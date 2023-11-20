
import asyncio
import logging
import time
from typing import Union
import numpy as np
import pandas as pd

from nodes.planning.consensus.bids import BidBuffer, UnconstrainedBid
from nodes.science.reqs import GroundPointMeasurementRequest, MeasurementRequest
from nodes.states import SatelliteAgentState, SimulationAgentState, UAVAgentState
from nodes.actions import *
from nodes.planning.consensus.consesus import ConsensusPlanner


class ACBBA(ConsensusPlanner):
    async def bundle_builder(self) -> None:
        try:
            plan = []
            results, bundle, path = {}, [], []
            prev_bundle_results = {}; prev_bundle = []
            
            level = logging.WARNING
            level = logging.DEBUG

            while True:
                # wait for incoming bids
                await self.replan.wait()

                incoming_bids = await self.listener_to_builder_buffer.pop_all()
                self.log_changes('builder - BIDS RECEIVED', incoming_bids, level)

                # Consensus Phase 
                t_0 = time.perf_counter()
                results, bundle, path, consensus_changes, \
                consensus_rebroadcasts = self.consensus_phase(  results, 
                                                                bundle, 
                                                                path, 
                                                                self.get_current_time(),
                                                                incoming_bids,
                                                                'builder',
                                                                level
                                                            )
                dt = time.perf_counter() - t_0
                self.stats['consensus'].append(dt)

                buffer = BidBuffer()
                await buffer.put_bids(consensus_rebroadcasts)
                consensus_rebroadcasts = await buffer.pop_all() 

                self.log_changes("builder - CHANGES MADE FROM CONSENSUS", consensus_changes, level)
                self.log_changes("builder - POTENTIAL REBROADCASTS TO BE DONE", consensus_rebroadcasts, level)

                # Planning Phase
                if(
                    len(consensus_rebroadcasts) > 0 
                    or self.t_plan + self.planning_horizon <= self.get_current_time() 
                    ):
                    t_0 = time.perf_counter()
                    results, bundle, path,\
                        planner_changes = self.planning_phase( self.agent_state, 
                                                                results, 
                                                                bundle, 
                                                                path, 
                                                                level
                                                            )
                    dt = time.perf_counter() - t_0
                    self.stats['planning'].append(dt)

                    broadcast_buffer = BidBuffer()
                    await broadcast_buffer.put_bids(planner_changes)
                    planner_changes = await broadcast_buffer.pop_all()
                    self.log_changes("builder - CHANGES MADE FROM PLANNING", planner_changes, level)
                    
                    # Check for convergence
                    same_bundle = self.compare_bundles(bundle, prev_bundle)
                    
                    same_bids = True
                    for key, bids in prev_bundle_results.items():
                        key : str; bids : list
                        for i in range(len(bids)):
                            prev_bid = prev_bundle_results[key][i]
                            current_bid = results[key][i]

                            if prev_bid is None:
                                continue

                            if prev_bid != current_bid:
                                same_bids = False
                                break

                            if not same_bids:
                                break

                    if not same_bundle:
                        await self.agent_state_lock.acquire()
                        plan = self.plan_from_path(self.agent_state, results, path)
                        self.agent_state_lock.release()

                    if isinstance(self.agent_state, UAVAgentState):
                        await self.agent_state_lock.acquire()
                        plan.insert(0, TravelAction(self.agent_state.pos, self.get_current_time()))
                        self.agent_state_lock.release()

                    # if not same_bundle or not same_bids:
                    #     # if not converged yet, await for 
                    #     plan.insert(0, WaitForMessages(self.get_current_time(), np.Inf))

                # Update iteration counter
                self.iter_counter += 1

                # save previous bundle for future convergence checks
                prev_bundle_results = {}
                prev_bundle = []
                for req, subtask_index in bundle:
                    req : MeasurementRequest; subtask_index : int
                    prev_bundle.append((req, subtask_index))
                    
                    if req.id not in prev_bundle_results:
                        prev_bundle_results[req.id] = [None for _ in results[req.id]]
                    prev_bundle_results[req.id][subtask_index] = results[req.id][subtask_index].copy()

                # Broadcast changes to bundle and any changes from consensus
                broadcast_bids : list = consensus_rebroadcasts
                broadcast_bids.extend(planner_changes)
                                
                broadcast_buffer = BidBuffer()
                await broadcast_buffer.put_bids(broadcast_bids)
                broadcast_bids = await broadcast_buffer.pop_all()
                self.log_changes("builder - REBROADCASTS TO BE DONE", broadcast_bids, level)

                await self.builder_to_broadcaster_buffer.put_bids(broadcast_bids)                                
                
                # Send plan to broadcaster
                await self.plan_inbox.put(plan)

        except asyncio.CancelledError:
            return

        finally:
            self.bundle_builder_results = results 

    def generate_bids_from_request(self, req : MeasurementRequest) -> list:
        return UnconstrainedBid.new_bids_from_request(req, self.get_parent_name())

    """
    ----------------------
        PLANNING PHASE 
    ----------------------
    """
    def planning_phase(self, state : SimulationAgentState, results : dict, bundle : list, path : list, level : int = logging.DEBUG) -> None:
        """
        Uses the most updated measurement request information to construct a path
        """
        self.log_results('builder - INITIAL BUNDLE RESULTS', results, level)
        self.log_task_sequence('bundle', bundle, level)
        self.log_task_sequence('path', path, level)

        changes = []
        changes_to_bundle = []

        if len(bundle) >= self.max_bundle_size:
            self.log_results('builder - MODIFIED BUNDLE RESULTS', results, level)
            self.log_task_sequence('bundle', bundle, level)
            self.log_task_sequence('path', path, level)

            return results, bundle, path, changes

        available_tasks : list = self.get_available_requests(state, bundle, results)
        
        current_bids = {req.id : {} for req, _ in bundle}
        for req, subtask_index in bundle:
            req : MeasurementRequest
            current_bid : UnconstrainedBid = results[req.id][subtask_index]
            current_bids[req.id][subtask_index] = current_bid.copy()

        max_path = [(req, subtask_index) for req, subtask_index in path]; 
        max_path_bids = {req.id : {} for req, _ in path}
        for req, subtask_index in path:
            req : MeasurementRequest
            max_path_bids[req.id][subtask_index] = results[req.id][subtask_index]

        max_task = -1

        while len(available_tasks) > 0 and max_task is not None and len(bundle) < self.max_bundle_size:                   
            # find next best task to put in bundle (greedy)
            max_task = None 
            max_subtask = None
            for measurement_req, subtask_index in available_tasks:
                # calculate bid for a given available task
                measurement_req : MeasurementRequest
                subtask_index : int

                if (    
                        isinstance(measurement_req, GroundPointMeasurementRequest) 
                    and isinstance(state, SatelliteAgentState)
                    ):
                    # check if the satellite can observe the GP
                    lat,lon,_ = measurement_req.lat_lon_pos
                    df : pd.DataFrame = self.orbitdata.get_ground_point_accesses_future(lat, lon, 0.0)
                    if df.empty:
                        continue

                projected_path, projected_bids, projected_path_utility = self.calc_path_bid(state, results, path, measurement_req, subtask_index)

                # check if path was found
                if projected_path is None:
                    continue
                
                # compare to maximum task
                projected_utility = projected_bids[measurement_req.id][subtask_index].winning_bid
                current_utility = results[measurement_req.id][subtask_index].winning_bid
                if ((max_task is None or projected_path_utility > max_path_utility)
                    and projected_utility > current_utility
                    ):

                    # check for cualition and mutex satisfaction
                    proposed_bid : UnconstrainedBid = projected_bids[measurement_req.id][subtask_index]
                    
                    max_path = projected_path
                    max_task = measurement_req
                    max_subtask = subtask_index
                    max_path_bids = projected_bids
                    max_path_utility = projected_path_utility
                    max_utility = proposed_bid.winning_bid

            if max_task is not None:
                # max bid found! place task with the best bid in the bundle and the path
                bundle.append((max_task, max_subtask))
                path = max_path

                # # remove bid task from list of available tasks
                # available_tasks.remove((max_task, max_subtask))
            
            # update results
            for measurement_req, subtask_index in path:
                measurement_req : MeasurementRequest
                subtask_index : int
                new_bid : UnconstrainedBid = max_path_bids[measurement_req.id][subtask_index]
                old_bid : UnconstrainedBid = results[measurement_req.id][subtask_index]

                if old_bid != new_bid:
                    changes_to_bundle.append((measurement_req, subtask_index))

                results[measurement_req.id][subtask_index] = new_bid

            # self.log_results('PRELIMINART MODIFIED BUNDLE RESULTS', results, level)
            # self.log_task_sequence('bundle', bundle, level)
            # self.log_task_sequence('path', path, level)

            available_tasks : list = self.get_available_requests(state, bundle, results)

        # broadcast changes to bundle
        for measurement_req, subtask_index in changes_to_bundle:
            measurement_req : MeasurementRequest
            subtask_index : int

            new_bid = results[measurement_req.id][subtask_index]

            # add to changes broadcast 
            changes.append(new_bid)

        self.log_results('builder - MODIFIED BUNDLE RESULTS', results, level)
        self.log_task_sequence('bundle', bundle, level)
        self.log_task_sequence('path', path, level)

        return results, bundle, path, changes

    

    """
    --------------------
    LOGGING AND TEARDOWN
    --------------------
    """
    def log_results(self, dsc : str, results : dict, level=logging.DEBUG) -> None:
        """
        Logs current results at a given time for debugging purposes

        ### Argumnents:
            - dsc (`str`): description of what is to be logged
            - results (`dict`): results to be logged
            - level (`int`): logging level to be used
        """
        if self._logger.getEffectiveLevel() <= level:
            headers = ['req_id', 'i', 'mmt', 'location', 'bidder', 'bid', 'winner', 'bid', 't_img', 'performed']
            data = []
            for req_id in results:
                if isinstance(results[req_id], list):
                    for bid in results[req_id]:
                        bid : UnconstrainedBid
                        req = MeasurementRequest.from_dict(bid.req)
                        split_id = req.id.split('-')
                        line = [split_id[0], bid.subtask_index, bid.main_measurement, req.pos, bid.bidder, round(bid.own_bid, 3), bid.winner, round(bid.winning_bid, 3), round(bid.t_img, 3), bid.performed]
                        data.append(line)
                elif isinstance(results[req_id], dict):
                    for bid_index in results[req_id]:
                        bid : UnconstrainedBid = results[req_id][bid_index]
                        req = MeasurementRequest.from_dict(bid.req)
                        split_id = req.id.split('-')
                        line = [split_id[0], bid.subtask_index, bid.main_measurement, req.pos, bid.bidder, round(bid.own_bid, 3), bid.winner, round(bid.winning_bid, 3), round(bid.t_img, 3), bid.performed]
                        data.append(line)
                else:
                    raise ValueError(f'`results` must be of type `list` or `dict`. is of type {type(results)}')

            df = pd.DataFrame(data, columns=headers).query('`bid` > 0.0')
            self.log(f'\n{dsc} [Iter {self.iter_counter}]\n{str(df)}\n', level)

    def log_changes(self, dsc: str, changes: list, level=logging.DEBUG) -> None:
        if self._logger.getEffectiveLevel() <= level:
            headers = ['req_id', 'i', 'mmt', 'location', 'bidder', 'bid', 'winner', 'bid', 't_update', 't_img', 'performed']
            data = []
            for bid in changes:
                bid : UnconstrainedBid
                req = MeasurementRequest.from_dict(bid.req)
                split_id = req.id.split('-')
                line = [split_id[0], bid.subtask_index, bid.main_measurement, req.pos, bid.bidder, round(bid.own_bid, 3), bid.winner, round(bid.winning_bid, 3), round(bid.t_update, 3), round(bid.t_img, 3), bid.performed]
                data.append(line)
        
            df = pd.DataFrame(data, columns=headers)
            self.log(f'\n{dsc} [Iter {self.iter_counter}]\n{str(df)}\n', level)