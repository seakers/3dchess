from abc import abstractmethod
import asyncio
import logging
import math
from scipy import optimize
import time
from typing import Any, Callable, Union
import pandas as pd
import zmq
from nodes.orbitdata import OrbitData
from nodes.planning.consensus.bids import Bid

from nodes.science.reqs import MeasurementRequest
from messages import *
from dmas.network import NetworkConfig
from nodes.states import *
from nodes.planning.consensus.bids import BidBuffer
from nodes.planning.planners import PlanningModule
from nodes.science.utility import synergy_factor

class ConsensusPlanner(PlanningModule):
    def __init__(   self, 
                    results_path: str, 
                    parent_name: str, 
                    parent_network_config: NetworkConfig, 
                    utility_func: Callable[[], Any], 
                    payload : list,
                    max_bundle_size = 3,
                    planning_horizon = 3600,
                    initial_reqs : list = [],
                    level: int = logging.INFO, 
                    logger: logging.Logger = None
                ) -> None:

        super().__init__(   results_path, 
                            parent_name, 
                            parent_network_config, 
                            utility_func, 
                            level, 
                            logger
                        )
        self.stats = {
                        "consensus" : [],
                        "planning" : [],
                        "doing" : [],
                        "c_comp_check" : [],
                        "c_t_end_check" : [],
                        "c_const_check" : []
                    }
        self.plan_history = []
        self.iter_counter = 0
        self.payload = payload
        self.max_bundle_size = max_bundle_size
        self.planning_horizon = planning_horizon
        self.parent_agent_type = None

        self.initial_reqs = []
        for req in initial_reqs:
            req : MeasurementRequest
            if req.t_start > 0:
                continue
            self.initial_reqs.append(req)
    
    async def setup(self) -> None:
        await super().setup()

        self.listener_to_builder_buffer = BidBuffer()
        self.listener_to_broadcaster_buffer = BidBuffer()
        self.builder_to_broadcaster_buffer = BidBuffer()
        self.broadcasted_bids_buffer = BidBuffer()

        self.t_curr = 0.0
        self.agent_state : SimulationAgentState = None
        self.agent_state_lock = asyncio.Lock()
        self.agent_state_updated = asyncio.Event()
        self.parent_agent_type = None
        self.orbitdata = None

        self.t_plan = 0.0
        self.plan_inbox = asyncio.Queue()
        self.replan = asyncio.Event()

    async def live(self) -> None:
        """
        Performs three concurrent tasks:
        - Listener: receives messages from the parent agent and updates internal results
        - Bundle-builder: plans and bids according to local information
        - Planner: listens for requests from parent agent and returns latest plan to perform
        """
        try:
            listener_task = asyncio.create_task(self.listener(), name='listener()')
            # bundle_builder_task = asyncio.create_task(self.bundle_builder(), name='bundle_builder()')
            planner_task = asyncio.create_task(self.planner(), name='planner()')
            
            tasks = [listener_task, 
                    # bundle_builder_task, 
                    planner_task]

            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        finally:
            for task in done:
                self.log(f'`{task.get_name()}` task finalized! Terminating all other tasks...')

            for task in pending:
                task : asyncio.Task
                if not task.done():
                    task.cancel()
                    await task

    async def listener(self) -> None:
        """
        Listens for any incoming messages, unpacks them and classifies them into 
        internal inboxes for future processing
        """
        try:
            # initiate results tracker
            results = {}
            # level = logging.WARNING
            level = logging.DEBUG

            # listen for broadcasts and place in the appropriate inboxes
            while True:
                self.log('listening to manager broadcast!')
                _, _, content = await self.listen_manager_broadcast()

                # if sim-end message, end agent `live()`
                if content['msg_type'] == ManagerMessageTypes.SIM_END.value:
                    self.log(f"received manager broadcast or type {content['msg_type']}! terminating `live()`...")
                    return

                elif content['msg_type'] == SimulationMessageTypes.SENSES.value:
                    self.log(f"received senses from parent agent!", level=logging.DEBUG)

                    # unpack message 
                    senses_msg : SensesMessage = SensesMessage(**content)

                    senses = []
                    senses.append(senses_msg.state)
                    senses.extend(senses_msg.senses)  

                    incoming_bids = []    

                    state : SimulationAgentState = None

                    for sense in senses:
                        if sense['msg_type'] == SimulationMessageTypes.AGENT_ACTION.value:
                            # unpack message 
                            action_msg = AgentActionMessage(**sense)
                            self.log(f"received agent action of status {action_msg.status}!")
                            
                            # send to planner
                            await self.action_status_inbox.put(action_msg)

                        elif sense['msg_type'] == SimulationMessageTypes.AGENT_STATE.value:
                            # unpack message 
                            state_msg : AgentStateMessage = AgentStateMessage(**sense)
                            self.log(f"received agent state message!")
                                                        
                            # update current state
                            await self.agent_state_lock.acquire()
                            state : SimulationAgentState = SimulationAgentState.from_dict(state_msg.state)

                            await self.update_current_time(state.t)
                            self.agent_state = state

                            if self.parent_agent_type is None:
                                if isinstance(state, SatelliteAgentState):
                                    # import orbit data
                                    self.orbitdata : OrbitData = self._load_orbit_data()
                                    self.parent_agent_type = SimulationAgentTypes.SATELLITE.value
                                elif isinstance(state, UAVAgentState):
                                    self.parent_agent_type = SimulationAgentTypes.UAV.value
                                else:
                                    raise NotImplementedError(f"states of type {state_msg.state['state_type']} not supported for greedy planners.")
                            
                            self.agent_state_lock.release()

                        elif sense['msg_type'] == SimulationMessageTypes.MEASUREMENT_REQ.value:
                            # unpack message 
                            req_msg = MeasurementRequestMessage(**sense)
                            req = MeasurementRequest.from_dict(req_msg.req)
                            self.log(f"received measurement request message!")
                            
                            # if not in send to planner
                            if req.id not in results:
                                # create task bid from measurement request and add to results
                                self.log(f"received new measurement request! Adding to results ledger...")

                                bids : list = self.generate_bids_from_request(req)
                                incoming_bids.extend(bids)

                        elif sense['msg_type'] == SimulationMessageTypes.MEASUREMENT_BID.value:
                            # unpack message 
                            bid_msg = MeasurementBidMessage(**sense)
                            bid : Bid = Bid.from_dict(bid_msg.bid)
                            self.log(f"received measurement request message!")
                            
                            incoming_bids.append(bid)              
                    
                    if len(incoming_bids) > 0:
                        sorting_buffer = BidBuffer()
                        await sorting_buffer.put_bids(incoming_bids)
                        incoming_bids = await sorting_buffer.pop_all()

                        await self.listener_to_builder_buffer.put_bids(incoming_bids)
                        await self.listener_to_broadcaster_buffer.put_bids(incoming_bids)

                    # inform planner of state update
                    self.agent_state_updated.set()
                    self.agent_state_updated.clear()
                    
                    await self.states_inbox.put(state) 

        except asyncio.CancelledError:
            return
        
        finally:
            self.listener_results = results    
    
    @abstractmethod
    def generate_bids_from_request(self, req : MeasurementRequest) -> list:
        pass

    @abstractmethod
    async def bundle_builder(self) -> None:
        """
        Waits for incoming bids to re-evaluate its current plan
        """
        pass

    async def planner(self) -> None:
        """
        Generates a plan for the parent agent to perform
        """
        try:
            # initialize results vectors
            plan = []
            bundle, path, results = [], [], {}

            # level = logging.WARNING
            level = logging.DEBUG

            while True:
                # wait for agent to update state
                _ : AgentStateMessage = await self.states_inbox.get()

                # --- Check Action Completion ---
                
                while not self.action_status_inbox.empty():
                    action_msg : AgentActionMessage = await self.action_status_inbox.get()
                    action : AgentAction = action_from_dict(**action_msg.action)

                    if action_msg.status == AgentAction.PENDING:
                        # if action wasn't completed, try again
                        plan_ids = [action.id for action in self.plan]
                        action_dict : dict = action_msg.action
                        if action_dict['id'] in plan_ids:
                            self.log(f'action {action_dict} not completed yet! trying again...')
                            plan_out.append(action_dict)

                    else:
                        # if action was completed or aborted, remove from plan
                        if action_msg.status == AgentAction.COMPLETED:
                            self.log(f'action of type `{action.action_type}` completed!', level)

                        action_dict : dict = action_msg.action
                        completed_action = AgentAction(**action_dict)
                        removed = None
                        for action in plan:
                            action : AgentAction
                            if action.id == completed_action.id:
                                removed = action
                                break
                        
                        if removed is not None:
                            removed : AgentAction
                            plan : list
                            plan.remove(removed)

                            if (isinstance(removed, BroadcastMessageAction) 
                                and action_msg.status == AgentAction.COMPLETED):
                                
                                bid_msg = MeasurementBidMessage(**removed.msg)
                                bid : Bid = Bid.from_dict(bid_msg.bid)
                                bid.set_performed(self.get_current_time())

                                if bid.performed:
                                    await self.listener_to_builder_buffer.put_bid(bid)
                
                # --- Look for Plan Updates ---

                replan = False
                plan_out = []
                
                if (len(self.initial_reqs) > 0                  # there are initial requests 
                    and self.get_current_time() <= 1e-3         # simulation just started
                    ):
                    # Generate Initial Plan
                    initial_reqs = self.initial_reqs
                    self.initial_reqs = []

                    await self.agent_state_lock.acquire()
                    path : list = self._initialize_observation_path(self.agent_state, initial_reqs)
                    plan : list = self.plan_from_path(self.agent_state, path)
                    self.agent_state_lock.release()

                if (len(self.listener_to_builder_buffer) > 0 ): # bids were received   
                    # Consensus Phase 
                    incoming_bids = await self.listener_to_builder_buffer.pop_all()
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

                    replan = len(consensus_rebroadcasts) > 0 

                # if (self.t_plan + self.planning_horizon <= self.get_current_time()): 
                #     # TODO implement case for planning horizon
                #     replan = True

                if replan:
                    await self.agent_state_lock.acquire()
                    results, bundle, path = self.replanner()         
                    plan : list = self.plan_from_path(self.agent_state, path)
                    self.agent_state_lock.release()
                           

                #     # wait for plan to be updated
                #     # self.replan.set(); self.replan.clear()
                #     # plan : list = await self.plan_inbox.get()
                #     # plan = []
                #     # plan_copy = [action for action in plan]
                #     # self.plan_history.append((self.get_current_time(), plan_copy))
                #     self.t_plan = self.get_current_time()

                #     # compule updated bids from the listener and bundle buiilder
                #     if len(self.builder_to_broadcaster_buffer) > 0:
                #         # received bids to rebroadcast from bundle-builder
                #         builder_bids : list = await self.builder_to_broadcaster_buffer.pop_all()
                                                
                #         # flush bids from listener    
                #         _ = await self.listener_to_broadcaster_buffer.pop_all()

                #         # compile bids to be rebroadcasted
                #         rebroadcast_bids : list = builder_bids.copy()
                #         self.log_changes("planner - REBROADCASTS TO BE DONE", rebroadcast_bids, level)
                        
                #         # create message broadcasts for every bid
                #         for rebroadcast_bid in rebroadcast_bids:
                #             rebroadcast_bid : Bid
                #             bid_message = MeasurementBidMessage(self.get_parent_name(), self.get_parent_name(), rebroadcast_bid.to_dict())
                #             plan_out.append( BroadcastMessageAction(bid_message.to_dict(), self.get_current_time()).to_dict() )
                #     else:
                #         # flush redundant broadcasts from listener
                #         _ = await self.listener_to_broadcaster_buffer.pop_all()

                # --- Execute plan ---

                # get next action to perform
                plan_out_ids = [action['id'] for action in plan_out]
                if len(plan_out_ids) > 0:
                    for action in plan:
                        action : AgentAction
                        if (action.t_start <= self.get_current_time()
                            and action.id not in plan_out_ids):
                            plan_out.append(action.to_dict())
                            break
                else:
                    for action in plan:
                        action : AgentAction
                        if (action.t_start <= self.get_current_time()
                            and action.id not in plan_out_ids):
                            plan_out.append(action.to_dict())

                if len(plan_out) == 0:
                    if len(plan) > 0:
                        # next action is yet to start, wait until then
                        next_action : AgentAction = plan[0]
                        t_idle = next_action.t_start if next_action.t_start > self.get_current_time() else self.get_current_time()
                    else:
                        # no more actions to perform
                        if isinstance(self.agent_state, SatelliteAgentState):
                            # wait until the next ground-point access
                            self.orbitdata : OrbitData
                            t_next = self.orbitdata.get_next_gs_access(self.get_current_time())
                        else:
                            # idle until the end of the simulation
                            t_next = np.Inf

                        # t_idle = self.t_plan + self.planning_horizon
                        t_idle = t_next

                    action = WaitForMessages(self.get_current_time(), t_idle)
                    plan_out.append(action.to_dict())

                # --- FOR DEBUGGING PURPOSES ONLY: ---
                out = f'\nPLAN\nid\taction type\tt_start\tt_end\n'
                for action in plan:
                    action : AgentAction
                    out += f"{action.id.split('-')[0]}, {action.action_type}, {action.t_start}, {action.t_end}\n"
                # self.log(out, level=logging.WARNING)

                out = f'\nPLAN OUT\nid\taction type\tt_start\tt_end\n'
                for action in plan_out:
                    action : dict
                    out += f"{action['id'].split('-')[0]}, {action['action_type']}, {action['t_start']}, {action['t_end']}\n"
                # self.log(out, level=logging.WARNING)
                # -------------------------------------

                self.log(f'sending {len(plan_out)} actions to agent...')
                plan_msg = PlanMessage(self.get_element_name(), self.get_network_name(), plan_out)
                await self._send_manager_msg(plan_msg, zmq.PUB)

                self.log(f'actions sent!')

        except asyncio.CancelledError:
            return

    async def replanner(  self, 
                    state : AbstractAgentState, 
                    original_plan : list, 
                    incoming_bids : list, 
                    level:int=logging.DEBUG
                    ) -> list:
        """
        Readjusts plan as needed
        """

        
        # Planning Phase
        # Only activated if relevant changes were made to the results or if the planning horizon 
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

    """
    -----------------------
        CONSENSUS PHASE
    -----------------------
    """
    def consensus_phase(  
                                self, 
                                results : dict, 
                                bundle : list, 
                                path : list, 
                                t : Union[int, float], 
                                new_bids : list,
                                process_name : str,
                                level : int = logging.DEBUG
                            ) -> None:
        """
        Evaluates incoming bids and updates current results and bundle
        """
        changes = []
        rebroadcasts = []
        self.log_results(f'\n{process_name} - INITIAL RESULTS', results, level)
        self.log_task_sequence('bundle', bundle, level)
        self.log_task_sequence('path', path, level)
        
        # compare bids with incoming messages
        t_0 = time.perf_counter()
        results, bundle, path, \
            comp_changes, comp_rebroadcasts = self.compare_results(results, bundle, path, t, new_bids, level)
        changes.extend(comp_changes)
        rebroadcasts.extend(comp_rebroadcasts)
        dt = time.perf_counter() - t_0
        self.stats['c_comp_check'].append(dt)

        self.log_changes(f'{process_name} - BIDS RECEIVED', new_bids, level)
        self.log_results(f'{process_name} - COMPARED RESULTS', results, level)
        self.log_task_sequence('bundle', bundle, level)
        self.log_task_sequence('path', path, level)
        
        # check for expired tasks
        t_0 = time.perf_counter()
        results, bundle, path, \
            exp_changes, exp_rebroadcasts = self.check_request_end_time(results, bundle, path, t, level)
        changes.extend(exp_changes)
        rebroadcasts.extend(exp_rebroadcasts)
        dt = time.perf_counter() - t_0
        self.stats['c_t_end_check'].append(dt)

        self.log_results(f'{process_name} - CHECKED EXPIRATION RESULTS', results, level)
        self.log_task_sequence('bundle', bundle, level)
        self.log_task_sequence('path', path, level)

        # check for already performed tasks
        t_0 = time.perf_counter()
        results, bundle, path, \
            done_changes, done_rebroadcasts = self.check_request_completion(results, bundle, path, t, level)
        changes.extend(done_changes)
        rebroadcasts.extend(done_rebroadcasts)
        dt = time.perf_counter() - t_0
        self.stats['c_t_end_check'].append(dt)

        self.log_results(f'{process_name} - CHECKED EXPIRATION RESULTS', results, level)
        self.log_task_sequence('bundle', bundle, level)
        self.log_task_sequence('path', path, level)

        return results, bundle, path, changes, rebroadcasts

    def compare_results(
                        self, 
                        results : dict, 
                        bundle : list, 
                        path : list, 
                        t : Union[int, float], 
                        new_bids : list,
                        level=logging.DEBUG
                    ) -> tuple:
        """
        Compares the existing results with any incoming task bids and updates the bundle accordingly

        ### Returns
            - results
            - bundle
            - path
            - changes
        """
        changes = []
        rebroadcasts = []

        for their_bid in new_bids:
            their_bid : Bid            

            # check bids are for new requests
            new_req = their_bid.req_id not in results

            req = MeasurementRequest.from_dict(their_bid.req)
            if new_req:
                # was not aware of this request; add to results as a blank bid
                results[req.id] = self.generate_bids_from_request(req)

                # add to changes broadcast
                my_bid : Bid = results[req.id][0]
                rebroadcasts.append(my_bid)
                                    
            # compare bids
            my_bid : Bid = results[their_bid.req_id][their_bid.subtask_index]
            self.log(f'comparing bids...\nmine:  {str(my_bid)}\ntheirs: {str(their_bid)}', level=logging.DEBUG)

            broadcast_bid, changed  = my_bid.update(their_bid.to_dict(), t)
            broadcast_bid : Bid; changed : bool

            self.log(f'\nupdated: {my_bid}\n', level=logging.DEBUG)
            results[their_bid.req_id][their_bid.subtask_index] = my_bid
                
            # if relevant changes were made, add to changes and rebroadcast
            if changed or new_req:
                changed_bid : Bid = broadcast_bid if not new_req else my_bid
                changes.append(changed_bid)

            if broadcast_bid or new_req:                    
                broadcast_bid : Bid = broadcast_bid if not new_req else my_bid
                rebroadcasts.append(broadcast_bid)

            # if outbid for a task in the bundle, release subsequent tasks in bundle and path
            if (
                (req, my_bid.subtask_index) in bundle 
                and my_bid.winner != self.get_parent_name()
                ):
                bid_index = bundle.index((req, my_bid.subtask_index))

                for _ in range(bid_index, len(bundle)):
                    # remove all subsequent tasks from bundle
                    measurement_req, subtask_index = bundle.pop(bid_index)
                    measurement_req : MeasurementRequest
                    path.remove((measurement_req, subtask_index))

                    # if the agent is currently winning this bid, reset results
                    current_bid : Bid = results[measurement_req.id][subtask_index]
                    if current_bid.winner == self.get_parent_name():
                        current_bid.reset(t)
                        results[measurement_req.id][subtask_index] = current_bid

                        rebroadcasts.append(current_bid)
                        changes.append(current_bid)
        
        return results, bundle, path, changes, rebroadcasts

    def check_request_end_time(self, results : dict, bundle : list, path : list, t : Union[int, float], level=logging.DEBUG) -> tuple:
        """
        Checks if measurement requests have expired and can no longer be performed

        ### Returns
            - results
            - bundle
            - path
            - changes
        """
        changes = []
        rebroadcasts = []
        # release tasks from bundle if t_end has passed
        task_to_remove = None
        for req, subtask_index in bundle:
            req : MeasurementRequest
            if req.t_end - req.duration < t:
                task_to_remove = (req, subtask_index)
                break

        if task_to_remove is not None:
            bundle_index = bundle.index(task_to_remove)
            for _ in range(bundle_index, len(bundle)):
                # remove all subsequent bids from bundle
                measurement_req, subtask_index = bundle.pop(bundle_index)

                # remove bids from path
                path.remove((measurement_req, subtask_index))

                # if the agent is currently winning this bid, reset results
                measurement_req : Bid
                current_bid : Bid = results[measurement_req.id][subtask_index]
                if current_bid.winner == self.get_parent_name():
                    current_bid.reset(t)
                    results[measurement_req.id][subtask_index] = current_bid
                    
                    rebroadcasts.append(current_bid)
                    changes.append(current_bid)

        return results, bundle, path, changes, rebroadcasts

    def check_request_completion(self, results : dict, bundle : list, path : list, t : Union[int, float], level=logging.DEBUG) -> tuple:
        """
        Checks if a subtask or a mutually exclusive subtask has already been performed 

        ### Returns
            - results
            - bundle
            - path
            - changes
        """

        changes = []
        rebroadcasts = []
        task_to_remove = None
        task_to_reset = None
        for req, subtask_index in bundle:
            req : MeasurementRequest

            # check if bid has been performed 
            subtask_bid : Bid = results[req.id][subtask_index]
            if self.is_bid_completed(req, subtask_bid, t):
                task_to_remove = (req, subtask_index)
                break

            # check if a mutually exclusive bid has been performed
            for subtask_bid in results[req.id]:
                subtask_bid : Bid

                bids : list = results[req.id]
                bid_index = bids.index(subtask_bid)
                bid : Bid = bids[bid_index]

                if self.is_bid_completed(req, bid, t) and req.dependency_matrix[subtask_index][bid_index] < 0:
                    task_to_remove = (req, subtask_index)
                    task_to_reset = (req, subtask_index) 
                    break   

            if task_to_remove is not None:
                break

        if task_to_remove is not None:
            if task_to_reset is not None:
                bundle_index = bundle.index(task_to_remove)
                
                # level=logging.WARNING
                self.log_results('PRELIMINARY PREVIOUS PERFORMER CHECKED RESULTS', results, level)
                self.log_task_sequence('bundle', bundle, level)
                self.log_task_sequence('path', path, level)

                for _ in range(bundle_index, len(bundle)):
                    # remove task from bundle and path
                    req, subtask_index = bundle.pop(bundle_index)
                    path.remove((req, subtask_index))

                    bid : Bid = results[req.id][subtask_index]
                    bid.reset(t)
                    results[req.id][subtask_index] = bid

                    rebroadcasts.append(bid)
                    changes.append(bid)

                    self.log_results('PRELIMINARY PREVIOUS PERFORMER CHECKED RESULTS', results, level)
                    self.log_task_sequence('bundle', bundle, level)
                    self.log_task_sequence('path', path, level)
            else: 
                # remove performed subtask from bundle and path 
                bundle_index = bundle.index(task_to_remove)
                req, subtask_index = bundle.pop(bundle_index)
                path.remove((req, subtask_index))

                # set bid as completed
                bid : Bid = results[req.id][subtask_index]
                bid.performed = True
                results[req.id][subtask_index] = bid

        return results, bundle, path, changes, rebroadcasts

    def is_bid_completed(self, req : MeasurementRequest, bid : Bid, t : float) -> bool:
        """
        Checks if a bid has been completed or not
        """
        return (bid.t_img >= 0.0 and bid.t_img + req.duration < t) or bid.performed

    """
    -----------------------
        PLANNING PHASE
    -----------------------
    """
    def _initialize_observation_path(self, state : AbstractAgentState, initial_reqs : list, level : int = logging.DEBUG) -> list:
        """ 
        Creates a preliminary observations plan to be performed by the agent. 

        ### Arguments:
            - state (:obj:`AbstractAgentState`) : Initial state of the agent at the start of the simulation
            - initial_reqs (`list`) : List of measurement requests available at the start of the simulation        
        """
        results = {req.id : [] for req in initial_reqs}
        path = [] 

        for req in initial_reqs:
            req : MeasurementRequest
            bids = self.generate_bids_from_request(req)
            results[req.id] = bids
        
        available_reqs : list = self.get_available_requests( state, [], results )

        if isinstance(state, SatelliteAgentState):
            # Generates a plan for observing GPs on a first-come first-served basis
            
            reqs = {req.id : req for req,_ in available_reqs}
            arrival_times = {req.id : {} for req,_ in available_reqs}

            for req, subtask_index in available_reqs:
                t_arrivals : list = self.calc_arrival_times(state, req, self.get_current_time())
                arrival_times[req.id][subtask_index] = t_arrivals
            
            path = []

            for req_id in arrival_times:
                for subtask_index in arrival_times[req_id]:
                    t_arrivals : list = arrival_times[req_id][subtask_index]
                    t_img = t_arrivals.pop(0)
                    req : MeasurementRequest = reqs[req_id]
                    path.append((req, subtask_index, t_img, req.s_max/len(req.measurements)))

            path.sort(key=lambda a: a[2])

            while True:
                
                conflict_free = True
                for i in range(len(path) - 1):
                    j = i + 1
                    req_i, _, t_i, __ = path[i]
                    req_j, subtask_index_j, t_j, s_j = path[j]

                    th_i = state.calc_off_nadir_agle(req_i)
                    th_j = state.calc_off_nadir_agle(req_j)

                    if abs(th_i - th_j) / state.max_slew_rate > t_j - t_i:
                        t_arrivals : list = arrival_times[req_j.id][subtask_index_j]
                        if len(t_arrivals) > 0:
                            t_img = t_arrivals.pop(0)

                            path[j] = (req_j, subtask_index_j, t_img, s_j)
                            path.sort(key=lambda a: a[2])
                        else:
                            #TODO remove request from path
                            raise Exception("Whoops. See Plan Initializer.")
                            path.pop(j) 
                        conflict_free = False
                        break

                if conflict_free:
                    break
                    
            out = '\n'
            for req, subtask_index, t_img, s in path:
                out += f"{req.id.split('-')[0]}\t{subtask_index}\t{np.round(t_img,3)}\t{np.round(s,3)}\n"
            self.log(out,level)

            return path
                
        else:
            raise NotImplementedError(f'initial planner for states of type `{type(state)}` not yet supported')
        
    def conflict_free(self, path) -> bool:
        """ Checks if path is conflict free """

    @abstractmethod
    async def planning_phase(self) -> None:
        pass


    def compare_bundles(self, bundle_1 : list, bundle_2 : list) -> bool:
        """
        Compares two bundles. Returns true if they are equal and false if not.
        """
        if len(bundle_1) == len(bundle_2):
            for req, subtask in bundle_1:
                if (req, subtask) not in bundle_2:            
                    return False
            return True
        return False
    
    def sum_path_utility(self, path : list, bids : dict) -> float:
        utility = 0.0
        for req, subtask_index in path:
            req : MeasurementRequest
            bid : Bid = bids[req.id][subtask_index]
            utility += bid.own_bid

        return utility

    def get_available_requests( self, 
                                state : SimulationAgentState, 
                                bundle : list, 
                                results : dict, 
                                planning_horizon : float = np.Inf
                                ) -> list:
        """
        Checks if there are any requests available to be performed

        ### Returns:
            - list containing all available and bidable tasks to be performed by the parent agent
        """
        available = []
        for req_id in results:
            for subtask_index in range(len(results[req_id])):
                subtaskbid : Bid = results[req_id][subtask_index]; 
                req = MeasurementRequest.from_dict(subtaskbid.req)

                is_biddable = self.can_bid(state, req, subtask_index, results[req_id], planning_horizon) 
                already_in_bundle = self.check_if_in_bundle(req, subtask_index, bundle)
                already_performed = self.request_has_been_performed(results, req, subtask_index, state.t)
                
                if is_biddable and not already_in_bundle and not already_performed:
                    available.append((req, subtaskbid.subtask_index))

        return available

    def can_bid(self, 
                state : SimulationAgentState, 
                req : MeasurementRequest, 
                subtask_index : int, 
                subtaskbids : list,
                planning_horizon : float
                ) -> bool:
        """
        Checks if an agent has the ability to bid on a measurement task
        """
        # check planning horizon
        if state.t + self.planning_horizon < req.t_start:
            return False

        # check capabilities - TODO: Replace with knowledge graph
        subtaskbid : Bid = subtaskbids[subtask_index]
        if subtaskbid.main_measurement not in [instrument.name for instrument in self.payload]:
            return False 

        # check time constraints
        ## Constraint 1: task must be able to be performed during or after the current time
        if req.t_end < state.t:
            return False

        elif isinstance(req, GroundPointMeasurementRequest):
            if isinstance(state, SatelliteAgentState):
                # check if agent can see the request location
                lat,lon,_ = req.lat_lon_pos
                df : pd.DataFrame = self.orbitdata.get_ground_point_accesses_future(lat, lon, state.t).sort_values(by='time index')
                
                can_access = False
                if not df.empty:                
                    times = df.get('time index')
                    for time in times:
                        time *= self.orbitdata.time_step 

                        if state.t + planning_horizon < time:
                            break

                        if req.t_start <= time <= req.t_end:
                            # there exists an access time before the request's availability ends
                            can_access = True
                            break
                
                if not can_access:
                    return False
        
        return True

    def check_if_in_bundle(self, req : MeasurementRequest, subtask_index : int, bundle : list) -> bool:
        for req_i, subtask_index_j in bundle:
            if req_i.id == req.id and subtask_index == subtask_index_j:
                return True
    
        return False

    def request_has_been_performed(self, results : dict, req : MeasurementRequest, subtask_index : int, t : Union[int, float]) -> bool:
        # check if subtask at hand has been performed
        current_bid : Bid = results[req.id][subtask_index]
        subtask_already_performed = t > current_bid.t_img >= 0 + req.duration and current_bid.winner != Bid.NONE
        if subtask_already_performed or current_bid.performed:
            return True
       
        return False

    def calc_path_bid(
                        self, 
                        state : SimulationAgentState, 
                        original_results : dict,
                        original_path : list, 
                        req : MeasurementRequest, 
                        subtask_index : int
                    ) -> tuple:
        state : SimulationAgentState = state.copy()
        winning_path = None
        winning_bids = None
        winning_path_utility = 0.0

        # check if the subtask is mutually exclusive with something in the bundle
        for req_i, subtask_j in original_path:
            req_i : MeasurementRequest; subtask_j : int
            if req_i.id == req.id:
                if req.dependency_matrix[subtask_j][subtask_index] < 0:
                    return winning_path, winning_bids, winning_path_utility

        # find best placement in path
        # self.log_task_sequence('original path', original_path, level=logging.WARNING)
        for i in range(len(original_path)+1):
            # generate possible path
            path = [scheduled_obs for scheduled_obs in original_path]
            
            path.insert(i, (req, subtask_index))
            # self.log_task_sequence('new proposed path', path, level=logging.WARNING)

            # calculate bids for each task in the path
            bids = {}
            for req_i, subtask_j in path:
                # calculate imaging time
                req_i : MeasurementRequest
                subtask_j : int
                t_img = self.calc_imaging_time(state, path, bids, req_i, subtask_j)

                # calc utility
                params = {"req" : req_i, "subtask_index" : subtask_j, "t_img" : t_img}
                utility = self.utility_func(**params) if t_img >= 0 else np.NINF
                utility *= synergy_factor(**params)

                # create bid
                bid : Bid = original_results[req_i.id][subtask_j].copy()
                bid.set_bid(utility, t_img, state.t)
                
                if req_i.id not in bids:
                    bids[req_i.id] = {}    
                bids[req_i.id][subtask_j] = bid                

            # look for path with the best utility
            path_utility = self.sum_path_utility(path, bids)
            if path_utility > winning_path_utility:
                winning_path = path
                winning_bids = bids
                winning_path_utility = path_utility

        return winning_path, winning_bids, winning_path_utility


    def calc_imaging_time(self, state : SimulationAgentState, path : list, bids : dict, req : MeasurementRequest, subtask_index : int) -> float:
        """
        Computes the ideal" time when a task in the path would be performed
        ### Returns
            - t_img (`float`): earliest available imaging time
        """
        # calculate the state of the agent prior to performing the measurement request
        i = path.index((req, subtask_index))
        if i == 0:
            t_prev = state.t
            prev_state = state.copy()
        else:
            prev_req, prev_subtask_index = path[i-1]
            prev_req : MeasurementRequest; prev_subtask_index : int
            bid_prev : Bid = bids[prev_req.id][prev_subtask_index]
            t_prev : float = bid_prev.t_img + prev_req.duration

            if isinstance(state, SatelliteAgentState):
                prev_state : SatelliteAgentState = state.propagate(t_prev)
                
                prev_state.attitude = [
                                        prev_state.calc_off_nadir_agle(prev_req),
                                        0.0,
                                        0.0
                                    ]
            elif isinstance(state, UAVAgentState):
                prev_state = state.copy()
                prev_state.t = t_prev
                
                if isinstance(prev_req, GroundPointMeasurementRequest):
                    prev_state.pos = prev_req.pos
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError(f"cannot calculate imaging time for agent states of type {type(state)}")

        return self.calc_arrival_times(prev_state, req, t_prev)[0]

    def calc_arrival_times(self, state : SimulationAgentState, req : MeasurementRequest, t_prev : Union[int, float]) -> float:
        """
        Estimates the quickest arrival time from a starting position to a given final position
        """
        if isinstance(req, GroundPointMeasurementRequest):
            # compute earliest time to the task
            if isinstance(state, SatelliteAgentState):
                t_imgs = []
                lat,lon,_ = req.lat_lon_pos
                df : pd.DataFrame = self.orbitdata.get_ground_point_accesses_future(lat, lon, t_prev)

                for _, row in df.iterrows():
                    t_img = row['time index'] * self.orbitdata.time_step
                    dt = t_img - state.t
                
                    # propagate state
                    propagated_state : SatelliteAgentState = state.propagate(t_img)

                    # compute off-nadir angle
                    thf = propagated_state.calc_off_nadir_agle(req)
                    dth = abs(thf - propagated_state.attitude[0])

                    # estimate arrival time using fixed angular rate TODO change to 
                    if dt >= dth / state.max_slew_rate: # TODO change maximum angular rate 
                        t_imgs.append(t_img)
                return t_imgs if len(t_imgs) > 0 else [-1]

            elif isinstance(state, UAVAgentState):
                dr = np.array(req.pos) - np.array(state.pos)
                norm = np.sqrt( dr.dot(dr) )
                return [norm / state.max_speed + t_prev]

            else:
                raise NotImplementedError(f"arrival time estimation for agents of type {self.parent_agent_type} is not yet supported.")

        else:
            raise NotImplementedError(f"cannot calculate imaging time for measurement requests of type {type(req)}")       


    """
    ------------------------
        EXECUTION PHASE
    ------------------------
    """
    # def plan_from_path( self, 
    #                     state : SimulationAgentState, 
    #                     results : dict, 
    #                     path : list
    #                 ) -> list:
    def plan_from_path( self, 
                        state : SimulationAgentState, 
                        path : list
                    ) -> list:
        """
        Generates a list of AgentActions from the current path.

        Agents look to move to their designated measurement target and perform the measurement.

        ## Arguments:
            - state (:obj:`SimulationAgentState`): state of the agent at the start of the path
            - path (`list`): list of tuples indicating the sequence of observations to be performed and time of observation
        """
        plan = []

        # if no requests left in the path, wait for the next planning horizon
        if len(path) == 0:
            t_0 = self.get_current_time()
            t_f = t_0 + self.planning_horizon
            return [WaitForMessages(t_0, t_f)]

        # TODO add wait timer if needed for plan convergende
        # t_conv_min = self.t_plan + self.planning_horizon
        # for measurement_req, subtask_index in path:
        #     bid : Bid = results[measurement_req.id][subtask_index]
        #     t_conv = bid.t_update + bid.dt_converge
        #     if t_conv < t_conv_min:
        #         t_conv_min = t_conv

        # if state.t <= t_conv_min:
        #     # if isinstance(self.agent_state, SatelliteAgentState):
        #     #     # wait until the next ground-point access
        #     #     self.orbitdata : OrbitData
        #     #     t_next = self.orbitdata.get_next_gs_access(self.get_current_time())
        #     # else:
        #     #     # idle until the end of the simulation
        #     #     t_next = np.Inf

        #     # t_conv_min = t_conv_min if t_conv_min < t_next else t_next            
        #     plan.append( WaitForMessages(state.t, t_conv_min) )

        # else:
        #     # plan.append( WaitForMessages(state.t, state.t) )
        #     t_conv_min = state.t

        # add actions per measurement
        for i in range(len(path)):
            plan_i = []

            measurement_req, subtask_index, t_img, u_exp = path[i]
            measurement_req : MeasurementRequest; subtask_index : int; t_img : float; u_exp : float

            if not isinstance(measurement_req, GroundPointMeasurementRequest):
                raise NotImplementedError(f"Cannot create plan for requests of type {type(measurement_req)}")
            
            # Estimate previous state
            if i == 0:
                if isinstance(state, SatelliteAgentState):
                    t_prev = state.t
                    prev_state : SatelliteAgentState = state.copy()

                elif isinstance(state, UAVAgentState):
                    t_prev = state.t #TODO consider wait time for convergence
                    prev_state : UAVAgentState = state.copy()

                else:
                    raise NotImplementedError(f"cannot calculate travel time start for agent states of type {type(state)}")
            else:
                prev_req = None
                for action in reversed(plan):
                    action : AgentAction
                    if isinstance(action, MeasurementAction):
                        prev_req = MeasurementRequest.from_dict(action.measurement_req)
                        break

                action_prev : AgentAction = plan[-1]
                t_prev = action_prev.t_end

                if isinstance(state, SatelliteAgentState):
                    prev_state : SatelliteAgentState = state.propagate(t_prev)
                    
                    if prev_req is not None:
                        prev_state.attitude = [
                                            prev_state.calc_off_nadir_agle(prev_req),
                                            0.0,
                                            0.0
                                        ]

                elif isinstance(state, UAVAgentState):
                    prev_state : UAVAgentState = state.copy()
                    prev_state.t = t_prev

                    if isinstance(prev_req, GroundPointMeasurementRequest):
                        prev_state.pos = prev_req.pos
                    else:
                        raise NotImplementedError(f"cannot calculate travel time start for requests of type {type(prev_req)} for uav agents")

                else:
                    raise NotImplementedError(f"cannot calculate travel time start for agent states of type {type(state)}")

            # maneuver to point to target
            t_maneuver_end = None
            if isinstance(state, SatelliteAgentState):
                prev_state : SatelliteAgentState

                t_maneuver_start = prev_state.t
                th_f = prev_state.calc_off_nadir_agle(measurement_req)
                t_maneuver_end = t_maneuver_start + abs(th_f - prev_state.attitude[0]) / prev_state.max_slew_rate

                if abs(t_maneuver_start - t_maneuver_end) > 0.0:
                    maneuver_action = ManeuverAction([th_f, 0, 0], t_maneuver_start, t_maneuver_end)
                    plan_i.append(maneuver_action)   
                else:
                    t_maneuver_end = None

            # move to target
            t_move_start = t_prev if t_maneuver_end is None else t_maneuver_end
            if isinstance(state, SatelliteAgentState):
                lat, lon, _ = measurement_req.lat_lon_pos
                df : pd.DataFrame = self.orbitdata.get_ground_point_accesses_future(lat, lon, t_move_start)
                
                t_move_end = None
                for _, row in df.iterrows():
                    if row['time index'] * self.orbitdata.time_step >= t_img:
                        t_move_end = row['time index'] * self.orbitdata.time_step
                        break

                if t_move_end is None:
                    # unpheasible path
                    self.log(f'Unheasible element in path. Cannot perform observation.', level=logging.DEBUG)
                    continue

                future_state : SatelliteAgentState = state.propagate(t_move_end)
                final_pos = future_state.pos

            elif isinstance(state, UAVAgentState):
                final_pos = measurement_req.pos
                dr = np.array(final_pos) - np.array(prev_state.pos)
                norm = np.sqrt( dr.dot(dr) )
                
                t_move_end = t_move_start + norm / state.max_speed

            else:
                raise NotImplementedError(f"cannot calculate travel time end for agent states of type {type(state)}")
            
            if t_move_end < t_img:
                plan_i.append( WaitForMessages(t_move_end, t_img) )
                
            t_img_start = t_img
            t_img_end = t_img_start + measurement_req.duration

            if isinstance(self._clock_config, FixedTimesStepClockConfig):
                dt = self._clock_config.dt
                if t_move_start < np.Inf:
                    t_move_start = dt * math.floor(t_move_start/dt)
                if t_move_end < np.Inf:
                    t_move_end = dt * math.ceil(t_move_end/dt)

                if t_img_start < np.Inf:
                    t_img_start = dt * math.floor(t_img_start/dt)
                if t_img_end < np.Inf:
                    t_img_end = dt * math.ceil((t_img_start + measurement_req.duration)/dt)
            
            if abs(t_move_start - t_move_end) >= 1e-3:
                move_action = TravelAction(final_pos, t_move_start, t_move_end)
                plan_i.append(move_action)
            
            # perform measurement
            main_measurement, _ = measurement_req.measurement_groups[subtask_index]
            measurement_action = MeasurementAction( 
                                                    measurement_req.to_dict(),
                                                    subtask_index, 
                                                    main_measurement,
                                                    u_exp,
                                                    t_img_start, 
                                                    t_img_end
                                                    )
            plan_i.append(measurement_action)  

            # TODO inform others of request completion
            # bid : Bid = subtask_bid.copy()
            # bid.set_performed(t_img_end)
            # plan.append(BroadcastMessageAction(MeasurementBidMessage(   self.get_parent_name(), 
            #                                                             self.get_parent_name(),
            #                                                             bid.to_dict() 
            #                                                         ).to_dict(),
            #                                     t_img_end
            #                                     )
            #             )
            plan.extend(plan_i)
        
        return plan

    """
    --------------------
    LOGGING AND TEARDOWN
    --------------------
    """
    # def record_runtime(self, func : function) -> Any:
    #     def wrapper(*args, **kwargs):
    #         t_0 = time.perf_counter()
    #         vals = func(*args, **kwargs)
    #         dt = time.perf_counter() - t_0
    #         self.stats[func.__name__].append(dt)
    #     return wrapper

    @abstractmethod
    def log_results(self, dsc : str, results : dict, level=logging.DEBUG) -> None:
        """
        Logs current results at a given time for debugging purposes

        ### Argumnents:
            - dsc (`str`): description of what is to be logged
            - results (`dict`): results to be logged
            - level (`int`): logging level to be used
        """
        pass

    def log_task_sequence(self, dsc : str, sequence : list, level=logging.DEBUG) -> None:
        """
        Logs a sequence of tasks at a given time for debugging purposes

        ### Argumnents:
            - dsc (`str`): description of what is to be logged
            - sequence (`list`): list of tasks to be logged
            - level (`int`): logging level to be used
        """
        if self._logger.getEffectiveLevel() <= level:
            out = f'\n{dsc} [Iter {self.iter_counter}] = ['
            for req, subtask_index in sequence:
                req : MeasurementRequest
                subtask_index : int
                split_id = req.id.split('-')
                
                if sequence.index((req, subtask_index)) > 0:
                    out += ', '
                out += f'({split_id[0]}, {subtask_index})'
            out += ']\n'

            self.log(out,level)

    @abstractmethod
    def log_changes(self, dsc : str, changes : list, level=logging.DEBUG) -> None:
        pass

    def log_plan(self, results : dict, plan : list, t : Union[int, float], level=logging.DEBUG) -> None:
        headers = ['t', 'req_id', 'subtask_index', 't_start', 't_end', 't_img', 'u_exp']
        data = []

        for action in plan:
            if isinstance(action, MeasurementAction):
                req : MeasurementRequest = MeasurementRequest.from_dict(action.measurement_req)
                task_id = req.id.split('-')[0]
                subtask_index : int = action.subtask_index
                subtask_bid : Bid = results[req.id][subtask_index]
                t_img = subtask_bid.t_img
                winning_bid = subtask_bid.winning_bid
            elif isinstance(action, TravelAction) or isinstance(action, ManeuverAction):
                task_id = action.id.split('-')[0]
                subtask_index = -1
                t_img = -1
                winning_bid = -1
            else:
                continue
            
            line_data = [   t,
                            task_id,
                            subtask_index,
                            np.round(action.t_start,3 ),
                            np.round(action.t_end,3 ),
                            np.round(t_img,3 ),
                            np.round(winning_bid,3)
            ]
            data.append(line_data)

        df = pd.DataFrame(data, columns=headers)
        self.log(f'\nPLANNER HISTORY\n{str(df)}\n', level)

    
    async def teardown(self) -> None:
        # log plan history
        headers = ['plan_index', 't_plan', 'req_id', 'subtask_index', 't_img', 'u_exp']
        data = []
        
        for i in range(len(self.plan_history)):
            t_plan, plan = self.plan_history[i]
            t_plan : float; plan : list

            for action in plan:
                if not isinstance(action, MeasurementAction):
                    continue

                req : MeasurementRequest = MeasurementRequest.from_dict(action.measurement_req)
                line_data = [   i,
                                np.round(t_plan,3),
                                req.id.split('-')[0],
                                action.subtask_index,
                                np.round(action.t_start,3 ),
                                np.round(action.u_exp,3)
                ]
                data.append(line_data)

        df = pd.DataFrame(data, columns=headers)
        self.log(f'\nPLANNER HISTORY\n{str(df)}\n', level=logging.WARNING)
        df.to_csv(f"{self.results_path}/{self.get_parent_name()}/planner_history.csv", index=False)

        await super().teardown()