import itertools
import logging
from queue import Queue
import time
from typing import Any, Callable
from numpy import Inf
import pandas as pd
from traitlets import Callable

from dmas.utils import runtime_tracker
from dmas.clocks import *

from chess3d.agents.states import SimulationAgentState
from chess3d.agents.planning.plan import Plan, Preplan, Replan
from chess3d.agents.planning.planners.consensus.bids import Bid, BidComparisonResults, RebroadcastComparisonResults
from chess3d.agents.planning.planner import AbstractReplanner
from chess3d.agents.science.utility import *
from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.science.requests import *
from chess3d.agents.states import *
from chess3d.messages import *

class AbstractConsensusReplanner(AbstractReplanner):    
    def __init__(self, 
                 utility_func: Callable, 
                 max_bundle_size : int = 1,
                 replan_threshold : int = 1,
                 planning_horizon : float = np.Inf,
                 logger: logging.Logger = None
                 ) -> None:
        super().__init__(logger=logger)

        # initialize variables
        self.bundle = []
        self.path = []
        self.results = {}
        self.bids_to_rebroadcasts = []
        self.completed_measurements = set()
        self.other_plans = {}
        self.recently_completed_measurements = set()
        self.planner_changes = []
        self.agent_orbitdata : OrbitData = None
        self.preplan : Preplan = None

        # set paremeters
        self.utility_func = utility_func
        self.max_bundle_size = max_bundle_size
        self.replan_threshold = replan_threshold
        self.planning_horizon = planning_horizon

    @runtime_tracker
    def update_percepts(self, 
                        state: SimulationAgentState, 
                        current_plan: Plan, 
                        incoming_reqs: list, 
                        relay_messages: list, 
                        misc_messages: list, 
                        completed_actions: list, 
                        aborted_actions: list, 
                        pending_actions: list, 
                        ) -> None:
       
        # update other percepts
        super().update_percepts(state,
                                current_plan,
                                incoming_reqs,
                                relay_messages,
                                misc_messages,
                                completed_actions,
                                aborted_actions,
                                pending_actions)

        # update latest preplan
        if state.t == current_plan.t and isinstance(current_plan, Preplan): 
            self.preplan : Preplan = current_plan.copy() 
            
            #TODO reset bundle and results
            self.bundle = []
            self.path = []
            self.results = {}

        # compile completed measurements
        completed_measurements = self._compile_completed_observations(completed_actions, 
                                                                        misc_messages)
        self.completed_measurements.update(completed_measurements)
        self.recently_completed_measurements = completed_measurements

        # update incoming bids
        self.incoming_bids = [  Bid.from_dict(msg.bid) 
                                for msg in misc_messages 
                                if isinstance(msg, MeasurementBidMessage)]

        # check if any new measurement requests have been received
        new_req_bids : list[Bid] = self.compile_new_measurement_request_bids(state)
        self.incoming_bids.extend(new_req_bids)
        
    def _compile_completed_observations(self, 
                                        completed_actions : list, 
                                        misc_messages : list) -> set:
        """ Reads incoming precepts and compiles all measurement actions performed by the parent agent or other agents """
        # checks measurements performed by the parent agent
        completed_observations = {action for action in completed_actions
                                  if isinstance(action, ObservationAction)
                                  and action not in self.completed_measurements}
        
        # checks measuremetns performed by other agents
        completed_observations.update({action_from_dict(**msg.observation_action) 
                                       for msg in misc_messages
                                       if isinstance(msg, ObservationPerformedMessage)})

        return completed_observations
    
    @runtime_tracker
    def compile_new_measurement_request_bids(self, 
                                             state : SimulationAgentState) -> list:
        """ Checks for all requests that havent been considered in current bidding process """
        
        new_request_bids : list[Bid] = []
        for req in [req for req in self.known_reqs if req.id not in self.results]:
            req : MeasurementRequest

            # create new bids for measurement request
            bids : list[Bid] = self._generate_bids_from_request(req, state)

            # add to list of new bids
            new_request_bids.extend(bids)

        return new_request_bids

    @abstractmethod
    def _generate_bids_from_request(self, req : MeasurementRequest, state : SimulationAgentState) -> list:
        """ Creages bids from given measurement request """    

    @runtime_tracker
    def needs_planning(self, 
                       state : SimulationAgentState, 
                       specs : object,
                       current_plan : Plan,
                       orbitdata : OrbitData
                       ) -> bool:   
                
        # perform consesus phase
        # ---------------------------------
        # DEBUGGING OUTPUTS 
        # self.log_results('PRE-CONSENSUS PHASE', state, self.results)
        # print(f'length of path: {len(self.path)}\nbids to rebradcast: {len(self.bids_to_rebroadcasts)}')
        # print(f'bundle:')
        # for req, subtask_index, bid in self.bundle:
        #     req : MeasurementRequest
        #     bid : Bid
        #     id_short = req.id.split('-')[0]
        #     print(f'\t{id_short}, {subtask_index}, {np.round(bid.t_img,3)}, {np.round(bid.bid)}')
        # print('')
        # ---------------------------------

        self.results, self.bundle, \
            self.path, _, bids_to_rebroadcasts = self.consensus_phase(state,
                                                                      self.results,
                                                                      self.bundle, 
                                                                      self.path, 
                                                                      self.incoming_bids,
                                                                      self.recently_completed_measurements)
        self.bids_to_rebroadcasts.extend(bids_to_rebroadcasts)
        
        assert self.is_task_path_valid(state, specs, self.path, orbitdata)
        
        # ---------------------------------
        # DEBUGGING OUTPUTS 
        # self.log_results('CONSENSUS PHASE', state, self.results)
        # print(f'length of path: {len(self.path)}\nbids to rebradcast: {len(self.bids_to_rebroadcasts)}')
        # print(f'bundle:')
        # for req, subtask_index, bid in self.bundle:
        #     req : MeasurementRequest
        #     bid : Bid
        #     id_short = req.id.split('-')[0]
        #     print(f'\t{id_short}, {subtask_index}, {np.round(bid.t_img,3)}, {np.round(bid.bid)}')
        # print('')
        # ---------------------------------

        if len(self.bids_to_rebroadcasts) >= self.replan_threshold:
            return True
        
        if len(self.bundle) < self.max_bundle_size and self.recently_completed_measurements:
            available_reqs = self._get_available_requests(state, specs, self.results, self.bundle, self.path, orbitdata)
            
            if len(available_reqs) > 0:
                x = 1

            return len(available_reqs) > 0
        
        return False
        
        # TODO: Consider preplan
    
    @runtime_tracker
    def generate_plan(  self, 
                        state : SimulationAgentState,
                        specs : object,
                        current_plan : Plan,
                        clock_config : ClockConfig,
                        orbitdata : dict = None
                    ) -> list:
        
        # -------------------------------
        # DEBUG PRINTOUTS
        # self.log_results('PRE-PLANNING PHASE', state, self.results)
        # -------------------------------

        # check if bundle is full
        if len(self.bundle) < self.max_bundle_size:
            # There is room in the bundle; perform bidding phase
            self.results, self.bundle, self.path, self.planner_changes = \
                self.planning_phase(state, specs, self.results, self.bundle, self.path, orbitdata)
        
        assert self.is_task_path_valid(state, specs, self.path, orbitdata)
        # TODO check convergence

        # generate plan from bids
        plan : Replan = self._plan_from_path(state, specs, self.results, self.path, clock_config, orbitdata)     
            
        # reset broadcast list
        # TODO only reset after broadcasts have been performed?
        self.bids_to_rebroadcasts = []

        # -------------------------------
        # DEBUG PRINTOUTS
        # self.log_results('PLANNING PHASE', state, self.results)
        # print(f'bundle:')
        # for req, subtask_index, bid in self.bundle:
        #     req : MeasurementRequest
        #     bid : Bid
        #     id_short = req.id.split('-')[0]
        #     print(f'\t{id_short}, {subtask_index}, {np.round(bid.t_img,3)}, {np.round(bid.bid)}')
        # print('')
        # -------------------------------

        # output plan
        return plan
        
    @runtime_tracker
    def _plan_from_path(self, 
                        state : SimulationAgentState, 
                        specs : object,
                        results : dict,
                        path : list,
                        clock_config: ClockConfig, 
                        orbitdata: dict = None
                        ) -> Replan:
        """ creates a new plan to be performed by the agent based on the results of the planning phase """

        # schedule observations
        observations : list = self._schedule_observations(state, specs, results, path, orbitdata)

        # schedule bruadcasts
        broadcasts : list = self._schedule_broadcasts(state, observations, orbitdata)

        # generate maneuver and travel actions from observations
        maneuvers : list = self._schedule_maneuvers(state, specs, observations, clock_config, orbitdata)

        # generate wait actions 
        waits : list = self._schedule_waits(state)
        
        return Replan(observations, broadcasts, maneuvers, waits, t=state.t, t_next=self.preplan.t_next)
    
    @runtime_tracker
    def _schedule_observations(self, state : SimulationAgentState, specs : object, results : dict, path : list, orbitdata : OrbitData) -> list:
        """ compiles and merges lists of measurement actions to be performed by the agent """
        

        # generate proposed observation actions
        proposed_observations = []
        for req, main_measurement, t_img, _ in path:
            req : MeasurementRequest
            th_imgs = [ th_img
                        for t,_,_,lat,lon,_,th_img,_,_,_,instrument,_ in orbitdata.gp_access_data.values
                        if  abs(req.target[0] - lat) <= 1e-3
                        and abs(req.target[1] - lon) <= 1e-3
                        and abs(t_img - t*orbitdata.time_step) <= 1e-3
                        and instrument == main_measurement
                        ]
            
            proposed_observations.append(ObservationAction(main_measurement,
                                                           req.target,
                                                           th_imgs.pop(),
                                                           t_img))
        proposed_observations.sort(key=lambda a : a.t_start)

        # check for if the right number of measurements was created
        assert len(proposed_observations) == len(path)

        # gather observations from preplan
        planned_observations = [action for action in self.preplan
                                if isinstance(action, ObservationAction)]
        planned_observations.sort(key=lambda a : a.t_start)

        # combine observation list
        observations = []
        while proposed_observations and planned_observations:
            # get next actions in the lists
            proposed_observation : ObservationAction = proposed_observations[0]
            planned_observation  : ObservationAction = planned_observations[0]

            # construct temporary observation sequence
            temp_observations = [action for action in observations]
            if proposed_observation.t_start <= planned_observation.t_start:
                temp_observations.append(proposed_observation)
            else:
                temp_observations.append(planned_observation)

            # check if the temporary observation sequence is valid
            if self.is_observation_path_valid(state, specs, temp_observations):
                # is valid; remove added action from queue
                if proposed_observation.t_start <= planned_observation.t_start:
                    observations.append(proposed_observations.pop(0))
                else:
                    if (    observations
                        and abs(observations[-1].target[0]-planned_observation.target[0]) <= 1e-3
                        and abs(observations[-1].target[1]-planned_observation.target[1]) <= 1e-3
                        and abs(observations[-1].target[2]-planned_observation.target[2]) <= 1e-3):
                        # is already in plan; ignore
                        planned_observations.pop(0)
                    else:
                        # add to new observations plan
                        observations.append(planned_observations.pop(0))
            else:
                if temp_observations[-1] == proposed_observation:
                    observations.pop()
                else:
                    planned_observations.pop(0)

        while proposed_observations:
            # get next actions in the lists
            proposed_observation : ObservationAction = proposed_observations[0]

            # construct temporary observation sequence
            temp_observations = [action for action in observations]
            temp_observations.append(proposed_observation)  

            # check if the temporary observation sequence is valid
            if self.is_observation_path_valid(state, specs, temp_observations):
                # is valid; remove added action from queue
                observations.append(proposed_observations.pop(0))
            else:
                observations.pop()

        while planned_observations:
            # get next actions in the lists
            planned_observation  : ObservationAction = planned_observations[0]

            # construct temporary observation sequence
            temp_observations = [action for action in observations]
            temp_observations.append(planned_observation) 

            # check if the temporary observation sequence is valid
            if self.is_observation_path_valid(state, specs, temp_observations):
                # is valid; check if already being observed:
                if (    observations
                    and abs(observations[-1].target[0]-planned_observation.target[0]) <= 1e-3
                    and abs(observations[-1].target[1]-planned_observation.target[1]) <= 1e-3
                    and abs(observations[-1].target[2]-planned_observation.target[2]) <= 1e-3):
                    # is already in plan; ignore
                    planned_observations.pop(0)
                else:
                    # add to new observations plan
                    observations.append(planned_observations.pop(0))
            else:
                planned_observations.pop(0)

        return observations
    
    @runtime_tracker
    def _schedule_broadcasts(self, 
                             state: SimulationAgentState, 
                             observations: list, 
                             orbitdata: dict
                             ) -> list:
        broadcasts : list[BroadcastMessageAction] = super()._schedule_broadcasts(state,
                                                                                 orbitdata)
        
        # compile bids to be broadcasted
        compiled_bids : list[Bid] = [bid for bid in self.bids_to_rebroadcasts]
        compiled_bids.extend(self.planner_changes)
        
        # sort and make sure the latest update of each task is broadcasted
        bids = {}
        for bid_to_rebroadcast in compiled_bids: 
            bid_to_rebroadcast : Bid
            req_id = bid_to_rebroadcast.req_id

            if req_id not in bids:
                bids[req_id] = {}
            
            if bid_to_rebroadcast.main_measurement not in bids[req_id]:
                bids[req_id][bid_to_rebroadcast.main_measurement] = bid_to_rebroadcast

            else:
                current_bid : Bid = bids[req_id][bid_to_rebroadcast.main_measurement]
                if current_bid.t_update <= bid_to_rebroadcast.t_update:
                    bids[req_id][bid_to_rebroadcast.main_measurement] = bid_to_rebroadcast

        # Find best path for broadcasts
        relay_path, t_start = self._create_broadcast_path(state, orbitdata, state.t)

        # Schedule bid re-broadcast and planner changes
        bids_out = [MeasurementBidMessage(state.agent_name, state.agent_name, bid_to_rebroadcast.to_dict(), path=relay_path) 
                    for req_id in bids 
                    for _, bid_to_rebroadcast in bids[req_id].items() 
                    if bid_to_rebroadcast is not None
                    and t_start >=0]
        broadcasts.extend([BroadcastMessageAction(msg.to_dict(), t_start) for msg in bids_out])

        # ensure the right number of broadcasts were created
        assert len(bids_out) <= len(compiled_bids)
        
        return broadcasts

    @runtime_tracker
    def _compile_broadcast_bids(self, planner_changes : list) -> list:        
        """ Compiles changes in bids from consensus and planning phase and returns a list of the most updated bids """
        broadcast_bids = {}

        bids = [bid for bid in self.bids_to_rebroadcasts]
        bids.extend(planner_changes)
        for bid in bids:
            bid : Bid

            if bid.req_id not in broadcast_bids:
                req : MeasurementRequest = MeasurementRequest.from_dict(bid.req)
                broadcast_bids[bid.req_id] = [None for _ in req.dependency_matrix]

            current_bid : Bid = broadcast_bids[bid.req_id][bid.subtask_index]
            
            if (current_bid is None 
                or current_bid in planner_changes
                or bid.bidder == current_bid.bidder
                or bid.t_update >= current_bid.t_update
                ):
                broadcast_bids[bid.req_id][bid.subtask_index] = bid.copy()       

        out = []
        for req_id in broadcast_bids:
            out.extend([bid for bid in broadcast_bids[req_id] if bid is not None])
            
        return out
    
    @runtime_tracker
    def _schedule_waits(self, state : SimulationAgentState) -> list:
        """ schedules periodic rescheduling checks """

        # horizon = self.planning_horizon if self.planning_horizon < np.Inf else 10*24*3600
        
        # # n_steps = int((state.t + horizon)/self.replan_period)
        # n_steps = 1
        
        # waits = [WaitForMessages(state.t+self.replan_period*step, 
        #                          state.t+self.replan_period*step)
        #          for step in range(1,n_steps+1)]

        # return waits 
        # t_horizon = state.t + self.planning_horizon
        # return [WaitForMessages(t_horizon, t_horizon)]
    
        return []

    """
    -----------------------
        CONSENSUS PHASE
    -----------------------
    """
    @runtime_tracker
    def consensus_phase(  
                                self, 
                                state : SimulationAgentState,
                                results : dict, 
                                bundle : list, 
                                path : list, 
                                bids_received : list,
                                completed_measurements : list,
                                level : int = logging.DEBUG
                            ) -> None:
        """
        Evaluates incoming bids and updates current results and bundle
        """
        
        # check if tasks were performed
        results, bundle, path, \
            done_changes, done_rebroadcasts = self.check_request_completion(state, results, bundle, path, completed_measurements, level)
        # ensure that path and bundle are in agreement
        assert all([(req, instrument_name, bid.t_img, bid.bid) in path for req, instrument_name, bid in bundle])

        # check if tasks expired
        results, bundle, path, \
            exp_changes, exp_rebroadcasts = self.check_request_end_time(state, results, bundle, path, level)
        # ensure that path and bundle are in agreement
        assert all([(req, instrument_name, bid.t_img, bid.bid) in path for req, instrument_name, bid in bundle])

        # compare bids with incoming messages
        results, bundle, path, \
            comp_changes, comp_rebroadcasts = self.compare_bids(state, results, bundle, path, bids_received, level)
        # ensure that path and bundle are in agreement
        assert all([(req, instrument_name, bid.t_img, bid.bid) in path for req, instrument_name, bid in bundle])

        # compile changes
        changes = []
        changes.extend(done_changes)
        changes.extend(exp_changes)
        changes.extend(comp_changes)

        # compile rebroadcasts
        rebroadcasts = []
        rebroadcasts.extend(done_rebroadcasts)
        rebroadcasts.extend(exp_rebroadcasts)
        rebroadcasts.extend(comp_rebroadcasts)

        return results, bundle, path, changes, rebroadcasts
    
    @runtime_tracker
    def check_request_completion(self, 
                                 state : SimulationAgentState,
                                 results : dict, 
                                 bundle : list, 
                                 path : list, 
                                 completed_measurements : list,
                                 level=logging.DEBUG
                                 ) -> tuple:
        """
        Checks if a subtask or a mutually exclusive subtask has already been performed 

        ### Returns
            - results
            - bundle
            - path
            - changes
        """
        # initialize bundle changes and rebroadcast lists
        changes = []
        rebroadcasts = []

        # update results using incoming messages
        for action in completed_measurements:
            action : ObservationAction

            # check if observation was performed to respond to a measurement request
            completed_req : MeasurementRequest = self._get_completed_request(action)

            if completed_req is None or action.instrument_name not in completed_req.observation_types:
                # observation was not performed to respond to a measurement request; ignore
                continue
            
            # set bid as completed           
            bid : Bid = results[completed_req.id][action.instrument_name]
            updated_bid : Bid = bid.copy()
            updated_bid._perform(state.t)
            results[completed_req.id][action.instrument_name] = updated_bid

            # add to changes and rebroadcast lists
            changes.append(updated_bid.copy())
            rebroadcasts.append(updated_bid.copy())

        # check for task completion in bundle
        for task in [(req, instrument_name, current_bid)
                    for req, instrument_name, current_bid in bundle
                    if results[req.id][instrument_name].performed]:
            ## remove all completed tasks from bundle
            self.bundle.remove(task)
        
        # check for task completion in path
        for task in [(req, instrument_name, t_img, u_exp)
                    for req, instrument_name, t_img, u_exp in path
                    if req.id in results
                    and results[req.id][instrument_name].performed]:
            ## remove all completed tasks from path
            self.path.remove(task) 

        return results, bundle, path, changes, rebroadcasts
    
    def _get_completed_request(self, action : ObservationAction) -> MeasurementRequest:
        reqs = {req 
                for req in self.known_reqs
                if isinstance(req, MeasurementRequest)
                and abs(req.target[0] - action.target[0]) <= 1e-3
                and abs(req.target[1] - action.target[1]) <= 1e-3
                and abs(req.target[2] - action.target[2]) <= 1e-3
                }
        return reqs.pop() if reqs else None

    def _get_matching_request(self, id : list) -> MeasurementRequest:
        reqs = {req for req in self.known_reqs if req.id == id}
        return reqs.pop() if reqs else None

    @runtime_tracker
    def check_request_end_time(self, 
                               state : SimulationAgentState,
                               results : dict, 
                               bundle : list, 
                               path : list, 
                               level=logging.DEBUG
                               ) -> tuple:
        """
        Checks if measurement requests have expired and can no longer be performed

        ### Returns a tuple with elements:
            - results
            - bundle
            - path
            - changes
            - rebroadcasts 
        """

        # initialize bundle changes and rebroadcast lists
        changes = []
        rebroadcasts = []

        # release tasks from bundle if `t_end` has passed
        task_to_remove = None
        for req, instrument_name, bid in bundle:
            req : MeasurementRequest; bid : Bid
            if req.t_end < state.t and not bid.performed:
                task_to_remove = (req, instrument_name, bid)
                break

        # if task has expired, release from bundle and path with all subsequent tasks
        if task_to_remove is not None:
            expired_index = bundle.index(task_to_remove)
            for _ in range(expired_index, len(bundle)):
                # remove from bundle
                measurement_req, instrument_name, current_bid = bundle.pop(expired_index)

                # remove from path
                path.remove((measurement_req, instrument_name, current_bid.t_img, current_bid.bid))

                # reset bid results
                current_bid : Bid; measurement_req : MeasurementRequest
                resetted_bid : Bid = current_bid.update(None, BidComparisonResults.RESET, state.t)
                results[measurement_req.id][instrument_name] = resetted_bid

                rebroadcasts.append(resetted_bid)
                changes.append(resetted_bid)

        return results, bundle, path, changes, rebroadcasts

    @runtime_tracker
    def compare_bids(
                    self, 
                    state : SimulationAgentState, 
                    results : dict, 
                    bundle : list, 
                    path : list, 
                    bids_received : list,
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

        for their_bid in bids_received:
            their_bid : Bid            
            
            # get matching request
            req : MeasurementRequest = self._get_matching_request(their_bid.req_id)

            # check bids are for new requests
            is_new_req : bool = their_bid.req_id not in results

            if is_new_req:
                # create a new blank bid and save it to results
                bids : list[Bid] = self._generate_bids_from_request(req, state)
                results[their_bid.req_id] = {bid.main_measurement : bid for bid in bids}

                # add to changes broadcast in case this request was generated by me
                if(their_bid.bidder == state.agent_name 
                   and their_bid.winner == their_bid.NONE):
                    my_bid : Bid = results[their_bid.req_id][their_bid.main_measurement]
                    rebroadcasts.append(my_bid.copy())
                                    
            # compare bids
            my_bid : Bid = results[their_bid.req_id][their_bid.main_measurement]
            # self.log(f'comparing bids...\nmine:  {str(my_bid)}\ntheirs: {str(their_bid)}', level=logging.DEBUG) #DEBUG PRINTOUT

            _, rebroadcast_result = my_bid.compare(their_bid)
            updated_bid : Bid = my_bid.update(their_bid, state.t)
            bid_changed = my_bid != updated_bid

            # update results with modified bid
            results[their_bid.req_id][their_bid.main_measurement] = updated_bid
            # self.log(f'\nupdated: {my_bid}\n', level=logging.DEBUG) #DEBUG PRINTOUT
                
            # if relevant changes were made, add to changes and rebroadcast lists respectively
            if bid_changed or is_new_req:
                # changed_bid : Bid = updated_bid if not is_new_req else my_bid
                # changes.append(changed_bid)
                changes.append(updated_bid)

            if (rebroadcast_result is RebroadcastComparisonResults.REBROADCAST_EMPTY 
                  or rebroadcast_result is RebroadcastComparisonResults.REBROADCAST_SELF):
                rebroadcasts.append(updated_bid)
            elif rebroadcast_result is RebroadcastComparisonResults.REBROADCAST_OTHER:
                rebroadcasts.append(their_bid)

            # if outbid for a bids in the bundle; release outbid and subsequent bids in bundle and path
            if ((req, my_bid.main_measurement, my_bid) in bundle 
                and updated_bid.winner != state.agent_name):

                outbid_index = bundle.index((req, my_bid.main_measurement, my_bid))

                for req, subtask_index, new_bid in bundle:
                    new_bid : Bid
                    assert (req, subtask_index, new_bid.t_img, new_bid.bid) in path

                # remove all subsequent bids
                for bundle_index in range(outbid_index, len(bundle)):
                    # remove from bundle
                    measurement_req, subtask_index, current_bid = bundle.pop(outbid_index)

                    # remove from path
                    path.remove((measurement_req, subtask_index, current_bid.t_img, current_bid.bid))

                    # reset bid results
                    current_bid : Bid; measurement_req : MeasurementRequest
                    if bundle_index > outbid_index:
                        reset_bid : Bid = current_bid.update(None, BidComparisonResults.RESET, state.t)
                        results[measurement_req.id][subtask_index] = reset_bid

                        rebroadcasts.append(reset_bid)
                        changes.append(reset_bid)
            
            # if outbid for a bid in the bundle; remove from path
            if ((req, my_bid.main_measurement, my_bid.t_img, my_bid.bid) in path
                and updated_bid.winner != state.agent_name):

                path.remove((req, my_bid.main_measurement, my_bid.t_img, my_bid.bid))
        
        return results, bundle, path, changes, rebroadcasts

    """
    -----------------------
        PLANNING PHASE
    -----------------------
    """
    def calc_path_times():
        pass

    @runtime_tracker
    def planning_phase( self, 
                        state : SimulationAgentState, 
                        specs : object,
                        results : dict, 
                        bundle : list,
                        path : list,
                        orbitdata : OrbitData
                    ) -> tuple:
        """
        Creates a modified plan from all known requests and current plan
        """
        # initialzie changes
        changes = []

        # get requests that can be bid on by this agent
        available_reqs : list = self._get_available_requests(state, specs, results, bundle, path, orbitdata)
                
        t_0 = time.perf_counter()
        # -------------------------------------
        # TEMPORARY FIX: resets bundle and replans from scratch
        if len(bundle) > 0:
            # reset path
            path = []

            # reset results
            for req, subtask_index, _ in bundle:
                bid : Bid = results[req.id][subtask_index]
                bid._reset(state.t)
                results[req.id][subtask_index] = bid

                available_reqs.append((req,subtask_index))

            # reset bundle 
            bundle = []
        # -------------------------------------

        if len(bundle) == 0: # create bundle from scratch
            # generate path 
            max_path = self._generate_path(state, specs, results, available_reqs, orbitdata)

            # update bundle and results 
            bundle = []
            for req, subtask_index, t_img, u_exp in max_path:
                req : MeasurementRequest

                # update bid
                old_bid : Bid = results[req.id][subtask_index]
                new_bid : Bid = old_bid.copy()
                new_bid.set(u_exp, t_img, state.t)

                # place in bundle
                bundle.append((req, subtask_index, new_bid))

                # if changed, update results
                if old_bid != new_bid:
                    changes.append(new_bid.copy())
                    results[req.id][subtask_index] = new_bid

            # update path
            path = [path_elem for path_elem in max_path]

        else: # add tasks to the bundle
            raise NotImplementedError('Repairing bundle not yet implemented')
        
        dt = time.perf_counter() - t_0
        
        # check if path is valid
        assert self.is_task_path_valid(state, specs, path, orbitdata)

        # ensure that bundle is within the allowed size
        assert len(bundle) <= self.max_bundle_size

        # return results
        return results, bundle, path, changes 
    
        # TODO implement bundle repair strategy
        #     max_path = [path_element for path_element in path]
        #     max_path_utility = sum([u_exp for _,_,_,u_exp in path])
        #     max_req = -1
        #     t_img = None
        #     u_exp = None

        #     while (len(bundle) < self.max_bundle_size   # there is room in the bundle
        #            and len(available_reqs) > 0          # there tasks available
        #            and max_req is not None              # a task was added in the previous iteration
        #            ):
        #         # search for best bid
        #         max_req = None
        #         for req, subtask_index in available_reqs:
        #             proposed_path, proposed_path_utility, \
        #                 proposed_t_img, proposed_utility = self.calc_path_bid(state, results, path, req, subtask_index)

        #             if proposed_path is None:
        #                 continue

        #             current_bid : Bid = results[req.id][subtask_index]

        #             if (max_path_utility < proposed_path_utility
        #                 and current_bid.winning_bid < proposed_utility):
        #                 max_path = [path_element for path_element in proposed_path]
        #                 max_path_utility = proposed_path_utility
        #                 max_req = (req, subtask_index)
        #                 t_img = proposed_t_img
        #                 u_exp = proposed_utility
                
        #         # check if a task to be added was found
        #         if max_req is not None:
        #             # update path 
        #             path = [path_element for path_element in max_path]

        #             # update bids
        #             for req, subtask_index, t_img, u_exp in max_path:
        #                 req : MeasurementRequest

        #                 # update bid
        #                 old_bid : Bid = results[req.id][subtask_index]
        #                 new_bid : Bid = old_bid.copy()
        #                 new_bid.set(u_exp, t_img, state.t)

        #                 # if changed, update results
        #                 if old_bid != new_bid:
        #                     changes.append(new_bid.copy())
        #                     results[req.id][subtask_index] = new_bid

        #             # add to bundle
        #             bid = results[req.id][subtask_index]
        #             bundle.append((req, subtask_index, bid))
    
    def _generate_path(self, state :SimulationAgentState, specs : object, results : dict, available_reqs : list, orbitdata : OrbitData) -> list:
        # count maximum number of paths to be searched
        n_max = 0
        for n in range(1,self.max_bundle_size+1):
            n_max += len(list(itertools.permutations(available_reqs, n)))
        
        # initilize search counter
        n_visited = 0

        # initialize tree search
        queue = Queue()
        queue.put([])
        max_path = []
        max_path_utility = 0.0

        # start tree search
        while not queue.empty():
            # update counter
            n_visited += 1 
            
            # get next path on the queue
            path_i : list = queue.get()
                
            # check if it out-performs the current best path
            if any([results[req.id][main_measurement].bid >= u_exp     # at least one bid is outbid by competitors
                    for req,main_measurement,_,u_exp in path_i]):
                # ignore path
                continue
            
            # check if it outbids competitors
            path_utility = sum([u_exp for _,_,_,u_exp in path_i])
            if (path_utility > max_path_utility                         # path utility is increased
                and all([results[req.id][main_measurement].bid < u_exp     # all new bids outbid competitors
                         for req,main_measurement,_,u_exp in path_i])
                ):
                max_path = [path_element for path_element in path_i]
                max_path_utility = path_utility


            # check if there is room to be added to the bundle
            if len(path_i) >= self.max_bundle_size:
                continue

            # add available requests to the path and place it in the queue
            path_reqs = [(req, main_measurement) for req,main_measurement,_,_ in path_i]
            reqs_to_add = [(req, main_measurement) 
                        for req, main_measurement in available_reqs
                        if (req, main_measurement) not in path_reqs
                        ]
            
            for req, main_measurement in reqs_to_add:
                req : MeasurementRequest
                # copy current path
                path_j = [path_element_i for path_element_i in path_i]
                
                # add request to path
                path_j.append((req, main_measurement, -1, -1))
                
                # calculate performance
                t_img = self.calc_imaging_time(state, specs, path_j, req, main_measurement, orbitdata)
                u_exp = self.utility_func(req.to_dict(), t_img)

                # update values
                path_j[-1] = (req, main_measurement, t_img, u_exp)

                # only add to queue if the path can be performed
                if self.is_task_path_valid(state, specs, path_j, orbitdata): queue.put(path_j)
        
        return max_path

    @runtime_tracker
    def _get_available_requests(self, 
                                state : SimulationAgentState, 
                                specs : object,
                                results : dict, 
                                bundle : list,
                                path : list,
                                orbitdata : OrbitData
                                ) -> list:
        """ Returns a list of known requests that can be performed within the current planning horizon """
        path_reqs = [(req, main_measurement) for req, main_measurement, _, _ in path]
        
        # find requests that can be bid on
        biddable_requests = []        
        for req_id in results:
            req = None
            for main_measurement in results[req_id]:
                bid : Bid = results[req_id][main_measurement]

                if req is None: req = self._get_matching_request(bid.req_id)

                # check if the agent can bid on the tasks
                if not self._can_bid(state, specs, results, req, main_measurement):
                    continue

                # check if the agent can access the task
                if not self._can_access(state, req, orbitdata):
                    continue 
                
                # check if task is already in the bundle
                if (req, main_measurement, bid) in bundle:
                    continue

                # check if task is already in the path
                if (req, main_measurement) in path_reqs:
                    continue

                # check if the task has already been performed
                if self.__request_has_been_performed(results, req, main_measurement, state.t):
                    continue

                biddable_requests.append((req, bid.main_measurement))  
                                    
        # find access intervals 
        n_intervals = self.max_bundle_size - len(bundle)
        intervals : list = self._get_available_intervals(biddable_requests, n_intervals, orbitdata)

        # find biddable requests that can be accessed in the next observation intervals
        available_requests = []
        for req, main_measurement in biddable_requests:
            req : MeasurementRequest
            bid : Bid = results[req.id][main_measurement]
            t_arrivals = [t*orbitdata.time_step 
                          for t,*_ in orbitdata.gp_access_data.values
                          for t_start, t_end in intervals
                          if req.t_start <= t*orbitdata.time_step <= req.t_end 
                          and t_start <= t*orbitdata.time_step <= t_end]

            if t_arrivals:
                available_requests.append((req, main_measurement))

        return available_requests
        
    @abstractmethod
    def is_task_path_valid(self, state : SimulationAgentState, specs : object, path : list, orbitdata : OrbitData) -> bool:
        pass
   
    def _can_access( self, 
                     state : SimulationAgentState, 
                     req : MeasurementRequest,
                     orbitdata : OrbitData = None
                     ) -> bool:
        """ Checks if an agent can access the location of a measurement request """
        if isinstance(state, SatelliteAgentState):
            t_arrivals = [t_img*orbitdata.time_step 
                          for t_img,_,_,lat,lon,*_,instrument,_ in orbitdata.gp_access_data.values
                          if req.t_start <= t_img*orbitdata.time_step <= req.t_end
                          and instrument in req.observation_types
                          and abs(lat - req.target[0]) <= 1e-3
                          and abs(lon - req.target[1]) <= 1e-3
                          ]
            
            return len(t_arrivals) > 0
        else:
            raise NotImplementedError(f"listing of available requests for agents with state of type {type(state)} not yet supported.")

        
    @abstractmethod
    def _can_bid(self, 
                state : SimulationAgentState, 
                specs : object,
                results : dict,
                req : MeasurementRequest, 
                main_measurement : str
                ) -> bool:
        """ Checks if an agent has the ability to bid on a measurement task """
        pass

    def __request_has_been_performed(self, 
                                     results : dict, 
                                     req : MeasurementRequest, 
                                     main_measurement : int, 
                                     t : Union[int, float]) -> bool:
        """ Check if subtask at hand has already been performed """
        current_bid : Bid = results[req.id][main_measurement]
        subtask_already_performed = t > current_bid.t_img and current_bid.winner != Bid.NONE

        return subtask_already_performed or current_bid.performed

    def _get_available_intervals(self, 
                                 available_requests : list, 
                                 n_intervals : int,
                                 orbitdata : OrbitData
                                 ) -> list:
        intervals = set()
        for req, _ in available_requests:
            req : MeasurementRequest

            t_arrivals = [t*orbitdata.time_step 
                          for t,*_ in orbitdata.gp_access_data.values
                          if req.t_start <= t*orbitdata.time_step <= req.t_end]
            
            t_start = None
            t_prev = None
            for t_arrival in t_arrivals:
                if t_prev is None:
                    t_prev = t_arrival
                    t_start = t_arrival
                    continue
                    
                if abs(t_arrivals[-1] - t_arrival) <= 1e-6: 
                    intervals.add((t_start, t_arrival))
                    continue

                dt = t_arrival - t_prev
                if abs(orbitdata.time_step - dt) <= 1e-6:
                    t_prev = t_arrival
                    continue
                else:
                    intervals.add((t_start, t_prev))
                    t_prev = t_arrival
                    t_start = t_arrival

        # split intervals if they overlap
        intervals_to_remove = set()
        intervals_to_add = set()
        for t_start, t_end in intervals:
            splits = [ [t_start_j, t_end_j, t_start, t_end] 
                        for t_start_j, t_end_j in intervals
                        if (t_start_j < t_start < t_end_j < t_end)
                        or (t_start < t_start_j < t_end < t_end_j)]
            
            for t_start_j, t_end_j, t_start, t_end in splits:
                intervals_to_remove.add((t_start,t_end))
                intervals_to_remove.add((t_start_j,t_end_j))
                
                split = [t_start_j, t_end_j, t_start, t_end]
                split.sort()

                intervals_to_add.add((split[0], split[1]))
                intervals_to_add.add((split[1], split[2]))
                intervals_to_add.add((split[2], split[3]))
        
        for interval in intervals_to_remove:
            intervals.remove(interval)

        for interval in intervals_to_add:
            intervals.add(interval)

        # remove overlaps
        intervals_to_remove = []
        for t_start, t_end in intervals:
            containers = [(t_start_j, t_end_j) 
                        for t_start_j, t_end_j in intervals
                        if (t_start_j < t_start and t_end <= t_end_j)
                        or (t_start_j <= t_start and t_end < t_end_j)]
            if containers:
                intervals_to_remove.append((t_start,t_end))
        
        for interval in intervals_to_remove:
            intervals.remove(interval)
        
        # sort intervals
        intervals = list(intervals)
        intervals.sort(key= lambda a: a[0])

        # return intervals 
        return intervals[:n_intervals]

    @abstractmethod
    def calc_path_bid(
                        self, 
                        state : SimulationAgentState, 
                        specs : object,
                        original_results : dict,
                        original_path : list, 
                        req : MeasurementRequest, 
                        subtask_index : int
                    ) -> tuple:
        """finds the best placement of a task at a given path. returns None if this is not possible"""
        pass

    @abstractmethod
    def calc_imaging_time(self, state : SimulationAgentState, specs : object, path : list, req : MeasurementRequest, main_measurement : str, orbitdata : OrbitData) -> float:
        """
        Computes the ideal" time when a task in the path would be performed
        ### Returns
            - t_img (`float`): earliest available imaging time
        """
        pass  

    """
    --------------------
    LOGGING AND TEARDOWN
    --------------------
    """
    @abstractmethod
    def log_results(self, dsc : str, state : SimulationAgentState, results : dict, level=logging.DEBUG) -> None:
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
        out = f'\n{dsc} = ['
        for req, subtask_index in sequence:
            req : MeasurementRequest
            subtask_index : int
            split_id = req.id.split('-')
            
            if sequence.index((req, subtask_index)) > 0:
                out += ', '
            out += f'({split_id[0]}, {subtask_index})'
        out += ']\n'
        print(out) 