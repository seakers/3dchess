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

from chess3d.agents.planning.planners.rewards import RewardGrid
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
                 max_bundle_size : int = 1,
                 replan_threshold : int = 1,
                 planning_horizon : float = np.Inf,
                 logger: logging.Logger = None
                 ) -> None:
        super().__init__(logger=logger)

        # initialize variables
        self.bundle = []
        # self.path = []
        self.results = {}
        self.bids_to_rebroadcasts = []
        self.completed_measurements = set()
        self.other_plans = {}
        self.recently_completed_observations = set()
        self.my_completed_observations = set()
        self.planner_changes = []
        self.ignored_requests = set()
        self.agent_orbitdata : OrbitData = None
        self.preplan : Preplan = None
        self.plan : Plan = None

        # set paremeters
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

        # check if new preplan was received
        if state.t == current_plan.t and isinstance(current_plan, Preplan): 
            # save latest preplan
            self.preplan : Preplan = current_plan.copy() 
            self.plan = current_plan.copy()
            
            # reset results
            for req, main_measurement, bid in self.bundle:
                # reset bid
                bid : Bid; bid._reset(state.t)

                # update results
                req : MeasurementRequest
                self.results[req.id][main_measurement] = bid

            # reset bundle 
            self.bundle = []

        # update preplan
        self.preplan.update_action_completion(completed_actions, aborted_actions, pending_actions, state.t)

        # compile completed measurements
        my_completed_observations, completed_measurements = self._compile_completed_observations(completed_actions, 
                                                                        misc_messages)
        self.completed_measurements.update(completed_measurements)
        self.recently_completed_observations = completed_measurements
        self.my_completed_observations = my_completed_observations

        # remove redundant measurement actions from preplan
        for completed_observation in my_completed_observations:
            completed_observation : ObservationAction

            # get matching preplanned observations
            preplanned_observations = [ preplanned_action 
                                        for preplanned_action in self.preplan 
                                        if isinstance(preplanned_action, ObservationAction)
                                        and abs(completed_observation.target[0] - preplanned_action.target[0]) <= 1e-3
                                        and abs(completed_observation.target[1] - preplanned_action.target[1]) <= 1e-3
                                        and abs(completed_observation.target[2] - preplanned_action.target[2]) <= 1e-3]
            
            # remove from plan
            for preplanned_observation in preplanned_observations:
                preplanned_observation : ObservationAction
                self.preplan.actions.remove(preplanned_observation)

        # check if any new measurement requests have been received
        self.incoming_bids : list[Bid] = self.compile_new_measurement_request_bids(state)

        # update incoming bids
        self.incoming_bids.extend([ Bid.from_dict(msg.bid) 
                                    for msg in misc_messages 
                                    if isinstance(msg, MeasurementBidMessage)])

        # check if any request refers to an action in the pre-plan
        for req in incoming_reqs:
            # find all observations that match a given request
            matching_planned_observations : list[ObservationAction] \
                = [action for action in self.preplan
                   if isinstance(action, ObservationAction)
                   and abs(action.target[0] - req.target[0]) <= 1e-3
                   and abs(action.target[1] - req.target[1]) <= 1e-3
                   and abs(action.target[2] - req.target[2]) <= 1e-3
                   ]
            
            # remove from pre-plan
            for observation in matching_planned_observations: 
                self.preplan.actions.remove(observation)

        if incoming_reqs:
            x = 1
        elif self.incoming_bids:
            x = 1
        
    def _compile_completed_observations(self, 
                                        completed_actions : list, 
                                        misc_messages : list) -> set:
        """ Reads incoming precepts and compiles all measurement actions performed by the parent agent or other agents """
        # checks measurements performed by the parent agent
        my_completed_observations = {action for action in completed_actions
                                  if isinstance(action, ObservationAction)
                                  and action not in self.completed_measurements}
        
        # checks measuremetns performed by other agents
        completed_observations = {action_from_dict(**msg.observation_action) 
                                    for msg in misc_messages
                                    if isinstance(msg, ObservationPerformedMessage)}
        completed_observations.update(my_completed_observations)

        return my_completed_observations, completed_observations
    
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
        self.log_results('PRE-CONSENSUS PHASE', state, self.results)
        print(f'length of bundle: {len(self.bundle)}\nbids to rebroadcast: {len(self.bids_to_rebroadcasts)}')
        print(f'bundle:')
        for req, subtask_index, bid in self.bundle:
            req : MeasurementRequest
            bid : Bid
            id_short = req.id.split('-')[0]
            print(f'\t{id_short}, {subtask_index}, {np.round(bid.t_img,3)}, {np.round(bid.bid)}')
        print('')
        # ---------------------------------

        self.results, self.bundle, \
            _, bids_to_rebroadcasts = self.consensus_phase(state,
                                                            self.results,
                                                            self.bundle, 
                                                            self.incoming_bids,
                                                            self.recently_completed_observations)
        self.bids_to_rebroadcasts.extend(bids_to_rebroadcasts)
        
        path = self.path_from_bundle(self.bundle)
        assert self.is_task_path_valid(state, specs, path, orbitdata)
        
        # ---------------------------------
        # DEBUGGING OUTPUTS 
        self.log_results('CONSENSUS PHASE', state, self.results)
        print(f'length of bundle: {len(self.bundle)}\nbids to rebroadcast: {len(self.bids_to_rebroadcasts)}')
        print(f'bundle:')
        for req, subtask_index, bid in self.bundle:
            req : MeasurementRequest
            bid : Bid
            id_short = req.id.split('-')[0]
            print(f'\t{id_short}, {subtask_index}, {np.round(bid.t_img,3)}, {np.round(bid.bid)}')
        print('')
        # ---------------------------------

        # check if relevant changes were made to the results
        if self.bids_to_rebroadcasts:
            return True
        
        if (len(self.bundle) < self.max_bundle_size # there is room in the bundle
            and self.my_completed_observations      # and I just performed an observation  
            ):
        # if len(self.bundle) < self.max_bundle_size:
            available_reqs : list = self._get_available_requests(state, specs, self.results, self.bundle, orbitdata)
            
            if available_reqs: 
                x = 1
            
            # check if there are requests available
            return len(available_reqs) >= self.replan_threshold
        
        return False
    
    def path_from_bundle(self, bundle : list) -> list:
        """ Extracts the sequence of observations being scheduled by a given bundle """
        path = [(req, instrument_name, bid.t_img, bid.th_img, bid.bid)
                for req, instrument_name, bid in bundle
                if isinstance(bid, Bid)]
        
        path.sort(key=lambda a : a[2])

        # TODO: add preplan path?

        return path

    @runtime_tracker
    def generate_plan(  self, 
                        state : SimulationAgentState,
                        specs : object,
                        reward_grid : RewardGrid,
                        current_plan : Plan,
                        clock_config : ClockConfig,
                        orbitdata : dict = None
                    ) -> list:
        

        # -------------------------------
        # DEBUG PRINTOUTS
        # self.log_results('PRE-PLANNING PHASE', state, self.results)
        # -------------------------------

        # perform bidding phase
        self.results, self.bundle, self.planner_changes = \
            self.planning_phase(state, specs, self.results, self.bundle, reward_grid, orbitdata)
        
        # -------------------------------
        # DEBUG PRINTOUTS
        # self.log_results('PLANNING PHASE', state, self.results)
        # print(f'bundle:')
        # for req, subtask_index, bid in self.bundle:
        #     req : MeasurementRequest
        #     bid : Bid
        #     print(f'\t{req.id.split('-')[0]}, {subtask_index}, {np.round(bid.t_img,3)}, {np.round(bid.bid)}')
        # print('')
        # -------------------------------

        # schedule observations from bids
        observations : list = self._schedule_observations(state, specs, self.bundle, orbitdata)

        # generate maneuver and travel actions from observations
        maneuvers : list = self._schedule_maneuvers(state, specs, observations, clock_config, orbitdata)

        # schedule broadcasts
        broadcasts : list = self._schedule_broadcasts(state, current_plan, orbitdata)       

        # generate wait actions 
        waits : list = self._schedule_waits(state)
        
        # compile and generate plan
        self.plan = Replan(observations, broadcasts, maneuvers, waits, t=state.t, t_next=self.preplan.t_next)

        return self.plan.copy()
    
    @runtime_tracker
    def _schedule_observations(self, 
                               state : SimulationAgentState, 
                               specs : object, 
                               bundle : list, 
                               orbitdata : OrbitData) -> list:
        """ compiles and merges lists of measurement actions to be performed by the agent """
        try:
            # generate plan from path        
            path : list = self.path_from_bundle(bundle)
            
            # merge with preplanned observations
            observations : list[ObservationAction] = self.merge_plans(state, specs, path)
            
            # return observations
            return observations
        
        finally:
            # ensure path given was valid
            assert self.is_task_path_valid(state, specs, path, orbitdata)

            # ensure resulting plan is valid
            assert self.is_observation_path_valid(state, specs, observations)

    def merge_plans(self, state : SimulationAgentState, specs : object, path : list) -> list:
        # generate proposed observation actions
        proposed_observations = [ObservationAction(main_measurement, req.target, th_img, t_img)
                                for req, main_measurement, t_img, th_img, _ in path
                                if isinstance(req,MeasurementRequest)]
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
        
        # check if compiled observations path is valid
        assert self.is_observation_path_valid(state, specs, observations)

        # check if there are still proposed observations to be added to the path
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

        # check if compiled observations path is valid
        assert self.is_observation_path_valid(state, specs, observations)

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

        # check if compiled observations path is valid
        assert self.is_observation_path_valid(state, specs, observations)

        # return compiled observations path

        planned_observations = [action for action in self.preplan
                                if isinstance(action, ObservationAction)]
        planned_observations.sort(key=lambda a : a.t_start)

        if len(planned_observations) > len(observations):
            x = 1

        return observations
    
    @runtime_tracker
    def _schedule_broadcasts(self, 
                             state: SimulationAgentState, 
                             current_plan: Plan, 
                             orbitdata: dict
                             ) -> list:
        # compile IDs of requests to be broadcasted
        req_ids = [req.id for req in self.pending_reqs_to_broadcast]
        
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

        # find best path for broadcasts
        relay_path, t_start = self._create_broadcast_path(state, orbitdata, state.t)       

        # schedule bid re-broadcast and planner changes
        bids_out = [MeasurementBidMessage(state.agent_name, state.agent_name, bid_to_rebroadcast.to_dict(), path=relay_path) 
                    for req_id in bids 
                    for _, bid_to_rebroadcast in bids[req_id].items() 
                    if bid_to_rebroadcast is not None
                    and t_start >=0]
        broadcasts.extend([BroadcastMessageAction(msg.to_dict(), t_start) for msg in bids_out])

        # reset broadcast list
        self.bids_to_rebroadcasts = []

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
                                bids_received : list,
                                completed_measurements : list,
                                level : int = logging.DEBUG
                            ) -> None:
        """
        Evaluates incoming bids and updates current results and bundle
        """
        
        # check if tasks were performed
        results, bundle, \
            done_changes, done_rebroadcasts = self.check_request_completion(state, results, bundle, completed_measurements, level)

        # check if tasks expired
        results, bundle, \
            exp_changes, exp_rebroadcasts = self.check_request_end_time(state, results, bundle, level)

        # compare bids with incoming messages
        results, bundle, \
            comp_changes, comp_rebroadcasts = self.compare_bids(state, results, bundle, bids_received, level)

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

        return results, bundle, changes, rebroadcasts
    
    @runtime_tracker
    def check_request_completion(self, 
                                 state : SimulationAgentState,
                                 results : dict, 
                                 bundle : list, 
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
            completed_req : MeasurementRequest = self._get_completed_request(results, action)

            if (completed_req is None 
                or action.instrument_name not in completed_req.observation_types):
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
        
        return results, bundle, changes, rebroadcasts
    
    def _get_completed_request(self,
                               results : dict, 
                               action : ObservationAction
                               ) -> MeasurementRequest:
        reqs : list[MeasurementRequest] = list({req 
                                                for req in self.known_reqs
                                                if isinstance(req, MeasurementRequest)
                                                and req.id in results
                                                and abs(req.target[0] - action.target[0]) <= 1e-3
                                                and abs(req.target[1] - action.target[1]) <= 1e-3
                                                and abs(req.target[2] - action.target[2]) <= 1e-3
                                                })
        reqs.sort(key=lambda a : a.t_start)
        return reqs.pop() if reqs else None

    def _get_matching_request(self, id : list) -> MeasurementRequest:
        reqs = {req for req in self.known_reqs if req.id == id}
        if not reqs:
            x = 1
        return reqs.pop() if reqs else None

    @runtime_tracker
    def check_request_end_time(self, 
                               state : SimulationAgentState,
                               results : dict, 
                               bundle : list, 
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

                # reset bid results
                current_bid : Bid; measurement_req : MeasurementRequest
                resetted_bid : Bid = current_bid.update(None, BidComparisonResults.RESET, state.t)
                results[measurement_req.id][instrument_name] = resetted_bid

                rebroadcasts.append(resetted_bid)
                changes.append(resetted_bid)

        return results, bundle, changes, rebroadcasts

    @runtime_tracker
    def compare_bids(
                    self, 
                    state : SimulationAgentState, 
                    results : dict, 
                    bundle : list, 
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

            if req is None:
                x= 1

            if is_new_req:
                # create a new blank bid and save it to results
                bids : list[Bid] = self._generate_bids_from_request(req, state)
                results[their_bid.req_id] = {bid.main_measurement : bid for bid in bids}

                # check who generated the request
                if(their_bid.bidder == state.agent_name   # bid belongs to me
                   and their_bid.winner == their_bid.NONE # has not been bid on
                   ):
                    # request was generated by me; add to broadcasts 
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

                # remove all subsequent bids
                for bundle_index in range(outbid_index, len(bundle)):
                    # remove from bundle
                    measurement_req, subtask_index, current_bid = bundle.pop(outbid_index)

                    # reset bid results
                    current_bid : Bid; measurement_req : MeasurementRequest
                    if bundle_index > outbid_index:
                        reset_bid : Bid = current_bid.update(None, BidComparisonResults.RESET, state.t)
                        results[measurement_req.id][subtask_index] = reset_bid

                        rebroadcasts.append(reset_bid)
                        changes.append(reset_bid)

        return results, bundle, changes, rebroadcasts

    """
    -----------------------
        PLANNING PHASE
    -----------------------
    """
    @runtime_tracker
    def planning_phase( self, 
                        state : SimulationAgentState, 
                        specs : object,
                        results : dict, 
                        bundle : list,
                        reward_grid : RewardGrid,
                        orbitdata : OrbitData
                    ) -> tuple:
        """
        Creates a modified plan from all known requests and current plan
        """
        try:
            # initialzie changes and resetted requirement lists
            changes = []
            reset_reqs = []
            
            # check if bundle is full
            if len(self.bundle) >= self.max_bundle_size:
                # no room left in bundle; return original results
                return results, bundle, changes 
            
            # -------------------------------------
            # TEMPORARY FIX: resets bundle and always replans from scratch
            if len(bundle) > 0:
                # reset results
                for req, main_measurement, bid in bundle:
                    # reset bid
                    bid : Bid; bid._reset(state.t)

                    # update results
                    results[req.id][main_measurement] = bid

                    # add to list of resetted requests
                    reset_reqs.append((req,main_measurement))

                # reset bundle 
                bundle = []
            # -------------------------------------
            
            # get requests that can be bid on by this agent
            available_reqs : list[tuple] = self._get_available_requests(state, specs, results, bundle, orbitdata)

            if not available_reqs: 
                # no tasks available to be bid on by this agent; return original results
                return results, bundle, changes 

            if not bundle: # bundle is empty; create bundle from scratch
                # generate path 
                max_path = self._generate_path(state, specs, results, available_reqs, reward_grid, orbitdata)

                # update bundle and results 
                for req, main_measurement, t_img, th_img, u_exp in max_path:
                    req : MeasurementRequest

                    # update bid
                    old_bid : Bid = results[req.id][main_measurement]
                    new_bid : Bid = old_bid.copy()
                    new_bid.set(u_exp, t_img, th_img, state.t)

                    # place in bundle
                    bundle.append((req, main_measurement, new_bid))

                    # if changed, update results
                    if old_bid != new_bid:
                        changes.append(new_bid.copy())
                        results[req.id][main_measurement] = new_bid

                # announce that tasks were reset and were not re-added to the plan
                path_reqs = {(req, main_measurement) 
                            for req,main_measurement,*_ in max_path}
                reset_bids : list[Bid] = [results[req.id][main_measurement]
                                        for req,main_measurement in reset_reqs
                                        if (req,main_measurement) not in path_reqs]
                
                changes.extend(reset_bids)

            else: # add tasks to the bundle
                raise NotImplementedError('Repairing bundle not yet implemented')

            # return results
            return results, bundle, changes 
        
        finally:
            # ensure that bundle is within the allowed size
            assert len(bundle) <= self.max_bundle_size
                    
            # construct observation sequence from bundle
            path = self.path_from_bundle(bundle)

            # check if path is valid
            assert self.is_task_path_valid(state, specs, path, orbitdata)
        
    def _generate_path(self, 
                       state :SimulationAgentState, 
                       specs : object, 
                       results : dict, 
                       available_reqs : list, 
                       reward_grid : RewardGrid,
                       orbitdata : OrbitData) -> list:
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

            # create potential bids for observations in the path
            path_bids : list[Bid] = [Bid(req.id, 
                                         main_measurement, 
                                         state.agent_name, 
                                         u_exp, 
                                         t_img=t_img, 
                                         th_img=th_img)
                                     for req,main_measurement,t_img,th_img,u_exp in path_i
                                     if isinstance(req,MeasurementRequest)]
                
            # check if it out-performs the current best path
            if any([results[bid.req_id][bid.main_measurement] >= bid
                    for bid in path_bids]):
                # at least one proposed bid is outbid by existing bid; ignore path
                continue
            
            # calculate path utility 
            path_utility : float = self.calc_path_utility(state, specs, path_i, reward_grid)

            # check if it outbids competitors
            if (path_utility > max_path_utility                             # path utility is increased
                and all([results[bid.req_id][bid.main_measurement] < bid    # all new bids outbid competitors
                         for bid in path_bids])
                ):
                # outbids competitors; set new max utility path
                max_path = [path_element for path_element in path_i]
                max_path_utility = path_utility

            # check if there is room to be added to the bundle
            if len(path_i) >= self.max_bundle_size:
                continue

            # add available requests to the path and place it in the queue
            path_reqs = [(req, main_measurement) 
                         for req,main_measurement,*_ in path_i]
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
                
                # estimate observation time and look angle
                t_img,th_img = self.calc_imaging_time(state, specs, path_j, req, main_measurement, orbitdata)
                
                # check if imaging time was found
                if t_img < 0.0: continue # skip

                # calculate performance
                observation = ObservationAction(main_measurement, req.target, th_img, t_img)
                u_exp = reward_grid.estimate_reward(observation)

                # update values
                path_j[-1] = (req, main_measurement, t_img, th_img, u_exp)

                # only add to queue if the path can be performed
                if self.is_task_path_valid(state, specs, path_j, orbitdata): queue.put(path_j)
        
        return max_path
    
    def calc_path_utility(self, 
                          state : SimulationAgentState, 
                          specs : object, 
                          path : list, 
                          reward_grid : RewardGrid
                          ) -> float:
        # merge current path with 
        observations : list[ObservationAction] = self.merge_plans(state, specs, path)
        return sum([reward_grid.estimate_reward(observation) for observation in observations])

    @runtime_tracker
    def _get_available_requests(self, 
                                state : SimulationAgentState, 
                                specs : object,
                                results : dict, 
                                bundle : list,
                                orbitdata : OrbitData
                                ) -> list:
        """ Returns a list of known requests that can be performed within the current planning horizon """
        
        # find requests that can be bid on
        biddable_reqs = []        
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
                
                # check if the task is already in the bundle
                if (req, main_measurement, bid) in bundle:
                    continue

                # check if the task has already been performed
                if self.__request_has_been_performed(results, req, main_measurement, state.t):
                    continue

                # check if the task can be placed in the bundle:
                if not self._can_add_to_bundle(bundle, req, main_measurement):
                    continue

                biddable_reqs.append((req, main_measurement))  

        # sort biddable requests by earliest access times
        biddable_reqs_accesses = []
        for req,main_measurement in biddable_reqs:
            # find access times for a given request
            req_accesses = [
                (t_img*orbitdata.time_step,req.id,req,main_measurement)
                for t_img,_,_,lat,lon,*_,instrument,_ in orbitdata.gp_access_data.values
                if req.t_start <= t_img*orbitdata.time_step <= req.t_end
                and instrument == main_measurement
                and abs(lat - req.target[0]) <= 1e-3
                and abs(lon - req.target[1]) <= 1e-3
            ]

            # sort in ascending order 
            req_accesses.sort()

            # save only the earliest access time to request target
            biddable_reqs_accesses.append(req_accesses.pop(0))
        
        # sort requests by ascending access time 
        biddable_reqs_accesses.sort()

        # return sorted list of access times
        return [(req,main_measurement) for *_,req,main_measurement in biddable_reqs_accesses]
        
    def _can_add_to_bundle(self, bundle : list, req : MeasurementRequest, main_measurement : str) -> bool:
        # TODO
        return True

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