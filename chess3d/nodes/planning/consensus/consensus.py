import logging
import time
from typing import Any, Callable
from numpy import Inf
import pandas as pd

from dmas.utils import runtime_tracker
from dmas.clocks import *
from traitlets import Callable
from chess3d.nodes.states import SimulationAgentState

from nodes.planning.plan import Plan, Preplan, Replan
from nodes.planning.consensus.bids import Bid, BidComparisonResults, RebroadcastComparisonResults, UnconstrainedBid
from nodes.planning.replanners import AbstractReplanner
from nodes.science.utility import *
from nodes.orbitdata import OrbitData
from nodes.science.reqs import *
from nodes.states import *
from messages import *

class AbstractConsensusReplanner(AbstractReplanner):
    def __init__(self, 
                 utility_func: Callable, 
                 max_bundle_size : int = 1,
                 dt_converge : float = 0.0,
                 logger: logging.Logger = None
                 ) -> None:
        super().__init__(utility_func, logger=logger)

        # initialize variables
        self.bundle = []
        self.path = []
        self.results = {}
        self.bids_to_rebroadcasts = []
        self.completed_measurements = []
        self.preplan : Preplan = Preplan(t=-1.0)

        # set paremeters
        self.max_bundle_size = max_bundle_size
        self.dt_converge = dt_converge

    def update_precepts(self, 
                        state: SimulationAgentState, 
                        current_plan: Plan, 
                        completed_actions: list, 
                        aborted_actions: list, 
                        pending_actions: list, 
                        incoming_reqs: list, 
                        generated_reqs: list, 
                        relay_messages: list, 
                        misc_messages: list, 
                        orbitdata: dict = None
                        ) -> None:
       
        # update other precepts
        super().update_precepts(state, 
                                current_plan,    
                                completed_actions, 
                                aborted_actions, 
                                pending_actions, 
                                incoming_reqs, 
                                generated_reqs, 
                                relay_messages, 
                                misc_messages, 
                                orbitdata)
        
        # update preplan
        if state.t == current_plan.t and isinstance(current_plan, Preplan): 
            self.preplan = current_plan.copy() 
        
        # compile received bids
        bids_received = self._compile_bids( state,
                                            incoming_reqs, 
                                            generated_reqs, 
                                            misc_messages)

        # compile completed measurements
        completed_measurements = self._compile_completed_measurements(completed_actions, misc_messages)

        # perform consesus phase
        self.results, self.bundle, \
            self.path, _, self.bids_to_rebroadcasts = self.consensus_phase( state,
                                                                            self.results,
                                                                            self.bundle, 
                                                                            self.path, 
                                                                            bids_received,
                                                                            completed_measurements)
        self.log_results('Updated precepts', self.results, logging.WARNING)

    @runtime_tracker
    def _update_access_times(  self,
                                state : SimulationAgentState,
                                t_plan : float,
                                agent_orbitdata : OrbitData) -> None:
        """
        Calculates and saves the access times of all known requests
        """
        if state.t == self.preplan.t or len(self.known_reqs) > len(self.results):
            # recalculate access times for all known requests            
            for req in self.known_reqs:
                req : MeasurementRequest

                if state.t == self.preplan.t or req.id not in self.access_times:
                    self.access_times[req.id] = {instrument : [] for instrument in req.measurements}

                # check access for each required measurement
                for instrument in self.access_times[req.id]:
                    if instrument not in state.payload:
                        # agent cannot perform this request TODO add KG support
                        continue

                    if (req, instrument) in self.completed_requests:
                        # agent has already performed this request
                        continue

                    if len(self.access_times[req.id][instrument]) > 0:
                        continue

                    t_arrivals : list = self._calc_arrival_times(   state, 
                                                                    req, 
                                                                    instrument,
                                                                    state.t,
                                                                    agent_orbitdata)
                    
                    self.access_times[req.id][instrument] = t_arrivals

    def _compile_bids(self, 
                      state : SimulationAgentState, 
                      incoming_reqs : list, 
                      generated_reqs : list, 
                      misc_messages : list
                      ) -> list:
        """ Reads incoming messages and requests and checks if bids were received """
        # get bids from misc messages
        bids = [Bid.from_dict(msg.bid) 
                for msg in misc_messages 
                if isinstance(msg, MeasurementBidMessage)]
        
        # check for new requests from incoming requests
        new_reqs = [req for req in incoming_reqs 
                    if req.id not in self.results]
        
        #check for new requests from generated requests
        new_reqs.extend([req for req in generated_reqs 
                         if req.id not in self.results 
                         and req not in new_reqs])
        
        # generate bids for new requests
        for new_req in new_reqs:
            new_req : MeasurementRequest
            req_bids : list = self._generate_bids_from_request(new_req, state)
            bids.extend(req_bids)

        return bids
    
    @abstractmethod
    def _generate_bids_from_request(self, req : MeasurementRequest, state : SimulationAgentState) -> list:
        """ Creages bids from given measurement request """
        pass

    def _compile_completed_measurements(self, 
                                        completed_actions : list, 
                                        misc_messages : list) -> list:
        """ Reads incoming precepts and compiles all measurement actions performed by the parent agent or other agents """
        # checks measurements performed by the parent agent
        completed_measurements = [action for action in completed_actions
                                  if isinstance(action, MeasurementAction)]
        
        # checks measuremetns performed by other agents
        completed_measurements.extend([action_from_dict(msg.measurement_action) 
                                       for msg in misc_messages
                                       if isinstance(msg, MeasurementPerformedMessage)])
        
        assert all([action.status == action.COMPLETED 
                    and isinstance(action, MeasurementAction)
                    for action in completed_measurements])

        return completed_measurements

    def needs_replanning(self, state : SimulationAgentState, plan : Plan) -> bool:   
        if not self.is_converged():
            x = 1
        if len(self.bids_to_rebroadcasts) > 0:
            x = 1

        # replan if relevant changes have been made to the bundle
        return len(self.bids_to_rebroadcasts) > 0 or not self.is_converged()

    @abstractmethod
    def is_converged(self) -> bool:
        """ Checks if consensus has been reached and plans are coverged """
        pass

    def generate_plan(  self, 
                        state : SimulationAgentState,
                        current_plan : Plan,
                        completed_actions : list,
                        aborted_actions : list,
                        pending_actions : list,
                        incoming_reqs : list,
                        generated_reqs : list,
                        relay_messages : list,
                        misc_messages : list,
                        clock_config : ClockConfig,
                        orbitdata : dict = None
                    ) -> list:

        # bidding phase
        self.results, self.bid, self.path, planner_changes = \
            self.planning_phase(state, self.results, self.bundle, self.path)
        
        # check convergence
        if self.is_converged():
            plan : Replan = self._plan_from_path(state, self.path, clock_config, orbitdata)
        else:
            plan : Preplan = self.preplan

        # broadcast changes to plan
        broadcast_bids : list = self._compile_broadcast_bids(planner_changes)       
        for bid in broadcast_bids:
            bid : Bid
            msg = MeasurementBidMessage(state.agent_name, state.agent_name, bid.to_dict())
            broadcast_action = BroadcastMessageAction(msg.to_dict(), state.t)
            plan.add(broadcast_action)
            
        # reset broadcast list
        self.rebroadcasts = []

        # -------------------------------
        # DEBUG PRINTOUTS
        print('Observations in Original Path:')
        original_obs_actions = [action for action in self.preplan if isinstance(action, MeasurementAction)]
        for action in original_obs_actions:
            print(action.measurement_req['id'].split('-')[0], action.subtask_index)
        print(len(original_obs_actions))
            
        print('Observations in Modified Path:')
        obs_actions = [action for action in plan if isinstance(action, MeasurementAction)]
        for action in obs_actions:
            print(action.measurement_req['id'].split('-')[0], action.subtask_index)
        print(len(obs_actions))

        if len(original_obs_actions) != len(obs_actions):
            x = 1
        # -------------------------------

        # output plan
        return plan
        
    def _plan_from_path(self, 
                        state : SimulationAgentState, 
                        path : list,
                        clock_config: ClockConfig, 
                        orbitdata: OrbitData = None
                        ) -> Replan:
        """ creates a new plan to be performed by the agent based on the results of the planning phase """

        # schedule measurements
        measurements : list = self._compile_measurements(path)

        # schedule broadcasts to be perfomed
        broadcasts : list = [action for action in self.preplan.actions if isinstance(action, BroadcastMessageAction)]

        # generate maneuver and travel actions from measurements
        maneuvers : list = self._schedule_maneuvers(state, measurements, broadcasts, clock_config)

        return Replan(measurements, broadcasts, maneuvers, t=state.t, t_next=self.preplan.t_next)
        
    def _compile_measurements(self, path : list) -> list:
        """ compiles and merges lists of measurement actions to be performed by the agent """
        # get list of preplanned measurements
        preplanned_measurements = [action for action in self.plan.actions 
                                   if isinstance(action, MeasurementAction)]
        preplanned_measurements.sort(key=lambda a : a.t_start)

        # get list of new measurements to add to plan
        new_measurements = []
        for req, subtask_index, t_img, u_exp in path:
            req : MeasurementRequest
            subtask_index : int

            instrument_name, _ = req.measurement_groups[subtask_index]
            new_measurements.append(MeasurementAction(req.to_dict(), 
                                                      subtask_index, 
                                                      instrument_name, 
                                                      u_exp, 
                                                      t_img, 
                                                      t_img+req.duration
                                                      )
                                    )    
        new_measurements.sort(key=lambda a : a.t_start)

        # merge measurements 
        measurements = [action for action in preplanned_measurements]
        for new_measurement in new_measurements:
            new_measurement : MeasurementAction
            i = -1
            replace = False
            insert = False
            
            for measurement in measurements:
                measurement : MeasurementAction
                
                replace = (measurement.t_start < new_measurement.t_end <= measurement.t_end
                            or measurement.t_start <= new_measurement.t_start < measurement.t_end)
                insert = new_measurement.t_end <= measurement.t_start
                
                if replace or insert:
                    i = measurements.index(measurement)
                    break

            if replace:
                measurements[i] = new_measurement
            elif insert:
                measurements.insert(i, new_measurement)
            else:
                measurements.append(new_measurement)

        return measurements
    
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

    """
    -----------------------
        CONSENSUS PHASE
    -----------------------
    """
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
        
        # check if tasks expired
        results, bundle, path, \
            exp_changes, exp_rebroadcasts = self.check_request_end_time(state, results, bundle, path, level)

        # compare bids with incoming messages
        results, bundle, path, \
            comp_changes, comp_rebroadcasts = self.compare_bids(state, results, bundle, path, bids_received, level)
        

        # compile changes
        changes = []
        changes.extend(comp_changes)
        changes.extend(exp_changes)
        changes.extend(done_changes)

        # compile rebroadcasts
        rebroadcasts = []
        rebroadcasts.extend(comp_rebroadcasts)
        rebroadcasts.extend(exp_rebroadcasts)
        rebroadcasts.extend(done_rebroadcasts)

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
            action : MeasurementAction

            # set bid as completed           
            completed_req : MeasurementRequest \
                = MeasurementRequest.from_dict(action.measurement_req)
            bid : Bid = results[completed_req.id][action.subtask_index]
            results[completed_req.id][action.subtask_index] \
                  = bid.update(None, BidComparisonResults.COMPLETED, state.t)

            # add to changes list
            changes.append(bid)

        # check for task completion in bundle
        for task_to_remove in [(req, subtask_index, current_bid)
                               for req, subtask_index, current_bid in bundle
                               if results[req.id][subtask_index].performed]:
            ## remove all completed tasks from bundle
            self.bundle.remove(task_to_remove)
        
        # check if any mutually exclusive tasks have been performed
        task_to_remove = None
        for req, subtask_index, current_bid in bundle:
            req : MeasurementRequest
            
            ## check for all known bids related to the relevant measurement request
            for bid_index in range(len(results[req.id])):
                bid : Bid = results[req.id][bid_index]
                if (bid.performed                                               # the other bid was performed
                    and req.dependency_matrix[subtask_index][bid_index] < 0):   # is mutually exclusive with the bid at hand
                    
                    ## a mutually exclusive bid was performed
                    task_to_remove = (req, subtask_index, current_bid)
                    break   

            if task_to_remove is not None:
                break
        
        if task_to_remove is not None:
            ## a mutually exclusive bid was performed; 
            ## remove mutually exclusive task from bundle and all subsequent tasks
            expired_index : int = bundle.index(task_to_remove)
            for _ in range(expired_index, len(bundle)):
                # remove from bundle
                measurement_req, subtask_index, current_bid = bundle.pop(expired_index)

                # remove from path
                path.remove((measurement_req, subtask_index, current_bid))

                # reset bid results
                current_bid : Bid; measurement_req : MeasurementRequest
                reset_bid : Bid = current_bid.update(None, BidComparisonResults.RESET, state.t)
                results[measurement_req.id][subtask_index] = reset_bid

                # add to changes and rebroadcast lists
                changes.append(reset_bid)
                rebroadcasts.append(reset_bid)

        return results, bundle, path, changes, rebroadcasts

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

        # release tasks from bundle if t_end has passed
        task_to_remove = None
        for req, subtask_index, bid in bundle:
            req : MeasurementRequest; bid : Bid
            if req.t_end - req.duration < state.t and not bid.performed:
                task_to_remove = (req, subtask_index, bid)
                break

        # if task has expired, release from bundle and path with all subsequent tasks
        if task_to_remove is not None:
            expired_index = bundle.index(task_to_remove)
            for _ in range(expired_index, len(bundle)):
                # remove from bundle
                measurement_req, subtask_index, current_bid = bundle.pop(expired_index)

                # remove from path
                path.remove((measurement_req, subtask_index, current_bid))

                # reset bid results
                current_bid : Bid; measurement_req : MeasurementRequest
                reset_bid : Bid = current_bid.update(None, BidComparisonResults.RESET, t)
                results[measurement_req.id][subtask_index] = reset_bid

                rebroadcasts.append(reset_bid)
                changes.append(reset_bid)

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

            # check bids are for new requests
            is_new_req : bool = their_bid.req_id not in results

            req : MeasurementRequest = MeasurementRequest.from_dict(their_bid.req)
            if is_new_req:
                # was not aware of this request; add to results as a blank bid
                results[req.id] = self._generate_bids_from_request(req, state)

                # add to changes broadcast
                my_bid : Bid = results[req.id][0]
                rebroadcasts.append(my_bid)
                                    
            # compare bids
            my_bid : Bid = results[their_bid.req_id][their_bid.subtask_index]
            # self.log(f'comparing bids...\nmine:  {str(my_bid)}\ntheirs: {str(their_bid)}', level=logging.DEBUG) #DEBUG PRINTOUT

            comp_result, rebroadcast_result = my_bid.compare(their_bid)
            updated_bid : Bid = my_bid.update(their_bid, comp_result, state.t)
            bid_changed = my_bid != updated_bid

            # update results with modified bid
            # self.log(f'\nupdated: {my_bid}\n', level=logging.DEBUG) #DEBUG PRINTOUT
            results[their_bid.req_id][their_bid.subtask_index] = updated_bid
                
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
            if (
                (req, my_bid.subtask_index, my_bid) in bundle 
                and updated_bid.winner != state.agent_name
                ):

                outbid_index = bundle.index((req, my_bid.subtask_index, my_bid))

                # remove all subsequent bids
                for bundle_index in range(outbid_index, len(bundle)):
                    # remove from bundle
                    measurement_req, subtask_index, current_bid = bundle.pop(outbid_index)

                    # remove from path
                    path.remove((measurement_req, subtask_index, current_bid))

                    # reset bid results
                    current_bid : Bid; measurement_req : MeasurementRequest
                    if bundle_index > outbid_index:
                        reset_bid : Bid = current_bid.update(None, BidComparisonResults.RESET, state.t)
                        results[measurement_req.id][subtask_index] = reset_bid

                        rebroadcasts.append(reset_bid)
                        changes.append(reset_bid)
        
        return results, bundle, path, changes, rebroadcasts

    """
    -----------------------
        PLANNING PHASE
    -----------------------
    """
    @runtime_tracker
    def planning_phase( self, 
                        state : SimulationAgentState, 
                        results : dict, 
                        bundle : list,
                        path : list
                    ) -> tuple:
        """
        Creates a modified plan from all known requests and current plan
        """
        # initialzie changes
        changes = []

        # get current measurement path from preplan
        path = []
        for measurement_action in [action for action in self.preplan.actions if isinstance(action, MeasurementAction)]:
            req : MeasurementRequest = MeasurementRequest.from_dict(measurement_action.measurement_req)
            path.append((req, 
                         measurement_action.subtask_index, 
                         measurement_action.t_start, 
                         measurement_action.u_exp))

        # check if bundle is full
        if len(bundle) >= self.max_bundle_size:
            # Bundle is full; cannot modify 
            return results, bundle, path, changes

        # get requests that can be bid on by this agent
        available_reqs : list = self._get_available_requests(state, results, bundle, path)

        # initialize path of maximum utility
        max_path = [path_element for path_element in path]; 
        max_path_bids = {req.id : {} for req, _, _, _ in path if isinstance(req, MeasurementRequest)}
        for req, subtask_index, _, _ in path:
            req : MeasurementRequest
            max_path_bids[req.id][subtask_index] = results[req.id][subtask_index]

        # begin bid process
        max_req = -1
        while ( len(available_reqs) > 0                     # there are available tasks to be bid on
                and len(bundle) < self.max_bundle_size      # there is space in the bundle
                and max_req is not None                     # there is a request that maximizes utility
            ):   
             # find next best task to put in bundle (greedy)
            max_req = None 
            max_subtask = None
            max_bid = None
            for req, subtask_index in available_reqs:
                
                # calculate best bid and path for a given request and subtask
                projected_path, projected_path_utility \
                     = self.calc_path_bid(  state, 
                                            results, 
                                            path, 
                                            req, 
                                            subtask_index)

                # check if path was found
                if projected_path is None:
                    continue

                # compare to maximum task
                if (
                    max_req is None or projected_path_utility > max_path_utility
                    ):
                    # get time and utility gained from said action
                    path_elements = [item 
                                     for item in path 
                                     if item[0] == req and item[1] == subtask_index
                                     ]
                    _, _, t_img, u_exp = path_elements[0]
                    old_bid : Bid = results[req.id][subtask_index]

                    # register maximum utility 
                    max_bid : Bid = old_bid.copy()
                    max_bid.set_bid(u_exp, t_img, state.t)
                    max_path = projected_path
                    max_path_utility = projected_path_utility
                    max_req = req
                    max_subtask = subtask_index
            
            if max_req is not None:
                # max bid found! place task with the best bid in the bundle and the path
                bundle.append((max_req, max_subtask, max_bid))
                path = max_path

                # update results
                for req, subtask_index, new_bid in bundle:
                    req : MeasurementRequest
                    subtask_index : int
                    new_bid : Bid

                    old_bid : Bid = results[req.id][subtask_index]

                    if old_bid != new_bid:
                        changes.append(new_bid.copy())
                        results[req.id][subtask_index] = new_bid

        return results, bundle, path, changes 
    
    def _get_available_requests(self, 
                                state : SimulationAgentState, 
                                results : dict, 
                                bundle : list,
                                path : list
                                ) -> list:
        """ Returns a list of known requests that can be performed within the current planning horizon """
        path_reqs = [(req, subtask_index) for req, subtask_index, _, _ in path]
        available = []        

        for req_id in results:
            for subtask_index in range(len(results[req_id])):
                subtask_bid : Bid = results[req_id][subtask_index]; 
                req = MeasurementRequest.from_dict(subtask_bid.req)

                is_accessible = self.__can_access(state, req, subtask_index)
                is_biddable = self._can_bid(state, results, req, subtask_index)
                already_in_bundle = (req, subtask_index, subtask_bid) in bundle
                already_in_path = (req, subtask_index) in path_reqs
                already_performed = self.__request_has_been_performed(results, req, subtask_index, state.t)
                
                if (is_accessible 
                    or is_biddable 
                    or not already_in_bundle 
                    or not already_performed
                    ) and not already_in_path:
                    x = 1
                if not already_in_path:
                    print(is_accessible, is_biddable, not already_in_bundle, not already_in_path, not already_performed)

                if (is_accessible 
                    and is_biddable 
                    and not already_in_bundle 
                    and not already_in_path
                    and not already_performed
                    ):
                    available.append((req, subtask_bid.subtask_index))       
                
        return available
   
    def __can_access(self, 
                     state : SimulationAgentState, 
                     req : MeasurementRequest, 
                     subtask_index : int
                     ) -> bool:
        """ Checks if an agent can access the location of a measurement request """
        if isinstance(state, SatelliteAgentState):
            main_instrument, _ = req.measurement_groups[subtask_index]
            t_arrivals = [t 
                          for t in self.access_times[req.id][main_instrument] 
                          if req.t_start <= t <= req.t_end
                          ]
            
            if len(t_arrivals) == 0 and len(self.access_times[req.id][main_instrument]) > 0:
                y = 1

            return len(t_arrivals) > 0
        else:
            raise NotImplementedError(f"listing of available requests for agents with state of type {type(state)} not yet supported.")

    @abstractmethod
    def _can_bid(self, 
                state : SimulationAgentState, 
                results : dict,
                req : MeasurementRequest, 
                subtask_index : int
                ) -> bool:
        """ Checks if an agent has the ability to bid on a measurement task """
        pass

    def __check_if_in_bundle(   self, 
                                req : MeasurementRequest, 
                                subtask_index : int, 
                                bid : Bid,
                                bundle : list
                                ) -> bool:
        """ Checks if the task is already in the bundle """
        for req_i, subtask_index_j, bid_j in bundle:
            if req_i.id == req.id and subtask_index == subtask_index_j and bid == bid_j:
                return True
    
        return False

    def __request_has_been_performed(self, 
                                     results : dict, 
                                     req : MeasurementRequest, 
                                     subtask_index : int, 
                                     t : Union[int, float]) -> bool:
        """ Check if subtask at hand has already been performed """
        current_bid : Bid = results[req.id][subtask_index]
        subtask_already_performed = t > current_bid.t_img >= 0 + req.duration and current_bid.winner != Bid.NONE
        if subtask_already_performed or current_bid.performed:
            return True
       
        return False


    @runtime_tracker
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
        winning_path_utility = 0.0

        # check if the subtask to be added is mutually exclusive with something in the bundle
        for req_i, subtask_j, _, _ in original_path:
            req_i : MeasurementRequest; subtask_j : int
            if req_i.id == req.id:
                if req.dependency_matrix[subtask_j][subtask_index] < 0:
                    return winning_path, winning_path_utility

        # find best placement in path
        # self.log_task_sequence('original path', original_path, level=logging.WARNING)

        ## insert strategy
        for i in range(len(original_path)+1):
            # generate possible path
            path = [scheduled_obs for scheduled_obs in original_path]
            
            path.insert(i, (req, subtask_index, -1, -1))
            # self.log_task_sequence('new proposed path', path, level=logging.WARNING)

            # recalculate bids for each task in the path if needed
            for j in range(len(path)):
                req_i, subtask_j, t_img, u = path[j]

                if j < i:
                    # element from previous path is unchanged; maintain current bid
                    path[j] = (req, subtask_index, t_img, u)
                    
                elif j == i:
                    # new request and subtask are being added; recalculate bid

                    # calculate imaging time
                    req_i : MeasurementRequest
                    subtask_j : int
                    t_img = self.calc_imaging_time(state, path, req_i, subtask_j)

                    # calc utility
                    params = {"req" : req_i.to_dict(), "subtask_index" : subtask_j, "t_img" : t_img}
                    utility = self.utility_func(**params) * synergy_factor(**params) if t_img >= 0 else np.NINF

                    # place bid in path
                    path[j] = (req, subtask_index, t_img, utility)

                else:
                    # elements from previous path are being adapted to new path

                    ## calculate imaging time
                    req_i : MeasurementRequest
                    subtask_j : int
                    t_img = self.calc_imaging_time(state, path, req_i, subtask_j)

                    _, _, t_img_prev, _ = original_path[j - 1]

                    if abs(t_img - t_img_prev) <= 1e-3:
                        # path was unchanged; keep the remaining elements of the previous path                    
                        break
                    else:
                        # calc utility
                        params = {"req" : req_i.to_dict(), "subtask_index" : subtask_j, "t_img" : t_img}
                        utility = self.utility_func(**params) * synergy_factor(**params) if t_img >= 0 else np.NINF

                        # place bid in path
                        path[j] = (req, subtask_index, t_img, utility)

            # look for path with the best utility
            path_utility = self.__sum_path_utility(path)
            if path_utility > winning_path_utility:
                winning_path = path
                winning_path_utility = path_utility
        
        ## replacement strategy
        if winning_path is None:
            # TODO add replacement strategy
            pass

        return winning_path, winning_path_utility

    @abstractmethod
    def calc_imaging_time(self, state : SimulationAgentState, path : list, req : MeasurementRequest, subtask_index : int) -> float:
        """
        Computes the ideal" time when a task in the path would be performed
        ### Returns
            - t_img (`float`): earliest available imaging time
        """
        pass 

    def __sum_path_utility(self, path : list) -> float:
        """ Gives the total utility of a proposed observations sequence """
        if -1 in [t_img for _,_,t_img,_ in path]:
            return 0.0
        
        return sum([u for _,_,_,u in path])  


    """
    --------------------
    LOGGING AND TEARDOWN
    --------------------
    """
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
