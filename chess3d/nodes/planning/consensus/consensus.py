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
        self.other_plans = {}
        self.recently_completed_measurments = []
        self.planner_changes = []

        # set paremeters
        self.max_bundle_size = max_bundle_size
        self.dt_converge = dt_converge

    def update_percepts(self, 
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
       
        # update other percepts
        super().update_percepts(state, 
                                current_plan,    
                                completed_actions, 
                                aborted_actions, 
                                pending_actions, 
                                incoming_reqs, 
                                generated_reqs, 
                                relay_messages, 
                                misc_messages, 
                                orbitdata)

        # update latest preplan
        if state.t == current_plan.t and isinstance(current_plan, Preplan): 
            self.preplan = current_plan.copy() 
            self.path = [(MeasurementRequest.from_dict(action.measurement_req), 
                         action.subtask_index, 
                         action.t_start, 
                         action.u_exp) 
                         for action in current_plan
                         if isinstance(action, MeasurementAction)]
            
            #TODO reset bundle and results
            self.bundle = []

        # compile completed measurements
        completed_measurements = self._compile_completed_measurements(completed_actions, 
                                                                        misc_messages)
        self.completed_measurements.extend(completed_measurements)
        self.recently_completed_measurments = completed_measurements


        # update incoming bids
        self.incoming_bids = [  Bid.from_dict(msg.bid) 
                                for msg in misc_messages 
                                if isinstance(msg, MeasurementBidMessage)]
        self_bids =   [Bid.from_dict(msg.bid) 
                        for msg in misc_messages 
                        if isinstance(msg, MeasurementBidMessage)
                        if msg.src == state.agent_name]
        

        # compile incoming plans
        incoming_plans = [(msg.src, 
                           Plan([action_from_dict(**action) for action in msg.plan],t=msg.t_plan))
                           for msg in misc_messages
                           if isinstance(msg, PlanMessage)]
    
        # update internal knowledge base of other agent's 
        for src, plan in incoming_plans:
            plan : Plan
            if src not in self.other_plans:
                # new agent has sent their current plan
                self.other_plans[src] = (plan, state.t)
            
            other_plan, _ = self.other_plans[src]
            other_plan : Plan
            if plan.t > other_plan.t:
                # only update if a newer plan was received
                self.other_plans[src] = (plan, state.t)

    def _compile_completed_measurements(self, 
                                        completed_actions : list, 
                                        misc_messages : list) -> list:
        """ Reads incoming precepts and compiles all measurement actions performed by the parent agent or other agents """
        # checks measurements performed by the parent agent
        completed_measurements = [action for action in completed_actions
                                  if isinstance(action, MeasurementAction)
                                  and action not in self.completed_actions]
        
        # checks measuremetns performed by other agents
        completed_measurements.extend([action_from_dict(**msg.measurement_action) 
                                       for msg in misc_messages
                                       if isinstance(msg, MeasurementPerformedMessage)])

        return completed_measurements

    def needs_planning(self, state : SimulationAgentState, plan : Plan) -> bool:   
    
        # compile any conflicts in newly received plans
        conflicts : list[Bid] = self.compile_planning_conflicts(state, plan)
        self.incoming_bids.extend(conflicts)
                    
        # check if any new measurement requests have been received
        new_req_bids : list[Bid] = self.compile_new_measurement_request_bids(state, plan)
        self.incoming_bids.extend(new_req_bids)
        
        # perform consesus phase
        # self.log_results('PRE-CONSENSUS PHASE', state, self.results)
        # print(f'length of path: {len(self.path)}\n')
        self.results, self.bundle, \
            self.path, _, self.bids_to_rebroadcasts = self.consensus_phase( state,
                                                                            self.results,
                                                                            self.bundle, 
                                                                            self.path, 
                                                                            self.incoming_bids,
                                                                            self.recently_completed_measurments)
        
        # self.log_results('CONSENSUS PHASE', state, self.results)
        # print(f'length of path: {len(self.path)}\n')

        if (state.t - self.preplan.t) < 1e-3 and not conflicts:
            # plan was just developed by preplanner; no need to replan
            return False
        
        if self.bids_to_rebroadcasts:
            # relevant changes were made to the results database; replan
            return True

        # no need to replan
        return False

    def compile_planning_conflicts(self, state : SimulationAgentState, plan : Plan) -> list:
        """checks for ay conflicts between the current plan and any incoming plans"""

        my_measurements = [action for action in plan if isinstance(action, MeasurementAction)]

        conflicts : list[Bid] = []
        other_plans = [(other_agent, other_plan) for other_agent, other_plan in self.other_plans.items()]
        

        for other_agent, other_plan in other_plans:
            their_plan, t_received = other_plan
            their_plan : Plan

            # if abs(t_received - state.t) >= 1e-3:
            #     # only checks for conflicts when when new plan was just received
            #     continue
                
            # check if another agent is planning on doing the same task as I am
            other_measurements = [action for action in their_plan 
                                    if isinstance(action, MeasurementAction)]
            for other_measurement in other_measurements:
                other_measurement : MeasurementAction
                conflicting_measurements = [my_measurement for my_measurement in my_measurements
                                            if isinstance(my_measurement,MeasurementAction)
                                            and my_measurement.measurement_req['id'] == other_measurement.measurement_req['id']
                                            and my_measurement.subtask_index == other_measurement.subtask_index]
                
                # there are scheduling conflicts; two agents are planning on performing the same task
                for conflict in conflicting_measurements:
                    conflict : MeasurementAction
                    # add measurement to list of conflicts
                    req = MeasurementRequest.from_dict(conflict.measurement_req)
                    bids : list[Bid] = self._generate_bids_from_request(req, state)

                    conflicting_bid : Bid = bids[other_measurement.subtask_index]
                    conflicting_bid.winner = other_agent
                    conflicting_bid.bidder = other_agent
                    conflicting_bid.set(other_measurement.u_exp, other_measurement.t_start, their_plan.t)
                                        
                    conflicts.append(conflicting_bid)
            
            # 
            self.other_plans.pop(other_agent)
        
        return conflicts
    
    @abstractmethod
    def _generate_bids_from_request(self, req : MeasurementRequest, state : SimulationAgentState) -> list:
        """ Creages bids from given measurement request """

    def compile_new_measurement_request_bids(self, state : SimulationAgentState, plan : Plan) -> list:
        """ Checks for all requests that havent been considered in current bidding process """
        # compile list of tasks currently planned to be performed 
        planned_reqs = [(action.measurement_req, action.subtask_index)
                        for action in plan 
                        if isinstance(action,MeasurementAction)]
        
        # compile list of new bids
        new_request_bids : list[Bid] = []
        for req in [req for req in self.known_reqs if req.id not in self.results]:
            req : MeasurementRequest

            # create new bids for measurement request
            bids : list[Bid] = self._generate_bids_from_request(req, state)

            # update bids with currently planned values
            for bid in bids:
                if (req.to_dict(), bid.subtask_index) in planned_reqs:
                    action = [action for action in plan 
                              if isinstance(action, MeasurementAction)
                              and action.measurement_req == req.to_dict()
                              and action.subtask_index == bid.subtask_index]
                    action : MeasurementAction = action[0]
                    bid.set(action.u_exp, action.t_start, state.t)

            # add to list of new bids
            new_request_bids.extend(bids)

        return new_request_bids

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
        
        # -------------------------------
        # DEBUG PRINTOUTS
        out = f'{state.agent_name} - Observations in Original Path:\n'
        original_obs_actions = [action for action in current_plan if isinstance(action, MeasurementAction)]
        # out += 'ID\t\tsubtask index\n'
        # for action in original_obs_actions:
        #     out += f"{action.measurement_req['id'].split('-')[0]}\t{action.subtask_index}\n"
        # out += f'n_observations: {len(original_obs_actions)}\n'
        # print(out)
        # -------------------------------

        # bidding phase
        self.results, self.bid, self.path, self.planner_changes = \
            self.planning_phase(state, self.results, self.bundle, self.path)
        
        # check convergence
        plan : Replan = self._plan_from_path(state, self.results, self.path, clock_config, orbitdata)     
            
        # reset broadcast list
        self.rebroadcasts = []

        # -------------------------------
        # DEBUG PRINTOUTS
        out = f'{state.agent_name} - Observations in Modified Path:\n'
        obs_actions = [action for action in plan if isinstance(action, MeasurementAction)]
        # out += 'ID\t\tsubtask index\n'
        # for action in obs_actions:
        #     out += f"{action.measurement_req['id'].split('-')[0]}\t{action.subtask_index}\n"
        out += f'n_observations: {len(obs_actions)}\n'
        # print(out)

        if len(original_obs_actions) != len(obs_actions):
            x = 1
        # -------------------------------

        # output plan
        return plan
        
    def _plan_from_path(self, 
                        state : SimulationAgentState, 
                        results : dict,
                        path : list,
                        clock_config: ClockConfig, 
                        orbitdata: OrbitData = None
                        ) -> Replan:
        """ creates a new plan to be performed by the agent based on the results of the planning phase """

        # schedule measurements
        measurements : list = self._schdule_measurements(results, path)

        # schedule bruadcasts
        broadcasts : list = self._schedule_broadcasts(state, measurements, orbitdata)

        # generate maneuver and travel actions from measurements
        maneuvers : list = self._schedule_maneuvers(state, measurements, broadcasts, clock_config)

        return Replan(measurements, broadcasts, maneuvers, t=state.t, t_next=self.preplan.t_next)
        
    def _schdule_measurements(self, results : dict, path : list) -> list:
        """ compiles and merges lists of measurement actions to be performed by the agent """

        measurements = []
        for req, subtask_index, t_img, u_exp in path:
            req : MeasurementRequest
            subtask_index : int

            bid : Bid = results[req.id][subtask_index]
            instrument_name = bid.main_measurement
            measurements.append(MeasurementAction(req.to_dict(), 
                                                      subtask_index, 
                                                      instrument_name, 
                                                      u_exp, 
                                                      t_img, 
                                                      t_img+req.duration
                                                      )
                                    )    
        # sort list of measurements
        measurements.sort(key=lambda a : a.t_start)

        # check for if the right number of measurements was created
        assert len(measurements) == len(path)

        return measurements
    
    def _schedule_broadcasts(self, 
                             state: SimulationAgentState, 
                             measurements: list, 
                             orbitdata: dict
                             ) -> list:
        broadcasts : list[BroadcastMessageAction] = super()._schedule_broadcasts(state, 
                                                                                 measurements, 
                                                                                 orbitdata)
        
        # compile bids to be broadcasted
        compiled_bids : list[Bid] = [bid for bid in self.bids_to_rebroadcasts]
        compiled_bids.extend(self.planner_changes)
        
        # sort and make sure the latest update of each task is broadcasted
        bids = {}
        for bid_to_rebroadcast in compiled_bids: 
            bid_to_rebroadcast : Bid
            req : MeasurementRequest = MeasurementRequest.from_dict(bid_to_rebroadcast.req)

            if req.id not in bids:
                bids[req.id] = [None for _ in range(len(self._generate_bids_from_request(req, state)))]
                bids[req.id][bid_to_rebroadcast.subtask_index] = bid_to_rebroadcast
                continue

            current_bid : Bid = bids[req.id][bid_to_rebroadcast.subtask_index]
            if current_bid.t_update <= bid_to_rebroadcast.t_update:
                bids[req.id][bid_to_rebroadcast.subtask_index] = bid_to_rebroadcast
            
        # Find best path for broadcasts
        relay_path, t_start = self._create_broadcast_path(state.agent_name, orbitdata, state.t)

        # Schedule bid re-broadcast and planner changes
        bids_out = [bid_to_rebroadcast 
                    for req_id in bids 
                    for bid_to_rebroadcast in bids[req_id] 
                    if bid_to_rebroadcast is not None]
        
        for bid_out in bids_out:
            bid_out : Bid
            msg = MeasurementBidMessage(state.agent_name, state.agent_name, bid_out.to_dict(), path=relay_path)
            
            if t_start >= 0:
                broadcasts.append(BroadcastMessageAction(msg.to_dict(), t_start))

        # ensure the right number of broadcasts were created
        if len(bids_out) <= len(compiled_bids):
            pass
        else:
            assert len(bids_out) <= len(compiled_bids)
        
        return broadcasts

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

        for req, subtask_index, bid in bundle:
            bid : Bid
            assert (req, subtask_index, bid.t_img, bid.winning_bid) in path

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
        for task in [(req, subtask_index, current_bid)
                    for req, subtask_index, current_bid in bundle
                    if results[req.id][subtask_index].performed]:
            ## remove all completed tasks from bundle
            self.bundle.remove(task)
        
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

        for req, subtask_index, bid in bundle:
            bid : Bid
            assert (req, subtask_index, bid.t_img, bid.winning_bid) in path

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

        for req, subtask_index, bid in bundle:
            bid : Bid
            assert (req, subtask_index, bid.t_img, bid.winning_bid) in path

        for their_bid in bids_received:
            their_bid : Bid            

            # check bids are for new requests
            is_new_req : bool = their_bid.req_id not in results

            req : MeasurementRequest = MeasurementRequest.from_dict(their_bid.req)
            if is_new_req:
                # was not aware of this request; add to results as a blank bid
                results[req.id] = self._generate_bids_from_request(req, state)

                # add to changes broadcast in case this request was generated by me
                my_bid : Bid = results[req.id][0]
                                    
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

                for req, subtask_index, bid in bundle:
                    bid : Bid
                    assert (req, subtask_index, bid.t_img, bid.winning_bid) in path

                # remove all subsequent bids
                for bundle_index in range(outbid_index, len(bundle)):
                    # remove from bundle
                    measurement_req, subtask_index, current_bid = bundle.pop(outbid_index)

                    # remove from path
                    path.remove((measurement_req, subtask_index, current_bid.t_img, current_bid.winning_bid))

                    # reset bid results
                    current_bid : Bid; measurement_req : MeasurementRequest
                    if bundle_index > outbid_index:
                        reset_bid : Bid = current_bid.update(None, BidComparisonResults.RESET, state.t)
                        results[measurement_req.id][subtask_index] = reset_bid

                        rebroadcasts.append(reset_bid)
                        changes.append(reset_bid)
            
            # if outbid for a bid in the bundle; remove from path
            if ((req, my_bid.subtask_index, my_bid.t_img, my_bid.winning_bid) in path
                and updated_bid.winner != state.agent_name):

                path.remove((req, my_bid.subtask_index, my_bid.t_img, my_bid.winning_bid))
        
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

        # check if bundle is full
        if len(bundle) >= self.max_bundle_size:
            # Bundle is full; cannot modify 
            return results, bundle, path, changes

        # get requests that can be bid on by this agent
        available_reqs : list = self._get_available_requests(state, results, bundle, path)

        # initialize path of maximum utility
        max_path = [path_element for path_element in path]
        max_path_utility = sum([u_exp for _,_,_,u_exp in path])

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

                # get time and utility gained from said action
                path_elements = [(path_req, path_subtask, path_t, path_u) 
                                    for path_req, path_subtask, path_t, path_u in projected_path 
                                    if path_req == req and path_subtask == subtask_index]
                _, _, t_img, u_exp = path_elements[0]
                old_bid : Bid = results[req.id][subtask_index]

                if old_bid.winner == state.agent_name:
                    is_max = projected_path_utility > max_path_utility
                else:
                    is_max = (projected_path_utility > max_path_utility 
                              and u_exp > old_bid.winning_bid)

                # compare to maximum task
                # if max_req is None or is_max:
                if is_max:
                    # register maximum utility 
                    max_bid : Bid = old_bid.copy()
                    max_bid.set(u_exp, t_img, state.t)
                    max_path = [path_elem for path_elem in projected_path]
                    max_path_utility = projected_path_utility
                    max_req = req
                    max_subtask = subtask_index

                    assert (req, subtask_index, t_img, u_exp) in max_path
            
            if max_req is not None: # max bid found! 
                #place task with the best bid in the bundle and the path
                bundle.append((max_req, max_subtask, max_bid))
                path = [path_elem for path_elem in max_path]

                assert (max_req, max_subtask, max_bid.t_img, max_bid.winning_bid) in path

                # update results
                for req, subtask_index, new_bid in bundle:
                    req : MeasurementRequest
                    subtask_index : int
                    new_bid : Bid

                    old_bid : Bid = results[req.id][subtask_index]

                    if old_bid != new_bid:
                        changes.append(new_bid.copy())
                        results[req.id][subtask_index] = new_bid

        for req, subtask_index, bid in bundle:
            assert (req, subtask_index, bid.t_img, bid.winning_bid) in path

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

                # cehck if the agent has access to the task
                if not self._can_access(state, results, req, subtask_index, path):
                    continue 

                # check if the agent can bid on the tasks
                if not self._can_bid(state, results, req, subtask_index):
                    continue
                
                # check if already in bundle
                if (req, subtask_index, subtask_bid) in bundle:
                    continue

                # check if task is already in the path
                if (req, subtask_index) in path_reqs:
                    continue

                # cchek if the task has already been performed
                if self.__request_has_been_performed(results, req, subtask_index, state.t):
                    continue

                available.append((req, subtask_bid.subtask_index))  
                    
                
        return available
   
    def _can_access(self, 
                     state : SimulationAgentState, 
                     results : dict,
                     req : MeasurementRequest, 
                     subtask_index : int,
                     path : list
                     ) -> bool:
        """ Checks if an agent can access the location of a measurement request """
        if isinstance(state, SatelliteAgentState):
            bid : Bid = results[req.id][subtask_index]
            t_arrivals = [t 
                          for t in self.access_times[req.id][bid.main_measurement] 
                          if req.t_start <= t <= req.t_end]
            
            return True if t_arrivals else False
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


    @abstractmethod
    def calc_path_bid(
                        self, 
                        state : SimulationAgentState, 
                        original_results : dict,
                        original_path : list, 
                        req : MeasurementRequest, 
                        subtask_index : int
                    ) -> tuple:
        """finds the best placement of a task at a given path. returns None if this is not possible"""
        pass

    @abstractmethod
    def calc_imaging_time(self, state : SimulationAgentState, path : list, req : MeasurementRequest, subtask_index : int) -> float:
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
