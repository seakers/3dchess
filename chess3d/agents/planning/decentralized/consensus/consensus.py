from abc import abstractmethod
from collections import defaultdict, deque
from itertools import chain
from typing import Any, Dict, List, Tuple

import logging

from dmas.messages import SimulationMessage
from dmas.utils import runtime_tracker
from dmas.agents import AgentAction
from dmas.clocks import ClockConfig

from chess3d.agents.actions import BroadcastMessageAction, FutureBroadcastMessageAction, ObservationAction, WaitForMessages
from chess3d.agents.planning.reactive import AbstractReactivePlanner
from chess3d.agents.planning.tasks import GenericObservationTask, EventObservationTask, SpecificObservationTask
from chess3d.agents.planning.tracker import ObservationHistory, ObservationTracker
from chess3d.agents.planning.plan import Plan, PeriodicPlan, ReactivePlan
from chess3d.agents.planning.decentralized.consensus.bids import Bid
from chess3d.agents.science.reward import *
from chess3d.messages import BusMessage, MeasurementBidMessage
from chess3d.mission.mission import Mission
from chess3d.agents.states import SatelliteAgentState, SimulationAgentState
from chess3d.orbitdata import OrbitData
from chess3d.utils import Interval

class ConsensusPlanner(AbstractReactivePlanner):    
    # Replanning models
    # NOTE break down into separate replanner classes that use this class as the parent class?
    HEURISTIC_INSERTION = 'heuristicInsertion'
    DYNAMIC_PROGRAMMING = 'dynamicProgramming'
    MILP = 'mixedIntegerLinearProgramming'
    MODELS = [HEURISTIC_INSERTION, DYNAMIC_PROGRAMMING, MILP]
    
    # Constants
    EPS = 1e-6

    def __init__(self, 
                 model : str = HEURISTIC_INSERTION,
                 replan_threshold : int = 1,
                 optimistic_bidding_threshold : int = 1,
                 debug : bool = False,
                 logger: logging.Logger = None
                 ) -> None:
        super().__init__(debug, logger)
        """
        # Consensus Couple-Constrained Planner
        
        ## Bundle
        The Bundle is defined as a list of a tuple indicating the specific task that was added to the plan, 
        and a dictionary that maps the observation number being bid on.
        
        """

        # validate inputs
        assert model in self.MODELS, f"Invalid model '{model}'. Must be one of {self.MODELS}."
        assert isinstance(replan_threshold, int) and replan_threshold > 0, "Replan threshold must be positive integer."
        assert isinstance(optimistic_bidding_threshold, int), "Optimistic bidding threshold must be an integer"
        assert optimistic_bidding_threshold >= 0, "Optimistic bidding threshold must be non-negative"

        # initialize consensus results
        self.bundle : List[Tuple[SpecificObservationTask, Dict[GenericObservationTask, int]]] = list()
        self.path : List[ObservationAction] = list()
        self.results : Dict[GenericObservationTask, List[Bid]] = defaultdict(list)
        self.optimistic_bidding_counters : Dict[GenericObservationTask, List[int]] = defaultdict(list)

        # initialize urgent tasks and bid inbox/outbox
        self.known_event_tasks : set[GenericObservationTask] = set()
        self.incoming_event_tasks : deque[GenericObservationTask] = deque()
        self.bid_inbox : deque[Bid] = deque()
        self.bid_outbox : Dict[GenericObservationTask, Dict[int,Bid]] = defaultdict(dict)
        self.relevant_updates : List[Bid] = list()

        # initialize known preplan and current plan
        self.preplan : PeriodicPlan = None
        self.plan : Plan = None

        # set parameters
        self.model = model
        self.replan_threshold = replan_threshold
        self.optimistic_bidding_threshold = optimistic_bidding_threshold
        self.t_share = -1   

        # replanning flags 
        self.results_changes_performed = False
        self.bundle_changes_performed = False
        self.new_periodic_plan_received = False

    """
    ---------------------------
    CONSENSUS PHASE
    ---------------------------
    """

    def update_percepts(self, 
                        state : SimulationAgentState,
                        current_plan : Plan,
                        incoming_reqs: List[TaskRequest], 
                        relay_messages: List[SimulationMessage], 
                        misc_messages : List[SimulationMessage],
                        completed_actions: List[AgentAction],
                        aborted_actions : List[AgentAction],
                        pending_actions : List[AgentAction]
                    ) -> None:
        
        # check if new base plan is available
        self.__update_preplan(state, current_plan)

        # check if new task requests have arrived and filter for available requests
        self.__update_urgent_tasks(state, incoming_reqs)

        # collect bids from incoming messages to inbox
        incoming_bids : List[Bid] = self.__collect_incoming_bids(misc_messages)  

        # -------------------------------
        # DEBUG PRINTOUTS
        if self.bid_inbox:
            self.__log_results('CONSENSUS PHASE (BEFORE)', state, self.results)
            self.__log_bundle('BUNDLE (BEFORE CONSENSUS)', state, self.bundle)
        # -------------------------------

        # perform consensus phase for incoming task bids
        results_updates, bundle_updates = self._consensus_phase(state, incoming_bids)

        # -------------------------------
        # DEBUG PRINTOUTS
        if results_updates or bundle_updates:
            self.__log_results('CONSENSUS PHASE (AFTER)', state, self.results)
            self.__log_bundle('BUNDLE (AFTER CONSENSUS)', state, self.bundle)
        # -------------------------------

        # set replanning flags
        # 1) there were relevant updates to bids/results
        self.results_changes_performed = len(results_updates) > 0
        # 2) incoming bids modified the bundle
        self.bundle_changes_performed = len(bundle_updates) > 0
        # 3) new periodic plan was received
        self.new_periodic_plan_received = isinstance(current_plan, PeriodicPlan) \
                                            and abs(state.t - current_plan.t) <= self.EPS      


    def __update_preplan(self, state : SimulationAgentState, current_plan : Plan) -> None:
        """ Update latest preplan if new plan is available. """
        if isinstance(current_plan, PeriodicPlan) and abs(state.t - current_plan.t) <= self.EPS:
            self.preplan : PeriodicPlan = current_plan.copy()

    def __update_urgent_tasks(self, state : SimulationAgentState, incoming_reqs : List[TaskRequest]) -> None:
        """ Remove completed tasks from urgent tasks set. """
        # TODO remove unavailable tasks from known task lists?
        if any([not task.is_available(state.t) for task in self.known_event_tasks]):
            raise NotImplementedError("Removal of unavailable urgent tasks not yet implemented.")
            # self.known_event_tasks = set([task for task in self.known_event_tasks 
            #                                 if task.is_available(state.t)])
            # self.incoming_event_tasks = set([task for task in self.incoming_event_tasks 
            #                                 if task.is_available(state.t)])

        # get active incoming tasks
        active_tasks = set([req.task for req in incoming_reqs 
                            if req.task.is_available(state.t)])
        
        # update urgent tasks
        self.known_event_tasks.update(active_tasks)
        self.incoming_event_tasks.extend(active_tasks)

    def __collect_incoming_bids(self, misc_messages : List[SimulationMessage]) -> List[Bid]:
        """ Collect bids from incoming messages and requests. """
        # TODO include support for BidResultsMessage when re-enabled
        
        # TEMP use only MeasurementBidMessages. Disable after `BidResultsMessage` is supported
        incoming_bids = [Bid.from_dict(msg.bid) 
                            for msg in misc_messages 
                            if isinstance(msg, MeasurementBidMessage)]
        
        if incoming_bids: 
            x = 1 # Placeholder implementation

        # sort bids by bid owner, task id, n_obs, t_img
        return sorted(incoming_bids, key=lambda b: (b.bidder, b.task.id, b.n_obs, b.t_img))

    def _consensus_phase(self,
                        state : SimulationAgentState,
                        incoming_bids : List[Bid]
                    ) -> List[Bid]:
        """ Perform consensus phase to update bids and bundle. """

        # check for new urgent tasks
        new_task_added = self._process_incoming_urgent_tasks(state)

        # check if planned tasks expired
        expired_tasks = self._remove_expired_tasks(state)
        
        # check if tasks in the bundle were performed by parent agent
        self.bundle, bundle_updates = self._update_performed_bundle(state)

        # compare results with incoming bids and update bundle
        comparison_updates = self._compare_incoming_bids(state, incoming_bids)
        
        # compile updates and return list of updates
        results_updates = list(chain.from_iterable([new_task_added, 
                                                    expired_tasks, 
                                                    comparison_updates, 
                                                    bundle_updates
                                                    ]))   

        # update bundle and enforce constraints iteratively on results
        while True:
            # update bundle from results updates
            self.bundle, constraint_bundle_updates = self._update_bundle_from_results(state)

            # enforce constraints in results
            constraint_violations = self._check_results_constraints(state)

            # append updates to compiled lists
            results_updates.extend(constraint_violations)
            bundle_updates.extend(constraint_bundle_updates)

            # check for further updates
            if not constraint_bundle_updates and not constraint_violations:
                break # no more updates; exit loop       

        return results_updates, bundle_updates

    def _process_incoming_urgent_tasks(self, state: SimulationAgentState) -> List[Bid]:
        """ Processes new urgent tasks and updates results accordingly. """
        # initialize list of newly added bids from new tasks
        new_task_added = []
        
        # identify new urgent tasks
        new_event_tasks = [task for task in self.incoming_event_tasks 
                           if task not in self.results]
        
        # check if new tasks exceed threshold
        if len(new_event_tasks) < self.replan_threshold: 
            return new_task_added # threshold not met; skip processing

        # threshold met; process new tasks
        while self.incoming_event_tasks:
            # remove tasks from incoming queue
            task : GenericObservationTask = self.incoming_event_tasks.popleft()

            # check if task is already in results
            if task in self.results: continue # already processed; skip

            # initialize results for new event tasks
            self.results[task] = []

            # initialize optimistic bidding counter for new task
            self.optimistic_bidding_counters[task] = []

            # create empty bid for new task and add to list of changes
            new_task_added.append(Bid(task, state.agent_name))

        # return list of new task bids added to results
        return new_task_added
    
    def _remove_expired_tasks(self, state : SimulationAgentState) -> List[Bid]:
        """ Remove expired tasks from results. """
        # initialize list of removed bids
        removed_bids = []

        # identify expired tasks
        expired_tasks = [task for task in self.results 
                         if not task.is_available(state.t)]
        
        if expired_tasks: raise NotImplementedError("Removal of expired tasks not yet tested.")
        
        # remove expired tasks from results
        for task in expired_tasks:
            # remove task from results
            bids_removed = self.results.pop(task)

            # remove optimistic bidding counters
            self.optimistic_bidding_counters.pop(task, None)

            # add removed bids to list
            removed_bids.extend(bids_removed)

        # return list of removed bids
        return removed_bids
    
    def _update_performed_bundle(self, state : SimulationAgentState) -> Tuple[list, List[Bid]]:
        """ Checks if planned tasks were performed by parent agent and updates results accordingly. """
        
        # initialize list of bundle updates
        bundle_updates = []

        # initialize list of performed tasks to remove from bundle
        performed_tasks = []

        # iterate through bundle to identify performed tasks
        for specific_task, obs_tasks in self.bundle:
            # check if imaging time for winning tasks has passed
            if any([self.results[task][n_obs].t_img < state.t 
                    for task,n_obs in obs_tasks.items()]):

                # TODO testing pending; raise error for now
                raise NotImplementedError("Marking bids as performed not yet implemented.")
                
                # imaging time has passed for task bids; assume tasks were performed by parent agent
                assert any([self.results[task][n_obs].winning_bidder == state.agent_name for task,n_obs in obs_tasks.items()]), \
                    "Cannot mark tasks as performed if this agent is not the winning bidder."
                
                # mark bids as performed
                performed_bids = []
                for task, n_obs in obs_tasks.items():
                    bid_to_perform : Bid = self.results[task][n_obs]

                    # mark bid as performed
                    performed_bid = bid_to_perform.set_performed(state.t, performed=True)

                    # update results
                    self.results[task][n_obs] = performed_bid

                    # add to list of performed bids
                    performed_bids.append(performed_bid)
                
                # add bids to list of bundle updates
                bundle_updates.append(performed_bids)

                # add tasks to list of performed tasks
                performed_tasks.append((specific_task, obs_tasks))
               
        # create revised bundle considering newly performed tasks
        revised_bundle = [entry for entry in self.bundle 
                          if entry not in performed_tasks] \
                            if performed_tasks else self.bundle
                
        # return revised bundle and list of performed bids
        return revised_bundle, bundle_updates

    def _compare_incoming_bids(self,
                       state : SimulationAgentState,
                       incoming_bids : List[Bid]
                       ) -> Tuple[List[Bid], List[Bid]]:
        """ Update results from incoming bids. """
        # initialize list of updates done to results
        results_updates = []        

        # process incoming bids
        for incoming_bid in incoming_bids:

            # check if bid is for a new task or higher observation number
            new_task : bool = incoming_bid.task not in self.results
            new_observation_number : bool = (not new_task) and (incoming_bid.n_obs >= len(self.results[incoming_bid.task]))

            # if new task or observation number, initialize in results
            if new_task or new_observation_number:
                # check if new task is even available at this time
                if not incoming_bid.task.is_available(state.t):
                    # task not available; skip bid consideration
                    continue

                # assume bids are received in order of observation numbers
                assert len(self.results[incoming_bid.task]) == incoming_bid.n_obs , \
                      "Received bids for non-consecutive observation numbers."

                # add an empty bid for each missing observation number
                empty_bid = Bid(incoming_bid.task, state.agent_name, incoming_bid.n_obs)
                self.results[incoming_bid.task].append(empty_bid)

                # initialize optimistic bidding counter for new bid
                self.optimistic_bidding_counters[incoming_bid.task].append(self.optimistic_bidding_threshold)

            # get current bid for this task and observation number
            current_bid : Bid = self.results[incoming_bid.task][incoming_bid.n_obs]

            # compare incoming bid with existing bids for the same task
            updated_bid : Bid = current_bid.update(incoming_bid, state.t)

            # update results with modified bid
            self.results[incoming_bid.task][incoming_bid.n_obs] = updated_bid

            # if bid was changed; add updated bid to results updates
            if updated_bid.has_different_winner_values(current_bid): 
                results_updates.append(updated_bid)
        
        # TEMP ensure all bids have this agent as the bidder and task matches. Remove after testing
        assert all(bid.bidder == state.agent_name and bid.task == task
                   for task, bids in self.results.items() for bid in bids)

        # return result changes
        return results_updates
         
    def _update_bundle_from_results(self,
                                    state : SimulationAgentState
                                    ) -> Tuple[list, List[List[Bid]]]:
        """ Update bundle according to latest results. """
        # initialize list of bundle updates
        bundle_updates = []

        # count initial bundle size
        init_bundle_size = len(self.bundle)

        # check if there are any results updates for tasks in the bundle
        min_updated_idx = min([ idx 
                                for idx,(_,obs_tasks) in enumerate(self.bundle)
                                if not all(len(self.results[task]) > n_obs
                                        and self.results[task][n_obs].is_bidder_winning() 
                                        for task, n_obs in obs_tasks.items())
                                ], 
                                default=None)
        
        # check if any updates were found
        if min_updated_idx is None:
            # no updates to bundle needed; return original bundle
            return self.bundle, bundle_updates

        # split bundle at first updated task
        revised_bundle = self.bundle[:min_updated_idx]

        # reset subsequent bids for all tasks in bundle if the bidder is still listed as the winner
        for _, obs_tasks in self.bundle[min_updated_idx:]:
            for task, n_obs in obs_tasks.items():
                # reset invalid bid along with all subsequent bids
                bids_reset = []
                for bid_idx in range(n_obs, len(self.results[task])):
                    # check if this agent is still listed as the winning bidder
                    if not self.results[task][bid_idx].is_bidder_winning():
                        continue # another agent is winning this bid; skip

                    # get bid to reset and remove from results
                    bid_to_reset : Bid = self.results[task][bid_idx]

                    # reset bid
                    reset_bid = bid_to_reset.reset(state.t)

                    # update results
                    self.results[task][bid_idx] = reset_bid

                    # add to list of resets
                    bids_reset.append(reset_bid)

                # add to violations list
                bundle_updates.append(bids_reset)

        assert len(revised_bundle) + len(self.bundle[min_updated_idx:]) == init_bundle_size, \
            "Revised bundle size does not match initial bundle size."

        # return updated bundle and list of updates
        return revised_bundle, bundle_updates

    def _check_results_constraints(self, state : SimulationAgentState) -> List[Bid]:
        """ Check results for constraint violations and return list of affected bids. """
        # initiate list of constraint violations
        bids_in_violation = []

        # check every task for constraint violations
        for bids in self.results.values():            
            # assume the index of every bid matches their observation number
            assert all(bid.n_obs == i_obs for i_obs, bid in enumerate(bids)), \
                "Results bids are not sorted by observation number."

            if len(bids) <= 1: continue # no observation sequence to check for constraints
            
            # initialize search for constraint violations
            invalid_bid_idx : int = None
            
            # check every bid for this task
            for n_obs_idx, bid in enumerate(bids[1:], start=1):
                # get previous bid to compare constraints with
                prev_bid : Bid = bids[n_obs_idx - 1]

                # define constraints
                constraints : List[bool] = [
                    # Constraint 0: Previous bid must be assigned to a winner
                    prev_bid.has_winner(),
                    # Constraint 1: Observation number must be consecutive
                    prev_bid.n_obs + 1 == bid.n_obs,
                    # Constraint 2: Imaging time must be after previous imaging time
                    prev_bid.t_img <= bid.t_img
                ]
                    
                # check if any constraint is violated
                if not all(constraints):
                    # mark this bid as invalid
                    invalid_bid_idx = n_obs_idx

                    # stop searching for constraint violations for this task
                    break
            
            # check if invalid bid was found
            if invalid_bid_idx is None: continue # no violations for this task; continue to next task

            # reset invalid bid along with all subsequent bids
            while len(bids) > invalid_bid_idx:
                # get bid to reset and remove from results
                bid_to_reset : Bid = bids.pop(invalid_bid_idx)

                # reset bid
                reset_bid = bid_to_reset.reset(state.t)

                # add to violations list
                bids_in_violation.append(reset_bid)                

        # return list of bids in violation
        return bids_in_violation   
    
    def needs_planning(self, 
                       state : SimulationAgentState,
                       specs : object,
                       current_plan : Plan,
                       orbitData : OrbitData
                    ) -> bool:
        try:
            # -------------------------------
            # DEBUG BREAKPOINTS
            if self.results_changes_performed:
                x = 1  # Placeholder implementation
            if self.bundle_changes_performed:
                x = 1  # Placeholder implementation
            if self.new_periodic_plan_received:
                x = 1  # Placeholder implementation
            # -------------------------------

            # trigger replan if either...
            return (                    
                    self.results_changes_performed      # 1) there were relevant updates to bids/results
                    or self.bundle_changes_performed    # 2) incoming bids modified the bundle
                    or self.new_periodic_plan_received  # 3) new periodic plan was received
                    )
        
        finally:
            # reset replanning flags
            self.results_changes_performed = False
            self.bundle_changes_performed = False
            self.new_periodic_plan_received = False

    """
    ---------------------------
    BUNDLE-BUILDING PHASE
    ---------------------------
    """
    def generate_plan(self, 
                      state : SimulationAgentState,
                      specs : object,
                      current_plan : Plan,
                      clock_config : ClockConfig,
                      orbitdata : OrbitData,
                      mission : Mission,
                      tasks : List[GenericObservationTask],
                      observation_history : ObservationHistory,
                    ) -> Plan:  
        """ Generate new plan according to consensus replanning model. """             
        # DEBUG return original preplan
        # return ReactivePlan.from_periodic_plan(self.preplan,t=state.t)

        # -------------------------------
        # DEBUG PRINTOUTS
        self.__log_results('PLANNING PHASE (BEFORE)', state, self.results)
        self.__log_bundle('BUNDLE (BEFORE PLANNING)', state, self.bundle)
        # -------------------------------

        # build new bundle
        new_bundle, new_path, new_bids = self.bundle_building_phase(state, specs, current_plan, clock_config, orbitdata, mission, observation_history)
        
        # check if new path is valid
        assert new_path is not None and len(new_path) > 0, "New observation path cannot be empty."
        assert self.is_observation_path_valid(state, new_path, None, None, specs), "New observation path is not valid."   

        # update bundle and path
        self.bundle, self.path = new_bundle, new_path

        # update results
        self.__update_results_from_bundle(new_bundle)

        # update bid outbox
        self.__update_outbox_from_bundle(new_bundle)

        # -------------------------------
        # DEBUG PRINTOUTS
        self.__log_results('PLANNING PHASE', state, self.results)
        self.__log_bundle('BUNDLE (AFTER PLANNING)', state, self.bundle)
        # -------------------------------
    
        # generate maneuver and travel actions from observations
        maneuvers : list = self._schedule_maneuvers(state, specs, new_path, clock_config, orbitdata)

        # schedule broadcasts
        # TODO decide on broadcast scheduling strategy
        broadcasts : list = self._schedule_broadcasts(state, orbitdata)
                
        # compile and generate plan
        self.plan = ReactivePlan(maneuvers, new_path, broadcasts, t=state.t, t_next=self.preplan.t_next)

        # clear new urgent tasks
        self.incoming_event_tasks = set()

        # return final plan
        return self.plan.copy()
    
    @abstractmethod
    def bundle_building_phase(self,
                       state : SimulationAgentState,
                       specs : object,
                       current_plan : Plan,
                       clock_config : ClockConfig,
                       orbitdata : OrbitData,
                       mission : Mission,
                       observation_history : ObservationHistory
                    ) -> tuple:        
        """ 
        Build bundle according to selected replanning model. 
        #### Returns:
            - `new_bundle` : List[List[Tuple[GenericObservationTask, int]]] -- New bundle of bids
            - `new_path` : List[GenericObservationTask] -- New observation path
            - `new_bids` : List[Bid] -- New bids generated during bundle building
        
        """
    
    def __update_results_from_bundle(self, new_bundle : List[List[Bid]]) -> None:
        """ Update results dictionary from new bundle. """
        
        # # check if this agent was the one who bid on the invalid bid
        # if self.results[task][invalid_bid_idx].bidder == state.agent_name:
        #     # decrement optimistic bidding counter for this bid (floor at 0)
        #     self.optimistic_bidding_counters[task][invalid_bid_idx] = \
        #         max(0, self.optimistic_bidding_counters[task][invalid_bid_idx] - 1)
        

        raise NotImplementedError("Updating results from new bundle not yet implemented.")
        # iterate through bids in new bundle
        for bids in new_bundle:
            for bid in bids:
                # check if bid for this task and observation number already exists
                if len(self.results[bid.task]) <= bid.n_obs:
                    # add empty bids up to `n_obs`
                    for i_obs in range(bid.n_obs - len(self.results[bid.task]) + 1):
                        n_obs = len(self.results[bid.task]) + i_obs
                        self.results[bid.task].append(AsynchronousBid(bid.task, bid.bidder, n_obs=n_obs))
                
                # get existing bid
                existing_bid = self.results[bid.task][bid.n_obs]
                
                # check if new bid is better than existing bid
                assert bid > existing_bid,\
                      "Generated a bid that is not better than existing bid in results."

                # update results with new bid
                self.results[bid.task][bid.n_obs] = bid

        # ensure that results bids match their observation number
        assert all(
                bid.n_obs == obs_idx
                for task in self.results
                for obs_idx, bid in enumerate(self.results[task])
            ), "Results bids are not sorted by observation number."
        
        # ensure that bids are consecutive in observation numbers
        assert all(
            all(bid.n_obs == idx for idx, bid in enumerate(bids))
            for bids in self.results.values()
        ), "Bids are not sorted in consecutive n_obs values."

    def __update_outbox_from_bundle(self, new_bundle : List[List[Bid]]) -> None:
        """ Update bid outbox from new bundle. """
        raise NotImplementedError("Updating bid outbox from new bundle not yet implemented.")
    
        # iterate through bids in new bundle
        for bids in new_bundle:
            for bid in bids:
                # check if bid for this task and observation number already exists in outbox
                if bid.n_obs not in self.bid_outbox[bid.task]:
                    # add new bid to outbox
                    self.bid_outbox[bid.task][bid.n_obs] = bid
                else:
                    # get existing bid
                    existing_bid = self.bid_outbox[bid.task][bid.n_obs]
                    
                    # update outbox with newest bid
                    self.bid_outbox[bid.task][bid.n_obs] = max(existing_bid, bid, key=lambda b: (b.t_stamp, b.bid_value))
       
    """
    BROADCAST SCHEDULING
    """
    def _schedule_broadcasts(self, state: SimulationAgentState, orbitdata: OrbitData) -> list:
        """ Schedules broadcasts to be done by this agent """
        try:
            if not isinstance(state, SatelliteAgentState):
                raise NotImplementedError(f'Broadcast scheduling for agents of type `{type(state)}` not yet implemented.')
            elif orbitdata is None:
                raise ValueError(f'`orbitdata` required for agents of type `{type(state)}`.')

            # initialize list of broadcasts to be done
            broadcasts = []       
            
            # iterate through communication targets
            for target in orbitdata.comms_links.keys():
                # get access intervals with target agent
                access_intervals : List[Interval] = orbitdata.get_next_agent_accesses(target, state.t, include_current=True)

                # create broadcast actions for each access interval
                for next_access in access_intervals:
                    # if no access opportunities in this planning horizon, skip scheduling
                    if next_access.is_empty(): continue

                    # get last access interval and calculate broadcast time
                    t_broadcast : float = max(next_access.left, state.t)

                    # generate plan message to share state
                    state_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.STATE, t_broadcast)

                    # generate plan message to share completed observations
                    observations_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.OBSERVATIONS, t_broadcast)

                    # generate plan message to share any task requests generated
                    task_requests_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.REQUESTS, t_broadcast)

                    # generate bid messages to share bids in results
                    bid_msgs : List[MeasurementBidMessage]= []
                    for bids in self.bid_outbox.values():
                        for bid in bids.values():
                            bid_msgs.append(MeasurementBidMessage(state.agent_name, state.agent_name, bid.to_dict()))
                    bid_bus_msg = BusMessage(state.agent_name, state.agent_name, [bid_msg.to_dict() for bid_msg in bid_msgs])
                    bid_msg_action = BroadcastMessageAction(bid_bus_msg.to_dict(), t_broadcast)
                    
                    # add to client broadcast list
                    broadcasts.extend([state_msg, observations_msg, task_requests_msg, bid_msg_action])

            if not orbitdata.comms_links:
                # no communication links available, broadcast task requests for future planning horizons
                t_broadcast : float = state.t

                # generate plan message to share any task requests generated
                task_requests_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.REQUESTS, t_broadcast)

                # add to client broadcast list
                broadcasts.append(task_requests_msg)

            # return scheduled broadcasts
            return broadcasts 
        
        finally:
            assert isinstance(broadcasts, list), "Scheduled broadcasts is not a list."
            assert all(isinstance(broadcast, BroadcastMessageAction) for broadcast in broadcasts), "Not all scheduled broadcasts are of type `BroadcastMessageAction`."

    """
    LOGGING
    """
    def __log_results(self, dsc : str, state : SimulationAgentState, level=logging.DEBUG) -> None:
        out = f'\nT{np.round(state.t,3)}[s]:\t\'{state.agent_name}\'\n{dsc}\n'
        line = 'Req ID\t n_obs\tins\twinner\tbid\tt_img\tt_stamp  performed\n'
        
        # count characters in line for formatting
        L_LINE = len(line)

        # header
        out += line 

        # divider 
        for _ in range(L_LINE + 25): out += '='
        out += '\n'

        n = 15
        i = 1
        for task, bids in self.results.items():
            task : GenericObservationTask
            req_id_short = task.id.split('-')[-1]

            # if all([bid.winner == bid.NONE for _,bid in bids.items()]): continue

            if not bids:
                out += f'{req_id_short} <none>\n'

            for bid in bids:
                # if i > n: break

                bid : Bid
                # if bid.winner == bid.NONE: continue

                if bid.winning_bidder != bid.NONE:
                    line = f'{req_id_short} {bid.n_obs}\t{bid.main_measurement}\t{bid.winning_bidder[0].lower()}{bid.winning_bidder[-1]}\t{np.round(bid.winning_bid,3)}\t{np.round(bid.t_img,3)}\t{np.round(bid.t_stamp,1)}\t  {(bid.performed)}\n'
                else:
                    line = f'{req_id_short} {bid.n_obs}\t{bid.main_measurement}\tn/a\t{np.round(bid.winning_bid,3)}\t{np.round(bid.t_img,3)}\t{np.round(bid.t_stamp,1)}\t  {(bid.performed)}\n'
                out += line
                i +=1

            for _ in range(L_LINE + 25):
                out += '-'
            out += '\n'

            if i > n:
                out += '\t\t\t...\n'
                for _ in range(L_LINE + 25):
                    out += '-'
                out += '\n'
                break

        print(out)

    def __log_bundle(self, dsc : str, state : SimulationAgentState, level=logging.DEBUG) -> None:
        out = f'\nT{np.round(state.t,3)}[s]:\t\'{state.agent_name}\'\n{dsc}\n'
        line = 'i\t Req IDs\n'
        
        # count characters in line for formatting
        L_LINE = len(line)

        # header
        out += line 

        # divider 
        for _ in range(L_LINE + 20): out += '='
        out += '\n'

        if not self.bundle:
            out += '\t<empty bundle>\n'
            for _ in range(L_LINE + 25): out += '-'
            out += '\n'

        n = 15
        for i,bids in enumerate(self.bundle):
            line = f'{i}\t['
            for bid in bids:
                # if i > n: break
                line += f'{bid.task.id.split("-")[-1]}({bid.n_obs}),'
            line = line[:-1] + ']\n'
            out += line

            for _ in range(L_LINE + 25):
                out += '-'
            out += '\n'

            if i > n:
                out += '\t\t\t...\n'
                for _ in range(L_LINE + 25):
                    out += '-'
                out += '\n'
                break

        print(out)