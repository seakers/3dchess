from abc import abstractmethod
from collections import defaultdict
from itertools import chain
from typing import Dict, List, Tuple

import logging

from dmas.messages import SimulationMessage
from dmas.utils import runtime_tracker
from dmas.agents import AgentAction
from dmas.clocks import ClockConfig

from chess3d.agents.actions import BroadcastMessageAction, FutureBroadcastMessageAction, ObservationAction
from chess3d.agents.planning.reactive import AbstractReactivePlanner
from chess3d.agents.planning.tasks import DefaultMissionTask, EventObservationTask, GenericObservationTask
from chess3d.agents.planning.observations import ObservationOpportunity
from chess3d.agents.planning.tracker import ObservationHistory
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
                 model : str,
                 replan_threshold : int,
                 optimistic_bidding_threshold : int,
                 debug : bool = False,
                 logger: logging.Logger = None
                 ) -> None:
        super().__init__(debug, logger)
        """
        # Consensus Couple-Constrained Planner
        
        ## Bundle
        The Bundle is defined as a list of a tuple indicating the specific observation opportunity that was added to the plan, 
        and a dictionary that maps the observation number being bid on.
        
        """

        # validate inputs
        assert isinstance(model, str), "Model must be a string."
        assert model in self.MODELS, f"Invalid model '{model}'. Must be one of {self.MODELS}."
        assert isinstance(replan_threshold, int) and replan_threshold > 0, "Replan threshold must be positive integer."
        assert isinstance(optimistic_bidding_threshold, int), "Optimistic bidding threshold must be an integer"
        assert optimistic_bidding_threshold >= 0, "Optimistic bidding threshold must be non-negative"

        # initialize consensus results
        self.bundle : List[Tuple[ObservationOpportunity, Dict[GenericObservationTask, int]]] = list()
        self.path : List[ObservationAction] = list()
        self.results : Dict[GenericObservationTask, List[Bid]] = defaultdict(list)
        self.optimistic_bidding_counters : Dict[GenericObservationTask, List[int]] = defaultdict(list)

        # initialize urgent tasks and bid inbox/outbox
        self.known_event_tasks : set[GenericObservationTask] = set()
        self.incoming_event_tasks : list[GenericObservationTask] = list()
        self.relevant_updates : List[Bid] = list()

        # initialize known preplan and current plan
        self.preplan : PeriodicPlan = PeriodicPlan([])
        self.plan : Plan = None

        # set parameters
        self.model = model
        self.replan_threshold = replan_threshold
        self.optimistic_bidding_threshold = optimistic_bidding_threshold
        self.t_share = -1   

        # replanning flags 
        self.results_changes_performed = False
        self.bundle_changes_performed = False

    """
    ---------------------------
    CONSENSUS PHASE
    ---------------------------
    """
    def update_percepts(self, 
                        state : SimulationAgentState,
                        current_plan : Plan,
                        tasks : List[GenericObservationTask],
                        incoming_reqs: List[TaskRequest], 
                        relay_messages: List[SimulationMessage], 
                        misc_messages : List[SimulationMessage],
                        completed_actions: List[AgentAction],
                        aborted_actions : List[AgentAction],
                        pending_actions : List[AgentAction]
                    ) -> None:
        """ Updates internal knowledge based on incoming percepts """

        # update base percepts
        super().update_percepts(state, incoming_reqs, relay_messages, completed_actions)

        # collect bids from incoming messages to inbox
        incoming_bids : List[Bid] = self.__collect_incoming_bids(misc_messages)  

        # collect performed observations from completed actions
        performed_observations : List[ObservationAction] = [action for action in completed_actions if isinstance(action, ObservationAction)]

        # -------------------------------
        # DEBUG PRINTOUTS
        if (incoming_bids or self.incoming_event_tasks) and self._debug:
            self._log_results('CONSENSUS PHASE - RESULTS (BEFORE)', state, self.results)
            self._log_bundle('CONSENSUS PHASE - BUNDLE (BEFORE)', state, self.bundle)

        if performed_observations and self._debug:
            x = 1 # debug breakpoint
        # -------------------------------

        # perform consensus phase for incoming task bids
        results_updates, bundle_updates, self.last_performed_observations \
              = self._consensus_phase(state, incoming_reqs, incoming_bids, tasks, current_plan, performed_observations)

        # assume bundle and results are now consistent
        assert len(self.bundle) == len(self.path), \
            "Bundle and path lengths do not match after consensus phase."

        # -------------------------------
        # DEBUG PRINTOUTS
        if (results_updates or bundle_updates) and self._debug:
            self._log_results('CONSENSUS PHASE - RESULTS (AFTER)', state, self.results)
            self._log_bundle('CONSENSUS PHASE - BUNDLE (AFTER)', state, self.bundle)
            x = 1 # debug breakpoint
        # -------------------------------

        # set replanning flags
        # 1) there were relevant updates to bids/results
        self.results_changes_performed = len(results_updates) > 0
        # 2) incoming bids modified the bundle
        self.bundle_changes_performed = len(bundle_updates) > 0

    def __collect_incoming_bids(self, misc_messages : List[SimulationMessage]) -> List[Bid]:
        """ Collect bids from incoming messages and requests. """
        # TODO include support for BidResultsMessage when re-enabled
        
        # TEMP use only MeasurementBidMessages. Disable after `BidResultsMessage` is supported
        incoming_bids = [Bid.from_dict(msg.bid) 
                            for msg in misc_messages 
                            if isinstance(msg, MeasurementBidMessage)]

        # sort bids by bid owner, task id, n_obs, t_img
        return sorted(incoming_bids, key=lambda b: (b.owner, b.task.id, b.n_obs, b.t_img))

    def _consensus_phase(self,
                        state : SimulationAgentState,
                        incoming_reqs : List[TaskRequest],
                        incoming_bids : List[Bid],
                        tasks : List[GenericObservationTask],
                        current_plan : Plan,
                        performed_observations : List[ObservationAction]
                    ) -> List[Bid]:
        """ Perform consensus phase to update bids and bundle. """

        # check for new default mission tasks
        new_default_tasks = self._process_default_tasks(state, tasks)

        # check for new urgent tasks
        new_urgent_task_added = self._process_incoming_urgent_tasks(state, incoming_reqs)

        # check if planned tasks expired
        expired_tasks = self._remove_expired_tasks(state)

        # check if new base plan is available
        self.bundle, self.path, preplan_updates = self.__update_bundle_from_preplan(state, current_plan)
        
        # check if tasks in the bundle were performed by parent agent
        self.bundle, self.path, performed_bundle_updates = self._update_performed_bundle(state, performed_observations)

        # compare results with incoming bids and update bundle
        comparison_updates = self._compare_incoming_bids(state, incoming_bids)

        # check if bids in results would have been performed by other agents
        performed_updates = self._update_performed_bids(state)
        
        # compile updates and return list of updates
        results_updates = list(chain.from_iterable([
                                                    new_default_tasks,
                                                    new_urgent_task_added, 
                                                    expired_tasks, 
                                                    preplan_updates,
                                                    performed_bundle_updates,
                                                    comparison_updates, 
                                                    performed_updates,
                                                   ]))   
        bundle_updates = list(chain.from_iterable([
                                                    preplan_updates,
                                                    performed_bundle_updates
                                                ]))

        # update bundle and enforce constraints iteratively on results
        while True:
            # update bundle from results updates
            self.bundle, self.path, constraint_bundle_updates = self._update_bundle_from_results(state)

            # enforce constraints in results
            constraint_violations = self._check_results_constraints(state)

            # append updates to compiled lists
            results_updates.extend(constraint_violations)
            bundle_updates.extend(constraint_bundle_updates)

            # check for further updates
            if not constraint_bundle_updates and not constraint_violations:
                break # no more updates; exit loop       
        
        # collect performed observation opportunities
        performed_observation_opportunities : List[ObservationOpportunity] \
            = [obs_opp for obs_opp,_ in performed_bundle_updates]
        
        # return lists of updates
        return results_updates, bundle_updates, performed_observation_opportunities   

    def __update_bundle_from_preplan(self, 
                                     state : SimulationAgentState, 
                                     current_plan : Plan
                                    ) -> Tuple[list, List[Bid]]:
        """ Update latest preplan if new plan is available. """
        # check if new periodic plan is available
        if isinstance(current_plan, PeriodicPlan) and abs(state.t - current_plan.t) <= self.EPS:
            # save new preplan
            self.preplan : PeriodicPlan = current_plan.copy()

            # obtain observations path from new preplan
            preplan_path : List[ObservationAction] = \
                  [action for action in current_plan if isinstance(action, ObservationAction)]

            # ensure all tasks in preplan observations are known in results
            assert all((task in self.results for obs_action in preplan_path for task in obs_action.obs_opp.tasks)), \
                "All tasks in preplan observations must be known in results."
            
            if any((isinstance(task, EventObservationTask) for obs_action in preplan_path for task in obs_action.obs_opp.tasks)):
                raise NotImplementedError("Updating preplan bids with urgent tasks not yet implemented.")

            # get series of observation number and time for each task in preplan
            n_obs, _ = self._count_observations_and_revisit_times_from_path(preplan_path)

            # calculate observation values for each preplanned observation
            obs_values = [{task : obs_action.obs_opp.get_priority() # TODO implement preplan observation value calculation
                           for task in obs_action.obs_opp.tasks}
                          for _, obs_action in enumerate(preplan_path)]

            # create bundle from list of bids from new preplan observations
            preplan_bundle_bids = [(obs_action.obs_opp, [Bid(task, state.agent_name, 
                                                   n_obs[obs_idx][task], 
                                                   obs_values[obs_idx][task],
                                                   obs_values[obs_idx][task],
                                                   state.agent_name, 
                                                   obs_action.t_start, 
                                                   current_plan.t, 
                                                   {state.agent_name: current_plan.t}, 
                                                   obs_action.instrument_name)
                                  for task in obs_action.obs_opp.tasks] )
                                  for obs_idx,obs_action in enumerate(preplan_path)]
            
            preplanned_bundle = [ (obs_opp, {bid.task: bid.n_obs for bid in bids}) 
                                 for obs_opp, bids in preplan_bundle_bids]

            # update results with new preplan bids
            for _, bids in preplan_bundle_bids:
                for bid in bids:
                    # add bid to results
                    if bid.n_obs >= len(self.results[bid.task]):
                        # assume bids are received in order of observation numbers
                        assert len(self.results[bid.task]) == bid.n_obs, \
                              "Received bids for non-consecutive observation numbers."
                        # add an empty bid for each missing observation number
                        self.results[bid.task].append(bid)
                    else:
                        # update existing bid
                        self.results[bid.task][bid.n_obs] = bid

                    # initialize optimistic bidding counter for new bid
                    self.optimistic_bidding_counters[bid.task].append(self.optimistic_bidding_threshold)

            # return new bundle and list of preplan updates
            return preplanned_bundle, preplan_path, [bid for _, bids in preplan_bundle_bids for bid in bids]

        # no new preplan available; return no updates
        return self.bundle, self.path, []
    
    def _process_default_tasks(self, state: SimulationAgentState, tasks: List[DefaultMissionTask]) -> List[Bid]:
        """ Processes new default mission tasks and updates results accordingly. """
        # initialize list of newly added bids from new tasks
        new_task_added = []

        # identify new default tasks
        unknown_tasks = [task for task in tasks if task not in self.results]
        
        # process each default task
        for task in unknown_tasks:
            # initialize results for new default tasks
            self.results[task] = []

            # initialize optimistic bidding counter for new task
            self.optimistic_bidding_counters[task] = []

            # create empty bid for new task and add to list of changes
            new_task_added.append(Bid(task, state.agent_name))

        # return list of new task bids added to results
        return new_task_added
    
    def _process_incoming_urgent_tasks(self, state: SimulationAgentState,  incoming_reqs : List[TaskRequest]) -> List[Bid]:
        """ Processes new urgent tasks and updates results accordingly. """
        # initialize list of newly added bids from new tasks
        new_task_added = []

        # get active incoming tasks
        active_tasks = set([req.task for req in incoming_reqs 
                            if req.task.is_available(state.t)])
        
        # update urgent tasks
        self.known_event_tasks.update(active_tasks)
        self.incoming_event_tasks.extend(active_tasks)
                
        # remove unavailable tasks from known task lists
        if any([not task.is_available(state.t) for task in self.known_event_tasks]):
            raise NotImplementedError("Removal of unavailable urgent tasks not yet implemented.")
        self.known_event_tasks = {task for task in self.known_event_tasks if task.is_available(state.t)}
        
        # identify new urgent tasks
        new_event_tasks = [task for task in self.incoming_event_tasks 
                           if task not in self.results]

        # check if new tasks exceed threshold
        if len(new_event_tasks) < self.replan_threshold: 
            return new_task_added # threshold not met; skip processing

        # threshold met; process new tasks
        for task in self.incoming_event_tasks:

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
        
        if expired_tasks: 
            # TODO implement removal of expired tasks
            raise NotImplementedError("Removal of expired tasks not yet tested.")
        
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
    
    def _update_performed_bundle(self, state : SimulationAgentState, performed_observations : List[ObservationAction]) -> Tuple[list, List[Bid]]:
        """ Checks if planned tasks were performed by parent agent and updates results accordingly. """
        
        # initialize list of bundle updates
        bundle_updates = []

        # initialize list of performed tasks to remove from bundle
        performed_task_bids = []

        # collect actions in bundle past their imaging time
        observed_opportunities : list[ObservationOpportunity] = [obs_action.obs_opp for obs_action in performed_observations]

        performed_bundle_tasks = [ (obs_opp, obs_tasks) 
                                    for obs_opp, obs_tasks in self.bundle
                                    if obs_opp in observed_opportunities]

        # iterate through performed bundle to mark bids as performed
        for obs_opp, obs_tasks in performed_bundle_tasks:     
                       
            # imaging time has passed for task bids; assume tasks were performed by parent agent
            assert any([self.results[task][n_obs].winner == state.agent_name for task,n_obs in obs_tasks.items()]), \
                "Cannot mark tasks as performed if this agent is not the winning bidder."
            
            # mark bids as performed
            performed_bids = []
            for task, n_obs in obs_tasks.items():
                bid_to_perform : Bid = self.results[task][n_obs]

                # mark bid as performed
                bid_to_perform.set_performed(state.t, performed=True)

                # update results
                self.results[task][n_obs] = bid_to_perform

                # add to list of performed bids
                performed_bids.append(bid_to_perform.copy())
            
            # add bids to list of bundle updates
            bundle_updates.append((obs_opp, performed_bids))

            # add tasks to list of performed tasks
            performed_task_bids.append((obs_opp, obs_tasks))
               
        # create revised bundle considering newly performed tasks
        revised_bundle = [entry for entry in self.bundle 
                          if entry not in performed_task_bids] \
                            if performed_task_bids else self.bundle
        
        # create revised path considering newly performed tasks
        performed_bundle_obs = [obs_opp for obs_opp,_ in performed_bundle_tasks]
        revised_path = [obs_action for obs_action in self.path
                        if obs_action.obs_opp not in performed_bundle_obs]
                
        # return revised bundle and list of performed bids
        return revised_bundle, revised_path, bundle_updates

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
        assert all(bid.owner == state.agent_name and bid.task == task
                   for task, bids in self.results.items() for bid in bids)

        # return result changes
        return results_updates
    
    def _update_performed_bids(self, state : SimulationAgentState) -> List[Bid]:
        """ Assumes tasks who were won by other agents and whose imaging time has passed were performed by those agents. """

        # initialize list of performed updates
        performed_updates = []

        # check every bid in results for performed status
        for task, bids in self.results.items():
            for n_obs, bid in enumerate(bids):
                # check if bid is already marked as performed
                if bid.was_performed():
                    continue # already marked or has no winner; skip
                
                # check if imaging time has passed
                if bid.t_img < state.t:
                    # assume bid has a winner different from this agent
                    assert bid.has_winner(), "Cannot mark bid as performed if it has no winner."
                    assert bid.winner != state.agent_name, "Bid should have been marked as performed by parent agent in previous step."
                    
                    # mark bid as performed
                    bid.set_performed(state.t, performed=True, performer=bid.winner)

                    # update results
                    self.results[task][n_obs] = bid

                    # add to list of performed updates
                    performed_updates.append(bid.copy())

        # return list of performed bids
        return performed_updates
         
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
            return self.bundle, self.path, bundle_updates

        # split bundle at first updated task
        revised_bundle = self.bundle[:min_updated_idx]

        # iterate through remaining bundle to update bids
        for _, obs_tasks in self.bundle[min_updated_idx:]:
            # reset subsequent bids for all tasks in bundle if the bidder is still listed as the winner
            for task, n_obs in obs_tasks.items():
                # initiate list of bids being reset for this task
                bids_reset = []

                # reset invalid bid along with all subsequent bids  
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

            # remove the remaining 

        assert len(revised_bundle) + len(self.bundle[min_updated_idx:]) == init_bundle_size, \
            "Revised bundle size does not match initial bundle size."
        
        # collect list of observation opportunities in the revised bundle 
        obs_opps_in_revised_bundle = {obs_opp for obs_opp,_ in revised_bundle}
        
        # revise path according to revised bundle
        revised_path = [obs_action for obs_action in self.path
                        if obs_action.obs_opp in obs_opps_in_revised_bundle]

        # return updated bundle and list of updates
        return revised_bundle, revised_path, bundle_updates
    
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
            # -------------------------------

            # trigger replan if either...
            return (                    
                    self.results_changes_performed      # 1) there were relevant updates to bids/results
                    or self.bundle_changes_performed    # 2) incoming bids modified the bundle
                    )
        
        finally:
            # reset replanning flags
            self.results_changes_performed = False
            self.bundle_changes_performed = False

            # reset new event task inbox
            self.incoming_event_tasks = list()

    """
    ---------------------------
    BUNDLE-BUILDING PHASE
    ---------------------------
    """
    def generate_plan(self, 
                      state : SimulationAgentState,
                      specs : object,
                      _ : Plan,
                      clock_config : ClockConfig,
                      orbitdata : OrbitData,
                      mission : Mission,
                      tasks : List[GenericObservationTask],
                      observation_history : ObservationHistory,
                    ) -> Plan:  
        """ Generate new plan according to consensus replanning model. """             

        # DEBUG RETURN------------------- 
        # do not modify plan
        # return current_plan.copy()
        # -------------------------------

        # DEBUG PRINTOUTS----------------
        if self._debug:
            self._log_results('PLANNING PHASE - RESULTS (BEFORE)', state, self.results)
            self._log_bundle('PLANNING PHASE - BUNDLE (BEFORE)', state, self.bundle)
            x = 1 # breakpoint
        # -------------------------------

        # build new bundle and path according to replanning model
        self.bundle, self.path, new_bids = self.bundle_building_phase(state, specs, tasks, orbitdata, mission, observation_history)
        
        # check if new path is valid
        assert len(self.bundle) == len(self.path), "New bundle and path lengths do not match."
        assert self.is_observation_path_valid(state, self.path, None, None, specs), "New observation path is not valid."   

        # update results
        self.__update_results_from_bundle(new_bids)

        # -------------------------------
        # DEBUG PRINTOUTS
        if self._debug:
            self._log_results('PLANNING PHASE - RESULTS (AFTER)', state, self.results)
            self._log_bundle('PLANNING PHASE - BUNDLE (AFTER)', state, self.bundle)
            x = 1 # breakpoint
        # -------------------------------
    
        # generate maneuver and travel actions from observations
        maneuvers : list = self._schedule_maneuvers(state, specs, self.path, clock_config, orbitdata)

        # schedule broadcasts
        broadcasts : list = self._schedule_broadcasts(state, orbitdata)
                
        # compile and generate plan
        self.plan = ReactivePlan(maneuvers, self.path, broadcasts, t=state.t, t_next=self.preplan.t_next)

        # clear new urgent tasks
        self.incoming_event_tasks = list()

        # return final plan
        return self.plan.copy()
    
    @abstractmethod
    def bundle_building_phase(self,
                       state : SimulationAgentState,
                       specs : object,
                       tasks : List[GenericObservationTask],
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
    
    def __update_results_from_bundle(self, 
                                     new_bids : Dict[GenericObservationTask, Dict[int, Bid]]
                                    #  ) -> Dict[GenericObservationTask, List[Bid]]
                                    ) -> None:
        """ Update results dictionary from new bundle. """
        
        # update results with new bids
        for task, bids_dict in new_bids.items():
            for n_obs, bid in bids_dict.items():
                # check if bid is for a new observation number
                if n_obs >= len(self.results[task]):
                    # assume bids are received in order of observation numbers
                    assert len(self.results[task]) == n_obs, \
                          "Generated bids for non-consecutive observation numbers."
                    
                    # add an empty bid for each missing observation number
                    self.results[task].append(bid)

                    # add optimistic bidding counter for new bid
                    self.optimistic_bidding_counters[task].append(self.optimistic_bidding_threshold)
                
                else:
                    # get previous bid
                    previous_bid : Bid = self.results[task][n_obs]

                    # reduce any optimistic bidding counters if needed  
                    if previous_bid > bid and previous_bid.t_img > bid.t_img:
                        # bid was worsened; reduce optimistic bidding counter
                        self.optimistic_bidding_counters[task][n_obs] -= 1

                        # ensure bid counters are possitive
                        assert self.optimistic_bidding_counters[task][n_obs] >= 0, \
                            "Bundle-builder attempted optimistic bid for task with optimistic bid attempt counter already at zero."

                    # update existing bid
                    self.results[task][n_obs] = bid

        # ensure all bids meet requirements
        # 1) previous task must be assigned
        # 2) observation numbers must be consecutive
        # 3) imaging times must be after previous imaging time
        for task, bids in self.results.items():
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
                assert all(constraints), \
                    "Generated bids violate constraints; cannot update results."
        
        # TODO enforce any constraints        

        return        

    def _calculate_path_utility(self,
                                state : SimulationAgentState,
                                specs : object,
                                cross_track_fovs : Dict[str, float],
                                path : List[ObservationAction],
                                observation_history : ObservationHistory,
                                orbitdata : OrbitData,
                                mission : Mission,
                                n_obs : List[Dict[GenericObservationTask, int]],
                                t_prev : List[Dict[GenericObservationTask, float]]
                            ) -> float:
        """ Calculate total expected utility of observation path. """
        
        # calculate path value
        path_value = self._calculate_path_value(specs, cross_track_fovs, path, observation_history, orbitdata, mission, n_obs, t_prev)
        
        # calculate path cost
        path_cost = self._calculate_path_cost(state, specs, path)

        # return path utility
        return path_value - path_cost

    def _calculate_path_value(self,
                              specs : object,
                              cross_track_fovs : Dict[str, float],
                              path : List[ObservationAction],
                              observation_history : ObservationHistory,
                              orbitdata : OrbitData,
                              mission : Mission,
                              n_obs : List[Dict[GenericObservationTask, int]],
                              t_prev : List[Dict[GenericObservationTask, float]]
                            ) -> float:
        """ Calculate total expected value of observation path. """
        # calculate and accumulate expected value of observation
        task_values = self._calculate_path_values(specs, cross_track_fovs, path, observation_history, orbitdata, mission, n_obs, t_prev)

        # return total task value
        return sum(task_values) 

    def _calculate_path_values(self,
                              specs : object,
                              cross_track_fovs : Dict[str, float],
                              path : List[ObservationAction],
                              observation_history : ObservationHistory,
                              orbitdata : OrbitData,
                              mission : Mission,
                              n_obs : List[Dict[GenericObservationTask, int]],
                              t_prev : List[Dict[GenericObservationTask, float]]
                            ) -> List[float]:
        """ Calculate expected value of each observation in the path. """
        return [self.estimate_observation_opportunity_value(obs.obs_opp,
                                                 obs.t_start,
                                                 obs.obs_opp.min_duration,
                                                 specs,
                                                 cross_track_fovs,
                                                 orbitdata,
                                                 mission,
                                                 observation_history,
                                                 n_obs[obs_idx],
                                                 t_prev[obs_idx])
                        for obs_idx, obs in enumerate(path)]

    def _count_observations_and_revisit_times_from_path(self,
                                                        path : List[ObservationAction]
                                                    ) -> Tuple[List[Dict[GenericObservationTask, int]],
                                                            List[Dict[GenericObservationTask, float]]]:
        """ Calculate observation number and revisit time for tasks in the given path given the known bids. """

        # initialize observation counters and previous observation time trackers
        n_obs = [defaultdict(int) for _ in path]
        t_prev = [defaultdict(lambda: np.NINF) for _ in path]

        # get all parent tasks in the given path
        path_tasks : set[GenericObservationTask]= {task 
                                                   for obs_action in path 
                                                   for task in obs_action.obs_opp.tasks}
        
        # ---HISTORICAL DATA FROM BID RESULTS---
        # initiate observation history for all parent tasks in path
        #  only considers performed bids as historical data
        n_obs_history = {task: 0 for task in path_tasks}
        t_prev_history = {task: np.NINF for task in path_tasks}

        # iterate through previous bids to populate initial observation numbers and previous observation times
        for task in path_tasks:
            # assume parent task is part of results
            assert task in self.results, \
                "Parent task in path must be part of results to count observation numbers and revisit times."

            # get previous matching observations for this task
            peformed_bids = [bid for bid in self.results[task]
                                if bid.was_performed()]
            
            assert all(bid.n_obs == idx for idx, bid in enumerate(peformed_bids)), \
                "Results bids are not sorted by observation number."
            
            # update previous observation counts
            n_obs_history[task] += len(peformed_bids)

            # calculate latest observation time from previous bids
            t_latest = max((bid.t_img for bid in peformed_bids), default=np.NINF)
            
            # update previous observation times
            t_prev_history[task] = max(t_prev_history[task], t_latest)

        # ---PATH DATA---
        # initiate observation counter for all parent tasks in path
        n_obs_in_path = {task: 0 for task in path_tasks}
        t_prev_in_path = {task: np.NINF for task in path_tasks}

        # initiate previous observations and times along path
        for obs_idx, obs in enumerate(path):           
            for task in obs.obs_opp.tasks:
                # update overall observation number and revisit times along path using historical and path data
                n_obs[obs_idx][task] = n_obs_history[task] + n_obs_in_path[task]
                t_prev[obs_idx][task] = max(t_prev_history[task], t_prev_in_path[task])               

                # update previous path observation counts 
                n_obs_in_path[task] += 1
                t_prev_in_path[task] = max(t_prev_in_path[task], obs.t_end)

        # return observation numbers and previous observation times
        return n_obs, t_prev
    
    def _count_observations_and_revisit_times_from_results( self,
                                                            state : SimulationAgentState,
                                                            path : List[ObservationAction]
                                                        ) -> Tuple[List[Dict[GenericObservationTask, int]],
                                                            List[Dict[GenericObservationTask, float]]]:
        

        # initialize observation counters and previous observation time trackers
        n_obs = [dict() for _ in path]
        t_prev = [dict() for _ in path]

        # ---HISTORICAL DATA FROM BID RESULTS---
        # iterate through path to populate observation numbers and previous observation times
        for obs_idx, obs_act in enumerate(path):
            for task in obs_act.obs_opp.tasks:
                # assume parent task is part of results
                assert task in self.results, \
                    "Parent task in path must be part of results to count observation numbers and revisit times."

                # get matching bid for this observation task
                matching_bids = [bid for bid in self.results[task]
                                if abs(bid.t_img - obs_act.t_start) <= self.EPS]
                
                assert matching_bids, \
                   "Matching bid for observation in path not found in results. Was assigned without updating results."
                assert len(matching_bids) <= 1, \
                    "There should be at most one matching bid for the current time step."

                matching_bid : Bid = matching_bids.pop()

                # get previous matching observations for this task
                prev_bids = [bid for bid in self.results[task]
                            if bid.t_img < obs_act.t_start]
                
                # update previous observation counts
                n_obs[obs_idx][task] = matching_bid.n_obs
                t_prev[obs_idx][task] = max((bid.t_img for bid in prev_bids), default=np.NINF)

        # ensure every parent task in path has values for `n_obs` and `t_prev`
        assert all(
            all(task in n_obs[obs_idx] and task in t_prev[obs_idx]
                for task in obs_action.obs_opp.tasks)
                for obs_idx,obs_action in enumerate(path)), \
            "Not all observation opportunity tasks in path have an assigned observation number values."
        
        # return observation numbers and previous observation times
        return n_obs, t_prev

    def _calculate_path_cost(self,
                             state : SimulationAgentState,
                             _ : object,
                             path : List[ObservationAction]
                            ) -> float:
        """ Calculate total expected cost of observation path. """

        # TODO implement realistic path cost calculation using agility specs to calculate power consumption between maneuvers.

        # initiate previus observation action with dummy action representing the current state
        prev_obs = None

        # compute total angle change
        total_angle_change = 0.0
        for obs in path:
            # get previous look angle
            prev_angle = state.attitude[0] if prev_obs is None else prev_obs.look_angle
            
            # calculate angle change
            total_angle_change += abs(obs.look_angle - prev_angle)

            # update previous observation
            prev_obs = obs
        
        # compute cost from total angle change
        return self.EPS * total_angle_change  # Placeholder implementation       

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

            # generate bid messages to share bids in results
            compiled_bid_msgs = [
                MeasurementBidMessage(state.agent_name, state.agent_name, bid.to_dict())
                for task,bids in self.results.items()
                if isinstance(task, EventObservationTask)  # only share bids for event-driven tasks
                for bid in bids
            ]
            compiled_results_msg = BusMessage(state.agent_name, 
                                              state.agent_name, 
                                              [bid_msg.to_dict() for bid_msg in compiled_bid_msgs])
            compiled_results_msg_dict = compiled_results_msg.to_dict()
            
            # compile broadcast times for communication opportunities
            t_broadcasts = []

            # compile broadcast times for each communication target
            for target in orbitdata.comms_links.keys():
                # get access intervals with target agent
                access_intervals : List[Interval] = orbitdata.get_next_agent_accesses(target, state.t, include_current=True)

                # create broadcast actions for each access interval
                for next_access in access_intervals:
                    # if no access opportunities in this planning horizon, skip scheduling
                    if next_access.is_empty(): continue

                    # get last access interval and calculate broadcast time
                    t_broadcast : float = max(next_access.left, state.t)

                    # add to list of broadcast times if not already present
                    if all(abs(t_broadcast - t_existing) > self.EPS for t_existing in t_broadcasts):
                        t_broadcasts.append(t_broadcast)

            for t_broadcast in t_broadcasts:
                # generate plan message to share state
                state_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.STATE, t_broadcast)

                # generate plan message to share completed observations
                observations_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.OBSERVATIONS, t_broadcast)

                # generate plan message to share any task requests generated
                task_requests_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.REQUESTS, t_broadcast)
                
                # generate results broadcast action
                bid_msg_action = BroadcastMessageAction(compiled_results_msg_dict, t_broadcast)
                
                # add to client broadcast list
                # TODO decide whether to broadcast state and observations as well
                broadcasts.extend([
                                #    state_msg, 
                                #    observations_msg, 
                                    task_requests_msg, 
                                    bid_msg_action
                                    ])

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
    def _log_results(self, 
                     dsc : str, 
                     state : SimulationAgentState, 
                     results : Dict[GenericObservationTask, List[Bid]],
                     level=logging.DEBUG, 
                     n : int = 50) -> None:
        out = f'\nT{np.round(state.t,3)}[s]:\t\'{state.agent_name}\'\n{dsc}\n'
        line = 'Task ID\t n_obs\tins\t\twinner\tbid\tt_img\tt_stamp\tv_opt\tperformed\n'
        
        # count characters in line for formatting
        L_LINE = len(line)
        L_LINE_PADding = 20

        # header
        out += line 

        # divider 
        for _ in range(L_LINE + L_LINE_PADding): out += '='
        out += '\n'

        i = 1
        for task, bids in results.items():
            task : GenericObservationTask

            if isinstance(task, EventObservationTask):
                req_id_short = task.id.split('-')[-1]
            else:
                req_id_short = f'Default({int(task.location[0][-2])},{int(task.location[0][-1])})'

            # if all([bid.winner == bid.NONE for _,bid in bids.items()]): continue

            if not bids:
                out += f'{req_id_short} <none>\n'

            if isinstance(bids, dict):
                printed_bids = sorted(bids.values(), key=lambda b: b.n_obs)
            elif isinstance(bids, list):
                printed_bids = bids
            else:
                raise TypeError("Bids in results must be either a list or a dictionary.")   

            for bid in printed_bids:
                # if i > n: break

                bid : Bid
                # if bid.winner == bid.NONE: continue

                if bid.winner != bid.NONE:
                    line = f'{req_id_short} {bid.n_obs}\t{bid.main_measurement}\t{bid.winner[0].lower()}{bid.winner[-1]}\t{np.round(bid.winning_bid,4)}\t{np.round(bid.t_img,1)}\t{np.round(bid.t_bid,1)}\t{self.optimistic_bidding_counters[bid.task][bid.n_obs]}\t{(bid.performed)}\n'
                else:
                    line = f'{req_id_short} {bid.n_obs}\t{bid.main_measurement}\tn/a\t{np.round(bid.winning_bid,4)}\t{np.round(bid.t_img,1)}\t{np.round(bid.t_bid,1)}\t{self.optimistic_bidding_counters[bid.task][bid.n_obs]}\t{(bid.performed)}\n'
                out += line
                i +=1

            for _ in range(L_LINE + L_LINE_PADding):
                out += '-'
            out += '\n'

            if i > n:
                out += '\t\t\t...\n'
                for _ in range(L_LINE + L_LINE_PADding):
                    out += '-'
                out += '\n'
                break

        print(out)

    def _log_path(self, 
                  dsc : str, 
                  state : SimulationAgentState, 
                  observation_path : List[ObservationAction], 
                  level=logging.DEBUG) -> None:
        out = f'\nT{np.round(state.t,3)}[s]:\t\'{state.agent_name}\'\n{dsc}\n'
        line = 'i\tt_img\t Task IDs\n'
        
        # count characters in line for formatting
        L_LINE = len(line)
        L_LINE_PADding = 20

        # header
        out += line 

        # divider 
        for _ in range(L_LINE + L_LINE_PADding): out += '='
        out += '\n'

        if not observation_path:
            out += '\t<empty path>\n'
            for _ in range(L_LINE + L_LINE_PADding): out += '-'
            out += '\n'

        n = 15
        for i,obs in enumerate(observation_path):
            spec_task : ObservationOpportunity = obs.obs_opp
            req_id_short = ""

            for task in spec_task.tasks:
                if isinstance(spec_task, EventObservationTask):
                    req_id_short += spec_task.id.split('-')[-1] + ","
                else:
                    req_id_short += f'Default({int(task.location[0][-2])},{int(task.location[0][-1])}),'

            line = f'{i}\t{np.round(obs.t_start,1)}\t[{req_id_short[:-1]}]\n'
            out += line

            for _ in range(L_LINE + L_LINE_PADding):
                out += '-'
            out += '\n'

            if i > n:
                out += '\t\t\t...\n'
                for _ in range(L_LINE + L_LINE_PADding):
                    out += '-'
                out += '\n'
                break

        print(out)

    def _log_bundle(self, 
                    dsc : str, 
                    state : SimulationAgentState, 
                    bundle : List[Tuple[GenericObservationTask, Dict[GenericObservationTask, int]]], 
                    level=logging.DEBUG) -> None:
        out = f'\nT{np.round(state.t,3)}[s]:\t\'{state.agent_name}\'\n{dsc}\n'
        line = 'i\t Task IDs\n'
        
        # count characters in line for formatting
        L_LINE = len(line)
        L_LINE_PADding = 20

        # header
        out += line 

        # divider 
        for _ in range(L_LINE + L_LINE_PADding): out += '='
        out += '\n'

        if not bundle:
            out += '\t<empty bundle>\n'
            for _ in range(L_LINE + L_LINE_PADding): out += '-'
            out += '\n'

        n = 15
        for i,(_,tasks) in enumerate(bundle):
            line = f'{i}\t['
            for task,n_obs in tasks.items():
                # if i > n: break

                if isinstance(task, EventObservationTask):
                    req_id_short = task.id.split('-')[-1]
                else:
                    req_id_short = f'Default({int(task.location[0][-2])},{int(task.location[0][-1])})'

                line += f'({req_id_short},{n_obs}),'
            line = line[:-1] + ']\n'
            out += line

            for _ in range(L_LINE + L_LINE_PADding):
                out += '-'
            out += '\n'

            if i > n:
                out += '\t\t\t...\n'
                for _ in range(L_LINE + L_LINE_PADding):
                    out += '-'
                out += '\n'
                break

        print(out)