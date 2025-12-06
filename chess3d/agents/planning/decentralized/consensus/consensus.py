from collections import defaultdict
from itertools import product
from typing import Dict, List, Tuple
from tqdm import tqdm

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
from chess3d.agents.planning.decentralized.consensus.bids import AsynchronousBid, Bid
from chess3d.agents.science.reward import *
from chess3d.messages import BusMessage, MeasurementBidMessage
from chess3d.mission.mission import Mission
from chess3d.agents.states import SatelliteAgentState, SimulationAgentState
from chess3d.orbitdata import OrbitData
from chess3d.utils import Interval

class ConsensusReplanner(AbstractReactivePlanner):    
    # Replanning models
    # NOTE break down into separate replanner classes that use this class as the parent class?
    HEURISTIC_INSERTION = 'heuristicInsertion'
    DYNAMIC_PROGRAMMING = 'dynamicProgramming'
    MILP = 'mixedIntegerLinearProgramming'
    MODELS = [HEURISTIC_INSERTION, DYNAMIC_PROGRAMMING, MILP]
    
    # Heuristic types for insertion model
    EARLIEST_ACCESS = 'earliestAccess'
    TASK_VALUE = 'taskValue'
    HEURISTICS = [EARLIEST_ACCESS, TASK_VALUE]

    # Constants
    EPS = 1e-6

    def __init__(self, 
                 model : str = HEURISTIC_INSERTION,
                 heuristic : str = EARLIEST_ACCESS,
                 replan_threshold : int = 1,
                 debug : bool = False,
                 logger: logging.Logger = None
                 ) -> None:
        super().__init__(debug, logger)

        # validate inputs
        assert model in self.MODELS, f"Invalid model '{model}'. Must be one of {self.MODELS}."
        assert heuristic in self.HEURISTICS, f"Invalid heuristic '{heuristic}'. Must be one of {self.HEURISTICS}."
        assert isinstance(replan_threshold, int) and replan_threshold > 0, "Replan threshold must be positive integer."

        # initialize results
        self.bundle : list[list[Bid]] = list()
        self.path : list[GenericObservationTask] = list()
        self.results : Dict[GenericObservationTask, List[Bid]] = defaultdict(list)
        self.preplan : PeriodicPlan = None
        self.plan : Plan = None
        self.known_urgent_tasks : set[GenericObservationTask] = set()
        self.new_urgent_tasks : set[GenericObservationTask] = set()
        self.relevant_updates : List[Bid] = list()
        self.bid_inbox : list[Bid] = list()
        self.bid_outbox : Dict[GenericObservationTask, Dict[int,Bid]] = defaultdict(dict)

        # set parameters
        self.model = model
        self.heuristic = heuristic
        self.replan_threshold = replan_threshold
        self.t_share = -1

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

        # TODO collect pending and performed task request announcement broadcasts 
        
        # convert incoming task requests to bids and add to inbox
        self.__generate_bids_from_reqs(state, incoming_reqs)

        # collect bids from incoming messages to inbox
        self.__collect_incoming_bids(misc_messages)  

        return # Placeholder implementation
        # raise NotImplementedError("Consensus replanner not yet implemented.")

    def __update_preplan(self, state : SimulationAgentState, current_plan : Plan) -> None:
        """ Update latest preplan if new plan is available. """
        if isinstance(current_plan, PeriodicPlan) and abs(state.t - current_plan.t) <= self.EPS:
            self.preplan : PeriodicPlan = current_plan.copy()

    def __update_urgent_tasks(self, state : SimulationAgentState, incoming_reqs : List[TaskRequest]) -> None:
        """ Remove completed tasks from urgent tasks set. """
        # remove unavailable tasks
        self.known_urgent_tasks = set([task for task in self.known_urgent_tasks 
                                        if task.is_available(state.t)])
        self.new_urgent_tasks = set([task for task in self.new_urgent_tasks 
                                        if task.is_available(state.t)])

        # get active incoming tasks
        active_tasks = set([req.task for req in incoming_reqs 
                            if req.task.is_available(state.t)])
        
        # update urgent tasks
        self.known_urgent_tasks.update(active_tasks)
        self.new_urgent_tasks.update(active_tasks)

    def __collect_incoming_bids(self, misc_messages : List[SimulationMessage]) -> None:
        """ Collect bids from incoming messages and requests. """
        incoming_bids = [AsynchronousBid.from_dict(msg.bid) 
                            for msg in misc_messages 
                            if isinstance(msg, MeasurementBidMessage)]
        
        if incoming_bids: 
            x = 1 # Placeholder implementation

        self.bid_inbox.extend(incoming_bids)

    def __generate_bids_from_reqs(self, state : SimulationAgentState, incoming_reqs : List[TaskRequest]) -> None:
        """ Generate bids from incoming task requests. """
        # extract bids from incoming requests
        # TODO make this an abstract method that can be overridden by subclasses that distinguish between
        # synchronous and asynchronous bidding strategies
        bids_from_reqs = [AsynchronousBid(req.task, state.agent_name) for req in incoming_reqs]
        
        if bids_from_reqs: 
            x = 1 # Placeholder implementation

        # update bid inbox
        self.bid_inbox.extend(bids_from_reqs)

    def needs_planning(self, 
                       state : SimulationAgentState,
                       specs : object,
                       current_plan : Plan,
                       orbitData : OrbitData
                    ) -> bool:
        # -------------------------------
        # DEBUG PRINTOUTS
        if self.bid_inbox:
            self.log_results('CONSENSUS PHASE (BEFORE)', state, self.results)
            self.log_bundle('BUNDLE (BEFORE CONSENSUS)', state, self.bundle)
        # -------------------------------

        # perform consensus phase for incoming bids and tasks
        changes, rebroadcasts = self.consensus_phase(state, specs, current_plan, orbitData)
        
        # update relevant updates
        self.relevant_updates.extend(rebroadcasts)

        # -------------------------------
        # DEBUG PRINTOUTS
        if self.relevant_updates:
            self.log_results('CONSENSUS PHASE (AFTER)', state, self.results)
            self.log_bundle('BUNDLE (AFTER CONSENSUS)', state, self.bundle)
        # -------------------------------

        # replan if...
        # 1) there were relevant updates to bids/results
        relevant_changes_received = len(self.relevant_updates) > 0
        # 2) or new periodic plan was received
        new_periodic_plan_received = isinstance(current_plan, PeriodicPlan) and abs(state.t - current_plan.t) <= self.EPS
        # 3) or new urgent tasks exceed threshold
        task_threshold_met = len(self.new_urgent_tasks) >= self.replan_threshold
        
        # -------------------------------
        # DEBUG BREAKPOINTS
        if relevant_changes_received:
            x = 1  # Placeholder implementation
        if new_periodic_plan_received:
            x = 1  # Placeholder implementation
        if task_threshold_met:
            x = 1  # Placeholder implementation
        # -------------------------------

        return (relevant_changes_received 
                # or new_periodic_plan_received 
                or task_threshold_met)

    def consensus_phase(self,
                        state : SimulationAgentState,
                        specs : object,
                        current_plan : Plan,
                        orbitdata : OrbitData
                    ) -> None:
        """ Perform consensus phase to update bids and bundle. """
        # check if tasks were performed
        # self.check_bid_completion()

        # check if tasks expired
        # self.check_task_end_time()

        # compare results with incoming bids and update bundle
        bid_changes, bid_rebroadcasts = self.update_results(state)

        # clear bid inbox
        self.bid_inbox = list()

        # compile changes and rebroadcasts
        changes = []
        changes.extend(bid_changes)

        rebroadcasts = []
        rebroadcasts.extend(bid_rebroadcasts)
        
        return changes, rebroadcasts

    def update_results(self,
                       state : SimulationAgentState,
                       ) -> Tuple[List[Bid], List[Bid]]:
        """ Update results from incoming bids. """
        # initialize bundle changes and rebroadcast lists
        outbid = []         # bids from results that were outbid 
        changes = []        # updated and modified bids
        rebroadcasts = []   # bids to be rebroadcast

        # process incoming bids
        for incoming_bid in self.bid_inbox:
            # check bids are for new requests
            if incoming_bid.task not in self.results:
                # add empty bid list for new task
                for n_obs in range(incoming_bid.n_obs+1):
                    empty_bid = AsynchronousBid(incoming_bid.task, state.agent_name, n_obs=n_obs)
                    self.results[incoming_bid.task].append(empty_bid)

            # compare incoming bid with existing bids for the same task
            current_bid : Bid = self.results[incoming_bid.task][incoming_bid.n_obs]

            _, rebroadcast_result = current_bid.rule_comparison(incoming_bid)
            updated_bid : Bid = current_bid.compare(incoming_bid, state.t)
            bid_changed = current_bid != updated_bid

            # update results with modified bid
            self.results[incoming_bid.task][incoming_bid.n_obs] = updated_bid

            # if bid was changed, add to changes list
            if bid_changed: 
                outbid.append(current_bid)
                changes.append(updated_bid)

            # if relevant changes were made, add appropriate bid to rebroadcast list
            if (rebroadcast_result == Bid.REBROADCAST_SELF
                or rebroadcast_result == Bid.REBROADCAST_SELF):
                rebroadcasts.append(updated_bid)
            elif rebroadcast_result == Bid.REBROADCAST_OTHER:
                rebroadcasts.append(updated_bid)

        # check if any bids in the bundle were modified
        if self.bundle and outbid: 
            raise NotImplementedError("Bundle update after bid comparison not yet implemented.")
        # for current_bid in changes:
        #     # search for outbids in bundle
        #     outbid_indices = [bid_idx for bid_idx,bundle_bids in enumerate(self.bundle)
        #                       if current_bid in bundle_bids                         # bid is in bundle
        #                       and current_bid.winning_bidder != state.agent_name]   # was outbid
        #     outbid_index = min(outbid_indices) if len(outbid_indices) > 0 else None
            
        #     if outbid_index is None: continue # no bid in bundle was outbid; continue to next incoming bid
            
        #     # if bid in bundle was outbid, remove all subsequent bids from bundle
        #     for bundle_index in range(outbid_index, len(self.bundle)):
        #         # remove bid from bundle
        #         bundle_bids = self.bundle.pop(bundle_index)

        #         # reset results for removed bids
        #         for bundle_bid in bundle_bids:
        #             # if already outbid, skip
        #             if bundle_bid in outbid: continue

        #             # reset bid
        #             bundle_bid.reset(state.t)

        #             # update results
        #             self.results[bundle_bid.task][bundle_bid.n_obs] = bundle_bid

        #             # add to changes and rebroadcast lists
        #             changes.append(bundle_bid)
        #             rebroadcasts.append(bundle_bid)
        
        # return result changes and bids to rebroadcasts
        return changes, rebroadcasts

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
        self.log_results('PLANNING PHASE (BEFORE)', state, self.results)
        self.log_bundle('BUNDLE (BEFORE PLANNING)', state, self.bundle)
        # -------------------------------

        # build new bundle
        new_bundle, new_path = self.bundle_building_phase(state, specs, current_plan, clock_config, orbitdata, mission, observation_history)
        
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
        self.log_results('PLANNING PHASE', state, self.results)
        self.log_bundle('BUNDLE (AFTER PLANNING)', state, self.bundle)
        # -------------------------------
    
        # generate maneuver and travel actions from observations
        maneuvers : list = self._schedule_maneuvers(state, specs, new_path, clock_config, orbitdata)

        # schedule broadcasts
        # TODO decide on broadcast scheduling strategy
        broadcasts : list = self._schedule_broadcasts(state, orbitdata)
                
        # compile and generate plan
        self.plan = ReactivePlan(maneuvers, new_path, broadcasts, t=state.t, t_next=self.preplan.t_next)

        # clear new urgent tasks
        self.new_urgent_tasks = set()

        # TEMP clear bid outbox 
        # TODO only clear bids that were successfully broadcasted?
        # self.bid_outbox = defaultdict(dict)

        # return final plan
        return self.plan.copy()
    
    def bundle_building_phase(self,
                       state : SimulationAgentState,
                       specs : object,
                       current_plan : Plan,
                       clock_config : ClockConfig,
                       orbitdata : OrbitData,
                       mission : Mission,
                       observation_history : ObservationHistory
                    ) -> tuple:
        
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self._collect_fov_specs(specs)

        # Outline planning horizon interval
        planning_horizon = Interval(state.t, self.preplan.t_next)

        # get only available tasks from existing plan and urgent tasks
        available_tasks : list[GenericObservationTask] = self.get_available_tasks(planning_horizon)
        
        # calculate coverage opportunities for available tasks
        access_opportunities : dict[tuple] = self.calculate_access_opportunities(state, planning_horizon, orbitdata)

        # create specific and merged tasks from scheduled tasks and urgent tasks
        schedulable_tasks : List[SpecificObservationTask] = self.create_tasks_from_accesses(available_tasks, access_opportunities, cross_track_fovs, orbitdata)

        # filter for only schedulable tasks with urgent parent tasks        
        schedulable_urgent_tasks : List[SpecificObservationTask] = [task for task in schedulable_tasks 
                                                                    if any(parent_task in self.known_urgent_tasks 
                                                                            for parent_task in task.parent_tasks)]

        # generate new plan according to selected model
        if self.model == self.HEURISTIC_INSERTION:
            if self.heuristic == self.EARLIEST_ACCESS:
                return self.earliest_access_heuristic_bundle_builder(state, specs, cross_track_fovs, current_plan, schedulable_urgent_tasks, orbitdata, mission, observation_history)
            
            elif self.heuristic == self.TASK_VALUE:
                return self.task_value_heuristic_bundle_builder(state, specs, cross_track_fovs, current_plan, schedulable_urgent_tasks, orbitdata, mission, observation_history)
            
        elif self.model == self.DYNAMIC_PROGRAMMING:
            raise NotImplementedError("Dynamic-programming consensus planner not yet implemented.")
        
        elif self.model == self.MILP:
            raise NotImplementedError("Mixed-integer-linear-programming consensus planner not yet implemented.")
        
        else:
            raise NotImplementedError(f"Model '{self.model}' not implemented.")            

    def get_available_tasks(self, planning_horizon : Interval) -> list:
        """ Get only tasks that are available within the planning horizon. """
        # get tasks present in current preplan
        planned_tasks = {parent_task
                         for action in self.preplan.actions 
                         if isinstance(action, ObservationAction)
                         for parent_task in action.task.parent_tasks
                         }

        # get urgent tasks that are available within planning horizon
        urgent_tasks = {task 
                           for task in self.known_urgent_tasks 
                            if task.availability.overlaps(planning_horizon)}
        
        # merge task sets
        available_tasks = {task for task in urgent_tasks}
        available_tasks.update(planned_tasks)

        # return tasks as a merged list
        return list(available_tasks)

    @runtime_tracker
    def calculate_access_opportunities(self, 
                                       state : SimulationAgentState, 
                                       planning_horizon : Interval,
                                       orbitdata : OrbitData
                                    ) -> dict:
        """ Calculate access opportunities for targets visible in the planning horizon """

        # check planning horizon span
        if planning_horizon.is_empty(): 
            return {}

        # compile coverage data
        raw_coverage_data : dict = orbitdata.gp_access_data.lookup_interval(planning_horizon.left, planning_horizon.right)

        # initiate access times
        access_opportunities = {}
        
        for i in tqdm(range(len(raw_coverage_data['time [s]'])), 
                        desc=f'{state.agent_name}/PREPLANNER: Compiling access opportunities', 
                        leave=False):
            t_img = raw_coverage_data['time [s]'][i]
            grid_index = raw_coverage_data['grid index'][i]
            gp_index = raw_coverage_data['GP index'][i]
            instrument = raw_coverage_data['instrument'][i]
            look_angle = raw_coverage_data['look angle [deg]'][i]
            
            # initialize dictionaries if needed
            if grid_index not in access_opportunities:
                access_opportunities[grid_index] = {}
                
            if gp_index not in access_opportunities[grid_index]:
                access_opportunities[grid_index][gp_index] = defaultdict(list)

            # compile time interval information 
            found = False
            for interval, t, th in access_opportunities[grid_index][gp_index][instrument]:
                interval : Interval
                t : list
                th : list

                overlap_interval = Interval(t_img - orbitdata.time_step, 
                                            t_img + orbitdata.time_step)
                
                if overlap_interval.overlaps(interval):
                    interval.extend(t_img)
                    t.append(t_img)
                    th.append(look_angle)
                    found = True
                    break      

            if not found:
                access_opportunities[grid_index][gp_index][instrument].append([Interval(t_img, t_img), [t_img], [look_angle]])
                
        # return access times and grid information
        return access_opportunities

    def earliest_access_heuristic_bundle_builder(self,
                                       state : SimulationAgentState,
                                       specs : object,
                                       cross_track_fovs : dict,
                                       current_plan : Plan,
                                       schedulable_urgent_tasks : List[SpecificObservationTask],
                                       orbitdata : OrbitData,
                                       mission : Mission,
                                       observation_history : ObservationHistory
                                    ) -> Tuple[list, list]:
        """ 
        Build bundle using earliest-access heuristic. 

        #### Returns
        - bundle : List[Tuple[GenericObservationTask, int, SpecificObservationTask, Bid]]
            List of tuples containing (task, observation number, observation time, expected utility).
        - path : List[ObservationAction]
            Updated observation path after bundle building.
        """ 
        # sort urgent tasks by earliest access time
        sorted_schedulable_urgent_tasks = sorted(schedulable_urgent_tasks, key=lambda task: task.accessibility.left)
    
        # build bundle using heuristic insertion method
        return self.__heuristic_insertion_bundle_builder(state, specs, cross_track_fovs, current_plan, sorted_schedulable_urgent_tasks, orbitdata, mission, observation_history, heuristic_evaluator=lambda task: task.accessibility.left)

    def task_value_heuristic_bundle_builder(self,
                                       state : SimulationAgentState,
                                       specs : object,
                                       cross_track_fovs : dict,
                                       current_plan : Plan,
                                       schedulable_urgent_tasks : List[SpecificObservationTask],
                                       orbitdata : OrbitData,
                                       mission : Mission,
                                       observation_history : ObservationHistory
                                    ) -> Tuple[list, list]:
        """ 
        Build bundle using task-value as main heuristic for order of task addition in bundle building process. 
         Considers the value of performing the task in isolation. Does not take into account any possible changes 
         in value due to in-schedule interactions.

        #### Returns
        - bundle : List[Tuple[GenericObservationTask, int, SpecificObservationTask, Bid]]
            List of tuples containing (task, observation number, observation time, expected utility).
        - path : List[ObservationAction]
            Updated observation path after bundle building.
        """ 
        # sort urgent tasks by expected task value
        task_values = [(task, self.estimate_specific_task_value(task,
                                                               task.accessibility.left,
                                                               task.min_duration,
                                                               specs,
                                                               cross_track_fovs,
                                                               orbitdata,
                                                               mission,
                                                               observation_history)) for task in schedulable_urgent_tasks]
        sorted_schedulable_urgent_tasks = [task for task, _ in sorted(task_values, key=lambda item: item[1], reverse=True)]
    
        # build bundle using heuristic insertion method
        return self.__heuristic_insertion_bundle_builder(state, specs, cross_track_fovs, current_plan, sorted_schedulable_urgent_tasks, orbitdata, mission, observation_history)

    def _is_task_mutually_exclusive_with_path(self, task : SpecificObservationTask, path : List[ObservationAction]):
        """ Check if task is mutually exclusive with any observations in the given path. """
        return any([task.is_mutually_exclusive(action.task) for action in path])

    def __heuristic_insertion_bundle_builder(self,
                                       state : SimulationAgentState,
                                       specs : object,
                                       cross_track_fovs : dict,
                                       current_plan : Plan,
                                       sorted_schedulable_urgent_tasks : List[SpecificObservationTask],
                                       orbitdata : OrbitData,
                                       mission : Mission,
                                       observation_history : ObservationHistory
                                    ) -> Tuple[list, list]:
        """ 
        Build bundle using a given heuristic. Attempts to insert tasks into existing path, right-shift existing tasks to accommodate for new 
         tasks or replaces tasks in the current plan if it leads to a feasible plan that can increase overall plan utility.  Tasks are added 
         according to heuristic evaluator. 

        #### Returns
        - bundle : List[Tuple[GenericObservationTask, int, SpecificObservationTask, Bid]]
            List of tuples containing (task, observation number, observation time, expected utility).
        - path : List[ObservationAction]
            Updated observation path after bundle building.        
        """

        # initialized bundle
        if isinstance(current_plan, PeriodicPlan) and abs(state.t - current_plan.t) <= self.EPS:
            raise NotImplementedError("Earliest-access bundle builder initializing for new preplans not yet implemented.")
        else:
            bundle : List[Tuple[GenericObservationTask, int, SpecificObservationTask, Bid]] = \
                [task_tuple for task_tuple in self.bundle]
            path = sorted([action for action in current_plan
                                if isinstance(action, ObservationAction)], 
                                key=lambda action: action.t_start)

        # iterate through urgent tasks and attempt to add to bundle
        for new_task in tqdm(sorted_schedulable_urgent_tasks, desc=f'{state.agent_name}-REPLANNER: Building bundle', leave=False):
            # Find best placement in path   
            proposed_path, t_img = self.__heuristic_insertion_path_builder(state, specs, path, new_task)
                
            # if no feasible path was found, ignore new urgent task
            if proposed_path is None: continue
            
            if len(new_task.parent_tasks) > 1:
                x = 1  # Placeholder implementation

            # create bids for relevant parent tasks
            new_bids : List[Bid] = self._generate_bids_for_task_in_path(state, specs, path, proposed_path, new_task, 
                                                                        t_img, cross_track_fovs, orbitdata, mission, observation_history)
                
            # if no new bids were generated, skip to next urgent task
            if not new_bids: continue

            # add bids to bundle
            bundle.append(new_bids)

            # update current path                    
            path = [action for action in proposed_path]

        return bundle, path
    
    def __heuristic_insertion_path_builder(self,
                                            state : SimulationAgentState,
                                            specs : object,
                                            current_path : List[ObservationAction],
                                            new_task : SpecificObservationTask
                                        ) -> Tuple[List[ObservationAction], float]:
        # compile agility specifications
        max_slew_rate, max_torque = self._collect_agility_specs(specs)

        # Option 1: Direct Insertion into existing path
        proposed_path, t_img = self._direct_insertion_into_path(state, specs, current_path, new_task, max_slew_rate, max_torque)

        # Option 2: Right-shifting existing path to accommodate new task
        if proposed_path is None:
            proposed_path, t_img = self._right_shift_path_for_new_task(state, specs, current_path, new_task, max_slew_rate, max_torque)
        
        # Option 3: Replace conflicting task with new urgent task
        if proposed_path is None:
            proposed_path, t_img = self._replace_conflicting_tasks_with_new_task(state, specs, current_path, new_task, max_slew_rate, max_torque)

        # TODO Option 4: Remove all conflicting tasks and insert new task

        # return proposed path and observation time
        return proposed_path, t_img
        
    
    def _generate_bids_for_task_in_path(self,
                                        state : SimulationAgentState,
                                        specs : object,
                                        current_path : List[ObservationAction],
                                        proposed_path : List[ObservationAction],
                                        task_to_schedule : SpecificObservationTask,
                                        t_img : float,
                                        cross_track_fovs : dict,
                                        orbitdata : OrbitData,
                                        mission : Mission,
                                        observation_history : ObservationHistory
                                    ) -> List[Bid]:
        """ Generate bid for given task in the context of the given path. 
        
        ### Returns 
            - bids : List[Tuple[GenericObservationTask, int]]] - List of bids for each parent task of the given task. Is None if no valid bids could be generated.
        """

        # calculate path utility without the new task
        old_path_utility : float = self._calculate_path_utility(state, specs, cross_track_fovs, current_path, observation_history, orbitdata, mission)

        # get relevant parent tasks from task being scheduled
        parent_tasks = [parent_task 
                        for parent_task in task_to_schedule.parent_tasks
                        if parent_task in self.known_urgent_tasks]

        # get bounds for min and maximum observation numbers for each parent task
        n_obs_per_task = {parent_task : list(range(len(self.results[parent_task])+(1 if self.results[parent_task][-1].has_winner() else 0)))
                            for parent_task in parent_tasks}

        # initiate possible observation number and revisit pair tracker for each parent task
        n_obs_revisit_pairs = defaultdict(list)
                
        # calculate revisit times for each possible observation number
        for parent_task, n_obs_list in n_obs_per_task.items():
            for n_obs in n_obs_list:             
                # check if observation has already been performed
                if self.results[parent_task][n_obs].was_performed(): 
                    continue # observation already performed; skip

                # check if agent is already scheduled to perform observation
                if self.results[parent_task][n_obs].winning_bidder == state.agent_name: 
                    continue # observation already scheduled by this agent; skip

                # calculate previous observation time
                t_prev = self.results[parent_task][n_obs-1].t_img if n_obs > 0 else np.NINF

                # check if previous observation time is prior to the chosen observation time 
                if t_img < t_prev: 
                    continue # incompatible observation time and observation number; skip

                # estimate revisit time
                t_revisit = t_img - t_prev if n_obs > 0 else np.NINF

                # add valid (n_obs, t_revisit) pair to list
                n_obs_revisit_pairs[parent_task].append((n_obs, t_revisit))

        assert all([parent_task in n_obs_revisit_pairs for parent_task in parent_tasks]), \
            "No valid (n_obs, t_revisit) pairs could be generated for all parent tasks."
        
        # enlist all possible (n_obs, t_revisit) options for each parent task
        options_lists = [n_obs_revisit_pairs[parent_task] for parent_task in parent_tasks]
        
        # initiate search for best (n_obs, t_revisit) combination
        best_combo : dict = dict()
        best_val : float = np.NINF

        # calculate bid for each parent task and each possible (n_obs, t_revisit) pair
        for combo in product(*options_lists):
            n_obs_bid = {parent_task : n_obs for parent_task, (n_obs, _) in zip(parent_tasks, combo)}
            t_prev_bid = {parent_task : t_prev for parent_task, (_, t_prev) in zip(parent_tasks, combo)}
            
            # calculate new path utility with proposed (n_obs, t_revisit) pairs
            new_path_utility : float = self._calculate_path_utility(state, specs, cross_track_fovs, proposed_path, observation_history, orbitdata, mission, task_to_schedule, n_obs_bid, t_prev_bid)

            # calculate bid value
            bid_value : float = new_path_utility - old_path_utility

            # calculate total subsequent bid values for competing bids
            subsequent_bid_values = {parent_task : sum([subsequent_bid.winning_bid 
                                            for subsequent_bid in self.results[parent_task][n_obs+1:]]) \
                                            if n_obs < len(self.results[parent_task]) else 0.0
                                    for parent_task in parent_tasks}
            
            # check if outbids all subsequent bids
            if any(bid_value <= subsequent_bid_value + self.EPS
                   for subsequent_bid_value in subsequent_bid_values.values()): 
                continue # does not outbid all subsequent bids; skip
            
            # check if bid value is best so far
            if bid_value > best_val + self.EPS:
                best_val = bid_value
                best_combo = {parent_task: (n_obs, t_prev) 
                              for parent_task, (n_obs, t_prev) in zip(parent_tasks, combo)}
        
        
        # create and return bids for each parent task based on best (n_obs, t_revisit) combination
        return [AsynchronousBid(parent_task, state.agent_name, n_obs, best_val, state.agent_name, 
                                 best_val, t_img, state.t, task_to_schedule.instrument_name) 
                for parent_task, (n_obs,_) in best_combo.items()]
        
    def _calculate_path_utility(self,
                                state : SimulationAgentState,
                                specs : object,
                                cross_track_fovs : Dict[str, float],
                                path : List[ObservationAction],
                                observation_history : ObservationHistory,
                                orbitdata : OrbitData,
                                mission : Mission,
                                task_to_schedule : SpecificObservationTask = None,
                                n_obs_bid : Dict[GenericObservationTask, int] = None,
                                t_prev_bid : Dict[GenericObservationTask, float] = None
                            ) -> float:
        """ Calculate total expected utility of observation path. """
        
        # validate input arguments
        if any(param is not None for param in [task_to_schedule, n_obs_bid, t_prev_bid]):
            assert all(param is not None for param in [task_to_schedule, n_obs_bid, t_prev_bid]), \
                "Proposed bid parameters must be provided if either is specified."
            
            assert all(parent_task in task_to_schedule.parent_tasks for parent_task in n_obs_bid.keys()), \
                "Proposed bid observation numbers contain parent tasks not associated with the task being scheduled."
            
            assert all(parent_task in task_to_schedule.parent_tasks for parent_task in t_prev_bid.keys()), \
                "Proposed bid previous observation times contain parent tasks not associated with the task being scheduled."

        # calculate path value
        path_value = self._calculate_path_value(state, specs, cross_track_fovs, path, observation_history, orbitdata, mission, task_to_schedule, n_obs_bid, t_prev_bid)
        
        # calculate path cost
        path_cost = self._calculate_path_cost(state, specs, path)

        # return path utility
        return path_value - path_cost

    def _calculate_path_value(self,
                              state : SimulationAgentState,
                              specs : object,
                              cross_track_fovs : Dict[str, float],
                              path : List[ObservationAction],
                              observation_history : ObservationHistory,
                              orbitdata : OrbitData,
                              mission : Mission,
                              task_to_schedule : SpecificObservationTask,
                              n_obs_bid : Dict[GenericObservationTask, int],
                              t_prev_bid : Dict[GenericObservationTask, float]
                            ) -> float:
        """ Calculate total expected value of observation path. """
        # initialize path value
        total_value = 0.0

        # calculate observation number and revisit time for tasks in path
        n_obs, t_prev = self._calculate_observation_number_and_revisit_in_path(path, observation_history)
     
        # replace observation number and revisit time values for task being scheduled if provided
        if any(param is not None for param in [task_to_schedule, n_obs_bid, t_prev_bid]):            
            # find index and observation action for task being scheduled
            matching_index,matching_obs = min([(obs_idx, obs) for obs_idx, obs in enumerate(path) 
                                                if obs.task == task_to_schedule],
                                                key=lambda item: item[0])
            
            # update observation number and previous observation time for task being scheduled
            for parent_task in n_obs_bid.keys():
                # get previous bids for this task
                previous_bids = [bid for bid in self.results[parent_task]
                                 if bid.t_img < matching_obs.t_start 
                                 and bid.winning_bidder == state.agent_name]
                latest_bid = max(previous_bids, key=lambda bid: bid.t_img, default=None)

                # check if overwrite values are valid
                assert n_obs_bid[parent_task] >= len(previous_bids), \
                    f"Proposed observation number {n_obs_bid[parent_task]} for task '{parent_task}' is less than the number of previous bids {len(previous_bids)} for the same task by this agent."
                assert t_prev_bid[parent_task] >= (latest_bid.t_img if latest_bid else np.NINF), \
                    f"Proposed previous observation time {t_prev_bid[parent_task]} [s] for task '{parent_task}' is earlier than the latest previous bid time {latest_bid.t_img if latest_bid else 'NINF'} for the same task by this agent."

                # overwrite observation number and previous observation time
                n_obs[matching_index][parent_task] = n_obs_bid[parent_task]
                t_prev[matching_index][parent_task] = t_prev_bid[parent_task]

            # update observation number and previous observation time for task being scheduled
            for parent_task in n_obs_bid.keys():
                n_obs[matching_index][parent_task] = n_obs_bid[parent_task]
                t_prev[matching_index][parent_task] = t_prev_bid[parent_task]

        # iterate through observations in path
        for obs_idx, obs in enumerate(path):
            # calculate expected value of observation
            obs_value = self.estimate_specific_task_value(obs.task,
                                                 obs.t_start,
                                                 obs.task.min_duration,
                                                 specs,
                                                 cross_track_fovs,
                                                 orbitdata,
                                                 mission,
                                                 observation_history,
                                                 n_obs[obs_idx],
                                                 t_prev[obs_idx])

            # accumulate total value
            total_value += obs_value

        return total_value    
    
    def _calculate_observation_number_and_revisit_in_path(self,
                                                          path : List[ObservationAction],
                                                          observation_history : ObservationHistory
                                                        ) -> Tuple[List[Dict[GenericObservationTask, int]],
                                                                    List[Dict[GenericObservationTask, float]]]:
        """ Calculate observation number and revisit time for tasks in the given path given the known bids. """

        # initialize observation counters and previous observation time trackers
        n_obs = [defaultdict(int) for _ in path]
        t_prev = [defaultdict(lambda: np.NINF) for _ in path]

        # ---HISTORICAL DATA---
        # get all parent tasks in the given path
        parent_tasks = {parent_task for action in path 
                        for parent_task in action.task.parent_tasks}

        # initiate observation history for all parent tasks in path
        n_obs_history = {parent_task: 0 for parent_task in parent_tasks}
        t_prev_history = {parent_task: np.NINF for parent_task in parent_tasks}

        # iterate through observation history to populate initial observation numbers and previous observation times
        for parent_task in parent_tasks:
            for *_,grid_idx,gp_idx in parent_task.location:                
                # get observation tracker for location
                obs_tracker : ObservationTracker = observation_history.get_observation_history(grid_idx,gp_idx)

                # get previous matching observations for this task
                obs_prev = [obs for obs in obs_tracker.observations 
                                if obs['t_start'] in parent_task.availability
                                or obs['t_end'] in parent_task.availability
                                or (obs['t_start'] < parent_task.availability.left
                                and obs['t_end'] > parent_task.availability.right)
                            ] if obs_tracker else []

                # update previous observation counts 
                n_obs_history[parent_task] += len(obs_prev)                                        
                
                # calculate latest observation time from previous observations
                obs_latest = max(obs_prev, key=lambda obs: obs['t_end'], default=None)
                t_latest = obs_latest['t_end'] if obs_latest else np.NINF
                
                # update previous observation times 
                t_prev_history[parent_task] = max(t_prev_history[parent_task], t_latest)
     
        # ---PATH DATA---
        # initiate observation counter for all parent tasks in path
        n_obs_in_path = {parent_task: 0 for parent_task in parent_tasks}
        t_prev_in_path = {parent_task: np.NINF for parent_task in parent_tasks}

        # initiate previous observations and times along path
        for obs_idx, obs in enumerate(path):           
            for parent_task in obs.task.parent_tasks:
                # check if parent task is being bid on
                if parent_task in self.results: # task is part of negotiations
                    # get previous bids for this task
                    previous_bids = [bid for bid in self.results[parent_task]
                                     if bid.t_img < obs.t_start and bid.has_winner()]

                    # update overall observation number and revisit times along path using previous bids
                    n_obs[obs_idx][parent_task] = len(previous_bids)
                    t_prev[obs_idx][parent_task] = max([bid.t_img for bid in previous_bids], default=np.NINF)                    
                
                else: # task is not part of negotiations
                    # update overall observation number and revisit times along path using historical and path data
                    n_obs[obs_idx][parent_task] = n_obs_history[parent_task] + n_obs_in_path[parent_task]
                    t_prev[obs_idx][parent_task] = max(t_prev_history[parent_task], t_prev_in_path[parent_task])               

                # update previous path observation counts 
                n_obs_in_path[parent_task] += 1
                t_prev_in_path[parent_task] = max(t_prev_in_path[parent_task], obs.t_end)

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
    
    def __update_results_from_bundle(self, new_bundle : List[List[Bid]]) -> None:
        """ Update results dictionary from new bundle. """
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
    BUNDLE-BUILDING PHASE - Path Insertion Methods
    """
    def _direct_insertion_into_path(self,
                                    state : SimulationAgentState,
                                    specs : object,
                                    current_path : List[ObservationAction],
                                    new_task : SpecificObservationTask,
                                    max_slew_rate : float,
                                    max_torque : float
                                ) -> Tuple[List[ObservationAction], float]:
        """ Try to directly insert new task into existing path. """
        # select observation loook angle for new task
        th_img = np.average([new_task.slew_angles.left, new_task.slew_angles.right])

        # initialize feasible observation time
        t_img = None 

        # find possible conflicts in current path
        ## find observations that are being performed during new task accessibility
        observations_during_task_access = [action for action in current_path
                                           if action.t_start in new_task.accessibility
                                           or action.t_end in new_task.accessibility]
        ## get latest observation before new task accessibility
        prev_observations = [action for action in current_path
                             if action.t_end <= new_task.accessibility.left]
        prev_observation = max(prev_observations, key=lambda action: action.t_end) if prev_observations else None
        ## get earliest observation after new task accessibility
        next_observations = [action for action in current_path
                             if action.t_start >= new_task.accessibility.right]
        next_observation = min(next_observations, key=lambda action: action.t_start) if next_observations else None

        # compile conflicting observations
        conflicting_observations = {prev_observation, next_observation} if prev_observation else {next_observation} if next_observation else set()
        ## get unique observations during new task access
        conflicting_observations.update(observations_during_task_access)
        ## sort conflicting observations by start time
        conflicting_observations = sorted([obs for obs in conflicting_observations 
                                           if obs is not None], key=lambda obs: obs.t_start)

        # set current state as a dummy previous observation
        obs_prev = ObservationAction(new_task.instrument_name,  state.attitude[0], state.t)

        # check if gaps between observations can accommodate new task
        for obs_next in conflicting_observations: 
            # check maneuver time between new task and current observations
            m_prev = abs(obs_prev.look_angle - th_img) / max_slew_rate
            m_next = abs(obs_next.look_angle - th_img) / max_slew_rate        
            
            # get earliest and latest feasible observation time
            t_earliest = max(new_task.accessibility.left, obs_prev.t_end + m_prev)
            t_latest = min(new_task.accessibility.right, obs_next.t_start - m_next) - new_task.min_duration

            # check if feasible observation time exists
            ## 1) must be able to maneuver from previous observation to new task
            ## 2) must be able to maneuver from new task to next observation
            ## 3) must fit within new task accessibility window
            earliest_is_feasible = (t_earliest + new_task.min_duration + m_next <= obs_next.t_start
                                    and obs_prev.t_end + m_prev <= t_earliest
                                    and new_task.accessibility.left <= t_earliest
                                    and t_earliest + new_task.min_duration <= new_task.accessibility.right)
            latest_is_feasible = (t_latest + new_task.min_duration + m_next <= obs_next.t_start
                                    and obs_prev.t_end + m_prev <= t_latest
                                    and new_task.accessibility.left <= t_latest 
                                    and t_latest + new_task.min_duration <= new_task.accessibility.right)
            
            # if feasible, select observation time
            if earliest_is_feasible:
                # choose earliest feasible time
                t_img = t_earliest
            elif latest_is_feasible:
                # choose latest feasible time
                t_img = t_latest

            # if feasible time found, break
            if earliest_is_feasible or latest_is_feasible: break    

            # else; update previous observation
            obs_prev = obs_next

        # no conflicting observations were found
        if not conflicting_observations:
            # schedule at earliest access time
            t_img = new_task.accessibility.left

        # check if observation time was found
        if t_img is None: return None, None # no time found; cannot insert new task into path

        # insert new observation into path
        ## create observation action for new task
        new_observation = ObservationAction(new_task.instrument_name, th_img, t_img, new_task.min_duration, new_task)

        ## create new path with inserted observation
        new_path = [action for action in current_path]
        new_path.append(new_observation)
        new_path = sorted(new_path, key=lambda action: action.t_start)
        
        # return new path if valid
        return (new_path, t_img) if self.is_observation_path_valid(state, new_path, max_slew_rate, max_torque, specs) else (None, None)

    def _right_shift_path_for_new_task(self,
                                        state : SimulationAgentState,
                                        specs : object,
                                        current_path : List[ObservationAction],
                                        new_task : SpecificObservationTask,
                                        max_slew_rate : float,
                                        max_torque : float
                                    ) -> Tuple[List[ObservationAction], float]:
        """ Try to right-shift existing path to accommodate new task. """
        # TODO Requires Testing
        # raise NotImplementedError("Replace conflicting tasks with new task method not yet implemented.")

        # check if path is empty
        assert len(current_path) > 0, "Current path is empty; cannot right-shift path for new task."
        # check if path is sorted by start time
        assert all(current_path[i].t_start <= current_path[i+1].t_start for i in range(len(current_path)-1)), "Current path is not sorted by start time."

        # select observation look angle for new task
        th_img = np.average([new_task.slew_angles.left, new_task.slew_angles.right])

        # find current path observations that occur before the end of the new task's accessibility
        preceeding_observations = [(path_idx,action) for path_idx,action in enumerate(current_path)
                                    if action.t_start <= new_task.accessibility.right]

        # add a dummy observation at the initial state
        preceeding_observations.insert(0, (-1, ObservationAction(new_task.instrument_name, state.attitude[0], state.t)))

        # initialize feasible path insertion index and observation time
        i_insert, t_img = None, None

        # iterate through previous observations to find insertion point
        for i_obs,obs_prev in preceeding_observations:
            # check maneuver time between new task and current observation
            m_prev = abs(obs_prev.look_angle - th_img) / max_slew_rate

            # calculate earliest feasible observation time
            t_earliest = max(new_task.accessibility.left, obs_prev.t_end + m_prev)

            # calculate observation feasibility
            ## 1) must be able to maneuver from previous observation to new task
            ## 2) must fit within new task accessibility window
            is_feasible = (obs_prev.t_end + m_prev <= t_earliest
                           and new_task.accessibility.left <= t_earliest
                           and t_earliest + new_task.min_duration <= new_task.accessibility.right)
            
            # check feasibility
            if is_feasible:
                # update insertion index to next location
                i_insert = i_obs + 1
                # update observation time
                t_img = t_earliest
                
            else:
                # stop searching
                break
        
        # check if insertion index was found
        if i_insert is None: return None, None # no insertion point found; cannot right-shift path for new task

        # initiate new path
        new_path = [action for action in current_path[:i_insert]]
        
        # create new observation action
        new_observation = ObservationAction(new_task.instrument_name, th_img, t_img, new_task.min_duration, new_task)
        
        # add new observation to new path
        new_path.append(new_observation)

        # right-shift remaining observations
        path_to_shift = [action for action in current_path[i_insert:]]
        for i_curr,obs_curr in enumerate(path_to_shift):
            # check previous observation in path
            obs_prev = new_path[-1]

            # compute maneuver time from previous observation
            m = abs(obs_prev.look_angle - obs_curr.look_angle) / max_slew_rate

            # calculate earliest start time for current observation
            t_earliest = max(obs_prev.t_end + m, obs_curr.task.accessibility.left)

            # check earliest time if feasible
            is_feasible = (obs_prev.t_end + m_prev <= t_earliest
                           and new_task.accessibility.left <= t_earliest
                           and t_earliest + new_task.min_duration <= new_task.accessibility.right)

            # check of new observation time is earlier the or the same as original
            if t_earliest < obs_curr.t_start or abs(t_earliest - obs_curr.t_start) <= self.EPS:
                # new task starts earlier, do not modify remaining plan and add to new path
                new_path.extend(path_to_shift[i_curr:])
                break

            # else if new observation time is feasible, add shifted observation to new path
            elif is_feasible: 
                # create shifted observation action
                shifted_observation = ObservationAction(obs_curr.instrument_name, obs_curr.look_angle, t_earliest, obs_curr.task.min_duration, obs_curr.task)

                # add shifted observation to new path
                new_path.append(shifted_observation)
                
            # else, task needs a later start time but is not feasible
            else: return None, None # cannot right-shift path for new task
            
        # return new path if valid
        return (new_path, t_img) if self.is_observation_path_valid(state, new_path, max_slew_rate, max_torque, specs) else (None, None)
    
    def _replace_conflicting_tasks_with_new_task(self,
                                                 state : SimulationAgentState,
                                                 specs : object,
                                                 current_path : List[ObservationAction],
                                                 new_task : SpecificObservationTask,
                                                 max_slew_rate : float,
                                                 max_torque : float
                                            ) -> Tuple[List[ObservationAction], float]:
        """ Try to replace conflicting tasks in existing path with new task. """
        # check if path is empty
        assert len(current_path) > 0, "Current path is empty; cannot replace conflicting tasks with new task."

        # find possible conflicts in current path
        ## find observations that are being performed during new task accessibility
        observations_during_task_access = [(obs_idx,obs) for obs_idx,obs in enumerate(current_path)
                                           if obs.t_start in new_task.accessibility
                                           or obs.t_end in new_task.accessibility
                                           or (obs.t_start < new_task.accessibility.left
                                               and obs.t_end > new_task.accessibility.right)]

        conflicting_observations = [obs_tup for obs_tup in observations_during_task_access]

        # check if conflicting observations were found
        assert conflicting_observations, "No conflicting observations found; direct insertion should have been possible."
        
        # select observation loook angle for new task
        th_img = np.average([new_task.slew_angles.left, new_task.slew_angles.right])

        # check if removing conflicting observations can accommodate new task
        for conflict_idx,conflicting_observation in conflicting_observations:                      
            # get preceeding observation action
            if conflict_idx == 0:
                # set previous observation as dummy action at current state
                obs_prev = ObservationAction(new_task.instrument_name, state.attitude[0], state.t)
            else:
                # select previous observation from path
                obs_prev = current_path[conflict_idx-1]

            # get succeeding observation action
            if conflict_idx == len(current_path) - 1:
                # set next observation as last action on the path
                obs_next = current_path[-1] 
            else:
                # select next observation from path
                obs_next = current_path[conflict_idx+1]

            # calculate maneuver time between new task and current observations
            m_prev = abs(obs_prev.look_angle - th_img) / max_slew_rate
            m_next = abs(obs_next.look_angle - th_img) / max_slew_rate

            # set earliest feasible observation time
            t_img = max(new_task.accessibility.left, obs_prev.t_end + m_prev)

            # check if earlist observation time is feasible
            ## 1) must be able to maneuver from previous observation to new task
            prev_to_earliest_can_maneuver = obs_prev.t_end + m_prev <= t_img
            ## 2) must be able to maneuver from new task to next observation
            earliest_to_next_can_maneuver = t_img + new_task.min_duration + m_next <= obs_next.t_start
            ## 3) must fit within new task accessibility window            
            earliest_in_access = (new_task.accessibility.left <= t_img
                                  and t_img + new_task.min_duration <= new_task.accessibility.right)
            is_observation_feasible : bool = (prev_to_earliest_can_maneuver and earliest_to_next_can_maneuver and earliest_in_access)

            # check if new task cannot be scheduled even when removing this conflicting observation
            if not is_observation_feasible: 
                continue 
            
            # create new observation action for new task
            new_observation = ObservationAction(new_task.instrument_name, th_img, t_img, new_task.min_duration, new_task)
            
            # replace conflicting observation with new task
            new_path = [action for action in current_path]
            new_path[conflict_idx] = new_observation

            # ensure conflicting observation was replaced
            assert conflicting_observation not in new_path, "Conflicting observation was not replaced in new path."

            # return new path if valid
            if self.is_observation_path_valid(state, new_path, max_slew_rate, max_torque, specs):
                return new_path, t_img

        # unable to accommodate new task by replacing conflicting observations
        return None, None
    
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
    def log_results(self, dsc : str, state : SimulationAgentState, level=logging.DEBUG) -> None:
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

    def log_bundle(self, dsc : str, state : SimulationAgentState, level=logging.DEBUG) -> None:
        out = f'\nT{np.round(state.t,3)}[s]:\t\'{state.agent_name}\'\n{dsc}\n'
        line = 'i\t Req IDs\n'
        
        # count characters in line for formatting
        L_LINE = len(line)

        # header
        out += line 

        # divider 
        for _ in range(L_LINE + 25): out += '='
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