from abc import abstractmethod
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
from chess3d.agents.planning.decentralized.consensus.consensus import ConsensusPlanner
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


class HeuristicInsertionConsensusPlanner(ConsensusPlanner):
    """
    # Heuristic Insertion Consensus Planner

    A decentralized consensus planner that utilizes heuristic insertion strategies for planning in multi-agent systems.
    """

    # Heuristics available for insertion model
    EARLIEST_ACCESS = 'earliestAccess'
    TASK_VALUE = 'taskValue'
    HEURISTICS = [EARLIEST_ACCESS, TASK_VALUE]

    def __init__(self,
                 heuristic : str = EARLIEST_ACCESS,
                 replan_threshold : int = 1, 
                 debug : bool = False, 
                 logger : bool = None):
        super().__init__(ConsensusPlanner.HEURISTIC_INSERTION, replan_threshold, debug, logger)

        # validate inputs
        assert heuristic in self.HEURISTICS, f"Invalid heuristic '{heuristic}'. Must be one of {self.HEURISTICS}."
                
        # set parameters
        self.heuristic = heuristic

    @runtime_tracker
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
                                                                    if any(parent_task in self.known_event_tasks 
                                                                            for parent_task in task.parent_tasks)]

        # generate new plan according to selected model
        if self.heuristic == self.EARLIEST_ACCESS:
            return self.earliest_access_heuristic_bundle_builder(state, specs, cross_track_fovs, current_plan, schedulable_urgent_tasks, orbitdata, mission, observation_history)
        
        elif self.heuristic == self.TASK_VALUE:
            return self.task_value_heuristic_bundle_builder(state, specs, cross_track_fovs, current_plan, schedulable_urgent_tasks, orbitdata, mission, observation_history)
        
        else:
            raise NotImplementedError(f"Heuristic '{self.heuristic}' not supported.")            
    
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
                           for task in self.known_event_tasks 
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
        # TEMP IMPLEMENTATION: copy existing bundle and path
        bundle : List[Tuple[GenericObservationTask, int, SpecificObservationTask, Bid]] = \
                [task_tuple for task_tuple in self.bundle]
        path = sorted([action for action in current_plan
                        if isinstance(action, ObservationAction)], 
                            key=lambda action: action.t_start)
        new_bids = []
        # if isinstance(current_plan, PeriodicPlan) and abs(state.t - current_plan.t) <= self.EPS:
        #     if sorted_schedulable_urgent_tasks:
        #         raise NotImplementedError("Earliest-access bundle builder initializing for new preplans not yet implemented.")
        #     bundle : List[Tuple[GenericObservationTask, int, SpecificObservationTask, Bid]] = []
        #     path = sorted([action for action in current_plan
        #                     if isinstance(action, ObservationAction)], 
        #                     key=lambda action: action.t_start)
        # else:
        #     bundle : List[Tuple[GenericObservationTask, int, SpecificObservationTask, Bid]] = \
        #         [task_tuple for task_tuple in self.bundle]
        #     path = sorted([action for action in current_plan
        #                         if isinstance(action, ObservationAction)], 
        #                         key=lambda action: action.t_start)

        # iterate through urgent tasks and attempt to add to bundle
        for new_task in tqdm(sorted_schedulable_urgent_tasks, desc=f'{state.agent_name}-REPLANNER: Building bundle', leave=False):
            # Generate proposed paths using heuristic insertion path builder
            proposed_paths = self.__heuristic_insertion_path_builder(state, specs, path, new_task)
            
            # Find best placement in path   
            for proposed_path, t_img in proposed_paths:
                
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

        # temp return
        return bundle, path, new_bids
        
    """
    BUNDLE-BUILDING PHASE - Path Insertion Methods
    """
    def __heuristic_insertion_path_builder(self,
                                            state : SimulationAgentState,
                                            specs : object,
                                            current_path : List[ObservationAction],
                                            new_task : SpecificObservationTask
                                        ) -> List[Tuple[List[ObservationAction], float]]:
        """ 
        Generates a list of proposed paths by applying the following operators to the path:
            1. Direct Insertion into existing path
            2. Right-shifting existing path to accommodate new task
            3. Replace conflicting task with new urgent task

        #### Returns:

            - `proposed_paths` : List[Tuple[List[ObservationAction], float]]
        """
        
        # compile agility specifications
        max_slew_rate, max_torque = self._collect_agility_specs(specs)

        proposed_paths = [
            # Option 1: Direct Insertion into existing path
            self._direct_insertion_into_path(state, specs, current_path, new_task, max_slew_rate, max_torque),

            # Option 2: Right-shifting existing path to accommodate new task
            self._right_shift_path_for_new_task(state, specs, current_path, new_task, max_slew_rate, max_torque),

            # Option 3: Replace conflicting task with new urgent task
            self._replace_conflicting_tasks_with_new_task(state, specs, current_path, new_task, max_slew_rate, max_torque),

            # TODO Option 4: Remove all conflicting tasks and insert new task
            # self._remove_conflicting_tasks_and_insert_new_task(state, specs, current_path, new_task, max_slew_rate, max_torque),
        ]

        # return proposed paths and the respective observation times for the new task in said paths
        return proposed_paths
        
    def _direct_insertion_into_path(self,
                                    state : SimulationAgentState,
                                    specs : object,
                                    current_path : List[ObservationAction],
                                    new_task : SpecificObservationTask,
                                    max_slew_rate : float,
                                    max_torque : float
                                ) -> Tuple[List[ObservationAction], float]:
        """ Try to directly insert new task into existing path. """
        # initialize feasible observation time and select observation loook angle for new task
        t_img, th_img = None, np.average([new_task.slew_angles.left, new_task.slew_angles.right])

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
                           and t_earliest in new_task.accessibility
                           and t_earliest + new_task.min_duration in new_task.accessibility)
            
            # check feasibility
            if not is_feasible: 
                break # cannot insert at this point; stop searching
            
            # update insertion index to next location
            i_insert = i_obs + 1
            # update observation time
            t_img = t_earliest
        
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
            is_feasible = (obs_prev.t_end + m <= t_earliest
                           and t_earliest in obs_curr.task.accessibility
                           and t_earliest + obs_curr.task.min_duration in obs_curr.task.accessibility)

            # check of new observation time is earlier the or the same as original
            if t_earliest < obs_curr.t_start or abs(t_earliest - obs_curr.t_start) <= self.EPS:
                # new task start time is earlier than original; 
                #  do not modify remaining plan and add to new path
                new_path.extend(path_to_shift[i_curr:])

                # stop shifting process
                break

            # else if new observation time is feasible, add shifted observation to new path
            elif is_feasible: 
                # create shifted observation action
                shifted_observation = ObservationAction(obs_curr.instrument_name, obs_curr.look_angle, t_earliest, obs_curr.task.min_duration, obs_curr.task)

                # add shifted observation to new path
                new_path.append(shifted_observation)
                
            # else, task needs a later start time but is not feasible
            else: 
                # do not add this task and try to shift remaining tasks
                continue
                # return None, None # cannot right-shift path for new task
            
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
            # create new proposed path
            new_path = [action for action in current_path]

            # remove conflicting observation
            new_path.pop(conflict_idx) 

            # get preceeding observation action
            if conflict_idx == 0:
                # set previous observation as dummy action at current state
                obs_prev = ObservationAction(new_task.instrument_name, state.attitude[0], state.t)
            else:
                # select previous observation from path
                obs_prev = new_path[conflict_idx-1]

            # calculate maneuver time between previous observation and new task
            m_prev = abs(obs_prev.look_angle - th_img) / max_slew_rate

            # estimate earliest feasible observation time
            t_img = max(new_task.accessibility.left, obs_prev.t_end + m_prev)

            # check if there is a succeeding observation action
            if conflict_idx < len(new_path):
                # there is a succeeding observation action
                obs_next = new_path[conflict_idx]

                # calculate maneuver time between new task and next observation
                m_next = abs(obs_next.look_angle - th_img) / max_slew_rate

                # check if earliest observation time is feasible
                feasibility_constraints = [
                    # 1) must be able to maneuver from previous observation to new task
                    obs_prev.t_end + m_prev <= t_img,   
                    # 2) must be able to maneuver from new task to next observation
                    t_img + new_task.min_duration + m_next <= obs_next.t_start, 
                    # 3) must fit within new task accessibility window
                    t_img in new_task.accessibility,
                    t_img + new_task.min_duration in new_task.accessibility
                ]
                
            else:
                # conflict was last observation in path; only consider previous observation constraints

                # check if earliest observation time is feasible
                feasibility_constraints = [
                    # 1) must be able to maneuver from previous observation to new task
                    obs_prev.t_end + m_prev <= t_img,   
                    # 2) must fit within new task accessibility window
                    t_img in new_task.accessibility,
                    t_img + new_task.min_duration in new_task.accessibility
                ]

            # check if new task cannot be scheduled even when removing this conflicting observation
            if not all(feasibility_constraints): 
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
    BUNDLE-BUILDING PHASE - Bid Generation Methods
    """
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
                        if parent_task in self.known_event_tasks]

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
    
    