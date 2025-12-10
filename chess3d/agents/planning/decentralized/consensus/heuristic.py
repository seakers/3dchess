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
from chess3d.agents.planning.tasks import DefaultMissionTask, GenericObservationTask, EventObservationTask, SpecificObservationTask
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
                       tasks : List[GenericObservationTask],
                       current_plan : Plan,
                       clock_config : ClockConfig,
                       orbitdata : OrbitData,
                       mission : Mission,
                       observation_history : ObservationHistory
                    ) -> tuple:
        
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self._collect_fov_specs(specs)

        # Outline planning horizon interval
        t_next = self.preplan.t_next if self.preplan is not None else np.Inf
        planning_horizon = Interval(state.t, t_next)

        # get only available tasks from existing plan and urgent tasks
        available_tasks : list[GenericObservationTask] = self.get_available_tasks(tasks, planning_horizon)
        
        # calculate coverage opportunities for available tasks
        access_opportunities : dict[tuple] = self.calculate_access_opportunities(state, planning_horizon, orbitdata)

        # create specific and merged tasks from scheduled tasks and urgent tasks
        schedulable_tasks : List[SpecificObservationTask] = self.create_tasks_from_accesses(available_tasks, access_opportunities, cross_track_fovs, orbitdata)

        # extract already planned specific tasks from current plan
        planned_specific_tasks = [obs.task for obs in current_plan if isinstance(obs,ObservationAction)]
        
        # filter tasks that are already in the current plan
        schedulable_tasks = [task for task in schedulable_tasks
                             if task not in planned_specific_tasks]
        
        # -------------------------------
        # DEBUG PRINTOUTS
        if self._debug:
            out = f'\nT{np.round(state.t,3)}[s]:\t\'{state.agent_name}\'\n'
            out += 'TASKS CONSIDERED FOR BUNDLE BUILDING:\n'
            # header
            line = 'i\tSpecTaskID\tAccess Start\tParentID(s)\t\n'
            out += line
            L_LINE = len(line)
            L_LINE_PADding = 30
            # divider 
            for _ in range(L_LINE + L_LINE_PADding): out += '='
            out += '\n'
            # task entries
            for task_idx,task in enumerate(schedulable_tasks):
                out += f'{task_idx}\t{task.id.split("-")[0]}\t{np.round(task.accessibility.left,1)}\t\t{[str(p) for p in task.parent_tasks]}\n'
            out += f'Total Tasks Considered: {len(schedulable_tasks)}\n'
            # print to console
            print(out)
        # -------------------------------

        # generate new plan according to selected model
        if self.heuristic == self.EARLIEST_ACCESS:
            # use earliest-access heuristic
            return self.earliest_access_heuristic_bundle_builder(state, specs, cross_track_fovs, current_plan, schedulable_tasks, orbitdata, mission, observation_history)
        
        elif self.heuristic == self.TASK_VALUE:
            # use task-value heuristic
            return self.task_value_heuristic_bundle_builder(state, specs, cross_track_fovs, current_plan, schedulable_tasks, orbitdata, mission, observation_history)
        
        # Fallback for unsupported heuristic
        raise NotImplementedError(f"Heuristic '{self.heuristic}' not supported.")            
    
    def get_available_tasks(self, tasks: List[GenericObservationTask], planning_horizon : Interval) -> list:
        """ Get only tasks that are available within the planning horizon. """
        # get known tasks that may already be part of the plan
        default_tasks = {task 
                         for task in tasks
                         if isinstance(task, DefaultMissionTask)
                         and task.availability.overlaps(planning_horizon)
                         }

        # get urgent event tasks that are available within planning horizon
        event_tasks = {task 
                        for task in self.known_event_tasks 
                        if task.availability.overlaps(planning_horizon)}
        
        # merge task sets
        available_tasks = {task for task in event_tasks}
        available_tasks.update(default_tasks)

        # return tasks as a merged list
        return list(available_tasks)   

    def earliest_access_heuristic_bundle_builder(self,
                                       state : SimulationAgentState,
                                       specs : object,
                                       cross_track_fovs : dict,
                                       current_plan : Plan,
                                       schedulable_tasks : List[SpecificObservationTask],
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
        sorted_schedulable_tasks = sorted(schedulable_tasks, key=lambda task: task.accessibility.left)
    
        # build bundle using heuristic insertion method
        return self.__heuristic_insertion_bundle_builder(state, specs, cross_track_fovs, current_plan, sorted_schedulable_tasks, orbitdata, mission, observation_history, heuristic_evaluator=lambda task: task.accessibility.left)

    def task_value_heuristic_bundle_builder(self,
                                       state : SimulationAgentState,
                                       specs : object,
                                       cross_track_fovs : dict,
                                       current_plan : Plan,
                                       schedulable_tasks : List[SpecificObservationTask],
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
                                                               observation_history)) for task in schedulable_tasks]
        sorted_schedulable_tasks = [task for task, _ in sorted(task_values, key=lambda item: item[1], reverse=True)]
    
        # build bundle using heuristic insertion method
        return self.__heuristic_insertion_bundle_builder(state, specs, cross_track_fovs, current_plan, sorted_schedulable_tasks, orbitdata, mission, observation_history)

    def _is_task_mutually_exclusive_with_path(self, task : SpecificObservationTask, path : List[ObservationAction]):
        """ Check if task is mutually exclusive with any observations in the given path. """
        return any([task.is_mutually_exclusive(action.task) for action in path])

    def __heuristic_insertion_bundle_builder(self,
                                       state : SimulationAgentState,
                                       specs : object,
                                       cross_track_fovs : dict,
                                       current_plan : Plan,
                                       sorted_schedulable_tasks : List[SpecificObservationTask],
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

        # initialized bundle from current plan
        proposed_bundle : List[Tuple[SpecificObservationTask, Dict[GenericObservationTask, int]]] = \
                [task_tuple for task_tuple in self.bundle]
        
        # initialize proposed path from current plan
        proposed_path = sorted([action for action in current_plan
                                if isinstance(action, ObservationAction)], 
                            key=lambda action: action.t_start)
        
        # extract existing bids from current bundle
        proposed_bids = [[self.results[task][n_obs] for task,n_obs in obs.items()] for _,obs in proposed_bundle] 

        # iterate through urgent tasks and attempt to add to bundle
        for proposed_task in tqdm(sorted_schedulable_tasks, desc=f'{state.agent_name}-REPLANNER: Building bundle', leave=False):
            # Generate proposed paths using heuristic insertion path builder
            candidate_paths = self.__heuristic_insertion_path_builder(state, specs, proposed_path, proposed_task)
            
            # Find best placement in path   
            for candidate_path, t_img in candidate_paths:
                # if no feasible path was found, ignore new urgent task
                if candidate_path is None: continue
                
                # find all possible task observation numbers for proposed task in candidate path
                valid_labelings = self._generate_valid_labelings_for_task_in_path(state, candidate_path, observation_history)

                # if any parent task has no valid observation numbers, skip to next candidate path
                if any((not labels for labels in valid_labelings.values())): continue

                # select observation number combination that maximizes value for that task
                best_labeling = self._get_best_labeling_for_task_in_path(state, specs, cross_track_fovs, orbitdata, mission, observation_history, candidate_path, valid_labelings)

                x = 1

                # create bids for relevant parent tasks
                # new_bids : List[Bid] = self._generate_bids_for_task_in_path(state, specs, proposed_path, candidate_path, proposed_task, 
                #                                                             t_img, cross_track_fovs, orbitdata, mission, observation_history)
                    
                # # if no new bids were generated, skip to next urgent task
                # if not new_bids: continue
            
            raise NotImplementedError("Heuristic insertion bundle builder finalization not yet implemented.")
        
            # add bids to bundle
            proposed_bundle.append([(bid.task, bid.n_obs) for bid in new_bids])

            # update current path                    
            proposed_path = [action for action in candidate_path]

        # temp return
        return proposed_bundle, proposed_path, proposed_bids
        
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
        if len(current_path) == 0: 
            # Current path is empty; cannot right-shift path for new task.
            return (None, None)

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
        if len(current_path) == 0: 
            # Current path is empty; cannot replace conflicting tasks in path for new task.
            return (None, None)

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
    def _generate_valid_labelings_for_task_in_path(self, 
                                                  state : SimulationAgentState,
                                                  proposed_path : List[ObservationAction],
                                                  observation_history : ObservationHistory
                                                ) -> Dict[GenericObservationTask, List[Tuple[List[int], List[float]]]]:
        """ Generate valid labelings for given task in the context of the given path. 
        
        ### Returns 
            - valid_labelings : Dict[GenericObservationTask, List[Tuple[List[int], List[float]]]] - List of valid labelings for the given task. Each labeling is a tuple containing a list of observation numbers and a list of previous observation times.
        """
        # enumerate the generic observation tasks being observed in the proposed path
        t_img_sequences : dict[GenericObservationTask, list[float]] = defaultdict(list)
        n_obs_candidates : dict[GenericObservationTask, list[int]] = defaultdict(list)
        obs_sequence_indices : dict[GenericObservationTask, list[int]] = defaultdict(list)
        
        # extract observation time and sequence indices for each parent 
        for obs_idx,obs in enumerate(proposed_path):
            for parent_task in obs.task.parent_tasks:
                t_img_sequences[parent_task].append(obs.t_start)
                obs_sequence_indices[parent_task].append(obs_idx)

        # count proposed path revisit and observation numbers
        n_obs_prop, _ = self._count_observation_number_and_revisit_times_from_path(state, proposed_path, observation_history)

        # enumerate candidate observation numbers for each parent task
        for parent_task,obs_indices in obs_sequence_indices.items():
            # check if task is being bid on
            assert parent_task in self.results, f"Parent task {parent_task} not being bid on by any agent; cannot generate bids."
            
            # calculate the maximum possible observation number for this parent task
            n_obs_max = max(len(self.results[parent_task])+1, len(obs_indices))

            for occurrance_idx,obs_idx in enumerate(obs_indices):
                # get previous observation number for this parent task before path started
                n_obs_prev = n_obs_prop[obs_sequence_indices[parent_task][0]].get(parent_task, 0)

                # calculate upper and lower bounds for candidate observation numbers
                n_obs_lower_bound = n_obs_prev + occurrance_idx
                n_obs_upper_bound = n_obs_max - (len(obs_indices) - occurrance_idx - 1)

                # generate candidate observation numbers for this parent task
                n_obs_occurance = list(range(n_obs_lower_bound, n_obs_upper_bound))
                n_obs_candidates[parent_task].append(n_obs_occurance)

        # return valid labelings for each parent task
        return {
                    parent_task: self.enumerate_labelings_for_task(state, parent_task, t_img_sequences[parent_task], n_obs_candidates[parent_task])
                    for parent_task in obs_sequence_indices.keys()
                }

    def enumerate_labelings_for_task(
                                        self,
                                        state : SimulationAgentState,
                                        task : GenericObservationTask,
                                        t_img_sequence,     # [t1, t2, ..., tm]
                                        n_obs_candidates,   # [F1, F2, ..., Fm], each Fi is a small set or range of ints
                                        # external_prev_time, # dict n -> t_ext[n] for already scheduled obs
                                        max_solutions=np.Inf  # optional limit
                                    ):
        """ Performs depth-first search to enumerate all valid labelings of observation numbers for a given task. """
        # get length of observation sequence
        task_obs_sequence_length = len(t_img_sequence)
        
        # initiate assignment lists
        n_obs_assignments = [None] * task_obs_sequence_length   # n_obs_r for r=0..m-1
        t_prev_assignments = [None] * task_obs_sequence_length  # t_prev_r for r=0..m-1
        
        # initiate valid solutions list
        valid_solutions = []

        # Initialize map for which `n_obs` have we already assigned and at what time (from this agent)
        local_obs_time : Dict[int, float] = {}  # n_obs -> t_img

        def prev_t_img(n_obs : int):
            # Find time of (n-1)-th obs, from external or our path
            n_obs_prev = n_obs - 1
            
            # check if there is no predecessor
            if n_obs_prev < 0: 
                return np.NINF # no previous observation; return negative infinity
            
            # check if predecessor is being considered by this agent
            if n_obs_prev in local_obs_time:
                return local_obs_time[n_obs_prev] # found locally
            
            # check if this task is being bid on by other agents
            if task not in self.results: 
                return None # no other agents bidding on this task; no predecessor available

            # else, lookup predecessor observation time from external bids
            prev_bid : Bid = self.results[task][n_obs_prev] if n_obs_prev < len(self.results.get(task, [])) else None

            # check if no one has bid for predecessor
            if prev_bid is None: return None

            # check if parent agent is winning predecessor
            if prev_bid.winning_bidder == state.agent_name:
                # if reaching here, it means that the predecessor should have been found locally
                return None
        
            # else return external observation time if predecessor has a winner
            return prev_bid.t_img if prev_bid.has_winner() else None

        def dfs(task_obs_sequence_idx : int):
            # initiate solution list
            nonlocal valid_solutions

            # timeout: check if maximum solutions reached
            if len(valid_solutions) >= max_solutions: return

            # base case: check if sequence is fully labeled
            if task_obs_sequence_idx == task_obs_sequence_length:
                # full labeling found; add to list of solutions and return to previous case
                return valid_solutions.append((n_obs_assignments.copy(), t_prev_assignments.copy()))

            # get current observation time
            t_img = t_img_sequence[task_obs_sequence_idx]

            # iterate candidate `n_obs` for this observation
            for n_obs in sorted(n_obs_candidates[task_obs_sequence_idx]):
                # get previous imaging time and observation number for candidate `n_obs`
                t_prev = prev_t_img(n_obs)
                n_obs_prev = n_obs_assignments[task_obs_sequence_idx-1] if task_obs_sequence_idx > 0 else None

                # define constraints
                constraints = [
                    (n_obs_prev is not None and n_obs_prev < n_obs) or n_obs == 0,  # observation number must be greater than sequence index
                    (t_prev is not None and t_prev <= t_img) or n_obs == 0          # previous observation time must be before current time
                ]

                # check constraints
                if not all(constraints): 
                    continue  # constraints not satisfied; try next candidate `n_obs`

                # commit to assignment lists
                n_obs_assignments[task_obs_sequence_idx] = n_obs
                t_prev_assignments[task_obs_sequence_idx] = t_prev

                # save previous local observation time value for future undo
                local_prev_value = local_obs_time.get(n_obs, None)

                # assign previous observation time to local value map
                local_obs_time[n_obs] = t_img

                # check next observation in sequence
                dfs(task_obs_sequence_idx + 1)

                # undo previous observation time assignment
                if local_prev_value is None:
                    del local_obs_time[n_obs]
                else:
                    local_obs_time[n_obs] = local_prev_value

                # undo observation sequence assignment
                n_obs_assignments[task_obs_sequence_idx] = None
                t_prev_assignments[task_obs_sequence_idx] = None
            
            # fallback;
            return

        dfs(0)
        return valid_solutions

    def _get_best_labeling_for_task_in_path(self,
                                            state : SimulationAgentState,
                                            specs : object,
                                            cross_track_fovs : dict,
                                            orbitdata : OrbitData,
                                            mission : Mission,
                                            observation_history : ObservationHistory,
                                            proposed_path : List[ObservationAction],
                                            valid_labelings : Dict[GenericObservationTask, List[Tuple[List[int], List[float]]]]
                                            ) -> Tuple[List[int], List[float]]:
        """ Select the best labeling for a given task in the context of the given path. 
        
        ### Returns 
            - best_labeling : Tuple[List[int], List[float]] - Best labeling for the given task. A tuple containing a list of observation numbers and a list of previous observation times.
        """
        # initiate imaging times and sequence for each task in the proposed path
        t_img_sequences : dict[GenericObservationTask, list[float]] = defaultdict(list)
        obs_sequence_indices : dict[GenericObservationTask, list[int]] = defaultdict(list)
        
        # extract observation time and sequence indices for each parent 
        for obs_idx,obs in enumerate(proposed_path):
            for parent_task in obs.task.parent_tasks:
                t_img_sequences[parent_task].append(obs.t_start)
                obs_sequence_indices[parent_task].append(obs_idx)

        # find labing that maximizes value for each task 
        for parent_task,task_lablings in valid_labelings.items():
            # count maximum number of observations for this parent task
            n_obs_max = max(len(self.results[parent_task])+1, len(obs_sequence_indices[parent_task]))
            
            # estimate value of proposed lablings
            for n_obs_sequences, t_prev_sequences in task_lablings:
                # initiate list of values for this labeling
                task_values = [0.0 for _ in range(n_obs_max)]
                task_t_imgs = [None for _ in range(n_obs_max)]

                # calculate value for this labeling
                for n_obs_in_seq,t_prev_in_seq,obs_idx_in_seq in zip(n_obs_sequences, t_prev_sequences, obs_sequence_indices[parent_task]):
                    # find matching observation action in proposed path
                    obs_action : ObservationAction = proposed_path[obs_idx_in_seq]

                    # calculate and accumulate value for this observation
                    task_values[n_obs_in_seq] = self._estimate_task_value(parent_task,
                                                                        obs_action.instrument_name,
                                                                        obs_action.look_angle,
                                                                        obs_action.t_start,
                                                                        obs_action.task.min_duration,
                                                                        specs,
                                                                        cross_track_fovs,
                                                                        orbitdata,
                                                                        mission,
                                                                        observation_history,
                                                                        n_obs_in_seq,
                                                                        t_prev_in_seq
                                                                        )
                    # assign observation time for this observation number
                    task_t_imgs[n_obs_in_seq] = obs_action.t_start

                # fill in gaps between assigned values with current bids if necessary
                for i,t in enumerate(task_t_imgs):
                    if i == 0: continue # skip first observation number

                    if t is None and task_t_imgs[i-1] is not None:
                        # missing value; try to fill in from existing bids
                        # check if bid exists for this observation number
                        if i < len(self.results.get(parent_task, [])):
                            # get bid for this observation number
                            bid : Bid = self.results[parent_task][i]

                            assert bid.winning_bidder != state.agent_name, \
                                "Internal error: trying to fill in missing bid from own agent."

                            # check if bid has a winner
                            if bid.has_winner():
                                # assign value and observation time from winning bid
                                task_values[i] = bid.winning_bid
                                task_t_imgs[i] = bid.t_img

                # ensure tasks with assigned times have assigned predecessor times
                assert all((t is not None) or (i == 0) or (task_t_imgs[i-1] is not None) for i,t in enumerate(task_t_imgs)), \
                    "Assigned observation times have missing predecessors."

                # compute total labeling value
                labeling_value = sum(task_values)

                # TODO check if it is the best labeling so far

                # if so, check for constraints
                #   if constraints are met, set as best labeling for this task
                #   else, check if constraint violations are still allowed for this task
                #       if so, set as best labeling for this task
                #       else, continue to next labeling
                # else, continue to next labeling
                x = 1            

        raise NotImplementedError("Best labeling selection not yet implemented.")

    # def _generate_bids_for_task_in_path(self,
    #                                     state : SimulationAgentState,
    #                                     specs : object,
    #                                     current_path : List[ObservationAction],
    #                                     proposed_path : List[ObservationAction],
    #                                     task_to_schedule : SpecificObservationTask,
    #                                     t_img : float,
    #                                     cross_track_fovs : dict,
    #                                     orbitdata : OrbitData,
    #                                     mission : Mission,
    #                                     observation_history : ObservationHistory
    #                                 ) -> List[Bid]:
    #     """ Generate bid for given task in the context of the given path. 
        
    #     ### Returns 
    #         - bids : List[Tuple[GenericObservationTask, int]]] - List of bids for each parent task of the given task. Is None if no valid bids could be generated.
    #     """

    #     # count current path revisit and observation numbers
    #     # TODO make and use a counter that only uses existing bidding results
    #     n_obs_curr, t_prev_curr = self._count_observation_number_and_revisit_times_from_path(state, current_path, observation_history)

    #     # calculate current path utility
    #     current_path_utility : float = self._calculate_path_utility(state, specs, cross_track_fovs, current_path, observation_history, orbitdata, mission, n_obs_curr, t_prev_curr)

        # # enumerate the generic observation tasks being observed in the proposed path
        # t_img_sequences : dict[GenericObservationTask, list[float]] = defaultdict(list)
        # n_obs_candidates : dict[GenericObservationTask, list[int]] = defaultdict(list)
        # obs_sequence_indices : dict[GenericObservationTask, list[int]] = defaultdict(list)
        
        # # extract observation time and sequence indices for each parent 
        # for obs_idx,obs in enumerate(proposed_path):
        #     for parent_task in obs.task.parent_tasks:
        #         t_img_sequences[parent_task].append(obs.t_start)
        #         obs_sequence_indices[parent_task].append(obs_idx)

    #     # count proposed path revisit and observation numbers
    #     n_obs_prop, _ = self._count_observation_number_and_revisit_times_from_path(state, proposed_path, observation_history)

    #     # enumerate candidate observation numbers for each parent task
    #     for parent_task,obs_indices in obs_sequence_indices.items():
    #         # check if task is being bid on
    #         assert parent_task in self.results, f"Parent task {parent_task} not being bid on by any agent; cannot generate bids."
            
    #         # calculate the maximum possible observation number for this parent task
    #         n_obs_max = max(len(self.results[parent_task])+1, len(obs_indices))

    #         for occurrance_idx,obs_idx in enumerate(obs_indices):
    #             # get previous observation number for this parent task before path started
    #             n_obs_prev = n_obs_prop[obs_sequence_indices[parent_task][0]].get(parent_task, 0)

    #             # calculate upper and lower bounds for candidate observation numbers
    #             n_obs_lower_bound = n_obs_prev + occurrance_idx
    #             n_obs_upper_bound = n_obs_max - (len(obs_indices) - occurrance_idx - 1)

    #             # generate candidate observation numbers for this parent task
    #             n_obs_occurance = list(range(n_obs_lower_bound, n_obs_upper_bound))
    #             n_obs_candidates[parent_task].append(n_obs_occurance)

    #     # enumerate valid labelings for each parent task
    #     valid_n_obs_sequences : dict[GenericObservationTask, list[list[int]]] = defaultdict(list)
    #     valid_t_prev_sequences : dict[GenericObservationTask, list[list[float]]] = defaultdict(list)
    #     for parent_task in obs_sequence_indices.keys():
    #         # enumerate valid labelings
    #         valid_solutions = self.enumerate_labelings_for_task(parent_task, t_img_sequences[parent_task], n_obs_candidates[parent_task])

    #         # calculate value of each valid labeling
    #         valid_solution_values : list[float] = []
    #         for sol_idx,(n_obs_sequences,t_prev_sequences) in enumerate(valid_solutions):
    #             valid_solution_value = 0.0

    #             # TODO count all of the observation numbers for a given task to calculate the total value.
    #             # Use newly computed value for valid labeling to compute bid value. If gaps exist in the sequence,
    #             # use a known bid from the current results to fill in the gaps. the goal is to find the label 
    #             # that maximizes the total value for the task, together with its proposed observation sequence
    #             # and the observations being bid on by other agents.

    #             # iterate through observation sequence
                # for n_obs_in_seq,t_prev_in_seq,obs_idx_in_seq in zip(n_obs_sequences, t_prev_sequences, obs_sequence_indices[parent_task]):
                #     # find matching observation action in proposed path
                #     obs_action : ObservationAction = proposed_path[obs_idx_in_seq]

                #     # calculate and accumulate value for this observation
                #     valid_solution_value += self._estimate_task_value(parent_task,
                #                                                 obs_action.instrument_name,
                #                                                 obs_action.look_angle,
                #                                                 obs_action.t_start,
                #                                                 obs_action.task.min_duration,
                #                                                 specs,
                #                                                 cross_track_fovs,
                #                                                 orbitdata,
                #                                                 mission,
                #                                                 observation_history,
                #                                                 n_obs_in_seq,
                #                                                 t_prev_in_seq
                #                                                 )

    #             valid_solution_values.append(valid_solution_value)
    #             # valid_n_obs_sequences[parent_task].append(n_obs_sequences)
    #             # valid_t_prev_sequences[parent_task].append(t_prev_sequences)

    #     x = 1

        # # get relevant parent tasks from task being scheduled
        # parent_tasks = [parent_task 
        #                 for parent_task in task_to_schedule.parent_tasks
        #                 if parent_task in self.known_event_tasks]

        # # get bounds for min and maximum observation numbers for each parent task
        # n_obs_per_task = {parent_task : list(range(len(self.results[parent_task])+1))
        #                     for parent_task in parent_tasks}

        # # initiate possible observation number and revisit pair tracker for each parent task
        # n_obs_revisit_pairs : Dict[SpecificObservationTask, List[Tuple[int, float]]] = defaultdict(list)
                

        # # calculate observation number and revisit time for tasks in path
        # n_obs, t_prev = self._calculate_observation_number_and_revisit_in_path(path, observation_history)
     
        # # replace observation number and revisit time values for task being scheduled if provided
        # if any(param is not None for param in [task_to_schedule, n_obs_bid, t_prev_bid]):            
        #     # find index and observation action for task being scheduled
        #     matching_index,matching_obs = min([(obs_idx, obs) for obs_idx, obs in enumerate(path) 
        #                                         if obs.task == task_to_schedule],
        #                                         key=lambda item: item[0])
            
        #     # update observation number and previous observation time for task being scheduled
        #     for parent_task in n_obs_bid.keys():
        #         # get previous bids for this task
        #         previous_bids = [bid for bid in self.results[parent_task]
        #                          if bid.t_img < matching_obs.t_start 
        #                          and bid.winning_bidder == state.agent_name]
        #         latest_bid = max(previous_bids, key=lambda bid: bid.t_img, default=None)

        #         # check if overwrite values are valid
        #         assert n_obs_bid[parent_task] >= len(previous_bids), \
        #             f"Proposed observation number {n_obs_bid[parent_task]} for task '{parent_task}' is less than the number of previous bids {len(previous_bids)} for the same task by this agent."
        #         assert t_prev_bid[parent_task] >= (latest_bid.t_img if latest_bid else np.NINF), \
        #             f"Proposed previous observation time {t_prev_bid[parent_task]} [s] for task '{parent_task}' is earlier than the latest previous bid time {latest_bid.t_img if latest_bid else 'NINF'} for the same task by this agent."

        #         # overwrite observation number and previous observation time
        #         n_obs[matching_index][parent_task] = n_obs_bid[parent_task]
        #         t_prev[matching_index][parent_task] = t_prev_bid[parent_task]

        #     # update observation number and previous observation time for task being scheduled
        #     for parent_task in n_obs_bid.keys():
        #         n_obs[matching_index][parent_task] = n_obs_bid[parent_task]
        #         t_prev[matching_index][parent_task] = t_prev_bid[parent_task]

        # # calculate revisit times for each possible observation number
        # for parent_task, n_obs_list in n_obs_per_task.items():
        #     for n_obs in n_obs_list:             
        #         # check if a bid even exists for this observation number
        #         if n_obs >= len(self.results[parent_task]): 
        #             # no bid exists; observation number is valid with infinite revisit time
        #             n_obs_revisit_pairs[parent_task].append((n_obs, np.Inf))
        #             continue

        #         # check if observation has already been performed
        #         if self.results[parent_task][n_obs].was_performed(): 
        #             continue # observation already performed; skip

        #         # check if agent is already scheduled to perform observation
        #         if self.results[parent_task][n_obs].winning_bidder == state.agent_name: 
        #             continue # observation already scheduled by this agent; skip

        #         # calculate previous observation time
        #         t_prev = self.results[parent_task][n_obs-1].t_img if n_obs > 0 else np.NINF

        #         # check if previous observation time is prior to the chosen observation time 
        #         if t_img < t_prev: 
        #             continue # incompatible observation time and observation number; skip

        #         # estimate revisit time
        #         t_revisit = t_img - t_prev if n_obs > 0 else np.NINF

        #         # add valid (n_obs, t_revisit) pair to list
        #         n_obs_revisit_pairs[parent_task].append((n_obs, t_revisit))

        # assert all([parent_task in n_obs_revisit_pairs for parent_task in parent_tasks]), \
        #     "No valid (n_obs, t_revisit) pairs could be generated for all parent tasks."
        
        # # enlist all possible (n_obs, t_revisit) options for each parent task
        # options_lists = [n_obs_revisit_pairs[parent_task] for parent_task in parent_tasks]
        
        # # initiate search for best (n_obs, t_revisit) combination
        # best_combo : dict = dict()
        # best_val : float = np.NINF

        # # calculate bid for each parent task and each possible (n_obs, t_revisit) pair
        # for combo in product(*options_lists):
        #     n_obs_bid = {parent_task : n_obs for parent_task, (n_obs, _) in zip(parent_tasks, combo)}
        #     t_prev_bid = {parent_task : t_prev for parent_task, (_, t_prev) in zip(parent_tasks, combo)}
            
        #     # calculate new path utility with proposed (n_obs, t_revisit) pairs
        #     new_path_utility : float = self._calculate_path_utility(state, specs, cross_track_fovs, proposed_path, observation_history, orbitdata, mission, task_to_schedule, n_obs_bid, t_prev_bid)

        #     # calculate bid value
        #     bid_value : float = new_path_utility - current_path_utility

        #     # create bids for each parent task based on current (n_obs, t_revisit) combination
        #     current_bids = [Bid(parent_task, state.agent_name, n_obs, bid_value, bid_value, 
        #                         state.agent_name, t_img, state.t, 
        #                         main_measurement=task_to_schedule.instrument_name)
        #                     for parent_task, (n_obs, _) in zip(parent_tasks, combo)]
            
        #     x = 1

        #     # calculate total subsequent bid values for competing bids
        #     subsequent_bid_values = {parent_task : sum([subsequent_bid.winning_bid 
        #                                     for subsequent_bid in self.results[parent_task][n_obs+1:]]) \
        #                                     if n_obs < len(self.results[parent_task]) else 0.0
        #                             for parent_task in parent_tasks}
            
        #     # check if outbids all subsequent bids
        #     if any(bid_value <= subsequent_bid_value + self.EPS
        #            for subsequent_bid_value in subsequent_bid_values.values()): 
        #         continue # does not outbid all subsequent bids; skip
            
        #     # check if bid value is best so far
        #     if bid_value > best_val + self.EPS:
        #         best_val = bid_value
        #         best_combo = {parent_task: (n_obs, t_prev) 
        #                       for parent_task, (n_obs, t_prev) in zip(parent_tasks, combo)}
        
        
        # # create and return bids for each parent task based on best (n_obs, t_revisit) combination
        # return [AsynchronousBid(parent_task, state.agent_name, n_obs, best_val, state.agent_name, 
        #                          best_val, t_img, state.t, task_to_schedule.instrument_name) 
        #         for parent_task, (n_obs,_) in best_combo.items()]
        
     
    
    