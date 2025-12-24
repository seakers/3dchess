from abc import abstractmethod
from collections import defaultdict, deque
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
    TASK_PRIORITY = 'taskPriority'
    HEURISTICS = [EARLIEST_ACCESS, TASK_VALUE, TASK_PRIORITY]

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
        
        elif self.heuristic == self.TASK_PRIORITY:
            # use task-priority heuristic
            return self.task_priority_heuristic_bundle_builder(state, specs, cross_track_fovs, current_plan, schedulable_tasks, orbitdata, mission, observation_history)

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
        sorted_schedulable_tasks = [task for task, _ in sorted(task_values, key=lambda item: (item[1], item[0].id), reverse=True)]
    
        # build bundle using heuristic insertion method
        return self.__heuristic_insertion_bundle_builder(state, specs, cross_track_fovs, current_plan, sorted_schedulable_tasks, orbitdata, mission, observation_history)

    def task_priority_heuristic_bundle_builder(self,
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
        Build bundle using task priority as main heuristic for order of task addition in bundle building process. 
         Considers the intrinsic priority of the tasks being considered in the bundle. 

        #### Returns
        - bundle : List[Tuple[GenericObservationTask, int, SpecificObservationTask, Bid]]
            List of tuples containing (task, observation number, observation time, expected utility).
        - path : List[ObservationAction]
            Updated observation path after bundle building.
        """ 
        # sort urgent tasks by intrinsic task priority
        task_priorities = [(task, task.get_priority()) for task in schedulable_tasks]
        sorted_schedulable_tasks = [task for task, _ in sorted(task_priorities, key=lambda item: (item[1], item[0].id), reverse=True)]
        
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
        proposed_bundle : List[Tuple[SpecificObservationTask, 
                                     Dict[GenericObservationTask, int]]] = \
                [task_tuple for task_tuple in self.bundle]
        
        # initialize proposed path from current plan
        proposed_path = sorted([action for action in current_plan
                                if isinstance(action, ObservationAction)], 
                            key=lambda action: action.t_start)
        
        # extract existing bids from current bundle
        proposed_bids = defaultdict(dict)
        for _,obs in proposed_bundle:
            for task,n_obs in obs.items():
                new_bid = self.results[task][n_obs]
                proposed_bids[task][n_obs] = new_bid.copy()

        # extract observation number assignments for current path
        n_obs_proposed, t_prev_proposed = self._count_observations_and_revisit_times_from_results(state, proposed_path)

        # calculate current path utility
        current_path_utility : float = self._calculate_path_utility(state, specs, cross_track_fovs, proposed_path, observation_history, orbitdata, mission, n_obs_proposed, t_prev_proposed)

        # Add tasks to path iteratively based on heuristic
        for proposed_task in tqdm(sorted_schedulable_tasks, desc=f'{state.agent_name}-REPLANNER: Building bundle', leave=False):
            # initialize search for best path for proposed task    
            best_path : List[ObservationAction] = None
            best_path_changes : List[ObservationAction] = None
            best_path_utility : float = current_path_utility # must outperform current path
            n_obs_best : List[Dict[GenericObservationTask, int]] = None
            t_prev_best : List[Dict[GenericObservationTask, float]] = None
            
            # Generate proposed paths using heuristic insertion path builder
            candidate_paths = self.__heuristic_insertion_path_builder(state, specs, proposed_path, proposed_task)

            # Find best placement in path   
            for candidate_path, path_changes in candidate_paths:

                # find best observation sequence for each parent task of the proposed task in this candidate path
                n_obs_candidate, t_prev_candidate = self._assign_best_observations_and_revisit_times_to_proposed_path(state, candidate_path, path_changes, specs, cross_track_fovs, orbitdata, mission, observation_history)

                # create bids from candidate path observation assignments


                # get path value for proposed path using best observation sequences
                proposed_path_utility : float = self._calculate_path_utility(state, specs, cross_track_fovs, candidate_path, observation_history, orbitdata, mission, n_obs_candidate, t_prev_candidate)

                # if path does not increase overall utility, skip
                if proposed_path_utility <= best_path_utility: continue
                
                # else: save as best path
                best_path = candidate_path
                best_path_changes = path_changes
                best_path_utility = proposed_path_utility
                n_obs_best = n_obs_candidate
                t_prev_best = t_prev_candidate

            # if no best path was found, continue to next proposed task
            if best_path is None: continue

            # intiate bids for new observations in best path changes 
            new_bids : Dict[GenericObservationTask, List[Bid]] = defaultdict(list)
            for obs in best_path_changes:
                # get path index for observation
                obs_idx = best_path.index(obs)

                # create bids for each parent task
                for parent_task in obs.task.parent_tasks:
                    # calc value of new bids for best proposed path for this task
                    bid_value = self._estimate_task_value(parent_task,
                                                          obs.instrument_name,
                                                          obs.look_angle,
                                                          obs.t_start,
                                                          obs.t_end-obs.t_start,
                                                          specs,
                                                          cross_track_fovs,
                                                          orbitdata,
                                                          mission,
                                                          observation_history,
                                                          n_obs_best[obs_idx][parent_task],
                                                          t_prev_best[obs_idx][parent_task])
                                        
                    # create bid for new observation
                    new_bid = Bid(
                        parent_task,
                        state.agent_name,
                        n_obs_best[obs_idx][parent_task],
                        bid_value,
                        bid_value, 
                        state.agent_name,
                        obs.t_start,
                        state.t,
                        main_measurement=obs.task.instrument_name
                    )

                    # add to list of new bids
                    new_bids[parent_task].append(new_bid)

            # intialize tracker for accepted bids
            accepted_bids : Dict[GenericObservationTask, Dict[Bid, bool]] = defaultdict(lambda: defaultdict(bool))

            # compare new bids with existing bids in results
            for parent_task,bids in new_bids.items():
                for new_bid in bids:
                    # get matching bids 
                    try:
                        existing_bid : Bid = self.results[parent_task][new_bid.n_obs]
                    except IndexError:
                        existing_bid = None

                    # check if bid outperforms existing bid
                    if existing_bid is None or new_bid > existing_bid:
                        # if new bids outperform the existing bids, check constraints
                        accepted_bids[parent_task][new_bid] = True
                    # check if it is an earlier observation
                    elif new_bid.t_img < existing_bid.t_img:
                        # if earlier observation, check constraint violation counters
                        x = 1 # TO-DO
                        raise NotImplementedError("Heuristic insertion bundle builder finalization not yet implemented.")
                    else:
                        accepted_bids[parent_task][new_bid] = False
                        raise NotImplementedError("Heuristic insertion bundle builder finalization not yet implemented.")

            #   if constraints are met, accept new path and bids. 
            #       Update bundle, results, and bids.
            #   else:
            #       continue?

            # if they do not outperform, check if bids are for earlier observations
            #   if they are for earlier observations, check constraint violation counters
            #       if counters allow, accept new path and bids
            #       else: continue

            # check if all bids were accepted
            if all([all(accepted.values()) for accepted in accepted_bids.values()]):
                # update proposed path
                proposed_path = best_path

                # update current path utility
                current_path_utility = best_path_utility

                # update proposed bundle and bids
                for parent_task,bids in new_bids.items():
                    for new_bid in bids:
                        # add to proposed bids
                        proposed_bids[parent_task][new_bid.n_obs] = new_bid.copy()

                        # TODO check if task was bid on before in the bundle building phase
                        
                        # update results
                        try:
                            existing_bid : Bid = self.results[parent_task][new_bid.n_obs]
                            self.results[parent_task][new_bid.n_obs] = existing_bid.update(new_bid, state.t)
                        except IndexError:
                            self.results[parent_task].append(new_bid.copy())
                
                # add new observations to proposed bundle
                obs_dict = {obs.task: n_obs_best[best_path.index(obs)] for obs in best_path_changes}
                proposed_bundle.append((proposed_task, obs_dict))        

                # -------------------------------
                # DEBUG PRINTOUTS
                if self._debug:
                    self._log_results('RESULTS (DURING BUNDLE-BUILDING PHASE)', state, self.results)
                    # self._log_bundle('BUNDLE (DURING BUNDLE-BUILDING PHASE)', state, self.bundle)
                    x = 1
                # -------------------------------               
        
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
                                        ) -> List[List[ObservationAction]]:
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

        # generate proposed paths
        proposed_paths : List[Tuple[List[ObservationAction], List[ObservationAction]]] = [
            # Option 1: Direct Insertion into existing path
            self._direct_insertion_into_path(state, specs, current_path, new_task, max_slew_rate, max_torque),

            # Option 2: Right-shifting existing path to accommodate new task
            self._right_shift_path_for_new_task(state, specs, current_path, new_task, max_slew_rate, max_torque),

            # Option 3: Replace conflicting task with new urgent task
            self._replace_conflicting_tasks_with_new_task(state, specs, current_path, new_task, max_slew_rate, max_torque),

            # TODO Option 4: Remove all conflicting tasks and insert new task
            # self._remove_conflicting_tasks_and_insert_new_task(state, specs, current_path, new_task, max_slew_rate, max_torque),
        ]

        # ensure new task was included in new paths
        assert not self._debug or all([(path is None or any([action.task == new_task for action in path])) for path,_ in proposed_paths]), \
              "New task not included in proposed paths."
        
        # ensure new tas was included in path changes
        assert not self._debug or all([(path is None or any([action.task == new_task for action in path_changes])) for path,path_changes in proposed_paths]), \
              "New task not included in proposed path changes."

        # return proposed paths and the respective observation times for the new task in said paths
        return [(path,path_changes) for path,path_changes in proposed_paths if path is not None]
        
    def _direct_insertion_into_path(self,
                                    state : SimulationAgentState,
                                    specs : object,
                                    current_path : List[ObservationAction],
                                    new_task : SpecificObservationTask,
                                    max_slew_rate : float,
                                    max_torque : float
                                ) -> Tuple[List[ObservationAction], List[ObservationAction]]:
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
        if t_img is None: return None,None # no time found; cannot insert new task into path

        # insert new observation into path
        ## create observation action for new task
        new_observation = ObservationAction(new_task.instrument_name, th_img, t_img, new_task.min_duration, new_task)

        ## create new path with inserted observation
        new_path = [action for action in current_path]
        new_path.append(new_observation)
        new_path = sorted(new_path, key=lambda action: action.t_start)
        
        # return new path if valid
        return new_path, [new_observation] if self.is_observation_path_valid(state, new_path, max_slew_rate, max_torque, specs) else None, None

    def _right_shift_path_for_new_task(self,
                                        state : SimulationAgentState,
                                        specs : object,
                                        current_path : List[ObservationAction],
                                        new_task : SpecificObservationTask,
                                        max_slew_rate : float,
                                        max_torque : float
                                    ) -> Tuple[List[ObservationAction], List[ObservationAction]]:
        """ Try to right-shift existing path to accommodate new task. """
        # check if path is empty
        if len(current_path) == 0: 
            # Current path is empty; cannot right-shift path for new task.
            return None, None

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
        path_changes = [new_observation]

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
                path_changes.append(shifted_observation)
                
            # else, task needs a later start time but is not feasible
            else: 
                # do not add this task and try to shift remaining tasks
                continue
                # return None, None # cannot right-shift path for new task
            
        # return new path if valid
        return new_path, path_changes if self.is_observation_path_valid(state, new_path, max_slew_rate, max_torque, specs) else (None, None)
    
    def _replace_conflicting_tasks_with_new_task(self,
                                                 state : SimulationAgentState,
                                                 specs : object,
                                                 current_path : List[ObservationAction],
                                                 new_task : SpecificObservationTask,
                                                 max_slew_rate : float,
                                                 max_torque : float
                                            ) -> Tuple[List[ObservationAction], List[ObservationAction]]:
        """ Try to replace conflicting tasks in existing path with new task. """
        # check if path is empty
        if len(current_path) == 0: 
            # Current path is empty; cannot replace conflicting tasks in path for new task.
            return None, None

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
                return new_path, [new_observation]

        # unable to accommodate new task by replacing conflicting observations
        return None, None

    """
    BUNDLE-BUILDING PHASE - Bid Generation Methods
    """
    def _assign_best_observations_and_revisit_times_to_proposed_path(self,
                                                                     state : SimulationAgentState,
                                                                    #  proposed_task : SpecificObservationTask,
                                                                     candidate_path : List[ObservationAction],
                                                                     path_changes : List[ObservationAction],
                                                                     specs : object,
                                                                     cross_track_fovs : dict,
                                                                     orbitdata : OrbitData,
                                                                     mission : Mission,
                                                                     observation_history : ObservationHistory
                                                                    ) -> Tuple[Dict[int, Dict[GenericObservationTask, int]], 
                                                                                Dict[int, Dict[GenericObservationTask, float]]]:
        """ Generate best observation numbers and revisit times for each observation in the proposed path. 
        
        ### Returns 
            - n_obs_best : Dict[int, Dict[GenericObservationTask, int]] - Best observation numbers for each observation in the proposed path.
            - t_prev_best : Dict[int, Dict[GenericObservationTask, float]] - Best previous observation times for each observation in the proposed path.
        """

        # extract modified specific tasks from path changes
        modified_specific_tasks : List[SpecificObservationTask] = [action.task for action in path_changes]   
        modified_parent_tasks = {parent_task for task in modified_specific_tasks 
                                 for parent_task in task.parent_tasks}

        # find observation time for proposed task in candidate path
        modified_parent_task_obs_times : Dict[GenericObservationTask, List[Tuple[float,str,float,SpecificObservationTask]]] \
                    = {parent_task : [
                        (action.t_start, state.agent_name, action.look_angle, action.task) 
                        for action in candidate_path 
                        if parent_task in action.task.parent_tasks
                    ] for parent_task in modified_parent_tasks}
        
        # initialize best observation numbers and previous observation times
        n_obs_best : Dict[GenericObservationTask, list[str]] = defaultdict(list)
        t_img_best : Dict[GenericObservationTask, list[float]] = defaultdict(list)
        t_prev_best : Dict[GenericObservationTask, list[float]] = defaultdict(list)
        vals_best : Dict[GenericObservationTask, list[float]] = defaultdict(list)

        # find best observation sequences for each parent task
        for parent_task in modified_parent_tasks:
            # assume parent task has been considered in results
            assert parent_task in self.results, f"Parent task {parent_task} not being bid on by any agent; cannot generate bids."
            
            # get all possible observation opportunities from results
            available_obs_times : list[Tuple[float,str,SpecificObservationTask]] = \
                  [(bid.t_img,bid.bidder,None,None) for bid in self.results[parent_task] if bid.winning_bidder != state.agent_name]

            # include proposed task imaging time 
            available_obs_times.extend(modified_parent_task_obs_times[parent_task])

            # sort by observation time
            available_obs_times.sort(key=lambda x: x[0])

            # collect feasible sequences
            feasible_sequences = self._find_feasible_observation_sequences_for_task(state, parent_task, available_obs_times)

            # initialize search for best sequence
            best_value = 0.0

            # find sequence that maximizes value for this agent
            for obs_names,obs_times,obs_look_angles,obs_tasks in feasible_sequences:
                # initiate sequence value tracker
                seq_values = []
                t_prev_seq = []
                is_sequence_valid = True

                # evaluate sequence value for this agent
                for n_obs,(agent_name,t_obs,look_angle,spec_task) in enumerate(zip(obs_names,obs_times,obs_look_angles,obs_tasks)):
                    # assume specific task was defined
                    assert isinstance(spec_task, SpecificObservationTask), "Specific task for observation not defined."

                    # get observation number and previous observation time
                    t_prev = obs_times[n_obs-1] if n_obs > 0 else np.NINF
                    
                    # get observation value
                    if agent_name != state.agent_name: 
                        # observation is to be performed by another agent; 
                        #   get matching bid for this observation
                        matching_bid : Bid = self.results[parent_task][n_obs]

                        # ensure matching bid is from correct agent
                        assert matching_bid.bidder == agent_name, \
                            "Matching bid bidder does not match agent assigned to observation."
                        assert abs(matching_bid.t_img - t_obs) <= self.EPS, \
                            "Matching bid observation time does not match assigned observation time."
                        
                        # get observation value from winning bid
                        obs_value = matching_bid.winning_bid

                    else:
                        # observation is to be performed by this agent;
                        #   estimate task value for this observation
                        obs_value = self._estimate_task_value(parent_task,
                                                            spec_task.instrument_name,
                                                            look_angle, 
                                                            t_obs,
                                                            spec_task.min_duration,
                                                            specs, 
                                                            cross_track_fovs,
                                                            orbitdata,
                                                            mission,
                                                            observation_history,
                                                            n_obs,
                                                            t_prev
                                                            )
                        
                        # compare against existing bids for this observation number

                        # if no existing bid, accept if 
                        #   1) proposed observation value is positive 

                        # if there is an existing bid, accept if either:
                        #   1) currently winning bid and proposed observation value is positive 
                        #   2) outperforms existing bid
                        #   3) proposed observation earlier observation time and optimistic bidding counter allows it
                    
                    #     # accept bid for further consideration if:
                    #     # 1) bid for this observation number does not exist yet and observation value is positive
                    #     if n_obs >= len(self.results[parent_task]):
                    #         if obs_value <= 0.0:
                    #             # observation does not benefit this agent; skip this sequence
                    #             is_sequence_valid = False
                    #             break
                    #     else:
                    #         # get existing bid for this observation number
                    #         existing_bid : Bid = self.results[parent_task][n_obs]
                            

                            
                    #         # 2) I'm currently winning this observation number
                    #         currently_winning = (existing_bid.winning_bidder == state.agent_name)

                    #         # 3) this observation outperforms existing bid for this observation number
                    #         outperforms_existing_bid = (obs_value > existing_bid.winning_bid)

                    #         # 4) this observation is earlier than existing bid for this observation number 
                    #         #       and optimistic bidding counter allows it
                    #         can_be_optimistic = (t_obs < existing_bid.t_img 
                    #                              and self.optimistic_bidding_counters[parent_task][n_obs] > 0)
                            
                    #         # if none of the acceptance criteria are met, skip this sequence
                    #         if not (currently_winning 
                    #                 or outperforms_existing_bid 
                    #                 or can_be_optimistic):
                    #             # skip this sequence
                    #             is_sequence_valid = False
                    #             break
                    
                    # # check if sequence is still valid
                    # if not is_sequence_valid: break

                    # # if observation value is non-positive, skip this sequence
                    # if obs_value <= 0.0:
                    #     is_sequence_valid = False
                    #     break

                    # accumulate sequence value
                    seq_values.append(obs_value)     
                    t_prev_seq.append(t_prev)      

                # skip to next sequence if current sequence is invalid
                if not is_sequence_valid: continue

                # compute total sequence value
                total_seq_value = sum(seq_values)                

                # check if this sequence outperforms previous best
                if total_seq_value > best_value:
                    # update best sequence value and sequence
                    best_value = total_seq_value

                    # update to best sequence trackers
                    n_obs_best[parent_task] = list(range(len(obs_names)))
                    t_img_best[parent_task] = obs_times
                    t_prev_best[parent_task] = t_prev_seq 
                    vals_best[parent_task] = seq_values  
            
        # compile best observation numbers and previous observation times for each observation in candidate path
        n_obs_candidate = [dict() for _ in candidate_path]
        t_prev_candidate = [dict() for _ in candidate_path]
        
        for obs_idx,obs in enumerate(candidate_path):
            for parent_task in obs.task.parent_tasks:
                # check if best sequences were found for this parent task
                if parent_task in n_obs_best:
                    # ensure there are enough observations in best sequence
                    assert n_obs_best[parent_task], f"No best observation number found for parent task {parent_task}."

                    # assign best observation number and previous observation time
                    n_obs_candidate[obs_idx][parent_task] = n_obs_best[parent_task].pop(0)
                    t_prev_candidate[obs_idx][parent_task] = t_prev_best[parent_task].pop(0)
                else:
                    # no best sequence found for this parent task; use existing bids from results
                    # get matching bid for this observation task
                    matching_bids = [bid for bid in self.results[parent_task]
                                    if abs(bid.t_img - obs.t_start) <= self.EPS]
                    
                    assert matching_bids, \
                        "Matching bid for observation in path not found in results. Was assigned without updating results."
                    assert len(matching_bids) <= 1, \
                        "There should be at most one matching bid for the current time step."

                    matching_bid : Bid = matching_bids.pop()

                    # get previous matching observations for this task
                    prev_bids = [bid for bid in self.results[parent_task]
                                if bid.t_img < obs.t_start]
                    
                    # update previous observation counts
                    n_obs_candidate[obs_idx][parent_task] = matching_bid.n_obs
                    t_prev_candidate[obs_idx][parent_task] = max((bid.t_img for bid in prev_bids), default=np.NINF)

        # TODO assure all best observation numbers have been assigned
        for obs_idx,obs in enumerate(candidate_path):
            for parent_task in obs.task.parent_tasks:
                assert parent_task in n_obs_candidate[obs_idx], \
                    f"Observation number for parent task {parent_task} not assigned for observation at time {obs.t_start}."
                assert parent_task in t_prev_candidate[obs_idx], \
                    f"Previous observation time for parent task {parent_task} not assigned for observation at time {obs.t_start}."

        # TODO assure assignments are consistent within candidate path

        # return updated observation numbers and previous observation times
        return n_obs_candidate, t_prev_candidate

    def _find_feasible_observation_sequences_for_task(self,
                                                      state : SimulationAgentState,
                                                      parent_task : GenericObservationTask,
                                                      available_obs : List[tuple]
                                                    ) -> List[Tuple[List[str], List[float]]]:
        """ Find feasible observation number sequences for a given task. """
        # initialize feasible sequence tracker
        feasible_sequences = []

        # count minimum sequence length; use number of occurrences of this agent in available observation times
        min_seq_length = sum(1 for _,agent_name,*_ in available_obs if agent_name == state.agent_name)

        # create dfs queue
        dfs_queue = deque()

        # seed dfs with initial observations from this agent
        for obs in available_obs: dfs_queue.append([obs])

        # perform dfs to find feasible sequences
        while dfs_queue:
            # pop current sequence from stack
            current_sequence = dfs_queue.pop()

            # check for available successors
            successors = [obs for obs in available_obs
                          if obs[0] > current_sequence[-1][0]]
            
            # base case: no more successors to add to sequence
            if not successors:
                # check if min length was achieved 
                if len(current_sequence) < min_seq_length: continue # min length not met; skip to next sequence 

                # ensure min number of observations from this agent are included
                n_obs_this_agent = sum(1 for _,agent_name,_,_ in current_sequence if agent_name == state.agent_name)
                if n_obs_this_agent != min_seq_length: continue # min observations from this agent not met; skip to next sequence

                # add to feasible sequences
                obs_names = [agent_name for _,agent_name,_,_ in current_sequence]
                obs_times = [t_img for t_img,_,_,_ in current_sequence]
                obs_look_angles = [look_angle for _,_,look_angle,_ in current_sequence]
                obs_tasks = [spec_task for _,_,_,spec_task in current_sequence]
                feasible_sequences.append((obs_names, obs_times, obs_look_angles, obs_tasks))               
            
            for obs_next in successors:
                # unpack proposed successor observation
                t_next,agent_next,*_ = obs_next
                
                # if successor is from another agent, check consistency with results
                if agent_next != state.agent_name:
                    # check successor's bid for this observation exists
                    n_obs_next = len(current_sequence)

                    if len(self.results[parent_task]) <= n_obs_next:
                        # matching no bids exist for this observation number; cannot add successor
                        continue
                    elif self.results[parent_task][n_obs_next].bidder != agent_next:
                        # bid for this observation number is from another agent; cannot add successor
                        continue
                    elif abs(self.results[parent_task][n_obs_next].t_img - t_next) > self.EPS:
                        # bid for this observation number does not match successor; cannot add successor
                        continue
                    # --- IGNORE ---

                # create new sequence with successor added
                new_sequence = [obs for obs in current_sequence] + [obs_next]

                # add new sequence to dfs stack
                dfs_queue.append(new_sequence)            

        return feasible_sequences
