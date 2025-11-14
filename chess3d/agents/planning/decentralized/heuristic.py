from typing import List
from orbitpy.util import Spacecraft

from dmas.utils import runtime_tracker
from dmas.clocks import *
from tqdm import tqdm

from chess3d.agents.planning.periodic import AbstractPeriodicPlanner
from chess3d.agents.planning.tracker import ObservationHistory
from chess3d.mission.mission import Mission
from chess3d.orbitdata import OrbitData
from chess3d.agents.planning.tasks import SpecificObservationTask
from chess3d.agents.states import *
from chess3d.agents.actions import *
from chess3d.agents.science.requests import *
from chess3d.agents.states import SimulationAgentState
from chess3d.orbitdata import OrbitData
from chess3d.messages import *

class HeuristicInsertionPlanner(AbstractPeriodicPlanner):
    """ Schedules observations iteratively based on the highest heuristic-scoring and feasible access point """

    @runtime_tracker
    def _schedule_observations(self, 
                               state : SimulationAgentState, 
                               specs : object, 
                               _ : ClockConfig, 
                               orbitdata : OrbitData, 
                               schedulable_tasks : list,
                               mission : Mission,
                               observation_history : ObservationHistory
                               ) -> list:
        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Naive planner not yet implemented for agents of type `{type(state)}.`')
        elif not isinstance(specs, Spacecraft):
            raise ValueError(f'`specs` needs to be of type `{Spacecraft}` for agents with states of type `{SatelliteAgentState}`')

        # compile list of instruments available in payload
        payload : dict = {instrument.name: instrument for instrument in specs.instrument}
        
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self._collect_fov_specs(specs)
        
        # sort tasks by heuristic
        schedulable_tasks : list[SpecificObservationTask] = self.__sort_tasks_by_heuristic(state, schedulable_tasks, specs, cross_track_fovs, orbitdata, mission, observation_history)

        # get pointing agility specifications
        adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
        assert adcs_specs, 'ADCS component specifications missing from agent specs object.'

        max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
        assert max_slew_rate, 'ADCS `maxRate` specification missing from agent specs object.'

        max_torque = float(adcs_specs['maxTorque']) if adcs_specs.get('maxTorque', None) is not None else None
        assert max_torque, 'ADCS `maxTorque` specification missing from agent specs object.'

        # generate plan
        plan_sequence : list[tuple[SpecificObservationTask, ObservationAction]] = []

        for task in tqdm(schedulable_tasks,
                         desc=f'{state.agent_name}-PLANNER: Pre-Scheduling Observations', 
                         leave=False):
            
            # check if agent has the payload to peform observation
            if task.instrument_name not in payload: continue

            # get previous and future observation actions' info
            th_prev,t_prev,d_prev,th_next,t_next,d_next \
                = self.__get_previous_and_future_observation_info(state, task, plan_sequence, max_slew_rate)
            
            # set task observation angle
            th_img = np.average((task.slew_angles.left, task.slew_angles.right))

            # calculate maneuver times
            m_prev = abs(th_prev - th_img) / max_slew_rate if max_slew_rate else 0.0
            m_next = abs(th_img - th_next) / max_slew_rate if max_slew_rate else 0.0
            
            # select task imaging time and duration # TODO room for improvement? Currently aims for earliest and shortest observation possible
            t_img = max(t_prev + d_prev + m_prev, task.accessibility.left)
            d_img = task.min_duration
            
            # check if the observation fits within the task's accessibility window
            if t_img + d_img not in task.accessibility: continue

            # check if the observation is feasible
            prev_action_feasible : bool = (t_prev + d_prev + m_prev <= t_img - 1e-6)
            next_action_feasible : bool = (t_img + d_img + m_next   <= t_next - 1e-6)
            if prev_action_feasible and next_action_feasible:
                # check if task is mutually exclusive with any already scheduled tasks
                if any(task.is_mutually_exclusive(task_j) for task_j,_ in plan_sequence): continue
                
                # create observation action
                action = ObservationAction(task.instrument_name, 
                                           th_img, 
                                           t_img, 
                                           d_img,
                                           task)

                # add to plan sequence
                plan_sequence.append((task, action))

        # return sorted by start time
        return sorted([action for _,action in plan_sequence], key=lambda a : a.t_start)
    
    def __get_previous_and_future_observation_info(self, 
                                                 state : SimulationAgentState, 
                                                 task : SpecificObservationTask, 
                                                 plan_sequence : list, 
                                                 max_slew_rate : float) -> tuple:
        
        # get latest previously scheduled observation
        action_prev : ObservationAction = self.__get_previous_observation_action(task, plan_sequence)

        # get values from previous action
        if action_prev:    
            th_prev = action_prev.look_angle
            t_prev = action_prev.t_end
            d_prev = action_prev.t_end - t_prev
            
        else:
            # no prior observation exists; compare with current state
            th_prev = state.attitude[0]
            t_prev = state.t
            d_prev = 0.0
        
        # get next earliest scheduled observation
        action_next : ObservationAction = self.__get_next_observation_action(task, plan_sequence)

        # get values from next action
        if action_next:
            th_next = action_next.look_angle
            t_next = action_next.t_start
            d_next = action_next.t_end - t_next
        else:
            # no future observation exists; compare with current task
            th_next = np.average((task.slew_angles.left, task.slew_angles.right))
            t_next = task.accessibility.right
            d_next = 0.0

        return th_prev, t_prev, d_prev, th_next, t_next, d_next

    def __get_previous_observation_action(self, task : SpecificObservationTask, plan_sequence : list) -> ObservationAction:
        """ find any previously scheduled observation """
        # set types
        observations : list[ObservationAction] = [observation for _,observation in plan_sequence]

        # filter for previous actions
        actions_prev : list[ObservationAction] = [observation for observation in observations
                                                 if observation.t_end - 1e-6 <= task.accessibility.right]

        # return latest observation action
        return max(actions_prev, key=lambda a: a.t_end) if actions_prev else None
    
    def __get_next_observation_action(self, task : SpecificObservationTask, plan_sequence : list) -> ObservationAction:
         # set types
        observations : list[ObservationAction] = [observation for _,observation in plan_sequence]

        # filter for next actions
        actions_next = [observation for observation in observations
                        if task.accessibility.left - 1e-6 <= observation.t_start]
        
        # return earliest observation action
        return min(actions_next, key=lambda a: a.t_start) if actions_next else None

    @runtime_tracker
    def __sort_tasks_by_heuristic(self, 
                                state : SimulationAgentState, 
                                tasks : List[SpecificObservationTask], 
                                specs : Spacecraft, 
                                cross_track_fovs : dict, 
                                orbitdata : OrbitData, 
                                mission : Mission, 
                                observation_history : ObservationHistory) -> list:
        """ Sorts tasks by heuristic value """
        
        # return if no tasks to schedule
        if not tasks: return tasks

        # estimate maximum number of tasks in the planning horizon
        min_task_duration = min([task.accessibility.span() for task in tasks])
        max_number_tasks = int(self.horizon / min_task_duration) if min_task_duration > 0 else -1
        
        # reduce number of tasks to be scheduled by using estimated max number of tasks 
        tasks.sort(key=lambda x: x.accessibility.span(),reverse=True)

        # calculate heuristic value for each task up to the maximum number of tasks
        heuristic_vals = [(task, self._calc_heuristic(task, specs, cross_track_fovs, orbitdata, mission, observation_history)) 
                          for task in tqdm(tasks[:max_number_tasks], 
                                           desc=f"{state.agent_name}-PREPLANNER: Calculating heuristic values", 
                                           leave=False)
                            ]
                
        # sort tasks by heuristic value
        sorted_data = sorted(heuristic_vals, key=lambda x: x[1:])
        
        # return sorted tasks
        return [task for task,*_ in sorted_data]
    
    @runtime_tracker
    def _calc_heuristic(self,
                        task : SpecificObservationTask, 
                        specs : Spacecraft, 
                        cross_track_fovs : dict, 
                        orbitdata : OrbitData, 
                        mission : Mission,
                        observation_history : ObservationHistory
                        ) -> tuple:
        """ Heuristic function to sort tasks by their heuristic value. """
        # calculate task priority
        priority = task.get_priority()
        
        # calculate task duration
        duration = task.accessibility.span()
        
        # choose task earliest possible start time
        t_start = task.accessibility.left

        # choose shortest allowable task duration
        duration = task.min_duration 

        # calculate task reward
        task_reward = self.estimate_task_value(task, t_start, duration, specs, cross_track_fovs, orbitdata, mission, observation_history)

        # return to sort using: highest task reward >> highest priority >> longest duration >> earliest start time
        return -task_reward, -priority, -duration, t_start
    

    @runtime_tracker
    def _schedule_broadcasts(self, 
                             state: SimulationAgentState, 
                             observations : list, 
                             orbitdata: OrbitData) -> list:
        
        # do not schedule broadcasts
        return super()._schedule_broadcasts(state, observations, orbitdata)