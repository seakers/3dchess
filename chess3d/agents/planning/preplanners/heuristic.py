from logging import Logger
from typing import Dict
from orbitpy.util import Spacecraft

from dmas.utils import runtime_tracker
from dmas.clocks import *
from tqdm import tqdm

from chess3d.orbitdata import OrbitData
from chess3d.agents.planning.tasks import EventObservationTask, ObservationHistory, SchedulableObservationTask
from chess3d.agents.states import *
from chess3d.agents.actions import *
from chess3d.agents.science.requests import *
from chess3d.agents.states import SimulationAgentState
from chess3d.orbitdata import OrbitData
from chess3d.agents.planning.planner import AbstractPreplanner
from chess3d.messages import *
from chess3d.utils import Interval

class HeuristicInsertionPlanner(AbstractPreplanner):
    """ Schedules observations iteratively based on the highest heuristic-scoring and feasible access point """

    @runtime_tracker
    def _schedule_observations(self, 
                               state : SimulationAgentState, 
                               specs : object, 
                               _ : ClockConfig, 
                               orbitdata : OrbitData, 
                               schedulable_tasks : list,
                               observation_history : ObservationHistory
                               ) -> list:
        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Naive planner not yet implemented for agents of type `{type(state)}.`')
        elif not isinstance(specs, Spacecraft):
            raise ValueError(f'`specs` needs to be of type `{Spacecraft}` for agents with states of type `{SatelliteAgentState}`')

        # compile list of instruments available in payload
        payload : dict = {instrument.name: instrument for instrument in specs.instrument}
        
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self.collect_fov_specs(specs)
        
        # sort tasks by heuristic
        schedulable_tasks : list[SchedulableObservationTask] = self.sort_tasks_by_heuristic(state, schedulable_tasks, specs, cross_track_fovs, orbitdata, observation_history)

        # get pointing agility specifications
        adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
        assert adcs_specs, 'ADCS component specifications missing from agent specs object.'

        max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
        assert max_slew_rate, 'ADCS `maxRate` specification missing from agent specs object.'

        max_torque = float(adcs_specs['maxTorque']) if adcs_specs.get('maxTorque', None) is not None else None
        assert max_torque, 'ADCS `maxTorque` specification missing from agent specs object.'

        # generate plan
        observations : list[ObservationAction] = []

        for task in tqdm(schedulable_tasks,
                         desc=f'{state.agent_name}-PLANNER: Pre-Scheduling Observations', 
                         leave=False):
            
            # check if agent has the payload to peform observation
            if task.instrument_name not in payload:
                continue
            
            # set task observation angle
            th_img = np.average((task.slew_angles.left, task.slew_angles.right))
            
            # check if there is overlap with previous scheduled observation
            potential_overlap = [Interval(observation.t_start,  observation.t_end) 
                                 for observation in observations
                                 if observation.t_start in task.accessibility 
                                  or observation.t_end in task.accessibility]
            if any([overlap.overlaps(task.accessibility) for overlap in potential_overlap]):
                continue

            # find any previous scheduled observation
            actions_prev = [observation for observation in observations
                            if observation.t_start - 1e-6 <= task.accessibility.left]
            
            if actions_prev:
                # sort by end time 
                actions_prev.sort(key=lambda a : a.t_end)

                action_prev : ObservationAction = actions_prev[-1]
                
                t_prev = action_prev.t_end
                th_prev = action_prev.look_angle
            else:
                # no prior observation exists, compare with current state
                t_prev = state.t
                th_prev = state.attitude[0]

            # check if there is a potential previous observation conflict
            prev_action_feasible = self.is_observation_feasible(state,
                                                                task.accessibility.left,
                                                                th_img,
                                                                t_prev,
                                                                th_prev,
                                                                max_slew_rate,
                                                                max_torque,
                                                                cross_track_fovs[task.instrument_name]
                                                                )

            # find any future scheduled observation
            actions_next = [observation for observation in observations
                            if task.accessibility.right - 1e-6 <= observation.t_end]
            if actions_next:
                # sort by start times
                actions_next.sort(key=lambda a : a.t_start)

                action_next : ObservationAction = actions_next[0]
                
                t_next = action_next.t_start
                th_next = action_next.look_angle
            else:
                # no future observation exists, compare with current state
                t_next = task.accessibility.right
                th_next = np.average((task.slew_angles.left, task.slew_angles.right))

            # check if there is a potential future observation conflict
            next_action_feasible = self.is_observation_feasible(state,
                                                                t_next,
                                                                th_next,
                                                                task.accessibility.right,
                                                                th_img,
                                                                max_slew_rate,
                                                                max_torque,
                                                                cross_track_fovs[task.instrument_name]
                                                                )
            
            # check if the observation is feasible
            if prev_action_feasible and next_action_feasible:
                targets = [[lat,lon,0.0] for parent_task in task.parent_tasks
                                        for lat,lon,*_ in parent_task.targets]
                objectives = list({parent_task.objective for parent_task in task.parent_tasks})
                action = ObservationAction(task.instrument_name, 
                                           targets, 
                                           objectives,
                                           th_img, 
                                           task.accessibility.left, 
                                           task.accessibility.span())
                observations.append(action)
        
        # sort by start time
        observations_sorted = sorted(observations, key=lambda a : a.t_start)

        # assert self.no_redundant_observations(state, observations_sorted, orbitdata)
        assert self.is_observation_path_valid(state, specs, observations_sorted)

        return observations_sorted
    
    @runtime_tracker
    def sort_tasks_by_heuristic(self, state : SimulationAgentState, tasks : list, specs : Spacecraft, cross_track_fovs : dict, orbitdata : OrbitData, observation_history : ObservationHistory) -> list:
        """ Sorts tasks by heuristic value """
        if tasks:
            # estimate maximum number of tasks in the planning horizon
            min_task_duration = min([task.accessibility.span() for task in tasks])
            max_number_tasks = int(self.horizon / min_task_duration) if min_task_duration > 0 else -1
            
            # reduce number of tasks to be scheduled by using estimated max number of tasks 
            tasks.sort(key=lambda x: x.accessibility.span(),reverse=True)
            tasks = tasks[:max_number_tasks]

        # calculate heuristic value for each task
        values = [[self.calc_heuristic(task, specs, cross_track_fovs, orbitdata, observation_history)] 
                  for task in tqdm(tasks, desc=f"{state.agent_name}-PREPLANNER: Calculating heuristic values", leave=False)]
        for i in range(len(values)): values[i].insert(0,tasks[i])
        
        # sort tasks by heuristic value
        sorted_data = sorted(values, key=lambda x: x[1:])
        
        # return sorted tasks
        return [task for task, *_ in sorted_data]
    
    @runtime_tracker
    def calc_heuristic(self,
                        task : SchedulableObservationTask, 
                        specs : Spacecraft, 
                        cross_track_fovs : dict, 
                        orbitdata : OrbitData, 
                        observation_history : ObservationHistory
                        ) -> float:
        """ Heuristic function to sort tasks by their heuristic value. """
        # calculate task priority
        priority = np.prod([parent_task.objective.priority  
                            for parent_task in task.parent_tasks])
        # calculate task duration
        duration = task.accessibility.span()
        
        # calculate task start time
        t_start = task.accessibility.left
        
        # calculate task reward
        task_reward = self.calc_task_reward(task, specs, cross_track_fovs, orbitdata, observation_history)

        # return to sort using: highest task reward >> highest priority >> longest duration >> earliest start time
        return -task_reward, -priority, -duration, t_start

    def is_observation_feasible(self, 
                                state : SimulationAgentState,
                                t_img : float, 
                                th_img : float, 
                                t_prev : float, 
                                th_prev : float, 
                                max_slew_rate : float, 
                                max_torque : float,
                                fov : float
                                ) -> bool:
        """ compares previous observation """
        
        # calculate inteval between observations
        dt_obs = t_img - t_prev

        # calculate maneuver angle 
        dth_img = abs(th_img - th_prev) 

        # estimate maneuver time
        dt_maneuver = dth_img / max_slew_rate

        # check slew constraint
        return dt_obs >= 0 and (dt_maneuver <= dt_obs or abs(dt_maneuver - dt_obs) <= 1e-6)
    
        #TODO check torque constraint
    
    def no_redundant_observations(self, 
                                 state : SimulationAgentState, 
                                 observations : list,
                                 orbitdata : OrbitData
                                 ) -> bool:
        if isinstance(state, SatelliteAgentState):
            for j in range(len(observations)):
                i = j - 1

                if i < 0: # there was no prior observation performed
                    continue                

                observation_prev : ObservationAction = observations[i]
                observation_curr : ObservationAction = observations[j]

                for target_prev in observation_prev.targets:
                    for target_curr in observation_curr.targets:
                        if (
                            abs(target_curr[0] - target_prev[0]) <= 1e-3
                            and abs(target_curr[1] - target_prev[1]) <= 1e-3
                            and (observation_curr.t_start - observation_prev.t_end) <= orbitdata.time_step):
                            return False
            
            return True

        else:
            raise NotImplementedError(f'Measurement path validity check for agents with state type {type(state)} not yet implemented.')
        
    @runtime_tracker
    def _schedule_broadcasts(self, 
                             state: SimulationAgentState, 
                             observations : list, 
                             orbitdata: OrbitData) -> list:
        
        # do not schedule broadcasts
        return super()._schedule_broadcasts(state, observations, orbitdata)