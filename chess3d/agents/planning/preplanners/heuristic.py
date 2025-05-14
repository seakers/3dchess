from logging import Logger
from typing import Dict
from orbitpy.util import Spacecraft

from dmas.utils import runtime_tracker
from dmas.clocks import *
from tqdm import tqdm

from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.planning.tasks import ObservationHistory, SchedulableObservationTask
from chess3d.agents.states import *
from chess3d.agents.actions import *
from chess3d.agents.science.requests import *
from chess3d.agents.states import SimulationAgentState
from chess3d.agents.orbitdata import OrbitData
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
        schedulable_tasks : list[SchedulableObservationTask] = self.sort_tasks_by_heuristic(schedulable_tasks, specs, cross_track_fovs, orbitdata, observation_history)

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
            
            # calculate task observation angle
            th_img = np.average((task.slew_angles.left, task.slew_angles.right))
            
            # check if there is overlap with previous scheduled observation
            potential_overlap = [Interval(observation.t_start,  observation.t_end) 
                                 for observation in observations
                                 if observation.t_start in task.accessibility 
                                  or observation.t_end in task.accessibility]
            if any([overlap.overlaps(task.accessibility)
                    for overlap in potential_overlap]):
                continue

            # compare with previous scheduled observation
            actions_prev = [observation for observation in observations
                            if observation.t_end <= task.accessibility.left]
            
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
            t_img = task.accessibility.left
            prev_action_feasible = self.is_observation_feasible(state,
                                                                t_img,
                                                                th_img,
                                                                t_prev,
                                                                th_prev,
                                                                max_slew_rate,
                                                                max_torque,
                                                                cross_track_fovs[task.instrument_name]
                                                                )

            # compare with future scheduled observation
            actions_next = [observation for observation in observations
                            if observation.t_start >= task.accessibility.right]
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
            t_img = task.accessibility.right
            next_action_feasible = self.is_observation_feasible(state,
                                                                t_next,
                                                                th_next,
                                                                t_img,
                                                                th_img,
                                                                max_slew_rate,
                                                                max_torque,
                                                                cross_track_fovs[task.instrument_name]
                                                                )
            
            # check if the observation is feasible
            if prev_action_feasible and next_action_feasible:
                targets = [[lat,lon,0.0] for parent_task in task.parent_tasks
                                        for lat,lon,*_ in parent_task.targets]
                objectives = [parent_task.objective for parent_task in task.parent_tasks]
                action = ObservationAction(task.instrument_name, 
                                           targets, 
                                           objectives,
                                           th_img, 
                                           task.accessibility.left, 
                                           task.accessibility.span())
                observations.append(action)
        
        # sort by start time
        observations_sorted = sorted(observations, key=lambda a : a.t_start)

        assert self.no_redundant_observations(state, observations_sorted, orbitdata)

        return observations_sorted

    # @abstractmethod
    def sort_tasks_by_heuristic(self, tasks : list, specs : Spacecraft, cross_track_fovs : dict, orbitdata : OrbitData, observation_history : ObservationHistory) -> list:
        def reward_heuristic(task : SchedulableObservationTask) -> float:
            """ Heuristic function to sort tasks by their heuristic value. """
            # calculate task priority
            priority = np.prod([parent_task.objective.priority  
                                for parent_task in task.parent_tasks])
            # calculate task duration
            duration = task.accessibility.span()
            # calculate task start time
            t_start = task.accessibility.left
            
            # estimate observation look angle
            th_img = np.average((task.slew_angles.left, task.slew_angles.right))
            
            t_l = task.accessibility.left / orbitdata.time_step - 0.5
            t_u = task.accessibility.right / orbitdata.time_step + 0.5

            # calculate task performance
            observation_performances = [row 
                                       for _,row in orbitdata.gp_access_data.iterrows()
                                       if row['instrument'] == task.instrument_name
                                       and any([(row['GP index'] == gp_index and row['grid index'] == grid_index)
                                                 for parent_task in task.parent_tasks
                                                 for *_,grid_index,gp_index in parent_task.targets])
                                       and t_l <= row['time index'] <= t_u
                                       and abs(row['look angle [deg]'] - th_img) <= cross_track_fovs[task.instrument_name] / 2
                                       ]
            if not observation_performances:
                return 0.0, -priority, -duration, t_start
            
            instrument_specs = [instr for instr in specs.instrument
                                if instr.name == task.instrument_name][0]
            observation_performance = observation_performances[0]
            grid_index = int(observation_performance['grid index'])
            gp_index = int(observation_performance['GP index'])

            prev_obs = observation_history.get_observation_history(grid_index, gp_index)

            measurement = {
                "instrument" : task.instrument_name,    
                "t_start" : task.accessibility.left,
                "t_end" : task.accessibility.right,
                "horizontal_spatial_resolution" : observation_performance['ground pixel cross-track resolution [m]'],
                "accuracy" : 50.0,
                "n_obs" : prev_obs.n_obs,
                "t_prev" : prev_obs.t_last,
            }

            if 'vnir' in task.instrument_name.lower():
                if 'hyp' in task.instrument_name.lower():
                    measurement['spectral_resolution'] = "Hyperspectral"
                elif 'multi' in task.instrument_name.lower():
                    measurement['spectral_resolution'] = "Multispectral"
                else:
                    measurement['spectral_resolution'] = "None"

            for parent_task in task.parent_tasks:
                performance = parent_task.objective.eval_performance(measurement)
                x = 1

            # calculate total task reward
            

            return -0.0, -priority, -duration, t_start

        # TEMP SOLUTION schedule based on largest duration and earliest start time
        return sorted(tasks, key=reward_heuristic)

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
        return dt_maneuver <= dt_obs or abs(dt_maneuver - dt_obs) <= 1e-6
    
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
                            abs(target_prev[0] - target_prev[0]) <= 1e-3
                            and abs(target_prev[1] - target_prev[1]) <= 1e-3
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