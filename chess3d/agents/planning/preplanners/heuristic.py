from logging import Logger
from orbitpy.util import Spacecraft

from dmas.utils import runtime_tracker
from dmas.clocks import *
from tqdm import tqdm

from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.planning.rewards import RewardGrid
from chess3d.agents.states import *
from chess3d.agents.actions import *
from chess3d.agents.science.requests import *
from chess3d.agents.states import SimulationAgentState
from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.planning.planner import AbstractPreplanner
from chess3d.messages import *

class HeuristicInsertionPlanner(AbstractPreplanner):
    """ Schedules observations iteratively based on the highest heuristic-scoring and feasible access point """

    def __init__(self, 
                 horizon: float = np.Inf, 
                 period: float = np.Inf, 
                 points : int = np.Inf, 
                #  sharing : bool = True,
                 debug: bool = False, 
                 logger: Logger = None
                 ) -> None:
        super().__init__(horizon, 
                         period, 
                        #  sharing, 
                         debug, 
                         logger)
        self.points = points

    @runtime_tracker
    def _schedule_observations(self, 
                               state: SimulationAgentState, 
                               specs : object,
                               reward_grid : RewardGrid, 
                               __,
                               orbitdata: OrbitData = None
                               ) -> list:
        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Naive planner not yet implemented for agents of type `{type(state)}.`')
        elif not isinstance(specs, Spacecraft):
            raise ValueError(f'`specs` needs to be of type `{Spacecraft}` for agents with states of type `{SatelliteAgentState}`')

        # compile access times for this planning horizon
        ground_points : dict = self.get_ground_points(orbitdata, self.points)
        access_times : list = self.calculate_access_opportunities(state, specs, ground_points, orbitdata)

        # sort by observation time
        access_times : list = self.sort_accesses(access_times, ground_points, reward_grid)

        # compile list of instruments available in payload
        payload : dict = {instrument.name: instrument for instrument in specs.instrument}
        
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self.collect_fov_specs(specs)
        
        # get pointing agility specifications
        adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
        if adcs_specs is None: raise ValueError('ADCS component specifications missing from agent specs object.')

        max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
        if max_slew_rate is None: raise ValueError('ADCS `maxRate` specification missing from agent specs object.')

        max_torque = float(adcs_specs['maxTorque']) if adcs_specs.get('maxTorque', None) is not None else None
        if max_torque is None: raise ValueError('ADCS `maxTorque` specification missing from agent specs object.')

        # generate plan
        observations : list[ObservationAction] = []

        for access_time in tqdm(access_times,
                                desc=f'{state.agent_name}-PLANNER: Pre-Scheduling Observations', 
                                leave=False):
            # get next available access interval
            grid_index, gp_index, instrument, _, t, th = access_time
            lat,lon = ground_points[grid_index][gp_index]
            target = [lat,lon,0.0]

            # check if agent has the payload to peform observation
            if instrument not in payload:
                continue

            # compare to previous measurement 
            actions_prev = [observation for observation in observations
                            if observation.t_end <= t[-1]]
            
            if actions_prev:
                # compare with previous scheduled observation
                action_prev : ObservationAction = actions_prev[-1]
                
                t_prev = action_prev.t_end
                th_prev = action_prev.look_angle
            else:
                # no prior observation exists, compare with current state
                t_prev = state.t
                th_prev = state.attitude[0]
        
            # find if feasible observation times exist
            feasible_obs = [(t[i], th[i]) 
                            for i in range(len(t))
                            if self.is_observation_feasible(state, 
                                                            t[i], th[i], 
                                                            t_prev, th_prev, 
                                                            max_slew_rate, max_torque, 
                                                            cross_track_fovs[instrument])]
            feasible_obs.sort(key=lambda a : a[0])

            while feasible_obs:
                # is feasible; create observation action with the earliest observation
                t_img, th_img = feasible_obs.pop(0)
                action = ObservationAction(instrument, target, th_img, t_img)

                # check if another observation was already scheduled at this time
                if not observations:
                    observations.append(action)
                    break
                else:
                    action_prev : ObservationAction = observations[-1]
                    if abs(action_prev.t_end - action.t_start) > 1e-3 and action_prev.t_end <= action.t_start:
                        observations.append(action)
                        break

        assert self.no_redundant_observations(state, observations, orbitdata)

        return observations
    
    def sort_accesses(self, access_times : list, ground_points : dict, reward_grid : RewardGrid) -> list:
        """ Sorts available accesses by a defined heuristic metric. By default it is sorted by expected reward """
        
        def __estimate_earliest_reward(access_time : tuple):
            # calculate earliest reward of a given access 
            grid_index, gp_index, instrument, _, t, th = access_time
            lat,lon = ground_points[grid_index][gp_index]
            target = [lat,lon,0.0]
            observation = ObservationAction(instrument, target, th[0], t[0])
            
            return reward_grid.estimate_reward(observation)
        
        # self.__print_access_times(access_times)
        access_times.sort(key=__estimate_earliest_reward)
        # self.__print_access_times(access_times)

        return access_times
            
    def __print_access_times(self, access_times : list):
        """ printouts for debugging purposes """
        for access_time in access_times:
            print(access_time)
        print()
    
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

                if (
                    abs(observation_curr.target[0] - observation_prev.target[0]) <= 1e-3
                    and abs(observation_curr.target[1] - observation_prev.target[1]) <= 1e-3
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