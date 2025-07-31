from dmas.modules import ClockConfig
import numpy as np
from chess3d.agents.actions import ObservationAction
from chess3d.orbitdata import OrbitData
from chess3d.agents.planning.preplanners.decentralized.earliest import EarliestAccessPlanner
from chess3d.agents.states import SatelliteAgentState, SimulationAgentState
from chess3d.messages import ClockConfig

from orbitpy.util import Spacecraft

from dmas.utils import runtime_tracker

class NadirPointingPlaner(EarliestAccessPlanner):
    """ Only points agents to """

    def is_observation_feasible(self, 
                                state : SimulationAgentState,
                                t_img: float, 
                                th_img: float, 
                                t_prev: float, 
                                th_prev: float, 
                                max_slew_rate: float,
                                max_torque: float, 
                                fov: float
                                ) -> bool:
        
        return (abs(th_img - state.attitude[0]) <= fov / 2.0 # is valid if no manuver is needed 
                and t_img >= t_prev)                         # is valid if it is done after the previous observation
    
    @runtime_tracker
    def _schedule_maneuvers(self, state: SimulationAgentState, specs: Spacecraft, observations: list, clock_config: ClockConfig, orbitdata: OrbitData = None) -> list:
        # schedule all travel maneuvers
        maneuvers = []

        # compile instrument field of view specifications   
        cross_track_fovs = self.collect_fov_specs(specs)

        # get pointing agility specifications
        adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
        if adcs_specs is None: raise ValueError('ADCS component specifications missing from agent specs object.')

        max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
        if max_slew_rate is None: raise ValueError('ADCS `maxRate` specification missing from agent specs object.')

        # ensure no attitude manuvers are required in plan
        assert self.is_maneuver_path_valid(state, specs, observations, maneuvers, max_slew_rate, cross_track_fovs)

        # return travel manuvers
        return maneuvers
    
    @runtime_tracker
    def is_observation_path_valid(self, 
                                  state : SimulationAgentState, 
                                  specs : object,
                                  observations : list
                                  ) -> bool:
        """ Checks if a given sequence of observations can be performed by a given agent """
        # return True
        if isinstance(state, SatelliteAgentState) and isinstance(specs, Spacecraft):

            # get pointing agility specifications
            adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
            if adcs_specs is None: raise ValueError('ADCS component specifications missing from agent specs object.')

            max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
            if max_slew_rate is None: raise ValueError('ADCS `maxRate` specification missing from agent specs object.')

            max_torque = float(adcs_specs['maxTorque']) if adcs_specs.get('maxTorque', None) is not None else None
            if max_torque is None: raise ValueError('ADCS `maxTorque` specification missing from agent specs object.')
            
            # compile instrument field of view specifications   
            cross_track_fovs : dict = self.collect_fov_specs(specs)

            # check if every observation can be reached from the prior measurement
            for j in range(len(observations)):

                # estimate the state of the agent at the given measurement
                observation_j : ObservationAction = observations[j]
                th_j = observation_j.look_angle
                t_j = observation_j.t_start
                fov = cross_track_fovs[observation_j.instrument_name]

                # compare to prior measurements and state
                th_i = state.attitude[0]
                
                if j > 0: # there was a prior observation performed
                    # estimate the state of the agent at the prior mesurement
                    observation_i : ObservationAction = observations[j-1]
                    t_i = observation_i.t_end

                else: # there was prior measurement
                    # use agent's current state as previous state
                    t_i = state.t                

                
                # check if desired instrument is contained within the satellite's specifications
                if observation_j.instrument_name not in [instrument.name for instrument in specs.instrument]:
                    return False 
                
                assert th_j != np.NAN and th_i != np.NAN # TODO: add case where the target is not visible by the agent at the desired time according to the precalculated orbitdata

                # estimate maneuver time betweem states
                dth_maneuver = abs(th_j - th_i)

                # calculate time between measuremnets
                dt_measurements = t_j - t_i

                # check if observation sequence is correct 
                if dt_measurements < 0.0:
                    return False

                # fov constraint: check if the target is within the instrument's fov
                if dth_maneuver > fov / 2:
                    # target is not in the fov of the instrument; flag current observation plan as unfeasible for rescheduling
                    return False              
                                            
            # if all measurements passed the check; observation path is valid
            return True
        else:
            raise NotImplementedError(f'Observation path validity check for agents with state type {type(state)} not yet implemented.')
        