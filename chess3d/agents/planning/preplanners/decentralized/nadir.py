from dmas.utils import runtime_tracker
import numpy as np
from orbitpy.util import Spacecraft

from chess3d.agents.actions import ObservationAction
from chess3d.agents.planning.preplanners.decentralized.earliest import EarliestAccessPlanner
from chess3d.agents.states import SimulationAgentState, SatelliteAgentState

class NadirPointingPlanner(EarliestAccessPlanner):
    """ Only points agents in the downward direction """

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
    def _schedule_maneuvers(self, *args) -> list:
        return []
    
    @runtime_tracker
    def is_observation_path_valid(self, 
                                  state : SimulationAgentState, 
                                  observations : list,
                                  max_slew_rate : float = None,
                                  max_torque : float = None,
                                  specs : object = None,
                                ) -> bool:
        """ Checks if a given sequence of observations can be performed by a given agent """
        # return True
        if isinstance(state, SatelliteAgentState):
            # validate inputs
            assert isinstance(specs, Spacecraft), 'Agent specs must be provided as a `Spacecraft` object from `orbitpy` package.'
            
            # get pointing agility specifications                
            if max_slew_rate is None or max_torque is None:
                if specs is None: raise ValueError('Either `specs` or both `max_slew_rate` and `max_torque` must be provided.')

                max_slew_rate, max_torque = self._collect_agility_specs(specs)

            # validate agility specifications
            if max_slew_rate is None: raise ValueError('ADCS `maxRate` specification missing from agent specs object.')
            if max_torque is None: raise ValueError('ADCS `maxTorque` specification missing from agent specs object.')
            assert max_slew_rate > 0.0
            # assert max_torque > 0.0

            # compile name of instruments onboard spacecraft
            instruments = [instrument.name for instrument in specs.instrument]

            # compile instrument field of view specifications   
            cross_track_fovs : dict = self._collect_fov_specs(specs)

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
                if observation_j.instrument_name not in instruments:
                    return False 

                assert not np.isnan(th_j) and not np.isnan(th_i) # TODO: add case where the target is not visible by the agent at the desired time according to the precalculated orbitdata

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
        