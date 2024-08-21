from dmas.modules import ClockConfig
from chess3d.agents.actions import ManeuverAction
from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.planning.planners.naive import NaivePlanner
from chess3d.agents.states import SimulationAgentState
from chess3d.messages import ClockConfig


class NadirPointingPlaner(NaivePlanner):
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
    
    def _schedule_maneuvers(self, state: SimulationAgentState, specs: object, observations: list, clock_config: ClockConfig, orbitdata: OrbitData = None) -> list:
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