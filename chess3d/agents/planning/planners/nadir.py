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
        maneuvers = super()._schedule_maneuvers(state, specs, observations, clock_config, orbitdata)

        # ensure no attitude manuvers exist in plan
        assert all([not isinstance(action, ManeuverAction) for action in maneuvers])

        # return travel manuvers
        return maneuvers