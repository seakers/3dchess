from chess3d.agents.planning.planners.naive import NaivePlanner
from chess3d.agents.states import SimulationAgentState


class NadirPointingPlaner(NaivePlanner):

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
        return abs(th_img - state.attitude[0]) <= fov / 2.0