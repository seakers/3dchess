from logging import Logger
from chess3d.agents.planning.planner import *


class MILPplanner(AbstractPreplanner):
    def __init__(self, 
                 payload: list, 
                 horizon: float = np.Inf,
                 period: float = np.Inf, 
                 logger: Logger = None
                 ) -> None:
        super().__init__(payload, horizon, period, logger)

    @runtime_tracker
    def _schedule_observations(self, 
                               state: SimulationAgentState, 
                               _: ClockConfig, 
                               orbitdata: OrbitData = None
                               ) -> list:
        pass