from logging import Logger
from chess3d.agents.planning.planner import *


class MILPObservationScheduler:
    def schedule_observations(self, 
                              states : dict,
                              orbitdata: dict
                              ) -> list:
        """
        Schedule observations using MILP (Mixed Integer Linear Programming).
        
        :param states: Current states of all simulation agents.
        :param orbitdata: Optional orbit data for scheduling.
        :return: List of scheduled observations.
        """
        # Implementation of MILP scheduling logic goes here
        pass

class SingleSatMILP(AbstractPreplanner):
    def __init__(self, 
                 orbitdata_dir : str,
                 horizon = np.Inf, 
                 period = np.Inf, 
                 debug = False, 
                 logger = None):
        super().__init__(horizon, period, debug, logger)

        # load observation data
        self.orbitdata : dict = OrbitData.from_directory(orbitdata_dir) if orbitdata_dir is not None else None

    @runtime_tracker
    def _schedule_observations(self, 
                               state: SimulationAgentState, 
                               _: ClockConfig, 
                               orbitdata: OrbitData = None
                               ) -> list:
        
        observation_schedules : dict = MILPObservationScheduler