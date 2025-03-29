import logging
import asyncio

from dmas.network import NetworkConfig
import pandas as pd

from agents.agent import RealtimeAgent, SimulatedAgent
from chess3d.agents.science.module import ScienceModule
from chess3d.agents.planning.module import PlanningModule
from chess3d.agents.science.requests import MeasurementRequest
from chess3d.agents.states import SimulationAgentState

from orbitpy.util import Spacecraft

# class GroundStationAgent(SimulationAgent):
#     def __init__(self, 
#                  agent_name: str, 
#                  results_path: str, 
#                  manager_network_config: NetworkConfig, 
#                  agent_network_config: NetworkConfig, 
#                  initial_state: SimulationAgentState, 
#                  specs: object, 
#                  planning_module: PlanningModule = None, 
#                  science_module: ScienceModule = None, 
#                  level: int = logging.INFO, 
#                  logger: logging.Logger = None
#                  ) -> None:
        
#         super().__init__(agent_name, 
#                          results_path, 
#                          manager_network_config, 
#                          agent_network_config, 
#                          initial_state, 
#                          specs, 
#                          planning_module, 
#                          science_module, 
#                          level, 
#                          logger)


#     async def setup(self) -> None:
#         # nothing to setup
#         return
    
#     async def live(self):
#         await asyncio.sleep(5e-2) # wait for others to connect 
        
#         await super().live()

#     async def teardown(self) -> None:
#         await super().teardown()

#         # print measurement requests from the ground
#         headers = ['id', 'lat','lon','alt','severity','obs_types','t_start','t_end','t_corr']
#         data = []
#         for req in self.measurement_reqs:
#             req : MeasurementRequest
#             line = [    
#                         req.id.split('-')[0],
#                         req.target[0],
#                         req.target[1],
#                         req.target[2],
#                         req.severity,
#                         f"{req.observation_types}",
#                         req.t_start,
#                         req.t_end,
#                         req.t_corr
#                     ]
#             data.append(line)

#         # log and save results
#         summary_df = pd.DataFrame(data, columns=headers)
#         self.log(f"\nMEASUREMENT REQUESTS:\n{str(summary_df)}\n\n", level=logging.WARNING)
#         summary_df.to_csv(f"{self.results_path}/../gpRequests.csv", index=False)

# class UAVAgent(SimulationAgent):
#     def __init__(   
#                     self, 
#                     agent_name: str,    
#                     results_path: str, 
#                     manager_network_config: NetworkConfig, 
#                     agent_network_config: NetworkConfig, 
#                     initial_state: SimulationAgentState, 
#                     payload: list, 
#                     planning_module: PlanningModule = None, 
#                     science_module: ScienceModule = None, 
#                     level: int = logging.INFO, 
#                     logger: logging.Logger = None
#                 ) -> None:

#         super().__init__(
#                         agent_name, 
#                         results_path, 
#                         manager_network_config, 
#                         agent_network_config, 
#                         initial_state, 
#                         payload, 
#                         planning_module, 
#                         science_module, 
#                         level, 
#                         logger
#                     )

#     async def setup(self) -> None:
#         # nothing to setup
#         return
    

class RealtimeSatelliteAgent(RealtimeAgent):
    def __init__(   self, 
                    agent_name: str, 
                    results_path: str, 
                    manager_network_config: NetworkConfig, 
                    agent_network_config: NetworkConfig,
                    initial_state: SimulationAgentState, 
                    specs: Spacecraft, 
                    planning_module : PlanningModule,
                    science_module: ScienceModule = None, 
                    level: int = logging.INFO, 
                    logger: logging.Logger = None
                    ) -> None:

        
        super().__init__(agent_name, 
                        results_path, 
                        manager_network_config, 
                        agent_network_config, 
                        initial_state, 
                        specs,
                        planning_module, 
                        science_module, 
                        level, 
                        logger)

    async def setup(self) -> None:
        # nothing to setup
        return
    
class SatelliteAgent(SimulatedAgent):
    def __init__(self, 
                 agent_name, 
                 results_path, 
                 agent_network_config, 
                 manager_network_config, 
                 initial_state, 
                 specs, 
                 orbitdata,
                 processor = None, 
                 preplanner = None, 
                 replanner = None, 
                 reward_grid = None,
                 level=logging.INFO, 
                 logger=None):
        
        super().__init__(agent_name, 
                         results_path, 
                         agent_network_config, 
                         manager_network_config, 
                         initial_state, 
                         specs, 
                         processor, 
                         preplanner, 
                         replanner, 
                         reward_grid, 
                         level, 
                         logger)
        
        self.orbitdata = orbitdata
    
    async def setup(self) -> None:
        # nothing to setup
        return
    