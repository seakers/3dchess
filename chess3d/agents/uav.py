import logging

from dmas.network import NetworkConfig

from agents.agent import SimulationAgent
from chess3d.agents.science.module import ScienceModule
from chess3d.agents.planning.module import PlanningModule
from chess3d.agents.states import SimulationAgentState


class UAVAgent(SimulationAgent):
    def __init__(   
                    self, 
                    agent_name: str,    
                    results_path: str, 
                    manager_network_config: NetworkConfig, 
                    agent_network_config: NetworkConfig, 
                    initial_state: SimulationAgentState, 
                    payload: list, 
                    planning_module: PlanningModule = None, 
                    science_module: ScienceModule = None, 
                    level: int = logging.INFO, 
                    logger: logging.Logger = None
                ) -> None:

        super().__init__(
                        agent_name, 
                        results_path, 
                        manager_network_config, 
                        agent_network_config, 
                        initial_state, 
                        payload, 
                        planning_module, 
                        science_module, 
                        level, 
                        logger
                    )

    async def setup(self) -> None:
        # nothing to setup
        return