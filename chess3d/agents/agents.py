import time
import logging
import asyncio

from matplotlib.pyplot import grid
import pandas as pd
from tqdm import tqdm

from chess3d.agents.agent import RealtimeAgent, SimulatedAgent
from chess3d.agents.planning.tracker import ObservationHistory
from chess3d.mission.objectives import DefaultMissionObjective
from chess3d.mission.requirements import GridTargetSpatialRequirement, SpatialRequirement
from chess3d.orbitdata import OrbitData
from chess3d.agents.planning.tasks import DefaultMissionTask, GenericObservationTask


class RealtimeGroundStationAgent(RealtimeAgent):
    async def setup(self) -> None:
        # nothing to setup
        return
    
    async def live(self):
        await asyncio.sleep(5e-2) # wait for others to connect 
        
        await super().live()

    async def teardown(self) -> None:
        await super().teardown()

class GroundStationAgent(SimulatedAgent):
    async def setup(self) -> None:
        # nothing to setup
        return
    
    async def live(self):
        await asyncio.sleep(5e-2) # wait for others to connect 
        
        await super().live()

    async def teardown(self) -> None:
        await super().teardown()

        # # print measurement requests from the ground
        # headers = ['id', 'lat','lon','alt','severity','obs_types','t_start','t_end','t_corr']
        # data = []
        # for req in self.measurement_reqs:
        #     req : MeasurementRequest
        #     line = [    
        #                 req.id.split('-')[0],
        #                 req.target[0],
        #                 req.target[1],
        #                 req.target[2],
        #                 req.severity,
        #                 f"{req.observation_types}",
        #                 req.t_start,
        #                 req.t_end,
        #                 req.t_corr
        #             ]
        #     data.append(line)

        # # log and save results
        # summary_df = pd.DataFrame(data, columns=headers)
        # self.log(f"\nMEASUREMENT REQUESTS:\n{str(summary_df)}\n\n", level=logging.WARNING)
        # summary_df.to_csv(f"{self.results_path}/../gpRequests.csv", index=False)    

class RealtimeSatelliteAgent(RealtimeAgent):
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
                 orbitdata : OrbitData,
                 mission,
                 processor = None, 
                 preplanner = None, 
                 replanner = None, 
                 level=logging.INFO, 
                 logger=None):
        
        super().__init__(agent_name, 
                         results_path, 
                         agent_network_config, 
                         manager_network_config, 
                         initial_state, 
                         specs,
                         mission,
                         processor, 
                         preplanner, 
                         replanner, 
                         level, 
                         logger)
        
        self.orbitdata : OrbitData = orbitdata
            
    async def setup(self) -> None:
        # initialize observation history
        self.observation_history = ObservationHistory(self.orbitdata)

        # create monitoring tasks from mission objectives
        monitoring_tasks : list[GenericObservationTask] = [
            DefaultMissionObjective(
                        objective.parameter,
                        requirements=[req for req in objective.requirements]
                    )
            for objective in self.mission
            if isinstance(objective, DefaultMissionObjective)
        ]

        # add to list of known tasks
        self.tasks.extend([ task for task in monitoring_tasks ])

        return
    