import time
import logging
import asyncio

import pandas as pd
from tqdm import tqdm

from chess3d.agents.agent import RealtimeAgent, SimulatedAgent
from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.planning.tasks import MonitoringObservationTask, ObservationHistory, SchedulableObservationTask


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

        self.tasks : list[SchedulableObservationTask]= [
            MonitoringObservationTask(
                                        self.mission.name,
                                        objective,
                                        (lat, lon, grid_index, gp_index),
                                        1.0,
                                        self.orbitdata.duration * 24 * 3600
                                    )
            for grid in tqdm(self.orbitdata.grid_data,
                         desc="SATELLITE: Generating monitoring tasks from ground targets",
                         leave=False)
            for lat,lon,grid_index,gp_index in grid.values
            for objective in self.mission
        ]

        self.observation_history = ObservationHistory(orbitdata, mission)
    
    async def setup(self) -> None:
        # get initial set of tasks from groud targets
        return
    