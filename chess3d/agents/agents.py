import time
import logging
import asyncio

import pandas as pd
from tqdm import tqdm

from chess3d.agents.agent import RealtimeAgent, SimulatedAgent
from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.planning.tasks import MonitoringObservationTask, ObservationTask


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

        self.tasks : list[ObservationTask]= [
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
    
    async def setup(self) -> None:
        # get initial set of tasks from groud targets
        return
    
    # @runtime_tracker
    # def calculate_access_opportunities(self, 
    #                                    state : SimulationAgentState, 
    #                                    specs : Spacecraft,
    #                                    ground_points : dict,
    #                                    orbitdata : OrbitData
    #                                 ) -> dict:
    #     # define planning horizon
    #     t_start = state.t
    #     t_end = self.plan.t_next+self.horizon
    #     t_index_start = t_start / orbitdata.time_step
    #     t_index_end = t_end / orbitdata.time_step

    #     # compile coverage data
    #     orbitdata_columns : list = list(orbitdata.gp_access_data.columns.values)
    #     raw_coverage_data = [(t_index*orbitdata.time_step, *_)
    #                          for t_index, *_ in orbitdata.gp_access_data.values
    #                          if t_index_start <= t_index <= t_index_end]
    #     raw_coverage_data.sort(key=lambda a : a[0])

    #     # initiate accestimes 
    #     access_opportunities = {}
        
    #     for data in tqdm(raw_coverage_data, 
    #                      desc='PREPLANNER: Compiling access opportunities', 
    #                      leave=False):
    #         t_img = data[orbitdata_columns.index('time index')]
    #         grid_index = data[orbitdata_columns.index('grid index')]
    #         gp_index = data[orbitdata_columns.index('GP index')]
    #         instrument = data[orbitdata_columns.index('instrument')]
    #         look_angle = data[orbitdata_columns.index('look angle [deg]')]

    #         # only consider ground points from the pedefined list of important groundopints
    #         if grid_index not in ground_points or gp_index not in ground_points[grid_index]:
    #             continue
            
    #         # initialize dictionaries if needed
    #         if grid_index not in access_opportunities:
    #             access_opportunities[grid_index] = {}
                
    #         if gp_index not in access_opportunities[grid_index]:
    #             access_opportunities[grid_index][gp_index] = {instr.name : [] 
    #                                                     for instr in specs.instrument}

    #         # compile time interval information 
    #         found = False
    #         for interval, t, th in access_opportunities[grid_index][gp_index][instrument]:
    #             interval : Interval
    #             t : list
    #             th : list

    #             if (   (t_img - orbitdata.time_step) in interval 
    #                 or (t_img + orbitdata.time_step) in interval):
    #                 interval.extend(t_img)
    #                 t.append(t_img)
    #                 th.append(look_angle)
    #                 found = True
    #                 break                        

    #         if not found:
    #             access_opportunities[grid_index][gp_index][instrument].append([Interval(t_img, t_img), [t_img], [look_angle]])

    #     # convert to `list`
    #     access_opportunities = [    (grid_index, gp_index, instrument, interval, t, th)
    #                                 for grid_index in access_opportunities
    #                                 for gp_index in access_opportunities[grid_index]
    #                                 for instrument in access_opportunities[grid_index][gp_index]
    #                                 for interval, t, th in access_opportunities[grid_index][gp_index][instrument]
    #                             ]
                
    #     # return access times and grid information
    #     return access_opportunities
    