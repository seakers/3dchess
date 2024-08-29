from logging import Logger
from queue import Queue
from numpy import Inf
from orbitpy.util import Spacecraft
from tqdm import tqdm
import matplotlib.pyplot as plt
import concurrent.futures

from dmas.clocks import ClockConfig
from dmas.utils import runtime_tracker
from dmas.clocks import *

from chess3d.agents.orbitdata import OrbitData, TimeInterval
from chess3d.agents.planning.plan import Plan
from chess3d.agents.planning.planners.rewards import RewardGrid
from chess3d.agents.states import *
from chess3d.agents.actions import *
from chess3d.agents.science.requests import *
from chess3d.agents.states import SimulationAgentState
from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.planning.planner import AbstractPreplanner
from chess3d.messages import *

class DynamicProgrammingPlanner(AbstractPreplanner):
    def __init__(self, 
                 sharing : bool = False,
                 horizon: float = np.Inf, 
                 period : float = np.Inf, 
                 debug : bool = False,
                 logger: Logger = None
                 ) -> None:
        super().__init__(horizon, period, debug, logger)

        # toggle for sharing plans
        self.sharing = sharing 

    def populate_adjacency_matrix(self, 
                                  state : SimulationAgentState, 
                                  specs : object,
                                  access_opportunities : list, 
                                  ground_points : dict,
                                  adjacency : list,
                                  j : int,
                                  pbar : tqdm = None):
               
        # get current observation
        curr_opportunity : tuple = access_opportunities[j]
        lat_curr,lon_curr = ground_points[curr_opportunity[0]][curr_opportunity[1]]
        curr_target = [lat_curr,lon_curr,0.0]

        # get any possibly prior observation
        prev_opportunities : list[tuple] = [prev_opportunity for prev_opportunity in access_opportunities
                                            if prev_opportunity[3].end <= curr_opportunity[3].end
                                            and prev_opportunity != curr_opportunity
                                            ]

        # construct adjacency matrix
        for prev_opportunity in prev_opportunities:
            prev_opportunity : tuple

            # get previous observation opportunity's target 
            lat_prev,lon_prev = ground_points[prev_opportunity[0]][prev_opportunity[1]]
            prev_target = [lat_prev,lon_prev,0.0]

            # assume earliest observation time from previous observation
            earliest_prev_observation = ObservationAction(prev_opportunity[2], 
                                                        prev_target, 
                                                        prev_opportunity[5][0], 
                                                        prev_opportunity[4][0])
            
            # check if observation can be reached from previous observation
            adjacent = any([self.is_observation_path_valid(state, 
                                                        specs, 
                                                        [earliest_prev_observation,
                                                            ObservationAction(curr_opportunity[2], 
                                                                            curr_target, 
                                                                            curr_opportunity[5][k], 
                                                                            curr_opportunity[4][k])])
                            for k in range(len(curr_opportunity[4]))
                            ])                               

            # update adjacency matrix
            adjacency[access_opportunities.index(prev_opportunity)][j] = adjacent

            # update progress bar
            if pbar is not None: pbar.update(1)

    def _schedule_observations(self, 
                               state: SimulationAgentState, 
                               specs: object, 
                               reward_grid: RewardGrid,
                               clock_config: ClockConfig, 
                               orbitdata: OrbitData = None
                               ) -> list:
        
        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Naive planner not yet implemented for agents of type `{type(state)}.`')
        elif not isinstance(specs, Spacecraft):
            raise ValueError(f'`specs` needs to be of type `{Spacecraft}` for agents with states of type `{SatelliteAgentState}`')

        t_0 = time.perf_counter()
        t_prev = t_0

        # compile access times for this planning horizon
        access_opportunities, ground_points = self.calculate_access_opportunities(state, specs, orbitdata)
        access_opportunities : list; ground_points : dict

        # sort by observation time
        access_opportunities.sort(key=lambda a: a[3])

        # initiate results arrays
        t_imgs = [np.NAN for _ in access_opportunities]
        th_imgs = [np.NAN for _ in access_opportunities]
        rewards = [0.0 for _ in access_opportunities]
        cumulative_rewards = [0.0 for _ in access_opportunities]
        preceeding_observations = [np.NAN for _ in access_opportunities]
        adjancency = [[False for _ in access_opportunities] for _ in access_opportunities]

        t_1 = time.perf_counter() - t_prev
        t_prev = time.perf_counter()

        # create adjancency matrix      
        with tqdm(total=len(access_opportunities), 
                    desc=f'{state.agent_name}-PLANNER: Generating Adjacency Matrix', 
                    leave=False) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                for j in range(len(access_opportunities)): 
                    executor.submit(self.populate_adjacency_matrix, state, specs, access_opportunities, ground_points, adjancency, j, pbar)
               
        t_2 = time.perf_counter() - t_prev
        t_prev = time.perf_counter()

        # calculate optimal path and update results
        n_prev_opp = []
        for j in tqdm(range(len(access_opportunities)), 
                      desc=f'{state.agent_name}-PLANNER: Calculating Optimal Path',
                      leave=False):
            
            # get current observation opportunity
            curr_opportunity : tuple = access_opportunities[j]
            lat,lon = ground_points[curr_opportunity[0]][curr_opportunity[1]]
            curr_target = [lat,lon,0.0]

            # get any possibly prior observation
            prev_opportunities : list[tuple] = [prev_opportunity for prev_opportunity in access_opportunities
                                                if adjancency[access_opportunities.index(prev_opportunity)][j]
                                                and not np.isnan(th_imgs[access_opportunities.index(prev_opportunity)])
                                                ]
            n_prev_opp.append(len(prev_opportunities))

            # calculate all possible observation actionss for this observation opportunity
            possible_observations = [ObservationAction( curr_opportunity[2], 
                                                        curr_target, 
                                                        curr_opportunity[5][k], 
                                                        curr_opportunity[4][k])
                                         for k in range(len(curr_opportunity[4]))]

            # update observation time and look angle
            if not prev_opportunities: # there are no previous possible observations
                possible_observations = [possible_observation for possible_observation in possible_observations
                                         if self.is_observation_path_valid(state, specs, [possible_observation])]

                if possible_observations:
                    t_imgs[j] = possible_observations[0].t_start
                    th_imgs[j] = possible_observations[0].look_angle
                    rewards[j] = reward_grid.estimate_reward(possible_observations[0])
            
            for prev_opportunity in prev_opportunities: # there are previous possible observations
                # get previous observation opportunity
                prev_opportunity : tuple
                i = access_opportunities.index(prev_opportunity)

                # create previous observation action using known information 
                lat_prev,lon_prev = ground_points[prev_opportunity[0]][prev_opportunity[1]]
                prev_target = [lat_prev,lon_prev,0.0]

                prev_observation = ObservationAction(prev_opportunity[2], 
                                                    prev_target, 
                                                    th_imgs[i], 
                                                    t_imgs[i])
                
                # get possible observation actions from the current observation opportuinty
                possible_observations = [possible_observation for possible_observation in possible_observations
                                        if self.is_observation_path_valid(state, specs, [prev_observation, possible_observation])]

                # check if an observation is possible
                if possible_observations: 
                    # update imaging time, look angle, and reward
                    t_imgs[j] = possible_observations[0].t_start
                    th_imgs[j] = possible_observations[0].look_angle
                    rewards[j] = reward_grid.estimate_reward(possible_observations[0])

                    # update results
                    if cumulative_rewards[i] + rewards[j] > cumulative_rewards[j] and not np.isnan(t_imgs[i]):
                        cumulative_rewards[j] += cumulative_rewards[i] + rewards[j]
                        preceeding_observations[j] = i
            
        t_3 = time.perf_counter() - t_prev
        t_prev = time.perf_counter()

        # extract sequence of observations from results
        visited_observation_opportunities = set()

        while preceeding_observations and np.isnan(preceeding_observations[-1]):
            preceeding_observations.pop()
        observation_sequence = [len(preceeding_observations)-1] if preceeding_observations else []
        
        while preceeding_observations and not np.isnan(preceeding_observations[observation_sequence[-1]]):
            prev_observation_index = preceeding_observations[observation_sequence[-1]]
            
            if prev_observation_index in visited_observation_opportunities:
                raise AssertionError('invalid sequence of observations generated by DP. Cycle detected.')
            
            visited_observation_opportunities.add(prev_observation_index)
            observation_sequence.append(prev_observation_index)

        observation_sequence.sort()

        t_4 = time.perf_counter() - t_prev
        t_prev = time.perf_counter()

        observations = []
        for j in observation_sequence:
            grid_index, gp_indx, instrument_name, *_ = access_opportunities[j]
            lat,lon = ground_points[grid_index][gp_indx]
            target = [lat,lon,0.0]
            observation = ObservationAction(instrument_name, target, th_imgs[j], t_imgs[j])
            observations.append(observation)
        
        if not self.is_observation_path_valid(state, specs, observations):
            y = self.is_observation_path_valid(state, specs, observations)
            x = 1

        t_5 = time.perf_counter() - t_prev
        t_f = time.perf_counter() - t_0
        return observations
    
    def _schedule_broadcasts(self, state: SimulationAgentState, observations: list, orbitdata: OrbitData) -> list:
        broadcasts =  super()._schedule_broadcasts(state, observations, orbitdata)

        if self.sharing:
            # TODO schedule reward grid broadcasts
            pass
            
        return broadcasts