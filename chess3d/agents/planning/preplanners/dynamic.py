from itertools import repeat
from logging import Logger
import os
from typing import Dict
from orbitpy.util import Spacecraft
from tqdm import tqdm
import concurrent.futures

from dmas.clocks import ClockConfig
from dmas.utils import runtime_tracker
from dmas.clocks import *

from chess3d.agents.planning.plan import Plan, Preplan
# from chess3d.agents.planning.rewards import RewardGrid
from chess3d.agents.states import *
from chess3d.agents.actions import *
from chess3d.agents.science.requests import *
from chess3d.agents.states import SimulationAgentState
from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.planning.planner import AbstractPreplanner
from chess3d.messages import *

class DynamicProgrammingPlanner(AbstractPreplanner):
    def __init__(self, 
                #  sharing : bool = False,
                 horizon: float = np.Inf, 
                 period : float = np.Inf, 
                #  points : float = np.Inf,
                 debug : bool = False,
                 logger: Logger = None
                 ) -> None:
        super().__init__(horizon, 
                         period, 
                        #  sharing, 
                         debug, 
                         logger)
        
    @runtime_tracker
    def _schedule_observations(self, 
                               state: SimulationAgentState, 
                               specs: object, 
                               reward_grid,
                               _: ClockConfig, 
                               orbitdata: OrbitData = None
                               ) -> list:
        
        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Dynamic Programming planner not yet implemented for agents of type `{type(state)}.`')
        elif not isinstance(specs, Spacecraft):
            raise ValueError(f'`specs` needs to be of type `{Spacecraft}` for agents with states of type `{SatelliteAgentState}`')

        t_0 = time.perf_counter()
        t_prev = t_0

        # compile access times for this planning horizon
        ground_points : dict = self.get_ground_points(orbitdata)
        access_opportunities : list  = self.calculate_access_opportunities(state, specs, ground_points, orbitdata)

        # sort by observation time
        access_opportunities.sort(key=lambda a: a[3])

        # initiate results arrays
        t_imgs = [np.NAN for _ in access_opportunities]
        th_imgs = [np.NAN for _ in access_opportunities]
        rewards = [0.0 for _ in access_opportunities]
        cumulative_rewards = [0.0 for _ in access_opportunities]
        preceeding_observations = [np.NAN for _ in access_opportunities]
        # adjacency = [[False for _ in access_opportunities] for _ in access_opportunities]

        t_1 = time.perf_counter() - t_prev
        t_prev = time.perf_counter()

        # create adjancency matrix      
        adjacency = [ [access_opportunities.index(prev_opportunity) < access_opportunities.index(curr_opportunity) 
                            and self.is_observation_path_valid(state, specs, 
                                                                [ObservationAction( prev_opportunity[2], 
                                                                                    [*ground_points[prev_opportunity[0]][prev_opportunity[1]],0.0], 
                                                                                    prev_opportunity[5][0], 
                                                                                    prev_opportunity[4][0]),
                                                                ObservationAction( curr_opportunity[2], 
                                                                                    [*ground_points[curr_opportunity[0]][curr_opportunity[1]],0.0], 
                                                                                    curr_opportunity[5][-1], 
                                                                                    curr_opportunity[4][-1])]
                                                                ) 
                        
                        for curr_opportunity in access_opportunities]
                    for prev_opportunity in tqdm(access_opportunities, 
                                                desc=f'{state.agent_name}-PLANNER: Generating Adjacency Matrix', 
                                                leave=False)]
               
        t_2 = time.perf_counter() - t_prev
        t_prev = time.perf_counter()

        # calculate optimal path and update results
        for j in tqdm(range(len(access_opportunities)), 
                      desc=f'{state.agent_name}-PLANNER: Calculating Optimal Path',
                      leave=False):
            self.find_optimal_path(state, 
                                   specs, 
                                   reward_grid, 
                                   access_opportunities, 
                                   ground_points, 
                                   adjacency, 
                                   t_imgs, 
                                   th_imgs, 
                                   rewards, 
                                   cumulative_rewards, preceeding_observations, j)
            
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

        # create observation actions
        observations : list[ObservationAction] = []
        for j in observation_sequence:
            grid_index, gp_indx, instrument_name, *_ = access_opportunities[j]
            lat,lon = ground_points[grid_index][gp_indx]
            target = [lat,lon,0.0]
            observation = ObservationAction(instrument_name, target, th_imgs[j], t_imgs[j])
            observations.append(observation)
                
        t_5 = time.perf_counter() - t_prev
        t_f = time.perf_counter() - t_0

        # return observations
        return observations    

    @runtime_tracker
    def find_optimal_path(self, 
                          state : SimulationAgentState, 
                          specs : dict, 
                          reward_grid ,
                          access_opportunities : list, 
                          ground_points : list, 
                          adjancency : list, 
                          t_imgs : list, 
                          th_imgs : list,
                          rewards : list,
                          cumulative_rewards : list,
                          preceeding_observations : list,
                          j : int
                          ) -> tuple:
                   
        # get current observation opportunity
        curr_opportunity : tuple = access_opportunities[j]
        lat,lon = ground_points[curr_opportunity[0]][curr_opportunity[1]]
        curr_target = [lat,lon,0.0]

        # get any possibly prior observation            
        prev_opportunities : list[tuple] = [(i,access_opportunities[i]) for i in range(0,j)
                                            if adjancency[i][j] and not np.isnan(th_imgs[i]) ]

        # calculate all possible observation actions for this observation opportunity
        possible_observations = [ObservationAction( curr_opportunity[2], 
                                                    curr_target, 
                                                    curr_opportunity[5][k], 
                                                    curr_opportunity[4][k])
                                        for k in range(len(curr_opportunity[4]))]

        # update observation time and look angle
        if not prev_opportunities: # there are no previous possible observations
            # get possible observation action from the current observation opportunity
            feasible_observation = next((possible_observation for possible_observation in possible_observations
                                            if self.is_observation_path_valid(state, specs, [possible_observation])),
                                        None)
            
            # check if an observation is possible
            if feasible_observation is not None:
                # update imaging time, look angle, and reward
                t_imgs[j] = feasible_observation.t_start
                th_imgs[j] = feasible_observation.look_angle
                rewards[j] = reward_grid.estimate_reward(feasible_observation)

            return

        prev_observations : Dict[int, ObservationAction] = {i : ObservationAction(prev_opportunity[2], 
                                                                                    [*ground_points[prev_opportunity[0]][prev_opportunity[1]],0.0], 
                                                                                    th_imgs[i], 
                                                                                    t_imgs[i])
                                                            for i,prev_opportunity in prev_opportunities}

        for i,_ in prev_opportunities: # there are previous possible observations
            # get previous observation
            prev_observation : ObservationAction = prev_observations[i]
                            
            # check if an observation is possible
            feasible_observation = next((possible_observation for possible_observation in possible_observations
                                        if self.is_observation_path_valid(state, specs, [prev_observation, possible_observation])),
                                        None)
            
            # update imaging time, look angle, and reward
            if feasible_observation is not None:
                # estimate reward for observation
                reward = reward_grid.estimate_reward(feasible_observation)

                # update results only if cumulative reward is increased
                if cumulative_rewards[i] + reward > cumulative_rewards[j] and not np.isnan(t_imgs[i]):
                    t_imgs[j] = feasible_observation.t_start
                    th_imgs[j] = feasible_observation.look_angle
                    rewards[j] = reward

                    cumulative_rewards[j] += cumulative_rewards[i] + rewards[j]
                    preceeding_observations[j] = i

    @runtime_tracker
    def populate_adjacency_matrix(self, 
                                  state : SimulationAgentState, 
                                  specs : object,
                                  access_opportunities : list, 
                                  ground_points : dict,
                                  adjacency : list,
                                  j : int
                                  ):
               
        # get current observation
        curr_opportunity : tuple = access_opportunities[j]
        lat_curr,lon_curr = ground_points[curr_opportunity[0]][curr_opportunity[1]]
        curr_target = [lat_curr,lon_curr,0.0]

        # get any possibly prior observation
        prev_opportunities : list[tuple] = [prev_opportunity 
                                            for prev_opportunity in access_opportunities[:j]
                                            if prev_opportunity[3].start <= curr_opportunity[3].end
                                            and prev_opportunity != curr_opportunity
                                            ]
        
        prev_opportunities_opt : list[tuple] = [prev_opportunity 
                                            for prev_opportunity in access_opportunities[:j]
                                            ]
        
        assert len(prev_opportunities_opt) == len(prev_opportunities) and all([opp in prev_opportunities for opp in prev_opportunities_opt])
 

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
            
            # check if observation can be reached from THE previous observation
            adjacent = self.is_observation_path_valid(state, specs, 
                                                          [earliest_prev_observation,
                                                          ObservationAction(curr_opportunity[2], 
                                                                            curr_target, 
                                                                            curr_opportunity[5][-1], 
                                                                            curr_opportunity[4][-1])])     

            # update adjacency matrix
            adjacency[access_opportunities.index(prev_opportunity)][j] = adjacent

    @runtime_tracker
    def _schedule_broadcasts(self, state: SimulationAgentState, observations: list, orbitdata: OrbitData) -> list:
        # schedule measurement requests
        broadcasts : list = super()._schedule_broadcasts(state, observations, orbitdata)

        # # set earliest broadcast time to end of replanning period
        # t_broadcast = self.plan.t+self.period
        
        # # gather observation plan to be sent out
        # plan_out = [action.to_dict() for action in self.pending_observations_to_broadcast]

        # # check if observations exist in plan
        # if plan_out or self.sharing:
        #     # find best path for broadcasts
        #     path, t_start = self._create_broadcast_path(state, orbitdata, t_broadcast)

        #     # check feasibility of path found
        #     if t_start >= 0:
                
        #         # add performing action broadcast to plan
        #         for action_dict in tqdm(plan_out, 
        #                                 desc=f'{state.agent_name}-PLANNER: Pre-Scheduling Broadcasts', 
        #                                 leave=False):
        #             # update action dict to indicate completion
        #             action_dict['status'] = AgentAction.COMPLETED
                    
        #             # create message
        #             msg = ObservationPerformedMessage(state.agent_name, state.agent_name, action_dict, path=path)

        #             # add message broadcast to plan
        #             broadcasts.append(BroadcastMessageAction(msg.to_dict(),t_start))
            
        return broadcasts