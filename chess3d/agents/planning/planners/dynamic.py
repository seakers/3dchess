from logging import Logger
from queue import Queue
from numpy import Inf
from orbitpy.util import Spacecraft

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

class ObservationOpportunity:
    def __init__(self, 
                 grid_index : int, 
                 gp_index : int, 
                 instrument : str, 
                 interval : TimeInterval, 
                 ts : list, 
                 ths : list,
                 prev_observations : list,
                #  grid_index : int = -1,
                #  gp_index : int = -1,
                #  instrument : str = None,
                 t_img : float = None,
                 th_img : float = None,
                 reward : float = None,
                 commulative_reward : float = None
                #  children : set = None,
                #  parent : object = None, 
                 ) -> None:
        
        self.grid_index = grid_index
        self.gp_index = gp_index
        self.instrument = instrument

        self.ts = ts
        self.ths = ths

        if t_img is not None:
            self.t_img = t_img
        elif ts is not None:
            self.t_img = ts[0]
        else:
            self.t_img = None
        assert (self.ts is None
                or self.t_img is None 
                or self.ts[0] <= self.t_img <= self.ts[-1])
        
        if th_img is not None:
            self.th_img = th_img
        elif ths is not None:
            self.th_img = ths[0]
        else:
            self.th_img = None
        assert (self.ths is None 
                or self.th_img is None 
                or min(self.ths) <= self.th_img <= max(self.ths))
        
        # self.parent = parent
        # self.children = children if children is not None else set()
    
    def is_root(self) -> bool:
        return (self.grid_index == -1 
                and self.gp_index == -1 
                and self.instrument is None
                and self.ts is None
                and self.ths is None)

    def add_child(self, __other : object) -> None:
        self.children.add(__other)

    def calc_manuever_time(self, __other : object, max_slew_rate : float) -> float:
        assert isinstance(__other, ObservationOpportunity)
        return abs(__other.th_img - self.th_img) / max_slew_rate
    
    def is_adjacent(self, __other : object, max_slew_rate : float, max_torque : float) -> bool:
        assert isinstance(__other, ObservationOpportunity)
                
        if __other.ts[-1] < self.t_img:
            return False
        if __other is self:
            return False
        
        dt_maneuver = self.calc_manuever_time(__other, max_slew_rate)

        slew_constraint = (self.t_img + dt_maneuver) <= __other.ts[-1]
        torque_constraint = True

        return  slew_constraint and torque_constraint
    
    def update_t(self, t_img : float) -> None:
        assert not self.is_root()

        # only update if its the latest anyone can reach me        
        if (self.ts[0] <= t_img <= self.ts[-1]
            and self.t_img < t_img):

            self.t_img = t_img
            self.th_img = np.interp(t_img, self.ts, self.ths)
                        
            # # update subsequent observations
            # children_to_remove = set()
            # for obs_j in self.children:
            #     obs_j : ObservationOpportunity
            #     t_img_j = self.t_img+self.calc_manuever_time(obs_j, max_slew_rate) 

            #     # update observation imaging time
            #     if t_img_j <= obs_j.t_interval[-1]:
            #         obs_j.update_t(t_img_j)
            #     else:
            #         children_to_remove.add(obs_j)
            
            # for obs_j in children_to_remove: self.children.remove(obs_j)

        assert t_img <= self.t_img

        assert (self.ts is None
                or self.t_img is None 
                or self.ts[0] <= self.t_img <= self.ts[-1])
        
        assert (self.ths is None 
                or self.th_img is None 
                or min(self.ths) <= self.th_img <= max(self.ths))
        

    def __repr__(self) -> str:
        return f'ObservationOpportunity_{self.grid_index}_{self.gp_index}_{self.ts[0]}-{self.ts[-1]}'

class DynamicProgrammingPlanner(AbstractPreplanner):
    def __init__(self, 
                 sharing : bool = False,
                 horizon: float = np.Inf, 
                 period: float = np.Inf, 
                 logger: Logger = None
                 ) -> None:
        super().__init__(horizon, period, logger)

        # toggle for sharing plans
        self.sharing = sharing 

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
        for j in range(len(access_opportunities)):
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
                adjancency[access_opportunities.index(prev_opportunity)][j] = adjacent

        t_2 = time.perf_counter() - t_prev
        t_prev = time.perf_counter()

        # update results
        for j in range(len(access_opportunities)): 
            # get current observation opportunity
            curr_opportunity : tuple = access_opportunities[j]
            lat,lon = ground_points[curr_opportunity[0]][curr_opportunity[1]]
            curr_target = [lat,lon,0.0]

            # get any possibly prior observation
            prev_opportunities : list[tuple] = [prev_observation for prev_observation in access_opportunities
                                                if adjancency[access_opportunities.index(prev_observation)][j]]

            # calculate all possible observation actionss for this observation opportunity
            possible_observations = [ObservationAction(curr_opportunity[2], 
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

                # check if previous opportunity has been checked yet
                if np.isnan(th_imgs[i]):

                    # get possible observation actions from the current observation opportuinty
                    possible_observations = [possible_observation for possible_observation in possible_observations
                                         if self.is_observation_path_valid(state, specs, [possible_observation])]

                else:
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

                if possible_observations: # an observation is possible
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
        while np.isnan(preceeding_observations[-1]):
            preceeding_observations.pop()
        observation_sequence = [len(preceeding_observations)-1]
        
        while not np.isnan(preceeding_observations[observation_sequence[-1]]):
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
            x = 1

        t_5 = time.perf_counter() - t_prev
        t_f = time.perf_counter() - t_0
        return observations

    def is_adjacent(self,
                    prev_observation : tuple,
                    curr_observation : tuple) -> bool:
        pass

    # def can_travel(self, 
    #                observation_i : tuple, 
    #                observation_j : tuple,
    #                state : SimulationAgentState,
    #                specs : object,
    #                ground_points : list
    #                ) -> bool:
    #     grid_i,gp_i,instrument_i,interval_i,th_i,t_i = observation_i
    #     grid_j,gp_j,instrument_j,interval_j,th_j,t_j = observation_j

    #     # if interval_i == interval_j:
    #     #     # avoid doing the same observation after 
    #     #     return False
        
    #     if t_i > t_j:
    #         # cannot travel to the past
    #         return False
        
    #     lat_i,lon_i = ground_points[grid_i][gp_i]
    #     lat_j,lon_j = ground_points[grid_j][gp_j]

    #     observations = [
    #                     ObservationAction(instrument_i, [lat_i,lon_i,0], th_i, t_i),
    #                     ObservationAction(instrument_j, [lat_j,lon_j,0], th_j, t_j)
    #                     ]
    #     return self.is_observation_path_valid(state, specs, observations)

    def _schedule_broadcasts(self, state: SimulationAgentState, observations: list, orbitdata: OrbitData) -> list:
        broadcasts =  super()._schedule_broadcasts(state, observations, orbitdata)

        if self.sharing:
            # TODO schedule reward grid broadcasts
            pass
            
        return broadcasts