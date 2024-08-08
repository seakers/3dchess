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

        # compile access times for this planning horizon
        access_opportunities, ground_points = self.calculate_access_opportunities(state, specs, orbitdata)
        access_opportunities : list; ground_points : dict

        # sort by observation time
        access_opportunities.sort(key=lambda a: a[3])

        # TEMP: 
        # access_opportunities = access_opportunities[:15]

        # compile list of instruments available in payload
        payload : dict = {instrument.name: instrument for instrument in specs.instrument}
        
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self.collect_fov_specs(specs)
        
        # get pointing agility specifications
        max_slew_rate, max_torque = self.collect_agility_specs(specs)

        # create results arrays
        t_imgs = [np.NAN for _ in access_opportunities]
        th_imgs = [np.NAN for _ in access_opportunities]
        rewards = [0.0 for _ in access_opportunities]
        commulative_rewards = [0.0 for _ in access_opportunities]
        preceeding_observations = [np.NAN for _ in access_opportunities]
        adjancency = [[False for _ in access_opportunities] for _ in access_opportunities]

        # create adjancency matrix
        for j in range(len(access_opportunities)):
            # get current observation
            curr_opportunity : tuple = access_opportunities[j]

            # get any possibly prior observation
            prev_opportunities : list[tuple] = [prev_opportunity for prev_opportunity in access_opportunities
                                                if prev_opportunity[3].end <= curr_opportunity[3].end
                                                and prev_opportunity != curr_opportunity]

            # construct adjacency matrix
            for prev_opportunity in prev_opportunities:
                prev_opportunity : tuple
                i = access_opportunities.index(prev_opportunity)

                # Step 1: check if manevuer can be performed between observation opportunities
                valid_indeces = [k for k in range(len(curr_opportunity[4]))
                                if (      curr_opportunity[4][k] - prev_opportunity[4][0]
                                    >= abs(curr_opportunity[5][k] - prev_opportunity[5][0])*max_slew_rate - 1e-6)
                                or abs( (curr_opportunity[4][k] - prev_opportunity[4][0]) 
                                      - abs(curr_opportunity[5][k] - prev_opportunity[5][0])*max_slew_rate) 
                                    < 1e-6]

                adjancency[i][j] = len(valid_indeces) > 0

        # update results
        for j in range(len(access_opportunities)):
            # get any possibly prior observation
            prev_opportunities : list[tuple] = [prev_observation for prev_observation in access_opportunities
                                                if adjancency[access_opportunities.index(prev_observation)][j]]

            if not prev_opportunities:
                t_imgs[j] = curr_opportunity[4][0]
                th_imgs[j] = curr_opportunity[5][0]

            for prev_opportunity in prev_opportunities:
                prev_opportunity : tuple
                i = access_opportunities.index(prev_opportunity)
                x  = 1

                # rewards[j] = reward_grid.estimate_reward(observation)
                
                # if commulative_rewards[i] + rewards[j] > commulative_rewards[j]:
                #     commulative_rewards[j] += commulative_rewards[i]
                #     preceeding_observations[j] = i

            # # find time of observation and adjancency to previous observation opportunities            
            # for prev_opportunity in prev_opportunities:
            #     prev_opportunity : tuple
            #     i = access_opportunities.index(prev_opportunity)

            #     # check for previous arrival time
            #     if np.isnan(t_imgs[i]): # no time has been set for the previous observation opportunities

            #         # ignore previous observation and set earliest time and look angle
                    # t_imgs[j] = curr_opportunity[4][0]
                    # th_imgs[j] = curr_opportunity[5][0]
                    
            #     else: # an observation time has been already set for the previous observation
                    
                    # # Step 1: check if manevuer can be performed between observation opportunities
                    
                    # # calculate maneuver time
                    # dt_maneuver = abs(curr_opportunity[5][0] - prev_opportunity[5][-1]) / max_slew_rate
                    # dt_intervals = curr_opportunity[4][0] - prev_opportunity[4][-1]

                    # if dt_maneuver < dt_intervals: # interval can be performed between observation opportunities
            #             # set earliest observation time and look angle 
            #             t_imgs[j] = curr_opportunity[4][0]
            #             th_imgs[j] = curr_opportunity[5][0]

            #         else:
            #             pass
                    
            #         x = 1

            # update results
            # for prev_opportunity in prev_opportunities:
            #     rewards[j] = reward_grid.estimate_reward(observation)
                
            #     if commulative_rewards[i] + rewards[j] > commulative_rewards[j]:
            #         commulative_rewards[j] += commulative_rewards[i]
            #         preceeding_observations[j] = i

        x = 1

        # observations = [ObservationOpportunity(grid_index, gp_index, instrument, t_interval, th_interval)
        #                 for grid_index, gp_index, instrument, _, t_interval, th_interval in access_opportunities]
        # graph = ObservationOpportunity(t_img=state.t,th_img=state.attitude[0])

        # assert all([len(obs.children) == 0 for obs in observations])

        # queue = []
        # queue.append(graph)

        # while queue:
        #     obs_i : ObservationOpportunity = queue.pop(0)
            
        #     # check if adjacent to nodes in the graph
        #     adjacent_obs = [obs_j for obs_j in observations
        #                     if obs_i.is_adjacent(obs_j, max_slew_rate, max_torque)]
        #     adjacent_obs.sort(key=lambda a: a.ts[0])
                        
        #     # update adjacent observations
        #     for obs_j in adjacent_obs:
        #         obs_j : ObservationOpportunity

        #         #calculate imaging times for adjacent observations
        #         t_img_j = obs_i.t_img + obs_i.calc_manuever_time(obs_j, max_slew_rate)
                
        #         # update observation imaging time
        #         t_img_j_prev = obs_j.t_img
        #         obs_j.update_t(t_img_j)

        #         # add observation to list of children in graph
        #         obs_i.add_child(obs_j)

        #         # add child to queue if time was updated and is not already in the queue
        #         if (obs_j not in queue 
        #             and (abs(t_img_j_prev - obs_j.t_img) > 1e-6
        #                  or obs_i.is_root())
        #             ): 
        #             queue.append(obs_j)
            
        #     x = 1       

        # queue = []
        # queue.append(graph)

        # while queue:      
        #     obs_i : ObservationOpportunity = queue.pop(0)   

        #     for obs_j in obs_i.children:
        #         if obs_i.t_img > obs_j.t_img: 
        #             x = 1


        #     assert all([obs_i.t_img <= obs_j.t_img for obs_j in obs_i.children])

        #     for obs_j in obs_i.children: queue.append(obs_j)

        # x = 1

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