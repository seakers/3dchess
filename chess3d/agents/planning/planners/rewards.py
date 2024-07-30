from typing import  Callable

import numpy as np
import pandas as pd

from orbitpy.util import Spacecraft

from chess3d.agents.actions import ObservationAction
from chess3d.agents.science.requests import MeasurementRequest

class GridPoint(object):
    """ Describes the reward of performing an observation of a given ground point """
    def __init__(self, 
                 instrument : str,
                 lat : float, 
                 lon : float, 
                 grid_index : int,
                 gp_index : int,
                 initial_reward : float,                  
                 alt : float = 0.0, 
                 observations : list = [],
                 events : list = [],
                 t_update : float = np.NAN
                 ) -> None:
        # set fixed parameters
        self.instrument : str = instrument
        self.target : list[float] = [lat, lon, alt]
        self.grid_index : int = grid_index
        self.gp_index : int = gp_index
        self.initial_reward : float = initial_reward

        # set variable parameters
        self.observations : set[ObservationAction] = {observation for observation in observations}
        self.events : set[MeasurementRequest] = {event for event in events}
        self.reward : float = initial_reward
        self.t_update : float = t_update

    def update_observations( self, observation : ObservationAction, t_update : float) -> None:
        assert self.is_update_time_valid(t_update)
        self.observations.add(observation)
        self.t_update = t_update

    def update_events(self, event : MeasurementRequest, t_update : float) -> None:
        assert self.is_update_time_valid(t_update)
        self.events.add(event)
        self.t_update = t_update

    def update_reward(self, reward : float, t_update : float) -> None:
        assert self.is_update_time_valid(t_update)
        self.reward = reward
        self.t_update = t_update

    def is_update_time_valid(self, t_update : float) -> bool:
        return t_update >= self.t_update or np.isnan(self.t_update)
    
    def reset(self) -> None:
        self.observations = set()
        self.events = set()
        self.reward = self.initial_reward
        self.t_update = np.NAN

    def to_dict(self) -> dict:
        return self.__dict__
    
    def __str__(self) -> str:
        return f'{self.t_update},{self.grid_index},{self.gp_index},{self.instrument},{self.reward},{len(self.observations)}'
    
    def __repr__(self) -> str:
        return f'GridPoint_{self.grid_index}_{self.gp_index}_{self.instrument}'

class RewardGrid(object):
    def __init__(self, 
                 reward_func : Callable,
                 specs : object, 
                 grid_data : list, 
                 initial_reward : float,
                 **grid_params : dict,              
                 ) -> None:       
        
        # save reward function
        self.reward_func = reward_func

        # load grid data
        self.grid_data : list[pd.DataFrame] = grid_data

        # ensure no repeats on grid indeces
        assert all([len([int(gp_index) for _,_,_,gp_index in grid_datum.values]) == len({int(gp_index) for _,_,_,gp_index in grid_datum.values})
                   for grid_datum in self.grid_data])
        
        # save additional parameters
        self.specs = specs
        self.initial_reward = initial_reward
        self.grid_params : dict = grid_params

        # initiate reward grid vectors
        if isinstance(specs, Spacecraft):
            self.rewards = [ [{instrument.name : GridPoint(instrument.name, 
                                                           lat, 
                                                           lon, 
                                                           int(grid_index), 
                                                           int(gp_index),
                                                           initial_reward) 
                               for instrument in specs.instrument} 
                            for lat,lon,grid_index,gp_index in grid_datum.values] 
                            for grid_datum in self.grid_data]
        elif isinstance(specs, dict):
            self.rewards = [ [{instrument['name'] : GridPoint(instrument['name'], 
                                                              lat, 
                                                              lon, 
                                                              int(grid_index), 
                                                              int(gp_index),
                                                              initial_reward)  
                               for instrument in specs['payload']} 
                            for lat,lon,grid_index,gp_index in grid_datum.values] 
                            for grid_datum in self.grid_data]
        else:
            raise ValueError(f'`specs` of type {type(specs)} not supported.')

        # update previous observations and events (if they exist)
        prev_observations = grid_params.get('prev_observations', [])
        prev_events = grid_params.get('prev_events', [])
        self.update(np.NAN, prev_observations, prev_events)

    def __get_target_indeces(self, lat : float, lon : float) -> tuple:
        # find ground points with matching latitude and longitude
        matches = [(grid_index,gp_index)
                   for grid_datum in self.grid_data
                   for gp_lat,gp_lon,grid_index,gp_index in grid_datum.values
                   if abs(gp_lat - lat) <= 1e-6
                   and abs(gp_lon - lon) <= 1e-6
                   ]
        
        # return findings
        match = matches.pop() if matches else (None,None)
        grid_index, gp_index = match
        
        # convert to integers
        grid_index = int(grid_index) if grid_index is not None else None
        gp_index = int(gp_index) if gp_index is not None else None
        
        # return values
        return grid_index, gp_index 
        
    def reset(self) -> None:
        for grid_rewards in self.rewards:
            for gp_rewards in grid_rewards:
                for _, grid_point in gp_rewards.items():
                    grid_point.reset()

    def update(self, t : float, observations : list = [], events : list = []) -> None:
        # update observations
        for observation in observations: self.update_observation(observation, t)

        # update events
        for event in events: self.update_requests(event, t)

        # update values for the entire grid
        for grid_rewards in self.rewards:
            for gp_rewards in grid_rewards:
                for _, grid_point in gp_rewards.items():
                    # estimate reward
                    reward : float = self.propagate_reward(grid_point, t)

                    # update grid point reward
                    grid_point.update_reward(reward, t)

    def update_observation(self, observation : ObservationAction, t : float) -> None:
        # get appropriate grid point object
        lat,lon,_ = observation.target
        grid_index,gp_index = self.__get_target_indeces(lat,lon)
        grid_point : GridPoint = self.rewards[grid_index][gp_index][observation.instrument_name]
        
        # update grid point observation list
        grid_point.update_observations(observation, t)

    def update_requests(self, request : MeasurementRequest, t : float) -> None:
        # get appropriate grid point object
        lat,lon,_ = request.target
        grid_index,gp_index = self.__get_target_indeces(lat,lon)

        for instrument in request.observation_types:
            grid_point : GridPoint = self.rewards[grid_index][gp_index][instrument]
            
            # update grid point observation list
            grid_point.update_events(request, t)

            # estimate reward
            reward : float = self.propagate_reward(grid_point, t)

            # update grid point reward
            grid_point.update_reward(reward, t)
       
    def propagate_reward(self, grid_point : GridPoint, t : float) -> float:
        params = grid_point.to_dict()
        params.update(self.grid_params)
        params['t'] = t

        return self.reward_func(**params)
    
    def estimate_reward(self, lat : float, lon : float, observation : ObservationAction) -> float:
        grid_index,gp_index = self.__get_target_indeces(lat,lon)
        grid_point : GridPoint = self.grid_data[grid_index][gp_index][observation.instrument_name]

        return self.propagate_reward(grid_point, observation.t_start)