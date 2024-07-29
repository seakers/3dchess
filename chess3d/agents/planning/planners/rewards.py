from typing import  Callable

import numpy as np
import pandas as pd

from orbitpy.util import Spacecraft

from chess3d.agents.actions import ObservationAction

class GridPoint(object):
    """ Describes the reward of performing an observation of a given ground point """
    def __init__(self, 
                 instrument : str,
                 lat : float, 
                 lon : float, 
                 grid_index : int,
                 gp_index : int,
                 alt : float = 0.0, 
                 observations : list = [],
                 reward : float = 0.0, 
                 t_update : float = np.NAN
                 ) -> None:
        # set fixed parameters
        self.instrument : str = instrument
        self.target : list[float] = [lat, lon, alt]
        self.grid_index : int = grid_index
        self.gp_index : int = gp_index

        # set variable parameters
        self.observations : list[ObservationAction] = [observation for observation in observations]
        self.reward : float = reward
        self.t_update : float = t_update

    def to_dict(self) -> dict:
        return self.__dict__
    
    def __str__(self) -> str:
        return f'{self.t_update},{self.grid_index},{self.gp_index},{self.instrument},{self.reward},{len(self.observations)}'
    
    def __repr__(self) -> str:
        return f'GridPoint_{self.grid_index}_{self.gp_index}_{self.instrument}'

    def update( self, 
                observation : ObservationAction, 
                reward : float, 
                t_update : float
                ) -> None:
        
        assert t_update >= self.t_update

        self.observations.append(observation)
        self.reward = reward
        self.t_update = t_update

class RewardGrid(object):
    def __init__(self, 
                 reward_func : Callable,
                 specs : dict, 
                 grid_data : list, 
                 prev_observations : list = []) -> None:       
        # save reward function
        self.reward_func = reward_func

        # load grid data
        self.grid_data : list[pd.DataFrame] = grid_data

        # ensure no repeats on grid indeces
        assert all([len([int(gp_index) for _,_,_,gp_index in grid_datum.values]) == len({int(gp_index) for _,_,_,gp_index in grid_datum.values})
                   for grid_datum in self.grid_data])

        # initiate reward grid vectors
        if isinstance(specs, Spacecraft):
            self.rewards = [ [{instrument.name : GridPoint(instrument.name, lat, lon, int(grid_index), int(gp_index)) 
                               for instrument in specs.instrument} 
                            for lat,lon,grid_index,gp_index in grid_datum.values] 
                            for grid_datum in self.grid_data]
        elif isinstance(specs, dict):
            self.rewards = [ [{instrument['name'] : GridPoint(instrument.name, lat, lon, int(grid_index), int(gp_index))  
                               for instrument in specs['payload']} 
                            for lat,lon,grid_index,gp_index in grid_datum.values] 
                            for grid_datum in self.grid_data]
        else:
            raise ValueError(f'`specs` of type {type(specs)} not supported.')

        # update previous observations
        self.update_all(prev_observations)

    def __get_target_indeces(self, lat : float, lon : float) -> tuple:
        # find ground points with matching latitude and longitude
        matches = [(grid_index,gp_index)
                   for grid_datum in self.grid_data
                   for gp_lat,gp_lon,grid_index,gp_index in grid_datum.values
                   if abs(gp_lat - lat) <= 1e-6
                   and abs(gp_lon - lon) <= 1e-6
                   ]
        
        # return findings
        return matches.pop() if matches else None,None
        
    def estimate_reward(self, grid_point : GridPoint, observation : ObservationAction) -> float:
        params = grid_point.to_dict()
        params['observation'] = observation
        return self.reward_func(**params)

    def update_all(self, observations : list, t : float = np.NAN) -> None:
        for observation in observations:
            observation : ObservationAction
            self.update(observation, t)

    def update(self, observation : ObservationAction, t : float) -> None:
        
            # get appropriate grid point object
        lat,lon,_ = observation.target
        grid_index,gp_index = self.__get_target_indeces(lat,lon)
        grid_point : GridPoint = self.grid_data[grid_index][gp_index][observation.instrument_name]
        
        # estimate reward
        reward : float = self.estimate_reward(grid_point, observation)

        # update grid point
        grid_point.update(observation, reward, t)