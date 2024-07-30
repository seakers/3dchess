import os
import unittest
import numpy as np
import pandas as pd

from orbitpy.util import Spacecraft

from chess3d.agents.actions import ObservationAction
from chess3d.agents.planning.planners.rewards import RewardGrid, GridPoint
from chess3d.agents.science.requests import MeasurementRequest

class RewardGridTester(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.grid_data : list[pd.DataFrame]

    def setUp(self) -> None:
        # load grid data
        grid_path = os.path.join('./tests', 'rewards', 'resources')
        self.grid_data = [pd.read_csv(os.path.join(grid_path, file_name))
                        for file_name in os.listdir(grid_path)
                        if 'points' in file_name]
        
        # add grid index and gp index
        i_grid = int(0)
        for grid_datum in self.grid_data:
            nrows, _ = grid_datum.shape
            grid_datum['grid index'] = [i_grid] * nrows
            grid_datum['GP index'] = [i for i in range(nrows)]
            i_grid += int(1)
        
        # create reward grid
        agent_dict = {
                    "@id": "thermal_sat_0_0",
                    "name": "thermal_0",
                    "spacecraftBus": {
                        "name": "BlueCanyon",
                        "mass": 20,
                        "volume": 0.5,
                        "orientation": {
                            "referenceFrame": "NADIR_POINTING",
                            "convention": "REF_FRAME_ALIGNED"
                        },
                        "components": {
                            "adcs" : {
                                "maxTorque" : 1000,
                                "maxRate" : 1
                            }
                        }
                    },
                    "instrument": {
                        "name": "thermal",
                        "mass": 10,
                        "volume": 12.45,
                        "dataRate": 40,
                        "bitsPerPixel": 8,
                        "power": 12,
                        "snr": 33,
                        "spatial_res": 50,
                        "spectral_res": 7e-09,
                        "orientation": {
                            "referenceFrame": "NADIR_POINTING",
                            "convention": "REF_FRAME_ALIGNED"
                        },
                        "fieldOfViewGeometry": {
                            "shape": "RECTANGULAR",
                            "angleHeight": 5,
                            "angleWidth": 10
                        },
                        "maneuver" : {
                            "maneuverType":"SINGLE_ROLL_ONLY",
                            "A_rollMin": -50,
                            "A_rollMax": 50
                        },
                        "@id": "therm1",
                        "@type": "Basic Sensor"
                    },
                    "orbitState": {
                        "date": {
                            "@type": "GREGORIAN_UT1",
                            "year": 2020,
                            "month": 1,
                            "day": 1,
                            "hour": 0,
                            "minute": 0,
                            "second": 0
                        },
                        "state": {
                            "@type": "KEPLERIAN_EARTH_CENTERED_INERTIAL",
                            "sma": 7078,
                            "ecc": 0.01,
                            "inc": 67,
                            "raan": 0.0,
                            "aop": 0.0,
                            "ta": 0.0
                        }
                    }
                }
        self.agent_specs : Spacecraft = Spacecraft.from_dict(agent_dict)

        self.grid_params = {
            'initial_reward' : 1.0,
            'min_reward' : 1.0,
            'unobserved_reward_rate' : 3600.0, # pts/hrs
            'max_unobserved_reward' : 10.0,
            'event_reward' : 10.0,
            'reobsevation_strategy' : lambda a : 1.0
        }        

    def test_grid_data(self) -> None:
        """ check if grid data was loaded correctly """
        ref_columns = ['lat [deg]', 'lon [deg]', 'grid index', 'GP index']
        
        for grid_datum in self.grid_data:
            self.assertEqual(len(ref_columns), len(grid_datum.columns))
            for column in grid_datum.columns:
                self.assertTrue(column in ref_columns)
                
    def test_reward_grid_init(self) -> None:
        """ checks if the reward grid was properly initialized """
        # load reward grid 
        reward_grid = RewardGrid(reward_func, self.agent_specs, self.grid_data, **self.grid_params)
        
        # check type
        self.assertIsInstance(reward_grid, RewardGrid)

        # check initial values
        for grid_datum in self.grid_data:
            for lat,lon,grid_index,gp_index in grid_datum.values:
                grid_index = int(grid_index)
                gp_index = int(gp_index)

                for _,reward_point in reward_grid.rewards[grid_index][gp_index].items():
                    reward_point : GridPoint
                    gp_lat,gp_lon,_ = reward_point.target

                    # check gp position
                    self.assertAlmostEqual(gp_lat, lat)
                    self.assertAlmostEqual(gp_lon, lon)
                    self.assertEqual(reward_point.gp_index, gp_index)
                    self.assertEqual(reward_point.grid_index, grid_index)

                    # check initial reward
                    self.assertAlmostEqual(reward_point.reward, self.grid_params['initial_reward'])

                    # check initial time
                    self.assertTrue(np.isnan(reward_point.t_update))

    def test_grid_propagation(self) -> None:
        """ checks that reward grid values are being propagated correctly """
        # set params
        n_steps = 11
        time_step = 1.0
        min_val = 1.0
        max_val = 10.0
        unobserved_reward_rate = 3600
        
        # create reference values
        ref_values = [min(i*time_step*unobserved_reward_rate/3600.0 + min_val, max_val)  for i in range(n_steps)]

        # load reward grid 
        reward_grid = RewardGrid(reward_func, self.agent_specs, self.grid_data, **self.grid_params)
        
        for i in range(n_steps):
            # calculate time
            t = i * time_step

            # update grid rewards
            reward_grid.update(t)

            # check values
            for grid_datum in self.grid_data:
                for *_,grid_index,gp_index in grid_datum.values:
                    grid_index,gp_index = int(grid_index), int(gp_index)

                    for _,reward_point in reward_grid.rewards[grid_index][gp_index].items():
                        reward_point : GridPoint

                        # check correct updated time
                        self.assertAlmostEqual(reward_point.t_update, t)

                        # check correct updated reward
                        self.assertAlmostEqual(reward_point.reward, ref_values[i])

    def test_reward_grid_reset(self) -> None:
        """ checks if all values are set to the default values when calling `reset()` """
        # set params
        n_steps = 11
        time_step = 1.0
        
        # load reward grid 
        reward_grid = RewardGrid(reward_func, self.agent_specs, self.grid_data, **self.grid_params)
        
        # propagate grid
        for i in range(n_steps):
            # calculate time
            t = i * time_step

            # update grid rewards
            reward_grid.update(t)

        # reset grid
        reward_grid.reset()

        # check values
        for grid_datum in self.grid_data:
            for lat,lon,grid_index,gp_index in grid_datum.values:
                grid_index = int(grid_index)
                gp_index = int(gp_index)

                for _,reward_point in reward_grid.rewards[grid_index][gp_index].items():
                    reward_point : GridPoint
                    gp_lat,gp_lon,_ = reward_point.target

                    # check gp position
                    self.assertAlmostEqual(gp_lat, lat)
                    self.assertAlmostEqual(gp_lon, lon)
                    self.assertEqual(reward_point.gp_index, gp_index)
                    self.assertEqual(reward_point.grid_index, grid_index)

                    # check initial reward
                    self.assertAlmostEqual(reward_point.reward, self.grid_params['initial_reward'])

                    # check initial time
                    self.assertTrue(np.isnan(reward_point.t_update))

    def test_observation_propagation(self) -> None:
        """ checks if the utility function properly """
        # create reference values
        ref_values = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0, 10.0]

        # set parameters
        n_steps = len(ref_values)
        time_step = 1.0

        # create observations
        observations = [ObservationAction('thermal', [0.0, 0.0, 0.0], 0.0, 4.0)]

        # load reward grid
        reward_grid = RewardGrid(reward_func, self.agent_specs, self.grid_data, **self.grid_params)

        for i in range(n_steps):
            # calculate time
            t = i * time_step

            # get relevant observations
            relevant_observations = [observation for observation in observations
                                     if observation.t_end <= t]

            # update grid rewards
            reward_grid.update(t, relevant_observations)

            for observation in relevant_observations: observations.remove(observation)

            # check values
            for grid_datum in self.grid_data:
                for *_,grid_index,gp_index in grid_datum.values:
                    grid_index,gp_index = int(grid_index), int(gp_index)

                    for _,reward_point in reward_grid.rewards[grid_index][gp_index].items():
                        reward_point : GridPoint

                        # check correct updated time
                        self.assertAlmostEqual(reward_point.t_update, t)

                        # check correct updated reward
                        self.assertAlmostEqual(reward_point.reward, ref_values[i])

        # reset grid
        reward_grid.reset()

    def test_event_propagation(self) -> None:
        """ checks if the utility function properly """
        # create reference values
        ref_values = [1.0, 2.0, 3.0, 4.0, 
                      10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 
                      2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0, 10.0]
        
        # set parameters
        n_steps = len(ref_values)
        time_step = 1.0

        # create events
        events = [MeasurementRequest('ADMIN', [0.0,0.0,0.0], 1.0, ['thermal'], 4.0, 13.0)]

        # load reward grid
        reward_grid = RewardGrid(reward_func, self.agent_specs, self.grid_data, **self.grid_params)

        for i in range(n_steps):
            # calculate time
            t = i * time_step

            # get relevant observations
            relevant_events = [event for event in events
                               if event.t_start <= t]

            # update grid rewards
            reward_grid.update(t, events=relevant_events)

            for event in relevant_events: events.remove(event)

            # check values
            for grid_datum in self.grid_data:
                for *_,grid_index,gp_index in grid_datum.values:
                    grid_index,gp_index = int(grid_index), int(gp_index)

                    for _,reward_point in reward_grid.rewards[grid_index][gp_index].items():
                        reward_point : GridPoint

                        # check correct updated time
                        self.assertAlmostEqual(reward_point.t_update, t)

                        # check correct updated reward
                        self.assertAlmostEqual(reward_point.reward, ref_values[i])

        # reset grid
        reward_grid.reset()

def reward_func(
                observations,
                events,
                reward, 
                min_reward, 
                unobserved_reward_rate, 
                max_unobserved_reward, 
                event_reward,
                t, 
                t_update, 
                reobsevation_strategy,
                **_):
    if np.isnan(t_update) and np.isnan(t):
        return reward

    # find latest event
    events : list[MeasurementRequest] = list(events)
    events.sort(key=lambda a : a.t_end)
    latest_event = events.pop() if events else None

    # find latest observations
    observations : list[ObservationAction] = list(observations)
    observations.sort(key= lambda a : a.t_start)
    
    # calculate utility
    if latest_event:
        # an event exists for this ground point
        assert latest_event.t_start <= t

        if latest_event.t_start <= t <= latest_event.t_end: # event is current
            # count previous observations
            latest_observations = [observation for observation in observations
                                   if latest_event.t_start <= observation.t_start <= latest_event.t_end]
            
            # calculate reward
            tp=[latest_event.t_start, latest_event.t_end]
            rp=[event_reward,min_reward]
            reward = np.interp(t,tp,rp) * reobsevation_strategy(len(latest_observations))
            
        else: # event has already passed
            # check if an observation has occurred since then
            latest_observations = [observation for observation in observations
                                   if latest_event.t_end < observation.t_start]
            latest_observation = latest_observations.pop() if latest_observations else None

            # calculate reward
            t_init = max(latest_event.t_end, latest_observation.t_end) if latest_observation else latest_event.t_end
            reward = (t - t_init) * unobserved_reward_rate / 3600  + min_reward
            reward = min(reward, max_unobserved_reward)

    else: # no events have been detected for this ground point
        # get latest observation if it exists
        latest_observation = observations.pop() if observations else None
        t_init = latest_observation.t_end if latest_observation else 0.0
        
        # calculate reward
        reward = (t - t_init) * unobserved_reward_rate / 3600  + min_reward
        reward = min(reward, max_unobserved_reward)

    return reward

if __name__ == '__main__':
    unittest.main()