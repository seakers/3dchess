import os
import unittest
import numpy as np
import pandas as pd

from orbitpy.util import Spacecraft

from chess3d.agents.planning.planners.rewards import RewardGrid, GridPoint

def reward_func(**_):
    return np.NINF

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
        i_grid = 0
        for grid_datum in self.grid_data:
            nrows, _ = grid_datum.shape
            grid_datum['grid index'] = [i_grid] * nrows
            grid_datum['GP index'] = [i for i in range(nrows)]
            i_grid += 1
        
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
        agent_specs : Spacecraft = Spacecraft.from_dict(agent_dict)

        self.reward_grid = RewardGrid(reward_func, agent_specs, self.grid_data)

    def test_grid_data(self) -> None:
        """ check if grid data was loaded correctly """
        ref_columns = ['lat [deg]', 'lon [deg]', 'grid index', 'GP index']
        
        for grid_datum in self.grid_data:
            self.assertEqual(len(ref_columns), len(grid_datum.columns))
            for column in grid_datum.columns:
                self.assertTrue(column in ref_columns)
                
    def test_reward_grid_init(self) -> None:
        """ checks if the reward grid was properly initialized """
        # check type
        self.assertIsInstance(self.reward_grid, RewardGrid)

        for grid_datum in self.grid_data:
            for lat,lon,grid_index,gp_index in grid_datum.values:
                grid_index = int(grid_index)
                gp_index = int(gp_index)

                for _,reward_point in self.reward_grid.rewards[grid_index][gp_index].items():
                    reward_point : GridPoint
                    gp_lat,gp_lon,_ = reward_point.target

                    self.assertAlmostEqual(gp_lat, lat)
                    self.assertAlmostEqual(gp_lon, lon)
                    self.assertEqual(reward_point.grid_index, grid_index)
                    self.assertEqual(reward_point.gp_index, gp_index)

if __name__ == '__main__':
    unittest.main()