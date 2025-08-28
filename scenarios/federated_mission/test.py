import json
import os
import unittest

from chess3d.mission.mission import Mission
from chess3d.simulation import Simulation
from chess3d.utils import print_welcome


class TestMissionLoading(unittest.TestCase):
    def test_mission_loading(self) -> None:
        mission_dict_path = './scenarios/federated_mission/resources/missions/missions.json'

        with open(mission_dict_path) as f:
            self.missions_dict = json.load(f)

        missions = [
            Mission.from_dict(mission_data) 
            for mission_data in self.missions_dict.get("missions", [])
        ]
        assert all(isinstance(mission, Mission) for mission in missions)

class TestFederatedMission(unittest.TestCase):
    def setUp(self) -> None:        
        # load scenario json file
        self.spacecraft_template = {
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
                        "name": "VNIR hyper",
                        "@id" : "vnir_hyp_imager",
                        "@type" : "VNIR",
                        "detectorWidth": 6.6e-6,
                        "focalLength": 3.6,  
                        "orientation": {
                            "referenceFrame": "NADIR_POINTING",
                            "convention": "REF_FRAME_ALIGNED"
                        },
                        "fieldOfViewGeometry": { 
                            "shape": "RECTANGULAR", 
                            "angleHeight": 2.5, 
                            "angleWidth": 45.0
                        },
                        "maneuver" : {
                            "maneuverType":"SINGLE_ROLL_ONLY",
                            "A_rollMin": -50,
                            "A_rollMax": 50
                        },
                        "spectral_resolution" : "Multispectral"
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
                            "inc": 60.0,
                            "raan": 0.0,
                            "aop": 0.0,
                            "ta": 95.0
                        }
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "earliest",
                            "period": 500,
                            # "horizon": 500,
                        },
                        # "replanner" : {
                        #     "@type" : "broadcaster",
                        #     "period" : 400
                        # },
                    },
                    # "science" : {
                    #     "@type": "lookup", 
                    #     "eventsPath" : "./scenarios/federated_mission/resources/events/toy_events.csv"
                    # },
                    "mission" : "Algal blooms monitoring"
            }
        
        # set outdir
        orbitdata_dir = os.path.join('./scenarios/federated_mission', 'orbit_data')
        if not os.path.isdir(orbitdata_dir): os.mkdir(orbitdata_dir)

    def setup_scenario_specs(self, 
                             duration : float, 
                             grid_name : str, 
                             scenario_name : str, 
                             connectivity : str, 
                             event_name : str, 
                             mission_name : str,
                             spacecraft : list = []
                             ) -> dict:
        scenario_specs : dict = {
            "epoch": {
                "@type": "GREGORIAN_UT1",
                "year": 2020,
                "month": 1,
                "day": 1,
                "hour": 0,
                "minute": 0,
                "second": 0
            },
            "duration": duration,
            "propagator": {
                "@type": "J2 ANALYTICAL PROPAGATOR",
            }
        }
        scenario_specs['grid'] = self.setup_grid(grid_name)
        scenario_specs['scenario'] = self.setup_scenario(scenario_name, connectivity, event_name, mission_name)
        scenario_specs['settings'] = self.setup_scenario_settings(scenario_name)
        scenario_specs['spacecraft'] = spacecraft

        return scenario_specs

    def setup_grid(self, grid_name : str) -> dict:
        """Setup the grid for the scenario. """

        assert isinstance(grid_name, str), "grid_name must be a string"

        assert os.path.isfile(f"./scenarios/federated_mission/resources/grids/{grid_name}.csv"), \
            f"Grid file not found: {grid_name}.csv"

        grid = {
            "@type": "customGrid",
            "covGridFilePath": f"./scenarios/federated_mission/resources/grids/{grid_name}.csv"
        }
        return [grid]

    def setup_scenario(self, scenario_name : str, connectivity : str, event_name : str, mission_name : str) -> dict:
        """Setup the scenario for the simulation. """

        assert isinstance(scenario_name, str), "scenario_name must be a string"
        assert isinstance(connectivity, str), "connectivity must be a string"
        assert isinstance(event_name, str), "event_name must be a string"
        assert isinstance(mission_name, str), "mission_name must be a string"

        assert os.path.isfile(f"./scenarios/federated_mission/resources/events/{event_name}.csv"), \
            f"Event file not found: {event_name}.csv"
        assert os.path.isfile(f"./scenarios/federated_mission/resources/missions/{mission_name}.json"), \
            f"Mission file not found: {mission_name}.json"

        scenario = {
            "connectivity": connectivity,
            "events": {
                "@type": "PREDEF",
                "eventsPath": f"./scenarios/federated_mission/resources/events/{event_name}.csv"
            },
            "clock" : {
                "@type" : "EVENT"
            },
            "scenarioPath" : "./scenarios/federated_mission/",
            "name" : scenario_name,
            "missionsPath" : f"./scenarios/federated_mission/resources/missions/{mission_name}.json"
        }
        return scenario
    
    def setup_scenario_settings(self, scenario_name : str) -> dict:
        """ Setup additional scenario settings for orbitpy propagator. """

        assert isinstance(scenario_name, str), "scenario_name must be a string"
        assert os.path.isdir(f"./scenarios/federated_mission/orbit_data"), \
            f"Orbit data directory not found."

        # create orbitdata output directory if needed
        scenario_orbitdata_dir = f"./scenarios/federated_mission/orbit_data/{scenario_name}"
        if not os.path.isdir(scenario_orbitdata_dir): os.mkdir(scenario_orbitdata_dir)

        # create orbitdata settings dictionary
        settings = {
                "coverageType": "GRID COVERAGE",
                "outDir" : f"./scenarios/federated_mission/orbit_data/{scenario_name}",
            }
        return settings

    def test_toy_scenario(self):
        # setup scenario parameters
        duration = 1.0 / 24.0
        grid_name = 'toy_points'
        scenario_name = 'toy_scenario'
        connectivity = 'FULL'
        event_name = 'toy_events'
        mission_name = 'toy_missions'

        # terminal welcome message
        print_welcome(f'Federated Mission Scenario Test: `{scenario_name}`')

        # Generate scenario
        satellite = self.spacecraft_template.copy()
        satellite['orbitState'] = {
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
                    "inc": 0.0,
                    "raan": 0.0,
                    "aop": 0.0,
                    "ta": 95.0
                }
            }

        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[satellite]
                                                   )


        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # check type of mission object
        self.assertTrue(isinstance(self.simulation, Simulation))

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.print_results()

        print('DONE')

    # def test_single_sat_scenario_heuristic(self):
    #     # setup scenario parameters
    #     duration = 2.0 / 24.0
    #     grid_name = 'lake_event_points'
    #     scenario_name = 'single_sat_scenario-heuristic'
    #     connectivity = 'FULL'
    #     event_name = 'lake_events_seed-1000'
    #     mission_name = 'toy_missions'

    #     spacecraft = self.spacecraft_template.copy()
    #     spacecraft['planner']['preplanner']['@type'] = 'heuristic'

    #     # terminal welcome message
    #     print_welcome(f'Federated Mission Scenario Test: `{scenario_name}`')

    #     # Generate scenario
    #     scenario_specs = self.setup_scenario_specs(duration,
    #                                                grid_name, 
    #                                                scenario_name, 
    #                                                connectivity,
    #                                                event_name,
    #                                                mission_name,
    #                                                spacecraft=[spacecraft]
    #                                                )


    #     # initialize mission
    #     self.simulation : Simulation = Simulation.from_dict(scenario_specs)

    #     # check type of mission object
    #     self.assertTrue(isinstance(self.simulation, Simulation))

    #     # execute mission
    #     self.simulation.execute()

    #     # print results
    #     self.simulation.print_results()

    #     print('DONE')

    # def test_single_sat_scenario_earliest(self):

    #     # setup scenario parameters
    #     duration = 2.0 / 24.0
    #     grid_name = 'lake_event_points'
    #     scenario_name = 'single_sat_scenario-earliest'
    #     connectivity = 'FULL'
    #     event_name = 'lake_events_seed-1000'
    #     mission_name = 'toy_missions'

    #     spacecraft = self.spacecraft_template.copy()
    #     spacecraft['planner']['preplanner']['@type'] = 'earliest'

    #     # terminal welcome message
    #     print_welcome(f'Federated Mission Scenario Test: `{scenario_name}`')

    #     # Generate scenario
    #     scenario_specs = self.setup_scenario_specs(duration,
    #                                                grid_name, 
    #                                                scenario_name, 
    #                                                connectivity,
    #                                                event_name,
    #                                                mission_name,
    #                                                spacecraft=[spacecraft]
    #                                                )


    #     # initialize mission
    #     self.simulation : Simulation = Simulation.from_dict(scenario_specs)

    #     # check type of mission object
    #     self.assertTrue(isinstance(self.simulation, Simulation))

    #     # execute mission
    #     self.simulation.execute()

    #     # print results
    #     self.simulation.print_results()

    #     print('DONE')

    # def test_single_sat_scenario_dynamic(self):
    #     # setup scenario parameters
    #     duration = 2.0 / 24.0
    #     grid_name = 'lake_event_points'
    #     scenario_name = 'single_sat_scenario-dynamic'
    #     connectivity = 'FULL'
    #     event_name = 'lake_events_seed-1000'
    #     mission_name = 'toy_missions'

    #     spacecraft = self.spacecraft_template.copy()
    #     spacecraft['planner']['preplanner']['@type'] = 'dynamic'

    #     # terminal welcome message
    #     print_welcome(f'Federated Mission Scenario Test: `{scenario_name}`')

    #     # Generate scenario
    #     scenario_specs = self.setup_scenario_specs(duration,
    #                                                grid_name, 
    #                                                scenario_name, 
    #                                                connectivity,
    #                                                event_name,
    #                                                mission_name,
    #                                                spacecraft=[spacecraft]
    #                                                )


    #     # initialize mission
    #     self.simulation : Simulation = Simulation.from_dict(scenario_specs)

    #     # check type of mission object
    #     self.assertTrue(isinstance(self.simulation, Simulation))

    #     # execute mission
    #     self.simulation.execute()

    #     # print results
    #     self.simulation.print_results()

    #     print('DONE')

    def test_single_sat_scenario_milp(self):
        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'lake_event_points'
        scenario_name = 'single_sat_scenario-milp'
        connectivity = 'FULL'
        event_name = 'lake_events_seed-1000'
        mission_name = 'toy_missions'

        spacecraft = self.spacecraft_template.copy()
        spacecraft['planner']['preplanner']['@type'] = 'milp'
        spacecraft['planner']['preplanner']['licensePath'] = './gurobi.lic'
        spacecraft['planner']['preplanner']['debug'] = 'False'
        spacecraft['planner']['preplanner']['horizon'] =  500

        # terminal welcome message
        print_welcome(f'Federated Mission Scenario Test: `{scenario_name}`')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[spacecraft]
                                                   )


        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # check type of mission object
        self.assertTrue(isinstance(self.simulation, Simulation))

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.print_results()

        print('DONE')

if __name__ == '__main__':

    # run tests
    unittest.main()

# class TestSingleSatNoEventsCase(unittest.TestCase):
#     def setUp(self) -> None:
#         # terminal welcome message
#         print_welcome('Simulation Loading Test')
        
#         # load scenario json file
#         scenario_specs : dict = {
#             "epoch": {
#                 "@type": "GREGORIAN_UT1",
#                 "year": 2020,
#                 "month": 1,
#                 "day": 1,
#                 "hour": 0,
#                 "minute": 0,
#                 "second": 0
#             },
#             "duration": 70 / 60 / 24.0,
#             "propagator": {
#                 "@type": "J2 ANALYTICAL PROPAGATOR",
#             },
#             "spacecraft": [
#                 {
#                     "@id": "thermal_sat_0_0",
#                     "name": "thermal_0",
#                     "spacecraftBus": {
#                         "name": "BlueCanyon",
#                         "mass": 20,
#                         "volume": 0.5,
#                         "orientation": {
#                             "referenceFrame": "NADIR_POINTING",
#                             "convention": "REF_FRAME_ALIGNED"
#                         },
#                         "components": {
#                             "adcs" : {
#                                 "maxTorque" : 1000,
#                                 "maxRate" : 1
#                             }
#                         }
#                     },
#                     "instrument": {
#                         "name": "TIR",
#                         "@id" : "vnir_hyp_imager",
#                         "@type" : "VNIR",
#                         "detectorWidth": 6.6e-6,
#                         "focalLength": 3.6,  
#                         "orientation": {
#                             "referenceFrame": "NADIR_POINTING",
#                             "convention": "REF_FRAME_ALIGNED"
#                         },
#                         "fieldOfViewGeometry": { 
#                             "shape": "RECTANGULAR", 
#                             "angleHeight": 2.5, 
#                             "angleWidth": 5.0
#                         },
#                         "maneuver" : {
#                             "maneuverType":"SINGLE_ROLL_ONLY",
#                             "A_rollMin": -50,
#                             "A_rollMax": 50
#                         },
#                         "spectral_resolution" : "Multispectral"
#                     },
#                     "orbitState": {
#                         "date": {
#                             "@type": "GREGORIAN_UT1",
#                             "year": 2020,
#                             "month": 1,
#                             "day": 1,
#                             "hour": 0,
#                             "minute": 0,
#                             "second": 0
#                         },
#                         "state": {
#                             "@type": "KEPLERIAN_EARTH_CENTERED_INERTIAL",
#                             "sma": 7078,
#                             "ecc": 0.01,
#                             "inc": 60.0,
#                             "raan": 0.0,
#                             "aop": 0.0,
#                             "ta": 0.0
#                         }
#                     },
#                     "planner" : {
#                         "preplanner" : {
#                             "@type" : "milp",
#                             "licensePath": "./gurobi.lic",
#                             "objective" : "duration",
#                             "period": 1000,
#                             # "horizon": 500,
#                         },
#                         # "replanner" : {
#                         #     "@type" : "broadcaster",
#                         #     "mode" : "opportunistic",
#                         #     "period" : 400
#                         # },
#                     },
#                     "science" : {
#                         "@type": "lookup", 
#                         "eventsPath" : "./scenarios/federated_mission/resources/events/no_events.csv"
#                     },
#                     "mission" : "Algal blooms monitoring"
#                 },
#                 # {
#                 #     "@id": "thermal_sat_0_1",
#                 #     "name": "thermal_1",
#                 #     "spacecraftBus": {
#                 #         "name": "BlueCanyon",
#                 #         "mass": 20,
#                 #         "volume": 0.5,
#                 #         "orientation": {
#                 #             "referenceFrame": "NADIR_POINTING",
#                 #             "convention": "REF_FRAME_ALIGNED"
#                 #         },
#                 #         "components": {
#                 #             "adcs" : {
#                 #                 "maxTorque" : 1000,
#                 #                 "maxRate" : 1
#                 #             }
#                 #         }
#                 #     },
#                 #     "instrument": {
#                 #         "name": "TIR",
#                 #         "@id" : "vnir_hyp_imager",
#                 #         "@type" : "VNIR",
#                 #         "detectorWidth": 6.6e-6,
#                 #         "focalLength": 3.6,  
#                 #         "orientation": {
#                 #             "referenceFrame": "NADIR_POINTING",
#                 #             "convention": "REF_FRAME_ALIGNED"
#                 #         },
#                 #         "fieldOfViewGeometry": { 
#                 #             "shape": "RECTANGULAR", 
#                 #             "angleHeight": 2.5, 
#                 #             "angleWidth": 5.0
#                 #         },
#                 #         "maneuver" : {
#                 #             "maneuverType":"SINGLE_ROLL_ONLY",
#                 #             "A_rollMin": -50,
#                 #             "A_rollMax": 50
#                 #         },
#                 #         "spectral_resolution" : "Multispectral"
#                 #     },
#                 #     "orbitState": {
#                 #         "date": {
#                 #             "@type": "GREGORIAN_UT1",
#                 #             "year": 2020,
#                 #             "month": 1,
#                 #             "day": 1,
#                 #             "hour": 0,
#                 #             "minute": 0,
#                 #             "second": 0
#                 #         },
#                 #         "state": {
#                 #             "@type": "KEPLERIAN_EARTH_CENTERED_INERTIAL",
#                 #             "sma": 7078,
#                 #             "ecc": 0.01,
#                 #             "inc": -120.0,
#                 #             "raan": 0.0,
#                 #             "aop": 0.0,
#                 #             "ta": 270.0
#                 #         }
#                 #     },
#                 #     "planner" : {
#                 #         # "preplanner" : {
#                 #         #     "@type" : "heuristic",
#                 #         #     "period": 1000,
#                 #         #     # "horizon": 500,
#                 #         # },
#                 #         "replanner" : {
#                 #             "@type" : "worker",
#                 #             # "mode" : "opportunistic",
#                 #             # "period" : 400
#                 #         },
#                 #     },
#                 #     "science" : {
#                 #         "@type": "lookup", 
#                 #         "eventsPath" : "./scenarios/federated_mission/resources/events/no_events.csv"
#                 #     },
#                 #     "mission" : "Algal blooms monitoring"
#                 # }
#             ],
#             "grid": [
#                 {
#                     "@type": "customGrid",
#                     "covGridFilePath": "./scenarios/federated_mission/resources/grids/lake_event_points.csv"
#                 }
#             ],
#             "scenario": {   
#                 "connectivity" : "LOS", 
#                 "events" : {
#                     "@type": "PREDEF", 
#                     "eventsPath" : "./scenarios/federated_mission/resources/events/no_events.csv"
#                 },
#                 "clock" : {
#                     "@type" : "EVENT"
#                 },
#                 "scenarioPath" : "./scenarios/federated_mission/",
#                 "name" : "single_sat_no_events_case",
#                 "missionsPath" : "./scenarios/federated_mission/resources/missions/missions.json"
#             },
#             "settings": {
#                 "coverageType": "GRID COVERAGE",
#                 "outDir" : "./scenarios/federated_mission/orbit_data/single_sat_no_events_case",
#             }
#         }

#         # set outdir
#         orbitdata_dir = os.path.join('./scenarios/federated_mission', 'orbit_data')
#         scenario_orbitdata_dir = os.path.join(orbitdata_dir, 'single_sat_no_events_case')
#         scenario_specs['settings']['outDir'] = scenario_orbitdata_dir
#         if not os.path.isdir(orbitdata_dir): os.mkdir(orbitdata_dir)
#         if not os.path.isdir(scenario_orbitdata_dir): os.mkdir(scenario_orbitdata_dir)

#         # initialize mission
#         self.simulation : Simulation = Simulation.from_dict(scenario_specs)

#         # check type of mission object
#         self.assertTrue(isinstance(self.simulation, Simulation))

