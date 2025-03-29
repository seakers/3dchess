import os
import unittest

from tqdm import tqdm

from chess3d.agents.actions import ObservationAction
from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.planning.planners.naive import EarliestAccessPlanner
from chess3d.agents.states import SimulationAgentState
from chess3d.mission import Mission
from chess3d.utils import print_welcome
from runtime_plots import plot_scenario_runtime

class TestToyCase(unittest.TestCase):
    def setUp(self) -> None:
        # terminal welcome message
        print_welcome('Naive Planner Test')
        
        # load scenario json file
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
            "duration": 3.0 / 24.0,
            "propagator": {
                "@type": "J2 ANALYTICAL PROPAGATOR",
                "stepSize": 10
            },
            "spacecraft": [
                {
                    "@id": "img_sat_0",
                    "name": "img_0",
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
                        "name": "visible",
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
                            "angleWidth": 45
                        },
                        "@id": "img1",
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
                            "inc": 0.0,
                            "raan": 0.0,
                            "aop": 0.0,
                            "ta": 0.0
                        }
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "naive"
                        },
                        # "replanner" : {
                        #     "@type" : "broadcaster"
                        # },
                        "rewardGrid":{
                            "reward_function" : 'event',
                            'initial_reward' : 1.0,
                            'min_reward' : 1.0,
                            'unobserved_reward_rate' : 2.0, # pts/hrs
                            'max_unobserved_reward' : 10.0,
                            'event_reward' : 10.0
                        }
                    },
                    "science" : {
                        "@type": "lookup", 
                        "eventsPath" : "./tests/naive/resources/toy_events.csv"
                    }
                }
            ],
            "grid": [
                {
                    "@type": "customGrid",
                    "covGridFilePath": "./tests/naive/resources/toy_points.csv"
                }
            ],
            "scenario": {   
                "connectivity" : "FULL", 
                "utility" : "LINEAR",
                "events" : {
                    "@type": "PREDEF", 
                    "eventsPath" : "./tests/naive/resources/toy_events.csv"
                },
                "clock" : {
                    "@type" : "EVENT"
                },
                "scenarioPath" : "./tests/naive/",
                "name" : "toy"
            },
            "settings": {
                "coverageType": "GRID COVERAGE",
                "outDir" : "./tests/naive/orbit_data/toy"
            }
        }

        # set outdir
        orbitdata_dir = os.path.join('./tests', 'naive', 'orbit_data')
        scenario_orbitdata_dir = os.path.join(orbitdata_dir, 'toy')
        scenario_specs['settings']['outDir'] = scenario_orbitdata_dir
        if not os.path.isdir(orbitdata_dir): os.mkdir(orbitdata_dir)
        if not os.path.isdir(scenario_orbitdata_dir): os.mkdir(scenario_orbitdata_dir)

        # initialize mission
        self.mission : Mission = Mission.from_dict(scenario_specs)

        # check type of mission object
        self.assertTrue(isinstance(self.mission, Mission))
        

    def test_planner(self) -> None:
        # execute mission
        self.mission.execute()

        # print results
        self.mission.print_results()

        print('DONE')

class TestBenCase(unittest.TestCase):
    def setUp(self) -> None:
        # terminal welcome message
        print_welcome('Naive Planner Test')
        
        # load scenario json file
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
            "duration": 3.0 / 24.0,
            "propagator": {
                "@type": "J2 ANALYTICAL PROPAGATOR",
                "stepSize": 10
            },
            "spacecraft": [
                {
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
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "naive",
                            # "period": 1000,
                            # "horizon": 1000,
                        },
                        # "replanner" : {
                        #     "@type" : "broadcaster"
                        # },
                        "rewardGrid":{
                            "reward_function" : 'event',
                            'initial_reward' : 1.0,
                            'min_reward' : 1.0,
                            'unobserved_reward_rate' : 2.0, # pts/hrs
                            'max_unobserved_reward' : 10.0,
                            'event_reward' : 10.0
                        }
                    },
                    "science" : {
                        "@type": "lookup", 
                        "eventsPath" : "./tests/naive/resources/all_events_formatted.csv"
                    }
                }
            ],
            "grid": [
                {
                    "@type": "customGrid",
                    "covGridFilePath": "./tests/naive/resources/lake_event_points.csv"
                }
            ],
            "scenario": {   
                "connectivity" : "FULL", 
                "utility" : "LINEAR",
                "events" : {
                    "@type": "PREDEF", 
                    "eventsPath" : "./tests/naive/resources/all_events_formatted.csv"
                },
                "clock" : {
                    "@type" : "EVENT"
                },
                "scenarioPath" : "./tests/naive/",
                "name" : "ben_case"
            },
            "settings": {
                "coverageType": "GRID COVERAGE",
                "outDir" : "./tests/naive/orbit_data/ben_case"
            }
        }

        # set outdir
        orbitdata_dir = os.path.join('./tests', 'naive', 'orbit_data')
        scenario_orbitdata_dir = os.path.join(orbitdata_dir, 'ben_case')
        scenario_specs['settings']['outDir'] = scenario_orbitdata_dir
        if not os.path.isdir(orbitdata_dir): os.mkdir(orbitdata_dir)
        if not os.path.isdir(scenario_orbitdata_dir): os.mkdir(scenario_orbitdata_dir)

        # initialize mission
        self.mission : Mission = Mission.from_dict(scenario_specs)

        # check type of mission object
        self.assertTrue(isinstance(self.mission, Mission))


    # def test_planner(self) -> None:
    #     # execute mission
    #     self.mission.execute()

    #     # print results
    #     self.mission.print_results()

    #     print('DONE')

class TestRandomCase(unittest.TestCase):
    def setUp(self) -> None:
        # terminal welcome message
        print_welcome('Naive Planner Test')
        
        # load scenario json file
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
            "duration": 2 / 24.0,
            "propagator": {
                "@type": "J2 ANALYTICAL PROPAGATOR",
                "stepSize": 10
            },
            "spacecraft": [
                {
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
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "naive",
                            # "period": 500,
                            # "horizon": 'Inf',
                        }
                    },
                    "science" : {
                        "@type": "lookup", 
                        "eventsPath" : "./tests/naive/resources/lake_events.csv"
                    }
                }
            ],
            "grid": [
                {
                    "@type": "customGrid",
                    "covGridFilePath": "./tests/naive/resources/lake_event_points.csv"
                }
            ],
            "scenario": {   
                "connectivity" : "FULL", 
                "utility" : "LINEAR",
                "events" : {
                    "@type": "PREDEF", 
                    "eventsPath" : "./tests/naive/resources/lake_events.csv"
                },
                "clock" : {
                    "@type" : "EVENT"
                },
                "scenarioPath" : "./tests/naive/",
                "name" : "naive"
            },
            "settings": {
                "coverageType": "GRID COVERAGE",
                "outDir" : "./tests/naive/orbit_data/naive"
            }
        }

        # initialize mission
        self.mission : Mission = Mission.from_dict(scenario_specs, overwrite=True)

        # check type of mission object
        self.assertTrue(isinstance(self.mission, Mission))

    # def test_planner(self) -> None:
    #     # execute mission
    #     self.mission.execute()

    #     # print results
    #     self.mission.print_results()

    #     # plot runtime
    #     scenarios = [
    #         'naive'
    #     ]

    #     agents = [
    #         'manager',
    #         'environment',
    #         'thermal_0'
    #     ]
            
    #     for agent in tqdm(agents, desc='Generating runtime performance plots for agents'):
    #         plot_scenario_runtime(scenarios, agent, False, True, './tests/naive/results', './tests/naive')

    #     print('DONE')

if __name__ == '__main__':
    unittest.main()

    # # plot runtime
    # scenarios = [
    #     'naive'
    # ]

    # agents = [
    #     'manager',
    #     'environment',
    #     'thermal_0'
    # ]
        
    # for agent in tqdm(agents, desc='Generating runtime performance plots for agents'):
    #     plot_scenario_runtime(scenarios, agent, False, True, './tests/naive/results')

    # print('DONE')