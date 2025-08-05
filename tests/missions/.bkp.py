import os
import unittest

from chess3d.mission.mission import *
from chess3d.mission.requirements import *
from chess3d.mission.objectives import *
from chess3d.simulation import Simulation
from chess3d.utils import print_welcome




class TestToySatCase(unittest.TestCase):
    def setUp(self) -> None:
        # terminal welcome message
        print_welcome('Simulation Loading Test')
        
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
            "duration": 1.0 / 24.0,
            "propagator": {
                "@type": "J2 ANALYTICAL PROPAGATOR",
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
                        # "name": "Altimeter",
                        # "@id" : "altimeter",
                        # "@type" : "Altimeter",
                        # "chirpBandwidth": 150e6,
                        # "pulseWidth": 50e-6,  
                        # "orientation": {
                        #     "referenceFrame": "NADIR_POINTING",
                        #     "convention": "REF_FRAME_ALIGNED"
                        # },
                        # "fieldOfViewGeometry": { 
                        #     "shape": "RECTANGULAR", 
                        #     "angleHeight": 2.5, 
                        #     "angleWidth": 45.0
                        # },
                        # "maneuver" : {
                        #     "maneuverType":"SINGLE_ROLL_ONLY",
                        #     "A_rollMin": -50,
                        #     "A_rollMax": 50
                        # }
                        "name": "TIR",
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
                            "inc": 0.0,
                            "raan": 0.0,
                            "aop": 0.0,
                            "ta": 95.0
                        }
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "heuristic",
                            "period": 1000,
                            # "horizon": 500,
                        },
                        # "replanner" : {
                        #     "@type" : "broadcaster",
                        #     "period" : 400
                        # },
                    },
                    "science" : {
                        "@type": "lookup", 
                        "eventsPath" : "./tests/missions/resources/events/toy_events.csv"
                    },
                    "mission" : "Algal blooms monitoring"
                }
            ],
            "grid": [
                {
                    "@type": "customGrid",
                    "covGridFilePath": "./tests/missions/resources/grids/toy_points.csv"
                }
            ],
            "scenario": {   
                "connectivity" : "FULL", 
                "events" : {
                    "@type": "PREDEF", 
                    "eventsPath" : "./tests/missions/resources/events/toy_events.csv"
                },
                "clock" : {
                    "@type" : "EVENT"
                },
                "scenarioPath" : "./tests/missions/",
                "name" : "toy_sat_case",
                "missionsPath" : "./tests/missions/resources/missions/missions.json"
            },
            "settings": {
                "coverageType": "GRID COVERAGE",
                "outDir" : "./tests/missions/orbit_data/toy_sat_case",
            }
        }

        # set outdir
        orbitdata_dir = os.path.join('./tests/missions', 'orbit_data')
        scenario_orbitdata_dir = os.path.join(orbitdata_dir, 'toy_sat_case')
        if not os.path.isdir(orbitdata_dir): os.mkdir(orbitdata_dir)
        if not os.path.isdir(scenario_orbitdata_dir): os.mkdir(scenario_orbitdata_dir)

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # check type of mission object
        self.assertTrue(isinstance(self.simulation, Simulation))


    # def test_planner(self) -> None:
    #     # execute mission
    #     self.simulation.execute()

    #     # print results
    #     self.simulation.print_results()

    #     print('DONE')

class TestSingleSatCase(unittest.TestCase):
    def setUp(self) -> None:
        # terminal welcome message
        print_welcome('Simulation Loading Test')
        
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
            "duration": 1500 / 3600 / 24.0,
            "propagator": {
                "@type": "J2 ANALYTICAL PROPAGATOR",
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
                        "name": "TIR",
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
                            "angleWidth": 5.0
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
                            "@type" : "heuristic",
                            "period": 1000,
                            # "horizon": 500,
                        },
                        # "replanner" : {
                        #     "@type" : "broadcaster",
                        #     "period" : 400
                        # },
                    },
                    "science" : {
                        "@type": "lookup", 
                        "eventsPath" : "./tests/missions/resources/events/toy_events.csv"
                    },
                    "mission" : "Algal blooms monitoring"
                }
            ],
            "grid": [
                {
                    "@type": "customGrid",
                    "covGridFilePath": "./tests/missions/resources/grids/lake_event_points.csv"
                }
            ],
            "scenario": {   
                "connectivity" : "FULL", 
                "events" : {
                    "@type": "PREDEF", 
                    "eventsPath" : "./tests/missions/resources/events/lake_events_seed-1000.csv"
                },
                "clock" : {
                    "@type" : "EVENT"
                },
                "scenarioPath" : "./tests/missions/",
                "name" : "single_sat_case",
                "missionsPath" : "./tests/missions/resources/missions/missions.json"
            },
            "settings": {
                "coverageType": "GRID COVERAGE",
                "outDir" : "./tests/missions/orbit_data/single_sat_case",
            }
        }

        # set outdir
        orbitdata_dir = os.path.join('./tests/missions', 'orbit_data')
        scenario_orbitdata_dir = os.path.join(orbitdata_dir, 'single_sat_case')
        scenario_specs['settings']['outDir'] = scenario_orbitdata_dir
        if not os.path.isdir(orbitdata_dir): os.mkdir(orbitdata_dir)
        if not os.path.isdir(scenario_orbitdata_dir): os.mkdir(scenario_orbitdata_dir)

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # check type of mission object
        self.assertTrue(isinstance(self.simulation, Simulation))


    # def test_planner(self) -> None:
    #     # execute mission
    #     self.simulation.execute()

    #     # print results
    #     self.simulation.print_results()

    #     print('DONE')

class TestSingleSatNoEventsCase(unittest.TestCase):
    def setUp(self) -> None:
        # terminal welcome message
        print_welcome('Simulation Loading Test')
        
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
            "duration": 70 / 60 / 24.0,
            "propagator": {
                "@type": "J2 ANALYTICAL PROPAGATOR",
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
                        "name": "TIR",
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
                            "angleWidth": 5.0
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
                            "ta": 0.0
                        }
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "milp",
                            "licensePath": "./gurobi.lic",
                            "objective" : "duration",
                            "period": 1000,
                            # "horizon": 500,
                        },
                        # "replanner" : {
                        #     "@type" : "broadcaster",
                        #     "mode" : "opportunistic",
                        #     "period" : 400
                        # },
                    },
                    "science" : {
                        "@type": "lookup", 
                        "eventsPath" : "./tests/missions/resources/events/no_events.csv"
                    },
                    "mission" : "Algal blooms monitoring"
                },
                # {
                #     "@id": "thermal_sat_0_1",
                #     "name": "thermal_1",
                #     "spacecraftBus": {
                #         "name": "BlueCanyon",
                #         "mass": 20,
                #         "volume": 0.5,
                #         "orientation": {
                #             "referenceFrame": "NADIR_POINTING",
                #             "convention": "REF_FRAME_ALIGNED"
                #         },
                #         "components": {
                #             "adcs" : {
                #                 "maxTorque" : 1000,
                #                 "maxRate" : 1
                #             }
                #         }
                #     },
                #     "instrument": {
                #         "name": "TIR",
                #         "@id" : "vnir_hyp_imager",
                #         "@type" : "VNIR",
                #         "detectorWidth": 6.6e-6,
                #         "focalLength": 3.6,  
                #         "orientation": {
                #             "referenceFrame": "NADIR_POINTING",
                #             "convention": "REF_FRAME_ALIGNED"
                #         },
                #         "fieldOfViewGeometry": { 
                #             "shape": "RECTANGULAR", 
                #             "angleHeight": 2.5, 
                #             "angleWidth": 5.0
                #         },
                #         "maneuver" : {
                #             "maneuverType":"SINGLE_ROLL_ONLY",
                #             "A_rollMin": -50,
                #             "A_rollMax": 50
                #         },
                #         "spectral_resolution" : "Multispectral"
                #     },
                #     "orbitState": {
                #         "date": {
                #             "@type": "GREGORIAN_UT1",
                #             "year": 2020,
                #             "month": 1,
                #             "day": 1,
                #             "hour": 0,
                #             "minute": 0,
                #             "second": 0
                #         },
                #         "state": {
                #             "@type": "KEPLERIAN_EARTH_CENTERED_INERTIAL",
                #             "sma": 7078,
                #             "ecc": 0.01,
                #             "inc": -120.0,
                #             "raan": 0.0,
                #             "aop": 0.0,
                #             "ta": 270.0
                #         }
                #     },
                #     "planner" : {
                #         # "preplanner" : {
                #         #     "@type" : "heuristic",
                #         #     "period": 1000,
                #         #     # "horizon": 500,
                #         # },
                #         "replanner" : {
                #             "@type" : "worker",
                #             # "mode" : "opportunistic",
                #             # "period" : 400
                #         },
                #     },
                #     "science" : {
                #         "@type": "lookup", 
                #         "eventsPath" : "./tests/missions/resources/events/no_events.csv"
                #     },
                #     "mission" : "Algal blooms monitoring"
                # }
            ],
            "grid": [
                {
                    "@type": "customGrid",
                    "covGridFilePath": "./tests/missions/resources/grids/lake_event_points.csv"
                }
            ],
            "scenario": {   
                "connectivity" : "LOS", 
                "events" : {
                    "@type": "PREDEF", 
                    "eventsPath" : "./tests/missions/resources/events/no_events.csv"
                },
                "clock" : {
                    "@type" : "EVENT"
                },
                "scenarioPath" : "./tests/missions/",
                "name" : "single_sat_no_events_case",
                "missionsPath" : "./tests/missions/resources/missions/missions.json"
            },
            "settings": {
                "coverageType": "GRID COVERAGE",
                "outDir" : "./tests/missions/orbit_data/single_sat_no_events_case",
            }
        }

        # set outdir
        orbitdata_dir = os.path.join('./tests/missions', 'orbit_data')
        scenario_orbitdata_dir = os.path.join(orbitdata_dir, 'single_sat_no_events_case')
        scenario_specs['settings']['outDir'] = scenario_orbitdata_dir
        if not os.path.isdir(orbitdata_dir): os.mkdir(orbitdata_dir)
        if not os.path.isdir(scenario_orbitdata_dir): os.mkdir(scenario_orbitdata_dir)

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # check type of mission object
        self.assertTrue(isinstance(self.simulation, Simulation))


    # def test_planner(self) -> None:
    #     # execute mission
    #     self.simulation.execute()

    #     # print results
    #     self.simulation.print_results()

    #     print('DONE')
