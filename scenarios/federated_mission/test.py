import json
import os
import unittest

from chess3d.mission.mission import Mission
from chess3d.utils import print_welcome


class TestMissionLoading(unittest.TestCase):
    def setUp(self) -> None:
        mission_dict_path = './scenarios/federated_mission/resources/missions/missions.json'

        with open(mission_dict_path) as f:
            self.missions_dict = json.load(f)

    def test_mission_loading(self) -> None:
        missions = [
            Mission.from_dict(mission_data) 
            for mission_data in self.missions_dict.get("missions", [])
        ]
        x = 1

# class TestToySatCase(unittest.TestCase):
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
#             "duration": 1.0 / 24.0,
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
#                         # "name": "Altimeter",
#                         # "@id" : "altimeter",
#                         # "@type" : "Altimeter",
#                         # "chirpBandwidth": 150e6,
#                         # "pulseWidth": 50e-6,  
#                         # "orientation": {
#                         #     "referenceFrame": "NADIR_POINTING",
#                         #     "convention": "REF_FRAME_ALIGNED"
#                         # },
#                         # "fieldOfViewGeometry": { 
#                         #     "shape": "RECTANGULAR", 
#                         #     "angleHeight": 2.5, 
#                         #     "angleWidth": 45.0
#                         # },
#                         # "maneuver" : {
#                         #     "maneuverType":"SINGLE_ROLL_ONLY",
#                         #     "A_rollMin": -50,
#                         #     "A_rollMax": 50
#                         # }
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
#                             "angleWidth": 45.0
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
#                             "inc": 0.0,
#                             "raan": 0.0,
#                             "aop": 0.0,
#                             "ta": 95.0
#                         }
#                     },
#                     "planner" : {
#                         "preplanner" : {
#                             "@type" : "heuristic",
#                             "period": 1000,
#                             # "horizon": 500,
#                         },
#                         # "replanner" : {
#                         #     "@type" : "broadcaster",
#                         #     "period" : 400
#                         # },
#                     },
#                     "science" : {
#                         "@type": "lookup", 
#                         "eventsPath" : "./scenarios/federated_mission/resources/events/toy_events.csv"
#                     },
#                     "mission" : "Algal blooms monitoring"
#                 }
#             ],
#             "grid": [
#                 {
#                     "@type": "customGrid",
#                     "covGridFilePath": "./scenarios/federated_mission/resources/grids/toy_points.csv"
#                 }
#             ],
#             "scenario": {   
#                 "connectivity" : "FULL", 
#                 "events" : {
#                     "@type": "PREDEF", 
#                     "eventsPath" : "./scenarios/federated_mission/resources/events/toy_events.csv"
#                 },
#                 "clock" : {
#                     "@type" : "EVENT"
#                 },
#                 "scenarioPath" : "./scenarios/federated_mission/",
#                 "name" : "toy_sat_case",
#                 "missionsPath" : "./scenarios/federated_mission/resources/missions/missions.json"
#             },
#             "settings": {
#                 "coverageType": "GRID COVERAGE",
#                 "outDir" : "./scenarios/federated_mission/orbit_data/toy_sat_case",
#             }
#         }

#         # set outdir
#         orbitdata_dir = os.path.join('./scenarios/federated_mission', 'orbit_data')
#         scenario_orbitdata_dir = os.path.join(orbitdata_dir, 'toy_sat_case')
#         if not os.path.isdir(orbitdata_dir): os.mkdir(orbitdata_dir)
#         if not os.path.isdir(scenario_orbitdata_dir): os.mkdir(scenario_orbitdata_dir)

#         # initialize mission
#         self.simulation : Simulation = Simulation.from_dict(scenario_specs)

#         # check type of mission object
#         self.assertTrue(isinstance(self.simulation, Simulation))


#     # def test_planner(self) -> None:
#     #     # execute mission
#     #     self.simulation.execute()

#     #     # print results
#     #     self.simulation.print_results()

#     #     print('DONE')

# class TestSingleSatCase(unittest.TestCase):
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
#             "duration": 1500 / 3600 / 24.0,
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
#                             "ta": 95.0
#                         }
#                     },
#                     "planner" : {
#                         "preplanner" : {
#                             "@type" : "heuristic",
#                             "period": 1000,
#                             # "horizon": 500,
#                         },
#                         # "replanner" : {
#                         #     "@type" : "broadcaster",
#                         #     "period" : 400
#                         # },
#                     },
#                     "science" : {
#                         "@type": "lookup", 
#                         "eventsPath" : "./scenarios/federated_mission/resources/events/toy_events.csv"
#                     },
#                     "mission" : "Algal blooms monitoring"
#                 }
#             ],
#             "grid": [
#                 {
#                     "@type": "customGrid",
#                     "covGridFilePath": "./scenarios/federated_mission/resources/grids/lake_event_points.csv"
#                 }
#             ],
#             "scenario": {   
#                 "connectivity" : "FULL", 
#                 "events" : {
#                     "@type": "PREDEF", 
#                     "eventsPath" : "./scenarios/federated_mission/resources/events/lake_events_seed-1000.csv"
#                 },
#                 "clock" : {
#                     "@type" : "EVENT"
#                 },
#                 "scenarioPath" : "./scenarios/federated_mission/",
#                 "name" : "single_sat_case",
#                 "missionsPath" : "./scenarios/federated_mission/resources/missions/missions.json"
#             },
#             "settings": {
#                 "coverageType": "GRID COVERAGE",
#                 "outDir" : "./scenarios/federated_mission/orbit_data/single_sat_case",
#             }
#         }

#         # set outdir
#         orbitdata_dir = os.path.join('./scenarios/federated_mission', 'orbit_data')
#         scenario_orbitdata_dir = os.path.join(orbitdata_dir, 'single_sat_case')
#         scenario_specs['settings']['outDir'] = scenario_orbitdata_dir
#         if not os.path.isdir(orbitdata_dir): os.mkdir(orbitdata_dir)
#         if not os.path.isdir(scenario_orbitdata_dir): os.mkdir(scenario_orbitdata_dir)

#         # initialize mission
#         self.simulation : Simulation = Simulation.from_dict(scenario_specs)

#         # check type of mission object
#         self.assertTrue(isinstance(self.simulation, Simulation))


#     # def test_planner(self) -> None:
#     #     # execute mission
#     #     self.simulation.execute()

#     #     # print results
#     #     self.simulation.print_results()

#     #     print('DONE')

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


#     # def test_case(self) -> None:
#     #     # execute mission
#     #     self.simulation.execute()

#     #     # print results
#     #     self.simulation.print_results()

#     #     print('DONE')


if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Federeated Mission Scenario Test')
    
    # run tests
    unittest.main()