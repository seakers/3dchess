import os
import unittest

import pandas as pd

from chess3d.simulation import Simulation
from chess3d.utils import print_welcome

class TestNaivePlanner(unittest.TestCase):
    def setUp(self) -> None:
        # terminal welcome message
        print_welcome('Naive Planner Test')
        
        # load scenario json file
        self.scenario_specs : dict = {
            "epoch": {
                "@type": "GREGORIAN_UT1",
                "year": 2020,
                "month": 1,
                "day": 1,
                "hour": 0,
                "minute": 0,
                "second": 0
            },
            "duration": 0.1,
            "propagator": {
                "@type": "J2 ANALYTICAL PROPAGATOR",
                "stepSize": 10
            },
            "spacecraft": [
                
            ],
            "grid": [
                {
                    "@type": "customGrid",
                    "covGridFilePath": "./tests/events/resources/points.csv"
                }
            ],
            "scenario": {   
                "connectivity" : "FULL", 
                "utility" : "LINEAR",
                "events" : {
                    "@type": "random", 
                    "numberOfEvents" : 1000,
                    "duration" : 3600,
                    "minSeverity" : 0.0,
                    "maxSeverity" : 100,
                    "measurements" : ['sar', 'visual', 'thermal']
                },
                "clock" : {
                    "@type" : "EVENT"
                },
                "scenarioPath" : "./tests/events/",
                "name" : "events"
            },
            "settings": {
                "coverageType": "GRID COVERAGE",
                "outDir" : "./tests/events/orbit_data"
            }
        }

        # initialize mission
        self.mission : Simulation = Simulation.from_dict(self.scenario_specs)

    def test_events(self) -> None:
        # load events parameters
        scenario_dict = self.scenario_specs['scenario']
        events_config_dict = scenario_dict['events']
        sim_duration = float(self.scenario_specs['duration'])

        n_events = int(events_config_dict.get('numberOfEvents', None))
        event_duration = float(events_config_dict.get('duration', None))
        max_severity = float(events_config_dict.get('maxSeverity', None)) 
        min_severity = float(events_config_dict.get('minSeverity', None)) 
        measurements = events_config_dict.get('measurements', None)

        # load generated events
        events_path = os.path.join('./tests/events/resources/','random_events.csv')
        events : pd.DataFrame = pd.read_csv(events_path)
        
        # TODO: load ground points

        # check event parameters 
        self.assertEqual(n_events,len(events.values))
        for lat,lon,t_start,duration,severity,event_measurements in events.values:
            self.assertLessEqual(sim_duration,t_start)
            self.assertAlmostEqual(event_duration, duration)
            self.assertGreaterEqual(severity,min_severity)
            self.assertGreaterEqual(max_severity,severity)
            
            event_measurements : str
            measurement_str = event_measurements.replace('[','')
            measurement_str = measurement_str.replace(']','')
            measurement_list : list = measurement_str.split(',')

            self.assertTrue(all([m in measurements for m in measurement_list]))

if __name__ == '__main__':
    unittest.main()