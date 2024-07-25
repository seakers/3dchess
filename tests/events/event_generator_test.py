import unittest

from chess3d.mission import Mission
from chess3d.utils import print_welcome

class TestNaivePlanner(unittest.TestCase):
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
                    "covGridFilePath": "./tests/nadir/resources/lake_event_points.csv"
                }
            ],
            "scenario": {   
                "connectivity" : "FULL", 
                "utility" : "LINEAR",
                "events" : {
                    "@type": "PREDEF", 
                    "eventsPath" : "./tests/nadir/resources/all_events_formatted.csv"
                },
                "clock" : {
                    "@type" : "EVENT"
                },
                "scenarioPath" : "./tests/nadir/",
                "name" : "nadir"
            },
            "settings": {
                "coverageType": "GRID COVERAGE",
                "outDir" : "./tests/nadir/orbit_data"
            }
        }

        # initialize mission
        # self.mission : Mission = Mission.from_dict(scenario_specs)

    def test_planner(self) -> None:
        # execute mission
        # self.mission.execute()
        print('DONE')

        self.assertTrue(True)

    def test_outputs(self) -> None:
        # TODO Check outputs
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()