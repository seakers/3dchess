import copy
import unittest

from chess3d.simulation import Simulation
from chess3d.utils import print_welcome
from tester import PlannerTester


class TestConsensusPlanner(PlannerTester, unittest.TestCase):
    def toy_planner_config(self):
        return {
            "preplanner": {
                "@type": "earliest",
                "debug": "False",
                "period" : 250,
            },
            "replanner": {
                "@type": "consensus",
                "model": "heuristic_insertion",
                "replan_threshold": 1,
                "debug": "False"
            }
        }

    def planner_name(self):
        return "consensus"
    
    def test_single_sat_toy(self):
        pass
    
    def test_multiple_sats_toy(self):
        pass

    def test_single_sat_lakes(self):
        """ Test case for a single satellite in a lake-monitoring scenario. """
        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'lake_event_points'
        scenario_name = f'single_sat_lake_scenario-{self.planner_name()}'
        connectivity = 'FULL'
        event_name = 'lake_events_seed-1000'
        mission_name = 'lake_missions'

        spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        spacecraft['planner'] = self.toy_planner_config()
        spacecraft['science'] = {
                        "@type": "lookup", 
                        "eventsPath" : "./tests/planners/resources/events/lake_events_seed-1000.csv"
                    }

        # terminal welcome message
        print_welcome(f'`{scenario_name}` PLANNER TEST')

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

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.print_results()

        print('DONE')

    def test_multiple_sats_lakes(self):
        pass

if __name__ == '__main__':
    # run tests
    unittest.main()