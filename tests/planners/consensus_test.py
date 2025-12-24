import copy
import os
import unittest

from chess3d.simulation import Simulation
from chess3d.utils import print_welcome
from tests.planners.tester import PlannerTester


class TestConsensusPlanner(PlannerTester, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        # test case toggles
        self.single_sat_toy = False
        self.single_sat_announcer_toy = True
        self.multiple_sat_toy = False
        self.single_sat_lakes = False
        self.multiple_sat_lakes = False

    def toy_planner_config(self):
        return {
            # "preplanner": {
            #     "@type": "heuristic",
            #     "debug": "False",
            #     # "period" : 250,
            # },
            "replanner": {
                "@type": "consensus",
                "model": "heuristicInsertion",
                "heuristic" : "taskPriority",
                "replanThreshold": 2,
                "debug": "True"
            }
        }
    
    def lakes_planner_config(self):
        return {
            "preplanner": {
                "@type": "heuristic",
                "debug": "False",
                # "period" : 250,
            },
            "replanner": {
                "@type": "consensus",
                "model": "heuristicInsertion",
                "heuristic" : "taskValue",
                "replanThreshold": 1,
                "debug": "False"
            }
        }
        
    def setup_announcer_config(self, event_name : str = None) -> dict:
        """ Setup announcer planner configuration for the scenario. """

        # default to no planner
        if event_name is None: return {}

        # validate event file exists
        assert isinstance(event_name, str), "event_name must be a string"
        assert os.path.isfile(f"./tests/planners/resources/events/{event_name}.csv"), \
            f"Event file not found: {event_name}.csv"
        
        # return event announcer planner config
        return {
                "preplanner": {
                    "@type": "eventAnnouncer",
                    "debug": "False",                        
                    "eventsPath" : f"./tests/planners/resources/events/{event_name}.csv"
                }
            }

    def planner_name(self):
        return "consensus"
    
    def test_single_sat_announcer_toy(self):
        """ Test case for single satellite receiving requests from the announcer agent. """
        # check for case toggle 
        if not self.single_sat_announcer_toy: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_points'
        scenario_name = f'single_sat_announcer_toy_scenario-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_events'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'SAT0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(None)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        # ractive_spacecraft_1['orbitState']['state']['ta'] = announcer_spacecraft['orbitState']['state']['ta'] - 2.0 # phase offset by 2.0[deg]

        # if 'replanner' in announcer_spacecraft['planner']: announcer_spacecraft["planner"].pop('replanner') # make announcer purely preplanner

        # terminal welcome message
        print_welcome(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                       ractive_spacecraft_1
                                                    ]
                                                   )


        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.print_results()

        print('DONE')

if __name__ == '__main__':
    # run tests
    unittest.main()