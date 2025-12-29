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
        ## common cases
        self.single_sat_toy = False
        self.multiple_sat_toy = False
        self.single_sat_lakes = False
        self.multiple_sat_lakes = False

        ## specific cases
        self.toy_1 = False
        self.toy_2 = False
        self.toy_3 = True

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
                "replanThreshold": 1,
                "optimisticBiddingThreshold": 1,
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
    
    def test_toy_case_1(self):
        """ 
        
        Test case for single satellite performing default mission tasks.

        ### GOALS
        - Validate basic functionality of the bundle-building phase of the consensus planner in a simple scenario. 
        - Ensure that the satellite can plan and execute observations of a single target without any events occurring.
        - Ensure repeated obsevations are being tracked properly by planner.
        
        ### Mission Details
        - Default objectives: continuous observation of a predefined target grid.
        - Event-driven objectives: None

        ### Agents
        - SAT1 : 
            - reactive satellite with narrow swath instrument
            - observation capability
            - onboard consensus planner
            - no onboard event-detection        

        ### Scenario  Description
        - Duration: 12 hours
        - Grid: One target at (lat=0.0°, lon=0.0°)
        - Events: No events

        ### Expected Outcomes
        - Satellite performs 7 observations, one per each access window.
        - All observations are successfully scheduled and executed without conflicts.
        - The planner effectively tracks the bidding and performance of the observations being scheduled.
        - Final planner results should indicate 7 completed bids and an empty bundle at the end of the simulation.
        - The environment results should reflect the successful completion of all scheduled observations.
        """

        # check for case toggle 
        if not self.toy_1: return

        # setup scenario parameters
        duration = 12.0 / 24.0
        grid_name = 'toy_1'
        scenario_name = f'toy_1-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_1'
        mission_name = 'toy_missions'

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['mission'] = "Algal bloom monitoring"

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

    def test_toy_case_2(self):
        """
        
        Test case for single satellite performing event-driven tasks from announcer.

        ### GOALS
        - Validate basic functionality of the bundle-building phase of the consensus planner in a reactive scenario. 
        - Ensure that the satellite can plan and execute observations of a single target without any default mission tasks.
        - Ensure event announcements are being processed properly by planner.
        - Ensure repeated obsevations are being tracked properly by planner.
        
        ### Mission Details
        - Default objectives: None
        - Event-driven objectives: respond to event announcements from an announcer satellite.

        ### Agents
        - SAT0 : 
            - announcer satellite
            - no observation capability
            - onboard event-announcer planner
            - no onboard consensus planner
        - SAT1 : 
            - reactive satellite with narrow swath instrument
            - observation capability
            - onboard consensus planner
            - no onboard event-detection        

        ### Scenario  Description
        - Duration: 12 hours
        - Grid: One target at (lat=0.0°, lon=0.0°)
        - Events: One event occurring at t=1000 s, lasting for 2 hours.

        ### Expected Outcomes

        """

        # check for case toggle 
        if not self.toy_2: return

        # setup scenario parameters
        duration = 12.0 / 24.0
        grid_name = 'toy_2'
        scenario_name = f'toy_2-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_2'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'SAT0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['mission'] = "Algal bloom response"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['mission'] = "Algal bloom response"

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

    def test_toy_case_3(self):
        """
        
        Test case for two satellite performing event-driven tasks from announcer.

        ### GOALS
        - Validate the functionality of the consensus planner in a multi-agent reactive scenario.
        
        ### Mission Details
        - Default objectives: None
        - Event-driven objectives: respond to event announcements from an announcer satellite.

        ### Agents
        - SAT0 : 
            - announcer satellite
            - no observation capability
            - onboard event-announcer planner
            - no onboard consensus planner
        - SAT1 : 
            - reactive satellite with narrow swath instrument
            - observation capability
            - onboard consensus planner
            - no onboard event-detection        
        - SAT2 : 
            - reactive satellite with narrow swath instrument
            - observation capability
            - onboard consensus planner
            - no onboard event-detection        

        ### Scenario  Description
        - Duration: 12 hours
        - Grid: One target at (lat=0.0°, lon=0.0°)
        - Events: One event occurring at t=2 hours, lasting for 1 hour.
        - Same instruments for both agents
        - Agents offset by 2 degrees in true anomaly
        
        ### Expected Outcomes

        """
        # check for case toggle 
        if not self.toy_3: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_3'
        scenario_name = f'toy_3-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_3'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'SAT0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['mission'] = "Algal bloom response"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['mission'] = "Algal bloom response"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_planner_config()
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] = ractive_spacecraft_1['orbitState']['state']['ta'] - 2.0 # phase offset by 2.0[deg]
        ractive_spacecraft_2['mission'] = "Algal bloom response"

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
                                                       ractive_spacecraft_1,
                                                       ractive_spacecraft_2
                                                    ]
                                                   )


        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.print_results()

    # def test_single_sat_announcer_toy(self):
        # # check for case toggle 
        # if not self.single_sat_announcer_toy: return

        # # setup scenario parameters
        # duration = 2.0 / 24.0
        # grid_name = 'toy_points'
        # scenario_name = f'single_sat_announcer_toy_scenario-{self.planner_name()}'
        # connectivity = 'LOS'
        # event_name = 'toy_events'
        # mission_name = 'toy_missions'

        # # SAT0 : announcer satellite 
        # announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        # announcer_spacecraft['@id'] = 'sat0_announcer'
        # announcer_spacecraft['name'] = 'SAT0'
        # announcer_spacecraft['planner'] = self.setup_announcer_config(None)
        # announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        # announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        # announcer_spacecraft['mission'] = "Algal bloom response"

        # # SAT1 : reactive satellite with narrow swath instrument
        # ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        # ractive_spacecraft_1['@id'] = 'sat1_vnir'
        # ractive_spacecraft_1['name'] = 'sat1'
        # ractive_spacecraft_1['planner'] = self.toy_planner_config()
        # ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        # ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        # # ractive_spacecraft_1['orbitState']['state']['ta'] = announcer_spacecraft['orbitState']['state']['ta'] - 2.0 # phase offset by 2.0[deg]
        # ractive_spacecraft_1['mission'] = "Algal bloom response"

        # # if 'replanner' in announcer_spacecraft['planner']: announcer_spacecraft["planner"].pop('replanner') # make announcer purely preplanner

        # # terminal welcome message
        # print_welcome(f'`{scenario_name}` PLANNER TEST')

        # # Generate scenario
        # scenario_specs = self.setup_scenario_specs(duration,
        #                                            grid_name, 
        #                                            scenario_name, 
        #                                            connectivity,
        #                                            event_name,
        #                                            mission_name,
        #                                            spacecraft=[
        #                                                announcer_spacecraft,
        #                                                ractive_spacecraft_1
        #                                             ]
        #                                            )


        # # initialize mission
        # self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # # execute mission
        # self.simulation.execute()

        # # print results
        # self.simulation.print_results()

        # print('DONE')

if __name__ == '__main__':
    # run tests
    unittest.main()