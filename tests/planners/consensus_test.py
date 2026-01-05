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
        self.toy_3 = False
        self.toy_4 = False
        self.toy_5 = True

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
                "optimisticBiddingThreshold": 2,
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
        ## TOY CASE 1
        Test case for single satellite performing default mission tasks.

        ### Goals
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
        mission_database = 'toy_missions'

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['mission'] = "toy_mission_1"

        # terminal welcome message
        print_welcome(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_database,
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

        print(f"{scenario_name}: DONE")

    def test_toy_case_2(self):
        """
        ## TOY CASE 2
        Test case for single satellite performing event-driven tasks from announcer.

        ### Goals
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
        - Only one observation opportunity should be available during the event duration.

        ### Expected Outcomes
        - Satellite performs 1 observation of the event.
        - The observation is successfully scheduled and executed without conflicts.
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
        announcer_spacecraft['mission'] = "toy_mission_2"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['mission'] = "toy_mission_2"

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

        print(f"{scenario_name}: DONE")

    def test_toy_case_3(self):
        """
        ## TOY CASE 3
        Test case for two satellite performing event-driven tasks from announcer.

        ### Goals
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
        - Duration: 2 hours
        - Grid: One target at (lat=0.0°, lon=0.0°)
        - Events: One event occurring at t=5 seconds, lasting for 2 hours.
        - Same instruments for both agents
        - Agents offset by 2 degrees in true anomaly
        
        ### Expected Outcomes
        - Both satellites should perform 2 observations of the event each, one per each access window.
        - Agent 1 performs first observation before Agent 2's first observation.
        - Agent 1 performs second observation before Agent 2's second observation but after its first observation.
        - All observations are successfully scheduled and executed without conflicts.
        - The planner effectively tracks the bidding and performance of the observations being scheduled.
        - Final planner results should indicate 2 completed bids per satellite and an empty bundle at the end of the simulation.
        - The environment results should reflect the successful completion of all scheduled observations.
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
        announcer_spacecraft['mission'] = "toy_mission_3"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['mission'] = "toy_mission_3"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_planner_config()
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] = ractive_spacecraft_1['orbitState']['state']['ta'] - 2.0 # phase offset by 2.0[deg]
        ractive_spacecraft_2['mission'] = "toy_mission_3"

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

        print(f"{scenario_name}: DONE")

    def test_toy_case_4(self):
        """
        
        Test case for optimisting bidding between two satellite performing event-driven tasks from announcer.

        ### Goals
        - Ensure that the optimistic bidding mechanism in the consensus planner functions correctly in a multi-agent reactive scenario.
        
        ### Mission Details
        - Default objectives: None
        - Event-driven objectives: respond to event announcements from an announcer satellite.

        ### Agents
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
        - Duration: 2 hours
        - Grid: One target at (lat=0.0°, lon=0.0°)
        - Events: One event occurring at t=5 seconds, lasting for 2 hours.
        - Mission objectives designed such that both agents have conflicting high-priority tasks.
        - Same instruments for both agents
        - Agents offset by 2 degrees in true anomaly

        ### Expected Outcomes
        - Both satellites have 2 observation windows for the event, making a total of 4 possible observations.
        - Agent 1's first observation window is earlier than Agent 2's first window.
        - Agent 1's second observation window is earlier than Agent 2's second window but after Agent 2's first window.
        - Agent 1 is expected to always out-bid Agent 2 for any observation opportunity due to mission objectives.
        - Agent 2 should lose bid to first and second observation in its first bidding round.
        - In the second bidding round, Agent 2 should be able to successfully bid for its second observation opportunity after Agent 1 has secured both of its observations.
        - The optimistic bidding mechanism should allow Agent 2 to plan for its second observation despite initial conflicts.
        - Final planner results should indicate 2 completed bids for Agent 1 and 2 completed bids for Agent 2, with an empty bundle at the end of the simulation.            
        """
        if not self.toy_4: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_4'
        scenario_name = f'toy_4-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_4'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'SAT0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['mission'] = "toy_mission_4"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['mission'] = "toy_mission_4"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_planner_config()
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] = ractive_spacecraft_1['orbitState']['state']['ta'] - 2.0 # phase offset by 2.0[deg]
        ractive_spacecraft_2['mission'] = "toy_mission_4"

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

        print(f"{scenario_name}: DONE")

    def test_toy_case_5(self):
        """
        
        Test case for two satellite performing event-driven tasks from announcer with communication delays.

        ### Goals
        - Validate basic functionality of the bundle-building phase of the consensus planner in a communications-limited scenario.        
        
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
        - Duration: 10 hours
        - Grid: One target at (lat=0.0°, lon=0.0°)
        - Events: One event occurring at t=TBD hours, lasting for TBD hours.
        - Same instruments for both agents
        - Agents orbits offset in true anomaly by TBD degrees and by inclination by TBD degrees.
        - Communication windows between agents 1 and 2 limited to LOS only.
        - Communication betwen announcer and agent 1 is constant. 
        - Communication windows expected to start after SAT1 was able to perform first observation of event.

        ### Expected Outcomes
        - Agent 1 has 3 observation opportunities for the event after its detection and announcement.
        - Agent 2 has only 1 observation opportunity for the event due to orbit phasing.
        - Agent 1's first two observation windows are earlier than Agent 2's only window.
        - Agent 1's third observation window is after Agent 2's only window.
        
        """        
        if not self.toy_5: return

        # setup scenario parameters
        duration = 6.0 / 24.0
        grid_name = 'toy_5'
        scenario_name = f'toy_5-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_5'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'SAT0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['orbitState']['state']['ta'] = -60.0 
        announcer_spacecraft['mission'] = "toy_mission_5"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = -60.0 
        ractive_spacecraft_1['mission'] = "toy_mission_5"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_planner_config()
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 60.0
        # ractive_spacecraft_2['orbitState']['state']['raan'] = 0.0 
        # ractive_spacecraft_2['orbitState']['state']['ta'] = ractive_spacecraft_1['orbitState']['state']['ta'] - 30.0 
        ractive_spacecraft_2['mission'] = "toy_mission_5"

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

        print(f"{scenario_name}: DONE")

    def test_toy_case_6(self):
        """
        ## TOY CASE 6
        Test case for two satellite agents performing two event-driven tasks from announcer without default mission tasks.

        ### Goals
        - Showcase decision-making between conflicting tasks.
        
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
        - Duration: 2 hours
        """

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

        # print(f"{scenario_name}: DONE")

if __name__ == '__main__':
    # run tests
    unittest.main()