import copy
import unittest

from tester import AgentTester

from chess3d.simulation import Simulation
from chess3d.utils import print_welcome

class TestGroundStationAgents(AgentTester, unittest.TestCase):
    def test_initializer(self):
        """ Test case for a single satellite in a lake-monitoring scenario. """
        # setup scenario parameters
        duration = 1.0 / 24.0
        grid_name = 'toy_points'
        scenario_name = f'toy_scenario_gs'
        connectivity = 'LOS'
        event_name = 'lake_events_seed-1000'
        mission_name = 'lake_missions'
        gs_network_name='gs_nen'

        # set toy satellites to equatorial orbit
        spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        spacecraft_1['name'] = 'sat-1'
        spacecraft_1['orbitState']['state']['inc'] = 0.0

        spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        spacecraft_2['name'] = 'sat-2'
        spacecraft_2['orbitState']['state']['inc'] = 0.0

        # terminal welcome message
        print_welcome(f'`{scenario_name}` AGENT TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   gs_network_name,
                                                   spacecraft=[spacecraft_1, spacecraft_2]
                                                   )


        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs)


if __name__ == '__main__':
    # run tests
    unittest.main()