import unittest

from chess3d.mission.events import GeophysicalEvent
from chess3d.mission.requirements import SinglePointSpatialRequirement, IntervalInterpolationRequirement, TemporalRequirementAttributes
from chess3d.mission.objectives import EventDrivenObjective, DefaultMissionObjective
from chess3d.mission.mission import Mission
from chess3d.utils import print_welcome


class TestMission(unittest.TestCase):
    def setUp(self):
        # Define event details
        self.event_type = 'Algal Bloom'
        self.parameter = "Chlorophyll-A"
        self.target_1 = (34.0522, -118.2437, 0, 0)  # Example target: (lat, lon, grid_index, gp_index)
        self.event = GeophysicalEvent(self.event_type, self.target_1, 0.0, 1000, 1.0)
        
        # Define requirements
        self.req_1_1 = SinglePointSpatialRequirement(target=self.target_1, distance_threshold=10.0)
        self.req_1_2 = IntervalInterpolationRequirement(TemporalRequirementAttributes.REVISIT_TIME.value, [0, 10], [1.0, 0.0])

        # Define objectives
        self.event_objective = EventDrivenObjective(
            event_type=self.event_type,
            parameter=self.parameter,
            requirements=[self.req_1_1, self.req_1_2]
        )
        self.default_objective = DefaultMissionObjective(
            parameter=self.parameter,
            requirements=[self.req_1_1, self.req_1_2]
        )

        # Create a mission with these objectives
        self.mission = Mission(
            name='TestMission',
            objectives=[self.event_objective, self.default_objective],
            weights=[0.6, 0.4]
        )

    def test_constructor(self):
        # Test mission attributes
        self.assertIsInstance(self.mission, Mission)
        self.assertEqual(self.mission.name, 'testmission')
        self.assertIn(self.event_objective, self.mission.objectives)
        self.assertIn(self.default_objective, self.mission.objectives)
        self.assertAlmostEqual(self.mission.objectives[self.event_objective], 0.6)
        self.assertAlmostEqual(self.mission.objectives[self.default_objective], 0.4)
        self.assertEqual(len(self.mission.objectives), 2)
        self.assertEqual(sum(self.mission.objectives.values()), 1.0)
        
        # Test invalid preferences
        self.assertRaises(AssertionError, Mission, name=12345, objectives=[self.event_objective, self.default_objective], weights=[0.6, 0.4]) # invalid mission name
        self.assertRaises(AssertionError, Mission, name='InvalidMission', objectives=[], weights=[]) # No objectives
        self.assertRaises(AssertionError, Mission, name='InvalidMission', objectives=[self.event_objective, self.default_objective, 'invalid_objective'], weights=[0.6, 0.4]) # invalid objective type
        self.assertRaises(AssertionError, Mission, name='InvalidMission', objectives=[self.event_objective, self.default_objective], weights=[0.6, 0.2, 0.2]) # mismatched objective and weight lengths
        self.assertRaises(AssertionError, Mission, name='InvalidMission', objectives=[self.event_objective, self.default_objective], weights=[0.6, 0.5]) # weights don't sum to 1.0

    def test_observation_opportunity_utility(self):
        # Create a mock observation opportunity and measurement
        class MockObservationOpportunity:
            def __init__(self):
                self.tasks = []
        
        obs_opportunity = MockObservationOpportunity()
        measurement = {self.parameter: 5.0}  # Example measurement data

        # Test utility calculation
        utility = self.mission.calc_observation_opportunity_utility(obs_opportunity, measurement, norm_param=0.01)
        self.assertIsInstance(utility, float)

        # Test invalid inputs
        self.assertRaises(AssertionError, self.mission.calc_observation_opportunity_utility, obs='invalid_obs', measurement=measurement, norm_param=0.01) # invalid obs type
        self.assertRaises(AssertionError, self.mission.calc_observation_opportunity_utility, obs=obs_opportunity, measurement='invalid_measurement', norm_param=0.01) # invalid measurement type
        self.assertRaises(AssertionError, self.mission.calc_observation_opportunity_utility, obs=obs_opportunity, measurement=measurement, norm_param=-0.01) # invalid norm_param type

if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Mission Definitions Test')
    
    # run tests
    unittest.main()