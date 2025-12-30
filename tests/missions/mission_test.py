import unittest

import numpy as np

from chess3d.agents.planning.observations import ObservationOpportunity
from chess3d.agents.planning.tasks import DefaultMissionTask, EventObservationTask
from chess3d.mission.attributes import TemporalRequirementAttributes
from chess3d.mission.events import GeophysicalEvent
from chess3d.mission.requirements import SinglePointSpatialRequirement, IntervalInterpolationRequirement, SpatialCoverageRequirement
from chess3d.mission.objectives import EventDrivenObjective, DefaultMissionObjective
from chess3d.mission.mission import Mission
from chess3d.utils import Interval, print_welcome


class TestMission(unittest.TestCase):
    def setUp(self):
        # Define event details
        self.event_type = 'Algal Bloom'
        self.parameter = "Chlorophyll-A"
        self.other_parameter = "Sea Surface Temperature"
        self.instrument = 'VNIR'
        self.target = (34.0522, -118.2437, 0, 0)  # Example target: (lat, lon, grid_index, gp_index)
        self.event = GeophysicalEvent(self.event_type, self.target, 0.0, 100, 1.0)
        
        # Define requirements
        self.req_1_1 = SinglePointSpatialRequirement(target=self.target, distance_threshold=10.0)
        self.req_1_2 = IntervalInterpolationRequirement(TemporalRequirementAttributes.REVISIT_TIME.value, [0, 10.0], [1.0, 0.0])
        self.req_2_1 = SinglePointSpatialRequirement(target=self.target, distance_threshold=10.0)
        self.req_2_2 = IntervalInterpolationRequirement(TemporalRequirementAttributes.REVISIT_TIME.value, [0, 5.0], [1.0, 0.0])

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
        self.other_default_objective = DefaultMissionObjective(
            parameter=self.parameter,
            requirements=[self.req_1_1, self.req_1_2]
        )
        self.other_param_default_objective = DefaultMissionObjective(
            parameter=self.other_parameter,
            requirements=[self.req_2_1, self.req_2_2]
        )

        # Create a mission with these objectives
        self.mission = Mission(
            name='TestMission',
            objectives=[self.event_objective, self.default_objective, self.other_default_objective, self.other_param_default_objective],
            weights=[0.6, 0.3, 0.1, 0.0]
        )

    def test_constructor(self):
        # Test mission attributes
        self.assertIsInstance(self.mission, Mission)
        self.assertEqual(self.mission.name, 'testmission')
        self.assertIn(self.event_objective, self.mission.objectives)
        self.assertIn(self.default_objective, self.mission.objectives)
        self.assertAlmostEqual(self.mission.objectives[self.event_objective], 0.6)
        self.assertAlmostEqual(self.mission.objectives[self.default_objective], 0.3)
        self.assertAlmostEqual(self.mission.objectives[self.other_default_objective], 0.1)
        self.assertAlmostEqual(self.mission.objectives[self.other_param_default_objective], 0.0)
        self.assertEqual(len(self.mission.objectives), 4)
        self.assertAlmostEqual(sum(self.mission.objectives.values()), 1.0)
        
        # Test invalid preferences
        self.assertRaises(AssertionError, Mission, name=12345, objectives=[self.event_objective, self.default_objective], weights=[0.6, 0.4]) # invalid mission name
        self.assertRaises(AssertionError, Mission, name='InvalidMission', objectives=[], weights=[]) # No objectives
        self.assertRaises(AssertionError, Mission, name='InvalidMission', objectives=[self.event_objective, self.default_objective, 'invalid_objective'], weights=[0.6, 0.4]) # invalid objective type
        self.assertRaises(AssertionError, Mission, name='InvalidMission', objectives=[self.event_objective, self.default_objective], weights=[0.6, 0.2, 0.2]) # mismatched objective and weight lengths
        self.assertRaises(AssertionError, Mission, name='InvalidMission', objectives=[self.event_objective, self.default_objective], weights=[0.6, 0.5]) # weights don't sum to 1.0

    def test_relate_objectives_to_task(self):
        # Create tasks
        default_task = DefaultMissionTask(
            parameter=self.parameter,
            location=self.target,
            mission_duration=100,
            priority=1.0,
            objective=self.default_objective
        )
        default_task_no_obj = DefaultMissionTask(
            parameter=self.parameter,
            location=self.target,
            mission_duration=100,
            priority=1.0
        )
        event_task = EventObservationTask(
            parameter=self.parameter,
            location=[self.target],
            availability=Interval(0, 100),
            priority=1.0,
            event=self.event,
            objective=self.event_objective
        )
        event_task_no_obj = EventObservationTask(
            parameter=self.parameter,
            location=[self.target],
            availability=Interval(0, 100),
            priority=1.0,
            event=self.event
        )

        # Test relating objectives to default task
        relevances_default = self.mission.relate_objectives_to_task(default_task)
        self.assertAlmostEqual(relevances_default[self.default_objective], 1) # exact match
        self.assertAlmostEqual(relevances_default[self.other_default_objective], 0.75) # same parameter, same type
        self.assertAlmostEqual(relevances_default[self.event_objective], 0.5) # same parameter, different type
        self.assertAlmostEqual(relevances_default[self.other_param_default_objective], 0.25) # different parameter, same type

        relevances_default = self.mission.relate_objectives_to_task(default_task_no_obj) # no specific objective
        self.assertAlmostEqual(relevances_default[self.default_objective], 0.5) # same parameter and type
        self.assertAlmostEqual(relevances_default[self.event_objective], 0.25) # same parameter, different type
        self.assertAlmostEqual(relevances_default[self.other_param_default_objective], 0.0) # different parameter

        relevances_default = self.mission.relate_objectives_to_task(event_task)
        self.assertAlmostEqual(relevances_default[self.default_objective], 0.5) # same parameter, different type
        self.assertAlmostEqual(relevances_default[self.event_objective], 1.0) # exact match
        self.assertAlmostEqual(relevances_default[self.other_param_default_objective], 0.0) # different parameter, different type

        relevances_default = self.mission.relate_objectives_to_task(event_task_no_obj) # no specific objective
        self.assertAlmostEqual(relevances_default[self.default_objective], 0.25) # same parameter, different type
        self.assertAlmostEqual(relevances_default[self.event_objective], 0.50) # same parameter and type
        self.assertAlmostEqual(relevances_default[self.other_param_default_objective], 0.0) # different parameter

    def test_task_value(self):
        # create observation opportunities
        self.task_1 = DefaultMissionTask(self.parameter, self.target, 100, 1.0, self.default_objective)
        self.task_2 = EventObservationTask(self.parameter, self.target, Interval(0, 100), 1.0, self.event, self.event_objective)
        self.obs = ObservationOpportunity([self.task_1, self.task_2], self.instrument, Interval(0, 100), 0.0, Interval(-90,90))

        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,                # best revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,              # zero duration; zero cost
            TemporalRequirementAttributes.OBS_TIME.value: 50.0          # observation time within task availability
        }
        # obs_val = self.mission.calc_observation_opportunity_value(self.obs, perf)
        task_1_val = self.mission.calc_task_value(self.task_1, perf)
        task_2_val = self.mission.calc_task_value(self.task_2, perf)
        self.assertAlmostEqual(self.mission.calc_observation_opportunity_utility(self.obs, perf), 1.0)


    # def test_observation_opportunity_utility(self):
        # # create observation opportunities
        # self.task_1 = DefaultMissionTask(self.parameter, self.target, np.Inf, 1.0, self.default_objective)
        # self.task_2 = EventObservationTask(self.parameter, self.target, Interval(0, np.Inf), 1.0, self.event, self.event_objective)
        # self.obs = ObservationOpportunity([self.task_1, self.task_2], self.instrument, Interval(0, 100), 0.0, Interval(-90,90))

        # perf = {
        #     SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
        #     TemporalRequirementAttributes.REVISIT_TIME.value: 0,                # best revisit time
        #     TemporalRequirementAttributes.DURATION.value: 0.0,              # zero duration; zero cost
        # }
        # # obs_val = self.mission.calc_observation_opportunity_value(self.obs, perf)
        # task_1_val = self.mission.calc_task_value(self.task_1, perf)
        # task_2_val = self.mission.calc_task_value(self.task_2, perf)
        # self.assertAlmostEqual(self.mission.calc_observation_opportunity_utility(self.obs, perf), 1.0)

    #     perf = {
    #         SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0),  # same target as requirement
    #         TemporalRequirementAttributes.REVISIT_TIME.value: 2.5,             # worse revisit time
    #         TemporalRequirementAttributes.DURATION.value: 0.0,              # zero duration; zero cost            
    #     }
    #     perf = {
    #         SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0),  # same target as requirement
    #         TemporalRequirementAttributes.REVISIT_TIME.value: 5.0,             # worse revisit time
    #         TemporalRequirementAttributes.DURATION.value: 0.0,              # zero duration; zero cost
    #     }
    #     perf = {
    #         SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0),  # same target as requirement
    #         TemporalRequirementAttributes.REVISIT_TIME.value: 10,             # worst revisit time
    #         TemporalRequirementAttributes.DURATION.value: 0.0,              # zero duration; zero cost
    #     }
    #     perf = {
    #         SpatialCoverageRequirement.ATTRIBUTE: (35.0522, -129.0, 0, 1),  # different target
    #         TemporalRequirementAttributes.REVISIT_TIME.value: 0,             # best revisit time
    #         TemporalRequirementAttributes.DURATION.value: 0.0,              # zero duration; zero cost
    #     }
    #     perf = {
    #         SpatialCoverageRequirement.ATTRIBUTE: (35.0522, -129.0, 0, 1),  # different target
    #         # missing revisit time
    #         TemporalRequirementAttributes.DURATION.value: 0.0,              # zero duration; zero cost
    #     }
    #     perf = {
    #         SpatialCoverageRequirement.ATTRIBUTE: (35.0522, -129.0, 0, 1),  # different target
    #         TemporalRequirementAttributes.REVISIT_TIME.value: 0,             # best revisit time
    #         12345 : 'invalid_key',  # invalid key type
    #         TemporalRequirementAttributes.DURATION.value: 0.0,              # zero duration; zero cost
    #     }


if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Mission Definitions Test')
    
    # run tests
    unittest.main()