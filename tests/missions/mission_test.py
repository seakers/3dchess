import unittest

import numpy as np

from chess3d.agents.planning.observations import ObservationOpportunity
from chess3d.agents.planning.tasks import DefaultMissionTask, EventObservationTask
from chess3d.mission.attributes import TemporalRequirementAttributes
from chess3d.mission.events import GeophysicalEvent
from chess3d.mission.requirements import SinglePointSpatialRequirement, IntervalInterpolationRequirement, SpatialCoverageRequirement
from chess3d.mission.objectives import EventDrivenObjective, DefaultMissionObjective, MissionObjective
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
        self.req_1_3 = IntervalInterpolationRequirement(TemporalRequirementAttributes.RESPONSE_TIME.value, [0, 100.0], [1.0, 0.0])
        
        self.req_2_1 = SinglePointSpatialRequirement(target=self.target, distance_threshold=10.0)
        self.req_2_2 = IntervalInterpolationRequirement(TemporalRequirementAttributes.REVISIT_TIME.value, [0, 5.0], [1.0, 0.0])
        self.req_2_3 = IntervalInterpolationRequirement(TemporalRequirementAttributes.RESPONSE_TIME.value, [0, 50.0], [1.0, 0.0])

        # Define objectives
        ## event-driven mission objective
        self.event_objective = EventDrivenObjective(
            event_type=self.event_type,
            parameter=self.parameter,
            requirements=[self.req_1_1, self.req_1_2, self.req_1_3]
        )
        ## default mission objective
        self.default_objective = DefaultMissionObjective(
            parameter=self.parameter,
            requirements=[self.req_1_1, self.req_1_2, self.req_1_3]
        )
        ## other default mission objective, not known by tasks
        self.other_default_objective = DefaultMissionObjective(
            parameter=self.parameter,
            requirements=[self.req_1_1, self.req_1_2, self.req_1_3]
        )
        ## different parameter default mission
        self.other_param_default_objective = DefaultMissionObjective(
            parameter=self.other_parameter,
            requirements=[self.req_2_1, self.req_2_2, self.req_2_3]
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
        self.assertAlmostEqual(relevances_default[self.other_default_objective], 0.5) # same parameter and type
        self.assertAlmostEqual(relevances_default[self.event_objective], 0.25) # same parameter, different type
        self.assertAlmostEqual(relevances_default[self.other_param_default_objective], 0.0) # different parameter

        relevances_event = self.mission.relate_objectives_to_task(event_task)
        self.assertAlmostEqual(relevances_event[self.default_objective], 0.5) # same parameter, different type
        self.assertAlmostEqual(relevances_event[self.other_default_objective], 0.5) # same parameter, different type
        self.assertAlmostEqual(relevances_event[self.event_objective], 1.0) # exact match
        self.assertAlmostEqual(relevances_event[self.other_param_default_objective], 0.0) # different parameter, different type
        
        relevances_event = self.mission.relate_objectives_to_task(event_task_no_obj) # no specific objective
        self.assertAlmostEqual(relevances_event[self.default_objective], 0.25) # same parameter, different type
        self.assertAlmostEqual(relevances_event[self.other_default_objective], 0.25) # same parameter, different type
        self.assertAlmostEqual(relevances_event[self.event_objective], 0.50) # same parameter and type
        self.assertAlmostEqual(relevances_event[self.other_param_default_objective], 0.0) # different parameter

    def test_eval_objective_performance(self):
        """ Sanity check for objective performance evaluation within mission context. """
        # Best Case Scenario
        ## create measurement performance 
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0                 # best observation time from start of task availability
        }

        ## compute performance for each objective
        default_obj_perf = self.default_objective.eval_measurement_performance(perf)
        event_obj_perf = self.event_objective.eval_measurement_performance(perf)

        ## validate performance values
        self.assertAlmostEqual(default_obj_perf, 1.0)
        self.assertAlmostEqual(event_obj_perf, 1.0)
        self.assertAlmostEqual(default_obj_perf, event_obj_perf)

        # Target Requirement 
        ## create measurement performance 
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (40.7128, -74.0060, 0, 1), # different target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0                 # observation time from start of task availability
        }

        ## compute performance for each objective
        default_obj_perf = self.default_objective.eval_measurement_performance(perf)
        event_obj_perf = self.event_objective.eval_measurement_performance(perf)

        ## validate performance values
        self.assertAlmostEqual(default_obj_perf, 0.0)
        self.assertAlmostEqual(event_obj_perf, 0.0)
        self.assertAlmostEqual(default_obj_perf, event_obj_perf)

        # Revisit Time Requirement 
        ## create measurement performance 
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 5.0,            # worse revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0                 # observation time from start of task availability
        }

        ## compute performance for each objective
        default_obj_perf = self.default_objective.eval_measurement_performance(perf)
        event_obj_perf = self.event_objective.eval_measurement_performance(perf)

        ## validate performance values
        self.assertAlmostEqual(default_obj_perf, 0.5)
        self.assertAlmostEqual(event_obj_perf, 0.5)
        self.assertAlmostEqual(default_obj_perf, event_obj_perf)

        ## create measurement performance 
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 10.0,           # worst revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0                 # observation time from start of task availability
        }

        ## compute performance for each objective
        default_obj_perf = self.default_objective.eval_measurement_performance(perf)
        event_obj_perf = self.event_objective.eval_measurement_performance(perf)

        ## validate performance values
        self.assertAlmostEqual(default_obj_perf, 0.0)
        self.assertAlmostEqual(event_obj_perf, 0.0)
        self.assertAlmostEqual(default_obj_perf, event_obj_perf)

        # Observation Time Requirement 
        ## create measurement performance 
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 50.0                # worse relative observation time from start of task availability
        }

        ## compute performance for each objective
        default_obj_perf = self.default_objective.eval_measurement_performance(perf)
        event_obj_perf = self.event_objective.eval_measurement_performance(perf)

        ## validate performance values
        self.assertAlmostEqual(default_obj_perf, 0.5)
        self.assertAlmostEqual(event_obj_perf, 0.5)
        self.assertAlmostEqual(default_obj_perf, event_obj_perf)

        ## create measurement performance 
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 100.0               # worst relative observation time from start of task availability
        }

        ## compute performance for each objective
        default_obj_perf = self.default_objective.eval_measurement_performance(perf)
        event_obj_perf = self.event_objective.eval_measurement_performance(perf)

        ## validate performance values
        self.assertAlmostEqual(default_obj_perf, 0.0)
        self.assertAlmostEqual(event_obj_perf, 0.0)
        self.assertAlmostEqual(default_obj_perf, event_obj_perf)

        # Effect of cost (should be none since cost not included in performance)
        ## create measurement performance 
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 1e6,                # unrealistically long duration; high cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0                # best relative observation time from start of task availability
        }

        ## compute performance for each objective
        default_obj_perf = self.default_objective.eval_measurement_performance(perf)
        event_obj_perf = self.event_objective.eval_measurement_performance(perf)

        ## validate performance values
        self.assertAlmostEqual(default_obj_perf, 1.0)
        self.assertAlmostEqual(event_obj_perf, 1.0)
        self.assertAlmostEqual(default_obj_perf, event_obj_perf)

    def test_task_value(self):
        # create observation opportunities
        task_1 = DefaultMissionTask(self.parameter, self.target, 100, 1.0, self.default_objective)
        task_2 = EventObservationTask(self.parameter, self.target, Interval(0, 100), 10.0, self.event, self.event_objective)
        
        # Error Case Scenario: wrong task type
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }
        self.assertRaises(AssertionError, self.mission.calc_task_value, "invalid_task_type", perf)
        
        # Error Case Scenario: performance type
        self.assertRaises(AssertionError, self.mission.calc_task_value, task_1, "invalid_performance_type")

        # Error Case Scenario: missing observation time in performance
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            # TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # missing observation time
        }
        self.assertRaises(AssertionError, self.mission.calc_task_value, task_1, perf)

        # Best Case Scenario
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }

        task_1_val = self.mission.calc_task_value(task_1, perf)
        task_2_val = self.mission.calc_task_value(task_2, perf)
        self.assertAlmostEqual(task_1_val, 1.0*(0.6*0.5*1.0*1.0+0.3*1.0*1.0*1.0+0.1*0.75*1.0*1.0))  # weight * relevance * performance * availability
        self.assertAlmostEqual(task_2_val, 10.0*(0.6*1.0*1.0*1.0+0.3*0.5*1.0*1.0+0.1*0.50*1.0*1.0))  # weight * relevance * performance * availability

        # Wront Target Scenario
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (85.0522, -40.50, 0, 2),    # different target than requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }

        task_1_val = self.mission.calc_task_value(task_1, perf)
        task_2_val = self.mission.calc_task_value(task_2, perf)
        self.assertAlmostEqual(task_1_val, 0.0)  # weight * relevance * performance * availability
        self.assertAlmostEqual(task_2_val, 0.0)  # weight * relevance * performance * availability

        # Poor Revisit Time Scenario
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 5.0,            # poor revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }

        task_1_val = self.mission.calc_task_value(task_1, perf)
        task_2_val = self.mission.calc_task_value(task_2, perf)
        self.assertAlmostEqual(task_1_val, 1.0*(0.6*0.5*0.5*1.0+0.3*1.0*0.5*1.0+0.1*0.75*0.5*1.0))  # weight * relevance * performance * availability
        self.assertAlmostEqual(task_2_val, 10.0*(0.6*1.0*0.5*1.0+0.3*0.5*0.5*1.0+0.1*0.50*0.5*1.0))  # weight * relevance * performance * availability

        # Worst Revisit Time Scenario
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 100.0,          # worst revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }

        task_1_val = self.mission.calc_task_value(task_1, perf)
        task_2_val = self.mission.calc_task_value(task_2, perf)
        self.assertAlmostEqual(task_1_val, 0.0)  # weight * relevance * performance * availability
        self.assertAlmostEqual(task_2_val, 0.0)  # weight * relevance * performance * availability

        # High Cost Scenario (should not affect value, only utility)
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 1e6,                # very long duration; high cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }

        task_1_val = self.mission.calc_task_value(task_1, perf)
        task_2_val = self.mission.calc_task_value(task_2, perf)
        self.assertAlmostEqual(task_1_val, 1.0*(0.6*0.5*1.0*1.0+0.3*1.0*1.0*1.0+0.1*0.75*1.0*1.0))  # weight * relevance * performance * availability
        self.assertAlmostEqual(task_2_val, 10.0*(0.6*1.0*1.0*1.0+0.3*0.5*1.0*1.0+0.1*0.50*1.0*1.0))  # weight * relevance * performance * availability

        # Outside Availability Scenario
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 101.0               # unavailable observation time
        }

        task_1_val = self.mission.calc_task_value(task_1, perf)
        task_2_val = self.mission.calc_task_value(task_2, perf)
        self.assertAlmostEqual(task_1_val, 0.0)  # weight * relevance * performance * availability
        self.assertAlmostEqual(task_2_val, 0.0)  # weight * relevance * performance * availability

    def test_observation_opportunity_value(self):
        # create observation opportunities
        task_1 = DefaultMissionTask(self.parameter, self.target, 100, 1.0, self.default_objective)
        task_2 = EventObservationTask(self.parameter, self.target, Interval(0, 100), 10.0, self.event, self.event_objective)
        obs = ObservationOpportunity([task_1, task_2], self.instrument, Interval(0, 100), 0.0, Interval(-90,90))
        
        # Error Case Scenario: wrong observation type
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }
        self.assertRaises(AssertionError, self.mission.calc_observation_opportunity_value, "invalid_observation_type", perf)
        
        # Error Case Scenario: wrong measurement type
        self.assertRaises(AssertionError, self.mission.calc_observation_opportunity_value, obs, "invalid_measurement_type")

        # Best Case Scenario
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }

        task_1_val = 0.6*0.5*1.0+0.3*1.0*1.0+0.1*0.75*1.0
        task_2_val = 0.6*1.0*1.0+0.3*0.5*1.0+0.1*0.50*1.0
        obs_val = self.mission.calc_observation_opportunity_value(obs, perf)
        self.assertAlmostEqual(obs_val, 1.0 * task_1_val + 10.0 * task_2_val) # priority * task value

        # Wront Target Scenario
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (85.0522, -40.50, 0, 2),    # different target than requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }

        task_1_val = 0.0
        task_2_val = 0.0
        obs_val = self.mission.calc_observation_opportunity_value(obs, perf)
        self.assertAlmostEqual(obs_val, 1.0 * task_1_val + 10.0 * task_2_val) # priority * task value

        # Poor Revisit Time Scenario
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 5.0,            # poor revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }

        task_1_val = 0.6*0.5*0.5*1.0+0.3*1.0*0.5*1.0+0.1*0.75*0.5*1.0
        task_2_val = 0.6*1.0*0.5*1.0+0.3*0.5*0.5*1.0+0.1*0.50*0.5*1.0
        obs_val = self.mission.calc_observation_opportunity_value(obs, perf)
        self.assertAlmostEqual(obs_val, 1.0 * task_1_val + 10.0 * task_2_val) # priority * task value

        # Worst Revisit Time Scenario
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 100.0,          # worst revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }

        task_1_val = 0.0
        task_2_val = 0.0
        obs_val = self.mission.calc_observation_opportunity_value(obs, perf)
        self.assertAlmostEqual(obs_val, 1.0 * task_1_val + 10.0 * task_2_val) # priority * task value

        # High Cost Scenario (should not affect value, only utility)
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 1e6,                # very long duration; high cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }

        task_1_val = 0.6*0.5*1.0*1.0+0.3*1.0*1.0*1.0+0.1*0.75*1.0*1.0
        task_2_val = 0.6*1.0*1.0*1.0+0.3*0.5*1.0*1.0+0.1*0.50*1.0*1.0
        obs_val = self.mission.calc_observation_opportunity_value(obs, perf)
        self.assertAlmostEqual(obs_val, 1.0 * task_1_val + 10.0 * task_2_val) # priority * task value

        # Outside Availability Scenario
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 101.0               # unavailable observation time
        }

        task_1_val = 0.0
        task_2_val = 0.0
        obs_val = self.mission.calc_observation_opportunity_value(obs, perf)
        self.assertAlmostEqual(obs_val, 1.0 * task_1_val + 10.0 * task_2_val) # priority * task value

    def test_observation_opportunity_utility(self):
        # create observation opportunities
        task_1 = DefaultMissionTask(self.parameter, self.target, 100, 1.0, self.default_objective)
        task_2 = EventObservationTask(self.parameter, self.target, Interval(0, 100), 10.0, self.event, self.event_objective)
        obs = ObservationOpportunity([task_1, task_2], self.instrument, Interval(0, 100), 0.0, Interval(-90,90))
        norm_param = 1.0  # example normalizing parameter
        
        # Error Case Scenario: wrong observation type
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }
        self.assertRaises(AssertionError, self.mission.calc_observation_opportunity_utility, "invalid_observation_type", perf, norm_param)
        
        # Error Case Scenario: wrong measurement type
        self.assertRaises(AssertionError, self.mission.calc_observation_opportunity_utility, obs, "invalid_measurement_type", norm_param)

        # Error Case Scenario: invalid normalizing parameter value
        self.assertRaises(AssertionError, self.mission.calc_observation_opportunity_utility, obs, perf, -123) # invalid value
        self.assertRaises(AssertionError, self.mission.calc_observation_opportunity_utility, obs, perf, "invalid_norm") # invalid type

        # Best Case Scenario
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }

        task_1_val = 0.6*0.5*1.0+0.3*1.0*1.0+0.1*0.75*1.0
        task_2_val = 0.6*1.0*1.0+0.3*0.5*1.0+0.1*0.50*1.0
        obs_util = self.mission.calc_observation_opportunity_utility(obs, perf, norm_param)
        self.assertAlmostEqual(obs_util, 1.0 * task_1_val + 10.0 * task_2_val) # priority * task value

        # Non-zero Cost Scenario
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 1.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }

        task_1_val = 0.6*0.5*1.0+0.3*1.0*1.0+0.1*0.75*1.0
        task_2_val = 0.6*1.0*1.0+0.3*0.5*1.0+0.1*0.50*1.0
        obs_util = self.mission.calc_observation_opportunity_utility(obs, perf, norm_param)
        self.assertAlmostEqual(obs_util, 1.0 * task_1_val + 10.0 * task_2_val - 1.0) # priority * task value - cost

        # Wront Target Scenario
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (85.0522, -40.50, 0, 2),    # different target than requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }

        task_1_val = 0.0
        task_2_val = 0.0
        obs_val = self.mission.calc_observation_opportunity_utility(obs, perf, norm_param)
        self.assertAlmostEqual(obs_val, 1.0 * task_1_val + 10.0 * task_2_val) # priority * task value

        # Wront Target Scenario with cost
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (85.0522, -40.50, 0, 2),    # different target than requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 1.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }

        task_1_val = 0.0
        task_2_val = 0.0
        obs_val = self.mission.calc_observation_opportunity_utility(obs, perf, norm_param)
        self.assertAlmostEqual(obs_val, 1.0 * task_1_val + 10.0 * task_2_val - 1.0) # priority * task value

        # Poor Revisit Time Scenario
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 5.0,            # poor revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }

        task_1_val = 0.6*0.5*0.5*1.0+0.3*1.0*0.5*1.0+0.1*0.75*0.5*1.0
        task_2_val = 0.6*1.0*0.5*1.0+0.3*0.5*0.5*1.0+0.1*0.50*0.5*1.0
        obs_val = self.mission.calc_observation_opportunity_utility(obs, perf, norm_param)
        self.assertAlmostEqual(obs_val, 1.0 * task_1_val + 10.0 * task_2_val) # priority * task value

        # Poor Revisit Time Scenario with cost
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 5.0,            # poor revisit time
            TemporalRequirementAttributes.DURATION.value: 1.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }

        task_1_val = 0.6*0.5*0.5*1.0+0.3*1.0*0.5*1.0+0.1*0.75*0.5*1.0
        task_2_val = 0.6*1.0*0.5*1.0+0.3*0.5*0.5*1.0+0.1*0.50*0.5*1.0
        obs_val = self.mission.calc_observation_opportunity_utility(obs, perf, norm_param)
        self.assertAlmostEqual(obs_val, 1.0 * task_1_val + 10.0 * task_2_val -1.0) # priority * task value - cost

        # Worst Revisit Time Scenario
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 100.0,          # worst revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }

        task_1_val = 0.0
        task_2_val = 0.0
        obs_val = self.mission.calc_observation_opportunity_utility(obs, perf, norm_param)
        self.assertAlmostEqual(obs_val, 1.0 * task_1_val + 10.0 * task_2_val) # priority * task value

        # Worst Revisit Time Scenario with cost
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 100.0,          # worst revisit time
            TemporalRequirementAttributes.DURATION.value: 1.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }

        task_1_val = 0.0
        task_2_val = 0.0
        obs_val = self.mission.calc_observation_opportunity_utility(obs, perf, norm_param)
        self.assertAlmostEqual(obs_val, 1.0 * task_1_val + 10.0 * task_2_val - 1.0) # priority * task value

        # High Cost Scenario
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 1e6,                # very long duration; high cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 0.0                 # best observation time from mission start
        }

        task_1_val = 0.6*0.5*1.0*1.0+0.3*1.0*1.0*1.0+0.1*0.75*1.0*1.0
        task_2_val = 0.6*1.0*1.0*1.0+0.3*0.5*1.0*1.0+0.1*0.50*1.0*1.0
        obs_val = self.mission.calc_observation_opportunity_utility(obs, perf, norm_param)
        self.assertAlmostEqual(obs_val, 1.0 * task_1_val + 10.0 * task_2_val - 1e6) # priority * task value

        # Outside Availability Scenario
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 0.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 101.0               # unavailable observation time
        }

        task_1_val = 0.0
        task_2_val = 0.0
        obs_val = self.mission.calc_observation_opportunity_utility(obs, perf, norm_param)
        self.assertAlmostEqual(obs_val, 1.0 * task_1_val + 10.0 * task_2_val) # priority * task value

        # Outside Availability Scenario with costs
        perf = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,              # best revisit time
            TemporalRequirementAttributes.DURATION.value: 1.0,                # zero duration; zero cost
            TemporalRequirementAttributes.RESPONSE_TIME.value: 0.0,       # best observation time from start of task availability
            TemporalRequirementAttributes.OBS_TIME.value: 101.0               # unavailable observation time
        }

        task_1_val = 0.0
        task_2_val = 0.0
        obs_val = self.mission.calc_observation_opportunity_utility(obs, perf, norm_param)
        self.assertAlmostEqual(obs_val, 1.0 * task_1_val + 10.0 * task_2_val - 1.0) # priority * task value - cost

    def test_representation(self):
        # test string representation
        mission_str = str(repr(self.mission))
        expected_str = f"Mission(name='testmission', n_objectives={len(self.mission.objectives)})"
        self.assertEqual(mission_str, expected_str)

    def test_iterator(self):
        # test iterator over mission objectives
        objs = list(self.mission)
        self.assertEqual(len(objs), len(self.mission.objectives))        
        for obj in self.mission:
            self.assertIsInstance(obj, MissionObjective)

    def test_to_dict(self):
        # test conversion to dictionary
        mission_dict = self.mission.to_dict()
        self.assertIsInstance(mission_dict, dict)
        self.assertEqual(mission_dict['name'], self.mission.name)
        self.assertEqual(len(mission_dict['objectives']), len(self.mission.objectives))
        for obj_dict, obj in zip(mission_dict['objectives'], self.mission.objectives):
            # compare objectives
            for key,val in obj.to_dict().items():
                self.assertEqual(obj_dict[key], val)

            # compare weights
            weight = self.mission.objectives[obj]
            self.assertAlmostEqual(obj_dict['weight'], weight)

    def test_from_dict(self):
        # test creation from dictionary
        mission_dict = self.mission.to_dict()
        mission_from_dict = Mission.from_dict(mission_dict)
        self.assertIsInstance(mission_from_dict, Mission)
        self.assertEqual(mission_from_dict.name, self.mission.name)
        self.assertEqual(len(mission_from_dict.objectives), len(self.mission.objectives))
        for obj_original, obj_from_dict in zip(self.mission.objectives, mission_from_dict.objectives):
            self.assertEqual(obj_original, obj_from_dict)
            self.assertAlmostEqual(self.mission.objectives[obj_original], mission_from_dict.objectives[obj_from_dict])

    def test_copy(self):
        # test deep copy of mission
        mission_copy = self.mission.copy()
        self.assertIsNot(mission_copy, self.mission)
        self.assertEqual(mission_copy.name, self.mission.name)
        self.assertEqual(len(mission_copy.objectives), len(self.mission.objectives))
        for original_obj, copied_obj in zip(self.mission.objectives, mission_copy.objectives):
            self.assertIsNot(original_obj, copied_obj)
            self.assertEqual(original_obj, copied_obj)



if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Mission Definitions Test')
    
    # run tests
    unittest.main()