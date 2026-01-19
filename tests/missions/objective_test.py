import unittest

from chess3d.mission.attributes import TemporalRequirementAttributes
from chess3d.mission.events import GeophysicalEvent
from chess3d.mission.requirements import *
from chess3d.mission.objectives import DefaultMissionObjective, EventDrivenObjective
from chess3d.utils import print_welcome

class TestDefaultObjectives(unittest.TestCase):
    def setUp(self):
        self.parameter = "Chlorophyll-A"
        self.target_1 = (34.0522, -118.2437, 0, 0)  # Example target: (lat, lon, grid_index, gp_index)
        
        self.req_1_1 = SinglePointSpatialRequirement(target=self.target_1, distance_threshold=10.0)
        self.req_1_2 = IntervalInterpolationRequirement(TemporalRequirementAttributes.REVISIT_TIME.value, [0, 10], [1.0, 0.0])

        self.objective = DefaultMissionObjective(
            parameter=self.parameter,
            requirements=[self.req_1_1, self.req_1_2]
        )

    def test_objective_constructor(self):
        # Verify that the objective is constructed correctly
        self.assertIsInstance(self.objective, DefaultMissionObjective)
        self.assertEqual(self.objective.objective_type, 'default_mission')
        self.assertEqual(self.objective.parameter, self.parameter.lower())
        self.assertEqual(len(self.objective.requirements), 2)
        self.assertIsInstance(self.objective.requirements[SpatialCoverageRequirement.ATTRIBUTE], SinglePointSpatialRequirement)
        self.assertIsInstance(self.objective.requirements[TemporalRequirementAttributes.REVISIT_TIME.value], IntervalInterpolationRequirement)

        # Test invalid preferences
        self.assertRaises(AssertionError, DefaultMissionObjective,
                          parameter=123, # invalid parameter type
                          requirements=[self.req_1_1, self.req_1_2])
        self.assertRaises(ValueError, DefaultMissionObjective,
                          parameter=self.parameter,
                          requirements=[])  # empty requirements list
        self.assertRaises(ValueError, DefaultMissionObjective,
                          parameter=self.parameter,
                          requirements=[self.req_1_1]) # missing temporal requirement
        # self.assertRaises(ValueError, DefaultMissionObjective,
        #                   parameter=self.parameter,
        #                   requirements=[self.req_1_2]) # missing spatial requirement
        self.assertRaises(AssertionError, DefaultMissionObjective,
                          parameter=self.parameter,
                          requirements=[self.req_1_1, self.req_1_2, 'invalid_requirement'])  # invalid requirement type
        self.assertRaises(AssertionError, DefaultMissionObjective,
                          parameter=self.parameter,
                          requirements=[self.req_1_1, self.req_1_2],
                          id=12345)  # invalid id type
        self.assertRaises(ValueError, DefaultMissionObjective,
                          parameter=self.parameter,
                          requirements=[self.req_1_1, self.req_1_2],
                          id="12345")  # invalid id format

    def test_measurement_performance(self):
        perf_1 = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0), # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,                # best revisit time
        }
        perf_2 = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0),  # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 5,             # worse revisit time
        }
        perf_3 = {
            SpatialCoverageRequirement.ATTRIBUTE: (34.0522, -118.2437, 0, 0),  # same target as requirement
            TemporalRequirementAttributes.REVISIT_TIME.value: 10,             # worst revisit time
        }
        perf_4 = {
            SpatialCoverageRequirement.ATTRIBUTE: (35.0522, -129.0, 0, 1),  # different target
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,             # best revisit time
        }
        perf_5 = {
            SpatialCoverageRequirement.ATTRIBUTE: (35.0522, -129.0, 0, 1),  # different target
            # missing revisit time
        }
        perf_6 = {
            SpatialCoverageRequirement.ATTRIBUTE: (35.0522, -129.0, 0, 1),  # different target
            TemporalRequirementAttributes.REVISIT_TIME.value: 0,             # best revisit time
            12345 : 'invalid_key'  # invalid key type
        }
        self.assertAlmostEqual(self.objective.eval_measurement_performance(perf_1), 1.0)
        self.assertAlmostEqual(self.objective.eval_measurement_performance(perf_2), 0.5)
        self.assertAlmostEqual(self.objective.eval_measurement_performance(perf_3), 0.0)
        self.assertAlmostEqual(self.objective.eval_measurement_performance(perf_4), 0.0)
        self.assertRaises(AssertionError, self.objective.eval_measurement_performance, 'perf')  # invalid measurement type
        self.assertRaises(AssertionError, self.objective.eval_measurement_performance, perf_5)  # missing requirement attribute
        self.assertRaises(AssertionError, self.objective.eval_measurement_performance, perf_6)  # invalid measurement keys type

    def test_representation(self):
        # Test string representation
        expected_str = f"DefaultMissionObjective(parameter={self.parameter.lower()}, n_reqs=2, id={self.objective.id.split('-')[0]})"
        self.assertEqual(repr(self.objective), expected_str)

    def test_from_dict(self):
        obj_dict = {
            "objective_type": "default_mission",
            "parameter": self.parameter.lower(),
            "requirements": [
                {
                    "req_type": RequirementTypes.SPATIAL.value,
                    "attribute": SpatialCoverageRequirement.ATTRIBUTE,
                    "strategy" : SpatialPreferenceStrategies.SINGLE_POINT.value,
                    "target": self.target_1,
                    "distance_threshold": 10.0,
                },
                {
                    "req_type": RequirementTypes.PERFORMANCE.value,
                    "attribute": TemporalRequirementAttributes.REVISIT_TIME.value,
                    "strategy": PerformancePreferenceStrategies.INTERVAL_INTERP.value,   
                    "thresholds": [0, 10],
                    "scores": [1.0, 0.0]
                }
            ],
            "id": self.objective.id
        }
        new_objective = DefaultMissionObjective.from_dict(obj_dict)
        self.assertIsInstance(new_objective, DefaultMissionObjective)
        self.assertEqual(new_objective.objective_type, 'default_mission')
        self.assertEqual(new_objective.parameter, self.parameter.lower())
        self.assertEqual(len(new_objective.requirements), 2)
        self.assertIsInstance(new_objective.requirements[SpatialCoverageRequirement.ATTRIBUTE], SinglePointSpatialRequirement)
        self.assertIsInstance(new_objective.requirements[TemporalRequirementAttributes.REVISIT_TIME.value], IntervalInterpolationRequirement)
        self.assertEqual(new_objective.id, self.objective.id)

        # obj_dict = {
        #     "objective_type": "default_mission",
        #     "parameter": self.parameter.lower(),
        #     "requirements": [
        #         {
        #             "req_type": RequirementTypes.PERFORMANCE.value,
        #             "attribute": TemporalRequirementAttributes.REVISIT_TIME.value,
        #             "strategy": PerformancePreferenceStrategies.INTERVAL_INTERP.value,   
        #             "thresholds": [0, 10],
        #             "scores": [1.0, 0.0]
        #         }
        #         # missing spatial coverage requirement
        #     ],
        #     "id": self.objective.id
        # }
        # self.assertRaises(ValueError, DefaultMissionObjective.from_dict, obj_dict)

        obj_dict = {
            "objective_type": "default_mission",
            "parameter": self.parameter.lower(),
            "requirements": [], # no requirements
            "id": self.objective.id
        }
        self.assertRaises(ValueError, DefaultMissionObjective.from_dict, obj_dict)

        obj_dict = {
            "objective_type": "default_mission",
            "parameter": self.parameter.lower(),
            "requirements": "invalid_reqs", # invalid requirements
            "id": self.objective.id
        }
        self.assertRaises(ValueError, DefaultMissionObjective.from_dict, obj_dict)

        obj_dict = {
            # missing objective type
            "parameter": self.parameter.lower(),
            "requirements": [
                {
                    "req_type": RequirementTypes.SPATIAL.value,
                    "attribute": SpatialCoverageRequirement.ATTRIBUTE,
                    "strategy" : SpatialPreferenceStrategies.SINGLE_POINT.value,
                    "target": self.target_1,
                    "distance_threshold": 10.0,
                },
                {
                    "req_type": RequirementTypes.PERFORMANCE.value,
                    "attribute": TemporalRequirementAttributes.REVISIT_TIME.value,
                    "strategy": PerformancePreferenceStrategies.INTERVAL_INTERP.value,   
                    "thresholds": [0, 10],
                    "scores": [1.0, 0.0]
                }
            ],
            "id": self.objective.id
        }
        self.assertRaises(AssertionError, DefaultMissionObjective.from_dict, obj_dict)
    
        obj_dict = {
            "objective_type": "default_mission",
            # missing parameter
            "requirements": [
                {
                    "req_type": RequirementTypes.SPATIAL.value,
                    "attribute": SpatialCoverageRequirement.ATTRIBUTE,
                    "strategy" : SpatialPreferenceStrategies.SINGLE_POINT.value,
                    "target": self.target_1,
                    "distance_threshold": 10.0,
                },
                {
                    "req_type": RequirementTypes.PERFORMANCE.value,
                    "attribute": TemporalRequirementAttributes.REVISIT_TIME.value,
                    "strategy": PerformancePreferenceStrategies.INTERVAL_INTERP.value,   
                    "thresholds": [0, 10],
                    "scores": [1.0, 0.0]
                }
            ],
            "id": self.objective.id
        }
        self.assertRaises(AssertionError, DefaultMissionObjective.from_dict, obj_dict)

        obj_dict = {
            "objective_type": "default_mission",
            "parameter": self.parameter.lower(),
            # missing requirements
            "id": self.objective.id
        }
        self.assertRaises(AssertionError, DefaultMissionObjective.from_dict, obj_dict)


    def test_to_dict(self):
        obj_dict = self.objective.to_dict()
        self.assertIsInstance(obj_dict, dict)
        self.assertEqual(obj_dict["objective_type"], 'default_mission')
        self.assertEqual(obj_dict["parameter"], self.parameter.lower())
        self.assertEqual(len(obj_dict["requirements"]), 2)
        self.assertEqual(obj_dict["id"], self.objective.id)

    def test_copy(self):
        obj_copy = self.objective.copy()
        self.assertIsInstance(obj_copy, DefaultMissionObjective)
        self.assertEqual(obj_copy.objective_type, self.objective.objective_type)
        self.assertEqual(obj_copy.parameter, self.objective.parameter)
        self.assertEqual(len(obj_copy.requirements), len(self.objective.requirements))
        for attr in self.objective.requirements:
            self.assertEqual(obj_copy.requirements[attr].to_dict(), self.objective.requirements[attr].to_dict())  # Ensure equal content
            self.assertIsNot(obj_copy.requirements[attr], self.objective.requirements[attr])  # Ensure deep copy
        self.assertEqual(obj_copy.id, self.objective.id)
        self.assertIsNot(obj_copy, self.objective)  # Ensure different instances

    def test_iter(self):
        # Test iteration over requirements
        reqs = list(self.objective)
        self.assertEqual(len(reqs), 2)
        self.assertIn(self.req_1_1, reqs)
        self.assertIn(self.req_1_2, reqs)

class TestEventDrivenObjectives(unittest.TestCase):
    def setUp(self):
        self.event_type = 'Algal Bloom'
        self.parameter = "Chlorophyll-A"
        self.target_1 = (34.0522, -118.2437, 0, 0)  # Example target: (lat, lon, grid_index, gp_index)
        
        self.req_1_1 = SinglePointSpatialRequirement(target=self.target_1, distance_threshold=10.0)
        self.req_1_2 = IntervalInterpolationRequirement(TemporalRequirementAttributes.REVISIT_TIME.value, [0, 10], [1.0, 0.0])

        self.event_objective = EventDrivenObjective(
            event_type=self.event_type,
            parameter=self.parameter,
            requirements=[self.req_1_1, self.req_1_2]
        )
        self.default_objective = DefaultMissionObjective(
            parameter=self.parameter,
            requirements=[self.req_1_1, self.req_1_2]
        )
        self.event = GeophysicalEvent(self.event_type, self.target_1, 0.0, 1000, 1.0)

    def test_objective_constructor(self):
        # Verify that the objective is constructed correctly
        self.assertIsInstance(self.event_objective, EventDrivenObjective)
        self.assertEqual(self.event_objective.objective_type, 'event_driven')
        self.assertEqual(self.event_objective.parameter, self.parameter.lower())
        self.assertEqual(len(self.event_objective.requirements), 2)
        self.assertIsInstance(self.event_objective.requirements[SpatialCoverageRequirement.ATTRIBUTE], SinglePointSpatialRequirement)
        self.assertIsInstance(self.event_objective.requirements[TemporalRequirementAttributes.REVISIT_TIME.value], IntervalInterpolationRequirement)

    def test_representation(self):
        # Test string representation
        expected_str = f"EventDrivenObjective(parameter={self.parameter.lower()}, event_type={self.event_type.lower()}, id={self.event_objective.id.split('-')[0]})"
        self.assertEqual(repr(self.event_objective), expected_str)

    def test_from_dict(self):
        obj_dict = {
            "objective_type": "event_driven",
            "event_type": self.event_type.lower(),
            "parameter": self.parameter.lower(),
            "requirements": [
                {
                    "req_type": RequirementTypes.SPATIAL.value,
                    "attribute": SpatialCoverageRequirement.ATTRIBUTE,
                    "strategy" : SpatialPreferenceStrategies.SINGLE_POINT.value,
                    "target": self.target_1,
                    "distance_threshold": 10.0,
                },
                {
                    "req_type": RequirementTypes.PERFORMANCE.value,
                    "attribute": TemporalRequirementAttributes.REVISIT_TIME.value,
                    "strategy": PerformancePreferenceStrategies.INTERVAL_INTERP.value,   
                    "thresholds": [0, 10],
                    "scores": [1.0, 0.0]
                }
            ],
            "id": self.event_objective.id
        }
        new_objective = EventDrivenObjective.from_dict(obj_dict)
        self.assertIsInstance(new_objective, EventDrivenObjective)
        self.assertEqual(new_objective.objective_type, 'event_driven')
        self.assertEqual(new_objective.parameter, self.parameter.lower())
        self.assertEqual(len(new_objective.requirements), 2)
        self.assertIsInstance(new_objective.requirements[SpatialCoverageRequirement.ATTRIBUTE], SinglePointSpatialRequirement)
        self.assertIsInstance(new_objective.requirements[TemporalRequirementAttributes.REVISIT_TIME.value], IntervalInterpolationRequirement)
        self.assertEqual(new_objective.id, self.event_objective.id)

        obj_dict = {
            "objective_type": "event_driven",
            "event_type": self.event_type.lower(),
            "parameter": self.parameter.lower(),
            "requirements": [
                {
                    "req_type": RequirementTypes.SPATIAL.value,
                    "attribute": SpatialCoverageRequirement.ATTRIBUTE,
                    "strategy" : SpatialPreferenceStrategies.SINGLE_POINT.value,
                    "target": self.target_1,
                    "distance_threshold": 10.0,
                },
                # missing temporal requirement
            ],
            "id": self.event_objective.id
        }
        self.assertRaises(ValueError, EventDrivenObjective.from_dict, obj_dict)

        obj_dict = {
            "objective_type": "event_driven",
            "event_type": self.event_type.lower(),
            "parameter": self.parameter.lower(),
            "requirements": [], # no requirements
            "id": self.event_objective.id
        }
        self.assertRaises(ValueError, EventDrivenObjective.from_dict, obj_dict)

        obj_dict = {
            # "objective_type": "event_driven", # missing objective type
            "event_type": self.event_type.lower(),
            "parameter": self.parameter.lower(),
            "requirements": [
                {
                    "req_type": RequirementTypes.SPATIAL.value,
                    "attribute": SpatialCoverageRequirement.ATTRIBUTE,
                    "strategy" : SpatialPreferenceStrategies.SINGLE_POINT.value,
                    "target": self.target_1,
                    "distance_threshold": 10.0,
                },
                {
                    "req_type": RequirementTypes.PERFORMANCE.value,
                    "attribute": TemporalRequirementAttributes.REVISIT_TIME.value,
                    "strategy": PerformancePreferenceStrategies.INTERVAL_INTERP.value,   
                    "thresholds": [0, 10],
                    "scores": [1.0, 0.0]
                }
            ],
            "id": self.event_objective.id
        }
        self.assertRaises(AssertionError, EventDrivenObjective.from_dict, obj_dict)

        obj_dict = {
            "objective_type": "event_driven",
            # "event_type": self.event_type.lower(), # missing event type
            "parameter": self.parameter.lower(),
            "requirements": [
                {
                    "req_type": RequirementTypes.SPATIAL.value,
                    "attribute": SpatialCoverageRequirement.ATTRIBUTE,
                    "strategy" : SpatialPreferenceStrategies.SINGLE_POINT.value,
                    "target": self.target_1,
                    "distance_threshold": 10.0,
                },
                {
                    "req_type": RequirementTypes.PERFORMANCE.value,
                    "attribute": TemporalRequirementAttributes.REVISIT_TIME.value,
                    "strategy": PerformancePreferenceStrategies.INTERVAL_INTERP.value,   
                    "thresholds": [0, 10],
                    "scores": [1.0, 0.0]
                }
            ],
            "id": self.event_objective.id
        }
        self.assertRaises(AssertionError, EventDrivenObjective.from_dict, obj_dict)

        obj_dict = {
            "objective_type": "event_driven",
            "event_type": self.event_type.lower(),
            # "parameter": self.parameter.lower(), # missing parameter
            "requirements": [
                {
                    "req_type": RequirementTypes.SPATIAL.value,
                    "attribute": SpatialCoverageRequirement.ATTRIBUTE,
                    "strategy" : SpatialPreferenceStrategies.SINGLE_POINT.value,
                    "target": self.target_1,
                    "distance_threshold": 10.0,
                },
                {
                    "req_type": RequirementTypes.PERFORMANCE.value,
                    "attribute": TemporalRequirementAttributes.REVISIT_TIME.value,
                    "strategy": PerformancePreferenceStrategies.INTERVAL_INTERP.value,   
                    "thresholds": [0, 10],
                    "scores": [1.0, 0.0]
                }
            ],
            "id": self.event_objective.id
        }
        self.assertRaises(AssertionError, EventDrivenObjective.from_dict, obj_dict)

        obj_dict = {
            "objective_type": "event_driven",
            "event_type": self.event_type.lower(),
            "parameter": self.parameter.lower(),
            # missing requirements
            "id": self.event_objective.id
        }
        self.assertRaises(AssertionError, EventDrivenObjective.from_dict, obj_dict)

    def test_from_default_objective(self):
        event_obj_from_default : EventDrivenObjective = EventDrivenObjective.from_default_objective(
            event=self.event,
            default_objective=self.default_objective,
        )
        self.assertIsInstance(event_obj_from_default, EventDrivenObjective)
        self.assertEqual(event_obj_from_default.objective_type, 'event_driven')
        self.assertEqual(event_obj_from_default.parameter, self.default_objective.parameter)
        self.assertEqual(len(event_obj_from_default.requirements), len(self.default_objective.requirements))
        for attr in self.default_objective.requirements:
            self.assertEqual(event_obj_from_default.requirements[attr].to_dict(), self.default_objective.requirements[attr].to_dict())  # Ensure equal content
        self.assertEqual(event_obj_from_default.event_type, self.event_type.lower())
        self.assertIsNot(event_obj_from_default, self.event_objective)  # Ensure different instances
        self.assertEqual(event_obj_from_default.event_type, self.event.event_type.lower())

        self.assertRaises(AssertionError, EventDrivenObjective.from_default_objective,
                          event="invalid_event",  # invalid event type
                          default_objective=self.default_objective)
        self.assertRaises(AssertionError, EventDrivenObjective.from_default_objective,
                          event=self.event,
                          default_objective="invalid_objective")  # invalid default objective type

if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Mission Objective Definitions Test')
    
    # run tests
    unittest.main()