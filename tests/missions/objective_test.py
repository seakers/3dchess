import unittest

from chess3d.mission.requirements import *
from chess3d.mission.objectives import *
from chess3d.utils import print_welcome

class TestObjectives(unittest.TestCase):
    def setUp(self):
        self.default_objective_dict = {
                "objective_type" : "default_mission",
                "parameter": "Chlorophyll-A",
                "priority": 1,
                "requirements" : [
                    {
                        "attribute": "horizontal_spatial_resolution",
                        "requirement_type": "discrete",
                        "thresholds": [10, 30, 100],
                        "scores": [1.0, 0.7, 0.1]
                    },
                    {
                        "attribute": "spectral_resolution",
                        "requirement_type": "categorical",
                        "thresholds": ["Hyperspectral", "Multispectral"],
                        "scores": [1, 0.5]
                    },
                    {
                        "attribute": "instrument",
                        "requirement_type": "capability",
                        "valid_values": ["VNIR", "TIR"]
                    },
                    {
                        "attribute": "revisit_time",
                        "requirement_type": "temporal",
                        "thresholds": [3600, 14400, 86400],
                        "scores": [1.0, 0.5, 0.0]
                    },
                    {
                        "requirement_type": "spatial",
                        "location_type": "grid",
                        "grid_name" : "grid_1",
                        "grid_index": 0,
                        "grid_size": 10
                    }
                ]
            }
        self.event_objective_dict = {
                "objective_type" : "event_driven",
                "event_type": "algal bloom",
                "parameter": "Chlorophyll-A",
                "priority": 10,
                "requirements" : [
                    {
                        "attribute": "horizontal_spatial_resolution",
                        "requirement_type": "discrete",
                        "thresholds": [10, 30, 100],
                        "scores": [1.0, 0.7, 0.1]
                    },
                    {
                        "attribute": "spectral_resolution",
                        "requirement_type": "categorical",
                        "thresholds": ["Hyperspectral", "Multispectral"],
                        "scores": [1, 0.5]
                    },
                    {
                        "attribute": "instrument",
                        "requirement_type": "capability",
                        "valid_values": ["VNIR", "TIR"]
                    },
                    {
                        "attribute": "revisit_time",
                        "requirement_type": "temporal",
                        "thresholds": [3600, 14400, 86400],
                        "scores": [1.0, 0.5, 0.0]
                    },
                    {
                        "requirement_type": "spatial",
                        "location_type": "grid",
                        "grid_name" : "grid_1",
                        "grid_index": 0,
                        "grid_size": 10
                    }
                ],
                "synergistic_parameters" : [
                    "Water temperature",
                    "Water level"
                ]
            }
    
    def test_default_mission_objective_from_dict(self):
        objective = MissionObjective.from_dict(self.default_objective_dict)
        self.assertEqual(objective.parameter, "Chlorophyll-A")
        self.assertEqual(objective.priority, 1)
        self.assertEqual(len(objective.requirements), 5)
        self.assertIsInstance(objective.requirements[0], DiscreteRequirement)
        self.assertIsInstance(objective.requirements[1], CategoricalRequirement)
        self.assertIsInstance(objective.requirements[2], CapabilityRequirement)
        self.assertIsInstance(objective.requirements[3], TemporalRequirement)
        self.assertIsInstance(objective.requirements[4], GridTargetSpatialRequirement)
        self.assertEqual(objective.requirements[4].grid_name, "grid_1")
        self.assertEqual(objective.requirements[4].grid_index, 0)
        self.assertEqual(objective.requirements[4].grid_size, 10)
    def test_default_mission_objective_copy(self):
        objective = MissionObjective.from_dict(self.default_objective_dict)
        objective_copy = objective.copy()
        self.assertEqual(objective_copy.parameter, "Chlorophyll-A")
        self.assertEqual(objective_copy.priority, 1)
        self.assertEqual(len(objective_copy.requirements), 5)
        self.assertIsInstance(objective_copy.requirements[0], DiscreteRequirement)
        self.assertIsInstance(objective_copy.requirements[1], CategoricalRequirement)
        self.assertIsInstance(objective_copy.requirements[2], CapabilityRequirement)
        self.assertIsInstance(objective_copy.requirements[3], TemporalRequirement)
        self.assertIsInstance(objective_copy.requirements[4], GridTargetSpatialRequirement)
        self.assertEqual(objective_copy.requirements[4].grid_name, "grid_1")
        self.assertEqual(objective_copy.requirements[4].grid_index, 0)
        self.assertEqual(objective_copy.requirements[4].grid_size, 10)
        self.assertNotEqual(objective, objective_copy)  # Ensure IDs are different
    def test_default_mission_objective_to_dict(self):
        objective = MissionObjective.from_dict(self.default_objective_dict)
        objective_dict = objective.to_dict()
        self.assertEqual(objective_dict['parameter'], "Chlorophyll-A")
        self.assertEqual(objective_dict['priority'], 1)
        self.assertEqual(len(objective_dict['requirements']), 5)
        self.assertEqual(objective_dict['requirements'][0]['attribute'], "horizontal_spatial_resolution")
        self.assertEqual(objective_dict['requirements'][1]['attribute'], "spectral_resolution")
        self.assertEqual(objective_dict['requirements'][2]['attribute'], "instrument")
        self.assertEqual(objective_dict['requirements'][3]['attribute'], "revisit_time")
        self.assertEqual(objective_dict['requirements'][4]['location_type'], "grid")
        self.assertEqual(objective_dict['requirements'][4]['grid_name'], "grid_1")
        self.assertEqual(objective_dict['requirements'][4]['grid_index'], 0)
        self.assertEqual(objective_dict['requirements'][4]['grid_size'], 10)

    def test_event_driven_objective_from_dict(self):
        objective = EventDrivenObjective.from_dict(self.event_objective_dict)
        self.assertEqual(objective.event_type, "algal bloom")
        self.assertEqual(objective.priority, 10)
        self.assertEqual(len(objective.requirements), 5)
        self.assertIsInstance(objective.requirements[0], DiscreteRequirement)
        self.assertIsInstance(objective.requirements[1], CategoricalRequirement)
        self.assertIsInstance(objective.requirements[2], CapabilityRequirement)
        self.assertIsInstance(objective.requirements[3], TemporalRequirement)
        self.assertIsInstance(objective.requirements[4], GridTargetSpatialRequirement)
        self.assertEqual(objective.requirements[4].grid_name, "grid_1")
        self.assertEqual(objective.requirements[4].grid_index, 0)
        self.assertEqual(objective.requirements[4].grid_size, 10)
        self.assertIn("water temperature", objective.synergistic_parameters)
        self.assertIn("water level", objective.synergistic_parameters)
        # self.assertEqual(objective.id, "EventDrivenObjective_algal bloom_0_1_EVENT-algal bloom")
    def test_event_driven_objective_copy(self):
        objective = EventDrivenObjective.from_dict(self.event_objective_dict)
        objective_copy = objective.copy()
        self.assertEqual(objective_copy.event_type, "algal bloom")
        self.assertEqual(objective_copy.priority, 10)
        self.assertEqual(len(objective_copy.requirements), 5)
        self.assertIsInstance(objective_copy.requirements[0], DiscreteRequirement)
        self.assertIsInstance(objective_copy.requirements[1], CategoricalRequirement)
        self.assertIsInstance(objective_copy.requirements[2], CapabilityRequirement)
        self.assertIsInstance(objective_copy.requirements[3], TemporalRequirement)
        self.assertIsInstance(objective_copy.requirements[4], GridTargetSpatialRequirement)
        self.assertEqual(objective_copy.requirements[4].grid_name, "grid_1")
        self.assertEqual(objective_copy.requirements[4].grid_index, 0)
        self.assertEqual(objective_copy.requirements[4].grid_size, 10)
        self.assertIn("water temperature", objective_copy.synergistic_parameters)
        self.assertIn("water level", objective_copy.synergistic_parameters)
        self.assertNotEqual(objective, objective_copy)
    def test_event_driven_objective_to_dict(self):
        objective = EventDrivenObjective.from_dict(self.event_objective_dict)
        objective_dict = objective.to_dict()
        self.assertEqual(objective_dict['event_type'], "algal bloom")
        self.assertEqual(objective_dict['priority'], 10)
        self.assertEqual(len(objective_dict['requirements']), 5)
        self.assertEqual(objective_dict['requirements'][0]['attribute'], "horizontal_spatial_resolution")
        self.assertEqual(objective_dict['requirements'][1]['attribute'], "spectral_resolution")
        self.assertEqual(objective_dict['requirements'][2]['attribute'], "instrument")
        self.assertEqual(objective_dict['requirements'][3]['attribute'], "revisit_time")
        self.assertEqual(objective_dict['requirements'][4]['attribute'], "location")
    def test_measurement_performance(self):
        objective = MissionObjective.from_dict(self.default_objective_dict)
        measurement = {
            "horizontal_spatial_resolution": 20,
            "spectral_resolution": "Hyperspectral",
            "instrument": "VNIR",
            "revisit_time": 7200,
            "location": (34.0522, -118.2437, 0, 0)
        }
        performance = objective.eval_measurement_performance(measurement)
        self.assertGreater(performance, 0)
        self.assertLessEqual(performance, 1)

if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Mission Definitions Test')
    
    # run tests
    unittest.main()