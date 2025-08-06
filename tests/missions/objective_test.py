import unittest

from chess3d.mission.events import GeophysicalEvent
from chess3d.mission.mission import *
from chess3d.mission.requirements import *
from chess3d.mission.objectives import *
from chess3d.utils import print_welcome

class TestObjective(unittest.TestCase):
    
    def test_default_mission_objective_from_dict(self):
        obj_dict = {
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
        objective = MissionObjective.from_dict(obj_dict)
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
        
    def test_event_driven_objective_from_dict(self):
        obj_dict = {
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
        objective = EventDrivenObjective.from_dict(obj_dict)
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

if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Mission Definitions Test')
    
    # run tests
    unittest.main()