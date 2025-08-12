import unittest

from chess3d.mission.requirements import *
from chess3d.mission.objectives import *
from chess3d.utils import print_welcome

class TestDefaultObjectives(unittest.TestCase):
    def setUp(self):
        self.default_objective_dict = {
                "objective_type" : "default_mission",
                "parameter": "Chlorophyll-A",
                "weight": 1,
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
    
    def test_default_mission_objective_from_dict(self):
        objective = MissionObjective.from_dict(self.default_objective_dict)
        self.assertEqual(objective.parameter, "Chlorophyll-A")
        self.assertEqual(objective.weight, 1)
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
        self.assertEqual(objective_copy.weight, 1)
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
        self.assertEqual(objective_dict['weight'], 1)
        self.assertEqual(len(objective_dict['requirements']), 5)
        self.assertEqual(objective_dict['requirements'][0]['attribute'], "horizontal_spatial_resolution")
        self.assertEqual(objective_dict['requirements'][1]['attribute'], "spectral_resolution")
        self.assertEqual(objective_dict['requirements'][2]['attribute'], "instrument")
        self.assertEqual(objective_dict['requirements'][3]['attribute'], "revisit_time")
        self.assertEqual(objective_dict['requirements'][4]['location_type'], "grid")
        self.assertEqual(objective_dict['requirements'][4]['grid_name'], "grid_1")
        self.assertEqual(objective_dict['requirements'][4]['grid_index'], 0)
        self.assertEqual(objective_dict['requirements'][4]['grid_size'], 10)
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
    def test_wrong_measurement_resolution_performance(self):
        objective = MissionObjective.from_dict(self.default_objective_dict)
        measurement = {
            "horizontal_spatial_resolution": 200, # deficient spatial resolution
            "spectral_resolution": "Hyperspectral",
            "instrument": "VNIR",
            "revisit_time": 7200,
            "location": (34.0522, -118.2437, 0, 0)
        }
        performance = objective.eval_measurement_performance(measurement)
        self.assertGreater(performance, 0)
        self.assertLessEqual(performance, 0.1)  # Should be low due to poor spatial resolution
    def test_wrong_measurement_spectral_performance(self):
        objective = MissionObjective.from_dict(self.default_objective_dict)
        measurement = {
            "horizontal_spatial_resolution": 20, 
            "spectral_resolution": "hypospectral", # deficient spectral resolution
            "instrument": "VNIR",
            "revisit_time": 7200,
            "location": (34.0522, -118.2437, 0, 0)
        }
        performance = objective.eval_measurement_performance(measurement)
        self.assertGreaterEqual(performance, 0) # Should be low due to poor spectral resolution
    def test_wrong_measurement_instrument_performance(self):
        objective = MissionObjective.from_dict(self.default_objective_dict)
        measurement = {
            "horizontal_spatial_resolution": 20, 
            "spectral_resolution": "Hyperspectral", 
            "instrument": "SAR", # deficient instrument
            "revisit_time": 7200,
            "location": (34.0522, -118.2437, 0, 0)
        }
        performance = objective.eval_measurement_performance(measurement)
        self.assertGreaterEqual(performance, 0) # Should be low due to poor instrument
    def test_wrong_measurement_instrument_performance(self):
        objective = MissionObjective.from_dict(self.default_objective_dict)
        measurement = {
            "horizontal_spatial_resolution": 20,
            "spectral_resolution": "Hyperspectral",
            "instrument": "VNIR",
            "revisit_time": 90000, # deficient revisit time
            "location": (34.0522, -118.2437, 0, 0)
        }
        performance = objective.eval_measurement_performance(measurement)
        self.assertGreaterEqual(performance, 0) # Should be low due to deficient revisit time
    def test_wrong_measurement_location_performance(self):
        objective = MissionObjective.from_dict(self.default_objective_dict)
        measurement = {
            "horizontal_spatial_resolution": 20,
            "spectral_resolution": "Hyperspectral",
            "instrument": "VNIR",
            "revisit_time": 7200,
            "location": (0, 0, 0, 20) # deficient measurement location
        }
        performance = objective.eval_measurement_performance(measurement)
        self.assertLessEqual(performance, 0.0) # Should be low due to deficient measurement location

class TestEventObjectives(unittest.TestCase):
    def setUp(self):
        self.default_objective_dict = {
                "objective_type" : "default_mission",
                "parameter": "Chlorophyll-A",
                "weight": 1,
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
                "weight": 10,
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
    
    def test_event_driven_objective_from_dict(self):
        objective = EventDrivenObjective.from_dict(self.event_objective_dict)
        self.assertEqual(objective.event_type, "algal bloom")
        self.assertEqual(objective.weight, 10)
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
        self.assertEqual(objective_copy.weight, 10)
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
        self.assertEqual(objective_dict['weight'], 10)
        self.assertEqual(len(objective_dict['requirements']), 5)
        self.assertEqual(objective_dict['requirements'][0]['attribute'], "horizontal_spatial_resolution")
        self.assertEqual(objective_dict['requirements'][1]['attribute'], "spectral_resolution")
        self.assertEqual(objective_dict['requirements'][2]['attribute'], "instrument")
        self.assertEqual(objective_dict['requirements'][3]['attribute'], "revisit_time")
        self.assertEqual(objective_dict['requirements'][4]['attribute'], "location")
    def test_event_driven_objective_from_default(self):
        event = GeophysicalEvent('Algal Bloom', 
                                 [
                                    (0.0,0.0,0,0),
                                    (1.0,1.0,0,1)
                                  ], 
                                 1.0, 
                                 0.5, 
                                 1.0
                                )
        default_objective = MissionObjective.from_dict(self.default_objective_dict)
        self.assertRaises(AssertionError, EventDrivenObjective.from_default_objective, event.to_dict(), default_objective, [], 10) # invalid event type
        self.assertRaises(AssertionError, EventDrivenObjective.from_default_objective, event, default_objective.to_dict(), [], 10) # invalid objective type
        self.assertRaises(AssertionError, EventDrivenObjective.from_default_objective, event, default_objective, 123, 10) # invalid synergistic parameter list type
        self.assertRaises(AssertionError, EventDrivenObjective.from_default_objective, event, default_objective, [], '10') # invalid objective weight type

        objective = EventDrivenObjective.from_default_objective(event, default_objective, [], 10)
        self.assertEqual(objective.event_type, "algal bloom")
        self.assertEqual(objective.weight, 10)
        self.assertEqual(len(objective.requirements), 5)
        self.assertIsInstance(objective.requirements[0], DiscreteRequirement)
        self.assertIsInstance(objective.requirements[1], CategoricalRequirement)
        self.assertIsInstance(objective.requirements[2], CapabilityRequirement)
        self.assertIsInstance(objective.requirements[3], TemporalRequirement)
        self.assertIsInstance(objective.requirements[4], GridTargetSpatialRequirement)
        self.assertEqual(objective.requirements[4].grid_name, "grid_1")
        self.assertEqual(objective.requirements[4].grid_index, 0)
        self.assertEqual(objective.requirements[4].grid_size, 10)
        self.assertEqual([], objective.synergistic_parameters)

        objective = EventDrivenObjective.from_default_objective(event, default_objective, [])
        self.assertEqual(objective.weight, 1)

if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Mission Definitions Test')
    
    # run tests
    unittest.main()