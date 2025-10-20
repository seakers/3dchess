import unittest

from chess3d.mission.requirements import *
from chess3d.utils import print_welcome

class TestRequirements(unittest.TestCase):
    # MissionRequirement base class
    def test_from_dict_invalid_type(self):
        bad_dict = {
            "req_type": "unknown_type",
            "attribute": "foo",
            "thresholds": [1, 2, 3],
            "scores": [1.0, 0.5, 0.0]
        }
        self.assertRaises(ValueError, MissionRequirement.from_dict, bad_dict)


class TestCategoricalRequirements(unittest.TestCase):
    # Categorical Requirements
    def test_categorical_requirement_valid(self):
        thresholds = ["low", "medium", "high"]
        scores = [1.0, 0.5, 0.0]
        req = CategoricalRequirement("cloud_cover", thresholds, scores)
        self.assertEqual(req.attribute, "cloud_cover")
        self.assertTrue(all(threshold in thresholds for threshold in req.thresholds))   
        self.assertTrue(all(score in scores for score in req.scores))
        self.assertEqual(req.calc_preference_value("Medium"),0.5)
    def test_categorical_requirement_invalid_score_range(self):
        self.assertRaises(AssertionError, CategoricalRequirement, "cloud_cover", ["Low", "High"], [1.2, 0.0])    
    def test_categorical_requirement_mismatched_lists(self):
        self.assertRaises(AssertionError, CategoricalRequirement, "cloud_cover", ["Low", "Medium"], [1.0])
    def test_categorical_copy(self):
        req = CategoricalRequirement("cloud", ["low", "high"], [1.0, 0.0])
        req_copy = req.copy()
        self.assertEqual(req.to_dict(), req_copy.to_dict())
        self.assertIsNot(req, req_copy)
    def test_categorical_to_from_dict(self):
        original = CategoricalRequirement("cloud", ["low", "medium", "high"], [1.0, 0.5, 0.0])
        as_dict = original.to_dict()
        reconstructed = MissionRequirement.from_dict(as_dict)
        self.assertIsInstance(reconstructed, CategoricalRequirement)
        self.assertEqual(original.to_dict(), reconstructed.to_dict())


class TestDiscreteRequirements(unittest.TestCase):
    # Discrete Value Requirements
    def test_discrete_requirement_valid(self):
        increasing_req = DiscreteRequirement("test", 
                                             thresholds =[0.1, 0.3, 0.5], 
                                             scores     =[1.0, 0.7, 0.4])
        self.assertEqual(increasing_req.attribute, "test")
        self.assertEqual(increasing_req.calc_preference_value(0.3), 0.7)
        self.assertEqual(increasing_req.calc_preference_value(0.2), 0.7)

        decreasing_req = DiscreteRequirement("test", 
                                             thresholds =[0.5, 0.3, 0.1], 
                                             scores     =[1.0, 0.7, 0.4])
        self.assertEqual(decreasing_req.attribute, "test")
        self.assertEqual(decreasing_req.calc_preference_value(0.3), 0.7)
        self.assertEqual(decreasing_req.calc_preference_value(0.2), 0.4)
    def test_discrete_requirement_unsorted_thresholds(self):
        self.assertRaises(AssertionError, DiscreteRequirement, "test",
                          thresholds=[0.3, 0.1, 0.5], scores=[1.0, 0.7, 0.4])
    def test_discrete_requirement_out_of_bounds_score(self):
        self.assertRaises(AssertionError, DiscreteRequirement, "test", 
                          thresholds=[0.1, 0.2], scores=[1.1, 0.9])   
    def test_discrete_copy(self):
        req = DiscreteRequirement("ndvi", [0.1, 0.3], [1.0, 0.5])
        req_copy = req.copy()
        self.assertEqual(req.to_dict(), req_copy.to_dict())
        self.assertIsNot(req, req_copy)
    def test_discrete_to_from_dict(self):
        original = DiscreteRequirement("ndvi", [0.1, 0.2], [1.0, 0.6])
        as_dict = original.to_dict()
        reconstructed = MissionRequirement.from_dict(as_dict)
        self.assertIsInstance(reconstructed, DiscreteRequirement)
        self.assertEqual(original.to_dict(), reconstructed.to_dict())


class TestContinousRequirements(unittest.TestCase):
    # Continuous Value Requirements
    def test_continuous_requirement_valid(self):
        req = ContinuousRequirement("temp", [10, 20, 30], [1.0, 0.5, 0.0])
        self.assertAlmostEqual(req.calc_preference_value(15), 0.75)
    def test_continuous_requirement_out_of_bounds(self):
        req = ContinuousRequirement("temp", [10, 20, 30], [1.0, 0.5, 0.0])
        self.assertAlmostEqual(req.calc_preference_value(35), 0.0)
    def test_continuous_copy(self):
        req = ContinuousRequirement("temp", [10, 20], [1.0, 0.0])
        req_copy = req.copy()
        self.assertEqual(req.to_dict(), req_copy.to_dict())
        self.assertIsNot(req, req_copy)
    def test_continuous_to_from_dict(self):
        original = ContinuousRequirement("temp", [5.0, 25.0], [1.0, 0.2])
        as_dict = original.to_dict()
        reconstructed = MissionRequirement.from_dict(as_dict)
        self.assertIsInstance(reconstructed, ContinuousRequirement)
        self.assertEqual(original.to_dict(), reconstructed.to_dict())

# Temporal Requirements
class TestTemporalRequirements(unittest.TestCase):
    def test_availability_requirement(self):
        req = AvailabilityRequirement(0,10)
        self.assertTrue(isinstance(req, AvailabilityRequirement))
        self.assertEqual(req.availability.left, 0)
        self.assertEqual(req.availability.right, 10)
    def test_availability_requirement_evaluation(self):
        req = AvailabilityRequirement(0, 10)
        self.assertEqual(req.calc_preference_value(0), 1.0)
        self.assertEqual(req.calc_preference_value(5), 1.0)
        self.assertEqual(req.calc_preference_value(10), 1.0)
        self.assertEqual(req.calc_preference_value(11), 0.0)
        self.assertRaises(AssertionError, req.calc_preference_value, -1)
    def test_availability_requirement_copy(self):
        req = AvailabilityRequirement(0, 10)
        req_copy = req.copy()
        self.assertEqual(req.to_dict(), req_copy.to_dict())
        self.assertIsNot(req, req_copy)
    def test_availability_requirement_to_from_dict(self):
        original = AvailabilityRequirement(0, 10)
        as_dict = original.to_dict()
        reconstructed = MissionRequirement.from_dict(as_dict)
        self.assertIsInstance(reconstructed, AvailabilityRequirement)
        self.assertEqual(original.to_dict(), reconstructed.to_dict())

    def test_measurement_duration_requirement(self):
        req = MeasurementDurationRequirement([10800, 7200, 3600], 
                                             [  1.0,  0.5,  0.1])
        self.assertTrue(isinstance(req, MeasurementDurationRequirement))
        self.assertEqual(req.requirement_type, MissionRequirement.TEMPORAL)
        self.assertEqual(req.attribute, TemporalRequirement.DURATION)
        self.assertEqual(req.calc_preference_value(0), 0.1)  # Should be 0.1
        self.assertEqual(req.calc_preference_value(3600), 0.1)  # Should be 0.1
        self.assertTrue(0.5 >= req.calc_preference_value(4000) >= 0.1)  # Should be between 0.5 and 0.1
        self.assertTrue(1.0 >= req.calc_preference_value(8000) >= 0.5)  # Should be between 1.0 and 0.5
        self.assertEqual(req.calc_preference_value(10800), 1.0)  # Should be 1.0
        self.assertEqual(req.calc_preference_value(20000), 1.0)  # Should be 0.1

    def test_revisit_time_requirement(self):
        req = RevisitTemporalRequirement([3600, 7200, 10800], [1.0, 0.5, 0.1])
        self.assertTrue(isinstance(req, RevisitTemporalRequirement))
        self.assertEqual(req.requirement_type, MissionRequirement.TEMPORAL)
        self.assertEqual(req.attribute, TemporalRequirement.REVISIT)
        self.assertTrue(1.0 >= req.calc_preference_value(4000) >= 0.5)  # Should be between 1.0 and 0.5
        self.assertEqual(req.calc_preference_value(3000), 1.0)  # Should be 1.0
        self.assertEqual(req.calc_preference_value(20000), 0.1)  # Should be 0.1
    def test_revisit_time_requirement_copy(self):
        req = RevisitTemporalRequirement([3600, 7200], [1.0, 0.5])
        req_copy = req.copy()
        self.assertEqual(req.to_dict(), req_copy.to_dict())
        self.assertIsNot(req, req_copy)
    def test_revisit_time_requirement_to_from_dict(self):
        original = RevisitTemporalRequirement([3600.0, 7200.0], [1.0, 0.5])
        as_dict = original.to_dict()
        reconstructed = MissionRequirement.from_dict(as_dict)
        self.assertIsInstance(reconstructed, RevisitTemporalRequirement)
        self.assertEqual(original.to_dict(), reconstructed.to_dict())
    
    def test_no_change_reobservation_strategy_requirement(self):
        req = NoChangeReobservationStrategy()
        self.assertTrue(isinstance(req, ReobservationStrategyRequirement))
        self.assertEqual(req.requirement_type, MissionRequirement.TEMPORAL)
        self.assertEqual(req.attribute, TemporalRequirement.N_OBS)
        self.assertTrue(isinstance(req, NoChangeReobservationStrategy))
        self.assertEqual(req.calc_preference_value(1), 1.0)
        self.assertEqual(req.calc_preference_value(2), 1.0)
    def test_no_change_reobservation_strategy_copy(self):
        req = NoChangeReobservationStrategy()
        req_copy = req.copy()
        self.assertEqual(req.to_dict(), req_copy.to_dict())
        self.assertIsNot(req, req_copy)
    def test_no_change_reobservation_strategy_to_from_dict(self):
        original = NoChangeReobservationStrategy()
        as_dict = original.to_dict()
        reconstructed = MissionRequirement.from_dict(as_dict)
        self.assertIsInstance(reconstructed, NoChangeReobservationStrategy)
        self.assertEqual(original.to_dict(), reconstructed.to_dict())
    
    def test_exponential_saturation_reobservation_strategy(self):
        req = ExpSaturationReobservationsStrategy(0.5)
        self.assertTrue(isinstance(req, ReobservationStrategyRequirement))
        self.assertEqual(req.requirement_type, MissionRequirement.TEMPORAL)
        self.assertEqual(req.attribute, TemporalRequirement.N_OBS)
        self.assertTrue(isinstance(req, ExpSaturationReobservationsStrategy))
        self.assertTrue(req.calc_preference_value(1) <= 0.5)
        self.assertTrue(0.5 <= req.calc_preference_value(2))
    def test_exponential_saturation_reobservation_strategy_copy(self):
        req = ExpSaturationReobservationsStrategy(0.5)
        req_copy = req.copy()
        self.assertEqual(req.to_dict(), req_copy.to_dict())
        self.assertIsNot(req, req_copy)
    def test_exponential_saturation_reobservation_strategy_to_from_dict(self):
        original = ExpSaturationReobservationsStrategy(0.5)
        as_dict = original.to_dict()
        reconstructed = MissionRequirement.from_dict(as_dict)
        self.assertIsInstance(reconstructed, ExpSaturationReobservationsStrategy)
        self.assertEqual(original.to_dict(), reconstructed.to_dict())
    
    def test_logarithmic_threshold_reobservation_strategy(self):
        req = LogThresholdReobservationsStrategy(2, 0.5)
        self.assertTrue(isinstance(req, ReobservationStrategyRequirement))
        self.assertEqual(req.requirement_type, MissionRequirement.TEMPORAL)
        self.assertEqual(req.attribute, TemporalRequirement.N_OBS)
        self.assertTrue(isinstance(req, LogThresholdReobservationsStrategy))        
        self.assertTrue(req.calc_preference_value(1) < 0.5)
        self.assertAlmostEqual(req.calc_preference_value(2), 0.5)
        self.assertAlmostEqual(req.calc_preference_value(10), 0.98, places=2)
    def test_logarithmic_threshold_reobservation_strategy_copy(self):
        req = LogThresholdReobservationsStrategy(2, 0.5)
        req_copy = req.copy()
        self.assertEqual(req.to_dict(), req_copy.to_dict())
        self.assertIsNot(req, req_copy)
    def test_logarithmic_threshold_reobservation_strategy_to_from_dict(self):
        original = LogThresholdReobservationsStrategy(2, 0.5)
        as_dict = original.to_dict()
        reconstructed = MissionRequirement.from_dict(as_dict)
        self.assertIsInstance(reconstructed, LogThresholdReobservationsStrategy)
        self.assertEqual(original.to_dict(), reconstructed.to_dict())
    
    def test_exponential_decay_reobservation_strategy(self):
        req = ExpDecayReobservationStrategy(2)
        self.assertTrue(isinstance(req, ReobservationStrategyRequirement))
        self.assertEqual(req.requirement_type, MissionRequirement.TEMPORAL)
        self.assertEqual(req.attribute, TemporalRequirement.N_OBS)
        self.assertTrue(isinstance(req, ExpDecayReobservationStrategy))
        self.assertEqual(req.calc_preference_value(0), 1.0)
        self.assertTrue(0 < req.calc_preference_value(1) <= 0.5)
        self.assertAlmostEqual(req.calc_preference_value(4), 0.0, places=2)
    def test_exponential_decay_reobservation_strategy_copy(self):
        req = ExpDecayReobservationStrategy(2)
        req_copy = req.copy()
        self.assertEqual(req.to_dict(), req_copy.to_dict())
        self.assertIsNot(req, req_copy)
    def test_exponential_decay_reobservation_strategy_to_from_dict(self):
        original = ExpDecayReobservationStrategy(2)
        as_dict = original.to_dict()
        reconstructed = MissionRequirement.from_dict(as_dict)
        self.assertIsInstance(reconstructed, ExpDecayReobservationStrategy)
        self.assertEqual(original.to_dict(), reconstructed.to_dict())
    
    def test_guassian_threshold_reobservation_strategy(self):
        req = GaussianThresholdReobservationsStrategy(4, 0.5)
        self.assertTrue(isinstance(req, ReobservationStrategyRequirement))
        self.assertEqual(req.requirement_type, MissionRequirement.TEMPORAL)
        self.assertEqual(req.attribute, TemporalRequirement.N_OBS)
        self.assertTrue(isinstance(req, GaussianThresholdReobservationsStrategy))
        self.assertAlmostEqual(req.calc_preference_value(0), 0.0, places=4)
        self.assertTrue(0 < req.calc_preference_value(3) <= 0.5, "Value should be between 0 and 0.5")
        self.assertAlmostEqual(req.calc_preference_value(4), 1, places=4)
        self.assertTrue(0 < req.calc_preference_value(5) <= 0.5, "Value should be between 0 and 0.5")
        self.assertAlmostEqual(req.calc_preference_value(8), 0.0, places=4)
    def test_guassian_threshold_reobservation_strategy_copy(self):
        req = GaussianThresholdReobservationsStrategy(4, 0.5)
        req_copy = req.copy()
        self.assertEqual(req.to_dict(), req_copy.to_dict())
        self.assertIsNot(req, req_copy)
    def test_guassian_threshold_reobservation_strategy_to_from_dict(self):
        original = GaussianThresholdReobservationsStrategy(4, 0.5)
        as_dict = original.to_dict()
        reconstructed = MissionRequirement.from_dict(as_dict)
        self.assertIsInstance(reconstructed, GaussianThresholdReobservationsStrategy)
        self.assertEqual(original.to_dict(), reconstructed.to_dict())

    def test_triangle_threshold_reobservation_strategy(self):
        req = TriangleThresholdReobservationsStrategy(4, 2.0)
        self.assertTrue(isinstance(req, ReobservationStrategyRequirement))
        self.assertEqual(req.requirement_type, MissionRequirement.TEMPORAL)
        self.assertEqual(req.attribute, TemporalRequirement.N_OBS)
        self.assertTrue(isinstance(req, TriangleThresholdReobservationsStrategy))
        self.assertAlmostEqual(req.calc_preference_value(0), 0.0, places=4)
        self.assertAlmostEqual(req.calc_preference_value(3), 0.5, places=4)
        self.assertAlmostEqual(req.calc_preference_value(4), 1, places=4)
        self.assertAlmostEqual(req.calc_preference_value(5), 0.5, places=4)
        self.assertAlmostEqual(req.calc_preference_value(8), 0.0, places=4)
    def test_triangle_threshold_reobservation_strategy_copy(self):
        req = TriangleThresholdReobservationsStrategy(4, 2.0)
        req_copy = req.copy()
        self.assertEqual(req.to_dict(), req_copy.to_dict())
        self.assertIsNot(req, req_copy)
    def test_triangle_threshold_reobservation_strategy_to_from_dict(self):
        original = TriangleThresholdReobservationsStrategy(4, 2.0)
        as_dict = original.to_dict()
        reconstructed = MissionRequirement.from_dict(as_dict)
        self.assertIsInstance(reconstructed, TriangleThresholdReobservationsStrategy)
        self.assertEqual(original.to_dict(), reconstructed.to_dict())

class TestSpatialRequirements(unittest.TestCase):
    # Spatial Requirements
    ## Point Target Spatial Requirement
    def test_point_target_spatial_requirement(self):
        target = (0.0, 0.0, 0, 0)
        req = PointTargetSpatialRequirement(target,
                                             distance_threshold=0.1)
        self.assertTrue(isinstance(req, PointTargetSpatialRequirement))
        self.assertEqual(req.requirement_type, MissionRequirement.SPATIAL)
        self.assertEqual(req.location_type, SpatialRequirement.POINT)
        self.assertEqual(req.target, target)
        self.assertEqual(req.distance_threshold, 0.1)
    def test_point_target_spatial_requirement_evaluation(self):
        target = (0.0, 0.0, 0, 0)
        req = PointTargetSpatialRequirement(target, distance_threshold=10.0)
        self.assertAlmostEqual(req.calc_preference_value((0.0, 0.0, 0, 0)), 1.0)
        self.assertAlmostEqual(req.calc_preference_value((0.05, 0.05, 0, 1000)), 1.0)
        self.assertAlmostEqual(req.calc_preference_value((0.2, 0.2, 0, 1000)), 0.0)
    def test_point_target_spatial_requirement_copy(self):
        target = (0.0, 0.0, 0, 0)
        req = PointTargetSpatialRequirement(target, distance_threshold=0.1)
        req_copy = req.copy()
        self.assertEqual(req.to_dict(), req_copy.to_dict())
        self.assertIsNot(req, req_copy)
    def test_point_target_spatial_requirement_to_from_dict(self):
        target = (0.0, 0.0, 0, 0)
        req = PointTargetSpatialRequirement(target, distance_threshold=0.1)
        as_dict = req.to_dict()
        reconstructed = MissionRequirement.from_dict(as_dict)
        self.assertIsInstance(reconstructed, PointTargetSpatialRequirement)
        self.assertEqual(req.to_dict(), reconstructed.to_dict())

    ## Target List Spatial Requirement
    def test_target_list_spatial_requirement(self):
        targets = [(0.0, 0.0, 0, 0), (1.0, 1.0, 0, 1)]
        req = TargetListSpatialRequirement(targets, 
                                           distance_threshold=0.1)
        
        self.assertTrue(isinstance(req, TargetListSpatialRequirement))
        self.assertEqual(req.requirement_type, MissionRequirement.SPATIAL)
        self.assertEqual(req.location_type, SpatialRequirement.LIST)
        self.assertTrue(all([target in req.targets for target in targets]))
        self.assertEqual(req.distance_threshold, 0.1)
    def test_target_list_spatial_requirement_evaluation(self):
        targets = [(0.0, 0.0, 0, 0), (1.0, 1.0, 0, 1)]
        req = TargetListSpatialRequirement(targets,
                                             distance_threshold=10.0)
        self.assertAlmostEqual(req.calc_preference_value((0.0, 0.0, 0, 0)), 1.0)
        self.assertAlmostEqual(req.calc_preference_value((0.05, 0.05, 0, 1000)), 1.0)
        self.assertAlmostEqual(req.calc_preference_value((0.2, 0.2, 0, 1000)), 0.0)
    def test_target_list_spatial_requirement_copy(self):
        targets = [(0.0, 0.0, 0, 0), (1.0, 1.0, 0, 1)]
        req = TargetListSpatialRequirement(targets, distance_threshold=0.1)
        req_copy = req.copy()
        self.assertEqual(req.to_dict(), req_copy.to_dict())
        self.assertIsNot(req, req_copy)
    def test_target_list_spatial_requirement_to_from_dict(self):
        targets = [(0.0, 0.0, 0, 0), (1.0, 1.0, 0, 1)]
        req = TargetListSpatialRequirement(targets, distance_threshold=0.1)
        as_dict = req.to_dict()
        reconstructed = MissionRequirement.from_dict(as_dict)
        self.assertIsInstance(reconstructed, TargetListSpatialRequirement)
        self.assertEqual(req.to_dict(), reconstructed.to_dict())

    ## Test Grid Target Spatial Requirement
    def test_grid_target_spatial_requirement(self):
        grid_name = "test_grid"
        grid_index = 0
        grid_size = 10
        req = GridTargetSpatialRequirement(grid_name, grid_index, grid_size)

        self.assertIsInstance(req, GridTargetSpatialRequirement)
        self.assertEqual(req.requirement_type, MissionRequirement.SPATIAL)
        self.assertEqual(req.location_type, SpatialRequirement.GRID)
        self.assertEqual(req.grid_name, grid_name)
        self.assertEqual(req.grid_index, grid_index)
        self.assertEqual(req.grid_size, grid_size)
    def test_grid_target_spatial_requirement_evaluation(self):
        grid_name = "test_grid"
        grid_index = 0
        grid_size = 10
        req = GridTargetSpatialRequirement(grid_name, grid_index, grid_size)
        # Test evaluation for a point within the grid cell
        self.assertAlmostEqual(req.calc_preference_value((0.0, 0.0, grid_index, 5)), 1.0)
        # Test evaluation for a point outside the grid cell
        self.assertAlmostEqual(req.calc_preference_value((0.2, 0.2, grid_index, 11)), 0.0)
        self.assertAlmostEqual(req.calc_preference_value((0.2, 0.2, grid_index+1, 11)), 0.0)
    def test_grid_target_spatial_requirement_copy(self):
        grid_name = "test_grid"
        grid_index = 0
        grid_size = 10
        req = GridTargetSpatialRequirement(grid_name, grid_index, grid_size)
        req_copy = req.copy()
        self.assertEqual(req.to_dict(), req_copy.to_dict())
        self.assertIsNot(req, req_copy)
    def test_grid_target_spatial_requirement_to_from_dict(self):
        grid_name = "test_grid"
        grid_index = 0
        grid_size = 10
        req = GridTargetSpatialRequirement(grid_name, grid_index, grid_size)
        as_dict = req.to_dict()
        reconstructed = MissionRequirement.from_dict(as_dict)
        self.assertIsInstance(reconstructed, GridTargetSpatialRequirement)
        self.assertEqual(req.to_dict(), reconstructed.to_dict())
    def test_grid_target_spatial_requirement_invalid_location(self):
        grid_name = "test_grid"
        grid_index = 0
        grid_size = 10
        req = GridTargetSpatialRequirement(grid_name, grid_index, grid_size)
        # Test with a location that is not a tuple/list of length 4
        with self.assertRaises(AssertionError):
            req.calc_preference_value((0.0, 0.0, grid_index))  # Missing grid index and gp index
    def test_grid_target_spatial_requirement_invalid_grid_index(self):
        grid_name = "test_grid"
        grid_index = 0
        grid_size = 10
        req = GridTargetSpatialRequirement(grid_name, grid_index, grid_size)
        # Test with a grid index that is not an integer or is negative
        with self.assertRaises(AssertionError):
            req.calc_preference_value((0.0, 0.0, -1, 5))  # Negative grid index
    def test_grid_target_spatial_requirement_invalid_gp_index(self):
        grid_name = "test_grid"
        grid_index = 0
        grid_size = 10
        req = GridTargetSpatialRequirement(grid_name, grid_index, grid_size)
        # Test with a gp index that is not an integer or is out of bounds
        with self.assertRaises(AssertionError):
            req.calc_preference_value((0.0, 0.0, grid_index, -1))  # Negative gp index
            with self.assertRaises(AssertionError):
                req.calc_preference_value((0.0, 0.0, grid_index, grid_size))  # gp index equal to grid size

class TestCapabilityRequirements(unittest.TestCase):
    def test_capability_requirement_valid(self):
        req = CapabilityRequirement("instrument", ["optical", "radar"])
        self.assertEqual(req.attribute, "instrument")
        self.assertTrue(all(val in req.valid_values for val in ["optical", "radar"]))
        self.assertEqual(req.calc_preference_value("optical"), 1.0)
        self.assertEqual(req.calc_preference_value("radar"), 1.0)
        self.assertEqual(req.calc_preference_value("thermal"), 0.0)
    def test_capability_requirement_invalid(self):
        # Test with an empty valid values list
        with self.assertRaises(AssertionError):
            CapabilityRequirement("instrument", [])
        # Test with a non-list valid values
        with self.assertRaises(AssertionError):
            CapabilityRequirement("instrument", "optical")
    def test_capability_requirement_copy(self):
        req = CapabilityRequirement("sensor_type", ["optical", "radar"])
        req_copy = req.copy()
        self.assertEqual(req.to_dict(), req_copy.to_dict())
        self.assertIsNot(req, req_copy)
    def test_capability_requirement_to_from_dict(self):
        original = CapabilityRequirement("sensor_type", ["optical", "radar"])
        as_dict = original.to_dict()
        reconstructed = MissionRequirement.from_dict(as_dict)
        self.assertIsInstance(reconstructed, CapabilityRequirement)
        self.assertEqual(original.to_dict(), reconstructed.to_dict())
    

if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Requirement Definition Test')
    
    # run tests
    unittest.main()