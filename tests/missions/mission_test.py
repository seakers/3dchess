import os
import unittest

import numpy as np

from chess3d.mission import *
from chess3d.simulation import Simulation
from chess3d.utils import print_welcome


class TestGeophysicalEvent(unittest.TestCase):
    """
    O4: Observe events “Algal Blooms” (w=10)
        Main parameter: Chl-A, MR = O1
        DA: See slides (Chl-A from VNIR radiances using formula, then compare to historical values for that location)
        CA: Severity proportional to lake area (as in paper) 
        CO: Secondary params = Water temperature and water level, MR as in O2 and O3
        RO: From Ben’s paper, rewards for subsequent observations or something simple like U(n) first increases to guarantee some reobs but then decreases exponentially beyond a certain #obs (e.g., 3)

    """
    def test_initialization_valid(self):
        event = GeophysicalEvent('Algal Bloom', 
                                 1.0, 
                                 [
                                    (0.0,0.0,0,0),
                                    (1.0,1.0,0,1)
                                  ], 
                                 0.5, 
                                 1.0
                                )
        self.assertEqual(event.event_type, "algal bloom")
        self.assertEqual(event.severity, 1.0)
        self.assertEqual(event.location, [(0.0,0.0,0,0), (1.0,1.0,0,1)])
        self.assertEqual(event.t_start, 0.5)
        self.assertEqual(event.t_end, 1.0)
        self.assertIsInstance(uuid.UUID(event.id), uuid.UUID)

    def test_invalid_event_type(self):
        with self.assertRaises(AssertionError):
            GeophysicalEvent(event_type=123, severity=0.5, location=[0, 0, 0, 0], t_start=0, t_end=1)

    def test_invalid_severity(self):
        with self.assertRaises(AssertionError):
            GeophysicalEvent(event_type="flood", severity="high", location=[0, 0, 0, 0], t_start=0, t_end=1)

    def test_invalid_location_type(self):
        with self.assertRaises(AssertionError):
            GeophysicalEvent(event_type="flood", severity=0.7, location="bad location", t_start=0, t_end=1)

    def test_invalid_location_length(self):
        with self.assertRaises(AssertionError):
            GeophysicalEvent(event_type="flood", severity=0.7, location=[0, 0], t_start=0, t_end=1)

    def test_invalid_t_start(self):
        with self.assertRaises(AssertionError):
            GeophysicalEvent(event_type="fire", severity=0.6, location=[1, 2, 3, 4], t_start="0", t_end=1)

    def test_invalid_t_end(self):
        with self.assertRaises(AssertionError):
            GeophysicalEvent(event_type="fire", severity=0.6, location=[1, 2, 3, 4], t_start=0, t_end="1")

    def test_invalid_time_order(self):
        with self.assertRaises(AssertionError):
            GeophysicalEvent(event_type="quake", severity=0.5, location=[1, 2, 3, 4], t_start=100, t_end=50)

    def test_temporal_status_methods(self):
        event = GeophysicalEvent("flood", 1.0, [1, 2, 3, 4], 100, 200)

        self.assertTrue(event.is_future(50))
        self.assertFalse(event.is_active(50))
        self.assertFalse(event.is_expired(50))

        self.assertTrue(event.is_active(150))
        self.assertFalse(event.is_future(150))
        self.assertFalse(event.is_expired(150))

        self.assertTrue(event.is_expired(250))
        self.assertFalse(event.is_active(250))
        self.assertFalse(event.is_future(250))

    def test_to_dict_and_from_dict(self):
        original = GeophysicalEvent("flood", 0.9, [0.0, 1.0, 2, 3], 100, 200)
        as_dict = original.to_dict()
        recreated = GeophysicalEvent.from_dict(as_dict)

        self.assertIsInstance(recreated, GeophysicalEvent)
        self.assertEqual(original, recreated)

    def test_event_equality_and_hash(self):
        e1 = GeophysicalEvent("fire", 0.6, [1, 1, 1, 1], 0, 10)
        e2 = GeophysicalEvent.from_dict(e1.to_dict())

        self.assertEqual(e1, e2)
        self.assertEqual(hash(e1), hash(e2))

    def test_repr_and_str(self):
        event = GeophysicalEvent("storm", 0.7, [10, 20, 1, 2], 0, 50)
        rep = repr(event)
        s = str(event)

        self.assertIn("storm", rep)
        self.assertIn("storm", s)
        self.assertIn("Severity", s)
        self.assertIn("t_start", rep)

class TestRequirements(unittest.TestCase):
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
    
    def test_reobservation_strategy_requirement(self):
        req = ReobservationStrategyRequirement("linear_increase")
        self.assertTrue(isinstance(req, ReobservationStrategyRequirement))
        self.assertEqual(req.requirement_type, MissionRequirement.TEMPORAL)
        self.assertEqual(req.attribute, TemporalRequirement.N_OBS)
        self.assertEqual(req.calc_preference_value(1), 1.0)
        self.assertEqual(req.calc_preference_value(2), 2.0)

    # def test_temporal_requirement(self):
    #     req = TemporalRequirement("time_since_obs", 
    #                               thresholds=[3600, 7200, 10800], 
    #                               scores    =[1.0, 0.5, 0.1])
    #     self.assertTrue(isinstance(req, TemporalRequirement))
    #     self.assertEqual(req.req_type, MissionRequirement.TEMPORAL)
    #     self.assertTrue(req.calc_preference_value(4000) <= 1.0)  
    # def test_temporal_copy(self):
    #     req = TemporalRequirement("time_since_obs", [3600, 7200], [1.0, 0.5])
    #     req_copy = req.copy()
    #     self.assertEqual(req.to_dict(), req_copy.to_dict())
    #     self.assertIsNot(req, req_copy)
    # def test_temporal_to_from_dict(self):
    #     original = TemporalRequirement("time_since_obs", [3600.0, 7200.0], [1.0, 0.5])
    #     as_dict = original.to_dict()
    #     reconstructed = MissionRequirement.from_dict(as_dict)
    #     self.assertIsInstance(reconstructed, TemporalRequirement)
    #     self.assertEqual(original.to_dict(), reconstructed.to_dict())

    # def test_from_dict_invalid_type(self):
    #     bad_dict = {
    #         "req_type": "unknown_type",
    #         "attribute": "foo",
    #         "thresholds": [1, 2, 3],
    #         "scores": [1.0, 0.5, 0.0]
    #     }
    #     self.assertRaises(ValueError, MissionRequirement.from_dict, bad_dict)

    # Spatial Requirements
    ## Point Target Spatial Requirement
    def test_point_target_spatial_requirement(self):
        target = (0.0, 0.0, 0, 0)
        req = PointTargetSpatialRequirement(target,
                                             distance_threshold=0.1)
        self.assertTrue(isinstance(req, PointTargetSpatialRequirement))
        self.assertEqual(req.requirement_type, MissionRequirement.SPATIAL)
        self.assertEqual(req.target_type, SpatialRequirement.POINT)
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
        self.assertEqual(req.target_type, SpatialRequirement.LIST)
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
        self.assertEqual(req.target_type, SpatialRequirement.GRID)
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


#     def test_objective(self):
#         """
#         O1: Measure Chlorophyll-A (w=1): 
#             Horizontal spatial resolution: Thresholds = [10, 30, 100] m, u = [1.0, 0.7, 0.1]
#             Spectral resolution: Thresholds = [Hyperspectral, Multispectral], u = [1, 0.5]

#         O2: Measure Water temperature (w=1)
#             Horizontal spatial resolution: Thresholds = [30, 100] m, u = [1.0, 0.3]
        
#         O3: Measure Water level (w=1)
#             Horizontal spatial resolution: Thresholds = [30, 100] m, u = [1.0, 0.5]
#             Accuracy: Thresholds = [10, 50, 100] cm, u = [1.0, 0.5, 0.1]

#         """
#         # Test the Objective class
#         # Create instances of MeasurementRequirement
#         req_1 = MeasurementRequirement('spatial_resolution', 
#                                        [10, 30, 100], 
#                                        [1.0, 0.7, 0.1])
#         req_2 = MeasurementRequirement('spectral_resolution', 
#                                        ["Hyperspectral", "Multispectral"], 
#                                        [1.0, 0.5])
#         req_3 = MeasurementRequirement('accuracy', 
#                                        [10, 50, 100], # [cm] 
#                                        [1.0, 0.5, 0.1])
        
#         # create objectives
#         o_1 = MissionObjective('Chlorophyll-A',
#                         1.0, 
#                         [req_1, req_2],
#                         ['VNIR'],
#                         'no_change'
#                     )
#         o_2 = MissionObjective('Water Temperature',
#                         1.0, 
#                         [req_1],
#                         ['VNIR'],
#                         'no_change'
#                     )
#         o_3 = MissionObjective('Water Level',
#                         1.0, 
#                         [req_1, req_3],
#                         ['VNIR'],
#                         'no_change'
#                     )
#         o_4 = MissionObjective('Water Level',
#                         1.0, 
#                         [req_1, req_3],
#                         ['TIR'],
#                         'no_change'
#                     )
        
#         # create example measurement
#         measurement = {
#             "instrument": "VNIR",
#             "spatial_resolution": 20,
#             "spectral_resolution": "Hyperspectral",
#             "n_obs" : 1,
#         }
  
#         self.assertAlmostEqual(o_1.eval_performance(measurement), 0.85*1.0)
#         self.assertAlmostEqual(o_2.eval_performance(measurement), 0.85)
#         self.assertAlmostEqual(o_3.eval_performance(measurement), 0.0)
#         self.assertAlmostEqual(o_4.eval_performance(measurement), 0.0)

#     def test_event_driven_objective(self):
#         """
#         O4: Observe events “Algal Blooms” (w=10)
#             Main parameter: Chl-A, MR = O1
#             DA: See slides (Chl-A from VNIR radiances using formula, then compare to historical values for that location)
#             CA: Severity proportional to lake area (as in paper) 
#             CO: Secondary params = Water temperature and water level, MR as in O2 and O3
#             RO: From Ben’s paper, rewards for subsequent observations or something simple like U(n) first increases to guarantee some reobs but then decreases exponentially beyond a certain #obs (e.g., 3)
#         """
#         # Create instances of MeasurementRequirement
#         req_1 = MeasurementRequirement('spatial_resolution', 
#                                        [10, 30, 100], 
#                                        [1.0, 0.7, 0.1])
#         req_2 = MeasurementRequirement('spectral_resolution', 
#                                        ["Hyperspectral", "Multispectral"], 
#                                        [1.0, 0.5])
        
#         # create event
#         event = GeophysicalEvent('Algal Bloom',
#                                     1.0, 
#                                     [
#                                         (0.0,0.0,0,0),
#                                         (1.0,1.0,0,1)
#                                     ], 
#                                     0.5, 
#                                     1.0, 
#                                     0.25)

#         # create objectives
#         o_4_1 = EventDrivenObjective('Chlorophyll-A',
#                                      10.0,
#                                      [req_1, req_2],
#                                      event.event_type,
#                                      ['VNIR'],
#                                      'linear_increase'
#                                      )
        
#         # create example measurements
#         measurement_1 = {
#             "instrument": "VNIR",
#             "spatial_resolution": 20,
#             "spectral_resolution": "Hyperspectral",
#             "n_obs" : 1
#         }
#         measurement_2 = {
#             "instrument": "VNIR",
#             "spatial_resolution": 20,
#             "spectral_resolution": "Hyperspectral",
#             "n_obs" : 2
#         }
#         measurement_3 = {
#             "instrument": "VNIR",
#             "accuracy": 1.0,
#             "n_obs" : 3
#         }
#         measurement_4 = {
#             "instrument": "TIR",
#             "spatial_resolution": 20,
#             "spectral_resolution": "Hyperspectral",
#             "n_obs" : 3
#         }

#         self.assertAlmostEqual(o_4_1.eval_performance(measurement_1), 0.85*1.0*1.0)
#         self.assertAlmostEqual(o_4_1.eval_performance(measurement_2), 0.85*1.0*2.0)
#         self.assertAlmostEqual(o_4_1.eval_performance(measurement_3), 0.0)
#         self.assertAlmostEqual(o_4_1.eval_performance(measurement_4), 0.0)


class TestToySatCase(unittest.TestCase):
    def setUp(self) -> None:
        # terminal welcome message
        print_welcome('Simulation Loading Test')
        
        # load scenario json file
        scenario_specs : dict = {
            "epoch": {
                "@type": "GREGORIAN_UT1",
                "year": 2020,
                "month": 1,
                "day": 1,
                "hour": 0,
                "minute": 0,
                "second": 0
            },
            "duration": 1.0 / 24.0,
            "propagator": {
                "@type": "J2 ANALYTICAL PROPAGATOR",
            },
            "spacecraft": [
                {
                    "@id": "thermal_sat_0_0",
                    "name": "thermal_0",
                    "spacecraftBus": {
                        "name": "BlueCanyon",
                        "mass": 20,
                        "volume": 0.5,
                        "orientation": {
                            "referenceFrame": "NADIR_POINTING",
                            "convention": "REF_FRAME_ALIGNED"
                        },
                        "components": {
                            "adcs" : {
                                "maxTorque" : 1000,
                                "maxRate" : 1
                            }
                        }
                    },
                    "instrument": {
                        # "name": "Altimeter",
                        # "@id" : "altimeter",
                        # "@type" : "Altimeter",
                        # "chirpBandwidth": 150e6,
                        # "pulseWidth": 50e-6,  
                        # "orientation": {
                        #     "referenceFrame": "NADIR_POINTING",
                        #     "convention": "REF_FRAME_ALIGNED"
                        # },
                        # "fieldOfViewGeometry": { 
                        #     "shape": "RECTANGULAR", 
                        #     "angleHeight": 2.5, 
                        #     "angleWidth": 45.0
                        # },
                        # "maneuver" : {
                        #     "maneuverType":"SINGLE_ROLL_ONLY",
                        #     "A_rollMin": -50,
                        #     "A_rollMax": 50
                        # }
                        "name": "TIR",
                        "@id" : "vnir_hyp_imager",
                        "@type" : "VNIR",
                        "detectorWidth": 6.6e-6,
                        "focalLength": 3.6,  
                        "orientation": {
                            "referenceFrame": "NADIR_POINTING",
                            "convention": "REF_FRAME_ALIGNED"
                        },
                        "fieldOfViewGeometry": { 
                            "shape": "RECTANGULAR", 
                            "angleHeight": 2.5, 
                            "angleWidth": 45.0
                        },
                        "maneuver" : {
                            "maneuverType":"SINGLE_ROLL_ONLY",
                            "A_rollMin": -50,
                            "A_rollMax": 50
                        },
                        "spectral_resolution" : "Multispectral"
                    },
                    "orbitState": {
                        "date": {
                            "@type": "GREGORIAN_UT1",
                            "year": 2020,
                            "month": 1,
                            "day": 1,
                            "hour": 0,
                            "minute": 0,
                            "second": 0
                        },
                        "state": {
                            "@type": "KEPLERIAN_EARTH_CENTERED_INERTIAL",
                            "sma": 7078,
                            "ecc": 0.01,
                            "inc": 0.0,
                            "raan": 0.0,
                            "aop": 0.0,
                            "ta": 95.0
                        }
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "heuristic",
                            "period": 1000,
                            # "horizon": 500,
                        },
                        # "replanner" : {
                        #     "@type" : "broadcaster",
                        #     "period" : 400
                        # },
                    },
                    "science" : {
                        "@type": "lookup", 
                        "eventsPath" : "./tests/missions/resources/events/toy_events.csv"
                    },
                    "mission" : "Algal blooms monitoring"
                }
            ],
            "grid": [
                {
                    "@type": "customGrid",
                    "covGridFilePath": "./tests/missions/resources/grids/toy_points.csv"
                }
            ],
            "scenario": {   
                "connectivity" : "FULL", 
                "events" : {
                    "@type": "PREDEF", 
                    "eventsPath" : "./tests/missions/resources/events/toy_events.csv"
                },
                "clock" : {
                    "@type" : "EVENT"
                },
                "scenarioPath" : "./tests/missions/",
                "name" : "toy_sat_case",
                "missionsPath" : "./tests/missions/resources/missions/missions.json"
            },
            "settings": {
                "coverageType": "GRID COVERAGE",
                "outDir" : "./tests/missions/orbit_data/toy_sat_case",
            }
        }

        # set outdir
        orbitdata_dir = os.path.join('./tests/missions', 'orbit_data')
        scenario_orbitdata_dir = os.path.join(orbitdata_dir, 'toy_sat_case')
        if not os.path.isdir(orbitdata_dir): os.mkdir(orbitdata_dir)
        if not os.path.isdir(scenario_orbitdata_dir): os.mkdir(scenario_orbitdata_dir)

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # check type of mission object
        self.assertTrue(isinstance(self.simulation, Simulation))


    # def test_planner(self) -> None:
    #     # execute mission
    #     self.simulation.execute()

    #     # print results
    #     self.simulation.print_results()

    #     print('DONE')

class TestSingleSatCase(unittest.TestCase):
    def setUp(self) -> None:
        # terminal welcome message
        print_welcome('Simulation Loading Test')
        
        # load scenario json file
        scenario_specs : dict = {
            "epoch": {
                "@type": "GREGORIAN_UT1",
                "year": 2020,
                "month": 1,
                "day": 1,
                "hour": 0,
                "minute": 0,
                "second": 0
            },
            "duration": 1500 / 3600 / 24.0,
            "propagator": {
                "@type": "J2 ANALYTICAL PROPAGATOR",
            },
            "spacecraft": [
                {
                    "@id": "thermal_sat_0_0",
                    "name": "thermal_0",
                    "spacecraftBus": {
                        "name": "BlueCanyon",
                        "mass": 20,
                        "volume": 0.5,
                        "orientation": {
                            "referenceFrame": "NADIR_POINTING",
                            "convention": "REF_FRAME_ALIGNED"
                        },
                        "components": {
                            "adcs" : {
                                "maxTorque" : 1000,
                                "maxRate" : 1
                            }
                        }
                    },
                    "instrument": {
                        "name": "TIR",
                        "@id" : "vnir_hyp_imager",
                        "@type" : "VNIR",
                        "detectorWidth": 6.6e-6,
                        "focalLength": 3.6,  
                        "orientation": {
                            "referenceFrame": "NADIR_POINTING",
                            "convention": "REF_FRAME_ALIGNED"
                        },
                        "fieldOfViewGeometry": { 
                            "shape": "RECTANGULAR", 
                            "angleHeight": 2.5, 
                            "angleWidth": 5.0
                        },
                        "maneuver" : {
                            "maneuverType":"SINGLE_ROLL_ONLY",
                            "A_rollMin": -50,
                            "A_rollMax": 50
                        },
                        "spectral_resolution" : "Multispectral"
                    },
                    "orbitState": {
                        "date": {
                            "@type": "GREGORIAN_UT1",
                            "year": 2020,
                            "month": 1,
                            "day": 1,
                            "hour": 0,
                            "minute": 0,
                            "second": 0
                        },
                        "state": {
                            "@type": "KEPLERIAN_EARTH_CENTERED_INERTIAL",
                            "sma": 7078,
                            "ecc": 0.01,
                            "inc": 60.0,
                            "raan": 0.0,
                            "aop": 0.0,
                            "ta": 95.0
                        }
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "heuristic",
                            "period": 1000,
                            # "horizon": 500,
                        },
                        # "replanner" : {
                        #     "@type" : "broadcaster",
                        #     "period" : 400
                        # },
                    },
                    "science" : {
                        "@type": "lookup", 
                        "eventsPath" : "./tests/missions/resources/events/toy_events.csv"
                    },
                    "mission" : "Algal blooms monitoring"
                }
            ],
            "grid": [
                {
                    "@type": "customGrid",
                    "covGridFilePath": "./tests/missions/resources/grids/lake_event_points.csv"
                }
            ],
            "scenario": {   
                "connectivity" : "FULL", 
                "events" : {
                    "@type": "PREDEF", 
                    "eventsPath" : "./tests/missions/resources/events/lake_events_seed-1000.csv"
                },
                "clock" : {
                    "@type" : "EVENT"
                },
                "scenarioPath" : "./tests/missions/",
                "name" : "single_sat_case",
                "missionsPath" : "./tests/missions/resources/missions/missions.json"
            },
            "settings": {
                "coverageType": "GRID COVERAGE",
                "outDir" : "./tests/missions/orbit_data/single_sat_case",
            }
        }

        # set outdir
        orbitdata_dir = os.path.join('./tests/missions', 'orbit_data')
        scenario_orbitdata_dir = os.path.join(orbitdata_dir, 'single_sat_case')
        scenario_specs['settings']['outDir'] = scenario_orbitdata_dir
        if not os.path.isdir(orbitdata_dir): os.mkdir(orbitdata_dir)
        if not os.path.isdir(scenario_orbitdata_dir): os.mkdir(scenario_orbitdata_dir)

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # check type of mission object
        self.assertTrue(isinstance(self.simulation, Simulation))


    # def test_planner(self) -> None:
    #     # execute mission
    #     self.simulation.execute()

    #     # print results
    #     self.simulation.print_results()

    #     print('DONE')

class TestSingleSatNoEventsCase(unittest.TestCase):
    def setUp(self) -> None:
        # terminal welcome message
        print_welcome('Simulation Loading Test')
        
        # load scenario json file
        scenario_specs : dict = {
            "epoch": {
                "@type": "GREGORIAN_UT1",
                "year": 2020,
                "month": 1,
                "day": 1,
                "hour": 0,
                "minute": 0,
                "second": 0
            },
            "duration": 70 / 60 / 24.0,
            "propagator": {
                "@type": "J2 ANALYTICAL PROPAGATOR",
            },
            "spacecraft": [
                {
                    "@id": "thermal_sat_0_0",
                    "name": "thermal_0",
                    "spacecraftBus": {
                        "name": "BlueCanyon",
                        "mass": 20,
                        "volume": 0.5,
                        "orientation": {
                            "referenceFrame": "NADIR_POINTING",
                            "convention": "REF_FRAME_ALIGNED"
                        },
                        "components": {
                            "adcs" : {
                                "maxTorque" : 1000,
                                "maxRate" : 1
                            }
                        }
                    },
                    "instrument": {
                        "name": "TIR",
                        "@id" : "vnir_hyp_imager",
                        "@type" : "VNIR",
                        "detectorWidth": 6.6e-6,
                        "focalLength": 3.6,  
                        "orientation": {
                            "referenceFrame": "NADIR_POINTING",
                            "convention": "REF_FRAME_ALIGNED"
                        },
                        "fieldOfViewGeometry": { 
                            "shape": "RECTANGULAR", 
                            "angleHeight": 2.5, 
                            "angleWidth": 5.0
                        },
                        "maneuver" : {
                            "maneuverType":"SINGLE_ROLL_ONLY",
                            "A_rollMin": -50,
                            "A_rollMax": 50
                        },
                        "spectral_resolution" : "Multispectral"
                    },
                    "orbitState": {
                        "date": {
                            "@type": "GREGORIAN_UT1",
                            "year": 2020,
                            "month": 1,
                            "day": 1,
                            "hour": 0,
                            "minute": 0,
                            "second": 0
                        },
                        "state": {
                            "@type": "KEPLERIAN_EARTH_CENTERED_INERTIAL",
                            "sma": 7078,
                            "ecc": 0.01,
                            "inc": 60.0,
                            "raan": 0.0,
                            "aop": 0.0,
                            "ta": 0.0
                        }
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "milp",
                            "licensePath": "./gurobi.lic",
                            "objective" : "duration",
                            "period": 1000,
                            # "horizon": 500,
                        },
                        # "replanner" : {
                        #     "@type" : "broadcaster",
                        #     "mode" : "opportunistic",
                        #     "period" : 400
                        # },
                    },
                    "science" : {
                        "@type": "lookup", 
                        "eventsPath" : "./tests/missions/resources/events/no_events.csv"
                    },
                    "mission" : "Algal blooms monitoring"
                },
                # {
                #     "@id": "thermal_sat_0_1",
                #     "name": "thermal_1",
                #     "spacecraftBus": {
                #         "name": "BlueCanyon",
                #         "mass": 20,
                #         "volume": 0.5,
                #         "orientation": {
                #             "referenceFrame": "NADIR_POINTING",
                #             "convention": "REF_FRAME_ALIGNED"
                #         },
                #         "components": {
                #             "adcs" : {
                #                 "maxTorque" : 1000,
                #                 "maxRate" : 1
                #             }
                #         }
                #     },
                #     "instrument": {
                #         "name": "TIR",
                #         "@id" : "vnir_hyp_imager",
                #         "@type" : "VNIR",
                #         "detectorWidth": 6.6e-6,
                #         "focalLength": 3.6,  
                #         "orientation": {
                #             "referenceFrame": "NADIR_POINTING",
                #             "convention": "REF_FRAME_ALIGNED"
                #         },
                #         "fieldOfViewGeometry": { 
                #             "shape": "RECTANGULAR", 
                #             "angleHeight": 2.5, 
                #             "angleWidth": 5.0
                #         },
                #         "maneuver" : {
                #             "maneuverType":"SINGLE_ROLL_ONLY",
                #             "A_rollMin": -50,
                #             "A_rollMax": 50
                #         },
                #         "spectral_resolution" : "Multispectral"
                #     },
                #     "orbitState": {
                #         "date": {
                #             "@type": "GREGORIAN_UT1",
                #             "year": 2020,
                #             "month": 1,
                #             "day": 1,
                #             "hour": 0,
                #             "minute": 0,
                #             "second": 0
                #         },
                #         "state": {
                #             "@type": "KEPLERIAN_EARTH_CENTERED_INERTIAL",
                #             "sma": 7078,
                #             "ecc": 0.01,
                #             "inc": -120.0,
                #             "raan": 0.0,
                #             "aop": 0.0,
                #             "ta": 270.0
                #         }
                #     },
                #     "planner" : {
                #         # "preplanner" : {
                #         #     "@type" : "heuristic",
                #         #     "period": 1000,
                #         #     # "horizon": 500,
                #         # },
                #         "replanner" : {
                #             "@type" : "worker",
                #             # "mode" : "opportunistic",
                #             # "period" : 400
                #         },
                #     },
                #     "science" : {
                #         "@type": "lookup", 
                #         "eventsPath" : "./tests/missions/resources/events/no_events.csv"
                #     },
                #     "mission" : "Algal blooms monitoring"
                # }
            ],
            "grid": [
                {
                    "@type": "customGrid",
                    "covGridFilePath": "./tests/missions/resources/grids/lake_event_points.csv"
                }
            ],
            "scenario": {   
                "connectivity" : "LOS", 
                "events" : {
                    "@type": "PREDEF", 
                    "eventsPath" : "./tests/missions/resources/events/no_events.csv"
                },
                "clock" : {
                    "@type" : "EVENT"
                },
                "scenarioPath" : "./tests/missions/",
                "name" : "single_sat_no_events_case",
                "missionsPath" : "./tests/missions/resources/missions/missions.json"
            },
            "settings": {
                "coverageType": "GRID COVERAGE",
                "outDir" : "./tests/missions/orbit_data/single_sat_no_events_case",
            }
        }

        # set outdir
        orbitdata_dir = os.path.join('./tests/missions', 'orbit_data')
        scenario_orbitdata_dir = os.path.join(orbitdata_dir, 'single_sat_no_events_case')
        scenario_specs['settings']['outDir'] = scenario_orbitdata_dir
        if not os.path.isdir(orbitdata_dir): os.mkdir(orbitdata_dir)
        if not os.path.isdir(scenario_orbitdata_dir): os.mkdir(scenario_orbitdata_dir)

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # check type of mission object
        self.assertTrue(isinstance(self.simulation, Simulation))


    # def test_planner(self) -> None:
    #     # execute mission
    #     self.simulation.execute()

    #     # print results
    #     self.simulation.print_results()

    #     print('DONE')

if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Mission Definitions Test')
    
    # run tests
    unittest.main()