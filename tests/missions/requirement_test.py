import unittest

import numpy as np

from chess3d.mission.requirements import CapabilityPreferenceStrategies, ConstantValueRequirement, ExpDecayRequirement, ExpSaturationRequirement, ExplicitCapabilityRequirement, GaussianRequirement, GridSpatialRequirement, IntervalInterpolationRequirement, LogThresholdRequirement, MissionRequirement, MultiPointSpatialRequirement, PerformancePreferenceStrategies, RequirementTypes, CategoricalRequirement, SinglePointSpatialRequirement, SpatialPreferenceStrategies, StepsRequirement, TriangleRequirement
from chess3d.utils import print_welcome

"""
---------------------------------
PERFORMANCE REQUIREMENT TESTS
---------------------------------
"""
class TestCategoricalRequirement(unittest.TestCase):
    def setUp(self):

        self.attribute = "test_attribute"
        self.req = CategoricalRequirement(
                                        attribute=self.attribute,
                                        preferences={"A": 1, "B": 0.5, "C": 0}
                                    )
        

    def test_constructor(self):
        # Test creation of a CategoricalRequirement        
        self.assertIsInstance(self.req, CategoricalRequirement)
        self.assertEqual(self.req.req_type, RequirementTypes.PERFORMANCE.value)
        self.assertEqual(self.req.attribute, "test_attribute")
        self.assertEqual(self.req.strategy, PerformancePreferenceStrategies.CATEGORICAL.value)
        self.assertEqual(self.req.preferences, {"a": 1, "b": 0.5, "c": 0}) # keys should be lowercase

        # Test invalid preferences
        self.assertRaises(AssertionError, CategoricalRequirement, 
                          attribute=123,                  # invalid attribute type
                          preferences={"A": 1, "B": 0.5, "C": 0})  
        self.assertRaises(AssertionError, CategoricalRequirement,
                          attribute=self.attribute,
                          preferences="{'A: 1, 'B': 1}")  # Invalid preference type
        self.assertRaises(AssertionError, CategoricalRequirement, 
                          attribute=self.attribute,
                          preferences={"A": 1, "B": 0.5, "C": 0},
                          id=123 # invalid id type
                          )  
        self.assertRaises(ValueError, CategoricalRequirement, 
                          attribute=self.attribute,
                          preferences={"A": 1, "B": 0.5, "C": 0},
                          id="123" # invalid id value
                          )  
        self.assertRaises(AssertionError, CategoricalRequirement,
                          attribute=self.attribute,
                          preferences={1: "A", 2 : "B"})  # Invalid preference key types
        self.assertRaises(AssertionError, CategoricalRequirement,
                          attribute=self.attribute,
                          preferences={"A": "1", "B": "1"})  # Invalid preference value types
        self.assertRaises(AssertionError, CategoricalRequirement,
                          attribute=self.attribute,
                          preferences={"A": 1, "B": 2})  # Invalid preference value

    def test_get_preference(self):
        # Test preference retrieval
        self.assertEqual(self.req.calc_preference(self.attribute, "A"), 1)
        self.assertEqual(self.req.calc_preference(self.attribute, "B"), 0.5)
        self.assertEqual(self.req.calc_preference(self.attribute, "C"), 0)
        self.assertEqual(self.req.calc_preference(self.attribute, "D"), 0)  # Not in preferences
        self.assertRaises(AssertionError, self.req.calc_preference, "different_attribute", "A") # wrong attribute
        self.assertRaises(AssertionError, self.req.calc_preference, 1234, "A") # wrong attribute type

    def test_representation(self):
        # Test string representation
        expected_str = "PerformanceRequirement(strategy=CATEGORICAL, attribute=test_attribute)"
        self.assertEqual(repr(self.req), expected_str)

    def test_to_dict(self):
        # Test dictionary conversion
        req_dict = self.req.to_dict()
        self.assertEqual(req_dict["req_type"], RequirementTypes.PERFORMANCE.value)
        self.assertEqual(req_dict["attribute"], self.attribute)
        self.assertEqual(req_dict["strategy"], PerformancePreferenceStrategies.CATEGORICAL.value)
        self.assertEqual(req_dict["preferences"], {"a": 1, "b": 0.5, "c": 0})
        self.assertEqual(req_dict["id"], self.req.id)

    def test_from_dict(self):
        # Test creation from dictionary
        req_dict = {
            "req_type": RequirementTypes.PERFORMANCE.value,
            "strategy" : "CATEGORICAL",
            "attribute": self.attribute,
            "preferences": {"A": 1, "B": 0.5, "C": 0}
        }
        # test class method
        req_from_dict : CategoricalRequirement= CategoricalRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, CategoricalRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.preferences, self.req.preferences)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

        # test parent class method
        req_from_dict : CategoricalRequirement = MissionRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, CategoricalRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.preferences, self.req.preferences)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

    def test_copy(self):
        # Test copying of the requirement
        req_copy = self.req.copy()
        self.assertIsInstance(req_copy, CategoricalRequirement)
        self.assertEqual(req_copy.to_dict(), self.req.to_dict())
        self.assertIsNot(req_copy, self.req)  # Ensure it's a different instance

class TestConstantValueRequirement(unittest.TestCase):
    def setUp(self):
        self.attribute = "constant_value_attribute"
        self.value = 0.75
        self.req = ConstantValueRequirement(
            attribute=self.attribute,
            value=self.value
        )

    def test_constructor(self):
        # Test creation of a ConstantValueRequirement        
        self.assertIsInstance(self.req, ConstantValueRequirement)
        self.assertEqual(self.req.req_type, RequirementTypes.PERFORMANCE.value)
        self.assertEqual(self.req.attribute, self.attribute)
        self.assertEqual(self.req.strategy, PerformancePreferenceStrategies.CONSTANT.value)
        self.assertEqual(self.req.value, self.value)

        # Test invalid value type
        self.assertRaises(AssertionError, ConstantValueRequirement, 
                          attribute=123, # invalid attribute type
                          value=self.value)
        self.assertRaises(AssertionError, ConstantValueRequirement, 
                          attribute=self.attribute,
                          value="not_a_float")  # invalid value type
        self.assertRaises(AssertionError, ConstantValueRequirement, 
                          attribute=self.attribute,
                          value=10)  # invalid value 
        self.assertRaises(AssertionError, ConstantValueRequirement, 
                          attribute=self.attribute,
                          value=self.value,
                          id=123)  # invalid id type
        self.assertRaises(ValueError, ConstantValueRequirement, 
                          attribute=self.attribute,
                          value=self.value,
                          id="123")  # invalid id value
        
    def test_get_preference(self):
        # Test preference retrieval
        self.assertEqual(self.req.calc_preference(self.attribute, 123456), self.value)
        self.assertEqual(self.req.calc_preference(self.attribute, 0.254), self.value)
        self.assertRaises(AssertionError, self.req.calc_preference, 12345, 10) # wrong attribute type
        self.assertRaises(AssertionError, self.req.calc_preference, "different_attribute", 10) # wrong attribute
        self.assertRaises(AssertionError, self.req.calc_preference, self.attribute, None) # wrong value type

    def test_representation(self):
        # Test string representation
        expected_str = "PerformanceRequirement(strategy=CONSTANT, attribute=constant_value_attribute)"
        self.assertEqual(repr(self.req), expected_str)

    def test_to_dict(self):
        # Test dictionary conversion
        req_dict = self.req.to_dict()
        self.assertEqual(req_dict["req_type"], RequirementTypes.PERFORMANCE.value)
        self.assertEqual(req_dict["attribute"], self.attribute)
        self.assertEqual(req_dict["strategy"], PerformancePreferenceStrategies.CONSTANT.value)
        self.assertEqual(req_dict["value"], self.value)
        self.assertEqual(req_dict["id"], self.req.id)

    def test_from_dict(self):
        # Test creation from dictionary
        req_dict = {
            "req_type": RequirementTypes.PERFORMANCE.value,
            "strategy" : PerformancePreferenceStrategies.CONSTANT.value,
            "attribute": self.attribute,
            "value": self.value
        }
        # test class method
        req_from_dict : ConstantValueRequirement= ConstantValueRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, ConstantValueRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.value, self.req.value)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

        # test parent class method
        req_from_dict : ConstantValueRequirement = MissionRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, ConstantValueRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.value, self.req.value)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

    def test_copy(self):
        # Test copying of the requirement
        req_copy = self.req.copy()
        self.assertIsInstance(req_copy, ConstantValueRequirement)
        self.assertEqual(req_copy.to_dict(), self.req.to_dict())
        self.assertIsNot(req_copy, self.req)  # Ensure it's a different instance

class TestExpSaturationRequirement(unittest.TestCase):
    def setUp(self):
        self.attribute = "exp_saturation_attribute"
        self.sat_rate = 0.1
        self.req = ExpSaturationRequirement(
            attribute=self.attribute,
            sat_rate=self.sat_rate
        )
    
    def test_constructor(self): 
        # Test creation of an ExpSaturationRequirement        
        self.assertIsInstance(self.req, ExpSaturationRequirement)
        self.assertEqual(self.req.req_type, RequirementTypes.PERFORMANCE.value)
        self.assertEqual(self.req.attribute, self.attribute)
        self.assertEqual(self.req.strategy, PerformancePreferenceStrategies.EXP_SATURATION.value)
        self.assertEqual(self.req.sat_rate, self.sat_rate)

        # Test invalid saturation_rate type
        self.assertRaises(AssertionError, ExpSaturationRequirement, 
                          attribute=123, # invalid attribute type
                          sat_rate=self.sat_rate)
        self.assertRaises(AssertionError, ExpSaturationRequirement, 
                          attribute=self.attribute,
                          sat_rate="not_a_float")  # invalid saturation_rate type
        self.assertRaises(AssertionError, ExpSaturationRequirement, 
                          attribute=self.attribute,
                          sat_rate=-0.5)  # invalid saturation_rate value 
        self.assertRaises(AssertionError, ExpSaturationRequirement, 
                          attribute=self.attribute,
                          sat_rate=self.sat_rate,
                          id=123)  # invalid id type
        self.assertRaises(ValueError, ExpSaturationRequirement, 
                          attribute=self.attribute,
                          sat_rate=self.sat_rate,
                          id="123")  # invalid id value
        
    def test_get_preference(self):
        # Test preference retrieval
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 0), 0.0)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 10), 1 - 2.718281828459045**(-1))
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 50), 1 - 2.718281828459045**(-5))
        self.assertRaises(AssertionError, self.req.calc_preference, 12345, 10) # wrong attribute type
        self.assertRaises(AssertionError, self.req.calc_preference, "different_attribute", 10) # wrong attribute
        self.assertRaises(AssertionError, self.req.calc_preference, self.attribute, "10") # wrong value type
        self.assertRaises(AssertionError, self.req.calc_preference, self.attribute, -1) # invalid value 

    def test_representation(self):
        # Test string representation
        expected_str = "PerformanceRequirement(strategy=EXP_SATURATION, attribute=exp_saturation_attribute, sat_rate=0.1)"
        self.assertEqual(repr(self.req), expected_str)

    def test_to_dict(self):
        # Test dictionary conversion
        req_dict = self.req.to_dict()
        self.assertEqual(req_dict["req_type"], RequirementTypes.PERFORMANCE.value)
        self.assertEqual(req_dict["attribute"], self.attribute)
        self.assertEqual(req_dict["strategy"], PerformancePreferenceStrategies.EXP_SATURATION.value)
        self.assertEqual(req_dict["sat_rate"], self.sat_rate)
        self.assertEqual(req_dict["id"], self.req.id)

    def test_from_dict(self):
        # Test creation from dictionary
        req_dict = {
            "req_type": RequirementTypes.PERFORMANCE.value,
            "strategy" : PerformancePreferenceStrategies.EXP_SATURATION.value,
            "attribute": self.attribute,
            "sat_rate": self.sat_rate
        }
        # test class method
        req_from_dict : ExpSaturationRequirement= ExpSaturationRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, ExpSaturationRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.sat_rate, self.req.sat_rate)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

        # test parent class method
        req_from_dict : ExpSaturationRequirement = MissionRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, ExpSaturationRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.sat_rate, self.req.sat_rate)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

    def test_copy(self):
        # Test copying of the requirement
        req_copy = self.req.copy()
        self.assertIsInstance(req_copy, ExpSaturationRequirement)
        self.assertEqual(req_copy.to_dict(), self.req.to_dict())
        self.assertIsNot(req_copy, self.req)  # Ensure it's a different instance

class TestLogThresholdRequirement(unittest.TestCase):
    def setUp(self):
        self.attribute = "log_threshold_attribute"
        self.threshold = 5.0
        self.slope = 0.2
        self.req = LogThresholdRequirement(
            attribute=self.attribute,
            slope=self.slope,
            threshold=self.threshold
        )

    def test_constructor(self):
        # Test creation of a LogThresholdRequirement        
        self.assertIsInstance(self.req, LogThresholdRequirement)
        self.assertEqual(self.req.req_type, RequirementTypes.PERFORMANCE.value)
        self.assertEqual(self.req.attribute, self.attribute)
        self.assertEqual(self.req.strategy, PerformancePreferenceStrategies.LOG_THRESHOLD.value)
        self.assertEqual(self.req.slope, self.slope)
        self.assertEqual(self.req.threshold, self.threshold)

        # Test invalid slope and threshold types
        self.assertRaises(AssertionError, LogThresholdRequirement, 
                          attribute=123, # invalid attribute type
                          slope=self.slope,
                          threshold=self.threshold)
        self.assertRaises(AssertionError, LogThresholdRequirement, 
                          attribute=self.attribute,
                          slope="not_a_float",  # invalid slope type
                          threshold=self.threshold)
        self.assertRaises(AssertionError, LogThresholdRequirement, 
                          attribute=self.attribute,
                          slope=-1.0,  # invalid slope value
                          threshold=self.threshold)
        self.assertRaises(AssertionError, LogThresholdRequirement, 
                          attribute=self.attribute,
                          slope=self.slope,
                          threshold="not_a_float")  # invalid threshold type
        self.assertRaises(AssertionError, LogThresholdRequirement, 
                          attribute=self.attribute,
                          slope=self.slope,
                          threshold=-1.0)  # invalid threshold value
        self.assertRaises(AssertionError, LogThresholdRequirement, 
                          attribute=self.attribute,
                          slope=self.slope,
                          threshold=self.threshold,
                          id=123)  # invalid id type
        self.assertRaises(ValueError, LogThresholdRequirement, 
                          attribute=self.attribute,
                          slope=self.slope,
                          threshold=self.threshold,
                          id="123")  # invalid id value
    
    def calc_preference(self, value):
        return 1 / (1 + np.exp(-self.slope * (value - self.threshold)))

    def test_get_preference(self):
        # Test preference retrieval
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 0), self.calc_preference(0))
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 5), self.calc_preference(5))
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 10), self.calc_preference(10))
        self.assertRaises(AssertionError, self.req.calc_preference, 12345, 10) # wrong attribute type
        self.assertRaises(AssertionError, self.req.calc_preference, "different_attribute", 10) # wrong attribute
        self.assertRaises(AssertionError, self.req.calc_preference, self.attribute, "10") # wrong value type
        self.assertRaises(AssertionError, self.req.calc_preference, self.attribute, -1) # invalid value

    def test_representation(self):
        # Test string representation
        expected_str = "PerformanceRequirement(strategy=LOG_THRESHOLD, attribute=log_threshold_attribute, slope=0.2, threshold=5.0)"
        self.assertEqual(repr(self.req), expected_str)

    def test_to_dict(self):
        # Test dictionary conversion
        req_dict = self.req.to_dict()
        self.assertEqual(req_dict["req_type"], RequirementTypes.PERFORMANCE.value)
        self.assertEqual(req_dict["attribute"], self.attribute)
        self.assertEqual(req_dict["strategy"], PerformancePreferenceStrategies.LOG_THRESHOLD.value)
        self.assertEqual(req_dict["slope"], self.slope)
        self.assertEqual(req_dict["threshold"], self.threshold)
        self.assertEqual(req_dict["id"], self.req.id)

    def test_from_dict(self):
        # Test creation from dictionary
        req_dict = {
            "req_type": RequirementTypes.PERFORMANCE.value,
            "strategy" : PerformancePreferenceStrategies.LOG_THRESHOLD.value,
            "attribute": self.attribute,
            "slope": self.slope,
            "threshold": self.threshold
        }
        # test class method
        req_from_dict : LogThresholdRequirement= LogThresholdRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, LogThresholdRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.slope, self.req.slope)
        self.assertEqual(req_from_dict.threshold, self.req.threshold)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

        # test parent class method
        req_from_dict : LogThresholdRequirement = MissionRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, LogThresholdRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.slope, self.req.slope)
        self.assertEqual(req_from_dict.threshold, self.req.threshold)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

    def test_copy(self):
        # Test copying of the requirement
        req_copy = self.req.copy()
        self.assertIsInstance(req_copy, LogThresholdRequirement)
        self.assertEqual(req_copy.to_dict(), self.req.to_dict())
        self.assertIsNot(req_copy, self.req)  # Ensure it's a different instance
    
class TestExpDecayRequirement(unittest.TestCase):
    def setUp(self):
        self.attribute = "exp_decay_attribute"
        self.decay_rate = 0.05

        self.req = ExpDecayRequirement(
            attribute=self.attribute,
            decay_rate=self.decay_rate
        )

    def test_constructor(self):
        # Test creation of an ExpDecayRequirement        
        self.assertIsInstance(self.req, ExpDecayRequirement)
        self.assertEqual(self.req.req_type, RequirementTypes.PERFORMANCE.value)
        self.assertEqual(self.req.attribute, self.attribute)
        self.assertEqual(self.req.strategy, PerformancePreferenceStrategies.EXP_DECAY.value)
        self.assertEqual(self.req.decay_rate, self.decay_rate)

        # Test invalid decay_rate type
        self.assertRaises(AssertionError, ExpDecayRequirement, 
                          attribute=123, # invalid attribute type
                          decay_rate=self.decay_rate)
        self.assertRaises(AssertionError, ExpDecayRequirement, 
                          attribute=self.attribute,
                          decay_rate="not_a_float")  # invalid decay_rate type
        self.assertRaises(AssertionError, ExpDecayRequirement, 
                          attribute=self.attribute,
                          decay_rate=-0.5)  # invalid decay_rate value 
        self.assertRaises(AssertionError, ExpDecayRequirement, 
                          attribute=self.attribute,
                          decay_rate=self.decay_rate,
                          id=123)  # invalid id type
        self.assertRaises(ValueError, ExpDecayRequirement, 
                          attribute=self.attribute,
                          decay_rate=self.decay_rate,
                          id="123")  # invalid id value
        
    def test_get_preference(self):
        # Test preference retrieval
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 0), 1.0)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 10), 2.718281828459045**(-0.5))
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 50), 2.718281828459045**(-2.5))
        self.assertRaises(AssertionError, self.req.calc_preference, 12345, 10) # wrong attribute type
        self.assertRaises(AssertionError, self.req.calc_preference, "different_attribute", 10) # wrong attribute
        self.assertRaises(AssertionError, self.req.calc_preference, self.attribute, "10") # wrong value type
        self.assertRaises(AssertionError, self.req.calc_preference, self.attribute, -1) # invalid value

    def test_representation(self):
        # Test string representation
        expected_str = "PerformanceRequirement(strategy=EXP_DECAY, attribute=exp_decay_attribute, decay_rate=0.05)"
        self.assertEqual(repr(self.req), expected_str)

    def test_to_dict(self):
        # Test dictionary conversion
        req_dict = self.req.to_dict()
        self.assertEqual(req_dict["req_type"], RequirementTypes.PERFORMANCE.value)
        self.assertEqual(req_dict["attribute"], self.attribute)
        self.assertEqual(req_dict["strategy"], PerformancePreferenceStrategies.EXP_DECAY.value)
        self.assertEqual(req_dict["decay_rate"], self.decay_rate)
        self.assertEqual(req_dict["id"], self.req.id)

    def test_from_dict(self):
        # Test creation from dictionary
        req_dict = {
            "req_type": RequirementTypes.PERFORMANCE.value,
            "strategy" : PerformancePreferenceStrategies.EXP_DECAY.value,
            "attribute": self.attribute,
            "decay_rate": self.decay_rate
        }
        # test class method
        req_from_dict : ExpDecayRequirement= ExpDecayRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, ExpDecayRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.decay_rate, self.req.decay_rate)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

        # test parent class method
        req_from_dict : ExpDecayRequirement = MissionRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, ExpDecayRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.decay_rate, self.req.decay_rate)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

    def test_copy(self):    
        # Test copying of the requirement
        req_copy = self.req.copy()
        self.assertIsInstance(req_copy, ExpDecayRequirement)
        self.assertEqual(req_copy.to_dict(), self.req.to_dict())
        self.assertIsNot(req_copy, self.req)  # Ensure it's a different instance

class TestGaussianRequirement(unittest.TestCase):
    def setUp(self):
        self.attribute = "gaussian_attribute"
        self.mean = 10.0
        self.stddev = 2.0

        self.req = GaussianRequirement(
            attribute=self.attribute,
            mean=self.mean,
            stddev=self.stddev
        )

    def test_constructor(self):
        # Test creation of a GaussianRequirement        
        self.assertIsInstance(self.req, GaussianRequirement)
        self.assertEqual(self.req.req_type, RequirementTypes.PERFORMANCE.value)
        self.assertEqual(self.req.attribute, self.attribute)
        self.assertEqual(self.req.strategy, PerformancePreferenceStrategies.GAUSSIAN.value)
        self.assertEqual(self.req.mean, self.mean)
        self.assertEqual(self.req.stddev, self.stddev)

        # Test invalid mean and std_dev types
        self.assertRaises(AssertionError, GaussianRequirement, 
                          attribute=123, # invalid attribute type
                          mean=self.mean,
                          stddev=self.stddev)
        self.assertRaises(AssertionError, GaussianRequirement, 
                          attribute=self.attribute,
                          mean="not_a_float",  # invalid mean type
                          stddev=self.stddev)
        self.assertRaises(AssertionError, GaussianRequirement, 
                          attribute=self.attribute,
                          mean=self.mean,
                          stddev="not_a_float")  # invalid stddev type
        self.assertRaises(AssertionError, GaussianRequirement, 
                          attribute=self.attribute,
                          mean=self.mean,
                          stddev=-1.0)  # invalid stddev value 
        self.assertRaises(AssertionError, GaussianRequirement, 
                          attribute=self.attribute,
                          mean=self.mean,
                          stddev=self.stddev,
                          id=123)  # invalid id type
        self.assertRaises(ValueError, GaussianRequirement, 
                          attribute=self.attribute,
                          mean=self.mean,
                          stddev=self.stddev,
                          id="123")  # invalid id value
        
    def test_get_preference(self):
        # Test preference retrieval
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 10), 1.0)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 12), np.exp(-0.5))
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 14), np.exp(-2.0))
        self.assertRaises(AssertionError, self.req.calc_preference, 12345, 10) # wrong attribute type
        self.assertRaises(AssertionError, self.req.calc_preference, "different_attribute", 10) # wrong attribute
        self.assertRaises(AssertionError, self.req.calc_preference, self.attribute, "10") # wrong value type
        self.assertRaises(AssertionError, self.req.calc_preference, self.attribute, -1) # invalid value 

    def test_representation(self):
        # Test string representation
        expected_str = "PerformanceRequirement(strategy=GAUSSIAN, attribute=gaussian_attribute, mean=10.0, stddev=2.0)"
        self.assertEqual(repr(self.req), expected_str)

    def test_to_dict(self):
        # Test dictionary conversion
        req_dict = self.req.to_dict()
        self.assertEqual(req_dict["req_type"], RequirementTypes.PERFORMANCE.value)
        self.assertEqual(req_dict["attribute"], self.attribute)
        self.assertEqual(req_dict["strategy"], PerformancePreferenceStrategies.GAUSSIAN.value)
        self.assertEqual(req_dict["mean"], self.mean)
        self.assertEqual(req_dict["stddev"], self.stddev)
        self.assertEqual(req_dict["id"], self.req.id)

    def test_from_dict(self):
        # Test creation from dictionary
        req_dict = {
            "req_type": RequirementTypes.PERFORMANCE.value,
            "strategy" : PerformancePreferenceStrategies.GAUSSIAN.value,
            "attribute": self.attribute,
            "mean": self.mean,
            "stddev": self.stddev
        }
        # test class method
        req_from_dict : GaussianRequirement= GaussianRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, GaussianRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.mean, self.req.mean)
        self.assertEqual(req_from_dict.stddev, self.req.stddev)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

        # test parent class method
        req_from_dict : GaussianRequirement = MissionRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, GaussianRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.mean, self.req.mean)
        self.assertEqual(req_from_dict.stddev, self.req.stddev)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

class TestTriangleRequirement(unittest.TestCase):
    def setUp(self):
        self.attribute = "triangle_attribute"
        self.reference = 10.0
        self.width = 10.0

        self.req = TriangleRequirement(
            attribute=self.attribute,
            reference=self.reference,
            width=self.width
        )

    def test_constructor(self):
        # Test creation of a TriangleRequirement        
        self.assertIsInstance(self.req, TriangleRequirement)
        self.assertEqual(self.req.req_type, RequirementTypes.PERFORMANCE.value)
        self.assertEqual(self.req.attribute, self.attribute)
        self.assertEqual(self.req.strategy, PerformancePreferenceStrategies.TRIANGLE.value)
        self.assertEqual(self.req.reference, self.reference)
        self.assertEqual(self.req.width, self.width)

        # Test invalid reference and width types
        self.assertRaises(AssertionError, TriangleRequirement, 
                          attribute=123, # invalid attribute type
                          reference=self.reference,
                          width=self.width)
        self.assertRaises(AssertionError, TriangleRequirement, 
                          attribute=self.attribute,
                          reference="not_a_float",  # invalid reference type
                          width=self.width)
        self.assertRaises(AssertionError, TriangleRequirement, 
                          attribute=self.attribute,
                          reference=self.reference,
                          width="not_a_float")  # invalid width type
        self.assertRaises(AssertionError, TriangleRequirement, 
                          attribute=self.attribute,
                          reference=self.reference,
                          width=-1.0)  # invalid width value 
        self.assertRaises(AssertionError, TriangleRequirement, 
                          attribute=self.attribute,
                          reference=self.reference,
                          width=self.width,
                          id=123)  # invalid id type
        self.assertRaises(ValueError, TriangleRequirement, 
                          attribute=self.attribute,
                          reference=self.reference,
                          width=self.width,
                          id="123")  # invalid id value
        
    def test_get_preference(self):
        # Test preference retrieval
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 10), 1.0)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 12.5), 0.5)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 7.5), 0.5)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 5), 0.0)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 15), 0.0)
        self.assertRaises(AssertionError, self.req.calc_preference, 12345, 10) # wrong attribute type
        self.assertRaises(AssertionError, self.req.calc_preference, "different_attribute", 10) # wrong attribute
        self.assertRaises(AssertionError, self.req.calc_preference, self.attribute, "10") # wrong value type
        self.assertRaises(AssertionError, self.req.calc_preference, self.attribute, -1) # invalid value
                
    def test_representation(self):  
        # Test string representation
        expected_str = "PerformanceRequirement(strategy=TRIANGLE, attribute=triangle_attribute, reference=10.0, width=10.0)"
        self.assertEqual(repr(self.req), expected_str)

    def test_to_dict(self):
        # Test dictionary conversion
        req_dict = self.req.to_dict()
        self.assertEqual(req_dict["req_type"], RequirementTypes.PERFORMANCE.value)
        self.assertEqual(req_dict["attribute"], self.attribute)
        self.assertEqual(req_dict["strategy"], PerformancePreferenceStrategies.TRIANGLE.value)
        self.assertEqual(req_dict["reference"], self.reference)
        self.assertEqual(req_dict["width"], self.width)
        self.assertEqual(req_dict["id"], self.req.id)

    def test_from_dict(self):
        # Test creation from dictionary
        req_dict = {
            "req_type": RequirementTypes.PERFORMANCE.value,
            "strategy" : PerformancePreferenceStrategies.TRIANGLE.value,
            "attribute": self.attribute,
            "reference": self.reference,
            "width": self.width
        }
        # test class method
        req_from_dict : TriangleRequirement= TriangleRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, TriangleRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.reference, self.req.reference)
        self.assertEqual(req_from_dict.width, self.req.width)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

        # test parent class method
        req_from_dict : TriangleRequirement = MissionRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, TriangleRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.reference, self.req.reference)
        self.assertEqual(req_from_dict.width, self.req.width)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

    def test_copy(self):
        # Test copying of the requirement
        req_copy = self.req.copy()
        self.assertIsInstance(req_copy, TriangleRequirement)
        self.assertEqual(req_copy.to_dict(), self.req.to_dict())
        self.assertIsNot(req_copy, self.req)  # Ensure it's a different instance
            
class TestStepsRequirement(unittest.TestCase):
    def setUp(self):
        self.attribute = "steps_attribute"
        self.thresholds = [5,   10,   15]
        self.scores =  [0.0, 0.5,  0.75,  1.0]

        self.req = StepsRequirement(
            attribute=self.attribute,
            thresholds=self.thresholds,
            scores=self.scores
        )

    def test_constructor(self):
        # Test creation of a StepsRequirement        
        self.assertIsInstance(self.req, StepsRequirement)
        self.assertEqual(self.req.req_type, RequirementTypes.PERFORMANCE.value)
        self.assertEqual(self.req.attribute, self.attribute)
        self.assertEqual(self.req.strategy, PerformancePreferenceStrategies.STEPS.value)
        self.assertEqual(self.req.thresholds, self.thresholds)
        self.assertEqual(self.req.scores, self.scores)

        # Test invalid steps and preferences types
        self.assertRaises(AssertionError, StepsRequirement, 
                          attribute=123, # invalid attribute type
                          thresholds=self.thresholds,
                          scores=self.scores)
        self.assertRaises(AssertionError, StepsRequirement, 
                          attribute=self.attribute,
                          thresholds="not_a_list",  # invalid thresholds type
                          scores=self.scores)
        self.assertRaises(AssertionError, StepsRequirement, 
                          attribute=self.attribute,
                          thresholds=self.thresholds,
                          scores="not_a_list")  # invalid scores type
        self.assertRaises(AssertionError, StepsRequirement, 
                          attribute=self.attribute,
                          thresholds=[0, 10, 5],  # invalid thresholds order
                          scores=self.scores)
        self.assertRaises(AssertionError, StepsRequirement, 
                          attribute=self.attribute,
                          thresholds=[0, 5, 10, 15],  # invalid thresholds length
                          scores=self.scores)
        self.assertRaises(AssertionError, StepsRequirement, 
                          attribute=self.attribute,
                          thresholds=self.thresholds,
                          scores=[0.0, 1.2, 0.8])  # invalid scores value 
        self.assertRaises(AssertionError, StepsRequirement, 
                          attribute=self.attribute,
                          thresholds=self.thresholds,
                          scores=[0.0, -1.2, 0.8])  # invalid scores value 
        self.assertRaises(AssertionError, StepsRequirement, 
                          attribute=self.attribute,
                          thresholds=self.thresholds,
                          scores=self.scores,
                          id=123)  # invalid id type
        self.assertRaises(ValueError, StepsRequirement, 
                          attribute=self.attribute,
                          thresholds=self.thresholds,
                          scores=self.scores,
                          id="123")  # invalid id value
        
    def test_get_preference(self):
        # self.thresholds = [ 5,   10,    15]
        # self.scores =  [0.0,  0.5,  0.75,  1.0]

        # Test preference retrieval
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 0.0), 0.0)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 5), 0.5)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 7.5), 0.5)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 10), 0.75)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 12.5), 0.75)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 15), 1.0)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 20), 1.0)
        self.assertRaises(AssertionError, self.req.calc_preference, 12345, 10) # wrong attribute type
        self.assertRaises(AssertionError, self.req.calc_preference, "different_attribute", 10) # wrong attribute
        self.assertRaises(AssertionError, self.req.calc_preference, self.attribute, "10") # wrong value type

    def test_representation(self):
        # Test string representation
        expected_str = "PerformanceRequirement(strategy=STEPS, attribute=steps_attribute, thresholds=[5, 10, 15], scores=[0.0, 0.5, 0.75, 1.0])"
        self.assertEqual(repr(self.req), expected_str)

    def test_to_dict(self):
        # Test dictionary conversion
        req_dict = self.req.to_dict()
        self.assertEqual(req_dict["req_type"], RequirementTypes.PERFORMANCE.value)
        self.assertEqual(req_dict["attribute"], self.attribute)
        self.assertEqual(req_dict["strategy"], PerformancePreferenceStrategies.STEPS.value)
        self.assertEqual(req_dict["thresholds"], self.thresholds)
        self.assertEqual(req_dict["scores"], self.scores)
        self.assertEqual(req_dict["id"], self.req.id)

    def test_from_dict(self):
        # Test creation from dictionary
        req_dict = {
            "req_type": RequirementTypes.PERFORMANCE.value,
            "strategy" : PerformancePreferenceStrategies.STEPS.value,
            "attribute": self.attribute,
            "thresholds": self.thresholds,
            "scores": self.scores
        }
        # test class method
        req_from_dict : StepsRequirement= StepsRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, StepsRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.thresholds, self.req.thresholds)
        self.assertEqual(req_from_dict.scores, self.req.scores)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

        # test parent class method
        req_from_dict : StepsRequirement = MissionRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, StepsRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.thresholds, self.req.thresholds)
        self.assertEqual(req_from_dict.scores, self.req.scores)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

    def test_copy(self):
        # Test copying of the requirement
        req_copy = self.req.copy()
        self.assertIsInstance(req_copy, StepsRequirement)
        self.assertEqual(req_copy.to_dict(), self.req.to_dict())
        self.assertIsNot(req_copy, self.req)  # Ensure it's a different instance

class TestIntervalInterpolationRequirement(unittest.TestCase):
    def setUp(self):
        self.attribute = "interval_interpolation_attribute"
        self.thresholds = [0,  10,  20]
        self.scores =  [0.0, 0.5, 1.0]

        self.req = IntervalInterpolationRequirement(
            attribute=self.attribute,
            thresholds=self.thresholds,
            scores=self.scores
        )

    def test_constructor(self):
        # Test creation of an IntervalInterpolationRequirement        
        self.assertIsInstance(self.req, IntervalInterpolationRequirement)
        self.assertEqual(self.req.req_type, RequirementTypes.PERFORMANCE.value)
        self.assertEqual(self.req.attribute, self.attribute)
        self.assertEqual(self.req.strategy, PerformancePreferenceStrategies.INTERVAL_INTERP.value)
        self.assertEqual(self.req.thresholds, self.thresholds)
        self.assertEqual(self.req.scores, self.scores)

        # Test invalid thresholds and scores types
        self.assertRaises(AssertionError, IntervalInterpolationRequirement, 
                          attribute=123, # invalid attribute type
                          thresholds=self.thresholds,
                          scores=self.scores)
        self.assertRaises(AssertionError, IntervalInterpolationRequirement, 
                          attribute=self.attribute,
                          thresholds="not_a_list",  # invalid thresholds type
                          scores=self.scores)
        self.assertRaises(AssertionError, IntervalInterpolationRequirement, 
                          attribute=self.attribute,
                          thresholds=self.thresholds,
                          scores="not_a_list")  # invalid scores type
        self.assertRaises(AssertionError, IntervalInterpolationRequirement,  
                            attribute=self.attribute,
                            thresholds=[0, 10, 5],  # invalid thresholds order
                            scores=self.scores)
        self.assertRaises(AssertionError, IntervalInterpolationRequirement,  
                            attribute=self.attribute,
                            thresholds=[0, 10, 20, 30],  # invalid thresholds length
                            scores=self.scores)
        self.assertRaises(AssertionError, IntervalInterpolationRequirement,  
                            attribute=self.attribute,
                            thresholds=self.thresholds,
                            scores=[0.0, 1.2, 0.8])  # invalid scores value 
        self.assertRaises(AssertionError, IntervalInterpolationRequirement,  
                            attribute=self.attribute,
                            thresholds=self.thresholds,
                            scores=[0.0, -1.2, 0.8])  # invalid scores value 
        self.assertRaises(AssertionError, IntervalInterpolationRequirement,  
                            attribute=self.attribute,
                            thresholds=self.thresholds,
                            scores=[0.0, 0.5, 0.8, 1.0])  # invalid scores length
        self.assertRaises(AssertionError, IntervalInterpolationRequirement,  
                            attribute=self.attribute,
                            thresholds=self.thresholds,
                            scores=self.scores,
                            id=123)  # invalid id type
        self.assertRaises(ValueError, IntervalInterpolationRequirement,  
                            attribute=self.attribute,
                            thresholds=self.thresholds,
                            scores=self.scores,
                            id="123")  # invalid id value

    def test_get_preference(self):
        # self.thresholds = [0,  10,  20]
        # self.scores =  [0.0, 0.5, 1.0]

        # Test preference retrieval
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 0.0), 0.0)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 5), 0.25)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 10), 0.5)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 15), 0.75)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 20), 1.0)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, 25), 1.0)
        self.assertRaises(AssertionError, self.req.calc_preference, 12345, 10) # wrong attribute type
        self.assertRaises(AssertionError, self.req.calc_preference, "different_attribute", 10) # wrong attribute
        self.assertRaises(AssertionError, self.req.calc_preference, self.attribute, "10") # wrong value type

    def test_representation(self):
        # Test string representation
        expected_str = "PerformanceRequirement(strategy=INTERVAL_INTERP, attribute=interval_interpolation_attribute, thresholds=[0, 10, 20], scores=[0.0, 0.5, 1.0])"
        self.assertEqual(repr(self.req), expected_str)

    def test_to_dict(self):
        # Test dictionary conversion
        req_dict = self.req.to_dict()
        self.assertEqual(req_dict["req_type"], RequirementTypes.PERFORMANCE.value)
        self.assertEqual(req_dict["attribute"], self.attribute)
        self.assertEqual(req_dict["strategy"], PerformancePreferenceStrategies.INTERVAL_INTERP.value)
        self.assertEqual(req_dict["thresholds"], self.thresholds)
        self.assertEqual(req_dict["scores"], self.scores)
        self.assertEqual(req_dict["id"], self.req.id)

    def test_from_dict(self):
        # Test creation from dictionary
        req_dict = {
            "req_type": RequirementTypes.PERFORMANCE.value,
            "strategy" : PerformancePreferenceStrategies.INTERVAL_INTERP.value,
            "attribute": self.attribute,
            "thresholds": self.thresholds,
            "scores": self.scores
        }
        # test class method
        req_from_dict : IntervalInterpolationRequirement = IntervalInterpolationRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, IntervalInterpolationRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.thresholds, self.req.thresholds)
        self.assertEqual(req_from_dict.scores, self.req.scores)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

        # test parent class method
        req_from_dict : IntervalInterpolationRequirement = MissionRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, IntervalInterpolationRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.thresholds, self.req.thresholds)
        self.assertEqual(req_from_dict.scores, self.req.scores)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

    def test_copy(self):
        # Test copying of the requirement
        req_copy = self.req.copy()
        self.assertIsInstance(req_copy, IntervalInterpolationRequirement)
        self.assertEqual(req_copy.to_dict(), self.req.to_dict())
        self.assertIsNot(req_copy, self.req)  # Ensure it's a different instance

"""
---------------------------------
CAPABILITY REQUIREMENT TESTS
---------------------------------
"""

class TestExplicitCapabilityRequirement(unittest.TestCase):
    def setUp(self):
        self.attribute = "instrument_type"
        self.valid_values = ["Camera", "Spectrometer", "Radiometer"]

        self.req = ExplicitCapabilityRequirement(
            attribute=self.attribute,
            valid_values=self.valid_values
        )

    def test_constructor(self):
        # Test creation of an ExplicitCapabilityRequirement        
        self.assertIsInstance(self.req, ExplicitCapabilityRequirement)
        self.assertEqual(self.req.req_type, RequirementTypes.CAPABILITY.value)
        self.assertEqual(self.req.attribute, self.attribute)
        self.assertEqual(self.req.valid_values, {val.lower() for val in self.valid_values})

        # Test invalid valid_values type
        self.assertRaises(AssertionError, ExplicitCapabilityRequirement, 
                          attribute=123, # invalid attribute type
                          valid_values=self.valid_values)
        self.assertRaises(AssertionError, ExplicitCapabilityRequirement, 
                          attribute=self.attribute,
                          valid_values="not_a_list")  # invalid valid_values type
        self.assertRaises(AssertionError, ExplicitCapabilityRequirement, 
                          attribute=self.attribute,
                          valid_values=[123, "Spectrometer"])  # invalid valid_values content
        self.assertRaises(AssertionError, ExplicitCapabilityRequirement, 
                          attribute=self.attribute,
                          valid_values=self.valid_values,
                          id=123)  # invalid id type
        self.assertRaises(ValueError, ExplicitCapabilityRequirement, 
                          attribute=self.attribute,
                          valid_values=self.valid_values,
                          id="123")  # invalid id value
        
    def test_get_preference(self):
        # Test preference retrieval
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, "Camera"), 1.0)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, "camera"), 1.0)  # should not be case insensitive
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, "Spectrometer"), 1.0)
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, "UnknownInstrument"), 0.0)
        self.assertRaises(AssertionError, self.req.calc_preference, 12345, "Camera") # wrong attribute type
        self.assertRaises(AssertionError, self.req.calc_preference, "different_attribute", "Camera") # wrong attribute
        self.assertRaises(AssertionError, self.req.calc_preference, self.attribute, 123) # wrong value type

    def test_representation(self):
        # Test string representation
        expected_str = "CapabilityRequirement(strategy=EXPLICIT, attribute=instrument_type)"
        self.assertEqual(repr(self.req), expected_str)

    def test_to_dict(self):
        # Test dictionary conversion
        req_dict = self.req.to_dict()
        self.assertEqual(req_dict["req_type"], RequirementTypes.CAPABILITY.value)
        self.assertEqual(req_dict["attribute"], self.attribute)
        self.assertEqual(req_dict["strategy"], CapabilityPreferenceStrategies.EXPLICIT.value)
        self.assertEqual(req_dict["valid_values"], set(self.req.valid_values))
        self.assertEqual(req_dict["id"], self.req.id)

    def test_from_dict(self):
        # Test creation from dictionary
        req_dict = {
            "req_type": RequirementTypes.CAPABILITY.value,
            "strategy" : CapabilityPreferenceStrategies.EXPLICIT.value,
            "attribute": self.attribute,
            "valid_values": list(self.req.valid_values)
        }
        # test class method
        req_from_dict : ExplicitCapabilityRequirement= ExplicitCapabilityRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, ExplicitCapabilityRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.valid_values, self.req.valid_values)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

        # test parent class method
        req_from_dict : ExplicitCapabilityRequirement = MissionRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, ExplicitCapabilityRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.valid_values, self.req.valid_values)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

    def test_copy(self):
        # Test copying of the requirement
        req_copy = self.req.copy()
        self.assertIsInstance(req_copy, ExplicitCapabilityRequirement)
        self.assertEqual(req_copy.to_dict(), self.req.to_dict())
        self.assertIsNot(req_copy, self.req)  # Ensure it's a different instance

"""
--------------------------
SPATIAL REQUIREMENT TESTS
--------------------------
"""

class TestSinglePointSpatialRequirement(unittest.TestCase):
    def setUp(self):
        self.target_point = (34.05, -118.25, 0, 1)  # Example: Los Angeles coordinates
        self.distance_threshold = 5.0  # in kilometers
        self.attribute = 'location'
    
        self.req = SinglePointSpatialRequirement(
            target=self.target_point,
            distance_threshold=self.distance_threshold
        )

        self.alt_req = SinglePointSpatialRequirement(
            target=[self.target_point],
            distance_threshold=self.distance_threshold
        )

    def test_constructor(self):
        # Test creation of a SinglePointSpatialRequirement        
        self.assertIsInstance(self.req, SinglePointSpatialRequirement)
        self.assertEqual(self.req.req_type, RequirementTypes.SPATIAL.value)
        self.assertEqual(self.req.attribute, self.attribute)
        self.assertEqual(self.req.target, self.target_point)
        self.assertEqual(self.req.distance_threshold, self.distance_threshold)

        # Test alternative target input
        self.assertIsInstance(self.alt_req, SinglePointSpatialRequirement)
        self.assertEqual(self.alt_req.req_type, RequirementTypes.SPATIAL.value)
        self.assertEqual(self.alt_req.attribute, self.attribute)
        self.assertEqual(self.alt_req.target, self.target_point)
        self.assertEqual(self.alt_req.distance_threshold, self.distance_threshold)

        # Test invalid target and distance_threshold types
        self.assertRaises(AssertionError, SinglePointSpatialRequirement, 
                          target="not_a_tuple",  # invalid target_point type
                          distance_threshold=self.distance_threshold)
        self.assertRaises(AssertionError, SinglePointSpatialRequirement, 
                          target=(34.05, -118.25),  # invalid target_point length
                          distance_threshold=self.distance_threshold)
        self.assertRaises(AssertionError, SinglePointSpatialRequirement, 
                          target=self.target_point,
                          distance_threshold="not_a_float")  # invalid tolerance type
        self.assertRaises(AssertionError, SinglePointSpatialRequirement, 
                          target=self.target_point,
                          distance_threshold=-1.0)  # invalid tolerance value 
        self.assertRaises(AssertionError, SinglePointSpatialRequirement, 
                          target=self.target_point,
                          distance_threshold=self.distance_threshold,
                          id=123)  # invalid id type
        self.assertRaises(ValueError, SinglePointSpatialRequirement, 
                          target=self.target_point,
                          distance_threshold=self.distance_threshold,
                          id="123")  # invalid id value
        
    def test_get_preference(self):
        # Test preference retrieval
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, (34.05, -118.25, 0, 1)), 1.0)   # at target
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, [(34.05, -118.25, 0, 1)]), 1.0) # at target
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, (34.10, -120.30, 0, 2)), 0.0)   # beyond threshold
        self.assertRaises(AssertionError, self.req.calc_preference, 12345, (34.05, -118.25, 0, 1))      # wrong attribute type
        self.assertRaises(AssertionError, self.req.calc_preference, "different_attribute", (34.05, -118.25, 0, 1)) # wrong attribute
        self.assertRaises(AssertionError, self.req.calc_preference, self.attribute, "not_a_tuple") # wrong value type
        self.assertRaises(AssertionError, self.req.calc_preference, self.attribute, (34.05,)) # invalid value length

    def test_representation(self):
        # Test string representation
        expected_str = "SpatialRequirement(strategy=SINGLE_POINT, target=(0, 1))"
        self.assertEqual(repr(self.req), expected_str)

    def test_to_dict(self):
        # Test dictionary conversion
        req_dict = self.req.to_dict()
        self.assertEqual(req_dict["req_type"], RequirementTypes.SPATIAL.value)
        self.assertEqual(req_dict["attribute"], self.attribute)
        self.assertEqual(req_dict["strategy"], SpatialPreferenceStrategies.SINGLE_POINT.value)
        self.assertEqual(req_dict["target"], self.target_point)
        self.assertEqual(req_dict["distance_threshold"], self.distance_threshold)
        self.assertEqual(req_dict["id"], self.req.id)

    def test_from_dict(self):
        # Test creation from dictionary
        req_dict = {
            "req_type": RequirementTypes.SPATIAL.value,
            "strategy" : SpatialPreferenceStrategies.SINGLE_POINT.value,
            "attribute": self.attribute,
            "target": self.target_point,
            "distance_threshold": self.distance_threshold
        }
        # test class method
        req_from_dict : SinglePointSpatialRequirement= SinglePointSpatialRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, SinglePointSpatialRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.target, self.req.target)
        self.assertEqual(req_from_dict.distance_threshold, self.req.distance_threshold)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

        # test parent class method
        req_from_dict : SinglePointSpatialRequirement = MissionRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, SinglePointSpatialRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.target, self.req.target)
        self.assertEqual(req_from_dict.distance_threshold, self.req.distance_threshold)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

    def test_copy(self):
        # Test copying of the requirement
        req_copy = self.req.copy()
        self.assertIsInstance(req_copy, SinglePointSpatialRequirement)
        self.assertEqual(req_copy.to_dict(), self.req.to_dict())
        self.assertIsNot(req_copy, self.req)  # Ensure it's a different instance

class TestMultiPointSpatialRequirement(unittest.TestCase):
    def setUp(self):
        self.target_points = [
            (34.05, -118.25, 0, 0),  # Los Angeles
            (40.71, -74.01, 0, 1),   # New York
            (51.51, -0.13, 0, 2)     # London
        ]
        self.distance_threshold = 10.0  # in kilometers
        self.attribute = 'location'
    
        self.req = MultiPointSpatialRequirement(
            targets=self.target_points,
            distance_threshold=self.distance_threshold
        )

    def test_constructor(self):
        # Test creation of a MultiPointSpatialRequirement        
        self.assertIsInstance(self.req, MultiPointSpatialRequirement)
        self.assertEqual(self.req.req_type, RequirementTypes.SPATIAL.value)
        self.assertEqual(self.req.attribute, self.attribute)
        self.assertEqual(self.req.targets, self.target_points)
        self.assertEqual(self.req.distance_threshold, self.distance_threshold)

        # Test invalid targets and distance_threshold types
        self.assertRaises(AssertionError, MultiPointSpatialRequirement, 
                          targets="not_a_list",  # invalid targets type
                          distance_threshold=self.distance_threshold)
        self.assertRaises(AssertionError, MultiPointSpatialRequirement, 
                          targets=[(34.05, -118.25)],  # invalid target point length
                          distance_threshold=self.distance_threshold)
        self.assertRaises(AssertionError, MultiPointSpatialRequirement, 
                          targets=self.target_points,
                          distance_threshold="not_a_float")  # invalid tolerance type
        self.assertRaises(AssertionError, MultiPointSpatialRequirement, 
                          targets=self.target_points,
                          distance_threshold=-1.0)  # invalid tolerance value 
        self.assertRaises(AssertionError, MultiPointSpatialRequirement, 
                          targets=self.target_points,
                          distance_threshold=self.distance_threshold,
                          id=123)  # invalid id type
        self.assertRaises(ValueError, MultiPointSpatialRequirement, 
                          targets=self.target_points,
                          distance_threshold=self.distance_threshold,
                          id="123")  # invalid id value
        
    def test_get_preference(self):
        # Test preference retrieval
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, (34.05, -118.25, 0, 0)), 1.0)   # at first target
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, [(40.71, -74.01, 0, 1)]), 1.0)   # at second target
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, [(34.05, -118.25, 0, 0), (40.71, -74.01, 0, 1)]), 1.0)   # at 1st and 2nd targets
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, (48.85, 2.35, 0, 2)), 0.0)        # beyond threshold
        self.assertRaises(AssertionError, self.req.calc_preference, 12345, (34.05, -118.25, 0, 0))      # wrong attribute type
        self.assertRaises(AssertionError, self.req.calc_preference, "different_attribute", (34.05, -118.25, 0, 0)) # wrong attribute
        self.assertRaises(AssertionError, self.req.calc_preference, self.attribute, "not_a_tuple") # wrong value type
        self.assertRaises(ValueError, self.req.calc_preference, self.attribute, (34.05,)) # invalid value length

    def test_representation(self):
        # Test string representation
        expected_str = "SpatialRequirement(strategy=MULTI_POINT, num_targets=3)"
        self.assertEqual(repr(self.req), expected_str)

    def test_to_dict(self):
        # Test dictionary conversion
        req_dict = self.req.to_dict()
        self.assertEqual(req_dict["req_type"], RequirementTypes.SPATIAL.value)
        self.assertEqual(req_dict["attribute"], self.attribute)
        self.assertEqual(req_dict["strategy"], SpatialPreferenceStrategies.MULTI_POINT.value)
        self.assertEqual(req_dict["targets"], self.target_points)
        self.assertEqual(req_dict["distance_threshold"], self.distance_threshold)
        self.assertEqual(req_dict["id"], self.req.id)

    def test_from_dict(self):
        # Test creation from dictionary
        req_dict = {
            "req_type": RequirementTypes.SPATIAL.value,
            "strategy" : SpatialPreferenceStrategies.MULTI_POINT.value,
            "attribute": self.attribute,
            "targets": self.target_points,
            "distance_threshold": self.distance_threshold
        }
        # test class method
        req_from_dict : MultiPointSpatialRequirement= MultiPointSpatialRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, MultiPointSpatialRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.targets, self.req.targets)
        self.assertEqual(req_from_dict.distance_threshold, self.req.distance_threshold)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

        # test parent class method
        req_from_dict : MultiPointSpatialRequirement = MissionRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, MultiPointSpatialRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.targets, self.req.targets)
        self.assertEqual(req_from_dict.distance_threshold, self.req.distance_threshold)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

    def test_copy(self):
        # Test copying of the requirement
        req_copy = self.req.copy()
        self.assertIsInstance(req_copy, MultiPointSpatialRequirement)
        self.assertEqual(req_copy.to_dict(), self.req.to_dict())
        self.assertIsNot(req_copy, self.req)  # Ensure it's a different instance

class TestGridSpatialRequirement(unittest.TestCase):
    def setUp(self):
        self.attribute = 'location'
        self.grid_name = 'global_grid'
        self.grid_index = 0
        self.grid_size = 1000
    
        self.req = GridSpatialRequirement(
            grid_name=self.grid_name,
            grid_index=self.grid_index,
            grid_size=self.grid_size
        )

    def test_constructor(self):
        # Test creation of a GridSpatialRequirement        
        self.assertIsInstance(self.req, GridSpatialRequirement)
        self.assertEqual(self.req.req_type, RequirementTypes.SPATIAL.value)
        self.assertEqual(self.req.attribute, self.attribute)
        self.assertEqual(self.req.grid_name, self.grid_name)
        self.assertEqual(self.req.grid_index, self.grid_index)
        self.assertEqual(self.req.grid_size, self.grid_size)

        # Test invalid grid_name, grid_index, and grid_size types
        self.assertRaises(AssertionError, GridSpatialRequirement, 
                          grid_name=123,  # invalid grid_name type
                          grid_index=self.grid_index,
                          grid_size=self.grid_size)
        self.assertRaises(AssertionError, GridSpatialRequirement, 
                          grid_name=self.grid_name,
                          grid_index="not_an_int",  # invalid grid_index type
                          grid_size=self.grid_size)
        self.assertRaises(AssertionError, GridSpatialRequirement, 
                          grid_name=self.grid_name,
                          grid_index=-1,  # invalid grid_index value
                          grid_size=self.grid_size)
        self.assertRaises(AssertionError, GridSpatialRequirement, 
                          grid_name=self.grid_name,
                          grid_index=self.grid_index,
                          grid_size="not_an_int")  # invalid grid_size type
        self.assertRaises(AssertionError, GridSpatialRequirement, 
                          grid_name=self.grid_name,
                          grid_index=self.grid_index,
                          grid_size=0)  # invalid grid_size value 
        self.assertRaises(AssertionError, GridSpatialRequirement, 
                          grid_name=self.grid_name,
                          grid_index=self.grid_index,
                          grid_size=self.grid_size,
                          id=123)  # invalid id type
        self.assertRaises(ValueError, GridSpatialRequirement, 
                          grid_name=self.grid_name,
                          grid_index=self.grid_index,
                          grid_size=self.grid_size,
                          id="123")  # invalid id value
        
    def test_get_preference(self):
        # Test preference retrieval
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, (34.05, -118.25, 0, 1)), 1.0) # inside grid cell
        self.assertAlmostEqual(self.req.calc_preference(self.attribute, (34.05, -20.25, 0, 1001)), 0.0) # outside grid cell
        self.assertRaises(AssertionError, self.req.calc_preference, 12345, (34.05, -118.25, 0, 1))      # wrong attribute type
        self.assertRaises(AssertionError, self.req.calc_preference, "different_attribute", (34.05, -118.25, 0, 1)) # wrong attribute
        self.assertRaises(AssertionError, self.req.calc_preference, self.attribute, "not_a_tuple") # wrong value type
        self.assertRaises(ValueError, self.req.calc_preference, self.attribute, (34.05,)) # invalid value length
    
    def test_representation(self):
        # Test string representation
        expected_str = "SpatialRequirement(strategy=GRID, grid_name=global_grid, grid_index=0, grid_size=1000)"
        self.assertEqual(repr(self.req), expected_str)

    def test_to_dict(self):
        # Test dictionary conversion
        req_dict = self.req.to_dict()
        self.assertEqual(req_dict["req_type"], RequirementTypes.SPATIAL.value)
        self.assertEqual(req_dict["attribute"], self.attribute)
        self.assertEqual(req_dict["strategy"], SpatialPreferenceStrategies.GRID.value)
        self.assertEqual(req_dict["grid_name"], self.grid_name)
        self.assertEqual(req_dict["grid_index"], self.grid_index)
        self.assertEqual(req_dict["grid_size"], self.grid_size)
        self.assertEqual(req_dict["id"], self.req.id)

    def test_from_dict(self):
        # Test creation from dictionary
        req_dict = {
            "req_type": RequirementTypes.SPATIAL.value,
            "strategy" : SpatialPreferenceStrategies.GRID.value,
            "attribute": self.attribute,
            "grid_name": self.grid_name,
            "grid_index": self.grid_index,
            "grid_size": self.grid_size
        }
        # test class method
        req_from_dict : GridSpatialRequirement= GridSpatialRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, GridSpatialRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.grid_name, self.req.grid_name)
        self.assertEqual(req_from_dict.grid_index, self.req.grid_index)
        self.assertEqual(req_from_dict.grid_size, self.req.grid_size)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

        # test parent class method
        req_from_dict : GridSpatialRequirement = MissionRequirement.from_dict(req_dict)
        self.assertIsInstance(req_from_dict, GridSpatialRequirement)
        self.assertEqual(req_from_dict.req_type, self.req.req_type)
        self.assertEqual(req_from_dict.attribute, self.req.attribute)
        self.assertEqual(req_from_dict.grid_name, self.req.grid_name)
        self.assertEqual(req_from_dict.grid_index, self.req.grid_index)
        self.assertEqual(req_from_dict.grid_size, self.req.grid_size)
        self.assertNotEqual(req_from_dict.id, self.req.id)  # IDs should differ

    def test_copy(self):
        # Test copying of the requirement
        req_copy = self.req.copy()
        self.assertIsInstance(req_copy, GridSpatialRequirement)
        self.assertEqual(req_copy.to_dict(), self.req.to_dict())
        self.assertIsNot(req_copy, self.req)  # Ensure it's a different instance

if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Mission Requirement Definition Test')
    
    # run tests
    unittest.main()