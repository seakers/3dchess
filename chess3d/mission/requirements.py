from collections.abc import Iterable
from enum import Enum
from numbers import Number
from typing import Any, Dict, List, Set, Tuple, Union
import uuid
import numpy as np
from pyparsing import ABC, Tuple, abstractmethod

"""
---------------------------------
ABSTRACT REQUIREMENT DEFINITION
---------------------------------
"""
        
class RequirementTypes(Enum):
    CAPABILITY = 'capability'
    SPATIAL = 'spatial'
    PERFORMANCE = 'performance'

class MissionRequirement(ABC):
    def __init__(self, req_type : str, attribute: str, id : str = None):
        """
        ### Mission Requirement 
        
        Initialize a mission requirement with a requirement type, attribute, strategy, and unique ID.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "temperature", "humidity").
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.    
        """
        # validate argument types
        assert isinstance(req_type, str), "Requirement type must be a string"
        assert isinstance(attribute, str), "Attribute must be a string"
        assert isinstance(id, str) or id is None, "ID must be a string or `None`"
        
        # validate argument values
        assert req_type.lower() in RequirementTypes._value2member_map_, f"Requirement type must be one of {list(RequirementTypes._value2member_map_.keys())}"
        
        # set attributes
        self.req_type : str = req_type.lower()
        self.attribute : str = attribute.lower()
        # TODO do we really need to enforce UUID format? Could the ID be attribute-dependent?
        self.id = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())

    def calc_preference(self, attribute : str, value : Any) -> float:
        """Evaluates the preference function for a given parameter-value pair."""
        
        # check if attribute matches requirement attribute
        assert isinstance(attribute, str), "Attribute must be a string"
        assert attribute.lower() == self.attribute, \
            f"Attribute '{attribute}' does not match requirement attribute '{self.attribute}'"
        
        # calculate preference value
        result = self._eval_preference_function(value)

        # validate the result
        if not isinstance(result, Number):
            raise TypeError(f"Expected a numeric return value, got {type(result).__name__}")

        if not (0.0 <= result <= 1.0):
            raise ValueError(f"Return value {result} is not in [0, 1]")

        # Return the preference value
        return result
    
    @abstractmethod
    def _eval_preference_function(self, value : Any) -> float:
        """Evaluate the preference function for a given value."""

    @abstractmethod
    def __repr__(self):
        """String representation of the measurement requirement."""
        # return f"MissionRequirement(type={RequirementTypes._value2member_map_[self.req_type].name}, attribute={self.attribute})"
    
    def copy(self) -> 'MissionRequirement':
        """Create a copy of the measurement requirement."""
        return self.from_dict(self.to_dict())
    
    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert the measurement requirement to a dictionary."""
        return dict(self.__dict__)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, float]]) -> 'MissionRequirement':
        """Create a measurement requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        
        # unpack dictionary
        req_type = d.get("req_type")

        # initiate approriate requirement 
        if req_type.lower() == RequirementTypes.PERFORMANCE.value:
            return PerformanceRequirement.from_dict(d)
        elif req_type.lower() == RequirementTypes.CAPABILITY.value:
            return CapabilityRequirement.from_dict(d)
        elif req_type.lower() == RequirementTypes.SPATIAL.value:
            return SpatialCoverageRequirement.from_dict(d)
        
        raise NotImplementedError(f"Requirement type '{req_type}' not yet supported.")
    
"""
------------------------------------
PERFORMANCE REQUIREMENT DEFINITIONS
------------------------------------
"""

class PerformancePreferenceStrategies(Enum):
    # Categorical
    CATEGORICAL = 'categorical'

    # Discrete
    DISCRETE = 'discrete'

    # No change
    CONSTANT = 'constant'
    
    # Higher val = better   
    EXP_SATURATION = 'exp_saturation'
    LOG_THRESHOLD = 'log_threshold'
    
    # Lower val = better
    EXP_DECAY = 'exp_decay'
    
    # Bounded
    GAUSSIAN = 'gaussian'
    TRIANGLE = 'triangle'
    
    # Interval Threshold-Based
    STEPS = 'discrete_steps'
    INTERVAL_INTERP = 'discrete_intervals'

class TemporalRequirementAttributes(Enum):
    START_TIME = 'start_time'
    END_TIME = 'end_time'
    DURATION = 'duration'
    REVISIT_TIME = 'revisit_time'
    OBS_TIME = 't_img'

class ObservationRequirementAttributes(Enum):
    NUM_OBSERVATIONS = 'n_obs'

class PerformanceRequirement(MissionRequirement):
    def __init__(self, 
                 attribute : str, 
                 strategy : str,
                 id = None):
        """
        ### Performance Requirement

        Initializes a generic measurement performance requirement
        - :`attribute`: The attribute being measured (e.g., "data collected", "observations made").
        - :`strategy`: Name of the preference function strategy to be used (e.g., "categorical", "exp_saturation").
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """

        # initiate parent class
        super().__init__(RequirementTypes.PERFORMANCE.value, attribute, id)

        # validate inputs
        assert isinstance(strategy, str), "Preference strategy must be a string"
        assert strategy.lower() in PerformancePreferenceStrategies._value2member_map_, f"Preference strategy must be one of {list(PerformancePreferenceStrategies._value2member_map_.keys())}"
        
        # set attributes
        self.strategy : str = strategy.lower()

    def __repr__(self):
        """String representation of the measurement requirement."""
        return f"PerformanceRequirement(strategy={PerformancePreferenceStrategies._value2member_map_[self.strategy].name}, attribute={self.attribute})"

    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, float]]) -> 'MissionRequirement':
        """Create a performance requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'strategy']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        
        # unpack dictionary
        strategy = d.get("strategy").lower()

        # initiate approriate requirement 
        if strategy == PerformancePreferenceStrategies.CATEGORICAL.value:
            return CategoricalRequirement.from_dict(d)
        
        elif strategy == PerformancePreferenceStrategies.CONSTANT.value:
            return ConstantValueRequirement.from_dict(d)
        
        elif strategy == PerformancePreferenceStrategies.EXP_SATURATION.value:
            return ExpSaturationRequirement.from_dict(d)

        elif strategy == PerformancePreferenceStrategies.LOG_THRESHOLD.value:
            return LogThresholdRequirement.from_dict(d)

        elif strategy == PerformancePreferenceStrategies.EXP_DECAY.value:
            return ExpDecayRequirement.from_dict(d)

        elif strategy == PerformancePreferenceStrategies.GAUSSIAN.value:
            return GaussianRequirement.from_dict(d)
        
        elif strategy == PerformancePreferenceStrategies.TRIANGLE.value:
            return TriangleRequirement.from_dict(d)

        elif strategy == PerformancePreferenceStrategies.STEPS.value:
            return StepsRequirement.from_dict(d)
        
        elif strategy == PerformancePreferenceStrategies.INTERVAL_INTERP.value:
            return IntervalInterpolationRequirement.from_dict(d)
        
        # Additional strategies can be implemented here
        raise NotImplementedError(f"Preference function for strategy '{strategy}' not yet supported.")

class CategoricalRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 preferences : Dict[str, float],
                 id = None):
        """
        ### Categorical Requirement

        Initializes a requirement that assigns preference scores to categorical values.
        - :`attribute`: The attribute being measured (e.g., instrument type, agent type, etc.).
        - :`preferences`: A dictionary mapping categorical values (strings) to preference scores (floats) in the range [0, 1].
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """

        # initiate parent class
        super().__init__(attribute, PerformancePreferenceStrategies.CATEGORICAL.value, id)
        
        # validate inputs
        assert isinstance(preferences, dict), "Preferences must be a dictionary"
        for key, val in preferences.items():
            assert isinstance(key, str), "Preference keys must be strings"
            assert isinstance(val, (int, float)), "Preference values must be numeric"
            assert 0.0 <= val <= 1.0, "Preference values must be in [0, 1]"
        
        # set attributes
        self.preferences : Dict[str, float] = {key.lower(): val for key,val in preferences.items()}
    
    def _eval_preference_function(self, value : str) -> float:
        # validate inputs
        assert isinstance(value, str), "Input value must be a string"

        # normalize value to lowercase string
        value = str(value).lower()

        # return preference value
        return self.preferences.get(value, 0.0) # default preference is 0.0 if category not found
        
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'CategoricalRequirement':
        """Create a categorical requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['attribute', 'strategy', 'preferences']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert dict.get("strategy").lower() == PerformancePreferenceStrategies.CATEGORICAL.value, \
            f"Strategy does not match requirement definition. Must be '{PerformancePreferenceStrategies.CATEGORICAL.value}'"
        
        # unpack dictionary
        attribute = dict.get("attribute")
        preferences = dict.get("preferences")
        id = dict.get("id", None)
        
        # initiate requirement
        return cls(attribute, preferences, id)

class ConstantValueRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str,
                 value : float = 1.0,
                 id = None
                ):
        """
        ### Constant Value Requirement

        Initializes a requirement that always returns the same preference score.
        - :`attribute`: The attribute being measured.
        - :`value`: The constant preference score to return (default is 1.0).
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """
        # initiate parent class
        super().__init__(attribute, PerformancePreferenceStrategies.CONSTANT.value, id)

        # validate inputs
        assert isinstance(value, (int, float)), "Value must be numeric"
        assert 0.0 <= value <= 1.0, "Value must be in [0, 1]"

        # set attributes
        self.value : float = value

    def _eval_preference_function(self, value : float) -> float:
        # validate inputs
        assert isinstance(value, (int, float)), "Input value must be numeric"

        # return preference value
        return self.value # always returns the constant preference value
       
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'ConstantValueRequirement':
        """Create a constant value requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'strategy']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert dict.get("strategy").lower() == PerformancePreferenceStrategies.CONSTANT.value, \
            f"Strategy does not match requirement definition. Must be '{PerformancePreferenceStrategies.CONSTANT.value}'"

        # unpack dictionary
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        value = dict.get("value", 1.0)  # default to 1.0 if not provided
        id = dict.get("id", None)

        # initiate requirement
        return cls(attribute, value, id)
    
class ExpSaturationRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 sat_rate : float,
                 id = None
                ):
        """
        ### Exponential Saturation Requirement

        Initializes a requirement that uses an exponential saturation preference function.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "data collected", "observations made").
        - :`sat_rate`: The rate at which preference saturates (higher values lead to quicker saturation). Must be non-negative.
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """
        # initiate parent class
        super().__init__(attribute, PerformancePreferenceStrategies.EXP_SATURATION.value, id)
        
        # validate inputs
        assert isinstance(sat_rate, (int, float)), "Saturation rate must be a number"
        assert sat_rate >= 0, "Saturation rate must be non-negative"

        # set attributes
        self.sat_rate : float = sat_rate

    def _eval_preference_function(self, value : float) -> float:
        # validate inputs
        assert isinstance(value, (int, float)), "Evaluated value must be a number"
        assert value >= 0, "Evaluated value must be non-negative"

        # return preference value
        return 1.0 - np.exp(-self.sat_rate * value)
    
    def __repr__(self):
        return super().__repr__()[:-1] + f", sat_rate={self.sat_rate})"
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'ExpSaturationRequirement':
        """Create an exponential saturation requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'strategy', 'sat_rate']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert dict.get("strategy").lower() == PerformancePreferenceStrategies.EXP_SATURATION.value, \
            f"Strategy does not match requirement definition. Must be '{PerformancePreferenceStrategies.EXP_SATURATION.value}'"
        
        # unpack dictionary
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        sat_rate = dict.get("sat_rate")
        id = dict.get("id", None) 

        # initiate requirement
        return cls(attribute, sat_rate, id)
    
class LogThresholdRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 slope : float, 
                 threshold : float, 
                 id = None
                ):
        """
        ### Logarithmic Threshold Requirement
        
        Initializes a requirement that uses a logarithmic threshold preference function.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "data collected", "observations made").
        - :`slope`: The slope of the logarithmic function (higher values lead to steeper transitions). Must be positive.
        - :`threshold`: The threshold value at which preference value is 0.5. Must be non-negative.
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """
        # initiate parent class
        super().__init__(attribute, PerformancePreferenceStrategies.LOG_THRESHOLD.value, id)
        
        # validate inputs
        assert isinstance(slope, (int, float)), "Slope must be a number"
        assert slope > 0, "Slope must be positive"
        assert isinstance(threshold, (int, float)), "Threshold must be a number"
        assert threshold >= 0, "Threshold must be non-negative"

        # set attributes
        self.slope : float = slope
        self.threshold : float = threshold
    
    def _eval_preference_function(self, value : float) -> float:
        # validate inputs
        assert isinstance(value, (int, float)), "Value must be a number"
        assert value >= 0, "Value must be non-negative"
        
        # return preference value
        return 1 / (1 + np.exp(-self.slope * (value - self.threshold)))
    
    def __repr__(self):
        return super().__repr__()[:-1] + f", slope={self.slope}, threshold={self.threshold})"
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'LogThresholdRequirement':
        """Create a log threshold requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'slope', 'threshold']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert dict.get("strategy").lower() == PerformancePreferenceStrategies.LOG_THRESHOLD.value, \
            f"Strategy does not match requirement definition. Must be '{PerformancePreferenceStrategies.LOG_THRESHOLD.value}'"
        
        # unpack dictionary
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        slope = dict.get("slope")
        threshold = dict.get("threshold")
        id = dict.get("id", None) 

        # initiate requirement
        return cls(attribute, slope, threshold, id)

class ExpDecayRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 decay_rate : float, 
                 id = None
                ):
        """
        ### Exponential Decay Requirement

        Initializes a requirement that uses an exponential decay preference function.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "data collected", "observations made").
        - :`decay_rate`: The rate at which preference decays (higher values lead to quicker decay). Must be non-negative. 
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """

        # initiate parent class
        super().__init__(attribute, PerformancePreferenceStrategies.EXP_DECAY.value, id)
        
        # validate inputs
        assert isinstance(decay_rate, (int, float)), "Decay rate must be a number"
        assert decay_rate >= 0, "Decay rate must be non-negative"
        
        # set attributes
        self.decay_rate : float = decay_rate

    def _eval_preference_function(self, value : float) -> float:
        # validate inputs
        assert isinstance(value, (int, float)), "Value must be a number"
        assert value >= 0, "Value must be non-negative"
        
        # return preference value
        return np.exp(-self.decay_rate * value)
    
    def __repr__(self):
        return super().__repr__()[:-1] + f", decay_rate={self.decay_rate})"
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'ExpDecayRequirement':
        """Create an exponential decay requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'decay_rate']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert dict.get("strategy").lower() == PerformancePreferenceStrategies.EXP_DECAY.value, \
            f"Strategy does not match requirement definition. Must be '{PerformancePreferenceStrategies.EXP_DECAY.value}'"
        
        # unpack dictionary
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        decay_rate = dict.get("decay_rate")
        id = dict.get("id", None) 

        # initiate requirement
        return cls(attribute, decay_rate, id)

class GaussianRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str,  
                 mean : float,
                 stddev : float,
                 id = None):
        """
        ### Gaussian Requirement

        Initializes a requirement that uses a Gaussian distribution as a threshold preference function.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "data collected", "observations made").
        - :`mean`: The mean value of the Gaussian function.
        - :`stddev`: The standard deviation of the Gaussian function. Must be positive.
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """
        # initiate parent class
        super().__init__(attribute, PerformancePreferenceStrategies.GAUSSIAN.value, id)

        # validate inputs
        assert isinstance(mean, (int, float)), "Average must be a number"
        assert isinstance(stddev, (int, float)), "Standard deviation must be a number"
        assert stddev > 0, "Standard deviation must be positive"

        # set attributes
        self.mean : float = mean
        self.stddev : float = stddev

    def _eval_preference_function(self, value : float) -> float:
        # validate inputs
        assert isinstance(value, (int, float)), "Number of observations must be a number"
        assert value >= 0, "Number of observations must be non-negative"

        # return preference value
        return np.exp(-0.5 * ((value - self.mean) / self.stddev) ** 2)
    
    def __repr__(self):
        return super().__repr__()[:-1] + f", mean={self.mean}, stddev={self.stddev})"

    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'GaussianRequirement':
        """Create a Gaussian requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'mean', 'stddev']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert dict.get("strategy").lower() == PerformancePreferenceStrategies.GAUSSIAN.value, \
            f"Strategy does not match requirement definition. Must be '{PerformancePreferenceStrategies.GAUSSIAN.value}'"
        
        # unpack dictionary
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        mean = dict.get("mean")
        stddev = dict.get("stddev")
        id = dict.get("id", None)

        # initiate requirement
        return cls(attribute, mean, stddev, id)

class TriangleRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 reference : float,
                 width : float, 
                 id = None):
        """
        ### Triangle Requirement

        Initializes a requirement that uses a triangular threshold preference function.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "data collected", "observations made").
        - :`reference`: The reference value at which preference is maximized.
        - :`width`: The width of the triangle base (preference drops to 0.0 at reference ± width / 2). Must be positive.
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """
        # initiate parent class
        super().__init__(attribute, PerformancePreferenceStrategies.TRIANGLE.value, id)

        # validate inputs
        assert isinstance(reference, (int, float)), "Reference must be a number"
        assert isinstance(width, (int, float)), "Width must be a number"
        assert width > 0, "Width must be positive"
        
        # set attributes
        self.reference : float = reference
        self.width : float = width
    
    def _eval_preference_function(self, value : float) -> float:
        # validate inputs
        assert isinstance(value, (int, float)), "Number of observations must be a number"
        assert value >= 0, "Number of observations must be non-negative"

        # return preference value
        return max(0.0, 1.0 - abs(value - self.reference) / (self.width / 2))

    def __repr__(self):
        return super().__repr__()[:-1] + f", reference={self.reference}, width={self.width})"
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'TriangleRequirement':
        """Create a triangle requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'reference', 'width']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert dict.get("strategy").lower() == PerformancePreferenceStrategies.TRIANGLE.value, \
            f"Strategy does not match requirement definition. Must be '{PerformancePreferenceStrategies.TRIANGLE.value}'"
        
        # unpack dictionary
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        reference = dict.get("reference")
        width = dict.get("width")
        id = dict.get("id", None)

        # initiate requirement
        return cls(attribute, reference, width, id)

class StepsRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 thresholds : List[float],
                 scores : List[float],
                 id = None):
        """
        ### Discrete Steps Requirement
        
        Initializes a requirement that uses discrete step functions for preference evaluation.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "data collected", "observations made").
        - :`thresholds`: A list of numeric thresholds defining the steps (must be in ascending order).
        - :`scores`: A list of preference scores corresponding to each threshold interval (must be in [0, 1]).
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """
        # initiate parent class
        super().__init__(attribute, PerformancePreferenceStrategies.STEPS.value, id)
        
        # validate inputs
        assert isinstance(thresholds, list), "Thresholds must be a list"
        assert isinstance(scores, list), "Scores must be a list"
        assert len(thresholds) + 1 == len(scores), \
            "Scores must have the same length as thresholds plus one"
        for threshold in thresholds:
            assert isinstance(threshold, (int, float)), "Thresholds must be numeric"
        assert all(thresholds[i] <= thresholds[i + 1] for i in range(len(thresholds) - 1)), "All values in `thresholds` must be ascending."
        for score in scores:
            assert isinstance(score, (int, float)), "Scores must be numeric"
            assert 0.0 <= score <= 1.0, "Scores must be in [0, 1]"

        # set attributes
        self.thresholds = [threshold for threshold in thresholds]
        self.scores = [score for score in scores] # assumes scores match thresholds in length and order

    def _eval_preference_function(self, value):
        # validate inputs
        assert isinstance(value, (int, float)), "Value must be numeric"

        # return preference value based on discrete levels
        for threshold,score in zip(self.thresholds,self.scores[:-1]):
            if value < threshold:
                return score
                    
        if self.thresholds[-1] <= value:
            return self.scores[-1] 

        # fallback; should not reach here
        raise ValueError("Value does not fall within any defined thresholds.")    

    def __repr__(self):
        return super().__repr__()[:-1] + f", thresholds={self.thresholds}, scores={self.scores})"
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'StepsRequirement':
        """Create a discrete levels requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'thresholds', 'scores']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        
        # unpack dictionary
        attribute = dict.get("attribute")
        thresholds = dict.get("thresholds")
        scores = dict.get("scores")
        id = dict.get("id", None)

        # initiate requirement
        return cls(attribute, thresholds, scores, id)

class IntervalInterpolationRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 thresholds : List[float],
                 scores : List[float],
                 id = None):
        """
        ### Interval Interpolation Requirement

        Initializes a requirement that uses interval-based linear interpolation for preference evaluation.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "data collected", "observations made").
        - :`thresholds`: A list of numeric thresholds defining the breakpoints (must be in ascending order).
        - :`scores`: A list of preference scores corresponding to each threshold (must be in [0, 1] and same length as thresholds).
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated
        """

        # initiate parent class
        super().__init__(attribute, PerformancePreferenceStrategies.INTERVAL_INTERP.value, id)
        
        # validate inputs
        assert isinstance(thresholds, list), "Intervals must be a list"
        assert isinstance(scores, list), "Scores must be a list"
        assert len(thresholds) == len(scores), "Intervals and scores must have the same length"
        for interval in thresholds:
            assert isinstance(interval, (int, float)), "Intervals must be numeric"
        assert all(thresholds[i] <= thresholds[i + 1] for i in range(len(thresholds) - 1)), "All values in `intervals` must be ascending."
        for score in scores:
            assert isinstance(score, (int, float)), "Scores must be numeric"
            assert 0.0 <= score <= 1.0, "Scores must be in [0, 1]"

        # set attributes
        self.thresholds = [threshold for threshold in thresholds]
        self.scores = [score for score in scores] # assumes scores match intervals in length and order

    def _eval_preference_function(self, value):
        # validate inputs
        assert isinstance(value, (int, float)), "Value must be numeric"

        # find if value is between two intervals and interpolate score
        if self.thresholds[-1] < value:
            return self.scores[-1]

        # initialize previous values
        prev_threshold,prev_score = np.NINF, self.scores[0]

        # iterate through intervals
        for threshold,score in zip(self.thresholds,self.scores):
            # check if value is within current interval
            if prev_threshold < value <= threshold:
                # do not interpolate if previous threshold is -inf
                if prev_threshold == np.NINF: return score
                
                # linear interpolation
                m = (score - prev_score) / (threshold - prev_threshold) # slope
                return prev_score + m * (value - prev_threshold)        # interpolated score
            
            # update previous values for next interval
            prev_threshold,prev_score = threshold, score
        
        # fallback; should not reach here
        raise ValueError("Value does not fall within any defined intervals.")
        
    def __repr__(self):
        return super().__repr__()[:-1] + f", thresholds={self.thresholds}, scores={self.scores})"
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'IntervalInterpolationRequirement':
        """Create a discrete intervals requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'thresholds', 'scores']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        
        # unpack dictionary
        attribute = dict.get("attribute")
        thresholds = dict.get("thresholds")
        scores = dict.get("scores")
        id = dict.get("id", None)

        # initiate requirement
        return cls(attribute, thresholds, scores, id)

"""
-----------------------------
CAPABILITY REQUIREMENT DEFINITIONS
-----------------------------
"""
class CapabilityPreferenceStrategies(Enum):
    # Explicit categorical matching
    EXPLICIT = 'explicit'

class CapabilityRequirement(MissionRequirement):
    def __init__(self, 
                 attribute : str, 
                 strategy : str,
                 id = None):
        """
        ### Capability Requirement

        Initializes a generic measurement capability requirement
        - :`attribute`: The attribute being evaluated (e.g., "instrument capability").
        - :`strategy`: Name of the preference function strategy to be used (e.g., "categorical", "exp_saturation").
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """

        # initiate parent class
        super().__init__(RequirementTypes.CAPABILITY.value, attribute, id)

        # validate inputs
        assert isinstance(strategy, str), "Preference strategy must be a string"
        assert strategy.lower() in CapabilityPreferenceStrategies._value2member_map_, f"Preference strategy must be one of {list(CapabilityPreferenceStrategies._value2member_map_.keys())}"
        
        # set attributes
        self.strategy : str = strategy.lower()

    def __repr__(self):
        """String representation of the capability requirement."""
        return f"CapabilityRequirement(strategy={CapabilityPreferenceStrategies._value2member_map_[self.strategy].name}, attribute={self.attribute})"
    
    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, float]]) -> 'MissionRequirement':
        """Create a capability requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'strategy']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"    
        # unpack dictionary
        strategy = d.get("strategy").lower()

        # initiate approriate requirement 
        if strategy == CapabilityPreferenceStrategies.EXPLICIT.value:
            return ExplicitCapabilityRequirement.from_dict(d)
        
        # Additional strategies can be implemented here
        raise NotImplementedError(f"Preference function for strategy '{strategy}' not yet supported.")

class ExplicitCapabilityRequirement(CapabilityRequirement):
    def __init__(self, 
                 attribute : str, 
                 valid_values : Union[List[str], Set[str]],
                 id = None):
        """
        ### Explicit Capability Requirement

        Initializes a requirement that accepts any value from a predefined set of valid categorical values.
        - :`attribute`: The attribute being measured (e.g., instrument type, agent type, etc.).
        - :`valid_values`: A set of valid categorical values (strings) that are acceptable.
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """

        # initiate parent class
        super().__init__(attribute, CapabilityPreferenceStrategies.EXPLICIT.value, id)

        # validate inputs
        assert isinstance(valid_values, (list, set)), "Valid values must be a list or set"
        assert all(isinstance(val, str) for val in valid_values), "All valid values must be strings"

        # set attributes
        self.valid_values : Set[str] = {val.lower() for val in valid_values}

    def _eval_preference_function(self, value : str) -> float:
        """Evaluate the preference function for a given capability value."""
        
        # validate inputs
        assert isinstance(value, str), "Input value must be a string"

        # normalize value to lowercase string
        value = str(value).lower()  

        # return preference value
        return 1.0 if value in self.valid_values else 0.0
    
    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, float]]) -> 'ExplicitCapabilityRequirement':
        """Create an explicit capability requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'valid_values']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert d.get("strategy") == CapabilityPreferenceStrategies.EXPLICIT.value, \
            f"Strategy does not match requirement definition. Must be '{CapabilityPreferenceStrategies.EXPLICIT.value}'"
        
        # unpack dictionary
        attribute = d.get("attribute")
        valid_values : list = d.get("valid_values")
        id = d.get("id", None)

        # initiate requirement
        return cls(attribute, valid_values, id)
    
"""
---------------------------------
SPATIAL REQUIREMENT DEFINITIONS
---------------------------------
"""
class SpatialPreferenceStrategies(Enum):
    SINGLE_POINT = 'single_point'
    MULTI_POINT = 'multi_point'
    GRID = 'grid'

class SpatialCoverageRequirement(MissionRequirement):
    ATTRIBUTE = 'location'

    def __init__(self, 
                 strategy : str,
                 id = None):
        """
        ### Spatial Coverage Requirement

        Initializes a generic coverage requirement.
        - :`strategy`: Name of the preference function strategy to be used (e.g., "categorical", "exp_saturation").
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """

        # initiate parent class
        super().__init__(RequirementTypes.SPATIAL.value, self.ATTRIBUTE, id)

        # validate inputs
        assert isinstance(strategy, str), "Preference strategy must be a string"
        assert strategy.lower() in SpatialPreferenceStrategies._value2member_map_, f"Preference strategy must be one of {list(SpatialPreferenceStrategies._value2member_map_.keys())}"

        # set attributes
        self.strategy : str = strategy.lower()

    def haversine_np(self, lat1 : float, lon1 : float, lat2 : float, lon2 : float) -> float:
        """
        Calculate the great circle distance between two points on the earth in [km] (specified in decimal degrees)
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Calculate angular difference in radians
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        # Haversine formula
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

        # Calculate the arc distance
        c = 2 * np.arcsin(np.sqrt(a))

        # Return great circle distance in kilometers
        return 6378.137 * c

    def __repr__(self):
        """String representation of the coverage requirement."""
        return f"SpatialRequirement(strategy={SpatialPreferenceStrategies._value2member_map_[self.strategy].name})"
      
    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, float]]) -> 'MissionRequirement':
        """Create a spatial requirement from a dictionary."""
        
        # validate input dictionary
        required_keys = ['req_type', 'strategy']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"    
        # unpack dictionary
        strategy = d.get("strategy").lower()

        # initiate approriate requirement 
        if strategy == SpatialPreferenceStrategies.SINGLE_POINT.value:
            return SinglePointSpatialRequirement.from_dict(d)
        elif strategy == SpatialPreferenceStrategies.MULTI_POINT.value:
            return MultiPointSpatialRequirement.from_dict(d)
        elif strategy == SpatialPreferenceStrategies.GRID.value:
            return GridSpatialRequirement.from_dict(d)
        
        # Additional strategies can be implemented here
        raise NotImplementedError(f"Preference function for strategy '{strategy}' not yet supported.")
    
class SinglePointSpatialRequirement(SpatialCoverageRequirement):
    def __init__(self, 
                 target : Union[Tuple, list],
                 distance_threshold : float,
                 id = None):
        """
        ### Single Point Spatial Requirement

        Initializes a requirement that evaluates preference based on proximity to a single target point.
        - :`target_point`: A tuple representing the target location as (latitude [deg], longitude [deg], grid idx, gp idx).
        - :`distance_threshold`: The distance threshold for full preference in [km].
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """

        # initiate parent class
        super().__init__(SpatialPreferenceStrategies.SINGLE_POINT.value, id)

        # validate inputs
        if isinstance(target, list):
            if len(target) == 1:
                target = target[0]
            else:
                raise ValueError("Target must be a single tuple of (latitude, longitude, grid idx, gp idx)")
            
        assert isinstance(target, tuple) and len(target) == 4, "Target point must be a tuple of (latitude, longitude)"
        lat, lon, grid_idx, gp_idx = target
        assert isinstance(lat, (int, float)) and isinstance(lon, (int, float)), "Latitude and longitude must be numeric"
        assert isinstance(grid_idx, int) and isinstance(gp_idx, int), "Grid index and GP index must be integers"
        assert -90.0 <= lat <= 90.0, "Latitude must be in [-90, 90]"
        assert -180.0 <= lon <= 180.0, "Longitude must be in [-180, 180]"
        assert grid_idx >= 0, "Grid index must be non-negative"
        assert gp_idx >= 0, "GP index must be non-negative"
        assert isinstance(distance_threshold, (int, float)), "Distance threshold must be numeric"
        assert distance_threshold >= 0, "Distance threshold must be non-negative"

        # set attributes
        self.target : Tuple[float, float, int, int] = target
        self.distance_threshold : float = distance_threshold

    def _eval_preference_function(self, location : Union[Tuple, list]) -> float:
        """Evaluate the preference function for a given location."""
        
        # validate inputs
        if isinstance(location, list):
            if len(location) == 1:
                location = location[0]
            else:
                raise ValueError("Location must be a single tuple of (latitude, longitude, grid idx, gp idx)")

        assert isinstance(location, tuple) and len(location) == 4, "Location must be a tuple of (latitude, longitude, grid idx, gp idx)"
        lat, lon, grid_idx, gp_idx = location
        assert isinstance(lat, (int, float)) and isinstance(lon, (int, float)), "Latitude and longitude must be numeric"
        assert -90.0 <= lat <= 90.0, "Latitude must be in [-90, 90]"
        assert -180.0 <= lon <= 180.0, "Longitude must be in [-180, 180]"
        assert isinstance(grid_idx, int) and isinstance(gp_idx, int), "Grid index and GP index must be integers"
        assert grid_idx >= 0, "Grid index must be non-negative"
        assert gp_idx >= 0, "GP index must be non-negative"

        # check for exact match
        if location == self.target: return 1.0

        # calculate distance to target point
        target_lat, target_lon, _, _ = self.target
        distance = self.haversine_np(lat, lon, target_lat, target_lon)

        # return preference value based on distance threshold
        return float(distance <= self.distance_threshold)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, float]]) -> 'SinglePointSpatialRequirement':
        """Create a single point spatial requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'strategy', 'target', 'distance_threshold']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert d.get("strategy").lower() == SpatialPreferenceStrategies.SINGLE_POINT.value, \
            f"Strategy does not match requirement definition. Must be '{SpatialPreferenceStrategies.SINGLE_POINT.value}'"
        
        # unpack dictionary
        target = d.get("target")
        distance_threshold = d.get("distance_threshold")
        id = d.get("id", None)

        # initiate requirement
        return cls(target, distance_threshold, id)
    
    def __repr__(self):
        """String representation of the single point spatial requirement."""
        return super().__repr__()[:-1] + f", target={self.target[2],self.target[3]})"
    
class MultiPointSpatialRequirement(SpatialCoverageRequirement):
    def __init__(self, 
                 targets : List[Tuple[float, float, int, int]],
                 distance_threshold : float,
                 id = None):
        """
        ### Multi Point Target Spatial Requirement

        Initializes a requirement that evaluates preference based on proximity to a list of target points.
        - :`targets`: A list of tuples representing target locations as (latitude [deg], longitude [deg], grid idx, gp idx).
        - :`distance_threshold`: The distance threshold for full preference in [km].
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """

        # initiate parent class
        super().__init__(SpatialPreferenceStrategies.MULTI_POINT.value, id)

        # validate inputs
        assert isinstance(targets, list) and len(targets) > 0, "Target list must be a non-empty list of target points"
        for target in targets:
            assert isinstance(target, tuple) and len(target) == 4, "Each target point must be a tuple of (latitude, longitude, grid idx, gp idx)"
            lat, lon, grid_idx, gp_idx = target
            assert isinstance(lat, (int, float)) and isinstance(lon, (int, float)), "Latitude and longitude must be numeric"
            assert isinstance(grid_idx, int) and isinstance(gp_idx, int), "Grid index and GP index must be integers"
            assert -90.0 <= lat <= 90.0, "Latitude must be in [-90, 90]"
            assert -180.0 <= lon <= 180.0, "Longitude must be in [-180, 180]"
            assert grid_idx >= 0, "Grid index must be non-negative"
            assert gp_idx >= 0, "GP index must be non-negative"
        assert isinstance(distance_threshold, (int, float)), "Distance threshold must be numeric"
        assert distance_threshold >= 0, "Distance threshold must be non-negative"

        # set attributes
        self.targets : List[Tuple[float, float, int, int]] = targets
        self.distance_threshold : float = distance_threshold

    def _eval_preference_function(self, location : Union[Tuple, list]) -> float:
        """Evaluate the preference function for a given location."""
        
        # validate inputs
        if isinstance(location, tuple):
            if len(location) == 4:
                location = [location]
            else:
                raise ValueError("Location must be a list of tuples of (latitude, longitude, grid idx, gp idx)")

        for loc in location:
            assert isinstance(loc, tuple) and len(loc) == 4, "Location must be a tuple of (latitude, longitude, grid idx, gp idx)"
            lat, lon, grid_idx, gp_idx = loc
            assert isinstance(lat, (int, float)) and isinstance(lon, (int, float)), "Latitude and longitude must be numeric"
            assert -90.0 <= lat <= 90.0, "Latitude must be in [-90, 90]"
            assert -180.0 <= lon <= 180.0, "Longitude must be in [-180, 180]"
            assert isinstance(grid_idx, int) and isinstance(gp_idx, int), "Grid index and GP index must be integers"
            assert grid_idx >= 0, "Grid index must be non-negative"
            assert gp_idx >= 0, "GP index must be non-negative"

        # check for exact match with any target
        for target in self.targets:
            if location == target:
                return 1.0

        # calculate distances to all target points
        for target in self.targets:
            target_lat, target_lon, _, _ = target
            distance = self.haversine_np(lat, lon, target_lat, target_lon)
            if distance <= self.distance_threshold:
                return 1.0

        # return preference value based on distance threshold
        return 0.0
    
    def __repr__(self):
        return super().__repr__()[:-1] + f", num_targets={len(self.targets)})"

    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, float]]) -> 'MultiPointSpatialRequirement':
        """Create a target list spatial requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'strategy', 'targets', 'distance_threshold']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert d.get("strategy").lower() == SpatialPreferenceStrategies.MULTI_POINT.value, \
            f"Strategy does not match requirement definition. Must be '{SpatialPreferenceStrategies.MULTI_POINT.value}'"

        # unpack dictionary
        targets = d.get("targets")
        distance_threshold = d.get("distance_threshold")
        id = d.get("id", None)
        
        # initiate requirement
        return cls(targets, distance_threshold, id)

class GridSpatialRequirement(SpatialCoverageRequirement):
    # TODO load grid definitions from file or external source and evaluate accordingly

    def __init__(self, 
                 grid_name : str,
                 grid_index : int,
                 grid_size : int,
                 id = None):
        """
        ### Grid Coverage Spatial Requirement

        Initializes a requirement that evaluates preference based on coverage of specified grid cells.
        - :`grid_name`: The name of the grid (e.g., "global", "regional").
        - :`grid_index`: The index of the grid cell.
        - :`grid_size`: The size of the grid cell in degrees.
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """

        # initiate parent class
        super().__init__(SpatialPreferenceStrategies.GRID.value, id)

        # validate inputs
        assert isinstance(grid_name, str), "Grid name must be a string"
        assert isinstance(grid_index, int) and grid_index >= 0, "Grid index must be a non-negative integer"
        assert isinstance(grid_size, int) and grid_size > 0, "Grid size must be a positive integer"
        
        # set attributes
        self.grid_name : str = grid_name
        self.grid_index : int = grid_index
        self.grid_size : int = grid_size

    def _eval_preference_function(self, location : Union[Tuple, list]) -> float:
        """Evaluate the preference function for a given location."""
        
        # validate inputs
        if isinstance(location, tuple):
            if len(location) == 4:
                location = [location]
            else:
                raise ValueError("Location must be a list of tuples of (latitude, longitude, grid idx, gp idx)")

        for loc in location:
            assert isinstance(loc, tuple) and len(loc) == 4, "Locations must be a tuple of (latitude, longitude, grid idx, gp idx)"
            lat, lon, grid_idx, gp_idx = loc
            assert isinstance(lat, (int, float)) and isinstance(lon, (int, float)), "Latitude and longitude must be numeric"
            assert -90.0 <= lat <= 90.0, "Latitude must be in [-90, 90]"
            assert -180.0 <= lon <= 180.0, "Longitude must be in [-180, 180]"
            assert isinstance(grid_idx, int) and isinstance(gp_idx, int), "Grid index and GP index must be integers"
            assert grid_idx >= 0, "Grid index must be non-negative"
            assert gp_idx >= 0, "GP index must be non-negative"

            # return preference value based on grid index match
            return 1.0 if grid_idx == self.grid_index and gp_idx < self.grid_size else 0.0        
        
        return 0.0
    
    def __repr__(self):
        return super().__repr__()[:-1] + f", grid_name={self.grid_name}, grid_index={self.grid_index}, grid_size={self.grid_size})"

    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, float]]) -> 'GridSpatialRequirement':
        """Create a grid coverage spatial requirement from a dictionary."""
        
        # validate input dictionary
        required_keys = ['req_type', 'strategy', 'grid_name', 'grid_index', 'grid_size']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert d.get("strategy").lower() == SpatialPreferenceStrategies.GRID.value, \
            f"Strategy does not match requirement definition. Must be '{SpatialPreferenceStrategies.GRID.value}'"  
        
        # unpack dictionary
        grid_name = d.get("grid_name")
        grid_index = d.get("grid_index")
        grid_size = d.get("grid_size")
        id = d.get("id", None)  

        # initiate requirement
        return cls(grid_name, grid_index, grid_size, id)