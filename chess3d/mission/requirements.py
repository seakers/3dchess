from enum import Enum
from numbers import Number
from typing import Any, Dict, List, Union
import uuid
import numpy as np
from pyparsing import ABC, abstractmethod
        
class RequirementTypes(Enum):
    CAPABILITY = 'capability'
    TEMPORAL = 'temporal'
    SPATIAL = 'spatial'
    PERFORMANCE = 'performance'

class PreferenceStrategies(Enum):
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

"""
---------------------------------
ABSTRACT REQUIREMENT DEFINITIONS
---------------------------------
"""

class MissionRequirement(ABC):
    def __init__(self, req_type : str, attribute: str, strategy : str, id : str = None):
        """
        ### Mission Requirement 
        
        Initialize a mission requirement with a requirement type, attribute, strategy, and unique ID.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "temperature", "humidity").
        - :`strategy`: Name of the preference function strategy to be used (e.g., "categorical", "exp_saturation").
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.    
        """
        # validate argument types
        assert isinstance(req_type, str), "Requirement type must be a string"
        assert isinstance(attribute, str), "Attribute must be a string"
        assert isinstance(strategy, str), "Preference strategy must be a string"
        assert isinstance(id, str) or id is None, "ID must be a string or `None`"
        
        # validate argument values
        assert req_type.lower() in RequirementTypes._value2member_map_, f"Requirement type must be one of {list(RequirementTypes._value2member_map_.keys())}"
        assert strategy.lower() in PreferenceStrategies._value2member_map_, f"Preference strategy must be one of {list(PreferenceStrategies._value2member_map_.keys())}"
        
        # set attributes
        self.req_type : str = req_type.lower()
        self.attribute : str = attribute.lower()
        self.strategy : str = strategy.lower()
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

    def __repr__(self):
        """String representation of the measurement requirement."""
        return f"MissionRequirement(type={RequirementTypes._value2member_map_[self.req_type].name}, strategy={PreferenceStrategies._value2member_map_[self.strategy].name}, attribute={self.attribute})"
    
    def copy(self) -> 'MissionRequirement':
        """Create a copy of the measurement requirement."""
        return self.from_dict(self.to_dict())
    
    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert the measurement requirement to a dictionary."""
        return dict(self.__dict__)

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
        if strategy == PreferenceStrategies.CATEGORICAL.value:
            return CategoricalRequirement.from_dict(d)
        
        elif strategy == PreferenceStrategies.CONSTANT.value:
            return ConstantValueRequirement.from_dict(d)
        
        elif strategy == PreferenceStrategies.EXP_SATURATION.value:
            return ExpSaturationRequirement.from_dict(d)

        elif strategy == PreferenceStrategies.LOG_THRESHOLD.value:
            return LogThresholdRequirement.from_dict(d)

        elif strategy == PreferenceStrategies.EXP_DECAY.value:
            return ExpDecayRequirement.from_dict(d)

        elif strategy == PreferenceStrategies.GAUSSIAN.value:
            return GaussianRequirement.from_dict(d)
        
        elif strategy == PreferenceStrategies.TRIANGLE.value:
            return TriangleRequirement.from_dict(d)

        elif strategy == PreferenceStrategies.STEPS.value:
            return StepsRequirement.from_dict(d)
        
        elif strategy == PreferenceStrategies.INTERVAL_INTERP.value:
            return IntervalInterpolationRequirement.from_dict(d)
        
        # Additional strategies can be implemented here
        raise NotImplementedError(f"Preference function for strategy '{strategy}' not yet supported.")

class CategoricalRequirement(MissionRequirement):
    def __init__(self, 
                 req_type : str,
                 attribute : str, 
                 preferences : Dict[str, float],
                 id = None):
        """
        ### Categorical Requirement

        Initializes a requirement that assigns preference scores to categorical values.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., instrument type, agent type, etc.).
        - :`preferences`: A dictionary mapping categorical values (strings) to preference scores (floats) in the range [0, 1].
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """

        # initiate parent class
        super().__init__(req_type, attribute, PreferenceStrategies.CATEGORICAL.value, id)
        
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
        required_keys = ['req_type', 'attribute', 'strategy', 'preferences']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert dict.get("strategy").lower() == PreferenceStrategies.CATEGORICAL.value, \
            f"Strategy does not match requirement definition. Must be '{PreferenceStrategies.CATEGORICAL.value}'"
        
        # unpack dictionary
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        preferences = dict.get("preferences")
        id = dict.get("id", None)
        
        # initiate requirement
        return cls(req_type, attribute, preferences, id)

class ConstantValueRequirement(MissionRequirement):
    def __init__(self, 
                 req_type : str,
                 attribute : str,
                 value : float = 1.0,
                 id = None
                ):
        """
        ### Constant Value Requirement

        Initializes a requirement that always returns the same preference score.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured.
        - :`value`: The constant preference score to return (default is 1.0).
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """
        # initiate parent class
        super().__init__(req_type, attribute, PreferenceStrategies.CONSTANT.value, id)

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
        assert dict.get("strategy").lower() == PreferenceStrategies.CONSTANT.value, \
            f"Strategy does not match requirement definition. Must be '{PreferenceStrategies.CONSTANT.value}'"

        # unpack dictionary
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        value = dict.get("value", 1.0)  # default to 1.0 if not provided
        id = dict.get("id", None)

        # initiate requirement
        return cls(req_type, attribute, value, id)
    
class ExpSaturationRequirement(MissionRequirement):
    def __init__(self, 
                 req_type : str,
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
        super().__init__(req_type, attribute, PreferenceStrategies.EXP_SATURATION.value, id)
        
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
        assert dict.get("strategy").lower() == PreferenceStrategies.EXP_SATURATION.value, \
            f"Strategy does not match requirement definition. Must be '{PreferenceStrategies.EXP_SATURATION.value}'"
        
        # unpack dictionary
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        sat_rate = dict.get("sat_rate")
        id = dict.get("id", None) 

        # initiate requirement
        return cls(req_type, attribute, sat_rate, id)
    
class LogThresholdRequirement(MissionRequirement):
    def __init__(self, 
                 req_type : str,
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
        super().__init__(req_type, attribute, PreferenceStrategies.LOG_THRESHOLD.value, id)
        
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
        assert dict.get("strategy").lower() == PreferenceStrategies.LOG_THRESHOLD.value, \
            f"Strategy does not match requirement definition. Must be '{PreferenceStrategies.LOG_THRESHOLD.value}'"
        
        # unpack dictionary
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        slope = dict.get("slope")
        threshold = dict.get("threshold")
        id = dict.get("id", None) 

        # initiate requirement
        return cls(req_type, attribute, slope, threshold, id)

class ExpDecayRequirement(MissionRequirement):
    def __init__(self, 
                 req_type : str,
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
        super().__init__(req_type, attribute, PreferenceStrategies.EXP_DECAY.value, id)
        
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
        assert dict.get("strategy").lower() == PreferenceStrategies.EXP_DECAY.value, \
            f"Strategy does not match requirement definition. Must be '{PreferenceStrategies.EXP_DECAY.value}'"
        
        # unpack dictionary
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        decay_rate = dict.get("decay_rate")
        id = dict.get("id", None) 

        # initiate requirement
        return cls(req_type, attribute, decay_rate, id)

class GaussianRequirement(MissionRequirement):
    def __init__(self, 
                 req_type : str,
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
        super().__init__(req_type, attribute, PreferenceStrategies.GAUSSIAN.value, id)

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
        assert dict.get("strategy").lower() == PreferenceStrategies.GAUSSIAN.value, \
            f"Strategy does not match requirement definition. Must be '{PreferenceStrategies.GAUSSIAN.value}'"
        
        # unpack dictionary
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        mean = dict.get("mean")
        stddev = dict.get("stddev")
        id = dict.get("id", None)

        # initiate requirement
        return cls(req_type, attribute, mean, stddev, id)

class TriangleRequirement(MissionRequirement):
    def __init__(self, 
                 req_type : str,
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
        super().__init__(req_type, attribute, PreferenceStrategies.TRIANGLE.value, id)

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
        assert dict.get("strategy").lower() == PreferenceStrategies.TRIANGLE.value, \
            f"Strategy does not match requirement definition. Must be '{PreferenceStrategies.TRIANGLE.value}'"
        
        # unpack dictionary
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        reference = dict.get("reference")
        width = dict.get("width")
        id = dict.get("id", None)

        # initiate requirement
        return cls(req_type, attribute, reference, width, id)

class StepsRequirement(MissionRequirement):
    def __init__(self, 
                 req_type : str,
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
        super().__init__(req_type, attribute, PreferenceStrategies.STEPS.value, id)
        
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
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        thresholds = dict.get("thresholds")
        scores = dict.get("scores")
        id = dict.get("id", None)

        # initiate requirement
        return cls(req_type, attribute, thresholds, scores, id)

class IntervalInterpolationRequirement(MissionRequirement):
    def __init__(self, 
                 req_type : str,
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
        super().__init__(req_type, attribute, PreferenceStrategies.INTERVAL_INTERP.value, id)
        
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
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        thresholds = dict.get("thresholds")
        scores = dict.get("scores")
        id = dict.get("id", None)

        # initiate requirement
        return cls(req_type, attribute, thresholds, scores, id)

"""
-----------------------------
REQUIREMENT IMPLEMENTATIONS
-----------------------------
"""

# class CapabilityRequirement(CategoricalRequirement):
#     def __init__(self, attribute, valid_values : List[str], id=None):
#         """
#         ### Capability Requirement

#         Initializes a requirement that accepts any value from a predefined set of valid categorical values.
#         - :`attribute`: The attribute being measured (e.g., instrument type, agent type, etc.).
#         - :`valid_values`: A list of valid categorical values (strings) that are acceptable.
#         - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
#         """
#         # validate inputs
#         assert isinstance(valid_values, list), "Valid values must be a list"
#         assert all(isinstance(val, str) for val in valid_values), "All valid values must be strings"

#         # create preferences dictionary with all valid values assigned a preference of 1.0
#         preferences = {val.lower(): 1.0 for val in valid_values}
        
#         # initiate parent class
#         super().__init__(RequirementTypes.CAPABILITY.value, attribute, preferences, id)

#     def __repr__(self):
#         return f"CapabilityRequirement(attribute={self.attribute}, valid_values={list(self.preferences.keys())})"
    
#     @classmethod
#     def from_dict(cls, d: Dict[str, Union[str, float]]) -> 'CapabilityRequirement':
#         """Create a capability requirement from a dictionary."""

#         # validate input dictionary
#         required_keys = ['req_type', 'attribute', 'preferences']
#         assert all(key in d for key in required_keys), \
#             f"Dictionary must contain the keys: {required_keys}"
#         assert d.get("strategy") == PreferenceStrategies.CATEGORICAL.value, \
#             f"Strategy does not match requirement definition. Must be '{PreferenceStrategies.CATEGORICAL.value}'"
#         assert isinstance(d.get("preferences"), dict), "Preferences must be a dictionary"
#         assert d.get("req_type") == RequirementTypes.CAPABILITY.value, \
#             f"Requirement type does not match requirement definition. Must be '{RequirementTypes.CAPABILITY.value}'"
        
#         # unpack dictionary
#         attribute = d.get("attribute")
#         preferences : dict = d.get("preferences")
#         id = d.get("id", None)

#         # extract valid values from preferences dictionary
#         valid_values = [key for key, val in preferences.items() if val == 1.0]
        
#         # initiate requirement
#         return cls(attribute, valid_values, id)
    

# class TemporalRequirement(MissionRequirement):
#     ...
    