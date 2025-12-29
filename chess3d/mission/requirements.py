from enum import Enum
from numbers import Number
from typing import Any, Callable, Dict, List, Tuple, Union
import uuid
import numpy as np
from pyparsing import ABC, abstractmethod

from chess3d.utils import Interval

class MissionRequirementType(Enum):
    CAPABILITY = 'capability'
    PERFORMANCE = 'performance'

class MissionRequirement(ABC):
    def __init__(self, requirement_type : str, attribute: str, id : str = None):
        """
        ### Mission Requirement 
        
        Initialize a mission requirement with an attribute, thresholds, and scores.
        - :`requirement_type`: The type of requirement (e.g., "categorical", "discrete", "continuous", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "temperature", "humidity").
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.    
        """
        # attributes
        requirement_type = requirement_type.lower()
        attribute = attribute.lower()

        # Validate inputs
        assert isinstance(requirement_type, str), "Requirement type must be a string"
        assert requirement_type in [MissionRequirementType.PERFORMANCE.value, MissionRequirementType.CAPABILITY.value], \
            f"Unknown requirement type: {requirement_type}"
        assert isinstance(attribute, str), "Attribute must be a string"
        assert isinstance(id, str) or id is None, "ID must be a string or `None`"
        
        # Set attributes
        self.requirement_type = requirement_type
        self.attribute : str = attribute.lower()
        self.id = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())

    def calc_preference_value(self, attribute : str, value : Any) -> float:
        """Evaluates the preference value for a given value."""
        # check if attribute matches
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
    def copy(self) -> 'MissionRequirement':
        """Create a copy of the measurement requirement."""
    
    @abstractmethod
    def __repr__(self):
        """String representation of the measurement requirement."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert the measurement requirement to a dictionary."""
        return {
            "requirement_type": self.requirement_type,
            "attribute": self.attribute,
            "id": self.id
        }

    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'MissionRequirement':
        """Create a measurement requirement from a dictionary."""
        # validate input dictionary 
        assert "requirement_type" in dict, "Dictionary must contain the key 'requirement_type'"
        
        # unpack dictionary
        requirement_type = dict.get("requirement_type")

        # initiate appropriate requirement
        if requirement_type == MissionRequirementType.PERFORMANCE.value:
            return PerformanceRequirement.from_dict(dict)

        elif requirement_type == MissionRequirementType.CAPABILITY.value:
            return CapabilityRequirement.from_dict(dict)

        raise ValueError(f"Unknown requirement type: {requirement_type}. Must be one of {MissionRequirement.CATEGORICAL}, {MissionRequirement.DISCRETE}, {MissionRequirement.CONTINUOUS}, {MissionRequirement.TEMPORAL}, {MissionRequirement.SPATIAL}, {MissionRequirement.CAPABILITY}.")

class PerformanceStrategies(Enum):
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
    
    # Thresholds
    STEP_THRESHOLD = 'step_threshold'
    LINEAR_THRESHOLD = 'linear_threshold'
    
    # Bounded
    GAUSSIAN_THRESHOLD = 'gaussian_threshold'
    TRIANGLE_THRESHOLD = 'triangle_threshold'
    
    # Discreet levels
    DISCRETE_LEVELS = 'discrete_levels'
    DISCRETE_INTERVALS = 'discrete_intervals'

class PerformanceRequirement(MissionRequirement):
    def __init__(self, 
                 strategy : str, 
                 attribute : str, 
                 id : str = None):        
        # initiate parent class
        super().__init__(MissionRequirementType.PERFORMANCE.value, attribute, id)
        
        # validate inputs
        assert isinstance(strategy, str), "Strategy must be a string"
        assert strategy in PerformanceStrategies._value2member_map_, f"Strategy must be one of {list(PerformanceStrategies._value2member_map_.keys())}"

        # set attributes
        self.strategy = strategy

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "strategy": self.strategy
        })
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, float]]) -> 'PerformanceRequirement':
        """Create a performance requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['requirement_type', 'attribute', 'strategy', 'id']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        
        # unpack dictionary
        strategy = d.get("strategy")

        # initiate approriate requirement 
        if strategy == PerformanceStrategies.CATEGORICAL.value:
            return CategoricalRequirement.from_dict(d)
        
        elif strategy == PerformanceStrategies.CONSTANT.value:
            return ConstantValueRequirement.from_dict(d)
        
        elif strategy == PerformanceStrategies.EXP_SATURATION.value:
            return ExpSaturationRequirement.from_dict(d)

        elif strategy == PerformanceStrategies.LOG_THRESHOLD.value:
            return LogThresholdRequirement.from_dict(d)

        elif strategy == PerformanceStrategies.EXP_DECAY.value:
            return ExpDecayRequirement.from_dict(d)

        elif strategy == PerformanceStrategies.STEP_THRESHOLD.value:
            pass

        elif strategy == PerformanceStrategies.LINEAR_THRESHOLD.value:
            pass

        elif strategy == PerformanceStrategies.GAUSSIAN_THRESHOLD.value:
            return GaussianThresholdRequirement.from_dict(d)
        
        elif strategy == PerformanceStrategies.TRIANGLE_THRESHOLD.value:
            return TriangleThresholdRequirement.from_dict(d)

        elif strategy == PerformanceStrategies.DISCRETE_LEVELS.value:
            return DiscreteLevelsRequirement.from_dict(d)
        
        elif strategy == PerformanceStrategies.DISCRETE_INTERVALS.value:
            return DiscreteIntervalsRequirement.from_dict(d)
        
        # Additional strategies can be implemented here
        raise NotImplementedError(f"Preference function for strategy '{strategy}' is not implemented.")

class CategoricalRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 preferences : Dict[str, float],
                 id = None):
        # initiate parent class
        super().__init__(PerformanceStrategies.CATEGORICAL.value, attribute, id)
        
        # validate inputs
        assert isinstance(preferences, dict), "Preferences must be a dictionary"
        for key, val in preferences.items():
            assert isinstance(key, str), "Preference keys must be strings"
            assert isinstance(val, (int, float)), "Preference values must be numeric"
            assert 0.0 <= val <= 1.0, "Preference values must be in [0, 1]"
        
        # set attributes
        self.preferences = preferences
    
    def _eval_preference_function(self, value):
        # normalize value to lowercase string
        value = str(value).lower()

        # return preference value
        return self.preferences.get(value, 0.0) # default preference is 0.0 if category not found

class ConstantValueRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 id : str = None):
        
        # initiate parent class
        super().__init__(PerformanceStrategies.CONSTANT.value, attribute, id)
    
    def _eval_preference_function(self, _ : float) -> float:
        return 1.0 # always returns maximum preference
       
    def __repr__(self):
        return f"PerformanceRequirement(strategy=CONSTANT_VALUE, attribute={self.attribute})"
    
    def copy(self) -> 'ConstantValueRequirement':
        """Create a deep copy of the constant value requirement."""
        return ConstantValueRequirement(
            self.requirement_type,
            self.attribute,
            self.id
        )
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'ConstantValueRequirement':
        """Create a constant value requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['attribute', 'id']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"

        # unpack dictionary
        attribute = dict.get("attribute")
        id = dict.get("id")

        # initiate requirement
        return cls(attribute, id)
    
class ExpSaturationRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 saturation_rate : float,
                 id = None
                ):
        # initiate parent class
        super().__init__(PerformanceStrategies.EXP_SATURATION.value, attribute, id)
        
        # validate inputs
        assert isinstance(saturation_rate, (int, float)), "Saturation rate must be a number"
        assert saturation_rate >= 0, "Saturation rate must be non-negative"

        # set attributes
        self.saturation_rate = saturation_rate

    def _eval_preference_function(self, value : float) -> float:
        # validate inputs
        assert value >= 0, "Value must be non-negative"

        # return preference value
        return 1.0 - np.exp(-self.saturation_rate * value)
    
    def __repr__(self):
        return f"PerformanceRequirement(strategy=EXP_SATURATION, attribute={self.attribute}, sat_rate={self.saturation_rate})"
    
    def copy(self) -> 'ExpSaturationRequirement':
        """Create a deep copy of the exponential saturation requirement."""
        return ExpSaturationRequirement(
            self.attribute,
            self.saturation_rate,
            self.id
        )
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'ExpSaturationRequirement':
        """Create an exponential saturation requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['attribute', 'saturation_rate', 'id']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        
        # unpack dictionary
        attribute = dict.get("attribute")
        saturation_rate = dict.get("saturation_rate")
        id = dict.get("id") 

        # initiate requirement
        return cls(attribute, saturation_rate, id)
    
    def to_dict(self):
        d = super().to_dict()
        d.update({
            "saturation_rate": self.saturation_rate
        })
        return d

class LogThresholdRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 slope : float, 
                 threshold : float, 
                 id = None
                ):
        # initiate parent class
        super().__init__(PerformanceStrategies.LOG_THRESHOLD.value, attribute, id)
        
        # validate inputs
        assert isinstance(slope, (int, float)), "Slope must be a number"
        assert slope > 0, "Slope must be positive"
        assert isinstance(threshold, (int, float)), "Threshold must be a number"
        assert threshold >= 0, "Threshold must be non-negative"

        # set attributes
        self.slope = slope
        self.threshold = threshold
    
    def _eval_preference_function(self, value : float) -> float:
        # validate inputs
        assert value >= 0, "Value must be non-negative"
        
        # return preference value
        return 1 / (1 + np.exp(-self.slope * (value - self.threshold)))
    
    def __repr__(self):
        return f"PerformanceRequirement(strategy=LOG_THRESHOLD, attribute={self.attribute}, slope={self.slope}, threshold={self.threshold})"
    
    def copy(self) -> 'LogThresholdRequirement':
        """Create a deep copy of the log threshold requirement."""
        return LogThresholdRequirement(
            self.attribute,
            self.slope,
            self.threshold,
            self.id
        )
    
    def to_dict(self):
        d = super().to_dict()
        d.update({
            "slope": self.slope,
            "threshold": self.threshold
        })
        return d
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'LogThresholdRequirement':
        """Create a log threshold requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['attribute', 'slope', 'threshold', 'id']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        
        # unpack dictionary
        attribute = dict.get("attribute")
        slope = dict.get("slope")
        threshold = dict.get("threshold")
        id = dict.get("id") 

        # initiate requirement
        return cls(attribute, slope, threshold, id)

class ExpDecayRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str,
                 decay_rate : float, 
                 id = None
                ):
        # initiate parent class
        super().__init__(PerformanceStrategies.EXP_DECAY.value, attribute, id)
        
        # validate inputs
        assert isinstance(decay_rate, (int, float)), "Decay rate must be a number"
        assert decay_rate >= 0, "Decay rate must be non-negative"
        
        # set attributes
        self.decay_rate = decay_rate

    def _eval_preference_function(self, value : float) -> float:
        # validate inputs
        assert value >= 0, "Value must be non-negative"
        
        # return preference value
        return np.exp(-self.decay_rate * value)
    
    def __repr__(self):
        return f"PerformanceRequirement(strategy=EXP_DECAY, attribute={self.attribute}, decay_rate={self.decay_rate})"
    
    def copy(self) -> 'ExpDecayRequirement':
        """Create a deep copy of the exponential decay requirement."""
        return ExpDecayRequirement(
            self.attribute,
            self.decay_rate,
            self.id
        )
    
    def to_dict(self):
        d = super().to_dict()
        d.update({
            "decay_rate": self.decay_rate
        })
        return d
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'ExpDecayRequirement':
        """Create an exponential decay requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['attribute', 'decay_rate', 'id']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        
        # unpack dictionary
        attribute = dict.get("attribute")
        decay_rate = dict.get("decay_rate")
        id = dict.get("id") 

        # initiate requirement
        return cls(attribute, decay_rate, id)

class StepThresholdRequirement(PerformanceRequirement):
    ...

class LinearThresholdRequirement(PerformanceRequirement):
    ...

class GaussianThresholdRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 mean : float = 0.0,
                 stddev : float = 1.0,
                 id = None):
        # initiate parent class
        super().__init__(PerformanceStrategies.GAUSSIAN_THRESHOLD.value, attribute, id)

        # validate inputs
        assert isinstance(mean, (int, float)), "Average must be a number"
        assert isinstance(stddev, (int, float)), "Standard deviation must be a number"
        assert mean >= 0, "Average must be non-negative"
        assert stddev > 0, "Standard deviation must be positive"

        # set attributes
        self.mean = mean
        self.stddev = stddev

    def _eval_preference_function(self, value : float) -> float:
        # validate inputs
        assert value >= 0, "Number of observations must be non-negative"

        # return preference value
        return np.exp(-0.5 * ((value - self.mean) / self.stddev) ** 2)
    
    def __repr__(self):
        return f"PerformanceRequirement(strategy=GAUSSIAN_THRESHOLD, attribute={self.attribute}, mean={self.mean}, stddev={self.stddev})"

    def copy(self) -> 'GaussianThresholdRequirement':
        """Create a deep copy of the Gaussian threshold requirement."""
        return GaussianThresholdRequirement(
            self.attribute,
            self.mean,
            self.stddev,
            self.id
        )
    
    def to_dict(self):
        d = super().to_dict()
        d.update({
            "mean": self.mean,
            "stddev": self.stddev
        })
        return d
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'GaussianThresholdRequirement':
        """Create a Gaussian threshold requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['attribute', 'mean', 'stddev', 'id']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        
        # unpack dictionary
        attribute = dict.get("attribute")
        mean = dict.get("mean")
        stddev = dict.get("stddev")
        id = dict.get("id")

        # initiate requirement
        return cls(attribute, mean, stddev, id)

class TriangleThresholdRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str,
                 reference : float,
                 width : float, 
                 id = None):
        # initiate parent class
        super().__init__(PerformanceStrategies.TRIANGLE_THRESHOLD.value, attribute, id)

        # validate inputs
        assert isinstance(reference, (int, float)), "Reference must be a number"
        assert isinstance(width, (int, float)), "Width must be a number"
        assert width > 0, "Width must be positive"
        
        # set attributes
        self.reference = reference
        self.width = width
    
    def _eval_preference_function(self, value):
        # validate inputs
        assert value >= 0, "Number of observations must be non-negative"

        # return preference value
        return max(0.0, 1.0 - abs(value - self.reference) / self.width)

    def __repr__(self):
        return f"PerformanceRequirement(strategy=TRIANGLE_THRESHOLD, attribute={self.attribute}, reference={self.reference}, width={self.width})"
    
    def copy(self) -> 'TriangleThresholdRequirement':
        """Create a deep copy of the triangle threshold requirement."""
        return TriangleThresholdRequirement(
            self.attribute,
            self.reference,
            self.width,
            self.id
        )

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "reference": self.reference,
            "width": self.width
        })
        return d
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'TriangleThresholdRequirement':
        """Create a triangle threshold requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['attribute', 'reference', 'width', 'id']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        
        # unpack dictionary
        attribute = dict.get("attribute")
        reference = dict.get("reference")
        width = dict.get("width")
        id = dict.get("id")

        # initiate requirement
        return cls(attribute, reference, width, id)

class DiscreteLevelsRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str,
                 levels : List[float],
                 scores : List[float],
                 id = None):
        # initiate parent class
        super().__init__(PerformanceStrategies.DISCRETE_LEVELS.value, attribute, id)
        
        # validate inputs
        assert isinstance(levels, list), "Levels must be a list"
        assert isinstance(scores, list), "Scores must be a list"
        assert len(levels) == len(scores), "Levels and scores must have the same length"
        for level in levels:
            assert isinstance(level, (int, float)), "Levels must be numeric"
        for score in scores:
            assert isinstance(score, (int, float)), "Scores must be numeric"
            assert 0.0 <= score <= 1.0, "Scores must be in [0, 1]"
        assert all(levels[i] <= levels[i + 1] for i in range(len(levels) - 1)), "All values in `levels` must be ascending."

        # set attributes
        self.levels = [level for level in levels]
        self.scores = [score for score in scores] # assumes scores match levels in length and order

    def _eval_preference_function(self, value):
        # validate inputs
        assert isinstance(value, (int, float)), "Value must be numeric"

        # return preference value based on discrete levels
        for level,score in zip(self.levels,self.scores):
            if value <= level:
                return score
        return self.scores[-1] 
    
    def __repr__(self):
        return f"PerformanceRequirement(strategy=DISCRETE_LEVELS, attribute={self.attribute})"
    
    def copy(self) -> 'DiscreteLevelsRequirement':
        """Create a deep copy of the discrete levels requirement."""
        return DiscreteLevelsRequirement(
            self.attribute,
            self.levels,
            self.scores,
            self.id
        )

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "levels": self.levels,
            "scores": self.scores
        })
        return d

    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'DiscreteLevelsRequirement':
        """Create a discrete levels requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['attribute', 'levels', 'scores', 'id']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        
        # unpack dictionary
        attribute = dict.get("attribute")
        levels = dict.get("levels")
        scores = dict.get("scores")
        id = dict.get("id")

        # initiate requirement
        return cls(attribute, levels, scores, id)

class DiscreteIntervalsRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str,
                 intervals : List[float],
                 scores : List[float],
                 id = None):
        # initiate parent class
        super().__init__(PerformanceStrategies.DISCRETE_INTERVALS.value, attribute, id)
        
        # validate inputs
        assert isinstance(intervals, list), "Intervals must be a list"
        assert isinstance(scores, list), "Scores must be a list"
        assert len(intervals) == len(scores), "Intervals and scores must have the same length"
        for interval in intervals:
            assert isinstance(interval, (int, float)), "Intervals must be numeric"
        for score in scores:
            assert isinstance(score, (int, float)), "Scores must be numeric"
            assert 0.0 <= score <= 1.0, "Scores must be in [0, 1]"
        assert all(intervals[i] <= intervals[i + 1] for i in range(len(intervals) - 1)), "All values in `intervals` must be ascending."

        # set attributes
        self.intervals = [interval for interval in intervals]
        self.scores = [score for score in scores] # assumes scores match intervals in length and order

    def _eval_preference_function(self, value):
        # validate inputs
        assert isinstance(value, (int, float)), "Value must be numeric"

        # find if value is between two intervals and interpolate score
        if value < self.intervals[0]:
            return self.scores[0]
        
        elif self.intervals[-1] < value:
            return self.scores[-1]
        
        for i in range(len(self.intervals) - 1):
            if self.intervals[i] <= value <= self.intervals[i + 1]:
                # linear interpolation
                m =  (self.scores[i + 1] - self.scores[i]) / (self.intervals[i + 1] - self.intervals[i])
                return self.scores[i] + m * (value - self.intervals[i])
        
        # fallback; should not reach here
        raise ValueError("Value does not fall within any defined intervals.")
        
    def __repr__(self):
        return f"PerformanceRequirement(strategy=DISCRETE_INTERVALS, attribute={self.attribute})"
    
    def copy(self) -> 'DiscreteIntervalsRequirement':
        """Create a deep copy of the discrete intervals requirement."""
        return DiscreteIntervalsRequirement(
            self.attribute,
            self.intervals,
            self.scores,
            self.id
        )

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "intervals": self.intervals,
            "scores": self.scores
        })
        return d

    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'DiscreteIntervalsRequirement':
        """Create a discrete intervals requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['attribute', 'intervals', 'scores', 'id']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        
        # unpack dictionary
        attribute = dict.get("attribute")
        intervals = dict.get("intervals")
        scores = dict.get("scores")
        id = dict.get("id")

        # initiate requirement
        return cls(attribute, intervals, scores, id)

class CapabilityRequirement(MissionRequirement):
    ...