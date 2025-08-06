from numbers import Number
from typing import Any, Callable, Dict, List, Tuple, Union
import uuid
import numpy as np
from pyparsing import ABC, abstractmethod


class MissionRequirement(ABC):
    CATEGORICAL = 'categorical'
    DISCRETE = 'discrete'
    CONTINUOUS = 'continuous'
    TEMPORAL = 'temporal'
    SPATIAL = 'spatial'
    CAPABILITY = 'capability'

    def __init__(self, requirement_type : str, attribute: str, preference_function : callable, id : str = None):
        """
        ### Mission Requirement 
        
        Initialize a mission requirement with an attribute, thresholds, and scores.
        - :`requirement_type`: The type of requirement (e.g., "categorical", "discrete", "continuous", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "temperature", "humidity").
        - :`preference_function`: maps values of perforamnce to requirement satisfaction score in [0,1].        
        """

        # Validate inputs
        assert any([requirement_type == t for t in [self.CATEGORICAL, self.DISCRETE, self.CONTINUOUS, self.TEMPORAL, self.SPATIAL, self.CAPABILITY]]), \
            f"Unknown requirement type: {requirement_type}"
        assert isinstance(attribute, str), "Attribute must be a string"
        assert callable(preference_function), "Preference function must be callable"
        
        # Set attributes
        self.requirement_type = requirement_type
        self.attribute : str = attribute.lower()
        self.preference_function : callable = preference_function
        self.id = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())

    def calc_preference_value(self, value : Any) -> float:
        """Calculate the preference value for a given value."""
        # calculate preference value
        result = self.preference_function(value)

        # Validate the result
        if not isinstance(result, Number):
            raise TypeError(f"Expected a numeric return value, got {type(result).__name__}")

        if not (0.0 <= result <= 1.0):
            raise ValueError(f"Return value {result} is not in [0, 1]")

        # Return the preference value
        return result
    
    @abstractmethod
    def copy(self) -> 'MissionRequirement':
        """Create a copy of the measurement requirement."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert the measurement requirement to a dictionary."""
        return {
            "requirement_type": self.requirement_type,
            "attribute": self.attribute,
            # "preference_function": self.preference_function.__name__,
            "id": self.id
        }
    
    @abstractmethod
    def __repr__(self):
        """String representation of the measurement requirement."""

    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'MissionRequirement':
        """Create a measurement requirement from a dictionary."""
        requirement_type = dict.get("requirement_type")

        if requirement_type == MissionRequirement.CATEGORICAL:
            return CategoricalRequirement.from_dict(dict)

        elif requirement_type == MissionRequirement.DISCRETE:
            return DiscreteRequirement.from_dict(dict)

        elif requirement_type == MissionRequirement.CONTINUOUS:
            return ContinuousRequirement.from_dict(dict)

        elif requirement_type == MissionRequirement.TEMPORAL:
            return TemporalRequirement.from_dict(dict)
        
        elif requirement_type == MissionRequirement.SPATIAL:
            return SpatialRequirement.from_dict(dict)
        
        elif requirement_type == MissionRequirement.CAPABILITY:
            return CapabilityRequirement.from_dict(dict)

        raise ValueError(f"Unknown requirement type: {requirement_type}")

class CategoricalRequirement(MissionRequirement):
    def __init__(self, attribute: str, thresholds: list, scores: list, id: str = None, **_):
        """
        ### Categorical Mission Requirement
        Initialize a categorical requirement with an attribute, thresholds, and scores.
        - :`attribute`: The attribute being measured (e.g., "temperature", "humidity").
        - :`thresholds`: A list of qualitative threshold values that define the performance levels threshold ordered [x_1=x_best, x_2,...,x_worst], e.g., ["low", "medium", "high"].
        - :`scores`: A list of scores corresponding to the thresholds, indicating performance ordered from highest to lowest, [u_1=u_best, u_2,...,u_worst], e.g., [1.0, 0.7, 0.2].
        """
        # Validate inputs
        assert len(thresholds) == len(scores), "Thresholds and scores must match in length"
        assert all(isinstance(threshold, str) for threshold in thresholds), "All thresholds must be strings"
        assert all(0 <= score <= 1 for score in scores), "Scores must be between 0 and 1"

        # Convert thresholds to lowercase
        self.thresholds = [threshold.lower() for threshold in thresholds]
        
        # Set scores attributes
        self.scores = scores

        # Build preference function
        preference_function = self._build_categorical_preference_function(thresholds, scores)

        # Initialize the parent class
        super().__init__(self.CATEGORICAL, attribute, preference_function, id)

    def _build_categorical_preference_function(self, thresholds: list, scores: list) -> callable:
        """Creates a categorical preference function."""
        def preference(value: str) -> float:
            value = value.lower()
            if value not in thresholds:
                return 0.0
            
            index = thresholds.index(value)
            return scores[index]
        
        return preference

    def copy(self) -> 'MissionRequirement':
        """Create a copy of the measurement requirement."""
        return CategoricalRequirement(
            attribute=self.attribute,
            thresholds=self.thresholds,
            scores=self.scores,
            id=self.id
        )
    
    def to_dict(self):
        d = super().to_dict()
        d.update({
            "thresholds": self.thresholds,
            "scores": self.scores
        })
        return d

    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'CategoricalRequirement':
        """Create a categorical requirement from a dictionary."""
        # Validate Inputs
        assert "attribute" in dict, "Attribute must be provided for categorical requirement"
        assert "thresholds" in dict, "Thresholds must be provided for categorical requirement"
        assert "scores" in dict, "Scores must be provided for categorical requirement"
        
        # Create and return the CategoricalRequirement
        return cls(
            attribute=dict["attribute"],
            thresholds=dict["thresholds"],
            scores=dict["scores"],
            id=dict.get("id")
        )

    
    def __repr__(self):
        return f"CategoricalRequirement({self.attribute}, thresholds={self.thresholds}, scores={self.scores})"

class DiscreteRequirement(MissionRequirement):
    def __init__(self, attribute: str, thresholds: list, scores: list, id : str = None, **kwargs):
        """
        ### Discrete Value Mission Requirement
        Initialize a discrete requirement with an attribute, thresholds, and scores.
        - :`attribute`: The attribute being measured (e.g., "temperature", "humidity").
        - :`thresholds`: A list of discrete threshold values that define the performance levels threshold ordered [x_1=x_best, x_2,...,x_worst], e.g., [10, 30, 100].
        - :`scores`: A list of scores corresponding to the thresholds, indicating performance ordered from highest to lowest, [u_1=u_best, u_2,...,u_worst], e.g., [1.0, 0.7, 0.2].
        """
        # Validate inputs
        assert all(isinstance(threshold, (int, float)) for threshold in thresholds), "All thresholds must be numbers"
        assert all(isinstance(score, (int, float)) for score in scores), "All scores must be numbers"
        assert len(thresholds) > 0, "At least one threshold is needed"
        assert len(scores) > 0, "At least one preference score value is needed"
        assert len(thresholds) == len(scores), "Thresholds and scores must match in length"
        assert all(0 <= score <= 1 for score in scores), "Scores must be values between 0 and 1"
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), "Scores must be sorted in descending order"
        assert all(thresholds[i] >= thresholds[i + 1] for i in range(len(thresholds) - 1)) or all(thresholds[i] <= thresholds[i + 1] for i in range(len(thresholds) - 1)), "Thresholds must be sorted."

        # Set thresholds and scores attributes
        self.thresholds = thresholds
        self.scores = scores

        # Build preference function
        preference_function = self._build_discrete_preference_function(thresholds, scores)

        # Initialize the parent class
        super().__init__(self.DISCRETE, attribute, preference_function, id)

    def _build_discrete_preference_function(self, thresholds: list, scores: list) -> callable:
        """Creates a discrete preference function."""
        # Check if thresholds are increasing or decreasing
        increasing : bool = all(thresholds[i] <= thresholds[i + 1] for i in range(len(thresholds) - 1)) if thresholds else True

        # Create preference function based on increasing or decreasing thresholds
        if increasing:
            def preference(value: float) -> float:
                for threshold in thresholds:
                    if value <= threshold:
                        index = thresholds.index(threshold)
                        return scores[index]
                
                return scores[-1]  # Beyond worst threshold: return worst score
        else:
            def preference(value: float) -> float:
                for threshold in thresholds:
                    if value >= threshold:
                        index = thresholds.index(threshold)
                        return scores[index]

                return scores[-1]  # Beyond worst threshold: return worst score
        
        # Return the preference function    
        return preference

    def copy(self) -> 'MissionRequirement':
        """Create a copy of the measurement requirement."""
        return DiscreteRequirement(
            attribute=self.attribute,
            thresholds=self.thresholds,
            scores=self.scores,
            id=self.id
        )
    
    def to_dict(self):
        d = super().to_dict()
        d.update({
            "thresholds": self.thresholds,
            "scores": self.scores
        })
        return d
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'DiscreteRequirement':
        """Create a discrete requirement from a dictionary."""
        # Validate Inputs
        assert "attribute" in dict, "Attribute must be provided for discrete requirement"
        assert "thresholds" in dict, "Thresholds must be provided for discrete requirement"
        assert "scores" in dict, "Scores must be provided for discrete requirement"
        # Create and return the DiscreteRequirement
        return cls(
            attribute=dict["attribute"],
            thresholds=dict["thresholds"],
            scores=dict["scores"],
            id=dict.get("id")
        )

    def __repr__(self):
        return f"DiscreteRequirement({self.attribute}, thresholds={self.thresholds}, scores={self.scores})"

class ContinuousRequirement(MissionRequirement):
    def __init__(self, attribute: str, thresholds: list, scores: list, id : str = None, **kwargs):
        """
        ### Continuous Value Mission Requirement
        Initialize a continuous requirement with an attribute, thresholds, and scores.
        - :`attribute`: The attribute being measured (e.g., "temperature", "humidity").
        - :`thresholds`: A list of continuous threshold values that define the performance levels threshold ordered [x_1=x_best, x_2,...,x_worst], e.g., [10.0, 30.0, 100.0].
        - :`scores`: A list of scores corresponding to the thresholds, indicating performance ordered from highest to lowest, [u_1=u_best, u_2,...,u_worst], e.g., [1.0, 0.7, 0.2].
        """
        # Validate inputs
        assert all(isinstance(threshold, (int, float)) for threshold in thresholds), "All thresholds must be numbers"
        assert all(isinstance(score, (int, float)) for score in scores), "All scores must be numbers"
        assert len(thresholds) > 0, "At least one threshold is needed"
        assert len(scores) > 0, "At least one preference score value is needed"
        assert len(thresholds) == len(scores), "Thresholds and scores must match in length"
        assert all(0 <= score <= 1 for score in scores), "Scores must be values between 0 and 1"
        if all(thresholds[i] >= thresholds[i + 1] for i in range(len(thresholds) - 1)):
            # thresholds are in descending order; reverse them
            thresholds = list(reversed(thresholds))
            scores = list(reversed(scores))
        assert all(thresholds[i] <= thresholds[i + 1] for i in range(len(thresholds) - 1)), \
                "Thresholds must be sorted."

        # Set thresholds and scores attributes
        self.thresholds = thresholds
        self.scores = scores

        # Build preference function
        preference_function = self._build_continuous_preference_function(thresholds, scores)

        # Initialize the parent class
        super().__init__(self.CONTINUOUS, attribute, preference_function, id)

    def _build_continuous_preference_function(self, thresholds: list, scores: list) -> callable:
        """Creates a continuous interpolated preference function."""

        def preference(value: float) -> float:
            return np.interp(value, thresholds, scores, left=scores[0], right=scores[-1])
        
        # Return the preference function
        return preference

    def copy(self):
        """Create a copy of the measurement requirement."""
        return ContinuousRequirement(
            attribute=self.attribute,
            thresholds=self.thresholds,
            scores=self.scores,
            id=self.id
        )
    
    def to_dict(self):
        d = super().to_dict()
        d.update({
            "thresholds": self.thresholds,
            "scores": self.scores
        })
        return d

    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'ContinuousRequirement':
        """Create a continuous requirement from a dictionary."""
        # Validate Inputs
        assert "attribute" in dict, "Attribute must be provided for continuous requirement"
        assert "thresholds" in dict, "Thresholds must be provided for continuous requirement"
        assert "scores" in dict, "Scores must be provided for continuous requirement"
        
        # Create and return the ContinuousRequirement
        return cls(
            attribute=dict["attribute"],
            thresholds=dict["thresholds"],
            scores=dict["scores"],
            id=dict.get("id")
        )

    def __repr__(self):
        return f"ContinuousRequirement({self.attribute}, thresholds={self.thresholds}, scores={self.scores})"

class TemporalRequirement(MissionRequirement):
    DURATION = 'measurement_duration'
    REVISIT = 'revisit_time'
    N_OBS = 'n_observations'

    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'TemporalRequirement':
        """Create a temporal requirement from a dictionary."""
        requirement_type = dict.get("requirement_type")
        attribute = dict.get("attribute")
        id = dict.get("id")

        if requirement_type == MissionRequirement.TEMPORAL:
            if attribute == cls.DURATION:
                assert "thresholds" in dict and "scores" in dict, "Thresholds and scores must be provided for `measurement_duration` requirement"
                thresholds = dict.get("thresholds", [])
                scores = dict.get("scores", [])
                return MeasurementDurationRequirement(thresholds, scores, id)

            elif attribute == cls.REVISIT:
                assert "thresholds" in dict and "scores" in dict, "Thresholds and scores must be provided for `revisit_time` requirement"
                thresholds = dict.get("thresholds", [])
                scores = dict.get("scores", [])
                return RevisitTemporalRequirement(thresholds, scores, id)
            
            elif attribute == cls.N_OBS:
                return ReobservationStrategyRequirement.from_dict(dict)
        
        raise ValueError(f"Unknown temporal requirement for attribute: {attribute}")

class MeasurementDurationRequirement(TemporalRequirement, ContinuousRequirement):
    def __init__(self, thresholds: list, scores: list, id: str = None, **kwargs):
        """
        ### Measurement Duration Requirement
        Initialize a measurement duration requirement with thresholds and scores.
        - :`thresholds`: A list of time thresholds that define the performance levels.
        - :`scores`: A list of scores corresponding to the thresholds.
        """
        super().__init__(TemporalRequirement.DURATION, thresholds, scores, id)
        
        # Validate inputs
        assert all(threshold >= 0 for threshold in thresholds), "All threshold values must be non-negative."

        # Set Requirement Type
        self.requirement_type = MissionRequirement.TEMPORAL
    
    def copy(self):
        return MeasurementDurationRequirement(self.thresholds, self.scores, self.id)
    
    def __repr__(self):
        return f"MeasurementDurationRequirement(thresholds={self.thresholds}, scores={self.scores}, id={self.id})"

class RevisitTemporalRequirement(TemporalRequirement, ContinuousRequirement):
    def __init__(self, thresholds: list, scores: list, id: str = None, **kwargs):
        """"
        ### Revisit Temporal Requirement
        Initialize a revisit temporal requirement with thresholds and scores.
        - :`thresholds`: A list of time thresholds that define the performance levels.
        - :`scores`: A list of scores corresponding to the thresholds.
        """
        super().__init__(TemporalRequirement.REVISIT, thresholds, scores, id)
        self.requirement_type = MissionRequirement.TEMPORAL
    
    def copy(self):
        return RevisitTemporalRequirement(self.thresholds, self.scores, self.id)
        
    def __repr__(self):
        return f"RevisitTemporalRequirement(thresholds={self.thresholds}, scores={self.scores}, id={self.id})"

class ReobservationStrategyRequirement(TemporalRequirement):
    # Reobservation Strategies
    ## No change
    NO_CHANGE = 'no_change'
    ## More obs = better    
    EXP_SATURATION = 'exp_saturation'
    LOG_THRESHOLD = 'log_threshold'
    ## Less obs = better    
    EXP_DECAY = 'exp_decay'
    ## Thresholds
    STEP_THRESHOLD = 'step_threshold'
    LINEAR_THRESHOLD = 'linear_threshold'
    GAUSSIAN_THRESHOLD = 'gaussian_threshold'
    TRIANGLE_THRESHOLD = 'triangle_threshold'

    # RO = {
    #     # Reobservation Strategies
    #     "linear_increase" : lambda n_obs : n_obs,
    #     "linear_decrease" : lambda n_obs : max((4 - n_obs)/4, 0),
    #     "decaying_increase" : lambda n_obs : np.log(n_obs) + 1,
    #     "decaying_decrease" : lambda n_obs : np.exp(1 - n_obs),
    #     "immediate_decrease" : lambda n_obs : 0.0 if n_obs > 0 else 1.0,
    #     "no_change" : lambda _ : 1.0,
    #     # "monitoring" : monitoring,
    # }

    def __init__(self, strategy : str, id : str = None, **_):
        """
        ### Reobservation Strategy Requirement
        Initialize a reobservation strategy requirement with a strategy and an ID.
        - :`id`: An optional unique identifier for the requirement.
        """

        # Validate inputs
        assert isinstance(strategy, str), "Strategy must be a string"
        assert strategy.lower() in [self.NO_CHANGE, self.EXP_SATURATION, self.LOG_THRESHOLD, self.EXP_DECAY, self.STEP_THRESHOLD, self.LINEAR_THRESHOLD, self.GAUSSIAN_THRESHOLD, self.TRIANGLE_THRESHOLD], \
            f"Unknown strategy: {strategy}. Must be one of {self.NO_CHANGE}, {self.EXP_SATURATION}, {self.LOG_THRESHOLD}, {self.EXP_DECAY}, {self.STEP_THRESHOLD}, {self.LINEAR_THRESHOLD}, {self.GAUSSIAN_THRESHOLD}, {self.TRIANGLE_THRESHOLD}."

        super().__init__(MissionRequirement.TEMPORAL, TemporalRequirement.N_OBS, self._build_reobservation_strategy(), id)
        self.strategy = strategy.lower()

    @abstractmethod
    def _build_reobservation_strategy(self) -> Callable[[Any], float]:
        """Creates a reobservation strategy preference function."""

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "strategy": self.strategy
        })
        return d

    def __repr__(self):
        return f"ReobservationStrategy(strategy={self.strategy}, id={self.id})"
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'ReobservationStrategyRequirement':
        """Create a reobservation strategy requirement from a dictionary."""
        assert "strategy" in dict, "Strategy must be provided for reobservation strategy requirement"
        strategy = dict.get("strategy")
        
        if strategy == cls.NO_CHANGE:
            return NoChangeReobservationStrategy.from_dict(dict)
        elif strategy == cls.EXP_SATURATION:
            return ExpSaturationReobservationsStrategy.from_dict(dict)
        elif strategy == cls.LOG_THRESHOLD:
            return LogThresholdReobservationsStrategy.from_dict(dict)
        elif strategy == cls.EXP_DECAY:
            return ExpDecayReobservationStrategy.from_dict(dict)
        # elif strategy == cls.STEP_THRESHOLD:
        #     TODO
        #     return StepThresholdReobservationsStrategy.from_dict(dict)
        # elif strategy == cls.LINEAR_THRESHOLD:
        #     TODO
        #     return LinearThresholdReobservationsStrategy.from_dict(dict)
        elif strategy == cls.GAUSSIAN_THRESHOLD:
            return GaussianThresholdReobservationsStrategy.from_dict(dict)
        elif strategy == cls.TRIANGLE_THRESHOLD:
            return TriangleThresholdReobservationsStrategy.from_dict(dict)

        raise ValueError(f"Unknown reobservation strategy: {strategy}")
    

class NoChangeReobservationStrategy(ReobservationStrategyRequirement):
    def __init__(self, id = None, **_):
        super().__init__(self.NO_CHANGE, id, **_)
    
    def _build_reobservation_strategy(self):
        def preference(n_obs: int) -> float:
            assert n_obs >= 0, "Number of observations must be non-negative"
            return 1.0
        return preference
    
    def copy(self):
        return NoChangeReobservationStrategy(self.id)
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'NoChangeReobservationStrategy':
        """Create a no change reobservation strategy from a dictionary."""
        # Validate Inputs
        assert "strategy" in dict and dict["strategy"] == cls.NO_CHANGE, "Strategy must be 'no_change' for NoChangeReobservationStrategy"
        
        # Get the ID if provided
        id = dict.get("id",None)

        # Create and return the NoChangeReobservationStrategy
        return cls(id=id)

class ExpSaturationReobservationsStrategy(ReobservationStrategyRequirement):
    def __init__(self, saturation_rate : float, id = None, **_):
        super().__init__(self.EXP_SATURATION, id)
        
        # Validate inputs
        assert isinstance(saturation_rate, (int, float)), "Saturation rate must be a number"
        assert saturation_rate >= 0, "Saturation rate must be non-negative"

        # Set attributes
        self.saturation_rate : float = saturation_rate
    
    def _build_reobservation_strategy(self):
        def preference(n_obs: int) -> float:
            assert n_obs >= 0, "Number of observations must be non-negative"
            return 1.0 - np.exp(-self.saturation_rate * n_obs)
        return preference
    
    def copy(self):
        return ExpSaturationReobservationsStrategy(self.saturation_rate, self.id)
    
    def to_dict(self):
        d = super().to_dict()
        d.update({
            "saturation_rate": self.saturation_rate
        })
        return d
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'ExpSaturationReobservationsStrategy':
        """Create an exponential saturation reobservation strategy from a dictionary."""
        # Validate Inputs
        assert "strategy" in dict and dict["strategy"] == cls.EXP_SATURATION, "Strategy must be 'exp_saturation' for ExpSaturationReobservationsStrategy"
        assert "saturation_rate" in dict, "Saturation rate must be provided for ExpSaturationReobservationsStrategy"
        
        # Get the saturation rate and ID if provided
        saturation_rate = dict.get("saturation_rate")
        id = dict.get("id", None)
        
        # Create and return the ExpSaturationReobservationsStrategy
        return cls(saturation_rate=saturation_rate, id=id)
    
    def __repr__(self):
        return f"ReobservationStrategy(strategy={self.strategy}, saturation_rate={self.saturation_rate}, id={self.id})"
    
class LogThresholdReobservationsStrategy(ReobservationStrategyRequirement):
    def __init__(self, threshold: float, slope: float, id = None, **_):
        super().__init__(self.LOG_THRESHOLD, id)
        
        # Validate inputs
        assert isinstance(threshold, (int, float)), "Threshold must be a number"
        assert isinstance(slope, (int, float)), "Slope must be a number"
        assert threshold >= 0, "Threshold must be non-negative"
        assert slope >= 0, "Slope must be non-negative"
        
        # Set attributes        
        self.threshold : float = threshold
        self.slope : float = slope

    def _build_reobservation_strategy(self):
        def preference(n_obs: int) -> float:
            assert n_obs >= 0, "Number of observations must be non-negative"
            return 1 / (1 + np.exp(-self.slope * (n_obs - self.threshold)))
        return preference
    
    def copy(self):
        return LogThresholdReobservationsStrategy(self.threshold, self.slope, self.id)
    
    def to_dict(self):
        d = super().to_dict()
        d.update({
            "threshold": self.threshold,
            "slope": self.slope
        })
        return d
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'LogThresholdReobservationsStrategy':
        """Create a log threshold reobservation strategy from a dictionary."""
        # Validate Inputs
        assert "strategy" in dict and dict["strategy"] == cls.LOG_THRESHOLD, "Strategy must be 'log_threshold' for LogThresholdReobservationsStrategy"
        assert "threshold" in dict, "Threshold must be provided for LogThresholdReobservationsStrategy"
        assert "slope" in dict, "Slope must be provided for LogThresholdReobservationsStrategy"
        
        # Get the threshold, slope and ID if provided
        threshold = dict.get("threshold")
        slope = dict.get("slope")
        id = dict.get("id", None)
        
        # Create and return the LogThresholdReobservationsStrategy
        return cls(threshold=threshold, slope=slope, id=id)
    
    def __repr__(self):
        return f"ReobservationStrategy(strategy={self.strategy}, threshold={self.threshold}, slope={self.slope}, id={self.id})"

class ExpDecayReobservationStrategy(ReobservationStrategyRequirement):
    def __init__(self, decay_rate: float, id = None, **_):
        super().__init__(self.EXP_DECAY, id)
        
        # Validate inputs
        assert isinstance(decay_rate, (int, float)), "Decay rate must be a number"
        assert decay_rate >= 0, "Decay rate must be non-negative"

        # Set attributes
        self.decay_rate : float = decay_rate

    def _build_reobservation_strategy(self):
        def preference(n_obs: int) -> float:
            assert n_obs >= 0, "Number of observations must be non-negative"
            return np.exp(-self.decay_rate * n_obs)
        return preference
    
    def copy(self):
        return ExpDecayReobservationStrategy(self.decay_rate, self.id)
    
    def to_dict(self):
        d = super().to_dict()
        d.update({
            "decay_rate": self.decay_rate
        })
        return d
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'ExpDecayReobservationStrategy':
        """Create an exponential decay reobservation strategy from a dictionary."""
        # Validate Inputs
        assert "strategy" in dict and dict["strategy"] == cls.EXP_DECAY, "Strategy must be 'exp_decay' for ExpDecayReobservationStrategy"
        assert "decay_rate" in dict, "Decay rate must be provided for ExpDecayReobservationStrategy"
        
        # Get the decay rate and ID if provided
        decay_rate = dict.get("decay_rate")
        id = dict.get("id", None)
        
        # Create and return the ExpDecayReobservationStrategy
        return cls(decay_rate=decay_rate, id=id)

    def __repr__(self):
        return f"ReobservationStrategy(strategy={self.strategy}, decay_rate={self.decay_rate}, id={self.id})"

class GaussianThresholdReobservationsStrategy(ReobservationStrategyRequirement):
    def __init__(self, n_target: int, stddev: float, id = None, **_):
        super().__init__(self.GAUSSIAN_THRESHOLD, id)

        # Validate inputs
        assert isinstance(n_target, (int, float)), "Target number of observations must be a number"
        assert isinstance(stddev, (int, float)), "Standard deviation must be a number"
        assert n_target >= 0, "Target number of observations must be non-negative"
        assert stddev > 0, "Standard deviation must be positive"

        self.n_target : int = n_target
        self.stddev : float = stddev

    def _build_reobservation_strategy(self):
        def preference(n_obs: int) -> float:
            assert n_obs >= 0, "Number of observations must be non-negative"
            return np.exp(-0.5 * ((n_obs - self.n_target) / self.stddev) ** 2)
        return preference
    
    def copy(self):
        return GaussianThresholdReobservationsStrategy(self.n_target, self.stddev, self.id)
    
    def to_dict(self):
        d = super().to_dict()
        d.update({
            "n_target": self.n_target,
            "stddev": self.stddev
        })
        return d
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'GaussianThresholdReobservationsStrategy':
        """Create a Gaussian threshold reobservation strategy from a dictionary."""
        # Validate Inputs
        assert "strategy" in dict and dict["strategy"] == cls.GAUSSIAN_THRESHOLD, "Strategy must be 'gaussian_threshold' for GaussianThresholdReobservationsStrategy"
        assert "n_target" in dict, "Target number of observations must be provided for GaussianThresholdReobservationsStrategy"
        assert "stddev" in dict, "Standard deviation must be provided for GaussianThresholdReobservationsStrategy"
        
        # Get the target number of observations, standard deviation and ID if provided
        n_target = dict.get("n_target")
        stddev = dict.get("stddev")
        id = dict.get("id", None)
        
        # Create and return the GaussianThresholdReobservationsStrategy
        return cls(n_target=n_target, stddev=stddev, id=id)
    
    def __repr__(self):
        return f"ReobservationStrategy(strategy={self.strategy}, mean={self.n_target}, stddev={self.stddev}, id={self.id})"

class TriangleThresholdReobservationsStrategy(ReobservationStrategyRequirement):
    def __init__(self, n_target : int, width : float, id = None, **_):
        super().__init__(self.TRIANGLE_THRESHOLD, id)

        # Validate inputs
        assert isinstance(n_target, int) and n_target >= 0, "Target number of observations must be a non-negative integer"
        assert isinstance(width, (int, float)) and width > 0, "Width must be a positive number"

        # Set attributes
        self.n_target : float = n_target
        self.width : float = width

    def _build_reobservation_strategy(self):
        def preference(n_obs: int) -> float:
            assert n_obs >= 0, "Number of observations must be non-negative"
            return max(0.0, 1.0 - abs(n_obs - self.n_target) / self.width)
        return preference
    
    def copy(self):
        return TriangleThresholdReobservationsStrategy(self.n_target, self.width, self.id)
    
    def to_dict(self):
        d = super().to_dict()
        d.update({
            "n_target": self.n_target,
            "width": self.width
        })
        return d
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'TriangleThresholdReobservationsStrategy':
        """Create a triangle threshold reobservation strategy from a dictionary."""
        # Validate Inputs
        assert "strategy" in dict and dict["strategy"] == cls.TRIANGLE_THRESHOLD, "Strategy must be 'triangle_threshold' for TriangleThresholdReobservationsStrategy"
        assert "n_target" in dict, "Target number of observations must be provided for TriangleThresholdReobservationsStrategy"
        assert "width" in dict, "Width must be provided for TriangleThresholdReobservationsStrategy"
        
        # Get the target number of observations, width and ID if provided
        n_target = dict.get("n_target")
        width = dict.get("width")
        id = dict.get("id", None)
        
        # Create and return the TriangleThresholdReobservationsStrategy
        return cls(n_target=n_target, width=width, id=id)

    def __repr__(self):
        return f"ReobservationStrategy(strategy={self.strategy}, n_target={self.n_target}, width={self.width}, id={self.id})"

class SpatialRequirement(MissionRequirement):
    POINT = 'point'
    LIST = 'list'
    GRID = 'grid'

    def __init__(self, location_type : str, distance_threshold: float = 1, id = None):
        """
        ### Spatial Mission Requirement
        Initialize a spatial requirement with a location type and a distance threshold.
        - :`location_type`: The type of target location (e.g., "point", "list", "grid").
        - :`distance_threshold`: The distance threshold for the requirement in [km].
        """
        super().__init__(self.SPATIAL, 'location', self._build_spatial_preference_function(distance_threshold), id)
        assert location_type in [self.POINT, self.LIST, self.GRID], f"Invalid location type: {location_type}. Must be one of {self.POINT}, {self.LIST}, or {self.GRID}."
        self.location_type = location_type
        self.distance_threshold = distance_threshold

    @abstractmethod
    def _build_spatial_preference_function(self, distance_threshold: float) -> callable:
        """Creates a spatial preference function based on a distance threshold."""
        raise NotImplementedError("Subclasses must implement this method")

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
        a = np.sin(dlat/2.0)**2 + np.cos(lon1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        
        # Calculate the arc distance
        c = 2 * np.arcsin(np.sqrt(a))

        # Return great circle distance in kilometers
        return 6378.137 * c
    
    def to_dict(self):
        d = super().to_dict()
        d.update({
            "location_type": self.location_type,
            "distance_threshold": self.distance_threshold
        })
        return d
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'SpatialRequirement':
        """Create a spatial requirement from a dictionary."""
        assert "location_type" in dict, "Location type must be provided for spatial requirement"
        location_type = dict.get("location_type")

        if location_type == cls.POINT:
            return PointTargetSpatialRequirement.from_dict(dict)

        elif location_type == cls.LIST:
            return TargetListSpatialRequirement.from_dict(dict)
        
        elif location_type == cls.GRID:
            return GridTargetSpatialRequirement.from_dict(dict)

        raise ValueError(f"Unknown spatial requirement type: {location_type}")
    
class PointTargetSpatialRequirement(SpatialRequirement):
    def __init__(self, target: Tuple[float, float, int, int], distance_threshold: float = 1.0, id: str = None, **kwargs):
        super().__init__(self.POINT, distance_threshold, id)
        
        # Validate inputs
        assert isinstance(target, (tuple, list)) and len(target) == 4, \
            "Target must be a tuple or list of length 4 (lat, lon, grid index, gp index)"
        
        # Set attributes
        self.target = target

    def _is_location_in_target(self, location: Tuple[float, float, int, int], distance_threshold: float) -> bool:
        if not (isinstance(location, (tuple, list)) and len(location) == 4):
            raise ValueError("Location must be a tuple/list of (lat, lon, grid index, gp index)")
        
        # Check for exact match
        if self.target == location: return True
        
        # Proximity match (lat/lon only)
        return self.haversine_np(self.target[0], self.target[1], location[0], location[1]) <= distance_threshold

    def _build_spatial_preference_function(self, distance_threshold: float) -> Callable[[Any], float]:
        """Creates a spatial preference function that returns 1.0 if a location is the target or within a distance threshold, else 0.0."""
        def preference(location: Any) -> float:
            # Validate location input
            if not (isinstance(location, (tuple, list)) and len(location) == 4):
                raise ValueError("Location must be a tuple/list of (lat, lon, grid index, gp index)")
            
            return float(self._is_location_in_target(location, distance_threshold))
        
        return preference
    
    def copy(self):
        """Create a copy of the measurement requirement."""
        return PointTargetSpatialRequirement(self.target, self.distance_threshold, self.id)
    
    def to_dict(self):
        d = super().to_dict()
        d.update({
            "target": self.target
        })
        return d
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'PointTargetSpatialRequirement':
        """Create a point target spatial requirement from a dictionary."""
        
        # Validate Inputs
        assert "target" in dict, "Target must be provided for point target spatial requirement"
        target = dict.get("target")
        
        # Validate target
        assert isinstance(target, (tuple, list)) and len(target) == 4, \
            "Target must be a tuple or list of length 4 (lat, lon, grid index, gp index)"
        target = tuple(target)
        distance_threshold = dict.get("distance_threshold", 1.0)

        # Return the PointTargetSpatialRequirement instance
        return cls(target=target, distance_threshold=distance_threshold, id=dict.get("id", None))

    def __repr__(self):
        return f"PointTargetSpatialRequirement(target={self.target}, distance_threshold={self.distance_threshold}, id={self.id})"

class TargetListSpatialRequirement(SpatialRequirement):
    def __init__(self, targets: List[Tuple[float, float, int, int]], distance_threshold: float, id=None, **kwargs):
        super().__init__(self.LIST, distance_threshold, id)

        # Validate inputs
        assert isinstance(targets, list) and len(targets) > 0, \
            "Targets must be a non-empty list"
        assert all(isinstance(t, (tuple, list)) and len(t) == 4 for t in targets), \
            "Each target must be a tuple/list of (lat, lon, grid index, gp index)"

        # Set attributes
        self.targets: List[Tuple[float, float, int, int]] = [tuple(t) for t in targets]

    def _is_location_in_targets(self, location: Tuple[float, float, int, int], distance_threshold: float) -> bool:
        """Check if a location is in the targets list or within a distance threshold."""
        # Validate location input
        if not (isinstance(location, (tuple, list)) and len(location) == 4):
            raise ValueError("Location must be a tuple/list of (lat, lon, grid index, gp index)")

        # Check for exact match
        if tuple(location) in self.targets:
            return True

        # Proximity match (lat/lon only)
        for loc in self.targets:
            if self.haversine_np(loc[0], loc[1], location[0], location[1]) <= distance_threshold:
                return True

        return False

    def _build_spatial_preference_function(self, distance_threshold: float) -> Callable[[Any], float]:
        """Creates a spatial preference function that returns 1.0 if a location is in targets (exact or within threshold), else 0.0."""
        def preference(location: Any) -> float:
            if isinstance(location, (list, tuple)) and len(location) == 4:
                return float(self._is_location_in_targets(location, distance_threshold))
            elif isinstance(location, list):
                return float(any(
                    self._is_location_in_targets(loc, distance_threshold)
                    for loc in location
                ))
            else:
                raise ValueError("Location must be a tuple/list of (lat, lon, grid index, gp index) or a list of such elements.")

        return preference

    def copy(self) -> 'MissionRequirement':
        """Create a copy of the measurement requirement."""
        return TargetListSpatialRequirement(
            targets=self.targets,
            distance_threshold=self.distance_threshold,
            id=self.id
        )
    
    def to_dict(self):
        d = super().to_dict()
        d.update({
            "targets": self.targets
        })
        return d
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'TargetListSpatialRequirement':
        """Create a target list spatial requirement from a dictionary."""
        # Validate Inputs
        assert "targets" in dict, "Targets must be provided for target list spatial requirement"
        targets = dict.get("targets")
        
        # Validate targets
        assert isinstance(targets, list) and len(targets) > 0, "Targets must be a non-empty list"
        assert all(isinstance(t, (tuple, list)) and len(t) == 4 for t in targets), \
            "Each target must be a tuple/list of (lat, lon, grid index, gp index)"
        
        distance_threshold = dict.get("distance_threshold", 1.0)

        # Return the TargetListSpatialRequirement instance
        return cls(targets=[tuple(t) for t in targets], distance_threshold=distance_threshold, id=dict.get("id", None))
    
    def __repr__(self):
        return f"SpatialRequirement(type={self.location_type},targets={self.targets}, distance_threshold={self.distance_threshold}, id={self.id})"

class GridTargetSpatialRequirement(SpatialRequirement):
    def __init__(self, grid_name: str, grid_index: int, grid_size : int = None, id: str = None, **kwargs):
        """
        ### Grid Target Spatial Requirement
        Initialize a grid target spatial requirement with a grid name, grid index, and grid size.
        - :`grid_name`: The name of the grid (e.g., "global", "regional").
        - :`grid_index`: The index of the grid cell.
        - :`grid_size`: The size of the grid cell in degrees.
        """
        super().__init__(self.GRID, id=id)
        
        # Validate inputs
        assert isinstance(grid_name, str), "Grid name must be a string"
        assert isinstance(grid_index, int) and grid_index >= 0, "Grid index must be a non-negative integer"
        assert grid_size is None or (isinstance(grid_size, (int, float)) and grid_size > 0), "Grid size must be a positive number"

        # Set attributes
        self.grid_name = grid_name
        self.grid_index = grid_index
        self.grid_size = grid_size if grid_size is not None else np.Inf

    def _build_spatial_preference_function(self, _: float) -> Callable[[Any], float]:
        """Creates a spatial preference function that checks if a location is within the grid cell."""
        def preference(location: Any) -> float:
            assert isinstance(location, (list, tuple)) and len(location) == 4, \
                "Location must be a tuple/list of (lat, lon, grid index, gp index)"
            
            *_, grid_idx, gp_idx = location

            assert isinstance(grid_idx, int) and grid_idx >= 0, "Grid index must be a non-negative integer"
            assert isinstance(gp_idx, int) and 0 <= gp_idx, "GP index must be a non-negative integer"

            # TODO add support for point outside of grid list but within tolerance distance 

            return float(
                grid_idx == self.grid_index and
                0 <= gp_idx < self.grid_size
            )
        
        return preference
    
    def copy(self) -> 'MissionRequirement':
        """Create a copy of the measurement requirement."""
        return GridTargetSpatialRequirement(
            grid_name=self.grid_name,
            grid_index=self.grid_index,
            grid_size=self.grid_size,
            id=self.id
        )
    
    def to_dict(self):
        d = super().to_dict()
        d.update({
            "grid_name": self.grid_name,
            "grid_index": self.grid_index,
            "grid_size": self.grid_size
        })
        return d
    
    @classmethod    
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'GridTargetSpatialRequirement':
        """Create a grid target spatial requirement from a dictionary."""
        # Validate Inputs
        assert "grid_name" in dict, "Grid name must be provided for grid target spatial requirement"
        assert "grid_index" in dict, "Grid index must be provided for grid target spatial requirement"
        assert "grid_size" in dict, "Grid size must be provided for grid target spatial requirement"
        
        grid_name = dict.get("grid_name")
        grid_index = dict.get("grid_index")
        grid_size = dict.get("grid_size", None)
        id = dict.get("id", None)

        # Return the GridTargetSpatialRequirement instance
        return cls(
            grid_name=grid_name,
            grid_index=grid_index,
            grid_size=grid_size,
            id=id
        )
    
    def __repr__(self):
        return f"GridTargetSpatialRequirement(grid_name={self.grid_name}, grid_index={self.grid_index}, grid_size={self.grid_size}, id={self.id})"

class CapabilityRequirement(MissionRequirement):
    def __init__(self, attribute: str, valid_values: List[str], id: str = None, **kwargs):
        """
        ### Predefined Capability Requirement
        Initialize a capability requirement with an attribute and a list of categories.
        - :`attribute`: The attribute being evaluated (e.g., "sensor_type", "platform").
        - :`valid_values`: A list of known values of the attribute that are considered to have a desired capability.

        TODO replace these requirements with a more generic capability requirement that can be used for any attribute, a Knowledge Graph, or other forms of knowledge representation.
        """

        # Validate inputs
        assert isinstance(valid_values, list), "Valid values must be a list"
        assert valid_values, "Valid values must be a non-empty list"
        if all(isinstance(cat, str) for cat in valid_values): valid_values = [cat.lower() for cat in valid_values]

        # Initialize parent class
        super().__init__(self.CAPABILITY, attribute, self._build_capability_preference_function(valid_values), id)
        
        # set attributes
        self.valid_values = valid_values

    def _build_capability_preference_function(self, valid_values: list) -> callable:
        """Creates a capability preference function."""
        def preference(value: str) -> float:
            # if the value is a string, convert it to lowercase
            value = value.lower() if isinstance(value, str) else value
            
            # only return 1.0 if the value is in the categories, else 0.0
            return 1.0 if value in valid_values else 0.0
        
        return preference
    
    def copy(self) -> 'MissionRequirement':
        """Create a copy of the measurement requirement."""
        return CapabilityRequirement(
            attribute=self.attribute,
            valid_values=self.valid_values,
            id=self.id
        )
    
    def to_dict(self):
        d = super().to_dict()
        d.update({
            "valid_values": self.valid_values
        })
        return d
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'CapabilityRequirement':
        """Create a capability requirement from a dictionary."""
        # Validate inputs
        assert "attribute" in dict, "Attribute must be provided for capability requirement"
        assert "valid_values" in dict, "Valid values must be provided for capability requirement"

        # Create and return the CapabilityRequirement
        return cls(
            attribute=dict["attribute"],
            valid_values=dict["valid_values"],
            id=dict.get("id", None)
        )
    
    def __repr__(self):
        return f"CapabilityRequirement({self.attribute}, valid_values={self.valid_values}, id={self.id})"
