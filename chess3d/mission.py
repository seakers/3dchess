
from typing import Dict, Union
import uuid
import numpy as np

RO = {
    # Reobservation Strategies
    "linear_increase" : lambda n_obs : n_obs,
    "linear_decrease" : lambda n_obs : max((4 - n_obs)/4, 0),
    "decaying_increase" : lambda n_obs : np.log(n_obs) + 1,
    "decaying_decrease" : lambda n_obs : np.exp(1 - n_obs),
    "immediate_decrease" : lambda n_obs : 0.0 if n_obs > 0 else 1.0,
    "no_change" : lambda _ : 1.0,
}

class GeophysicalEvent:
    def __init__(self,
                 event_type : str,
                 severity : str,
                 location : list,
                 t_start : float,
                 t_end : float,
                 t_corr : float = None,
                 id : str = None
                 ):
        """ 
        Geophysical Event

        Initialize a geophysical event with a type, severity, start time, end time, and correlation time.
        - :`event_type`: The type of event (e.g., "algal bloom", "flood").
        - :`severity`: The severity of the event.
        - :`location`: The location of the event as a list of lat-lon-alt coordinates.
        - :`t_start`: The start time of the event.
        - :`t_end`: The end time of the event.
        - :`t_corr`: The decorrelation time of the event. If None, it is set to (t_end - t_start).
        
        """

        # Validate inputs
        assert isinstance(event_type, str), "Event type must be a string"
        assert isinstance(severity, (int, float)), "Severity must be a number"
        assert isinstance(location, tuple) or isinstance(location, list), "Location must be a tuple or a list"
        if isinstance(location, tuple):
            assert len(location) == 3, "Location must be a lat-lon-alt tuple of length 3"
        elif isinstance(location, list):
            assert all(len(loc) == 3 for loc in location), "Location must be a list of lat-lon-alt tuples of length 3"
        assert isinstance(t_start, (int, float)), "Start time must be a number"
        assert isinstance(t_end, (int, float)), "End time must be a number"
        assert t_start < t_end, "Start time must be less than end time"
        assert t_corr is None or isinstance(t_corr, (int, float)), "Correlation time must be a number or None"
        assert t_corr is None or t_corr >= 0, "Correlation time must be greater than 0 or None"
        assert t_corr is None or t_corr <= (t_end - t_start), "Correlation time must be less than or equal to (t_end - t_start) or None"

        # Set attributes
        self.event_type : str = event_type.lower()
        self.severity : float = severity
        self.location : list = location
        self.t_start : float = t_start
        self.t_end : float = t_end
        self.t_corr : float = t_corr if t_corr is not None else (t_end - t_start)
        self.id = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())

    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert the event to a dictionary."""
        return self.__dict__
    
    def from_dict(event_dict: Dict[str, Union[str, float]]) -> 'GeophysicalEvent':
        """Create an event from a dictionary."""
        return GeophysicalEvent(**event_dict)
    
    def __repr__(self) -> str:
        """String representation of the event."""
        return f"GeophysicalEvent({self.event_type}, severity={self.severity}, t_start={self.t_start}, t_end={self.t_end}, t_corr={self.t_corr})"
    
    def __str__(self) -> str:
        """String representation of the event."""
        return f"Event: {self.event_type}, Severity: {self.severity}, Start: {self.t_start}, End: {self.t_end}, Correlation Time: {self.t_corr}"
    
    def __eq__(self, other) -> bool:
        """Check if two events are equal."""
        if not isinstance(other, GeophysicalEvent):
            return False
        return self.to_dict() == other.to_dict()
    
    def __hash__(self) -> int:
        """Hash the event for use in sets and dictionaries."""
        return hash((self.event_type, self.severity, self.t_start, self.t_end, self.t_corr, self.id))
    

class MeasurementRequirement:
    def __init__(self, attribute: str, thresholds: list, scores: list):
        """
        ### Measurement Requirement 
        
        Initialize a measurement requirement with an attribute, thresholds, and scores.
        - :`attribute`: The attribute being measured (e.g., "temperature", "humidity").
        - :`thresholds`: A list of threshold values that define the performance levels.
        - :`scores`: A list of scores corresponding to the thresholds, indicating performance.
                
        """
        # Validate inputs
        assert len(thresholds) == len(scores), "Thresholds and scores must match in length"
        
        # Check if scores are sorted
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), "Scores must be sorted in descending order"

        # Check if score values are between 0 and 1
        assert all(0 <= score <= 1 for score in scores), "Scores must be between 0 and 1"

        # Set attributes
        self.attribute : str = attribute
        self.thresholds : list = thresholds
        self.scores : list[float] = scores

        # Determine if thresholds are categorical or numeric
        if isinstance(thresholds[0], str): # Categorical thresholds          
            # Check if all thresholds are strings
            assert all(isinstance(threshold, str) for threshold in thresholds), "All thresholds must be strings"

            # Convert all thresholds to lowercase
            self.thresholds = [threshold.lower() for threshold in thresholds]

            # Create performance function
            self.preference_function = self._build_categorical_preference_function()

        else: # Numerical thresholds
            # Check if thresholds are sorted
            assert all(thresholds[i] <= thresholds[i + 1] for i in range(len(thresholds) - 1)), "Thresholds must be sorted"

            # Create performance function
            self.preference_function = self._build_continous_preference_function()

    def _build_categorical_preference_function(self) -> callable:
        """Creates a categorical preference function."""
        def preference(value : str) -> float:
            if value.lower() not in self.thresholds: return 0.0 
            
            index = self.thresholds.index(value.lower())
            return self.scores[index]
            
        return preference

    def _build_continous_preference_function(self) -> callable:
        """Creates a piecewise-linear + exponential tail preference function."""
        def preference(x : float) -> float:
            # Unpack thresholds and scores
            T, S = self.thresholds, self.scores
            
            # Check if x is below the first threshold
            if x <= T[0]:
                # Maximum score for values below the first threshold
                return S[0]
            
            # Check if x is between thresholds
            for i in range(1, len(T)):
                if x <= T[i]:
                    # Linear interpolation between T[i-1] and T[i]
                    slope = (S[i] - S[i - 1]) / (T[i] - T[i - 1])
                    return S[i - 1] + slope * (x - T[i - 1])
                
            # Beyond worst threshold: exponential drop-off
            return S[-1] * np.exp(-(x - T[-1]))
        
        return preference

    def calc_preference_value(self, value: float) -> float:
        return self.preference_function(value)

class Objective:
    def __init__(self, parameter: str, priority: float, requirements: list):
        """ 
        ### Objective
         
        Initialize an objective with a priority, parameter, and requirements.
        - :`parameter`: The primary geophysical parameter to be measured (e.g., "Chl-A concentration").
        - :`priority`: The priority of the objective.
        - :`requirements`: A list of `MeasurementRequirement` instances that define the requirements for the objective.
        """
        
        # Validate inputs
        assert isinstance(priority, (int, float)), "Priority must be a number"
        assert isinstance(parameter, str), "Parameter must be a string"
        assert len(requirements) > 0, "At least one requirement is needed"
        assert all(isinstance(req, MeasurementRequirement) for req in requirements), "All requirements must be instances of `MeasurementRequirement`"

        # Set attributes
        self.priority : float = priority
        self.parameter : str = parameter
        self.requirements : Dict[str, MeasurementRequirement] = {requirement.attribute : requirement for requirement in requirements}

    def eval_performance(self, measurement: dict) -> float:
        """Evaluate the product of satisfaction scores given a measurement dict {attr: value}"""
        scores = []
        for attribute,req in self.requirements.items():
            if attribute not in measurement:
                # measurement does not meet requirement
                scores.append(0)
            else:
                # calculate measurement performance score
                scores.append(req.calc_preference_value(measurement[attribute]))
        
        return np.prod(scores)

class EventDrivenObjective(Objective):
    def __init__(self, 
                 parameter: str, 
                 priority: float, 
                 requirements: list, 
                 event_type: str,
                 reobservation_strategy: str = "no_change"
                 ):
        """ 
        ### Event Driven Objective
         
        Initialize an event-driven objective with a priority, parameter, and requirements.
        - :`priority`: The priority of the objective.
        - :`parameter`: The primary geophysical parameter to be measured (e.g., "Chl-A concentration").
        - :`requirements`: A list of `MeasurementRequirement` instances that define the requirements for the objective.
        - :`event_type`: The type of geophysical event associated with the objective.
        - :`reobservation_strategy`: The strategy for reobserving the event. Can be one of the following:
            - "linar_increase"
            - "linar_decrease"
            - "decaying_increase"
            - "decaying_decrease"
            - "immediate_decrease"
            - "no_change"
        """
        super().__init__(parameter, priority, requirements)
        
        # Validate inputs
        assert isinstance(event_type, str), "Event type must be a string"
        assert reobservation_strategy in RO, f"Invalid reobservation strategy: {reobservation_strategy}. Available strategies: {list(RO.keys())}"

        # Set attributes
        self.event_type = event_type.lower() 
        self.reobservation_strategy = reobservation_strategy

    def eval_performance(self, measurement):
        return super().eval_performance(measurement) * RO[self.reobservation_strategy](measurement["n_obs"]) * self.event.severity

class Mission:
    def __init__(self, name : str, objectives: list):
        # Validate inputs
        assert isinstance(name, str), "Mission name must be a string"
        assert len(objectives) > 0, "At least one objective is needed"
        assert all(isinstance(obj, Objective) for obj in objectives), "All objectives must be instances of `Objective`"

        # Set attributes
        self.name : str = name
        self.objectives : list[Objective] = objectives

    def evaluate_measurement(self, measurements: dict) -> float:
        """Sum weighted objective scores across all objectives"""
        return sum([obj.priority * obj.eval_performance(measurements) 
                    for obj in self.objectives])

