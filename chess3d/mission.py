
from numbers import Number
from typing import Any, Callable, Dict, List, Tuple, Union
import uuid
import numpy as np
from pyparsing import ABC, abstractmethod

def monitoring(kwargs) -> float:
    """
    ### Monitoring Utility Function

    This function calculates the utility of a monitoring observation based on the time since the last observation.
    The utility is calculated as the time since the last observation divided by the total time available for monitoring.

    - :`observation`: The current observation.
    - :`unobserved_reward_rate`: The rate at which the reward decreases for unobserved events.    - :`latest_observation`: The latest observation.
    - :`kwargs`: Additional keyword arguments (not used in this function).
    
    """
    t_img = kwargs['t_start']
    t_prev = kwargs.get('t_prev',0.0)
    unobserved_reward_rate = kwargs.get('unobserved_reward_rate', 1.0)
    n_obs = kwargs.get('n_obs', 0)
    if n_obs > 0:
        x = 1
        
    assert (t_img - t_prev) >= 0.0 # TODO fix acbba triggering this

    # calculate reward
    reward = (t_img - t_prev) * unobserved_reward_rate / 3600 
    
    # clip reward to [0, 1]
    reward = np.clip(reward, 0.0, 1.0)

    # return reward
    return reward

RO = {
    # Reobservation Strategies
    "linear_increase" : lambda kwargs : kwargs['n_obs'],
    "linear_decrease" : lambda kwargs : max((4 - kwargs['n_obs'])/4, 0),
    "decaying_increase" : lambda kwargs : np.log(kwargs['n_obs']) + 1,
    "decaying_decrease" : lambda kwargs : np.exp(1 - kwargs['n_obs']),
    "immediate_decrease" : lambda kwargs : 0.0 if kwargs['n_obs'] > 0 else 1.0,
    "no_change" : lambda _ : 1.0,
    "monitoring" : monitoring,
}

CO = {
    # Coobservation Strategies
    "no_change" : lambda _ : 1.0
}

class GeophysicalEvent:
    def __init__(self,
                 event_type : str,
                 severity : str,
                 location : list,
                 t_start : float,
                 t_end : float,
                 id : str = None
                 ):
        """ 
        ### Geophysical Event

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
            assert len(location) == 4, "Location must be a lat-lon-grid index-gp index tuple of length 4"
        elif isinstance(location, list):
            if len(location) == 4:
                # Check if all elements are numbers
                assert all(isinstance(loc, (int, float)) for loc in location), "All elements of location must be numbers"
            else:                
                # Check if all elements are tuples of length 4
                assert all(isinstance(loc, (tuple, list)) and len(loc) == 4 for loc in location), "Location must be a list of tuples or lists of length 4"
        assert isinstance(t_start, (int, float)), "Start time must be a number"
        assert isinstance(t_end, (int, float)), "End time must be a number"
        assert t_start < t_end, "Start time must be less than end time"
        
        # Set attributes
        self.event_type : str = event_type.lower()
        self.severity : float = severity
        self.location : list = location
        self.t_start : float = t_start
        self.t_end : float = t_end
        self.id = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())

    def is_active(self, t: float) -> bool:
        """Check if the event is active at time t."""
        return self.t_start <= t <= self.t_end
    
    def is_expired(self, t: float) -> bool:
        """Check if the event is expired at time t."""
        return t > self.t_end
    
    def is_future(self, t: float) -> bool:
        """Check if the event is in the future at time t."""
        return t < self.t_start

    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert the event to a dictionary."""
        return self.__dict__
    
    def from_dict(event_dict: Dict[str, Union[str, float]]) -> 'GeophysicalEvent':
        """Create an event from a dictionary."""
        return GeophysicalEvent(**event_dict)
    
    def __repr__(self) -> str:
        """String representation of the event."""
        return f"GeophysicalEvent({self.event_type}, severity={self.severity}, t_start={self.t_start}, t_end={self.t_end}, id={self.id})"
    
    def __str__(self) -> str:
        """String representation of the event."""
        return f"Event: {self.event_type}, Severity: {self.severity}, Start: {self.t_start}, End: {self.t_end}, Location: {self.location}, ID: {self.id}"
    
    def __eq__(self, other) -> bool:
        """Check if two events are equal."""
        if not isinstance(other, GeophysicalEvent):
            return False
        return self.to_dict() == other.to_dict()
    
    def __hash__(self) -> int:
        """Hash the event for use in sets and dictionaries."""
        return hash((self.event_type, self.severity, self.t_start, self.t_end, self.id))
    
    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert the event to a dictionary."""
        return {
            "event_type": self.event_type,
            "severity": self.severity,
            "location": self.location,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "id": self.id
        }

class MissionRequirement(ABC):
    CATEG = 'categorical'
    DISCRETE = 'discrete'
    CONTINUOUS = 'continuous'
    TEMPORAL = 'temporal'
    SPATIAL = 'spatial'

    def __init__(self, req_type : str, attribute: str, preference_function : callable, id : str = None):
        """
        ### Mission Requirement 
        
        Initialize a mission requirement with an attribute, thresholds, and scores.
        - :`attribute`: The attribute being measured (e.g., "temperature", "humidity").
        - :`preference_function`: maps values of perforamnce to requirement satisfaction score in [0,1].        
        """
        assert any([req_type == t for t in [self.CATEG, self.DISCRETE, self.CONTINUOUS, self.TEMPORAL, self.SPATIAL]]), \
            f"Unknown requirement type: {req_type}"

        self.req_type = req_type
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
    
    @abstractmethod
    def __repr__(self):
        """String representation of the measurement requirement."""

    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'MissionRequirement':
        """Create a measurement requirement from a dictionary."""
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        thresholds = dict.get("thresholds", [])
        scores = dict.get("scores", [])
        id = dict.get("id")

        if req_type == MissionRequirement.CATEG:
            return CategoricalRequirement(attribute, thresholds, scores, id)

        elif req_type == MissionRequirement.DISCRETE:
            return DiscreteRequirement(attribute, thresholds, scores, id)

        elif req_type == MissionRequirement.CONTINUOUS:
            return ContinuousRequirement(attribute, thresholds, scores, id)

        elif req_type == MissionRequirement.TEMPORAL:
            return TemporalRequirement(attribute, thresholds, scores, id)
        
        elif req_type == MissionRequirement.SPATIAL:
            target_type = dict.get("target_type", None)
            if target_type == SpatialRequirement.POINT:
                pass
            elif target_type == SpatialRequirement.LIST:
                targets = dict.get("targets", [])
                return TargetListSpatialRequirement(targets, id)
            elif target_type == SpatialRequirement.GRID:
                pass
            else:
                raise ValueError(f"Unknown spatial requirement type: {target_type}")

        raise ValueError(f"Unknown requirement type: {req_type}")

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
        super().__init__(self.CATEG, attribute, preference_function, id)

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
        return {
            "req_type": self.req_type,
            "attribute": self.attribute,
            "thresholds": self.thresholds,
            "scores": self.scores,
            "id": self.id
        }
    
    def __repr__(self):
        return f"CategoricalRequirement({self.attribute}, thresholds={self.thresholds}, scores={self.scores})"

class DiscreteRequirement(MissionRequirement):
    def __init__(self, attribute: str, thresholds: list, scores: list, id : str = None, **_):
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
        return {
            "req_type": self.req_type,
            "attribute": self.attribute,
            "thresholds": self.thresholds,
            "scores": self.scores,
            "id": self.id
        }
    
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
        assert all(thresholds[i] >= thresholds[i + 1] for i in range(len(thresholds) - 1)) or all(thresholds[i] <= thresholds[i + 1] for i in range(len(thresholds) - 1)), "Thresholds must be sorted."

        # Set thresholds and scores attributes
        self.thresholds = thresholds
        self.scores = scores

        # Build preference function
        preference_function = self._build_continuous_preference_function(thresholds, scores)

        # Initialize the parent class
        super().__init__(self.CONTINUOUS, attribute, preference_function, id)

    def _build_continuous_preference_function(self, thresholds: list, scores: list) -> callable:
        """Creates a piecewise-linear + exponential tail preference function."""
        def preference(x: float) -> float:
            return np.interp(x, thresholds, scores, left=scores[0], right=scores[-1])
                
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
        return {
            "req_type": self.req_type,
            "attribute": self.attribute,
            "thresholds": self.thresholds,
            "scores": self.scores,
            "id": self.id
        }

    def __repr__(self):
        return f"ContinousRequirement({self.attribute}, thresholds={self.thresholds}, scores={self.scores})"

class TemporalRequirement(ContinuousRequirement):
    def __init__(self, attribute: str, thresholds: list, scores: list, id : str = None, **kwargs):
        """
        ### Temporal Mission Requirement
        Initialize a temporal requirement with an attribute, thresholds, and scores.
        - :`attribute`: The attribute being measured (e.g., "time since last observation").
        - :`thresholds`: A list of time thresholds that define the performance levels threshold ordered [x_1=x_best, x_2,...,x_worst], e.g., [3600.0, 7200.0, 10800.0] seconds.
        - :`scores`: A list of scores corresponding to the thresholds, indicating performance ordered from highest to lowest, [u_1=u_best, u_2,...,u_worst], e.g., [1.0, 0.7, 0.2].
        """
        super().__init__(attribute, thresholds, scores, id)
        self.req_type = self.TEMPORAL

    def copy(self):
        return TemporalRequirement(self.attribute, self.thresholds, self.scores, self.id)

    def __repr__(self):
        return f"TemporalRequirement({self.attribute}, thresholds={self.thresholds}, scores={self.scores})"

class SpatialRequirement(MissionRequirement):
    POINT = 'point'
    LIST = 'list'
    GRID = 'grid'

    def __init__(self, target_type : str, distance_threshold: float = 1, id = None):
        """
        ### Spatial Mission Requirement
        Initialize a spatial requirement with a target type and a distance threshold.
        - :`target_type`: The type of target location (e.g., "point", "list", "grid").
        - :`distance_threshold`: The distance threshold for the requirement in [km].
        """
        super().__init__(self.SPATIAL, 'location', self._build_spatial_preference_function(distance_threshold), id)
        assert target_type in [self.POINT, self.LIST, self.GRID], f"Invalid target type: {target_type}. Must be one of {self.POINT} or {self.GRID}."
        self.target_type = target_type
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

    def _is_location_in_targets(self, location: Tuple[float, float, int, int], distance_threshold: float) -> float:
        """Check if a location is in the targets list or within a distance threshold."""
        # Validate location input
        if not (isinstance(location, (tuple, list)) and len(location) == 4):
            raise ValueError("Location must be a tuple/list of (lat, lon, grid index, gp index)")

        # Exact match
        if tuple(location) in self.targets:
            return 1.0

        # Proximity match (lat/lon only)
        for loc in self.targets:
            if self.haversine_np(loc[0], loc[1], location[0], location[1]) <= distance_threshold:
                return 1.0

        return 0.0

    def _build_spatial_preference_function(self, distance_threshold: float) -> Callable[[Any], float]:
        """Creates a spatial preference function that returns 1.0 if a location is in targets (exact or within threshold), else 0.0."""
        def preference(location: Any) -> float:
            if isinstance(location, (list, tuple)) and len(location) == 4:
                return self._is_location_in_targets(location, distance_threshold)
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
        return {
            "req_type": self.req_type,
            "attribute": self.attribute,
            "target_type": self.target_type,
            "targets": self.targets,
            "distance_threshold": self.distance_threshold,
            "id": self.id
        }
    
    def __repr__(self):
        return f"SpatialRequirement(type={self.target_type},targets={self.targets}, distance_threshold={self.distance_threshold}, id={self.id})"

# class TargetListSpatialRequirement(SpatialRequirement):
#     def __init__(self, targets : list, id=None):
#         super().__init__(self.LIST, id)
        
#         # Validate inputs
#         assert isinstance(targets, list), "Location must be a list of coordinates"
#         assert len(targets) > 0, "Location list must not be empty"
#         assert all(isinstance(loc, (tuple, list)) and len(loc) == 4 for loc in targets), "Each location must be a tuple or list of length 4 (lat, lon, grid index, gp index)"

#         # Set attributes
#         self.targets : list[tuple] = [pos for pos in targets]

#     def __is_location_in_targets(self, location: tuple, distance_threshold : float) -> bool:
#         if isinstance(location,(list, tuple)) and len(location) == 4:
#             for tar in self.targets:
#                 # Check if the location matches one of the locations in self.targets
#                 if all([tar[i] == location[i] for i in range(4)]): return 1.0

#             # If not exact match, check haversine distance
#             if any([self.haversine_np(loc[1], loc[0], location[1], location[0]) <= distance_threshold for loc in self.targets]): return 1.0

#             return 0.0
        
#         raise ValueError("Target must be a tuple or list of length 4 (lat, lon, grid index, gp index)")

#     def _build_spatial_preference_function(self, distance_threshold: float = 1e-6) -> callable:
#         """Creates a spatial preference function based on a distance threshold."""
#         def preference(location: Any) -> float:
#             if isinstance(location,(list, tuple)) and len(location) == 4:
#                 return self.__is_location_in_targets(location, distance_threshold)
#             elif isinstance(location, list):
#                 return any(self.__is_location_in_targets(loc, distance_threshold) for loc in location)
#             else:
#                 raise ValueError("Location must be a tuple or list of length 4 (lat, lon, grid index, gp index) or a list containing such tuples or lists")
            
#         return preference

#     def __init__(self, location_type : str, location : Any, distance_threshold: float = 1e-6, id : str = None, **kwargs):
#         """
#         ### Spatial Mission Requirement
#         Initialize a spatial requirement with a location and a distance threshold.
#         - :`location_type`: The type of location (e.g., "point" or "grid").
#         - :`location`: The specific location (e.g., coordinates for a point or a grid cell).
#         - :`distance_threshold`: The distance threshold for the requirement.
#         - :`id`: An optional ID for the requirement.
#         """
#         super().__init__(self.SPATIAL, 'location', self._build_spatial_preference_function(distance_threshold), id)
        
#         assert location_type in [self.POINT, self.GRID], f"Invalid location type: {location_type}. Must be one of {self.POINT} or {self.GRID}."
#         assert isinstance(distance_threshold, (int, float)), "Distance threshold must be a number"
#         assert distance_threshold >= 0, "Distance threshold must be non-negative"
        
#         self.location_type = location_type
#         self.location = location
#         self.distance_threshold = distance_threshold

#     def _build_spatial_preference_function(self, distance_threshold: float) -> callable:
#         """Creates a spatial preference function based on a distance threshold."""
#         if location 
#             def preference(location: Any) -> float:
#                 # Calculate the distance from the location to the target
#                 distance = self._calculate_distance(location, self.target_location)
#                 # Apply the distance threshold
#                 if distance < distance_threshold:
#                     return 1.0  # Full score if within threshold
#                 else:
#                     return 0.0  # No score if outside threshold

#         return preference

class MissionObjective:
    def __init__(self, 
                 parameter: str, 
                 priority: float, 
                 requirements: list, 
                 valid_instruments : list, 
                 id : str = None):
        """ 
        ### Objective
         
        Initialize an objective with a priority, parameter, and requirements.
        - :`parameter`: The primary geophysical parameter to be measured (e.g., "Chl-A concentration").
        - :`priority`: The priority of the objective.
        - :`requirements`: A list of `MeasurementRequirement` instances that define the requirements for the objective.
        - :`valid_instruments`: A list of valid instruments that can be used to measure the parameter.
        - :`id`: An optional ID for the objective. If None, a new UUID is generated.
        """
        if all([isinstance(req, dict) for req in requirements]):
            requirements = [MissionRequirement(**req) for req in requirements]

        # Validate inputs
        assert isinstance(priority, (int, float)), "Priority must be a number"
        assert isinstance(parameter, str), "Parameter must be a string"
        assert len(requirements) > 0, "At least one requirement is needed"
        assert all(isinstance(req, MissionRequirement) for req in requirements), "All requirements must be instances of `MeasurementRequirement`"
        assert any(isinstance(req, TemporalRequirement) for req in requirements), "At least one requirement must be a `TemporalRequirement`"
        assert isinstance(id, str) or id is None, f"ID must be a string or None. is of type {type(id)}"

        # Set attributes
        self.priority : float = priority
        self.parameter : str = parameter
        self.requirements : Dict[str, MissionRequirement] = {requirement.attribute : requirement for requirement in requirements}
        self.valid_instruments = [instrument.lower() for instrument in valid_instruments] # TODO remove this and implement knoledge graph in agent
        self.id = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())

    def eval_measurement_performance(self, measurement: dict) -> float:
        """Evaluate the product of satisfaction scores given a measurement dict {attr: value}"""
        try:
            # Check if the measurement contains the required parameter
            # TODO replace with knowledge graph
            if measurement['instrument'].lower() not in self.valid_instruments:
                # measurement does not meet requirement
                return 0.0

            # Calculate the performance score for each requirement
            # If the attribute is not in the measurement, return 0
            # Otherwise, calculate the preference value using the requirement's preference function
            scores = [
                0 if attribute not in measurement
                else req.calc_preference_value(measurement[attribute])
                for attribute, req in self.requirements.items()
            ]

            # return the product of all scores
            return np.prod(scores)
        
        except Exception as e:
            raise(e)
        
    def calc_reward(self, measurement: dict) -> float:
        """Calculate the reward for the objective based on the measurement."""
        return RO[self.reobservation_strategy](measurement)
    
    def __repr__(self):
        """String representation of the objective."""
        return f"MissionObjective({self.parameter}, priority={self.priority}, requirements={self.requirements})"
    
    def __str__(self):
        """String representation of the objective."""
        return f"Objective: {self.parameter}, Priority: {self.priority}, Requirements: {self.requirements}"

    def copy(self) -> 'MissionObjective':
        """Create a copy of the objective."""
        return MissionObjective(self.parameter, self.priority, [req.copy() for req in self.requirements.values()], self.valid_instruments, self.reobservation_strategy, self.id)

    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert the objective to a dictionary."""
        out : dict = dict(self.__dict__)
        out['requirements'] = {key: req.to_dict() 
                            for key,req in self.requirements.items()}
        return out
    
    def from_dict(obj_dict: Dict[str, Union[str, float]]) -> 'MissionObjective':
        """Create an objective from a dictionary."""
        if 'event_type' in obj_dict:
            # EventDrivenObjective
            requirements = [MissionRequirement(**req) for _,req in obj_dict['requirements'].items()]
            return EventDrivenObjective(obj_dict['parameter'], obj_dict['priority'], requirements, obj_dict['event_type'], obj_dict['valid_instruments'], obj_dict['reobservation_strategy'], obj_dict['id'])
        else:
            requirements = [MissionRequirement(**req) for _,req in obj_dict['requirements'].items()]
            return MissionObjective(obj_dict['parameter'], obj_dict['priority'], requirements, obj_dict['valid_instruments'], obj_dict['reobservation_strategy'], obj_dict['id'])

    def can_perform(self, instrument: str) -> bool:
        """Check if the objective can be performed by the given instrument."""
        return instrument.lower() in self.valid_instruments

class EventDrivenObjective(MissionObjective):
    def __init__(self, 
                 parameter: str, 
                 priority: float, 
                 requirements: list, 
                 event_type: str,
                 valid_instruments : list,
                 reobservation_strategy: str,
                 synergistic_parameters: list,
                 coobservation_strategy: str,
                 t_corr : float = None,
                 id : str = None
                 ):
        """ 
        ### Event Driven Objective
         
        Initialize an event-driven objective with a priority, parameter, and requirements.
        - :`priority`: The priority of the objective.
        - :`parameter`: The primary geophysical parameter to be measured (e.g., "Chl-A concentration").
        - :`requirements`: A list of `MeasurementRequirement` instances that define the requirements for the objective.
        - :`event_type`: The type of geophysical event associated with the objective.
        - :`valid_instruments`: A list of valid instruments that can be used to measure the parameter.
        - :`reobservation_strategy`: The strategy for reobserving the event. Can be one of the following:
            - "linar_increase"
            - "linar_decrease"
            - "decaying_increase"
            - "decaying_decrease"
            - "immediate_decrease"
            - "no_change"
            - "monitoring"
        - :`id`: An optional ID for the objective. If None, a new UUID is generated.
        """
        super().__init__(parameter, priority, requirements, valid_instruments, reobservation_strategy, id)
        
        # Validate inputs
        assert isinstance(event_type, str), "Event type must be a string"
        assert isinstance(synergistic_parameters, list), "Synergistic parameters must be a list"
        assert all(isinstance(param, str) for param in synergistic_parameters), "Synergistic parameters must be strings"
        assert parameter not in synergistic_parameters, "Main parameter cannot be in list of synergistic parameters."
        assert isinstance(coobservation_strategy, str), "Coobservation strategy must be a string"
        assert coobservation_strategy in CO, f"Invalid coobservation strategy: {coobservation_strategy}. Available strategies: {list(CO.keys())}"
        
        # Set attributes
        self.event_type = event_type.lower() 
        self.synergistic_parameters = [param.lower() for param in synergistic_parameters]
        self.coobservation_strategy = coobservation_strategy.lower()
        self.t_corr = t_corr if t_corr is not None else np.Inf

    def __repr__(self):
        """String representation of the objective."""
        return f"EventDrivenObjective({self.parameter}, priority={self.priority}, requirements={self.requirements})"
    
    def __str__(self):
        """String representation of the objective."""
        return f"Event-driven Objective: {self.parameter}, Priority: {self.priority}, Requirements: {self.requirements}"

    def copy(self):
        return EventDrivenObjective(self.parameter, 
                                    self.priority, 
                                    [req.copy() for req in self.requirements.values()], 
                                    self.event_type, 
                                    self.valid_instruments, 
                                    self.reobservation_strategy, 
                                    self.synergistic_parameters,
                                    self.coobservation_strategy,
                                    self.t_corr,
                                    self.id)

    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert the objective to a dictionary."""
        out : dict = dict(self.__dict__)
        out['requirements'] = {key: req.to_dict() 
                            for key,req in self.requirements.items()}
        return out

class Mission:
    def __init__(self, name : str, objectives: list):
        # Validate inputs
        assert isinstance(name, str), "Mission name must be a string"
        assert len(objectives) > 0, "At least one objective is needed"
        assert all(isinstance(obj, MissionObjective) for obj in objectives), "All objectives must be instances of `Objective`"

        # Set attributes
        self.name : str = name.lower()
        self.objectives : list[MissionObjective] = objectives

    def evaluate_measurement(self, measurements: dict) -> float:
        """Sum weighted objective scores across all objectives"""
        return sum([obj.priority * obj.eval_measurement_performance(measurements) 
                    for obj in self.objectives
                    if not isinstance(obj, EventDrivenObjective)
                    ])
    
    def evaluate_event_measurement(self, measurements: dict) -> float:
        """Sum weighted objective scores across all objectives"""
        return sum([obj.priority * obj.eval_measurement_performance(measurements) 
                    for obj in self.objectives
                    if isinstance(obj, EventDrivenObjective)
                    ])

    def __repr__(self):
        """String representation of the mission."""
        return f"Mission({self.name}, objectives={self.objectives})"
    
    def __str__(self):
        """String representation of the mission."""
        return f"Mission: {self.name}, Objectives: {self.objectives}"
    
    def __iter__(self):
        """Iterate over the objectives."""
        return iter(self.objectives)
    
    def copy(self) -> 'Mission':
        """Create a copy of the mission."""
        return Mission(self.name, [obj.copy() for obj in self.objectives])
    
    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert the mission to a dictionary."""
        return self.__dict__