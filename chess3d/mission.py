
from typing import Dict, Union
import uuid
import numpy as np

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
    t_prev = kwargs.get('t_prev',-1)
    t_prev = t_prev if t_prev > 0 else 0.0
    unobserved_reward_rate = kwargs.get('unobserved_reward_rate', 1.0)
        
    # assert (t - t_init) >= 0.0 # TODO fix acbba triggering this

    # calculate reward
    reward = (t_img - t_prev) * unobserved_reward_rate / 3600 
    reward = min(max(reward, 0.0), 1.0 )

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
            assert len(location) == 4, "Location must be a lat-lon-grid index-gp index tuple of length 4"
        elif isinstance(location, list):
            if len(location) == 4:
                # Check if all elements are numbers
                assert all(isinstance(loc, (int, float)) for loc in location), "All elements of location must be numbers"
            else:
                
                assert all(len(loc) == 4 for loc in location), "Location must be a lat-lon-grid index-gp index tuple of length 4"
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
    
    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert the event to a dictionary."""
        return {
            "event_type": self.event_type,
            "severity": self.severity,
            "location": self.location,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "t_corr": self.t_corr,
            "id": self.id
        }

class MeasurementRequirement:
    def __init__(self, attribute: str, thresholds: list, scores: list, id : str = None):
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
        self.attribute : str = attribute.lower()
        self.thresholds : list = thresholds
        self.scores : list[float] = scores
        self.id = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())

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
    
    def copy(self) -> 'MeasurementRequirement':
        """Create a copy of the measurement requirement."""
        return MeasurementRequirement(self.attribute, self.thresholds, self.scores, self.id)

    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert the measurement requirement to a dictionary."""
        return {
            "attribute": self.attribute,
            "thresholds": self.thresholds,
            "scores": self.scores,
            "id": self.id
        }
    
    def __repr__(self):
        return f"MeasurementRequirement({self.attribute}, thresholds={self.thresholds}, scores={self.scores})"

class MissionObjective:
    def __init__(self, 
                 parameter: str, 
                 priority: float, 
                 requirements: list, 
                 valid_instruments : list, 
                 reobservation_strategy : str = "monitoring", 
                 id : str = None):
        """ 
        ### Objective
         
        Initialize an objective with a priority, parameter, and requirements.
        - :`parameter`: The primary geophysical parameter to be measured (e.g., "Chl-A concentration").
        - :`priority`: The priority of the objective.
        - :`requirements`: A list of `MeasurementRequirement` instances that define the requirements for the objective.
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
        if all([isinstance(req, dict) for req in requirements]):
            requirements = [MeasurementRequirement(**req) for req in requirements]

        # Validate inputs
        assert isinstance(priority, (int, float)), "Priority must be a number"
        assert isinstance(parameter, str), "Parameter must be a string"
        assert len(requirements) > 0, "At least one requirement is needed"
        assert all(isinstance(req, MeasurementRequirement) for req in requirements), "All requirements must be instances of `MeasurementRequirement`"
        assert reobservation_strategy in RO, f"Invalid reobservation strategy: {reobservation_strategy}. Available strategies: {list(RO.keys())}"
        assert isinstance(id, str) or id is None, f"ID must be a string or None. is of type {type(id)}"

        # Set attributes
        self.priority : float = priority
        self.parameter : str = parameter
        self.requirements : Dict[str, MeasurementRequirement] = {requirement.attribute : requirement for requirement in requirements}
        self.valid_instruments = [instrument.lower() for instrument in valid_instruments]
        self.reobservation_strategy = reobservation_strategy
        self.id = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())

    def eval_performance(self, measurement: dict) -> float:
        """Evaluate the product of satisfaction scores given a measurement dict {attr: value}"""
        try:
            scores = []
            if measurement['instrument'].lower() not in self.valid_instruments:
                # measurement does not meet requirement
                return 0.0

            for attribute,req in self.requirements.items():
                if attribute not in measurement:
                    # measurement does not meet requirement
                    scores.append(0)
                else:
                    # calculate measurement performance score
                    scores.append(req.calc_preference_value(measurement[attribute]))
            
            # reobs_val = RO[self.reobservation_strategy](measurement)
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
            requirements = [MeasurementRequirement(**req) for _,req in obj_dict['requirements'].items()]
            return EventDrivenObjective(obj_dict['parameter'], obj_dict['priority'], requirements, obj_dict['event_type'], obj_dict['valid_instruments'], obj_dict['reobservation_strategy'], obj_dict['id'])
        else:
            requirements = [MeasurementRequirement(**req) for _,req in obj_dict['requirements'].items()]
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
                 reobservation_strategy: str = "no_change",
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
        - :`id`: An optional ID for the objective. If None, a new UUID is generated.
        """
        super().__init__(parameter, priority, requirements, valid_instruments, reobservation_strategy, id)
        
        # Validate inputs
        assert isinstance(event_type, str), "Event type must be a string"
        
        # Set attributes
        self.event_type = event_type.lower() 

    def __repr__(self):
        """String representation of the objective."""
        return f"EventDrivenObjective({self.parameter}, priority={self.priority}, requirements={self.requirements})"
    
    def __str__(self):
        """String representation of the objective."""
        return f"Event-driven Objective: {self.parameter}, Priority: {self.priority}, Requirements: {self.requirements}"

    def copy(self):
        return EventDrivenObjective(self.parameter, self.priority, [req.copy() for req in self.requirements.values()], self.event_type, self.valid_instruments, self.reobservation_strategy, self.id)

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
        return sum([obj.priority * obj.eval_performance(measurements) 
                    for obj in self.objectives
                    if not isinstance(obj, EventDrivenObjective)
                    ])
    
    def evaluate_event_measurement(self, measurements: dict) -> float:
        """Sum weighted objective scores across all objectives"""
        return sum([obj.priority * obj.eval_performance(measurements) 
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