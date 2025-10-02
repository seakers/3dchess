from typing import Dict, Union
import uuid

class GeophysicalEvent:
    def __init__(self,
                 event_type : str,
                 location : list,
                 t_detect : float,
                 d_exp : float,
                 severity : str,
                 t_start : float = None,
                 id : str = None
                 ):
        """ 
        ### Geophysical Event

        Initialize a geophysical event with a type, severity, start time, end time, and correlation time.
        - :`event_type`: The type of event (e.g., "algal bloom", "flood").
        - :`location`: The location of the event as a list of lat-lon-grid_index-gp_index coordinates.
        - :`t_detect`: The detection time of the event.
        - :`d_exp`: The duration of the event.
        - :`severity`: The severity of the event.
        - :`t_start`: The expected start time of the event. If not provided, it defaults to the detection time.
        - :`id`: An optional unique identifier for the event. If not provided, a new UUID is generated.
        """

        # Validate inputs
        assert isinstance(event_type, str), "Event type must be a string"
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
        assert isinstance(t_detect, (int, float)), "Detection time must be a number"
        assert t_detect >= 0, "Detection time must be non-negative"
        assert isinstance(d_exp, (int, float)), "Duration must be a number"
        assert d_exp > 0, "Duration must be greater than zero"
        assert isinstance(severity, (int, float)), "Severity must be a number"
        assert severity >= 0, "Severity must be non-negative"
        assert t_start is None or isinstance(t_start, (int, float)), "Start time must be a number or None"
        

        # Set attributes
        self.event_type : str = event_type.lower()
        self.severity : float = severity
        self.location : list = location
        self.t_detect : float = t_detect
        self.d_exp : float = d_exp
        self.t_start : float = t_detect if t_start is None else t_start
        
        # Generate a unique ID if not provided
        assert id is None or isinstance(id, str) or uuid.UUID(id), "ID must be a string, a valid UUID, or None"
        self.id = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())

    def is_active(self, t: float) -> bool:
        """Check if the event is active at time t."""
        return self.t_start <= t <= (self.t_start + self.d_exp)
    
    def is_expired(self, t: float) -> bool:
        """Check if the event is expired at time t."""
        return t > (self.t_start + self.d_exp)

    def is_future(self, t: float) -> bool:
        """Check if the event is in the future at time t."""
        return t < self.t_start
    
    def is_available(self, t: float) -> bool:
        """Check if the event is available for observation at time t."""
        return self.is_active(t) or self.is_future(t)

    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert the event to a dictionary."""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, event_dict: Dict[str, Union[str, float]]) -> 'GeophysicalEvent':
        """Create an event from a dictionary."""
        return cls(**event_dict)
    
    def __repr__(self) -> str:
        """String representation of the event."""
        return f"GeophysicalEvent({self.event_type}, severity={self.severity}, t_detect={self.t_detect}, d_exp={self.d_exp}, t_start={self.t_start}, id={self.id})"

    def __str__(self) -> str:
        """String representation of the event."""
        return f"Event: {self.event_type}, Severity: {self.severity}, Detection: {self.t_detect}, Duration: {self.d_exp}, Start: {self.t_start}, End: {self.t_start + self.d_exp}, Location: {self.location}, ID: {self.id}"
    
    def __eq__(self, other) -> bool:
        """Check if two events are equal."""
        if not isinstance(other, GeophysicalEvent):
            return False
        return self.to_dict() == other.to_dict()
    
    def __hash__(self) -> int:
        """Hash the event for use in sets and dictionaries."""
        return hash((self.event_type, self.severity, self.t_detect, self.d_exp, self.t_start, self.id))
    
    # def to_dict(self) -> Dict[str, Union[str, float]]:
    #     """Convert the event to a dictionary."""
    #     return {
    #         "event_type": self.event_type,
    #         "severity": self.severity,
    #         "location": self.location,
    #         "t_start": self.t_start,
    #         "id": self.id
    #     }
