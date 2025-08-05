from typing import Dict, Union
import uuid

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
