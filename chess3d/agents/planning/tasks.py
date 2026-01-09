from abc import ABC, abstractmethod
import uuid

from chess3d.mission.events import GeophysicalEvent
from chess3d.mission.objectives import MissionObjective, EventDrivenObjective, DefaultMissionObjective
from chess3d.mission.requirements import SpatialCoverageRequirement, SinglePointSpatialRequirement
from chess3d.utils import Interval


class GenericObservationTask(ABC):
    DEFAULT = 'default_mission_task'
    EVENT = 'event_driven_task'

    def __init__(self,
                 task_type : str,
                 parameter : str,
                 location: list,
                 availability: Interval,
                 priority : float, 
                 objective : MissionObjective = None,
                 id : str = None,
                ):
        """
        Generic observation task to be scheduled by an agent.
        - :`task_type`: The type of the task, either 'default_mission_task' or 'event_driven_task'.
        - :`parameter`: The parameter to be observed (e.g., "temperature", "humidity").
        - :`location`: Location or list of locations to be observed, each represented as a tuple of (lat[deg], lon[deg], grid index, gp index).
        - :`availability`: The time interval during which the task is available.
        - :`reward`: The reward for completing the task.
        - :`relevant_objective`: The relevant mission objective associated with the task by the agent who initialized it.
        - :`priority`: The priority of the task, which can be used to determine its importance relative to other tasks.
        - :`objective`: The objective of the mission, which defines the goal of the task.
        - :`id`: A unique identifier for the task. If not provided, a new ID will be generated.
        """

        if isinstance(location, (tuple, list)) and len(location) == 4 and all([isinstance(coordinate, (float,int)) for coordinate in location]):
            # single location provided; convert to list
            if isinstance(location, tuple):
                location = [location]
            else:
                location = [tuple(location)]

        # validate inputs
        assert isinstance(task_type, str), "Task type must be a string."
        assert task_type in [self.DEFAULT, self.EVENT], "Task type must be either 'default_mission_task' or 'event_driven_task'."
        assert isinstance(parameter, str), "Parameter must be a string."
        assert isinstance(location, list), "Locations must be a list."
        assert all([isinstance(location, tuple) for location in location]), \
            "All locations must tuples of type (lat[deg], lon[deg], grid index, gp index)."
        assert all([len(location) == 4 for location in location]), "All locations must tuples of type (lat[deg], lon[deg], grid index, gp index)."
        assert isinstance(availability, Interval), "Availability must be an Interval."
        assert availability.left >= 0.0, "Start of availability must be non-negative."
        assert isinstance(priority, (float, int)), "Priority must be a number."
        assert priority >= 0, "Priority must be non-negative."
        assert isinstance(objective, MissionObjective) or objective is None, f"If specified, objective must be a `MissionObjective`. Is of type {type(objective)}."
        assert id is None or isinstance(id, str), "ID must be a string or None."

        if objective is not None:
            # Objective specified; check objective attributes 
            assert parameter.lower() == objective.parameter, "Target parameter must match the objective's parameter."

        # Set attributes
        self.task_type : str = task_type
        self.parameter : str = parameter.lower()
        self.location : list[tuple] = location
        self.availability : Interval = availability
        self.priority : float = priority
        self.objective : MissionObjective = objective
        self.id : str = id if id is not None else self.generate_id()

    @abstractmethod
    def generate_id(self) -> str:
        """ Generate a unique identifier for the task. """
        pass

    @abstractmethod
    def copy(self) -> object:
        """ Create a deep copy of the task. """
        pass

    def is_available(self, time : float) -> bool:
        """ Check if the task is available at a given time. """
        assert time >= 0, "Time must be non-negative."
        return time in self.availability    
    
    def is_expired(self, time : float) -> bool:
        """ Check if the task is expired at a given time. """
        assert time >= 0, "Time must be non-negative."
        return self.availability.is_before(time)
    
    def to_dict(self) -> dict:
        """ Convert the task to a dictionary. """
        return {
            "task_type": self.task_type,
            "parameter": self.parameter,
            "location": [loc for loc in self.location],
            "availability": self.availability.to_dict(),
            "priority": self.priority,
            "objective" : self.objective.to_dict() if self.objective else None,
            "id": self.id,
        }
    
    @abstractmethod
    def __repr__(self):
        """ String representation of the task. """

    def __str__(self):
        return self.__repr__()

    @classmethod
    def from_dict(cls, task_dict: dict) -> 'GenericObservationTask':
        """ Create a task from a dictionary. """
        assert 'task_type' in task_dict, "Task type must be specified in the dictionary."
        task_type = task_dict['task_type']        

        if task_type == cls.DEFAULT:
            return DefaultMissionTask.from_dict(task_dict)
        
        elif task_type == cls.EVENT:
            return EventObservationTask.from_dict(task_dict)

        return ValueError(f"Unknown task type: {task_type}")
        
    def __eq__(self, other: object) -> bool:
        """ Check if two tasks are equal. """
        assert isinstance(other, GenericObservationTask), "Can only compare GenericObservationTask objects."
        return self.to_dict() == other.to_dict()

    def __hash__(self):
        return hash(self.id)

class DefaultMissionTask(GenericObservationTask):
    def __init__(self,
                 parameter : str,
                 location: list,
                 mission_duration : float,
                 priority : float = 1.0,
                 objective : MissionObjective = None,
                 id : str = None
                ):
        """
        ### Default Observation Task
        Represents a default observation task of a point location to be scheduled by an agent.
        - :`parameter`: The parameter to be observed (e.g., "temperature", "humidity").
        - :`location`: The location to be observed, represented as a tuple of (lat[deg], lon[deg], grid index, gp index).
        - :`mission_duration`: The duration of the mission in seconds.
        - :`priority`: The priority of the task, which can be used to determine its importance relative to other tasks. Is 1 by default.
        - :`objective`: The objective of the mission, which defines the goal of the task.
        - :`id`: A unique identifier for the task. If not provided, a new ID will be generated.
        """

        # validate inputs
        assert isinstance(location, tuple), "Location must be a tuple of type (lat[deg], lon[deg], grid index, gp index)."
        assert len(location) == 4, "Location must be a tuple of type (lat[deg], lon[deg], grid index, gp index)."
        assert all([isinstance(coordinate, float) or isinstance(coordinate, int) for coordinate in location]), \
            "All locations must tuples of type (lat[deg], lon[deg], grid index, gp index)."

        # initialte parent class
        super().__init__(GenericObservationTask.DEFAULT, parameter, [location], Interval(0.0, mission_duration), priority, objective, id)

    def generate_id(self) -> str:
        """ Generate a unique identifier for the task. """
        # return f"GenericObservation_{self.parameter}_{self.priority}_{int(self.location[0][2])}_{int(self.location[0][3])}"
        return str(uuid.uuid1())

    def copy(self) -> object:
        """ Create a deep copy of the task. """
        return DefaultMissionTask(
            self.parameter,
            self.location[0],
            self.availability.right,
            self.priority,
            self.objective,
            self.id,
        )
    
    def __repr__(self):
        return f"DefaultMissionTask-'{self.parameter}'@({int(self.location[0][-2])},{int(self.location[0][-1])})"

    @classmethod
    def from_dict(cls, task_dict: dict) -> 'DefaultMissionTask':
        """ Create a task from a dictionary. """
        assert 'task_type' in task_dict, "Task type must be specified in the dictionary."
        assert task_dict['task_type'] == GenericObservationTask.DEFAULT, "Task type must be 'default_mission_task'."
        assert 'parameter' in task_dict, "Task observation parameter must be specified in the dictionary."
        assert 'location' in task_dict, "Task location must be specified in the dictionary."
        assert 'availability' in task_dict, "Task availability must be specified in the dictionary."

        if 'objective' in task_dict:
            assert isinstance(task_dict['objective'], dict) or task_dict['objective'] is None, "Objective must be a dictionary."
            objective = MissionObjective.from_dict(task_dict['objective']) if task_dict['objective'] is not None else None
        else:
            objective = None

        return cls(
            parameter=task_dict['parameter'],
            location=tuple(task_dict['location'][0]),
            mission_duration=task_dict['availability']['right'],
            priority=task_dict.get('priority', 1.0),
            objective=objective,
            id=task_dict.get('id',None),
        )

class EventObservationTask(GenericObservationTask):
    def __init__(self,  
                 parameter : str, 
                 location : list = None,
                 availability : Interval = None,
                 priority : float = None,
                 event : GeophysicalEvent = None,
                 objective : MissionObjective = None,
                 id = None
                 ):
        """
        ### Event Observation Task
        Represents an event observation task to be scheduled by an agent.
        - :`parameter`: The parameter to be observed (e.g., "temperature", "humidity").
        - :`location`: The location(s) to be observed, represented as a list of tuples of (lat[deg], lon[deg], grid index, gp index).
        - :`availability`: The time interval during which the task is available. If not provided, it will be set to the event's duration.
        - :`priority`: The priority of the task, which can be used to determine its importance relative to other tasks. If not provided, it will be set to the event's severity.
        - :`event`: The geophysical event to be observed, represented as a `GeophysicalEvent` object.
        - :`objective`: Reference mission objective associated with the task by the requesting agent, represented as a `MissionObjective` object.
        - :`id`: A unique identifier for the task. If not provided, a new ID will be generated.
        """

        # Validate Inputs
        assert isinstance(event, GeophysicalEvent) or event is None, "If specified, event must be a `GeophysicalEvent`."
        assert isinstance(objective, EventDrivenObjective) or objective is None, "If specified, objective must be a `EventDrivenObjective`."

        if event is None and objective is None:
            assert location is not None, "If no event or objective is specified, locations must be provided."
            assert availability is not None, "If no event or objective is specified, availability must be provided."
            assert priority is not None, "If no event or objective is specified, priority must be provided."

        if event is not None and objective is not None:
            assert event.event_type == objective.event_type, "If both event and objective are specified, their event types must match."

        # Extract event or objective attributes if given
        if event is not None: 
            # Event specified; use event attributes
            location = event.location if location is None else location
            availability = Interval(event.t_start, event.t_start + event.d_exp) if availability is None else availability   
            priority = event.severity if priority is None else priority

        elif objective is not None:
            # TODO: implement objective-based task creation
            raise NotImplementedError("`EventObservationTask` creation from only an objective is not implemented yet.")

            # # Objective specified; use objective attributes
            # assert objective.parameter == parameter, "If objective is specified, target parameter must match the objective's parameter."

            # ## Extract spatial measurement requirements
            # spatial_req = [req for req in objective.requirements
            #                if isinstance(req, SpatialCoverageRequirement)]
            # spatial_req : SpatialCoverageRequirement = spatial_req[0] if spatial_req else None
            # assert spatial_req is not None or location is not None, \
            #     "If no event is specified, either a specified location or a spatial requirement must be provided."
            
            # if isinstance(spatial_req, SinglePointSpatialRequirement):
            #     location = [spatial_req.target] if location is None else location
            # else:
            #     raise NotImplementedError(f"Default task creation for spatial requirement type {type(spatial_req)} is not implemented yet")
            
            # ## Extract temporal measurement requirements
            # availability_req = [req for req in objective.requirements
            #                     if isinstance(req, AvailabilityRequirement)]
            # availability_req : AvailabilityRequirement = availability_req[0] if availability_req else None
            # assert availability_req is not None or availability is not None, \
            #     "If no event is specified, either a specified availability or an availability requirement must be provided."
            # availability = availability_req.availability if availability is None else availability

            # ## Validate task priority 
            # assert priority is not None, "If no event is specified, priority must be provided."

        # Set attributes
        self.event : GeophysicalEvent = event

        # Initialize parent class
        super().__init__(GenericObservationTask.EVENT, parameter, location, availability, priority, objective, id)

    def generate_id(self) -> str:
        """ Generate a unique identifier for the task. `Mission-Parameter-Grid Index-Ground Point Index` """
        return f"EventObservationTask-'{self.parameter}'@({self.location[0][2]},{self.location[0][3]})-EVENT-{self.event.id.split('-')[0] if self.event else 'None'}"

    def copy(self) -> object:
        """ Create a deep copy of the task. """
        return EventObservationTask(
            parameter=self.parameter,
            location=self.location,
            availability=self.availability,
            priority=self.priority,
            event=self.event,
            objective=self.objective,
            id=self.id
        )

    def __repr__(self):
        return self.id
    
    def to_dict(self) -> dict:
        """ Convert the task to a dictionary. """
        d = super().to_dict()
        d.update({
            "event": self.event.to_dict() if self.event else None
        })
        return d

    @classmethod
    def from_dict(cls, task_dict: dict) -> 'EventObservationTask':
        """ Create a task from a dictionary. """
        # Validate Inputs
        assert 'task_type' in task_dict, "Task type must be specified in the dictionary."
        assert task_dict['task_type'] == GenericObservationTask.EVENT, "Task type must be 'event_observation_task'."
        assert 'parameter' in task_dict, "Parameter must be specified in the dictionary."
        assert 'event' in task_dict, "Event must be specified in the dictionary."

        # Unpack dictionary
        event = GeophysicalEvent.from_dict(task_dict['event']) if 'event' in task_dict else None
        objective = MissionObjective.from_dict(task_dict['objective']) if 'objective' in task_dict else None
        
        # Return task
        return cls(
            parameter=task_dict['parameter'],
            event=event,
            objective=objective,
            id=task_dict.get('id',None),
        )        

