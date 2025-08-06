from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from chess3d.agents.actions import ObservationAction
from chess3d.mission.events import GeophysicalEvent
from chess3d.orbitdata import OrbitData
from chess3d.mission.mission import *
from chess3d.utils import Interval


class GenericObservationTask(ABC):
    DEFAULT = 'default_mission_task'
    EVENT = 'event_driven_task'

    def __init__(self,
                 task_type : str,
                 parameter : str,
                 location: list,
                 availability: Interval,
                 id : str = None,
                ):
        """
        Generic observation task to be scheduled by an agent.
        - :`parameter`: The parameter to be observed (e.g., "temperature", "humidity").
        - :`location`: Location or list of locations to be observed, each represented as a tuple of (lat[deg], lon[deg], grid index, gp index).
        - :`availability`: The time interval during which the task is available.
        - :`reward`: The reward for completing the task.
        - :`relevant_objective`: The relevant mission objective associated with the task by the agent who initialized it.
        - :`duration_requirements`: The duration requirements for the task.
        - :`id`: A unique identifier for the task. If not provided, a new ID will be generated.
        """

        # validate inputs
        assert isinstance(task_type, str), "Task type must be a string."
        assert task_type in [self.DEFAULT, self.EVENT], "Task type must be either 'default_mission_task' or 'event_driven_task'."
        assert isinstance(parameter, str), "Parameter must be a string."
        assert isinstance(location, list), "Locations must be a list."
        assert all([isinstance(location, tuple) for location in location]), "All locations must tuples of type (lat[deg], lon[deg], grid index, gp index)."
        assert all([len(location) == 4 for location in location]), "All locations must tuples of type (lat[deg], lon[deg], grid index, gp index)."
        assert isinstance(availability, Interval), "Availability must be an Interval."
        assert availability.left >= 0.0, "Start of availability must be non-negative."

        # Set attributes
        self.task_type : str = task_type
        self.parameter : str = parameter
        self.location : list[tuple] = location
        self.availability : Interval = availability
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
    
    def to_dict(self) -> dict:
        """ Convert the task to a dictionary. """
        return {
            "task_type": self.task_type,
            "parameter": self.parameter,
            "location": [loc for loc in self.location],
            "availability": self.availability.to_dict(),
            "id": self.id,
        }
    
    @abstractmethod
    def __repr__(self):
        """ String representation of the task. """

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
        
class DefaultMissionTask(GenericObservationTask):
    def __init__(self,
                 parameter : str,
                 location: list,
                 mission_duration : float,
                 id : str = None
                ):
        """
        ### Default Observation Task
        Represents a default observation task of a point location to be scheduled by an agent.
        - :`parameter`: The parameter to be observed (e.g., "temperature", "humidity").
        - :`location`: The location to be observed, represented as a tuple of (lat[deg], lon[deg], grid index, gp index).
        - :`mission_duration`: The duration of the mission in seconds.
        - :`id`: A unique identifier for the task. If not provided, a new ID will be generated.
        """

        # validate inputs
        assert isinstance(location, tuple), "Location must be a tuple of type (lat[deg], lon[deg], grid index, gp index)."
        assert len(location) == 4, "Location must be a tuple of type (lat[deg], lon[deg], grid index, gp index)."
        assert all([isinstance(coordinate, float) or isinstance(coordinate, int) for coordinate in location]), \
            "All locations must tuples of type (lat[deg], lon[deg], grid index, gp index)."

        # initialte parent class
        super().__init__(GenericObservationTask.DEFAULT, parameter, [location], Interval(0.0, mission_duration), id)

    def generate_id(self) -> str:
        """ Generate a unique identifier for the task. `Mission-Parameter-Grid Index-Ground Point Index` """
        return f"GenericObservation_{self.parameter}_{self.location[0][2]}_{self.location[0][3]}"

    def copy(self) -> object:
        """ Create a deep copy of the task. """
        return DefaultMissionTask(
            self.parameter,
            self.location[0],
            self.availability.right,
            self.id,
        )
    
    def __repr__(self):
        return f"DefaultMissionTask(parameter={self.parameter}, location={self.location}, availability={self.availability}, id={self.id})"

    @classmethod
    def from_dict(cls, task_dict: dict) -> 'DefaultMissionTask':
        """ Create a task from a dictionary. """
        assert 'task_type' in task_dict, "Task type must be specified in the dictionary."
        assert task_dict['task_type'] == GenericObservationTask.DEFAULT, "Task type must be 'default_mission_task'."
        assert 'parameter' in task_dict, "Task observation parameter must be specified in the dictionary."
        assert 'location' in task_dict, "Task location must be specified in the dictionary."
        assert 'availability' in task_dict, "Task availability must be specified in the dictionary."

        return cls(
            parameter=task_dict['parameter'],
            location=task_dict['location'][0],
            mission_duration=task_dict['availability']['right'],
            id=task_dict.get('id',None),
        )

class EventObservationTask(GenericObservationTask):
    def __init__(self,  
                 parameter : str, 
                 event : GeophysicalEvent,
                 objective : MissionObjective = None,
                 id = None
                 ):
        """
        ### Event Observation Task
        Represents an event observation task to be scheduled by an agent.
        - :`parameter`: The parameter to be observed (e.g., "temperature", "humidity").
        - :`event`: The geophysical event to be observed, represented as a `GeophysicalEvent` object.
        - :`objective`: Reference mission objective associated with the task by the requesting agent, represented as a `MissionObjective` object.
        - :`id`: A unique identifier for the task. If not provided, a new ID will be generated.
        """

        # Validate Inputs
        assert isinstance(event, GeophysicalEvent), "Event must be a GeophysicalEvent."
        assert isinstance(objective, MissionObjective) or objective is None, "Objective must be a MissionObjective."
        if objective is not None: assert parameter == objective.parameter, "Target parameter must match the objective's parameter."

        # Set attributes
        self.event = event
        self.severity = event.severity
        self.objective = objective

        # Extract event attributes
        availability = Interval(event.t_start, event.t_start + event.d_exp)
        locations = [tuple(target) for target in event.location]

        # Initialize parent class
        super().__init__(GenericObservationTask.EVENT, parameter, locations, availability, id)


    def generate_id(self) -> str:
        """ Generate a unique identifier for the task. `Mission-Parameter-Grid Index-Ground Point Index` """
        return f"EventObservationTask_{self.parameter}_{self.location[0][2]}_{self.location[0][3]}_EVENT-{self.event.id.split('-')[0]}"

    def copy(self) -> object:
        """ Create a deep copy of the task. """
        return EventObservationTask(
            parameter=self.parameter,
            event=self.event,
            objective=self.objective,
            id=self.id
        )

    def __repr__(self):
        return f"EventObservationTask(parameter={self.parameter}, event={self.event}, location={self.location}, availability={self.availability}, id={self.id})"
    
    def to_dict(self) -> dict:
        """ Convert the task to a dictionary. """
        d = super().to_dict()
        d.update({
            "event": self.event.to_dict(),
            "objective": self.objective.to_dict() if self.objective else None,
        })
        return d

    @classmethod
    def from_dict(cls, task_dict: dict) -> 'EventObservationTask':
        """ Create a task from a dictionary. """
        assert 'task_type' in task_dict, "Task type must be specified in the dictionary."
        assert task_dict['task_type'] == GenericObservationTask.EVENT, "Task type must be 'event_observation_task'."
        assert 'parameter' in task_dict, "Parameter must be specified in the dictionary."
        assert 'event' in task_dict, "Event must be specified in the dictionary."

        event = GeophysicalEvent.from_dict(task_dict['event'])
        objective = MissionObjective.from_dict(task_dict['objective']) if 'objective' in task_dict else None
        
        return cls(
            parameter=task_dict['parameter'],
            event=event,
            objective=objective,
            id=task_dict.get('id',None),
        )
        
# TODO: Update specific observation tasks
class SpecificObservationTask:
    def __init__(self,
                 parent_tasks : Union[GenericObservationTask, set],
                 instrument_name : str, 
                 accessibility : Interval,
                 slew_angles : Interval,
                 id : str = None,
                 ):
        """ Represents an observation task to be scheduled by a particular agent """

        # format inputs
        if isinstance(parent_tasks, GenericObservationTask): parent_tasks = {parent_tasks}
        
        # validate inputs
        assert isinstance(parent_tasks, set), "Parent tasks must be a set of GenericObservationTask."
        assert all([isinstance(task, GenericObservationTask) for task in parent_tasks]), "All parent tasks must be instances of GenericObservationTask."
        parent_tasks : set[GenericObservationTask]

        assert isinstance(accessibility, Interval), "Accessibility must be an Interval."
        assert isinstance(slew_angles, Interval), "Slew angles must be an Interval."
        assert all([accessibility.overlaps(parent_task.availability) for parent_task in parent_tasks]), "Accesibility interval must be within the parent tasks' availability interval."

        # set parameters
        self.parent_tasks : set[GenericObservationTask] = parent_tasks
        self.instrument_name : str = instrument_name
        self.accessibility : Interval = accessibility
        self.slew_angles : Interval = slew_angles
        self.id : str = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())
    
    def can_combine(self, other_task : object) -> bool:
        """ Check if two tasks can be combined based on their time and slew angle. """
        
        # Check if the other task is an instance of ObservationTask
        if not isinstance(other_task, SpecificObservationTask):
            raise ValueError("The other task must be an instance of Task.")
        
        # Check if the instrument names are the same
        if self.instrument_name != other_task.instrument_name:
            return False
        
        # Check if the parent tasks are the same
        parent_task_types = {type(task) for task in self.parent_tasks}
        assert len(parent_task_types) == 1, "All parent tasks must be of the same type."
        parent_task_type = parent_task_types.pop()
        if any([type(task) != parent_task_type for task in other_task.parent_tasks]):
            return False

        # Check if parent tasks have the same valid instruments
        my_valid_instruments = {instrument_name 
                                for task in self.parent_tasks
                                for instrument_name in task.objective.valid_instruments}
        their_valid_instruments = {instrument_name 
                                for task in other_task.parent_tasks
                                for instrument_name in task.objective.valid_instruments}
        if my_valid_instruments != their_valid_instruments: 
            return False

        # Check if the availability time intervals overlap
        accessibility_union : Interval = self.accessibility.union(other_task.accessibility, extend=True)
        # accessibility_union : Interval = self.accessibility.union(other_task.accessibility)
        
        if not accessibility_union.is_empty():
            # Check if the time intervals are within the maximum duration
            max_parent_duration_self = min([task.duration_requirements.right for task in self.parent_tasks])
            max_parent_duration_other = min([task.duration_requirements.right for task in other_task.parent_tasks])
            
            if accessibility_union.span() > min(max_parent_duration_self, max_parent_duration_other):
                return False

        # Check if the slew angles overlap
        slew_angle_overlap : Interval = self.slew_angles.intersection(other_task.slew_angles) 

        return not (accessibility_union.is_empty() or slew_angle_overlap.is_empty())

    def merge(self, other_task : object) -> object:
        try:
            """ Merge two tasks into one. """
            assert isinstance(other_task, SpecificObservationTask), "The other task must be an instance of ObservationTask."
            assert self.can_combine(other_task), "Tasks cannot be combined."

            # Combine the time intervals and slew angles
            combined_time_interval : Interval = self.accessibility.union(other_task.accessibility, extend=True)
            combined_slew_angles : Interval  = self.slew_angles.intersection(other_task.slew_angles)
                    
            # Update the task attributes
            parent_tasks = {task for task in self.parent_tasks}
            parent_tasks.update({task for task in other_task.parent_tasks})
            accessibility = combined_time_interval
            slew_angles = combined_slew_angles
            
            return SpecificObservationTask(parent_tasks, self.instrument_name, accessibility, slew_angles, self.id)
        except AssertionError as e:
            x = 1
            self.can_combine(other_task)
            raise e
    
    def __repr__(self):
        return f"ObservationTask(parent_tasks={self.parent_tasks}, accessibility={self.accessibility}, slew_angles={self.slew_angles})"
    
    def __str__(self):
        return f"ObservationTask(parent_tasks={self.parent_tasks}, accessibility={self.accessibility}, slew_angles={self.slew_angles})"
    
    def __hash__(self):
        return hash(self.id)

