from abc import ABC, abstractmethod
from typing import Union
import uuid

from chess3d.mission.events import GeophysicalEvent
from chess3d.mission.objectives import *
from chess3d.utils import EmptyInterval, Interval


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
        - :`parameter`: The parameter to be observed (e.g., "temperature", "humidity").
        - :`location`: Location or list of locations to be observed, each represented as a tuple of (lat[deg], lon[deg], grid index, gp index).
        - :`availability`: The time interval during which the task is available.
        - :`reward`: The reward for completing the task.
        - :`relevant_objective`: The relevant mission objective associated with the task by the agent who initialized it.
        - :`priority`: The priority of the task, which can be used to determine its importance relative to other tasks.
        - :`objective`: The objective of the mission, which defines the goal of the task.
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
        assert isinstance(priority, (float, int)), "Priority must be a number."
        assert priority >= 0, "Priority must be non-negative."
        assert isinstance(objective, MissionObjective) or objective is None, f"If specified, objective must be a `MissionObjective`. Is of type {type(objective)}."
        assert id is None or isinstance(id, str), "ID must be a string or None."

        if objective is not None:
            # Objective specified; check objective attributes 
            assert parameter == objective.parameter, "Target parameter must match the objective's parameter."

        # Set attributes
        self.task_type : str = task_type
        self.parameter : str = parameter
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
        """ Generate a unique identifier for the task. `Mission-Parameter-Grid Index-Ground Point Index` """
        return f"GenericObservation_{self.parameter}_{self.priority}_{self.location[0][2]}_{self.location[0][3]}"

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
        return f"DefaultMissionTask(parameter={self.parameter}, priority={self.priority}, location={self.location}, availability={self.availability}, id={self.id})"

    @classmethod
    def from_dict(cls, task_dict: dict) -> 'DefaultMissionTask':
        """ Create a task from a dictionary. """
        assert 'task_type' in task_dict, "Task type must be specified in the dictionary."
        assert task_dict['task_type'] == GenericObservationTask.DEFAULT, "Task type must be 'default_mission_task'."
        assert 'parameter' in task_dict, "Task observation parameter must be specified in the dictionary."
        assert 'location' in task_dict, "Task location must be specified in the dictionary."
        assert 'availability' in task_dict, "Task availability must be specified in the dictionary."

        if 'objective' in task_dict:
            assert isinstance(task_dict['objective'], dict), "Objective must be a dictionary."
            objective = MissionObjective.from_dict(task_dict['objective'])
        else:
            objective = None

        return cls(
            parameter=task_dict['parameter'],
            location=task_dict['location'][0],
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
        assert isinstance(objective, MissionObjective) or objective is None, "If specified, objective must be a `MissionObjective`."

        if event is None and objective is None:
            assert location is not None, "If no event or objective is specified, locations must be provided."
            assert availability is not None, "If no event or objective is specified, availability must be provided."
            assert priority is not None, "If no event or objective is specified, priority must be provided."

        # Extract event or objective attributes if given
        if event is not None: 
            # Event specified; use event attributes
            location = event.location if location is None else location
            availability = Interval(event.t_start, event.t_start + event.d_exp) if availability is None else availability   
            priority = event.severity if priority is None else priority

        elif objective is not None:
            # Objective specified; use objective attributes

            ## Extract spatial measurement requirements
            spatial_req = [req for req in objective.requirements
                           if isinstance(req, SpatialRequirement)]
            spatial_req : SpatialRequirement = spatial_req[0] if spatial_req else None
            assert spatial_req is not None or location is not None, \
                "If no event is specified, either a specified location or a spatial requirement must be provided."
            
            if isinstance(spatial_req, PointTargetSpatialRequirement):
                location = [spatial_req.target] if location is None else location
            else:
                raise NotImplementedError(f"Default task creation for spatial requirement type {type(spatial_req)} is not implemented yet")
            
            ## Extract temporal measurement requirements
            availability_req = [req for req in objective.requirements
                                if isinstance(req, AvailabilityRequirement)]
            availability_req : AvailabilityRequirement = availability_req[0] if availability_req else None
            assert availability_req is not None or availability is not None, \
                "If no event is specified, either a specified availability or an availability requirement must be provided."
            availability = availability_req.availability if availability is None else availability

            ## Validate task priority 
            assert priority is not None, "If no event is specified, priority must be provided."

        # Set attributes
        self.event : GeophysicalEvent = event

        # Initialize parent class
        super().__init__(GenericObservationTask.EVENT, parameter, location, availability, priority, objective, id)

    def generate_id(self) -> str:
        """ Generate a unique identifier for the task. `Mission-Parameter-Grid Index-Ground Point Index` """
        return f"EventObservationTask_{self.parameter}_{self.priority}_{self.location[0][2]}_{self.location[0][3]}_EVENT-{self.event.id.split('-')[0] if self.event else 'None'}"

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
        return f"EventObservationTask(parameter={self.parameter}, priority={self.priority}, event={self.event}, location={self.location}, availability={self.availability}, id={self.id})"

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

class SpecificObservationTask:
    def __init__(self,
                 parent_task : Union[GenericObservationTask, set],
                 instrument_name : str, 
                 accessibility : Interval,
                 duration_requirements : Interval,
                 slew_angles : Interval,
                 id : str = None,
                 ):
        """ Represents an observation task to be scheduled by a particular agent """

        # format inputs
        
        # validate inputs
        assert isinstance(parent_task, (GenericObservationTask, set)), "Parent task(s) must be a `GenericObservationTask` or a set of `GenericObservationTask`."
        assert isinstance(instrument_name, str), "Instrument name must be a string."
        assert isinstance(accessibility, Interval), "Accessibility must be an Interval."
        assert isinstance(duration_requirements, Interval), "Duration requirements must be an Interval."
        assert duration_requirements.left >= 0.0, "Start of duration requirements must be non-negative."
        assert isinstance(slew_angles, Interval), "Slew angles must be an Interval."
        
        if isinstance(parent_task, set):
            assert all([isinstance(task, GenericObservationTask) 
                        for task in parent_task]), \
                "All parent tasks must be instances of GenericObservationTask."
            assert all([accessibility.overlaps(task.availability) 
                        for task in parent_task 
                        if isinstance(task, GenericObservationTask)]),\
                "Accesibility interval must be within the parent tasks' availability interval."
        else:
            assert accessibility.overlaps(parent_task.availability), \
                "Accessibility interval must be within the parent task's availability interval."

        # set parametersparent_task}
        self.parent_tasks : set[GenericObservationTask] = \
              {parent_task} if isinstance(parent_task, GenericObservationTask) else parent_task
        self.instrument_name : str = instrument_name
        self.accessibility : Interval = accessibility
        self.duration_requirements : Interval = duration_requirements
        self.slew_angles : Interval = slew_angles
        self.id : str = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())
    
    def copy(self) -> 'SpecificObservationTask':
        """ Create a deep copy of the task. """
        return SpecificObservationTask(
            parent_task=self.parent_tasks,
            instrument_name=self.instrument_name,
            accessibility=self.accessibility,
            duration_requirements=self.duration_requirements,
            slew_angles=self.slew_angles,
            id=self.id
        )
    
    def get_location(self) -> List[tuple]:
        """ Collects the location information of all parent tasks. """
        return list({loc for task in self.parent_tasks for loc in task.location})
    
    def get_objectives(self) -> List[MissionObjective]:
        """ Collects the objectives of all parent tasks. """
        return list({task.objective for task in self.parent_tasks})
    
    def get_priority(self) -> float:
        """ Collects the priority of all parent tasks. """
        return sum(task.priority for task in self.parent_tasks) if self.parent_tasks else 0.0

    def is_mutually_exclusive(self, other_task : 'SpecificObservationTask') -> bool:
        """ Check if two tasks are mutually exclusive. """
        common_parents : set = self.parent_tasks.intersection(other_task.parent_tasks)
        return len(common_parents) > 0

    def can_merge(self, other_task : object, must_overlap : bool = False) -> bool:
        """ Check if two tasks can be merged based on their time and slew angle. """
        
        # Check if the other task is an instance of ObservationTask
        if not isinstance(other_task, SpecificObservationTask):
            raise ValueError("The other task must be an instance of Task.")
        
        # Check if the instrument names are the same
        if self.instrument_name != other_task.instrument_name: return False
        
        # Check if the slew angles overlap
        slew_angles : Interval = self.slew_angles.intersection(other_task.slew_angles) 

        # Calculate accessibility overlap
        accessibility : Interval = self.accessibility.intersection(other_task.accessibility)

        # Check if accesibility has any overlap
        if accessibility.is_empty(): # There is no overlap between the tasks' availability
            # If accessibility time windows are not allowed to be extended, we cannot merge the tasks
            if must_overlap: return False 

            # Find new accessibility bounds
            accessibility_start = min(self.accessibility.right - self.duration_requirements.left, 
                                      other_task.accessibility.right - other_task.duration_requirements.left)
            accessibility_end = max(self.accessibility.left + self.duration_requirements.left, 
                                    other_task.accessibility.left + other_task.duration_requirements.left)

            # Expand accessibility
            accessibility : Interval = Interval(accessibility_start, accessibility_end) if accessibility_end > accessibility_start else EmptyInterval()

            # Calculate new observation duration requirement
            min_duration_req = accessibility.span()

        else: # There is an overlap between the tasks' availability
            # Calculate new observation duration requirement
            min_duration_req = max(self.duration_requirements.left, other_task.duration_requirements.left)

        # Check if merge can occur
        return (    not slew_angles.is_empty()                  # slew angles overlap
                and min_duration_req <= accessibility.span()    # accessibility overlap encompasses the duration requirements
                # TODO minimum duration requirements do not exceed maximum requirements?
                )           

    def merge(self, other_task : 'SpecificObservationTask', must_overlap : bool = True) -> object:
        """ Merge two tasks into one. """

        # Check if tasks can be merged
        assert self.can_merge(other_task, must_overlap), "Tasks cannot be combined."

        # Merge parent tasks
        merged_parent_tasks = {task for task in self.parent_tasks}
        merged_parent_tasks.update({task for task in other_task.parent_tasks})

        # Merge accessibility
        merged_accessibility : Interval = self.accessibility.intersection(other_task.accessibility)

        # Check if accesibility has any overlap
        if merged_accessibility.is_empty(): # There is no overlap between the tasks' availability
            # If accessibility time windows are not allowed to be extended, we cannot merge the tasks
            if must_overlap: raise ValueError("Tasks cannot be combined without overlapping accessibility intervals unless explicitly specified.") 

            # Find new accessibility bounds
            accessibility_start = min(self.accessibility.right - self.duration_requirements.left, 
                                      other_task.accessibility.right - other_task.duration_requirements.left)
            accessibility_end = max(self.accessibility.left + self.duration_requirements.left, 
                                    other_task.accessibility.left + other_task.duration_requirements.left)

            # Expand accessibility
            merged_accessibility : Interval = Interval(accessibility_start, accessibility_end) if accessibility_end > accessibility_start else EmptyInterval()

            # Calculate new observation duration requirement
            max_duration_req = np.Inf # TODO improve calculation of this; currently not used but should be.
            min_duration_req = merged_accessibility.span()

        else: # There is an overlap between the tasks' availability
            # Check if either task availability is contained within the other
            if (other_task.accessibility.is_subset(self.accessibility)
                or self.accessibility.is_subset(other_task.accessibility)): # one task's availability is fully within the other's

                # Find task which task is encompassing the other
                encompassing_task : SpecificObservationTask = \
                    max([self, other_task], key=lambda task: task.accessibility.span())
                encompassed_task : SpecificObservationTask = other_task if encompassing_task == self else self

                # Check if encompassed task's availability  fits within encompassing task's duration requirements
                if encompassed_task.accessibility.span() <= encompassing_task.duration_requirements.left: 
                    # Set combined accessibility interval to encompassing task's accessibility
                    merged_accessibility = encompassing_task.accessibility

            # Calculate new observation duration requirement
            max_duration_req = min(self.duration_requirements.right, other_task.duration_requirements.right)
            min_duration_req = max(self.duration_requirements.left, other_task.duration_requirements.left)

            # max_duration_req = max(min(self.duration_requirements.right, 
            #                         other_task.duration_requirements.right), 
            #                     merged_accessibility.span()
            #                     ) # TODO improve calculation of this; currently not used but should be. 
            # min_duration_req = max(self.duration_requirements.left, other_task.duration_requirements.left)

        # Merge observation duration requirements
        merged_duration_requirements = Interval(min_duration_req, max_duration_req)
        
        # Merge slew angles overlap
        merged_slew_angles : Interval = self.slew_angles.intersection(other_task.slew_angles) 

        # Return merged task
        return SpecificObservationTask(merged_parent_tasks, self.instrument_name, merged_accessibility, merged_duration_requirements, merged_slew_angles, self.id)
    
    def __repr__(self):
        return f"SpecificObservationTask(parent_tasks={self.parent_tasks}, accessibility={self.accessibility}, slew_angles={self.slew_angles})"
    
    def __hash__(self):
        return hash(self.id)
