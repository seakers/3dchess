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
                 targets: list,
                 availability: Interval,
                 id : str = None,
                ):
        """
        Generic observation task to be scheduled by an agent.
        - :`parameter`: The parameter to be observed (e.g., "temperature", "humidity").
        - :`targets`: A list of targets to be observed, each represented as a tuple of (lat[deg], lon[deg], grid index, gp index).
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
        assert isinstance(targets, list), "Targets must be a list."
        assert all([isinstance(target, tuple) for target in targets]), "All targets must tuples of type (lat[deg], lon[deg], grid index, gp index)."
        assert all([len(target) == 4 for target in targets]), "All targets must tuples of type (lat[deg], lon[deg], grid index, gp index)."
        assert isinstance(availability, Interval), "Availability must be an Interval."
        assert availability.left >= 0.0, "Start of availability must be non-negative."

        # Set attributes
        self.task_type : str = task_type
        self.parameter : str = parameter
        self.targets : list[tuple] = targets
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

    def available(self, time : float) -> bool:
        """ Check if the task is available at a given time. """
        return time in self.availability    
    
    def to_dict(self) -> dict:
        """ Convert the task to a dictionary. """
        return {
            "parameter": self.parameter,
            "targets": [target for target in self.targets],
            "availability": self.availability.to_dict(),
            "id": self.id,
        }
    
    @classmethod
    def from_dict(cls, task_dict: dict) -> 'GenericObservationTask':
        """ Create a task from a dictionary. """
        task_type = task_dict['task_type']
        

        if task_type == cls.DEFAULT:
            return DefaultObservationTask.from_dict(task_dict)
        
        # elif task_type == cls.EVENT:
        #     return EventObservationTask.from_dict(task_dict)

        return ValueError(f"Unknown task type: {task_type}")
        
class DefaultObservationTask(GenericObservationTask):
    def __init__(self,
                 parameter : str,
                 target: list,
                 mission_duration : float,
                 id : str = None
                ):
        """
        ### Default Observation Task
        Represents a default observation task of a point target to be scheduled by an agent.
        - :`parameter`: The parameter to be observed (e.g., "temperature", "humidity").
        - :`target`: The target to be observed, represented as a tuple of (lat[deg], lon[deg], grid index, gp index).
        - :`mission_duration`: The duration of the mission in seconds.
        - :`id`: A unique identifier for the task. If not provided, a new ID will be generated.
        """

        # validate inputs
        assert isinstance(target, tuple), "Target must be a tuple of type (lat[deg], lon[deg], grid index, gp index)."
        assert len(target) == 4, "Target must be a tuple of type (lat[deg], lon[deg], grid index, gp index)."
        assert all([isinstance(coordinate, float) or isinstance(coordinate, int) for coordinate in target]), \
            "All targets must tuples of type (lat[deg], lon[deg], grid index, gp index)."

        # initialte default values
        targets = [target]
        availability = Interval(0.0, mission_duration)

        # initialte parent class
        super().__init__(GenericObservationTask.DEFAULT,parameter, targets, availability, id)

    def generate_id(self) -> str:
        """ Generate a unique identifier for the task. `Mission-Parameter-Grid Index-Ground Point Index` """
        return f"GenericObservation_{self.parameter}_{self.targets[0][2]}_{self.targets[0][3]}"

    def copy(self) -> object:
        """ Create a deep copy of the task. """
        return DefaultObservationTask(
            self.parameter,
            self.targets[0],
            self.availability.right,
            id=self.id,
        )
    
    def __repr__(self):
        return f"DefaultObservationTask(parameter={self.parameter}, targets={self.targets}, availability={self.availability}, id={self.id})"

    @classmethod
    def from_dict(cls, task_dict: dict) -> 'DefaultObservationTask':
        """ Create a task from a dictionary. """
        return cls(
            parameter=task_dict['parameter'],
            target=task_dict['targets'][0],
            mission_duration=task_dict['availability']['right'],
            id=task_dict.get('id'),
        )

class EventObservationTask(GenericObservationTask):
    def __init__(self,
                 event : GeophysicalEvent,
                 mission : str,
                 objective : MissionObjective,
                 reobservation_strategy : str,
                 id : str = None,
                 duration_requirements = Interval(0.0, np.Inf)
                ):
        # validate inputs
        assert isinstance(event, GeophysicalEvent), "Event must be a GeophysicalEvent."
        
        # set default values
        targets = [tuple(target for target in event.location)]
        availability = Interval(event.t_start, event.t_end)
        reward = event.severity

        # initialte default values
        self.event = event
        super().__init__(mission, objective, targets, availability, reward, reobservation_strategy, id, duration_requirements)

    def copy(self):
        return EventObservationTask(
            self.event,
            self.mission,
            self.objective,
            self.reobservation_strategy,
            id=self.id,
            duration_requirements=self.duration_requirements
        )
    
    def generate_id(self) -> str:
        """ Generate a unique identifier for the task. `Mission-Parameter-Grid Index-Ground Point Index` """
        return f"{self.mission}_{self.objective.parameter}_{self.targets[0][2]}_{self.targets[0][3]}_EVENT-{self.event.id.split('-')[0]}"
    
    def __repr__(self):
        return f"EventObservationTask(event={self.event}, mission={self.mission}, objective={self.objective}, targets={self.targets}, availability={self.availability}, reward={self.reward}, reobservation_strategy={self.reobservation_strategy}, id={self.id})"

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

class ObservationTracker:
    def __init__(self, lat : float, lon : float, grid_index : int, gp_index : int, t_last : str = -1, n_obs : int = 0, latest_observation : dict = None):
        """ 
        Class to track the observation tasks and their history.
        """
        # validate inputs
        assert isinstance(lat, (float, int)), "Latitude must be a float or int."
        assert isinstance(lon, (float, int)), "Longitude must be a float or int."
        assert isinstance(grid_index, int), "Grid index must be an integer."
        assert isinstance(gp_index, int), "Ground point index must be an integer."
        assert isinstance(t_last, (int, float)), "Last observation time must be a float or int."
        assert isinstance(n_obs, int), "Number of observations must be an integer."
        assert n_obs >= 0, "Number of observations must be non-negative."
        assert lat >= -90 and lat <= 90, "Latitude must be between -90 and 90 degrees."
        assert lon >= -180 and lon <= 180, "Longitude must be between -180 and 180 degrees."
        assert grid_index >= 0, "Grid index must be non-negative."
        assert gp_index >= 0, "Ground point index must be non-negative."

        # assign parameters
        self.lat = lat
        self.lon = lon
        self.grid_index = grid_index
        self.gp_index = gp_index
        self.t_last = t_last
        self.n_obs = n_obs
        self.latest_observation = latest_observation
        self.observations : list[dict] = []
    
    def update(self, observation : dict) -> None:
        """ Update the observation tracker with a new observation."""        
        # update number of observations at this target
        self.n_obs += 1

        # update list of known observations 
        self.observations.append(observation)

        # update last observation time
        if observation['t_end'] >= self.t_last:
            self.t_last = observation['t_end']
            self.latest_observation = observation

    def __repr__(self):
        return f"ObservationTracker(grid_index={self.grid_index}, gp_index={self.gp_index}, lat={self.lat}, lon={self.lon}, t_last={self.t_last}, n_obs={self.n_obs})"

class ObservationHistory:
    def __init__(self, orbitdata : OrbitData):
        """
        Class to track the observation history of the agent.
        """
        self.history = {}
        self.grid_lookup = {}

        for gp_index in range(len(orbitdata.grid_data)):
            grid : pd.DataFrame = orbitdata.grid_data[gp_index]
            
            for _,row in grid.iterrows():
                lat = row["lat [deg]"]
                lon = row["lon [deg]"]
                grid_index = int(row["grid index"])
                gp_index = int(row["GP index"])

                # create a new entry for the grid point
                if grid_index not in self.history:
                    self.history[grid_index] = {}
                
                # create a new entry for the grid point
                if gp_index not in self.history[grid_index]:
                    self.history[grid_index][gp_index] = ObservationTracker(lat, lon, grid_index, gp_index) 
                
                # create a lookup table for the grid points
                lat_key = round(row["lat [deg]"], 6)
                lon_key = round(row["lon [deg]"], 6)
                self.grid_lookup[(lat_key, lon_key)] = (
                    int(row["grid index"]),
                    int(row["GP index"])
                )

    def update(self, observations : list) -> None:
        """
        Update the observation history with the new observations.
        """
        for _,observations_data in observations:
            for observation in observations_data:
                grid_index = observation['grid index']
                gp_index = observation['GP index']
                
                tracker : ObservationTracker = self.history[grid_index][gp_index]
                tracker.update(observation)

                # grid_index = observation['grid index']
                # gp_index = observation['GP index']
                # t_end = observation['t_end']
                
                # tracker : ObservationTracker = self.history[grid_index][gp_index]

                # tracker.t_last = t_end
                # tracker.n_obs += 1
                # tracker.latest_observation = observation


    def get_observation_history(self, grid_index : int, gp_index : int) -> ObservationTracker:
        if grid_index in self.history and gp_index in self.history[grid_index]:
            return self.history[grid_index][gp_index]
        else:
            raise ValueError(f"Observation history for grid index {grid_index} and ground point index {gp_index} not found.")

        