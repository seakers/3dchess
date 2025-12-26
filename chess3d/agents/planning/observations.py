from collections.abc import Iterable
import math
from typing import List, Tuple, Union
import uuid

import numpy as np

from chess3d.agents.planning.tasks import GenericObservationTask
from chess3d.mission.objectives import MissionObjective
from chess3d.utils import EmptyInterval, Interval

class ObservationOpportunity:
    def __init__(self,
                 tasks : Union[GenericObservationTask, set],
                 instrument_name : str, 
                 accessibility : Interval,
                 min_duration : float,
                 slew_angles : Interval,
                 id : str = None,
                 ):
        """ 
        Represents an observation opportunity that can be scheduled by an agent in order to fulfill one or more generic observation tasks.

        **Arguments:**
        - :`tasks` : : The parent generic observation task(s) that this opportunity can fulfill.
        - :`instrument_name`: The name of the instrument to be used for the observation.
        - :`accessibility`: The time interval during which the observation opportunity is accessible.
        - :`min_duration`: The minimum duration required for the observation in seconds [s].
        - :`slew_angles`: The interval of slew angles [deg] available for the observation.
        - :`id`: A unique identifier for the observation opportunity. If not provided, a new ID will be generated.
        """

        # validate inputs
        assert isinstance(tasks, (GenericObservationTask, Iterable)), "Parent task(s) must be a `GenericObservationTask` or a collection of `GenericObservationTask`."
        assert isinstance(instrument_name, str), "Instrument name must be a string."
        assert isinstance(accessibility, Interval), "Accessibility must be an Interval."
        assert not accessibility.is_empty(), "Accessibility must not be empty."
        assert accessibility.left >= 0.0, "Start of accessibility must be non-negative."
        assert isinstance(min_duration, (float, int)), "Minimum duration must be a number."
        assert min_duration >= 0.0, "Minimum duration must be non-negative."
        assert min_duration <= accessibility.span(), "Minimum duration must not exceed accessibility interval span."
        assert isinstance(slew_angles, Interval), "Slew angles must be an Interval."
        
        if isinstance(tasks, Iterable):
            assert all([isinstance(task, GenericObservationTask) 
                        for task in tasks]), \
                "All parent tasks must be instances of `GenericObservationTask`."
            assert all([accessibility.overlaps(task.availability) 
                        for task in tasks 
                        if isinstance(task, GenericObservationTask)]),\
                "Accesibility interval must be within the parent tasks' availability interval."
        else:
            assert accessibility.overlaps(tasks.availability), \
                "Accessibility interval must be within the parent task's availability interval."

        # set parametersparent_task}
        self.tasks : set[GenericObservationTask] = \
              {tasks} if isinstance(tasks, GenericObservationTask) else tasks
        self.instrument_name : str = instrument_name
        self.accessibility : Interval = accessibility
        self.min_duration : Interval = min_duration
        self.slew_angles : Interval = slew_angles
        self.id : str = id if id is not None else self.generate_id(tasks, instrument_name, accessibility)

    @staticmethod
    def generate_id(parent_tasks : Union[GenericObservationTask, set],
                    instrument_name : str,
                    accessibility : Interval) -> str:
        """
        Deterministic ID for a specific observation task based on:
        - parent generic task IDs
        - access interval [start, end)
        - optional agent/instrument/index
        """
        # check inputs
        if isinstance(parent_tasks, GenericObservationTask):
            parent_tasks = {parent_tasks}
        assert isinstance(parent_tasks, set), "parent_tasks must be a set of `GenericObservationTask`"

        # Define namespace for UUID generation
        SPECIFIC_TASK_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

        # Collect parent IDs (sorted string representation)
        parent_part = ",".join(sorted([str(p.id) for p in parent_tasks]))

        # Collect access interval [start, end]
        start, end = accessibility.left, accessibility.right
        interval_part = f"{np.round(start,3)}_{np.round(end,3)}"

        # Collect additional info for disambiguation
        extras = []
        extras.append(f"instrument={instrument_name}")
        extras_part = ";".join(extras)

        # Create canonical name string
        name = f"{parent_part}|{interval_part}|{extras_part}"

        # Return deterministic UUID derived from name
        return str(uuid.uuid5(SPECIFIC_TASK_NAMESPACE, name))
    
    def copy(self) -> 'ObservationOpportunity':
        """ Create a deep copy of the task. """
        return ObservationOpportunity(
            tasks=self.tasks,
            instrument_name=self.instrument_name,
            accessibility=self.accessibility,
            min_duration=self.min_duration,
            slew_angles=self.slew_angles,
            id=self.id
        )
    
    def get_location(self) -> List[tuple]:
        """ Collects the location information of all parent tasks. """
        return list({loc for task in self.tasks for loc in task.location})
    
    def get_objectives(self) -> List[MissionObjective]:
        """ Collects the objectives of all parent tasks. """
        return list({task.objective for task in self.tasks})
    
    def get_priority(self) -> float:
        """ Collects the priority of all parent tasks. """
        return sum(task.priority for task in self.tasks) if self.tasks else 0.0

    def is_mutually_exclusive(self, other_task : 'ObservationOpportunity') -> bool:
        """ Check if two tasks are mutually exclusive. """
        # validate inputs
        assert isinstance(other_task, ObservationOpportunity), "The other task must be an instance of `SpecificObservationTask`."

        # check if they are the same task        
        same_task : bool = (other_task == self)

        # check if they have the same parent tasks
        common_parents : bool = (len(self.tasks.intersection(other_task.tasks)) > 0)
        
        # check if they refer to the same access opportunity
        common_access : bool = self.accessibility.overlaps(other_task.accessibility)
        
        # return true if all conditions are met
        return (not same_task) and common_parents and common_access

    def _calc_time_requirements(self, other_task : 'ObservationOpportunity', must_overlap : bool = False) -> Tuple[Interval,float]:
        """
        Calculates the joint availability and duration requirements for two tasks if merged. 

        **Arguments:**
        - `other_task`: The other task to merge with.
        - `must_overlap`: If True, the tasks must overlap in accessibility to be merged.
        """
        # Calculate accessibility overlap
        accessibility_overlap : Interval = self.accessibility.intersection(other_task.accessibility)

        # If accessibility time windows are not allowed to be extended, we cannot merge the tasks
        if accessibility_overlap.is_empty() and must_overlap: return accessibility_overlap, np.NaN

        # Determine which task starts first
        preceeding_task, proceeding_task = sorted([self, other_task], key=lambda t: t.accessibility.left)

        # Check if accesibility has any overlap
        if not accessibility_overlap.is_empty():  # There is an overlap between the tasks' availability
            # Use tasks' accessibility and duration requirements to find new accessibility bounds
            accessibility_start = max(preceeding_task.accessibility.left, 
                                        min(proceeding_task.accessibility.left + proceeding_task.min_duration - preceeding_task.min_duration,
                                            proceeding_task.accessibility.left)
                                    )
            accessibility_end   = min(preceeding_task.accessibility.right,
                                        max(proceeding_task.accessibility.right - proceeding_task.min_duration + preceeding_task.min_duration,
                                            proceeding_task.accessibility.right)
                                    )
            
            # Merge accessibility
            merged_accessibility = Interval(accessibility_start, accessibility_end) if accessibility_start <= accessibility_end else EmptyInterval()

            # Calculate new observation duration requirement
            min_duration_req = max(preceeding_task.min_duration, proceeding_task.min_duration)
        
        else: # There is no overlap between the tasks' availability; we can only merge by extending the accessibility window
            # Find new accessibility bounds
            accessibility_start = preceeding_task.accessibility.right - preceeding_task.min_duration
            accessibility_end = proceeding_task.accessibility.left + proceeding_task.min_duration

            # Extend accessibility
            merged_accessibility : Interval = Interval(accessibility_start, accessibility_end) if accessibility_end > accessibility_start else EmptyInterval()

            # Calculate new observation duration requirement
            min_duration_req = merged_accessibility.span()

        # Return Time Requirements
        return merged_accessibility, min_duration_req

    def can_merge(self, other : 'ObservationOpportunity', must_overlap : bool = False, max_duration : float = 5*60) -> bool:
        """ 
        Check if two tasks can be merged based on their time and slew angle. 
        
        **Arguments:**
        - `other`: Other task to merge with.
        - `must_overlap`: If True, the tasks must overlap in accessibility to be merged.
        - `max_duration`: The maximum allowed duration for the merged task in seconds [s].
        """
               
        # Validate inputs
        assert isinstance(other, ObservationOpportunity), "The other task must be an instance of SpecificObservationTask."
        assert isinstance(must_overlap, bool), "must_overlap must be a boolean."
        assert isinstance(max_duration, (float, int)), "max_duration must be a number."
        assert max_duration > 0.0, "max_duration must be positive."
        
        # Calculate slew angles overlap
        merged_slew_angles : Interval = self.slew_angles.intersection(other.slew_angles) 
        slew_angles_overlap : bool = self.slew_angles.overlaps(other.slew_angles)

        # Calculate accessibility overlap and duration requirements
        merged_accessibility, min_duration_req = self._calc_time_requirements(other, must_overlap)

        # Gather joint observation targets
        my_targets = set(self.get_location())
        other_targets = set(other.get_location())
        location_overlap : bool = len(my_targets.intersection(other_targets)) > 0

        # Define necessary merge conditions
        merge_conditions = [
                self.instrument_name == other.instrument_name,      # same instrument
                min_duration_req <= max_duration,                   # duration requirements do not exceed maximum allowed duration
                not math.isnan(min_duration_req),                   # joint minimum duration requirements is valid
                min_duration_req <= merged_accessibility.span(),    # accessibility window encompasses the duration requirements
                # and not merged_slew_angles.is_empty()               # slew angles overlap
                slew_angles_overlap,                                # slew angles overlap
                not merged_accessibility.is_empty(),                # there exist a valid joint accessibility window 
                # and not self.is_mutually_exclusive(other_task)      # TODO tasks with common parent tasks cannot be merged
                not location_overlap                                # tasks do not observe the same target(s)
        ]   

        # Return if merge can occur
        return all(merge_conditions)        
        
    def merge(self, other : 'ObservationOpportunity', must_overlap : bool = False, max_duration : float = 2*60) -> 'ObservationOpportunity':
        """ 
        Merge two tasks into one. 
                
        **Arguments:**
        - `other_task`: The other task to merge with.
        - `must_overlap`: If True, the tasks must overlap in accessibility to be merged.
        - `max_duration`: The maximum allowed duration for the merged task in seconds [s] (default: 120 seconds = 2 minutes).
        """
        try:
            # Check other task's type
            assert isinstance(other, ObservationOpportunity), "can only merge with tasks of type `SpecificObservationTask`."

            # Merge parent tasks
            merged_parent_tasks = {task for task in self.tasks}
            merged_parent_tasks.update({task for task in other.tasks})

            # Check instrument compatibility
            assert self.instrument_name == other.instrument_name, "tasks pertain to observations with different instruments."

            # Calculate accessibility overlap and duration requirements
            merged_accessibility, min_duration_req = self._calc_time_requirements(other, must_overlap)
            assert isinstance(merged_accessibility, Interval), "merged accessibility is not an Interval."
            assert isinstance(min_duration_req, (float,int)), "minimum duration requirement is not a number."
            assert not merged_accessibility.is_empty(), "joint task availability is empty."
            assert not math.isnan(min_duration_req) , "minimum duration requirement is invalid."
            assert min_duration_req <= max_duration, "minimum duration requirements exceed maximum allowed duration."
            assert min_duration_req <= merged_accessibility.span(), "minimum duration requirements exceed accessibility span."

            # Calculate slew angles overlap
            merged_slew_angles : Interval = self.slew_angles.intersection(other.slew_angles) 
            assert not merged_slew_angles.is_empty(), "slew angles do not overlap."
        
            # Return merged task
            # TODO which id should we be using for new tasks? Ref task clustering in planners
            return ObservationOpportunity(merged_parent_tasks, self.instrument_name, merged_accessibility, min_duration_req, merged_slew_angles, self.id) 
        
        except AssertionError as e:
            raise AssertionError(f"Cannot merge tasks; {e}")

    def __repr__(self):
        return f"SpecificObservationTask_{self.id.split('-')[0]}"
            
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tasks": [task.to_dict() for task in self.tasks],
            "instrument_name": self.instrument_name,
            "accessibility": self.accessibility.to_dict(),
            "min_duration": self.min_duration,
            "slew_angles": self.slew_angles.to_dict()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ObservationOpportunity":
        return cls(
            tasks={GenericObservationTask.from_dict(task) for task in data["tasks"]},
            instrument_name=data["instrument_name"],
            accessibility=Interval.from_dict(data["accessibility"]),
            min_duration=data["min_duration"],
            slew_angles=Interval.from_dict(data["slew_angles"]),
            id=data["id"]
        )
    
    def __eq__(self, other : 'ObservationOpportunity') -> bool:
        assert isinstance(other, ObservationOpportunity), "Can only compare with another SpecificObservationTask."
        return self.to_dict() == other.to_dict()
    
    def __hash__(self):
        return hash(self.id)