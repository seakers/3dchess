import uuid
import numpy as np
from sympy import Interval

from chess3d.agents.actions import ObservationAction

class ObservationTask:
    def __init__(self, 
                 instrument_name : str,
                 time_interval : Interval,
                 slew_angles : Interval,
                 targets : list,
                 reward : float,
                 id: str = None,
                 max_duration: float = np.Inf,
                 ):
        """ Represents an observation task in a planning system. """

        self.instrument_name = instrument_name              # name of instrument being used
        self.time_interval = time_interval                  # start and end time of the observation task
        self.slew_angles = slew_angles                      # feasible slew angles for the instrument at the time of observation
        self.targets = targets                              # list of target ground points to be observed        
        self.reward = reward                                # reward associated with the observation task
        self.max_duration = max_duration                    # maximum duration of the observation task
        self.id = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1()) # unique identifier for the task

    def can_cluster(self, other_task : object) -> bool:
        """ Check if two tasks can be clustered based on their time and slew angle. """
        try:
        
            # Check if the other task is an instance of ObservationTask
            if not isinstance(other_task, ObservationTask):
                raise ValueError("The other task must be an instance of Task.")
            
            # Check if the tasks have the same instrument name
            if self.instrument_name != other_task.instrument_name:
                raise ValueError("Tasks must have the same instrument name to be clustered.")

            if self.id == other_task.id:
                return False

            # Check if the time intervals overlap
            time_overlap : Interval = self.time_interval.intersection(other_task.time_interval)

            if not time_overlap.is_empty:
                # Check if the time intervals are within the maximum duration
                if (time_overlap.measure) > min(self.max_duration, other_task.max_duration):
                    return False

            # Check if the slew angles overlap
            slew_angle_overlap : Interval = self.slew_angles.intersection(other_task.slew_angles) 

            return not (time_overlap.is_empty or slew_angle_overlap.is_empty)
        except Exception as e:
            x = 1

    def to_observation_action(self) -> ObservationAction:
        """ Convert the task to an observation action. """
        
        # calculate look angle and action duration
        targets = [target for target in self.targets]
        look_angle = (self.slew_angles.start + self.slew_angles.end) / 2
        duration = self.time_interval.end - self.time_interval.start
        
        # Create an ObservationAction object
        return ObservationAction(self.instrument_name, targets, look_angle, self.time_interval.start, duration)
    
if __name__ == "__main__":
    # Example usage
    # task = Task("Sample Task", "This is a sample task description.")
    x = 1
    # print(f"Task Name: {task.name}")
    # print(f"Task Description: {task.description}")
    # task.execute()  # This will raise NotImplementedError