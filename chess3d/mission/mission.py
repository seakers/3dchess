
from typing import  Dict, Union

from chess3d.agents.states import SatelliteAgentState, SimulationAgentState
from chess3d.agents.planning.tasks import *
from chess3d.mission.objectives import *

class Mission:
    def __init__(self, name : str, objectives: list, normalizing_parameter : float = 1e-6):
        # Validate inputs
        assert isinstance(name, str), "Mission name must be a string"
        assert len(objectives) > 0, "At least one objective is needed"
        assert all(isinstance(obj, MissionObjective) for obj in objectives), "All objectives must be instances of `Objective`"

        # Set attributes
        self.name : str = name.lower()
        self.objectives : list[MissionObjective] = [obj for obj in objectives]
        self.normalizing_parameter : float = normalizing_parameter

    def calc_task_utility(self, task : SpecificObservationTask, measurement: dict, prev_state : SimulationAgentState) -> float:
        """Calculate the utility of a task based on the mission's objectives and the measurement."""
        
        # Validate inputs
        assert isinstance(task, SpecificObservationTask), "Task must be an instance of `SpecificObservationTask`"
        assert isinstance(measurement, dict), "Measurement must be a dictionary"
        assert isinstance(prev_state, SimulationAgentState), "Previous state must be an instance of `SimulationAgentState`"

        # Calculate utility = specific_task_value - norm * task_cost
        return self.calc_specific_task_value(task, measurement) - self.normalizing_parameter * self.calc_task_cost(task, prev_state)

    def calc_specific_task_value(self, task: SpecificObservationTask, measurement: dict) -> float:
        """Calculate the utility of a specific observation task based on the mission's objectives and the measurement."""

        # Validate inputs
        assert isinstance(task, SpecificObservationTask), "Task must be an instance of `SpecificObservationTask`"
        assert isinstance(measurement, dict), "Measurement must be a dictionary"        
    
        # Calculate the value of a specific task by summing the value of parent tasks
        return sum([self.calc_task_value(gen_task, measurement) for gen_task in task.parent_tasks])

    def calc_task_value(self, task: GenericObservationTask, measurement : dict) -> float:
        """Calculate the value of a task based on the mission's objectives."""
        assert isinstance(task, GenericObservationTask), "Task must be an instance of `GenericObservationTask`"
        assert isinstance(measurement, dict), "Measurement must be a dictionary"

        # Maps objectives to their relevance to the task at hand
        obj_relevances : Dict[MissionObjective, float] = self.relate_objectives_to_task(task)

        # Calculate the value of the task based on the objectives and their relevance
        values = [objective.weight * obj_relevances[objective] * objective.eval_measurement_performance(measurement)
                 for objective in self.objectives]
        
        # Return the sum of values for all objectives
        return task.priority * sum(values)

    def relate_objectives_to_task(self, task: GenericObservationTask) -> Dict[MissionObjective, float]:
        """Relate objectives to a task based on the task's parameters."""
        # TODO move to science module? Allow for more complex relationships using Knowledge Graphs or other methods
        
        # Validate task type
        assert isinstance(task, GenericObservationTask), "Task must be an instance of `GenericObservationTask`"

        # Define mapping of task types to objective types
        type_map = {
            DefaultMissionTask: DefaultMissionObjective,
            EventObservationTask: EventDrivenObjective
        }

        # Check if task type is supported
        assert type(task) in type_map, f"Task type {type(task).__name__} not supported for objective relation"
        
        # Initialize relevances
        obj_relevances = {
            obj: (1.0 if isinstance(obj, type_map[type(task)]) else 0.5)
            if obj.parameter == task.parameter else 0.0
            for obj in self.objectives
        }

        # Validate outputs
        assert all(0 <= val <= 1 for val in obj_relevances.values()), "Objective relevance values must be between 0 and 1"

        return obj_relevances

    def calc_task_cost(self, task: SpecificObservationTask, prev_state: SimulationAgentState) -> float:
        """Calculate the cost of a task based on the previous state."""
        assert isinstance(task, SpecificObservationTask), "Task must be an instance of `SpecificObservationTask`"
        assert isinstance(prev_state, SimulationAgentState), "Previous state must be an instance of `SimulationAgentState`"

        if not isinstance(prev_state, SatelliteAgentState):
            raise NotImplementedError("Cost calculation is currently only implemented for `SatelliteAgentState`")

        th_i : float = np.average(task.slew_angles.left, task.slew_angles.right)
        th_prev : float = prev_state.attitude[0]

        return np.abs(th_i - th_prev)

    def tasks_from_event(self, event: GeophysicalEvent) -> List[GenericObservationTask]:
        """Generate tasks based on a geophysical event."""
        assert isinstance(event, GeophysicalEvent), "Event must be an instance of `GeophysicalEvent`"
    
        # TODO event task from objective and events
        raise NotImplementedError("Task generation from event is not implemented yet")

    def task_from_event(self, event: GeophysicalEvent) -> GenericObservationTask:
        """Generate a task from a geophysical event."""
        assert isinstance(event, GeophysicalEvent), "Event must be an instance of `GeophysicalEvent`"
        
        # TODO event task from event and mission objectives
        raise NotImplementedError("Task generation from event is not implemented yet")

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