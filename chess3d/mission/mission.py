
from typing import  Dict, Union

from chess3d.agents.planning.tasks import *
from chess3d.agents.states import SimulationAgentState
from chess3d.mission.objectives import *

class Mission:
    def __init__(self, name : str, objectives: list, normalizing_parameter : float = 1e-9):
        # Validate inputs
        assert isinstance(name, str), "Mission name must be a string"
        assert len(objectives) > 0, "At least one objective is needed"
        assert all(isinstance(obj, MissionObjective) for obj in objectives), "All objectives must be instances of `Objective`"

        # Set attributes
        self.name : str = name.lower()
        self.objectives : list[MissionObjective] = objectives
        self.normalizing_parameter : float = normalizing_parameter

    # def calc_measurement_performance(self, measurements: dict, objectives : list) -> float:
    #     """Evaluate the performance of the mission based on the measurements."""
    #     # Sum weighted objective scores across all objectives
    #     default_performance = sum([obj.weight * obj.calc_requirement_satisfaction(measurements)
    #                                for obj in objectives
    #                                if not isinstance(obj, DefaultMissionObjective)])

    #     event_performance = sum([obj.weight * obj.calc_requirement_satisfaction(measurements)
    #                              for obj in objectives
    #                              if isinstance(obj, EventDrivenObjective)])
        
    #     return default_performance + event_performance

    def calc_task_utility(self, task : GenericObservationTask, measurement: dict, prev_state : SimulationAgentState) -> float:
        """Calculate the utility of a task based on the mission's objectives and the measurement."""
        return self.calc_task_value(task, measurement) - self.normalizing_parameter * self.calc_task_cost(task, prev_state)

    def calc_task_value(self, task: GenericObservationTask, measurement : dict) -> float:
        """Calculate the value of a task based on the mission's objectives."""
        assert isinstance(task, GenericObservationTask), "Task must be an instance of `GenericObservationTask`"
        assert isinstance(measurement, dict), "Measurement must be a dictionary"

        # Maps objectives to their relevance to the task at hand
        obj_relevances : Dict[MissionObjective, float] = self.relate_objectives_to_task(task)

        # Calculate the value of the task based on the objectives and their relevance
        values = [objective.priority * self.calc_measurement_task_performance(task, 
                                                                              measurement, 
                                                                              objective, 
                                                                              obj_relevances[objective])
                 for objective in self.objectives]
        
        # Return the sum of values for all objectives
        return sum(values)
    
    def calc_measurement_task_performance(self, 
                                          task: GenericObservationTask, 
                                          measurement: dict, 
                                          objective: MissionObjective,
                                          relevance : float) -> float:
        """Calculate the performance of a measurement based on the task and objective."""
        assert isinstance(task, GenericObservationTask), "Task must be an instance of `GenericObservationTask`"
        assert isinstance(measurement, dict), "Measurement must be a dictionary"
        assert isinstance(objective, MissionObjective), "Objective must be an instance of `MissionObjective`"

        if isinstance(task, DefaultMissionTask):
            return relevance *  objective.calc_requirement_satisfaction(measurement)
        elif isinstance(task, EventObservationTask):
            return task.severity * relevance *  objective.calc_requirement_satisfaction(measurement)
        else:
            raise ValueError("Unsupported task type for performance calculation")

    def relate_objectives_to_task(self, task: GenericObservationTask) -> Dict[MissionObjective, float]:
        """Relate objectives to a task based on the task's parameters."""
        assert isinstance(task, GenericObservationTask), "Task must be an instance of `GenericObservationTask`"

        # TODO # Implement logic to relate objectives to the task based on the task's parameters
        # For now, return a dummy dictionary with all objectives having equal relevance
        # This should be replaced with actual logic based on the task's parameters and objectives
        if not self.objectives:
            return {}
        return {obj: 1.0 for obj in self.objectives}

    def calc_task_cost(self, task: GenericObservationTask, prev_state: SimulationAgentState) -> float:
        """Calculate the cost of a task based on the previous state."""
        assert isinstance(task, GenericObservationTask), "Task must be an instance of `GenericObservationTask`"
        assert isinstance(prev_state, SimulationAgentState), "Previous state must be an instance of `SimulationAgentState`"

        # TODO # Implement logic to calculate the cost of the task based on the previous state
        # For now, return a dummy cost value
        return 0.0  # Replace with actual cost calculation logic

    def tasks_from_event(self, event: GeophysicalEvent) -> list[GenericObservationTask]:
        """Generate tasks based on a geophysical event."""
        assert isinstance(event, GeophysicalEvent), "Event must be an instance of `GeophysicalEvent`"
    
        # TODO event task from objective and events
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


    # TODO event task from objective and events

    # Include Rel_{jk} for mapping tasks to objectives in the definition or in the evaluation of tasks

    # @abstractmethod
    # def calc_measurement_performance(self, measurement: dict) -> float:
    #     """Calculate the performance of the measurement based on this objective's requirements."""
