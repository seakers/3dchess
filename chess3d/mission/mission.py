
from typing import  Dict, Union

from chess3d.agents.states import SatelliteAgentState, SimulationAgentState
from chess3d.agents.planning.tasks import GenericObservationTask, DefaultMissionTask, EventObservationTask
from chess3d.agents.planning.observations import ObservationOpportunity
from chess3d.mission.objectives import *

class Mission:
    def __init__(self, 
                 name : str, 
                 objectives: List[MissionObjective], 
                 weights : List[float]
                 ):
        # Validate inputs
        assert isinstance(name, str), "Mission name must be a string"
        assert len(objectives) > 0, "At least one objective is needed"
        assert all(isinstance(obj, MissionObjective) for obj in objectives), "All objectives must be instances of `Objective`"
        assert len(objectives) == len(weights), "Objectives and weights must have the same length"
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"

        # Set attributes
        self.name : str = name.lower()
        self.objectives : Dict[MissionObjective, float] = {o: w for o, w in zip(objectives, weights)}

    def calc_observation_opportunity_utility(self, obs : ObservationOpportunity, measurement: dict, norm_param : float = 1e-6) -> float:
        """Calculate the utility of a task based on the mission's objectives and the measurement."""
        
        # Validate inputs
        assert isinstance(obs, ObservationOpportunity), "Task must be an instance of `SpecificObservationTask`"
        assert isinstance(measurement, dict), "Measurement must be a dictionary"
        assert isinstance(norm_param, (int,float)) and norm_param >= 0, "Normalizing parameter must be a positive value"

        # Calculate utility = specific_task_value - norm * task_cost
        return self.calc_observation_opportunity_value(obs, measurement) - norm_param * self.calc_observation_cost(obs)

    def calc_observation_opportunity_value(self, task: ObservationOpportunity, measurement: dict) -> float:
        """Calculate the utility of a specific observation task based on the mission's objectives and the measurement."""

        # Validate inputs
        assert isinstance(task, ObservationOpportunity), "Task must be an instance of `SpecificObservationTask`"
        assert isinstance(measurement, dict), "Measurement must be a dictionary"        

        # Calculate the value of a specific task by summing the value of parent tasks
        return sum([self.calc_task_value(gen_task, measurement) for gen_task in task.tasks])

    def calc_task_value(self, task: GenericObservationTask, measurement : dict) -> float:
        """Calculate the value of a task based on the mission's objectives."""
        assert isinstance(task, GenericObservationTask), "Task must be an instance of `GenericObservationTask`"
        assert isinstance(measurement, dict), "Measurement must be a dictionary"

        # Maps objectives to their relevance to the task at hand
        obj_relevances : Dict[MissionObjective, float] = self.relate_objectives_to_task(task)

        # Calculate the value of the task based on the objectives and their relevance
        values = [weight * obj_relevances[objective] * objective.eval_measurement_performance(measurement)
                 for objective, weight in self.objectives.items()]
        
        # Return the sum of values for all objectives
        return task.priority * sum(values)

    def relate_objectives_to_task(self, task: GenericObservationTask) -> Dict[MissionObjective, float]:
        """Relate objectives to a task based on the task's parameters."""
        # TODO Allow for more complex relationships using Knowledge Graphs or other methods. Move to science module?
        
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
            obj: (1.0 if isinstance(obj, type_map[type(task)]) else 0.25)
            if obj.parameter == task.parameter else 0.0
            for obj in self.objectives
        }

        # Validate outputs
        assert all(0 <= val <= 1 for val in obj_relevances.values()), "Objective relevance values must be between 0 and 1"

        return obj_relevances

    def calc_observation_cost(self, obs: ObservationOpportunity) -> float:
        """Calculate the intrinsic cost of a task based on the previous state."""
        
        # Validate Inputs
        assert isinstance(obs, ObservationOpportunity), "Task must be an instance of `SpecificObservationTask`"

        # Calculate the cost of a specific task by summing the cost of parent tasks
        costs = [self.calc_task_cost(task) for task in obs.tasks]

        # return the sum of costs for all objectives
        return sum(costs)
    
    def calc_task_cost(self, task: GenericObservationTask) -> float:
        """Calculate the intrinsic cost of a task."""
        # TODO Define task cost model

        # Validate Inputs
        assert isinstance(task, GenericObservationTask), "Task must be an instance of `GenericObservationTask`"

        # For now, return 0.0 as a placeholder
        return 0.0

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

    @classmethod
    def from_dict(cls, data: Dict[str, Union[str, float]]) -> 'Mission':
        """Create a mission from a dictionary."""

        assert isinstance(data, dict), "Input must be a dictionary"
        assert 'name' in data, "Name is a required field"
        assert 'objectives' in data, "Objectives are a required field"

        return cls(
            name=data.get("name", ""),
            objectives=[MissionObjective.from_dict(obj) 
                        for obj in data.get("objectives", [])],
            normalizing_parameter=data.get("normalizing_parameter", None)
        )