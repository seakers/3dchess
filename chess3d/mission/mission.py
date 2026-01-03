
from typing import  Dict

from chess3d.agents.planning.tasks import GenericObservationTask, DefaultMissionTask, EventObservationTask
from chess3d.agents.planning.observations import ObservationOpportunity
from chess3d.mission.objectives import *
from chess3d.mission.attributes import TemporalRequirementAttributes

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

        # Calculate utility = value - cost
        return self.calc_observation_opportunity_value(obs, measurement) - self.calc_measurement_cost(measurement, norm_param)

    def calc_observation_opportunity_value(self, obs: ObservationOpportunity, measurement: dict) -> float:
        """Calculate the utility of a specific observation opportunity based on its predicted performance compared to the mission's objectives."""

        # Validate inputs
        assert isinstance(obs, ObservationOpportunity), "Task must be an instance of `SpecificObservationTask`"
        assert isinstance(measurement, dict), "Measurement must be a dictionary"        

        # Calculate the value of a specific task by summing the value of each task
        values = [self.calc_task_value(task, measurement) for task in obs.tasks]

        # return sum of values
        return sum(values)

    def calc_task_value(self, task: GenericObservationTask, measurement : dict) -> float:
        """Calculate the value of a task based on the mission's objectives."""
        # Validate inputs
        assert isinstance(task, GenericObservationTask), "Task must be an instance of `GenericObservationTask`"
        assert isinstance(measurement, dict), "Measurement must be a dictionary"
        assert TemporalRequirementAttributes.OBS_TIME.value in measurement, "Measurement must contain 't_img [s]' key for observation time"

        # Maps objectives to their relevance to the task at hand
        obj_relevances : Dict[MissionObjective, float] = self.relate_objectives_to_task(task)

        # Check for availability of measurement at observation time
        if measurement.get(TemporalRequirementAttributes.OBS_TIME.value) not in task.availability: return 0.0

        # Clip duration to task availability if applicable
        if TemporalRequirementAttributes.DURATION.value in measurement:
            d_prev = measurement[TemporalRequirementAttributes.DURATION.value]
            measurement[TemporalRequirementAttributes.DURATION.value] = min(
                measurement[TemporalRequirementAttributes.DURATION.value], 
                task.availability.right - measurement.get(TemporalRequirementAttributes.OBS_TIME.value)
            )

        # Calculate the value of the task based on the objectives and their relevance
        task_values = {objective : [
                            weight,                                             # weight of the objective
                            obj_relevances[objective],                          # relevance of the objective to the task
                            objective.eval_measurement_performance(measurement),# performance of the measurement for the objective
                        ]
                for objective, weight in self.objectives.items()}
        
        # Restore original duration if it was modified
        if TemporalRequirementAttributes.DURATION.value in measurement and 'd_prev' in locals():
            measurement[TemporalRequirementAttributes.DURATION.value] = d_prev
    
        # Return the sum of values for all objectives times the task priority
        return task.priority * sum([np.prod(values) for values in task_values.values()])

    def relate_objectives_to_task(self, task: GenericObservationTask) -> Dict[MissionObjective, float]:
        """Relate objectives to a task based on the task's parameters."""
        # TODO Allow for more complex relationships using Knowledge Graphs or other methods. Move to science module?
        
        # Validate task type
        assert isinstance(task, GenericObservationTask), "Task must be an instance of `GenericObservationTask`"

        # initiate objective-to-task relevance mapping
        obj_relevances = dict()

        # Check if task has a defined objective
        if task.objective is None:
            # No specific objective; relate based on parameter matching
            for obj in self.objectives:                
                if obj.parameter == task.parameter:
                    if (isinstance(task, DefaultMissionTask) and isinstance(obj, DefaultMissionObjective)) or \
                       (isinstance(task, EventObservationTask) and isinstance(obj, EventDrivenObjective)):
                        # if same parameter and objective type matches task type -> 0.50
                        obj_relevances[obj] = 0.50
                    else:
                        # if same parameter but objective type does not match task type -> 0.25
                        obj_relevances[obj] = 0.25
                else:
                    # else -> 0.0
                    obj_relevances[obj] = 0.0
        else:
            # Task has a specific objective defined; directly relate task objective to mission objectives
            for obj in self.objectives:
                if obj == task.objective:
                    # if same objective -> 1.0
                    obj_relevances[obj] = 1.0

                elif type(obj) == type(task.objective) and obj.parameter == task.objective.parameter:
                    # if different objective but of the same kind and for the same parameter -> 0.75
                    obj_relevances[obj] = 0.75

                elif obj.parameter == task.objective.parameter:
                    # if different objective and different kind but for the same parameter -> 0.50
                    obj_relevances[obj] = 0.50

                elif type(obj) == type(task.objective):
                    # if different objective and of for different parameters but the same kind -> 0.25
                    obj_relevances[obj] = 0.25

                else:
                    # else it must be a different objective with different parameter and type -> 0.0
                    obj_relevances[obj] = 0.0
            

        # Validate outputs
        assert all(0 <= val <= 1 for val in obj_relevances.values()), "Objective relevance values must be between 0 and 1"

        # Return objective relevances
        return obj_relevances

    def calc_measurement_cost(self, measurement: dict, norm_param: float) -> float:
        """Calculate the intrinsic cost of a measurement."""
        
        # Validate Inputs
        assert isinstance(measurement, dict), "Measurement must be a dictionary"

        # Calculate the cost of a specific task by summing the cost of parent tasks
        # TODO Define cost model; currently using measurement duration as a placeholder
        return norm_param * measurement.get(TemporalRequirementAttributes.DURATION.value, 0.0)        
    
    def __repr__(self):
        """String representation of the mission."""
        return f"Mission(name='{self.name}', n_objectives={len(self.objectives)})"
 
    def __iter__(self):
        """Iterate over the objectives."""
        return iter(self.objectives)
    
    def copy(self) -> 'Mission':
        """Create a copy of the mission."""
        return Mission.from_dict(self.to_dict())
    
    def to_dict(self) -> dict:
        """Convert the mission to a dictionary."""
        d = dict(self.__dict__)
        d['objectives'] = [
            {
                **obj.to_dict(),
                "weight": self.objectives[obj]
            } for obj in self.objectives
        ]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'Mission':
        """Create a mission from a dictionary."""

        # validate inputs
        assert isinstance(d, dict), "Input must be a dictionary"
        assert 'name' in d, "Name is a required field"
        assert 'objectives' in d, "Objectives are a required field"
        assert all("weight" in obj for obj in d.get("objectives")), "All objectives must have a weight field"

        # unpack dictionary
        objective_dicts : List[Dict] = d.get("objectives")
        objectives = [MissionObjective.from_dict(obj) 
                        for obj in objective_dicts]
        weights = [obj.get("weight") for obj in objective_dicts]

        # return mission instance
        return cls(
            name=d.get("name"),
            objectives=objectives,
            weights=weights
        )
    
    def __eq__(self, value : 'Mission') -> bool:
        assert isinstance(value, Mission), "Can only compare Mission instances"
        return self.to_dict() == value.to_dict()