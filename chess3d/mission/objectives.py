
from typing import Dict, Union
import uuid
import numpy as np

from chess3d.mission.requirements import *


class MissionObjective(ABC):
    DEFAULT = "default_mission"
    EVENT = "event_driven"

    def __init__(self, 
                 objective_type: str,
                 parameter: str, 
                 priority: float, 
                 requirements: list, 
                 id : str = None):
        """ 
        ### Objective
         
        Initialize an objective with a priority, parameter, and requirements.
        - :`parameter`: The primary geophysical parameter to be measured (e.g., "Chl-A concentration").
        - :`priority`: The priority of the objective.
        - :`requirements`: A list of `MeasurementRequirement` instances that define the requirements for the objective.
        - :`id`: An optional ID for the objective. If None, a new UUID is generated.
        """

        # Validate inputs
        assert isinstance(objective_type, str), "Objective type must be a string"
        assert objective_type in [self.DEFAULT, self.EVENT], f"Objective type must be one of {self.DEFAULT} or {self.EVENT}"
        assert isinstance(parameter, str), "Parameter must be a string"
        assert isinstance(priority, (int, float)), "Priority must be a number"
        assert len(requirements) > 0, "At least one requirement is needed"
        assert all(isinstance(req, MissionRequirement) for req in requirements), "All requirements must be instances of `MeasurementRequirement`"
        assert any(isinstance(req, TemporalRequirement) for req in requirements), "At least one requirement must be a `TemporalRequirement`"
        assert any(isinstance(req, SpatialRequirement) for req in requirements), "At least one requirement must be a `SpatialRequirement`"
        assert isinstance(id, str) or id is None, f"ID must be a string or None. is of type {type(id)}"

        # Set attributes
        self.objective_type : str = objective_type.lower()
        self.priority : float = priority
        self.parameter : str = parameter
        self.requirements : list[MissionRequirement] = [requirement for requirement in requirements]
        self.id = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())

    def calc_requirement_satisfaction(self, measurement: dict) -> float:
        """Calculate the satisfaction score for the objective based on the preference scores of the measurement to the objective's requirements."""

        # Validate measurement input
        assert isinstance(measurement, dict), "Measurement must be a dictionary"

        assert all(req.attribute in measurement for req in self.requirements),\
            f"Measurement must contain all required attributes: {[req.attribute for req in self.requirements]}"

        return np.prod([req.calc_preference_value(measurement[req.attribute])
                        for req in self.requirements])   
    
    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert the objective to a dictionary."""
        return {
            "objective_type": self.objective_type,
            "parameter": self.parameter,
            "priority": self.priority,
            "requirements": [req.to_dict() for req in self.requirements],
            "id": self.id
        }

    @classmethod
    def from_dict(cls, obj_dict: Dict[str, Union[str, float]]) -> 'MissionObjective':
        """Create an objective from a dictionary."""
        assert 'objective_type' in obj_dict, "Objective type must be specified in the dictionary"
        
        if obj_dict['objective_type'] == cls.EVENT:
            # EventDrivenObjective
            return EventDrivenObjective.from_dict(obj_dict)
        
        elif obj_dict['objective_type'] == cls.DEFAULT:
            # DefaultMissionObjective
            return DefaultMissionObjective.from_dict(obj_dict)
        
        raise ValueError(f"Unknown objective type: {obj_dict['objective_type']}")

    @abstractmethod
    def copy(self) -> 'MissionObjective':
        """Create a copy of the objective."""
    
    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the objective."""

class DefaultMissionObjective(MissionObjective):
    def __init__(self, 
                 parameter: str, 
                 priority: float, 
                 requirements: list, 
                 id : str = None):
        """ 
        ### Monitoring Objective
         
        Initialize a monitoring objective with a priority, parameter, and requirements.
        - :`parameter`: The primary geophysical parameter to be measured (e.g., "Chl-A concentration").
        - :`priority`: The priority of the objective.
        - :`requirements`: A list of `MeasurementRequirement` instances that define the requirements for the objective.
        - :`id`: An optional ID for the objective. If None, a new UUID is generated.
        """
        # Validate inputs
        if not any(isinstance(req, SpatialRequirement) for req in requirements):
            print("WARNING:No spatial requirement found, adding default grid target spatial requirement.")
            requirements.append(GridTargetSpatialRequirement('grid', 0, 1))
        if not any(isinstance(req, TemporalRequirement) for req in requirements):
            print("WARNING: No temporal requirement found, adding default temporal requirement.")
            requirements.append(RevisitTemporalRequirement([3600, 3600*4, 24*3600], [1, 0.5, 0.0]))

        super().__init__(MissionObjective.DEFAULT, parameter, priority, requirements, id)

    def copy(self) -> 'DefaultMissionObjective':
        """Create a copy of the objective."""
        return DefaultMissionObjective(self.parameter, 
                                       self.priority, 
                                       [req.copy() for req in self.requirements], 
                                       self.id)

    def __repr__(self) -> str:
        """String representation of the objective."""
        return f"DefaultMissionObjective({self.parameter}, priority={self.priority}, requirements={self.requirements})"

    @classmethod
    def from_dict(cls, obj_dict: Dict[str, Union[str, float]]) -> 'DefaultMissionObjective':
        """Create a default mission objective from a dictionary."""
        assert 'objective_type' in obj_dict and obj_dict['objective_type'] == MissionObjective.DEFAULT, "Objective type must be 'default' for DefaultMissionObjective"
        assert 'parameter' in obj_dict, "Parameter must be specified in the dictionary"
        assert 'priority' in obj_dict, "Priority must be specified in the dictionary"
        assert 'requirements' in obj_dict, "Requirements must be specified in the dictionary"

        # Convert requirements to MissionRequirement instances
        if all(isinstance(req, dict) for req in obj_dict['requirements']):
            requirements = [MissionRequirement.from_dict(req) for req in obj_dict['requirements']]
        elif all(isinstance(req, MissionRequirement) for req in obj_dict['requirements']):
            requirements = obj_dict['requirements']
        else:
            raise ValueError("Requirements must be a list of dictionaries or `MissionRequirement` instances")

        id = obj_dict.get('id', None)

        return cls(obj_dict['parameter'], obj_dict['priority'], requirements, id)

class EventDrivenObjective(MissionObjective):
    def __init__(self, 
                 event_type: str,
                 parameter: str, 
                 priority: float, 
                 requirements: list, 
                 synergistic_parameters: list = [],
                 id : str = None
                 ):
        """ 
        ### Event Driven Objective
         
        Initialize an event-driven objective with a priority, parameter, and requirements.
        - :`event_type`: The type of geophysical event associated with the objective.
        - :`parameter`: The primary geophysical parameter to be measured (e.g., "Chl-A concentration").
        - :`priority`: The priority of the objective.
        - :`requirements`: A list of `MeasurementRequirement` instances that define the requirements for the objective.
        - :`synergistic_parameters`: A list of additional parameters that are synergistic with the main parameter.
        - :`id`: An optional ID for the objective. If None, a new UUID is generated.
        """
        # Initialize the parent class
        super().__init__(MissionObjective.EVENT, parameter, priority, requirements, id)
        
        # Validate inputs
        assert isinstance(event_type, str), "Event type must be a string"
        assert isinstance(synergistic_parameters, list), "Synergistic parameters must be a list"
        assert all(isinstance(param, str) for param in synergistic_parameters), "Synergistic parameters must be strings"
        assert parameter not in synergistic_parameters, "Main parameter cannot be in list of synergistic parameters."
        
        # Set attributes
        self.event_type = event_type.lower() 
        self.synergistic_parameters = [param.lower() for param in synergistic_parameters]

    def copy(self):
        return EventDrivenObjective(self.event_type, 
                                    self.parameter, 
                                    self.priority, 
                                    [req.copy() for req in self.requirements], 
                                    self.synergistic_parameters, 
                                    self.id)
    
    def __repr__(self):
        return f"EventDrivenObjective({self.parameter}, priority={self.priority}, event_type={self.event_type}, synergistic_parameters={self.synergistic_parameters}, requirements={self.requirements})"

    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert the objective to a dictionary."""
        d = super().to_dict()
        d.update({
            "event_type": self.event_type,
            "synergistic_parameters": self.synergistic_parameters
        })
        return d

    @classmethod
    def from_dict(cls, obj_dict: Dict[str, Union[str, float]]) -> 'EventDrivenObjective':
        """Create an event-driven objective from a dictionary."""
        assert 'objective_type' in obj_dict and obj_dict['objective_type'] == MissionObjective.EVENT, "Objective type must be 'event' for EventDrivenObjective"
        assert 'parameter' in obj_dict, "Parameter must be specified in the dictionary"
        assert 'priority' in obj_dict, "Priority must be specified in the dictionary"
        assert 'requirements' in obj_dict, "Requirements must be specified in the dictionary"
        assert 'event_type' in obj_dict, "Event type must be specified in the dictionary"
        
        # Convert requirements to MissionRequirement instances
        if all(isinstance(req, dict) for req in obj_dict['requirements']):
            requirements = [MissionRequirement.from_dict(req) for req in obj_dict['requirements']]
        elif all(isinstance(req, MissionRequirement) for req in obj_dict['requirements']):
            requirements = obj_dict['requirements']
        else:
            raise ValueError("Requirements must be a list of dictionaries or `MissionRequirement` instances")
        
        return EventDrivenObjective(event_type=obj_dict['event_type'],
                                    parameter=obj_dict['parameter'],
                                    priority=obj_dict['priority'],
                                    requirements=requirements,
                                    synergistic_parameters=obj_dict.get('synergistic_parameters', []),
                                    id=obj_dict.get('id', None))
