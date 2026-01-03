
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Union
import uuid
import numpy as np

from chess3d.mission.events import GeophysicalEvent
from chess3d.mission.requirements import MissionRequirement, PerformanceRequirement, SpatialCoverageRequirement

class MissionObjective(ABC):
    DEFAULT = "default_mission"
    EVENT = "event_driven"

    def __init__(self, 
                 objective_type: str,
                 parameter: str, 
                 requirements: List[MissionRequirement], 
                 id : str = None):
        """ 
        ### Objective
         
        Initialize an objective with a parameter, weight, and requirements.
        - :`parameter`: The primary geophysical parameter to be measured (e.g., "Chl-A concentration").
        - :`weight`: The relative objective weight.
        - :`requirements`: A list of `MissionRequirement` instances that define the requirements for the objective.
        - :`id`: An optional ID for the objective. If None, a new UUID is generated.
        """

        # Validate inputs
        assert isinstance(objective_type, str), "Objective type must be a string"
        assert objective_type in [self.DEFAULT, self.EVENT], f"Objective type must be one of {self.DEFAULT} or {self.EVENT}"
        assert isinstance(parameter, str), "Parameter must be a string"
        assert len(requirements) > 0, "At least one requirement is needed"
        assert all(isinstance(req, MissionRequirement) for req in requirements), "All requirements must be instances of `MeasurementRequirement`"
        assert isinstance(id, str) or id is None, f"ID must be a string or None. is of type {type(id)}"

        # Set attributes
        self.objective_type : str = objective_type.lower()
        self.parameter : str = parameter.lower()
        self.requirements : Dict[str, MissionRequirement] = {requirement.attribute: requirement for requirement in requirements}
        self.id = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())

    def eval_measurement_performance(self, measurement: dict) -> float:
        """Calculate the satisfaction score for the objective based on the preference scores of the measurement to the objective's requirements."""

        # Validate measurement input
        assert isinstance(measurement, dict), "Measurement must be a dictionary"
        assert all(isinstance(k, str) for k in measurement.keys()), "Measurement keys must be strings"
        assert all(attribute in measurement for attribute in self.requirements.keys()), "Measurement must contain all requirement attributes"

        # Evaluate measurement performance for each requirement attribute
        pref_values = [
            req.calc_preference(attribute, measurement[attribute]) 
            for attribute,req in self.requirements.items()
        ]

        # Return product of all preference values
        return np.prod(pref_values)

    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert the objective to a dictionary."""
        d = dict(self.__dict__)
        d['requirements'] = [req.to_dict() for req in self.requirements.values()]
        return d

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

    def copy(self) -> 'MissionObjective':
        """Create a copy of the objective."""
        return self.from_dict(self.to_dict())
    
    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the objective."""

    def __iter__(self) -> 'Iterator[MissionRequirement]':
        """Iterate over the objectives."""
        return iter(self.requirements.values())
    
    def __eq__(self, other):
        """Check equality of two objectives."""
        if not isinstance(other, MissionObjective):
            return False
        return self.to_dict() == other.to_dict()
    
    def __hash__(self):
        return hash(self.id)

class DefaultMissionObjective(MissionObjective):
    def __init__(self, 
                 parameter: str, 
                 requirements: list = [], 
                 id : str = None
                 ):
        """ 
        ### Monitoring Objective
         
        Initialize a monitoring objective with a weight, parameter, and requirements.
        - :`parameter`: The primary geophysical parameter to be measured (e.g., "Chl-A concentration").
        - :`weight`: The weight of the objective.
        - :`requirements`: A list of `MeasurementRequirement` instances that define the requirements for the objective.
        - :`id`: An optional ID for the objective. If None, a new UUID is generated.
        """
        # Validate inputs
        # if not any(isinstance(req, SpatialCoverageRequirement) for req in requirements):
        #     raise ValueError("No spatial requirement found, please add a spatial coverage requirement to objective definition.")
        if not any(isinstance(req, PerformanceRequirement) for req in requirements):
            raise ValueError("No performance requirement found, please add a performance requirement to objective definition.")

        super().__init__(MissionObjective.DEFAULT, parameter, requirements, id)

    def __repr__(self) -> str:
        """String representation of the objective."""
        return f"DefaultMissionObjective(parameter={self.parameter}, n_reqs={len(self.requirements)}, id={self.id.split('-')[0]})"

    @classmethod
    def from_dict(cls, obj_dict: Dict[str, Union[str, float]]) -> 'DefaultMissionObjective':
        """Create a default mission objective from a dictionary."""

        # validate input dictionary
        assert 'objective_type' in obj_dict and obj_dict['objective_type'] == MissionObjective.DEFAULT, "Objective type must be 'default' for DefaultMissionObjective"
        assert 'parameter' in obj_dict, "Parameter must be specified in the dictionary"
        assert 'requirements' in obj_dict, "Requirements must be specified in the dictionary"

        # Convert requirements to MissionRequirement instances
        if all(isinstance(req, dict) for req in obj_dict['requirements']):
            requirements = [MissionRequirement.from_dict(req) for req in obj_dict['requirements']]
        elif all(isinstance(req, MissionRequirement) for req in obj_dict['requirements']):
            requirements = obj_dict['requirements']
        else:
            raise ValueError("Requirements must be a list of dictionaries or `MissionRequirement` instances")

        # Unpack other attributes
        parameter = obj_dict.get('parameter')
        id = obj_dict.get('id', None)

        # Return DefaultMissionObjective
        return cls(parameter, requirements, id)

class EventDrivenObjective(MissionObjective):
    def __init__(self, 
                 event_type: str,
                 parameter: str,
                 requirements: List[MissionRequirement], 
                #  synergistic_parameters: List[str] = [],
                 id : str = None
                 ):
        """ 
        ### Event Driven Objective
         
        Initialize an event-driven objective with a parameter and requirements.
        - :`event_type`: The type of geophysical event associated with the objective.
        - :`parameter`: The primary geophysical parameter to be measured (e.g., "Chl-A concentration").
        - :`requirements`: A list of `MeasurementRequirement` instances that define the requirements for the objective.
        # - :`synergistic_parameters`: A list of additional parameters that are synergistic with the main parameter.
        - :`id`: An optional ID for the objective. If None, a new UUID is generated.
        """
        # Validate inputs
        if not any(isinstance(req, SpatialCoverageRequirement) for req in requirements):
            raise ValueError("No spatial requirement found, please add a spatial coverage requirement to objective definition.")
        if not any(isinstance(req, PerformanceRequirement) for req in requirements):
            raise ValueError("No performance requirement found, please add a performance requirement to objective definition.")

        # Initialize the parent class
        super().__init__(MissionObjective.EVENT, parameter, requirements, id)
        
        # Validate inputs
        assert isinstance(event_type, str), "Event type must be a string"
        # assert isinstance(synergistic_parameters, list), "Synergistic parameters must be a list"
        # assert all(isinstance(param, str) for param in synergistic_parameters), "Synergistic parameters must be strings"
        # assert parameter not in synergistic_parameters, "Main parameter cannot be in list of synergistic parameters."
        
        # Set attributes
        self.event_type = event_type.lower() 
        # self.synergistic_parameters = [param.lower() for param in synergistic_parameters]
    
    def __repr__(self):
        return f"EventDrivenObjective(parameter={self.parameter}, event_type={self.event_type}, id={self.id.split('-')[0]})"

    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, float]]) -> 'EventDrivenObjective':
        """Create an event-driven objective from a dictionary."""
        assert 'objective_type' in d and d['objective_type'] == MissionObjective.EVENT, "Objective type must be 'event' for EventDrivenObjective"
        assert 'event_type' in d, "Event type must be specified in the dictionary"
        assert 'parameter' in d, "Parameter must be specified in the dictionary"
        assert 'requirements' in d, "Requirements must be specified in the dictionary"
        
        # Convert requirements to MissionRequirement instances
        if all(isinstance(req, dict) for req in d['requirements']):
            requirements = [MissionRequirement.from_dict(req) for req in d['requirements']]
        elif all(isinstance(req, MissionRequirement) for req in d['requirements']):
            requirements = d['requirements']
        else:
            raise ValueError("Requirements must be a list of dictionaries or `MissionRequirement` instances")
        
        return EventDrivenObjective(event_type=d['event_type'],
                                    parameter=d['parameter'],
                                    requirements=requirements,
                                    # synergistic_parameters=d.get('synergistic_parameters', []),
                                    id=d.get('id', None))

    @classmethod
    def from_default_objective(cls, 
                               event : GeophysicalEvent, 
                               default_objective: DefaultMissionObjective, 
                            #    synergistic_parameters : list = []
                            ) -> 'EventDrivenObjective':
        """Create an `EventDrivenObjective` from a default objective and an event."""

        # Validate Inputs
        assert isinstance(event, GeophysicalEvent), "Event must be an instance of GeophysicalEvent"
        assert isinstance(default_objective, DefaultMissionObjective), "Default objective must be an instance of DefaultMissionObjective"
        # assert isinstance(synergistic_parameters, list), "Synergistic parameters must be a list"

        # Return Event Objective
        return cls(event_type=event.event_type,
                   parameter=default_objective.parameter,
                   requirements=[req for req in default_objective],
                #    synergistic_parameters=synergistic_parameters
                   )