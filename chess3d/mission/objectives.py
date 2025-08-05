
from typing import Dict, Union
import uuid
import numpy as np

from chess3d.mission.requirements import *


class MissionObjective:
    def __init__(self, 
                 parameter: str, 
                 priority: float, 
                 requirements: list, 
                 valid_instruments : list, 
                 id : str = None):
        """ 
        ### Objective
         
        Initialize an objective with a priority, parameter, and requirements.
        - :`parameter`: The primary geophysical parameter to be measured (e.g., "Chl-A concentration").
        - :`priority`: The priority of the objective.
        - :`requirements`: A list of `MeasurementRequirement` instances that define the requirements for the objective.
        - :`valid_instruments`: A list of valid instruments that can be used to measure the parameter.
        - :`id`: An optional ID for the objective. If None, a new UUID is generated.
        """
        if all([isinstance(req, dict) for req in requirements]):
            requirements = [MissionRequirement(**req) for req in requirements]

        # Validate inputs
        assert isinstance(priority, (int, float)), "Priority must be a number"
        assert isinstance(parameter, str), "Parameter must be a string"
        assert len(requirements) > 0, "At least one requirement is needed"
        assert all(isinstance(req, MissionRequirement) for req in requirements), "All requirements must be instances of `MeasurementRequirement`"
        assert any(isinstance(req, TemporalRequirement) for req in requirements), "At least one requirement must be a `TemporalRequirement`"
        assert any(isinstance(req, SpatialRequirement) for req in requirements), "At least one requirement must be a `SpatialRequirement`"
        assert isinstance(id, str) or id is None, f"ID must be a string or None. is of type {type(id)}"

        # Set attributes
        self.priority : float = priority
        self.parameter : str = parameter
        self.requirements : Dict[str, MissionRequirement] = {requirement.attribute : requirement 
                                                             for requirement in requirements 
                                                             if isinstance(requirement, MissionRequirement)}
        self.valid_instruments = [instrument.lower() 
                                  for instrument in valid_instruments
                                  if isinstance(instrument, str)] # TODO remove this and implement knoledge graph in agent
        self.id = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())

    def eval_measurement_performance(self, measurement: dict) -> float:
        """Evaluate the product of satisfaction scores given a measurement dict {attr: value}"""
        try:
            # Check if the measurement contains the required parameter
            # TODO replace with knowledge graph
            if measurement['instrument'].lower() not in self.valid_instruments:
                # measurement does not meet requirement
                return 0.0

            # Calculate the performance score for each requirement
            # If the attribute is not in the measurement, return 0
            # Otherwise, calculate the preference value using the requirement's preference function
            scores = [
                0 if attribute not in measurement
                else req.calc_preference_value(measurement[attribute])
                for attribute, req in self.requirements.items()
            ]

            # return the product of all scores
            return np.prod(scores)
        
        except Exception as e:
            raise(e)
        
    def calc_reward(self, measurement: dict) -> float:
        """Calculate the reward for the objective based on the measurement."""
        return RO[self.reobservation_strategy](measurement)
    
    def __repr__(self):
        """String representation of the objective."""
        return f"MissionObjective({self.parameter}, priority={self.priority}, requirements={self.requirements})"
    
    def __str__(self):
        """String representation of the objective."""
        return f"Objective: {self.parameter}, Priority: {self.priority}, Requirements: {self.requirements}"

    def copy(self) -> 'MissionObjective':
        """Create a copy of the objective."""
        return MissionObjective(self.parameter, self.priority, [req.copy() for req in self.requirements.values()], self.valid_instruments, self.reobservation_strategy, self.id)

    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert the objective to a dictionary."""
        out : dict = dict(self.__dict__)
        out['requirements'] = {key: req.to_dict() 
                            for key,req in self.requirements.items()}
        return out
    
    def from_dict(obj_dict: Dict[str, Union[str, float]]) -> 'MissionObjective':
        """Create an objective from a dictionary."""
        if 'event_type' in obj_dict:
            # EventDrivenObjective
            requirements = [MissionRequirement(**req) for _,req in obj_dict['requirements'].items()]
            return EventDrivenObjective(obj_dict['parameter'], obj_dict['priority'], requirements, obj_dict['event_type'], obj_dict['valid_instruments'], obj_dict['reobservation_strategy'], obj_dict['id'])
        else:
            requirements = [MissionRequirement(**req) for _,req in obj_dict['requirements'].items()]
            return MissionObjective(obj_dict['parameter'], obj_dict['priority'], requirements, obj_dict['valid_instruments'], obj_dict['reobservation_strategy'], obj_dict['id'])

    def can_perform(self, instrument: str) -> bool:
        """Check if the objective can be performed by the given instrument."""
        return instrument.lower() in self.valid_instruments

class EventDrivenObjective(MissionObjective):
    def __init__(self, 
                 parameter: str, 
                 priority: float, 
                 requirements: list, 
                 event_type: str,
                 valid_instruments : list,
                 reobservation_strategy: str,
                 synergistic_parameters: list,
                 coobservation_strategy: str,
                 t_corr : float = None,
                 id : str = None
                 ):
        """ 
        ### Event Driven Objective
         
        Initialize an event-driven objective with a priority, parameter, and requirements.
        - :`priority`: The priority of the objective.
        - :`parameter`: The primary geophysical parameter to be measured (e.g., "Chl-A concentration").
        - :`requirements`: A list of `MeasurementRequirement` instances that define the requirements for the objective.
        - :`event_type`: The type of geophysical event associated with the objective.
        - :`valid_instruments`: A list of valid instruments that can be used to measure the parameter.
        - :`reobservation_strategy`: The strategy for reobserving the event. Can be one of the following:
            - "linar_increase"
            - "linar_decrease"
            - "decaying_increase"
            - "decaying_decrease"
            - "immediate_decrease"
            - "no_change"
            - "monitoring"
        - :`id`: An optional ID for the objective. If None, a new UUID is generated.
        """
        super().__init__(parameter, priority, requirements, valid_instruments, reobservation_strategy, id)
        
        # Validate inputs
        assert isinstance(event_type, str), "Event type must be a string"
        assert isinstance(synergistic_parameters, list), "Synergistic parameters must be a list"
        assert all(isinstance(param, str) for param in synergistic_parameters), "Synergistic parameters must be strings"
        assert parameter not in synergistic_parameters, "Main parameter cannot be in list of synergistic parameters."
        assert isinstance(coobservation_strategy, str), "Coobservation strategy must be a string"
        assert coobservation_strategy in CO, f"Invalid coobservation strategy: {coobservation_strategy}. Available strategies: {list(CO.keys())}"
        
        # Set attributes
        self.event_type = event_type.lower() 
        self.synergistic_parameters = [param.lower() for param in synergistic_parameters]
        self.coobservation_strategy = coobservation_strategy.lower()
        self.t_corr = t_corr if t_corr is not None else np.Inf

    def __repr__(self):
        """String representation of the objective."""
        return f"EventDrivenObjective({self.parameter}, priority={self.priority}, requirements={self.requirements})"
    
    def __str__(self):
        """String representation of the objective."""
        return f"Event-driven Objective: {self.parameter}, Priority: {self.priority}, Requirements: {self.requirements}"

    def copy(self):
        return EventDrivenObjective(self.parameter, 
                                    self.priority, 
                                    [req.copy() for req in self.requirements.values()], 
                                    self.event_type, 
                                    self.valid_instruments, 
                                    self.synergistic_parameters,
                                    self.coobservation_strategy,
                                    self.t_corr,
                                    self.id)

    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert the objective to a dictionary."""
        out : dict = dict(self.__dict__)
        out['requirements'] = {key: req.to_dict() 
                            for key,req in self.requirements.items()}
        return out
