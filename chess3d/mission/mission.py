
from typing import  Dict, Union

from chess3d.mission.objectives import *


CO = {
    # Coobservation Strategies
    "no_change" : lambda _ : 1.0
}

class Mission:
    def __init__(self, name : str, objectives: list):
        # Validate inputs
        assert isinstance(name, str), "Mission name must be a string"
        assert len(objectives) > 0, "At least one objective is needed"
        assert all(isinstance(obj, MissionObjective) for obj in objectives), "All objectives must be instances of `Objective`"

        # Set attributes
        self.name : str = name.lower()
        self.objectives : list[MissionObjective] = objectives

    def evaluate_measurement(self, measurements: dict) -> float:
        """Sum weighted objective scores across all objectives"""
        return sum([obj.priority * obj.eval_measurement_performance(measurements) 
                    for obj in self.objectives
                    if not isinstance(obj, EventDrivenObjective)
                    ])
    
    def evaluate_event_measurement(self, measurements: dict) -> float:
        """Sum weighted objective scores across all objectives"""
        return sum([obj.priority * obj.eval_measurement_performance(measurements) 
                    for obj in self.objectives
                    if isinstance(obj, EventDrivenObjective)
                    ])

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

    # TODO match geophysical event to objective

    # TODO event task from objective and events