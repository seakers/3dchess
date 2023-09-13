from abc import ABC, abstractmethod

from nodes.orbitdata import OrbitData
from nodes.states import AbstractAgentState

class AbstractReplanner(ABC):
    """
    # Replanner
    
    
    """
    @abstractmethod 
    def needs_replanning(   self, 
                            state : AbstractAgentState
                        ) -> bool:
        checks 