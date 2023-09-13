from abc import ABC, abstractmethod
import logging
from nodes.orbitdata import OrbitData
from nodes.states import AbstractAgentState

class AbstractPreplanner(ABC):
    """
    # Preplanner
    
    
    """
    
    @abstractmethod
    def initialize_plan(    self, 
                            state : AbstractAgentState, 
                            initial_reqs : list, 
                            orbitdata : OrbitData,
                            level : int = logging.DEBUG
                        ) -> list:
        """
        Creates an initial 
        """
        pass