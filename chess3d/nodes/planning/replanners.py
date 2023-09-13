from abc import ABC, abstractmethod
import logging

from nodes.states import SimulationAgentState
from nodes.orbitdata import OrbitData
from nodes.states import AbstractAgentState

class AbstractReplanner(ABC):
    """
    # Replanner
    
    
    """
    @abstractmethod 
    def needs_replanning(   self, 
                            state : AbstractAgentState,
                            curent_plan : list,
                            incoming_reqs : list
                        ) -> bool:
        """
        Returns `True` if the current plan needs replanning
        """

    @abstractmethod
    def revise_plan(    self, 
                        state : AbstractAgentState, 
                        current_plan : list,
                        incoming_reqs : list, 
                        orbitdata : OrbitData,
                        level : int = logging.DEBUG
                    ) -> list:
        """
        Revises the current plan 
        """
        pass

    @abstractmethod
    def plan_from_path( self, 
                        state : SimulationAgentState, 
                        path : list,
                        orbitdata : OrbitData
                    ) -> list:
        """
        Generates a list of AgentActions from the current path.

        Agents look to move to their designated measurement target and perform the measurement.

        ## Arguments:
            - state (:obj:`SimulationAgentState`): state of the agent at the start of the path
            - path (`list`): list of tuples indicating the sequence of observations to be performed and time of observation
        """