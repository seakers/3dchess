    
from abc import abstractmethod
from logging import Logger
from typing import Any, Callable
import numpy as np

from dmas.clocks import ClockConfig

from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.planning.plan import Plan, Replan
from chess3d.agents.planning.planner import AbstractPreplanner, AbstractReplanner
from chess3d.agents.states import SimulationAgentState


class ReactivePlanner(AbstractPreplanner):    
    def __init__(self, 
                 utility_func: Callable[[], Any], 
                 horizon: float = 3600, 
                 period: float = 3600, 
                 logger: Logger = None
                 ) -> None:
        super().__init__(horizon, period, logger)

        self.utility_func = utility_func
        self.reward_grid = {}

    def update_percepts(self, 
                        state: SimulationAgentState, 
                        current_plan: Plan, 
                        incoming_reqs: list, 
                        relay_messages: list, 
                        misc_messages: list, 
                        completed_actions: list, 
                        aborted_actions: list, 
                        pending_actions: list
                        ) -> None:
        super().update_percepts(state, current_plan, incoming_reqs, relay_messages, misc_messages, completed_actions, aborted_actions, pending_actions)

        self.update_reward_grid(completed_actions, misc_messages)

    def update_reward_grid(self, completed_actions : list, misc_messages : list) -> None:
        """updates internal knowledge of the reward grid for planning purposes """
        raise NotImplementedError('TODO: Implement reward grid')
    
    @abstractmethod
    def _schedule_measurements(self, state : SimulationAgentState, current_plan : list, clock_config : ClockConfig, orbitdata : dict) -> list:
        pass

    def _schedule_broadcasts(self, 
                             state: SimulationAgentState, 
                             observations: list, 
                             orbitdata: OrbitData) -> list:
        broadcasts = super()._schedule_broadcasts(state, observations, orbitdata)
