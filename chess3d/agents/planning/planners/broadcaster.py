import logging

from dmas.clocks import ClockConfig
from dmas.utils import runtime_tracker

from chess3d.messages import MeasurementRequestMessage
from chess3d.agents.actions import BroadcastMessageAction
from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.planning.plan import Plan, Replan
from chess3d.agents.planning.planner import AbstractReplanner
from chess3d.agents.science.requests import MeasurementRequest
from chess3d.agents.states import SimulationAgentState

class Broadcaster(AbstractReplanner):
       
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
        return super().update_percepts(state, current_plan, incoming_reqs, relay_messages, misc_messages, completed_actions, aborted_actions, pending_actions)
    
    @runtime_tracker
    def needs_planning( self, 
                        state : SimulationAgentState,
                        _ : object,
                        current_plan : Plan
                        ) -> bool:
        """ only replans whenever there are any pending relays or requests to broadcasts to perform """

        return (len(self.pending_relays) > 0
                or len(self.pending_reqs_to_broadcast)) and abs(state.t - current_plan.t) > 1e-3
    
    def generate_plan(self, 
                      state : SimulationAgentState,
                      specs : object,
                      current_plan : Plan,
                      _ : ClockConfig,
                      orbitdata : OrbitData
                      ) -> Plan:
        
        # initialize list of broadcasts to be done
        broadcasts = self._schedule_broadcasts(state, orbitdata)
                                        
        # update and return plan with new broadcasts
        return Replan.from_preplan(current_plan, broadcasts, t=state.t)
