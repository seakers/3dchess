import logging

from dmas.clocks import ClockConfig
from dmas.utils import runtime_tracker

from chess3d.messages import MeasurementRequestMessage, message_from_dict
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
        super().update_percepts(state, current_plan, incoming_reqs, relay_messages, misc_messages, completed_actions, aborted_actions, pending_actions)

        # check if broadcasts of new requests have already been scheduled       
        planned_broadcasts = {message_from_dict(**action.msg) 
                              for action in current_plan
                              if isinstance(action, BroadcastMessageAction)}
        planned_reqs = {MeasurementRequest.from_dict(msg.req) 
                        for msg in planned_broadcasts 
                        if isinstance(msg, MeasurementRequestMessage)}

        for req in planned_reqs: 
            if req in self.pending_reqs_to_broadcast:
                self.pending_reqs_to_broadcast.remove(req)
    
    @runtime_tracker
    def needs_planning( self, 
                        state : SimulationAgentState,
                        _ : object,
                        current_plan : Plan,
                        __ : OrbitData
                        ) -> bool:
        """ only replans whenever there are any pending relays or requests to broadcasts to perform """

        return len(self.pending_relays) > 0 or len(self.pending_reqs_to_broadcast)
    
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
