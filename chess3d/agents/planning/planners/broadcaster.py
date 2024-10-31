import logging

from dmas.clocks import ClockConfig
from dmas.utils import runtime_tracker

from chess3d.agents.planning.planners.rewards import RewardGrid
from chess3d.messages import MeasurementRequestMessage, message_from_dict
from chess3d.agents.actions import BroadcastMessageAction
from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.planning.plan import Plan, Preplan, Replan
from chess3d.agents.planning.planner import AbstractReplanner
from chess3d.agents.science.requests import MeasurementRequest
from chess3d.agents.states import SimulationAgentState

class Broadcaster(AbstractReplanner):
    def __init__(self, debug: bool = False, logger: logging.Logger = None) -> None:
        super().__init__(debug, logger)

        # initialize set of preplanned broadcasts
        self.planned_req_broadcasts : set = set()

        self.latest_plan : Plan = None

    @runtime_tracker
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
        
        # check if latest known plan has been updated
        if (self.latest_plan is None 
            or self.latest_plan.t < current_plan.t
            ):

            # plan has been updated; update latest known plan
            self.latest_plan = current_plan.copy()

            # get list of broadcasts to be performed
            planned_broadcasts = {message_from_dict(**action.msg) 
                                for action in current_plan
                                if isinstance(action, BroadcastMessageAction)}

            # get list of measurement requests scheduled to be broadcasted
            planned_reqs = {MeasurementRequest.from_dict(msg.req) 
                            for msg in planned_broadcasts 
                            if isinstance(msg, MeasurementRequestMessage)}

            # update list of planned request broadcasts
            self.planned_req_broadcasts = planned_reqs
    
    @runtime_tracker
    def needs_planning( self, 
                        state : SimulationAgentState,
                        _ : object,
                        current_plan : Plan,
                        __ : OrbitData
                        ) -> bool:
        """ only replans whenever there are any pending relays or requests to broadcasts to perform """

        pending_request_broadcasts : set[MeasurementRequest] = self.pending_reqs_to_broadcast.difference(self.planned_req_broadcasts)

        return len(self.pending_relays) > 0 or len(pending_request_broadcasts)
    
    @runtime_tracker
    def generate_plan(self, 
                      state : SimulationAgentState,
                      _ : object,
                      __ : RewardGrid,
                      current_plan : Plan,
                      ____ : ClockConfig,
                      orbitdata : OrbitData
                      ) -> Plan:
        
        # initialize list of broadcasts to be done
        broadcasts = self._schedule_broadcasts(state, orbitdata)

        # update and return plan with new broadcasts
        self.latest_plan =  Replan.from_preplan(current_plan, broadcasts, t=state.t)
        return self.latest_plan.copy()
