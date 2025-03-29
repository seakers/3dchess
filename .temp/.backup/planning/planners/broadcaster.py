# import logging

# from dmas.clocks import ClockConfig
# from dmas.utils import runtime_tracker
# from tqdm import tqdm

# from chess3d.agents.planning.planners.rewards import RewardGrid
# from chess3d.messages import MeasurementRequestMessage, message_from_dict
# from chess3d.agents.actions import BroadcastMessageAction
# from chess3d.agents.orbitdata import OrbitData
# from chess3d.agents.planning.plan import Plan, Preplan, Replan
# from chess3d.agents.planning.planner import AbstractReplanner
# from chess3d.agents.science.requests import MeasurementRequest
# from chess3d.agents.states import SatelliteAgentState, SimulationAgentState

# class Broadcaster(AbstractReplanner):
#     def __init__(self, debug: bool = False, logger: logging.Logger = None) -> None:
#         super().__init__(debug, logger)

#         # initialize set of preplanned broadcasts
#         self.planned_reqs_to_broadcasts : set = set()

#         self.latest_plan : Plan = None
#         self.n_prev_reqs = 0

#     # @runtime_tracker
#     # def update_percepts(self, 
#     #                     state: SimulationAgentState, 
#     #                     current_plan: Plan, 
#     #                     incoming_reqs: list, 
#     #                     relay_messages: list, 
#     #                     misc_messages: list, 
#     #                     completed_actions: list, 
#     #                     aborted_actions: list, 
#     #                     pending_actions: list
#     #                     ) -> None:
    
#     #     super().update_percepts(state, current_plan, incoming_reqs, relay_messages, misc_messages, completed_actions, aborted_actions, pending_actions)
        
#     #     # check if latest known plan has been updated
#     #     if (self.latest_plan is None 
#     #         or self.latest_plan.t < current_plan.t
#     #         ):

#     #         # plan has been updated; update latest known plan
#     #         self.latest_plan = current_plan.copy()

#     #         # get list of broadcasts to be performed
#     #         planned_broadcasts = {message_from_dict(**action.msg) 
#     #                             for action in current_plan
#     #                             if isinstance(action, BroadcastMessageAction)}

#     #         # get list of measurement requests scheduled to be broadcasted
#     #         planned_reqs = {MeasurementRequest.from_dict(msg.req) 
#     #                         for msg in planned_broadcasts 
#     #                         if isinstance(msg, MeasurementRequestMessage)}

#     #         # update list of planned request broadcasts
#     #         self.planned_reqs_to_broadcasts = planned_reqs

#     #     # remove from list of planned requests to breadcasts if they've been broadcasted already
#     #     self.planned_reqs_to_broadcasts.difference_update(self.broadcasted_reqs)
    
#     # @runtime_tracker
#     # def needs_planning( self, 
#     #                     *_
#     #                     ) -> bool:
#     #     """ only replans whenever there are any pending relays or requests to broadcasts to perform """

#     #     # unscheduled_reqs_to_broadcasts : set[MeasurementRequest] = self.pending_reqs_to_broadcast.difference(self.planned_reqs_to_broadcasts)

#     #     return len(self.pending_relays) > 0 \
#     #             or (len(self.pending_reqs_to_broadcast) != len(self.planned_reqs_to_broadcasts) and len(self.pending_reqs_to_broadcast) > 0)
    
#     # @runtime_tracker
#     # def generate_plan(self, 
#     #                   state : SimulationAgentState,
#     #                   _ : object,
#     #                   __ : RewardGrid,
#     #                   current_plan : Plan,
#     #                   ____ : ClockConfig,
#     #                   orbitdata : OrbitData
#     #                   ) -> Plan:
        
#     #     # initialize list of broadcasts to be done
#     #     broadcasts = self._schedule_broadcasts(state, orbitdata)

#     #     # update and return plan with new broadcasts
#     #     self.latest_plan =  Replan.from_preplan(current_plan, broadcasts, t=state.t)
#     #     return self.latest_plan.copy()

#     # @runtime_tracker
#     # def _schedule_broadcasts(self, state: SimulationAgentState, orbitdata: OrbitData) -> list:
#     #     if not isinstance(state, SatelliteAgentState):
#     #         raise NotImplementedError(f'Broadcast scheduling for agents of type `{type(state)}` not yet implemented.')
#     #     elif orbitdata is None:
#     #         raise ValueError(f'`orbitdata` required for agents of type `{type(state)}`.')

#     #     # initialize list of broadcasts to be done
#     #     broadcasts = []       

#     #     # get set of requests that have not been broadcasted yet 
#     #     unscheduled_reqs_to_broadcasts = self.pending_reqs_to_broadcast.difference(self.planned_reqs_to_broadcasts)

#     #     # sort requests based on their start time
#     #     pending_reqs_to_broadcast = list(unscheduled_reqs_to_broadcasts) 
#     #     pending_reqs_to_broadcast.sort(key=lambda a : a.t_start)

#     #     # schedule generated measurement request broadcasts
#     #     t_start = None
#     #     for req in tqdm(pending_reqs_to_broadcast,
#     #                     desc=f'{state.agent_name}-PLANNER: Pre-Scheduling Broadcasts', 
#     #                     leave=False):
#     #         req : MeasurementRequest
            
#     #         # calculate broadcast start time
#     #         if t_start is None or req.t_start > t_start:
#     #             path, t_start = self._create_broadcast_path(state, orbitdata, req.t_start)
                
#     #         # check broadcast feasibility
#     #         if t_start < 0:
#     #             break

#     #         # create broadcast action
#     #         msg = MeasurementRequestMessage(state.agent_name, state.agent_name, req.to_dict(), path=path)
#     #         broadcast_action = BroadcastMessageAction(msg.to_dict(), t_start)
            
#     #         # add to list of broadcasts
#     #         broadcasts.append(broadcast_action)

#     #         # update list of planned request broadcasts
#     #         self.planned_reqs_to_broadcasts.add(req)

#     #     # schedule message relay
#     #     relay_broadcasts = [self._schedule_relay(relay) for relay in self.pending_relays]
#     #     broadcasts.extend(relay_broadcasts)    
                        
#     #     # return scheduled broadcasts
#     #     return broadcasts 