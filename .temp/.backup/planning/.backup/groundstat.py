import asyncio
import logging
from typing import Any, Callable
from nodes.planning.backup.fixed import FixedPlanner
from nodes.states import GroundStationAgentState, SatelliteAgentState, SimulationAgentTypes, UAVAgentState
from nodes.agent import *
from messages import *
from dmas.network import NetworkConfig


class GroundStationPlanner(FixedPlanner):
    def __init__(self, 
                results_path: str, 
                parent_name: str, 
                measurement_reqs : list,
                parent_network_config: NetworkConfig, 
                utility_func: Callable[[], Any], 
                level: int = logging.INFO, 
                logger: logging.Logger = None
                ) -> None:
        
        # create an initial plan
        self.measurement_reqs = measurement_reqs
        plan = []
        for measurement_req in measurement_reqs:
            # broadcast every initialy known measurement requests
            measurement_req : MeasurementRequest
            msg = MeasurementRequestMessage(parent_name, parent_name, measurement_req.to_dict())
            
            # TODO schedule broadcasts depending on agent access
            if measurement_req.t_start == 0.0:
                continue
            action = BroadcastMessageAction(msg.to_dict(), measurement_req.t_start)

            if len(plan) == 0:
                plan.append(action)
            else:
                i_insert = -1
                for i in range(len(plan)):
                    action_i : AgentAction = plan[i]
                    if action.t_start <= action_i.t_start:
                        i_insert = i
                        break
                if i_insert < 0:
                    plan.append(action)
                else:
                    plan.insert(i_insert, action)
        
        super().__init__(   results_path, 
                            parent_name, 
                            plan, 
                            parent_network_config, 
                            utility_func, 
                            level, 
                            logger
                        )
        
    async def planner(self) -> None:
        try:
            t_curr = 0
            while True:
                plan_out = []
                msg : AgentStateMessage = await self.states_inbox.get()

                # update current time:
                if msg.state['state_type'] == SimulationAgentTypes.SATELLITE.value:
                    state = SatelliteAgentState(**msg.state)
                elif msg.state['state_type'] == SimulationAgentTypes.UAV.value:
                    state = UAVAgentState(**msg.state)
                elif msg.state['state_type'] == SimulationAgentTypes.GROUND_STATION.value:
                    state = GroundStationAgentState(**msg.state)
                else:
                    raise NotImplementedError(f"`state_type` {msg.state['state_type']} not supported.")

                if t_curr < state.t:
                    t_curr = state.t

                while not self.action_status_inbox.empty():
                    msg : AgentActionMessage = await self.action_status_inbox.get()

                    if msg.status != AgentAction.COMPLETED and msg.status != AgentAction.ABORTED:
                        # if action wasn't completed, re-try
                        action_dict : dict = msg.action
                        self.log(f'action {action_dict} not completed yet! trying again...')
                        plan_out.append(action_dict)

                    else:
                        # if action was completed or aborted, remove from plan
                        action_dict : dict = msg.action
                        completed_action = AgentAction(**action_dict)
                        removed = None
                        for action in self.plan:
                            action : AgentAction
                            if action.id == completed_action.id:
                                removed = action
                                break

                        if removed is not None:
                            self.plan.remove(removed)

                while not self.req_inbox.empty():
                    # TODO: rebroadcast measurement requests that were not known to this GS
                    msg : MeasurementRequestMessage = await self.req_inbox.get()

                plan_out_id = [action['id'] for action in plan_out]
                for action in self.plan:
                    action : AgentAction
                    if (action.t_start <= t_curr <= action.t_end
                        and action.id not in plan_out_id):
                        plan_out.append(action.to_dict())

                if len(plan_out) == 0:
                    # if no plan left, just idle for a time-step
                    self.log('no more actions to perform. instruct agent to idle for the remainder of the simulation.')
                    if len(self.plan) > 0:
                        next_action : AgentAction = self.plan[0]
                        t_idle = next_action.t_start
                    else:
                        t_idle = t_curr + 1e8 # TODO find end of simulation time        
                    action = WaitForMessages(t_curr, t_idle)
                    plan_out.append(action.to_dict())
                    
                self.log(f'sending {len(plan_out)} actions to agent...')
                plan_msg = PlanMessage(self.get_element_name(), self.get_network_name(), plan_out)
                await self._send_manager_msg(plan_msg, zmq.PUB)

                self.log(f'actions sent!')

        except asyncio.CancelledError:
            return

        except Exception as e:
            self.log(f'routine failed. {e}')
            raise e