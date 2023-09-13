import math
import os
import re
from typing import Any, Callable
from nodes.planning.preplanners import AbstractPreplanner
from nodes.planning.replanners import AbstractReplanner
from nodes.orbitdata import OrbitData
from nodes.states import *
from nodes.science.reqs import *
from messages import *
from dmas.modules import *
import pandas as pd

class PlannerTypes(Enum):
    FIXED = 'FIXED'
    GREEDY = 'GREEDY'
    MACCBBA = 'MACCBBA'
    MCCBBA = 'MCCBBA'
    ACBBA = 'ACBBA'

class PlanningModule(InternalModule):
    def __init__(self, 
                results_path : str, 
                parent_name : str,
                parent_network_config: NetworkConfig, 
                utility_func : Callable[[], Any],
                preplanner : AbstractPreplanner = None,
                replanner : AbstractReplanner = None,
                initial_reqs : list = [],
                level: int = logging.INFO, 
                logger: logging.Logger = None
                ) -> None:
                       
        addresses = parent_network_config.get_internal_addresses()        
        sub_addesses = []
        sub_address : str = addresses.get(zmq.PUB)[0]
        sub_addesses.append( sub_address.replace('*', 'localhost') )

        if len(addresses.get(zmq.SUB)) > 1:
            sub_address : str = addresses.get(zmq.SUB)[1]
            sub_addesses.append( sub_address.replace('*', 'localhost') )

        pub_address : str = addresses.get(zmq.SUB)[0]
        pub_address = pub_address.replace('localhost', '*')

        addresses = parent_network_config.get_manager_addresses()
        push_address : str = addresses.get(zmq.PUSH)[0]

        planner_network_config =  NetworkConfig(parent_name,
                                        manager_address_map = {
                                        zmq.REQ: [],
                                        zmq.SUB: sub_addesses,
                                        zmq.PUB: [pub_address],
                                        zmq.PUSH: [push_address]})

        super().__init__(f'{parent_name}-PLANNING_MODULE', 
                        planner_network_config, 
                        parent_network_config, 
                        level, 
                        logger)
        

        self.results_path = results_path
        self.parent_name = parent_name
        self.utility_func = utility_func

        self.preplanner : AbstractPreplanner = preplanner
        self.replanner : AbstractReplanner = replanner
        
        self.plan_history = []
        self.plan = []
        self.stats = {
                        "planning" : [],
                        "preplanning" : [],
                        "replanning" : []
                    }

        self.agent_state : SimulationAgentState = None
        self.parent_agent_type = None
        self.orbitdata : OrbitData = None

        self.initial_reqs = []
        for req in initial_reqs:
            req : MeasurementRequest
            if req.t_start > 0:
                continue
            self.initial_reqs.append(req)

    async def sim_wait(self, delay: float) -> None:
        # does nothing
        return

    async def setup(self) -> None:
        # initialize internal messaging queues
        self.states_inbox = asyncio.Queue()
        self.action_status_inbox = asyncio.Queue()
        self.req_inbox = asyncio.Queue()

        # setup agent state locks
        self.agent_state_lock = asyncio.Lock()
        self.agent_state_updated = asyncio.Event()

    async def live(self) -> None:
        """
        Performs two concurrent tasks:
        - Listener: receives messages from the parent agent and checks results
        - Bundle-builder: plans and bids according to local information
        """
        try:
            listener_task = asyncio.create_task(self.listener(), name='listener()')
            bundle_builder_task = asyncio.create_task(self.planner(), name='planner()')
            
            tasks = [listener_task, bundle_builder_task]

            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        finally:
            for task in done:
                self.log(f'`{task.get_name()}` task finalized! Terminating all other tasks...')

            for task in pending:
                task : asyncio.Task
                if not task.done():
                    task.cancel()
                    await task

    async def listener(self) -> None:
        """
        Listens for any incoming messages, unpacks them and classifies them into 
        internal inboxes for future processing
        """
        try:
            # listen for broadcasts and place in the appropriate inboxes
            while True:
                self.log('listening to manager broadcast!')
                _, _, content = await self.listen_manager_broadcast()

                # if sim-end message, end agent `live()`
                if content['msg_type'] == ManagerMessageTypes.SIM_END.value:
                    self.log(f"received manager broadcast or type {content['msg_type']}! terminating `live()`...")
                    return

                elif content['msg_type'] == SimulationMessageTypes.SENSES.value:
                    self.log(f"received senses from parent agent!", level=logging.DEBUG)

                    # unpack message 
                    senses_msg : SensesMessage = SensesMessage(**content)

                    senses = []
                    senses.append(senses_msg.state)
                    senses.extend(senses_msg.senses)     

                    for sense in senses:
                        if sense['msg_type'] == SimulationMessageTypes.AGENT_ACTION.value:
                            # unpack message 
                            action_msg = AgentActionMessage(**sense)
                            self.log(f"received agent action of status {action_msg.status}!")
                            
                            # send to planner
                            await self.action_status_inbox.put(action_msg)

                        elif sense['msg_type'] == SimulationMessageTypes.AGENT_STATE.value:
                            # unpack message 
                            state_msg : AgentStateMessage = AgentStateMessage(**sense)
                            self.log(f"received agent state message!")
                                                        
                            # send to planner
                            await self.states_inbox.put(state_msg) 

                        elif sense['msg_type'] == SimulationMessageTypes.MEASUREMENT_REQ.value:
                            # request received directly from another agent
                            
                            # unpack message 
                            req_msg = MeasurementRequestMessage(**sense)
                            req : MeasurementRequest = MeasurementRequest.from_dict(req_msg.req)
                            self.log(f"received measurement request message!")
                            
                            # send to planner
                            await self.req_inbox.put(req)

                        # TODO support down-linked information processing

                elif content['msg_type'] == SimulationMessageTypes.MEASUREMENT_REQ.value:
                    # request received directly from another module

                    # unpack message 
                    req_msg = MeasurementRequestMessage(**content)
                    req : MeasurementRequest = MeasurementRequest.from_dict(req_msg.req)
                    self.log(f"received measurement request message from another module!")
                    
                    # send to planner
                    await self.req_inbox.put(req)

        except asyncio.CancelledError:
            return

    async def planner(self) -> None:
        """
        Generates a plan for the parent agent to perform
        """
        try:
            # initialize results vectors
            plan, known_reqs = [], []
            bundle, path, results = [], [], {}

            # level = logging.WARNING
            level = logging.DEBUG

            while True:
                # wait for agent to update state
                _ : AgentStateMessage = await self.states_inbox.get()

                # --- Check Action Completion ---
                
                while not self.action_status_inbox.empty():
                    action_msg : AgentActionMessage = await self.action_status_inbox.get()
                    action : AgentAction = action_from_dict(**action_msg.action)

                    if action_msg.status == AgentAction.PENDING:
                        # if action wasn't completed, try again
                        plan_ids = [action.id for action in self.plan]
                        action_dict : dict = action_msg.action
                        if action_dict['id'] in plan_ids:
                            self.log(f'action {action_dict} not completed yet! trying again...')
                            plan_out.append(action_dict)

                    else:
                        # if action was completed or aborted, remove from plan
                        if action_msg.status == AgentAction.COMPLETED:
                            self.log(f'action of type `{action.action_type}` completed!', level)

                        action_dict : dict = action_msg.action
                        completed_action = AgentAction(**action_dict)
                        removed = None
                        for action in plan:
                            action : AgentAction
                            if action.id == completed_action.id:
                                removed = action
                                break
                        
                        if removed is not None:
                            removed : AgentAction
                            plan : list
                            plan.remove(removed)
                
                # --- Look for Plan Updates ---

                plan_out = []
                
                if (len(self.initial_reqs) > 0                  # there are initial requests 
                    and self.get_current_time() <= 1e-3         # simulation just started
                    and self.preplanner is not None             # there is a preplanner assigned to this planner
                    ):
                    # Generate Initial Plan
                    initial_reqs = self.initial_reqs
                    self.initial_reqs = []

                    await self.agent_state_lock.acquire()
                    plan : list = self.preplanner.initialize_plan(  self.agent_state, 
                                                                    initial_reqs,
                                                                    self.orbitdata,
                                                                    level
                                                                    )
                    self.agent_state_lock.release()

                if (
                    len(self.req_inbox) > 0                     # there was a new request received
                    ):
                    # Replan
                    incoming_reqs = []
                    while not self.req_inbox.empty():
                        incoming_reqs.append(await self.req_inbox.get())

                    await self.agent_state_lock.acquire()
                    plan : list = self.replanner.revise_plan(   self.agent_state, 
                                                                incoming_reqs,
                                                                self.orbitdata,
                                                                level
                                                            )
                    self.agent_state_lock.release()

                # --- Execute plan ---

                # get next action to perform
                plan_out_ids = [action['id'] for action in plan_out]
                for action in plan:
                    action : AgentAction
                    if (action.t_start <= self.get_current_time()
                        and action.id not in plan_out_ids):
                        
                        plan_out.append(action.to_dict())

                        if len(plan_out_ids) > 0:
                            break

                if len(plan_out) == 0:
                    if len(plan) > 0:
                        # next action is yet to start, wait until then
                        next_action : AgentAction = plan[0]
                        t_idle = next_action.t_start if next_action.t_start > self.get_current_time() else self.get_current_time()
                    else:
                        # no more actions to perform
                        if isinstance(self.agent_state, SatelliteAgentState):
                            # wait until the next ground-point access
                            self.orbitdata : OrbitData
                            t_next = self.orbitdata.get_next_gs_access(self.get_current_time())
                        else:
                            # idle until the end of the simulation
                            t_next = np.Inf

                        # t_idle = self.t_plan + self.planning_horizon
                        t_idle = t_next

                    action = WaitForMessages(self.get_current_time(), t_idle)
                    plan_out.append(action.to_dict())

                # --- FOR DEBUGGING PURPOSES ONLY: ---
                out = f'\nPLAN\nid\taction type\tt_start\tt_end\n'
                for action in plan:
                    action : AgentAction
                    out += f"{action.id.split('-')[0]}, {action.action_type}, {action.t_start}, {action.t_end}\n"
                # self.log(out, level=logging.WARNING)

                out = f'\nPLAN OUT\nid\taction type\tt_start\tt_end\n'
                for action in plan_out:
                    action : dict
                    out += f"{action['id'].split('-')[0]}, {action['action_type']}, {action['t_start']}, {action['t_end']}\n"
                # self.log(out, level=logging.WARNING)
                # -------------------------------------

                self.log(f'sending {len(plan_out)} actions to agent...')
                plan_msg = PlanMessage(self.get_element_name(), self.get_network_name(), plan_out)
                await self._send_manager_msg(plan_msg, zmq.PUB)

                self.log(f'actions sent!')

        except asyncio.CancelledError:
            return

    def plan_from_path( self, 
                        state : SimulationAgentState, 
                        path : list
                    ) -> list:
        """
        Generates a list of AgentActions from the current path.

        Agents look to move to their designated measurement target and perform the measurement.

        ## Arguments:
            - state (:obj:`SimulationAgentState`): state of the agent at the start of the path
            - path (`list`): list of tuples indicating the sequence of observations to be performed and time of observation
        """
        plan = []

        # if no requests left in the path, wait for the next planning horizon
        if len(path) == 0:
            t_0 = self.get_current_time()
            t_f = t_0 + self.planning_horizon
            return [WaitForMessages(t_0, t_f)]

        # TODO add wait timer if needed for plan convergende

        # add actions per measurement
        for i in range(len(path)):
            plan_i = []

            measurement_req, subtask_index, t_img, u_exp = path[i]
            measurement_req : MeasurementRequest; subtask_index : int; t_img : float; u_exp : float

            if not isinstance(measurement_req, GroundPointMeasurementRequest):
                raise NotImplementedError(f"Cannot create plan for requests of type {type(measurement_req)}")
            
            # Estimate previous state
            if i == 0:
                if isinstance(state, SatelliteAgentState):
                    t_prev = state.t
                    prev_state : SatelliteAgentState = state.copy()

                elif isinstance(state, UAVAgentState):
                    t_prev = state.t #TODO consider wait time for convergence
                    prev_state : UAVAgentState = state.copy()

                else:
                    raise NotImplementedError(f"cannot calculate travel time start for agent states of type {type(state)}")
            else:
                prev_req = None
                for action in reversed(plan):
                    action : AgentAction
                    if isinstance(action, MeasurementAction):
                        prev_req = MeasurementRequest.from_dict(action.measurement_req)
                        break

                action_prev : AgentAction = plan[-1]
                t_prev = action_prev.t_end

                if isinstance(state, SatelliteAgentState):
                    prev_state : SatelliteAgentState = state.propagate(t_prev)
                    
                    if prev_req is not None:
                        prev_state.attitude = [
                                            prev_state.calc_off_nadir_agle(prev_req),
                                            0.0,
                                            0.0
                                        ]

                elif isinstance(state, UAVAgentState):
                    prev_state : UAVAgentState = state.copy()
                    prev_state.t = t_prev

                    if isinstance(prev_req, GroundPointMeasurementRequest):
                        prev_state.pos = prev_req.pos
                    else:
                        raise NotImplementedError(f"cannot calculate travel time start for requests of type {type(prev_req)} for uav agents")

                else:
                    raise NotImplementedError(f"cannot calculate travel time start for agent states of type {type(state)}")

            # maneuver to point to target
            t_maneuver_end = None
            if isinstance(state, SatelliteAgentState):
                prev_state : SatelliteAgentState

                t_maneuver_start = prev_state.t
                th_f = prev_state.calc_off_nadir_agle(measurement_req)
                t_maneuver_end = t_maneuver_start + abs(th_f - prev_state.attitude[0]) / prev_state.max_slew_rate

                if abs(t_maneuver_start - t_maneuver_end) > 0.0:
                    maneuver_action = ManeuverAction([th_f, 0, 0], t_maneuver_start, t_maneuver_end)
                    plan_i.append(maneuver_action)   
                else:
                    t_maneuver_end = None

            # move to target
            t_move_start = t_prev if t_maneuver_end is None else t_maneuver_end
            if isinstance(state, SatelliteAgentState):
                lat, lon, _ = measurement_req.lat_lon_pos
                df : pd.DataFrame = self.orbitdata.get_ground_point_accesses_future(lat, lon, t_move_start)
                
                t_move_end = None
                for _, row in df.iterrows():
                    if row['time index'] * self.orbitdata.time_step >= t_img:
                        t_move_end = row['time index'] * self.orbitdata.time_step
                        break

                if t_move_end is None:
                    # unpheasible path
                    self.log(f'Unheasible element in path. Cannot perform observation.', level=logging.DEBUG)
                    continue

                future_state : SatelliteAgentState = state.propagate(t_move_end)
                final_pos = future_state.pos

            elif isinstance(state, UAVAgentState):
                final_pos = measurement_req.pos
                dr = np.array(final_pos) - np.array(prev_state.pos)
                norm = np.sqrt( dr.dot(dr) )
                
                t_move_end = t_move_start + norm / state.max_speed

            else:
                raise NotImplementedError(f"cannot calculate travel time end for agent states of type {type(state)}")
            
            if t_move_end < t_img:
                plan_i.append( WaitForMessages(t_move_end, t_img) )
                
            t_img_start = t_img
            t_img_end = t_img_start + measurement_req.duration

            if isinstance(self._clock_config, FixedTimesStepClockConfig):
                dt = self._clock_config.dt
                if t_move_start < np.Inf:
                    t_move_start = dt * math.floor(t_move_start/dt)
                if t_move_end < np.Inf:
                    t_move_end = dt * math.ceil(t_move_end/dt)

                if t_img_start < np.Inf:
                    t_img_start = dt * math.floor(t_img_start/dt)
                if t_img_end < np.Inf:
                    t_img_end = dt * math.ceil((t_img_start + measurement_req.duration)/dt)
            
            if abs(t_move_start - t_move_end) >= 1e-3:
                move_action = TravelAction(final_pos, t_move_start, t_move_end)
                plan_i.append(move_action)
            
            # perform measurement
            main_measurement, _ = measurement_req.measurement_groups[subtask_index]
            measurement_action = MeasurementAction( 
                                                    measurement_req.to_dict(),
                                                    subtask_index, 
                                                    main_measurement,
                                                    u_exp,
                                                    t_img_start, 
                                                    t_img_end
                                                    )
            plan_i.append(measurement_action)  

            # TODO inform others of request completion

            plan.extend(plan_i)
        
        return plan