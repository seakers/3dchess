from typing import Any, Callable
import pandas as pd

from dmas.modules import *
from dmas.utils import runtime_tracker

from nodes.planning.plan import Plan, Preplan
from nodes.planning.preplanners import AbstractPreplanner
from nodes.planning.replanners import AbstractReplanner
from nodes.orbitdata import OrbitData
from nodes.states import *
from nodes.science.reqs import *
from messages import *

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
        """ Internal Agent Module in charge of planning an scheduling. """
                       
        # setup network settings
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

        # intialize module
        super().__init__(f'{parent_name}-PLANNING_MODULE', 
                        planner_network_config, 
                        parent_network_config, 
                        level, 
                        logger)
        
        # initialize default attributes
        self.results_path = results_path
        self.parent_name = parent_name
        self.utility_func = utility_func

        self.preplanner : AbstractPreplanner = preplanner
        self.replanner : AbstractReplanner = replanner
        
        self.plan_history = []
        self.stats = {
                    }
        self.agent_state : SimulationAgentState = None
        self.parent_agent_type = None
        self.orbitdata : dict = None
        self.other_modules_exist : bool = False

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
        self.measurement_inbox = asyncio.Queue()
        self.relay_inbox = asyncio.Queue()
        self.misc_inbox = asyncio.Queue()

        # setup agent state locks
        self.agent_state_lock = asyncio.Lock()
        self.agent_state_updated = asyncio.Event()

        # place all initial requests into requests inbox
        for req in self.initial_reqs:
            await self.req_inbox.put(req)

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

    """
    -----------------
        LISTENER
    -----------------
    """
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
                    # received agent sense message from parent
                    self.log(f"received senses from parent agent!", level=logging.DEBUG)

                    # unpack message 
                    senses_msg : SensesMessage = SensesMessage(**content)

                    # sort agent senses to be processed
                    senses = [senses_msg.state]
                    senses.extend(senses_msg.senses)     

                    # process agent senses
                    for sense in senses:
                        # unpack message
                        msg : SimulationMessage = message_from_dict(**sense)

                        # check if message needs to be relayed
                        if msg.path:
                            msg.path.pop(0)
                            if msg.path:
                                msg_copy : SimulationMessage = message_from_dict(**msg.to_dict())
                                await self.relay_inbox.put(msg_copy)

                        # check type of message being received
                        if isinstance(msg, AgentActionMessage):
                            # agent action received
                            self.log(f"received agent action of status {msg.status}!")
                            
                            # send to planner
                            await self.action_status_inbox.put(msg)

                        elif isinstance(msg, AgentStateMessage):
                            # agent state received
                            self.log(f"received agent state message!")
                            
                            # unpack state
                            state : SimulationAgentState = SimulationAgentState.from_dict(msg.state)

                            # update parent agent information if the type of parent agent is unknown
                            if self.parent_agent_type is None:
                                if isinstance(state, SatelliteAgentState):
                                    # import orbit data
                                    self.orbitdata : dict = self._load_orbit_data()
                                    
                                    # parent is a satellite-type agent
                                    self.parent_agent_type = SimulationAgentTypes.SATELLITE.value
                                elif isinstance(state, UAVAgentState):
                                    # parent is a uav-type agent
                                    self.parent_agent_type = SimulationAgentTypes.UAV.value

                                elif isinstance(state, GroundStationAgentState):
                                    # parent is a ground station-type agent
                                    self.parent_agent_type = SimulationAgentTypes.GROUND_STATION.value

                                else:
                                    # parent is an agent of an unknown type; raise exception
                                    raise NotImplementedError(f"states of type {msg.state['state_type']} not supported for planners.")
                            
                            await self.states_inbox.put(state)

                        elif isinstance(msg, MeasurementRequestMessage):
                            # request received directly from another agent
                            req : MeasurementRequest = MeasurementRequest.from_dict(msg.req)
                            self.log(f"received measurement request message!")
                            
                            # send to planner
                            await self.req_inbox.put(req)

                        elif isinstance(msg, MeasurementResultsRequestMessage):
                            # measurement was just performed by agent
                            self.log(f"received measurement data from agent!")

                            # senf to planner
                            await self.measurement_inbox.put(msg)

                        # TODO support down-linked information processing
                        # elif isisntance(msg, DOWNLINKED MESSAGE CONFIRMATION):

                        else:
                            # other type of message was received
                            self.log(f"received some other kind of message!")

                            # send to planner
                            await self.misc_inbox.put(msg)

                else:
                    if not self.other_modules_exist:
                        # another type of message was received from another module 
                        self.other_modules_exist = True

                    # unpack message
                    msg : SimulationMessage = message_from_dict(**content)

                    # send to planner
                    await self.internal_inbox.put(msg)

        except asyncio.CancelledError:
            return
        
    def _load_orbit_data(self) -> dict:
        """
        Loads agent orbit data from pre-computed csv files in scenario directory
        """
        if self.parent_agent_type != None:
            raise RuntimeError(f"orbit data already loaded. It can only be assigned once.")            

        results_path_list = self.results_path.split('/')
        while "" in results_path_list:
            results_path_list.remove("")
        if 'results' in results_path_list[-1]:
            results_path_list.pop()
        if '.' in results_path_list[0]:
            results_path_list.pop(0)
        if 'scenarios' in results_path_list[0]:
            results_path_list.pop(0)


        scenario_name = []
        for name_element in results_path_list:
            scenario_name.extend([name_element, '/'])
        scenario_name = ''.join(scenario_name)

        scenario_dir = f'./scenarios/{scenario_name}'
        
        return OrbitData.from_directory(scenario_dir)
        
    """
    -----------------
         PLANNER
    -----------------
    """
    async def planner(self) -> None:
        """
        Generates a plan for the parent agent to perform
        """
        try:
            # initialize plan
            plan : Plan = Preplan(t=-1.0)

            # level = logging.WARNING
            level = logging.DEBUG

            while True:
                # wait for agent to update state
                state : SimulationAgentState = await self.states_inbox.get()
                
                # update internal clock
                await self.update_current_time(state.t)
                assert abs(self.get_current_time() - state.t) <= 1e-2

                # --- Check incoming information ---
                # Read incoming messages
                incoming_reqs, generated_reqs, relay_messages, misc_messages \
                    = await self._read_incoming_messages()

                # check action completion
                completed_actions, aborted_actions, pending_actions \
                    = await self.__check_action_completion(level)

                # remove aborted or completed actions from plan
                plan.update_action_completion(  
                                                completed_actions, 
                                                aborted_actions, 
                                                pending_actions, 
                                                self.get_current_time()
                                            )
                
                # --- FOR DEBUGGING PURPOSES ONLY: ---
                # self.__log_actions(completed_actions, aborted_actions, pending_actions)
                # -------------------------------------
                
                # --- Create plan ---
                if self.preplanner is not None:
                    # there is a preplanner assigned to this planner

                    # update preplanner precepts
                    self.preplanner.update_percepts(state,
                                                    plan, 
                                                    completed_actions,
                                                    aborted_actions,
                                                    pending_actions,
                                                    incoming_reqs,
                                                    generated_reqs,
                                                    relay_messages,
                                                    misc_messages,
                                                    self.orbitdata
                                                )
                    
                    # check if there is a need to construct a new plan
                    if self.preplanner.needs_planning(state, plan):     
                        # initialize plan      
                        plan : Plan = self.preplanner.generate_plan(    state, 
                                                                        plan,
                                                                        completed_actions,
                                                                        aborted_actions,
                                                                        pending_actions,
                                                                        incoming_reqs,
                                                                        generated_reqs,
                                                                        relay_messages,
                                                                        misc_messages,
                                                                        self._clock_config,
                                                                        self.orbitdata
                                                                        )

                        # save copy of plan for post-processing
                        plan_copy = [action for action in plan]
                        self.plan_history.append((state.t, plan_copy))
                        
                        # --- FOR DEBUGGING PURPOSES ONLY: ---
                        self.__log_plan(plan, "PRE-PLAN", logging.WARNING)
                        x = 1
                        # -------------------------------------

                # --- Modify plan ---
                # Check if reeplanning is needed
                if self.replanner is not None:
                    # there is a replanner assigned to this planner

                    # update replanner precepts
                    self.replanner.update_percepts( state,
                                                    plan, 
                                                    completed_actions,
                                                    aborted_actions,
                                                    pending_actions,
                                                    incoming_reqs,
                                                    generated_reqs,
                                                    relay_messages,
                                                    misc_messages,
                                                    self.orbitdata
                                                )
                    
                    if self.replanner.needs_planning(state, plan):
                        # --- FOR DEBUGGING PURPOSES ONLY: ---
                        self.__log_plan(plan, "ORIGINAL PLAN", logging.WARNING)
                        x = 1
                        # -------------------------------------

                        # Modify current Plan      
                        plan : Plan = self.replanner.generate_plan( state, 
                                                                    plan,
                                                                    completed_actions,
                                                                    aborted_actions,
                                                                    pending_actions,
                                                                    incoming_reqs,
                                                                    generated_reqs,
                                                                    relay_messages,
                                                                    misc_messages,
                                                                    self._clock_config,
                                                                    self.orbitdata
                                                                    )

                        # update last time plan was updated
                        self.t_plan = self.get_current_time()

                        # save copy of plan for post-processing
                        plan_copy = [action for action in plan]
                        self.plan_history.append((self.t_plan, plan_copy))
                    
                        # clear pending actions
                        pending_actions = []

                        # --- FOR DEBUGGING PURPOSES ONLY: ---
                        # self.__log_plan(plan, "REPLAN", logging.WARNING)
                        x = 1
                        # -------------------------------------

                # --- Execute plan ---
                # get next action to perform
                plan_out : list = plan.get_next_actions(self.get_current_time())

                # --- FOR DEBUGGING PURPOSES ONLY: ---
                # self.__log_plan(plan_out, "PLAN OUT", logging.WARNING)
                x = 1
                # -------------------------------------

                # send plan to parent agent
                self.log(f'sending {len(plan_out)} actions to parent agent...')
                plan_msg = PlanMessage(self.get_element_name(), self.get_network_name(), plan_out, self.get_current_time())
                await self._send_manager_msg(plan_msg, zmq.PUB)

                self.log(f'actions sent!')

        except asyncio.CancelledError:
            return
                
    @runtime_tracker
    async def __check_action_completion(self, level : int = logging.DEBUG) -> tuple:
        """
        Checks incoming messages from agent to check which actions from its plan have been completed, aborted, or are still pending

        ### Arguments:
            - level (`int`): logging level

        ### Returns:
            - `tuple` of action lists `completed_actions, aborted_actions, pending_actions`
        """
        completed_actions = []
        aborted_actions = []
        pending_actions = []

        while not self.action_status_inbox.empty():
            action_msg : AgentActionMessage = await self.action_status_inbox.get()
            action : AgentAction = action_from_dict(**action_msg.action)

            if action_msg.status == AgentAction.COMPLETED:
                # planned action completed! 
                self.log(f'action of type `{action.action_type}` completed!', level)
                completed_actions.append(action)
            elif action_msg.status == AgentAction.ABORTED:
                # planned action aborted! 
                self.log(f'action of type `{action.action_type}` aborted!', level)
                aborted_actions.append(action)
                pass
            elif action_msg.status == AgentAction.PENDING:
                # planned action wasn't completed
                self.log(f"action {action.id.split('-')[0]} not completed yet! trying again...", level)
                pending_actions.append(action)
            else: 
                # unknowhn action status; ignore
                continue
        
        return completed_actions, aborted_actions, pending_actions
    
    @runtime_tracker
    async def _read_incoming_messages(self) -> tuple:
        """
        Collects and classifies incoming messages. If a measurement was made by the parent agent,
        it waits until other modules to process the measurement's data and send a response.

        ## Returns:
            incoming_reqs 
            generated_reqs 
            misc_messages
        """
        incoming_reqs = []
        while not self.req_inbox.empty():
            req : MeasurementRequest = await self.req_inbox.get()
            incoming_reqs.append(req)

        incoming_measurements = []
        while not self.measurement_inbox.empty():
            incoming_measurements.append(await self.measurement_inbox.get())

        generated_reqs = []
        misc_messages = []
        if (self.other_modules_exist                # other modules exist within the parent agent
            and len(incoming_measurements) > 0      # some agent just performed a measurement
            ):

            # wait for science module to send their assesment of the measurement 
            while True:
                internal_msg = await self.internal_inbox.get()

                if not isinstance(internal_msg, MeasurementRequestMessage):
                    await self.misc_inbox.put(internal_msg)
                else:
                    generated_reqs.append( MeasurementRequest.from_dict(internal_msg.req) )

                if self.internal_inbox.empty():
                    break

        while not self.misc_inbox.empty():
            misc_messages.append(await self.misc_inbox.get())

        relay_messages= []
        while not self.relay_inbox.empty():
            relay_messages.append(await self.relay_inbox.get())

        return incoming_reqs, generated_reqs, relay_messages, misc_messages
    
    def __log_actions(self, completed_actions : list, aborted_actions : list, pending_actions : list) -> None:
        all_actions = [action for action in completed_actions]
        all_actions.extend([action for action in aborted_actions])
        all_actions.extend([action for action in pending_actions])

        if len(all_actions) > 0:
            out = '\nACTION STATUS:\n'
            for action in all_actions:
                action : AgentAction
                out += f"{action.id.split('-')[0]}, {action.action_type}, {action.t_start}, {action.t_end}\n"
            out += '\n'
            self.log(out, logging.WARNING)

    def __log_plan(self, plan : Plan, title : str, level : int = logging.DEBUG) -> None:
        out = f'\n{title}\n'

        if isinstance(plan, Plan):
            out += str(plan)
        else:
            for action in plan:
                if isinstance(action, AgentAction):
                    out += f"{action.id.split('-')[0]}, {action.action_type}, {action.t_start}, {action.t_end}\n"
                elif isinstance(action, dict):
                    out += f"{action['id'].split('-')[0]}, {action['action_type']}, {action['t_start']}, {action['t_end']}\n"
        

        self.log(out, level)
               
    async def teardown(self) -> None:
        # log plan history
        headers = ['plan_index', 't_plan', 'req_id', 'subtask_index', 't_img', 'u_exp']
        data = []
        
        for i in range(len(self.plan_history)):
            t_plan, plan = self.plan_history[i]
            t_plan : float; plan : list

            for action in plan:
                if not isinstance(action, MeasurementAction):
                    continue

                req : MeasurementRequest = MeasurementRequest.from_dict(action.measurement_req)
                line_data = [   i,
                                np.round(t_plan,3),
                                req.id.split('-')[0],
                                action.subtask_index,
                                np.round(action.t_start,3 ),
                                np.round(action.u_exp,3)
                ]
                data.append(line_data)

        df = pd.DataFrame(data, columns=headers)
        self.log(f'\nPLANNER HISTORY\n{str(df)}\n', level=logging.WARNING)
        df.to_csv(f"{self.results_path}/{self.get_parent_name()}/planner_history.csv", index=False)

        # log performance stats
        n_decimals = 3
        headers = ['routine','t_avg','t_std','t_med','n']
        data = []

        for routine in self.stats:
            n = len(self.stats[routine])
            t_avg = np.round(np.mean(self.stats[routine]),n_decimals) if n > 0 else -1
            t_std = np.round(np.std(self.stats[routine]),n_decimals) if n > 0 else 0.0
            t_median = np.round(np.median(self.stats[routine]),n_decimals) if n > 0 else -1

            line_data = [ 
                            routine,
                            t_avg,
                            t_std,
                            t_median,
                            n
                            ]
            data.append(line_data)

        if isinstance(self.preplanner, AbstractPreplanner):
            for routine in self.preplanner.stats:
                n = len(self.preplanner.stats[routine])
                t_avg = np.round(np.mean(self.preplanner.stats[routine]),n_decimals) if n > 0 else -1
                t_std = np.round(np.std(self.preplanner.stats[routine]),n_decimals) if n > 0 else 0.0
                t_median = np.round(np.median(self.preplanner.stats[routine]),n_decimals) if n > 0 else -1

                line_data = [ 
                                f"preplanner/{routine}",
                                t_avg,
                                t_std,
                                t_median,
                                n
                                ]
                data.append(line_data)

        if isinstance(self.replanner, AbstractReplanner):
            for routine in self.replanner.stats:
                n = len(self.replanner.stats[routine])
                t_avg = np.round(np.mean(self.replanner.stats[routine]),n_decimals) if n > 0 else -1
                t_std = np.round(np.std(self.replanner.stats[routine]),n_decimals) if n > 0 else 0.0
                t_median = np.round(np.median(self.replanner.stats[routine]),n_decimals) if n > 0 else -1

                line_data = [ 
                                f"replanner/{routine}",
                                t_avg,
                                t_std,
                                t_median,
                                n
                                ]
                data.append(line_data)

        stats_df = pd.DataFrame(data, columns=headers)
        self.log(f'\nPLANNER RUN-TIME STATS\n{str(stats_df)}\n', level=logging.WARNING)
        stats_df.to_csv(f"{self.results_path}/{self.get_parent_name()}/planner_runtime_stats.csv", index=False)

        await super().teardown()
