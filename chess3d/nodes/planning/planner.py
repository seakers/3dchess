from typing import Any, Callable
import pandas as pd

from dmas.modules import *
from dmas.utils import runtime_tracker

from nodes.planning.plan import Plan, Preplan
from nodes.planning.preplanners import AbstractPreplanner
from nodes.planning.replanners import AbstractReplanner
from nodes.orbitdata import OrbitData
from nodes.states import *
from chess3d.nodes.science.requests import *
from messages import *

class PlanningModule(InternalModule):
    def __init__(self, 
                results_path : str, 
                parent_name : str,
                parent_network_config: NetworkConfig, 
                preplanner : AbstractPreplanner = None,
                replanner : AbstractReplanner = None,
                orbitdata : OrbitData = None,
                level: int = logging.INFO, 
                logger: logging.Logger = None
                ) -> None:
        """ Internal Agent Module in charge of planning an scheduling. """
                       
        # setup network settings
        planner_network_config : NetworkConfig = self._setup_planner_network_config(parent_name, parent_network_config)

        # intialize module
        super().__init__(f'{parent_name}-PLANNING_MODULE', 
                        planner_network_config, 
                        parent_network_config, 
                        level, 
                        logger)
        
        # initialize default attributes
        self.plan_history = []
        self.stats = {
                    }
        self.agent_state : SimulationAgentState = None
        self.other_modules_exist : bool = False
        
        # set parameters
        self.results_path = results_path
        self.parent_name = parent_name
        self.preplanner : AbstractPreplanner = preplanner
        self.replanner : AbstractReplanner = replanner
        self.orbitdata : OrbitData = orbitdata


    def _setup_planner_network_config(self, parent_name : str, parent_network_config : NetworkConfig) -> dict:
        """ Sets up network configuration for intra-agent module communication """
        # addresses : dict = parent_network_config.get_internal_addresses()        
        # sub_addesses = []
        # sub_address : str = addresses.get(zmq.PUB)[0]
        # sub_addesses.append( sub_address.replace('*', 'localhost') )

        # if len(addresses.get(zmq.SUB)) > 1:
        #     sub_address : str = addresses.get(zmq.SUB)[1]
        #     sub_addesses.append( sub_address.replace('*', 'localhost') )

        # pub_address : str = addresses.get(zmq.SUB)[0]
        # pub_address = pub_address.replace('localhost', '*')

        # addresses = parent_network_config.get_manager_addresses()
        # push_address : str = addresses.get(zmq.PUSH)[0]

        # return  NetworkConfig(  parent_name,
        #                         manager_address_map = {
        #                             zmq.REQ: [],
        #                             zmq.SUB: sub_addesses,
        #                             zmq.PUB: [pub_address],
        #                             zmq.PUSH: [push_address]
        #                         }
        #                     )

        # get full list of addresses from parent agent
        addresses : dict = parent_network_config.get_internal_addresses()        
        
        # subscribe to parent agent's bradcasts
        sub_address : str = addresses.get(zmq.PUB)[0]
        sub_addesses = [ sub_address.replace('*', 'localhost') ]

        # subscribe to science module's broadcasts
        if len(addresses.get(zmq.SUB)) > 1:
            sub_address : str = addresses.get(zmq.SUB)[1]
            sub_addesses.append( sub_address.replace('*', 'localhost') )

        # obtain address for intra-agent broadcasts
        pub_address : str = addresses.get(zmq.SUB)[0]
        pub_address = pub_address.replace('localhost', '*')

        # obtain results logger address
        addresses = parent_network_config.get_manager_addresses()
        push_address : str = addresses.get(zmq.PUSH)[0]

        # return network configuration
        return  NetworkConfig(  parent_name,
                                manager_address_map = {
                                    zmq.REQ: [],
                                    zmq.SUB: sub_addesses,
                                    zmq.PUB: [pub_address],
                                    zmq.PUSH: [push_address]
                                }
                            )

    async def setup(self) -> None:
        # initialize internal messaging queues
        self.states_inbox = asyncio.Queue()
        self.action_status_inbox = asyncio.Queue()
        self.req_inbox = asyncio.Queue()
        self.observations_inbox = asyncio.Queue()
        self.relay_inbox = asyncio.Queue()
        self.misc_inbox = asyncio.Queue()

        # setup agent state locks
        self.agent_state_lock = asyncio.Lock()
        self.agent_state_updated = asyncio.Event()

    async def sim_wait(self, delay: float) -> None:
        # does nothing; planner module is designed to be event-driven, not time-driven
        return

    async def live(self) -> None:
        """
        Performs two concurrent tasks:
        - Listener: receives messages from the parent agent and checks results
        - Planner: plans and bids according to local information
        """
        try:
            listener_task = asyncio.create_task(self.listener(), name='listener()')
            planner_task = asyncio.create_task(self.planner(), name='planner()')
            
            tasks = [listener_task, planner_task]

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
                # wait for next message from the parent agent
                self.log('listening to manager broadcast!')
                _, _, content = await self.listen_manager_broadcast()

                # classify mesasge content based of type of message
                if content['msg_type'] == ManagerMessageTypes.SIM_END.value:
                    # sim-end message; end agent `live()`
                    self.log(f"received manager broadcast or type {content['msg_type']}! terminating `live()`...")
                    return

                elif content['msg_type'] == SimulationMessageTypes.SENSES.value:
                    # received agent sense message
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

                        # check relay path
                        if msg.path:
                            # message contains relay path; forward message copy
                            await self.relay_inbox.put(message_from_dict(**sense))

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
                            
                            # send to planner
                            await self.states_inbox.put(state)

                        elif isinstance(msg, MeasurementRequestMessage):
                            # measurement request message received
                            self.log(f"received measurement request message!")

                            # unapack measurement request
                            req : MeasurementRequest = MeasurementRequest.from_dict(msg.req)
                            
                            # send to planner
                            await self.req_inbox.put(req)

                        elif isinstance(msg, ObservationResultsMessage):
                            # observation data from another agent was received
                            self.log(f"received observation data from agent!")

                            # send to planner
                            await self.observations_inbox.put(msg)

                        # TODO support down-linked information processing
                        ## elif isinstance(msg, DOWNLINKED MESSAGE CONFIRMATION):
                        ##     pass

                        else:
                            # other type of message was received
                            self.log(f"received some other kind of message!")

                            # send to planner
                            await self.misc_inbox.put(msg)

                else:
                    if not self.other_modules_exist:
                        # another type of message was received from another module 
                        self.other_modules_exist = True
                        continue

                    # unpack message
                    msg : SimulationMessage = message_from_dict(**content)

                    # send to planner
                    await self.internal_inbox.put(msg)

        except asyncio.CancelledError:
            return
        
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
                incoming_reqs, relay_messages, misc_messages \
                    = await self._read_incoming_messages()

                # check action completion
                completed_actions, aborted_actions, pending_actions \
                    = await self.__check_action_completion(level)

                # remove aborted or completed actions from plan
                plan.update_action_completion(  completed_actions, 
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
                                                    incoming_reqs,
                                                    relay_messages,
                                                    misc_messages,
                                                    completed_actions,
                                                    aborted_actions,
                                                    pending_actions
                                                )
                    
                    # check if there is a need to construct a new plan
                    if self.preplanner.needs_planning(state, plan):     
                        # initialize plan      
                        plan : Plan = self.preplanner.generate_plan(    state, 
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
                                                    relay_messages,
                                                    misc_messages
                                                )
                    
                    if self.replanner.needs_planning(state, plan):
                        # --- FOR DEBUGGING PURPOSES ONLY: ---
                        # self.__log_plan(plan, "ORIGINAL PLAN", logging.WARNING)
                        x = 1
                        # -------------------------------------

                        # Modify current Plan      
                        plan : Plan = self.replanner.generate_plan( state, 
                                                                    plan,
                                                                    completed_actions,
                                                                    aborted_actions,
                                                                    pending_actions,
                                                                    incoming_reqs,
                                                                    relay_messages,
                                                                    misc_messages,
                                                                    self._clock_config
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

        # collect all action statuses from inbox
        actions = []
        while not self.action_status_inbox.empty():
            action_msg : AgentActionMessage = await self.action_status_inbox.get()
            actions.append(action_from_dict(**action_msg.action))

        # classify by action completion
        completed_actions = [action for action in actions
                            if isinstance(action, AgentAction)
                            and action.status == AgentAction.COMPLETED] # planned action completed
                
        aborted_actions = [action for action in actions 
                           if isinstance(action, AgentAction)
                           and action.status == AgentAction.ABORTED]    # planned action aborted
                
        pending_actions = [action for action in actions
                           if isinstance(action, AgentAction)
                           and action_msg.status == AgentAction.PENDING] # planned action wasn't completed

        # return classified lists
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
        requests = []
        while not self.req_inbox.empty():
            requests.append(await self.req_inbox.get())

        incoming_obsevations = []
        while not self.observations_inbox.empty():
            incoming_obsevations.append(await self.observations_inbox.get())

        if (self.other_modules_exist                # science module exists within the parent agent
            and len(incoming_obsevations) > 0       # some agent just performed an observation
            ):

            while True:
                # wait for science module to send their assesment of the observation 
                internal_msg = await self.internal_inbox.get()

                # check the type of response from the science module
                if isinstance(internal_msg, MeasurementRequestMessage):
                    # the science module analized out latest observation(s)
                    request : MeasurementRequest = MeasurementRequest.from_dict(internal_msg.req)
                    
                    # check if outlier was deteced
                    if request.severity > 0.0:
                        # event was detected and an observation was requested
                        requests.append(request)
                else:
                    # the science module generated a different response; process later
                    await self.misc_inbox.put(internal_msg)

                # check for if there are more responses from the science module
                if self.internal_inbox.empty():
                    # no more messages from the science module; stop wait
                    break

        relay_messages= []
        while not self.relay_inbox.empty():
            relay_messages.append(await self.relay_inbox.get())

        misc_messages = []
        while not self.misc_inbox.empty():
            misc_messages.append(await self.misc_inbox.get())

        return requests, relay_messages, misc_messages
    
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
                if not isinstance(action, ObservationAction):
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
