import logging
import os
from dmas.messages import SimulationMessage
import numpy as np
from pandas import DataFrame

from instrupy.base import Instrument
from orbitpy.util import Spacecraft

from dmas.agents import *
from dmas.network import NetworkConfig
from dmas.utils import runtime_tracker
from zmq import SocketType

from chess3d.agents.states import SimulationAgentState
from chess3d.agents.science.requests import MeasurementRequest
from chess3d.agents.planning.module import PlanningModule
from chess3d.agents.science.module import ScienceModule
from chess3d.agents.actions import *
from chess3d.messages import *

class SimulationAgent(Agent):
    """
    # Abstract Simulation Agent

    #### Attributes:
        - agent_name (`str`): name of the agent
        - scenario_name (`str`): name of the scenario being simulated
        - manager_network_config (:obj:`NetworkCdo(nfig`): netwdo(rk configuration of the simulation manager
        - agent_network_config (:obj:`NetworkConfig`): network configuration for this agent
        - initial_state (:obj:`SimulationAgentState`): initial state for this agent
        - specs (`object`): describes the physical specifications of this agent
        - planning_module (`PlanningModule`): planning module assigned to this agent
        - science_module (`ScienceModule): science module assigned to this agent
        - level (int): logging level
        - logger (logging.Logger): simulation logger 
    """
    def __init__(   self, 
                    agent_name: str, 
                    results_path : str,
                    manager_network_config: NetworkConfig, 
                    agent_network_config: NetworkConfig,
                    initial_state : SimulationAgentState,
                    specs : object,
                    planning_module : PlanningModule = None,
                    science_module : ScienceModule = None,
                    level: int = logging.INFO, 
                    logger: logging.Logger = None
                    ) -> None:
        """
        Initializes an instance of a Simulation Agent Object

        #### Arguments:
            - agent_name (`str`): name of the agent
            - scenario_name (`str`): name of the scenario being simulated
            - manager_network_config (:obj:`NetworkConfig`): network configuration of the simulation manager
            - agent_network_config (:obj:`NetworkConfig`): network configuration for this agent
            - initial_state (:obj:`SimulationAgentState`): initial state for this agent
            - payload (`list): list of instruments on-board the spacecraft
            - planning_module (`PlanningModule`): planning module assigned to this agent
            - science_module (`ScienceModule): science module assigned to this agent
            - level (int): logging level
            - logger (logging.Logger): simulation logger 
        """
        # load agent modules
        modules = []
        if planning_module is not None:
            if not isinstance(planning_module, PlanningModule):
                raise AttributeError(f'`planning_module` must be of type `PlanningModule`; is of type {type(planning_module)}')
            modules.append(planning_module)
        if science_module is not None:
            if not isinstance(science_module, ScienceModule):
                raise AttributeError(f'`science_module` must be of type `ScienceModule`; is of type {type(science_module)}')
            modules.append(science_module)

        # initialize agent
        super().__init__(agent_name, 
                        agent_network_config, 
                        manager_network_config, 
                        initial_state, 
                        modules, 
                        level, 
                        logger)

        self.specs = specs
        if isinstance(self.specs, Spacecraft):
            self.payload = {instrument.name: instrument for instrument in self.specs.instrument}
        elif isinstance(self.specs, dict):
            self.payload = {instrument.name: instrument for instrument in self.specs['instrument']} if 'instrument' in self.specs else dict()
        else:
            raise ValueError(f'`specs` must be of type `Spacecraft` or `dict`. Is of type `{type(specs)}`.')


        self.state_history : list = []
        
        # setup results folder:
        self.results_path = os.path.join(results_path, self.get_element_name())
    
    """
    --------------------
            SENSE       
    --------------------
    """
    
    @runtime_tracker
    async def sense(self, statuses: list) -> list:
        # initiate senses array
        senses = []

        # check status of previously performed tasks
        action_statuses = self.check_action_statuses(statuses)
        senses.extend(action_statuses)

        # update state
        self.update_state()

        # sense environment
        env_updates = await self.sense_environment()
        env_resp = BusMessage(**env_updates)

        # update agent with environment senses
        env_senses = await self.update_state_environment(env_resp)
        senses.extend(env_senses)

        # handle environment broadcasts
        env_broadcasts = await self.get_environment_broadcasts()
        senses.extend(env_broadcasts)

        # handle peer broadcasts
        agent_broadcasts = await self.get_agent_broadcasts()
        senses.extend(agent_broadcasts)

        return senses
    
    @runtime_tracker
    def check_action_statuses(self, statuses) -> list:
        senses = []
        for action, status in statuses:
            # sense and compile updated task status for planner 
            action : AgentAction
            status : str
            msg = AgentActionMessage(   self.get_element_name(), 
                                        self.get_element_name(), 
                                        action.to_dict(),
                                        status)
            senses.append(msg)    
        return senses
    
    @runtime_tracker
    def update_state(self) -> None:
        self.state.update_state(self.get_current_time(), 
                                status=SimulationAgentState.SENSING)
        self.state_history.append(self.state.to_dict())

    @runtime_tracker
    async def sense_environment(self) -> dict:
        state_msg = AgentStateMessage(  self.get_element_name(), 
                                        SimulationElementRoles.ENVIRONMENT.value,
                                        self.state.to_dict()
                                    )
        _, _, content = await self.send_peer_message(state_msg)

        return content
    
    @runtime_tracker
    async def update_state_environment(self, env_resp : BusMessage)-> list:
        senses = []
        for resp in env_resp.msgs:
            # unpackage message
            resp : dict
            resp_msg : SimulationMessage = message_from_dict(**resp)

            if isinstance(resp_msg, AgentStateMessage):
                # update state
                state_msg = AgentStateMessage(  self.get_element_name(), 
                                                self.get_element_name(),
                                                self.state.to_dict()
                                            )
                senses.append(state_msg)               

            elif isinstance(resp_msg, AgentConnectivityUpdate):
                if resp_msg.connected == 1:
                    self.subscribe_to_broadcasts(resp_msg.target)
                else:
                    self.unsubscribe_to_broadcasts(resp_msg.target)
        return senses 
    
    async def __empty_queue(self, q : asyncio.Queue) -> list:
        msgs = []
        while not q.empty():
            # save as senses to forward to planner
            _, _, d = await q.get()
            msgs.append(message_from_dict(**d))

            # give other agents time to finish sending their messages
            await asyncio.sleep(1e-2)
        return msgs
    
    @runtime_tracker
    async def get_environment_broadcasts(self) -> list:
        return await self.__empty_queue(self.environment_inbox)
    
    @runtime_tracker
    async def get_agent_broadcasts(self) -> list:
        return await self.__empty_queue(self.external_inbox)

    """
    --------------------
            THINK       
    --------------------
    """
    @runtime_tracker
    async def think(self, senses: list) -> list:
        # send all sensed messages to planner
        self.log(f'sending {len(senses)} senses to planning module...', level=logging.DEBUG)
        senses_dict = []
        state_dict = None
        for sense in senses:
            sense : SimulationMessage
            if isinstance(sense, AgentStateMessage):
                state_dict = sense.to_dict()
            else:
                senses_dict.append(sense.to_dict())

        senses_msg = SenseMessage( self.get_element_name(), 
                                    self.get_element_name(),
                                    state_dict, 
                                    senses_dict)
        await self.send_internal_message(senses_msg)

        # wait for planner to send list of tasks to perform
        self.log(f'senses sent! waiting on response from planner module...')
        actions = []
        
        while True:
            _, _, content = await self.internal_inbox.get()
            
            if content['msg_type'] == SimulationMessageTypes.PLAN.value:
                msg = PlanMessage(**content)

                # assert self.get_current_time() - msg.t_plan <= 1e-3

                for action_dict in msg.plan:
                    self.log(f"received an action of type {action_dict['action_type']}", level=logging.DEBUG)
                    actions.append(action_dict)  
                break
        
        self.log(f"plan of {len(actions)} actions received from planner module!")
        return actions

    """
    --------------------
            DO       
    --------------------
    """
    @runtime_tracker
    async def do(self, actions: list) -> dict:
        self.log(f'performing {len(actions)} actions', level=logging.DEBUG)

        # perform each action and record action status
        statuses = []
        for action_dict in actions:
            action : AgentAction = action_from_dict(**action_dict)

            # check action start time
            if (action.t_start - self.get_current_time()) > 1e-6:
                self.log(f"action of type {action.action_type} has NOT started yet (start time {action.t_start}[s]). waiting for start time...", level=logging.ERROR)
                action.status = AgentAction.PENDING
                statuses.append((action, action.status))

                raise RuntimeError(f"agent {self.get_element_name()} attempted to perform action of type {action.action_type} before it started (start time {action.t_start}[s]) at time {self.get_current_time()}[s]")
            
            # check action end time
            if (self.get_current_time() - action.t_end) > 1e-6:
                self.log(f"action of type {action.action_type} has already occureed (start/end times {action.t_start}[s], {action.t_end}[s]). could not perform task before...", level=logging.ERROR)
                action.status = AgentAction.ABORTED
                statuses.append((action, action.status))

                raise RuntimeError(f"agent {self.get_element_name()} attempted to perform action of type {action.action_type} after it ended (start/end times {action.t_start}[s], {action.t_end}[s]) at time {self.get_current_time()}[s]")

            # perform each action depending on 
            self.log(f"performing action of type {action.action_type}...", level=logging.INFO)    
            if (action.action_type == ActionTypes.IDLE.value         
                or action.action_type == ActionTypes.TRAVEL.value
                or action.action_type == ActionTypes.MANEUVER.value): 
                # update agent state
                action.status = await self.perform_state_change(action)

            elif action.action_type == ActionTypes.BROADCAST.value:
                # perform message broadcast
                action.status = await self.perform_broadcast(action)
            
            elif action.action_type == ActionTypes.WAIT.value:
                # wait for incoming messages
                action.status = await self.perform_wait_for_messages(action)

            elif action.action_type == ActionTypes.OBSERVE.value:                              
                # perform observation
                action.status = await self.perform_observation(action)
                
            else: # unknown action type; ignore action
                self.log(f"action of type {action.action_type} not yet supported. ignoring...", level=logging.INFO)
                action.status = AgentAction.ABORTED  
                
            # record action completion status 
            self.log(f"finished performing action of type {action.action_type}! action completion status: {action.status}", level=logging.INFO)
            statuses.append((action, action.status))

        # return list of statuses
        self.log(f'returning {len(statuses)} statuses', level=logging.DEBUG)
        return statuses

    @runtime_tracker
    async def perform_state_change(self, action : AgentAction) -> str:
        """ Performs actions that have an effect on the agent's state """
        t = self.get_current_time()

        # modify the agent's state
        status, dt = self.state.perform_action(action, t)
        self.state_history.append(self.state.to_dict())

        if dt > 0:
            # perfrom time wait if needed
            await self.perform_wait_for_messages(WaitForMessages(t, t+dt), False)

            # update the agent's state
            status, _ = self.state.perform_action(action, self.get_current_time())
            self.state_history.append(self.state.to_dict())

        # return completion status
        return status
    
    @runtime_tracker
    async def perform_broadcast(self, action : BroadcastMessageAction) -> str:
        # unpackage action
        msg_out : SimulationMessage = message_from_dict(**action.msg)

        # update state
        self.state.update_state(self.get_current_time(), status=SimulationAgentState.MESSAGING)
        self.state_history.append(self.state.to_dict())
        
        # perform action
        msg_out.src = self.get_element_name()
        msg_out.dst = self.get_network_name()
        await self.send_peer_broadcast(msg_out)

        self.log(f'\n\tSent broadcast!\n\tfrom:\t{msg_out.src}\n\tto:\t{msg_out.dst}',level=logging.DEBUG)

        return AgentAction.COMPLETED
    
    @runtime_tracker
    async def perform_wait_for_messages(self, action : WaitForMessages, save : bool = True) -> str:
        """ Waits for a message from another agent to be received. """
        
        # get current start time
        t_curr = self.get_current_time()
        self.state.update_state(t_curr, status=SimulationAgentState.LISTENING)
        if save: self.state_history.append(self.state.to_dict())

        # check if messages have already been received
        if not self.external_inbox.empty(): # messages in inbox; end wait
            return AgentAction.COMPLETED
        
        else: # no messages in inbox; wait for incoming messages

            # check type of simulation clock
            if ((isinstance(self._clock_config, FixedTimesStepClockConfig) 
                or isinstance(self._clock_config, EventDrivenClockConfig)) 
                and self.external_inbox.empty()
                ):
                # give the agent time to finish processing messages before submitting a tic-request
                t_wait = 1e-3 if t_curr < 1e-3 else 1e-5
                await asyncio.sleep(t_wait)

            # initiate broadcast wait and timeout tasks
            receive_broadcast = asyncio.create_task(self.external_inbox.get())
            timeout = asyncio.create_task(self.sim_wait(action.t_end - t_curr))

            # wait for first task to be completed
            done, _ = await asyncio.wait([timeout, receive_broadcast], return_when=asyncio.FIRST_COMPLETED)

            # check which task was finished first 
            if receive_broadcast in done:
                # messages were received before timeout
                try:
                    # cancel timeout timer and end wait
                    timeout.cancel()
                    await timeout

                except asyncio.CancelledError:
                    # restore message to inbox so it can be processed during `sense()`
                    await self.external_inbox.put(receive_broadcast.result())    

                    # update action completion status
                    return AgentAction.COMPLETED                

            else:
                # timouet ended
                try:
                    # cancel message wait
                    receive_broadcast.cancel()
                    await receive_broadcast

                except asyncio.CancelledError:
                    # update action completion status
                    if self.external_inbox.empty():
                        return AgentAction.ABORTED
                    else:
                        return AgentAction.COMPLETED
                    
    @runtime_tracker
    async def perform_observation(self, action : ObservationAction) -> str:
        # update agent state and state history
        self.state : SimulationAgentState
        self.state.update_state(self.get_current_time(), 
                                status=SimulationAgentState.MEASURING)
        self.state_history.append(self.state.to_dict())
        
        try:
            # find relevant instrument information 
            instrument : Instrument = self.payload[action.instrument_name]

            # create observation data request
            observation_req = ObservationResultsMessage(
                                                    self.get_element_name(),
                                                    SimulationElementRoles.ENVIRONMENT.value,
                                                    self.state.to_dict(),
                                                    action.to_dict(),
                                                    instrument.to_dict()
                                                    )

            # request measurement data from the environment
            dst,src,observation_results = await self.send_peer_message(observation_req)
            msg_sci = ObservationResultsMessage(**observation_results)
            
            # send measurement data to results logger
            # await self._send_manager_msg(msg_sci, zmq.PUB)
            await self._send_manager_msg(msg_sci, zmq.PUSH)

            # send measurement to environment inbox to be processed during `sensing()`
            await self.environment_inbox.put((dst, src, observation_results))

            # wait for the designated duration of the measurmeent 
            dt = action.t_end - self.get_current_time()
            if dt > 0: await self.sim_wait(dt) 

            # return action completion            
            return AgentAction.COMPLETED

        except asyncio.CancelledError:
            # action was aborted 
            return AgentAction.ABORTED

    """
    --------------------
          TEARDOWN       
    --------------------
    """
    async def _publish_deactivate(self) -> None:
        # notify monitor
        await super()._publish_deactivate()

        # notify manager
        manager_message = NodeDeactivatedMessage(self.get_element_name(), SimulationElementRoles.MANAGER.value)
        await self._send_manager_msg(manager_message, zmq.PUB)

        x = 1

    async def teardown(self) -> None:
        try:
            # TODO log agent capabilities

            # log states
            n_decimals = 3
            headers = ['t', 'x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel', 'z_vel', 'attitude', 'status']
            data = []

            for state_dict in self.state_history:
                line_data = [
                                np.round(state_dict['t'],3),

                                np.round(state_dict['pos'][0],n_decimals),
                                np.round(state_dict['pos'][1],n_decimals),
                                np.round(state_dict['pos'][2],n_decimals),

                                np.round(state_dict['vel'][0],n_decimals),
                                np.round(state_dict['vel'][1],n_decimals),
                                np.round(state_dict['vel'][2],n_decimals),
                                
                                np.round(state_dict['attitude'][0],n_decimals),

                                state_dict['status']
                            ]
                data.append(line_data)
            
            state_df = DataFrame(data,columns=headers)
            # self.log(f'\nSTATE HISTORY\n{str(state_df)}\n', level=logging.WARNING)
            state_df.to_csv(f"{self.results_path}/states.csv", index=False)

            # log performance stats
            runtime_dir = os.path.join(self.results_path, "runtime")
            if not os.path.isdir(runtime_dir): os.mkdir(runtime_dir)

            headers = ['routine','t_avg','t_std','t_med','t_max','t_min','n', 't_total']
            data = []

            for routine in self.stats:
                # compile stats
                t_avg = np.mean(self.stats[routine])
                t_std = np.std(self.stats[routine])
                t_median = np.median(self.stats[routine])
                t_max = max(self.stats[routine])
                t_min = min(self.stats[routine])
                n = len(self.stats[routine])
                t_total = n * t_avg

                line_data = [ 
                                routine,
                                np.round(t_avg,n_decimals),
                                np.round(t_std,n_decimals),
                                np.round(t_median,n_decimals),
                                t_max,
                                t_min,
                                n,
                                t_total
                                ]
                data.append(line_data)

                # save time-series
                time_series = [[v] for v in self.stats[routine]]
                routine_df = DataFrame(data=time_series, columns=['dt'])
                routine_dir = os.path.join(runtime_dir, f"time_series-{routine}.csv")
                routine_df.to_csv(routine_dir,index=False)

            stats_df = DataFrame(data, columns=headers)
            # self.log(f'\nAGENT RUN-TIME STATS\n{str(stats_df)}\n', level=logging.WARNING)
            stats_df.to_csv(f"{self.results_path}/agent_runtime_stats.csv", index=False)
        except Exception as e:
            x = 1

    async def sim_wait(self, delay: float) -> None:
        try:  
            if (
                isinstance(self._clock_config, FixedTimesStepClockConfig) 
                or isinstance(self._clock_config, EventDrivenClockConfig)
                ):
                if delay == 0.0: 
                    ignored = None
                    return

                # desired time not yet reached
                t0 = self.get_current_time()
                tf = t0 + delay

                # check if failure or critical state will be reached first
                self.state : SimulationAgentState
                if self.state.engineering_module is not None:
                    t_failure = self.state.engineering_module.predict_failure() 
                    t_crit = self.state.engineering_module.predict_critical()
                    t_min = min(t_failure, t_crit)
                    tf = t_min if t_min < tf else tf
                
                # wait for time update        
                ignored = []   

                while self.get_current_time() <= t0:
                    # send tic request
                    tic_req = TicRequest(self.get_element_name(), t0, tf)
                    toc_msg = None
                    confirmation = None
                    confirmation = await self._send_manager_msg(tic_req, zmq.PUB)

                    self.log(f'tic request for {tf}[s] sent! waiting on toc broadcast...')
                    dst, src, content = await self.manager_inbox.get()
                    
                    if content['msg_type'] == ManagerMessageTypes.TOC.value:
                        # update clock
                        toc_msg = TocMessage(**content)
                        await self.update_current_time(toc_msg.t)
                        self.log(f'toc received! time updated to: {self.get_current_time()}[s]')

                    else:
                        # ignore message
                        self.log(f'some other manager message was received. ignoring...')
                        ignored.append((dst, src, content))

            elif isinstance(self._clock_config, AcceleratedRealTimeClockConfig):
                await asyncio.sleep(delay / self._clock_config.sim_clock_freq)

            else:
                raise NotImplementedError(f'`sim_wait()` for clock of type {type(self._clock_config)} not yet supported.')
        
        except asyncio.CancelledError as e:
            # if still waiting on  cancel request
            if confirmation is not None and toc_msg is None:
                tic_cancel = CancelTicRequest(self.get_element_name(), t0, tf)
                await self._send_manager_msg(tic_cancel, zmq.PUB)

            raise e

        finally:
            if (
                isinstance(self._clock_config, FixedTimesStepClockConfig) 
                or isinstance(self._clock_config, EventDrivenClockConfig)
                ) and ignored is not None:

                # forward all ignored messages as manager messages
                for dst, src, content in ignored:
                    await self.manager_inbox.put((dst,src,content))
    
    @runtime_tracker
    async def _send_manager_msg(self, msg: SimulationMessage, socket_type: SocketType) -> bool:
        return await super()._send_manager_msg(msg, socket_type)

    @runtime_tracker
    async def send_internal_message(self, msg: SimulationMessage) -> tuple:
        return await super().send_internal_message(msg)

    @runtime_tracker
    async def send_peer_message(self, msg: SimulationMessage) -> tuple:
        return await super().send_peer_message(msg)

    @runtime_tracker
    async def send_peer_broadcast(self, msg: SimulationMessage) -> None:
        return await super().send_peer_broadcast(msg)