import logging
import os
from dmas.messages import SimulationMessage
import numpy as np
import pandas as pd

from instrupy.base import Instrument
from orbitpy.util import Spacecraft

from dmas.agents import *
from dmas.modules import InternalModule
from dmas.utils import runtime_tracker
from zmq import SocketType

from chess3d.agents.planning.plan import *
from chess3d.agents.planning.planner import AbstractPreplanner, AbstractReplanner
from chess3d.agents.planning.rewards import RewardGrid
from chess3d.agents.science.requests import MeasurementRequest
from chess3d.agents.states import SimulationAgentState
from chess3d.agents.actions import *
from chess3d.messages import *
from chess3d.agents.planning.module import PlanningModule
from chess3d.agents.science.module import ScienceModule
from chess3d.agents.science.processing import DataProcessor

class AbstractAgent(Agent):
    """
    # Abstract Simulation Agent
    """
    def __init__(self, 
                 agent_name, 
                 results_path : str,
                 agent_network_config, 
                 manager_network_config, 
                 initial_state, 
                 specs : object,
                 modules, 
                 level = logging.INFO, 
                 logger = None):
        super().__init__(agent_name, agent_network_config, manager_network_config, initial_state, modules, level, logger)
        
        self.specs = specs
        if isinstance(self.specs, Spacecraft):
            self.payload = {instrument.name: instrument for instrument in self.specs.instrument}
        elif isinstance(self.specs, dict):
            self.payload = {instrument['name']: instrument for instrument in self.specs['instrument']} if 'instrument' in self.specs else dict()
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
            
            state_df = pd.DataFrame(data,columns=headers)
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
                routine_df = pd.DataFrame(data=time_series, columns=['dt'])
                routine_dir = os.path.join(runtime_dir, f"time_series-{routine}.csv")
                routine_df.to_csv(routine_dir,index=False)

            stats_df = pd.DataFrame(data, columns=headers)
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
    

class RealtimeAgent(AbstractAgent):
    """
    Implements 
    """

    def __init__(self, 
                 agent_name, 
                 results_path, 
                 agent_network_config, 
                 manager_network_config, 
                 initial_state, 
                 specs, 
                 planning_module : InternalModule = None,
                 science_module : InternalModule = None,
                 level=logging.INFO, 
                 logger=None):
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

        super().__init__(agent_name, results_path, agent_network_config, manager_network_config, initial_state, specs, modules, level, logger)

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


class SimulatedAgent(AbstractAgent):
    def __init__(self, 
                 agent_name, 
                 results_path, 
                 agent_network_config, 
                 manager_network_config, 
                 initial_state, 
                 specs, 
                 processor : DataProcessor = None, 
                 preplanner : AbstractPreplanner = None,
                 replanner : AbstractReplanner = None,
                 reward_grid : RewardGrid = None,
                 level=logging.INFO, 
                 logger=None):
        
        super().__init__(agent_name, 
                         results_path, 
                         agent_network_config, 
                         manager_network_config, 
                         initial_state, 
                         specs, 
                         [], 
                         level, 
                         logger)
        
        # set parameters
        self.processor : DataProcessor = processor
        self.preplanner : AbstractPreplanner = preplanner
        self.replanner : AbstractReplanner = replanner
        self.reward_grid : RewardGrid = reward_grid

        # initialize parameters
        self.plan : Plan = Preplan(t=-1.0)
        self.orbitdata = None
        self.plan_history = []

    @runtime_tracker
    async def think(self, senses : list):

        # unpack and sort senses
        relay_messages, incoming_reqs, observations, \
            states, action_statuses, misc_messages = self._read_incoming_messages(senses)
        incoming_reqs : list[MeasurementRequest]
        states : list[SimulationAgentState]

        # check action completion
        completed_actions, aborted_actions, pending_actions = self._check_action_completion(action_statuses)

        # extract latest state from senses
        states.sort(key = lambda a : a.state['t'])
        state : SimulationAgentState = SimulationAgentState.from_dict(states[-1].state)                                                          

        # update plan completion
        self.plan.update_action_completion(completed_actions, 
                                           aborted_actions, 
                                           pending_actions, 
                                           state.t
                                           )    

        # process performed observations
        generated_reqs : list[MeasurementRequest] = self.processor.process(incoming_reqs, observations)
        incoming_reqs.extend(generated_reqs)

        # compile measurements performed by myself or other agents
        completed_observations = {action for action in completed_actions
                        if isinstance(action, ObservationAction)}
        completed_observations.update({action_from_dict(**msg.observation_action) 
                            for msg in misc_messages
                            if isinstance(msg, ObservationPerformedMessage)})
        
        # update reward grid
        if self.reward_grid: self.reward_grid.update(self.get_current_time(), completed_observations, incoming_reqs)

        # --- Create plan ---
        if self.preplanner is not None:
            # there is a preplanner assigned to this planner
            
            # update preplanner precepts
            self.preplanner.update_percepts(state,
                                            self.plan, 
                                            incoming_reqs,
                                            relay_messages,
                                            misc_messages,
                                            completed_actions,
                                            aborted_actions,
                                            pending_actions
                                        )
            
            # check if there is a need to construct a new plan
            if self.preplanner.needs_planning(state, 
                                              self.specs, 
                                              self.plan):  
                # initialize plan      
                self.plan : Plan = self.preplanner.generate_plan(state, 
                                                            self.specs,
                                                            self.reward_grid,
                                                            self._clock_config,
                                                            self.orbitdata
                                                            )

                # save copy of plan for post-processing
                plan_copy = [action for action in self.plan]
                self.plan_history.append((state.t, plan_copy))
                
                # --- FOR DEBUGGING PURPOSES ONLY: ---
                # self.__log_plan(self.plan, "PRE-PLAN", logging.WARNING)
                # x = 1 # breakpoint
                # -------------------------------------

        t_3 = time.perf_counter()

        # --- Modify plan ---
        # Check if reeplanning is needed
        if self.replanner is not None:
            # there is a replanner assigned to this planner

            # update replanner precepts
            self.replanner.update_percepts( state,
                                            self.plan, 
                                            incoming_reqs,
                                            relay_messages,
                                            misc_messages,
                                            completed_actions,
                                            aborted_actions,
                                            pending_actions
                                        )
            
            if self.replanner.needs_planning(state, 
                                             self.specs,
                                             self.plan,
                                             self.orbitdata):    
                # --- FOR DEBUGGING PURPOSES ONLY: ---
                # self.__log_plan(plan, "ORIGINAL PLAN", logging.WARNING)
                # x = 1 # breakpoint
                # -------------------------------------

                # Modify current Plan      
                self.plan : Plan = self.replanner.generate_plan(state, 
                                                                self.specs,    
                                                                self.reward_grid,
                                                                self.plan,
                                                                self._clock_config,
                                                                self.orbitdata
                                                                )

                # update last time plan was updated
                self.t_plan = self.get_current_time()

                # save copy of plan for post-processing
                plan_copy = [action for action in self.plan]
                self.plan_history.append((self.t_plan, plan_copy))
            
                # clear pending actions
                pending_actions = []

                # --- FOR DEBUGGING PURPOSES ONLY: ---
                # self.__log_plan(self.plan, "REPLAN", logging.WARNING)
                x = 1 # breakpoint
                # -------------------------------------

        plan_out : list = self.plan.get_next_actions(state.t)
        # --- FOR DEBUGGING PURPOSES ONLY: ---
        # self.__log_plan(plan_out, "PLAN OUT", logging.WARNING)
        # x = 1 # breakpoint
        # -------------------------------------
        
        return plan_out

    def _read_incoming_messages(self, senses : list) -> tuple:
        ## extract relay messages
        relay_messages = [msg for msg in senses if msg.path]
        for relay_msg in relay_messages: senses.remove(relay_msg)

        ## remove bus messages and add their contents to senses 
        bus_messages : list[BusMessage] = [msg for msg in senses if isinstance(msg, BusMessage)]
        for bus_msg in bus_messages: 
            senses.extend([message_from_dict(**msg) for msg in bus_msg.msgs])
            senses.remove(bus_msg)

        ## classify senses
        incoming_reqs : list[MeasurementRequest] = [MeasurementRequest.from_dict(msg.req) 
                                                    for msg in senses 
                                                    if isinstance(msg, MeasurementRequestMessage)
                                                    and msg.req['severity'] > 0.0]
        observation_msgs : list [ObservationResultsMessage] = [sense for sense in senses 
                                                                if isinstance(sense, ObservationResultsMessage)]
        observations : list[tuple] = [(Instrument.from_dict(msg.instrument), msg.observation_data) 
                                        for msg in senses 
                                        if isinstance(msg, ObservationResultsMessage)
                                        and msg.dst == self.get_element_name()]
        states : list[AgentStateMessage] = [sense for sense in senses 
                                            if isinstance(sense, AgentStateMessage)
                                            and sense.src == self.get_element_name()]
        action_statuses : list[AgentActionMessage] = [sense for sense in senses
                                                      if isinstance(sense, AgentActionMessage)]
                
        misc_messages = set(senses)
        misc_messages.difference_update(incoming_reqs)
        misc_messages.difference_update(observation_msgs)
        misc_messages.difference_update(states)
        misc_messages.difference_update(action_statuses)
        misc_messages = list(misc_messages)

        return relay_messages, incoming_reqs, observations, states, action_statuses, misc_messages

    def _check_action_completion(self, action_statuses : list) -> tuple:
        
        # collect all action statuses from messages
        actions = [action_from_dict(**action_msg.action) for action_msg in action_statuses]

        # classify by action completion
        completed_actions = [action for action in actions
                            if isinstance(action, AgentAction)
                            and action.status == AgentAction.COMPLETED] # planned action completed
                
        aborted_actions = [action for action in actions 
                           if isinstance(action, AgentAction)
                           and action.status == AgentAction.ABORTED]    # planned action aborted
                
        pending_actions = [action for action in actions
                           if isinstance(action, AgentAction)
                           and action.status == AgentAction.PENDING] # planned action wasn't completed

        # return classified lists
        return completed_actions, aborted_actions, pending_actions

    def __log_plan(self, plan : Plan, title : str, level : int = logging.DEBUG) -> None:
        try:
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
        except Exception as e:
            print(e)
            raise e

    async def teardown(self):
        try:
            await super().teardown()

            # log known and generated requests
            if self.processor is not None:
                columns = ['ID','Requester','lat [deg]','lon [deg]','Severity','t start','t end','t corr','Measurment Types']
                data = [(req.id, req.requester, req.target[0], req.target[1], req.severity, req.t_start, req.t_end, req.t_corr, str(req.observation_types))
                        for req in self.processor.known_reqs
                        if req.requester == self.get_element_name()]
                
                df = pd.DataFrame(data=data, columns=columns)        
                df.to_csv(f"{self.results_path}/events_known.csv", index=False)   

                columns = ['ID','Requester','lat [deg]','lon [deg]','Severity','t start','t end','t corr','Measurment Types']
                data = [(req.id, req.requester, req.target[0], req.target[1], req.severity, req.t_start, req.t_end, req.t_corr, str(req.observation_types))
                        for req in self.processor.generated_reqs
                        if req.requester == self.get_element_name()]
                
                df = pd.DataFrame(data=data, columns=columns)        
                df.to_csv(f"{self.results_path}/events_detected.csv", index=False)   
        
            # log plan history
            headers = ['plan_index', 't_plan', 'desc', 't_start', 't_end']
            data = []
            
            for i in range(len(self.plan_history)):
                t_plan, plan = self.plan_history[i]
                t_plan : float; plan : list[AgentAction]

                for action in plan:
                    desc = f'{action.action_type}'
                    if isinstance(action, ObservationAction):
                        desc += f'_{action.instrument_name}'
                        
                    line_data = [   i,
                                    np.round(t_plan,3),
                                    desc,
                                    np.round(action.t_start,3 ),
                                    np.round(action.t_end,3 )
                                ]
                    data.append(line_data)

            df = pd.DataFrame(data, columns=headers)
            # self.log(f'\nPLANNER HISTORY\n{str(df)}\n', level=logging.WARNING)
            df.to_csv(f"{self.results_path}/planner_history.csv", index=False)

            # log reward grid history
            if self.reward_grid is not None:
                headers = ['t_update','grid_index','GP index','lat [deg]', 'log [deg]','instrument','reward','n_observations','n_events']
                data = self.reward_grid.get_history()
                df = pd.DataFrame(data, columns=headers)
                # self.log(f'\nREWARD GRID HISTORY\n{str(df)}\n', level=logging.DEBUG)
                df.to_csv(f"{self.results_path}/reward_grid_history.csv", index=False)

            # log performance stats
            n_decimals = 5
            headers = ['routine','t_avg','t_std','t_med', 't_max', 't_min', 'n', 't_total']
            data = []

            for routine in self.stats:
                n = len(self.stats[routine])
                t_avg = np.round(np.mean(self.stats[routine]),n_decimals) if n > 0 else -1
                t_std = np.round(np.std(self.stats[routine]),n_decimals) if n > 0 else 0.0
                t_median = np.round(np.median(self.stats[routine]),n_decimals) if n > 0 else -1
                t_max = np.round(max(self.stats[routine]),n_decimals) if n > 0 else -1
                t_min = np.round(min(self.stats[routine]),n_decimals) if n > 0 else -1
                t_total = t_avg * n

                line_data = [ 
                                routine,
                                t_avg,
                                t_std,
                                t_median,
                                t_max,
                                t_min,
                                n,
                                t_total
                                ]
                data.append(line_data)

                # save time-series
                time_series = [[v] for v in self.stats[routine]]
                routine_df = pd.DataFrame(data=time_series, columns=['dt'])
                routine_dir = os.path.join(f"{self.results_path}/runtime", f"time_series-planner_{routine}.csv")
                routine_df.to_csv(routine_dir,index=False)

            if isinstance(self.preplanner, AbstractPreplanner):
                for routine in self.preplanner.stats:
                    n = len(self.preplanner.stats[routine])
                    t_avg = np.round(np.mean(self.preplanner.stats[routine]),n_decimals) if n > 0 else -1
                    t_std = np.round(np.std(self.preplanner.stats[routine]),n_decimals) if n > 0 else 0.0
                    t_median = np.round(np.median(self.preplanner.stats[routine]),n_decimals) if n > 0 else -1
                    t_max = np.round(max(self.preplanner.stats[routine]),n_decimals) if n > 0 else -1
                    t_min = np.round(min(self.preplanner.stats[routine]),n_decimals) if n > 0 else -1
                    t_total = t_avg * n

                    line_data = [ 
                                    f"preplanner/{routine}",
                                    t_avg,
                                    t_std,
                                    t_median,
                                    t_max,
                                    t_min,
                                    n,
                                    t_total
                                    ]
                    data.append(line_data)

                    # save time-series
                    time_series = [[v] for v in self.preplanner.stats[routine]]
                    routine_df = pd.DataFrame(data=time_series, columns=['dt'])
                    routine_dir = os.path.join(f"{self.results_path}/runtime", f"time_series-preplanner_{routine}.csv")
                    routine_df.to_csv(routine_dir,index=False)

            if isinstance(self.replanner, AbstractReplanner):
                for routine in self.replanner.stats:
                    n = len(self.replanner.stats[routine])
                    t_avg = np.round(np.mean(self.replanner.stats[routine]),n_decimals) if n > 0 else -1
                    t_std = np.round(np.std(self.replanner.stats[routine]),n_decimals) if n > 0 else 0.0
                    t_median = np.round(np.median(self.replanner.stats[routine]),n_decimals) if n > 0 else -1
                    t_max = np.round(max(self.replanner.stats[routine]),n_decimals) if n > 0 else -1
                    t_min = np.round(min(self.replanner.stats[routine]),n_decimals) if n > 0 else -1
                    t_total = t_avg * n

                    line_data = [ 
                                    f"replanner/{routine}",
                                    t_avg,
                                    t_std,
                                    t_median,
                                    t_max,
                                    t_min,
                                    n,
                                    t_total
                                    ]
                    data.append(line_data)

                    # save time-series
                    time_series = [[v] for v in self.replanner.stats[routine]]
                    routine_df = pd.DataFrame(data=time_series, columns=['dt'])
                    routine_dir = os.path.join(f"{self.results_path}/runtime", f"time_series-replanner_{routine}.csv")
                    routine_df.to_csv(routine_dir,index=False)

            if self.reward_grid:
                for routine in self.reward_grid.stats:
                    n = len(self.reward_grid.stats[routine])
                    t_avg = np.round(np.mean(self.reward_grid.stats[routine]),n_decimals) if n > 0 else -1
                    t_std = np.round(np.std(self.reward_grid.stats[routine]),n_decimals) if n > 0 else 0.0
                    t_median = np.round(np.median(self.reward_grid.stats[routine]),n_decimals) if n > 0 else -1
                    t_max = np.round(max(self.reward_grid.stats[routine]),n_decimals) if n > 0 else -1
                    t_min = np.round(min(self.reward_grid.stats[routine]),n_decimals) if n > 0 else -1
                    t_total = t_avg * n

                    line_data = [ 
                                    f"reward_grid/{routine}",
                                    t_avg,
                                    t_std,
                                    t_median,
                                    t_max,
                                    t_min,
                                    n,
                                    t_total
                                    ]
                    data.append(line_data)

            stats_df = pd.DataFrame(data, columns=headers)
            # self.log(f'\nPLANNER RUN-TIME STATS\n{str(stats_df)}\n', level=logging.WARNING)
            # self.log(f'total: {sum(stats_df["t_total"])}', level=logging.WARNING)
            stats_df.to_csv(f"{self.results_path}/planner_runtime_stats.csv", index=False)


        except Exception as e:
            x = 1