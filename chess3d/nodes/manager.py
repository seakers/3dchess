import logging
import math
import os
import numpy as np
import pandas as pd

from dmas.clocks import ClockConfig
from dmas.elements import ClockConfig, NetworkConfig, logging
from dmas.managers import *

from chess3d.messages import *

class SimulationManager(AbstractManager):
    """
    ## Simulation Manager

    Describes the simulation manager of a 3D CHESS simulation

    In charge of keeping time in the simulation.
        - Listens for all agents to request a time to fast-forward to.
        - Once all agents have done so, the manager will perform a time-step increase in the simulation clock and announce the new time to every member of the simulation.
        - This will be repeated until the final time has been reached.

    """
    def __init__(   self, 
                    results_path : str,
                    simulation_element_name_list: list, 
                    clock_config: ClockConfig, 
                    network_config: NetworkConfig, 
                    level: int = logging.INFO, 
                    logger: logging.Logger = None) -> None:
        super().__init__(simulation_element_name_list, clock_config, network_config, level, logger)

        self.results_path : str = results_path
        self.stats = {f"{name}_wait" : [] for name in simulation_element_name_list}
        self.stats["clock_wait"] = []
        self.stats["sim_runtime"] = []

    def _check_element_list(self):
        env_count = 0
        for sim_element_name in self._simulation_element_name_list:
            if SimulationElementRoles.ENVIRONMENT.value in sim_element_name:
                env_count += 1
        
        if env_count > 1:
            raise AttributeError(f'`simulation_element_name_list` must only contain one {SimulationElementRoles.ENVIRONMENT.value}. contains {env_count}')
        elif env_count < 1:
            raise AttributeError(f'`simulation_element_name_list` must contain {SimulationElementRoles.ENVIRONMENT.value}.')
        
    async def setup(self) -> None:
        # nothing to set-up
        return
    
    @runtime_tracker
    async def _execute(self) -> None:
        return await super()._execute()

    async def sim_wait(self, delay: float) -> None:
        """
        Waits for the total number of seconds in the simulation.
        Time waited depends on length of simulation and clock type in use.
        """
        try:
            t_0_sim = time.perf_counter()

            desc = f'{self.name}: Simulating'
            if isinstance(self._clock_config, AcceleratedRealTimeClockConfig):
                for _ in tqdm (range (10), desc=desc):
                    await asyncio.sleep(delay/10)

            elif isinstance(self._clock_config, FixedTimesStepClockConfig):
                dt = self._clock_config.dt
                t = 0
                tf = t + delay
                
                with tqdm(total=delay, desc=desc, leave=True) as pbar:

                    while t < tf:
                        t_0 = time.perf_counter()

                        # wait for everyone to ask to fast forward            
                        self.log(f'waiting for tic requests...')
                        reqs = await self.wait_for_tic_requests()
                        self.log(f'tic requests received!')

                        # announce new time to simulation elements
                        self.log(f'sending toc for time {t}[s]...', level=logging.INFO)
                        if reqs is None: break
                        toc = TocMessage(self.get_network_name(), t)

                        await self.send_manager_broadcast(toc)

                        # announce new time to simulation monitor
                        self.log(f'sending toc for time {t}[s] to monitor...')
                        toc.dst = SimulationElementRoles.MONITOR.value
                        await self.send_monitor_message(toc) 

                        self.log(f'toc for time {t}[s] sent!')

                        # updete time and display
                        pbar.update(dt)
                        t += dt

                        dt = time.perf_counter() - t_0
                        self.stats['clock_wait'].append(dt)

                    self.log('TIMER DONE!', level=logging.INFO)
            
            elif isinstance(self._clock_config, EventDrivenClockConfig):  
                t = 0
                tf = self._clock_config.get_total_seconds()
                iter_counter = 0
                with tqdm(total=tf , desc=desc, leave=True) as pbar:
                    while t < tf:
                        
                        t_0 = time.perf_counter()

                        # wait for everyone to ask to fast forward            
                        self.log(f'waiting for tic requests...')
                        reqs = await self.wait_for_tic_requests()
                        self.log(f'tic requests received!')

                        if reqs is None: break # an agent in the simulation is offline; terminate sim

                        tic_reqs = [reqs[src].tf for src in reqs]
                        tic_reqs.append(tf)
                        t_next = min(tic_reqs)
                        
                        # announce new time to simulation elements
                        self.log(f'sending toc for time {t_next}[s]...', level=logging.INFO)
                        toc = TocMessage(self.get_network_name(), t_next)

                        await self.send_manager_broadcast(toc)

                        # # announce new time to simulation monitor
                        # self.log(f'sending toc for time {t_next}[s] to monitor...')
                        # toc.dst = SimulationElementRoles.MONITOR.value
                        # await self.send_monitor_message(toc) 

                        self.log(f'toc for time {t_next}[s] sent!')
                        
                        # updete time and display
                        pbar.update(t_next - t)
                        t = t_next
                        iter_counter += 1
                        
                        dt = time.perf_counter() - t_0
                        self.stats['clock_wait'].append(dt)

            else:
                raise NotImplemented(f'clock configuration of type {type(self._clock_config)} not yet supported.')

            
            dt = time.perf_counter() - t_0_sim
            self.stats['sim_runtime'].append(dt)

        except asyncio.CancelledError:
            return
        
    async def wait_for_tic_requests(self, timeout : float=10*60):
        """
        Awaits for all agents to send tic requests
        
        #### Returns:
            - `dict` mapping simulation elements' names to the messages they sent.
        """
        try:
            t_0 = time.perf_counter()
            received_messages : dict = {}
            read_task = None

            with tqdm(total=len(self._simulation_element_name_list) - 1, 
                      desc='Waiting for tic requests', 
                      leave=False) as pbar:

                while(
                        len(received_messages) < len(self._simulation_element_name_list) - 1
                        and len(self._simulation_element_name_list) > 1
                    ):                

                    # reset tasks
                    read_task = None

                    # wait for incoming messages with a timeout
                    timeout_task = asyncio.create_task( asyncio.sleep(timeout) )
                    read_task = asyncio.create_task( self._receive_manager_msg(zmq.SUB) )
                    done,_ = await asyncio.wait([read_task, timeout_task], return_when=asyncio.FIRST_COMPLETED)

                    if timeout_task in done:
                        # timeout task is done
                        self.log(f'wait_for_tic_requests: timeout task done')
                        # for task in pending: 
                        #     task.cancel()
                        #     await task

                        missing_reqs = [sim_element for sim_element in self._simulation_element_name_list
                                        if sim_element not in received_messages
                                        and sim_element != self.get_element_name()
                                        and sim_element != SimulationElementRoles.ENVIRONMENT.value]
                        raise asyncio.TimeoutError(f'wait for tic request timed out. Missing requests from {missing_reqs}.')

                    _, src, msg_dict = read_task.result()
                    msg_type = msg_dict['msg_type']

                    if NodeMessageTypes[msg_type] == NodeMessageTypes.DEACTIVATED:
                        return None

                    if ((NodeMessageTypes[msg_type] != NodeMessageTypes.TIC_REQ
                        and NodeMessageTypes[msg_type] != NodeMessageTypes.CANCEL_TIC_REQ)
                        or SimulationElementRoles.ENVIRONMENT.value in src):
                        # ignore all incoming messages that are not of the desired type 
                        self.log(f'Received {msg_type} message from node {src}! Ignoring message...')
                        continue

                    # unpack and message
                    self.log(f'Received {msg_type} message from node {src}!')
                    if NodeMessageTypes[msg_type] == NodeMessageTypes.TIC_REQ:
                        # unpack message
                        tic_req = TicRequest(**msg_dict)

                        # log subscriber confirmation
                        if src not in self._simulation_element_name_list and self.get_network_name() + '/' + src not in self._simulation_element_name_list:
                            # node is not a part of the simulation
                            self.log(f'{src} is not part of this simulation. Wait status: ({len(received_messages)}/{len(self._simulation_element_name_list) - 1})')

                        elif src in received_messages:
                            # node is a part of the simulation but has already communicated with me
                            self.log(f'{src} has already reported its tic request to the simulation manager. Wait status: ({len(received_messages)}/{len(self._simulation_element_name_list) - 1})')

                        else:
                            # node is a part of the simulation and has not yet been synchronized
                            received_messages[src] = tic_req
                            self.log(f'{src} has now reported reported its tic request  to the simulation manager. Wait status: ({len(received_messages)}/{len(self._simulation_element_name_list) - 1})')

                            dt = time.perf_counter() - t_0
                            self.stats[f'{src}_wait'].append(dt)

                            pbar.update(1)

                    elif NodeMessageTypes[msg_type] == NodeMessageTypes.CANCEL_TIC_REQ:

                        # log subscriber cancellation
                        if src not in self._simulation_element_name_list and self.get_network_name() + '/' + src not in self._simulation_element_name_list:
                            # node is not a part of the simulation
                            self.log(f'{src} is not part of this simulation. Wait status: ({len(received_messages)}/{len(self._simulation_element_name_list) - 1})')

                        elif src not in received_messages:
                            # node is a part of the simulation but ha not yet communicated with me
                            self.log(f'{src} has not reported its tic request to the simulation manager yet. Wait status: ({len(received_messages)}/{len(self._simulation_element_name_list) - 1})')

                        else:
                            # node is a part of the simulation and has already been synchronized
                            received_messages.pop(src)
                            self.log(f'{src} has cancelled its tic request to the simulation manager. Wait status: ({len(received_messages)}/{len(self._simulation_element_name_list) - 1})')

                            self.stats[f'{src}_wait'].pop(-1)

                            pbar.update(-1)

            return received_messages

        except asyncio.CancelledError:            
            # wait cancelled
            return

        except Exception as e:
            self.log(f'wait failed. {e}', level=logging.ERROR)
            raise e

        finally: 
            # cancel read message task in case it is still being performed
            if read_task is not None and not read_task.done(): 
                read_task.cancel()
                await read_task
    
    async def teardown(self) -> None:
        # log performance stats
        results_dir = os.path.join(self.results_path, self.get_element_name().lower())
        if not os.path.isdir(results_dir): os.mkdir(results_dir)
        
        runtime_dir = os.path.join(results_dir, "runtime")
        if not os.path.isdir(runtime_dir): os.mkdir(runtime_dir)
    
        n_decimals = 3
        headers = ['routine','t_avg','t_std','t_med', 't_max', 't_min', 'n', 't_total']
        data = []

        for routine in tqdm(self.stats, desc="MANAGER: Compiling runtime statistics", leave=False):
            n = len(self.stats[routine])
            t_avg = np.round(np.mean(self.stats[routine]),n_decimals) if n > 0 else -1
            t_std = np.round(np.std(self.stats[routine]),n_decimals) if n > 0 else 0.0
            t_median = np.round(np.median(self.stats[routine]),n_decimals) if n > 0 else -1
            t_max = np.round(max(self.stats[routine]),n_decimals) if n > 0 else -1
            t_min = np.round(min(self.stats[routine]),n_decimals) if n > 0 else -1
            t_total = n * t_avg

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
            routine_dir = os.path.join(runtime_dir, f"time_series-{routine}.csv")
            routine_df.to_csv(routine_dir,index=False)

        stats_df = pd.DataFrame(data, columns=headers)
        # self.log(f'\nMANAGER RUN-TIME STATS\n{str(stats_df)}\n', level=logging.WARNING)
        stats_df.to_csv(f"{results_dir}/runtime_stats.csv", index=False)
        
        self.log(f'sucessfully shutdown ', level=logging.WARNING)
