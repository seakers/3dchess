import copy
import csv
import time
from typing import Any, Callable
import numpy as np
import pandas as pd
from zmq import asyncio as azmq

from pandas import DataFrame
from nodes.science.utility import synergy_factor
from nodes.science.reqs import *
from nodes.orbitdata import OrbitData
from nodes.states import GroundStationAgentState, UAVAgentState, SatelliteAgentState
from nodes.actions import MeasurementAction
from nodes.agent import SimulationAgentState
from messages import *
from utils import setup_results_directory

from dmas.environments import *
from dmas.messages import *


class SimulationEnvironment(EnvironmentNode):
    """
    ## Simulation Environment

    Environment in charge of creating task requests and notifying agents of their exiance
    Tracks the current state of the agents and checks if they are in communication range 
    of eachother.
    
    """
    SPACECRAFT = 'SPACECRAFT'
    UAV = 'UAV'
    GROUND_STATION = 'GROUND_STATION'

    def __init__(self, 
                scenario_path : dict,
                results_path : str, 
                env_network_config: NetworkConfig, 
                manager_network_config: NetworkConfig, 
                utility_func : Callable[[], Any], 
                measurement_reqs : list = [],
                events_path : str = None,
                level: int = logging.INFO, 
                logger: logging.Logger = None) -> None:
        super().__init__(env_network_config, manager_network_config, [], level, logger)

        # setup results folder:
        self.results_path = setup_results_directory(results_path+'/'+ self.get_element_name().swapcase())

        # load observation data
        self.orbitdata = OrbitData.from_directory(scenario_path)

        # load agent names and types
        self.agents = {}
        agent_names = []
        with open(scenario_path + 'MissionSpecs.json', 'r') as scenario_specs:
            scenario_dict : dict = json.load(scenario_specs)
            
            # load satellite names
            sat_names = []
            sat_list : dict = scenario_dict.get('spacecraft', None)
            if sat_list:
                for sat in sat_list:
                    sat : dict
                    sat_name = sat.get('name')
                    sat_names.append(sat_name)
                    agent_names.append(sat_name)
            self.agents[self.SPACECRAFT] = sat_names

            # load uav names
            uav_names = []
            uav_list : dict = scenario_dict.get('uav', None)
            if uav_list:
                for uav in uav_list:
                    uav : dict
                    uav_name = uav.get('name')
                    uav_names.append(uav_name)
                    agent_names.append(uav_name)
            self.agents[self.UAV] = uav_names

            # load GS agent names
            gs_names = []
            gs_list : dict = scenario_dict.get('groundStation', None)
            if gs_list:
                for gs in gs_list:
                    gs : dict
                    gs_name = gs.get('name')
                    gs_names.append(gs_name)
                    agent_names.append(gs_name)
            self.agents[self.GROUND_STATION] = gs_names

        # initialize parameters
        self.utility_func = utility_func
        self.measurement_history = []
        self.agent_connectivity = {}
        for src in agent_names:
            for target in agent_names:
                if src not in self.agent_connectivity:
                    self.agent_connectivity[src] = {}    
                
                self.agent_connectivity[src][target] = -1

        self.measurement_reqs = []
        self.initial_reqs = []
        for req in measurement_reqs:
            req : MeasurementRequest
            self.initial_reqs.append(req.copy())
            # self.measurement_reqs.append(req.copy())

        self.stats = {}
        self.events_path = events_path

    async def setup(self) -> None:
        # nothing to set up
        return

    async def live(self) -> None:
        try:
            # create port poller 
            poller = azmq.Poller()

            manager_socket, _ = self._manager_socket_map.get(zmq.SUB)
            agent_socket, _ = self._external_socket_map.get(zmq.REP)
            agent_broadcasts, _ = self._external_socket_map.get(zmq.SUB)

            poller.register(manager_socket, zmq.POLLIN)
            poller.register(agent_socket, zmq.POLLIN)
            poller.register(agent_broadcasts, zmq.POLLIN)
            
            # track agent and simulation states
            while True:
                socks = dict(await poller.poll())

                if agent_socket in socks:
                    # read message from agents
                    dst, src, content = await self.listen_peer_message()

                    t_0 = time.perf_counter()
                    
                    if content['msg_type'] == SimulationMessageTypes.MEASUREMENT.value:
                        # unpack message
                        msg = MeasurementResultsRequestMessage(**content)
                        self.log(f'received masurement data request from {msg.src}. quering measurement results...')

                        # find/generate measurement results
                        measurement_action = MeasurementAction(**msg.measurement_action)
                        agent_state = SimulationAgentState.from_dict(msg.agent_state)
                        measurement_req = MeasurementRequest.from_dict(measurement_action.measurement_req)
                        measurement_data = self.query_measurement_data(src, agent_state, measurement_req, measurement_action)

                        # repsond to request
                        self.log(f'measurement results obtained! responding to request')
                        resp = copy.deepcopy(msg)
                        resp.dst = resp.src
                        resp.src = self.get_element_name()
                        resp.measurement = measurement_data
                        self.measurement_history.append(resp)

                        await self.respond_peer_message(resp) 

                    elif content['msg_type'] == SimulationMessageTypes.AGENT_STATE.value:
                        # unpack message
                        msg = AgentStateMessage(**content)
                        self.log(f'state message received from {msg.src}. updating state tracker...')

                        # initiate response
                        resp_msgs = []

                        # check current state
                        updated_state = None
                        if src in self.agents[self.SPACECRAFT]:
                            # look up orbitdata
                            current_state = SatelliteAgentState(**msg.state)
                            data : OrbitData = self.orbitdata[src]
                            pos, vel, eclipse = data.get_orbit_state(current_state.t)

                            # update state
                            updated_state = current_state
                            updated_state.pos = pos
                            updated_state.vel = vel
                            updated_state.eclipse = int(eclipse)

                        elif src in self.agents[self.UAV]:
                            # Do NOT update
                            updated_state = UAVAgentState(**msg.state)

                        elif src in self.agents[self.GROUND_STATION]:
                            # Do NOT update state
                            updated_state = GroundStationAgentState(**msg.state)

                        updated_state.t = max(self.get_current_time(), current_state.t)
                        
                        updated_state_msg = AgentStateMessage(self.get_element_name(), src, updated_state.to_dict())
                        resp_msgs.append(updated_state_msg.to_dict())

                        # check connectivity status
                        for target_type in self.agents:
                            for target in self.agents[target_type]:
                                if target == src:
                                    continue
                                
                                connected = self.check_agent_connectivity(src, target, target_type)
                                
                                if connected == 0 and self.agent_connectivity[src][target] == -1:
                                    self.agent_connectivity[src][target] = connected
                                    continue

                                if self.agent_connectivity[src][target] != connected:
                                    # only announce if changes to connectivity have been made
                                    connectivity_update = AgentConnectivityUpdate(src, target, connected)
                                    resp_msgs.append(connectivity_update.to_dict())
                                    self.agent_connectivity[src][target] = connected

                        # package response
                        resp_msg = BusMessage(self.get_element_name(), src, resp_msgs)

                        # send response
                        await self.respond_peer_message(resp_msg)

                    else:
                        # message is of an unsopported type. send blank response
                        self.log(f"received message of type {content['msg_type']}. ignoring message...")
                        resp = NodeReceptionIgnoredMessage(self.get_element_name(), src)

                        # respond to request
                        await self.respond_peer_message(resp)
                    
                    dt = time.perf_counter() - t_0
                    if src not in self.stats:
                        self.stats[src] = []
                    self.stats[src].append(dt)

                elif agent_broadcasts in socks:
                    dst, src, content = await self.listen_peer_broadcast()

                    if content['msg_type'] == SimulationMessageTypes.MEASUREMENT_REQ.value:
                        req_msg = MeasurementRequestMessage(**content)
                        measurement_req = MeasurementRequest.from_dict(req_msg.req)

                        if measurement_req.s_max <= 0.0:
                            continue

                        initial_req_ids = [req.id for req in self.initial_reqs]
                        gen_req_ids = [req.id for req in self.measurement_reqs]
                        if (measurement_req.id not in gen_req_ids and measurement_req.id not in initial_req_ids):
                            self.measurement_reqs.append(measurement_req)

                elif manager_socket in socks:
                    # check if manager message is received:
                    dst, src, content = await self.listen_manager_broadcast()

                    if (dst in self.name 
                        and SimulationElementRoles.MANAGER.value in src 
                        and content['msg_type'] == ManagerMessageTypes.SIM_END.value
                        ):
                        # sim end message received
                        self.log(f"received message of type {content['msg_type']}. ending simulation...")
                        return

                    elif content['msg_type'] == ManagerMessageTypes.TOC.value:
                        # toc message received

                        # unpack message
                        msg = TocMessage(**content)

                        # update internal clock
                        self.log(f"received message of type {content['msg_type']}. updating internal clock to {msg.t}[s]...")
                        await self.update_current_time(msg.t)

                        # wait for all agent's to send their updated states
                        self.log(f"internal clock uptated to time {self.get_current_time()}[s]!")
                    
                    else:
                        # ignore message
                        self.log(f"received message of type {content['msg_type']}. ignoring message...")

        except asyncio.CancelledError:
            self.log(f'`live()` interrupted. {e}', level=logging.DEBUG)
            return

        except Exception as e:
            self.log(f'`live()` failed. {e}', level=logging.ERROR)
            raise e

    def check_agent_connectivity(self, src : str, target : str, target_type : str) -> bool:
        """
        Checks if an agent is in communication range with another agent

        #### Arguments:
            - src (`str`): name of agent starting the connection
            - target (`str`): name of agent receving the connection
            - target_type (`str`): type of agent receving the connection

        #### Returns:
            - connected (`int`): binary value representing if the `src` and `target` are connected
        """
        connected = False
        if target_type == self.SPACECRAFT:
            if src in self.agents[self.SPACECRAFT]:
                # check orbit data
                src_data : OrbitData = self.orbitdata[src]
                connected = src_data.is_accessing_agent(target, self.get_current_time())
                
            elif src in self.agents[self.UAV]:
                # check orbit data with nearest GS
                target_data : OrbitData = self.orbitdata[target]
                connected = target_data.is_accessing_ground_station(target, self.get_current_time())
            
            elif src in self.agents[self.GROUND_STATION]:
                # check orbit data
                target_data : OrbitData = self.orbitdata[target]
                connected = target_data.is_accessing_ground_station(target, self.get_current_time())
        
        elif target_type == self.UAV:
            if src in self.agents[self.SPACECRAFT]:
                # check orbit data with nearest GS
                src_data : OrbitData = self.orbitdata[src]
                connected = src_data.is_accessing_ground_station(target, self.get_current_time())

            elif src in self.agents[self.UAV]:
                # always connected
                connected = True

            elif src in self.agents[self.GROUND_STATION]:
                # always connected
                connected = True
        
        elif target_type == self.GROUND_STATION:
            if src in self.agents[self.SPACECRAFT]:
                # check orbit data
                src_data : OrbitData = self.orbitdata[src]
                connected = src_data.is_accessing_ground_station(target, self.get_current_time())

            elif src in self.agents[self.UAV]:
                # always connected
                connected = True

            elif src in self.agents[self.GROUND_STATION]:
                # always connected
                connected = True

        return int(connected)

    @runtime_tracker
    def query_measurement_data( self,
                                agent_name : str, 
                                agent_state : SimulationAgentState, 
                                measurement_req : MeasurementRequest, 
                                measurement_action : MeasurementAction
                                ) -> dict:
        """
        Queries internal models or data and returns observation information being sensed by the agent
        """
        # if measurement_req.id not in self.measurement_reqs:
        #     self.measurement_reqs[measurement_req.id] = measurement_req

        if isinstance(measurement_req, GroundPointMeasurementRequest):
            if isinstance(agent_state, SatelliteAgentState):
                orbitdata : OrbitData = self.orbitdata[agent_name]
                lat, lon, _ = measurement_req.lat_lon_pos
                
                # query ground point access data
                observation_data : dict = orbitdata.get_groundpoint_access_data(lat, 
                                                                                lon, 
                                                                                measurement_action.instrument_name, 
                                                                                agent_state.t)
                
                # include measurement request info
                observation_data['t'] = agent_state.t
                observation_data['state'] = agent_state.to_dict()
                observation_data['req'] = measurement_req.to_dict()
                observation_data['subtask_index'] = measurement_action.subtask_index
                observation_data['t_img'] = agent_state.t
                
                # calculate measurement utility
                observation_data['u_max'] = measurement_req.s_max
                observation_data['u_exp'] = measurement_action.u_exp

                ## check time constraints
                if (agent_state.t < measurement_req.t_start
                    or measurement_req.t_end < agent_state.t
                    ):
                    observation_data['u'] = 0.0
                ## check observation metrics
                elif (  observation_data['observation range [km]'] is None
                        or observation_data['look angle [deg]'] is None
                        or observation_data['incidence angle [deg]'] is None
                    ):
                    observation_data['u'] = 0.0
                # TODO check attitude pointing
                # elif

                ## calculate utility
                else:
                    observation_data['u'] = self.utility_func(**observation_data) * synergy_factor(**observation_data)

                return observation_data

            else:
                raise NotImplementedError(f"Measurement results query not yet supported for agents with state of type {type(agent_state)}")

        else:
            raise NotImplementedError(f"Measurement results query not yet supported for measurment requests of type {type(measurement_req)}.")

    async def teardown(self) -> None:
        try:
            # print final time
            self.log(f'Environment shutdown with internal clock of {self.get_current_time()}[s]', level=logging.WARNING)
            
            # print received measurements
            headers = ['req_id','measurer','measurement','pos','t_start','t_end','t_corr','t_img','u_max','u_exp','u']
            data = []
            for msg in self.measurement_history:
                msg : MeasurementResultsRequestMessage
                measurement_action = MeasurementAction(**msg.measurement_action)
                req : MeasurementRequest = MeasurementRequest.from_dict(measurement_action.measurement_req)
                measurement_data : dict = msg.measurement
                measurer = msg.dst
                t_img = msg.measurement['t_img']           

                line_data = [req.id.split('-')[0],
                                measurer,
                                measurement_action.instrument_name,
                                msg.measurement_action["measurement_req"]["pos"],
                                req.t_start,
                                req.t_end,
                                req.t_corr,
                                t_img,
                                measurement_data['u_max'],
                                measurement_data['u_exp'],
                                measurement_data['u']]
                data.append(line_data)

            received_measurements_df = DataFrame(data, columns=headers)

            # count total number of events in the simulation
            if self.events_path is not None:
                df = pd.read_csv(self.events_path)
                n_events, _ = df.shape
                

                events_detected = []
                for _, row in df.iterrows():
                    for req in self.measurement_reqs:
                        req : MeasurementRequest
                        lat,lon,_ = req.lat_lon_pos

                        if (    
                                abs(lat - row['lat [deg]']) <= 1e-2 and
                                abs(lon - row['lon [deg]']) <= 1e-2 and
                                row['start time [s]'] <= req.t_start and
                                abs(req.t_end - (row['start time [s]'] + row['duration [s]'])) <= 1e-2 and
                                abs(req.s_max == row['severity']) <= 1e-2
                        ):
                            measurements : str = row['measurements']
                            measurements : str = measurements.replace('[','')
                            measurements : str = measurements.replace(']','')
                            measurements : str = measurements.replace(' ','')
                            measurements = measurements.split(',')

                            if len(req.measurements) != len(measurements):
                                continue

                            measurement_not_found = False
                            for measurement in req.measurements:
                                if measurement not in measurements:
                                    measurement_not_found = True
                                    break
                                
                            if measurement_not_found:
                                continue

                            # n_events_detected += 1 
                            events_detected.append((req, row))
                            break

                n_events_detected = len(events_detected)

                n_events_obs = 0
                for event_req, _ in events_detected:
                    for msg in self.measurement_history:
                        msg : MeasurementResultsRequestMessage
                        measurement_action = MeasurementAction(**msg.measurement_action)
                        req : MeasurementRequest = MeasurementRequest.from_dict(measurement_action.measurement_req)
                        
                        if req == event_req:
                            n_events_obs += 1
                            break

            else:
                n_events = 0
                n_events_detected = 0
                n_events_obs = 0

            # calculate utility achieved by measurements
            utility_total = 0.0
            max_utility = 0.0
            n_obervations_max = 0
            co_observations = []
            unique_observations = []

            measurement_reqs = [req.copy() for req in self.measurement_reqs]
            measurement_reqs.extend([req.copy() for req in self.initial_reqs])

        
            for req in measurement_reqs:
                req_id : str = req.id
                req_id_short = req_id.split('-')[0]
                req_measurements = received_measurements_df \
                                    .query('@req_id_short == `req_id`')

                
                req_utility = 0
                for idx, row_i in req_measurements.iterrows():
                    t_img_i = row_i['t_img']
                    measurement_i = row_i['measurement']                   

                    correlated_measurements = []
                    for _, row_j in req_measurements.iterrows():
                        measurement_j = row_j['measurement']
                        t_img_j = row_j['t_img']

                        if measurement_i == measurement_j:
                            continue

                        if abs(t_img_i - t_img_j) <= req.t_corr:
                            correlated_measurements.append( measurement_j )

                    subtask_index = None
                    while subtask_index == None:
                        for main_measurement, dependent_measurements in req.measurement_groups:
                            if (main_measurement == measurement_i 
                                and len(np.setdiff1d(correlated_measurements, dependent_measurements)) == 0):
                                subtask_index = req.measurement_groups.index((main_measurement, dependent_measurements))
                                break
                        
                        if subtask_index == None:
                            correlated_measurements == []

                    if len(correlated_measurements) > 0:
                        co_observation : list = copy.copy(dependent_measurements)
                        co_observation.append(main_measurement)
                        co_observation.append(req_id)
                        co_observation = set(co_observation) 

                        if co_observation not in co_observations:
                            co_observations.append(co_observation)
                                    
                    params = {
                                "req" : req.to_dict(), 
                                "subtask_index" : subtask_index,
                                "t_img" : t_img_i
                            }
                    utility = self.utility_func(**params) * synergy_factor(**params)

                    if (req_id_short, measurement_i) not in unique_observations:
                        unique_observations.append( (req_id_short, measurement_i) )
                        req_utility += utility

                    received_measurements_df.loc[idx,'u']=utility

                utility_total += req_utility
                max_utility += req.s_max
                n_obervations_max += len(req.measurements)

            # calculate possible number of measurements given coverage metrics
            n_obervations_pos = 0
            for req in measurement_reqs:
                if not isinstance(req, GroundPointMeasurementRequest):
                    self.log(f"WARNING cannot process results for requests of type {type(req)}", logging.WARNING)
                    continue

                req : GroundPointMeasurementRequest
                lat,lon,_ = req.lat_lon_pos

                observable_measurements = []
                for _, coverage_data in self.orbitdata.items():
                    coverage_data : OrbitData
                    req_start = req.t_start/coverage_data.time_step
                    req_end = req.t_end/coverage_data.time_step
                    grid_index, gp_index, gp_lat, gp_lon = coverage_data.find_gp_index(lat,lon)

                    df = coverage_data.gp_access_data.query('`time index` >= @req_start & `time index` <= @req_end & `GP index` == @gp_index & `grid index` == @grid_index')

                    # if not df.empty:
                    #     print(df['time index'] * coverage_data.time_step)

                    for _, row in df.iterrows():
                        instrument : str = row['instrument']
                        if (instrument in req.measurements 
                            and instrument not in observable_measurements):
                            observable_measurements.append(instrument)

                        if len(observable_measurements) == len(req.measurements):
                            break

                    if len(observable_measurements) == len(req.measurements):
                        break

                n_obervations_pos += len(observable_measurements)

            # Generate summary
            summary_headers = ['stat_name', 'val']
            summary_data = [
                        ['t_start', self._clock_config.start_date], 
                        ['t_end', self._clock_config.end_date], 
                        ['n_events', n_events],
                        ['n_events_detected', n_events_detected],
                        ['n_events_obs', n_events_obs],
                        ['n_reqs_total', len(self.measurement_reqs) + len(self.initial_reqs)],
                        ['n_reqs_init', len(self.initial_reqs)],
                        ['n_reqs_gen', len(self.measurement_reqs)],
                        ['n_obs_unique_max', n_obervations_max],
                        ['n_obs_unique_pos', n_obervations_pos],
                        ['n_obs_unique', len(unique_observations)],
                        ['n_obs_co', len(co_observations)],
                        ['n_obs', len(self.measurement_history)],
                        ['u_max', max_utility], 
                        ['u_total', utility_total],
                        ['u_norm', utility_total/max_utility]
                    ]

            # log and save results
            self.log(f"MEASUREMENTS RECEIVED:\n{str(received_measurements_df)}\n\n", level=logging.WARNING)
            received_measurements_df.to_csv(f"{self.results_path}/measurements.csv", index=False)

            summary_df = DataFrame(summary_data, columns=summary_headers)
            self.log(f"\nSIMULATION RESULTS SUMMARY:\n{str(summary_df)}\n\n", level=logging.WARNING)
            summary_df.to_csv(f"{self.results_path}/../summary.csv", index=False)

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

            stats_df = pd.DataFrame(data, columns=headers)
            self.log(f'\nENVIRONMENT RUN-TIME STATS\n{str(stats_df)}\n', level=logging.WARNING)
            stats_df.to_csv(f"{self.results_path}/runtime_stats.csv", index=False)
        
        except asyncio.CancelledError as e:
            raise e
        except Exception as e:
            print(e.with_traceback())
            raise e

    async def sim_wait(self, delay: float) -> None:
        try:
            if isinstance(self._clock_config, FixedTimesStepClockConfig):
                tf = self.get_current_time() + delay
                while tf > self.get_current_time():
                    # listen for manager's toc messages
                    _, _, msg_dict = await self.listen_manager_broadcast()

                    if msg_dict is None:
                        raise asyncio.CancelledError()

                    msg_dict : dict
                    msg_type = msg_dict.get('msg_type', None)

                    # check if message is of the desired type
                    if msg_type != ManagerMessageTypes.TOC.value:
                        continue
                    
                    # update time
                    msg = TocMessage(**msg_type)
                    self.update_current_time(msg.t)

            elif isinstance(self._clock_config, AcceleratedRealTimeClockConfig):
                await asyncio.sleep(delay / self._clock_config.sim_clock_freq)

            else:
                raise NotImplementedError(f'`sim_wait()` for clock of type {type(self._clock_config)} not yet supported.')
                
        except asyncio.CancelledError:
            return
