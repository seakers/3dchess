import copy
import os
import time
from dmas.elements import SimulationMessage
from dmas.messages import SimulationMessage
import numpy as np
import pandas as pd
from tqdm import tqdm
from zmq import asyncio as azmq
import concurrent.futures

from instrupy.base import Instrument
from instrupy.base import BasicSensorModel
from instrupy.passive_optical_scanner_model import PassiveOpticalScannerModel
from instrupy.util import SphericalGeometry, ViewGeometry

from chess3d.agents.science.requests import *
from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.states import *
from chess3d.agents.states import SimulationAgentState
from chess3d.messages import *

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
                results_path : str, 
                orbitdata_dir : str,
                sat_list : list,
                uav_list : list,
                gs_list : list,
                env_network_config: NetworkConfig, 
                manager_network_config: NetworkConfig, 
                connectivity : str = 'full',
                events_path : str = None,
                level: int = logging.INFO, 
                logger: logging.Logger = None) -> None:
        super().__init__(env_network_config, manager_network_config, [], level, logger)

        # setup results folder:
        self.results_path : str = os.path.join(results_path, self.get_element_name().lower())

        # load observation data
        self.orbitdata : dict = OrbitData.from_directory(orbitdata_dir) if orbitdata_dir is not None else None

        # load agent names and classify by type of agent
        self.agents = {}
        agent_names = []
            
        # load satellite names
        sat_names = []
        if sat_list:
            for sat in sat_list:
                sat : dict
                sat_name = sat.get('name')
                sat_names.append(sat_name)
                agent_names.append(sat_name)
        self.agents[self.SPACECRAFT] = sat_names

        # load uav names
        uav_names = []
        if uav_list:
            for uav in uav_list:
                uav : dict
                uav_name = uav.get('name')
                uav_names.append(uav_name)
                agent_names.append(uav_name)
        self.agents[self.UAV] = uav_names

        # load GS agent names
        gs_names : list = []
        if gs_list:
            for gs in gs_list:
                gs : dict
                gs_name = gs.get('name')
                gs_names.append(gs_name)
                agent_names.append(gs_name)
        self.agents[self.GROUND_STATION] = gs_names

        # load events
        self.events_path : str = events_path
        self.events : pd.DataFrame = self.load_events(events_path)

        # initialize parameters
        self.connectivity = connectivity
        self.observation_history = []
        self.agent_connectivity = {}
        for src in agent_names:
            for target in agent_names:
                if src not in self.agent_connectivity:
                    self.agent_connectivity[src] = {}    
                
                self.agent_connectivity[src][target] = -1
        self.agent_state_update_times = {}

        self.measurement_reqs : set[TaskRequest] = set()
        self.stats = {}
        self.t_0 = None
        self.t_f = None
        self.t_update = None

        self.broadcasts_history = []
        
    def load_events(self, events_path : str) -> pd.DataFrame:
        """ Loads events present in the simulation """
        # checks if event path exists
        if events_path is None: return None
        if not os.path.isfile(events_path): raise ValueError(f'List of events not found in `{events_path}`')

        # get simulation duration 
        agent_names = list(self.orbitdata.keys())
        if agent_names:
            temp_agent = agent_names.pop()
            temp_agent_orbit_data : OrbitData = self.orbitdata[temp_agent]
            sim_duration : float = temp_agent_orbit_data.duration*24*3600
        else:
            sim_duration = np.Inf

        if not os.path.isfile(events_path):
            raise ValueError('`events_path` must point to an existing file.')
        
        events_df : pd.DataFrame = pd.read_csv(events_path)

        events = []
        for _,row in events_df.iterrows():
            # convert event to GeophysicalEvent
            if row['start time [s]'] > sim_duration:
                # event is not in the simulation time frame
                continue

            event = GeophysicalEvent(
                row['event type'],
                row['severity'],
                (row['lat [deg]'], row['lon [deg]'], row.get('grid index', 0), row['gp_index']),
                row['start time [s]'],
                (row['start time [s]'] + row['duration [s]']),
                row['decorrelation time [s]'],
                row['id']
            )
            events.append(event)

        return events

    async def setup(self) -> None:
        # nothing to set up
        return

    async def live(self) -> None:
        try:
            self.t_0 = time.perf_counter()

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
                # get list of sockets with incoming messages or requests
                socks = dict(await poller.poll())

                # handle incoming messages or requests
                req_status : bool = await self.handle_request(socks, agent_socket, agent_broadcasts, manager_socket)
                
                # check if request was processed correctly
                if not req_status: return                    

        except asyncio.CancelledError:
            self.log(f'`live()` interrupted. {e}', level=logging.DEBUG)
            return

        except Exception as e:
            self.log(f'`live()` failed. {e}', level=logging.ERROR)
            raise e
    
    @runtime_tracker
    async def handle_request(self, 
                             socks : dict, 
                             agent_socket : zmq.Socket, 
                             agent_broadcasts : zmq.Socket, 
                             manager_socket : zmq.Socket
                             ) -> bool:
        
        # read message from agents
        if agent_socket in socks:
            return await self.handle_agent_request()

        # check if agents broadcast any information
        elif agent_broadcasts in socks:
            return await self.handle_agent_broadcast()

        # check if manager message is received
        elif manager_socket in socks:
            return await self.handle_manager_broadcast()

    @runtime_tracker
    async def handle_agent_request(self) -> bool:
        _, src, content = await self.listen_peer_message()

        if content['msg_type'] == SimulationMessageTypes.OBSERVATION.value:
            resp = self.handle_observation(content)

        elif content['msg_type'] == SimulationMessageTypes.AGENT_STATE.value:
            resp = self.handle_agent_state(content)

        else:
            # message is of an unsopported type. send blank response
            self.log(f"received message of type {content['msg_type']}. ignoring message...")
            resp = NodeReceptionIgnoredMessage(self.get_element_name(), src)

        await self.respond_peer_message(resp)
                            
        return True
    
    @runtime_tracker
    async def respond_peer_message(self, resp: SimulationMessage) -> None:
        return await super().respond_peer_message(resp)

    @runtime_tracker
    async def handle_agent_broadcast(self) -> bool:
        *_, content = await self.listen_peer_broadcast()

        if content['msg_type'] == SimulationMessageTypes.MEASUREMENT_REQ.value:
            # some agent broadcasted a measurement request
            measurement_req : TaskRequest = TaskRequest.from_dict(content['req'])

            # add to list of received measurement requests 
            self.measurement_reqs.add(measurement_req)

        elif content['msg_type'] == SimulationMessageTypes.BUS.value:
            # an agent made a broadcast of multiple measurements
            bus_msg : BusMessage = message_from_dict(**content)
            
            # filter out measurement request messages
            measurement_reqs : list[TaskRequest] \
                = [ TaskRequest.from_dict(msg['req'])
                    for msg in bus_msg.msgs
                    if msg['msg_type'] == SimulationMessageTypes.MEASUREMENT_REQ.value]
            
            # add to list of received measurement requests 
            self.measurement_reqs.update(measurement_reqs)

        # add to list of received broadcasts
        content['t_msg'] = self.get_current_time()
        self.broadcasts_history.append(content)

        return True
    
    @runtime_tracker
    async def handle_manager_broadcast(self) -> bool:
        dst, src, content = await self.listen_manager_broadcast()

        if (dst in self.name 
            and SimulationElementRoles.MANAGER.value in src 
            and content['msg_type'] == ManagerMessageTypes.SIM_END.value
            ):
            # sim end message received
            self.log(f"received message of type {content['msg_type']}. ending simulation...")
            return False

        elif content['msg_type'] == ManagerMessageTypes.TOC.value:
            # toc message received

            # unpack message
            t = content['t']
            
            # update internal databases
            time_step = self.get_orbitdata_time_step()
            if self.t_update is None or abs(self.t_update - t) / time_step > 10:
            # if self.get_current_time() < msg.t:
                self.update_databases(t)

            # update internal clock
            self.log(f"received message of type {content['msg_type']}. updating internal clock to {t}[s]...")
            await self.update_current_time(t)


            # wait for all agent's to send their updated states
            self.log(f"internal clock uptated to time {self.get_current_time()}[s]!")
        
        else:
            # ignore message
            self.log(f"received message of type {content['msg_type']}. ignoring message...")

        return True
    
    def get_orbitdata_time_step(self) -> float:
        for _,agent_orbitdata in self.orbitdata.items():
            agent_orbitdata : OrbitData
            return agent_orbitdata.time_step

    @runtime_tracker
    def update_databases(self, t : float) -> None:
        # TODO fix update orbitdata to consider two time steps ago
        for _,agent_orbitdata in self.orbitdata.items(): 
            agent_orbitdata : OrbitData
            t_update = self.t_update if self.t_update is not None else 0.0
            agent_orbitdata.update_databases(t_update)

        # update events
        self.events = [event for event in self.events if event.is_active(t) or event.is_future(t)]
        
        # update time tracker
        self.t_update = t

    @runtime_tracker
    def handle_observation(self, content : dict) -> SimulationMessage:
        # unpack message
        msg = ObservationResultsMessage(**content)
        self.log(f'received masurement data request from {msg.src}. quering measurement results...')
        agent_state = SimulationAgentState.from_dict(msg.agent_state)
        instrument = Instrument.from_dict(msg.instrument) if isinstance(msg.instrument, dict) else msg.instrument

        # find/generate measurement results
        observation_data = self.query_measurement_data(agent_state, instrument, msg.t_start, msg.t_end)

        # repsond to request
        self.log(f'measurement results obtained! responding to request')
        resp : ObservationResultsMessage = copy.deepcopy(msg)
        resp.dst = resp.src
        resp.src = self.get_element_name()
        resp.observation_data = observation_data

        # save observation
        self.observation_history.append(resp)

        # return observation response
        return resp
    
    @runtime_tracker
    def handle_agent_state(self, content : dict) -> SimulationMessage:
        # unpack message
        msg = AgentStateMessage(**content)
        self.log(f'state message received from {msg.src}. updating state tracker...')

        # update agent state
        updated_state = self.update_agent_state(msg)

        # create state response message
        updated_state_msg = AgentStateMessage(self.get_element_name(), msg.src, updated_state)

        # initiate response message list
        resp_msgs = [updated_state_msg.to_dict()]

        # update agent connectivity 
        resp_msgs.extend(self.update_agent_connectivity(msg))

        # send response
        return BusMessage(self.get_element_name(), msg.src, resp_msgs)

    @runtime_tracker
    def get_current_time(self) -> float:
        return super().get_current_time()

    @runtime_tracker
    def update_agent_state(self, msg : AgentStateMessage) -> dict:
        # 
        if msg.src not in self.agent_state_update_times: 
            self.agent_state_update_times[msg.src] = -1.0

        # TODO support ground stations
        
        # check if time has passed between state updates
        sat_orbitdata : OrbitData = self.orbitdata[msg.src]
        t_state_update = round(self.agent_state_update_times[msg.src] / sat_orbitdata.time_step)
        t_curr = round(self.get_current_time() / sat_orbitdata.time_step)

        if abs(t_state_update - t_curr) < 1 and self.agent_state_update_times[msg.src] >= 0.0: 
            updated_state = msg.state

        else:
            # check current state
            if msg.src in self.agents[self.SPACECRAFT]:
                # look up orbit state
                pos, vel, eclipse = self.get_updated_orbit_state(sat_orbitdata, self.get_current_time())

                # update state
                updated_state = msg.state
                updated_state['pos'] = pos
                updated_state['vel'] = vel
                updated_state['eclipse'] = int(eclipse)

            elif msg.src in self.agents[self.UAV]:
                # Do NOT update
                updated_state = msg.state

            elif msg.src in self.agents[self.GROUND_STATION]:
                # Do NOT update state
                updated_state = msg.state

            else:
                raise ValueError(f'Unrecognized agent performed an update state request. Agent {msg.src} is not part of this simulation.')

            updated_state['t'] = max(self.get_current_time(), updated_state['t'])
            self.agent_state_update_times[msg.src] = updated_state['t']

        return updated_state
    
    @runtime_tracker
    def get_updated_orbit_state(self, orbitdata : OrbitData, t : float) -> tuple:
        # look up orbit state
        return orbitdata.get_orbit_state(t)
    
    @runtime_tracker
    def update_agent_connectivity(self, msg : SimulationMessage) -> list:
        # initiate update list
        resp_msgs = []

        # check connectivity status
        for target_type in self.agents:
            for target in self.agents[target_type]:
                # do not compare to self
                if target == msg.src: continue
                
                # check updated connectivity
                connected = self.check_agent_connectivity(msg.src, target, target_type)
                
                # check if it changes from previously known connectivity state
                if connected == 0 and self.agent_connectivity[msg.src][target] == -1:
                    # no change found; do not announce
                    pass

                    # update internal state
                    self.agent_connectivity[msg.src][target] = connected
                    continue

                if self.agent_connectivity[msg.src][target] != connected:
                    # change found; make announcement 
                    connectivity_update = AgentConnectivityUpdate(msg.src, target, connected)
                    resp_msgs.append(connectivity_update.to_dict())

                    # update internal state
                    self.agent_connectivity[msg.src][target] = connected

        return resp_msgs
    
    @runtime_tracker
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
        if self.connectivity == 'FULL': return True

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
                                agent_state : SimulationAgentState, 
                                instrument : Instrument,
                                t_start : float,
                                t_end : float
                                ) -> dict:
        """
        Queries internal models or data and returns observation information being sensed by the agent
        """

        if isinstance(agent_state, SatelliteAgentState):
            agent_orbitdata : OrbitData = self.orbitdata[agent_state.agent_name]

            # get time indexes and bounds
            t_u = t_end
            t_l = t_start
            
            # get names of the columns of the available coverage data
            orbitdata_columns : list = list(agent_orbitdata.gp_access_data.columns.values)
            
            # get satellite's off-axis angle
            satellite_off_axis_angle = agent_state.attitude[0]
            
            # collect data for every instrument model onboard
            obs_data = []
            for instrument_model in instrument.mode:
                # get observation FOV from instrument model
                if isinstance(instrument_model, BasicSensorModel):
                    instrument_fov : ViewGeometry = instrument_model.get_field_of_view()
                    instrument_fov_geometry : SphericalGeometry = instrument_fov.sph_geom
                    instrument_off_axis_fov = instrument_fov_geometry.angle_width / 2.0
                elif isinstance(instrument_model, PassiveOpticalScannerModel):
                    instrument_fov : ViewGeometry = instrument_model.get_field_of_view()
                    instrument_fov_geometry : SphericalGeometry = instrument_fov.sph_geom
                    instrument_off_axis_fov = instrument_fov_geometry.angle_width / 2.0

                else:
                    raise NotImplementedError(f'measurement data query not yet suported for sensor models of type {type(instrument_model)}.')

                # query coverage data of everything that is within the field of view of the agent
                matching_data = [ (time_index,gp_index,pnt_opt,lat,lon,obs_range,look_angle,*_,instrument_name,agent_name)
                                  for time_index,gp_index,pnt_opt,lat,lon,obs_range,look_angle,*_,instrument_name,agent_name in agent_orbitdata.gp_access_data.values
                                  if t_l < time_index*agent_orbitdata.time_step <= t_u                            # is being observed at this given time
                                  and instrument_name == instrument.name                # is being observed by the correct instrument
                                  and abs(satellite_off_axis_angle - look_angle) <= instrument_off_axis_fov + 1e-6      # agent is pointing at the ground point
                                ]
                
                if not matching_data:
                    desired_accesses = [ (time_index,gp_index,*_,instrument_name,agent_name)
                                  for time_index,gp_index,*_,instrument_name,agent_name in agent_orbitdata.gp_access_data.values
                                  if t_l < time_index*agent_orbitdata.time_step <= t_u  # is being observed at this given time
                                  and instrument_name == instrument.name                # is being observed by the correct instrument
                    ]
                    x = 1
                    for time_index,gp_index,pnt_opt,lat,lon,obs_range,look_angle,*_,instrument_name,agent_name in desired_accesses:
                        off_axis_angle = abs(satellite_off_axis_angle - look_angle)
                        if off_axis_angle <= instrument_off_axis_fov + 1e-6:
                            x =1
                        else:
                            x = 1


                # compile data
                unique_targets = {(lat,lon) 
                                  for time_index,gp_index,pnt_opt,lat,lon,*_ in matching_data}
                for lat,lon in unique_targets:
                    matching_observations = [(time_index,gp_index,pnt_opt,lat,lon,*_) 
                                             for time_index,gp_index,pnt_opt,lat_obs,lon_obs,*_ in matching_data
                                             if abs(lat_obs - lat ) < 1e-3
                                             and abs(lon_obs - lon) < 1e-3]
                    merged_observation = { column : [] for column in orbitdata_columns }
                    merged_observation['t_start'] = np.Inf
                    merged_observation['t_end'] = np.NINF
                    for matching_observation in matching_observations:
                        datum = { column : matching_observation[orbitdata_columns.index(column)] for column in orbitdata_columns }
                        
                        datum['t_start'] = matching_observation[0] * agent_orbitdata.time_step
                        datum['t_end'] = matching_observation[0] * agent_orbitdata.time_step
                        
                        for column in orbitdata_columns:
                            if (column in ['instrument', 'agent name', 'grid index', 'GP index', 'lat [deg]', 'lon [deg]', 'pnt-opt index']
                                and len(merged_observation[column]) > 0): 
                                continue
                            merged_observation[column].append(datum[column])

                        merged_observation['t_start'] = min(datum['t_start'], merged_observation['t_start'])
                        merged_observation['t_end'] = max(datum['t_end'], merged_observation['t_end'])

                    for key in merged_observation:
                        if isinstance(merged_observation[key], list) and len(merged_observation[key]) == 1:
                            merged_observation[key] = merged_observation[key][0]

                    obs_data.append(merged_observation)

            # return processed observation data
            return obs_data

        else:
            raise NotImplementedError(f"Measurement results query not yet supported for agents with state of type {type(agent_state)}")

    def query_event_data(self, lat_img, lon_img, t_img, instrument_name) -> list:
        """ Checks any of the events in its database is being observed and return its severity and required measurements """

        return [{"severity" : severity, "measurements" : measurements }
                for lat,lon,t_start,duration,severity,measurements in self.events.values
                if lat==lat_img 
                and lon==lon_img
                and t_start<= t_img <=t_start+duration
                and instrument_name in measurements  #TODO include better reasoning]
                ]
    
    async def teardown(self) -> None:
        try:
            self.t_f = time.perf_counter()

            self.log('Compiling results...',level=logging.WARNING)

            # compile observations performed
            observations_performed : pd.DataFrame = self.compile_observations()

            # log and save results
            # self.log(f"MEASUREMENTS RECEIVED:\n{len(observations_performed.values)}\n\n", level=logging.WARNING)
            observations_performed.to_csv(f"{self.results_path}/measurements.csv", index=False)
            
            # commpile list of broadcasts performed
            broadcasts_performed : pd.DataFrame = self.compile_broadcasts()

            # log and save results
            # self.log(f"BROADCASTS RECEIVED:\n{len(broadcasts_performed.values)}\n\n", level=logging.WARNING)
            broadcasts_performed.to_csv(f"{self.results_path}/broadcasts.csv", index=False)

            # compile list of measurement requests 
            measurement_reqs : pd.DataFrame = self.compile_requests()

            # log and save results
            # self.log(f"MEASUREMENT REQUESTS RECEIVED:\n{len(measurement_reqs.values)}\n\n", level=logging.WARNING)
            measurement_reqs.to_csv(f"{self.results_path}/requests.csv", index=False)

            # log performance stats
            runtime_dir = os.path.join(self.results_path, "runtime")
            if not os.path.isdir(runtime_dir): os.mkdir(runtime_dir)

            columns = ['routine','t_avg','t_std','t_med','t_max','t_min','n','t_total']
            data = []

            n_decimals = 3
            for routine in tqdm(self.stats, desc="ENVIRONMENT: Compiling runtime statistics", leave=False):
                # compile stats
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

            stats_df = pd.DataFrame(data, columns=columns)
            # self.log(f'\nENVIRONMENT RUN-TIME STATS\n{str(stats_df)}\n', level=logging.WARNING)
            stats_df.to_csv(f"{self.results_path}/runtime_stats.csv", index=False)

            # print final time
            print('\n')
            self.log(f'successfully shutdown', level=logging.WARNING)
        
        except asyncio.CancelledError as e:
            raise e
        except Exception as e:
            print('\n','\n','\n')
            print(e.with_traceback())
            raise e        
            
    def compile_observations(self) -> pd.DataFrame:
        columns = None
        data = []
        for msg in tqdm(self.observation_history, desc='Compiling observations results', leave=True):
            msg : ObservationResultsMessage
            observation_data : list = msg.observation_data
            observer = msg.dst

            for obs in observation_data:
                
                # find column names 
                if columns is None:
                    columns = [key for key in obs]
                    columns.insert(0, 'observer')
                    columns.insert(2, 't_img')
                    columns.remove('t_start')
                    columns.remove('t_end')

                # add observation to data list
                obs['observer'] = observer
                for key in columns:
                    val = obs.get(key, None)
                    if isinstance(val, list):
                        if len(val) == 1:
                            obs[key] = val[0]
                        else:
                            obs[key] = [val[0], val[-1]]

                obs['t_img'] = [obs['t_start'], obs['t_end']]
                obs.pop('t_start')
                obs.pop('t_end')
                data.append([obs[key] for key in columns])

        return pd.DataFrame(data=data, columns=columns)
    
    def compile_broadcasts(self) -> pd.DataFrame:
        columns = ['t_msg', 'Sender', 'Message Type', 
                #    'Message'
                   ]
        data = [[msg['t_msg'], 
                 msg['src'], 
                 msg['msg_type'],
                 #  json.dumps(msg)
                 ]
                for msg in self.broadcasts_history]
            
        return pd.DataFrame(data=data, columns=columns)
    
    def compile_requests(self) -> pd.DataFrame:
        columns = ['ID', 'Requester', 'lat [deg]', 'lon [deg]', 'Severity', 't start', 't end', 't corr', 'Measurment Types']
        data = [[req.id,
                 req.requester,
                 req.target[0],
                 req.target[1],
                 req.severity,
                 req.t_start,
                 req.t_end,
                 req.t_corr,
                 req.observation_types] 
                 for req in self.measurement_reqs]

        return pd.DataFrame(data=data, columns=columns)

    def calc_coverage_metrics(self) -> tuple:
        # TODO improve performance or load precomputed vals
        return np.NAN, np.NAN, np.NAN
            
        # compile coverage calcs 
        consolidated_orbitdata = None

        for _,agent_orbitdata in self.orbitdata.items():
            agent_orbitdata : OrbitData
            if consolidated_orbitdata is None:
                consolidated_orbitdata : OrbitData = agent_orbitdata.copy()
                consolidated_orbitdata.agent_name = 'all'
                continue

            consolidated_orbitdata.gp_access_data = pd.concat([consolidated_orbitdata.gp_access_data, agent_orbitdata.gp_access_data],
                                                               axis=0)

        # calculate coverage metrics          
        if consolidated_orbitdata is not None:
            return consolidated_orbitdata.calculate_percent_coverage() 
        else: 
            return np.NAN, np.NAN, np.NAN
       
        # # calculate possible number of measurements given coverage metrics
        # n_obervations_pos = 0
        # for req in measurement_reqs:

        #     req : MeasurementRequest
        #     lat,lon,_ = req.target

        #     observable_measurements = []
        #     for _, coverage_data in self.orbitdata.items():
        #         coverage_data : OrbitData
        #         req_start = req.t_start/coverage_data.time_step
        #         req_end = req.t_end/coverage_data.time_step
        #         grid_index, gp_index, gp_lat, gp_lon = coverage_data.find_gp_index(lat,lon)

        #         df = coverage_data.gp_access_data.query('`time index` >= @req_start & `time index` <= @req_end & `GP index` == @gp_index & `grid index` == @grid_index')

        #         # if not df.empty:
        #         #     print(df['time index'] * coverage_data.time_step)

        #         for _, row in df.iterrows():
        #             instrument : str = row['instrument']
        #             if (instrument in req.observations_types 
        #                 and instrument not in observable_measurements):
        #                 observable_measurements.append(instrument)

        #             if len(observable_measurements) == len(req.observations_types):
        #                 break

        #         if len(observable_measurements) == len(req.observations_types):
        #             break

        #     n_obervations_pos += len(observable_measurements)

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

    @runtime_tracker
    async def listen_internal_broadcast(self) -> tuple:
        return await super().listen_internal_broadcast()
    
    @runtime_tracker
    async def listen_manager_broadcast(self) -> tuple:
        return await super().listen_manager_broadcast()
    
    @runtime_tracker
    async def listen_peer_broadcast(self) -> tuple:
        return await super().listen_peer_broadcast()
    
    @runtime_tracker
    async def listen_internal_message(self) -> tuple:
        return await super().listen_internal_message()
    
    @runtime_tracker
    async def listen_peer_message(self) -> tuple:
        return await super().listen_peer_message()