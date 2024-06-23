import copy
import csv
import os
import time
from typing import Any, Callable
import numpy as np
import pandas as pd
from pandas import DataFrame
from zmq import asyncio as azmq

from instrupy.base import Instrument
from instrupy.base import BasicSensorModel
from instrupy.util import SphericalGeometry, ViewGeometry

from chess3d.agents.science.requests import *
from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.states import GroundStationAgentState, UAVAgentState, SatelliteAgentState
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
                scenario_path : dict,
                results_path : str, 
                env_network_config: NetworkConfig, 
                manager_network_config: NetworkConfig, 
                utility_func : Callable[[], Any], 
                # measurement_reqs : list = [],
                events_path : str = None,
                level: int = logging.INFO, 
                logger: logging.Logger = None) -> None:
        super().__init__(env_network_config, manager_network_config, [], level, logger)

        # setup results folder:
        self.results_path : str = os.path.join(results_path, self.get_element_name().lower())

        # load observation data
        self.orbitdata : dict = OrbitData.from_directory(scenario_path)

        # load agent names and classify by type of agent
        self.agents = {}
        agent_names = []
        scenario_specs_filename = os.path.join(scenario_path, 'MissionSpecs.json')

        with open(scenario_specs_filename, 'r') as scenario_specs:
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
        self.observation_history = []
        self.agent_connectivity = {}
        for src in agent_names:
            for target in agent_names:
                if src not in self.agent_connectivity:
                    self.agent_connectivity[src] = {}    
                
                self.agent_connectivity[src][target] = -1

        self.measurement_reqs = []

        self.stats = {}
        self.events : pd.DataFrame = pd.read_csv(events_path) if events_path is not None else None

        self.t_0 = None
        self.t_f = None

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
                socks = dict(await poller.poll())

                if agent_socket in socks:
                    # read message from agents
                    dst, src, content = await self.listen_peer_message()

                    t_0 = time.perf_counter()
                    
                    if content['msg_type'] == SimulationMessageTypes.OBSERVATION.value:
                        # unpack message
                        msg = ObservationResultsMessage(**content)
                        self.log(f'received masurement data request from {msg.src}. quering measurement results...')
                        agent_state = SimulationAgentState.from_dict(msg.agent_state)
                        instrument = Instrument.from_dict(msg.instrument)

                        # find/generate measurement results
                        observation_data = self.query_measurement_data(agent_state, 
                                                                       instrument)

                        # repsond to request
                        self.log(f'measurement results obtained! responding to request')
                        resp : ObservationResultsMessage = copy.deepcopy(msg)
                        resp.dst = resp.src
                        resp.src = self.get_element_name()
                        resp.observation_data = observation_data

                        # save observation
                        self.observation_history.append(resp)

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
                    # check if agents broadcast any information
                    dst, src, content = await self.listen_peer_broadcast()

                    if content['msg_type'] == SimulationMessageTypes.MEASUREMENT_REQ.value:
                        # some agent broadcasted a measurement request
                        req_msg = MeasurementRequestMessage(**content)
                        measurement_req : MeasurementRequest = MeasurementRequest.from_dict(req_msg.req)

                        # add to list of received measurement requests 
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
                                agent_state : SimulationAgentState, 
                                instrument : Instrument
                                ) -> dict:
        """
        Queries internal models or data and returns observation information being sensed by the agent
        """

        if isinstance(agent_state, SatelliteAgentState):
            agent_orbitdata : OrbitData = self.orbitdata[agent_state.agent_name]

            # get time indexes and bounds
            t = agent_state.t/agent_orbitdata.time_step
            t_u = t + 1
            t_l = t - 1
            
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
                else:
                    raise NotImplementedError(f'measurement data query not yet suported for sensor models of type {type(instrument_model)}.')

                # query coverage data of everything that is within the field of view of the agent
                raw_coverage_data = [list(data)
                                    for data in agent_orbitdata.gp_access_data.values
                                    if t_l < data[orbitdata_columns.index('time index')] < t_u # is being observed at this given time
                                    and data[orbitdata_columns.index('instrument')] == instrument.name # is being observed by the correct instrument
                                    and abs(satellite_off_axis_angle - data[orbitdata_columns.index('look angle [deg]')]) <= instrument_off_axis_fov # agent is pointing at the ground point
                                    ]
                
                raw_coverage_data_no_fov = [
                                    list(data)
                                    for data in agent_orbitdata.gp_access_data.values
                                    if t_l < data[orbitdata_columns.index('time index')] < t_u # is being observed at this given time
                                    and data[orbitdata_columns.index('instrument')] == instrument.name # is being observed by the correct instrument
                                    ]
                
                if not raw_coverage_data and raw_coverage_data_no_fov:
                    # for data in raw_coverage_data_no_fov:
                    #     look_angle_index = orbitdata_columns.index('look angle [deg]')
                    #     look_angle = data[look_angle_index]
                    #     pointing_angle = satellite_off_axis_angle

                    #     in_fov = abs(look_angle - pointing_angle) <= instrument_off_axis_fov/2.0
                    x = 1

                # compile data
                for data in raw_coverage_data:                    
                    obs_data.append({
                        "t_img"     : data[orbitdata_columns.index('time index')]*agent_orbitdata.time_step,
                        "lat"       : data[orbitdata_columns.index('lat [deg]')],
                        "lon"       : data[orbitdata_columns.index('lon [deg]')],
                        "range"     : data[orbitdata_columns.index('observation range [km]')],
                        "look"      : data[orbitdata_columns.index('look angle [deg]')],
                        "incidence" : data[orbitdata_columns.index('incidence angle [deg]')],
                        "zenith"         : data[orbitdata_columns.index('solar zenith [deg]')],
                        "instrument_name": data[orbitdata_columns.index('instrument')]
                    })

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

            # print final time
            self.log(f'Environment shutdown with internal clock of {self.get_current_time()}[s]', level=logging.WARNING)
            
            # compile observations performed
            observations_performed : pd.DataFrame = self.compile_observations()

            # log and save results
            self.log(f"MEASUREMENTS RECEIVED:\n{str(observations_performed)}\n\n", level=logging.WARNING)
            observations_performed.to_csv(f"{self.results_path}/measurements.csv", index=False)
            
            # TODO: move this section to external results compiler script -------------
            # count observations performed
            n_events, n_events_obs, n_events_partially_co_obs, \
                n_events_fully_co_obs, n_events_detected \
                    = self.count_observations(observations_performed)

            # calculate coverage
            n_gps, n_gps_accessible, n_gps_access_ptg = self.calc_coverage_metrics()
            
            # count number of GPs observed
            gps_observed : set = {(lat,lon) for _,_,lat,lon,*_ in observations_performed.values}
            n_gps_observed = len(gps_observed)

            # Generate summary
            summary_headers = ['stat_name', 'val']
            summary_data = [
                        ['t_start', self._clock_config.start_date], 
                        ['t_end', self._clock_config.end_date], 
                        ['n_events', n_events],
                        ['n_gps', n_gps],
                        ['n_gps_accessible', n_gps_accessible],
                        ['n_gps_access_ptg', n_gps_access_ptg],
                        ['n_gps_obs', n_gps_observed],
                        ['n_events_obs', n_events_obs],
                        ['n_event_partially_co_obs', n_events_partially_co_obs],
                        ['n_events_fully_co_obs', n_events_fully_co_obs],
                        ['n_co_obs', n_events_fully_co_obs + n_events_partially_co_obs],
                        ['n_events_detected', n_events_detected],
                        ['t_runtime', self.t_f - self.t_0]
                    ]

            summary_df = DataFrame(summary_data, columns=summary_headers)
            self.log(f"\nSIMULATION RESULTS SUMMARY:\n{str(summary_df)}\n\n", level=logging.WARNING)
            summary_df.to_csv(f"{self.results_path}/../summary.csv", index=False)
            # --------------------------------------------------------------------

            # log performance stats
            n_decimals = 3
            columns = ['routine','t_avg','t_std','t_med','n']
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

            stats_df = pd.DataFrame(data, columns=columns)
            self.log(f'\nENVIRONMENT RUN-TIME STATS\n{str(stats_df)}\n', level=logging.WARNING)
            stats_df.to_csv(f"{self.results_path}/runtime_stats.csv", index=False)
        
        except asyncio.CancelledError as e:
            raise e
        except Exception as e:
            print('\n','\n','\n')
            print(e.with_traceback())
            raise e        
            

    def compile_observations(self) -> pd.DataFrame:
        columns = None
        data = []
        for msg in self.observation_history:
            msg : ObservationResultsMessage
            observation_data : list = msg.observation_data
            observer = msg.dst

            for obs in observation_data:
                
                # find column names 
                if columns is None:
                    columns = [key for key in obs]
                    columns.insert(0, 'observer')
                    
                # add observation to data list
                obs['observer'] = observer
                data.append([obs[key] for key in columns])

        return DataFrame(data, columns=columns)

    def count_observations(self, observations_performed : pd.DataFrame) -> tuple:
        if self.events is not None: # events present in the simulation
            # initalize list of events observed
            events_observed = []
            events_detected = []
            
            # count total number of events in the simulation
            for lat,lon,t_start,duration,severity,observations_req in self.events.values:
                
                # find observations that overlooked a given event's location
                matching_observations = [(lat, lon, t_start, duration, severity, observer, t_img, instrument, observations_req)
                                            for observer,t_img,lat_img,lon_img,*_,instrument in observations_performed.values
                                            if abs(lat - lat_img) < 1e-3 
                                        and abs(lon - lon_img) < 1e-3
                                        and t_start <= t_img <= t_start+duration
                                        and instrument in observations_req  #TODO include better reasoning!
                                            ]                    

                # check if observations were found
                if matching_observations:
                    # add to list of observed events
                    events_observed.append(matching_observations)

                # find measurement requests that match the event bieng
                matching_requests = [req
                                        for req in self.measurement_reqs
                                        if isinstance(req, MeasurementRequest)
                                        and abs(lat - req.target[0]) < 1e-3 
                                        and abs(lon - req.target[1]) < 1e-3
                                        and t_start <= req.t_start <= req.t_end <= t_start+duration
                                        and all([instrument in observations_req for instrument in req.observations_types])
                                    ]

                if matching_requests:
                    events_detected.append(matching_requests)

            # count number of events
            n_events = len(self.events.values)                

            # count number of observed events
            n_events_obs = len(events_observed)

            # count number of co-observed events 
            n_events_partially_co_obs = 0
            n_events_fully_co_obs = 0
            for matching_observations in events_observed:
                # ge required measurements for a given event
                *_,observations_req = matching_observations[-1]
                observations_req : str 
                observations_req = observations_req.replace('[','')
                observations_req = observations_req.replace(']','')
                observations_req = observations_req.split(',')

                observations = {instrument for *_, instrument, _ in matching_observations}
                if all([obs in observations_req for obs in observations]):
                    if len(observations) == len(observations_req):
                        n_events_fully_co_obs += 1
                    elif len(observations) > 2:
                        n_events_partially_co_obs += 1

            # count number of events detected
            n_events_detected = len(events_detected)

        else: # no events present in the simulation
            n_events = 0
            n_events_obs = 0
            n_events_partially_co_obs = 0
            n_events_fully_co_obs = 0
            n_events_detected = 0   

        return n_events, n_events_obs, n_events_partially_co_obs, n_events_fully_co_obs, n_events_detected

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
