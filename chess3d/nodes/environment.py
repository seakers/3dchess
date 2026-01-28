from collections import defaultdict
import copy
import os
import time
from typing import Dict
from dmas.elements import SimulationMessage
from dmas.messages import SimulationMessage
import numpy as np
import pandas as pd
from tqdm import tqdm
from zmq import asyncio as azmq

from instrupy.base import Instrument
from instrupy.base import BasicSensorModel
from instrupy.passive_optical_scanner_model import PassiveOpticalScannerModel
from instrupy.util import SphericalGeometry, ViewGeometry

from execsatm.events import GeophysicalEvent
from execsatm.tasks import EventObservationTask

from chess3d.agents.science.requests import *
from chess3d.orbitdata import OrbitData
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
    
    def __init__(self, 
                results_path : str, 
                orbitdata_dir : str,
                scenario_duration : float,
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
        self.orbitdata : Dict[str,OrbitData] = OrbitData.from_directory(orbitdata_dir, scenario_duration) if orbitdata_dir is not None else None

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
        self.agents[SimulationAgentTypes.SATELLITE] = sat_names

        # load uav names
        uav_names = []
        if uav_list:
            for uav in uav_list:
                uav : dict
                uav_name = uav.get('name')
                uav_names.append(uav_name)
                agent_names.append(uav_name)
        self.agents[SimulationAgentTypes.UAV] = uav_names

        # load GS agent names
        gs_names : list = []
        if gs_list:
            for gs in gs_list:
                gs : dict
                gs_name = gs.get('name')
                gs_names.append(gs_name)
                agent_names.append(gs_name)
        self.agents[SimulationAgentTypes.GROUND_OPERATOR] = gs_names

        # load events
        self.events_path : str = events_path
        self.events : List[GeophysicalEvent] = self.load_events(events_path)

        # initialize parameters
        self.connectivity = connectivity
        self.observation_history = []
        self.agent_connectivity = defaultdict(lambda: defaultdict(lambda: -1))
        for src in agent_names:
            for target in agent_names:
                if src not in self.agent_connectivity:
                    self.agent_connectivity[src] = {}    
                if target != src:
                    self.agent_connectivity[src][target] = -1

        self.agent_state_update_times = {}

        # self.measurement_reqs : set[TaskRequest] = set()
        # self.measurement_reqs : set[dict] = set()
        self.task_reqs : list[dict] = list()
        
        self.t_0 = None
        self.t_f = None
        self.t_update = None

        self.broadcasts_history = []
        
    def load_events(self, events_path : str) -> List[GeophysicalEvent]:
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
                (row['lat [deg]'], row['lon [deg]'], row.get('grid index', 0), row['gp_index']),
                row['start time [s]'],
                row['duration [s]'],
                row['severity'],
                row['start time [s]'],
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
                # listen for requests
                req_status : bool = await self.listen_for_requests(poller, agent_socket, agent_broadcasts, manager_socket)
                
                # check if end of simulation message was received
                if not req_status: 
                    # if so, print final results and exit live loop
                    return self.print_results()                   

        except asyncio.CancelledError:
            self.log(f'`live()` interrupted. {e}', level=logging.DEBUG)
            return

        except Exception as e:
            self.log(f'`live()` failed. {e}', level=logging.ERROR)
            raise e
        
    @runtime_tracker
    async def listen_for_requests(self, poller : azmq.Poller, agent_socket : zmq.Socket, agent_broadcasts : zmq.Socket, manager_socket : zmq.Socket) -> bool:
        # get list of sockets with incoming messages or requests
        socks = dict(await poller.poll())

        # handle incoming messages or requests
        return await self.handle_request(socks, agent_socket, agent_broadcasts, manager_socket)
            
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
        # get incoming message
        _, src, content = await self.listen_peer_message()

        # get appropriate handler
        if content['msg_type'] == SimulationMessageTypes.OBSERVATION.value:
            resp = self.handle_observation(content)

        elif content['msg_type'] == SimulationMessageTypes.AGENT_STATE.value:
            resp = self.handle_agent_state(content)

        else:
            # message is of an unsopported type. send blank response
            self.log(f"received message of type {content['msg_type']}. ignoring message...")
            resp = NodeReceptionIgnoredMessage(self.get_element_name(), src)

        # send response
        await self.respond_peer_message(resp)
                            
        return True
    
    @runtime_tracker
    async def listen_peer_message(self) -> tuple:
        return await super().listen_peer_message()

    @runtime_tracker
    async def respond_peer_message(self, resp: SimulationMessage) -> None:
        return await super().respond_peer_message(resp)

    @runtime_tracker
    async def handle_agent_broadcast(self) -> bool:
        # listen for broadcast
        *_, content = await self.listen_peer_broadcast()

        if content['msg_type'] == SimulationMessageTypes.MEASUREMENT_REQ.value:
            # add to list of received broadcasts
            self.task_reqs.append(content['req'])

        elif content['msg_type'] == SimulationMessageTypes.BUS.value:
            # an agent made a broadcast of multiple measurements;            
            # filter out measurement request messages
            measurement_reqs : list[dict] \
                = [ msg['req']
                    for msg in content['msgs']
                    if msg['msg_type'] == SimulationMessageTypes.MEASUREMENT_REQ.value]

            # add to list of received measurement requests 
            self.task_reqs.extend(measurement_reqs)

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
            
            # update internal databases if needed
            # if self.t_update is None or abs(self.t_update - t) // self.get_orbitdata_time_step() > 100: # update every 100 time steps
            
            # if self.t_update is None or abs(self.t_update - t) / 3600 > 1: # update every hour
            #     self.update_databases(t)

            # update internal clock
            self.log(f"received message of type {content['msg_type']}. updating internal clock to {t}[s]...")
            await self.update_current_time(t)

            # wait for all agent's to send their updated states
            self.log(f"internal clock uptated to time {self.get_current_time()}[s]!")

            # TODO TEMPORARY: breakpoint for debugging
            if 94.0 < t < 95.0:
                # update connectivity matrix for debugging
                connectivity = defaultdict(lambda: defaultdict(lambda: 0))
                for sender in self.agent_connectivity:
                    for receiver in self.agent_connectivity:
                        connectivity[sender][receiver] = self.check_agent_connectivity(sender, receiver)

                # print connectivity matrix
                print('\n\n\n')
                for sender in self.agent_connectivity:
                    line = ""
                    for receiver in self.agent_connectivity:
                        if sender == receiver:
                            line += "0,"
                            continue
                        line += f"{connectivity[sender][receiver]},"                        

                    print(line[:-1])
                
                # check for mismatches
                for sender in self.agent_connectivity:
                    for receiver in self.agent_connectivity:
                        if sender == receiver:
                            continue
                        
                        if connectivity[sender][receiver] != self.agent_connectivity[receiver][sender]:
                            a_to_b = self.check_agent_connectivity(sender, receiver)
                            b_to_a = self.check_agent_connectivity(receiver, sender)
                            x = 1 # breakpoint
                            

                        assert connectivity[sender][receiver] == self.agent_connectivity[receiver][sender], \
                            f'Connectivity mismatch between {sender} and {receiver}: {connectivity[sender][receiver]} vs {self.agent_connectivity[receiver][sender]}'

                x = 1 # breakpoint
        
        else:
            # ignore message
            self.log(f"received message of type {content['msg_type']}. ignoring message...")

        return True
    
    def get_simulation_duration(self) -> float:
        return min(agent_orbitdata.duration * 24 * 3600 
                    for agent_orbitdata in self.orbitdata.values())
    
    def get_orbitdata_time_step(self) -> float:
        for agent_orbitdata in self.orbitdata.values():
            return agent_orbitdata.time_step

    @runtime_tracker
    def update_databases(self, t : float) -> None:
        # update orbit databases
        for agent_orbitdata in self.orbitdata.values(): 
            agent_orbitdata.update_databases(t)

        # update events; only keep active and future events
        self.events = [event for event in self.events 
                       if event.is_active(t) or event.is_future(t)]
        
        # update time tracker
        self.t_update = t

    @runtime_tracker
    def handle_observation(self, content : dict) -> SimulationMessage:
        # unpack message
        agent_state_dict = content['agent_state']
        instrument_dict = content['instrument']
        t_start = content['t_start']
        t_end = content['t_end']

        # find/generate measurement results
        observation_data = self.query_measurement_data(agent_state_dict, instrument_dict, t_start, t_end)

        # repsond to request
        self.log(f'measurement results obtained! responding to request')
        resp : ObservationResultsMessage = copy.deepcopy(content)
        resp['dst'] = resp['src']
        resp['src'] = self.get_element_name()
        resp['observation_data'] = observation_data

        # save observation
        self.observation_history.append(resp)

        # return observation response
        return resp

        # TEMP original implementation commented out
        # # unpack message
        # msg = ObservationResultsMessage(**content)
        # self.log(f'received masurement data request from {msg.src}. quering measurement results...')
        # agent_state = SimulationAgentState.from_dict(msg.agent_state)
        # instrument = Instrument.from_dict(msg.instrument) if isinstance(msg.instrument, dict) else msg.instrument

        # # find/generate measurement results
        # observation_data = self.query_measurement_data(agent_state, instrument, msg.t_start, msg.t_end)

        # # DEBUG ----------------
        # # targets_requested : set = {(np.round(lat,3),np.round(lon,3)) for lat,lon,_ in msg.observation_action['targets']}
        # # targets_observed : set = {(obs['lat [deg]'], obs['lon [deg]']) for obs in observation_data}
        # # additional_targets = targets_observed.difference(targets_requested)
        # # if len(targets_requested) > len(targets_observed):
        # #     print(f'\nWARNING: number of targets requested ({len(targets_requested)}) is larger than observed ({len(targets_observed)}) at T={np.round(self.get_current_time(),3)} [s].')
        # # elif additional_targets:
        # #     print(f'\nWARNING: number of targets observed ({len(targets_observed)}) does not match requested targets ({len(msg.observation_action["targets"])}) at T={np.round(self.get_current_time(),3)} [s].')
        # # ----------------------

        # # repsond to request
        # self.log(f'measurement results obtained! responding to request')
        # resp : ObservationResultsMessage = copy.deepcopy(msg)
        # resp.dst = resp.src
        # resp.src = self.get_element_name()
        # resp.observation_data = observation_data

        # # save observation
        # self.observation_history.append(resp)

        # # return observation response
        # return resp
    
    @runtime_tracker
    def handle_agent_state(self, content : dict) -> SimulationMessage:
        try:
            # unpack message
            self.log(f'state message received from {content["src"]}. updating state tracker...')

            # update agent state
            updated_state = self.update_agent_state(content)

            # create state response message
            updated_state_msg = content.copy()
            updated_state_msg['src'] = self.get_element_name()
            updated_state_msg['dst'] = content["src"]
            updated_state_msg['state'] = updated_state

            # initiate response message list
            resp_msgs = [updated_state_msg]

            # update agent connectivity 
            resp_msgs.extend(self.update_agent_connectivity(content))

            # send response
            return BusMessage(self.get_element_name(), content["src"], resp_msgs)
        except Exception as e:
            self.log(f'Error handling agent state: {e}', level=logging.ERROR)
            raise e
    
    @runtime_tracker
    def get_current_time(self) -> float:
        return super().get_current_time()

    @runtime_tracker
    def update_agent_state(self, msg_dict : dict) -> dict:
        # 
        if msg_dict["src"] not in self.agent_state_update_times: 
            self.agent_state_update_times[msg_dict["src"]] = -1.0

        # TODO support ground stations
        
        # check if time has passed between state updates
        sat_orbitdata : OrbitData = self.orbitdata[msg_dict["src"]]
        t_state_update = round(self.agent_state_update_times[msg_dict["src"]] / sat_orbitdata.time_step)
        t_curr = round(self.get_current_time() / sat_orbitdata.time_step)

        if abs(t_state_update - t_curr) < 1 and self.agent_state_update_times[msg_dict["src"]] >= 0.0: 
            updated_state = msg_dict["state"]

        else:
            # check current state
            if msg_dict["src"] in self.agents[SimulationAgentTypes.SATELLITE]:
                # look up orbit state
                pos, vel, eclipse = self.get_updated_orbit_state(sat_orbitdata, self.get_current_time())

                # update state
                updated_state = msg_dict["state"]
                updated_state['pos'] = pos
                updated_state['vel'] = vel
                updated_state['eclipse'] = int(eclipse)

            elif msg_dict["src"] in self.agents[SimulationAgentTypes.GROUND_OPERATOR]:
                # Do NOT update state
                updated_state = msg_dict["state"]

            else:
                raise ValueError(f'Unrecognized agent performed an update state request. Agent {msg_dict["src"]} is not part of this simulation.')

            updated_state['t'] = max(self.get_current_time(), updated_state['t'])
            self.agent_state_update_times[msg_dict["src"]] = updated_state['t']

        return updated_state
    
    @runtime_tracker
    def get_updated_orbit_state(self, orbitdata : OrbitData, t : float) -> tuple:
        # look up orbit state
        return orbitdata.get_orbit_state(t)
    
    @runtime_tracker
    def update_agent_connectivity(self, msg_dict : dict) -> list:
        # initiate update list
        resp_msgs = []

        # check connectivity of sender agent status with all other agents
        for target in self.agent_connectivity[msg_dict["src"]]:
            # check updated connectivity
            connected = self.check_agent_connectivity(msg_dict["src"], target)
            
            # check if it changes from previously known connectivity state
            if connected == 0 and self.agent_connectivity[msg_dict["src"]][target] == -1:
                # no change found; do not announce
                pass

            elif self.agent_connectivity[msg_dict["src"]][target] != connected:
                # change found; make announcement 
                connectivity_update = AgentConnectivityUpdate(msg_dict["src"], target, connected)
                resp_msgs.append(connectivity_update.to_dict())

            # update internal state
            self.agent_connectivity[msg_dict["src"]][target] = connected   

        # TODO use bfs or dfs to propagate connectivity changes through the network
        # queue = [[src] for src in self.agent_connectivity.keys()]
        # visited = set()
        # while queue:
        #     path = queue.pop(0)
        #     node = path[-1]

        #     if node not in visited:
        #         visited.add(node)
                
        #         for adjacent in self.agent_connectivity.get(node, {}):
                    
        #             if self.agent_connectivity[node][adjacent] == 1:
        #                 new_path = list(path)
        #                 new_path.append(adjacent)
        #                 queue.append(new_path)

        #                 # check if connectivity to adjacent has changed
        #                 if self.agent_connectivity[msg_dict["src"]][adjacent] != 1:
        #                     # change found; make announcement 
        #                     connectivity_update = AgentConnectivityUpdate(msg_dict["src"], adjacent, 1)
        #                     resp_msgs.append(connectivity_update.to_dict())
        #                     # update internal state
        #                     self.agent_connectivity[msg_dict["src"]][adjacent] = 1

        return resp_msgs
    
    @runtime_tracker
    def check_agent_connectivity(self, src : str, target : str) -> int:
        """
        Checks if an agent is in communication range with another agent

        #### Arguments:
            - src (`str`): name of agent starting the connection
            - target (`str`): name of agent receving the connection

        #### Returns:
            - connected (`int`): binary value representing if the `src` and `target` are connected
        """
        # check if full connectivity has been assumed
        if self.connectivity == 'FULL': return 1

        # check if orbit data is available for the source agent
        assert src in self.orbitdata, f'No orbit data found for agent `{src}`.'
        
        # check if target is in the list of comms links for the source agent
        if target not in self.orbitdata[src].comms_links: return 0

        # check connectivity based on orbit data
        src_data : OrbitData = self.orbitdata[src]
        return int(src_data.is_accessing_agent(target, self.get_current_time()))

    @runtime_tracker
    def query_measurement_data( self,
                                agent_state_dict : dict, 
                                instrument_dict : dict,
                                t_start : float,
                                t_end : float
                                ) -> dict:
        """
        Queries internal models or data and returns observation information being sensed by the agent
        """

        # if isinstance(agent_state, SatelliteAgentState):
        if agent_state_dict['state_type'] == SimulationAgentTypes.SATELLITE.value:
            # get orbit data for the agent
            agent_orbitdata : OrbitData = self.orbitdata[agent_state_dict['agent_name']]

            # get access data for the agent
            raw_access_data : Dict[str, list] = agent_orbitdata.gp_access_data.lookup_interval(t_start, t_end)
                        
            # get satellite's off-axis angle
            satellite_off_axis_angle = agent_state_dict['attitude'][0]
            
            # collect instrument information
            instrument_name = instrument_dict["name"]
            instruments = np.asarray(raw_access_data["instrument"])
            ID_COLS = {'instrument', 'agent name', 'grid index', 'GP index',
           'lat [deg]', 'lon [deg]', 'pnt-opt index'}
            
            # create instrument mask for data filtering
            inst_mask = (instruments == instrument_name)

            # collect data for every instrument model onboard
            obs_data = []
            for instrument_model in instrument_dict['mode']:
                # get observation FOV from instrument model
                if instrument_model['@type'] == 'Basic Sensor':
                    instrument_off_axis_fov = instrument_model['fieldOfViewGeometry']['angleWidth'] / 2.0
                elif instrument_model['@type'] == 'Passive Optical Scanner':
                    instrument_off_axis_fov = instrument_model['fieldOfViewGeometry']['angleWidth'] / 2.0
                else:
                    raise NotImplementedError(f"measurement data query not yet suported for sensor models of type {instrument_model['model_type']}.")

                # query coverage data of everything that is within the field of view of the agent
                # TODO Add along-track angle checking. Currently assumes that only cross-track maneuverability is available

                angles = np.asarray(raw_access_data["off-nadir axis angle [deg]"])
                angles_inst = angles[inst_mask]            # smaller array

                mask = np.abs(angles_inst - satellite_off_axis_angle) <= instrument_off_axis_fov

                matching_data = {col: np.asarray(vals)[inst_mask][mask] 
                                 for col, vals in tqdm(raw_access_data.items(), 
                                                       desc=f"{self.get_element_name()}-Filtering access data for instrument {instrument_name}...", 
                                                       leave=False)}
                
                # convert columns to arrays once
                cols = {k: np.asarray(v) for k, v in matching_data.items()}
                grid = cols['grid index'].astype(np.int64, copy=False)
                gp   = cols['GP index'].astype(np.int64, copy=False)
                time = cols['time [s]']

                # check if there is any data to process
                if len(time) == 0: continue

                # ---- Build unique groups for (grid, gp) efficiently ----
                # Stack into (n,2) and unique rows
                pairs = np.column_stack((grid, gp))  # shape (n,2)
                _, inv = np.unique(pairs, axis=0, return_inverse=True)
                # inv[i] = group id of row i, groups are 0..G-1

                # Sort rows by group id so each group is contiguous
                order = np.argsort(inv, kind="mergesort")
                inv_sorted = inv[order]

                # Find group boundaries in the sorted order
                # starts: indices in `order` where a new group begins
                starts = np.r_[0, np.flatnonzero(inv_sorted[1:] != inv_sorted[:-1]) + 1]
                ends   = np.r_[starts[1:], len(order)]

                obs_data: list[dict] = []

                # Iterate groups (G is usually much smaller than N)
                for s,e in tqdm(zip(starts, ends), desc=f"{self.get_element_name()}-Merging observation data for instrument {instrument_name}...", unit=' obs', leave=False):
                    idx = order[s:e]  # row indices for this group

                    merged = {
                        't_start': float(np.min(time[idx])),
                        't_end':   float(np.max(time[idx])),
                    }

                    # For ID columns: take first value
                    # For other columns: collect list (or scalar if length 1)
                    for col, arr in cols.items():
                        if col in ID_COLS:
                            v = arr[idx[0]]
                            merged[col] = v.item() if hasattr(v, "item") else v
                        else:
                            v = arr[idx]
                            # Convert numpy scalars to Python types if needed
                            lst = [x.item() if hasattr(x, "item") else x for x in v.tolist()]
                            merged[col] = lst[0] if len(lst) == 1 else lst

                    obs_data.append(dict(merged))

            # return processed observation data
            return obs_data

        else:
            raise NotImplementedError(f"Measurement results query not yet supported for agents with state of type {agent_state_dict['state_type']}")

    # def query_event_data(self, lat_img, lon_img, t_img, instrument_name) -> list:
    #     """ Checks any of the events in its database is being observed and return its severity and required measurements """

    #     return [{"severity" : severity, "measurements" : measurements }
    #             for lat,lon,t_start,duration,severity,measurements in self.events.values
    #             if lat==lat_img 
    #             and lon==lon_img
    #             and t_start<= t_img <=t_start+duration
    #             and instrument_name in measurements  #TODO include better reasoning
    #             ]
    
    def print_results(self) -> None:
        try:
            # set final simulation time
            self.t_f = time.perf_counter()

            # log results compilation start
            self.log('Compiling results...',level=logging.WARNING)

            # compile observations performed
            observations_performed : pd.DataFrame = self.compile_observations()

            # log and save results
            # self.log(f"MEASUREMENTS RECEIVED:\n{len(observations_performed.values)}\n\n", level=logging.WARNING)
            observations_performed.to_parquet(f"{self.results_path}/measurements.parquet", index=False)
            
            # commpile list of broadcasts performed
            broadcasts_performed : pd.DataFrame = self.compile_broadcasts()

            # log and save results
            # self.log(f"BROADCASTS RECEIVED:\n{len(broadcasts_performed.values)}\n\n", level=logging.WARNING)
            broadcasts_performed.to_parquet(f"{self.results_path}/broadcasts.parquet", index=False)

            # compile list of measurement requests 
            measurement_reqs : pd.DataFrame = self.compile_requests()

            # log and save results
            # self.log(f"MEASUREMENT REQUESTS RECEIVED:\n{len(measurement_reqs.values)}\n\n", level=logging.WARNING)
            measurement_reqs.to_parquet(f"{self.results_path}/requests.parquet", index=False)

            # log performance stats
            runtime_dir = os.path.join(self.results_path, "runtime")
            if not os.path.isdir(runtime_dir): os.mkdir(runtime_dir)

            columns = ['routine','t_avg','t_std','t_med','t_max','t_min','n','t_total']
            data = []

            n_decimals = 5
            for routine in tqdm(self.stats, desc="ENVIRONMENT: Compiling runtime statistics", leave=False):
                # compile stats
                n = len(self.stats[routine])
                t_avg = np.round(np.mean(self.stats[routine]),n_decimals) if n > 0 else -1
                t_std = np.round(np.std(self.stats[routine]),n_decimals) if n > 0 else 0.0
                t_median = np.round(np.median(self.stats[routine]),n_decimals) if n > 0 else -1
                t_max = np.round(max(self.stats[routine]),n_decimals) if n > 0 else -1
                t_min = np.round(min(self.stats[routine]),n_decimals) if n > 0 else -1
                t_total = np.round(sum(self.stats[routine]),n_decimals) if n > 0 else 0

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
                routine_dir = os.path.join(runtime_dir, f"time_series-{routine}.parquet")
                routine_df.to_parquet(routine_dir,index=False)

            stats_df = pd.DataFrame(data, columns=columns)
            self.log(f'\nENVIRONMENT RUN-TIME STATS\n{str(stats_df)}\n', level=logging.WARNING)
            stats_df.to_parquet(f"{self.results_path}/runtime_stats.parquet", index=False)
        
        except Exception as e:
            print('\n','\n','\n')
            print(e.with_traceback())
            raise e       

    async def teardown(self) -> None:
        # print final time
        print('\n')
        self.log(f'successfully shutdown', level=logging.WARNING)
        
    #     try:
    #         self.t_f = time.perf_counter()

    #         self.log('Compiling results...',level=logging.WARNING)

    #         # compile observations performed
    #         observations_performed : pd.DataFrame = self.compile_observations()

    #         # log and save results
    #         # self.log(f"MEASUREMENTS RECEIVED:\n{len(observations_performed.values)}\n\n", level=logging.WARNING)
    #         observations_performed.to_csv(f"{self.results_path}/measurements.csv", index=False)
            
    #         # commpile list of broadcasts performed
    #         broadcasts_performed : pd.DataFrame = self.compile_broadcasts()

    #         # log and save results
    #         # self.log(f"BROADCASTS RECEIVED:\n{len(broadcasts_performed.values)}\n\n", level=logging.WARNING)
    #         broadcasts_performed.to_csv(f"{self.results_path}/broadcasts.csv", index=False)

    #         # compile list of measurement requests 
    #         measurement_reqs : pd.DataFrame = self.compile_requests()

    #         # log and save results
    #         # self.log(f"MEASUREMENT REQUESTS RECEIVED:\n{len(measurement_reqs.values)}\n\n", level=logging.WARNING)
    #         measurement_reqs.to_csv(f"{self.results_path}/requests.csv", index=False)

    #         # log performance stats
    #         runtime_dir = os.path.join(self.results_path, "runtime")
    #         if not os.path.isdir(runtime_dir): os.mkdir(runtime_dir)

    #         columns = ['routine','t_avg','t_std','t_med','t_max','t_min','n','t_total']
    #         data = []

    #         n_decimals = 3
    #         for routine in tqdm(self.stats, desc="ENVIRONMENT: Compiling runtime statistics", leave=False):
    #             # compile stats
    #             n = len(self.stats[routine])
    #             t_avg = np.round(np.mean(self.stats[routine]),n_decimals) if n > 0 else -1
    #             t_std = np.round(np.std(self.stats[routine]),n_decimals) if n > 0 else 0.0
    #             t_median = np.round(np.median(self.stats[routine]),n_decimals) if n > 0 else -1
    #             t_max = np.round(max(self.stats[routine]),n_decimals) if n > 0 else -1
    #             t_min = np.round(min(self.stats[routine]),n_decimals) if n > 0 else -1
    #             t_total = n * t_avg

    #             line_data = [ 
    #                             routine,
    #                             t_avg,
    #                             t_std,
    #                             t_median,
    #                             t_max,
    #                             t_min,
    #                             n,
    #                             t_total
    #                             ]
    #             data.append(line_data)

    #             # save time-series
    #             time_series = [[v] for v in self.stats[routine]]
    #             routine_df = pd.DataFrame(data=time_series, columns=['dt'])
    #             routine_dir = os.path.join(runtime_dir, f"time_series-{routine}.csv")
    #             routine_df.to_csv(routine_dir,index=False)

    #         stats_df = pd.DataFrame(data, columns=columns)
    #         # self.log(f'\nENVIRONMENT RUN-TIME STATS\n{str(stats_df)}\n', level=logging.WARNING)
    #         stats_df.to_csv(f"{self.results_path}/runtime_stats.csv", index=False)

    #         # print final time
    #         print('\n')
    #         self.log(f'successfully shutdown', level=logging.WARNING)
        
    #     except asyncio.CancelledError as e:
    #         raise e
    #     except Exception as e:
    #         print('\n','\n','\n')
    #         print(e.with_traceback())
    #         raise e        
            
    def compile_observations(self) -> pd.DataFrame:
        try:
            columns = None
            data = []
            
            for msg_dict in tqdm(self.observation_history, 
                            desc='Compiling observations results', 
                            leave=True):
                msg : ObservationResultsMessage = ObservationResultsMessage(**msg_dict)
                observation_data : List[dict] = msg.observation_data
                observer = msg.dst

                for obs in observation_data:
                    
                    # find column names 
                    if columns is None:
                        columns = sorted([key for key in obs])
                        columns.insert(0, 'observer')
                        # columns.insert(2, 't_img')
                        # columns.remove('t_start')
                        # columns.remove('t_end')

                    # add observation to data list
                    obs['observer'] = observer.lower()
                    for key in columns:
                        val = obs.get(key, None)
                        if isinstance(val, list):
                            obs[key] = val[0]
                            # if len(val) == 1:
                            #     obs[key] = val[0]
                            # else:
                            #     obs[key] = [val[0], val[-1]]

                    # obs['t_img'] = [obs['t_start'], obs['t_end']]
                    # obs.pop('t_start')
                    # obs.pop('t_end')

                    data.append([obs[key] for key in columns])

            observations_df = pd.DataFrame(data=data, columns=columns)
            observations_df = observations_df.sort_index(axis=1)

            return observations_df
        
        except Exception as e:
            print(e.with_traceback())
            raise e
    
    def compile_broadcasts(self) -> pd.DataFrame:
        columns = ['t_msg', 'sender', 'message type', 
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
        # convert measurement request dictionaries to Task Requests
        self.task_reqs : list[TaskRequest] = list({
            TaskRequest.from_dict(req_dict)
            for req_dict in self.task_reqs})

        columns = ['request id', 'requester', 'event id', 'parameter', 't_req', 'mission name']
        data = [[req.id,
                 req.requester,
                 req.task.event.id,
                 req.task.parameter,
                 req.t_req,
                 req.mission_name,
                 ] 
                 for req in self.task_reqs
                 if isinstance(req.task, EventObservationTask)]
        
        assert all(isinstance(req.task, EventObservationTask) for req in self.task_reqs), \
            'Only `EventObservationTask` measurement requests are currently supported in the results compilation.'

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
    
   