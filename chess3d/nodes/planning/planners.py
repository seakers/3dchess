
import math
import queue
from typing import Callable, Any

from nodes.planning.plan import Plan, Preplan
from nodes.orbitdata import OrbitData, TimeInterval
from nodes.states import *
from nodes.science.reqs import *
from messages import *
from dmas.modules import *
from dmas.utils import runtime_tracker
import pandas as pd


class AbstractPlanner(ABC):
    def __init__(self, 
                 utility_func : Callable[[], Any], 
                 orbitdata : OrbitData,
                 logger : logging.Logger = None
                 ) -> None:
        
        # initialize object
        super().__init__()

        # initialize attributes
        self.generated_reqs = []
        self.completed_requests = []
        self.completed_broadcasts = []
        self.completed_actions = []
        self.pending_relays = []
        self.access_times = {}
        self.known_reqs = []
        self.stats = {}
        self.plan : Plan = None

        # set attribute parameters
        self.utility_func = utility_func    # utility function
        self.orbitdata = orbitdata          # parent agent orbitdata
        self._logger = logger               # logger for debugging

    @abstractmethod
    def update_percepts(self, **kwargs) -> None:
        """ Updates internal knowledge based on incoming percepts """

    @abstractmethod
    def needs_planning(self, **kwargs) -> bool:
        """ Determines whether planning is triggered """ 
        
    @abstractmethod
    def generate_plan(self, **kwargs) -> Plan:
        """ Creates a plan for the agent to perform """

    def _schedule_broadcasts(self, 
                             state : SimulationAgentState, 
                             **_
                            ) -> list:
        """ 
        Schedules any broadcasts to be done. 
        
        By default it schedules the broadcast of any newly generated requests
        and the relay of any incoming relay messages
        """
        # initialize list of broadcasts to be done
        broadcasts = []       

        # schedule generated measurement request broadcasts
        ## check which requests have not been broadcasted yet
        requests_broadcasted = [msg.req['id'] for msg in self.completed_broadcasts 
                                if isinstance(msg, MeasurementRequestMessage)]
        requests_to_broadcast = [req for req in self.generated_reqs
                                 if isinstance(req, MeasurementRequest)
                                 and req.id not in requests_broadcasted]

        # Find best path for broadcasts
        path, t_start = self._create_broadcast_path(state)

        ## create a broadcast action for all unbroadcasted measurement requests
        for req in requests_to_broadcast:        
            # if found, create broadcast action
            if t_start >= 0:
                msg = MeasurementRequestMessage(state.agent_name, state.agent_name, req.to_dict(), path=path)
                broadcast_action = BroadcastMessageAction(msg.to_dict(), t_start)
                
                broadcasts.append(broadcast_action)

        # schedule message relay
        relay_broadcasts = [self._schedule_relay(relay) for relay in self.pending_relays]
        broadcasts.extend(relay_broadcasts)    
                        
        # return scheduled broadcasts
        return broadcasts   
    
    def _create_broadcast_path(self, state : SimulationAgentState) -> tuple:
        """ 
        Finds the best path for broadcasting a message to all agents using depth-first-search
        """
        # populate list of agents
        target_agents = [target_agent 
                         for target_agent in self.orbitdata.isl_data 
                         if target_agent != state.agent_name]
        
        # check if other agents exist in the simulation
        if not target_agents: 
            # no other agents in the simulation; no need for relays
            return ([], state.t)
        
        earliest_accesses = [self.orbitdata.get_next_agent_access(target_agent, state.t) 
                                for target_agent in target_agents]           
        same_access_start = [access.start == earliest_accesses[0].start 
                                for access in earliest_accesses 
                                if isinstance(access, TimeInterval)]
        same_access_end = [access.end == earliest_accesses[0].end 
                            for access in earliest_accesses 
                            if isinstance(access, TimeInterval)]

        # check if broadcast needs to be routed
        if all(same_access_start) and all(same_access_end):
            # all agents are accessing eachother at the same time; no need for mesasge relays
            return ([], state.t)   

        # initialize queue
        q = queue.Queue()
        
        # initialize min path and min path cost
        min_path = []
        min_times = []
        min_cost = np.Inf

        # add parent agent as the root node
        q.put((state.agent_name, [], [], 0.0))

        # iterate through depth-first search
        while not q.empty():
            # get next node in the search
            _, current_path, current_times, path_cost = q.get()

            # check if path is complete
            if len(target_agents) == len(current_path):
                # check if minimum cost
                if path_cost < min_cost:
                    min_cost = path_cost
                    min_path = [path_element for path_element in current_path]
                    min_times = [path_time for path_time in current_times]

            # add children nodes to queue
            for receiver_agent in [receiver_agent for receiver_agent in target_agents 
                                    if receiver_agent not in current_path
                                    and receiver_agent != state.agent_name
                                    ]:
                # query next access interval to children nodes
                t_access : float = state.t + path_cost

                sender_orbitdata : OrbitData = self.orbitdata
                access_interval : TimeInterval = sender_orbitdata.get_next_agent_access(receiver_agent, t_access)
                
                if access_interval.start < np.Inf:
                    new_path = [path_element for path_element in current_path]
                    new_path.append(receiver_agent)

                    new_cost = access_interval.start - state.t

                    new_times = [path_time for path_time in current_times]
                    new_times.append(new_cost + state.t)

                    q.put((receiver_agent, new_path, new_times, new_cost))

        # check if earliest broadcast time is valid
        if min_times: assert state.t <= min_times[0]

        # return path and broadcast start time
        return (min_path, min_times[0]) if min_path else ([], np.Inf)
    
    @runtime_tracker
    def _schedule_relay(self, relay_message : SimulationMessage) -> list:
        raise NotImplementedError('Relay scheduling not yet supported.')
        
        # check if relay message has a valid relay path
        assert relay.path

        # find next destination and access time
        next_dst = relay.path[0]
        
        # query next access interval to children nodes
        sender_orbitdata : OrbitData = orbitdata[state.agent_name]
        access_interval : TimeInterval = sender_orbitdata.get_next_agent_access(next_dst, state.t)
        t_start : float = access_interval.start

        if t_start < np.Inf:
            # if found, create broadcast action
            broadcast_action = BroadcastMessageAction(relay.to_dict(), t_start)
            
            # check broadcast start; only add to plan if it's within the planning horizon
            if t_start <= state.t + self.horizon:
                broadcasts.append(broadcast_action)


    @runtime_tracker
    def _schedule_maneuvers(    self, 
                                state : SimulationAgentState, 
                                observations : list,
                                clock_config : ClockConfig
                            ) -> list:
        """
        Generates a list of AgentActions from the current path.

        Agents look to move to their designated measurement target and perform the measurement.

        ## Arguments:
            - state (:obj:`SimulationAgentState`): state of the agent at the start of the path
            - path (`list`): list of tuples indicating the sequence of observations to be performed and time of observation
            - t_init (`float`): start time for plan
            - clock_config (:obj:`ClockConfig`): clock being used for this simulation
        """

        # initialize maneuver list
        maneuvers = []

        for i in range(len(observations)):
            action_sequence_i = []

            measurement_action : MeasurementAction = observations[i]
            measurement_req = MeasurementRequest.from_dict(measurement_action.measurement_req)
            t_img = measurement_action.t_start
            
            if not isinstance(measurement_req, GroundPointMeasurementRequest):
                raise NotImplementedError(f"Cannot create plan for requests of type {type(measurement_req)}")
            
            # Estimate previous state
            if i == 0:
                if isinstance(state, SatelliteAgentState):
                    t_prev = state.t
                    prev_state : SatelliteAgentState = state.copy()

                # elif isinstance(state, UAVAgentState):
                #     t_prev = state.t # TODO consider wait time for convergence
                #     prev_state : UAVAgentState = state.copy()

                else:
                    raise NotImplemented(f"maneuver scheduling for states of type `{type(state)}` not yet supported")
            else:
                prev_measurement : MeasurementAction = observations[i-1]
                prev_req = MeasurementRequest.from_dict(prev_measurement.measurement_req)
                t_prev = prev_measurement.t_end if prev_measurement is not None else state.t

                if isinstance(state, SatelliteAgentState):
                    prev_state : SatelliteAgentState = state.propagate(t_prev)

                    prev_req : GroundPointMeasurementRequest
                    lat,lon,_ =  prev_req.lat_lon_pos
                    main_instrument = prev_measurement.instrument_name

                    obs_prev = self.orbitdata.get_groundpoint_access_data(lat, lon, main_instrument, t_prev)
                    th_f = obs_prev['look angle [deg]']
                    
                    prev_state.attitude = [th_f, 0.0, 0.0]

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
                
                lat,lon, _ =  measurement_req.lat_lon_pos
                main_instrument = measurement_action.instrument_name

                obs_j = self.orbitdata.get_groundpoint_access_data(lat, lon, main_instrument, t_img)
                th_f = obs_j['look angle [deg]']

                dt = abs(th_f - prev_state.attitude[0]) / prev_state.max_slew_rate
                t_maneuver_end = t_maneuver_start + dt

                if abs(t_maneuver_start - t_maneuver_end) >= 1e-3:
                    action_sequence_i.append(ManeuverAction([th_f, 0, 0], 
                                                            t_maneuver_start, 
                                                            t_maneuver_end))   
                else:
                    t_maneuver_end = None

            # move to target
            t_move_start = t_prev if t_maneuver_end is None else t_maneuver_end
            if isinstance(state, SatelliteAgentState):
                t_move_end = t_img
                future_state : SatelliteAgentState = state.propagate(t_move_end)
                final_pos = future_state.pos

            elif isinstance(state, UAVAgentState):
                final_pos = measurement_req.pos
                dr = np.array(final_pos) - np.array(prev_state.pos)
                norm = np.sqrt( dr.dot(dr) )
                
                t_move_end = t_move_start + norm / state.max_speed

            else:
                raise NotImplementedError(f"cannot calculate travel time end for agent states of type {type(state)}")
            
            # quantize travel maneuver times if needed
            if isinstance(clock_config, FixedTimesStepClockConfig):
                dt = clock_config.dt
                if t_move_start < np.Inf:
                    t_move_start = dt * math.floor(t_move_start/dt)
                if t_move_end < np.Inf:
                    t_move_end = dt * math.ceil(t_move_end/dt)
            
            # add travel maneuver if required
            if abs(t_move_start - t_move_end) >= 1e-3:
                move_action = TravelAction(final_pos, t_move_start, t_move_end)
                action_sequence_i.append(move_action)
            
            # wait for measurement action to start
            if t_move_end < t_img:
                action_sequence_i.append( WaitForMessages(t_move_end, t_img) )

            maneuvers.extend(action_sequence_i)

        return maneuvers
        
    def _print_observation_path(self, state : SatelliteAgentState, path : list) -> None :
        """ Debugging tool. Prints current observations plan being considered. """

        out = f'\n{state.agent_name}:\n\n\nID\t  j\tt_img\tth\tdt_mmt\tdt_mvr\tValid\tu_exp\n'

        out_temp = [f"N\A       ",
                    f"N\A",
                    f"\t{np.round(state.t,3)}",
                    f"\t{np.round(state.attitude[0],3)}",
                    f"\t-",
                    f"\t-",
                    f"\t-"
                    f"\t{0.0}",
                    f"\n"
                    ]
        out += ''.join(out_temp)

        for i in range(len(path)):
            if i > 0:
                measurement_prev : MeasurementAction = path[i-1]
                t_prev = measurement_prev.t_end
                req_prev : MeasurementRequest = MeasurementRequest.from_dict(measurement_prev.measurement_req)
                state_prev : SatelliteAgentState = state.propagate(t_prev)
                th_prev = state_prev.calc_off_nadir_agle(req_prev)
            else:
                t_prev = state.t
                state_prev : SatelliteAgentState = state
                th_prev = state.attitude[0]

            measurement_i : MeasurementAction = path[i]
            t_i = measurement_i.t_start
            req_i : MeasurementRequest = MeasurementRequest.from_dict(measurement_i.measurement_req)
            state_i : SatelliteAgentState = state.propagate(measurement_i.t_start)
            th_i = state_i.calc_off_nadir_agle(req_i)

            dt_maneuver = abs(th_i - th_prev) / state.max_slew_rate
            dt_measurements = t_i - t_prev

            out_temp = [f"{req_i.id.split('-')[0]}",
                            f"  {measurement_i.subtask_index}",
                            f"\t{np.round(measurement_i.t_start,3)}",
                            f"\t{np.round(th_i,3)}",
                            f"\t{np.round(dt_measurements,3)}",
                            f"\t{np.round(dt_maneuver,3)}",
                            f"\t{dt_maneuver <= dt_measurements}"
                            f"\t{np.round(measurement_i.u_exp,3)}",
                            f"\n"
                            ]
            out += ''.join(out_temp)
        out += f'\nn measurements: {len(path)}\n'

        print(out)

    @runtime_tracker
    def is_observation_path_valid(self, state : SimulationAgentState, measurements : list) -> bool:
        """ 
        Checks if a given measurement or observation plan is valid given the type of agent performing them 
        """
        
        if isinstance(state, SatelliteAgentState):

            # check for 
            for j in range(len(measurements)):
                i = j - 1

                # estimate maneuver time 
                if i >= 0:
                    measurement_i : MeasurementAction = measurements[i]
                    req_i : GroundPointMeasurementRequest = MeasurementRequest.from_dict(measurement_i.measurement_req)
                    main_instrument_i = measurement_i.instrument_name
                    lat,lon,_ =  req_i.lat_lon_pos

                    obs_prev = self.orbitdata.get_groundpoint_access_data(lat, lon, main_instrument_i, measurement_i.t_end)
                    th_i = obs_prev['look angle [deg]']
                    t_i = measurement_i.t_end

                else:
                    th_i = state.attitude[0]
                    t_i = state.t

                measurement_j : MeasurementAction = measurements[j]
                req_j : GroundPointMeasurementRequest = MeasurementRequest.from_dict(measurement_j.measurement_req)
                main_instrument_j = measurement_j.instrument_name
                lat,lon,_ =  req_j.lat_lon_pos

                obs_prev = self.orbitdata.get_groundpoint_access_data(lat, lon, main_instrument_j, measurement_j.t_end)
                th_j = obs_prev['look angle [deg]']
                t_j = measurement_j.t_start

                dt_maneuver = abs(th_j - th_i) / state.max_slew_rate
                dt_measurements = t_j - t_i

                # check if there's enough time to maneuver from one observation to another
                if dt_maneuver - dt_measurements >= 1e-9:
                    # there is not enough time to maneuver; flag current observation plan as unfeasible for rescheduling
                    return False
        else:
            raise NotImplementedError(f'Measurement path validity check for agents with state type {type(state)} not yet implemented.')

        return True