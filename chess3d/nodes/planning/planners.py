
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
                             orbitdata : OrbitData,
                             **_
                            ) -> list:
        """ 
        Schedules any broadcasts to be done. 
        
        By default it schedules the broadcast of any newly generated requests
        and the relay of any incoming relay messages
        """
        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Broadcast scheduling for agents of type `{type(state)}` not yet implemented.')
        elif orbitdata is None:
            raise ValueError(f'`orbitdata` required for agents of type `{type(state)}`.')

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
        path, t_start = self._create_broadcast_path(state, orbitdata)

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
    
    def _create_broadcast_path(self, 
                               state : SimulationAgentState, 
                               orbitdata : OrbitData) -> tuple:
        """ 
        Finds the best path for broadcasting a message to all agents using depth-first-search
        """
        # populate list of agents
        target_agents = [target_agent 
                         for target_agent in orbitdata.isl_data 
                         if target_agent != state.agent_name]
        
        # check if other agents exist in the simulation
        if not target_agents: 
            # no other agents in the simulation; no need for relays
            return ([], state.t)
        
        earliest_accesses = [orbitdata.get_next_agent_access(target_agent, state.t) 
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

                access_interval : TimeInterval = orbitdata.get_next_agent_access(receiver_agent, t_access)
                
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
                                clock_config : ClockConfig,
                                orbitdata : OrbitData = None
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

        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Maneuver scheduling for agents of type `{type(state)}` not yet implemented.')
        elif orbitdata is None:
            raise ValueError(f'`orbitdata` required for agents of type `{type(state)}`.')

        # initialize maneuver list
        maneuvers = []

        for i in range(len(observations)):
            action_sequence_i = []

            curr_observation : ObservationAction = observations[i]
            t_img = curr_observation.t_start
                        
            # estimate previous state
            if i == 0:
                t_prev = state.t
                prev_state : SatelliteAgentState = state.copy()
                
            else:
                prev_observation : ObservationAction = observations[i-1]
                t_prev = prev_observation.t_end if prev_observation is not None else state.t

                prev_state : SatelliteAgentState = state.propagate(t_prev)

                lat,lon,_ =  prev_observation.target
                main_instrument = prev_observation.instrument_name

                obs_prev = orbitdata.get_groundpoint_access_data(lat, lon, main_instrument, t_prev)
                th_f = obs_prev['look angle [deg]']
                
                prev_state.attitude = [th_f, 0.0, 0.0]

            # maneuver to point to target
            t_maneuver_end = None
            if isinstance(state, SatelliteAgentState):
                prev_state : SatelliteAgentState

                t_maneuver_start = prev_state.t
                
                lat,lon, _ =  curr_observation.target
                main_instrument = curr_observation.instrument_name

                obs_curr = orbitdata.get_groundpoint_access_data(lat, lon, main_instrument, t_img)
                th_f = obs_curr['look angle [deg]']

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
            t_move_end = t_img
            future_state : SatelliteAgentState = state.propagate(t_move_end)
            final_pos = future_state.pos
            
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
        
    def _print_observation_sequence(self, 
                                    state : SatelliteAgentState, 
                                    path : list, 
                                    orbitdata : OrbitData = None
                                    ) -> None :
        """ Debugging tool. Prints current observation sequence being considered. """

        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError('Observation sequence printouts for non-satellite agents not yet supported.')
        elif orbitdata is None:
            raise ValueError(f'`orbitdata` required for agents of type `{type(state)}`.')

        out = f'\n{state.agent_name}:\n\n\ntarget\tinstr\tt_img\tth\tdt_mmt\tdt_mvr\tValid?\n'

        out_temp = [f"N\A       ",
                    f"N\A",
                    f"\t{np.round(state.t,3)}",
                    f"\t{np.round(state.attitude[0],3)}",
                    f"\t-",
                    f"\t-",
                    f"\t-",
                    f"\n"
                    ]
        out += ''.join(out_temp)

        for i in range(len(path)):
            if i > 0:
                measurement_prev : ObservationAction = path[i-1]
                t_prev = measurement_i.t_end
                lat,lon,_ = measurement_prev.target
                obs_prev = orbitdata.get_groundpoint_access_data(lat, lon, measurement_prev.instrument_name, t_prev)
                th_prev = obs_prev['look angle [deg]']
            else:
                t_prev = state.t
                th_prev = state.attitude[0]

            measurement_i : ObservationAction = path[i]
            t_i = measurement_i.t_start
            lat,lon,alt = measurement_i.target
            obs_i = orbitdata.get_groundpoint_access_data(lat, lon, measurement_i.instrument_name, t_i)
            th_i = obs_i['look angle [deg]']

            dt_maneuver = abs(th_i - th_prev) / state.max_slew_rate
            dt_measurements = t_i - t_prev

            out_temp = [f"({round(lat,3)}, {round(lon,3)}, {round(alt,3)})",
                            f"  {measurement_i.instrument_name}",
                            f"\t{np.round(measurement_i.t_start,3)}",
                            f"\t{np.round(th_i,3)}",
                            f"\t{np.round(dt_measurements,3)}",
                            f"\t{np.round(dt_maneuver,3)}",
                            f"\t{dt_maneuver <= dt_measurements}",
                            f"\n"
                            ]
            out += ''.join(out_temp)
        out += f'\nn measurements: {len(path)}\n'

        print(out)

    @runtime_tracker
    def is_observation_path_valid(self, 
                                  state : SimulationAgentState, 
                                  observations : list,
                                  orbitdata : OrbitData = None
                                  ) -> bool:
        """ Checks if a given sequence of observations can be performed by a given agent """
        
        if isinstance(state, SatelliteAgentState):

            # check if every observation can be reached from the prior measurement
            for j in range(len(observations)):
                i = j - 1

                # check if there was an observation performed previously
                if i >= 0: # there was a prior observation performed

                    # estimate the state of the agent at the prior mesurement
                    observation_i : ObservationAction = observations[i]
                    lat_i,lon_i,_ =  observation_i.target
                    obs_i : dict = orbitdata.get_groundpoint_access_data(lat_i, lon_i, observation_i.instrument_name, observation_i.t_end)

                    th_i = obs_i.get('look angle [deg]', np.NAN)
                    t_i = observation_i.t_end

                else: # there was prior measurement

                    # use agent's current state as previous state
                    th_i = state.attitude[0]
                    t_i = state.t

                # estimate the state of the agent at the given measurement
                observation_j : ObservationAction = observations[j]
                lat_j,lon_j,_ =  observation_j.target
                obs_j : dict = orbitdata.get_groundpoint_access_data(lat_j, lon_j, observation_j.instrument_name, observation_i.t_end)

                th_j = obs_j.get('look angle [deg]', np.NAN)
                t_j = observation_j.t_start

                assert th_j != np.NAN and th_i != np.NAN # TODO: add case where the target is not visible by the agent at the desired time according to the precalculated orbitdata

                # estimate maneuver time betweem states
                dt_maneuver = abs(th_j - th_i) / state.max_slew_rate
                dt_measurements = t_j - t_i

                assert dt_measurements >= 0.0 and dt_maneuver >= 0.0

                # check if there's enough time to maneuver from one observation to another
                if dt_maneuver - dt_measurements >= 1e-9:
                    # there is not enough time to maneuver; flag current observation plan as unfeasible for rescheduling
                    return False
            
            # if all measurements passed the check; measurement path
            return True
        else:
            raise NotImplementedError(f'Measurement path validity check for agents with state type {type(state)} not yet implemented.')
        