
from logging import Logger
import queue
import random

from instrupy.base import BasicSensorModel
from instrupy.util import ViewGeometry, SphericalGeometry
from orbitpy.util import Spacecraft

from dmas.modules import *
from dmas.utils import runtime_tracker
from tqdm import tqdm

from chess3d.agents.planning.plan import Plan, Preplan
from chess3d.agents.orbitdata import OrbitData, TimeInterval
from chess3d.agents.planning.rewards import GridPoint, RewardGrid
from chess3d.agents.states import *
from chess3d.agents.science.requests import *
from chess3d.messages import *

class AbstractPlanner(ABC):
    """ 
    Describes a generic planner that, given a new set of percepts, decides whether to generate a new plan
    """
    def __init__(self, 
                 debug : bool = False,
                 logger : logging.Logger = None) -> None:
        # initialize object
        super().__init__()

        # check inputs
        if not isinstance(logger,logging.Logger) and logger is not None: 
            raise ValueError(f'`logger` must be of type `Logger`. Is of type `{type(logger)}`.')

        # initialize attributes
        self.known_reqs : set[MeasurementRequest] = set()                   # set of known measurement requests
        self.stats : dict = dict()                                          # collector for runtime performance statistics
        
        # set attribute parameters
        self._debug = debug                 # toggles debugging features
        self._logger = logger               # logger for debugging

    @abstractmethod
    def update_percepts( self,
                         state : SimulationAgentState,
                         incoming_reqs : list,
                         relay_messages : list,
                         completed_actions : list,
                         **kwargs
                        ) -> None:
        """ Updates internal knowledge based on incoming percepts """
        
        # check parameters
        assert all([isinstance(req, MeasurementRequest) for req in incoming_reqs])

        # update list of known requests
        self.known_reqs.update(incoming_reqs)
        
    @abstractmethod
    def needs_planning(self, **kwargs) -> bool:
        """ Determines whether planning is triggered """ 
        
    @abstractmethod
    def generate_plan(self, **kwargs) -> Plan:
        """ Creates a plan for the agent to perform """

    @abstractmethod
    def _schedule_broadcasts(self, 
                             state : SimulationAgentState, 
                             orbitdata : OrbitData,
                             **kwargs
                            ) -> list:
        """ 
        Schedules any broadcasts to be done. 
        
        By default it schedules any pending measurement requests or message relay messages.
        """
    
    @runtime_tracker
    def _create_broadcast_path(self, 
                               state : SimulationAgentState, 
                               orbitdata : OrbitData = None,
                               t_init : float = None
                               ) -> tuple:
        """ Finds the best path for broadcasting a message to all agents using depth-first-search 
        
        ### Arguments:
            - state (`SimulationAgentState`): current state of the agent
            - orbitdata (`OrbitData`): coverage data of agent if it is of type `SatelliteAgent`
            - t_init (`float`): ealiest desired broadcast time
        """
        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Broadcast routing path not yet supported for agents of type `{type(state)}`')
        
        # get earliest desired broadcast time 
        t_init = state.t if t_init is None or t_init < state.t else t_init

        # populate list of all agents except the parent agent
        target_agents = [target_agent 
                         for target_agent in orbitdata.isl_data 
                         if target_agent != state.agent_name]
        
        if not target_agents: 
            # no other agents in the simulation; no need for relays
            return ([], t_init)
        
        # check if broadcast needs to be routed
        earliest_accesses = [   orbitdata.get_next_agent_access(target_agent, t_init) 
                                for target_agent in target_agents]           
        
        same_access_start = [   abs(access.start - earliest_accesses[0].start) < 1e-3
                                for access in earliest_accesses 
                                if isinstance(access, TimeInterval)]
        same_access_end = [     abs(access.end - earliest_accesses[0].end) < 1e-3
                                for access in earliest_accesses 
                                if isinstance(access, TimeInterval)]

        if all(same_access_start) and all(same_access_end):
            # all agents are accessing eachother at the same time; no need for mesasge relays
            return ([], t_init)   

        # look for relay path using depth-first search

        # initialize queue
        q = queue.Queue()
        
        # initialize min path and min path cost
        min_path = []
        min_times = []
        min_cost = np.Inf

        # add parent agent as the root node
        q.put((state.agent_name, [], [], 0.0))

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
        next_dst = relay.path.pop(0)
        
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
                                specs : object,
                                observations : list,
                                clock_config : ClockConfig,
                                orbitdata : OrbitData = None
                            ) -> list:
        """
        Generates a list of AgentActions from the current path.

        Agents look to move to their designated measurement target and perform the measurement.

        ## Arguments:
            - state (:obj:`SimulationAgentState`): state of the agent at the start of the path
            - specs (`dict` or `Sapcecraft`): contains information regarding the physical specifications of the agent
            - path (`list`): list of tuples indicating the sequence of observations to be performed and time of observation
            - t_init (`float`): start time for plan
            - clock_config (:obj:`ClockConfig`): clock being used for this simulation
        """

        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Maneuver scheduling for agents of type `{type(state)}` not yet implemented.')
        elif not isinstance(specs, Spacecraft):
            raise ValueError(f'`specs` needs to be of type `Spacecraft` for agents of state type `{type(state)}`. Is of type `{type(specs)}`.')
        elif orbitdata is None:
            raise ValueError(f'`orbitdata` required for agents of type `{type(state)}`.')

        # compile instrument field of view specifications   
        cross_track_fovs = self.collect_fov_specs(specs)

        # get pointing agility specifications
        adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
        if adcs_specs is None: raise ValueError('ADCS component specifications missing from agent specs object.')

        max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
        if max_slew_rate is None: raise ValueError('ADCS `maxRate` specification missing from agent specs object.')

        max_torque = float(adcs_specs['maxTorque']) if adcs_specs.get('maxTorque', None) is not None else None
        if max_torque is None: raise ValueError('ADCS `maxTorque` specification missing from agent specs object.')

        # initialize maneuver list
        maneuvers : list[ManeuverAction] = []

        for i in tqdm(range(len(observations)), 
                      desc=f'{state.agent_name}-PLANNER: Scheduling Maneuvers', 
                      leave=False):

            curr_observation : ObservationAction = observations[i]

            # estimate previous state
            if i == 0:
                t_prev = state.t
                prev_state : SatelliteAgentState = state.copy()
                
            else:
                prev_observation : ObservationAction = observations[i-1]
                t_prev = prev_observation.t_end
                prev_state : SimulationAgentState = state.propagate(t_prev)
                prev_state.attitude = [prev_observation.look_angle, 0.0, 0.0]

            # maneuver to point to target
            if isinstance(state, SatelliteAgentState):
                prev_state : SatelliteAgentState
                
                dth_req = abs(curr_observation.look_angle - prev_state.attitude[0])
                dth_max = (curr_observation.t_start - prev_state.t) * max_slew_rate

                if dth_req > dth_max and abs(dth_req - dth_max) >= 1e-6: 
                    # maneuver impossible within timeframe
                    raise ValueError(f'Cannot schedule maneuver. Not enough time between observations')\
                
                # check if attitude maneuver is required
                if abs(dth_req) <= 1e-3: continue # already pointing in the same direction; ignore maneuver

                # calculate attitude duration    
                th_f = curr_observation.look_angle
                slew_rate = (curr_observation.look_angle - prev_state.attitude[0]) / dth_req * max_slew_rate
                dt = abs(th_f - prev_state.attitude[0]) / max_slew_rate

                # calculate maneuver time
                t_maneuver_start = curr_observation.t_start - dt
                t_maneuver_end = curr_observation.t_start

                # check if mnaeuver time is non-zero
                if abs(t_maneuver_start - t_maneuver_end) >= 1e-3:
                    # maneuver has non-zero duration; perform maneuver
                    maneuvers.append(ManeuverAction([th_f, 0, 0], 
                                                    [slew_rate, 0, 0],
                                                    t_maneuver_start, 
                                                    t_maneuver_end)) 

        maneuvers.sort(key=lambda a: a.t_start)

        assert self.is_maneuver_path_valid(state, specs, observations, maneuvers, max_slew_rate, cross_track_fovs)

        return maneuvers
    
    def is_maneuver_path_valid(self, 
                               state : SimulationAgentState, 
                               specs : object, 
                               observations : list, 
                               maneuvers : list,
                               max_slew_rate : float,
                               cross_track_fovs : dict
                               ) -> bool:

        for observation in observations:
            observation : ObservationAction

            # get fov for this observation's instrument
            cross_track_fov : float = cross_track_fovs[observation.instrument_name]

            # check if previous maneuvers were performed
            prev_maneuvers = [maneuver for maneuver in maneuvers
                              if maneuver.t_start <= observation.t_start]
            prev_maneuvers.sort(key=lambda a : a.t_start)

            if prev_maneuvers: # there was a maneuver performed before this observation
                # get latest maneuver
                latest_maneuver : ManeuverAction = prev_maneuvers.pop()

                # check status of completion of this maneuver
                if latest_maneuver.t_end < observation.t_start: # maneuver ended before observation started
                    # compare to final state after meneuver
                    dth = abs(observation.look_angle - latest_maneuver.final_attitude[0])

                else: # maneuver was being performed during meneuver
                    if prev_maneuvers:
                        prev_maneuver : ManeuverAction = prev_maneuvers.pop()
                        th_0 = prev_maneuver.final_attitude[0]
                    else:
                        th_0 = state.attitude[0]

                    dth = abs(observation.look_angle - th_0) - max_slew_rate * (observation.t_start - latest_maneuver.t_start) 

            else: # there were no maneuvers performed before this observation
                # compare to initial state
                dth = abs(observation.look_angle - state.attitude[0])

            if dth > cross_track_fov / 2.0 and abs(dth - cross_track_fov / 2.0) >= 1e-6:
                # latest state does not point towards the target at the intended look angle
                return False

        # all maneuvers passed checks; path is valid        
        return True
    
    def collect_fov_specs(self, specs : Spacecraft) -> dict:
        # compile instrument field of view specifications   
        cross_track_fovs = {instrument.name: np.NAN for instrument in specs.instrument}
        for instrument in specs.instrument:
            cross_track_fov = []
            for instrument_model in instrument.mode:
                if isinstance(instrument_model, BasicSensorModel):
                    instrument_fov : ViewGeometry = instrument_model.get_field_of_view()
                    instrument_fov_geometry : SphericalGeometry = instrument_fov.sph_geom
                    if instrument_fov_geometry.shape == 'RECTANGULAR':
                        cross_track_fov.append(instrument_fov_geometry.angle_width)
                    else:
                        raise NotImplementedError(f'Extraction of FOV for instruments with view geometry of shape `{instrument_fov_geometry.shape}` not yet implemented.')
                else:
                    raise NotImplementedError(f'measurement data query not yet suported for sensor models of type {type(instrument_model)}.')
            cross_track_fovs[instrument.name] = max(cross_track_fov)

        return cross_track_fovs

    def collect_agility_specs(self, specs : Spacecraft) -> tuple:
        adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
        if adcs_specs is None: raise ValueError('ADCS component specifications missing from agent specs object.')

        max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
        if max_slew_rate is None: raise ValueError('ADCS `maxRate` specification missing from agent specs object.')

        max_torque = float(adcs_specs['maxTorque']) if adcs_specs.get('maxTorque', None) is not None else None
        if max_torque is None: raise ValueError('ADCS `maxTorque` specification missing from agent specs object.')

        return max_slew_rate, max_torque
        
    @runtime_tracker
    def is_observation_path_valid(self, 
                                  state : SimulationAgentState, 
                                  specs : object,
                                  observations : list
                                  ) -> bool:
        """ Checks if a given sequence of observations can be performed by a given agent """

        if isinstance(state, SatelliteAgentState) and isinstance(specs, Spacecraft):

            # get pointing agility specifications
            adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
            if adcs_specs is None: raise ValueError('ADCS component specifications missing from agent specs object.')

            max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
            if max_slew_rate is None: raise ValueError('ADCS `maxRate` specification missing from agent specs object.')

            max_torque = float(adcs_specs['maxTorque']) if adcs_specs.get('maxTorque', None) is not None else None
            if max_torque is None: raise ValueError('ADCS `maxTorque` specification missing from agent specs object.')
            
            # check if every observation can be reached from the prior measurement
            for j in range(len(observations)):

                # estimate the state of the agent at the given measurement
                observation_j : ObservationAction = observations[j]
                th_j = observation_j.look_angle
                t_j = observation_j.t_start
                # fov = cross_track_fovs[observation_j.instrument_name]

                # compare to prior measurements
                
                if j > 0: # there was a prior observation performed

                    # estimate the state of the agent at the prior mesurement
                    observation_i : ObservationAction = observations[j-1]
                    th_i = observation_i.look_angle
                    t_i = observation_i.t_end

                else: # there was prior measurement

                    # use agent's current state as previous state
                    th_i = state.attitude[0]
                    t_i = state.t                

                # check if desired instrument is contained within the satellite's specifications
                if observation_j.instrument_name not in [instrument.name for instrument in specs.instrument]:
                    return False 
                
                assert th_j != np.NAN and th_i != np.NAN # TODO: add case where the target is not visible by the agent at the desired time according to the precalculated orbitdata

                # estimate maneuver time betweem states
                dt_maneuver = abs(th_j - th_i) / max_slew_rate

                # calculate time between measuremnets
                dt_measurements = t_j - t_i

                # check if observation sequence is correct 
                if dt_measurements < 0.0:
                    return False

                # Slewing constraint: check if there's enough time to maneuver from one observation to another
                if dt_maneuver > dt_measurements and abs(dt_maneuver - dt_measurements) > 1e-6:
                    # there is not enough time to maneuver; flag current observation plan as unfeasible for rescheduling
                    return False              
                
                # Torque constraint:
                # TODO check if the agent has enough torque in its reaction wheels to perform the maneuver
                            
            # if all measurements passed the check; observation path is valid
            return True
        else:
            raise NotImplementedError(f'Observation path validity check for agents with state type {type(state)} not yet implemented.')
        
    # def _print_observation_sequence(self, 
    #                                 state : SatelliteAgentState, 
    #                                 path : list, 
    #                                 orbitdata : OrbitData = None
    #                                 ) -> None :
    #     """ Debugging tool. Prints current observation sequence being considered. """

    #     if not isinstance(state, SatelliteAgentState):
    #         raise NotImplementedError('Observation sequence printouts for non-satellite agents not yet supported.')
    #     elif orbitdata is None:
    #         raise ValueError(f'`orbitdata` required for agents of type `{type(state)}`.')

    #     out = f'\n{state.agent_name}:\n\n\ntarget\tinstr\tt_img\tth\tdt_mmt\tdt_mvr\tValid?\n'

    #     out_temp = [f"N\A       ",
    #                 f"N\A",
    #                 f"\t{np.round(state.t,3)}",
    #                 f"\t{np.round(state.attitude[0],3)}",
    #                 f"\t-",
    #                 f"\t-",
    #                 f"\t-",
    #                 f"\n"
    #                 ]
    #     out += ''.join(out_temp)

    #     for i in range(len(path)):
    #         if i > 0:
    #             measurement_prev : ObservationAction = path[i-1]
    #             t_prev = measurement_i.t_end
    #             lat,lon,_ = measurement_prev.target
    #             obs_prev = orbitdata.get_groundpoint_access_data(lat, lon, measurement_prev.instrument_name, t_prev)
    #             th_prev = obs_prev['look angle [deg]']
    #         else:
    #             t_prev = state.t
    #             th_prev = state.attitude[0]

    #         measurement_i : ObservationAction = path[i]
    #         t_i = measurement_i.t_start
    #         lat,lon,alt = measurement_i.target
    #         obs_i = orbitdata.get_groundpoint_access_data(lat, lon, measurement_i.instrument_name, t_i)
    #         th_i = obs_i['look angle [deg]']

    #         dt_maneuver = abs(th_i - th_prev) / state.max_slew_rate
    #         dt_measurements = t_i - t_prev

    #         out_temp = [f"({round(lat,3)}, {round(lon,3)}, {round(alt,3)})",
    #                         f"  {measurement_i.instrument_name}",
    #                         f"\t{np.round(measurement_i.t_start,3)}",
    #                         f"\t{np.round(th_i,3)}",
    #                         f"\t{np.round(dt_measurements,3)}",
    #                         f"\t{np.round(dt_maneuver,3)}",
    #                         f"\t{dt_maneuver <= dt_measurements}",
    #                         f"\n"
    #                         ]
    #         out += ''.join(out_temp)
    #     out += f'\nn measurements: {len(path)}\n'

    #     print(out)

class AbstractPreplanner(AbstractPlanner):
    """
    # Preplanner

    Conducts operations planning for an agent at the beginning of a planning period. 
    """
    def __init__(   self, 
                    horizon : float = np.Inf,
                    period : float = np.Inf,
                    # sharing : bool = False,
                    debug : bool = False,
                    logger: logging.Logger = None
                ) -> None:
        """
        ## Preplanner 
        
        Creates an instance of a preplanner class object.

        #### Arguments:
            - horizon (`float`) : planning horizon in seconds [s]
            - period (`float`) : period of replanning in seconds [s]
            - logger (`logging.Logger`) : debugging logger
        """
        # initialize planner
        super().__init__(debug, logger)    

        # set parameters
        self.horizon = horizon                                                      # planning horizon
        self.period = period                                                        # replanning period         
        # self.sharing = sharing                                                      # toggle for sharing plans
        self.plan = Preplan(t=-1,horizon=horizon,t_next=0.0)                        # initialized empty plan
                
        # initialize attributes
        self.pending_reqs_to_broadcast : set[MeasurementRequest] = set()            # set of observation requests that have not been broadcasted
        
    @runtime_tracker
    def update_percepts(self, 
                        state : SimulationAgentState,
                        current_plan : Plan,
                        incoming_reqs: list, 
                        relay_messages: list, 
                        misc_messages : list,
                        completed_actions: list,
                        aborted_actions : list,
                        pending_actions : list
                        ) -> None:
        # update percepts
        super().update_percepts(state, incoming_reqs, relay_messages, completed_actions)

        # update completion of current plan
        # self.plan.update_action_completion(completed_actions, aborted_actions, pending_actions, state.t)

        # # update list of pending observation requests to broadcasts
        # if self.sharing:
        #     # collect list of measurement requests newly generated by the parent agent
        #     my_reqs = {req 
        #                 for req in incoming_reqs
        #                 if isinstance(req, MeasurementRequest)
        #                 and req.requester == state.agent_name}

        #     # update list of pending requests to broadcast
        #     self.pending_reqs_to_broadcast.update(my_reqs)

        #     # collect list of completed broadcasts
        #     completed_broadcasts = {message_from_dict(**action.msg) 
        #                         for action in completed_actions
        #                         if isinstance(action, BroadcastMessageAction)}
            
        #     # collect list of completed request broadcasts
        #     broadcasted_reqs = {MeasurementRequest.from_dict(msg.req) 
        #                         for msg in completed_broadcasts
        #                         if isinstance(msg, MeasurementRequestMessage)}
            
        #     # remove from list of pending requests to breadcasts if they've been broadcasted already
        #     self.pending_reqs_to_broadcast.difference_update(broadcasted_reqs)
    
    @runtime_tracker
    def needs_planning( self, 
                        state : SimulationAgentState,
                        __ : object,
                        current_plan : Plan
                        ) -> bool:
        """ Determines whether a new plan needs to be initalized """    

        if (current_plan.t < 0                  # simulation just started
            or state.t >= current_plan.t_next):    # or periodic planning period has been reached
            
            pending_actions = [action for action in current_plan
                               if action.t_start <= current_plan.t_next]
            
            return not bool(pending_actions)     # no actions left to do before the end of the replanning period 
        return False

    @runtime_tracker
    def generate_plan(  self, 
                        state : SimulationAgentState,
                        specs : object,
                        reward_grid : RewardGrid,
                        clock_config : ClockConfig,
                        orbitdata : OrbitData,
                    ) -> Plan:
        
        # schedule observations
        observations : list = self._schedule_observations(state, specs, reward_grid, clock_config, orbitdata)
        assert self.is_observation_path_valid(state, specs, observations)

        # schedule broadcasts to be perfomed
        broadcasts : list = self._schedule_broadcasts(state, observations, orbitdata)

        # generate maneuver and travel actions from measurements
        maneuvers : list = self._schedule_maneuvers(state, specs, observations, clock_config, orbitdata)
        
        # generate plan from actions
        self.plan : Preplan = Preplan(observations, maneuvers, broadcasts, t=state.t, horizon=self.horizon, t_next=state.t+self.period)    
        
        # wait for next planning period to start
        replan : list = self._schedule_periodic_replan(state, self.plan, state.t+self.period)
        self.plan.add_all(replan, t=state.t)

        # return plan and save local copy
        return self.plan.copy()
        
    @abstractmethod
    def _schedule_observations(self, state : SimulationAgentState, specs : object, reward_grid : RewardGrid, clock_config : ClockConfig, orbitdata : OrbitData = None) -> list:
        """ Creates a list of observation actions to be performed by the agent """    

    @abstractmethod
    def _schedule_broadcasts(self, state: SimulationAgentState, observations : list, orbitdata: OrbitData, t : float = None) -> list:
        """ Schedules broadcasts to be done by this agent """
        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Broadcast scheduling for agents of type `{type(state)}` not yet implemented.')
        elif orbitdata is None:
            raise ValueError(f'`orbitdata` required for agents of type `{type(state)}`.')

        # initialize list of broadcasts to be done
        broadcasts = []       

        # # schedule generated measurement request broadcasts
        # if self.sharing and self.pending_reqs_to_broadcast:
        #     # sort requests based on their start time
        #     pending_reqs_to_broadcast : list[MeasurementRequest] = list(self.pending_reqs_to_broadcast) 
        #     pending_reqs_to_broadcast.sort(key=lambda a : a.t_start)

        #     # find best path for broadcasts at the current time
        #     t = state.t if t is None else t
        #     path, t_start = self._create_broadcast_path(state, orbitdata, t)

        #     for req in tqdm(pending_reqs_to_broadcast,
        #                     desc=f'{state.agent_name}-PLANNER: Scheduling Measurement Request Broadcasts', 
        #                     leave=False):
                
        #         # calculate broadcast start time
        #         if req.t_start > t_start:
        #             path, t_start = self._create_broadcast_path(state, orbitdata, req.t_start)
                    
        #         # check broadcast feasibility
        #         if t_start < 0: continue

        #         # create broadcast action
        #         msg = MeasurementRequestMessage(state.agent_name, state.agent_name, req.to_dict(), path=path)
        #         broadcast_action = BroadcastMessageAction(msg.to_dict(), t_start)
                
        #         # add to list of broadcasts
        #         broadcasts.append(broadcast_action) 
                        
        # return scheduled broadcasts
        return broadcasts 

    @runtime_tracker
    def _schedule_periodic_replan(self, state : SimulationAgentState, prelim_plan : Plan, t_next : float) -> list:
        """ Creates and schedules a waitForMessage action such that it triggers a periodic replan """

        # find wait start time
        if prelim_plan.is_empty():
            t_wait_start = state.t 
        
        else:
            actions_within_period = [action for action in prelim_plan 
                                 if  isinstance(action, AgentAction)
                                 and action.t_start < t_next]

            if actions_within_period:
                # last_action : AgentAction = actions_within_period.pop()
                t_wait_start = min(max([action.t_end for action in actions_within_period]), t_next)
                                
            else:
                t_wait_start = state.t

        # create wait action
        return [WaitForMessages(t_wait_start, t_next)] if t_wait_start < t_next else []
    
    @runtime_tracker
    def get_ground_points(self,
                          orbitdata : OrbitData,
                          max_points : int = np.Inf,
                        ) -> dict:
        # initiate accestimes 
        all_ground_points = list({
            (grid_index, gp_index, lat, lon)
            for grid_datum in orbitdata.grid_data
            for lat, lon, grid_index, gp_index in grid_datum.values
        })
        
        # sample required number of points to consider
        ground_point_set : list = random.sample(all_ground_points, max_points) if max_points < np.Inf else all_ground_points
    
        # organize into a `dict`
        ground_points = dict()
        for grid_index, gp_index, lat, lon in ground_point_set: 
            if grid_index not in ground_points: ground_points[grid_index] = dict()
            if gp_index not in ground_points[grid_index]: ground_points[grid_index][gp_index] = dict()

            ground_points[grid_index][gp_index] = (lat,lon)

        # return grid information
        return ground_points

    @runtime_tracker
    def calculate_access_opportunities(self, 
                                       state : SimulationAgentState, 
                                       specs : Spacecraft,
                                       ground_points : dict,
                                       orbitdata : OrbitData
                                    ) -> dict:
        # define planning horizon
        t_start = state.t
        t_end = self.plan.t_next+self.horizon
        t_index_start = t_start / orbitdata.time_step
        t_index_end = t_end / orbitdata.time_step

        # compile coverage data
        orbitdata_columns : list = list(orbitdata.gp_access_data.columns.values)
        raw_coverage_data = [(t_index*orbitdata.time_step, *_)
                             for t_index, *_ in orbitdata.gp_access_data.values
                             if t_index_start <= t_index <= t_index_end]
        raw_coverage_data.sort(key=lambda a : a[0])

        # initiate accestimes 
        access_opportunities = {}
        
        for data in tqdm(raw_coverage_data, 
                         desc='PREPLANNER: Compiling access opportunities', 
                         leave=False):
            t_img = data[orbitdata_columns.index('time index')]
            grid_index = data[orbitdata_columns.index('grid index')]
            gp_index = data[orbitdata_columns.index('GP index')]
            instrument = data[orbitdata_columns.index('instrument')]
            look_angle = data[orbitdata_columns.index('look angle [deg]')]

            # only consider ground points from the pedefined list of important groundopints
            if grid_index not in ground_points or gp_index not in ground_points[grid_index]:
                continue
            
            # initialize dictionaries if needed
            if grid_index not in access_opportunities:
                access_opportunities[grid_index] = {}
                
            if gp_index not in access_opportunities[grid_index]:
                access_opportunities[grid_index][gp_index] = {instr.name : [] 
                                                        for instr in specs.instrument}

            # compile time interval information 
            found = False
            for interval, t, th in access_opportunities[grid_index][gp_index][instrument]:
                interval : TimeInterval
                t : list
                th : list

                if (   interval.is_during(t_img - orbitdata.time_step) 
                    or interval.is_during(t_img + orbitdata.time_step)):
                    interval.extend(t_img)
                    t.append(t_img)
                    th.append(look_angle)
                    found = True
                    break                        

            if not found:
                access_opportunities[grid_index][gp_index][instrument].append([TimeInterval(t_img, t_img), [t_img], [look_angle]])

        # convert to `list`
        access_opportunities = [    (grid_index, gp_index, instrument, interval, t, th)
                                    for grid_index in access_opportunities
                                    for gp_index in access_opportunities[grid_index]
                                    for instrument in access_opportunities[grid_index][gp_index]
                                    for interval, t, th in access_opportunities[grid_index][gp_index][instrument]
                                ]
                
        # return access times and grid information
        return access_opportunities

class AbstractReplanner(AbstractPlanner):
    """ Repairs plans previously constructed by another planner """

    def __init__(self, debug: bool = False, logger: Logger = None) -> None:
        super().__init__(debug, logger)

        self.preplan : Preplan = None

    @abstractmethod
    def update_percepts(self, 
                        state : SimulationAgentState,
                        current_plan : Plan,
                        incoming_reqs: list, 
                        relay_messages: list, 
                        misc_messages : list,
                        completed_actions: list,
                        aborted_actions : list,
                        pending_actions : list
                        ) -> None:
        
        super().update_percepts(state, incoming_reqs, relay_messages, completed_actions)
        
        # update latest preplan
        if abs(state.t - current_plan.t) <= 1e-3 and isinstance(current_plan, Preplan): 
            self.preplan : Preplan = current_plan.copy() 

    @abstractmethod
    def generate_plan(  self, 
                        state : SimulationAgentState,
                        specs : object,
                        reward_grid : RewardGrid,
                        current_plan : Plan,
                        clock_config : ClockConfig,
                        orbitdata : OrbitData,
                    ) -> Plan:
        pass

    def get_broadcast_contents(self,
                               broadcast_action : FutureBroadcastMessageAction,
                               state : SimulationAgentState,
                               reward_grid : RewardGrid,
                               **kwargs
                               ) -> BroadcastMessageAction:
        """  Generates a broadcast message to be sent to other agents """
        if broadcast_action.broadcast_type == FutureBroadcastTypes.REWARD:
            latest_observations : list[ObservationAction] = []
            for grid in reward_grid.rewards:
                for target in grid:
                    for instrument,grid_point in target.items():
                        grid_point : GridPoint
                        
                        # collect latest known observation for each ground point
                        if grid_point.observations:
                            observations : list[dict] = list(grid_point.observations)
                            observations.sort(key= lambda a: a['t_img'])
                            latest_observations.append((instrument, observations[-1]))

            instruments_used : set = {instrument for instrument,_ in latest_observations}

            msgs = [ObservationResultsMessage(state.agent_name, 
                                              state.agent_name, 
                                              state.to_dict(), 
                                              {}, 
                                              instrument_used,
                                              [observation_data
                                                for instrument, observation_data in latest_observations
                                                if instrument == instrument_used]
                                              )
                    for instrument_used in instruments_used]
            
        elif broadcast_action.broadcast_type == FutureBroadcastTypes.REQUESTS:
            msgs = [MeasurementRequestMessage(state.agent_name, state.agent_name, req.to_dict())
                    for req in self.known_reqs
                    if isinstance(req, MeasurementRequest)
                    and req.t_start <= state.t <= req.t_end]
        else:
            raise ValueError(f'`{broadcast_action.broadcast_type}` broadcast type not supported.')

        # construct bus message
        bus_msg = BusMessage(state.agent_name, state.agent_name, [msg.to_dict() for msg in msgs])

        # return bus message broadcast (if not empty)
        return BroadcastMessageAction(bus_msg.to_dict(), broadcast_action.t_start)