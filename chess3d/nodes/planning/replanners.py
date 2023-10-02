from abc import ABC, abstractmethod
import math
import pandas as pd

from dmas.utils import runtime_tracker
from dmas.clocks import *

from nodes.orbitdata import OrbitData
from nodes.science.reqs import *
from nodes.states import *

class AbstractReplanner(ABC):
    """
    # Replanner    
    """
    def __init__(self) -> None:
        super().__init__()
        self.stats = {}

    @abstractmethod 
    def needs_replanning(   self, 
                            state : AbstractAgentState,
                            current_plan : list,
                            performed_actions : list,
                            incoming_reqs : list,
                            generated_reqs : list,
                            misc_messages : list,
                            t_plan : float,
                            planning_horizon : float,
                            orbitdata : OrbitData = None
                        ) -> bool:
        """
        Returns `True` if the current plan needs replanning
        """

    @abstractmethod
    def replan( self, 
                state : AbstractAgentState, 
                current_plan : list,
                performed_actions : list,
                incoming_reqs : list, 
                generated_reqs : list,
                misc_messages : list,
                t_plan : float,
                planning_horizon : float,
                clock_config : ClockConfig,
                orbitdata : OrbitData = None
            ) -> list:
        """
        Revises the current plan 
        """
        pass

    @runtime_tracker
    def _plan_from_path(    self, 
                            state : SimulationAgentState, 
                            path : list,
                            orbitdata : OrbitData,
                            t_init : float,
                            clock_config : ClockConfig
                    ) -> list:
        """
        Generates a list of AgentActions from the current path.

        Agents look to move to their designated measurement target and perform the measurement.

        ## Arguments:
            - state (:obj:`SimulationAgentState`): state of the agent at the start of the path
            - path (`list`): list of tuples indicating the sequence of observations to be performed and time of observation
        """
        plan = []

        # if no requests in the path, wait indefinitely
        if len(path) == 0:
            t_0 = t_init
            t_f = np.Inf
            return [WaitForMessages(t_0, t_f)]

        # create appropriate actions for every measurement in the path
        for i in range(len(path)):
            plan_i = []

            measurement_req, subtask_index, t_img, u_exp = path[i]
            measurement_req : MeasurementRequest; subtask_index : int; t_img : float; u_exp : float

            if not isinstance(measurement_req, GroundPointMeasurementRequest):
                raise NotImplementedError(f"Cannot create plan for requests of type {type(measurement_req)}")
            
            # Estimate previous state
            if i == 0:
                if isinstance(state, SatelliteAgentState):
                    t_prev = t_init
                    prev_state : SatelliteAgentState = state.copy()

                elif isinstance(state, UAVAgentState):
                    t_prev = t_init #TODO consider wait time for convergence
                    prev_state : UAVAgentState = state.copy()

                else:
                    raise NotImplementedError(f"cannot calculate travel time start for agent states of type {type(state)}")
            else:
                prev_req = None
                for action in reversed(plan):
                    action : AgentAction
                    if isinstance(action, MeasurementAction):
                        prev_req = MeasurementRequest.from_dict(action.measurement_req)
                        break
                
                action_prev : AgentAction = plan[-1]
                t_prev = action_prev.t_end

                if isinstance(state, SatelliteAgentState):
                    prev_state : SatelliteAgentState = state.propagate(t_prev)
                    
                    if prev_req is not None:
                        prev_state.attitude = [
                                            prev_state.calc_off_nadir_agle(prev_req),
                                            0.0,
                                            0.0
                                        ]

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
                th_f = prev_state.calc_off_nadir_agle(measurement_req)
                t_maneuver_end = t_maneuver_start + abs(th_f - prev_state.attitude[0]) / prev_state.max_slew_rate

                if abs(t_maneuver_start - t_maneuver_end) > 0.0:
                    maneuver_action = ManeuverAction([th_f, 0, 0], t_maneuver_start, t_maneuver_end)
                    plan_i.append(maneuver_action)   
                else:
                    t_maneuver_end = None

            # move to target
            t_move_start = t_prev if t_maneuver_end is None else t_maneuver_end
            if isinstance(state, SatelliteAgentState):
                lat, lon, _ = measurement_req.lat_lon_pos
                instrument = measurement_req.measurements[subtask_index]
                df : pd.DataFrame = orbitdata.get_ground_point_accesses_future(lat, lon, instrument, t_move_start)
                
                t_move_end = None
                for _, row in df.iterrows():
                    if row['time index'] * orbitdata.time_step >= t_img:
                        t_move_end = row['time index'] * orbitdata.time_step
                        break

                if t_move_end is None:
                    # unpheasible path
                    # self.log(f'Unheasible element in path. Cannot perform observation.', level=logging.DEBUG)
                    continue

                future_state : SatelliteAgentState = state.propagate(t_move_end)
                final_pos = future_state.pos

            elif isinstance(state, UAVAgentState):
                final_pos = measurement_req.pos
                dr = np.array(final_pos) - np.array(prev_state.pos)
                norm = np.sqrt( dr.dot(dr) )
                
                t_move_end = t_move_start + norm / state.max_speed

            else:
                raise NotImplementedError(f"cannot calculate travel time end for agent states of type {type(state)}")
            
            if t_move_end < t_img:
                plan_i.append( WaitForMessages(t_move_end, t_img) )
                
            t_img_start = t_img
            t_img_end = t_img_start + measurement_req.duration

            if isinstance(clock_config, FixedTimesStepClockConfig):
                dt = clock_config.dt
                if t_move_start < np.Inf:
                    t_move_start = dt * math.floor(t_move_start/dt)
                if t_move_end < np.Inf:
                    t_move_end = dt * math.ceil(t_move_end/dt)

                if t_img_start < np.Inf:
                    t_img_start = dt * math.floor(t_img_start/dt)
                if t_img_end < np.Inf:
                    t_img_end = dt * math.ceil((t_img_start + measurement_req.duration)/dt)
            
            if abs(t_move_start - t_move_end) >= 1e-3:
                move_action = TravelAction(final_pos, t_move_start, t_move_end)
                plan_i.append(move_action)
            
            # perform measurement
            main_measurement, _ = measurement_req.measurement_groups[subtask_index]
            measurement_action = MeasurementAction( 
                                                    measurement_req.to_dict(),
                                                    subtask_index, 
                                                    main_measurement,
                                                    u_exp,
                                                    t_img_start, 
                                                    t_img_end
                                                    )
            plan_i.append(measurement_action)  

            # TODO inform others of request completion

            plan.extend(plan_i)
        
        return plan

    @runtime_tracker
    def _get_available_requests( self, 
                                state : SimulationAgentState, 
                                requests : list,
                                orbitdata : OrbitData,
                                t_plan : float,
                                planning_horizon : float = np.Inf
                                ) -> list:
        """
        Checks if there are any requests available to be performed

        ### Returns:
            - list containing all available and bidable tasks to be performed by the parent agent
        """
        available = []
        for req in requests:
            req : MeasurementRequest
            for subtask_index in range(len(req.measurements)):
                if self.__can_bid(  state, 
                                    req, 
                                    subtask_index, 
                                    orbitdata, 
                                    t_plan,
                                    planning_horizon
                                ):
                    available.append((req, subtask_index))

        return available

    @runtime_tracker
    def __can_bid(self, 
                state : SimulationAgentState, 
                req : MeasurementRequest, 
                subtask_index : int, 
                orbitdata : OrbitData,
                t_plan : float,
                planning_horizon : float = np.Inf
                ) -> bool:
        """
        Checks if an agent has the ability to bid on a measurement task
        """
        # check planning horizon
        if state.t + planning_horizon < req.t_start:
            return False

        # check capabilities - TODO: Replace with knowledge graph
        main_measurement = req.measurements[subtask_index]
        if main_measurement not in [instrument for instrument in state.payload]:
            return False 

        # check time constraints
        ## Constraint 1: task must be able to be performed during or after the current time
        if req.t_end < state.t:
            return False

        elif isinstance(req, GroundPointMeasurementRequest):
            if isinstance(state, SatelliteAgentState):
                # check if agent can see the request location
                lat,lon,_ = req.lat_lon_pos
                instrument = req.measurements[subtask_index]
                df : pd.DataFrame = orbitdata.get_ground_point_accesses_future(lat, lon, instrument, req.t_start, req.t_end)
                
                if not df.empty:                
                    times = df.get('time index')
                    for time in times:
                        time *= orbitdata.time_step 

                        if state.t + planning_horizon < time:
                            break

                        return True
                
                return False
        
        return True

class FIFOReplanner(AbstractReplanner):
    def __init__(self):
        self.performed_requests = []
        self.access_times = {}
        self.known_reqs = []
        self.prev_path = None

    @runtime_tracker
    def needs_replanning(   self, 
                            state : SimulationAgentState,
                            current_plan : list,
                            performed_actions : list,
                            incoming_reqs : list,
                            generated_reqs : list,
                            misc_messages : list,
                            t_plan : float,
                            planning_horizon : float,
                            orbitdata : OrbitData = None
                        ) -> bool:
        
        # update list of known requests
        new_reqs : list = self._update_known_requests(  current_plan, 
                                                        incoming_reqs,
                                                        generated_reqs)
        
        # update list of performed measurements
        self.__update_performed_requests(performed_actions)

        self.__update_access_times( state, 
                                    new_reqs, 
                                    performed_actions,
                                    t_plan,
                                    planning_horizon,
                                    orbitdata)        

        # check if incoming or generated measurement requests are already accounted for
        _, unscheduled_reqs = self._compare_requests(current_plan)

        # replan if planning horizon has been reached or if there are requests to be scheduled
        if state.t >= t_plan + planning_horizon or len(unscheduled_reqs) > 0:            
            x = 1

        return state.t >= t_plan + planning_horizon or len(unscheduled_reqs) > 0
   
    @runtime_tracker
    def _update_known_requests( self, 
                                current_plan : list,
                                incoming_reqs : list,
                                generated_reqs : list
                                ) -> list:
        """
        Reads incoming requests and current plan to keep track of all known requests
        """
        ## list all requests in current plan
        scheduled_reqs = []
        for action in current_plan:
            if isinstance(action, MeasurementAction):
                req = MeasurementRequest.from_dict(action.measurement_req)
                if req not in scheduled_reqs:
                    scheduled_reqs.append(req)

        ## compare with incoming or generated requests
        new_reqs = []
        for req in incoming_reqs:
            req : MeasurementRequest
            if req not in scheduled_reqs and req.s_max > 0.0:
                new_reqs.append(req)
        for req in generated_reqs:
            if req not in scheduled_reqs and req.s_max > 0.0:
                new_reqs.append(req)

        scheduled_reqs = list(filter(lambda req : req not in self.known_reqs, scheduled_reqs))
        new_reqs = list(filter(lambda req : req not in self.known_reqs, new_reqs))
        new_reqs.extend(scheduled_reqs)

        # update intenal list of known requests
        self.known_reqs.extend(new_reqs)

        return new_reqs
    
    @runtime_tracker
    def __update_performed_requests(self, performed_actions : list) -> None:
        """ Updates an internal list of requests performed by the parent agent """
        for action in performed_actions:
            if isinstance(action, MeasurementAction):
                req : MeasurementRequest = MeasurementRequest.from_dict(action.measurement_req)
                if action.status == action.COMPLETED and req not in self.performed_requests:
                    self.performed_requests.append(req)

        return

    @runtime_tracker
    def __update_access_times(  self,
                                state : AbstractAgentState,
                                new_reqs : list,
                                performed_actions : list,  
                                t_plan : float,
                                planning_horizon : float,
                                orbitdata : OrbitData) -> None:
        """
        Calculates and saves the access times of all known requests
        """
        if state.t < t_plan + planning_horizon:
            # calculate new access times for new requests
            for req in new_reqs:
                if req.id not in self.access_times:
                    self.access_times[req.id] = [[] for _ in range(len(req.measurements))]
                    for subtask_index in range(len(req.measurements)):
                        self.access_times[req.id][subtask_index] = self._calc_arrival_times(   state, 
                                                                                                req,
                                                                                                subtask_index, 
                                                                                                t_plan, 
                                                                                                planning_horizon, 
                                                                                                orbitdata)
                            
            # update access times if a measurement was completed
            for action in performed_actions:
                if isinstance(action, MeasurementAction):
                    req : MeasurementRequest = MeasurementRequest.from_dict(action.measurement_req)
                    if action.status == action.COMPLETED:
                        self.access_times[req.id] = [[] for _ in range(len(req.measurements))]

            # update latest available access time for each known request
            for req_id in self.access_times:
                for t_arrivals in self.access_times[req_id]:
                    while len(t_arrivals) > 0 and t_arrivals[0] < state.t:
                        t_arrivals.pop(0)
        else:
            # recalculate access times for all known requests
            for req in self.known_reqs:
                self.access_times[req.id] = [[] for _ in range(len(req.measurements))]
                for subtask_index in range(len(req.measurements)):
                    self.access_times[req.id][subtask_index] = self._calc_arrival_times(    state, 
                                                                                            req, 
                                                                                            subtask_index,
                                                                                            state.t,
                                                                                            planning_horizon, 
                                                                                            orbitdata)

        # for req_id in self.access_times:
        #     print(req_id.split('-')[0], self.access_times[req_id])

        return
    
    def _calc_arrival_times(self, 
                            state : SimulationAgentState, 
                            req : MeasurementRequest, 
                            subtask_index : int,
                            t_prev : Union[int, float],
                            planning_horizon : Union[int, float], 
                            orbitdata : OrbitData) -> float:
        """
        Estimates the quickest arrival time from a starting position to a given final position
        """
        if isinstance(req, GroundPointMeasurementRequest):
            # compute earliest time to the task
            if isinstance(state, SatelliteAgentState):
                t_imgs = []
                lat,lon,_ = req.lat_lon_pos
                instrument = req.measurements[subtask_index]
                t_end = t_prev + planning_horizon
                df : pd.DataFrame = orbitdata.get_ground_point_accesses_future(lat, lon, instrument, t_prev, t_end)

                for _, row in df.iterrows():
                    t_img = row['time index'] * orbitdata.time_step
                    dt = t_img - state.t
                
                    # propagate state
                    propagated_state : SatelliteAgentState = state.propagate(t_img)

                    # compute off-nadir angle
                    thf = propagated_state.calc_off_nadir_agle(req)
                    dth = abs(thf - propagated_state.attitude[0])

                    # estimate arrival time using fixed angular rate TODO change to 
                    if dt >= dth / state.max_slew_rate: # TODO change maximum angular rate 
                        t_imgs.append(t_img)
                        
                return t_imgs

            elif isinstance(state, UAVAgentState):
                dr = np.array(req.pos) - np.array(state.pos)
                norm = np.sqrt( dr.dot(dr) )
                return [norm / state.max_speed + t_prev]

            else:
                raise NotImplementedError(f"arrival time estimation for agents of type `{type(state)}` is not yet supported.")

        else:
            raise NotImplementedError(f"cannot calculate imaging time for measurement requests of type {type(req)}")       

    def _compare_requests(  self, 
                            current_plan : list
                        ) -> tuple:
        """ Separates scheduled and unscheduled requests """
        
        ## list all requests in current plan
        scheduled_reqs = []
        for action in current_plan:
            if isinstance(action, MeasurementAction):
                req = MeasurementRequest.from_dict(action.measurement_req)
                if req not in scheduled_reqs:
                    scheduled_reqs.append(req)

        ## list all known available request that have not been scheduled yet
        def is_new_req(req : MeasurementRequest) -> bool:
            if req in scheduled_reqs:
                return False
            
            for t_arrivals in self.access_times[req.id]:
                if len(t_arrivals) > 0:
                    return True
                
            return False

        new_reqs = list(filter(is_new_req, self.known_reqs))
        
        return scheduled_reqs, new_reqs
    
    def replan( self, 
                state : AbstractAgentState, 
                current_plan : list,
                performed_actions : list,
                incoming_reqs : list, 
                generated_reqs : list,
                misc_messages : list,
                t_plan : float,
                planning_horizon : float,
                clock_config : ClockConfig,
                orbitdata : OrbitData = None
            ) -> list:
        
        # initialize plan
        path = []         
        
        # # compile requests
        # scheduled_reqs, unscheduled_reqs = self._compare_requests(current_plan)            
        # available_reqs : list = self._get_available_requests( state, self.known_reqs, orbitdata, t_plan, planning_horizon )

        if isinstance(state, SatelliteAgentState):
            # Generates a plan for observing GPs on a first-come first-served basis            
            path = []
            reqs = {req.id : req for req in self.known_reqs}

            for req_id in self.access_times:
                for subtask_index in range(len(self.access_times[req_id])):
                    t_arrivals : list = self.access_times[req_id][subtask_index]

                    if len(t_arrivals) > 0:
                        t_img = t_arrivals.pop(0)
                        req : MeasurementRequest = reqs[req_id]
                        path.append((req, subtask_index, t_img, req.s_max/len(req.measurements)))

            path.sort(key=lambda a: a[2])

            out = '\n'
            for req, subtask_index, t_img, s in path:
                out += f"{req.id.split('-')[0]}\t{subtask_index}\t{np.round(t_img,3)}\t{np.round(s,3)}\n"
            # print(out)
            x = 1

            while True:
                
                conflict_free = True
                for i in range(len(path) - 1):
                    j = i + 1
                    req_i, _, t_i, __ = path[i]
                    req_j, subtask_index_j, t_j, s_j = path[j]

                    th_i = state.calc_off_nadir_agle(req_i)
                    th_j = state.calc_off_nadir_agle(req_j)

                    if abs(th_i - th_j) / state.max_slew_rate > t_j - t_i:
                        t_arrivals : list = self.access_times[req_j.id][subtask_index_j]
                        if len(t_arrivals) > 0:
                            t_img = t_arrivals.pop(0)

                            path[j] = (req_j, subtask_index_j, t_img, s_j)
                            path.sort(key=lambda a: a[2])
                        else:
                            #TODO remove request from path
                            raise Exception("Whoops. See Plan Initializer.")
                            path.pop(j) 
                        conflict_free = False
                        break

                if conflict_free:
                    break
                    
        out = '\n'
        for req, subtask_index, t_img, s in path:
            out += f"{req.id.split('-')[0]}\t{subtask_index}\t{np.round(t_img,3)}\t{np.round(s,3)}\n"
        # print(out)

        # generate plan from path
        plan = self._plan_from_path(state, path, orbitdata, state.t, clock_config)

        # wait for next planning horizon 
        if len(plan) > 0 and plan[-1].t_end < state.t + planning_horizon:
            plan.append(WaitForMessages(plan[-1].t_end, state.t + planning_horizon))
        else:
            plan.append(WaitForMessages(state.t, state.t + planning_horizon))

        self.prev_path = [req for req, _, __, ___, in path]
            
        return plan
    