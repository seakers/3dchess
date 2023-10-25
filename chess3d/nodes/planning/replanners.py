from abc import ABC, abstractmethod
import logging
import math
from typing import Any, Callable
import pandas as pd

from dmas.utils import runtime_tracker
from dmas.clocks import *

from nodes.orbitdata import OrbitData
from nodes.science.reqs import *
from nodes.science.utility import synergy_factor
from nodes.states import *

class AbstractReplanner(ABC):
    """
    # Replanner    
    """
    def __init__(   self, 
                    utility_func : Callable[[], Any], 
                    logger: logging.Logger = None
                ) -> None:
        super().__init__()

        self.t_plan = -1
        self.t_next = np.Inf

        self.performed_requests = []
        self.access_times = {}
        self.known_reqs = []
        
        self.stats = {}

        self.utility_func = utility_func
        self._logger = logger

    @abstractmethod 
    def needs_replanning(   self, 
                            state : SimulationAgentState,
                            current_plan : list,
                            performed_actions : list,
                            incoming_reqs : list,
                            generated_reqs : list,
                            misc_messages : list,
                            t_plan : float,
                            t_next : float,
                            planning_horizon = np.Inf,
                            orbitdata : OrbitData = None
                        ) -> bool:
        """
        Returns `True` if the current plan needs replanning
        """

    @abstractmethod
    def replan( self, 
                state : SimulationAgentState, 
                current_plan : list,
                performed_actions : list,
                incoming_reqs : list, 
                generated_reqs : list,
                misc_messages : list,
                t_plan : float,
                t_next : float,
                clock_config : ClockConfig,
                orbitdata : OrbitData = None
            ) -> list:
        """
        Revises the current plan 
        """
        pass

    @abstractmethod
    def _get_available_requests(self, *args, **kwargs) -> list:
        """ Returns a list of known requests that can be performed within the current planning horizon """
        pass

    @runtime_tracker
    def _plan_from_path(    self, 
                            state : SimulationAgentState, 
                            path : list,
                            t_init : float,
                            clock_config : ClockConfig,
                            dt_wait : float = 0.0
                    ) -> list:
        """
        Generates a list of AgentActions from the current path.

        Agents look to move to their designated measurement target and perform the measurement.

        ## Arguments:
            - state (:obj:`SimulationAgentState`): state of the agent at the start of the path
            - path (`list`): list of tuples indicating the sequence of observations to be performed and time of observation
        """
        # create appropriate actions for every measurement in the path
        plan = []

        for i in range(len(path)):
            plan_i = []

            measurement_req, subtask_index, t_img, u_exp = path[i]
            measurement_req : MeasurementRequest; subtask_index : int; t_img : float; u_exp : float

            if not isinstance(measurement_req, GroundPointMeasurementRequest):
                raise NotImplementedError(f"Cannot create plan for requests of type {type(measurement_req)}")
            
            # Estimate previous state
            if i == 0:
                if isinstance(state, SatelliteAgentState):
                    if dt_wait >= 1e-3:
                        t_prev = state.t + dt_wait
                        prev_state : SatelliteAgentState = state.propagate(t_prev)
                        plan_i.append(WaitForMessages(state.t, t_prev))
                    else:
                        t_prev = state.t
                        prev_state : SatelliteAgentState = state.copy()

                # elif isinstance(state, UAVAgentState):
                #     t_prev = state.t #TODO consider wait time for convergence
                #     prev_state : UAVAgentState = state.copy()

                else:
                    raise NotImplementedError(f"cannot calculate travel time start for agent states of type {type(state)}")
            else:
                prev_req = None
                for action in reversed(plan):
                    action : AgentAction
                    if isinstance(action, MeasurementAction):
                        prev_req = MeasurementRequest.from_dict(action.measurement_req)
                        break
                
                action_prev : AgentAction = plan[-1] if len(plan) > 0 else None
                t_prev = action_prev.t_end if action_prev is not None else t_init

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
                dt = abs(th_f - prev_state.attitude[0]) / prev_state.max_slew_rate
                t_maneuver_end = t_maneuver_start + dt

                if abs(t_maneuver_start - t_maneuver_end) >= 1e-3:
                    maneuver_action = ManeuverAction([th_f, 0, 0], t_maneuver_start, t_maneuver_end)
                    plan_i.append(maneuver_action)   
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
                if t_move_start > t_move_end:
                    continue

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
    def _update_known_requests( self, 
                                current_plan : list,
                                incoming_reqs : list,
                                generated_reqs : list
                                ) -> list:
        """
        Reads incoming requests and current plan to keep track of all known requests
        """
        ## list all requests in current plan)
        measurement_actions = [action for action in current_plan if isinstance(action, MeasurementAction)]
        scheduled_reqs = []
        for action in measurement_actions:
            action : MeasurementAction
            if action not in scheduled_reqs:
                scheduled_reqs.append(MeasurementRequest.from_dict(action.measurement_req)) 
        new_scheduled_reqs = [req for req in scheduled_reqs if req not in self.known_reqs]
        
        # update intenal list of known requests with scheduled requests
        self.known_reqs.extend(new_scheduled_reqs)

        ## compare with incoming or generated requests
        new_reqs = []
        new_incoming_reqs = [req for req in incoming_reqs if    req not in self.known_reqs and
                                                                req not in new_reqs and 
                                                                req.s_max > 0.0]
        new_reqs.extend(new_incoming_reqs)
        new_generated_reqs = [req for req in generated_reqs if  req not in self.known_reqs and
                                                                req not in new_reqs and 
                                                                req.s_max > 0.0]
        new_reqs.extend(new_generated_reqs)

        # update intenal list of known requests with new requests
        self.known_reqs.extend(new_reqs)

        return new_reqs

    @runtime_tracker
    def _update_access_times(  self,
                                state : SimulationAgentState,
                                new_reqs : list,
                                performed_actions : list,  
                                t_plan : float,
                                t_next : float,
                                planning_horizon : float,
                                orbitdata : OrbitData) -> None:
        """
        Calculates and saves the access times of all known requests
        """

        if t_next - state.t >= planning_horizon:
            # recalculate access times for all known requests
            for req in self.known_reqs:
                req : MeasurementRequest
                if req.id in self.access_times:
                    self.access_times.pop(req.id)

        # calculate new access times for new requests
        uncalculated_reqs = [req for req in self.known_reqs if req.id not in self.access_times]
        for req in uncalculated_reqs:
            req : MeasurementRequest
            self.access_times[req.id] = {instrument : [] for instrument in req.measurements}
            for instrument in self.access_times[req.id]:
                if instrument not in state.payload:
                    # agent cannot perform this request
                    continue

                if (req, instrument) in self.performed_requests:
                    # agent has already performed this request
                    continue

                t_arrivals : list = self.__calc_arrival_times(   state, 
                                                                req,
                                                                instrument, 
                                                                state.t, 
                                                                t_next - state.t, 
                                                                orbitdata)
                self.access_times[req.id][instrument] = t_arrivals

        # update access times if a measurement was completed
        for action in performed_actions:
            if isinstance(action, MeasurementAction):
                req : MeasurementRequest = MeasurementRequest.from_dict(action.measurement_req)
                if action.status == action.COMPLETED:
                    self.access_times[req.id] = {instrument : [] for instrument in req.measurements}
                    # self.access_times.pop(req.id)

        # update latest available access time for each known request
        for req_id in self.access_times:
            for instrument in self.access_times[req_id]:
                t_arrivals : list = self.access_times[req_id][instrument]
                while len(t_arrivals) > 0 and t_arrivals[0] < state.t:
                    t_arrivals.pop(0)
    
    @runtime_tracker
    def __calc_arrival_times(self, 
                            state : SimulationAgentState, 
                            req : MeasurementRequest, 
                            instrument : str,
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
                # t_end = t_prev + planning_horizon
                # df : pd.DataFrame = orbitdata.get_ground_point_accesses_future(lat, lon, instrument, t_prev, t_end)

                t_end = min(t_prev + planning_horizon, req.t_end)
                t_start = min(max(t_prev, req.t_start), t_prev + planning_horizon)
                df : pd.DataFrame = orbitdata.get_ground_point_accesses_future(lat, lon, instrument, t_start, t_end)

                for _, row in df.iterrows():
                    t_img = row['time index'] * orbitdata.time_step
                    dt = t_img - state.t
                
                    # propagate state
                    propagated_state : SatelliteAgentState = state.propagate(t_prev)

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


class FIFOReplanner(AbstractReplanner):

    @runtime_tracker
    def needs_replanning(   self, 
                            state : SimulationAgentState,
                            current_plan : list,
                            performed_actions : list,
                            incoming_reqs : list,
                            generated_reqs : list,
                            misc_messages : list,
                            t_plan : float,
                            t_next : float,
                            planning_horizon : float = np.Inf,
                            orbitdata : OrbitData = None
                        ) -> bool:
        
        # update list of known requests
        new_reqs : list = self._update_known_requests( current_plan, 
                                                        incoming_reqs,
                                                        generated_reqs)
        
        # update list of performed measurements
        self.__update_performed_requests(performed_actions)

        # update access times for known requests
        self._update_access_times( state, 
                                    new_reqs, 
                                    performed_actions,
                                    t_plan,
                                    t_next,
                                    planning_horizon,
                                    orbitdata)        

        # check if incoming or generated measurement requests are already accounted for
        _, unscheduled_reqs = self._compare_requests(current_plan)

        # replan if there are new requests to be scheduled
        return len(unscheduled_reqs) > 0
    
    @runtime_tracker
    def __update_performed_requests(self, performed_actions : list) -> None:
        """ Updates an internal list of requests performed by the parent agent """
        for action in performed_actions:
            if isinstance(action, MeasurementAction):
                req : MeasurementRequest = MeasurementRequest.from_dict(action.measurement_req)
                if( action.status == action.COMPLETED                                   
                    and (req, action.instrument_name) not in self.performed_requests
                    ):
                    self.performed_requests.append((req, action.instrument_name))

        return

    
    def _compare_requests(  self, 
                            current_plan : list
                        ) -> tuple:
        """ Separates scheduled and unscheduled requests """
        
        ## list all unique requests in current plan
        scheduled_reqs = []
        for action in current_plan:
            if isinstance(action, MeasurementAction):
                req = MeasurementRequest.from_dict(action.measurement_req)
                if (req, action.instrument_name) not in scheduled_reqs:
                    scheduled_reqs.append((req, action.instrument_name))

        ## list all known available request that have not been scheduled yet
        def is_new_req(req : MeasurementRequest) -> bool:
            scheduled_measurements = []
            for instrument in req.measurements:
                if (req, instrument) in scheduled_reqs:
                    scheduled_measurements.append(instrument)
            unscheduled_measurement = [measurement for measurement in req.measurements if measurement not in scheduled_measurements]
                        
            for instrument in unscheduled_measurement:
                t_arrivals = self.access_times[req.id][instrument]
                if len(t_arrivals) > 0:
                    return True
                
            return False

        unscheduled_reqs = list(filter(is_new_req, self.known_reqs))
        
        return scheduled_reqs, unscheduled_reqs
    
    def replan( self, 
                state : AbstractAgentState, 
                current_plan : list,
                performed_actions : list,
                incoming_reqs : list, 
                generated_reqs : list,
                misc_messages : list,
                t_plan : float,
                t_next : float,
                clock_config : ClockConfig,
                orbitdata : OrbitData = None
            ) -> list:
        
        path = []         
        available_reqs : list = self._get_available_requests()

        if isinstance(state, SatelliteAgentState):
            # Generates a plan for observing GPs on a first-come first-served basis
            reqs = {req.id : req for req, _ in available_reqs}

            for req, subtask_index in available_reqs:
                instrument, _ = req.measurement_groups[subtask_index]  
                t_arrivals : list = self.access_times[req.id][instrument]

                if len(t_arrivals) > 0:
                    t_img = t_arrivals.pop(0)
                    req : MeasurementRequest = reqs[req.id]
                    s_j = self.utility_func(req.to_dict(), t_img) * synergy_factor(req.to_dict(), subtask_index)
                    path.append((req, subtask_index, t_img, s_j))

            path.sort(key=lambda a: a[2])

            # out = '\n'
            # for req, subtask_index, t_img, s in path:
            #     out += f"{req.id.split('-')[0]}\t{subtask_index}\t{np.round(t_img,3)}\t{np.round(s,3)}\n"
            # print(out)

            while True:
                
                conflict_free = True
                i_remove = None

                for i in range(len(path) - 1):
                    j = i + 1
                    req_i, _, t_i, __ = path[i]
                    req_j, subtask_index_j, t_j, s_j = path[j]

                    th_i = state.calc_off_nadir_agle(req_i)
                    th_j = state.calc_off_nadir_agle(req_j)

                    if abs(th_i - th_j) / state.max_slew_rate > t_j - t_i:
                        instrument, _ = req.measurement_groups[subtask_index_j]  
                        t_arrivals : list = self.access_times[req_j.id][instrument]
                        if len(t_arrivals) > 0:
                            # pick next arrival time
                            t_img = t_arrivals.pop(0)
                            s_j = self.utility_func(req.to_dict(), t_img) * synergy_factor(req.to_dict(), subtask_index)

                            path[j] = (req_j, subtask_index_j, t_img, s_j)
                            path.sort(key=lambda a: a[2])
                        else:
                            # remove request from path
                            i_remove = j
                            conflict_free = False
                            break
                            # raise Exception("Whoops. See Plan Initializer.")
                        conflict_free = False
                        break
                
                if i_remove is not None:
                    path.pop(j) 

                if conflict_free:
                    break
                    
            # out = '\n'
            # for req, subtask_index, t_img, s in path:
            #     out += f"{req.id.split('-')[0]}\t{subtask_index}\t{np.round(t_img,3)}\t{np.round(s,3)}\n"
            # print(out)

            # generate plan from path
            plan = self._plan_from_path(state, path, state.t, clock_config)

            return plan
                
        else:
            raise NotImplementedError(f'initial planner for states of type `{type(state)}` not yet supported')
    
    @runtime_tracker
    def _get_available_requests(self) -> list:
        """ Returns a list of known requests that can be performed within the current planning horizon """

        reqs = {req.id : req for req in self.known_reqs}
        available_reqs = []

        for req_id in self.access_times:
            req : MeasurementRequest = reqs[req_id]
            for instrument in self.access_times[req_id]:
                t_arrivals : list = self.access_times[req_id][instrument]

                if len(t_arrivals) > 0:
                    for subtask_index in range(len(req.measurement_groups)):
                        main_instrument, _ = req.measurement_groups[subtask_index]
                        if main_instrument == instrument:
                            available_reqs.append((reqs[req_id], subtask_index))
                            break

        return available_reqs