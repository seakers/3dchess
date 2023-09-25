from abc import ABC, abstractmethod
import logging
import math
import pandas as pd

from dmas.clocks import *

from nodes.states import AbstractAgentState
from nodes.orbitdata import OrbitData
from nodes.states import *
from nodes.actions import *
from nodes.science.reqs import *
from nodes.states import SimulationAgentState
from nodes.orbitdata import OrbitData
from nodes.states import AbstractAgentState

class AbstractPreplanner(ABC):
    """
    # Preplanner
    """
    def __init__(   self, 
                    logger: logging.Logger = None
                ) -> None:
        super().__init__()
        self._logger = logger

    @abstractmethod
    def initialize_plan(    self, 
                            state : AbstractAgentState, 
                            initial_reqs : list, 
                            orbitdata : OrbitData,
                            clock_config : ClockConfig,
                            level : int = logging.DEBUG
                        ) -> list:
        """
        Creates an initial plan for the agent to perform
        """

    def _plan_from_path(    self, 
                            state : SimulationAgentState, 
                            path : list,
                            orbitdata : OrbitData,
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
            t_0 = state.t
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
                    t_prev = state.t
                    prev_state : SatelliteAgentState = state.copy()

                elif isinstance(state, UAVAgentState):
                    t_prev = state.t #TODO consider wait time for convergence
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
                df : pd.DataFrame = orbitdata.get_ground_point_accesses_future(lat, lon, t_move_start)
                
                t_move_end = None
                for _, row in df.iterrows():
                    if row['time index'] * orbitdata.time_step >= t_img:
                        t_move_end = row['time index'] * orbitdata.time_step
                        break

                if t_move_end is None:
                    # unpheasible path
                    self.log(f'Unheasible element in path. Cannot perform observation.', level=logging.DEBUG)
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

    def _get_available_requests( self, 
                                state : SimulationAgentState, 
                                requests : list,
                                orbitdata : OrbitData
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
                if self.__can_bid(state, req, subtask_index, orbitdata):
                    available.append((req, subtask_index))

        return available

    def __can_bid(self, 
                state : SimulationAgentState, 
                req : MeasurementRequest, 
                subtask_index : int, 
                orbitdata : OrbitData,
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
                df : pd.DataFrame = orbitdata.get_ground_point_accesses_future(lat, lon, state.t).sort_values(by='time index')
                
                can_access = False
                if not df.empty:                
                    times = df.get('time index')
                    for time in times:
                        time *= orbitdata.time_step 

                        if state.t + planning_horizon < time:
                            break

                        if req.t_start <= time <= req.t_end:
                            # there exists an access time before the request's availability ends
                            can_access = True
                            break
                
                if not can_access:
                    return False
        
        return True

    def _calc_arrival_times(self, 
                            state : SimulationAgentState, 
                            req : MeasurementRequest, 
                            t_prev : Union[int, float],
                            orbitdata : OrbitData) -> float:
        """
        Estimates the quickest arrival time from a starting position to a given final position
        """
        if isinstance(req, GroundPointMeasurementRequest):
            # compute earliest time to the task
            if isinstance(state, SatelliteAgentState):
                t_imgs = []
                lat,lon,_ = req.lat_lon_pos
                df : pd.DataFrame = orbitdata.get_ground_point_accesses_future(lat, lon, t_prev)

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
                        
                return t_imgs if len(t_imgs) > 0 else [-1]

            elif isinstance(state, UAVAgentState):
                dr = np.array(req.pos) - np.array(state.pos)
                norm = np.sqrt( dr.dot(dr) )
                return [norm / state.max_speed + t_prev]

            else:
                raise NotImplementedError(f"arrival time estimation for agents of type `{type(state)}` is not yet supported.")

        else:
            raise NotImplementedError(f"cannot calculate imaging time for measurement requests of type {type(req)}")       

    def log(self, msg : str, level=logging.DEBUG) -> None:
        """
        Logs a message to the desired level.
        """
        try:
            return 
            #TODO 
            # t = t if t is None else round(t,3)

            # if self._logger is None:
            #     return

            # if level is logging.DEBUG:
            #     self._logger.debug(f'T={t}[s] | {self.name}: {msg}')
            # elif level is logging.INFO:
            #     self._logger.info(f'T={t}[s] | {self.name}: {msg}')
            # elif level is logging.WARNING:
            #     self._logger.warning(f'T={t}[s] | {self.name}: {msg}')
            # elif level is logging.ERROR:
            #     self._logger.error(f'T={t}[s] | {self.name}: {msg}')
            # elif level is logging.CRITICAL:
            #     self._logger.critical(f'T={t}[s] | {self.name}: {msg}')
        
        except Exception as e:
            raise e

class IdlePlanner(AbstractPreplanner):
    def initialize_plan(    self, 
                            state : AbstractAgentState, 
                            initial_reqs : list, 
                            orbitdata : OrbitData,
                            clock_config : ClockConfig,
                            level : int = logging.DEBUG
                        ) -> list:
        return [IdleAction(0.0, np.Inf)]

class FIFOPreplanner(AbstractPreplanner):
    def initialize_plan(    self, 
                            state: AbstractAgentState, 
                            initial_reqs: list, 
                            orbitdata: OrbitData, 
                            clock_config : ClockConfig,
                            level: int = logging.DEBUG
                        ) -> list:
        path = []         
        available_reqs : list = self._get_available_requests( state ,initial_reqs, orbitdata )

        if isinstance(state, SatelliteAgentState):
            # Generates a plan for observing GPs on a first-come first-served basis
            
            reqs = {req.id : req for req, _ in available_reqs}
            arrival_times = {req.id : {} for req, _ in available_reqs}

            for req, subtask_index in available_reqs:
                t_arrivals : list = self._calc_arrival_times(state, req, state.t, orbitdata)
                arrival_times[req.id][subtask_index] = t_arrivals
            
            path = []

            for req_id in arrival_times:
                for subtask_index in arrival_times[req_id]:
                    t_arrivals : list = arrival_times[req_id][subtask_index]
                    t_img = t_arrivals.pop(0)
                    req : MeasurementRequest = reqs[req_id]
                    path.append((req, subtask_index, t_img, req.s_max/len(req.measurements)))

            path.sort(key=lambda a: a[2])

            while True:
                
                conflict_free = True
                for i in range(len(path) - 1):
                    j = i + 1
                    req_i, _, t_i, __ = path[i]
                    req_j, subtask_index_j, t_j, s_j = path[j]

                    th_i = state.calc_off_nadir_agle(req_i)
                    th_j = state.calc_off_nadir_agle(req_j)

                    if abs(th_i - th_j) / state.max_slew_rate > t_j - t_i:
                        t_arrivals : list = arrival_times[req_j.id][subtask_index_j]
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
            # self.log(out,level)

            return self._plan_from_path(state, path, orbitdata, clock_config)
                
        else:
            raise NotImplementedError(f'initial planner for states of type `{type(state)}` not yet supported')
    