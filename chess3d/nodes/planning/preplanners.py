from abc import abstractmethod
import logging
from typing import Any, Callable

from dmas.utils import runtime_tracker
from dmas.clocks import *
import pandas as pd

from messages import *

from nodes.planning.plan import Plan, Preplan
from nodes.orbitdata import OrbitData, TimeInterval
from nodes.states import *
from nodes.actions import *
from nodes.science.reqs import *
from nodes.science.utility import synergy_factor
from nodes.states import SimulationAgentState
from nodes.orbitdata import OrbitData
from nodes.planning.planners import AbstractPlanner

class AbstractPreplanner(AbstractPlanner):
    """
    # Preplanner

    Conducts operations planning for an agent at the beginning of a planning horizon. 
    """
    def __init__(   self, 
                    utility_func : Callable[[], Any], 
                    horizon : float = np.Inf,
                    period : float = np.Inf,
                    logger: logging.Logger = None
                ) -> None:
        """
        ## Preplanner 
        
        Creates an instance of a preplanner class object.

        #### Arguments:
            - utility_func (`Callable`): desired utility function for evaluating observations
            - horizon (`float`) : planning horizon in seconds [s]
            - period (`float`) : period of replanning in seconds [s]
            - logger (`logging.Logger`) : debugging logger
        """
        # initialize planner
        super().__init__(utility_func, logger)    

        # set parameters
        self.horizon = horizon              # planning horizon
        self.period = period                # replanning period         
        self.plan = Preplan(t=-1,horizon=horizon,t_next=0.0)
        
    @runtime_tracker
    def needs_planning( self, 
                        state : SimulationAgentState,
                        current_plan : Plan, 
                        *_
                        ) -> bool:
        """ Determines whether a new plan needs to be initalized """    

        return (current_plan.t < 0                  # simulation just started
                or state.t >= self.plan.t_next)     # periodic planning period has been reached

    @runtime_tracker
    def _update_access_times(self,
                             state : SimulationAgentState,
                             agent_orbitdata : OrbitData) -> None:
        """
        Calculates and saves the access times of all known requests
        """
        if state.t >= self.plan.t_next or self.plan.t < 0:
            # recalculate access times for all known requests            
            for req in self.known_reqs:
                req : MeasurementRequest
                self.access_times[req.id] = {instrument : [] for instrument in req.measurements}

                # check access for each required measurement
                for instrument in self.access_times[req.id]:
                    if instrument not in state.payload:
                        # agent cannot perform this request TODO add KG support
                        continue

                    if (req, instrument) in self.completed_requests:
                        # agent has already performed this request
                        continue

                    if isinstance(req, GroundPointMeasurementRequest):
                        lat,lon,_ = req.lat_lon_pos 
                        t_start = state.t
                        t_end = self.plan.t + self.horizon if self.plan else state.t + self.horizon
                        if isinstance(state, SatelliteAgentState):
                            df : pd.DataFrame = agent_orbitdata \
                                            .get_ground_point_accesses_future(lat, lon, instrument, t_start, t_end)
                            t_arrivals = [row['time index'] * agent_orbitdata.time_step
                                        for _, row in df.iterrows()]
                            self.access_times[req.id][instrument] = t_arrivals
                        else:
                            raise NotImplementedError(f"access time estimation for agents of type `{type(state)}` not yet supported.")    

                    else:
                        raise NotImplementedError(f"access time estimation for measurement requests of type `{type(req)}` not yet supported.")
        elif any([req.id not in self.access_times for req in self.known_reqs]):
            for req in self.known_reqs:
                if req.id not in self.access_times:
                    self.access_times[req.id] = {instrument : [] for instrument in req.measurements}


    @runtime_tracker
    def generate_plan(  self, 
                        state : SimulationAgentState,
                        current_plan : Plan,
                        completed_actions : list,
                        aborted_actions : list,
                        pending_actions : list,
                        incoming_reqs : list,
                        generated_reqs : list,
                        relay_messages : list,
                        misc_messages : list,
                        clock_config : ClockConfig,
                        orbitdata : dict = None
                    ) -> Plan:
        
        # schedule measurements
        measurements : list = self._schedule_measurements(state, clock_config)

        # schedule broadcasts to be perfomed
        broadcasts : list = self._schedule_broadcasts(state, measurements, orbitdata)

        # generate maneuver and travel actions from measurements
        maneuvers : list = self._schedule_maneuvers(state, measurements, broadcasts, clock_config)
        
        # wait for next planning period to start
        replan : list = self.__schedule_periodic_replan(state, measurements, maneuvers)

        # generate plan from actions
        self.plan : Preplan = Preplan(measurements, maneuvers, broadcasts, replan, t=state.t, horizon=self.horizon, t_next=state.t+self.period)    

        # return plan
        return self.plan
        
    @abstractmethod
    def _schedule_measurements(self, state : SimulationAgentState, clock_config : ClockConfig) -> list:
        """ Creates a list of measurement actions to be performed by the agent """

    @runtime_tracker
    def _generate_broadcasts(self, 
                             state : SimulationAgentState, 
                             measurements : list, 
                             generated_reqs : list, 
                             orbitdata : OrbitData
                             ) -> list:
        """ Creates broadcast actions """
        # initialize list of broadcasts to be done
        broadcasts = []

        # schedule generated measurement request broadcasts
        for req in generated_reqs:
            req : MeasurementRequest
            msg = MeasurementRequestMessage(state.agent_name, state.agent_name, req.to_dict())
            
            # place broadcasts in plan
            if isinstance(state, SatelliteAgentState):
                if not orbitdata:
                    raise ValueError('orbitdata required for satellite agents')
                
                # get next access windows to all agents
                isl_accesses = [orbitdata.get_next_agent_access(target, state.t) for target in orbitdata.isl_data]

                # TODO merge accesses to find overlaps
                isl_accesses_merged = isl_accesses

                # place requests in pending broadcasts list
                for interval in isl_accesses_merged:
                    interval : TimeInterval
                    broadcast_action = BroadcastMessageAction(msg.to_dict(), max(interval.start, state.t))

                    broadcasts.append(broadcast_action)
                    
            else:
                raise NotImplementedError(f"Scheduling of broadcasts for agents with state of type {type(state)} not yet implemented.")
        
        return broadcasts   

    @runtime_tracker
    def __schedule_periodic_replan(self, state : SimulationAgentState, measurement_actions : list, maneuver_actions : list) -> list:
        """ Creates and schedules a waitForMessage action such that it triggers a periodic replan """
        # calculate next period for planning
        t_next = state.t + self.period

        # find wait start time
        if not measurement_actions and not maneuver_actions:
            t_wait_start = state.t 
        
        else:
            prelim_plan = Preplan(measurement_actions, maneuver_actions, t=state.t)

            actions_in_period = [action for action in prelim_plan.actions 
                                 if action.t_start < t_next]

            if actions_in_period:
                last_action : AgentAction = actions_in_period.pop()
                t_wait_start = min(last_action.t_end, t_next)
                                
            else:
                t_wait_start = state.t

        # create wait action
        return [WaitForMessages(t_wait_start, t_next)]
        
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

class IdlePlanner(AbstractPreplanner):
    @runtime_tracker
    def generate_plan(  self, 
                        *_
                    ) -> tuple:
        return [IdleAction(0.0, np.Inf)]

class FIFOPreplanner(AbstractPreplanner):
    def __init__(self, 
                 utility_func: Callable[[], Any], 
                 period : float = np.Inf,
                 horizon : float = np.Inf,
                 collaboration : bool = False,
                 logger: logging.Logger = None, 
                 **kwargs
                 ) -> None:
        """
        ### First Come, First Served Preplanner
        """

        super().__init__(utility_func, horizon, period, logger)
        self.collaboration = collaboration    

    @runtime_tracker
    def _schedule_measurements(self, state : SimulationAgentState, _ : ClockConfig) -> list:
        """ 
        Schedule a sequence of observations based on the current state of the agent 
        
        ### Arguments:
            - state (:obj:`SimulationAgentState`): state of the agent at the time of planning
        """

        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'FIFO preplanner for agents of type `{type(state)}` not yet supported.')
        
        # initialize measurement path
        measurements = []         

        # get available requests
        available_reqs : list = self._get_available_requests()

        if isinstance(state, SatelliteAgentState):

            # create first assignment of observations
            for req, subtask_index in available_reqs:
                req : MeasurementRequest
                main_measurement, _ = req.measurement_groups[subtask_index]  
                t_arrivals : list = self.access_times[req.id][main_measurement]

                if len(t_arrivals) > 0:
                    t_img = t_arrivals.pop(0)
                    u_exp = self.utility_func(req.to_dict(), t_img) * synergy_factor(req.to_dict(), subtask_index)

                    # perform measurement
                    measurements.append(MeasurementAction( req.to_dict(),
                                                   subtask_index, 
                                                   main_measurement,
                                                   u_exp,
                                                   t_img, 
                                                   t_img + req.duration
                                                 ))

            # sort from ascending start time
            measurements.sort(key=lambda a: a.t_start)

            # ensure conflict-free path
            while True:                
                # ----- FOR DEBUGGING PURPOSES ONLY ------
                # self._print_observation_path(state, measurements)
                # ----------------------------------------

                conflict_free = True
                j_remove = None

                for j in range(len(measurements)):
                    i = j - 1

                    if i >= 0:
                        measurement_i : MeasurementAction = measurements[i]
                        state_i : SatelliteAgentState = state.propagate(measurement_i.t_start)
                        req_i : MeasurementRequest = MeasurementRequest.from_dict(measurement_i.measurement_req)
                        th_i = state_i.calc_off_nadir_agle(req_i)
                        t_i = measurement_i.t_end
                    else:
                        th_i = state.attitude[0]
                        t_i = state.t

                    measurement_j : MeasurementAction = measurements[j]
                    state_j : SatelliteAgentState = state.propagate(measurement_j.t_start)
                    req_j : MeasurementRequest = MeasurementRequest.from_dict(measurement_j.measurement_req)
                    th_j = state_j.calc_off_nadir_agle(req_j)

                    dt_maneuver = abs(th_j - th_i) / state.max_slew_rate
                    dt_measurements = measurement_j.t_start - t_i

                    # check if there's enough time to maneuver from one observation to another
                    if dt_maneuver > dt_measurements:

                        # there is not enough time to maneuver 
                        main_measurement, _ = req_j.measurement_groups[measurement_j.subtask_index]  
                        t_arrivals : list = self.access_times[req_j.id][main_measurement]
                        
                        if len(t_arrivals) > 0:     # try again for the next access period
                            # pick next access time
                            t_img = t_arrivals.pop(0)
                            u_exp = self.utility_func(req.to_dict(), t_img) * synergy_factor(req.to_dict(), subtask_index)

                            # update measurement action
                            measurement_j.t_start = t_img
                            measurement_j.t_end = t_img + req_j.duration
                            measurement_j.u_exp = u_exp
                            
                            # update path list
                            measurements[j] = measurement_j

                            # sort from ascending start time
                            measurements.sort(key=lambda a: a.t_start)
                        else:                       # no more future accesses for this GP
                            j_remove = j

                        # flag current observation plan as unfeasible for rescheduling
                        conflict_free = False
                        break
                
                if j_remove is not None:
                    measurements.pop(j) 

                    # sort from ascending start time
                    measurements.sort(key=lambda a: a.t_start)

                if conflict_free:
                    break
            
        else:
            raise NotImplementedError(f'initial planner for states of type `{type(state)}` not yet supported')
    
        return measurements
    
    @runtime_tracker
    def _schedule_broadcasts(self, 
                             state : SimulationAgentState, 
                             measurements : list, 
                             orbitdata : dict
                            ) -> list:
        # initialize broadcasts
        broadcasts : list = super()._schedule_broadcasts(state,
                                                         measurements, 
                                                         orbitdata)

        # schedule performed measurement broadcasts
        if self.collaboration:

            # schedule the announcement of the current plan
            if measurements:
                # if there are measurements in plan, create plan broadcast action
                path, t_start = self._create_broadcast_path(state.agent_name, orbitdata, state.t)
                msg = PlanMessage(state.agent_name, 
                                state.agent_name, 
                                [action.to_dict() for action in measurements],
                                state.t,
                                path=path)
                
                if t_start >= 0.0:
                    broadcast_action = BroadcastMessageAction(msg.to_dict(), t_start)
                    
                    # check broadcast start; only add to plan if it's within the planning horizon
                    if t_start <= state.t + self.horizon:
                        broadcasts.append(broadcast_action)

            # schedule the broadcast of each scheduled measurement's completion after it's been performed
            for measurement in measurements:
                measurement : MeasurementAction
                path, t_start = self._create_broadcast_path(state.agent_name, orbitdata, measurement.t_end)

                msg = MeasurementPerformedMessage(state.agent_name, state.agent_name, measurement.to_dict(), path=path)

                if t_start >= 0.0:
                    broadcast_action = BroadcastMessageAction(msg.to_dict(), t_start)
                    
                    # check broadcast start; only add to plan if it's within the planning horizon
                    if t_start <= state.t + self.horizon:
                        broadcasts.append(broadcast_action)
           
            # check which measurements that have been performed and already broadcasted
            measurements_broadcasted = [action_from_dict(**msg.measurement_action)
                                        for msg in self.completed_broadcasts 
                                        if isinstance(msg, MeasurementPerformedMessage)]

            # search for measurements that have been performed but not yet been broadcasted
            measurements_to_broadcast = [action for action in self.completed_actions 
                                        if isinstance(action, MeasurementAction)
                                        and action not in measurements_broadcasted]
            
            # create a broadcast action for all unbroadcasted measurements
            for completed_measurement in measurements_to_broadcast:       
                completed_measurement : MeasurementAction
                t_end = max(completed_measurement.t_end, state.t)
                path, t_start = self._create_broadcast_path(state.agent_name, orbitdata, t_end)
                
                msg = MeasurementPerformedMessage(state.agent_name, state.agent_name, completed_measurement.to_dict())
                
                if t_start >= 0.0:
                    broadcast_action = BroadcastMessageAction(msg.to_dict(), t_start, path=path)
                    
                    # check broadcast start; only add to plan if it's within the planning horizon
                    if t_start <= state.t + self.horizon:
                        broadcasts.append(broadcast_action)

                    assert completed_measurement.t_end <= broadcast_action.t_start
        
        # sort broadcasts by start time
        broadcasts.sort(key=lambda a : a.t_start)

        # return broadcast list
        return broadcasts