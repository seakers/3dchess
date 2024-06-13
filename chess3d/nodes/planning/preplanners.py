from abc import abstractmethod
import logging
from typing import Any, Callable

from dmas.utils import runtime_tracker
from dmas.clocks import *
from numpy import Inf
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
        self.horizon = horizon                               # planning horizon
        self.period = period                                 # replanning period         
        self.plan = Preplan(t=-1,horizon=horizon,t_next=0.0) # initialized empty plan
        
    @runtime_tracker
    def needs_planning( self, 
                        state : SimulationAgentState,
                        current_plan : Plan, 
                        **_
                        ) -> bool:
        """ Determines whether a new plan needs to be initalized """    

        return (current_plan.t < 0                  # simulation just started
                or state.t >= self.plan.t_next)     # periodic planning period has been reached

    @runtime_tracker
    def generate_plan(  self, 
                        state : SimulationAgentState,
                        clock_config : ClockConfig,
                        orbitdata : OrbitData,
                        **_
                    ) -> Plan:
        
        # schedule measurements
        measurements : list = self._schedule_measurements(state, clock_config, orbitdata)
        assert self.is_observation_path_valid(state, measurements, orbitdata)

        # schedule broadcasts to be perfomed
        broadcasts : list = self._schedule_broadcasts(state, measurements, orbitdata)

        # generate maneuver and travel actions from measurements
        maneuvers : list = self._schedule_maneuvers(state, measurements, broadcasts, clock_config, orbitdata)
        
        # wait for next planning period to start
        replan : list = self.__schedule_periodic_replan(state, measurements, maneuvers)

        # generate plan from actions
        self.plan : Preplan = Preplan(measurements, maneuvers, broadcasts, replan, t=state.t, horizon=self.horizon, t_next=state.t+self.period)    

        # return plan
        return self.plan
        
    @abstractmethod
    def _schedule_measurements(self, state : SimulationAgentState, clock_config : ClockConfig, orbitdata : dict = None) -> list:
        """ Creates a list of measurement actions to be performed by the agent """
        pass    

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
                                 if  isinstance(action, AgentAction)
                                 and action.t_start < t_next]

            if actions_in_period:
                last_action : AgentAction = actions_in_period.pop()
                t_wait_start = min(last_action.t_end, t_next)
                                
            else:
                t_wait_start = state.t

        # create wait action
        return [WaitForMessages(t_wait_start, t_next)]

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
    def _schedule_measurements(self, state : SimulationAgentState, _ : ClockConfig, orbitdata : dict = None) -> list:
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

            # available_req_times = []
            # for req, subtask_index in available_reqs:
            #     req : MeasurementRequest
            #     main_measurement, _ = req.measurement_groups[subtask_index]  
            #     t_arrivals : list = self.access_times[req.id][main_measurement]

            #     available_req_times.append((req, subtask_index, t_arrivals))

            # while available_req_times:
            #     req_min, subtask_index_min, t_arrivals_min, measurement_min = None, None, None, None
            #     req_times_to_delete = []
            #     for req, subtask_index, t_arrivals in available_req_times:
            #         req : MeasurementRequest
            #         t_arrivals : list

            #         # if measurements:
            #         #     t_arrivals = [t_img for t_img in t_arrivals
            #         #                 for measurement in measurements
            #         #                 if isinstance(measurement, MeasurementAction)
            #         #                 and not (measurement.t_start < t_img < measurement.t_end)]

            #         if len(t_arrivals) == 0:
            #             req_times_to_delete.append((req, subtask_index, t_arrivals))
            #             continue

            #         proposed_measurement = None
            #         while len(t_arrivals) > 0:
            #             t_img = t_arrivals.pop(0)
            #             u_exp = self.utility_func(req.to_dict(), t_img) * synergy_factor(req.to_dict(), subtask_index)

            #             proposed_path = [action for action in measurements]
            #             proposed_measurement = MeasurementAction(req.to_dict(),
            #                                                      subtask_index, 
            #                                                      main_measurement,
            #                                                      u_exp,
            #                                                      t_img, 
            #                                                      t_img + req.duration
            #                                                     )
            #             proposed_path.append(proposed_measurement)
                        
            #             if self.is_observation_path_valid(state, proposed_path, orbitdata):
            #                 break

                    
            #         if (isinstance(proposed_measurement, MeasurementAction)
            #             and (measurement_min is None or proposed_measurement.t_end < measurement_min.t_start)
            #             ):
            #             req_min = req
            #             subtask_index_min = subtask_index
            #             t_arrivals_min = t_arrivals
            #             measurement_min = proposed_measurement
                
            #     if isinstance(measurement_min, MeasurementAction):
            #         measurements.append(measurement_min)
            #         available_req_times.remove((req_min, subtask_index_min, t_arrivals_min))

            #     for req, subtask_index, t_arrivals in req_times_to_delete:
            #         available_req_times.remove((req, subtask_index, t_arrivals))
            #     x = 1

            # create first assignment of observations
            for req, subtask_index in available_reqs:
                req : MeasurementRequest
                main_measurement, _ = req.observation_groups[subtask_index]  
                t_arrivals : list = self.access_times[req.id][main_measurement]

                if len(t_arrivals) > 0:
                    t_img = t_arrivals.pop(0)
                    u_exp = self.utility_func(req.to_dict(), t_img) * synergy_factor(req.to_dict(), subtask_index)

                    # perform measurement
                    measurements.append(ObservationAction( req.to_dict(),
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
                        measurement_i : ObservationAction = measurements[i]
                        req_i : MeasurementRequest = MeasurementRequest.from_dict(measurement_i.measurement_req)

                        lat, lon, _ = req_i.lat_lon_pos
                        main_instrument_i = measurement_i.instrument_name
                        t_i = measurement_i.t_start

                        agent_orbitdata : OrbitData = orbitdata[state.agent_name]
                        obs_i = agent_orbitdata.get_groundpoint_access_data(lat, lon, main_instrument_i, t_i)
                        th_i = obs_i['look angle [deg]']

                        t_i = measurement_i.t_end
                    else:
                        th_i = state.attitude[0]
                        t_i = state.t

                    measurement_j : ObservationAction = measurements[j]
                    req_j : GroundPointMeasurementRequest = MeasurementRequest.from_dict(measurement_j.measurement_req)
                    
                    lat, lon, _ = req_j.lat_lon_pos
                    main_instrument_j = measurement_j.instrument_name
                    t_j = measurement_j.t_start

                    agent_orbitdata : OrbitData = orbitdata[state.agent_name]
                    obs_j = agent_orbitdata.get_groundpoint_access_data(lat, lon, main_instrument_j, t_j)
                    th_j = obs_j['look angle [deg]']

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
                measurement : ObservationAction
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
                                        if isinstance(action, ObservationAction)
                                        and action not in measurements_broadcasted]
            
            # create a broadcast action for all unbroadcasted measurements
            for completed_measurement in measurements_to_broadcast:       
                completed_measurement : ObservationAction
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

class CommonToDoPreplanner(FIFOPreplanner):
    def __init__(self, 
                 utility_func: Callable[[], Any], 
                 period: float = np.Inf, 
                 horizon: float = np.Inf,  
                 logger: logging.Logger = None, 
                 **kwargs) -> None:
        super().__init__(utility_func, period, horizon, True, logger, **kwargs)


class PredefinedPlanner(AbstractPreplanner):
    def __init__(self, 
                 utility_func: Callable[[], Any], 
                 horizon: float = np.Inf, 
                 period: float = np.Inf, 
                 logger: logging.Logger = None
                 ) -> None:
        super().__init__(utility_func, horizon, period, logger)