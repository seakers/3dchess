import logging
from dmas.clocks import ClockConfig
from dmas.utils import runtime_tracker
from dmas.clocks import *
from numpy import Inf
import pandas as pd
from pyparsing import Any
from traitlets import Callable
from chess3d.messages import ClockConfig
from chess3d.nodes.planning.plan import Plan
from chess3d.nodes.science.utility import synergy_factor
from chess3d.nodes.states import SimulationAgentState

from nodes.orbitdata import OrbitData
from nodes.science.reqs import *
from nodes.planning.plan import Plan, Preplan, Replan
from nodes.planning.planners import AbstractPlanner
from nodes.states import *
from messages import *

class AbstractReplanner(AbstractPlanner):
    def __init__(self, 
                 utility_func: Callable = None, 
                 logger: logging.Logger = None
                 ) -> None:
        """ 
        # Abstract Replanner 

        Only schedules the breoadcast of newly generated measurement requests into the current plan
        """
        super().__init__(utility_func, logger)
        
        self.preplan : Preplan = Preplan(t=-1.0)
        self.plan : Replan = Replan(t=-1)

    def update_precepts(self, 
                        state: SimulationAgentState, 
                        current_plan: Plan, 
                        completed_actions: list, 
                        aborted_actions: list, 
                        pending_actions: list, 
                        incoming_reqs: list, 
                        generated_reqs: list, 
                        relay_messages: list, 
                        misc_messages: list, 
                        orbitdata: dict = None
                        ) -> None:
        # update preplan
        if state.t == current_plan.t and isinstance(current_plan, Preplan): 
            self.preplan = current_plan.copy() 
        
        return super().update_precepts(state, current_plan, completed_actions, aborted_actions, pending_actions, incoming_reqs, generated_reqs, relay_messages, misc_messages, orbitdata)

    def needs_planning(self, *_) -> bool:
        # check if there any requests that have not been broadcasted yet
        requests_broadcasted = [msg.req['id'] for msg in self.completed_broadcasts 
                                if isinstance(msg, MeasurementRequestMessage)]
        requests_to_broadcast = [req for req in self.generated_reqs
                                    if isinstance(req, MeasurementRequest)
                                    and req.id not in requests_broadcasted]

        # replans if relays need to be sent or if requests have to be announced
        return len(requests_to_broadcast) > 0
        
    @runtime_tracker
    def _update_access_times(self,
                             state : SimulationAgentState,
                             agent_orbitdata : OrbitData
                            ) -> None:
        """
        Calculates and saves the access times of all known requests
        """
        if state.t == self.preplan.t or any([req.id not in self.access_times for req in self.known_reqs]):
            # recalculate access times for all known requests            
            for req in self.known_reqs:
                req : MeasurementRequest

                if state.t == self.preplan.t or req.id not in self.access_times:
                    self.access_times[req.id] = {instrument : [] for instrument in req.measurements}

                # check access for each required measurement
                for instrument in self.access_times[req.id]:
                    if instrument not in state.payload:
                        # agent cannot perform this request TODO add KG support
                        continue

                    if (req, instrument) in self.completed_requests:
                        # agent has already performed this request
                        continue

                    if len(self.access_times[req.id][instrument]) > 0:
                        # access times for this request have already been calculated for this period
                        continue

                    if isinstance(req, GroundPointMeasurementRequest):
                        lat,lon,_ = req.lat_lon_pos 
                        t_start = state.t
                        t_end = self.preplan.t_next

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

        else:
            # remove past access times
            for req_id in self.access_times:
                for instrument in self.access_times[req_id]:
                    t_imgs : list = self.access_times[req_id][instrument]
                    if not t_imgs:
                        continue

                    while len(t_imgs) > 0 and t_imgs[0] < state.t:
                        t_imgs.pop(0)

                    if len(t_imgs) == 0:
                        x=1

                    self.access_times[req_id][instrument] = t_imgs

        return
    
class RelayReplanner(AbstractReplanner):
    def needs_planning(self, state : SimulationAgentState, plan : Plan) -> bool:
        return super().needs_planning(state, plan) or len(self.pending_relays) > 0
    
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
        
        # initialize list of broadcasts to be done
        broadcasts = self._schedule_broadcasts(state, None, orbitdata)
                                                
        # update plan with new broadcasts
        self.plan : Replan = Replan.from_preplan(current_plan, broadcasts, t=state.t)

        # return scheduled broadcasts
        return self.plan
        
class ReactivePlanner(AbstractReplanner):    
    def generate_plan(self, 
                      state: SimulationAgentState, 
                      current_plan: Plan, 
                      completed_actions: list, 
                      aborted_actions: list, 
                      pending_actions: list, 
                      incoming_reqs: list, 
                      generated_reqs: list, 
                      relay_messages: list, 
                      misc_messages: list, 
                      clock_config: ClockConfig, 
                      orbitdata: dict = None
                      ) -> Plan:
        
        # schedule measurements
        measurements : list = self._schedule_measurements(state, clock_config)

        # schedule broadcasts to be perfomed
        broadcasts : list = self._schedule_broadcasts(state, measurements, orbitdata)

        # generate maneuver and travel actions from measurements
        maneuvers : list = self._schedule_maneuvers(state, measurements, broadcasts, clock_config)
        
        # generate plan from actions
        self.plan : Preplan = Replan(measurements, maneuvers, broadcasts, t=state.t, t_next=self.preplan.t_next)    

        # return plan
        return self.plan

    @abstractmethod
    def _schedule_measurements(self, state : SimulationAgentState, clock_config : ClockConfig) -> list:
        pass

class FIFOReplanner(ReactivePlanner):
    def __init__(self, 
                 utility_func: Callable = None, 
                 collaboration : bool = False,
                 logger: logging.Logger = None
                 ) -> None:
        super().__init__(utility_func, logger)

        self.collaboration = collaboration
        self.other_plans = {}
        self.ignored_reqs = []

    def generate_plan(self, state: SimulationAgentState, current_plan: Plan, completed_actions: list, aborted_actions: list, pending_actions: list, incoming_reqs: list, generated_reqs: list, relay_messages: list, misc_messages: list, clock_config: ClockConfig, orbitdata: dict = None) -> Plan:
        
        # schedule measurements
        prev_measurements = [action for action in current_plan if isinstance(action, MeasurementAction)]
        available_measurements = self._get_available_requests()

        if len(prev_measurements) > len(available_measurements):
            x = 1 

        measurements : list = self._schedule_measurements(state, clock_config)

        if len(prev_measurements) > len(measurements):
            x = 1
        
        return super().generate_plan(state, current_plan, completed_actions, aborted_actions, pending_actions, incoming_reqs, generated_reqs, relay_messages, misc_messages, clock_config, orbitdata)

    def update_precepts(self, 
                        state: SimulationAgentState, 
                        current_plan: Plan, 
                        completed_actions: list, 
                        aborted_actions: list, 
                        pending_actions: list, 
                        incoming_reqs: list, 
                        generated_reqs: list, 
                        relay_messages: list, 
                        misc_messages: list, 
                        orbitdata: dict = None
                        ) -> None:
        
        # initialize update
        super().update_precepts(state, 
                                current_plan, 
                                completed_actions, 
                                aborted_actions, 
                                pending_actions, 
                                incoming_reqs, 
                                generated_reqs, 
                                relay_messages, 
                                misc_messages, 
                                orbitdata)
        
        # compile incoming plans
        incoming_plans = [(msg.src, Plan([action_from_dict(**action) for action in msg.plan], t=state.t))
                            for msg in misc_messages
                            if isinstance(msg, PlanMessage)]
        
        # update internal knowledge base of other agent's 
        for src, plan in incoming_plans:
            self.other_plans[src] = plan
        self.plan = self.preplan.copy()

    def needs_planning(self, state: SimulationAgentState, plan: Plan) -> bool:
        if self.collaboration:
            # check if other agents have plans with tasks considered by the current plan
            my_measurements = [(MeasurementRequest.from_dict(action.measurement_req), action.subtask_index, action.t_start)
                                for action in self.plan
                                if isinstance(action, MeasurementAction)]
            
            for agent_name in self.other_plans:
                plan : Plan = self.other_plans[agent_name]
                their_measurements = [(MeasurementRequest.from_dict(action.measurement_req), action.subtask_index, action.t_start)
                                        for action in plan
                                        if isinstance(action, MeasurementAction)]
                
                for their_req, their_subtaskt_index, their_t_img in their_measurements:

                    for my_req, my_subtaskt_index, my_t_img in my_measurements:
                        
                        if (their_req == my_req 
                            and their_subtaskt_index == my_subtaskt_index
                            and their_t_img < my_t_img):
                            # other agent is performing a task I was considering but does it sooner
                            return True
                
        # compile requests that have not been considered by the current plan
        inconsidered_req = [req 
                            for req in self.known_reqs
                            if req not in self.plan]
        
        # check if unconsidered requests are accessible
        for req in inconsidered_req:
            req : MeasurementRequest
            for instrument in self.access_times[req.id]:
                t_accesses = self.access_times[req.id][instrument]
                if t_accesses:
                    # measurement request is accessible and hasent been considered yet
                    return True

        return super().needs_planning(state, plan) or len(self.pending_relays) > 0
        
    
    @runtime_tracker
    def _get_available_requests(self) -> list:
        """ Returns a list of known requests that can be performed within the current planning horizon """

        reqs = {req.id : req 
                for req in self.known_reqs}

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

    @runtime_tracker
    def _schedule_measurements(self, state : SimulationAgentState, _ : ClockConfig) -> list:
        """ 
        Schedule a sequence of observations based on the current state of the agent 
        
        ### Arguments:
            - state (:obj:`SimulationAgentState`): state of the agent at the time of planning
        """
        
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

                # t_others = []
                # for plan in self.other_plans:
                #     measurements = [action
                #                     for action in plan
                #                     if isinstance(action, MeasurementAction) 
                #                     and MeasurementRequest.from_dict(action.measurement_req) == req
                #                     and action.subtask_index == subtask_index]
                    
                #     t_others.extend([measurement.t_start 
                #                      for measurement in measurements
                #                      if isinstance(measurement, MeasurementAction)])
                
                # t_other = min(t_others) if t_others else np.Inf
                # if t_other < np.Inf:
                #     x = 1

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
                self._print_observation_path(state, measurements)
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
                             state: SimulationAgentState, 
                             measurements: list, 
                             orbitdata: dict
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
                    if t_start <= self.preplan.t_next:
                        broadcasts.append(broadcast_action)

            # schedule the broadcast of each scheduled measurement's completion after it's been performed
            for measurement in measurements:
                measurement : MeasurementAction
                path, t_start = self._create_broadcast_path(state.agent_name, orbitdata, measurement.t_end)

                msg = MeasurementPerformedMessage(state.agent_name, state.agent_name, measurement.to_dict(), path=path)

                if t_start >= 0.0:
                    broadcast_action = BroadcastMessageAction(msg.to_dict(), t_start)
                    
                    # check broadcast start; only add to plan if it's within the planning horizon
                    if t_start <= self.preplan.t_next:
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
                    if t_start <= self.preplan.t_next:
                        broadcasts.append(broadcast_action)

                    assert completed_measurement.t_end <= broadcast_action.t_start
        
        # sort broadcasts by start time
        broadcasts.sort(key=lambda a : a.t_start)

        # return broadcast list
        return broadcasts