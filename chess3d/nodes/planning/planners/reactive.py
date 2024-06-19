    


#     def generate_plan(  self, 
#                         state : SimulationAgentState,
#                         current_plan : Plan,
#                         completed_actions : list,
#                         aborted_actions : list,
#                         pending_actions : list,
#                         incoming_reqs : list,
#                         generated_reqs : list,
#                         relay_messages : list,
#                         misc_messages : list,
#                         clock_config : ClockConfig,
#                         orbitdata : dict = None
#                     ) -> Plan:
        
#         # initialize list of broadcasts to be done
#         broadcasts = self._schedule_broadcasts(state, None, orbitdata)
                                                
#         # update and return plan with new broadcasts
#         return Replan.from_preplan(current_plan, broadcasts, t=state.t)
        
# class ReactivePlanner(AbstractReplanner):    
#     def generate_plan(self, 
#                       state: SimulationAgentState, 
#                       current_plan: Plan, 
#                       completed_actions: list, 
#                       aborted_actions: list, 
#                       pending_actions: list, 
#                       incoming_reqs: list, 
#                       generated_reqs: list, 
#                       relay_messages: list, 
#                       misc_messages: list, 
#                       clock_config: ClockConfig, 
#                       orbitdata: dict = None
#                       ) -> Plan:
        
#         # schedule measurements
#         measurements : list = self._schedule_measurements(state, current_plan, clock_config, orbitdata)
#         assert self.is_observation_path_valid(state, measurements, orbitdata)

#         # schedule broadcasts to be perfomed
#         broadcasts : list = self._schedule_broadcasts(state, measurements, orbitdata)

#         # generate maneuver and travel actions from measurements
#         maneuvers : list = self._schedule_maneuvers(state, measurements, broadcasts, clock_config, orbitdata)
        
#         # generate plan from actions
#         return Replan(measurements, maneuvers, broadcasts, t=state.t, t_next=self.preplan.t_next)    

#     @abstractmethod
#     def _schedule_measurements(self, state : SimulationAgentState, current_plan : list, clock_config : ClockConfig, orbitdata : dict) -> list:
#         pass

# class FIFOReplanner(ReactivePlanner):
#     def __init__(self, 
#                  utility_func: Callable = None, 
#                  collaboration : bool = False,
#                  logger: logging.Logger = None
#                  ) -> None:
#         super().__init__(utility_func, logger)

#         self.collaboration = collaboration
#         self.other_plans = {}
#         self.ignored_reqs = []

#     def update_percepts(self, 
#                         state: SimulationAgentState, 
#                         current_plan: Plan, 
#                         completed_actions: list, 
#                         aborted_actions: list, 
#                         pending_actions: list, 
#                         incoming_reqs: list, 
#                         generated_reqs: list, 
#                         relay_messages: list, 
#                         misc_messages: list, 
#                         orbitdata: dict = None
#                         ) -> None:
        
#         # initialize update
#         super().update_percepts(state, 
#                                 current_plan, 
#                                 completed_actions, 
#                                 aborted_actions, 
#                                 pending_actions, 
#                                 incoming_reqs, 
#                                 generated_reqs, 
#                                 relay_messages, 
#                                 misc_messages, 
#                                 orbitdata)
        
#         # compile incoming plans
#         incoming_plans = [(msg.src, Plan([action_from_dict(**action) for action in msg.plan], t=msg.t_plan))
#                             for msg in misc_messages
#                             if isinstance(msg, PlanMessage)]
        
#         # update internal knowledge base of other agent's 
#         for src, plan in incoming_plans:
#             plan : Plan
#             if src not in self.other_plans:
#                 self.other_plans[src] = plan
#             elif plan.t > self.other_plans[src].t:
#                 # only update if a newer plan was received
#                 self.other_plans[src] = plan

#     def needs_planning(self, state: SimulationAgentState, current_plan: Plan) -> bool:
#         if self.collaboration:
#             # check if other agents have plans with tasks considered by the current plan
#             my_measurements = [(MeasurementRequest.from_dict(action.measurement_req), 
#                                                              action.subtask_index, 
#                                                              action.t_start)
#                                 for action in current_plan
#                                 if isinstance(action, ObservationAction)]
            
#             for my_req, my_subtaskt_index, my_t_img in my_measurements:
#                 for _, their_plan in self.other_plans.items():                
#                     their_measurements = [(MeasurementRequest.from_dict(action.measurement_req), 
#                                            action.subtask_index, 
#                                            action.t_start)
                                           
#                                            for action in their_plan
#                                            if isinstance(action, ObservationAction)
#                                            and action.t_start < my_t_img]
                
#                     for their_req, their_subtaskt_index, their_t_img in their_measurements:
                        
#                         if (their_req == my_req 
#                             and their_subtaskt_index == my_subtaskt_index):
#                             # other agent is performing a task I was considering but does it sooner
#                             # short_id = my_req.id.split('-')[0]
#                             # print(f'CONFLICT AT: {short_id}, {my_subtaskt_index}, {their_t_img} < {my_t_img}')
#                             return True
                
#         # compile requests that have not been considered by the current plan
#         considered_reqs = [MeasurementRequest.from_dict(action.measurement_req)
#                            for action in current_plan
#                            if isinstance(action, ObservationAction)]
#         considered_reqs.extend([req 
#                                 for req, _ in self.completed_requests
#                                 if req not in considered_reqs])
#         inconsidered_req = [req 
#                             for req in self.known_reqs
#                             if req not in considered_reqs
#                             and req not in self.ignored_reqs]
        
#         # check if unconsidered requests are accessible
#         for req in inconsidered_req:
#             req : MeasurementRequest
#             for instrument in self.access_times[req.id]:
#                 t_accesses = self.access_times[req.id][instrument]
#                 if t_accesses and req not in self.ignored_reqs:
#                     # measurement request is accessible and hasent been considered yet
#                     return True

#         return super().needs_planning(state, current_plan) or len(self.pending_relays) > 0
                
#     @runtime_tracker
#     def _get_available_requests(self) -> list:
#         """ Returns a list of known requests that can be performed within the current planning horizon """

#         reqs = {req.id : req 
#                 for req in self.known_reqs}

#         available_reqs = []
#         for req_id in self.access_times:
#             req : MeasurementRequest = reqs[req_id]

#             if req in self.ignored_reqs:
#                 continue

#             for instrument in self.access_times[req_id]:
#                 t_arrivals : list = self.access_times[req_id][instrument]

#                 if len(t_arrivals) > 0:
#                     for subtask_index in range(len(req.observation_groups)):
#                         main_instrument, _ = req.observation_groups[subtask_index]
#                         if main_instrument == instrument:
#                             available_reqs.append((req, subtask_index))
#                             break

#         return available_reqs

#     @runtime_tracker
#     def _schedule_measurements(self, state : SimulationAgentState, current_plan : list, _ : ClockConfig, orbitdata : dict) -> list:
#         """ 
#         Schedule a sequence of observations based on the current state of the agent 
        
#         ### Arguments:
#             - state (:obj:`SimulationAgentState`): state of the agent at the time of planning
#         """
        
#         my_measurements = [ action
#                             for action in current_plan
#                             if isinstance(action, ObservationAction)]

#         if self.collaboration:
#             # check if other agents have plans with tasks considered by the current plan
            
#             conflicts = []

#             for my_action in my_measurements:
#                 my_req = MeasurementRequest.from_dict(my_action.measurement_req)
#                 my_subtaskt_index = my_action.subtask_index
#                 my_t_img = my_action.t_start

#                 conflict = None
#                 for _, their_plan in self.other_plans.items():                
#                     their_measurements = [(MeasurementRequest.from_dict(action.measurement_req), 
#                                            action.subtask_index, 
#                                            action.t_start)
#                                         for action in their_plan
#                                         if isinstance(action, ObservationAction)
#                                         and action.t_start < my_t_img]
                
#                     for their_req, their_subtaskt_index, their_t_img in their_measurements:
#                         if (their_req == my_req 
#                             and their_subtaskt_index == my_subtaskt_index
#                             and their_t_img < my_t_img):
#                             # other agent is performing a task I was considering but does it sooner
#                             conflict = my_action
#                             break

#                     if conflict:
#                         conflicts.append(my_action)
#                         break

#             for conflict in conflicts:
#                 conflict : ObservationAction
#                 conflict_req = MeasurementRequest.from_dict(conflict.measurement_req)

#                 # remove conflict from measurement plan
#                 my_measurements.remove(conflict)

#                 # add conflict to list of ignored requests
#                 if conflict_req not in self.ignored_reqs: self.ignored_reqs.append(conflict_req)
                
#         # add any new requrests to the plan
#         considered_reqs = [MeasurementRequest.from_dict(action.measurement_req)
#                            for action in my_measurements
#                            if isinstance(action, ObservationAction)]
#         considered_reqs.extend([req 
#                                 for req, _ in self.completed_requests
#                                 if req not in considered_reqs])
#         inconsidered_req = [req 
#                             for req in self.known_reqs
#                             if req not in considered_reqs
#                             and req not in self.ignored_reqs]
        
#         available_reqs = []
#         for req in inconsidered_req:
#             req : MeasurementRequest
#             for instrument in self.access_times[req.id]:
#                 if self.access_times[req.id][instrument]:
#                     # measurement request is accessible and hasent been considered yet
#                     accessible = False
#                     for subtask_index in range(len(req.observation_groups)):                        
#                         main_instrument, _ = req.observation_groups[subtask_index]
#                         if main_instrument == instrument:
#                             available_reqs.append((req, subtask_index))
#                             accessible = True
#                             break
                    
#                     if not accessible and req not in self.ignored_reqs:
#                         # gp is not accessible by agent; add to list of ignored requests
#                         self.ignored_reqs.append(req)

#                 elif req not in self.ignored_reqs:
#                     # gp is not accessible by agent; add to list of ignored requests
#                     self.ignored_reqs.append(req)

#         planned_reqs = [(MeasurementRequest.from_dict(action.measurement_req), action.subtask_index)
#                         for action in my_measurements]

#         if isinstance(state, SatelliteAgentState):
#             for req, subtask_index in available_reqs:
#                 req : MeasurementRequest
#                 if req.id not in self.access_times:
#                     continue
#                 if (req, subtask_index) in planned_reqs:
#                     continue
                
#                 # get access times for all available measurements
#                 main_measurement, _ = req.observation_groups[subtask_index]  
#                 access_times = [t_img for t_img in self.access_times[req.id][main_measurement]]

#                 # remove access times that overlap with existing measurements
#                 for action in my_measurements:
#                     action : ObservationAction
#                     access_times = [t_img 
#                                     for t_img in access_times
#                                     if not (action.t_start < t_img < action.t_end)]
                    
#                     if not access_times:
#                         break

#                 while access_times:
#                     t_img = access_times.pop(0)
#                     u_exp = self.utility_func(req.to_dict(), t_img) * synergy_factor(req.to_dict(), subtask_index)

#                     proposed_measurement = ObservationAction(req.to_dict(), subtask_index, main_measurement, u_exp, t_img, t_img+req.duration)
                    
#                     # see if it fits between measurements 
#                     i = -1
#                     for measurement in my_measurements:
#                         measurement : ObservationAction
#                         if measurement.t_start > proposed_measurement.t_end:
#                             i = my_measurements.index(measurement)
#                             break
                    
#                     proposed_path = [action for action in my_measurements]
#                     proposed_path.insert(i, proposed_measurement)

#                     if self.is_observation_path_valid(state, proposed_path, orbitdata):
#                         my_measurements = [action for action in proposed_path]
#                         break

#         else:
#             raise NotImplementedError(f'fifo replanner not yet supported for agents with state of type {type(state)}.')

#         return my_measurements
    
#     @runtime_tracker
#     def _schedule_broadcasts(self, 
#                              state: SimulationAgentState, 
#                              measurements: list, 
#                              orbitdata: dict
#                              ) -> list:
        
#         # initialize broadcasts
#         broadcasts : list = super()._schedule_broadcasts(state,
#                                                          measurements, 
#                                                          orbitdata)
                
#         # schedule performed measurement broadcasts
#         if self.collaboration:

#             # schedule the announcement of the current plan
#             if measurements:
#                 # if there are measurements in plan, create plan broadcast action
#                 path, t_start = self._create_broadcast_path(state.agent_name, orbitdata, state.t)
#                 msg = PlanMessage(state.agent_name, 
#                                 state.agent_name, 
#                                 [action.to_dict() for action in measurements],
#                                 state.t,
#                                 path=path)
                
#                 broadcast_action = BroadcastMessageAction(msg.to_dict(), t_start)
                
#                 # check broadcast start; only add to plan if it's within the planning horizon
#                 if t_start <= self.preplan.t_next:
#                     broadcasts.append(broadcast_action)

#             # schedule the broadcast of each scheduled measurement's completion after it's been performed
#             for measurement in measurements:
#                 measurement : ObservationAction
#                 path, t_start = self._create_broadcast_path(state.agent_name, orbitdata, measurement.t_end)

#                 msg = MeasurementPerformedMessage(state.agent_name, state.agent_name, measurement.to_dict(), path=path)

#                 broadcast_action = BroadcastMessageAction(msg.to_dict(), t_start)
                
#                 # check broadcast start; only add to plan if it's within the planning horizon
#                 if t_start <= self.preplan.t_next:
#                     broadcasts.append(broadcast_action)
           
#             # check which measurements that have been performed and already broadcasted
#             measurements_broadcasted = [action_from_dict(**msg.measurement_action)
#                                         for msg in self.completed_broadcasts 
#                                         if isinstance(msg, MeasurementPerformedMessage)]

#             # search for measurements that have been performed but not yet been broadcasted
#             measurements_to_broadcast = [action for action in self.completed_actions 
#                                         if isinstance(action, ObservationAction)
#                                         and action not in measurements_broadcasted]
            
#             # create a broadcast action for all unbroadcasted measurements
#             for completed_measurement in measurements_to_broadcast:       
#                 completed_measurement : ObservationAction
#                 t_end = max(completed_measurement.t_end, state.t)
#                 path, t_start = self._create_broadcast_path(state.agent_name, orbitdata, t_end)
                
#                 msg = MeasurementPerformedMessage(state.agent_name, state.agent_name, completed_measurement.to_dict())
                
#                 broadcast_action = BroadcastMessageAction(msg.to_dict(), t_start, path=path)
                
#                 # check broadcast start; only add to plan if it's within the planning horizon
#                 if t_start <= self.preplan.t_next:
#                     broadcasts.append(broadcast_action)

#                 assert completed_measurement.t_end <= broadcast_action.t_start
        
#         # sort broadcasts by start time
#         broadcasts.sort(key=lambda a : a.t_start)

#         # return broadcast list
#         return broadcasts