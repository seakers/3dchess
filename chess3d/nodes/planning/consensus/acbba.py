# import numpy as np

# from dmas.utils import runtime_tracker

# from nodes.states import *
# from nodes.planning.consensus.bids import UnconstrainedBid
# from nodes.planning.consensus.consensus import AbstractConsensusReplanner
# from nodes.science.reqs import MeasurementRequest



# class ACBBAReplanner(AbstractConsensusReplanner):
    
#     def _generate_bids_from_request(self, req : MeasurementRequest) -> list:
#         return UnconstrainedBid.new_bids_from_request(req, self.parent_name, self.dt_converge)

#     # def planning_phase(self, state: SimulationAgentState, current_plan: list, t_next: float) -> tuple:
#     #     return [], [], [], []

#     def _can_bid(self, 
#                 state : SimulationAgentState, 
#                 results : dict,
#                 req : MeasurementRequest, 
#                 subtask_index : int
#                 ) -> bool:
#         """
#         Checks if an agent has the ability to bid on a measurement task
#         """
#         # check capabilities - TODO: Replace with knowledge graph
#         bid : UnconstrainedBid = results[req.id][subtask_index]
#         if bid.main_measurement not in [instrument for instrument in state.payload]:
#             return False 

#         # check time constraints
#         ## Constraint 1: task must be able to be performed during or after the current time
#         if req.t_end < state.t:
#             return False
        
#         return True

#     @runtime_tracker
#     def calc_imaging_time(  self, 
#                             state : SimulationAgentState, 
#                             path : list,
#                             req : MeasurementRequest, 
#                             subtask_index : int
#                          ) -> float:
#         """
#         Computes the ideal" time when a task in the path would be performed
#         ### Returns
#             - t_img (`float`): earliest available imaging time
#         """
#         # calculate the state of the agent prior to performing the measurement request
#         def is_path_element(path_element) -> bool:
#             path_req, path_subtask_index, _, _ = path_element
#             return path_req == req and path_subtask_index == subtask_index

#         path_element = list(filter(is_path_element, path))
#         if len(path_element) != 1:
#             # path contains more than one measurement of the same request and subtask or does not contain it at all
#             return -1

#         i = path.index(path_element[0])
#         if i == 0:
#             t_prev = state.t
#             prev_state = state.copy()
#         else:
#             prev_req, _, t_img, _ = path[i-1]
#             prev_req : MeasurementRequest
#             t_prev : float = t_img + prev_req.duration

#             if isinstance(state, SatelliteAgentState):
#                 prev_state : SatelliteAgentState = state.propagate(t_prev)
                
#                 prev_state.attitude = [
#                                         prev_state.calc_off_nadir_agle(prev_req),
#                                         0.0,
#                                         0.0
#                                     ]
#             elif isinstance(state, UAVAgentState):
#                 prev_state = state.copy()
#                 prev_state.t = t_prev
                
#                 if isinstance(prev_req, GroundPointMeasurementRequest):
#                     prev_state.pos = prev_req.pos
#                 else:
#                     raise NotImplementedError
#             else:
#                 raise NotImplementedError(f"cannot calculate imaging time for agent states of type {type(state)}")

#         if isinstance(state, SatelliteAgentState):
#             instrument, _ = req.measurement_groups[subtask_index]
#             t_img = -1

#             for t_arrival in [t for t in self.access_times[req.id][instrument] if t >= t_prev] :
#                 th_i = prev_state.attitude[0]
#                 th_j = state.calc_off_nadir_agle(req)
                
#                 if abs(th_j - th_i) / state.max_slew_rate <= t_arrival - t_prev:
#                     return t_arrival

#             return t_img

#         # elif isinstance(state, UAVAgentState):
#         #     pass
        
#         else:
#             raise NotImplementedError(f"cannot calculate imaging time for agent states of type {type(state)}")

#     def _is_bundle_converged(   self, 
#                                 state : SimulationAgentState, 
#                                 results : dict, 
#                                 bundle : list, 
#                                 prev_bundle: list
#                             ) -> bool:
#         """ Checks if the constructed bundle is ready for excution"""
#         return self._compare_bundles(bundle, prev_bundle)

#     def _compare_bundles(self, bundle_1 : list, bundle_2 : list) -> bool:
#         """ Compares two bundles. Returns true if they are equal and false if not. """
#         if len(bundle_1) == len(bundle_2):
#             for req, subtask in bundle_1:
#                 if (req, subtask) not in bundle_2:            
#                     return False
#             return True
#         return False


    # def calc_arrival_times(self, state : SimulationAgentState, req : MeasurementRequest, t_prev : Union[int, float]) -> float:
    #     """
    #     Estimates the quickest arrival time from a starting position to a given final position
    #     """
    #     if isinstance(req, GroundPointMeasurementRequest):
    #         # compute earliest time to the task
    #         if isinstance(state, SatelliteAgentState):
    #             t_imgs = []
    #             lat,lon,_ = req.lat_lon_pos
    #             df : pd.DataFrame = self.orbitdata.get_ground_point_accesses_future(lat, lon, t_prev)

    #             for _, row in df.iterrows():
    #                 t_img = row['time index'] * self.orbitdata.time_step
    #                 dt = t_img - state.t
                
    #                 # propagate state
    #                 propagated_state : SatelliteAgentState = state.propagate(t_img)

    #                 # compute off-nadir angle
    #                 thf = propagated_state.calc_off_nadir_agle(req)
    #                 dth = abs(thf - propagated_state.attitude[0])

    #                 # estimate arrival time using fixed angular rate TODO change to 
    #                 if dt >= dth / state.max_slew_rate: # TODO change maximum angular rate 
    #                     t_imgs.append(t_img)
    #             return t_imgs if len(t_imgs) > 0 else [-1]

    #         elif isinstance(state, UAVAgentState):
    #             dr = np.array(req.pos) - np.array(state.pos)
    #             norm = np.sqrt( dr.dot(dr) )
    #             return [norm / state.max_speed + t_prev]

    #         else:
    #             raise NotImplementedError(f"arrival time estimation for agents of type {self.parent_agent_type} is not yet supported.")

    #     else:
    #         raise NotImplementedError(f"cannot calculate imaging time for measurement requests of type {type(req)}")       
