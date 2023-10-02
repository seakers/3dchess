import logging
import time
from typing import Any, Callable

from dmas.utils import runtime_tracker
from dmas.clocks import *

from nodes.planning.consensus.bids import *
from nodes.planning.replanners import AbstractReplanner
from nodes.orbitdata import OrbitData
from nodes.science.reqs import *
from nodes.states import *


class AbstractConsensusReplanner(AbstractReplanner):
    def __init__(   self,
                    parent_name : str,
                    utility_func : Callable[[], Any],
                    max_bundle_size : int = 3
                    ) -> None:
        super().__init__()

        self.parent_name = parent_name
        self.utility_func = utility_func
        self.max_bundle_size = max_bundle_size

        self.path = []
        self.bundle = []
        self.results = {}

        self.performed_requests = []
        self.access_times = {}
        self.known_reqs = []
        self.prev_path = None

    def get_parent_name(self) -> str:
        return self.parent_name

    @runtime_tracker
    def needs_replanning(   self, 
                            state: AbstractAgentState, 
                            current_plan: list, 
                            performed_actions: list, 
                            incoming_reqs: list, 
                            generated_reqs: list, 
                            misc_messages: list, 
                            t_plan: float, 
                            planning_horizon: float, 
                            orbitdata: OrbitData = None
                        ) -> bool:
        pass

    def replan( self, 
                state: AbstractAgentState, 
                current_plan: list, 
                performed_actions: list,
                incoming_reqs: list, 
                generated_reqs: list, 
                misc_messages: list, 
                t_plan: float, 
                planning_horizon: float, 
                clock_config: ClockConfig, 
                orbitdata: OrbitData = None
            ) -> list:
        pass


    """
    -----------------------
        CONSENSUS PHASE
    -----------------------
    """
    # def consensus_phase(  
    #                             self, 
    #                             results : dict, 
    #                             bundle : list, 
    #                             path : list, 
    #                             t : Union[int, float], 
    #                             new_bids : list,
    #                             level : int = logging.DEBUG
    #                         ) -> None:
    #     """
    #     Evaluates incoming bids and updates current results and bundle
    #     """
    #     changes = []
    #     rebroadcasts = []
    #     # self.log_results(f'\n{process_name} - INITIAL RESULTS', results, level)
    #     # self.log_task_sequence('bundle', bundle, level)
    #     # self.log_task_sequence('path', path, level)
        
    #     # compare bids with incoming messages
    #     results, bundle, path, \
    #         comp_changes, comp_rebroadcasts = self.compare_results(results, bundle, path, t, new_bids, level)
    #     changes.extend(comp_changes)
    #     rebroadcasts.extend(comp_rebroadcasts)

    #     # self.log_changes(f'{process_name} - BIDS RECEIVED', new_bids, level)
    #     # self.log_results(f'{process_name} - COMPARED RESULTS', results, level)
    #     # self.log_task_sequence('bundle', bundle, level)
    #     # self.log_task_sequence('path', path, level)
        
    #     # check for expired tasks
    #     results, bundle, path, \
    #         exp_changes, exp_rebroadcasts = self.check_request_end_time(results, bundle, path, t, level)
    #     changes.extend(exp_changes)
    #     rebroadcasts.extend(exp_rebroadcasts)

    #     # self.log_results(f'{process_name} - CHECKED EXPIRATION RESULTS', results, level)
    #     # self.log_task_sequence('bundle', bundle, level)
    #     # self.log_task_sequence('path', path, level)

    #     # check for already performed tasks
    #     results, bundle, path, \
    #         done_changes, done_rebroadcasts = self.check_request_completion(results, bundle, path, t, level)
    #     changes.extend(done_changes)
    #     rebroadcasts.extend(done_rebroadcasts)

    #     # self.log_results(f'{process_name} - CHECKED EXPIRATION RESULTS', results, level)
    #     # self.log_task_sequence('bundle', bundle, level)
    #     # self.log_task_sequence('path', path, level)

    #     return results, bundle, path, changes, rebroadcasts

    # @runtime_tracker
    # def compare_results(
    #                     self, 
    #                     results : dict, 
    #                     bundle : list, 
    #                     path : list, 
    #                     t : Union[int, float], 
    #                     new_bids : list,
    #                     level=logging.DEBUG
    #                 ) -> tuple:
    #     """
    #     Compares the existing results with any incoming task bids and updates the bundle accordingly

    #     ### Returns
    #         - results
    #         - bundle
    #         - path
    #         - changes
    #     """
    #     changes = []
    #     rebroadcasts = []

    #     for their_bid in new_bids:
    #         their_bid : Bid            

    #         # check bids are for new requests
    #         new_req = their_bid.req_id not in results

    #         req = MeasurementRequest.from_dict(their_bid.req)
    #         if new_req:
    #             # was not aware of this request; add to results as a blank bid
    #             results[req.id] = self.generate_bids_from_request(req)

    #             # add to changes broadcast
    #             my_bid : Bid = results[req.id][0]
    #             rebroadcasts.append(my_bid)
                                    
    #         # compare bids
    #         my_bid : Bid = results[their_bid.req_id][their_bid.subtask_index]
    #         # self.log(f'comparing bids...\nmine:  {str(my_bid)}\ntheirs: {str(their_bid)}', level=logging.DEBUG)

    #         broadcast_bid, changed  = my_bid.update(their_bid.to_dict(), t)
    #         broadcast_bid : Bid; changed : bool

    #         # self.log(f'\nupdated: {my_bid}\n', level=logging.DEBUG)
    #         results[their_bid.req_id][their_bid.subtask_index] = my_bid
                
    #         # if relevant changes were made, add to changes and rebroadcast
    #         if changed or new_req:
    #             changed_bid : Bid = broadcast_bid if not new_req else my_bid
    #             changes.append(changed_bid)

    #         if broadcast_bid or new_req:                    
    #             broadcast_bid : Bid = broadcast_bid if not new_req else my_bid
    #             rebroadcasts.append(broadcast_bid)

    #         # if outbid for a task in the bundle, release subsequent tasks in bundle and path
    #         if (
    #             (req, my_bid.subtask_index) in bundle 
    #             and my_bid.winner != self.get_parent_name()
    #             ):
    #             bid_index = bundle.index((req, my_bid.subtask_index))

    #             for _ in range(bid_index, len(bundle)):
    #                 # remove all subsequent tasks from bundle
    #                 measurement_req, subtask_index = bundle.pop(bid_index)
    #                 measurement_req : MeasurementRequest
    #                 path.remove((measurement_req, subtask_index))

    #                 # if the agent is currently winning this bid, reset results
    #                 current_bid : Bid = results[measurement_req.id][subtask_index]
    #                 if current_bid.winner == self.get_parent_name():
    #                     current_bid.reset(t)
    #                     results[measurement_req.id][subtask_index] = current_bid

    #                     rebroadcasts.append(current_bid)
    #                     changes.append(current_bid)
        
    #     return results, bundle, path, changes, rebroadcasts

    # @runtime_tracker
    # def check_request_end_time(self, results : dict, bundle : list, path : list, t : Union[int, float], level=logging.DEBUG) -> tuple:
    #     """
    #     Checks if measurement requests have expired and can no longer be performed

    #     ### Returns
    #         - results
    #         - bundle
    #         - path
    #         - changes
    #     """
    #     changes = []
    #     rebroadcasts = []
    #     # release tasks from bundle if t_end has passed
    #     task_to_remove = None
    #     for req, subtask_index in bundle:
    #         req : MeasurementRequest
    #         if req.t_end - req.duration < t:
    #             task_to_remove = (req, subtask_index)
    #             break

    #     if task_to_remove is not None:
    #         bundle_index = bundle.index(task_to_remove)
    #         for _ in range(bundle_index, len(bundle)):
    #             # remove all subsequent bids from bundle
    #             measurement_req, subtask_index = bundle.pop(bundle_index)

    #             # remove bids from path
    #             path.remove((measurement_req, subtask_index))

    #             # if the agent is currently winning this bid, reset results
    #             measurement_req : Bid
    #             current_bid : Bid = results[measurement_req.id][subtask_index]
    #             if current_bid.winner == self.get_parent_name():
    #                 current_bid.reset(t)
    #                 results[measurement_req.id][subtask_index] = current_bid
                    
    #                 rebroadcasts.append(current_bid)
    #                 changes.append(current_bid)

    #     return results, bundle, path, changes, rebroadcasts

    # @runtime_tracker
    # def check_request_completion(self, results : dict, bundle : list, path : list, t : Union[int, float], level=logging.DEBUG) -> tuple:
    #     """
    #     Checks if a subtask or a mutually exclusive subtask has already been performed 

    #     ### Returns
    #         - results
    #         - bundle
    #         - path
    #         - changes
    #     """

    #     changes = []
    #     rebroadcasts = []
    #     task_to_remove = None
    #     task_to_reset = None
    #     for req, subtask_index in bundle:
    #         req : MeasurementRequest

    #         # check if bid has been performed 
    #         subtask_bid : Bid = results[req.id][subtask_index]
    #         if self.is_bid_completed(req, subtask_bid, t):
    #             task_to_remove = (req, subtask_index)
    #             break

    #         # check if a mutually exclusive bid has been performed
    #         for subtask_bid in results[req.id]:
    #             subtask_bid : Bid

    #             bids : list = results[req.id]
    #             bid_index = bids.index(subtask_bid)
    #             bid : Bid = bids[bid_index]

    #             if self.is_bid_completed(req, bid, t) and req.dependency_matrix[subtask_index][bid_index] < 0:
    #                 task_to_remove = (req, subtask_index)
    #                 task_to_reset = (req, subtask_index) 
    #                 break   

    #         if task_to_remove is not None:
    #             break

    #     if task_to_remove is not None:
    #         if task_to_reset is not None:
    #             bundle_index = bundle.index(task_to_remove)
                
    #             # level=logging.WARNING
    #             # self.log_results('PRELIMINARY PREVIOUS PERFORMER CHECKED RESULTS', results, level)
    #             # self.log_task_sequence('bundle', bundle, level)
    #             # self.log_task_sequence('path', path, level)

    #             for _ in range(bundle_index, len(bundle)):
    #                 # remove task from bundle and path
    #                 req, subtask_index = bundle.pop(bundle_index)
    #                 path.remove((req, subtask_index))

    #                 bid : Bid = results[req.id][subtask_index]
    #                 bid.reset(t)
    #                 results[req.id][subtask_index] = bid

    #                 rebroadcasts.append(bid)
    #                 changes.append(bid)

    #                 # self.log_results('PRELIMINARY PREVIOUS PERFORMER CHECKED RESULTS', results, level)
    #                 # self.log_task_sequence('bundle', bundle, level)
    #                 # self.log_task_sequence('path', path, level)
    #         else: 
    #             # remove performed subtask from bundle and path 
    #             bundle_index = bundle.index(task_to_remove)
    #             req, subtask_index = bundle.pop(bundle_index)
    #             path.remove((req, subtask_index))

    #             # set bid as completed
    #             bid : Bid = results[req.id][subtask_index]
    #             bid.performed = True
    #             results[req.id][subtask_index] = bid

    #     return results, bundle, path, changes, rebroadcasts

    # def is_bid_completed(self, req : MeasurementRequest, bid : Bid, t : float) -> bool:
    #     """
    #     Checks if a bid has been completed or not
    #     """
    #     return (bid.t_img >= 0.0 and bid.t_img + req.duration < t) or bid.performed

    # @abstractmethod
    # def generate_bids_from_request(self, req : MeasurementRequest) -> list:
    #     pass
    
    """
    -----------------------
        PLANNING PHASE
    -----------------------
    """
    # @runtime_tracker
    # def conflict_free(self, path) -> bool:
    #     """ Checks if path is conflict free """

    # @abstractmethod
    # async def planning_phase(self) -> None:
    #     pass


    # def compare_bundles(self, bundle_1 : list, bundle_2 : list) -> bool:
    #     """
    #     Compares two bundles. Returns true if they are equal and false if not.
    #     """
    #     if len(bundle_1) == len(bundle_2):
    #         for req, subtask in bundle_1:
    #             if (req, subtask) not in bundle_2:            
    #                 return False
    #         return True
    #     return False
    
    # def sum_path_utility(self, path : list, bids : dict) -> float:
    #     utility = 0.0
    #     for req, subtask_index in path:
    #         req : MeasurementRequest
    #         bid : Bid = bids[req.id][subtask_index]
    #         utility += bid.own_bid

    #     return utility

    # def get_available_requests( self, 
    #                             state : SimulationAgentState, 
    #                             bundle : list, 
    #                             results : dict, 
    #                             planning_horizon : float = np.Inf
    #                             ) -> list:
    #     """
    #     Checks if there are any requests available to be performed

    #     ### Returns:
    #         - list containing all available and bidable tasks to be performed by the parent agent
    #     """
    #     available = []
    #     for req_id in results:
    #         for subtask_index in range(len(results[req_id])):
    #             subtaskbid : Bid = results[req_id][subtask_index]; 
    #             req = MeasurementRequest.from_dict(subtaskbid.req)

    #             is_biddable = self.can_bid(state, req, subtask_index, results[req_id], planning_horizon) 
    #             already_in_bundle = self.check_if_in_bundle(req, subtask_index, bundle)
    #             already_performed = self.request_has_been_performed(results, req, subtask_index, state.t)
                
    #             if is_biddable and not already_in_bundle and not already_performed:
    #                 available.append((req, subtaskbid.subtask_index))

    #     return available

    # def can_bid(self, 
    #             state : SimulationAgentState, 
    #             req : MeasurementRequest, 
    #             subtask_index : int, 
    #             subtaskbids : list,
    #             planning_horizon : float
    #             ) -> bool:
    #     """
    #     Checks if an agent has the ability to bid on a measurement task
    #     """
    #     # check planning horizon
    #     if state.t + self.planning_horizon < req.t_start:
    #         return False

    #     # check capabilities - TODO: Replace with knowledge graph
    #     subtaskbid : Bid = subtaskbids[subtask_index]
    #     if subtaskbid.main_measurement not in [instrument.name for instrument in self.payload]:
    #         return False 

    #     # check time constraints
    #     ## Constraint 1: task must be able to be performed during or after the current time
    #     if req.t_end < state.t:
    #         return False

    #     elif isinstance(req, GroundPointMeasurementRequest):
    #         if isinstance(state, SatelliteAgentState):
    #             # check if agent can see the request location
    #             lat,lon,_ = req.lat_lon_pos
    #             df : pd.DataFrame = self.orbitdata.get_ground_point_accesses_future(lat, lon, state.t).sort_values(by='time index')
                
    #             can_access = False
    #             if not df.empty:                
    #                 times = df.get('time index')
    #                 for time in times:
    #                     time *= self.orbitdata.time_step 

    #                     if state.t + planning_horizon < time:
    #                         break

    #                     if req.t_start <= time <= req.t_end:
    #                         # there exists an access time before the request's availability ends
    #                         can_access = True
    #                         break
                
    #             if not can_access:
    #                 return False
        
    #     return True

    # def check_if_in_bundle(self, req : MeasurementRequest, subtask_index : int, bundle : list) -> bool:
    #     for req_i, subtask_index_j in bundle:
    #         if req_i.id == req.id and subtask_index == subtask_index_j:
    #             return True
    
    #     return False

    # def request_has_been_performed(self, results : dict, req : MeasurementRequest, subtask_index : int, t : Union[int, float]) -> bool:
    #     # check if subtask at hand has been performed
    #     current_bid : Bid = results[req.id][subtask_index]
    #     subtask_already_performed = t > current_bid.t_img >= 0 + req.duration and current_bid.winner != Bid.NONE
    #     if subtask_already_performed or current_bid.performed:
    #         return True
       
    #     return False

    # def calc_path_bid(
    #                     self, 
    #                     state : SimulationAgentState, 
    #                     original_results : dict,
    #                     original_path : list, 
    #                     req : MeasurementRequest, 
    #                     subtask_index : int
    #                 ) -> tuple:
    #     state : SimulationAgentState = state.copy()
    #     winning_path = None
    #     winning_bids = None
    #     winning_path_utility = 0.0

    #     # check if the subtask is mutually exclusive with something in the bundle
    #     for req_i, subtask_j in original_path:
    #         req_i : MeasurementRequest; subtask_j : int
    #         if req_i.id == req.id:
    #             if req.dependency_matrix[subtask_j][subtask_index] < 0:
    #                 return winning_path, winning_bids, winning_path_utility

    #     # find best placement in path
    #     # self.log_task_sequence('original path', original_path, level=logging.WARNING)
    #     for i in range(len(original_path)+1):
    #         # generate possible path
    #         path = [scheduled_obs for scheduled_obs in original_path]
            
    #         path.insert(i, (req, subtask_index))
    #         # self.log_task_sequence('new proposed path', path, level=logging.WARNING)

    #         # calculate bids for each task in the path
    #         bids = {}
    #         for req_i, subtask_j in path:
    #             # calculate imaging time
    #             req_i : MeasurementRequest
    #             subtask_j : int
    #             t_img = self.calc_imaging_time(state, path, bids, req_i, subtask_j)

    #             # calc utility
    #             params = {"req" : req_i, "subtask_index" : subtask_j, "t_img" : t_img}
    #             utility = self.utility_func(**params) if t_img >= 0 else np.NINF
    #             utility *= synergy_factor(**params)

    #             # create bid
    #             bid : Bid = original_results[req_i.id][subtask_j].copy()
    #             bid.set_bid(utility, t_img, state.t)
                
    #             if req_i.id not in bids:
    #                 bids[req_i.id] = {}    
    #             bids[req_i.id][subtask_j] = bid                

    #         # look for path with the best utility
    #         path_utility = self.sum_path_utility(path, bids)
    #         if path_utility > winning_path_utility:
    #             winning_path = path
    #             winning_bids = bids
    #             winning_path_utility = path_utility

    #     return winning_path, winning_bids, winning_path_utility


    # def calc_imaging_time(self, state : SimulationAgentState, path : list, bids : dict, req : MeasurementRequest, subtask_index : int) -> float:
    #     """
    #     Computes the ideal" time when a task in the path would be performed
    #     ### Returns
    #         - t_img (`float`): earliest available imaging time
    #     """
    #     # calculate the state of the agent prior to performing the measurement request
    #     i = path.index((req, subtask_index))
    #     if i == 0:
    #         t_prev = state.t
    #         prev_state = state.copy()
    #     else:
    #         prev_req, prev_subtask_index = path[i-1]
    #         prev_req : MeasurementRequest; prev_subtask_index : int
    #         bid_prev : Bid = bids[prev_req.id][prev_subtask_index]
    #         t_prev : float = bid_prev.t_img + prev_req.duration

    #         if isinstance(state, SatelliteAgentState):
    #             prev_state : SatelliteAgentState = state.propagate(t_prev)
                
    #             prev_state.attitude = [
    #                                     prev_state.calc_off_nadir_agle(prev_req),
    #                                     0.0,
    #                                     0.0
    #                                 ]
    #         elif isinstance(state, UAVAgentState):
    #             prev_state = state.copy()
    #             prev_state.t = t_prev
                
    #             if isinstance(prev_req, GroundPointMeasurementRequest):
    #                 prev_state.pos = prev_req.pos
    #             else:
    #                 raise NotImplementedError
    #         else:
    #             raise NotImplementedError(f"cannot calculate imaging time for agent states of type {type(state)}")

    #     return self.calc_arrival_times(prev_state, req, t_prev)[0]

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

    """
    --------------------
    LOGGING AND TEARDOWN
    --------------------
    """

    
