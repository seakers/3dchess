import logging
import time
from typing import Any, Callable
import pandas as pd

from dmas.utils import runtime_tracker
from dmas.clocks import *

from nodes.planning.consensus.bids import Bid
from nodes.planning.replanners import AbstractReplanner
from nodes.science.utility import *
from nodes.orbitdata import OrbitData
from nodes.science.reqs import *
from nodes.states import *
from messages import *


class AbstractConsensusReplanner(AbstractReplanner):
    def __init__(   self,
                    parent_name : str,
                    utility_func : Callable[[], Any],
                    max_bundle_size : int = 3,
                    dt_converge : float = 0.0
                    ) -> None:
        super().__init__(utility_func)

        self.parent_name = parent_name
        self.max_bundle_size = max_bundle_size
        self.dt_converge = dt_converge

        self.bundle = []
        self.path = []
        self.results = {}

        self.pre_plan = []
        self.pre_path = []

        self.rebroadcasts = []
        self.converged = True

    def get_parent_name(self) -> str:
        return self.parent_name

    @runtime_tracker
    def needs_replanning(self, 
                        state: SimulationAgentState, 
                        current_plan: list, 
                        performed_actions: list, 
                        incoming_reqs: list, 
                        generated_reqs: list, 
                        misc_messages: list, 
                        t_plan: float, 
                        t_next: float, 
                        planning_horizon=np.Inf, 
                        orbitdata: OrbitData = None
                    ) -> bool:

        # update list of known requests
        new_reqs : list = self._update_known_requests(  current_plan, 
                                                        incoming_reqs,
                                                        generated_reqs)

        # update access times for known requests
        self._update_access_times(  state, 
                                    new_reqs, 
                                    performed_actions,
                                    t_plan,
                                    t_next,
                                    planning_horizon,
                                    orbitdata)        

        # compile received bids
        bids_received = self._compile_bids( incoming_reqs, 
                                            generated_reqs, 
                                            misc_messages, 
                                            state.t)
        
        # perform consesus phase
        self.results, self.bundle, self.path, _, self.rebroadcasts = self.consensus_phase(self.results, self.bundle, self.path, state.t, bids_received)

        # replan if relevant changes have been made to the bundle
        if not self.converged:
            x = 1
        if len(self.rebroadcasts) > 0:
            x = 1

        return len(self.rebroadcasts) > 0 or not self.converged

    def _compile_bids(self, incoming_reqs : list, generated_reqs : list, misc_messages : list, t : float) -> list:
        """ Reads incoming messages and requests and checks if bids were received """
        # get bids from misc messages
        bids = [Bid.from_dict(msg.bid) for msg in misc_messages if isinstance(msg, MeasurementBidMessage)]
        
        # check for new requests from incoming requests
        new_reqs = [req for req in incoming_reqs if req.id not in self.results]
        
        #check for new requests from generated requests
        new_reqs.extend([req for req in generated_reqs if req.id not in self.results and req not in new_reqs])
        
        # generate bids for new requests
        for new_req in new_reqs:
            new_req : MeasurementRequest
            req_bids : list = self._generate_bids_from_request(new_req)
            bids.extend(req_bids)

        return bids

    def replan( self, 
                state: SimulationAgentState,
                original_plan: list, 
                performed_actions: list, 
                incoming_reqs: list, 
                generated_reqs: list, 
                misc_messages: list, 
                t_plan: float, 
                t_next: float, 
                clock_config: ClockConfig, 
                orbitdata: OrbitData = None
            ) -> list:
        # reset convergence if needed
        # self.converged = False if self.converged else self.converged

        # save previous bundle for future convergence checks
        _, prev_bundle = self._save_previous_bundle(self.results, self.bundle)
        
        # perform bundle-building phase
        self.results, self.bundle, self.path, planner_changes \
                = self.planning_phase(  state, 
                                        original_plan,
                                        self.results,
                                        self.bundle,
                                        t_plan,
                                        t_next
                )       

        # chose modified plan if planning has converged
        self.converged = self._is_bundle_converged(state, self.results, self.bundle, prev_bundle)
        # plan = [action for action in original_plan] if not self.converged else self._plan_from_path(state, self.path, state.t, clock_config)
        if self.converged:
            dt_converge = 0.0 
        else:
            dt_converge = min([self.results[req.id][subtask_index].dt_converge 
                                + self.results[req.id][subtask_index].t_update 
                                for req, subtask_index in self.bundle
                                ]
                            ) - state.t
        plan = self._plan_from_path(state, self.path, state.t, clock_config, dt_converge)
        
        # add broadcast changes to plan
        broadcast_bids : list = self._compile_broadcast_bids(planner_changes)       
        for bid in broadcast_bids:
            bid : Bid
            msg = MeasurementBidMessage(self.parent_name, self.parent_name, bid.to_dict())
            broadcast_action = BroadcastMessageAction(msg.to_dict(), state.t)
            plan.insert(0, broadcast_action)
            
        # reset broadcast list
        self.rebroadcasts = []
        
        # return plan
        # print('Observations in Original Path:')
        # original_obs_actions = [action for action in original_plan if isinstance(action, MeasurementAction)]
        # for action in original_obs_actions:
        #     print(action.measurement_req['id'].split('-')[0], action.subtask_index)
        # print(len(original_obs_actions))
            
        # print('Observations in Modified Path:')
        # obs_actions = [action for action in plan if isinstance(action, MeasurementAction)]
        # for action in obs_actions:
        #     print(action.measurement_req['id'].split('-')[0], action.subtask_index)
        # print(len(obs_actions))

        # if len(original_obs_actions) != len(obs_actions):
        #     x = 1

        return plan

    def _save_previous_bundle(self, results : dict, bundle : list) -> tuple:
        """ creates a copy of the current bids of a given bundle  """
        prev_results = {}
        prev_bundle = []
        for req, subtask_index in bundle:
            req : MeasurementRequest; subtask_index : int
            prev_bundle.append((req, subtask_index))
            
            if req.id not in prev_results:
                prev_results[req.id] = [None for _ in results[req.id]]

            prev_results[req.id][subtask_index] = results[req.id][subtask_index].copy()
        
        return prev_results, prev_bundle

    @abstractmethod
    def _is_bundle_converged(   self, 
                                state : SimulationAgentState, 
                                results : dict, 
                                bundle : list, 
                                prev_bundle: list
                            ) -> bool:
        """ Checks if the constructed bundle is ready for excution"""
        pass

    def _compile_broadcast_bids(self, planner_changes : list) -> list:        
        """ Compiles changes in bids from consensus and planning phase and returns a list of the most updated bids """
        broadcast_bids = {}

        planner_changes.extend([bid for bid in self.rebroadcasts])
        for new_bid in planner_changes:
            new_bid : Bid

            if new_bid.req_id not in broadcast_bids:
                req : MeasurementRequest = MeasurementRequest.from_dict(new_bid.req)
                broadcast_bids[new_bid.req_id] = [None for _ in req.dependency_matrix]

            current_bid : Bid = broadcast_bids[new_bid.req_id][new_bid.subtask_index]
            
            if (current_bid is None 
                or new_bid.bidder == current_bid.bidder
                or new_bid.t_update >= current_bid.t_update
                ):
                broadcast_bids[new_bid.req_id][new_bid.subtask_index] = new_bid.copy()       

        out = []
        for req_id in broadcast_bids:
            out.extend([bid for bid in broadcast_bids[req_id] if bid is not None])
            
        return out
    
    @abstractmethod
    def _generate_bids_from_request(self, req : MeasurementRequest) -> list:
        """ Creages bids from given measurement request """
        pass

    def _get_available_requests(self, 
                                state : SimulationAgentState, 
                                results : dict, 
                                bundle : list,
                                path : list
                                ) -> list:
        """ Returns a list of known requests that can be performed within the current planning horizon """
        path_reqs = [(req, subtask_index) for req, subtask_index, _, _ in path]
        available = []        

        for req_id in results:
            for subtask_index in range(len(results[req_id])):
                subtaskbid : Bid = results[req_id][subtask_index]; 
                req = MeasurementRequest.from_dict(subtaskbid.req)

                is_accessible = self.__can_access(state, req, subtask_index)
                is_biddable = self._can_bid(state, results, req, subtask_index)
                already_in_bundle = self.__check_if_in_bundle(req, subtask_index, bundle)
                already_in_path = (req, subtask_index) in path_reqs
                already_performed = self.__request_has_been_performed(results, req, subtask_index, state.t)
                
                if (is_accessible 
                    and is_biddable 
                    and not already_in_bundle 
                    and not already_in_path
                    and not already_performed
                    ):
                    available.append((req, subtaskbid.subtask_index))       
                
        return available

    def __can_access(self, state : SimulationAgentState, req : MeasurementRequest, subtask_index : int) -> bool:
        """ Checks if an agent can access the location of a measurement request """
        if isinstance(state, SatelliteAgentState):
            main_instrument, _ = req.measurement_groups[subtask_index]
            t_arrivals = [t for t in self.access_times[req.id][main_instrument] if req.t_start <= t <= req.t_end]

            return len(t_arrivals)
        else:
            # available_reqs = self.known_reqs
            raise NotImplementedError(f"listing of available requests for agents with state of type {type(state)} not yet supported.")

    @abstractmethod
    def _can_bid(self, 
                state : SimulationAgentState, 
                results : dict,
                req : MeasurementRequest, 
                subtask_index : int
                ) -> bool:
        """ Checks if an agent has the ability to bid on a measurement task """
        pass

    def __check_if_in_bundle(   self, 
                                req : MeasurementRequest, 
                                subtask_index : int, 
                                bundle : list
                                ) -> bool:
        for req_i, subtask_index_j in bundle:
            if req_i.id == req.id and subtask_index == subtask_index_j:
                return True
    
        return False

    def __request_has_been_performed(self, results : dict, req : MeasurementRequest, subtask_index : int, t : Union[int, float]) -> bool:
        # check if subtask at hand has been performed
        current_bid : Bid = results[req.id][subtask_index]
        subtask_already_performed = t > current_bid.t_img >= 0 + req.duration and current_bid.winner != Bid.NONE
        if subtask_already_performed or current_bid.performed:
            return True
       
        return False

    """
    -----------------------
        CONSENSUS PHASE
    -----------------------
    """
    def consensus_phase(  
                                self, 
                                results : dict, 
                                bundle : list, 
                                path : list, 
                                t : Union[int, float], 
                                bids_received : list,
                                level : int = logging.DEBUG
                            ) -> None:
        """
        Evaluates incoming bids and updates current results and bundle
        """
        changes = []
        rebroadcasts = []
        
        # compare bids with incoming messages
        results, bundle, path, \
            comp_changes, comp_rebroadcasts = self.compare_results(results, bundle, path, t, bids_received, level)
        changes.extend(comp_changes)
        rebroadcasts.extend(comp_rebroadcasts)
        
        # check for expired tasks
        results, bundle, path, \
            exp_changes, exp_rebroadcasts = self.check_request_end_time(results, bundle, path, t, level)
        changes.extend(exp_changes)
        rebroadcasts.extend(exp_rebroadcasts)

        # check for already performed tasks
        results, bundle, path, \
            done_changes, done_rebroadcasts = self.check_request_completion(results, bundle, path, t, level)
        changes.extend(done_changes)
        rebroadcasts.extend(done_rebroadcasts)

        return results, bundle, path, changes, rebroadcasts

    @runtime_tracker
    def compare_results(
                        self, 
                        results : dict, 
                        bundle : list, 
                        path : list, 
                        t : Union[int, float], 
                        bids_received : list,
                        level=logging.DEBUG
                    ) -> tuple:
        """
        Compares the existing results with any incoming task bids and updates the bundle accordingly

        ### Returns
            - results
            - bundle
            - path
            - changes
        """
        changes = []
        rebroadcasts = []

        for their_bid in bids_received:
            their_bid : Bid            

            # check bids are for new requests
            new_req = their_bid.req_id not in results

            req = MeasurementRequest.from_dict(their_bid.req)
            if new_req:
                # was not aware of this request; add to results as a blank bid
                results[req.id] = self._generate_bids_from_request(req)

                # add to changes broadcast
                my_bid : Bid = results[req.id][0]
                rebroadcasts.append(my_bid)
                                    
            # compare bids
            my_bid : Bid = results[their_bid.req_id][their_bid.subtask_index]
            # self.log(f'comparing bids...\nmine:  {str(my_bid)}\ntheirs: {str(their_bid)}', level=logging.DEBUG)

            broadcast_bid, changed  = my_bid.update(their_bid.to_dict(), t)
            broadcast_bid : Bid; changed : bool

            # self.log(f'\nupdated: {my_bid}\n', level=logging.DEBUG)
            results[their_bid.req_id][their_bid.subtask_index] = my_bid
                
            # if relevant changes were made, add to changes and rebroadcast
            if changed or new_req:
                changed_bid : Bid = broadcast_bid if not new_req else my_bid
                changes.append(changed_bid)

            if broadcast_bid or new_req:                    
                broadcast_bid : Bid = broadcast_bid if not new_req else my_bid
                rebroadcasts.append(broadcast_bid)

            # if outbid for a task in the bundle, release subsequent tasks in bundle and path
            if (
                (req, my_bid.subtask_index) in bundle 
                and my_bid.winner != self.get_parent_name()
                ):
                bid_index = bundle.index((req, my_bid.subtask_index))

                for _ in range(bid_index, len(bundle)):
                    # remove all subsequent tasks from bundle
                    measurement_req, subtask_index = bundle.pop(bid_index)
                    measurement_req : MeasurementRequest
                    path.remove((measurement_req, subtask_index))

                    # if the agent is currently winning this bid, reset results
                    current_bid : Bid = results[measurement_req.id][subtask_index]
                    if current_bid.winner == self.get_parent_name():
                        current_bid.reset(t)
                        results[measurement_req.id][subtask_index] = current_bid

                        rebroadcasts.append(current_bid)
                        changes.append(current_bid)
        
        return results, bundle, path, changes, rebroadcasts

    @runtime_tracker
    def check_request_end_time(self, results : dict, bundle : list, path : list, t : Union[int, float], level=logging.DEBUG) -> tuple:
        """
        Checks if measurement requests have expired and can no longer be performed

        ### Returns
            - results
            - bundle
            - path
            - changes
        """
        changes = []
        rebroadcasts = []
        # release tasks from bundle if t_end has passed
        task_to_remove = None
        for req, subtask_index in bundle:
            req : MeasurementRequest
            if req.t_end - req.duration < t:
                task_to_remove = (req, subtask_index)
                break

        if task_to_remove is not None:
            bundle_index = bundle.index(task_to_remove)
            for _ in range(bundle_index, len(bundle)):
                # remove all subsequent bids from bundle
                measurement_req, subtask_index = bundle.pop(bundle_index)

                # remove bids from path
                path.remove((measurement_req, subtask_index))

                # if the agent is currently winning this bid, reset results
                measurement_req : Bid
                current_bid : Bid = results[measurement_req.id][subtask_index]
                if current_bid.winner == self.get_parent_name():
                    current_bid.reset(t)
                    results[measurement_req.id][subtask_index] = current_bid
                    
                    rebroadcasts.append(current_bid)
                    changes.append(current_bid)

        return results, bundle, path, changes, rebroadcasts

    @runtime_tracker
    def check_request_completion(self, results : dict, bundle : list, path : list, t : Union[int, float], level=logging.DEBUG) -> tuple:
        """
        Checks if a subtask or a mutually exclusive subtask has already been performed 

        ### Returns
            - results
            - bundle
            - path
            - changes
        """

        changes = []
        rebroadcasts = []
        req_to_remove = None
        req_to_reset = None

        for req, subtask_index in bundle:
            req : MeasurementRequest

            # check if bid has been performed 
            subtask_bid : Bid = results[req.id][subtask_index]
            if self.is_bid_completed(req, subtask_bid, t):
                req_to_remove = (req, subtask_index)
                break

            # check if a mutually exclusive bid has been performed
            for subtask_bid in results[req.id]:
                subtask_bid : Bid

                bids : list = results[req.id]
                bid_index = bids.index(subtask_bid)
                bid : Bid = bids[bid_index]

                if self.is_bid_completed(req, bid, t) and req.dependency_matrix[subtask_index][bid_index] < 0:
                    req_to_remove = (req, subtask_index)
                    req_to_reset = (req, subtask_index) 
                    break   

            if req_to_remove is not None:
                break

        if req_to_remove is not None:
            if req_to_reset is not None:
                bundle_index = bundle.index(req_to_remove)
                
                # level=logging.WARNING
                # self.log_results('PRELIMINARY PREVIOUS PERFORMER CHECKED RESULTS', results, level)
                # self.log_task_sequence('bundle', bundle, level)
                # self.log_task_sequence('path', path, level)

                for _ in range(bundle_index, len(bundle)):
                    # remove task from bundle and path
                    req, subtask_index = bundle.pop(bundle_index)
                    path.remove((req, subtask_index))

                    bid : Bid = results[req.id][subtask_index]
                    bid.reset(t)
                    results[req.id][subtask_index] = bid

                    rebroadcasts.append(bid)
                    changes.append(bid)

                    # self.log_results('PRELIMINARY PREVIOUS PERFORMER CHECKED RESULTS', results, level)
                    # self.log_task_sequence('bundle', bundle, level)
                    # self.log_task_sequence('path', path, level)
            else: 
                # remove performed subtask from bundle
                if req_to_remove in bundle:
                    bundle_index = bundle.index(req_to_remove)
                    req, subtask_index = bundle.pop(bundle_index)

                # remove performed subtask from path
                req, subtask_index = req_to_remove
                path_elements = [item for item in path if item[0] == req and item[1] == subtask_index]
                path.remove(path_elements[0])

                # set bid as completed
                bid : Bid = results[req.id][subtask_index]
                bid.performed = True
                results[req.id][subtask_index] = bid

        return results, bundle, path, changes, rebroadcasts

    def is_bid_completed(self, req : MeasurementRequest, bid : Bid, t : float) -> bool:
        """ Checks if a bid has already been completed or not """
        return (bid.t_img >= 0.0 and bid.t_img + req.duration < t) or bid.performed
    
    """
    -----------------------
        PLANNING PHASE
    -----------------------
    """
    @runtime_tracker
    def planning_phase( self, 
                        state : SimulationAgentState, 
                        current_plan : list,
                        results : dict, 
                        bundle : list,
                        t_plan : float,
                        t_next : float
                    ) -> tuple:
        """
        Creates a modified plan from all known requests and current plan
        """
        changes = []

        path = []
        for measurement_action in [action for action in current_plan if isinstance(action, MeasurementAction)]:
            measurement_action : MeasurementAction
            req : MeasurementRequest = MeasurementRequest.from_dict(measurement_action.measurement_req)
            path.append((req, measurement_action.subtask_index, measurement_action.t_start, measurement_action.u_exp))

        if len(bundle) >= self.max_bundle_size:
            # Bundle is full; cannot modify 
            return results, bundle, path, changes

        available_reqs : list = self._get_available_requests(state, results, bundle, path)

        current_bids = {req.id : {} for req, _ in bundle}
        for req, subtask_index in bundle:
            req : MeasurementRequest
            current_bid : UnconstrainedBid = results[req.id][subtask_index]
            current_bids[req.id][subtask_index] = current_bid.copy()

        max_path = [(req, subtask_index, t_img, u_exp) for req, subtask_index, t_img, u_exp in path]; 
        max_path_bids = {req.id : {} for req, _, _, _ in path}
        for req, subtask_index, _, _ in path:
            req : MeasurementRequest
            max_path_bids[req.id][subtask_index] = results[req.id][subtask_index]

        max_req = -1
        while ( len(available_reqs) > 0                     # there are available tasks to be bid on
                and len(bundle) < self.max_bundle_size      # there is space in the bundle
                and max_req is not None                     # there is a request that maximizes utility
            ):   
             # find next best task to put in bundle (greedy)
            max_req = None 
            max_subtask = None
            for req, subtask_index in available_reqs:
                
                # calculate best bid and path for a given request and subtask
                projected_path, projected_path_utility \
                     = self.calc_path_bid(  state, 
                                            results, 
                                            path, 
                                            req, 
                                            subtask_index)

                # check if path was found
                if projected_path is None:
                    continue

                # compare to maximum task
                if (
                    max_req is None or projected_path_utility > max_path_utility
                    ):

                    # check for cualition and mutex satisfaction                    
                    max_path = projected_path
                    max_path_utility = projected_path_utility
                    max_req = req
                    max_subtask = subtask_index
            
            if max_req is not None:
                # max bid found! place task with the best bid in the bundle and the path
                bundle.append((max_req, max_subtask))
                path = max_path

                # update results
                for req, subtask_index in bundle:
                    req : MeasurementRequest
                    subtask_index : int
                    
                    path_elements = [item for item in path if item[0] == req and item[1] == subtask_index]
                    _, _, t_img, u_exp = path_elements[0]

                    old_bid : Bid = results[req.id][subtask_index]
                    new_bid : Bid = old_bid.copy()
                    new_bid.set_bid(u_exp, t_img, state.t)

                    if old_bid != new_bid:
                        changes.append(new_bid.copy())
                        results[req.id][subtask_index] = new_bid

        return results, bundle, path, changes 
   
    @runtime_tracker
    def calc_path_bid(
                        self, 
                        state : SimulationAgentState, 
                        original_results : dict,
                        original_path : list, 
                        req : MeasurementRequest, 
                        subtask_index : int
                    ) -> tuple:
        state : SimulationAgentState = state.copy()
        winning_path = None
        winning_path_utility = 0.0

        # check if the subtask to be added is mutually exclusive with something in the bundle
        for req_i, subtask_j, _, _ in original_path:
            req_i : MeasurementRequest; subtask_j : int
            if req_i.id == req.id:
                if req.dependency_matrix[subtask_j][subtask_index] < 0:
                    return winning_path, winning_path_utility

        # find best placement in path
        # self.log_task_sequence('original path', original_path, level=logging.WARNING)

        ## insert strategy
        for i in range(len(original_path)+1):
            # generate possible path
            path = [scheduled_obs for scheduled_obs in original_path]
            
            path.insert(i, (req, subtask_index, -1, -1))
            # self.log_task_sequence('new proposed path', path, level=logging.WARNING)

            # recalculate bids for each task in the path if needed
            for j in range(len(path)):
                req_i, subtask_j, t_img, u = path[j]

                if j < i:
                    # element from previous path is unchanged; maintain current bid
                    path[j] = (req, subtask_index, t_img, u)
                    
                elif j == i:
                    # new request and subtask are being added; recalculate bid

                    ## calculate imaging time
                    req_i : MeasurementRequest
                    subtask_j : int
                    t_img = self.calc_imaging_time(state, path, req_i, subtask_j)

                    ## calc utility
                    params = {"req" : req_i.to_dict(), "subtask_index" : subtask_j, "t_img" : t_img}
                    utility = self.utility_func(**params) if t_img >= 0 else np.NINF
                    utility *= synergy_factor(**params)

                    ## place bid in path
                    path[j] = (req, subtask_index, t_img, utility)

                else:
                    # elements from previous path are being adapted to new path

                    ## calculate imaging time
                    req_i : MeasurementRequest
                    subtask_j : int
                    t_img = self.calc_imaging_time(state, path, req_i, subtask_j)

                    _, _, t_img_prev, _ = original_path[j - 1]

                    if abs(t_img - t_img_prev) <= 1e-3:
                        # path was unchanged; keep the remaining elements of the previous path                    
                        pass
                    else:
                        # path was changed; set as unfeasible 
                        path[j] = (req, subtask_index, -1, -1)
                    break

            # look for path with the best utility
            path_utility = self.__sum_path_utility(path)
            if path_utility > winning_path_utility:
                winning_path = path
                winning_path_utility = path_utility
        
        ## replacement strategy
        if winning_path is None:
            # TODO add replacement strategy
            pass

        return winning_path, winning_path_utility

    @abstractmethod
    def calc_imaging_time(self, state : SimulationAgentState, path : list, req : MeasurementRequest, subtask_index : int) -> float:
        """
        Computes the ideal" time when a task in the path would be performed
        ### Returns
            - t_img (`float`): earliest available imaging time
        """
        pass 

    def __sum_path_utility(self, path : list) -> float:
        """ Gives the total utility of a proposed observations sequence """
        if -1 in [t_img for _,_,t_img,_ in path]:
            return 0.0
        
        return sum([u for _,_,_,u in path])  
    
    """
    --------------------
    LOGGING AND TEARDOWN
    --------------------
    """
    # @abstractmethod
    # def log_results(self, dsc : str, results : dict, level=logging.DEBUG) -> None:
    #     """
    #     Logs current results at a given time for debugging purposes

    #     ### Argumnents:
    #         - dsc (`str`): description of what is to be logged
    #         - results (`dict`): results to be logged
    #         - level (`int`): logging level to be used
    #     """
    #     pass
    
    def log_task_sequence(self, dsc : str, sequence : list, level=logging.DEBUG) -> None:
        """
        Logs a sequence of tasks at a given time for debugging purposes

        ### Argumnents:
            - dsc (`str`): description of what is to be logged
            - sequence (`list`): list of tasks to be logged
            - level (`int`): logging level to be used
        """
        out = f'\n{dsc} = ['
        for req, subtask_index in sequence:
            req : MeasurementRequest
            subtask_index : int
            split_id = req.id.split('-')
            
            if sequence.index((req, subtask_index)) > 0:
                out += ', '
            out += f'({split_id[0]}, {subtask_index})'
        out += ']\n'
        print(out)    
