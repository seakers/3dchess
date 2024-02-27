import logging
import math
from typing import Callable
import numpy as np

from dmas.utils import runtime_tracker
from traitlets import Callable

from nodes.science.utility import synergy_factor
from nodes.states import *
from nodes.planning.consensus.bids import Bid, UnconstrainedBid
from nodes.planning.consensus.consensus import AbstractConsensusReplanner
from nodes.science.reqs import MeasurementRequest



class ACBBAReplanner(AbstractConsensusReplanner):
    def __init__(self, utility_func: Callable, max_bundle_size: int = 1, dt_converge: float = 0, logger: logging.Logger = None) -> None:
        super().__init__(utility_func, max_bundle_size, dt_converge, logger)
        self.prev_bundle = []
        self.converged = True

    def _generate_bids_from_request(self, req : MeasurementRequest, state : SimulationAgentState) -> list:
        """ Creages bids from given measurement request """
        return UnconstrainedBid.new_bids_from_request(req, state.agent_name, self.dt_converge)

    def is_converged(self) -> bool:
        """ Checks if consensus has been reached and plans are coverged """       
        return True # TODO
        
        self.prev_bundle = [b for b in self.bundle]

        if self.converged:
            self.converged = False
            return True
        else:
            self.converged = self._compare_bundles(self.bundle, self.prev_bundle)
            return False

    def _compare_bundles(self, bundle_1 : list, bundle_2 : list) -> bool:
        """ Compares two bundles. Returns true if they are equal and false if not. """
        if len(bundle_1) == len(bundle_2):
            for req, subtask, bid in bundle_1:
                if (req, subtask, bid) not in bundle_2:            
                    return False
            return True
        return False
    
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
                    path[j] = (req_i, subtask_j, t_img, u)
                    
                elif j == i:
                    # new request and subtask are being added; recalculate bid

                    # calculate imaging time
                    req_i : MeasurementRequest
                    subtask_j : int
                    t_img = self.calc_imaging_time(state, path, req_i, subtask_j)

                    # calc utility
                    params = {"req" : req_i.to_dict(), "subtask_index" : subtask_j, "t_img" : t_img}
                    utility = self.utility_func(**params) * synergy_factor(**params) if t_img >= 0 else np.NINF

                    # place bid in path
                    path[j] = (req_i, subtask_j, t_img, utility)

                else:
                    # elements from previous path are being adapted to new path

                    ## calculate imaging time
                    req_i : MeasurementRequest
                    subtask_j : int
                    t_img = self.calc_imaging_time(state, path, req_i, subtask_j)

                    _, _, t_img_prev, _ = original_path[j - 1]

                    if abs(t_img - t_img_prev) <= 1e-3:
                        # path was unchanged; keep the remaining elements of the previous path                    
                        break
                    else:
                        # calc utility
                        params = {"req" : req_i.to_dict(), "subtask_index" : subtask_j, "t_img" : t_img}
                        utility = self.utility_func(**params) * synergy_factor(**params) if t_img >= 0 else np.NINF

                        # place bid in path
                        path[j] = (req_i, subtask_j, t_img, utility)

            # look for path with the best utility
            path_utility = self.__sum_path_utility(path)
            if path_utility > winning_path_utility:
                winning_path = [scheduled_obs for scheduled_obs in path]
                winning_path_utility = path_utility
        
        ## replacement strategy
        if winning_path is None:
            # TODO add replacement strategy
            pass
        
        # ensure winning path contains desired task 
        if winning_path:
            assert len(winning_path) == len(original_path) + 1

            tasks = [(path_req, path_subtask_index) for path_req, path_subtask_index, _, __ in winning_path]
            assert (req, subtask_index) in tasks

            tasks_unique = []
            for task in tasks:
                if task not in tasks_unique: tasks_unique.append(task)
                
            assert len(tasks) == len(tasks_unique)
            assert len(tasks) == len(winning_path)
            assert self.is_path_valid(state, winning_path)

        return winning_path, winning_path_utility
    
    def __sum_path_utility(self, path : list) -> float:
        """ Gives the total utility of a proposed observations sequence """
        if -1 in [t_img for _,_,t_img,_ in path]:
            return 0.0
        
        return sum([u for _,_,_,u in path])  
    
    def log_results(self, dsc : str, state : SimulationAgentState, results : dict, level=logging.DEBUG) -> None:
        """
        Logs current results at a given time for debugging purposes

        ### Argumnents:
            - dsc (`str`): description of what is to be logged
            - results (`dict`): results to be logged
            - level (`int`): logging level to be used
        """
        out = f'T{state.t}:\t\'{state.agent_name}\'\n{dsc}\n'
        line = 'Req ID\t  j\tins\tdep\twinner\tbid\tt_img\tt_update\n'
        L_LINE = len(line)
        out += line 
        for _ in range(L_LINE + 25):
            out += '='
        out += '\n'
        
        n = 15
        i = 1
        for req_id in results:
            req_id : str
            req_id_short = req_id.split('-')[0]

            bids : list[Bid] = results[req_id]
            # if all([bid.winner == bid.NONE for bid in bids]): continue

            for bid in bids:
                # if i > n: break

                bid : UnconstrainedBid
                # if bid.winner == bid.NONE: continue

                req : MeasurementRequest = MeasurementRequest.from_dict(bid.req)
                ins, deps = req.measurement_groups[bid.subtask_index]
                line = f'{req_id_short}  {bid.subtask_index}\t{ins}\t{deps}\t{bid.winner}\t{np.round(bid.winning_bid,3)}\t{np.round(bid.t_img,3)}\t{np.round(bid.t_update,1)}\n'
                out += line
                i +=1

            for _ in range(L_LINE + 25):
                out += '-'
            out += '\n'

            if i > n:
                out += '\t\t\t...\n'
                for _ in range(L_LINE + 25):
                    out += '-'
                out += '\n'
                break

        print(out)

    def _can_bid(self, 
                state : SimulationAgentState, 
                results : dict,
                req : MeasurementRequest, 
                subtask_index : int
                ) -> bool:
        """ Checks if an agent has the ability to bid on a measurement task """

        # check capabilities - TODO: Replace with knowledge graph
        bid : UnconstrainedBid = results[req.id][subtask_index]
        if bid.main_measurement not in [instrument for instrument in state.payload]:
            return False 

        # check time constraints
        ## Constraint 1: task must be able to be performed during or after the current time
        if req.t_end < state.t:
            return False
        
        return True

    @runtime_tracker
    def calc_imaging_time(self, state : SimulationAgentState, path : list, req : MeasurementRequest, subtask_index : int) -> float:
        """
        Computes the ideal time when a task in the path would be performed
        
        ### Returns
            - t_img (`float`): earliest available imaging time
        """
        main_measurement = req.measurements[subtask_index]
        proposed_path = [path_element for path_element in path]
        proposed_path_tasks = [(path_req, path_j) for path_req, path_j,_,_ in path]
        j = proposed_path_tasks.index((req,subtask_index))  
        
        # get access times for all available measurements
        if j > 0:
            _,_,t_prev,_ = path[j-1]
            t_start = max(t_prev, req.t_start)
            access_times = [t_img 
                            for t_img in self.access_times[req.id][main_measurement]
                            if t_start <= t_img <= req.t_end]

        else:
            t_start = max(state.t, req.t_start)
            access_times = [t_img 
                            for t_img in self.access_times[req.id][main_measurement]
                            if t_start <= t_img <= req.t_end]


        while access_times:
            t_img = access_times.pop(0)
            u_exp = self.utility_func(req.to_dict(), t_img) * synergy_factor(req.to_dict(), subtask_index)
    
            proposed_path[j] = (req, subtask_index, t_img, u_exp)

            if self.is_path_valid(state, proposed_path):
                return t_img

        return -1
        # # calculate the state of the agent prior to performing the measurement request
        # path_element = [(path_req, path_subtask_index, _, __ )
        #                 for path_req, path_subtask_index, _, __ in path
        #                 if path_req == req and path_subtask_index == subtask_index]
        
        # # if len(path_element) != 1:
        # #     # path contains more than one measurement of the same request and subtask or does not contain it at all
        # #     return -1
        
        # i = path.index(path_element[0])
        # if i == 0:
        #     t_prev = state.t
        #     prev_state = state.copy()
        # else:
        #     prev_req, _, t_img, _ = path[i-1]
        #     prev_req : MeasurementRequest
        #     t_prev : float = t_img + prev_req.duration

        #     if isinstance(state, SatelliteAgentState):
        #         prev_state : SatelliteAgentState = state.propagate(t_prev)
                
        #         prev_state.attitude = [
        #                                 prev_state.calc_off_nadir_agle(prev_req),
        #                                 0.0,
        #                                 0.0
        #                             ]
        #     elif isinstance(state, UAVAgentState):
        #         prev_state = state.copy()
        #         prev_state.t = t_prev
                
        #         if isinstance(prev_req, GroundPointMeasurementRequest):
        #             prev_state.pos = prev_req.pos
        #         else:
        #             raise NotImplementedError
        #     else:
        #         raise NotImplementedError(f"cannot calculate imaging time for agent states of type {type(state)}")

        # # calculate arrival time
        # if isinstance(state, SatelliteAgentState):
        #     instrument, _ = req.measurement_groups[subtask_index]

        #     for t_arrival in [t for t in self.access_times[req.id][instrument] if t >= t_prev] :
        #         th_i = prev_state.attitude[0]
        #         state_j : SatelliteAgentState = state.propagate(t_arrival)
        #         th_j = state_j.calc_off_nadir_agle(req)
                
        #         if th_j - th_i / state.max_slew_rate <= t_arrival - t_prev:
        #             return t_arrival

        #     return -1
        
        # else:
        #     raise NotImplementedError(f"cannot calculate imaging time for agent states of type {type(state)}")

    