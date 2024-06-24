from logging import Logger
import logging
from typing import Callable
import numpy as np
from traitlets import Callable

from orbitpy.util import Spacecraft

from dmas.utils import runtime_tracker

from chess3d.agents.actions import ObservationAction
from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.planning.planners.consensus.bids import Bid
from chess3d.agents.planning.planners.consensus.consensus import AbstractConsensusReplanner
from chess3d.agents.science.requests import MeasurementRequest
from chess3d.agents.states import MeasurementRequest, SatelliteAgentState, SimulationAgentState


class ACBBAPlanner(AbstractConsensusReplanner):
    def __init__(self, 
                 utility_func: Callable, 
                 max_bundle_size: int = 1, 
                 replan_threshold: int = 1,
                 planning_horizon: float = np.Inf, 
                 logger: Logger = None) -> None:
        super().__init__(utility_func, max_bundle_size, replan_threshold, planning_horizon, logger)
        self.prev_bundle = []

    def _generate_bids_from_request(self, 
                                    req: MeasurementRequest, 
                                    state: SimulationAgentState
                                    ) -> list:
        return [Bid(req.id, observation_type, state.agent_name) 
                for observation_type in req.observation_types]

    def _compare_bundles(self, bundle_1 : list, bundle_2 : list) -> bool:
        """ Compares two bundles. Returns true if they are equal and false if not. """
        if len(bundle_1) == len(bundle_2):
            for req, subtask, bid in bundle_1:
                if (req, subtask, bid) not in bundle_2:            
                    return False
            return True
        return False
    
    # @runtime_tracker
    # def calc_path_bid(
    #                     self, 
    #                     state : SimulationAgentState, 
    #                     specs : object,
    #                     original_results : dict,
    #                     original_path : list, 
    #                     req : MeasurementRequest, 
    #                     subtask_index : int
    #                 ) -> tuple:
    #     state : SimulationAgentState = state.copy()
    #     winning_path = None
    #     winning_path_utility = 0.0
    #     winning_t_img = -1
    #     winning_utility = 0.0

    #     # TODO: Improve runtime efficiency:
    #     for i in range(len(original_path)+1):
    #         # generate possible path
    #         path = [scheduled_obs for scheduled_obs in original_path]
    #         path.insert(i, (req, subtask_index, -1, -1))
            
    #         # self.log_task_sequence('new proposed path', path, level=logging.WARNING)

    #         # recalculate bids for each task in the path if needed
    #         for j in range(i, len(path), 1):
    #             req_i, subtask_j, t_img_prev, _ = path[j]
                    
    #             if j == i:
    #                 # new request and subtask are being added; recalculate bid

    #                 # calculate imaging time
    #                 req_i : MeasurementRequest
    #                 subtask_j : int
    #                 t_img = self.calc_imaging_time(state, specs, path, req_i, subtask_j)

    #                 # calc utility
    #                 params = {"req" : req_i.to_dict(), 
    #                           "subtask_index" : subtask_j, 
    #                           "t_img" : t_img}
    #                 utility = self.utility_func(**params) if t_img >= 0 else np.NINF

    #                 # place bid in path
    #                 path[j] = (req_i, subtask_j, t_img, utility)

    #             else:
    #                 # elements from previous path are being adapted to new path

    #                 ## calculate imaging time
    #                 req_i : MeasurementRequest
    #                 subtask_j : int
    #                 t_img = self.calc_imaging_time(state, specs, path, req_i, subtask_j)

    #                 if abs(t_img - t_img_prev) <= 1e-3:
    #                     # path was unchanged; keep the remaining elements of the previous path                    
    #                     break
    #                 else:
    #                     # calc utility
    #                     params = {"req" : req_i.to_dict(), "subtask_index" : subtask_j, "t_img" : t_img}
    #                     utility = self.utility_func(**params) if t_img >= 0 else np.NINF

    #                     # place bid in path
    #                     path[j] = (req_i, subtask_j, t_img, utility)

    #         # look for path with the best utility
    #         path_utility = self.__sum_path_utility(path)
    #         if path_utility > winning_path_utility:
    #             winning_path = [scheduled_obs for scheduled_obs in path]
    #             winning_path_utility = path_utility
    #             _, _, winning_t_img, winning_utility = path[i]
        
    #     ## replacement strategy
    #     if winning_path is None:
    #         # TODO add replacement strategy
    #         pass
        
    #     # ensure winning path contains desired task 
    #     if winning_path:
    #         assert len(winning_path) == len(original_path) + 1

    #         tasks = [(path_req, path_subtask_index) for path_req, path_subtask_index, _, __ in winning_path]
    #         assert (req, subtask_index) in tasks

    #         tasks_unique = []
    #         for task in tasks:
    #             if task not in tasks_unique: tasks_unique.append(task)
                
    #         assert len(tasks) == len(tasks_unique)
    #         assert len(tasks) == len(winning_path)
    #         assert self.is_task_path_valid(state, winning_path)
    #     else:
    #         x = 1

    #     return winning_path, winning_path_utility, winning_t_img, winning_utility
    
    def __sum_path_utility(self, path : list) -> float:
        """ Gives the total utility of a proposed observations sequence """
        if -1 in [t_img for _,_,t_img,_ in path]:
            return 0.0
        
        return sum([u for _,_,_,u in path])  
    
    def log_results(self, dsc : str, state : SimulationAgentState, results : dict, 
                    level=logging.DEBUG) -> None:
        """
        Logs current results at a given time for debugging purposes

        ### Argumnents:
            - dsc (`str`): description of what is to be logged
            - results (`dict`): results to be logged
            - level (`int`): logging level to be used
        """
        out = f'T{state.t}:\t\'{state.agent_name}\'\n{dsc}\n'
        line = 'Req ID\tins\twinner\tbid\tt_img\tt_update  performed\n'
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

            bids : dict[str,Bid] = results[req_id]
            if all([bid.winner == bid.NONE for _,bid in bids.items()]): continue

            for _,bid in bids.items():
                # if i > n: break

                bid : Bid
                if bid.winner == bid.NONE: continue

                line = f'{req_id_short}  \t{bid.main_measurement}\t{bid.winner}\t{np.round(bid.bid,3)}\t{np.round(bid.t_img,3)}\t{np.round(bid.t_update,1)}\t  {int(bid.performed)}\n'
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
                specs : object,
                results : dict,
                req : MeasurementRequest, 
                main_measurement : str
                ) -> bool:
        """ Checks if an agent has the ability to bid on a measurement task """
        if isinstance(specs, dict):
            payload = specs['instrument']
        elif isinstance(specs, Spacecraft):
            payload = specs.instrument
        else:
            raise ValueError('`specs` type not supported.')

        # check capabilities - TODO: Replace with knowledge graph
        bid : Bid = results[req.id][main_measurement]
        if bid.main_measurement not in [instrument.name for instrument in payload]:
            return False 

        # check time constraints
        ## Constraint 1: task must be able to be performed during or after the current time
        if req.t_end < state.t:
            return False
        
        return True

    @runtime_tracker
    def calc_imaging_time(self, state : SimulationAgentState, specs : object, path : list, req : MeasurementRequest, main_measurement : str, orbitdata : OrbitData) -> float:
        """
        Computes the ideal time when a task in the path would be performed
        
        ### Returns
            - t_img (`float`): earliest available imaging time
        """
        proposed_path = [path_element for path_element in path]
        proposed_path_tasks = [(path_req, path_j) for path_req, path_j,_,_ in path]
        j = proposed_path_tasks.index((req,main_measurement))  
        
        # get access times for all available measurements
        if j > 0:
            _,_,t_prev,_ = path[j-1]
            t_start = max(t_prev, req.t_start)

        else:
            t_start = max(state.t, req.t_start)

        access_times = [t_img*orbitdata.time_step 
                        for t_img,_,_,lat,lon,_,_,_,_,_,instrument,_ in orbitdata.gp_access_data.values
                        if  abs(req.target[0] - lat) <= 1e-3
                        and abs(req.target[1] - lon) <= 1e-3
                        and t_start <= t_img*orbitdata.time_step <= req.t_end
                        and instrument == main_measurement]

        # find earliest time that is allowed
        while access_times:
            t_img = access_times.pop(0)
            u_exp = self.utility_func(req.to_dict(), t_img)
    
            proposed_path[j] = (req, main_measurement, t_img, u_exp)

            if self.is_task_path_valid(state, specs, proposed_path, orbitdata):
                return t_img

        return -1

    def is_task_path_valid(self, state : SimulationAgentState, specs : object, path : list, orbitdata : OrbitData) -> bool:
        if isinstance(state, SatelliteAgentState):
            # check if no suitable time was found for this observation
            if any([t_img < 0.0 for _,_,t_img,_ in path]):
                return False

            # gather list of observation actions
            observations = []
            for req, main_measurement, t_img, _ in path:
                req : MeasurementRequest
                th_imgs = [ th_img
                            for t,_,_,lat,lon,_,th_img,_,_,_,instrument,_ in orbitdata.gp_access_data.values
                            if  abs(req.target[0] - lat) <= 1e-3
                            and abs(req.target[1] - lon) <= 1e-3
                            and abs(t_img - t*orbitdata.time_step) <= 1e-3
                            and instrument == main_measurement
                            ]
                observations.append(ObservationAction(main_measurement, req.target, th_imgs.pop(), t_img))

            return self.is_observation_path_valid(state, specs, observations)
        
        else:
            raise NotImplementedError(f'Check for path validity for agents of type {type(state)} not yet supported.')
