import logging
import math
from typing import Callable
import numpy as np

from dmas.utils import runtime_tracker
from traitlets import Callable

from nodes.states import *
from nodes.planning.consensus.bids import UnconstrainedBid
from nodes.planning.consensus.consensus import AbstractConsensusReplanner
from nodes.science.reqs import MeasurementRequest



class ACBBAReplanner(AbstractConsensusReplanner):
    def __init__(self, utility_func: Callable, max_bundle_size: int = 1, dt_converge: float = 0, logger: logging.Logger = None) -> None:
        super().__init__(utility_func, max_bundle_size, dt_converge, logger)
        self.prev_bundle = []
        self.converged = False

    def _generate_bids_from_request(self, req : MeasurementRequest, state : SimulationAgentState) -> list:
        """ Creages bids from given measurement request """
        return UnconstrainedBid.new_bids_from_request(req, state.agent_name, self.dt_converge)

    def is_converged(self) -> bool:
        """ Checks if consensus has been reached and plans are coverged """       
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
    
    def log_results(self, dsc : str, state : SimulationAgentState, results : dict, level=logging.DEBUG) -> None:
        """
        Logs current results at a given time for debugging purposes

        ### Argumnents:
            - dsc (`str`): description of what is to be logged
            - results (`dict`): results to be logged
            - level (`int`): logging level to be used
        """
        out = f'T{state.t}:\t\'{state.agent_name}\'\n{dsc}\n'
        line = 'Req ID\t\tj\tins\tdep\twinner\tbid\tt_img\tt_update\n'
        out += line 
        for _ in range(len(line)):
            out += '-'
        out += '\n'
        
        for req_id in results:
            req_id : str
            req_id_short = req_id.split('-')[0]

            for bid in results[req_id]:
                bid : UnconstrainedBid
                req : MeasurementRequest = MeasurementRequest.from_dict(bid.req)
                ins, deps = req.measurement_groups[bid.subtask_index]
                line = f'{req_id_short}\t{bid.subtask_index}\t{ins}\t{deps}\t{bid.winner}\t{np.round(bid.winning_bid,3)}\t{np.round(bid.t_img,3)}\t{np.round(bid.t_update,1)}\n'
                out += line

            for _ in range(len(line)):
                out += '--'
            out += '\n'

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
        # calculate the state of the agent prior to performing the measurement request
        path_element = [(path_req, path_subtask_index, _, __ )
                        for path_req, path_subtask_index, _, __ in path
                        if path_req == req and path_subtask_index == subtask_index]
        if len(path_element) != 1:
            # path contains more than one measurement of the same request and subtask or does not contain it at all
            return -1
        
        i = path.index(path_element[0])
        if i == 0:
            t_prev = state.t
            prev_state = state.copy()
        else:
            prev_req, _, t_img, _ = path[i-1]
            prev_req : MeasurementRequest
            t_prev : float = t_img + prev_req.duration

            if isinstance(state, SatelliteAgentState):
                prev_state : SatelliteAgentState = state.propagate(t_prev)
                
                prev_state.attitude = [
                                        prev_state.calc_off_nadir_agle(prev_req),
                                        0.0,
                                        0.0
                                    ]
            elif isinstance(state, UAVAgentState):
                prev_state = state.copy()
                prev_state.t = t_prev
                
                if isinstance(prev_req, GroundPointMeasurementRequest):
                    prev_state.pos = prev_req.pos
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError(f"cannot calculate imaging time for agent states of type {type(state)}")

        # calculate arrival time
        if isinstance(state, SatelliteAgentState):
            instrument, _ = req.measurement_groups[subtask_index]
            t_img = -1

            for t_arrival in [t for t in self.access_times[req.id][instrument] if t >= t_prev] :
                th_i = prev_state.attitude[0]
                th_j = state.calc_off_nadir_agle(req)
                
                if abs(th_j - th_i) / state.max_slew_rate <= t_arrival - t_prev:
                    return t_arrival

            return t_img
        
        else:
            raise NotImplementedError(f"cannot calculate imaging time for agent states of type {type(state)}")

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
