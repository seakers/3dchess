from logging import DEBUG
import logging
from nodes.planning.consensus.bids import Bid
from nodes.science.reqs import MeasurementRequest
from nodes.states import SimulationAgentState
from nodes.planning.consensus.consensus import AbstractConsensusReplanner
from nodes.planning.consensus.bids import Bid, BidComparisonResults


class ACCBBAReplanner(AbstractConsensusReplanner):
    # TODO not yet implemented

    def check_request_completion(self, 
                                 state: SimulationAgentState, 
                                 results: dict, bundle: list, 
                                 path: list, 
                                 completed_measurements: list, 
                                 level=logging.DEBUG) -> tuple:
        results, bundle, path, changes, rebroadcasts \
            = super().check_request_completion(state, results, bundle, path, completed_measurements, level)
        results : dict; bundle : list; path : list
        changes : list; rebroadcasts : list

        # check if any mutually exclusive tasks have been performed
        task_to_remove = None
        for req, subtask_index, current_bid in bundle:
            req : MeasurementRequest
            
            ## check for all known bids related to the relevant measurement request
            for bid_index in range(len(results[req.id])):
                bid : Bid = results[req.id][bid_index]
                if (bid.performed                                               # the other bid was performed
                    and req.dependency_matrix[subtask_index][bid_index] < 0):   # is mutually exclusive with the bid at hand
                    
                    ## a mutually exclusive bid was performed
                    task_to_remove = (req, subtask_index, current_bid)
                    break   

            if task_to_remove is not None:
                break
        
        if task_to_remove is not None:
            ## a mutually exclusive bid was performed; remove mutually exclusive task from bundle and all subsequent tasks
            expired_index : int = bundle.index(task_to_remove)
            for _ in range(expired_index, len(bundle)):
                # remove from bundle
                measurement_req, subtask_index, current_bid = bundle.pop(expired_index)

                # remove from path
                path.remove((measurement_req, subtask_index, current_bid))

                # reset bid results
                current_bid : Bid; measurement_req : MeasurementRequest
                reset_bid : Bid = current_bid.update(None, BidComparisonResults.RESET, state.t)
                results[measurement_req.id][subtask_index] = reset_bid

                # add to changes and rebroadcast lists
                changes.append(reset_bid)
                rebroadcasts.append(reset_bid)    

        return results, bundle, path, changes, rebroadcasts