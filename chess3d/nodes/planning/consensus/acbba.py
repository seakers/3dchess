from nodes.states import SimulationAgentState
from nodes.planning.consensus.bids import UnconstrainedBid
from nodes.planning.consensus.consensus import AbstractConsensusReplanner
from nodes.science.reqs import MeasurementRequest


class ACBBAReplanner(AbstractConsensusReplanner):
    
    def _generate_bids_from_request(self, req : MeasurementRequest) -> list:
        return UnconstrainedBid.new_bids_from_request(req, self.parent_name)

    # def planning_phase(self, state: SimulationAgentState, current_plan: list, t_next: float) -> tuple:
    #     return [], [], [], []

    def _can_bid(self, 
                state : SimulationAgentState, 
                results : dict,
                req : MeasurementRequest, 
                subtask_index : int
                ) -> bool:
        """
        Checks if an agent has the ability to bid on a measurement task
        """
        # check capabilities - TODO: Replace with knowledge graph
        bid : UnconstrainedBid = results[req.id][subtask_index]
        if bid.main_measurement not in [instrument for instrument in state.payload]:
            return False 

        # check time constraints
        ## Constraint 1: task must be able to be performed during or after the current time
        if req.t_end < state.t:
            return False
        
        return True