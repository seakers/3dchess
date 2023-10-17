from nodes.states import SimulationAgentState
from nodes.planning.consensus.bids import UnconstrainedBid
from nodes.planning.consensus.consensus import AbstractConsensusReplanner
from nodes.science.reqs import MeasurementRequest


class ACBBAReplanner(AbstractConsensusReplanner):
    
    def _generate_bids_from_request(self, req : MeasurementRequest) -> list:
        return UnconstrainedBid.new_bids_from_request(req, self.parent_name)

    def planning_phase(self, state: SimulationAgentState, current_plan: list, t_next: float) -> tuple:
        return [], [], [], []