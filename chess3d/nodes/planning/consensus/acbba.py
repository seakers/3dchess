from nodes.planning.consensus.bids import UnconstrainedBid
from nodes.planning.consensus.consensus import AbstractConsensusReplanner
from nodes.science.reqs import MeasurementRequest


class ACBBAReplanner(AbstractConsensusReplanner):
    
    def _generate_bids_from_request(self, req : MeasurementRequest) -> list:
        return UnconstrainedBid.new_bids_from_request(req, self.get_parent_name())