from typing import Any
from dmas.utils import runtime_tracker

from chess3d.agents.planning.decentralized.heuristic import HeuristicInsertionPlanner
from chess3d.agents.planning.observations import ObservationOpportunity

class EarliestAccessPlanner(HeuristicInsertionPlanner):
    """ Schedules observations based on the earliest feasible access point """
    @runtime_tracker
    def _calc_heuristic(self,
                        observation_opportunity : ObservationOpportunity, 
                        *_ : Any
                        ) -> tuple:
        """ Heuristic function to sort observation opportunities by their earliest access time. """
        # return to sort using: earliest start time >> longest duration
        return (
                observation_opportunity.accessibility.left, 
                -observation_opportunity.min_duration, 
                observation_opportunity.get_priority()
                )