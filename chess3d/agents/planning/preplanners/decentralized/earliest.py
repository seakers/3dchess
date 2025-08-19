from dmas.utils import runtime_tracker

from chess3d.agents.planning.preplanners.decentralized.heuristic import HeuristicInsertionPlanner
from chess3d.agents.planning.tasks import SpecificObservationTask

class EarliestAccessPlanner(HeuristicInsertionPlanner):
    """ Schedules observations based on the earliest feasible access point """
    @runtime_tracker
    def _calc_heuristic(self,
                        task : SpecificObservationTask, 
                        *args
                        ) -> tuple:
        """ Heuristic function to sort tasks by their heuristic value. """
        # return to sort using: earliest start time >> longest duration
        return task.accessibility.left, -task.duration_requirements.left 