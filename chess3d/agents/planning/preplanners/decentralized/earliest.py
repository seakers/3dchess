from dmas.utils import runtime_tracker

from chess3d.agents.planning.preplanners.decentralized.heuristic import HeuristicInsertionPlanner
from chess3d.agents.planning.tasks import SpecificObservationTask

class EarliestAccessPlanner(HeuristicInsertionPlanner):
    """ Schedules observations based on the earliest feasible access point """
    @runtime_tracker
    def sort_tasks_by_heuristic(self, _, tasks : SpecificObservationTask, *__):
        return sorted(tasks, key=lambda task: task.accessibility.left)