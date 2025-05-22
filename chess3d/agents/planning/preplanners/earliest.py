from logging import Logger
from orbitpy.util import Spacecraft

from dmas.utils import runtime_tracker
from dmas.clocks import *
from tqdm import tqdm

from chess3d.orbitdata import OrbitData
from chess3d.agents.planning.preplanners.heuristic import HeuristicInsertionPlanner

from chess3d.agents.states import *
from chess3d.agents.actions import *
from chess3d.agents.science.requests import *
from chess3d.agents.states import SimulationAgentState
from chess3d.orbitdata import OrbitData
from chess3d.messages import *

class EarliestAccessPlanner(HeuristicInsertionPlanner):
    """ Schedules observations based on the earliest feasible access point """
    @runtime_tracker
    def sort_tasks_by_heuristic(self, tasks, *_):
        return sorted(tasks, key=lambda task: task.accessibility.left)