from logging import Logger
from orbitpy.util import Spacecraft

from dmas.utils import runtime_tracker
from dmas.clocks import *
from tqdm import tqdm

from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.planning.preplanners.heuristic import HeuristicInsertionPlanner

from chess3d.agents.states import *
from chess3d.agents.actions import *
from chess3d.agents.science.requests import *
from chess3d.agents.states import SimulationAgentState
from chess3d.agents.orbitdata import OrbitData
from chess3d.messages import *

class EarliestAccessPlanner(HeuristicInsertionPlanner):
    """ Schedules observations based on the earliest feasible access point """

    @runtime_tracker
    def sort_accesses(self, access_times : list, *_) -> list:
        # sort by earliest access
        access_times.sort(key = lambda a: a[3])
        
        # return 
        return access_times