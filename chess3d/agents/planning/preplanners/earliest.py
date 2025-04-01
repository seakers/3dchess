from logging import Logger
from orbitpy.util import Spacecraft

from dmas.utils import runtime_tracker
from dmas.clocks import *
from tqdm import tqdm

from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.planning.preplanners.heuristic import HeuristicInsertionPlanner
from chess3d.agents.planning.rewards import RewardGrid
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

    # @runtime_tracker
    # def _schedule_broadcasts(self, 
    #                          state: SimulationAgentState, 
    #                          observations : list, 
    #                          orbitdata: OrbitData) -> list:
        # # schedule measurement request broadcasts 
        # broadcasts : list = super()._schedule_broadcasts(state, observations, orbitdata)

        # # gather observation plan to be sent out
        # plan_out = [action.to_dict()
        #             for action in observations
        #             if isinstance(action,ObservationAction)]

        # # check if broadcasts are enabled 
        # if self.sharing and plan_out:
        #     # find best path for broadcasts
        #     path, t_start = self._create_broadcast_path(state, orbitdata)
                
        #     # share current plan to other agents
        #     if t_start >= 0:
        #         # create plan message
        #         msg = PlanMessage(state.agent_name, state.agent_name, plan_out, state.t, path=path)
                
        #         # add plan broadcast to list of broadcasts
        #         broadcasts.append(BroadcastMessageAction(msg.to_dict(),t_start))

        #     # add action performance broadcast to plan based on their completion
        #     for action_dict in tqdm(plan_out, 
        #             desc=f'{state.agent_name}-PLANNER: Pre-Scheduling Broadcasts', 
        #             leave=False):
                
        #         # calculate broadcast start time
        #         if action_dict['t_end'] > t_start:
        #             path, t_start = self._create_broadcast_path(state, orbitdata, action_dict['t_end'])
                    
        #         # check broadcast feasibility
        #         if t_start < 0: continue
                
        #         action_dict['status'] = AgentAction.COMPLETED
        #         # t_start = action_dict['t_end'] # TODO temp solution
        #         msg = ObservationPerformedMessage(state.agent_name, state.agent_name, action_dict)
        #         if t_start >= 0: broadcasts.append(BroadcastMessageAction(msg.to_dict(),t_start))

        # # return broadcast plan
        # return broadcasts
    