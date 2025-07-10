from typing import Dict
import numpy as np

from instrupy.base import Instrument

from dmas.modules import ClockConfig
from dmas.messages import SimulationMessage
import pandas as pd

from chess3d.orbitdata import OrbitData
from chess3d.agents.planning.plan import *
from chess3d.agents.planning.planner import AbstractReplanner
# from chess3d.agents.planning.rewards import  RewardGrid
from chess3d.agents.planning.tasks import ObservationHistory
from chess3d.agents.states import SimulationAgentState
from chess3d.messages import BusMessage
from chess3d.mission import Mission
from chess3d.utils import Interval


class AccessDrivenBroadcasterReplanner(AbstractReplanner):
    def __init__(self, debug = False, logger = None):
        super().__init__(debug, logger)

    def needs_planning(self, 
                       state : SimulationAgentState,
                       specs : object,
                       current_plan : Plan,
                       orbitdata : OrbitData
                       ) -> bool:

        # check if a new preplan was just generated
        return isinstance(current_plan, Preplan)
    
    def generate_plan(  self, 
                        state : SimulationAgentState,
                        specs : object,
                        current_plan : Plan,
                        clock_config : ClockConfig,
                        orbitdata : OrbitData,
                        mission : Mission,
                        tasks : list,
                        observation_history : ObservationHistory,
                    ) -> Plan:
        
        # determine the next acceptable broadcast times
        t_start = state.t
        t_end = current_plan.t_next if isinstance(current_plan, Preplan) else np.Inf

        # gather the access opportunities per agent
        access_opportunities : Dict[str, list] = self.__calculate_broadcast_opportunities(orbitdata, t_start, t_end)

    # def _schedule_broadcasts(self, state : SimulationAgentState, reward_grid , orbitdata : OrbitData):
    #     msgs : list[SimulationMessage] = []

    #     # # prepare broadcasts latest known and active requests # <= MOVED TO ABSTRACT PREPLANNER CLASS
    #     # req_msgs = [MeasurementRequestMessage(state.agent_name, state.agent_name, req.to_dict())
    #     #             for req in self.known_reqs
    #     #             if isinstance(req, MeasurementRequest)
    #     #             and req.t_start <= state.t <= req.t_end]
    #     # msgs.extend(req_msgs)
        
    #     # # prepare reward grid info broadcasts
    #     # latest_observations : list[ObservationAction] = []
    #     # for grid in reward_grid.rewards:
    #     #     for target in grid:
    #     #         for _,grid_point in target.items():
    #     #             grid_point : GridPoint
                    
    #     #             # collect latest known observation for each ground point
    #     #             if grid_point.observations:
    #     #                 observations = list(grid_point.observations)
    #     #                 observations.sort(key= lambda a: a['t_img'])
    #     #                 latest_observations.append(observations[-1])

    #     # obs_msgs = [ObservationPerformedMessage(state.agent_name, state.agent_name, observation.to_dict())
    #     #             for observation in latest_observations]
    #     # msgs.extend(obs_msgs)

    #     # construct bus message
    #     bus_msg = BusMessage(state.agent_name, state.agent_name, [msg.to_dict() for msg in msgs])

    #     # return bus message broadcast (if not empty)
    #     return [BroadcastMessageAction(bus_msg.to_dict(), state.t)] if msgs else []

    def __calculate_broadcast_opportunities(self, orbitdata: OrbitData, t_start : float, t_end : float) -> dict:
        """ calculates future broadcast times based on inter-agent access opportunities """

        # compile all inter-agent access opportunities
        intervals : list[Interval] = { agent_name : sorted([Interval(max(t_start, t_access_start),t_access_end)
                                                    for t_access_start,t_access_end in data.values
                                                    if t_start <= t_access_end <= t_end], key=lambda x: x.left)
                                        for agent_name,data in orbitdata.isl_data.items()}
        

        # TODO include ground station support

        # collect unique start times of every access opportunity
        return intervals
        
        # IDEA only broadcast during overlaps? 
        # merge access intervals
        merged_intervals : list[Interval] = []
        for interval in intervals:
            overlaps = [merged_interval for merged_interval in merged_intervals
                        if interval.has_overlap(merged_interval)]
            
            if not overlaps:
                merged_intervals.append(interval)
            
            for overlap in overlaps:
                overlap.merge(interval)

        return [interval.start for interval in merged_intervals]

class PeriodicBroadcasterReplanner(AbstractReplanner):
    def __init__(self, mode : str, period : float = np.Inf, debug = False, logger = None):
        super().__init__(debug, logger)

        # check if valid mode was selected
        if mode not in ['periodic', 'opportunistic']: raise ValueError(f'`mode` of type `{mode}` not supported.')

        # set parameters
        self.mode = mode
        self.period = period

        # initialize attributes
        self.t_next = 0                                                             # time of next broacast
        # self.pending_observations_to_broadcast : set[ObservationAction] = set()     # set of completed observations that have not been broadcasted
        self.broadcast_times = None

    def update_percepts(self, 
                        state, 
                        current_plan, 
                        incoming_reqs, 
                        relay_messages, 
                        misc_messages, 
                        completed_actions, 
                        aborted_actions, 
                        pending_actions):
        super().update_percepts(state, 
                                current_plan, 
                                incoming_reqs, 
                                relay_messages, 
                                misc_messages, 
                                completed_actions, 
                                aborted_actions, 
                                pending_actions)
        
    def needs_planning(self, 
                       state : SimulationAgentState,
                       specs : object,
                       current_plan : Plan,
                       orbitdata : OrbitData
                       ) -> bool:

        # check if a new preplan was just generated
        if isinstance(current_plan, Preplan): return True
            
        # check if the broadcast time has been reached 
        return state.t >= self.t_next

    def generate_plan(  self, 
                        state : SimulationAgentState,
                        specs : object,
                        current_plan : Plan,
                        clock_config : ClockConfig,
                        orbitdata : OrbitData,
                        mission : Mission,
                        tasks : list,
                        observation_history : ObservationHistory,
                    ) -> Plan:
        
        if self.mode == 'periodic':
            if state.t >= self.t_next:
                # replanning was triggered periodically

                # update next broadcast time
                self.t_next += self.period

            broadcasts = [FutureBroadcastMessageAction(FutureBroadcastTypes.REWARD,self.t_next),
                          FutureBroadcastMessageAction(FutureBroadcastTypes.REQUESTS,self.t_next)]
            
            return Replan.from_preplan(current_plan, broadcasts, t=state.t)
        else:
            raise NotImplementedError(f'`{self.mode}` information broadcaster not supported.')
    
    
        