from typing import Dict
import numpy as np

from instrupy.base import Instrument

from dmas.modules import ClockConfig
from dmas.messages import SimulationMessage
import pandas as pd

from chess3d.agents.orbitdata import OrbitData, TimeInterval
from chess3d.agents.planning.plan import *
from chess3d.agents.planning.planner import AbstractReplanner
from chess3d.agents.planning.rewards import GridPoint, RewardGrid
from chess3d.agents.science.requests import MeasurementRequest
from chess3d.agents.states import SimulationAgentState
from chess3d.messages import BusMessage, MeasurementRequestMessage, ObservationPerformedMessage, ObservationResultsMessage, message_from_dict


class BroadcasterReplanner(AbstractReplanner):
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
        
        # return isinstance(current_plan, Preplan)

        # check if a new preplan was just generated
        if isinstance(current_plan, Preplan): return True
            
        # check if the broadcast time has been reached 
        return state.t >= self.t_next

    def generate_plan(self, 
                      state : SimulationAgentState,
                      specs : object,
                      reward_grid : RewardGrid,
                      current_plan : Plan,
                      clock_config : ClockConfig,
                      orbitdata : OrbitData,
                    ) -> Plan:
        
        if self.mode == 'periodic':
            if state.t >= self.t_next:
                # replanning was triggered periodically

                # update next broadcast time
                self.t_next += self.period
                
            #     # generate messages to broadcast
            #     broadcasts = self._schedule_broadcasts(state, reward_grid, orbitdata)
            
            #     # insert wait to trigger next broadcasts
            #     waits = [WaitForMessages(self.t_next, self.t_next)]

            # elif isinstance(current_plan, Preplan):
            #     # replanning was triggered because of a new preplan was generated
                
            #     # do not broadcast yet
            #     broadcasts = []
                
            #     # insert wait to trigger next broadcasts
            #     waits = [WaitForMessages(self.t_next, self.t_next)]                

            # return Replan.from_preplan(current_plan, broadcasts, waits, t=state.t)

            broadcasts = [FutureBroadcastMessageAction(FutureBroadcastTypes.REWARD,self.t_next),
                          FutureBroadcastMessageAction(FutureBroadcastTypes.REQUESTS,self.t_next)]
            return Replan.from_preplan(current_plan, broadcasts, t=state.t)
        else:
            raise NotImplementedError(f'`{self.mode}` information broadcaster not supported.')
    
    def get_broadcast_contents(self,
                               broadcast_action : FutureBroadcastMessageAction,
                               state : SimulationAgentState,
                               reward_grid : RewardGrid,
                               ) -> BroadcastMessageAction:
        
        if broadcast_action.broadcast_type == FutureBroadcastTypes.REWARD:
            latest_observations : list[ObservationAction] = []
            for grid in reward_grid.rewards:
                for target in grid:
                    for instrument,grid_point in target.items():
                        grid_point : GridPoint
                        
                        # collect latest known observation for each ground point
                        if grid_point.observations:
                            observations : list[dict] = list(grid_point.observations)
                            observations.sort(key= lambda a: a['t_img'])
                            latest_observations.append((instrument, observations[-1]))

            instruments_used : set = {instrument for instrument,_ in latest_observations}

            msgs = [ObservationResultsMessage(state.agent_name, 
                                              state.agent_name, 
                                              state.to_dict(), 
                                              {}, 
                                              instrument_used,
                                              [observation_data
                                                for instrument, observation_data in latest_observations
                                                if instrument == instrument_used]
                                              )
                    for instrument_used in instruments_used]
            x = 1
            
        elif broadcast_action.broadcast_type == FutureBroadcastTypes.REQUESTS:
            msgs = [MeasurementRequestMessage(state.agent_name, state.agent_name, req.to_dict())
                    for req in self.known_reqs
                    if isinstance(req, MeasurementRequest)
                    and req.t_start <= state.t <= req.t_end]
        else:
            raise ValueError(f'`{broadcast_action.broadcast_type}` broadcast type not supported.')

        # construct bus message
        bus_msg = BusMessage(state.agent_name, state.agent_name, [msg.to_dict() for msg in msgs])

        # return bus message broadcast (if not empty)
        return BroadcastMessageAction(bus_msg.to_dict(), broadcast_action.t_start)

    def _schedule_broadcasts(self, state : SimulationAgentState, reward_grid : RewardGrid, orbitdata : OrbitData):
        msgs : list[SimulationMessage] = []

        # prepare broadcasts latest known and active requests
        req_msgs = [MeasurementRequestMessage(state.agent_name, state.agent_name, req.to_dict())
                    for req in self.known_reqs
                    if isinstance(req, MeasurementRequest)
                    and req.t_start <= state.t <= req.t_end]
        msgs.extend(req_msgs)
        
        # prepare reward grid info broadcasts
        latest_observations : list[ObservationAction] = []
        for grid in reward_grid.rewards:
            for target in grid:
                for instrument,grid_point in target.items():
                    grid_point : GridPoint
                    
                    # collect latest known observation for each ground point
                    if grid_point.observations:
                        observations = list(grid_point.observations)
                        observations.sort(key= lambda a: a.t_start)
                        latest_observations.append(observations[-1])

        obs_msgs = [ObservationPerformedMessage(state.agent_name, state.agent_name, observation.to_dict())
                    for observation in latest_observations]
        msgs.extend(obs_msgs)

        # construct bus message
        bus_msg = BusMessage(state.agent_name, state.agent_name, [msg.to_dict() for msg in msgs])

        # return bus message broadcast (if not empty)
        return [BroadcastMessageAction(bus_msg.to_dict(), state.t)] if msgs else []

    def __calculate_broadcast_opportunities(self, orbitdata: OrbitData) -> list:
        """ calculates broadcast times that overlap with """
        # compile all inter-agent access opportunities
        intervals : list[TimeInterval] = [TimeInterval(t_start,t_end)
                                          for _,data in orbitdata.isl_data.items()
                                          for t_start,t_end in data.values]
        # TODO include ground station support

        # collect unique start times of every access opportunity
        interval_starts = list({interval.start for interval in intervals})
        interval_starts.sort(reverse=True)

        return interval_starts
        
        # IDEA only broadcast during overlaps? 
        # merge access intervals
        merged_intervals : list[TimeInterval] = []
        for interval in intervals:
            overlaps = [merged_interval for merged_interval in merged_intervals
                        if interval.has_overlap(merged_interval)]
            
            if not overlaps:
                merged_intervals.append(interval)
            
            for overlap in overlaps:
                overlap.merge(interval)

        return [interval.start for interval in merged_intervals]
        