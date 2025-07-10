from abc import abstractmethod
from typing import Dict
import numpy as np

from dmas.modules import ClockConfig

from chess3d.agents.science.requests import TaskRequest
from chess3d.messages import BusMessage, MeasurementRequestMessage, ObservationResultsMessage
from chess3d.orbitdata import IntervalData, OrbitData
from chess3d.agents.planning.plan import *
from chess3d.agents.planning.planner import AbstractReplanner
from chess3d.agents.planning.tasks import ObservationHistory, ObservationTracker
from chess3d.agents.states import SimulationAgentState
from chess3d.mission import Mission
from chess3d.utils import Interval

class BroadcasterReplanner(AbstractReplanner):
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
        
    def get_broadcast_contents(self,
                               broadcast_action : FutureBroadcastMessageAction,
                               state : SimulationAgentState,
                               observation_history : ObservationHistory,
                               **kwargs
                               ) -> BroadcastMessageAction:
        """  Generates a broadcast message to be sent to other agents """
        # raise NotImplementedError('Broadcast contents generation not yet implemented for this planner.')

        if broadcast_action.broadcast_type == FutureBroadcastTypes.REWARD:
            # raise NotImplementedError('Reward broadcast not yet implemented.')

            # compile latest observations from the observation history
            latest_observations : list[ObservationAction] = self.get_latest_observations(state, observation_history)

            # index by instrument name
            instruments_used : set = {latest_observation['instrument'].lower() 
                                      for latest_observation in latest_observations}
            indexed_observations = {instrument_used: [latest_observation for latest_observation in latest_observations
                                                      if latest_observation['instrument'].lower() == instrument_used]
                                    for instrument_used in instruments_used}

            # create ObservationResultsMessage for each instrument
            msgs = [ObservationResultsMessage(state.agent_name, 
                                              state.agent_name, 
                                              state.to_dict(), 
                                              {}, 
                                              instrument,
                                              state.t,
                                              state.t,
                                              observations
                                              )
                    for instrument, observations in indexed_observations.items()]
            
        elif broadcast_action.broadcast_type == FutureBroadcastTypes.REQUESTS:

            msgs = [MeasurementRequestMessage(state.agent_name, state.agent_name, req.to_dict())
                    for req in self.known_reqs
                    if isinstance(req, TaskRequest)
                    and req.event.t_start <= state.t <= req.event.t_end
                    and req.requester == state.agent_name]
        else:
            raise ValueError(f'`{broadcast_action.broadcast_type}` broadcast type not supported.')

        # construct bus message
        bus_msg = BusMessage(state.agent_name, state.agent_name, [msg.to_dict() for msg in msgs])

        # return bus message broadcast (if not empty)
        return BroadcastMessageAction(bus_msg.to_dict(), broadcast_action.t_start)

    def _calculate_broadcast_opportunities(self, orbitdata: OrbitData, t_start : float, t_end : float) -> dict:
        """ calculates future broadcast times based on inter-agent access opportunities """

        # compile all inter-agent access opportunities
        intervals : list[Interval] = { agent_name : sorted([Interval(max(t_start, t_access_start),min(t_end, t_access_end))
                                                    for t_access_start,t_access_end,_ in data.data
                                                    if isinstance(data, IntervalData)
                                                    and (t_start <= t_access_start <= t_end
                                                         or t_start <= t_access_end <= t_end)], key=lambda x: x.left)
                                        for agent_name,data in orbitdata.isl_data.items()}
        

        # TODO include ground station support

        # collect unique start times of every access opportunity
        return intervals
        
        # IDEA aim to broadcasts during intervals where more agents will listen? 
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


    @abstractmethod
    def get_latest_observations(self, 
                                    state : SimulationAgentState,
                                    observation_history : ObservationHistory
                                    ) -> list:
        """ Returns the latest observations from the observation history """

class OpportunisticBroadcasterReplanner(BroadcasterReplanner):
    def __init__(self, debug = False, logger = None):
        # initialize the base class
        super().__init__(debug, logger)
        
        # initialize parameters
        self.t_prev = 0         # time of last broadcast in [s]    

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
        # update time of last broadcast
        self.t_prev = state.t

        # determine the next acceptable broadcast times
        t_start = state.t
        t_end = current_plan.t_next if isinstance(current_plan, Preplan) else np.Inf

        # gather the access opportunities per agent
        access_opportunities : Dict[str, list] = self._calculate_broadcast_opportunities(orbitdata, t_start, t_end)

        # prepare the broadcast messages
        broadcasts : list[FutureBroadcastMessageAction] = []
        for _, intervals in access_opportunities.items():
            # for each agent, prepare a broadcast message for each access opportunity
            for interval in intervals:
                interval : Interval
                # create a future broadcast message for the next access opportunity
                broadcasts.append(FutureBroadcastMessageAction(FutureBroadcastTypes.REWARD, interval.left))
                broadcasts.append(FutureBroadcastMessageAction(FutureBroadcastTypes.REQUESTS, interval.left))
        
        # retrn modified plan
        return Replan.from_preplan(current_plan, broadcasts, t=state.t)

    def get_latest_observations(self, 
                                state : SimulationAgentState,
                                observation_history : ObservationHistory
                                ) -> list:

        return [observation_tracker.latest_observation
                for _,grid in observation_history.history.items()
                for _, observation_tracker in grid.items()
                if isinstance(observation_tracker, ObservationTracker)
                and observation_tracker.latest_observation is not None
                and observation_tracker.latest_observation['t_end'] <= state.t
                # ONLY include observations performed after the lastest broadcast
                and self.t_prev <= observation_tracker.latest_observation['t_end']
                ]

class PeriodicBroadcasterReplanner(BroadcasterReplanner):
    def __init__(self, period : float = np.Inf, debug = False, logger = None):
        super().__init__(debug, logger)

        # set parameters
        self.period = period

        # initialize attributes
        self.t_next = 0             # time of next broacast
        
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
        
        # update next broadcast time
        self.t_next += self.period if state.t >= self.t_next else 0

        # add future broadcast messages to the plan
        broadcasts = [FutureBroadcastMessageAction(FutureBroadcastTypes.REWARD,self.t_next),
                        FutureBroadcastMessageAction(FutureBroadcastTypes.REQUESTS,self.t_next)]
        
        # return modified plan
        return Replan.from_preplan(current_plan, broadcasts, t=state.t)
    
    def get_latest_observations(self, 
                                    state : SimulationAgentState,
                                    observation_history : ObservationHistory
                                    ) -> list:

        return [observation_tracker.latest_observation
                for _,grid in observation_history.history.items()
                for _, observation_tracker in grid.items()
                if isinstance(observation_tracker, ObservationTracker)
                and observation_tracker.latest_observation is not None
                and observation_tracker.latest_observation['t_end'] <= state.t
                # ONLY include observations that are within the period of the broadcast
                and state.t - self.period <= observation_tracker.latest_observation['t_end']
                ]
        