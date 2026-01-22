
import os
from typing import List
import numpy as np
import pandas as pd

from dmas.modules import ClockConfig
from dmas.utils import runtime_tracker
from tqdm import tqdm

from chess3d.agents.actions import BroadcastMessageAction, WaitForMessages
from chess3d.agents.planning.periodic import AbstractPeriodicPlanner
from chess3d.agents.planning.plan import PeriodicPlan, Plan
from chess3d.agents.planning.tasks import EventObservationTask
from chess3d.agents.science.requests import TaskRequest
from chess3d.agents.states import SimulationAgentState
from chess3d.messages import BusMessage, MeasurementRequestMessage
from chess3d.mission.events import GeophysicalEvent
from chess3d.mission.mission import Mission
from chess3d.mission.objectives import EventDrivenObjective
from chess3d.orbitdata import OrbitData
from chess3d.utils import Interval


class EventAnnouncerPlanner(AbstractPeriodicPlanner):
    def __init__(self, 
                 events_path : str,
                 mission : Mission,
                 debug = False, 
                 logger = None):
        """
        # Event Announcer Planner
        Announces geophysical events to other agents in the mission as they become available.
        
        TODO : expand to announce events at different times, not just when they become available.
        """

        super().__init__(np.Inf, np.Inf, AbstractPeriodicPlanner.OPPORTUNISTIC, debug, logger)

        # validate inputs
        if not os.path.isfile(events_path):
            raise ValueError('`events_path` must point to an existing file.')
        if not isinstance(mission, Mission):
            raise ValueError('`mission` must be of type `Mission`.')


        # load predefined events
        self.events : list[GeophysicalEvent] = self.load_events(events_path)
        self.parent_mission : Mission = mission
    
    def load_events(self, events_path : str) -> pd.DataFrame:        
        events_df : pd.DataFrame = pd.read_csv(events_path)

        events = []
        for _,row in events_df.iterrows():
            # convert event to GeophysicalEvent
            event = GeophysicalEvent(
                row['event type'],
                (row['lat [deg]'], row['lon [deg]'], row.get('grid index', 0), row['gp_index']),
                row['start time [s]'],
                row['duration [s]'],
                row['severity'],
                row['start time [s]'],
                row['id']
            )
            events.append(event)

        return events
    
    @runtime_tracker
    def generate_plan(  self, 
                        state : SimulationAgentState,
                        specs : object,
                        clock_config : ClockConfig,
                        orbitdata : OrbitData,
                        *_
                    ) -> Plan:
        """ Generates a new plan for the agent """            
        # schedule broadcasts to be perfomed
        broadcasts : list = self._schedule_broadcasts(state, [], orbitdata)
        
        # TODO add maneuvers for pointing-dependent transmissions
        # currently assumes omnidirectional antennas

        # generate plan from actions
        self.plan : PeriodicPlan = PeriodicPlan(broadcasts, t=state.t, horizon=self.horizon, t_next=state.t+self.period)    
        
        # wait for next planning period to start
        replan : list = self._schedule_periodic_replan(state, self.plan, state.t + self.period)
        
        # add replan actions to plan
        self.plan.add_all(replan, t=state.t)

        # return plan and save local copy
        return self.plan.copy()
    
    def _schedule_observations(self, *_) -> list:
        return [] # No scheduling, only announcing events
    
    def _schedule_broadcasts(self, state : SimulationAgentState, _, orbitdata : OrbitData, __ = None) -> List[BroadcastMessageAction]:
        # initialize broadcasts from parent planner
        # broadcasts : List[BroadcastMessageAction] = super()._schedule_broadcasts(state, observations, orbitdata, t)
        broadcasts : List[BroadcastMessageAction] = []

        # get list of future events
        future_events : List[GeophysicalEvent] = [event for event in self.events if event.is_available(state.t)]

        # create requests for each event
        task_requests : List[TaskRequest] = []
        for event in tqdm(future_events, 
                          desc=f'{state.agent_name}/PREPLANNER: Generating task request from known events',
                          leave=False):
            # get event objetives from mission
            objectives  : list[EventDrivenObjective] = [objective for objective in self.parent_mission.objectives
                                                        if isinstance(objective, EventDrivenObjective)
                                                        and objective.event_type == event.event_type]

            for objective in objectives:
                objective : EventDrivenObjective
                # create task
                task = EventObservationTask(objective.parameter, event=event, objective=objective)

                # generate task request 
                task_request = TaskRequest(task,
                                            requester = state.agent_name,
                                            mission_name = self.parent_mission.name,
                                            t_req = event.t_start)
                
                # update list of generated requests 
                task_requests.append(task_request)

        # check if no comms links are available
        if len(orbitdata.comms_links.keys()) == 0: 
            # set broadcast time to immediate
            t_broadcast : float = state.t # immediate broadcast if no comms links available

            # initiate broadcasts list 
            task_requests_msgs : List[MeasurementRequestMessage] = []

            # create broadcasts for each future request
            for req in tqdm(task_requests, 
                        desc=f'{state.agent_name}/PREPLANNER: Scheduling broadcasts for generated task requests',
                        leave=False):

                # generate plan message to share any task requests generated
                task_requests_msg = MeasurementRequestMessage(state.agent_name, state.agent_name, req.to_dict())

                # add to list of task request messages
                task_requests_msgs.append(task_requests_msg.to_dict())

            # compile all requests into single broadcast
            bus_broadcast = BusMessage(state.agent_name, state.agent_name, task_requests_msgs)

            # create single broadcast action for all requests
            broadcasts.append(BroadcastMessageAction(bus_broadcast.to_dict(), t_broadcast))

        # initialize set of times when broadcasts are scheduled
        t_access_starts = set()    

        # create broadcasts for each request
        for req in tqdm(task_requests, 
                        desc=f'{state.agent_name}/PREPLANNER: Scheduling broadcasts for generated task requests',
                        leave=False):
            
            # schedule broadcasts to all available agents
            for target in orbitdata.comms_links.keys():
                # get access intervals with the client agent within the planning horizon
                access_intervals : List[Interval] = orbitdata.get_next_agent_accesses(target, req.t_req, include_current=True)

                # collect access start times for future reference
                t_access_starts.update([access.left for access in access_intervals if not access.is_empty()])

                # create broadcast actions for each access interval
                for next_access in access_intervals:
                    # if no access opportunities in this planning horizon, skip scheduling
                    if next_access.is_empty(): continue

                    # get last access interval and calculate broadcast time
                    # t_broadcast : float = max(next_access.left, req.t_req)
                    t_broadcast : float = max(
                                              min(next_access.left + 5*self.EPS,    # give buffer time for access to start
                                                  next_access.right),               # ensure broadcast is before access ends
                                            state.t)                                # ensure broadcast is not in the past

                    # generate plan message to share any task requests generated
                    task_requests_msg = MeasurementRequestMessage(state.agent_name, state.agent_name, req.to_dict())

                    # create broadcast action and add to client broadcast list
                    broadcast = BroadcastMessageAction(task_requests_msg.to_dict(), t_broadcast)

                    broadcasts.append(broadcast)

        # connection waits; allows for messages to be received right after access start times
        waits = [WaitForMessages(t_access_start, t_access_start) for t_access_start in t_access_starts]
        broadcasts.extend(waits)

        return broadcasts