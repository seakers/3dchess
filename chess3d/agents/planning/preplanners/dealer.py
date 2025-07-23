import numpy as np
from abc import abstractmethod

from dmas.modules import ClockConfig
from dmas.utils import runtime_tracker
from dmas.agents import AgentAction

from chess3d.agents.actions import BroadcastMessageAction, ObservationAction
from chess3d.agents.planning.plan import Plan, Preplan
from chess3d.agents.planning.preplanners.preplanner import AbstractPreplanner
from chess3d.agents.planning.tasks import ObservationHistory
from chess3d.agents.states import SimulationAgentState
from chess3d.messages import  PlanMessage
from chess3d.mission import Mission
from chess3d.orbitdata import IntervalData, OrbitData
from chess3d.utils import Interval


class DealerPreplanner(AbstractPreplanner):
    """
    A preplanner that generates plans for other agents.
    """
    def __init__(self, clients : dict, horizon = np.Inf, period = np.Inf, debug = False, logger = None):
        super().__init__(horizon, period, debug, logger)

        # check parameters
        assert isinstance(clients, dict), "Clients must be a dictionary mapping agent names to OrbitData instances."
        assert all(isinstance(client, str) for client in clients.keys()), \
            "All keys in clients must be strings representing agent names."
        assert all(isinstance(client, OrbitData) for client in clients.values()), \
            "All clients must be instances of OrbitData."

        # store clients
        self.clients : list[OrbitData] = {client.lower(): clients[client] for client in clients}

    def _schedule_observations(self, state, specs, clock_config, orbitdata, schedulable_tasks, observation_history):
        """ Boilerplate method for scheduling observations."""
        # does not schedule observations for parent agent
        return []

    @runtime_tracker
    def generate_plan(  self, 
                        state : SimulationAgentState,
                        specs : object,
                        clock_config : ClockConfig,
                        orbitdata : OrbitData,
                        mission : Mission,
                        tasks : list,
                        observation_history : ObservationHistory,
                    ) -> Plan:

        # generate plans for all agents
        client_plans : dict = self._generate_client_plans(state, specs, clock_config, orbitdata, mission, tasks, observation_history)

        # schedule broadcasts to be perfomed
        broadcasts : list = self._schedule_broadcasts(state, client_plans, orbitdata)
        
        # generate plan from actions
        self.plan : Preplan = Preplan(broadcasts, t=state.t, horizon=self.horizon, t_next=state.t+self.period)    
        
        # wait for next planning period to start
        replan : list = self._schedule_periodic_replan(state, self.plan, state.t+self.period)
        self.plan.add_all(replan, t=state.t)

        # return plan and save local copy
        return self.plan.copy()
    
    @abstractmethod
    def _generate_client_plans(self, state, specs, clock_config, orbitdata, mission, tasks, observation_history):
        """
        Generates plans for each agent based on the provided parameters.
        """

    def _schedule_broadcasts(self, state : SimulationAgentState, client_plans : dict, orbitdata : OrbitData):
        """
        Schedules broadcasts to be performed based on the generated plans for each agent.
        """
        broadcasts = []
        for client in client_plans:
            # get access intervals with the client agent within the planning horizon         
            access_intervals : list[Interval] = self._calculate_broadcast_opportunities(client, orbitdata, state.t, state.t + self.period)
            
            # if no access opportunities in this planning horizon, skip scheduling
            if not access_intervals: continue
            
            # get next access interval and calculate broadcast time
            next_access : Interval = access_intervals[0]
            t_broadcast : float = max(next_access.left, state.t)

            # schedule broadcasts for the client
            client_plan : list[AgentAction]= client_plans[client]
            plan_msg = PlanMessage(state.agent_name, client, [action.to_dict() for action in client_plan], state.t)

            # create broadcast action
            plan_broadcast = BroadcastMessageAction(plan_msg.to_dict(), t_broadcast)
            broadcasts.append(plan_broadcast)

        return broadcasts

    def _calculate_broadcast_opportunities(self, client : str, orbitdata: OrbitData, t_start : float, t_end : float) -> dict:
        """ calculates future broadcast times based on inter-agent access opportunities """

        # get access intervals for the client
        data : IntervalData = orbitdata.isl_data.get(client.lower(), None)
            
        # if no data for the client, return empty list
        if data is None or not isinstance(data, IntervalData): return []
        
        # compile and sort access intervals for the desired planning horizon
        intervals : list[Interval] = sorted([Interval(max(t_start, t_access_start),min(t_end, t_access_end))
                                                    for t_access_start,t_access_end,_ in data.data
                                                    if (t_start <= t_access_start <= t_end
                                                         or t_start <= t_access_end <= t_end)])

        # return access intervals
        return intervals if intervals else []

class TestingDealer(DealerPreplanner):
    """
    A preplanner that generates plans for testing purposes.
    """

    @runtime_tracker
    def _generate_client_plans(self, state, specs, clock_config, orbitdata, mission, tasks, observation_history):
        """
        Generates plans for each agent based on the provided parameters.
        """
        # For testing purposes, just return an empty dictionary
        return {client: [ObservationAction('test_observation',
                                           [[0,0,0]],
                                           [],
                                           0.0,
                                           state.t+500,
                                           0.0
                                           )] 
                for client in self.clients.keys()}