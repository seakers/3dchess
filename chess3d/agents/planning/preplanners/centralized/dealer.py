import logging
import os
from typing import Dict, List
from abc import abstractmethod

import numpy as np

from datetime import datetime

from orbitpy.util import OrbitState, StateType
from orbitpy.util import Spacecraft

from dmas.modules import ClockConfig
from dmas.utils import runtime_tracker
from dmas.agents import AgentAction

from chess3d.agents import states
from chess3d.agents.actions import BroadcastMessageAction, ObservationAction
from chess3d.agents.planning.plan import Plan, Preplan
from chess3d.agents.planning.preplanners.preplanner import AbstractPreplanner
from chess3d.agents.planning.tasks import GenericObservationTask, SpecificObservationTask
from chess3d.agents.planning.tracker import ObservationHistory
from chess3d.agents.states import SatelliteAgentState, SimulationAgentState
from chess3d.messages import  PlanMessage
from chess3d.mission.mission import Mission
from chess3d.orbitdata import IntervalData, OrbitData
from chess3d.utils import Interval


class DealerPreplanner(AbstractPreplanner):
    """
    A preplanner that generates plans for other agents.
    """
    def __init__(self, 
                 client_orbitdata : Dict[str, OrbitData], 
                 client_specs : Dict[str, object],
                 horizon = np.Inf, 
                 period = np.Inf, 
                 debug = False, 
                 logger = None):
        super().__init__(horizon, period, debug, logger)

        # check parameters
        assert isinstance(client_orbitdata, dict), "Clients must be a dictionary mapping agent names to OrbitData instances."
        assert all(isinstance(client, str) for client in client_orbitdata.keys()), \
            "All keys in clients must be strings representing agent names."
        assert all(isinstance(orbitdata, OrbitData) for orbitdata in client_orbitdata.values()), \
            "All clients must be instances of OrbitData."
        assert all(isinstance(client, str) for client in client_specs.keys()), \
            "All keys in clients must be strings representing agent names."
        assert len(client_orbitdata) == len(client_specs), \
            "Clients and client_specs must have the same number of entries."
        assert all(client in client_specs for client in client_orbitdata), \
            "Clients and client_specs must have the same keys."

        # store client information
        self.client_orbitdata : Dict[str, OrbitData] = {client.lower(): client_orbitdata[client] for client in client_orbitdata}
        self.client_specs : Dict[str, object] = {client.lower(): client_specs[client] for client in client_specs}
        self.cross_track_fovs : Dict[str, Dict[str, float]] = self._collect_client_cross_track_fovs(client_specs)
        self.client_states : Dict[str, SatelliteAgentState] = self.__initiate_client_states(client_orbitdata, client_specs)
        self.plans : Dict[str, Preplan] = {client : Preplan([], t=0.0, horizon=self.horizon, t_next=np.Inf) 
                                           for client in self.client_orbitdata}

    def _collect_client_cross_track_fovs(self, client_specs : Dict[str, Spacecraft]) -> Dict[str, Dict[str, float]]:
        """ get instrument field of view specifications from agent specs object """
        return {client: self._collect_fov_specs(client_specs[client]) 
                for client in client_specs}

    def __initiate_client_states(self, 
                                 client_orbitdata : Dict[str, OrbitData], 
                                 client_specs : Dict[str, Spacecraft]
                                 ) -> Dict[str, SatelliteAgentState]:
        """ initiate client agent states at the start of the simulation """
        states: Dict[str, SatelliteAgentState] = {client_name : SatelliteAgentState(client_name, 
                                                                                    client_specs[client_name].orbitState.to_dict(), 
                                                                                    client_orbitdata[client_name].time_step)
                                                   for client_name in client_orbitdata
                                                }
        # check for cases in which the orbitstate does not match the simulation epoch
        for client_name,state in states.items():
            assert client_orbitdata[client_name].epoch == state.orbit_state['date']['jd'], \
                f"Epoch mismatch between client '{client_name}' orbitdata ({client_orbitdata[client_name].epoch}) and specs ({state.orbit_state['date']['jd']})."
            # TODO assert that both epoch times are in the same format
            # assert client_orbitdata[client_name].epoch_type == state.orbit_state['date']['@type'], \
            #     f"Epoch type mismatch between client '{client_name}' orbitdata ({client_orbitdata[client_name].epoch_type}) and specs ({state.orbit_state['date']['@type']})."

        #TODO adjust/propagate states to simulation start time if needed

        # return states
        return states
    
    def update_percepts(self, 
                        state, 
                        current_plan, 
                        incoming_reqs, 
                        relay_messages, 
                        misc_messages, 
                        completed_actions, 
                        aborted_actions, 
                        pending_actions):
        super().update_percepts(state, current_plan, incoming_reqs, relay_messages, misc_messages, completed_actions, aborted_actions, pending_actions)

        # TODO check if any client broadcasted their state or plan
        
        # TODO if so, update the stored state 

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

        # TODO update client states

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
    
    def _generate_client_plans(self, 
                               state : SimulationAgentState, 
                               specs : object, 
                               clock_config : ClockConfig, 
                               orbitdata : OrbitData, 
                               mission : Mission, 
                               tasks : list, 
                               observation_history : ObservationHistory):
        """
        Generates plans for each agent based on the provided parameters.
        """
        # Outline planning horizon interval
        planning_horizon = Interval(state.t, state.t + self.horizon)

        # get only available tasks
        available_tasks : list[GenericObservationTask] = self.get_available_tasks(tasks, planning_horizon)
        
        # calculate coverage opportunities for tasks
        access_opportunities : Dict[str, int, int, str, tuple] = {client : self.calculate_access_opportunities(state, planning_horizon, client_orbitdata)
                                                                    for client, client_orbitdata in self.client_orbitdata.items()
                                                                }

        # create schedulable tasks from known tasks and future access opportunities
        schedulable_tasks : Dict[str, list[SpecificObservationTask]] = {client : self.create_tasks_from_accesses(available_tasks, client_access_opportunities, self.cross_track_fovs[client], orbitdata)
                                                                        for client, client_access_opportunities in access_opportunities.items()
                                                                        }
        
        # schedule observations for each client
        client_observations : Dict[str, List[ObservationAction]] = self._schedule_client_observations()
        
        # validate observation paths for each client
        for client,observations in client_observations.items():
            max_slew_rate, max_torque = self._collect_agility_specs(self.client_specs[client])
            assert self.is_observation_path_valid(self.client_states[client], observations, max_slew_rate, max_torque, self.client_specs[client]), \
                f'Generated observation path/sequence is not valid. Overlaps or mutually exclusive tasks detected.'
            
        # schedule maneuvers for each client
        client_maneuvers : Dict[str, List[AgentAction]] = {client: self._schedule_maneuvers(self.client_states[client], 
                                                                                            self.client_specs[client], 
                                                                                            client_observations[client],
                                                                                            clock_config,
                                                                                            self.client_orbitdata[client])
                                                          for client in self.client_orbitdata
                                                        }

        # schedule broadcasts for each client
        # TODO implement client broadcast scheduling if needed
        client_broadcasts : Dict[str, List[BroadcastMessageAction]] = {client: []
                                                                      for client in self.client_orbitdata
                                                                    }

        # combine scheduled actions to create plans for each client
        client_plans : Dict[str, List[AgentAction]] = {client: Preplan(client_observations[client], client_maneuvers[client], client_broadcasts[client], 
                                                                       t=state.t, horizon=self.horizon, t_next=state.t+self.horizon).actions
                                                          for client in self.client_orbitdata
                                                        }

        # return plans
        return client_plans

    @abstractmethod
    def _schedule_client_observations(self, *args) -> Dict[str, List[ObservationAction]]:
        """ schedules observations for all clients """
    
    def _schedule_broadcasts(self, state : SimulationAgentState, client_plans : dict, orbitdata : OrbitData):
        """
        Schedules broadcasts to be performed based on the generated plans for each agent.
        """
        broadcasts : list[BroadcastMessageAction] = []
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

        # return sorted broadcasts by broadcast start time
        return sorted(broadcasts, key=lambda x: x.t_start)

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
    
    def _schedule_observations(self, *_) -> list:
        """ Boilerplate method for scheduling observations for dealer agent. """
        return [] # dealer does not schedule its own observations, only its clients'

class TestingDealer(DealerPreplanner):
    """
    A preplanner that generates plans for testing purposes.
    """

    @runtime_tracker
    def _generate_client_plans(self, state, specs, clock_config, orbitdata, mission, tasks, observation_history):
        """
        Generates plans for each agent based on the provided parameters.
        """
        # For testing purposes, just return an generic observation action for each client
        return {client: [ObservationAction('VNIR hyper',
                                           0.0,
                                           state.t+500,
                                           0.0,
                                           )] 
                for client in self.client_orbitdata.keys()}
    