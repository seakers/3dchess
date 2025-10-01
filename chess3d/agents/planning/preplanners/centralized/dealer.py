from collections import defaultdict
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
import pandas as pd

from chess3d.agents import states
from chess3d.agents.actions import BroadcastMessageAction, ManeuverAction, ObservationAction, WaitForMessages
from chess3d.agents.planning.plan import Plan, Preplan
from chess3d.agents.planning.preplanners.preplanner import AbstractPreplanner
from chess3d.agents.planning.tasks import DefaultMissionTask, GenericObservationTask, SpecificObservationTask
from chess3d.agents.planning.tracker import ObservationHistory
from chess3d.agents.states import SatelliteAgentState, SimulationAgentState
from chess3d.messages import  PlanMessage
from chess3d.mission.mission import Mission
from chess3d.mission.objectives import DefaultMissionObjective
from chess3d.mission.requirements import GridTargetSpatialRequirement, PointTargetSpatialRequirement, SpatialRequirement, TargetListSpatialRequirement
from chess3d.orbitdata import IntervalData, OrbitData
from chess3d.utils import Interval


class DealerPreplanner(AbstractPreplanner):
    """
    A preplanner that generates plans for other agents.
    """
    def __init__(self, 
                 client_orbitdata : Dict[str, OrbitData], 
                 client_specs : Dict[str, object],
                 client_missions : Dict[str, Mission],
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
        assert all(isinstance(specs, Spacecraft) for specs in client_specs.values()), \
            "All client specs must be instances of Spacecraft."
        assert all(isinstance(client, str) for client in client_missions.keys()), \
            "All keys in clients must be strings representing agent names."
        assert all(isinstance(mission, Mission) for mission in client_missions.values()), \
            "All client missions must be instances of Mission."
        assert len(client_orbitdata) == len(client_specs), \
            "Clients and client_specs must have the same number of entries."
        assert len(client_orbitdata) == len(client_missions), \
            "Clients and client_missions must have the same number of entries."
        assert all(client in client_specs for client in client_orbitdata), \
            "Clients and client_specs must have the same keys."
        assert all(client in client_missions for client in client_orbitdata), \
            "Clients and client_missions must have the same keys."

        # store client information
        self.client_orbitdata : Dict[str, OrbitData] = {client.lower(): client_orbitdata[client] for client in client_orbitdata}
        self.client_specs : Dict[str, object] = {client.lower(): client_specs[client] for client in client_specs}
        self.client_missions : Dict[str, Mission] = {client.lower(): client_missions[client] for client in client_missions}
        self.cross_track_fovs : Dict[str, Dict[str, float]] = self._collect_client_cross_track_fovs(client_specs)
        self.client_states : Dict[str, SatelliteAgentState] = self.__initiate_client_states(client_orbitdata, client_specs)
        self.client_plans : Dict[str, Preplan] = {client : Preplan([], t=0.0, horizon=self.horizon, t_next=np.Inf) 
                                           for client in self.client_orbitdata}
        self.client_tasks : Dict[Mission, List[GenericObservationTask]] = self.__generate_default_client_tasks(client_missions, client_orbitdata)

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
        if misc_messages: # update the stored state 
            raise NotImplementedError('Updating of internal knowledge of agent states based on state update messages not yet implemented.')
        
        else: # else, estimate states
            # propagate position and velocity
            self.client_states : Dict[str, SatelliteAgentState] = {client : client_state.propagate(state.t) 
                                                                    for client,client_state in self.client_states.items()}
            
            # check latest action in their plan
            last_actions : Dict[str, list[AgentAction]] = {client : plan.get_next_actions(state.t)
                                                            for client,plan in self.client_plans.items()}

            # update attitude based on latest scheduled action
            if any([len([action for action in actions if not isinstance(action, WaitForMessages)]) > 0 
                    for actions in last_actions.values()]):
                raise NotImplementedError('Estimation of agent\'s actions based on previously scheduled tasks not yet implemented.')


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

        # TODO update client agent states

        # generate plans for all client agents
        client_plans : Dict[str, Preplan] = self._generate_client_plans(state, specs, clock_config, orbitdata, mission, tasks, observation_history)

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
        available_tasks : Dict[Mission, GenericObservationTask] = \
            self._get_available_client_tasks(planning_horizon)
        
        # calculate coverage opportunities for tasks
        access_opportunities : Dict[str, List[ List[ Dict[str, tuple]]]] = \
              self._calculate_client_access_opportunities(planning_horizon)

        # create schedulable tasks from known tasks and future access opportunities
        schedulable_tasks : Dict[str, list[SpecificObservationTask]] = \
              self._collect_schedulable_client_tasks(available_tasks, access_opportunities)
        
        # schedule observations for each client
        client_observations : Dict[str, List[ObservationAction]] = \
              self._schedule_client_observations(state, available_tasks, schedulable_tasks, observation_history)
        
        # validate observation paths for each client
        for client,observations in client_observations.items():
            assert all(isinstance(obs, ObservationAction) for obs in observations), \
                f'All scheduled observations for client {client} must be instances of `ObservationAction`.'
            assert all(obs.task.parent_tasks for obs in observations), \
                f'All scheduled observations for client {client} must have a parent task.'
            assert self.is_observation_path_valid(self.client_states[client], observations, None, None, self.client_specs[client]), \
                f'Generated observation path/sequence is not valid. Overlaps or mutually exclusive tasks detected.'
            
        # schedule maneuvers for each client
        client_maneuvers : Dict[str, List[ManeuverAction]] = self._schedule_client_maneuvers(client_observations, clock_config)
        
        # validate maneuver paths for each client
        for client,maneuvers in client_maneuvers.items():
            assert all(isinstance(maneuver, ManeuverAction) for maneuver in maneuvers), \
                f'All scheduled maneuvers for client {client} must be instances of `ManeuverAction`.'
            max_slew_rate, max_torque = self._collect_agility_specs(self.client_specs[client])
            assert self.is_maneuver_path_valid(self.client_states[client],
                                               self.client_specs[client],
                                               client_observations[client],
                                               maneuvers,
                                               max_slew_rate,
                                               self.cross_track_fovs[client]), \
                f'Generated maneuver path/sequence is not valid. Overlaps or mutually exclusive tasks detected.'

        # schedule broadcasts for each client
        client_broadcasts : Dict[str, List[BroadcastMessageAction]] = self._schedule_client_broadcasts()

        # validate broadcast paths for each client
        for client,broadcasts in client_broadcasts.items():
            assert all(isinstance(broadcast, BroadcastMessageAction) for broadcast in broadcasts), \
                f'All scheduled broadcasts for client {client} must be instances of `BroadcastMessageAction`.'
            # TODO validate broadcast times if needed

        # combine scheduled actions to create plans for each client
        client_plans : Dict[str, Preplan] = {client: Preplan(client_observations[client], 
                                                            client_maneuvers[client], 
                                                            client_broadcasts[client], 
                                                            t=state.t, horizon=self.horizon, t_next=state.t+self.horizon)
                                                for client in self.client_orbitdata
                                            }

        # return plans
        return client_plans

    def __generate_default_client_tasks(self, client_missions : Dict[str, Mission], client_orbitdata : Dict[str, OrbitData]) -> List[GenericObservationTask]:
        """ generate default tasks for all clients based on their missions and orbitdata """

        # map missions to clients
        mission_clients : Dict[Mission, list[str]] = {mission : list({client 
                                                                      for client, m in client_missions.items() 
                                                                      if m == mission}) 
                                                     for mission in client_missions.values()}

        # collect coverage grids for each client
        client_coverage_grids : Dict[str, list[pd.DataFrame]] = {client : orbitdata.grid_data 
                                                                 for client,orbitdata in client_orbitdata.items()}
        
        # validate that all clients with the same mission have the same coverage grids in their orbitdata
        for mission, clients in mission_clients.items():
            if len(clients) > 0:
                # check same number of grids
                assert all([len(client_coverage_grids[clients[0]]) == len(client_coverage_grids[client]) for client in clients]), \
                    f"All clients with the same mission must have the same number of coverage grids in their orbitdata."

                # check same grid values
                for i,grid_i in enumerate(client_coverage_grids[clients[0]]):
                    assert all([grid_i.equals(client_coverage_grids[client][i]) for client in clients]), \
                        f"All clients with the same mission must have the same coverage grids in their orbitdata. Clients {clients} do not."
                    
                # check same mission duration for clients with the same mission
                assert all([client_orbitdata[clients[0]].duration == client_orbitdata[client].duration for client in clients]), \
                    f"All clients with the same mission must have the same mission duration. Clients {clients} do not."

        # map mission durations
        mission_durations : Dict[Mission, float] = {mission : client_orbitdata[clients[0]].duration 
                                                    for mission,clients in mission_clients.items()}

        # map missions to grids
        mission_grids : Dict[Mission, list[pd.DataFrame]] = {mission : client_coverage_grids[clients[0]] if len(clients) > 0 else []
                                                             for mission,clients in mission_clients.items()}

        # initialize list of tasks
        tasks : Dict[Mission, List[GenericObservationTask]] = defaultdict(list)

        # for each mission and targets, generate default tasks
        for mission,grids in mission_grids.items():
            # gather targets for default mission tasks
            objective_targets = { objective for objective in mission
                                 # ignore non-default objectives
                                 if isinstance(objective, DefaultMissionObjective)
                                 }
            for objective in objective_targets:         
                for req in objective:
                    # ignore non-spatial requirements
                    if not isinstance(req, SpatialRequirement): continue
                    
                    elif isinstance(req, PointTargetSpatialRequirement):
                        raise NotImplementedError("Default task creation for `PointTargetSpatialRequirement` is not implemented yet")
                    
                    elif isinstance(req, TargetListSpatialRequirement):
                        raise NotImplementedError("Default task creation for `TargetListSpatialRequirement` is not implemented yet")
                    
                    elif isinstance(req, GridTargetSpatialRequirement):
                        req_targets = [
                            (lat, lon, grid_index, gp_index)
                            for grid in grids
                            for lat,lon,grid_index,gp_index in grid.values
                            if grid_index == req.grid_index and gp_index < req.grid_size
                        ]
                        
                    else: 
                        raise TypeError(f"Unknown spatial requirement type: {type(req)}")
                        
                # create monitoring tasks from each location in this mission objective
                mission_tasks = [DefaultMissionTask(objective.parameter,
                                            location=(lat, lon, grid_index, gp_index),
                                            mission_duration=mission_durations[mission]*24*3600,
                                            objective=objective,
                                            )
                            for lat,lon,grid_index,gp_index in req_targets
                        ]
                
                # add to list of known tasks
                tasks[mission] = mission_tasks

        return tasks

    def _get_available_client_tasks(self, planning_horizon : Interval) -> Dict[Mission, List[GenericObservationTask]]:
        """ get all known and active tasks for all clients within the planning horizon """
        return {mission: [task for task in tasks
                          if isinstance(task, GenericObservationTask)
                          and task.availability.overlaps(planning_horizon)]
                for mission, tasks in self.client_tasks.items()}

    def _calculate_client_access_opportunities(self, planning_horizon : Interval) -> Dict[str, List[ List[ Dict[str, tuple]]]]:
        """ calculates future access opportunities for all clients within the planning horizon """
        return {client : self.calculate_access_opportunities(self.client_states[client], 
                                                             planning_horizon, 
                                                             client_orbitdata)
                for client, client_orbitdata in self.client_orbitdata.items()}

    def _collect_schedulable_client_tasks(self, 
                                          available_tasks : Dict[Mission, List[GenericObservationTask]], 
                                          access_opportunities : dict
                                        ) -> Dict:
        return {client : self.create_tasks_from_accesses(available_tasks[self.client_missions[client]], 
                                                         client_access_opportunities, 
                                                         self.cross_track_fovs[client], 
                                                         self.client_orbitdata[client])
                for client, client_access_opportunities in access_opportunities.items()}

    @abstractmethod
    def _schedule_client_observations(self, 
                                      state : SimulationAgentState, 
                                      available_tasks : Dict[Mission, List[GenericObservationTask]],
                                      schedulable_tasks: Dict[str, List[SpecificObservationTask]], 
                                      observation_history : ObservationHistory
                                    ) -> Dict[str, List[ObservationAction]]:
        """ schedules observations for all clients """        
    
    @runtime_tracker
    def _schedule_client_maneuvers(self, client_observations : Dict[str, List[ObservationAction]], clock_config : ClockConfig) -> Dict[str, List[ManeuverAction]]:
        return {client: self._schedule_maneuvers(self.client_states[client], 
                                                self.client_specs[client], 
                                                client_observations[client],
                                                clock_config,
                                                self.client_orbitdata[client])
                for client in self.client_orbitdata }

    def _schedule_client_broadcasts(self, *_) -> Dict[str, List[BroadcastMessageAction]]:
        """ schedules broadcasts for all clients """
        return {client: [] for client in self.client_orbitdata} # TODO implement client broadcast scheduling if needed

    def _schedule_broadcasts(self, state : SimulationAgentState, client_plans : Dict[str, Preplan], orbitdata : OrbitData):
        """
        Schedules broadcasts to be performed based on the generated plans for each agent.
        """
        broadcasts : list[BroadcastMessageAction] = []
        for client,client_plan in client_plans.items():
            # get access intervals with the client agent within the planning horizon         
            access_intervals : list[Interval] = self._calculate_broadcast_opportunities(client, orbitdata, state.t, state.t + self.period)
            
            # if no access opportunities in this planning horizon, skip scheduling
            if not access_intervals: continue
            
            # get next access interval and calculate broadcast time
            next_access : Interval = access_intervals[0]
            t_broadcast : float = max(next_access.left, state.t)

            # schedule broadcasts for the client
            plan_msg = PlanMessage(state.agent_name, client, [action.to_dict() for action in client_plan.actions], state.t)

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
    