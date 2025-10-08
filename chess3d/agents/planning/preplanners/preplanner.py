from collections import defaultdict
import logging
import numpy as np
from tqdm import tqdm
from abc import abstractmethod

from orbitpy.util import Spacecraft

from dmas.modules import ClockConfig
from dmas.utils import runtime_tracker
from dmas.agents import AgentAction

from chess3d.agents.actions import ObservationAction, WaitForMessages
from chess3d.agents.planning.plan import Plan, Preplan
from chess3d.agents.planning.planner import AbstractPlanner
from chess3d.agents.planning.tasks import GenericObservationTask, SpecificObservationTask
from chess3d.agents.planning.tracker import ObservationHistory
from chess3d.agents.science.requests import TaskRequest
from chess3d.agents.states import SatelliteAgentState, SimulationAgentState
from chess3d.mission.mission import Mission
from chess3d.orbitdata import OrbitData
from chess3d.utils import Interval


class AbstractPreplanner(AbstractPlanner):
    """
    # Preplanner

    Conducts operations planning for an agent at the beginning of a planning period. 
    """
    def __init__(   self, 
                    horizon : float = np.Inf,
                    period : float = np.Inf,
                    # sharing : bool = False,
                    debug : bool = False,
                    logger: logging.Logger = None
                ) -> None:
        """
        ## Preplanner 
        
        Creates an instance of a preplanner class object.

        #### Arguments:
            - horizon (`float`) : planning horizon in seconds [s]
            - period (`float`) : period of replanning in seconds [s]
            - logger (`logging.Logger`) : debugging logger
        """
        # initialize planner
        super().__init__(debug, logger)    

        # set parameters
        self.horizon = horizon                                                      # planning horizon
        self.period = period                                                        # replanning period         
        # self.sharing = sharing                                                      # toggle for sharing plans
        self.plan = Preplan(t=-1,horizon=horizon,t_next=0.0)                        # initialized empty plan
                
        # initialize attributes
        self.pending_reqs_to_broadcast : set[TaskRequest] = set()            # set of observation requests that have not been broadcasted

    @runtime_tracker
    def update_percepts(self, 
                        state : SimulationAgentState,
                        current_plan : Plan,
                        incoming_reqs: list, 
                        relay_messages: list, 
                        misc_messages : list,
                        completed_actions: list,
                        aborted_actions : list,
                        pending_actions : list
                        ) -> None:
        # update percepts
        super().update_percepts(state, incoming_reqs, relay_messages, completed_actions)
    
    @runtime_tracker
    def needs_planning( self, 
                        state : SimulationAgentState,
                        __ : object,
                        current_plan : Plan
                        ) -> bool:
        """ Determines whether a new plan needs to be initalized """    

        if (current_plan.t < 0                  # simulation just started
            or state.t >= current_plan.t_next):    # or periodic planning period has been reached
            
            pending_actions = [action for action in current_plan
                               if action.t_start <= current_plan.t_next]
            
            return not bool(pending_actions)     # no actions left to do before the end of the replanning period 
        return False

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
        
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self._collect_fov_specs(specs)

        # compile agility specifications
        max_slew_rate, max_torque = self._collect_agility_specs(specs)

        # Outline planning horizon interval
        planning_horizon = Interval(state.t, state.t + self.horizon)

        # get only available tasks
        available_tasks : list[GenericObservationTask] = self.get_available_tasks(tasks, planning_horizon)
        
        # calculate coverage opportunities for tasks
        access_opportunities : dict[tuple] = self.calculate_access_opportunities(state, planning_horizon, orbitdata)

        # create schedulable tasks from known tasks and future access opportunities
        schedulable_tasks : list[SpecificObservationTask] = self.create_tasks_from_accesses(available_tasks, access_opportunities, cross_track_fovs, orbitdata)

        # schedule observation tasks
        observations : list = self._schedule_observations(state, specs, clock_config, orbitdata, schedulable_tasks, mission, observation_history)

        assert isinstance(observations, list) and all([isinstance(obs, ObservationAction) for obs in observations]), \
            f'Observation actions not generated correctly. Is of type `{type(observations)}` with elements of type `{type(observations[0])}`.'
        assert self.is_observation_path_valid(state, observations, max_slew_rate, max_torque, specs), \
            f'Generated observation path/sequence is not valid. Overlaps or mutually exclusive tasks detected.'

        # schedule broadcasts to be perfomed
        broadcasts : list = self._schedule_broadcasts(state, observations, orbitdata)

        # generate maneuver and travel actions from measurements
        maneuvers : list = self._schedule_maneuvers(state, specs, observations, clock_config, orbitdata)
        
        # generate plan from actions
        self.plan : Preplan = Preplan(observations, maneuvers, broadcasts, t=state.t, horizon=self.horizon, t_next=state.t+self.period)    
        
        # wait for next planning period to start
        replan : list = self._schedule_periodic_replan(state, self.plan, state.t+self.period)
        self.plan.add_all(replan, t=state.t)

        # return plan and save local copy
        return self.plan.copy()
            
    @runtime_tracker
    def get_available_tasks(self, tasks : list, planning_horizon : Interval) -> list:
        """ Returns a list of tasks that are available at the given time """
        if not isinstance(tasks, list):
            raise ValueError(f'`tasks` needs to be of type `list`. Is of type `{type(tasks)}`.')
        
        # TODO add check for capability of the agent to perform the task?      

        # Check if task is available within the proposed planning horizon
        return [task for task in tasks 
                if isinstance(task, GenericObservationTask)
                and task.availability.overlaps(planning_horizon)]
    
    @runtime_tracker
    def calculate_access_opportunities(self, 
                                       state : SimulationAgentState, 
                                       planning_horizon : Interval,
                                       orbitdata : OrbitData
                                    ) -> dict:
        """ Calculate access opportunities for targets visible in the planning horizon """

        # compile coverage data
        raw_coverage_data : dict = orbitdata.gp_access_data.lookup_interval(planning_horizon.left, planning_horizon.right)

        # initiate access times
        access_opportunities = {}
        
        for i in tqdm(range(len(raw_coverage_data['time [s]'])), 
                        desc=f'{state.agent_name}/PREPLANNER: Compiling access opportunities', 
                        leave=False):
            t_img = raw_coverage_data['time [s]'][i]
            grid_index = raw_coverage_data['grid index'][i]
            gp_index = raw_coverage_data['GP index'][i]
            instrument = raw_coverage_data['instrument'][i]
            look_angle = raw_coverage_data['look angle [deg]'][i]
            
            # initialize dictionaries if needed
            if grid_index not in access_opportunities:
                access_opportunities[grid_index] = {}
                
            if gp_index not in access_opportunities[grid_index]:
                access_opportunities[grid_index][gp_index] = defaultdict(list)

            # compile time interval information 
            found = False
            for interval, t, th in access_opportunities[grid_index][gp_index][instrument]:
                interval : Interval
                t : list
                th : list

                overlap_interval = Interval(t_img - orbitdata.time_step, 
                                            t_img + orbitdata.time_step)
                
                if overlap_interval.overlaps(interval):
                    interval.extend(t_img)
                    t.append(t_img)
                    th.append(look_angle)
                    found = True
                    break      

            if not found:
                access_opportunities[grid_index][gp_index][instrument].append([Interval(t_img, t_img), [t_img], [look_angle]])
                
        # return access times and grid information
        return access_opportunities

    @abstractmethod
    def _schedule_observations(self, state : SimulationAgentState, specs : object, clock_config : ClockConfig, orbitdata : OrbitData, schedulable_tasks : list, mission : Mission, observation_history : ObservationHistory) -> list:
        """ Creates a list of observation actions to be performed by the agent """    

    @abstractmethod
    def _schedule_broadcasts(self, state: SimulationAgentState, observations : list, orbitdata: OrbitData, t : float = None) -> list:
        """ Schedules broadcasts to be done by this agent """
        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Broadcast scheduling for agents of type `{type(state)}` not yet implemented.')
        elif orbitdata is None:
            raise ValueError(f'`orbitdata` required for agents of type `{type(state)}`.')

        # initialize list of broadcasts to be done
        broadcasts = []       

        # # schedule generated measurement request broadcasts
        # if self.sharing and self.pending_reqs_to_broadcast:
        #     # sort requests based on their start time
        #     pending_reqs_to_broadcast : list[MeasurementRequest] = list(self.pending_reqs_to_broadcast) 
        #     pending_reqs_to_broadcast.sort(key=lambda a : a.t_start)

        #     # find best path for broadcasts at the current time
        #     t = state.t if t is None else t
        #     path, t_start = self._create_broadcast_path(state, orbitdata, t)

        #     for req in tqdm(pending_reqs_to_broadcast,
        #                     desc=f'{state.agent_name}-PLANNER: Scheduling Measurement Request Broadcasts', 
        #                     leave=False):
                
        #         # calculate broadcast start time
        #         if req.t_start > t_start:
        #             path, t_start = self._create_broadcast_path(state, orbitdata, req.t_start)
                    
        #         # check broadcast feasibility
        #         if t_start < 0: continue

        #         # create broadcast action
        #         msg = MeasurementRequestMessage(state.agent_name, state.agent_name, req.to_dict(), path=path)
        #         broadcast_action = BroadcastMessageAction(msg.to_dict(), t_start)
                
        #         # add to list of broadcasts
        #         broadcasts.append(broadcast_action) 
                        
        # return scheduled broadcasts
        return broadcasts 

    @runtime_tracker
    def _schedule_periodic_replan(self, state : SimulationAgentState, prelim_plan : Plan, t_next : float) -> list:
        """ Creates and schedules a waitForMessage action such that it triggers a periodic replan """

        # find wait start time
        if prelim_plan.is_empty():
            t_wait_start = state.t 
        
        else:
            actions_within_period = [action for action in prelim_plan 
                                 if  isinstance(action, AgentAction)
                                 and action.t_start < t_next]

            if actions_within_period:
                # last_action : AgentAction = actions_within_period.pop()
                t_wait_start = min(max([action.t_end for action in actions_within_period]), t_next)
                                
            else:
                t_wait_start = state.t

        # create wait action
        return [WaitForMessages(t_wait_start, t_next)] if t_wait_start < t_next else []
    
    @runtime_tracker
    def get_ground_points(self,
                          orbitdata : OrbitData
                        ) -> dict:
        # initiate accestimes 
        all_ground_points = list({
            (grid_index, gp_index, lat, lon)
            for grid_datum in orbitdata.grid_data
            for lat, lon, grid_index, gp_index in grid_datum.values
        })
        
        # organize into a `dict`
        ground_points = dict()
        for grid_index, gp_index, lat, lon in all_ground_points: 
            if grid_index not in ground_points: ground_points[grid_index] = dict()
            if gp_index not in ground_points[grid_index]: ground_points[grid_index][gp_index] = dict()

            ground_points[grid_index][gp_index] = (lat,lon)

        # return grid information
        return ground_points
