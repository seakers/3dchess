from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

import logging

from dmas.messages import SimulationMessage
from dmas.utils import runtime_tracker
from dmas.agents import AgentAction
from dmas.clocks import ClockConfig

from chess3d.agents.actions import FutureBroadcastMessageAction, ObservationAction, WaitForMessages
from chess3d.agents.planning.reactive import AbstractReactivePlanner
from chess3d.agents.planning.tasks import GenericObservationTask, EventObservationTask, SpecificObservationTask
from chess3d.agents.planning.tracker import ObservationHistory
from chess3d.agents.planning.plan import Plan, PeriodicPlan, ReactivePlan
from chess3d.agents.planning.decentralized.consensus.bids import AsynchronousBid, Bid
from chess3d.agents.science.reward import *
from chess3d.messages import MeasurementBidMessage
from chess3d.mission.mission import Mission
from chess3d.agents.states import SimulationAgentState
from chess3d.orbitdata import OrbitData
from chess3d.utils import Interval

class ConsensusReplanner(AbstractReactivePlanner):    
    # Replanning models
    EARLIEST_ACCESS = 'earliest_access'
    HEURISTIC_INSERTION = 'heuristic_insertion'
    DYNAMIC_PROGRAMMING = 'dynamic_programming'
    MILP = 'mixed-integer_linear_programming'
    MODELS = [EARLIEST_ACCESS, HEURISTIC_INSERTION, DYNAMIC_PROGRAMMING, MILP]

    # Constants
    EPS = 1e-6

    def __init__(self, 
                 model : str = HEURISTIC_INSERTION,
                 replan_threshold : int = 1,
                 debug : bool = False,
                 logger: logging.Logger = None
                 ) -> None:
        super().__init__(debug, logger)

        # validate inputs
        assert model in self.MODELS, f"Invalid model '{model}'. Must be one of {self.MODELS}."
        assert isinstance(replan_threshold, int) and replan_threshold > 0, "Replan threshold must be positive integer."

        # initialize results
        self.bundle : list[GenericObservationTask] = list()
        self.path : list[GenericObservationTask] = list()
        self.results : Dict[GenericObservationTask, List[Bid]] = defaultdict(list)
        self.preplan : PeriodicPlan = None
        self.plan : Plan = None
        self.known_urgent_tasks : set[GenericObservationTask] = set()
        self.new_urgent_tasks : set[GenericObservationTask] = set()
        self.bid_inbox : list[Bid] = list()

        # set parameters
        self.model = model
        self.replan_threshold = replan_threshold
        self.t_share = -1


    """
    ---------------------------
    CONSENSUS PHASE
    ---------------------------
    """

    def update_percepts(self, 
                        state : SimulationAgentState,
                        current_plan : Plan,
                        incoming_reqs: List[TaskRequest], 
                        relay_messages: List[SimulationMessage], 
                        misc_messages : List[SimulationMessage],
                        completed_actions: List[AgentAction],
                        aborted_actions : List[AgentAction],
                        pending_actions : List[AgentAction]
                    ) -> None:
        
        # check if new base plan is available
        self.__update_preplan(state, current_plan)

        # check if new task requests have arrived and filter for available requests
        self.__update_urgent_tasks(state, incoming_reqs)

        # TODO collect pending and performed task request announcement broadcasts 
        
        # convert incoming task requests to bids and add to inbox
        self.__generate_bids_from_reqs(state, incoming_reqs)

        # collect bids from incoming messages to inbox
        self.__collect_incoming_bids(misc_messages)  

        return # Placeholder implementation
        # raise NotImplementedError("Consensus replanner not yet implemented.")

    def __update_preplan(self, state : SimulationAgentState, current_plan : Plan) -> None:
        """ Update latest preplan if new plan is available. """
        if isinstance(current_plan, PeriodicPlan) and abs(state.t - current_plan.t) <= self.EPS:
            self.preplan : PeriodicPlan = current_plan.copy()

    def __update_urgent_tasks(self, state : SimulationAgentState, incoming_reqs : List[TaskRequest]) -> None:
        """ Remove completed tasks from urgent tasks set. """
        # remove unavailable tasks
        self.known_urgent_tasks = set([task for task in self.known_urgent_tasks 
                                        if task.is_available(state.t)])
        self.new_urgent_tasks = set([task for task in self.new_urgent_tasks 
                                        if task.is_available(state.t)])

        # get active incoming tasks
        active_tasks = set([req.task for req in incoming_reqs 
                            if req.task.is_available(state.t)])
        
        # update urgent tasks
        self.known_urgent_tasks.update(active_tasks)
        self.new_urgent_tasks.update(active_tasks)

    def __collect_incoming_bids(self, misc_messages : List[SimulationMessage]) -> None:
        """ Collect bids from incoming messages and requests. """
        incoming_bids = { Bid.from_dict(msg.bid) 
                            for msg in misc_messages 
                            if isinstance(msg, MeasurementBidMessage)}
        
        self.bid_inbox.extend(incoming_bids)

    def __generate_bids_from_reqs(self, state : SimulationAgentState, incoming_reqs : List[TaskRequest]) -> None:
        """ Generate bids from incoming task requests. """
        # extract bids from incoming requests
        # TODO make this an abstract method that can be overridden by subclasses that distinguish between
        # synchronous and asynchronous bidding strategies
        bids_from_reqs = [AsynchronousBid(req.task, state.agent_name) for req in incoming_reqs]
        
        # update bid inbox
        self.bid_inbox.extend(bids_from_reqs)

    def needs_planning(self, 
                       state : SimulationAgentState,
                       specs : object,
                       current_plan : Plan,
                       orbitData : OrbitData
                    ) -> bool:
        # perform consensus phase for incoming bids and tasks
        # TODO
        
        # replan if number of urgent tasks exceeds threshold
        if len(self.new_urgent_tasks) >= self.replan_threshold: 
            return True
        
        # replan if changes were made to the bundle

        # replan if broadcasts need to be scheduled
        # TODO

        # replan if new preplan was just received
        # if isinstance(current_plan, PeriodicPlan) and abs(state.t - current_plan.t) <= self.EPS:
        #     return True

        # otherwise, no replanning needed
        return False 

    """
    ---------------------------
    BUNDLE-BUILDING PHASE
    ---------------------------
    """
    def generate_plan(self, 
                      state : SimulationAgentState,
                      specs : object,
                      current_plan : Plan,
                      clock_config : ClockConfig,
                      orbitdata : OrbitData,
                      mission : Mission,
                      tasks : list,
                      observation_history : ObservationHistory,
                    ) -> Plan:               
    
        # -------------------------------
        # DEBUG PRINTOUTS
        # self.log_results('PRE-PLANNING PHASE', state, self.results)
        # -------------------------------

        # update bundle
        new_bundle, new_path = self.bundle_building_phase(state, specs, current_plan, clock_config, orbitdata, mission, observation_history)
        
        # update results
        # TODO 

        # -------------------------------
        # DEBUG PRINTOUTS
        # self.log_results('PLANNING PHASE', state, self.results)
        # print(f'bundle:')
        # for req, subtask_index, bid in self.bundle:
        #     req : MeasurementRequest
        #     bid : Bid
        #     req_id_short = req.id.split('-')[0]
        #     print(f'\t{req_id_short}, {subtask_index}, {np.round(bid.t_img,3)}, {np.round(bid.bid)}')
        # print('')
        # -------------------------------
    
        return ReactivePlan.from_periodic_plan(current_plan, t=state.t) # Placeholder implementation; no changes to plan

        # schedule observations from bids
        observations : list = self._schedule_observations(state, specs, self.bundle, orbitdata)

        # generate maneuver and travel actions from observations
        maneuvers : list = self._schedule_maneuvers(state, specs, observations, clock_config, orbitdata)

        # schedule broadcasts
        broadcasts : list = self._schedule_broadcasts(state, current_plan, observation_history, orbitdata)       

        # generate wait actions 
        waits : list = self._schedule_waits(state)
        
        # compile and generate plan
        self.plan = ReactivePlan(maneuvers, waits, observations, broadcasts, t=state.t, t_next=self.preplan.t_next)

        # clear new urgent tasks
        self.new_urgent_tasks = set()

        return self.plan.copy()
    
    def bundle_building_phase(self,
                       state : SimulationAgentState,
                       specs : object,
                       current_plan : Plan,
                       clock_config : ClockConfig,
                       orbitdata : OrbitData,
                       mission : Mission,
                       observation_history : ObservationHistory
                    ) -> tuple:
        
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self._collect_fov_specs(specs)

        # Outline planning horizon interval
        planning_horizon = Interval(state.t, self.preplan.t_next)

        # get only available tasks from existing plan and urgent tasks
        available_tasks : list[GenericObservationTask] = self.get_available_tasks(planning_horizon)
        
        # calculate coverage opportunities for available tasks
        access_opportunities : dict[tuple] = self.calculate_access_opportunities(state, planning_horizon, orbitdata)

        # create specific and merged tasks from scheduled tasks and urgent tasks
        schedulable_tasks : list[SpecificObservationTask] = self.create_tasks_from_accesses(available_tasks, access_opportunities, cross_track_fovs, orbitdata)

        # filter for only schedulable tasks with urgent parent tasks        
        schedulable_urgent_tasks : list[SpecificObservationTask] = [task for task in schedulable_tasks 
                                                                    if any(parent_task in self.known_urgent_tasks 
                                                                            for parent_task in task.parent_tasks)]

        # generate new plan according to selected model
        if self.model == self.EARLIEST_ACCESS:
            return self.earliest_access_bundle_builder(state, specs, cross_track_fovs, current_plan, schedulable_urgent_tasks, orbitdata, mission, observation_history)
        elif self.model == self.HEURISTIC_INSERTION:
            raise NotImplementedError("Heuristic-insertion consensus planner not yet implemented.")
        elif self.model == self.DYNAMIC_PROGRAMMING:
            raise NotImplementedError("Dynamic-programming consensus planner not yet implemented.")
        elif self.model == self.MILP:
            raise NotImplementedError("Mixed-integer-linear-programming consensus planner not yet implemented.")
        else:
            raise NotImplementedError(f"Model '{self.model}' not implemented.")            

    def get_available_tasks(self, planning_horizon : Interval) -> list:
        """ Get only tasks that are available within the planning horizon. """
        # get tasks present in current preplan
        planned_tasks = {parent_task
                         for action in self.preplan.actions 
                         if isinstance(action, ObservationAction)
                         for parent_task in action.task.parent_tasks
                         }

        # get urgent tasks that are available within planning horizon
        urgent_tasks = {task 
                           for task in self.known_urgent_tasks 
                            if task.availability.overlaps(planning_horizon)}
        
        # merge task sets
        available_tasks = {task for task in urgent_tasks}
        available_tasks.update(planned_tasks)

        # return tasks as a merged list
        return list(available_tasks)

    @runtime_tracker
    def calculate_access_opportunities(self, 
                                       state : SimulationAgentState, 
                                       planning_horizon : Interval,
                                       orbitdata : OrbitData
                                    ) -> dict:
        """ Calculate access opportunities for targets visible in the planning horizon """

        # check planning horizon span
        if planning_horizon.is_empty(): 
            return {}

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

    def earliest_access_bundle_builder(self,
                                       state : SimulationAgentState,
                                       specs : object,
                                       cross_track_fovs : dict,
                                       current_plan : Plan,
                                       schedulable_urgent_tasks : List[SpecificObservationTask],
                                       orbitdata : OrbitData,
                                       mission : Mission,
                                       observation_history : ObservationHistory
                                    ) -> List:
        """ 
        Build bundle using earliest-access heuristic. 

        #### Returns
        - bundle : List[Tuple[GenericObservationTask, int, float, float]]
            List of tuples containing (task, observation number, observation time, expected utility).
        
        """
        # compile agility specifications
        max_slew_rate, max_torque = self._collect_agility_specs(specs)

        # initialized bundle
        if isinstance(current_plan, PeriodicPlan) and abs(state.t - current_plan.t) <= self.EPS:
            raise NotImplementedError("Earliest-access bundle builder initializing for new preplans not yet implemented.")
        else:
            bundle : List[Tuple[GenericObservationTask, int, float, float]] = \
                 [task_tuple for task_tuple in self.bundle]
            current_path = sorted([action for action in current_plan
                                   if isinstance(action, ObservationAction)], 
                                   key=lambda action: action.t_start)

        # Find best placement in path
        for urgent_task in sorted(schedulable_urgent_tasks, key=lambda task: task.accessibility.left):
            # Option 1: Direct Insertion into existing path
            new_path, t_img = self._direct_insertion_into_path(current_path, urgent_task, max_slew_rate)

            # Option 2: Right-shifting existing path to accommodate new task
            if new_path is None:
                new_path, t_img = self._right_shift_path_for_new_task(state, specs, current_path, urgent_task, max_slew_rate, max_torque, orbitdata, mission, observation_history)
            
            # Option 3: Replace conflicting tasks with new urgent task
            if new_path is None:
                new_path, t_img = self._replace_conflicting_tasks_with_new_task(state, specs, current_path, urgent_task, max_slew_rate, max_torque, orbitdata, mission, observation_history)
            # if no feasible path was found, ignore new urgent task
            if new_path is None: continue
            
            # calculate bids for new path
            new_path_value : float = self._calculate_path_value(specs, new_path, observation_history, orbitdata, mission)
            old_path_value : float = self._calculate_path_value(specs, current_path, observation_history, orbitdata, mission)
            bid_value : float = new_path_value - old_path_value

            # if bid is negative do NOT add to bundle
            if bid_value < 0: continue

            # create bids for relevant parent tasks
            ## get parent tasks from current task
            parent_tasks = [parent_task for parent_task in urgent_task.parent_tasks
                            if parent_task in self.known_urgent_tasks]
            ## create bid for each parent task
            new_bids = []
            for parent_task in parent_tasks:
                # create new bid
                bid = AsynchronousBid(parent_task, state.agent_name, n_img, bid_value, state.agent_name, bid_value, t_img, state.t, urgent_task.instrument_name)
                
                # if bid beats existing bids for parent task, add to `new_bids`
                new_bids.append((parent_task, bid))
            
            # update bundle and current path                    

        return bundle
    
    def _calculate_path_value(self,
                              specs : object,
                              cross_track_fovs : Dict[str, float],
                              path : List[ObservationAction],
                              observation_history : ObservationHistory,
                              orbitdata : OrbitData,
                              mission : Mission
                            ) -> float:
        """ Calculate total expected value of observation path. """
        total_value = 0.0

        for i_obs,obs in enumerate(path):
            # get all previous observations with the same parent tasks

            # calculate number of previous observations for this task

            # estimate previous observation time

            # calculate expected value of observation
            obs_value = self.estimate_specific_task_value(obs.task,
                                                 obs.t_start,
                                                 obs.task.min_duration,
                                                 specs,
                                                 cross_track_fovs[obs.instrument_name],
                                                 orbitdata,
                                                 mission,
                                                 observation_history,
                                                 n_obs,
                                                 t_prev)

            # accumulate total value
            total_value += obs_value

        return total_value

    """
    BUNDLE-BUILDING PHASE - Path Insertion Methods
    """
    def _direct_insertion_into_path(self,
                                    current_path : List[ObservationAction],
                                    new_task : SpecificObservationTask,
                                    max_slew_rate : float
                                ) -> List[ObservationAction]:
        """ Try to directly insert new task into existing path. """
        # select observation loook angle for new task
        th_img = np.average([new_task.slew_angles.left, new_task.slew_angles.right])

        # initialize feasible observation time
        t_img = None 

        # find possible conflicts in current path
        ## find observations that are being performed during new task accessibility
        observations_during_task_access = [action for action in current_path
                                           if action.t_start in new_task.accessibility
                                           or action.t_end in new_task.accessibility]
        ## get latest observation before new task accessibility
        prev_observations = [action for action in current_path
                             if action.t_end <= new_task.accessibility.left]
        prev_observation = max(prev_observations, key=lambda action: action.t_end) if prev_observations else None
        ## get earliest observation after new task accessibility
        next_observations = [action for action in current_path
                             if action.t_start >= new_task.accessibility.right]
        next_observation = min(next_observations, key=lambda action: action.t_start) if next_observations else None

        # compile conflicting observations
        conflicting_observations = {prev_observation, next_observation} if prev_observation else {next_observation} if next_observation else set()
        ## get unique observations during new task access
        conflicting_observations.update(observations_during_task_access)
        ## sort conflicting observations by start time
        conflicting_observations = sorted([obs for obs in conflicting_observations 
                                           if obs is not None], key=lambda obs: obs.t_start)

        # check if gaps between observations can accommodate new task
        obs_prev = None
        for obs_next in conflicting_observations:
            # set previous observation
            if obs_prev is None: 
                obs_prev = obs_next
                continue
            
            # check maneuver time between new task and current observations
            m_prev = abs(obs_prev.look_angle - th_img) / max_slew_rate
            m_next = abs(obs_next.look_angle - th_img) / max_slew_rate        
            
            # get earliest and latest feasible observation time
            t_earliest = max(new_task.accessibility.left, obs_prev.t_end + m_prev)
            t_latest = min(new_task.accessibility.right, obs_next.t_start - m_next) - new_task.min_duration

            # check if feasible observation time exists
            ## 1) must be able to maneuver from previous observation to new task
            ## 2) must be able to maneuver from new task to next observation
            ## 3) must fit within new task accessibility window
            earliest_is_feasible = (t_earliest + new_task.min_duration + m_next <= obs_next.t_start
                                    and obs_prev.t_end + m_prev <= t_earliest
                                    and new_task.accessibility.left <= t_earliest
                                    and t_earliest + new_task.min_duration <= new_task.accessibility.right)
            latest_is_feasible = (t_latest + new_task.min_duration + m_next <= obs_next.t_start
                                    and obs_prev.t_end + m_prev <= t_latest
                                    and new_task.accessibility.left <= t_latest 
                                    and t_latest + new_task.min_duration <= new_task.accessibility.right)
            
            # if feasible, select observation time
            if earliest_is_feasible:
                # choose earliest feasible time
                t_img = t_earliest
            elif latest_is_feasible:
                # choose latest feasible time
                t_img = t_latest

            # if feasible time found, break
            if earliest_is_feasible or latest_is_feasible: break    

        # no conflicting observations were found
        if not conflicting_observations:
            # schedule at earliest access time
            t_img = new_task.accessibility.left

        # check if observation time was found
        if t_img is None: return None, None # no time found; cannot insert new task into path

        # insert new observation into path
        ## create observation action for new task
        new_observation = ObservationAction(new_task.instrument_name, th_img, t_img, new_task.min_duration, new_task)

        ## create new path with inserted observation
        new_path = [action for action in current_path]
        new_path.append(new_observation)
        new_path = sorted(new_path, key=lambda action: action.t_start)
        
        # return new path 
        return new_path

    def _right_shift_path_for_new_task(self,
                                        current_path : List[ObservationAction],
                                        new_task : SpecificObservationTask,
                                        max_slew_rate : float
                                    ) -> List[ObservationAction]:
        """ Try to right-shift existing path to accommodate new task. """
        # select observation loook angle for new task
        th_img = np.average([new_task.slew_angles.left, new_task.slew_angles.right])

        # initialize feasible observation time
        t_img = None 

        # find previous observations before new task accessibility
        prev_observations = sorted([action for action in current_path
                                    if action.t_end <= (new_task.accessibility.right - new_task.min_duration)],
                                    key=lambda action: action.t_end,
                                    reverse=True)

        i_insert = 0
        for i_obs,prev_obs in enumerate(prev_observations):
            # check maneuver time between new task and current observation
            m_prev = abs(prev_obs.look_angle - th_img) / max_slew_rate

            # get earliest feasible observation time
            t_earliest = max(new_task.accessibility.left, prev_obs.t_end + m_prev)

            # check if feasible observation time exists
            ## 1) must be able to maneuver from previous observation to new task
            ## 2) must fit within new task accessibility window
            is_feasible = (prev_obs.t_end + m_prev <= t_earliest
                           and new_task.accessibility.left <= t_earliest
                           and t_earliest + new_task.min_duration <= new_task.accessibility.right)
            if is_feasible:     
                # set insertion index after current observation
                i_insert = len(prev_observations) - i_obs
                # choose earliest feasible time
                t_img = t_earliest

                # stop searching
                break

        # create new observation action
        new_observation = ObservationAction(new_task.instrument_name, th_img, t_img, new_task.min_duration, new_task)
        
        # initiate new path
        new_path = [action for action in current_path[:i_insert]]
        new_path.append(new_observation)

        # right-shift remaining observations
        for i_curr,obs_curr in enumerate(current_path[i_insert:]):
            # check previous observation in path
            prev_obs = new_path[-1]

            # compute maneuver time from previous observation
            m = abs(prev_obs.look_angle - obs_curr.look_angle) / max_slew_rate

            # calculate earliest start time for current observation
            t_earliest = max(prev_obs.t_end + m, obs_curr.task.accessibility.left)

            # check earliest time if feasible
            is_feasible = (prev_obs.t_end + m_prev <= t_earliest
                           and new_task.accessibility.left <= t_earliest
                           and t_earliest + new_task.min_duration <= new_task.accessibility.right)

            # check of new observation time is earlier the or the same as original
            if t_earliest < obs_curr.t_start or abs(t_earliest - obs_curr.t_start) <= self.EPS:
                # new task starts earlier, do not modify remaining plan
                new_path.extend(current_path[i_insert + i_curr:])
                break

            # if later start time and not feasible, do not add to new path
            elif not is_feasible: 
                continue
                
            # else, add shifted observation to new path
            else:
                # create shifted observation action
                shifted_observation = ObservationAction(obs_curr.instrument_name, obs_curr.look_angle, t_earliest, obs_curr.task.min_duration, obs_curr.task)

                # add shifted observation to new path
                new_path.append(shifted_observation)

        # return new path
        return new_path
    
    def _replace_conflicting_tasks_with_new_task(self,
                                    current_path : List[ObservationAction],
                                    new_task : SpecificObservationTask,
                                    max_slew_rate : float
                                ) -> List[ObservationAction]:
        """ Try to replace conflicting tasks in existing path with new task. """
        # TODO 
        raise NotImplementedError("Replace conflicting tasks with new task method not yet implemented.")
    
        # select observation loook angle for new task
        th_img = np.average([new_task.slew_angles.left, new_task.slew_angles.right])

        # initialize feasible observation time
        t_img = None 

        # find possible conflicts in current path
        ## find observations that are being performed during new task accessibility
        observations_during_task_access = [action for action in current_path
                                           if action.t_start in new_task.accessibility
                                           or action.t_end in new_task.accessibility]
        ## get latest observation before new task accessibility
        prev_observations = [action for action in current_path
                             if action.t_end <= new_task.accessibility.left]
        prev_observation = max(prev_observations, key=lambda action: action.t_end) if prev_observations else None
        ## get earliest observation after new task accessibility
        next_observations = [action for action in current_path
                             if action.t_start >= new_task.accessibility.right]
        next_observation = min(next_observations, key=lambda action: action.t_start) if next_observations else None

        # compile conflicting observations
        conflicting_observations = {prev_observation, next_observation} if prev_observation else {next_observation} if next_observation else set()
        ## get unique observations during new task access
        conflicting_observations.update(observations_during_task_access)
        ## sort conflicting observations by start time
        conflicting_observations = sorted([obs for obs in conflicting_observations 
                                           if obs is not None], key=lambda obs: obs.t_start)
        
        # check if gaps between observations can accommodate new task
        obs_prev = None
        obs_to_remove = set()
        for obs_next in conflicting_observations:
            # set previous observation
            if obs_prev is None: 
                obs_prev = obs_next
                continue
            
            # check maneuver time between new task and current observations
            m_prev = abs(obs_prev.look_angle - th_img) / max_slew_rate
            m_next = abs(obs_next.look_angle - th_img) / max_slew_rate        
            
            # set earliest feasible observation time
            t_img = max(new_task.accessibility.left, obs_prev.t_end + m_prev)

            # check if earlist observation time is feasible
            ## 1) must be able to maneuver from previous observation to new task
            prev_to_earliest_can_maneuver = obs_prev.t_end + m_prev <= t_img
            ## 2) must be able to maneuver from new task to next observation
            earliest_to_next_can_maneuver = t_img + new_task.min_duration + m_next <= obs_next.t_start
            ## 3) must fit within new task accessibility window            
            earliest_in_access = (new_task.accessibility.left <= t_img
                                  and t_img + new_task.min_duration <= new_task.accessibility.right)
            
            # check feasibility
            if (prev_to_earliest_can_maneuver and earliest_to_next_can_maneuver and earliest_in_access):
                # no conflicting observations in this gap need to be removed
                break    

            # earliest observation time is unfeasible; check which observation to remove
            if not prev_to_earliest_can_maneuver:
                # earliest time cannot maneuver from previous observation; remove previous observation
                obs_to_remove.add(obs_prev)

            elif not earliest_to_next_can_maneuver:
                # earliest time cannot maneuver to next observation; remove next observation
                obs_to_remove.add(obs_next)
        
        # no conflicting observations were found
        if not conflicting_observations:
            # schedule at earliest access time
            t_img = new_task.accessibility.left

        # check if observation time was found
        if t_img is None: return None, None # no time found; cannot insert new task into path

        # insert new observation into path
        ## create observation action for new task
        new_observation = ObservationAction(new_task.instrument_name, th_img, t_img, new_task.min_duration, new_task)

        ## create new path with inserted observation
        new_path = [action for action in current_path]
        new_path.append(new_observation)
        new_path = sorted(new_path, key=lambda action: action.t_start)
        
        # return new path 
        return new_path