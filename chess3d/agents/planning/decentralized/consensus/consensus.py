from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

import logging

from dmas.messages import SimulationMessage
from dmas.utils import runtime_tracker
from dmas.agents import AgentAction
from dmas.clocks import *

from chess3d.agents.actions import FutureBroadcastMessageAction, ObservationAction, WaitForMessages
from chess3d.agents.planning.reactive import AbstractReactivePlanner
from chess3d.agents.planning.tasks import GenericObservationTask, SpecificObservationTask
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

        # compile agility specifications
        max_slew_rate, max_torque = self._collect_agility_specs(specs)

        # Outline planning horizon interval
        planning_horizon = Interval(state.t, self.preplan.t_next)

        # get only available tasks from existing plan and urgent tasks
        available_tasks : list[GenericObservationTask] = self.get_available_tasks(planning_horizon)
        
        # calculate coverage opportunities for available tasks
        access_opportunities : dict[tuple] = self.calculate_access_opportunities(state, planning_horizon, orbitdata)

        # create schedulable tasks from known tasks and future access opportunities
        schedulable_tasks : list[SpecificObservationTask] = self.create_tasks_from_accesses(available_tasks, access_opportunities, cross_track_fovs, orbitdata)

        # filter for only schedulable tasks with urgent parent tasks        
        schedulable_urgent_tasks : list[SpecificObservationTask] = [task for task in schedulable_tasks 
                                                                    if any(parent_task in self.known_urgent_tasks 
                                                                            for parent_task in task.parent_tasks)]

        # generate new plan according to selected model
        if self.model == self.EARLIEST_ACCESS:
            return self.earliest_access_bundle_builder(state, specs, current_plan, schedulable_urgent_tasks, orbitdata, mission, observation_history)
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
        # initialized bundle
        if isinstance(current_plan, PeriodicPlan) and abs(state.t - current_plan.t) <= self.EPS:
            raise NotImplementedError("Earliest-access bundle builder initializing for new preplans not yet implemented.")
        else:
            bundle : List[Tuple[GenericObservationTask, int, float, float]] = \
                 [task_tuple for task_tuple in self.bundle]
            
        # get current path from existing plan
        current_path = [action
                for action in current_plan
                if isinstance(action, ObservationAction)]

        # select an observation time for each urgent task
        for urgent_task in sorted(schedulable_urgent_tasks, key=lambda task: task.accessibility.left):
            # Find best placement in path
            # Option 1: Direct Insertion into existing path
            new_path = self._direct_insertion_into_path(state, specs, current_path, urgent_task, orbitdata, mission, observation_history, bundle)

            # Option 2: Right-shifting existing path to accommodate new task
            if new_path is None:
                new_path = self._right_shift_path_for_new_task(state, specs, current_path, urgent_task, orbitdata, mission, observation_history, bundle)
            # Option 3: Replace conflicting tasks with new urgent task
            if new_path is None:
                new_path = self._replace_conflicting_tasks_with_new_task(state, specs, current_path, urgent_task, orbitdata, mission, observation_history, bundle)

            # ignore new urgent task if cannot be scheduled           
            if new_path is None: continue

            # calculate expected utility of new task observation
            x = 1    

            # estimate observation number and revistit time 

            # estimate expected utility of new task observation

            # check if bid can be improved
            # if yes, update bundle

            # else, continue

        return bundle
