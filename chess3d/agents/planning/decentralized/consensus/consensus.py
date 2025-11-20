from collections import defaultdict
from typing import Dict, List
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
    
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self._collect_fov_specs(specs)

        # compile agility specifications
        max_slew_rate, max_torque = self._collect_agility_specs(specs)

        # Outline planning horizon interval
        planning_horizon = Interval(state.t, self.preplan.t_next)

        # get only available tasks
        available_tasks : list[GenericObservationTask] = self.get_available_tasks(planning_horizon)
        
        # calculate coverage opportunities for tasks
        access_opportunities : dict[tuple] = self.calculate_access_opportunities(state, planning_horizon, orbitdata)

        # create schedulable tasks from known tasks and future access opportunities
        schedulable_tasks : list[SpecificObservationTask] = self.create_tasks_from_accesses(available_tasks, access_opportunities, cross_track_fovs, orbitdata)

        # filter for only schedulable tasks with urgent parent tasks
        schedulable_urgent_tasks : list[SpecificObservationTask] = [task for task in schedulable_tasks 
                                                                    if any(parent_task in self.known_urgent_tasks 
                                                                            for parent_task in task.parent_tasks)]

        # generate new plan according to selected model
        if self.model == self.EARLIEST_ACCESS:
            foo = self.earliest_access_planner(state, specs, current_plan, clock_config, orbitdata, mission, tasks, observation_history)
        elif self.model == self.HEURISTIC_INSERTION:
            pass
        elif self.model == self.DYNAMIC_PROGRAMMING:
            pass
        elif self.model == self.MILP:
            pass
        else:
            raise NotImplementedError(f"Model '{self.model}' not implemented.")
        
        # clear new urgent tasks
        self.new_urgent_tasks = set()

        return ReactivePlan.from_periodic_plan(current_plan, t=state.t) # Placeholder implementation; no changes to plan
    
    def get_available_tasks(self, planning_horizon : Interval) -> list:
        """ Get only tasks that are available within the planning horizon. """
        # get tasks present in current preplan
        planned_tasks = {parent_task
                         for action in self.preplan.actions 
                         if isinstance(action, ObservationAction)
                         for parent_task in action.task.parent_tasks
                         }

        # get urgent tasks that are available within planning horizon
        available_tasks = {task 
                           for task in self.known_urgent_tasks 
                            if task.availability.overlaps(planning_horizon)}
        
        # include planned tasks
        available_tasks.update(planned_tasks)

        # return tasks as a list
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

    def earliest_access_planner(self,
                                state : SimulationAgentState,
                                specs : object,
                                current_plan : Plan,
                                clock_config : ClockConfig,
                                orbitdata : OrbitData,
                                mission : Mission,
                                tasks : list,
                                observation_history : ObservationHistory
                            ) -> List:
        # initialized bundle
        bundle = []

        # select an observation time for each urgent task
        t_imgs = [np.NINF for _ in self.new_urgent_tasks]
        for urgent_task in self.new_urgent_tasks:
            x = 1
