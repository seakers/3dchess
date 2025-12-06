from abc import abstractmethod
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple

import logging

from dmas.messages import SimulationMessage
from dmas.utils import runtime_tracker
from dmas.agents import AgentAction
from dmas.clocks import ClockConfig

from chess3d.agents.actions import BroadcastMessageAction, FutureBroadcastMessageAction, ObservationAction, WaitForMessages
from chess3d.agents.planning.reactive import AbstractReactivePlanner
from chess3d.agents.planning.tasks import GenericObservationTask, EventObservationTask, SpecificObservationTask
from chess3d.agents.planning.tracker import ObservationHistory, ObservationTracker
from chess3d.agents.planning.plan import Plan, PeriodicPlan, ReactivePlan
from chess3d.agents.planning.decentralized.consensus.bids import Bid
from chess3d.agents.science.reward import *
from chess3d.messages import BusMessage, MeasurementBidMessage
from chess3d.mission.mission import Mission
from chess3d.agents.states import SatelliteAgentState, SimulationAgentState
from chess3d.orbitdata import OrbitData
from chess3d.utils import Interval

class ConsensusPlanner(AbstractReactivePlanner):    
    # Replanning models
    # NOTE break down into separate replanner classes that use this class as the parent class?
    HEURISTIC_INSERTION = 'heuristicInsertion'
    DYNAMIC_PROGRAMMING = 'dynamicProgramming'
    MILP = 'mixedIntegerLinearProgramming'
    MODELS = [HEURISTIC_INSERTION, DYNAMIC_PROGRAMMING, MILP]
    
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

        # initialize consensus results
        self.bundle : list[list[Bid]] = list()
        self.path : list[GenericObservationTask] = list()
        self.results : Dict[GenericObservationTask, List[Bid]] = defaultdict(list)

        # initialize urgent tasks and bid inbox/outbox
        self.known_event_tasks : set[GenericObservationTask] = set()
        self.incoming_event_tasks : deque[GenericObservationTask] = deque()
        self.bid_inbox : deque[Bid] = deque()
        self.bid_outbox : Dict[GenericObservationTask, Dict[int,Bid]] = defaultdict(dict)
        self.relevant_updates : List[Bid] = list()

        # initialize known preplan and current plan
        self.preplan : PeriodicPlan = None
        self.plan : Plan = None

        # set parameters
        self.model = model
        self.replan_threshold = replan_threshold
        self.t_share = -1   

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

        # collect bids from incoming messages to inbox
        self.__collect_incoming_bids(misc_messages)  

    def __update_preplan(self, state : SimulationAgentState, current_plan : Plan) -> None:
        """ Update latest preplan if new plan is available. """
        if isinstance(current_plan, PeriodicPlan) and abs(state.t - current_plan.t) <= self.EPS:
            self.preplan : PeriodicPlan = current_plan.copy()

    def __update_urgent_tasks(self, state : SimulationAgentState, incoming_reqs : List[TaskRequest]) -> None:
        """ Remove completed tasks from urgent tasks set. """
        # # remove unavailable tasks
        # self.known_event_tasks = set([task for task in self.known_event_tasks 
        #                                 if task.is_available(state.t)])
        # self.incoming_event_tasks = set([task for task in self.incoming_event_tasks 
        #                                 if task.is_available(state.t)])

        # get active incoming tasks
        active_tasks = set([req.task for req in incoming_reqs 
                            if req.task.is_available(state.t)])
        
        # update urgent tasks
        self.known_event_tasks.update(active_tasks)
        self.incoming_event_tasks.extend(active_tasks)

    def __collect_incoming_bids(self, misc_messages : List[SimulationMessage]) -> None:
        """ Collect bids from incoming messages and requests. """
        incoming_bids = [Bid.from_dict(msg.bid) 
                            for msg in misc_messages 
                            if isinstance(msg, MeasurementBidMessage)]
        
        if incoming_bids: 
            x = 1 # Placeholder implementation

        self.bid_inbox.extend(incoming_bids)

    """
    ---------------------------
    CONSENSUS PHASE
    ---------------------------
    """

    def needs_planning(self, 
                       state : SimulationAgentState,
                       specs : object,
                       current_plan : Plan,
                       orbitData : OrbitData
                    ) -> bool:
        # -------------------------------
        # DEBUG PRINTOUTS
        if self.bid_inbox:
            self.log_results('CONSENSUS PHASE (BEFORE)', state, self.results)
            self.log_bundle('BUNDLE (BEFORE CONSENSUS)', state, self.bundle)
        # -------------------------------

        # perform consensus phase for incoming task bids
        relevant_changes = self.consensus_phase(state, specs, current_plan, orbitData)
       
        # -------------------------------
        # DEBUG PRINTOUTS
        if relevant_changes:
            self.log_results('CONSENSUS PHASE (AFTER)', state, self.results)
            self.log_bundle('BUNDLE (AFTER CONSENSUS)', state, self.bundle)
        # -------------------------------

        # replan if...
        # 1) there were relevant updates to bids/results
        relevant_changes_received = len(relevant_changes) > 0
        # 2) or new periodic plan was received
        new_periodic_plan_received = isinstance(current_plan, PeriodicPlan) and abs(state.t - current_plan.t) <= self.EPS
        
        # -------------------------------
        # DEBUG BREAKPOINTS
        if relevant_changes_received:
            x = 1  # Placeholder implementation
        if new_periodic_plan_received:
            x = 1  # Placeholder implementation
        # -------------------------------

        return (relevant_changes_received 
                # or new_periodic_plan_received 
                )

    def consensus_phase(self,
                        state : SimulationAgentState,
                        specs : object,
                        current_plan : Plan,
                        orbitdata : OrbitData
                    ) -> List[Bid]:
        """ Perform consensus phase to update bids and bundle. """
        # initalize list of changes
        changes = []

        # check for new urgent tasks
        new_task_added = self.check_incoming_urgent_tasks(state)
        
        # TODO check if tasks were performed

        # TODO check if tasks expired        

        # compare results with incoming bids and update bundle
        results_updates = self.update_results(state)

        # TODO update bundle from results if necessary        

        # compile changes and rebroadcasts
        changes.extend(new_task_added)
        changes.extend(results_updates)
        
        return changes

    def check_incoming_urgent_tasks(self, state: SimulationAgentState) -> List[Bid]:
        """ Check for new urgent tasks and update results accordingly. """
        # initialize list of newly added bids from new tasks
        new_task_added = []
        
        # identify new urgent tasks
        new_event_tasks = [task for task in self.incoming_event_tasks 
                           if task not in self.results]
        
        # check if new tasks exceed threshold
        if len(new_event_tasks) >= self.replan_threshold:
            # threshold met; process new tasks
            while self.incoming_event_tasks:
                # remove tasks from incoming queue
                task : GenericObservationTask = self.incoming_event_tasks.popleft()

                # initialize results for new event tasks
                self.results[task] = []
 
                # create empty bid for new task and add to list of changes
                new_task_added.append(Bid(task, state.agent_name))

        # return list of new task bids added to results
        return new_task_added

    def update_results(self,
                       state : SimulationAgentState,
                       ) -> Tuple[List[Bid], List[Bid]]:
        """ Update results from incoming bids. """
        # initialize list of updates done to results
        results_updates = []        

        # process incoming bids
        while self.bid_inbox:
            # get next incoming bid
            incoming_bid : Bid = self.bid_inbox.popleft()

            # check if bid is for a new task or higher observation number
            new_task : bool = incoming_bid.task not in self.results
            new_observation_number : bool = (not new_task) and (incoming_bid.n_obs >= len(self.results[incoming_bid.task]))

            if new_task or new_observation_number:
                # TODO is this correct?
                # get bounds for observation number not being conseidered up to incoming bid's `n_obs`
                n_obs_init = len(self.results[incoming_bid.task]) if not new_task else 0
                n_obs_max = incoming_bid.n_obs + 1

                # add an empty bid for each missing observation number
                for n_obs in range(n_obs_init, n_obs_max):
                    empty_bid = Bid(incoming_bid.task, state.agent_name, n_obs)
                    self.results[incoming_bid.task].append(empty_bid)

            # get current bid for this task and observation number
            current_bid : Bid = self.results[incoming_bid.task][incoming_bid.n_obs]

            # compare incoming bid with existing bids for the same task
            updated_bid : Bid = current_bid.compare(incoming_bid, state.t)

            # update results with modified bid
            self.results[incoming_bid.task][incoming_bid.n_obs] = updated_bid

            # if bid was changed, add to changes list
            if updated_bid.is_different(current_bid): results_updates.append(updated_bid)
        
        # return result changes and bids to rebroadcasts
        return results_updates
    
    def update_bundle_from_results(self,
                                    state : SimulationAgentState,
                                    specs : object,
                                    current_plan : Plan,
                                    clock_config : ClockConfig,
                                    orbitdata : OrbitData,
                                    mission : Mission,
                                    observation_history : ObservationHistory
                                    ) -> Any:
        """ Update bundle according to latest results. """
        raise NotImplementedError("Updating bundle from results not yet implemented.")
    
        # check if any bids in the bundle were modified
        if self.bundle and outbid: 
            raise NotImplementedError("Bundle update after bid comparison not yet implemented.")
        
        # for current_bid in changes:
        #     # search for outbids in bundle
        #     outbid_indices = [bid_idx for bid_idx,bundle_bids in enumerate(self.bundle)
        #                       if current_bid in bundle_bids                         # bid is in bundle
        #                       and current_bid.winning_bidder != state.agent_name]   # was outbid
        #     outbid_index = min(outbid_indices) if len(outbid_indices) > 0 else None
            
        #     if outbid_index is None: continue # no bid in bundle was outbid; continue to next incoming bid
            
        #     # if bid in bundle was outbid, remove all subsequent bids from bundle
        #     for bundle_index in range(outbid_index, len(self.bundle)):
        #         # remove bid from bundle
        #         bundle_bids = self.bundle.pop(bundle_index)

        #         # reset results for removed bids
        #         for bundle_bid in bundle_bids:
        #             # if already outbid, skip
        #             if bundle_bid in outbid: continue

        #             # reset bid
        #             bundle_bid.reset(state.t)

        #             # update results
        #             self.results[bundle_bid.task][bundle_bid.n_obs] = bundle_bid

        #             # add to changes and rebroadcast lists
        #             changes.append(bundle_bid)
        #             rebroadcasts.append(bundle_bid)

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
                      tasks : List[GenericObservationTask],
                      observation_history : ObservationHistory,
                    ) -> Plan:  
        """ Generate new plan according to consensus replanning model. """             
        # DEBUG return original preplan
        # return ReactivePlan.from_periodic_plan(self.preplan,t=state.t)

        # -------------------------------
        # DEBUG PRINTOUTS
        self.log_results('PLANNING PHASE (BEFORE)', state, self.results)
        self.log_bundle('BUNDLE (BEFORE PLANNING)', state, self.bundle)
        # -------------------------------

        # build new bundle
        new_bundle, new_path, new_bids = self.bundle_building_phase(state, specs, current_plan, clock_config, orbitdata, mission, observation_history)
        
        # check if new path is valid
        assert new_path is not None and len(new_path) > 0, "New observation path cannot be empty."
        assert self.is_observation_path_valid(state, new_path, None, None, specs), "New observation path is not valid."   

        # update bundle and path
        self.bundle, self.path = new_bundle, new_path

        # update results
        self.__update_results_from_bundle(new_bundle)

        # update bid outbox
        self.__update_outbox_from_bundle(new_bundle)

        # -------------------------------
        # DEBUG PRINTOUTS
        self.log_results('PLANNING PHASE', state, self.results)
        self.log_bundle('BUNDLE (AFTER PLANNING)', state, self.bundle)
        # -------------------------------
    
        # generate maneuver and travel actions from observations
        maneuvers : list = self._schedule_maneuvers(state, specs, new_path, clock_config, orbitdata)

        # schedule broadcasts
        # TODO decide on broadcast scheduling strategy
        broadcasts : list = self._schedule_broadcasts(state, orbitdata)
                
        # compile and generate plan
        self.plan = ReactivePlan(maneuvers, new_path, broadcasts, t=state.t, t_next=self.preplan.t_next)

        # clear new urgent tasks
        self.incoming_event_tasks = set()

        # return final plan
        return self.plan.copy()
    
    @abstractmethod
    def bundle_building_phase(self,
                       state : SimulationAgentState,
                       specs : object,
                       current_plan : Plan,
                       clock_config : ClockConfig,
                       orbitdata : OrbitData,
                       mission : Mission,
                       observation_history : ObservationHistory
                    ) -> tuple:        
        """ Build bundle according to selected replanning model. """
    
    def __update_results_from_bundle(self, new_bundle : List[List[Bid]]) -> None:
        """ Update results dictionary from new bundle. """
        raise NotImplementedError("Updating results from new bundle not yet implemented.")
        # iterate through bids in new bundle
        for bids in new_bundle:
            for bid in bids:
                # check if bid for this task and observation number already exists
                if len(self.results[bid.task]) <= bid.n_obs:
                    # add empty bids up to `n_obs`
                    for i_obs in range(bid.n_obs - len(self.results[bid.task]) + 1):
                        n_obs = len(self.results[bid.task]) + i_obs
                        self.results[bid.task].append(AsynchronousBid(bid.task, bid.bidder, n_obs=n_obs))
                
                # get existing bid
                existing_bid = self.results[bid.task][bid.n_obs]
                
                # check if new bid is better than existing bid
                assert bid > existing_bid,\
                      "Generated a bid that is not better than existing bid in results."

                # update results with new bid
                self.results[bid.task][bid.n_obs] = bid

        # ensure that results bids match their observation number
        assert all(
                bid.n_obs == obs_idx
                for task in self.results
                for obs_idx, bid in enumerate(self.results[task])
            ), "Results bids are not sorted by observation number."
        
        # ensure that bids are consecutive in observation numbers
        assert all(
            all(bid.n_obs == idx for idx, bid in enumerate(bids))
            for bids in self.results.values()
        ), "Bids are not sorted in consecutive n_obs values."

    def __update_outbox_from_bundle(self, new_bundle : List[List[Bid]]) -> None:
        """ Update bid outbox from new bundle. """
        raise NotImplementedError("Updating bid outbox from new bundle not yet implemented.")
    
        # iterate through bids in new bundle
        for bids in new_bundle:
            for bid in bids:
                # check if bid for this task and observation number already exists in outbox
                if bid.n_obs not in self.bid_outbox[bid.task]:
                    # add new bid to outbox
                    self.bid_outbox[bid.task][bid.n_obs] = bid
                else:
                    # get existing bid
                    existing_bid = self.bid_outbox[bid.task][bid.n_obs]
                    
                    # update outbox with newest bid
                    self.bid_outbox[bid.task][bid.n_obs] = max(existing_bid, bid, key=lambda b: (b.t_stamp, b.bid_value))
       
    """
    BROADCAST SCHEDULING
    """
    def _schedule_broadcasts(self, state: SimulationAgentState, orbitdata: OrbitData) -> list:
        """ Schedules broadcasts to be done by this agent """
        try:
            if not isinstance(state, SatelliteAgentState):
                raise NotImplementedError(f'Broadcast scheduling for agents of type `{type(state)}` not yet implemented.')
            elif orbitdata is None:
                raise ValueError(f'`orbitdata` required for agents of type `{type(state)}`.')

            # initialize list of broadcasts to be done
            broadcasts = []       
            
            # iterate through communication targets
            for target in orbitdata.comms_links.keys():
                # get access intervals with target agent
                access_intervals : List[Interval] = orbitdata.get_next_agent_accesses(target, state.t, include_current=True)

                # create broadcast actions for each access interval
                for next_access in access_intervals:
                    # if no access opportunities in this planning horizon, skip scheduling
                    if next_access.is_empty(): continue

                    # get last access interval and calculate broadcast time
                    t_broadcast : float = max(next_access.left, state.t)

                    # generate plan message to share state
                    state_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.STATE, t_broadcast)

                    # generate plan message to share completed observations
                    observations_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.OBSERVATIONS, t_broadcast)

                    # generate plan message to share any task requests generated
                    task_requests_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.REQUESTS, t_broadcast)

                    # generate bid messages to share bids in results
                    bid_msgs : List[MeasurementBidMessage]= []
                    for bids in self.bid_outbox.values():
                        for bid in bids.values():
                            bid_msgs.append(MeasurementBidMessage(state.agent_name, state.agent_name, bid.to_dict()))
                    bid_bus_msg = BusMessage(state.agent_name, state.agent_name, [bid_msg.to_dict() for bid_msg in bid_msgs])
                    bid_msg_action = BroadcastMessageAction(bid_bus_msg.to_dict(), t_broadcast)
                    
                    # add to client broadcast list
                    broadcasts.extend([state_msg, observations_msg, task_requests_msg, bid_msg_action])

            if not orbitdata.comms_links:
                # no communication links available, broadcast task requests for future planning horizons
                t_broadcast : float = state.t

                # generate plan message to share any task requests generated
                task_requests_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.REQUESTS, t_broadcast)

                # add to client broadcast list
                broadcasts.append(task_requests_msg)

            # return scheduled broadcasts
            return broadcasts 
        
        finally:
            assert isinstance(broadcasts, list), "Scheduled broadcasts is not a list."
            assert all(isinstance(broadcast, BroadcastMessageAction) for broadcast in broadcasts), "Not all scheduled broadcasts are of type `BroadcastMessageAction`."

    """
    LOGGING
    """
    def log_results(self, dsc : str, state : SimulationAgentState, level=logging.DEBUG) -> None:
        out = f'\nT{np.round(state.t,3)}[s]:\t\'{state.agent_name}\'\n{dsc}\n'
        line = 'Req ID\t n_obs\tins\twinner\tbid\tt_img\tt_stamp  performed\n'
        
        # count characters in line for formatting
        L_LINE = len(line)

        # header
        out += line 

        # divider 
        for _ in range(L_LINE + 25): out += '='
        out += '\n'

        n = 15
        i = 1
        for task, bids in self.results.items():
            task : GenericObservationTask
            req_id_short = task.id.split('-')[-1]

            # if all([bid.winner == bid.NONE for _,bid in bids.items()]): continue

            for bid in bids:
                # if i > n: break

                bid : Bid
                # if bid.winner == bid.NONE: continue

                if bid.winning_bidder != bid.NONE:
                    line = f'{req_id_short} {bid.n_obs}\t{bid.main_measurement}\t{bid.winning_bidder[0].lower()}{bid.winning_bidder[-1]}\t{np.round(bid.winning_bid,3)}\t{np.round(bid.t_img,3)}\t{np.round(bid.t_stamp,1)}\t  {(bid.performed)}\n'
                else:
                    line = f'{req_id_short} {bid.n_obs}\t{bid.main_measurement}\tn/a\t{np.round(bid.winning_bid,3)}\t{np.round(bid.t_img,3)}\t{np.round(bid.t_stamp,1)}\t  {(bid.performed)}\n'
                out += line
                i +=1

            for _ in range(L_LINE + 25):
                out += '-'
            out += '\n'

            if i > n:
                out += '\t\t\t...\n'
                for _ in range(L_LINE + 25):
                    out += '-'
                out += '\n'
                break

        print(out)

    def log_bundle(self, dsc : str, state : SimulationAgentState, level=logging.DEBUG) -> None:
        out = f'\nT{np.round(state.t,3)}[s]:\t\'{state.agent_name}\'\n{dsc}\n'
        line = 'i\t Req IDs\n'
        
        # count characters in line for formatting
        L_LINE = len(line)

        # header
        out += line 

        # divider 
        for _ in range(L_LINE + 25): out += '='
        out += '\n'

        if not self.bundle:
            out += '\t<empty bundle>\n'
            for _ in range(L_LINE + 25): out += '-'
            out += '\n'

        n = 15
        for i,bids in enumerate(self.bundle):
            line = f'{i}\t['
            for bid in bids:
                # if i > n: break
                line += f'{bid.task.id.split("-")[-1]}({bid.n_obs}),'
            line = line[:-1] + ']\n'
            out += line

            for _ in range(L_LINE + 25):
                out += '-'
            out += '\n'

            if i > n:
                out += '\t\t\t...\n'
                for _ in range(L_LINE + 25):
                    out += '-'
                out += '\n'
                break

        print(out)