
from collections import defaultdict
from logging import Logger
import queue
import random

from instrupy.base import BasicSensorModel
from instrupy.passive_optical_scanner_model import PassiveOpticalScannerModel
from instrupy.util import ViewGeometry, SphericalGeometry
from orbitpy.util import Spacecraft

from dmas.modules import *
from dmas.utils import runtime_tracker
from tqdm import tqdm

from chess3d.agents.planning.plan import Plan, Preplan
from chess3d.orbitdata import OrbitData, TimeIndexedData
from chess3d.agents.planning.tasks import EventObservationTask, GenericObservationTask, ObservationHistory, ObservationTracker, SchedulableObservationTask
from chess3d.agents.states import *
from chess3d.agents.science.requests import *
from chess3d.messages import *
from chess3d.utils import Interval

class AbstractPlanner(ABC):
    """ 
    Describes a generic planner that, given a new set of percepts, decides whether to generate a new plan
    """
    def __init__(self, 
                 debug : bool = False,
                 logger : logging.Logger = None) -> None:
        # initialize object
        super().__init__()

        # check inputs
        if not isinstance(logger,logging.Logger) and logger is not None: 
            raise ValueError(f'`logger` must be of type `Logger`. Is of type `{type(logger)}`.')

        # initialize attributes
        self.known_reqs : set[TaskRequest] = set()                   # set of known measurement requests
        self.stats : dict = dict()                                          # collector for runtime performance statistics
        
        # set attribute parameters
        self._debug = debug                 # toggles debugging features
        self._logger = logger               # logger for debugging

    @abstractmethod
    def update_percepts( self,
                         state : SimulationAgentState,
                         incoming_reqs : list,
                         relay_messages : list,
                         completed_actions : list,
                         **kwargs
                        ) -> None:
        """ Updates internal knowledge based on incoming percepts """
        
        # check parameters
        assert all([isinstance(req, TaskRequest) for req in incoming_reqs])

        # update list of known requests
        self.known_reqs.update(incoming_reqs)
        
    @abstractmethod
    def needs_planning(self, **kwargs) -> bool:
        """ Determines whether planning is triggered """ 
        
    @abstractmethod
    def generate_plan(self, **kwargs) -> Plan:
        """ Creates a plan for the agent to perform """

    @runtime_tracker
    def create_tasks_from_accesses(self, 
                                    available_tasks : list,
                                    access_times : list, 
                                    cross_track_fovs : dict
                                    ) -> list:
        """ Creates tasks from access times. """

        # generate schedulable tasks from access times
        schedulable_tasks : list[SchedulableObservationTask] \
            = self.single_tasks_from_accesses(available_tasks, access_times, cross_track_fovs)
        
        # check if tasks are clusterable
        adj : Dict[str, set[SchedulableObservationTask]] = self.check_task_clusterability(schedulable_tasks)
   
        # cluster tasks based on adjacency
        combined_tasks : list[SchedulableObservationTask] = self.cluster_tasks(schedulable_tasks, adj)

        # add clustered tasks to the final list of tasks available for scheduling
        schedulable_tasks.extend(combined_tasks)

        assert all([task.slew_angles.span()-1e-6 <= cross_track_fovs[task.instrument_name] for task in schedulable_tasks]), \
            f"Tasks have slew angles larger than the maximum allowed field of view."

        # return tasks
        return schedulable_tasks
    
    @runtime_tracker
    def single_tasks_from_accesses(self,
                                     available_tasks : list,
                                     access_times : list, 
                                     cross_track_fovs : dict) -> list:
        
        # index access times by ground point and grid index
        indexed_access_times = defaultdict(list)
        for access_time in access_times:
            grid_index, gp_index = access_time[0], access_time[1]
            indexed_access_times[(int(grid_index), int(gp_index))].append(access_time)

        # create one task per each access opportunity
        schedulable_tasks : list[SchedulableObservationTask] = []
        for task in tqdm(available_tasks, desc="Calculating access times to known tasks", leave=False):
            task : GenericObservationTask

            # find access time for this task
            for *__,grid_index,gp_index in task.targets:
                # get access times for this ground point and grid index
                matching_access_times = indexed_access_times.get((int(grid_index), int(gp_index)), [])
                
                # create a schedulable task for each access time
                for access_time in matching_access_times:
                    # unpack access time
                    instrument = access_time[2]
                    accessibility = access_time[3]
                    th = access_time[-1]
                    slew_angles = Interval(np.mean(th)-cross_track_fovs[instrument]/2, 
                                        np.mean(th)+cross_track_fovs[instrument]/2)

                    # check if instrument can perform the task                    
                    if not task.objective.can_perform(instrument): 
                        continue # skip if not

                    # check if access time matches task availability
                    if not task.availability.overlaps(accessibility):
                        continue # skip if not

                    # create and add schedulable task to list of schedulable tasks
                    schedulable_tasks.append(SchedulableObservationTask(task, 
                                                            instrument,
                                                            accessibility,
                                                            slew_angles))
        
        # return list of schedulable tasks
        return schedulable_tasks
            
    @runtime_tracker
    def check_task_clusterability(self, schedulable_tasks : list) -> dict:
        schedulable_tasks : list[SchedulableObservationTask]
        adj : Dict[str, set[SchedulableObservationTask]] = {task.id : set() for task in schedulable_tasks}
        
        for i in tqdm(range(len(schedulable_tasks)), leave=False, desc="Checking task clusterability"):
            for j in range(i + 1, len(schedulable_tasks)):
                if schedulable_tasks[i].can_combine(schedulable_tasks[j]):
                    adj[schedulable_tasks[i].id].add(schedulable_tasks[j])
                    adj[schedulable_tasks[j].id].add(schedulable_tasks[i]) 

        return adj

    @runtime_tracker
    def cluster_tasks(self, schedulable_tasks : list, adj : dict) -> list:
        """ Clusters tasks based on adjacency. """ 

        schedulable_tasks : list[SchedulableObservationTask]
        adj : Dict[str, set[SchedulableObservationTask]]

        # sort tasks by degree of adjacency 
        v : list[SchedulableObservationTask] = self.sort_tasks_by_degree(schedulable_tasks, adj)
        
        # only keep tasks that have at least one clusterable task
        v = [task for task in v if len(adj[task.id]) > 0]
        # print(f'\n\n{len(v)} tasks are clusterable out of {len(tasks)} tasks.')
        
        # combine tasks into clusters
        combined_tasks : list[SchedulableObservationTask] = []

        with tqdm(total=len(v), desc="Merging overlapping tasks", leave=False) as pbar:
            while len(v) > 0:
                p : SchedulableObservationTask = v.pop(0)
                n_p : list[SchedulableObservationTask] = self.sort_tasks_by_degree(list(adj[p.id]), adj)

                clique : set = {p}
                pbar.update(1)

                while len(n_p) > 0:
                    # Pick a neighbor q of p, q \in n_p, such that the number of their common neighbors is maximum: If such p are not unique, pick the p with least edges being deleted
                    q : SchedulableObservationTask = n_p.pop(0)

                    # Combine q and p into a new p
                    # p.combine(q)
                    clique.add(q)

                    # Delete edges from q and p that are not connected to their common neighbors
                    common_neighbors : set[SchedulableObservationTask] = adj[p.id].intersection(adj[q.id])

                    neighbors_to_delete_p : set[SchedulableObservationTask] = adj[p.id].difference(common_neighbors)
                    for neighbor in neighbors_to_delete_p: 
                        adj[p.id].remove(neighbor)
                        adj[neighbor.id].remove(p)

                    neighbors_to_delete_q : set[SchedulableObservationTask] = adj[q.id].difference(common_neighbors)
                    for neighbor in neighbors_to_delete_q: 
                        adj[q.id].remove(neighbor)
                        adj[neighbor.id].remove(q)

                    # Reset neighbor collection N_p for the new p;
                    n_p : list[SchedulableObservationTask] = self.sort_tasks_by_degree(list(adj[p.id]), adj)

                for q in clique: 
                    p = p.merge(q)
                    if q in v: 
                        v.remove(q)
                        pbar.update(1)

                v : list[SchedulableObservationTask] = self.sort_tasks_by_degree(v, adj)

                # Add p to the list of combined tasks
                combined_tasks.append(p)

        return combined_tasks

    @runtime_tracker
    def sort_tasks_by_degree(self, tasks : list, adjacency : dict) -> list:
        """ Sorts tasks by degree of adjacency. """
        # calculate degree of each task
        degrees : dict = {task : len(adjacency[task.id]) for task in tasks}

        # sort tasks by degree and return
        return sorted(tasks, key=lambda p: (degrees[p], sum([parent_task.reward for parent_task in p.parent_tasks]), p.accessibility), reverse=True)

    @runtime_tracker
    def calc_task_reward(self, 
                         task : SchedulableObservationTask, 
                         specs : Spacecraft, 
                         cross_track_fovs : dict,
                         orbitdata : OrbitData,
                         observation_history : ObservationHistory
                         ) -> float:
        # estimate observation look angle
        th_img = np.average((task.slew_angles.left, task.slew_angles.right))

        # get ground points accessesible during the availability of the task
        raw_access_data = orbitdata.gp_access_data.lookup_interval(task.accessibility.left, task.accessibility.right)

        # get ground points that are within the agent's field of view
        accessible_gps_data_indeces = [i for i in range(len(raw_access_data['time [s]']))
                                        if abs(raw_access_data['look angle [deg]'][i] - th_img) \
                                            <= cross_track_fovs[task.instrument_name] / 2]
        accessible_gps_performances = {col : [raw_access_data[col][i] for i in accessible_gps_data_indeces]
                                    for col in raw_access_data}

        # calculate total task reward          
        task_reward = 0
        for parent_task in task.parent_tasks:

            # get task targets
            task_targets = {(int(grid_index), int(gp_index))
                            for *_,grid_index,gp_index in parent_task.targets}
            
            # get accesses of desired targets within the task's accessibility and agent's field of view
            valid_access_data_indeces = [i for i in range(len(accessible_gps_performances['time [s]']))
                                        if (int(accessible_gps_performances['grid index'][i]), \
                                            int(accessible_gps_performances['GP index'][i])) in task_targets]
            observation_performances = {col : [accessible_gps_performances[col][i] for i in valid_access_data_indeces]
                                        for col in accessible_gps_performances}
            
            # check if there are no valid observations for this task
            if any([len(observation_performances[col]) == 0 for col in observation_performances]): 
                # no accesses; no reward added
                continue
            
            # get target information
            instrument_spec : BasicSensorModel = next(instr for instr in specs.instrument).mode[0]
            grid_index = observation_performances['grid index'][0] if 'grid index' in observation_performances else None
            gp_index = observation_performances['GP index'][0] if 'GP index' in observation_performances else None

            # get previous observation information
            prev_obs = observation_history.get_observation_history(grid_index, gp_index)

            # get current observation information 
            # TODO : add support for multiple targets within the observation
            observation_performance = {col : observation_performances[col][0] for col in observation_performances}

            # package observation performance information
            if task.instrument_name.lower() in ['vnir', 'tir']:
                observation_performance = {
                    "instrument" : task.instrument_name,    
                    "t_start" : task.accessibility.left,
                    "t_end" : task.accessibility.right,
                    "n_obs" : prev_obs.n_obs,
                    "t_prev" : prev_obs.t_last,
                    "horizontal_spatial_resolution" : observation_performance['ground pixel cross-track resolution [m]'],
                    'spectral_resolution' : instrument_spec.spectral_resolution
                }
            elif task.instrument_name.lower() == 'altimeter':
                observation_performance = {
                    "instrument" : task.instrument_name,    
                    "t_start" : task.accessibility.left,
                    "t_end" : task.accessibility.right,
                    "n_obs" : prev_obs.n_obs,
                    "t_prev" : prev_obs.t_last,
                    "horizontal_spatial_resolution" : observation_performance['ground pixel cross-track resolution [m]'],
                    "accuracy" : observation_performance['accuracy [m]'],
                }
            else:
                raise NotImplementedError(f'Calculation of task reward not yet supported for instruments of type `{task.instrument_name}`.')
            
            # calculate task priority, performance, score and severity
            task_priority = parent_task.objective.priority
            task_performance = parent_task.objective.eval_performance(observation_performance)
            task_score = parent_task.objective.calc_reward(observation_performance)
            task_severity = parent_task.event.severity if isinstance(parent_task, EventObservationTask) else 1.0
            
            # calculate utility of the task
            u_ijk = task_priority * task_performance * task_score * task_severity
            
            # update task reward
            task_reward += u_ijk

        # return total task reward
        return task_reward

    @abstractmethod
    def _schedule_broadcasts(self, 
                             state : SimulationAgentState, 
                             orbitdata : OrbitData,
                             **kwargs
                            ) -> list:
        """ 
        Schedules any broadcasts to be done. 
        
        By default it schedules any pending measurement requests or message relay messages.
        """
    
    @runtime_tracker
    def _create_broadcast_path(self, 
                               state : SimulationAgentState, 
                               orbitdata : OrbitData = None,
                               t_init : float = None
                               ) -> tuple:
        """ Finds the best path for broadcasting a message to all agents using depth-first-search 
        
        ### Arguments:
            - state (`SimulationAgentState`): current state of the agent
            - orbitdata (`OrbitData`): coverage data of agent if it is of type `SatelliteAgent`
            - t_init (`float`): ealiest desired broadcast time
        """
        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Broadcast routing path not yet supported for agents of type `{type(state)}`')
        
        # get earliest desired broadcast time 
        t_init = state.t if t_init is None or t_init < state.t else t_init

        # populate list of all agents except the parent agent
        target_agents = [target_agent 
                         for target_agent in orbitdata.isl_data 
                         if target_agent != state.agent_name]
        
        if not target_agents: 
            # no other agents in the simulation; no need for relays
            return ([], t_init)
        
        # check if broadcast needs to be routed
        earliest_accesses = [   orbitdata.get_next_agent_access(target_agent, t_init) 
                                for target_agent in target_agents]           
        
        same_access_start = [   abs(access.left - earliest_accesses[0].left) < 1e-3
                                for access in earliest_accesses 
                                if isinstance(access, Interval)]
        same_access_end = [     abs(access.right - earliest_accesses[0].right) < 1e-3
                                for access in earliest_accesses 
                                if isinstance(access, Interval)]

        if all(same_access_start) and all(same_access_end):
            # all agents are accessing eachother at the same time; no need for mesasge relays
            return ([], t_init)   

        # look for relay path using depth-first search

        # initialize queue
        q = queue.Queue()
        
        # initialize min path and min path cost
        min_path = []
        min_times = []
        min_cost = np.Inf

        # add parent agent as the root node
        q.put((state.agent_name, [], [], 0.0))

        while not q.empty():
            # get next node in the search
            _, current_path, current_times, path_cost = q.get()

            # check if path is complete
            if len(target_agents) == len(current_path):
                # check if minimum cost
                if path_cost < min_cost:
                    min_cost = path_cost
                    min_path = [path_element for path_element in current_path]
                    min_times = [path_time for path_time in current_times]

            # add children nodes to queue
            for receiver_agent in [receiver_agent for receiver_agent in target_agents 
                                    if receiver_agent not in current_path
                                    and receiver_agent != state.agent_name
                                    ]:
                # query next access interval to children nodes
                t_access : float = state.t + path_cost

                access_interval : Interval = orbitdata.get_next_agent_access(receiver_agent, t_access)
                
                if access_interval.left < np.Inf:
                    new_path = [path_element for path_element in current_path]
                    new_path.append(receiver_agent)

                    new_cost = access_interval.left - state.t

                    new_times = [path_time for path_time in current_times]
                    new_times.append(new_cost + state.t)

                    q.put((receiver_agent, new_path, new_times, new_cost))

        # check if earliest broadcast time is valid
        if min_times: assert state.t <= min_times[0]

        # return path and broadcast start time
        return (min_path, min_times[0]) if min_path else ([], np.Inf)
    
    @runtime_tracker
    def _schedule_relay(self, relay_message : SimulationMessage) -> list:
        raise NotImplementedError('Relay scheduling not yet supported.')
        
        # check if relay message has a valid relay path
        assert relay.path

        # find next destination and access time
        next_dst = relay.path.pop(0)
        
        # query next access interval to children nodes
        sender_orbitdata : OrbitData = orbitdata[state.agent_name]
        access_interval : Interval = sender_orbitdata.get_next_agent_access(next_dst, state.t)
        t_start : float = access_interval.start

        if t_start < np.Inf:
            # if found, create broadcast action
            broadcast_action = BroadcastMessageAction(relay.to_dict(), t_start)
            
            # check broadcast start; only add to plan if it's within the planning horizon
            if t_start <= state.t + self.horizon:
                broadcasts.append(broadcast_action)

    @runtime_tracker
    def _schedule_maneuvers(    self, 
                                state : SimulationAgentState, 
                                specs : object,
                                observations : list,
                                clock_config : ClockConfig,
                                orbitdata : OrbitData = None
                            ) -> list:
        """
        Generates a list of AgentActions from the current path.

        Agents look to move to their designated measurement target and perform the measurement.

        ## Arguments:
            - state (:obj:`SimulationAgentState`): state of the agent at the start of the path
            - specs (`dict` or `Sapcecraft`): contains information regarding the physical specifications of the agent
            - path (`list`): list of tuples indicating the sequence of observations to be performed and time of observation
            - t_init (`float`): start time for plan
            - clock_config (:obj:`ClockConfig`): clock being used for this simulation
        """

        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Maneuver scheduling for agents of type `{type(state)}` not yet implemented.')
        elif not isinstance(specs, Spacecraft):
            raise ValueError(f'`specs` needs to be of type `Spacecraft` for agents of state type `{type(state)}`. Is of type `{type(specs)}`.')
        elif orbitdata is None:
            raise ValueError(f'`orbitdata` required for agents of type `{type(state)}`.')

        # compile instrument field of view specifications   
        cross_track_fovs = self.collect_fov_specs(specs)

        # get pointing agility specifications
        adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
        if adcs_specs is None: raise ValueError('ADCS component specifications missing from agent specs object.')

        max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
        if max_slew_rate is None: raise ValueError('ADCS `maxRate` specification missing from agent specs object.')

        max_torque = float(adcs_specs['maxTorque']) if adcs_specs.get('maxTorque', None) is not None else None
        if max_torque is None: raise ValueError('ADCS `maxTorque` specification missing from agent specs object.')

        # initialize maneuver list
        maneuvers : list[ManeuverAction] = []

        for i in tqdm(range(len(observations)), 
                      desc=f'{state.agent_name}-PLANNER: Scheduling Maneuvers', 
                      leave=False):

            curr_observation : ObservationAction = observations[i]

            # estimate previous state
            if i == 0:
                t_prev = state.t
                prev_state : SatelliteAgentState = state.copy()
                
            else:
                prev_observation : ObservationAction = observations[i-1]
                t_prev = prev_observation.t_end
                prev_state : SimulationAgentState = state.propagate(t_prev)
                prev_state.attitude = [prev_observation.look_angle, 0.0, 0.0]

            # maneuver to point to target
            if isinstance(state, SatelliteAgentState):
                prev_state : SatelliteAgentState
                
                dth_req = abs(curr_observation.look_angle - prev_state.attitude[0])
                dth_max = (curr_observation.t_start - prev_state.t) * max_slew_rate

                if dth_req > dth_max and abs(dth_req - dth_max) >= 1e-6: 
                    # maneuver impossible within timeframe
                    raise ValueError(f'Cannot schedule maneuver. Not enough time between observations')\
                
                # check if attitude maneuver is required
                if abs(dth_req) <= 1e-3: continue # already pointing in the same direction; ignore maneuver

                # calculate attitude duration    
                th_f = curr_observation.look_angle
                slew_rate = (curr_observation.look_angle - prev_state.attitude[0]) / dth_req * max_slew_rate
                dt = abs(th_f - prev_state.attitude[0]) / max_slew_rate

                # calculate maneuver time
                t_maneuver_start = curr_observation.t_start - dt
                t_maneuver_end = curr_observation.t_start

                # check if mnaeuver time is non-zero
                if abs(t_maneuver_start - t_maneuver_end) >= 1e-3:
                    # maneuver has non-zero duration; perform maneuver
                    maneuvers.append(ManeuverAction([th_f, 0, 0], 
                                                    [slew_rate, 0, 0],
                                                    t_maneuver_start, 
                                                    t_maneuver_end)) 

        maneuvers.sort(key=lambda a: a.t_start)

        assert self.is_maneuver_path_valid(state, specs, observations, maneuvers, max_slew_rate, cross_track_fovs)

        return maneuvers
    
    def is_maneuver_path_valid(self, 
                               state : SimulationAgentState, 
                               specs : object, 
                               observations : list, 
                               maneuvers : list,
                               max_slew_rate : float,
                               cross_track_fovs : dict
                               ) -> bool:

        for observation in observations:
            observation : ObservationAction

            # get fov for this observation's instrument
            cross_track_fov : float = cross_track_fovs[observation.instrument_name]

            # check if previous maneuvers were performed
            prev_maneuvers = [maneuver for maneuver in maneuvers
                              if maneuver.t_start <= observation.t_start]
            prev_maneuvers.sort(key=lambda a : a.t_start)

            if prev_maneuvers: # there was a maneuver performed before this observation
                # get latest maneuver
                latest_maneuver : ManeuverAction = prev_maneuvers.pop()

                # check status of completion of this maneuver
                if latest_maneuver.t_end < observation.t_start: # maneuver ended before observation started
                    # compare to final state after meneuver
                    dth = abs(observation.look_angle - latest_maneuver.final_attitude[0])

                else: # maneuver was being performed during meneuver
                    if prev_maneuvers:
                        prev_maneuver : ManeuverAction = prev_maneuvers.pop()
                        th_0 = prev_maneuver.final_attitude[0]
                    else:
                        th_0 = state.attitude[0]

                    dth = abs(observation.look_angle - th_0) - max_slew_rate * (observation.t_start - latest_maneuver.t_start) 

            else: # there were no maneuvers performed before this observation
                # compare to initial state
                dth = abs(observation.look_angle - state.attitude[0])

            if dth > cross_track_fov / 2.0 and abs(dth - cross_track_fov / 2.0) >= 1e-6:
                # latest state does not point towards the target at the intended look angle
                return False

        # all maneuvers passed checks; path is valid        
        return True
    
    def collect_fov_specs(self, specs : Spacecraft) -> dict:
        # compile instrument field of view specifications   
        cross_track_fovs = {instrument.name: np.NAN for instrument in specs.instrument}
        for instrument in specs.instrument:
            cross_track_fov = []
            for instrument_model in instrument.mode:
                if isinstance(instrument_model, BasicSensorModel):
                    instrument_fov : ViewGeometry = instrument_model.get_field_of_view()
                    instrument_fov_geometry : SphericalGeometry = instrument_fov.sph_geom
                    if instrument_fov_geometry.shape == 'RECTANGULAR':
                        cross_track_fov.append(instrument_fov_geometry.angle_width)
                    else:
                        raise NotImplementedError(f'Extraction of FOV for instruments with view geometry of shape `{instrument_fov_geometry.shape}` not yet implemented.')
                elif isinstance(instrument_model, PassiveOpticalScannerModel):
                    instrument_fov : ViewGeometry = instrument_model.get_field_of_view()
                    instrument_fov_geometry : SphericalGeometry = instrument_fov.sph_geom
                    if instrument_fov_geometry.shape == 'RECTANGULAR':
                        cross_track_fov.append(instrument_fov_geometry.angle_width)
                    else:
                        raise NotImplementedError(f'Extraction of FOV for instruments with view geometry of shape `{instrument_fov_geometry.shape}` not yet implemented.')
                else:
                    raise NotImplementedError(f'measurement data query not yet suported for sensor models of type {type(instrument_model)}.')
            cross_track_fovs[instrument.name] = max(cross_track_fov)

        return cross_track_fovs

    def collect_agility_specs(self, specs : Spacecraft) -> tuple:
        adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
        if adcs_specs is None: raise ValueError('ADCS component specifications missing from agent specs object.')

        max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
        if max_slew_rate is None: raise ValueError('ADCS `maxRate` specification missing from agent specs object.')

        max_torque = float(adcs_specs['maxTorque']) if adcs_specs.get('maxTorque', None) is not None else None
        if max_torque is None: raise ValueError('ADCS `maxTorque` specification missing from agent specs object.')

        return max_slew_rate, max_torque
        
    @runtime_tracker
    def is_observation_path_valid(self, 
                                  state : SimulationAgentState, 
                                  specs : object,
                                  observations : list
                                  ) -> bool:
        """ Checks if a given sequence of observations can be performed by a given agent """

        if isinstance(state, SatelliteAgentState) and isinstance(specs, Spacecraft):

            # get pointing agility specifications
            adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
            if adcs_specs is None: raise ValueError('ADCS component specifications missing from agent specs object.')

            max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
            if max_slew_rate is None: raise ValueError('ADCS `maxRate` specification missing from agent specs object.')

            max_torque = float(adcs_specs['maxTorque']) if adcs_specs.get('maxTorque', None) is not None else None
            if max_torque is None: raise ValueError('ADCS `maxTorque` specification missing from agent specs object.')
            
            # check if every observation can be reached from the prior measurement
            for j in range(len(observations)):

                # estimate the state of the agent at the given measurement
                observation_j : ObservationAction = observations[j]
                th_j = observation_j.look_angle
                t_j = observation_j.t_start
                # fov = cross_track_fovs[observation_j.instrument_name]

                # compare to prior measurements
                
                if j > 0: # there was a prior observation performed

                    # estimate the state of the agent at the prior mesurement
                    observation_i : ObservationAction = observations[j-1]
                    th_i = observation_i.look_angle
                    t_i = observation_i.t_end

                else: # there was prior measurement

                    # use agent's current state as previous state
                    th_i = state.attitude[0]
                    t_i = state.t                

                # check if desired instrument is contained within the satellite's specifications
                if observation_j.instrument_name not in [instrument.name for instrument in specs.instrument]:
                    return False 
                
                assert th_j != np.NAN and th_i != np.NAN # TODO: add case where the target is not visible by the agent at the desired time according to the precalculated orbitdata

                # estimate maneuver time betweem states
                dt_maneuver = abs(th_j - th_i) / max_slew_rate

                # calculate time between measuremnets
                dt_measurements = t_j - t_i

                # check if observation sequence is correct 
                if dt_measurements < 0.0:
                    return False

                # Slewing constraint: check if there's enough time to maneuver from one observation to another
                if dt_maneuver > dt_measurements and abs(dt_maneuver - dt_measurements) > 1e-6:
                    # there is not enough time to maneuver; flag current observation plan as unfeasible for rescheduling
                    return False              
                
                # Torque constraint:
                # TODO check if the agent has enough torque in its reaction wheels to perform the maneuver
                            
            # if all measurements passed the check; observation path is valid
            return True
        else:
            raise NotImplementedError(f'Observation path validity check for agents with state type {type(state)} not yet implemented.')
        
    # def _print_observation_sequence(self, 
    #                                 state : SatelliteAgentState, 
    #                                 path : list, 
    #                                 orbitdata : OrbitData = None
    #                                 ) -> None :
    #     """ Debugging tool. Prints current observation sequence being considered. """

    #     if not isinstance(state, SatelliteAgentState):
    #         raise NotImplementedError('Observation sequence printouts for non-satellite agents not yet supported.')
    #     elif orbitdata is None:
    #         raise ValueError(f'`orbitdata` required for agents of type `{type(state)}`.')

    #     out = f'\n{state.agent_name}:\n\n\ntarget\tinstr\tt_img\tth\tdt_mmt\tdt_mvr\tValid?\n'

    #     out_temp = [f"N\A       ",
    #                 f"N\A",
    #                 f"\t{np.round(state.t,3)}",
    #                 f"\t{np.round(state.attitude[0],3)}",
    #                 f"\t-",
    #                 f"\t-",
    #                 f"\t-",
    #                 f"\n"
    #                 ]
    #     out += ''.join(out_temp)

    #     for i in range(len(path)):
    #         if i > 0:
    #             measurement_prev : ObservationAction = path[i-1]
    #             t_prev = measurement_i.t_end
    #             lat,lon,_ = measurement_prev.target
    #             obs_prev = orbitdata.get_groundpoint_access_data(lat, lon, measurement_prev.instrument_name, t_prev)
    #             th_prev = obs_prev['look angle [deg]']
    #         else:
    #             t_prev = state.t
    #             th_prev = state.attitude[0]

    #         measurement_i : ObservationAction = path[i]
    #         t_i = measurement_i.t_start
    #         lat,lon,alt = measurement_i.target
    #         obs_i = orbitdata.get_groundpoint_access_data(lat, lon, measurement_i.instrument_name, t_i)
    #         th_i = obs_i['look angle [deg]']

    #         dt_maneuver = abs(th_i - th_prev) / state.max_slew_rate
    #         dt_measurements = t_i - t_prev

    #         out_temp = [f"({round(lat,3)}, {round(lon,3)}, {round(alt,3)})",
    #                         f"  {measurement_i.instrument_name}",
    #                         f"\t{np.round(measurement_i.t_start,3)}",
    #                         f"\t{np.round(th_i,3)}",
    #                         f"\t{np.round(dt_measurements,3)}",
    #                         f"\t{np.round(dt_maneuver,3)}",
    #                         f"\t{dt_maneuver <= dt_measurements}",
    #                         f"\n"
    #                         ]
    #         out += ''.join(out_temp)
    #     out += f'\nn measurements: {len(path)}\n'

    #     print(out)

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
        cross_track_fovs : dict = self.collect_fov_specs(specs)

        # calculate coverage opportunities for tasks
        access_opportunities = self.calculate_access_opportunities(state, specs, orbitdata)

        # get only available tasks
        available_tasks : list[GenericObservationTask] = self.get_available_tasks(tasks, state.t)

        # create schedulable tasks from known tasks and future access opportunities
        schedulable_tasks : list[SchedulableObservationTask] = self.create_tasks_from_accesses(available_tasks, access_opportunities, cross_track_fovs)
        
        # schedule observation tasks
        observations : list = self._schedule_observations(state, specs, clock_config, orbitdata, schedulable_tasks, observation_history)
        assert isinstance(observations, list) and all([isinstance(obs, ObservationAction) for obs in observations]), \
            f'Observation actions not generated correctly. Is of type `{type(observations)}` with elements of type `{type(observations[0])}`.'
        assert self.is_observation_path_valid(state, specs, observations)

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
    def calculate_access_opportunities(self, 
                                       state : SimulationAgentState, 
                                       specs : Spacecraft,
                                       orbitdata : OrbitData
                                    ) -> dict:
        # define planning horizon
        t_start = state.t
        t_end = t_start+self.horizon

        # compile coverage data
        raw_coverage_data : dict = orbitdata.gp_access_data.lookup_interval(t_start, t_end)

        # initiate accestimes 
        access_opportunities = {}
        
        for i in tqdm(range(len(raw_coverage_data['time [s]'])), 
                        desc=f'{state.agent_name}-PREPLANNER: Compiling access opportunities', 
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
                access_opportunities[grid_index][gp_index] = {instr.name : [] 
                                                        for instr in specs.instrument}

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

        # convert to `list`
        access_opportunities = [    (grid_index, gp_index, instrument, interval, t, th)
                                    for grid_index in access_opportunities
                                    for gp_index in access_opportunities[grid_index]
                                    for instrument in access_opportunities[grid_index][gp_index]
                                    for interval, t, th in access_opportunities[grid_index][gp_index][instrument]
                                ]
                
        # return access times and grid information
        return access_opportunities
        
    @runtime_tracker
    def get_available_tasks(self, tasks : list, t : float) -> list:
        """ Returns a list of tasks that are available at the given time """
        if not isinstance(tasks, list):
            raise ValueError(f'`tasks` needs to be of type `list`. Is of type `{type(tasks)}`.')
        
        # TODO add check for capability of the agent to perform the task

        return [task for task in tasks if task.available(t)]

    @abstractmethod
    def _schedule_observations(self, state : SimulationAgentState, specs : object, clock_config : ClockConfig, orbitdata : OrbitData, schedulable_tasks : list, observation_history : ObservationHistory) -> list:
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


class AbstractReplanner(AbstractPlanner):
    """ Repairs plans previously constructed by another planner """

    def __init__(self, debug: bool = False, logger: Logger = None) -> None:
        super().__init__(debug, logger)

        self.preplan : Preplan = None

    @abstractmethod
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
        
        super().update_percepts(state, incoming_reqs, relay_messages, completed_actions)
        
        # update latest preplan
        if abs(state.t - current_plan.t) <= 1e-3 and isinstance(current_plan, Preplan): 
            self.preplan : Preplan = current_plan.copy() 

    @abstractmethod
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
        pass

    def get_broadcast_contents(self,
                               broadcast_action : FutureBroadcastMessageAction,
                               state : SimulationAgentState,
                               observation_history : ObservationHistory,
                               **kwargs
                               ) -> BroadcastMessageAction:
        """  Generates a broadcast message to be sent to other agents """
        raise NotImplementedError('Broadcast contents generation not yet implemented for this planner.')

        if broadcast_action.broadcast_type == FutureBroadcastTypes.REWARD:
            # raise NotImplementedError('Reward broadcast not yet implemented.')

            # compile latest observations from the observation history
            latest_observations : list[ObservationAction] = [observation_tracker.latest_observation
                                                             for grid_index,grid in observation_history.history.items()
                                                             for gp_index, observation_tracker in grid.items()
                                                             if observation_tracker.latest_observation is not None
                                                             # TODO ONLY include observations that are within the period of the broadcast
                                                             # and state.t - self.period <= observation_tracker.latest_observation.t_start
                                                             ]
            
            # index by instrument name
            instruments_used : set = {latest_observation['instrument_name'] 
                                      for latest_observation in latest_observations}
            indexed_observations = {instrument_used: [latest_observation for latest_observation in latest_observations
                                                      if latest_observation['instrument_name'] == instrument_used]
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
            if self.known_reqs:
                x =1
            
            msgs = [MeasurementRequestMessage(state.agent_name, state.agent_name, req.to_dict())
                    for req in self.known_reqs
                    if isinstance(req, TaskRequest)
                    and req.event.t_start <= state.t <= req.event.t_end]
        else:
            raise ValueError(f'`{broadcast_action.broadcast_type}` broadcast type not supported.')

        # construct bus message
        bus_msg = BusMessage(state.agent_name, state.agent_name, [msg.to_dict() for msg in msgs])

        # return bus message broadcast (if not empty)
        return BroadcastMessageAction(bus_msg.to_dict(), broadcast_action.t_start)