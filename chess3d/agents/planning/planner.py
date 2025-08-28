
from collections import defaultdict
from functools import reduce
import queue

from instrupy.base import BasicSensorModel
from instrupy.passive_optical_scanner_model import PassiveOpticalScannerModel
from instrupy.util import ViewGeometry, SphericalGeometry
from orbitpy.util import Spacecraft

from dmas.modules import *
from dmas.utils import runtime_tracker
from tqdm import tqdm

from chess3d.agents.planning.plan import Plan
from chess3d.agents.planning.tasks import EventObservationTask, GenericObservationTask, SpecificObservationTask
from chess3d.agents.planning.tracker import ObservationHistory, ObservationTracker
from chess3d.agents.states import *
from chess3d.agents.science.requests import *
from chess3d.messages import *
from chess3d.orbitdata import OrbitData
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
                                    cross_track_fovs : dict,
                                    orbitdata : OrbitData,
                                    must_overlap : bool = True,
                                    threshold : float = 5*60
                                    ) -> list:
        """ 
        Creates specific observation tasks from precalculated access times of known generic task targets. 

        #### Arguments
        - `available_tasks` : List of known and available generic observation tasks.
        - `access_times` : List of access times for each generic observation task.
        - `cross_track_fovs` : Dictionary of cross-track fields of view for each instrument.
        - `orbitdata` : Precalculated orbit and coverage data for the mission.
        - `must_overlap` : Whether tasks' availability must overlap in availability time to be considered for clustering. (default : True)
        - `threshold` : The time threshold for clustering tasks in seconds [s] (default : 300 seconds = 5 minutes).
        """

        if not must_overlap: raise NotImplementedError('Clustering without overlap is not yet fully implemented.')

        # generate schedulable tasks from access times
        schedulable_tasks : list[SpecificObservationTask] \
            = self.single_tasks_from_accesses(available_tasks, access_times, cross_track_fovs, orbitdata)
        
        # check if tasks are clusterable
        task_adjacency : Dict[str, set[SpecificObservationTask]] \
            = self.check_task_clusterability(schedulable_tasks, must_overlap, threshold)
   
        # cluster tasks based on adjacency
        combined_tasks : list[SpecificObservationTask] = self.cluster_tasks(schedulable_tasks, task_adjacency, must_overlap, threshold)

        # add clustered tasks to the final list of tasks available for scheduling
        schedulable_tasks.extend(combined_tasks) 

        assert all([task.slew_angles.span()-1e-6 <= cross_track_fovs[task.instrument_name] 
                    for task in schedulable_tasks]), \
            f"Tasks have slew angles larger than the maximum allowed field of view."

        # return tasks
        return schedulable_tasks
    
    @runtime_tracker
    def single_tasks_from_accesses(self,
                                   available_tasks : list,
                                   access_times : list, 
                                   cross_track_fovs : dict,
                                   orbitdata : OrbitData,
                                   threshold : float = 1e-9
                                   ) -> list:
        """ Creates one specific task per each access opportunity for every available task """

        # initialize list of schedulable tasks
        schedulable_tasks : list[SpecificObservationTask] = []

        # create one task per each access opportunity
        for task in tqdm(available_tasks, desc="Calculating access times to known tasks", leave=False):
            task : GenericObservationTask

            # TODO improve minimum and maximum measurement duration requirement calculation
            if task.objective is not None:
                duration_reqs = [req for req in task.objective
                                if isinstance(req, MeasurementDurationRequirement)]
                duration_req : MeasurementDurationRequirement = duration_reqs[0] if duration_reqs else None
            else:
                duration_req = None
            min_duration_req : float = min(duration_req.thresholds) if duration_req is not None else orbitdata.time_step
            assert isinstance(min_duration_req, (int,float)) and min_duration_req >= 0.0, "minimum duration requirement must be a positive number."

            # find access time for each target location for this task
            for *__,grid_index,gp_index in task.location:
                # ensure grid_index and gp_index are integers
                grid_index,gp_index = int(grid_index), int(gp_index)
                
                # check if target is accessible  
                if grid_index not in access_times or gp_index not in access_times[grid_index]:
                    continue

                # get access times for this ground point and grid index
                matching_access_times = [
                                        (instrument, access_interval, t, th)
                                        for instrument in access_times[grid_index][gp_index]
                                        for access_interval,t,th in access_times[grid_index][gp_index][instrument]
                                        if task.availability.overlaps(access_interval)
                                        ]
                
                # create a schedulable task for each access time
                for access_time in matching_access_times:
                    # unpack access time
                    instrument_name,accessibility,_,th = access_time
                    accessibility : Interval
                    
                    if max(th) - min(th) > cross_track_fovs[instrument_name]:
                        # not all of the accessibility is observable with a single pass
                        continue
                        # TODO raise NotImplementedError('No support for tasks that require multiple passes yet.')
                    else:
                        off_axis_angles = [Interval(off_axis_angle - cross_track_fovs[instrument_name]/2,
                                                    off_axis_angle + cross_track_fovs[instrument_name]/2)
                                                    for off_axis_angle in th]
                        slew_angles : Interval = reduce(lambda a, b: a.intersection(b), off_axis_angles)

                    if slew_angles.is_empty(): continue  # skip if no valid slew angles

                    # check if instrument can perform the task                    
                    if not self.can_perform_task(task, instrument_name): 
                        continue # skip if not

                    # check if access time matches task availability
                    if not task.availability.overlaps(accessibility):
                        continue # skip if not

                    # check if access time is enough to perform the task
                    if min_duration_req > accessibility.span():
                        # check if accessibility span is non-zero
                        if accessibility.span() <= 0.0: continue # accessibility time is too short; skip
    
                        # check if available timespan longer than the minimum observation duration
                        if accessibility.span() - min_duration_req >= threshold: continue # is over the threshold; skip 

                        # create and add schedulable task to list of schedulable tasks with a different minimum observation requirement
                        schedulable_tasks.append(SpecificObservationTask(task,
                                                                        instrument_name,
                                                                        accessibility,
                                                                        accessibility.span(), # slightly shorter than `min_duration_req`
                                                                        slew_angles
                                                                        ))
                    else:
                        # create and add schedulable task to list of schedulable tasks
                        schedulable_tasks.append(SpecificObservationTask(task,
                                                                        instrument_name,
                                                                        accessibility,
                                                                        min_duration_req,
                                                                        slew_angles
                                                                        ))

        
        # return list of schedulable tasks
        return schedulable_tasks
            
    def can_perform_task(self, task : GenericObservationTask, instrument_name : str) -> bool:
        """ Checks if the agent can perform the task at hand with the given instrument """
        # TODO Replace this with KG for better reasoning capabilities

        # Check if task has specified objectives
        if task.objective is not None:
            # Extract capability requirements from the objective
            capability_reqs = [req for req in task.objective
                               if isinstance(req, CapabilityRequirement)]
            capability_req: CapabilityRequirement = capability_reqs[0] if capability_reqs else None

            # Evaluate capability requirement
            if capability_req is not None:
                return capability_req.calc_preference_value(instrument_name) >= 0.5

        # No capability objectives specified; check if instrument has general capability
        # TODO replace with better reasoning; currently assumes instrument has general capability
        return True
        
    @runtime_tracker
    def check_task_clusterability(self, schedulable_tasks : list, must_overlap : bool, threshold : float) -> dict:
        """ 
        Creates adjacency list for a given list of specific observation tasks.

        #### Arguments
        - `schedulable_tasks` : A list of specific observation tasks to create the adjacency list for.
        - `must_overlap` : Whether tasks' availability must overlap in availability time to be considered for clustering.
        - `threshold` : The time threshold for clustering tasks in seconds [s].
        """
        schedulable_tasks : list[SpecificObservationTask] = schedulable_tasks

        # create adjacency list for tasks
        adj : Dict[str, set[SpecificObservationTask]] = {task.id : set() for task in schedulable_tasks}
                
        if schedulable_tasks:
            # sort tasks by accessibility
            schedulable_tasks.sort(key=lambda a : a.accessibility) 
            
            # get min and max accessibility times
            t_min = schedulable_tasks[0].accessibility.left

            # initialize bins
            bins = defaultdict(list)
            
            # group task in bins by accessibility
            for task in tqdm(schedulable_tasks, leave=False, desc="Grouping tasks into bins"):
                task : SpecificObservationTask
                center_time = (task.accessibility.left + task.accessibility.right) / 2 - t_min
                bin_key = int(center_time // threshold)
                bins[bin_key].append(task)

            # populate adjacency list
            with tqdm(total=len(schedulable_tasks), desc="Checking task clusterability", leave=False) as pbar:
                for b in bins:
                    candidates : list[SpecificObservationTask]\
                          = bins[b] + bins.get(b + 1, [])  # optionally add b-1 for symmetry
                    for i in range(len(candidates)):
                        for j in range(i + 1, len(candidates)):
                            t1, t2 = candidates[i], candidates[j]
                            if t1.can_merge(t2, must_overlap=must_overlap, max_duration=threshold):
                                adj[t1.id].add(t2)
                                adj[t2.id].add(t1)
                        pbar.update(1)

        # check if adjacency list is symmetric
        for p in schedulable_tasks:
            assert p not in adj[p.id], f'Task {p.id} is in its own adjacency list.'
            for q in adj[p.id]:
                assert p in adj[q.id], f'Task {p.id} is in the adjacency list of task {q.id} but not vice versa.'

        return adj

    @runtime_tracker
    def cluster_tasks(self, schedulable_tasks : list, adj : dict, must_overlap : bool, threshold : float) -> list:
        """ 
        Clusters tasks based on adjacency. 
        
        ```
        while V!=Ø do
            Pick a vertex p with largest degree from V. 
                If such p are not unique, pick the p with highest priority.
            
            while N(p)=Ø do
                Pick a neighbor q of p, q ∈ N(p), such that the number of their common neighbors is maximum. 
                    If such p are not unique, pick the p with least edges being deleted.
                    Again, if such p are still not unique, pick the p with highest priority.
                Combine q and p into a new p
                Delete edges from q and p that are not connected to their common neighbors
                Reset neighbor collection N(p) for the new p
            end while
            
            Output the cluster-task denoted by p
            Delete p from V
        end while
        ```
        
        """         
        schedulable_tasks : list[SpecificObservationTask]
        adj : Dict[str, set[SpecificObservationTask]] = adj

        # only keep tasks that have at least one clusterable task
        v = [task for task in schedulable_tasks if len(adj[task.id]) > 0]
        
        # sort tasks by degree of adjacency 
        v : list[SpecificObservationTask] = self.sort_tasks_by_degree(schedulable_tasks, adj)
        
        # combine tasks into clusters
        combined_tasks : list[SpecificObservationTask] = []

        with tqdm(total=len(v), desc="Merging overlapping tasks", leave=False) as pbar:
            while len(v) > 0:
                # pop first task from the list of tasks to be scheduled
                p : SpecificObservationTask = v.pop()

                # get list of neighbors of p sorted by number of common neighbors
                n_p : list[SpecificObservationTask] = self.sort_tasks_by_common_neighbors(p, list(adj[p.id]), adj)

                # initialize clique with p
                clique = set()

                # update progress bar
                pbar.update(1)

                # while there are neighbors of p
                while len(n_p) > 0:
                    # pop first neighbor q from the list of neighbors
                    q : SpecificObservationTask = n_p.pop()

                    # Combine q and p into a new p                 
                    clique.add(q)

                    # find common neighbors of p and q
                    common_neighbors : set[SpecificObservationTask] = adj[p.id].intersection(adj[q.id])
                   
                    # remove edges to p and q that do not include common neighbors
                    for neighbor in adj[p.id].difference(common_neighbors): adj[neighbor.id].discard(p)
                    for neighbor in adj[q.id]: adj[neighbor.id].discard(q)              
                    
                    # update edges of p and q to only include common neighbors
                    adj[p.id].intersection_update(common_neighbors)
                    
                    # remove q from the adjacency list
                    adj.pop(q.id)

                    # remove q from the list of tasks to be scheduled
                    v.remove(q)

                    # Reset neighbor collection N_p for the new p;
                    n_p : list[SpecificObservationTask] = self.sort_tasks_by_common_neighbors(p, list(adj[p.id]), adj)               

                for q in clique: 
                    # TODO: look into ID being used. Ideally we would want a new ID for the combined task.

                    # merge all tasks in the clique into a single task p
                    p = p.merge(q, must_overlap=must_overlap, max_duration=threshold)  # max duration of 5 minutes

                    # update progress bar
                    pbar.update(1)

                # DEBUGGING--------- 
                # clique.add(p)
                # cliques.append(sorted([schedulable_tasks.index(t)+1 for t in clique]))
                # ------------------

                # add merged task to the list of combined tasks
                combined_tasks.append(p)

                # sort remaining schedulable tasks by degree of adjacency 
                v : list[SpecificObservationTask] = self.sort_tasks_by_degree(v, adj)
        
        return combined_tasks

    @runtime_tracker
    def sort_tasks_by_degree(self, tasks : list, adjacency : dict) -> list:
        """ Sorts tasks by degree of adjacency. """
        # calculate degree of each task
        degrees : dict = {task : len(adjacency[task.id]) for task in tasks}

        # sort tasks by degree and return
        return sorted(tasks, key=lambda p: (degrees[p], sum([parent_task.priority for parent_task in p.parent_tasks]), -p.accessibility.left))

    def sort_tasks_by_common_neighbors(self, p : SpecificObservationTask, n_p : list, adjacency : dict) -> list:
        # specify types
        n_p : list[SpecificObservationTask] = n_p
        adjacency : Dict[str, set[SpecificObservationTask]] = adjacency

        # calculate common neighbors
        common_neighbors : dict = {q : adjacency[p.id].intersection(adjacency[q.id]) 
                                   for q in n_p}
        
        # calculate neighbors to delete
        neighbors_to_delete : dict = {q : adjacency[p.id].difference(adjacency[q.id])
                                      for q in n_p}
        
        # sort neighbors by number of common neighbors, number of edges to delete, priority and accessibility
        return sorted(n_p, 
                      key=lambda p: (len(common_neighbors[p]), 
                                     -len(neighbors_to_delete[p]),
                                     sum([parent_task.priority for parent_task in p.parent_tasks]), 
                                     -p.accessibility.left))

    @runtime_tracker
    def estimate_task_value(self, 
                            task : SpecificObservationTask, 
                            t_img : float,
                            d_img : float,
                            specs : Spacecraft, 
                            cross_track_fovs : dict,
                            orbitdata : OrbitData,
                            mission : Mission,
                            observation_history : ObservationHistory
                            ) -> float:
        """ Estimates task value based on predicted observation performance. """

        # estimate measurement performance metrics
        measurement_performance_metrics : dict = self.estimate_observation_performance_metrics(task, t_img, d_img, specs, cross_track_fovs, orbitdata, observation_history)

        # check if measurement performance is valid
        if measurement_performance_metrics is None: return 0.0

        # calculate and return total task reward
        return mission.calc_specific_task_value(task, measurement_performance_metrics)

    @runtime_tracker    
    def estimate_observation_performance_metrics(self, 
                                         task : SpecificObservationTask, 
                                         t_img : float,
                                         d_img : float,
                                         specs : Spacecraft, 
                                         cross_track_fovs : dict,
                                         orbitdata : OrbitData,
                                         observation_history : ObservationHistory
                                        ) -> dict:

        # get available access metrics
        observation_performances = self.get_available_accesses(task, t_img, d_img, orbitdata, cross_track_fovs)
        observed_locations = list({(observation_performances['lat [deg]'][i], 
                                    observation_performances['lon [deg]'][i],
                                    observation_performances['grid index'][i], 
                                    observation_performances['GP index'][i]) 
                                     for i in range(len(observation_performances['time [s]']))
                                     })

        # check if there are no valid observations for this task
        if any([len(observation_performances[col]) == 0 for col in observation_performances]): 
            # no valid accesses; no reward added
            return None

        # get instrument specifications
        instrument_spec : BasicSensorModel = next(instr 
                                                  for instr in specs.instrument
                                                  if instr.name.lower() == task.instrument_name.lower()).mode[0]
                
        # get previous observation information
        prev_obs : list[ObservationTracker] = [observation_history.get_observation_history(grid_index, gp_index)
                    for *_,grid_index,gp_index in observed_locations]

        # get current observation information 
        observation_performance_metrics = {col.lower() : observation_performances[col][-1] 
                                    for col in observation_performances}
        observation_performance_metrics.update({ 
                "location" : observed_locations,
                "t_start" : t_img,
                "t_end" : t_img + d_img,
                "duration" : d_img,
                "n_obs" : sum(obs.n_obs for obs in prev_obs),
                "revisit_time" : min(t_img - obs.t_last for obs in prev_obs),
                "horizontal_spatial_resolution" : observation_performance_metrics['ground pixel cross-track resolution [m]'],
            })

        # package observation performance information
        if 'vnir' in task.instrument_name.lower() or 'tir' in task.instrument_name.lower():
            observation_performance_metrics.update({
                'spectral_resolution' : instrument_spec.spectral_resolution
            })
        elif 'altimeter' in task.instrument_name.lower():
            observation_performance_metrics.update({
                "accuracy" : observation_performance_metrics['accuracy [m]'],
            })
        else:
            raise NotImplementedError(f'Calculation of task reward not yet supported for instruments of type `{task.instrument_name}`.')

        return observation_performance_metrics

    def get_available_accesses(self, 
                               task : SpecificObservationTask, 
                               t_img : float,
                               d_img : float,
                               orbitdata : OrbitData, 
                               cross_track_fovs : dict
                            ) -> dict:
        """ Uses pre-computed orbitdata to estimate observation metrics for a given task. """
        
        # get task targets
        task_targets = {(int(grid_index), int(gp_index))
                        for parent_task in task.parent_tasks
                        for *_,grid_index,gp_index in parent_task.location}
        
        # estimate observation look angle
        th_img = np.average((task.slew_angles.left, task.slew_angles.right))

        # get ground points accessesible during the availability of the task
        raw_access_data : Dict[str,list] = orbitdata.gp_access_data.lookup_interval(t_img, t_img + d_img)

        # extract ground point accesses that are within the agent's field of view
        accessible_gps_data_indeces = [i for i in range(len(raw_access_data['time [s]']))
                                        if abs(raw_access_data['look angle [deg]'][i] - th_img) \
                                            <= cross_track_fovs[task.instrument_name] / 2
                                        and raw_access_data['instrument'][i] == task.instrument_name]
        accessible_gps_performances = {col : [raw_access_data[col][i] 
                                              for i in accessible_gps_data_indeces]
                                    for col in raw_access_data}
        
        # extract gp accesses of the desired targets within the task's accessibility and agent's field of view
        valid_access_data_indeces = [i for i in range(len(accessible_gps_performances['time [s]']))
                                    if (int(accessible_gps_performances['grid index'][i]), \
                                        int(accessible_gps_performances['GP index'][i])) in task_targets]
        observation_performances = {col : [accessible_gps_performances[col][i] 
                                           for i in valid_access_data_indeces]
                                    for col in accessible_gps_performances}
        
        # return estimated observation performances
        return observation_performances

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
                                  observations : list,
                                  **kwargs
                                  ) -> bool:
        """ Checks if a given sequence of observations can be performed by a given agent """
        try:
            if isinstance(state, SatelliteAgentState) and isinstance(specs, Spacecraft):

                # get pointing agility specifications
                adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
                if adcs_specs is None: raise ValueError('ADCS component specifications missing from agent specs object.')

                max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
                if max_slew_rate is None: raise ValueError('ADCS `maxRate` specification missing from agent specs object.')

                max_torque = float(adcs_specs['maxTorque']) if adcs_specs.get('maxTorque', None) is not None else None
                if max_torque is None: raise ValueError('ADCS `maxTorque` specification missing from agent specs object.')

                # construct observation sequence parameter list
                observation_parameters = []
                for j,observation_j in enumerate(observations):
                    # estimate the state of the agent at the given measurement
                    observation_j : ObservationAction
                    th_j = observation_j.look_angle
                    t_j = observation_j.t_start
                    d_j = observation_j.t_end - t_j

                    # compare to prior measurements
                    if j > 0: # there was a prior observation performed
                        # estimate the state of the agent at the prior mesurement
                        observation_i : ObservationAction = observations[j-1]
                        th_i = observation_i.look_angle
                        t_i = observation_i.t_start
                        d_i = observation_i.t_end - t_i

                    else: # there was no prior measurement
                        # use agent's current state as previous state
                        th_i = state.attitude[0]
                        t_i = state.t
                        d_i = 0.0

                    observation_parameters.append((th_i, t_i, d_i, th_j, t_j, d_j, max_slew_rate))

                # check if all observations are valid
                return all([self.is_observation_pair_valid(*params) 
                            for params in observation_parameters])

            else:
                raise NotImplementedError(f'Observation path validity check for agents with state type {type(state)} not yet implemented.')
        finally:
            for th_i,t_i,d_i,th_j,t_j,d_j,max_slew_rate in observation_parameters:
                if not self.is_observation_pair_valid(th_i, t_i, d_i, th_j, t_j, d_j, max_slew_rate):
                    x = 1

    def is_observation_pair_valid(self, 
                                  th_i, t_i, d_i, 
                                  th_j, t_j, d_j, 
                                  max_slew_rate):
        # check inputs
        assert not np.isnan(th_j) and not np.isnan(th_i) # TODO: add case where the target is not visible by the agent at the desired time according to the precalculated orbitdata

        # calculate maneuver time betweem states
        dt_maneuver = abs(th_j - th_i) / max_slew_rate
        
        # calculate time between measuremnets
        dt_measurements = t_j - (t_i + d_i)

        return ((dt_measurements > dt_maneuver 
                or abs(dt_measurements - dt_maneuver) < 1e-6)   # there is enough time to maneuver
                and dt_measurements >= -1e-6)                   # measurement time is after the previous measurement
                 
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
