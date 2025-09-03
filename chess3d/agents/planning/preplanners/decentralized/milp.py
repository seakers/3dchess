import os
from chess3d.agents.planning.planner import *
from chess3d.agents.planning.preplanners.preplanner import AbstractPreplanner

from collections import defaultdict, deque
import gurobipy as gp
import numpy as np

class SingleSatMILP(AbstractPreplanner):
    def __init__(self, 
                 objective : str, 
                 licence_path : str, 
                 horizon = np.Inf, 
                 period = np.Inf, 
                 max_tasks = np.Inf,
                 debug = False, 
                 logger = None
                ):
        super().__init__(horizon, period, debug, logger)

        if not debug:
            # Check for Gurobi license
            assert os.path.isfile(licence_path), f"Provided Gurobi licence path `{licence_path}` is not a valid file."

            # Set Gurobi license environment variable
            os.environ['GRB_LICENSE_FILE'] = licence_path

        # Validate inputs
        assert objective in ["reward", "duration"], "Objective must be either 'reward' or 'duration'."
        assert (isinstance(max_tasks, int) and max_tasks > 0) or max_tasks == np.Inf, "Max tasks must be a positive integer."

        # Set attributes
        self.objective = objective
        self.max_tasks = max_tasks

    @runtime_tracker
    def _schedule_observations(self, 
                               state : SimulationAgentState, 
                               specs : object, 
                               _ : ClockConfig, 
                               orbitdata : OrbitData, 
                               schedulable_tasks : list, 
                               mission : Mission, 
                               observation_history : ObservationHistory
                               ) -> list:
        # Check if there are no tasks to schedule
        if not schedulable_tasks: return []
        
        # Set type for `schedulable_tasks` and sort in ascending start time access
        schedulable_tasks : list[SpecificObservationTask] = sorted(schedulable_tasks, key=lambda x: x.accessibility.left)
        
        # validate Inputs
        assert all(isinstance(task, SpecificObservationTask) for task in schedulable_tasks), "All tasks must be of type `SpecificObservationTask`."
        assert all(task.min_duration > 0 for task in schedulable_tasks), "All tasks must have positive duration requirements."
        t_max = max((task.accessibility.right for task in schedulable_tasks if isinstance(task, SpecificObservationTask)))
        assert t_max <= state.t + self.horizon, f"Tasks exceed the planning horizon of {self.horizon} seconds."

        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Naive planner not yet implemented for agents of type `{type(state)}.`')
        elif not isinstance(specs, Spacecraft):
            raise ValueError(f'`specs` needs to be of type `{Spacecraft}` for agents with states of type `{SatelliteAgentState}`')
                
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self.collect_fov_specs(specs)
        
        # get pointing agility specifications
        adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
        assert adcs_specs, 'ADCS component specifications missing from agent specs object.'

        max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
        assert max_slew_rate, 'ADCS `maxRate` specification missing from agent specs object.'

        max_torque = float(adcs_specs['maxTorque']) if adcs_specs.get('maxTorque', None) is not None else None
        assert max_torque, 'ADCS `maxTorque` specification missing from agent specs object.'

        # Decide how to split tasks into chunks according to `max_tasks`
        if self.max_tasks < np.Inf:
            task_chunks = [
                schedulable_tasks[i:i + self.max_tasks] 
                for i in range(0, len(schedulable_tasks), self.max_tasks)
            ]
        else:
            task_chunks = [schedulable_tasks]

        # Run optimization on each chunk and flatten results
        observations : list[ObservationAction] = []
        curr_state : SimulationAgentState = state.copy()
        for tasks in task_chunks:
            # Update state
            if observations:
                last_action : ObservationAction = observations[-1]
                curr_state = curr_state.propagate(last_action.t_end)
                curr_state.attitude[0] = last_action.look_angle

            # Filter tasks based on current state
            reduced_tasks = [task for task in tasks
                     if task.accessibility.right - max(task.accessibility.left, curr_state.t) >= task.min_duration
                     and all(not task.is_mutually_exclusive(obs.task) for obs in observations)]
                     
            for task in reduced_tasks:
                if curr_state.t in task.accessibility:
                    task.accessibility = Interval(curr_state.t, task.accessibility.right)

            # Calculate optimized observation schedule
            t_ref = time.perf_counter()
            _,x_0,__,tau_0,obj_0 = self.__initialize_observations_schedule(curr_state, 
                                                                        specs,
                                                                        cross_track_fovs,
                                                                        max_slew_rate, 
                                                                        max_torque, 
                                                                        orbitdata,
                                                                        reduced_tasks, 
                                                                        mission, 
                                                                        observation_history
                                                                    )
            t_0 = time.perf_counter() - t_ref

            t_ref = time.perf_counter()
            _,x_1,__,tau_1,obj_1 = self.__optimize_observations_schedule(curr_state, 
                                                                        specs,
                                                                        cross_track_fovs,
                                                                        max_slew_rate, 
                                                                        max_torque, 
                                                                        orbitdata,
                                                                        reduced_tasks, 
                                                                        mission, 
                                                                        observation_history
                                                                    )
            t_1 = time.perf_counter() - t_ref

            # t_ref = time.perf_counter()
            # _,x_2,__,tau_2,obj_2 = self.__optimize_observations_schedule(curr_state, 
            #                                                             specs,
            #                                                             cross_track_fovs,
            #                                                             max_slew_rate, 
            #                                                             max_torque, 
            #                                                             orbitdata,
            #                                                             reduced_tasks, 
            #                                                             mission, 
            #                                                             observation_history,
            #                                                             x_0
            #                                                         )
            # t_2 = time.perf_counter() - t_ref

            # Extract observation sequence
            d = [task.min_duration for task in reduced_tasks if isinstance(task, SpecificObservationTask)]
            th_imgs = [np.average((task.slew_angles.left, task.slew_angles.right)) for task in reduced_tasks if isinstance(task, SpecificObservationTask)]

            # Append to list of observations
            observations.extend(self.__extract_observation_sequence(reduced_tasks, curr_state, x_1, tau_1, d, th_imgs))

            # DEBUG
            obs_0 = self.__extract_observation_sequence(reduced_tasks, curr_state, x_0, tau_0, d, th_imgs)
            obs_1 = self.__extract_observation_sequence(reduced_tasks, curr_state, x_1, tau_1, d, th_imgs)
            # obs_2 = self.__extract_observation_sequence(reduced_tasks, curr_state, x_2, tau_2, d, th_imgs)
            
            valid_0 : bool = self.is_observation_path_valid(state, specs, obs_0)
            valid_1 : bool = self.is_observation_path_valid(state, specs, obs_1)
            # valid_2 : bool = self.is_observation_path_valid(state, specs, obs_2)
            y = 1
            
        return observations
    
    def __initialize_observations_schedule(self, 
                                         state : SimulationAgentState, 
                                         specs : object, 
                                         cross_track_fovs : dict, 
                                         max_slew_rate : float, 
                                         max_torque : float,
                                         orbitdata : OrbitData, 
                                         schedulable_tasks : list, 
                                         mission : Mission, 
                                         observation_history : ObservationHistory
                                         ) -> tuple:
        """ Uses a simplified MILP formulation to generate an initial solution for the full MILP optimization """

        # Check if there are no tasks to schedule
        if not schedulable_tasks: return None, [], [], [], np.NAN

        # Add dummy task to represent initial state
        dummy_task = SpecificObservationTask(set([]),
                                             schedulable_tasks[0].instrument_name,
                                             Interval(state.t,state.t), 
                                             0.0, 
                                             Interval(state.attitude[0],state.attitude[0]))
        tasks : list[SpecificObservationTask] = [dummy_task]
        tasks.extend(schedulable_tasks)

        # Create a new model
        model = gp.Model("single-sat_milp_planner")

        # Set parameter to suppress output
        # model.setParam('OutputFlag', int(self._debug))

        # List tasks by their index
        indexed_tasks = list(enumerate(tasks))
        task_indices = [j for j,_ in indexed_tasks]

        # Define constants
        rewards = np.array([self.estimate_task_value(task, 
                                                     task.accessibility.left, 
                                                     task.min_duration, 
                                                     specs, cross_track_fovs, orbitdata, 
                                                     mission, 
                                                     observation_history)
                            for task in tqdm(tasks,leave=False,desc='SATELLITE: Calculating task rewards')
                            if isinstance(task,SpecificObservationTask)])
        t_start   = np.array([task.accessibility.left-state.t for task in tasks if isinstance(task, SpecificObservationTask)])
        d         = np.array([task.min_duration for task in tasks if isinstance(task, SpecificObservationTask)])
        th_imgs   = np.array([np.average((task.slew_angles.left, task.slew_angles.right)) for task in tasks if isinstance(task, SpecificObservationTask)])
        slew_time = np.array([[abs(th_imgs[j_p]-th_imgs[j]) / max_slew_rate 
                               for j in task_indices]
                               for j_p in task_indices
                             ])        
        Z = [(j, j_p) 
             for j      in task_indices 
             for j_p    in task_indices
             if t_start[j] + d[j] + slew_time[j, j_p] <= t_start[j_p]   # sequence j->j' is feasible
             and j != j_p                                               # ensure distinct tasks
             ] 
        
        E = [(j,j_p) 
             for j,task_j       in indexed_tasks 
             for j_p,task_j_p   in indexed_tasks
             if ((task_j.is_mutually_exclusive(task_j_p) # mutual exclusivity constraint
                    or (
                        t_start[j] + d[j] + slew_time[j, j_p] > t_start[j_p]    # sequence j->j' is not feasible
                    and  t_start[j_p] + d[j_p] + slew_time[j_p, j] > t_start[j] # sequence j'->j is not feasible
                    ))                                      
                and j < j_p)                               # ensure distinct tasks
             ]
        
        # Create decision variables
        x : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.BINARY, name="x")

        # Validate constants and decision variables to ensure convergence
        assert all([reward >= 0 for reward in rewards])
        assert all([0 <= t_start[j] <= self.horizon for j in task_indices])
        assert all([slew_time[j][j_p] >= 0 for j in task_indices for j_p in task_indices])
        assert all([j in x.keys() and j_p in x.keys() for j,j_p in Z])

        # Set objective
        if self.objective == "reward": 
            model.setObjective(gp.quicksum( rewards[j] * x[j] for j in task_indices), gp.GRB.MAXIMIZE)
                
        # Add constraints
        ## Always assign first observation
        model.addConstr(x[0] == 1)

        ## Enforce exclusivity
        for j,j_p in tqdm(E, desc=f"{state.agent_name}/PREPLANNER: Adding exclusivity constraints", unit='task pairs', leave=False):
            model.addConstr(x[j] + x[j_p] <= 1)

        # Optimize model
        model.optimize()

        # Print results
        self.__print_model_results(model)

        # Convert decision variables to arrays
        x_array = np.array([int(x[j].X) for j,_ in indexed_tasks if j > 0])
        z_array = np.array([(j,j_p, int(x[j].X * x[j_p].X)) for j,j_p in Z if j > 0 and j_p > 0])
        tau_array = np.array([t_start[j]+state.t for j,_ in indexed_tasks if j > 0])

        # Return solved model
        return model, x_array, z_array, tau_array, model.getObjective().getValue()
    
    def __optimize_observations_schedule_temp(self,
                                     state : SimulationAgentState,
                                     specs : object,
                                     cross_track_fovs : dict,
                                     max_slew_rate : float,
                                     max_torque : float,
                                     orbitdata : OrbitData,
                                     schedulable_tasks : list,
                                     mission : Mission,
                                     observation_history : ObservationHistory,
                                     x_init : np.ndarray = None
                                     ) -> tuple:
        """
        Refactored MILP: sparse arcs, pruning, flow constraints, indicators, clique exclusivity.
        Returns: model, x_array, z_array, tau_array, objective_value
        x_array / z_array / tau_array correspond to tasks excluding the dummy (i.e., original schedulable_tasks order).
        """

        # --- quick guards
        if not schedulable_tasks:
            return None, np.array([]), np.array([]), np.array([]), np.NAN

        # --- build tasks list with dummy task at index 0
        dummy_task = SpecificObservationTask(set([]),
                                            schedulable_tasks[0].instrument_name,
                                            Interval(state.t, state.t),
                                            0.0,
                                            Interval(state.attitude[0], state.attitude[0]))
        tasks = [dummy_task] + list(schedulable_tasks)  # index 0 = dummy

        n_total = len(tasks)
        orig_indices = list(range(n_total))  # mapping for later (identity now)

        # --- Build arrays aligned with tasks indices (0..n_total-1)
        rewards = np.array([self.estimate_task_value(task,
                                                    task.accessibility.left,
                                                    task.min_duration,
                                                    specs, cross_track_fovs, orbitdata,
                                                    mission, observation_history)
                            for task in tasks], dtype=float)

        # times relative to state.t
        t_start = np.array([task.accessibility.left - state.t for task in tasks], dtype=float)
        t_end   = np.array([task.accessibility.right - state.t for task in tasks], dtype=float)
        d       = np.array([task.min_duration for task in tasks], dtype=float)
        th_imgs = np.array([0.5 * (task.slew_angles.left + task.slew_angles.right) for task in tasks], dtype=float)

        # slew_time[i,j] = time required to slew from i to j
        # careful with division by zero max_slew_rate
        if max_slew_rate is None or max_slew_rate <= 0:
            raise ValueError("max_slew_rate must be > 0")
        slew_time = np.abs(th_imgs.reshape(-1, 1) - th_imgs.reshape(1, -1)) / float(max_slew_rate)

        # --- Candidate forward arcs Z (sparse): only j->j' where it is possible in time and j != j'
        # enforce forward time ordering to reduce symmetry: require t_start[j] <= t_start[j_p]
        Z = []
        for j in range(n_total):
            for jp in range(n_total):
                if j == jp: 
                    continue
                # require finishing earliest at j plus slew can be <= latest feasible start of jp
                # i.e., there exists tau_j in [t_start[j], t_end[j]-d[j]] and tau_jp in [t_start[jp], t_end[jp]-d[jp]]
                # A sufficient fast check: earliest_finish_j + slew <= latest_start_jp
                earliest_finish_j = t_start[j] + d[j]
                latest_start_jp   = t_end[jp] - d[jp]
                if earliest_finish_j + slew_time[j, jp] <= latest_start_jp and t_start[j] <= t_start[jp]:
                    Z.append((j, jp))

        # --- Build exclusivity pair list (unordered pairs j<jp) E_pairs
        exclusivity_matrix = [[tasks[i].is_mutually_exclusive(tasks[jp]) for jp in range(n_total)] for i in range(n_total)]
        E_pairs = []
        for i in range(n_total):
            for j in range(i+1, n_total):
                if exclusivity_matrix[i][j]:
                    E_pairs.append((i, j))

        # --- Build directed adjacency (succs/preds) from Z
        succs = defaultdict(list)
        preds = defaultdict(list)
        for i, j in Z:
            succs[i].append(j)
            preds[j].append(i)

        # --- Reachability prune: nodes not reachable from dummy(0) cannot be scheduled
        reachable = set([0])
        dq = deque([0])
        while dq:
            u = dq.popleft()
            for v in succs.get(u, []):
                if v not in reachable:
                    reachable.add(v)
                    dq.append(v)

        # If some tasks (excluding dummy) unreachable, prune them from consideration.
        # We'll build a keep_mask and remap indices to a compact new indexing.
        keep_mask = [i in reachable for i in range(n_total)]
        # Always keep dummy
        keep_mask[0] = True

        # If nothing except dummy reachable, return trivial plan (no tasks)
        if sum(keep_mask) <= 1:
            # no feasible tasks
            return None, np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float), 0.0

        # Remap old->new indices
        old_to_new = {}
        new_to_old = []
        for old_idx, keep in enumerate(keep_mask):
            if keep:
                old_to_new[old_idx] = len(new_to_old)
                new_to_old.append(old_idx)
        n = len(new_to_old)  # new number of tasks after pruning

        # Build compact arrays for the kept tasks
        tasks_comp = [tasks[old] for old in new_to_old]
        rewards_comp = np.array([rewards[old] for old in new_to_old], dtype=float)
        t_start_comp = np.array([t_start[old] for old in new_to_old], dtype=float)
        t_end_comp   = np.array([t_end[old] for old in new_to_old], dtype=float)
        d_comp       = np.array([d[old] for old in new_to_old], dtype=float)
        th_comp      = np.array([th_imgs[old] for old in new_to_old], dtype=float)

        # recompute slew_time_comp and Z_comp, preds/succs_comp
        slew_time_comp = np.abs(th_comp.reshape(-1,1) - th_comp.reshape(1,-1)) / float(max_slew_rate)
        Z_comp = []
        for i_new, i_old in enumerate(new_to_old):
            for j_new, j_old in enumerate(new_to_old):
                if i_new == j_new:
                    continue
                earliest_finish_i = t_start_comp[i_new] + d_comp[i_new]
                latest_start_j = t_end_comp[j_new] - d_comp[j_new]
                if earliest_finish_i + slew_time_comp[i_new, j_new] <= latest_start_j and t_start_comp[i_new] <= t_start_comp[j_new]:
                    Z_comp.append((i_new, j_new))

        succs_comp = defaultdict(list)
        preds_comp = defaultdict(list)
        for i, j in Z_comp:
            succs_comp[i].append(j)
            preds_comp[j].append(i)

        # exclusivity pairs mapped to compact indices (unordered i<j)
        E_pairs_comp = []
        for (i_old, j_old) in E_pairs:
            if i_old in old_to_new and j_old in old_to_new:
                i_new, j_new = old_to_new[i_old], old_to_new[j_old]
                if i_new < j_new:
                    E_pairs_comp.append((i_new, j_new))
                else:
                    E_pairs_comp.append((j_new, i_new))

        # --- Build clique cover for exclusivity (greedy) to reduce constraints
        # We build cliques such that inside a clique all pairs are mutually exclusive.
        # Simple greedy: iterate nodes, add node to first clique where it's exclusive with all members; else start new clique.
        exclusivity_sets = defaultdict(set)
        for a,b in E_pairs_comp:
            exclusivity_sets[a].add(b)
            exclusivity_sets[b].add(a)

        cliques : list[list[int]] = []
        for node in range(n):
            placed = False
            for clique in cliques:
                # check if node is exclusive to all members in clique (i.e., node cannot coexist with members)
                if all((member in exclusivity_sets[node]) for member in clique):
                    clique.append(node)
                    placed = True
                    break
            if not placed and node in exclusivity_sets and exclusivity_sets[node]:
                cliques.append([node])

        # Note: cliques currently are small; to be safe, also add single-edge exclusivity if still unmatched
        # We'll keep E_pairs_comp too for completeness if cliques empty

        # --- Build Gurobi model (compact)
        model = gp.Model("single-sat_milp_planner_compact")
        # model.setParam('OutputFlag', int(self._debug))

        idxs = list(range(n))  # compact indices 0..n-1
        x = model.addVars(idxs, vtype=gp.GRB.BINARY, name="x")
        z = model.addVars(Z_comp, vtype=gp.GRB.BINARY, name="z")
        tau = model.addVars(idxs, vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=float(self.horizon), name="tau")

        # --- Warm start mapping if provided (x_init assumed for original schedulable_tasks; we must map)
        if x_init is not None:
            # x_init is expected to have length len(schedulable_tasks) (no dummy)
            # Construct original length vector including dummy at start:
            x_init_full = np.insert(np.asarray(x_init, dtype=float), 0, 1.0)
            # Map to compact indices
            x_init_comp = np.array([x_init_full[old_idx] for old_idx in new_to_old], dtype=float)
        else:
            x_init_comp = None

        # objective
        model.setObjective(gp.quicksum(rewards_comp[j] * x[j] for j in idxs), gp.GRB.MAXIMIZE)

        # Always assign dummy start
        model.addConstr(x[0] == 1, name="assign_dummy")
        model.addConstr(tau[0] == 0.0, name="tau_dummy_zero")

        # tau bounds and tie tau when x==0
        for j in idxs:
            lb = float(t_start_comp[j])
            ub = float(t_end_comp[j] - d_comp[j])
            if ub < lb:
                # this task cannot be scheduled; force off
                model.addConstr(x[j] == 0, name=f"unsched_{j}")
                model.addConstr(tau[j] == lb, name=f"tau_fix_{j}")
                continue
            tau[j].LB = lb
            tau[j].UB = ub
            # if x[j] == 0, force tau to lb to avoid free floats
            model.addGenConstrIndicator(x[j], 0, tau[j] == lb)

        # flow / degree constraints
        # For source (dummy index 0), require exactly one outgoing arc from source (one path start)
        source_outgoing = [ (i,j) for (i,j) in Z_comp if i==0 ]
        if source_outgoing:
            model.addConstr(gp.quicksum(z[i,j] for (i,j) in source_outgoing) == 1, name="source_out_deg")
        else:
            # No outgoing from source -> nothing schedulable
            for j in idxs:
                if j > 0: model.addConstr(x[j] == 0)
        # For all non-source nodes, incoming == x and outgoing == x
        for j in idxs:
            if j == 0:
                continue
            incoming = [ (i,j) for i in preds_comp.get(j, []) ]
            outgoing = [ (j,k) for k in succs_comp.get(j, []) ]
            # equality constraints: sum incoming == x[j] and sum outgoing == x[j]
            model.addConstr(gp.quicksum(z[i,j] for (i,j) in incoming) == x[j], name=f"in_deg_{j}")
            model.addConstr(gp.quicksum(z[j,k] for (j,k) in outgoing) == x[j], name=f"out_deg_{j}")

        # indicator arc timing constraints and linking z->x (redundant with degrees, but keep for clarity)
        for (i, j) in Z_comp:
            model.addGenConstrIndicator(z[i,j], 1, tau[i] + d_comp[i] + float(slew_time_comp[i,j]) <= tau[j], name=f"prec_{i}_{j}")
            # linking (redundant if degrees used): z <= x[i], z <= x[j]
            model.addConstr(z[i,j] <= x[i], name=f"z_le_x_i_{i}_{j}")
            model.addConstr(z[i,j] <= x[j], name=f"z_le_x_j_{i}_{j}")

        # exclusivity via clique constraints (if cliques found) else fallback to pairwise
        if cliques:
            for idx, clique in enumerate(cliques):
                if len(clique) > 1:
                    model.addConstr(gp.quicksum(x[j] for j in clique) <= 1, name=f"excl_clique_{idx}")
            # also include any remaining pairwise exclusivity not covered by cliques
            covered = set()
            for c in cliques:
                for v in c:
                    covered.add(v)
            # fallback add pairwise for any exclusivity not covered
            for (i,j) in E_pairs_comp:
                if not (i in covered and j in covered):
                    model.addConstr(x[i] + x[j] <= 1, name=f"excl_pair_{i}_{j}")
        else:
            # no cliques computed; just add pairwise constraints
            for (i,j) in E_pairs_comp:
                model.addConstr(x[i] + x[j] <= 1, name=f"excl_pair_{i}_{j}")

        # set warm start if available (assign starts only for compact indices)
        if x_init_comp is not None:
            for j in idxs:
                x[j].Start = float(x_init_comp[j])
                tau[j].Start = float(max(tau[j].LB, min(tau[j].UB, t_start_comp[j])))  # earliest feasible
            # z start: for each arc, if both endpoints selected in start, set order by tau
            for (i,j) in Z_comp:
                if x_init_comp[i] > 0.5 and x_init_comp[j] > 0.5:
                    # choose ordering consistent with initial tau assumptions
                    z[i,j].Start = 1 if (t_start_comp[i] + d_comp[i] + slew_time_comp[i,j] <= t_start_comp[j]) else 0

        # optimize
        model.optimize()

        # Post-solve handling
        self.__print_model_results(model)

        # Extract solution mapped back to original task indices (exclude dummy)
        # x_comp is for compact indices; we want boolean per original schedulable_tasks (without dummy)
        x_comp_vals = np.array([int(round(x[j].X)) for j in idxs])
        tau_comp_vals = np.array([float(tau[j].X) for j in idxs])
        z_comp_list = [(i, j, int(round(z[i,j].X))) for (i,j) in Z_comp]

        # Map back to original tasks excluding dummy: new_to_old[1:] corresponds to original tasks indices
        # Build arrays for original-schedulable_tasks order (exclude dummy)
        x_array = np.array([x_comp_vals[old_to_new[old_idx]] for old_idx in range(1, n_total) if old_idx in old_to_new], dtype=int)
        tau_array = np.array([tau_comp_vals[old_to_new[old_idx]] + state.t for old_idx in range(1, n_total) if old_idx in old_to_new], dtype=float)
        # build z array as list of tuples for scheduled arcs where both endpoints are >0 original (exclude dummy arcs)
        z_array = []
        for (i_new, j_new, val) in z_comp_list:
            if val == 1:
                old_i = new_to_old[i_new]
                old_j = new_to_old[j_new]
                # only include arcs between real tasks (exclude arcs to/from dummy if desired)
                if old_i > 0 and old_j > 0:
                    # map to original-task indices (1..)
                    z_array.append((old_i-1, old_j-1, val))
        obj_val = model.ObjVal if model.SolCount > 0 else float('nan')

        return model, x_array, np.array(z_array), tau_array, obj_val


    def __optimize_observations_schedule(self, 
                                         state : SimulationAgentState, 
                                         specs : object, 
                                         cross_track_fovs : dict, 
                                         max_slew_rate : float, 
                                         max_torque : float,
                                         orbitdata : OrbitData, 
                                         schedulable_tasks : list, 
                                         mission : Mission, 
                                         observation_history : ObservationHistory,
                                         x_init : np.ndarray = None
                                         ) -> tuple:
        # Check if there are no tasks to schedule
        if not schedulable_tasks: return None, [], [], [], np.NAN

        # Add dummy task to represent initial state
        dummy_task = SpecificObservationTask(set([]),
                                             schedulable_tasks[0].instrument_name,
                                             Interval(state.t,state.t), 
                                             0.0, 
                                             Interval(state.attitude[0],state.attitude[0]))
        tasks : list[SpecificObservationTask] = [dummy_task]
        tasks.extend(schedulable_tasks)

        # Create a new model
        model = gp.Model("single-sat_milp_planner")

        # Set parameter to suppress output
        # model.setParam('OutputFlag', int(self._debug))

        # List tasks by their index
        indexed_tasks = list(enumerate(tasks))
        task_indices = [j for j,_ in indexed_tasks]

        # Define constants
        rewards = np.array([self.estimate_task_value(task, 
                                                     task.accessibility.left, 
                                                     task.min_duration, 
                                                     specs, cross_track_fovs, orbitdata, 
                                                     mission, 
                                                     observation_history)
                            for task in tqdm(tasks,leave=False,desc='SATELLITE: Calculating task rewards')
                            if isinstance(task,SpecificObservationTask)])
        t_start   = np.array([task.accessibility.left-state.t for task in tasks if isinstance(task, SpecificObservationTask)])
        t_end     = np.array([task.accessibility.right-state.t for task in tasks if isinstance(task, SpecificObservationTask)])
        d         = np.array([task.min_duration for task in tasks if isinstance(task, SpecificObservationTask)])
        th_imgs   = np.array([np.average((task.slew_angles.left, task.slew_angles.right)) for task in tasks if isinstance(task, SpecificObservationTask)])
        slew_time = np.array([[abs(th_imgs[j_p]-th_imgs[j]) / max_slew_rate 
                               for j in task_indices]
                               for j_p in task_indices
                             ])        
        Z = [(j, j_p) 
             for j      in task_indices 
             for j_p    in task_indices
             if t_start[j] + d[j] + slew_time[j, j_p] <= t_end[j_p] - d[j_p] # sequence j->j' is feasible
             and j != j_p                                                    # ensure distinct tasks
             ] 
        
        E = [(j,j_p) 
             for j,task_j       in indexed_tasks 
             for j_p,task_j_p   in indexed_tasks
             if (task_j.is_mutually_exclusive(task_j_p) # mutual exclusivity constraint
                or (
                    t_start[j] + d[j] + slew_time[j, j_p] > t_end[j_p] - d[j_p]
                and  t_start[j_p] + d[j_p] + slew_time[j_p, j] > t_end[j] - d[j]
                ))                                      # sequence j->j' and j'->j are not feasible
                and j < j_p                             # ensure distinct tasks and non-repeating pairs
             ]
        
        # Create decision variables
        x : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.BINARY, name="x")
        z : gp.tupledict = model.addVars(Z, vtype=gp.GRB.BINARY, name="z")
        tau : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.CONTINUOUS, name="tau")

        # Validate constants and decision variables to ensure convergence
        assert all([reward >= 0 for reward in rewards])
        assert all([0 <= t_start[j] <= self.horizon for j in task_indices])
        assert all([0 <= t_end[j] <= self.horizon for j in task_indices])
        assert all([t_end[j] - t_start[j] >= d[j] for j in task_indices])
        assert all([t_end[j] - d[j] >= 0.0 for j in task_indices])
        assert all([slew_time[j][j_p] >= 0 for j in task_indices for j_p in task_indices])
        assert all([j in x.keys() and j_p in x.keys() for j,j_p in Z])

        # if initial solution is provided, use it
        if x_init is not None:
            x_init = np.insert(x_init, 0, 1) # compensate for missing initial task
            assert len(x_init) == len(x), "Initial solution length does not match number of tasks."

            for j in task_indices:
                x[j].Start = x_init[j]
                tau[j].Start = float(t_start[j])

            for j,j_p in Z:
                if x_init[j] > 0.5 and x_init[j_p] > 0.5 and j != j_p:
                    z[j, j_p].Start = int(t_start[j] + d[j] + slew_time[j, j_p] <= t_start[j_p])

        # Set objective
        if self.objective == "reward": 
            model.setObjective(gp.quicksum( rewards[j] * x[j] for j in task_indices), gp.GRB.MAXIMIZE)
                
        # # Precalculate preemptive pruning
        # # --- Build directed adjacency (succs/preds) from Z
        # succs = defaultdict(list)
        # preds = defaultdict(list)
        # for i, j in Z:
        #     succs[i].append(j)
        #     preds[j].append(i)

        # # --- Reachability prune: nodes not reachable from dummy(0) cannot be scheduled
        # reachable = set([0])
        # dq = deque([0])
        # while dq:
        #     u = dq.popleft()
        #     for v in succs.get(u, []):
        #         if v not in reachable:
        #             reachable.add(v)
        #             dq.append(v)

        # # If some tasks (excluding dummy) unreachable, prune them from consideration.
        # # We'll build a keep_mask and remap indices to a compact new indexing.
        # keep_mask = [j in reachable for j in task_indices]
        # discard_mask = [j not in reachable for j in task_indices]
        # # Always keep dummy
        # keep_mask[0] = True

        # # If nothing except dummy reachable, return trivial plan (no tasks)
        # if sum(keep_mask) <= 1:
        #     # no feasible tasks
        #     return None, np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float), 0.0

        # Add constraints
        ## Always assign first observation
        model.addConstr(x[0] == 1)
        model.addConstr(tau[0] == 0.0)

        # ## Avoid unreachable tasks
        # for j,discarded in tqdm(enumerate(discard_mask), desc=f"{state.agent_name}/PREPLANNER: Adding exclusivity constraints", unit='task pairs', leave=False):
        #     if discarded: model.addConstr(x[j] == 0)

        ## Enforce exclusivity
        for j,j_p in tqdm(E, desc=f"{state.agent_name}/PREPLANNER: Adding exclusivity constraints", unit='task pairs', leave=False):
            model.addConstr(x[j] + x[j_p] <= 1)

        ## Observation time and accessibility constraints
        for j in tqdm(x.keys(), desc=f"{state.agent_name}/PREPLANNER: Adding observation time constraints", unit='tasks', leave=False):
            tau[j].LB = t_start[j]
            tau[j].UB = t_end[j] - d[j]
            # If x[j] == 0, force tau to LB (or any fixed value in the feasible interval)
            model.addGenConstrIndicator(x[j], 0, tau[j] == float(t_start[j]))
           
        ## Slew time constraints for observation task sequence j->j'
        pairs_considered : set = set()
        for j,j_p in tqdm(z.keys(), desc=f"{state.agent_name}/PREPLANNER: Adding slewing constraints", unit='task pairs', leave=False):
           
            # If z[j, j_p] == 1, enforce slew constraint for task sequence j->j'
            model.addGenConstrIndicator(z[j,j_p], 1, tau[j] + d[j] + slew_time[j,j_p] <= tau[j_p])  

            # enforce lower bound for z[j, j_p]
            if (j_p, j) in z.keys(): # j->j' and j'->j sequences are both valid
                if (min(j, j_p),max(j, j_p)) not in pairs_considered:
                    # at least one of z[j, j_p] or z[j_p, j] must be 1 if x[j] and x[j_p] are assigned
                    model.addConstr(x[j] + x[j_p] - 1 <= z[j, j_p] + z[j_p, j])
                    
                    # enforce mutual exclusivity if (j, j_p) and (j_p, j) are both in z
                    model.addConstr(z[j, j_p] + z[j_p, j] <= 1) 

                    # add pair to considered pairs
                    pairs_considered.add((min(j, j_p),max(j, j_p)))

            else: # only j->j' sequence is valid
                # x[j] and x[j_p] are mutually exclusive unless z[j,j_p] is assigned
                model.addConstr(x[j] + x[j_p] <= 1 + z[j, j_p])                

            # # enforce upper bound for z[j, j_p]; x[j] and x[j_p] must be assigned if z[j, j_p] is to be assigned
            # model.addConstr(z[j, j_p] <= x[j])
            # model.addConstr(z[j, j_p] <= x[j_p])

        # Optimize model
        model.optimize()

        # Print results
        self.__print_model_results(model)

        # Convert decision variables to arrays
        x_array = np.array([int(x[j].X) for j,_ in indexed_tasks if j > 0])
        z_array = np.array([(j,j_p,int(z[j,j_p].X)) for j,j_p in Z if j > 0 and j_p > 0])
        tau_array = np.array([float(tau[j].X)+state.t for j,_ in indexed_tasks if j > 0])

        # Return solved model
        return model, x_array, z_array, tau_array, model.getObjective().getValue()

    def __print_model_results(self, model : gp.Model) -> None:
        """ Prints model results to console and outputs error logs in case of infeasibility """

        # Print model status
        print("\nStatus code:", model.Status)

        # Check model status
        if model.Status == gp.GRB.OPTIMAL:
            print("Optimal solution found.")
            print(f"Obj: {model.ObjVal:g}")

            # delete any `.ilp` files that may have been generated when debugging the model
            class_path = inspect.getfile(type(self))
            class_path = class_path.replace('milp.py', '')
            for filename in os.listdir(class_path):
                if ".ilp" in filename:
                    os.remove(os.path.join(class_path, filename))
        else:
            if model.Status == gp.GRB.INFEASIBLE:
                print("Model is infeasible.")
            
                # diagnose problems with model
                print("Computing IIS...")
                model.computeIIS()

                # print results to file
                filepath = f"./chess3d/agents/planning/preplanners/decentralized/{model.ModelName}.ilp"
                model.write(filepath)
                model.write(f"{filepath}.json")
                print(f"IIS written to `{filepath}`")

            if model.Status == gp.GRB.UNBOUNDED:
                print("Model is unbounded.")

            if model.Status == gp.GRB.INTERRUPTED:
                print("Model was interrupted.")

            raise ValueError(f"Unexpected model status: {model.Status}")
        
        return

    def __extract_observation_sequence(self, schedulable_tasks : list, state : SimulationAgentState, x : gp.tupledict, tau : gp.tupledict, d : np.array, th_imgs : np.array) -> list:
        """ Generates a list of observations from the outputs of MILP solver"""

        # Convert model outputs into observation actions
        observations = [ObservationAction(task.instrument_name,
                              th_imgs[j],
                              tau[j],
                              d[j],
                              task
                              )
                        for j,task in enumerate(schedulable_tasks) 
                        if x[j] > 0.5            # select only assigned tasks
                        ]

        # Return sorted observations in start time
        return sorted(observations, key=lambda obs: obs.t_start)

    def _schedule_broadcasts(self, state, observations, orbitdata, t = None):
        # TODO 
        return super()._schedule_broadcasts(state, observations, orbitdata, t)