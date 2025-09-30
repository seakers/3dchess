import inspect
import logging
import os
from typing import Dict, Tuple

from dmas.utils import runtime_tracker

from pyparsing import List
import gurobipy as gp
import numpy as np
from tqdm import tqdm

from chess3d.agents.actions import ObservationAction
from chess3d.agents.planning import tasks
from chess3d.agents.planning.preplanners.centralized.dealer import DealerPreplanner
from chess3d.agents.planning.tasks import GenericObservationTask, SpecificObservationTask
from chess3d.agents.planning.tracker import ObservationHistory
from chess3d.agents.states import SimulationAgentState
from chess3d.mission.mission import Mission
from chess3d.orbitdata import OrbitData
from chess3d.utils import Interval


class DealerMILPPreplanner(DealerPreplanner):
    """
    A preplanner that generates plans for other agents using MILP models.

    ### Models Available
    | Model Name  | Time Assignment          | Reward Function                       | Reobs / Revisit Handling        | Notes                                      |
    |-------------|--------------------------|---------------------------------------|---------------------------------|--------------------------------------------|
    | Static      | Fixed at start of window | Static, precomputed per task          | None                            | Simplest, fastest; ignores dynamics        |
    | Linear      | Any time in window       | Linear variation across access window | None                            | Captures time-dependence, no reobs         |
    | Reobs       | Any time in window       | Depends on # of reobservations        | Reobs only                      | Tracks constellation-wide redundancy       |
    | Revisit     | Any time in window       | Depends on reobs + revisit intervals  | Reobs + Revisit timing          | Richest model; most realistic, most costly |

    """
    STATIC = 'static'
    LINEAR = 'linear'
    REOBS = 'reobs'
    REVISIT = 'revisit'
    MODELS = [STATIC, LINEAR, REOBS, REVISIT]

    def __init__(self, 
                 client_orbitdata : Dict[str, OrbitData],
                 client_specs : Dict[str, object],
                 client_missions : Dict[str, Mission],
                 model : str, 
                 licence_path : str = None, 
                 horizon : float = np.Inf,
                 period : float = np.Inf,
                 max_tasks : float = np.Inf,
                 max_observations : int = 10, 
                 debug : bool = False,
                 logger : logging.Logger = None):
        super().__init__(client_orbitdata, client_specs, client_missions, horizon, period, debug, logger)

        if not debug or licence_path is not None:
            # Check for Gurobi license
            assert os.path.isfile(licence_path), f"Provided Gurobi licence path `{licence_path}` is not a valid file."

            # Set Gurobi license environment variable
            os.environ['GRB_LICENSE_FILE'] = licence_path

        # Validate inputs
        assert model in self.MODELS, f"Model must be one of {self.MODELS}."
        assert (isinstance(max_tasks, int) and max_tasks > 0) or max_tasks == np.Inf, "Max tasks must be a positive integer."

        # Set attributes
        self.model = model
        self.max_tasks = max_tasks
        self.max_observations = max_observations

    @runtime_tracker
    def _schedule_client_observations(self, 
                                      state : SimulationAgentState,
                                      available_tasks : Dict[Mission, List[GenericObservationTask]],
                                      schedulable_tasks: Dict[str, List[SpecificObservationTask]], 
                                      observation_history : ObservationHistory
                                    ) -> Dict[str, List[ObservationAction]]:
        """ schedules observations for all clients """
        
        # short-circuit if no tasks to schedule
        if all([len(tasks) == 0 for tasks in schedulable_tasks.values()]) or len(available_tasks) == 0:
                return {client: [] for client in self.client_orbitdata} # no tasks to schedule
        
        # collect max slew rates for each client
        max_slew_rates : Dict[str, float] = {client : self._collect_agility_specs(specs)[0] 
                                             for client,specs in self.client_specs.items()}
        max_torques : Dict[str, float] = {client : self._collect_agility_specs(specs)[1] 
                                             for client,specs in self.client_specs.items()}

        # add dummy tasks to each client's task list
        for client,client_tasks in schedulable_tasks.items():
            # Check if there are no tasks to schedule
            if not client_tasks: raise NotImplementedError("Client observation scheduling not yet implemented.")

            # Add dummy task to represent initial state of each client
            dummy_task = SpecificObservationTask(set(),                                             # empty set of parent tasks
                                                client_tasks[0].instrument_name, 
                                                Interval(self.client_states[client].t,
                                                         self.client_states[client].t),             # set accessibility to current time only
                                                0.0,                                                # zero duration
                                                Interval(self.client_states[client].attitude[0],
                                                         self.client_states[client].attitude[0]))   # set slew angle to current angle
            client_tasks.insert(0, dummy_task)      

        # collect list of unique instruments across all clients
        instruments : list[str] = list({instrument.name #.lower() 
                                        for specs in self.client_specs.values() 
                                        for instrument in specs.instrument})

        # collect list of unique parent tasks across all missions
        parent_tasks : list[GenericObservationTask] = list({ptask for mission_ptasks in available_tasks.values() for ptask in mission_ptasks})
        
        # count max number of reobservations per parent task and instrument
        K : list[list[int]] = [len([task 
                                        for client_tasks in schedulable_tasks.values() 
                                        for task in client_tasks 
                                        if ptask in task.parent_tasks 
                                    ])
                                for ptask in parent_tasks]

        # index task and client info 
        parent_task_indices : list[int] = [task_idx for task_idx,_ in enumerate(parent_tasks)]

        reobservation_indices : list[tuple[int,int]] = [(task_idx,obs_idx) 
                                                        for task_idx,_ in enumerate(parent_tasks)
                                                        for obs_idx in range(1,K[task_idx]+1)]

        indexed_clients : list[str] = list(schedulable_tasks.keys())
        client_indices : Dict[str, int] = {client : idx 
                                           for idx,client in enumerate(indexed_clients)}

        indexed_missions : list[Mission] = list(available_tasks.keys())
        mission_indices : Dict[Mission, int] = {mission : idx 
                                                for idx,mission in enumerate(indexed_missions)}

        task_indices : list[tuple[int,int]] = [(client_idx,task_idx) 
                                                for client_idx,client in enumerate(indexed_clients)
                                                for task_idx,_ in enumerate(schedulable_tasks[client])]
        
        # Initialize constants
        t_start   = [[task.accessibility.left-state.t 
                              for task in schedulable_tasks[client]  if isinstance(task, SpecificObservationTask)]
                              for client in indexed_clients]
        t_end     = [[task.accessibility.right-state.t 
                              for task in schedulable_tasks[client]  if isinstance(task, SpecificObservationTask)]
                              for client in indexed_clients]
        d         = [[task.min_duration 
                              for task in schedulable_tasks[client]  if isinstance(task, SpecificObservationTask)]
                              for client in indexed_clients]
        th_imgs   = [[np.average((task.slew_angles.left, task.slew_angles.right)) 
                              for task in schedulable_tasks[client]  if isinstance(task, SpecificObservationTask)]
                              for client in indexed_clients]
        slew_times= [[[abs(th_imgs[client_index][j_p]-th_imgs[client_index][j]) / max_slew_rates[client]
                                for j_p,task_j_p in enumerate(schedulable_tasks[client]) if isinstance(task_j_p, SpecificObservationTask)]
                                for j,task_j in enumerate(schedulable_tasks[client]) if isinstance(task_j, SpecificObservationTask)]
                                for client_index, client in enumerate(indexed_clients)]
        
        # Map sparse arch matrix of feasible task sequences
        A : list = [(s,j,j_p)
                        for s,client in enumerate(indexed_clients)
                        for j,task_j in enumerate(schedulable_tasks[client]) if isinstance(task_j, SpecificObservationTask)
                        for j_p,task_j_p in enumerate(schedulable_tasks[client]) if isinstance(task_j_p, SpecificObservationTask)
                        if t_start[s][j] + d[s][j] + slew_times[s][j][j_p]
                            <= t_end[s][j_p] - d[s][j_p]    # sequence j->j' is feasible
                            and j != j_p                  # ensure distinct tasks
                    ]
        
        # Map task mutual exclusivity constraints
        E : list = [(s,j,j_p)
                    for s,client in enumerate(indexed_clients)
                    for j,task_j in enumerate(schedulable_tasks[client]) if isinstance(task_j, SpecificObservationTask)
                    for j_p,task_j_p in enumerate(schedulable_tasks[client]) if isinstance(task_j_p, SpecificObservationTask)
                    if (task_j.is_mutually_exclusive(task_j_p) 
                        or (t_start[s][j] + d[s][j] + slew_times[s][j][j_p] > t_end[s][j_p] - d[s][j_p]    # sequence j->j' is not feasible
                            and t_start[s][j_p] + d[s][j_p] + slew_times[s][j_p][j] > t_end[s][j] - d[s][j] # sequence j'->j is not feasible
                        )
                    ) and j < j_p   # avoid duplicates
                ]

        # Run optimization on each chunk and flatten results
        if self.model == self.STATIC:
            y,t = self.__static_milp_planner(state, schedulable_tasks, observation_history, indexed_clients, task_indices, A, E, t_start, d, slew_times)
        # elif self.model == self.LINEAR:
        #     y,t = self.__linear_milp_planner(state, schedulable_tasks, observation_history, indexed_clients, task_indices, A, E, t_start, t_end, d, slew_times)
        # elif self.model == self.REOBS:
        #     y,t = self.__reobs_milp_planner(state, available_tasks, schedulable_tasks, observation_history)
        # elif self.model == self.REVISIT:
        #     y,t = self.__revisit_milp_planner(state, available_tasks, schedulable_tasks, observation_history)
        else: # should never happen due to checks in `__init__`
            raise ValueError(f"Unknown model type `{self.model}`.")

        # remove dummy tasks to each client's task list
        for client,client_tasks in schedulable_tasks.items(): client_tasks.pop(0)

        # Extract and return observation sequence
        observations : Dict[str, list[ObservationAction]] = \
            self.__extract_observation_sequence(state, schedulable_tasks, observation_history, indexed_clients, task_indices, d, th_imgs, y, t)

        return observations

    def __static_milp_planner(self,
                              state : SimulationAgentState,
                              schedulable_tasks: Dict[str, List[ObservationAction]], 
                              observation_history : ObservationHistory,
                              indexed_clients : List[str],
                              task_indices : List[Tuple[int,int]],
                              A : List[Tuple[int,int,int]],
                              E : List[Tuple[int,int,int]],
                              t_start : np.ndarray,
                              d : np.ndarray,
                              slew_times : np.ndarray
                            ) -> Tuple:
        """ Implements a MILP planner with static time assignment and static rewards """

        # Compute Rewards at start of access window
        rewards : np.ndarray = self.__estimate_static_task_rewards(indexed_clients, schedulable_tasks, observation_history)

        # Create a new model
        model = gp.Model(f"multi-mission_milp_planner_{self.model.upper()}")

        # Set parameter to suppress output
        model.setParam('OutputFlag', int(self._debug))

        # Create decision variables
        y : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.BINARY, name="y")
        t : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.CONTINUOUS, name="t")
        z : gp.tupledict = model.addVars(A, vtype=gp.GRB.BINARY, name="z")

        # Set objective
        model.setObjective(gp.quicksum( rewards[s][j] * y[s,j] for s,j in task_indices), gp.GRB.MAXIMIZE)

        # Add constraints
        ## Always assign first observation
        for s,_ in enumerate(indexed_clients): model.addConstr(y[s,0] == 1, name=f"assign_initial_dummy_task_client{s}")

        ## set unreachable tasks to y=0
        for s,j in tqdm(y.keys(), desc=f"{state.agent_name}/PREPLANNER: Avoiding unreachable tasks", unit='tasks', leave=False):
            if j == 0: continue # skip dummy task
            j_p_predecessors = [j_p for a,j_p,b in A if a == s and b == j]
            if not j_p_predecessors: model.addConstr(y[s,j] == 0, name=f"unreachable_task_client{s}_task{j}")

        ## Enforce exclusivity
        for s,j,j_p in tqdm(E, desc=f"{state.agent_name}/PREPLANNER: Adding mutual exclusivity constraints", unit='task pairs', leave=False):
            model.addConstr(y[s,j] + y[s,j_p] <= 1)

        ## Observation time and accessibility constraints; force task scheduling at start of visibility window
        for s,j in tqdm(y.keys(), desc=f"{state.agent_name}/PREPLANNER: Adding observation time constraints", unit='tasks', leave=False):            
            model.addConstr(t[s,j] == t_start[s][j], name=f"static_time_assignment_client{s}_task{j}")

        ## Task sequencing constraints
        # for s,j,j_p in tqdm(z.keys(), desc=f"{state.agent_name}/PREPLANNER: Adding sequencing time constraints", unit='sequences', leave=False):
        #      # If z[s, j, j_p] == 1, enforce slew constraint for task sequence j->j' for client s
        #     model.addGenConstrIndicator(z[s,j,j_p], 1, t[s,j] + d[s,j] + slew_times[s,j,j_p] <= t[s,j_p])  

        # for s,j in tqdm(y.keys(), desc=f"{state.agent_name}/PREPLANNER: Adding task sequencing constraints", unit='tasks', leave=False):
        #     # Ensure that each task can have at most one successor
        #     j_p_successors = [j_p for a,b,j_p in A if a == s and b == j]
        #     model.addConstr(gp.quicksum(z[s,j,j_p] for j_p in j_p_successors) <= y[s,j],
        #                     name=f"max_one_successor_client{s}_task{j}")

        #     # Ensure that each task must have only one predecessor
        #     if j > 0: # dummy task cannot have a predecessor
        #         j_p_predecessors = [j_p for a,j_p,b in A if a == s and b == j]
        #         model.addConstr(gp.quicksum(z[s,j_p,j] for j_p in j_p_predecessors) == y[s,j],
        #                         name=f"max_one_predecessor_client{s}_task{j}")

        ## Slew time constraints for observation task sequence j->j'
        pairs_considered : set = set()
        for s,j,j_p in tqdm(z.keys(), desc=f"{state.agent_name}/PREPLANNER: Adding slewing constraints", unit='task pairs', leave=False):
           # If z[j, j_p] == 1, enforce slew constraint for task sequence j->j'
            model.addGenConstrIndicator(z[s,j,j_p], 1, t[s,j] + d[s][j] + slew_times[s][j][j_p] <= t[s,j_p])  

            # enforce lower bound for z[j, j_p]
            if (s,j_p, j) in z.keys(): # j->j' and j'->j sequences are both valid
                # create unique pair for considered pairs
                unique_pair : tuple = (s,min(j, j_p),max(j, j_p))

                # check if unique pair has been considered already
                if unique_pair not in pairs_considered:
                    # at least one of z[j, j_p] or z[j_p, j] must be 1 if y[s,j] and y[s,j_p] are assigned
                    model.addConstr(y[s,j] + y[s,j_p] - 1 <= z[s,j,j_p] + z[s,j_p,j])
                    
                    # enforce mutual exclusivity if (s,j, j_p) and (s,j_p, j) are both in z
                    model.addConstr(z[s,j,j_p] + z[s,j_p,j] <= 1)

                    # add pair to considered pairs
                    pairs_considered.add(unique_pair)

            else: # only j->j' sequence is valid
                # y[s,j] and y[s,j_p] are mutually exclusive unless z[s,j,j_p] is assigned
                model.addConstr(y[s,j] + y[s,j_p] <= 1 + z[s,j,j_p])   

        # Optimize model
        model.optimize()

        # Print results
        self.__print_model_results(model)
    
        # Convert decision variables to arrays and ignore dummy tasks
        y_array = [[int(y[s,j].X) 
                             for j,_ in enumerate(schedulable_tasks[client]) if j > 0] 
                            for s,client in enumerate(indexed_clients)]
        t_array = [[float(t[s,j].X)+state.t 
                             for j,_ in enumerate(schedulable_tasks[client]) if j > 0] 
                            for s,client in enumerate(indexed_clients)]
        
        # DEBUG
        # TODO: there seem to be issues scheduling a sequence of tasks with a single predecesor and at most a single successor in the sequence
        z_array = {(s,j,j_p): int(z[s,j,j_p].X) for s,j,j_p in z.keys()}

        # Return solved model
        return y_array, t_array

    def __estimate_static_task_rewards(self, 
                                       indexed_clients: List[str], 
                                       schedulable_tasks: Dict[str, List[ObservationAction]], 
                                       observation_history: ObservationHistory) -> list:
        """ Estimate static task rewards for each client and task based on parent tasks """
        
        return [[self.estimate_task_value(task, 
                                                      task.accessibility.left, 
                                                      task.min_duration, 
                                                      self.client_specs[client], 
                                                      self.cross_track_fovs[client], 
                                                      self.client_orbitdata[client], 
                                                      self.client_missions[client], 
                                                     observation_history)
                            for task in tqdm(schedulable_tasks[client],leave=False,desc=f'SATELLITE: Calculating task rewards for client agent `{client}`')
                            if isinstance(task,SpecificObservationTask)]
                            for client in indexed_clients]

    def __linear_milp_planner(self,
                              state : SimulationAgentState,
                              schedulable_tasks: Dict[str, List[ObservationAction]], 
                              observation_history : ObservationHistory,
                              indexed_clients : List[str],
                              task_indices : List[Tuple[int,int]],
                              A : List[Tuple[int,int,int]],
                              E : List[Tuple[int,int,int]],
                              t_start : np.ndarray,
                              t_end : np.ndarray,
                              d : np.ndarray,
                              slew_times : np.ndarray
                            ) -> Tuple:
        """ Implements a MILP planner with linear time assignment and linear rewards """
        
        raise NotImplementedError("Linear MILP planner not yet implemented.")

        # Compute rewards at start and end of access window
        rewards : np.ndarray = self.__estimate_linear_task_rewards(indexed_clients, schedulable_tasks, observation_history)

        # Compute slope of linear reward function during the access windows
        slopes : np.ndarray = np.array([[ (rewards[s,j,1]-rewards[s,j,0]) / (t_end[s,j]-t_start[s,j]) 
                                          if t_end[s,j] > t_start[s,j] else 0.0
                                          for j in range(rewards.shape[1])]
                                         for s in range(rewards.shape[0])])
        
        # Create a new model
        model = gp.Model(f"multi-mission_milp_planner_{self.model.upper()}")

        # Set parameter to suppress output
        model.setParam('OutputFlag', int(self._debug))

        # Create decision variables
        y : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.BINARY, name="y")
        t : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.CONTINUOUS, name="t")
        z : gp.tupledict = model.addVars(A, vtype=gp.GRB.BINARY, name="z")

        # Set objective
        model.setObjective(gp.quicksum( rewards[s,j,0] * y[s,j] + slopes[s,j] * (t[s,j] - t_start[s,j]) for s,j in task_indices), gp.GRB.MAXIMIZE)

        # Add constraints
        ## Always assign first observation
        for s,_ in enumerate(indexed_clients): model.addConstr(y[s,0] == 1, name=f"assign_initial_dummy_task_client{s}")

        ## set unreachable tasks to y=0
        for s,j in tqdm(y.keys(), desc=f"{state.agent_name}/PREPLANNER: Avoiding unreachable tasks", unit='tasks', leave=False):
            if j == 0: continue # skip dummy task
            j_p_predecessors = [j_p for a,j_p,b in A if a == s and b == j]
            if not j_p_predecessors: model.addConstr(y[s,j] == 0, name=f"unreachable_task_client{s}_task{j}")

        ## Enforce exclusivity
        for s,j,j_p in tqdm(E, desc=f"{state.agent_name}/PREPLANNER: Adding mutual exclusivity constraints", unit='task pairs', leave=False):
            model.addConstr(y[s,j] + y[s,j_p] <= 1)

        ## Observation time and accessibility constraints
        for s,j in tqdm(y.keys(), desc=f"{state.agent_name}/PREPLANNER: Adding observation time constraints", unit='tasks', leave=False):
            # Enforce task scheduling within visibility window
            t[s,j].LB = t_start[s,j]
            t[s,j].UB = t_end[s,j] - d[s,j]
            
            # If task is not assigned (y[s,j] == 0), force tau to LB (or any fixed value in the feasible interval)
            model.addGenConstrIndicator(y[s,j], 0, t[s,j] == float(t_start[s,j]))

        ## Task sequencing constraints
        for s,j,j_p in tqdm(z.keys(), desc=f"{state.agent_name}/PREPLANNER: Adding sequencing time constraints", unit='sequences', leave=False):
             # If z[s, j, j_p] == 1, enforce slew constraint for task sequence j->j' for client s
            model.addGenConstrIndicator(z[s,j,j_p], 1, t[s,j] + d[s,j] + slew_times[s,j,j_p] <= t[s,j_p])  

        for s,j in tqdm(y.keys(), desc=f"{state.agent_name}/PREPLANNER: Adding task sequencing constraints", unit='tasks', leave=False):
            # Ensure that each task can have at most one successor
            j_p_successors = [j_p for a,b,j_p in A if a == s and b == j]
            model.addConstr(gp.quicksum(z[s,j,j_p] for j_p in j_p_successors) <= y[s,j],
                            name=f"max_one_successor_client{s}_task{j}")

            # Ensure that each task must have only one predecessor
            if j > 0: # dummy task cannot have a predecessor
                j_p_predecessors = [j_p for a,j_p,b in A if a == s and b == j]
                model.addConstr(gp.quicksum(z[s,j_p,j] for j_p in j_p_predecessors) == y[s,j],
                                name=f"max_one_predecessor_client{s}_task{j}")
        
        # Optimize model
        model.optimize()

        # Print results
        self.__print_model_results(model)
    
        # Convert decision variables to arrays
        y_array = np.array([[int(y[s,j].X) 
                             for j,_ in enumerate(schedulable_tasks[client]) if j > 0] 
                            for s,client in enumerate(indexed_clients)])
        t_array = np.array([[float(t[s,j].X)+state.t 
                             for j,_ in enumerate(schedulable_tasks[client]) if j > 0] 
                            for s,client in enumerate(indexed_clients)])

        # Return solved model
        return y_array, t_array

    def __estimate_linear_task_rewards(self, 
                                       indexed_clients: List[str], 
                                       schedulable_tasks: Dict[str, List[ObservationAction]], 
                                       observation_history: ObservationHistory) -> np.ndarray:
        """ Estimate linear task rewards for each client and task based on parent tasks """

        return np.array([[[self.estimate_task_value(task, 
                                                      task.accessibility.left, 
                                                      task.min_duration, 
                                                      self.client_specs[client], 
                                                      self.cross_track_fovs[client], 
                                                      self.client_orbitdata[client], 
                                                      self.client_missions[client], 
                                                     observation_history),
                            self.estimate_task_value(task, 
                                                      task.accessibility.right, 
                                                      task.min_duration, 
                                                      self.client_specs[client], 
                                                      self.cross_track_fovs[client], 
                                                      self.client_orbitdata[client], 
                                                      self.client_missions[client], 
                                                     observation_history)]
                            for task in tqdm(schedulable_tasks[client],leave=False,desc=f'SATELLITE: Calculating task rewards for client agent `{client}`')
                            if isinstance(task,SpecificObservationTask)]
                            for client in indexed_clients])
    
    def __reobs_milp_planner(self,
                              state : SimulationAgentState,
                              available_tasks : Dict[Mission, List[GenericObservationTask]],
                              schedulable_tasks: Dict[str, List[ObservationAction]], 
                              observation_history : ObservationHistory
                            ) -> Tuple:
        """ Implements a MILP planner with static time assignment and reobservation-dependent rewards """
        
        raise NotImplementedError("MILP preplanner not yet implemented. Working on it...")
    
    def __revisit_milp_planner(self,
                              state : SimulationAgentState,
                              available_tasks : Dict[Mission, List[GenericObservationTask]],
                              schedulable_tasks: Dict[str, List[ObservationAction]], 
                              observation_history : ObservationHistory
                            ) -> Tuple:
        """ Implements a MILP planner with static time assignment and reobservation + revisit-dependent rewards """ 
        raise NotImplementedError("MILP preplanner not yet implemented. Working on it...")


    def __extract_observation_sequence(self,
                              state : SimulationAgentState,
                              schedulable_tasks: Dict[str, List[ObservationAction]], 
                              observation_history : ObservationHistory,
                              indexed_clients : List[str],
                              task_indices : List[Tuple[int,int]],
                              d : np.ndarray,
                              th_imgs : np.ndarray,
                              y : List[List[int]],
                              t : List[List[float]]
                              ) -> Dict[str, List[ObservationAction]]:
        """ Extracts the observation sequence from the solved MILP model """
        return {client : sorted([ObservationAction(task.instrument_name,
                                    th_imgs[s][j],
                                    t[s][j],
                                    d[s][j],
                                    task
                                    ) 
                                 for j,task in enumerate(schedulable_tasks[client]) 
                                 if y[s][j] > 0.5
                            ], key=lambda action: action.t_start)
                        for s,client in enumerate(indexed_clients)}
        
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
            

        # # Create a new model
        # model = gp.Model("multi-mission_milp_planner")

        # # Set parameter to suppress output
        # model.setParam('OutputFlag', int(self._debug))

        
        # # Calculate incremental rewards for each task based on parent tasks and reobservation number 
        # del_rewards : np.ndarray = self.__estimate_task_rewards(available_tasks, schedulable_tasks, parent_tasks, instruments, indexed_clients, K, observation_history)

        # # Create decision variables
        # y : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.BINARY, name="y")
        # tau : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.CONTINUOUS, name="tau")
        # z : gp.tupledict = model.addVars(A, vtype=gp.GRB.BINARY, name="z")
        # n : gp.tupledict = model.addVars(parent_task_indices, vtype=gp.GRB.INTEGER, name="n", ub=self.max_observations)
        # r : gp.tupledict = model.addVars(reobservation_indices, vtype=gp.GRB.BINARY, name="r")

        # # set objective
        # model.setObjective(gp.quicksum(del_rewards[ptask_idx,reobs_idx] * r[ptask_idx,reobs_idx] 
        #                                for ptask_idx in parent_tasks 
        #                                for reobs_idx in range(1,K[ptask_idx]+1)), 
        #                     gp.GRB.MAXIMIZE)

        # # Add constraints
        # ## Always assign first observation
        # for s,_ in enumerate(indexed_clients):
        #     model.addConstr(y[s,0] == 1, name=f"assign_initial_dummy_task_client{s}")

        # ## set unreachable tasks to y=0
        # for s,j in tqdm(y.keys(), desc=f"{state.agent_name}/PREPLANNER: Avoiding unreachable tasks", unit='tasks', leave=False):
        #     j_p_predecessors = [j_p for a,j_p,b in A if a == s and b == j]
        #     if not j_p_predecessors: model.addConstr(y[s,j] == 0, name=f"unreachable_task_client{s}_task{j}")

        # ## Enforce exclusivity
        # for s,j,j_p in tqdm(E, desc=f"{state.agent_name}/PREPLANNER: Adding mutual exclusivity constraints", unit='task pairs', leave=False):
        #     model.addConstr(y[s,j] + y[s,j_p] <= 1)

        # ## Observation time and accessibility constraints
        # for s,j in tqdm(y.keys(), desc=f"{state.agent_name}/PREPLANNER: Adding observation time constraints", unit='tasks', leave=False):
        #     # Enforce task scheduling within visibility window
        #     tau[s,j].LB = t_start[s,j]
        #     tau[s,j].UB = t_end[s,j] - d[s,j]
            
        #     # If task is not assigned (y[s,j] == 0), force tau to LB (or any fixed value in the feasible interval)
        #     model.addGenConstrIndicator(y[s,j], 0, tau[s,j] == float(t_start[s,j]))

        # ## Task sequencing constraints
        # for s,j,j_p in tqdm(z.keys(), desc=f"{state.agent_name}/PREPLANNER: Adding sequencing time constraints", unit='sequences', leave=False):
        #      # If z[s, j, j_p] == 1, enforce slew constraint for task sequence j->j' for client s
        #     model.addGenConstrIndicator(z[s,j,j_p], 1, tau[s,j] + d[s,j] + slew_times[s,j,j_p] <= tau[s,j_p])  

        # for s,j in tqdm(y.keys(), desc=f"{state.agent_name}/PREPLANNER: Adding task sequencing constraints", unit='tasks', leave=False):
        #     # Ensure that each task can have at most one successor
        #     j_p_successors = [j_p for a,b,j_p in A if a == s and b == j]
        #     if j == 0: # dummy task must have at least one successor
        #         model.addConstr(gp.quicksum(z[s,j,j_p] for j_p in j_p_successors) == 1,
        #                         name=f"min_one_successor_client{s}_task{j}")
        #     else:
        #         model.addConstr(y[s,j] == gp.quicksum(z[s,j,j_p] for j_p in j_p_successors),
        #                         name=f"max_one_successor_client{s}_task{j}")

        #     # Ensure that each task can have at most one predecessor
        #     j_p_predecessors = [j_p for a,j_p,b in A if a == s and b == j]
        #     if j > 0: # dummy task cannot have a predecessor
        #         model.addConstr(gp.quicksum(z[s,j_p,j] for j_p in j_p_predecessors) == y[s,j],
        #                         name=f"max_one_predecessor_client{s}_task{j}")

        # ## Reobservation constraints
        # for ptask_idx in tqdm(parent_task_indices, desc=f"{state.agent_name}/PREPLANNER: task reobservation counters", leave=False):
        #     ## Reobservation counter
        #     model.addConstr(n[ptask_idx] == sum([y[client_idx,task_idx] 
        #                                                   for client_idx,task_idx in task_indices
        #                                                   if parent_tasks[ptask_idx] in schedulable_tasks[indexed_clients[client_idx]][task_idx].parent_tasks]),
        #                     name=f"reobservation_count_ptask{ptask_idx}")

        #     ## Reobservation upper bound
        #     model.addConstr(n[ptask_idx] <= K[ptask_idx],
        #                     name=f"max_reobservations_ptask{ptask_idx}")

        #     ## Reobservation sequence
        #     if K[ptask_idx] == 0: continue

        #     for obs_idx in range(1,K[ptask_idx]):
        #         model.addConstr(r[ptask_idx,obs_idx] >= r[ptask_idx,obs_idx+1],
        #                         name=f"reobservation_sequence_ptask{ptask_idx}_obs{obs_idx}")

        #     ## Link reobservation counter to binary indicators
        #     model.addConstr(sum([r[ptask_idx,obs_idx] for obs_idx in range(1,K[ptask_idx]+1)]) == n[ptask_idx],
        #                     name=f"link_reob_count_to_indicators_ptask{ptask_idx}")
            
        # raise NotImplementedError("Client observation scheduling not yet implemented.") # TODO remove when finishing section
    
        # # Optimize model
        # model.optimize()

        # # Print results
        # self.__print_model_results(model)

    # def __estimate_task_rewards(self,
    #                             available_tasks : Dict[Mission, List[GenericObservationTask]],
    #                             schedulable_tasks : Dict[str, List[ObservationAction]],
    #                             parent_tasks : List[GenericObservationTask],
    #                             instruments : List[str],
    #                             indexed_clients : List[str],
    #                             K : List[int],
    #                             observation_history : ObservationHistory
    #                         ) -> np.ndarray:
    #     """ Estimate task rewards for each client and task based on parent tasks and reobservation number """
    #     # Initialize rewards array
    #     rewards = np.array([[0 for _ in range(K[ptask_idx])] for ptask_idx,_ in enumerate(parent_tasks)])

    #     # Calculate rewards for each observation
    #     for ptask_idx, ptask in enumerate(parent_tasks):
    #         # gather
    #         relevant_specfic_tasks : list[tuple[str,SpecificObservationTask]] \
    #             = sorted([(client,task)
    #                         for client,tasks in schedulable_tasks.items() 
    #                         for task in tasks
    #                         if ptask in task.parent_tasks], 
    #                     key=lambda x: x[1].accessibility.left)

    #         for reobs_idx,(client,task) in enumerate(relevant_specfic_tasks):
    #             task : SpecificObservationTask = task  # type casting for clarity
    #             rewards[ptask_idx,reobs_idx] = self.estimate_task_value(task, 
    #                                                                     task.accessibility.left, 
    #                                                                     task.min_duration, 
    #                                                                     self.client_specs[client], 
    #                                                                     self.cross_track_fovs[client], 
    #                                                                     self.client_orbitdata[client], 
    #                                                                     self.client_missions[client], 
    #                                                                     observation_history,
    #                                                                     reobs_idx)
                
    #         x = 1

    #     # Populate rewards array
    #     # for s,client in enumerate(indexed_clients):
    #     #     for j,task_j in enumerate(schedulable_tasks[client]):
    #     #         if not isinstance(task_j, SpecificObservationTask): continue
    #     #         for (m,mission),mission_tasks in available_tasks.items():
    #     #             if task_j not in mission_tasks: continue
    #     #             for p,ptask in enumerate(parent_tasks):
    #     #                 if ptask not in mission_tasks: continue
    #     #                 if not task_j.is_reobservation_of(ptask): continue
    #     #                 for i,instr in enumerate(instruments):
    #     #                     if instr != task_j.instrument_name: continue
    #     #                     for o in range(self.max_observations):
    #     #                         if o == 0:
    #     #                             rewards[s,j,p,i,o] = task_j.base_reward
    #     #                         else:
    #     #                             # Example: exponential decay of reward for reobservations
    #     #                             rewards[s,j,p,i,o] = task_j.base_reward * (0.5 ** o)
    #     # return rewards

    #     raise NotImplementedError("Task reward estimation not yet implemented. Working on it...")