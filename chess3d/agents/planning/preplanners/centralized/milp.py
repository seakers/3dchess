import inspect
import logging
import os
from typing import Dict

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
    """
    EARLIEST = 'earliest'
    BEST = 'best'

    def __init__(self, 
                 client_orbitdata : Dict[str, OrbitData],
                 client_specs : Dict[str, object],
                 client_missions : Dict[str, Mission],
                 objective : str, 
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
        assert objective in ["reward", "duration"], "Objective must be either 'reward' or 'duration'."
        assert model in [self.EARLIEST, self.BEST], f"Model must be either '{self.EARLIEST}' or '{self.BEST}'."
        assert (isinstance(max_tasks, int) and max_tasks > 0) or max_tasks == np.Inf, "Max tasks must be a positive integer."

        # Set attributes
        self.objective = objective
        self.model = model
        self.max_tasks = max_tasks
        self.max_observations = max_observations

    @runtime_tracker
    def _schedule_client_observations(self, 
                                      state : SimulationAgentState,
                                      available_tasks : Dict[Mission, List[GenericObservationTask]],
                                      schedulable_tasks: Dict[str, List[ObservationAction]], 
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
        K : list[list[int]] = [[len([task 
                                        for client_tasks in schedulable_tasks.values() 
                                        for task in client_tasks 
                                        if ptask in task.parent_tasks and task.instrument_name == instrument])
                                    for instrument in instruments]
                                for ptask in parent_tasks]

        # index task and client info 
        parent_task_indices : list[int] = [(task_idx,instr_idx) 
                                           for task_idx,_ in enumerate(parent_tasks)
                                           for instr_idx,_ in enumerate(instruments)]

        reobservation_indices : list[tuple[int,int]] = [(task_idx,instr_idx,obs_idx) 
                                                        for task_idx,_ in enumerate(parent_tasks) 
                                                        for instr_idx,_ in enumerate(instruments)
                                                        for obs_idx in range(1,K[task_idx][instr_idx]+1)]

        indexed_clients : list[str] = list(schedulable_tasks.keys())
        client_indices : Dict[str, int] = {client : idx 
                                           for idx,client in enumerate(indexed_clients)}

        indexed_missions : list[Mission] = list(available_tasks.keys())
        mission_indices : Dict[Mission, int] = {mission : idx 
                                                for idx,mission in enumerate(indexed_missions)}

        task_indices : list[tuple[int,int]] = [(client_idx,task_idx) 
                                                for client_idx,client in enumerate(indexed_clients)
                                                for task_idx,_ in enumerate(schedulable_tasks[client])]
        
        # Create a new model
        model = gp.Model("multi-mission_milp_planner")

        # Set parameter to suppress output
        model.setParam('OutputFlag', int(self._debug))

        # Initialize constants
        t_start   = np.array([[task.accessibility.left-state.t 
                              for task in schedulable_tasks[client]  if isinstance(task, SpecificObservationTask)]
                              for client in indexed_clients])
        t_end     = np.array([[task.accessibility.right-state.t 
                              for task in schedulable_tasks[client]  if isinstance(task, SpecificObservationTask)]
                              for client in indexed_clients])
        d         = np.array([[task.min_duration 
                              for task in schedulable_tasks[client]  if isinstance(task, SpecificObservationTask)]
                              for client in indexed_clients])
        th_imgs   = np.array([[np.average((task.slew_angles.left, task.slew_angles.right)) 
                              for task in schedulable_tasks[client]  if isinstance(task, SpecificObservationTask)]
                              for client in indexed_clients])
        slew_times= np.array([[[abs(th_imgs[client_index][j_p]-th_imgs[client_index][j]) / max_slew_rates[indexed_clients[client_index]]
                                for j_p,task_j_p in enumerate(schedulable_tasks[client]) if isinstance(task_j_p, SpecificObservationTask)]
                                for j,task_j in enumerate(schedulable_tasks[client]) if isinstance(task_j, SpecificObservationTask)]
                                for client_index, client in enumerate(indexed_clients)])
        
        # TODO Calculate incremental rewards for each task based on parent tasks and reobservation number 
        # rewards = self.__estimate_task_rewards(available_tasks, schedulable_tasks, parent_tasks, instruments, indexed_clients)

        # Map sparce arch matrix of feasible task sequences
        A : list = [(s,j,j_p)
                        for s,client in enumerate(indexed_clients)
                        for j,task_j in enumerate(schedulable_tasks[client]) if isinstance(task_j, SpecificObservationTask)
                        for j_p,task_j_p in enumerate(schedulable_tasks[client]) if isinstance(task_j_p, SpecificObservationTask)
                        if t_start[s,j] + d[s,j] + slew_times[s,j,j_p]
                            <= t_end[s,j_p] - d[s,j_p]    # sequence j->j' is feasible
                            and j != j_p                  # ensure distinct tasks
                    ]
        
        # Map task mutual exclusivity constraints
        E : list = [(s,j,j_p)
                    for s,client in enumerate(indexed_clients)
                    for j,task_j in enumerate(schedulable_tasks[client]) if isinstance(task_j, SpecificObservationTask)
                    for j_p,task_j_p in enumerate(schedulable_tasks[client]) if isinstance(task_j_p, SpecificObservationTask)
                    if (task_j.is_mutually_exclusive(task_j_p) 
                        or (t_start[s,j] + d[s,j] + slew_times[s,j,j_p]
                            <= t_end[s,j_p] - d[s,j_p]    # sequence j->j' is feasible
                            and j != j_p                 # ensure distinct tasks
                        )
                    ) and j < j_p   # avoid duplicates
                ]

        # Create decision variables
        y : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.BINARY, name="y")
        tau : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.CONTINUOUS, name="tau")
        z : gp.tupledict = model.addVars(A, vtype=gp.GRB.BINARY, name="z")
        n : gp.tupledict = model.addVars(parent_task_indices, vtype=gp.GRB.INTEGER, name="n", ub=self.max_observations)
        r : gp.tupledict = model.addVars(reobservation_indices, vtype=gp.GRB.BINARY, name="r")

        # TODO set objective

        # Add constraints
        ## Always assign first observation
        for s,_ in enumerate(indexed_clients):
            model.addConstr(y[s,0] == 1, name=f"assign_initial_dummy_task_client{s}")

        ## set unreachable tasks to y=0
        for s,j in tqdm(y.keys(), desc=f"{state.agent_name}/PREPLANNER: Avoiding unreachable tasks", unit='tasks', leave=False):
            j_p_predecessors = [j_p for a,j_p,b in A if a == s and b == j]
            if not j_p_predecessors:
                model.addConstr(y[s,j] == 0, name=f"unreachable_task_client{s}_task{j}")

        ## Enforce exclusivity
        for s,j,j_p in tqdm(E, desc=f"{state.agent_name}/PREPLANNER: Adding mutual exclusivity constraints", unit='task pairs', leave=False):
            model.addConstr(y[s,j] + y[s,j_p] <= 1)

        ## Observation time and accessibility constraints
        for s,j in tqdm(y.keys(), desc=f"{state.agent_name}/PREPLANNER: Adding observation time constraints", unit='tasks', leave=False):
            # Enforce task scheduling within visibility window
            tau[s,j].LB = t_start[s,j]
            tau[s,j].UB = t_end[s,j] - d[s,j]
            
            # If task is not assigned (y[s,j] == 0), force tau to LB (or any fixed value in the feasible interval)
            model.addGenConstrIndicator(y[s,j], 0, tau[s,j] == float(t_start[s,j]))

        ## Task sequencing constraints
        for s,j,j_p in tqdm(z.keys(), desc=f"{state.agent_name}/PREPLANNER: Adding sequencing time constraints", unit='sequences', leave=False):
             # If z[s, j, j_p] == 1, enforce slew constraint for task sequence j->j' for client s
            model.addGenConstrIndicator(z[s,j,j_p], 1, tau[s,j] + d[s,j] + slew_times[s,j,j_p] <= tau[s,j_p])  

        for s,j in tqdm(y.keys(), desc=f"{state.agent_name}/PREPLANNER: Adding task sequencing constraints", unit='tasks', leave=False):
            # Ensure that each task can have at most one successor
            j_p_successors = [j_p for a,b,j_p in A if a == s and b == j]
            if j == 0: # dummy task must have at least one successor
                model.addConstr(gp.quicksum(z[s,j,j_p] for j_p in j_p_successors) == 1,
                                name=f"min_one_successor_client{s}_task{j}")
            else:
                model.addConstr(gp.quicksum(z[s,j,j_p] for j_p in j_p_successors) == y[s,j],
                                name=f"max_one_successor_client{s}_task{j}")

            # Ensure that each task can have at most one predecessor
            j_p_predecessors = [j_p for a,j_p,b in A if a == s and b == j]
            if j > 0: # dummy task cannot have a predecessor
                model.addConstr(gp.quicksum(z[s,j_p,j] for j_p in j_p_predecessors) == y[s,j],
                                name=f"max_one_predecessor_client{s}_task{j}")

        ## Reobservation constraints
        for ptask_idx,instr_idx in tqdm(parent_task_indices, desc=f"{state.agent_name}/PREPLANNER: task reobservation counters", leave=False):
            ## Reobservation counter
            model.addConstr(n[ptask_idx,instr_idx] == sum([y[client_idx,task_idx] 
                                                          for client_idx,task_idx in task_indices
                                                          if parent_tasks[ptask_idx] in schedulable_tasks[indexed_clients[client_idx]][task_idx].parent_tasks
                                                          and schedulable_tasks[indexed_clients[client_idx]][task_idx].instrument_name == instruments[instr_idx]]),
                            name=f"reobservation_count_ptask{ptask_idx}_instr{instr_idx}")

            ## Reobservation upper bound
            model.addConstr(n[ptask_idx,instr_idx] <= K[ptask_idx][instr_idx],
                            name=f"max_reobservations_ptask{ptask_idx}_instr{instr_idx}")
            
            ## Reobservation sequence
            if K[ptask_idx][instr_idx] == 0: continue

            for obs_idx in range(1,K[ptask_idx][instr_idx]):
                model.addConstr(r[ptask_idx,instr_idx,obs_idx] >= r[ptask_idx,instr_idx,obs_idx+1],
                                name=f"reobservation_sequence_ptask{ptask_idx}_instr{instr_idx}_obs{obs_idx}")
                
            ## Link reobservation counter to binary indicators
            model.addConstr(n[ptask_idx,instr_idx] == sum([r[ptask_idx,instr_idx,obs_idx] for obs_idx in range(1,K[ptask_idx][instr_idx]+1)]),
                            name=f"link_reob_count_to_indicators_ptask{ptask_idx}_instr{instr_idx}")
            
        raise NotImplementedError("Client observation scheduling not yet implemented.") # TODO remove when finishing section
    
        # Optimize model
        model.optimize()

        # Print results
        self.__print_model_results(model)

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

    
    def __estimate_task_rewards(self,
                                available_tasks : Dict[Mission, List[GenericObservationTask]],
                                schedulable_tasks : Dict[str, List[ObservationAction]],
                                parent_tasks : List[GenericObservationTask],
                                instruments : List[str],
                                indexed_clients : List[str]
                            ) -> np.ndarray:
        """ Estimate task rewards for each client and task based on parent tasks and reobservation number """
        # Initialize rewards array
        rewards = np.zeros((len(indexed_clients), 
                            max([len(schedulable_tasks[client]) 
                                 for client in indexed_clients]), 
                            len(parent_tasks), 
                            len(instruments), 
                            self.max_observations))

        # Populate rewards array
        # for s,client in enumerate(indexed_clients):
        #     for j,task_j in enumerate(schedulable_tasks[client]):
        #         if not isinstance(task_j, SpecificObservationTask): continue
        #         for (m,mission),mission_tasks in available_tasks.items():
        #             if task_j not in mission_tasks: continue
        #             for p,ptask in enumerate(parent_tasks):
        #                 if ptask not in mission_tasks: continue
        #                 if not task_j.is_reobservation_of(ptask): continue
        #                 for i,instr in enumerate(instruments):
        #                     if instr != task_j.instrument_name: continue
        #                     for o in range(self.max_observations):
        #                         if o == 0:
        #                             rewards[s,j,p,i,o] = task_j.base_reward
        #                         else:
        #                             # Example: exponential decay of reward for reobservations
        #                             rewards[s,j,p,i,o] = task_j.base_reward * (0.5 ** o)
        return rewards