import logging
import os
from typing import Dict

from dmas.utils import runtime_tracker

import pandas as pd
from pyparsing import List
import gurobipy as gp
import numpy as np

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
        
        # index task and client info 
        instruments : list[str] = list({instrument.name 
                                        for client_name,specs in self.client_specs.items() 
                                        for instrument in specs.instrument})

        parent_tasks : list[GenericObservationTask] = list({ptask for mission_ptasks in available_tasks.values() for ptask in mission_ptasks})
        parent_task_indices : list[int] = [(task_idx,istr_idx) 
                                           for task_idx,_ in enumerate(parent_tasks)
                                           for istr_idx,_ in enumerate(instruments)]

        reobservation_indices : list[tuple[int,int]] = [(task_idx,istr_ids,obs_idx) 
                                                        for task_idx,_ in enumerate(parent_tasks) 
                                                        for istr_ids,_ in enumerate(instruments)
                                                        for obs_idx in range(self.max_observations)]

        indexed_clients : list[str] = list(schedulable_tasks.keys())
        client_indices : Dict[str, int] = {client : idx for idx,client in enumerate(indexed_clients)}

        indexed_missions : list[Mission] = list(available_tasks.keys())
        mission_indices : Dict[Mission, int] = {mission : idx for idx,mission in enumerate(indexed_missions)}

        task_indices : list[tuple[int,int]] = [(client_index,task_index) 
                                                for client_index,client in enumerate(indexed_clients)
                                                for task_index,_ in enumerate(schedulable_tasks[client])]
        
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

        # Map sparce arch matrix of feasible task sequences
        A = [(s,j,j_p)
             for s,client in enumerate(indexed_clients)
             for j,task_j in enumerate(schedulable_tasks[client]) if isinstance(task_j, SpecificObservationTask)
             for j_p,task_j_p in enumerate(schedulable_tasks[client]) if isinstance(task_j_p, SpecificObservationTask)
             if t_start[s,j] + d[s,j] + slew_times[s,j,j_p]
                <= t_end[s,j_p] - d[s,j_p]    # sequence j->j' is feasible
                and j != j_p                   # ensure distinct tasks
            ]

        # Create decision variables
        y : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.BINARY, name="y")
        tau : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.CONTINUOUS, name="tau")
        z : gp.tupledict = model.addVars(A, vtype=gp.GRB.BINARY, name="z")
        n : gp.tupledict = model.addVars(parent_task_indices, vtype=gp.GRB.CONTINUOUS, name="n", ub=self.max_observations)
        r : gp.tupledict = model.addVars(reobservation_indices, vtype=gp.GRB.BINARY, name="r")

        # return {client: [] for client in self.client_orbitdata} # temporarily disable MILP planner
        raise NotImplementedError("Client observation scheduling not yet implemented.")

        # for each client, create variables and constraints
        for mission,generic_tasks in available_tasks.items():
            x =1 

        # return {client: [] for client in self.client_orbitdata} # temporarily disable MILP planner
        raise NotImplementedError("Client observation scheduling not yet implemented.")