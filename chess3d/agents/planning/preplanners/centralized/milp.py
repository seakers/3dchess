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

    @runtime_tracker
    def _schedule_client_observations(self, 
                                      state : SimulationAgentState,
                                      available_tasks : Dict[Mission, List[GenericObservationTask]],
                                      schedulable_tasks: Dict[str, List[ObservationAction]], 
                                      observation_history : ObservationHistory
                                    ) -> Dict[str, List[ObservationAction]]:
        """ schedules observations for all clients """

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

        # Create a new model
        model = gp.Model("multi-mission_milp_planner")

        # Set parameter to suppress output
        model.setParam('OutputFlag', int(self._debug))

        # Initialize constants
        t_start   = {client : np.array([task.accessibility.left-state.t 
                              for task in tasks if isinstance(task, SpecificObservationTask)
                              ])
                              for client,tasks in schedulable_tasks.items()
                            }
        t_end     = {client : np.array([task.accessibility.right-state.t for task in tasks if isinstance(task, SpecificObservationTask)])
                     for client,tasks in schedulable_tasks.items()
                    }
        d         = {client : np.array([task.min_duration for task in tasks if isinstance(task, SpecificObservationTask)])
                     for client,tasks in schedulable_tasks.items()
                    }
        th_imgs   = {client : np.array([np.average((task.slew_angles.left, task.slew_angles.right)) for task in tasks if isinstance(task, SpecificObservationTask)])
                     for client,tasks in schedulable_tasks.items()
                    }
        slew_times= {client :  np.array([[abs(th_imgs[client][j_p]-th_imgs[client][j]) / max_slew_rates[client] 
                                           for j,_ in enumerate(tasks)]
                                         for j_p,_ in enumerate(tasks)]
                                        )
                     for client,tasks in schedulable_tasks.items()
                    }
        
        Z = {client : [(j, j_p) 
                        for j,_    in enumerate(tasks) 
                        for j_p,_  in enumerate(tasks)
                        if t_start[client][j] + d[client][j] + slew_times[client][j, j_p] <= t_end[client][j_p] - d[client][j_p] # sequence j->j' is feasible
                        and j != j_p                                                                                            # ensure distinct tasks
                    ]
             for client,tasks in schedulable_tasks.items()
             } 
        
        # index tasks and clients
        indexed_clients = list(schedulable_tasks.keys())
        client_indices = {client : idx for idx,client in enumerate(indexed_clients)}

        indexed_missions = list(available_tasks.keys())
        mission_indices = {mission : idx for idx,mission in enumerate(indexed_missions)}

        # indexed_parent_tasks = [pt for pt in range(sum(len(available_tasks[mission]) for mission in indexed_missions))]
        indexed_parent_tasks = []
        parent_task_indices = {pt : idx for idx,pt in enumerate(indexed_parent_tasks)}
        
        indexed_tasks = [list(range(len(schedulable_tasks[client]))) 
                        for client in indexed_clients]
        task_indices = [(client,task_index) 
                        for client,tasks in enumerate(indexed_tasks)
                        for task_index in tasks
                        ]

        # Create decision variables
        x : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.BINARY, name="x")
        z : gp.tupledict = model.addVars(Z, vtype=gp.GRB.BINARY, name="z")
        tau : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.CONTINUOUS, name="tau")

        # return {client: [] for client in self.client_orbitdata} # temporarily disable MILP planner
        raise NotImplementedError("Client observation scheduling not yet implemented.")

        # for each client, create variables and constraints
        for mission,generic_tasks in available_tasks.items():
            x =1 

        # return {client: [] for client in self.client_orbitdata} # temporarily disable MILP planner
        raise NotImplementedError("Client observation scheduling not yet implemented.")