import os
from chess3d.agents.planning.planner import *
from chess3d.agents.planning.preplanners.preplanner import AbstractPreplanner

import gurobipy as gp
from gurobipy import GRB
import numpy as np

class SingleSatMILP(AbstractPreplanner):
    def __init__(self, 
                 objective : str, 
                 licence_path : str, 
                 horizon = np.Inf, period = np.Inf, debug = False, logger = None):
        super().__init__(horizon, period, debug, logger)

        assert os.path.isfile(licence_path), f"Provided Gurobi licence path `{licence_path}` is not a valid file."
        os.environ['GRB_LICENSE_FILE'] = licence_path

        assert objective in ["reward", "duration"], "Objective must be either 'reward' or 'duration'."

        self.objective = objective

    def calc_big_m(self, schedulable_tasks : list, specs : object, orbitdata : OrbitData, observation_history : ObservationHistory):
        """
        Calculate the big-M constant for the MILP formulation.
        This is a placeholder function and should be implemented based on specific requirements.
        """
        t_end = max(task.accessibility.right for task in schedulable_tasks if isinstance(task, SpecificObservationTask)) if schedulable_tasks else 0
        t_start = min(task.accessibility.left for task in schedulable_tasks if isinstance(task, SpecificObservationTask)) if schedulable_tasks else 0
        return t_end + self.horizon  # Example calculation, adjust as needed

    @runtime_tracker
    def _schedule_observations(self, 
                               state : SimulationAgentState, 
                               specs : object, 
                               _ : ClockConfig, 
                               orbitdata : OrbitData, 
                               schedulable_tasks : list,
                               observation_history : ObservationHistory
                               ) -> list:
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
        
        assert max(task.accessibility.right for task in schedulable_tasks if isinstance(task, SpecificObservationTask)) <= state.t + self.horizon, \
            f"Tasks exceed the planning horizon of {self.horizon} seconds. "

        # Create a new model
        model = gp.Model("single-sat_milp_planner")

        # Set parameter to suppress output
        # model.setParam('OutputFlag', int(self.debug))

        # List tasks by their index
        task_indices = list(range(len(schedulable_tasks)))

        # Create decision variables
        x = model.addVars(task_indices, vtype=GRB.BINARY, name="x")
        z = model.addVars(task_indices, task_indices, vtype=GRB.BINARY, name="z")
        tau = model.addVars(task_indices, vtype=GRB.CONTINUOUS, lb=0, name="tau")
        delta = model.addVars(task_indices, vtype=GRB.CONTINUOUS, lb=0, name="delta")
        
        # Set constants
        rewards = np.array([self.calc_task_reward(task, specs, cross_track_fovs, orbitdata, observation_history)
                            for task in tqdm(schedulable_tasks,leave=False,desc='SATELLITE: Calculating task rewards')])
        t_start = np.array([task.accessibility.left for task in schedulable_tasks if isinstance(task, SpecificObservationTask)])
        t_end   = np.array([task.accessibility.right for task in schedulable_tasks if isinstance(task, SpecificObservationTask)])
        d       = np.array([0 for _ in schedulable_tasks])  # Assuming no minimum measurement duration for simplicity
        th_imgs = np.array([np.average((task.slew_angles.left, task.slew_angles.right)) for task in schedulable_tasks if isinstance(task, SpecificObservationTask)])
        m       = np.array([[abs(th_imgs[i]-th_imgs[j]) / max_slew_rate 
                            for j,_ in enumerate(schedulable_tasks)]
                            for i,__ in enumerate(schedulable_tasks)
                            ])
        M = self.calc_big_m(schedulable_tasks, specs, orbitdata, observation_history)

        # Set objective
        if self.objective == "reward": 
            model.setObjective(gp.quicksum(rewards[i] * x[i] for i in task_indices), GRB.MAXIMIZE)
        elif self.objective == "duration":
            model.setObjective(gp.quicksum(rewards[i] * delta[i] for i in task_indices), GRB.MAXIMIZE)
        
        # Add constraints
        for j in tqdm(x.keys(), desc=f"{state.agent_name}/PREPLANNER: Building optimization model", unit='tasks', leave=False):
            # Observation time constraint
            model.addConstr(t_start[j] * x[j] <= tau[j], 
                            name=f"observation_start_time_constraint_1_{j}")
            model.addConstr(tau[j] <= t_end[j] * x[j], 
                            name=f"observation_start_time_constraint_2_{j}")
            
            model.addConstr(tau[j] + delta[j] <= t_end[j] * x[j], 
                            name=f"observation_start_and_duration_constraint_{j}")
            
            # Observation duration constraints
            model.addConstr(d[j] * x[j] <= delta[j], 
                            name=f"min_duration_constraint_{j}")
            model.addConstr(delta[j] <= x[j] * (t_end[j] - t_start[j]), 
                            name=f"max_duration_constraint_{j}")
            
            # Slew time constraints
            for j_p in x.keys():
                if j != j_p:
                    

                    model.addConstr(tau[j] + delta[j] + m[j, j_p] <= tau[j_p] + M * (1 - z[j, j_p]),
                                    name=f"slew_time_constraint_1_{j}_{j_p}")
                    model.addConstr(tau[j_p] + delta[j_p] + m[j_p, j] <= tau[j] + M * (1 - z[j, j_p]), 
                                    name=f"slew_time_constraint_2_{j}_{j_p}")
                    
                else:
                    model.addConstr(z[j, j_p] == 0, 
                                    name=f"slew_binary_constraint_2_{j}_{j_p}_self")
                    model.addConstr(z[j_p, j] == 0, 
                                    name=f"slew_binary_constraint_1_{j_p}_{j}_self")
                    
                #TODO Add constraints for tasks with the same parent task
                # if schedulable_tasks[j].parent == schedulable_tasks[j_p].parent:
                #     model.addConstr(x[j] + x[j_p] <= 1, name=f"parent_task_sequence_{j}_{j_p}")

                model.addConstr(z[j, j_p] + z[j_p, j] <= 1, name=f"sequence_direction_{j}_{j_p}")
                model.addConstr(z[j, j_p] <= x[j], name=f"z_x_link1_{j}_{j_p}")
                model.addConstr(z[j, j_p] <= x[j_p], name=f"z_x_link2_{j}_{j_p}")
                model.addConstr(z[j_p, j] <= x[j], name=f"z_x_link3_{j_p}_{j}")
                model.addConstr(z[j_p, j] <= x[j_p], name=f"z_x_link4_{j_p}_{j}")

        # Optimize model
        model.optimize()

        # Print results
        print("\nStatus code:", model.Status)

        if model.Status == GRB.OPTIMAL:
            print("Optimal solution found.")
            # for i in task_indices:
            #     if x[i].X > 0.5:
            #         print(f"Task {i} is scheduled.")
            print(f"Obj: {model.ObjVal:g}")
        else:
            if model.Status == GRB.INFEASIBLE:
                print("Model is infeasible.")
            
                print("Computing IIS...")
                model.computeIIS()
                model.write("model.ilp")
                model.write("model.ilp.json")
                print("IIS written to model.ilp")

            elif model.Status == GRB.UNBOUNDED:
                print("Model is unbounded.")

            elif model.Status == GRB.INTERRUPTED:
                print("Model was interrupted.")
            
            raise ValueError(f"Unexpected model status: {model.Status}")

        # Extract scheduled tasks
        scheduled_task_indices = [i for i in task_indices if x[i].X > 0.5]
        scheduled_tasks : list[SpecificObservationTask] = \
            [schedulable_tasks[i] for i in scheduled_task_indices]
        
        # Create observation actions from scheduled tasks
        observations = []

        # TODO Remove exception and return scheduled tasks
        raise NotImplementedError("MILP scheduling is not yet fully implemented.")
        return observations

    def _schedule_broadcasts(self, state, observations, orbitdata, t = None):
        # TODO 
        return super()._schedule_broadcasts(state, observations, orbitdata, t)