import os
from chess3d.agents.planning.planner import *
from chess3d.agents.planning.preplanners.preplanner import AbstractPreplanner

import gurobipy as gp
import numpy as np

class SingleSatMILP(AbstractPreplanner):
    def __init__(self, 
                 objective : str, 
                 licence_path : str, 
                 horizon = np.Inf, 
                 period = np.Inf, 
                 debug = False, 
                 logger = None
                ):
        super().__init__(horizon, period, debug, logger)

        if not debug:
            assert os.path.isfile(licence_path), f"Provided Gurobi licence path `{licence_path}` is not a valid file."
            os.environ['GRB_LICENSE_FILE'] = licence_path

        assert objective in ["reward", "duration"], "Objective must be either 'reward' or 'duration'."

        self.objective = objective

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
        # Set type for `schedulable_tasks`
        schedulable_tasks : list[SpecificObservationTask] = schedulable_tasks
        if not schedulable_tasks: return []

        self.__class__.__name__
        
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
        
        # Add dummy task to represent initial state
        dummy_task = SpecificObservationTask(set([]), 
                                             schedulable_tasks[0].instrument_name, 
                                             Interval(state.t,state.t), 
                                             0.0, 
                                             Interval(state.attitude[0],state.attitude[0]))
        schedulable_tasks.insert(0,dummy_task)

        # Create a new model
        model = gp.Model("single-sat_milp_planner")

        # Set parameter to suppress output
        model.setParam('OutputFlag', int(self._debug))

        # List tasks by their index
        indexed_tasks = list(enumerate(schedulable_tasks))
        task_indices = [j for j,_ in indexed_tasks]
        # task_indices = list(range(len(schedulable_tasks)))

        # Create decision variables
        x : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.BINARY, name="x")
        z : gp.tupledict = model.addVars(task_indices, task_indices, vtype=gp.GRB.BINARY, name="z")
        tau : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.CONTINUOUS, lb=0, ub=self.horizon, name="tau")
        
        # Set constants
        init_rewards = np.array([self.estimate_task_value(task, 
                                                     task.accessibility.left, 
                                                     task.min_duration, 
                                                     specs, cross_track_fovs, orbitdata, 
                                                     mission, 
                                                     observation_history)
                            for task in tqdm(schedulable_tasks,leave=False,desc='SATELLITE: Calculating task rewards')
                            if isinstance(task,SpecificObservationTask)])
        final_rewards = np.array([self.estimate_task_value(task, 
                                                     task.accessibility.right-task.min_duration, 
                                                     task.min_duration, 
                                                     specs, cross_track_fovs, orbitdata, 
                                                     mission, 
                                                     observation_history)
                            for task in tqdm(schedulable_tasks,leave=False,desc='SATELLITE: Calculating task rewards')
                            if isinstance(task,SpecificObservationTask)])
        t_start   = np.array([task.accessibility.left-state.t for task in schedulable_tasks if isinstance(task, SpecificObservationTask)])
        t_end     = np.array([task.accessibility.right-state.t for task in schedulable_tasks if isinstance(task, SpecificObservationTask)])
        d         = np.array([task.min_duration for task in schedulable_tasks if isinstance(task, SpecificObservationTask)])
        th_imgs   = np.array([np.average((task.slew_angles.left, task.slew_angles.right)) for task in schedulable_tasks if isinstance(task, SpecificObservationTask)])
        slew_time = np.array([[abs(th_imgs[j_p]-th_imgs[j]) / max_slew_rate 
                               for j,_ in enumerate(schedulable_tasks)]
                               for j_p,__ in enumerate(schedulable_tasks)
                             ])
        M         = np.array([[max((t_end[j], t_end[j_p])) + slew_time[j, j_p]  
                               for j,_ in enumerate(schedulable_tasks)]
                               for j_p,__ in enumerate(schedulable_tasks)
                             ]) # conservative estimate of big M as a function of each task
        exclusive = np.array([[int(task_i.is_mutually_exclusive(task_j)) for task_j in schedulable_tasks]
                               for task_i in schedulable_tasks])

        # Validate constants to ensure convergence
        assert all([reward >= 0 for reward in init_rewards])
        assert all([reward >= 0 for reward in final_rewards])
        assert all([0 <= t_start[j] <= self.horizon for j in task_indices])
        assert all([0 <= t_end[j] <= self.horizon for j in task_indices])
        assert all([t_end[j] - t_start[j] >= d[j] for j in task_indices])
        assert all([t_end[j] - d[j] >= 0.0 for j in task_indices])
        assert all([slew_time[j][j_p] >= 0 for j in task_indices for j_p in task_indices])
        assert len(x)**2 == len(z) and len(z) == len(tau)**2 and len(tau) == len(init_rewards)

        # Set objective
        if self.objective == "reward": 
            model.setObjective(gp.quicksum( init_rewards[j] * x[j]
                                           for j in task_indices), gp.GRB.MAXIMIZE)
        
        # Assign dummy observation
        model.addConstr(x[0] == 1)
        
        # Add constraints
        for j in tqdm(x.keys(), desc=f"{state.agent_name}/PREPLANNER: Building optimization model", unit='tasks', leave=False):
            
            # Observation time constraint
            model.addConstr(t_start[j] * x[j] <= tau[j])
            model.addConstr(tau[j] <= (t_end[j] - d[j]) * x[j])
           
            # Slew time constraints for observation task sequence j->j'
            for j_p in x.keys():
                # # set slew constraint for task sequence j->j'
                # model.addGenConstrIndicator(z[j,j_p], 1,
                #     tau[j] + d[j] + slew_time[j,j_p] <= tau[j_p], name=f"prec_{j}_{j_p}")

                # set slew constraint for task sequence j->j'
                model.addConstr(tau[j] + d[j] + slew_time[j, j_p] <= tau[j_p] + M[j, j_p] * (1 - z[j, j_p]))

                if j != j_p:
                    # enforce lower bound for z[j, j'] and z[j', j]
                    model.addConstr(x[j] + x[j_p] - 1 <= z[j, j_p] + z[j_p, j])

                    # enforce upper bounds for z[j, j_p] and z[j_p, j]
                    model.addConstr(z[j, j_p] + z[j_p, j] <= 1) # only z[j, j'] or z[j', j] can be assigned at a time
                    model.addConstr(z[j, j_p] <= x[j])          # z[j, j'] cannot be assigned if x[j] is not assigned
                    model.addConstr(z[j, j_p] <= x[j_p])        # z[j, j'] cannot be assigned if x[j'] is not assigned
                else:
                    # cannot form a sequence of observations between the same task; constrain z[j, j] == 0
                    model.addConstr(z[j, j] == 0)

                # enforce exclusivity
                if exclusive[j, j_p]: model.addConstr(x[j] + x[j_p] <= 1)

        # Optimize model
        model.optimize()

        # Print results
        print("\nStatus code:", model.Status)

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
            
                print("Computing IIS...")
                model.computeIIS()
                model.write("./chess3d/agents/planning/preplanners/decentralized/milp_model.ilp")
                model.write("./chess3d/agents/planning/preplanners/decentralized/milp_model.ilp.json")
                print("IIS written to `./chess3d/agents/planning/preplanners/decentralized/milp_model.ilp`")

            if model.Status == gp.GRB.UNBOUNDED:
                print("Model is unbounded.")

            if model.Status == gp.GRB.INTERRUPTED:
                print("Model was interrupted.")

            raise ValueError(f"Unexpected model status: {model.Status}")

        # Extract scheduled tasks
        scheduled_task_indices : list[Tuple[int,float,float,float,SpecificObservationTask]] \
            = [(j,tau[j].X,d[j],th_imgs[j],schedulable_tasks[j]) 
               for j in task_indices 
               if x[j].X > 0.5          # select only assigned tasks
               and j > 0                # exclude dummy task
               ]

        # Create observation actions from scheduled tasks
        observations = [
            ObservationAction(task.instrument_name,
                              th_img,
                              t_img+state.t,
                              d_img,
                              task
                              )
            for _,t_img,d_img,th_img,task in sorted(scheduled_task_indices,key=lambda x: x[1]) # Sorted by start time
        ]

        # Return scheduled tasks
        return sorted(observations, key=lambda obs: obs.t_start)

    def _schedule_broadcasts(self, state, observations, orbitdata, t = None):
        # TODO 
        return super()._schedule_broadcasts(state, observations, orbitdata, t)