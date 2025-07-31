from itertools import repeat
from logging import Logger
import math
from tqdm import tqdm

from orbitpy.util import Spacecraft

from dmas.clocks import ClockConfig
from dmas.utils import runtime_tracker
from dmas.clocks import *

from chess3d.agents.planning.preplanners.preplanner import AbstractPreplanner
from chess3d.agents.planning.tasks import ObservationHistory, SchedulableObservationTask
from chess3d.agents.states import *
from chess3d.agents.actions import *
from chess3d.agents.science.requests import *
from chess3d.agents.states import SimulationAgentState
from chess3d.orbitdata import OrbitData
from chess3d.messages import *
from chess3d.utils import Interval

class DynamicProgrammingPlanner(AbstractPreplanner):
    def __init__(self, 
                #  sharing : bool = False,
                 horizon: float = np.Inf, 
                 period : float = np.Inf, 
                #  points : float = np.Inf,
                 debug : bool = False,
                 logger: Logger = None
                 ) -> None:
        super().__init__(horizon, 
                         period, 
                        #  sharing, 
                         debug, 
                         logger)
        
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
        
        # compile list of instruments available in payload
        payload : dict = {instrument.name: instrument for instrument in specs.instrument}
        
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self.collect_fov_specs(specs)
        
        # get pointing agility specifications
        adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
        assert adcs_specs, 'ADCS component specifications missing from agent specs object.'

        max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
        assert max_slew_rate, 'ADCS `maxRate` specification missing from agent specs object.'

        max_torque = float(adcs_specs['maxTorque']) if adcs_specs.get('maxTorque', None) is not None else None
        assert max_torque, 'ADCS `maxTorque` specification missing from agent specs object.'

        # sort tasks by start time
        schedulable_tasks : list[SchedulableObservationTask] = sorted(schedulable_tasks, key=lambda t: (t.accessibility.left, -len(t.parent_tasks)))

        # add dummy task to represent initial state
        instrument_names = list(payload.keys())
        dummy_task = SchedulableObservationTask(set([]), instrument_names[0], Interval(state.t,state.t), Interval(state.attitude[0],state.attitude[0]))
        schedulable_tasks.insert(0,dummy_task)

        # initiate results arrays
        # t_imgs : list[Interval]             = [task.accessibility for task in schedulable_tasks]
        th_imgs : list[float]               = [np.average((task.slew_angles.left, task.slew_angles.right)) for task in schedulable_tasks]
        rewards : list[float]               = [self.calc_task_reward(task, specs, cross_track_fovs, orbitdata, observation_history)
                                                for task in tqdm(schedulable_tasks,leave=False,desc='SATELLITE: Calculating task rewards')]
        cumulative_rewards : list[float]    = [0.0 for _ in schedulable_tasks]
        preceeding_observations : list[int] = [np.NAN for _ in schedulable_tasks]

        # initialize observation actions list
        observation_actions : list[ObservationAction] = [None for _ in schedulable_tasks]
        
        # populate observation action list 
        for i in tqdm(range(len(schedulable_tasks)), 
                        desc=f'{state.agent_name}-PLANNER: Generating Observation Actions from Tasks',
                        leave=False):
            # get task i
            task_i : SchedulableObservationTask = schedulable_tasks[i]

            # collect all targets and objectives
            targets_i = [[target[0], target[1], 0.0] 
                         for parent_task in task_i.parent_tasks
                         for target in parent_task.targets]
            objectives = [parent_task.objective for parent_task in task_i.parent_tasks]
            th_i = th_imgs[i]

            # estimate observation action for task i
            observation_i = ObservationAction(task_i.instrument_name,
                                              targets_i,
                                              objectives,
                                              th_i,
                                              task_i.accessibility.left,
                                              task_i.accessibility.span(),
                                              )

            # update observation action list
            observation_actions[i] = observation_i if observation_actions[i] is None else observation_actions[i]

        # initialize adjancency matrix     
        adjacency = [ [False for _ in schedulable_tasks] for _ in schedulable_tasks]

        # populate adjacency matrix 
        for i in tqdm(range(len(schedulable_tasks)), 
                        desc=f'{state.agent_name}-PLANNER: Generating Adjacency Matrix',
                        leave=False):
            for j in range(i + 1, len(schedulable_tasks)):
                # update adjacency matrix for sequence i->j
                adjacency[i][j] = self.is_observation_path_valid(state, specs, [observation_actions[i], observation_actions[j]])

                # check if observation j starts after observation i ends
                if observation_actions[i].t_end < observation_actions[j].t_start:
                    # observation sequence j->i cannot be performed; skip
                    continue

                # update adjacency matrix for sequence j->i
                adjacency[j][i] = self.is_observation_path_valid(state, specs, [observation_actions[j], observation_actions[i]])

        # calculate optimal path and update results
        for j in tqdm(range(len(schedulable_tasks)), 
                      desc=f'{state.agent_name}-PLANNER: Calculating Optimal Path',
                      leave=False):
            # get any possibly prior observation            
            prev_opportunities : list[tuple] = [i
                                                for i in range(0,j)
                                                if adjacency[i][j]]
            # there are no previous possible observations; skip
            if not prev_opportunities: 
                continue

            # update cummulative reward
            for i in prev_opportunities:
                # update cumulative reward
                if cumulative_rewards[i] + rewards[j] > cumulative_rewards[j]:
                    cumulative_rewards[j] = cumulative_rewards[i] + rewards[j]
                    preceeding_observations[j] = i
            
        # get task with highest cummulative reward
        best_task_index = self.argmax(cumulative_rewards)

        # extract sequence of observations from results
        visited_observation_opportunities = set()
        observation_sequence = [best_task_index] if preceeding_observations else []

        while (preceeding_observations 
               and not np.isnan(preceeding_observations[observation_sequence[-1]])):
            prev_observation_index = preceeding_observations[observation_sequence[-1]]
            
            if prev_observation_index in visited_observation_opportunities:
                raise AssertionError('invalid sequence of observations generated by DP. Cycle detected.')
            
            visited_observation_opportunities.add(prev_observation_index)
            observation_sequence.append(prev_observation_index)

        assert 0 in observation_sequence, 'Invalid observation sequence generated by DP. No starting point found.'
        
        # reverse the sequence to start from the first observation
        observation_sequence.reverse()

        # remove dummy observation from sequence
        observation_sequence.remove(0)

        # get matching observation actions
        observations : list[ObservationAction] = [observation_actions[j] for j in observation_sequence]

        # check if all observation was scheduled are valid
        assert all([obs is not None for obs in observations]), 'Invalid observation sequence generated by DP. No observations scheduled.'

        # sort by start time
        observations.sort(key=lambda o: o.t_start)      

        # return observations
        return observations  


    def argmax(self, values, rel_tol=1e-9, abs_tol=0.0):
        """ returns the index of the highest value in a list of values """
        max_val = max(values)
        for i, val in enumerate(values):
            if math.isclose(val, max_val, rel_tol=rel_tol, abs_tol=abs_tol):
                return i        

    @runtime_tracker
    def populate_adjacency_matrix(self, 
                                  state : SimulationAgentState, 
                                  specs : object,
                                  access_opportunities : list, 
                                  ground_points : dict,
                                  adjacency : list,
                                  j : int
                                  ):
               
        # get current observation
        curr_opportunity : tuple = access_opportunities[j]
        lat_curr,lon_curr = ground_points[curr_opportunity[0]][curr_opportunity[1]]
        curr_target = [lat_curr,lon_curr,0.0]

        # get any possibly prior observation
        prev_opportunities : list[tuple] = [prev_opportunity 
                                            for prev_opportunity in access_opportunities[:j]
                                            if prev_opportunity[3].start <= curr_opportunity[3].end
                                            and prev_opportunity != curr_opportunity
                                            ]
        
        prev_opportunities_opt : list[tuple] = [prev_opportunity 
                                            for prev_opportunity in access_opportunities[:j]
                                            ]
        
        assert len(prev_opportunities_opt) == len(prev_opportunities) and all([opp in prev_opportunities for opp in prev_opportunities_opt])
 

        # construct adjacency matrix
        for prev_opportunity in prev_opportunities:
            prev_opportunity : tuple

            # get previous observation opportunity's target 
            lat_prev,lon_prev = ground_points[prev_opportunity[0]][prev_opportunity[1]]
            prev_target = [lat_prev,lon_prev,0.0]

            # assume earliest observation time from previous observation
            earliest_prev_observation = ObservationAction(prev_opportunity[2], 
                                                        prev_target, 
                                                        prev_opportunity[5][0], 
                                                        prev_opportunity[4][0])
            
            # check if observation can be reached from THE previous observation
            adjacent = self.is_observation_path_valid(state, specs, 
                                                          [earliest_prev_observation,
                                                          ObservationAction(curr_opportunity[2], 
                                                                            curr_target, 
                                                                            curr_opportunity[5][-1], 
                                                                            curr_opportunity[4][-1])])     

            # update adjacency matrix
            adjacency[access_opportunities.index(prev_opportunity)][j] = adjacent

    @runtime_tracker
    def _schedule_broadcasts(self, state: SimulationAgentState, observations: list, orbitdata: OrbitData) -> list:
        # schedule measurement requests
        broadcasts : list = super()._schedule_broadcasts(state, observations, orbitdata)

        # # set earliest broadcast time to end of replanning period
        # t_broadcast = self.plan.t+self.period
        
        # # gather observation plan to be sent out
        # plan_out = [action.to_dict() for action in self.pending_observations_to_broadcast]

        # # check if observations exist in plan
        # if plan_out or self.sharing:
        #     # find best path for broadcasts
        #     path, t_start = self._create_broadcast_path(state, orbitdata, t_broadcast)

        #     # check feasibility of path found
        #     if t_start >= 0:
                
        #         # add performing action broadcast to plan
        #         for action_dict in tqdm(plan_out, 
        #                                 desc=f'{state.agent_name}-PLANNER: Pre-Scheduling Broadcasts', 
        #                                 leave=False):
        #             # update action dict to indicate completion
        #             action_dict['status'] = AgentAction.COMPLETED
                    
        #             # create message
        #             msg = ObservationPerformedMessage(state.agent_name, state.agent_name, action_dict, path=path)

        #             # add message broadcast to plan
        #             broadcasts.append(BroadcastMessageAction(msg.to_dict(),t_start))
            
        return broadcasts