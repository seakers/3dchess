from collections import defaultdict
from itertools import repeat
from logging import Logger
import math
from typing import List, Dict
from tqdm import tqdm

from orbitpy.util import Spacecraft

from dmas.clocks import ClockConfig
from dmas.utils import runtime_tracker
from dmas.clocks import *

from chess3d.agents.planning.periodic import AbstractPeriodicPlanner
from chess3d.agents.planning.tasks import SpecificObservationTask
from chess3d.agents.planning.tracker import ObservationHistory
from chess3d.agents.states import *
from chess3d.agents.actions import *
from chess3d.agents.science.requests import *
from chess3d.agents.states import SimulationAgentState
from chess3d.mission.mission import Mission
from chess3d.orbitdata import OrbitData
from chess3d.messages import *
from chess3d.utils import Interval, argmax

class DynamicProgrammingPlanner(AbstractPeriodicPlanner):
    # models
    DISCRETE = 'discrete'
    CONTINUOUS = 'continuous'

    # sharing modes #TODO move to parent class later
    PERIODIC = 'periodic'
    OPPORTUNISTIC = 'opportunistic'

    def __init__(self, 
                 horizon: float, 
                 period : float, 
                 model : str = CONTINUOUS,
                 sharing : str = None,
                 debug : bool = False,
                 logger: Logger = None
                 ) -> None:
        super().__init__(horizon, 
                         period, 
                         debug, 
                         logger)
        
        # validate inputs
        assert model in [self.DISCRETE, self.CONTINUOUS], f'Invalid `model` type `{model}`. Must be one of {[self.DISCRETE, self.CONTINUOUS]}.'
        assert sharing in [None, self.PERIODIC, self.OPPORTUNISTIC], f'Invalid `sharing` type `{sharing}`. Must be one of {[None, self.PERIODIC, self.OPPORTUNISTIC]}.'

        # set planner parameters
        self.model = model
        self.sharing = sharing
        
    @runtime_tracker
    def _schedule_observations(self, 
                               state : SimulationAgentState, 
                               specs : object, 
                               _ : ClockConfig, 
                               orbitdata : OrbitData, 
                               schedulable_tasks : list,
                               mission : Mission,
                               observation_history : ObservationHistory
                               ) -> List[ObservationAction]:
        
        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Naive planner not yet implemented for agents of type `{type(state)}.`')
        elif not isinstance(specs, Spacecraft):
            raise ValueError(f'`specs` needs to be of type `{Spacecraft}` for agents with states of type `{SatelliteAgentState}`')
        
        # compile list of instruments available in payload
        payload : dict = {instrument.name: instrument for instrument in specs.instrument}
        
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self._collect_fov_specs(specs)
        
        # get pointing agility specifications
        max_slew_rate, max_torque = self._collect_agility_specs(specs)
        assert max_slew_rate, 'ADCS `maxRate` specification missing from agent specs object.'
        assert max_torque, 'ADCS `maxTorque` specification missing from agent specs object.'

        # sort tasks by start time
        schedulable_tasks : list[SpecificObservationTask] = sorted(schedulable_tasks, key=lambda t: (t.accessibility.left, -t.get_priority(), -len(t.parent_tasks)))

        # call appropriate model to generate observation schedule
        if self.model == self.DISCRETE:
            observations = self.__discrete_model(state, 
                                         schedulable_tasks, 
                                         orbitdata, 
                                         mission, 
                                         observation_history, 
                                         specs, 
                                         payload,
                                         cross_track_fovs, 
                                         max_slew_rate, 
                                         max_torque)
        
        elif self.model == self.CONTINUOUS:
            observations = self.__continuous_model(state, 
                                          schedulable_tasks, 
                                          orbitdata, 
                                          mission, 
                                          observation_history, 
                                          specs, 
                                          payload,
                                          cross_track_fovs, 
                                          max_slew_rate, 
                                          max_torque)
        else:
            raise ValueError(f'Invalid `model` type `{self.model}`. Must be one of {[self.DISCRETE, self.CONTINUOUS]}.')

        # check if all observation was scheduled are valid
        assert all([obs is not None for obs in observations]), 'Invalid observation sequence generated by DP. No observations scheduled.'

        # sort by start time
        observations.sort(key=lambda o: o.t_start) 

        # extract sequence of observations
        return observations

    def __discrete_model(self, 
                         state : SatelliteAgentState, 
                         schedulable_tasks : List[SpecificObservationTask], 
                         orbitdata : OrbitData,
                         mission : Mission, 
                         observation_history : ObservationHistory,
                         specs : Spacecraft, 
                         payload : dict,
                         cross_track_fovs : Dict[str, float],
                         max_slew_rate : float, 
                         max_torque : float,
                        ) -> List[ObservationAction]:
        """ schedules observations using a discrete-time dynamic programming approach """
        # remove dummy task from schedulable tasks #TODO needed?
        # schedulable_tasks.pop(0)

        # add dummy task to represent initial state
        instrument_names = list(payload.keys())
        dummy_task = SpecificObservationTask(set([]), instrument_names[0], Interval(state.t,state.t), 0.0, Interval(state.attitude[0],state.attitude[0]))
        schedulable_tasks.insert(0,dummy_task)

        
        # initiate constants
        d_imgs : list[float]        = [task.min_duration for task in schedulable_tasks]
        th_imgs : list[float]       = [np.average((task.slew_angles.left, task.slew_angles.right)) for task in schedulable_tasks]
        slew_times : list[float]    = [[abs(th_imgs[i] - th_imgs[j]) / max_slew_rate if max_slew_rate else np.Inf
                                        for j,_ in enumerate(schedulable_tasks)]
                                        for i,_ in enumerate(schedulable_tasks)
                                        ]        

        # create time discretization pairs
        task_pairs : list[tuple] = [] # of the form (task_index, t_img, reward)
        for i,task_i in tqdm(enumerate(schedulable_tasks), desc=f'{state.agent_name}-PLANNER: Generating Task-Time Pairs', leave=False):
            # set initial imaging time
            t_img = task_i.accessibility.left

            # iterate through all possible imaging times for task i with time step of orbitdata
            while t_img + task_i.min_duration <= min(task_i.accessibility.right, state.t + self.horizon):
                # add pair to list                
                task_pairs.append((i, t_img))

                # update imaging time
                t_img += orbitdata.time_step
        
        # sort pairs by start time, task index
        task_pairs.sort(key=lambda x: (x[1], x[0]))

        # initiate results lists
        rewards : list[float]               = [np.NAN for _ in task_pairs]  
        cumulative_rewards : list[float]    = [0.0 for _ in task_pairs]
        preceeding_observations : list[int] = [np.NAN for _ in task_pairs]

        # initialize adjacency matrix
        adjacency_dict : Dict[tuple,list] = defaultdict(list)

        # populate adjacency matrix
        for j,pair_j in tqdm(enumerate(task_pairs), 
                                desc=f'{state.agent_name}-PLANNER: Generating Adjacency Matrix',
                                leave=False,
                                total=len(task_pairs)
                                ):
            
            # skip first task-time pair
            if i == 0: continue # dummy task has no preceeding tasks

            # unpack task-time pair j
            idx_j,t_img_j = pair_j
            task_j : SpecificObservationTask = schedulable_tasks[idx_j]

            # find pairs that can preceed j
            idx_prev = max(0, j-1)
            preceeding_task_pairs = [(i,(idx_i,t_img_i) )
                                   for i,(idx_i,t_img_i) in enumerate(task_pairs[:idx_prev])
                                   if i < j
                                   and idx_i != idx_j
                                   and not task_j.is_mutually_exclusive(schedulable_tasks[idx_i])
                                   and t_img_i + d_imgs[idx_i] + slew_times[idx_i][idx_j] <= t_img_j
                                ]   
            
            # add to adjacency dict
            if preceeding_task_pairs: adjacency_dict[(j,pair_j)] = preceeding_task_pairs
            
        # TODO ensure acyclic graph
        # for (j,pair_j),preceeding_pairs in adjacency_dict.items():
        #     for i,pair_i in preceeding_pairs:
        #         assert (i,pair_i) not in adjacency_dict or all(j2 > j for j2,_ in adjacency_dict[(i,pair_i)]), \
        #             'invalid sequence of observations generated by DP. Cycle detected.'
        #     adjacency_dict[(j,pair_j)] = [pair for pair in preceeding_pairs if pair[0] < j]

        # calculate optimal path and update results
        for j,pair_j in tqdm(enumerate(task_pairs), 
                              desc=f'{state.agent_name}-PLANNER: Evaluating Path Reward',
                              leave=False,
                              total=len(task_pairs)
                              ):
            # unpack task-time pair j
            idx_j,t_img_j = pair_j
            task_j : SpecificObservationTask = schedulable_tasks[idx_j]

            # get feasible next task-time pairs
            preceeding_pairs = adjacency_dict.get((j,pair_j), [])

            # update cumulative rewards
            for i,_ in preceeding_pairs:

                # reconstruct path leading to i
                path_i = self.__get_path_to_index(preceeding_observations, i)
            
                # check if new task j conflicts with any in path leading to i
                if any(schedulable_tasks[task_pairs[k][0]].is_mutually_exclusive(task_j) for k in path_i):
                    continue  # skip this candidate extension
                
                # estimate task value of task j if done after i if it hasn't been estimated yet
                if np.isnan(rewards[j]):
                    rewards[j] = self.estimate_task_value(task_j, 
                                                        t_img_j, 
                                                        d_imgs[idx_j], 
                                                        specs, 
                                                        cross_track_fovs, 
                                                        orbitdata, 
                                                        mission, 
                                                        observation_history)

                # calculate cumulative reward for path i->j
                if cumulative_rewards[i] + rewards[j] > cumulative_rewards[j]:
                    # update cumulative reward
                    cumulative_rewards[j] = cumulative_rewards[i] + rewards[j]

                    # update preceeding observation
                    preceeding_observations[j] = i

        # extract sequence of observations from results
        observation_sequence : List[int] = self.__extract_observation_sequence(preceeding_observations, cumulative_rewards)

        # get matching observation actions
        observations : list[ObservationAction] = [ObservationAction(schedulable_tasks[task_pairs[k][0]].instrument_name,
                                                                    th_imgs[task_pairs[k][0]],
                                                                    task_pairs[k][1],
                                                                    d_imgs[task_pairs[k][0]],
                                                                    schedulable_tasks[task_pairs[k][0]]
                                                                    )
                                                  for k in observation_sequence]

        # return observations
        return observations  
    
    def __continuous_model(self, 
                           state : SatelliteAgentState, 
                           schedulable_tasks : List[SpecificObservationTask], 
                           orbitdata : OrbitData,
                           mission : Mission, 
                           observation_history : ObservationHistory,
                           specs : Spacecraft, 
                           payload : dict,
                           cross_track_fovs : Dict[str, float],
                           max_slew_rate : float, 
                           max_torque : float,
                        ) -> List[ObservationAction]:
        """ schedules observations using a continuous-time dynamic programming approach """
        # add dummy task to represent initial state
        instrument_names = list(payload.keys())
        dummy_task = SpecificObservationTask(set([]), instrument_names[0], Interval(state.t,state.t), 0.0, Interval(state.attitude[0],state.attitude[0]))
        schedulable_tasks.insert(0,dummy_task)
        
        # initiate results arrays
        t_imgs : list[Interval]             = [max(task.accessibility.left, state.t) for task in schedulable_tasks]
        d_imgs : list[float]                = [task.min_duration for task in schedulable_tasks]
        th_imgs : list[float]               = [np.average((task.slew_angles.left, task.slew_angles.right)) for task in schedulable_tasks]
        rewards : list[float]               = [0.0 for _ in schedulable_tasks]
        cumulative_rewards : list[float]    = [0.0 for _ in schedulable_tasks]
        preceeding_observations : list[int] = [np.NAN for _ in schedulable_tasks]
        slew_times : list[float]            = [[abs(th_imgs[i] - th_imgs[j]) / max_slew_rate if max_slew_rate else np.Inf
                                                for j,_ in enumerate(schedulable_tasks)]
                                               for i,_ in enumerate(schedulable_tasks)
                                               ]

        # initialize observation actions list
        earliest_observation_actions : list[ObservationAction] = [None for _ in schedulable_tasks]
        latest_observation_actions : list[ObservationAction] = [None for _ in schedulable_tasks]
        
        # populate observation action list 
        for i, task_i in tqdm(enumerate(schedulable_tasks), 
                                desc=f'{state.agent_name}-PLANNER: Generating Observation Actions from Tasks',
                                leave=False):
            
            # collect all targets and objectives
            th_i = th_imgs[i]
            d_imgs_i = d_imgs[i]

            # estimate observation action for task i
            earliest_observation_i = ObservationAction(task_i.instrument_name,
                                                        th_i,
                                                        task_i.accessibility.left,
                                                        d_imgs_i,
                                                        task_i
                                                        )
            latest_observation_i = ObservationAction(task_i.instrument_name,
                                                        th_i,
                                                        task_i.accessibility.right-d_imgs_i,
                                                        d_imgs_i,
                                                        task_i
                                                        )

            # update observation action list
            earliest_observation_actions[i] = earliest_observation_i
            latest_observation_actions[i] = latest_observation_i

        # initialize adjacency matrix
        adjacency = [[False for _ in schedulable_tasks] for _ in schedulable_tasks]

        # populate adjacency matrix 
        for i in tqdm(range(len(schedulable_tasks)), 
                        desc=f'{state.agent_name}-PLANNER: Generating Adjacency Matrix',
                        leave=False):
            for j in range(i + 1, len(schedulable_tasks)):
                # check mutual exclusivity
                if schedulable_tasks[i].is_mutually_exclusive(schedulable_tasks[j]):
                    # tasks i and j are mutually exclusive, cannot perform sequence i->j
                    continue

                # update adjacency matrix for sequence i->j
                adjacency[i][j] = self.is_observation_path_valid(state, [earliest_observation_actions[i], latest_observation_actions[j]], max_slew_rate, max_torque)

                if adjacency[i][j]:
                    # if sequence i->j is valid; enforce acyclical behavior and skip j->i
                    continue
                
                elif latest_observation_actions[i].t_end < earliest_observation_actions[j].t_start:
                    # observation sequence j->i cannot be performed; skip
                    continue
                
                # update adjacency matrix for sequence j->i
                adjacency[j][i] = self.is_observation_path_valid(state, [earliest_observation_actions[j], latest_observation_actions[i]], max_slew_rate, max_torque)

        # calculate optimal path and update results
        for j in tqdm(range(len(schedulable_tasks)), 
                      desc=f'{state.agent_name}-PLANNER: Evaluating Path Reward',
                      leave=False):
            # get indeces of possible prior observations
            prev_indices : list[int] = [i for i in range(0,j) if adjacency[i][j]]

            # update cumulative reward
            for i in prev_indices:
                # reconstruct path leading to i
                path_i = self.__get_path_to_index(preceeding_observations, i)

                # check if new task j conflicts with any in path leading to i
                if any(schedulable_tasks[j].is_mutually_exclusive(schedulable_tasks[k]) for k in path_i):
                    continue  # skip this candidate extension
                
                # calculate earliest imaging time for task j assuming task i is done before
                t_img_j = max(t_imgs[i] + d_imgs[i] + slew_times[i][j], t_imgs[j]) 

                # check if imaging time is valid
                if not (t_img_j <= state.t + self.horizon                           # imaging start time within planning horizon
                    and t_img_j in schedulable_tasks[j].accessibility               # imaging start time within task availability
                    and t_img_j + d_imgs[j] <= state.t + self.horizon               # imaging end time within planning horizon
                    and t_img_j + d_imgs[j] in schedulable_tasks[j].accessibility): # imaging end time within task availability
                    continue

                # estimate task value of task j if done after i
                reward_j = self.estimate_task_value(schedulable_tasks[j], 
                                                    t_img_j, 
                                                    d_imgs[j], 
                                                    specs, 
                                                    cross_track_fovs, 
                                                    orbitdata, 
                                                    mission, 
                                                    observation_history)

                # compare cumulative rewards
                if cumulative_rewards[i] + reward_j > cumulative_rewards[j]:
                    # update imaging time
                    t_imgs[j] = t_img_j

                    # update individual task reward
                    rewards[j] = reward_j

                    # update cumulative reward
                    cumulative_rewards[j] = cumulative_rewards[i] + rewards[j]
                    
                    # update preceeding observation 
                    preceeding_observations[j] = i
            
        # extract sequence of observations from results
        observation_sequence : List[int] = self.__extract_observation_sequence(preceeding_observations, cumulative_rewards)

        # get matching observation actions
        observations : list[ObservationAction] = [ObservationAction(schedulable_tasks[j].instrument_name,
                                                                    th_imgs[j],
                                                                    t_imgs[j],
                                                                    d_imgs[j],
                                                                    schedulable_tasks[j]
                                                                    )
                                                  for j in observation_sequence]
          
        # return observations
        return observations  

    def __get_path_to_index(self, preceeding_observations : list, index : int) -> List[int]:
        """ retrieves path of observation indices leading to given index """        
        # initialize path
        path = [index]

        if np.isnan(preceeding_observations[index]): return path

        # backtrack to get full path
        k = index
        while not np.isnan(preceeding_observations[k]):
            # get preceeding observation
            k = preceeding_observations[k]

            # validate no cycles
            assert k not in path, \
                'invalid sequence of observations generated by DP. Cycle detected.'

            # add to path
            path.append(int(k))

        # return path in correct order
        return list(reversed(path))

    def __extract_observation_sequence(self, preceeding_observations : list, cumulative_rewards : list) -> List[int]:
        """ extracts observation index sequence from DP results """
        # get task with highest cummulative reward
        best_task_index = argmax(cumulative_rewards)

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

        # return observation sequence
        return observation_sequence

    # @runtime_tracker
    # def populate_adjacency_matrix(self, 
    #                               state : SimulationAgentState, 
    #                               specs : object,
    #                               access_opportunities : list, 
    #                               ground_points : dict,
    #                               adjacency : list,
    #                               j : int
    #                               ):
               
    #     # get satellite agility specifications
    #     max_slew_rate, max_torque = self._collect_agility_specs(specs)
    #     assert max_slew_rate, 'ADCS `maxRate` specification missing from agent specs object.'
    #     assert max_torque, 'ADCS `maxTorque` specification missing from agent specs object.'

    #     # get current observation
    #     curr_opportunity : tuple = access_opportunities[j]
    #     lat_curr,lon_curr = ground_points[curr_opportunity[0]][curr_opportunity[1]]
    #     curr_target = [lat_curr,lon_curr,0.0]

    #     # get any possibly prior observation
    #     prev_opportunities : list[tuple] = [prev_opportunity 
    #                                         for prev_opportunity in access_opportunities[:j]
    #                                         if prev_opportunity[3].start <= curr_opportunity[3].end
    #                                         and prev_opportunity != curr_opportunity
    #                                         ]
        
    #     prev_opportunities_opt : list[tuple] = [prev_opportunity 
    #                                         for prev_opportunity in access_opportunities[:j]
    #                                         ]
        
    #     assert len(prev_opportunities_opt) == len(prev_opportunities) and all([opp in prev_opportunities for opp in prev_opportunities_opt])
 
    #     # construct adjacency matrix
    #     for prev_opportunity in prev_opportunities:
    #         prev_opportunity : tuple

    #         # get previous observation opportunity's target 
    #         lat_prev,lon_prev = ground_points[prev_opportunity[0]][prev_opportunity[1]]
    #         prev_target = [lat_prev,lon_prev,0.0]

    #         # assume earliest observation time from previous observation
    #         earliest_prev_observation = ObservationAction(prev_opportunity[2], 
    #                                                     prev_target, 
    #                                                     prev_opportunity[5][0], 
    #                                                     prev_opportunity[4][0])
            
    #         # check if observation can be reached from THE previous observation
    #         adjacent = self.is_observation_path_valid(state,  
    #                                                       [earliest_prev_observation,
    #                                                       ObservationAction(curr_opportunity[2], 
    #                                                                         curr_target, 
    #                                                                         curr_opportunity[5][-1], 
    #                                                                         curr_opportunity[4][-1])],
    #                                                     max_slew_rate,
    #                                                     max_torque)     
    #         mutually_exclusive = prev_opportunity[2].is_mutually_exclusive(curr_opportunity[2])

    #         # update adjacency matrix
    #         adjacency[access_opportunities.index(prev_opportunity)][j] = adjacent and not mutually_exclusive

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