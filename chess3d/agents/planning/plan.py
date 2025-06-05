from abc import ABC
import copy
import uuid
import numpy as np
from tqdm import tqdm

from chess3d.agents.actions import *

class Plan(ABC):
    """ Describes a plan to be performed by an agent """
    
    def __init__(   self, 
                    *actions,  
                    t : float = 0.0,
                    t_next : float = np.Inf,
                ) -> None:
        # initialize values
        self.t = t
        self.t_next = t_next
        self.actions : list[AgentAction] = []
        
        # load preplan
        if actions: self.update(*actions, t=t)
            
    def counters(self) -> dict:
        counts = {}
        for action in self.actions:
            action : AgentAction

            if type(action) not in counts:
                counts[type(action)] = 0
            
            counts[type(action)] += 1
        
        return counts
    
    def empty(self, t : float) -> None:
        """ rmeoves all actions from plan """
        self.actions : list[AgentAction] = []
        self.t = t

    def remove(self, action : AgentAction, t : float) -> None:
        """ removes action from plan """
        if action in self.actions: 
            self.actions.remove(action)
            self.t = t

    def update(self, *action_lists, t : float) -> None:
        """ Updates the current plan to a new list of actions """
        
        # reset current plan
        self.actions = []

        # add actions from iterable set of actions
        for actions in tqdm(action_lists, desc='Adding action lists to plan', leave=False):
            # check argument types
            if not isinstance(actions, list):
                raise ValueError(f'updated plan must be of type `list`.')
            
            # check feasiblity
            try:
                self.__is_feasible(actions)
            except RuntimeError:
                raise RuntimeError("Cannot update plan: new plan is unfeasible.")
        
            # update plan
            self.add_all(actions, t)

        # update plan update time
        self.t = t            

        # check feasibility of new plan
        assert self.__is_feasible(self.actions)
        
    def add_all(self, actions : list, t : float) -> None:
        """ adds a set of actions to plan """
        
        # sort new set of actions by start time 
        actions.sort(key= lambda a : a.t_start)

        # if there are no actions in current plan
        if not self.actions:
            self.actions.extend(actions)
        
        else:
            # create preliminary plan
            prelim_plan = []

            curr_actions = [action for action in self.actions]
            new_actions = [action for action in actions]

            with tqdm(total=len(curr_actions) + len(new_actions),
                      desc='Planner-Adding actions to plan',
                      leave=False) as pbar:
                while curr_actions and new_actions:
                    curr_action : AgentAction = curr_actions[0]
                    new_action : AgentAction = new_actions[0]

                    if curr_action.t_start < new_action.t_start:
                        prelim_plan.append(curr_actions.pop(0))
                        pbar.update(1)
                    
                    elif curr_action.t_start > new_action.t_start:
                        prelim_plan.append(new_actions.pop(0))
                        pbar.update(1)

                    elif abs(curr_action.t_end - curr_action.t_start) <= 1e-3:
                        prelim_plan.append(curr_actions.pop(0))
                        pbar.update(1)
                    
                    elif abs(new_action.t_end - new_action.t_start) <= 1e-3:
                        prelim_plan.append(new_actions.pop(0))
                        pbar.update(1)

                if curr_actions: 
                    prelim_plan.extend(curr_actions)
                    pbar.update(len(curr_actions))
                if new_actions: 
                    prelim_plan.extend(new_actions)
                    pbar.update(len(new_actions))

            # check feasibility
            try:
                self.__is_feasible(prelim_plan)
                feasible = True
            except ValueError:
                feasible = False
            
            if feasible:
                # is feasible, no need to repair 
                self.actions = [action for action in prelim_plan]
                
            else:
                # possible conflicts exist, may need to repair 
                for action in actions: self.add(action, t)
        
    def add(self, action : AgentAction, t : float) -> None:
        """ adds action to plan """

        if action.t_end < t:
            x =1

        # check argument types
        if not isinstance(action, AgentAction):
            raise ValueError(f"Cannot place action of type `{type(action)}` in plan. Must be of type `{AgentAction}`.")

        try:
            # check if action is scheduled to occur during while another action is being performed
            interrupted_actions = [interrupted_action 
                                   for interrupted_action in self.actions
                                   if isinstance(interrupted_action, AgentAction)
                                   and interrupted_action.t_start < action.t_start < interrupted_action.t_end]
            
            # check if another action is schduled during this action
            concurrent_actions = [concurrent_action 
                                  for concurrent_action in self.actions
                                  if isinstance(concurrent_action, AgentAction)
                                  and action.t_start < concurrent_action.t_start
                                  and concurrent_action.t_end < action.t_end
                                  ]
            concurrent_actions.sort(key=lambda a : a.t_start)

            # check if action occurs after another action was completed 
            previous_actions = [completed_action 
                                for completed_action in self.actions
                                if completed_action.t_end <= action.t_start]

            if interrupted_actions:
                earliest_interrupted_action : AgentAction = interrupted_actions.pop(0)
                if abs(earliest_interrupted_action.t_end - earliest_interrupted_action.t_start) < 1e-6:
                    # interrupted action has no duration, schedule task for right after
                    self.actions.insert(self.actions.index(earliest_interrupted_action) + 1, action)
                    
                else:
                    # interrupted action has a non-zero duration; split interrupted action into two 
                    i_plan = self.actions.index(earliest_interrupted_action)

                    # create duplciate of interrupted action with a new ID
                    continued_action : AgentAction = action_from_dict(**earliest_interrupted_action.to_dict())
                    continued_action.t_end = earliest_interrupted_action.t_end
                    continued_action.id = str(uuid.uuid1())

                    # modify interrupted action
                    if isinstance(earliest_interrupted_action, TravelAction):
                        ## change start and end positions TODO
                        pass

                    ## change start and end times for the interrupted and continued actions
                    earliest_interrupted_action.t_end = action.t_start
                    continued_action.t_start = action.t_end

                    # place action in between the two split parts
                    self.actions[i_plan] = earliest_interrupted_action
                    self.actions.insert(i_plan + 1, continued_action)
                    self.actions.insert(i_plan + 1, action)

            elif concurrent_actions:
                # split action between concurent actions
                for concurrent_action in concurrent_actions:
                    if type(concurrent_action) == type(action):
                        raise RuntimeError(f"Another action of the same type is being performed concurrently to another.")
                    elif abs(concurrent_action.t_end - concurrent_action.t_start) > 1e-6:
                        raise RuntimeError(f"Another action of non-zero is being performed concurrently to another.")

                    # create a copy of the action and assign it to start after concurrent action
                    shortened_action : AgentAction = copy.copy(action)
                    shortened_action.t_end = concurrent_action.t_start

                    # check if new action has a non-zero time duration
                    if abs(shortened_action.t_end - shortened_action.t_start) >= 1e-6:   
                        # update current action's end time
                        action.t_start = concurrent_action.t_end
                        
                        # add new action to plan
                        self.actions.insert(self.actions.index(concurrent_action), shortened_action)
                    else:
                        x = 1

                # check if there is still time left in the action
                if abs(action.t_end - action.t_start) >= 1e-6:   
                    # place action after last concurrent_action 
                    latest_action : AgentAction = concurrent_actions[-1]
                    self.actions.insert(self.actions.index(latest_action) + 1, action) 
                else:
                    # action is too short after splitting; do not add to plan
                    pass
            
            elif previous_actions:
                # place action after latest action ends
                latest_action : AgentAction = previous_actions[-1]
                self.actions.insert(self.actions.index(latest_action) + 1, action)

            else:
                # place action at the start of the plan
                self.actions.insert(0, action)

        finally:
            # sort actions chronollogically 
            self.actions.sort(key=lambda a : a.t_start)
            
            # update latest update time
            self.t = t  

            try:
                # check plan feasibility
                self.__is_feasible(self.actions)
            except ValueError as e:
                # self.__is_feasible(self.actions)
                raise RuntimeError(f"Cannot place action in plan. {e} \n {str(self)}\ncurrent plan:\n{str(self)}")

    def update_action_completion(   self, 
                                    completed_actions : list, 
                                    aborted_actions : list, 
                                    pending_actions : list,
                                    t : float
                                ) -> None:
        """ 
        
        Checks if any of the actions completed by the agent were in the current plan. 
        If so, they are removed.
        
        """
        # compile performed actions
        performed_actions : list[AgentAction] = [action for action in completed_actions]
        performed_actions.extend(aborted_actions)

        # remove performed actions from plan
        for performed_action in performed_actions:
            matching_actions = [action for action in self.actions
                                if performed_action.id == action.id]
            for action in matching_actions: self.actions.remove(action)

        # removed expired actions
        expired_actions = [action for action in self.actions
                           if action.t_end < t]
        for action in expired_actions: self.actions.remove(action)

    def get_next_actions(self, t : float) -> list:
        """ returns a list of dicts """

        # get next available action to perform
        eps = 1e-6
        plan_out : list[AgentAction] = [action
                                        for action in self.actions 
                                        if action.t_start - eps <= t <= action.t_end + eps]
        
        # sort plan in order of ascending start time 
        plan_out.sort(key=lambda a: (a.t_start, a.t_end))

        # check if actions are contained in output plan
        if plan_out:
            # there are actions in output plan; return plan
            return plan_out
        else:
            # no actions in output plan; wait for future actions
            t_idle = self.actions[0].t_start if not self.is_empty() else self.t_next
            if t > t_idle:
                x =1
            return [WaitForMessages(t, t_idle)]

        # if plan_out:
        #     plan_out = [action.to_dict()
        #                 for action in plan_out]
        # else:
        #     # idle if no more actions can be performed at this time
        #     t_idle = self.actions[0].t_start if not self.is_empty() else self.t_next
        #     action = WaitForMessages(t, t_idle)
        #     plan_out.append(action.to_dict())     
        
        # if (len(plan_out) > 1 and 
        #     any([action_out['t_start'] == action_out['t_end'] for action_out in plan_out])):
        #     plan_out = [action_out for action_out in plan_out
                    #    if action_out['t_start'] == action_out['t_end']]
        
        # return plan_out    
    
    def copy(self) -> object:
        """ Copy contructor. Creates a deep copy of this oject. """
        return Plan(self.actions, t=self.t, t_next=self.t_next)

    def __is_feasible(self, plan : list) -> bool:
        """ Checks if the current plan can be performed by the agent """
        
        # initialize previous action's start and end times
        t_start_prev, t_end_prev = None, None

        for action in plan:
            # get action's start and end times 
            if isinstance(action, AgentAction):
                t_start = action.t_start
                t_end = action.t_end

            elif isinstance(action, dict):
                t_start = action['t_start']
                t_end = action['t_end']

            else:
                raise ValueError(f"Cannot check plan of actions of type {type(action)}")

            # check if there is no overlap between tasks
            if t_start_prev is not None and t_end_prev is not None:
                if t_start_prev > t_start:
                    continue 
                elif t_end_prev > t_start:
                    raise ValueError(f"Plan contains action with start time prior to its previous action's end time.")

            # save current action start and endtimes for comparison with the following action
            t_start_prev = t_start
            t_end_prev = t_end

        return True
    
    def __eq__(self, __value: object) -> bool:
        if len(self.actions) != len(__value.actions):
            return False
        
        for action in self.actions:
            if action not in __value.actions:
                return False
            
        return self.t == __value.t

    def __str__(self) -> str:
        out = f't_plan = {self.t}[s]\n'
        out += f'id\t  action type\tt_start\tt_end\n'

        if self.is_empty():
            out += 'EMPTY\n\n'
        else:
            for action in self.actions:
                if isinstance(action, AgentAction):
                    out += f"{action.id.split('-')[0]}  {action.action_type}\t{round(action.t_start,1)}\t{round(action.t_end,1)}\n"
        out += f'\nn actions in plan: {len(self)}'
        out += f'\nn measurements in plan: {len([action for action in self if isinstance(action, ObservationAction)])}'
        out += f'\nn broadcasts in plan: {len([action for action in self if isinstance(action, BroadcastMessageAction)])}'
        out += f'\nn maneuvers in plan: {len([action for action in self if isinstance(action, ManeuverAction)])}'
        out += f'\nn travel actions in plan: {len([action for action in self if isinstance(action, TravelAction)])}\n'
        return out

    def get_horizon(self) -> float:
        """ Returns current planning horizon """
        return 0.0 if self.is_empty() else self.actions[-1].t_end - self.actions[0].t_start

    def is_empty(self) -> bool:
        """ Checks if the current plan is empty """
        return not bool(self.actions)
    
    def __iter__(self):
        for action in self.actions:
            yield action

    def __len__(self) -> int:
        return len(self.actions)    

class Preplan(Plan):
    def __init__(self, 
                 *actions, 
                 t: float = 0,
                 horizon : float = np.Inf,
                 t_next : float = np.Inf
                 ) -> None:
        
        self.horizon = horizon

        super().__init__(*actions, t=t, t_next=t_next)

    def copy(self) -> object:
        return Preplan(self.actions, t=self.t, horizon=self.horizon, t_next=self.t_next)
    
    def add(self, action: AgentAction, t: float) -> None:
        if self.t + self.horizon < action.t_end:
            raise ValueError(f'cannot add action scheduled to be done past the planning horizon of this plan')

        super().add(action, t)

class Replan(Plan):
    def copy(self) -> object:
        return Replan(self.actions, t=self.t, t_next=self.t_next)
    
    def from_preplan(preplan : Preplan, *actions, t : float) -> object:
        """ creates a modified plan from an existing preplan and a set of new actions to be added to said plan """
        return Replan(preplan.actions, *actions, t=t, t_next=preplan.t_next)
    