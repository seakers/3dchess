from abc import ABC, abstractmethod
import uuid
import numpy as np
from chess3d.nodes.actions import AgentAction
from nodes.actions import *


class Plan(ABC):
    """ Describes a plan to be performed by an agent """
    
    def __init__(   self, 
                    *actions,  
                    t : float = 0.0
                ) -> None:
        # initialize values
        self.t = t
        self.actions = []
        
        # load preplan
        if actions:
            self.update(*actions, t=t)

    def update(self, *action_lists, t : float) -> None:
        """ Updates the current plan to a new list of actions """
        
        # reset current plan
        self.actions = []

        # add actions from iterable set of actions
        for actions in action_lists:
            # check argument types
            if not isinstance(actions, list):
                raise ValueError(f'updated plan must be of type `list`.')
            
            # check feasiblity
            try:
                self.__is_feasible(actions)
            except RuntimeError as e:
                raise RuntimeError("Cannot update plan: new plan is unfeasible.")
        
            # update plan
            self.add_all(actions, t)
                
        # add indefinite wait at the end of the plan
        t_wait_start = t if self.empty() else self.actions[-1].t_end
        if t_wait_start < np.Inf:
            self.add(WaitForMessages(t_wait_start, np.Inf), t)

        # update plan update time
        self.t = t              
        
    def add(self, action : AgentAction, t : float) -> None:
        """ adds action to plan """

        # check argument types
        if not isinstance(action, AgentAction):
            raise ValueError(f"Cannot place action of type `{type(action)}` in plan. Must be of type `{AgentAction}`.")

        try:
            # check if action is scheduled to occur during while another action is being performed
            interrupted_actions = [interrupted_action for interrupted_action in self.actions
                                   if interrupted_action.t_start < action.t_start < interrupted_action.t_end]
            if interrupted_actions:
                interrupted_action : AgentAction = interrupted_actions.pop(0)
                if (    
                        isinstance(interrupted_action, MeasurementAction) 
                    or  isinstance(interrupted_action, BroadcastMessageAction)
                    ):
                    # interrupted action has no duration, schedule broadcast for right after
                    self.actions.insert(self.actions.index(interrupted_action) + 1, action)
                    
                else:
                    # interrupted action has a non-zero duration; split interrupted action into two 
                    i_plan = self.actions.index(interrupted_action)

                    # create duplciate of interrupted action with a new ID
                    continued_action : AgentAction = action_from_dict(**interrupted_action.to_dict())
                    continued_action.id = str(uuid.uuid1())

                    # modify interrupted action
                    if isinstance(interrupted_action, TravelAction):
                        ## change start and end positions TODO
                        pass

                    ## change start and end times for the interrupted and continued actions
                    interrupted_action.t_end = action.t_start
                    continued_action.t_start = action.t_end

                    # place action in between the two split parts
                    self.actions[i_plan] = interrupted_action
                    self.actions.insert(i_plan + 1, continued_action)
                    self.actions.insert(i_plan + 1, action)
                
                return

            ## check if access occurs after an action was completed 
            completed_actions = [completed_action for completed_action in self.actions
                            if completed_action.t_end <= action.t_start]
            
            if completed_actions:
                # place action after action ends
                completed_action : AgentAction = completed_actions.pop()
                self.actions.insert(self.actions.index(completed_action) + 1, action)

            else:
                self.actions.insert(0, action)
            
        finally:
            # update latest update time
            self.t = t  

            try:
                # check plan feasibility
                self.__is_feasible(self.actions)
            except ValueError as e:
                raise RuntimeError(f"Cannot place action in plan. {e} \n {str(self)}\ncurrent plan:\n{str(self)}")

    def add_all(self, actions : list, t : float) -> None:
        """ adds a set of actions to plan """
        for action in actions:
            self.add(action, t)

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
        performed_actions = [action for action in completed_actions]
        performed_actions.extend(aborted_actions)

        # remove performed actions from plan
        for performed_action in performed_actions:
            performed_action : AgentAction
            for action in self.actions:
                action : AgentAction
                if performed_action.id == action.id:
                    self.actions.remove(action)
                    break

    def get_next_actions(self, t : float) -> list:
        """ returns a list of dicts """

        # get next available action to perform
        plan_out = [action.to_dict() 
                    for action in self.actions 
                    if action.t_start <= t <= action.t_end]

        # idle if no more actions can be performed
        if not plan_out:
            t_idle = self.actions[0].t_start if not self.empty() else np.Inf
            action = WaitForMessages(t, t_idle)
            plan_out.append(action.to_dict())     

        # sort plan in order of ascending start time 
        plan_out.sort(key=lambda a: a['t_end'])

        if any([action_out['t_start'] == action_out['t_end']
                for action_out in plan_out]):
            plan_out = [action_out for action_out in plan_out
                       if action_out['t_start'] == action_out['t_end']]

        # check plan feasibility
        try:
            self.__is_feasible(plan_out)
        except ValueError as e:
            raise RuntimeError(f"Unfeasible plan. {e}")

        return plan_out    
    
    def copy(self) -> object:
        """ Copy contructor. Creates a deep copy of this oject. """
        return Plan(self.actions, t=self.t)

    def __is_feasible(self, plan : list) -> bool:
        """ Checks if the current plan can be performed by the agent """

        # check if actions dont overlap
        t_start_prev, t_end_prev = None, None
        for action in plan:
            if isinstance(action, AgentAction):
                t_start = action.t_start
                t_end = action.t_end

            elif isinstance(action, dict):
                t_start = action['t_start']
                t_end = action['t_end']

            else:
                raise ValueError(f"Cannot check plan of actions of type {type(action)}")

            if t_start_prev is not None and t_end_prev is not None:
                if t_start_prev > t_start:
                    raise ValueError(f"Plan contains action with start time prior to its previous action's start time.")
                elif t_end_prev > t_start:
                    raise ValueError(f"Plan contains action with start time prior to its previous action's end time.")

            t_start_prev = t_start
            t_end_prev = t_end

        return True
    
    def __eq__(self, __value: object) -> bool:
        if len(self.actions) != len(__value.actions):
            return False
        
        for action in self.actions:
            if action not in __value.actions:
                return False
            
        return self.t == __value.t_update

    def __str__(self) -> str:
        out = f't_plan = {self.t}[s]\n'
        out += f'id\t  action type\tt_start\tt_end\n'

        if self.empty():
            out += 'EMPTY\n\n'
        else:
            for action in self.actions:
                if isinstance(action, AgentAction):
                    out += f"{action.id.split('-')[0]}  {action.action_type}\t{round(action.t_start,1)}\t{round(action.t_end,1)}\n"
        out += f'\nn actions in plan: {len(self)}'
        out += f'\nn measurements in plan: {len([action for action in self if isinstance(action, MeasurementAction)])}'
        out += f'\nn bradcasts in plan: {len([action for action in self if isinstance(action, BroadcastMessageAction)])}'
        out += f'\nn maneuvers in plan: {len([action for action in self if isinstance(action, ManeuverAction)])}'
        out += f'\nn travel actions in plan: {len([action for action in self if isinstance(action, TravelAction)])}\n'
        return out

    def get_horizon(self) -> float:
        """ Returns current planning horizon """
        return 0.0 if self.empty() else self.actions[-1].t_end - self.actions[0].t_start

    def empty(self) -> bool:
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
        self.t_next = t_next

        super().__init__(*actions, t=t)

    def copy(self) -> object:
        return Preplan(self.actions, t=self.t, horizon=self.horizon, t_next=self.t_next)
    
    def add(self, action: AgentAction, t: float) -> None:
        if self.t + self.horizon < action.t_end:
            raise ValueError(f'cannot add action scheduled to be done past the planning horizon of this plan')

        super().add(action, t)

class Replan(Plan):
    def __init__(self, 
                 *actions, 
                 t: float = 0,
                 t_next : float = np.Inf
                 ) -> None:
        
        self.t_next = t_next

        super().__init__(*actions, t=t)

    def add(self, action: AgentAction, t: float) -> None:
        # if self.t_next < action.t_end:
        #     return
        #     # raise ValueError(f'cannot add action scheduled to be done past the next scheduled replan for this plan')

        super().add(action, t)

    def copy(self) -> object:
        return Replan(self.actions, t=self.t, t_next=self.t_next)
    
    def from_preplan(preplan : Preplan, *actions, t : float) -> object:
        """ creates a modified plan from an existing preplan and a set of new actions to be added to said plan """
        return Replan(preplan.actions, *actions, t=t, t_next=preplan.t_next)
    