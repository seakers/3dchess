import uuid
import numpy as np
from nodes.actions import *


class Plan(object):
    """ Describes a plan to be performed by an agent """
    
    def __init__(   self, 
                    *actions,  
                    # horizon : float = np.Inf,
                    t : float = 0.0
                ) -> None:
        # initialize values
        self.t_update = t
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
            for action in actions:
                self.put(action, t)
                
        # add indefinite wait at the end of the plan
        t_wait_start = t if self.empty() else self.actions[-1].t_end
        self.put(WaitForMessages(t_wait_start, np.Inf), t)

        # update plan update time
        self.t_update = t              
        
    def put(self, action : AgentAction, t : float) -> None:
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
            self.t_update = t  

            try:
                # check plan feasibility
                self.__is_feasible(self.actions)
            except ValueError as e:
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
        plan_out = [action.to_dict() for action in self.actions if action.t_start <= t <= action.t_end]

        # idle if no more actions can be performed
        if not plan_out:
            t_idle = self.actions[0].t_start if not self.empty() else np.Inf
            action = WaitForMessages(t, t_idle)
            plan_out.append(action.to_dict())     

        # sort plan in order of ascending start time 
        plan_out.sort(key=lambda a: a['t_start'])

        # check plan feasibility
        try:
            self.__is_feasible(plan_out)
        except ValueError as e:
            raise RuntimeError(f"Unfeasible plan. {e}")

        return plan_out    

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
    
    def __str__(self) -> str:
        out = f'\nid\taction type\tt_start\tt_end\n'

        if self.empty():
            out += 'EMPTY\n\n'
        else:
            for action in self.actions:
                if isinstance(action, AgentAction):
                    out += f"{action.id.split('-')[0]}, {action.action_type}, {action.t_start}, {action.t_end}\n"

        return out

    def get_horizon(self) -> float:
        """ Returns current planning horizon """
        return 0.0 if self.empty() else self.actions[-1].t_end - self.actions[0].t_start

    def empty(self) -> bool:
        """ Checks if the current plan is empty """
        return not bool(self.actions)
    
    def __iter__(self) -> list:
        for action in self.actions:
            yield action