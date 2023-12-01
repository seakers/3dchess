import uuid
import numpy as np
from nodes.actions import *


class Plan(object):
    """ Describes a plan to be performed by an agent """
    
    def __init__(   self, 
                    actions : list = [],  
                    # horizon : float = np.Inf,
                    t : float = 0.0
                ) -> None:
        # check for argument types
        if actions:
            if not isinstance(actions, list):
                raise ValueError(f'pre-defined plan must be of type `list`.')

            for action in actions:
                if not isinstance(action, AgentAction):
                    raise ValueError(f'pre-defined plan must be comprised of objects of type AgentAction`.')
                
        # initialize values
        self.t_update = t
        # self.horizon = horizon
        self.actions = []
        
        # load preplan
        if actions:
            self.update_plan(actions, t)

    def update_plan(self, actions : list, t : float) -> None:
        """ Updates the current plan to a new list of actions """

        # check argument types
        if not isinstance(actions, list):
            raise ValueError(f'updated plan must be of type `list`.')
        
        if any([not isinstance(action, AgentAction) for action in actions]):
            raise ValueError(f'updated plan must be comprised of objects of type AgentAction`.')

        # check feasiblity
        if not self.__is_feasible(actions):
            raise RuntimeError("Cannot update plan: new plan is unfeasible.")
        
        # update plan
        self.actions = [action for action in actions]
        
        t_wait_start = t if self.empty() else self.actions[-1].t_end
        self.actions.append(WaitForMessages(t_wait_start, np.Inf))

        # update latest update time
        self.t_update = t              
        
    def put(self, action : AgentAction, t : float) -> None:
        """ adds action to plan """
        try:
            # check if action is scheduled to occur during while another action is being performed
            interrupted_actions = [interrupted_action for interrupted_action in self.actions 
                                   if interrupted_action.t_start <= action.t_start < interrupted_action.t_end]
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

                    # change start and end times for the interrupted and continued actions
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
            
        finally:
            # update latest update time
            self.t_update = t  

            try:
                # check plan feasibility
                self.__is_feasible(self.actions)
            except ValueError as e:
                raise RuntimeError(f"Unfeasible plan. {e}")

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
        plan_ids = [action.id for action in self.actions]

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

        # update pending actions
        # for pending_action in pending_actions:
        #     pending_action : AgentAction
            
        #     # update start time to current time
        #     pending_action.t_start = t

        #     # update plan with updated action
        #     i_pending = plan_ids.index(pending_action.id)
        #     self.actions[i_pending] = pending_action

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

    def get_horizon(self) -> float:
        """ Returns current planning horizon """
        return 0.0 if self.empty() else self.actions[-1].t_end - self.actions[0].t_start

    def empty(self) -> bool:
        """ Checks if the current plan is empty """
        return not bool(self.actions)
    
    def __iter__(self) -> list:
        for action in self.actions:
            yield action