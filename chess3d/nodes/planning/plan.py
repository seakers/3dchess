import numpy as np
from nodes.actions import AgentAction, WaitForMessages


class Plan(object):
    """ Describes a plan to be performed by an agent """
    
    def __init__(self, actions : list = None, t : float = 0.0, horizon : float = np.Inf) -> None:
        # check for argument types
        if actions:
            if not isinstance(actions, list):
                raise ValueError(f'pre-defined plan must be of type `list`.')

            for action in actions:
                if not isinstance(action, AgentAction):
                    raise ValueError(f'pre-defined plan must be comprised of objects of type AgentAction`.')
                
        # initialize values
        self.t_update = t
        self.horizon = horizon
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
        self.actions = [action for action in actions 
                     if action.t_start <= self.t_update + self.horizon]
        
        t_next = self.t_update + self.horizon
        if not self.empty():
            if self.actions[-1].t_end < t_next:
                self.actions.append(WaitForMessages(self.actions[-1].t_end, t_next))
        else:
            self.actions.append(WaitForMessages(t, t_next))

        # update latest update time
        self.t_update = t              
        
    def add_to_plan(self, action : AgentAction, t : float) -> None:
        """ adds action to plan """
        
        # check if action fits within planning horizon
        if action.t_start > self.t_update + self.horizon:
            return

        # TODO place action in plan

        pass

        if not self.__is_feasible(self.actions):
            raise RuntimeError("Unfeasible plan.")

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
            t_next = self.t_update + self.horizon
            t_idle = self.actions[0].t_start if not self.empty() else t_next
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

    def empty(self) -> bool:
        """ Checks if the current plan is empty """
        return not bool(self.actions)
    
    def __iter__(self) -> list:
        for action in self.actions:
            yield action