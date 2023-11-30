import numpy as np
from nodes.actions import AgentAction, WaitForMessages


class Plan(object):
    """ Describes a plan to be performed by an agent """
    
    def __init__(self, preplan : list = None, t : float = 0.0, horizon : float = np.Inf) -> None:
        # check for argument types
        if preplan:
            if not isinstance(preplan, list):
                raise ValueError(f'pre-defined plan must be of type `list`.')

            for action in preplan:
                if not isinstance(action, AgentAction):
                    raise ValueError(f'pre-defined plan must be comprised of objects of type AgentAction`.')
                
        # initialize values
        self.t_update = t
        self.horizon = horizon
        self.plan = []
        
        # load preplan
        if preplan:
            self.update_plan(preplan, t)

    def __iter__(self) -> list:
        for action in self.plan:
            yield action

    def update_plan(self, new_plan : list, t : float) -> None:
        """ Updates the current plan to a new list of actions """

        # check argument types
        if not isinstance(new_plan, list):
            raise ValueError(f'updated plan must be of type `list`.')
        
        if any([not isinstance(action, AgentAction) for action in new_plan]):
            raise ValueError(f'updated plan must be comprised of objects of type AgentAction`.')

        # check feasiblity
        if not self.__is_feasible(new_plan):
            raise RuntimeError("Cannot update plan: new plan is unfeasible.")
        
        # update plan
        self.plan = [action for action in new_plan 
                     if action.t_start <= self.t_update + self.horizon]
        
        t_next = self.t_update + self.horizon
        if self.plan:
            if self.plan[-1].t_end < t_next:
                self.plan.append(WaitForMessages(self.plan[-1].t_end, t_next))
        else:
            self.plan.append(WaitForMessages(t, t_next))

        # update latest update time
        self.t_update = t              
        
    def add_to_plan(self, action : AgentAction, t : float) -> None:
        """ adds action to plan """
        
        # check if action fits within planning horizon
        if action.t_start > self.t_update + self.horizon:
            return

        # TODO place action in plan

        pass

        if not self.__is_feasible(self.plan):
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
        plan_ids = [action.id for action in self.plan]

        # compile performed actions
        performed_actions = [action for action in completed_actions]
        performed_actions.extend(aborted_actions)

        # remove performed actions from plan
        for performed_action in performed_actions:
            performed_action : AgentAction
            for action in self.plan:
                action : AgentAction
                if performed_action.id == action.id:
                    self.plan.remove(action)

        # update pending actions
        for pending_action in pending_actions:
            pending_action : AgentAction
            
            # update start time to current time
            pending_action.t_start = t

            # update plan with updated action
            i_pending = plan_ids.index(pending_action.id)
            pending_action[i_pending] = pending_action

    def get_next_actions(self, t : float) -> list:
        """ returns a list of dicts """

        # get next available action to perform
        plan_out = [action.to_dict() for action in self.plan if action.t_start <= t <= action.t_end]

        # idle if no more actions can be performed
        if not plan_out:
            t_next = self.t_update + self.horizon
            t_idle = self.plan[0].t_start if len(self.plan) > 0 else t_next
            action = WaitForMessages(t, t_idle)
            plan_out.append(action.to_dict())     

        # sort plan in order of ascending start time 
        plan_out.sort(key=lambda a: a['t_start'])

        # check plan feasibility
        if not self.__is_feasible(plan_out):
            raise RuntimeError("Unfeasible plan.")

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
                    return False
                elif t_end_prev > t_start:
                    return False

            t_start_prev = t_start
            t_end_prev = t_end

        return True
    
    def empty(self) -> bool:
        """ Checks if the current plan is empty """
        return len(self.plan) == 0