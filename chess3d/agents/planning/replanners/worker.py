from dmas.agents import AgentAction

from chess3d.agents.actions import action_from_dict
from chess3d.agents.planning.plan import Plan, Replan
from chess3d.agents.planning.replanners.replanner import AbstractReplanner
from chess3d.agents.states import SimulationAgentState
from chess3d.messages import PlanMessage


class WorkerReplanner(AbstractReplanner):
    """
    Worker Replanner class that handles replanning tasks for agents.
    It processes the replanning requests and updates the agent's plan accordingly.
    """
    def __init__(self, debug = False, logger = None):
        super().__init__(debug, logger)
        self.plan_message : PlanMessage = None

    def update_percepts(self, state, current_plan, incoming_reqs, relay_messages, misc_messages, completed_actions, aborted_actions, pending_actions):       
        super().update_percepts(state, current_plan, incoming_reqs, relay_messages, misc_messages, completed_actions, aborted_actions, pending_actions)

        if misc_messages or relay_messages:
            x = 1 # breakpoint

        # get the plan messages for this agent
        plan_messages = {msg for msg in misc_messages 
                         if isinstance(msg, PlanMessage)
                         and msg.agent_name == state.agent_name}
        
        # update the latest plan message
        for plan_message in plan_messages:
            if self.plan_message is None or plan_message.t_plan > self.plan_message.t_plan:
                self.plan_message = plan_message        
        
    def needs_planning(self, _, __, current_plan : Plan, ___, **kwargs):
        # only replans if there is a plan message
        return current_plan.t < self.plan_message.t_plan if self.plan_message else False

    def generate_plan(self, state, *_):
        # get actions from latest plan message
        actions : list[AgentAction] = [action_from_dict(**action) 
                                       for action in self.plan_message.plan]
        
        # only keep actions that start after the current time
        actions = [action for action in actions if action.t_start >= self.plan_message.t_plan]

        # create a plan from plan message actions
        self.plan = Replan(actions, t=self.plan_message.t_plan)

        # return the generated plan
        return self.plan.copy()

    def _schedule_observations(self, state, specs, clock_config, orbitdata, schedulable_tasks, observation_history):
        """ Boilerplate method for scheduling observations."""
        # does not schedule observations for parent agent
        return []
    