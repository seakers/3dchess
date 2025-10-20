from dmas.agents import AgentAction

from chess3d.agents.actions import action_from_dict
from chess3d.agents.planning.plan import Plan, Replan
from chess3d.agents.planning.reactive import AbstractReactivePlanner
from chess3d.agents.states import SimulationAgentState
from chess3d.messages import PlanMessage


class WorkerReplanner(AbstractReactivePlanner):
    """
    Worker Replanner class that handles replanning tasks for agents.
    It processes the replanning requests and updates the agent's plan accordingly.
    """
    def __init__(self, dealer_name : str, debug = False, logger = None):
        super().__init__(debug, logger)
        
        # validate inputs
        assert isinstance(dealer_name, str), "dealer_name must of type `str`"

        self.dealer_name = dealer_name
        self.plan_message : PlanMessage = None

    def update_percepts(self, state, current_plan, incoming_reqs, relay_messages, misc_messages, completed_actions, aborted_actions, pending_actions):       
        super().update_percepts(state, current_plan, incoming_reqs, relay_messages, misc_messages, completed_actions, aborted_actions, pending_actions)

        # check if there are any plan messages for this agent
        plan_messages = {msg for msg in misc_messages 
                         if isinstance(msg, PlanMessage) # filter by message type
                         and msg.agent_name == state.agent_name # filter by agent name                         
                        #  and msg.src == self.dealer_name # TODO filter by dealer name
                         }
        
        # update the latest plan message
        for plan_message in plan_messages:
            if self.plan_message is None or plan_message.t_plan > self.plan_message.t_plan:
                self.plan_message = plan_message        
        
    def needs_planning(self, _, __, current_plan : Plan, ___, **kwargs):
        # only replans if there is an unprocessed plan message
        return self.plan_message is not None

    def generate_plan(self, state : SimulationAgentState, *_):
        # get actions from latest plan message
        actions : list[AgentAction] = [action_from_dict(**action) 
                                       for action in self.plan_message.plan]
        
        # only keep actions that start after the current time
        actions = [action for action in actions if action.t_start >= state.t]

        # create a plan from plan message actions
        self.plan = Replan(actions, t=self.plan_message.t_plan)

        # remove the plan message after processing
        del self.plan_message
        self.plan_message = None

        # return the generated plan
        return self.plan.copy()

    def _schedule_observations(self, *_):
        """ Boilerplate method for scheduling observations."""
        # does not schedule observations for parent agent
        return []
    