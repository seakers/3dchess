from abc import abstractmethod
import logging

from dmas.clocks import ClockConfig
from dmas.utils import runtime_tracker
from dmas.clocks import *

from messages import ClockConfig
from messages import *

from nodes.planning.plan import Plan, Preplan
from nodes.orbitdata import OrbitData
from nodes.states import *
from nodes.actions import *
from nodes.science.requests import *
from nodes.states import SimulationAgentState
from nodes.orbitdata import OrbitData
from nodes.planning.planners import AbstractPlanner

class AbstractPreplanner(AbstractPlanner):
    """
    # Preplanner

    Conducts operations planning for an agent at the beginning of a planning horizon. 
    """
    def __init__(   self, 
                    horizon : float = np.Inf,
                    period : float = np.Inf,
                    logger: logging.Logger = None
                ) -> None:
        """
        ## Preplanner 
        
        Creates an instance of a preplanner class object.

        #### Arguments:
            - utility_func (`Callable`): desired utility function for evaluating observations
            - horizon (`float`) : planning horizon in seconds [s]
            - period (`float`) : period of replanning in seconds [s]
            - logger (`logging.Logger`) : debugging logger
        """
        # initialize planner
        super().__init__(logger)    

        # set parameters
        self.horizon = horizon                               # planning horizon
        self.period = period                                 # replanning period         
        self.plan = Preplan(t=-1,horizon=horizon,t_next=0.0) # initialized empty plan
        
    @runtime_tracker
    def needs_planning( self, 
                        state : SimulationAgentState,
                        current_plan : Plan, 
                        **_
                        ) -> bool:
        """ Determines whether a new plan needs to be initalized """    

        return (current_plan.t < 0                  # simulation just started
                or state.t >= self.plan.t_next)     # periodic planning period has been reached

    @runtime_tracker
    def generate_plan(  self, 
                        state : SimulationAgentState,
                        clock_config : ClockConfig,
                        orbitdata : OrbitData,
                        **_
                    ) -> Plan:
        
        # schedule observations
        observations : list = self._schedule_observations(state, clock_config, orbitdata)
        assert self.is_observation_path_valid(state, observations)

        # schedule broadcasts to be perfomed
        broadcasts : list = self._schedule_broadcasts(state, observations, orbitdata)

        # generate maneuver and travel actions from measurements
        maneuvers : list = self._schedule_maneuvers(state, observations, clock_config, orbitdata)
        
        # wait for next planning period to start
        replan : list = self.__schedule_periodic_replan(state, observations, maneuvers)

        # generate plan from actions
        self.plan : Preplan = Preplan(observations, maneuvers, broadcasts, replan, t=state.t, horizon=self.horizon, t_next=state.t+self.period)    

        # return plan and save local copy
        return self.plan.copy()
        
    @abstractmethod
    def _schedule_observations(self, state : SimulationAgentState, clock_config : ClockConfig, orbitdata : OrbitData = None) -> list:
        """ Creates a list of observation actions to be performed by the agent """    

    @abstractmethod
    def _schedule_broadcasts(self, state: SimulationAgentState, observations : list, orbitdata: OrbitData) -> list:
        return super()._schedule_broadcasts(state, orbitdata)

    @runtime_tracker
    def __schedule_periodic_replan(self, state : SimulationAgentState, observations : list, maneuvers : list) -> list:
        """ Creates and schedules a waitForMessage action such that it triggers a periodic replan """
        
        # calculate next period for planning
        t_next = state.t + self.period

        # find wait start time
        if not observations and not maneuvers:
            t_wait_start = state.t 
        
        else:
            prelim_plan = Preplan(observations, maneuvers, t=state.t)

            actions_in_period = [action for action in prelim_plan.actions 
                                 if  isinstance(action, AgentAction)
                                 and action.t_start < t_next]

            if actions_in_period:
                last_action : AgentAction = actions_in_period.pop()
                t_wait_start = min(last_action.t_end, t_next)
                                
            else:
                t_wait_start = state.t

        # create wait action
        return [WaitForMessages(t_wait_start, t_next)]
