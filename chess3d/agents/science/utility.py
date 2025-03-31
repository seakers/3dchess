import random
from typing import Callable
import numpy as np
from chess3d.agents.actions import ObservationAction
from chess3d.agents.science.requests import MeasurementRequest

"""
List of utility functions used to evalute the value of observations
"""

# def synergy_factor(req : dict, subtask_index : int, **_) -> float:
#     # unpack request
#     req : MeasurementRequest = MeasurementRequest.from_dict(req)
    
#     _, dependent_measurements = req.observation_groups[subtask_index]
#     k = len(dependent_measurements) + 1

#     if k / len(req.observations_types) == 1.0:
#         return 1.0
#     else:
#         return 1.0/3.0

def no_utility(**_) -> float:
    return 0.0

def fixed_utility(req : dict, t_img : float, **_) -> float:
    # unpack request
    req : MeasurementRequest = MeasurementRequest.from_dict(req)

    return req.severity if req.t_start <= t_img <= req.t_end else 0.0

def random_utility(req : dict, **_) -> float:    
    # unpack request
    req : MeasurementRequest = MeasurementRequest.from_dict(req)

    return req.severity * random.random()

def linear_utility(   
                    req : dict, 
                    t_img : float,
                    **kwargs
                ) -> float:
    """
    Calculates the expected utility of performing a measurement task.
    Its value decays lineraly with the time of observation

    ### Arguments:
        - state (:obj:`SimulationAgentState`): agent state before performing the task
        - task (:obj:`MeasurementRequest`): task request to be performed 
        - subtask_index (`int`): index of subtask to be performed
        - t_img (`float`): time at which the task will be performed

    ### Retrurns:
        - utility (`float`): estimated normalized utility 
    """
    # unpack request
    req : MeasurementRequest = MeasurementRequest.from_dict(req)
    
    # calculate urgency factor from task
    utility = req.severity * (t_img - req.t_end) / (req.t_start - req.t_end)

    return utility if req.t_start <= t_img <= req.t_end else 0.0

def exp_utility(   
                    req : dict, 
                    t_img : float,
                    **_
                ) -> float:
    """
    Calculates the expected utility of performing a measurement task.
    Its value decays exponentially with the time of observation

    ### Arguments:
        - state (:obj:`SimulationAgentState`): agent state before performing the task
        - task (:obj:`MeasurementRequest`): task request to be performed 
        - subtask_index (`int`): index of subtask to be performed
        - t_img (`float`): time at which the task will be performed

    ### Retrurns:
        - utility (`float`): estimated normalized utility 
    """
    # unpack request
    req : MeasurementRequest = MeasurementRequest.from_dict(req)

    # check time constraints
    if t_img < req.t_start or req.t_end < t_img:
        return 0.0
    
    # calculate urgency factor from task
    utility = req.severity * np.exp( - (t_img - req.t_start) )

    return utility

def event_driven(
                observations : set,
                events : set,
                reward : float, 
                min_reward : float, 
                unobserved_reward_rate : float, 
                max_unobserved_reward : float, 
                reobservation_strategy : Callable,
                event_reward : float,
                t : float, 
                t_update : float, 
                instrument : str,
                **_):
    
    # check if simulation has started yet
    if np.isnan(t_update) and np.isnan(t):
        # reward grid was jut initialized; return initial reward
        return reward

    # find latest event
    latest_events : list[MeasurementRequest] = [event 
                                                for event in events 
                                                if event.t_start <= t]
    latest_events.sort(key=lambda a : a.t_end)
    latest_event : MeasurementRequest = latest_events.pop() if latest_events else None

    # find latest observations
    observations : list[ObservationAction] = list(observations)
    observations.sort(key= lambda a : a.t_end)
    
    # calculate utility
    if latest_event:
        # an event exists for this ground point
        assert latest_event.t_start <= t

        if t <= latest_event.t_end: # event is current
            # count previous observations
            latest_observations = [observation for observation in observations
                                   if latest_event.t_start <= observation.t_start <= latest_event.t_end]
            
            # calculate reward
            if instrument in latest_event.observation_types:
                tp=[latest_event.t_start, latest_event.t_end]
                rp=[event_reward,min_reward]
                reward = np.interp(t,tp,rp) * reobservation_strategy(len(latest_observations))
            else:
                reward = min_reward
            
        else: # event has already passed
            # check if an observation has occurred since then
            latest_observations = [observation for observation in observations
                                   if latest_event.t_end <= observation.t_start <= t]
            latest_observation = latest_observations.pop() if latest_observations else None

            # calculate reward
            t_init = max(latest_event.t_end, latest_observation.t_end) if latest_observation else latest_event.t_end
            reward = (t - t_init) * unobserved_reward_rate / 3600  + min_reward
            reward = min(reward, max_unobserved_reward)

    else: # no events have been detected for this ground point
        # get latest observation if it exists
        latest_observation = observations.pop() if observations else None
        t_init = latest_observation.t_end if latest_observation else 0.0
        
        # assert (t - t_init) >= 0.0 # TODO fix acbba triggering this

        # calculate reward
        reward = (t - t_init) * unobserved_reward_rate / 3600  + min_reward
        reward = min(reward, max_unobserved_reward) * int((t - t_init) >= 0.0)

    return reward

    # TODO time penalty (?)
    day = 24 * 3600
    rho = max(1 - t / day, 0.0)

    return reward * rho

utility_function = {
    "none" : no_utility,
    "fixed" : fixed_utility,
    "random" : random_utility,
    "linear" : linear_utility,
    "exponential" : exp_utility,
    "event" : event_driven
}

"""
List of reobservation strategy functions used to evalute the value of reobservations
"""

def constant_reobs(_) -> float:
    return 1.0

def linear_reobs(n) -> float:
    return n

def log_reobs(n) -> float:
    if n < 1:
        return 1.0
    else:
        return np.log(n) + 1
    
reobservation_strategy = {
    'constant' : constant_reobs,
    'linear_increase' : linear_reobs,
    'log' : log_reobs
}