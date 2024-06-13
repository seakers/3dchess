import random
import numpy as np
from nodes.science.reqs import MeasurementRequest

"""
List of utility functions used to evalute the value of observations
"""

def synergy_factor(req : dict, subtask_index : int, **_) -> float:
    # unpack request
    req : MeasurementRequest = MeasurementRequest.from_dict(req)
    
    _, dependent_measurements = req.observation_groups[subtask_index]
    k = len(dependent_measurements) + 1

    if k / len(req.observations_types) == 1.0:
        return 1.0
    else:
        return 1.0/3.0

def no_utility(**_) -> float:
    return 0.0

def fixed_utility(req : dict, **_) -> float:
    # unpack request
    req : MeasurementRequest = MeasurementRequest.from_dict(req)

    return req.s_max / len(req.observations_types)

def random_utility(req : dict, **_) -> float:    
    # unpack request
    req : MeasurementRequest = MeasurementRequest.from_dict(req)

    return req.s_max * random.random() / len(req.observations_types)

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
    utility = req.s_max * (t_img - req.t_end) / (req.t_start - req.t_end)

    return utility / len(req.observations_types)

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
    utility = req.s_max * np.exp( - req.urgency * (t_img - req.t_start) )

    return utility / len(req.observations_types)

utility_function = {
    "NONE" : no_utility,
    "FIXED" : fixed_utility,
    "RANDOM" : random_utility,
    "LINEAR" : linear_utility,
    "EXPONENTIAL" : exp_utility
}