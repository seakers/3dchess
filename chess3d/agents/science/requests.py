from typing import Union
import numpy as np
import uuid

from chess3d.mission import *

# class TaskRequest(object):
#     """
#     Indicates the existance of an event of interest at a given target point
#     and requests agents to perform an observatrion with a given set of instruments

#     ### Attributes:
#         - requester (`str`): name of agent requesting the observations
#         - target (`list`): location of the target area of interest expressed in [lat[deg], lon[deg], alt[km]]
#         - severity (`float`): severity of the event being measured
#         - observation_types (`list`): measurement types required to perform this task
#         - t_start (`float`): start time of the availability of this task in [s] from the beginning of the simulation
#         - t_end (`float`): end time of the availability of this task in [s] from the beginning of the simulation
#         - t_corr (`float`): maximum decorralation time between different observations
#         - id (`str`) : identifying number for this task in uuid format
#     """        
#     def __init__(self, 
#                  requester : str,
#                  target : list,
#                  severity : float,
#                  observation_types : list,
#                  t_start: Union[float, int], 
#                  t_end: Union[float, int] = np.Inf, 
#                  t_corr: Union[float, int] = np.Inf, 
#                  id: str = None, 
#                  **_
#                 ) -> None:
#         """
#         Creates an instance of a measurement request 

#         ### Arguments:
#             - requester (`str`): name of agent requesting the observations
#             - target (`list`): location of the target area of interest expressed in [lat[deg], lon[deg], alt[km]]
#             - severity (`float`): severity of the event being measured
#             - observation_types (`list`): measurement types of observations required to perform this task
#             - t_start (`float`): start time of the availability of this task in [s] from the beginning of the simulation
#             - t_end (`float`): end time of the availability of this task in [s] from the beginning of the simulation
#             - t_corr (`float`): maximum decorralation time between different observations
#             - id (`str`) : identifying number for this task in uuid format
#         """
#         # check arguments 
#         if not isinstance(requester, str):
#             raise ValueError(f'`rqst` must be of type `str`. Is of type {type(requester)}.')
#         if not isinstance(target, list):
#             raise ValueError(f'`target` must be of type `list`. is of type {type(target)}.')
#         if any([not isinstance(target_val, float) and not isinstance(target_val, int) for target_val in target]):
#             raise ValueError(f'`target` must be a `list` of elements of type `float` or type `int`.')
#         if len(target) != 3:
#             raise ValueError(f'`target` must be a list of size 3. Is of size {len(target)}.')
#         if not isinstance(severity, float) and not isinstance(severity, int):
#             raise ValueError(f'`severity` must be of type `float` or type `int`. is of type {type(severity)}.')
#         if not isinstance(observation_types, list):
#             raise ValueError(f'`instruments` must be of type `list`. is of type {type(observation_types)}.')
#         if any([not isinstance(observations_type, str) for observations_type in observation_types]):
#             raise ValueError(f'`measurements` must a `list` of elements of type `str`.')
        
#         if isinstance(t_start, str) and t_start.lower() == "inf":   t_start = np.Inf
#         if isinstance(t_end, str)   and t_end.lower() == "inf":     t_end = np.Inf
#         if isinstance(t_corr, str)  and t_corr.lower() == "inf":    t_corr = np.Inf
        
#         if t_start > t_end: raise ValueError(f"`t_start` must be lesser than `t_end`")
#         if t_corr < 0:      raise ValueError(f"`t_corr` must have a non-negative value.")
        
#         # initialize attributes
#         self.requester = requester
#         self.target = [target_val for target_val in target]
#         self.severity = severity
#         self.observation_types = [obs_type for obs_type in observation_types]    
#         self.t_start = t_start
#         self.t_end = t_end
#         self.t_corr = t_corr
#         self.id = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())
        
#     def __repr__(self):
#         task_id = self.id.split('-')
#         return f'MeasurementReq_{task_id[0]}'

#     def to_dict(self) -> dict:
#         """
#         Crates a dictionary containing all information contained in this measurement request object
#         """
#         return dict(self.__dict__)

#     def from_dict(d : dict) -> object:
#         return TaskRequest(**d)
    
#     def __eq__(self, other : object) -> bool:
#         if not isinstance(other, TaskRequest):
#             raise ValueError(f'cannot compare `MeasurementRequest` object to an object of type {type(other)}.')
        
#         my_dict : dict = self.to_dict()
#         other_dict : dict = other.to_dict()

#         my_dict.pop('id')
#         other_dict.pop('id')

#         return my_dict == other_dict
            
#     def same_event(self, other : object) -> bool:
#         """ compares the events being requested for observation between two measurement requests """

#         if not isinstance(other, TaskRequest):
#             raise ValueError(f'cannot compare `MeasurementRequest` object to an object of type {type(other)}.')

#         same_target = all([abs(self.target[i]-other.target[i]) <= 1e-3 for i in range(len(self.target))])
#         same_severity = abs(self.severity - other.severity) <= 1e-3
#         same_observations = (len(self.observation_types) == len(other.observation_types)
#                              and all([observation in other.observation_types for observation in self.observation_types]))
#         same_time = abs(self.t_end - other.t_end) <= 1e-3
#         same_decorrelation = abs(self.t_corr - other.t_corr) <= 1e-3

#         if not (same_target and same_severity and same_time):
#             x = 1

#         return (
#                 same_target
#                 and same_severity
#                 # and same_observations
#                 and same_time
#                 # and same_decorrelation
#                 )

#     def __hash__(self) -> int:
#         return hash(repr(self))

#     def copy(self) -> object:
#         return TaskRequest.from_dict(self.to_dict())
    
class TaskRequest:
    """
        Indicates the existance of an event of interest at a given target point
    """
    def __init__(self,
                 requester : str,
                 event : Union[GeophysicalEvent, dict],
                 mission_name : str,
                 objectives : list,
                 t_req : Union[float, int],
                 id : str = None,
                 **_ 
                 ):
        """
        Creates an instance of a measurement request

        ### Arguments:
            - requester (`str`): name of agent requesting the observations
            - event (`GeophysicalEvent` or `dict`): event being requested for observation
            - parameters (`list`): list of parameters to be observed
            - t_req (`float`): time at which the request is made in [s] from the beginning of the simulation
            - id (`str`) : identifying number for this task in uuid format
        
        """
        if not isinstance(requester, str):
            raise ValueError(f'`requester` must be of type `str`. Is of type {type(requester)}.')
        if isinstance(event, dict):
            event = GeophysicalEvent.from_dict(event)
        if not isinstance(event, GeophysicalEvent):
            raise ValueError(f'`event` must be of type `GeophysicalEvent`. Is of type {type(event)}.')
        if not isinstance(objectives, list):
            raise ValueError(f'`objectives` must be of type `list`. Is of type {type(objectives)}.')
        if any([not isinstance(objective, dict) for objective in objectives]) and any([not isinstance(objective, MissionObjective) for objective in objectives]):
            raise ValueError(f'`objectives` must be a `list` of elements of type `dict` or `MissionObjective`.')
        if len(objectives) == 0:
            raise ValueError(f'`objectives` must be a non-empty `list` of elements of type `dict` or `MissionObjective`.')
        if isinstance(t_req, str) and t_req.lower() == "inf": t_req = np.Inf
        if not isinstance(t_req, float) and not isinstance(t_req, int):
            raise ValueError(f'`t_req` must be of type `float` or `int`. Is of type {type(t_req)}.')
        if t_req < 0: raise ValueError(f"`t_req` must have a non-negative value.")

        # initialize attributes
        self.requester : str = requester
        self.event : GeophysicalEvent= event
        self.mission_name : str = mission_name
        # convert list of dicts to list of MissionObjectives if needed 
        if isinstance(objectives[0], dict): objectives = [MissionObjective.from_dict(objective) for objective in objectives]
        self.objectives : list[EventDrivenObjective] = [objective for objective in objectives]
        self.t_req : float = t_req
        self.id : str = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())

    def __repr__(self):
        task_id = self.id.split('-')
        return f'TaskRequest_{task_id[0]}'

    def to_dict(self) -> dict:
        """
        Crates a dictionary containing all information contained in this measurement request object
        """
        out = dict(self.__dict__)
        out['objetives'] = [objective.to_dict() for objective in out['objetives']]
        return out

    def from_dict(d : dict) -> object:
        return TaskRequest(**d)
    
    def same_event(self, other : object) -> bool:
        """ compares the events being requested for observation between two measurement requests """

        if not isinstance(other, TaskRequest):
            raise ValueError(f'cannot compare `TaskRequest` object to an object of type {type(other)}.')

        same_type = self.event.event_type == other.event.event_type
        same_target = all([abs(self.event.location[i]-other.event.location[i]) <= 1e-3 
                           for i in range(len(self.event.location))])
        same_severity = abs(self.event.severity - other.event.severity) <= 1e-3
        same_time = abs(self.event.t_end - other.event.t_end) <= 1e-3

        return (
                same_type
                and same_target
                and same_severity
                and same_time
                )