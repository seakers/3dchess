from typing import Union
import numpy as np
import uuid

from chess3d.agents.planning.tasks import EventObservationTask
from chess3d.mission.events import GeophysicalEvent
from chess3d.mission.mission import *
    
class TaskRequest:
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
        ### Task Request 
        Indicates the existance of an event of interest at a given target point and requests a series of objectives to be performed by agents.

        ### Arguments:
            - requester (`str`): name of agent requesting the observations
            - event (`GeophysicalEvent` or `dict`): event being requested for observation
            - parameters (`list`): list of parameters to be observed
            - t_req (`float`): time at which the request is made in [s] from the beginning of the simulation
            - id (`str`) : identifying number for this task in uuid format
        
        """
        # convert arguments if needed
        if isinstance(event, dict): event = GeophysicalEvent.from_dict(event)
        if isinstance(objectives[0], dict): objectives = [MissionObjective.from_dict(objective) for objective in objectives]
        if isinstance(t_req, str) and t_req.lower() == "inf": t_req = np.Inf

        # check argument types
        assert isinstance(requester, str), f'`requester` must be of type `str`. Is of type {type(requester)}.'
        assert isinstance(event, (GeophysicalEvent, dict)), f'`event` must be of type `GeophysicalEvent` or `dict`. Is of type {type(event)}.'
        assert isinstance(mission_name, str), f'`mission_name` must be of type `str`. Is of type {type(mission_name)}.'
        assert isinstance(objectives, list), f'`objectives` must be of type `list`. Is of type {type(objectives)}.'
        assert all([isinstance(objective, (dict, MissionObjective)) for objective in objectives]), \
               f'`objectives` must be a `list` of elements of type `dict` or `MissionObjective`.'
        assert len(objectives) > 0, f'`objectives` must be a non-empty `list` of elements of type `dict` or `MissionObjective`.'
        assert isinstance(t_req, (float, int)), f'`t_req` must be of type `float` or `int`. Is of type {type(t_req)}.'
        assert t_req >= 0, f"`t_req` must have a non-negative value."
        
        # initialize attributes
        self.requester : str = requester
        self.event : GeophysicalEvent= event
        self.mission_name : str = mission_name
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
        out['event'] = out['event'].to_dict() if isinstance(out['event'], GeophysicalEvent) else out['event']
        out['objectives'] = [objective.to_dict() for objective in out['objectives']]
        return out
    
    def to_tasks(self) -> list:
        """
        Converts this task request into a list of `Task` objects, one for each objective.
        """
        return [EventObservationTask(self.event, 
                                    self.mission_name, 
                                    objective, 
                                    objective.reobservation_strategy)
                for objective in self.objectives]

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