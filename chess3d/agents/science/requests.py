from typing import Union
import numpy as np
import uuid

from chess3d.agents.planning.tasks import EventObservationTask, GenericObservationTask
from chess3d.utils import Interval

    
class TaskRequest:
    def __init__(self,
                 task : Union[GenericObservationTask, dict],
                 requester : str,
                 mission_name : str,
                 t_req : Union[float, int],
                 id : str = None,
                 **_ 
                 ):
        """
        ### Task Request 
        Indicates the existance of an event of interest at a given target point and requests a series of objectives to be performed by agents.

        ### Arguments:
            - task (`GenericObservationTask` or `dict`): task being requested
            - requester (`str`): name of agent requesting the observations
            - mission_name (`str`): name of the mission
            - t_req (`float`): time at which the request is made in [s] from the beginning of the simulation
            - id (`str`) : identifying number for this task request in uuid format
        
        """
        # convert arguments if needed
        if isinstance(task, dict): task = GenericObservationTask.from_dict(task)
        if isinstance(t_req, str) and t_req.lower() == "inf": t_req = np.Inf

        # check argument types
        assert isinstance(task, GenericObservationTask), f'`task` must be of type `GenericObservationTask`. Is of type {type(task)}.'
        assert isinstance(requester, str), f'`requester` must be of type `str`. Is of type {type(requester)}.'
        assert isinstance(mission_name, str), f'`mission_name` must be of type `str`. Is of type {type(mission_name)}.'
        assert isinstance(t_req, (float, int)), f'`t_req` must be of type `float` or `int`. Is of type {type(t_req)}.'
        assert t_req >= 0, f"`t_req` must have a non-negative value."
        
        # initialize attributes
        self.requester : str = requester
        self.task : GenericObservationTask = task
        self.mission_name : str = mission_name
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
        out['task'] = self.task.to_dict()
        return out
    
    @classmethod
    def from_dict(cls, d : dict) -> 'TaskRequest':
        return cls(**d)
    
    def same_event(self, other_req : 'TaskRequest') -> bool:
        """ compares the events being requested for observation between two measurement requests """

        # validate inputs
        if not isinstance(other_req, TaskRequest):
            raise ValueError(f'cannot compare `TaskRequest` object to an object of type {type(other_req)}.')
        if not isinstance(self.task, EventObservationTask) or not isinstance(other_req.task, EventObservationTask):
            raise ValueError(f'`same_event` can only be used to compare `EventObservationTask` objects. One of the tasks is of type {type(self.task)} and the other is of type {type(other_req.task)}.')

        if self.task.event is None and other_req.task.event is None:
            # both requests have no event associated; compare their task parameters instead
            same_type = self.task.task_type == other_req.task.task_type
            same_target = all([abs(self.task.location[i][j]-other_req.task.location[i][j]) <= 1e-3
                            for i in range(len(self.task.location))
                            for j in range(len(self.task.location[i]))
                            ])
            same_severity = True   # no event, so severity is not applicable
            same_time = self.task.availability.overlaps(other_req.task.availability)

        else:
            same_type = self.task.event.event_type == other_req.task.event.event_type
            same_target = all([abs(self.task.event.location[i]-other_req.task.event.location[i]) <= 1e-3
                            for i in range(len(self.task.event.location))])
            same_severity = abs(self.task.event.severity - other_req.task.event.severity) <= 1e-3

            my_event_availability = Interval(self.task.event.t_start, self.task.event.t_start+self.task.event.d_exp)
            other_event_availability = Interval(other_req.task.event.t_start, other_req.task.event.t_start+other_req.task.event.d_exp)
            same_time = my_event_availability.overlaps(other_event_availability)

        return (
                same_type
                and same_target
                and same_severity
                and same_time
                )