from enum import Enum
from typing import Union

import numpy as np
from chess3d.mission import MissionObjective
from chess3d.utils import CoordinateTypes
from dmas.agents import AgentAction
   
class ActionTypes(Enum):
    IDLE = 'IDLE'
    TRAVEL = 'TRAVEL'
    MANEUVER = 'MANEUVER'
    BROADCAST = 'BROADCAST'
    WAIT = 'WAIT'
    OBSERVE = 'OBSERVE'
    REPLAN = 'REPLAN'

class FutureBroadcastTypes(Enum):
    PLAN = 'PLAN'                   # broadcast latest planner information
    BIDS = 'BIDS'                   # broadcast latest bids for a task
    REQUESTS = 'REQUESTS'           # broadcast latest known active measurement requests 
    OBSERVATIONS = 'OBSERVATIONS'   # broadcast latest observation info
    REWARD = 'REWARD'               # broadcast latest reward grid information

def action_from_dict(action_type : str, **kwargs) -> AgentAction:
    if action_type == ActionTypes.IDLE.value:
        return IdleAction(**kwargs)
    elif action_type == ActionTypes.TRAVEL.value:
        return TravelAction(**kwargs)
    elif action_type == ActionTypes.MANEUVER.value:
        return ManeuverAction(**kwargs)
    elif action_type == ActionTypes.BROADCAST.value:
        return BroadcastMessageAction(**kwargs)
    elif action_type == ActionTypes.WAIT.value:
        return WaitForMessages(**kwargs)
    elif action_type == ActionTypes.OBSERVE.value:
        return ObservationAction(**kwargs)
    else:
        raise NotImplementedError(f'Action of type {action_type} not yet implemented.')
    
class IdleAction(AgentAction):
    """
    ## Idle Action

    Instructs an agent to idle for a given amount of time

    ### Attributes:
        - action_type (`str`): type of action to be performed
        - t_start (`float`): start time of this action in [s] from the beginning of the simulation
        - t_end (`float`): end time of this this action in [s] from the beginning of the simulation
        - status (`str`): completion status of the action
        - id (`str`) : identifying number for this action in uuid format
    """
    def __init__(   self, 
                    t_start : Union[float, int],
                    t_end : Union[float, int], 
                    status : str = 'PENDING',
                    id: str = None, 
                    **_
                ) -> None:
        """
        Creates an isntance of an Idle Action

        ### Arguments:
            - t_start (`float`): start time of this task in [s] from the beginning of the simulation
            - t_end (`float`): end time of this this task in [s] from the beginning of the simulation
            - status (`str`): completion status of the task
            - id (`str`) : identifying number for this task in uuid format
        """
        super().__init__(ActionTypes.IDLE.value, t_start, t_end, status, id)

class TravelAction(AgentAction):
    """
    ## Travel Action

    Instructs an agent to travel to a particular position
    
    ### Attributes:
        - action_type (`str`): type of action to be performed
        - t_start (`float`): start time of this action in [s] from the beginning of the simulation
        - t_end (`float`): end time of this this action in [s] from the beginning of the simulation
        - status (`str`): completion status of the action
        - id (`str`) : identifying number for this action in uuid format
        - final_pos (`list`): coordinates desired destination
        - pos_type (`str`): coordinate basis being used for the desired destination
    """
    def __init__(self,
                final_pos : list, 
                t_start : Union[float, int],
                t_end : Union[float, int] = np.Inf,
                pos_type : str = CoordinateTypes.CARTESIAN.value,
                status : str = 'PENDING',
                id: str = None, 
                **_) -> None:
        """
        Creates an instance of a Travel Action

        ### Arguments:
            - final_pos (`list`): coordinates desired destination
            - t_start (`float`): start time of this action in [s] from the beginning of the simulation
            - pos_type (`str`): coordinate basis being used for the desired destination
            - status (`str`): completion status of the action
            - id (`str`) : identifying number for this action in uuid format
        """
            
        super().__init__(ActionTypes.TRAVEL.value, t_start, t_end, status=status, id=id)
        
        if not isinstance(final_pos, list):
            raise AttributeError(f'`final_pos` must be of type `list`. is of type {type(final_pos)}.')
        
        if pos_type == CoordinateTypes.CARTESIAN.value and len(final_pos) != 3:
            raise ValueError(f'`final_pos` must be a list of 3 values (x, y, z). is of length {len(final_pos)}.')
        elif pos_type == CoordinateTypes.KEPLERIAN.value and len(final_pos) != 5:
            raise ValueError(f'`final_pos` must be a list of 5 values (lat, lon, alt). is of length {len(final_pos)}.')
        elif pos_type == CoordinateTypes.LATLON.value and len(final_pos) != 3:
            raise ValueError(f'`final_pos` must be a list of 3 values (lat, lon, alt). is of length {len(final_pos)}.')
        elif (pos_type != CoordinateTypes.CARTESIAN.value
             and pos_type != CoordinateTypes.KEPLERIAN.value
             and pos_type != CoordinateTypes.LATLON.value):
            raise NotImplemented(f'`pos_type` or type `{pos_type}` not yet supported for `MoveAction`.')

        self.final_pos = final_pos
        self.pos_type = pos_type

class ManeuverAction(AgentAction):
    """
    ## Maneuver Action

    Instructs a satellite agent to perform an attitude maneuver
    
    ### Attributes:
        - action_type (`str`): type of action to be performed
        - t_start (`float`): start time of this action in [s] from the beginning of the simulation
        - t_end (`float`): end time of this this action in [s] from the beginning of the simulation
        - status (`str`): completion status of the action
        - id (`str`) : identifying number for this action in uuid format
        - final_attitude (`float`): desired off-nadir angle parallel to velocity vector
    """
    def __init__(self,
                final_attitude : list, 
                attitude_rates : list,
                t_start : Union[float, int],
                t_end : Union[float, int] = np.Inf,
                status : str = 'PENDING',
                id: str = None, 
                **_) -> None:
        super().__init__(ActionTypes.MANEUVER.value, t_start, t_end, status=status, id=id)
        
        # check values
        if not isinstance(final_attitude, list): raise ValueError(f'`final_attitude` must be of type `list`. Is of type {type(final_attitude)}.')
        if len(final_attitude) != 3: raise ValueError(f'`final_attitude` must be of type `list` of length 3. Is of length {len(final_attitude)}.')

        # set parameters
        self.final_attitude = [th for th in final_attitude]
        self.attitude_rates = [dth for dth in attitude_rates]

class BroadcastMessageAction(AgentAction):
    """
    ## Broadcast Message Action 

    Instructs an agent to broadcast a message to all of its peers

    ### Attributes:
        - action_type (`str`): type of action to be performed
        - msg (`dict`): message to be broadcasted to other agents in the network
        - t_start (`float`): start time of this action in [s] from the beginning of the simulation
        - t_end (`float`): start time of this actrion in[s] from the beginning of the simulation
        - status (`str`): completion status of the task
        - id (`str`) : identifying number for this task in uuid format
    """
    def __init__(self, 
                msg : dict,
                t_start : Union[float, int],
                status : str = 'PENDING',
                id: str = None, 
                **_) -> None:
        """
        Creates an instance of a Broadcast Message Action

        ### Arguments
            - msg (`dict`): message to be broadcasted to other agents in the network
            - t_start (`float`): start time of this action in [s] from the beginning of the simulation
            - status (`str`): completion status of the task
            - id (`str`) : identifying number for this task in uuid format
        """
        super().__init__(ActionTypes.BROADCAST.value, t_start, t_start, status=status, id=id)
        self.msg = msg

class FutureBroadcastMessageAction(BroadcastMessageAction):
    """
    ## Future Broadcast Message Action 

    Instructs an agent that a message is to be broadcast to all of its peers but the contents are not yet kown

    ### Attributes:
        - broadcast_type (`str`): type of broadcast to be performed
        - t_start (`float`): start time of this action in [s] from the beginning of the simulation
        - t_end (`float`): start time of this actrion in[s] from the beginning of the simulation
        - status (`str`): completion status of the task
        - id (`str`) : identifying number for this task in uuid format
    """
        
    def __init__(self, 
                broadcast_type : str,
                t_start : Union[float, int],
                status : str = 'PENDING',
                id: str = None, 
                **_) -> None:
        """
        Creates an instance of a Future Broadcast Message Action

        ### Arguments
            - broadcast_type (`dict`): type of broadcast to be performed
            - t_start (`float`): start time of this action in [s] from the beginning of the simulation
            - status (`str`): completion status of the task
            - id (`str`) : identifying number for this task in uuid format
        """
        super().__init__(dict(), t_start, status, id, **_)
        self.broadcast_type = broadcast_type

class ObservationAction(AgentAction):
    """
    Describes an observation to be performed by agents in the simulation

    ### Attributes:
        - instrument_name (`str`): name of the instrument_name that will perform this action
        - target (`list`): coordinates for the intended observation target in (lat [deg], lon [deg], alt [km]) 
        - look_angle (`float`): look angle of the observation in [deg]
        - t_start (`float`): start time of the measurement of this action in [s] from the beginning of the simulation
        - t_end (`float`): end time of the measurment of this action in [s] from the beginning of the simulation
        - id (`str`) : identifying number for this task in uuid format
    """  
    def __init__(   self,
                    instrument_name : str,
                    targets : list, 
                    objectives : list,
                    look_angle : float, 
                    t_start: Union[float, int], 
                    duration: Union[float, int] = 0.0, 
                    status: str = 'PENDING', 
                    id: str = None, 
                    **_) -> None:
        """
        Creates an instance of an Observation Action
        ### Arguments:
            - instrument_name (`str`): name of the instrument_name that will perform this action
            - targets (`list`): list of coordinates for the intended observation target in (lat [deg], lon [deg], alt [km]) 
            - look_angle (`float`): look angle of the observation in [deg]
            - t_start (`float`): start time of the measurement of this action in [s] from the beginning of the simulation
            - t_end (`float`): end time of the measurment of this action in [s] from the beginning of the simulation
            - id (`str`) : identifying number for this task in uuid format
        """
        t_end = t_start + duration
        super().__init__(ActionTypes.OBSERVE.value, t_start, t_end, status, id)
        
        # check parameters
        if not isinstance(instrument_name,str): raise ValueError(f'`instrument_name` must be of type `str`. Is of type `{type(instrument_name)}`.')
        if not isinstance(targets, list): raise ValueError(f'`targets` must be of type `list`. Is of type `{type(targets)}`.')
        
        if not all(isinstance(target, list) for target in targets): 
            raise ValueError(f'`target` must be a `list` of numerical values of type `float`. Is of type `{type(targets)}`.')
        if any([len(target) != 3 for target in targets]): raise ValueError(f'`target` must be a `list` of length 3 (lat, lon, alt). Is of length {len(targets)}.')
        if not isinstance(look_angle,float) and not isinstance(look_angle,int): raise ValueError(f'`look_angle` must be a numerical value of type `float`. Is of type `{type(look_angle)}`')

        if all([isinstance(objective, dict) for objective in objectives]):
            objectives = [MissionObjective.from_dict(objective) for objective in objectives]
        
        # get unique targets and objectives
        objectives = list(set(objectives))
        targets = {tuple(target) for target in targets}
        targets = [list(target) for target in targets]

        # set parameters
        self.instrument_name = instrument_name
        self.objectives : list[MissionObjective] = [objective.copy() for objective in objectives]
        self.targets = [[coordinate for coordinate in target] for target in targets]
        self.look_angle = look_angle

    def to_dict(self):
        out = super().to_dict()
        out['objectives'] = [ojective.to_dict() for ojective in self.objectives]
        return out

class WaitForMessages(AgentAction):
    """
    ## Wait for Messages Action

    Instructs an agent to idle until a roadcast from a peer is received or a timer runs out

    ### Attributes:
        - action_type (`str`): type of action to be performed
        - t_start (`float`): start time of the waiting period in [s] from the beginning of the simulation
        - t_end (`float`): start time of the waiting period in[s] from the beginning of the simulation
        - status (`str`): completion status of the task
        - id (`str`) : identifying number for this task in uuid format
    """
    def __init__(   self, 
                    t_start: Union[float, int], 
                    t_end: Union[float, int] = np.Inf, 
                    status: str = 'PENDING', 
                    id: str = None, 
                    **_
                ) -> None:
        """
        Creates an isntance of a Waif For Message Action

         ### Arguments:
            - t_start (`float`): start time of the waiting period in [s] from the beginning of the simulation
            - t_end (`float`): start time of the waiting period in[s] from the beginning of the simulation
            - status (`str`): completion status of the task
            - id (`str`) : identifying number for this task in uuid format
        """
        super().__init__(ActionTypes.WAIT.value, t_start, t_end, status, id)

class TriggerReplan(AgentAction):
    """
    ## Replan Action

    Instructs the planner to generate a new plan for the agent. Used to schedule future replanning events.
    """
    def __init__(self, 
                t_start: Union[float, int], 
                status: str = 'PENDING', 
                id: str = None, 
                **_
            ) -> None:
        super().__init__(ActionTypes.REPLAN.value, t_start, t_start, status, id)