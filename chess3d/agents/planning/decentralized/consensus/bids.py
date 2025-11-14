from abc import ABC
from ast import Tuple
import asyncio
from enum import Enum
from typing import Any, Union

import numpy as np

from chess3d.agents.science.requests import TaskRequest
from chess3d.agents.planning.tasks import GenericObservationTask

class Bid: 
    """
    ## Measurement Task Bid for Consensus Planners

    Describes a bid placed on a task by a given agent
    """
    # Constants
    NONE = 'none' # NONE value used in various bid attributes
    ## Bid comparison results
    UPDATE_TIME = 'update_time'
    UPDATE = 'update'
    LEAVE = 'leave'
    RESET = 'reset'
    COMPLETED = 'completed'
    ## Rebroadcast comparison results
    REBROADCAST_SELF = 'rebroadcast self'
    REBROADCAST_OTHER = 'rebroadcast other'
    REBROADCAST_EMPTY = 'rebroadcast empty'
    NO_REBROADCAST = 'no rebroadcast'

    def __init__(self,
                 task : GenericObservationTask,
                 main_measurement : str,
                 bidder: str,
                 bid_value: Union[float, int] = 0,
                 winning_bidder: str = NONE,
                 winning_bid: Union[float, int] = 0,
                 t_img: Union[float, int] = np.NINF,
                 t_update: Union[float, int] = np.NINF,
                 performed : bool = False,
                 ):
        """
        ## Measurement Task Bid for Consensus Planners

        Creates an instance of a task bid

        ### Attributes:
            - task (`GenericObservationTask`): observation task being bid on
            - main_measurement (`str`): name of the main measurement assigned by this subtask bid
            - bidder (`bidder`): name of the agent keeping track of this bid information
            - bid_value (`float` or `int`): latest bid value from bidder
            - winning_bidder (`str`): name of current the winning agent
            - winning_bid (`float` or `int`): current winning bid value
            - t_img (`float` or `int`): time where the task is set to be performed by the winning agent
            - t_update (`float` or `int`): latest time when this bid was updated
            - performed (`bool`): indicates if the winner of this bid has performed the measurement request at hand
        """

        # Validate inputs
        assert isinstance(task, GenericObservationTask), f'`task` must be of type `GenericObservationTask`, got `{type(task)}`'
        assert isinstance(main_measurement, str), f'`main_measurement` must be of type `str`, got `{type(main_measurement)}`'
        assert isinstance(bidder, str), f'`bidder` must be of type `str`, got `{type(bidder)}`'
        assert isinstance(bid_value, (float, int)), f'`bid_value` must be of type `float` or `int`, got `{type(bid_value)}`'
        assert isinstance(winning_bidder, str), f'`winning_bidder` must be of type `str`, got `{type(winning_bidder)}`'
        assert isinstance(winning_bid, (float, int)), f'`winning_bid` must be of type `float` or `int`, got `{type(winning_bid)}`'
        assert isinstance(t_img, (float, int)), f'`t_img` must be of type `float` or `int`, got `{type(t_img)}`'
        assert isinstance(t_update, (float, int)), f'`t_update` must be of type `float` or `int`, got `{type(t_update)}`'
        assert isinstance(performed, bool), f'`performed` must be of type `bool`, got `{type(performed)}`'

        # Assign attributes
        self.task = task
        self.main_measurement = main_measurement
        self.bidder = bidder
        self.bid_value = bid_value
        self.winning_bidder = winning_bidder
        self.winning_bid = winning_bid
        self.t_img = t_img
        self.t_update = t_update
        self.performed = performed

    """
    ---------------------------
    COPY AND OTHER CONSTRUCTORS
    ---------------------------
    """

    def to_dict(self) -> dict:
        """
        Crates a dictionary containing all information contained in this bid
        """
        bid_dict = {
            'task': self.task.to_dict(),
            'main_measurement': self.main_measurement,
            'bidder': self.bidder,
            'bid_value': self.bid_value,
            'winning_bidder': self.winning_bidder,
            'winning_bid': self.winning_bid,
            't_img': self.t_img,
            't_update': self.t_update,
            'performed': self.performed
        }
        return bid_dict

    @classmethod
    def from_dict(cls, bid_dict: dict) -> 'Bid':
        """
        Creates a bid class object from a dictionary
        """
        # check if all required keys are present
        required_keys = ['task', 'main_measurement', 'bidder', 'bid_value', 'winning_bidder', 'winning_bid', 't_img', 't_update', 'performed']
        assert all(key in bid_dict for key in required_keys), f'Bid dictionary is missing required keys. Required keys: {required_keys}'
        
        # convert task from dict if necessary
        if isinstance(bid_dict['task'], dict):
            task = GenericObservationTask.from_dict(bid_dict['task'])
        elif isinstance(bid_dict['task'], GenericObservationTask):
            task = bid_dict['task']

        # return bid object
        return cls(
            task=task,
            main_measurement=bid_dict['main_measurement'],
            bidder=bid_dict['bidder'],
            bid_value=bid_dict['bid_value'],
            winning_bidder=bid_dict['winning_bidder'],
            winning_bid=bid_dict['winning_bid'],
            t_img=bid_dict['t_img'],
            t_update=bid_dict['t_update'],
            performed=bid_dict['performed']
        )
    
    def copy(self) -> 'Bid':
        """ Creates a deep copy of this bid object """
        return Bid.from_dict(self.to_dict())

    """
    ------------------
    COMPARISON METHODS
    ------------------
    """

    def __lt__(self, other : object) -> bool:
        # validate inputs
        assert isinstance(other, Bid), f'can only compare bids to other bids.'
        assert self.task == other.task, f'cannot compare bids intended for different tasks (expected task id: {self.task.id}, given id: {other.task.id})'

        # compare bids
        if abs(other.bid_value - self.bid_value) < 1e-3:
            # if there's a tie, use tie-breaker
            return self != self._tie_breaker(self, other)

        return other.bid_value > self.bid_value

    def __gt__(self, other : object) -> bool:
        # validate inputs
        assert isinstance(other, Bid), f'can only compare bids to other bids.'
        assert self.task == other.task, f'cannot compare bids intended for different tasks (expected task id: {self.task.id}, given id: {other.task.id})'

        # compare bids
        if abs(other.bid_value - self.bid_value) < 1e-3:
            # if there's a tie, use tie-breaker
            return self == self._tie_breaker(self, other)

        return other.bid_value < self.bid_value

    def __le__(self, other : object) -> bool:
        # validate inputs
        assert isinstance(other, Bid), f'can only compare bids to other bids.'
        assert self.task == other.task, f'cannot compare bids intended for different tasks (expected task id: {self.task.id}, given id: {other.task.id})'

        # compare bids
        return other.bid_value >= self.bid_value or abs(other.bid_value - self.bid_value) < 1e-3
    
    def __ge__(self, other : object) -> bool:
        # validate inputs
        assert isinstance(other, Bid), f'can only compare bids to other bids.'
        assert self.task == other.task, f'cannot compare bids intended for different tasks (expected task id: {self.task.id}, given id: {other.task.id})'

        # compare bids
        return other.bid_value <= self.bid_value or abs(other.bid_value - self.bid_value) < 1e-3

    def __eq__(self, other : object) -> bool:
        # validate inputs
        assert isinstance(other, Bid), f'can only compare bids to other bids.'

        # check if they are for the same task
        if self.task != other.task: return False

        # compare bids
        return (abs(other.winning_bid - self.winning_bid) < 1e-3    # same bid value
                and other.winning_bidder == self.winning_bidder)    # same winner

    def __ne__(self, other : object) -> bool:
        # validate inputs
        assert isinstance(other, Bid), f'can only compare bids to other bids.'

        # check if they are for the same task
        if self.task != other.task: return True

        # compare bids
        return (abs(other.winning_bid - self.winning_bid) > 1e-3    # different bid value
                or other.winning_bidder != self.winning_bidder)     # different winner
    
    def _tie_breaker(self, bid1 : object, bid2 : object) -> 'Bid':
        """
        Tie-breaking criteria for determining which bid is GREATER in case winning bids are equal
        """
        # validate inputs
        assert isinstance(bid1, Bid) and isinstance(bid2, Bid), f'cannot calculate tie breaker. Both objects must be bids.'

        # compare bids
        ## Check for NONE winning bidders
        if bid2.winning_bidder == self.NONE and bid1.winning_bidder != self.NONE:
            return bid2
        elif bid2.winning_bidder != self.NONE and bid1.winning_bidder == self.NONE:
            return bid1

        ## Compare bidders alphabetically
        return max(bid1, bid2, key=lambda b: b.bidder)

    def __compare(self, other : Any) -> Tuple[str,str]:
        """
        Compares bid with another and either updates, resets, or leaves the information contained in this bid
        depending on the rules specified in:
            - Luke B. Johnson, Sameera S. Ponda, Han-Lim Choi, Jonathan P. How "Asynchronous Decentralized Task Allocation for Dynamic Environments".

        ### Arguments:
            - other_dict (`dict`): dictionary representing the bid being compared to

        ### Returns: Tuple(comparison, rebroadcast)
            - comparison (`self`) : action to perform to this bid upon comparing this bid to the other bid
            - rebroadcast (`self`): rebroadcast action to perform after comparing this bid to the other bid
        """
        # convert other bid to `Bid` class type if necessary
        other : Bid = Bid.from_dict(other) if isinstance(other, dict) else other

        # validate inputs
        assert isinstance(other, Bid), f'can only compare bids to other bids.'
        assert self.task == other.task, f'can only compare bids for the same task (expected task id: {self.task.id}, given id: {other.task.id})'

        if other.performed and not self.performed:
            # update and rebroadcast
            return self.UPDATE, True
        
        if other.bidder == self.bidder:
            if other.t_update > self.t_update:
                # update & rebroadcast other's bid
                return self.UPDATE, \
                    self.REBROADCAST_OTHER
            else:
                # leave & no-rebroadcast
                return self.UPDATE, \
                    self.NO_REBROADCAST
        
        # Total of 31 cases
        elif other.winning_bidder == other.bidder:
            if self.winning_bidder == self.bidder:
                if other.winning_bid > self.winning_bid:
                    # update & rebroadcast other's bid
                    return self.UPDATE,\
                        self.REBROADCAST_OTHER
                
                elif other.winning_bid == self.winning_bid and self != self._tie_breaker(other, self):
                    # update & rebroadcast other's bid
                    return self.UPDATE, \
                        self.REBROADCAST_OTHER
                
                elif other.winning_bid < self.winning_bid:
                    # update time & rebroadcast own bid
                    return self.UPDATE_TIME,\
                        self.REBROADCAST_SELF

            elif self.winning_bidder == other.bidder:
                if abs(other.t_update - self.t_update) < 1e-3:
                    # leave & no-rebroadcast
                    return self.LEAVE,\
                        self.NO_REBROADCAST
                
                elif other.t_update > self.t_update:
                    # update & rebroadcast other's bid
                    return self.UPDATE,\
                        self.REBROADCAST_OTHER
                
                elif other.t_update < self.t_update:
                    # leave & no-rebroadcast
                    return self.LEAVE,\
                        self.NO_REBROADCAST
                
            elif self.winning_bidder not in [other.bidder, self.bidder, self.NONE]:
                if other.winning_bid > self.winning_bid:
                    if other.t_update >= self.t_update:
                        # update & rebroadcast other's bid
                        return self.UPDATE,\
                        self.REBROADCAST_OTHER
                    else:
                        # update & rebroadcast other's bid
                        return self.UPDATE,\
                        self.REBROADCAST_OTHER
                
                elif other.winning_bid == self.winning_bid:
                    # leave & rebroadcast own information
                    return self.LEAVE,\
                        self.REBROADCAST_SELF
                
                elif other.winning_bid < self.winning_bid:
                    if other.t_update <= self.t_update:
                        # leave & rebroadcast own information
                        return self.LEAVE,\
                        self.REBROADCAST_SELF
                    else:
                        # update & rebroadcast other's bid
                        return self.UPDATE,\
                        self.REBROADCAST_OTHER

            elif self.winning_bidder is self.NONE:
                # update & rebroadcast other's bid
                return self.UPDATE,\
                    self.REBROADCAST_OTHER

        elif other.winning_bidder == self.bidder:
            if self.winning_bidder == self.bidder:
                if abs(other.t_update - self.t_update) < 1e-3:
                    # leave & no-rebroadcast
                    return self.LEAVE,\
                        self.NO_REBROADCAST

            elif self.winning_bidder == other.bidder:
                # reset & rebroadcast empty bid with current time
                return self.RESET,\
                    self.REBROADCAST_EMPTY

            elif self.winning_bidder not in [other.bidder, self.bidder, self.NONE]:
                # leave & rebroadcast own information
                return self.LEAVE,\
                    self.REBROADCAST_SELF

            elif self.winning_bidder is self.NONE:
                # leave & rebroadcast emtpy bid with current time
                return self.LEAVE,\
                    self.REBROADCAST_EMPTY

        elif other.winning_bidder not in [other.bidder, self.bidder, self.NONE]:
            if self.winning_bidder == self.bidder:
                if other.winning_bid > self.winning_bid:
                    # update & rebroadcast other's bid
                    return self.UPDATE,\
                        self.REBROADCAST_OTHER
                
                elif other.winning_bid == self.winning_bid and self != self._tie_breaker(other, self): 
                    # update & rebroadcast other's bid
                    return self.UPDATE,\
                        self.REBROADCAST_OTHER
                
                elif other.winning_bid < self.winning_bid:
                    # update time & rebroadcast own bid
                    return self.UPDATE_TIME,\
                        self.REBROADCAST_SELF
                    
            elif self.winning_bidder == other.bidder:
                # update & rebroadcast other's bid
                return self.UPDATE,\
                    self.REBROADCAST_OTHER
            
            elif self.winning_bidder == other.winning_bidder:
                if abs(other.t_update - self.t_update) < 1e-3:
                    # leave & no-rebroadcast
                    return self.LEAVE,\
                        self.NO_REBROADCAST
                
                elif other.t_update > self.t_update:
                    # update & rebroadcast other's bid
                    return self.UPDATE,\
                        self.REBROADCAST_OTHER
                
                elif other.t_update < self.t_update:
                    # leave & no-rebroadcast
                    return self.LEAVE,\
                        self.NO_REBROADCAST
                
            elif self.winning_bidder not in [other.bidder, self.bidder, other.winning_bidder, self.NONE]:
                if other.winning_bid > self.winning_bid:
                    if other.t_update >= self.t_update:
                        # update & rebroadcast other's bid
                        return self.UPDATE,\
                        self.REBROADCAST_OTHER

                    elif other.t_update < self.t_update:
                        # leave & rebroadcast own bid
                        return self.LEAVE,\
                        self.REBROADCAST_SELF

                elif other.winning_bid < self.winning_bid:
                    if other.t_update <= self.t_update:
                        # leave & rebroadcast own bid
                        return self.LEAVE,\
                        self.REBROADCAST_SELF
                    
                    elif other.t_update > self.t_update:
                        # update & rebroadcast other's bid
                        return self.UPDATE,\
                        self.REBROADCAST_OTHER
                
            elif self.winning_bidder is self.NONE:
                # update & rebroadcast other's bid
                return self.UPDATE,\
                    self.REBROADCAST_OTHER

        elif other.winning_bidder is self.NONE:
            if self.winning_bidder == self.bidder:
                # leave & rebroadcast own bid
                return self.LEAVE,\
                    self.REBROADCAST_SELF

            elif self.winning_bidder == other.bidder:
                # update & rebroadcast other's bid
                return self.UPDATE,\
                    self.REBROADCAST_OTHER

            elif self.winning_bidder not in [other.bidder, self.bidder, self.NONE]:
                if other.t_update > self.t_update:
                    # update & rebroadcast other's bid
                    return self.UPDATE,\
                        self.REBROADCAST_OTHER

            elif self.winning_bidder is self.NONE:
                # leave & no-rebroadcast
                return self.LEAVE,\
                    self.NO_REBROADCAST
            
        return self.LEAVE,\
            self.NO_REBROADCAST
    
    """
    ---------------------------
    MODIFIERS
    ---------------------------
    """

    def update(self, other : object, t : float) -> 'Bid':
        """ Returns a bid with the updated information after comparing this bid to another bid """
        # compare bids
        comp_result, _ = self.__compare(other)
        comp_result : str

        # update accordingly 
        new_bid : Bid = self.copy()
        if comp_result is self.UPDATE_TIME:
            new_bid.__update_time(t)
        elif comp_result is self.UPDATE:
            new_bid._update_info(other, t)
        elif comp_result is self.RESET:
            new_bid._reset(t)
        elif comp_result is self.LEAVE:
            new_bid._leave(t)
        elif comp_result is self.COMPLETED:
            new_bid._perform(t)
        else:
            raise ValueError(f'cannot perform update of type `{comp_result}`')
        
        # return updated bid
        return new_bid
    
    def __update_time(self, t_update : float) -> None:
        """Records the lastest time this bid was updated"""
        self.t_update = t_update
    
    def _update_info(self, other : 'Bid', t : float) -> None:
        """
        Updates all of the variable bid information

        ### Arguments:
            - other (`Bid`): equivalent bid being used to update information
        """
        # validate inputs
        assert isinstance(other, Bid), f'can only update bids from other bids.'
        assert self.task == other.task, f'cannot update bid with information from another bid intended for another task (expected task id: {self.task}, given id: {other.task}).'

        # update bid information
        self.winning_bid = other.winning_bid
        self.winning_bidder = other.winning_bidder
        self.t_img = other.t_img
        self.th_img = other.th_img

        self.t_update = t
        self.performed = other.performed if not self.performed else True # Check if this hold true for all values

    def _reset(self, t_update) -> None:
        """
        Resets the values of this bid while keeping track of lates update time
        """
        self.winning_bid = 0
        self.winning_bidder = self.NONE
        self.t_img = -1
        self.th_img = np.NAN
        self.t_update = t_update

    def _leave(self, _, **__) -> None:
        """
        Leaves bid as is (used for code readibility).

        ### Arguments:
            - t_update (`float` or `int`): latest time when this bid was updated
        """
        return
    
    def _perform(self, t_update : float) -> None:
        """ Indicates that this action has been performed """
        self.performed = True
        self.t_update = t_update
        
    def set(self, 
            new_bid : Union[int, float], 
            t_img : Union[int, float], 
            th_img : Union[int, float], 
            t_update : Union[int, float]
        ) -> None:
        """
        Sets new values for this bid

        ### Arguments: 
            - new_bid (`int` or `float`): new bid value
            - t_img (`int` or `float`): new imaging time
            - t_update (`int` or `float`): update time
        """
        self.winning_bid = new_bid
        self.winning_bidder = self.bidder
        self.t_img = t_img
        self.th_img = th_img
        self.t_update = t_update
    
    def has_winner(self) -> bool:
        """
        Checks if this bid has a winner
        """
        return self.winning_bidder != Bid.NONE

    def set_performed(self, t : float, performed : bool = True, performer : str = None) -> None:
        """
        Sets the performed status of this bid

        ### Arguments:
            - t (`float` or `int`): latest time when this bid was updated
            - performed (`bool`): indicates if the winner of this bid has performed the measurement request at hand
            - performer (`str`): name of the agent that performed the task
        """
        self.__update_time(t)
        self.winning_bidder = performer if performer is not None else self.bidder
        self.performed = performed
        self.t_img = t

    """
    ---------------------------
    STRING REPRESENTATION
    ---------------------------
    """

    def __str__(self) -> str:
        """
        Returns a string representation of this task bid in the following format:
        - `task_id`, `main_measurement`, `target`, `bidder`, `bid`, `winner`, `t_img`, `t_update`
        """
        
        split_id = self.task.id.split('-')
        line_data = {   "task_id" : split_id[0], 
                        "main_measurement" : self.main_measurement, 
                        "target" : self.task.location(), 
                        "bidder" : self.bidder, 
                        "bid" : round(self.winning_bid, 3), 
                        "winner" : self.winning_bidder, 
                        "t_img" : round(self.t_img, 3),
                        "t_update" : round(self.t_update, 3)
                    }
        out = "Bid("
        for key, value in line_data.items():
            out += f"{key}={value}, "
        out = out[:-2] + ")"

        return out
    
    def __repr__(self):
        task_id = self.task.id.split('-')
        return f'Bid_{task_id[0]}_{self.bidder}_{self.main_measurement}_{round(self.winning_bid,1)}'

    def __hash__(self) -> int:
        return hash(repr(self))       

# class BidBuffer(object):
#     """
#     Asynchronous buffer that holds bid information for use by processes within the MACCBBA
#     """
#     def __init__(self) -> None:
#         self.bid_access_lock = asyncio.Lock()
#         self.bid_buffer = {}
#         self.updated = asyncio.Event()             

#     def __len__(self) -> int:
#         l = 0
#         for req_id in self.bid_buffer:
#             for bid in self.bid_buffer[req_id]:
#                 bid : Bid
#                 l += 1 if bid is not None else 0
#         return l

#     async def pop_all(self) -> list:
#         """
#         Returns latest bids for all requests and empties buffer
#         """
#         await self.bid_access_lock.acquire()

#         out = []
#         for req_id in self.bid_buffer:
#             for bid in self.bid_buffer[req_id]:
#                 bid : Bid
#                 if bid is not None:
#                     # place bid in outgoing list
#                     out.append(bid)

#             # reset bids in buffer
#             self.bid_buffer[req_id] = [None for _ in self.bid_buffer[req_id]]

#         self.bid_access_lock.release()

#         return out

#     async def put_bid(self, new_bid : Bid) -> None:
#         """
#         Adds bid to the appropriate buffer if it's a more updated bid information than the one at hand
#         """
#         await self.bid_access_lock.acquire()

#         if new_bid.req_id not in self.bid_buffer:
#             req : TaskRequest = TaskRequest.from_dict(new_bid.req)
#             self.bid_buffer[new_bid.req_id] = [None for _ in req.dependency_matrix]

#         current_bid : Bid = self.bid_buffer[new_bid.req_id][new_bid.subtask_index]
        
#         if (    current_bid is None 
#                 or new_bid.bidder == current_bid.bidder
#                 or new_bid.t_update >= current_bid.t_update
#             ):
#             self.bid_buffer[new_bid.req_id][new_bid.subtask_index] = new_bid.copy()

#         self.bid_access_lock.release()

#         self.updated.set()
#         self.updated.clear()

#     async def put_bids(self, new_bids : list) -> None:
#         """
#         Adds bid to the appropriate buffer if it's a more updated bid information than the one at hand
#         """
#         if len(new_bids) == 0:
#             return

#         await self.bid_access_lock.acquire()

#         for new_bid in new_bids:
#             new_bid : Bid

#             if new_bid.req_id not in self.bid_buffer:
#                 req : TaskRequest = TaskRequest.from_dict(new_bid.req)
#                 self.bid_buffer[new_bid.req_id] = [None for _ in req.dependency_matrix]

#             current_bid : Bid = self.bid_buffer[new_bid.req_id][new_bid.subtask_index]

#             if (    current_bid is None 
#                  or (new_bid.bidder == current_bid.bidder and new_bid.t_update >= current_bid.t_update)
#                  or (new_bid.bidder != new_bid.NONE and current_bid.winning_bidder == new_bid.NONE and new_bid.t_update >= current_bid.t_update)
#                 ):
#                 self.bid_buffer[new_bid.req_id][new_bid.subtask_index] = new_bid.copy()

#         self.bid_access_lock.release()

#         self.updated.set()
#         self.updated.clear()

#     async def wait_for_updates(self, min_len : int = 1) -> list:
#         """
#         Waits for the contents of this buffer to be updated and to contain more updates than the given minimum
#         """
#         while True:
#             await self.updated.wait()

#             if len(self) >= min_len:
#                 break

#         return await self.pop_all()
