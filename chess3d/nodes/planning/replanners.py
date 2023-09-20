from abc import ABC, abstractmethod
import logging
import pandas as pd

from nodes.science.reqs import *
from nodes.states import *
from nodes.orbitdata import OrbitData

class AbstractReplanner(ABC):
    """
    # Replanner    
    """
    @abstractmethod 
    def needs_replanning(   self, 
                            state : AbstractAgentState,
                            curent_plan : list,
                            incoming_reqs : list,
                            generated_reqs : list,
                            misc_messages : list
                        ) -> bool:
        """
        Returns `True` if the current plan needs replanning
        """

    @abstractmethod
    def revise_plan(    self, 
                        state : AbstractAgentState, 
                        current_plan : list,
                        incoming_reqs : list, 
                        generated_reqs : list,
                        misc_messages : list,
                        **kwargs
                    ) -> list:
        """
        Revises the current plan 
        """
        pass

    @abstractmethod
    def plan_from_path( self, 
                        state : SimulationAgentState, 
                        path : list,
                        **kwargs
                    ) -> list:
        """
        Generates a list of AgentActions from the current path.

        Agents look to move to their designated measurement target and perform the measurement.

        ## Arguments:
            - state (:obj:`SimulationAgentState`): state of the agent at the start of the path
            - path (`list`): list of tuples indicating the sequence of observations to be performed and time of observation
        """

    def _get_available_requests( self, 
                                state : SimulationAgentState, 
                                requests : list,
                                orbitdata : OrbitData
                                ) -> list:
        """
        Checks if there are any requests available to be performed

        ### Returns:
            - list containing all available and bidable tasks to be performed by the parent agent
        """
        available = []
        for req in requests:
            req : MeasurementRequest
            for subtask_index in range(len(req.measurements)):
                if self.__can_bid(state, req, subtask_index, orbitdata):
                    available.append((req, subtask_index))

        return available

    def __can_bid(self, 
                state : SimulationAgentState, 
                req : MeasurementRequest, 
                subtask_index : int, 
                orbitdata : OrbitData,
                planning_horizon : float = np.Inf
                ) -> bool:
        """
        Checks if an agent has the ability to bid on a measurement task
        """
        # check planning horizon
        if state.t + planning_horizon < req.t_start:
            return False

        # check capabilities - TODO: Replace with knowledge graph
        main_measurement = req.measurements[subtask_index]
        if main_measurement not in [instrument for instrument in state.payload]:
            return False 

        # check time constraints
        ## Constraint 1: task must be able to be performed during or after the current time
        if req.t_end < state.t:
            return False

        elif isinstance(req, GroundPointMeasurementRequest):
            if isinstance(state, SatelliteAgentState):
                # check if agent can see the request location
                lat,lon,_ = req.lat_lon_pos
                df : pd.DataFrame = orbitdata.get_ground_point_accesses_future(lat, lon, state.t).sort_values(by='time index')
                
                can_access = False
                if not df.empty:                
                    times = df.get('time index')
                    for time in times:
                        time *= orbitdata.time_step 

                        if state.t + planning_horizon < time:
                            break

                        if req.t_start <= time <= req.t_end:
                            # there exists an access time before the request's availability ends
                            can_access = True
                            break
                
                if not can_access:
                    return False
        
        return True

    def _calc_arrival_times(self, 
                            state : SimulationAgentState, 
                            req : MeasurementRequest, 
                            t_prev : Union[int, float],
                            orbitdata : OrbitData) -> float:
        """
        Estimates the quickest arrival time from a starting position to a given final position
        """
        if isinstance(req, GroundPointMeasurementRequest):
            # compute earliest time to the task
            if isinstance(state, SatelliteAgentState):
                t_imgs = []
                lat,lon,_ = req.lat_lon_pos
                df : pd.DataFrame = orbitdata.get_ground_point_accesses_future(lat, lon, t_prev)

                for _, row in df.iterrows():
                    t_img = row['time index'] * orbitdata.time_step
                    dt = t_img - state.t
                
                    # propagate state
                    propagated_state : SatelliteAgentState = state.propagate(t_img)

                    # compute off-nadir angle
                    thf = propagated_state.calc_off_nadir_agle(req)
                    dth = abs(thf - propagated_state.attitude[0])

                    # estimate arrival time using fixed angular rate TODO change to 
                    if dt >= dth / state.max_slew_rate: # TODO change maximum angular rate 
                        t_imgs.append(t_img)
                        
                return t_imgs if len(t_imgs) > 0 else [-1]

            elif isinstance(state, UAVAgentState):
                dr = np.array(req.pos) - np.array(state.pos)
                norm = np.sqrt( dr.dot(dr) )
                return [norm / state.max_speed + t_prev]

            else:
                raise NotImplementedError(f"arrival time estimation for agents of type `{type(state)}` is not yet supported.")

        else:
            raise NotImplementedError(f"cannot calculate imaging time for measurement requests of type {type(req)}")       


class FIFOReplanner(AbstractReplanner):
    
    def revise_plan(    self, 
                        state : AbstractAgentState, 
                        current_plan : list,
                        incoming_reqs : list, 
                        generated_reqs : list,
                        misc_messages : list,
                        orbitdata : OrbitData,
                        level : int = logging.DEBUG
                    ) -> list:

        # initialize plan
        path = []         
        
        # compile requests
        reqs = []
        for 

        available_reqs : list = self._get_available_requests( state, initial_reqs, orbitdata )

        if isinstance(state, SatelliteAgentState):
            # Generates a plan for observing GPs on a first-come first-served basis
            
            reqs = {req.id : req for req, _ in available_reqs}
            arrival_times = {req.id : {} for req, _ in available_reqs}

            for req, subtask_index in available_reqs:
                t_arrivals : list = self._calc_arrival_times(state, req, state.t, orbitdata)
                arrival_times[req.id][subtask_index] = t_arrivals
            
            path = []

            for req_id in arrival_times:
                for subtask_index in arrival_times[req_id]:
                    t_arrivals : list = arrival_times[req_id][subtask_index]
                    t_img = t_arrivals.pop(0)
                    req : MeasurementRequest = reqs[req_id]
                    path.append((req, subtask_index, t_img, req.s_max/len(req.measurements)))

            path.sort(key=lambda a: a[2])

            while True:
                
                conflict_free = True
                for i in range(len(path) - 1):
                    j = i + 1
                    req_i, _, t_i, __ = path[i]
                    req_j, subtask_index_j, t_j, s_j = path[j]

                    th_i = state.calc_off_nadir_agle(req_i)
                    th_j = state.calc_off_nadir_agle(req_j)

                    if abs(th_i - th_j) / state.max_slew_rate > t_j - t_i:
                        t_arrivals : list = arrival_times[req_j.id][subtask_index_j]
                        if len(t_arrivals) > 0:
                            t_img = t_arrivals.pop(0)

                            path[j] = (req_j, subtask_index_j, t_img, s_j)
                            path.sort(key=lambda a: a[2])
                        else:
                            #TODO remove request from path
                            raise Exception("Whoops. See Plan Initializer.")
                            path.pop(j) 
                        conflict_free = False
                        break

                if conflict_free:
                    break
                    
            out = '\n'
            for req, subtask_index, t_img, s in path:
                out += f"{req.id.split('-')[0]}\t{subtask_index}\t{np.round(t_img,3)}\t{np.round(s,3)}\n"
            # self.log(out,level)

            return self._plan_from_path(state, path, orbitdata, clock_config)
                
        else:
            raise NotImplementedError(f'initial planner for states of type `{type(state)}` not yet supported')