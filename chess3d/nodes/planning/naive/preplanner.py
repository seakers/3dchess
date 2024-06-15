import logging

from dmas.clocks import ClockConfig
from dmas.utils import runtime_tracker
from dmas.clocks import *
import pandas as pd

from chess3d.messages import ClockConfig
from messages import *

from nodes.planning.plan import Plan
from nodes.orbitdata import OrbitData, TimeInterval
from nodes.states import *
from nodes.actions import *
from chess3d.nodes.science.requests import *
from nodes.states import SimulationAgentState
from nodes.orbitdata import OrbitData
from nodes.planning.preplanners import AbstractPreplanner

class NaivePlanner(AbstractPreplanner):
    def __init__(self, 
                 horizon: float = np.Inf, 
                 period: float = np.Inf, 
                 logger: logging.Logger = None
                 ) -> None:
        """ Schedules observations based on the earliest feasible access point and broadcasts plan to all agents in the network """
        super().__init__(horizon, period, logger)

    @runtime_tracker
    def update_percepts(self, 
                        state : SimulationAgentState,
                        current_plan : Plan,
                        incoming_reqs: list, 
                        relay_messages: list, 
                        misc_messages : list,
                        completed_actions: list,
                        aborted_actions : list,
                        pending_actions : list
                        ) -> None:
        
        super().update_percepts(incoming_reqs, relay_messages, completed_actions)
    
    @runtime_tracker
    def _schedule_observations(self, 
                               state: SimulationAgentState, 
                               _: ClockConfig, 
                               orbitdata: OrbitData = None
                               ) -> list:
        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Naive planner not yet implemented for agents of type `{type(state)}.`')
        
        # compile access times for this planning horizon
        access_times, ground_points = self.calculate_access_times(state, orbitdata)
        access_times : list; ground_points : dict
        
        # generate plan
        observations = []
        while access_times:
            # get next available access interval
            grid_index, gp_index, instrument, _, t, th = access_times.pop()
            lat,lon = ground_points[grid_index][gp_index]
            target = [lat,lon,0.0]

            # find if feasible observation time exists 
            for i in range(len(t)):
                t_img = t[i]
                th_img = th[i]
                action = ObservationAction(instrument, target, th_img, t_img)

                # check feasibility given the prior obvservation
                if not observations:
                    # no prior observation exists, is feasible
                    observations.append(action)
                    break                
                
                # get the previous scheduled observation
                action_prev : ObservationAction = observations[-1]

                # estimate maneuver time
                dt_obs = action.t_start - action_prev.t_end
                dt_maneuver = abs(action.look_angle - action_prev.look_angle)/state.max_slew_rate

                # check feasibility
                if dt_maneuver <= dt_obs:
                    observations.append(action)
                    break

        return observations
    
    @runtime_tracker
    def calculate_access_times(self, state : SimulationAgentState, orbitdata : OrbitData) -> dict:
        # compile list of all ground points
        grid_data = {
                        (row[0], row[1], int(row[2]), int(row[3]))
                        for grid in orbitdata.grid_data
                        for row in grid.values
                    }
        
        # define planning horizon
        t_start = state.t
        t_end = self.plan.t_next+self.horizon

        # initiate accestimes 
        access_times = {}
        ground_points = {}

        # calculate accesstimes to all groundpoints
        for lat,lon,grid_index,gp_index in grid_data:
            for instrument in state.payload:

                accesses : pd.DataFrame \
                    = orbitdata.get_ground_point_accesses_future(lat,
                                                                 lon,
                                                                 instrument,
                                                                 t_start,
                                                                 t_end,
                                                                 grid_index,
                                                                 gp_index)
                    
                for t_index,_,_,lat_access,lon_access,_,th_img,*_ in accesses.values:
                    # make sure that the right ground pount data was queried
                    assert abs(round(lat,3) - lat_access) < 1e-9
                    assert abs(round(lon,3) - lon_access) < 1e-9
                    
                    # calculate observation time
                    t_img = t_index * orbitdata.time_step

                    # initialize dictionaries if needed
                    if grid_index not in access_times:
                        access_times[grid_index] = {}
                        ground_points[grid_index] = {}
                    if gp_index not in access_times[grid_index]:
                        access_times[grid_index][gp_index] = {instrument : [] 
                                                              for instrument in state.payload}
                        ground_points[grid_index][gp_index] = (lat_access, lon_access)

                    # compile time interval information 
                    found = False
                    for interval, t, th in access_times[grid_index][gp_index][instrument]:
                        interval : TimeInterval
                        t : list
                        th : list

                        if (interval.is_during(t_img - orbitdata.time_step) 
                            or interval.is_during(t_img + orbitdata.time_step)):
                            interval.extend(t_img)
                            t.append(t_img)
                            th.append(th_img)
                            found = True
                            break                        

                    if not found:
                        access_times[grid_index][gp_index][instrument].append([TimeInterval(t_img, t_img), [t_img], [th_img]])
        
        # convert to `list`
        access_times = [    (grid_index, gp_index, instrument, interval, t, th)
                            for grid_index in access_times
                            for gp_index in access_times[grid_index]
                            for instrument in access_times[grid_index][gp_index]
                            for interval, t, th in access_times[grid_index][gp_index][instrument]
                        ]
        # sort by observation time
        access_times.sort(key = lambda a: a[3],reverse=True)
        
        # return access times and grid information
        return access_times, ground_points
    
    @runtime_tracker
    def _schedule_broadcasts(self, 
                             state: SimulationAgentState, 
                             observations : list, 
                             orbitdata: OrbitData) -> list:
        # schedule relays 
        broadcasts : list = super()._schedule_broadcasts(state, observations, orbitdata)

        # gather observation plan to be sent out
        plan_out = [action.to_dict()
                    for action in observations
                    if isinstance(action,ObservationAction)]

        # check if observations exist in plan
        if plan_out:
            # find best path for broadcasts
            path, t_start = self._create_broadcast_path(state, orbitdata)

            # check feasibility of path found
            if t_start >= 0:
                # create plan message
                msg = PlanMessage(state.agent_name, state.agent_name, plan_out, state.t, path=path)
                
                # add to broadcast plan
                broadcasts.append(BroadcastMessageAction(msg.to_dict(),t_start))
        
        # return broadcast plan
        return broadcasts