from orbitpy.util import Spacecraft

from dmas.clocks import ClockConfig
from dmas.utils import runtime_tracker
from dmas.clocks import *

from chess3d.agents.orbitdata import OrbitData, TimeInterval
from chess3d.agents.states import *
from chess3d.agents.actions import *
from chess3d.agents.science.requests import *
from chess3d.agents.states import SimulationAgentState
from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.planning.planner import AbstractPreplanner
from chess3d.messages import *

class NaivePlanner(AbstractPreplanner):
    """ Schedules observations based on the earliest feasible access point and broadcasts plan to all agents in the network """

    @runtime_tracker
    def _schedule_observations(self, 
                               state: SimulationAgentState, 
                               specs : object,
                               _: ClockConfig, 
                               orbitdata: OrbitData = None
                               ) -> list:
        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Naive planner not yet implemented for agents of type `{type(state)}.`')
        elif not isinstance(specs, Spacecraft):
            raise ValueError(f'`specs` needs to be of type `{Spacecraft}` for agents with states of type `{SatelliteAgentState}`')

        # compile access times for this planning horizon
        access_times, ground_points = self.calculate_access_times(state, specs, orbitdata)
        access_times : list; ground_points : dict

        # compile list of instruments available in payload
        payload : dict = {instrument.name: instrument for instrument in specs.instrument}
        
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self.collect_fov_specs(specs)
        
        # get pointing agility specifications
        adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
        if adcs_specs is None: raise ValueError('ADCS component specifications missing from agent specs object.')

        max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
        if max_slew_rate is None: raise ValueError('ADCS `maxRate` specification missing from agent specs object.')

        max_torque = float(adcs_specs['maxTorque']) if adcs_specs.get('maxTorque', None) is not None else None
        if max_torque is None: raise ValueError('ADCS `maxTorque` specification missing from agent specs object.')

        # generate plan
        observations = []
        while access_times:
            # get next available access interval
            grid_index, gp_index, instrument, _, t, th = access_times.pop()
            lat,lon = ground_points[grid_index][gp_index]
            target = [lat,lon,0.0]

            # check if agent has the payload to peform observation
            if instrument not in payload:
                continue

            # compare to previous measurement 
            if not observations:
                # no prior observation exists, compare with current state
                t_prev = state.t
                th_prev = state.attitude[0]
            else:
                # compare with previous scheduled observation
                action_prev : ObservationAction = observations[-1]
                t_prev = action_prev.t_end
                th_prev = action_prev.look_angle
          
            # find if feasible observation time exists 
            feasible_obs = [(t[i], th[i]) 
                            for i in range(len(t))
                            if self.is_observation_feasible(state, t[i], th[i], t_prev, th_prev, 
                                                            max_slew_rate, max_torque, 
                                                            cross_track_fovs[instrument])]
            
            if feasible_obs:
                # is feasible; create observation action
                t_img, th_img = feasible_obs[0]
                action = ObservationAction(instrument, target, th_img, t_img)

                # check if another observation was already scheduled at this time
                if observations:
                    action_prev : ObservationAction = observations[-1]
                    if abs(action_prev.t_start - action.t_start) <= 1e-3:
                        continue

                observations.append(action)

        assert self.no_redundant_observations(state, observations, orbitdata)

        return observations
    
    def is_observation_feasible(self, 
                                state : SimulationAgentState,
                                t_img : float, 
                                th_img : float, 
                                t_prev : float, 
                                th_prev : float, 
                                max_slew_rate : float, 
                                max_torque : float,
                                fov : float
                                ) -> bool:
        """ compares previous observation """
        # calculate inteval between observations
        dt_obs = t_img - t_prev

        # estimate maneuver time
        if abs(th_img - th_prev) <= fov / 2.0:
            # observations within the field-of-view of the instrument; no need to maneuver
            dth_img = 0
        else:
            # maneuver needed
            # dth_img = abs(th_img - th_prev)
            dth_img = abs(th_img - th_prev) + fov

        dt_maneuver = dth_img / max_slew_rate

        #TODO implement torque constraint

        # check feasibility
        return dt_maneuver <= dt_obs
    
    def no_redundant_observations(self, 
                                 state : SimulationAgentState, 
                                 observations : list,
                                 orbitdata : OrbitData
                                 ) -> bool:
        if isinstance(state, SatelliteAgentState):
            for j in range(len(observations)):
                i = j - 1

                if i < 0: # there was no prior observation performed
                    continue                

                observation_prev : ObservationAction = observations[i]
                observation_curr : ObservationAction = observations[j]

                if (
                    abs(observation_curr.target[0] - observation_prev.target[0]) <= 1e-3
                    and abs(observation_curr.target[1] - observation_prev.target[1]) <= 1e-3
                    and (observation_curr.t_start - observation_prev.t_end) <= orbitdata.time_step):
                    return False
            
            return True

        else:
            raise NotImplementedError(f'Measurement path validity check for agents with state type {type(state)} not yet implemented.')
            
    @runtime_tracker
    def calculate_access_times(self, 
                               state : SimulationAgentState, 
                               specs : Spacecraft,
                               orbitdata : OrbitData
                               ) -> dict:
        # define planning horizon
        t_start = state.t
        t_end = self.plan.t_next+self.horizon
        t_index_start = t_start / orbitdata.time_step
        t_index_end = t_end / orbitdata.time_step

        # compile coverage data
        orbitdata_columns : list = list(orbitdata.gp_access_data.columns.values)
        raw_coverage_data = [(t_index*orbitdata.time_step, *_)
                             for t_index, *_ in orbitdata.gp_access_data.values
                             if t_index_start <= t_index <= t_index_end]
                
        # initiate accestimes 
        access_times = {}
        ground_points = {}
        
        for data in raw_coverage_data:
            t_img = data[orbitdata_columns.index('time index')]
            grid_index = data[orbitdata_columns.index('grid index')]
            gp_index = data[orbitdata_columns.index('GP index')]
            lat = data[orbitdata_columns.index('lat [deg]')]
            lon = data[orbitdata_columns.index('lon [deg]')]
            instrument = data[orbitdata_columns.index('instrument')]
            look_angle = data[orbitdata_columns.index('look angle [deg]')]
            
            # initialize dictionaries if needed
            if grid_index not in access_times:
                access_times[grid_index] = {}
                ground_points[grid_index] = {}
            if gp_index not in access_times[grid_index]:
                access_times[grid_index][gp_index] = {instr.name : [] 
                                                        for instr in specs.instrument}
                ground_points[grid_index][gp_index] = (lat, lon)

            # compile time interval information 
            found = False
            for interval, t, th in access_times[grid_index][gp_index][instrument]:
                interval : TimeInterval
                t : list
                th : list

                if (   interval.is_during(t_img - orbitdata.time_step) 
                    or interval.is_during(t_img + orbitdata.time_step)):
                    interval.extend(t_img)
                    t.append(t_img)
                    th.append(look_angle)
                    found = True
                    break                        

            if not found:
                access_times[grid_index][gp_index][instrument].append([TimeInterval(t_img, t_img), [t_img], [look_angle]])

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
                
                # add plan broadcast to list of broadcasts
                broadcasts.append(BroadcastMessageAction(msg.to_dict(),t_start))
                
                # add action performance broadcast to plan
                for action_dict in plan_out:
                    path, t_start = self._create_broadcast_path(state, orbitdata, action_dict['t_end'])
                    msg = ObservationPerformedMessage(state.agent_name, state.agent_name, action_dict)
                    if t_start >= 0: broadcasts.append(BroadcastMessageAction(msg.to_dict(),t_start))

        # return broadcast plan
        return broadcasts