import copy
import json
import os
import random
import re
import time
import pandas as pd
import numpy as np

from datetime import timedelta
from orbitpy.mission import Mission

class TimeInterval:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        if self.end < self.start:
            raise Exception('The end of time interval must be later than beginning of the interval.')

    def is_after(self, t):
        return t < self.start

    def is_before(self, t):
        return self.end < t

    def is_during(self, t):
        return self.start <= t <= self.end
    
    def has_overlap(self, __other : object) -> bool:
        """ checks if a this time interval has an overlap with another """
        return (    __other.start <= self.start <= __other.end
                or  __other.start <= self.end <= __other.end 
                or  (self.start <= __other.start and __other.end <= self.end)) 

    def merge(self, __other : object) -> object:
        """ merges two time intervals and returns their intersect """

        __other : TimeInterval
        if not self.has_overlap(__other):
            raise ValueError("cannot merge two time intervals with no overlap")

        self.start = max(self.start, __other.start)
        self.end = min(self.min, __other.min)

    def extend(self, t: float) -> None:
        """ extends time interval """

        if t < self.start:
            self.start = t
        elif t > self.end:
            self.end = t

    def duration(self) -> int:
        return self.end - self.start

    def __eq__(self, __value: object) -> bool:
        return abs(self.start - __value.start) < 1e-6 and abs(self.end - __value.end) < 1e-6

    def __gt__(self, __value: object) -> bool:
        if abs(self.start - __value.start) < 1e-6:
            return self.duration() > __value.duration()
        
        return self.start > __value.start

    def __ge__(self, __value: object) -> bool:
        if abs(self.start - __value.start) < 1e-6:
            return self.duration() >= __value.duration()
        
        return self.start >= __value.start
    
    def __lt__(self, __value: object) -> bool:
        if abs(self.start - __value.start) < 1e-6:
            return self.duration() < __value.duration()
        
        return self.start < __value.start

    def __le__(self, __value: object) -> bool:
        if abs(self.start - __value.start) < 1e-6:
            return self.duration() <= __value.duration()
        
        return self.start <= __value.start
    
    def __repr__(self) -> str:
        return f'TimeInterval({self.start},{self.end})'
    
    def __hash__(self) -> int:
        return hash(str(self))

class OrbitData:
    """
    Stores and queries data regarding an agent's orbital data. 

    TODO: add support to load ground station agents' data
    """
    def __init__(self, agent_name : str, 
                 time_data : pd.DataFrame, 
                 eclipse_data : pd.DataFrame, 
                 position_data : pd.DataFrame, 
                 isl_data : dict,
                 gs_access_data : pd.DataFrame, 
                 gp_access_data : pd.DataFrame, 
                 grid_data : list
                ):
        # name of agent being represented by this object
        self.agent_name = agent_name

        # propagation time specifications
        self.time_step = time_data['time step']
        self.epoc_type = time_data['epoc type']
        self.epoc = time_data['epoc']
        self.duration = time_data['duration']

        # agent position and eclipse information
        self.eclipse_data = eclipse_data.sort_values(by=['start index'])
        self.position_data = position_data.sort_values(by=['time index'])

        # inter-satellite communication access times
        self.isl_data = { satellite_name : isl_data[satellite_name].sort_values(by=['start index']) 
                         for satellite_name in isl_data }
        
        # ground station access times
        self.gs_access_data = gs_access_data.sort_values(by=['start index'])
        
        # ground point access times
        self.gp_access_data = gp_access_data.sort_values(by=['time index'])

        # grid information
        self.grid_data = grid_data
    
    def copy(self) -> object:
        return OrbitData(self.agent_name, 
                         {'time step': self.time_step, 'epoc type' : self.epoc_type, 'epoc' : self.epoc},
                         self.eclipse_data,
                         self.position_data,
                         self.isl_data,
                         self.gs_access_data,
                         self.gp_access_data,
                         self.grid_data
                         )
    
    def update_databases(self, t : float) -> None:
        # calculate time-step
        t_index = t / self.time_step 

        # exclude outdated data
        self.eclipse_data = self.eclipse_data[self.eclipse_data['end index'] >= t_index - 1]
        self.position_data = self.position_data[self.position_data['time index'] >= t_index - 1]
        self.isl_data = {   satellite_name : self.isl_data[satellite_name][self.isl_data[satellite_name]['end index'] >= t_index - 1]
                            for satellite_name in self.isl_data}
        self.gs_access_data = self.gs_access_data[self.gs_access_data['end index'] >= t_index - 1]
        self.gp_access_data = self.gp_access_data[self.gp_access_data['time index'] >= t_index - 1]

    """
    GET NEXT methods
    """
    def get_next_agent_access(self, target : str, t: float):
        src = self.agent_name

        if target in self.isl_data.keys():
            return self.get_next_isl_access_interval(target, t)
        else:
            raise ValueError(f'{src} cannot access {target}.')

    def get_next_isl_access_interval(self, target : str, t : float) -> TimeInterval:
        t = t/self.time_step
        isl_data : pd.DataFrame = self.isl_data[target]
        isl_access : pd.DataFrame = isl_data.query('@t <= `end index`').sort_values('start index')
        
        for _, row in isl_access.iterrows():
            t_start = max(t, row['start index']) * self.time_step
            t_end = row['end index'] * self.time_step

            return TimeInterval(t_start, t_end)

        return TimeInterval(np.Inf, np.Inf)

    def get_next_gs_access(self, t):
        t = t/self.time_step
        accesses = self.gp_access_data.query('`time index` >= @t').sort_values(by='time index')        
        for _, row in accesses.iterrows():
            return row['time index'] * self.time_step
        return np.Inf

    def get_next_gp_access_interval(self, lat: float, lon: float, t: float):
        """
        Returns the next access to a ground point
        """
        # find closest gridpoint 
        grid_index, gp_index, _, _ = self.find_gp_index(lat, lon)

        # find next access
        # TODO
        raise NotImplementedError('TODO: need to implement.')

        interval = TimeInterval(-np.Inf, np.Inf)
        instruments = []
        modes = dict()
        return interval, instruments, modes

    def get_next_eclipse_interval(self, t: float):
        for _, row in self.eclipse_data.iterrows():
            t_start = row['start index'] * self.time_step
            t_end = row['end index'] * self.time_step
            if t_end <= t:
                continue
            elif t < t_start:
                return t_start
            elif t < t_end:
                return t_end
        return np.Infinity

    """
    STATE QUERY methods
    """
    def is_accessing_agent(self, target: str, t: float) -> bool:
        if target in self.agent_name:
            return True

        if target not in self.isl_data.keys():
            return False 

        t = t/self.time_step

        return any([t_start <= t <= t_end for t_start,t_end in self.isl_data[target].values])

    def is_accessing_ground_station(self, target : str, t: float) -> bool:
        t = t/self.time_step
        nrows, _ = self.gs_access_data.query('`start index` <= @t & @t <= `end index` & `gndStn name` == @target').shape
        return bool(nrows > 0)

    def is_eclipse(self, t: float):
        """ checks if a satellite is currently in eclise at time `t`. """
        t_e = t / self.time_step

        return any([t_start <= t_e <= t_end for t_start,t_end in self.eclipse_data.values])

    def get_position(self, t: float):
        pos, _, _ = self.get_orbit_state(t)
        return pos

    def get_velocity(self, t: float):
        _, vel, _ = self.get_orbit_state(t)
        return vel
        
    def get_orbit_state(self, t: float):
        is_eclipse = self.is_eclipse(t)

        t = t / self.time_step
        t_u = t + 0.5
        t_l = t - 0.5

        data = [(t_i,x,y,z,vx,vy,vz) 
                for t_i,x,y,z,vx,vy,vz in self.position_data.values
                if t_l < t_i <= t_u]
        
        if not data:
            x = 1

        _,x,y,z,vx,vy,vz = data[0]
        pos = [x, y, z]
        vel = [vx, vy, vz]
        
        return (pos, vel, is_eclipse)

    """
    LOAD FROM PRE-COMPUTED DATA
    """
    def load(orbitdata_path : str, agent_name : str) -> object:
        """
        Loads agent orbit data from pre-computed csv files in scenario directory
        """
        with open(os.path.join(orbitdata_path, 'MissionSpecs.json'), 'r') as mission_specs:
            # load json file as dictionary
            mission_dict : dict = json.load(mission_specs)
            spacecraft_list : list = mission_dict.get('spacecraft', None)
            ground_station_list = mission_dict.get('groundStation', None)
            
            if agent_name in [spacecraft.get('name') for spacecraft in spacecraft_list]:
                return OrbitData.load_spacecraft_data(agent_name, spacecraft_list, ground_station_list, orbitdata_path, mission_dict)
            elif agent_name in [gstat.get('name') for gstat in ground_station_list]:
                return OrbitData.load_gstat_data(agent_name, spacecraft_list, ground_station_list, orbitdata_path, mission_dict)
            else:
                raise ValueError(f'Orbitdata for agent `{agent_name}` not found in precomputed data.')
            
    def load_spacecraft_data(
                             agent_name : str, 
                             spacecraft_list : list, 
                             ground_station_list: list,
                             orbitdata_path : str,
                             mission_dict : dict
                             ) -> object:
        for spacecraft in spacecraft_list:
            spacecraft : dict
            name = spacecraft.get('name')
            index = spacecraft_list.index(spacecraft)
            agent_folder = "sat" + str(index) + '/'

            if name != agent_name:
                continue

            # load eclipse data
            eclipse_file = os.path.join(orbitdata_path, agent_folder, "eclipses.csv")
            eclipse_data = pd.read_csv(eclipse_file, skiprows=range(3))
            
            # load position data
            position_file = os.path.join(orbitdata_path, agent_folder, "state_cartesian.csv")
            position_data = pd.read_csv(position_file, skiprows=range(4))

            # load propagation time data
            time_data =  pd.read_csv(position_file, nrows=3)
            _, epoc_type, _, epoc = time_data.at[0,time_data.axes[1][0]].split(' ')
            epoc_type = epoc_type[1 : -1]
            epoc = float(epoc)
            _, _, _, _, time_step = time_data.at[1,time_data.axes[1][0]].split(' ')
            time_step = float(time_step)
            _, _, _, _, duration = time_data.at[2,time_data.axes[1][0]].split(' ')
            duration = float(duration)

            time_data = { "epoc": epoc, 
                        "epoc type": epoc_type, 
                        "time step": time_step,
                        "duration" : duration }

            # load inter-satellite link data
            isl_data = dict()
            comms_path = os.path.join(orbitdata_path, 'comm')
            for file in os.listdir(comms_path):                
                isl = re.sub(".csv", "", file)
                sender, _, receiver = isl.split('_')

                if 'sat' + str(index) in sender or 'sat' + str(index) in receiver:
                    isl_file = os.path.join(comms_path, file)
                    if 'sat' + str(index) in sender:
                        receiver_index = int(re.sub("[^0-9]", "", receiver))
                        receiver_name = spacecraft_list[receiver_index].get('name')
                        if (
                            (scenario_dict := mission_dict.get('scenario', None)) 
                            and scenario_dict.get('connectivity', None).upper() == "FULL"
                            ):
                            # modify connectivity if specified 
                            columns = ['start index', 'end index']
                            duration = timedelta(days=float(mission_dict["duration"]))
                            data = [[0.0, duration.total_seconds()]]
                            assert data[0][1] > 0.0
                            isl_data[receiver_name] = pd.DataFrame(data=data, columns=columns)
                        else:
                            # load connectivity
                            isl_data[receiver_name] = pd.read_csv(isl_file, skiprows=range(3))
                    else:
                        sender_index = int(re.sub("[^0-9]", "", sender))
                        sender_name = spacecraft_list[sender_index].get('name')
                        if (
                            (scenario_dict := mission_dict.get('scenario', None)) 
                            and scenario_dict.get('connectivity', None).upper() == "FULL"
                            ):
                            # modify connectivity if specified 
                            columns = ['start index', 'end index']
                            duration = timedelta(days=float(mission_dict["duration"]))
                            data = [[0.0, duration.total_seconds()]]
                            assert data[0][1] > 0.0
                            isl_data[sender_name] = pd.DataFrame(data=data, columns=columns)
                        else:
                            # load connectivity
                            isl_data[sender_name] = pd.read_csv(isl_file, skiprows=range(3))

            # load ground station access data
            gs_access_data = pd.DataFrame(columns=['start index', 'end index', 'gndStn id', 'gndStn name','lat [deg]','lon [deg]'])
            agent_orbitdata_path = os.path.join(orbitdata_path, agent_folder)
            for file in os.listdir(agent_orbitdata_path):
                if 'gndStn' in file:
                    gndStn_access_file = os.path.join(orbitdata_path, agent_folder, file)
                    gndStn_access_data = pd.read_csv(gndStn_access_file, skiprows=range(3))
                    nrows, _ = gndStn_access_data.shape

                    if nrows > 0:
                        gndStn, _ = file.split('_')
                        gndStn_index = int(re.sub("[^0-9]", "", gndStn))
                        
                        gndStn_name = ground_station_list[gndStn_index].get('name')
                        gndStn_id = ground_station_list[gndStn_index].get('@id')
                        gndStn_lat = ground_station_list[gndStn_index].get('latitude')
                        gndStn_lon = ground_station_list[gndStn_index].get('longitude')

                        gndStn_name_column = [gndStn_name] * nrows
                        gndStn_id_column = [gndStn_id] * nrows
                        gndStn_lat_column = [gndStn_lat] * nrows
                        gndStn_lon_column = [gndStn_lon] * nrows

                        gndStn_access_data['gndStn name'] = gndStn_name_column
                        gndStn_access_data['gndStn id'] = gndStn_id_column
                        gndStn_access_data['lat [deg]'] = gndStn_lat_column
                        gndStn_access_data['lon [deg]'] = gndStn_lon_column

                        if len(gs_access_data) == 0:
                            gs_access_data = gndStn_access_data
                        else:
                            gs_access_data = pd.concat([gs_access_data, gndStn_access_data])

            # land coverage data metrics data
            payload = spacecraft.get('instrument', None)
            if not isinstance(payload, list):
                payload = [payload]

            gp_access_data = pd.DataFrame(columns=['time index','GP index','pnt-opt index','lat [deg]','lon [deg]', 'agent','instrument',
                                                            'observation range [km]','look angle [deg]','incidence angle [deg]','solar zenith [deg]'])

            for instrument in payload:
                if instrument is None: continue 

                i_ins = payload.index(instrument)
                gp_acces_by_mode = []

                # TODO implement different viewing modes for payloads
                # modes = spacecraft.get('instrument', None)
                # if not isinstance(modes, list):
                #     modes = [0]
                modes = [0]

                gp_acces_by_mode = pd.DataFrame(columns=['time index','GP index','pnt-opt index','lat [deg]','lon [deg]','instrument',
                                                            'observation range [km]','look angle [deg]','incidence angle [deg]','solar zenith [deg]'])
                for mode in modes:
                    i_mode = modes.index(mode)
                    gp_access_by_grid = pd.DataFrame(columns=['time index','GP index','pnt-opt index','lat [deg]','lon [deg]',
                                                            'observation range [km]','look angle [deg]','incidence angle [deg]','solar zenith [deg]'])

                    for grid in mission_dict.get('grid'):
                        i_grid = mission_dict.get('grid').index(grid)
                        metrics_file = os.path.join(orbitdata_path, agent_folder, f'datametrics_instru{i_ins}_mode{i_mode}_grid{i_grid}.csv')
                        
                        try:
                            metrics_data = pd.read_csv(metrics_file, skiprows=range(4))
                            
                            nrows, _ = metrics_data.shape
                            grid_id_column = [i_grid] * nrows
                            metrics_data['grid index'] = grid_id_column

                            if len(gp_access_by_grid) == 0:
                                gp_access_by_grid = metrics_data
                            else:
                                gp_access_by_grid = pd.concat([gp_access_by_grid, metrics_data])
                        except pd.errors.EmptyDataError:
                            continue

                    nrows, _ = gp_access_by_grid.shape
                    gp_access_by_grid['pnt-opt index'] = [mode] * nrows

                    if len(gp_acces_by_mode) == 0:
                        gp_acces_by_mode = gp_access_by_grid
                    else:
                        gp_acces_by_mode = pd.concat([gp_acces_by_mode, gp_access_by_grid])
                    # gp_acces_by_mode.append(gp_access_by_grid)

                nrows, _ = gp_acces_by_mode.shape
                gp_access_by_grid['instrument'] = [instrument['name']] * nrows
                # gp_access_data[ins_name] = gp_acces_by_mode

                if len(gp_access_data) == 0:
                    gp_access_data = gp_acces_by_mode
                else:
                    gp_access_data = pd.concat([gp_access_data, gp_acces_by_mode])
            
            nrows, _ = gp_access_data.shape
            gp_access_data['agent name'] = [spacecraft['name']] * nrows

            grid_data_compiled = []
            for grid in mission_dict.get('grid'):
                grid : dict
                i_grid = mission_dict.get('grid').index(grid)
                
                if grid.get('@type').lower() == 'customgrid':
                    grid_file = grid.get('covGridFilePath')
                    
                elif grid.get('@type').lower() == 'autogrid':
                    grid_file = os.path.join(orbitdata_path, f'grid{i_grid}.csv')
                else:
                    raise NotImplementedError(f"Loading of grids of type `{grid.get('@type')} not yet supported.`")

                grid_data = pd.read_csv(grid_file)
                nrows, _ = grid_data.shape
                grid_data['grid index'] = [i_grid] * nrows
                grid_data['GP index'] = [i for i in range(nrows)]
                grid_data_compiled.append(grid_data)

            return OrbitData(name, time_data, eclipse_data, position_data, isl_data, gs_access_data, gp_access_data, grid_data_compiled)
        
        raise ValueError(f'Orbitdata for satellite `{agent_name}` not found in precalculated data.')
    
    def load_gstat_data(
                             agent_name : str, 
                             spacecraft_list : list, 
                             ground_station_list: list,
                             orbitdata_path : str,
                             mission_dict : dict
                             ) -> object:
        raise NotImplementedError('not implemented yet')
        for gstat in ground_station_list:
            spacecraft : dict
            name = gstat.get('name')
            index = ground_station_list.index(gstat)
            agent_folder = "sat" + str(index) + '/'

            if name != agent_name:
                continue

            # load eclipse data
            eclipse_data = pd.DataFrame(columns=['start index', 'end index'])
            
            # load position data
            position_data = pd.DataFrame(columns=['time index','x [km]','y [km]','z [km]','vx [km/s]','vy [km/s]','vz [km/s]'])

            # # load propagation time data

            # time_data =  pd.read_csv(position_file, nrows=3)
            # _, epoc_type, _, epoc = time_data.at[0,time_data.axes[1][0]].split(' ')
            # epoc_type = epoc_type[1 : -1]
            # epoc = float(epoc)
            # _, _, _, _, time_step = time_data.at[1,time_data.axes[1][0]].split(' ')
            # time_step = float(time_step)
            # _, _, _, _, duration = time_data.at[2,time_data.axes[1][0]].split(' ')
            # duration = float(duration)

            # time_data = { "epoc": epoc, 
            #             "epoc type": epoc_type, 
            #             "time step": time_step,
            #             "duration" : duration }

            # load inter-satellite link data
            isl_data = dict()
            
            # load ground station access data
            gs_access_data = pd.DataFrame(columns=['start index', 'end index', 'gndStn id', 'gndStn name','lat [deg]','lon [deg]'])
            agent_orbitdata_path = os.path.join(orbitdata_path, agent_folder)
            for file in os.listdir(agent_orbitdata_path):
                if 'gndStn' in file:
                    gndStn_access_file = os.path.join(orbitdata_path, agent_folder, file)
                    gndStn_access_data = pd.read_csv(gndStn_access_file, skiprows=range(3))
                    nrows, _ = gndStn_access_data.shape

                    if nrows > 0:
                        gndStn, _ = file.split('_')
                        gndStn_index = int(re.sub("[^0-9]", "", gndStn))
                        
                        gndStn_name = ground_station_list[gndStn_index].get('name')
                        gndStn_id = ground_station_list[gndStn_index].get('@id')
                        gndStn_lat = ground_station_list[gndStn_index].get('latitude')
                        gndStn_lon = ground_station_list[gndStn_index].get('longitude')

                        gndStn_name_column = [gndStn_name] * nrows
                        gndStn_id_column = [gndStn_id] * nrows
                        gndStn_lat_column = [gndStn_lat] * nrows
                        gndStn_lon_column = [gndStn_lon] * nrows

                        gndStn_access_data['gndStn name'] = gndStn_name_column
                        gndStn_access_data['gndStn id'] = gndStn_id_column
                        gndStn_access_data['lat [deg]'] = gndStn_lat_column
                        gndStn_access_data['lon [deg]'] = gndStn_lon_column

                        if len(gs_access_data) == 0:
                            gs_access_data = gndStn_access_data
                        else:
                            gs_access_data = pd.concat([gs_access_data, gndStn_access_data])

            # land coverage data metrics data
            payload = spacecraft.get('instrument', None)
            if not isinstance(payload, list):
                payload = [payload]

            gp_access_data = pd.DataFrame(columns=['time index','GP index','pnt-opt index','lat [deg]','lon [deg]', 'agent','instrument',
                                                            'observation range [km]','look angle [deg]','incidence angle [deg]','solar zenith [deg]'])

            for instrument in payload:
                i_ins = payload.index(instrument)
                gp_acces_by_mode = []

                # TODO implement different viewing modes for payloads
                # modes = spacecraft.get('instrument', None)
                # if not isinstance(modes, list):
                #     modes = [0]
                modes = [0]

                gp_acces_by_mode = pd.DataFrame(columns=['time index','GP index','pnt-opt index','lat [deg]','lon [deg]','instrument',
                                                            'observation range [km]','look angle [deg]','incidence angle [deg]','solar zenith [deg]'])
                for mode in modes:
                    i_mode = modes.index(mode)
                    gp_access_by_grid = pd.DataFrame(columns=['time index','GP index','pnt-opt index','lat [deg]','lon [deg]',
                                                            'observation range [km]','look angle [deg]','incidence angle [deg]','solar zenith [deg]'])

                    for grid in mission_dict.get('grid'):
                        i_grid = mission_dict.get('grid').index(grid)
                        metrics_file = os.path.join(orbitdata_path, agent_folder, f'datametrics_instru{i_ins}_mode{i_mode}_grid{i_grid}.csv')
                        metrics_data = pd.read_csv(metrics_file, skiprows=range(4))
                        
                        nrows, _ = metrics_data.shape
                        grid_id_column = [i_grid] * nrows
                        metrics_data['grid index'] = grid_id_column

                        if len(gp_access_by_grid) == 0:
                            gp_access_by_grid = metrics_data
                        else:
                            gp_access_by_grid = pd.concat([gp_access_by_grid, metrics_data])

                    nrows, _ = gp_access_by_grid.shape
                    gp_access_by_grid['pnt-opt index'] = [mode] * nrows

                    if len(gp_acces_by_mode) == 0:
                        gp_acces_by_mode = gp_access_by_grid
                    else:
                        gp_acces_by_mode = pd.concat([gp_acces_by_mode, gp_access_by_grid])
                    # gp_acces_by_mode.append(gp_access_by_grid)

                nrows, _ = gp_acces_by_mode.shape
                gp_access_by_grid['instrument'] = [instrument['name']] * nrows
                # gp_access_data[ins_name] = gp_acces_by_mode

                if len(gp_access_data) == 0:
                    gp_access_data = gp_acces_by_mode
                else:
                    gp_access_data = pd.concat([gp_access_data, gp_acces_by_mode])
            
            nrows, _ = gp_access_data.shape
            gp_access_data['agent name'] = [spacecraft['name']] * nrows

            grid_data_compiled = []
            for grid in mission_dict.get('grid'):
                grid : dict
                if grid.get('@type').lower() == 'customgrid':
                    grid_file = grid.get('covGridFilePath')
                    
                elif grid.get('@type').lower() == 'autogrid':
                    i_grid = mission_dict.get('grid').index(grid)
                    grid_file = os.path.join(orbitdata_path, f'grid{i_grid}.csv')
                else:
                    raise NotImplementedError(f"Loading of grids of type `{grid.get('@type')} not yet supported.`")

                grid_data = pd.read_csv(grid_file)
                nrows, _ = grid_data.shape
                grid_data['grid index'] = [i_grid] * nrows
                grid_data['GP index'] = [i for i in range(nrows)]
                grid_data_compiled.append(grid_data)

            return OrbitData(name, time_data, eclipse_data, position_data, isl_data, gs_access_data, gp_access_data, grid_data_compiled)
        
        raise ValueError(f'Orbitdata for satellite `{agent_name}` not found in precalculated data.')
    
    def from_directory(orbitdata_dir: str):
        """
        Loads orbit data from a directory containig a json file specifying the details of the mission being simulated.
        If the data has not been previously propagated, it will do so and store it in the same directory as the json file
        being used.

        The data gets stored as a dictionary, with each entry containing the orbit data of each agent in the mission 
        indexed by the name of the agent.
        """
        orbitdata_specs : str = os.path.join(orbitdata_dir, 'MissionSpecs.json')
        with open(orbitdata_specs, 'r') as scenario_specs:
            
            # load json file as dictionary
            mission_dict : dict = json.load(scenario_specs)
            data = dict()
            spacecraft_list : list = mission_dict.get('spacecraft', None)
            uav_list : list = mission_dict.get('uav', None)
            ground_station_list = mission_dict.get('groundStation', None)

            # load pre-computed data
            if spacecraft_list:
                for spacecraft in spacecraft_list:
                    spacecraft : dict
                    agent_name = spacecraft.get('name')

                    data[agent_name] = OrbitData.load(orbitdata_dir, agent_name)

            if uav_list:
                raise NotImplementedError('Orbitdata for UAVs not yet supported')

            if ground_station_list:
                for ground_station in ground_station_list:
                    ground_station : dict
                    agent_name = ground_station.get('name')

                    data[agent_name] = OrbitData.load(orbitdata_dir, agent_name)

                raise NotImplementedError('Orbitdata for ground stations not yet supported')

            return data
               
    def precompute(scenario_specs : dict) -> str:
        """
        Pre-calculates coverage and position data for a given scenario
        """
        
        # get desired orbit data path
        scenario_dir = scenario_specs['scenario']['scenarioPath']
        settings_dict : dict = scenario_specs.get('settings', None)
        if settings_dict is None:
            data_dir = None
        else:
            data_dir = settings_dict.get('outDir', None)

        if data_dir is None:
            data_dir = os.path.join(scenario_dir, 'orbit_data')
    
        if not os.path.exists(data_dir):
            # if directory does not exists, create it
            os.mkdir(data_dir)
            changes_to_scenario = True
        else:
            changes_to_scenario : bool = OrbitData._check_changes_to_scenario(scenario_specs, data_dir)

        if not changes_to_scenario:
            # if propagation data files already exist, load results
            print('Orbit data found!')
        else:
            # if propagation data files do not exist, propagate and then load results
            if os.path.exists(data_dir):
                print('Existing orbit data does not match scenario.')
            else:
                print('Orbit data not found.')

            # clear files if they exist
            print('Clearing \'orbitdata\' directory...')    
            if os.path.exists(data_dir):
                for f in os.listdir(data_dir):
                    f_dir = os.path.join(data_dir, f)
                    if os.path.isdir(f_dir):
                        for h in os.listdir(f_dir):
                            os.remove(os.path.join(f_dir, h))
                        os.rmdir(f_dir)
                    else:
                        os.remove(f_dir) 
            print('\'orbitdata\' cleared!')

            # set grid 
            grid_dicts : list = scenario_specs.get("grid", None)
            for grid_dict in grid_dicts:
                grid_dict : dict
                if grid_dict is not None:
                    grid_type : str = grid_dict.get('@type', None)
                    
                    if grid_type.lower() == 'customgrid':
                        # do nothing
                        pass
                    elif grid_type.lower() == 'uniform':
                        # create uniform grid
                        lat_spacing = grid_dict.get('lat_spacing', 1)
                        lon_spacing = grid_dict.get('lon_spacing', 1)
                        grid_index  = grid_dicts.index(grid_dict)
                        grid_path : str = OrbitData._create_uniform_grid(scenario_dir, grid_index, lat_spacing, lon_spacing)

                        # set to customgrid
                        grid_dict['@type'] = 'customgrid'
                        grid_dict['covGridFilePath'] = grid_path
                        
                    elif grid_type.lower() in ['cluster', 'clustered']:
                        # create clustered grid
                        n_clusters          = grid_dict.get('n_clusters', 100)
                        n_cluster_points    = grid_dict.get('n_cluster_points', 1)
                        variance            = grid_dict.get('variance', 1)
                        grid_index          = grid_dicts.index(grid_dict)
                        grid_path : str = OrbitData._create_clustered_grid(scenario_dir, grid_index, n_clusters, n_cluster_points, variance)

                        # set to customgrid
                        grid_dict['@type'] = 'customgrid'
                        grid_dict['covGridFilePath'] = grid_path
                        
                    else:
                        raise ValueError(f'Grids of type \'{grid_type}\' not supported.')
                else:
                    pass
            scenario_specs['grid'] = grid_dicts

            # set output directory to orbit data directory
            if scenario_specs.get("settings", None) is not None:
                if scenario_specs["settings"].get("outDir", None) is None:
                    scenario_specs["settings"]["outDir"] = scenario_dir + '/orbit_data/'
            else:
                scenario_specs["settings"] = {}
                scenario_specs["settings"]["outDir"] = scenario_dir + '/orbit_data/'

            # propagate data and save to orbit data directory
            print("Propagating orbits...")
            mission : Mission = Mission.from_json(scenario_specs)  
            mission.execute()                
            print("Propagation done!")

            # save specifications of propagation in the orbit data directory
            with open(os.path.join(data_dir,'MissionSpecs.json'), 'w') as mission_specs:
                mission_specs.write(json.dumps(scenario_specs, indent=4))

        return data_dir
    
    def _check_changes_to_scenario(scenario_dict : dict, orbitdata_dir : str) -> bool:
        """ 
        Checks if the scenario has already been pre-computed 
        or if relevant changes have been made 
        """
        # check if directory exists
        filename = 'MissionSpecs.json'
        orbitdata_filename = os.path.join(orbitdata_dir, filename)
        if not os.path.exists(orbitdata_filename):
            return True
        
        # copy scenario specs
        scenario_specs : dict = copy.deepcopy(scenario_dict)
            
        # compare specifications
        with open(orbitdata_filename, 'r') as orbitdata_specs:
            orbitdata_dict : dict = json.load(orbitdata_specs)

            scenario_specs.pop('settings')
            orbitdata_dict.pop('settings')
            scenario_specs.pop('scenario')
            orbitdata_dict.pop('scenario')

            if (
                    scenario_specs['epoch'] != orbitdata_dict['epoch']
                or scenario_specs['duration'] > orbitdata_dict['duration']
                or scenario_specs.get('groundStation', None) != orbitdata_dict.get('groundStation', None)
                # or scenario_dict['grid'] != orbitdata_dict['grid']
                # or scenario_dict['scenario']['connectivity'] != mission_dict['scenario']['connectivity']
                ):
                return True
            
            if scenario_specs['grid'] != orbitdata_dict['grid']:
                if len(scenario_specs['grid']) != len(orbitdata_dict['grid']):
                    return True
                
                for i in range(len(scenario_specs['grid'])):
                    scenario_grid : dict = scenario_specs['grid'][i]
                    mission_grid : dict = orbitdata_dict['grid'][i]

                    scenario_gridtype = scenario_grid['@type'].lower()
                    mission_gridtype = mission_grid['@type'].lower()

                    if scenario_gridtype != mission_gridtype == 'customgrid':
                        if scenario_gridtype not in mission_grid['covGridFilePath']:
                            return True

            if scenario_specs['spacecraft'] != orbitdata_dict['spacecraft']:
                if len(scenario_specs['spacecraft']) != len(orbitdata_dict['spacecraft']):
                    return True
                
                for i in range(len(scenario_specs['spacecraft'])):
                    scenario_sat : dict = scenario_specs['spacecraft'][i]
                    mission_sat : dict = orbitdata_dict['spacecraft'][i]
                    
                    if "planner" in scenario_sat:
                        scenario_sat.pop("planner")
                    if "science" in scenario_sat:
                        scenario_sat.pop("science")
                    if "notifier" in scenario_sat:
                        scenario_sat.pop("notifier") 
                    if "missionProfile" in scenario_sat:
                        scenario_sat.pop("missionProfile")

                    if "planner" in mission_sat:
                        mission_sat.pop("planner")
                    if "science" in mission_sat:
                        mission_sat.pop("science")
                    if "notifier" in mission_sat:
                        mission_sat.pop("notifier") 
                    if "missionProfile" in mission_sat:
                        mission_sat.pop("missionProfile")

                    if scenario_sat != mission_sat:
                        return True
                        
        return False

    def _create_uniform_grid(scenario_dir : str, grid_index : int, lat_spacing : float, lon_spacing : float) -> str:
        # create uniform grid
        groundpoints = [(lat, lon) 
                        for lat in np.linspace(-90, 90, int(180/lat_spacing)+1)
                        for lon in np.linspace(-180, 180, int(360/lon_spacing)+1)
                        if lon < 180
                        ]
                
        # create datagrame
        df = pd.DataFrame(data=groundpoints, columns=['lat [deg]','lon [deg]'])

        # save to csv
        grid_path : str = os.path.join(scenario_dir, 'resources', f'uniform_grid{grid_index}.csv')
        df.to_csv(grid_path,index=False)

        # return address
        return grid_path

    def _create_clustered_grid(scenario_dir : str, grid_index : int, n_clusters : int, n_cluster_points : int, variance : float) -> str:
        # create clustered grid of gound points
        std = np.sqrt(variance)
        groundpoints = []
        
        for _ in range(n_clusters):
            # find cluster center
            lat_cluster = (90 - -90) * random.random() -90
            lon_cluster = (180 - -180) * random.random() -180
            
            for _ in range(n_cluster_points):
                # sample groundpoint
                lat = random.normalvariate(lat_cluster, std)
                lon = random.normalvariate(lon_cluster, std)
                groundpoints.append((lat,lon))

        # create datagrame
        df = pd.DataFrame(data=groundpoints, columns=['lat [deg]','lon [deg]'])

        # save to csv
        grid_path : str = os.path.join(scenario_dir, 'resources', f'clustered_grid{grid_index}.csv')
        df.to_csv(grid_path,index=False)

        # return address
        return grid_path
    
    """
    COVERAGE Metrics
    """
    def calculate_percent_coverage(self) -> float:
        n_observed = 0
        n_points = sum([len(grid) for grid in self.grid_data])
        t_0 = time.perf_counter()

        # for grid in self.grid_data:
        #     grid : pd.DataFrame

        #     for lat,lon,grid_index,gp_index in grid.values:

        #         accesses = [(t_img,lat,lon,lat_img,lon_img)
        #                     for t_img, gp_index_img, _, lat_img, lon_img, _, _, _, _, grid_index_img, *_ 
        #                     in self.gp_access_data.values
        #                     if abs(lat - lat_img) < 1e-3
        #                     and abs(lon - lon_img) < 1e-3
        #                     and gp_index == gp_index_img 
        #                     and grid_index == grid_index_img]
                
        #         if accesses:
        #             n_observed += 1

        gp_observed = { (lat,lon) 
                        for grid in self.grid_data
                        for lat,lon,grid_index,gp_index in grid.values
                        for _, gp_index_img, _, _, _, _, _, _, _, grid_index_img, *_ in self.gp_access_data.values
                        if gp_index == gp_index_img 
                        and grid_index == grid_index_img}
        n_observed = len(gp_observed)
        
        dt = time.perf_counter() - t_0        
        
        return n_points, n_observed, float(n_observed) / float(n_points)

"""
TESTING
"""
if __name__ == '__main__':
    scenario_dir = './scenarios/sim_test/'
    
    orbit_data_list = OrbitData.from_directory(scenario_dir)

    # expected val: (grid, point) = 0, 0
    for agent in orbit_data_list:
        lat = 1.0
        lon = 158.0
        t = 210.5

        grid, point, gp_lat, gp_lon = orbit_data_list[agent].find_gp_index(lat, lon)
        print(f'({lat}, {lon}) = G{grid}, P{point}, Lat{gp_lat}, Lon{gp_lon}')

        print(orbit_data_list[agent].is_accessing_ground_point(lat, lon, t))
        break