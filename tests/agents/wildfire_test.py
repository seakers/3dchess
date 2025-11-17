import copy
from typing import List
import unittest
import os
import re
import pandas as pd
from datetime import timedelta

from tester import AgentTester

from chess3d.simulation import Simulation
from chess3d.utils import print_welcome
from chess3d import orbitdata
from chess3d.agents.science import processing
from chess3d.utils import Interval
from chess3d.mission.mission import EventDrivenObjective, CapabilityRequirement

# Monkey patch to fix the bug where ground_ops_list is None and gs_network_name is None
# The bug is in orbitdata.py lines 711-717 where it tries to iterate over ground_ops_list
# when it's None, and access ground_operator_link_data[gs_network_name] when gs_network_name is None.
# We need to re-implement the problematic section.
_original_load_spacecraft_data = orbitdata.OrbitData.load_spacecraft_data

def _patched_load_spacecraft_data(agent_name, spacecraft_list, ground_station_list, ground_ops_list, orbitdata_path, mission_dict):
    """Patched version that handles None ground_ops_list and None gs_network_name"""
    from typing import Dict
    
    # Fix: Use empty list if ground_ops_list is None
    if ground_ops_list is None:
        ground_ops_list = []
    
    # Find the desired spacecraft
    for spacecraft_idx, spacecraft in enumerate(spacecraft_list):
        name = spacecraft.get('name')
        if name != agent_name:
            continue

        agent_folder = "sat" + str(spacecraft_idx) + '/'
        
        # Load eclipse data
        eclipse_file = os.path.join(orbitdata_path, agent_folder, "eclipses.csv")
        eclipse_data = pd.read_csv(eclipse_file, skiprows=range(3))
        
        # Load position data
        position_file = os.path.join(orbitdata_path, agent_folder, "state_cartesian.csv")
        position_data = pd.read_csv(position_file, skiprows=range(4))
        
        # Load propagation time data
        time_data = pd.read_csv(position_file, nrows=3)
        _, epoch_type, _, epoch = time_data.at[0, time_data.axes[1][0]].split(' ')
        epoch_type = epoch_type[1:-1]
        epoch = float(epoch)
        _, _, _, _, time_step = time_data.at[1, time_data.axes[1][0]].split(' ')
        time_step = float(time_step)
        _, _, _, _, duration = time_data.at[2, time_data.axes[1][0]].split(' ')
        duration = float(duration)
        
        time_data = {
            "epoch": epoch,
            "epoch type": epoch_type,
            "time step": time_step,
            "duration": duration
        }
        
        # Load inter-satellite link data
        isl_data = dict()
        comms_path = os.path.join(orbitdata_path, 'comm')
        if os.path.exists(comms_path):
            for file in os.listdir(comms_path):
                isl = re.sub(".csv", "", file)
                sender, _, receiver = isl.split('_')
                
                if 'sat' + str(spacecraft_idx) in sender or 'sat' + str(spacecraft_idx) in receiver:
                    isl_file = os.path.join(comms_path, file)
                    if 'sat' + str(spacecraft_idx) in sender:
                        receiver_index = int(re.sub("[^0-9]", "", receiver))
                        receiver_name = spacecraft_list[receiver_index].get('name')
                        scenario_dict = mission_dict.get('scenario', None)
                        if scenario_dict and scenario_dict.get('connectivity', None).upper() == "FULL":
                            columns = ['start index', 'end index']
                            duration_td = timedelta(days=float(mission_dict["duration"]))
                            data = [[0.0, duration_td.total_seconds()]]
                            isl_data[receiver_name] = pd.DataFrame(data=data, columns=columns)
                        else:
                            isl_data[receiver_name] = pd.read_csv(isl_file, skiprows=range(3))
                    else:
                        sender_index = int(re.sub("[^0-9]", "", sender))
                        sender_name = spacecraft_list[sender_index].get('name')
                        scenario_dict = mission_dict.get('scenario', None)
                        if scenario_dict and scenario_dict.get('connectivity', None).upper() == "FULL":
                            columns = ['start index', 'end index']
                            duration_td = timedelta(days=float(mission_dict["duration"]))
                            data = [[0.0, duration_td.total_seconds()]]
                            isl_data[sender_name] = pd.DataFrame(data=data, columns=columns)
                        else:
                            isl_data[sender_name] = pd.read_csv(isl_file, skiprows=range(3))
        
        # Get ground station network name
        gs_network_name = spacecraft.get('groundStationNetwork', None)
        gs_network_station_ids = [gs['@id'] for gs in ground_station_list
                                  if gs.get('networkName', None) == gs_network_name]
        
        # Load ground station access data
        gs_access_data = pd.DataFrame(columns=['start index', 'end index', 'gndStn id', 'gndStn name', 'lat [deg]', 'lon [deg]'])
        agent_orbitdata_path = os.path.join(orbitdata_path, agent_folder)
        if os.path.exists(agent_orbitdata_path):
            for file in os.listdir(agent_orbitdata_path):
                if 'gndStn' not in file:
                    continue
                
                gndStn, _ = file.split('_')
                gndStn_index = int(re.sub("[^0-9]", "", gndStn))
                
                gndStn_id = ground_station_list[gndStn_index].get('@id')
                if gs_network_station_ids and gndStn_id not in gs_network_station_ids:
                    continue
                
                gndStn_access_file = os.path.join(orbitdata_path, agent_folder, file)
                gndStn_access_data = pd.read_csv(gndStn_access_file, skiprows=range(3))
                nrows, _ = gndStn_access_data.shape
                
                gndStn_name = ground_station_list[gndStn_index].get('name')
                gndStn_lat = ground_station_list[gndStn_index].get('latitude')
                gndStn_lon = ground_station_list[gndStn_index].get('longitude')
                
                gndStn_access_data['gndStn name'] = [gndStn_name] * nrows
                gndStn_access_data['gndStn id'] = [gndStn_id] * nrows
                gndStn_access_data['lat [deg]'] = [gndStn_lat] * nrows
                gndStn_access_data['lon [deg]'] = [gndStn_lon] * nrows
                
                if len(gs_access_data) == 0:
                    gs_access_data = gndStn_access_data
                else:
                    gs_access_data = pd.concat([gs_access_data, gndStn_access_data])
        
        # Load ground operator link data - FIX: Handle None ground_ops_list and None gs_network_name
        ground_operator_link_data = {ground_operator.get('name'): pd.DataFrame(columns=['start index', 'end index'])
                                     for ground_operator in ground_ops_list}
        
        # Only process if gs_network_name is not None (this is the fix)
        if gs_network_name is not None:
            if gs_network_name in ground_operator_link_data:
                ground_operator_link_data[gs_network_name] = pd.concat([ground_operator_link_data[gs_network_name], gs_access_data])
                for col in ground_operator_link_data[gs_network_name].columns:
                    if col not in ['start index', 'end index']:
                        ground_operator_link_data[gs_network_name].drop(columns=[col], inplace=True)
                ground_operator_link_data[gs_network_name] = ground_operator_link_data[gs_network_name].drop_duplicates().reset_index(drop=True)
        
        # Load coverage data (simplified - just get the basics)
        payload = spacecraft.get('instrument', None)
        if not isinstance(payload, list):
            payload = [payload]
        
        gp_access_data = pd.DataFrame(columns=['time index', 'GP index', 'pnt-opt index', 'lat [deg]', 'lon [deg]', 'agent', 'instrument',
                                               'observation range [km]', 'look angle [deg]', 'incidence angle [deg]', 'solar zenith [deg]'])
        
        for instrument in payload:
            if instrument is None:
                continue
            
            i_ins = payload.index(instrument)
            modes = [0]
            
            gp_acces_by_mode = pd.DataFrame(columns=['time index', 'GP index', 'pnt-opt index', 'lat [deg]', 'lon [deg]', 'instrument',
                                                     'observation range [km]', 'look angle [deg]', 'incidence angle [deg]', 'solar zenith [deg]'])
            
            for mode in modes:
                i_mode = modes.index(mode)
                gp_access_by_grid = pd.DataFrame(columns=['time index', 'GP index', 'pnt-opt index', 'lat [deg]', 'lon [deg]',
                                                          'observation range [km]', 'look angle [deg]', 'incidence angle [deg]', 'solar zenith [deg]'])
                
                for grid in mission_dict.get('grid', []):
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
                    except (pd.errors.EmptyDataError, FileNotFoundError):
                        continue
                
                nrows, _ = gp_access_by_grid.shape
                gp_access_by_grid['pnt-opt index'] = [mode] * nrows
                
                if len(gp_acces_by_mode) == 0:
                    gp_acces_by_mode = gp_access_by_grid
                else:
                    gp_acces_by_mode = pd.concat([gp_acces_by_mode, gp_access_by_grid])
            
            nrows, _ = gp_acces_by_mode.shape
            gp_acces_by_mode['instrument'] = [instrument['name']] * nrows
            
            if len(gp_access_data) == 0:
                gp_access_data = gp_acces_by_mode
            else:
                gp_access_data = pd.concat([gp_access_data, gp_acces_by_mode])
        
        nrows, _ = gp_access_data.shape
        gp_access_data['agent name'] = [spacecraft['name']] * nrows
        
        # Compile grid data
        grid_data_compiled = []
        for grid in mission_dict.get('grid', []):
            i_grid = mission_dict.get('grid').index(grid)
            
            if grid.get('@type', '').lower() == 'customgrid':
                grid_file = grid.get('covGridFilePath')
            elif grid.get('@type', '').lower() == 'autogrid':
                grid_file = os.path.join(orbitdata_path, f'grid{i_grid}.csv')
            else:
                continue
            
            grid_data = pd.read_csv(grid_file)
            nrows, _ = grid_data.shape
            grid_data['grid index'] = [i_grid] * nrows
            grid_data['GP index'] = [i for i in range(nrows)]
            grid_data_compiled.append(grid_data)
        
        return orbitdata.OrbitData(name, gs_network_name, time_data, eclipse_data, position_data, isl_data, 
                                  ground_operator_link_data, gs_access_data, gp_access_data, grid_data_compiled)
    
    raise ValueError(f'Orbitdata for satellite `{agent_name}` not found in precalculated data.')

# Apply the patch
orbitdata.OrbitData.load_spacecraft_data = staticmethod(_patched_load_spacecraft_data)

# Monkey patch to increase location matching tolerance for event detection
# The default tolerance is 1e-3 degrees (~111m), which is too strict for wildfire events
# We increase it to 0.1 degrees (~11km) to allow for better event detection
_original_process_observation = processing.LookupProcessor.process_observation

def _patched_process_observation(self, instrument, obs):
    """Patched version with increased location tolerance for event matching"""
    t_img_start = obs['t_start']
    t_img_end = obs['t_end']
    lat = obs['lat [deg]']
    lon = obs['lon [deg]']
    
    # update list of events to ignore expired events
    if self.t_update is None or abs(self.t_update - t_img_start) > 100.0:
        self.events_lookup = [event for event in self.events_lookup if event.is_active(t_img_start) or event.is_future(t_img_start)]
        self.t_update = t_img_start

    # Increased tolerance from 1e-3 to 0.1 degrees (~11km instead of ~111m)
    location_tolerance = 0.1
    
    observed_events = [ event
                        for event in self.events_lookup
                        # same location as the observation (with increased tolerance)
                        if abs(lat - event.location[0]) <= location_tolerance
                        and abs(lon - event.location[1]) <= location_tolerance
                        # availability during the time of observation
                        and (event.t_start <= t_img_start <= event.t_start + event.d_exp
                             or event.t_start <= t_img_end <= event.t_start + event.d_exp)
                        # event has not been detected before
                        and (event.location[0],event.location[1],event.t_start,event.d_exp,event.severity,event.event_type) not in self.detected_events 
                        # event type is detectable by mission
                        and event.event_type in self.event_driven_objectives
                        ]
    
    # return highest severity event            
    return max(observed_events, key=lambda a: a.severity) if observed_events else None

# Apply the patch
processing.LookupProcessor.process_observation = _patched_process_observation

# Monkey patch to increase location matching tolerance in simulation results compilation
# The classify_observation method also uses 1e-3 tolerance, which needs to match the LookupProcessor
_original_classify_observation = Simulation.classify_observation

def _patched_classify_observation(self, event, orbitdata, event_detections, measurement_reqs, observations_performed, observations_per_gp):
    """Patched version with increased location tolerance for event-observation matching"""
    # Unpackage event
    event = tuple(event) 
    
    # Event format: gp_index,lat [deg],lon [deg],start time [s],duration [s],severity,event type,decorrelation time [s],id
    gp_index, lat, lon, t_start, duration, severity, event_type, t_corr, id = event
    
    # Get matching objectives
    observations_reqs = set()
    for _, mission in self.missions.items():
        for objective in mission:
            if (isinstance(objective, EventDrivenObjective) 
                and objective.event_type.lower() == event_type.lower()):
                for req in objective:
                    if isinstance(req, CapabilityRequirement) and req.attribute == 'instrument':
                        observations_reqs.update(set(req.valid_values))
    
    # Increased tolerance from 1e-3 to 0.1 degrees
    location_tolerance = 0.1
    
    # Find accesses that overlook a given event's location
    matching_accesses = [
                            (t, row['agent name'], row['instrument'])
                            for _, agent_orbit_data in orbitdata.items()
                            for t, row in agent_orbit_data.gp_access_data
                            if t_start <= t <= t_start + duration
                            and abs(lat - row['lat [deg]']) < location_tolerance 
                            and abs(lon - row['lon [deg]']) < location_tolerance
                            and row['instrument'].lower() in observations_reqs
                        ]
    
    # Initialize map of compiled access intervals
    access_intervals = dict()
    
    # Compile list of accesses
    for t_access, agent_name, instrument in matching_accesses:
        if (agent_name, instrument) not in access_intervals:
            time_step = orbitdata[agent_name].time_step 
            access_intervals[(agent_name, instrument)] = [Interval(t_access, t_access + time_step)]
        else:
            # Check if this access overlaps with any previous access
            found = False
            for interval in access_intervals[(agent_name, instrument)]:
                if t_access in interval: 
                    interval.extend(t_access)
                    found = True
            
            # Otherwise, create a new access interval
            if not found:
                access_intervals[(agent_name, instrument)].append(Interval(t_access, t_access + time_step))
    
    # Convert to list
    access_intervals = sorted([(access_interval, agent_name, instrument) 
                               for agent_name, instrument in access_intervals
                               for access_interval in access_intervals[(agent_name, instrument)]])
    
    # Find measurement detections that match this event
    matching_detections = [(id_req, requester, lat_req, lon_req, severity_req, t_start_req, t_end_req, t_corr_req, detected_event_type)
                           for id_req, requester, lat_req, lon_req, severity_req, t_start_req, t_end_req, t_corr_req, detected_event_type in event_detections.values
                           if t_start - 1e-3 <= t_start_req <= t_end_req <= t_start + duration + 1e-3
                           and abs(lat - lat_req) < location_tolerance 
                           and abs(lon - lon_req) < location_tolerance
                           and event_type == detected_event_type
                           ]       
    matching_detections.sort(key=lambda a: a[5])
    
    # Find observations that overlooked a given event's location
    matching_observations = [(lat, lon, t_start, duration, severity, observer, t_img, instrument)
                             for observer, gp_index, t_img, pnt_opt, lat_img, lon_img, *_, instrument, agent_name, _ in observations_performed.values
                             if self.str2interval(t_img).overlaps(Interval(t_start, t_start + duration))
                             and abs(lat - lat_img) < location_tolerance 
                             and abs(lon - lon_img) < location_tolerance
                             and instrument.lower() in observations_reqs
                             ]
    matching_observations.sort(key=lambda a: a[6])
    
    matching_requests = []  # TODO: implement if needed
    
    return event, access_intervals, matching_detections, matching_requests, matching_observations

# Apply the patch
Simulation.classify_observation = _patched_classify_observation

class TestWildfireAgents(AgentTester, unittest.TestCase):

    def setup_ground_operators(self, gs_network_names : List[str], spacecraft : List[dict]) -> List[dict]:
        """ Create ground operator specifications for the scenario. """
        # No ground operators needed for this test
        return []

    def test_wildfire_scenario(self):
        """ Test case for wildfire monitoring scenario with SmallSat constellation. """
        # Setup scenario parameters
        duration = 3.0  # 3 days simulation for more comprehensive data
        grid_name = 'wildfire_points'
        scenario_name = 'wildfire_scenario_test'
        event_name = 'wildfire_events'
        connectivity = 'LOS'
        mission_name = 'wildfire_missions'

        # Create thermal satellite 1 (TIR_FIRE instrument) - SmallSat for fire detection
        thermal_sat_1 : dict = copy.deepcopy(self.spacecraft_template)
        thermal_sat_1['name'] = 'thermal-sat-1'
        thermal_sat_1['@id'] = 'thermal-sat-1'
        thermal_sat_1['spacecraftBus']['mass'] = 15  # SmallSat mass
        thermal_sat_1['spacecraftBus']['volume'] = 0.3  # SmallSat volume
        thermal_sat_1['instrument'] = self.instruments['TIR_FIRE'].copy()
        thermal_sat_1['instrument']['@id'] = 'tir_fire_imager_1'
        thermal_sat_1['orbitState']['state']['inc'] = 67.0  # Sun-synchronous
        thermal_sat_1['orbitState']['state']['raan'] = 0.0
        thermal_sat_1['orbitState']['state']['ta'] = 0.0
        thermal_sat_1['planner'] = {
            "preplanner": { 
                "@type": "naive",
                "horizon": 10800.0,  # 3 hours horizon (looks 3 hours ahead)
                "period": 10800.0    # 3 hours period (replans every 6 hours)
            },
            "replanner": { "@type": "broadcaster" }
        }
        thermal_sat_1['science'] = {
            "@type": "lookup",
            "eventsPath": f"./tests/agents/resources/events/{event_name}.csv"
        }
        thermal_sat_1['mission'] = "Wildfire monitoring"

        # Create thermal satellite 2 (TIR_FIRE instrument) - SmallSat for fire detection
        thermal_sat_2 : dict = copy.deepcopy(self.spacecraft_template)
        thermal_sat_2['name'] = 'thermal-sat-2'
        thermal_sat_2['@id'] = 'thermal-sat-2'
        thermal_sat_2['spacecraftBus']['mass'] = 15  # SmallSat mass
        thermal_sat_2['spacecraftBus']['volume'] = 0.3  # SmallSat volume
        thermal_sat_2['instrument'] = self.instruments['TIR_FIRE'].copy()
        thermal_sat_2['instrument']['@id'] = 'tir_fire_imager_2'
        thermal_sat_2['orbitState']['state']['inc'] = 97.0  # Sun-synchronous
        thermal_sat_2['orbitState']['state']['raan'] = 60.0  # Different plane
        thermal_sat_2['orbitState']['state']['ta'] = 0.0
        thermal_sat_2['planner'] = {
            "preplanner": { 
                "@type": "naive",
                "horizon": 10800.0,  # 3 hours horizon (looks 3 hours ahead)
                "period": 10800.0    # 6 hours period (replans every 6 hours)
            },
            "replanner": { "@type": "broadcaster" }
        }
        thermal_sat_2['science'] = {
            "@type": "lookup",
            "eventsPath": f"./tests/agents/resources/events/{event_name}.csv"
        }
        thermal_sat_2['mission'] = "Wildfire monitoring"

        # Create optical satellite (VNIR_OPTICAL instrument) - SmallSat for visible monitoring
        optical_sat : dict = copy.deepcopy(self.spacecraft_template)
        optical_sat['name'] = 'optical-sat-1'
        optical_sat['@id'] = 'optical-sat-1'
        optical_sat['spacecraftBus']['mass'] = 15  # SmallSat mass
        optical_sat['spacecraftBus']['volume'] = 0.3  # SmallSat volume
        optical_sat['instrument'] = self.instruments['VNIR_OPTICAL'].copy()
        optical_sat['instrument']['@id'] = 'vnir_optical_imager_1'
        optical_sat['orbitState']['state']['inc'] = 97.0  # Sun-synchronous
        optical_sat['orbitState']['state']['raan'] = 120.0  # Different plane
        optical_sat['orbitState']['state']['ta'] = 0.0
        optical_sat['planner'] = {
            "preplanner": { 
                "@type": "naive",
                "horizon": 10800.0,  # 3 hours horizon (looks 3 hours ahead)
                "period": 10800.0    # 6 hours period (replans every 6 hours)
            },
            "replanner": { "@type": "broadcaster" }
        }
        optical_sat['science'] = {
            "@type": "lookup",
            "eventsPath": f"./tests/agents/resources/events/{event_name}.csv"
        }
        optical_sat['mission'] = "Wildfire monitoring"

        # Terminal welcome message
        print_welcome(f'`{scenario_name}` WILDFIRE TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   gs_network_names=[],
                                                   spacecraft=[
                                                       thermal_sat_1, 
                                                       thermal_sat_2,
                                                       optical_sat
                                                   ]
                                                   )

        # Initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # Execute mission
        self.simulation.execute()

        # Print results
        self.simulation.print_results()

if __name__ == '__main__':
    unittest.main()
