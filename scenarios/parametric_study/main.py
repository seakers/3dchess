import argparse
import copy
import json
import os
import random
import shutil

import numpy as np
import pandas as pd
import tqdm

from chess3d.mission import Mission
from chess3d.utils import print_welcome, LEVELS


def main(lower_bound : int, upper_bound : int, level : int):
    # set scenario name
    parent_scenario_name = 'parametric_study'
    scenario_dir = os.path.join('./scenarios', parent_scenario_name)
    
    # load base scenario json file
    template_file = os.path.join(scenario_dir,'MissionSpecs.json')
    with open(template_file, 'r') as template_file:
        template_specs : dict = json.load(template_file)

    # read scenario parameters file
    scenarios_file = os.path.join(scenario_dir, 'resources', 'lhs_scenarios.csv')
    scenarios_df : pd.DataFrame = pd.read_csv(scenarios_file)
    # scenarios_df = scenarios_df.sort_values(by=['num_planes','num_sats_per_plane'], ascending=False)

    # check if bounds are valid
    assert 0 <= lower_bound <= upper_bound
    assert lower_bound <= len(scenarios_df) - 1
    assert upper_bound <= len(scenarios_df) - 1 or np.isinf(upper_bound)

    # set fixed parameters
    same_events = True
    overwrite = False
    sim_duration = 24.0 / 24.0 # in days
    preplanners = [
                        'naive',
                        # 'dynamic'
                        # 'nadir'                    
                        ]
    preplanner_settings = [ # (period[s], horizon [s])
        (np.Inf, np.Inf),
        (100, 100),         
        (500, 500)
    ]
    replanners = [
                    'broadcaster', 
                    # 'acbba',
                    ]
    bundle_sizes = [
                    # 1,
                    # 2, 
                    3 
                    # 5
                    ]   

    # count number of runs to be made
    n_runs : int = min(len(scenarios_df), upper_bound-lower_bound) * 3 * (len(replanners)-1 + len(bundle_sizes))
    print(F'NUMBER OF RUNS TO PERFORM: {n_runs}')

    # run simulation for each set of parameters
    with tqdm.tqdm(total=n_runs) as pbar:
        for scenario_i,row in scenarios_df.iterrows():
            if scenario_i < lower_bound:
                continue
            elif upper_bound < scenario_i:
                break

            # extract parameters
            n_planes = row['num_planes']
            sats_per_plane = row['num_sats_per_plane']
            tas = [360 * i / sats_per_plane for i in range(sats_per_plane)]
            raans = [360 * j / n_planes for j in range(n_planes)]

            field_of_regard = row['for (deg)']
            field_of_view = row['fov (deg)']
            agility = row['agility (deg/s)']
            max_torque = 0.0
            instruments = [
                            'visual', 
                            'thermal', 
                            'sar'
                            ]
            abbreviations = {
                            'visual' : 'vis', 
                            'thermal' : 'therm', 
                            'sar' : 'sar'
                            }

            event_duration = row['event_duration (s)']
            n_events = row['num_events']
            event_clustering = row['event_clustering']
            min_severity = 0.0
            max_severity = 100
            measurement_list = ['sar', 'visual', 'thermal']

            # remove previous orbit data
            # clear_orbitdata(scenario_dir)

            # run cases
            pregenerated_events = False
            for preplanner in preplanners:
                for replanner in replanners:
                    for period, horizon in preplanner_settings:
                        for bundle_size in bundle_sizes:
                            # check simulation requirements
                            if replanner == 'broadcaster' and bundle_size > 1 and len(bundle_sizes) > 1:
                                break # skip

                            if preplanner == 'dynamic' and horizon == np.Inf:
                                continue # skip

                            if preplanner == 'naive' and horizon != np.Inf:
                                continue # skip

                            # if preplanner == 'naive' and replanner == 'broadcaster':
                            #     continue # skip

                            if replanner == 'acbba' and horizon != np.Inf:
                                replanner += '-dp'

                            # create specs from template
                            scenario_specs : dict = copy.deepcopy(template_specs)

                            # create scenario name
                            scenario_name = f'{parent_scenario_name}_{scenario_i}_{preplanner}-{period}-{horizon}_{replanner}'
                            if replanner != 'broadcaster': scenario_name += f'-{bundle_size}'
                            
                            # set scenario name
                            scenario_specs['scenario']['name'] = scenario_name

                            # set outdir
                            orbitdata_dir = os.path.join('./scenarios', parent_scenario_name, 'orbit_data', f'{parent_scenario_name}_{scenario_i}')
                            scenario_specs['settings']['outDir'] = orbitdata_dir

                            # check overwrite toggle
                            results_dir = os.path.join('./scenarios', parent_scenario_name, 'results', scenario_name)
                            if not overwrite and os.path.exists(os.path.join(results_dir, 'summary.csv')):
                                # scenario already ran; skip to avoid overwrite
                                pbar.update(1)
                                continue

                            # set simulation duration
                            scenario_specs['duration'] = sim_duration

                            # set events
                            if not (same_events and pregenerated_events):
                                if event_clustering == 'uniform':
                                    grid_path = create_uniform_grid(scenario_dir, scenario_i, n_events)
                                elif event_clustering == 'clustered':
                                    grid_path = create_clustered_grid(scenario_dir, scenario_i, n_events)
                                else:
                                    raise NotImplementedError(f'event clustering of type {event_clustering} not supported.')
                                create_events(scenario_dir, scenario_i, grid_path, sim_duration, n_events, event_duration, min_severity, max_severity, measurement_list)
                                pregenerated_events = True
                                
                            scenario_specs['grid'] = [{
                                                        "@type": "customGrid",
                                                        "covGridFilePath": os.path.join(scenario_dir, 'resources', f'random_grid_{scenario_i}.csv')
                                                    }]
                            scenario_specs['scenario']['events'] = {
                                                        "@type": "PREDEF", 
                                                        "eventsPath" : os.path.join(scenario_dir, 'resources', f'random_events_{scenario_i}.csv')
                                                    }

                            # set spacecraft specs
                            sats = []
                            for j in range(n_planes):
                                instr_counter = {instrument : 0 for instrument in instruments}
                                for i in range(sats_per_plane):
                                    sat : dict = copy.deepcopy(scenario_specs['spacecraft'][-1])

                                    # choose agent instrument
                                    i_instrument = np.mod(i, len(instruments))
                                    instrument = instruments[i_instrument] 

                                    # set agent name and id
                                    sat['@id'] = f'{abbreviations[instrument]}_sat_{j}_{instr_counter[instrument]}'
                                    sat['name'] = f'{abbreviations[instrument]}_sat_{j}_{instr_counter[instrument]}'

                                    # set slew rate
                                    sat['spacecraftBus']['components']['adcs']['maxRate'] = agility
                                    sat['spacecraftBus']['components']['adcs']['maxTorque'] = max_torque

                                    # set instrument properties
                                    sat['instrument']['name'] = instrument
                                    sat['instrument']['fieldOfViewGeometry'] ['angleHeight'] = field_of_view
                                    sat['instrument']['fieldOfViewGeometry'] ['angleWidth'] = field_of_view
                                    sat['instrument']['maneuver']['A_rollMin'] = - field_of_regard / 2.0
                                    sat['instrument']['maneuver']['A_rollMax'] = field_of_regard / 2.0
                                    sat['instrument']['@id'] = f'{abbreviations[instrument]}1'

                                    # set orbit
                                    sat['orbitState']['state']['raan'] = raans[j]
                                    sat['orbitState']['state']['ta'] = tas[i]

                                    # set preplanner
                                    sat['planner']['preplanner']['@type'] = preplanner
                                    sat['planner']['preplanner']['period'] = period
                                    sat['planner']['preplanner']['horizon'] = horizon

                                    # set replanner
                                    sat['planner']['replanner']['@type'] = replanner
                                    sat['planner']['replanner']['bundle size'] = bundle_size

                                    # set science
                                    sat['science']['eventsPath'] = os.path.join(scenario_dir, 'resources', f'random_events_{scenario_i}.csv')
                                    
                                    # add to list of sats
                                    sats.append(sat)

                                    # update counter 
                                    instr_counter[instrument] += 1
                            
                            # update list of satellites
                            scenario_specs['spacecraft'] = sats

                            # print welcome message
                            print_welcome(scenario_name)

                            # initialize mission
                            mission : Mission = Mission.from_dict(scenario_specs)

                            # # execute mission
                            mission.execute()
                                
                            # update progress bad
                            pbar.update(1)

def clear_orbitdata(scenario_dir : str) -> None:
    orbitdata_path = os.path.join(scenario_dir, 'orbit_data')

    if os.path.exists(orbitdata_path):       
        # clear results in case it already exists
        for filename in os.listdir(orbitdata_path):
            file_path = os.path.join(orbitdata_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

def create_uniform_grid(scenario_dir : str, scenario_i : str, n_events : int, lat_spacing : float = 0.1, lon_spacing : float = 0.1) -> str:
    # set grid name
    grid_path : str = os.path.join(scenario_dir, 'resources', f'random_grid_{scenario_i}.csv')
    
    # check if grid already exists
    if os.path.isfile(grid_path): return grid_path

    # generate grid
    all_groundpoints = [(lat, lon) 
                        for lat in np.linspace(-90, 90, int(180/lat_spacing)+1)
                        for lon in np.linspace(-180, 180, int(360/lon_spacing)+1)
                        if lon < 180
                        ]

    groundpoints : list = random.sample(all_groundpoints, n_events)
    groundpoints.sort()
    
    # create dataframe
    df = pd.DataFrame(data=groundpoints, columns=['lat [deg]','lon [deg]'])

    # save to csv
    df.to_csv(grid_path,index=False)

    # return address
    return grid_path

def create_clustered_grid(scenario_dir : str, scenario_i : str, n_events : int, variance : float = 1.0, n_clusters : float = 100, lat_spacing : float = 0.1, lon_spacing : float = 0.1) -> str:
    # set grid name
    grid_path : str = os.path.join(scenario_dir, 'resources', f'random_grid_{scenario_i}.csv')
    
    # check if grid already exists
    if os.path.isfile(grid_path): return grid_path
    
    # generate cluster grid
    all_clusters = [(lat, lon) 
                        for lat in np.linspace(-90, 90, int(180/lat_spacing)+1)
                        for lon in np.linspace(-180, 180, int(360/lon_spacing)+1)
                        if lon < 180
                        ]
    clusters : list = random.sample(all_clusters, n_clusters)
    clusters.sort()

    # create clustered grid of gound points
    std = np.sqrt(variance)
    groundpoints = []

    for lat_cluster,lon_cluster in tqdm.tqdm(clusters, desc='generating clustered grid', leave=False):
        for _ in range(int(n_events / n_clusters)):
            # sample groundpoint
            lat = random.normalvariate(lat_cluster, std)
            lon = random.normalvariate(lon_cluster, std)
            groundpoints.append((lat,lon))

    # create datagrame
    df = pd.DataFrame(data=groundpoints, columns=['lat [deg]','lon [deg]'])

    # save to csv
    df.to_csv(grid_path,index=False)

    # return address
    return grid_path

def create_events(scenario_dir : str, scenario_i : str, grid_path : str, sim_duration : float, n_events : int, event_duration : float, min_severity : float, max_severity : float, measurements : list) -> str:
    # set events path
    events_path = os.path.join(scenario_dir, 'resources', f'random_events_{scenario_i}.csv')
    
    # check if events have already been generated
    if os.path.isfile(events_path): return events_path

    # load coverage grid
    grid : pd.DataFrame = pd.read_csv(grid_path)

    # generate events
    events = []
    for _ in tqdm.tqdm(range(n_events), desc='generating events', leave=False):
        # select ground points for events
        gp_index = random.randint(0, len(grid)-1)
        gp = grid.iloc[gp_index]
        
        # generate start time 
        t_start = sim_duration * 24 * 3600 * random.random()

        # generate severity
        severity = max_severity * random.random() + min_severity

        # generate required measurements
        if len(measurements) < 2: raise ValueError('`measurements` must include more than one sensor')
        n_measurements = random.randint(2,len(measurements)-1)
        required_measurements = random.sample(measurements,k=n_measurements)
        measurements_str = '['
        for req in required_measurements: 
            if required_measurements.index(req) == 0:
                measurements_str += req
            else:
                measurements_str += f',{req}'
        measurements_str += ']'
        
        # create event
        event = [
            gp['lat [deg]'],
            gp['lon [deg]'],
            t_start,
            event_duration,
            severity,
            measurements_str
        ]

        # add to list of events
        events.append(event)

    # compile list of events
    events_df = pd.DataFrame(data=events, columns=['lat [deg]','lon [deg]','start time [s]','duration [s]','severity','measurements'])
    
    # save list of events to events path 
    events_df.to_csv(events_path,index=False)

    # return path address
    return events_path

if __name__ == "__main__":
    
    # create argument parser
    parser = argparse.ArgumentParser(prog='3D-CHESS - ACBBA Parametric Study',
                                     description='Study performance of ACBBA as a reactive planning strategy in the context of Earth-observing missions.',
                                     epilog='- TAMU')
    
    # set parser arguments
    parser.add_argument('-l',
                        '--lower-bound', 
                        help='lower bound of simulation indeces to be run',
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument('-u',
                        '--upper-bound', 
                        help='upper bound of simulation indeces to be run',
                        type=int,
                        required=False,
                        default=np.Inf)
    parser.add_argument('-L', 
                        '--level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL', 'ERROR'],
                        default='WARNING',
                        help='logging level',
                        required=False,
                        type=str) 
    
    # parse arguments
    args = parser.parse_args()
    
    # extract arguments
    lower_bound = args.lower_bound
    upper_bound = args.upper_bound
    level = LEVELS.get(args.level)

    lower_bound = 3

    # run simulation
    main(lower_bound, upper_bound, level)

    # print DONE
    print(f'Sims {lower_bound}-{upper_bound} DONE')
    