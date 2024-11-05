import argparse
import copy
import json
import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

from chess3d.mission import Mission
from chess3d.utils import print_welcome, LEVELS


def main(
         experiments_name : str,
         lower_bound : int, 
         upper_bound : int, 
         level : int, 
         overwrite : bool = True,
         debug : bool = True):
    
    # get experiments name
    _,seeds = experiments_name.split('_')
    _,seed =  seeds.split('-')

    # set scenario name
    parent_scenario_name = 'replanner_study'
    scenario_dir = os.path.join('./scenarios', parent_scenario_name)

    # check if orbitdata folder exists
    if not os.path.isdir(os.path.join(scenario_dir, 'orbit_data')): os.mkdir(os.path.join(scenario_dir, 'orbit_data'))
    
    # remove previous orbit data
    if overwrite: clear_orbitdata(scenario_dir)
    
    # load base scenario json file
    template_file = os.path.join(scenario_dir,'MissionSpecs.json')
    with open(template_file, 'r') as template_file:
        template_specs : dict = json.load(template_file)

    # read scenario parameters file
    experiments_file = os.path.join(scenario_dir, 'resources', 'experiments', f'{experiments_name}.csv')
    experiments_df : pd.DataFrame = pd.read_csv(experiments_file)
    # experiments_df = experiments_df.sort_values(by=['Number Planes','Number of Satellites per Plane'], ascending=True)

    # check if bounds are valid
    assert 0 <= lower_bound <= upper_bound
    if len(experiments_df) <= lower_bound: raise ValueError('Lower bound exceeds number of experiments. None will be run.')

    # set fixed parameters
    sim_duration = 1.0 / 24.0 if debug else 1.0 # in days
    period, horizon = np.Inf, np.Inf

    # count number of runs to be made
    experiments_to_eval = [ (i,row) for i,row in experiments_df.iterrows()
                            if lower_bound <= i < upper_bound] 
    n_runs : int = len(experiments_to_eval) if not debug else 1
    print(F'NUMBER OF RUNS TO PERFORM: {n_runs}')

    # run simulation for each set of parameters
    for i_experiment,row in tqdm(experiments_to_eval, desc='Evaluating experiments'):          

        # extract constellation parameters
        n_planes = row['Number Planes']
        sats_per_plane = row['Number of Satellites per Plane']
        tas = [360 * i / sats_per_plane for i in range(sats_per_plane)]
        raans = [360 * j / n_planes for j in range(n_planes)]

        # extract planner info
        preplanner = row['Preplanner']
        replanner = row['Replanner']
        n_points = row['Number of Ground-Points']
        fraction_points_considered = row['Percent Ground-Points Considered']
        grid_name = f"{row['Grid Type']}_grid_{row['Number of Ground-Points']}"
        if 'hydrolakes' in grid_name: grid_name += f'_seed-{seed}'

        # extract satellite capability parameters
        field_of_regard = 1e-6 if preplanner == 'nadir' and replanner == 'broadcaster' else row['Field of Regard (deg)']
        field_of_view = row['Field of View (deg)']
        agility = row['Maximum Slew Rate (deg/s)']
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
        
        # run cases

        # create specs from template
        scenario_specs : dict = copy.deepcopy(template_specs)

        # create scenario name
        experiment_name = row['Name']
        results_path = os.path.join(scenario_dir, 'results', experiment_name, 'summary.csv')
        if os.path.isfile(results_path) and not overwrite: continue
        
        # set scenario name
        scenario_specs['scenario']['name'] = experiment_name

        # set outdir
        orbitdata_dir = os.path.join('./scenarios', parent_scenario_name, 'orbit_data', experiment_name)
        scenario_specs['settings']['outDir'] = orbitdata_dir

        # check overwrite toggle
        results_dir = os.path.join('./scenarios', parent_scenario_name, 'results', experiments_name)
        if not overwrite and os.path.exists(os.path.join(results_dir, 'summary.csv')):
            # scenario already simulated and told not to overwrite; skip
            continue

        # set simulation duration
        scenario_specs['duration'] = sim_duration

        # set coverage grid and events 
        grid_file = os.path.join(scenario_dir, 'resources', 'grids', f'{grid_name}.csv')
        scenario_specs['grid'] = [{
                                    "@type": "customGrid",
                                    "covGridFilePath": grid_file
                                }]
        scenario_specs['scenario']['events'] = {
                                    "@type": "PREDEF", 
                                    "eventsPath" : os.path.join(scenario_dir, 'resources', 'events', experiments_name, f"{row['Name']}_events.csv")
                                }

        # set spacecraft specs
        sats = []
        sat_index = 0
        for j in range(n_planes):
            instr_counter = {instrument : 0 for instrument in instruments}
            for i in range(sats_per_plane):
                # initiate satellite dictionary from template
                sat : dict = copy.deepcopy(scenario_specs['spacecraft'][-1])

                # choose agent instrument
                i_instrument = np.mod(sat_index, len(instruments))
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
                sat['instrument']['@id'] = f'{abbreviations[instrument]}_1'

                # set orbit
                sat['orbitState']['state']['raan'] = raans[j]
                sat['orbitState']['state']['ta'] = tas[i]

                # set preplanner
                sat['planner']['preplanner']['@type'] = preplanner
                sat['planner']['preplanner']['period'] = period
                sat['planner']['preplanner']['horizon'] = horizon
                sat['planner']['preplanner']['numGroundPoints'] = int(np.floor(n_points * fraction_points_considered))

                # set replanner
                sat['planner']['replanner']['@type'] = replanner

                # set science
                events_path = os.path.join(scenario_dir, 'resources', 'events', experiments_name, f"{row['Name']}_events.csv")
                sat['science']['eventsPath'] = events_path
                
                # add to list of sats
                sats.append(sat)

                # update instrument counter 
                instr_counter[instrument] += 1

                # pudate satellite index
                sat_index += 1
        
        # update list of satellites
        scenario_specs['spacecraft'] = sats

        # print welcome message
        print_welcome(experiment_name)

        # initialize mission
        mission : Mission = Mission.from_dict(scenario_specs)

        # execute mission
        mission.execute()

        # print results
        mission.print_results()

        # check if summary file was properly generated at the end of the simulation
        if not os.path.isfile(results_path): raise Exception(f'`{row["Name"]}` not executed properly.')

        # stop if debugging mode is on
        if debug and i_experiment > 3: return

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

if __name__ == "__main__":
    
    # create argument parser
    parser = argparse.ArgumentParser(prog='3D-CHESS - Replanner Parametric Study',
                                     description='Study performance of ACBBA as a reactive planning strategy in the context of Earth-observing missions.',
                                     epilog='- TAMU')
    
    # set parser arguments
    parser.add_argument('-s',
                        '--experiments-name', 
                        help='name of set of experiments being used to select the location of events',
                        type=str,
                        required=False,
                        default='experiments_seed-1000')
    parser.add_argument('-l',
                        '--lower-bound', 
                        help='lower bound of simulation indeces to be run (inclusive)',
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument('-u',
                        '--upper-bound', 
                        help='upper bound of simulation indeces to be run (non-inclusive)',
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
    parser.add_argument('-o', 
                        '--overwrite',
                        default=False,
                        help='results overwrite toggle',
                        required=False,
                        type=bool) 
    parser.add_argument('-debug', 
                        '--debug',
                        default=False,
                        help='toggles to run just one experiment for debugging purposes',
                        required=False,
                        type=bool) 
    
    # parse arguments
    args = parser.parse_args()
    
    # extract arguments
    scenario_name = args.experiments_name
    lower_bound = args.lower_bound
    upper_bound = args.upper_bound
    level = LEVELS.get(args.level)
    overwrite = args.overwrite
    debug = args.debug

    # run simulation
    main(scenario_name, 
         lower_bound, 
         upper_bound, 
         level, 
         overwrite, 
         debug
         )

    # print DONE
    print(f'Sims {lower_bound}-{upper_bound} DONE')
    