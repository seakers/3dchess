import argparse

import os
import random
import shutil

import numpy as np
import pandas as pd
import tqdm

from chess3d.utils import print_welcome, LEVELS


def main(experiments_name : str,
         grid_name : str,
         lower_bound : int, 
         upper_bound : int, 
         overwrite : bool = True,
         seed : int = 1000
         ):    
    # read scenario parameters file
    experiment_path = os.path.join('./experiments', f'{experiments_name}.csv')
    experiments_df : pd.DataFrame = pd.read_csv(experiment_path)

    # check if bounds are valid
    assert 0 <= lower_bound <= upper_bound
    assert lower_bound <= len(experiments_df) - 1
    assert upper_bound <= len(experiments_df) - 1 or np.isinf(upper_bound)

    # make output directory
    events_dir = os.path.join('./events', experiments_name)
    if overwrite or not os.path.isdir(events_dir):
        clear_events(events_dir)
        os.mkdir(events_dir)

    # count number of runs to be made
    n_runs : int = min(len(experiments_df), upper_bound-lower_bound)
    print(F'NUMBER OF EVENTS TO GENERATE: {n_runs}')

    # get grid
    grid_path : str = os.path.join('./grids', f'{grid_name}.csv')
    grid : pd.DataFrame = pd.read_csv(grid_path)

    # set simulation duration in hours
    sim_duration : float = 24.0 / 24.0

    # run simulation for each set of parameters
    for experiment_i,row in tqdm.tqdm(experiments_df.iterrows(), 
                                      desc = 'Generating Events'):
        if experiment_i < lower_bound:
            continue
        elif upper_bound < experiment_i:
            break

        # extract event parameters
        experiment_name = row['Name']
        event_duration = row['Event Duration (hrs)']
        n_events = row['Number of Events per Day']
        min_severity = 0.0
        max_severity = 100
        measurement_list = ['sar', 'visual', 'thermal']

        # run cases
        create_events(events_dir,
                      experiment_name, 
                      grid, 
                      sim_duration, 
                      n_events, 
                      event_duration, 
                      min_severity, 
                      max_severity, 
                      measurement_list, 
                      overwrite,
                      seed)

def clear_events(events_path : str) -> None:
    if os.path.exists(events_path):       
        # clear results in case it already exists
        for filename in os.listdir(events_path):
            file_path = os.path.join(events_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

def create_events(experiments_dir : str,
                  experiment_name : str, 
                  grid : pd.DataFrame, 
                  sim_duration : float, 
                  n_events : int, 
                  event_duration : float, 
                  min_severity : float, 
                  max_severity : float, 
                  measurements : list,
                  overwrite : bool = False,
                  seed : int = 1000
                  ) -> str:
    # set random seed
    random.seed(seed)
    
    # set events path
    experiments_path = os.path.join(experiments_dir, f'{experiment_name}_events.csv')
    
    # check if events have already been generated
    if os.path.isfile(experiments_path) and not overwrite: return experiments_path

    # check if measurements list contains more than one measurement
    if len(measurements) < 2: raise ValueError('`measurements` must include more than one sensor')

    # generate events
    events = []
    for _ in tqdm.tqdm(range(int(n_events * sim_duration)), 
                       desc=f'Generating Events for {experiment_name}', 
                       leave=False):
        
        while True:
            # select a random ground point for this event
            gp_index = random.randint(0, len(grid)-1)
            gp = grid.iloc[gp_index]
            
            # generate start time 
            t_start = sim_duration * 24 * 3600 * random.random()

            # check if time overlap exists in the same ground point
            overlapping_events = [(t_start_overlap,duration_overlap)
                                for gp_index_overlap,_,_,t_start_overlap,duration_overlap,_,_ in events
                                if gp_index == gp_index_overlap
                                and (t_start_overlap <= t_start <= t_start_overlap + duration_overlap
                                or   t_start <= t_start_overlap <= t_start + event_duration*3600)]
            
            # if no overlaps, break random generation cycle
            if not overlapping_events: break

        # generate severity
        severity = max_severity * random.random() + min_severity

        # generate required measurements        
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
            gp_index,
            gp['lat [deg]'],
            gp['lon [deg]'],
            t_start,
            event_duration * 3600,
            severity,
            measurements_str
        ]

        # add to list of events
        events.append(event)

    # validate event generation constraints
    for gp_index,_,_,t_start,duration,_,_ in tqdm.tqdm(events, 
                       desc=f'validating {experiment_name} events', 
                       leave=False):
        
        # check if time overlap exists in the same ground point
        overlapping_events = [(t_start_overlap,duration_overlap)
                            for gp_index_overlap,_,_,t_start_overlap,duration_overlap,_,_ in events
                            if gp_index == gp_index_overlap
                            and abs(t_start - t_start_overlap) > 1e-3
                            and (t_start_overlap <= t_start <= t_start_overlap + duration_overlap
                            or   t_start <= t_start_overlap <= t_start + duration)]
        
        assert not overlapping_events

    assert len(events) == n_events

    # compile list of events
    events_df = pd.DataFrame(data=events, columns=['gp_index','lat [deg]','lon [deg]','start time [s]','duration [s]','severity','measurements'])
    
    # save list of events to events path 
    events_df.to_csv(experiments_path,index=False)

    # return path address
    return experiments_path

if __name__ == "__main__":
    
    # create argument parser
    parser = argparse.ArgumentParser(prog='3D-CHESS - Parametric Event Generator Study',
                                     description='Generates events for a given grid.',
                                     epilog='- TAMU')
    
    # set parser arguments
    parser.add_argument('-s',
                        '--scenario-name', 
                        help='name of scenario being used to select the location of events',
                        type=str,
                        required=False,
                        default='experiments_seed-1000')
    parser.add_argument('-g',
                        '--grid-name', 
                        help='name of grid being used to select the location of events',
                        type=str,
                        required=False,
                        default='hydrolakes_dataset')
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
    parser.add_argument('-o', 
                        '--overwrite',
                        default=False,
                        help='results overwrite toggle',
                        required=False,
                        type=bool) 
    
    # parse arguments
    args = parser.parse_args()
    
    # extract arguments
    scenario_name = args.scenario_name
    grid_name = args.grid_name
    lower_bound = args.lower_bound
    upper_bound = args.upper_bound
    overwrite = args.overwrite

    # print welcome
    print_welcome('Event generator for Parametric Study')

    # run simulation
    main(scenario_name, grid_name, lower_bound, upper_bound, overwrite)

    # print DONE
    print(f'Event generation for sims {lower_bound}-{upper_bound} DONE')
    