import argparse

import os
import random
import shutil

import numpy as np
import pandas as pd
import tqdm

from chess3d.utils import print_welcome, LEVELS


def main(experiments_name : str,
         overwrite : bool = True,
         seed : int = 1000
         ):    
    # read scenario parameters file
    experiment_path = os.path.join('./experiments', f'{experiments_name}.csv')
    experiments_df : pd.DataFrame = pd.read_csv(experiment_path)

    # make output directory
    events_dir = os.path.join('./events', experiments_name)
    if overwrite or not os.path.isdir(events_dir):
        clear_events(events_dir)
        if not os.path.isdir(events_dir): os.mkdir(events_dir)

    # count number of runs to be made
    n_runs : int = len(experiments_df) / (len(experiments_df['Preplanner'].unique()) * len(experiments_df['Replanner'].unique()))
    print(F'NUMBER OF EVENTS TO GENERATE: {n_runs}')

    # set simulation duration in hours
    sim_duration : float = 24.0 / 24.0

    experiments_df.pop('Preplanner')
    experiments_df.pop('Replanner')
    experiments_df.pop('Name')
    experiments_df = experiments_df.drop_duplicates()

    # run simulation for each set of parameters
    for _,row in tqdm.tqdm(experiments_df.iterrows(), 
                                      desc = 'Generating Events'):

        # extract event parameters
        scenario_id = row['Scenario ID']
        grid_name = f"{row['Grid Type']}_grid_{row['Number of Ground-Points']}"
        event_duration = row['Event Duration (hrs)']
        n_events = row['Number of Events per Day']
        min_severity = 0.0
        max_severity = 100
        measurement_list = ['sar', 'visual', 'thermal']


        # get grid
        if 'hydrolakes' in grid_name: grid_name += f'_seed-{seed}'
        grid_path : str = os.path.join('./grids', f'{grid_name}.csv')
        grid : pd.DataFrame = pd.read_csv(grid_path)

        # run cases
        create_events(events_dir,
                      scenario_id, 
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
                  scenario_id : str, 
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
    experiments_path = os.path.join(experiments_dir, f'scenario_{scenario_id}_events.csv')
    
    # check if events have already been generated
    if os.path.isfile(experiments_path) and not overwrite: return experiments_path

    # check if measurements list contains more than one measurement
    if len(measurements) < 2: raise ValueError('`measurements` must include more than one sensor')

    # generate events
    events = []
    for _ in tqdm.tqdm(range(int(n_events * sim_duration)), 
                       desc=f'Generating Events for `scenario_{scenario_id}`', 
                       leave=False):
        
        while True:
            # generate start time 
            t_start = sim_duration * 24 * 3600 * random.random()
            
            gp_history = set()
            while True:
                # select a random ground point for this event
                gp_index = random.randint(0, len(grid)-1)

                if gp_index in gp_history: continue

                gp_history.add(gp_index)
                gp = grid.iloc[gp_index]

                # check if time overlap exists in the same ground point
                overlapping_events = [(t_start_overlap,duration_overlap)
                                    for gp_index_overlap,_,_,t_start_overlap,duration_overlap,_,_ in events
                                    if gp_index == gp_index_overlap
                                    and (t_start_overlap <= t_start <= t_start_overlap + duration_overlap
                                    or   t_start <= t_start_overlap <= t_start + event_duration*3600)]
                
                # if no overlaps, break random generation cycle
                if not overlapping_events: break

                # if all ground points have overlaps at this time, try another start time
                if len(gp_history) == len(grid): break

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
    # for gp_index,_,_,t_start,duration,_,_ in tqdm.tqdm(events, 
    #                    desc=f'Validating {experiment_name} events', 
    #                    leave=False):
        
    #     # check if time overlap exists in the same ground point
    #     overlapping_events = [(t_start_overlap,duration_overlap)
    #                         for gp_index_overlap,_,_,t_start_overlap,duration_overlap,_,_ in events
    #                         if gp_index == gp_index_overlap
    #                         and abs(t_start - t_start_overlap) > 1e-3
    #                         and (t_start_overlap <= t_start <= t_start_overlap + duration_overlap
    #                         or   t_start <= t_start_overlap <= t_start + duration)]
        
    #     assert not overlapping_events

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
    overwrite = args.overwrite

    # print welcome
    print_welcome('Event generator for Parametric Study')

    # run simulation
    main(scenario_name, 
        #  overwrite
         )

    # print DONE
    print(f'Event generation for sims DONE')
    