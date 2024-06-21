import argparse
from functools import reduce
import json
import logging
import os

from dmas.messages import SimulationElementRoles
import pandas as pd
import tqdm
from chess3d.agents.orbitdata import OrbitData, TimeInterval
from chess3d.utils import print_welcome, setup_results_directory

def main(   scenario_name : str, 
            scenario_path : str,
            level : int = logging.WARNING
        ) -> None:
    
    # load scenario json file
    scenario_filename = os.path.join(scenario_path, 'MissionSpecs.json')
    with open(scenario_filename, 'r') as scenario_file:
        scenario_dict : dict = json.load(scenario_file)

    # unpack agent info
    spacecraft_dict = scenario_dict.get('spacecraft', None)
    uav_dict        = scenario_dict.get('uav', None)
    gstation_dict   = scenario_dict.get('groundStation', None)
    settings_dict   = scenario_dict.get('settings', None)

    # unpack scenario info
    scenario_config_dict : dict = scenario_dict['scenario']
    grid_config_dict : dict = scenario_dict['grid']
    
    # load agent names
    agent_names = [SimulationElementRoles.ENVIRONMENT.value]
    if spacecraft_dict: agent_names.extend([spacecraft['name'] for spacecraft in spacecraft_dict])
    if uav_dict:        agent_names.extend([uav['name'] for uav in uav_dict])
    if gstation_dict:   agent_names.extend([gstation['name'] for gstation in gstation_dict])

    # create results directory
    setup_results_directory(scenario_path, agent_names)

    # compute orbit data
    OrbitData.precompute(scenario_name) if spacecraft_dict is not None else None

    # load agent data
    agent_orbitdata : OrbitData = OrbitData.load(scenario_path, agent_names[1])
    
    # calculate % coverage 
    n_points, n_observed, ptg_cov = calculate_percent_coverage(agent_orbitdata)
    print(f'Percent coverage: {round(ptg_cov*100,3)}% ({n_observed}/{n_points})')

    # calculate avg revisit time
    avg_rev = calculate_revisit_time(agent_orbitdata)

    """
    TEST RESULTS
    Max Roll        coverage %
    ~0°         1.332% (140/10512)
    ±5°         1.332% (140/10512)
    ±15°        1.988% (209/10512)
    ±25°        4.11% (432/10512)
    ±45°        9.056% (952/10512)
    """
    
def calculate_percent_coverage(orbitdata : OrbitData) -> float:
    n_observed = 0
    n_points = sum([len(grid) for grid in orbitdata.grid_data])

    with tqdm.tqdm(total=n_points,desc="Calculating percent coverage") as pbar:
        for grid in orbitdata.grid_data:
            grid : pd.DataFrame

            for lat,lon,grid_index,gp_index in grid.values:

                accesses = [t_img
                            for t_img, gp_index_img, _, lat_img, lon_img, _, _, _, _, grid_index_img, *_ in orbitdata.gp_access_data.values
                            if lat == lat_img 
                            and lon == lon_img 
                            and gp_index == gp_index_img 
                            and grid_index == grid_index_img]
                
                if accesses:
                    n_observed += 1
            
                pbar.update(1)
    
    return n_points, n_observed, float(n_observed) / float(n_points)

def calculate_revisit_time(orbitdata : OrbitData) -> float:
    access_times = [{} for _ in orbitdata.grid_data]
    n_points = sum([len(grid) for grid in orbitdata.grid_data])

    with tqdm.tqdm(total=n_points,desc="Calculating revisit time") as pbar:
        for grid in orbitdata.grid_data:
            grid : pd.DataFrame

            for lat,lon,grid_index,gp_index in grid.values:
                grid_index=int(grid_index); gp_index=int(gp_index)

                accesses = [t_img*orbitdata.time_step
                            for t_img, gp_index_img, _, lat_img, lon_img, _, _, _, _, grid_index_img, *_ in orbitdata.gp_access_data.values
                            if lat == lat_img 
                            and lon == lon_img 
                            and gp_index == gp_index_img 
                            and grid_index == grid_index_img]
                
                for t_img in accesses:
                    if gp_index not in access_times[grid_index]:
                        access_times[grid_index][gp_index] = [TimeInterval(t_img,t_img)]
                        continue
                    
                    time_interval : TimeInterval = access_times[grid_index][gp_index][-1]
                    if t_img < time_interval.start and time_interval.start - t_img <= orbitdata.time_step:
                        time_interval.extend(t_img)
                    elif time_interval.end < t_img and t_img - time_interval.end <= orbitdata.time_step:
                        time_interval.extend(t_img)
                    else:
                        access_times[grid_index][gp_index].append(TimeInterval(t_img,t_img))

                
                pbar.update(1)
    
    return None

if __name__ == "__main__":
    
    # read system arguments
    parser = argparse.ArgumentParser(
                    prog='DMAS for 3D-CHESS',
                    description='Simulates an autonomous Earth-Observing satellite mission.',
                    epilog='- TAMU')

    parser.add_argument(    'scenario_name', 
                            help='name of the scenario being simulated',
                            type=str)
    parser.add_argument(    '-d', 
                            '--no-graphic',
                            action='store_true',
                            help='does not draws ascii welcome screen graphic',
                            required=False,
                            default=False)  
    parser.add_argument(    '-l', 
                            '--level',
                            choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL', 'ERROR'],
                            default='WARNING',
                            help='logging level',
                            required=False,
                            type=str)  
                    
    args = parser.parse_args()
    
    scenario_name = args.scenario_name
    no_grapgic = args.no_graphic

    levels = {  'DEBUG' : logging.DEBUG, 
                'INFO' : logging.INFO, 
                'WARNING' : logging.WARNING, 
                'CRITICAL' : logging.CRITICAL, 
                'ERROR' : logging.ERROR
            }
    level = levels.get(args.level)

    # terminal welcome message
    if not no_grapgic:
        print_welcome(scenario_name)
    
    # load scenario json file
    scenario_path = f"{scenario_name}" if "./scenarios/" in scenario_name else f'./scenarios/{scenario_name}/'
    scenario_file = open(scenario_path + '/MissionSpecs.json', 'r')
    scenario_dict : dict = json.load(scenario_file)
    scenario_file.close()

    main(scenario_name, scenario_path, level)