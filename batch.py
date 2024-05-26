import argparse
import json
import logging
import os

from chess3d.main import main, precompute_orbitdata, print_welcome


if __name__ == "__main__":
    # read system arguments
    parser = argparse.ArgumentParser(
                    prog='DMAS for 3D-CHESS',
                    description='Simulates an autonomous Earth-Observing satellite mission.',
                    epilog='- TAMU')

    parser.add_argument(    'scenarios_name', 
                            help='name of the directory containing all of the scenarios being simulated',
                            type=str)
    parser.add_argument(    '-p', 
                            '--plot-result',
                            action='store_true',
                            help='creates animated plot of the simulation',
                            required=False,
                            default=False)    
    parser.add_argument(    '-s', 
                            '--save-plot',
                            action='store_true',
                            help='saves animated plot of the simulation as a gif',
                            required=False,
                            default=False) 
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
    
    scenarios_name = args.scenarios_name
    plot_results = args.plot_result
    save_plot = args.save_plot
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
        print_welcome(scenarios_name)

    # run sims for each scenario in the given dir
    scenarios_path = f"{scenarios_name}" if "./scenarios/" in scenarios_name else f'./scenarios/{scenarios_name}/'

    for scenario_name in os.listdir(scenarios_path):
        scenario_dir = scenarios_path + f'{scenario_name}/MissionSpecs.json'
        scenario_file = open(scenario_dir, 'r')
        scenario_dict : dict = json.load(scenario_file)
        scenario_file.close()


        # precompute orbit data
        spacecraft_dict = scenario_dict.get('spacecraft', None)
        orbitdata_dir = precompute_orbitdata(f'{scenarios_name}/{scenario_name}/') if spacecraft_dict is not None else None

        print(f'\n\nSIMULATING SCENARIO: `{scenario_name}`')
        main(scenario_name, f'{scenarios_path}{scenario_name}/', orbitdata_dir, plot_results, save_plot, levels)
        