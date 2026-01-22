import argparse
from enum import Enum
import logging
import math
import os
import shutil

import numpy as np

class CoordinateTypes(Enum):
    """
    # Coordinate Type

    Describes the type of coordinate being described by a position vector
    """
    CARTESIAN = 'CARTESIAN'
    KEPLERIAN = 'KEPLERIAN'
    LATLON = 'LATLON'

class ModuleTypes(Enum):
    """
    # Types of Internal Modules for agents 
    """
    PLANNER = 'PLANNER'
    SCIENCE = 'SCIENCE'
    ENGINEERING = 'ENGINEERING'

def setup_results_directory(scenario_path : str, scenario_name : str, agent_names : list, overwrite : bool = True) -> str:
    """
    Creates an empty results directory within the current working directory
    """
    results_path = os.path.join(scenario_path, 'results', scenario_name)

    if not os.path.exists(results_path):
        # create results directory if it doesn't exist
        os.makedirs(results_path)

    elif overwrite:
        # clear results in case it already exists
        for filename in os.listdir(results_path):
            file_path = os.path.join(results_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        # path exists and no overwrite 
        return results_path

    # create a results directory for all agents
    for agent_name in agent_names:
        agent_name : str
        agent_results_path : str = os.path.join(results_path, agent_name.lower())
        os.makedirs(agent_results_path)

    return results_path

def print_welcome(scenario_name = None) -> None:
    os.system('cls' if os.name == 'nt' else 'clear')
    out = "\n======================================================"
    out += '\n   _____ ____        ________  __________________\n  |__  // __ \      / ____/ / / / ____/ ___/ ___/\n   /_ </ / / /_____/ /   / /_/ / __/  \__ \\__ \ \n ___/ / /_/ /_____/ /___/ __  / /___ ___/ /__/ / \n/____/_____/      \____/_/ /_/_____//____/____/ (v1.1)'
    out += "\n======================================================"
    out += '\n\tTexas A&M University - SEAK Lab ©'
    out += "\n======================================================"
    if scenario_name is not None: out += f"\nSCENARIO: {scenario_name}"
    print(out)

def arg_parser() -> tuple:
    """
    Parses the input arguments to the command line when starting a simulation
    
    ### Returns:
        `scenario_name`, `plot_results`, `save_plot`, `no_graphic`, `level`
    """
    parser : argparse.ArgumentParser = argparse.ArgumentParser(prog='DMAS for 3D-CHESS',
                                                               description='Simulates an autonomous Earth-Observing satellite mission.',
                                                               epilog='- TAMU')

    parser.add_argument(    '-n',
                            '--scenario-name', 
                            help='name of the scenario being simulated',
                            type=str,
                            required=False,
                            default='none')
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
                            '--welcome-graphic',
                            action='store_true',
                            help='draws ascii welcome screen graphic',
                            required=False,
                            default=True)  
    parser.add_argument(    '-l', 
                            '--level',
                            choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL', 'ERROR'],
                            default='WARNING',
                            help='logging level',
                            required=False,
                            type=str)  
                    
    args = parser.parse_args()
    
    scenario_name = args.scenario_name
    plot_results = args.plot_result
    save_plot = args.save_plot
    no_graphic = args.welcome_graphic

    levels = {  'DEBUG' : logging.DEBUG, 
                'INFO' : logging.INFO, 
                'WARNING' : logging.WARNING, 
                'CRITICAL' : logging.CRITICAL, 
                'ERROR' : logging.ERROR
            }
    level = levels.get(args.level)

    return scenario_name, plot_results, save_plot, no_graphic, level

def argmax(values, rel_tol=1e-9, abs_tol=0.0):
        """ returns the index of the highest value in a list of values """
        max_val = max(values)
        for i, val in enumerate(values):
            if math.isclose(val, max_val, rel_tol=rel_tol, abs_tol=abs_tol):
                return i        
            
        raise ValueError("No maximum value found in the list.")

def str_to_list(lst_string : str, list_type : type = str) -> list:
    """ reverts a list that has been printed into a string back into a list """
    
    # remove printed list brackets and quotes
    lst = lst_string.replace('[','')
    lst = lst.replace(']','')
    lst = lst.replace('\'','')
    lst = lst.replace(' ','')

    # convert into a string
    return [list_type(val) for val in lst.split(',')]

LEVELS = {  'DEBUG' : logging.DEBUG, 
            'INFO' : logging.INFO, 
            'WARNING' : logging.WARNING, 
            'CRITICAL' : logging.CRITICAL, 
            'ERROR' : logging.ERROR
        }


def monitoring(kwargs) -> float:
    """
    ### Monitoring Utility Function

    This function calculates the utility of a monitoring observation based on the time since the last observation.
    The utility is calculated as the time since the last observation divided by the total time available for monitoring.

    - :`observation`: The current observation.
    - :`unobserved_reward_rate`: The rate at which the reward decreases for unobserved events.    - :`latest_observation`: The latest observation.
    - :`kwargs`: Additional keyword arguments (not used in this function).
    
    """
    t_img = kwargs['t_start']
    t_prev = kwargs.get('t_prev',0.0)
    unobserved_reward_rate = kwargs.get('unobserved_reward_rate', 1.0)
    n_obs = kwargs.get('n_obs', 0)
        
    assert (t_img - t_prev) >= 0.0 # TODO fix acbba triggering this

    # calculate reward
    reward = (t_img - t_prev) * unobserved_reward_rate / 3600 
    
    # clip reward to [0, 1]
    reward = np.clip(reward, 0.0, 1.0)

    # return reward
    return reward


