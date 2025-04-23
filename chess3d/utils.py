import argparse
from enum import Enum
import logging
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

class Interval:
    """ Represents an linear interval set of real numbers """

    def __init__(self, left : float, right : float, left_open : bool=False, right_open : bool=False):
        self.left : float = left
        self.left_open : bool = left_open

        self.right : float = right
        self.right_open : bool = right_open

        if self.right < self.left:
            raise Exception('The right side of interval must be later than the left side of the interval.')

    def is_after(self, x : float) -> bool:
        """ checks if the interval starts after the value `x` """
        return x < self.left

    def is_before(self, x : float) -> bool:
        """ checks if the time ends before the value `x` """
        return self.right < x
    
    def is_empty(self) -> bool:
        """ checks if the interval is empty """
        return self.left == self.right
        
    def overlaps(self, __other : object) -> bool:
        """ checks if this interval has an overlap with another """
        
        if not isinstance(__other, Interval):
            raise TypeError(f'Cannot check overlap with object of type `{type(__other)}`.')

        return ( self.left in __other
                or self.right in __other
                or __other.left in self
                or __other.right in self)

        # return (    __other.left <= self.left <= __other.right
        #         or  __other.left <= self.right <= __other.right 
        #         or  (self.left <= __other.left and __other.right <= self.right)) 

    def intersect(self, __other : object) -> object:
        """ merges intervals and returns their intersect """

        if not isinstance(__other, Interval):
            raise TypeError(f'Cannot merge with object of type `{type(__other)}`.')

        if not self.overlaps(__other):
            raise ValueError("cannot merge intervals with no overlap")

        # find the left and right bounds of the intersection
        left = max(self.left, __other.left)
        right = min(self.right, __other.right)

        # check if the left and right bounds are open
        left_open = self.left_open if left == self.left else __other.left_open
        right_open = __other.right_open if right == __other.right else self.right_open

        # create a new interval object
        return Interval(left, right, left_open, right_open)

    def union(self, __other : object) -> object:
        """ merges overlapping intervals and returns their union """

        if not isinstance(__other, Interval):
            raise TypeError(f'Cannot merge with object of type `{type(__other)}`.')

        if not self.overlaps(__other):
            raise ValueError("cannot merge intervals with no overlap")

        # find the left and right bounds of the union
        left = min(self.left, __other.left)
        right = max(self.right, __other.right)

        # check if the left and right bounds are open
        left_open = self.left_open if left == self.left else __other.left_open
        right_open = __other.right_open if right == __other.right else self.right_open

        # create a new interval object
        return Interval(left, right, left_open, right_open)


    def extend(self, t: float) -> None:
        """ extends time interval """

        if t < self.left:
            self.left = t
        elif t > self.right:
            self.right = t

        return
    
    def __len__(self) -> int:
        return self.right - self.left
    
    def __contains__(self, x: float) -> bool:
        """ checks if `x` is contained in the interval """
        l = self.left < x if self.left_open else self.left <= x
        r = x < self.right  if self.right_open else x <= self.right
        
        return l and r

    def __eq__(self, __other: object) -> bool:

        if not isinstance(__other, Interval):
            raise TypeError(f'Cannot compare with object of type `{type(__other)}`.')

        return abs(self.left - __other.left) < 1e-6 and abs(self.right - __other.right) < 1e-6

    def __gt__(self, __value: object) -> bool:
        if not isinstance(__value, Interval):
            raise TypeError(f'Cannot compare with object of type `{type(__value)}`.')

        if abs(self.left - __value.left) < 1e-6:
            return len(self) > len(__value)
        
        return self.left > __value.left

    def __ge__(self, __value: object) -> bool:
        if not isinstance(__value, Interval):
            raise TypeError(f'Cannot compare with object of type `{type(__value)}`.')

        if abs(self.left - __value.left) < 1e-6:
            return len(self) >= len(__value)
        
        return self.left >= __value.left
    
    def __lt__(self, __value: object) -> bool:
        if not isinstance(__value, Interval):
            raise TypeError(f'Cannot compare with object of type `{type(__value)}`.')
        
        if abs(self.left - __value.left) < 1e-6:
            return len(self) < len(__value)
        
        return self.left < __value.left

    def __le__(self, __value: object) -> bool:
        if not isinstance(__value, Interval):
            raise TypeError(f'Cannot compare with object of type `{type(__value)}`.')
        
        if abs(self.left - __value.left) < 1e-6:
            return len(self) <= len(__value)
        
        return self.left <= __value.left
    
    def __repr__(self) -> str:
        l = '(' if self.left_open else '['
        r = ')' if self.left_open else ']'
        return f'Interval{l}{self.left},{self.right}{r}'
    
    def __hash__(self) -> int:
        return hash(str(self))
    
class EmptyInterval(Interval):
    """ Represents an empty interval """

    def __init__(self):
        super().__init__(np.NAN, np.NAN)
        self.left_open = True
        self.right_open = True

    def is_empty(self) -> bool:
        return True

    def __repr__(self) -> str:
        return 'EmptyInterval()'

    def __contains__(self, x: float) -> bool:
        return False
    

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
    out += '\n   _____ ____        ________  __________________\n  |__  // __ \      / ____/ / / / ____/ ___/ ___/\n   /_ </ / / /_____/ /   / /_/ / __/  \__ \\__ \ \n ___/ / /_/ /_____/ /___/ __  / /___ ___/ /__/ / \n/____/_____/      \____/_/ /_/_____//____/____/ (v1.0)'
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