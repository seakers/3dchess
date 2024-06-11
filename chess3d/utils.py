from enum import Enum
import json
import os
import shutil
import time

import numpy as np

from orbitpy.mission import Mission

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
