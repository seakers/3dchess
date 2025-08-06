import unittest

from chess3d.mission.events import GeophysicalEvent
from chess3d.mission.mission import *
from chess3d.mission.requirements import *
from chess3d.mission.objectives import *
from chess3d.utils import print_welcome

if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Mission Definitions Test')
    
    # run tests
    unittest.main()