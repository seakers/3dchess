import unittest
import copy

import numpy as np

from chess3d.simulation import Simulation
from chess3d.utils import print_welcome
from tests.planners.tester import PlannerTester


class TestACBBAPlanner(PlannerTester, unittest.TestCase):
    ...


if __name__ == '__main__':
    # run tests
    unittest.main()