import copy
import unittest

from chess3d.simulation import Simulation
from chess3d.utils import print_welcome
from tests.planners.tester import PlannerTester


class TestConsensusPlanner(PlannerTester, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        # test case toggles
        self.multiple_sat_toy = True
        self.multiple_sat_lakes : bool = False

    def toy_planner_config(self):
        return {
            "preplanner": {
                "@type": "heuristic",
                "debug": "False",
                "period" : 250,
            },
            "replanner": {
                "@type": "consensus",
                "model": "heuristic_insertion",
                "replanThreshold": 1,
                "debug": "False"
            }
        }

    def planner_name(self):
        return "consensus"

if __name__ == '__main__':
    # run tests
    unittest.main()