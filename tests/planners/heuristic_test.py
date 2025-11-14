import unittest

from tester import PlannerTester


class TestHeuristic(PlannerTester, unittest.TestCase):
    def planner_name(self) -> str:
        return "heuristic"

    def toy_planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "heuristic",
                "debug": "False",
                # "horizon": 1000,
                "period" : 500,
            }
        }

if __name__ == '__main__':

    # run tests
    unittest.main()