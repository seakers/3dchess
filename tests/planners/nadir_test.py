import unittest

from test_planners import TestPlanners


class TestNadir(TestPlanners, unittest.TestCase):
    def planner_name(self) -> str:
        return "nadir"

    def toy_planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "nadir",
                "debug": "False",
                "horizon": 500,
                "period" : 500,
            }
        }

if __name__ == '__main__':

    # run tests
    unittest.main()