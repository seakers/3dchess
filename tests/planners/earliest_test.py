import unittest

from test_planners import TestPlanners


class TestEarliest(TestPlanners, unittest.TestCase):
    def planner_name(self) -> str:
        return "earliest"

    def planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "earliest",
                "debug": "False",
                # "horizon": 1000,
                "period" : 500,
            }
        }

if __name__ == '__main__':

    # run tests
    unittest.main()