import unittest

from tester import PlannerTester

class TestDynamicProgramming(PlannerTester, unittest.TestCase):
    def planner_name(self) -> str:
        return "dynamic-programming"

    def toy_planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "dynamic",
                "debug": "False",
                "model" : "earliest",
                "sharing": "periodic",
                # "horizon": 250,
                "period" : 250,
            }
        }

if __name__ == '__main__':

    # run tests
    unittest.main()