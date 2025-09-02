import unittest

from test_planners import TestPlanners

class TestSingleSatMILP(TestPlanners, unittest.TestCase):
    def planner_name(self) -> str:
        return "single-sat-milp"

    def planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "milp",
                "licensePath": "./gurobi.lic",
                "debug": "False",
                # "horizon": 500,
                "period" : 250,
                "maxTasks": 15,
            }
        }
    
if __name__ == '__main__':
    # run tests
    unittest.main()