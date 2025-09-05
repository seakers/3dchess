import unittest

from test_planners import TestPlanners

class TestSingleSatMILP(TestPlanners, unittest.TestCase):
    def planner_name(self) -> str:
        return "single-sat-milp"

    def planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "milp",
                "model": "earliest",
                "licensePath": "./gurobi.lic",
                # "horizon": 500,
                "period" : 250,
                "maxTasks": 100,
                "debug" : "True"
            }
        }
    
if __name__ == '__main__':
    # run tests
    unittest.main()
    