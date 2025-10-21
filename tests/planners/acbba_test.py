import unittest

from tester import PlannerTester


class TestACBBAPlanner(PlannerTester, unittest.TestCase):
    def toy_planner_config(self):
        return {
            "preplanner": {
                "@type": "earliest",
                "debug": "False",
                "period" : 500,
            },
            "replanner": {
                "@type": "acbba",
                "debug": "False"
            }
        }

    def planner_name(self):
        return "acbba"
    
    def test_multiple_sats_toy(self):
        pass

    def test_single_sat_lakes(self):
        pass

    def test_multiple_sats_lakes(self):
        pass

if __name__ == '__main__':
    # run tests
    unittest.main()