import unittest

from tests.agents.tester import AgentTester

class TestGroundStationAgents(AgentTester, unittest.TestCase):
    def planner_name(self) -> str:
        return "dynamic-programming"

    def toy_planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "dynamic",
                "debug": "False",
                "horizon": 500,
                "period" : 500,
            }
        }

if __name__ == '__main__':

    # run tests
    unittest.main()