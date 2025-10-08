import unittest

from tester import AgentTester

class TestGroundStationAgents(AgentTester, unittest.TestCase):
    def test_initializer(self):
        x = 1 # breakpoint

if __name__ == '__main__':
    # run tests
    unittest.main()