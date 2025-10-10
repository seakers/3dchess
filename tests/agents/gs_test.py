from collections import defaultdict
import copy
from typing import List
import unittest

from tester import AgentTester

from chess3d.simulation import Simulation
from chess3d.utils import print_welcome

class TestGroundStationAgents(AgentTester, unittest.TestCase):

    def setup_ground_operators(self, gs_network_names : List[str], spacecraft : List[dict]) -> List[dict]:
        """ create ground operator specifications for the scenario. """
        gs_clients = defaultdict(list)
        for sat in spacecraft:
            sat : dict
            dealer_name = sat.get('planner', {}).get('replanner', {}).get('dealerName', None)
            if dealer_name is not None and dealer_name in gs_network_names:
                gs_clients[dealer_name].append(sat['name'])
        
        ground_ops = [
            {
                "name" : gs_network_name,
                "@id" : gs_network_name.lower(),
                # "science" : {
                #     "@type": "lookup", 
                #     "eventsPath" : "./tests/agents/resources/events/toy_events.csv"
                # },
                "planner" : {
                "preplanner": {
                    "@type": "dealer",
                    "@mode": "milp",
                    "model": "static",
                    "licensePath": "./gurobi.lic",
                    # "horizon": 500,
                    "period" : 100,
                    "maxTasks": 100,
                    "debug" : "False",
                    "clients" : gs_clients[gs_network_name]
                    },
                },
                "mission" : "Algal blooms monitoring"
            }

            for gs_network_name in gs_network_names
        ]
        return ground_ops

    def test_initializer(self):
        """ Test case for a single satellite in a lake-monitoring scenario. """
        # setup scenario parameters
        duration = 1.0 / 24.0
        grid_name = 'toy_points'
        scenario_name = f'toy_scenario_gs'
        connectivity = 'FULL' # TODO fix LOS cases
        event_name = 'lake_events_seed-1000'
        mission_name = 'lake_missions'
        nen = 'gs_nen'
        lakes = 'gs_lakes'

        # set toy satellites to equatorial orbit
        spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        spacecraft_1['name'] = 'sat-1'
        spacecraft_1['@id'] = 'sat-1'
        spacecraft_1['orbitState']['state']['sma'] = 42164.0
        spacecraft_1['orbitState']['state']['inc'] = 0.0
        spacecraft_1['groundStationNetwork'] = nen
        spacecraft_1['planner']['replanner'] = {"@type": "worker", "dealerName" : nen}

        spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        spacecraft_2['name'] = 'sat-2'
        spacecraft_2['@id'] = 'sat-2'
        spacecraft_2['orbitState']['state']['inc'] = 0.0
        spacecraft_2['orbitState']['state']['ta'] = 90.0
        spacecraft_2['groundStationNetwork'] = lakes
        spacecraft_2['planner']['replanner'] = {"@type": "worker", "dealerName" : lakes}

        # terminal welcome message
        print_welcome(f'`{scenario_name}` AGENT TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   gs_network_names=[
                                                       nen,
                                                       lakes
                                                    ],
                                                   spacecraft=[
                                                       spacecraft_1, 
                                                       spacecraft_2
                                                    ]
                                                   )


        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.print_results()

if __name__ == '__main__':
    # run tests
    unittest.main()