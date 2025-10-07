import unittest
import copy

import numpy as np

from chess3d.simulation import Simulation
from chess3d.utils import print_welcome
from test_planners import TestPlanners


class TestDealerWorker(TestPlanners, unittest.TestCase):
    def planner_name(self) -> str:
        return "dealer-worker"

    def toy_planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "dealer",
                "@mode": "milp",
                "model": "linear",
                "licensePath": "./gurobi.lic",
                # "horizon": 500,
                "period" : 250,
                "maxTasks": 100,
                "debug" : "False"
            }
        }
    
    def lake_planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "dealer",
                "@mode": "milp",
                "model": "static",
                "licensePath": "./gurobi.lic",
                # "horizon": 500,
                "period" : 100,
                "maxTasks": 100,
                "debug" : "False"
            }
        }
    
    def test_single_sat_toy(self):
        # do nothing, cannot test only one sat for these planners
        return

    def test_single_sat_lakes(self):
        # do nothing, cannot test only one sat for these planners
        return

    def test_multiple_sats_toy(self):
        # setup scenario parameters
        duration = 1.0 / 24.0
        grid_name = 'toy_points'
        scenario_name = f'multiple_sat_toy_scenario-{self.planner_name()}'
        connectivity = 'FULL'
        event_name = 'toy_events'
        mission_name = 'toy_missions'

        dealer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        dealer_spacecraft['name'] = 'dealer-sat'
        dealer_spacecraft['@id'] = 'dealer-sat_0'
        dealer_spacecraft['planner'] = self.toy_planner_config()
        dealer_spacecraft['orbitState']['state']['inc'] = 0.0
        dealer_spacecraft['instrument'] = []

        worker_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        worker_spacecraft_1['name'] = 'worker_sat_1'
        worker_spacecraft_1['@id'] = 'worker-sat_1'
        worker_spacecraft_1['planner'] = {}
        worker_spacecraft_1['planner']['replanner'] = {"@type": "worker"}
        worker_spacecraft_1['orbitState']['state']['inc'] = 0.0         # equatorial
        worker_spacecraft_1['instrument'] = self.instruments['TIR']  # thermal infrared instrument

        worker_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        worker_spacecraft_2['name'] = 'worker_sat_2'
        worker_spacecraft_2['@id'] = 'worker-sat_2'
        worker_spacecraft_2['planner'] = {}
        worker_spacecraft_2['planner']['replanner'] = {"@type": "worker"}
        worker_spacecraft_2['orbitState']['state']['inc'] = 0.0     # equatorial
        worker_spacecraft_2['orbitState']['state']['ta'] = 90.0     # 5 deg before worker 1
        worker_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # hyperspectral imager instrument
        
        # terminal welcome message
        print_welcome(f'Planner Test: `{scenario_name}`')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[dealer_spacecraft, 
                                                               worker_spacecraft_1,
                                                               worker_spacecraft_2]
                                                   )


        # # initialize mission
        # self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # # execute mission
        # self.simulation.execute()

        # # print results
        # self.simulation.print_results()

    def test_multiple_sats_lakes(self):
        # setup scenario parameters
        duration = 1.0 / 24.0
        grid_name = 'lake_event_points'
        scenario_name = f'multiple_sat_lake_scenario-{self.planner_name()}'
        connectivity = 'FULL'
        event_name = 'lake_events_seed-1000'
        mission_name = 'lake_missions'

        dealer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        dealer_spacecraft['name'] = 'dealer-sat'
        dealer_spacecraft['@id'] = 'dealer-sat_0'
        dealer_spacecraft['planner'] = self.lake_planner_config()
        dealer_spacecraft['instrument'] = []
        dealer_spacecraft['orbitState']['state']['ta'] = np.average([95,93])  # between both workers
        dealer_spacecraft['science'] = {
                        "@type": "lookup", 
                        "eventsPath" : f"./tests/planners/resources/events/{event_name}.csv"
                    }

        worker_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        worker_spacecraft_1['name'] = 'worker_sat_1'
        worker_spacecraft_1['@id'] = 'worker-sat_1'
        worker_spacecraft_1['planner'] = {}
        worker_spacecraft_1['planner']['replanner'] = {"@type": "worker"}
        worker_spacecraft_1['orbitState']['state']['ta'] = 95.0     
        worker_spacecraft_1['instrument'] = self.instruments['TIR']  # thermal infrared instrument
        
        worker_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        worker_spacecraft_2['name'] = 'worker_sat_2'
        worker_spacecraft_2['@id'] = 'worker-sat_2'
        worker_spacecraft_2['planner'] = {}
        worker_spacecraft_2['planner']['replanner'] = {"@type": "worker"}
        worker_spacecraft_2['orbitState']['state']['ta'] = 93.0     # 3 [deg] before worker 1
        worker_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # hyperspectral imager instrument
        
        # terminal welcome message
        print_welcome(f'Planner Test: `{scenario_name}`')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[dealer_spacecraft, 
                                                               worker_spacecraft_1,
                                                               worker_spacecraft_2]
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
    