from abc import ABC, abstractmethod
import os
import copy

from chess3d.simulation import Simulation
from chess3d.utils import print_welcome


class TestPlanners(ABC):
    def setUp(self) -> None:        
        # load scenario json file
        self.spacecraft_template = {
                    "@id": "thermal_sat_0_0",
                    "name": "thermal_0",
                    "spacecraftBus": {
                        "name": "BlueCanyon",
                        "mass": 20,
                        "volume": 0.5,
                        "orientation": {
                            "referenceFrame": "NADIR_POINTING",
                            "convention": "REF_FRAME_ALIGNED"
                        },
                        "components": {
                            "adcs" : {
                                "maxTorque" : 1000,
                                "maxRate" : 1
                            }
                        }
                    },
                    "instrument": {
                        "name": "VNIR hyper",
                        "@id" : "vnir_hyp_imager",
                        "@type" : "VNIR",
                        "detectorWidth": 6.6e-6,
                        "focalLength": 3.6,  
                        "orientation": {
                            "referenceFrame": "NADIR_POINTING",
                            "convention": "REF_FRAME_ALIGNED"
                        },
                        "fieldOfViewGeometry": { 
                            "shape": "RECTANGULAR", 
                            "angleHeight": 2.5, 
                            "angleWidth": 45.0
                        },
                        "maneuver" : {
                            "maneuverType":"SINGLE_ROLL_ONLY",
                            "A_rollMin": -50,
                            "A_rollMax": 50
                        },
                        "spectral_resolution" : "Multispectral"
                    },
                    "orbitState": {
                        "date": {
                            "@type": "GREGORIAN_UT1",
                            "year": 2020,
                            "month": 1,
                            "day": 1,
                            "hour": 0,
                            "minute": 0,
                            "second": 0
                        },
                        "state": {
                            "@type": "KEPLERIAN_EARTH_CENTERED_INERTIAL",
                            "sma": 7078,
                            "ecc": 0.01,
                            "inc": 60.0,
                            "raan": 0.0,
                            "aop": 0.0,
                            "ta": 95.0
                        }
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "earliest",
                            "period": 500,
                            # "horizon": 500,
                        },
                        # "replanner" : {
                        #     "@type" : "broadcaster",
                        #     "period" : 400
                        # },
                    },
                    # "science" : {
                    #     "@type": "lookup", 
                    #     "eventsPath" : "./tests/planners/resources/events/toy_events.csv"
                    # },
                    "mission" : "Algal blooms monitoring"
            }
        
        # set outdir
        orbitdata_dir = os.path.join('./tests/planners', 'orbit_data')
        if not os.path.isdir(orbitdata_dir): os.mkdir(orbitdata_dir)

        # define known list of instruments
        self.instruments = {
                            "VNIR hyp" : {
                                "name": "VNIR hyper",
                                "@id" : "vnir_hyp_imager",
                                "@type" : "VNIR",
                                "detectorWidth": 6.6e-6,
                                "focalLength": 3.6,  
                                "orientation": {
                                    "referenceFrame": "NADIR_POINTING",
                                    "convention": "REF_FRAME_ALIGNED"
                                },
                                "fieldOfViewGeometry": { 
                                    "shape": "RECTANGULAR", 
                                    "angleHeight": 2.5, 
                                    "angleWidth": 45.0
                                },
                                "maneuver" : {
                                    "maneuverType":"SINGLE_ROLL_ONLY",
                                    "A_rollMin": -50,
                                    "A_rollMax": 50
                                },
                                "spectral_resolution" : "Hyperspectral"
                            },
                            "VNIR multi" : {
                                "name": "VNIR multi",
                                "@id" : "vnir_multi_imager",
                                "@type" : "VNIR",
                                "detectorWidth": 6.6e-6,
                                "focalLength": 3.6,  
                                "orientation": {
                                    "referenceFrame": "NADIR_POINTING",
                                    "convention": "REF_FRAME_ALIGNED"
                                },
                                "fieldOfViewGeometry": { 
                                    "shape": "RECTANGULAR", 
                                    "angleHeight": 2.5, 
                                    "angleWidth": 45.0
                                },
                                "maneuver" : {
                                    "maneuverType":"SINGLE_ROLL_ONLY",
                                    "A_rollMin": -50,
                                    "A_rollMax": 50
                                },
                                "spectral_resolution" : "Multispectral"
                            },
                            "TIR": {
                                "name": "TIR",
                                "@id" : "tir_imager",
                                "@type" : "VNIR",
                                "detectorWidth": 6.6e-6,
                                "focalLength": 3.6,  
                                "orientation": {
                                    "referenceFrame": "NADIR_POINTING",
                                    "convention": "REF_FRAME_ALIGNED"
                                },
                                "fieldOfViewGeometry": { 
                                    "shape": "RECTANGULAR", 
                                    "angleHeight": 2.5, 
                                    "angleWidth": 45.0
                                },
                                "maneuver" : {
                                    "maneuverType":"SINGLE_ROLL_ONLY",
                                    "A_rollMin": -50,
                                    "A_rollMax": 50
                                },
                                "spectral_resolution" : "Multispectral"
                            },
                            "Altimeter": {
                                "name": "Altimeter",
                                "@id" : "altimeter",
                                "@type" : "Altimeter",
                                "chirpBandwidth": 150e6,
                                "pulseWidth": 50e-6,  
                                "orientation": {
                                    "referenceFrame": "NADIR_POINTING",
                                    "convention": "REF_FRAME_ALIGNED"
                                },
                                "fieldOfViewGeometry": { 
                                    "shape": "RECTANGULAR", 
                                    "angleHeight": 2.5, 
                                    "angleWidth": 45.0
                                },
                                "maneuver" : {
                                    "maneuverType":"SINGLE_ROLL_ONLY",
                                    "A_rollMin": -50,
                                    "A_rollMax": 50
                                }
                            }
                        }
        
    def setup_scenario_specs(self, 
                             duration : float, 
                             grid_name : str, 
                             scenario_name : str, 
                             connectivity : str, 
                             event_name : str, 
                             mission_name : str,
                             spacecraft : list = []
                             ) -> dict:
        scenario_specs : dict = {
            "epoch": {
                "@type": "GREGORIAN_UT1",
                "year": 2020,
                "month": 1,
                "day": 1,
                "hour": 0,
                "minute": 0,
                "second": 0
            },
            "duration": duration,
            "propagator": {
                "@type": "J2 ANALYTICAL PROPAGATOR",
            }
        }
        scenario_specs['grid'] = self.setup_grid(grid_name)
        scenario_specs['scenario'] = self.setup_scenario(scenario_name, connectivity, event_name, mission_name)
        scenario_specs['settings'] = self.setup_scenario_settings(scenario_name)
        scenario_specs['spacecraft'] = spacecraft

        return scenario_specs

    def setup_grid(self, grid_name : str) -> dict:
        """Setup the grid for the scenario. """

        assert isinstance(grid_name, str), "grid_name must be a string"

        assert os.path.isfile(f"./tests/planners/resources/grids/{grid_name}.csv"), \
            f"Grid file not found: {grid_name}.csv"

        grid = {
            "@type": "customGrid",
            "covGridFilePath": f"./tests/planners/resources/grids/{grid_name}.csv"
        }
        return [grid]

    def setup_scenario(self, scenario_name : str, connectivity : str, event_name : str, mission_name : str) -> dict:
        """Setup the scenario for the simulation. """

        assert isinstance(scenario_name, str), "scenario_name must be a string"
        assert isinstance(connectivity, str), "connectivity must be a string"
        assert isinstance(event_name, str), "event_name must be a string"
        assert isinstance(mission_name, str), "mission_name must be a string"

        assert os.path.isfile(f"./tests/planners/resources/events/{event_name}.csv"), \
            f"Event file not found: {event_name}.csv"
        assert os.path.isfile(f"./tests/planners/resources/missions/{mission_name}.json"), \
            f"Mission file not found: {mission_name}.json"

        scenario = {
            "connectivity": connectivity,
            "events": {
                "@type": "PREDEF",
                "eventsPath": f"./tests/planners/resources/events/{event_name}.csv"
            },
            "clock" : {
                "@type" : "EVENT"
            },
            "scenarioPath" : "./tests/planners/",
            "name" : scenario_name,
            "missionsPath" : f"./tests/planners/resources/missions/{mission_name}.json"
        }
        return scenario
    
    def setup_scenario_settings(self, full_scenario_name : str) -> dict:
        """ Setup additional scenario settings for orbitpy propagator. """

        assert isinstance(full_scenario_name, str), "scenario_name must be a string"
        assert os.path.isdir(f"./tests/planners/orbit_data"), \
            f"Orbit data directory not found."
        
        # extract relevant scenario name
        scenario_name,*_ = full_scenario_name.split('-')

        # create orbitdata output directory if needed
        scenario_orbitdata_dir = f"./tests/planners/orbit_data/{scenario_name}"
        if not os.path.isdir(scenario_orbitdata_dir): os.mkdir(scenario_orbitdata_dir)

        # create orbitdata settings dictionary
        settings = {
                "coverageType": "GRID COVERAGE",
                "outDir" : f"./tests/planners/orbit_data/{scenario_name}",
            }
        return settings

    @abstractmethod
    def toy_planner_config(self) -> dict:
        """ Returns the planner configuration for the test case. """
    
    @abstractmethod
    def planner_name(self) -> str:
        """ Returns the planner name for the test case. """

    def test_single_sat_toy(self):
        """ Test case for a single satellite with toy events. """
        # setup scenario parameters
        duration = 1.0 / 24.0
        grid_name = 'toy_points'
        scenario_name = f'single_sat_toy_scenario-{self.planner_name()}'
        connectivity = 'FULL'
        event_name = 'toy_events'
        mission_name = 'toy_missions'

        spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        spacecraft['planner'] = self.toy_planner_config()
        spacecraft['orbitState']['state']['inc'] = 0.0

        # terminal welcome message
        print_welcome(f'Planner Test: `{scenario_name}`')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[spacecraft]
                                                   )


        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.print_results()

        print('DONE')

    def test_multiple_sats_toy(self):
        """ Test case for multiple satellites with toy events. """
        pass

    def test_single_sat_lakes(self):
        """ Test case for a single satellite in a lake-monitoring scenario. """
        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'lake_event_points'
        scenario_name = f'single_sat_lake_scenario-{self.planner_name()}'
        connectivity = 'FULL'
        event_name = 'lake_events_seed-1000'
        mission_name = 'lake_missions'

        spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        spacecraft['planner'] = self.toy_planner_config()

        # terminal welcome message
        print_welcome(f'Planner Test: `{scenario_name}`')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[spacecraft]
                                                   )


        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.print_results()

        print('DONE')

    def test_multiple_sats_lakes(self):
        """ Test case for multiple satellites in a lake-monitoring scenario. """
        pass
