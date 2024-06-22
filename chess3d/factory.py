from datetime import timedelta
import logging
import os
import numpy as np
import pandas as pd
import zmq

import orbitpy.util
from instrupy.base import Instrument
from orbitpy.util import Spacecraft

from dmas.network import NetworkConfig
from dmas.clocks import *

from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.planning.module import PlanningModule
from chess3d.agents.planning.planners.broadcaster import Broadcaster
from chess3d.agents.planning.planners.naive import NaivePlanner
from chess3d.agents.satellite import SatelliteAgent
from chess3d.agents.science.module import OracleScienceModule
from chess3d.agents.states import SatelliteAgentState, SimulationAgentTypes, UAVAgentState
from chess3d.agents.agent import SimulationAgent
from chess3d.agents.science.utility import utility_function

class SimulationFactory:
    """
    Generates simulation elements according to input file
    """

    def generate_agent(     scenario_name : str, 
                            scenario_path : str,
                            results_path : str,
                            orbitdata_dir : str,
                            agent_dict : dict, 
                            agent_index : int,
                            manager_network_config : NetworkConfig, 
                            port : int, 
                            agent_type : SimulationAgentTypes,
                            level : int,
                            logger : logging.Logger
                        ) -> SimulationAgent:
        """
        Creates an agent from a list of parameters
        """

        # unpack mission specs
        agent_name = agent_dict.get('name', None)
        planner_dict = agent_dict.get('planner', None)
        science_dict = agent_dict.get('science', None)
        instruments_dict = agent_dict.get('instrument', None)
        orbit_state_dict = agent_dict.get('orbitState', None)

        # create agent network config
        manager_addresses : dict = manager_network_config.get_manager_addresses()
        req_address : str = manager_addresses.get(zmq.REP)[0]
        req_address = req_address.replace('*', 'localhost')

        sub_address : str = manager_addresses.get(zmq.PUB)[0]
        sub_address = sub_address.replace('*', 'localhost')

        pub_address : str = manager_addresses.get(zmq.SUB)[0]
        pub_address = pub_address.replace('*', 'localhost')

        push_address : str = manager_addresses.get(zmq.PUSH)[0]

        agent_network_config = NetworkConfig( 	scenario_name,
                                                manager_address_map = {
                                                        zmq.REQ: [req_address],
                                                        zmq.SUB: [sub_address],
                                                        zmq.PUB: [pub_address],
                                                        zmq.PUSH: [push_address]},
                                                external_address_map = {
                                                        zmq.REQ: [],
                                                            zmq.SUB: [f'tcp://localhost:{port+1}'],
                                                            zmq.PUB: [f'tcp://*:{port+2}']},
                                                internal_address_map = {
                                                        zmq.REP: [f'tcp://*:{port+3}'],
                                                        zmq.PUB: [f'tcp://*:{port+4}'],
                                                        zmq.SUB: [  
                                                                    f'tcp://localhost:{port+5}',
                                                                    f'tcp://localhost:{port+6}'
                                                                ]
                                                })

        # load orbitdata
        if orbitdata_dir is not None:
            agent_orbitdata : OrbitData = OrbitData.load(scenario_path, agent_name)
        else:
            agent_orbitdata = None

        # load payload
        if agent_type == SimulationAgentTypes.SATELLITE:
            agent_specs : Spacecraft = Spacecraft.from_dict(agent_dict)
        else:
            agent_specs : dict = {key: val for key,val in agent_dict.items()}
            agent_specs['payload'] = orbitpy.util.dictionary_list_to_object_list(instruments_dict, Instrument) if instruments_dict else []

        # load science module
        if science_dict is not None:
            science_dict : dict

            # load science module type
            science_module_type : str = science_dict.get('@type', None)
            if science_module_type is None: raise ValueError(f'science module type not specified in input file.')

            # create an instance of the science module based on the specified science module type
            if science_module_type.lower() == "oracle":
                events_path : str = science_dict.get('eventsPath', None)

                if events_path is None: raise ValueError(f'predefined events path not specified in input file.')

                science = OracleScienceModule(results_path, 
                                              events_path, 
                                              agent_name, 
                                              agent_network_config, 
                                              logger)
            else:
                raise NotImplementedError(f'science module of type `{science_module_type}` not yet supported.')
        else:
            science = None

        # load planner module
        if planner_dict is not None:
            planner_dict : dict
            
            preplanner_dict = planner_dict.get('preplanner', None)
            if isinstance(preplanner_dict, dict):
                preplanner_type : str = preplanner_dict.get('@type', None)
                if preplanner_type is None: raise ValueError(f'preplanner type within planner module not specified in input file.')

                period = preplanner_dict.get('period', np.Inf)
                horizon = preplanner_dict.get('horizon', np.Inf)

                if preplanner_type.lower() == "naive":
                    preplanner = NaivePlanner(period, horizon, logger)
                # elif...
                else:
                    raise NotImplementedError(f'preplanner of type `{preplanner_dict}` not yet supported.')
            else:
                preplanner = None

            replanner_dict = planner_dict.get('replanner', None)
            if isinstance(replanner_dict, dict):
                replanner_type : str = replanner_dict.get('@type', None)
                if replanner_type is None: raise ValueError(f'replanner type within planner module not specified in input file.')

                if replanner_type.lower() == 'broadcaster':
                    replanner = Broadcaster(logger)

        #         # elif replanner_type == 'ACBBA': #TODO
        #         #     max_bundle_size = replanner_dict.get('bundle size', 3)
        #         #     dt_converge = replanner_dict.get('dt_convergence', 0.0)
        #         #     period = replanner_dict.get('period', 60.0)
        #         #     threshold = replanner_dict.get('threshold', 1)
        #         #     horizon = replanner_dict.get('horizon', delta.total_seconds())

        #         #     replanner = ACBBAReplanner(utility_func, max_bundle_size, dt_converge, period, threshold, horizon)
                
                else:
                    raise NotImplementedError(f'replanner of type `{replanner_dict}` not yet supported.')
            else:
                # replanner = None
                replanner = None
        else:
            preplanner, replanner, = None, None
        
        # create planning module
        planner = PlanningModule(   results_path, 
                                    agent_specs,
                                    agent_network_config, 
                                    preplanner,
                                    replanner,
                                    agent_orbitdata,
                                    level,
                                    logger
                                )    
        
        # create agent
        if agent_type == SimulationAgentTypes.SATELLITE:
            position_file = os.path.join(orbitdata_dir, f'sat{agent_index}', 'state_cartesian.csv')
            time_data =  pd.read_csv(position_file, nrows=3)
            l : str = time_data.at[1,time_data.axes[1][0]]
            _, _, _, _, dt = l.split(' '); dt = float(dt)

            initial_state = SatelliteAgentState(agent_name,
                                                orbit_state_dict,
                                                time_step=dt) 
            
            return SatelliteAgent(
                                    agent_name,
                                    results_path,
                                    manager_network_config,
                                    agent_network_config,
                                    initial_state, 
                                    agent_specs,
                                    planner,
                                    science,
                                    logger=logger
                                )
        else:
            raise NotImplementedError(f"agents of type `{agent_type}` not yet supported by agent factory.")

    def generate_clock(scenario_dict : dict, 
                       spacecraft_dict : list, 
                       orbitdata_dir : str) -> ClockConfig:
        """
        Generates a `ClockConfig` object based on the given parameters
        """
        # unpack clock config information
        clock_dict : dict = scenario_dict['scenario'].get('clock', None)
        clock_type : str = clock_dict.get('@type', None)
        if not clock_type: raise ValueError('Clock type not defined in inpt file.')

        # load simulation start and end dates
        epoch_dict : dict = scenario_dict.get("epoch"); epoch_dict.pop('@type')
        start_date = datetime(**epoch_dict)
        delta = timedelta(days=scenario_dict.get("duration"))
        end_date = start_date + delta

        # generate simulation clock config 
        if clock_type.lower() == 'step': # generate fixed time-step clock
            
            # check if spacecraft are present in the simulation
            if spacecraft_dict: # use propagator time-step
                # check for existance of orbitdata
                if not orbitdata_dir: raise ImportError('Cannot initialize spacecraft agents. Orbit data was not loaded successfully.')

                # load orbit data info
                position_file = os.path.join(orbitdata_dir, "sat0", 'state_cartesian.csv')
                time_data =  pd.read_csv(position_file, nrows=3)
                l : str = time_data.at[1,time_data.axes[1][0]]
                
                # get timestep from propagated orbit data
                _, _, _, _, dt = l.split(' ')
                dt = float(dt)
            
            else: # use user-defined time-step
                dt = float(clock_dict.get('stepSize', None))
                if dt is None: raise ValueError('`stepSize` not defined in input file.')

            # return clock config
            return FixedTimesStepClockConfig(start_date, end_date, dt)

        else:
            # return event-driven clock config
            return EventDrivenClockConfig(start_date, end_date)