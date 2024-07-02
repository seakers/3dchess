
import concurrent.futures
import datetime
from datetime import timedelta
import logging
import os
import random
import zmq
import numpy as np
import pandas as pd

import orbitpy.util
from instrupy.base import Instrument
from orbitpy.util import Spacecraft

from dmas.messages import SimulationElementRoles
from dmas.network import NetworkConfig
from dmas.clocks import *
from dmas.network import NetworkConfig
from dmas.clocks import *

from chess3d.nodes.manager import SimulationManager
from chess3d.nodes.monitor import ResultsMonitor
from chess3d.nodes.environment import SimulationEnvironment
from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.states import *
from chess3d.agents.agent import SimulationAgent
from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.planning.module import PlanningModule
from chess3d.agents.planning.planners.broadcaster import Broadcaster
from chess3d.agents.planning.planners.consensus.acbba import ACBBAPlanner
from chess3d.agents.planning.planners.naive import NaivePlanner
from chess3d.agents.satellite import SatelliteAgent
from chess3d.agents.science.module import OracleScienceModule
from chess3d.agents.science.utility import utility_function
from chess3d.agents.states import SatelliteAgentState, SimulationAgentTypes, UAVAgentState
from chess3d.agents.agent import SimulationAgent
from chess3d.utils import *


class Mission:
    def __init__(self,
                 manager : SimulationManager,
                 environment : SimulationEnvironment,
                 agents : list,
                 monitor : ResultsMonitor
            ) -> None:
        self.manager : SimulationManager = manager
        self.environment : SimulationEnvironment = environment
        self.agents : list[SimulationAgent] = agents
        self.monitor : ResultsMonitor = monitor

    def execute(self, plot_results : bool = False, save_plot : bool = False) -> None:
        """ executes the simulation """
        with concurrent.futures.ThreadPoolExecutor(len(self.agents) + 3) as pool:
            pool.submit(self.monitor.run, *[])
            pool.submit(self.manager.run, *[])
            pool.submit(self.environment.run, *[])
            for agent in self.agents:                
                agent : SimulationAgent
                pool.submit(agent.run, *[])  
        
    def from_dict(d : dict, level=logging.WARNING):
        """ Loads simulation from input json """

        # select unsused ports
        port = random.randint(5555, 9999)

        # unpack agent info
        spacecraft_dict : dict = d.get('spacecraft', None)
        uav_dict        : dict = d.get('uav', None)
        gstation_dict   : dict = d.get('groundStation', None)
        scenario_dict   : dict = d.get('scenario', None)

        # unpack scenario info
        scenario_dict : dict = d.get('scenario', None)
        grid_dict : dict = d.get('grid', None)
        
        # load agent names
        agent_names = [SimulationElementRoles.ENVIRONMENT.value]
        if spacecraft_dict: agent_names.extend([spacecraft['name'] for spacecraft in spacecraft_dict])
        if uav_dict:        agent_names.extend([uav['name'] for uav in uav_dict])
        if gstation_dict:   agent_names.extend([gstation['name'] for gstation in gstation_dict])

        # ------------------------------------
        # get scenario name
        scenario_name = scenario_dict.get('name', 'test')
        
        # get scenario path
        scenario_path : str = scenario_dict.get('scenarioPath', None)
        if scenario_path is None: raise ValueError(f'`scenarioPath` not contained in input file.')

        # create results directory
        results_path : str = setup_results_directory(scenario_path, agent_names)

        # precompute orbit data
        orbitdata_dir = OrbitData.precompute(d) if spacecraft_dict is not None else None

        # load simulation clock configuration
        clock_config : ClockConfig = SimulationFactory.generate_clock(d, spacecraft_dict, orbitdata_dir)
        
        # load events
        events_path = SimulationFactory.load_events(scenario_dict, grid_dict, clock_config)

        # ------------------------------------
        # initialize manager
        manager_network_config = NetworkConfig( scenario_name,
                                                manager_address_map = {
                                                                        zmq.REP: [f'tcp://*:{port}'],
                                                                        zmq.PUB: [f'tcp://*:{port+1}'],
                                                                        zmq.SUB: [f'tcp://*:{port+2}'],
                                                                        zmq.PUSH: [f'tcp://localhost:{port+3}']
                                                                        }
                                                )


        manager = SimulationManager(results_path, agent_names, clock_config, manager_network_config, level)
        logger = manager.get_logger()

        # ------------------------------------
        # create results monitor
        monitor_network_config = NetworkConfig( scenario_name,
                                        external_address_map = {zmq.SUB: [f'tcp://localhost:{port+1}'],
                                                                zmq.PULL: [f'tcp://*:{port+3}']}
                                        )
        
        monitor = ResultsMonitor(clock_config, monitor_network_config, logger=logger)
        
        # ------------------------------------
        # create agents 
        agents : list[SimulationAgent] = []
        agent_port = port + 6
        if isinstance(spacecraft_dict, list):
            for spacecraft in spacecraft_dict:
                agent = SimulationFactory.generate_agent(
                                                    scenario_name, 
                                                    scenario_path,
                                                    results_path,
                                                    orbitdata_dir,
                                                    spacecraft,
                                                    spacecraft_dict.index(spacecraft), 
                                                    manager_network_config, 
                                                    agent_port, 
                                                    SimulationAgentTypes.SATELLITE, 
                                                    level,
                                                    logger
                                                )
                agents.append(agent)
                agent_port += 7

        if uav_dict is not None:
            # TODO Implement UAV agents
            raise NotImplementedError('UAV agents not yet implemented.')

        if gstation_dict is not None:
            # TODO Implement ground station agents
            raise NotImplementedError('Ground Station agents not yet implemented.')
        
        # ------------------------------------
        # create environment
        
        ## subscribe to all elements in the network
        env_subs = []
        for agent in agents:
            agent_pubs : str = agent._network_config.external_address_map[zmq.PUB]
            for agent_pub in agent_pubs:
                env_sub : str = agent_pub.replace('*', 'localhost')
                env_subs.append(env_sub)
        
        ## create network config
        env_network_config = NetworkConfig( manager.get_network_config().network_name,
                                                manager_address_map = {
                                                        zmq.REQ: [f'tcp://localhost:{port}'],
                                                        zmq.SUB: [f'tcp://localhost:{port+1}'],
                                                        zmq.PUB: [f'tcp://localhost:{port+2}'],
                                                        zmq.PUSH: [f'tcp://localhost:{port+3}']},
                                                external_address_map = {
                                                        zmq.REP: [f'tcp://*:{port+4}'],
                                                        zmq.PUB: [f'tcp://*:{port+5}'],
                                                        zmq.SUB: env_subs
                                                })
        
        ## initialize environment
        environment = SimulationEnvironment(scenario_path, 
                                            results_path, 
                                            env_network_config, 
                                            manager_network_config,
                                            events_path,
                                            logger=logger)
        
        # return initialized mission
        return Mission(manager, environment, agents, monitor)

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

                elif replanner_type.lower() == 'acbba': 
                    max_bundle_size = replanner_dict.get('bundle size', 3)
                    threshold = replanner_dict.get('threshold', 1)
                    horizon = replanner_dict.get('horizon', np.Inf)
                    utility_func_name = replanner_dict.get('utility', 'fixed')
                    utility_func = utility_function[utility_func_name]

                    replanner = ACBBAPlanner(utility_func, 
                                             max_bundle_size, 
                                             threshold, 
                                             horizon,
                                             logger)
                
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

    def generate_clock(mission_specs : dict, 
                       spacecraft_dict : list, 
                       orbitdata_dir : str) -> ClockConfig:
        """
        Generates a `ClockConfig` object based on the given parameters
        """
        # unpack clock config information
        clock_dict : dict = mission_specs['scenario'].get('clock', None)
        clock_type : str = clock_dict.get('@type', None)
        if not clock_type: raise ValueError('Clock type not defined in inpt file.')

        # load simulation start and end dates
        epoch_dict : dict = mission_specs.get("epoch"); epoch_dict.pop('@type')
        start_date = datetime(**epoch_dict)
        delta = timedelta(days=mission_specs.get("duration"))
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

    def load_events(scenario_dict : dict, 
                    grid_dict : list,
                    clock_config : ClockConfig
                    ) -> str:

        # get events configuration dictionary
        events_config_dict : dict = scenario_dict.get('events', None)

        # check if events configuration exists in input file
        if not events_config_dict: raise ValueError('Missing events configuration in Mission Specs input file.')

        # check events configuration format
        events_type : str = events_config_dict.get('@type', None)
        if events_type is None:
            raise ValueError('Event type missing in Mission Specs input file.')
        
        if events_type.lower() == 'predef': # load predefined events
            events_path : str = events_config_dict.get('eventsPath', None) 
            if not events_path: 
                raise ValueError('Path to predefined events not goind in Mission Specs input file.')
            else:
                return events_path
            
        if events_type.lower() == 'random': # generate random events
            # get path to resources directory
            scenario_path = scenario_dict['scenarioPath']
            resources_path = os.path.join(scenario_path, 'resources')
            
            # load coverage grids
            grids = []
            for grid_dict in grid_dict:
                grid_dict : dict
                grid_type : str = grid_dict.get('@type', None)

                if grid_type is None: raise ValueError('Grid type missing from grid specifications in Mission Specs input file.')

                if grid_type.lower() == 'customgrid':
                    # load custom grid
                    grid_path = grid_dict['covGridFilePath']
                    grid = pd.read_csv(grid_path)
                else:
                    # load random grid
                    grid_index = grid_dict.index(grid_dict)
                    grid_filename = f'{grid_type}_grid{grid_index}.csv'
                    grid_path = os.path.join(resources_path, grid_filename)
                    grid = pd.read_csv(grid_path)

                grids.append(grid)

            # load number of events to be generated
            n_events = int(events_config_dict.get('n_events', None))
            if not n_events: raise ValueError('Number of random events not specified in Mission Specs input file.')
            
            # load event parameters
            sim_duration = clock_config.get_total_seconds()
            event_duration = float(events_config_dict.get('duration', None)) * 3600
            severity = float(events_config_dict.get('severity', None))
            measurements = events_config_dict.get('measurements', None)
            n_measurements = int(events_config_dict.get('n_measurements', None))

            # generate random events
            events = []
            for _ in range(n_events):
                # select ground points for events
                grid : pd.DataFrame = random.choice(grids)
                gp_index = random.randint(0, len(grid)-1)
                gp = grid.iloc[gp_index]
                
                # create event
                event = [
                    gp['lat [deg]'],
                    gp['lon [deg]'],
                    random.random()*sim_duration,
                    event_duration,
                    severity,
                    random.sample(measurements,k=n_measurements)
                ]

                # add to list of events
                events.append(event)
            
            # compile list of events
            events_path = os.path.join(resources_path, 'random_events.csv')
            events_df = pd.DataFrame(events, columns=['lat [deg]','lon [deg]','start time [s]','duration [s]','severity','measurements'])
            
            # save list of events to events path 
            events_df.to_csv(events_path)

            # return path address
            return events_path
       
        