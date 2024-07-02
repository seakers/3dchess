
import concurrent.futures
import json
import logging
import os
import random
import zmq

from dmas.messages import SimulationElementRoles
from dmas.network import NetworkConfig
from dmas.clocks import ClockConfig
import pandas as pd

from chess3d.factory import SimulationFactory
from chess3d.nodes.manager import SimulationManager
from chess3d.nodes.monitor import ResultsMonitor
from chess3d.nodes.environment import SimulationEnvironment
from chess3d.agents.orbitdata import OrbitData
from chess3d.agents.states import *
from chess3d.agents.agent import SimulationAgent
from chess3d.utils import *

class Mission:
    def __init__(self,
                 scenario_name : str,
                 scenario_path : str,
                 plot_results : bool = False, 
                 save_plot : bool = False, 
                 level : int = logging.WARNING
            ) -> None:
        # select unsused ports
        port = random.randint(5555, 9999)
        
        # load scenario json file
        scenario_filename = os.path.join(scenario_path, 'MissionSpecs.json')
        with open(scenario_filename, 'r') as scenario_file:
            scenario_dict : dict = json.load(scenario_file)

        # unpack agent info
        spacecraft_dict = scenario_dict.get('spacecraft', None)
        uav_dict        = scenario_dict.get('uav', None)
        gstation_dict   = scenario_dict.get('groundStation', None)
        settings_dict   = scenario_dict.get('settings', None)

        # unpack scenario info
        scenario_config_dict : dict = scenario_dict['scenario']
        grid_config_dict : dict = scenario_dict['grid']
        
        # load agent names
        agent_names = [SimulationElementRoles.ENVIRONMENT.value]
        if spacecraft_dict: agent_names.extend([spacecraft['name'] for spacecraft in spacecraft_dict])
        if uav_dict:        agent_names.extend([uav['name'] for uav in uav_dict])
        if gstation_dict:   agent_names.extend([gstation['name'] for gstation in gstation_dict])

        # load logger level
        if isinstance(settings_dict, dict):
            level = settings_dict.get('logger', logging.WARNING)
            if not isinstance(level, int):
                levels = {
                            'DEBUG': logging.DEBUG, 
                            'WARNING' : logging.WARNING,
                            'CRITICAL' : logging.CRITICAL,
                            'ERROR' : logging.ERROR
                        }
                level = levels[level]
        else:
            level = logging.WARNING

        # ------------------------------------
        # create results directory
        results_path : str = setup_results_directory(scenario_path, agent_names)

        # precompute orbit data
        orbitdata_dir = OrbitData.precompute(scenario_name) if spacecraft_dict is not None else None

        # load simulation clock configuration
        clock_config : ClockConfig = SimulationFactory.generate_clock(scenario_dict,
                                                                    spacecraft_dict, 
                                                                    orbitdata_dir)
        
        # load events
        events_path = self.load_events(scenario_path, clock_config, scenario_config_dict, grid_config_dict)

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
        self.agents : list[SimulationAgent] = []
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
                self.agents.append(agent)
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
        for agent in self.agents:
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
        self.environment = SimulationEnvironment(scenario_path, 
                                            results_path, 
                                            env_network_config, 
                                            manager_network_config,
                                            events_path,
                                            logger=logger)

        # self.monitor : ResultsMonitor = None
        # self.manager : SimulationManager = None
        # self.environment : SimulationEnvironment = None
        # self.agents : list[SimulationAgent] = None

    def load_events(scenario_path : str, 
                clock_config : ClockConfig,
                scenario_config_dict : dict, 
                grid_config_dict : list) -> str:
    
        # get events configuration dictionary
        events_config_dict : dict = scenario_config_dict.get('events', None)

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
            resources_path = os.path.join(scenario_path, 'resources')
            
            # load coverage grids
            grids = []
            for grid_dict in grid_config_dict:
                grid_dict : dict
                grid_type : str = grid_dict.get('@type', None)

                if grid_type is None: raise ValueError('Grid type missing from grid specifications in Mission Specs input file.')

                if grid_type.lower() == 'customgrid':
                    # load custom grid
                    grid_path = grid_config_dict['covGridFilePath']
                    grid = pd.read_csv(grid_path)
                else:
                    # load random grid
                    grid_index = grid_config_dict.index(grid_dict)
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
    
    def execute(self) -> None:
        with concurrent.futures.ThreadPoolExecutor(len(self.agents) + 3) as pool:
            pool.submit(self.monitor.run, *[])
            pool.submit(self.manager.run, *[])
            pool.submit(self.environment.run, *[])
            for agent in self.agents:                
                agent : SimulationAgent
                pool.submit(agent.run, *[])  
        