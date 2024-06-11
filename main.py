import argparse
import csv
from datetime import datetime, timedelta
import json
import logging
import pandas as pd
import random
import zmq
import concurrent.futures

from dmas.messages import SimulationElementRoles
from dmas.network import NetworkConfig
from dmas.clocks import FixedTimesStepClockConfig, EventDrivenClockConfig
from chess3d.factory import SimulationFactory
from chess3d.nodes.planning.preplanners import *
from chess3d.nodes.planning.replanners import *
from chess3d.manager import SimulationManager
from chess3d.monitor import ResultsMonitor

from chess3d.nodes.states import *
from chess3d.nodes.agent import SimulationAgent
from chess3d.nodes.science.utility import utility_function
from chess3d.nodes.science.reqs import GroundPointMeasurementRequest
from chess3d.nodes.environment import SimulationEnvironment
from chess3d.utils import *

# from satplan.visualizer import Visualizer

"""
======================================================
   _____ ____  ________  __________________
  |__  // __ \/ ____/ / / / ____/ ___/ ___/
   /_ </ / / / /   / /_/ / __/  \__ \\__ \ 
 ___/ / /_/ / /___/ __  / /___ ___/ /__/ / 
/____/_____/\____/_/ /_/_____//____/____/       (v1.0)
======================================================
                Texas A&M - SEAK Lab
======================================================

Wrapper for running DMAS simulations for the 3DCHESS project
"""

def main(   scenario_name : str, 
            scenario_path : str,
            plot_results : bool = False, 
            save_plot : bool = False, 
            level : int = logging.WARNING
        ) -> None:
    """
    Runs Simulation 
    """
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
    events_path = load_events(scenario_path, clock_config, scenario_config_dict, grid_config_dict)

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
    agents = []
    agent_port = port + 6
    if spacecraft_dict is not None:
        for spacecraft in spacecraft_dict:
            # Create spacecraft agents
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
                                                clock_config,
                                                events_path,
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
    ## unpack config
    env_utility_function = scenario_config_dict.get('utility', 'LINEAR')
    
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
                                        utility_function[env_utility_function], 
                                        events_path,
                                        logger=logger)
    
    # ------------------------------------
    # run simulation
    with concurrent.futures.ThreadPoolExecutor(len(agents) + 3) as pool:
        pool.submit(monitor.run, *[])
        pool.submit(manager.run, *[])
        pool.submit(environment.run, *[])
        for agent in agents:                
            agent : SimulationAgent
            pool.submit(agent.run, *[])    

    # TODO convert outputs for satplan visualizer
    # measurements_path = os.path.join(results_path, environment.get_element_name().lower(), 'measurements.csv')
    # performed_measurements : pd.DataFrame = pd.read_csv(measurements_path)

    # spacecraft_names = [spacecraft['name'] for spacecraft in spacecraft_dict]
    # spacecraft_ids = os.listdir(orbitdata_dir)

    # for spacecraft in spacecraft_dict:
    #     spacecraft : dict
    #     name = spacecraft.get('name')
    #     index = spacecraft_dict.index(spacecraft)
    #     spacecraft_id =  "sat" + str(index)
    #     plan_path = orbitdata_dir+'/'+spacecraft_id+'/plan.csv'

    #     with open(plan_path,'w') as csvfile:
    #         csvwriter = csv.writer(csvfile, delimiter=',',
    #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
            
    #         sat_measurements : pd.DataFrame = performed_measurements.query('`measurer` == @name').sort_values(by=['t_img'])
    #         for _, row in sat_measurements.iterrows():
    #             lat,lon = None, None
    #             for measurement_req in measurement_reqs:
    #                 measurement_req : GroundPointMeasurementRequest
    #                 if row['req_id'] in measurement_req.id:
    #                     lat,lon,_ = measurement_req.lat_lon_pos
                
    #             if lat is None and lon is None:
    #                 continue

    #             obs = {
    #                 "start" : row['t_img'],
    #                 "end" : row['t_img'] + dt,
    #                 "location" : {
    #                                 "lat" : lat,
    #                                 "lon" : lon
    #                                 }
    #             }
    #             row_out = [obs["start"],obs["end"],obs["location"]["lat"],obs["location"]["lon"]]
    #             csvwriter.writerow(row_out)

    # if plot_results:
    #     raise NotImplementedError("Visualization integration with `satplan` not yet supported.")
        
    #     visualizer = Visualizer(scenario_path+'/',
    #                             results_path+'/',
    #                             start_date,
    #                             dt,
    #                             scenario_dict.get("duration"),
    #                             orbitdata_dir+'/grid0.csv'
    #                             )
    #     visualizer.process_mission_data()
    #     visualizer.plot_mission()
        
    #     if save_plot:
    #         raise NotImplementedError("Saving of `satplan` animations not yet supported.")
    
    print(f'\nSIMULATION FOR SCENARIO `{scenario_name}` DONE')


# def load_measurement_reqs(scenario_dict : dict, spacecraft_dict : dict, uav_dict : dict, delta : timedelta) -> list:
#     measurement_reqs = []
#     if scenario_dict['scenario']['@type'] == 'RANDOM':
#         reqs_dict = scenario_dict['scenario']['requests']
#         for i_req in range(reqs_dict['n']):
#             if spacecraft_dict:
#                 raise NotImplementedError('random spacecraft scenario not yet implemented.')

#             elif uav_dict: 
#                 max_distance = np.sqrt((reqs_dict['x_bounds'][1] - reqs_dict['x_bounds'][0]) **2 + (reqs_dict['y_bounds'][1] - reqs_dict['y_bounds'][0])**2)
#                 x_pos = reqs_dict['x_bounds'][0] + (reqs_dict['x_bounds'][1] - reqs_dict['x_bounds'][0]) * random.random()
#                 y_pos = reqs_dict['y_bounds'][0] + (reqs_dict['y_bounds'][1] - reqs_dict['y_bounds'][0]) * random.random()
#                 z_pos = 0.0
#                 pos = [x_pos, y_pos, z_pos]
#                 lan_lon_pos = [0.0, 0.0, 0.0]

#             s_max = 100
#             measurements = []
#             required_measurements = reqs_dict['measurement_reqs']
#             for _ in range(random.randint(1, len(required_measurements))):
#                 measurement = random.choice(required_measurements)
#                 while measurement in measurements:
#                     measurement = random.choice(required_measurements)
#                 measurements.append(measurement)


#             # t_start = delta.seconds * random.random()
#             t_start = 0.0
#             # t_end = t_start + (delta.seconds - t_start) * random.random()
#             # t_end = t_end if t_end - t_start > max_distance else t_start + max_distance
#             t_end = delta.seconds
#             t_corr = t_end - t_start

#             req = GroundPointMeasurementRequest(lan_lon_pos, s_max, measurements, t_start, t_end, t_corr, pos=pos)
#             measurement_reqs.append(req)

#     else:
#         initialRequestsPath = scenario_dict['scenario'].get('initialRequestsPath', scenario_path + '/gpRequests.csv')
#         df = pd.read_csv(initialRequestsPath)
            
#         for _, row in df.iterrows():
#             s_max = row.get('severity',row.get('s_max', None))
            
#             measurements_str : str = row['measurements']
#             measurements_str = measurements_str.replace('[','')         # remove left bracket
#             measurements_str = measurements_str.replace(']','')         # remove right bracket
#             measurements_str = measurements_str.replace(', ',',')       # remove spaces
#             measurements_str = measurements_str.replace('\'','')        # remove quotes if any
#             measurements = measurements_str.split(',')                  # plit into measurements

#             t_start = row.get('start time [s]',row.get('t_start', None))
#             t_end =  row.get('t_end', t_start + row.get('duration [s]', None) )
#             t_corr = row.get('t_corr',t_end - t_start)

#             lat = row.get('lat', row.get('lat [deg]', None)) 
#             lon = row.get('lon', row.get('lon [deg]', None))
#             alt = row.get('alt', 0.0)
#             if lat is None or lon is None or alt is None: 
#                 x_pos, y_pos, z_pos = row.get('x_pos', None), row.get('y_pos', None), row.get('z_pos', None)
#                 if x_pos is not None and y_pos is not None and z_pos is not None:
#                     pos = [x_pos, y_pos, z_pos]
#                     lat, lon, alt = 0.0, 0.0, 0.0
#                 else:
#                     raise ValueError('GP Measurement Requests in `gpRequest.csv` must specify a ground position as lat-lon-alt or cartesian coordinates.')
#             else:
#                 pos = None

#             lan_lon_pos = [lat, lon, alt]
#             req = GroundPointMeasurementRequest(lan_lon_pos, s_max, measurements, t_start, t_end, t_corr, pos=pos)
#             measurement_reqs.append(req)

#     return measurement_reqs


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


if __name__ == "__main__":
    
    # read system arguments
    parser = argparse.ArgumentParser(
                    prog='DMAS for 3D-CHESS',
                    description='Simulates an autonomous Earth-Observing satellite mission.',
                    epilog='- TAMU')

    parser.add_argument(    'scenario_name', 
                            help='name of the scenario being simulated',
                            type=str)
    parser.add_argument(    '-p', 
                            '--plot-result',
                            action='store_true',
                            help='creates animated plot of the simulation',
                            required=False,
                            default=False)    
    parser.add_argument(    '-s', 
                            '--save-plot',
                            action='store_true',
                            help='saves animated plot of the simulation as a gif',
                            required=False,
                            default=False) 
    parser.add_argument(    '-d', 
                            '--no-graphic',
                            action='store_true',
                            help='does not draws ascii welcome screen graphic',
                            required=False,
                            default=False)  
    parser.add_argument(    '-l', 
                            '--level',
                            choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL', 'ERROR'],
                            default='WARNING',
                            help='logging level',
                            required=False,
                            type=str)  
                    
    args = parser.parse_args()
    
    scenario_name = args.scenario_name
    plot_results = args.plot_result
    save_plot = args.save_plot
    no_grapgic = args.no_graphic

    levels = {  'DEBUG' : logging.DEBUG, 
                'INFO' : logging.INFO, 
                'WARNING' : logging.WARNING, 
                'CRITICAL' : logging.CRITICAL, 
                'ERROR' : logging.ERROR
            }
    level = levels.get(args.level)

    # terminal welcome message
    if not no_grapgic:
        print_welcome(scenario_name)
    
    # load scenario json file
    scenario_path = f"{scenario_name}" if "./scenarios/" in scenario_name else f'./scenarios/{scenario_name}/'
    scenario_file = open(scenario_path + '/MissionSpecs.json', 'r')
    scenario_dict : dict = json.load(scenario_file)
    scenario_file.close()

    main(scenario_name, scenario_path, plot_results, save_plot, levels)
    