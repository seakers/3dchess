import csv
from datetime import datetime, timedelta
import json
import logging
from instrupy.base import Instrument
import orbitpy.util
import pandas as pd
import random
import sys
import zmq
import concurrent.futures

from dmas.messages import SimulationElementRoles
from dmas.network import NetworkConfig
from dmas.clocks import FixedTimesStepClockConfig, EventDrivenClockConfig
from chess3d.nodes.planning.preplanners import FIFOPreplanner
from manager import SimulationManager
from monitor import ResultsMonitor

from nodes.states import *
from nodes.uav import UAVAgent
from nodes.agent import SimulationAgent
from nodes.groundstat import GroundStationAgent
from nodes.satellite import SatelliteAgent
from nodes.planning.planners import PlanningModule
from nodes.science.science import ScienceModule
from nodes.science.utility import utility_function
from nodes.science.reqs import GroundPointMeasurementRequest
from nodes.environment import SimulationEnvironment
from utils import *

from satplan.visualizer import Visualizer

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

Preliminary wrapper used for debugging purposes
"""
def agent_factory(  scenario_name : str, 
                    scenario_path : str,
                    results_path : str,
                    orbitdata_dir : str,
                    agent_dict : dict, 
                    manager_network_config : NetworkConfig, 
                    port : int, 
                    agent_type : SimulationAgentTypes,
                    clock_config : float,
                    logger : logging.Logger,
                    initial_reqs : list
                ) -> SimulationAgent:
    ## unpack mission specs
    agent_name = agent_dict['name']
    planner_dict = agent_dict.get('planner', None)
    science_dict = agent_dict.get('science', None)
    instruments_dict = agent_dict.get('instrument', None)
    orbit_state_dict = agent_dict.get('orbitState', None)

    ## create agent network config
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

    ## load payload
    if instruments_dict:
        payload = orbitpy.util.dictionary_list_to_object_list(instruments_dict, Instrument) # list of instruments
    else:
        payload = []

    ## load science module
    if science_dict is not None and science_dict == "True":
        science = ScienceModule(results_path,scenario_path,agent_name,agent_network_config,logger=logger)
    else:
        science = None
        # raise NotImplementedError(f"Science module not yet implemented.")

    ## load planner module
    if planner_dict is not None:
        planner_dict : dict
        preplanner_type = planner_dict.get('preplanner', None)
        replanner_type = planner_dict.get('replanner', None)
        
        if preplanner_type is not None:
            if preplanner_type == 'FIFO':
                preplanner = FIFOPreplanner()
            else:
                raise NotImplementedError(f'preplanner of type `{preplanner_type}` not yet supported.')
        else:
            preplanner = None

        if replanner_type is not None:
            raise NotImplementedError(f'replanner of type `{replanner_type}` not yet supported.')
        else:
            replanner = None
    else:
        preplanner, replanner = None, None

    planner = PlanningModule(   results_path, 
                                agent_name, 
                                agent_network_config, 
                                utility_function, 
                                preplanner,
                                replanner,
                                initial_reqs)    
        
    ## create agent
    if agent_type == SimulationAgentTypes.UAV:
        ## load initial state 
            pos = agent_dict['pos']
            max_speed = agent_dict['max_speed']
            if isinstance(clock_config, FixedTimesStepClockConfig):
                eps = max_speed * clock_config.dt / 2.0
            else:
                eps = 1e-6

            initial_state = UAVAgentState(  [instrument.name for instrument in payload], 
                                            pos, 
                                            max_speed, 
                                            eps=eps )

            ## create agent
            return UAVAgent(   agent_name, 
                                results_path,
                                manager_network_config,
                                agent_network_config,
                                initial_state,
                                payload,
                                planner,
                                science,
                                logger=logger
                            )

    elif agent_type == SimulationAgentTypes.SATELLITE:
        agent_folder = "sat" + str(0) + '/'

        position_file = orbitdata_dir + agent_folder + 'state_cartesian.csv'
        time_data =  pd.read_csv(position_file, nrows=3)
        l : str = time_data.at[1,time_data.axes[1][0]]
        _, _, _, _, dt = l.split(' ')
        dt = float(dt)

        initial_state = SatelliteAgentState(orbit_state_dict, 
                                            [instrument.name for instrument in payload], 
                                            time_step=dt) 
        
        return SatelliteAgent(
                                agent_name,
                                results_path,
                                manager_network_config,
                                agent_network_config,
                                initial_state, 
                                planner,
                                payload,
                                science,
                                logger=logger
                            )
    else:
        raise NotImplementedError(f"agents of type `{agent_type}` not yet supported by agent factory.")


if __name__ == "__main__":
    
    # read system arguments
    scenario_name = sys.argv[1]
    plot_results = True
    save_plot = False
    level = logging.WARNING

    # terminal welcome message
    print_welcome(scenario_name)

    # create results directory
    results_path = setup_results_directory(scenario_name)

    # select unsused port
    port = random.randint(5555, 9999)
    
    # load scenario json file
    scenario_path = f"{scenario_name}" if "./scenarios/" in scenario_name else f'./scenarios/{scenario_name}/'
    scenario_file = open(scenario_path + '/MissionSpecs.json', 'r')
    scenario_dict : dict = json.load(scenario_file)
    scenario_file.close()

    # read agent names
    spacecraft_dict = scenario_dict.get('spacecraft', None)
    uav_dict = scenario_dict.get('uav', None)
    gstation_dict = scenario_dict.get('groundStation', None)
    settings_dict = scenario_dict.get('settings', None)

    agent_names = [SimulationElementRoles.ENVIRONMENT.value]
    if spacecraft_dict:
        for spacecraft in spacecraft_dict:
            agent_names.append(spacecraft['name'])
    if uav_dict:
        for uav in uav_dict:
            agent_names.append(uav['name'])
    if gstation_dict:
        for gstation in gstation_dict:
            agent_names.append(gstation['name'])

    # precompute orbit data
    orbitdata_dir = precompute_orbitdata(scenario_name) if spacecraft_dict is not None else None

    # read logger level
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

    # read clock configuration
    epoch_dict : dict = scenario_dict.get("epoch")
    year = epoch_dict.get('year', None)
    month = epoch_dict.get('month', None)
    day = epoch_dict.get('day', None)
    hh = epoch_dict.get('hour', None)
    mm = epoch_dict.get('minute', None)
    ss = epoch_dict.get('second', None)
    start_date = datetime(year, month, day, hh, mm, ss)
    delta = timedelta(days=scenario_dict.get("duration"))
    end_date = start_date + delta

    ## define simulation clock
    if spacecraft_dict:
        for spacecraft in spacecraft_dict:
            spacecraft_dict : list
            spacecraft : dict
            index = spacecraft_dict.index(spacecraft)
            agent_folder = "sat" + str(index) + '/'

            position_file = orbitdata_dir + agent_folder + 'state_cartesian.csv'
            time_data =  pd.read_csv(position_file, nrows=3)
            l : str = time_data.at[1,time_data.axes[1][0]]
            _, _, _, _, dt = l.split(' ')
            dt = float(dt)
    else:
        dt = delta.total_seconds()/100

    ## load initial measurement request
    measurement_reqs = []
    if scenario_dict['scenario']['@type'] == 'RANDOM':
        reqs_dict = scenario_dict['scenario']['requests']
        for i_req in range(reqs_dict['n']):
            if spacecraft_dict:
                raise NotImplementedError('random spacecraft scenario not yet implemented.')

            elif uav_dict: 
                max_distance = np.sqrt((reqs_dict['x_bounds'][1] - reqs_dict['x_bounds'][0]) **2 + (reqs_dict['y_bounds'][1] - reqs_dict['y_bounds'][0])**2)
                x_pos = reqs_dict['x_bounds'][0] + (reqs_dict['x_bounds'][1] - reqs_dict['x_bounds'][0]) * random.random()
                y_pos = reqs_dict['y_bounds'][0] + (reqs_dict['y_bounds'][1] - reqs_dict['y_bounds'][0]) * random.random()
                z_pos = 0.0
                pos = [x_pos, y_pos, z_pos]
                lan_lon_pos = [0.0, 0.0, 0.0]

            s_max = 100
            measurements = []
            required_measurements = reqs_dict['measurement_reqs']
            for _ in range(random.randint(1, len(required_measurements))):
                measurement = random.choice(required_measurements)
                while measurement in measurements:
                    measurement = random.choice(required_measurements)
                measurements.append(measurement)


            # t_start = delta.seconds * random.random()
            t_start = 0.0
            # t_end = t_start + (delta.seconds - t_start) * random.random()
            # t_end = t_end if t_end - t_start > max_distance else t_start + max_distance
            t_end = delta.seconds
            t_corr = t_end - t_start

            req = GroundPointMeasurementRequest(lan_lon_pos, s_max, measurements, t_start, t_end, t_corr, pos=pos)
            measurement_reqs.append(req)

    else:
        initialRequestsPath = scenario_dict['scenario'].get('initialRequestsPath', scenario_path + '/gpRequests.csv')
        df = pd.read_csv(initialRequestsPath)
            
        for _, row in df.iterrows():
            s_max = row.get('severity',row.get('s_max', None))
            
            measurements_str : str = row['measurements']
            measurements_str = measurements_str.replace('[','')
            measurements_str = measurements_str.replace(']','')
            measurements_str = measurements_str.replace(', ',',')
            measurements_str = measurements_str.replace('\'','')
            measurements = measurements_str.split(',')

            t_start = row.get('start time [s]',row.get('t_start', None))
            t_end =  row.get('t_end', t_start + row.get('duration [s]', None) )
            t_corr = row.get('t_corr',t_end - t_start)

            lat = row.get('lat', row.get('lat [deg]', None)) 
            lon = row.get('lon', row.get('lon [deg]', None))
            alt = row.get('alt', 0.0)
            if lat is None or lon is None or alt is None: 
                x_pos, y_pos, z_pos = row.get('x_pos', None), row.get('y_pos', None), row.get('z_pos', None)
                if x_pos is not None and y_pos is not None and z_pos is not None:
                    pos = [x_pos, y_pos, z_pos]
                    lat, lon, alt = 0.0, 0.0, 0.0
                else:
                    raise ValueError('GP Measurement Requests in `gpRequest.csv` must specify a ground position as lat-lon-alt or cartesian coordinates.')
            else:
                pos = None

            lan_lon_pos = [lat, lon, alt]
            req = GroundPointMeasurementRequest(lan_lon_pos, s_max, measurements, t_start, t_end, t_corr, pos=pos)
            measurement_reqs.append(req)

    # clock_config = FixedTimesStepClockConfig(start_date, end_date, dt)
    clock_config = EventDrivenClockConfig(start_date, end_date)

    # initialize manager
    manager_network_config = NetworkConfig( scenario_name,
											manager_address_map = {
																	zmq.REP: [f'tcp://*:{port}'],
																	zmq.PUB: [f'tcp://*:{port+1}'],
                                                                    zmq.SUB: [f'tcp://*:{port+2}'],
																	zmq.PUSH: [f'tcp://localhost:{port+3}']
                                                                    }
                                            )


    manager = SimulationManager(agent_names, clock_config, manager_network_config, level)
    logger = manager.get_logger()

    # create results monitor
    monitor_network_config = NetworkConfig( scenario_name,
                                    external_address_map = {zmq.SUB: [f'tcp://localhost:{port+1}'],
                                                            zmq.PULL: [f'tcp://*:{port+3}']}
                                    )
    
    monitor = ResultsMonitor(clock_config, monitor_network_config, logger=logger)

    # create environment
    scenario_config_dict : dict = scenario_dict['scenario']
    env_utility_function = scenario_config_dict.get('utility', 'LINEAR')
    env_network_config = NetworkConfig( manager.get_network_config().network_name,
											manager_address_map = {
													zmq.REQ: [f'tcp://localhost:{port}'],
													zmq.SUB: [f'tcp://localhost:{port+1}'],
                                                    zmq.PUB: [f'tcp://localhost:{port+2}'],
													zmq.PUSH: [f'tcp://localhost:{port+3}']},
											external_address_map = {
													zmq.REP: [f'tcp://*:{port+4}'],
													zmq.PUB: [f'tcp://*:{port+5}']
											})
    
    environment = SimulationEnvironment(scenario_path, 
                                        results_path, 
                                        env_network_config, 
                                        manager_network_config,
                                        utility_function[env_utility_function],
                                        measurement_reqs, 
                                        logger=logger)
    port += 6
    
    # Create agents 
    agents = []
    if spacecraft_dict is not None:
        for d in spacecraft_dict:
            # Create spacecraft agents
            agent = agent_factory(  scenario_name, 
                                    scenario_path, 
                                    results_path, 
                                    orbitdata_dir, 
                                    d, 
                                    manager_network_config, 
                                    port, 
                                    SimulationAgentTypes.SATELLITE, 
                                    clock_config, 
                                    logger,
                                    measurement_reqs
                                )
            agents.append(agent)
            port += 6

    if uav_dict is not None:
        # Create uav agents
        for d in uav_dict:
            agent = agent_factory(  scenario_name, 
                                    scenario_path, 
                                    results_path, 
                                    orbitdata_dir, 
                                    d, 
                                    manager_network_config, 
                                    port, 
                                    SimulationAgentTypes.UAV, 
                                    clock_config, 
                                    logger,
                                    measurement_reqs
                                )
            agents.append(agent)
            port += 6

    if gstation_dict is not None:
        # Create ground station agents
        for d in gstation_dict:
            d : dict
            agent_name = d['name']
            lat = d['latitude']
            lon = d['longitude']
            alt = d['altitude']
            initial_state = GroundStationAgentState(lat,
                                                    lon,
                                                    alt)

            agent = GroundStationAgent( agent_name, 
                                        results_path,
                                        scenario_name,
                                        port,
                                        manager_network_config,
                                        initial_state,
                                        utility_function[env_utility_function],
                                        initial_reqs=measurement_reqs,
                                        logger=logger)
            agents.append(agent)
            port += 6
            
    # run simulation
    with concurrent.futures.ThreadPoolExecutor(len(agents) + 3) as pool:
        pool.submit(monitor.run, *[])
        pool.submit(manager.run, *[])
        pool.submit(environment.run, *[])
        for agent in agents:                
            agent : SimulationAgent
            pool.submit(agent.run, *[])    

    # convert outputs for satplan visualizer
    measurements_path = results_path + '/' + environment.get_element_name().lower() + '/measurements.csv'
    performed_measurements : pd.DataFrame = pd.read_csv(measurements_path)

    spacecraft_names = [spacecraft['name'] for spacecraft in spacecraft_dict]
    spacecraft_ids = os.listdir(orbitdata_dir)

    for spacecraft in spacecraft_dict:
        spacecraft : dict
        name = spacecraft.get('name')
        index = spacecraft_dict.index(spacecraft)
        spacecraft_id =  "sat" + str(index)
        plan_path = orbitdata_dir+'/'+spacecraft_id+'/plan.csv'

        with open(plan_path,'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            
            sat_measurements : pd.DataFrame = performed_measurements.query('`measurer` == @name').sort_values(by=['t_img'])
            for _, row in sat_measurements.iterrows():
                lat,lon = None, None
                for measurement_req in measurement_reqs:
                    measurement_req : GroundPointMeasurementRequest
                    if row['req_id'] in measurement_req.id:
                        lat,lon,_ = measurement_req.lat_lon_pos
                
                if lat is None and lon is None:
                    continue

                obs = {
                    "start" : row['t_img'],
                    "end" : row['t_img'] + dt,
                    "location" : {
                                    "lat" : lat,
                                    "lon" : lon
                                    }
                }
                row_out = [obs["start"],obs["end"],obs["location"]["lat"],obs["location"]["lon"]]
                csvwriter.writerow(row_out)

    # visualizer = Visualizer(scenario_path+'/',
    #                         results_path+'/',
    #                         start_date,
    #                         dt,
    #                         scenario_dict.get("duration"),
    #                         orbitdata_dir+'/grid0.csv'
    #                         )
    # visualizer.process_mission_data()
    # visualizer.plot_mission()
    
    print('\nSIMULATION DONE')