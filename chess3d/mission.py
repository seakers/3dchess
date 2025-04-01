
import concurrent.futures
import datetime
from datetime import timedelta
from itertools import repeat
import logging
import os
import random
from typing import Any
from tqdm import tqdm
import zmq
import numpy as np
import pandas as pd
from typing import Dict

import orbitpy.util
from instrupy.base import Instrument
from orbitpy.util import Spacecraft

from dmas.messages import SimulationElementRoles
from dmas.network import NetworkConfig
from dmas.clocks import *
from dmas.network import NetworkConfig
from dmas.clocks import *

from chess3d.agents.agents import *
from chess3d.agents.planning.replanners.broadcaster import BroadcasterReplanner
from chess3d.agents.planning.rewards import RewardGrid
from chess3d.agents.science.processing import LookupProcessor
from chess3d.nodes.manager import SimulationManager
from chess3d.nodes.monitor import ResultsMonitor
from chess3d.nodes.environment import SimulationEnvironment
from chess3d.agents.orbitdata import OrbitData, TimeInterval
from chess3d.agents.states import *
from chess3d.agents.agent import SimulatedAgent
from chess3d.agents.planning.module import PlanningModule
from chess3d.agents.planning.preplanners.heuristic import HeuristicInsertionPlanner
from chess3d.agents.planning.preplanners.earliest import EarliestAccessPlanner
from chess3d.agents.planning.preplanners.nadir import NadirPointingPlaner
from chess3d.agents.planning.preplanners.dynamic import DynamicProgrammingPlanner
from chess3d.agents.planning.replanners.consensus.acbba import ACBBAPlanner
from chess3d.agents.planning.replanners.consensus.dynamic import DynamicProgrammingACBBAReplanner
from chess3d.agents.science.module import *
from chess3d.agents.science.utility import utility_function, reobservation_strategy
from chess3d.agents.states import SatelliteAgentState, SimulationAgentTypes, UAVAgentState
from chess3d.agents.agent import SimulatedAgent
from chess3d.utils import *


class Mission:
    def __init__(self,
                 results_path : str,
                 orbitdata_dir : str,
                 manager : SimulationManager,
                 environment : SimulationEnvironment,
                 agents : list,
                 monitor : ResultsMonitor,
                 level : int
            ) -> None:
        self.results_path : str = results_path
        self.orbitdata_dir : str = orbitdata_dir
        self.manager : SimulationManager = manager
        self.environment : SimulationEnvironment = environment
        self.agents : list[SimulatedAgent] = agents
        self.monitor : ResultsMonitor = monitor
        self.level=level
        
    def from_dict(mission_specs : dict, overwrite : bool = False, level=logging.WARNING):
        """ Loads simulation from input json """

        # select unsused ports
        port = random.randint(5555, 9999)

        # unpack agent info
        spacecraft_dict : dict = mission_specs.get('spacecraft', None)
        uav_dict        : dict = mission_specs.get('uav', None)
        gstation_dict   : dict = mission_specs.get('groundStation', None)
        scenario_dict   : dict = mission_specs.get('scenario', None)

        # unpack scenario info
        scenario_dict : dict = mission_specs.get('scenario', None)
        grid_dict : dict = mission_specs.get('grid', None)
        settings_dict : dict = mission_specs.get('settings', None)
        
        # load agent names
        agent_names = [SimulationElementRoles.ENVIRONMENT.value]
        if spacecraft_dict: agent_names.extend([spacecraft['name'] for spacecraft in spacecraft_dict])
        if uav_dict:        agent_names.extend([uav['name'] for uav in uav_dict])
        if gstation_dict:   agent_names.extend([gstation['name'] for gstation in gstation_dict])

        # ------------------------------------
        # get scenario name
        scenario_name = scenario_dict.get('name', 'test')
        
        # get scenario path and name
        scenario_path : str = scenario_dict.get('scenarioPath', None)
        if scenario_path is None: raise ValueError(f'`scenarioPath` not contained in input file.')

        # create results directory
        results_path : str = setup_results_directory(scenario_path, scenario_name, agent_names, overwrite)

        # precompute orbit data
        orbitdata_dir = OrbitData.precompute(mission_specs) if spacecraft_dict is not None else None

        # load simulation clock configuration
        clock_config : ClockConfig = SimulationElementsFactory.generate_clock(mission_specs, spacecraft_dict, orbitdata_dir)
        
        # load events
        events_path = SimulationElementsFactory.load_events(scenario_dict, grid_dict, clock_config)

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
        agents : list[SimulatedAgent] = []
        agent_port = port + 6
        if isinstance(spacecraft_dict, list):
            for spacecraft in spacecraft_dict:
                # create satellite agents
                agent = SimulationElementsFactory.generate_agent(
                                                    scenario_name, 
                                                    results_path,
                                                    orbitdata_dir,
                                                    spacecraft,
                                                    spacecraft_dict.index(spacecraft), 
                                                    manager_network_config, 
                                                    agent_port, 
                                                    SimulationAgentTypes.SATELLITE, 
                                                    clock_config,
                                                    level,
                                                    logger
                                                )
                agents.append(agent)
                agent_port += 7

        if uav_dict is not None:
            # TODO Implement UAV agents
            raise NotImplementedError('UAV agents not yet implemented.')

        if isinstance(gstation_dict, list):
            for gndstat in gstation_dict:
                # create ground station agents
                agent = SimulationElementsFactory.generate_agent(
                                                    scenario_name, 
                                                    results_path,
                                                    orbitdata_dir,
                                                    gndstat,
                                                    gstation_dict.index(gndstat), 
                                                    manager_network_config, 
                                                    agent_port, 
                                                    SimulationAgentTypes.GROUND_STATION, 
                                                    clock_config,
                                                    level,
                                                    logger
                                                )
                agents.append(agent)
                agent_port += 7
        
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
        connectivity = scenario_dict.get('connectivity','full').upper()
        environment = SimulationEnvironment(results_path, 
                                            orbitdata_dir,
                                            spacecraft_dict,
                                            uav_dict,
                                            gstation_dict,
                                            env_network_config, 
                                            manager_network_config,
                                            connectivity,
                                            events_path,
                                            level,
                                            logger)
        
        # return initialized mission
        return Mission(results_path, orbitdata_dir, manager, environment, agents, monitor, level)
    
    def execute(self, plot_results : bool = False, save_plot : bool = False) -> None:
        """ executes the simulation """
        # run each simulation element in parallel
        n_pools = len(self.agents) + 3
        with concurrent.futures.ThreadPoolExecutor(n_pools) as pool:
            pool.submit(self.monitor.run, *[])
            pool.submit(self.manager.run, *[])
            pool.submit(self.environment.run, *[])
            for agent in self.agents:                
                agent : SimulatedAgent
                pool.submit(agent.run, *[])  
    
    def print_results(self, precission : int = 5) -> None:
        # define file name
        summary_path = os.path.join(f"{self.results_path}","summary.csv")

        # collect results
        orbitdata : dict = OrbitData.from_directory(self.orbitdata_dir) if self.orbitdata_dir is not None else None

        try:
            observations_performed = pd.read_csv((os.path.join(self.environment.results_path, 'measurements.csv')))
        except pd.errors.EmptyDataError:
            columns = ['observer','t_img','lat','lon','range','look','incidence','zenith','instrument_name']
            observations_performed = pd.DataFrame(data=[],columns=columns)

        # load all senario events
        events = pd.read_csv(self.environment.events_path) 

        # filter out events that do not occurr during this simulation
        events = events[events['start time [s]'] <= self.manager._clock_config.get_total_seconds()] 

        # compile events detected
        event_detections = None
        for agent in self.agents:
            _,agent_name = agent.name.split('/')
            events_detected_path = os.path.join(self.results_path, agent_name, 'events_detected.csv')
            events_detected_temp = pd.read_csv(events_detected_path)

            if event_detections is None: 
                event_detections = events_detected_temp
            else:
                event_detections = pd.concat([event_detections, events_detected_temp], axis=0)

        # compile mesurement requests
        try:
            measurement_reqs = pd.read_csv((os.path.join(self.environment.results_path, 'requests.csv')))
        except pd.errors.EmptyDataError:
            columns = ['ID','Requester','lat [deg]','lon [deg]','Severity','t start','t end','t corr','Measurment Types']
            measurement_reqs = pd.DataFrame(data=[],columns=columns)

        # summarize results
        results_summary : pd.DataFrame = self.summarize_results(orbitdata, observations_performed, events, event_detections, measurement_reqs, precission)

        # log and save results summary
        print(f"\nSIMULATION RESULTS SUMMARY:\n{str(results_summary)}\n\n")
        results_summary.to_csv(summary_path, index=False)

    def summarize_results(self, 
                          orbitdata : dict,
                          observations_performed : pd.DataFrame, 
                          events : pd.DataFrame,
                          event_detections : pd.DataFrame,
                          measurement_reqs : pd.DataFrame,
                          n_decimals : int = 5) -> pd.DataFrame:
        
        # classify observations
        observations_per_gp, events_per_gp, \
            events_observable, events_observed, events_detected, events_requested, \
                events_re_observable, events_re_obs, \
                    events_co_observable, events_co_obs, \
                    events_co_observable_fully, events_co_obs_fully, \
                        events_co_observable_partially, events_co_obs_partially \
                                = self.classify_observations(orbitdata,
                                                             observations_performed, 
                                                             events, 
                                                             event_detections,
                                                             measurement_reqs)       

        # count observations performed
        # n_events, n_unique_event_obs, n_total_event_obs,
        n_observations, n_gps, n_gps_accessible, n_gps_observed, n_gps_with_events, \
            n_events, n_events_observable, n_events_detected, n_events_requested, n_events_observed, n_total_event_obs, \
                n_events_reobservable, n_events_reobserved, n_total_event_re_obs, \
                    n_events_co_observable, n_events_co_obs, n_total_event_co_obs, \
                        n_events_co_observable_fully, n_events_fully_co_obs, n_total_event_fully_co_obs, \
                            n_events_co_observable_partially, n_events_partially_co_obs, n_total_event_partially_co_obs \
                                = self.count_observations(  orbitdata, 
                                                            observations_performed, 
                                                            observations_per_gp,
                                                            events, 
                                                            events_per_gp,
                                                            events_observable,
                                                            events_detected, 
                                                            events_requested,
                                                            events_observed, 
                                                            events_re_observable,
                                                            events_re_obs, 
                                                            events_co_observable,
                                                            events_co_obs, 
                                                            events_co_observable_fully,
                                                            events_co_obs_fully, 
                                                            events_co_observable_partially,
                                                            events_co_obs_partially)
        
        # count probabilities of observations performed
        p_gp_accessible, p_gp_observed, p_event_at_gp, p_event_detected, \
            p_event_obs_if_obs, p_event_re_obs_if_obs, p_event_co_obs_if_obs, p_event_co_obs_fully_if_obs, p_event_co_obs_partially_if_obs, \
                p_event_observable, p_event_observed, p_event_observed_if_observable, p_event_observed_if_detected, p_event_observed_if_observable_and_detected, \
                    p_event_re_observable, p_event_re_obs, p_event_re_obs_if_re_observable, p_event_re_obs_if_detected, p_event_re_obs_if_reobservable_and_detected, \
                        p_event_co_observable, p_event_co_obs, p_event_co_obs_if_co_observable, p_event_co_obs_if_detected, p_event_co_obs_if_co_observable_and_detected, \
                            p_event_co_observable_fully, p_event_co_obs_fully, p_event_co_obs_fully_if_co_observable_fully, p_event_co_obs_fully_if_detected, p_event_co_obs_fully_if_co_observable_fully_and_detected, \
                                p_event_co_observable_partial, p_event_co_obs_partial, p_event_co_obs_partial_if_co_observable_partially, p_event_co_obs_partial_if_detected, p_event_co_obs_partial_if_co_observable_partially_and_detected \
                                    = self.calc_event_probabilities(orbitdata, 
                                                                    observations_performed, 
                                                                    observations_per_gp,
                                                                    events, 
                                                                    events_per_gp,
                                                                    events_observable,
                                                                    events_detected, 
                                                                    events_requested,
                                                                    events_observed, 
                                                                    events_re_observable,
                                                                    events_re_obs, 
                                                                    events_co_observable,
                                                                    events_co_obs, 
                                                                    events_co_observable_fully,
                                                                    events_co_obs_fully, 
                                                                    events_co_observable_partially,
                                                                    events_co_obs_partially)
        
        # calculate event revisit times
        t_gp_reobservation = self.calc_groundpoint_coverage_metrics(observations_per_gp)
        t_event_reobservation = self.calc_event_coverage_metrics(events_observed)

        # count number of GPs observed
        gps_observed : set = {(lat,lon) for _,_,lat,lon,*_ in observations_performed.values}
        n_gps_observed = len(gps_observed)

        # Generate summary
        summary_headers = ['stat_name', 'val']
        summary_data = [
                    # Dates
                    # ['Simulation Start Date', self.environment._clock_config.start_date], 
                    # ['Simulation End Date', self.environment._clock_config.end_date], 

                    # Coverage Metrics #TODO add more
                    ['Ground Points', n_gps],
                    ['Ground Points Accessible', n_gps_accessible],
                    ['Ground Points Observed', n_gps_observed],
                    ['Ground Point Observations', n_observations],
                    ['Ground Points with Events', n_gps_with_events],

                    ['Average GP Reobservation Time [s]', t_gp_reobservation['mean']],
                    ['Standard Deviation of GP Reobservation Time [s]', t_gp_reobservation['std']],
                    ['Median GP Reobservation Time [s]', t_gp_reobservation['median']],
                    ['Average Event Reobservation Time [s]', t_event_reobservation['mean']],
                    ['Standard Deviation of Event Reobservation Time [s]', t_event_reobservation['std']],
                    ['Median Event Reobservation Time [s]', t_event_reobservation['median']],

                    # Counters
                    ['Events', n_events],
                    ['Events Observable', n_events_observable],
                    ['Events Observed', n_events_observed],
                    ['Events Detected', n_events_detected],
                    ['Events Requested', n_events_requested],
                    ['Event Observations', n_total_event_obs],
                    ['Events Re-observable', n_events_reobservable],
                    ['Events Re-observed', n_events_reobserved],
                    ['Event Re-observations', n_total_event_re_obs],
                    ['Events Co-observable', n_events_co_observable],
                    ['Events Co-observed', n_events_co_obs],
                    ['Event Co-observations', n_total_event_co_obs],
                    ['Events Fully Co-observable', n_events_co_observable_fully],
                    ['Events Fully Co-observed', n_events_fully_co_obs],
                    ['Event Full Co-observations', n_total_event_fully_co_obs],
                    ['Events Only Partially Co-observable', n_events_co_observable_partially],
                    ['Events Partially Co-observed', n_events_partially_co_obs],
                    ['Event Partial Co-observations', n_total_event_partially_co_obs],

                    # Ground-Point Coverage Probabilities
                    ['P(Ground Point Accessible)', np.round(p_gp_accessible,n_decimals)],
                    ['P(Ground Point Observed)', np.round(p_gp_observed,n_decimals)],
                    ['P(Event at a GP)', np.round(p_event_at_gp,n_decimals)],

                    # Event Observation Probabilities
                    ['P(Event Observable)', np.round(p_event_observable,n_decimals)],
                    ['P(Event Re-observable)', np.round(p_event_re_observable,n_decimals)],
                    ['P(Event Co-observable)', np.round(p_event_co_observable,n_decimals)],
                    ['P(Event Fully Co-observable)', np.round(p_event_co_observable_fully,n_decimals)],
                    ['P(Event Partially Co-observable)', np.round(p_event_co_observable_partial,n_decimals)],
                    
                    ['P(Event Detected)', np.round(p_event_detected,n_decimals)],
                    ['P(Event Observed)', np.round(p_event_observed,n_decimals)],
                    ['P(Event Re-observed)', np.round(p_event_re_obs,n_decimals)],
                    ['P(Event Co-observed)', np.round(p_event_co_obs,n_decimals)],
                    ['P(Event Fully Co-observed)', np.round(p_event_co_obs_fully,n_decimals)],
                    ['P(Event Partially Co-observed)', np.round(p_event_co_obs_partial,n_decimals)],

                    ['P(Event Observation | Observation)', np.round(p_event_obs_if_obs,n_decimals)],
                    ['P(Event Re-observation | Observation)', np.round(p_event_re_obs_if_obs,n_decimals)],
                    ['P(Event Co-observation | Observation)', np.round(p_event_co_obs_if_obs,n_decimals)],
                    ['P(Event Full Co-observation | Observation)', np.round(p_event_co_obs_partially_if_obs,n_decimals)],
                    ['P(Event Partial Co-observation | Observation)', np.round(p_event_co_obs_fully_if_obs,n_decimals)],

                    ['P(Event Observed | Observable)', np.round(p_event_observed_if_observable,n_decimals)],
                    ['P(Event Re-observed | Re-observable)', np.round(p_event_re_obs_if_re_observable,n_decimals)],
                    ['P(Event Co-observed | Co-observable)', np.round(p_event_co_obs_if_co_observable,n_decimals)],
                    ['P(Event Fully Co-observed | Fully Co-observable)', np.round(p_event_co_obs_fully_if_co_observable_fully,n_decimals)],
                    ['P(Event Partially Co-observed | Partially Co-observable)', np.round(p_event_co_obs_partial_if_co_observable_partially,n_decimals)],
                    
                    ['P(Event Observed | Event Detected)', np.round(p_event_observed_if_detected,n_decimals)],
                    ['P(Event Re-observed | Event Detected)', np.round(p_event_re_obs_if_detected,n_decimals)],
                    ['P(Event Co-observed | Event Detected)', np.round(p_event_co_obs_if_detected,n_decimals)],
                    ['P(Event Co-observed Fully | Event Detected)', np.round(p_event_co_obs_fully_if_detected,n_decimals)],
                    ['P(Event Co-observed Partially | Event Detected)', np.round(p_event_co_obs_partial_if_detected,n_decimals)],

                    ['P(Event Observed | Event Observable and Detected)', np.round(p_event_observed_if_detected,n_decimals)],
                    ['P(Event Re-observed | Event Re-observable and Detected)', np.round(p_event_re_obs_if_detected,n_decimals)],
                    ['P(Event Co-observed | Event Co-observable and Detected)', np.round(p_event_co_obs_if_detected,n_decimals)],
                    ['P(Event Co-observed Fully | Event Fully Co-observable and Detected)', np.round(p_event_co_obs_fully_if_detected,n_decimals)],
                    ['P(Event Co-observed Partially | Event Partially Co-observable and Detected)', np.round(p_event_co_obs_partial_if_detected,n_decimals)],

                    # Simulation Runtime
                    # ['Total Runtime [s]', round(self.environment.t_f - self.environment.t_0, n_decimals)]
                ]

        # TODO: print time-series or reobservations/coobservations

        return pd.DataFrame(summary_data, columns=summary_headers)
                       
    def classify_observations(self, 
                              orbitdata : dict,
                              observations_performed : pd.DataFrame,
                              events : pd.DataFrame,
                              event_detections : pd.DataFrame,
                              measurement_reqs : pd.DataFrame
                              ) -> tuple:
               
        # classify observations per GP
        observations_per_gp : Dict[tuple, list] = { (lat,lon) :   [(observer,t_img,lat_img,lon_img,__,instrument)
                                                                    for observer,t_img,lat_img,lon_img,*__,instrument in observations_performed.values
                                                                    if abs(lat - lat_img) <= 1e-3
                                                                    and abs(lon - lon_img) <= 1e-3]
                                                                    for _,_,lat,lon,*_ in observations_performed.values

                                                    }
        # classify events per ground point
        events_per_gp : Dict[tuple, list] = {(lat,lon): [[t_start,duration,severity,observations_req] 
                                                        for *_,lat_event,lon_event,t_start,duration,severity,observations_req in events.values
                                                        if abs(lat - lat_event) <= 1e-3
                                                        and abs(lon - lon_event) <= 1e-3] 
                                                        for *_,lat,lon,_,_,_,_ in events.values}
        
        # count event presense, detections, and observations
        events_observable : Dict[tuple, list] = {}
        events_detected : Dict[tuple, list] = {}
        events_requested : Dict[tuple, list] = {}
        events_observed : Dict[tuple, list] = {}

        for event in tqdm(events.values, 
                          desc='Calssifying event accesses, detections, and observations', 
                          leave=True):

            event_tuple, access_intervals, matching_detections, matching_requests, matching_observations \
                = self.classify_observation(event, orbitdata, event_detections, measurement_reqs, observations_performed)

            if access_intervals: events_observable[event_tuple] = access_intervals
            if matching_detections: events_detected[event_tuple] = matching_detections
            if matching_requests: events_requested[event_tuple] = matching_requests
            if matching_observations: events_observed[event_tuple] = matching_observations

        assert all([event in events_observable for event in events_observed])

        # find reobserved events
        events_re_observable : dict = { event: access_intervals 
                                        for event,access_intervals in events_observable.items()
                                        if len(access_intervals) > 1}
        events_re_obs : dict = {event: observations[1:] 
                                for event,observations in events_observed.items()
                                if len(observations) > 1}
        
        # find co-observable events
        events_co_observable : Dict[tuple, list] = {}
        events_co_observable_fully : Dict[tuple, list] = {}
        events_co_observable_partially : Dict[tuple, list] = {}

        for event,access_intervals in tqdm(events_observable.items(), desc='Compiling possible co-observations', leave=False):
            # get required measurements for a given event
            *_,observations_req = event
            observations_req : list = str_to_list(observations_req)
            
            # get types of observations that can be performed for this event
            observation_types = list({instrument
                                    for *_, instrument in access_intervals
                                    if instrument in observations_req})
            
            # check if there are observations that satisfy the requirements of the request
            if len(observation_types) > 1:
                # check if valid co-observations match this event
                co_observation_opportunities : set = {  (*_, instrument) 
                                                        for *_, instrument in access_intervals
                                                        if instrument in observations_req
                                                    }                

                if all([observation_req in observation_types for observation_req in observations_req]):
                    # all required observation types were performed; event was fully co-observed
                    events_co_observable_fully[event] = co_observation_opportunities

                else:
                    # some required observation types were performed; event was parially co-observed
                    events_co_observable_partially[event] = co_observation_opportunities

                # event is co-observed
                events_co_observable[event] = co_observation_opportunities

        # find co-observed events
        events_co_obs : Dict[tuple, list] = {}
        events_co_obs_fully : Dict[tuple, list] = {}
        events_co_obs_partially : Dict[tuple, list] = {}

        for event,observations in tqdm(events_observed.items(), desc='Compiling co-observations', leave=False):
            # get required measurements for a given event
            *_,observations_req = event
            observations_req : list = str_to_list(observations_req)
            
            # get types of observations that were performed for this event
            observation_types = list({instrument
                                    for *_, instrument in observations
                                    if instrument in observations_req})
            
            # check if there are observations that satisfy the requirements of the request
            if len(observation_types) > 1:
                # check if valid co-observations match this event
                co_observations : set = {(*_, instrument) 
                                        for *_, instrument in observations
                                        if instrument in observations_req}
                
                # find which observations may have triggered co-observations
                if event in events_requested:
                    requesting_observations = {(lat, lon, t_start, duration, severity, observer, t_img, instrument)
                                            for lat, lon, t_start, duration, severity, observer, t_img, instrument in co_observations
                                            for _, requester, _, _, _, t_start_req, *_ in events_requested[event]
                                            if abs(t_start_req - t_img) <= 1e-3
                                            and requester == observer
                                            }
                    # remove requesting observations from co-observations (if any)
                    co_observations.difference_update(requesting_observations)

                if all([observation_req in observation_types for observation_req in observations_req]):
                    # all required observation types were performed; event was fully co-observed
                    events_co_obs_fully[event] = co_observations

                else:
                    # some required observation types were performed; event was parially co-observed
                    events_co_obs_partially[event] = co_observations

                # event is co-observed
                events_co_obs[event] = co_observations

        assert all([event in events_co_observable for event in events_co_obs])
        assert all([event in events_co_observable and event in events_co_observable_fully for event in events_co_obs_fully])
        assert all([event in events_co_observable for event in events_co_obs_partially])

        return observations_per_gp, events_per_gp, \
                events_observable, events_observed, events_detected, events_requested, \
                    events_re_observable, events_re_obs, \
                        events_co_observable, events_co_obs, \
                        events_co_observable_fully, events_co_obs_fully, \
                            events_co_observable_partially, events_co_obs_partially
    
    def classify_observation(self, 
                             event : tuple, 
                             orbitdata : Dict[str, OrbitData],
                             event_detections : pd.DataFrame, 
                             measurement_reqs : pd.DataFrame, 
                             observations_performed : pd.DataFrame) -> tuple:
        # unpackage event
        event = tuple(event)
        *_,lat,lon,t_start,duration,severity,observations_req = event

        # find accesses that overlook a given event's location
        matching_accesses = [
                                (t_index*agent_orbit_data.time_step, agent_name, instrument)
                                for _, agent_orbit_data in orbitdata.items()
                                for t_index,gp_index,pnt_opt_index,gp_lat,gp_lon,*_,instrument,agent_name in agent_orbit_data.gp_access_data.values
                                if abs(lat - gp_lat) < 1e-3 
                                and abs(lon - gp_lon) < 1e-3
                                and t_start <= t_index*agent_orbit_data.time_step <= t_start+duration
                                and instrument in observations_req
                            ]
        
        # initialize map of compiled access intervals
        access_intervals : Dict[tuple,TimeInterval] = dict()

        # compile list of accesses
        for t_access, agent_name, instrument in matching_accesses:
            if (agent_name,instrument) not in access_intervals:
                time_step = orbitdata[agent_name].time_step 
                access_intervals[(agent_name,instrument)] = TimeInterval(t_access,t_access+time_step)

            elif access_intervals[(agent_name,instrument)].is_during(t_access):
                access_intervals[(agent_name,instrument)].extend(t_access)
        
        # convert to list
        access_intervals : list = [ (access_interval,agent_name,instrument) 
                                    for (agent_name,instrument),access_interval in access_intervals.items()]
        
        # sort in chronological order
        access_intervals.sort()

        # find measurement detections that match this event
        matching_detections = [ (id_req, requester, lat_req, lon_req, severity_req, t_start_req, t_end_req, t_corr_req, observation_types)
                                for id_req, requester, lat_req, lon_req, severity_req, t_start_req, t_end_req, t_corr_req, observation_types in event_detections.values
                                if  abs(lat - lat_req) < 1e-3 
                                and abs(lon - lon_req) < 1e-3
                                and t_start-1e-3 <= t_start_req <= t_end_req <= t_start+duration+1e-3
                                and all([instrument in observations_req for instrument in str_to_list(observation_types)])
                            ]       
        matching_detections.sort(key= lambda a : a[5])

        # find measurement requests that match this event
        matching_requests = [   (id_req, requester, lat_req, lon_req, severity_req, t_start_req, t_end_req, t_corr_req, observation_types)
                                for id_req, requester, lat_req, lon_req, severity_req, t_start_req, t_end_req, t_corr_req, observation_types in measurement_reqs.values
                                if  abs(lat - lat_req) < 1e-3 
                                and abs(lon - lon_req) < 1e-3
                                and t_start-1e-3 <= t_start_req <= t_end_req <= t_start+duration+1e-3
                                and all([instrument in observations_req for instrument in str_to_list(observation_types)])
                            ]       
        matching_requests.sort(key= lambda a : a[5])

        # find observations that overlooked a given event's location
        matching_observations = [   (lat, lon, t_start, duration, severity, observer, t_img, instrument)
                                    for observer,t_img,lat_img,lon_img,*_,instrument in observations_performed.values
                                    if abs(lat - lat_img) < 1e-3 
                                    and abs(lon - lon_img) < 1e-3
                                    and t_start <= t_img <= t_start+duration
                                    and instrument in observations_req  #TODO include better reasoning!
                                ]
        matching_observations.sort(key= lambda a : a[6])

        return event, access_intervals, matching_detections, matching_requests, matching_observations

    def count_observations(self, 
                           orbitdata : dict, 
                           observations_performed : pd.DataFrame, 
                           observations_per_gp : dict,
                           events : pd.DataFrame,
                           events_per_gp : dict,
                           events_observable : dict,
                           events_detected : dict, 
                           events_requested : dict,
                           events_observed : dict, 
                           events_re_observable : dict,
                           events_re_obs : dict, 
                           events_co_observable : dict,
                           events_co_obs : dict, 
                           events_co_observable_fully : dict,
                           events_co_obs_fully : dict, 
                           events_co_observable_partially : dict,
                           events_co_obs_partially : dict
                        ) -> tuple:
        
        # count number of groundpoints and their accessibility
        n_gps = None
        gps_accessible_compiled = set()
        for _,agent_orbitdata in orbitdata.items():
            agent_orbitdata : OrbitData

            # count number of ground points
            n_gps = sum([len(gps.values) for gps in agent_orbitdata.grid_data]) if n_gps is None else n_gps

            # get set of accessible ground points
            gps_accessible : set = {gp_index for _,gp_index,*__ in agent_orbitdata.gp_access_data.values}

            # update set of accessible ground points
            gps_accessible_compiled.update(gps_accessible)
        n_gps_accessible = len(gps_accessible_compiled)
        n_gps_observed = len(observations_per_gp)
        
        # count number of observations performed
        n_observations = len(observations_performed)
        
        # count number of events
        n_events = len(events.values)

        # count number of groundpoints with events
        n_gps_with_events = len(events_per_gp)

        # count events observable
        n_events_observable = len(events_observable)

        # count event detections
        n_events_detected = len(events_detected)

        # count events with meaurement requests
        n_events_requested = len(events_requested)

        # count event observations
        n_events_observed = len(events_observed)
        n_total_event_obs = sum([len(observations) for _,observations in events_observed.items()])

        assert n_events_detected <= n_events_observed
        assert n_total_event_obs <= n_observations

        # count events reobservable
        n_events_reobservable = len(events_re_observable)

        # count event reobservations
        n_events_reobserved = len(events_re_obs)
        n_total_event_re_obs = sum([len(re_observations) for _,re_observations in events_re_obs.items()])

        assert n_events_reobserved <= n_events_observed
        assert n_total_event_re_obs <= n_observations

        # count events co-observable
        n_events_co_observable = len(events_co_observable)
        n_events_co_observable_fully = len(events_co_observable_fully)
        n_events_co_observable_partially = len(events_co_observable_partially)

        # count event co-observations
        n_events_co_obs = len(events_co_obs)
        n_total_event_co_obs = sum([len(co_observations) for _,co_observations in events_co_obs.items()])        

        assert n_events_co_obs <= n_events_observed
        assert n_total_event_co_obs <= n_observations

        n_events_fully_co_obs = len(events_co_obs_fully)
        n_total_event_fully_co_obs = sum([len(full_co_observations) for _,full_co_observations in events_co_obs_fully.items()])        
        
        assert n_events_fully_co_obs <= n_events_observed
        assert n_total_event_fully_co_obs <= n_total_event_co_obs

        n_events_partially_co_obs = len(events_co_obs_partially)
        n_total_event_partially_co_obs = sum([len(partial_co_observations) for _,partial_co_observations in events_co_obs_partially.items()])        

        assert n_events_partially_co_obs <= n_events_observed
        assert n_total_event_partially_co_obs <= n_total_event_co_obs

        assert n_events_co_obs == n_events_fully_co_obs + n_events_partially_co_obs
        assert n_total_event_co_obs == n_total_event_fully_co_obs + n_total_event_partially_co_obs


        # return values
        return n_observations, n_gps, n_gps_accessible, n_gps_observed, n_gps_with_events, \
                n_events, n_events_observable, n_events_detected, n_events_requested, n_events_observed, n_total_event_obs, \
                    n_events_reobservable, n_events_reobserved, n_total_event_re_obs, \
                        n_events_co_observable, n_events_co_obs, n_total_event_co_obs, \
                            n_events_co_observable_fully, n_events_fully_co_obs, n_total_event_fully_co_obs, \
                                n_events_co_observable_partially, n_events_partially_co_obs, n_total_event_partially_co_obs

    def calc_event_probabilities(self,
                                 orbitdata : dict, 
                                 observations_performed : pd.DataFrame, 
                                 observations_per_gp : dict,
                                 events : pd.DataFrame,
                                 events_per_gp : dict,
                                 events_observable : dict,
                                 events_detected : dict, 
                                 events_requested : dict,
                                 events_observed : dict, 
                                 events_re_observable : dict,
                                 events_re_obs : dict, 
                                 events_co_observable : dict,
                                 events_co_obs : dict, 
                                 events_co_observable_fully : dict,
                                 events_co_obs_fully : dict, 
                                 events_co_observable_partially : dict,
                                 events_co_obs_partially : dict
                                ) -> tuple:
    
        # count observations by type
        n_observations, n_gps, n_gps_accessible, n_gps_observed, n_gps_with_events, \
            n_events, n_events_observable, n_events_detected, n_events_requested, n_events_observed, n_total_event_obs, \
                n_events_reobservable, n_events_reobserved, n_total_event_re_obs, \
                    n_events_co_observable, n_events_co_obs, n_total_event_co_obs, \
                        n_events_co_observable_fully, n_events_fully_co_obs, n_total_event_fully_co_obs, \
                            n_events_co_observable_partially, n_events_partially_co_obs, n_total_event_partially_co_obs \
                                = self.count_observations(  orbitdata, 
                                                            observations_performed, 
                                                            observations_per_gp,
                                                            events, 
                                                            events_per_gp,
                                                            events_observable,
                                                            events_detected, 
                                                            events_requested,
                                                            events_observed, 
                                                            events_re_observable,
                                                            events_re_obs, 
                                                            events_co_observable,
                                                            events_co_obs, 
                                                            events_co_observable_fully,
                                                            events_co_obs_fully, 
                                                            events_co_observable_partially,
                                                            events_co_obs_partially)
                    
        # count number of obseved and detected events
        n_events_observed_and_observable = len([event for event in events_observed
                                                if event in events_observable])
        n_events_observed_and_detected = len([event for event in events_observed
                                                if event in events_detected])
        n_events_observed_and_observable_and_detected = len([event for event in events_observed
                                                            if event in events_observable
                                                            and event in events_detected])
        n_events_observable_and_detected = len([event for event in events_observable
                                                if event in events_detected])
            
        # count number of re-obseved and events
        n_events_re_obs_and_reobservable = len([event for event in events_re_obs
                                                if event in  events_re_observable])
        n_events_re_obs_and_detected = len([event for event in events_re_obs
                                            if event in events_detected])
        n_events_re_obs_and_reobservable_and_detected = len([event for event in events_re_obs
                                                            if event in  events_re_observable
                                                            and event in events_detected])
        n_events_re_observable_and_detected = len([event for event in events_re_observable
                                                  if event in events_detected])

        # count number of co-observed and detected events 
        n_events_co_obs_and_co_observable = len([event for event in events_co_obs
                                                if event in events_co_observable])
        n_events_co_obs_and_detected = len([event for event in events_co_obs
                                                if event in events_detected])
        n_events_co_obs_and_co_observable_and_detected = len([event for event in events_co_obs
                                                                if event in events_co_observable
                                                                and event in events_detected])
        n_events_co_observable_and_detected = len([event for event in events_co_observable
                                                    if event in events_detected])

        
        n_events_fully_co_obs_and_fully_co_observable = len([event for event in events_co_obs_fully
                                                            if event in events_co_observable_fully])
        n_events_fully_co_obs_and_detected = len([event for event in events_co_obs_fully
                                                  if event in events_detected])
        n_events_fully_co_obs_and_fully_co_observable_and_detected = len([event for event in events_co_obs_fully
                                                                          if event in events_co_observable_fully
                                                                          and event in events_detected])
        n_events_fully_co_observable_and_detected = len([event for event in events_co_observable_fully
                                                        if event in events_detected])
        
        n_events_partially_co_obs_and_partially_co_observable = len([event for event in events_co_obs_partially
                                                                    if event in events_co_observable_partially])
        n_events_partially_co_obs_and_detected = len([event for event in events_detected
                                                    if event in events_co_obs_partially])
        n_events_partially_co_obs_and_partially_co_observable_and_detected = len([event for event in events_co_observable_partially
                                                                          if event in events_co_obs_partially
                                                                          and event in events_detected])
        n_events_partially_co_observable_and_detected = len([event for event in events_co_observable_partially
                                                            if event in events_detected])

        # calculate probabilities
        p_gp_accessible = n_gps_accessible / n_gps if n_gps > 0 else np.NAN
        p_gp_observed = n_gps_observed / n_gps if n_gps > 0 else np.NAN
        p_event_at_gp = n_gps_with_events / n_gps if n_gps > 0 else np.NAN
        p_event_observable = n_events_observable / n_events if n_events > 0 else np.NAN
        p_event_detected = n_events_detected / n_events if n_events > 0 else np.NAN
        p_event_observed = n_events_observed / n_events if n_events > 0 else np.NAN
        p_event_re_observable = n_events_reobservable / n_events if n_events > 0 else np.NAN
        p_event_re_obs = n_events_reobserved / n_events if n_events > 0 else np.NAN
        p_event_co_observable = n_events_co_observable / n_events if n_events > 0 else np.NAN
        p_event_co_obs = n_events_co_obs / n_events if n_events > 0 else np.NAN
        p_event_co_observable_fully = n_events_co_observable_fully / n_events if n_events > 0 else np.NAN
        p_event_co_obs_fully = n_events_fully_co_obs / n_events if n_events > 0 else np.NAN
        p_event_co_observable_partial = n_events_co_observable_partially / n_events if n_events > 0 else np.NAN
        p_event_co_obs_partial = n_events_partially_co_obs / n_events if n_events > 0 else np.NAN    

        # calculate joint probabilities
        p_event_observed_and_observable = n_events_observed_and_observable / n_events if n_events > 0 else np.NAN
        p_event_observed_and_detected = n_events_observed_and_detected / n_events if n_events > 0 else np.NAN
        p_event_observed_and_observable_and_detected = n_events_observed_and_observable_and_detected / n_events if n_events > 0 else np.NAN
        p_event_observable_and_detected = n_events_observable_and_detected / n_events if n_events > 0 else np.NAN

        p_event_re_obs_and_reobservable = n_events_re_obs_and_reobservable / n_events if n_events > 0 else np.NAN
        p_event_re_obs_and_detected = n_events_re_obs_and_detected / n_events if n_events > 0 else np.NAN
        p_event_re_obs_and_reobservable_and_detected = n_events_re_obs_and_reobservable_and_detected / n_events if n_events > 0 else np.NAN
        p_event_re_observable_and_detected = n_events_re_observable_and_detected / n_events if n_events > 0 else np.NAN

        p_event_co_obs_and_co_observable = n_events_co_obs_and_co_observable / n_events if n_events > 0 else np.NAN
        p_event_co_obs_and_detected = n_events_co_obs_and_detected / n_events if n_events > 0 else np.NAN
        p_event_co_obs_and_co_observable_and_detected = n_events_co_obs_and_co_observable_and_detected / n_events if n_events > 0 else np.NAN
        p_event_co_observable_and_detected = n_events_co_observable_and_detected / n_events if n_events > 0 else np.NAN

        p_event_co_obs_fully_and_co_observable_fully = n_events_fully_co_obs_and_fully_co_observable / n_events if n_events > 0 else np.NAN
        p_event_co_obs_fully_and_detected = n_events_fully_co_obs_and_detected / n_events if n_events > 0 else np.NAN
        p_event_fully_co_obs_and_fully_co_observable_and_detected = n_events_fully_co_obs_and_fully_co_observable_and_detected / n_events if n_events > 0 else np.NAN
        p_event_fully_co_observable_and_detected = n_events_fully_co_observable_and_detected / n_events if n_events > 0 else np.NAN

        p_event_co_obs_partially_and_co_observable_partially = n_events_partially_co_obs_and_partially_co_observable / n_events if n_events > 0 else np.NAN
        p_event_co_obs_partial_and_detected = n_events_partially_co_obs_and_detected / n_events if n_events > 0 else np.NAN
        p_event_partially_co_obs_and_partially_co_observable_and_detected = n_events_partially_co_obs_and_partially_co_observable_and_detected / n_events if n_events > 0 else np.NAN
        p_event_partially_co_observable_and_detected = n_events_partially_co_observable_and_detected / n_events if n_events > 0 else np.NAN

        # calculate conditional probabilities
        p_event_obs_if_obs = n_total_event_obs / n_observations if n_observations > 0 else np.NAN
        p_event_re_obs_if_obs = n_total_event_re_obs / n_observations if n_observations > 0 else np.NAN
        p_event_co_obs_if_obs = n_total_event_co_obs / n_observations if n_observations > 0 else np.NAN
        p_event_co_obs_fully_if_obs = n_total_event_fully_co_obs / n_observations if n_observations > 0 else np.NAN
        p_event_co_obs_partially_if_obs = n_total_event_partially_co_obs / n_observations if n_observations > 0 else np.NAN
        
        p_event_observed_if_observable = p_event_observed_and_observable / p_event_observable if p_event_observable > 0.0 else np.NAN
        p_event_observed_if_detected = p_event_observed_and_detected / p_event_detected if p_event_detected > 0.0 else np.NAN
        p_event_observed_if_observable_and_detected = p_event_observed_and_observable_and_detected / p_event_observable_and_detected if p_event_observable_and_detected > 0.0 else np.NAN
        
        p_event_re_obs_if_re_observable = p_event_re_obs_and_reobservable / p_event_re_observable if p_event_re_observable > 0.0 else np.NAN
        p_event_re_obs_if_detected = p_event_re_obs_and_detected / p_event_detected if p_event_detected > 0.0 else np.NAN
        p_event_re_obs_if_reobservable_and_detected = p_event_re_obs_and_reobservable_and_detected / p_event_re_observable_and_detected if p_event_re_observable_and_detected > 0.0 else np.NAN
        
        p_event_co_obs_if_co_observable = p_event_co_obs_and_co_observable / p_event_co_observable if p_event_co_observable > 0.0 else np.NAN
        p_event_co_obs_if_detected = p_event_co_obs_and_detected / p_event_detected if p_event_detected > 0.0 else np.NAN
        p_event_co_obs_if_co_observable_and_detected = p_event_co_obs_and_co_observable_and_detected / p_event_co_observable_and_detected if p_event_co_observable_and_detected > 0 else np.NAN

        p_event_co_obs_fully_if_co_observable_fully = p_event_co_obs_fully_and_co_observable_fully / p_event_co_observable_fully if p_event_co_observable_fully > 0.0 else np.NAN
        p_event_co_obs_fully_if_detected = p_event_co_obs_fully_and_detected / p_event_detected if p_event_detected > 0.0 else np.NAN
        p_event_co_obs_fully_if_co_observable_fully_and_detected = p_event_fully_co_obs_and_fully_co_observable_and_detected / p_event_fully_co_observable_and_detected if p_event_fully_co_observable_and_detected > 0 else np.NAN

        p_event_co_obs_partial_if_co_observable_partially = p_event_co_obs_partially_and_co_observable_partially / p_event_co_observable_partial if p_event_co_observable_partial > 0.0 else np.NAN
        p_event_co_obs_partial_if_detected = p_event_co_obs_partial_and_detected / p_event_detected if p_event_detected > 0.0 else np.NAN
        p_event_co_obs_partial_if_co_observable_partially_and_detected = p_event_partially_co_obs_and_partially_co_observable_and_detected / p_event_partially_co_observable_and_detected if p_event_partially_co_observable_and_detected > 0 else np.NAN

        return p_gp_accessible, p_gp_observed, p_event_at_gp, p_event_detected, \
                p_event_obs_if_obs, p_event_re_obs_if_obs, p_event_co_obs_if_obs, p_event_co_obs_fully_if_obs, p_event_co_obs_partially_if_obs, \
                    p_event_observable, p_event_observed, p_event_observed_if_observable, p_event_observed_if_detected, p_event_observed_if_observable_and_detected, \
                        p_event_re_observable, p_event_re_obs, p_event_re_obs_if_re_observable, p_event_re_obs_if_detected, p_event_re_obs_if_reobservable_and_detected, \
                            p_event_co_observable, p_event_co_obs, p_event_co_obs_if_co_observable, p_event_co_obs_if_detected, p_event_co_obs_if_co_observable_and_detected, \
                                p_event_co_observable_fully, p_event_co_obs_fully, p_event_co_obs_fully_if_co_observable_fully, p_event_co_obs_fully_if_detected, p_event_co_obs_fully_if_co_observable_fully_and_detected, \
                                    p_event_co_observable_partial, p_event_co_obs_partial, p_event_co_obs_partial_if_co_observable_partially, p_event_co_obs_partial_if_detected, p_event_co_obs_partial_if_co_observable_partially_and_detected

    def calc_groundpoint_coverage_metrics(self,
                                    observations_per_gp: dict
                                    ) -> tuple:
        # event reobservation times
        t_reobservations : list = []
        for _,observations in observations_per_gp.items():
            prev_observation = None
            for observation in observations:
                if prev_observation is None:
                    prev_observation = observation
                    continue

                # get observation times
                _,t,*_ = observation
                _,t_prev,*_ = prev_observation

                # calculate revisit
                t_reobservation = t-t_prev

                # add to list
                t_reobservations.append(t_reobservation)

                # update previous observation
                prev_observation = observation
        
        # compile statistical data
        t_reobservation : dict = {
            'mean' : np.average(t_reobservations) if t_reobservations else np.NAN,
            'std' : np.std(t_reobservations) if t_reobservations else np.NAN,
            'median' : np.median(t_reobservations) if t_reobservations else np.NAN,
            'data' : t_reobservations
        }

        return t_reobservation
    
    def calc_event_coverage_metrics(self, events_observed : dict) -> tuple:
        t_reobservations : list = []
        for event in events_observed:
            prev_observation = None
            for observation in events_observed[event]:
                if prev_observation is None:
                    prev_observation = observation
                    continue

                # get observation times - (lat, lon, t_start, duration, severity, observer, t_img, instrument, observations_req)
                *_,t_prev,_ = prev_observation
                *_,t,_ = observation

                # calculate revisit
                t_reobservation = t-t_prev

                # add to list
                t_reobservations.append(t_reobservation)

                # update previous observation
                prev_observation = observation

        # compile statistical data
        t_reobservation : dict = {
            'mean' : np.average(t_reobservations) if t_reobservations else np.NAN,
            'std' : np.std(t_reobservations) if t_reobservations else np.NAN,
            'median' : np.median(t_reobservations) if t_reobservations else np.NAN,
            'data' : t_reobservations
        }

        return t_reobservation

class SimulationElementsFactory:
    """
    Generates simulation elements according to input file
    """

    def generate_agent(     scenario_name : str, 
                            results_path : str,
                            orbitdata_dir : Any,
                            agent_dict : dict, 
                            agent_index : int,
                            manager_network_config : NetworkConfig, 
                            port : int, 
                            agent_type : SimulationAgentTypes,
                            clock_config : ClockConfig,
                            level : int,
                            logger : logging.Logger
                        ) -> SimulatedAgent:
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
        agent_network_config : NetworkConfig \
            = SimulationElementsFactory.create_agent_network_config(manager_network_config, 
                                                            scenario_name, 
                                                            port)
        
        # load payload
        if agent_type == SimulationAgentTypes.SATELLITE:
            # load orbitdata
            agent_orbitdata : OrbitData = OrbitData.load(orbitdata_dir, agent_name) if orbitdata_dir is not None else None

            # load satellite specs
            agent_specs : Spacecraft = Spacecraft.from_dict(agent_dict)
        else:
            agent_specs : dict = {key: val for key,val in agent_dict.items()}
            agent_specs['payload'] = orbitpy.util.dictionary_list_to_object_list(instruments_dict, Instrument) \
                                     if instruments_dict else []

        if isinstance(clock_config, RealTimeClockConfig):
            # load science module
            science = SimulationElementsFactory.load_science_module(science_dict,
                                                            results_path,
                                                            agent_name,
                                                            agent_network_config,
                                                            logger)

            # load planner module
            planner = SimulationElementsFactory.load_planner_module(planner_dict,
                                                            results_path,
                                                            agent_specs,
                                                            agent_network_config,
                                                            agent_orbitdata, 
                                                            level, 
                                                            logger)

            # create agent
            if agent_type == SimulationAgentTypes.SATELLITE:

                # define initial state
                position_file = os.path.join(orbitdata_dir, f'sat{agent_index}', 'state_cartesian.csv')
                time_data =  pd.read_csv(position_file, nrows=3)
                l : str = time_data.at[1,time_data.axes[1][0]]
                _, _, _, _, dt = l.split(' '); dt = float(dt)

                initial_state = SatelliteAgentState(agent_name,
                                                    orbit_state_dict,
                                                    time_step=dt) 
                
                # return satellite agent
                return RealtimeSatelliteAgent(
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
            processor : DataProcessor = \
                SimulationElementsFactory.load_data_processor(science_dict, agent_name)
            # preplanner : AbstractPreplanner = 

            if agent_type == SimulationAgentTypes.SATELLITE:

                # define initial state
                position_file = os.path.join(orbitdata_dir, 
                                             f'sat{agent_index}', 
                                             'state_cartesian.csv')
                time_data =  pd.read_csv(position_file, nrows=3)
                l : str = time_data.at[1,time_data.axes[1][0]]
                _, _, _, _, dt = l.split(' '); dt = float(dt)

                initial_state = SatelliteAgentState(agent_name,
                                                    orbit_state_dict,
                                                    time_step=dt) 
                
                preplanner, replanner, reward_grid = \
                    SimulationElementsFactory.load_planners(planner_dict, agent_specs, agent_orbitdata, logger)
                
                return SatelliteAgent(agent_name, 
                                      results_path,
                                      agent_network_config,
                                      manager_network_config,
                                      initial_state,
                                      agent_specs,
                                      agent_orbitdata,
                                      processor,
                                      preplanner,
                                      replanner,
                                      reward_grid,
                                      level,
                                      logger)     

            elif agent_type == SimulationAgentTypes.GROUND_STATION:
                lat = agent_specs['latitude']
                lon = agent_specs['longitude']
                alt = agent_specs.get('altitude', 0.0)
                initial_state = GroundStationAgentState(agent_name, lat, lon, alt)

                return GroundStationAgent(agent_name,
                                          results_path,
                                          agent_network_config,
                                          manager_network_config,
                                          initial_state,
                                          agent_specs, 
                                          processor,
                                        #   preplanner,
                                        #   replanner,
                                          level=level,
                                          logger=logger
                                          )

        
        raise NotImplementedError(f"agents of type `{agent_type}` not yet supported by agent factory.")

    def create_agent_network_config(manager_network_config : NetworkConfig, 
                                    scenario_name : str, 
                                    port : int
                                    ) -> NetworkConfig:
        manager_addresses : dict = manager_network_config.get_manager_addresses()
        req_address : str = manager_addresses.get(zmq.REP)[0]
        req_address = req_address.replace('*', 'localhost')

        sub_address : str = manager_addresses.get(zmq.PUB)[0]
        sub_address = sub_address.replace('*', 'localhost')

        pub_address : str = manager_addresses.get(zmq.SUB)[0]
        pub_address = pub_address.replace('*', 'localhost')

        push_address : str = manager_addresses.get(zmq.PUSH)[0]

        return NetworkConfig( 	scenario_name,
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
            n_events = int(events_config_dict.get('numberOfEvents', None))
            if not n_events: raise ValueError('Number of random events not specified in Mission Specs input file.')
            
            # load event parameters
            event_duration = float(events_config_dict.get('duration', None))
            max_severity = float(events_config_dict.get('maxSeverity', None)) 
            min_severity = float(events_config_dict.get('minSeverity', None)) 
            measurements = events_config_dict.get('measurements', None)

            # generate random events
            events = []
            for _ in range(n_events):
                # select ground points for events
                grid : pd.DataFrame = random.choice(grids)
                gp_index = random.randint(0, len(grid)-1)
                gp = grid.iloc[gp_index]
                
                # generate start time 
                t_start = clock_config.get_total_seconds() * random.random()

                # generate severity
                severity = max_severity * random.random() + min_severity

                # generate required measurements
                if len(measurements) < 2: raise ValueError('`measurements` must include more than one sensor')
                n_measurements = random.randint(2,len(measurements)-1)
                required_measurements = random.sample(measurements,k=n_measurements)
                measurements_str = '['
                for req in required_measurements: 
                    if required_measurements.index(req) == 0:
                        measurements_str += req
                    else:
                        measurements_str += f',{req}'
                measurements_str += ']'
                
                # create event
                event = [
                    gp['lat [deg]'],
                    gp['lon [deg]'],
                    t_start,
                    event_duration,
                    severity,
                    measurements_str
                ]

                # add to list of events
                events.append(event)
            
            # compile list of events
            events_path = os.path.join(resources_path, 'random_events.csv')
            events_df = pd.DataFrame(data=events, columns=['lat [deg]','lon [deg]','start time [s]','duration [s]','severity','measurements'])
            
            # save list of events to events path 
            events_df.to_csv(events_path,index=False)

            # return path address
            return events_path
        
    def load_data_processor(science_dict : dict, agent_name : str) -> DataProcessor:
        if science_dict is not None:
            science_dict : dict

            # load science module type
            science_module_type : str = science_dict.get('@type', None)
            if science_module_type is None: raise ValueError(f'science module type not specified in input file.')

            # create an instance of the science module based on the specified science module type
            if science_module_type.lower() == "lookup":
                # load events path
                events_path : str = science_dict.get('eventsPath', None)

                if events_path is None: raise ValueError(f'predefined events path not specified in input file.')

                # create science module
                processor = LookupProcessor(events_path, agent_name)

            else:
                raise NotImplementedError(f'science module of type `{science_module_type}` not yet supported.')
            
            # return science module
            return processor

        # return nothing
        return None          

    def load_science_module(science_dict : dict, 
                            results_path : str, 
                            agent_name : str, 
                            agent_network_config : NetworkConfig, 
                            logger : logging.Logger) -> ScienceModule:
        
        if science_dict is not None:
            science_dict : dict

            # load selected data processor
            processor : DataProcessor = SimulationElementsFactory.load_data_processor(science_dict, agent_name)

            if processor is None: return None

            # return science module
            return ScienceModule(results_path, agent_name, agent_network_config, processor, logger)

        # return nothing
        return None            
   
    def load_planners(planner_dict : dict,
                        agent_specs : object,
                        agent_orbitdata : OrbitData,
                        logger : logging.Logger) -> tuple:
        if planner_dict is not None:
            # get reward grid spes
            reward_grid_params : dict = planner_dict.get('rewardGrid', None)

            if reward_grid_params:
                assert agent_orbitdata is not None

                # get utility function 
                reward_func_name = reward_grid_params.get('reward_function', 'fixed')
                reward_func = utility_function[reward_func_name]

                # get observation startegy
                reobsevation_strategy_name = reward_grid_params.get('reobservation', 'constant')
                reobs_strategy = reobservation_strategy[reobsevation_strategy_name]

                # add parameters
                reward_grid_params['reward_function'] = reward_func
                reward_grid_params['specs'] = agent_specs
                reward_grid_params['grid_data'] = agent_orbitdata.grid_data
                reward_grid_params['reobservation_strategy'] = reobs_strategy

                # create reward gri
                reward_grid = RewardGrid(**reward_grid_params)
            else:
                reward_grid = None

            # get preplanner specs
            preplanner_dict = planner_dict.get('preplanner', None)
            
            if isinstance(preplanner_dict, dict): # preplanner exists
                # get preplanner parameters
                preplanner_type : str = preplanner_dict.get('@type', None)
                if preplanner_type is None: raise ValueError(f'preplanner type within planner module not specified in input file.')

                period = preplanner_dict.get('period', np.Inf)
                horizon = preplanner_dict.get('horizon', period)
                horizon = np.Inf if isinstance(horizon, str) and 'inf' in horizon.lower() else horizon
                debug = bool(preplanner_dict.get('debug', 'false').lower() in ['true', 't'])
                # sharing = bool(preplanner_dict.get('sharing', 'false').lower() in ['true', 't'])

                # initialize preplanner
                if preplanner_type.lower() in ["heuristic"]:
                    period = preplanner_dict.get('period', 500)
                    horizon = preplanner_dict.get('horizon', period)
                    points = preplanner_dict.get('numGroundPoints', np.Inf)

                    if period > horizon: raise ValueError('replanning period must be greater than planning horizon.')

                    preplanner = HeuristicInsertionPlanner(horizon, period, points, debug, logger)

                elif preplanner_type.lower() in ["naive", "fifo", "earliest"]:
                    points = preplanner_dict.get('numGroundPoints', np.Inf)
                    preplanner = EarliestAccessPlanner(horizon, period, points, debug, logger)

                elif preplanner_type.lower() == 'nadir':
                    points = preplanner_dict.get('numGroundPoints', np.Inf)
                    preplanner = NadirPointingPlaner(horizon, period, points, debug, logger)

                elif preplanner_type.lower() in ["dynamic", "dp"]:
                    period = preplanner_dict.get('period', 500)
                    horizon = preplanner_dict.get('horizon', period)
                    
                    if period > horizon: raise ValueError('replanning period must be greater than planning horizon.')

                    preplanner = DynamicProgrammingPlanner(horizon, period, debug, logger)
                
                # elif... # add more preplanners here
                
                else:
                    raise NotImplementedError(f'preplanner of type `{preplanner_dict}` not yet supported.')
            
            else: # no preplanner exists in agent specs
                preplanner = None

            replanner_dict = planner_dict.get('replanner', None)
            if isinstance(replanner_dict, dict):
                replanner_type : str = replanner_dict.get('@type', None)
                if replanner_type is None: raise ValueError(f'replanner type within planner module not specified in input file.')
                debug = bool(replanner_dict.get('debug', 'false').lower() in ['true', 't'])

                if replanner_type.lower() == 'broadcaster':
                    mode = replanner_dict.get('mode', 'periodic').lower()
                    period = replanner_dict.get('period', 500) if mode == 'periodic' else np.Inf

                    replanner = BroadcasterReplanner(mode, period, debug, logger)

                elif replanner_type.lower() == 'acbba': 
                    threshold = replanner_dict.get('threshold', 1)

                    replanner = ACBBAPlanner(
                                             threshold, 
                                             debug,
                                             logger
                                             )
                    
                # elif replanner_type.lower() == 'acbba-dp': 
                #     max_bundle_size = replanner_dict.get('bundle size', 3)
                #     threshold = replanner_dict.get('threshold', 1)
                #     horizon = replanner_dict.get('horizon', np.Inf)

                #     replanner = DynamicProgrammingACBBAReplanner(max_bundle_size, 
                #                                                 threshold, 
                #                                                 horizon,
                #                                                 debug,
                #                                                 logger)
                
                else:
                    raise NotImplementedError(f'replanner of type `{replanner_dict}` not yet supported.')
            else:
                # replanner = None
                replanner = None
        else:
            preplanner, replanner, reward_grid = None, None, None
        
        return preplanner, replanner, reward_grid

    def load_planner_module(planner_dict : dict,
                            results_path : str,
                            agent_specs : object,
                            agent_network_config : NetworkConfig,
                            agent_orbitdata : OrbitData,
                            level : int,
                            logger : logging.Logger
                            ) -> PlanningModule:
        
        preplanner, replanner, reward_grid = SimulationElementsFactory.load_planners(planner_dict, agent_specs, agent_orbitdata, logger)
        
        # create planning module
        return PlanningModule(results_path, 
                              agent_specs,
                              agent_network_config, 
                              reward_grid,
                              preplanner,
                              replanner,
                              agent_orbitdata,
                              level,
                              logger
                            )    
    