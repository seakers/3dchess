from typing import List, Tuple
from copy import deepcopy
import os

import json
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

import networkx as nx

import seaborn as sns
import matplotlib.pyplot as plt

from orbitpy.mission import Mission
from chess3d.utils import print_welcome

# ========================================
#               Constants
# ========================================
R = 6378 # Earth radius in km    
GM = 3.986004418e14 * (1/1000)**3 # [km^3/s^2]
ORBITAL_PARAM = {
                    "@type": "KEPLERIAN_EARTH_CENTERED_INERTIAL",
                    "sma": R,
                    "ecc": 0.01,
                    "inc": 60.0,
                    "raan": 0.0,
                    "aop": 98.0,
                    "ta": 0.0
                }
MISSION_TEMPLATE = {
                    "epoch": {
                        "@type": "GREGORIAN_UT1",
                        "year": 2020,
                        "month": 1,
                        "day": 1,
                        "hour": 0,
                        "minute": 0,
                        "second": 0
                    },
                    "duration": 0.75,
                    "propagator": {
                        "@type": "J2 ANALYTICAL PROPAGATOR",
                        # "stepSize": 10
                    },
                    "spacecraft": [],
                    "grid": [
                        {
                            "@type": "customGrid",
                            "covGridFilePath": "./grids/toy.csv"
                        }
                    ],
                    "scenario" : {
                        "connectivity": "LOS"
                    },
                    "settings": {
                        "coverageType": "GRID COVERAGE",
                        "outDir" : "./orbits"
                    }
                }
SPACECRAFT_TEMPLATE = {
                    "@id": "sat_temp",
                    "name": "sat_temp",
                    "spacecraftBus": {
                        "name": "BlueCanyon",
                        "mass": 20,
                        "volume": 0.5,
                        "orientation": {
                            "referenceFrame": "NADIR_POINTING",
                            "convention": "REF_FRAME_ALIGNED"
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
                            "angleWidth": 2.5
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
                            "sma": 7078, # ~700 km altitude
                            "ecc": 0.01,
                            "inc": 60.0,
                            "raan": 0.0,
                            "aop": 98.0,
                            "ta": 0.0
                        }
                    }
            }

# ========================================
#   Walker Delta Constellation Functions
# ========================================
def generate_walker_delta_specifications(i : float, t : int, f : int = 1) -> List[tuple]:
    """ 
    Generate Walker delta constellation specifications for a low, a medium, and a high connectivity ISL constellation 

    `i = n_sats_total/p/f`
    
    - `i` is the inclination
    - `t` is the total number of satellites;
    - `p` is the number of equally spaced planes; 
    - `f` is the relative spacing between satellites in adjacent planes. 
            The change in true anomaly (in degrees) for equivalent satellites 
            in neighbouring planes is equal to f * 360° / t.
    """
    specifications = [
        (i,t,p,f) 
        for p in range(1, t+1)
        if t % p == 0
    ]

    # return all specifications
    return specifications
    
def walker_to_orbital_params(alt : float, i : float, t : float, p : int, f : int) -> List[dict]:
    """
    Converts Walker delta constellation specifications to orbital parameters.

    - `i` is the inclination
    - `t` is the total number of satellites;
    - `p` is the number of equally spaced planes; 
    - `f` is the relative spacing between satellites in adjacent planes. 
            The change in true anomaly (in degrees) for equivalent satellites 
            in neighbouring planes is equal to f * 360° / t.
    """
    assert t % p == 0, "Number of satellites must be divisible by number of planes."

    # initialize list of orbital parameters
    orbital_params = []

    # calculate number of satellites per plane and RAAN spacing
    sats_per_plane = t // p
    raan_spacing = calc_raan_spacing(p)
    aop_phasing = calc_initial_phasing(t, f)

    # generate orbital parameters for every satellite in every plane
    for plane_idx in range(p):
        # calculate plane RAAN and initial aop
        raan = plane_idx * raan_spacing
        aop = plane_idx * aop_phasing 

        # generate orbital parameters for each satellite in plane
        for sat_idx in range(sats_per_plane):
            ta = sat_idx * calc_ta_spacing(t, p)

            params = {
                "@type": "KEPLERIAN_EARTH_CENTERED_INERTIAL",
                "sma": R + alt,
                "ecc": 0.001,
                "inc": i,
                "raan": raan,
                "aop": aop,
                "ta": ta
            }
            orbital_params.append(params)

    return orbital_params

def calc_initial_phasing(t: int, f : int) -> float:
    """ Calculate the initial phases for each satellite in degrees. """
    return f * 360 / t

def calc_ta_spacing(t: int, p : int) -> float:
    """ Calculate the in-plane spacing between satellites in degrees. """
    # ensure valid inputs
    assert t % p == 0, "Number of satellites must be divisible by number of planes."

    # calculate number of satellites per plane
    sats_per_plane = t // p

    # return spacing
    return 360.0 / sats_per_plane

def calc_raan_spacing(n_planes : int) -> float:
    """ Calculate the RAAN spacing between planes in degrees. """
    # return spacing
    return 360.0 / n_planes

def propagate_walker_delta_constellation(alt : float, i : float, t : int, p : int, f : int, T : float, debug : bool = True) -> str:
    """ Propagate a Walker delta constellation and save the orbit data to disk. """
    # define data directory
    data_dir = f'./orbits/walker_delta_{t}sat_{p}planes'
    data_filename = os.path.join(data_dir,'MissionSpecs.json')
    
    # convert to orbital parameters
    orbital_params = walker_to_orbital_params(alt, i, t, p, f)

    # DEBUG print results
    print("="*80 + "\n")
    print(f"Walker Delta Constellation - i:t/p/f = {i}°:{t}/{p}/{f}")
    if debug:
        print("-"*82)
        prev_param = None
        for idx, params in enumerate(orbital_params):
            if prev_param and prev_param['raan'] != params['raan']:
                print("  " + "-"*80)
            print(f"  Sat-{idx+1}:\tinc={params['inc']}°\traan={params['raan']}°\taop={params['aop']}°\tta={params['ta']}°")
            prev_param = params
        print("="*80 + "\n")
    else:
        print("\n")

    # create satellite specification
    spacecraft = []
    for idx, params in enumerate(orbital_params):
        sat_spec = deepcopy(SPACECRAFT_TEMPLATE)
        sat_spec['@id'] = f'sat_{idx+1}'
        sat_spec['name'] = f'sat_{idx+1}'
        sat_spec['orbitState']['state'] = params
        spacecraft.append(sat_spec)

    # create mission specification
    mission_spec = deepcopy(MISSION_TEMPLATE)
    mission_spec['duration'] = T / 3600.0 / 24 # convert duration from seconds to days
    mission_spec['spacecraft'] = spacecraft
    mission_spec['settings']['outDir'] = data_dir

    if os.path.exists(data_filename):
        # load existing mission specifications
        existing_mission_spec : dict = json.load(open(data_filename,'r'))
        existing_mission_spec['propagator'].pop('stepSize', None)

        # compare with current specifications
        if existing_mission_spec == mission_spec: 
            print(f"Propagation already exists for walker_delta_{t}sat_{p}planes. Skipping propagation...\n\n")
            return data_dir
        else:
            for key in mission_spec:
                if existing_mission_spec[key] != mission_spec[key]:
                    print(f"Difference found in key: `{key}`")
            print(f"Existing propagation specifications differ from current specifications for walker_delta_{t}sat_{p}planes. Re-propagating...\n\n")

    # create mission for propagation
    mission : Mission = Mission.from_json(mission_spec)  

    # propagate data and save to orbit data directory
    print("Propagating orbits...")
    mission.execute(coverage_propagation=False, data_metrics_calculation=False,eclipse_finder=False)                
    print("Propagation done!")

    # delete state propagation from printed data for space savings
    for dir_name in os.listdir(data_dir):
        x = 1 # breakpoint
        dir_path = os.path.join(data_dir,dir_name)
        if 'sat' in dir_name.lower() and os.path.isdir(dir_path):
            shutil.rmtree(dir_path)            

    # remove step size from propagator for comparison purposes
    mission_spec['propagator'].pop('stepSize', None)

    # save specifications of propagation in the orbit data directory
    with open(data_filename, 'w') as mission_specs_file:
        mission_specs_file.write(json.dumps(mission_spec, indent=4))

    return data_dir


# ========================================
# String of Pearls Constellation Functions
# ========================================
# def generate_string_of_pearls_specifications(n_sats : int, alt : float, inclination : float) -> Dict[dict]:
#     pass

# ========================================
#       Orbit Evaluation Functions
# ========================================

def load_access_event_intervals(data_dir : str) -> Tuple[list,list,float]:
    # initiale list of events for temporal graph
    events : List[tuple] = []

    # initiate set of nodes
    nodes = set()

    # define path to comms data directory
    comms_data_dir = os.path.join(data_dir, 'comm')
    
    # initiate time step variable
    time_step = np.NAN

    # load comms data for every inter-satellite link
    for filename in tqdm(os.listdir(comms_data_dir), desc=f'Loading inter-satellite link data'):
        # parse filename
        isl_names = filename.split('.')[0]
        sat1, _, sat2 = isl_names.split('_')

        # add nodes to list
        nodes.add(sat1)
        nodes.add(sat2)
        
        # define full path to comms data file
        comms_data_path = os.path.join(comms_data_dir, filename)

        # read propagation time-step
        time_data =  pd.read_csv(comms_data_path, nrows=2)
        _, _, _, _, time_step = time_data.at[1,time_data.axes[1][0]].split(' ')
        time_step = float(time_step)

        # load communications data
        df : pd.DataFrame = pd.read_csv(comms_data_path, skiprows=range(3))

        # skip if dataframe is empty
        if df.empty: continue

        # add edges to temporal graph
        for t_start,t_end in df.values:
            # convert to integer time-steps
            t_start = int(t_start)
            t_end = int(t_end)

            # add contact as an event to list 
            events.append( (sat1, sat2, t_start, t_end) )
    
    # sort events by start time
    events.sort(key=lambda e: e[2])
    
    # return events and approximate time-step
    return events, list(nodes), time_step

def load_graph_snapshot_series(data_dir : str, T : float) -> Tuple[List[nx.Graph],list,float]:
    # load access events and nodes
    events, nodes, time_step = load_access_event_intervals(data_dir)

    # extract event times
    event_times : list = sorted(set([e[2] for e in events]))

    # create linear time space
    t_start = max(min(event_times,default=0), 0)
    t_end = max(max(event_times,default=0), int(T / time_step))
    times = list(range(t_start, t_end+1))

    # initiate list of graph snapshots    
    snapshots: list[nx.Graph] = []

    # create snapshot for each time step
    for t in times:
        # get active events at time t
        active = [(u,b,t_start,t_end) for u,b,t_start,t_end in events 
                  if t_start <= t <= t_end]

        # create graph snapshot
        G = nx.Graph()
        
        # specify snapshot time
        G.graph["time"] = float(t)

        # add nodes
        G.add_nodes_from(nodes)

        # add edges
        G.add_edges_from(zip([e[0] for e in active], [e[1] for e in active]))

        # append to snapshots list
        snapshots.append(G)

    # return list of graph snapshots and corresponding times with time step
    return snapshots, times, time_step

def generate_time_series_metrics(i : float, t : int, p : int, f : int, 
                                 TG : List[nx.Graph], 
                                 times : List[int], 
                                 time_step : float,
                                 data_dir : str, 
                                 overwrite : bool = False
                                ) -> pd.DataFrame:
    """ Generate time-series connectivity metrics from temporal graph snapshots. """
    # define path to save metrics
    metrics_path = os.path.join(data_dir, 'connectivity_series.csv')

    # check if file already exists
    if os.path.exists(metrics_path) and not overwrite:
        # file exists and no overwrite is required, skip computation
        print(f"Connectivity metrics file already exists at: {metrics_path}. Loading existing metrics...\n")
        
        # load existing metrics
        metrics_series_df = pd.read_csv(metrics_path)
        
        # print metrics summary
        assert not metrics_series_df.empty, "Loaded connectivity metrics dataframe is empty!"
        print('Connectivity Metrics:')
        print(metrics_series_df.describe().iloc[1:].round(2))
        print(f"\nConnectivity metrics loaded from:\n   `{metrics_path}`\n")

    else:# file does not exist or require overwrite, compute metrics
        # initialize metrics
        n_components_series = []
        n_components_fraction_series = []
        largest_cc_size_series = []
        largest_cc_norm_series = []

        # Evaluate constellation connectivity metrics
        for G in tqdm(TG, desc=f'Evaluating connectivity metrics', unit='time steps'):
            # compute connectivity metrics
            components = list(nx.connected_components(G))
            n_components = nx.number_connected_components(G)
            n_components_fraction = n_components / G.number_of_nodes()
            largest_cc_size = len(max(components, key=len)) if n_components > 0 else 0
            largest_cc_norm = largest_cc_size / G.number_of_nodes() 

            # store metrics
            n_components_series.append(n_components)
            n_components_fraction_series.append(n_components_fraction)
            largest_cc_size_series.append(largest_cc_size)
            largest_cc_norm_series.append(largest_cc_norm)

        # compile to dataframe
        metrics_series_df = pd.DataFrame({
            'time index' : times,
            'time [s]' : [t * time_step for t in times],
            "inc [deg]" : [i] * len(times),
            "num sats" : [t] * len(times),
            "num planes" : [p] * len(times),
            "phasing param" : [f] * len(times),
            'num components' : n_components_series,
            'lcc' : largest_cc_size_series,
            'lcc [norm]' : largest_cc_norm_series
        })

        # save to csv
        metrics_series_df.to_csv(metrics_path, index=False)
        
        # print metrics summary
        print('Connectivity Metrics:')
        print(metrics_series_df.describe().iloc[1:].round(2))
        print(f"\nConnectivity metrics saved to: \n   `{metrics_path}`\n")

    return metrics_series_df

# ========================================
#              Main Execution
# ========================================
if __name__ == "__main__":
    
    # terminal welcome message
    print_welcome(f'Internal Validation Orbit Generator')

    # define common orbital parameters
    alt = 550.0     # altitude [km]
    i = 98.0        # inclination [degrees]
    # n_sats_candidates = [8, 12, 24, 36, 48, 60, 96, 144]  # total number of satellites
    # n_sats_candidates = [12, 36, 60, 144]  # total number of satellites
    # n_sats_candidates = [8, 24, 48, 96]  # total number of satellites
    n_sats_candidates = [144]  # total number of satellites
    # n_sats_candidates = [8, 12, 24]  # total number of satellites

    # calculate orbital period
    T = 2 * np.pi * np.sqrt( (R + alt)**3 / GM )
    T /= 2 # propagate for half an orbital period

    # initiate results compilation lists
    time_series_df : pd.DataFrame = None
    compiled_df : pd.DataFrame = None    

    # evaluate each candidate number of satellites
    for n_sats in n_sats_candidates:
        # generate walker delta specifications
        walker_specs = generate_walker_delta_specifications(i, n_sats)

        # initialize results compilation lists
        metrics_columns : List[str] = ['alt [km]', 'inc [deg]', 'num sats', 'num planes', 'phasing param',
                                        # 'fully connected time frac', 
                                        'max lcc [norm]', 
                                        'max lcc frac', 
                                        'avg lcc [norm]', 
                                        'avg num components'
                                        ]
        metrics_data : list = []    

        # evaluate each specification
        for i,t,p,f in walker_specs:
            # ensure total number of satellites matches specifications
            assert t == n_sats, "Total number of satellites does not match specified value."
    
            # propagate constellation and save to disk
            data_dir : str = propagate_walker_delta_constellation(alt, i, t, p, f, T)

            # Load temporal graph and event times
            TG, times, time_step = load_graph_snapshot_series(data_dir,T)

            # Generate time-series connectivity metrics
            metrics_series_df = generate_time_series_metrics(i, t, p, f, TG, times, time_step, data_dir, overwrite=True)
            
            # Add to results dataframe
            time_series_df = pd.concat([time_series_df, metrics_series_df], ignore_index=False) \
                             if time_series_df is not None else metrics_series_df
 
            # TODO Evaluate scalar metrics

            # Largest Connected Component 
            avg_largest_cc_norm = metrics_series_df['lcc [norm]'].mean()
                        
            max_largest_cc_norm = metrics_series_df['lcc [norm]'].max()
            largest_cc_norm_series = metrics_series_df[metrics_series_df["lcc [norm]"] == max_largest_cc_norm]
            max_largest_cc_norm_fraction = len(largest_cc_norm_series) / len(metrics_series_df)

            # Connected Component Count
            avg_n_components = metrics_series_df['num components'].mean()

            # Percentage Time Fully Connected
            n_fully_connected = len(metrics_series_df[metrics_series_df['num components'] == 1])
            fully_connected_fraction = n_fully_connected / len(metrics_series_df)

            # compile results
            scalar_metrics = [
                        # fully_connected_fraction, 
                        max_largest_cc_norm, 
                        max_largest_cc_norm_fraction, 
                        avg_largest_cc_norm, 
                        avg_n_components
                    ]

            metrics = [alt,i,t,p,f]
            metrics.extend(scalar_metrics)
            
            # Add to results list
            metrics_data.append(metrics)

        # sort metrics data 
        # metrics_data.sort(key=lambda x: (-x[5],-x[6],-x[7],-x[8],x[9])) # sort by metrics

        # generate metrics dataframe for this number of satellites
        metrics_df = pd.DataFrame(data=metrics_data, columns=metrics_columns) 

        # compile results dataframe
        compiled_df = pd.concat([compiled_df, metrics_df], ignore_index=True) \
                        if compiled_df is not None else metrics_df
    
    # compile final results metrics
    print("Compiled Scalar Metrics:")
    print(compiled_df)

    # Save results metrics to csv
    compiled_metrics_path = os.path.join('./orbits/compiled_walker_delta_connectivity_metrics.csv')
    compiled_df.to_csv(compiled_metrics_path, index=False)
    print(f"Compiled connectivity scalar metrics saved to: \n   `{compiled_metrics_path}`\n")

    # plot 
    print("Plotting compiled time-series metrics...")
    sns.set_theme(style="whitegrid")

    # Largest Connected Component over time
    ax1 = sns.relplot(data=time_series_df, x='time [s]', y='lcc [norm]', row="num sats", hue="num planes", kind="line", palette="Set2")
    ax1.set_titles("Largest Connected Component Over Time (n_sats={row_name})")

    # Number of Components over time
    ax2 = sns.relplot(data=time_series_df, x='time [s]', y='num components', row="num sats", hue="num planes", kind="line", palette="Set2")
    ax2.set_titles("Number of Components Over Time (n_sats={row_name})")

    # Scatter Plots
    ax3 = sns.relplot(data=compiled_df, x='max lcc [norm]', y='max lcc frac', row='num sats', hue='num planes', size='avg lcc [norm]', kind='scatter', palette="Set2")
    ax3.set_titles("Max LCC vs. Avg LCC (n_sats={row_name})")

    plt.show()

    x = 1