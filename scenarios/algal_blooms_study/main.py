import copy
import json
import os

import numpy as np
import pandas as pd
import tqdm

from chess3d.simulation import Simulation
from chess3d.utils import print_welcome


if __name__ == "__main__":
    # set scenario name
    parent_scenario_name = 'algal_blooms_study'
    
    # load base scenario json file
    template_file = os.path.join('./scenarios', parent_scenario_name,'MissionSpecs.json')
    with open(template_file, 'r') as template_file:
        template_specs : dict = json.load(template_file)

    # read scenarios file
    scenarios_file = os.path.join('./scenarios', parent_scenario_name, 'resources', 'lhs_scenarios.csv')
    scenarios_df : pd.DataFrame = pd.read_csv(scenarios_file)

    # set event parameters
    event_type = 'random'
    number_of_events = [500, 1000, 2500]
    event_durations = [3600 * i for i in range(1,5)]
    min_severity = 0.0
    max_severity = 100
    measurement_list = ['sar', 'visual', 'thermal']
    # same_events = True
    
    # get agent parameters from previous study
    params = {c: scenarios_df[c].unique() for c in scenarios_df}
    fovs = [float(val) for val in params['fov (deg)']]
    fors = [float(val) for val in params['for (deg)']]
    agility = [float(val) for val in params['agility (deg/s)']]
    num_planes = [int(val) for val in params['num_planes']]; num_planes.sort()
    num_sats_per_plane = [int(val) for val in params['num_planes']]; num_sats_per_plane.sort()

    # set agent parameters manually
    fovs = [5]
    fors = [max(fors)]
    agility = [1.0]
    n_sats_min = 6
    ta_spacing = 'even'

    duration = 24.0 / 24.0
    max_torque = 0.0
        
    instruments = [
                    'visual', 
                    'thermal', 
                    'sar'
                   ]
    abbreviations = {
                    'visual' : 'vis', 
                    'thermal' : 'therm', 
                    'sar' : 'sar'
                   }
    preplanners = [
                    'naive',
                    'dynamic'
                    # 'nadir'                    
                    ]
    periods = [
        100,
        500,
        np.Inf
    ]
    horizons = [
        100,
        500,
        np.Inf
    ]
    replanners = [
                  'broadcaster', 
                  'acbba',
                  ]
    bundle_sizes = [
                    1,
                    # 2, 
                    3 
                    # 5
                    ]   
    
    # set results parameters
    same_events = True
    overwrite = False

    # count number of runs to be made
    n_runs : int = int(len(number_of_events) * len(event_durations)
                       * len(num_planes) * len(num_sats_per_plane) 
                       * len(fovs) * len(fors) * len(agility)
                       * len(preplanners) * len(periods) * len(horizons)
                       * len(replanners) * len(bundle_sizes)
                       
                    #    - n_sats_min
                    #    * len(fovs) * len(fors) * len(agility)
                    #    * len(preplanners) 
                    #    * ((len(replanners) - 1) * len(bundle_sizes) + 1)
                       )
    
    print(F'NUMBER OF RUNS TO PERFORM: {n_runs}')
    
    # run simulations
    with tqdm.tqdm(total=n_runs) as pbar:
        for n_events in number_of_events:

            for event_duration in event_durations:
                pregenerated_events = False

                for n_planes in num_planes:

                    # calculate list of RAANs
                    raans = [360 * j / n_planes for j in range(n_planes)]

                    for n_sats_per_plane in num_sats_per_plane:

                        # calculate list of true anomalies for each sat 
                        ta_spacing = 360 / n_sats_per_plane if ta_spacing == 'even' else ta_spacing
                        tas = [ta_spacing * i for i in range(n_sats_per_plane)]

                        for field_of_view in fovs:

                            for field_of_regard in fors:

                                for max_slew_rate in agility:
                                    max_slew_rate = float(max_slew_rate)

                                    for preplanner in preplanners:
                                            
                                        for period in periods:

                                            for horizon in horizons:

                                                for replanner in replanners:

                                                    for bundle_size in bundle_sizes:

                                                        # check simulation requirements
                                                        if n_planes * n_sats_per_plane < n_sats_min:
                                                            # skip
                                                            pbar.update(1)
                                                            continue

                                                        if replanner == 'broadcaster' and bundle_size > 1:
                                                            # skip
                                                            pbar.update(1)
                                                            break

                                                        if preplanner == 'np' and horizon == np.Inf:
                                                            # skip
                                                            pbar.update(1)
                                                            continue

                                                        if preplanner == 'naive' and (horizon < np.Inf or period < np.Inf):
                                                            # skip
                                                            pbar.update(1)
                                                            continue

                                                        scenario_specs : dict = copy.deepcopy(template_specs)

                                                        # set simulation duration
                                                        scenario_specs['duration'] = duration

                                                        # set events
                                                        if event_type == 'random':
                                                            if same_events:
                                                                if not pregenerated_events:
                                                                    events = {
                                                                                "@type": "random", 
                                                                                "numberOfEvents" : n_events,
                                                                                "duration" : event_duration,
                                                                                "minSeverity" : min_severity,
                                                                                "maxSeverity" : max_severity,
                                                                                "measurements" : measurement_list
                                                                            }
                                                                    pregenerated_events = True
                                                                else:
                                                                    events = {
                                                                                "@type": "PREDEF", 
                                                                                "eventsPath" : "./scenarios/algal_blooms_study/resources/random_events.csv"
                                                                            }
                                                            else:
                                                                events = {
                                                                            "@type": "random", 
                                                                            "numberOfEvents" : n_events,
                                                                            "duration" : event_duration,
                                                                            "minSeverity" : min_severity,
                                                                            "maxSeverity" : max_severity,
                                                                            "measurements" : measurement_list
                                                                        }
                                                            scenario_specs['scenario']['events'] = events

                                                        # set spacecraft specs
                                                        sats = []
                                                        for j in range(n_planes):
                                                            instr_counter = {instrument : 0 for instrument in instruments}
                                                            for i in range(n_sats_per_plane):
                                                                sat : dict = copy.deepcopy(scenario_specs['spacecraft'][-1])

                                                                # choose agent instrument
                                                                i_instrument = np.mod(i, len(instruments))
                                                                instrument = instruments[i_instrument] 

                                                                # set agent name and id
                                                                sat['@id'] = f'{abbreviations[instrument]}_sat_{j}_{instr_counter[instrument]}'
                                                                sat['name'] = f'{abbreviations[instrument]}_sat_{j}_{instr_counter[instrument]}'

                                                                # set slew rate
                                                                sat['spacecraftBus']['components']['adcs']['maxRate'] = max_slew_rate
                                                                sat['spacecraftBus']['components']['adcs']['maxTorque'] = max_torque

                                                                # set instrument properties
                                                                sat['instrument']['name'] = instrument
                                                                sat['instrument']['fieldOfViewGeometry'] ['angleHeight'] = field_of_view
                                                                sat['instrument']['fieldOfViewGeometry'] ['angleWidth'] = field_of_view
                                                                sat['instrument']['maneuver']['A_rollMin'] = - field_of_regard / 2.0
                                                                sat['instrument']['maneuver']['A_rollMax'] = field_of_regard / 2.0
                                                                sat['instrument']['@id'] = f'{abbreviations[instrument]}1'

                                                                # set orbit
                                                                sat['orbitState']['state']['raan'] = raans[j]
                                                                sat['orbitState']['state']['ta'] = tas[i]

                                                                # set preplanner
                                                                sat['planner']['preplanner']['@type'] = preplanner
                                                                sat['planner']['preplanner']['period'] = period
                                                                sat['planner']['preplanner']['horizon'] = horizon

                                                                # set replanner
                                                                sat['planner']['replanner']['@type'] = replanner
                                                                sat['planner']['replanner']['bundle size'] = bundle_size
                                                                
                                                                # add to list of sats
                                                                sats.append(sat)

                                                                # update counter 
                                                                instr_counter[instrument] += 1
                                                        
                                                        # update list of satellites
                                                        scenario_specs['spacecraft'] = sats

                                                        # create scenario name
                                                        scenario_name = f'{parent_scenario_name}_{n_events}_{event_duration}_{n_planes}_{n_sats_per_plane}_{int(field_of_view)}_{int(field_of_regard)}_{int(max_slew_rate)}_{preplanner}-{period}-{horizon}_{replanner}'
                                                        if replanner != 'broadcaster': scenario_name += f'-{bundle_size}'
                                                        
                                                        # set scenario name
                                                        scenario_specs['scenario']['name'] = scenario_name

                                                        # check overwrite toggle
                                                        results_dir = os.path.join('./scenarios',parent_scenario_name, 'results', scenario_name)
                                                        if not overwrite and os.path.exists(os.path.join(results_dir, 'summary.csv')):
                                                            # scenario already ran; skip to avoid overwrite
                                                            pbar.update(1)
                                                            continue

                                                        # initialize mission
                                                        mission : Simulation = Simulation.from_dict(scenario_specs)

                                                        # print welcome message
                                                        print_welcome(scenario_name)

                                                        # execute mission
                                                        mission.execute()

                                                        # # jump mission specifications into json file
                                                        # with open(os.path.join(results_dir,'MissionSpecs.json'), 'w') as mission_specs:
                                                        #     json.dump(scenario_specs, mission_specs, indent=4)
                                                        #     # mission_specs.write()
                                                            
                                                        # update progress bad
                                                        pbar.update(1)

    print('DONE')