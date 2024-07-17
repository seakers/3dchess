import copy
import json
import os

from numpy import mod
import pandas as pd
import tqdm

from chess3d.mission import Mission
from chess3d.utils import print_welcome


if __name__ == "__main__":
    # set scenario name
    scenario_name = 'algal_blooms_study'
    
    # load base scenario json file
    template_file = os.path.join('./scenarios', scenario_name, 'MissionSpecs.json')
    with open(template_file, 'r') as template_file:
        template_specs : dict = json.load(template_file)

    # read scenarios file
    scenarios_file = os.path.join('./scenarios', scenario_name, 'lhs_scenarios.csv')
    scenarios_df : pd.DataFrame = pd.read_csv(scenarios_file)

    # extract parameters from previous
    params = {c: scenarios_df[c].unique() for c in scenarios_df}
    fovs = list(params['fov (deg)'])
    fors = list(params['for (deg)'])
    agility = list(params['agility (deg/s)'])
    num_planes = list(params['num_planes'])
    num_sats_per_plane = list(params['num_sats_per_plane'])

    # set parameters
    duration = 1
    max_torque = 0.0
        
    instruments = [
                    'visual', 
                    'thermal', 
                    'sar'
                   ]
    abbreviations = {'visual' : 'vis', 
                   'thermal' : "therm", 
                   'sar' : 'sar'}
    preplanners = ['naive']
    replanners = [
                #   'broadcaster', 
                  'acbba'
                  ]
    bundle_sizes = [
                    1,
                    # 2, 
                    3 
                    # 5
                    ]
    utility = 'fixed'

    # count number of runs to be made
    n_runs : int = int(len(num_planes) * len(num_sats_per_plane) 
                       * len(fovs) * len(fors) * len(agility)
                       * len(preplanners) 
                       * ((len(replanners) - 1) * len(bundle_sizes) + 1))
    print(F'NUMBER OF RUNS TO PERFORM: {n_runs}')
    
    # run simulations
    with tqdm.tqdm(total=n_runs) as pbar:
        for n_planes in num_planes:
            n_planes = int(n_planes)

            # calculate list of RAANs
            raans = [360 * j / n_planes for j in range(n_planes)]

            for n_sats_per_plane in num_sats_per_plane:
                n_sats_per_plane = int(n_sats_per_plane)

                # calculate list of true anomalies for each sat 
                tas = [360 * i / n_sats_per_plane for i in range(n_sats_per_plane)]

                if n_planes * n_sats_per_plane < 10:
                    pbar.update(1)
                    continue

                for field_of_view in fovs:
                    field_of_view = float(field_of_view)

                    for field_of_regard in fors:
                        field_of_regard = float(field_of_regard) 

                        for max_slew_rate in agility:
                            max_slew_rate = float(max_slew_rate)

                            for preplanner in preplanners:

                                for replanner in replanners:

                                    for bundle_size in bundle_sizes:

                                        if replanner == 'broadcaster' and bundle_size > 1:
                                            break

                                        scenario_specs : dict = copy.deepcopy(template_specs)

                                        # set simulation duration
                                        scenario_specs['duration'] = duration

                                        # set spacecraft specs
                                        sats = []
                                        for j in range(n_planes):
                                            instr_counter = {instrument : 0 for instrument in instruments}
                                            for i in range(n_sats_per_plane):
                                                sat : dict = copy.deepcopy(scenario_specs['spacecraft'][-1])

                                                # choose agent instrument
                                                i_instrument = mod(i, len(instruments))
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
                                                sat['orbitState']['state']['ta'] = tas[j]

                                                # set preplanner
                                                sat['planner']['preplanner']['@type'] = preplanner

                                                # set replanner
                                                sat['planner']['replanner']['@type'] = replanner
                                                sat['planner']['replanner']['utility'] = utility
                                                sat['planner']['replanner']['bundle size'] = bundle_size
                                                
                                                # add to list of sats
                                                sats.append(sat)

                                                # update counter 
                                                instr_counter[instrument] += 1
                                        
                                        # update list of satellites
                                        scenario_specs['spacecraft'] = sats

                                        # set scenario name
                                        if replanner == 'broadcaster':
                                            scenario_specs['scenario']['name'] = f'{scenario_name}_{n_planes}_{field_of_view}_{field_of_regard}_{int(max_slew_rate)}_{preplanner}_{replanner}'
                                        else:
                                            scenario_specs['scenario']['name'] = f'{scenario_name}_{n_planes}_{field_of_view}_{field_of_regard}_{int(max_slew_rate)}_{preplanner}_{replanner}_{bundle_size}'

                                        # initialize mission
                                        mission : Mission = Mission.from_dict(scenario_specs)

                                        # print welcome message
                                        print_welcome(scenario_specs['scenario']['name'])

                                        # execute mission
                                        mission.execute()

                                        # update progress bad
                                        pbar.update(1)

    print('DONE')