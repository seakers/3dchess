import copy
import json
import os

from numpy import mod

from chess3d.mission import Mission
from chess3d.utils import print_welcome


if __name__ == "__main__":
    # set scenario name
    scenario_name = 'algal_bloom'
    
    # load base scenario json file
    scenario_file = os.path.join('./scenarios', scenario_name, 'MissionSpecs.json')
    with open(scenario_file, 'r') as scenario_file:
        template_specs : dict = json.load(scenario_file)

    # set parameters
    duration = 1
    n_planes = 1
    n_sats_per_plane = 6
    raans = [360 * j / n_planes for j in range(n_planes)]
    tas = [360 * i / n_sats_per_plane for i in range(n_sats_per_plane)]
    
    max_slew_rate = 1
    max_torque = 1

    instruments = [
                    'visual', 
                    'thermal', 
                    'sar'
                   ]
    abbreviations = {'visual' : 'vis', 
                   'thermal' : "therm", 
                   'sar' : 'sar'}
    field_of_view = 5
    field_of_regard = 45

    preplanners = ['nadir']
    replanners = [
                #   'broadcaster', 
                  'acbba'
                  ]
    bundle_sizes = [
                    # 1
                    # 2, 
                    3
                    # 5
                    ]
    utility = 'fixed'

    n_runs = len(preplanners) * ((len(replanners) - 1) * len(bundle_sizes) + 1)

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
                        sat['name'] = f'{abbreviations[instrument]}_{instr_counter[instrument]}'

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
                    scenario_specs['scenario']['name'] = f'{preplanner}_{replanner}'
                else:
                    scenario_specs['scenario']['name'] = f'{preplanner}_{replanner}_{bundle_size}'

                # initialize mission
                mission : Mission = Mission.from_dict(scenario_specs)

                # print welcome message
                print_welcome(scenario_specs['scenario']['name'])

                # execute mission
                mission.execute()

    print('DONE')