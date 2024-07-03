import copy
import json
import os

from chess3d.mission import Mission
from chess3d.utils import print_welcome


if __name__ == "__main__":
    # set scenario name
    scenario_name = 'algal_bloom'

    # terminal welcome message
    print_welcome(scenario_name)
    
    # load base scenario json file
    scenario_file = os.path.join('./scenarios', scenario_name, 'MissionSpecs.json')
    with open(scenario_file, 'r') as scenario_file:
        template_specs : dict = json.load(scenario_file)

    # set parameters
    duration = 1
    n_planes = 3
    n_sats_per_plane = 3
    raans = [360 * j / n_planes for j in range(n_planes)]
    tas = [360 * i / n_sats_per_plane for i in range(n_sats_per_plane)]
    
    max_slew_rate = 1
    max_torque = 1

    instruments = ['visual', 'thermal', 'sar']
    abbreviations = {'visual' : 'vis', 
                   'thermal' : "therm", 
                   'sar' : 'sar'}
    field_of_view = 5
    field_of_regard = 45

    preplanners = ['naive']
    replanners = ['broadcaster', 'acbba']
    bundle_sizes = [1, 2, 3, 5]

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
                    for i in range(n_sats_per_plane):
                        sat : dict = copy.deepcopy(scenario_specs['spacecraft'][-1])

                        # set slew rate
                        sat['spacecraftBus']['components']['adcs']['maxRate'] = max_slew_rate
                        sat['spacecraftBus']['components']['adcs']['maxTorque'] = max_torque

                        # set instrument properties
                        sat['instrument']['name'] = instruments[j]
                        sat['instrument']['fieldOfViewGeometry'] ['angleHeight'] = field_of_view
                        sat['instrument']['fieldOfViewGeometry'] ['angleWidth'] = field_of_view
                        sat['instrument']['maneuver']['A_rollMin'] = - field_of_regard / 2.0
                        sat['instrument']['maneuver']['A_rollMax'] = field_of_regard / 2.0
                        sat['instrument']['@id'] = f'{abbreviations[instruments[j]]}1'

                        # set orbit
                        sat['orbitState']['state']['raan'] = raans[j]
                        sat['orbitState']['state']['ta'] = tas[j]
                        
                        # add to list of sats
                        sats.append(sat)
                
                # update list of satellites
                scenario_specs['spacecraft'] = sats

                # initialize mission
                mission : Mission = Mission.from_dict(scenario_specs)

                # execute mission
                # mission.execute()

    print('DONE')