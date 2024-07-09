import json
import os

from chess3d.mission import Mission
from chess3d.utils import print_welcome


if __name__ == "__main__":
    # terminal welcome message
    print_welcome()
    
    # load scenario json file
    scenario_file = os.path.join('./examples', 'toy_acbba', 'MissionSpecs.json')
    with open(scenario_file, 'r') as scenario_file:
        scenario_specs : dict = json.load(scenario_file)

    # initialize mission
    mission : Mission = Mission.from_dict(scenario_specs)

    # execute mission
    mission.execute()

    print('DONE')