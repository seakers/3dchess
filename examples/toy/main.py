import json
import os

from chess3d.mission import Mission
from chess3d.utils import print_welcome, arg_parser


if __name__ == "__main__":
    
    # read system arguments
    scenario_name, plot_results, save_plot, welcome_graphic, level = arg_parser()

    # terminal welcome message
    if welcome_graphic: print_welcome(scenario_name)
    
    # load scenario json file
    scenario_file = os.path.join('./examples', scenario_name, 'MissionSpecs.json')
    with open(scenario_file, 'r') as scenario_file:
        scenario_specs : dict = json.load(scenario_file)

    # initialize mission
    mission : Mission = Mission.from_dict(scenario_specs)

    # execute mission
    mission.execute(plot_results, save_plot)

    print('DONE')
    x = 1