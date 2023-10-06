import csv
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # clear terminal
    os.system('cls' if os.name == 'nt' else 'clear')

    # read results
    print('loading results...')
    results = {}
    for dir in os.listdir():
        if os.path.isdir(f'./{dir}'):
            with open(f'./{dir}/results/summary.csv', newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                
                skip_first = True
                result = {}
                for prop, val in spamreader:
                    if skip_first:
                        skip_first = False
                        continue
                                        
                    result[prop] = val
            
                results[dir] = result
    
    # create case-to-directory map
    cases = ['No Replanning', 'Periodic\nPlanning [1hr]', 'Event-driven\nReplanning', 'Periodic Planning [1hr] \nw/Event-driven Replanning']
    case_dirs = list(filter(lambda case : os.path.isdir(f'./{case}'), os.listdir()))

    cases_map = {}
    for case_dir in case_dirs:
        if "no_replan" in case_dir:
            if "periodic" in case_dir:
                cases_map[cases[1]] = case_dir
            else:
                cases_map[cases[0]] = case_dir
        else:
            if "periodic" in case_dir:
                cases_map[cases[3]] = case_dir
            else:
                cases_map[cases[2]] = case_dir

    # plot events detected/observed 
    print('plotting events detected/observed...')
    events_detected = [int(results[cases_map[case_name]]['n_events_detected']) 
                        for case_name in cases]
    events_observed = [int(results[cases_map[case_name]]['n_events_obs']) 
                        for case_name in cases]
    n_events_max = max([int(results[case]['n_events']) for case in results])
        
    n_events = {
                    "Events\nDetected" : tuple(events_detected),
                    "Events\nObserved" : tuple(events_observed)
                }
    x = np.arange(len(cases))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in n_events.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1     
    
    ax.set_ylabel('#')
    ax.set_title(f"Algae-Bloom Event Detection (max {n_events_max} events)")
    ax.set_xticks(x + width, cases)
    ax.legend(loc='upper right', ncols=2)
    ax.set_ylim(0, max(events_detected) + 2)

    print('plot done.')
    # plt.savefig('./events.png')
    # plt.show()
    
    # plot observations possible/made
    print('plotting observations...')
    obs_unique_max = [int(results[cases_map[case_name]]['n_obs_unique_max']) 
                        for case_name in cases]
    obs_unique_pos = [int(results[cases_map[case_name]]['n_obs_unique_pos']) 
                        for case_name in cases]
    obs_unique = [int(results[cases_map[case_name]]['n_obs_unique']) 
                        for case_name in cases]
    obs_co = [int(results[cases_map[case_name]]['n_obs_co']) 
                        for case_name in cases]
    obs = [int(results[cases_map[case_name]]['n_obs']) 
                        for case_name in cases]

    n_obs = {
                    "Observations\nMade" : tuple(obs),
                    "Max Unique\nObservations" : tuple(obs_unique_max),
                    # "Possible Unique\nObservations" : tuple(obs_unique_pos),
                    "Unique\nObservations Made" : tuple(obs_unique),
                    "Co-observations\nMade" : tuple(obs_co)
                }

    x = np.arange(len(cases))  # the label locations
    width = 0.10  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in n_obs.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1     
    
    ax.set_ylabel('#')
    ax.set_title(f"Ground-Point Observations")
    ax.set_xticks(x + width, cases)
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0, max(obs) + 250)


    print('plot done.')
    fig.set_figwidth(8)
    plt.show()
    plt.savefig('./observations.png')

    print('DONE')
    x = 1
