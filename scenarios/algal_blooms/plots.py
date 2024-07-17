"""
Compiles results and creates plots
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__  == "__main__":
    results_path = './results'
    
    run_names = [run_name for run_name in os.listdir(results_path)]
    run_names.sort(reverse=True)

    events = {
        'Events Detected': [],
        'Total Event Observations' : [],
        'Unique Event Observations' : []
    }

    coobservations = {
        "Co-observations" : [],
        "Partial Co-observations" : [],
        "Full Co-observations" : []
    }

    n_events = None
    for run_name in run_names:
        # load results summary
        summary_path = os.path.join(results_path, run_name, 'summary.csv')
        summary : pd.DataFrame = pd.read_csv(summary_path)

        # load list of observations
        observations_path = os.path.join(results_path, run_name, 'environment', 'measurements.csv')
        observations : pd.DataFrame = pd.read_csv(observations_path)

        # add data to plot inputs
        n_events = [int(val) for key,val in summary.values if key == 'n_events'][0] if n_events is None else n_events

        events['Events Detected'].append([int(val) for key,val in summary.values if key == 'n_events_detected'][0])
        events['Total Event Observations'].append([int(val) for key,val in summary.values if key == 'n_total_event_obs'][0])
        events['Unique Event Observations'].append([int(val) for key,val in summary.values if key == 'n_unique_event_obs'][0])

        coobservations['Co-observations'].append([int(val) for key,val in summary.values if key == 'n_co_obs'][0])
        coobservations['Partial Co-observations'].append([int(val) for key,val in summary.values if key == 'n_event_partially_co_obs'][0])
        coobservations['Full Co-observations'].append([int(val) for key,val in summary.values if key == 'n_events_fully_co_obs'][0])
    
    # set x-axis
    x = np.arange(len(run_names))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    # event observation plot
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in events.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('n')
    ax.set_title(f'Events ({n_events} total)')
    ax.set_xticks(x + width, run_names)
    ax.legend(loc='upper left')

    plt.show()

    # event observation plot
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in coobservations.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('n')
    ax.set_title(f'Event Co-observations ({n_events} total)')
    ax.set_xticks(x + width, run_names)
    ax.legend(loc='upper left')

    plt.show()
