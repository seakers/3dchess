"""
Compiles results and creates plots
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__  == "__main__":
    # set params
    results_path = './results'
    show_plots = False
    save_plots = True
    
    # get run names
    run_names = list({run_name for run_name in os.listdir(results_path)
                 if os.path.isfile(os.path.join(results_path, run_name, 'summary.csv'))})

    # get run parameters
    parameters_df = pd.read_csv('./resources/lhs_scenarios.csv')
    parameters = { name : (field_of_regard,fov,agility,event_duration,num_events,distribution,num_planes,sats_per_plane)
        for name,field_of_regard,fov,_,agility,event_duration,num_events,_,distribution,num_planes,sats_per_plane,*_ in parameters_df.values
    }

    # organize data
    for run_name in run_names:
        # load results summary
        summary_path = os.path.join(results_path, run_name, 'summary.csv')
        summary : pd.DataFrame = pd.read_csv(summary_path)

        # get run parameters
        *_,experiment_id,preplanner,replanner = run_name.split('_')
        experiment_id = int(experiment_id)
        preplanner,period,horizon = preplanner.split('-')
        if 'acbba-dp' in replanner:
            replanner,_,bundle_size = replanner.split('-')
        elif 'acbba' in replanner:
            replanner,bundle_size = replanner.split('-')

        field_of_regard,fov,agility,event_duration, \
            num_events,distribution,num_planes,sats_per_plane \
                = parameters[f'updated_experiment_{experiment_id}']
        x = 1
    x = 1

    # # initialize metrics
    # ## Bar plots
    # events = {
    #     'Events Detected': [],
    #     'Total Event Observations' : [],
    #     'Unique Event Observations' : []
    # }

    # coobservations = {
    #     "Events Co-observed" : [],
    #     "Events Partially Co-observed" : [],
    #     "Events Fully Co-observed" : [],
    #     "Total Event Co-observations" : []
    # }

    # # Scatter plots
    # scatter_metrics = { planner : {
    #                                 "N sats" : [],
    #                                 "N planes" : [],
    #                                 "N sats per plane" : [],
    #                                 "Events Detected" : [],
    #                                 "Total Event Observations" : [],
    #                                 "Unique Event Observations" : [],
    #                                 "Events Co-observed" : [],
    #                                 "Events Partially Co-observed" : [],
    #                                 "Events Fully Co-observed" : [],
    #                                 "Total Event Co-observations" : []
    #                                 } 
    #                     for planner in ['naive_fifo', 'acbba-1', 'acbba-3']
    #                 }

    # # collect metrics
    # n_events = None
    # ignored_runs = []
    # for run_name in run_names:
    #     # load run specs
    #     *_,n_planes,n_sats_per_plane,field_of_view,field_of_regard,max_slew_rate,preplanner,replanner = run_name.split('_')
    #     n_planes = int(n_planes)
    #     n_sats_per_plane = int(n_sats_per_plane)
    #     field_of_view = float(field_of_view)
    #     field_of_regard = float(field_of_regard)

        # # load results summary
        # summary_path = os.path.join(results_path, run_name, 'summary.csv')
        # summary : pd.DataFrame = pd.read_csv(summary_path)

    #     # # load list of observations
    #     # observations_path = os.path.join(results_path, run_name, 'environment', 'measurements.csv')
    #     # observations : pd.DataFrame = pd.read_csv(observations_path)

    #     # collect data for plots
    #     n_events = [int(val) for key,val in summary.values if key == 'Events'][0] if n_events is None else n_events

    #     events['Events Detected'].append([int(val) for key,val in summary.values if key == 'Events Detected'][0])
    #     events['Unique Event Observations'].append([int(val) for key,val in summary.values if key == 'Unique Event Observations'][0])
    #     events['Total Event Observations'].append([int(val) for key,val in summary.values if key == 'Total Event Observations'][0])

    #     coobservations['Events Co-observed'].append([int(val) for key,val in summary.values if key == 'Events Co-observed'][0])
    #     coobservations['Events Partially Co-observed'].append([int(val) for key,val in summary.values if key == 'Events Partially Co-observed'][0])
    #     coobservations['Events Fully Co-observed'].append([int(val) for key,val in summary.values if key == 'Events Fully Co-observed'][0])
    #     coobservations['Total Event Co-observations'].append([int(val) for key,val in summary.values if key == 'Total Event Co-observations'][0])

    #     planner = replanner if 'acbba' in replanner else 'naive_fifo'
    #     scatter_metrics[planner]['N sats'].append(n_planes*n_sats_per_plane)
    #     scatter_metrics[planner]['N planes'].append(n_planes)
    #     scatter_metrics[planner]['N sats per plane'].append(n_sats_per_plane)

    #     scatter_metrics[planner]['Events Detected'].append([int(val) for key,val in summary.values if key == 'Events Detected'][0])
    #     scatter_metrics[planner]['Unique Event Observations'].append([int(val) for key,val in summary.values if key == 'Unique Event Observations'][0])
    #     scatter_metrics[planner]['Total Event Observations'].append([int(val) for key,val in summary.values if key == 'Total Event Observations'][0])

    #     scatter_metrics[planner]['Events Co-observed'].append([int(val) for key,val in summary.values if key == 'Events Co-observed'][0])
    #     scatter_metrics[planner]['Events Partially Co-observed'].append([int(val) for key,val in summary.values if key == 'Events Partially Co-observed'][0])
    #     scatter_metrics[planner]['Events Fully Co-observed'].append([int(val) for key,val in summary.values if key == 'Events Fully Co-observed'][0])
    #     scatter_metrics[planner]['Total Event Co-observations'].append([int(val) for key,val in summary.values if key == 'Total Event Co-observations'][0])

    # # ---- BAR PLOTS ----
    # # set x-axis
    # x = np.arange(len(run_names))  # the label locations
    # width = 0.25  # the width of the bars
    # multiplier = 0

    # # event observation plot
    # fig, ax = plt.subplots(layout='constrained')

    # for attribute, measurement in events.items():
    #     offset = width * multiplier
    #     rects = ax.bar(x + offset, measurement, width, label=attribute)
    #     ax.bar_label(rects, padding=3)
    #     multiplier += 1

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('n')
    # ax.set_title(f'Events ({n_events} total)')
    # ax.set_xticks(x + width, run_names)
    # ax.legend(loc='upper left')

    # # show/save plots
    # # if show_plots: plt.show()
    # # if save_plots: plt.savefig('./plots/events.png')

    # # event observation plot
    # fig, ax = plt.subplots(layout='constrained')

    # for attribute, measurement in coobservations.items():
    #     offset = width * multiplier
    #     rects = ax.bar(x + offset, measurement, width, label=attribute)
    #     ax.bar_label(rects, padding=3)
    #     multiplier += 1

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('n')
    # ax.set_title(f'Event Co-observations ({n_events} total)')
    # ax.set_xticks(x + width, run_names)
    # ax.legend(loc='upper left')

    # # show/save plots
    # # if show_plots: plt.show()
    # # if save_plots: plt.savefig('./plots/observations.png')

    # # ---- SCATTER PLOTS ----
    # titles = [
    #            "Events Detected",
    #            "Total Event Observations",
    #            "Unique Event Observations",
    #            "Events Co-observed",
    #            "Events Partially Co-observed",
    #            "Events Fully Co-observed",
    #            "Total Event Co-observations"
    #           ]
    # counter = 1
    # for title in titles:
    #     # create plot
    #     fig, ax = plt.subplots(layout='constrained')
    #     for planner in scatter_metrics:
    #         x = scatter_metrics[planner]['N sats']
    #         y = scatter_metrics[planner][title]
    #         ax.scatter(x, y, label=planner, edgecolors='none')

    #     # set axis and title
    #     ax.set_title(f'{title} per N sats')
    #     ax.set_xlabel('N sats')
    #     ax.set_ylabel(title)
    #     ax.legend()
    #     ax.grid(True)

    #     # show/save plots
    #     if show_plots: plt.show()
    #     if save_plots: plt.savefig(f'./plots/scatter{counter}.png')
        
    #     # increase counter
    #     counter += 1 
