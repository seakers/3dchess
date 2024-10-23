"""
Compiles results and creates plots
"""

import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__  == "__main__":
    # set params
    results_path = './results'
    show_plots = True
    save_plots = True
    
    # get run names
    run_names = list({run_name for run_name in os.listdir(results_path)
                 if os.path.isfile(os.path.join(results_path, run_name, 'summary.csv'))})

    # get run parameters
    parameters_df = pd.read_csv('./resources/experiments/experiments_seed-1000.csv')
    parameters = { name : params for name,*params in parameters_df.values } # Name,Number Planes,Number of Satellites per Plane,Field of Regard (deg),Field of View (deg),Maximum Slew Rate (deg/s),Number of Events per Day,Event Duration (hrs),Grid Type,Number of Grid-points,Preplanner,Points Considered

    # define performance metrics
    columns = list(parameters_df.columns.values)
    
    # organize data
    data = []
    for experiment_name in tqdm(run_names, desc='Compiling results', leave=False):
        # load results summary
        summary_path = os.path.join(results_path, experiment_name, 'summary.csv')
        summary : pd.DataFrame = pd.read_csv(summary_path)

        # get experiment parameters
        matching_experiment = [[name,*_] for name,*_ in parameters_df.values if name==experiment_name]
        datum = matching_experiment.pop()
        
        # collect xperiment results
        for key,value in summary.values:
            key : str; value : str

            if key not in columns: columns.append(key)           

            if not isinstance(value, float):
                try:
                    value = float(value)  
                except ValueError:
                    pass
        
            datum.append(value)

        # add to data
        data.append(datum)

    # create compiled data frame
    df = pd.DataFrame(data=data, columns=columns)

    df['Percent Ground Points Observed'] = df['Ground Points Observed'] / df['Ground Points']
    df['Number of Satellites'] = df['Number Planes'] * df['Number of Satellites per Plane']

    # save to csv
    df.to_csv(os.path.join(results_path, 'study_results_compiled.csv'),index=False)

    # create plots directory if necessary
    if not os.path.isdir('./plots'): os.mkdir('./plots')

    # scatter plots
    scatterplot_path = os.path.join('./plots', 'scatterplots')
    if not os.path.isdir(scatterplot_path): os.mkdir(scatterplot_path)
    
    # Apply the default theme
    sns.set_theme(style="whitegrid", palette="pastel")

    ys = [
            "Percent Ground Points Observed",
            "P(Event Detected)",
            "P(Event Observed)",
            "P(Event Re-observed)",
            "P(Event Co-observed)",
            "P(Event Partially Co-observed)",
            "P(Event Fully Co-observed)"
          ]
    xs = ["Points Considered"]

    for y in ys:
        for x in xs:
            # Create a visualization
            sns.relplot(
                data=df,
                x=x, 
                y=y, 
                col="Grid Type",
                hue="Constellation", 
                style="Preplanner", 
                # size="Number of Satellites",
            )

            # save or show graph
            if show_plots: plt.show()
            if save_plots: 
                plot_path = os.path.join(scatterplot_path, f'{y.lower()}_vs_{x.lower()}.png')
                plt.savefig(plot_path)

    x = 1 

    # # density histograms
    # histogram_path = os.path.join('./plots', 'histograms')
    # if not os.path.isdir(histogram_path): os.mkdir(histogram_path)

    # for metric in tqdm(columns, desc='Histogram Plots'):
    #     if 'obs' not in metric.lower() and 'events' not in metric.lower(): continue
    #     if 'Strategy' in metric: continue

    #     # create histogram
    #     sns.displot(df, x=metric, hue='Strategy', kind="kde", warn_singular=False)
    #     plt.xlim(left=0)
    #     if 'P(' in metric: plt.xlim(right=1)
    #     plt.grid(visible=True)
        
        # # save or show graph
        # if show_plots: plt.show()
        # if save_plots: 
        #     plot_path = os.path.join(histogram_path, f'{metric}.png')
        #     plt.savefig(plot_path)

    #     # close plot
    #     plt.close()

    # # grid layouts
    # grids_path = os.path.join('./plots', 'grids')
    # if not os.path.isdir(grids_path): os.mkdir(grids_path)

    # grid_names = list({grid_name.replace('.csv','') for grid_name in os.listdir('./resources')
    #              if 'grid' in grid_name})
    # grid_names.sort()

    # for grid_name in tqdm(grid_names, desc='Coverage Grid Plots'):
    #     # load grid
    #     grid_path = os.path.join('./resources', f'{grid_name}.csv')
    #     grid : pd.DataFrame = pd.read_csv(grid_path)

    #     # get lats and lons
    #     lats = [lat for lat,_ in grid.values]
    #     lons = [lon for _,lon in grid.values]

    #     # get grid info
    #     *_,grid_i = grid_name.split('_')
    #     field_of_regard,fov,agility,event_duration, \
    #         num_events,distribution,num_planes,sats_per_plane \
    #             = parameters[f'updated_experiment_{grid_i}']

    #     # plot ground points
    #     fig, ax = plt.subplots()
    #     m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
    #                 llcrnrlon=-180,urcrnrlon=180,resolution='c',ax=ax)
        
    #     x, y = m(lons,lats)
    #     m.drawmapboundary(fill_color='#99ffff')
    #     m.fillcontinents(color='#cc9966',lake_color='#99ffff')
    #     m.scatter(x,y,3,marker='o',color='r')

    #     m.drawcoastlines()
    #     # m.fillcontinents(color='coral',lake_color='aqua')
    #     # draw parallels and meridians.
    #     m.drawparallels(np.arange(-90.,91.,30.))
    #     m.drawmeridians(np.arange(-180.,181.,60.))
    #     m.drawmapboundary(fill_color='aqua') 

    #     # set title
    #     plt.title(f"{distribution} distribution (n={num_events})")
        
    #     # save or show graph
    #     if show_plots: plt.show()
    #     if save_plots: 
    #         plot_path = os.path.join(grids_path, f'{distribution}_grid_{grid_i}.png')
    #         plt.savefig(plot_path)
        
    #     # close plot
    #     plt.close()

    # x = 1

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
