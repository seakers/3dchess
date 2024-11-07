"""
Compiles results and creates plots
"""

from itertools import combinations
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

def main(results_path : str, show_plots : bool, save_plots : bool, overwrite : bool) -> None:
    # get run names
    run_names = list({run_name for run_name in os.listdir(results_path)
                 if os.path.isfile(os.path.join(results_path, run_name, 'summary.csv'))})
    run_names.sort()

    # get run parameters
    parameters_df = pd.read_csv('./resources/experiments/experiments_seed-1000.csv')
    parameters = { params for params in parameters_df.columns.values }

    # define performance metrics
    columns = list(parameters_df.columns.values)
    
    # organize data
    data = []
    for experiment_name in tqdm(run_names, desc='Compiling Results'):
        # load results summary
        summary_path = os.path.join(results_path, experiment_name, 'summary.csv')
        summary : pd.DataFrame = pd.read_csv(summary_path)

        # get experiment parameters
        matching_experiment = [[name,*_] for name,*_ in parameters_df.values if name==experiment_name]
        datum = matching_experiment.pop()
        
        # collect xperiment results
        for key,value in summary.values:
            key : str; value : str

            if key not in columns: 
                columns.append(key)           
                # ys.append(key)

            if not isinstance(value, float):
                try:
                    value = float(value)  
                except ValueError:
                    pass
        
            datum.append(value)

        # add to data
        if not len(datum) == len(columns):
            x = 1
        assert len(datum) == len(columns)
        data.append(datum)

    # create compiled data frame
    df = pd.DataFrame(data=data, columns=columns)
    df.sort_values('Name')

    df['Percent Ground Points Observed'] = df['Ground Points Observed'] / df['Ground Points']
    df['Percent Ground Points Accessible'] = df['Ground Points Accessible'] / df['Ground Points']
    df['Percent Events Detected'] = df['Events Detected'] / df["Number of Events per Day"]
    df['Percent Events Observed'] = df['Events Observed'] / df["Number of Events per Day"]
    df['Percent Events Re-observed'] = df['Events Re-observed'] / df["Number of Events per Day"]
    df['Percent Events Co-observed'] = df['Events Co-observed'] / df["Number of Events per Day"]
    df['Percent Events Fully Co-observed'] = df['Events Fully Co-observed'] / df["Number of Events per Day"]
    df['Percent Events Partially Co-observed'] = df['Events Partially Co-observed'] / df["Number of Events per Day"]
    df['Percent Events Observed'] = df['Events Observed'] / df["Number of Events per Day"]
    df['Ground-Points Considered'] = df['Percent Ground-Points Considered'] * df['Number of Ground-Points']
    df['Number of Satellites'] = df['Number Planes'] * df['Number of Satellites per Plane']

    # save to csv
    df.to_csv(os.path.join(results_path, 'study_results_compiled.csv'),index=False)

    # create plots directory if necessary
    if not os.path.isdir('./plots'): os.mkdir('./plots')

    # SCATTER PLOTS
    scatterplot_path = os.path.join('./plots', 'scatterplots')
    if not os.path.isdir(scatterplot_path): os.mkdir(scatterplot_path)
    
    # Apply the default theme
    sns.set_theme(style="whitegrid", palette="Set2")

    # select data to be plotted
    ys : set[str] = {val for val in df.columns.values}
    ys.difference_update(parameters)
    ys.remove('Events')
    ys.remove('Number of Satellites')
    ys.remove('Ground Points')
    xs = [
          "Ground Points Accessible",
          "Percent Ground Points Accessible",
          "Ground-Points Considered",
          "Percent Ground-Points Considered",
          "Number of Ground-Points",
          "Number of Events per Day",
          "Events Detected",
          "Percent Events Detected"
        ]
    ys.difference_update(xs)
    
    vals = [(x,y) for y in ys for x in xs if x != y]

    for x,y in tqdm(vals, desc='Generating Scatter Plots'):
        # set plot name and path
        dep_var = y.replace(' ', '_')
        indep_var = x.replace(' ', '_')
        
        dep_var_path = os.path.join(scatterplot_path, y)
        if not os.path.isdir(dep_var_path): os.mkdir(dep_var_path)
        plot_path = os.path.join(dep_var_path, f'{dep_var}_vs_{indep_var}.png')

        # check if plot has already been generated
        if (show_plots or save_plots) and os.path.isfile(plot_path) and not overwrite: continue

        # create plot
        sns.relplot(
            data=df,
            x=x, 
            y=y, 
            col="Grid Type",
            hue="Number of Satellites", 
            # size="Number of Grid-points",
            style="Preplanner",
            palette="flare"
        )

        plt.xlim(left=0)
        plt.ylim(bottom=0)

        # save or show graph
        if show_plots: plt.show()
        if save_plots: plt.savefig(plot_path)

        # close plot
        plt.close()

    # HISTOGRAMS
    histogram_path = os.path.join('./plots', 'histograms')
    if not os.path.isdir(histogram_path): os.mkdir(histogram_path)

    vals_histogram = [y for y in ys if 'p(' in y.lower() or 'percent' in y.lower()]
    for y in tqdm(vals_histogram, desc='Generating Histogram Plots'):
        
        # set plot name and path
        histogram_name = y.replace(' ', '_')
        plot_path = os.path.join(histogram_path, f'{histogram_name}.png')

        # check if plot has already been generated
        if (show_plots or save_plots) and os.path.isfile(plot_path) and not overwrite: 
            continue

        # create histogram
        sns.displot(df, 
                    x=y, 
                    # hue='Strategy', 
                    kind="kde", 
                    col="Grid Type",
                    hue="Preplanner", 
                    # size="Number of Grid-points",<
                    # palette="flare",
                    warn_singular=False
                    )
        plt.xlim(left=0)
        plt.xlim(right=1)
        plt.grid(visible=True)
        
        # save or show graph
        if show_plots: plt.show()
        if save_plots: 
            plt.savefig(plot_path)
            print(f'saved {plot_path}')

        # close plot
        plt.close()

    # BOX PLOTS
    boxplot_path = os.path.join('./plots', 'boxplots')
    if not os.path.isdir(boxplot_path): os.mkdir(boxplot_path)

    for y in tqdm(ys, desc='Generating Box Plots'):
        # set plot name and path
        dep_var = y.replace(' ', '_')
        
        plot_path = os.path.join(boxplot_path, f'{dep_var}.png')

        # check if plot has already been generated
        if (show_plots or save_plots) and os.path.isfile(plot_path) and not overwrite: continue

        # create plot
        # sns.relplot(
        #     data=df,
        #     x=x, 
        #     y=y, 
        #     col="Grid Type",
        #     hue="Number of Satellites", 
        #     # size="Number of Grid-points",
        #     style="Preplanner",
        #     palette="flare"
        # )

        sns.boxplot(
            df, y="Preplanner", x=y, hue="Number of Satellites",
            whis=[0, 100], width=.6, palette="vlag"
        )

        plt.xlim(left=0)
        plt.ylim(bottom=0)

        plt.show()

        # save or show graph
        # if show_plots: plt.show()
        # if save_plots: plt.savefig(plot_path)

        # close plot
        plt.close()

if __name__  == "__main__":
    # set params
    results_path = './results'
    show_plots = False
    save_plots = True
    overwrite = False
    
    main(results_path, show_plots, save_plots, overwrite)