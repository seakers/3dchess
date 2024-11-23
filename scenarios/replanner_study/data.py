"""
Compiles results and creates plots
"""

import os
from typing import Dict
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
from tqdm import tqdm

def main(results_path : str, show_plots : bool, save_plots : bool, overwrite : bool) -> None:
    
    # load and compile data
    experiments_path = './resources/experiments/experiments_seed-1000.csv'
    parameters, results_data = compile_data(results_path, experiments_path)
    results_data : pd.DataFrame

    # define preformance metrics and their optimality
    metrics = [                 
                ("Ground Points Accessible",'max'),
                ("Ground Points Observed",'max'),
                ("Ground Point Observations",'max'),
                ("Events Observable",'max'),
                ("Events Detected",'max'),
                ("Events Observed",'max'),
                ("Event Observations",'max'),
                ("Events Re-observable",'max'),
                ("Events Re-observed",'max'),
                ("Event Re-observations",'max'),
                ("Events Co-observable",'max'),
                ("Events Co-observed",'max'),
                ("Event Co-observations",'max'),
                ("Events Fully Co-observable",'max'),
                ("Events Fully Co-observed",'max'),
                ("Event Full Co-observations",'max'),
                ("Events Partially Co-observable",'max'),
                ("Events Partially Co-observed",'max'),
                ("Event Partial Co-observations",'max'),
                ("P(Ground Point Accessible)",'max'),
                ("P(Ground Point Observed)",'max'),
                ("P(Event at a GP)",'max'),
                ("P(Event Observable)",'max'),
                ("P(Event Re-observable)",'max'),
                ("P(Event Co-observable)",'max'),
                ("P(Event Fully Co-observable)",'max'),
                ("P(Event Partially Co-observable)",'max'),
                ("P(Event Detected)",'max'),
                ("P(Event Observed)",'max'),
                ("P(Event Re-observed)",'max'),
                ("P(Event Co-observed)",'max'),
                ("P(Event Fully Co-observed)",'max'),
                ("P(Event Partially Co-observed)",'max'),
                ("P(Event Observed | Observable)",'max'),
                ("P(Event Re-observed | Re-observable)",'max'),
                ("P(Event Co-observed | Co-observable)",'max'),
                ("P(Event Fully Co-observed | Fully Co-observable)",'max'),
                ("P(Event Partially Co-observed | Partially Co-observable)",'max'),
                ("P(Event Observation | Observation)",'max'),
                ("P(Event Re-observation | Observation)",'max'),
                ("P(Event Co-observation | Observation)",'max'),
                ("P(Event Full Co-observation | Observation)",'max'),
                ("P(Event Partial Co-observation | Observation)",'max'),
                ("P(Event Observed | Event Detected)",'max'),
                ("P(Event Co-observed | Event Detected)",'max'),
                ("P(Event Co-observed Fully | Event Detected)",'max'),
                ("P(Event Co-observed Partially | Event Detected)",'max'),
                ("Ground-Points Considered",'max')

                # ('Ground Points Accessible', 'max'),
                # # ('Percent Ground Points Accessible', 'max'),
                # ('Ground Points Observed', 'max'),
                # # ('Percent Ground Points Observed', 'max'),
                # ('Ground Point Observations', 'max'),
                # ('Average GP Reobservation Time [s]', 'min'),
                # # ('Standard Deviation of GP Reobservation Time [s]', 'none'),
                # # ('Median GP Reobservation Time [s]', 'min'),
                # ('Events Detected', 'max'),
                # ('P(Event Detected)', 'max'),
                # # ('Percent Events Detected', 'max'),
                # # ('Events Observed', 'max'),
                # # ('P(Event Observed)', 'max'),
                # # ('Percent Events Observed', 'max'),
                # # ('Event Observations', 'max'),
                # ('Events Re-observed', 'max'),
                # ('P(Event Re-observed)', 'max'),
                # # ('Percent Events Re-observed', 'max'),
                # # ('Event Re-observations', 'max'),
                # ('Average Event Reobservation Time [s]', 'min'),
                # # ('Standard Deviation of Event Reobservation Time [s]', 'max'),
                # # ('Median Event Reobservation Time [s]', 'max'),
                # # ('Events Co-observed', 'max'),
                # # ('P(Event Co-observed)', 'max'),
                # # ('Percent Events Co-observed', 'max'),
                # # ('Events Partially Co-observed', 'max'),
                # # ('P(Event Partially Co-observed)', 'max'),
                # # ('Percent Events Partially Co-observed', 'max'),
                # ('Events Fully Co-observed', 'max'),
                # ('P(Event Fully Co-observed)', 'max'),
                # # ('Percent Events Fully Co-observed', 'max'),
                # # ('Event Co-observations', 'max'),
                # # ('P(Event Observation | Observation)', 'max'),
                # # ('P(Event Observed | Event Detected)', 'max'),
                # ('P(Event Re-observed | Event Detected)', 'max'),
                # # ('P(Event Co-observed | Event Detected)', 'max'),
                # ('P(Event Co-observed Fully | Event Detected)', 'max')
                # # ('P(Event Co-observed Partially | Event Detected)', 'max'),
                ]

    # process results    
    processed_results, differential_results = process_results(results_path, results_data, parameters, metrics)

    # generate scenario reports
    generate_scenario_report(results_path, processed_results, parameters, metrics)
    
    # generate overal experiment report
    generate_experiment_report(results_path, processed_results, parameters, metrics)

    # create plots
    plot_results(processed_results, differential_results, parameters, metrics, show_plots, save_plots, overwrite)    


def compile_data(results_path : str, experiments_path : str) -> tuple:
    # get run names
    run_names = list({run_name for run_name in os.listdir(results_path)
                 if os.path.isfile(os.path.join(results_path, run_name, 'summary.csv'))})
    run_names.sort()

    # get run parameters
    parameters_df = pd.read_csv(experiments_path)
    parameters = { params for params in parameters_df.columns.values }

    # set compiled results file name
    compiled_results_path = os.path.join(results_path, 'study_results_compiled.csv')

    if os.path.isfile(compiled_results_path):
        df : pd.DataFrame = pd.read_csv(compiled_results_path)

    else:
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
            assert len(datum) == len(columns)
            data.append(datum)

        # create compiled data frame
        df = pd.DataFrame(data=data, columns=columns)

        # calculate additional metrics to results data
        df['Planner'] = df['Preplanner'] + '-' + df['Replanner']
        df['Ground Points Considered'] = df['Percent Ground-Points Considered'] * df['Number of Ground-Points']
        df['Number of Satellites'] = df['Number Planes'] * df['Number of Satellites per Plane']

        # save to csv
        df.to_csv(compiled_results_path,index=False)

    # set compiled results file name
    failed_results_path = os.path.join(results_path, 'failed_scenarios.csv')

    df_eval = df.copy()
    df_eval['Scenario ID'] = [row['Name'].split('_')[-1] for _,row in df.iterrows()]
    df_eval.sort_values('Scenario ID')
    df_eval : pd.DataFrame = df_eval[df_eval['P(Event Co-observable)'] == 0.0]

    parameters.add('Scenario ID')

    failed_scenarios = []
        
    unique_vals : list = df_eval['Scenario ID'].unique()
    unique_vals.sort()

    for unique_val in unique_vals:
        count = len([val for val in df_eval['Scenario ID'] if val == unique_val])

        if count == 4:
            failed_scenarios.append(int(unique_val))

    failed_scenarios.sort()
    print('failed_scenarios:', failed_scenarios)

    params_to_remove = [param for param in df.columns.values if param not in parameters]
    params_to_remove.extend(['Name', 'Preplanner', 'Replanner', 'Scenario ID'])
    df_eval = df_eval.drop(columns=params_to_remove)
    df_eval = df_eval.drop_duplicates()

    df_eval.to_csv(failed_results_path, index=False)

    return parameters, df

def process_results(results_path : str, results_data : pd.DataFrame, parameters : set, metrics : list):
    
    # initialized processed data 
    temp_data : pd.DataFrame = results_data.copy()

    # add scenario IDs
    temp_data['Scenario ID'] = [row['Name'].split('_')[-1] for _,row in results_data.iterrows()]
    temp_data.sort_values('Scenario ID')

    # enumerate planning strategies 
    planners = [(preplanner,replanner) 
                for preplanner in temp_data['Preplanner'].unique() 
                for replanner in temp_data['Replanner'].unique()]

    # initialize results
    processed_results = pd.DataFrame(data=[], columns=temp_data.columns)

    columns = ['Scenario ID','Preplanner', 'Grid Type']; columns.extend([f'Δ {metric}' for metric,_ in metrics])
    differential_results = pd.DataFrame(data=[], columns=columns)
    
    # processes scenarios
    scenario_ids = [int(id) for id in temp_data['Scenario ID'].unique()]
    scenario_ids.sort()
    
    for scenario_id in scenario_ids:
        # get results entries for a given scenario
        scenario_results : pd.DataFrame = temp_data[temp_data['Scenario ID'] == str(scenario_id)]

        scenario_grid_type = scenario_results['Grid Type'].unique()[-1]
        scenario_planners = { tuple(planner.split('-')) for planner in scenario_results['Planner']}
        
        # check if all planner runs were completed in this scenario
        if (    
            len(scenario_planners) != len(planners) 
            or any([planner not in scenario_planners for planner in planners])
            ): 
            continue
        
        # check if any of the metrics has invalid values as their best value
        invalid_metric = False
        for metric,optimum in metrics:
            # find best value
            best_results = [val for val in scenario_results[metric]
                            if val >= 0.0 and not np.isnan(val)]
            if not best_results:
                continue

            # result is invalid if leq 0.0
            best_val = max(best_results) if optimum.lower() == 'max' else min(best_results) 
            if best_val < 0.0: 
                invalid_metric = True
                break
        
        # if invalid, skip
        if invalid_metric: 
            continue                
        
        # add to list of completed scenarios
        processed_results = pd.concat([processed_results, scenario_results])

        # calculate differential values between metrics 
        preplanners = {preplanner for preplanner,_ in scenario_planners}
        
        for preplanner in preplanners:
            row = [scenario_id, preplanner, scenario_grid_type]
            replanner_results : pd.DataFrame = scenario_results[scenario_results['Preplanner'] == preplanner] 

            for metric,_ in metrics:
                vals = replanner_results[metric].values
                replanners = list(replanner_results['Replanner'].values)

                i_acbba = replanners.index('acbba')
                i_none = replanners.index('none')
                
                assert len(vals) == 2
                
                row.append(vals[i_acbba] - vals[i_none])

            differential_results.loc[-1] = row
            differential_results.index = differential_results.index + 1  # shifting index
            differential_results = differential_results.sort_index()  # sorting by index
                    

    # save to csv
    processed_results_path = os.path.join(results_path, 'processed_results.csv')
    processed_results.to_csv(processed_results_path,index=False)
    
    differential_results_path = os.path.join(results_path, 'differential_results.csv')
    differential_results.to_csv(differential_results_path,index=False)

    # return processed results
    return processed_results, differential_results

def generate_scenario_report(results_path : str, processed_results : pd.DataFrame, parameters : set, metrics : list):
    
    report_path = os.path.join(results_path, 'scenario_report.md')
    with open(report_path, 'w') as report:
        # Initialize report 
        report.write("# Scenario Report\n")

        # initialized processed data 
        processed_data : pd.DataFrame = processed_results.copy()

        # add scenario ID to results
        processed_data['Scenario ID'] = [row['Name'].split('_')[-1] for _,row in processed_results.iterrows()]
        processed_data.sort_values('Scenario ID')

        # processes completed scenarios
        scenario_ids = [int(id) for id in processed_data['Scenario ID'].unique()]
        scenario_ids.sort()

        report.write(f'Number of candidate scenarios: {len(scenario_ids)}\n')
        
        for scenario_id in scenario_ids:
            scenario_results : pd.DataFrame = processed_data[processed_data['Scenario ID'] == str(scenario_id)]
            scenario_results = scenario_results.sort_values('Name')

            # add scenario to report
            report.write(f'## Scenario {scenario_id}\n|Parameter|Value|\n|-|-|\n')

            # write scenario parameters
            for param in scenario_results.columns.values:
                if param not in parameters: continue
                if param in ['Name', 'Constellation', 'Preplanner', 'Replanner']: continue
                
                val : str = max(scenario_results[param])
                
                report.write(f'|{param}| \t {val}|\n')
            
            # get best performing planner 
            best_planner : pd.DataFrame = scenario_results[scenario_results['P(Event Co-observed Fully | Event Detected)'] 
                                                        == max(scenario_results['P(Event Co-observed Fully | Event Detected)'])]
            
            for _,row in best_planner.iterrows(): best_planner = row

            # create result table
            scenario_planners = [(preplanner,replanner) 
                            for preplanner in scenario_results['Preplanner'].unique() 
                            for replanner in scenario_results['Replanner'].unique()]
            
            report.write(f'\n| Performance Metric |')
            for preplanner,replanner in scenario_planners:
                report.write(f' {preplanner}-{replanner} | ')
            report.write('\n| - |')
            for preplanner,replanner in scenario_planners:
                report.write(f'- | ')
            report.write('\n')
            
            best_planners = {(preplanner,replanner) : 0 for preplanner,replanner in scenario_planners}
            for metric,optimum in metrics:
                
                # find maximum value
                if optimum.lower() == 'max':
                    best_val = max(scenario_results[metric]) 
                    if best_val <= 0.0 or np.isnan(best_val): continue
                else:
                    best_results : pd.DataFrame = scenario_results[scenario_results[metric] > 0.0]
                    if not best_results.empty:
                        best_val = min(best_results[metric]) 
                    else:
                        best_val = None
                        continue

                metric_name = metric.replace("|","\|")
                report.write(f'| {metric_name} |')

                for preplanner,replanner in scenario_planners:
                    # get metric value for this planner combination
                    matching_row = scenario_results[scenario_results['Preplanner']==preplanner]
                    matching_row = matching_row[matching_row['Replanner']==replanner]
                    
                    val = max(matching_row[metric])

                    # write to report
                    if val != best_val:
                        report.write(f' {np.round(val,3)} |')
                    else:
                        report.write(f' **{np.round(val,3)}** |')
                        best_planners[(preplanner,replanner)] += 1

                report.write('\n')

def generate_experiment_report(results_path : str, processed_results : pd.DataFrame, parameters : set, metrics : list):
    
    report_path = os.path.join(results_path, 'experiment_report.md')
    with open(report_path, 'w') as report:
        
        # Initialize report 
        report.write("# Experiments Report\n")

        # initialized processed data 
        processed_data : pd.DataFrame = processed_results.copy()

        # add scenario ID to results
        processed_data['Scenario ID'] = [row['Name'].split('_')[-1] for _,row in processed_results.iterrows()]
        processed_data.sort_values('Scenario ID')

        # processes completed scenarios
        scenario_ids = [int(id) for id in processed_data['Scenario ID'].unique()]
        scenario_ids.sort()

        report.write(f'Number of candidate scenarios: {len(scenario_ids)}\n')

        # add experiment parameters 
        report.write(f'## Experiment Parameters \n|Parameter|Value|\n|-|-|\n')

        # write scenario parameters
        for param in processed_data.columns.values:
            if param not in parameters: continue
            if param in ['Name', 'Constellation', 'Preplanner', 'Replanner']: continue
            
            vals : list = list(processed_data[param].unique())
            vals.sort()
            
            report.write(f'|{param}| \t {vals}|\n')

        # create result table
        scenario_planners = [(preplanner,replanner) 
                            for preplanner in processed_data['Preplanner'].unique() 
                            for replanner in processed_data['Replanner'].unique()]
        
        report.write(f'\n| Performance Metric |')
        for preplanner,replanner in scenario_planners:
            report.write(f' {preplanner}-{replanner} | ')
        report.write('\n| - |')
        for preplanner,replanner in scenario_planners:
            report.write(f'- | ')
        report.write('\n')
            
        # generate compiled table
        for metric,optimum in metrics:
            if 'P(' not in metric and 'Percent' not in metric: continue

            metric_name = metric.replace("|","\|")
            report.write(f'| {metric_name} |')

            for preplanner,replanner in scenario_planners:
                data = processed_data[processed_data['Preplanner'] == preplanner]
                data = data[data['Replanner'] == replanner]
                
                vals = [val for val in data[metric] 
                        if not np.isnan(val)
                        ]
                avg = np.round(np.average(vals),3)
                dev = np.round(np.std(vals),3)

                report.write(f' {avg} ± {dev} | ')

            report.write('\n')

def plot_results(processed_results : pd.DataFrame, differential_results : pd.DataFrame, parameters : list, metrics : list, show_plots : bool, save_plots : bool, overwrite : bool) -> None:
    # create plots directory if necessary
    if not os.path.isdir('./plots'): os.mkdir('./plots')
    
    # select data to be plotted
    ys : set[str] = {val for val in processed_results.columns.values}
    # ys : set[str] = {val for val,_ in metrics}
    ys.difference_update(parameters)
    ys.difference_update({'Events',
                          'Number of Satellites',
                          'Ground Points',
                          'Planner',
                          'Scenario ID'})
    
    xs = [
          "Ground Points Accessible",
          "P(Ground Point Accessible)",
          "Ground-Points Considered",
          "Percent Ground-Points Considered",
          "Number of Ground-Points",
          "Number of Events per Day",
          "Events Detected",
          "P(Event Detected)",
          "P(Event Observable)"
        ]
    # ys.difference_update(xs)
       
    # # SCATTER PLOTS
    generate_scatterplots(processed_results, ys, xs, show_plots, save_plots, overwrite, 'completed')

    # DENSITY HISTOGRAMS
    generate_density_histograms(processed_results, ys, show_plots, save_plots, overwrite, 'completed')

    # HISTOGRAMS
    generate_histograms(processed_results, ys, show_plots, save_plots, overwrite, 'completed')

    # Plot differential data
    ys : set[str] = {val for val in differential_results.columns.values}
    ys.remove('Scenario ID')
    ys.difference_update(xs)

    # DENSITY HISTOGRAMS
    generate_density_histograms(differential_results, ys, show_plots, save_plots, overwrite, 'differential')

    # HISTOGRAMS
    generate_histograms(differential_results, ys, show_plots, save_plots, overwrite, 'differential')

def generate_scatterplots(results_data : pd.DataFrame, ys : set, xs : list, show_plots : bool, save_plots : bool, overwrite : bool, dir_name : str) -> None:
    """ Creates scatter plots from results data """
    # set ouput path
    scatterplot_path = './plots'
    for path_element in ['scatterplots', dir_name]:
        scatterplot_path = os.path.join(scatterplot_path, path_element)
        if not os.path.isdir(scatterplot_path): os.mkdir(scatterplot_path)
    
    # apply the default theme
    sns.set_theme(style="whitegrid", palette="Set2")

    # generate plots
    vals = [(x,y) for y in ys for x in xs if x != y]
    for x,y_vals in tqdm(vals, desc='Generating Scatter Plots'):
        # set plot name and path
        dep_var = y_vals.replace(' ', '_')
        indep_var = x.replace(' ', '_')
        
        dep_var_path = os.path.join(scatterplot_path, y_vals)
        if not os.path.isdir(dep_var_path): os.mkdir(dep_var_path)
        plot_path = os.path.join(dep_var_path, f'{dep_var}_vs_{indep_var}.png')

        # check if plot has already been generated
        if (show_plots or save_plots) and os.path.isfile(plot_path) and not overwrite: continue
        if all([np.isnan(val) for val in results_data[y_vals].values]): continue

        # create plot
        sns.relplot(
            data=results_data,
            x=x, 
            y=y_vals, 
            col="Grid Type",
            hue="Number of Satellites", 
            # size="Number of Grid-points",
            style="Replanner",
            palette="flare"
        )

        plt.xlim(left=0)
        plt.ylim(bottom=0)

        # save or show graph
        if show_plots: plt.show()
        if save_plots: plt.savefig(plot_path)

        # close plot
        plt.close()

def generate_density_histograms(results_data : pd.DataFrame, ys : list,  show_plots : bool, save_plots : bool, overwrite : bool, dir_name : str) -> None:
    # set ouput path
    kde_histogram_path = './plots'
    for path_element in ['kde_histograms', dir_name]:
        kde_histogram_path = os.path.join(kde_histogram_path, path_element)
        if not os.path.isdir(kde_histogram_path): os.mkdir(kde_histogram_path)

    # apply the default theme
    sns.set_theme(style="whitegrid", palette="Set2")

    # generate plots
    vals_histogram = [y for y in ys if 'p(' in y.lower() or 'percent' in y.lower()]
    for y_vals in tqdm(vals_histogram, desc='Generating Density Histogram Plots'):
        # set plot name and path
        kde_histogram_name = y_vals.replace(' ', '_')
        plot_path = os.path.join(kde_histogram_path, f'{kde_histogram_name}.png')

        # check if plot has already been generated
        if (show_plots or save_plots) and os.path.isfile(plot_path) and not overwrite: continue
        if all([np.isnan(val) for val in results_data[y_vals].values]): continue

        # create histogram
        left,right = plt.xlim()
        if 'dif' not in dir_name:
            sns.displot(results_data, 
                        x=y_vals, 
                        kind="kde", 
                        col="Grid Type",
                        # row="Planner", 
                        hue='Replanner',
                        warn_singular=False,
                        )
            plt.xlim(left=0)
            if right > 0.50: 
                plt.xlim(right=1)
        else:
            sns.displot(results_data, 
                        x=y_vals, 
                        kind="kde", 
                        col="Grid Type",
                        row="Preplanner", 
                        # hue='Replanner',
                        warn_singular=False
                        )
            
            if left < -1 or 1 < right:
                plt.xlim(left=-1)
                plt.xlim(right=1)
            else:
                bound = max([abs(left),abs(right)])
                plt.xlim(left=-bound)
                plt.xlim(right=bound)
            # plt.suptitle(f'Difference of `{y_vals}` (ACBBA - None)')
        
        # save or show graph
        if show_plots: plt.show()
        if save_plots: 
            plt.savefig(plot_path)

        # close plot
        plt.close()

def generate_histograms(results_data : pd.DataFrame, ys : list,  show_plots : bool, save_plots : bool, overwrite : bool, dir_name : str) -> None:
    # set ouput path
    histogram_path = './plots'
    for path_element in ['histograms', dir_name]:
        histogram_path = os.path.join(histogram_path, path_element)
        if not os.path.isdir(histogram_path): os.mkdir(histogram_path)

    # apply the default theme
    sns.set_theme(style="whitegrid", palette="Set2")

    # generate plots
    vals_histogram = [y for y in ys if 'p(' in y.lower() or 'percent' in y.lower()]
    for y_val in tqdm(vals_histogram, desc='Generating Histogram Plots'):
        # set plot name and path
        histogram_name = y_val.replace(' ', '_')
        plot_path = os.path.join(histogram_path, f'{histogram_name}.png')

        # check if plot has already been generated
        if (show_plots or save_plots) and os.path.isfile(plot_path) and not overwrite: continue        
        if all([np.isnan(val) for val in results_data[y_val].values]): continue

        # create histogram
        if 'dif' not in dir_name:
            sns.displot(results_data, 
                        x=y_val, 
                        kind="hist", 
                        col="Grid Type",
                        row="Replanner",
                        bins=10,
                        # size="Number of Grid-points",
                        # palette="flare",
                        # warn_singular=False
                        )
            plt.xlim(left=0)
            plt.xlim(right=1)
        else:
            sns.displot(results_data, 
                        x=y_val, 
                        kind="hist", 
                        col="Grid Type",
                        row="Preplanner",
                        bins=10,
                        # size="Number of Grid-points",
                        # palette="flare",
                        # warn_singular=False
                        )
            left,right = plt.xlim()
            # if left < -1 or 1 < right:
            #     plt.xlim(left=-1)
            #     plt.xlim(right=1)
            # else:
            bound = max([abs(left),abs(right)])
            plt.xlim(left=-bound)
            plt.xlim(right=bound)
        
        # save or show graph
        if show_plots: plt.show()
        if save_plots: 
            plt.savefig(plot_path)

        # close plot
        plt.close()

def generate_box_plots(results_data : pd.DataFrame, ys : list, show_plots : bool, save_plots : bool, overwrite : bool) -> None:
    boxplot_path = os.path.join('./plots', 'boxplots')
    if not os.path.isdir(boxplot_path): os.mkdir(boxplot_path)
   
    grid_types = list(results_data['Grid Type'].unique())
    box_vals = [val for val in ys 
                if 'Average' not in val
                and 'Standard' not in val
                and 'Mean' not in val
                ]
    
    for y_vals in tqdm(box_vals, desc='Generating Box Sublots'):
        fig, axs = plt.subplots(nrows=1, 
                                ncols=len(grid_types), 
                                sharex=True,
                                figsize=(12, 6))
        
        for grid_type in grid_types:
            # set plot name and path
            dep_var = y_vals.replace(' ', '_')
            plot_path = os.path.join(boxplot_path, f'{dep_var}.png')

            # check if plot has already been generated
            if (show_plots or save_plots) and os.path.isfile(plot_path) and not overwrite: continue
            
            # get grid indeces
            grid_index = grid_types.index(grid_type)
            ax : Axes = axs[grid_index]

            sns.boxplot(
                results_data[results_data['Grid Type'] == grid_type], 
                x="Planner", 
                y=y_vals, 
                hue="Number of Satellites",
                whis=[0, 100], 
                # width=.6, 
                palette="flare",
                ax=ax
            )   
            ax.title.set_text(grid_type)

        y_lim_bottom,y_lim_top = max([ax.get_ylim() for ax in axs])
        for ax in axs:  
            if ax != axs[0]: 
                ax.set_ylabel('')
            if ax == axs[-1]:
                # Put a legend to the right of the current axis
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Num of Satellites')
            else:
                ax.get_legend().remove()

            ax.set_ylim(bottom=y_lim_bottom,top=y_lim_top)

        fig.set_figwidth(25)
        fig.set_figheight(6)

        # save or show graph
        if show_plots: plt.show()
        if save_plots: plt.savefig(plot_path)

        # close plot
        plt.close()

if __name__  == "__main__":
    # set params
    results_path = './results'
    show_plots = False
    save_plots = True
    overwrite = False
    
    main(results_path, show_plots, save_plots, overwrite)