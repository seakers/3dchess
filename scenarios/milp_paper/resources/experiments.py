import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats._qmc import LatinHypercube
from tqdm import tqdm

from chess3d.utils import print_banner

def main(experiment_name : str = "milp_generic_testcase", n_samples : int = 1, seed : int = 1000):
    """
    Generates a set of samples for experiments in a parameterized study using 
    a latin hypercube  approach and saves it to a csv file within the `resources` directory.
    """
    
    # set parameters
    params = [
        ('Constellation',                       [
                                                 (1,8), 
                                                 (2,4), 
                                                 (3,4), 
                                                 (8,3)
                                                 ]),
        ('Field of Regard (deg)',               [60]),
        ('Field of View (deg)',                 [1.0,10.0]),
        ('Maximum Slew Rate (deg/s)',           [1.0,10.0]),
        ('Number of Events per Day',            [10**(i) for i in range(2,5)]),
        ('Event Duration (hrs)',                [
                                                 0.25, 
                                                 1, 
                                                 3, 
                                                 6
                                                ]),
        ('Grid Type',                           [
                                                 'fibonacci',
                                                 'hydrolakes'
                                                 ]),
        ('Number of Ground-Points',             [
                                                 1000, 
                                                #  2500, 
                                                #  5000, 
                                                #  10000
                                                 ]),
        ('Percent Ground-Points Considered',    [1]),
        ('Preplanning Period',                  [
                                                 200,
                                                 500, 
                                                 1000,
                                                #  np.Inf
                                                 ]),
        ('Ground-Station Network',              ['gs_lakes',
                                                 'gs_nen'])
    ]

    # calculate lowest-common-multiple for estimating number of samples
    lcm = np.lcm.reduce([len(vals) for _,vals in params])

    # load failed scenarios
    failed_scenarios_path = './experiments/failed_scenarios.csv'
    failed_scenarios : pd.DataFrame = pd.read_csv(failed_scenarios_path) if os.path.isfile(failed_scenarios_path) else pd.DataFrame()

    # generate experiments
    n_samples_init = n_samples
    while True:
        # sample latin hypercube
        n = n_samples*lcm
        sampler : LatinHypercube = LatinHypercube(d=len(params),seed=seed)
        samples = sampler.integers(l_bounds=[0 for _ in params], 
                                u_bounds=[len(vals) for _,vals in params], 
                                n=n)

        # interpret samples and generate experiments
        columns = [param for param,_ in params]
        if 'Constellation' in columns:
            i_constellation = columns.index('Constellation')
            # columns.pop(i_constellation)
            columns.insert(i_constellation+1, 'Number Planes')
            columns.insert(i_constellation+2, 'Number of Satellites per Plane')
        columns.insert(0,'Scenario ID')
        data = []
        j = 0
        for sample in tqdm(samples, desc='Generating experiments'):
            if len(data) >= lcm*n_samples_init: 
                continue
            
            # create row of values 
            row = [j]
            for i in range(len(sample)):
                _,vals = params[i]
                value = vals[sample[i]]

                if i == i_constellation:
                    row.append(sample[i])
                    row.extend(list(value))
                else:
                    row.append(value)
            
            # check if experiment is feasible
            if is_feasible(row) and not has_failed(columns, row, failed_scenarios): 
                # add to list of experiments
                data.append(row)

                # update experiment index
                j += 1

        # create data frame
        feasible_scenarios = pd.DataFrame(data=data, columns=columns)

        # check if enough samples are contained in the experiment list
        if len(feasible_scenarios) >= lcm*n_samples_init: break
        n_samples += 1
   
    # create compiled data frame
    df = pd.DataFrame(data=[], columns=feasible_scenarios.columns.values)

    planners = [
                    'nadir', 
                    # 'heuristic', 
                    # 'dp',
                    'worker'
                    ]
    models = [
                    'static', 
                    'linear',
                    # 'reobs',
                    # 'revisit'
                ]

    for planner in planners:
        if planner != 'worker':
            df_temp : pd.DataFrame = feasible_scenarios.copy()
            
            df_temp['Planner'] = planner
            df_temp['Model'] = 'none'
            df_temp['Name'] = [f'scenario_{planner}-none_{j}' for j in df_temp['Scenario ID'].values]

            df = pd.concat([df,df_temp],axis=0)
            continue

        for model in models:
            df_temp : pd.DataFrame = feasible_scenarios.copy()
            df_temp['Planner'] = planner
            df_temp['Model'] = model
            df_temp['Name'] = [f'scenario_{planner}-{model}_{j}' for j in df_temp['Scenario ID'].values]

            df = pd.concat([df,df_temp],axis=0)

    name_column = df.pop('Name')
    df.insert(0, 'Name', name_column)
    df = df.sort_values('Scenario ID')

    # make dir if it doesn't exist
    if not os.path.isdir('./experiments'): os.mkdir('./experiments')

    # save to csv
    df.to_csv(f'./experiments/{experiment_name}_seed-{seed}.csv',index=False)

def is_feasible(row : list) -> bool:

    events_frequeny = row[7]    # [per day]
    event_duration = row[8]     # [hrs]
    gp_distribution = row[9]
    n_gps = row[10]
    
    # check if number of events can be acheived with number of ground points and event duration
    if not events_frequeny <= n_gps * (24 / event_duration) * (2/3): 
        return False

    # check if there are enough ground points in hydrolakes database
    if gp_distribution == 'hydrolakes' and n_gps > 5000: 
        return False 

    return True

def has_failed(columns : list, row : list, failed_scenarios : pd.DataFrame) -> bool:
    
    for _,failed_row in failed_scenarios.iterrows():
        if all([row[columns.index(param)]==failed_row[param] 
                for param in failed_row.index.values
                if param in columns]):
            return True
    
    return False

if __name__ == "__main__":
    # print welcome
    print_banner('Experiment generator for Preplanner Parametric Study')

    # generate experiments
    main("milp_lakes_testcase_nov_2025", 1)