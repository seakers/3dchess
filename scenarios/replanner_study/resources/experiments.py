import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats._qmc import LatinHypercube
from tqdm import tqdm

from chess3d.utils import print_welcome

def main(n_samples : int = 1, seed : int = 1000):
    """
    Generates a set of samples for experiments in a parameterized study using 
    a latin hypercube  approach and saves it to a csv file within the `resources` directory.
    """
    
    # set parameters
    # params = [
    #     ('Constellation',                       [
    #                                              (1,8), 
    #                                              (2,4), 
    #                                              (3,4), 
    #                                             #  (8,3)
    #                                              ]),
    #     ('Field of Regard (deg)',               [30,60]),
    #     ('Field of View (deg)',                 [1,10]),
    #     ('Maximum Slew Rate (deg/s)',           [1,10]),
    #     ('Number of Events per Day',            [10**(i) for i in range(2,5)]),
    #     ('Event Duration (hrs)',                [
    #                                             #  0.25, 
    #                                              1, 
    #                                              3, 
    #                                              6
    #                                             ]),
    #     ('Grid Type',                           [
    #                                              'fibonacci', 
    #                                             #  'inland', 
    #                                             #  'hydrolakes'
    #                                              ]),
    #     ('Number of Ground-Points',             [
    #                                              1000, 
    #                                             #  2500, 
    #                                              5000, 
    #                                              10000
    #                                              ]),
    #     ('Percent Ground-Points Considered',    [1]),
    #     ('Preplanning Period',                  [
    #                                             #  500, 
    #                                              np.Inf
    #                                              ])
    # ]

    params = [
        ('Constellation',                       [
                                                 (1,8), 
                                                 (2,4), 
                                                 (3,4), 
                                                 (8,3)
                                                 ]),
        ('Field of Regard (deg)',               [30,60]),
        ('Field of View (deg)',                 [1,10]),
        ('Maximum Slew Rate (deg/s)',           [1,10]),
        ('Number of Events per Day',            [10**(i) for i in range(2,5)]),
        ('Event Duration (hrs)',                [
                                                #  0.25, 
                                                 1, 
                                                 3, 
                                                 6
                                                ]),
        ('Grid Type',                           [
                                                #  'fibonacci', 
                                                 'inland', 
                                                 'hydrolakes'
                                                 ]),
        ('Number of Ground-Points',             [
                                                 1000, 
                                                 2500, 
                                                 5000, 
                                                 10000
                                                 ]),
        ('Percent Ground-Points Considered',    [1]),
        ('Preplanning Period',                  [
                                                 500, 
                                                #  np.Inf
                                                 ])
    ]

    # calculate lowest-common-multiple for estimating number of samples
    lcm = np.lcm.reduce([len(vals) for _,vals in params])

    # load failed scenarios
    failed_scenarios : pd.DataFrame = pd.read_csv('./experiments/failed_scenarios.csv')

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

    preplanners = [
                    'fifo', 
                    'heuristic', 
                    # 'dp'
                    ]
    replanners = [
                    'none', 
                    # 'broadcaster', 
                    'acbba'
                ]

    for preplanner in preplanners:
        for replanner in replanners:
            
            if preplanner == 'fifo' and replanner == 'broadcaster':
                continue

            df_temp : pd.DataFrame = feasible_scenarios.copy()

            df_temp['Preplanner'] = preplanner
            df_temp['Replanner'] = replanner
            df_temp['Name'] = [f'scenario_{preplanner}-{replanner}_{j}' for j in df_temp['Scenario ID'].values]

            df = pd.concat([df,df_temp],axis=0)

    name_column = df.pop('Name')
    df.insert(0, 'Name', name_column)
    df = df.sort_values('Scenario ID')

    # make dir if it doesn't exist
    if not os.path.isdir('./experiments'): os.mkdir('./experiments')

    # save to csv
    df.to_csv(f'./experiments/test_case_2_seed-{seed}.csv',index=False)

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
    print_welcome('Experiment generator for Preplanner Parametric Study')

    # generate experiments
    main(1)