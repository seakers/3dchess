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
    params = [
        ('Constellation',                       [(1,8), (2,4), (3,4), (8,3)]),
        ('Field of Regard (deg)',               [30,60]),
        ('Field of View (deg)',                 [1,5,10]),
        ('Maximum Slew Rate (deg/s)',           [1,10]),
        ('Number of Events per Day',            [10**(i) for i in range(1,4)]),
        ('Event Duration (hrs)',                [0.25, 1, 3, 6]),
        ('Grid Type',                           ['hydrolakes', 'uniform', 'fibonacci']),
        ('Number of Ground-Points',             [100, 1000, 5000, 10000]),
        ('Preplanner',                          ['nadir', 'fifo']),
        ('Replanner',                           ['acbba', 'broadcaster']),
        ('Percent Ground-Points Considered',    [i/10 for i in range(1,11)])
    ]

    # calculate lowest-common-multiple for estimating number of samples
    lcm = np.lcm.reduce([len(vals) for _,vals in params])

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
        columns.insert(0,'Name')
        data = []
        j = 0
        for sample in tqdm(samples, desc='Generating experiments'):
            # create row of values 
            row = [f'experiment_{j}']
            for i in range(len(sample)):
                _,vals = params[i]
                value = vals[sample[i]]

                if i == i_constellation:
                    row.append(sample[i])
                    row.extend(list(value))
                else:
                    row.append(value)
            
            # check if experiment is feasible
            if is_feasible(row): 
                # add to list of experiments
                data.append(row)

                # update experiment index
                j += 1

        # create data frame
        df = pd.DataFrame(data=data, columns=columns)

        # check if enough samples are contained in the experiment list
        if len(df) >= lcm: break
        n_samples += 1

    # make dir if it doesn't exist
    if not os.path.isdir('./experiments'): os.mkdir('./experiments')

    # save to csv
    df.to_csv(f'./experiments/experiments_seed-{seed}.csv',index=False)

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

if __name__ == "__main__":
    # print welcome
    print_welcome('Experiment generator for Preplanner Parametric Study')

    # generate experiments
    main()