import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats._qmc import LatinHypercube
from tqdm import tqdm

def main(n_samples : int = 1, seed : int = 1000):
    """
    Generates a set of samples for experiments in a parameterized study using 
    a latin hypercube  approach and saves it to a csv file within the `resources` directory.
    """
    
    # set parameters
    params = [
        ('Constellation',               [(1,8), (2,4), (3,4), (8,3)]),
        ('Field of Regard (deg)',       [30,60]),
        ('Field of View (deg)',         [1,5,10]),
        ('Maximum Slew Rate (deg/s)',   [1,10]),
        ('Number of Events per Day',    [10, 100, 1000, 10000]),
        ('Event Duration (hrs)',        [0.25, 1, 3, 6]),
        ('Grid Type',                   ['hydrolakes', 'uniform', 'fibonacci']),
        ('Number of Grid-points',       [100, 500, 1000, 5000]),
        ('Preplanner',                  ['nadir', 'naive']),
        ('Points Considered',           [1/10, 1/4, 1/3, 1/2, 1.0])
    ]

    # calculate lowest-common-multiple for estimating number of samples
    lcm = np.lcm.reduce([len(vals) for _,vals in params])
    n = n_samples*lcm

    # sample latin hypercube
    sampler : LatinHypercube = LatinHypercube(d=len(params),seed=seed)
    samples = sampler.integers(l_bounds=[0 for _ in params], 
                              u_bounds=[len(vals) for _,vals in params], 
                              n=n)

    # interpret samples and generate experiments
    columns = [param for param,_ in params]
    if 'Constellation' in columns:
        i_constellation = columns.index('Constellation')
        columns.pop(i_constellation)
        columns.insert(i_constellation, 'Number Planes')
        columns.insert(i_constellation+1, 'Number of Satellites per Plane')
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
                row.extend(list(value))
            else:
                row.append(value)
        
        # if experiment is feasible, add to list of experiments
        if is_feasible(row): data.append(row)

        # update experiment index
        j += 1

    # create data frame
    df = pd.DataFrame(data=data, columns=columns)

    # make dir if it doesn't exist
    if not os.path.isdir('./experiments'): os.mkdir('./experiments')

    # save to csv
    df.to_csv(f'./experiments/experiments_seed-{seed}.csv',index=False)

def is_feasible(row : list) -> bool:
    # check if number of events can be acheived with number of ground points and event duration
    return row[6] <= row[9] * (24 / row[7]) * (2/3)

if __name__ == "__main__":
    main(1)