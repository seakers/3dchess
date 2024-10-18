import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats._qmc import LatinHypercube

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
        ('Event Duration (hrs)',        [0.25, 1, 3, 6])
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
    for sample in samples:
        row = [f'experiment_{j}']
        for i in range(len(sample)):
            _,vals = params[i]
            value = vals[sample[i]]
            if i == i_constellation:
                row.extend(list(value))
            else:
                row.append(value)
        data.append(row)
        j += 1

    # create data frame
    df = pd.DataFrame(data=data, columns=columns)

    # make dir if it doesn't exist
    if not os.path.isdir('./experiments'): os.mkdir('./experiments')

    # save to csv
    df.to_csv(f'./experiments/experiments_seed-{seed}.csv',index=False)

if __name__ == "__main__":
    main(4)