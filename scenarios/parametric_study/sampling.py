from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats._qmc import LatinHypercube

def main(n : int = 1):
    """
    Generates a set of samples for a parameterized study using a latin hypercube 
    approach and saves it to a csv file within the `resources` directory.
    """
    params = [
        ('Constellation',               [(1,8), (2,4), (3,4), (8,3)]),
        ('Field of Regard (deg)',       [30,60]),
        ('Field of View (deg)',         [1,5,10]),
        ('Maximum Slew Rate (deg/s)',   [1,10]),
        ('Number of Events per Day',    [10, 100, 1000, 10000]),
        ('Event Duration (hrs)',        [0.25, 1, 3, 6])
    ]
    lcm = np.lcm.reduce([len(vals) for _,vals in params])

    sampler = LatinHypercube(d=len(params))
    samples = sampler.integers(l_bounds=[0 for _ in params], 
                              u_bounds=[len(vals) for _,vals in params], 
                              n=n*lcm)

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

    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv('./resources/scenarios.csv',index=False)

if __name__ == "__main__":
    main(4)