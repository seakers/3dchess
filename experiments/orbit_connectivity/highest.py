
from typing import List
import pandas as pd
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from chess3d.utils import print_welcome

from constellation import WalkerDeltaConstellation

def pareto_front(df: pd.DataFrame, objectives: dict) -> pd.DataFrame:
    """
    objectives: dict like {"mean_LCC": "max", "frac_LCC": "max", "P": "min"}
    """
    vals = df[list(objectives.keys())].to_numpy()

    # Convert all objectives to maximization
    for j, (_, sense) in enumerate(objectives.items()):
        if "min" in sense:
            vals[:, j] = -vals[:, j]

    # initiate pareto condition array
    is_pareto = np.ones(len(vals), dtype=bool)

    # check each point
    for i,val in tqdm(enumerate(vals), total=len(vals), desc="Finding Pareto front", leave=False):
        # skip non-pareto points
        if not is_pareto[i]: continue

        # any point dominates i?
        dominates = np.all(vals >= val, axis=1) & np.any(vals > val, axis=1)
        dominates[i] = False
        if np.any(dominates):
            is_pareto[i] = False

    return df[is_pareto].copy()

# def print_results(trial_params : List[dict]):
#     print("Optimal Parameters Found:")
    
#     print(header)
#     print("-" * len(header))

#     for params in trial_params:
#         print(f" - {params['num sats']} sats: {params['num planes']} planes, phasing {params['phasing param']}")

if __name__ == "__main__":
    """
    Optimizing Walker Delta Constellation Connectivity Experiment

    GOAL: find the optimal Walker Delta constellation parameters (i.e., number of planes and
    phasing parameter) that MAXIMIZE the connectivity within a satellite constellation of a
    given size.
    
    """
    # terminal welcome message
    print_welcome(f'Walker Delta Constellation Connectivity Experiment')

    # set inclination and altitude
    inc = 98.0  # [deg]
    alt = 550.0 # [km]

    # define number of satellites for each constellation
    # sats = [12, 48, 96, 192]
    trials = [2,4,6]

    trial_params = [None for _ in trials]
       
    # find optimal parameters for each trial
    for i, num_sats in enumerate(trials):
        # define search space
        PF_space = [ (p,f) 
                    for p in range(1, num_sats+1) # up to `num_sats` planes
                    for f in range(0, p)          # phasing param from 0 to p-1
                ]

        if len(PF_space) <= 100: # small search space, evaluate all options
            # initialize tracking variables
            metrics_df = None

            # evaluate all options
            for (num_planes, phasing) in tqdm(PF_space, desc=f"Evaluating all options for {num_sats} sats"):
                # create constellation
                constellation = WalkerDeltaConstellation(alt, inc, num_sats, num_planes, phasing)
                
                # evaluate connectivity
                metrics_series_df,scalar_metrics = constellation.evaluate_connectivity(debug=True)

                # convert to dataframe 
                scalar_metrics_dict = {
                    "alt [km]" : alt,
                    "inc [deg]" : inc,
                    "num sats" : num_sats,
                    "num planes" : num_planes,
                    "phasing param" : phasing,
                }
                scalar_metrics_dict.update({key : [val] for key, val in scalar_metrics.items()})
                scalar_metrics_df = pd.DataFrame(scalar_metrics_dict)

                # store results
                if metrics_df is None:
                    metrics_df = scalar_metrics_df
                else:
                    metrics_df = pd.concat([metrics_df, scalar_metrics_df], ignore_index=True)
    
            # find pareto front
            objectives = {
                "max lcc [norm]": "max", 
                "max lcc time-fraction [norm]": "max", 
                "avg lcc [norm]": "max", 
                "avg num components": "min",
            }
            pareto_df = pareto_front(metrics_df, objectives)

            # pick best option (highest max lcc)
            best_idx = pareto_df["max lcc [norm]"].idxmax()
            best_params = pareto_df.loc[best_idx]

            # store best params
            trial_params[i] = dict(best_params)
        
        else:
            # large search space, implement smarter search (TBD)
            raise NotImplementedError("Smarter search not yet implemented for large search spaces.")
    x = 1

