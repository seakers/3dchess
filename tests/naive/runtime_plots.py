import os
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

def create_plot_folder(scenario : str, agent : str):
    if not os.path.isdir('plots'): os.mkdir('plots')
    if not os.path.isdir(f'plots/{scenario}'): os.mkdir(f'plots/{scenario}')
    if not os.path.isdir(f'plots/{scenario}/{agent}'): os.mkdir(f'plots/{scenario}/{agent}')

def main(scenarios : list, agent : str, show_plot : bool = False, save_plot : bool = False):
    # get list of routines
    if not scenarios: raise ValueError('Must have at least one scenario in scenario list')
    dir_path = os.path.join('results', scenarios[0], agent, 'runtime')
    routines = os.listdir(dir_path)
    
    # generate a plot per routine
    for routine in tqdm(routines, f'Generating runtime performance plots for {agent}\'s routines', leave=False):
        # compile data
        data = None
        for scenario in scenarios:
            file_path = os.path.join(dir_path, routine)
            df : pd.DataFrame = pd.read_csv(file_path)
            df['t'] = range(len(df))
            df['scenario'] = scenario

            if data is None:
                data = df
            else:
                data = pd.concat([data, df], axis=0)

        # if data is incomplete, skip
        if len(data) / len(scenarios) < len(df): 
            continue
        if len(data) == 0:
            continue

        # create plots
        sns.set_theme()
        sns.relplot(
                    data=data, x='t', y="dt", col="scenario",
                    kind="line", errorbar=('ci', 95)
                    )

        # show or save plot    
        if show_plot: 
            plt.show()    
        if save_plot: 
            create_plot_folder(scenario, agent)
            routine = routine.replace('.csv','')
            routine = routine.replace('time_series-', '')
            plt.savefig(f'plots/{scenario}/{agent}/{routine}.png')
            plt.close()

if __name__ == '__main__':

    show_plot = False
    save_plot = True

    scenarios = [
        'ben_case'
    ]

    agents = [
        'manager',
        'environment',
        'thermal_0'
    ]
        
    for agent in tqdm(agents, desc='Generating runtime performance plots for agents'):
        main(scenarios, agent, False, True)