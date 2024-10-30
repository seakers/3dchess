import os
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def main(scenarios : list, agent : str, routine : str):
    data = None
    for scenario in scenarios:
        file_path = os.path.join('results', scenario, agent, 'runtime', f'time_series-{routine}.csv')
        df : pd.DataFrame = pd.read_csv(file_path)
        df['t'] = range(len(df))
        df['scenario'] = scenario


        if data is None:
            data = df
        else:
            data = pd.concat([data, df], axis=0)

    if len(data) <= len(scenarios) : return

    sns.set_theme()
    
    sns.relplot(
                data=data, x='t', y="dt", col="scenario",
                kind="line", errorbar=('ci', 95)
                )
    print(f'{scenario} : {agent} `{routine}()` runtime')
    plt.show()
    x = 1

if __name__ == '__main__':

    scenarios = [
        'naive_normal',
        'naive_selective',
        'naive_updated'
    ]

    agents = [
        'manager',
        'environment',
        'thermal_0'
    ]
    

    routines = {
        'manager' : [
            '_execute',
            'sim_wait',
            'clock_wait',
            'thermal_0_wait',
            'ENVIRONMENT_wait'
        ],

        'thermal_0' : [
            'sense',
            'think',
            'do',
            'perform_broadcast',
            'perform_observation',
            'perform_state_change',
            'perform_wait_for_messages'
        ],
        
        'environment' : [
            'handle_agent_broadcast',
            'handle_agent_request',
            'handle_agent_state',
            'handle_manager_broadcast',
            'handle_observation',
            'handle_request',
            'query_measurement_data'
        ]
    }
    
    for agent in agents:
        for routine in routines[agent]:
            main(scenarios, agent, routine)