import pandas
import matplotlib.pyplot as plt
import numpy as np


def plot_bars(name : str, data : dict, show=False, save=True) -> None:
    x = np.arange(len(scenarios))   # the label locations
    width = 0.25                    # the width of the bars
    multiplier = 0

    _, ax = plt.subplots(layout='constrained')
    max_vals = []
    for attribute, val in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, val, width, label=attribute)
        ax.bar_label(rects)
        multiplier += 1
        max_vals.append(max(val))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('#')
    ax.set_title(name)
    ax.set_xticks(x + width, scenarios)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim([0, 1.25*max(max_vals)])

    if show: plt.show()
    if save: plt.savefig(f'./plots/{name.lower()}.png')

if __name__ == '__main__':

    scenarios = ['fifo_no_colab',
                 'fifo_colab',
                 'cbba_1',
                 'cbba_2',
                 'cbba_3',
                #  'cbba_5'
                 ]
    
    summaries = {scenario : {} for scenario in scenarios}
    for scenario in scenarios:
        summary_path = f'./scenario2_{scenario}/results/summary.csv'
        summary_df : pandas.DataFrame = pandas.read_csv(summary_path)
        summaries[scenario] = {row['stat_name']: row['val'] 
                               for _, row in summary_df.iterrows()}
            
    # events detected 
    events = {
                # 'n_events'          : [summary['n_events'] for _, summary in summaries.items()],
                'n_events_detected' : [int(summary['n_events_detected']) for _, summary in summaries.items()],
                'n_events_obs'      : [int(summary['n_events_obs']) for _, summary in summaries.items()],
                'n_reqs_gen'        : [int(summary['n_reqs_gen']) for _, summary in summaries.items()]
    }
    plot_bars('Events', events)    

    # observations performed
    observations = {
                'n_obs'         : [int(summary['n_obs']) for _, summary in summaries.items()],
                'n_obs_unique'  : [int(summary['n_obs_unique']) for _, summary in summaries.items()],
                'n_obs_co'      : [int(summary['n_obs_co']) for _, summary in summaries.items()]
    }
    plot_bars('Observations', observations)

    # utility 
    utility = {
                'u_max'     : [float(summary['u_max']) for _, summary in summaries.items()],
                'u_total'   : [float(summary['u_total']) for _, summary in summaries.items()]
    }
    plot_bars('Utility', utility)

    print('DONE')