from chess3d.study import ParametricStudy


if __name__ == '__main__':
    # set parameters
    params = {
        'Constellation' :                      [(1,8), (2,4), (3,4), (8,3)],
        'Field of Regard (deg)' :              [30,60],
        'Field of View (deg)' :                [1,5,10],
        'Maximum Slew Rate (deg/s)' :          [1,10],
        'Number of Events per Day' :           [10**(i) for i in range(1,4)],
        'Event Duration (hrs)' :               [0.25, 1, 3, 6],
        'Grid Type' :                          ['grid0', 'uniform', 'fibonacci'],
        'Number of Ground-Points' :            [100, 1000, 5000, 10000],
        'Preplanner' :                         ['nadir', 'fifo'],
        'Percent Ground-Points Considered' :   [i/10 for i in range(1,11)]
    }

    study = ParametricStudy(params, 1.0, './experiments', './resources')
    experiments, grids, events = study.generate_study(print=True)