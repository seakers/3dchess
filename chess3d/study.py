import os
import random
from typing import Dict
import numpy as np
import pandas as pd
from scipy.stats._qmc import LatinHypercube
from tqdm import tqdm


class ParametricStudy:
    """ Contains tools for creating parametric studies for different planners """
    def __init__(self, params : dict, sim_duration : float = 1.0, outdir : str = '.', resources_dir : str = '.') -> None:
        """
        Creates an instance of a parametric study to be generated
        """
        super().__init__()

        required_vals = [
            'Constellation',
            'Number of Events per Day',
            'Event Duration (hrs)',
            'Grid Type'
        ]
        for req in required_vals:
            if req not in params: raise ValueError(f'`params` must contain `{req}` information`')
            
        if any([not isinstance(val, list) for _,val in params.items()]):
            raise ValueError(f'`params` must only contain lists of values')

        self.params = params
        self.sim_duration = sim_duration
        self.outdir = outdir
        self.resources_dir = resources_dir

    def generate_study(self, n_samples : int = 1, seed : int = 1000, print_to_csv : bool = False) -> tuple:
        # set random seed
        random.seed(seed)

        # generate experiments
        experiments : pd.DataFrame = self.generate_experiments(n_samples, seed)        

        # generate grids 
        grids : Dict[str, pd.DataFrame] = self.generate_grids(experiments, seed)

        # generate events
        events : Dict[str, pd.DataFrame] = self.generate_events(experiments, grids, seed)

        if print_to_csv:
            if not os.path.isdir(self.outdir): os.mkdir(self.outdir)

            seed_dir = os.path.join(self.outdir, f'seed-{seed}')
            if not os.path.isdir(seed_dir): os.mkdir(seed_dir)

            # print experiments to `csv` file
            experiments_path = os.path.join(seed_dir, f'experiments.csv')
            experiments.to_csv(experiments_path,index=False)

            # print grids to `csv` file
            grids_path = os.path.join(seed_dir, 'grids')
            if not os.path.isdir(grids_path): os.mkdir(grids_path)

            for experiment_name,grid in tqdm(grids.items(), desc='Printing coverage grids'):
                grid_path = os.path.join(grids_path,f'{experiment_name}_grid.csv')
                grid.to_csv(grid_path,index=False)

            # print events to `csv` file
            events_path = os.path.join(seed_dir, 'events')
            if not os.path.isdir(events_path): os.mkdir(events_path)

            for experiment_name,grid in tqdm(events.items(), desc='Printing experiment events'):
                grid_path = os.path.join(events_path,f'{experiment_name}_events.csv')
                grid.to_csv(grid_path,index=False)

        return experiments, grids, events

    
    def generate_experiments(self, n_samples : int, seed : int) -> pd.DataFrame:
        # calculate lowest-common-multiple for estimating number of samples
        lcm = np.lcm.reduce([len(vals) for _,vals in self.params.items()])

        while True:
            # sample latin hypercube
            n = n_samples*lcm
            sampler : LatinHypercube = LatinHypercube(d=len(self.params),seed=seed)
            samples = sampler.integers(l_bounds=[0 for _ in self.params], 
                                    u_bounds=[len(vals) for _,vals in self.params.items()], 
                                    n=n)

            # interpret samples and generate experiments
            columns = [param for param in self.params]
            i_constellation = columns.index('Constellation')
            data = []
            j = 0
            for sample in tqdm(samples, desc='Generating experiments'):
                # create row of values 
                row = [self.params[columns[i]][sample[i]]
                       for i in range(len(sample))]
                
                # check if experiment is feasible
                if self._is_feasible(row, columns): 
                    # expand constellation information
                    n_planes,n_sats = row[i_constellation]
                    row[i_constellation] = self.params['Constellation'].index((n_planes,n_sats))
                    row.insert(i_constellation, n_planes)
                    row.insert(i_constellation+2, n_sats)

                    # add experiment name 
                    row.insert(0, f'experiment_{j}')

                    # add to list of experiments
                    data.append(row)

                    # update experiment index
                    j += 1

            # update columns info
            columns.insert(i_constellation+1, 'Number Planes')
            columns.insert(i_constellation+2, 'Number of Satellites per Plane')
            columns.insert(0,'Name')

            # create data frame
            df = pd.DataFrame(data=data, columns=columns)

            # check if enough samples are contained in the experiment list
            if len(df) >= lcm: return df
            n_samples += 1

            if n_samples > 100: raise RuntimeError('Could not generate enough feasible experiments.')


    def _is_feasible(self, experiment_row : list, columns : list) -> bool:
        """Checks if an experiment is feasible """
        events_frequeny = experiment_row[columns.index('Number of Events per Day')]    # [per day]
        event_duration = experiment_row[columns.index('Event Duration (hrs)')]     # [hrs]
        gp_distribution = experiment_row[columns.index('Grid Type')]
        n_gps = experiment_row[columns.index('Number of Ground-Points')]
        
        # check if number of events can be acheived with number of ground points and event duration
        if not events_frequeny <= n_gps * (24 / event_duration) * (2/3): 
            return False

        # check if there are enough ground points in hydrolakes database
        if gp_distribution == 'hydrolakes' and n_gps > 5000: 
            return False 

        return True
    
    def generate_grids(self, experiments : pd.DataFrame, seed : int) -> list:
        # initialize grids
        grids : dict = {}

        # generate one grid per experiment
        for _,row in tqdm(experiments.iterrows(),
                          desc='Generating Coverate Grids'):
            # get grid parameters
            experiment_name : str = row['Name']
            grid_type : str = row['Grid Type']
            n_points : int = row['Number of Ground-Points']

            # generate grids 
            if grid_type.lower() == 'uniform':
                grid : pd.DataFrame = self.create_uniform_grid(n_points)
            
            elif grid_type.lower() == 'clustered':
                raise NotImplementedError(f'Cannot generate grid of type `{grid_type}`. Type not yet supported.')
            
            elif grid_type.lower() == 'fibonacci':
                grid : pd.DataFrame = self.create_fibonacci_grid(n_points)
            
            else:
                grid : pd.DataFrame = self.sample_custom_grid(grid_type, n_points)
            
            # add to compiled list of grids
            grid = grid.sort_values(by=['lat [deg]', 'lon [deg]'])
            grids[experiment_name] = grid
        
        # return list of grids
        return grids
            
    def create_uniform_grid(self,
                            n_points : int
                            ) -> str:
        # calculate spacing
        k_1 = (1/2) * (1 - np.sqrt( 2*n_points - 3 ))
        k_2 = (1/2) * (np.sqrt( 2*n_points - 3 ) + 1)
        k = np.floor(max(k_1,k_2))

        spacing = 180/k # deg / plane

        # generate grid
        groundpoints = [[lat, lon] 
                        for lat in np.linspace(-90, 90, int(180/spacing)+1)
                        for lon in np.linspace(-180, 180, int(360/spacing)+1)
                        if lon < 180
                        ]

        assert len(groundpoints) >= n_points

        # create dataframe
        return pd.DataFrame(data=groundpoints, columns=['lat [deg]','lon [deg]'])

    def create_fibonacci_grid(self, 
                              n_points : int
                              ) -> str:        
        # set parameters
        golden_ratio = (1 + np.sqrt(5)) / 2
        N = int(np.round((n_points - 1) / 2))
        Ns = [i for i in range(-N,N+1)]

        # generate grid
        groundpoints = []
        for i in Ns:
            lat = np.arcsin( 2*i / (2*N + 1) ) * 180 / np.pi
            lon = np.mod(i,golden_ratio) * 360 / golden_ratio

            if lon < -180:
                lon += 360
            if lon > 180:
                lon -= 360

            groundpoints.append([lat,lon])

        # return grid
        return pd.DataFrame(data=groundpoints, columns=['lat [deg]','lon [deg]'])

    def sample_custom_grid(self,
                           grid_name : str,
                           n_points : int
                           ) -> str:

        # load original grid dataset
        path_to_grid = os.path.join(self.resources_dir, f'{grid_name}.csv')
        grid : pd.DataFrame = pd.read_csv(path_to_grid)

        # check inputs
        assert n_points <= len(grid)

        # collect groundpoints
        all_groundpoints = [[lat,lon] for lat,lon in grid.values]
        groundpoints = random.sample(all_groundpoints, n_points)

        # create dataframe
        return pd.DataFrame(data=groundpoints, columns=['lat [deg]','lon [deg]'])

    def generate_events(self, experiments : pd.DataFrame, grids : Dict[str, pd.DataFrame], seed : int) -> dict:

        # run simulation for each set of parameters
        events : Dict[str, pd.DataFrame] = {}
        for _,row in tqdm(experiments.iterrows(), 
                          desc = 'Generating Events'):

            # extract event parameters
            experiment_name = row['Name']
            event_duration = row['Event Duration (hrs)']
            n_events = row['Number of Events per Day']
            min_severity = 0.0
            max_severity = 100
            measurement_list = ['sar', 'visual', 'thermal']


            # get grid
            grid : pd.DataFrame = grids[experiment_name]

            # generate events
            experiment_events = self.create_events(experiment_name, 
                                                    grid, 
                                                    n_events, 
                                                    event_duration, 
                                                    min_severity, 
                                                    max_severity, 
                                                    measurement_list, 
                                                    seed)
            
            events[experiment_name] = experiment_events
        
        return events            
            
    def create_events(self,
                      experiment_name : str, 
                      grid : pd.DataFrame, 
                      n_events : int, 
                      event_duration : float, 
                      min_severity : float, 
                      max_severity : float, 
                      measurements : list,
                      seed : int = 1000
                    ) -> str:
        # set random seed
        random.seed(seed)
        
        # check if measurements list contains more than one measurement
        if len(measurements) < 2: raise ValueError('`measurements` must include more than one sensor')

        # generate events
        events = []
        for _ in tqdm(range(int(n_events * self.sim_duration)), 
                        desc=f'Generating Events for {experiment_name}', 
                        leave=False):
            
            while True:
                # generate start time 
                t_start = self.sim_duration * 24 * 3600 * random.random()
                
                gp_history = set()
                while True:
                    # select a random ground point for this event
                    gp_index = random.randint(0, len(grid)-1)

                    if gp_index in gp_history: continue

                    gp_history.add(gp_index)
                    gp = grid.iloc[gp_index]

                    # check if time overlap exists in the same ground point
                    overlapping_events = [(t_start_overlap,duration_overlap)
                                        for gp_index_overlap,_,_,t_start_overlap,duration_overlap,_,_ in events
                                        if gp_index == gp_index_overlap
                                        and (t_start_overlap <= t_start <= t_start_overlap + duration_overlap
                                        or   t_start <= t_start_overlap <= t_start + event_duration*3600)]
                    
                    # if no overlaps, break random generation cycle
                    if not overlapping_events: break

                    # if all ground points have overlaps at this time, try another start time
                    if len(gp_history) == len(grid): break

                # if no overlaps, break random generation cycle
                if not overlapping_events: break

            # generate severity
            severity = max_severity * random.random() + min_severity

            # generate required measurements        
            n_measurements = random.randint(2,len(measurements)-1)
            required_measurements = random.sample(measurements,k=n_measurements)
            measurements_str = '['
            for req in required_measurements: 
                if required_measurements.index(req) == 0:
                    measurements_str += req
                else:
                    measurements_str += f',{req}'
            measurements_str += ']'
            
            # create event
            event = [
                gp_index,
                gp['lat [deg]'],
                gp['lon [deg]'],
                t_start,
                event_duration * 3600,
                severity,
                measurements_str
            ]

            # add to list of events
            events.append(event)

        # compile list of events
        return pd.DataFrame(data=events, columns=['gp_index','lat [deg]','lon [deg]','start time [s]','duration [s]','severity','measurements'])
    
    def compile_results(self, experiments : pd.DataFrame, results_path : str, print_to_csv : bool = True) -> pd.DataFrame:
        # get run names
        run_names = list({
                          run_name 
                          for run_name in os.listdir(results_path)
                          if os.path.isfile(os.path.join(results_path, run_name, 'summary.csv'))
                          }
                        )

        # define performance metrics
        columns = list(experiments.columns.values)
        
        # organize data
        data = []
        for experiment_name in tqdm(run_names, desc='Compiling Results'):
            # load results summary
            summary_path = os.path.join(results_path, experiment_name, 'summary.csv')
            summary : pd.DataFrame = pd.read_csv(summary_path)

            # get experiment parameters
            matching_experiment = [[name,*_] for name,*_ in experiments.values if name==experiment_name]
            datum = matching_experiment.pop()
            
            # collect xperiment results
            for key,value in summary.values:
                key : str; value : str

                if key not in columns: 
                    columns.append(key)           
                    # ys.append(key)

                if not isinstance(value, float):
                    try:
                        value = float(value)  
                    except ValueError:
                        pass
            
                datum.append(value)

            # add to data
            data.append(datum)

        # create compiled data frame
        results = pd.DataFrame(data=data, columns=columns)
        results.sort_values('Name')

        results['Percent Ground Points Observed'] = results['Ground Points Observed'] / results['Ground Points']
        results['Percent Ground Points Accessible'] = results['Ground Points Accessible'] / results['Ground Points']
        results['Percent Events Detected'] = results['Events Detected'] / results["Number of Events per Day"]
        results['Percent Events Observed'] = results['Events Observed'] / results["Number of Events per Day"]
        results['Percent Events Re-observed'] = results['Events Re-observed'] / results["Number of Events per Day"]
        results['Percent Events Co-observed'] = results['Events Co-observed'] / results["Number of Events per Day"]
        results['Percent Events Fully Co-observed'] = results['Events Fully Co-observed'] / results["Number of Events per Day"]
        results['Percent Events Partially Co-observed'] = results['Events Partially Co-observed'] / results["Number of Events per Day"]
        results['Percent Events Observed'] = results['Events Observed'] / results["Number of Events per Day"]
        results['Ground-Points Considered'] = results['Percent Ground-Points Considered'] * results['Number of Ground-Points']
        results['Number of Satellites'] = results['Number Planes'] * results['Number of Satellites per Plane']

        # save to csv
        if print_to_csv: results.to_csv(os.path.join(results_path, 'study_results_compiled.csv'),index=False)

        # return results dataframe
        return results