import os
import random
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
from tqdm import tqdm


def main(
         n_points : int,
         grid_type : str,
         overwrite : bool = True,
         plot : bool = False,
         seed : int = 1000
         ) -> None:
        
    # generate grids
    if grid_type.lower() == 'uniform':
        grid_path = create_uniform_grid(n_points, overwrite)
    
    elif grid_type.lower() == 'clustered':
        raise NotImplementedError(f'Cannot generate grid of type `{grid_type}`. Type not yet supported.')
    
    elif grid_type.lower() == 'fibonacci':
        grid_path = create_fibonacci_grid(n_points, overwrite)
    
    elif grid_type.lower() == 'hydrolakes':
        grid_path = sample_hyrolakes(n_points, overwrite, seed)
    
    else:
        raise ValueError(f'Cannot generate grid of type `{grid_type}`. Type not supported.')
    
    # plot grids
    if plot: plot_grid(grid_path, grid_type, n_points, overwrite)
        
        
def create_uniform_grid(n_points : int,
                        overwrite : bool
                        ) -> str:
    # set grid name
    grid_path : str = os.path.join('grids', f'uniform_grid_{n_points}.csv')
    
    # check if grid already exists
    if os.path.isfile(grid_path) and not overwrite: return grid_path

    # calculate spacing
    k_1 = (1/2) * (1 - np.sqrt( 2*n_points - 3 ))
    k_2 = (1/2) * (np.sqrt( 2*n_points - 3 ) + 1)
    k = np.floor(max(k_1,k_2))

    spacing = 180/k # deg / plane
    n_meridians = 2*k
    n_parallels = k - 1

    # generate grid
    groundpoints = [[lat, lon] 
                    for lat in np.linspace(-90, 90, int(180/spacing)+1)
                    for lon in np.linspace(-180, 180, int(360/spacing)+1)
                    if lon < 180
                    ]

    assert len(groundpoints) >= n_points

    # create dataframe
    df = pd.DataFrame(data=groundpoints, columns=['lat [deg]','lon [deg]'])

    # save to csv
    df.to_csv(grid_path,index=False)

    # return address
    return grid_path

def create_fibonacci_grid(n_points : int,
                          overwrite : bool
                          ) -> str:
    # set grid name
    grid_path : str = os.path.join('grids', f'fibonacci_grid_{n_points}.csv')
    
    # check if grid already exists
    if os.path.isfile(grid_path) and not overwrite: return grid_path

    golden_ratio = (1 + np.sqrt(5)) / 2
    N = int(np.round((n_points - 1) / 2))
    Ns = [i for i in range(-N,N+1)]

    groundpoints = []
    for i in Ns:
        lat = np.arcsin( 2*i / (2*N + 1) ) * 180 / np.pi
        lon = np.mod(i,golden_ratio) * 360 / golden_ratio

        if lon < -180:
            lon += 360
        if lon > 180:
            lon -= 360

        groundpoints.append([lat,lon])

    # create dataframe
    df = pd.DataFrame(data=groundpoints, columns=['lat [deg]','lon [deg]'])

    # save to csv
    df.to_csv(grid_path,index=False)

    # return address
    return grid_path

def sample_hyrolakes(n_points : int,
                     overwrite : bool,
                     seed : int,
                    ) -> str:
    # set random seed
    random.seed(seed)

    # set grid name
    grid_path : str = os.path.join('grids', f'hydrolakes_grid_{n_points}_seed-{seed}.csv')
    
    # check if grid already exists
    if os.path.isfile(grid_path) and not overwrite: return grid_path

    # load original hydrolakes dataset
    original_grid_path = os.path.join('grids', f'hydrolakes_dataset.csv')
    hydrolakes : pd.DataFrame = pd.read_csv(original_grid_path)

    # check inputs
    assert n_points <= len(hydrolakes)

    # collect groundpoints
    all_groundpoints = [[lat,lon] for lat,lon in hydrolakes.values]
    groundpoints = random.sample(all_groundpoints, n_points)

    # create dataframe
    df = pd.DataFrame(data=groundpoints, columns=['lat [deg]','lon [deg]'])

    # save to csv
    df.to_csv(grid_path,index=False)

    # return address
    return grid_path

# def create_clustered_grid(n_points : int, 
#                           overwrite : bool,
#                           seed : int,
#                           n_clusters : float = 100, 
#                           variance : float = 1.0
#                           ) -> str:
    
    # # set random seed
    # random.seed(seed)
    
#     # set grid name
#     grid_path : str = os.path.join('grids', f'clustered_grid_{n_points}_seed-{seed}.csv')
    
#     # check if grid already exists
#     if os.path.isfile(grid_path) and not overwrite: return grid_path
    
#     # generate cluster grid
#     all_clusters = [[lat, lon ]
#                     for lat in np.linspace(-90, 90, int(180/spacing)+1)
#                     for lon in np.linspace(-180, 180, int(360/spacing)+1)
#                     if lon < 180
#                     ]
#     clusters : list = random.sample(all_clusters, n_clusters)
#     clusters.sort()

#     # create clustered grid of gound points
#     std = np.sqrt(variance)
#     groundpoints = []

#     for lat_cluster,lon_cluster in tqdm.tqdm(clusters, desc='generating clustered grid', leave=False):
#         for _ in range(int(n_points / n_clusters)):
#             # sample groundpoint
#             lat = random.normalvariate(lat_cluster, std)
#             lon = random.normalvariate(lon_cluster, std)
#             groundpoints.append((lat,lon))

#     assert len(groundpoints) == n_points

#     # create datagrame
#     df = pd.DataFrame(data=groundpoints, columns=['lat [deg]','lon [deg]'])

#     # save to csv
#     df.to_csv(grid_path,index=False)

#     # return address
#     return grid_path

def plot_grid(grid_path : str, grid_type : str, n_points : int, overwrite : bool) -> None:
    # get plot path
    plot_path = grid_path.replace('.csv', '.png')

    # check if plot already exists
    if os.path.isfile(plot_path) and not overwrite: return 

    # load grid data
    df : pd.DataFrame = pd.read_csv(grid_path)
    lons = [lon for _,lon in df.values]
    lats = [lat for lat,_ in df.values]

    # generate plot
    m = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='l')
    x, y = m(lons,lats)
    m.drawmapboundary(fill_color='#99ffff')
    m.fillcontinents(color='#cc9966',lake_color='#99ffff')
    m.scatter(x,y,3,marker='o',color='k')

    # set title
    plt.title(f"{grid_type} grid of ~{n_points} points")

    # save plot
    plt.savefig(plot_path)

    # close plot
    plt.close()


if __name__ == "__main__":
    # set seed
    seed = 1000

    # load experiments
    experiments_path = os.path.join('experiments', f'experiments_seed-{seed}.csv')
    experiments : pd.DataFrame = pd.read_csv(experiments_path)

    grid_types : list = experiments['Grid Type'].unique(); grid_types.sort()
    points : list = experiments['Number of Grid-points'].unique(); points.sort()

    # plot original hydrolakes database
    plot_grid('./grids/hydrolakes_dataset.csv', 'hydrolakes', 5000, overwrite=True)

    # generate grids and plots for all types and number of groundpoints
    with tqdm(range(len(grid_types) * len(points)), desc='Generating coverage grids') as pbar:
        for grid_type in grid_types:
            for n_points in points:
                
                if grid_type == 'hydrolakes' and n_points > 5000:
                    pbar.update(1)
                    continue

                main(n_points, grid_type, overwrite=False, plot=True)
                pbar.update(1)