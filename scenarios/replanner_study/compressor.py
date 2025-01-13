import os
import shutil

from tqdm import tqdm

def main(results_path : str = './results'):
    """ Removes unnecessary data from results to save hard drive space. """    

    for experiment in tqdm(os.listdir(results_path), desc='Clearing runtime stats from results'):
        experiment_path : str = os.path.join(results_path, experiment)

        for sim_element_name in os.listdir(experiment_path):
            runtime_path = os.path.join(experiment_path, sim_element_name, 'runtime')

            if not os.path.isdir(runtime_path): continue

            for filename in os.listdir(runtime_path):
                file_path = os.path.join(runtime_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

            os.rmdir(runtime_path)

    print('DONE')

if __name__ == '__main__':
    main()