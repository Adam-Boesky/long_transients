import os
import re
import numpy as np

from Extracting.utils import load_ecsv, prepare_table_for_write


def convert_directory(directory: str, depth: int = 1, regex: str = ''):
    if depth == 0:
        return

    # Recursively convert subdirectories
    for subdir in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, subdir)):
            convert_directory(os.path.join(directory, subdir), depth - 1, regex=regex)

    # Convert files
    for file in os.listdir(directory):
        if file.endswith('.ecsv') and re.match(regex, file):

            ecsv_path = os.path.join(directory, file)
            hdf5_path = os.path.join(directory, file.replace('.ecsv', '.hdf5'))

            if os.path.exists(hdf5_path):
                print(f'{hdf5_path} already exists. Skipping...')
                continue

            print(f'Converting {ecsv_path}...')
            table = prepare_table_for_write(load_ecsv(ecsv_path))
            table.write(hdf5_path, path='data', serialize_meta=True, overwrite=True)

if __name__ == '__main__':
    # convert_directory('/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/catalog_results/field_results')
    convert_directory('/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/catalog_results', depth=2)
