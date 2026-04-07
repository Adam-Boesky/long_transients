import os
import numpy as np

from astropy.table import MaskedColumn
from Extracting.utils import load_ecsv


def convert_directory(directory: str, depth: int = 1):
    if depth == 0:
        return

    # Recursively convert subdirectories
    for subdir in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, subdir)):
            convert_directory(os.path.join(directory, subdir), depth - 1)

    # Convert files
    for file in os.listdir(directory):
        if file.endswith('.ecsv'):

            hdf5_path = os.path.join(directory, file.replace('.ecsv', '.hdf5'))

            if os.path.exists(hdf5_path):
                print(f'{hdf5_path} already exists. Skipping...')
                continue

            print(f'Converting {file}...')
            table = load_ecsv(os.path.join(directory, file))

            for col in table.colnames:
                if col == 'PSTARR_PanSTARR_ID':
                    data = np.array(table[col])
                    mask = np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in data])
                    fill = -1
                    values = np.array([v if not (v is None or (isinstance(v, float) and np.isnan(v))) else fill for v in data], dtype=np.int64)                                                                       
                    table[col] = MaskedColumn(values, mask=mask)   
                    print(col, type(table[col][0]))

            table.write(hdf5_path, path='data', serialize_meta=True, overwrite=True)

if __name__ == '__main__':
    convert_directory('/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/catalog_results/field_results')
