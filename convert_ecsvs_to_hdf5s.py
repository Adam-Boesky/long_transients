import os
import numpy as np

from astropy.table import MaskedColumn
from Extracting.utils import load_ecsv


def convert_directory(directory: str):
    for file in os.listdir(directory):
        if file.endswith('.ecsv'):

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

            table.write(os.path.join(directory, file.replace('.ecsv', '.hdf5')), path='data', serialize_meta=True, overwrite=True)

if __name__ == '__main__':
    convert_directory('/Users/adamboesky/Research/long_transients/Data/debugging')
