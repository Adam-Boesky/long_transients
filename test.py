from Extracting.utils import get_data_path, load_ecsv

import numpy as np

def check():
    pstarr_tab = load_ecsv('/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/catalog_results/000373_01_3/PSTARR.hdf5')
    print(pstarr_tab[np.isclose(pstarr_tab['ra'], 204.87893) & np.isclose(pstarr_tab['dec'], -12.79083)][['gPSFMag', 'rPSFMag', 'iPSFMag']])

    merged_tab = load_ecsv('/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/catalog_results/000373_01_3/r_associated.hdf5')
    row = merged_tab[np.isclose(merged_tab['ra'], 204.87893) & np.isclose(merged_tab['dec'], -12.79083)]
    for col in row.colnames:
        print(f'{col}: {row[col][0]}')

if __name__ == '__main__':
    check()
