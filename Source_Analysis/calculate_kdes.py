import sys
import os
import pickle
import ztffields
import numpy as np

from ztffields.fields import get_fieldid

sys.path.append('/Users/adamboesky/Research/long_transients')
sys.path.append('/n/home04/aboesky/berger/long_transients')

from Source_Analysis.utils import get_kde, get_data_path


def calculate_kdes():

    # Get the fields that we want KDEs for
    # imaged_fields = {}
    data_path = get_data_path()
    # for band in ('g', 'r', 'i'):
    #     imaged_fields[band] = np.load(os.path.join(data_path, f'{band}_imaged_fields.npy'))
    # fs = np.intersect1d(get_fieldid(grid='main', dec_range=[-100, -5], ra_range=[75, 100]), imaged_fields['g'])
    fs = [571, 572, 573, 616, 617, 618, 619, 620, 621, 622]
    print(f'Calculating KDEs for fields (n = {len(fs)}): {fs}')

    # Calculate the KDEs
    kdes = {}
    for band in ('g', 'r', 'i'):
        kdes[band] = get_kde(fs, band, allow_missing=True)

    # Save the KDEs
    for band, kde in kdes.items():
        with open(os.path.join(data_path, f'{band}_kde.pkl'), 'wb') as f:
            pickle.dump(kde, f)


if __name__ == '__main__':
    calculate_kdes()
