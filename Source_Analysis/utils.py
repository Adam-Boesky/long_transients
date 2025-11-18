import os
import sys
import numpy as np

from typing import Literal, List
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
from astropy.table import Table, vstack, join

sys.path.append('/Users/adamboesky/Research/long_transients')

from Extracting.utils import load_ecsv, get_data_path
from Source_Analysis.filter_fields import Filters


def get_sources_filtered_by(
    filter_name: str,
    catalogs: List[Literal['in_both', 'in_ztf', 'in_pstarr']],
    in_bands: List[Literal['g', 'r', 'i']],
    filter_dirpath: str = '/Users/adamboesky/Research/long_transients/Data/filter_results',
):
    """Get the sources filtered by a given filter name and catalog."""

    # Get the fields in the filtered directory
    fields = os.listdir(filter_dirpath)
    fields.remove('combined')
    fields.remove('.DS_Store')

    # Set up sources with empty tables to fill in
    empty_coord_table = Table(data={'ra':[], 'dec':[], 'filter':[]}, dtype=['float64', 'float64', 'str'])
    srcs = {catalog: empty_coord_table.copy() for catalog in catalogs}
    catalog_map = {'in_both': 0, 'in_ztf': 1, 'in_pstarr': 2}

    # Loop through fields, catalogs, and bands to fill in the sources
    for field in fields:
        print(f'Checking field: {field}')
        for catalog in catalogs:

            # Get sources that are in the filtered out table for all required bands
            filtered_out_tab = load_ecsv(
                    f'{filter_dirpath}/{field}/{catalog_map[catalog]}_{in_bands[0]}_filtered_out.ecsv'
                )
            for band in in_bands[1:]:
                filtered_out_tab_band = load_ecsv(
                    f'{filter_dirpath}/{field}/{catalog_map[catalog]}_{band}_filtered_out.ecsv'
                )
                if len(filtered_out_tab) == 0 or len(filtered_out_tab_band) == 0:
                    filtered_out_tab = empty_coord_table.copy()
                else:
                    filtered_out_tab = join(filtered_out_tab, filtered_out_tab_band, join_type='inner', keys=['ra', 'dec', 'filter'])

            # Get the sources with the desired filter
            filter_mask = filtered_out_tab['filter'] == filter_name
            srcs[catalog] = vstack([srcs[catalog], filtered_out_tab[filter_mask]])

    if len(catalogs) == 1:
        return srcs[catalogs[0]]
    return srcs


def get_kde(fields: List[int], band: str, return_data: bool = False, allow_missing: bool = False) -> gaussian_kde:
    """Get the fitted kde object for given band and list of given fields."""
    fields = [str(int(f)).zfill(6) for f in fields]

    # Set up filters and data stuff
    filters = Filters()
    data_path = get_data_path()
    # Collect arrays in lists to avoid O(n²) concatenation operations
    x_arrays, y_arrays = [], []

    print('Loading and filtering data...')
    for i, field in enumerate(fields):

        print(f'Loading field {i+1} / {len(fields)}...')
        field_tab_path = os.path.join(data_path, 'catalog_results', 'field_results', f'{field}_{band}.ecsv')
        if os.path.exists(field_tab_path):
            field_tab = load_ecsv(field_tab_path)
        else:
            if allow_missing:
                print(f'WARNING: Field {field} not found at {field_tab_path}. Skipping...')
                continue
            else:
                raise FileNotFoundError(f'Field {field} not found in {field_tab_path}')

        # Only use sources in both catalogs
        field_tab = field_tab[field_tab['Catalog_Flag'] == 0]

        # Drop nan rows
        mag_data = field_tab[[f'PSTARR_{band}PSFMag', f'ZTF_{band}PSFMag']].as_array().view(np.float64).reshape(len(field_tab), -1)
        nan_mask = ~np.any(np.isnan(mag_data), axis=1)
        field_tab = field_tab[nan_mask]

        # Apply filters
        temp_dict = {band: field_tab}  # dict needed for filter sytax
        filt_names = ['sep_extraction_filter', 'snr_filter', 'shape_filter', 'psf_fit_filter', 'pstarr_not_saturated']
        for filt_name in filt_names:
            temp_dict = filters.filter(temp_dict, filt_name)
        field_tab = temp_dict[band]

        # Extract columns and store arrays (much more memory efficient than concatenation)
        pstarr_mag = np.asarray(field_tab[f'PSTARR_{band}PSFMag'], dtype=np.float64)
        ztf_mag = np.asarray(field_tab[f'ZTF_{band}PSFMag'], dtype=np.float64)

        # Store arrays in lists (will concatenate once at the end)
        x_arrays.append(pstarr_mag)
        y_arrays.append(ztf_mag - pstarr_mag)

    print('Calculating the KDE...')
    # Concatenate all arrays only once at the end (O(n) instead of O(n²))
    x = np.concatenate(x_arrays) if x_arrays else np.array([], dtype=np.float64)
    y = np.concatenate(y_arrays) if y_arrays else np.array([], dtype=np.float64)
    kde = gaussian_kde(np.vstack([x, y]))

    if return_data:
        return kde, x, y
    return kde
