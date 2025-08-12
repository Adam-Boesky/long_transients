import os
import sys

from typing import Literal, List
from astropy.table import Table, vstack, join

sys.path.append('/Users/adamboesky/Research/long_transients')

from Extracting.utils import load_ecsv


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
