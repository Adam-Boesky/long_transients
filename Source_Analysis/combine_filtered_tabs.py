import os
import sys

import numpy as np
import pandas as pd

from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, match_coordinates_sky
from typing import Iterable

sys.path.append('/Users/adamboesky/Research/long_transients')

from Extracting.utils import load_ecsv, get_data_path
from Source_Analysis.filter_fields import combine_stats, create_filter_flowchart


def combine_filtered_tabs():
    # Dicts for stats, tabs, and tabs_wide
    stat_dfs = {0: [], 1: [], 2: []}
    tabs = {0: [], 1: [], 2: []}
    tabs_wide = {1: [], 2: []}

    # Combine everything
    fnames = os.listdir(os.path.join(get_data_path(), 'filter_results'))
    for cat in (0, 1, 2):
        for field_name in fnames:
            if field_name in ('combined', 'candidates', '.DS_Store'):
                continue

            # Load the stats
            stat_df = pd.read_csv(os.path.join(get_data_path(), 'filter_results', field_name, f'{cat}_filter_stats.csv'))
            stat_df['branch'] = stat_df['branch'].fillna('')
            stat_dfs[cat].append(stat_df)

            # Load the tabs
            if cat != 2:  # TODO: temporary for bad pstarr extraction
                tabs[cat].append(load_ecsv(os.path.join(get_data_path(), 'filter_results', field_name, f'{cat}.ecsv')))

    # Deal with wide associations
    for cat in (1, 2):
        for field_name in fnames:
            if field_name in ('combined', 'candidates', '.DS_Store'):
                continue

            # Load the tabs_wide
            if cat != 2:  # TODO: temporary for bad pstarr extraction
                tabs_wide[cat].append(load_ecsv(os.path.join(get_data_path(), 'filter_results', field_name, f'{cat}_wide_association.ecsv')))

    # Combine the stats
    stat_dfs = {cat: combine_stats(stat_dfs[cat]) for cat in (0, 1, 2)}

    # Combine the tabs
    for cat in (0, 1):  # TODO: add 2 back
        for tab in tabs[cat]:
            if 'filter_info' in tab.colnames:
                tab['filter_info'] = tab['filter_info'].astype(str)
            if 'Catalog' in tab.colnames:
                tab['Catalog'] = tab['Catalog'].astype(str)
        tabs[cat] = vstack(tabs[cat])

    # Do the same casting for tabs_wide
    for cat in (1,):  # TODO: add 2 back
        for tab in tabs_wide[cat]:
            if 'filter_info' in tab.colnames:
                tab['filter_info'] = tab['filter_info'].astype(str)
            if 'Catalog' in tab.colnames:
                tab['Catalog'] = tab['Catalog'].astype(str)
    tabs_wide = {cat: vstack(tabs_wide[cat]) for cat in (1, 2) if cat != 2}  # TODO: add 2 back

    # Save everything to a new directory
    for cat in (0, 1, 2):
        stat_dfs[cat].to_csv(os.path.join(get_data_path(), 'filter_results', 'combined', f'{cat}_filter_stats.csv'))
        if cat != 2:  # TODO: temporary for bad pstarr extraction
            tabs[cat].write(os.path.join(get_data_path(), 'filter_results', 'combined', f'{cat}.ecsv'), overwrite=True)

    for cat in (1, 2):
        if cat != 2:  # TODO: temporary for bad pstarr extraction
            tabs_wide[cat].write(os.path.join(get_data_path(), 'filter_results', 'combined', f'{cat}_wide_association.ecsv'), overwrite=True)

    for cat in (0, 1, 2):
        create_filter_flowchart(stat_dfs[cat]).save(f'/Users/adamboesky/Research/long_transients/Data/filter_results/combined/{cat}_flowchart.pdf')


if __name__ == '__main__':
    combine_filtered_tabs()
