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

FILTER_RESULTS_DIRNAME = 'filter_results_gemini'


def combine_filtered_tabs():
    # Dicts for stats, tabs, and tabs_wide
    stat_dfs = {0: [], 1: [], 2: []}
    tabs = {0: [], 1: [], 2: []}
    tabs_wide = {1: [], 2: []}

    # Combine everything
    fnames = os.listdir(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME))
    for cat in (0, 1, 2):
        for field_name in fnames:
            if field_name in ('combined', 'candidates', '.DS_Store'):
                continue

            # Load the stats
            stat_df = pd.read_csv(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, field_name, f'{cat}_filter_stats.csv'))
            stat_df['branch'] = stat_df['branch'].fillna('')
            stat_dfs[cat].append(stat_df)

            # Load the tabs
            if cat != 2:  # TODO: temporary for bad pstarr extraction
                tabs[cat].append(load_ecsv(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, field_name, f'{cat}.ecsv')))

    # Deal with wide associations
    for cat in (1, 2):
        for field_name in fnames:
            if field_name in ('combined', 'candidates', '.DS_Store'):
                continue

            # Load the tabs_wide
            if cat != 2:  # TODO: temporary for bad pstarr extraction
                tabs_wide[cat].append(load_ecsv(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, field_name, f'{cat}_wide_association.ecsv')))

    # Drop length 0 tabs
    for cat in (0, 1, 2):
        tabs[cat] = [tab for tab in tabs[cat] if len(tab) > 0]
    for cat in (1, 2):
        tabs_wide[cat] = [tab for tab in tabs_wide[cat] if len(tab) > 0]

    # Combine the stats
    stat_dfs = {cat: combine_stats(stat_dfs[cat]) for cat in (0, 1, 2)}

    # Combine the tabs
    for cat in (0, 1):  # TODO: add 2 back
        for tab in tabs[cat]:
            if 'filter_info' in tab.colnames:
                tab['filter_info'] = tab['filter_info'].astype(str)
            if 'Catalog' in tab.colnames:
                tab['Catalog'] = tab['Catalog'].astype(str)
        if len(tabs[cat]) > 0:
            tabs[cat] = vstack(tabs[cat])
        else:
            tabs[cat] = Table()

    # Do the same casting for tabs_wide
    for cat in (1,):  # TODO: add 2 back
        for tab in tabs_wide[cat]:
            if 'filter_info' in tab.colnames:
                tab['filter_info'] = tab['filter_info'].astype(str)
            if 'Catalog' in tab.colnames:
                tab['Catalog'] = tab['Catalog'].astype(str)
        if len(tabs_wide[cat]) > 0:
            tabs_wide[cat] = vstack(tabs_wide[cat])
        else:
            tabs_wide[cat] = Table()

    # Save everything to a new directory
    if not os.path.exists(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, 'combined')):
        os.makedirs(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, 'combined'))
    for cat in (0, 1, 2):
        stat_dfs[cat].to_csv(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, 'combined', f'{cat}_filter_stats.csv'))
        if cat != 2:  # TODO: temporary for bad pstarr extraction
            tabs[cat].write(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, 'combined', f'{cat}.ecsv'), overwrite=True)

    for cat in (1, 2):
        if cat != 2:  # TODO: temporary for bad pstarr extraction
            tabs_wide[cat].write(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, 'combined', f'{cat}_wide_association.ecsv'), overwrite=True)

    for cat in (0, 1, 2):
        create_filter_flowchart(stat_dfs[cat]).save(f'/Users/adamboesky/Research/long_transients/Data/{FILTER_RESULTS_DIRNAME}/combined/{cat}_flowchart.pdf')


if __name__ == '__main__':
    combine_filtered_tabs()
