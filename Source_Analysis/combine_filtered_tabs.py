import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, match_coordinates_sky
from typing import Iterable

sys.path.append('/Users/adamboesky/Research/long_transients')

from Extracting.utils import load_ecsv, get_data_path
from Source_Analysis import filtered_out_summary
from Source_Analysis.filter_fields import combine_stats, create_filter_flowchart

FILTER_RESULTS_DIRNAME = 'filter_results_7_30_2026_gaia_propagation'


def _field_reasons(field_name: str, run_dirname: str, max_arcsec: float):
    """Worker: the nine filter-reason columns for one field, per catalog.

    Runs in a subprocess because the cost is reading ~535 MB of reject tables
    per field; the crossmatch itself is a fraction of a second. Returns
    `{cat: {column: values}}`, or `{}` when the field has no survivors.
    """
    base = os.path.join(get_data_path(), run_dirname, field_name)
    out = {}
    for cat in (0, 1):   # TODO: add 2 back with pstarr extraction
        path = os.path.join(base, f'{cat}.ecsv')
        # `load_ecsv` silently prefers a .hdf5 sibling, and the cluster runs
        # write only .hdf5 -- so existence must be checked on both, or every
        # field is skipped and the columns come back empty.
        if not os.path.exists(path) and not os.path.exists(path[:-5] + '.hdf5'):
            continue
        tab = load_ecsv(path)
        if len(tab) == 0:
            continue
        out[cat] = filtered_out_summary.reasons_for(
            tab['ra'], tab['dec'], base, max_arcsec)
    return field_name, out


def combine_filtered_tabs(add_filtered_out: bool = False,
                          processes: int = 1,
                          max_arcsec: float = filtered_out_summary.DEFAULT_MATCH_ARCSEC,
                          out_subdir: str = 'combined'):
    """Combine the per-field outputs into `combined/`.

    With `add_filtered_out`, also reduce the per-field reject tables to nine
    `filtered_out_{catalog}_{band}` columns per surviving source. That has to
    happen here: the reject tables are ~200 GB across a run, live only on the
    cluster, and are deleted once this script has produced `combined/`. Nine
    short strings per source survive instead, and they ride through
    `enrich_combined_tabs` into the database with no further plumbing.
    """
    # Dicts for stats, tabs, and tabs_wide
    stat_dfs = {0: [], 1: [], 2: []}
    tabs = {0: [], 1: [], 2: []}
    tabs_wide = {1: [], 2: []}
    # Which field each entry of tabs[cat] came from, so the filtered-out
    # columns are attached by name rather than by relying on list order.
    tab_fields = {0: [], 1: [], 2: []}

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
                tab_fields[cat].append(field_name)

    # Deal with wide associations
    for cat in (1, 2):
        for field_name in fnames:
            if field_name in ('combined', 'candidates', '.DS_Store'):
                continue

            # Load the tabs_wide
            if cat != 2:  # TODO: temporary for bad pstarr extraction
                tabs_wide[cat].append(load_ecsv(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, field_name, f'{cat}_wide_association.ecsv')))

    # Filter reasons, computed per field while the reject tables are still
    # next to the survivors. Attached before the vstack so the columns simply
    # come along.
    if add_filtered_out:
        field_names = [f for f in fnames
                       if f not in ('combined', 'candidates', '.DS_Store')]
        print(f'Summarising filtered-out reasons for {len(field_names)} fields '
              f'on {processes} process(es)...')
        worker = partial(_field_reasons, run_dirname=FILTER_RESULTS_DIRNAME,
                         max_arcsec=max_arcsec)
        if processes > 1:
            with Pool(processes) as pool:
                results = pool.map(worker, field_names)
        else:
            results = [worker(f) for f in field_names]
        by_field = dict(results)

        for cat in (0, 1):
            assert len(tab_fields[cat]) == len(tabs[cat])
            for field_name, tab in zip(tab_fields[cat], tabs[cat]):
                cols = by_field.get(field_name, {}).get(cat)
                if cols is None:
                    cols = {c: np.full(len(tab), filtered_out_summary.NO_TABLE)
                            for c in filtered_out_summary.ALL_COLUMNS}
                for name, values in cols.items():
                    tab[name] = np.asarray(values, dtype=str)

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

    # Save everything to a new directory. `out_subdir` lets a validation run
    # write alongside an existing `combined/` instead of overwriting it.
    if not os.path.exists(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, out_subdir)):
        os.makedirs(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, out_subdir))
    for cat in (0, 1, 2):
        stat_dfs[cat].to_csv(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, out_subdir, f'{cat}_filter_stats.csv'))
        if cat != 2:  # TODO: temporary for bad pstarr extraction
            tabs[cat].write(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, out_subdir, f'{cat}.ecsv'), overwrite=True)

    for cat in (1, 2):
        if cat != 2:  # TODO: temporary for bad pstarr extraction
            tabs_wide[cat].write(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, out_subdir, f'{cat}_wide_association.ecsv'), overwrite=True)

    for cat in (0, 1, 2):
        # get_data_path() rather than a hardcoded local path: this script runs
        # on the cluster, where that directory does not exist.
        create_filter_flowchart(stat_dfs[cat]).save(
            os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, out_subdir,
                         f'{cat}_flowchart.pdf'))


def main():
    ap = argparse.ArgumentParser(description=combine_filtered_tabs.__doc__.splitlines()[0])
    ap.add_argument('--filtered-out', action='store_true',
                    help='also reduce the per-field reject tables to nine '
                         'filter-reason columns (adds ~1 min/field serially; '
                         'use --processes)')
    ap.add_argument('--processes', type=int, default=1,
                    help='parallelism for the filtered-out pass; the field '
                         'loop is independent so this scales nearly linearly')
    ap.add_argument('--out-subdir', default='combined',
                    help="output directory under the run (default 'combined'); "
                         'set it to write a validation run alongside the real one')
    ap.add_argument('--max-arcsec', type=float,
                    default=filtered_out_summary.DEFAULT_MATCH_ARCSEC,
                    help='match radius, matching Source.max_arcsec')
    a = ap.parse_args()
    combine_filtered_tabs(add_filtered_out=a.filtered_out,
                          processes=a.processes, max_arcsec=a.max_arcsec,
                          out_subdir=a.out_subdir)


if __name__ == '__main__':
    main()
