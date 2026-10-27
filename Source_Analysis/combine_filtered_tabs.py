import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, match_coordinates_sky, search_around_sky
import astropy.units as u
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, on any machine

from Extracting.utils import load_ecsv, get_data_path, prepare_table_for_write
from Source_Analysis import filtered_out_summary
from Source_Analysis.filter_fields import combine_stats, create_filter_flowchart

FILTER_RESULTS_DIRNAME = os.environ.get('FILTER_RESULT_DIR', 'filter_results_final_10_22_2026')


def _field_reasons(field_name: str, run_dirname: str, max_arcsec: float):
    """Worker: the nine filter-reason columns for one field, per catalog.

    Runs in a subprocess because the cost is reading ~535 MB of reject tables
    per field; the crossmatch itself is a fraction of a second. Returns
    `{cat: {column: values}}`, or `{}` when the field has no survivors.
    """
    base = os.path.join(get_data_path(), run_dirname, field_name)
    out = {}
    # Both the ordinary survivors and the wide-association ones: they end up
    # in the same database table, so they must carry the same columns.
    targets = [(cat, f'{cat}.ecsv') for cat in (0, 1)]
    # targets += [(f'wide{cat}', f'{cat}_wide_association.ecsv') for cat in (1,)]  # wide-association tables disabled (#4)
    for key, fname in targets:
        path = os.path.join(base, fname)
        # `load_ecsv` silently prefers a .hdf5 sibling, and the cluster runs
        # write only .hdf5 -- so existence must be checked on both, or every
        # field is skipped and the columns come back empty.
        if not os.path.exists(path) and not os.path.exists(path[:-5] + '.hdf5'):
            continue
        tab = load_ecsv(path)
        if len(tab) == 0:
            continue
        out[key] = filtered_out_summary.reasons_for(
            tab['ra'], tab['dec'], base, max_arcsec)
    return field_name, out


GROUP_RADIUS_ARCSEC = 1.0


def add_group_ids(tables: Iterable[Table], radius_arcsec: float = GROUP_RADIUS_ARCSEC) -> None:
    """Link rows within radius_arcsec of each other, across fields and across tables (plan_filter_review.md #1).

    Overlapping fields find the same object more than once. Linked rows (chains included) share an integer
    `group_id` (1, 2, ..., numbered by the group's mean RA); a row with no partner gets -1. Rows are never
    dropped or merged.
    """
    tables = [t for t in tables if len(t) > 0]
    if not tables:
        return
    ra = np.concatenate([np.asarray(t['ra'], dtype=float) for t in tables])
    dec = np.concatenate([np.asarray(t['dec'], dtype=float) for t in tables])
    coords = SkyCoord(ra, dec, unit='deg')
    i, j, _, _ = search_around_sky(coords, coords, radius_arcsec * u.arcsec)

    # Union-find over the pairs, so chains A-B-C end up in one group
    parent = np.arange(len(ra))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(i[i < j], j[i < j]):
        ra_, rb_ = find(a), find(b)
        if ra_ != rb_:
            parent[max(ra_, rb_)] = min(ra_, rb_)
    roots = np.array([find(x) for x in range(len(ra))])

    group_ids = np.full(len(ra), -1, dtype=np.int64)
    uniq, inv, counts = np.unique(roots, return_inverse=True, return_counts=True)
    xyz = coords.cartesian.xyz.value.T  # average on the unit sphere, safe across RA=0
    sums = np.zeros((len(uniq), 3))
    np.add.at(sums, inv, xyz)
    multi = np.where(counts > 1)[0]
    if len(multi):
        # Number groups by mean RA (then Dec) so the ids don't depend on file order
        mean = SkyCoord(*sums[multi].T, representation_type='cartesian').spherical
        order = np.lexsort((mean.lat.deg, mean.lon.deg))
        number = np.zeros(len(uniq), dtype=np.int64)
        number[multi[order]] = np.arange(1, len(multi) + 1)
        in_group = counts[inv] > 1
        group_ids[in_group] = number[inv[in_group]]

    start = 0
    for t in tables:
        t['group_id'] = group_ids[start:start + len(t)]
        start += len(t)


BAND_SETS = ('gri', 'gr', 'gi', 'ri')
MERGE_STEP = 'merge_duplicates'


def _band_sets(tab: Table) -> np.ndarray:
    """Each row's detected band set ('gri', 'gr', ...), as the filter stats count it (Source.in_g etc.)."""
    det = {b: np.isfinite(np.asarray(tab[f'ZTF_{b}_ra'], dtype=float)) for b in 'gri'}
    return np.array([''.join(b for b in 'gri' if det[b][k]) for k in range(len(tab))])


def add_merge_duplicates_step(stats: pd.DataFrame, tab: Table) -> pd.DataFrame:
    """Append a final flowchart step collapsing each `group_id` to one object.

    Every earlier step is summed over fields, so objects in overlap regions count once per field. This
    step's counts are unique objects; a group whose detections differ in band set takes the one with the
    most bands. Replaces an existing step, so it can be re-run on a combined stats table.
    """
    if set(stats.index.names) != {'filter', 'branch'}:  # combine_stats leaves a single field ungrouped
        stats = stats.groupby(['filter', 'branch']).sum()
    stats = stats.drop(index=MERGE_STEP, level='filter', errors='ignore')
    sets = _band_sets(tab) if len(tab) else np.array([], dtype=str)
    gid = np.asarray(tab['group_id']) if len(tab) else np.array([], dtype=np.int64)
    # Most bands first (gri), then file order; keep the first row of each group
    order = np.lexsort((np.arange(len(sets)), -np.char.str_len(sets.astype(str))))
    dup = pd.Series(gid[order]).duplicated().to_numpy() & (gid[order] > 0)
    kept = np.zeros(len(sets), dtype=bool)
    kept[order[~dup]] = True

    # The candidates' own branch: the last one, after the wide-association ('in_both') branch
    branch = sorted(set(stats.index.get_level_values('branch')))[-1]
    row = pd.Series(0.0, index=stats.columns)
    for b in BAND_SETS:
        row[f'n_{b}_before'] = np.sum(sets == b)
        row[f'n_{b}_after'] = np.sum(kept & (sets == b))
    stats.loc[(MERGE_STEP, branch), :] = row
    return stats


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
    tab_wide_fields = {1: [], 2: []}

    # Combine everything
    # Only field folders (e.g. 000400); leftover copies like combined_pre_filtered_out/ are skipped
    fnames = sorted(f for f in os.listdir(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME)) if f.isdigit())
    for cat in (0, 1):  # PS1-only branch (2) disabled
        for field_name in fnames:
            # Load the stats
            stat_df = pd.read_csv(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, field_name, f'{cat}_filter_stats.csv'))
            stat_df['branch'] = stat_df['branch'].fillna('')
            stat_dfs[cat].append(stat_df)

            # Load the tabs
            if cat != 2:  # TODO: temporary for bad pstarr extraction
                tabs[cat].append(load_ecsv(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, field_name, f'{cat}.ecsv')))
                tab_fields[cat].append(field_name)

    # Wide-association tables disabled (plan_filter_review.md #4): filtering no longer writes them.
    # # Deal with wide associations
    # for cat in (1, 2):
    #     for field_name in fnames:
    #         # Load the tabs_wide
    #         if cat != 2:  # TODO: temporary for bad pstarr extraction
    #             tabs_wide[cat].append(load_ecsv(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, field_name, f'{cat}_wide_association.ecsv')))
    #             tab_wide_fields[cat].append(field_name)

    # Filter reasons, computed per field while the reject tables are still
    # next to the survivors. Attached before the vstack so the columns simply
    # come along.
    if add_filtered_out:
        field_names = fnames
        print(f'Summarising filtered-out reasons for {len(field_names)} fields '
              f'on {processes} process(es)...', flush=True)
        worker = partial(_field_reasons, run_dirname=FILTER_RESULTS_DIRNAME,
                         max_arcsec=max_arcsec)
        if processes > 1:
            with Pool(processes) as pool:
                results = pool.map(worker, field_names)
        else:
            results = [worker(f) for f in field_names]
        by_field = dict(results)

        def _attach(field_list, tab_list, key):
            assert len(field_list) == len(tab_list)
            for field_name, tab in zip(field_list, tab_list):
                cols = by_field.get(field_name, {}).get(key)
                if cols is None:
                    cols = {c: np.full(len(tab), filtered_out_summary.NO_TABLE)
                            for c in filtered_out_summary.ALL_COLUMNS}
                for name, values in cols.items():
                    tab[name] = np.asarray(values, dtype=str)

        for cat in (0, 1):
            _attach(tab_fields[cat], tabs[cat], cat)
        # for cat in (1,):  # wide-association tables disabled (#4)
        #     _attach(tab_wide_fields[cat], tabs_wide[cat], f'wide{cat}')

    # Drop length 0 tabs
    for cat in (0, 1):  # PS1-only branch (2) disabled
        tabs[cat] = [tab for tab in tabs[cat] if len(tab) > 0]
    for cat in (1, 2):
        tabs_wide[cat] = [tab for tab in tabs_wide[cat] if len(tab) > 0]

    # Combine the stats
    stat_dfs = {cat: combine_stats(stat_dfs[cat]) for cat in (0, 1)}  # PS1-only branch (2) disabled

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

    # Link duplicates of one object (overlapping fields, or the same object in both tables)
    add_group_ids([tabs[0], tabs[1]])
    for cat in (0, 1):
        n = int(np.sum(np.asarray(tabs[cat]['group_id']) > 0)) if 'group_id' in tabs[cat].colnames else 0
        print(f'Catalog {cat}: {n} of {len(tabs[cat])} rows share a group_id with another row', flush=True)

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
    for cat in (0, 1):  # PS1-only branch (2) disabled
        stat_dfs[cat] = add_merge_duplicates_step(stat_dfs[cat], tabs[cat])
        stat_dfs[cat].to_csv(os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, out_subdir, f'{cat}_filter_stats.csv'))
        if cat != 2:  # TODO: temporary for bad pstarr extraction
            prepare_table_for_write(tabs[cat]).write(
                os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, out_subdir, f'{cat}.hdf5'),
                path='data', serialize_meta=True, overwrite=True)

    # for cat in (1, 2):  # wide-association tables disabled (#4)
    #     if cat != 2:  # TODO: temporary for bad pstarr extraction
    #         prepare_table_for_write(tabs_wide[cat]).write(
    #             os.path.join(get_data_path(), FILTER_RESULTS_DIRNAME, out_subdir, f'{cat}_wide_association.hdf5'),
    #             path='data', serialize_meta=True, overwrite=True)

    for cat in (0, 1):  # PS1-only branch (2) disabled
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
