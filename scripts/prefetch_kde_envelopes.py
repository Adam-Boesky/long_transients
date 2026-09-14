"""Pre-populate KDE envelope caches for all gri fields.

Runs the same quality-filter pipeline as filter_field() up to the KDE-build
step, saves {data_path}/kde_envelopes/{field}/{band}.npz, then exits.
Fields whose .npz files already exist for all three bands are skipped.

Safe to run in parallel with the main filter job — both produce identical
deterministic output, so a write race at most wastes one computation.
"""

import os
import sys
import numpy as np

from astropy.table import Table, vstack
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append('/n/home04/aboesky/berger/long_transients')

from Source_Analysis.filter_fields import (
    Filters,
    remove_mask,
    load_or_build_kde_envelopes,
    EXTRACTED_CATALOG_DIR,
)
from Extracting.utils import get_data_path

BANDS = ('g', 'r', 'i')
MIN_DEC = -29.5


def _kde_cached(field_name: str, data_path: str) -> bool:
    cache_dir = os.path.join(data_path, 'kde_envelopes', field_name)
    return all(os.path.exists(os.path.join(cache_dir, f'{b}.npz')) for b in BANDS)


def build_kde_for_field(field_name: str):
    data_path = get_data_path()

    if _kde_cached(field_name, data_path):
        print(f'[{field_name}] cache hit, skipping', flush=True)
        return

    print(f'[{field_name}] loading tables...', flush=True)
    tables = {}
    for band in BANDS:
        path = os.path.join(data_path, f'{EXTRACTED_CATALOG_DIR}/{field_name}_{band}.hdf5')
        try:
            tables[band] = Table.read(path, path='data')
        except FileNotFoundError:
            print(f'[{field_name}] WARNING: missing band {band}, skipping field', flush=True)
            return

    # Mirror the upper-limit and mask-removal logic from filter_field()
    for band in BANDS:
        for b in BANDS:
            if f'PSTARR_{b}PSFMag' not in tables[band].colnames:
                tables[band][f'PSTARR_{b}PSFMag'] = -999 * np.ones(len(tables[band]))
        tab = tables[band]
        upper_lim_mask = tab[f'ZTF_{band}PSFFlags'] == 4
        tab[f'ZTF_{band}PSFMag'][upper_lim_mask] = tab[f'ZTF_{band}_mag_limit'][upper_lim_mask]
        tab[f'ZTF_{band}_upper_lim_flag'] = False
        tab[f'ZTF_{band}_upper_lim_flag'][upper_lim_mask] = True
        tables[band] = remove_mask(tab)

    # Quality-filter pipeline (mirrors filter_field exactly)
    _qf = Filters()
    all_q = {band: tab.copy() for band, tab in tables.items()}
    all_q = _qf.filter(all_q, 'sep_extraction_filter')
    all_q, ztf_low, pstarr_low = _qf.filter(all_q, 'snr_filter', snr_min=5, both_cat=True)

    # tabs0 mirrors `tabs` in filter_field() — the pre-quality-filter Catalog_Flag==0
    # subset used only as a schema source for the empty-band fallback below.
    tabs0 = {band: tables[band][tables[band]['Catalog_Flag'] == 0] for band in tables}
    processed_bands = list(tabs0.keys())  # bands present in the raw tables (always all 3)
    for band in processed_bands:
        if band in ztf_low:
            ztf_low[band]['Catalog_Flag'] = 1
        if band in pstarr_low:
            pstarr_low[band]['Catalog_Flag'] = 2
        if band not in all_q:
            all_q[band] = tabs0[band][:0].copy()
        if band not in ztf_low:
            ztf_low[band] = tabs0[band][:0].copy()
            ztf_low[band]['Catalog_Flag'] = 1
        if band not in pstarr_low:
            pstarr_low[band] = tabs0[band][:0].copy()
            pstarr_low[band]['Catalog_Flag'] = 2
    all_q = {band: vstack([all_q[band], ztf_low[band], pstarr_low[band]]) for band in processed_bands}
    del ztf_low, pstarr_low, tabs0

    all_q = _qf.filter(all_q, 'shape_filter')
    all_q = _qf.filter(all_q, 'pstarr_not_saturated')
    all_q = _qf.filter(all_q, 'psf_fit_filter')
    all_q = _qf.filter(all_q, 'dec_greater_than', min_dec=MIN_DEC)

    in_both = {band: tab[tab['Catalog_Flag'] == 0] for band, tab in all_q.items()}
    del all_q

    print(f'[{field_name}] building KDE envelopes...', flush=True)
    load_or_build_kde_envelopes(field_name, in_both)
    del in_both
    print(f'[{field_name}] done', flush=True)


def main():
    data_path = get_data_path()

    g_fields = np.load(os.path.join(data_path, 'g_imaged_fields.npy'))
    r_fields = np.load(os.path.join(data_path, 'r_imaged_fields.npy'))
    i_fields = np.load(os.path.join(data_path, 'i_imaged_fields.npy'))
    gri_fields = np.intersect1d(np.intersect1d(g_fields, r_fields), i_fields)
    fields = [str(f).zfill(6) for f in sorted(gri_fields)]

    todo = [f for f in fields if not _kde_cached(f, data_path)]
    print(f'{len(todo)} / {len(fields)} fields need KDE pre-population', flush=True)

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(build_kde_for_field, f): f for f in todo}
        for future in as_completed(futures):
            field = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f'ERROR: {field} failed with {type(e).__name__}: {e}', flush=True)


if __name__ == '__main__':
    main()
