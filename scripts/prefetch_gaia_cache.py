"""Warm the Gaia query cache for all fields used by filter_fields.filter_fields().

Replicates the exact table-loading, preprocessing, and Gaia box-query logic from
filter_field() (Source_Analysis/filter_fields.py) so the cached .ecsv files are
byte-for-byte what the real filtering run would produce -- just run in parallel
across fields instead of serially inside the single-worker filtering pipeline.

Usage
-----
    python scripts/prefetch_gaia_cache.py
"""

import gc
import os
import sys
import numpy as np
from astropy.table import Table
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append('/n/home04/aboesky/berger/long_transients')

from Source_Analysis.filter_fields import remove_mask, query_gaia_for_field, EXTRACTED_CATALOG_DIR
from Extracting.utils import get_data_path

FIELDS = ['000616', '000617', '000619', '000303', '000304', '000373',
   '000374', '000375', '000377', '000363', '000368', '000339',
   '000516', '000517', '000518', '000324', '000326', '000337',
   '000338', '000567', '000568', '000569', '000305', '000313',
   '000315', '000318']


def prefetch_field(field_name: str) -> int:
    """Load a field's catalogs and trigger+cache its Gaia query. Returns n Gaia sources."""
    tables = {}
    for band in ('g', 'r', 'i'):
        try:
            tables[band] = Table.read(
                os.path.join(get_data_path(), f'{EXTRACTED_CATALOG_DIR}/{field_name}_{band}.hdf5'),
                path='data',
            )
        except FileNotFoundError:
            continue

    for band in tables.keys():
        for b in ('g', 'r', 'i'):
            if f'PSTARR_{b}PSFMag' not in tables[band].colnames:
                tables[band][f'PSTARR_{b}PSFMag'] = -999 * np.ones(len(tables[band]))

        tab = tables[band]
        upper_lim_mask = tab[f'ZTF_{band}PSFFlags'] == 4
        tab[f'ZTF_{band}PSFMag'][upper_lim_mask] = tab[f'ZTF_{band}_mag_limit'][upper_lim_mask]
        tab[f'ZTF_{band}_upper_lim_flag'] = False
        tab[f'ZTF_{band}_upper_lim_flag'][upper_lim_mask] = True
        tables[band] = remove_mask(tab)

    all_ras = np.concatenate([np.asarray(t['ra']) for t in tables.values()])
    all_decs = np.concatenate([np.asarray(t['dec']) for t in tables.values()])

    # These per-field catalogs are large (multi-GB on disk, larger still once
    # remove_mask() converts masked columns to object dtype) and only ra/dec
    # are needed past this point -- free them before the next field starts so
    # peak memory across concurrent workers doesn't stack up.
    del tables
    gc.collect()

    gaia_table = query_gaia_for_field(field_name, all_ras, all_decs)
    return len(gaia_table)


def main():
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(prefetch_field, f): f for f in FIELDS}
        for future in as_completed(futures):
            field = futures[future]
            try:
                n = future.result()
                print(f'DONE {field}: {n} Gaia sources cached', flush=True)
            except Exception as e:
                print(f'FAILED {field}: {type(e).__name__}: {e}', flush=True)


if __name__ == '__main__':
    main()
