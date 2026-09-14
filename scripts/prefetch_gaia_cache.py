"""Warm the Gaia query cache for every field we have extracted on.

filter_field() (Source_Analysis/filter_fields.py) issues one bulk Gaia box query
per field and caches it to {get_data_path()}/gaia_cache/{field}.ecsv.  Doing that
inline is the slowest part of a filtering run, so this script warms the cache for
all fields ahead of time, in parallel.

The Gaia box is derived solely from the min/max of the field's `ra`/`dec` columns,
so -- unlike filter_field() -- we read only those two columns straight out of the
HDF5 compound dataset and skip the mag preprocessing entirely.  None of that
preprocessing touches ra/dec, so the resulting box (and therefore the cached
table) is identical to what a real filtering run would produce, while reading a
few MB per field instead of several GB.

Fields are discovered from the extraction output directory rather than hardcoded,
so this keeps up with the full-sky rollout automatically.

Usage
-----
    python scripts/prefetch_gaia_cache.py [--workers N] [--dry-run] [--fields F [F ...]]
"""

import argparse
import os
import re
import sys
import numpy as np
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append('/n/home04/aboesky/berger/long_transients')

from astroquery.gaia import Gaia

from Source_Analysis.filter_fields import query_gaia_for_field, EXTRACTED_CATALOG_DIR
from Extracting.utils import get_data_path, get_credentials

BANDS = ('g', 'r', 'i')
FIELD_FILE_RE = re.compile(r'^(\d+)_([gri])\.hdf5$')


def discover_fields() -> list:
    """Return every field name with at least one extracted band, sorted."""
    extract_dir = os.path.join(get_data_path(), EXTRACTED_CATALOG_DIR)
    fields = set()
    for fname in os.listdir(extract_dir):
        match = FIELD_FILE_RE.match(fname)
        if match:
            fields.add(match.group(1))
    return sorted(fields)


def is_cached(field_name: str) -> bool:
    return os.path.exists(os.path.join(get_data_path(), 'gaia_cache', f'{field_name}.ecsv'))


def read_field_coords(field_name: str) -> tuple:
    """Read just the ra/dec columns for every readable band of a field.

    A band whose HDF5 file is unreadable (truncated or corrupt from a failed
    extraction) is skipped with a warning rather than losing the whole field:
    the bands cover the same footprint, so the remaining ones still give the
    right Gaia box.
    """
    extract_dir = os.path.join(get_data_path(), EXTRACTED_CATALOG_DIR)
    ras, decs = [], []
    for band in BANDS:
        path = os.path.join(extract_dir, f'{field_name}_{band}.hdf5')
        if not os.path.exists(path):
            continue
        try:
            with h5py.File(path, 'r') as f:
                # Field selection on the compound dataset pulls only these two
                # columns off disk instead of the full multi-GB row set.
                coords = f['data'].fields(['ra', 'dec'])[:]
        except OSError as e:
            print(f'WARNING: {field_name}_{band}.hdf5 is unreadable, skipping band '
                  f'({type(e).__name__}: {e})', flush=True)
            continue
        ras.append(np.asarray(coords['ra'], dtype=float))
        decs.append(np.asarray(coords['dec'], dtype=float))

    if not ras:
        raise FileNotFoundError(f'No readable extracted bands for field {field_name}')

    return np.concatenate(ras), np.concatenate(decs)


def prefetch_field(field_name: str) -> int:
    """Trigger + cache the Gaia query for a field. Returns n Gaia sources."""
    all_ras, all_decs = read_field_coords(field_name)
    gaia_table = query_gaia_for_field(field_name, all_ras, all_decs)
    return len(gaia_table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3,
                        help='Concurrent Gaia queries (default: 3). The ESA TAP '
                             'server is the bottleneck; keep this modest.')
    parser.add_argument('--dry-run', action='store_true',
                        help='List what would be queried and exit.')
    parser.add_argument('--fields', nargs='+', default=None,
                        help='Explicit field names instead of directory discovery.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-query fields that are already cached.')
    args = parser.parse_args()

    fields = args.fields if args.fields is not None else discover_fields()
    print(f'Found {len(fields)} extracted fields.', flush=True)

    if not args.overwrite:
        todo = [f for f in fields if not is_cached(f)]
        print(f'{len(fields) - len(todo)} already cached, {len(todo)} to query.', flush=True)
    else:
        todo = list(fields)
        print(f'--overwrite set: re-querying all {len(todo)} fields.', flush=True)

    if args.dry_run:
        print('Fields to query:', ' '.join(todo))
        return

    if not todo:
        print('Nothing to do -- cache is fully warm.')
        return

    # Log in once up front so concurrent workers don't race on Gaia.login().
    username, password = get_credentials('gaia_login.txt')
    Gaia.login(user=username, password=password)

    n_done = n_failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(prefetch_field, f): f for f in todo}
        for future in as_completed(futures):
            field = futures[future]
            try:
                n = future.result()
                n_done += 1
                print(f'[{n_done + n_failed}/{len(todo)}] DONE {field}: {n} Gaia sources cached', flush=True)
            except Exception as e:
                n_failed += 1
                print(f'[{n_done + n_failed}/{len(todo)}] FAILED {field}: {type(e).__name__}: {e}', flush=True)

    print(f'\nFinished: {n_done} cached, {n_failed} failed.', flush=True)


if __name__ == '__main__':
    main()
