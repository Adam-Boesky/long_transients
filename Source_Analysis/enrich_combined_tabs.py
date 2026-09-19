"""Append catalog crossmatch columns to a combined filter-results table.

Runs between `combine_filtered_tabs.py` and `store_src_plots.py`. Each source
gets its nearest match in SDSS DR17, DESI DR1, TNS, AllWISE, AGN-DB and SIMBAD
appended as columns, so cuts can be made on classification without querying
anything at plot time.

    python -m Source_Analysis.enrich_combined_tabs <path to a combined .ecsv>

Writes `<stem>_enriched.ecsv` alongside the input; pass --in-place to overwrite
the input instead. Re-running is safe: existing crossmatch columns are dropped
and recreated rather than duplicated.

`--catalogs` restricts the run to a subset, which is how a newly added catalog
gets folded into an already-enriched table without redoing the expensive ones:

    python -m Source_Analysis.enrich_combined_tabs \
        Data/.../combined/0_enriched.ecsv --in-place --catalogs simbad

Columns for the named catalogs are replaced and all others are carried through
untouched.

Memory peaks around 25 GB with every catalog loaded -- DESI's in-memory KD-tree
is ~13 GB and SIMBAD adds ~8 GB resident, with a ~10 GB transient spike while
its parquet is read -- so run this as a single process, never inside a
multiprocessing Pool. A `--catalogs` subset only pays for what it names. The
first AllWISE run queries IRSA for every position (roughly 10 s per 1000) and
caches the answers, so later runs are fast.
"""
import argparse
import os
import sys
from typing import Optional, Sequence

import pandas as pd
from astropy.table import Table, hstack

sys.path.append('/Users/adamboesky/Research/long_transients')

from Extracting.utils import load_ecsv
from Source_Analysis.catalogs import agn, desi, sdss, simbad, tns, wise
from Source_Analysis.catalogs._base import text_column_names, to_astropy_table

# Matches Source.max_arcsec, the association radius every per-source lookup in
# Sources.py already uses (SDSS query_region, the SPARCL box, AGN-DB and TNS).
MATCH_RADIUS_ARCSEC = 1.5

CATALOGS = {'sdss': sdss, 'desi': desi, 'tns': tns, 'wise': wise, 'agn_db': agn,
            'simbad': simbad}


def selected_catalogs(catalogs: Optional[Sequence[str]] = None) -> list[str]:
    """Validate a catalog subset, returned in `CATALOGS` order.

    Ordering follows `CATALOGS` rather than the caller's argument order, so the
    appended columns land in the same place no matter how the subset was spelled.
    """
    if not catalogs:
        return list(CATALOGS)
    unknown = sorted(set(catalogs) - set(CATALOGS))
    if unknown:
        raise KeyError(f'unknown catalog(s) {unknown}; choose from {sorted(CATALOGS)}')
    return [name for name in CATALOGS if name in catalogs]


def crossmatch_columns(
    ra, dec,
    radius_arcsec: float = MATCH_RADIUS_ARCSEC,
    catalogs: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Nearest match in each catalog, one row per input position, in input order."""
    frames = []
    for name in selected_catalogs(catalogs):
        frame = CATALOGS[name].cone_search_many(ra, dec, radius_arcsec=radius_arcsec)
        n = int(frame[f'{name}_matched'].sum())
        print(f'  {name:6s} {n:6,} / {len(frame):,} matched')
        frames.append(frame)
    return pd.concat(frames, axis=1)


def enrich(
    in_path: str,
    out_path: str,
    radius_arcsec: float = MATCH_RADIUS_ARCSEC,
    catalogs: Optional[Sequence[str]] = None,
) -> str:
    """Crossmatch `in_path` against `catalogs` (default all) and write `out_path`.

    Only the selected catalogs are touched: their columns are replaced, and any
    other catalog's columns already on the table are carried through untouched.
    That is what makes it safe to add one catalog to an already-enriched table
    without paying to rebuild the rest -- notably DESI's ~13 GB KD-tree.
    """
    names = selected_catalogs(catalogs)
    table = load_ecsv(in_path)
    print(f'{os.path.basename(in_path)}: {len(table):,} sources, '
          f'{len(table.colnames)} columns')
    print(f'  crossmatching {", ".join(names)} at {radius_arcsec}"')

    # Drop only the *selected* catalogs' columns, so re-running replaces rather
    # than duplicating them. Widening this to every catalog in CATALOGS would
    # delete the unselected ones' columns without recreating them.
    stale = [c for c in table.colnames if any(c.startswith(f'{p}_') for p in names)]
    if stale:
        print(f'  dropping {len(stale)} columns from a previous run')
        table.remove_columns(stale)

    block = to_astropy_table(
        crossmatch_columns(table['ra'], table['dec'], radius_arcsec, names))
    out = hstack([table, block])

    # An .hdf5 sibling shadows the .ecsv in `load_ecsv`, so leaving a stale one
    # next to a freshly written table makes the new columns invisible.
    hdf5_sibling = f'{os.path.splitext(out_path)[0]}.hdf5'
    if os.path.exists(hdf5_sibling):
        print(f'  WARNING: {os.path.basename(hdf5_sibling)} exists and load_ecsv '
              f'prefers it -- regenerate or remove it or these columns will not '
              f'be seen')

    # Over the whole output, not just `block`. ECSV cannot distinguish an empty
    # string from a masked value without 'data_mask', and a --catalogs subset
    # carries through text columns from the catalogs it did not touch -- those
    # need it too, or a matched-but-empty value (e.g. sdss_subclass on a source
    # SDSS matched without recording a subclass) comes back masked, silently
    # turning "matched, no value" into "no match".
    text = text_column_names(out)

    # Staged then swapped: an interrupted write would otherwise truncate the
    # input itself when out_path == in_path.
    staging = f'{out_path}.incoming'
    out.write(staging, format='ascii.ecsv', overwrite=True,
              serialize_method={c: 'data_mask' for c in text})
    os.replace(staging, out_path)
    print(f'  wrote {out_path} ({len(out.colnames)} columns, '
          f'{len(text)} of them text) [{os.path.getsize(out_path) / 1e6:.1f} MB]')
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('table', help='combined filter-results .ecsv to enrich')
    parser.add_argument('--in-place', action='store_true',
                        help='overwrite the input instead of writing <stem>_enriched.ecsv')
    parser.add_argument('--radius-arcsec', type=float, default=MATCH_RADIUS_ARCSEC,
                        help=f'match radius (default {MATCH_RADIUS_ARCSEC}, = Source.max_arcsec)')
    parser.add_argument('--catalogs', nargs='+', metavar='NAME', choices=sorted(CATALOGS),
                        help='crossmatch only these catalogs instead of all of them. '
                             'Their columns are replaced; every other catalog already '
                             'on the table is left alone, so one catalog can be added '
                             'to an --in-place enriched table cheaply.')
    args = parser.parse_args()

    out_path = args.table if args.in_place else f'{os.path.splitext(args.table)[0]}_enriched.ecsv'
    enrich(args.table, out_path, args.radius_arcsec, args.catalogs)


if __name__ == '__main__':
    main()
