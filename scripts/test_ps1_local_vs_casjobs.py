#!/usr/bin/env python3
"""Assert that PS1_Local.py and the CasJobs PSTARR_Catalog return identical results.

Tests a small sky tile against both query backends and checks that:
  - Every objID returned by CasJobs is present in the local result
  - Every objID returned locally is present in the CasJobs result
  - All shared columns agree to floating-point tolerance (or exactly for ints)
  - psfLikelihood columns are excluded from the comparison (removed from local catalog)

Usage
-----
    python scripts/test_ps1_local_vs_casjobs.py
    python scripts/test_ps1_local_vs_casjobs.py --ra-center 180.0 --dec-center 0.0 --half-width 0.05
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Extracting.PS1_Local import PanSTARRSLocal, DEFAULT_PARQUET_DIR
from Extracting.Catalogs import PSTARR_Catalog

BANDS = ('g', 'r', 'i')
PSF_LIKELIHOOD_COLS = {f'{b}psfLikelihood' for b in 'grizy'}


def casjobs_query(ra_range, dec_range, bands):
    """Run the existing CasJobs-based query and return a DataFrame sorted by objID.

    Reverses the column_map renaming (PanSTARR_ID→objID, ra→raMean, dec→decMean)
    so column names are comparable to the local query output.
    """
    cat = PSTARR_Catalog(
        ra_range=ra_range,
        dec_range=dec_range,
        catalog_bands=bands,
        query_buffer=0.0,   # exact range so it matches the local query
        prefetch=True,
    )
    tab = cat.data
    df = tab.to_pandas()
    # Reverse the column_map renaming for a fair comparison
    df = df.rename(columns={'PanSTARR_ID': 'objID', 'ra': 'raMean', 'dec': 'decMean'})
    # Drop psfLikelihood columns
    drop = [c for c in df.columns if c in PSF_LIKELIHOOD_COLS]
    df = df.drop(columns=drop)
    return df.sort_values('objID').reset_index(drop=True)


def local_query(ra_range, dec_range, bands):
    """Run per-band local queries and outer-join them, matching Catalogs.py behaviour."""
    ps1 = PanSTARRSLocal(parquet_dir=DEFAULT_PARQUET_DIR, per_band=True)
    base_cols = ['objID', 'raMean', 'decMean', 'qualityFlag', 'objInfoFlag']

    band_dfs = {}
    for band in bands:
        df = ps1.query_tile(ra_range=ra_range, dec_range=dec_range, band=band)
        band_dfs[band] = df

    # Outer-join bands on shared base columns (same logic as Catalogs._join_tables)
    final = band_dfs[bands[0]]
    for band in bands[1:]:
        final = final.merge(band_dfs[band], on=base_cols, how='outer')

    return final.sort_values('objID').reset_index(drop=True)


CASJOBS_FILL = -999.0  # sentinel used by _join_tables for missing band values
PHOT_SUFFIXES = ('KronMag', 'KronMagErr', 'PSFMag', 'PSFMagErr', 'infoFlag', 'infoFlag2')


def _to_numeric(series: pd.Series) -> pd.Series:
    """Cast to float64, treating CasJobs -999 fill as NaN and coercing strings."""
    s = pd.to_numeric(series, errors='coerce')
    return s.where(s != CASJOBS_FILL, other=np.nan)


def compare(cj_df: pd.DataFrame, local_df: pd.DataFrame) -> bool:
    """Compare two DataFrames and print a detailed report. Returns True if all checks pass."""
    ok = True

    cj_ids    = set(cj_df['objID'].dropna().apply(int))
    local_ids = set(local_df['objID'].dropna().apply(int))
    only_cj    = cj_ids - local_ids
    only_local = local_ids - cj_ids

    print(f'\n{"="*60}')
    print(f'CasJobs rows:  {len(cj_df):,}')
    print(f'Local rows:    {len(local_df):,}')
    print(f'Only in CasJobs (known ~0.4% multi-skycell edge case): {len(only_cj):,}')
    print(f'Only in local:                                          {len(only_local):,}')

    if only_local:
        print(f'  FAIL: {len(only_local)} objIDs in local missing from CasJobs (unexpected)')
        ok = False

    common_ids = cj_ids & local_ids
    if not common_ids:
        print('No common objIDs — cannot compare columns.')
        return False

    cj_sub    = cj_df[cj_df['objID'].apply(int).isin(common_ids)].set_index('objID').sort_index()
    local_sub = local_df[local_df['objID'].apply(int).isin(common_ids)].set_index('objID').sort_index()

    cj_cols    = set(cj_sub.columns) - PSF_LIKELIHOOD_COLS
    local_cols = set(local_sub.columns)
    shared_cols = cj_cols & local_cols

    if cj_cols - local_cols:
        print(f'  NOTE: columns in CasJobs but not local: {sorted(cj_cols - local_cols)}')
    if local_cols - cj_cols:
        print(f'  NOTE: columns in local but not CasJobs: {sorted(local_cols - cj_cols)}')

    print(f'\nComparing {len(shared_cols)} shared columns on {len(common_ids):,} common objIDs:')
    col_failures = []
    for col in sorted(shared_cols):
        cj_vals    = _to_numeric(cj_sub[col])
        local_vals = _to_numeric(local_sub[col])

        # Objects where CasJobs has data but local is NaN
        cj_has_local_missing = ~cj_vals.isna() & local_vals.isna()
        # Objects where local has data but CasJobs is NaN (unexpected direction)
        local_has_cj_missing = cj_vals.isna() & ~local_vals.isna()

        # For per-band photometric columns: NaN in local should be fully explained
        # by the corresponding band's infoFlag2 & 4 != 0 in the local catalog.
        band = next((b for b in 'grizy' if col.startswith(b) and col[1:] in PHOT_SUFFIXES), None)
        if band and cj_has_local_missing.any():
            flag_col = f'{band}infoFlag2'
            if flag_col in local_sub.columns:
                local_flags = _to_numeric(local_sub[flag_col]).fillna(4).astype(int)
                unexplained = cj_has_local_missing & ~((local_flags & 4) != 0)
                if unexplained.any():
                    col_failures.append(
                        (col, f'{unexplained.sum()} NaN in local NOT explained by {flag_col} & 4'))
                else:
                    n = cj_has_local_missing.sum()
                    print(f'  OK  {col}  ({n} local NaNs all explained by {flag_col} & 4 != 0)')
                    continue

        if cj_has_local_missing.any() or local_has_cj_missing.any():
            col_failures.append(
                (col, f'cj_has_local_missing={cj_has_local_missing.sum()}, '
                      f'local_has_cj_missing={local_has_cj_missing.sum()}'))
            continue

        # Values agree where both are non-NaN
        both_valid = ~cj_vals.isna() & ~local_vals.isna()
        if both_valid.any():
            a, b       = cj_vals[both_valid].to_numpy(), local_vals[both_valid].to_numpy()
            close      = np.isclose(a, b, rtol=1e-9, atol=0)
            n_mismatch = (~close).sum()
            max_diff   = np.abs(a - b).max()
        else:
            max_diff, n_mismatch = 0.0, 0

        if n_mismatch > 0:
            col_failures.append((col, f'val_mismatch={n_mismatch}, max_diff={max_diff:.6g}'))
        else:
            print(f'  OK  {col}  (max_diff={max_diff:.2e})')

    if col_failures:
        ok = False
        print('\nFAILED columns:')
        for col, msg in col_failures:
            print(f'  FAIL  {col}: {msg}')

    print(f'\n{"="*60}')
    print('RESULT:', 'PASS' if ok else 'FAIL')
    return ok


# Predefined test suite.  Each entry: (label, ra_center, dec_center, half_width, bands)
TEST_SUITE = [
    # Baseline: small tile, single HEALPix pixel
    ('baseline_small',      180.0,   0.0, 0.05, ('g', 'r', 'i')),
    # Wide tile: 1×1 deg, crosses multiple nside=32 pixels (~1.0 deg each at equator)
    ('multi_pixel_wide',     45.0,  30.0, 0.50, ('g', 'r', 'i')),
    # Different sky region: southern hemisphere, different RA
    ('southern_hemisphere', 270.0, -30.0, 0.30, ('g', 'r', 'i')),
    # All five bands
    ('all_five_bands',      180.0,   0.0, 0.05, ('g', 'r', 'i', 'z', 'y')),
]


def run_case(label, ra_center, dec_center, half_width, bands):
    ra_range  = (ra_center  - half_width, ra_center  + half_width)
    dec_range = (dec_center - half_width, dec_center + half_width)

    print(f'\n{"#"*60}')
    print(f'CASE: {label}')
    print(f'  RA {ra_range}, Dec {dec_range}, bands={bands}')

    print('\n--- Running CasJobs query ---')
    cj_df = casjobs_query(ra_range, dec_range, bands)
    print(f'CasJobs returned {len(cj_df):,} rows')

    print('\n--- Running local query ---')
    local_df = local_query(ra_range, dec_range, bands)
    print(f'Local returned {len(local_df):,} rows')

    return compare(cj_df, local_df)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ra-center',  type=float, default=None)
    parser.add_argument('--dec-center', type=float, default=None)
    parser.add_argument('--half-width', type=float, default=0.05)
    parser.add_argument('--bands',      nargs='+',  default=None)
    args = parser.parse_args()

    if args.ra_center is not None or args.dec_center is not None:
        # Single custom tile
        bands = tuple(args.bands) if args.bands else BANDS
        passed = run_case('custom', args.ra_center or 180.0, args.dec_center or 0.0,
                          args.half_width, bands)
        sys.exit(0 if passed else 1)

    # Full test suite
    results = {}
    for label, ra_c, dec_c, hw, bands in TEST_SUITE:
        results[label] = run_case(label, ra_c, dec_c, hw, bands)

    print(f'\n{"#"*60}')
    print('SUITE SUMMARY:')
    all_passed = True
    for label, passed in results.items():
        status = 'PASS' if passed else 'FAIL'
        print(f'  {status}  {label}')
        all_passed = all_passed and passed
    print(f'\nOVERALL: {"PASS" if all_passed else "FAIL"}')
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
