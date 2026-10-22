"""Test that bulk Gaia query produces the same parallax/PM filter decisions as per-source queries.

Usage:
    conda run -n long_transients python scripts/test_bulk_gaia.py
"""
import sys
import time
import numpy as np

sys.path.append('/Users/adamboesky/Research/long_transients')
sys.path.append('/n/home04/aboesky/berger/long_transients')

from astropy.coordinates import SkyCoord
from astropy.table import Table

from Extracting.utils import get_data_path, load_ecsv
from Source_Analysis.filter_fields import query_gaia_for_field, _match_sources_to_gaia
from Source_Analysis.Sources import Sources

TEST_FILE   = '/Users/adamboesky/Research/long_transients/Data/debugging/filter_results/kde_000573/0.hdf5'
FIELD_NAME  = 'kde_000573'
N_PER_SRC   = 10       # cap for the slow per-source path
MAX_ARCSEC  = 5.0
PARALLAX_THRESH = 5.0


def per_source_parallax_keep(src) -> bool:
    """Replicate _parallax_in_srcs per-source logic. Returns True if source should be kept."""
    info = src.GAIA_info
    if len(info) == 0:
        return True
    return (
        info[0]['parallax_over_error'] < PARALLAX_THRESH
        or info['parallax_over_error'].mask[0]
    )


def per_source_pm_keep(src) -> bool:
    """Replicate _proper_motion_in_srcs per-source logic. Returns True if source should be kept."""
    info = src.GAIA_info
    if len(info) == 0:
        return True
    return (
        info[0]['pm'] / (info[0]['pmra_error'] + info[0]['pmdec_error']) < PARALLAX_THRESH
    ) or (
        info['pm'].mask[0]
        or info['pmra_error'].mask[0]
        or info['pmdec_error'].mask[0]
    )


def bulk_parallax_keep(row) -> bool:
    """Bulk path: True = keep."""
    if row is None:
        return True
    poe = row['parallax_over_error']
    if np.ma.is_masked(poe):
        return True
    return float(poe) < PARALLAX_THRESH


def bulk_pm_keep(row) -> bool:
    """Bulk path: True = keep."""
    if row is None:
        return True
    pm, pmra_e, pmdec_e = row['pm'], row['pmra_error'], row['pmdec_error']
    if np.ma.is_masked(pm) or np.ma.is_masked(pmra_e) or np.ma.is_masked(pmdec_e):
        return True
    return float(pm) / (float(pmra_e) + float(pmdec_e)) < PARALLAX_THRESH


def main():
    print(f'Loading sources from {TEST_FILE}')
    srcs = Sources.from_file(TEST_FILE)
    print(f'Loaded {len(srcs)} sources.')

    # Collect RA/Dec for bulk query
    ras  = np.array([s.ra  for s in srcs])
    decs = np.array([s.dec for s in srcs])

    # ── Bulk approach ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    gaia_table = query_gaia_for_field(FIELD_NAME, ras, decs)
    idx = _match_sources_to_gaia(srcs, gaia_table, max_arcsec=MAX_ARCSEC)
    bulk_elapsed = time.perf_counter() - t0

    print(f'Bulk Gaia query: {len(gaia_table)} Gaia sources, {np.sum(idx >= 0)} / {len(srcs)} matched.')

    # Compute bulk decisions for all sources
    bulk_parallax = np.array([
        bulk_parallax_keep(gaia_table[idx[i]] if idx[i] >= 0 else None)
        for i in range(len(srcs))
    ])
    bulk_pm = np.array([
        bulk_pm_keep(gaia_table[idx[i]] if idx[i] >= 0 else None)
        for i in range(len(srcs))
    ])

    # ── Per-source approach (capped at N_PER_SRC) ─────────────────────────────
    cap = min(N_PER_SRC, len(srcs))
    print(f'\nRunning per-source Gaia queries for first {cap} sources...')
    t1 = time.perf_counter()
    per_src_parallax = []
    per_src_pm       = []
    for i in range(cap):
        per_src_parallax.append(per_source_parallax_keep(srcs[i]))
        per_src_pm.append(per_source_pm_keep(srcs[i]))
    per_src_elapsed = time.perf_counter() - t1
    per_src_parallax = np.array(per_src_parallax)
    per_src_pm       = np.array(per_src_pm)

    # ── Compare decisions ─────────────────────────────────────────────────────
    parallax_match = np.all(bulk_parallax[:cap] == per_src_parallax)
    pm_match       = np.all(bulk_pm[:cap]       == per_src_pm)

    print('\n── Results ───────────────────────────────────────────────────────────────')
    print(f'  Parallax filter decisions match:      {"PASS" if parallax_match else "FAIL"}')
    if not parallax_match:
        mismatch = np.where(bulk_parallax[:cap] != per_src_parallax)[0]
        for j in mismatch:
            print(f'    source {j}: bulk={bulk_parallax[j]}, per-src={per_src_parallax[j]}')

    print(f'  Proper-motion filter decisions match: {"PASS" if pm_match else "FAIL"}')
    if not pm_match:
        mismatch = np.where(bulk_pm[:cap] != per_src_pm)[0]
        for j in mismatch:
            print(f'    source {j}: bulk={bulk_pm[j]}, per-src={per_src_pm[j]}')

    print()
    print(f'  Timing ({cap} sources):')
    print(f'    Bulk  (full field + match): {bulk_elapsed:.2f}s')
    print(f'    Per-source ({cap} sources): {per_src_elapsed:.2f}s  '
          f'(projected full field: {per_src_elapsed / cap * len(srcs):.1f}s)')
    if per_src_elapsed > 0:
        speedup = (per_src_elapsed / cap * len(srcs)) / bulk_elapsed
        print(f'    Estimated speedup: {speedup:.1f}×')

    all_pass = parallax_match and pm_match
    print(f'\n  Overall: {"ALL PASS" if all_pass else "FAILURES DETECTED"}')
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
