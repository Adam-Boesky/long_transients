"""Each source's W1-W2 mean and median, for the website's filters.

    python -m source_db.build_wise_colors

Writes `Data/db/wise_colors.parquet`:

    source_uuid, wise_w1w2_mean, wise_w1w2_median

These are the two numbers on an analysis page's W1-W2 histogram, computed the
same way: the WISE and NEOWISE epochs are shaped by `local_store`, visit-binned
by `Light_Curve._finish_catalog_lc`, stacked, and cut to sigma(W1-W2) < 0.5 by
`wise_colors`. Every step is the page's own code, not a copy of it. Distinct
from the catalog's `wise_w1w2`, which is the single AllWISE colour.

Sources with no W1/W2 epoch passing the cut are left out, so they read as
missing in the filter rather than as zero.
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from astropy.table import Table, vstack

from Source_Analysis.Light_Curve import Light_Curve, wise_colors
from Source_Analysis.local_store import shape_catalog_lc
from source_db.paths import DEFAULT_OUT_DIR

SURVEYS = ('wise', 'neowise')
OUT_NAME = 'wise_colors.parquet'

# `_finish_catalog_lc` reads no instance state, so a bare instance will do.
_LC = Light_Curve.__new__(Light_Curve)


def colors_of(uuid: str, rows: pd.DataFrame) -> tuple[float, float] | None:
    """(mean, median) W1-W2 for one source's WISE+NEOWISE rows, as the page has them."""
    lcs = []
    for survey in SURVEYS:
        r = rows[rows['survey'] == survey].sort_values('mjd').reset_index(drop=True)
        if len(r):
            lc = _LC._finish_catalog_lc(survey, shape_catalog_lc(r, survey, uuid))
            if len(lc):
                lcs.append(lc)
    if not lcs:
        return None
    # As `Light_Curve.get_lc` joins catalogs. Masked explicitly: on the page the
    # other catalogs in the stack are what make these columns masked.
    lc = Table(vstack(lcs), masked=True)
    d = wise_colors(lc)
    if d is None:
        return None
    d = np.ma.filled(np.ma.asarray(d, dtype=float), np.nan)
    d = d[np.isfinite(d)]
    return (float(np.mean(d)), float(np.median(d))) if len(d) else None


def _partition(path: str) -> list[tuple[str, float, float]]:
    rows = duckdb.connect().execute(
        f"SELECT source_uuid, survey, band, mjd, mag, magerr, snr "
        f"FROM read_parquet('{path}') WHERE survey IN {SURVEYS}").fetchdf()
    out = []
    for uuid, group in rows.groupby('source_uuid', sort=False):
        stats = colors_of(uuid, group)
        if stats is not None:
            out.append((uuid, *stats))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument('--processes', type=int, default=8)
    args = ap.parse_args()

    parts = sorted(str(p) for p in (args.out_dir / 'light_curves').glob('uuid_prefix=*/part-0.parquet'))
    t0 = time.time()
    with Pool(args.processes) as pool:
        results = [r for part in pool.imap_unordered(_partition, parts) for r in part]
    frame = pd.DataFrame(results, columns=['source_uuid', 'wise_w1w2_mean', 'wise_w1w2_median'])
    frame.to_parquet(args.out_dir / OUT_NAME, index=False)
    print(f'{len(frame):,} sources -> {args.out_dir / OUT_NAME} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
