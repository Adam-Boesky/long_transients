"""Normalise the raw per-catalog pulls into the bucketed light-curve tier.

Pure function of `Data/db/raw/`, so it is freely re-runnable: a bug in the melt
logic costs minutes rather than another overnight backfill.

    python -m source_db.build_lightcurve_table
    python -m source_db.build_lightcurve_table --catalogs ztf

Writes `Data/db/light_curves/uuid_prefix=XX/part-0.parquet`:

    source_uuid, survey, band, mjd, mag, magerr, snr, catalog_id
    + every survey's extras (null where a survey does not emit them)

Why bucketed, and why streaming
-------------------------------
The combined table is ~286M rows / ~7.5 GB -- ZTF alone is ~264M epochs, not
the ~4.9M an early estimate assumed. Concatenating that in pandas peaks around
64 GB, so this build never holds more than one batch: raw parts are read in
groups, normalised, and appended straight to the bucket they belong to.

This tier is for analysis. The website reads per-source payloads built by
`source_db.serving.build_serving_db`, never these files.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from source_db.lightcurves import _base  # noqa: E402
from source_db.paths import DEFAULT_OUT_DIR, RAW_DIR  # noqa: E402

# wise/neowise/ptf/gaia all come from the one IRSA module, which takes the
# survey name because its melt differs per catalog (Gaia is a mean flux at a
# reference epoch, PTF a time series, WISE/NEOWISE one row per band per epoch).
_IRSA_SURVEYS = ('wise', 'neowise', 'ptf', 'gaia')


def normalise_batch(name: str, raw: pd.DataFrame) -> pd.DataFrame:
    """Melt one batch of a catalog's raw rows to the shared schema.

    Every normaliser is row-independent, so a batch melts exactly as the whole
    catalog would.
    """
    if name in _IRSA_SURVEYS:
        from source_db.lightcurves import irsa
        return irsa.normalise(name, raw)
    if name == 'ztf':
        from source_db.lightcurves import ztf
        return ztf.normalise(raw)
    if name == 'panstarrs':
        from source_db.lightcurves import panstarrs
        return panstarrs.normalise(raw)
    if name == 'custom':
        from source_db.lightcurves import custom
        return custom.normalise(raw)
    if name == 'sdss':
        from source_db.lightcurves import sdss
        return sdss.normalise(raw)
    raise NotImplementedError(f'no normaliser for {name!r} yet')


def build(catalogs: list[str], raw_dir: Path, out_dir: Path,
          parts_per_batch: int = 40) -> dict:
    """Stream every catalog into the bucketed tier."""
    totals: dict[str, int] = {}
    t0 = time.time()
    with _base.BucketedWriter(out_dir) as writer:
        for name in catalogs:
            n_parts = _base.n_raw_parts(name, raw_dir)
            t1, rows, done = time.time(), 0, 0
            for raw in _base.iter_raw_batches(name, raw_dir, parts_per_batch):
                df = normalise_batch(name, raw)
                writer.write(df)
                rows += len(df)
                done += parts_per_batch
                del raw, df
                print(f'    {name} {min(done, n_parts):,}/{n_parts:,} parts  '
                      f'{rows:,} rows  {(time.time() - t1) / 60:.1f} min',
                      end='\r', flush=True)
            totals[name] = rows
            print(f'  {name:10} {rows:>13,} rows  '
                  f'{(time.time() - t1) / 60:>6.1f} min'.ljust(70), flush=True)
    print(f'\ntotal {sum(totals.values()):,} rows into '
          f'{_base.lightcurve_dir(out_dir)} in {(time.time() - t0) / 60:.1f} min')
    return totals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--catalogs', nargs='+', default=None,
                    help=f'subset of {_base.CATALOGS} (default: whatever is in raw/)')
    ap.add_argument('--raw-dir', type=Path, default=RAW_DIR)
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument('--parts-per-batch', type=int, default=40,
                    help='raw parts normalised at once; lower it if memory is tight')
    args = ap.parse_args()

    available = [c for c in _base.CATALOGS
                 if _base.n_raw_parts(c, args.raw_dir) > 0]
    wanted = args.catalogs or available
    missing = [c for c in wanted if c not in available]
    if missing:
        sys.exit(f'no raw data for {missing} -- fetch it first. '
                 f'Available: {available or "(none)"}')

    build(wanted, args.raw_dir, args.out_dir, args.parts_per_batch)


if __name__ == '__main__':
    main()
