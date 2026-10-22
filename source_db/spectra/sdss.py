"""SDSS DR17 spectra -> `Data/db/raw/sdss_spectra/`.

No queries: `sources.parquet` already carries plate/mjd/fiberid/run2d for
every matched source, so the SAS file URL is built offline and the fetch is
just parallel HTTP GETs of the `lite` coadds.

Straight to the Science Archive Server rather than through astroquery's
SkyServer endpoint, for the reason `Sources.spectrum` documents: same data
release and bit-identical files, but SkyServer is frequently unreachable
(it is returning 500s for x_results.aspx as of this build) and the SAS is not.
"""
from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from source_db.paths import RAW_DIR
from source_db.spectra import _base

SURVEY = 'sdss'
CATALOG = 'sdss_spectra'

SAS_BASE = 'https://data.sdss.org/sas/dr17/sdss/spectro/redux'

# Sources per resumable part file.
CHUNK_SIZE = 2_000

# Modest: this is a shared archive and we are one user. Also keeps the fetch
# from competing with the light-curve pulls for bandwidth.
N_THREADS = 8
TIMEOUT_S = 120
MAX_ATTEMPTS = 3
RETRY_BACKOFF_S = 5


def spectrum_url(row) -> Optional[str]:
    """The SAS `lite` coadd URL, or None if the identifiers are incomplete."""
    plate, mjd, fiber, run2d = (row['sdss_plate'], row['sdss_mjd'],
                                row['sdss_fiberid'], row['sdss_run2d'])
    if pd.isna(plate) or pd.isna(mjd) or pd.isna(fiber) or pd.isna(run2d):
        return None
    return (f'{SAS_BASE}/{run2d}/spectra/lite/{int(plate):04d}/'
            f'spec-{int(plate):04d}-{int(mjd)}-{int(fiber):04d}.fits')


def _fetch_one(row, session: requests.Session) -> Optional[dict]:
    from astropy.io import fits

    url = spectrum_url(row)
    if url is None:
        return None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = session.get(url, timeout=TIMEOUT_S)
            if resp.status_code == 404:
                # A matched specObj row whose coadd is absent from the lite
                # tree; real and not worth retrying.
                return None
            resp.raise_for_status()
            break
        except Exception:
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(RETRY_BACKOFF_S * 2 ** (attempt - 1))

    with fits.open(io.BytesIO(resp.content), memmap=False) as hdul:
        d = hdul[1].data
        # HDU1 carries loglam/flux/model; convert to linear Angstroms so the
        # frontend sees one wavelength convention across surveys.
        wave = 10.0 ** np.asarray(d['loglam'], dtype='float64')
        flux = np.asarray(d['flux'])
        model = np.asarray(d['model']) if 'model' in d.names else None

    return _base.spectrum_row(
        source_uuid=row['source_uuid'], survey=SURVEY,
        spec_id=f"{int(row['sdss_plate'])}-{int(row['sdss_mjd'])}-{int(row['sdss_fiberid'])}",
        wavelength=wave, flux=flux, model=model,
        spec_class=row.get('sdss_class'), redshift=row.get('sdss_z'))


def fetch_chunk(sub: pd.DataFrame, n_threads: int = N_THREADS) -> tuple[pd.DataFrame, int]:
    rows, failed = [], 0
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = {ex.submit(_fetch_one, r, session): r['source_uuid']
                       for _, r in sub.iterrows()}
            for fut in as_completed(futures):
                try:
                    row = fut.result()
                except Exception as exc:
                    failed += 1
                    print(f'    {futures[fut][:8]} failed: {exc!r}', flush=True)
                    continue
                if row is not None:
                    rows.append(row)
    return (pd.DataFrame(rows) if rows else _base.empty_spectra()), failed


def fetch(limit: Optional[int] = None, raw_dir: Path = RAW_DIR,
          chunk_size: int = CHUNK_SIZE, n_threads: int = N_THREADS,
          resume: bool = True) -> dict:
    from source_db.lightcurves._base import pending_chunks, row_chunks

    targets = _base.load_targets(SURVEY, limit=limit)
    chunks = row_chunks(targets, chunk_size)
    # Coverage is checked against every fetched uuid, not per chunk id,
    # so adding sources later costs new chunks rather than a refetch.
    todo = pending_chunks(CATALOG, targets, chunk_size, raw_dir, resume)
    print(f'{CATALOG}: {len(targets):,} sources with SDSS spectra, '
          f'{len(chunks)} chunks, {len(todo)} to do', flush=True)

    t0, n_spec, n_failed = time.time(), 0, 0
    for cid, sub in todo:
        t1 = time.time()
        df, failed = fetch_chunk(sub, n_threads)
        _base.write_raw(CATALOG, cid, df, raw_dir=raw_dir, n_targets=len(sub))
        _base.write_coverage(CATALOG, cid, sub.source_uuid, raw_dir=raw_dir)
        n_spec += len(df)
        n_failed += failed
        print(f'  chunk {cid}: {len(sub):,} targets -> {len(df):,} spectra '
              f'({failed} failed) [{time.time() - t1:.0f}s]', flush=True)

    dt = time.time() - t0
    print(f'{CATALOG}: {n_spec:,} spectra, {n_failed} failures, '
          f'{dt / 60:.1f} min', flush=True)
    return {'survey': CATALOG, 'n_spectra': n_spec, 'n_failed': n_failed,
            'seconds': dt}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    ap.add_argument('--threads', type=int, default=N_THREADS)
    ap.add_argument('--raw-dir', type=Path, default=RAW_DIR)
    ap.add_argument('--no-resume', action='store_true')
    a = ap.parse_args()
    fetch(limit=a.limit, raw_dir=a.raw_dir, chunk_size=a.chunk_size,
          n_threads=a.threads, resume=not a.no_resume)
