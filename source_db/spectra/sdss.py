"""SDSS DR20 spectra -> `Data/db/raw/sdss_spectra/`.

No queries: `sources.parquet` already carries `sdss_spec_url`, the allspec SAS
URL of each matched spectrum (SDSS-I--IV and SDSS-V alike), so the fetch is
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

from source_db.paths import DEFAULT_OUT_DIR, RAW_DIR
from source_db.spectra import _base

SURVEY = 'sdss'
CATALOG = 'sdss_spectra'


# Spectra per resumable part file.
CHUNK_SIZE = 2_000

# Modest: this is a shared archive and we are one user. Also keeps the fetch
# from competing with the light-curve pulls for bandwidth.
N_THREADS = 8
TIMEOUT_S = 120
MAX_ATTEMPTS = 3
RETRY_BACKOFF_S = 5


def targets(out_dir: Path = DEFAULT_OUT_DIR, limit: Optional[int] = None) -> pd.DataFrame:
    """One row per (source, SDSS spectrum): every spectrum the crossmatch saw.

    The catalog tier records only that a source matched; its spectra are
    re-listed here from the same catalog at the same radius, since one source
    can have hundreds.
    """
    from Source_Analysis.catalogs import sdss as sdss_catalog
    from Source_Analysis.enrich_combined_tabs import MATCH_RADIUS_ARCSEC

    src = pd.read_parquet(Path(out_dir) / 'sources.parquet',
                          columns=['source_uuid', 'ra', 'dec', 'sdss_matched'])
    src = src[src['sdss_matched'].fillna(False).astype(bool)].reset_index(drop=True)
    if limit:
        src = src.head(limit)
    spec = sdss_catalog.spectra_within_many(src['ra'], src['dec'], MATCH_RADIUS_ARCSEC)
    spec['source_uuid'] = src['source_uuid'].to_numpy()[spec['input_index'].to_numpy()]
    return spec[['source_uuid', 'sdss_specobjid', 'sdss_spec_url', 'sdss_class',
                 'sdss_z', 'sdss_mjd']].reset_index(drop=True)


def spectrum_url(row) -> Optional[str]:
    """The SAS `lite` coadd URL, or None if the source has none."""
    url = row.get('sdss_spec_url')
    return None if url is None or pd.isna(url) or not url else str(url)


def _fetch_one(row, session: requests.Session) -> Optional[dict]:
    from astropy.io import fits

    url = spectrum_url(row)
    if url is None:
        return None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = session.get(url, timeout=TIMEOUT_S)
            if resp.status_code == 404:
                # A matched catalog row whose coadd is absent from the lite
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
        # HDU1 carries loglam/flux/model (upper case in SDSS-V files; astropy
        # field lookup ignores case); convert to linear Angstroms so the
        # frontend sees one wavelength convention across surveys.
        wave = 10.0 ** np.asarray(d['loglam'], dtype='float64')
        flux = np.asarray(d['flux'])
        has_model = 'model' in (n.lower() for n in d.names)
        model = np.asarray(d['model']) if has_model else None

    return _base.spectrum_row(
        source_uuid=row['source_uuid'], survey=SURVEY,
        spec_id=str(row['sdss_specobjid']),
        wavelength=wave, flux=flux, model=model,
        spec_class=row.get('sdss_class'), redshift=row.get('sdss_z'),
        mjd=row.get('sdss_mjd'))


def coverage_keys(targets: pd.DataFrame) -> pd.Series:
    """`uuid|specobjid`: what one fetch covered."""
    return targets['source_uuid'].astype(str) + '|' + targets['sdss_specobjid'].astype(str)


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
    from source_db.lightcurves._base import all_covered, next_chunk_index

    targets = _base.load_targets(SURVEY, limit=limit)
    # Coverage is keyed by source *and* spectrum, so a source whose match
    # moves to another spectrum (a catalog rebuild) is fetched again.
    keys = coverage_keys(targets)
    done = all_covered(CATALOG, raw_dir) if resume else set()
    pending = targets[~keys.isin(done)].reset_index(drop=True)
    start = next_chunk_index(CATALOG, raw_dir) if resume else 0
    todo = [(f'{start + i // chunk_size:04d}', pending.iloc[i:i + chunk_size])
            for i in range(0, len(pending), chunk_size)]
    print(f'{CATALOG}: {len(targets):,} SDSS spectra of '
          f'{targets["source_uuid"].nunique():,} sources, '
          f'{len(pending):,} to fetch in {len(todo)} chunks', flush=True)

    t0, n_spec, n_failed = time.time(), 0, 0
    for cid, sub in todo:
        t1 = time.time()
        df, failed = fetch_chunk(sub, n_threads)
        _base.write_raw(CATALOG, cid, df, raw_dir=raw_dir, n_targets=len(sub))
        _base.write_coverage(CATALOG, cid, coverage_keys(sub), raw_dir=raw_dir)
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
