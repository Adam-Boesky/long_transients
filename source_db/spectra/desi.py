"""DESI DR1 spectra -> `Data/db/raw/desi_spectra/`.

`sources.parquet` already carries `desi_targetid` for every matched source, so
this skips the position search that `Sources.desi_spectrum` has to do
per-source (SPARCL's `find()` supports only range constraints, so that path
boxes around the source and post-filters by separation). Here the crossmatch
is already done and `retrieve_by_specid` takes a list, which turns 42,500
individual searches into a few hundred batch retrievals.

Records come back unordered, so they are mapped back by `specid`.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from source_db.paths import RAW_DIR
from source_db.spectra import _base

SURVEY = 'desi'
CATALOG = 'desi_spectra'

DATASET = 'DESI-DR1'

# Spectra per SPARCL call. Each DESI spectrum is ~7,800 pixels x 3 arrays, so
# 100 is ~9 MB on the wire -- large enough to amortise the round trip, small
# enough that one failure is cheap to retry.
BATCH_SIZE = 100

# Sources per resumable part file.
CHUNK_SIZE = 2_000

MAX_ATTEMPTS = 3
RETRY_BACKOFF_S = 10

INCLUDE = ['specid', 'ra', 'dec', 'wavelength', 'flux', 'model',
           'spectype', 'redshift']

_client = None


def _get_client():
    global _client
    if _client is None:
        from sparcl.client import SparclClient
        _client = SparclClient()
    return _client


def _retrieve(specids: list[int]):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _get_client().retrieve_by_specid(
                specid_list=specids, dataset_list=[DATASET], include=INCLUDE)
        except Exception as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            print(f'    sparcl attempt {attempt} failed ({exc!r}); retrying',
                  flush=True)
            time.sleep(RETRY_BACKOFF_S * 2 ** (attempt - 1))


def fetch_chunk(sub: pd.DataFrame, batch_size: int = BATCH_SIZE) -> pd.DataFrame:
    by_specid = {int(r['desi_targetid']): r for _, r in sub.iterrows()
                 if not pd.isna(r['desi_targetid'])}
    specids = list(by_specid)
    rows = []
    # SPARCL can hand back several records for one specid -- near-identical
    # coadds of the same spectrum, differing in redshift at the 1e-4 level.
    # Keeping them all put 2-4 copies of one spectrum on the page. Take the
    # first per specid; the serving build dedups again in case this misses.
    seen_specids: set[int] = set()
    for i in range(0, len(specids), batch_size):
        got = _retrieve(specids[i:i + batch_size])
        for rec in got.records:
            tgt = by_specid.get(int(rec.specid))
            if tgt is None:
                # SPARCL returned a specid we did not ask for; ignore rather
                # than guess which source it belongs to.
                continue
            if int(rec.specid) in seen_specids:
                continue
            seen_specids.add(int(rec.specid))
            rows.append(_base.spectrum_row(
                source_uuid=tgt['source_uuid'], survey=SURVEY,
                spec_id=int(rec.specid),
                wavelength=rec.wavelength, flux=rec.flux,
                model=getattr(rec, 'model', None),
                spec_class=getattr(rec, 'spectype', None),
                redshift=getattr(rec, 'redshift', None)))
    return pd.DataFrame(rows) if rows else _base.empty_spectra()


def fetch(limit: Optional[int] = None, raw_dir: Path = RAW_DIR,
          chunk_size: int = CHUNK_SIZE, batch_size: int = BATCH_SIZE,
          resume: bool = True) -> dict:
    from source_db.lightcurves._base import pending_chunks, row_chunks

    targets = _base.load_targets(SURVEY, limit=limit)
    chunks = row_chunks(targets, chunk_size)
    # Coverage is checked against every fetched uuid, not per chunk id,
    # so adding sources later costs new chunks rather than a refetch.
    todo = pending_chunks(CATALOG, targets, chunk_size, raw_dir, resume)
    print(f'{CATALOG}: {len(targets):,} sources with DESI spectra, '
          f'{len(chunks)} chunks, {len(todo)} to do', flush=True)

    t0, n_spec, failed = time.time(), 0, []
    for cid, sub in todo:
        t1 = time.time()
        try:
            df = fetch_chunk(sub, batch_size)
        except Exception as exc:
            print(f'  chunk {cid}: giving up ({exc!r}); leaving uncovered',
                  flush=True)
            failed.append(cid)
            continue
        _base.write_raw(CATALOG, cid, df, raw_dir=raw_dir, n_targets=len(sub))
        _base.write_coverage(CATALOG, cid, sub.source_uuid, raw_dir=raw_dir)
        n_spec += len(df)
        print(f'  chunk {cid}: {len(sub):,} targets -> {len(df):,} spectra '
              f'[{time.time() - t1:.0f}s]', flush=True)

    dt = time.time() - t0
    if failed:
        print(f'{CATALOG}: {len(failed)} chunks failed ({", ".join(failed)}); '
              f'rerun to retry', flush=True)
    print(f'{CATALOG}: {n_spec:,} spectra in {dt / 60:.1f} min', flush=True)
    return {'survey': CATALOG, 'n_spectra': n_spec, 'n_failed': len(failed),
            'seconds': dt}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    ap.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    ap.add_argument('--raw-dir', type=Path, default=RAW_DIR)
    ap.add_argument('--no-resume', action='store_true')
    a = ap.parse_args()
    fetch(limit=a.limit, raw_dir=a.raw_dir, chunk_size=a.chunk_size,
          batch_size=a.batch_size, resume=not a.no_resume)
