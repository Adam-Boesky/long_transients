"""PanSTARRS stack-image filenames -> `Data/db/raw/ps1_cutouts/`.

Why precompute at all
---------------------
A PS1 cutout URL needs the stack filename for the sky cell the source falls
in, which comes from `ps1filenames.py`. Measured at 0.74 s per source -- far
too slow to do while someone waits for a page. The filename depends only on
position, and our positions never move, so it is resolved once here and the
page is handed a ready `<img src>`.

The service also accepts a POSTed list of positions: 100 at a time returns in
~1.0 s (~98 positions/s), so the whole catalog is about half an hour rather
than the 35 hours a per-source loop would take.

Only `g`, `r`, `i` are fetched -- the three bands the analysis pages show.
"""
from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from source_db.lightcurves import _base
from source_db.paths import RAW_DIR

CATALOG = 'ps1_cutouts'

SERVICE = 'https://ps1images.stsci.edu/cgi-bin/ps1filenames.py'
BANDS = 'gri'

# Positions per POST. 100 measured at ~1.0 s; larger batches risk a timeout
# on a service that is not documented to have a limit.
BATCH_SIZE = 100
# Sources per resumable part file.
CHUNK_SIZE = 20_000

MAX_ATTEMPTS = 4
RETRY_BACKOFF_S = 10
TIMEOUT_S = 180


def _post(positions: pd.DataFrame) -> pd.DataFrame:
    payload = '\n'.join(f'{ra} {dec}' for ra, dec
                        in zip(positions['ra'], positions['dec']))
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = requests.post(SERVICE, data={'filters': BANDS, 'type': 'stack'},
                              files={'file': ('pos.txt', payload)},
                              timeout=TIMEOUT_S)
            r.raise_for_status()
            break
        except Exception as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            wait = RETRY_BACKOFF_S * 2 ** (attempt - 1)
            print(f'    attempt {attempt} failed ({exc!r}); retrying in {wait}s',
                  flush=True)
            time.sleep(wait)
    text = r.text.strip()
    if not text:
        return pd.DataFrame(columns=['ra', 'dec', 'filter', 'filename'])
    return pd.read_csv(io.StringIO(text), sep=r'\s+')


def fetch_chunk(sub: pd.DataFrame, batch_size: int = BATCH_SIZE) -> pd.DataFrame:
    """One row per (source, band) with the stack filename.

    The service echoes the *input* ra/dec it was given, so results are matched
    back by position rather than by row order -- it returns one row per band
    and drops positions with no coverage, so the ordering is not one-to-one.
    """
    rows = []
    for k in range(0, len(sub), batch_size):
        part = sub.iloc[k:k + batch_size]
        got = _post(part)
        if got.empty:
            continue
        # Key on the exact float the service was handed.
        by_pos = {(round(float(ra), 6), round(float(dec), 6)): uuid
                  for ra, dec, uuid in zip(part['ra'], part['dec'],
                                           part['source_uuid'])}
        for rec in got.itertuples(index=False):
            uuid = by_pos.get((round(float(rec.ra), 6), round(float(rec.dec), 6)))
            if uuid is None:
                continue
            rows.append({'source_uuid': uuid, 'band': str(rec.filter),
                         'filename': str(rec.filename)})
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=['source_uuid', 'band', 'filename'])


def fetch(limit: Optional[int] = None, raw_dir: Path = RAW_DIR,
          chunk_size: int = CHUNK_SIZE, batch_size: int = BATCH_SIZE,
          resume: bool = True) -> dict:
    positions = _base.load_positions(limit=limit)
    todo = _base.pending_chunks(CATALOG, positions, chunk_size, raw_dir, resume)
    print(f'{CATALOG}: {len(positions):,} sources, {len(todo)} chunks to do',
          flush=True)

    t0, n_rows = time.time(), 0
    for cid, sub in todo:
        t1 = time.time()
        df = fetch_chunk(sub.reset_index(drop=True), batch_size)
        _base.write_raw(CATALOG, cid, df, raw_dir=raw_dir, n_positions=len(sub))
        _base.write_coverage(CATALOG, cid, sub.source_uuid, raw_dir=raw_dir)
        n_rows += len(df)
        print(f'  chunk {cid}: {len(sub):,} positions -> {len(df):,} band-images '
              f'[{time.time() - t1:.0f}s]', flush=True)

    dt = time.time() - t0
    print(f'{CATALOG}: {n_rows:,} band-images in {dt / 60:.1f} min', flush=True)
    return {'catalog': CATALOG, 'n_rows': n_rows, 'seconds': dt}


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
