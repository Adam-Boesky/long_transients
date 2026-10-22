"""Which ZTF reference images exist, resolved ahead of time.

The mirror of `panstarrs.py`. PanSTARRS needs a lookup because a stack
filename is not derivable from a position; ZTF's address *is* derivable --
field, ccd and quadrant give the IBE path directly -- but whether a
reference image was ever built for a given band is not. Most fields have no
i-band, and the only way to find out used to be to render the stamp and see
whether IRSA returned anything, which meant availability was a side effect
of the render cache rather than data. A rebuild lost it.

Keyed by pointing, not by source: a reference image belongs to a
field/ccd/quadrant and ~8 sources share each one. IBE's search endpoint
answers a whole field at a time -- all 64 ccd/quadrant combinations and
their filters in one query -- so the 169k sources cost 380 queries rather
than half a million probes.

    python -m source_db.cutouts.ztf
"""
from __future__ import annotations

import base64
import concurrent.futures as cf
import io
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

from source_db.lightcurves import _base
from source_db.paths import RAW_DIR

CATALOG = 'ztf_cutouts'

SEARCH = 'https://irsa.ipac.caltech.edu/ibe/search/ztf/products/deep'
# IRSA's filtercode -> the band the app asks for.
BAND_OF = {'zg': 'g', 'zr': 'r', 'zi': 'i'}

# Fields per resumable part file. Small because a field is ~450 pointings.
CHUNK_SIZE = 50
WORKERS = 8
MAX_ATTEMPTS = 4
RETRY_BACKOFF_S = 10
TIMEOUT_S = 300


def _auth() -> str:
    from Extracting.utils import get_credentials
    user, password = get_credentials('irsa_login.txt')
    return 'Basic ' + base64.b64encode(f'{user}:{password}'.encode()).decode()


def fetch_field(field: int, auth: str) -> pd.DataFrame:
    """Every (ccd, quadrant, band) with a reference image in one field."""
    url = f'{SEARCH}?WHERE=field={int(field)}&ct=csv'
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(url)
            req.add_header('Authorization', auth)
            text = urllib.request.urlopen(req, timeout=TIMEOUT_S).read().decode()
            break
        except urllib.error.HTTPError as e:
            # A field with no deep products at all is an empty answer, not a
            # failure worth retrying.
            if e.code == 404:
                return pd.DataFrame(columns=['fieldid', 'ccdid', 'qid', 'band'])
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(RETRY_BACKOFF_S * 2 ** (attempt - 1))
        except Exception:
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(RETRY_BACKOFF_S * 2 ** (attempt - 1))

    if not text.strip():
        return pd.DataFrame(columns=['fieldid', 'ccdid', 'qid', 'band'])
    df = pd.read_csv(io.StringIO(text))
    need = {'field', 'ccdid', 'qid', 'filtercode'}
    if not need <= set(df.columns):
        return pd.DataFrame(columns=['fieldid', 'ccdid', 'qid', 'band'])
    df = df[list(need)].dropna()
    df['band'] = df['filtercode'].map(BAND_OF)
    df = df.dropna(subset=['band'])
    out = df.rename(columns={'field': 'fieldid'})[['fieldid', 'ccdid', 'qid', 'band']]
    return out.astype({'fieldid': 'int64', 'ccdid': 'int64', 'qid': 'int64'}) \
              .drop_duplicates().reset_index(drop=True)


def fetch_chunk(fields: list[int], auth: str, workers: int = WORKERS) -> pd.DataFrame:
    frames = []
    with cf.ThreadPoolExecutor(workers) as ex:
        for df in ex.map(lambda f: fetch_field(f, auth), fields):
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['fieldid', 'ccdid', 'qid', 'band'])
    return pd.concat(frames, ignore_index=True)


def pointings(limit: Optional[int] = None) -> pd.DataFrame:
    """The distinct field/ccd/quadrant the catalog actually uses."""
    df = _base.load_positions(limit=limit)
    cols = ['fieldid', 'ccdid', 'qid']
    if not set(cols) <= set(df.columns):
        from source_db.paths import DEFAULT_OUT_DIR
        df = pd.read_parquet(DEFAULT_OUT_DIR / 'sources.parquet',
                             columns=['source_uuid'] + cols)
        if limit:
            df = df.head(limit)
    return df[cols].dropna().astype('int64').drop_duplicates()


def fetch(limit: Optional[int] = None, raw_dir: Path = RAW_DIR,
          chunk_size: int = CHUNK_SIZE, workers: int = WORKERS,
          resume: bool = True) -> dict:
    pts = pointings(limit)
    fields = sorted(pts['fieldid'].unique().tolist())
    print(f'{CATALOG}: {len(pts):,} pointings across {len(fields)} fields',
          flush=True)

    done = _base.completed_chunks(CATALOG, raw_dir) if resume else set()
    t0, n_rows = time.time(), 0
    auth = _auth()
    for k in range(0, len(fields), chunk_size):
        cid = f'{k // chunk_size:04d}'
        part = fields[k:k + chunk_size]
        if cid in done:
            continue
        t1 = time.time()
        got = fetch_chunk(part, auth, workers)
        # Stored whole, not filtered to the pointings the catalog currently
        # uses. Filtering here looked harmless -- a field carries 64
        # quadrants and most have sources -- but the resume key is the
        # *field*, so a later catalog that puts a source on a quadrant we
        # discarded would skip the field as done and never learn its bands.
        # That is exactly what happened: 69 pointings across 51 crawled
        # fields came back empty. The whole answer costs ~2% more rows.
        _base.write_raw(CATALOG, cid, got, raw_dir=raw_dir, n_positions=len(part))
        # Coverage is the *fields* asked for, not sources: availability is a
        # property of the pointing. Without this sidecar `completed_chunks`
        # sees nothing done and the whole crawl repeats on every run.
        _base.write_coverage(CATALOG, cid, [str(f) for f in part], raw_dir=raw_dir)
        n_rows += len(got)
        print(f'  chunk {cid}: {len(part)} fields -> {len(got):,} '
              f'pointing-bands [{time.time() - t1:.0f}s]', flush=True)

    dt = time.time() - t0
    print(f'{CATALOG}: {n_rows:,} pointing-bands in {dt / 60:.1f} min', flush=True)
    return {'catalog': CATALOG, 'n_rows': n_rows, 'seconds': dt}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    ap.add_argument('--workers', type=int, default=WORKERS)
    ap.add_argument('--raw-dir', type=Path, default=RAW_DIR)
    ap.add_argument('--no-resume', action='store_true')
    a = ap.parse_args()
    fetch(limit=a.limit, raw_dir=a.raw_dir, chunk_size=a.chunk_size,
          workers=a.workers, resume=not a.no_resume)
