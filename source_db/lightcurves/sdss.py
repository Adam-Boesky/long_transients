"""SDSS PhotoObj photometry -> `Data/db/raw/sdss/`.

Not a light curve: SDSS contributes one PhotoObj row per source, so this is a
single epoch in five bands. It is stored in the same melted schema as the rest
so the pages can plot it alongside everything else.

Why not astroquery
------------------
`Light_Curve` calls `SDSS.query_crossid`, which posts to
`/en/tools/search/x_results.aspx`. That endpoint has been returning 500 for
days. `SkyServerWS/SearchTools/SqlSearch` is up and serves the same database,
so this module goes straight there with SQL.

The service is GET-only (POST 500s whatever the payload), so the batch size is
set by the IIS query-string cap: 200 positions is ~5.8 kB of URL, and the
service starts 404ing somewhere above 6 kB. 200 positions costs ~0.5 s, so all
169k sources take roughly 850 requests / ~10 minutes.

Match radius
------------
`query_crossid` defaults to 5" and `Light_Curve` never overrides it, which made
SDSS the only catalog here not matching at 1.5". Measured on 2,000 sources,
5" adds 4.5% more matches, all of them in the 1.5"-5" annulus -- chance
neighbours rather than the source (median separation inside 1.5" is 0.45").
This module uses 1.5" for consistency; `--radius-arcsec` restores the old
behaviour.
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

SURVEY = 'sdss'

DATA_RELEASE = 'dr18'
BASE_URL = f'https://skyserver.sdss.org/{DATA_RELEASE}/SkyServerWS/SearchTools/SqlSearch'

# Matches Source.max_arcsec and every other catalog here; see the module
# docstring for why this departs from query_crossid's 5" default.
MATCH_RADIUS_ARCSEC = 1.5

# Positions per GET. The service is GET-only and IIS caps the query string:
# 200 positions is ~5.8 kB and works, 250 (~7.1 kB) 404s.
BATCH_SIZE = 200

# Sources per resumable part file (one part = BATCH_SIZE-sized GETs).
CHUNK_SIZE = 20_000

MAX_ATTEMPTS = 4
RETRY_BACKOFF_S = 15

BANDS = ['u', 'g', 'r', 'i', 'z']

# Exactly the PhotoObj fields Light_Curve asks query_crossid for.
PHOTO_COLUMNS = ['objID', 'ra', 'dec', 'mjd'] + [
    f'{kind}Mag{err}_{b}'
    for b in BANDS for kind in ('psf', 'model') for err in ('', 'err')
]


def _select_list() -> str:
    cols = ['p.objID', 'p.ra', 'p.dec', 'p.mjd'] + [
        f'p.{c}' for c in PHOTO_COLUMNS if c not in ('objID', 'ra', 'dec', 'mjd')]
    return ','.join(cols)


def _sql(batch: pd.DataFrame, radius_arcsec: float) -> str:
    """One positional crossmatch for up to BATCH_SIZE positions.

    The row index travels as `s.i` rather than the uuid: a uuid costs 38 URL
    characters against the query-string cap and buys nothing the caller cannot
    recover by position.
    """
    vals = ','.join(f'({i},{ra:.6f},{dec:.6f})' for i, (ra, dec)
                    in enumerate(zip(batch['ra'], batch['dec'])))
    # fGetNearestObjEq takes ARCMINUTES. *Nearest*, not *nearby*: one row per
    # source, the same choice the PanSTARRS module makes for the same reason.
    return (f'SELECT s.i,{_select_list()},n.distance '
            f'FROM (VALUES {vals}) AS s(i,sra,sdec) '
            f'CROSS APPLY dbo.fGetNearestObjEq(s.sra,s.sdec,{radius_arcsec / 60.0:.6f}) AS n '
            f'JOIN PhotoObj p ON p.objID=n.objID')


def _get(sql: str) -> pd.DataFrame:
    """Run one SQL query, returning the (possibly empty) result frame."""
    r = requests.get(BASE_URL, params={'cmd': sql, 'format': 'csv'}, timeout=300)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    # The CSV is preceded by a '#Table1' banner line.
    body = '\n'.join(lines[1:]) if lines and lines[0].startswith('#') else r.text
    if not body.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(body))


def fetch_batch(batch: pd.DataFrame, radius_arcsec: float) -> pd.DataFrame:
    """Crossmatch one batch, retrying transient failures."""
    sql = _sql(batch, radius_arcsec)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            df = _get(sql)
            break
        except Exception as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            wait = RETRY_BACKOFF_S * 2 ** (attempt - 1)
            print(f'    batch attempt {attempt} failed ({exc!r}); retrying in {wait}s',
                  flush=True)
            time.sleep(wait)
    if df.empty:
        return df
    # `i` indexes into `batch`, so the uuid is recovered by position.
    df['source_uuid'] = batch['source_uuid'].to_numpy()[df['i'].to_numpy()]
    # fGetNearestObjEq reports `distance` in ARCMINUTES.
    df['sep_arcsec'] = df['distance'] * 60.0
    return df.drop(columns=['i', 'distance'])


def fetch(limit: Optional[int] = None, raw_dir: Path = RAW_DIR,
          chunk_size: int = CHUNK_SIZE, resume: bool = True,
          radius_arcsec: float = MATCH_RADIUS_ARCSEC) -> dict:
    """Fetch SDSS photometry for every source, in resumable chunks."""
    positions = _base.load_positions(limit=limit)
    chunks = _base.row_chunks(positions, chunk_size)
    # Coverage is checked against every fetched uuid, not per chunk id, so
    # adding sources later costs new chunks rather than a full refetch.
    todo = _base.pending_chunks(SURVEY, positions, chunk_size, raw_dir, resume)
    print(f'{SURVEY}: {len(positions):,} sources, {len(chunks)} chunks, '
          f'{len(todo)} to do (radius {radius_arcsec}")', flush=True)

    t0 = time.time()
    n_rows = 0
    failed = []
    for cid, sub in todo:
        t1 = time.time()
        sub = sub.reset_index(drop=True)
        parts = []
        try:
            for k in range(0, len(sub), BATCH_SIZE):
                parts.append(fetch_batch(sub.iloc[k:k + BATCH_SIZE].reset_index(drop=True),
                                         radius_arcsec))
        except Exception as exc:
            print(f'  chunk {cid}: giving up ({exc!r}); leaving uncovered', flush=True)
            failed.append(cid)
            continue
        parts = [p for p in parts if not p.empty]
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        _base.write_raw(SURVEY, cid, df, raw_dir=raw_dir, n_positions=len(sub))
        _base.write_coverage(SURVEY, cid, sub.source_uuid, raw_dir=raw_dir)
        n_rows += len(df)
        print(f'  chunk {cid}: {len(sub):,} positions -> {len(df):,} matches '
              f'[{time.time() - t1:.0f}s]', flush=True)

    dt = time.time() - t0
    if failed:
        print(f'{SURVEY}: {len(failed)} chunks failed ({", ".join(failed)}); '
              f'rerun to retry them', flush=True)
    print(f'{SURVEY}: {n_rows:,} matches in {dt / 60:.1f} min', flush=True)
    return {'survey': SURVEY, 'n_rows': n_rows, 'n_failed': len(failed), 'seconds': dt}


def read_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return _base.read_raw(SURVEY, raw_dir)


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Wide PhotoObj rows -> one row per (source, band).

    `mag` is modelMag, because that is what `Light_Curve.ALL_BAND_DF` maps to
    the SDSS bands ({band}Model_mag) and therefore what the pages plot. psfMag
    is carried alongside rather than dropped.

    SDSS sentinels non-detections as -9999; those become null.
    """
    if raw.empty:
        return _base.empty_lightcurves()
    frames = []
    for b in BANDS:
        sub = pd.DataFrame({
            'source_uuid': raw['source_uuid'].astype('string'),
            'survey': pd.array([SURVEY] * len(raw), dtype='string'),
            'band': pd.array([b] * len(raw), dtype='string'),
            'mjd': raw['mjd'].astype('float64'),
            'mag': raw[f'modelMag_{b}'].astype('float64'),
            'magerr': raw[f'modelMagerr_{b}'].astype('float64'),
            'snr': pd.Series([float('nan')] * len(raw), dtype='float64'),
            'catalog_id': raw['objID'].astype('string'),
        })
        sub['psf_mag'] = raw[f'psfMag_{b}'].astype('float64')
        sub['psf_magerr'] = raw[f'psfMagerr_{b}'].astype('float64')
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True)
    for c in ('mag', 'magerr', 'psf_mag', 'psf_magerr'):
        out.loc[out[c] <= -999, c] = float('nan')
    return out[out['mag'].notna()].reset_index(drop=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--raw-dir', type=Path, default=RAW_DIR)
    ap.add_argument('--radius-arcsec', type=float, default=MATCH_RADIUS_ARCSEC)
    ap.add_argument('--no-resume', action='store_true')
    a = ap.parse_args()
    fetch(limit=a.limit, raw_dir=a.raw_dir, resume=not a.no_resume,
          radius_arcsec=a.radius_arcsec)
