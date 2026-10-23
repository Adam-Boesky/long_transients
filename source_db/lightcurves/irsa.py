"""WISE, NEOWISE, PTF and Gaia light curves, in bulk, via IRSA async TAP.

Four catalogs, one mechanism
----------------------------
All four live in IRSA's TAP service and all four are queried the same way:
upload our positions as a table and let the server do the cone join, instead
of one HTTP round trip per source. `Light_Curve.get_lc` does the latter --
169,052 sources x 4 catalogs would be ~676,000 requests.

    sync   times out after 5 minutes
    async  allows 3 days  ->  https://irsa.ipac.caltech.edu/TAP/async

So every query here is async. IRSA documents table upload for exactly this:
"IRSA's TAP offers the option of uploading a table containing multiple
positions so that the user can search multiple sky regions in a single query."

Keyed by source_uuid
--------------------
The uploaded table carries `source_uuid` alongside ra/dec, so results come
back already attached to our sources. Without it every result row would have
to be re-crossmatched against the source list -- a second chance to
mis-associate, for no benefit.

Chunking
--------
Positions go up in chunks (default 25k). This is not for memory -- it is so a
failed or throttled job costs one chunk rather than the whole catalog, and so
`raw/` fills incrementally. Coverage is recorded per chunk, so adding sources
later only re-fetches the chunks that gained them.

Column sets come from `Light_Curve`
-----------------------------------
Each catalog's `desired_colnames` is copied from the corresponding branch of
`Light_Curve.get_catalog_lc`, so the stored light curves are the same
measurements the pages already plot. Where that code renames a column on the
way out (`obsmjd` -> `mjd`, `source_id_mf` -> `wise_id`), the rename happens
in `normalise`, not at fetch time: `raw/` keeps the service's own names.

WISE and NEOWISE are stored UNBINNED -- bin at render time
-----------------------------------------------------------
`Light_Curve` does not plot raw WISE/NEOWISE exposures. It groups them into
*visits* first, via `_cluster_by_gap(mjds, max_gap=5.0)`: WISE revisits a
position several times over a couple of days, and those exposures are stacked
into one point. The effect is large --

    wise     23 raw exposures (MJD 55300-55492)  ->   2 visits
    neowise  95 raw exposures                    ->  20 visits

-- and it is the entire difference between this module's output and
`Light_Curve.get_lc()`. The underlying queries are identical: astroquery's
cone and this TAP upload both return 23 rows / 1 object at 1.5", 3" and 6".
PTF and Gaia get no clustering and match exactly.

We store the raw exposures deliberately. Binning is lossy and parameterised
(`max_gap=5.0`), so baking it in would freeze that choice and make the
individual exposures unrecoverable without re-fetching.

**Anything rendering WISE or NEOWISE from this database must apply
`Light_Curve._cluster_by_gap(max_gap=5.0)` to match the current pages.**
This is the same kind of consumer-side step as ZTF's `catflags & 6141` cut.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source_db.lightcurves import _base  # noqa: E402
from source_db.paths import RAW_DIR  # noqa: E402

TAP_URL = 'https://irsa.ipac.caltech.edu/TAP'

# Matches Source.max_arcsec / enrich_combined_tabs.MATCH_RADIUS_ARCSEC.
MATCH_RADIUS_ARCSEC = 1.5
CHUNK_SIZE = 25_000

# IRSA 5xxs under load; a NEOWISE chunk is ~75 min, so retry before losing it.
MAX_ATTEMPTS = 4
RETRY_BACKOFF_S = 60

# How long a single poll blocks, and how long one chunk may take overall.
# `run_async` would block indefinitely; bounding the poll is what turns a
# dropped connection into a retry rather than a silent hang.
POLL_SECONDS = 60.0
MAX_WAIT_SECONDS = 3 * 60 * 60

# Gaia DR3 zero points, Table 5.4 of the GDR3 photometry documentation.
# Duplicated from Light_Curve.GAIA_ZERO_PTS deliberately: importing that module
# pulls in ztfquery, sparcl and astroquery, none of which this needs.
GAIA_ZERO_PTS = {'g': 25.8010, 'bp': 25.3540, 'rp': 25.1040}

PTF_FID_TO_BAND = {1: 'g', 2: 'R'}


class Catalog:
    """One IRSA table, its columns, and how its rows become light-curve epochs."""

    def __init__(self, name, table, id_col, mjd_col, columns, bands):
        self.name = name          # our survey label
        self.table = table        # IRSA table name
        self.id_col = id_col      # the service's object identifier
        self.mjd_col = mjd_col    # time column as the service names it
        self.columns = columns    # everything to select
        self.bands = bands        # {band: (mag_col, magerr_col, snr_col|None)}


_w = lambda n: (f'w{n}mpro_ep', f'w{n}sigmpro_ep', None)
WISE = Catalog(
    'wise', 'allwise_p3as_mep', 'source_id_mf', 'mjd',
    ['source_id_mf', 'ra', 'dec', 'mjd']
    + [c for n in range(1, 5)
       for c in (f'w{n}mpro_ep', f'w{n}sigmpro_ep', f'w{n}flux_ep', f'w{n}sigflux_ep')],
    {f'w{n}': _w(n) for n in range(1, 5)},
)

NEOWISE = Catalog(
    'neowise', 'neowiser_p1bs_psd', 'source_id', 'mjd',
    ['source_id', 'ra', 'dec', 'mjd']
    + [c for n in range(1, 3)
       for c in (f'w{n}mpro', f'w{n}sigmpro', f'w{n}snr')],
    {f'w{n}': (f'w{n}mpro', f'w{n}sigmpro', f'w{n}snr') for n in range(1, 3)},
)

PTF = Catalog(
    'ptf', 'ptf_lightcurves', 'oid', 'obsmjd',
    ['oid', 'ra', 'dec', 'obsmjd', 'fid', 'mag_autocorr', 'magerr_auto'],
    {},   # band comes from `fid`, handled in normalise
)

GAIA = Catalog(
    'gaia', 'gaia_dr3_source', 'source_id', 'ref_epoch',
    ['source_id', 'ra', 'dec', 'ref_epoch']
    + [c for b in ('g', 'rp', 'bp')
       for c in (f'phot_{b}_mean_flux', f'phot_{b}_mean_flux_error')],
    {},   # fluxes -> magnitudes in normalise
)

CATALOGS = {c.name: c for c in (WISE, NEOWISE, PTF, GAIA)}


def _adql(cat: Catalog, radius_arcsec: float) -> str:
    """The upload cone join.

    Comma join with `WHERE CONTAINS(...) = 1`, not `JOIN ... ON CONTAINS(...)`:
    IRSA rejects the latter outright as "Invalid or unsupported ADQL query
    string", while accepting this form. Both are legal ADQL; only one is
    implemented here.
    """
    cols = ', '.join(f'c.{c}' for c in cat.columns)
    return (f"SELECT u.source_uuid, {cols} "
            f"FROM TAP_UPLOAD.pos AS u, {cat.table} AS c "
            f"WHERE CONTAINS(POINT('ICRS', c.ra, c.dec), "
            f"CIRCLE('ICRS', u.ra, u.dec, {radius_arcsec / 3600.0})) = 1")


def _upload_table(positions: pd.DataFrame):
    """Positions as a VOTable IRSA will accept.

    `source_uuid` is ASCII-encoded to bytes on purpose. A normal pandas/numpy
    str column serialises as VOTable `unicodeChar`, and IRSA's TAP rejects the
    whole query with "INTERNAL_SERVER_ERROR: Unimplemented data type:
    unicodeChar". Bytes serialise as `char`, which it accepts.

    Carrying the uuid through the query means results come back already keyed
    to our sources; the alternative (upload a row index, join back locally) is
    an extra chance to mis-associate for no gain.
    """
    from astropy.table import Table
    return Table({
        'source_uuid': np.char.encode(
            positions.source_uuid.to_numpy().astype(str), 'ascii'),
        'ra': positions.ra.to_numpy(float),
        'dec': positions.dec.to_numpy(float),
    })


def fetch_chunk(cat: Catalog, positions: pd.DataFrame,
                radius_arcsec: float = MATCH_RADIUS_ARCSEC,
                poll_seconds: float = POLL_SECONDS,
                max_wait_seconds: float = MAX_WAIT_SECONDS) -> pd.DataFrame:
    """One async TAP upload-join for one chunk of positions.

    Driven manually rather than with `run_async`, which blocks in a single
    `WAIT=-1` long poll with no socket timeout: if the connection dies under
    it -- a laptop going offline is enough -- it waits forever. That cost a
    NEOWISE chunk 7.5 hours of doing nothing, twice. `run_async` exposes no
    timeout, but `AsyncTAPJob.wait` does, so the poll is bounded here and the
    whole job is capped by a wall clock.
    """
    import pyvo

    svc = pyvo.dal.TAPService(TAP_URL)
    # async, not sync: sync caps at 5 minutes, async allows 3 days.
    job = svc.submit_job(_adql(cat, radius_arcsec),
                         uploads={'pos': _upload_table(positions)},
                         maxrec=100_000_000)
    try:
        job.run()
        deadline = time.time() + max_wait_seconds
        while True:
            phase = job.phase
            if phase in ('COMPLETED', 'ERROR', 'ABORTED'):
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f'{cat.name}: job still {phase} after '
                    f'{max_wait_seconds / 60:.0f} min; abandoning so the '
                    f'chunk can be retried')
            # Bounded: a dead connection surfaces as an exception on this
            # call instead of hanging, and the loop simply polls again.
            try:
                job.wait(phases={'COMPLETED', 'ERROR', 'ABORTED'},
                         timeout=poll_seconds)
            except pyvo.DALServiceError as exc:
                print(f'    poll failed ({exc!r}); retrying', flush=True)
                time.sleep(min(poll_seconds, 30))
        job.raise_if_error()
        res = job.fetch_result()
        df = res.to_table().to_pandas()
    finally:
        # Free the server-side job whatever happened; IRSA keeps results for
        # three days otherwise.
        try:
            job.delete()
        except Exception:
            pass
    if 'source_uuid' in df.columns:
        # comes back as bytes, since that is how it went up
        df['source_uuid'] = df['source_uuid'].apply(
            lambda v: v.decode() if isinstance(v, bytes) else str(v))
    return df


def fetch(catalog: str, limit: Optional[int] = None, raw_dir: Path = RAW_DIR,
          chunk_size: int = CHUNK_SIZE, resume: bool = True,
          radius_arcsec: float = MATCH_RADIUS_ARCSEC, workers: int = 1,
          max_wait_seconds: float = MAX_WAIT_SECONDS) -> dict:
    """Fetch one catalog for every source, in resumable chunks.

    `workers` > 1 keeps that many async jobs open at IRSA at once. The chunks
    are disjoint and each writes its own part, so this only changes wall time.
    """
    cat = CATALOGS[catalog]
    positions = _base.load_positions(limit=limit)
    chunks = _base.row_chunks(positions, chunk_size)
    # Coverage is checked against every fetched uuid, not per chunk id, so
    # adding sources later costs new chunks rather than a full refetch.
    todo = _base.pending_chunks(cat.name, positions, chunk_size, raw_dir, resume)
    print(f'{cat.name}: {len(positions):,} sources, {len(chunks)} chunks, '
          f'{len(todo)} to do', flush=True)

    t0 = time.time()

    def one(item):
        cid, sub = item
        t1 = time.time()
        # IRSA returns transient 5xx under load, and a NEOWISE chunk costs
        # ~75 min -- so one bad response must not throw the run away. Retry
        # with a backoff, then move on and leave the chunk uncovered for the
        # next --resume.
        df = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                df = fetch_chunk(cat, sub, radius_arcsec, max_wait_seconds=max_wait_seconds)
                break
            except Exception as exc:
                if attempt == MAX_ATTEMPTS:
                    print(f'  chunk {cid}: giving up after {attempt} attempts '
                          f'({exc!r})', flush=True)
                    return cid, None
                wait = RETRY_BACKOFF_S * 2 ** (attempt - 1)
                print(f'  chunk {cid}: attempt {attempt} failed ({exc!r}); '
                      f'retrying in {wait}s', flush=True)
                time.sleep(wait)
        # n_rows is recorded by write_raw itself -- passing it here too would
        # collide in record_chunk(**meta).
        _base.write_raw(cat.name, cid, df, raw_dir=raw_dir,
                        n_positions=len(sub))
        _base.write_coverage(cat.name, cid, sub.source_uuid, raw_dir=raw_dir)
        print(f'  chunk {cid}: {len(sub):,} positions -> {len(df):,} rows '
              f'[{time.time() - t1:.0f}s]', flush=True)
        return cid, len(df)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        results = list(ex.map(one, todo))
    failed = [cid for cid, n in results if n is None]
    n_rows = sum(n for _, n in results if n is not None)
    dt = time.time() - t0
    if failed:
        print(f'{cat.name}: {len(failed)} chunks failed ({", ".join(failed)}); '
              f'rerun to retry them', flush=True)
    print(f'{cat.name}: {n_rows:,} rows in {dt / 60:.1f} min', flush=True)
    return {'survey': cat.name, 'n_rows': n_rows,
            'n_failed': len(failed), 'seconds': dt}


def normalise(catalog: str, raw: pd.DataFrame) -> pd.DataFrame:
    """Raw service rows -> the shared light-curve schema, one row per band."""
    cat = CATALOGS[catalog]
    if raw.empty:
        return _base.empty_lightcurves()

    out = []
    if cat.name == 'ptf':
        m = raw['mag_autocorr'].notna()
        sub = raw[m]
        out.append(pd.DataFrame({
            'source_uuid': sub['source_uuid'].astype('string'),
            'survey': 'ptf', 'band': sub['fid'].map(PTF_FID_TO_BAND).astype('string'),
            'mjd': sub['obsmjd'].astype('float64'),
            'mag': sub['mag_autocorr'].astype('float64'),
            'magerr': sub['magerr_auto'].astype('float64'),
            'snr': np.nan, 'catalog_id': sub['oid'].astype('string')}))

    elif cat.name == 'gaia':
        from astropy.time import Time
        # Gaia gives one mean flux per band at a reference epoch, not a time
        # series -- three rows per source, matching what Light_Curve plots.
        mjd = Time(raw['ref_epoch'].to_numpy(float), format='jyear', scale='tcb').mjd
        for band in ('g', 'rp', 'bp'):
            f = raw[f'phot_{band}_mean_flux'].to_numpy(float)
            fe = raw[f'phot_{band}_mean_flux_error'].to_numpy(float)
            ok = np.isfinite(f) & (f > 0)
            if not ok.any():
                continue
            mag = GAIA_ZERO_PTS[band] - 2.5 * np.log10(f[ok])
            out.append(pd.DataFrame({
                'source_uuid': raw['source_uuid'].to_numpy()[ok].astype(str),
                'survey': 'gaia', 'band': {'rp': 'r', 'bp': 'b'}.get(band, band),
                'mjd': mjd[ok], 'mag': mag,
                'magerr': 2.5 / np.log(10) * fe[ok] / f[ok],
                'snr': f[ok] / fe[ok],
                'catalog_id': raw['source_id'].to_numpy()[ok].astype(str)}))

    else:   # wise, neowise: one row per band per epoch
        for band, (magc, errc, snrc) in cat.bands.items():
            if magc not in raw.columns:
                continue
            ok = raw[magc].notna()
            if not ok.any():
                continue
            sub = raw[ok]
            out.append(pd.DataFrame({
                'source_uuid': sub['source_uuid'].astype('string'),
                'survey': cat.name, 'band': band,
                'mjd': sub[cat.mjd_col].astype('float64'),
                'mag': sub[magc].astype('float64'),
                'magerr': sub[errc].astype('float64'),
                'snr': sub[snrc].astype('float64') if snrc in sub.columns else np.nan,
                'catalog_id': sub[cat.id_col].astype('string')}))

    if not out:
        return _base.empty_lightcurves()
    df = pd.concat(out, ignore_index=True)
    for c in ('source_uuid', 'survey', 'band', 'catalog_id'):
        df[c] = df[c].astype('string')
    return df[_base.LIGHTCURVE_COLUMNS]


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--catalogs', nargs='+', default=list(CATALOGS),
                    choices=list(CATALOGS))
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    ap.add_argument('--raw-dir', type=Path, default=RAW_DIR)
    ap.add_argument('--workers', type=int, default=1,
                    help='async jobs open at IRSA at once (chunks are disjoint)')
    ap.add_argument('--max-wait-hours', type=float, default=MAX_WAIT_SECONDS / 3600,
                    help='abandon and retry a job still running after this long')
    a = ap.parse_args()
    for name in a.catalogs:
        fetch(name, limit=a.limit, raw_dir=a.raw_dir, chunk_size=a.chunk_size,
              workers=a.workers, max_wait_seconds=a.max_wait_hours * 3600)
