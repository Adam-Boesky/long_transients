"""ZTF light curves, in bulk, from the DR24 HATS collection on public S3.

Why not the light-curve API
---------------------------
`irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves` takes one position per
request and has no async mode. IRSA's own docs redirect anything over ~1000
light curves to the HATS collection, which is the same DR24 that
`Light_Curve.catalog_astroquery_map` already points at (`ztf_objects_dr24`).
It also survives IRSA outages -- the bucket answered while IRSA TAP was 502.

Why pyarrow rather than lsdb
----------------------------
IRSA lists pyarrow first for advanced use: "files can be efficiently queried
in place (no need to download them first)", and "query efficiency is strongly
impacted by dataset partitioning -- include partition key/value pairs in row
filters whenever possible."

lsdb was tried and abandoned. Measured on one 172 MB / 275k-row partition:

    pyarrow, position columns only      1.2 s
    pyarrow, including lightcurve       5-7 s
    lsdb crossmatch, same work         30-98 s, and RSS grew without bound
                                       (2.1 -> 9.7 GB across three batches)

So the cost was lsdb's overhead, not S3 or the data volume. Reading directly
also keeps this in the normal `long_transients` env -- lsdb/hats would have
forced numpy>=2.3 and astropy>=7 -- and lets the match use the same astropy
`search_around_sky` as `catalogs/_base.py` instead of a second crossmatch
implementation.

Two passes, both column-pruned
------------------------------
Row-group pruning on position does *not* work here: the files are sorted by
`objectid` (`hats_cols_sort=objectid`), so the three row groups per partition
overlap in `_healpix_29`. Column pruning is what pays.

    pass 1  read objectid/objra/objdec/filterid from each partition our
            sources touch (1.2 s each), match locally, keep the hits
    pass 2  re-read only partitions that matched, this time with `lightcurve`,
            and keep only the matched objectids

Pass 1 is ~5,120 partitions at ~1.2 s. Both passes parallelise: each partition
is independent, and the work is I/O-bound, so threads are enough.

One source, several objectids
-----------------------------
ZTF assigns a separate `objectid` per band *and* per field overlap, so one
position routinely returns 3-8 objects. All of them are kept; `Light_Curve`
does the same thing by associating everything within 1.5".
"""
from __future__ import annotations

import io
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source_db.lightcurves import _base  # noqa: E402
from source_db.paths import RAW_DIR  # noqa: E402

SURVEY = 'ztf'

BUCKET = 'ipac-irsa-ztf'
PREFIX = 'ztf/enhanced/dr24/lc/hats/ztf_dr24_lc-hats'
REGION = 'us-east-1'

# Position pass: everything needed to match, and nothing heavy.
LIGHT_COLUMNS = ['objectid', 'filterid', 'objra', 'objdec', 'nepochs']
# Light-curve pass: adds the nested column.
FULL_COLUMNS = ['objectid', 'filterid', 'objra', 'objdec', 'lightcurve']
EPOCH_FIELDS = ['hmjd', 'mag', 'magerr', 'clrcoeff', 'catflags']

FILTERID_TO_BAND = {1: 'g', 2: 'r', 3: 'i'}

# Matches Source.max_arcsec and enrich_combined_tabs.MATCH_RADIUS_ARCSEC.
MATCH_RADIUS_ARCSEC = 1.5

# I/O-bound, so threads rather than processes; S3 measured ~24 MB/s per stream.
N_THREADS = 8


# One filesystem for the whole run, not one per read. Each S3FileSystem owns
# a connection pool and background IO threads, so building one per partition
# leaked both: a 5,072-partition run wedged after ~1,100 reads with 58 threads,
# every socket in CLOSED, and no progress for five hours. The timeouts matter
# for the same reason -- the default is to wait forever, so one dropped
# connection silently costs a worker for the rest of the run.
_FS = None
_FS_LOCK = threading.Lock()


def _fs():
    global _FS
    if _FS is None:
        with _FS_LOCK:
            if _FS is None:
                import pyarrow.fs as pafs
                _FS = pafs.S3FileSystem(
                    anonymous=True, region=REGION,
                    connect_timeout=10.0, request_timeout=300.0,
                    retry_strategy=pafs.AwsStandardS3RetryStrategy(max_attempts=5),
                )
    return _FS


def partition_table() -> pd.DataFrame:
    """The collection's partition list: Norder, Npix, num_rows."""
    import pyarrow.parquet as pq  # noqa: F401  (kept for symmetry)
    import requests
    url = (f'https://{BUCKET}.s3.{REGION}.amazonaws.com/{PREFIX}/partition_info.csv')
    return pd.read_csv(io.StringIO(requests.get(url, timeout=120).text))


def partition_path(norder: int, npix: int) -> str:
    return (f'{BUCKET}/{PREFIX}/dataset/Norder={norder}/'
            f'Dir={npix // 10000 * 10000}/Npix={npix}/part0.snappy.parquet')


def partitions_for(ra: np.ndarray, dec: np.ndarray,
                   parts: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Which partitions contain these positions.

    This is the "include partition key/value pairs in row filters" step: the
    HEALPix directory structure *is* the partition key, so selecting the right
    Norder/Npix directories prunes 12,485 partitions down to the ~5,120 our
    sources actually touch.
    """
    import hpgeom
    parts = partition_table() if parts is None else parts
    hit = np.zeros(len(parts), dtype=bool)
    for order in sorted(parts.Norder.unique()):
        occupied = set(np.unique(hpgeom.angle_to_pixel(2 ** order, ra, dec,
                                                       nest=True)).tolist())
        hit |= ((parts.Norder == order) & parts.Npix.isin(occupied)).to_numpy()
    return parts[hit].reset_index(drop=True)


def _read(path: str, columns: list[str], fs=None):
    import pyarrow.parquet as pq
    fs = fs or _fs()
    return pq.read_table(fs.open_input_file(path), columns=columns)


def match_partition(norder: int, npix: int, positions: pd.DataFrame,
                    radius_arcsec: float = MATCH_RADIUS_ARCSEC) -> pd.DataFrame:
    """Pass 1 for one partition: which of our sources match which objectids."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord, search_around_sky

    tbl = _read(partition_path(norder, npix), LIGHT_COLUMNS)
    if tbl.num_rows == 0 or positions.empty:
        return pd.DataFrame(columns=['source_uuid', 'objectid', 'sep_arcsec'])

    objra = tbl.column('objra').to_numpy(zero_copy_only=False).astype(float)
    objdec = tbl.column('objdec').to_numpy(zero_copy_only=False).astype(float)
    ours = SkyCoord(positions.ra.to_numpy(float) * u.deg,
                    positions.dec.to_numpy(float) * u.deg)
    theirs = SkyCoord(objra * u.deg, objdec * u.deg)
    i_ours, i_theirs, sep, _ = search_around_sky(ours, theirs,
                                                 radius_arcsec * u.arcsec)
    if len(i_ours) == 0:
        return pd.DataFrame(columns=['source_uuid', 'objectid', 'sep_arcsec'])
    oid = tbl.column('objectid').to_numpy(zero_copy_only=False)
    return pd.DataFrame({
        'source_uuid': positions.source_uuid.to_numpy()[i_ours],
        'objectid': oid[i_theirs],
        'sep_arcsec': sep.arcsec,
        'Norder': norder, 'Npix': npix,
    })


def fetch_lightcurves(norder: int, npix: int,
                      wanted: Iterable[int]) -> pd.DataFrame:
    """Pass 2 for one partition: epochs for the matched objectids only."""
    wanted = set(int(x) for x in wanted)
    tbl = _read(partition_path(norder, npix), FULL_COLUMNS)
    oid = tbl.column('objectid').to_numpy(zero_copy_only=False)
    keep = np.flatnonzero(np.isin(oid, list(wanted)))
    if len(keep) == 0:
        del tbl
        return pd.DataFrame(columns=EPOCH_FIELDS + ['objectid', 'filterid'])
    # `tbl` is ~360 MB for a typical partition and we keep ~30 of 275k rows.
    # Drop it immediately: with several readers in flight, holding these is
    # what drove RSS to 11 GB in testing.
    sub = tbl.take(keep).combine_chunks()
    del tbl

    # Flatten in Arrow, not Python. `lightcurve` is struct<hmjd: list<...>,
    # mag: list<...>, ...>, so each field is a ListArray sharing one set of
    # offsets: list_flatten gives the concatenated epoch values and
    # list_parent_indices maps each epoch back to its object. An earlier
    # version used `.to_pylist()` and built a DataFrame per object -- that is
    # pure-Python work holding the GIL, which serialised every worker thread
    # and materialised the whole nested column as Python objects.
    import pyarrow.compute as pc

    lc = sub.column('lightcurve').combine_chunks()
    if hasattr(lc, 'chunk'):
        lc = lc.chunk(0) if lc.num_chunks else None
    if lc is None or len(lc) == 0:
        return pd.DataFrame(columns=EPOCH_FIELDS + ['objectid', 'filterid'])

    first = lc.field(EPOCH_FIELDS[0])
    parent = pc.list_parent_indices(first).to_numpy(zero_copy_only=False)
    if len(parent) == 0:
        return pd.DataFrame(columns=EPOCH_FIELDS + ['objectid', 'filterid'])

    data = {f: pc.list_flatten(lc.field(f)).to_numpy(zero_copy_only=False)
            for f in EPOCH_FIELDS}
    data['objectid'] = sub.column('objectid').to_numpy(zero_copy_only=False)[parent]
    data['filterid'] = sub.column('filterid').to_numpy(zero_copy_only=False)[parent]
    return pd.DataFrame(data)


def fetch(limit: Optional[int] = None, raw_dir: Path = RAW_DIR,
          n_threads: int = N_THREADS, resume: bool = True) -> dict:
    """Both passes over every partition our sources touch, written per partition."""
    positions = _base.load_positions(limit=limit)
    ra = positions.ra.to_numpy(float)
    dec = positions.dec.to_numpy(float)

    import hpgeom
    parts = partitions_for(ra, dec)
    print(f'{len(positions):,} sources -> {len(parts):,} partitions', flush=True)

    # Bucket our sources by the partition that contains them, so each task
    # only carries the positions it needs.
    by_part: dict[tuple[int, int], pd.DataFrame] = {}
    for order in sorted(parts.Norder.unique()):
        pix = hpgeom.angle_to_pixel(2 ** order, ra, dec, nest=True)
        sel = parts[parts.Norder == order]
        for npix in sel.Npix.to_numpy():
            m = pix == npix
            if m.any():
                by_part[(int(order), int(npix))] = positions[m]

    # Resume on *sources*, not partitions: a partition already fetched still
    # needs redoing if sources have since been added to it, which is where most
    # newly added sources land.
    covered = _base.covered_sources(SURVEY, raw_dir) if resume else {}
    todo = []
    for key, sub in by_part.items():
        chunk_id = f'{key[0]}_{key[1]}'
        if not set(sub.source_uuid) <= covered.get(chunk_id, set()):
            todo.append(key)
    n_stale = sum(1 for k in by_part
                  if f'{k[0]}_{k[1]}' in covered and k in todo)
    print(f'{len(todo):,} partitions to do '
          f'({len(by_part) - len(todo):,} already cover their sources'
          + (f', {n_stale:,} being redone for newly added sources' if n_stale else '')
          + ')', flush=True)

    # Largest partitions first: cost tracks partition size (measured 5-68 s),
    # not our source count, so this keeps the long tail off the end of the run.
    size = {(int(r.Norder), int(r.Npix)): int(r.num_rows)
            for r in parts.itertuples(index=False)}
    todo.sort(key=lambda k: -size.get(k, 0))

    t0 = time.time()
    n_epochs = 0
    failed: list[tuple[int, int]] = []

    def one(key):
        (order, npix) = key
        hits = match_partition(order, npix, by_part[key])
        if hits.empty:
            return key, pd.DataFrame()
        epochs = fetch_lightcurves(order, npix, hits.objectid.unique())
        if epochs.empty:
            return key, pd.DataFrame()
        # attach source_uuid; one objectid can serve several of our sources
        return key, epochs.merge(hits[['source_uuid', 'objectid', 'sep_arcsec']],
                                 on='objectid', how='inner')

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = {ex.submit(one, k): k for k in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            # A partition that raises is left uncovered on purpose: nothing is
            # written for it, so the next --resume picks it up again rather
            # than the whole run dying on one bad read.
            try:
                key, df = fut.result()
            except Exception as exc:
                key = futures[fut]
                failed.append(key)
                print(f'  partition {key[0]}_{key[1]} FAILED: {exc!r}', flush=True)
                continue
            chunk_id = f'{key[0]}_{key[1]}'
            _base.write_raw(SURVEY, chunk_id, df, raw_dir=raw_dir,
                            n_positions=len(by_part[key]), n_epochs=len(df))
            # Every source the partition was *asked about*, including ones with
            # no ZTF match -- otherwise an unmatched source is retried forever.
            _base.write_coverage(SURVEY, chunk_id, by_part[key].source_uuid,
                                 raw_dir=raw_dir)
            n_epochs += len(df)
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                print(f'  {i:,}/{len(todo):,} partitions  {n_epochs:,} epochs  '
                      f'{el / 60:.1f} min  (eta {el / i * (len(todo) - i) / 60:.0f} min)',
                      flush=True)

    dt = time.time() - t0
    if failed:
        print(f'{len(failed)} partitions failed; rerun with --resume to retry them',
              flush=True)
    print(f'done: {n_epochs:,} epochs in {dt / 60:.1f} min', flush=True)
    return {'survey': SURVEY, 'n_positions': len(positions),
            'n_partitions': len(todo), 'n_epochs': n_epochs,
            'n_failed': len(failed), 'seconds': dt}


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw ZTF epochs -> the shared light-curve schema.

    `catflags` is carried through rather than applied. The quality cut
    (`catflags & (6141 | 32768)`, ZTF Explanatory Supplement 10.3, see
    `Source_Analysis.ZTF_Local_LC`) belongs to the consumer: baking it in here
    would make the stored light curves unrecoverable if the mask changed.
    """
    if raw.empty:
        return _base.empty_lightcurves()
    out = pd.DataFrame({
        'source_uuid': raw['source_uuid'].astype('string'),
        'survey': pd.array([SURVEY] * len(raw), dtype='string'),
        'band': raw['filterid'].map(FILTERID_TO_BAND).astype('string'),
        'mjd': raw['hmjd'].astype('float64'),      # heliocentric; see ZTF_Local_LC
        'mag': raw['mag'].astype('float64'),
        'magerr': raw['magerr'].astype('float64'),
        'snr': pd.Series([float('nan')] * len(raw), dtype='float64'),
        'catalog_id': raw['objectid'].astype('string'),
    })
    out['catflags'] = raw['catflags'].astype('Int32')
    out['clrcoeff'] = raw['clrcoeff'].astype('float64')
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--threads', type=int, default=N_THREADS)
    ap.add_argument('--raw-dir', type=Path, default=RAW_DIR)
    ap.add_argument('--no-resume', action='store_true')
    a = ap.parse_args()
    fetch(limit=a.limit, raw_dir=a.raw_dir, n_threads=a.threads,
          resume=not a.no_resume)
