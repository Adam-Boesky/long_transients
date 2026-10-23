"""ZTF light curves from the local DR24 per-quadrant copy on the cluster.

Same output as `ztf.py` (the S3 HATS fetch): rows in `raw/ztf/`, one per
epoch, with the same columns and dtypes, every ZTF object within 1.5" of each
source kept, and coverage recorded per source. Only *where* the epochs are read
from differs, and a ~1 TB S3 re-read becomes local disk reads.

    python -m source_db.lightcurves.ztf_local --uuids uncovered.txt

The local copy (`ztf_lc_dir`, IRSA's lc_dr24 tree) has one Parquet file per
field / CCD / quadrant / band. Verified against the S3 path on 7,253 sources
(31,662 objects, 5,075,044 epochs): identical objects and bit-identical epochs.
Two details matter for that:

* Which files to open comes from each file's real sky extent (Parquet footer
  min/max of objra/objdec), not ztffields' nominal outlines. Those can sit
  ~30" off the real footprint and miss whole files near the pole and in
  secondary-grid overlaps.
* Separations are computed in float64, as `ztf.py` does. The files store
  positions as float32, and float32 arithmetic moves objects at exactly 1.5"
  across the cut.

Chunk ids are `local_NNNN`, so they never collide with the S3 path's sky
partition ids.
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source_db.lightcurves import _base  # noqa: E402
from source_db.lightcurves.ztf import EPOCH_FIELDS, MATCH_RADIUS_ARCSEC, SURVEY  # noqa: E402
from source_db.paths import RAW_DIR  # noqa: E402

LOCAL_DIR = Path('/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/ztf_lc_dir')
N_THREADS = 32
CHUNK_SOURCES = 5000

# Raw dtypes written by `ztf.py`, so both paths normalise identically.
RAW_DTYPES = {'hmjd': 'float64', 'mag': 'float32', 'magerr': 'float32', 'clrcoeff': 'float32',
              'catflags': 'int32', 'objectid': 'int64', 'filterid': 'int8', 'sep_arcsec': 'float64'}


def extent_index(local_dir: Path = LOCAL_DIR, cache: Optional[Path] = None,
                 n_threads: int = N_THREADS) -> pd.DataFrame:
    """Sky extent of every local file, from its Parquet footer statistics."""
    import pyarrow.parquet as pq
    if cache is not None and cache.exists():
        return pd.read_parquet(cache)
    rel = [u.strip().split('/lc_dr24/')[1]
           for u in open(local_dir / 'parquet_urls_from_checksum.txt') if u.strip()]

    def one(r):
        md = pq.read_metadata(local_dir / r)
        names = md.schema.names
        ia, idc = names.index('objra'), names.index('objdec')
        st = [(md.row_group(g).column(ia).statistics, md.row_group(g).column(idc).statistics)
              for g in range(md.num_row_groups)]
        if md.num_rows == 0 or not all(a is not None and a.has_min_max and d is not None and d.has_min_max
                                       for a, d in st):
            return r, md.num_rows, np.nan, np.nan, np.nan, np.nan
        return (r, md.num_rows, min(a.min for a, _ in st), max(a.max for a, _ in st),
                min(d.min for _, d in st), max(d.max for _, d in st))

    with ThreadPoolExecutor(n_threads) as ex:
        idx = pd.DataFrame(list(ex.map(one, rel)),
                           columns=['file', 'rows', 'ramin', 'ramax', 'decmin', 'decmax'])
    idx = idx[idx.rows > 0].reset_index(drop=True)  # empty files hold nothing to match
    if cache is not None:
        idx.to_parquet(cache, index=False)
    return idx


def files_for(positions: pd.DataFrame, idx: pd.DataFrame,
              radius_arcsec: float = MATCH_RADIUS_ARCSEC) -> dict[str, list[int]]:
    """`{file: [row indices into positions]}` for every file whose extent reaches a source."""
    r = radius_arcsec / 3600
    dec_lo, dec_hi = idx.decmin.to_numpy() - r, idx.decmax.to_numpy() + r
    ramin, ramax = idx.ramin.to_numpy(), idx.ramax.to_numpy()
    wide = ramax - ramin > 180  # straddles RA 0/360 or circles the pole: select on Dec alone
    files = idx.file.to_numpy()
    out = defaultdict(list)
    for i, (ra, dec) in enumerate(zip(positions.ra.to_numpy(float), positions.dec.to_numpy(float))):
        dra = r / max(np.cos(np.radians(dec)), 1e-3)
        in_ra = wide.copy()
        for shift in (0.0, 360.0, -360.0):
            in_ra |= (ra + shift >= ramin - dra) & (ra + shift <= ramax + dra)
        for f in files[(dec >= dec_lo) & (dec <= dec_hi) & in_ra]:
            out[f].append(i)
    return out


def match_file(path: Path, positions: pd.DataFrame,
               radius_arcsec: float = MATCH_RADIUS_ARCSEC) -> pd.DataFrame:
    """Every object in one file within the radius of these sources, exploded to epochs."""
    import astropy.units as u
    import pyarrow.parquet as pq
    from astropy.coordinates import SkyCoord, search_around_sky

    t = pq.read_table(path, columns=['objectid', 'objra', 'objdec']).to_pandas()
    ours = SkyCoord(positions.ra.to_numpy(float) * u.deg, positions.dec.to_numpy(float) * u.deg)
    # float64 before any arithmetic: the files store float32, see the module docstring
    theirs = SkyCoord(t.objra.to_numpy(float) * u.deg, t.objdec.to_numpy(float) * u.deg)
    i_ours, i_theirs, sep, _ = search_around_sky(ours, theirs, radius_arcsec * u.arcsec)
    if len(i_ours) == 0:
        return pd.DataFrame()
    hits = pd.DataFrame({'source_uuid': positions.source_uuid.to_numpy()[i_ours],
                         'objectid': t.objectid.to_numpy()[i_theirs],
                         'sep_arcsec': sep.arcsec})
    full = pq.read_table(path, columns=['objectid', 'filterid'] + EPOCH_FIELDS,
                         filters=[('objectid', 'in', hits.objectid.unique().tolist())])
    import pyarrow.compute as pc
    parent = pc.list_parent_indices(full.column('hmjd')).to_numpy(zero_copy_only=False)
    epochs = pd.DataFrame({f: pc.list_flatten(full.column(f)).to_numpy(zero_copy_only=False)
                           for f in EPOCH_FIELDS})
    epochs['objectid'] = full.column('objectid').to_numpy()[parent]
    epochs['filterid'] = full.column('filterid').to_numpy()[parent]
    return epochs.merge(hits, on='objectid', how='inner')


def fetch(uuids: Optional[set[str]] = None, raw_dir: Path = RAW_DIR,
          local_dir: Path = LOCAL_DIR, n_threads: int = N_THREADS,
          index_cache: Optional[Path] = None) -> dict:
    """Light curves for the given sources (default: every source not yet covered)."""
    positions = _base.load_positions()
    done = _base.all_covered(SURVEY, raw_dir)
    want = positions[~positions.source_uuid.isin(done)]
    if uuids is not None:
        want = want[want.source_uuid.isin(uuids)]
    want = want.sort_values('source_uuid').reset_index(drop=True)
    print(f'{len(want):,} sources to do ({len(done):,} already covered)', flush=True)
    if want.empty:
        return {'n_sources': 0}

    t0 = time.time()
    idx = extent_index(local_dir, index_cache, n_threads)
    print(f'extent index: {len(idx):,} non-empty files [{time.time() - t0:.0f}s]', flush=True)
    # `_base.next_chunk_index` only sees numeric ids, so number our own prefix
    taken = [int(p.stem.removeprefix('part-local_')) for p in _base.raw_dir_for(SURVEY, raw_dir).glob('part-local_*.parquet')]
    start = max(taken) + 1 if taken else 0
    for c, lo in enumerate(range(0, len(want), CHUNK_SOURCES)):
        chunk = want.iloc[lo:lo + CHUNK_SOURCES].reset_index(drop=True)
        by_file = files_for(chunk, idx)
        with ThreadPoolExecutor(n_threads) as ex:
            parts = list(ex.map(lambda kv: match_file(local_dir / kv[0], chunk.iloc[kv[1]]),
                                by_file.items()))
        parts = [p for p in parts if len(p)]
        df = (pd.concat(parts, ignore_index=True) if parts
              else pd.DataFrame(columns=EPOCH_FIELDS + ['objectid', 'filterid', 'source_uuid', 'sep_arcsec']))
        df = df.astype(RAW_DTYPES)[EPOCH_FIELDS + ['objectid', 'filterid', 'source_uuid', 'sep_arcsec']]
        chunk_id = f'local_{start + c:04d}'
        _base.write_raw(SURVEY, chunk_id, df, raw_dir=raw_dir, n_positions=len(chunk),
                        n_epochs=len(df), source='ztf_lc_dir')
        _base.write_coverage(SURVEY, chunk_id, chunk.source_uuid, raw_dir=raw_dir)
        print(f'  {chunk_id}: {len(chunk):,} sources, {len(by_file):,} files, '
              f'{df.source_uuid.nunique():,} with ZTF, {len(df):,} epochs '
              f'[{(time.time() - t0) / 60:.1f} min]', flush=True)
    return {'n_sources': len(want), 'seconds': time.time() - t0}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--uuids', type=Path, default=None,
                    help='file of source_uuids to fetch (default: every uncovered source)')
    ap.add_argument('--raw-dir', type=Path, default=RAW_DIR)
    ap.add_argument('--local-dir', type=Path, default=LOCAL_DIR)
    ap.add_argument('--threads', type=int, default=N_THREADS)
    ap.add_argument('--index-cache', type=Path, default=None)
    a = ap.parse_args()
    ids = set(a.uuids.read_text().split()) if a.uuids else None
    fetch(ids, a.raw_dir, a.local_dir, a.threads, a.index_cache)
