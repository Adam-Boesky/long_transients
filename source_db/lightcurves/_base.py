"""Shared machinery for the light-curve backfill.

Two stages, deliberately separate:

    fetch      per-catalog bulk pull -> Data/db/raw/<catalog>/part-NNNN.parquet
               native schema, fetched once, never re-fetched
    normalise  raw/ -> Data/db/light_curves/, the melted schema below
               pure function of raw/, so it is freely re-runnable

Keeping them apart means a bug in the melt logic costs minutes rather than
another overnight backfill against six external services.

The normalised schema
---------------------
    source_uuid  the key, carried through the fetch so results never need
                 re-crossmatching back onto sources
    survey       ztf | wise | neowise | ptf | sdss | panstarrs | gaia | custom
    band         g r i z y u b w1 w2 w3 w4 R ...
    mjd, mag, magerr, snr
    catalog_id   the service's own object id, for provenance back to raw/

This is the melted form of what `Light_Curve.get_lc()` returns. The native
table is one row per epoch with ~30 magnitude columns of which one or two are
populated; `Light_Curve.ALL_BAND_DF` is exactly the
``column -> (band, survey)`` map needed to pivot back to the wide form the
plotting code expects.

Correctness gate
----------------
Every catalog module must provide `verify()`, which runs `Light_Curve.get_lc()`
the old way on a sample and diffs against the bulk result. The bulk path
reimplements the per-catalog closest-object filtering that `get_lc` does inline
(the 1.5" ZTF cross-band association, the NEOWISE special case); if that drifts
the stored light curves silently stop matching the plotted ones.
"""
from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from source_db.paths import DEFAULT_OUT_DIR, RAW_DIR

LIGHTCURVE_COLUMNS = ['source_uuid', 'survey', 'band', 'mjd',
                      'mag', 'magerr', 'snr', 'catalog_id']

# Per-survey extras kept alongside the shared schema, because they are needed
# to reproduce what the pages plot and cannot be recovered without re-fetching.
# ZTF: `catflags` drives the quality cut (catflags & 6141, ZTF Explanatory
# Supplement 10.3, see Source_Analysis.ZTF_Local_LC) and `clrcoeff` is carried
# through by that same reader.
# Documentation only -- nothing reads this. Each module's `normalise` decides
# what it emits, so keep the two in step by hand.
SURVEY_EXTRA_COLUMNS = {
    'ztf': ['catflags', 'clrcoeff'],
    # custom: `is_upper_limit` marks a non-detection, which the pages draw
    # as a limit arrow rather than a point; telescope/instrument are the
    # only provenance these hand-assembled files carry.
    'custom': ['is_upper_limit', 'telescope', 'instrument'],
    # sdss: `mag` is modelMag (what the pages plot); psfMag is the other
    # standard choice and is cheap to carry, so it rides along.
    'sdss': ['psf_mag', 'psf_magerr'],
    # panstarrs: the two detection quality bitmasks, carried rather than
    # applied for the same reason as ZTF's catflags.
    'panstarrs': ['infoFlag', 'infoFlag2'],
    # wise/neowise/ptf/gaia emit the shared columns only.
}

# Reductions the *consumer* must apply to reproduce what the analysis pages
# show. Everything here is stored raw on purpose: each step is lossy and
# parameterised, so baking it in would freeze the choice and make the
# underlying measurements unrecoverable without re-fetching.
#
#   ztf              drop epochs where `catflags & 6141` (ZTF Explanatory
#                    Supplement 10.3; see Source_Analysis.ZTF_Local_LC)
#   wise, neowise    group exposures into visits with
#                    Light_Curve._cluster_by_gap(mjd, max_gap=5.0) before
#                    plotting -- WISE revisits a position several times over a
#                    couple of days. Unbinned vs binned is a ~10x difference in
#                    point count (95 raw NEOWISE exposures -> 20 visits), and
#                    it is the only reason stored light curves differ from
#                    Light_Curve.get_lc() output.
#   ptf, gaia        none
RENDER_TIME_REDUCTIONS = {
    'ztf': 'catflags & 6141 == 0',
    'wise': '_cluster_by_gap(max_gap=5.0)',
    'neowise': '_cluster_by_gap(max_gap=5.0)',
}

# Bucketed, not one sorted file. An earlier estimate put ZTF at ~4.9M rows /
# 0.14 GB; the real pull is ~264M epochs and the combined table is ~286M rows
# / ~7.5 GB, so a single file can neither be built in memory (~64 GB peak) nor
# read cheaply. Buckets are keyed on the first two hex characters of
# `source_uuid`, giving 256 files of ~30 MB that can each be written and read
# independently.
#
# Note this tier is for *analysis*. The website reads the per-source payloads
# in `source_db.serving`, never these files.
BUCKET_CHARS = 2


def bucket_of(source_uuid: str) -> str:
    """The bucket a source's rows live in."""
    return source_uuid[:BUCKET_CHARS]

# The catalogs Light_Curve queries by default, and which the pages plot.
CATALOGS = ['ztf', 'wise', 'neowise', 'ptf', 'sdss', 'panstarrs', 'gaia', 'custom']


# ── positions ─────────────────────────────────────────────────────────────────

def load_positions(out_dir: Path = DEFAULT_OUT_DIR,
                   limit: Optional[int] = None) -> pd.DataFrame:
    """`source_uuid`, `ra`, `dec` for every source in the built database.

    The uuid travels with the position into every bulk query, so results come
    back already keyed and no second crossmatch is needed to attach them.
    """
    path = Path(out_dir) / 'sources.parquet'
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found -- run `python -m source_db.build_catalog_tier` first')
    df = pd.read_parquet(path, columns=['source_uuid', 'ra', 'dec'])
    df = df[df['ra'].notna() & df['dec'].notna()].reset_index(drop=True)
    return df.head(limit) if limit else df


def row_chunks(positions: pd.DataFrame, size: int) -> list[tuple[str, pd.DataFrame]]:
    """Split positions into `(chunk_id, frame)` pairs for resumable fetching."""
    return [(f'{i // size:04d}', positions.iloc[i:i + size])
            for i in range(0, len(positions), size)]


# ── raw storage ───────────────────────────────────────────────────────────────

def all_covered(catalog: str, raw_dir: Path = RAW_DIR) -> set[str]:
    """Every source already fetched for `catalog`, across all chunks."""
    cov = covered_sources(catalog, raw_dir)
    return set().union(*cov.values()) if cov else set()


def next_chunk_index(catalog: str, raw_dir: Path = RAW_DIR) -> int:
    """One past the highest numeric chunk id on disk.

    New chunks must not reuse an id, or `write_raw` would overwrite a part
    file that is already accounted for.
    """
    idx = []
    for p in raw_dir_for(catalog, raw_dir).glob('part-*.parquet'):
        stem = p.stem.removeprefix('part-')
        if stem.isdigit():
            idx.append(int(stem))
    return max(idx) + 1 if idx else 0


def pending_chunks(catalog: str, positions: pd.DataFrame, size: int,
                   raw_dir: Path = RAW_DIR,
                   resume: bool = True) -> list[tuple[str, pd.DataFrame]]:
    """Chunks covering only the sources not yet fetched.

    Chunks here are *row-positional*, so inserting a source shifts every later
    row into a different chunk. Checking coverage per chunk id would then see
    every chunk as changed and refetch the lot. Coverage is therefore checked
    against the union of all fetched uuids, and the sources that remain are
    packed into new chunks numbered after the existing ones -- so adding 50
    sources to a finished catalog costs one chunk, not 169,052.

    ZTF does not use this: its chunk id is a sky partition, which is stable
    under insertion.
    """
    done = all_covered(catalog, raw_dir) if resume else set()
    todo = positions[~positions['source_uuid'].isin(done)].reset_index(drop=True)
    start = next_chunk_index(catalog, raw_dir) if resume else 0
    return [(f'{start + i // size:04d}', todo.iloc[i:i + size])
            for i in range(0, len(todo), size)]


def raw_dir_for(catalog: str, raw_dir: Path = RAW_DIR) -> Path:
    d = Path(raw_dir) / catalog
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw_path(catalog: str, chunk_id: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir_for(catalog, raw_dir) / f'part-{chunk_id}.parquet'


def write_raw(catalog: str, chunk_id: str, df: pd.DataFrame,
              raw_dir: Path = RAW_DIR, **meta) -> Path:
    """Write one raw chunk and record it in the manifest.

    Staged then swapped, so an interrupted write cannot leave a truncated part
    file that the next run would skip as complete.
    """
    path = raw_path(catalog, chunk_id, raw_dir)
    staging = path.with_suffix('.parquet.incoming')
    df.to_parquet(staging, compression='zstd', index=False)
    staging.replace(path)
    record_chunk(catalog, chunk_id, n_rows=len(df),
                 n_bytes=path.stat().st_size, raw_dir=raw_dir, **meta)
    return path


def iter_raw(catalog: str, raw_dir: Path = RAW_DIR) -> Iterator[pd.DataFrame]:
    for path in sorted(raw_dir_for(catalog, raw_dir).glob('part-*.parquet')):
        yield pd.read_parquet(path)


def iter_raw_batches(catalog: str, raw_dir: Path = RAW_DIR,
                     parts_per_batch: int = 40) -> Iterator[pd.DataFrame]:
    """Raw parts in groups, so a 264M-epoch catalog never lands in memory at once.

    ZTF writes ~5,072 parts of ~50k rows; 40 at a time is ~2M rows (~450 MB),
    which is the unit the streaming build normalises and writes.
    """
    paths = sorted(raw_dir_for(catalog, raw_dir).glob('part-*.parquet'))
    for i in range(0, len(paths), parts_per_batch):
        frames = [pd.read_parquet(p) for p in paths[i:i + parts_per_batch]]
        frames = [f for f in frames if len(f)]
        if frames:
            yield pd.concat(frames, ignore_index=True)


def n_raw_parts(catalog: str, raw_dir: Path = RAW_DIR) -> int:
    return len(list(raw_dir_for(catalog, raw_dir).glob('part-*.parquet')))


def read_raw(catalog: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    parts = list(iter_raw(catalog, raw_dir))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ── manifest ──────────────────────────────────────────────────────────────────

def manifest_path(raw_dir: Path = RAW_DIR) -> Path:
    return Path(raw_dir) / '_manifest.json'


def load_manifest(raw_dir: Path = RAW_DIR) -> dict:
    path = manifest_path(raw_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def record_chunk(catalog: str, chunk_id: str, raw_dir: Path = RAW_DIR, **meta) -> None:
    """Append one chunk's provenance to the shared manifest.

    Read-modify-write on a file every catalog shares, so it is done under an
    exclusive lock and swapped in atomically. Without the lock, two catalogs
    fetching at once lose each other's entries -- which is exactly what
    happened to `sdss_spectra` chunk 0004 while ztf and irsa were writing.
    """
    path = manifest_path(raw_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix('.json.lock')
    with open(lock, 'w') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            man = load_manifest(raw_dir)
            man.setdefault(catalog, {})[chunk_id] = {
                'fetched_at': datetime.now(timezone.utc).isoformat(), **meta}
            staging = path.with_suffix('.json.incoming')
            staging.write_text(json.dumps(man, indent=2, sort_keys=True))
            staging.replace(path)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def completed_chunks(catalog: str, raw_dir: Path = RAW_DIR) -> set[str]:
    """Chunk ids with both a part file and its coverage sidecar.

    Deliberately does not consult `_manifest.json`. That file is one shared
    read-modify-write across every catalog, so a concurrent fetch can drop an
    entry -- and a dropped entry used to mean re-fetching a chunk whose data
    was sitting right there. The part file and its sidecar are each written
    once and never rewritten, which makes them the honest record.
    """
    d = raw_dir_for(catalog, raw_dir)
    parts = {p.stem.removeprefix('part-') for p in d.glob('part-*.parquet')}
    covered = {p.name[len('part-'):-len('.sources')] for p in d.glob('part-*.sources')}
    return parts & covered


# ── source-level coverage ─────────────────────────────────────────────────────
#
# Resume has to be keyed on *sources*, not chunks. Chunks here are sky
# partitions shared by many sources, so "this partition is done" would silently
# skip a source added to it later -- and most newly added sources land in
# partitions that already exist. That failure looks like success, which is the
# worst kind.
#
# Coverage is stored beside each chunk rather than in `_manifest.json`, because
# `record_chunk` rewrites that whole file per chunk: folding ~29 uuids x 5,120
# chunks into it would mean rewriting a growing multi-MB JSON thousands of
# times. A sidecar is ~1 KB and written once.

def coverage_path(catalog: str, chunk_id: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir_for(catalog, raw_dir) / f'part-{chunk_id}.sources'


def write_coverage(catalog: str, chunk_id: str, source_uuids: Iterable[str],
                   raw_dir: Path = RAW_DIR) -> None:
    """Record every source this chunk *covered*, matched or not."""
    path = coverage_path(catalog, chunk_id, raw_dir)
    staging = path.with_suffix('.sources.incoming')
    staging.write_text('\n'.join(sorted(source_uuids)))
    staging.replace(path)


def read_coverage(catalog: str, chunk_id: str, raw_dir: Path = RAW_DIR) -> set[str]:
    path = coverage_path(catalog, chunk_id, raw_dir)
    if not path.exists():
        return set()
    return {ln for ln in path.read_text().splitlines() if ln}


def covered_sources(catalog: str, raw_dir: Path = RAW_DIR) -> dict[str, set[str]]:
    """`{chunk_id: covered source_uuids}` for everything already on disk."""
    done = completed_chunks(catalog, raw_dir)
    return {c: read_coverage(catalog, c, raw_dir) for c in done}


def empty_lightcurves() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(
        LIGHTCURVE_COLUMNS,
        ['string', 'string', 'string', 'float64', 'float64', 'float64',
         'float64', 'string'])})


# ── the normalised table ──────────────────────────────────────────────────────

# Every extra any survey emits. The bucket files share one schema so the 256
# of them can be read as a single dataset; a survey that does not emit a given
# extra leaves it null.
_EXTRA_TYPES = {
    'catflags': 'int32', 'clrcoeff': 'float64',
    'is_upper_limit': 'bool', 'telescope': 'string', 'instrument': 'string',
    'psf_mag': 'float64', 'psf_magerr': 'float64',
    'infoFlag': 'int64', 'infoFlag2': 'int64',
}

_SHARED_TYPES = {
    'source_uuid': 'string', 'survey': 'string', 'band': 'string',
    'mjd': 'float64', 'mag': 'float64', 'magerr': 'float64',
    'snr': 'float64', 'catalog_id': 'string',
}


def unified_schema() -> 'pa.Schema':
    """One schema for every bucket file: shared columns then all extras."""
    import pyarrow as pa
    tmap = {'string': pa.string(), 'float64': pa.float64(), 'int32': pa.int32(),
            'int64': pa.int64(), 'bool': pa.bool_()}
    fields = [pa.field(c, tmap[t]) for c, t in _SHARED_TYPES.items()]
    fields += [pa.field(c, tmap[t]) for c, t in _EXTRA_TYPES.items()]
    return pa.schema(fields)


def conform(df: pd.DataFrame) -> 'pa.Table':
    """Reindex a normalised frame onto `unified_schema()`."""
    import pyarrow as pa
    schema = unified_schema()
    out = {}
    for f in schema:
        if f.name in df.columns:
            col = df[f.name]
        else:
            col = pd.Series([None] * len(df), dtype='object')
        out[f.name] = pa.array(col, type=f.type, from_pandas=True)
    return pa.Table.from_arrays(list(out.values()), schema=schema)


def lightcurve_dir(out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    return Path(out_dir) / 'light_curves'


def bucket_path(bucket: str, out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    return lightcurve_dir(out_dir) / f'uuid_prefix={bucket}' / 'part-0.parquet'


class BucketedWriter:
    """Streams normalised rows into one parquet file per uuid prefix.

    Holding 256 open writers costs one row-group buffer each, which is what
    keeps the build flat in memory: rows are appended as they are normalised
    and never all held at once.
    """

    def __init__(self, out_dir: Path = DEFAULT_OUT_DIR, row_group_size: int = 50_000):
        import pyarrow.parquet as pq  # noqa: F401
        self.out_dir = Path(out_dir)
        self.row_group_size = row_group_size
        self.schema = unified_schema()
        self._writers: dict[str, object] = {}
        self.n_rows = 0

    def _writer(self, bucket: str):
        import pyarrow.parquet as pq
        if bucket not in self._writers:
            path = bucket_path(bucket, self.out_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._writers[bucket] = pq.ParquetWriter(
                path, self.schema, compression='zstd')
        return self._writers[bucket]

    def write(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        buckets = df['source_uuid'].str.slice(0, BUCKET_CHARS)
        for bucket, idx in df.groupby(buckets, sort=False).groups.items():
            sub = df.loc[idx]
            self._writer(str(bucket)).write_table(
                conform(sub), row_group_size=self.row_group_size)
            self.n_rows += len(sub)

    def close(self) -> None:
        for w in self._writers.values():
            w.close()
        self._writers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
