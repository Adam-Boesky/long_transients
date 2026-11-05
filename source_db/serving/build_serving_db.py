"""Build `science.sqlite`: one row per source, ready to serve.

    python -m source_db.serving.build_serving_db
    python -m source_db.serving.build_serving_db --limit 5000

Reads the analysis tiers -- `Data/db/sources.parquet` and the bucketed
`Data/db/light_curves/` -- and writes one payload per source. Nothing here is
a source of truth: delete the file and rebuild it.

Why precompute
--------------
Every page load is "give me everything for source X" on data that never
changes. Materialising that exact answer turns a page load into one
primary-key fetch, and shrinks the ~7.5 GB analysis tier to ~2 GB because the
payloads compress well and carry no per-row keys.

It also makes the storage choice reversible: the unit of storage is now the
page payload, so moving from SQLite to Postgres or flat files later is a
change to one function, not to the frontend or the build.

Memory stays flat because buckets are processed one at a time; a bucket is
~25 MB on disk at full ZTF.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import orjson  # noqa: E402
import pandas as pd  # noqa: E402

from source_db.lightcurves import _base  # noqa: E402
from source_db.paths import DEFAULT_OUT_DIR, RAW_DIR  # noqa: E402
from source_db.serving import payload as payload_mod  # noqa: E402
from source_db.serving.schema import LIST_COLUMNS, SCIENCE_DB, SCIENCE_SCHEMA  # noqa: E402


def _connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    # Bulk-load settings; this file is rebuilt wholesale, so durability during
    # the build buys nothing.
    con.execute('PRAGMA journal_mode=OFF')
    con.execute('PRAGMA synchronous=OFF')
    return con


# JSON numbers are IEEE doubles once a browser parses them, so an integer
# beyond 2^53 comes back changed: `row_hash` 6867325581823111965 reads as
# ...112192 in JavaScript. Three catalog columns are affected, and
# `PSTARR_PanSTARR_ID` is 100% affected -- every PanSTARRS identifier would
# be wrong in the UI. Emit those as strings; they are identifiers, never
# arithmetic operands.
JS_SAFE_INTEGER = 2 ** 53


def _json_safe(value):
    """An int too large for a double becomes a string, everything else passes."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and abs(value) > JS_SAFE_INTEGER:
        return str(value)
    return value


def _write_sources(con: sqlite3.Connection, out_dir: Path, limit: int | None) -> int:
    """The catalog row: list columns promoted, the other ~300 as JSON."""
    src = pd.read_parquet(Path(out_dir) / 'sources.parquet')
    if limit:
        src = src.head(limit)
    promoted = [c for c in LIST_COLUMNS if c in src.columns]
    rest = [c for c in src.columns if c not in promoted and c != 'source_uuid']

    rows = []
    for rec in src.to_dict(orient='records'):
        uuid = rec['source_uuid']
        blob = orjson.dumps({k: _json_safe(rec[k]) for k in rest},
                            option=orjson.OPT_SERIALIZE_NUMPY)
        tier = rec.get('catalog_tier')
        rows.append((uuid,
                     rec.get('coord_slug'),
                     None if pd.isna(rec.get('ra')) else float(rec['ra']),
                     None if pd.isna(rec.get('dec')) else float(rec['dec']),
                     None if tier is None or pd.isna(tier) else str(tier),
                     blob))
    con.executemany(
        'INSERT OR REPLACE INTO sources '
        '(source_uuid, coord_slug, ra, dec, catalog_tier, catalog_json) '
        'VALUES (?,?,?,?,?,?)',
        rows)
    con.commit()
    return len(rows)


def _write_lightcurves(con: sqlite3.Connection, out_dir: Path,
                       wanted: set[str] | None) -> tuple[int, int]:
    """One payload per source, a bucket at a time."""
    lc_dir = _base.lightcurve_dir(out_dir)
    buckets = sorted(lc_dir.glob('uuid_prefix=*/part-0.parquet'))
    if not buckets:
        raise FileNotFoundError(
            f'no buckets under {lc_dir} -- run `python -m '
            f'source_db.build_lightcurve_table` first')

    n_src = n_rows = 0
    t0 = time.time()
    for i, path in enumerate(buckets, 1):
        df = pd.read_parquet(path)
        if wanted is not None:
            df = df[df['source_uuid'].isin(wanted)]
        if df.empty:
            continue
        batch = []
        for uuid, g in df.groupby('source_uuid', sort=False):
            blob, n, surveys = payload_mod.build_payload(g, str(uuid))
            batch.append((str(uuid), n, ','.join(surveys), blob))
            n_rows += n
        con.executemany(
            'INSERT OR REPLACE INTO light_curves '
            '(source_uuid, n_epochs, surveys, payload) VALUES (?,?,?,?)', batch)
        con.commit()
        n_src += len(batch)
        print(f'  bucket {i:>3}/{len(buckets)}  {n_src:,} sources  '
              f'{n_rows:,} epochs  {(time.time() - t0) / 60:.1f} min',
              end='\r', flush=True)
        del df, batch
    print()
    return n_src, n_rows


def current_sdss_spectra(all_spec: pd.DataFrame, current: set[str]) -> pd.DataFrame:
    """Drop SDSS spectra that are not among the source's current matches.

    `raw/` keeps every spectrum ever fetched, so after a catalog rebuild a
    source can hold a spectrum it no longer matches (an unplugged fibre, say).
    `current` holds `uuid|specobjid` keys, as `spectra.sdss.coverage_keys`.
    """
    key = all_spec['source_uuid'].astype(str) + '|' + all_spec['spec_id'].astype(str)
    stale = (all_spec['survey'] == 'sdss') & ~key.isin(current)
    if stale.any():
        print(f'  dropped {int(stale.sum()):,} SDSS spectra that are no longer '
              f'their source\'s matches')
    return all_spec[~stale]


def current_sdss_keys(out_dir: Path) -> set[str]:
    """`uuid|specobjid` for every SDSS spectrum the sources currently match."""
    from source_db.spectra import sdss as spec_sdss
    return set(spec_sdss.coverage_keys(spec_sdss.targets(out_dir)))


def _write_spectra(con: sqlite3.Connection, raw_dir: Path, out_dir: Path,
                   wanted: set[str] | None) -> tuple[int, int]:
    """One payload per source that has any spectrum.

    Spectra are read straight from `raw/`: unlike the light curves, the fetch
    modules already emit the normalised schema, so there is no separate melt
    stage to sit between them.
    """
    from source_db.spectra import _base as spec_base

    frames = []
    for catalog in ('sdss_spectra', 'desi_spectra'):
        if spec_base.raw_dir_for(catalog, raw_dir).exists():
            df = spec_base.read_raw(catalog, raw_dir)
            if len(df):
                frames.append(df)
    if not frames:
        return 0, 0

    all_spec = current_sdss_spectra(pd.concat(frames, ignore_index=True),
                                    current_sdss_keys(out_dir))
    if wanted is not None:
        all_spec = all_spec[all_spec['source_uuid'].isin(wanted)]

    # One spectrum per (source, survey, spec_id). SPARCL can return the same
    # specid more than once for a DESI targetid, and the fetch keeps whatever
    # it returns, so 2,262 sources carried 2-4 copies of a single spectrum --
    # overplotted on the page and repeated in the legend. Every affected
    # source had exactly one distinct spec_id, differing only in redshift at
    # the 1e-4 level, so these are duplicate retrievals rather than genuine
    # repeat coadds and dropping them loses no observation.
    #
    # Deduplicated here rather than only at fetch time so a rebuild repairs
    # the raw store already on disk, and so a future fetch that duplicates
    # again cannot reach the page.
    before = len(all_spec)
    all_spec = all_spec.drop_duplicates(
        subset=['source_uuid', 'survey', 'spec_id'], keep='first')
    if len(all_spec) != before:
        print(f'  dropped {before - len(all_spec):,} duplicate spectra '
              f'(same source, survey and spec_id)')

    n_src = n_spec = 0
    batch = []
    for uuid, g in all_spec.groupby('source_uuid', sort=False):
        blob, n, surveys = payload_mod.build_spectrum_payload(g, str(uuid))
        if n == 0:
            continue
        batch.append((str(uuid), ','.join(surveys), n, blob))
        n_spec += n
        n_src += 1
    if batch:
        con.executemany(
            'INSERT OR REPLACE INTO spectra '
            '(source_uuid, surveys, n_spectra, payload) VALUES (?,?,?,?)', batch)
        con.commit()
    return n_src, n_spec


def _write_cutouts(con: sqlite3.Connection, raw_dir: Path,
                   wanted: set[str] | None) -> int:
    """Where every stamp's pixels come from, both surveys, from `raw/`.

    Symmetric on purpose. PanSTARRS resolution is per source and gives a
    stack filename; ZTF resolution is per *pointing* -- ~8 sources share one
    -- and gives the bands that have a reference image, which is expanded
    back out to sources here. Neither is discovered by rendering, so a
    rebuild reproduces both instead of losing one.
    """
    from source_db.lightcurves import _base as store

    rows: list[tuple[str, str, str, str]] = []

    d = store.raw_dir_for('ps1_cutouts', raw_dir)
    if d.exists() and list(d.glob('part-*.parquet')):
        ps1 = store.read_raw('ps1_cutouts', raw_dir)
        if wanted is not None:
            ps1 = ps1[ps1['source_uuid'].isin(wanted)]
        rows += [(str(u), 'panstarrs', str(b), str(f)) for u, b, f
                 in zip(ps1['source_uuid'], ps1['band'], ps1['filename'])]

    d = store.raw_dir_for('ztf_cutouts', raw_dir)
    if d.exists() and list(d.glob('part-*.parquet')):
        avail = store.read_raw('ztf_cutouts', raw_dir)
        pointings = _source_pointings(con)
        if not avail.empty and not pointings.empty:
            joined = pointings.merge(avail, on=['fieldid', 'ccdid', 'qid'],
                                     how='inner')
            rows += [(str(u), 'ztf', str(b), f'{int(f)}/{int(c)}/{int(q)}')
                     for u, b, f, c, q in zip(
                         joined['source_uuid'], joined['band'],
                         joined['fieldid'], joined['ccdid'], joined['qid'])]

    if not rows:
        return 0
    con.executemany('INSERT OR REPLACE INTO cutout_sources '
                    '(source_uuid, survey, band, origin) VALUES (?,?,?,?)', rows)
    con.commit()
    return len({r[0] for r in rows})


def _source_pointings(con: sqlite3.Connection) -> 'pd.DataFrame':
    """source_uuid -> its ZTF field/ccd/quadrant, read back off the blob."""
    import pandas as pd

    out = []
    for uuid, blob in con.execute('SELECT source_uuid, catalog_json FROM sources'):
        cat = orjson.loads(blob)
        try:
            out.append((uuid, int(cat['fieldid']), int(cat['ccdid']),
                        int(cat['qid'])))
        except (KeyError, TypeError, ValueError):
            continue
    return pd.DataFrame(out, columns=['source_uuid', 'fieldid', 'ccdid', 'qid'])


def build(out_dir: Path = DEFAULT_OUT_DIR, limit: int | None = None,
          db_path: Path | None = None) -> Path:
    path = Path(db_path) if db_path else Path(out_dir) / SCIENCE_DB
    if path.exists():
        path.unlink()
    con = _connect(path)
    con.executescript(SCIENCE_SCHEMA)

    t0 = time.time()
    n_src = _write_sources(con, out_dir, limit)
    print(f'  sources: {n_src:,}')

    wanted = None
    if limit:
        wanted = {r[0] for r in con.execute('SELECT source_uuid FROM sources')}
    n_lc, n_rows = _write_lightcurves(con, out_dir, wanted)

    n_spec_src, n_spec = _write_spectra(con, RAW_DIR, out_dir, wanted)
    print(f'  spectra: {n_spec:,} for {n_spec_src:,} sources')

    n_cut = _write_cutouts(con, RAW_DIR, wanted)
    print(f'  cutouts: {n_cut:,} sources with image addresses')

    # Denormalise the light-curve and spectrum summaries onto `sources` so the
    # candidate list can sort and filter on them without touching the blobs.
    con.execute("""
        UPDATE sources SET
          n_epochs = COALESCE((SELECT n_epochs FROM light_curves
                               WHERE light_curves.source_uuid = sources.source_uuid), 0),
          surveys  = (SELECT surveys FROM light_curves
                      WHERE light_curves.source_uuid = sources.source_uuid),
          n_spectra = COALESCE((SELECT n_spectra FROM spectra
                                WHERE spectra.source_uuid = sources.source_uuid), 0),
          spec_surveys = (SELECT surveys FROM spectra
                          WHERE spectra.source_uuid = sources.source_uuid)
    """)
    # Built after the UPDATE above, so the bulk load never maintains it.
    # Without this, ORDER BY n_epochs scans all 169k rows (~140 ms/request).
    con.execute('CREATE INDEX IF NOT EXISTS idx_sources_epochs ON sources(n_epochs)')
    con.execute('CREATE INDEX IF NOT EXISTS idx_sources_nspec ON sources(n_spectra)')
    con.executemany('INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)', [
        ('built_at', time.strftime('%Y-%m-%dT%H:%M:%S')),
        ('n_sources', str(n_src)),
        ('n_sources_with_lc', str(n_lc)),
        ('n_epochs', str(n_rows)),
        ('n_sources_with_spectra', str(n_spec_src)),
        ('n_spectra', str(n_spec)),
        ('n_sources_with_cutouts', str(n_cut)),
    ])
    con.commit()
    # No VACUUM: the file is unlinked and recreated every build, so there is
    # no free space to reclaim and it would just rewrite several GB.
    con.close()

    print(f'\n{path} ({path.stat().st_size / 1e9:.2f} GB)')
    print(f'  {n_src:,} sources, {n_lc:,} with light curves, {n_rows:,} epochs, '
          f'{(time.time() - t0) / 60:.1f} min')
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument('--db-path', type=Path, default=None)
    ap.add_argument('--limit', type=int, default=None,
                    help='first N sources only, for a quick smoke build')
    a = ap.parse_args()
    build(a.out_dir, a.limit, a.db_path)


if __name__ == '__main__':
    main()
