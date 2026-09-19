"""Query helpers for the local SIMBAD object catalog.

SIMBAD's `basic` table (~22.2M rows) is the CDS cross-identification database:
one row per astronomical object, carrying its canonical name, object type,
redshift, astrometry and literature count. Despite its reputation it is
*smaller* than DESI DR1, so it can be held locally rather than needing WISE's
remote-with-cache arrangement.

Bulk and per-source lookups take different routes, as in `catalogs.agn`:
`cone_search_many` matches against `_base.LocalCatalog`'s in-memory KD-tree,
while `cone_search` queries the parquet through DuckDB so a `multiprocessing`
worker never has to hold the ~8 GB catalog. See the per-source section below.

What it adds over the other catalogs is a name and a type for objects that never
got a spectroscopic fibre. Roughly a quarter of source positions match at 1.5",
and the matches SDSS/DESI/AGN-DB/TNS all miss come overwhelmingly from
photometric and morphological compilations -- HyperLEDA, 2MASX, the Zwicky
catalogs, NGC/IC/UGC -- which no other catalog here covers.

`simbad_otype` is SIMBAD's terse type code ('QSO', 'EB*', 'AG?'). The
`otypedef` table is joined in at build time so each row also carries the
human-readable `simbad_otype_label` and the full hierarchy in
`simbad_otype_path` ('G > AGN > QSO', '* > ** > CV*'). Prefer the path over
matching type codes by hand:

    stars = df.simbad_otype_path.str.startswith('*')
    agn   = df.simbad_otype_path.str.startswith('G > AGN')

`simbad_is_candidate` separates SIMBAD's candidate types from confirmed ones.
Check it whenever confirmation matters, because neither of the other two type
columns will tell you: 'AG?' (AGN candidate) and 'AGN' share the path
'G > AGN', so a path prefix filter silently includes candidates alongside
confirmed objects. `simbad_otype_label` does spell it out ('Active Galaxy
Nucleus Candidate'), but only as free text.

No `row_filter`: unlike SDSS's specprimary or DESI's zcat_primary there is
nothing to de-duplicate, since `basic` is already one row per object. Position
quality is carried in `simbad_coo_qual` ('A' best, 'E' worst) for callers that
want to cut on it -- about 2.6% of the catalog is 'E', and those coordinates are
poor enough that they mostly fail to match at all rather than matching wrongly.

`scripts/build_simbad_catalog.py` downloads the raw chunks and calls
`build_parquet()`. Like TNS -- and unlike the frozen SDSS DR17 and DESI DR1
releases -- SIMBAD grows continuously, so this catalog goes stale and wants an
occasional refresh.

Memory, measured rather than estimated: the DataFrame itself is 3.3 GB, of
which 1.5 GB is `simbad_main_id` alone (22M unique strings as object dtype).
Process RSS is higher -- ~6.4 GB after the load, ~8.3 GB once `coords()` has
built the KD-tree, with a ~10.3 GB transient spike during the parquet read
while pyarrow holds both the Arrow table and the pandas copy. Reading in row
groups would flatten that spike, but only by changing `_base.frame()` for every
catalog. It stacks with DESI's ~13 GB in a single process, so budget ~25 GB for
an enrichment run that touches both.

A pyarrow string backend would roughly halve `simbad_main_id`, but pandas does
not preserve that dtype across a parquet round trip (it reads back as
`string[python]`), so it would again mean changing `_base.frame()` for all
catalogs.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import astropy.units as u
import duckdb
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from Source_Analysis.catalogs._base import LocalCatalog

_DATA_DIR = Path(__file__).resolve().parents[2] / 'Data' / 'catalogs'
CATALOG_RAW_DIR = _DATA_DIR / 'raw' / 'simbad'
CATALOG_PARQUET = _DATA_DIR / 'simbad_basic_slim.parquet'

TAP_SYNC_URL = 'https://simbad.cds.unistra.fr/simbad/sim-tap/sync'

# CDS caps one TAP response at 2M rows, so the download pages through `oid` --
# the primary key. Paging on sky position instead would spike over the Galactic
# plane and overrun the cap; `oid` is uniform. About 70% of the id space is
# populated, so this stride yields ~1.05M rows per page, leaving good headroom.
OID_STRIDE = 1_500_000
TAP_MAXREC = 2_000_000

# (SIMBAD TAP column, output column). The download queries exactly these, so
# this list is the single source of truth for the schema.
_COLUMNS: list[tuple[str, str]] = [
    ('main_id', 'simbad_main_id'),
    ('ra', 'ra'),
    ('dec', 'dec'),
    ('otype', 'simbad_otype'),
    ('sp_type', 'simbad_sp_type'),
    ('morph_type', 'simbad_morph_type'),
    ('rvz_redshift', 'simbad_z'),
    ('rvz_type', 'simbad_z_type'),
    ('rvz_qual', 'simbad_z_qual'),
    ('plx_value', 'simbad_plx'),
    ('plx_err', 'simbad_plx_err'),
    ('pmra', 'simbad_pmra'),
    ('pmdec', 'simbad_pmdec'),
    ('nbref', 'simbad_nbref'),
    ('coo_qual', 'simbad_coo_qual'),
]

# Derived at build time from the 226-row `otypedef` table, inserted after
# `simbad_otype` so the type columns stay together.
_DERIVED = ['simbad_otype_label', 'simbad_otype_path', 'simbad_is_candidate']

# All low cardinality. `simbad_sp_type` has the most unique values but is null
# for most rows, so it still collapses well.
_CATEGORICAL = {'simbad_otype', 'simbad_otype_label', 'simbad_otype_path',
                'simbad_sp_type', 'simbad_morph_type', 'simbad_z_type',
                'simbad_z_qual', 'simbad_coo_qual'}

# SIMBAD's text fields, read with an explicit dtype rather than left to
# inference. A page in which one of these is entirely empty is inferred as
# float64, and concatenating a float-categoried column with a string-categoried
# one produces categories of mixed type that pyarrow cannot map to an Arrow
# type. Not hypothetical: `morph_type` is empty on 6 of the 21 pages.
_TEXT_NATIVE = {'main_id', 'otype', 'sp_type', 'morph_type',
                'rvz_type', 'rvz_qual', 'coo_qual'}

# Inverse of the build-time rename, for callers who want SIMBAD's own field
# names back when reporting a single object.
NATIVE_NAMES: dict[str, str] = {dst: src for src, dst in _COLUMNS}

_CATALOG = LocalCatalog(CATALOG_PARQUET, prefix='simbad')


def basic_adql(oid_lo: int, oid_hi: int) -> str:
    """ADQL for one `oid` page of the `basic` columns this module keeps."""
    cols = ', '.join(src for src, _ in _COLUMNS)
    return f'SELECT {cols} FROM basic WHERE oid BETWEEN {oid_lo} AND {oid_hi}'


OTYPEDEF_ADQL = 'SELECT otype, otype_longname, path, is_candidate FROM otypedef'
COUNT_ADQL = 'SELECT COUNT(*) AS n FROM basic'
OID_RANGE_ADQL = 'SELECT MIN(oid) AS lo, MAX(oid) AS hi FROM basic'


def build_parquet(
    raw_dir: Path = CATALOG_RAW_DIR,
    out_path: Path = CATALOG_PARQUET,
) -> Path:
    """Convert the downloaded SIMBAD pages into the slim crossmatching parquet.

    Reads every `basic_*.csv` page in `raw_dir` plus `otypedef.csv`. Categoricals
    are applied per page so the concatenation never holds 22M rows of Python
    strings for the low-cardinality columns, then re-applied afterwards because
    concatenating categoricals with differing categories falls back to object.
    """
    chunk_paths = sorted(raw_dir.glob('basic_*.csv'))
    if not chunk_paths:
        raise FileNotFoundError(
            f'no basic_*.csv pages in {raw_dir} -- run scripts/build_simbad_catalog.py')
    otypedef_path = raw_dir / 'otypedef.csv'
    if not otypedef_path.exists():
        raise FileNotFoundError(
            f'missing {otypedef_path} -- run scripts/build_simbad_catalog.py')

    native = [src for src, _ in _COLUMNS]
    dtypes = {c: str for c in _TEXT_NATIVE}
    pieces = []
    for path in chunk_paths:
        df = pd.read_csv(path, usecols=native, dtype=dtypes, low_memory=False)
        df = df.rename(columns=dict(_COLUMNS))[[dst for _, dst in _COLUMNS]]
        for c in _CATEGORICAL & set(df.columns):
            df[c] = df[c].astype('category')
        pieces.append(df)
        print(f'  {path.name}: {len(df):,} rows')
    df = pd.concat(pieces, ignore_index=True)
    del pieces
    print(f'  {len(df):,} rows total')

    # Denormalise the object-type hierarchy. Mapping through object dtype rather
    # than merging keeps categorical join semantics out of it.
    defs = pd.read_csv(otypedef_path)
    # `?` (unknown nature) and `var` (variable source) are top-level types that
    # otypedef leaves with a null path. Every other top level -- 'G', '*' -- is
    # its own path, so fill these in the same way; otherwise the column cannot
    # be used for prefix filtering without null handling at every call site.
    defs['path'] = defs['path'].fillna(defs['otype'])
    otype = df['simbad_otype'].astype(object)
    df['simbad_otype_label'] = otype.map(dict(zip(defs['otype'], defs['otype_longname'])))
    df['simbad_otype_path'] = otype.map(dict(zip(defs['otype'], defs['path'])))
    df['simbad_is_candidate'] = otype.map(
        dict(zip(defs['otype'], defs['is_candidate'].astype(bool)))).astype('boolean')
    unknown = df['simbad_otype_path'].isna() & df['simbad_otype'].notna()
    if unknown.any():
        print(f'  {unknown.sum():,} rows carry an otype absent from otypedef: '
              f'{sorted(df.loc[unknown, "simbad_otype"].astype(str).unique())[:10]}')

    n_before = len(df)
    df = df[np.isfinite(df['ra']) & np.isfinite(df['dec'])].reset_index(drop=True)
    if len(df) != n_before:
        print(f'  dropped {n_before - len(df):,} rows with missing coordinates')

    order: list[str] = []
    for _, dst in _COLUMNS:
        order.append(dst)
        if dst == 'simbad_otype':
            order.extend(_DERIVED)
    df = df[order]

    for c in _CATEGORICAL:
        df[c] = df[c].astype('category')

    df.to_parquet(out_path, compression='zstd', index=False)
    return out_path


def load_catalog() -> pd.DataFrame:
    """The slim SIMBAD catalog, cached. Read-only -- see `_base`."""
    return _CATALOG.frame()


def cone_search_many(
    ra: Sequence[float],
    dec: Sequence[float],
    radius_arcsec: float = 2.0,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Nearest SIMBAD object within `radius_arcsec` for each input position."""
    return _CATALOG.match_nearest(ra, dec, radius_arcsec, columns=columns)


# ---------------------------------------------------------------------------
# Per-source path
#
# `cone_search` goes through DuckDB against the parquet rather than the
# in-memory LocalCatalog that backs `cone_search_many`. Holding SIMBAD costs
# ~8 GB, which is acceptable once in a bulk enrichment process but not once per
# worker in `store_src_plots.py`'s Pool. DuckDB reads only the columns a query
# touches -- for the coordinate pre-filter that is just `ra` and `dec` -- so a
# worker pays nothing to keep the catalog resident. Measured at 0.05-0.09 s per
# position, against multi-second plot generation.
#
# `catalogs.agn` splits the same way for the same reason. Both functions emit
# `simbad_sep_arcsec`, so they stay drop-in compatible with each other.
# ---------------------------------------------------------------------------

def cone_search(
    ra: float,
    dec: float,
    radius_arcsec: float = 2.0,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """All SIMBAD objects within `radius_arcsec` of one position, nearest first.

    Pre-filters on a padded RA/Dec box, which DuckDB answers by reading only the
    two coordinate columns, then applies an exact angular cut to the remainder.
    """
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return pd.DataFrame()

    radius_deg = radius_arcsec / 3600.0
    dec_pad = radius_deg
    # RA degrees shrink as cos(dec), so the box has to widen by the same factor
    # to still cover `radius_arcsec` on the sky.
    ra_pad = radius_deg / max(math.cos(math.radians(dec)), 1e-6)

    box_cols = list(columns) if columns else None
    if box_cols is not None:
        box_cols = ['ra', 'dec'] + [c for c in box_cols if c not in ('ra', 'dec')]
    select = ', '.join(f'"{c}"' for c in box_cols) if box_cols else '*'

    box = duckdb.sql(f"""
        SELECT {select} FROM read_parquet('{CATALOG_PARQUET}')
        WHERE ra BETWEEN {ra - ra_pad} AND {ra + ra_pad}
          AND dec BETWEEN {dec - dec_pad} AND {dec + dec_pad}
    """).df()
    if box.empty:
        return box

    sep = SkyCoord(ra=box['ra'].values * u.deg, dec=box['dec'].values * u.deg).separation(
        SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    ).arcsec
    keep = sep <= radius_arcsec
    box = box.loc[keep].copy()
    box['simbad_sep_arcsec'] = sep[keep]
    return box.sort_values('simbad_sep_arcsec').reset_index(drop=True)


if __name__ == '__main__':
    print(f'Building {CATALOG_PARQUET.name} from {CATALOG_RAW_DIR}...')
    path = build_parquet()
    print(f'Wrote {path} ({path.stat().st_size / 1e6:.0f} MB)')
    frame = load_catalog()
    print(f'  {len(frame):,} objects')
