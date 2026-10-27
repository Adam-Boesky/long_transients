"""The process-wide filter engine, and the translation into it.

`CatalogFilters` reads `sources.parquet`, which carries the 314 science
columns but not the four the serving build counts afterwards from the
light-curve and spectrum tables. Those are loaded out of `science.sqlite`
once, at startup, and attached -- see `CatalogFilters.attach` for why they are
handed over rather than read in place.

Startup cost is ~4 s, almost all of it reading 198k rows of four columns out
of SQLite. It happens in the lifespan hook so the first request does not wear
it, and so a broken catalog fails the process rather than one endpoint.

Everything the list endpoints used to express as ad-hoc SQL is translated into
a filter document here instead, so there is one filter path rather than two
that can disagree. `legacy_filter` is that translation.
"""
from __future__ import annotations

import math
import re
import sqlite3
from typing import Any

from api.derive import CLASS_SOURCES, MATCHED_COLUMN
from source_db.serving.filters import UUID, CatalogFilters, FilterError
from source_db.constants import PSTARR_UPPER_LIM, SIGMA_CAP

# The columns counted at serving-build time, which therefore live in
# `science.sqlite` rather than in the parquet.
SERVING_COLUMNS = ('n_epochs', 'surveys', 'n_spectra', 'spec_surveys')

# PanSTARRS 5-sigma limits, from `Source_Analysis/store_src_plots`. Duplicated
# rather than imported for the reason `api.routers.table` gives: that module
# pulls in matplotlib with usetex at import time.


# Cells that mean "absent" but arrive as text, mirroring `derive._BLANK`.
_BLANK_SQL = "'', '<NA>', 'nan', 'None'"


def _finite(expr: str) -> str:
    """`_number`'s rule in SQL: a non-finite value reads as missing."""
    return f'CASE WHEN isfinite({expr}) THEN {expr} END'


def _classification_sql(available: set[str]) -> str:
    """`derive.classes_of`, rendered the way the table's column renders it.

    One string per source -- `SDSS QSO · SIMBAD Quasar · AGN-DB generic_agn`
    -- so a filter can say `contains 'QSO'` and mean what the column shows.
    The parts a caller would otherwise have to know: each catalog's classes
    are deduplicated and sorted, a catalog that matched but did not type the
    source contributes `unknown`, and AGN-DB's cell is a JSON list because it
    records one class per contributing sub-catalog.

    Checked against the Python implementation over 30,000 sampled sources:
    identical on every one.
    """
    parts = []
    for prefix, column, is_list in CLASS_SOURCES:
        matched = MATCHED_COLUMN[prefix]
        if column not in available or matched not in available:
            continue
        col, flag = f'"{column}"', f'"{matched}"'
        # `_truthy`: the crossmatch flag arrives as a bool, a number or text.
        truthy = (f"(CASE WHEN typeof({flag}) = 'VARCHAR' "
                  f"THEN lower(trim(CAST({flag} AS VARCHAR))) "
                  f"IN ('true','t','1','yes') "
                  f"ELSE COALESCE(TRY_CAST({flag} AS BOOLEAN), FALSE) END)")
        if is_list:
            items = (f"list_sort(list_distinct(list_filter("
                     f"  list_transform(COALESCE(TRY_CAST({col} AS VARCHAR[]), "
                     f"    CASE WHEN {col} IS NULL THEN [] "
                     f"    ELSE [CAST({col} AS VARCHAR)] END), x -> trim(x)), "
                     f"  x -> x NOT IN ({_BLANK_SQL}))))")
        else:
            items = (f"CASE WHEN {col} IS NULL "
                     f"  OR trim(CAST({col} AS VARCHAR)) IN ({_BLANK_SQL}) THEN [] "
                     f"ELSE [trim(CAST({col} AS VARCHAR))] END")
        parts.append(
            f"CASE WHEN len({items}) > 0 "
            f"THEN '{prefix} ' || array_to_string({items}, '/') "
            f"WHEN {truthy} THEN '{prefix} unknown' END")
    if not parts:
        return "CAST(NULL AS VARCHAR)"
    # NULL, not '', for a source no catalog typed. The empty string would read
    # as 100% filled and make `is_null` match nothing, when "nothing has typed
    # this" is one of the more useful things to ask for -- it is 89,358
    # sources, and the fill rate is how the picker warns you about a cut.
    return (f"nullif(array_to_string(list_filter([{', '.join(parts)}], "
            f"x -> x IS NOT NULL), ' · '), '')")


def _mean_abs_dmag_sql(available: set[str]) -> str:
    """`table._mean_abs_dmag` as a SQL expression.

    The mean of |ZTF - PanSTARRS| over the bands that have both, with the
    substitutions the Python version makes: a ZTF-only detection (flag 1) is
    differenced against the PanSTARRS limit, a PanSTARRS-only one (flag 2)
    against ZTF's own limiting magnitude.

    Derived rather than computed per row because it is the table's default
    sort. In Python it could only rank the rows already fetched, so "the
    largest |dmag|" meant "the largest among an arbitrary 5,000". Checked
    against the Python implementation over 20,000 sampled sources: identical
    on every one.

    Built from the bands this release actually has. SQL cannot name a column
    that is not there, so a catalog without the photometry -- the test
    fixtures, and any early build -- would fail to define the view at all
    rather than reporting no value. The Python version returns None for that
    case; this returns NULL, which is the same statement.
    """
    parts = []
    for band in 'gri':
        needed = {f'{band}_Catalog_Flag', f'PSTARR_{band}PSFMag',
                  f'ZTF_{band}PSFMag', f'ZTF_{band}_mag_limit'}
        if not needed <= available:
            continue
        flag = _finite(f'"{band}_Catalog_Flag"')
        ps = (f'CASE WHEN {flag} = 1 THEN {PSTARR_UPPER_LIM[band]} '
              f'ELSE {_finite(f'"PSTARR_{band}PSFMag"')} END')
        ztf = (f'CASE WHEN {flag} = 2 THEN {_finite(f'"ZTF_{band}_mag_limit"')} '
               f'ELSE {_finite(f'"ZTF_{band}PSFMag"')} END')
        parts.append(f'abs(({ztf}) - ({ps}))')
    if not parts:
        return 'CAST(NULL AS DOUBLE)'
    return f'list_avg(list_filter([{", ".join(parts)}], x -> x IS NOT NULL))'

def _mean_abs_dmag_sigma_sql(available: set[str]) -> str:
    """Mean |Δmag σ| over the bands that have one: the default sort.

    `{band}_dmag_sigma` is the significance of each band's Δmag against its
    field's KDE envelope, written by filtering (capped at 8). NULL when no band
    has one, as for releases built before the column existed.

    A band flagged `{band}_dmag_sigma_capped` counts as the cap: filtering
    before the Envelope_KDE.sigma fix stored the grid edge's value for a Δmag
    off the grid, so the most extreme sources ranked below moderate ones.
    """
    parts = []
    for b in 'gri':
        if f'{b}_dmag_sigma' not in available:
            continue
        sig = f'abs({_finite(f'"{b}_dmag_sigma"')})'
        if f'{b}_dmag_sigma_capped' in available:
            sig = f'CASE WHEN {sig} IS NOT NULL AND CAST("{b}_dmag_sigma_capped" AS BOOLEAN) THEN {SIGMA_CAP} ELSE {sig} END'
        parts.append(sig)
    if not parts:
        return 'CAST(NULL AS DOUBLE)'
    return f'list_avg(list_filter([{", ".join(parts)}], x -> x IS NOT NULL))'


# The pipeline's sub-catalogues, as tagged in `catalog_tier` by
# `source_db.catalog_tier.TIERS`, grouped into the names the `catalog=`
# parameter takes. Defined here alone -- `api.routers.sources` imports it --
# because two copies of this map is how the candidate list and the table come
# to disagree about what `in_ztf` means.
#
# A partition, not overlapping sets: these are offered as a segmented control,
# which promises mutually exclusive views.
CATALOG_TIERS = {
    'in_both': ('0',),
    'in_ztf': ('1',),
}

# Which name to show for a source, given its tier. Well defined because the
# groups above do not overlap.
CATALOG_OF_TIER = {tier: name for name, tiers in CATALOG_TIERS.items()
                   for tier in tiers}

_engine: CatalogFilters | None = None


def startup(science: sqlite3.Connection, parquet=None) -> CatalogFilters:
    """Build the engine and attach the serving columns. Call once.

    `parquet` defaults to the configured catalog; tests pass the one their
    fixture build wrote, so the engine and `science.sqlite` describe the same
    five synthetic sources rather than the engine filtering the real 198k.
    """
    global _engine
    engine = CatalogFilters(parquet)
    rows = science.execute(
        f'SELECT {UUID}, {", ".join(SERVING_COLUMNS)} FROM sources').fetchall()
    engine.attach('serving', _as_table(rows))
    # The analysis page's W1-W2 mean and median, from
    # `python -m source_db.build_wise_colors`. Optional: without it the
    # filters simply do not offer those two columns.
    wise = engine.parquet.parent / 'wise_colors.parquet'
    if wise.exists():
        import pyarrow.parquet as pq
        engine.attach('wise_colors', pq.read_table(wise))
    # One class per source, pooled from every catalogue's verdict. Voted in
    # Python rather than SQL -- the rules parse a JSON list, walk SIMBAD's
    # otype path and apply an ordered precedence -- and attached, so it
    # filters and sorts like any other column. 0.6 s for 198k rows.
    #
    # Optional, like `wise_colors`: a catalog missing any of the inputs (the
    # test fixtures' synthetic one) simply does not offer the column. Voting
    # on a subset would be worse than not voting, because the answer would
    # look authoritative.
    unified = _unified_table(engine.parquet)
    if unified is not None:
        engine.attach('unified', unified)
    available = {c.name for c in engine.columns()}
    engine.derive('mean_abs_dmag', _mean_abs_dmag_sql(available))
    engine.derive('mean_abs_dmag_sigma', _mean_abs_dmag_sigma_sql(available))
    engine.derive('classification', _classification_sql(available))
    engine.columns()                      # warm the schema so page 1 is fast
    _engine = engine
    return engine


def _unified_table(parquet) -> Any | None:
    """`source_uuid` -> `unified_class`, or None if the catalog lacks inputs."""
    import duckdb
    import pyarrow as pa
    from source_db.unified_class import INPUT_COLUMNS, unified_classes

    con = duckdb.connect()
    try:
        have = {r[0] for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{parquet}')").fetchall()}
        missing = [c for c in INPUT_COLUMNS if c not in have]
        if missing:
            print(f'  unified_class: not offered, catalog lacks {missing}')
            return None
        df = con.execute(
            f'SELECT {UUID}, {", ".join(INPUT_COLUMNS)} '
            f"FROM read_parquet('{parquet}')").df()
    finally:
        con.close()
    return pa.table({UUID: df[UUID].tolist(),
                     'unified_class': list(unified_classes(df))})


def _as_table(rows) -> Any:
    """Rows from sqlite3 into something DuckDB can register.

    A dict of columns rather than a DataFrame: this is four columns of 198k
    values and pandas is not needed to carry them.
    """
    import pyarrow as pa

    names = (UUID, *SERVING_COLUMNS)
    return pa.table({name: [r[i] for r in rows] for i, name in enumerate(names)})


def shutdown() -> None:
    global _engine
    if _engine is not None:
        _engine.close()
        _engine = None


def get_filters() -> CatalogFilters:
    """FastAPI dependency. Read-only after startup, so it is shared."""
    if _engine is None:
        raise RuntimeError('filter engine not started')
    return _engine


# ── translating the old query parameters ──────────────────────────────────────

# Default search radius for an RA/Dec query.
SEARCH_RADIUS_ARCSEC = 5.0


def _slug_form(q: str) -> str:
    """Decimal coordinates as the slug spells them: '-3.29' -> 'n3p29'."""
    return re.sub(r'-(?=\d)', 'n', q.strip()).replace('.', 'p')


def _parse_position(q: str) -> tuple[float, float, float] | None:
    """(ra, dec, radius_deg) from 'ra dec [radius_arcsec]' in decimal
    degrees, or a sexagesimal position; None if q is not a position."""
    tokens = q.replace(',', ' ').split()
    try:
        nums = [float(t) for t in tokens]
    except ValueError:
        nums = None
    if nums is not None and len(nums) in (2, 3):
        ra, dec = nums[0], nums[1]
        radius = nums[2] if len(nums) == 3 else SEARCH_RADIUS_ARCSEC
    elif re.search(r'\d[:hdms]', q) or (nums is not None and len(nums) == 6):
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        try:
            c = SkyCoord(q, unit=(u.hourangle, u.deg))
        except Exception:
            return None
        ra, dec, radius = c.ra.deg, c.dec.deg, SEARCH_RADIUS_ARCSEC
    else:
        return None
    if not (0 <= ra < 360 and -90 <= dec <= 90 and radius > 0):
        return None
    return ra, dec, radius / 3600


def _position_clause(ra: float, dec: float, r: float) -> dict[str, Any]:
    """A box of half-width r (deg) on the sky around (ra, dec)."""
    cos_dec = math.cos(math.radians(min(abs(dec) + r, 90)))
    dra = 180 if cos_dec < r / 180 else r / cos_dec
    lo, hi = ra - dra, ra + dra
    # Split the RA range where it wraps through 0/360.
    if dra >= 180:
        ra_arms = []
    elif lo < 0:
        ra_arms = [[lo + 360, 360], [0, hi]]
    elif hi >= 360:
        ra_arms = [[lo, 360], [0, hi - 360]]
    else:
        ra_arms = [[lo, hi]]
    clauses: list[dict[str, Any]] = [
        {'column': 'dec', 'op': 'between', 'value': [dec - r, dec + r]}]
    if ra_arms:
        clauses.append({'op': 'or', 'clauses': [
            {'column': 'ra', 'op': 'between', 'value': v} for v in ra_arms]})
    return {'op': 'and', 'clauses': clauses}

def legacy_filter(*, survey: str | None = None, min_epochs: int | None = None,
                  has_lightcurve: bool = False, has_spectrum: bool = False,
                  catalog: str | None = None, q: str | None = None,
                  exclude_uuids: list[str] | None = None) -> dict[str, Any] | None:
    """The list endpoint's original parameters as a filter document.

    Each of these used to be a hand-written fragment of SQL in `_filter_sql`.
    Expressing them in the same language as a user-built filter is what lets
    the two compose -- "in_ztf, at least 50 epochs, and whatever I just built
    in the sidebar" is one document, one query, one pagination.
    """
    clauses: list[dict[str, Any]] = []

    if q:
        # Substring on the coord slug or a uuid prefix, as before. Both arms
        # are kept because people paste either. Decimal numbers also match
        # the slug's spelling, and a position matches within a radius.
        arms: list[dict[str, Any]] = [
            {'column': 'coord_slug', 'op': 'contains', 'value': q},
            {'column': UUID, 'op': 'startswith', 'value': q},
        ]
        slug = _slug_form(q)
        if slug != q:
            arms.append({'column': 'coord_slug', 'op': 'contains', 'value': slug})
        pos = _parse_position(q)
        if pos is not None:
            arms.append(_position_clause(*pos))
        clauses.append({'op': 'or', 'clauses': arms})
    if catalog:
        if catalog not in CATALOG_TIERS:
            raise FilterError(f'catalog must be one of {sorted(CATALOG_TIERS)}')
        clauses.append({'column': 'catalog_tier', 'op': 'in',
                        'value': list(CATALOG_TIERS[catalog])})
    if has_spectrum:
        clauses.append({'column': 'n_spectra', 'op': '>', 'value': 0})
    if survey:
        clauses.append({'column': 'surveys', 'op': 'has_element', 'value': survey})
    if min_epochs:
        clauses.append({'column': 'n_epochs', 'op': '>=', 'value': min_epochs})
    if has_lightcurve:
        clauses.append({'column': 'n_epochs', 'op': '>', 'value': 0})
    if exclude_uuids:
        clauses.append({'column': UUID, 'op': 'not_in', 'value': exclude_uuids})

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {'op': 'and', 'clauses': clauses}



def combine(*docs: Any) -> dict[str, Any] | None:
    """AND together the documents that are actually present."""
    present = [d for d in docs if d]
    if not present:
        return None
    return present[0] if len(present) == 1 else {'op': 'and', 'clauses': present}


# DuckDB binds a filter's values untouched, so a type mismatch surfaces at
# execution as `duckdb.Error` rather than `FilterError` -- and a hand-edited
# deep link with `{"column":"ra","op":"between","value":["a","b"]}` became a
# 500 on every list endpoint. Routers catch this tuple instead.
def _duckdb_error() -> type[Exception]:
    import duckdb
    return duckdb.Error


BAD_FILTER = (FilterError, _duckdb_error())
