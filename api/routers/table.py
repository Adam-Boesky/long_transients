"""One compact row per source, for the Table page.

Split in two, because the two halves have opposite lifetimes:

  `/table`              the catalog half. Static for a release, so it is
                        derived once, cached as gzipped bytes and handed to
                        the client untouched -- the same contract the light
                        curves use.
  `/table/annotations`  the mutable half. Small, uncached, and fetched
                        separately so that someone filing a verdict does not
                        invalidate a megabyte of catalog data that did not
                        change.

Before the split, every request re-read `catalog_json` for every row and
orjson-parsed ~300 columns to read about thirty of them. Reading those thirty
out of the columnar catalog instead removes the inflation entirely.
"""
from __future__ import annotations

import gzip
import math
import sqlite3
import threading

import orjson
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from api.catalog_filters import BAD_FILTER, combine, get_filters, legacy_filter
from api.db import get_annotations, get_science
from api.ztforce_store import get_ztforce_store
from api.derive import CLASS_SOURCES, MATCHED_COLUMN, classes_of
from api.annotation_filters import annotation_filter
from api.routers.sources import _parse_filter, _sort_column
from source_db.serving.filters import UUID, CatalogFilters, FilterError, restrict_to
from source_db.constants import PSTARR_UPPER_LIM

router = APIRouter()


# ── the table view ────────────────────────────────────────────────────────────
#
# One compact row per source. The catalog row is 314 columns and the table
# shows nine fields, two of them derived, so the reduction happens here rather
# than shipping every column to the browser. Filtering, sorting and search
# stay client-side within the window that was fetched, which is what keeps a
# slider drag instant; narrowing *which* window you get is what the `filter`
# parameter is for.

# Copied from `Source_Analysis/store_src_plots`, which cannot be imported here:
# it pulls in matplotlib with usetex at module scope.

# (crossmatch prefix, display prefix, class column, column holds a list).
# Mirrors `classify_app/lib/classes.CLASS_SOURCES`; the crossmatch prefix is
# what `{prefix}_matched` is built from.
# The crossmatch flags the require/exclude filters offer, in display order.
CATALOG_PREFIXES = ['sdss', 'desi', 'tns', 'wise', 'agn_db']



def _number(value) -> float | None:
    """A finite float, or None for missing/NaN/non-numeric cells."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean_abs_dmag(cat: dict) -> float | None:
    """Mean |ZTF - PanSTARRS| across g/r/i: the manifest's triage measure.

    Ports `store_src_plots.mean_abs_dmag`, substitutions included -- a
    ZTF-only detection (Catalog_Flag 1) is differenced against the PanSTARRS
    upper limit, a PanSTARRS-only one (flag 2) against ZTF's own limiting
    magnitude. Recomputed rather than read: the science database carries the
    inputs but not the `dmag_order.csv` sidecar the manifest merged in.
    """
    dmags = []
    for band in ('g', 'r', 'i'):
        per_band = f'{band}_Catalog_Flag'
        flag = _number(cat.get(per_band if per_band in cat else 'Catalog_Flag'))
        ps = (PSTARR_UPPER_LIM[band] if flag == 1
              else _number(cat.get(f'PSTARR_{band}PSFMag')))
        ztf = (_number(cat.get(f'ZTF_{band}_mag_limit')) if flag == 2
               else _number(cat.get(f'ZTF_{band}PSFMag')))
        if ps is not None and ztf is not None:
            dmags.append(abs(ztf - ps))
    return sum(dmags) / len(dmags) if dmags else None


# The thirty-odd columns the table actually reads, out of 314. Named
# explicitly so the parquet scan touches only these -- that is the whole
# reason the endpoint no longer needs to inflate a JSON blob per row.
#
# `Catalog_Flag` without a band is `_mean_abs_dmag`'s fallback for catalogs
# written before the per-band flags existed. It is not in the current release,
# so the list is resolved against the live schema rather than asserted: a
# column that is not there should mean "that fallback does not apply", not a
# 422 on every request.
WANTED_COLUMNS = [
    # `wise_w1w2_mean`, not `wise_w1w2`: the column is headed W1-W2 and sat
    # beside a histogram showing the light-curve colour while reporting the
    # AllWISE coadd one, which differ by a median 0.10 mag. One quantity
    # across the table, the histogram and the filter.
    # `mean_abs_dmag` is a derived column on the catalog view (see
    # `api.catalog_filters`), which is also what the table sorts and filters
    # by. Reading it rather than recomputing it here removes a second
    # implementation of the same formula -- and with it the twelve input
    # columns that only existed to feed it.
    'coord_slug', 'ra', 'dec', 'catalog_tier', 'wise_w1w2_mean', 'mean_abs_dmag',
    *(c for _, c, _ in CLASS_SOURCES),
    *MATCHED_COLUMN.values(),
    *(f'{p}_matched' for p in CATALOG_PREFIXES),
    *(f'{b}_Catalog_Flag' for b in 'gri'),
    *(f'PSTARR_{b}PSFMag' for b in 'gri'),
    *(f'ZTF_{b}PSFMag' for b in 'gri'),
    *(f'ZTF_{b}_mag_limit' for b in 'gri'),
    'Catalog_Flag',
]

_COLUMNS: dict[tuple[str, float], list[str]] = {}


def _table_columns(cat: CatalogFilters) -> list[str]:
    """`WANTED_COLUMNS`, deduplicated and narrowed to what this release has."""
    global _COLUMNS
    stamp = _catalog_stamp(cat)
    cached = _COLUMNS.get(stamp)
    if cached is None:
        have = {c.name for c in cat.columns()}
        cached = [c for c in dict.fromkeys(WANTED_COLUMNS) if c in have]
        # Rebind rather than `clear()` + assign: one thread clearing while
        # another was between `in` and `[...]` raised KeyError. Rebinding is
        # atomic, and keeping only the current stamp still bounds the dict.
        _COLUMNS = {stamp: cached}
    return cached

# Derived rows are static for a release, so they are built once and kept as
# the bytes that go on the wire. Keyed by everything that changes the answer,
# including the catalog's mtime so a rebuild cannot serve stale rows.
_CACHE: dict[tuple, bytes] = {}
_CACHE_LOCK = threading.Lock()
# A window is ~0.4 MB gzipped at 5k rows; a handful of distinct windows is all
# the page ever asks for, and an unbounded dict here would be a slow leak.
_CACHE_MAX = 16


def _derive(row: dict) -> dict:
    """One table row from one catalog row.

    Every lookup is a `.get`, because the selected columns are whatever this
    release actually has -- a catalog written without `catalog_tier` should
    render a blank cell, not 500 the page.
    """
    return {
        'source_uuid': row['source_uuid'], 'coord_slug': row.get('coord_slug'),
        'ra': row.get('ra'), 'dec': row.get('dec'),
        'catalog_tier': row.get('catalog_tier'),
        'mean_abs_dmag': _number(row.get('mean_abs_dmag')),
        'wise_w1w2': _number(row.get('wise_w1w2_mean')),
        'matched': {p: bool(row.get(f'{p}_matched')) for p in CATALOG_PREFIXES},
        'classes': classes_of(row),
    }


@router.get('/table')
def table_rows(request: Request,
               offset: int = Query(0, ge=0),
               limit: int = Query(5000, ge=1, le=50000),
               sort: str = Query('uuid'),
               desc: bool = False,
               catalog: str | None = None,
               min_epochs: int | None = None,
               has_spectrum: bool = False,
               q: str | None = None,
               filter: str | None = None,
               label_state: str | None = None,
               author: str | None = None,
               bucket_in: list[str] | None = Query(None),
               bucket_out: list[str] | None = Query(None),
               status_in: list[str] | None = Query(None),
               status_out: list[str] | None = Query(None),
               cat: CatalogFilters = Depends(get_filters),
               ann: sqlite3.Connection = Depends(get_annotations)):
    """Every source in the window as one compact row, without verdicts.

    Verdicts live at `/table/annotations`: they are the only part that
    changes, and keeping them out is what makes this response cacheable at
    all. The frontend joins the two on `source_uuid`.

    The default window is 5,000 rows of 197,933. That is a real limit, not a
    performance accident -- the full catalog is ~12 MB gzipped and ~200 MB of
    objects once parsed, which is more than a browser should hold to draw a
    table. Narrow it with `filter` rather than raising `limit`.
    """
    doc = _table_filter(filter, survey=None, min_epochs=min_epochs,
                        has_spectrum=has_spectrum, catalog=catalog, q=q)
    doc = combine(doc, _label_filter(ann, label_state, author),
                  annotation_filter(ann, bucket_in=bucket_in,
                                    bucket_out=bucket_out,
                                    status_in=status_in, status_out=status_out))
    try:
        key = (_catalog_stamp(cat), offset, limit, sort, desc,
               orjson.dumps(doc, option=orjson.OPT_SORT_KEYS) if doc else b'')
    except (TypeError, ValueError) as e:
        # `orjson.dumps` refuses past 254 levels of nesting.
        raise HTTPException(422, f'filter is too deeply nested: {e}') from e

    blob = _CACHE.get(key)
    if blob is None:
        try:
            rows = cat.select(doc, _table_columns(cat), sort=_sort_column(sort),
                              desc=desc, limit=limit, offset=offset)
            total = cat.count(doc)
        except BAD_FILTER as e:
            raise HTTPException(422, str(e)) from e
        blob = gzip.compress(orjson.dumps({
            'total': total, 'offset': offset, 'limit': limit,
            'rows': [_derive(r) for r in rows],
        }), 6)
        with _CACHE_LOCK:
            if len(_CACHE) >= _CACHE_MAX:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = blob

    if 'gzip' in request.headers.get('accept-encoding', ''):
        return Response(content=blob, media_type='application/json',
                        headers={'Content-Encoding': 'gzip'})
    return Response(content=gzip.decompress(blob), media_type='application/json')


@router.get('/table/annotations')
def table_annotations(author: str | None = None,
                      ann: sqlite3.Connection = Depends(get_annotations),
                      zt: sqlite3.Connection = Depends(get_ztforce_store)):
    """Every verdict, by source.

    Returned whole rather than per-window: the table filters on label state
    across the whole sample, and this is one author's worth of rows per
    author -- far smaller than the catalog half it used to be embedded in.
    """
    out: dict[str, list[dict]] = {}
    sql = ('SELECT source_uuid, author, label, note, status FROM annotations'
           + (' WHERE author=?' if author else '')
           + ' ORDER BY author')
    for a in ann.execute(sql, (author,) if author else ()):
        out.setdefault(a['source_uuid'], []).append(
            {'author': a['author'], 'label': a['label'], 'note': a['note'],
             'status': a['status']})
    # Live like the verdicts, so it rides here rather than in the cached half.
    # "Ran" means a stored result exists, even one now queued to be redone.
    done = [r[0] for r in zt.execute(
        'SELECT source_uuid FROM runs WHERE stacks IS NOT NULL')]
    return {'annotations': out, 'ztforce_done': done}


def _catalog_stamp(cat: CatalogFilters) -> tuple[str, float]:
    """Changes when the catalog does, so a rebuild cannot serve stale rows.

    The path is part of it, not just the mtime: two catalogs written in the
    same second -- which is every pair of fixtures in the test suite -- would
    otherwise share a cache key and serve each other's rows.
    """
    try:
        return (str(cat.parquet), cat.parquet.stat().st_mtime)
    except OSError:
        return (str(cat.parquet), 0.0)


# The table's label cuts, answered over the whole catalog rather than the
# 5,000-row window: cut client-side, "Labeled by me" across all catalogs
# showed 47 of 111. Each cut is (keep or drop the matching sources, SQL that
# picks them). A source is filed when its label is non-blank.
_LABEL_CUTS = {
    'Unlabeled':       (False, ''),
    # The complement of 'Unlabeled': anyone's verdict counts, which is also
    # how `bucket` already behaves.
    'Labeled':         (True, ''),
    'Labeled by me':   (True, 'AND author = :author'),
    'Unlabeled by me': (False, 'AND author = :author'),
    'Disagreements':   (True, 'GROUP BY source_uuid '
                              'HAVING COUNT(DISTINCT trim(label)) > 1'),
}



def _label_filter(ann: sqlite3.Connection, state: str | None,
                  author: str | None) -> dict | None:
    """The label-state cut. Bucket and follow-up live in
    `api.annotation_filters`, which both routers share."""
    docs = []
    if state and state != 'All':
        if state not in _LABEL_CUTS:
            raise HTTPException(422, f'label_state must be All or one of {list(_LABEL_CUTS)}')
        # The per-author cuts bind `author` into the SQL; without one it binds
        # NULL, matches nothing, and the page silently shows zero rows. The
        # `/sources` equivalent already 400s, so this makes the two agree.
        if 'by me' in state and not author:
            raise HTTPException(400, f'label_state={state!r} requires author')
        keep, where = _LABEL_CUTS[state]
        uuids = [r[0] for r in ann.execute(
            "SELECT DISTINCT source_uuid FROM annotations "
            f"WHERE label IS NOT NULL AND trim(label) != '' {where}",
            {'author': author})]
        if keep:
            # `in` refuses an empty list; an empty OR is "match nothing".
            docs.append(restrict_to(uuids) if uuids else {'op': 'or', 'clauses': []})
        elif uuids:
            docs.append({'column': UUID, 'op': 'not_in', 'value': uuids})

    return combine(*docs)



def _table_filter(raw, **legacy):
    """As `/sources`, including the length cap.

    This used to `orjson.loads` whatever arrived, with no cap: `/sources`
    answered 413 for an oversized document while `/table` answered 500,
    because a deeply nested one reaches `orjson.dumps` when the cache key is
    built and that refuses past 254 levels.
    """
    doc = _parse_filter(raw)
    try:
        base = legacy_filter(**legacy)
    except BAD_FILTER as e:
        raise HTTPException(422, str(e)) from e
    if base and doc:
        return {'op': 'and', 'clauses': [base, doc]}
    return base or doc
