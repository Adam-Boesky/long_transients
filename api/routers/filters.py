"""What the filter UI needs to offer a filter it can trust.

Three things, and each exists because the alternative is a UI that lies:

  `/filters/columns`  what you may filter on, and how full each column is
  `/filters/stats`    a column's real range, or its actual vocabulary
  `/filters/preview`  how many sources match, and which clause did the cutting

The fill rate is the one that matters most. SIMBAD matches 26% of sources,
DESI 25%, TNS 0.6%; a cut on any of them is a cut on that subset, and a UI
that does not say so lets you select 183 sources and believe you selected a
population.
"""
from __future__ import annotations

from typing import Any

import orjson
from fastapi import APIRouter, Body, Depends, HTTPException, Query

from api.catalog_filters import BAD_FILTER, combine, get_filters, legacy_filter
from api import catalog_schema
from source_db.serving.filters import (MISSING_POLICIES, OPERATORS,
                                       CatalogFilters, FilterError)

router = APIRouter()


@router.get('/filters/columns')
def list_columns(cat: CatalogFilters = Depends(get_filters)):
    """Every filterable column, grouped the way the detail page groups them.

    Columns that are NULL for every source are reported rather than hidden:
    they are a fact about this release -- `PSTARR_primaryDetection` and the
    ZTF `ccdid`/`qid`/`field` triple never made it through the build -- and
    silently dropping them would turn "why can't I filter on that" into a
    question with no answer on screen.
    """
    columns = cat.columns()
    return {
        'operators': OPERATORS,
        'missing_policies': list(MISSING_POLICIES),
        'groups': [
            {'key': key, 'label': catalog_schema.LABELS.get(key, key),
             'columns': [
                 {'name': c.name, 'type': c.type, 'n_non_null': c.n_non_null,
                  'fill': round(c.fill, 4), 'filterable': c.filterable}
                 for c in sorted(columns, key=lambda c: c.name)
                 if c.group == key]}
            for key in sorted({c.group for c in columns})
        ],
        'n_columns': len(columns),
    }


@router.get('/filters/stats/{column}')
def column_stats(column: str, top: int = Query(20, ge=1, le=200),
                 cat: CatalogFilters = Depends(get_filters)):
    """A column's range and quantiles, or its most common values.

    Fetched when a column is picked rather than up front: 318 columns of
    quantiles is a lot of work for a panel that shows one at a time.
    """
    try:
        return cat.stats(column, top=top)
    except BAD_FILTER as e:
        raise HTTPException(404, str(e)) from e


@router.post('/filters/preview')
def preview(doc: Any = Body(None, description='a filter document'),
            survey: str | None = None,
            catalog: str | None = None,
            min_epochs: int | None = None,
            has_lightcurve: bool = False,
            has_spectrum: bool = False,
            q: str | None = None,
            cat: CatalogFilters = Depends(get_filters)):
    """How many sources match, and how many each top-level clause keeps alone.

    Counted *within the selection the page is already showing*, not across the
    whole catalog. The sidebar's other filters are passed in and ANDed into
    every count here, because a panel reading "111,121 sources match" next to
    a header reading "1,270" is worse than no count at all -- both numbers are
    true, and the reader has no way to know that.

    The per-clause counts answer the question a filter returning three sources
    always raises -- *which cut did that* -- and they are affordable only
    because each is a few milliseconds.
    """
    try:
        scope = legacy_filter(survey=survey, min_epochs=min_epochs,
                              has_lightcurve=has_lightcurve,
                              has_spectrum=has_spectrum, catalog=catalog, q=q)
        clauses = doc.get('clauses') if isinstance(doc, dict) else None
        if not clauses:
            clauses = [doc] if doc else []
        return {
            'total': cat.count(combine(scope, doc)),
            'scope_total': cat.count(scope),
            'clauses': [_clause_count(cat, scope, c) for c in clauses],
            'sql': cat.compile(combine(scope, doc))[0],
        }
    except BAD_FILTER as e:
        raise HTTPException(422, str(e)) from e


def _clause_count(cat: CatalogFilters, scope: Any, clause: Any) -> dict:
    try:
        return {'clause': clause, 'matches': cat.count(combine(scope, clause))}
    except BAD_FILTER as e:
        return {'clause': clause, 'error': str(e)}




