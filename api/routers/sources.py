"""Candidate list, detail, and the click-through queue.

The three endpoints the classification loop actually uses are `/sources`
(the list), `/sources/{uuid}/lightcurve` (the chart data), and
"""
from __future__ import annotations

import gzip
import json

import orjson
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from api import coord_lists
from api.catalog_filters import (BAD_FILTER, CATALOG_OF_TIER, CATALOG_TIERS,
                                 combine, get_filters, legacy_filter,
                                 one_per_object as _one_per_object)
from api.db import get_annotations, get_coord_lists, get_science
from api.annotation_filters import annotation_filter
from api.derive import class_redshifts_of, classes_of, redshifts_of
from source_db.serving.filters import UUID, CatalogFilters, FilterError, restrict_to
from api import catalog_schema
from api.config import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from api.models import SourceDetail, SourcePage, SourceSummary
from api.routers.annotations import _row_to_annotation
from api.routers.coord_lists import coord_list_filter

router = APIRouter()

# Short names the frontend has always sent. Any real column name is also
# accepted now -- the filter engine validates it against the catalog schema
# and refuses one that is NULL for every source -- so these are aliases for
# the common four rather than the whole permitted set.
SORT_ALIASES = {
    'uuid': 'source_uuid',
    'ra': 'ra',
    'dec': 'dec',
    'epochs': 'n_epochs',
}


def _sort_column(sort: str) -> str:
    return SORT_ALIASES.get(sort, sort)


# `sort=list:<column>` orders by a value column of the applied coordinate list.
LIST_SORT_PREFIX = 'list:'


def _resolve_sort(sort: str, cat: CatalogFilters, store: sqlite3.Connection,
                  coord_list: str | None) -> str:
    """What the engine orders by: an alias, a catalog column, or a list value.

    A list's values are registered with the engine the first time a request
    sorts by them, under a name made of the list id and the column, so every
    later request -- and every endpoint that has to agree on the order --
    reuses the same table.
    """
    if not sort.startswith(LIST_SORT_PREFIX):
        return _sort_column(sort)
    column = sort[len(LIST_SORT_PREFIX):]
    if not coord_list:
        raise HTTPException(422, f'sorting by {sort!r} needs a coordinate list (coord_list=)')
    found = coord_lists.values(store, coord_list)
    if found is None:
        raise HTTPException(404, f'no coordinate list {coord_list}; it may have been deleted')
    if column not in found['columns']:
        raise HTTPException(422, f'coordinate list has no column {column!r}; '
                                 f'it has {found["columns"] or "none"}')
    i = found['columns'].index(column)
    return cat.register_values(f'coordlist_{coord_list}_{column}',
                               {u: row[i] for u, row in found['rows'].items()})


def _parse_filter(raw: str | None):
    """The `filter` query parameter: a filter document, JSON-encoded.

    Sent in the URL rather than a request body so that a filtered view stays a
    link -- the thing you paste to a collaborator, and the thing the classify
    page's deep links already rely on. Capped because a URL is not a place to
    put an unbounded document; a filter that needs more than this wants to be
    saved and referenced by id, which is the next thing to build.
    """
    if not raw:
        return None
    if len(raw) > MAX_FILTER_CHARS:
        raise HTTPException(413, f'filter document over {MAX_FILTER_CHARS} characters; '
                                 f'save it and reference it by id instead')
    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError as e:
        raise HTTPException(422, f'filter is not valid JSON: {e}') from e


def _labelled_uuids(ann, author: str) -> list[str]:
    return [r['source_uuid'] for r in ann.execute(
        'SELECT source_uuid FROM annotations '
        'WHERE author=? AND label IS NOT NULL', (author,))]


def _build(doc, *, author=None, unlabelled_only=False, ann=None,
           bucket_in=None, bucket_out=None, status_in=None, status_out=None,
           one_per_object=False, **legacy):
    """Everything the caller asked for, as one filter document.

    The legacy query parameters, the user's own filter, and "not yet labelled
    by me" all end up in the same language, which is what lets them compose
    and be paginated once rather than filtered in two places that can
    disagree about what the total is.
    """
    exclude = _labelled_uuids(ann, author) if (unlabelled_only and author) else None
    verdicts = annotation_filter(ann, bucket_in=bucket_in, bucket_out=bucket_out,
                                 status_in=status_in, status_out=status_out)
    try:
        # One row per object is picked on the catalog side only, so labels
        # then apply to that row: labelling it takes the object out of an
        # "unbucketed" queue rather than promoting its group-mate.
        scope = combine(legacy_filter(**legacy), doc)
        if one_per_object:
            scope = _one_per_object(scope)
        return combine(scope, legacy_filter(exclude_uuids=exclude), verdicts)
    except BAD_FILTER as e:
        raise HTTPException(422, str(e)) from e


MAX_FILTER_CHARS = 8000


def _split_surveys(value: str | None) -> list[str]:
    return [s for s in (value or '').split(',') if s]


# One definition, in `api.catalog_filters`, so the list and the table cannot
# come to disagree about what `in_ztf` means.
_CATALOG_OF_TIER = CATALOG_OF_TIER


def _summary(row, label=None, n_ann=0) -> SourceSummary:
    keys = row.keys()
    return SourceSummary(
        source_uuid=row['source_uuid'], coord_slug=row['coord_slug'],
        ra=row['ra'], dec=row['dec'], n_epochs=row['n_epochs'] or 0,
        surveys=_split_surveys(row['surveys']),
        n_spectra=(row['n_spectra'] or 0) if 'n_spectra' in keys else 0,
        spec_surveys=_split_surveys(row['spec_surveys'] if 'spec_surveys' in keys else None),
        label=label, n_annotations=n_ann,
        catalog=(_CATALOG_OF_TIER.get(row['catalog_tier'])
                 if 'catalog_tier' in keys else None))


@router.get('/catalog/schema')
def get_catalog_schema(sci: sqlite3.Connection = Depends(get_science)):
    """How to group the ~300 catalog columns on the detail page.

    Static across rebuilds, so the frontend fetches it once and caches it,
    rather than every source response repeating the group names.
    """
    row = sci.execute('SELECT catalog_json FROM sources LIMIT 1').fetchone()
    if row is None:
        return {'groups': [], 'n_columns': 0}
    return catalog_schema.build_schema(list(orjson.loads(row['catalog_json'])))


@router.get('/sources', response_model=SourcePage)
def list_sources(
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    sort: str = Query('uuid', description="'uuid', 'ra', 'dec', 'epochs', "
                                          'or any catalog column'),
    desc: bool = False,
    survey: str | None = None,
    catalog: str | None = Query(
        None, description="sub-catalogue: 'in_both', 'in_ztf' or 'in_ztf_wide'; omit for all"),
    min_epochs: int | None = None,
    has_lightcurve: bool = False,
    has_spectrum: bool = False,
    author: str | None = Query(None, description="annotate rows with this author's label"),
    unlabelled_only: bool = Query(False, description='requires `author`'),
    q: str | None = Query(None, description='substring of coord_slug, or a uuid prefix'),
    filter: str | None = Query(None, description='a filter document, JSON-encoded'),
    one_per_object: bool = Query(False, description='one detection per duplicate group'),
    bucket_in: list[str] | None = Query(None),
    bucket_out: list[str] | None = Query(None),
    status_in: list[str] | None = Query(None),
    status_out: list[str] | None = Query(None),
    sci: sqlite3.Connection = Depends(get_science),
    ann: sqlite3.Connection = Depends(get_annotations),
    coord_list: str | None = Query(None, description='id from POST /coord-lists: only those sources'),
    store: sqlite3.Connection = Depends(get_coord_lists),
    cat: CatalogFilters = Depends(get_filters),
):
    """One page of the candidate list.

    Filtering and paging happen in the catalog engine, which can see all 318
    columns; SQLite is then asked only for the handful of rows that survived.
    The split matters because the page is a window into ~198k sources and the
    filter may name a column that only exists in the parquet.
    """
    if unlabelled_only and not author:
        raise HTTPException(400, 'unlabelled_only requires author')

    doc = _build(combine(_parse_filter(filter), coord_list_filter(store, coord_list)), author=author,
                 unlabelled_only=unlabelled_only, ann=ann,
                 bucket_in=bucket_in, bucket_out=bucket_out,
                 status_in=status_in, status_out=status_out,
                 survey=survey, min_epochs=min_epochs,
                 has_lightcurve=has_lightcurve, has_spectrum=has_spectrum,
                 catalog=catalog, q=q, one_per_object=one_per_object)
    try:
        window = cat.page(doc, sort=_resolve_sort(sort, cat, store, coord_list), desc=desc,
                          limit=limit, offset=offset)
    except BAD_FILTER as e:
        raise HTTPException(422, str(e)) from e

    uuids = window['uuids']
    rows_by_uuid = {}
    if uuids:
        qs = ','.join('?' * len(uuids))
        rows_by_uuid = {r['source_uuid']: r for r in sci.execute(
            f'SELECT source_uuid, coord_slug, ra, dec, n_epochs, surveys, '
            f'n_spectra, spec_surveys, catalog_tier '
            f'FROM sources WHERE source_uuid IN ({qs})',
            uuids)}

    labels, counts = {}, {}
    if uuids:
        qs = ','.join('?' * len(uuids))
        for r in ann.execute(
                f'SELECT source_uuid, author, label FROM annotations '
                f'WHERE source_uuid IN ({qs})', uuids):
            counts[r['source_uuid']] = counts.get(r['source_uuid'], 0) + 1
            if author and r['author'] == author:
                labels[r['source_uuid']] = r['label']

    # Progress ("n of total labelled") over the *filtered* set. One more count
    # in the same engine: how many of this author's labelled sources survive
    # the filter. `unlabelled_only` has excluded them all, so it is 0 there.
    n_labelled = 0
    if author and not unlabelled_only:
        mine = _labelled_uuids(ann, author)
        if mine:
            n_labelled = cat.count(combine(doc, restrict_to(mine)))

    # The engine decided the order; SQLite returned a set. Re-impose it.
    items = [_summary(rows_by_uuid[u], labels.get(u), counts.get(u, 0))
             for u in uuids if u in rows_by_uuid]
    return SourcePage(total=window['total'], offset=offset, limit=limit,
                      n_labelled=n_labelled, items=items)


@router.get('/sources/{source_uuid}/group')
def get_group(source_uuid: str, cat: CatalogFilters = Depends(get_filters)):
    """The other detections of the same object: rows within 1\" of each other
    from overlapping fields share a `group_id`. `members` includes this source
    and is empty when it has no group."""
    return {'members': cat.group_members(source_uuid)}


@router.get('/sources-locate')
def locate_source(
    target: str = Query(..., description='find this source_uuid in the filtered order'),
    sort: str = Query('uuid'),
    desc: bool = False,
    survey: str | None = None,
    catalog: str | None = None,
    min_epochs: int | None = None,
    has_lightcurve: bool = False,
    has_spectrum: bool = False,
    q: str | None = None,
    filter: str | None = None,
    one_per_object: bool = False,
    author: str | None = None,
    unlabelled_only: bool = False,
    bucket_in: list[str] | None = Query(None),
    bucket_out: list[str] | None = Query(None),
    status_in: list[str] | None = Query(None),
    status_out: list[str] | None = Query(None),
    ann: sqlite3.Connection = Depends(get_annotations),
    coord_list: str | None = Query(None, description='id from POST /coord-lists: only those sources'),
    store: sqlite3.Connection = Depends(get_coord_lists),
    cat: CatalogFilters = Depends(get_filters),
):
    """The offset of `target` within the current filter, or null if filtered out.

    The client holds a window of the list rather than all of it, so it cannot
    find a source's position by scanning what it has. This answers "where does
    this one sit" so a link to a specific source still lands on the right index.

    Counting the rows that sort before the target, rather than scanning for
    it, is what keeps this O(filter) instead of O(offset).
    """
    doc = _build(combine(_parse_filter(filter), coord_list_filter(store, coord_list)), author=author,
                 unlabelled_only=unlabelled_only, ann=ann,
                 bucket_in=bucket_in, bucket_out=bucket_out,
                 status_in=status_in, status_out=status_out,
                 survey=survey, min_epochs=min_epochs,
                 has_lightcurve=has_lightcurve, has_spectrum=has_spectrum,
                 catalog=catalog, q=q, one_per_object=one_per_object)
    try:
        offset = cat.locate(target, doc, sort=_resolve_sort(sort, cat, store, coord_list), desc=desc)
    except BAD_FILTER as e:
        raise HTTPException(422, str(e)) from e
    return {'offset': offset}


@router.get('/sources-next-unlabelled')
def next_unlabelled(
    author: str = Query(..., description='whose labels count'),
    after: int = Query(0, ge=0, description='first offset to consider'),
    sort: str = Query('uuid'),
    desc: bool = False,
    survey: str | None = None,
    catalog: str | None = None,
    min_epochs: int | None = None,
    has_lightcurve: bool = False,
    has_spectrum: bool = False,
    q: str | None = None,
    filter: str | None = None,
    one_per_object: bool = False,
    bucket_in: list[str] | None = Query(None),
    bucket_out: list[str] | None = Query(None),
    status_in: list[str] | None = Query(None),
    status_out: list[str] | None = Query(None),
    ann: sqlite3.Connection = Depends(get_annotations),
    coord_list: str | None = Query(None, description='id from POST /coord-lists: only those sources'),
    store: sqlite3.Connection = Depends(get_coord_lists),
    cat: CatalogFilters = Depends(get_filters),
):
    """Offset of the next source this author has not labelled, wrapping once.

    Same reason as `/sources-locate`: with only a window in hand the client
    cannot scan ahead for one. The labelled set is one author's work, so it
    goes in as a `not_in` clause and the engine does the rest -- the answer is
    the first row of the filter-plus-unlabelled order at or after `after`.
    """
    base = _build(combine(_parse_filter(filter), coord_list_filter(store, coord_list)),
                  survey=survey, min_epochs=min_epochs,
                  bucket_in=bucket_in, bucket_out=bucket_out,
                  status_in=status_in, status_out=status_out,
                  has_lightcurve=has_lightcurve, has_spectrum=has_spectrum,
                  catalog=catalog, q=q, ann=ann, one_per_object=one_per_object)
    mine = _labelled_uuids(ann, author)
    unlabelled = combine(base, {'column': UUID, 'op': 'not_in', 'value': mine}
                         if mine else None)
    key = _resolve_sort(sort, cat, store, coord_list)
    try:
        # Positions are offsets into the *filtered* order, which still counts
        # the labelled ones -- that is the order the client is paging through.
        # So find the next unlabelled source in its own order, then ask where
        # it sits in the client's.
        for start in (after, 0):        # from here, then wrap to the beginning
            page = cat.page(unlabelled, sort=key, desc=desc, limit=1,
                            offset=_rank(cat, base, unlabelled, start, key, desc))
            if page['uuids']:
                found = page['uuids'][0]
                offset = cat.locate(found, base, sort=key, desc=desc)
                if offset is not None and (start == 0 or offset >= after):
                    return {'offset': offset, 'source_uuid': found}
    except BAD_FILTER as e:
        raise HTTPException(422, str(e)) from e
    return {'offset': None, 'source_uuid': None}


def _rank(cat, base, unlabelled, offset_in_base, key, desc) -> int:
    """Translate an offset in the full order into one in the unlabelled order.

    The client counts positions including sources it has already filed; the
    search for "next unlabelled" has to happen in the order that excludes
    them. Anchoring on the source currently at `offset_in_base` converts
    between the two without materialising either list.
    """
    if offset_in_base <= 0:
        return 0
    anchor = cat.page(base, sort=key, desc=desc, limit=1, offset=offset_in_base)
    if not anchor['uuids']:
        return 0
    at = cat.locate(anchor['uuids'][0], unlabelled, sort=key, desc=desc)
    if at is not None:
        return at
    # The anchor is itself labelled, so it has no place in the unlabelled
    # order; count how many unlabelled sources sort before it instead.
    return cat.rank_before(anchor['uuids'][0], unlabelled, sort=key, desc=desc)


@router.get('/sources/{source_uuid}', response_model=SourceDetail)
def get_source(source_uuid: str,
               sci: sqlite3.Connection = Depends(get_science),
               ann: sqlite3.Connection = Depends(get_annotations)):
    """The full catalog row plus every author's annotation."""
    row = sci.execute(
        'SELECT source_uuid, coord_slug, ra, dec, n_epochs, surveys, '
        'n_spectra, spec_surveys, catalog_json '
        'FROM sources WHERE source_uuid=?', (source_uuid,)).fetchone()
    if row is None:
        raise HTTPException(404, f'no source {source_uuid}')
    anns = [_row_to_annotation(r) for r in ann.execute(
        'SELECT * FROM annotations WHERE source_uuid=? ORDER BY author',
        (source_uuid,))]
    cat = orjson.loads(row['catalog_json'])
    return SourceDetail(
        source_uuid=row['source_uuid'], coord_slug=row['coord_slug'],
        ra=row['ra'], dec=row['dec'], n_epochs=row['n_epochs'] or 0,
        surveys=_split_surveys(row['surveys']),
        n_spectra=row['n_spectra'] or 0,
        spec_surveys=_split_surveys(row['spec_surveys']),
        catalog=cat, classes=classes_of(cat),
        redshifts=redshifts_of(cat), class_redshifts=class_redshifts_of(cat),
        annotations=anns)


@router.get('/sources/{source_uuid}/lightcurve')
def get_lightcurve(source_uuid: str, request: Request,
                   sci: sqlite3.Connection = Depends(get_science)):
    """The chart payload.

    Stored gzipped in exactly the shape the frontend wants, so the normal path
    hands the bytes to the client untouched -- no decompress, no reserialise.
    Only a client that cannot accept gzip pays for a decompression.
    """
    row = sci.execute(
        'SELECT payload FROM light_curves WHERE source_uuid=?',
        (source_uuid,)).fetchone()
    if row is None:
        # A source with no epochs is normal, not an error.
        if sci.execute('SELECT 1 FROM sources WHERE source_uuid=?',
                       (source_uuid,)).fetchone() is None:
            raise HTTPException(404, f'no source {source_uuid}')
        return Response(
            content=json.dumps({'source_uuid': source_uuid, 'n_epochs': 0,
                                'surveys': {}}),
            media_type='application/json')

    blob = row['payload']
    if 'gzip' in request.headers.get('accept-encoding', ''):
        return Response(content=blob, media_type='application/json',
                        headers={'Content-Encoding': 'gzip',
                                 'Cache-Control': 'public, max-age=31536000, immutable'})
    return Response(content=gzip.decompress(blob), media_type='application/json')


@router.get('/sources/{source_uuid}/spectra')
def get_spectra(source_uuid: str, request: Request,
                sci: sqlite3.Connection = Depends(get_science)):
    """SDSS and DESI spectra for one source.

    Same gzip-passthrough contract as the light curves. Only ~34% of sources
    have a spectrum, so an empty result is the common case, not an error.
    """
    row = sci.execute('SELECT payload FROM spectra WHERE source_uuid=?',
                      (source_uuid,)).fetchone()
    if row is None:
        if sci.execute('SELECT 1 FROM sources WHERE source_uuid=?',
                       (source_uuid,)).fetchone() is None:
            raise HTTPException(404, f'no source {source_uuid}')
        return Response(
            content=json.dumps({'source_uuid': source_uuid, 'n_spectra': 0,
                                'spectra': []}),
            media_type='application/json')

    blob = row['payload']
    if 'gzip' in request.headers.get('accept-encoding', ''):
        return Response(content=blob, media_type='application/json',
                        headers={'Content-Encoding': 'gzip',
                                 'Cache-Control': 'public, max-age=31536000, immutable'})
    return Response(content=gzip.decompress(blob), media_type='application/json')






