"""Ordering the table by what the annotators filed.

Bucket, follow-up status, comments and the ztforce flag are not catalog
columns: they live in `annotations.sqlite` and the ztforce store, and used to
sort only within the page that had been fetched -- a ranking of 5,000 rows
presented as a ranking of 200,000. The engine can order by a per-uuid values
table (`CatalogFilters.register_values`), so each of these is rendered to one
value per annotated source and registered under a fixed name, rebuilt on every
request because they change under the running server. Sources with no value
sort last whichever way the column points, as the engine does for NULLs.
"""
from __future__ import annotations

import sqlite3

from api.followup import RANK
from source_db.serving.filters import CatalogFilters

PREFIX = 'ann:'
COLUMNS = ('bucket', 'status', 'comments', 'ztforce')


def is_annotation_sort(sort: str) -> bool:
    return sort.startswith(PREFIX)


def stamp(ann: sqlite3.Connection, zt: sqlite3.Connection) -> str:
    """Changes whenever any of the values below could; for cache keys."""
    parts = [
        ann.execute('SELECT COUNT(*), MAX(updated_at) FROM annotations').fetchone(),
        ann.execute('SELECT COUNT(*), MAX(updated_at) FROM followup').fetchone(),
        zt.execute('SELECT COUNT(*), MAX(finished_at) FROM runs WHERE stacks IS NOT NULL'
                   ).fetchone(),
    ]
    return '|'.join(f'{n}:{t or ""}' for n, t in parts)


def values(column: str, ann: sqlite3.Connection, zt: sqlite3.Connection) -> dict:
    """One sortable value per source that has one, matching what the table shows."""
    if column == 'bucket':
        # The cell: every bucket anyone filed it under, sorted, joined.
        by_uuid: dict[str, set[str]] = {}
        for r in ann.execute("SELECT source_uuid, label FROM annotations "
                             "WHERE label IS NOT NULL AND trim(label) != ''"):
            by_uuid.setdefault(r[0], set()).add(r[1].strip())
        return {u: ' / '.join(sorted(s)) for u, s in by_uuid.items()}
    if column == 'status':
        # Progress order, not alphabetical: to queue < queued < observed.
        return {r[0]: RANK[r[1]] for r in ann.execute('SELECT source_uuid, status FROM followup')
                if r[1] in RANK}
    if column == 'comments':
        # How many people have said something: the cell is their notes.
        return {r[0]: r[1] for r in ann.execute(
            "SELECT source_uuid, COUNT(*) FROM annotations "
            "WHERE note IS NOT NULL AND trim(note) != '' GROUP BY source_uuid")}
    if column == 'ztforce':
        return {r[0]: 1.0 for r in zt.execute(
            'SELECT source_uuid FROM runs WHERE stacks IS NOT NULL')}
    raise ValueError(column)


def resolve(sort: str, ann: sqlite3.Connection, zt: sqlite3.Connection,
            cat: CatalogFilters) -> str:
    """The engine's sort key for an `ann:<column>` sort, with the values fresh."""
    column = sort[len(PREFIX):]
    if column not in COLUMNS:
        raise ValueError(f'sort must be one of {[PREFIX + c for c in COLUMNS]}')
    return cat.register_values(f'ann_{column}', values(column, ann, zt), replace=True)
