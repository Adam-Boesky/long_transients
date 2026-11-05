"""Filter documents derived from `annotations.sqlite`.

The catalog engine filters parquet, but verdicts live in SQLite. These resolve
a verdict question to a uuid list the engine can restrict on, so the table and
the candidate list ask it the same way rather than in two places that can
disagree about the total.

Every cut is any-author: one person filing a source is enough.
"""
from __future__ import annotations

import sqlite3

from fastapi import HTTPException

from api.catalog_filters import combine
from api.models import FOLLOWUP_STATUSES
from source_db.serving.filters import UUID, restrict_to

# The pseudo-value both checklists offer: "nobody has set one". It is the
# absence of a row, so it cannot be looked up and is answered as a complement.
NONE = 'none'

# `in` refuses an empty list; an empty OR is "match nothing".
_MATCH_NOTHING: dict = {'op': 'or', 'clauses': []}


def _clean(names: list[str] | None) -> list[str]:
    return sorted({n.strip() for n in (names or []) if n.strip()})


def _uuids(ann: sqlite3.Connection, column: str, names: list[str]) -> list[str]:
    """Sources any author gave one of `names` in `column`."""
    if not names:
        return []
    qs = ','.join('?' * len(names))
    # NOCASE because `labels.name` is declared COLLATE NOCASE precisely so
    # "trash" cannot sit beside "Trash" -- a verdict written under a differently
    # cased spelling would otherwise be invisible to these filters while showing
    # up everywhere else.
    return [r[0] for r in ann.execute(
        'SELECT DISTINCT source_uuid FROM annotations '
        f'WHERE {column} IS NOT NULL AND trim({column}) COLLATE NOCASE IN ({qs})',
        names)]


def _uuids_any(ann: sqlite3.Connection, column: str) -> list[str]:
    """Sources any author gave any value at all in `column`."""
    return [r[0] for r in ann.execute(
        'SELECT DISTINCT source_uuid FROM annotations '
        f"WHERE {column} IS NOT NULL AND trim({column}) != ''")]


def _include(ann, column: str, names: list[str]) -> dict | None:
    """Sources matching any of `names`; `NONE` means "has no value here"."""
    real = [n for n in names if n != NONE]
    clauses: list[dict] = []
    if real:
        hits = _uuids(ann, column, real)
        clauses.append(restrict_to(hits) if hits else _MATCH_NOTHING)
    if NONE in names:
        taken = _uuids_any(ann, column)
        if not taken:
            return None        # nothing is set, so "none" is everything
        clauses.append({'column': UUID, 'op': 'not_in', 'value': taken})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {'op': 'or', 'clauses': clauses}


def _exclude(ann, column: str, names: list[str]) -> dict | None:
    """Drop sources matching any of `names`."""
    real = [n for n in names if n != NONE]
    docs: list[dict] = []
    if real:
        hits = _uuids(ann, column, real)
        if hits:
            docs.append({'column': UUID, 'op': 'not_in', 'value': hits})
    if NONE in names:
        # Dropping "has no value" is keeping the ones that have one.
        taken = _uuids_any(ann, column)
        docs.append(restrict_to(taken) if taken else _MATCH_NOTHING)
    return combine(*docs)


def annotation_filter(ann: sqlite3.Connection, *,
                      bucket_in: list[str] | None = None,
                      bucket_out: list[str] | None = None,
                      status_in: list[str] | None = None,
                      status_out: list[str] | None = None) -> dict | None:
    """The bucket and follow-up include/exclude lists, as one document.

    Within a list the values are an OR; the four lists combine with AND, so
    excluding beats including for a source carrying both.
    """
    si, so = _clean(status_in), _clean(status_out)
    known = {*FOLLOWUP_STATUSES, NONE}
    for value in (*si, *so):
        if value not in known:
            raise HTTPException(422, f'status must be one of {sorted(known)}')
    return combine(
        _include(ann, 'label', _clean(bucket_in)),
        _exclude(ann, 'label', _clean(bucket_out)),
        _include(ann, 'status', si),
        _exclude(ann, 'status', so),
    )
