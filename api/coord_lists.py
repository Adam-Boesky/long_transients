"""Uploaded coordinate lists: positions from a .txt, matched to sources and saved under a name.

A list of a few hundred positions does not fit in a URL, so it is matched here
and the matching source uuids are stored. The list endpoints then take
`coord_list=<id>`, which keeps a filtered view a link like every other cut.

Lists are shared: anyone can see, apply, rename or delete any of them, and no
owner is recorded. Deleting only hides a list (`deleted_at`), so a mistaken
delete can be undone in the database. Stored in its own SQLite file, never in
annotations.sqlite.

A file may carry numbers beside each position -- a score, a statistic from a
notebook -- declared by a header line such as `name ra dec stetson_J`. They
are kept per matched source and the list endpoints can order by them
(`sort=list:stetson_J`), so a ranking made elsewhere can be walked here.
"""
from __future__ import annotations

import re
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import orjson

from api.catalog_filters import CATALOG_OF_TIER, _parse_position
from source_db.serving.filters import UUID, restrict_to

DEFAULT_RADIUS_ARCSEC = 2.0
MAX_RADIUS_ARCSEC = 60.0
MAX_COORDS = 20000
MAX_TEXT_CHARS = 2_000_000
MAX_NAME_CHARS = 120
# How many unparsed lines and unmatched positions a summary lists; the counts are always complete.
MAX_LISTED = 50
# Value columns a header may declare. Names are identifiers because they become sort keys.
MAX_VALUE_COLUMNS = 20
MAX_COLUMN_CHARS = 40
_IDENTIFIER = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

SCHEMA = """
CREATE TABLE IF NOT EXISTS coord_lists (
    id TEXT PRIMARY KEY,
    name TEXT,
    created TEXT,
    summary_json TEXT NOT NULL,
    uuids_json TEXT NOT NULL
)
"""
# Added after the first version; `connect` adds any that an older file lacks.
COLUMNS = {'radius_arcsec': 'REAL', 'positions_json': 'TEXT', 'deleted_at': 'TEXT',
           'values_json': 'TEXT'}


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, check_same_thread=False, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute(SCHEMA)
    have = {r[1] for r in con.execute('PRAGMA table_info(coord_lists)')}
    for col, kind in COLUMNS.items():
        if col not in have:
            con.execute(f'ALTER TABLE coord_lists ADD COLUMN {col} {kind}')
    con.commit()
    return con


def _header(line: str) -> list[str] | None:
    """Column names from a header line, or None if the line is not one.

    A header is `ra dec` or `name ra dec`, with or without a leading `#`,
    followed by the names of the value columns: `name ra dec stetson_J rho`.
    """
    tokens = line.lstrip('#').replace(',', ' ').replace('\t', ' ').split()
    low = [t.lower() for t in tokens]
    if low[:3] == ['name', 'ra', 'dec']:
        extra = tokens[3:]
    elif low[:2] == ['ra', 'dec']:
        extra = tokens[2:]
    else:
        return None
    if len(extra) > MAX_VALUE_COLUMNS:
        return None
    if not all(_IDENTIFIER.match(t) and len(t) <= MAX_COLUMN_CHARS for t in extra):
        return None
    return list(dict.fromkeys(extra))


def _position_and_rest(tokens: list[str], n_values: int):
    """(ra, dec, remaining tokens) from a line's tokens, or None.

    The position is two tokens (decimal or `hh:mm:ss dd:mm:ss`) or six (`hh mm
    ss dd mm ss`), after an optional name. With value columns declared, the
    reading that leaves exactly that many tokens is preferred, so `10 05 3 ...`
    is not mistaken for a sexagesimal hour when it is a position plus a score.
    """
    found = []
    for start in (0, 1):
        for width in (6, 2):
            part = tokens[start:start + width]
            if len(part) < width:
                continue
            pos = _parse_position(' '.join(part))
            if pos is not None:
                found.append((pos[0], pos[1], tokens[start + width:]))
    for ra, dec, rest in found:
        if len(rest) == n_values:
            return ra, dec, rest
    return found[0] if found else None


def _value(token: str) -> float | None:
    try:
        v = float(token)
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def parse(text: str) -> tuple[list[tuple[float, float]], list[str], list[str], list[list[float | None]]]:
    """Positions, one per line, as 'ra dec' in decimal degrees or sexagesimal.

    Commas and tabs separate as well as spaces, a leading name is skipped, and
    blank lines and '#' comments are ignored. A header line before the first
    position (see `_header`) declares value columns, read from the tokens
    after each position; a missing or non-numeric value is None. Returns the
    positions, the lines that were not one, the column names, and one list of
    values per position.
    """
    coords, bad, values = [], [], []
    columns: list[str] | None = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if columns is None and not coords:
            header = _header(stripped)
            if header is not None:
                columns = header
                continue
        line = stripped.split('#', 1)[0].strip()
        if not line:
            continue
        tokens = line.replace(',', ' ').replace('\t', ' ').split()
        n_values = len(columns or [])
        found = _position_and_rest(tokens, n_values)
        if found is None:
            bad.append(line)
            continue
        ra, dec, rest = found
        coords.append((float(ra), float(dec)))
        vals = [_value(t) for t in rest[:n_values]]
        values.append(vals + [None] * (n_values - len(vals)))
    return coords, bad, columns or [], values


class _Sky:
    """Every source's position, sorted by dec, cached until the science DB changes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stamp = None

    def get(self, sci: sqlite3.Connection, stamp) -> '_Sky':
        with self._lock:
            if stamp != self._stamp:
                rows = sci.execute('SELECT source_uuid, ra, dec, catalog_tier FROM sources '
                                   'WHERE ra IS NOT NULL AND dec IS NOT NULL').fetchall()
                dec = np.array([r['dec'] for r in rows], float)
                order = np.argsort(dec)
                self.dec = dec[order]
                self.ra = np.array([r['ra'] for r in rows], float)[order]
                self.uuid = np.array([r['source_uuid'] for r in rows], object)[order]
                self.tier = np.array([r['catalog_tier'] for r in rows], object)[order]
                self._stamp = stamp
            return self


_SKY = _Sky()


def match(sky: _Sky, coords: list[tuple[float, float]],
          radius_arcsec: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each position, the indices into `sky` of the sources within radius_arcsec, and their separations (deg)."""
    r = radius_arcsec / 3600
    out = []
    for ra, dec in coords:
        lo = np.searchsorted(sky.dec, dec - r, side='left')
        hi = np.searchsorted(sky.dec, dec + r, side='right')
        if hi <= lo:
            out.append((np.array([], int), np.array([], float)))
            continue
        d1, d2 = np.radians(dec), np.radians(sky.dec[lo:hi])
        dra = np.radians(sky.ra[lo:hi] - ra)
        # Haversine: well conditioned at arcsecond separations
        h = np.sin((d2 - d1) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin(dra / 2) ** 2
        sep = np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))))
        near = sep <= r
        out.append((lo + np.flatnonzero(near), sep[near]))
    return out


def evaluate(sci: sqlite3.Connection, sci_stamp, text: str, radius_arcsec: float):
    """Parse and match without storing: (summary, positions, matched uuids, values).

    `values` is {'columns': [...], 'rows': {uuid: [...]}}: each matched source
    takes the values of the nearest position that matched it.
    """
    coords, bad, columns, per_position = parse(text)
    sky = _SKY.get(sci, sci_stamp)
    hits = match(sky, coords, radius_arcsec)
    best: dict[int, tuple[float, list]] = {}
    for (idx, sep), vals in zip(hits, per_position):
        for i, d in zip(idx.tolist(), sep.tolist()):
            if i not in best or d < best[i][0]:
                best[i] = (d, vals)
    matched = sorted(best)
    uuids = [sky.uuid[i] for i in matched]
    values = {'columns': columns,
              'rows': {sky.uuid[i]: best[i][1] for i in matched} if columns else {}}
    by_catalog: dict[str, int] = {}
    for i in matched:
        key = CATALOG_OF_TIER.get(sky.tier[i]) or 'other'
        by_catalog[key] = by_catalog.get(key, 0) + 1
    unmatched = [c for c, (idx, _) in zip(coords, hits) if len(idx) == 0]
    summary = {
        'radius_arcsec': radius_arcsec,
        'n_coords': len(coords),
        'n_unparsed': len(bad),
        'unparsed': bad[:MAX_LISTED],
        'n_matched_coords': len(coords) - len(unmatched),
        'unmatched': [f'{a:.6f} {b:+.6f}' for a, b in unmatched[:MAX_LISTED]],
        'n_unmatched': len(unmatched),
        'n_sources': len(uuids),
        'by_catalog': by_catalog,
        'columns': columns,
    }
    return summary, coords, uuids, values


def _with_row(row: sqlite3.Row) -> dict:
    """The stored summary plus the row's own fields, which a rename changes."""
    # `columns` arrived after the first lists were saved; those have none.
    return {'columns': [], **orjson.loads(row['summary_json']), 'id': row['id'],
            'name': row['name'], 'created': row['created']}


def create(store: sqlite3.Connection, sci: sqlite3.Connection, sci_stamp, text: str,
           name: str, radius_arcsec: float) -> dict:
    """Match and save a list under a new id; returns its summary."""
    summary, coords, uuids, values = evaluate(sci, sci_stamp, text, radius_arcsec)
    list_id = secrets.token_hex(6)
    with store:
        store.execute('INSERT INTO coord_lists (id, name, created, summary_json, uuids_json, '
                      'radius_arcsec, positions_json, values_json) VALUES (?,?,?,?,?,?,?,?)',
                      (list_id, name, datetime.now(timezone.utc).isoformat(timespec='seconds'),
                       orjson.dumps(summary).decode(), orjson.dumps(uuids).decode(),
                       radius_arcsec, orjson.dumps(coords).decode(),
                       orjson.dumps(values).decode()))
    return get(store, list_id)


def get(store: sqlite3.Connection, list_id: str) -> dict | None:
    row = store.execute('SELECT * FROM coord_lists WHERE id=? AND deleted_at IS NULL',
                        (list_id,)).fetchone()
    return None if row is None else _with_row(row)


def all_lists(store: sqlite3.Connection) -> list[dict]:
    """Every saved list, newest first."""
    rows = store.execute('SELECT * FROM coord_lists WHERE deleted_at IS NULL '
                         'ORDER BY created DESC, id').fetchall()
    return [_with_row(r) for r in rows]


def _check_exists(store: sqlite3.Connection, list_id: str) -> None:
    if get(store, list_id) is None:
        raise KeyError(list_id)


def rename(store: sqlite3.Connection, list_id: str, name: str) -> dict:
    _check_exists(store, list_id)
    with store:
        store.execute('UPDATE coord_lists SET name=? WHERE id=?', (name, list_id))
    return get(store, list_id)


def delete(store: sqlite3.Connection, list_id: str) -> None:
    """Hide the list. The row stays, so clearing `deleted_at` restores it."""
    _check_exists(store, list_id)
    with store:
        store.execute('UPDATE coord_lists SET deleted_at=? WHERE id=?',
                      (datetime.now(timezone.utc).isoformat(timespec='seconds'), list_id))


def uuids(store: sqlite3.Connection, list_id: str) -> list[str] | None:
    row = store.execute('SELECT uuids_json FROM coord_lists WHERE id=? AND deleted_at IS NULL',
                        (list_id,)).fetchone()
    return None if row is None else orjson.loads(row['uuids_json'])


def values(store: sqlite3.Connection, list_id: str) -> dict | None:
    """The list's value columns: {'columns': [...], 'rows': {uuid: [...]}}; None for an unknown id."""
    row = store.execute('SELECT values_json FROM coord_lists WHERE id=? AND deleted_at IS NULL',
                        (list_id,)).fetchone()
    if row is None:
        return None
    if not row['values_json']:          # saved before value columns existed
        return {'columns': [], 'rows': {}}
    return orjson.loads(row['values_json'])


def as_filter(store: sqlite3.Connection, list_id: str | None) -> dict | None:
    """The filter clause for `coord_list=<id>`: None without one; KeyError for an unknown id."""
    if not list_id:
        return None
    found = uuids(store, list_id)
    if found is None:
        raise KeyError(list_id)
    # An empty `in` would be a list matching nothing; spell that so every engine agrees
    return restrict_to(found) if found else {'column': UUID, 'op': 'in', 'value': ['']}
