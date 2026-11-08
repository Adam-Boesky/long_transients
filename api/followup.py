"""One follow-up status per object, shared by everyone.

A verdict (the bucket, the comment) is one author's opinion and stays
per-author in `annotations`. Whether an object is to be queued, is queued, or
has been observed is a fact about the object: two people cannot usefully hold
different ones, and the same object seen in two overlapping fields (a
`group_id`) must not be queued twice. So the status lives in its own table,
keyed by source but written to every member of the source's group, and the
last person to set it sets it for all.

The statuses are ordered. Where two have to be merged -- the per-author
values an older database holds, or group-mates that drifted apart between
releases -- the later stage wins: an object someone has observed is observed
whatever anyone else said, and one already in a queue should not be queued
again.
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from typing import Iterable, Optional

from api.models import FOLLOWUP_STATUSES

RANK = {s: i for i, s in enumerate(FOLLOWUP_STATUSES)}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def merged(statuses: Iterable[Optional[str]]) -> Optional[str]:
    """The furthest-along of `statuses`; None when none is set."""
    known = [s for s in statuses if s in RANK]
    return max(known, key=RANK.__getitem__) if known else None


def read(con: sqlite3.Connection, source_uuid: str) -> Optional[dict]:
    row = con.execute('SELECT source_uuid, status, author, updated_at '
                      'FROM followup WHERE source_uuid=?', (source_uuid,)).fetchone()
    return dict(row) if row else None


def read_all(con: sqlite3.Connection) -> dict[str, dict]:
    return {r['source_uuid']: dict(r) for r in con.execute(
        'SELECT source_uuid, status, author, updated_at FROM followup')}


def set_status(con: sqlite3.Connection, source_uuids: Iterable[str],
               status: Optional[str], author: Optional[str],
               record: bool = True) -> None:
    """Give every uuid in `source_uuids` (one object's detections) `status`.

    None clears it. Each change is appended to `followup_history`, one row per
    uuid, so the trail says which detection was on screen when it was set.
    """
    now = _now()
    uuids = list(dict.fromkeys(source_uuids))
    if status is None:
        con.executemany('DELETE FROM followup WHERE source_uuid=?',
                        [(u,) for u in uuids])
    else:
        con.executemany(
            'INSERT INTO followup (source_uuid, status, author, updated_at) '
            'VALUES (?,?,?,?) ON CONFLICT(source_uuid) DO UPDATE SET '
            'status=excluded.status, author=excluded.author, '
            'updated_at=excluded.updated_at',
            [(u, status, author, now) for u in uuids])
    if record:
        con.executemany(
            'INSERT INTO followup_history (source_uuid, status, author, recorded_at) '
            'VALUES (?,?,?,?)', [(u, status, author, now) for u in uuids])


def backfill_from_annotations(con: sqlite3.Connection) -> int:
    """Fold the per-author `annotations.status` column into `followup`.

    For a database from before the shared status existed. Per source, the
    furthest-along of what anyone set wins, credited to whoever set it most
    recently; a source `followup` already has keeps what it has. The column is
    then cleared in the same transaction, so the database never holds two
    answers and a re-run finds nothing to do. `history` keeps every per-author
    value as it was set. Returns how many sources were given a status.
    """
    have = {r[0] for r in con.execute('SELECT source_uuid FROM followup')}
    best: dict[str, tuple[str, str, str]] = {}
    rows = con.execute(
        'SELECT source_uuid, author, status, updated_at FROM annotations '
        "WHERE status IS NOT NULL AND trim(status) != '' "
        'ORDER BY updated_at').fetchall()
    for r in rows:
        if r['source_uuid'] in have or r['status'] not in RANK:
            continue
        cur = best.get(r['source_uuid'])
        # Later stage wins; within a stage the latest edit is the one credited.
        if cur is None or RANK[r['status']] >= RANK[cur[0]]:
            best[r['source_uuid']] = (r['status'], r['author'], r['updated_at'])
    con.executemany(
        'INSERT INTO followup (source_uuid, status, author, updated_at) VALUES (?,?,?,?)',
        [(u, s, a, t) for u, (s, a, t) in best.items()])
    con.execute('UPDATE annotations SET status = NULL WHERE status IS NOT NULL')
    con.commit()
    return len(best)


def sync_groups(con: sqlite3.Connection, cat) -> int:
    """Make every group agree with its furthest-along member.

    `cat` is a `CatalogFilters` for the current release. Needed after a
    rebuild changes which detections share a `group_id`, and after a backfill
    (which knows nothing about groups). Returns how many rows were written.
    """
    rows = read_all(con)
    seen: set[str] = set()
    written = 0
    for uuid, row in rows.items():
        if uuid in seen:
            continue
        members = [m['source_uuid'] for m in cat.group_members(uuid)] or [uuid]
        seen.update(members)
        present = [rows[m] for m in members if m in rows]
        status = merged(r['status'] for r in present)
        if status is None:        # a value outside the vocabulary: leave it be
            continue
        # Credit the latest setter of the winning status.
        setter = max((r for r in present if r['status'] == status),
                     key=lambda r: r['updated_at'])
        stale = [m for m in members
                 if m not in rows or rows[m]['status'] != status]
        if stale:
            con.executemany(
                'INSERT INTO followup (source_uuid, status, author, updated_at) '
                'VALUES (?,?,?,?) ON CONFLICT(source_uuid) DO UPDATE SET '
                'status=excluded.status, author=excluded.author, '
                'updated_at=excluded.updated_at',
                [(m, status, setter['author'], setter['updated_at']) for m in stale])
            written += len(stale)
    con.commit()
    return written


def main(argv: list[str] | None = None) -> None:
    """`python -m api.followup sync`: make every group agree, against the
    configured science database and catalog. (The per-author backfill needs
    no command: `api.db` runs it whenever it opens the annotations database.)
    """
    import contextlib

    from api.catalog_filters import startup
    from api.config import ANNOTATIONS_PATH, SCIENCE_PATH
    from api.db import connect_annotations, connect_science

    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sync', help='make every group agree with its furthest-along member')
    s.add_argument('--db', default=str(ANNOTATIONS_PATH))
    args = ap.parse_args(argv)

    with contextlib.closing(connect_science(SCIENCE_PATH)) as sci:
        cat = startup(sci)
    with contextlib.closing(connect_annotations(args.db)) as con:
        print(f'{sync_groups(con, cat)} rows rewritten to agree with their group')


if __name__ == '__main__':
    main()
