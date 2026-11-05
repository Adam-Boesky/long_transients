"""The ztforce queue and its results: ``ztforce.sqlite``.

One row per source. The row's life:

    queued  -- labelled "Run ztforce", or "run now" pressed on the page
    running -- claimed by the worker
    done    -- finished; the page's stacks stored
    error   -- failed; kept with the message until someone retries

The "Run ztforce" label is the source of truth for label-driven runs: labelling
a source queues it, and removing the label before it starts dequeues it. A
finished result made by an older ztforce (a different ``result_version``) is
stale; if the source is still labelled it is queued again, and its old result
keeps being served until the new one lands.

This database is written by the server and is fully regenerable, so it lives
apart from both science.sqlite (read-only on the server) and annotations.sqlite
(the master copy of the labels).
"""
from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    source_uuid   TEXT PRIMARY KEY,
    ra            REAL NOT NULL,
    dec           REAL NOT NULL,
    status        TEXT NOT NULL,              -- queued | running | done | error
    origin        TEXT NOT NULL,              -- label | manual
    priority      INTEGER NOT NULL DEFAULT 0, -- higher runs first ("run now" = 1)
    requested_at  TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    version       TEXT,                       -- result_version() of the stored result
    n_epochs      INTEGER,
    note          TEXT,                       -- e.g. epochs IRSA could not serve
    error         TEXT,
    stacks        BLOB                        -- gzipped JSON the page draws
);
CREATE INDEX IF NOT EXISTS runs_queue ON runs (status, priority DESC, requested_at);
CREATE TABLE IF NOT EXISTS settings (
    key    TEXT PRIMARY KEY,
    value  TEXT
);
"""

_bootstrapped: set[Path] = set()
_bootstrap_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def connect(path: Path) -> sqlite3.Connection:
    path = Path(path)
    with _bootstrap_lock:
        if path not in _bootstrapped:
            path.parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(path)
            try:
                # WAL: the worker writes while page requests read.
                con.execute('PRAGMA journal_mode=WAL')
                con.executescript(SCHEMA)
                con.commit()
            finally:
                con.close()
            _bootstrapped.add(path)
    # isolation_level=None: explicit transactions only, so claim() can take the
    # write lock with BEGIN IMMEDIATE before it reads.
    con = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def get_ztforce_store() -> Iterator[sqlite3.Connection]:
    """Read-write handle on ztforce.sqlite, for one request."""
    from api.config import ZTFORCE_PATH

    con = connect(ZTFORCE_PATH)
    try:
        yield con
    finally:
        con.close()


# ── reads ─────────────────────────────────────────────────────────────────────


def get(con: sqlite3.Connection, source_uuid: str) -> sqlite3.Row | None:
    return con.execute('SELECT * FROM runs WHERE source_uuid = ?', (source_uuid,)).fetchone()


def queue_position(con: sqlite3.Connection, source_uuid: str) -> int | None:
    """0-based place in the queue, or None if the source is not queued."""
    row = con.execute(
        "SELECT priority, requested_at FROM runs WHERE source_uuid = ? AND status = 'queued'",
        (source_uuid,),
    ).fetchone()
    if row is None:
        return None
    return con.execute(
        "SELECT COUNT(*) FROM runs WHERE status = 'queued' AND "
        '(priority > ? OR (priority = ? AND requested_at < ?))',
        (row['priority'], row['priority'], row['requested_at']),
    ).fetchone()[0]


def mean_run_seconds(con: sqlite3.Connection, last: int = 20) -> float | None:
    """Mean duration of the most recent finished runs, for rough ETAs."""
    rows = con.execute(
        "SELECT started_at, finished_at FROM runs WHERE status = 'done' AND started_at IS NOT NULL "
        'ORDER BY finished_at DESC LIMIT ?',
        (last,),
    ).fetchall()
    secs = [
        (datetime.fromisoformat(r['finished_at']) - datetime.fromisoformat(r['started_at'])).total_seconds()
        for r in rows
    ]
    return sum(secs) / len(secs) if secs else None


def counts(con: sqlite3.Connection) -> dict[str, int]:
    out = dict.fromkeys(('queued', 'running', 'done', 'error'), 0)
    for status, n in con.execute('SELECT status, COUNT(*) FROM runs GROUP BY status'):
        out[status] = n
    return out


def running(con: sqlite3.Connection) -> list[sqlite3.Row]:
    return con.execute(
        "SELECT source_uuid, started_at FROM runs WHERE status = 'running' ORDER BY started_at"
    ).fetchall()


def is_paused(con: sqlite3.Connection) -> bool:
    row = con.execute("SELECT value FROM settings WHERE key = 'paused'").fetchone()
    return row is not None and row['value'] == '1'


# ── writes ────────────────────────────────────────────────────────────────────


def set_paused(con: sqlite3.Connection, paused: bool) -> None:
    con.execute(
        "INSERT INTO settings (key, value) VALUES ('paused', ?) "
        'ON CONFLICT (key) DO UPDATE SET value = excluded.value',
        ('1' if paused else '0',),
    )


def sync_labels(
    con: sqlite3.Connection, labelled: dict[str, tuple[float, float]], version: str
) -> tuple[int, int]:
    """Make the queue match the set of labelled sources. Returns (queued, dequeued).

    Labelled sources with no row, or only a stale result, are queued. Label-queued
    sources whose label has gone are dequeued if they have not started: dropped if
    they never had a result, returned to their stale result otherwise. Errors and
    manual requests are left alone.
    """
    now = _now()
    queued = dequeued = 0
    con.execute('BEGIN IMMEDIATE')
    try:
        for source_uuid, (ra, dec) in labelled.items():
            row = get(con, source_uuid)
            if row is None:
                con.execute(
                    "INSERT INTO runs (source_uuid, ra, dec, status, origin, requested_at) "
                    "VALUES (?, ?, ?, 'queued', 'label', ?)",
                    (source_uuid, ra, dec, now),
                )
                queued += 1
            elif row['status'] == 'done' and row['version'] != version:
                con.execute(
                    "UPDATE runs SET status = 'queued', origin = 'label', priority = 0, requested_at = ? "
                    'WHERE source_uuid = ?',
                    (now, source_uuid),
                )
                queued += 1
        for row in con.execute(
            "SELECT source_uuid, stacks FROM runs WHERE status = 'queued' AND origin = 'label'"
        ).fetchall():
            if row['source_uuid'] in labelled:
                continue
            if row['stacks'] is None:
                con.execute('DELETE FROM runs WHERE source_uuid = ?', (row['source_uuid'],))
            else:
                con.execute("UPDATE runs SET status = 'done' WHERE source_uuid = ?", (row['source_uuid'],))
            dequeued += 1
        con.execute('COMMIT')
    except BaseException:
        con.execute('ROLLBACK')
        raise
    return queued, dequeued


def request_now(con: sqlite3.Connection, source_uuid: str, ra: float, dec: float, version: str) -> None:
    """"Run now" from the page: queue at the front, unless a current result exists.

    Under `BEGIN IMMEDIATE` for the same reason `claim` is: this reads the row
    and then writes it. Interleaved with the worker, the read could see
    'queued', the worker claim it to 'running' and start downloading, and this
    write put it back to 'queued' -- so the next tick runs it a second time,
    doubling the IRSA load the module exists to bound.
    """
    con.execute('BEGIN IMMEDIATE')
    try:
        row = get(con, source_uuid)
        if row is not None and row['status'] == 'done' and row['version'] == version:
            return
        if row is not None and row['status'] == 'running':
            return
        if row is None:
            con.execute(
                "INSERT INTO runs (source_uuid, ra, dec, status, origin, priority, requested_at) "
                "VALUES (?, ?, ?, 'queued', 'manual', 1, ?)",
                (source_uuid, ra, dec, _now()),
            )
        else:
            # A queued source jumps ahead; a stale or failed one is queued again.
            con.execute(
                "UPDATE runs SET status = 'queued', origin = 'manual', priority = 1, error = NULL, "
                'requested_at = CASE WHEN status = ? THEN requested_at ELSE ? END WHERE source_uuid = ?',
                ('queued', _now(), source_uuid),
            )
    finally:
        con.execute('COMMIT')


def claim(con: sqlite3.Connection, n: int) -> list[sqlite3.Row]:
    """Atomically move up to n queued sources to running, highest priority first."""
    if n <= 0:
        return []
    con.execute('BEGIN IMMEDIATE')
    try:
        rows = con.execute(
            "SELECT source_uuid, ra, dec FROM runs WHERE status = 'queued' "
            'ORDER BY priority DESC, requested_at LIMIT ?',
            (n,),
        ).fetchall()
        now = _now()
        for row in rows:
            con.execute(
                "UPDATE runs SET status = 'running', started_at = ?, finished_at = NULL WHERE source_uuid = ?",
                (now, row['source_uuid']),
            )
        con.execute('COMMIT')
    except BaseException:
        con.execute('ROLLBACK')
        raise
    return rows


def finish(
    con: sqlite3.Connection,
    source_uuid: str,
    version: str,
    stacks: bytes,
    n_epochs: int,
    note: str | None,
) -> None:
    con.execute(
        "UPDATE runs SET status = 'done', finished_at = ?, version = ?, stacks = ?, "
        'n_epochs = ?, note = ?, error = NULL, priority = 0 WHERE source_uuid = ?',
        (_now(), version, stacks, n_epochs, note, source_uuid),
    )


def fail(con: sqlite3.Connection, source_uuid: str, error: str) -> None:
    con.execute(
        "UPDATE runs SET status = 'error', finished_at = ?, error = ?, priority = 0 WHERE source_uuid = ?",
        (_now(), error, source_uuid),
    )


def recover(con: sqlite3.Connection) -> int:
    """Return runs left 'running' by a stopped worker to the queue."""
    return con.execute("UPDATE runs SET status = 'queued' WHERE status = 'running'").rowcount


def row_summary(row: sqlite3.Row | None) -> dict[str, Any]:
    """The plain fields of a run row, for responses."""
    if row is None:
        return {'status': 'none'}
    keys = ('status', 'origin', 'requested_at', 'started_at', 'finished_at', 'version', 'n_epochs', 'note', 'error')
    return {k: row[k] for k in keys}
