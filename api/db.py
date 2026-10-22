"""SQLite connections for the two tiers, as FastAPI dependencies.

Exposed as `Depends(...)` callables rather than module-level globals so tests
can swap in a throwaway database with `app.dependency_overrides` instead of
manipulating environment variables and import order.

One connection per *request*, not per thread. The per-thread version of this
looked right -- sqlite3 connections are not safe to use concurrently, and
FastAPI runs sync endpoints in a threadpool -- but it is not: FastAPI resolves
a sync dependency in one threadpool task and runs the endpoint body in
another, and nothing pins those to the same thread. So a connection cached
against the thread that resolved the dependency could be handed to a handler
running on a different thread, and to a second request's handler at the same
time. Two handlers then interleaved `execute`/`fetchone` on one connection,
which does not raise: it returns the other query's row, a row with NULL
columns, or no row at all. That surfaced as sporadic 404s from `/sources/
{uuid}` for sources that exist and 500s from `orjson.loads(catalog_json)`
being handed None, under the concurrent fetches the classify page makes
(window, detail, light curve, spectra, six cutouts, and a prefetch of the
next source, all in flight together).

`check_same_thread=False` stays, and is now load-bearing for a different
reason than it was: the connection still legitimately crosses threads once,
from the task that resolved the dependency to the task that runs the handler.
Dropping it makes SQLite say so out loud -- "created in thread A ... this is
thread B" on most requests -- which is how the hand-off above was confirmed.
Crossing threads is fine as long as only one request holds the connection,
and that is exactly what per-request connections guarantee.

Connecting per request costs microseconds -- SQLite parses a five-table
schema and the OS page cache is shared across connections, so the 10 GB file
is not re-read -- and it removes the sharing entirely rather than papering
over it with a lock.

The science database is opened read-only so a bug in a handler cannot corrupt
a rebuildable artefact, and so several API processes can share one file
safely.
"""
from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

from api.config import ANNOTATIONS_PATH, SCIENCE_PATH
from source_db.serving.schema import ANNOTATIONS_SCHEMA

# The annotations schema and its WAL setting are applied once per process
# rather than on every connection: `executescript` implicitly commits, which
# has no business running inside a request.
_bootstrapped: set[Path] = set()
_bootstrap_lock = threading.Lock()


def connect_science(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True,
                          check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _bootstrap_annotations(path: Path) -> None:
    with _bootstrap_lock:
        if path in _bootstrapped:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path, timeout=30)
        try:
            # WAL lets the three of you annotate concurrently with readers.
            con.execute('PRAGMA journal_mode=WAL')
            # Before the schema, not after: it indexes `annotations.status`,
            # which an older table does not have yet. A fresh database has no
            # tables, so these no-op and the schema creates them whole.
            for table in ('annotations', 'history'):
                have = {r[1] for r in con.execute(f'PRAGMA table_info({table})')}
                if have and 'status' not in have:
                    con.execute(f'ALTER TABLE {table} ADD COLUMN status TEXT')
            con.executescript(ANNOTATIONS_SCHEMA)
            # CREATE TABLE IF NOT EXISTS leaves an older labels table as it
            # was, so add the columns added since: order, then colour.
            cols = {r[1] for r in con.execute('PRAGMA table_info(labels)')}
            for col, kind in (('position', 'INTEGER'), ('color', 'TEXT')):
                if col not in cols:
                    con.execute(f'ALTER TABLE labels ADD COLUMN {col} {kind}')
            con.commit()
        finally:
            con.close()
        _bootstrapped.add(path)


def connect_annotations(path: Path) -> sqlite3.Connection:
    _bootstrap_annotations(Path(path))
    # `timeout=30` to match the ztforce store: under WAL a writer waits for
    # the lock, and the default 5 s raises `database is locked` -- which
    # surfaces as a 500 on a PUT, i.e. a silently lost verdict.
    con = sqlite3.connect(path, check_same_thread=False, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def get_science() -> Iterator[sqlite3.Connection]:
    """Read-only handle on the static science database, for one request."""
    if not SCIENCE_PATH.exists():
        raise FileNotFoundError(
            f'{SCIENCE_PATH} not found -- run '
            f'`python -m source_db.serving.build_serving_db`')
    con = connect_science(SCIENCE_PATH)
    try:
        yield con
    finally:
        con.close()


def get_annotations() -> Iterator[sqlite3.Connection]:
    """Read-write handle on the annotation database, for one request."""
    con = connect_annotations(ANNOTATIONS_PATH)
    try:
        yield con
    finally:
        con.close()
