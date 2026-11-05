"""Each request gets its own SQLite connection.

This guards a bug that is invisible in a single-request test and silent when
it fires. `get_science` used to cache one connection per thread, on the
reasoning that FastAPI runs sync endpoints in a threadpool. But FastAPI
resolves a sync dependency in one threadpool task and runs the handler in
another, with no guarantee they share a thread -- so a connection cached
against the resolving thread could be handed to two handlers at once.

Concurrent use of one sqlite3 connection does not raise. It returns the other
query's row, a row whose columns are NULL, or no row at all, which reached the
browser as sporadic 404s for sources that plainly exist and 500s from
`orjson.loads(catalog_json)` being given None.

The test asserts the structural property rather than trying to race threads,
which would be flaky in both directions: distinct connection objects per call,
and closed again when the request ends.
"""
from __future__ import annotations

import sqlite3

from api import db as api_db


def _drain(gen):
    """Run a dependency generator to its yield, as FastAPI does."""
    con = next(gen)
    return con, gen


def test_science_connections_are_not_shared_between_requests():
    a, ga = _drain(api_db.get_science())
    b, gb = _drain(api_db.get_science())
    assert a is not b
    for g in (ga, gb):
        next(g, None)            # run the finally: block


def test_annotation_connections_are_not_shared_between_requests(tmp_path,
                                                                monkeypatch):
    monkeypatch.setattr(api_db, 'ANNOTATIONS_PATH', tmp_path / 'ann.sqlite')
    a, ga = _drain(api_db.get_annotations())
    b, gb = _drain(api_db.get_annotations())
    assert a is not b
    for g in (ga, gb):
        next(g, None)


def test_the_connection_is_closed_when_the_request_ends():
    gen = api_db.get_science()
    con = next(gen)
    con.execute('SELECT 1').fetchone()       # usable while the request runs
    next(gen, None)
    try:
        con.execute('SELECT 1')
    except sqlite3.ProgrammingError:
        return
    raise AssertionError('connection outlived the request')
