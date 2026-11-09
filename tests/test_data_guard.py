"""The suite cannot reach the real data (see `tests/_data_guard.py`).

Each probe is harmless even if the guard regressed: a read-only URI on a file
that does not exist, or a write into a directory that does not exist, so
nothing under `Data/` can be created or changed by these tests.
"""
from __future__ import annotations

import os
import sqlite3

import pytest

from tests import _data_guard

LIVE = _data_guard.REPO / 'Data' / 'db'


def test_every_configured_path_is_redirected():
    _data_guard.assert_redirected()


def test_sqlite_cannot_open_live_data():
    with pytest.raises(_data_guard.RealDataAccess):
        sqlite3.connect(f'file:{LIVE}/__guard_probe__.sqlite?mode=ro', uri=True)


def test_files_under_live_data_cannot_be_read():
    with pytest.raises(_data_guard.RealDataAccess):
        open(LIVE / '__guard_probe__.txt')


def test_nothing_under_data_can_be_written():
    target = _data_guard.REPO / 'Data' / '__guard_no_such_dir__' / 'x'
    with pytest.raises(_data_guard.RealDataAccess):
        open(target, 'x')
    with pytest.raises(_data_guard.RealDataAccess):
        os.open(target, os.O_CREAT | os.O_WRONLY)


def test_the_app_starts_against_the_test_catalog():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        assert c.get('/api/health').json()['science_db'].startswith(
            str(_data_guard.TEST_DATA_DIR))
