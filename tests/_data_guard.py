"""Keep the test suite away from the real data.

Imported by `conftest.py` before anything else, so it is in force before any
application module computes a path. It does two things: points every data
path at a throwaway directory (see "redirect" below), and installs a Python
audit hook that refuses, in this process:

  - any SQLite connection to, or file opened under, a live data directory
    (`Data/db/`, and wherever `LT_DATA_DIR` or the `LT_*_DB` overrides pointed
    before the tests changed them), read or write: SQLite can write on a
    "read" (journals, migrations), and a test that reads the real catalog is
    a test whose result depends on it;
  - any write anywhere else under `Data/`.

An audit hook cannot be removed, and it sees opens from C extensions that go
through Python's file layer. It does not see a native library opening a file
itself (DuckDB, PyArrow), which is why the redirect check below also asserts
that every configured path points into the throwaway directory.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class RealDataAccess(RuntimeError):
    """A test reached for the real data."""


def _env_file_data_dir() -> str | None:
    try:
        lines = (REPO / '.env').read_text().splitlines()
    except OSError:
        return None
    vals = [ln.split('=', 1)[1].strip() for ln in lines if ln.startswith('LT_DATA_DIR=')]
    return vals[-1] if vals else None


def _real(p: str | os.PathLike) -> str:
    return os.path.realpath(os.fspath(p))


# Captured before anything below rewrites the environment.
_ORIGINAL = {k: os.environ[k] for k in os.environ if k.startswith('LT_')}
LIVE_DIRS = {_real(REPO / 'Data' / 'db')}
for _d in (_ORIGINAL.get('LT_DATA_DIR'), _env_file_data_dir()):
    if _d:
        LIVE_DIRS.add(_real(_d))
LIVE_FILES = {_real(v) for k, v in _ORIGINAL.items()
              if k.endswith('_DB') or k == 'LT_CUTOUT_CACHE'}
DATA_ROOT = _real(REPO / 'Data')


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + os.sep)


def check(path: str, *, write: bool) -> None:
    """Raise if `path` is real data (any access) or under `Data/` (writes)."""
    p = _real(path)
    if any(_under(p, d) for d in LIVE_DIRS) or any(_under(p, f) for f in LIVE_FILES):
        raise RealDataAccess(f'test tried to open live data: {p}')
    if write and _under(p, DATA_ROOT):
        raise RealDataAccess(f'test tried to write under Data/: {p}')


_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _hook(event: str, args: tuple) -> None:
    if event == 'open':
        path, mode, flags = args
        if not isinstance(path, (str, bytes, os.PathLike)):
            return                              # an fd
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        write = (any(c in mode for c in 'wax+') if isinstance(mode, str)
                 else bool((flags or 0) & _WRITE_FLAGS))
        check(path, write=write)
    elif event == 'sqlite3.connect':
        db = os.fspath(args[0])
        if isinstance(db, bytes):
            db = os.fsdecode(db)
        if db and db != ':memory:':
            if db.startswith('file:'):
                db = db[5:].split('?', 1)[0]
            if db:
                check(db, write=True)


sys.addaudithook(_hook)


# ── redirect ──────────────────────────────────────────────────────────────
# Every database and cache path derives from LT_DATA_DIR (`source_db.paths`),
# so it is forced -- not defaulted -- to a throwaway directory, and the
# individual overrides that would win over it are dropped. `conftest` builds
# the synthetic catalog here, so the app's startup loads five sources rather
# than the real 198k.
TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix='lt-test-data-'))
os.environ['LT_DATA_DIR'] = str(TEST_DATA_DIR)
for _k in [k for k in os.environ if k.startswith('LT_') and (k.endswith('_DB') or k == 'LT_CUTOUT_CACHE')]:
    del os.environ[_k]


def assert_redirected() -> None:
    """Every configured path points into TEST_DATA_DIR; raise otherwise."""
    from api import config
    from source_db import paths
    from source_db.cutouts import render
    root = _real(TEST_DATA_DIR)
    for name, p in [('DEFAULT_OUT_DIR', paths.DEFAULT_OUT_DIR), ('RAW_DIR', paths.RAW_DIR),
                    ('SCIENCE_PATH', config.SCIENCE_PATH), ('ANNOTATIONS_PATH', config.ANNOTATIONS_PATH),
                    ('ZTFORCE_PATH', config.ZTFORCE_PATH), ('COORD_LISTS_PATH', config.COORD_LISTS_PATH),
                    ('CUTOUT_CACHE', render.CACHE_DIR)]:
        if not _under(_real(p), root):
            raise RealDataAccess(f'{name} is not redirected: {p}')


def remove_test_data_dir() -> None:
    """Delete TEST_DATA_DIR, and only it."""
    root = _real(TEST_DATA_DIR)
    if (os.path.basename(root).startswith('lt-test-data-')
            and _under(root, _real(tempfile.gettempdir()))
            and not any(_under(root, d) or _under(d, root) for d in LIVE_DIRS | {DATA_ROOT})):
        shutil.rmtree(root, ignore_errors=True)
