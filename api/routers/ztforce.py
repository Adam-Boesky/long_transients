"""ztforce forced photometry on the candidate page, via a background queue.

ztforce measures flux at a fixed sky position in every available ZTF science
image, so it finds a source whether or not anything was ever detected there. It
downloads and PSF-fits images and takes minutes, so runs go through a queue
(``api.ztforce_store``) drained by a background worker (``api.ztforce_worker``):

- labelling a source "Run ztforce" queues it; removing the label dequeues it;
- "run now" on the page queues a source at the front;
- finished results are stored, so the page loads them instantly.

The payload is ztforce's stack for every window (see ``api.ztforce_results``).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api import ztforce_results as results
from api import ztforce_store as store
from api import ztforce_worker
from api.config import ZTFORCE_CONCURRENCY, ZTFORCE_LABEL
from api.db import get_science

router = APIRouter()


def _seconds_since(iso: str | None) -> float | None:
    if not iso:
        return None
    return round((datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds(), 1)


def _payload(con: sqlite3.Connection, source_uuid: str) -> dict[str, Any]:
    row = store.get(con, source_uuid)
    out: dict[str, Any] = {
        'source_uuid': source_uuid,
        **store.row_summary(row),
        'windows': list(results.WINDOWS_DAYS),
        'stacks': None,
        'stale': False,
        'queue_position': None,
        'eta_s': None,
        'elapsed_s': None,
    }
    if row is None:
        return out
    # A stored result is served even while a newer one is queued or running.
    out['stacks'] = results.unpack(row['stacks'])
    out['stale'] = row['stacks'] is not None and row['version'] != results.result_version()
    if row['status'] == 'running':
        out['elapsed_s'] = _seconds_since(row['started_at'])
    elif row['status'] == 'queued':
        pos = store.queue_position(con, source_uuid)
        out['queue_position'] = pos
        mean = store.mean_run_seconds(con)
        if pos is not None and mean is not None:
            out['eta_s'] = round(mean * (pos // ZTFORCE_CONCURRENCY + 1))
    return out


def _source_coords(sci: sqlite3.Connection, source_uuid: str) -> tuple[float, float]:
    row = sci.execute('SELECT ra, dec FROM sources WHERE source_uuid=?', (source_uuid,)).fetchone()
    if row is None:
        raise HTTPException(404, f'no source {source_uuid}')
    if row['ra'] is None or row['dec'] is None:
        raise HTTPException(422, f'source {source_uuid} has no coordinates')
    return float(row['ra']), float(row['dec'])


@router.get('/sources/{source_uuid}/ztforce')
def get_ztforce(source_uuid: str, con: sqlite3.Connection = Depends(store.get_ztforce_store)) -> dict[str, Any]:
    """This source's ztforce status and, if there is one, its stored result."""
    return _payload(con, source_uuid)


@router.post('/sources/{source_uuid}/ztforce')
def run_ztforce_now(
    source_uuid: str,
    sci: sqlite3.Connection = Depends(get_science),
    con: sqlite3.Connection = Depends(store.get_ztforce_store),
) -> dict[str, Any]:
    """Queue this source at the front, unless it already has a current result."""
    ra, dec = _source_coords(sci, source_uuid)
    store.request_now(con, source_uuid, ra, dec, results.result_version())
    ztforce_worker.nudge()
    return _payload(con, source_uuid)


class QueueSettings(BaseModel):
    paused: bool


def _queue(con: sqlite3.Connection) -> dict[str, Any]:
    n = store.counts(con)
    mean = store.mean_run_seconds(con)
    return {
        'label': ZTFORCE_LABEL,
        'paused': store.is_paused(con),
        'concurrency': ZTFORCE_CONCURRENCY,
        'n_queued': n['queued'],
        'n_running': n['running'],
        'n_done': n['done'],
        'n_error': n['error'],
        'running': [
            {'source_uuid': r['source_uuid'], 'elapsed_s': _seconds_since(r['started_at'])}
            for r in store.running(con)
        ],
        # Rough time to drain what is queued now.
        'eta_s': round(mean * -(-n['queued'] // ZTFORCE_CONCURRENCY)) if mean and n['queued'] else None,
    }


@router.get('/ztforce/queue')
def get_queue(con: sqlite3.Connection = Depends(store.get_ztforce_store)) -> dict[str, Any]:
    return _queue(con)


@router.put('/ztforce/queue')
def set_queue(settings: QueueSettings, con: sqlite3.Connection = Depends(store.get_ztforce_store)) -> dict[str, Any]:
    """Pause or resume the queue. Runs already started finish either way."""
    store.set_paused(con, settings.paused)
    if not settings.paused:
        ztforce_worker.nudge()
    return _queue(con)
