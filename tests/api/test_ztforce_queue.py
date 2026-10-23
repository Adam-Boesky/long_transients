"""The ztforce queue: label-driven queueing, "run now", the worker, and the page API."""
from __future__ import annotations

import time

import numpy as np
import pytest

from api import ztforce_results as results
from api import ztforce_store as store
from api import ztforce_worker
from tests.conftest import UUIDS

VERSION = 'v-test'
A, B, C = UUIDS[0], UUIDS[1], UUIDS[2]


def _status(con, uuid):
    row = store.get(con, uuid)
    return None if row is None else row['status']


def _finish(con, uuid, version=VERSION, stacks=None):
    store.finish(con, uuid, version, results.pack(stacks or {'0': {}}), 5, None)


# ── store: label sync ─────────────────────────────────────────────────────────


def test_labelling_queues_and_unlabelling_dequeues(ztforce_con):
    store.sync_labels(ztforce_con, {A: (1.0, 2.0), B: (3.0, 4.0)}, VERSION)
    assert _status(ztforce_con, A) == _status(ztforce_con, B) == 'queued'

    store.sync_labels(ztforce_con, {A: (1.0, 2.0)}, VERSION)  # B's label removed before it ran
    assert _status(ztforce_con, A) == 'queued'
    assert _status(ztforce_con, B) is None


def test_sync_leaves_running_errors_and_manual_requests_alone(ztforce_con):
    store.sync_labels(ztforce_con, {A: (1.0, 2.0), B: (3.0, 4.0)}, VERSION)
    store.claim(ztforce_con, 1)  # A starts
    store.fail(ztforce_con, B, 'boom')
    store.request_now(ztforce_con, C, 5.0, 6.0, VERSION)

    store.sync_labels(ztforce_con, {}, VERSION)  # every label removed
    assert _status(ztforce_con, A) == 'running'
    assert _status(ztforce_con, B) == 'error'
    assert _status(ztforce_con, C) == 'queued'


def test_stale_result_is_requeued_and_kept_until_replaced(ztforce_con):
    store.sync_labels(ztforce_con, {A: (1.0, 2.0)}, 'old')
    store.claim(ztforce_con, 1)
    _finish(ztforce_con, A, version='old')

    store.sync_labels(ztforce_con, {A: (1.0, 2.0)}, VERSION)  # ztforce was upgraded
    row = store.get(ztforce_con, A)
    assert row['status'] == 'queued' and row['stacks'] is not None

    store.sync_labels(ztforce_con, {}, VERSION)  # label removed before the re-run
    row = store.get(ztforce_con, A)
    assert row['status'] == 'done' and row['version'] == 'old'  # back to the stale result


def test_current_result_is_not_requeued(ztforce_con):
    store.sync_labels(ztforce_con, {A: (1.0, 2.0)}, VERSION)
    store.claim(ztforce_con, 1)
    _finish(ztforce_con, A)
    store.sync_labels(ztforce_con, {A: (1.0, 2.0)}, VERSION)
    assert _status(ztforce_con, A) == 'done'


# ── store: run now, claiming, recovery ────────────────────────────────────────


def test_run_now_jumps_the_queue(ztforce_con):
    store.sync_labels(ztforce_con, {A: (1.0, 2.0), B: (3.0, 4.0)}, VERSION)
    store.request_now(ztforce_con, B, 3.0, 4.0, VERSION)
    assert store.queue_position(ztforce_con, B) == 0
    assert [r['source_uuid'] for r in store.claim(ztforce_con, 2)] == [B, A]


def test_run_now_retries_errors_but_not_current_results(ztforce_con):
    store.request_now(ztforce_con, A, 1.0, 2.0, VERSION)
    store.claim(ztforce_con, 1)
    store.fail(ztforce_con, A, 'boom')
    store.request_now(ztforce_con, A, 1.0, 2.0, VERSION)
    row = store.get(ztforce_con, A)
    assert row['status'] == 'queued' and row['error'] is None

    store.claim(ztforce_con, 1)
    _finish(ztforce_con, A)
    store.request_now(ztforce_con, A, 1.0, 2.0, VERSION)
    assert _status(ztforce_con, A) == 'done'


def test_claim_takes_at_most_n_and_recover_requeues(ztforce_con):
    store.sync_labels(ztforce_con, {A: (1.0, 2.0), B: (3.0, 4.0), C: (5.0, 6.0)}, VERSION)
    assert len(store.claim(ztforce_con, 2)) == 2
    assert store.counts(ztforce_con)['running'] == 2
    assert store.claim(ztforce_con, 0) == []
    assert store.recover(ztforce_con) == 2
    assert store.counts(ztforce_con)['queued'] == 3


def test_pause_flag(ztforce_con):
    assert not store.is_paused(ztforce_con)
    store.set_paused(ztforce_con, True)
    assert store.is_paused(ztforce_con)
    store.set_paused(ztforce_con, False)
    assert not store.is_paused(ztforce_con)


# ── API ───────────────────────────────────────────────────────────────────────


def test_get_unknown_source_has_no_run(client):
    body = client.get(f'/api/sources/{A}/ztforce').json()
    assert body['status'] == 'none' and body['stacks'] is None
    assert body['windows'] == list(results.WINDOWS_DAYS)


def test_run_now_queues_a_source(client, ztforce_con):
    body = client.post(f'/api/sources/{A}/ztforce').json()
    assert body['status'] == 'queued' and body['origin'] == 'manual'
    assert body['queue_position'] == 0


def test_run_now_unknown_source_is_404(client):
    assert client.post('/api/sources/not-a-uuid/ztforce').status_code == 404


def test_finished_result_is_served(client, ztforce_con):
    client.post(f'/api/sources/{A}/ztforce')
    store.claim(ztforce_con, 1)
    stacks = {'0': {'g': {'mjd': [1.0], 'mag': [20.0], 'magerr': [0.1], 'is_upper_limit': [False]}}}
    store.finish(ztforce_con, A, results.result_version(), results.pack(stacks), 1, 'note')

    body = client.get(f'/api/sources/{A}/ztforce').json()
    assert body['status'] == 'done' and not body['stale']
    assert body['stacks'] == stacks and body['n_epochs'] == 1 and body['note'] == 'note'


def test_old_version_result_is_marked_stale(client, ztforce_con):
    client.post(f'/api/sources/{A}/ztforce')
    store.claim(ztforce_con, 1)
    _finish(ztforce_con, A, version='old')
    assert client.get(f'/api/sources/{A}/ztforce').json()['stale']


def test_queue_summary_and_pause(client):
    client.post(f'/api/sources/{A}/ztforce')
    q = client.get('/api/ztforce/queue').json()
    assert q['n_queued'] == 1 and not q['paused']
    assert client.put('/api/ztforce/queue', json={'paused': True}).json()['paused']


# ── worker ────────────────────────────────────────────────────────────────────


def _fake_lightcurves(ra, dec):
    """Real ztforce Lightcurves (so stacking runs for real), with fake epochs."""
    from ztforce.lightcurve import Lightcurve

    lcs = {}
    rng = np.random.default_rng(0)
    for band in ('g', 'r'):
        lc = Lightcurve(ra=ra, dec=dec)
        for i in range(40):
            flux = 1000.0 + rng.normal(0, 30)
            lc.add_epoch(2459000.0 + 10 * i, band, flux, 30.0, 26.3 - 2.5 * np.log10(flux), 0.03, 26.3, 0)
        lcs[band] = lc
    return lcs


def test_worker_runs_labelled_sources_and_stores_results(
    client, built_db, ztforce_con, tmp_path, monkeypatch
):
    calls = []

    def fake_run_one(uuid, ra, dec, data_dir=None):
        calls.append(uuid)
        return results.summarise(_fake_lightcurves(ra, dec))

    monkeypatch.setattr(ztforce_worker, 'run_one', fake_run_one)
    client.put(f'/api/sources/{A}/annotations/adam', json={'label': 'Run ztforce'})

    worker = ztforce_worker.Worker(
        tmp_path / 'ztforce.sqlite', tmp_path / 'annotations.sqlite', built_db, 'Run ztforce', concurrency=1
    )
    worker.tick()  # sync the label, claim, start the run
    for _ in range(100):
        if _status(ztforce_con, A) == 'done':
            break
        time.sleep(0.05)
    worker.stop()

    assert calls == [A]
    body = client.get(f'/api/sources/{A}/ztforce').json()
    assert body['status'] == 'done' and body['n_epochs'] == 80
    assert set(body['stacks']) == {str(w) for w in results.WINDOWS_DAYS}
    assert set(body['stacks']['365']) == {'g', 'r'}
    assert all(m == pytest.approx(18.8, abs=0.05) for m in body['stacks']['365']['g']['mag'])


def test_worker_records_failures(client, built_db, ztforce_con, tmp_path, monkeypatch):
    def boom(uuid, ra, dec, data_dir=None):
        raise RuntimeError('IRSA down')

    monkeypatch.setattr(ztforce_worker, 'run_one', boom)
    client.post(f'/api/sources/{A}/ztforce')
    worker = ztforce_worker.Worker(
        tmp_path / 'ztforce.sqlite', tmp_path / 'annotations.sqlite', built_db, 'Run ztforce', concurrency=1
    )
    worker.tick()
    for _ in range(100):
        if _status(ztforce_con, A) == 'error':
            break
        time.sleep(0.05)
    worker.stop()
    assert store.get(ztforce_con, A)['error'] == 'RuntimeError: IRSA down'


def test_paused_worker_starts_nothing(client, built_db, ztforce_con, tmp_path, monkeypatch):
    monkeypatch.setattr(ztforce_worker, 'run_one', lambda *a, **k: pytest.fail('should not run'))
    client.post(f'/api/sources/{A}/ztforce')
    client.put('/api/ztforce/queue', json={'paused': True})
    worker = ztforce_worker.Worker(
        tmp_path / 'ztforce.sqlite', tmp_path / 'annotations.sqlite', built_db, 'Run ztforce', concurrency=1
    )
    worker.tick()
    worker.stop()
    assert _status(ztforce_con, A) == 'queued'


def test_note_counts_unmeasured_epochs_from_flags():
    """The stored note says how many epochs could not be measured, from the flags."""
    from ztforce._constants import FLAG_DOWNLOAD_FAILED, FLAG_UNAVAILABLE

    lcs = _fake_lightcurves(1.0, 2.0)
    for jd, flag in ((2460000.0, FLAG_UNAVAILABLE), (2460001.0, FLAG_UNAVAILABLE), (2460002.0, FLAG_DOWNLOAD_FAILED)):
        lcs['g'].add_epoch(jd, 'g', float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), flag)
    note = results.summarise(lcs)['note']
    assert note == '2 epochs not served by IRSA, 1 epochs download failed'
    assert results.summarise(_fake_lightcurves(1.0, 2.0))['note'] is None
