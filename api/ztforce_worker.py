"""Background worker that drains the ztforce queue.

Every few seconds it syncs the queue with the "Run ztforce" label, then, unless
the queue is paused, starts queued sources until ``ZTFORCE_CONCURRENCY`` are
running. Each finished source is written to ztforce.sqlite as soon as it is
done, so the page shows results one by one.

Runs inside the API process, started from its lifespan (``LT_ZTFORCE_WORKER=0``
turns it off, as the tests do). Run only one API process per ztforce.sqlite:
claiming is atomic, so a second worker would not duplicate work, but it would
double the load on IRSA.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from api import ztforce_results as results
from api import ztforce_store as store

log = logging.getLogger(__name__)

POLL_SECONDS = 5.0

_current: Worker | None = None


def nudge() -> None:
    """Wake the in-process worker now (e.g. after "run now"), if there is one."""
    if _current is not None:
        _current.wake.set()


def labelled_sources(annotations_path: Path, science_path: Path, label: str) -> dict[str, tuple[float, float]]:
    """source_uuid -> (ra, dec) for every source any author has labelled ``label``."""
    ann = sqlite3.connect(f'file:{annotations_path}?mode=ro', uri=True)
    try:
        uuids = [r[0] for r in ann.execute('SELECT DISTINCT source_uuid FROM annotations WHERE label = ?', (label,))]
    except sqlite3.OperationalError:  # no annotations table yet
        uuids = []
    finally:
        ann.close()
    if not uuids:
        return {}
    sci = sqlite3.connect(f'file:{science_path}?mode=ro', uri=True)
    try:
        out = {}
        for i in range(0, len(uuids), 500):
            chunk = uuids[i : i + 500]
            marks = ','.join('?' * len(chunk))
            for uuid, ra, dec in sci.execute(
                f'SELECT source_uuid, ra, dec FROM sources WHERE source_uuid IN ({marks})', chunk
            ):
                if ra is not None and dec is not None:
                    out[uuid] = (float(ra), float(dec))
        return out
    finally:
        sci.close()


def run_one(source_uuid: str, ra: float, dec: float, data_dir: Path = results.ZTFORCE_DIR) -> dict:
    """Photometer one source and summarise it for storage. Raises on failure."""
    from ztforce import build_config, run_forced_photometry

    data_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        # The unmeasured-epoch warning is summarised in the stored note instead.
        warnings.simplefilter('ignore')
        lcs = run_forced_photometry(
            ra, dec, bands=results.BANDS, data_dir=data_dir, config=build_config(), show_progress=False
        )
    return results.summarise(lcs)


class Worker:
    def __init__(
        self,
        store_path: Path,
        annotations_path: Path,
        science_path: Path,
        label: str,
        concurrency: int = 2,
        poll_seconds: float = POLL_SECONDS,
    ) -> None:
        self.store_path = store_path
        self.annotations_path = annotations_path
        self.science_path = science_path
        self.label = label
        self.concurrency = max(1, concurrency)
        self.poll_seconds = poll_seconds
        self.wake = threading.Event()
        self._stop = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix='ztforce-run')
        self._active: dict[str, Future] = {}
        self._thread: threading.Thread | None = None
        self.version = results.result_version()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        global _current
        con = store.connect(self.store_path)
        try:
            n = store.recover(con)
        finally:
            con.close()
        if n:
            log.info('ztforce worker: re-queued %d runs left running by a previous worker', n)
        self._thread = threading.Thread(target=self._loop, name='ztforce-worker', daemon=True)
        self._thread.start()
        _current = self

    def stop(self) -> None:
        """Stop taking work. Runs in progress are abandoned and re-queued on the next start."""
        global _current
        self._stop.set()
        self.wake.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._pool.shutdown(wait=False, cancel_futures=True)
        if _current is self:
            _current = None

    # ── loop ──────────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # keep the worker alive through a bad tick
                log.exception('ztforce worker tick failed')
            self.wake.wait(self.poll_seconds)
            self.wake.clear()

    def tick(self) -> None:
        """One pass: sync with the labels, then start work if there is room."""
        self._active = {u: f for u, f in self._active.items() if not f.done()}
        labelled = labelled_sources(self.annotations_path, self.science_path, self.label)
        con = store.connect(self.store_path)
        try:
            store.sync_labels(con, labelled, self.version)
            if store.is_paused(con):
                return
            for row in store.claim(con, self.concurrency - len(self._active)):
                uuid = row['source_uuid']
                self._active[uuid] = self._pool.submit(self._process, uuid, row['ra'], row['dec'])
        finally:
            con.close()

    def _process(self, source_uuid: str, ra: float, dec: float) -> None:
        try:
            summary = run_one(source_uuid, ra, dec)
        except Exception as exc:  # noqa: BLE001 - stored for the page to show
            log.exception('ztforce run failed for %s', source_uuid)
            con = store.connect(self.store_path)
            try:
                store.fail(con, source_uuid, f'{type(exc).__name__}: {exc}')
            finally:
                con.close()
            return
        con = store.connect(self.store_path)
        try:
            store.finish(
                con,
                source_uuid,
                self.version,
                results.pack(summary['stacks']),
                summary['n_epochs'],
                summary['note'],
            )
        finally:
            con.close()
        self.wake.set()  # a slot is free: start the next source now
