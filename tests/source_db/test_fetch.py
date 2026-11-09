"""Retry and concurrency behaviour of `source_db.cutouts.fetch`, network stubbed."""
from __future__ import annotations

import http.client
import io
import threading
import time
import urllib.error

import pytest

from source_db.cutouts import fetch

FITS = b'SIMPLE  =' + b' ' * (2880 - 9)


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code):
    return urllib.error.HTTPError('http://x', code, 'err', {}, io.BytesIO(b''))


@pytest.fixture
def stub(monkeypatch):
    """Serve the given outcomes in order; record how many calls were made."""
    calls = []
    monkeypatch.setattr(fetch.time, 'sleep', lambda s: None)

    def install(*outcomes):
        seq = list(outcomes)

        def urlopen(req, timeout):
            calls.append(req.full_url)
            o = seq.pop(0) if len(seq) > 1 else seq[0]
            if isinstance(o, BaseException):
                raise o
            return _Resp(o)
        monkeypatch.setattr(fetch.urllib.request, 'urlopen', urlopen)
        return calls
    return install


@pytest.mark.parametrize('err', [
    _http_error(503), _http_error(502), _http_error(504), _http_error(429), _http_error(500),
    ConnectionResetError(), http.client.IncompleteRead(b''), http.client.RemoteDisconnected(),
    TimeoutError(), urllib.error.URLError('refused'),
])
def test_transient_failures_are_retried(stub, err):
    calls = stub(err, FITS)
    assert fetch.get('http://h/a') == FITS
    assert len(calls) == 2


@pytest.mark.parametrize('code', [400, 404])
def test_missing_is_none_without_retrying(stub, code):
    calls = stub(_http_error(code))
    assert fetch.get('http://h/a') is None
    assert len(calls) == 1


def test_auth_errors_raise_without_retrying(stub):
    calls = stub(_http_error(403))
    with pytest.raises(urllib.error.HTTPError):
        fetch.get('http://h/a')
    assert len(calls) == 1


def test_an_invalid_body_is_retried(stub):
    calls = stub(b'<html>busy</html>', FITS[:1000], FITS)
    assert fetch.get('http://h/a', validate=fetch.is_fits) == FITS
    assert len(calls) == 3


def test_gives_up_as_a_urlerror_after_the_attempts(stub):
    calls = stub(_http_error(503))
    with pytest.raises(urllib.error.URLError) as exc:
        fetch.get('http://h/a', policy=fetch.Policy(attempts=3, budget_s=1e9))
    assert isinstance(exc.value, fetch.FetchError)
    assert len(calls) == 3


def test_no_retry_starts_past_the_budget(stub):
    calls = stub(_http_error(503))
    with pytest.raises(fetch.FetchError):
        fetch.get('http://h/a', policy=fetch.Policy(attempts=50, budget_s=0.1, base_delay_s=1))
    assert len(calls) == 1


def test_in_flight_requests_are_capped_per_host(monkeypatch):
    monkeypatch.setitem(fetch.HOST_LIMITS, 'capped.example', 2)
    monkeypatch.setattr(fetch, '_sems', {})
    live = peak = 0
    lock = threading.Lock()

    def urlopen(req, timeout):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.05)
        with lock:
            live -= 1
        return _Resp(FITS)
    monkeypatch.setattr(fetch.urllib.request, 'urlopen', urlopen)
    threads = [threading.Thread(target=fetch.get, args=(f'http://capped.example/{i}',))
               for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 2


def test_is_fits():
    assert fetch.is_fits(FITS)
    assert not fetch.is_fits(FITS[:-1])
    assert not fetch.is_fits(b'<html>' + b' ' * 2874)


def _count_in_flight(monkeypatch, host, limit, hold_s=0.05):
    """Stub urlopen to track peak in-flight requests to `host`."""
    monkeypatch.setitem(fetch.HOST_LIMITS, host, limit)
    monkeypatch.setattr(fetch, '_sems', {})
    state = {'live': 0, 'peak': 0}
    lock = threading.Lock()

    def urlopen(req, timeout):
        with lock:
            state['live'] += 1
            state['peak'] = max(state['peak'], state['live'])
        time.sleep(hold_s)
        with lock:
            state['live'] -= 1
        return _Resp(FITS)
    monkeypatch.setattr(fetch.urllib.request, 'urlopen', urlopen)
    return state


def _in_threads(n, target):
    threads = [threading.Thread(target=target, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_background_work_holds_only_half_the_slots(monkeypatch):
    state = _count_in_flight(monkeypatch, 'bg.example', 4)

    def prefetch(i):
        with fetch.priority(background=True):
            fetch.get(f'http://bg.example/{i}')
    _in_threads(8, prefetch)
    assert state['peak'] == 2


def test_foreground_keeps_headroom_while_background_is_busy(monkeypatch):
    state = _count_in_flight(monkeypatch, 'mix.example', 4, hold_s=0.2)

    def either(i):
        with fetch.priority(background=i < 6):
            fetch.get(f'http://mix.example/{i}')
    _in_threads(8, either)
    assert state['peak'] == 4             # 2 background + 2 foreground


def test_a_promoted_request_stops_waiting_for_a_background_slot(monkeypatch):
    monkeypatch.setitem(fetch.HOST_LIMITS, 'p.example', 2)   # one background slot
    monkeypatch.setattr(fetch, '_sems', {})
    monkeypatch.setattr(fetch.urllib.request, 'urlopen', lambda req, timeout: _Resp(FITS))
    bg = fetch._sem('p.example', background=True)
    bg.acquire()                          # the only background slot is taken
    got = []

    def queued():
        with fetch.priority(background=True) as p:
            prio.append(p)
            got.append(fetch.get('http://p.example/x'))
    prio = []
    t = threading.Thread(target=queued)
    t.start()
    time.sleep(0.2)
    assert not got                        # stuck behind the background share
    prio[0].background = False            # a foreground request joined it
    t.join(timeout=2)
    bg.release()
    assert got == [FITS]
