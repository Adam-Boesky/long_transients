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
