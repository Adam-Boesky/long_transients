"""HTTP fetching for the cutout services, with retries and a per-host cap.

IRSA's IBE answers bursts with instant nginx 503s (measured Nov 2026: ~8% of
serial requests, 70-90% at 8 in flight), and any request can also die to a
reset, a truncated body or a stall. All of these clear on a retry, so:

  - 429/5xx, connection errors, timeouts and bodies failing `validate` are
    retried with jittered exponential backoff inside a time budget;
  - 400/404 return None at once (a missing band is ordinary);
  - at most `HOST_LIMITS[host]` requests are in flight per host, released
    while backing off.

Stdlib only: `requests` is not in the serving requirements.
"""
from __future__ import annotations

import http.client
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

RETRY_STATUS = {429, 500, 502, 503, 504}
MISSING_STATUS = {400, 404}

# In-flight requests per host. Loose on purpose until IRSA's limit is known
# on a day it is not degraded.
HOST_LIMITS = {'irsa.ipac.caltech.edu': 4}
DEFAULT_HOST_LIMIT = 8


@dataclass(frozen=True)
class Policy:
    attempts: int
    budget_s: float          # no new retry starts once this much time has passed
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0


# A page request someone is waiting on, and a batch job that can wait.
INTERACTIVE = Policy(attempts=5, budget_s=20)
BATCH = Policy(attempts=10, budget_s=600, base_delay_s=2, max_delay_s=120)
DEFAULT_POLICY = INTERACTIVE


class FetchError(urllib.error.URLError):
    """Gave up after retries. A URLError, so existing handlers map it to 502."""


class InvalidBody(Exception):
    pass


_sems: dict[str, threading.BoundedSemaphore] = {}
_sems_lock = threading.Lock()


def _sem(host: str) -> threading.BoundedSemaphore:
    with _sems_lock:
        if host not in _sems:
            _sems[host] = threading.BoundedSemaphore(HOST_LIMITS.get(host, DEFAULT_HOST_LIMIT))
        return _sems[host]


def is_fits(raw: bytes) -> bool:
    """A complete FITS file: the SIMPLE card and whole 2880-byte blocks."""
    return raw[:9] == b'SIMPLE  =' and len(raw) % 2880 == 0


def get(url: str, headers: Optional[dict] = None, *, timeout: float = 60,
        validate: Optional[Callable[[bytes], bool]] = None,
        policy: Optional[Policy] = None) -> Optional[bytes]:
    """The body, or None for 400/404. Raises FetchError once retries run out.

    `timeout` bounds the connect and each socket read, so it catches a stalled
    transfer. Other HTTP errors (401, 403) raise unretried.
    """
    policy = policy or DEFAULT_POLICY
    req = urllib.request.Request(url)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    sem = _sem(urllib.parse.urlsplit(url).hostname or '')
    t0 = time.monotonic()
    for attempt in range(policy.attempts):
        retry_after = None
        try:
            with sem:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read()
            if validate is not None and not validate(raw):
                raise InvalidBody(f'{len(raw)} B body failed validation: {raw[:60]!r}')
            return raw
        except urllib.error.HTTPError as e:
            if e.code in MISSING_STATUS:
                return None
            if e.code not in RETRY_STATUS:
                raise
            retry_after = _retry_after(e)
            e.close()                # free the socket before backing off
            err: Exception = e
        except (urllib.error.URLError, http.client.HTTPException, OSError, InvalidBody) as e:
            # OSError covers resets and TimeoutError; HTTPException covers
            # IncompleteRead and RemoteDisconnected.
            err = e
        delay = min(policy.max_delay_s, policy.base_delay_s * 2 ** attempt) * random.uniform(0.5, 1.5)
        if retry_after is not None:
            delay = max(delay, min(retry_after, policy.max_delay_s))
        if attempt == policy.attempts - 1 or time.monotonic() - t0 + delay > policy.budget_s:
            break
        time.sleep(delay)
    raise FetchError(f'{type(err).__name__} after {attempt + 1} attempts '
                     f'({time.monotonic() - t0:.1f} s): {err}; {url}')


def _retry_after(e: urllib.error.HTTPError) -> Optional[float]:
    try:
        return float(e.headers.get('Retry-After'))
    except (TypeError, ValueError):
        return None
