"""Serving the compiled frontend from the API process.

Deployed there is one process on one port: `npm run build` writes `web/dist`
and `api.main` hands it out next to the data. These cover the ways that
arrangement goes wrong quietly -- a refresh 404ing, a mistyped API call
answered with a page, a stale bundle pinned by a cache header, and `..`
reaching outside the build.

The catch-all is registered at import time only when a build exists, so each
test skips rather than fails on a checkout that has never run `npm run build`.
"""
from __future__ import annotations

import pytest

from api.config import WEB_DIST

INDEX = WEB_DIST / 'index.html'

pytestmark = pytest.mark.skipif(
    not INDEX.is_file(), reason='no web/dist build to serve')


def _is_the_page(res) -> bool:
    return (res.status_code == 200
            and res.headers['content-type'].startswith('text/html'))


@pytest.mark.parametrize('path', ['/', '/table', '/classify', '/progress'])
def test_client_routes_all_serve_the_page(client, path):
    """React owns the URL bar, so every route it knows must return the page.

    Without the fallback only `/` works: opening a shared link to `/table`,
    or just refreshing it, 404s.
    """
    assert _is_the_page(client.get(path))


def test_an_unknown_deep_path_still_serves_the_page(client):
    """The server cannot enumerate client routes, so anything unclaimed is one."""
    assert _is_the_page(client.get('/some/route/added/later'))


def test_an_unknown_api_path_is_404_not_the_page(client):
    """`/api` is the server's own namespace.

    Falling through to the page would answer a mistyped endpoint with 200 and
    a lump of HTML, which surfaces later as a JSON parse error pointing at the
    wrong thing entirely.
    """
    res = client.get('/api/no-such-endpoint')
    assert res.status_code == 404
    assert not res.headers['content-type'].startswith('text/html')


def test_a_real_api_route_is_not_shadowed(client):
    """The catch-all matches every path, so registration order is load-bearing."""
    assert client.get('/api/health').json()['status'] == 'ok'


def test_the_page_is_not_cached_but_hashed_assets_are():
    """index.html names the current bundle; the bundle's name is its hash.

    A cached index.html pins the browser to a bundle that no longer exists --
    a blank page after every deploy, fixed only by a hard refresh.
    """
    from fastapi.testclient import TestClient

    from api.main import app
    # No dependency overrides: these paths never touch a database, and the
    # `client` fixture's lifespan would start the real catalog engine.
    with TestClient(app) as c:
        assert 'no-cache' in c.get('/').headers['cache-control']
        assets = WEB_DIST / 'assets'
        hashed = next((p for p in assets.glob('*.js')), None) if assets.is_dir() else None
        if hashed is None:
            pytest.skip('build has no hashed assets')
        res = c.get(f'/assets/{hashed.name}')
        assert res.status_code == 200
        assert 'immutable' in res.headers['cache-control']


@pytest.mark.parametrize('path', [
    '/../api/config.py',
    '/../../requirements.txt',
    '/%2e%2e%2f%2e%2e%2fapi%2fconfig.py',
])
def test_dot_dot_cannot_escape_the_build(client, path):
    """Serving the repo's own source would be the worst version of this bug."""
    res = client.get(path)
    # Either rejected outright or treated as an unknown client route; what
    # matters is that no file from outside `web/dist` comes back.
    assert _is_the_page(res) or res.status_code in (307, 400, 404)
    assert b'LT_SCIENCE_DB' not in res.content
    assert b'fastapi==' not in res.content
