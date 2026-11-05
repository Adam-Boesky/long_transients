"""Single-epoch exposure lookup and its endpoints.

The network is monkeypatched, as in `test_cutouts.py`. The ZTF example is a
real epoch: objectid 876314200025364 at hmjd 59189.10434 is IRSA exposure
ztf_20201206102060_000876_zi_c14_o_q2, starting at JD 2459189.6021065.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import db as api_db
from api.main import app
from source_db.cutouts import epochs, render

UUID = 'cccc3333-0000-4000-8000-000000000003'


@pytest.fixture(autouse=True)
def _fresh_caches():
    epochs._find.cache_clear()
    epochs._night_exposures.cache_clear()
    yield


def test_objectid_layout():
    assert epochs.parse_objectid('876314200025364') == (876, 3, 14, 2)
    # Four-digit fields keep the same tail layout.
    assert epochs.parse_objectid('1523201400000001') == (1523, 2, 1, 4)
    assert epochs.parse_objectid('not-an-id') is None


def test_sci_path_matches_irsa_naming():
    row = {'filefracday': '20201206102060', 'field': 876, 'filtercode': 'zi',
           'ccdid': 14, 'imgtypecode': 'o', 'qid': 2}
    assert epochs.ztf_sci_path(row) == (
        '2020/1206/102060/ztf_20201206102060_000876_zi_c14_o_q2_sciimg.fits')


def test_hmjd_converts_to_the_exposure_midpoint():
    jd = epochs.ztf_mid_exposure_jd(59189.10434, 314.86217, 87.26282)
    midpoint = 2459189.6021065 + 15 / 86400
    assert abs(jd - midpoint) * 86400 < 2      # seconds


def test_listing_names_the_exposure_from_the_night(monkeypatch):
    listing = b'<a href="101597/">101597/</a><a href="102060/">102060/</a>'
    seen = []

    def fake_get(url, headers=None):           # noqa: ANN001
        seen.append(url)
        return listing if url.endswith('/2020/1206/') else b'FITS'
    monkeypatch.setattr(render, '_get', fake_get)
    jd = epochs.ztf_mid_exposure_jd(59189.10434, 314.86217, 87.26282)
    exp = epochs._find_ztf_by_listing('i', 314.86217, 87.26282, jd, (876, 14, 2))
    # The later of the two directories before the midpoint, not the earlier.
    assert exp.image_id == 'ztf_20201206102060_000876_zi_c14_o_q2'
    assert any('size=3arcsec' in u for u in seen)    # existence was checked


def test_listing_gives_up_when_the_pointing_has_no_such_file(monkeypatch):
    listing = b'<a href="102060/">102060/</a>'
    monkeypatch.setattr(render, '_get',
                        lambda url, headers=None: listing if url.endswith('/') else None)
    jd = epochs.ztf_mid_exposure_jd(59189.10434, 314.86217, 87.26282)
    assert epochs._find_ztf_by_listing('i', 314.86217, 87.26282, jd, (500, 1, 1)) is None


def test_ps1_removes_the_obstime_lag_before_matching(monkeypatch):
    # The detection sits 58 s after warp A; warp B is 58 s before it, so the
    # raw nearest is a tie and only the lag correction picks A.
    t = 56549.436784
    a, b = t - 58 / 86400, t + 58 / 86400
    table = ('projcell subcell ra dec filter mjd type filename shortname badflag\n'
             f'795 4 1 2 r {b} warp /x/B.fits B.fits 0\n'
             f'795 4 1 2 r {a} warp /x/A.fits A.fits 0\n').encode()
    monkeypatch.setattr(render, '_get', lambda url, headers=None: table)
    exp = epochs.find_exposure('panstarrs', 'r', 1.0, 2.0, t)
    assert exp.image_id == 'A.fits'
    assert 'format=fits' in exp.cutout_url and 'red=/x/A.fits' in exp.cutout_url


def test_reproject_identity_and_outside_is_nan():
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [10, 10]
    w.wcs.crpix = [5, 5]
    w.wcs.cd = [[1 / 3600, 0], [0, 1 / 3600]]
    img = np.arange(100, dtype=np.float32).reshape(10, 10)
    same = epochs.reproject(img, w, w, (10, 10))
    assert np.allclose(same[:9, :9], img[:9, :9])
    assert np.isnan(epochs.reproject(img, w, w, (12, 12))[11, 11])


# ── endpoints ─────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(render, 'CACHE_DIR', tmp_path / 'cache')
    con = sqlite3.connect(':memory:', check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute('CREATE TABLE sources (source_uuid TEXT PRIMARY KEY, ra REAL,'
                ' dec REAL, catalog_json TEXT)')
    con.execute('CREATE TABLE cutout_sources (source_uuid TEXT, survey TEXT,'
                ' band TEXT, origin TEXT, PRIMARY KEY (source_uuid, survey, band))')
    con.execute('INSERT INTO sources VALUES (?,?,?,?)', (UUID, 10.0, -5.0, '{}'))
    con.execute('INSERT INTO cutout_sources VALUES (?,?,?,?)', (UUID, 'ztf', 'g', '501/15/3'))
    app.dependency_overrides[api_db.get_science] = lambda: con
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    con.close()


EXP = epochs.Exposure(survey='ztf', band='g', image_id='ztf_x', obs_mjd=60000.0,
                      cutout_url='https://irsa/x.fits?c', full_url='https://irsa/x.fits',
                      details={'field': 501})


def test_epoch_uses_the_objectid_pointing(client, monkeypatch):
    got = {}

    def fake_find(survey, band, ra, dec, mjd, pointing=None):   # noqa: ANN001
        got['pointing'] = pointing
        return EXP
    monkeypatch.setattr(epochs, 'find_exposure', fake_find)
    r = client.get(f'/api/sources/{UUID}/epoch/ztf/g',
                   params={'mjd': 60000.01, 'objectid': '876314200025364'})
    assert r.status_code == 200
    body = r.json()
    assert got['pointing'] == (876, 14, 2)
    assert body['image_id'] == 'ztf_x'
    assert body['png'].startswith(f'/api/sources/{UUID}/epoch/ztf/g/image.png?mjd=')
    assert 'objectid=876314200025364' in body['png']


def test_epoch_falls_back_to_the_reference_pointing(client, monkeypatch):
    got = {}
    monkeypatch.setattr(epochs, 'find_exposure',
                        lambda s, b, ra, dec, mjd, p=None: got.setdefault('p', p) and EXP or EXP)
    client.get(f'/api/sources/{UUID}/epoch/ztf/r', params={'mjd': 60000.01})
    # No r-band reference: any ZTF band's pointing is still the source's field.
    assert got['p'] == (501, 15, 3)


def test_epoch_404s_when_nothing_matches(client, monkeypatch):
    monkeypatch.setattr(epochs, 'find_exposure', lambda *a, **k: None)
    r = client.get(f'/api/sources/{UUID}/epoch/panstarrs/r', params={'mjd': 56000.0})
    assert r.status_code == 404
    assert 'no panstarrs r exposure' in r.json()['detail']


@pytest.mark.parametrize('path', ['wise/w1', 'ztf/z', 'sdss/g'])
def test_epoch_rejects_surveys_without_single_exposures(client, path):
    r = client.get(f'/api/sources/{UUID}/epoch/{path}', params={'mjd': 60000.0})
    assert r.status_code == 404


def test_epoch_png_renders_once_then_serves_the_cache(client, monkeypatch):
    calls = []
    monkeypatch.setattr(epochs, 'find_exposure', lambda *a, **k: EXP)

    def fake_fetch(exp, ra, dec, ref_origin=None):              # noqa: ANN001
        calls.append(ref_origin)
        return np.arange(36, dtype=np.float32).reshape(6, 6)
    monkeypatch.setattr(epochs, 'fetch_epoch', fake_fetch)
    url = f'/api/sources/{UUID}/epoch/ztf/g/image.png'
    first = client.get(url, params={'mjd': 60000.01})
    second = client.get(url, params={'mjd': 60000.01})
    assert first.status_code == 200 and first.headers['content-type'] == 'image/png'
    assert second.content == first.content
    assert calls == [(501, 15, 3)]       # resampled onto the g reference; fetched once


def test_service_errors_become_502(client, monkeypatch):
    import urllib.error

    def down(*a, **k):                   # noqa: ANN001
        raise urllib.error.HTTPError('u', 503, 'Service Temporarily Unavailable', {}, None)
    monkeypatch.setattr(epochs, 'find_exposure', down)
    r = client.get(f'/api/sources/{UUID}/epoch/ztf/g', params={'mjd': 60000.0})
    assert r.status_code == 502


def test_payload_ships_ztf_objectids_only():
    import gzip
    import orjson
    import pandas as pd
    from source_db.serving.payload import build_payload
    df = pd.DataFrame({
        'survey': ['ztf', 'ztf', 'panstarrs'], 'band': ['g', 'g', 'r'],
        'mjd': [1.0, 2.0, 3.0], 'mag': [18.0, 18.1, 19.0], 'magerr': [0.1] * 3,
        'snr': [np.nan] * 3,
        'catalog_id': pd.array(['876314200025364', '876314200025364', '1234'], dtype='string'),
    })
    blob, n, _ = build_payload(df, UUID)
    doc = orjson.loads(gzip.decompress(blob))
    # Strings, not numbers: 15 digits is past what a JavaScript double holds exactly.
    assert doc['surveys']['ztf']['g']['catalog_id'] == ['876314200025364'] * 2
    assert 'catalog_id' not in doc['surveys']['panstarrs']['r']
