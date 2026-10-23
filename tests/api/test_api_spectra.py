"""The spectra tier: fetch identifiers, payload encoding, and the endpoint."""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from source_db.serving import payload as P
from source_db.spectra import _base as spec_base
from source_db.spectra import sdss as spec_sdss
from tests.conftest import NO_SPEC_UUID, SPEC_UUIDS, UUIDS


# ── SDSS URL construction (offline, no network) ───────────────────────────────

def test_sdss_spectrum_url_is_built_from_identifiers():
    row = pd.Series({'sdss_plate': 6740, 'sdss_mjd': 56401,
                     'sdss_fiberid': 518, 'sdss_run2d': 'v5_13_2'})
    url = spec_sdss.spectrum_url(row)
    assert url.endswith('/v5_13_2/spectra/lite/6740/spec-6740-56401-0518.fits')
    # plate and fiber are zero-padded to 4; mjd is not
    assert 'spec-6740-56401-0518' in url


def test_sdss_spectrum_url_is_none_when_identifiers_are_missing():
    row = pd.Series({'sdss_plate': None, 'sdss_mjd': 56401,
                     'sdss_fiberid': 518, 'sdss_run2d': 'v5_13_2'})
    assert spec_sdss.spectrum_url(row) is None


# ── payload encoding ──────────────────────────────────────────────────────────

def _one(model=True, n=20):
    w = np.linspace(3600.0, 9200.0, n)
    return pd.DataFrame([spec_base.spectrum_row(
        source_uuid='u', survey='sdss', spec_id='1-2-3', wavelength=w,
        flux=np.linspace(1.0, 2.0, n),
        model=np.linspace(1.1, 2.1, n) if model else None,
        spec_class='QSO', redshift=1.5)])


def test_spectrum_payload_round_trips():
    blob, n, surveys = P.build_spectrum_payload(_one(), 'u')
    doc = P.decode_payload(blob)
    assert n == 1 and surveys == ['sdss']
    s = doc['spectra'][0]
    assert s['spec_class'] == 'QSO' and s['redshift'] == 1.5
    assert s['n_pix'] == 20 and len(s['flux']) == 20


# ── the analytic wavelength grid ──────────────────────────────────────────────

def test_regular_grid_is_replaced_by_start_and_step():
    doc = P.decode_payload(P.build_spectrum_payload(_one(), 'u')[0])
    s = doc['spectra'][0]
    assert s['wave_scale'] == 'linear'
    assert 'wavelength' not in s, 'a regular grid should not ship the array'
    rebuilt = [s['wave_start'] + i * s['wave_step'] for i in range(s['n_pix'])]
    assert rebuilt[0] == 3600.0
    assert abs(rebuilt[-1] - 9200.0) < 1e-6


def test_log_grid_is_detected():
    n = 30
    loglam = np.linspace(np.log10(3600.0), np.log10(9200.0), n)
    df = pd.DataFrame([spec_base.spectrum_row(
        source_uuid='u', survey='sdss', spec_id='1-2-3',
        wavelength=10 ** loglam, flux=np.ones(n))])
    s = P.decode_payload(P.build_spectrum_payload(df, 'u')[0])['spectra'][0]
    assert s['wave_scale'] == 'log10'
    rebuilt = 10 ** (s['wave_start'] + np.arange(s['n_pix']) * s['wave_step'])
    assert np.abs(rebuilt - 10 ** loglam).max() < 1e-2


def test_irregular_grid_falls_back_to_the_explicit_array():
    """Never silently resample: an uneven grid must ship its wavelengths."""
    w = np.array([3600.0, 3601.0, 3610.0, 3611.0, 3650.0])
    df = pd.DataFrame([spec_base.spectrum_row(
        source_uuid='u', survey='sdss', spec_id='1-2-3',
        wavelength=w, flux=np.ones(w.size))])
    s = P.decode_payload(P.build_spectrum_payload(df, 'u')[0])['spectra'][0]
    assert s['wave_scale'] is None
    assert s['wavelength'] == [3600.0, 3601.0, 3610.0, 3611.0, 3650.0]


def test_fit_grid_returns_none_for_irregular_input():
    assert P.fit_grid(np.array([1.0, 2.0, 10.0, 11.0])) is None
    assert P.fit_grid(np.array([1.0])) is None


def test_spectrum_payload_is_gzip():
    blob, _, _ = P.build_spectrum_payload(_one(), 'u')
    assert blob[:2] == b'\x1f\x8b'


def test_spectrum_payload_rounds_to_configured_precision():
    doc = P.decode_payload(P.build_spectrum_payload(_one(), 'u')[0])
    for v in doc['spectra'][0]['flux']:
        assert round(v, P.SPECTRUM_FLUX_DP) == v


def test_spectrum_payload_keeps_a_null_model():
    doc = P.decode_payload(P.build_spectrum_payload(_one(model=False), 'u')[0])
    assert doc['spectra'][0]['model'] is None, (
        'a missing pipeline fit must stay null, not become zeros')


def test_spectrum_payload_of_nothing_is_empty():
    blob, n, surveys = P.build_spectrum_payload(spec_base.empty_spectra(), 'u')
    assert n == 0 and surveys == []
    assert P.decode_payload(blob)['spectra'] == []


# ── the built database ────────────────────────────────────────────────────────

def test_only_sources_with_spectra_get_rows(built_db):
    con = sqlite3.connect(built_db)
    got = {r[0] for r in con.execute('SELECT source_uuid FROM spectra')}
    assert got == set(SPEC_UUIDS)
    assert NO_SPEC_UUID not in got


def test_spectrum_summary_is_denormalised_onto_sources(built_db):
    con = sqlite3.connect(built_db)
    rows = dict((r[0], (r[1], r[2])) for r in con.execute(
        'SELECT source_uuid, n_spectra, spec_surveys FROM sources'))
    # UUIDS[0] has both an SDSS and a DESI spectrum.
    assert rows[UUIDS[0]] == (2, 'desi,sdss')
    assert rows[UUIDS[1]] == (1, 'desi')
    assert rows[NO_SPEC_UUID] == (0, None)


def test_spectra_meta_is_recorded(built_db):
    con = sqlite3.connect(built_db)
    meta = dict(con.execute('SELECT key, value FROM meta'))
    assert int(meta['n_sources_with_spectra']) == len(SPEC_UUIDS)
    assert int(meta['n_spectra']) == 3


# ── the endpoint ──────────────────────────────────────────────────────────────

def test_spectra_endpoint_returns_both_surveys(client):
    doc = client.get(f'/api/sources/{UUIDS[0]}/spectra').json()
    assert doc['n_spectra'] == 2
    assert sorted(s['survey'] for s in doc['spectra']) == ['desi', 'sdss']


def test_spectra_endpoint_is_gzipped(client):
    r = client.get(f'/api/sources/{UUIDS[0]}/spectra',
                   headers={'accept-encoding': 'gzip'})
    assert r.status_code == 200
    assert r.headers.get('content-encoding') == 'gzip'


def test_spectra_endpoint_decompresses_when_client_cannot_gzip(client):
    r = client.get(f'/api/sources/{UUIDS[0]}/spectra',
                   headers={'accept-encoding': 'identity'})
    assert r.headers.get('content-encoding') != 'gzip'
    assert json.loads(r.content)['n_spectra'] == 2


def test_source_without_spectra_gets_an_empty_document(client):
    r = client.get(f'/api/sources/{NO_SPEC_UUID}/spectra')
    assert r.status_code == 200
    assert r.json() == {'source_uuid': NO_SPEC_UUID, 'n_spectra': 0, 'spectra': []}


def test_spectra_404s_for_unknown_source(client):
    assert client.get('/api/sources/not-a-uuid/spectra').status_code == 404


def test_list_exposes_spectrum_summary(client):
    row = next(i for i in client.get('/api/sources').json()['items']
               if i['source_uuid'] == UUIDS[0])
    assert row['n_spectra'] == 2
    assert row['spec_surveys'] == ['desi', 'sdss']


def test_list_can_filter_to_sources_with_spectra(client):
    ids = {i['source_uuid'] for i in
           client.get('/api/sources?has_spectrum=true').json()['items']}
    assert ids == set(SPEC_UUIDS)


def test_detail_exposes_spectrum_summary(client):
    d = client.get(f'/api/sources/{UUIDS[1]}').json()
    assert d['n_spectra'] == 1 and d['spec_surveys'] == ['desi']


def test_stats_counts_spectra(client):
    s = client.get('/api/stats').json()
    assert s['n_with_spectra'] == len(SPEC_UUIDS)
    assert s['n_spectra'] == 3
