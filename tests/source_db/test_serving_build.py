"""The build behind the API: payload encoding and the serving database.

These guard the contract the API depends on, so a build regression fails here
rather than as a confusing 500 in an endpoint test.
"""
from __future__ import annotations

import gzip
import json
import sqlite3

import numpy as np
import pandas as pd

from source_db.lightcurves import _base
from source_db.serving import payload as P
from tests.conftest import LC_UUIDS, NO_LC_UUID, UUIDS


# ── payload encoding ──────────────────────────────────────────────────────────

def _frame(**over) -> pd.DataFrame:
    base = {'source_uuid': 'u', 'survey': 'ztf', 'band': 'g',
            'mjd': [3.0, 1.0, 2.0], 'mag': [20.0, 21.0, 22.0],
            'magerr': [0.1, 0.1, 0.1], 'snr': [np.nan] * 3,
            'catalog_id': 'x', 'catflags': [0, 0, 6141]}
    base.update(over)
    return pd.DataFrame(base)


def test_payload_round_trips():
    blob, n, surveys = P.build_payload(_frame(), 'u')
    doc = P.decode_payload(blob)
    assert n == 3 and surveys == ['ztf']
    assert doc['source_uuid'] == 'u'
    assert doc['surveys']['ztf']['g']['mag'] == [21.0, 22.0, 20.0]


def test_payload_sorts_epochs_by_time():
    doc = P.decode_payload(P.build_payload(_frame(), 'u')[0])
    assert doc['surveys']['ztf']['g']['mjd'] == [1.0, 2.0, 3.0]


def test_payload_is_gzip():
    blob, _, _ = P.build_payload(_frame(), 'u')
    assert blob[:2] == b'\x1f\x8b', 'must be gzip so the API can pass it through'
    json.loads(gzip.decompress(blob))


def test_payload_drops_all_null_columns():
    """snr is null for ZTF; shipping a column of nulls is wasted bytes."""
    doc = P.decode_payload(P.build_payload(_frame(), 'u')[0])
    assert 'snr' not in doc['surveys']['ztf']['g']


def test_payload_keeps_reduction_inputs():
    doc = P.decode_payload(P.build_payload(_frame(), 'u')[0])
    assert doc['surveys']['ztf']['g']['catflags'] == [0, 6141, 0]


def test_payload_converts_nan_to_null():
    doc = P.decode_payload(P.build_payload(
        _frame(mag=[20.0, np.nan, 22.0]), 'u')[0])
    assert None in doc['surveys']['ztf']['g']['mag'], 'NaN is not valid JSON'


def test_empty_frame_yields_an_empty_payload():
    blob, n, surveys = P.build_payload(_base.empty_lightcurves(), 'u')
    assert n == 0 and surveys == []
    assert P.decode_payload(blob)['surveys'] == {}


def test_to_long_frame_inverts_the_payload():
    doc = P.decode_payload(P.build_payload(_frame(), 'u')[0])
    back = P.to_long_frame(doc)
    assert len(back) == 3
    assert sorted(back['mjd']) == [1.0, 2.0, 3.0]
    assert set(back['survey']) == {'ztf'}


# ── the built database ────────────────────────────────────────────────────────

def test_every_source_is_present(built_db):
    con = sqlite3.connect(built_db)
    got = {r[0] for r in con.execute('SELECT source_uuid FROM sources')}
    assert got == set(UUIDS)


def test_only_sources_with_epochs_get_payloads(built_db):
    con = sqlite3.connect(built_db)
    got = {r[0] for r in con.execute('SELECT source_uuid FROM light_curves')}
    assert got == set(LC_UUIDS)
    assert NO_LC_UUID not in got


def test_summary_columns_are_denormalised_onto_sources(built_db):
    con = sqlite3.connect(built_db)
    for uuid, n_epochs, surveys in con.execute(
            'SELECT source_uuid, n_epochs, surveys FROM sources'):
        if uuid == NO_LC_UUID:
            assert n_epochs == 0 and surveys is None
            continue
        row = con.execute('SELECT n_epochs, surveys FROM light_curves '
                          'WHERE source_uuid=?', (uuid,)).fetchone()
        assert (n_epochs, surveys) == row, (
            'the list view sorts on these, so they must match the payload')


def test_catalog_json_holds_the_unpromoted_columns(built_db):
    con = sqlite3.connect(built_db)
    blob = con.execute('SELECT catalog_json FROM sources WHERE source_uuid=?',
                       (UUIDS[0],)).fetchone()[0]
    cat = json.loads(blob)
    assert cat['some_float'] == 1.5
    assert 'ra' not in cat and 'coord_slug' not in cat


def test_epochs_index_exists(built_db):
    """Without it, ORDER BY n_epochs scans every row (~140 ms at full size)."""
    con = sqlite3.connect(built_db)
    names = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='index'")}
    assert 'idx_sources_epochs' in names


def test_meta_records_the_build(built_db):
    con = sqlite3.connect(built_db)
    meta = dict(con.execute('SELECT key, value FROM meta'))
    assert int(meta['n_sources']) == len(UUIDS)
    assert int(meta['n_sources_with_lc']) == len(LC_UUIDS)
    assert meta['built_at']


# ── bucketing ─────────────────────────────────────────────────────────────────

def test_bucketing_splits_sources_across_files(tmp_path):
    df = pd.DataFrame({
        'source_uuid': ['00aaaaaa', 'ffbbbbbb', '00cccccc'],
        'survey': 'ztf', 'band': 'g', 'mjd': [1.0, 2.0, 3.0],
        'mag': [20.0, 20.0, 20.0], 'magerr': [0.1, 0.1, 0.1],
        'snr': [np.nan] * 3, 'catalog_id': 'x',
    })
    with _base.BucketedWriter(tmp_path) as w:
        w.write(df)
    buckets = sorted(p.parent.name for p in
                     _base.lightcurve_dir(tmp_path).glob('uuid_prefix=*/part-0.parquet'))
    assert buckets == ['uuid_prefix=00', 'uuid_prefix=ff']
    # The two '00' sources must share one file.
    got = pd.read_parquet(_base.bucket_path('00', tmp_path))
    assert set(got['source_uuid']) == {'00aaaaaa', '00cccccc'}


def test_buckets_share_one_schema(tmp_path):
    """Surveys emit different extras; the bucket files must still align."""
    ztf = pd.DataFrame({'source_uuid': ['00a'], 'survey': ['ztf'], 'band': ['g'],
                        'mjd': [1.0], 'mag': [20.0], 'magerr': [0.1],
                        'snr': [np.nan], 'catalog_id': ['x'], 'catflags': [0],
                        'clrcoeff': [0.1]})
    sdss = pd.DataFrame({'source_uuid': ['00b'], 'survey': ['sdss'], 'band': ['u'],
                         'mjd': [1.0], 'mag': [20.0], 'magerr': [0.1],
                         'snr': [np.nan], 'catalog_id': ['y'], 'psf_mag': [20.5],
                         'psf_magerr': [0.06]})
    with _base.BucketedWriter(tmp_path) as w:
        w.write(ztf)
        w.write(sdss)
    got = pd.read_parquet(_base.bucket_path('00', tmp_path))
    assert len(got) == 2
    # The ZTF row has no psf_mag and the SDSS row no catflags; both are null.
    assert got.loc[got.survey == 'ztf', 'psf_mag'].isna().all()
    assert got.loc[got.survey == 'sdss', 'catflags'].isna().all()


def test_bucket_of_matches_the_written_layout():
    assert _base.bucket_of('ab123456-0000') == 'ab'
    assert len(_base.bucket_of('ab123456-0000')) == _base.BUCKET_CHARS


def test_sdss_spectra_of_a_former_match_are_not_published():
    from source_db.serving.build_serving_db import current_sdss_spectra
    spec = pd.DataFrame({'source_uuid': ['a', 'a', 'a', 'b', 'b'],
                         'survey': ['sdss', 'sdss', 'sdss', 'sdss', 'desi'],
                         'spec_id': ['111', '112', '222', '333', '444']})
    kept = current_sdss_spectra(spec, {'a|111', 'a|112'})
    # Both of a's current spectra stay; its old one and b's go; DESI is untouched.
    assert kept['spec_id'].tolist() == ['111', '112', '444']
