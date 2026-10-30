"""Grouping for the ~300-column catalog row.

The detail page renders `catalog` as sections, so the grouping must cover
every column exactly once and stay stable across rebuilds.
"""
from __future__ import annotations

from api import catalog_schema as CS


# ── the rules ─────────────────────────────────────────────────────────────────

def test_known_prefixes_map_to_their_group():
    cases = {
        'ZTF_mag_g': 'ztf',
        'PSTARR_PanSTARR_ID': 'panstarrs',
        'PanSTARR_ID': 'panstarrs',
        'wise_w1mpro': 'wise',
        'simbad_main_id': 'simbad',
        'sdss_class': 'sdss',
        'tns_name': 'tns',
        'desi_targetid': 'desi',
        'agn_db_type': 'agn',
    }
    for col, group in cases.items():
        assert CS.group_of(col) == group, col


def test_unprefixed_extraction_columns_fall_back_to_detection():
    for col in ('g_x', 'r_Catalog_Flag', 'fieldid', 'ccdid', 'qid',
                'i_association_separation_arcsec', 'filter_info'):
        assert CS.group_of(col) == 'detection', col


def test_provenance_is_separated_from_science():
    assert CS.group_of('catalog_tier') == 'provenance'
    assert CS.group_of('row_hash') == 'provenance'


def test_panstarrs_prefix_is_matched_before_looser_rules():
    """Both PSTARR_ and PanSTARR_ must land in one group, not two."""
    assert CS.group_of('PSTARR_gMeanPSFMag') == CS.group_of('PanSTARR_ID')


# ── the schema ────────────────────────────────────────────────────────────────

def _cols():
    return ['ZTF_a', 'ZTF_b', 'PSTARR_c', 'wise_d', 'simbad_e', 'sdss_f',
            'tns_g', 'desi_h', 'agn_i', 'g_x', 'fieldid',
            'catalog_tier', 'row_hash', 'ra', 'dec', 'source_uuid']


def test_schema_covers_every_column_exactly_once():
    sch = CS.build_schema(_cols())
    listed = [c for g in sch['groups'] for c in g['columns']]
    expected = [c for c in _cols() if c not in CS.PROMOTED]
    assert sorted(listed) == sorted(expected)
    assert len(listed) == len(set(listed)), 'a column must not appear twice'
    assert sch['n_columns'] == len(expected)


def test_schema_omits_promoted_columns():
    """ra/dec/uuid are fields on the source, not rows in the catalog table."""
    sch = CS.build_schema(_cols())
    listed = {c for g in sch['groups'] for c in g['columns']}
    assert not (listed & CS.PROMOTED)


def test_schema_groups_are_ordered_and_labelled():
    sch = CS.build_schema(_cols())
    keys = [g['key'] for g in sch['groups']]
    assert keys.index('ztf') < keys.index('detection') < keys.index('provenance')
    assert all(g['label'] for g in sch['groups'])
    assert next(g for g in sch['groups'] if g['key'] == 'ztf')['n_columns'] == 2


def test_schema_skips_groups_with_no_columns():
    sch = CS.build_schema(['ZTF_a', 'ra'])
    assert [g['key'] for g in sch['groups']] == ['ztf']


def test_empty_input_yields_an_empty_schema():
    assert CS.build_schema([]) == {'groups': [], 'n_columns': 0}


# ── the endpoint ──────────────────────────────────────────────────────────────

def test_endpoint_returns_a_schema_matching_the_detail_payload(client):
    from tests.conftest import UUIDS
    sch = client.get('/api/catalog/schema').json()
    catalog = client.get(f'/api/sources/{UUIDS[0]}').json()['catalog']
    listed = {c for g in sch['groups'] for c in g['columns']}
    assert listed == set(catalog), (
        'the schema must describe exactly the keys the detail endpoint returns')


def test_endpoint_reports_column_counts(client):
    sch = client.get('/api/catalog/schema').json()
    assert sch['n_columns'] == sum(g['n_columns'] for g in sch['groups'])
    assert sch['n_columns'] > 0
