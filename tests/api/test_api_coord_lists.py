"""Uploaded coordinate lists, and the `coord_list=` filter on the list endpoints."""
import sqlite3

import orjson
import pandas as pd
import pytest

# Six sources at (10 + i, -5 + i) deg: two in_both, two in_ztf, one in_ztf_wide, one with no tier
TIERS = ['0', '0', '1', '1', '1_wide_association', None]


def _uuid(i):
    return f'{i:08d}-0000-4000-8000-000000000000'


@pytest.fixture
def coord_client(tmp_path):
    from fastapi.testclient import TestClient
    from api import catalog_filters as api_filters
    from api import coord_lists
    from api import db as api_db
    from api.main import app

    con = sqlite3.connect(':memory:', check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute('CREATE TABLE sources (source_uuid TEXT PRIMARY KEY, coord_slug TEXT,'
                ' ra REAL, dec REAL, n_epochs INT, surveys TEXT, n_spectra INT,'
                ' spec_surveys TEXT, catalog_tier TEXT, catalog_json TEXT)')
    for i, tier in enumerate(TIERS):
        con.execute('INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?,?)',
                    (_uuid(i), f'slug{i}', 10.0 + i, -5.0 + i, 3, 'ztf', 0, '', tier,
                     orjson.dumps({} if tier is None else {'catalog_tier': tier}).decode()))
    parquet = tmp_path / 'sources.parquet'
    pd.DataFrame({'source_uuid': [_uuid(i) for i in range(len(TIERS))],
                  'coord_slug': [f'slug{i}' for i in range(len(TIERS))],
                  'ra': [10.0 + i for i in range(len(TIERS))],
                  'dec': [-5.0 + i for i in range(len(TIERS))],
                  'catalog_tier': TIERS}).to_parquet(parquet, index=False)
    engine = api_filters.startup(con, parquet)
    store = coord_lists.connect(tmp_path / 'coord_lists.sqlite')
    ann = api_db.connect_annotations(tmp_path / 'annotations.sqlite')

    app.dependency_overrides[api_db.get_science] = lambda: con
    app.dependency_overrides[api_db.get_annotations] = lambda: ann
    app.dependency_overrides[api_db.get_coord_lists] = lambda: store
    app.dependency_overrides[api_filters.get_filters] = lambda: engine
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    api_filters.shutdown()
    for c in (con, store, ann):
        c.close()


def _upload(client, text, **kw):
    res = client.post('/api/coord-lists', json={'text': text, 'name': 'mine', **kw})
    assert res.status_code == 200, res.text
    return res.json()


def test_upload_matches_decimal_and_sexagesimal_positions(coord_client):
    text = '\n'.join([
        '# ra dec',
        '10.0003 -5.0002',               # source 0, ~1.3" away
        '00:48:00 -03:00:00',            # (12, -3): source 2
        '14.0, -1.0',                    # source 4, comma separated
        '200 40',                        # nothing there
        'not a position',
        '',
    ])
    s = _upload(coord_client, text)
    assert (s['n_coords'], s['n_matched_coords'], s['n_unmatched'], s['n_sources']) == (4, 3, 1, 3)
    assert s['unparsed'] == ['not a position']
    assert s['by_catalog'] == {'in_both': 1, 'in_ztf': 1, 'in_ztf_wide': 1}


def test_a_leading_name_is_skipped(coord_client):
    s = _upload(coord_client, 'cand_1 10 -5\ncand_2, 00:48:00, -03:00:00\ncand 3 not here')
    assert (s['n_coords'], s['n_sources'], s['unparsed']) == (2, 2, ['cand 3 not here'])


def test_radius_decides_what_matches(coord_client):
    assert _upload(coord_client, '10.0003 -5.0002', radius_arcsec=1.0)['n_sources'] == 0
    assert _upload(coord_client, '10.0003 -5.0002', radius_arcsec=2.0)['n_sources'] == 1


def test_list_shows_only_the_matched_sources(coord_client):
    s = _upload(coord_client, '10 -5\n12 -3\n14 -1')
    page = coord_client.get('/api/sources', params={'coord_list': s['id']}).json()
    assert sorted(i['source_uuid'] for i in page['items']) == [_uuid(0), _uuid(2), _uuid(4)]
    # And it composes with the catalog tabs
    page = coord_client.get('/api/sources', params={'coord_list': s['id'], 'catalog': 'in_ztf'}).json()
    assert [i['source_uuid'] for i in page['items']] == [_uuid(2)]


def test_a_list_matching_nothing_shows_nothing(coord_client):
    s = _upload(coord_client, '200 40')
    assert coord_client.get('/api/sources', params={'coord_list': s['id']}).json()['total'] == 0


def test_table_and_preview_respect_the_list(coord_client):
    s = _upload(coord_client, '10 -5\n11 -4')
    rows = coord_client.get('/api/table', params={'coord_list': s['id']}).json()['rows']
    assert sorted(r['source_uuid'] for r in rows) == [_uuid(0), _uuid(1)]
    preview = coord_client.post('/api/filters/preview', params={'coord_list': s['id']}, json=None).json()
    assert preview['scope_total'] == 2


def test_unknown_list_is_a_404(coord_client):
    assert coord_client.get('/api/sources', params={'coord_list': 'nope'}).status_code == 404
    assert coord_client.get('/api/coord-lists/nope').status_code == 404


def test_preview_matches_without_saving(coord_client):
    res = coord_client.post('/api/coord-lists/preview', json={'text': '10 -5\n12 -3'}).json()
    assert res['n_sources'] == 2 and 'id' not in res
    assert coord_client.get('/api/coord-lists').json() == []


def test_saved_lists_are_named_and_visible_to_everyone(coord_client):
    a = _upload(coord_client, '10 -5\n12 -3', name='bright ones')
    b = _upload(coord_client, '10 -5\n12 -3', name='same positions, another copy')
    assert a['id'] != b['id']                     # each upload is its own list
    listed = {l['id']: (l['name'], l['n_sources']) for l in coord_client.get('/api/coord-lists').json()}
    assert listed == {a['id']: ('bright ones', 2), b['id']: ('same positions, another copy', 2)}
    assert 'author' not in a


def test_a_list_needs_a_name(coord_client):
    assert coord_client.post('/api/coord-lists', json={'text': '10 -5', 'name': ' '}).status_code == 422


def test_anyone_can_rename(coord_client):
    s = _upload(coord_client, '10 -5')
    url = f"/api/coord-lists/{s['id']}"
    assert coord_client.patch(url, json={'name': 'renamed'}).json()['name'] == 'renamed'
    assert coord_client.get(url).json()['name'] == 'renamed'
    assert coord_client.patch('/api/coord-lists/nope', json={'name': 'x'}).status_code == 404


def test_delete_hides_the_list_but_keeps_the_row(coord_client):
    from api import db as api_db
    from api.main import app
    s = _upload(coord_client, '10 -5')
    url = f"/api/coord-lists/{s['id']}"
    assert coord_client.delete(url).status_code == 200
    assert coord_client.get(url).status_code == 404
    assert coord_client.delete(url).status_code == 404
    assert coord_client.get('/api/coord-lists').json() == []
    assert coord_client.get('/api/sources', params={'coord_list': s['id']}).status_code == 404
    # Recoverable: the row is only marked
    store = app.dependency_overrides[api_db.get_coord_lists]()
    assert store.execute('SELECT deleted_at FROM coord_lists WHERE id=?', (s['id'],)).fetchone()[0]


def test_an_older_store_is_migrated(tmp_path):
    import sqlite3
    from api import coord_lists
    old = sqlite3.connect(tmp_path / 'old.sqlite')
    old.execute('CREATE TABLE coord_lists (id TEXT PRIMARY KEY, name TEXT, created TEXT, '
                'summary_json TEXT NOT NULL, uuids_json TEXT NOT NULL)')
    old.execute("INSERT INTO coord_lists VALUES ('abc', 'old.txt', '2026-10-27', '{\"n_sources\": 1}', '[\"u\"]')")
    old.commit(); old.close()
    con = coord_lists.connect(tmp_path / 'old.sqlite')
    assert coord_lists.get(con, 'abc')['name'] == 'old.txt'
    assert coord_lists.get(con, 'abc')['columns'] == []
    assert coord_lists.uuids(con, 'abc') == ['u']
    assert coord_lists.values(con, 'abc') == {'columns': [], 'rows': {}}


# ── value columns ─────────────────────────────────────────────────────────────

VALUED = '\n'.join([
    'name ra dec score other',
    'A 10 -5 0.5 1',           # source 0
    'B 12 -3 2.5 nan',         # source 2; `other` missing
    'C, 14, -1, 1.5',          # source 4; `other` absent
    'D 200 40 9 9',            # nothing there
])


def test_a_header_declares_value_columns(coord_client):
    s = _upload(coord_client, VALUED)
    assert s['columns'] == ['score', 'other']
    assert (s['n_coords'], s['n_sources'], s['n_unparsed']) == (4, 3, 0)
    # Commented, name-less and tab-separated headers are the same header
    s = _upload(coord_client, '# ra dec\tscore\n10\t-5\t0.5\n')
    assert s['columns'] == ['score'] and s['n_sources'] == 1


def test_without_a_header_extra_numbers_are_ignored(coord_client):
    s = _upload(coord_client, '10 -5 0.5\nX 12 -3 2.5 7')
    assert s['columns'] == [] and s['n_sources'] == 2


def test_sexagesimal_positions_keep_their_values(coord_client):
    s = _upload(coord_client, 'ra dec score\n00:48:00 -03:00:00 2.5\n00 48 00 -03 00 00 2.5')
    assert s['n_sources'] == 1
    rows = coord_client.get('/api/table', params={'coord_list': s['id']}).json()['rows']
    assert rows[0]['list_values'] == {'score': 2.5}


def test_sort_by_a_list_value_orders_the_whole_list(coord_client):
    s = _upload(coord_client, VALUED)
    order = lambda **kw: [i['source_uuid'] for i in coord_client.get(
        '/api/sources', params={'coord_list': s['id'], **kw}).json()['items']]
    assert order(sort='list:score', desc=True) == [_uuid(2), _uuid(4), _uuid(0)]
    assert order(sort='list:score') == [_uuid(0), _uuid(4), _uuid(2)]
    # A missing value sorts last whichever way, then uuid order among the missing
    assert order(sort='list:other', desc=True) == [_uuid(0), _uuid(2), _uuid(4)]
    assert order(sort='list:other') == [_uuid(0), _uuid(2), _uuid(4)]
    # And composes with the other cuts
    assert order(sort='list:score', desc=True, catalog='in_ztf') == [_uuid(2)]


def test_locate_agrees_with_the_list_order(coord_client):
    s = _upload(coord_client, VALUED)
    for target, want in [(_uuid(2), 0), (_uuid(4), 1), (_uuid(0), 2)]:
        r = coord_client.get('/api/sources-locate', params={
            'coord_list': s['id'], 'sort': 'list:score', 'desc': True, 'target': target})
        assert r.json()['offset'] == want
    nxt = coord_client.get('/api/sources-next-unlabelled', params={
        'coord_list': s['id'], 'sort': 'list:score', 'desc': True, 'author': 'me', 'after': 1}).json()
    assert nxt == {'offset': 1, 'source_uuid': _uuid(4)}


def test_table_rows_carry_and_sort_by_list_values(coord_client):
    s = _upload(coord_client, VALUED)
    body = coord_client.get('/api/table', params={'coord_list': s['id'], 'sort': 'list:score',
                                                 'desc': True}).json()
    assert [r['source_uuid'] for r in body['rows']] == [_uuid(2), _uuid(4), _uuid(0)]
    assert body['rows'][0]['list_values'] == {'score': 2.5, 'other': None}
    assert body['rows'][1]['list_values'] == {'score': 1.5, 'other': None}
    # Without a list applied, rows do not grow the key
    assert 'list_values' not in coord_client.get('/api/table').json()['rows'][0]


def test_the_nearest_position_decides_a_shared_source(coord_client):
    s = _upload(coord_client, 'ra dec score\n10.0003 -5.0002 1\n10.0001 -5.0001 2', radius_arcsec=2)
    rows = coord_client.get('/api/table', params={'coord_list': s['id']}).json()['rows']
    assert rows[0]['list_values'] == {'score': 2.0}


def test_list_sorts_need_a_list_with_that_column(coord_client):
    s = _upload(coord_client, VALUED)
    assert coord_client.get('/api/sources', params={'sort': 'list:score'}).status_code == 422
    assert coord_client.get('/api/sources', params={'coord_list': s['id'],
                                                   'sort': 'list:nope'}).status_code == 422
    assert coord_client.get('/api/table', params={'coord_list': s['id'],
                                                 'sort': 'list:nope'}).status_code == 422
    plain = _upload(coord_client, '10 -5')
    assert coord_client.get('/api/sources', params={'coord_list': plain['id'],
                                                   'sort': 'list:score'}).status_code == 422


def test_a_header_with_a_bad_column_name_is_not_a_header(coord_client):
    s = _upload(coord_client, 'name ra dec not-a-name\n10 -5 1')
    assert s['columns'] == [] and s['unparsed'] == ['name ra dec not-a-name'] and s['n_sources'] == 1
