"""The shared label vocabulary."""
from __future__ import annotations


def test_starts_with_the_two_defaults(client):
    assert [l['name'] for l in client.get('/api/labels').json()] == \
        ['Trash', 'Follow-up']


def test_added_labels_come_after_the_defaults(client):
    r = client.post('/api/labels', json={'name': ' Nuclear ', 'author': 'adam'})
    assert r.status_code == 201
    assert r.json()['name'] == 'Nuclear', 'surrounding whitespace is trimmed'
    got = client.get('/api/labels').json()
    assert [l['name'] for l in got] == ['Trash', 'Follow-up', 'Nuclear']
    assert got[-1]['created_by'] == 'adam'


def test_duplicates_are_refused_whatever_the_case(client):
    assert client.post('/api/labels', json={'name': 'trash'}).status_code == 409


def test_blank_names_are_refused(client):
    assert client.post('/api/labels', json={'name': '   '}).status_code == 422
    assert client.post('/api/labels', json={'name': ''}).status_code == 422


def test_reorder_sticks_and_new_labels_join_the_end(client):
    client.post('/api/labels', json={'name': 'Nuclear'})
    r = client.put('/api/labels/order',
                   json={'names': ['Nuclear', 'trash', 'Follow-up']})
    assert r.status_code == 200
    assert [l['name'] for l in r.json()] == ['Nuclear', 'Trash', 'Follow-up'], (
        'names match case-insensitively but keep their stored spelling')
    client.post('/api/labels', json={'name': 'TDE'})
    assert [l['name'] for l in client.get('/api/labels').json()] == \
        ['Nuclear', 'Trash', 'Follow-up', 'TDE']


def test_reorder_must_name_every_label_exactly_once(client):
    for names in (['Trash'], ['Trash', 'Follow-up', 'Nope'],
                  ['Trash', 'Trash', 'Follow-up']):
        r = client.put('/api/labels/order', json={'names': names})
        assert r.status_code == 409, names
    assert [l['name'] for l in client.get('/api/labels').json()] == \
        ['Trash', 'Follow-up'], 'a refused reorder changes nothing'


def test_position_is_added_to_a_labels_table_that_predates_it(tmp_path):
    import sqlite3
    from api import db
    path = tmp_path / 'old.sqlite'
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE labels (name TEXT PRIMARY KEY COLLATE NOCASE, '
                'created_by TEXT, created_at TEXT NOT NULL)')
    con.commit(); con.close()
    con = db.connect_annotations(path)
    cols = {r[1] for r in con.execute('PRAGMA table_info(labels)')}
    assert {'position', 'color'} <= cols


def test_colour_is_set_for_everyone_and_can_be_reset(client):
    r = client.patch('/api/labels/trash', json={'color': '#AA00FF'})
    assert r.status_code == 200 and r.json()['color'] == '#aa00ff'
    got = {l['name']: l['color'] for l in client.get('/api/labels').json()}
    assert got == {'Trash': '#aa00ff', 'Follow-up': None}
    client.patch('/api/labels/Trash', json={'color': None})
    assert client.get('/api/labels').json()[0]['color'] is None


def test_colour_must_be_hex_and_label_must_exist(client):
    for bad in ('red', '#abc', '#12345g'):
        assert client.patch('/api/labels/Trash',
                            json={'color': bad}).status_code == 422, bad
    assert client.patch('/api/labels/Nope',
                        json={'color': '#000000'}).status_code == 404
