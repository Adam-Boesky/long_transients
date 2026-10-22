"""Exporting annotations — the output the labelling campaign exists to produce."""
from __future__ import annotations

import csv
import io

from tests.conftest import UUIDS


def _label(client, uuid, author, label, tags=None):
    return client.put(f'/api/sources/{uuid}/annotations/{author}',
                      json={'label': label, 'tags': tags or []})


def test_export_is_empty_before_anything_is_labelled(client):
    body = client.get('/api/annotations/export').json()
    assert body['n'] == 0 and body['rows'] == []


def test_export_returns_current_state(client):
    _label(client, UUIDS[0], 'adam', 'SN', ['nuclear'])
    _label(client, UUIDS[1], 'beth', 'AGN')
    body = client.get('/api/annotations/export').json()
    assert body['n'] == 2 and body['table'] == 'annotations'
    got = {(r['source_uuid'], r['author'], r['label']) for r in body['rows']}
    assert got == {(UUIDS[0], 'adam', 'SN'), (UUIDS[1], 'beth', 'AGN')}


def test_export_decodes_tags_to_a_list(client):
    _label(client, UUIDS[0], 'adam', 'SN', ['nuclear', 'blue'])
    row = client.get('/api/annotations/export').json()['rows'][0]
    assert row['tags'] == ['nuclear', 'blue'], 'tags must not leak as a JSON string'


def test_export_can_filter_by_author(client):
    _label(client, UUIDS[0], 'adam', 'SN')
    _label(client, UUIDS[1], 'beth', 'AGN')
    body = client.get('/api/annotations/export?author=adam').json()
    assert body['n'] == 1 and body['rows'][0]['author'] == 'adam'


def test_export_reflects_only_the_latest_edit(client):
    _label(client, UUIDS[0], 'adam', 'SN')
    _label(client, UUIDS[0], 'adam', 'AGN')
    body = client.get('/api/annotations/export').json()
    assert body['n'] == 1 and body['rows'][0]['label'] == 'AGN'


def test_history_export_keeps_every_edit(client):
    _label(client, UUIDS[0], 'adam', 'SN')
    _label(client, UUIDS[0], 'adam', 'AGN')
    body = client.get('/api/annotations/export?include_history=true').json()
    assert body['table'] == 'history'
    assert [r['label'] for r in body['rows']] == ['SN', 'AGN']


def test_history_survives_a_retraction(client):
    _label(client, UUIDS[0], 'adam', 'SN')
    client.delete(f'/api/sources/{UUIDS[0]}/annotations/adam')
    assert client.get('/api/annotations/export').json()['n'] == 0
    hist = client.get('/api/annotations/export?include_history=true').json()
    assert hist['n'] == 1 and hist['rows'][0]['label'] == 'SN'


def test_csv_export_is_parseable(client):
    _label(client, UUIDS[0], 'adam', 'SN', ['nuclear', 'blue'])
    _label(client, UUIDS[1], 'beth', 'AGN')
    r = client.get('/api/annotations/export?fmt=csv')
    assert r.status_code == 200
    assert 'text/csv' in r.headers['content-type']
    assert 'attachment' in r.headers.get('content-disposition', '')
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert len(rows) == 2
    assert {'source_uuid', 'author', 'label', 'tags'} <= set(rows[0])
    adam = next(x for x in rows if x['author'] == 'adam')
    assert adam['tags'] == 'nuclear,blue', 'list tags must flatten for CSV'


def test_csv_export_of_nothing_is_empty_not_an_error(client):
    r = client.get('/api/annotations/export?fmt=csv')
    assert r.status_code == 200 and r.text == ''


def test_export_rejects_an_unknown_format(client):
    assert client.get('/api/annotations/export?fmt=xlsx').status_code == 422
