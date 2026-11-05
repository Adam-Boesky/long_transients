"""The writable tier: labels, history, and isolation from the science data."""
from __future__ import annotations

from tests.conftest import LC_UUIDS, UUIDS

TARGET = UUIDS[0]


def test_put_creates_an_annotation(client):
    r = client.put(f'/api/sources/{TARGET}/annotations/adam',
                   json={'label': 'SN', 'confidence': 'high',
                         'note': 'clear rise', 'tags': ['nuclear', 'blue']})
    assert r.status_code == 200
    body = r.json()
    assert body['label'] == 'SN'
    assert body['tags'] == ['nuclear', 'blue']
    assert body['created_at'] and body['updated_at']


def test_put_is_idempotent_and_updates_in_place(client):
    client.put(f'/api/sources/{TARGET}/annotations/adam', json={'label': 'SN'})
    first = client.get(f'/api/sources/{TARGET}/annotations').json()
    client.put(f'/api/sources/{TARGET}/annotations/adam', json={'label': 'AGN'})
    second = client.get(f'/api/sources/{TARGET}/annotations').json()

    assert len(second) == 1, 'a second PUT must update, not insert a duplicate'
    assert second[0]['label'] == 'AGN'
    # created_at is preserved across edits; updated_at moves.
    assert second[0]['created_at'] == first[0]['created_at']


def test_authors_are_independent(client):
    client.put(f'/api/sources/{TARGET}/annotations/adam', json={'label': 'SN'})
    client.put(f'/api/sources/{TARGET}/annotations/beth', json={'label': 'AGN'})
    got = {a['author']: a['label'] for a in
           client.get(f'/api/sources/{TARGET}/annotations').json()}
    assert got == {'adam': 'SN', 'beth': 'AGN'}, (
        'disagreement between authors is the useful state, not a conflict')


def test_put_404s_for_unknown_source(client):
    r = client.put('/api/sources/not-a-uuid/annotations/adam', json={'label': 'SN'})
    assert r.status_code == 404


def test_delete_removes_the_annotation(client):
    client.put(f'/api/sources/{TARGET}/annotations/adam', json={'label': 'SN'})
    assert client.delete(f'/api/sources/{TARGET}/annotations/adam').status_code == 200
    assert client.get(f'/api/sources/{TARGET}/annotations').json() == []


def test_delete_404s_when_there_is_nothing_to_delete(client):
    assert client.delete(f'/api/sources/{TARGET}/annotations/nobody').status_code == 404


def test_history_keeps_every_edit_including_deleted_ones(client, annotations_con):
    client.put(f'/api/sources/{TARGET}/annotations/adam', json={'label': 'SN'})
    client.put(f'/api/sources/{TARGET}/annotations/adam', json={'label': 'AGN'})
    client.delete(f'/api/sources/{TARGET}/annotations/adam')

    labels = [r['label'] for r in annotations_con.execute(
        'SELECT label FROM history WHERE source_uuid=? AND author=? '
        'ORDER BY id', (TARGET, 'adam'))]
    assert labels == ['SN', 'AGN'], (
        'the labelling process is itself data; retracting must not erase it')


def test_annotations_never_touch_the_science_database(client, science_con):
    """The science tier is opened read-only, so a write must fail loudly."""
    import sqlite3
    client.put(f'/api/sources/{TARGET}/annotations/adam', json={'label': 'SN'})
    try:
        science_con.execute("UPDATE sources SET ra = 999 WHERE source_uuid=?",
                            (TARGET,))
        science_con.commit()
        raised = False
    except sqlite3.OperationalError:
        raised = True
    assert raised, 'science.sqlite must be read-only in the API process'


def test_each_test_gets_a_clean_annotations_db(client):
    """Guards the fixture scoping: leakage here would make the suite order-dependent."""
    assert client.get('/api/stats').json()['n_annotated'] == 0


# ── stats ─────────────────────────────────────────────────────────────────────

def test_stats_counts_sources_and_epochs(client):
    s = client.get('/api/stats').json()
    assert s['n_sources'] == len(UUIDS)
    assert s['n_with_lightcurves'] == len(LC_UUIDS)
    assert s['n_epochs'] > 0
    assert s['built_at']


def test_stats_aggregates_labels_and_authors(client):
    client.put(f'/api/sources/{UUIDS[0]}/annotations/adam', json={'label': 'SN'})
    client.put(f'/api/sources/{UUIDS[1]}/annotations/adam', json={'label': 'SN'})
    client.put(f'/api/sources/{UUIDS[1]}/annotations/beth', json={'label': 'AGN'})

    s = client.get('/api/stats').json()
    assert s['by_label'] == {'SN': 2, 'AGN': 1}
    assert s['by_author'] == {'adam': 2, 'beth': 1}
    # two distinct sources carry annotations, though there are three rows
    assert s['n_annotated'] == 2


# ── progress ──────────────────────────────────────────────────────────────────

def test_progress_is_empty_before_anyone_labels(client):
    p = client.get('/api/progress').json()
    assert p['n_sources'] == len(UUIDS)
    assert p['n_labelled'] == 0
    assert p['by_label'] == {} and p['by_member'] == []
    assert p['disagreements'] == []


def test_progress_cross_tabs_labels_against_authors(client):
    client.put(f'/api/sources/{UUIDS[0]}/annotations/adam', json={'label': 'SN'})
    client.put(f'/api/sources/{UUIDS[1]}/annotations/adam', json={'label': 'SN'})
    client.put(f'/api/sources/{UUIDS[1]}/annotations/beth', json={'label': 'AGN'})

    p = client.get('/api/progress').json()
    assert p['by_label'] == {'SN': {'adam': 2}, 'AGN': {'beth': 1}}
    assert p['authors'] == ['adam', 'beth']
    assert p['n_labelled'] == 2


def test_progress_ignores_comment_only_annotations(client):
    """A note with no label says nothing about which bucket a source is in."""
    client.put(f'/api/sources/{UUIDS[0]}/annotations/adam',
               json={'note': 'worth a second look'})
    p = client.get('/api/progress').json()
    assert p['n_labelled'] == 0 and p['by_member'] == []


def test_progress_per_member_counts_sources_labels_and_comments(client):
    client.put(f'/api/sources/{UUIDS[0]}/annotations/adam',
               json={'label': 'SN', 'note': 'clear rise'})
    client.put(f'/api/sources/{UUIDS[1]}/annotations/adam', json={'label': 'AGN'})
    client.put(f'/api/sources/{UUIDS[2]}/annotations/adam', json={'label': 'AGN'})
    client.put(f'/api/sources/{UUIDS[0]}/annotations/beth', json={'label': 'SN'})

    rows = {m['author']: m for m in client.get('/api/progress').json()['by_member']}
    assert rows['adam'] == {'author': 'adam', 'n_sources': 3, 'n_labels': 2,
                            'n_comments': 1}
    assert rows['beth']['n_sources'] == 1
    # Busiest first, so the table opens on whoever has done the most.
    assert [m['author'] for m in
            client.get('/api/progress').json()['by_member']] == ['adam', 'beth']


def test_progress_reports_disagreements_with_both_verdicts(client):
    client.put(f'/api/sources/{UUIDS[0]}/annotations/adam',
               json={'label': 'SN', 'note': 'rises then fades'})
    client.put(f'/api/sources/{UUIDS[0]}/annotations/beth', json={'label': 'AGN'})
    client.put(f'/api/sources/{UUIDS[1]}/annotations/adam', json={'label': 'SN'})
    client.put(f'/api/sources/{UUIDS[1]}/annotations/beth', json={'label': 'SN'})

    p = client.get('/api/progress').json()
    assert p['n_disagreements'] == 1, 'agreement is not a disagreement'
    (d,) = p['disagreements']
    assert d['source_uuid'] == UUIDS[0]
    assert d['labels'] == ['AGN', 'SN']
    assert d['ra'] is not None and d['coord_slug']
    assert {v['author']: v['label'] for v in d['verdicts']} == {
        'adam': 'SN', 'beth': 'AGN'}
    assert [v['note'] for v in d['verdicts'] if v['author'] == 'adam'] == [
        'rises then fades']


def test_progress_truncates_disagreements_but_still_counts_them(client):
    for uuid in UUIDS[:3]:
        client.put(f'/api/sources/{uuid}/annotations/adam', json={'label': 'SN'})
        client.put(f'/api/sources/{uuid}/annotations/beth', json={'label': 'AGN'})

    p = client.get('/api/progress', params={'limit': 2}).json()
    assert p['n_disagreements'] == 3
    assert len(p['disagreements']) == 2


# ── follow-up status ──────────────────────────────────────────────────────────

def test_status_round_trips(client):
    r = client.put(f'/api/sources/{UUIDS[0]}/annotations/adam',
                   json={'label': 'SN', 'status': 'queued'})
    assert r.status_code == 200 and r.json()['status'] == 'queued'
    (a,) = client.get(f'/api/sources/{UUIDS[0]}/annotations').json()
    assert a['status'] == 'queued'


def test_status_is_rejected_outside_the_vocabulary(client):
    r = client.put(f'/api/sources/{UUIDS[0]}/annotations/adam',
                   json={'label': 'SN', 'status': 'observer'})
    assert r.status_code == 422


def test_empty_status_clears_it(client):
    """'nothing' is NULL, and the UI sends '' to mean it."""
    client.put(f'/api/sources/{UUIDS[0]}/annotations/adam',
               json={'label': 'SN', 'status': 'observed'})
    r = client.put(f'/api/sources/{UUIDS[0]}/annotations/adam',
                   json={'label': 'SN', 'status': ''})
    assert r.json()['status'] is None


def test_a_label_edit_does_not_clear_the_status(client):
    """The PUT replaces the row, so the client has to resend status -- this
    pins the server half: what is sent is what is stored."""
    client.put(f'/api/sources/{UUIDS[0]}/annotations/adam',
               json={'label': 'SN', 'status': 'queued'})
    r = client.put(f'/api/sources/{UUIDS[0]}/annotations/adam',
                   json={'label': 'AGN', 'status': 'queued'})
    assert r.json() == {**r.json(), 'label': 'AGN', 'status': 'queued'}


def test_status_defaults_to_none(client):
    r = client.put(f'/api/sources/{UUIDS[0]}/annotations/adam',
                   json={'label': 'SN'})
    assert r.json()['status'] is None


def test_history_records_each_status_change(client):
    for st in ('to queue', 'queued', 'observed'):
        client.put(f'/api/sources/{UUIDS[0]}/annotations/adam',
                   json={'label': 'SN', 'status': st})
    rows = client.get('/api/annotations/export',
                      params={'include_history': True, 'author': 'adam'}).json()
    assert [r['status'] for r in rows['rows']] == ['to queue', 'queued', 'observed']


def test_export_carries_status(client):
    client.put(f'/api/sources/{UUIDS[0]}/annotations/adam',
               json={'label': 'SN', 'status': 'queued'})
    assert client.get('/api/annotations/export').json()['rows'][0]['status'] == 'queued'
    csv_text = client.get('/api/annotations/export', params={'fmt': 'csv'}).text
    assert 'status' in csv_text.splitlines()[0]


def test_table_annotations_carry_status(client):
    client.put(f'/api/sources/{UUIDS[0]}/annotations/adam',
               json={'label': 'SN', 'status': 'observed'})
    body = client.get('/api/table/annotations').json()
    assert body['annotations'][UUIDS[0]][0]['status'] == 'observed'
