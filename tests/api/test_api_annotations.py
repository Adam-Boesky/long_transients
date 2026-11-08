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
#
# One per object, shared by everyone, and written to every detection of the
# object (UUIDS[0] and UUIDS[2] share a group_id in the fixture).

def _put(client, uuid, status, author='adam'):
    return client.put(f'/api/sources/{uuid}/followup',
                      json={'status': status, 'author': author})


def test_followup_round_trips(client):
    r = _put(client, UUIDS[0], 'queued')
    assert r.status_code == 200
    assert r.json() == {**r.json(), 'source_uuid': UUIDS[0], 'status': 'queued',
                        'author': 'adam'}
    assert client.get(f'/api/sources/{UUIDS[0]}/followup').json()['status'] == 'queued'
    assert client.get(f'/api/sources/{UUIDS[0]}').json()['followup']['status'] == 'queued'


def test_followup_is_null_until_someone_sets_it(client):
    assert client.get(f'/api/sources/{UUIDS[1]}/followup').json() is None
    assert client.get(f'/api/sources/{UUIDS[1]}').json()['followup'] is None


def test_followup_is_rejected_outside_the_vocabulary(client):
    assert _put(client, UUIDS[0], 'observer').status_code == 422


def test_followup_is_one_per_object_not_per_author(client):
    _put(client, UUIDS[1], 'to queue', author='adam')
    r = _put(client, UUIDS[1], 'observed', author='beth')
    assert r.json()['status'] == 'observed' and r.json()['author'] == 'beth'
    assert client.get(f'/api/sources/{UUIDS[1]}/followup').json()['status'] == 'observed'


def test_followup_covers_every_detection_of_the_object(client):
    _put(client, UUIDS[0], 'queued')
    mate = client.get(f'/api/sources/{UUIDS[2]}/followup').json()
    assert mate['status'] == 'queued', 'the group-mate carries the same status'
    assert client.get(f'/api/sources/{UUIDS[1]}/followup').json() is None


def test_empty_followup_clears_it_for_the_whole_object(client):
    _put(client, UUIDS[2], 'observed')
    r = _put(client, UUIDS[0], '')
    assert r.json() is None
    assert client.get(f'/api/sources/{UUIDS[2]}/followup').json() is None


def test_followup_does_not_touch_the_verdict(client):
    client.put(f'/api/sources/{UUIDS[0]}/annotations/adam', json={'label': 'SN'})
    _put(client, UUIDS[0], 'queued')
    (a,) = client.get(f'/api/sources/{UUIDS[0]}/annotations').json()
    assert a['label'] == 'SN' and 'status' not in a


def test_followup_history_records_each_change(client, annotations_con):
    for st in ('to queue', 'queued', 'observed', ''):
        _put(client, UUIDS[1], st)
    rows = annotations_con.execute(
        'SELECT status FROM followup_history WHERE source_uuid=? ORDER BY id',
        (UUIDS[1],)).fetchall()
    assert [r[0] for r in rows] == ['to queue', 'queued', 'observed', None]


def test_followup_404s_for_an_unknown_source(client):
    assert _put(client, 'nope', 'queued').status_code == 404


def test_followup_needs_an_author(client):
    r = client.put(f'/api/sources/{UUIDS[0]}/followup', json={'status': 'queued'})
    assert r.status_code == 422


def test_table_annotations_carry_followup(client):
    _put(client, UUIDS[1], 'observed', author='beth')
    body = client.get('/api/table/annotations').json()
    assert body['followup'][UUIDS[1]] == {**body['followup'][UUIDS[1]],
                                          'status': 'observed', 'author': 'beth'}
    assert UUIDS[3] not in body['followup']


# ── migrating the per-author column ───────────────────────────────────────────

def _legacy_db(path):
    """An annotations database from before `followup` existed, with the
    per-author `status` column filled in."""
    import re
    import sqlite3
    from source_db.serving.schema import ANNOTATIONS_SCHEMA
    con = sqlite3.connect(path)
    con.executescript(re.sub(
        r'CREATE (TABLE|INDEX) IF NOT EXISTS (followup|idx_followup|idx_fhist)[^;]*;',
        '', ANNOTATIONS_SCHEMA))
    con.executemany(
        'INSERT INTO annotations (source_uuid, author, label, status, created_at, updated_at) '
        'VALUES (?,?,?,?,?,?)', [
            (UUIDS[0], 'adam', 'SN', 'to queue', 't1', 't1'),
            (UUIDS[0], 'beth', 'SN', 'observed', 't2', 't2'),
            (UUIDS[0], 'cara', 'AGN', 'queued', 't3', 't3'),
            (UUIDS[1], 'adam', 'SN', 'queued', 't1', 't1'),
            (UUIDS[1], 'beth', 'SN', 'queued', 't4', 't4'),
            (UUIDS[3], 'adam', 'SN', None, 't1', 't1'),
        ])
    con.commit()
    con.close()


def test_bootstrap_backfills_followup_from_the_per_author_column(tmp_path):
    from api import db as api_db
    path = tmp_path / 'legacy.sqlite'
    _legacy_db(path)
    con = api_db.connect_annotations(path)
    rows = {r[0]: (r[1], r[2]) for r in con.execute(
        'SELECT source_uuid, status, author FROM followup')}
    # The furthest-along stage wins, whoever set it ...
    assert rows[UUIDS[0]] == ('observed', 'beth')
    # ... and within a stage the latest edit is credited.
    assert rows[UUIDS[1]] == ('queued', 'beth')
    assert UUIDS[3] not in rows
    # One source of truth: the per-author column is consumed ...
    assert con.execute('SELECT COUNT(*) FROM annotations WHERE status IS NOT NULL'
                       ).fetchone()[0] == 0
    # ... so a later clear is not undone by the next open.
    con.execute('DELETE FROM followup')
    con.commit()
    con.close()
    api_db._bootstrapped.discard(path)
    con = api_db.connect_annotations(path)
    assert con.execute('SELECT COUNT(*) FROM followup').fetchone()[0] == 0
    con.close()


def test_sync_groups_makes_every_detection_agree(client, annotations_con, filter_engine):
    from api import followup
    # Two detections of one object, set apart (as a backfill would leave them).
    followup.set_status(annotations_con, [UUIDS[0]], 'to queue', 'adam', record=False)
    followup.set_status(annotations_con, [UUIDS[2]], 'queued', 'beth', record=False)
    annotations_con.commit()
    assert followup.sync_groups(annotations_con, filter_engine) == 1
    assert client.get(f'/api/sources/{UUIDS[0]}/followup').json() == {
        **client.get(f'/api/sources/{UUIDS[0]}/followup').json(),
        'status': 'queued', 'author': 'beth'}
    assert client.get(f'/api/sources/{UUIDS[2]}/followup').json()['status'] == 'queued'


def test_merged_prefers_the_later_stage():
    from api.followup import merged
    assert merged(['to queue', 'observed', 'queued']) == 'observed'
    assert merged(['to queue', 'queued']) == 'queued'
    assert merged([None, 'to queue']) == 'to queue'
    assert merged([None, None]) is None
