"""`/api/table`: the one-row-per-source payload behind the Table page.

Split in two since the catalog half became cacheable: `/api/table` carries
what is static for a release and `/api/table/annotations` the verdicts. The
tests hold that boundary, because the value of the split is entirely that the
cached half contains nothing that can go stale.

The session fixture's sources carry none of the crossmatch or photometry
columns, which is the case the endpoint has to survive rather than the case
it has to compute, so the derivations are exercised as units against literal
catalog rows.
"""
from __future__ import annotations

from api.derive import CLASS_SOURCES, classes_of

CLASS_PREFIXES = [p for p, _, _ in CLASS_SOURCES]
from api.routers.table import _mean_abs_dmag
from tests.conftest import UUIDS


# ── the endpoint ──────────────────────────────────────────────────────────────

def test_table_returns_every_source(client):
    body = client.get('/api/table').json()
    assert body['total'] == len(UUIDS)
    assert {r['source_uuid'] for r in body['rows']} == set(UUIDS)


def test_table_row_has_the_columns_the_page_reads(client):
    row = client.get('/api/table').json()['rows'][0]
    # No `annotations`: they are the mutable half and live at their own
    # endpoint, so that a verdict filed by one person does not invalidate a
    # megabyte of catalog data that did not change.
    assert set(row) == {'source_uuid', 'coord_slug', 'ra', 'dec',
                        'catalog_tier', 'mean_abs_dmag', 'wise_w1w2',
                        'matched', 'classes'}
    assert set(row['matched']) == {'sdss', 'desi', 'tns', 'wise', 'agn_db'}
    # Only catalogs that actually said something appear; the fixture rows
    # are crossmatched to nothing, so the map is empty.
    assert set(row['classes']) <= set(CLASS_PREFIXES)


def test_table_tolerates_a_catalog_row_with_none_of_the_columns(client):
    """Nothing in the fixture is crossmatched or has ZTF/PanSTARRS mags."""
    rows = client.get('/api/table').json()['rows']
    assert all(r['mean_abs_dmag'] is None for r in rows)
    assert all(r['wise_w1w2'] is None for r in rows)
    assert all(not any(r['matched'].values()) for r in rows)
    assert all(v == [] for r in rows for v in r['classes'].values())


def test_table_pools_verdicts_across_authors(client):
    uuid = UUIDS[0]
    client.put(f'/api/sources/{uuid}/annotations/ada',
               json={'label': 'SN', 'note': 'rises then fades'})
    client.put(f'/api/sources/{uuid}/annotations/bob',
               json={'label': 'AGN', 'note': ''})

    got = client.get('/api/table/annotations').json()['annotations']
    assert sorted(a['author'] for a in got[uuid]) == ['ada', 'bob']
    assert {a['label'] for a in got[uuid]} == {'SN', 'AGN'}
    # Sources nobody has filed are absent rather than present-and-empty; the
    # page defaults them when it joins, and sending 198k empty lists to say
    # "no verdict" is the thing this endpoint exists to stop doing.
    assert UUIDS[1] not in got


def test_table_annotations_can_be_narrowed_to_one_author(client):
    client.put(f'/api/sources/{UUIDS[0]}/annotations/ada', json={'label': 'SN'})
    client.put(f'/api/sources/{UUIDS[0]}/annotations/bob', json={'label': 'AGN'})
    got = client.get('/api/table/annotations?author=ada').json()['annotations']
    assert [a['author'] for a in got[UUIDS[0]]] == ['ada']


def test_a_new_verdict_is_visible_immediately(client):
    """The catalog half is cached; the verdicts must not be.

    This is the failure the split is designed to make impossible -- filing a
    label and seeing the table still show the old one because a cache served
    the row it was embedded in.
    """
    assert client.get('/api/table/annotations').json()['annotations'] == {}
    client.get('/api/table')            # warm the cached half
    client.put(f'/api/sources/{UUIDS[2]}/annotations/ada', json={'label': 'SN'})
    got = client.get('/api/table/annotations').json()['annotations']
    assert got[UUIDS[2]][0]['label'] == 'SN'


def test_the_catalog_half_is_served_from_cache_unchanged(client):
    first = client.get('/api/table').json()
    second = client.get('/api/table').json()
    assert first == second


def test_table_narrows_with_a_filter_document(client):
    """The same filter document the classify page builds.

    The window is the only thing standing between the table and 198k rows, so
    narrowing it has to be possible without raising `limit`.
    """
    body = client.get('/api/table', params={
        'filter': '{"column": "source_uuid", "op": "=", "value": "%s"}' % UUIDS[0],
    }).json()
    assert body['total'] == 1
    assert [r['source_uuid'] for r in body['rows']] == [UUIDS[0]]


def test_table_rejects_a_filter_it_cannot_compile(client):
    assert client.get('/api/table',
                      params={'filter': '{"column": "nope", "op": "=", '
                                        '"value": 1}'}).status_code == 422
    assert client.get('/api/table',
                      params={'filter': '{oops'}).status_code == 422


def test_table_windows_rows_but_reports_the_full_count(client):
    body = client.get('/api/table?limit=2&offset=1').json()
    assert body['total'] == len(UUIDS)
    assert len(body['rows']) == 2
    assert body['rows'][0]['source_uuid'] == sorted(UUIDS)[1]


def test_table_rejects_an_oversized_window(client):
    assert client.get('/api/table?limit=100000').status_code == 422


# ── label state, cut before the window ────────────────────────────────────────

def _uuids(client, **params) -> set[str]:
    body = client.get('/api/table', params=params).json()
    assert body['total'] == len(body['rows'])
    return {r['source_uuid'] for r in body['rows']}


def test_labeled_by_me_picks_the_window_from_my_labels(client):
    """A labelled source past the window must still come back.

    Sorted by uuid with a one-row window, the last source is never in it --
    exactly how labels outside the top 5,000 by |dmag| went missing.
    """
    last = sorted(UUIDS)[-1]
    client.put(f'/api/sources/{last}/annotations/ada', json={'label': 'SN'})
    got = client.get('/api/table', params={
        'label_state': 'Labeled by me', 'author': 'ada', 'limit': 1}).json()
    assert got['total'] == 1
    assert [r['source_uuid'] for r in got['rows']] == [last]


def test_label_states_partition_the_catalog(client):
    a, b, c = sorted(UUIDS)[:3]
    client.put(f'/api/sources/{a}/annotations/ada', json={'label': 'SN'})
    client.put(f'/api/sources/{b}/annotations/bob', json={'label': 'AGN'})
    client.put(f'/api/sources/{c}/annotations/ada', json={'label': 'SN'})
    client.put(f'/api/sources/{c}/annotations/bob', json={'label': 'AGN'})

    assert _uuids(client, label_state='Labeled by me', author='ada') == {a, c}
    assert _uuids(client, label_state='Unlabeled by me', author='ada') == \
        set(UUIDS) - {a, c}
    assert _uuids(client, label_state='Unlabeled') == set(UUIDS) - {a, b, c}
    assert _uuids(client, label_state='Disagreements') == {c}


def test_labeled_by_me_with_no_labels_is_empty_not_an_error(client):
    assert _uuids(client, label_state='Labeled by me', author='nobody') == set()
    assert _uuids(client, label_state='Disagreements') == set()


def test_a_cleared_label_does_not_count_as_labelled(client):
    client.put(f'/api/sources/{UUIDS[0]}/annotations/ada', json={'label': None,
                                                                'note': 'hmm'})
    assert _uuids(client, label_state='Labeled by me', author='ada') == set()


def test_ztforce_done_lists_sources_with_a_stored_result(client, ztforce_con):
    from api import ztforce_store
    done, queued = sorted(UUIDS)[:2]
    ztforce_store.request_now(ztforce_con, done, 1.0, 2.0, 'v1')
    ztforce_store.finish(ztforce_con, done, 'v1', b'stacks', 10, None)
    ztforce_store.request_now(ztforce_con, queued, 3.0, 4.0, 'v1')
    got = client.get('/api/table/annotations').json()['ztforce_done']
    assert got == [done]


def test_label_state_rejects_nonsense(client):
    assert client.get('/api/table?label_state=Maybe').status_code == 422


def test_bucket_keeps_sources_anyone_filed_there(client):
    a, b, c = sorted(UUIDS)[:3]
    client.put(f'/api/sources/{a}/annotations/ada', json={'label': 'SN'})
    client.put(f'/api/sources/{b}/annotations/bob', json={'label': 'SN'})
    client.put(f'/api/sources/{c}/annotations/ada', json={'label': 'AGN'})

    assert _uuids(client, bucket_in='SN') == {a, b}
    assert _uuids(client, bucket_in='SN', label_state='Labeled by me',
                  author='ada') == {a}
    assert _uuids(client, bucket_in='Nothing') == set()


# ── mean |dmag| ───────────────────────────────────────────────────────────────

def test_mean_abs_dmag_averages_the_bands_present():
    cat = {'g_Catalog_Flag': 0, 'ZTF_gPSFMag': 20.0, 'PSTARR_gPSFMag': 18.0,
           'r_Catalog_Flag': 0, 'ZTF_rPSFMag': 19.0, 'PSTARR_rPSFMag': 20.0}
    assert _mean_abs_dmag(cat) == 1.5


def test_mean_abs_dmag_substitutes_the_panstarrs_limit_for_ztf_only():
    """Flag 1 is a ZTF-only detection: PanSTARRS contributes its limit."""
    assert _mean_abs_dmag(
        {'g_Catalog_Flag': 1, 'ZTF_gPSFMag': 20.3}) == 3.0


def test_mean_abs_dmag_substitutes_the_ztf_limit_for_panstarrs_only():
    assert _mean_abs_dmag(
        {'r_Catalog_Flag': 2, 'ZTF_r_mag_limit': 20.5,
         'PSTARR_rPSFMag': 21.5}) == 1.0


def test_mean_abs_dmag_falls_back_to_a_shared_flag_column():
    """Older rows carry one `Catalog_Flag` rather than one per band."""
    assert _mean_abs_dmag({'Catalog_Flag': 1, 'ZTF_iPSFMag': 22.1}) == 1.0


def test_mean_abs_dmag_is_none_when_no_band_is_measurable():
    assert _mean_abs_dmag({}) is None
    assert _mean_abs_dmag({'ZTF_gPSFMag': 20.0, 'PSTARR_gPSFMag': None}) is None
    assert _mean_abs_dmag({'g_Catalog_Flag': 0, 'ZTF_gPSFMag': float('nan'),
                           'PSTARR_gPSFMag': 18.0}) is None


# ── classes ───────────────────────────────────────────────────────────────────

def test_classes_deduplicate_the_agn_db_list():
    """AGN-DB records one class per contributing sub-catalog."""
    cat = {'agn_db_matched': True,
           'agn_db_best_class_all': '["generic_agn", "unknown", "generic_agn"]'}
    assert classes_of(cat)['AGN-DB'] == ['generic_agn', 'unknown']


def test_classes_read_a_scalar_column_whole():
    assert classes_of({'tns_matched': True, 'tns_type': 'SN IIn'})['TNS'] == ['SN IIn']


def test_a_matched_but_untyped_catalog_reads_as_unknown():
    """Otherwise TNS would vanish from the column as though it had never
    heard of the source, which is a different statement."""
    assert classes_of({'tns_matched': True, 'tns_type': None})['TNS'] == ['unknown']


def test_an_unmatched_catalog_contributes_nothing():
    """Absent rather than present-and-empty, so a caller can render the map
    directly without filtering blanks back out."""
    assert 'SDSS' not in classes_of({'sdss_matched': False, 'sdss_class': None})


def test_simbad_classifications_are_included():
    """`classify_app/lib/classes.py` lists four catalogs and omits SIMBAD.

    We cross-match it and it types sources, so leaving it out reported a
    SIMBAD-identified quasar as unclassified.
    """
    assert classes_of({'simbad_matched': True,
                       'simbad_otype_label': 'Quasar'})['SIMBAD'] == ['Quasar']

# ── sorting ───────────────────────────────────────────────────────────────────

def test_table_sorts_across_the_whole_catalog_not_the_window(client):
    """The reason sorting moved to the server.

    Ordering in the browser can only rank the rows already fetched, so "the
    largest |dmag|" meant "the largest among the first 5,000 by uuid". On the
    real catalog that answered 7.73 where the truth is 8.95.
    """
    body = client.get('/api/table', params={'sort': 'ra', 'desc': True,
                                            'limit': 2}).json()
    ras = [r['ra'] for r in body['rows']]
    assert ras == sorted(ras, reverse=True)
    every = client.get('/api/table').json()['rows']
    assert ras[0] == max(r['ra'] for r in every)


def test_table_sorts_ascending_and_descending(client):
    up = [r['ra'] for r in client.get(
        '/api/table', params={'sort': 'ra'}).json()['rows']]
    down = [r['ra'] for r in client.get(
        '/api/table', params={'sort': 'ra', 'desc': True}).json()['rows']]
    assert up == sorted(up)
    assert down == sorted(down, reverse=True)
    assert set(up) == set(down)


def test_table_can_sort_on_the_derived_dmag_column(client):
    """`mean_abs_dmag` is computed in the query engine, not per row.

    The fixture's sources have no photometry, so every value is NULL -- which
    is the case worth pinning: a column that is empty for this release must
    still be a legal sort rather than a 422.
    """
    body = client.get('/api/table', params={'sort': 'mean_abs_dmag',
                                            'desc': True}).json()
    assert len(body['rows']) == len(UUIDS)


def test_table_rejects_a_sort_on_an_unknown_column(client):
    assert client.get('/api/table',
                      params={'sort': 'not_a_column'}).status_code == 422


# ── the derived classification column ─────────────────────────────────────────

def test_classification_sql_matches_the_python_implementation():
    """The filterable column has to say what the table's column says.

    Built as SQL so it can be filtered and sorted across the whole catalog
    rather than within a fetched window. Checked here against `classes_of`
    on the cases that actually differ -- a JSON list, a matched-but-untyped
    catalog, blanks in their several spellings -- because "identical on the
    real catalog" is a claim that only holds until the next release.
    """
    import duckdb
    import pandas as pd
    from api.catalog_filters import _classification_sql
    from api.derive import CLASS_SOURCES, MATCHED_COLUMN, classes_of

    prefixes = [p for p, _, _ in CLASS_SOURCES]
    columns = [c for _, c, _ in CLASS_SOURCES] + list(MATCHED_COLUMN.values())

    def rendered(cat):
        got = classes_of(cat)
        return ' \u00b7 '.join(f"{p} {'/'.join(got[p])}" for p in prefixes if p in got)

    cases = [
        {},                                                     # nothing at all
        {'sdss_class': 'QSO', 'sdss_matched': True},
        {'tns_matched': True, 'tns_type': None},                # matched, untyped
        {'tns_matched': True, 'tns_type': '<NA>'},              # blank-as-text
        {'tns_matched': True, 'tns_type': 'nan'},
        {'simbad_matched': True, 'simbad_otype_label': 'Quasar'},
        {'agn_db_matched': True,
         'agn_db_best_class_all': '["generic_agn", "unknown", "generic_agn"]'},
        {'sdss_class': 'QSO', 'sdss_matched': True,
         'simbad_otype_label': 'Quasar', 'simbad_matched': True,
         'agn_db_best_class_all': '["generic_agn"]', 'agn_db_matched': True},
        {'sdss_matched': 'true', 'sdss_class': 'GALAXY'},       # flag as text
    ]
    frame = pd.DataFrame([{c: case.get(c) for c in columns} for case in cases],
                         columns=columns).astype(object)
    con = duckdb.connect()
    con.register('cases', frame)
    sql = _classification_sql(set(columns))
    got = [r[0] for r in con.execute(f'SELECT {sql} FROM cases').fetchall()]
    con.close()

    for case, produced in zip(cases, got):
        assert (produced or '') == rendered(case), case


def test_classification_is_null_when_nothing_typed_the_source():
    """Not the empty string: `is_null` is how you ask for "unclassified", and
    a blank would report the column as fully populated."""
    import duckdb
    import pandas as pd
    from api.catalog_filters import _classification_sql
    from api.derive import CLASS_SOURCES, MATCHED_COLUMN

    columns = [c for _, c, _ in CLASS_SOURCES] + list(MATCHED_COLUMN.values())
    frame = pd.DataFrame([{c: None for c in columns}], columns=columns).astype(object)
    con = duckdb.connect()
    con.register('cases', frame)
    got = con.execute(f'SELECT {_classification_sql(set(columns))} FROM cases').fetchone()
    con.close()
    assert got[0] is None


def test_classification_is_skipped_when_the_release_lacks_its_columns():
    """Same rule as `mean_abs_dmag`: a missing column means no value, not a
    view that fails to define."""
    from api.catalog_filters import _classification_sql
    assert _classification_sql(set()) == 'CAST(NULL AS VARCHAR)'


# ── redshifts ─────────────────────────────────────────────────────────────────

def test_redshifts_are_keyed_like_classes():
    """The chip pairs the two by display prefix, so the keys must line up."""
    from api.derive import REDSHIFT_COLUMN, redshifts_of
    assert set(REDSHIFT_COLUMN) <= {p for p, _, _ in CLASS_SOURCES}
    cat = {'sdss_z': 2.4737, 'desi_z': 2.4741, 'simbad_z': 2.475,
           'agn_db_best_z_merged': 2.4737, 'tns_redshift': 0.06}
    assert redshifts_of(cat) == {'SDSS': 2.4737, 'DESI': 2.4741, 'SIMBAD': 2.475,
                                 'AGN-DB': 2.4737, 'TNS': 0.06}


def test_a_catalog_without_a_redshift_is_absent_not_zero():
    """SIMBAD types ~6,000 sources it has no redshift for, and TNS all but 182
    of its matches. Defaulting those to 0.0 would render `(z=0.0000)`."""
    from api.derive import redshifts_of
    assert redshifts_of({'simbad_z': None, 'tns_redshift': float('nan')}) == {}


def test_a_negative_redshift_is_kept():
    """A few hundred SIMBAD/SDSS/DESI entries are blueshifted galactic
    objects; dropping them would misreport what the catalog says."""
    from api.derive import redshifts_of
    assert redshifts_of({'simbad_z': -0.00255})['SIMBAD'] == -0.00255


def test_a_redshift_of_zero_survives():
    """0.0 is a value, and `if z:` would swallow it."""
    from api.derive import redshifts_of
    assert redshifts_of({'desi_z': 0.0}) == {'DESI': 0.0}


def test_redshifts_parse_text_cells():
    """The catalog blob round-trips through JSON, so a numeric column can
    arrive as a string, and the blank sentinels as text."""
    from api.derive import redshifts_of
    assert redshifts_of({'sdss_z': '1.5', 'desi_z': '<NA>',
                         'simbad_z': 'nan', 'tns_redshift': ''}) == {'SDSS': 1.5}


def test_table_filters_by_followup_status(client):
    """`none` is the complement of the rest, not a stored value."""
    from tests.conftest import UUIDS
    client.put(f'/api/sources/{UUIDS[0]}/annotations/adam',
               json={'label': 'SN', 'status': 'queued'})
    total = client.get('/api/table?limit=1').json()['total']

    queued = client.get('/api/table?limit=1&status_in=queued').json()['total']
    observed = client.get('/api/table?limit=1&status_in=observed').json()['total']
    none = client.get('/api/table?limit=1&status_in=none').json()['total']
    assert queued == 1 and observed == 0
    assert queued + none == total, 'none must be everything else'


def test_table_rejects_an_unknown_status(client):
    assert client.get('/api/table?limit=1&status_in=observer').status_code == 422


def test_status_filter_matches_a_source_with_no_label(client):
    """A source can be queued without being filed, which `_label_filter`
    would have excluded."""
    from tests.conftest import UUIDS
    client.put(f'/api/sources/{UUIDS[1]}/annotations/adam',
               json={'status': 'to queue'})
    body = client.get('/api/table?limit=10&status_in=to+queue').json()
    assert [r['source_uuid'] for r in body['rows']] == [UUIDS[1]]


def test_labeled_is_any_author_and_complements_unlabeled(client):
    """'Labeled' must count anyone's verdict, not just the caller's, and the
    two cuts must partition the catalog."""
    from tests.conftest import UUIDS
    client.put(f'/api/sources/{UUIDS[0]}/annotations/beth', json={'label': 'SN'})
    total = client.get('/api/table?limit=1').json()['total']

    get = lambda st: client.get(f'/api/table?limit=50&label_state={st}&author=adam').json()
    labeled, unlabeled = get('Labeled'), get('Unlabeled')
    assert [r['source_uuid'] for r in labeled['rows']] == [UUIDS[0]], \
        "beth's verdict must count for adam"
    assert labeled['total'] + unlabeled['total'] == total
    assert UUIDS[0] not in [r['source_uuid'] for r in unlabeled['rows']]

    # ... and it is not the same question as 'Labeled by me'
    assert get('Labeled by me')['total'] == 0


def test_a_blank_label_does_not_count_as_labeled(client):
    from tests.conftest import UUIDS
    client.put(f'/api/sources/{UUIDS[0]}/annotations/adam', json={'label': '   '})
    assert client.get('/api/table?limit=1&label_state=Labeled').json()['total'] == 0


# ── bucket include / exclude ──────────────────────────────────────────────────

def _file(client, uuid, author, label):
    client.put(f'/api/sources/{uuid}/annotations/{author}', json={'label': label})


def test_bucket_in_is_any_author(client):
    from tests.conftest import UUIDS
    _file(client, UUIDS[0], 'beth', 'Trash')
    body = client.get('/api/table?limit=50&bucket_in=Trash&author=adam').json()
    assert [r['source_uuid'] for r in body['rows']] == [UUIDS[0]]


def test_several_includes_are_an_or(client):
    from tests.conftest import UUIDS
    _file(client, UUIDS[0], 'adam', 'Trash')
    _file(client, UUIDS[1], 'beth', 'Interesting')
    body = client.get(
        '/api/table?limit=50&bucket_in=Trash&bucket_in=Interesting').json()
    assert sorted(r['source_uuid'] for r in body['rows']) == sorted(UUIDS[:2])


def test_bucket_out_removes_the_source(client):
    from tests.conftest import UUIDS
    _file(client, UUIDS[0], 'adam', 'Trash')
    total = client.get('/api/table?limit=1').json()['total']
    body = client.get('/api/table?limit=50&bucket_out=Trash').json()
    assert body['total'] == total - 1
    assert UUIDS[0] not in [r['source_uuid'] for r in body['rows']]


def test_include_and_exclude_combine_with_and(client):
    """A source filed under both is excluded: exclude wins."""
    from tests.conftest import UUIDS
    _file(client, UUIDS[0], 'adam', 'Interesting')
    _file(client, UUIDS[0], 'beth', 'Trash')
    _file(client, UUIDS[1], 'adam', 'Interesting')
    body = client.get(
        '/api/table?limit=50&bucket_in=Interesting&bucket_out=Trash').json()
    assert [r['source_uuid'] for r in body['rows']] == [UUIDS[1]]


def test_an_unknown_bucket_matches_nothing(client):
    """Rather than being ignored, which would silently widen the cut."""
    assert client.get('/api/table?limit=1&bucket_in=NoSuchBucket').json()['total'] == 0


def test_an_unknown_exclude_removes_nothing(client):
    total = client.get('/api/table?limit=1').json()['total']
    assert client.get(
        '/api/table?limit=1&bucket_out=NoSuchBucket').json()['total'] == total
