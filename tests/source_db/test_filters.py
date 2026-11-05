"""The catalog filter engine.

Built against a small synthetic parquet rather than `Data/db/sources.parquet`,
for the reasons the rest of this suite gives: it runs anywhere, in
milliseconds, and cannot be broken by a half-finished build.

The fixture deliberately mirrors the shape that makes real filtering
error-prone -- columns that are mostly NULL, a mixed-case name, a boolean, and
a text column with a small vocabulary -- because those are what the filter
semantics have to get right.
"""
from __future__ import annotations

import concurrent.futures

import pandas as pd
import pytest

from source_db.serving.filters import (CatalogFilters, FilterError, restrict_to)


# uuid, tier, mag (mixed case), colour (60% null), otype (40% null), matched
ROWS = [
    # source_uuid                             tier    ZTF_gPSFMag  w1w2  otype    matched
    ('00000000-0000-4000-8000-000000000001', '0',      18.0,       1.2,  'Quasar',  True),
    ('00000000-0000-4000-8000-000000000002', '0',      19.5,       0.3,  'Galaxy',  True),
    ('00000000-0000-4000-8000-000000000003', '1',      20.0,       None, 'Star',    False),
    ('00000000-0000-4000-8000-000000000004', '1',      21.0,       None, None,      False),
    ('00000000-0000-4000-8000-000000000005', '1_wide_association',
                                                      22.5,       2.0,  None,      True),
]
COLUMNS = ['source_uuid', 'catalog_tier', 'ZTF_gPSFMag', 'wise_w1w2',
           'simbad_otype_label', 'sdss_matched']
UUIDS = [r[0] for r in ROWS]


@pytest.fixture(scope='module')
def catalog(tmp_path_factory):
    path = tmp_path_factory.mktemp('catalog') / 'sources.parquet'
    pd.DataFrame(ROWS, columns=COLUMNS).to_parquet(path, index=False)
    f = CatalogFilters(path)
    yield f
    f.close()


def uuids_of(result) -> list[str]:
    return result['uuids'] if isinstance(result, dict) else result


# ── introspection ─────────────────────────────────────────────────────────────

def test_columns_report_type_group_and_fill(catalog):
    by_name = {c.name: c for c in catalog.columns()}
    assert by_name['ZTF_gPSFMag'].type == 'number'
    assert by_name['simbad_otype_label'].type == 'text'
    assert by_name['sdss_matched'].type == 'bool'
    assert by_name['wise_w1w2'].group == 'wise'
    assert by_name['catalog_tier'].group == 'provenance'
    # 2 of 5 rows have a colour, 3 of 5 have an otype.
    assert by_name['wise_w1w2'].n_non_null == 3
    assert by_name['simbad_otype_label'].fill == pytest.approx(0.6)


def test_numeric_stats_give_a_slider_its_bounds(catalog):
    s = catalog.stats('ZTF_gPSFMag')
    assert (s['min'], s['max']) == (18.0, 22.5)
    assert s['quantiles']['p50'] == pytest.approx(20.0)


def test_text_stats_give_the_vocabulary(catalog):
    s = catalog.stats('simbad_otype_label')
    assert s['n_distinct'] == 3
    assert {v['value'] for v in s['values']} == {'Quasar', 'Galaxy', 'Star'}
    assert all(v['count'] == 1 for v in s['values'])


def test_an_all_null_column_reports_itself_as_unfilterable(tmp_path):
    path = tmp_path / 'empty.parquet'
    pd.DataFrame({'source_uuid': ['a'], 'dead': [None]}).to_parquet(path, index=False)
    f = CatalogFilters(path)
    try:
        assert not {c.name: c for c in f.columns()}['dead'].filterable
        # No range and no vocabulary -- there is nothing to offer, and asking
        # for a quantile over zero rows is how that turns into a crash. The
        # declared type is whatever the writer chose for an empty column, so
        # it is not part of the contract here.
        stats = f.stats('dead')
        assert stats['n_non_null'] == 0 and stats['fill'] == 0.0
        assert 'min' not in stats and 'values' not in stats
    finally:
        f.close()


# ── operators ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('doc, expected', [
    ({'column': 'catalog_tier', 'op': '=', 'value': '0'}, 2),
    ({'column': 'catalog_tier', 'op': '!=', 'value': '0'}, 3),
    ({'column': 'ZTF_gPSFMag', 'op': '<', 'value': 20.0}, 2),
    ({'column': 'ZTF_gPSFMag', 'op': '<=', 'value': 20.0}, 3),
    ({'column': 'ZTF_gPSFMag', 'op': '>', 'value': 20.0}, 2),
    ({'column': 'ZTF_gPSFMag', 'op': '>=', 'value': 20.0}, 3),
    ({'column': 'ZTF_gPSFMag', 'op': 'between', 'value': [19.0, 21.0]}, 3),
    ({'column': 'catalog_tier', 'op': 'in', 'value': ['1', '1_wide_association']}, 3),
    ({'column': 'catalog_tier', 'op': 'not_in', 'value': ['0']}, 3),
    ({'column': 'simbad_otype_label', 'op': 'contains', 'value': 'ala'}, 1),
    ({'column': 'simbad_otype_label', 'op': 'startswith', 'value': 'Q'}, 1),
    ({'column': 'simbad_otype_label', 'op': 'endswith', 'value': 'r'}, 2),
    ({'column': 'wise_w1w2', 'op': 'is_null'}, 2),
    ({'column': 'wise_w1w2', 'op': 'is_not_null'}, 3),
    ({'column': 'sdss_matched', 'op': 'is_true'}, 3),
    ({'column': 'sdss_matched', 'op': 'is_false'}, 2),
])
def test_each_operator_counts_what_it_should(catalog, doc, expected):
    assert catalog.count(doc) == expected


def test_contains_is_case_insensitive(catalog):
    assert catalog.count(
        {'column': 'simbad_otype_label', 'op': 'contains', 'value': 'QUASAR'}) == 1


def test_values_are_bound_not_interpolated(catalog):
    sql, params = catalog.compile(
        {'column': 'catalog_tier', 'op': '=', 'value': "0' OR 1=1 --"})
    assert '?' in sql and "OR 1=1" not in sql
    assert params == ["0' OR 1=1 --"]
    assert catalog.count(
        {'column': 'catalog_tier', 'op': '=', 'value': "0' OR 1=1 --"}) == 0


# ── boolean structure ─────────────────────────────────────────────────────────

def test_and_or_and_nesting(catalog):
    doc = {'op': 'and', 'clauses': [
        {'column': 'catalog_tier', 'op': 'in', 'value': ['1', '1_wide_association']},
        {'op': 'or', 'clauses': [
            {'column': 'sdss_matched', 'op': 'is_true'},
            {'column': 'ZTF_gPSFMag', 'op': '<', 'value': 20.5},
        ]},
    ]}
    assert catalog.count(doc) == 2          # rows 3 and 5


def test_an_empty_group_does_not_error_mid_edit(catalog):
    """A half-built filter in a UI should show everything, not a red box."""
    assert catalog.count({'op': 'and', 'clauses': []}) == len(ROWS)
    assert catalog.count({'op': 'or', 'clauses': []}) == 0
    assert catalog.count(None) == len(ROWS)


def test_not_keeps_rows_the_inner_clause_could_not_judge(catalog):
    """`NOT (x = 'Star')` must not quietly drop the rows where x is NULL.

    Plain SQL negation leaves NULL as NULL, so a cut meant to remove stars
    would also remove every source SIMBAD never matched -- which on the real
    catalog is 74% of them.
    """
    assert catalog.count(
        {'op': 'not', 'clauses': [
            {'column': 'simbad_otype_label', 'op': '=', 'value': 'Star'}]}) == 4


def test_not_takes_exactly_one_clause(catalog):
    with pytest.raises(FilterError):
        catalog.count({'op': 'not', 'clauses': [
            {'column': 'catalog_tier', 'op': '=', 'value': '0'},
            {'column': 'catalog_tier', 'op': '=', 'value': '1'}]})


# ── the missing-value policy ──────────────────────────────────────────────────

def test_missing_defaults_to_excluding_unmatched_sources(catalog):
    assert catalog.count({'column': 'wise_w1w2', 'op': '>', 'value': 1.0}) == 2


def test_missing_include_keeps_them(catalog):
    assert catalog.count({'column': 'wise_w1w2', 'op': '>', 'value': 1.0,
                          'missing': 'include'}) == 4


def test_an_unknown_missing_policy_is_refused(catalog):
    with pytest.raises(FilterError):
        catalog.count({'column': 'wise_w1w2', 'op': '>', 'value': 1.0,
                       'missing': 'maybe'})


# ── the expression escape hatch ───────────────────────────────────────────────

def test_an_expression_can_span_columns(catalog):
    assert catalog.count({'expr': 'ZTF_gPSFMag - wise_w1w2 > 17'}) == 2


def test_an_expression_may_call_whitelisted_maths(catalog):
    assert catalog.count({'expr': 'abs(wise_w1w2 - 1.2) < 0.01'}) == 1


@pytest.mark.parametrize('expr', [
    "1=1; DROP TABLE catalog",
    "ZTF_gPSFMag > (SELECT max(ZTF_gPSFMag) FROM catalog)",
    "ZTF_gPSFMag > 1 -- comment",
    # `--` and `/* */` tokenise as arithmetic but are comments to the engine,
    # so a validator that let them through would be approving text it had not
    # actually read. Both spellings use only column names, to make the point
    # that the rejection is about the marker, not the words after it.
    "ZTF_gPSFMag > 1 -- ZTF_gPSFMag",
    "ZTF_gPSFMag > 1 /* ZTF_gPSFMag */",
    "read_csv('/etc/passwd') IS NOT NULL",
    "nosuchcolumn > 1",
    "ZTF_gPSFMag > 1 /* block */",
    "$$1$$ = '1'",
])
def test_expressions_that_are_not_arithmetic_over_known_columns_are_refused(
        catalog, expr):
    with pytest.raises(FilterError):
        catalog.count({'expr': expr})


def test_an_expression_must_mention_a_column(catalog):
    with pytest.raises(FilterError):
        catalog.count({'expr': '1 = 1'})


@pytest.mark.parametrize('expr', ['', '   '])
def test_a_blank_expression_matches_everything(catalog, expr):
    """What the builder sends the moment "+ expression" is clicked."""
    assert catalog.count({'expr': expr}) == catalog.count(None)
    assert catalog.count({'op': 'and', 'clauses': [
        {'column': 'wise_w1w2', 'op': '>', 'value': 1.0}, {'expr': expr}]}) == 2


def test_a_very_long_expression_is_refused(catalog):
    with pytest.raises(FilterError):
        catalog.count({'expr': 'ZTF_gPSFMag > 1 AND ' * 500 + 'ZTF_gPSFMag > 1'})


# ── malformed documents ───────────────────────────────────────────────────────

@pytest.mark.parametrize('doc', [
    {'column': 'nope', 'op': '=', 'value': 1},
    {'column': 'catalog_tier', 'op': 'regex', 'value': 'x'},
    {'column': 'catalog_tier', 'op': '='},                     # no value
    {'column': 'ZTF_gPSFMag', 'op': 'between', 'value': [1]},   # not a pair
    {'column': 'catalog_tier', 'op': 'in', 'value': []},        # empty list
    {'op': 'xor', 'clauses': []},
    {'op': 'and', 'clauses': 'not a list'},
    {'column': 'simbad_otype_label', 'op': 'contains', 'value': 7},
    'a bare string',
    42,
])
def test_a_malformed_document_raises_rather_than_matching_everything(catalog, doc):
    with pytest.raises(FilterError):
        catalog.count(doc)


# ── paging and sorting ────────────────────────────────────────────────────────

def test_page_reports_the_total_not_the_page_size(catalog):
    page = catalog.page(None, limit=2, offset=0)
    assert page['total'] == len(ROWS)
    assert len(page['uuids']) == 2


def test_paging_covers_every_row_exactly_once(catalog):
    seen = []
    for off in range(0, len(ROWS), 2):
        seen += catalog.page(None, limit=2, offset=off)['uuids']
    assert seen == sorted(UUIDS)


def test_sorting_on_another_column_is_still_a_total_order(catalog):
    """Ties must break on uuid, or two pages can disagree about one row."""
    asc = catalog.page(None, sort='catalog_tier', limit=10)['uuids']
    desc = catalog.page(None, sort='catalog_tier', desc=True, limit=10)['uuids']
    assert len(asc) == len(desc) == len(ROWS)
    assert set(asc) == set(desc)
    # Within tier '0' the two uuids stay in uuid order in both directions.
    assert asc[:2] == sorted(asc[:2])


def test_nulls_sort_last_in_both_directions(catalog):
    """A sort by redshift should not open with everything that has none."""
    for desc in (False, True):
        order = catalog.page(None, sort='wise_w1w2', desc=desc, limit=10)['uuids']
        missing = {UUIDS[2], UUIDS[3]}
        assert set(order[-2:]) == missing


def test_sorting_on_an_unknown_column_is_refused(catalog):
    with pytest.raises(FilterError):
        catalog.page(None, sort='nope')


def test_sorting_on_an_all_null_column_is_allowed(tmp_path):
    """An empty column is a fact about the release, not a typo.

    It orders fine -- every row ties and the uuid tiebreak decides -- and
    refusing it turned "this build has no photometry" into a 422 on the
    table page's default sort.
    """
    path = tmp_path / 'sparse.parquet'
    pd.DataFrame({'source_uuid': ['b', 'a'], 'dead': [None, None]}).to_parquet(
        path, index=False)
    f = CatalogFilters(path)
    try:
        assert f.page(None, sort='dead')['uuids'] == ['a', 'b']
    finally:
        f.close()


# ── the other readers ─────────────────────────────────────────────────────────

def test_select_returns_only_the_columns_asked_for(catalog):
    rows = catalog.select({'column': 'catalog_tier', 'op': '=', 'value': '0'},
                          ['wise_w1w2'], sort='wise_w1w2')
    assert [set(r) for r in rows] == [{'source_uuid', 'wise_w1w2'}] * 2


def test_select_refuses_an_unknown_column(catalog):
    with pytest.raises(FilterError):
        catalog.select(None, ['nope'])


def test_explain_counts_each_clause_on_its_own(catalog):
    out = catalog.explain({'op': 'and', 'clauses': [
        {'column': 'catalog_tier', 'op': '=', 'value': '1'},
        {'column': 'sdss_matched', 'op': 'is_true'},
    ]})
    assert [d['matches'] for d in out] == [2, 3]


def test_explain_reports_a_bad_clause_instead_of_raising(catalog):
    out = catalog.explain({'op': 'and', 'clauses': [
        {'column': 'catalog_tier', 'op': '=', 'value': '1'},
        {'column': 'nope', 'op': '=', 'value': 1},
    ]})
    assert out[0]['matches'] == 2 and 'error' in out[1]


def test_restrict_to_turns_a_uuid_list_into_a_filter(catalog):
    assert catalog.count(restrict_to(UUIDS[:2])) == 2
    assert catalog.uuids(restrict_to(UUIDS[:2])) == sorted(UUIDS[:2])


def test_uuids_honours_a_limit(catalog):
    assert len(catalog.uuids(None, limit=3)) == 3


# ── concurrency ───────────────────────────────────────────────────────────────

def test_concurrent_queries_do_not_cross_results(catalog):
    """One DuckDB connection shared across threads returns crossed results.

    Measured on the real catalog: 7 of 240 concurrent queries failed or came
    back wrong. `CatalogFilters` runs every query on its own `cursor()`, which
    is the supported way to fan out, so this holds the property in place.
    """
    wanted = {
        '0': 2, '1': 2, '1_wide_association': 1,
    }
    def run(tier):
        return tier, catalog.count(
            {'column': 'catalog_tier', 'op': '=', 'value': tier})

    jobs = list(wanted) * 60
    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        results = list(ex.map(run, jobs))
    assert all(n == wanted[tier] for tier, n in results)
    assert len(results) == len(jobs)


def test_a_missing_catalog_says_which_build_step_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match='build_catalog_tier'):
        CatalogFilters(tmp_path / 'nope.parquet')
