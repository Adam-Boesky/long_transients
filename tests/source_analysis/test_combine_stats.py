"""combine_stats: summing per-field flowchart stats when some fields predate a step."""
import pandas as pd
import pytest

from Source_Analysis.filter_fields import combine_stats


def _stats(rows):
    """rows: (filter, branch, before, after) -> a per-field stats frame with one count column."""
    return pd.DataFrame([{'filter': f, 'branch': b, 'n_g_before': float(x), 'n_g_after': float(y)}
                         for f, b, x, y in rows])


FULL = [('sep', '', 100, 90), ('shape', '', 90, 80),
        ('big_dmag', 'in_ztf', 80, 20), ('nearby', 'in_ztf', 20, 15), ('ps1_region', 'in_ztf', 15, 14),
        ('big_dmag', 'in_both', 80, 30), ('nearby', 'in_both', 30, 25), ('ps1_region', 'in_both', 25, 25)]


def test_missing_step_is_filled_from_its_own_branch():
    # The old field lacks ps1_region on both branches; each fill must use that branch's
    # previous after-count, not the row above it in the template (which crosses branches).
    old = _stats([r for r in FULL if r[0] != 'ps1_region'])
    got = combine_stats([_stats(FULL), old])  # indexed by (filter, branch)
    assert got.loc[('ps1_region', 'in_ztf'), 'n_g_before'] == 15 + 15
    assert got.loc[('ps1_region', 'in_ztf'), 'n_g_after'] == 14 + 15
    assert got.loc[('ps1_region', 'in_both'), 'n_g_before'] == 25 + 25
    assert got.loc[('ps1_region', 'in_both'), 'n_g_after'] == 25 + 25


def test_missing_first_step_of_a_branch_is_an_error():
    old = _stats([r for r in FULL if not (r[0] == 'big_dmag' and r[1] == 'in_both')])
    with pytest.raises(ValueError, match='first step'):
        combine_stats([_stats(FULL), old])
