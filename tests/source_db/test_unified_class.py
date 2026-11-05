"""The pooled classification, ported from `Notebooks/demographics.ipynb`."""
from __future__ import annotations

import pandas as pd
import pytest

from source_db.unified_class import (
    GROUPS, INPUT_COLUMNS, agn_db_groups, simbad_group, stern12_pass,
    unified_classes,
)

BLANK = {'tns_matched': False, 'tns_designation': '', 'tns_type': '',
         'wise_matched': False, 'wise_w1mpro': None, 'wise_w2mpro': None,
         'wise_cc_flags': '', 'wise_w2snr': None, 'wise_nb': None,
         'agn_db_matched': False, 'agn_db_best_class_all': '',
         'sdss_matched': False, 'sdss_classes': '',
         'desi_matched': False, 'desi_spectype': '',
         'simbad_matched': False, 'simbad_otype_path': '',
         'simbad_is_candidate': False}


def classify(**over) -> str:
    row = {**BLANK, **over}
    assert set(row) == set(INPUT_COLUMNS), 'test row must match the inputs'
    return unified_classes(pd.DataFrame([row]))[0]


# ── the no-information groups ─────────────────────────────────────────────────

def test_no_catalog_saw_it_is_unmatched():
    assert classify() == 'Unmatched'


def test_seen_but_untyped_is_unclassified():
    """Distinct from Unmatched: somebody looked and said nothing."""
    assert classify(sdss_matched=True, sdss_classes='') == 'Unclassified'


# ── single votes ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize('col,value,expected', [
    ('sdss_classes', '["QSO"]', 'AGN'),
    ('sdss_classes', '["QSO_BAL"]', 'AGN'),
    ('sdss_classes', '["GALAXY"]', 'Galaxy'),
    ('sdss_classes', '["STAR"]', 'Star'),
])
def test_a_spectrum_classifies(col, value, expected):
    assert classify(sdss_matched=True, **{col: value}) == expected


def test_a_tns_designation_is_a_transient():
    assert classify(tns_matched=True, tns_designation='2020abc') == 'Transient'


def test_tns_typed_agn_names_the_host_not_an_event():
    assert classify(tns_matched=True, tns_designation='2020abc',
                    tns_type='AGN') == 'AGN'


# ── WISE votes only on a full Stern+12 pass ───────────────────────────────────

def _wise(**over):
    base = dict(wise_matched=True, wise_w1mpro=16.0, wise_w2mpro=15.0,
                wise_cc_flags='0000', wise_w2snr=20.0, wise_nb=1)
    return {**base, **over}


def test_a_full_stern12_pass_votes_agn():
    assert classify(**_wise()) == 'AGN'


@pytest.mark.parametrize('spoil', [
    {'wise_w2mpro': 15.6},        # colour 0.4, below the cut
    {'wise_w2snr': 4.0},          # too noisy
    {'wise_nb': 5},               # blended
    {'wise_cc_flags': 'DD00'},    # contaminated in W1/W2
])
def test_a_failed_condition_votes_for_nothing(spoil):
    """Colour alone is not the criterion, and a blue colour is not a class."""
    assert classify(**_wise(**spoil)) == 'Unmatched'


def test_cc_flags_only_in_w3_w4_still_passes():
    """Only the first two characters matter."""
    assert classify(**_wise(wise_cc_flags='00DD')) == 'AGN'


# ── reconciliation ────────────────────────────────────────────────────────────

def test_transient_wins_over_agn():
    """A transient at an AGN is the interesting part."""
    assert classify(tns_matched=True, tns_designation='2020abc',
                    sdss_matched=True, sdss_classes='["QSO"]') == 'Transient'


def test_a_spectrum_outranks_a_generic_star():
    """SIMBAD's plain Star is often an unresolved point source."""
    assert classify(simbad_matched=True, simbad_otype_path='* > Star',
                    sdss_matched=True, sdss_classes='["QSO"]') == 'AGN'


def test_agn_beats_galaxy():
    """SDSS calls any AGN without broad lines GALAXY."""
    assert classify(sdss_matched=True, sdss_classes='["GALAXY"]',
                    simbad_matched=True,
                    simbad_otype_path='G > AGN > QSO') == 'AGN'


def test_spectra_disagreeing_is_a_real_conflict():
    assert classify(sdss_matched=True, sdss_classes='["QSO"]',
                    desi_matched=True, desi_spectype='GALAXY') == 'Multiple Classes'


def test_every_sdss_spectrum_votes():
    """A source's SDSS spectra can disagree; each class they give votes."""
    assert classify(sdss_matched=True, sdss_classes='["QSO", "STAR"]') == 'Multiple Classes'


def test_star_and_galaxy_is_multiple_classes():
    assert classify(simbad_matched=True, simbad_otype_path='* > Star',
                    agn_db_matched=True,
                    agn_db_best_class_all='["galaxy"]') == 'Multiple Classes'


# ── the pieces ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('path,expected', [
    ('G > AGN > QSO', 'AGN'), ('G > GinGroup', 'Galaxy'),
    ('* > SN*', 'Transient'), ('ev > SN*', 'Transient'),
    ('* > Star', 'Star'), ('X', ''), ('Radio', ''),
])
def test_simbad_paths(path, expected):
    assert simbad_group(path) == expected


def test_a_simbad_candidate_does_not_vote():
    assert classify(simbad_matched=True, simbad_otype_path='G > AGN',
                    simbad_is_candidate=True) == 'Unclassified'


@pytest.mark.parametrize('entry,expected', [
    ('["generic_agn"]', {'AGN'}),
    ('["type1", "blazar"]', {'AGN'}),
    ('["unknown"]', set()),          # absence of a class, not one of them
    ('["lens_system"]', set()),
    ('["star", "galaxy"]', {'Star', 'Galaxy'}),
    ('', set()),
])
def test_agn_db_list_votes(entry, expected):
    assert agn_db_groups(entry) == expected


def test_every_row_gets_a_known_group():
    rows = pd.DataFrame([BLANK, {**BLANK, 'sdss_matched': True,
                                 'sdss_classes': '["QSO"]'}])
    assert set(unified_classes(rows)) <= set(GROUPS)


def test_stern12_needs_a_wise_match():
    row = pd.DataFrame([{**BLANK, **_wise(wise_matched=False)}])
    assert not stern12_pass(row)[0]
