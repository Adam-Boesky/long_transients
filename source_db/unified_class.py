"""One class per source, pooled from every catalogue's verdict.

Ported from `Notebooks/demographics.ipynb` ("Unified Classification") so the
website and the paper figure cannot drift apart: the notebook should import
`unified_classes` from here rather than keep its own copy.

Six catalogues vote and `_reconcile` picks one group. A WISE match alone is not
a vote -- WISE gives colours, not classes -- so only a full Stern+12 pass
counts. With no votes a source is `Unclassified` if some catalogue saw it and
`Unmatched` if none did.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

#: The seven groups, in the notebook's wedge order.
GROUPS = ('AGN', 'Galaxy', 'Transient', 'Star', 'Multiple Classes',
          'Unclassified', 'Unmatched')

# Spectroscopic classes that map onto a group; anything else says nothing.
_SPEC = {'QSO': 'AGN', 'GALAXY': 'Galaxy', 'STAR': 'Star'}
_SPEC_CATALOGS = ('SDSS', 'DESI')
# TNS types that name the host rather than a transient.
_TNS_AGN = {'AGN', 'QSO'}
_AGN_TYPES = {'generic_agn', 'type1', 'type2', 'blazar'}
_AGN_DB_LABELS = {'star': 'Star', 'galaxy': 'Galaxy'}
# "some catalogue saw it", which separates Unclassified from Unmatched.
_MATCH_COLS = ('tns_matched', 'agn_db_matched', 'sdss_matched',
               'desi_matched', 'simbad_matched')

#: Every column the vote reads.
INPUT_COLUMNS = (
    'tns_matched', 'tns_designation', 'tns_type',
    'wise_matched', 'wise_w1mpro', 'wise_w2mpro', 'wise_cc_flags',
    'wise_w2snr', 'wise_nb',
    'agn_db_matched', 'agn_db_best_class_all',
    'sdss_matched', 'sdss_class', 'desi_matched', 'desi_spectype',
    'simbad_matched', 'simbad_otype_path', 'simbad_is_candidate',
)


def _s(col) -> np.ndarray:
    return np.asarray(pd.Series(col).fillna('').astype(str))


def _f(col) -> np.ndarray:
    return np.asarray(pd.to_numeric(pd.Series(col), errors='coerce'), dtype=float)


def _b(col) -> np.ndarray:
    return np.asarray(pd.Series(col).fillna(False).astype(bool))


def stern12_pass(data) -> np.ndarray:
    """The full Stern+2012 AGN criterion: colour *and* quality.

    Colour alone is not the criterion. A blue colour says "not an AGN by this
    test", which is not a classification, so only a pass votes.
    """
    colour = _f(data['wise_w1mpro']) - _f(data['wise_w2mpro'])
    cc = _s(data['wise_cc_flags'])
    return (_b(data['wise_matched'])
            & (colour >= 0.8)
            & (_f(data['wise_w2snr']) >= 10.0)
            & (_f(data['wise_nb']) <= 2)
            # cc_flags is one character per band in W1 W2 W3 W4 order; only the
            # first two matter, so a source flagged solely at 12 or 22 um is kept.
            & np.array([len(c) >= 2 and c[:2] == '00' for c in cc]))


def simbad_group(path: str) -> str:
    """The group a SIMBAD otype path ('G > AGN > QSO', '* > SN*') belongs to."""
    levels = [p.strip() for p in path.split('>')]
    if 'AGN' in levels:
        return 'AGN'
    if levels[0] == 'G':
        return 'Galaxy'
    if 'SN*' in levels or levels[0] == 'ev':
        return 'Transient'
    if levels[0] == '*':
        return 'Star'
    return ''


def agn_db_groups(entry: str) -> set[str]:
    """What AGN-DB's per-catalogue class list votes for.

    'unknown' and 'lens_system' state no class, so neither votes.
    """
    claims = (set(json.loads(entry)) if entry.startswith('[')
              else ({entry} if entry else set()))
    out = set()
    for c in claims:
        if c in _AGN_TYPES:
            out.add('AGN')
        elif c in _AGN_DB_LABELS:
            out.add(_AGN_DB_LABELS[c])
    return out


def _reconcile(votes: dict[str, set[str]]) -> str:
    """One group from one source's {catalogue: groups}, or '' if nobody voted.

    In order: a Transient wins, because a transient at an AGN or in a galaxy
    is the interesting part. Spectra outrank a generic 'star' -- SIMBAD's
    plain Star is often an unresolved point source and AGN-DB's a flag. AGN
    beats Galaxy, because SDSS/DESI call any AGN without broad lines GALAXY,
    unless the spectra themselves disagree, which is a real conflict.
    """
    groups = set().union(*votes.values()) if votes else set()
    spec = set().union(*(votes.get(c, set()) for c in _SPEC_CATALOGS)) if votes else set()
    if 'Transient' in groups:
        return 'Transient'
    if spec and 'Star' not in spec:
        groups.discard('Star')
    if {'AGN', 'Galaxy'} <= groups and not {'AGN', 'Galaxy'} <= spec:
        groups.discard('Galaxy')
    if not groups:
        return ''
    return next(iter(groups)) if len(groups) == 1 else 'Multiple Classes'


def unified_classes(data) -> np.ndarray:
    """One group per row, as an array of strings. Never empty: see GROUPS."""
    n = len(data[next(iter(INPUT_COLUMNS))])
    votes: list[dict[str, set[str]]] = [{} for _ in range(n)]

    def vote(i: int, catalog: str, group: str) -> None:
        votes[i].setdefault(catalog, set()).add(group)

    # TNS: a designation means somebody reported a transient here, unless TNS
    # typed it as an AGN or QSO, which names what is there.
    tns_type = _s(data['tns_type'])
    for i in np.flatnonzero(_s(data['tns_designation']) != ''):
        vote(i, 'TNS', 'AGN' if tns_type[i] in _TNS_AGN else 'Transient')

    for i in np.flatnonzero(stern12_pass(data)):
        vote(i, 'WISE', 'AGN')

    # Each contributing catalogue votes separately, so a source can disagree
    # with itself.
    agn_cls = _s(data['agn_db_best_class_all'])
    for i in np.flatnonzero(_b(data['agn_db_matched'])):
        for g in agn_db_groups(agn_cls[i]):
            vote(i, 'AGN-DB', g)

    for catalog, matched_col, class_col in (
            ('SDSS', 'sdss_matched', 'sdss_class'),
            ('DESI', 'desi_matched', 'desi_spectype')):
        spec = _s(data[class_col])
        for i in np.flatnonzero(_b(data[matched_col])):
            if spec[i] in _SPEC:
                vote(i, catalog, _SPEC[spec[i]])

    # Candidates ('AG?', 'Q?') do not vote.
    path = _s(data['simbad_otype_path'])
    candidate = _b(data['simbad_is_candidate'])
    for i in np.flatnonzero(_b(data['simbad_matched']) & ~candidate):
        if group := simbad_group(path[i]):
            vote(i, 'SIMBAD', group)

    seen = np.logical_or.reduce([_b(data[c]) for c in _MATCH_COLS])
    return np.array([_reconcile(v) or ('Unclassified' if s else 'Unmatched')
                     for v, s in zip(votes, seen)], dtype=object)
