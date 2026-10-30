"""UUID assignment and the identity map.

`identity.parquet` is the only artefact that cannot be regenerated: it is the
sole link between an annotation and a source across rebuilds. These pin the
rules that keep that link intact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from source_db.identity import (
    IDENTITY_MATCH_ARCSEC, assign_uuids, coord_slugs, empty_identity,
    update_identity,
)

ARCSEC = 1 / 3600


def build(positions, release=1, identity=None):
    """Run one build's worth of assignment + fold-in."""
    ra = np.array([p[0] for p in positions], dtype=float)
    dec = np.array([p[1] for p in positions], dtype=float)
    identity = empty_identity() if identity is None else identity
    uuids, is_new = assign_uuids(ra, dec, identity)
    slugs = coord_slugs(ra, dec)
    return uuids, update_identity(identity, uuids, is_new, ra, dec, slugs, release)


# ── carrying UUIDs across rebuilds ────────────────────────────────────────────

def test_a_source_at_the_same_position_keeps_its_uuid():
    first, ident = build([(10.0, 20.0)])
    second, _ = build([(10.0, 20.0)], release=2, identity=ident)
    assert second[0] == first[0]


def test_a_source_that_drifted_within_the_radius_keeps_its_uuid():
    first, ident = build([(10.0, 20.0)])
    drift = 0.4 * ARCSEC          # inside IDENTITY_MATCH_ARCSEC (0.5")
    second, _ = build([(10.0, 20.0 + drift)], release=2, identity=ident)
    assert second[0] == first[0]


def test_a_source_beyond_the_radius_is_new():
    first, ident = build([(10.0, 20.0)])
    second, _ = build([(10.0, 20.0 + 5 * ARCSEC)], release=2, identity=ident)
    assert second[0] != first[0]


def test_coincident_rows_recover_the_same_set_of_uuids():
    """The catalog has 215 groups of byte-identical positions; a plain
    nearest-neighbour match would give them all one UUID."""
    first, ident = build([(10.0, 20.0), (10.0, 20.0), (10.0, 20.0)])
    assert len(set(first)) == 3
    second, _ = build([(10.0, 20.0)] * 3, release=2, identity=ident)
    assert set(second) == set(first)


# ── retirement ────────────────────────────────────────────────────────────────

def test_a_source_absent_from_a_build_is_retired_not_deleted():
    _, ident = build([(10.0, 20.0), (11.0, 21.0)])
    _, ident = build([(10.0, 20.0)], release=2, identity=ident)
    assert len(ident) == 2, 'nothing is ever deleted'
    gone = ident[ident['ra'] == 11.0].iloc[0]
    assert gone['retired_at'] == 2


def test_a_returning_source_reclaims_its_uuid():
    """The trapdoor: retirement excluded a UUID from matching, so a source
    that dropped out and came back got a new one and orphaned its labels."""
    first, ident = build([(10.0, 20.0), (11.0, 21.0)])
    _, ident = build([(10.0, 20.0)], release=2, identity=ident)          # drops out
    third, ident = build([(10.0, 20.0), (11.0, 21.0)], release=3, identity=ident)
    assert third[1] == first[1], 'the returning source must keep its UUID'


def test_a_reclaimed_uuid_is_un_retired():
    _, ident = build([(10.0, 20.0), (11.0, 21.0)])
    _, ident = build([(10.0, 20.0)], release=2, identity=ident)
    _, ident = build([(10.0, 20.0), (11.0, 21.0)], release=3, identity=ident)
    back = ident[ident['ra'] == 11.0].iloc[0]
    assert pd.isna(back['retired_at'])
    assert back['last_release'] == 3


def test_a_live_row_outranks_a_nearer_retired_one():
    """Greedy-by-separation over one pool would let a dead UUID win and push
    the live source onto a fresh one."""
    first, ident = build([(10.0, 20.0)])                      # A, lives at 20.0
    _, ident = build([(10.0, 20.0), (10.0, 20.0 + 0.3 * ARCSEC)],
                     release=2, identity=ident)               # B appears nearby
    # B retires; A is still live and slightly further from the new position
    _, ident = build([(10.0, 20.0)], release=3, identity=ident)
    live_uuid = ident[pd.isna(ident['retired_at'])].iloc[0]['source_uuid']
    assert live_uuid == first[0]
    # A position between the two: the live row must claim it
    got, _ = build([(10.0, 20.0 + 0.1 * ARCSEC)], release=4, identity=ident)
    assert got[0] == first[0]


# ── release numbering ─────────────────────────────────────────────────────────

def test_first_and_last_release_are_recorded():
    _, ident = build([(10.0, 20.0)], release=4)
    row = ident.iloc[0]
    assert row['first_release'] == 4 and row['last_release'] == 4


# ── sanity ────────────────────────────────────────────────────────────────────

def test_uuids_are_unique_within_a_build():
    uuids, _ = build([(10.0, 20.0), (30.0, 40.0), (50.0, -10.0)])
    assert len(set(uuids)) == 3


def test_non_finite_positions_get_a_new_uuid_rather_than_crashing():
    uuids, _ = build([(10.0, 20.0), (float('nan'), 20.0)])
    assert all(u is not None for u in uuids)
    assert len(set(uuids)) == 2


def test_the_match_radius_is_the_documented_one():
    assert IDENTITY_MATCH_ARCSEC == pytest.approx(0.5)
