"""Stable `source_uuid` assignment and the identity map.

`identity.parquet` is the one artifact in the database that **cannot be
regenerated**. Everything else is a pure function of the pipeline output; this
is the only link between an annotation and a source across releases.

Why the UUID is random rather than derived
------------------------------------------
There is no stable input to hash. The only identifier column,
`PSTARR_PanSTARR_ID`, is non-unique (151,551 / 158,876) and absent for ZTF-only
sources. Position is not stable across pipeline changes -- successive
`filter_results_*` runs differ in astrometric handling, so centroids move. And
the measurement columns that do discriminate are exactly the ones that change
between builds. So UUIDs are assigned once and remembered, then carried forward
on rebuild by positional crossmatch.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Sequence

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, search_around_sky

from Source_Analysis.coord_utils import coord_stem
from source_db.paths import DEFAULT_OUT_DIR


# Slug precision. 6 dp (~4 mas) rather than the 4 dp used for filenames: at 4 dp
# 3,673 genuinely separated sources collide, which would make the duplicate view
# group unrelated objects. At 6 dp only the 215 truly coincident groups remain.
SLUG_PRECISION = 6


# Radius for carrying UUIDs across rebuilds. Generous enough to absorb the
# astrometric drift between pipeline versions, tight enough not to merge
# neighbours -- the closest distinct sources in the catalog sit ~0.09" apart, so
# this is not a safe radius for *identifying* sources, only for re-finding ones
# already known at a nearly identical position.
IDENTITY_MATCH_ARCSEC = 0.5


IDENTITY_COLUMNS = ['source_uuid', 'ra', 'dec', 'coord_slug',
                    'first_release', 'last_release', 'retired_at']


def coord_slug(ra: float, dec: float) -> str:
    """Display/URL slug for a position. Not unique -- see the module docstring."""
    return coord_stem(ra, dec, precision=SLUG_PRECISION)


def coord_slugs(ra: Sequence[float], dec: Sequence[float]) -> np.ndarray:
    """Vectorised `coord_slug`."""
    return np.array([coord_slug(a, b) for a, b in zip(ra, dec)], dtype=object)


def empty_identity() -> pd.DataFrame:
    """An identity table with no rows, for a first build."""
    return pd.DataFrame({
        'source_uuid': pd.Series(dtype=object),
        'ra': pd.Series(dtype=float),
        'dec': pd.Series(dtype=float),
        'coord_slug': pd.Series(dtype=object),
        'first_release': pd.Series(dtype='Int64'),
        'last_release': pd.Series(dtype='Int64'),
        'retired_at': pd.Series(dtype='Int64'),
    })


def load_identity(out_dir: Path = DEFAULT_OUT_DIR) -> pd.DataFrame:
    """The persisted identity table, or an empty one if this is the first build."""
    path = Path(out_dir) / 'identity.parquet'
    if not path.exists():
        return empty_identity()
    return pd.read_parquet(path)


def assign_uuids(
    ra: Sequence[float],
    dec: Sequence[float],
    identity: pd.DataFrame,
    radius_arcsec: float = IDENTITY_MATCH_ARCSEC,
) -> tuple[np.ndarray, np.ndarray]:
    """Give every input position a UUID, reusing `identity` wherever possible.

    Returns `(uuids, is_new)`, both length-N in input order.

    Matching is **one-to-one and greedy by separation**: each identity row can
    be claimed by at most one input position. That matters because the catalog
    contains 215 groups of rows sharing a byte-identical position -- a
    nearest-neighbour match would hand all of them the same UUID and violate the
    primary key. With one-to-one claiming, N coincident rows recover the N UUIDs
    previously assigned at that position. Which row gets which UUID is arbitrary
    for an exact positional tie, but the *set* is preserved, and those rows are
    distinguished by measurements rather than position anyway.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    n = len(ra)
    uuids = np.array([None] * n, dtype=object)

    # Live rows first, then retired ones for whatever is still unclaimed.
    #
    # Two passes rather than one pool, because the claiming is greedy by
    # separation: in a single pass a retired row 0.1" away would outrank a live
    # row 0.2" away, resurrecting a dead UUID *and* pushing the live source onto
    # a new one. Live always wins; a retired UUID is only reused when nothing
    # live is in range.
    #
    # Retired rows have to be candidates at all because retirement would
    # otherwise be a trapdoor: `update_identity` clears `retired_at` for any
    # UUID it sees again, but a UUID excluded from matching can never be seen,
    # so a source that drops out of one pipeline run and returns in the next
    # got a fresh UUID and silently orphaned every annotation filed against it.
    if len(identity):
        is_retired = identity['retired_at'].notna().to_numpy()
        finite = np.isfinite(ra) & np.isfinite(dec)
        finite_idx = np.flatnonzero(finite)
        new_coords = SkyCoord(ra[finite] * u.deg, dec[finite] * u.deg)

        for pool in (identity[~is_retired], identity[is_retired]):
            if not len(pool) or all(u_ is not None for u_ in uuids):
                continue
            id_coords = SkyCoord(pool['ra'].to_numpy() * u.deg,
                                 pool['dec'].to_numpy() * u.deg)
            # Every candidate pair within the radius, not just nearest, so that
            # a second coincident row can still claim a second identity entry.
            i_new, i_id, sep, _ = search_around_sky(
                new_coords, id_coords, radius_arcsec * u.arcsec)

            id_uuid = pool['source_uuid'].to_numpy()
            taken_id = np.zeros(len(pool), dtype=bool)
            for k in np.argsort(sep.arcsec):      # closest pairs win
                src, tgt = finite_idx[i_new[k]], i_id[k]
                if uuids[src] is None and not taken_id[tgt]:
                    uuids[src] = id_uuid[tgt]
                    taken_id[tgt] = True

    is_new = np.array([u_ is None for u_ in uuids], dtype=bool)
    for i in np.flatnonzero(is_new):
        uuids[i] = str(uuid.uuid4())
    return uuids, is_new


def update_identity(
    identity: pd.DataFrame,
    uuids: np.ndarray,
    is_new: np.ndarray,
    ra: Sequence[float],
    dec: Sequence[float],
    slugs: np.ndarray,
    release_id: int,
) -> pd.DataFrame:
    """Fold this build's assignments into the identity table.

    Rows seen in this build get `last_release` bumped and `retired_at` cleared;
    live rows absent from it are retired. Nothing is ever deleted, because an
    annotation may still point at a retired source.
    """
    identity = identity.copy() if len(identity) else empty_identity()
    seen = set(uuids.tolist())

    if len(identity):
        mask = identity['source_uuid'].isin(seen)
        identity.loc[mask, 'last_release'] = release_id
        identity.loc[mask, 'retired_at'] = pd.NA
        retire = (~mask) & identity['retired_at'].isna()
        identity.loc[retire, 'retired_at'] = release_id

    if is_new.any():
        idx = np.flatnonzero(is_new)
        identity = pd.concat([identity, pd.DataFrame({
            'source_uuid': uuids[idx],
            'ra': np.asarray(ra, dtype=float)[idx],
            'dec': np.asarray(dec, dtype=float)[idx],
            'coord_slug': slugs[idx],
            'first_release': pd.array([release_id] * len(idx), dtype='Int64'),
            'last_release': pd.array([release_id] * len(idx), dtype='Int64'),
            'retired_at': pd.array([pd.NA] * len(idx), dtype='Int64'),
        })], ignore_index=True)

    return identity[IDENTITY_COLUMNS]
