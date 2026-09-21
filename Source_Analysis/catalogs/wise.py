"""AllWISE crossmatching via IRSA, with a local cache.

AllWISE is ~750M sources, far too large to slim into a local parquet the way
`sdss`, `desi` and `tns` are handled. Instead this queries IRSA's TAP service
with an uploaded position list -- the match happens server-side and only the
matched rows come back -- and caches the answers in
`Data/catalogs/allwise_matches.parquet`.

Two consequences of that design worth knowing:

* Positions are queried at a fixed `CACHE_RADIUS_ARCSEC`, wider than any match
  radius you would actually use, and the requested radius is applied locally.
  So changing the match radius never invalidates the cache, and each position
  is queried exactly once, ever.
* Because the cache stores every AllWISE source near a position rather than
  just the nearest, the count of neighbours inside the WISE beam comes for
  free. At AllWISE's source density that is only ~0.4 rows per position.

Colours are Vega, as AllWISE reports them, which is also the system Stern et
al. (2012) defined the W1-W2 >= 0.8 AGN criterion in.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import requests
from astropy.table import Table

_DATA_DIR = Path(__file__).resolve().parents[2] / 'Data' / 'catalogs'
CACHE_PARQUET = _DATA_DIR / 'allwise_matches.parquet'

IRSA_TAP = 'https://irsa.ipac.caltech.edu/TAP/sync'
IRSA_TABLE = 'allwise_p3as_psd'

# Every position is cached out to this radius, so any smaller match radius can
# be served from the cache without re-querying.
CACHE_RADIUS_ARCSEC = 10.0
# WISE's W1 beam is ~6"; neighbours inside this radius share the same blend.
BEAM_ARCSEC = 8.0
# Positions per TAP request. IRSA handles this comfortably in sync mode.
CHUNK = 1000

# (IRSA column, output column).
_COLUMNS: list[tuple[str, str]] = [
    ('designation', 'wise_designation'),
    ('ra', 'wise_ra'),
    ('dec', 'wise_dec'),
    ('w1mpro', 'wise_w1mpro'),
    ('w1sigmpro', 'wise_w1sigmpro'),
    ('w2mpro', 'wise_w2mpro'),
    ('w2sigmpro', 'wise_w2sigmpro'),
    ('w3mpro', 'wise_w3mpro'),
    ('w3sigmpro', 'wise_w3sigmpro'),
    ('w4mpro', 'wise_w4mpro'),
    ('w4sigmpro', 'wise_w4sigmpro'),
    ('w1snr', 'wise_w1snr'),
    ('w2snr', 'wise_w2snr'),
    ('ph_qual', 'wise_ph_qual'),
    ('cc_flags', 'wise_cc_flags'),
    ('ext_flg', 'wise_ext_flg'),
    ('var_flg', 'wise_var_flg'),
    ('nb', 'wise_nb'),
    ('na', 'wise_na'),
]


# Cache dtypes are pinned explicitly. Without this, a batch whose positions all
# miss writes all-null payload columns, parquet types them float64, and the next
# batch carrying real strings fails to append.
_TEXT = {'key', 'wise_designation', 'wise_ph_qual', 'wise_cc_flags', 'wise_var_flg'}
_INT = {'wise_ext_flg', 'wise_nb', 'wise_na'}


def _cache_dtypes() -> dict[str, str]:
    cols = ['key', 'query_ra', 'query_dec', 'sep_arcsec'] + [dst for _, dst in _COLUMNS]
    return {c: 'string' if c in _TEXT else 'Int64' if c in _INT else 'float64' for c in cols}


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Force the cache schema, so appends never hit a dtype conflict."""
    dtypes = _cache_dtypes()
    for col, dtype in dtypes.items():
        if col not in df.columns:
            # No fill value: an empty Series of the dtype is null-filled, and
            # picks the right null (NaN vs pd.NA) for that dtype on its own.
            df[col] = pd.Series(index=df.index, dtype=dtype)
        elif dtype == 'Int64':
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        else:
            df[col] = df[col].astype(dtype)
    return df[list(dtypes)]


def _key(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Stable cache key for a position. 1e-7 deg is ~0.4 mas -- far finer than
    any real separation between distinct sources, and stable through parquet."""
    return np.char.add(np.char.add(np.round(ra, 7).astype('U16'), '_'),
                       np.round(dec, 7).astype('U16'))


def _query_irsa(ra: np.ndarray, dec: np.ndarray) -> pd.DataFrame:
    """Upload positions to IRSA and return every AllWISE source within
    CACHE_RADIUS_ARCSEC of each, tagged with the input index in `qidx`."""
    selects = ', '.join(f'w.{src}' for src, _ in _COLUMNS)
    adql = (
        f'SELECT u.qidx, {selects} '
        f'FROM TAP_UPLOAD.pos AS u, {IRSA_TABLE} AS w '
        f"WHERE CONTAINS(POINT('ICRS', w.ra, w.dec), "
        f"CIRCLE('ICRS', u.qra, u.qdec, {CACHE_RADIUS_ARCSEC / 3600.0})) = 1"
    )

    frames = []
    for start in range(0, len(ra), CHUNK):
        sl = slice(start, start + CHUNK)
        buf = io.BytesIO()
        Table({'qidx': np.arange(len(ra))[sl], 'qra': ra[sl], 'qdec': dec[sl]}).write(
            buf, format='votable'
        )
        buf.seek(0)
        resp = requests.post(
            IRSA_TAP,
            data={'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv',
                  'UPLOAD': 'pos,param:upfile', 'QUERY': adql},
            files={'upfile': ('pos.xml', buf, 'application/x-votable+xml')},
            timeout=600,
        )
        resp.raise_for_status()
        if resp.text.lstrip().startswith('<'):
            raise RuntimeError(f'IRSA returned an error document: {resp.text[:400]}')
        frames.append(pd.read_csv(io.StringIO(resp.text)))
        print(f'  queried {min(start + CHUNK, len(ra)):,} / {len(ra):,} positions')

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out.rename(columns=dict(_COLUMNS))


def _angular_sep_arcsec(ra1, dec1, ra2, dec2) -> np.ndarray:
    """Great-circle separation in arcsec, vectorised over equal-length arrays."""
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    return np.degrees(2 * np.arcsin(np.sqrt(
        np.sin((d2 - d1) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2
    ))) * 3600.0


def _load_cache() -> pd.DataFrame:
    if CACHE_PARQUET.exists():
        return _coerce(pd.read_parquet(CACHE_PARQUET))
    return _coerce(pd.DataFrame(index=pd.RangeIndex(0)))


def update_cache(ra: Sequence[float], dec: Sequence[float]) -> int:
    """Query IRSA for any of these positions not already cached. Returns the
    number of newly queried positions."""
    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    finite = np.isfinite(ra) & np.isfinite(dec)

    cache = _load_cache()
    known = set(cache['key']) if len(cache) else set()
    keys = _key(ra, dec)
    todo = np.array([k not in known for k in keys]) & finite
    if not todo.any():
        return 0

    # Unique positions only -- duplicated coordinates would be queried twice.
    _, first = np.unique(keys[todo], return_index=True)
    idx = np.flatnonzero(todo)[np.sort(first)]
    print(f'querying IRSA for {len(idx):,} new positions ({int(todo.sum()) - len(idx)} duplicates skipped)')

    found = _query_irsa(ra[idx], dec[idx])

    rows = []
    if len(found):
        found['query_ra'] = ra[idx][found['qidx'].to_numpy()]
        found['query_dec'] = dec[idx][found['qidx'].to_numpy()]
        found['sep_arcsec'] = _angular_sep_arcsec(
            found['query_ra'].to_numpy(), found['query_dec'].to_numpy(),
            found['wise_ra'].to_numpy(), found['wise_dec'].to_numpy())
        found['key'] = _key(found['query_ra'].to_numpy(), found['query_dec'].to_numpy())
        rows.append(found.drop(columns=['qidx']))

    # Record positions with no AllWISE source nearby, so they are never re-queried.
    hit_keys = set(rows[0]['key']) if rows else set()
    misses = [k for k in keys[idx] if k not in hit_keys]
    if misses:
        blank = pd.DataFrame({'key': misses})
        blank['query_ra'] = [float(k.split('_')[0]) for k in misses]
        blank['query_dec'] = [float(k.split('_')[1]) for k in misses]
        rows.append(blank)

    CACHE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    merged = _coerce(pd.concat([cache] + [_coerce(r) for r in rows], ignore_index=True))
    merged.to_parquet(CACHE_PARQUET, index=False)
    return len(idx)


def cone_search_many(
    ra: Sequence[float],
    dec: Sequence[float],
    radius_arcsec: float = 2.0,
    beam_arcsec: float = BEAM_ARCSEC,
    refresh: bool = True,
) -> pd.DataFrame:
    """Nearest AllWISE source within `radius_arcsec` of each input position.

    One row per input, in input order, matching the contract of the other
    catalog modules. `refresh` queries IRSA for positions not yet cached; set
    it False to work purely offline from what is already cached.

    `wise_n_in_beam` counts AllWISE sources within `beam_arcsec` -- more than
    one means the photometry is blended and the colour should not be trusted.
    """
    if radius_arcsec > CACHE_RADIUS_ARCSEC:
        raise ValueError(
            f'radius_arcsec={radius_arcsec} exceeds the cached radius '
            f'({CACHE_RADIUS_ARCSEC}); widen CACHE_RADIUS_ARCSEC and rebuild the cache'
        )
    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    if ra.shape != dec.shape:
        raise ValueError(f'ra and dec must be the same length, got {ra.shape} and {dec.shape}')

    if refresh:
        update_cache(ra, dec)

    cache = _load_cache()
    payload = [dst for _, dst in _COLUMNS]
    by_key: dict[str, pd.DataFrame] = (
        {k: g for k, g in cache.groupby('key', sort=False)} if len(cache) else {}
    )

    records, seps, n_beam = [], [], []
    blank = {c: None for c in payload}
    for k in _key(ra, dec):
        g = by_key.get(k)
        if g is None or g['wise_designation'].isna().all():
            records.append(blank)
            seps.append(np.nan)
            n_beam.append(0 if g is not None else pd.NA)
            continue
        g = g[g['wise_designation'].notna()]
        n_beam.append(int((g['sep_arcsec'] <= beam_arcsec).sum()))
        near = g[g['sep_arcsec'] <= radius_arcsec]
        if near.empty:
            records.append(blank)
            seps.append(np.nan)
        else:
            best = near.loc[near['sep_arcsec'].idxmin()]
            records.append({c: best[c] for c in payload})
            seps.append(float(best['sep_arcsec']))

    out = pd.DataFrame(records, columns=payload)
    seps = np.asarray(seps, dtype=float)
    out.insert(0, 'wise_matched', np.isfinite(seps))
    out.insert(1, 'wise_sep_arcsec', seps)
    out['wise_n_in_beam'] = pd.array(n_beam, dtype='Int64')

    # Vega colour and its propagated uncertainty.
    w1, w2 = out['wise_w1mpro'].astype(float), out['wise_w2mpro'].astype(float)
    e1, e2 = out['wise_w1sigmpro'].astype(float), out['wise_w2sigmpro'].astype(float)
    out['wise_w1w2'] = w1 - w2
    out['wise_w1w2_err'] = np.sqrt(e1 ** 2 + e2 ** 2)
    # ph_qual is one letter per band; W1 and W2 are the ones the colour needs.
    q = out['wise_ph_qual'].astype('string')
    out['wise_w1_ph_qual'] = q.str[0]
    out['wise_w2_ph_qual'] = q.str[1]
    return out
