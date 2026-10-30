"""Custom follow-up photometry -> `Data/db/raw/custom/`.

The only catalog here with no network call: the epochs are each follow-up
target's `photometry.txt` (`Data/followup/<slug>/`, see
`Source_Analysis.followup`), positioned by the target's `target.yaml`.

`Light_Curve` matches a source to the *nearest* such target within
`query_rad_arcsec`; this module does the same match for every source at once.
Six files and 60 epochs today, so it runs in well under a second -- it exists
as a module rather than a one-off so that newly added photometry is picked up
by the same `python -m source_db.lightcurves.custom` as everything else.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from Source_Analysis import followup
from source_db.lightcurves import _base
from source_db.paths import RAW_DIR

SURVEY = 'custom'

MATCH_RADIUS_ARCSEC = followup.MATCH_RADIUS_ARCSEC

# The header every file carries. `UL` marks a non-detection upper limit.
FIELDS = ['mjd', 'mag', 'magerr', 'band', 'ul', 'telescope', 'instrument']

# Only these reach the plots: Light_Curve.ALL_BAND_DF maps custom_{g,r,i,z}_mag
# and nothing else, so an epoch in another filter is stored but never drawn.

_CHUNK_ID = '0000'


# ── the files ─────────────────────────────────────────────────────────────────

def file_positions(root: Path = followup.FOLLOWUP_DIR) -> pd.DataFrame:
    """`custom_id` (the target slug), `path`, `ra`, `dec` for every target with photometry."""
    rows = [{'custom_id': t.slug, 'path': str(t.photometry_file), 'ra': t.ra, 'dec': t.dec}
            for t in followup.iter_targets(root) if t.photometry_file is not None]
    return pd.DataFrame(rows, columns=['custom_id', 'path', 'ra', 'dec'])


def read_photometry(path: Path, custom_id: str) -> pd.DataFrame:
    """One file -> one frame, tolerating a row that omits the `UL` field.

    #14's file has a row with six fields rather than seven. Read
    positionally (as `Light_Curve` does) that row shifts, landing the telescope
    in `ul` and dropping the instrument. The measurement columns are unaffected
    either way, so this only rescues the metadata -- but storing `ul` as
    "Magellan" would be worse than storing nothing.
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        f = line.split()
        if len(f) < 4:
            print(f'  {custom_id}: skipping short row {line!r}')
            continue
        ul_present = f[4].lower() in ('true', 'false') if len(f) >= 5 else False
        ul = f[4].lower() == 'true' if ul_present else False
        rest = f[5:] if ul_present else f[4:]
        rows.append({
            'custom_id': custom_id,
            'mjd': float(f[0]),
            'mag': float(f[1]),
            'magerr': float(f[2]),
            'band': f[3],
            'is_upper_limit': ul,
            'telescope': rest[0] if len(rest) > 0 else None,
            'instrument': rest[1] if len(rest) > 1 else None,
        })
    return pd.DataFrame(rows)


# ── fetch ─────────────────────────────────────────────────────────────────────

def fetch(limit: Optional[int] = None, raw_dir: Path = RAW_DIR,
          root: Path = followup.FOLLOWUP_DIR,
          radius_arcsec: float = MATCH_RADIUS_ARCSEC) -> pd.DataFrame:
    """Attach every source to its nearest photometry file within `radius`."""
    positions = _base.load_positions(limit=limit)
    files = file_positions(root)
    if files.empty:
        print(f'{SURVEY}: no photometry in {root}')
        return pd.DataFrame()

    src = SkyCoord(positions['ra'].to_numpy(), positions['dec'].to_numpy(), unit='deg')
    fil = SkyCoord(files['ra'].to_numpy(), files['dec'].to_numpy(), unit='deg')

    # Nearest file per source, mirroring Light_Curve's argmin-over-separations.
    idx, sep, _ = src.match_to_catalog_sky(fil)
    hit = sep.arcsec <= radius_arcsec

    print(f'{SURVEY}: {len(files)} files, {len(positions):,} sources, '
          f'{int(hit.sum())} matched within {radius_arcsec}"')

    frames = []
    for fi in np.unique(idx[hit]):
        row = files.iloc[int(fi)]
        phot = read_photometry(Path(row['path']), row['custom_id'])
        if phot.empty:
            continue
        uuids = positions.loc[hit & (idx == fi), 'source_uuid'].to_numpy()
        seps = sep.arcsec[hit & (idx == fi)]
        for uuid, s in zip(uuids, seps):
            f = phot.copy()
            f.insert(0, 'source_uuid', uuid)
            f['sep_arcsec'] = s
            frames.append(f)
            print(f'  {row["custom_id"]:<26} -> {uuid[:8]}  '
                  f'{len(f):>3} epochs  ({s:.2f}")')

    if not frames:
        print(f'{SURVEY}: no sources matched a photometry file')
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    _base.write_raw(SURVEY, _CHUNK_ID, df, raw_dir=raw_dir, n_files=len(files))
    _base.write_coverage(SURVEY, _CHUNK_ID, positions['source_uuid'], raw_dir=raw_dir)
    print(f'{SURVEY}: {len(df):,} epochs for {df["source_uuid"].nunique()} sources')
    return df


def read_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return _base.read_raw(SURVEY, raw_dir)


# ── normalise ─────────────────────────────────────────────────────────────────

def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw follow-up epochs -> the shared light-curve schema.

    `is_upper_limit` is carried, not applied: a limit is a real constraint the
    pages may want to draw as an arrow rather than a point.
    """
    if raw.empty:
        return _base.empty_lightcurves()
    out = pd.DataFrame({
        'source_uuid': raw['source_uuid'].astype('string'),
        'survey': pd.array([SURVEY] * len(raw), dtype='string'),
        'band': raw['band'].astype('string'),
        'mjd': raw['mjd'].astype('float64'),
        'mag': raw['mag'].astype('float64'),
        'magerr': raw['magerr'].astype('float64'),
        'snr': pd.Series([float('nan')] * len(raw), dtype='float64'),
        'catalog_id': raw['custom_id'].astype('string'),
    })
    out['is_upper_limit'] = raw['is_upper_limit'].astype('boolean')
    out['telescope'] = raw['telescope'].astype('string')
    out['instrument'] = raw['instrument'].astype('string')
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--raw-dir', type=Path, default=RAW_DIR)
    a = ap.parse_args()
    fetch(limit=a.limit, raw_dir=a.raw_dir)
