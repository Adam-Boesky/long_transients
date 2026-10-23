"""eROSITA-DE DR2 X-ray counterparts: is there an X-ray source at this position?

DR2 (Ramos-Ceja et al. 2026) is the eRASS:3 stack -- the first three all-sky
scans, 12 Dec 2019 to 16 Jun 2021, co-added -- detected in 0.2-2.3 keV. There
is no time information per source: no per-scan fluxes, no light curves, and no
MJD column (DR1, the eRASS1-only catalogue, is the one with `mjd`).

Coverage is only the German half of the sky, Galactic 179.944 < l < 359.944
deg. Outside it a missing match means "not observed", not "no X-ray source", so
`erosita_covered` is written beside `erosita_matched` to tell the two apart.

Matching is a plain 10" cone, a dirty first pass. eROSITA's PSF is ~30" HEW
and its positional errors are a few arcsec, so the 1.5" the optical catalogs
use would miss most real counterparts. The DR2 paper's own cross-matches use
4x the positional error, 5-16" (Ramos-Ceja et al. 2026, Sect. 5); a per-source
rule like that, or the DR2 NWAY counterpart catalogues, is the refinement.

`scripts/build_erosita_catalog.py` downloads the catalogue and calls
`build_parquet()`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from Source_Analysis.catalogs._base import LocalCatalog

_DATA_DIR = Path(__file__).resolve().parents[2] / 'Data' / 'catalogs'
CATALOG_RAW_DIR = _DATA_DIR / 'raw' / 'erosita'
CATALOG_FITS = CATALOG_RAW_DIR / 'eRASS3_Main_v1.3.fits.gz'
CATALOG_PARQUET = _DATA_DIR / 'erosita_dr2_main_slim.parquet'
CATALOG_URL = ('https://erosita.mpe.mpg.de/dr2/AllSkySurveyData_dr2/Catalogues_dr2/'
               'RamosM_DR2/eRASS3_Main_v1.3.fits.gz')

# The DR2 release footprint (eROSITA-DE), from the DR2 description page.
L_MIN_DEG, L_MAX_DEG = 179.94423568, 359.94423568

MATCH_RADIUS_ARCSEC = 10.0

# (catalogue column, output column).
_COLUMNS: list[tuple[str, str]] = [
    ('IAUNAME', 'erosita_name'),
    ('UID', 'erosita_uid'),
    ('RA', 'ra'),
    ('DEC', 'dec'),
    ('POS_ERR', 'erosita_pos_err'),
    ('DET_LIKE_0', 'erosita_det_like'),
    ('EXT', 'erosita_ext'),
    ('EXT_LIKE', 'erosita_ext_like'),
    ('ML_FLUX_1', 'erosita_flux'),            # 0.2-2.3 keV, erg/s/cm^2
    ('ML_FLUX_ERR_1', 'erosita_flux_err'),
    ('ML_EXP_1', 'erosita_exposure'),          # s, vignetted
    ('UID_DR1', 'erosita_uid_dr1'),            # eRASS1 match: >0 yes, <0 weak, 0 none
    ('FLAG_OPT', 'erosita_flag_opt'),          # possible optical loading by a bright star
]

# The main catalogue is already cut at DET_LIKE_0 >= 6 (Sect. 4.1), so no filter.
_CATALOG = LocalCatalog(CATALOG_PARQUET, prefix='erosita')


def build_parquet(fits_path: Path = CATALOG_FITS,
                  out_path: Path = CATALOG_PARQUET) -> Path:
    """The slim crossmatching parquet, from the downloaded DR2 main catalogue.

    Reads the whole gzipped FITS into memory (~2 GB) to pull 13 of its 250 columns.
    """
    from astropy.io import fits
    if not fits_path.exists():
        raise FileNotFoundError(f'{fits_path} missing -- run scripts/build_erosita_catalog.py')
    with fits.open(fits_path, memmap=False) as hdul:
        data = hdul[1].data
        names = {n.upper(): n for n in data.columns.names}
        missing = [src for src, _ in _COLUMNS if src not in names]
        if missing:
            raise KeyError(f'DR2 catalogue lacks {missing}; the column map needs updating')
        df = pd.DataFrame({dst: _native(data[names[src]]) for src, dst in _COLUMNS})
    df['erosita_name'] = df['erosita_name'].astype('string').str.strip()
    df['erosita_flag_opt'] = df['erosita_flag_opt'].astype(bool)
    df = df[np.isfinite(df['ra']) & np.isfinite(df['dec'])].reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression='zstd', index=False)
    return out_path


def _native(col) -> np.ndarray:
    """FITS columns are big-endian; pandas and pyarrow want native byte order."""
    a = np.asarray(col)
    return a.astype(a.dtype.newbyteorder('=')) if a.dtype.byteorder == '>' else a


def load_catalog() -> pd.DataFrame:
    """The DR2 main catalogue, cached. Read-only -- see `_base`."""
    return _CATALOG.frame()


def in_footprint(ra: Sequence[float], dec: Sequence[float]) -> np.ndarray:
    """True where a position lies in the eROSITA-DE half of the sky."""
    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    out = np.zeros(len(ra), dtype=bool)
    ok = np.isfinite(ra) & np.isfinite(dec)
    if ok.any():
        lon = SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg).galactic.l.deg
        out[ok] = (lon > L_MIN_DEG) & (lon < L_MAX_DEG)
    return out


def cone_search_many(
    ra: Sequence[float],
    dec: Sequence[float],
    radius_arcsec: float = MATCH_RADIUS_ARCSEC,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Nearest DR2 X-ray source within `radius_arcsec` of each input position."""
    out = _CATALOG.match_nearest(ra, dec, radius_arcsec, columns=columns)
    out.insert(1, 'erosita_covered', in_footprint(ra, dec))
    return out


def cone_search(ra: float, dec: float, radius_arcsec: float = MATCH_RADIUS_ARCSEC) -> pd.DataFrame:
    """All DR2 sources within `radius_arcsec` of one position, nearest first."""
    return _CATALOG.within(ra, dec, radius_arcsec)


if __name__ == '__main__':
    print(f'Building {CATALOG_PARQUET.name} from {CATALOG_FITS}...')
    path = build_parquet()
    print(f'Wrote {path} ({path.stat().st_size / 1e6:.0f} MB)')
    print(f'  {len(load_catalog()):,} sources')
