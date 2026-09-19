"""Query helpers for the local SDSS DR17 spectroscopic catalog.

`specObj-dr17.fits` (6.7 GB, 5.8M rows, 133 columns) is the full SDSS-I--IV
spectroscopic catalog. It holds classifications and redshifts, *not* spectra --
the spectra are separate per-object files keyed by (plate, mjd, fiberid), which
this catalog provides and `spectrum_url()` turns into a download URL.

`build_parquet()` slims the FITS to the crossmatching columns and writes
`Data/catalogs/sdss_specobj_dr17_slim.parquet` (~180 MB). Matching itself
lives in `_base.LocalCatalog`, shared with the DESI and TNS modules.

Download the source FITS with:

    curl -C - -O https://data.sdss.org/sas/dr17/sdss/spectro/redux/specObj-dr17.fits
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from Source_Analysis.catalogs._base import LocalCatalog

_DATA_DIR = Path(__file__).resolve().parents[2] / 'Data' / 'catalogs'
CATALOG_FITS = _DATA_DIR / 'raw' / 'specObj-dr17.fits'
CATALOG_PARQUET = _DATA_DIR / 'sdss_specobj_dr17_slim.parquet'

# (FITS column, output column). PLUG_RA/PLUG_DEC are the fiber plug positions --
# specObj has no plain RA/DEC. The *_NOQSO variants are the fits made without QSO
# templates, the recommended classification for BOSS/eBOSS galaxy targets.
_COLUMNS: list[tuple[str, str]] = [
    ('PLUG_RA', 'ra'),
    ('PLUG_DEC', 'dec'),
    ('CLASS', 'sdss_class'),
    ('SUBCLASS', 'sdss_subclass'),
    ('Z', 'sdss_z'),
    ('Z_ERR', 'sdss_z_err'),
    ('ZWARNING', 'sdss_zwarning'),
    ('CLASS_NOQSO', 'sdss_class_noqso'),
    ('Z_NOQSO', 'sdss_z_noqso'),
    ('ZWARNING_NOQSO', 'sdss_zwarning_noqso'),
    ('SPECPRIMARY', 'sdss_specprimary'),
    ('SN_MEDIAN_ALL', 'sdss_sn_median'),
    ('SURVEY', 'sdss_survey'),
    ('PLATE', 'sdss_plate'),
    ('MJD', 'sdss_mjd'),
    ('FIBERID', 'sdss_fiberid'),
    ('RUN2D', 'sdss_run2d'),
]

_CATEGORICAL = {'sdss_class', 'sdss_subclass', 'sdss_class_noqso', 'sdss_survey', 'sdss_run2d'}

_CATALOG = LocalCatalog(
    CATALOG_PARQUET,
    prefix='sdss',
    row_filter=lambda df: df['sdss_specprimary'].astype(bool),
)


def build_parquet(fits_path: Path = CATALOG_FITS, out_path: Path = CATALOG_PARQUET) -> Path:
    """Convert the full specObj FITS into the slim parquet used for crossmatching."""
    from astropy.io import fits

    with fits.open(fits_path, memmap=True) as hdul:
        data = hdul[1].data
        out = {}
        for src, dst in _COLUMNS:
            col = np.asarray(data[src])
            # FITS is big-endian; parquet/arrow only accepts native byte order.
            if col.dtype.byteorder not in ('=', '|'):
                col = col.astype(col.dtype.newbyteorder('='))
            if col.dtype.kind == 'S':
                col = np.char.decode(col, 'utf-8')
            if col.dtype.kind == 'U':
                col = np.char.strip(col)
            out[dst] = col

    df = pd.DataFrame(out)
    df['sdss_specprimary'] = df['sdss_specprimary'].astype(bool)

    n_before = len(df)
    df = df[np.isfinite(df['ra']) & np.isfinite(df['dec'])].reset_index(drop=True)
    if len(df) != n_before:
        print(f'  dropped {n_before - len(df)} rows with missing coordinates')

    for c in _CATEGORICAL:
        df[c] = df[c].astype('category')

    df.to_parquet(out_path, compression='zstd', index=False)
    return out_path


def load_catalog(primary_only: bool = True) -> pd.DataFrame:
    """The slim catalog, cached. Read-only -- see `_local_catalog`."""
    return _CATALOG.frame(primary_only)


def cone_search_many(
    ra: Sequence[float],
    dec: Sequence[float],
    radius_arcsec: float = 2.0,
    primary_only: bool = True,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Nearest SDSS spectrum within `radius_arcsec` for each input position."""
    return _CATALOG.match_nearest(ra, dec, radius_arcsec, filtered=primary_only, columns=columns)


def cone_search(
    ra: float,
    dec: float,
    radius_arcsec: float = 2.0,
    primary_only: bool = True,
) -> pd.DataFrame:
    """All SDSS spectra within `radius_arcsec` of one position, nearest first."""
    return _CATALOG.within(ra, dec, radius_arcsec, filtered=primary_only)


def spectrum_url(row: pd.Series) -> Optional[str]:
    """SAS URL of the spectrum file for a matched row, or None if unmatched."""
    plate, mjd, fiber = row['sdss_plate'], row['sdss_mjd'], row['sdss_fiberid']
    if pd.isna(plate) or pd.isna(mjd) or pd.isna(fiber):
        return None
    return (
        f"https://data.sdss.org/sas/dr17/sdss/spectro/redux/{row['sdss_run2d']}/spectra/lite/"
        f"{int(plate):04d}/spec-{int(plate):04d}-{int(mjd)}-{int(fiber):04d}.fits"
    )


if __name__ == '__main__':
    print(f'Building {CATALOG_PARQUET.name} from {CATALOG_FITS.name}...')
    path = build_parquet()
    print(f'Wrote {path} ({path.stat().st_size / 1e6:.0f} MB)')
