"""Query helpers for the local DESI DR1 (Iron) redshift catalog.

`zall-pix-iron.fits` (22.4 GB, 28.4M rows, 136 columns) is the HEALPix-based
redshift catalog combining every DESI DR1 survey and program. Like SDSS's
specObj it holds redshifts and classifications, *not* spectra; DR1 publishes no
per-object spectra, so `spectrum_url()` points at the ~500 MB HEALPix coadd
bundle that contains one (SPARCL is the better route for a single spectrum).

`build_parquet()` slims it to the crossmatching columns and writes
`Data/catalogs/desi_dr1_zpix_slim.parquet`. The FITS is too large to load at once, so the
conversion streams it in row chunks; low-cardinality string columns become
categoricals, which keeps 28M rows in the hundreds of MB rather than several GB.
Matching lives in `_base.LocalCatalog`, shared with the SDSS and TNS modules.

Download the source FITS with:

    curl -C - -O https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from Source_Analysis.catalogs._base import LocalCatalog

_DATA_DIR = Path(__file__).resolve().parents[2] / 'Data' / 'catalogs'
CATALOG_FITS = _DATA_DIR / 'raw' / 'zall-pix-iron.fits'
CATALOG_PARQUET = _DATA_DIR / 'desi_dr1_zpix_slim.parquet'

_COLUMNS: list[tuple[str, str]] = [
    ('TARGET_RA', 'ra'),
    ('TARGET_DEC', 'dec'),
    ('TARGETID', 'desi_targetid'),
    ('SPECTYPE', 'desi_spectype'),
    ('SUBTYPE', 'desi_subtype'),
    ('Z', 'desi_z'),
    ('ZERR', 'desi_zerr'),
    ('ZWARN', 'desi_zwarn'),
    ('DELTACHI2', 'desi_deltachi2'),
    ('COADD_FIBERSTATUS', 'desi_fiberstatus'),
    ('ZCAT_PRIMARY', 'desi_zcat_primary'),
    ('OBJTYPE', 'desi_objtype'),
    ('SURVEY', 'desi_survey'),
    ('PROGRAM', 'desi_program'),
    ('HEALPIX', 'desi_healpix'),
]

# Few unique values each, so categoricals collapse them from ~3 GB to ~150 MB.
_CATEGORICAL = {'desi_spectype', 'desi_subtype', 'desi_objtype', 'desi_survey', 'desi_program'}

# OBJTYPE 'TGT' drops sky and calibration fibres, which zall includes and which
# would otherwise turn up as matches.
_CATALOG = LocalCatalog(
    CATALOG_PARQUET,
    prefix='desi',
    row_filter=lambda df: df['desi_zcat_primary'].astype(bool) & (df['desi_objtype'] == 'TGT'),
)


def build_parquet(
    fits_path: Path = CATALOG_FITS,
    out_path: Path = CATALOG_PARQUET,
    chunk_rows: int = 2_000_000,
) -> Path:
    """Stream the zall FITS into the slim parquet used for crossmatching."""
    from astropy.io import fits

    chunks = []
    with fits.open(fits_path, memmap=True) as hdul:
        hdu = hdul[1]
        n_rows = hdu.header['NAXIS2']
        for start in range(0, n_rows, chunk_rows):
            block = hdu.data[start:start + chunk_rows]
            piece = {}
            for src, dst in _COLUMNS:
                col = np.asarray(block[src])
                # FITS is big-endian; parquet/arrow only accepts native byte order.
                if col.dtype.byteorder not in ('=', '|'):
                    col = col.astype(col.dtype.newbyteorder('='))
                if col.dtype.kind == 'S':
                    col = np.char.decode(col, 'utf-8')
                if col.dtype.kind == 'U':
                    col = np.char.strip(col)
                piece[dst] = col
            df = pd.DataFrame(piece)
            for c in _CATEGORICAL:
                df[c] = df[c].astype('category')
            chunks.append(df)
            print(f'  {min(start + chunk_rows, n_rows):,} / {n_rows:,} rows')

    out = pd.concat(chunks, ignore_index=True)

    n_before = len(out)
    out = out[np.isfinite(out['ra']) & np.isfinite(out['dec'])].reset_index(drop=True)
    if len(out) != n_before:
        print(f'  dropped {n_before - len(out)} rows with missing coordinates')

    for c in _CATEGORICAL:
        out[c] = out[c].astype('category')
    out.to_parquet(out_path, compression='zstd', index=False)
    return out_path


def load_catalog(primary_only: bool = True) -> pd.DataFrame:
    """The slim catalog, cached. Read-only -- see `_local_catalog`.

    `primary_only` keeps the best redshift per target (ZCAT_PRIMARY) and drops
    non-science fibres (OBJTYPE != 'TGT').
    """
    return _CATALOG.frame(primary_only)


def cone_search_many(
    ra: Sequence[float],
    dec: Sequence[float],
    radius_arcsec: float = 2.0,
    primary_only: bool = True,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Nearest DESI redshift within `radius_arcsec` for each input position."""
    return _CATALOG.match_nearest(ra, dec, radius_arcsec, filtered=primary_only, columns=columns)


def cone_search(
    ra: float,
    dec: float,
    radius_arcsec: float = 2.0,
    primary_only: bool = True,
) -> pd.DataFrame:
    """All DESI redshifts within `radius_arcsec` of one position, nearest first."""
    return _CATALOG.within(ra, dec, radius_arcsec, filtered=primary_only)


def spectrum_url(row: pd.Series) -> Optional[str]:
    """URL of the HEALPix coadd holding this row's spectrum, or None if unmatched.

    DR1 ships no per-object spectra, so this is a ~500 MB bundle of roughly 500
    targets; prefer SPARCL when you only want one spectrum.
    """
    survey, program, hpx = row['desi_survey'], row['desi_program'], row['desi_healpix']
    if pd.isna(survey) or pd.isna(program) or pd.isna(hpx):
        return None
    hpx = int(hpx)
    return (
        'https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix/'
        f'{survey}/{program}/{hpx // 100}/{hpx}/coadd-{survey}-{program}-{hpx}.fits'
    )


if __name__ == '__main__':
    print(f'Building {CATALOG_PARQUET.name} from {CATALOG_FITS.name}...')
    path = build_parquet()
    print(f'Wrote {path} ({path.stat().st_size / 1e6:.0f} MB)')
