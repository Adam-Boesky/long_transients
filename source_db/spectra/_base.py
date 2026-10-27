"""Shared machinery for the spectra backfill.

Same two-stage split as the light curves -- fetch once into `raw/`, normalise
as often as you like -- and it reuses their chunk storage (`write_raw`,
coverage sidecars, the manifest) because that machinery is about resumable
chunked fetching, not about light curves specifically.

Why arrays rather than one row per pixel
----------------------------------------
A spectrum is ~4,600 (SDSS) or ~7,800 (DESI) pixels. Melting 71,000 spectra to
one row per pixel would be ~430M rows to express 71,000 objects, and every
read would re-group them. Parquet list columns keep a spectrum as a spectrum.

Everything is float32: these are fluxes with ~3 significant figures, so float64
would double the size to store noise.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# The chunk-storage helpers are survey-agnostic; reused rather than duplicated.
from source_db.lightcurves._base import (  # noqa: F401
    completed_chunks, coverage_path, covered_sources, iter_raw, load_manifest,
    raw_dir_for, raw_path, read_coverage, read_raw, record_chunk, write_coverage,
    write_raw,
)
from source_db.paths import DEFAULT_OUT_DIR

SURVEYS = ['sdss', 'desi']

# One row per spectrum.
SPECTRUM_COLUMNS = [
    'source_uuid',    # the key
    'survey',         # sdss | desi
    'spec_id',        # SDSS specobjid, DESI specid
    'spec_class',     # SDSS CLASS / DESI SPECTYPE
    'redshift',
    'mjd',            # observation date; SDSS only (a source can have many)
    'n_pix',
    'wavelength',     # list<float32>, Angstroms (linear, not log)
    'flux',           # list<float32>
    'model',          # list<float32>, the pipeline fit; may be null
]

# SDSS stores wavelength as log10(Angstroms); DESI stores linear Angstroms.
# Normalising to linear here means the frontend never has to know which is
# which -- the same choice `Sources.plot_spectrum` makes before overlaying them.


def spectra_dir(out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    return Path(out_dir) / 'spectra'


def load_targets(survey: str, out_dir: Path = DEFAULT_OUT_DIR,
                 limit: Optional[int] = None) -> pd.DataFrame:
    """Sources with a spectrum in `survey`, with the identifiers to fetch it.

    The crossmatch already happened in the catalog tier, so this is a filter,
    not a query.
    """
    path = Path(out_dir) / 'sources.parquet'
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found -- run `python -m source_db.build_catalog_tier` first')

    if survey == 'sdss':
        # A source can have many SDSS spectra; `spectra.sdss.targets` lists them.
        from source_db.spectra import sdss
        return sdss.targets(out_dir, limit)
    if survey == 'desi':
        cols = ['source_uuid', 'ra', 'dec', 'desi_matched', 'desi_targetid',
                'desi_spectype', 'desi_z']
    else:
        raise ValueError(f'unknown survey {survey!r}')

    df = pd.read_parquet(path, columns=cols)
    df = df[df[f'{survey}_matched'].fillna(False).astype(bool)].reset_index(drop=True)
    return df.head(limit) if limit else df


def empty_spectra() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(
        SPECTRUM_COLUMNS,
        ['string', 'string', 'string', 'string', 'float64', 'float64', 'Int32',
         'object', 'object', 'object'])})


def as_float32(a) -> Optional[np.ndarray]:
    if a is None:
        return None
    arr = np.asarray(a, dtype='float32')
    return arr if arr.size else None


def spectrum_row(source_uuid: str, survey: str, spec_id: str,
                 wavelength, flux, model=None, spec_class=None,
                 redshift=None, mjd=None) -> dict:
    """One normalised spectrum, with the arrays already float32."""
    w = as_float32(wavelength)
    f = as_float32(flux)
    m = as_float32(model)
    return {
        'source_uuid': source_uuid, 'survey': survey, 'spec_id': str(spec_id),
        'spec_class': None if spec_class is None else str(spec_class),
        'redshift': None if redshift is None else float(redshift),
        'mjd': None if mjd is None or pd.isna(mjd) else float(mjd),
        'n_pix': 0 if w is None else int(w.size),
        'wavelength': None if w is None else w,
        'flux': None if f is None else f,
        'model': None if m is None else m,
    }
