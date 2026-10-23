"""Turn a source's normalised rows into the blob the frontend charts read.

Shape: columnar arrays per (survey, band), not an array of row objects. For
~1,700 epochs the columnar form is roughly a third the JSON of the row form,
because the keys are written once instead of once per epoch.

    {"source_uuid": "...",
     "n_epochs": 1694,
     "surveys": {
       "ztf":  {"g": {"mjd": [...], "mag": [...], "magerr": [...],
                      "catflags": [...]},
                "r": {...}},
       "wise": {"w1": {"mjd": [...], "mag": [...], "magerr": [...],
                       "snr": [...]}}}}

Reductions are deliberately *not* applied. `_base.RENDER_TIME_REDUCTIONS`
lists what the pages must do at draw time -- the ZTF `catflags & 6141` cut and
the WISE/NEOWISE visit binning -- and the flags needed to do it travel in the
payload. Baking them in would freeze the choice and lose the measurements.
"""
from __future__ import annotations

from source_db.lightcurves._base import SURVEY_EXTRA_COLUMNS

import gzip
from typing import Optional

import numpy as np
import orjson
import pandas as pd

from source_db.lightcurves import _base

# Per-survey extras worth shipping. Everything else in the unified schema is
# null for that survey and would just be a column of nulls in the JSON.
# The same map `lightcurves._base` normalises against. Imported rather than
# restated: the two were byte-identical and kept in step by hand.
PAYLOAD_EXTRAS = SURVEY_EXTRA_COLUMNS

_CORE = ['mjd', 'mag', 'magerr', 'snr']

# Surveys whose per-epoch `catalog_id` ships with the payload.
ID_SURVEYS = {'ztf'}

# gzip rather than zstd: every browser and HTTP stack speaks it, so the API can
# return these bytes untouched with Content-Encoding: gzip and do no work.
GZIP_LEVEL = 6


def _clean(values: np.ndarray) -> list:
    """NaN/NaT -> None, numpy scalars -> python, so orjson can serialise it."""
    out = values.tolist()
    return [None if isinstance(v, float) and np.isnan(v) else v for v in out]


def build_payload(df: pd.DataFrame, source_uuid: str) -> tuple[bytes, int, list[str]]:
    """One source's rows -> (gzipped JSON, n_epochs, surveys present)."""
    surveys: dict[str, dict] = {}
    n = 0
    for survey, s_df in df.groupby('survey', sort=True):
        extras = [c for c in PAYLOAD_EXTRAS.get(str(survey), []) if c in s_df.columns]
        bands: dict[str, dict] = {}
        for band, b_df in s_df.groupby('band', sort=True):
            if band is None or (isinstance(band, float) and np.isnan(band)):
                continue
            b_df = b_df.sort_values('mjd', kind='stable')
            series = {}
            for col in _CORE:
                vals = b_df[col].to_numpy()
                # snr is null for several surveys; drop rather than ship nulls
                if col == 'snr' and pd.isna(b_df[col]).all():
                    continue
                series[col] = _clean(vals)
            for col in extras:
                if pd.isna(b_df[col]).all():
                    continue
                series[col] = _clean(b_df[col].to_numpy())
            # ZTF's objectid encodes the field/ccd/quadrant an epoch was taken
            # on, which is what finds its exposure when a point is clicked.
            if survey in ID_SURVEYS and 'catalog_id' in b_df.columns:
                series['catalog_id'] = [None if pd.isna(v) else str(v)
                                        for v in b_df['catalog_id']]
            bands[str(band)] = series
            n += len(b_df)
        if bands:
            surveys[str(survey)] = bands

    doc = {'source_uuid': source_uuid, 'n_epochs': n, 'surveys': surveys}
    raw = orjson.dumps(doc, option=orjson.OPT_SERIALIZE_NUMPY)
    return gzip.compress(raw, GZIP_LEVEL), n, sorted(surveys)


def decode_payload(blob: bytes) -> dict:
    """Inverse of `build_payload`, for tests and PDF rendering."""
    return orjson.loads(gzip.decompress(blob))


def to_long_frame(doc: dict) -> pd.DataFrame:
    """Payload -> the melted frame, i.e. back to `_base.LIGHTCURVE_COLUMNS`.

    Lets the on-demand PDF path reuse the serving blob instead of re-reading
    the parquet tier.
    """
    rows = []
    for survey, bands in doc.get('surveys', {}).items():
        for band, series in bands.items():
            n = len(series.get('mjd', []))
            base = {'source_uuid': doc['source_uuid'], 'survey': survey, 'band': band}
            for i in range(n):
                rows.append({**base, **{k: v[i] for k, v in series.items()}})
    if not rows:
        return _base.empty_lightcurves()
    return pd.DataFrame(rows)


# ── spectra ───────────────────────────────────────────────────────────────────

# Spectra are ~4,600 (SDSS) or ~7,781 (DESI) pixels, so unlike light curves the
# float repr dominates the payload: a full float32 round-trips to ~9 JSON
# characters, most of them noise. Fluxes carry ~3 significant figures and
# wavelengths are a known grid, so rounding here is lossless in any sense that
# matters for a plot and roughly halves the bytes on the wire.
SPECTRUM_FLUX_DP = 4
SPECTRUM_WAVE_DP = 3


def _round(values, dp: int) -> list:
    if values is None:
        return None
    arr = np.asarray(values, dtype='float64').round(dp)
    return [None if np.isnan(v) else v for v in arr.tolist()]


# Both surveys sample on an analytic grid -- DESI linearly in wavelength,
# SDSS linearly in log10(wavelength) with dloglam = 1e-4 -- so the array can be
# replaced by (start, step, n). Measured residual is 4e-4 A for DESI and 1e-2 A
# for SDSS, i.e. ~100x below one pixel, and it is storage noise rather than a
# real irregularity. Dropping the array takes a page from 40 to 22 KB (SDSS)
# and 50 to 34 KB (DESI).
#
# Only the *serving* payload does this. The analysis tier keeps explicit
# wavelengths so it stays self-describing.
GRID_TOLERANCE_FRAC = 0.05   # of one pixel


def fit_grid(wavelength) -> Optional[dict]:
    """(scale, start, step) if the grid is regular to within tolerance.

    Returns None when it is not, so the caller falls back to the explicit
    array rather than silently resampling someone's spectrum.
    """
    w = np.asarray(wavelength, dtype='float64')
    if w.size < 2:
        return None
    for scale, fwd, inv in (('linear', lambda x: x, lambda x: x),
                            ('log10', np.log10, lambda x: 10.0 ** x)):
        if scale == 'log10' and w.min() <= 0:
            continue
        g = fwd(w)
        step = (g[-1] - g[0]) / (g.size - 1)
        err = np.abs(inv(g[0] + np.arange(g.size) * step) - w).max()
        pixel = np.abs(np.diff(w)).min()
        if err <= GRID_TOLERANCE_FRAC * pixel:
            return {'wave_scale': scale, 'wave_start': float(g[0]),
                    'wave_step': float(step)}
    return None


def build_spectrum_payload(df: pd.DataFrame,
                           source_uuid: str) -> tuple[bytes, int, list[str]]:
    """One source's spectra -> (gzipped JSON, n_spectra, surveys present).

    Same contract as `build_payload`: the API returns these bytes untouched.
    The client rebuilds wavelength from `wave_start`/`wave_step`/`n_pix`, or
    reads `wavelength` directly when the grid was irregular.
    """
    out = []
    for _, r in df.iterrows():
        if r['wavelength'] is None or r['flux'] is None:
            continue
        rec = {
            'survey': str(r['survey']),
            'spec_id': str(r['spec_id']),
            'spec_class': None if pd.isna(r['spec_class']) else str(r['spec_class']),
            'redshift': None if pd.isna(r['redshift']) else float(r['redshift']),
            'n_pix': int(r['n_pix']),
            'flux': _round(r['flux'], SPECTRUM_FLUX_DP),
            'model': _round(r['model'], SPECTRUM_FLUX_DP),
        }
        grid = fit_grid(r['wavelength'])
        if grid is None:
            rec['wavelength'] = _round(r['wavelength'], SPECTRUM_WAVE_DP)
            rec['wave_scale'] = None
        else:
            rec.update(grid)
        out.append(rec)
    doc = {'source_uuid': source_uuid, 'n_spectra': len(out), 'spectra': out}
    raw = orjson.dumps(doc, option=orjson.OPT_SERIALIZE_NUMPY)
    return (gzip.compress(raw, GZIP_LEVEL), len(out),
            sorted({s['survey'] for s in out}))
