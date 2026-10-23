"""Turning a ztforce run into what the candidate page draws, and what is stored.

The page gets the light curve already stacked by ztforce itself, for every
window it offers, so it draws the same points as anything else built on ztforce
and changing the window in the browser never re-runs anything. Stacking follows
the ZFPS user guide (Masci et al. 2023, section 6.6): every good epoch, detected
or not, rescaled to a common zero point and inverse-variance averaged; a window
below S/N 3 comes back as a 5-sigma upper limit. "Unbinned" is per epoch:
detections, and 5-sigma limits for the rest. Error bars are statistical only,
so underestimated for bright sources (see ztforce's README).
"""
from __future__ import annotations

import gzip
import math
from pathlib import Path
from typing import Any

import os

import orjson

# Where ztforce keeps its image and photometry cache. Same directory
# `Light_Curve.get_catalog_lc` and `store_src_plots.prefetch_ztforce` use, so a
# source fetched by either path is already warm for the others. The queue's
# results live in ztforce.sqlite; this is only ztforce's own scratch cache.
#
# Overridable because it is written to and can be large: it was the one data
# path that stayed repo-relative when everything else moved to `LT_DATA_DIR`.
ZTFORCE_DIR = Path(os.environ.get(
    'LT_ZTFORCE_CACHE',
    Path(__file__).resolve().parents[1] / 'Data' / 'followup' / 'ztforce_photometry'))

BANDS = ('g', 'r', 'i')

# Stacking windows offered on the page, in days; 0 is unbinned. The frontend
# builds its selector from the payload, so this is the only list.
WINDOWS_DAYS = (0, 7, 30, 90, 182, 365, 730)

JD_TO_MJD = 2_400_000.5


def result_version() -> str:
    """Identifies how a stored result was made: ztforce's photometry version and
    the windows stacked. A stored result from another version is stale."""
    from ztforce._constants import _PHOTOMETRY_VERSION

    return f'p{_PHOTOMETRY_VERSION}-w{"-".join(map(str, WINDOWS_DAYS))}'


def _finite_or_none(values) -> list[float | None]:
    """JSON has no NaN; send missing values as null."""
    return [float(v) if math.isfinite(v) else None for v in values]


def stack_bands(lcs: dict[str, Any], window_days: int) -> dict[str, dict[str, list]]:
    """One window's light curve per band, in the chart's BandSeries shape.

    Window 0 is per epoch: good detections, and good non-detections as upper
    limits. Otherwise it is ztforce's non-overlapping stack (step = window);
    stacks below S/N 3 are upper limits.
    """
    out: dict[str, dict[str, list]] = {}
    for band, lc in lcs.items():
        if window_days == 0:
            df = lc.df
            # Detections, plus each good non-detection as its 5-sigma limit at the
            # target (ZFPS guide 6.4). Without the limits the unbinned curve is
            # only the bright tail of the noise and reads too bright.
            ep = df[df['detection'] | df['upper_limit'].notna()]
            if ep.empty:
                continue
            upper = (~ep['detection'].astype(bool)).tolist()
            mjd = ep['obsjd'] - JD_TO_MJD
            mag = ep['mag'].where(ep['detection'], ep['upper_limit'])
            magerr = ep['mag_err'].where(ep['detection'])
        else:
            st = lc.rolling_stack(window=window_days, window_unit='days', step=window_days)
            if st.empty:
                continue
            upper = (~st['detection'].astype(bool)).tolist()
            mjd = st['obsjd_center'] - JD_TO_MJD
            mag = st['mag_stack'].where(st['detection'], st['upper_limit_stack'])
            magerr = st['mag_err_stack']
        out[band] = {
            'mjd': _finite_or_none(mjd),
            'mag': _finite_or_none(mag),
            'magerr': _finite_or_none(magerr),
            'is_upper_limit': upper,
        }
    return out


def summarise(lcs: dict[str, Any]) -> dict[str, Any]:
    """Everything stored for one finished run: the page's stacks for every window,
    the number of good epochs, and a note on epochs that could not be measured.
    (The epochs themselves stay in ztforce's own cache, under ZTFORCE_DIR.)
    """
    return {
        'stacks': {str(w): stack_bands(lcs, w) for w in WINDOWS_DAYS},
        'n_epochs': sum(int((lc.df['flags'] == 0).sum()) for lc in lcs.values()),
        'note': _unmeasured_note(lcs),
    }


def _unmeasured_note(lcs: dict[str, Any]) -> str | None:
    """Which epochs could not be measured, counted from the flags (so it is right
    whether the run computed the photometry or read ztforce's cache)."""
    from ztforce._constants import FLAG_DOWNLOAD_FAILED, FLAG_PROCESSING_ERROR, FLAG_UNAVAILABLE

    counts = {'not served by IRSA': 0, 'download failed': 0, 'unreadable': 0}
    for lc in lcs.values():
        flags = lc.df['flags'].astype(int)
        counts['not served by IRSA'] += int((flags & FLAG_UNAVAILABLE > 0).sum())
        counts['download failed'] += int((flags & FLAG_DOWNLOAD_FAILED > 0).sum())
        counts['unreadable'] += int((flags & FLAG_PROCESSING_ERROR > 0).sum())
    parts = [f'{n} epochs {what}' for what, n in counts.items() if n]
    return ', '.join(parts) or None


def pack(obj: Any) -> bytes:
    return gzip.compress(orjson.dumps(obj))


def unpack(blob: bytes | None) -> Any:
    return None if blob is None else orjson.loads(gzip.decompress(blob))
