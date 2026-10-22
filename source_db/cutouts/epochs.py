"""Single-epoch images: the exposure behind one light-curve point.

The stored light curves keep only a time and a band per epoch, not which image
the photometry came from, so the image is found by asking each survey what it
observed at that position and moment:

    ZTF   The epoch's field/ccd/quadrant (from its DR objectid, or the
          source's reference image) plus IRSA's directory listing for that
          night names the file directly. IBE's metadata search is exact
          without a pointing but takes 13-35 s, so it is only the fallback.
    PS1   `ps1filenames.py?type=warp`, which lists every warp covering the
          position with its MJD.

Both are public, so no credentials are needed to find or fetch an epoch. The
pixels are then put on the same grid as the reference stamp beside them, so
the two can be compared pixel for pixel:

    ZTF   science images are rotated ~12 deg against the sky and, for a
          secondary-grid field, against the reference too, so the epoch is
          resampled onto the reference cutout's own WCS.
    PS1   warps and stacks share the skycell projection, and `fitscut` crops
          both to the same grid, so the stack's flip is all it takes.
"""
from __future__ import annotations

import csv
import io
import re
import urllib.parse
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np

from source_db.cutouts import render

IBE_SCI_SEARCH = 'https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci'
IBE_SCI = 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci'
PS1_FILENAMES = 'https://ps1images.stsci.edu/cgi-bin/ps1filenames.py'

# ZTF exposures are 30 s; IRSA's `obsjd` is the start, our time the midpoint.
ZTF_SEARCH_WINDOW_DAYS = 0.01
ZTF_MATCH_TOL_S = 60.0

# PS1 Detection `obsTime` sits 56-60 s after the warp's MJD in every detection
# checked (12 of 12), and a neighbouring warp can be as close as ~58 s on the
# other side, so the lag is removed before taking the nearest rather than
# trusting the nearest raw difference.
PS1_OBSTIME_LAG_S = 58.0
PS1_MATCH_TOL_S = 20.0

EPOCH_SURVEYS = ('ztf', 'panstarrs')
EPOCH_BANDS = {'ztf': ('g', 'r', 'i'), 'panstarrs': ('g', 'r', 'i', 'z', 'y')}


@dataclass(frozen=True)
class Exposure:
    survey: str
    band: str
    image_id: str
    obs_mjd: float               # exposure start, UTC
    cutout_url: str              # FITS crop around the source
    full_url: str                # ZTF: the whole image; PS1: a JPEG view at STScI
    details: dict = field(default_factory=dict, compare=False, hash=False)


def find_exposure(survey: str, band: str, ra: float, dec: float, mjd: float,
                  ztf_pointing: Optional[tuple[int, int, int]] = None) -> Optional[Exposure]:
    """The exposure a light-curve point came from, or None if none matches.

    `ztf_pointing` is the (field, ccd, qid) the epoch was taken on, from its
    objectid or, failing that, the source's reference image. With it the
    exposure is named from IRSA's directory listing; without it, or if it is
    wrong, IRSA's much slower metadata search is the fallback.
    """
    # Rounded so float noise from the browser does not defeat the cache.
    return _find(survey, band, round(ra, 6), round(dec, 6), round(mjd, 6), ztf_pointing)


@lru_cache(maxsize=4096)
def _find(survey: str, band: str, ra: float, dec: float, mjd: float,
          ztf_pointing: Optional[tuple[int, int, int]]) -> Optional[Exposure]:
    if survey == 'ztf':
        return _find_ztf(band, ra, dec, mjd, ztf_pointing)
    if survey == 'panstarrs':
        return _find_ps1(band, ra, dec, mjd)
    raise ValueError(f'no single-epoch images for {survey}')


# ── ZTF ───────────────────────────────────────────────────────────────────

def ztf_mid_exposure_jd(hmjd: float, ra: float, dec: float) -> float:
    """Geocentric UTC JD of mid-exposure for a ZTF DR `hmjd`.

    `hmjd` is heliocentric on the UTC scale: treating it as TDB leaves every
    match off by exactly TT-UTC (69.2 s), which is how this was found.

    The light time is (r_earth - r_sun) . n / c from the built-in ephemeris,
    not `Time.light_travel_time`, which needs an observatory and so pulls IERS
    Earth-rotation tables off the network (13 s, and fails for recent dates).
    Palomar's offset from the geocentre is 21 ms, far inside the match window.
    """
    import astropy.constants as const
    import astropy.units as u
    from astropy.coordinates import SkyCoord, get_body_barycentric
    from astropy.time import Time
    t = Time(hmjd, format='mjd', scale='utc')
    n = SkyCoord(ra, dec, unit='deg').cartesian.xyz
    r = get_body_barycentric('earth', t).xyz - get_body_barycentric('sun', t).xyz
    ltt_days = float(((r * n).sum() / const.c).to(u.day).value)
    return float(t.jd) - ltt_days


def ztf_sci_path(row: dict) -> str:
    ffd = row['filefracday']
    return (f"{ffd[:4]}/{ffd[4:8]}/{ffd[8:]}/ztf_{ffd}_{int(row['field']):06d}_"
            f"{row['filtercode']}_c{int(row['ccdid']):02d}_{row['imgtypecode']}"
            f"_q{int(row['qid'])}_sciimg.fits")


def parse_objectid(objectid: str) -> Optional[tuple[int, int, int, int]]:
    """(field, fid, ccdid, qid) from a ZTF DR objectid.

    The layout is field | fid | ccd (2) | qid | 8-digit serial, e.g.
    876|3|14|2|00025364, checked against the exposure IRSA returned for it.
    """
    s = str(objectid).strip()
    if not s.isdigit() or len(s) < 13:
        return None
    return int(s[:-12]), int(s[-12]), int(s[-11:-9]), int(s[-9])


@lru_cache(maxsize=256)
def _night_exposures(day: str) -> tuple[int, ...]:
    """Exposure directories IRSA holds for one UTC night (`YYYYMMDD`).

    Each is the 6-digit day fraction the exposure started at. One listing
    (3-7 s) replaces IBE's metadata search (13-35 s), and a night's clicks
    then share it.
    """
    raw = _get(f'{IBE_SCI}/{day[:4]}/{day[4:]}/', _auth())
    if not raw:
        return ()
    return tuple(sorted(int(m) for m in re.findall(r'href="(\d{6})/"', raw.decode())))


def _find_ztf_by_listing(band: str, ra: float, dec: float, jd: float,
                         pointing: tuple[int, int, int]) -> Optional[Exposure]:
    """Name the exposure directly from the night's listing and a known pointing."""
    from astropy.time import Time
    t = Time(jd, format='jd', scale='utc')
    day = t.strftime('%Y%m%d')
    frac = int((jd - 0.5) % 1 * 1e6)      # JD days start at noon; IRSA's at midnight
    # The directory is stamped a few seconds before the exposure opens, so the
    # midpoint sits ~20 s after it; exposures are ~40 s apart.
    starts = [s for s in _night_exposures(day) if 0 <= frac - s <= ZTF_MATCH_TOL_S / 86400 * 1e6]
    if not starts:
        return None
    ffd = f'{day}{max(starts):06d}'
    field, ccd, qid = pointing
    row = {'filefracday': ffd, 'field': field, 'filtercode': render.ZTF_FILTERCODE[band],
           'ccdid': ccd, 'imgtypecode': 'o', 'qid': qid}
    path = ztf_sci_path(row)
    # A tiny crop proves the file exists and covers the position; a wrong
    # pointing (an epoch from an overlapping field) comes back 404.
    if not _get(f'{IBE_SCI}/{path}?center={ra},{dec}&size=3arcsec&gzip=false', _auth()):
        return None
    return Exposure(
        survey='ztf', band=band,
        image_id=path.rsplit('/', 1)[-1].removesuffix('_sciimg.fits'),
        obs_mjd=float(Time.strptime(day, '%Y%m%d').mjd) + max(starts) / 1e6,
        cutout_url=_ztf_cutout(path, ra, dec), full_url=f'{IBE_SCI}/{path}',
        details={'field': field, 'ccdid': ccd, 'qid': qid},
    )


def _ztf_cutout(path: str, ra: float, dec: float) -> str:
    # 1.5x the stamp: the science image is rotated against the reference, so
    # a same-size crop leaves the corners of the resampled grid empty.
    return (f'{IBE_SCI}/{path}?center={ra},{dec}'
            f'&size={render.CUTOUT_ARCSEC * 1.5}arcsec&gzip=false')


def _find_ztf(band: str, ra: float, dec: float, hmjd: float,
              pointing: Optional[tuple[int, int, int]] = None) -> Optional[Exposure]:
    jd = ztf_mid_exposure_jd(hmjd, ra, dec)
    if pointing is not None:
        exp = _find_ztf_by_listing(band, ra, dec, jd, pointing)
        if exp is not None:
            return exp
    # IRSA's own search: exact whatever the pointing, but 13-35 s.
    where = (f'obsjd BETWEEN {jd - ZTF_SEARCH_WINDOW_DAYS} AND {jd + ZTF_SEARCH_WINDOW_DAYS}'
             f" AND filtercode='{render.ZTF_FILTERCODE[band]}'")
    url = IBE_SCI_SEARCH + '?' + urllib.parse.urlencode(
        {'POS': f'{ra},{dec}', 'WHERE': where, 'ct': 'csv'})
    raw = _get(url, _auth())
    rows = list(csv.DictReader(io.StringIO(raw.decode()))) if raw else []

    def offset_s(r: dict) -> float:
        return abs(float(r['obsjd']) + float(r['exptime']) / 2 / 86400 - jd) * 86400

    rows = [r for r in rows if offset_s(r) <= ZTF_MATCH_TOL_S]
    if not rows:
        return None
    r = min(rows, key=offset_s)
    path = ztf_sci_path(r)
    return Exposure(
        survey='ztf', band=band,
        image_id=path.rsplit('/', 1)[-1].removesuffix('_sciimg.fits'),
        obs_mjd=float(r['obsjd']) - 2400000.5,
        cutout_url=_ztf_cutout(path, ra, dec),
        full_url=f'{IBE_SCI}/{path}',
        details={'field': int(r['field']), 'ccdid': int(r['ccdid']), 'qid': int(r['qid']),
                 'exptime': float(r['exptime']), 'seeing': _num(r.get('seeing')),
                 'maglimit': _num(r.get('maglimit')), 'airmass': _num(r.get('airmass'))},
    )


def _get(url: str, headers: Optional[dict] = None) -> Optional[bytes]:
    """`render._get`, retried once after a 503: IRSA sheds load intermittently."""
    import time
    import urllib.error
    try:
        return render._get(url, headers)
    except urllib.error.HTTPError as e:
        if e.code != 503:
            raise
        time.sleep(1.5)
        return render._get(url, headers)


def _auth() -> dict:
    a = render._irsa_auth()
    return {'Authorization': a} if a else {}


# ── PanSTARRS ─────────────────────────────────────────────────────────────

def _find_ps1(band: str, ra: float, dec: float, obs_time: float) -> Optional[Exposure]:
    url = PS1_FILENAMES + '?' + urllib.parse.urlencode(
        {'ra': ra, 'dec': dec, 'filters': band, 'type': 'warp'})
    raw = _get(url)
    if not raw:
        return None
    lines = raw.decode().split('\n')
    header = lines[0].split()
    warps = [dict(zip(header, ln.split())) for ln in lines[1:] if ln.strip()]
    target = obs_time - PS1_OBSTIME_LAG_S / 86400

    def offset_s(w: dict) -> float:
        return abs(float(w['mjd']) - target) * 86400

    warps = [w for w in warps if offset_s(w) <= PS1_MATCH_TOL_S]
    if not warps:
        return None
    w = min(warps, key=offset_s)
    base = (f"{render.PS1_FITSCUT}?ra={ra}&dec={dec}&size={render.PS1_SIZE_PX}"
            f"&red={w['filename']}")
    return Exposure(
        survey='panstarrs', band=band, image_id=w['shortname'],
        obs_mjd=float(w['mjd']),
        cutout_url=base + '&format=fits', full_url=base + '&format=jpg',
        details={'projcell': int(w['projcell']), 'skycell': int(w['subcell'])},
    )


# ── pixels ────────────────────────────────────────────────────────────────

def fetch_epoch(exp: Exposure, ra: float, dec: float,
                ref_origin: Optional[tuple] = None) -> Optional[np.ndarray]:
    """The epoch's pixels, oriented like the reference stamp of its band.

    `ref_origin` is the ZTF (field, ccd, qid) of the reference stamp; without
    it the epoch goes onto a north-up 1"/px grid oriented the same way.
    """
    raw = _get(exp.cutout_url, _auth() if exp.survey == 'ztf' else None)
    if not raw:
        return None
    from astropy.io import fits
    hdu = fits.open(io.BytesIO(raw))[0]
    if hdu.data is None:
        return None
    img = hdu.data.astype(np.float32)
    if not render._has_signal(img):
        return None
    if exp.survey == 'panstarrs':
        return np.flipud(render.ps1_linear_flux(img, hdu.header))

    from astropy.wcs import WCS
    grid = _ztf_reference_grid(exp.band, ra, dec, ref_origin)
    out = reproject(img, WCS(hdu.header), *grid)
    if not render._has_signal(out):
        return None
    # Same median background and flip as `render.fetch_ztf`.
    return np.fliplr(out - np.nanmedian(out))


def _ztf_reference_grid(band: str, ra: float, dec: float,
                        ref_origin: Optional[tuple]):
    """(WCS, shape) of the reference stamp, or a stand-in on the same axes."""
    from astropy.io import fits
    from astropy.wcs import WCS
    auth = render._irsa_auth()
    if ref_origin is not None and auth is not None:
        raw = _get(render.ztf_cutout_url(*ref_origin, band, ra, dec),
                   {'Authorization': auth})
        if raw:
            hdu = fits.open(io.BytesIO(raw))[0]
            if hdu.data is not None:
                return WCS(hdu.header), hdu.data.shape
    # RA increasing with x and Dec decreasing with y, as the references are,
    # so the same left-right flip turns it east-left, north-up.
    n = int(render.CUTOUT_ARCSEC) + 1
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    w.wcs.cd = [[1 / 3600, 0], [0, -1 / 3600]]
    return w, (n, n)


def reproject(img: np.ndarray, src_wcs, dst_wcs, shape: tuple) -> np.ndarray:
    """Bilinear resample of `img` onto `dst_wcs`; NaN where it has no data.

    Plain numpy rather than scipy, which the deployed host does not install.
    A NaN anywhere under the 2x2 footprint makes the output pixel NaN.
    """
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    sx, sy = src_wcs.world_to_pixel(dst_wcs.pixel_to_world(xx, yy))
    # Rounded so a WCS round trip landing at 4.9999999 or -1e-10 counts as
    # the pixel it is, rather than its neighbour or off the edge.
    sx, sy = np.round(sx, 6), np.round(sy, 6)
    h, w = img.shape
    inside = (sx >= 0) & (sy >= 0) & (sx <= w - 1) & (sy <= h - 1)
    xc = np.clip(np.floor(sx).astype(int), 0, w - 2)
    yc = np.clip(np.floor(sy).astype(int), 0, h - 2)
    fx, fy = sx - xc, sy - yc
    out = (img[yc, xc] * (1 - fx) * (1 - fy) + img[yc, xc + 1] * fx * (1 - fy)
           + img[yc + 1, xc] * (1 - fx) * fy + img[yc + 1, xc + 1] * fx * fy)
    out[~inside] = np.nan
    return out.astype(np.float32)


def _num(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None
