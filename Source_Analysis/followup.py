"""The follow-up campaign's data: one directory per target.

Layout (`FOLLOWUP_DIR`, normally `Data/followup/`)::

    <slug>/                     slug = source_db.identity.coord_slug(ra, dec)
      target.yaml               position, redshift, notes, and the spectra listed
      photometry.txt            our own photometry (Light_Curve's `custom` catalog)
      spectra/
        raw/<telescope file>    as delivered, original names
        <instrument>_<date>.txt reduced: wavelength [A], flux [erg/s/cm^2/A]
        <instrument>_<date>.pdf plot
      zfps/
        batchfp_req*_lc.txt     IPAC ZTF forced-photometry service results
        zfps.pdf
    ztforce_photometry/         ztforce's cache for any position; not a target

`target.yaml` is the only place a target's position lives. File names never
encode coordinates, so a typo in one cannot misplace a source.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from Extracting.utils import get_data_path
from source_db.identity import coord_slug

FOLLOWUP_DIR = Path(os.environ.get('LT_FOLLOWUP_DIR', Path(get_data_path()) / 'followup'))
ZTFORCE_CACHE_DIR = FOLLOWUP_DIR / 'ztforce_photometry'

TARGET_FILE = 'target.yaml'
PHOTOMETRY_FILE = 'photometry.txt'
SPECTRA_DIR = 'spectra'
ZFPS_DIR = 'zfps'
ZFPS_PLOT = 'zfps.pdf'

# 1.5" to match Light_Curve.query_rad_arcsec and every other catalog here.
MATCH_RADIUS_ARCSEC = 1.5

# Keys written first, in this order; anything else follows.
_KEY_ORDER = ('slug', 'ra', 'dec', 'source_uuid', 'list_id', 'redshift', 'notes', 'spectra')


@dataclass
class Target:
    slug: str
    ra: float
    dec: float
    path: Path
    meta: dict = field(default_factory=dict)

    @property
    def redshift(self) -> Optional[float]:
        return self.meta.get('redshift')

    @property
    def photometry_file(self) -> Optional[Path]:
        p = self.path / PHOTOMETRY_FILE
        return p if p.exists() else None

    @property
    def spectra(self) -> list[dict]:
        """`spectra` entries from target.yaml, with `file`/`plot`/`raw` as absolute paths."""
        out = []
        for s in self.meta.get('spectra') or []:
            s = dict(s)
            for k in ('file', 'plot', 'raw'):
                if s.get(k):
                    s[k] = self.path / s[k]
            out.append(s)
        return out

    @property
    def zfps_files(self) -> list[Path]:
        return sorted((self.path / ZFPS_DIR).glob('batchfp_req*_lc.txt'))


def _read(path: Path) -> Target:
    meta = yaml.safe_load(path.read_text()) or {}
    for k in ('slug', 'ra', 'dec'):
        if k not in meta:
            raise ValueError(f'{path}: missing `{k}`')
    slug = path.parent.name
    if meta['slug'] != slug:
        raise ValueError(f'{path}: slug {meta["slug"]!r} does not match its directory {slug!r}')
    if coord_slug(meta['ra'], meta['dec']) != slug:
        raise ValueError(f'{path}: ra/dec give slug {coord_slug(meta["ra"], meta["dec"])!r}, '
                         f'not the directory name {slug!r}')
    return Target(slug=slug, ra=float(meta['ra']), dec=float(meta['dec']), path=path.parent, meta=meta)


def iter_targets(root: Path = FOLLOWUP_DIR) -> list[Target]:
    """Every target directory under `root`, sorted by slug."""
    return [_read(p) for p in sorted(Path(root).glob(f'*/{TARGET_FILE}'))]


def get_target(slug: str, root: Path = FOLLOWUP_DIR) -> Target:
    return _read(Path(root) / slug / TARGET_FILE)


def load_targets(root: Path = FOLLOWUP_DIR) -> pd.DataFrame:
    """One row per target: slug, ra, dec, path, and every scalar in target.yaml."""
    rows = []
    for t in iter_targets(root):
        row = {k: v for k, v in t.meta.items() if not isinstance(v, (list, dict))}
        row.update(slug=t.slug, ra=t.ra, dec=t.dec, path=str(t.path),
                   n_spectra=len(t.meta.get('spectra') or []),
                   has_photometry=t.photometry_file is not None,
                   n_zfps=len(t.zfps_files))
        rows.append(row)
    return pd.DataFrame(rows)


def nearest_target(ra: float, dec: float, radius_arcsec: float = MATCH_RADIUS_ARCSEC,
                   root: Path = FOLLOWUP_DIR) -> Optional[Target]:
    """The closest target within `radius_arcsec`, or None."""
    targets = iter_targets(root)
    if not targets:
        return None
    t_ra = np.array([t.ra for t in targets])
    t_dec = np.array([t.dec for t in targets])
    dra = ((t_ra - ra + 180) % 360 - 180) * np.cos(np.radians(dec))
    sep = np.hypot(dra, t_dec - dec) * 3600
    i = int(np.argmin(sep))
    return targets[i] if sep[i] <= radius_arcsec else None


def write_target(target: Target) -> None:
    """Write `target.meta` back to its target.yaml."""
    meta = {k: target.meta[k] for k in _KEY_ORDER if k in target.meta}
    meta.update({k: v for k, v in target.meta.items() if k not in meta})
    (target.path / TARGET_FILE).write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))


def create_target(ra: float, dec: float, root: Path = FOLLOWUP_DIR, **meta) -> Target:
    """Make a new target directory and its target.yaml. Refuses to overwrite one.

    Pass the catalog position, so the directory matches the website's coord_slug.
    """
    slug = coord_slug(ra, dec)
    path = Path(root) / slug
    if (path / TARGET_FILE).exists():
        raise FileExistsError(f'{path / TARGET_FILE} already exists')
    path.mkdir(parents=True, exist_ok=True)
    t = Target(slug=slug, ra=float(ra), dec=float(dec), path=path,
               meta={'slug': slug, 'ra': float(ra), 'dec': float(dec), **meta})
    write_target(t)
    return t
