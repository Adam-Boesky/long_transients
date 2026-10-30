"""Query helpers for the local SDSS DR20 spectroscopic catalog.

DR20 is cumulative, but no single DR20 file carries every spectrum *with* its
classification, so `build_parquet()` stacks the two summary catalogs that do:

    SDSS-I--IV   specObj-dr17.fits          5.8M spectra (Legacy, SEGUE, BOSS,
                                            eBOSS; run2d 26/103/104/v5_13_2). Not
                                            re-reduced since DR17: this is the
                                            file DR20 itself ships, byte-identical
    SDSS-V       spAll-lite-v6_2_1.fits     8.7M daily coadds, MJD 59146-60708
                 boss_vi_results-1.0.3.fits visual inspection of 45.8k SDSS-V
                                            objects (VAC #37), joined on sdss_id

and takes each spectrum's file URL from `allspec-dr20-1.0.2.parquet`, SDSS's
index of every released spectrum, joined on specobjid. The build checks that
every kept spectrum is in allspec, so the result is "every optical SDSS
spectrum in DR20".

Rows whose position is not where the light came from are dropped: sky fibres
(ZWARNING bit 0, OBJTYPE SKY) and unplugged fibres (ZWARNING bit 7). In SDSS-V
the latter are the fibres no target was assigned to (sdss_id -999).

`sdss_class` / `sdss_z` are the secure (z_conf 3) visual-inspection values
where they exist, as SDSS recommends, and the pipeline's otherwise; the
pipeline values are kept in `sdss_class_pipeline` / `sdss_z_pipeline`.

Matching does not prefer one spectrum over another: a source gets *every*
kept spectrum within the radius (repeat SDSS-V visits included; up to ~250 for
reverberation-mapped quasars). `cone_search_many` reports them as
`sdss_n_spectra`, `sdss_classes` (every class they give) and `sdss_class_z`
(median redshift per class); its other `sdss_*` columns describe the nearest
one, for the catalog panel and numeric filters. `spectra_within_many` lists
them all, which is what the spectra tier fetches. SPECPRIMARY is kept as a
column but not used for matching.

Built from the October 2026 inputs: 9.65M spectra kept (5.30M SDSS-I--IV,
4.35M SDSS-V), 7.21M of them primary.

Not included: APOGEE (infrared stellar spectra) and MaNGA (galaxy IFU cubes),
which have no CLASS, and the DR20 value-added catalogs other than VAC #37
(quasar properties, eROSITA counterparts, blazars, CVs, ...) and the older
quasar catalogs (DR16Q, DR7Q).

Writes `Data/catalogs/sdss_dr20_slim.parquet`. Matching itself lives in
`_base.LocalCatalog`, shared with the DESI and TNS modules. Inputs, into
`Data/catalogs/raw/` (the spAll is gunzipped by the build):

    curl -C - -O https://dr20.sdss.org/sas/dr20/prior-surveys/sdss2-dr8-sdss/spectro/redux/specObj-dr17.fits
    cd sdss_dr20
    curl -C - -O https://dr20.sdss.org/sas/dr20/spectro/boss/redux/v6_2_1/summary/daily/spAll-lite-v6_2_1.fits.gz
    curl -C - -O https://data.sdss.org/sas/dr20/vac/bhm/visual-inspect/1.0.3/boss_vi_results-1.0.3.fits
    curl -C - -O https://dr20.sdss.org/sas/dr20/spectro/allspec/1.0.2/allspec-dr20-1.0.2.parquet
"""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from Source_Analysis.catalogs._base import LocalCatalog

_DATA_DIR = Path(__file__).resolve().parents[2] / 'Data' / 'catalogs'
_RAW = _DATA_DIR / 'raw'
SPECOBJ_FITS = _RAW / 'specObj-dr17.fits'
SPALL_FITS = _RAW / 'sdss_dr20' / 'spAll-lite-v6_2_1.fits'
VI_FITS = _RAW / 'sdss_dr20' / 'boss_vi_results-1.0.3.fits'
ALLSPEC_PARQUET = _RAW / 'sdss_dr20' / 'allspec-dr20-1.0.2.parquet'
CATALOG_PARQUET = _DATA_DIR / 'sdss_dr20_slim.parquet'

# ZWARNING bits meaning the spectrum's position is not the target's.
SKY, UNPLUGGED = 1 << 0, 1 << 7
BAD_POSITION_MASK = SKY | UNPLUGGED

# Secure redshift in the visual-inspection VAC.
VI_SECURE = 3

# (FITS column, output column), common to specObj and spAll. The *_NOQSO
# variants are the fits made without QSO templates, which SDSS recommends for
# galaxy targets.
_SHARED: list[tuple[str, str]] = [
    ('SPECOBJID', 'sdss_specobjid'),
    ('CLASS', 'sdss_class_pipeline'),
    ('SUBCLASS', 'sdss_subclass'),
    ('Z', 'sdss_z_pipeline'),
    ('Z_ERR', 'sdss_z_err'),
    ('ZWARNING', 'sdss_zwarning'),
    ('CLASS_NOQSO', 'sdss_class_noqso'),
    ('Z_NOQSO', 'sdss_z_noqso'),
    ('ZWARNING_NOQSO', 'sdss_zwarning_noqso'),
    ('SPECPRIMARY', 'sdss_specprimary'),
    ('SN_MEDIAN_ALL', 'sdss_sn_median'),
    ('SURVEY', 'sdss_survey'),
    ('MJD', 'sdss_mjd'),
    ('RUN2D', 'sdss_run2d'),
]
# specObj has no plain RA/DEC: PLUG_RA/DEC are the drilled fibre positions.
_SPECOBJ = _SHARED + [('PLUG_RA', 'ra'), ('PLUG_DEC', 'dec'),
                      ('PLATE', 'sdss_field'), ('FIBERID', 'sdss_fiberid')]
# RACAT/DECCAT, the catalog position: FIBER_RA is NaN for some DR19-era rows.
_SPALL = _SHARED + [('RACAT', 'ra'), ('DECCAT', 'dec'), ('FIELD', 'sdss_field'),
                    ('SDSS_ID', 'sdss_id'), ('OBJTYPE', 'objtype')]

_CATEGORICAL = {'sdss_class', 'sdss_class_pipeline', 'sdss_subclass', 'sdss_class_noqso',
                'sdss_survey', 'sdss_run2d', 'sdss_class_source'}

_CATALOG = LocalCatalog(
    CATALOG_PARQUET,
    prefix='sdss',
    row_filter=lambda df: df['sdss_specprimary'].astype(bool),
)


def _read_fits(path: Path, columns: list[tuple[str, str]]) -> pd.DataFrame:
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        out = {}
        for src, dst in columns:
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
    # specObj stores SPECOBJID as an integer, spAll as a 28-digit string.
    df['sdss_specobjid'] = df['sdss_specobjid'].astype(str).str.strip()
    df['sdss_specprimary'] = df['sdss_specprimary'].astype(bool)
    return df


def _drop(df: pd.DataFrame, bad: np.ndarray, why: str) -> pd.DataFrame:
    print(f'  dropping {int(bad.sum()):,} {why}')
    return df[~bad].reset_index(drop=True)


def _gunzipped(path: Path) -> Path:
    """`path`, gunzipping `path.gz` beside it first if needed (memmap needs it plain)."""
    gz = path.with_name(path.name + '.gz')
    if not path.exists():
        print(f'  gunzipping {gz.name}')
        tmp = path.with_name(path.name + '.part')
        with gzip.open(gz, 'rb') as src, open(tmp, 'wb') as dst:
            shutil.copyfileobj(src, dst, length=16 << 20)
        tmp.replace(path)
    return path


def _spec_urls(allspec: Path, phase5: bool) -> pd.DataFrame:
    """specobjid -> SAS URL from allspec: the daily coadds for SDSS-V, all of SDSS-I--IV."""
    import duckdb

    # Unassigned SDSS-V fibres (sdss_id -999) share specobjids; they are
    # dropped from spAll too, so they are dropped here.
    where = ("sdss_phase = 5 AND coadd = 'daily' AND sdss_id > 0" if phase5
             else 'sdss_phase < 5')
    return duckdb.sql(
        f"SELECT CAST(specobjid AS VARCHAR) AS sdss_specobjid, sas_url AS sdss_spec_url "
        f"FROM '{allspec}' WHERE instrument IN ('sdss', 'boss') AND {where}").df()


def _with_urls(df: pd.DataFrame, allspec: Path, phase5: bool) -> pd.DataFrame:
    urls = _spec_urls(allspec, phase5)
    out = df.merge(urls, on='sdss_specobjid', how='left', validate='one_to_one')
    missing = out['sdss_spec_url'].isna()
    if missing.any():
        raise ValueError(f'{int(missing.sum()):,} spectra are not in allspec, e.g. '
                         f'{out.loc[missing, "sdss_specobjid"].head(3).tolist()}')
    return out


def _sdss_v(spall: Path, vi_path: Path, allspec: Path) -> pd.DataFrame:
    print(f'{spall.name}:')
    df = _read_fits(_gunzipped(spall), _SPALL)
    print(f'  {len(df):,} spectra')
    df = _drop(df, (df['sdss_id'] <= 0).to_numpy(), 'unassigned fibres (sdss_id -999)')
    df = _drop(df, ((df['sdss_zwarning'] & BAD_POSITION_MASK) != 0).to_numpy()
               | (df['objtype'] == 'SKY').to_numpy(), 'sky or unplugged fibres')
    df = _with_urls(df.drop(columns='objtype'), allspec, phase5=True)

    from astropy.table import Table
    vi = Table.read(vi_path).to_pandas()
    vi = vi[vi['has_vi'].astype(bool)]
    vi = pd.DataFrame({
        'sdss_id': vi['sdss_id'].astype('int64'),
        'sdss_class_vi': vi['class_best'].astype(str).str.strip(),
        'sdss_z_vi': vi['z_vi'].astype('float64'),
        'sdss_z_conf_vi': vi['z_conf_vi'].astype('int64'),
    })
    df = df.merge(vi, on='sdss_id', how='left', validate='many_to_one')
    print(f'  {df["sdss_z_conf_vi"].notna().sum():,} spectra of '
          f'{vi["sdss_id"].nunique():,} visually inspected objects')
    return df


def _sdss_iv(specobj: Path, allspec: Path) -> pd.DataFrame:
    print(f'{specobj.name}:')
    df = _read_fits(specobj, _SPECOBJ)
    print(f'  {len(df):,} spectra')
    df = _drop(df, ((df['sdss_zwarning'] & BAD_POSITION_MASK) != 0).to_numpy(),
               'sky or unplugged fibres')
    return _with_urls(df, allspec, phase5=False)


def build_parquet(out_path: Path = CATALOG_PARQUET, specobj: Path = SPECOBJ_FITS,
                  spall: Path = SPALL_FITS, vi: Path = VI_FITS,
                  allspec: Path = ALLSPEC_PARQUET) -> Path:
    """Stack SDSS-I--IV and SDSS-V into the slim parquet used for crossmatching."""
    df = pd.concat([_sdss_iv(specobj, allspec), _sdss_v(spall, vi, allspec)],
                   ignore_index=True)
    df['sdss_id'] = df['sdss_id'].astype('Int64')
    df['sdss_fiberid'] = df['sdss_fiberid'].astype('Int64')
    df['sdss_z_conf_vi'] = df['sdss_z_conf_vi'].astype('Int64')

    vi = (df['sdss_z_conf_vi'] == VI_SECURE).fillna(False).to_numpy()
    df['sdss_class'] = np.where(vi, df['sdss_class_vi'], df['sdss_class_pipeline'])
    df['sdss_z'] = np.where(vi, df['sdss_z_vi'], df['sdss_z_pipeline'])
    df['sdss_class_source'] = np.where(vi, 'visual', 'pipeline')

    n_before = len(df)
    df = df[np.isfinite(df['ra']) & np.isfinite(df['dec'])].reset_index(drop=True)
    if len(df) != n_before:
        print(f'  dropped {n_before - len(df)} rows with missing coordinates')

    for c in _CATEGORICAL:
        df[c] = df[c].astype('category')

    df.to_parquet(out_path, compression='zstd', index=False)
    print(f'{len(df):,} spectra, {int(df["sdss_specprimary"].sum()):,} primary')
    return out_path


def load_catalog(primary_only: bool = False) -> pd.DataFrame:
    """The slim catalog, cached. Read-only -- see `_base.LocalCatalog`."""
    return _CATALOG.frame(primary_only)


def spectra_within_many(
    ra: Sequence[float],
    dec: Sequence[float],
    radius_arcsec: float = 2.0,
) -> pd.DataFrame:
    """Every SDSS spectrum within `radius_arcsec` of each input position.

    One row per (input, spectrum), the input's position in `input_index`.
    """
    return _CATALOG.within_many(ra, dec, radius_arcsec, filtered=False)


def _summaries(spectra: pd.DataFrame, n_inputs: int) -> pd.DataFrame:
    """Per input: how many spectra, which classes, the median redshift of each."""
    out = pd.DataFrame({'sdss_n_spectra': np.zeros(n_inputs, dtype='int64'),
                        'sdss_classes': pd.Series([None] * n_inputs, dtype='object'),
                        'sdss_class_z': pd.Series([None] * n_inputs, dtype='object')})
    if spectra.empty:
        return out
    cls = spectra.assign(sdss_class=spectra['sdss_class'].astype(str))
    z = cls.groupby(['input_index', 'sdss_class'])['sdss_z'].median()
    for i, g in z.groupby(level=0):
        per_class = {c: round(float(v), 5) for (_, c), v in g.items() if np.isfinite(v)}
        out.at[i, 'sdss_classes'] = json.dumps(sorted(g.index.get_level_values(1)))
        out.at[i, 'sdss_class_z'] = json.dumps(per_class, sort_keys=True)
    counts = spectra['input_index'].value_counts()
    out.loc[counts.index, 'sdss_n_spectra'] = counts.values
    return out


def cone_search_many(
    ra: Sequence[float],
    dec: Sequence[float],
    radius_arcsec: float = 2.0,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Per input: the nearest SDSS spectrum, plus a summary of all of them.

    The nearest spectrum's columns are one representative row, not a
    preference; `sdss_n_spectra`, `sdss_classes` and `sdss_class_z` cover
    every spectrum within `radius_arcsec`.
    """
    nearest = _CATALOG.match_nearest(ra, dec, radius_arcsec, filtered=False, columns=columns)
    summary = _summaries(spectra_within_many(ra, dec, radius_arcsec), len(nearest))
    return pd.concat([nearest, summary.set_index(nearest.index)], axis=1)


def cone_search(
    ra: float,
    dec: float,
    radius_arcsec: float = 2.0,
    primary_only: bool = False,
) -> pd.DataFrame:
    """All SDSS spectra within `radius_arcsec` of one position, nearest first."""
    return _CATALOG.within(ra, dec, radius_arcsec, filtered=primary_only)


def spectrum_url(row: pd.Series) -> Optional[str]:
    """SAS URL of the spectrum file for a matched row, or None if unmatched."""
    url = row.get('sdss_spec_url')
    return None if url is None or pd.isna(url) or not url else str(url)


if __name__ == '__main__':
    print(f'Building {CATALOG_PARQUET.name}...')
    path = build_parquet()
    print(f'Wrote {path} ({path.stat().st_size / 1e6:.0f} MB)')
