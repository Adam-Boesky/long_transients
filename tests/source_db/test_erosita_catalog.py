"""eROSITA DR2 counterpart matching, on a synthetic catalogue (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

from Source_Analysis.catalogs import _base, erosita

# A position in eROSITA-DE (Galactic l ~ 230 deg) and one outside it (l ~ 60 deg).
DE = SkyCoord(l=230 * u.deg, b=30 * u.deg, frame='galactic').icrs
RU = SkyCoord(l=60 * u.deg, b=30 * u.deg, frame='galactic').icrs


def _offset(c: SkyCoord, arcsec: float) -> SkyCoord:
    return c.directional_offset_by(0 * u.deg, arcsec * u.arcsec)


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    """Two X-ray sources near DE, two arcminutes apart."""
    pos = [DE, _offset(DE, 120)]
    cols = [
        fits.Column('IAUNAME', '23A', array=['1eRASS J000000.0+000000', '1eRASS J000000.0+000200']),
        fits.Column('UID', 'K', array=[300000100001, 300000100002]),
        fits.Column('RA', 'D', array=[p.ra.deg for p in pos]),
        fits.Column('DEC', 'D', array=[p.dec.deg for p in pos]),
        fits.Column('POS_ERR', 'D', array=[2.0, 6.0]),
        fits.Column('DET_LIKE_0', 'E', array=[20.0, 6.0]),
        fits.Column('EXT', 'E', array=[0.0, 0.0]),
        fits.Column('EXT_LIKE', 'E', array=[0.0, 0.0]),
        fits.Column('ML_FLUX_1', 'E', array=[1e-13, 3e-14]),
        fits.Column('ML_FLUX_ERR_1', 'E', array=[2e-14, 1e-14]),
        fits.Column('ML_EXP_1', 'E', array=[250.0, 240.0]),
        # >0 a DR1 match, <0 a weak association, 0 none.
        fits.Column('UID_DR1', 'K', array=[100000700001, 0]),
        fits.Column('FLAG_OPT', 'B', array=[0, 1]),
    ]
    fits_path = tmp_path / 'cat.fits'
    fits.BinTableHDU.from_columns(cols).writeto(fits_path)
    parquet = erosita.build_parquet(fits_path, tmp_path / 'cat.parquet')
    monkeypatch.setattr(erosita, '_CATALOG', _base.LocalCatalog(parquet, prefix='erosita'))
    return parquet


def test_footprint_is_the_german_half_of_the_sky():
    assert erosita.in_footprint([DE.ra.deg, RU.ra.deg], [DE.dec.deg, RU.dec.deg]).tolist() == [True, False]


def test_build_keeps_every_row_with_native_types(catalog):
    df = pd.read_parquet(catalog)
    assert len(df) == 2
    assert df['erosita_flag_opt'].tolist() == [False, True]
    assert df['erosita_uid'].iloc[0] == 300000100001        # 12 digits survive intact


def test_cone_is_a_fixed_ten_arcsec(catalog):
    probes = [_offset(DE, 9), _offset(DE, 11), _offset(_offset(DE, 120), 3), RU]
    out = erosita.cone_search_many([p.ra.deg for p in probes], [p.dec.deg for p in probes])
    assert out['erosita_matched'].tolist() == [True, False, True, False]
    assert out['erosita_covered'].tolist() == [True, True, True, False]
    assert out.loc[0, 'erosita_uid_dr1'] == 100000700001
    # An unmatched row carries no payload, so its fields cannot be mistaken for a match.
    assert out.loc[1, ['erosita_name', 'erosita_flux', 'erosita_sep_arcsec']].isna().all()


def test_enrichment_uses_the_erosita_radius_not_the_shared_one(catalog, monkeypatch):
    from Source_Analysis import enrich_combined_tabs as enrich
    p = _offset(DE, 9)
    out = enrich.crossmatch_columns([p.ra.deg], [p.dec.deg], radius_arcsec=1.5, catalogs=['erosita'])
    assert out['erosita_matched'].tolist() == [True]
