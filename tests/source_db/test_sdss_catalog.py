"""The SDSS DR20 catalog build (`Source_Analysis.catalogs.sdss.build_parquet`)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from Source_Analysis.catalogs import sdss

URL = 'https://sas/{}.fits'


def _shared(n, specobjid, zwarning, primary, cls):
    return {
        'SPECOBJID': specobjid, 'CLASS': cls, 'SUBCLASS': [''] * n,
        'Z': np.linspace(0.1, 0.5, n), 'Z_ERR': np.full(n, 1e-4),
        'ZWARNING': np.array(zwarning, dtype='int32'),
        'CLASS_NOQSO': cls, 'Z_NOQSO': np.linspace(0.1, 0.5, n),
        'ZWARNING_NOQSO': np.zeros(n, dtype='int32'),
        'SPECPRIMARY': np.array(primary, dtype='int16'),
        'SN_MEDIAN_ALL': np.full(n, 5.0), 'SURVEY': ['x'] * n,
        'MJD': np.full(n, 59500, dtype='int32'), 'RUN2D': ['r'] * n,
    }


@pytest.fixture
def inputs(tmp_path):
    # SDSS-I--IV: a good spectrum, a sky fibre, an unplugged fibre.
    old = _shared(3, np.array([11, 12, 13], dtype='int64'), [0, 1, 128 | 4], [1, 1, 1],
                  ['GALAXY', 'GALAXY', 'QSO'])
    old.update({'PLUG_RA': [10.0, 11.0, 12.0], 'PLUG_DEC': [0.0, 0.0, 0.0],
                'PLATE': np.array([1, 1, 1], dtype='int32'),
                'FIBERID': np.array([1, 2, 3], dtype='int32')})
    Table(old).write(tmp_path / 'specObj.fits')

    # SDSS-V: inspected object, plain object, unassigned fibre, ops_sky fibre.
    new = _shared(4, ['2001', '2002', '2003', '2004'], [0, 0, 132, 5], [1, 1, 0, 1],
                  ['GALAXY', 'STAR', 'QSO', 'GALAXY'])
    new.update({'RACAT': [20.0, 21.0, 22.0, 23.0], 'DECCAT': [1.0, 1.0, 1.0, 1.0],
                'FIELD': np.array([5, 5, 5, 5], dtype='int32'),
                'SDSS_ID': np.array([7, 8, -999, 9], dtype='int64'),
                'OBJTYPE': ['QSO', 'STAR', '', 'SKY']})
    Table(new).write(tmp_path / 'spAll.fits')

    Table({'sdss_id': np.array([7, 8], dtype='int64'), 'has_vi': [True, False],
           'class_best': ['QSO_BAL', 'STAR'], 'z_vi': np.array([2.1, -9999.0]),
           'z_conf_vi': np.array([3, 0], dtype='int16')}).write(tmp_path / 'vi.fits')

    ids = [11, 12, 13, 2001, 2002, 2003, 2004]
    pd.DataFrame({
        'sdss_phase': [4 if i < 2000 else 5 for i in ids],
        'coadd': [None if i < 2000 else 'daily' for i in ids],
        'instrument': 'boss',
        'specobjid': [str(i) for i in ids],
        'sdss_id': [None if i < 2000 else {2003: -999}.get(i, i) for i in ids],
        'sas_url': [URL.format(i) for i in ids],
    }).to_parquet(tmp_path / 'allspec.parquet')
    return tmp_path


def _build(root):
    out = root / 'out.parquet'
    sdss.build_parquet(out, specobj=root / 'specObj.fits', spall=root / 'spAll.fits',
                       vi=root / 'vi.fits', allspec=root / 'allspec.parquet')
    return pd.read_parquet(out).set_index('sdss_specobjid')


def test_sky_unplugged_and_unassigned_fibres_are_dropped(inputs):
    df = _build(inputs)
    assert sorted(df.index) == ['11', '2001', '2002']


def test_every_spectrum_carries_its_allspec_url(inputs):
    df = _build(inputs)
    assert df.loc['2002', 'sdss_spec_url'] == URL.format('2002')
    assert df.loc['11', 'sdss_spec_url'] == URL.format('11')


def test_secure_visual_inspection_overrides_the_pipeline(inputs):
    df = _build(inputs)
    assert df.loc['2001', ['sdss_class', 'sdss_class_pipeline', 'sdss_class_source']].tolist() \
        == ['QSO_BAL', 'GALAXY', 'visual']
    assert df.loc['2001', 'sdss_z'] == pytest.approx(2.1)
    # Not inspected: the pipeline stands.
    assert df.loc['2002', ['sdss_class', 'sdss_class_source']].tolist() == ['STAR', 'pipeline']


def test_a_spectrum_missing_from_allspec_is_an_error(inputs):
    pd.read_parquet(inputs / 'allspec.parquet').iloc[1:].to_parquet(inputs / 'allspec.parquet')
    with pytest.raises(ValueError, match='not in allspec'):
        _build(inputs)


def test_within_many_lists_every_row_per_input(tmp_path):
    from Source_Analysis.catalogs._base import LocalCatalog
    pd.DataFrame({'ra': [10.0, 10.0001, 50.0], 'dec': [0.0, 0.0, 0.0],
                  'x_v': [1, 2, 3]}).to_parquet(tmp_path / 'c.parquet')
    cat = LocalCatalog(tmp_path / 'c.parquet', prefix='x')
    got = cat.within_many([50.0, np.nan, 10.0, 80.0], [0.0, 0.0, 0.0, 0.0], 1.0)
    # Input 2 sees both nearby rows, nearest first; input 0 sees one; the NaN
    # and the empty patch see none.
    assert got['input_index'].tolist() == [0, 2, 2]
    assert got['x_v'].tolist() == [3, 1, 2]
