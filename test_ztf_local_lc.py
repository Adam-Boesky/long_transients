"""Tests for ZTF_LC class."""
import sys
import numpy as np
import pandas as pd

sys.path.append('/n/home04/aboesky/berger/long_transients')

from Source_Analysis.ZTF_Local_LC import ZTF_LC

DATA_DIR = '/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/ztf_lc_dir'
SAMPLE_FILE = f'{DATA_DIR}/0/field000812/ztf_000812_zg_c01_q1_dr24.parquet'


def get_sample_source():
    """Pull the first objectid and coordinates from a known parquet file."""
    df = pd.read_parquet(SAMPLE_FILE, columns=['objectid', 'objra', 'objdec'])
    row = df.iloc[0]
    return int(row['objectid']), float(row['objra']), float(row['objdec'])


def test_coord_lookup():
    print('\n--- Test 1: coordinate lookup ---')
    objectid, ra, dec = get_sample_source()
    print(f'  target: objectid={objectid}, ra={ra:.4f}, dec={dec:.4f}')

    lc = ZTF_LC(DATA_DIR, ra=ra, dec=dec)
    t = lc.lc
    print(f'  got {len(t)} epochs, bands: {set(t["filtercode"])}')
    print(f'  columns: {t.colnames}')
    assert len(t) > 0, 'Expected at least one epoch'
    assert 'hmjd' in t.colnames
    assert 'mag' in t.colnames
    print('  PASSED')


def test_objectid_with_coords():
    print('\n--- Test 2: objectid + coord lookup (disambiguates by objectid) ---')
    objectid, ra, dec = get_sample_source()
    print(f'  target: objectid={objectid}, ra={ra:.4f}, dec={dec:.4f}')

    lc = ZTF_LC(DATA_DIR, ra=ra, dec=dec, objectid=objectid)
    t = lc.lc
    print(f'  got {len(t)} epochs, bands: {set(t["filtercode"])}')
    assert len(t) > 0, 'Expected at least one epoch'
    assert np.all(t['ztf_id'] == objectid), 'Unexpected objectid in result'
    print('  PASSED')


if __name__ == '__main__':
    test_coord_lookup()
    test_objectid_with_coords()
    print('\nAll tests passed.')
