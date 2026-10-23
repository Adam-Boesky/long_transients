"""
Targeted repair for fields 000788 and 000859.

Root cause: ZTF_g.hdf5 was extracted before the PS1 parquet backend was
available (2026-08-31). When r and i were extracted later (2026-09-10) with
the new PSTARR_Local_Catalog, catalog_bands=['r','i'] was used (g already
existed), so PSTARR.hdf5 got no g-band photometry. This repair rebuilds
PSTARR.hdf5 with all three bands, then re-runs cross-matching and field merge.
"""

import os
import sys
sys.path.insert(0, '/n/home04/aboesky/berger/long_transients')

from astropy.table import Table

from Extracting.Catalogs import PSTARR_Local_Catalog
from Extracting.utils import get_data_path, prepare_table_for_write, load_ecsv
from Extracting.cross_match import cross_match_quadrant, merge_field, initializer

# Set cross_match module globals so cross_match_quadrant and merge_field work
import Extracting.cross_match as xm
DATA_PATH = get_data_path()
CATALOG_DIR = os.path.join(DATA_PATH, 'catalog_results')
xm.CATALOG_DIR = CATALOG_DIR
xm.BANDS = ['g', 'r', 'i']
xm.OVERWRITE = True
xm.N_THREADS = 1

AFFECTED = {
    '000788': '000788_01_2',
    '000859': '000859_01_2',
}


def rebuild_pstarr(quad_dirname: str):
    quad_dir = os.path.join(CATALOG_DIR, quad_dirname)
    print(f'\n=== Rebuilding PSTARR.hdf5 for {quad_dirname} ===')

    # Infer RA/Dec range from the existing ZTF_g.hdf5 (always present)
    ztf_g = Table.read(os.path.join(quad_dir, 'ZTF_g.hdf5'), path='data')
    ra_range  = (float(ztf_g['ra'].min()), float(ztf_g['ra'].max()))
    dec_range = (float(ztf_g['dec'].min()), float(ztf_g['dec'].max()))
    print(f'  RA: {ra_range}, Dec: {dec_range}')

    ps1 = PSTARR_Local_Catalog(ra_range, dec_range, catalog_bands=['g', 'r', 'i'])
    pstarr_data = ps1.data
    print(f'  PS1 rows: {len(pstarr_data)}, cols: {pstarr_data.colnames}')

    out_path = os.path.join(quad_dir, 'PSTARR.hdf5')
    prepare_table_for_write(pstarr_data).write(
        out_path, path='data', serialize_meta=True, overwrite=True
    )
    print(f'  Wrote {out_path}')


def main():
    # Step 1: rebuild PSTARR.hdf5 with all three bands
    for field, quad in AFFECTED.items():
        rebuild_pstarr(quad)

    # Step 2: re-cross-match quadrants (OVERWRITE=True already set in xm module)
    for field, quad in AFFECTED.items():
        quad_dirpath = os.path.join(CATALOG_DIR, quad)
        print(f'\n=== Cross-matching {quad} ===')
        cross_match_quadrant(quad_dirpath)

    # Step 3: re-merge field files
    all_quad_dirs = os.listdir(CATALOG_DIR)
    for field in AFFECTED:
        print(f'\n=== Merging field {field} ===')
        merge_field(field, all_quad_dirs, field_subdir='field_results')

    print('\nRepair complete. Now re-run the filter for fields 000788 and 000859.')


if __name__ == '__main__':
    main()
