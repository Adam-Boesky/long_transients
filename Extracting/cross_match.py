#!/usr/bin/env python3
import os
import re
import pickle
import argparse
import numpy as np

from typing import List
from astropy.wcs import WCS
from astropy.io import ascii
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, match_coordinates_sky

try:
    from utils import get_data_path, true_nearby
except ModuleNotFoundError:
    from .utils import get_data_path, true_nearby


def associate_tables(table1: Table, table2: Table, ztf_nan_mask: np.ndarray, wcs: WCS, max_sep: float = 1.0) -> Table:
    """
    Associate rows of two Astropy tables by ensuring that the objects are within a specified separation (in arcseconds).

    Parameters:
    - table1: First Astropy Table with RA and DEC columns.
    - table2: Second Astropy Table with RA and DEC columns.
    - max_sep: Maximum separation in arcseconds to consider two objects as the same.

    Returns:
    - A new Astropy Table with rows from table1 and corresponding rows from table2 that are within the specified separation.
    """

    # Make copies of the input tables
    table1 = table1.copy()
    table2 = table2.copy()

    # Add prefixes to all the columns
    prefix1 = 'ZTF'
    prefix2 = 'PSTARR'
    for colname in table1.colnames:
        prefixed_colname1 = f'{prefix1}_{colname}'
        if colname in ('ra', 'dec') and prefixed_colname1 not in table1.colnames:
            table1[prefixed_colname1] = table1[colname]  # Make a copy of the ra and dec columns
        elif colname[:len(prefix1)] != prefix1:  # Make sure that the prefixed column isn't already in the table
            table1.columns[colname].name = prefixed_colname1
    for colname in table2.colnames:
        prefixed_colname2 = f'{prefix2}_{colname}'
        if colname in ('ra', 'dec') and prefixed_colname2 not in table2.colnames:
            table2[prefixed_colname2] = table2[colname]  # Make a copy of the ra and dec columns
        elif colname[:len(prefix2)] != prefix2:  # Make sure that the prefixed column isn't already in the table
                table2.columns[colname].name = prefixed_colname2

    coords1 = SkyCoord(ra=table1['ra'], dec=table1['dec'], unit='deg')
    coords2 = SkyCoord(ra=table2['ra'], dec=table2['dec'], unit='deg')
    idx, sep2d, _ = match_coordinates_sky(coords1, coords2)
    mask = sep2d.arcsecond <= max_sep

    associated_table = table1[mask]
    for col in table2.colnames:
        associated_table[col] = table2[idx[mask]][col]

    # Add a column for the separations between the sources
    associated_table['association_separation_arcsec'] = sep2d.arcsecond[mask]

    # Find the indices of the unmatched rows in both tables
    unmatched_idx1 = ~mask
    unmatched_idx2 = np.full(len(table2), True)
    unmatched_idx2[idx[mask]] = False

    # Create tables for the unmatched rows
    unmatched_table1 = table1[unmatched_idx1]
    unmatched_table2 = table2[unmatched_idx2]

    # Add columns from table2 to unmatched_table1 with NaN values and stack the tables
    for col in table2.colnames:
        unmatched_table1[col] = float('nan')
    for col in table1.colnames:
        unmatched_table2[col] = float('nan')
    associated_table['Catalog'] = 'Both'
    unmatched_table1['Catalog'] = prefix1
    unmatched_table2['Catalog'] = prefix2
    combined_table = vstack([associated_table, unmatched_table1, unmatched_table2])

    # Identify rows where 'association_separation_arcsec' should be NaN
    nan_mask = combined_table['association_separation_arcsec'].mask
    combined_table['association_separation_arcsec'][nan_mask] = float('nan')

    # Reorder combined_table so that 'ra' and 'dec' are the first two columns
    combined_table['ra'] = (combined_table['ZTF_ra'] + combined_table['PSTARR_ra']) / 2
    combined_table['dec'] = (combined_table['ZTF_dec'] + combined_table['PSTARR_dec']) / 2
    cols = combined_table.colnames
    cols.remove('ra')
    cols.remove('dec')
    combined_table = combined_table[['ra', 'dec'] + cols]

    # Flagging the catalogs --- use the following key:
    # 0 — Both catalogs
    # 1 — ZTF, not Pan-STARRS, and not nan in either
    # 2 — Pan-STARRS, not ZTF, and not nan in either
    # 3 — Nan in either Pan-STARRS or ZTF
    combined_table['Catalog_Flag'] = combined_table['Catalog'].copy()
    combined_table['Catalog_Flag'][combined_table['Catalog_Flag'] == 'Both'] = 0
    in_ztf_mask = combined_table['Catalog_Flag'] == 'ZTF'
    in_pstarr_mask = combined_table['Catalog_Flag'] == 'PSTARR'

    # Check if panstarrs is nan -- we will approximate this by checking if there are any sources within an arcminute
    pstarr_coords = SkyCoord(ra=combined_table[in_pstarr_mask]['PSTARR_ra'], dec=combined_table[in_pstarr_mask]['PSTARR_dec'], unit='deg')
    ztf_coords = SkyCoord(ra=combined_table[in_ztf_mask]['ZTF_dec'], dec=combined_table[in_ztf_mask]['ZTF_dec'], unit='deg')
    _, sep_to_other_sources, _ = match_coordinates_sky(ztf_coords, pstarr_coords)
    combined_table['Catalog_Flag'][in_ztf_mask][sep_to_other_sources.arcminute > 1.0] = 3

    # Check if nan in ZTF
    pstarr_pix_coords = wcs.world_to_pixel(pstarr_coords)
    pstarr_xs = pstarr_pix_coords[0].round().astype(int)
    pstarr_ys = pstarr_pix_coords[1].round().astype(int)
    in_wcs = (pstarr_xs < ztf_nan_mask.shape[0]) & (pstarr_ys < ztf_nan_mask.shape[1])
    is_nan_in_ztf = np.array([
        true_nearby(
            row=y,
            column=x,
            radius=5,
            mask=ztf_nan_mask,
        ) if b else True for x, y, b in zip(pstarr_xs, pstarr_ys, in_wcs)
    ])
    combined_table['Catalog_Flag'][in_pstarr_mask][is_nan_in_ztf] = 3

    # Map the remaining clean sources
    combined_table['Catalog_Flag'][combined_table['Catalog_Flag'] == 'ZTF'] = 1
    combined_table['Catalog_Flag'][combined_table['Catalog_Flag'] == 'PSTARR'] = 2
    combined_table['Catalog_Flag'] = combined_table['Catalog_Flag'].astype(int)

    # Add x and y to PSTARR sources and make a new column
    combined_table['x'] = combined_table['ZTF_x']
    combined_table['y'] = combined_table['ZTF_y']
    combined_table['x'][in_pstarr_mask] = pstarr_xs
    combined_table['y'][in_pstarr_mask] = pstarr_ys

    # Make sure that the entire maglimit column is not NaN
    for band in ('g', 'r', 'i'):
        k = f'ZTF_{band}_mag_limit'
        if k in combined_table.colnames:
            combined_table[k] = combined_table[k][~np.isnan(combined_table[k])][0]

    return combined_table


def cross_match_quadrant(quadrant_dirpath: str):
    """Cross match the g, r, i catalogs with the panstarrs catalog for a single quadrant."""
    print(f'Cross matching quadrant {quadrant_dirpath.split("/")[-1]}')
    pstar_tab = ascii.read(os.path.join(quadrant_dirpath, 'PSTARR.ecsv'))

    # Iterate through the ZTF band catalogs
    for band, fname in zip(BANDS, [f'ZTF_{band}.ecsv' for band in BANDS]):

        full_fpath = os.path.join(quadrant_dirpath, fname)
        if os.path.exists(full_fpath):

            # Load the ZTF data, nan mask, and WCS
            ztf_tab = ascii.read(full_fpath)
            ztf_nan_mask = np.load(os.path.join(quadrant_dirpath, 'nan_masks', f'ZTF_{band}_nan_mask.npy'))
            with open(os.path.join(quadrant_dirpath, 'WCSs', f'ZTF_{band}_wcs.pkl'), 'rb') as f:
                wcs = pickle.load(f)

            # Save the associated table
            associated_tab = associate_tables(ztf_tab, pstar_tab, ztf_nan_mask, wcs)
            associated_tab.write(
                os.path.join(quadrant_dirpath, f'{band}_associated.ecsv'),
                format='ascii.ecsv',
                overwrite=True
            )


def merge_field(field_name: str, field_quad_dirs: List[str], field_subdir: str = 'field_results'):
    """Merge all quadrants from a field into one table."""
    print(f'Merging quadrant results for field {field_name}')
    for band in BANDS:
        tab = ascii.read(os.path.join(CATALOG_DIR, field_quad_dirs[0], f'{band}_associated.ecsv'))
        for fqdir in field_quad_dirs[1:]:
            fqpath = os.path.join(CATALOG_DIR, fqdir, f'{band}_associated.ecsv')
            if os.path.exists(fqpath):
                tab = vstack((tab, ascii.read(fqpath)))
            else:
                print(f'Warning: {fqpath} does not exist. Skipping...')
        tab.write(
            os.path.join(CATALOG_DIR, field_subdir, f'{field_name}_{band}.ecsv'),
            format='ascii.ecsv',
            overwrite=True
        )


def associate_quadrants():
    """Cross match the g, r, i catalogs with the panstarrs catalog for each quadrant."""
    for quad_dir in os.listdir(CATALOG_DIR):
        if re.match(r'[0-9]{6}_[0-9]{2}_[0-9]', quad_dir):
            cross_match_quadrant(os.path.join(CATALOG_DIR, quad_dir))


def merge_fields(output_directory: str = 'field_results'):
    """Merge all quadrants from a field into one table."""
    # Make separate directory for field results
    field_results_dir = os.path.join(CATALOG_DIR, output_directory)
    if not os.path.exists(field_results_dir):
        os.makedirs(field_results_dir)

    # Get the directories for each quadrant and the field names
    quad_dirs = os.listdir(CATALOG_DIR)
    field_names = list(set([quad_dir.split('_')[0] for quad_dir in quad_dirs]))

    # Associate and store each field
    for field_name in field_names:
        field_quad_dirs = [quad_dir for quad_dir in quad_dirs if quad_dir.startswith(field_name) and (re.match(r'[0-9]{6}_[0-9]{2}_[0-9]', quad_dir) is not None)]
        if len(field_quad_dirs) > 0:
            merge_field(field_name, field_quad_dirs)


def cross_match():

    # Configure parser
    parser = argparse.ArgumentParser(description='Cross match ZTF catalogs with Pan-STARRS catalog.')
    parser.add_argument(
        '-aq',
        '--associate_quadrants',
        action='store_true',
        default=False,
        help='Associate the g, r, i catalogs with the Pan-STARRS catalog.'
    )
    parser.add_argument(
        '-mf',
        '--merge_fields',
        action='store_true',
        default=False,
        help='Merge all quadrants from a field into one table.'
    )
    parser.add_argument(
        '-outdir',
        '--output_directory',
        type=str,
        default='field_results',
        help='The output directory for merged field results.'
    )
    parser.add_argument(
        '-catdir',
        '--catalog_subdirectory',
        type=str,
        default='catalog_results',
        help='The subdirectory for TODO.'
    )
    parser.add_argument(
        '-bs',
        '--bands',
        type=str,
        default='gri',
        help='The photometric bands to store for.'
    )
    
    args = parser.parse_args()

    # Set up
    global CATALOG_DIR
    global BANDS
    CATALOG_DIR = os.path.join(get_data_path(), args.catalog_subdirectory)
    BANDS = list(args.bands)

    # Actions
    if args.associate_quadrants:
        print('Associating sources in quadrants...')
        associate_quadrants()
    if args.merge_fields:
        print('Merging quadrants from all fields...')
        merge_fields(args.output_directory)


if __name__=='__main__':
    cross_match()
