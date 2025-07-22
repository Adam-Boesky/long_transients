#!/usr/bin/env python3
import os
import re
import sys
import pickle
import argparse
import numpy as np

from typing import List, Callable
from astropy.wcs import WCS
from astropy.io import ascii
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, match_coordinates_sky

try:
    from utils import get_data_path, true_nearby, metadata_from_field_dirname, load_ecsv
except ModuleNotFoundError:
    from .utils import get_data_path, true_nearby, metadata_from_field_dirname, load_ecsv
import multiprocessing

sys.stdout.flush()


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

    # Add columns from table2 to unmatched_table1 with NaN values, cast cols to correct types, and stack the tables
    for col in table2.colnames:
        if 'PanSTARR_ID' not in col:
            unmatched_table1[col] = float('nan')
    for col in table1.colnames:
        if 'PanSTARR_ID' not in col:
            unmatched_table2[col] = float('nan')
    associated_table['Catalog'] = 'Both'
    unmatched_table1['Catalog'] = prefix1
    unmatched_table2['Catalog'] = prefix2

    # Cast PSTARR_primaryDetection to float64 in associated_table, unmatched_table1, unmatched_table2
    for table in [associated_table, unmatched_table1, unmatched_table2]:
        if 'PSTARR_primaryDetection' in table.colnames:
            table['PSTARR_primaryDetection'] = table['PSTARR_primaryDetection'].astype(np.float64)
    combined_table = vstack([associated_table, unmatched_table1, unmatched_table2])

    # Identify rows where 'association_separation_arcsec' should be NaN
    nan_mask = combined_table['association_separation_arcsec'].mask
    combined_table['association_separation_arcsec'][nan_mask] = float('nan')

    # Reorder combined_table so that 'ra' and 'dec' are the first two columns
    combined_table['ra'] = np.nanmean(np.vstack((combined_table['ZTF_ra'], combined_table['PSTARR_ra'])), axis=0)
    combined_table['dec'] = np.nanmean(np.vstack((combined_table['ZTF_dec'], combined_table['PSTARR_dec'])), axis=0)
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
    ztf_coords = SkyCoord(ra=combined_table[in_ztf_mask]['ZTF_ra'], dec=combined_table[in_ztf_mask]['ZTF_dec'], unit='deg')
    _, sep_to_other_sources, _ = match_coordinates_sky(ztf_coords, pstarr_coords)
    indices_in_ztf = np.where(in_ztf_mask)[0]
    sep_condition = sep_to_other_sources.arcminute > 1.0
    indices_sep_condition = np.where(sep_condition)[0]
    original_indices = indices_in_ztf[indices_sep_condition]
    combined_table['Catalog_Flag'][original_indices] = 3

    # Check if PSTARR coords are nan in ZTF
    pstarr_pix_coords = wcs.world_to_pixel(pstarr_coords)
    pstarr_xs = pstarr_pix_coords[0].round().astype(int)
    pstarr_ys = pstarr_pix_coords[1].round().astype(int)
    in_wcs = (pstarr_xs < ztf_nan_mask.shape[0]) & (pstarr_ys < ztf_nan_mask.shape[1]) & (pstarr_xs > 0) & (pstarr_ys > 0)
    is_nan_in_ztf = np.array([
        true_nearby(
            row=y,
            column=x,
            radius=5,
            mask=ztf_nan_mask,
        ) if b else True for x, y, b in zip(pstarr_xs, pstarr_ys, in_wcs)
    ])
    indices_in_pstarr = np.where(in_pstarr_mask)[0]
    indices_is_nan_in_ztf = np.where(is_nan_in_ztf)[0]
    original_indices = indices_in_pstarr[indices_is_nan_in_ztf]
    combined_table['Catalog_Flag'][original_indices] = 3

    # Check if ZTF coords are nan in ZTF
    ztf_pix_coords = wcs.world_to_pixel(ztf_coords)
    ztf_xs = ztf_pix_coords[0].round().astype(int)
    ztf_ys = ztf_pix_coords[1].round().astype(int)
    is_nan_in_ztf = np.array([
        true_nearby(
            row=y,
            column=x,
            radius=15,  # nan within 15 pixels is bad!
            mask=ztf_nan_mask,
        ) for x, y in zip(ztf_xs, ztf_ys)
    ])
    indices_in_ztf = np.where(in_ztf_mask)[0]
    indices_is_nan_in_ztf = np.where(is_nan_in_ztf)[0]
    original_indices = indices_in_ztf[indices_is_nan_in_ztf]
    combined_table['Catalog_Flag'][original_indices] = 3

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


def first_not_nan(v: np.ndarray) -> np.float64:
    """Helper function to get the first non-nan value from a 1D numpy array. Returns nan if all values are nan."""
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    not_nans = v[~np.isnan(v) & (v != -999)]
    return not_nans[0] if len(not_nans) > 0 else np.nan


def first_not_nan_vectorized(arr):
    """Vectorized version of first_not_nan applied column-wise."""
    return np.apply_along_axis(first_not_nan, axis=0, arr=arr)


def collapse_nonunique_srcs(tab: Table) -> Table:
    """Collapse the non-unique rows of a table based on duplicate coordinates and Pan-STARR ID."""
    
    # Step 1: Collapse by (ra, dec)
    tab.sort(['ra', 'dec'])
    grouped_by_coords = tab.group_by(['ra', 'dec'])

    # Extract indices for slicing
    indices = grouped_by_coords.groups.indices
    collapsed_data = {}

    # Process each column efficiently
    for col in tab.colnames:
        col_data = np.array(tab[col])  # Extract column as NumPy array
        grouped_col_data = [col_data[indices[i]:indices[i+1]] for i in range(len(indices)-1)]
        if 'PanSTARR_ID' not in col:
            collapsed_data[col] = np.array([first_not_nan(group) for group in grouped_col_data])
        else:
            collapsed_data[col] = np.array([c[0] for c in grouped_col_data])

    collapsed_by_coords = Table(collapsed_data)

    # Step 2: Collapse by Pan-STARR ID
    collapsed_by_coords.sort(['PanSTARR_ID'])
    grouped_by_id = collapsed_by_coords.group_by(['PanSTARR_ID'])

    # Extract indices for slicing
    indices = grouped_by_id.groups.indices
    collapsed_final_data = {}

    # Process each column again
    for col in collapsed_by_coords.colnames:
        col_data = np.array(collapsed_by_coords[col])
        grouped_col_data = [col_data[indices[i]:indices[i+1]] for i in range(len(indices)-1)]
        collapsed_final_data[col] = np.array([first_not_nan(group) for group in grouped_col_data])

    return Table(collapsed_final_data)


def cross_match_quadrant(quadrant_dirpath: str):
    """Cross match the g, r, i catalogs with the panstarrs catalog for a single quadrant."""
    if OVERWRITE is False:
        if (
            (
                os.path.exists(os.path.join(quadrant_dirpath, 'g_associated.ecsv')) or not
                os.path.exists(os.path.join(quadrant_dirpath, 'ZTF_g.ecsv'))
            )
            and (
                os.path.exists(os.path.join(quadrant_dirpath, 'r_associated.ecsv')) or not
                os.path.exists(os.path.join(quadrant_dirpath, 'ZTF_r.ecsv'))
            )
            and (
                os.path.exists(os.path.join(quadrant_dirpath, 'i_associated.ecsv')) or not
                os.path.exists(os.path.join(quadrant_dirpath, 'ZTF_i.ecsv'))
            )
        ):
            print(f'Skipping {quadrant_dirpath.split('/')[-1]} because all bands are associated and overwrite is set to False.')
            return

    print(f'Cross matching quadrant {quadrant_dirpath.split("/")[-1]}')
    pstar_tab = load_ecsv(os.path.join(quadrant_dirpath, 'PSTARR.ecsv'), careful_load=True)

    # Cast flags to floats TODO: delete after running extraction again
    for col in pstar_tab.colnames:
        if 'flag' in col.lower():
            pstar_tab[col] = pstar_tab[col].astype(float)

    # Collapse the non-unique Pan-STARRS sources
    pstar_tab.remove_column('primaryDetection')  # we can remove primaryDetection cuz it's a nuissance and always true
    pstar_tab = collapse_nonunique_srcs(pstar_tab)

    # Iterate through the ZTF band catalogs
    for band, fname in zip(BANDS, [f'ZTF_{band}.ecsv' for band in BANDS]):

        # Check if already associated
        if os.path.exists(os.path.join(quadrant_dirpath, f'{band}_associated.ecsv')) and OVERWRITE is False:
            print(
                f'Skipping {band} association for {quadrant_dirpath.split("/")[-1]} because it is already associated ' +
                'and overwrite is set to False.'
            )
            continue

        full_fpath = os.path.join(quadrant_dirpath, fname)
        if os.path.exists(full_fpath):

            # Load the ZTF data, nan mask, and WCS
            ztf_tab = load_ecsv(full_fpath)
            ztf_nan_mask = np.load(os.path.join(quadrant_dirpath, 'nan_masks', f'ZTF_{band}_nan_mask.npy'))
            with open(os.path.join(quadrant_dirpath, 'WCSs', f'ZTF_{band}_wcs.pkl'), 'rb') as f:
                wcs = pickle.load(f)

            # Save the associated table
            associated_tab = associate_tables(ztf_tab, pstar_tab, ztf_nan_mask, wcs)
            associated_tab['PSTARR_PanSTARR_ID'] = associated_tab['PSTARR_PanSTARR_ID'].astype(object)
            associated_tab['PSTARR_PanSTARR_ID'][associated_tab['PSTARR_PanSTARR_ID'].mask] = np.nan 
            associated_tab.write(
                os.path.join(quadrant_dirpath, f'{band}_associated.ecsv'),
                format='ascii.ecsv',
                overwrite=True,
            )


def merge_field(field_name: str, quad_dirs: List[str], field_subdir: str = 'field_results'):
    """Merge all quadrants from a field into one table."""

    # Get the quadrant directories
    field_quad_dirs = [quad_dir for quad_dir in quad_dirs if quad_dir.startswith(field_name) and (re.match(r'[0-9]{6}_[0-9]{2}_[0-9]', quad_dir) is not None)]
    if len(field_quad_dirs) == 0:
        return

    print(f'Merging quadrant results for field {field_name}')
    for band in BANDS:

        # Check if the merged file already exists
        if OVERWRITE is False:
            if os.path.exists(os.path.join(CATALOG_DIR, field_subdir, f'{field_name}_{band}.ecsv')):
                print(
                    f'Skipping merging for {band} band for {field_name} because it already exists and overwrite is ' +
                    'set to False.'
                )
                return

        # Start with first available quadrant
        getting_first_tab = True
        while getting_first_tab and len(field_quad_dirs) > 0:
            first_tab_path = os.path.join(CATALOG_DIR, field_quad_dirs[0], f'{band}_associated.ecsv')
            if os.path.exists(first_tab_path):
                tab = load_ecsv(os.path.join(CATALOG_DIR, field_quad_dirs[0], f'{band}_associated.ecsv'), careful_load=True)
                getting_first_tab = False

                # Add on the field info for each source
                field_quadrant_metadata = metadata_from_field_dirname(field_quad_dirs[0])
                for k, v in field_quadrant_metadata.items():
                    tab[k] = v
            else:
                print(f'WARNING: {first_tab_path} does not exist. Skipping...')
                field_quad_dirs.remove(field_quad_dirs[0])

        if len(field_quad_dirs) == 0:
            print(f'WARNING: No valid quadrants found for field {field_name} in the {band} band. Skipping...')
            continue

        # Merge the rest of them
        for fqdir in field_quad_dirs[1:]:
            fqpath = os.path.join(CATALOG_DIR, fqdir, f'{band}_associated.ecsv')
            if os.path.exists(fqpath):
                tab_to_stack = load_ecsv(fqpath, careful_load=True)

                # Add on the field info for each source
                field_quadrant_metadata = metadata_from_field_dirname(fqdir)
                for k, v in field_quadrant_metadata.items():
                    tab_to_stack[k] = v

                # Stack
                tab = vstack((tab, tab_to_stack))
            else:
                print(f'WARNING: {fqpath} does not exist. Skipping...')
        tab.write(
            os.path.join(CATALOG_DIR, field_subdir, f'{field_name}_{band}.ecsv'),
            format='ascii.ecsv',
            overwrite=True
        )


def associate_quadrants(initializer: Callable):
    """Cross match the g, r, i catalogs with the panstarrs catalog for each quadrant."""
    quad_dirpaths = [os.path.join(CATALOG_DIR, quad_dir) for quad_dir in os.listdir(CATALOG_DIR) \
                     if re.match(r'[0-9]{6}_[0-9]{2}_[0-9]', quad_dir)]

    with multiprocessing.Pool(processes=N_THREADS, initializer=initializer, initargs=(CATALOG_DIR, BANDS, OVERWRITE, N_THREADS)) as pool:
        pool.map(cross_match_quadrant, quad_dirpaths)


def merge_fields(initializer: Callable, output_directory: str = 'field_results'):
    """Merge all quadrants from a field into one table."""
    # Make separate directory for field results
    field_results_dir = os.path.join(CATALOG_DIR, output_directory)
    if not os.path.exists(field_results_dir):
        os.makedirs(field_results_dir)

    # Get the directories for each quadrant and the field names
    quad_dirs = os.listdir(CATALOG_DIR)
    field_names = list(set([quad_dir.split('_')[0] for quad_dir in quad_dirs]))

    # Merge fields in parallel
    with multiprocessing.Pool(processes=N_THREADS, initializer=initializer, initargs=(CATALOG_DIR, BANDS, OVERWRITE, N_THREADS)) as pool:
        pool.starmap(
            merge_field,
            [(field_name, quad_dirs, output_directory) for field_name in field_names]
        )


def initializer(catalog_dir, bands, overwrite, n_threads):
    """Initializer function for setting globals in the multiprocessing pool."""
    global CATALOG_DIR
    global BANDS
    global OVERWRITE
    global N_THREADS
    CATALOG_DIR = catalog_dir
    BANDS = bands
    OVERWRITE = overwrite
    N_THREADS = n_threads


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
        '-ow',
        '--overwrite',
        action='store_true',
        default=False,
        help='Overwrite the existing outputs.'
    )
    parser.add_argument(
        '-nt',
        '--n_threads',
        type=int,
        default=8,
        help='The number of threads to use.'
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

    # Set up initializer and call
    global CATALOG_DIR
    global BANDS
    global OVERWRITE
    global N_THREADS
    CATALOG_DIR = os.path.join(get_data_path(), args.catalog_subdirectory)
    BANDS = list(args.bands)
    OVERWRITE = args.overwrite
    N_THREADS = args.n_threads

    # Actions
    if args.associate_quadrants:
        print('Associating sources in quadrants...')
        associate_quadrants(initializer)
    if args.merge_fields:
        print('Merging quadrants from all fields...')
        merge_fields(initializer, output_directory=args.output_directory)


if __name__=='__main__':
    cross_match()
