import numpy as np

from typing import Dict
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table, vstack


def add_prefixed_coords(table: Table, band: str) -> Table:
    new_table = table.copy()
    new_table[f'{band}_ra'] = new_table['ra']
    new_table[f'{band}_dec'] = new_table['dec']
    return new_table


# def match_band_tables(tables: Dict[str, Table], max_sep: float = 1.0) -> Table:
    # # Get first table and add band prefixes to ra and dec columns
    # bands = list(tables.keys())
    # final_table = add_prefixed_coords(tables[bands[0]], bands[0])

    # for b in bands[1:]:

    #     # Get table and add band prefixes to ra and dec columns
    #     band_tab = tables[b].copy()
    #     band_tab = add_prefixed_coords(band_tab, b)

    #     # Match
    #     coords1 = SkyCoord(ra=final_table['ra'], dec=final_table['dec'], unit='deg')
    #     coords2 = SkyCoord(ra=band_tab['ra'], dec=band_tab['dec'], unit='deg')
    #     idx, sep2d, _ = match_coordinates_sky(coords1, coords2)
    #     mask = sep2d.arcsecond <= max_sep

    #     new_table = final_table[mask]
    #     for col in band_tab.colnames:
    #         new_table[col] = band_tab[idx[mask]][col]

    #     # Find the indices of the unmatched rows in both tables
    #     unmatched_idx1 = ~mask
    #     unmatched_idx2 = np.full(len(band_tab), True)
    #     unmatched_idx2[idx[mask]] = False

    #     # Create tables for the unmatched rows
    #     unmatched_table1 = final_table[unmatched_idx1]
    #     unmatched_table2 = band_tab[unmatched_idx2]

    #     # Add columns from table2 to unmatched_table1 with NaN values and stack the tables
    #     for col in band_tab.colnames:
    #         unmatched_table1[col] = float('nan')
    #     for col in final_table.colnames:
    #         unmatched_table2[col] = float('nan')
    #     new_table = vstack([new_table, unmatched_table1, unmatched_table2])

    #     # Reorder combined_table so that 'ra' and 'dec' are the first two columns
    #     new_table['ra'] = np.nanmean([new_table['ra'], ])
    #     combined_table['ra'] = (combined_table['ZTF_ra'] + combined_table['PSTARR_ra']) / 2
    #     combined_table['dec'] = (combined_table['ZTF_dec'] + combined_table['PSTARR_dec']) / 2
    #     cols = combined_table.colnames
    #     cols.remove('ra')
    #     cols.remove('dec')
    #     combined_table = combined_table[['ra', 'dec'] + cols]

    #     # Check if panstarrs is nan -- we will approximate this by checking if there are any sources within an arcminute
    #     pstarr_coords = SkyCoord(ra=combined_table[in_pstarr_mask]['PSTARR_ra'], dec=combined_table[in_pstarr_mask]['PSTARR_dec'], unit='deg')
    #     ztf_coords = SkyCoord(ra=combined_table[in_ztf_mask]['ZTF_dec'], dec=combined_table[in_ztf_mask]['ZTF_dec'], unit='deg')
    #     _, sep_to_other_sources, _ = match_coordinates_sky(ztf_coords, pstarr_coords)
    #     combined_table['Catalog_Flag'][in_ztf_mask][sep_to_other_sources.arcminute > 1.0] = 3


    # return combined_table


# if __name__=='__main__':
#     match_between_bands()
