"""Objects that get the information from astrophysical catalogs."""
import atexit
import os
import tempfile
from io import StringIO, BytesIO
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table, join, vstack
from astropy.io import ascii
from mastcasjobs import MastCasJobs

try:
    from Source_Extractor import Source_Extractor
    from utils import get_credentials
except ModuleNotFoundError:
    from .Source_Extractor import Source_Extractor
    from .utils import get_credentials


class Catalog():
    def __init__(self):
        self.catalog_name = None
        self.column_map = {}
        self.ra_range = None        # (ra_min, ra_max) [deg]
        self.dec_range = None       # (dec_min, dec_max) [deg]
        self._data: Table = None

    @property
    def data(self) -> Table:
        if self._data is None:
            self._data = self.get_data()
            self.rename_columns()
        return self._data

    def rename_columns(self):
        """Function used to rename the columns of the catalog to some common values."""
        for old_col, new_col in self.column_map.items():
            if old_col in self._data.columns:
                self._data.columns[old_col].name = new_col

    def get_coordinate_range(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return self.ra_range, self.dec_range

    def prefetch(self):
        self.data


class PSTARR_Catalog(Catalog):
    def __init__(self, ra_range: Tuple[float], dec_range: Tuple[float], prefetch: bool = False):
        super().__init__()
        self.catalog_name = 'PanSTARR'
        self.ra_range = ra_range    # (ra_min, ra_max) [deg]
        self.dec_range = dec_range  # (dec_min, dec_max) [deg]
        self.column_map = {'objID': 'PanSTARR_ID', 'raMean': 'ra', 'decMean': 'dec'}

        assert (self.ra_range[1] > self.ra_range[0]) and (self.dec_range[1] > self.dec_range[0]), "The ranges must be \
            of the form (min, max) but are (max, min)."

        if prefetch:
            self.prefetch()

    def get_data(self) -> Table:
        # Query Pan-STARRS DR2 using the specified RA and DEC range
        # 'panstarrs_dr2' is the catalog name for Pan-STARRS DR2
        # Limit the search to the desired RA and DEC range
        # return Catalogs.query_criteria(catalog='PanSTARRS', criteria=f'RA > {ra_range[0]} and RA < {ra_range[1]} and '
        #                                 f'DEC > {dec_range[0]} and DEC < {dec_range[1]}', limit=100)
        # SQL query to get sources within the RA and DEC range
        # Get the PS1 MAST username and password
        wsid, password = get_credentials('mast_login.txt')

        # Set up casjobs object
        jobs = MastCasJobs(context="PanSTARRS_DR2", userid=wsid, password=password)

        # Make the table name
        ra_range_strs = [str(ra).replace('.', 'p').replace('-', 'n')[:6] for ra in self.ra_range]      # formatting coords
        dec_range_strs = [str(dec).replace('.', 'p').replace('-', 'n')[:6] for dec in self.dec_range]  # formatting coords
        table_name = f'pstarr_sources_ra{ra_range_strs[0]}_{ra_range_strs[1]}_dec{dec_range_strs[0]}_{dec_range_strs[1]}'

        # If the table is there, grab it. Else, submit the query and then grab it
        try:
            final_table = jobs.get_table(table_name, format='csv')
        except ValueError:

            # Execute the query and wait for it to complete
            # note that this query gives us a 0.003 padding in RA and DEC to ensure we get all sources
            print(f"Querying Pan-STARRS DR2 for the RA and DEC range in the ZTF image.")
            query = f"""
WITH ranked AS (
    SELECT
        o.objID, o.raMean, o.decMean,
        m.gKronMag, m.rKronMag, m.iKronMag,
        m.gKronMagErr, m.rKronMagErr, m.iKronMagErr,
        m.gApMag, m.rApMag, m.iApMag,
        m.gApMagErr, m.rApMagErr, m.iApMagErr,
        m.gPSFMag, m.rPSFMag, m.iPSFMag,
        m.gPSFMagErr, m.rPSFMagErr, m.iPSFMagErr,
        m.primaryDetection,
        ROW_NUMBER() OVER (PARTITION BY o.objID ORDER BY m.primaryDetection DESC) as rn into mydb.{table_name}
    FROM ObjectThin o
    INNER JOIN StackObjectThin m ON o.objID = m.objID
    WHERE o.raMean BETWEEN {self.ra_range[0] - 0.003} AND {self.ra_range[1] + 0.003}
    AND o.decMean BETWEEN {self.dec_range[0] - 0.003} AND {self.dec_range[1] + 0.003}
    AND (o.nStackDetections > 0 OR o.nDetections > 1)
)
SELECT * FROM ranked
WHERE rn = 1
            """
            jobid = jobs.submit(query, task_name=f"PanSTARRS_DR2_RA_DEC_Query")
            jobs.monitor(jobid)

            # Get the results of the query
            final_table = jobs.get_table(table_name, format='csv')

        # Drop residual row number column
        if 'rn' in final_table.columns: final_table.remove_column('rn')


        # # # Take the primary detections for each object, and if there are none, take the first detection
        # # final_table = None
        # # for id in np.unique(results['objID']):
        # #     objs = results[results['objID'] == id]

        # #     # If there's a primary detection for the object ID, take it
        # #     if np.any(objs['primaryDetection'] == 1):
        # #         objs = objs[objs['primaryDetection'] == 1]
        # #     else:
        # #         objs = objs[[0]]
        # #     final_table = objs if final_table == None else vstack((final_table, objs))

        # results_list.append(results)

        # # Combine the results into one table by matching objID
        # combined_results: Table = results_list[0]
        # for result in results_list[1:]:
        #     combined_results = join(combined_results, result, keys='objID', join_type='inner', metadata_conflicts='silent')

        return final_table


class ZTF_Catalog(Catalog):
    def __init__(self, ra: float, dec: float, data_dir: Optional[str] = None, catalog_bands: Optional[Iterable[str]] = None):
        super().__init__()
        self.catalog_name = 'ZTF'
        self.column_map = {'objID': 'PanSTARR_ID', 'raMean': 'ra', 'decMean': 'dec'}
        self.sextractors: Dict[str, Source_Extractor] = {}
        self.image_metadata = {}

        # Set up the data directory and download the images
        if data_dir is None:
            self.data_dirpath = tempfile.mkdtemp()
            atexit.register(self._cleanup)
        else:
            if not os.path.exists(data_dir):
                raise ValueError(f"Data directory {data_dir} does not exist.")
            self.data_dirpath = data_dir
        self.bands = catalog_bands if catalog_bands is not None else ('g', 'r', 'i')
        for band in self.bands:
            fname, self.image_metadata[band] = self.download_image(ra, dec, band=band)
            if fname is not None:
                self.sextractors[band] = Source_Extractor(fname, band=band)
        if len(self.sextractors) == 0:
            raise ValueError("No g, r, or i found for the specified RA and DEC.")
        self.ra_range, self.dec_range = self.sextractors[self.bands[0]].get_coord_range()

    def _cleanup(self):
        if os.path.exists(self.data_dirpath):
            for root, dirs, files in os.walk(self.data_dirpath, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
                os.rmdir(self.data_dirpath)

    def get_data(self) -> Table:
        big_table = None
        # for band in self.bands:  TODO: replace
        for band in self.bands:
            table = self.sextractors[band].get_data_table()
            if big_table is None:
                big_table = table
            else:
                big_table = associate_tables_by_coordinates(big_table, table)

        return big_table

    def download_image(self, ra: float, dec: float, band: str) -> Tuple[str, dict]:

        # Get the authentication credentials
        username, password = get_credentials('irsa_login.txt')

        # Get a metadata table for the specified RA and DEC
        cutout_halfwidth = 0.8888889 / 2  # 1/2 of the quadrant width in fields
        filtercode = f'z{band}'
        ra_min, ra_max = ra - cutout_halfwidth, ra + cutout_halfwidth
        dec_min, dec_max = dec - cutout_halfwidth, dec + cutout_halfwidth
        metadata_url = f"https://irsa.ipac.caltech.edu/ibe/search/ztf/products/deep?WHERE=ra>{ra_min}+AND+ra<{ra_max}+AND+dec>{dec_min}+AND+dec<{dec_max}+AND+filtercode='{filtercode}'"
        print(f"Querying metadata from {metadata_url}")
        metadata_response = requests.get(metadata_url, auth=(username, password), params={'ct': 'csv'})
        metadata_table = pd.read_csv(StringIO(metadata_response.content.decode("utf-8")))
        metadata_table.sort_values(by=['nframes'], ascending=False, inplace=True)
        if len(metadata_table) == 0:
            print(f"No {band} images found for the specified RA and DEC.")
            return None, None

        # Download the image
        qid = metadata_table['qid'].iloc[0]
        field = metadata_table['field'].iloc[0]
        paddedccdid = str(metadata_table['ccdid'].iloc[0]).zfill(2)
        paddedfield = str(field).zfill(6)
        fieldprefix = paddedfield[:6 - len(str(field))]
        image_url = f'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/deep/{fieldprefix}/field{paddedfield}/{filtercode}/ccd{paddedccdid}/q{qid}/ztf_{paddedfield}_{filtercode}_c{paddedccdid}_q{qid}_refimg.fits'
        metadata_dict = {'field': paddedfield, 'ccid': paddedccdid, 'qid': qid, 'filtercode': filtercode}

        print(f"Downloading image from {image_url}")

        # File path where the image will be saved
        file_path = os.path.join(self.data_dirpath, f'ztf_{paddedfield}_{filtercode}_c{paddedccdid}_q{qid}_refimg.fits')
        if os.path.exists(file_path):
            print(f"Image already downloaded and saved at {file_path}")
            return file_path, metadata_dict
        response = requests.get(image_url, auth=(username, password), stream=True)
        if response.status_code == 200:
            if os.path.exists(file_path):
                os.remove(file_path)
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            print(f"Image downloaded and saved at {file_path}")
        else:
            print(f"Failed to download image from {image_url}. Status code: {response.status_code}")

        return file_path, metadata_dict


def associate_tables_by_coordinates(table1: Table, table2: Table, max_sep: float = 2.0, prefix1: str = '', prefix2: str = '') -> Table:
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
    if prefix1 != '':
        for colname in table1.colnames:
            prefixed_colname1 = f'{prefix1}_{colname}'
            if colname in ('ra', 'dec') and prefixed_colname1 not in table1.colnames:
                table1[prefixed_colname1] = table1[colname]  # Make a copy of the ra and dec columns
            elif colname[:len(prefix1)] != prefix1:  # Make sure that the prefixed column isn't already in the table
                table1.columns[colname].name = prefixed_colname1
    if prefix2 != '':
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
    unmatched_table1['Catalog'] = prefix1 if prefix1 != '' else 'table1'
    unmatched_table2['Catalog'] = prefix2 if prefix2 != '' else 'table2'
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

    return combined_table
