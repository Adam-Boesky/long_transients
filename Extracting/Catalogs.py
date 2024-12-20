"""Objects that get the information from astrophysical catalogs."""
from PIL import Image
import atexit
import os
import tempfile
from io import StringIO, BytesIO
from typing import Dict, Iterable, Optional, Tuple, List
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table, join, vstack
from astropy.io import ascii, fits
from mastcasjobs import MastCasJobs

try:
    from Source_Extractor import Source_Extractor
    from utils import get_credentials
except ModuleNotFoundError:
    from .Source_Extractor import Source_Extractor
    from .utils import get_credentials

ZTF_CUTOUT_HALFWIDTH = 0.8888889 / 2


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

    @data.setter
    def data(self, table: Table):
        self._data = table
        self.rename_columns()

    def get_data(self) -> Table:
        raise NotImplementedError('This method must be implemented in subclass.')

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
    def __init__(
            self,
            ra_range: Tuple[float],
            dec_range: Tuple[float],
            prefetch: bool = False,
            catalog_bands: Iterable[str] = ('g', 'r', 'i'),
        ):
        super().__init__()
        self.catalog_name = 'PanSTARR'
        self.ra_range = ra_range    # (ra_min, ra_max) [deg]
        self.dec_range = dec_range  # (dec_min, dec_max) [deg]
        self.column_map = {'objID': 'PanSTARR_ID', 'raMean': 'ra', 'decMean': 'dec'}
        self.bands = catalog_bands

        assert (self.ra_range[1] > self.ra_range[0]) and (self.dec_range[1] > self.dec_range[0]), "The ranges must be \
            of the form (min, max) but are (max, min)."

        if prefetch:
            self.prefetch()

#     def _get_sql_query(self, table_name: str, query_type: str = 'vanilla') -> str:
#         """Get casjobs sql query to get PanSTARRS data.

#         NOTE: that this query gives us a 0.003 padding in RA and DEC to ensure we get all sources.
#         """
#         if query_type not in ('vanilla', 'fancy'):
#             raise ValueError(f"Query type {query_type} not recognized.")

#         if query_type == 'fancy':
#             query = f"""
# WITH
#     base AS (
#         SELECT DISTINCT o.objID, o.raMean, o.decMean
#         FROM ObjectThin o
#         WHERE o.raMean BETWEEN {self.ra_range[0] - 0.003} AND {self.ra_range[1] + 0.003}
#             AND o.decMean BETWEEN {self.dec_range[0] - 0.003} AND {self.ra_range[1] + 0.003}
#             AND (o.nStackDetections > 0 OR o.nDetections > 1)
#     ),"""

#             for band in self.bands:
#                 query += f"""
#     {band}_band AS (
#         SELECT
#             o.objID,
#             m.{band}KronMag,
#             m.{band}KronMagErr,
#             m.{band}PSFMag,
#             m.{band}PSFMagErr,
#             a.{band}psfLikelihood,
#             m.{band}infoFlag2,
#             ROW_NUMBER() OVER (
#                 PARTITION BY o.objID 
#                 ORDER BY 
#                     CASE WHEN m.{band}infoFlag2 = 0 THEN 0 ELSE 1 END,
#                     m.primaryDetection DESC
#             ) AS rn
#         FROM base o
#         INNER JOIN StackObjectThin m ON o.objID = m.objID
#         INNER JOIN StackObjectAttributes a ON o.objID = a.objID
#         WHERE (m.{band}infoFlag2 & 4) = 0
#     ),"""

#             for band in self.bands:
#                 query += f"""
#     {band}_table AS (
#         SELECT * FROM {band}_band WHERE rn = 1
#     ),"""

#             # Remove the trailing comma after the last CTE
#             query = query.rstrip(',') + "\n"
#             query += """

#     SELECT
#         base.objID,
#         base.raMean,
#         base.decMean,"""

#             for band in self.bands:
#                 query += f"""
#         {band}_table.{band}KronMag,
#         {band}_table.{band}KronMagErr,
#         {band}_table.{band}PSFMag,
#         {band}_table.{band}PSFMagErr,
#         {band}_table.{band}psfLikelihood,
#         {band}_table.{band}infoFlag2,"""

#             # Remove the trailing comma from the last column
#             query = query.rstrip(',')

#             query += f"""

#             INTO mydb.{table_name}

#             FROM base"""

#             for band in self.bands:
#                 query += f"""
#             LEFT JOIN {band}_table ON base.objID = {band}_table.objID"""

#             query += ";"

#         elif query_type == 'vanilla':
#             # Get the keys for the bands we want
#             band_mags_str = ''
#             for band in self.bands:
#                 band_mags_str += (
#                     f'\t\tm.{band}KronMag, m.{band}KronMagErr,\n'
#                     f'\t\tm.{band}ApMag, m.{band}ApMagErr,\n'
#                     f'\t\tm.{band}PSFMag, m.{band}PSFMagErr,\n'
#                     f'\t\ta.{band}psfLikelihood, m.{band}infoFlag2,\n'
#                 )

#             query = f"""
# WITH ranked AS (
#     SELECT
#         o.objID, o.raMean, o.decMean,
# {band_mags_str}\t\tm.primaryDetection,
#         ROW_NUMBER() OVER (PARTITION BY o.objID ORDER BY m.primaryDetection DESC) as rn into mydb.{table_name}
#     FROM ObjectThin o
#     INNER JOIN StackObjectThin m ON o.objID = m.objID
#     INNER JOIN StackObjectAttributes a ON o.objID = a.objID
#     WHERE o.raMean BETWEEN {self.ra_range[0] - 0.003} AND {self.ra_range[1] + 0.003}
#     AND o.decMean BETWEEN {self.dec_range[0] - 0.003} AND {self.dec_range[1] + 0.003}
#     AND (o.nStackDetections > 0 OR o.nDetections > 1)
# )
# SELECT * FROM ranked
# WHERE rn = 1
#             """
#         else:
#             query = None

#         return query

    def _get_band_query(self, band: str, tab_name: str) -> str:
        # Get the keys for the bands we want
        band_mags_str = (
        f'\tm.{band}KronMag, m.{band}KronMagErr,\n'
        f'\tm.{band}ApMag, m.{band}ApMagErr,\n'
        f'\tm.{band}PSFMag, m.{band}PSFMagErr,\n'
        f'\ta.{band}psfLikelihood, m.{band}infoFlag2,\n'
        )
        query = f"""
WITH ranked AS (
    SELECT
    o.objID, o.raMean, o.decMean,
    {band_mags_str}\tm.primaryDetection,
    ROW_NUMBER() OVER (PARTITION BY o.objID ORDER BY m.primaryDetection DESC) as rn into mydb.{tab_name}
    FROM ObjectThin o
    INNER JOIN StackObjectThin m ON o.objID = m.objID
    INNER JOIN StackObjectAttributes a ON o.objID = a.objID
    WHERE o.raMean BETWEEN {self.ra_range[0] - 0.003} AND {self.ra_range[1] + 0.003}
    AND o.decMean BETWEEN {self.dec_range[0] - 0.003} AND {self.dec_range[1] + 0.003}
    AND (o.nStackDetections > 0 OR o.nDetections > 1)
    AND (m.{band}infoFlag2 & 4) = 0
)
SELECT * FROM ranked
WHERE rn = 1
    """

        return query

    def _join_tables(self, tabs: List[Table]) -> Table:
        # Join the tables
        final_table = None
        for tab in tabs:
            if final_table is None:
                final_table = tab

            if len(final_table) == 0:
                final_table = tab
            elif len(tab) != 0:
                final_table = join(final_table, tab, join_type='outer')
            elif len(tab) == 0:
                for col in tab.colnames:
                    if col not in final_table.colnames:
                        final_table[col] = np.zeros(len(final_table)) * np.nan

        return final_table

    def _submit_table_query(self, jobs: MastCasJobs, band: str) -> str:
        # Make the table name
        ra_range_strs = [str(ra).replace('.', 'p').replace('-', 'n')[:6] for ra in self.ra_range]      # formatting coords
        dec_range_strs = [str(dec).replace('.', 'p').replace('-', 'n')[:6] for dec in self.dec_range]  # formatting coords
        table_name = f'pstarr_sources_ra{ra_range_strs[0]}_{ra_range_strs[1]}_dec{dec_range_strs[0]}_{dec_range_strs[1]}'

        # Submit the queries
        table_name = f'{band}_{table_name}'

        # Submit the query and monitor it
        band_query = self._get_band_query(band, tab_name=table_name)
        jobid = jobs.submit(band_query, task_name=f"{band}_PanSTARRS_DR2_TILE_QUERY")
        jobs.monitor(jobid)

        return table_name

    def get_data(self) -> Table:
        # Query Pan-STARRS DR2 using the specified RA and DEC range
        # 'panstarrs_dr2' is the catalog name for Pan-STARRS DR2
        # Limit the search to the desired RA and DEC range
        # return Catalogs.query_criteria(catalog='PanSTARRS', criteria=f'RA > {ra_range[0]} and RA < {ra_range[1]} and '
        #                                 f'DEC > {dec_range[0]} and DEC < {dec_range[1]}', limit=100)
        # SQL query to get sources within the RA and DEC range
        # Get the PS1 MAST username and password
        wsid, password = get_credentials('mast_login.txt')
        jobs = MastCasJobs(context="PanSTARRS_DR2", userid=wsid, password=password, request_type='POST')

        # Make the table name
        ra_range_strs = [str(ra).replace('.', 'p').replace('-', 'n')[:6] for ra in self.ra_range]      # formatting coords
        dec_range_strs = [str(dec).replace('.', 'p').replace('-', 'n')[:6] for dec in self.dec_range]  # formatting coords

        # Get tables for each band
        band_tables = []
        for band in self.bands:

            # Construct the table name
            table_name = f'{band}_pstarr_sources_ra{ra_range_strs[0]}_{ra_range_strs[1]}_dec{dec_range_strs[0]}_{dec_range_strs[1]}'

            # If the table is there, grab it. Else, submit the query and then grab it
            try:
                print(f'Retrieving {table_name} from MyDB!')
                band_tables.append(jobs.get_table(table_name))
            except ValueError:
                print(f'{table_name} not in MyDB, submitting request to make it!')
                self._submit_table_query(jobs, band)
                band_tables.append(jobs.get_table(table_name))
        
        # Join the band tables
        final_table = self._join_tables(band_tables)

        # Drop residual row number column
        if 'rn' in final_table.columns: final_table.remove_column('rn')

        # Fill in mask with nans
        final_table = Table(final_table, masked=True)
        for col in final_table.colnames:
            column = final_table[col]
            if np.issubdtype(column.dtype, np.number):
                # Convert column to float if numeric to allow np.nan
                final_table[col] = column.astype(float)
                final_table[col][column.mask] = np.nan
            else:
                # Convert to object dtype for non-numeric columns
                final_table[col] = column.astype(object)
                final_table[col][column.mask] = np.nan

        # Unmask the table (removes the mask attribute entirely)
        final_table = Table(final_table, masked=False)

        return final_table


class ZTF_Catalog(Catalog):
    def __init__(
            self,
            ra: float,
            dec: float,
            data_dir: Optional[str] = None,
            catalog_bands: Iterable[str] = ('g', 'r', 'i'),
        ):
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
        self.bands = catalog_bands
        for band in self.bands:
            fname, self.image_metadata[band] = self.download_image(ra, dec, band=band)
            if fname is not None:
                self.sextractors[band] = Source_Extractor(
                    fname,
                    band=band,
                    maglimit=self.image_metadata[band]['limiting_mag'],
                )
        if len(self.sextractors) == 0:
            raise ValueError(f"No {', '.join(catalog_bands)} found for the specified RA and DEC.")
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
        filtercode = f'z{band}'
        ra_min, ra_max = ra - ZTF_CUTOUT_HALFWIDTH, ra + ZTF_CUTOUT_HALFWIDTH
        dec_min, dec_max = dec - ZTF_CUTOUT_HALFWIDTH, dec + ZTF_CUTOUT_HALFWIDTH
        metadata_table = get_ztf_metadata(
            ra_range=(ra_min, ra_max),
            dec_range=(dec_min, dec_max),
            filter=filtercode,
            username=username,
            password=password,
        )
        if len(metadata_table) == 0:
            print(f"No {band} images found for the specified RA and DEC.")
            return None, None
        
        # Get the limiting magnitude
        limiting_mag = metadata_table['maglimit'].iloc[0]

        # Download the image
        qid = metadata_table['qid'].iloc[0]
        field = metadata_table['field'].iloc[0]
        paddedccdid = str(metadata_table['ccdid'].iloc[0]).zfill(2)
        paddedfield = str(field).zfill(6)
        fieldprefix = paddedfield[:6 - len(str(field))]
        image_url = (f'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/deep/{fieldprefix}/field{paddedfield}/'
                     f'{filtercode}/ccd{paddedccdid}/q{qid}/ztf_{paddedfield}_{filtercode}_c{paddedccdid}'
                     f'_q{qid}_refimg.fits')
        metadata_dict = {
            'field': paddedfield,
            'ccid': paddedccdid,
            'qid': qid,
            'filtercode': filtercode,
            'limiting_mag': limiting_mag,
        }

        print(f"Downloading image from {image_url}")

        # File path where the image will be saved
        file_path = os.path.join(self.data_dirpath, f'ztf_{paddedfield}_{filtercode}_c{paddedccdid}_q{qid}_refimg.fits')
        if os.path.exists(file_path):
            print(f"Image already downloaded and saved at {file_path}")
            return file_path, metadata_dict

        # Try downloading 3 times
        for _ in range(3):
            response = requests.get(image_url, auth=(username, password), stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            file.write(chunk)
                print(f"Image downloaded and saved at {file_path}")
                return file_path, metadata_dict

        print(f"Failed to download image from {image_url}. Status code: {response.status_code}")
        return None, None


def associate_tables_by_coordinates(
        table1: Table,
        table2: Table,
        max_sep: float = 2.0,
        prefix1: str = '',
        prefix2: str = '',
    ) -> Table:
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


@lru_cache(maxsize=None)
def get_ztf_metadata(
        ra_range: Tuple,
        dec_range: Tuple,
        filter: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> pd.DataFrame:
    """Query the ZTF metadata for a specific RA and DEC range."""
    # Set up the query
    if username is None or password is None:
        username, password = get_credentials('irsa_login.txt')

    # Make sure filtercode is setup correctly
    if filter[0] != 'z' or len(filter) == 1:
        filter = f'z{filter}'

    metadata_url = f"https://irsa.ipac.caltech.edu/ibe/search/ztf/products/deep?WHERE=ra>{ra_range[0]}+AND+ra<{ra_range[1]}+AND+dec>{dec_range[0]}+AND+dec<{dec_range[1]}+AND+filtercode='{filter}'"
    print(f"Querying metadata from {metadata_url}")

    # Query
    metadata_response = requests.get(metadata_url, auth=(username, password), params={'ct': 'csv'})
    metadata_table = pd.read_csv(StringIO(metadata_response.content.decode("utf-8")))
    metadata_table.sort_values(by=['nframes'], ascending=False, inplace=True)

    return metadata_table


### THE FOLLOWING FUNCTIONS ARE TAKEN FROM https://ps1images.stsci.edu/ps1image.html ###
def getimages(ra,dec,filters="grizy"):
    
    """Query ps1filenames.py service to get a list of images
    
    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """
    
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    table = Table.read(url, format='ascii')
    return table


def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False):
    
    """Get URL for images in the table
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """
    
    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg","png","fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra,dec,filters=filters)
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           f"ra={ra}&dec={dec}&size={size}&format={format}&type=stack")
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0,len(table)//2,len(table)-1]]
        for i, param in enumerate(["red","green","blue"]):
            url = url + "&{}={}".format(param,table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)
    return url


def get_pstarr_cutout(ra, dec, size=240, output_size=None, filter="g", format="fits"):
    """Get grayscale image at a sky position

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filter = string with filter to extract (one of grizy)
    format = data format
    Returns the image
    """
    if filter not in list("grizy"):
        raise ValueError("filter must be one of grizy")
    size = int(size)
    url = geturl(ra,dec,size=size,filters=filter,output_size=output_size,format=format)[0]
    print(f'Downloading image from {url}')
    flux = None
    for _ in range(3):
        try:
            # Get the data
            fh = fits.open(url)
            flux = fh[0].data

            # Scale back to flux
            # source = https://outerspace.stsci.edu/display/PANSTARRS/PS1+Image+Cutout+Service#PS1ImageCutoutService-ImportantFITSimageformat,WCS,andflux-scalingnotes
            if 'BSOFTEN' in fh[0].header and 'BOFFSET' in fh[0].header:
                bzero   =   fh[0].header['BZERO'] # Scaling: TRUE = BZERO + BSCALE * DISK
                bscale  =   fh[0].header['BSCALE'] # Scaling: TRUE = BZERO + BSCALE * DISK
                bsoften =   fh[0].header['BSOFTEN'] # Scaling: LINEAR = 2 * BSOFTEN * sinh(TRUE/a)
                boffset =   fh[0].header['BOFFSET'] # Scaling: UNCOMP = BOFFSET + LINEAR
                v = bzero + bscale * flux
                a = 2.5/np.log(10)
                x = v/a
                flux = boffset + bsoften * (np.exp(x) - np.exp(-x))
            
            # Load the WCS
            wcs = WCS(fh[0].header)

        except Exception as e:
            print(f'Pan-STARRS fits url query failed! Exception: {e}')

    return flux, wcs
