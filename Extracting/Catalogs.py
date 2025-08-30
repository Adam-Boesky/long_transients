"""Objects that get the information from astrophysical catalogs."""
import subprocess
from PIL import Image
import atexit
import time
import random
import os
import tempfile
import traceback
from io import StringIO, BytesIO
from typing import Dict, Iterable, Optional, Tuple, List, Union
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table, join, vstack
from astropy.io import ascii, fits
from astropy.utils.data import clear_download_cache

try:
    from Source_Extractor import Source_Extractor
    from utils import get_credentials, MASTCASJOBS
except ModuleNotFoundError:
    from .Source_Extractor import Source_Extractor
    from .utils import get_credentials, MASTCASJOBS

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
            query_buffer: float = 0.003,
            overwrite_mydb: bool = False,
        ):
        super().__init__()
        self.catalog_name = 'PanSTARR'
        self.ra_range = ra_range    # (ra_min, ra_max) [deg]
        self.dec_range = dec_range  # (dec_min, dec_max) [deg]
        self.column_map = {'objID': 'PanSTARR_ID', 'raMean': 'ra', 'decMean': 'dec'}
        self.bands = catalog_bands
        self.query_buffer = query_buffer
        self.overwrite_mydb = overwrite_mydb

        assert (self.ra_range[1] >= self.ra_range[0]) and (self.dec_range[1] >= self.dec_range[0]), "The ranges must be \
            of the form (min, max) but are (max, min)."

        if prefetch:
            self.prefetch()

    def _get_band_query(self, band: str, tab_name: str) -> str:
        # Get the string for RAs
        if self.ra_range[1] < 360.0:
            ra_str = f"""o.raMean BETWEEN {self.ra_range[0] - self.query_buffer} AND {self.ra_range[1] + self.query_buffer}"""
        else:
            ra_str = f"""(
        (o.raMean BETWEEN {self.ra_range[0] - self.query_buffer} AND {self.ra_range[1] + self.query_buffer})
        OR (o.raMean BETWEEN {self.ra_range[0] - self.query_buffer - 360} AND {self.ra_range[1] + self.query_buffer - 360})
    )
"""

        # Get the keys for the bands we want
        band_mags_str = (
        f'\tm.{band}KronMag, m.{band}KronMagErr,\n'
        f'\tm.{band}PSFMag, m.{band}PSFMagErr,\n'
        f'\ta.{band}psfLikelihood, m.{band}infoFlag, m.{band}infoFlag2,\n'
        )
        query = f"""
WITH ranked AS (
    SELECT
    o.objID, o.raMean, o.decMean, o.qualityFlag, o.objInfoFlag,
    {band_mags_str}\tm.primaryDetection,
    ROW_NUMBER() OVER (PARTITION BY o.objID ORDER BY m.primaryDetection DESC) as rn
    FROM ObjectThin o
    INNER JOIN StackObjectThin m ON o.objID = m.objID
    INNER JOIN StackObjectAttributes a ON o.objID = a.objID
    WHERE {ra_str}
    AND o.decMean BETWEEN {self.dec_range[0] - self.query_buffer} AND {self.dec_range[1] + self.query_buffer}
    AND (o.nStackDetections > 0 OR o.nDetections > 1)
    AND (m.{band}infoFlag2 & 4) = 0
)
SELECT * FROM ranked
WHERE rn = 1 into mydb.{tab_name}
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

                # First cast the common columns to the same dtype
                for k, _ in final_table.items():
                    if k in tab.columns:
                        final_table[k] = final_table[k].astype(tab[k].dtype)

                # Fill missing values in 'gKronMag' with a placeholder, e.g., -999 or another sentinel value
                final_table = final_table.filled(-999)

                # Join!
                final_table = join(final_table, tab, join_type='outer')

            elif len(tab) == 0:
                for col in tab.colnames:
                    if col not in final_table.colnames:
                        final_table[col] = np.zeros(len(final_table)) * np.nan

        return final_table

    def _submit_table_query(self, band: str, table_name: str) -> int:

        # Submit the query and monitor it
        band_query = self._get_band_query(band, tab_name=table_name)
        jobid = MASTCASJOBS.submit(band_query, task_name=f"{band}_PanSTARRS_DR2_TILE_QUERY")
        print(f'Submitted query for {table_name} with id={jobid}')
        for attempt in range(3):
            try:
                MASTCASJOBS.monitor(jobid)
            except Exception as e:
                if attempt == 2:
                    raise e

        return jobid

    def get_data(self) -> Table:
        # Query Pan-STARRS DR2 using the specified RA and DEC range
        # 'panstarrs_dr2' is the catalog name for Pan-STARRS DR2
        # Limit the search to the desired RA and DEC range
        # return Catalogs.query_criteria(catalog='PanSTARRS', criteria=f'RA > {ra_range[0]} and RA < {ra_range[1]} and '
        #                                 f'DEC > {dec_range[0]} and DEC < {dec_range[1]}', limit=100)
        # SQL query to get sources within the RA and DEC range

        # Make the table name
        ra_range_strs = [str(ra).replace('.', 'p').replace('-', 'n')[:6] for ra in self.ra_range]      # formatting coords
        dec_range_strs = [str(dec).replace('.', 'p').replace('-', 'n')[:6] for dec in self.dec_range]  # formatting coords

        # Get tables for each band
        band_tables = []
        for band in self.bands:

            # Construct the table name
            ra_range_strs = [str(ra).replace('.', 'p').replace('-', 'n')[:6] for ra in self.ra_range]      # formatting coords
            dec_range_strs = [str(dec).replace('.', 'p').replace('-', 'n')[:6] for dec in self.dec_range]  # formatting coords
            table_name = f'{band}_pstarr_sources_ra{ra_range_strs[0]}_{ra_range_strs[1]}_dec{dec_range_strs[0]}_{dec_range_strs[1]}'

            # If overwriting myDB, drop the table
            if self.overwrite_mydb:
                print(f'Overwriting mydb.{table_name}')
                MASTCASJOBS.drop_table_if_exists(table_name)

            # If the table is there, grab it. Else, submit the query and then grab it
            for _ in range(3):
                try:

                    # If the table is not in mydb, submit a request to make it
                    mydb_table_list = MASTCASJOBS.list_tables()
                    if table_name not in mydb_table_list:
                        print(f'{table_name} not in MyDB, submitting request to make it!')
                        self._submit_table_query(band, table_name)

                    # Retrieve the table
                    print(f'Retrieving {table_name} from MyDB!')
                    band_tables.append(MASTCASJOBS.get_table(table_name))
                    break

                except Exception as e:
                    print(f'Exception retrieving {table_name} from MyDB printed below. Trying again.')
                    print(f'Exception type: {type(e).__name__}')
                    print(f'Exception message: {str(e)}')
                    print('Traceback:')
                    traceback.print_exc()

                    # Check for Astropy's "not enough free space" error
                    if isinstance(e, OSError) and "Not enough free space" in str(e):
                       print('WARNING: Not enough free space in cache. Clearing Astropy download cache to free up space...')
                       clear_download_cache()

                    # Delete some tables if the local database is too full
                    mydb_table_list = MASTCASJOBS.list_tables()
                    if len(mydb_table_list) > 50:
                        delete_prop = 0.5
                        print(f'WARNING: mydb appears to be pretty cluttered. Deleting some ({delete_prop * 100}% of '
                              'mydb) first...')
                        tabs_to_delete = random.sample(mydb_table_list, k=int(len(mydb_table_list) * delete_prop))
                        for t in tabs_to_delete:
                            MASTCASJOBS.drop_table_if_exists(t)

        # Join the band tables
        final_table = self._join_tables(band_tables)

        # Drop residual row number column
        if 'rn' in final_table.columns: final_table.remove_column('rn')

        # Fill in mask with nans
        final_table = Table(final_table, masked=True)
        for col in final_table.colnames:
            column = final_table[col]
            if (np.issubdtype(column.dtype, np.number) and col not in ('PanSTARR_ID', 'objID')) or ('flag' in column.name):
                # Take care of mask first
                final_table[col] = column.astype(object)
                final_table[col][column.mask] = np.nan

                # Convert column to float if numeric to allow np.nan
                final_table[col] = column.astype(float)
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
            band: str,
            data_dir: Optional[str] = None,
            image_metadata: Dict[str, Dict] = {},
        ):
        super().__init__()
        self.catalog_name = 'ZTF'
        self.column_map = {'objID': 'PanSTARR_ID', 'raMean': 'ra', 'decMean': 'dec'}
        self._image_metadata = image_metadata

        # Initial ra, dec values to use if there is no given metadata
        self.init_ra, self.init_dec = ra, dec

        # Set up the data directory and download the images
        if data_dir is None:
            self.data_dirpath = tempfile.mkdtemp()
            atexit.register(self._cleanup)
        else:
            if not os.path.exists(data_dir):
                raise ValueError(f"Data directory {data_dir} does not exist.")
            self.data_dirpath = data_dir
        self.band = band
        fname = self.download_image(band=band)
        if fname is not None:
            self.sextractor = Source_Extractor(
                fname,
                band=band,
            )
        if fname is None:
            raise ValueError(f"No ZTF {band} image found for the specified RA and DEC.")
        self.ra_range, self.dec_range = self.sextractor.get_coord_range()

    @property
    def image_metadata(self) -> Dict[str, Dict]:
        if len(self._image_metadata) == 0:
            # Get the authentication credentials
            username, password = get_credentials('irsa_login.txt')

            # Get a metadata table for the specified RA and DEC
            filtercode = f'z{self.band}'
            ra_min, ra_max = self.init_ra - ZTF_CUTOUT_HALFWIDTH, self.init_ra + ZTF_CUTOUT_HALFWIDTH
            dec_min, dec_max = self.init_dec - ZTF_CUTOUT_HALFWIDTH, self.init_dec + ZTF_CUTOUT_HALFWIDTH
            self._image_metadata = get_ztf_metadata_from_coords(
                ra_range=(ra_min, ra_max),
                dec_range=(dec_min, dec_max),
                filter=filtercode,
                username=username,
                password=password,
            ).iloc[0].to_dict()

            # Take metadata out of list form
            for k, v in self._image_metadata.items():
                self._image_metadata[k] = v

        return self._image_metadata

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
        table = self.sextractor.get_data_table()
        if big_table is None:
            big_table = table
        else:
            big_table = associate_tables_by_coordinates(big_table, table)

        return big_table

    def download_image(self, band: str) -> str:

        # Get the authentication credentials
        username, password = get_credentials('irsa_login.txt')

        # Get a metadata table for the specified RA and DEC
        filtercode = f'z{band}'
        if len(self.image_metadata) == 0:
            print(f"No {band} images found for the specified RA and DEC.")
            return None

        # Download the image
        qid = self.image_metadata['qid']
        field = self.image_metadata.get('fieldid', self.image_metadata.get('field'))
        paddedccdid = str(self.image_metadata['ccdid']).zfill(2)
        paddedfield = str(field).zfill(6)
        fieldprefix = paddedfield[:6 - len(str(field))]
        image_url = (f'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/deep/{fieldprefix}/field{paddedfield}/'
                     f'{filtercode}/ccd{paddedccdid}/q{qid}/ztf_{paddedfield}_{filtercode}_c{paddedccdid}'
                     f'_q{qid}_refimg.fits')

        print(f"Downloading image from {image_url}")

        # File path where the image will be saved
        file_path = os.path.join(self.data_dirpath, f'ztf_{paddedfield}_{filtercode}_c{paddedccdid}_q{qid}_refimg.fits')

        # Check if the image already exists and is valid
        if os.path.exists(file_path):
            print(f"Checking if image already downloaded and valid: {file_path.split('/')[-1]}")

            # Make sure that the image is not truncated
            result = subprocess.run(
                ["fitscheck", "--ignore-missing", "--compliance", file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if "truncated" in result.stdout.lower() or "truncated" in result.stderr.lower():
                print(f"{file_path.split('/')[-1]} is not valid. Deleting and re-downloading...")
                os.remove(file_path)
            else:
                print(f"Image is valid at {file_path}")
                return file_path

        # Try downloading 3 times
        for _ in range(3):
            response = requests.get(image_url, auth=(username, password), stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            file.write(chunk)
                print(f"Image downloaded and saved at {file_path}")
                return file_path

        print(f"Failed to download image from {image_url}. Status code: {response.status_code}")
        return None


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
def get_ztf_metadata_from_coords(
        ra_range: Tuple,
        dec_range: Tuple,
        filter: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> pd.DataFrame:
    """Query the ZTF metadata for a specific RA and DEC range."""
    # Set up the query
    if username is None or password is None:
        username, password = get_credentials('irsa_login.txt')

    # Construct url
    metadata_url = f"https://irsa.ipac.caltech.edu/ibe/search/ztf/products/deep?WHERE=ra>{ra_range[0]}+AND+ra<{ra_range[1]}+AND+dec>{dec_range[0]}+AND+dec<{dec_range[1]}"
    if filter is not None:

        # Make sure filtercode is setup correctly
        if filter[0] != 'z' or len(filter) == 1:
            filter = f'z{filter}'
        metadata_url = metadata_url + f"+AND+filtercode='{filter}'"

    print(f"Querying metadata from {metadata_url}")

    # Query
    metadata_response = requests.get(metadata_url, auth=(username, password), params={'ct': 'csv'})
    metadata_table = pd.read_csv(StringIO(metadata_response.content.decode("utf-8")))
    metadata_table.sort_values(by=['nframes'], ascending=False, inplace=True, ignore_index=True)

    return metadata_table.reset_index(drop=True)


def get_ztf_metadata_from_metadata(
        ztf_metadata: Dict[str, Union[int, str]],
        username: Optional[str] = None,
        password: Optional[str] = None,
        verbose: int = 1,
    ) -> pd.DataFrame:
    """Query the ZTF metadata for a specific RA and DEC range."""
    # Set up the query
    if username is None or password is None:
        username, password = get_credentials('irsa_login.txt')

    # Collect info from metadata and construct url
    ztf_metadata['fieldid'] = str(ztf_metadata.get('field', ztf_metadata.get('fieldid'))).zfill(6)
    metadata_url = f"https://irsa.ipac.caltech.edu/ibe/search/ztf/products/deep?WHERE=field={ztf_metadata['fieldid']}+AND+"
    if ztf_metadata.get('ccdid', -1) >= 0:
        metadata_url += f"ccdid={str(ztf_metadata['ccdid']).zfill(2)}+AND+"
    if ztf_metadata.get('qid', -1) >= 0:
        metadata_url += f"qid={ztf_metadata['qid']}+AND+"
    if not ztf_metadata.get('filtercode', ztf_metadata.get('filter', None)) is None:
        # Make sure the filter code starts with a z
        filtercode = ztf_metadata.get('filtercode', ztf_metadata.get('filter'))
        if filtercode[0] != 'z' or len(filtercode) == 1:
            filtercode = f'z{filtercode}'

        metadata_url += f"filtercode=\'{filtercode}\'+AND+"

    metadata_url = metadata_url[:-5]  # delete the +AND+ from the end of the url

    # Query
    if verbose > 0:
        print(f"Querying metadata from {metadata_url}")
    metadata_response = requests.get(metadata_url, auth=(username, password), params={'ct': 'csv'})
    metadata_table = pd.read_csv(StringIO(metadata_response.content.decode("utf-8")))
    if len(metadata_table) > 1:
        metadata_table.sort_values(by=['nframes'], ascending=False, inplace=True, ignore_index=True)

    return metadata_table.reset_index(drop=True)


def ztf_image_exists(
        ztf_metadata: Dict[str, Union[int, str]],
        username: Optional[str] = None,
        password: Optional[str] = None,
        verbose: int = 1,
    ) -> pd.DataFrame:
    return len(get_ztf_metadata_from_metadata(ztf_metadata, username=username, password=password, verbose=verbose)) > 0


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
            if 'RADESYS' not in fh[0].header:
                fh[0].header.rename_keyword('RADECSYS', 'RADESYS')
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
