import os
import pathlib
from contextlib import nullcontext

from astropy.io import ascii
from astropy.table import Table, MaskedColumn
from functools import lru_cache
from mastcasjobs import MastCasJobs
from typing import Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd

MAST_CREDENTIAL_FNAME = 'mast_dino_login.txt'
print(f'CasJobs will use the credentials from {MAST_CREDENTIAL_FNAME}')


def get_credentials(fname: str) -> Union[Tuple[str, str], str]:
    """Retrieves credentials from a specified file in my ~/vault/directory."""
    key_location = os.path.join(pathlib.Path.home(), f'vault/{fname}')
    return np.genfromtxt(key_location, dtype = 'str')


def img_flux_to_ab_mag(flux: np.ndarray, zero_point: float, fluxerr: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert image flux to AB magnitude.

    Parameters:
    flux (np.ndarray): The flux values of the image.
    zero_point (float): The zero point for the magnitude calculation.
    fluxerr (Optional[np.ndarray]): The flux error values of the image, by default None.

    Returns:
        1. The AB magnitudes corresponding to the input fluxes.
        2. If `fluxerr` is provided, the AB magnitude error.
    """
    mag = np.full_like(flux, np.nan, dtype=np.float64)
    positive_flux = flux > 0
    mag[positive_flux] = -2.5 * np.log10(flux[positive_flux]) + zero_point

    if fluxerr is not None:
        magerr = np.full_like(flux, np.nan, dtype=np.float64)
        magerr[positive_flux] = 2.5 * fluxerr[positive_flux] / (np.log(10) * flux[positive_flux])
        return mag, magerr

    return mag


def img_ab_mag_to_flux(mag: np.ndarray, zero_point: np.ndarray, magerr: Optional[np.ndarray] = None) -> np.ndarray:
    """Inverse of img_flux_to_ab_mag."""
    flux = 10 ** ((mag - zero_point) / -2.5)
    if magerr is not None:
        fluxerr = (magerr * (np.log(10) * flux)) / 2.5
        return flux, fluxerr
    return flux


def get_snr_from_mag(mag: np.ndarray, magerr: np.ndarray, zp: float) -> np.ndarray:
        flux, fluxerr = img_ab_mag_to_flux(mag, zero_point=zp, magerr=magerr)
        return flux / fluxerr


def on_cluster() -> bool:
    """Check if the code is running on the cluster."""
    return os.path.exists('/n/home04')


def get_data_path() -> str:
    """Get the path to the data directory."""
    if on_cluster():
        return '/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients'
    return '/Users/adamboesky/Research/long_transients/Data'


def true_nearby(row: int, column: int, radius: int, mask: np.ndarray) -> bool:

    # Adjust for negative indexing issues
    if radius > row:
        row = radius
    if radius > column:
        column = radius

    return np.any(
        mask[int(row - radius):int(row + radius + 1), int(column - radius):int(column + radius + 1)]
    )


def nan_nearby(row: int, column: int, radius: int, arr: np.ndarray) -> bool:
    """Check if there are any NaN values in the nearby pixels."""
    return true_nearby(row, column, radius, np.isnan(arr))


@lru_cache(maxsize=None)
def load_cached_table(table_path: str) -> Table:
    """Load a table from disk and cache it in memory."""
    return load_ecsv(table_path)


# Useful paths
RESULT_DIRPATH = os.path.join(get_data_path(), 'catalog_results')
MERGED_RESULTS_DIRPATH = os.path.join(RESULT_DIRPATH, 'field_results')

def get_n_quadrants_merged(field_name: str) -> Dict[str, int]:
    """Get the number of quadrants for a merged field. Returns a dictionary of the form {'band': number_of_quadrants}"""

    # Iterate through the extracted results and count the number of quadrants
    bands = ('g', 'r', 'i')
    quad_counts_per_band = {band: 0 for band in bands}
    for dirname in os.listdir(RESULT_DIRPATH):
    
        # If the field name is in the extracted directory
        if field_name == dirname[:6]:

            # Check if merged g, r, and i are in the directory
            for band in bands:
                if os.path.exists(os.path.join(RESULT_DIRPATH, dirname, f'{band}_associated.ecsv')):
                    quad_counts_per_band[band] += 1

    return quad_counts_per_band


def get_n_quadrants_merged_all_fields() -> Dict[str, Dict[str, int]]:
    """Get number of quadrants in all merged fields. Returns dict of form {'field': {'band': number_of_quadrants}}
    NOTE: ~0.73 sq deg / quadrant
    """
    quad_counts_per_band_all_fields = {}
    for merged_field_fname in os.listdir(MERGED_RESULTS_DIRPATH):
        field_name = merged_field_fname[:6]
        quad_counts_per_band_all_fields[field_name] = get_n_quadrants_merged(field_name)

    return quad_counts_per_band_all_fields


def metadata_from_field_dirname(field_dirname: str) -> Dict[str, str]:
    """Get the ZTF image metadata given the field's dirname in our saved data."""
    field, ccdid, qid = field_dirname.split('_')
    return {'fieldid': field, 'ccdid': ccdid, 'qid': qid}


def rcid_to_ccdid_qid(rcid: int) -> tuple:
    """Convert a ZTF readout-channel ID to (ccdid, qid).

    ZTF has 16 CCDs each with 4 quadrants, giving 64 readout channels.
    The mapping is: rcid = (ccdid - 1) * 4 + (qid - 1).
    """
    return rcid // 4 + 1, rcid % 4 + 1


# Columns that are logically integers but may arrive as nullable floats (None/NaN fill).
# These must be cast to masked int64 before writing to avoid float rounding of large IDs.
_INT64_COLUMNS = frozenset({'PSTARR_PanSTARR_ID', 'PanSTARR_ID', 'qualityFlag', 'primaryDetection'})


def prepare_table_for_write(table: Table) -> Table:
    """Return a copy of *table* with columns cast for safe serialization.

    Two fixes are applied:
    - Byte-string columns (dtype kind 'S' or object-of-bytes) are converted to str
      so ECSV writers don't choke on them.
    - Columns in ``_INT64_COLUMNS`` that arrived as nullable floats (None / NaN fill)
      are cast to masked int64, preserving integer precision for large IDs like
      PanSTARRS objIDs that would otherwise be rounded.
    """
    table = table.copy()
    for colname in table.colnames:
        col = table[colname]
        # Bytes or any remaining object dtype -> str (HDF5 has no object equivalent)
        if col.dtype.kind == 'S' or col.dtype.kind == 'O':
            table[colname] = col.astype(str)
        # Nullable float -> masked int64 for known integer columns
        elif colname in _INT64_COLUMNS:
            data = np.array(col)
            mask = np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in data])
            values = np.array(
                [v if not (v is None or (isinstance(v, float) and np.isnan(v))) else -1 for v in data],
                dtype=np.int64,
            )
            table[colname] = MaskedColumn(values, mask=mask)
    return table


def load_ecsv(fpath: str, careful_load: bool = True) -> Table:
    """Load a ecsv file as an astropy table. If you want speed and are okay with roundoff, careful_load can be False."""
    # If the file is a hdf5 file, load it directly
    if fpath.endswith('.hdf5'):
        return Table.read(fpath, path='data')

    # Try getting the hdf5 version if it exists
    if fpath.endswith('.ecsv'):
        hdf5_version_fpath = f'{fpath[:-5]}.hdf5'
        if os.path.exists(hdf5_version_fpath):
            print(f'hdf5 version of {fpath.split("/")[-1]} found, loading instead of ecsv...')
            return Table.read(hdf5_version_fpath, path='data')

    if careful_load:
        return Table.read(fpath, format='ascii.ecsv')
    return Table.from_pandas(pd.read_csv(fpath, comment='#', delimiter=' '))


# Set up casjobs object
wsid, password = get_credentials(MAST_CREDENTIAL_FNAME)
MASTCASJOBS = MastCasJobs(context="PanSTARRS_DR2", userid=wsid, password=password, request_type='POST')

# Per-process CasJobs lock (set by init_worker_lock in parallel contexts).
# Only one process should talk to the CasJobs service at a time because the
# shared account cannot have multiple open DataReaders simultaneously.
_casjobs_lock = None


def init_worker_lock(lock) -> None:
    """Initializer for ProcessPoolExecutor workers — stores the shared lock."""
    global _casjobs_lock
    _casjobs_lock = lock


def casjobs_query_lock():
    """Return the active CasJobs lock, or a no-op context if none was set."""
    return _casjobs_lock if _casjobs_lock is not None else nullcontext()


def _add_pstarr_mag_cols(tab: Table) -> Table:
    mags, magerrs = img_flux_to_ab_mag(tab['psfFlux'], fluxerr=tab['psfFluxErr'], zero_point=8.9)  # 8.9 is used by 
    tab['mag'] = mags
    tab['magerr'] = magerrs

    return tab


def _drop_bad_pstarr_flags(tab: Table) -> Table:
    """Drop bad flags. For flag references see https://outerspace.stsci.edu/display/PANSTARRS/PS1+Detection+Flags."""
    tab = tab[tab['infoFlag'] & 2048 == 0]  # source is thought to be a defect
    tab = tab[tab['infoFlag2'] & 4 == 0]    # weird forced photometry thing

    return tab


def get_pstarr_lc_from_id(objid: int) -> Table:
    if not isinstance(objid, int) and not np.isnan(objid):
        objid = int(objid)

    # Construct the query
    query = f"""
SELECT d.objID, d.ra, d.dec, d.obsTime, d.filterID, d.psfFlux, d.psfFluxErr, d.infoFlag, d.infoFlag2, d.zp FROM Detection d WHERE d.objID = {objid}
"""

    # Get table and add mag cols
    tab = MASTCASJOBS.quick(query, task_name=f"PanSTARRS_lc")
    tab = _drop_bad_pstarr_flags(tab)
    tab = _add_pstarr_mag_cols(tab)

    return tab


def get_pstarr_lc_from_coord(ra: float, dec: float, rad_arcsec: float = 1.0) -> Table:
    # Construct the query
    query = f"""
SELECT d.objID, d.ra, d.dec, d.obsTime, d.filterID, d.psfFlux, d.psfFluxErr, d.infoFlag, d.infoFlag2, d.zp
FROM fGetNearbyObjEq({ra}, {dec}, {rad_arcsec}) nb
INNER JOIN Detection d on d.objID = nb.objID
"""

    # Get table and add mag cols
    tab = MASTCASJOBS.quick(query, task_name=f"PanSTARRS_lc")
    tab = _drop_bad_pstarr_flags(tab)
    tab = _add_pstarr_mag_cols(tab)

    return tab
