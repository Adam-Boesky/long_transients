import os
import sys
import traceback
import numpy as np
import pandas as pd
import astropy.units as u

from typing import List, Iterable, Tuple, Optional
from ztfquery import lightcurve
from astropy.time import Time
from astropy.table import Table, vstack
from astroquery.sdss import SDSS
from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
from concurrent.futures import ThreadPoolExecutor

from Extracting.utils import get_pstarr_lc_from_coord, get_pstarr_lc_from_id, img_flux_to_ab_mag, get_data_path

ALL_LC_COLNAMES = ['ptf_id', 'wise_id', 'ztf_id', 'ra', 'dec', 'mjd', 'g_mag', 'g_magerr', 'r_mag', 'r_magerr', 'i_mag',
                   'i_magerr', 'w1_mag', 'w1_magerr', 'w2_mag', 'w2_magerr', 'w3_mag', 'w3_magerr', 'w4_mag',
                   'w4_magerr', 'R_mag', 'R_magerr']
LC_MARKER_INFO = {
    'ptf': '.',
    'ztf': '<',
    'sdss': 'd',
    'panstarrs': '3',
    'gaia': 'X',
    'custom': '*',
}
LC_COLOR_INFO = {
    'u': 'black',
    'g': 'forestgreen',
    'r': 'firebrick',
    'i': 'darkorchid',
    'z': 'lightcoral',
    'y': 'gold',
    'b': 'steelblue',
}
ALL_BAND_DF = pd.DataFrame(
    data={
        'g_mag': ['g', 'ztf'],
        'r_mag': ['r', 'ztf'],
        'i_mag': ['i', 'ztf'],
        'G_mag': ['g', 'ptf'],
        'R_mag': ['r', 'ptf'],
        'uModel_mag': ['u', 'sdss'],
        'gModel_mag': ['g', 'sdss'],
        'rModel_mag': ['r', 'sdss'],
        'iModel_mag': ['i', 'sdss'],
        'zModel_mag': ['z', 'sdss'],
        'yModel_mag': ['y', 'sdss'],
        'pstarr_g_mag': ['g', 'panstarrs'],
        'pstarr_r_mag': ['r', 'panstarrs'],
        'pstarr_i_mag': ['i', 'panstarrs'],
        'pstarr_z_mag': ['z', 'panstarrs'],
        'pstarr_y_mag': ['y', 'panstarrs'],
        'gaia_g_mag': ['g', 'gaia'],
        'gaia_rp_mag': ['r', 'gaia'],
        'gaia_bp_mag': ['b', 'gaia'],
        'custom_g_mag': ['g', 'custom'],
        'custom_r_mag': ['r', 'custom'],
        'custom_i_mag': ['i', 'custom'],
        'custom_z_mag': ['z', 'custom'],
    },
    index=['band', 'survey'],
)
GAIA_ZERO_PTS = {'g': 25.8010, 'bp': 25.3540, 'rp': 25.1040}  # from Table 5.4 in https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photProc/cu5pho_ssec_photCal.html#SSS3.P2
CUSTOM_PHOT_DIR = os.path.join(get_data_path(), 'followup', 'photometry')


class Light_Curve:
    def __init__(
            self,
            ra: float,
            dec: float,
            catalogs: List[str] = ['ztf', 'wise', 'neowise', 'ptf', 'sdss', 'panstarrs', 'gaia', 'custom'],
            query_rad_arcsec: float = 1.5,
            query_in_parallel: bool = True,
            pstarr_objid: Optional[int] = None,
            pstarr_coord: Optional[Tuple[float]] = None,
        ):
        self.ra = ra
        self.dec = dec
        self.skycoord = SkyCoord(ra, dec, unit="deg")
        self.catalogs = catalogs
        self.query_rad_arcsec = query_rad_arcsec  # query radius in arcseconds
        self.query_in_parallel = query_in_parallel  # whether we should query the apis in parallel or not
        self.pstarr_objid = pstarr_objid
        self.pstarr_coord = pstarr_coord

        # Map class catalog names to astroquery catalog names
        self.catalog_astroquery_map = {
            'ztf': 'ztf_objects_dr23',
            'wise': 'allwise_p3as_mep',
            'neowise': 'neowiser_p1bs_psd',
            'ptf': 'ptf_lightcurves',
            'gaia': 'gaia_dr3_source',
        }

        self._lc = None

    @property
    def lc(self) -> Table:
        if self._lc is None:
            self._lc = self.get_lc()
        return self._lc

    def get_catalog_lc(self, catalog: str) -> Optional[Table]:
        """Query the IRSA service for the specified catalog and return the light curve data."""
        # Construct the lightcurve
        if catalog == 'ztf':
            # Desired column names
            desired_colnames = ['oid', 'ra', 'dec', 'mjd', 'filtercode', 'mag', 'magerr']

            # Rename columns
            column_rename_map = [('oid', 'ztf_id')]

        elif catalog == 'wise':
            # Desired column names
            mag_colnames = [f'w{band_num}mpro_ep' for band_num in range(1, 5)]
            magerr_colnames = [f'w{band_num}sigmpro_ep' for band_num in range(1, 5)]
            f_colnames = [f'w{band_num}flux_ep' for band_num in range(1, 5)]
            ferr_colnames = [f'w{band_num}sigflux_ep' for band_num in range(1, 5)]
            desired_colnames = ['source_id_mf', 'ra', 'dec', 'mjd'] + [
                item for sublist in list(zip(
                    mag_colnames, magerr_colnames, f_colnames, ferr_colnames,
                )) for item in sublist
            ]

            # Rename columns
            column_rename_map = [('source_id_mf', 'wise_id')] + [(f'w{band_num}mpro_ep', f'w{band_num}_mag') for band_num in range(1, 5)] + \
                [(f'w{band_num}sigmpro_ep', f'w{band_num}_magerr') for band_num in range(1, 5)]

        elif catalog == 'neowise':
            # Desired column names
            mag_colnames = [f'w{band_num}mpro' for band_num in range(1, 3)]
            magerr_colnames = [f'w{band_num}sigmpro' for band_num in range(1, 3)]
            snr_colnames = [f'w{band_num}snr' for band_num in range(1, 3)]
            desired_colnames = ['source_id', 'ra', 'dec', 'mjd'] + [
                item for sublist in list(zip(
                    mag_colnames, magerr_colnames, snr_colnames
                )) for item in sublist
            ]

            # Rename columns
            column_rename_map = [('source_id', 'neowise_id')] + [(f'w{band_num}mpro', f'w{band_num}_mag') for band_num in range(1, 3)] + \
                [(f'w{band_num}sigmpro', f'w{band_num}_magerr') for band_num in range(1, 3)] + \
                [(f'w{band_num}snr', f'w{band_num}_snr') for band_num in range(1, 3)]

        elif catalog == 'ptf':
            # Desired column names
            desired_colnames = ['oid', 'ra', 'dec', 'obsmjd', 'fid', 'mag_autocorr', 'magerr_auto'] 

            # Rename columns
            column_rename_map = [('oid', 'ptf_id'), ('obsmjd', 'mjd')]

        elif catalog == 'sdss':
            # Desired column names
            sdss_bands = ['u', 'g', 'r', 'i', 'z']
            desired_colnames = ['objID', 'ra', 'dec', 'mjd'] + [item for sublist in zip(
                [f'psfMag_{band}' for band in sdss_bands],
                [f'psfMagerr_{band}' for band in sdss_bands],
                [f'modelMag_{band}' for band in sdss_bands],
                [f'modelMagerr_{band}' for band in sdss_bands],
            ) for item in sublist]

            # Rename columns
            column_rename_map = [('objID', 'sdss_id')] + [(f'psfMag_{b}', f'{b}Psf_mag') for b in sdss_bands] + \
                [(f'psfMagerr_{b}', f'{b}Psf_magerr') for b in sdss_bands] + \
                [(f'modelMag_{b}', f'{b}Model_mag') for b in sdss_bands] + \
                [(f'modelMagerr_{b}', f'{b}Model_magerr') for b in sdss_bands]

        elif catalog == 'panstarrs':
            # Desired columns
            desired_colnames = ['objID', 'ra', 'dec', 'obsTime', 'infoFlag2'] + [
                    item for sublist in zip(
                        [f'pstarr_{band}_mag' for band in 'grizy'],
                        [f'pstarr_{band}_magerr' for band in 'grizy'],
                    ) for item in sublist
                ]

            # Rename columns
            column_rename_map = [('objID', 'panstarrs_id'), ('obsTime', 'mjd'), ('infoFlag2', 'pstarr_infoFlag2')]

        elif catalog == 'gaia':
            # Desired columns
            gaia_bands = ('g', 'rp', 'bp')
            desired_colnames = ['source_id', 'ra', 'dec', 'ref_epoch'] + [item for sublist in zip(
                [f'phot_{band}_mean_flux' for band in gaia_bands],
                [f'phot_{band}_mean_flux_error' for band in gaia_bands],
            ) for item in sublist]

            # Rename columns
            column_rename_map = [('source_id', 'gaia_id')]
        
        elif catalog == 'custom':
            # Desired columns
            desired_colnames = ['custom_id', 'mjd'] + [item for sublist in zip(
                [f'custom_{band}_mag' for band in 'griz'],
                [f'custom_{band}_magerr' for band in 'griz'],
            ) for item in sublist]

            # Rename columns
            column_rename_map = []

        # Query the catalog
        print(f"Querying {catalog} catalog for light curve...")
        max_attempts = 3
        for i_attempt in range(max_attempts):
            try:
                if catalog == 'ztf':
                    # See Section 10.3 for BAD_CATFLAG_MASK info
                    # https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_explanatory_supplement.pdf
                    lightcurve_tab = Table.from_pandas(
                        lightcurve.LCQuery.from_position(
                            self.ra,
                            self.dec,
                            self.query_rad_arcsec,
                            BAD_CATFLAGS_MASK=6141,
                        ).data
                    )
                    lightcurve_tab = lightcurve_tab[desired_colnames]
                elif catalog == 'panstarrs':

                    # Check if we have an objid
                    if self.pstarr_objid is not None and not np.isnan(self.pstarr_objid):
                        lightcurve_tab = get_pstarr_lc_from_id(self.pstarr_objid)
                    else:
                        print('Pan-STARRS objid not found. Falling back on coordinate query...')
                        
                        # Sometimes we have a more precise panstarrs coordinate
                        if self.pstarr_coord is None:
                            ra, dec = self.ra, self.dec
                        else:
                            ra, dec = self.pstarr_coord
                        lightcurve_tab = get_pstarr_lc_from_coord(ra, dec, rad_arcsec=0.1)
                elif catalog == 'sdss':
                    lightcurve_tab: Table = SDSS.query_crossid(
                        SkyCoord(self.ra, self.dec, unit='deg'),
                        photoobj_fields=desired_colnames,
                    )
                elif catalog == 'custom':
                    # Get the filename -> coordinate mapping
                    custom_phot_fnames = [fname.split('.')[0] for fname in os.listdir(CUSTOM_PHOT_DIR) if fname != 'README.md'] # dropping file extensions
                    fname_coords = [(
                        float(fname.split('_')[1].replace("p", ".").replace("n", "-")),
                        float(fname.split('_')[2].replace("p", ".").replace("n", "-"))
                    ) for fname in custom_phot_fnames]

                    # Get the lightcurve table
                    fname_skycoords = SkyCoord(
                        [c[0] for c in fname_coords],
                        [c[1] for c in fname_coords],
                        unit='deg',
                        )
                    seps = self.skycoord.separation(fname_skycoords)
                    if np.min(seps.arcsecond > self.query_rad_arcsec):
                        lightcurve_tab = None
                    else:
                        # Read the custom file format into a pandas DataFrame
                        custom_phot_fname = custom_phot_fnames[np.argmin(seps.arcsec)]
                        lightcurve_tab = pd.read_csv(
                            os.path.join(CUSTOM_PHOT_DIR, f'{custom_phot_fname}.txt'),
                            sep=r'\s+',  # delim_whitespace=True
                            comment='#',
                            names=['mjd', 'custom_mag', 'custom_magerr', 'filter', 'ul', 'telescope', 'instrument']
                        )
                        lightcurve_tab['custom_id'] = custom_phot_fname
                        lightcurve_tab = Table.from_pandas(lightcurve_tab)
                else:
                    lightcurve_tab: Table = Irsa.query_region(
                        coordinates=self.skycoord,
                        catalog=self.catalog_astroquery_map[catalog],
                        spatial='Cone',
                        radius=self.query_rad_arcsec * u.arcsec,
                        columns=','.join(desired_colnames),
                    )

                # Escape the loop if no error was raised
                break
            except:
                if i_attempt >= max_attempts - 1:
                    raise
                else:
                    print(f'Attempt {i_attempt + 1} / {max_attempts} to query for light curves failed. Trying again...')
                    traceback.print_exc()

        # Add the wise SNRs and drop the flux columns
        if catalog == 'wise':
            for band_num in range(1, 5):
                f_colname, ferr_colname = f'w{band_num}flux_ep', f'w{band_num}sigflux_ep'
                lightcurve_tab[f'w{band_num}_snr'] = lightcurve_tab[f_colname] / lightcurve_tab[ferr_colname]
                lightcurve_tab.remove_columns([f_colname, ferr_colname])

        # Adjustment for PTF catalog
        if catalog == 'ptf':
            # fid Filter identifier (1 = g; 2 = R).
            # g band
            g_mask = lightcurve_tab['fid'] == 1
            lightcurve_tab['G_mag'] = np.ones(len(lightcurve_tab)) * np.nan
            lightcurve_tab['G_magerr'] = np.ones(len(lightcurve_tab)) * np.nan
            lightcurve_tab['G_mag'][g_mask] = lightcurve_tab['mag_autocorr'][g_mask]
            lightcurve_tab['G_magerr'][g_mask] = lightcurve_tab['magerr_auto'][g_mask]

            # R band
            R_mask = lightcurve_tab['fid'] == 2
            lightcurve_tab['R_mag'] = np.ones(len(lightcurve_tab)) * np.nan
            lightcurve_tab['R_magerr'] = np.ones(len(lightcurve_tab)) * np.nan
            lightcurve_tab['R_mag'][R_mask] = lightcurve_tab['mag_autocorr'][R_mask]
            lightcurve_tab['R_magerr'][R_mask] = lightcurve_tab['magerr_auto'][R_mask]
        elif catalog == 'panstarrs':  # filterID (1=g, 2=r, 3=i, 4=z, 5=y)
            # Iterate through bands and add band magnitude columns
            for i, band in enumerate('grizy'):

                # Get a mask for the filter ID
                filter_id = i + 1
                filter_mask = lightcurve_tab['filterID'].astype(int) == filter_id

                # Fill in the table
                lightcurve_tab[f'pstarr_{band}_mag'] = np.ones(len(lightcurve_tab)) * np.nan
                lightcurve_tab[f'pstarr_{band}_magerr'] = np.ones(len(lightcurve_tab)) * np.nan
                lightcurve_tab[f'pstarr_{band}_mag'][filter_mask] = lightcurve_tab['mag'][filter_mask]
                lightcurve_tab[f'pstarr_{band}_magerr'][filter_mask] = lightcurve_tab['magerr'][filter_mask]

            # Drop unnecessary columns
            lightcurve_tab = lightcurve_tab[desired_colnames]
        elif catalog == 'ztf':
            for band in 'gri':
                band_mask = lightcurve_tab[f'filtercode'] == f'z{band}'
                lightcurve_tab[f'{band}_mag'] = np.ones(len(lightcurve_tab)) * np.nan
                lightcurve_tab[f'{band}_magerr'] = np.ones(len(lightcurve_tab)) * np.nan
                lightcurve_tab[f'{band}_mag'][band_mask] = lightcurve_tab['mag'][band_mask]
                lightcurve_tab[f'{band}_magerr'][band_mask] = lightcurve_tab['magerr'][band_mask]
        elif catalog == 'gaia':
            # Convert time column to mjd
            lightcurve_tab['mjd'] = Time(lightcurve_tab['ref_epoch'], format='jyear', scale='tcb').mjd
            lightcurve_tab.remove_column('ref_epoch')

            # Add magnitude columns
            flux_cols = [f'phot_{band}_mean_flux' for band in gaia_bands]
            mag_cols = [f'gaia_{band}_mag' for band in gaia_bands]
            for band, flux_colname, mag_colname in zip(gaia_bands, flux_cols, mag_cols):
                mag, magerr = img_flux_to_ab_mag(
                    lightcurve_tab[flux_colname],
                    fluxerr=lightcurve_tab[f'{flux_colname}_error'],
                    zero_point=GAIA_ZERO_PTS[band],
                )  # zp from https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photProc/cu5pho_ssec_photCal.html#SSS3.P2
                lightcurve_tab[mag_colname] = mag
                lightcurve_tab[f'{mag_colname}err'] = magerr

            # Drop flux and flux error columns
            lightcurve_tab.remove_columns(flux_cols + [f'{col}_error' for col in flux_cols])
        elif catalog == 'custom' and lightcurve_tab is not None:
            # Add magnitude columns by band
            for band in 'griz':
                band_mask = lightcurve_tab['filter'] == band
                lightcurve_tab[f'custom_{band}_mag'] = np.ones(len(lightcurve_tab)) * np.nan
                lightcurve_tab[f'custom_{band}_magerr'] = np.ones(len(lightcurve_tab)) * np.nan
                lightcurve_tab[f'custom_{band}_mag'][band_mask] = lightcurve_tab['custom_mag'][band_mask]
                lightcurve_tab[f'custom_{band}_magerr'][band_mask] = lightcurve_tab['custom_magerr'][band_mask]

            # Drop unnecessary columns
            lightcurve_tab = lightcurve_tab[desired_colnames]

        if lightcurve_tab is None:
            return Table()

        # Rename columns
        for (cname, new_cname) in column_rename_map:
            lightcurve_tab.rename_column(cname, new_cname)

        # Mask NaN values
        for col in lightcurve_tab.colnames:
            if lightcurve_tab[col].dtype.kind in 'f':  # check if the column is of float type
                lightcurve_tab[col] = np.ma.masked_invalid(lightcurve_tab[col])

        # Collapse two measurements from the same night. Pan-STARRS single-epoch measurements visit the same spot
        # twice every night so we can just count that as one measurement
        if catalog == 'panstarrs':

            # The mag and magerr columns
            pstarr_mag_cols = [f'pstarr_{band}_mag' for band in 'grizy']
            pstarr_magerr_cols = [f'pstarr_{band}_magerr' for band in 'grizy']

            # Iterate through the tables of different panstarrs magnitudes
            collapsed_tab = Table(names=lightcurve_tab.columns, dtype=lightcurve_tab.dtype)
            for mag_col, magerr_col in zip(pstarr_mag_cols, pstarr_magerr_cols):
                mag_tab = lightcurve_tab[~lightcurve_tab.mask[mag_col]]

                # Collapse
                while len(mag_tab) > 0:
                    row = mag_tab[0]
                    dt = np.abs(row['mjd'] - mag_tab['mjd'])
                    same_night_mask = (0 < dt) & (dt <= 1)

                    # If there's another observation that night, combine into one row and drop the second one
                    if np.sum(same_night_mask) > 0:

                        # Take the mean of the mag columns and propagate their error
                        row['mjd'] = np.nanmean(np.hstack((row['mjd'], mag_tab[same_night_mask]['mjd'])))
                        row[mag_col] = np.nanmean(np.hstack((row[mag_col], mag_tab[same_night_mask][mag_col])))
                        row[magerr_col] = 1 / (np.sum(same_night_mask) + 1) * np.sqrt(
                            np.nansum(
                                np.hstack((row[magerr_col]**2, mag_tab[same_night_mask][magerr_col]**2))
                            )
                        )  # 1/n * sqrt(sum(sigma_i^2))

                        # Drop the other observation
                        mag_tab = mag_tab[~same_night_mask]

                    # Stack and drop the row
                    collapsed_tab = vstack((collapsed_tab, row))
                    mag_tab.remove_row(0)
            lightcurve_tab = collapsed_tab

        # Ensure mjd column is float dtype
        lightcurve_tab['mjd'] = lightcurve_tab['mjd'].astype(float)

        # For WISE, take the mean of mags for 3-day windows
        if catalog == 'wise' and len(lightcurve_tab) > 0:
            # Sort by MJD to ensure time order
            lightcurve_tab.sort('mjd')
            new_tab = Table(names=lightcurve_tab.colnames, dtype=lightcurve_tab.dtype)
            mjds = np.array(lightcurve_tab['mjd'])
            used = np.zeros(len(lightcurve_tab), dtype=bool)
            i = 0
            while i < len(lightcurve_tab):
                # Start window at current unprocessed row
                window_start = mjds[i]
                in_window_mask = (mjds >= window_start) & (mjds < window_start + 3) & (~used)
                if not np.any(in_window_mask):
                    i += 1
                    continue
                # Compute the mean for each column manually and create a new row
                mean_row = []
                for col in lightcurve_tab.colnames:
                    col_data = np.array(lightcurve_tab[col][in_window_mask], copy=True)
                    if np.issubdtype(col_data.dtype, np.number):
                        mean_val = np.nanmean(col_data)
                    else:
                        mean_val = col_data[0] if len(col_data) > 0 else None
                    mean_row.append(mean_val)
                new_tab.add_row(mean_row)
                used[in_window_mask] = True
                # Move to next unused row
                next_indices = np.where(~used)[0]
                if len(next_indices) == 0:
                    break
                i = next_indices[0]
            lightcurve_tab = new_tab

        return lightcurve_tab

    def get_lc(self) -> Table:
        """Get the light curve."""
        # Query the lightcurves for each catalog in parallel
        lcs: List[Table] = []
        def get_lc_from_cat(catalog_name):
            # Grab the data
            cat_lc = self.get_catalog_lc(catalog_name)

            # Data only for the closest object
            if len(cat_lc) > 0:

                # ZTF uses different oids for different bands, so we must associate between bands
                # we will use an angular separation of <0.5arcseconds as our between-band requirement
                if catalog_name == 'ztf':
                    src_coord = SkyCoord(cat_lc[0]['ra'], cat_lc[0]['dec'], unit='deg')
                    all_coords = SkyCoord(cat_lc['ra'], cat_lc['dec'], unit='deg')
                    mask = src_coord.separation(all_coords).arcsec < 0.5
                else:  # otherwise, we can just ensure that the IDs line up
                    mask = cat_lc[f'{catalog_name}_id'] == cat_lc[f'{catalog_name}_id'][0]
                cat_lc = cat_lc[mask]
                return cat_lc
            return None

        if self.query_in_parallel:
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(get_lc_from_cat, self.catalogs))
        else:
            results = [get_lc_from_cat(cat) for cat in self.catalogs]

        # Filter out None results
        lcs = [result for result in results if result is not None]

        # Join the tables
        if len(lcs) == 1:
            tab = lcs[0]
        elif len(lcs) > 1:
            tab = vstack(lcs)
        else:
            tab = Table(names=ALL_LC_COLNAMES, masked=True)

        # Drop unecessary colnames
        for colname in tab.colnames:
            if not colname.endswith('_id') and not colname.endswith('_mag') and not colname.endswith('_magerr') and \
                not colname.endswith('_snr') not in ['mjd', 'ra', 'dec']:
                tab.remove_column(colname)

        # Reorder columns for convenience
        ordered_cols = tab.colnames
        for colname in ordered_cols:
            if colname.endswith('_id'):
                ordered_cols.remove(colname)
                ordered_cols.insert(0, colname)
        tab = tab[ordered_cols]

        return tab
