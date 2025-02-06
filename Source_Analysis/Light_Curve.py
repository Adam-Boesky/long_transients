import os
import sys
import numpy as np
import pandas as pd
import astropy.units as u

from typing import List, Optional
from ztfquery import lightcurve
from astropy.table import Table, vstack
from astroquery.sdss import SDSS
from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
from concurrent.futures import ThreadPoolExecutor

ALL_LC_COLNAMES = ['ptf_id', 'wise_id', 'ztf_id', 'ra', 'dec', 'mjd', 'g_mag', 'g_magerr', 'r_mag', 'r_magerr', 'i_mag',
                   'i_magerr', 'w1_mag', 'w1_magerr', 'w2_mag', 'w2_magerr', 'w3_mag', 'w3_magerr', 'w4_mag',
                   'w4_magerr', 'R_mag', 'R_magerr']
LC_MARKER_INFO = {
    'ptf': '.',
    'ztf': 'v',
    'sdss': 'd',
}
LC_COLOR_INFO = {
    'u': 'black',
    'g': 'forestgreen',
    'r': 'firebrick',
    'i': 'indigo',
    'z': 'lightcoral',
    'y': 'gold',
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
    },
    index=['band', 'survey'],
)


class Light_Curve:
    def __init__(
            self,
            ra: float,
            dec: float,
            catalogs: List[str] = ['ztf', 'wise', 'ptf', 'sdss'],
            query_rad_arcsec: float = 1.5,
            query_in_parallel: bool = True,
        ):
        self.ra = ra
        self.dec = dec
        self.skycoord = SkyCoord(ra, dec, unit="deg")
        self.catalogs = catalogs
        self.query_rad_arcsec = query_rad_arcsec  # query radius in arcseconds
        self.query_in_parallel = query_in_parallel  # whether we should query the apis in parallel or not

        # Map class catalog names to astroquery catalog names
        self.catalog_astroquery_map = {
            'ztf': 'ztf_objects_dr23',
            'wise': 'allwise_p3as_mep',
            'ptf': 'ptf_lightcurves'
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

        # Query the catalog
        print(f"Querying {catalog} catalog for light curve...")
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
        elif catalog == 'sdss':
            lightcurve_tab: Table = SDSS.query_crossid(
                SkyCoord(self.ra, self.dec, unit='deg'),
                photoobj_fields=desired_colnames,
            )
        else:
            lightcurve_tab: Table = Irsa.query_region(
                coordinates=self.skycoord,
                catalog=self.catalog_astroquery_map[catalog],
                spatial='Cone',
                radius=self.query_rad_arcsec * u.arcsec,
                columns=','.join(desired_colnames),
            )

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
        elif catalog == 'ztf':
            for band in ('g', 'r', 'i'):
                band_mask = lightcurve_tab[f'filtercode'] == f'z{band}'
                lightcurve_tab[f'{band}_mag'] = np.ones(len(lightcurve_tab)) * np.nan
                lightcurve_tab[f'{band}_magerr'] = np.ones(len(lightcurve_tab)) * np.nan
                lightcurve_tab[f'{band}_mag'][band_mask] = lightcurve_tab['mag'][band_mask]
                lightcurve_tab[f'{band}_magerr'][band_mask] = lightcurve_tab['magerr'][band_mask]

        if lightcurve_tab is None:
            return Table()

        # Rename columns
        for (cname, new_cname) in column_rename_map:
            lightcurve_tab.rename_column(cname, new_cname)

        # Mask NaN values
        for col in lightcurve_tab.colnames:
            if lightcurve_tab[col].dtype.kind in 'f':  # check if the column is of float type
                lightcurve_tab[col] = np.ma.masked_invalid(lightcurve_tab[col])

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
