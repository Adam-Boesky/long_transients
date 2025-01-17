import os
import sys
import numpy as np
import astropy.units as u

from typing import List
from ztfquery import lightcurve
from astropy.table import Table, vstack
from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord


class Light_Curve:
    def __init__(
            self,
            ra: float,
            dec: float,
            catalogs: List[str] = ['ztf', 'wise', 'ptf'],
            query_rad_arcsec: float = 1.5,
        ):
        self.ra = ra
        self.dec = dec
        self.skycoord = SkyCoord(ra, dec, unit="deg")
        self.catalogs = catalogs
        self.query_rad_arcsec = query_rad_arcsec  # query radius in arcseconds

        # Map class catalog names to astroquery catalog names
        self.catalog_astroquery_map = {
            'ztf': 'ztf_objects_dr22',
            'wise': 'allwise_p3as_mep',
            'ptf': 'ptf_lightcurves'
        }

        self._lc = None

    @property
    def lc(self) -> Table:
        if self._lc is None:
            self._lc = self.get_lc()
        return self._lc

    def get_catalog_lc(self, catalog: str) -> Table:
        """Query the IRSA service for the specified catalog and return the light curve data."""

        # Construct the lightcurve
        if catalog == 'ztf':
            # Desired column names
            desired_colnames = ['oid', 'ra', 'dec', 'mjd', 'filtercode', 'mag', 'magerr']

            # Rename columns
            column_rename_map = [('oid', 'ztf_id')]

        elif catalog == 'wise':
            # Desired column names
            mag_colnams = [f'w{band_num}mpro_ep' for band_num in range(1, 5)]
            magerr_colnams = [f'w{band_num}sigmpro_ep' for band_num in range(1, 5)]
            desired_colnames = ['source_id_mf', 'ra', 'dec', 'mjd'] + [item for sublist in list(zip(mag_colnams, magerr_colnams)) for item in sublist]

            # Rename columns
            column_rename_map = [('source_id_mf', 'wise_id')] + [(f'w{band_num}mpro_ep', f'w{band_num}_mag') for band_num in range(1, 5)] + \
                [(f'w{band_num}sigmpro_ep', f'w{band_num}_magerr') for band_num in range(1, 5)]

        elif catalog == 'ptf':
            # Desired column names
            desired_colnames = ['oid', 'ra', 'dec', 'obsmjd', 'fid', 'mag_autocorr', 'magerr_auto'] 

            # Rename columns
            column_rename_map = [('oid', 'ptf_id'), ('obsmjd', 'mjd')]

        # Query the catalog
        print(f"Querying {catalog} catalog...")
        if catalog == 'ztf':
            lightcurve_tab = Table.from_pandas(
                lightcurve.LCQuery.from_position(self.ra, self.dec, self.query_rad_arcsec).data
            )
            lightcurve_tab = lightcurve_tab[desired_colnames]
        else:
            lightcurve_tab: Table = Irsa.query_region(
                coordinates=self.skycoord,
                catalog=self.catalog_astroquery_map[catalog],
                spatial='Cone',
                radius=self.query_rad_arcsec * u.arcsec,
                columns=','.join(desired_colnames),
            )

        # Adjustment for PTF catalog
        if catalog == 'ptf':
            # fid Filter identifier (1 = g; 2 = R).
            # g band
            g_mask = lightcurve_tab['fid'] == 1
            lightcurve_tab['g_mag'] = np.ones(len(lightcurve_tab)) * np.nan
            lightcurve_tab['g_magerr'] = np.ones(len(lightcurve_tab)) * np.nan
            lightcurve_tab['g_mag'][g_mask] = lightcurve_tab['mag_autocorr'][g_mask]
            lightcurve_tab['g_magerr'][g_mask] = lightcurve_tab['magerr_auto'][g_mask]

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

        # Rename columns
        for (cname, new_cname) in column_rename_map:
            lightcurve_tab.rename_column(cname, new_cname)
        
        # Mask NaN values
        for col in lightcurve_tab.colnames:
            if lightcurve_tab[col].dtype.kind in 'f':  # Check if the column is of float type
                lightcurve_tab[col] = np.ma.masked_invalid(lightcurve_tab[col])

        return lightcurve_tab

    def get_lc(self) -> Table:
        """Get the light curve."""
        # Query the lightcurves for each catalog
        lcs: List[Table] = []
        for catalog_name in self.catalogs:

            # Grab the data
            cat_lc = self.get_catalog_lc(catalog_name)

            # Data only for the closest object
            cat_lc = cat_lc[cat_lc[f'{catalog_name}_id'] == cat_lc[f'{catalog_name}_id'][0]]
            lcs.append(cat_lc)

        # Join the tables
        tab = lcs[0]
        for lc in lcs[1:]:
            tab = vstack([tab, lc])

        # Drop unecessary colnames
        for colname in tab.colnames:
            if not colname.endswith('_id') and '_mag' not in colname and '_magerr' not in colname and colname \
                not in ['mjd', 'ra', 'dec']:
                tab.remove_column(colname)

        # Reorder columns for convenience
        ordered_cols = tab.colnames
        for colname in ordered_cols:
            if colname.endswith('_id'):
                ordered_cols.remove(colname)
                ordered_cols.insert(0, colname)
        tab = tab[ordered_cols]

        return tab
