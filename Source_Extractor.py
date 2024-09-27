import os
import sep
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mastcasjobs import MastCasJobs
from astroquery.mast import Catalogs
from photutils.psf import PSFPhotometry
from scipy.interpolate import NearestNDInterpolator

from utils import get_credentials
from typing import Union, Tuple, Optional
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table, unique


def img_flux_to_ab_mag(flux: np.ndarray, zero_point: float, fluxerr: Optional[np.ndarray] = None) -> float:
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
    mag = -2.5 * np.log10(flux) + zero_point
    if fluxerr is not None:
        magerr = 2.5 * fluxerr / (np.log(10) * flux)
        return mag, magerr
    return mag


class Source_Extractor():
    def __init__(self, fits_fpath: str, band: Optional[str] = None):
        hdul = fits.open(fits_fpath)
        self.image_data = hdul[0].data.byteswap().newbyteorder()
        self.header = hdul[0].header
        self.wcs = WCS(self.header)

        # Interpolate NaNs
        # mask = np.where(~np.isnan(self.image_data))
        # interp = NearestNDInterpolator(np.transpose(mask), self.image_data[mask])
        # self.image_data = interp(*np.indices(self.image_data.shape))
        # self.image_data[np.isnan(self.image_data)] = np.nanmax(self.image_data)
        notnan_inds = np.where(~np.isnan(self.image_data))
        interp = NearestNDInterpolator(np.transpose(notnan_inds), self.image_data[notnan_inds])
        self.image_data = interp(*np.indices(self.image_data.shape))

        # Hyperparameters (many of which were hand-tuned)
        self.gain = 5.8 * self.header['NFRAMES']  # gain from ZTF paper https://iopscience.iop.org/article/10.1088/1538-3873/aaecbe/pdf
        try:
            self.r_fwhm = self.header['MEDFWHM']
        except KeyError:
            self.r_fwhm = self.header['SEEING']
        self.zero_pt_mag = self.header['MAGZP']
        self.deblend_cont = 0.00075
        self.minarea = 5
        self.deblend_nthresh = 32
        self.thresh = 1.0
        self.band = '' if band is None else band  # the band that the image is in

        # Properties defined later
        self._ztf_sources = None
        self._image_sub = None
        self._bkg = None

    @property
    def bkg(self) -> sep.Background:
        if self._bkg is None:
            self._bkg = sep.Background(self.image_data, bw=64, bh=64, fw=3, fh=3)
        return self._bkg

    @property
    def ztf_sources(self) -> np.ndarray:
        if self._ztf_sources is None:
            self.get_sources()
        return self._ztf_sources

    @property
    def image_sub(self) -> np.ndarray:
        if self._image_sub is None:
            self._image_sub = self.image_data - self.bkg.back()
        return self._image_sub

    def get_sources(self, get_segmap: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Extract sources from the image using Source Extractor (SEP).

        Parameters:
            get_segmap (bool, optional): If True, return the segmentation map along with the sources. Default is False.

        Returns:
            1. An array of detected sources.
            2. If `get_segmap` is True, the segmentation map.
        """

        # Extract self.ztf_sources
        res = sep.extract(
            self.image_sub,
            thresh=self.thresh,
            err=self.bkg.globalrms,
            deblend_cont=self.deblend_cont,
            minarea=self.minarea,
            segmentation_map=get_segmap,
            gain=self.gain,
            deblend_nthresh=self.deblend_nthresh,
        )

        if isinstance(res, tuple):
            self._ztf_sources = res[0]
        else:
            self._ztf_sources = res

        return res

    def get_kron_mags(self) -> np.ndarray:
        """
        Perform photometry on detected self.ztf_sources in the image and update the self.ztf_sources array with photometric results.
        This method calculates the Kron flux and circular flux for detected self.ztf_sources in the image, converts the flux to 
        magnitudes, and updates the self.ztf_sources array to include the new photometric fields.

        Returns:
            np.ndarray: Updated self.ztf_sources array with additional fields for Kron magnitude, Kron flux, and Kron flux error.
        """
        # Get Kron radius
        self.ztf_sources['theta'][self.ztf_sources['theta'] > np.pi / 2] -= np.pi
        self.ztf_sources['theta'][self.ztf_sources['theta'] < -1 * np.pi / 2] += np.pi / 2
        kronrad, _ = sep.kron_radius(
            self.image_sub,
            self.ztf_sources['x'],
            self.ztf_sources['y'],
            self.ztf_sources['a'],
            self.ztf_sources['b'],
            self.ztf_sources['theta'],
            6.0,
        )

        # Kron flux for sources with a radius smaller than 1.0 are circular
        r_min = 0.5  # minimum diameter = 1
        use_circle = kronrad * np.sqrt(self.ztf_sources['a'] * self.ztf_sources['b']) < r_min
        flux, fluxerr, _ = sep.sum_ellipse(self.image_sub,
                                            self.ztf_sources['x'][~use_circle],
                                            self.ztf_sources['y'][~use_circle],
                                            self.ztf_sources['a'][~use_circle],
                                            self.ztf_sources['b'][~use_circle],
                                            self.ztf_sources['theta'][~use_circle],
                                            2.5*kronrad[~use_circle],
                                            err=self.bkg.globalrms,
                                            gain=self.gain,
        )
        cflux, cfluxerr, _ = sep.sum_circle(self.image_sub,
                                                self.ztf_sources['x'][use_circle],
                                                self.ztf_sources['y'][use_circle],
                                                2.5*self.r_fwhm,
                                                subpix=1,
                                                err=self.bkg.globalrms,
                                                gain=self.gain,
        )
        flux = np.hstack((flux, cflux))
        fluxerr = np.hstack((fluxerr, cfluxerr))

        # Add mag and magerrs
        mag, magerr = img_flux_to_ab_mag(flux, self.zero_pt_mag, fluxerr=fluxerr)

        return mag, magerr

    def get_psf_mags(self) -> np.ndarray:
        """
        Perform PSF photometry on detected self.ztf_sources in the image and update the self.ztf_sources array with photometric results.
        This method calculates the PSF flux for detected self.ztf_sources in the image, converts the flux to magnitudes, and updates 
        the self.ztf_sources array to include the new photometric fields.

        Returns:
            np.ndarray: Updated self.ztf_sources array with additional fields for PSF magnitude and PSF flux error.
        """

        psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
        fit_shape = (5, 5)
        finder = DAOStarFinder(6.0, 2.0)
        psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                                aperture_radius=4)

        # Initialize arrays to store PSF flux and flux error
        psf_flux = np.zeros(len(self.ztf_sources))
        psf_fluxerr = np.zeros(len(self.ztf_sources))

        # Perform PSF photometry on each source
        for i, source in enumerate(self.ztf_sources):
            x, y = source['x'], source['y']
            psf_flux[i], psf_fluxerr[i] = sep.sum_circle(self.image_sub, x, y, self.r_fwhm, err=self.bkg.globalrms, gain=self.gain)

        # Convert flux to magnitudes
        psf_mag, psf_magerr = img_flux_to_ab_mag(psf_flux, self.zero_pt_mag, fluxerr=psf_fluxerr)

        return psf_mag, psf_magerr

    def get_sources_ra_dec(self) -> np.ndarray:
        coords = self.wcs.pixel_to_world(self.ztf_sources['x'], self.ztf_sources['y'])
        return np.vstack((coords.ra.deg, coords.dec.deg)).T

    def store_coords(self, fpath: str, coord_system: str = 'ra_dec'):
        """Store the coordinates of the detected self.ztf_sources in a text file."""
        # Get the coordinates of the self.ztf_sources
        if coord_system == 'ra_dec':
            coords = self.get_sources_ra_dec()
        elif coord_system == 'pixel':
            coords = np.vstack((self.ztf_sources['x'], self.ztf_sources['y'])).T
            coords += 1  # 0-indexed to 1-indexed
        else:
            raise ValueError("Invalid coordinate system. Must be 'ra_dec' or 'pixel'.")

        # Store the coordinates in a text file
        with open(fpath, 'w') as f:
            for s in coords:
                f.write(f"{s[0]} {s[1]}\n")

    def plot_segmap(self, show_sources: bool = True, fpath: Optional[str] = None, cmap: str = None):
        """Plot the segmentation map of the image."""
        fig, ax = plt.subplots(figsize=(15, 15))
        segmap = self.get_sources(get_segmap=True)[1]
        ax.imshow(segmap, cmap=cmap)
        if show_sources:
            for j in range(len(self.ztf_sources)):
                e = Ellipse(xy=(self.ztf_sources['x'][j], self.ztf_sources['y'][j]),
                            width=6*self.ztf_sources['a'][j],
                            height=6*self.ztf_sources['b'][j],
                            angle=self.ztf_sources['theta'][j] * 180. / np.pi)
                e.set_facecolor('none')
                e.set_edgecolor('red')
                ax.add_artist(e)
                ax.scatter(self.ztf_sources['x'][j], self.ztf_sources['y'][j], color='k')
        if fpath is not None:
            plt.savefig(fpath)

    def get_coord_range(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get the RA and DEC range of the image."""
        ra_range = (self.wcs.pixel_to_world(0, 0).ra.deg, self.wcs.pixel_to_world(self.image_data.shape[1], 0).ra.deg)
        dec_range = (self.wcs.pixel_to_world(0, self.image_data.shape[0]).dec.deg, self.wcs.pixel_to_world(0, 0).dec.deg)
        return ra_range, dec_range

    def get_data_table(self, include_kron: bool = True, include_psf: bool = True) -> Table:
        """
        Get a table of the data extracted from the image.

        Parameters:
            include_kron (bool, optional): If True, include Kron magnitudes in the table. Default is True.
            include_psf (bool, optional): If True, include PSF magnitudes in the table. Default is True.

        Returns:
            astropy.Table: A table of the data extracted from the image.
        """
        # Get the sources and coordinates
        self.get_sources()
        coords = self.get_sources_ra_dec()

        # Get the photometry
        kron_mags, kron_magerrs = self.get_kron_mags()

        # Create the table
        data_table = Table(self.ztf_sources)
        if include_kron:
            data_table[f'{self.band}KronMag'] = kron_mags
            data_table[f'{self.band}KronMagErr'] = kron_magerrs
        if include_psf:
            ... # TODO: implement psf photometry stuff
        data_table['ra'] = coords[:, 0]
        data_table['dec'] = coords[:, 1]

        return data_table

    # def match_panstarrs(self, verbose: bool = False) -> SkyCoord:
    #     """
    #     Matches sources to the Pan-STARRS catalog.
    #     This method checks if each source in the `self.ztf_sources` list is present in the Pan-STARRS catalog.
    #     It uses the World Coordinate System (WCS) to convert pixel coordinates to sky coordinates and then
    #     queries the Pan-STARRS catalog for each source.

    #     Parameters:
    #         verbose (bool, optional): If True, prints progress messages every 10 sources checked, else every  100.

    #     Returns:
    #         1. A boolean array indicating whether each source is found in the Pan-STARRS catalog.
    #     """
    #     in_panstarrs = np.zeros(len(self.ztf_sources), dtype=bool)
    #     log_interval = 10 if verbose else 100

    #     # Check if each source is in PanSTARRS
    #     source_coords = self.wcs.pixel_to_world(self.ztf_sources['x'], self.ztf_sources['y'])
    #     for i, coord in enumerate(source_coords):
    #         ps1_res = query_cone_ps1(coord.ra.deg, coord.dec.deg, 0.5)
    #         in_panstarrs[i] = ps1_res is not None
    #         if i % log_interval == 0:
    #             print(f'Checked for {i} / {len(self.ztf_sources)} sources...')

    #     return in_panstarrs

    # def get_pstar_catalog_difference(self, verbose: bool = False) -> SkyCoord:
    #     """
    #     Matches sources to the Pan-STARRS catalog.
    #     This method checks if each source in the `self.ztf_sources` list is present in the Pan-STARRS catalog.
    #     It uses the World Coordinate System (WCS) to convert pixel coordinates to sky coordinates and then
    #     queries the Pan-STARRS catalog for each source.

    #     Parameters:
    #         verbose (bool, optional): If True, prints progress messages every 10 sources checked, else every  100.

    #     Returns:
    #         1. A boolean array indicating whether each source is found in the Pan-STARRS catalog.
    #     """
    #     # Get the panstarrs coordiantes
    #     pstar_coords = SkyCoord(ra=self.pstar_sources['raMean'], dec=self.pstar_sources['decMean'], unit='deg')

    #     # Get ZTF coordinates
    #     ztf_coords = self.get_sources_ra_dec()
    #     ztf_coords = SkyCoord(ra=ztf_coords[:, 0], dec=ztf_coords[:, 1], unit='deg')
    #     print(f'ZTF sources: {ztf_coords.shape}, PanSTARRS sources: {pstar_coords.shape}')

    #     # Cross-match panstarrs and my ZTF sources
    #     idx, d2d, _ = ztf_coords.match_to_catalog_sky(pstar_coords)
    #     sep_constraint = d2d.arcsecond > 1.0
    #     ztf_unmatched = ztf_coords[sep_constraint]
    #     pstar_unmatched = pstar_coords[idx[sep_constraint]]

    #     return ztf_unmatched, pstar_unmatched, sep_constraint


def make_nan(catalog, replace = np.nan):
    '''
    Go through an astropy table and covert any empty values
    into a single aunified value specified by 'replace'
    '''
    for i in range(len(catalog)):
        for j in catalog[i].colnames:
            if str(catalog[i][j]) in [False, 'False', '', '-999', '-999.0', '--', 'n', '-9999.0', 'nan', b'']:
                catalog[i][j] = replace

    return catalog


def query_cone_ps1(ra_deg: Union[float, np.ndarray], dec_deg: Union[float, np.ndarray], search_radius_arcmin: float) -> Table:
    '''
    Adapted from FLEET.
    '''

    # Get the PS1 MAST username and password from /Users/username/3PI_key.txt
    key_location = os.path.join(pathlib.Path.home(), 'vault/mast_login.txt')
    wsid, password = np.genfromtxt(key_location, dtype = 'str')

    # 3PI query
    # Kron Magnitude, ps_score (we define galaxies as ps_score < 0.9)
    the_query = """
    SELECT o.objID,o.raStack,o.decStack,m.primaryDetection,psc.ps_score
    FROM fGetNearbyObjEq(%s, %s, %s) nb
    INNER JOIN ObjectThin o on o.objid=nb.objid
    INNER JOIN StackObjectThin m on o.objid=m.objid
    LEFT JOIN HLSP_PS1_PSC.pointsource_scores psc on o.objid=psc.objid
    FULL JOIN StackModelFitSer s on o.objid=s.objid
    INNER JOIN StackObjectAttributes b on o.objid=b.objid WHERE m.primaryDetection = 1
    """
    la_query = the_query%(ra_deg, dec_deg, search_radius_arcmin)

    # Format Query
    jobs    = MastCasJobs(userid=wsid, password=password, context="PanSTARRS_DR2")
    results = jobs.quick(la_query, task_name="python cone search")

    # For New format
    if type(results) != str:
        catalog_3pi = Table(results, dtype=[str] * len(results.columns))
        if len(catalog_3pi) == 0:
            print('Found %s objects'%len(catalog_3pi))
            return None
    else:
        raise TypeError(f'Query returned type {type(results)}, not astropy.Table.')

    # Clean up 3pi's empty cells
    catalog_3pi = make_nan(catalog_3pi)

    # Append '3pi' to column name
    for i in range(len(catalog_3pi.colnames)):
        catalog_3pi[catalog_3pi.colnames[i]].name = catalog_3pi.colnames[i] + '_3pi'

    # Remove duplicates
    catalog_3pi = unique(catalog_3pi, keys = 'objID_3pi', keep = 'first')

    return catalog_3pi
