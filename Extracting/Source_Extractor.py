import os
import pathlib
from typing import Optional, Tuple, Union, Iterable

import matplotlib.pyplot as plt
import numpy as np
import sep
from numpy.lib.recfunctions import rename_fields
from matplotlib.axes._axes import Axes

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, unique
from astropy.wcs import WCS
from astropy.nddata import NDData
from astropy.visualization import simple_norm
from matplotlib.patches import Ellipse
from photutils.psf import PSFPhotometry, EPSFBuilder, EPSFFitter, EPSFStars, extract_stars
from scipy.interpolate import NearestNDInterpolator

try:
    from utils import img_ab_mag_to_flux, img_flux_to_ab_mag, get_snr_from_mag, true_nearby, MASTCASJOBS, MAST_CREDENTIAL_FNAME
except ModuleNotFoundError:
    from .utils import img_ab_mag_to_flux, img_flux_to_ab_mag, get_snr_from_mag, true_nearby, MASTCASJOBS, MAST_CREDENTIAL_FNAME


class Source_Extractor():
    def __init__(self, fits_fpath: str, band: Optional[str] = None):
        self.fits_fpath = fits_fpath

        # Make header valid
        hdul = fits.open(self.fits_fpath)
        if 'RADECSYS' in hdul[0].header:
            if 'RADESYS' not in hdul[0].header:
                hdul[0].header.rename_keyword('RADECSYS', 'RADESYS')
            else:
                del hdul[0].header['RADECSYS']

        self.image_data = hdul[0].data.byteswap().newbyteorder()
        self.header = hdul[0].header
        self.wcs = WCS(self.header)

        # Interpolate NaNs
        self.nan_mask = np.isnan(self.image_data)
        # notnan_inds = np.where(~self.nan_mask)
        # interp = NearestNDInterpolator(np.transpose(notnan_inds), self.image_data[notnan_inds])
        # self.image_data = interp(*np.indices(self.image_data.shape))

        # Hyperparameters (many of which were hand-tuned)
        self.gain = 5.8 * self.header['NFRAMES']  # gain from ZTF paper https://iopscience.iop.org/article/10.1088/1538-3873/aaecbe/pdf
        try:
            self.r_fwhm = self.header['MEDFWHM']
        except KeyError:
            self.r_fwhm = self.header['SEEING']
        self.zero_pt_mag = self.header['MAGZP']
        self.maglimit = dict(self.header).get('MAGLIM')
        self.deblend_cont = 0.00075
        self.minarea = 5
        self.deblend_nthresh = 32
        self.thresh = 1.0
        self.band = '' if band is None else band  # the band that the image is in

        # PSF fitting parameters (hand-tuned as function of FWHM)
        self.psf_cutout_size = np.ceil(self.r_fwhm * 14 // 2 * 2 + 1).astype(int)  # rounding to nearest odd integer
        self.fit_boxsize = np.ceil(self.r_fwhm * 2 // 2 * 2 + 1).astype(int)

        # Properties defined later
        self._sources = None
        self._image_sub = None
        self._bkg = None
        self._point_source_coords = None
        self._segmap = None
        self._segid = None

        # The PSF photometry tool and the PSF model
        self.psfphot = None
        self.epsf = None

    @property
    def bkg(self) -> sep.Background:
        """Background estimation of the image."""
        if self._bkg is None:
            self._bkg = sep.Background(self.image_data, bw=64, bh=64, fw=3, fh=3)
        return self._bkg

    @property
    def sources(self) -> Table:
        """Detected sources in the image."""
        if self._sources is None:
            self.get_sources()
        return self._sources

    @sources.setter
    def sources(self, new_sources):
        self._sources = new_sources

    @property
    def image_sub(self) -> np.ndarray:
        """Background-subtracted image data."""
        if self._image_sub is None:
            self._image_sub = self.image_data - self.bkg.back()
        return self._image_sub

    def _is_truncated(self, flag: Union[Iterable[int], int]) -> Iterable[bool]:
        """Return if source is truncated based on sep flags."""
        if isinstance(flag, int):
            flag = np.array([flag])
        return np.logical_or((flag & sep.APER_TRUNC) != 0, (flag & sep.OBJ_TRUNC) != 0)

    def get_sources(self) -> Union[Table, Tuple[Table, np.ndarray]]:
        """
        Extract sources from the image using Source Extractor (SEP).

        Returns:
            1. An array of detected sources.

        NOTE: Flag key is here: https://sextractor.readthedocs.io/en/latest/Flagging.html
        """

        # Extract self.sources
        print('Extracting sources...')
        self._sources, self._segmap = sep.extract(
            self.image_sub,
            thresh=self.thresh,
            err=self.bkg.globalrms,
            deblend_cont=self.deblend_cont,
            minarea=self.minarea,
            segmentation_map=True,
            gain=self.gain,
            deblend_nthresh=self.deblend_nthresh,
            mask=self.nan_mask,                     # don't detect sources in NaN regions
        )
        self._segid = np.arange(1, len(self._sources)+1, dtype=np.int32)  # arbitrary unique IDs for each source

        # Rename columns and set _sources
        self._sources = rename_fields(self._sources, {'flag': 'sepExtractionFlag'})
        self._sources = Table(self._sources)

        return self._sources, self._segmap

    def set_sources_for_psf(self, pstarr_table: Table):
        """Set the sources for the PSF photometry. Will do so by making a cut on SNR > 10, psf mag - kron mag < 0.05,
        and psf mag brighter than 10.
        """
        # Distance to other sources mask (> 10 pixels)
        xs, ys = self.ra_dec_to_pix(pstarr_table['ra'], pstarr_table['dec'])
        min_dists = []
        for x, y in zip(xs, ys):
            dists = np.sqrt((xs - x)*(xs - x) + (ys - y)*(ys - y))
            min_dists.append(np.min(dists[dists != 0]))
        min_dists = np.array(min_dists)
        pstarr_table = pstarr_table[min_dists > 10]

        # Make sure sources aren't don't have any nans in the pixels around them
        xs, ys = self.ra_dec_to_pix(pstarr_table['ra'], pstarr_table['dec'])
        half_cutout_size = int(self.psf_cutout_size / 2)
        not_nan = np.zeros(len(pstarr_table))
        for i, (x, y) in enumerate(zip(xs, ys)):
            x, y = int(x), int(y)
            cutout = self.nan_mask[
                y - half_cutout_size : y + half_cutout_size,
                x - half_cutout_size : x + half_cutout_size
            ]
            if np.sum(cutout) / (self.psf_cutout_size * self.psf_cutout_size) < 0.05 and \
                len(cutout) != 0:  
                # <5% of the pixels in the cutout are NaN, and the cutout isn't empty, which would happen for sources
                # that are not in the field (which is allowed because the panstarrs query has some padding around the
                # ZTF cutouts).
                not_nan[i] = 1
        not_nan = not_nan.astype(bool)
        pstarr_table = pstarr_table[not_nan]

        # Mask on detected bands
        pstarr_table = Table(pstarr_table.copy(), masked=False)
        detected_mask = (
            (pstarr_table[f'{self.band}KronMag'] != -999.0) &
            (pstarr_table[f'{self.band}KronMagErr'] != -999.0) &
            (pstarr_table[f'{self.band}PSFMag'] != -999.0)
        )
        pstarr_table = pstarr_table[detected_mask]

        # Cut on SNR, psf mag - kron mag, and psf mag upper limit
        pstarr_snr = get_snr_from_mag(
            np.array(pstarr_table[f'{self.band}KronMag']),
            np.array(pstarr_table[f'{self.band}KronMagErr']),
            self.zero_pt_mag
        )
        pstarr_table = pstarr_table[(
            (pstarr_snr >= 10) &
            (pstarr_table[f'{self.band}PSFMag'] - pstarr_table[f'{self.band}KronMag'] < 0.05) &
            (pstarr_table[f'{self.band}PSFMag'] < 17)
        )]

        # Convert to x and y coords, and store
        self._point_source_coords = SkyCoord(pstarr_table['ra'], pstarr_table['dec'], unit='deg')

    def get_kron_mags(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform photometry on detected self.sources in the image.

        Returns:
            np.ndarray: AB Kron magnitudes.
            np.ndarray: AB Kron magnitude errors.
            np.ndarray: Flag indicating whether the used Kron apeture is circular.
            np.ndarray: Sep photometry flags.

        NOTE: Sometimes the flux is calculated to be negative for sources because background subtraction makes faint
              sources ever so slightly negative. In these cases, we set the magnitude to -999.0.
        """
        print('Calculating Kron magnitudes...')

        # Get Kron radius
        self.sources['theta'][self.sources['theta'] > np.pi / 2] -= np.pi
        self.sources['theta'][self.sources['theta'] < -1 * np.pi / 2] += np.pi / 2
        kronrad, kr_flag = sep.kron_radius(
            self.image_sub,
            self.sources['x'],
            self.sources['y'],
            self.sources['a'],
            self.sources['b'],
            self.sources['theta'],
            6.0,
            mask=self.nan_mask,
            seg_id=self._segid,
        )
        self.sources['KronRad'] = kronrad

        # Kron flux for sources with a radius smaller than 1.0 are circular
        r_min = 1.5  # minimum diameter = 1
        use_circle = kronrad * np.sqrt(self.sources['a'] * self.sources['b']) < r_min
        ncflux, ncfluxerr, ncflag = sep.sum_ellipse(
            self.image_sub,
            self.sources['x'][~use_circle],
            self.sources['y'][~use_circle],
            self.sources['a'][~use_circle],
            self.sources['b'][~use_circle],
            self.sources['theta'][~use_circle],
            2.5*kronrad[~use_circle],
            err=self.bkg.globalrms,
            gain=self.gain,
            mask=self.nan_mask,
            segmap=self._segmap,
            seg_id=self._segid[~use_circle],
        )
        cflux, cfluxerr, cflag = sep.sum_circle(
            self.image_sub,
            self.sources['x'][use_circle],
            self.sources['y'][use_circle],
            2.5*self.r_fwhm,
            subpix=1,
            err=self.bkg.globalrms,
            gain=self.gain,
            mask=self.nan_mask,
            segmap=self._segmap,
            seg_id=self._segid[use_circle],
        )

        # Concatenate the fluxes
        flux = np.zeros(len(self.sources)) * np.nan
        fluxerr = np.zeros(len(self.sources)) * np.nan
        flag = np.zeros(len(self.sources)) * np.nan
        flux[~use_circle] = ncflux
        flux[use_circle] = cflux
        fluxerr[~use_circle] = ncfluxerr
        fluxerr[use_circle] = cfluxerr
        flag[~use_circle] = ncflag
        flag[use_circle] = cflag
        flag = flag.astype(int)
        flag |= kr_flag.astype(int)

        # Add mag and magerrs
        mag, magerr = img_flux_to_ab_mag(flux, self.zero_pt_mag, fluxerr=fluxerr)

        # Make all the negative fluxes have mag -999.0
        mag[flux < 0] = -999.0

        return mag, magerr, use_circle.astype(int), flag

    def get_psf_mags(
            self,
            kron_mags: np.ndarray,
            kron_flag: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform PSF photometry on detected self.sources in the image.

        Returns:
            np.ndarray: PSF Kron magnitudes.
            np.ndarray: PSF Kron magnitude errors.
            np.ndarray: Flags from https://photutils.readthedocs.io/en/stable/api/photutils.psf.PSFPhotometry.html#photutils.psf.PSFPhotometry.__call__
            astropy.Table: Quality of the fit (qfit and cfit) for each source.

        NOTE:
            qfit - the absolute value of the sum of the fit residuals divided by the fit flux.
            cfit - the fit residual in the central pixel divided by the fit flux.
        """
        print('Calculating PSF magnitudes...')

        # Can't have NaN init mags
        psf_fit_width = 5
        nan_nearby_mask = np.array([
            true_nearby(
                y,
                x,
                radius=int(psf_fit_width / 2),
                mask=self.nan_mask,
            ) for x, y in zip(self.sources['x'], self.sources['y'])
        ])
        truncated_mask = self._is_truncated(kron_flag)
        nan_init_flux_mask = np.isnan(kron_mags)
        bad_src_mask = np.logical_or(np.logical_or(nan_nearby_mask, truncated_mask), nan_init_flux_mask)
        print(f'Info for {self.band} EPSF fitting in coordinate range {self.get_coord_range()}:')
        print(f'\tNumber of sources: {len(self.sources)}')
        print(f'\tNumber of sources with NaN nearby: {np.sum(nan_nearby_mask)}')
        print(f'\tNumber of sources with truncated: {np.sum(truncated_mask)}')
        print(f'\tNumber of sources with NaN init flux: {np.sum(nan_init_flux_mask)}')

        # Convert kron mags to fluxes
        init_fluxes = img_ab_mag_to_flux(kron_mags[~bad_src_mask], self.zero_pt_mag)
        init_fluxes[kron_mags[~bad_src_mask] == -999.0] = 0.0

        # Get the initial parameters for the PSF model
        init_params = self.sources[~bad_src_mask][['x', 'y']]
        init_params['flux_init'] = init_fluxes

        # Fit the PSF model on a sample of just stars
        print(f'Fitting PSF model using {len(self._point_source_coords)} stars...')
        data_for_fit = NDData(data=self.image_sub, wcs=self.wcs)
        try:
            self.stars = extract_stars(
                data_for_fit,
                Table([self._point_source_coords], names=['skycoord']),
                size=self.psf_cutout_size,
            )
            self.stars = EPSFStars([s for s in self.stars if not np.any(np.isnan(s.data))])  # drop cutouts with nans
            fitter = EPSFFitter(fit_boxsize=self.fit_boxsize)
            epsf_builder = EPSFBuilder(fitter=fitter)
            self.epsf, _ = epsf_builder(self.stars) # self.epsf is an EPSFModel
        except Exception as e:
            e.add_note(f'Band: {self.band}\nCoordinate Range: {self.get_coord_range()}')
            raise e

        # Perform the PSF photometry with the fitted model
        self.psfphot = PSFPhotometry(self.epsf, fit_shape=psf_fit_width)
        phot = self.psfphot(self.image_sub, error=self.bkg.rms(), init_params=init_params, mask=self.nan_mask)
        phot_flags = phot['flags']
        phot_cfit = phot['cfit']
        phot_qfit = phot['qfit']

        # Convert to magnitudes
        mag, magerr = img_flux_to_ab_mag(phot['flux_fit'], self.zero_pt_mag, fluxerr=phot['flux_err'])

        # Put nans in for bad sources
        mag_final = np.zeros(len(bad_src_mask)) * np.nan
        magerr_final = np.zeros(len(bad_src_mask)) * np.nan
        phot_flags_final = np.zeros(len(bad_src_mask)) * np.nan
        phot_cfit_final = np.zeros(len(bad_src_mask)) * np.nan
        phot_qfit_final = np.zeros(len(bad_src_mask)) * np.nan
        mag_final[~bad_src_mask] = mag
        magerr_final[~bad_src_mask] = magerr
        phot_flags_final[~bad_src_mask] = phot_flags
        phot_cfit_final[~bad_src_mask] = phot_cfit
        phot_qfit_final[~bad_src_mask] = phot_qfit

        return mag_final, magerr_final, phot_flags_final, phot_cfit_final, phot_qfit_final

    def get_sources_ra_dec(self) -> np.ndarray:
        """Get the RA and DEC of the detected self.sources."""
        coords = self.pix_to_ra_dec(self.sources['x'], self.sources['y'])
        return np.vstack(coords).T

    def pix_to_ra_dec(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert pixel coordinates to RA and DEC."""
        coords = self.wcs.pixel_to_world(x, y)
        return np.array(coords.ra.deg), np.array(coords.dec.deg)

    def ra_dec_to_pix(self, ra: Union[float, np.ndarray], dec: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert RA and DEC to pixel coordinates."""
        coords = SkyCoord(ra=ra, dec=dec, unit='deg')
        coords = self.wcs.world_to_pixel(coords)
        return tuple(np.array(c) for c in coords)

    def store_coords(self, fpath: str, coord_system: str = 'ra_dec', mask: Optional[np.ndarray] = None):
        """Store the coordinates of the detected self.sources in a text file."""
        # Get the coordinates of the self.sources
        if coord_system == 'ra_dec':
            coords = self.get_sources_ra_dec()
        elif coord_system == 'pixel':
            coords = np.vstack((self.sources['x'], self.sources['y'])).T
            coords += 1  # 0-indexed to 1-indexed
        else:
            raise ValueError("Invalid coordinate system. Must be 'ra_dec' or 'pixel'.")

        # Store the coordinates in a text file
        mask = np.ones(len(self.sources), dtype=bool) if mask is None else mask
        with open(fpath, 'w') as f:
            for s in coords[mask]:
                f.write(f"{s[0]} {s[1]}\n")

    def plot_segmap(
            self,
            show_sources: bool = True,
            show_source_shapes: bool = False,
            image_is_subtracted: bool = False,
            fpath: Optional[str] = None,
            source_mask: Optional[np.ndarray] = None,
            color='red',
            ax: Optional[Axes] = None,
            **kwargs,
        ) -> Axes:
        """Plot the segmentation map of the image."""
        if ax is None:
            _, ax = plt.subplots(figsize=(15, 15))
        if image_is_subtracted:
            norm = simple_norm(self.image_sub, 'log', percent=99.75)
            ax.imshow(self.image_sub, norm=norm, origin='lower', cmap='gray')
        else:
            if self._segmap is None:
                self.get_sources()
            segmap = (self._segmap != 0).astype(float)
            ax.imshow(segmap, cmap='gray')
        source_mask = np.ones(len(self.sources), dtype=bool) if source_mask is None else source_mask
        if show_source_shapes:
            self.get_kron_mags()  # call so that kron rads are added to the sources
            sources = self.sources.copy()[source_mask]
            for j in range(len(sources)):
                e = Ellipse(xy=(sources['x'][j], sources['y'][j]),
                            width=2.5*sources['KronRad'][j]*sources['a'][j],
                            height=2.5*sources['KronRad'][j]*sources['b'][j],
                            angle=sources['theta'][j] * 180. / np.pi,
                            linewidth=2)
                e.set_facecolor('none')
                e.set_edgecolor(color)
                ax.add_artist(e)
        if show_sources:
            sources = self.sources[source_mask]
            ax.scatter(sources['x'], sources['y'], color=color, **kwargs)
        if fpath is not None:
            plt.savefig(fpath)

    def get_coord_range(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get the RA and DEC range of the image."""
        corners = [
            self.wcs.pixel_to_world(0, 0),
            self.wcs.pixel_to_world(self.image_data.shape[1], 0),
            self.wcs.pixel_to_world(self.image_data.shape[1], self.image_data.shape[0]),
            self.wcs.pixel_to_world(0, self.image_data.shape[0]),
        ]
        corner_ras = np.array([c.ra.deg for c in corners])
        corner_decs = np.array([c.dec.deg for c in corners])
        ra_range = (np.min(corner_ras), np.max(corner_ras))
        dec_range = (np.min(corner_decs), np.max(corner_decs))

        # Account for wrapping at 360 deg
        if ra_range[0] < 10.0 and ra_range[1] > 350.0:
            ra_range = (ra_range[1], ra_range[0] + 360.0)

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

        # Add the magnitudes to the table
        if include_kron or include_psf:  # kron mags are used when calculating the psf mags
            kron_mags, kron_magerrs, circle_flag, kron_flag = self.get_kron_mags()
            # kron_mags, kron_magerrs, kron_flag = self.get_kron_mags()
            data_table = Table(self.sources)
            data_table[f'{self.band}KronMag'] = kron_mags
            data_table[f'{self.band}KronMagErr'] = kron_magerrs
            data_table[f'{self.band}KronCircleFlag'] = circle_flag
            data_table[f'{self.band}KronFlag'] = kron_flag
        if include_psf:
            psf_mags, psf_magerrs, psf_flags, cfit, qfit = self.get_psf_mags(kron_mags, kron_flag)
            data_table[f'{self.band}PSFMag'] = psf_mags
            data_table[f'{self.band}PSFMagErr'] = psf_magerrs
            data_table[f'{self.band}PSFFlags'] = psf_flags
            data_table['qfit'] = qfit
            data_table['cfit'] = cfit

        # Add some additional info
        coords = self.get_sources_ra_dec()
        data_table['ra'] = coords[:, 0]
        data_table['dec'] = coords[:, 1]
        data_table[f'{self.band}_zero_pt_mag'] = self.zero_pt_mag
        maglim = self.maglimit if self.maglimit is not None else np.nan
        data_table[f'{self.band}_mag_limit'] = maglim

        return data_table


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


def query_cone_ps1(
        ra_deg: Union[float, np.ndarray],
        dec_deg: Union[float, np.ndarray],
        search_radius_arcmin: float,
    ) -> Table:
    '''
    Adapted from FLEET.
    '''

    # Get the PS1 MAST username and password from /Users/username/3PI_key.txt
    key_location = os.path.join(pathlib.Path.home(), f'vault/{MAST_CREDENTIAL_FNAME}')

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
    results = MASTCASJOBS.quick(la_query, task_name="python cone search")

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
