import os
import sys
import ast
import traceback
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from concurrent import futures
from typing import List, Tuple, Union, Optional, Iterable, Dict
from matplotlib.axes._axes import Axes

from astropy.wcs import WCS
from astropy.time import Time
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.visualization import simple_norm, time_support
from astropy.io.fits import HDUList
from astroquery.gaia import Gaia
from astroquery.sdss import SDSS

sys.path.append('/Users/adamboesky/Research/long_transients')

from Extracting.utils import get_data_path, load_cached_table
from Extracting.Catalogs import ZTF_Catalog, ZTF_CUTOUT_HALFWIDTH, get_ztf_metadata_from_coords, get_ztf_metadata_from_metadata, get_pstarr_cutout
from ztf_fp_query.Forced_Photo_Map import Forced_Photo_Map
from ztf_fp_query.query import ZTFFP_Service
try:
    from Light_Curve import Light_Curve, LC_MARKER_INFO, LC_COLOR_INFO, ALL_BAND_DF
except ModuleNotFoundError:
    from .Light_Curve import Light_Curve, LC_MARKER_INFO, LC_COLOR_INFO, ALL_BAND_DF
time_support()

ACCEPTABLE_PROC_STATUS = [0]
MANDATORY_SOURCE_COLUMNS = [
    'ra', 'dec', 'PSTARR_rPSFMag', 'PSTARR_iKronMagErr', 'PSTARR_rApMagErr', 'PSTARR_iApMag', 'PSTARR_primaryDetection',
    'PSTARR_gApMag', 'PSTARR_rinfoFlag2', 'PSTARR_rpsfLikelihood', 'PSTARR_gApMagErr', 'PSTARR_gPSFMagErr',
    'PSTARR_gKronMag', 'PSTARR_gKronMagErr', 'PSTARR_iinfoFlag2', 'PSTARR_dec', 'PSTARR_rKronMagErr', 'PSTARR_rApMag',
    'PSTARR_iApMagErr', 'PSTARR_ginfoFlag2', 'PSTARR_ipsfLikelihood', 'PSTARR_rKronMag', 'PSTARR_iPSFMagErr',
    'PSTARR_ra', 'PSTARR_gpsfLikelihood', 'PSTARR_iPSFMag', 'PSTARR_PanSTARR_ID', 'PSTARR_gPSFMag', 'PSTARR_rPSFMagErr',
    'PSTARR_iKronMag', 'ZTF_g_b', 'ZTF_r_cpeak', 'ZTF_r_errx2', 'ZTF_i_xmin', 'ZTF_i_xmax', 'ZTF_rKronCircleFlag',
    'ZTF_r_a', 'ZTF_i_xpeak', 'ZTF_i_theta', 'ZTF_i_cxy', 'ZTF_g_thresh', 'ZTF_iPSFMag', 'ZTF_i_mag_limit',
    'ZTF_g_npix', 'ZTF_iKronCircleFlag', 'ZTF_g_ra', 'ZTF_r_npix', 'ZTF_g_cxy', 'ZTF_r_sepExtractionFlag',
    'ZTF_i_errxy', 'ZTF_g_flux', 'ZTF_i_a', 'ZTF_iKronFlag', 'ZTF_r_mag_limit', 'ZTF_g_cpeak', 'ZTF_gPSFMag',
    'ZTF_iPSFFlags', 'ZTF_gKronCircleFlag', 'ZTF_i_y', 'ZTF_r_thresh', 'ZTF_i_thresh', 'ZTF_i_x', 'ZTF_g_ymin',
    'ZTF_i_upper_lim_flag', 'ZTF_r_tnpix', 'ZTF_r_xcpeak', 'ZTF_i_xcpeak', 'ZTF_r_ycpeak', 'ZTF_i_erry2', 'ZTF_g_peak',
    'ZTF_r_dec', 'ZTF_r_xpeak', 'ZTF_i_xy', 'ZTF_i_ymax', 'ZTF_r_qfit', 'ZTF_g_tnpix', 'ZTF_g_errx2', 'ZTF_r_cyy',
    'ZTF_iPSFMagErr', 'ZTF_g_xy', 'ZTF_i_zero_pt_mag', 'ZTF_i_cpeak', 'ZTF_gKronMag', 'ZTF_i_y2', 'ZTF_g_a', 'ZTF_i_ra',
    'ZTF_r_cxx', 'ZTF_r_flux', 'ZTF_r_peak', 'ZTF_r_xy', 'ZTF_g_ycpeak', 'ZTF_r_ymin', 'ZTF_r_x2', 'ZTF_i_cyy',
    'ZTF_KronRad', 'ZTF_rPSFMag', 'ZTF_g_erry2', 'ZTF_rKronMagErr', 'ZTF_g_dec', 'ZTF_g_qfit', 'ZTF_i_qfit',
    'ZTF_i_sepExtractionFlag', 'ZTF_r_cfit', 'ZTF_i_cfit', 'ZTF_i_x2', 'ZTF_i_cflux', 'ZTF_r_errxy', 'ZTF_g_errxy',
    'ZTF_g_xmax', 'ZTF_r_cxy', 'ZTF_i_peak', 'ZTF_r_ypeak', 'ZTF_g_mag_limit', 'ZTF_i_b', 'ZTF_i_ypeak', 'ZTF_g_xpeak',
    'ZTF_rKronFlag', 'ZTF_g_cyy', 'ZTF_g_sepExtractionFlag', 'ZTF_r_b', 'ZTF_r_upper_lim_flag', 'ZTF_i_ymin',
    'ZTF_r_ra', 'ZTF_r_xmin', 'ZTF_r_y2', 'ZTF_r_cflux', 'ZTF_gPSFMagErr', 'ZTF_gPSFFlags', 'ZTF_i_ycpeak', 'ZTF_g_x',
    'ZTF_g_y2', 'ZTF_g_zero_pt_mag', 'ZTF_rKronMag', 'ZTF_g_y', 'ZTF_g_upper_lim_flag', 'ZTF_iKronMag', 'ZTF_g_xmin',
    'ZTF_r_theta', 'ZTF_g_x2', 'ZTF_i_errx2', 'ZTF_i_dec', 'ZTF_gKronMagErr', 'ZTF_r_erry2', 'ZTF_i_cxx',
    'ZTF_rPSFFlags', 'ZTF_iKronMagErr', 'ZTF_g_cflux', 'ZTF_r_y', 'ZTF_r_zero_pt_mag', 'ZTF_i_flux', 'ZTF_r_x',
    'ZTF_r_ymax', 'ZTF_g_cxx', 'ZTF_i_npix', 'ZTF_r_xmax', 'ZTF_g_cfit', 'ZTF_g_theta', 'ZTF_rPSFMagErr', 'ZTF_g_ypeak',
    'ZTF_g_xcpeak', 'ZTF_g_ymax', 'ZTF_i_tnpix', 'ZTF_gKronFlag', 'x', 'y', 'association_separation_arcsec',
    'Catalog_Flag', 'Catalog', 'filter_info', 'ZTF_g_field', 'ZTF_g_ccdid', 'ZTF_g_qid', 'ZTF_r_field', 'ZTF_r_ccdid',
    'ZTF_r_qid', 'ZTF_i_field', 'ZTF_i_ccdid', 'ZTF_i_qid',
]
Gaia.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'


def closest_within_radius(coord: SkyCoord, coords: SkyCoord, max_arcsec: float = 1.0) -> Tuple[int, SkyCoord]:
    """Finds the closest coordinate in 'coords' to 'coord' that is less than a given distance away."""
    # Calculate separations between coord and each coordinate in coords
    seps = coord.separation(coords)
    within_one_arcsecond = seps < max_arcsec * u.arcsec

    # If there are no coordinates within one arcsecond, return None
    if not any(within_one_arcsecond):
        return None, None

    # Find the index of the closest coordinate within the one-arcsecond range
    seps[np.isnan(seps)] = np.inf * u.arcsec
    closest_index = seps.argmin()
    closest_coord = coords[closest_index]

    return closest_index, closest_coord


class Postage_Stamp():
    def __init__(
            self,
            ra: float,
            dec: float,
            stamp_width_arcsec: int = 50,
            bands: list = ['g', 'r', 'i'],
            ztf_data_dir: Optional[str] = None,
        ):
        self.ra = ra
        self.dec = dec
        self.coord = SkyCoord(ra=ra, dec=dec, unit='deg')
        self.bands = bands
        self.stamp_width_arcsec = stamp_width_arcsec
        self.arcsec_per_pixel = None
        self.ztf_data_dir = ztf_data_dir

        # Offset used for coordinates in the cutout image and the WCS
        self.x_origin_offsets, self.y_origin_offsets = {band: 0 for band in self.bands}, {band: 0 for band in self.bands}

        # Properties
        self._images = None
        self._WCSs = None

    @property
    def images(self) -> Dict[str, np.ndarray]:
        if self._images is None:
            self._images, self._WCSs = self.get_images()

        return self._images

    @property
    def WCSs(self) -> Dict[str, np.ndarray]:
        if self._WCSs is None:
            self._images, self._WCSs = self.get_images()

        return self._WCSs

    def get_images(self) -> Tuple[Dict[str, np.ndarray], Dict[str, WCS]]:
        raise NotImplementedError('This method must be implemented in a subclass!')

    def transform_pix_coords(
            self,
            x: Union[float, np.ndarray],
            y: Union[float, np.ndarray],
            band: str,
        ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        return x, y

    def origpix_to_current(
            self,
            x_orig: Union[float, np.ndarray],
            y_orig: Union[float, np.ndarray],
            band: str,
        ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        x, y = self.transform_pix_coords(
            x_orig - self.x_origin_offsets[band],
            y_orig - self.y_origin_offsets[band],
            band=band,
        )
    
        return x, y

    def currentpix_to_orig(
            self,
            x_current: Union[float, np.ndarray],
            y_current: Union[float, np.ndarray],
            band: str,
        ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        x_current, y_current = self.transform_pix_coords(x_current, y_current, band=band)

        return x_current + self.x_origin_offsets[band], y_current + self.y_origin_offsets[band]

    def pix_to_ra_dec(
            self,
            x: Union[float, np.ndarray],
            y: Union[float, np.ndarray],
            band: str,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert pixel coordinates to RA and DEC."""
        x, y = self.currentpix_to_orig(x, y, band=band)
        coords = self.WCSs[band].pixel_to_world(x, y)

        return np.array(coords.ra.deg), np.array(coords.dec.deg)

    def ra_dec_to_pix(
            self,
            ra: Union[float, np.ndarray],
            dec: Union[float, np.ndarray],
            band: str,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert RA and DEC to pixel coordinates."""
        coords = SkyCoord(ra=ra, dec=dec, unit='deg')
        coords = self.WCSs[band].world_to_pixel(coords)
        x, y = self.origpix_to_current(coords[0], coords[1], band=band)

        return x, y

    def plot_cutout(
            self,
            band: str,
            xs: Optional[Iterable] = None,
            ys: Optional[Iterable] = None,
            ax: Optional[Axes] = None,
            show_center: bool = True,
            **kwargs
        ) -> Axes:
        """Plot the cutout image."""
        # Make an axis if not given
        if ax is None:
            _, ax = plt.subplots()

        # If no image, just return
        if self.images[band] is None:
            return ax

        # Set up plotting parameters
        if 'origin' not in kwargs:
            kwargs['origin'] = 'lower'
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'viridis'

        # Plot the image
        norm = simple_norm(self.images[band], 'log', percent=99.5)  # scaling for visual purposes
        ax.imshow(self.images[band], norm=norm, **kwargs)

        # Show the center
        # NOTE: -1 offset is due to the 0-based indexing of the image
        reticle_offset = self.images[band].shape[0] * 0.03
        if show_center:
            x, y = self.ra_dec_to_pix(self.ra, self.dec, band)
            ax.scatter(x - reticle_offset - 1, y - 1, color='red', marker='_')
            ax.scatter(x - 1, y - reticle_offset - 1, color='red', marker='|')

        # Show the given sources
        # NOTE: -1 offset is due to the 0-based indexing of the image
        if xs is not None and ys is not None:
            x, y = self.ra_dec_to_pix(xs, ys, band)
            ax.scatter(x - reticle_offset - 1, y - 1, color='k', marker='_')
            ax.scatter(x - 1, y - reticle_offset - 1, color='k', marker='|')

        return ax


class ZTF_Postage_Stamp(Postage_Stamp):
    def __init__(self, *args, image_metadata: Dict[str, Dict] = {}, **kwargs):
        super().__init__(*args, **kwargs)
        self.arcsec_per_pixel = 1.01
        self.image_metadata = image_metadata

    def transform_pix_coords(
            self,
            x: Union[float, np.ndarray],
            y: Union[float, np.ndarray],
            band: str,
        ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        return self.images[band].shape[1] - x - 1, self.images[band].shape[0] - y - 1


    def get_images(self) -> Tuple[Dict[str, np.ndarray], Dict[str, WCS]]:

        # Get the bands with images for the given metadata and load ztf catalogs
        bands_with_images = [
            b[1] for b in get_ztf_metadata_from_metadata(ztf_metadata={
                'fieldid': self.image_metadata.get('fieldid', self.image_metadata.get('field')),
                'ccdid': self.image_metadata['ccdid'],
                'qid': self.image_metadata['qid'],
            })['filtercode']
        ]
        bands_with_images = [b for b in bands_with_images if b in self.bands]

        # Preload ztf catalogs in parallel to speed up image downloads
        with futures.ThreadPoolExecutor() as executor:
            ztf_catalogs = dict(zip(
            bands_with_images,
            executor.map(lambda band: ZTF_Catalog(
                self.ra,
                self.dec,
                band=band,
                image_metadata=self.image_metadata,
                data_dir=self.ztf_data_dir,
            ), bands_with_images)
        ))

        # Get the images and crop to postage stamps
        self._images, self._WCSs = {}, {}
        for band in self.bands:
            if band not in bands_with_images:
                self._images[band] = None
            else:
                im = ztf_catalogs[band].sextractor.image_sub
                self._WCSs[band] = ztf_catalogs[band].sextractor.wcs

                # Get the pixel location of the center of the image
                x, y = ztf_catalogs[band].sextractor.ra_dec_to_pix(self.ra, self.dec)

                # Get the desired region
                halfwidth_pixels = 0.5 * self.stamp_width_arcsec / self.arcsec_per_pixel
                self._images[band] = im[
                    int(y - halfwidth_pixels):int(y + halfwidth_pixels),
                    int(x - halfwidth_pixels):int(x + halfwidth_pixels),
                ]

                # Adjust the origin
                self.x_origin_offsets[band] += x - (halfwidth_pixels)
                self.y_origin_offsets[band] += y - (halfwidth_pixels)

                # Rotate and flip the image to align with PanSTARRS orientation
                self._images[band] = np.flipud(np.fliplr(self._images[band]))

        return self._images, self._WCSs


class PSTARR_Postage_Stamp(Postage_Stamp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arcsec_per_pixel = 0.25

    def get_images(self) -> Tuple[Dict[str, np.ndarray], Dict[str, WCS]]:

        # Get the images from the ZTF cutouts
        images = {}
        wcss = {}
        with futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(
            lambda band: (band, get_pstarr_cutout(self.ra, self.dec, size=self.stamp_width_arcsec / self.arcsec_per_pixel, filter=band)),
            self.bands
            ))

        for band, (image, wcs) in results:
            images[band] = image
            wcss[band] = wcs

        return images, wcss


class Source():
    def __init__(
            self,
            ra: float,
            dec: float,
            bands: list = ['g', 'r', 'i'],
            cutout_bands: Optional[List[str]] = None,
            merged_field_basedir: str = '/Users/adamboesky/Research/long_transients/Data/catalog_results/field_results',
            ztf_data_dir: Optional[str] = None,
            field_catalogs: Optional[dict[str, Table]] = None,
            max_arcsec: float = 1.0,
            gaia_max_arcsec: float = 5.0,
            verbose: int = 1,
            catch_plotting_exceptions: bool = True,
            lc_catalogs: List[str] = ['ztf', 'wise', 'ptf', 'sdss', 'panstarrs', 'gaia'],
        ):
        self.ra = ra
        self.dec = dec
        self.coord = SkyCoord(ra=ra, dec=dec, unit='deg')
        self.verbose = verbose
        self.catch_plotting_exceptions = catch_plotting_exceptions

        # Get the field for the given RA and DEC
        self.bands = bands
        self.cutout_bands = bands if cutout_bands is None else cutout_bands
        self.merged_field_basedir = merged_field_basedir
        self.ztf_data_dir = ztf_data_dir
        self.lc_catalogs = lc_catalogs

        # The maximum distance for an object to be considered a match
        # We will use a different value for our GAIA queries because sources are likely in motion or have a considerable
        # parallax
        self.max_arcsec = max_arcsec
        self.gaia_max_arcsec = gaia_max_arcsec

        # Properties
        self._field_catalogs = field_catalogs
        self._data = None
        self._postage_stamps = None
        self._ztf_lightcurve = None
        self._paddedfield = None
        self._GAIA_info = None
        self.filter_info = {}
        self._image_metadata = None
        self._spectrum = None
        self._has_spectrum = True
        self._light_curve = None

    @property
    def light_curve(self) -> Light_Curve:
        """The lightcurve class for the source"""
        if self._light_curve is None:
            self._light_curve = Light_Curve(
                self.ra,
                self.dec,
                query_rad_arcsec=self.max_arcsec,
                catalogs=self.lc_catalogs,
                pstarr_coord=(
                    self.data['PSTARR_ra'][0],
                    self.data['PSTARR_dec'][0],
                ) if not np.any(
                    np.isnan((
                        self.data['PSTARR_ra'][0],
                        self.data['PSTARR_dec'][0],
                    ))
                ) else None,
            )

        return self._light_curve

    @property
    def spectrum(self) -> Union[List[HDUList], None]:
        if self._has_spectrum and self._spectrum is None:
            # The source spectrum
            print('Getting source spectrum from SDSS...')
            sdss_res = SDSS.query_region(self.coord, radius=self.max_arcsec*u.arcsec, spectro=True)

            if sdss_res is None:
                print(f'Source at ({self.coord.ra.deg}, {self.coord.dec.deg}) has no spectrum in SDSS.')
                self._has_spectrum = False
            else:
                self._spectrum = SDSS.get_spectra(matches=sdss_res) if len(sdss_res) > 0 else None

        return self._spectrum

    @property
    def image_metadata(self) -> Dict[str, Dict]:
        if self._image_metadata is None:

            # Iterate through bands and get the first metadata that works
            metadata = {}
            field_vals = np.array([self.data[k][0] for k in self.data.columns if 'field' in k])
            ccd_vals = np.array([self.data[k][0] for k in self.data.columns if 'ccd' in k])
            qid_vals = np.array([self.data[k][0] for k in self.data.columns if 'qid' in k])
            if np.sum(~np.isnan(field_vals)) > 0:
                metadata['fieldid'] = int(field_vals[~np.isnan(field_vals)][0])
                metadata['ccdid'] = int(ccd_vals[~np.isnan(ccd_vals)][0])
                metadata['qid'] = int(qid_vals[~np.isnan(qid_vals)][0])

            if len(metadata) >= 3:
                self._image_metadata = metadata

            # for band in self.bands:
            #     if isinstance(self.data[f'ZTF_{band}_fieldid'][0], (float, int)) and not np.isnan(self.data[f'ZTF_{band}_fieldid'][0]):
            #         metadata['fieldid'] = int(self.data[f'ZTF_{band}_fieldid'][0])
            #     if isinstance(self.data[f'ZTF_{band}_ccdid'][0], (float, int)) and not np.isnan(self.data[f'ZTF_{band}_ccdid'][0]):
            #         metadata['ccdid'] = int(self.data[f'ZTF_{band}_ccdid'][0])
            #     if isinstance(self.data[f'ZTF_{band}_qid'][0], (float, int)) and not np.isnan(self.data[f'ZTF_{band}_qid'][0]):
            #         metadata['qid'] = int(self.data[f'ZTF_{band}_qid'][0])

            #     if len(metadata) >= 3:
            #         self._image_metadata = metadata
            #         break

        # If was not set, use (ra, dec) query
        if self._image_metadata is None:
            print('Falling back on ZTF image metadata with coordinate query...')
            self._image_metadata = get_ztf_metadata_from_coords(
                ra_range=(self.ra - ZTF_CUTOUT_HALFWIDTH * 1, self.ra + ZTF_CUTOUT_HALFWIDTH * 1),
                dec_range=(self.dec - ZTF_CUTOUT_HALFWIDTH * 1, self.dec + ZTF_CUTOUT_HALFWIDTH * 1),
            )

            # Filter for fields that we have actually extracted and stored
            inds_extracted = []
            test_paths = []
            for ind, field in self._image_metadata['field'].items():
                field_id = str(field).zfill(6)
                if os.path.exists(os.path.join(self.merged_field_basedir, f'{field_id}_g.ecsv')):
                    inds_extracted.append(ind)
                    test_paths.append(os.path.join(self.merged_field_basedir, f'{field_id}_g.ecsv'))
            if len(inds_extracted) > 0:
                self._image_metadata = self._image_metadata.iloc[inds_extracted].copy().reset_index(drop=True)
            else:
                self._image_metadata = self._image_metadata.iloc[[0]].copy().reset_index(drop=True)

            # If we haven't extracted this field, throw and error
            if len(self._image_metadata) == 0 or self._image_metadata is None:
                raise ValueError(f'Metadata for source at {self.coord} appears to be for a field that has not been extracted.')

            # Sort by the biggest stack, and make into a dict
            self._image_metadata.sort_values(by=['nframes'], ascending=False, inplace=True, ignore_index=True)
            self._image_metadata = self._image_metadata.reset_index(drop=True).iloc[0].to_dict()

            # Take metadata out of list form
            for k, v in self._image_metadata.items():
                self._image_metadata[k] = v

        return self._image_metadata

    @property
    def field_catalogs(self) -> dict[str, Table]:
        """Dictionary containing the catalogs for each band in the ZTF field that contains the current source."""
        if self._field_catalogs is None:
            print('Loading catalogs!')
            self._field_catalogs = {}

            def load_catalog(band):
                padded_field = self.image_metadata[band]["field"].zfill(6)
                print(f'Loading {band} catalog from locally stored catalog {padded_field}_{band}...')
                return band, load_cached_table(os.path.join(self.merged_field_basedir, f'{padded_field}_{band}.ecsv')).copy()

            with futures.ThreadPoolExecutor() as executor:
                future_to_band = {executor.submit(load_catalog, band): band for band in self.bands}
                for future in futures.as_completed(future_to_band):
                    band, catalog = future.result()
                    self._field_catalogs[band] = catalog

        return self._field_catalogs

    @property
    def data(self) -> Table:
        if self._data is None:

            # Get the unique columns
            unique_colnames = []
            for band, tab in self.field_catalogs.items():
                for cname in tab.colnames:

                    # The ZTF magnitude columns already have band names in them, the others don't
                    if 'ZTF' in cname and (
                        'Kron' not in cname and
                        'PSF' not in cname and
                        'mag_limit' not in cname and
                        'zero_pt_mag' not in cname
                    ) and cname[:6] != f'ZTF_{band}_':  # also making sure that column isn't already modified
                        new_cname = f'ZTF_{band}_{cname[4:]}'
                        self.field_catalogs[band].rename_column(cname, new_cname)
                        unique_colnames.append(new_cname)
                    else:
                        unique_colnames.append(cname)

            # Add the mandatory columns and make sure they're unique
            unique_colnames += MANDATORY_SOURCE_COLUMNS
            unique_colnames = set(unique_colnames)

            # Make the empty table to fill in
            data_dict = {k: [np.nan] for k in unique_colnames}
            str_cols = ['Catalog']  # need to have types align
            for col in str_cols:
                data_dict[col] = [str(data_dict[col][0])]
            self._data = Table(data_dict)

            # Make sure the catalog column is a long enough string
            if 'Catalog' in self._data.colnames:
                self._data['Catalog'] = self._data['Catalog'].astype('S10')

            if self.verbose > 0: print('Searching for source in the catalogs!')
            for band, cat in self.field_catalogs.items():
                if self.verbose > 0: print(f'Searching {band} catalog for source...')

                # Get the source from each catalog
                coords = SkyCoord(ra=cat['ra'], dec=cat['dec'], unit='deg')
                ind_closest, _ = closest_within_radius(self.coord, coords, max_arcsec=self.max_arcsec)

                # If a coord was found, join the tables
                if ind_closest is not None:

                    # Fill in table
                    for cname in cat.colnames:
                        if cname == 'filter_info':
                            # Convert the filter information to a dictionary
                            if isinstance(cat[cname][ind_closest], str):
                                self.filter_info = ast.literal_eval(cat[cname][ind_closest])['in_bands']
                            entry_is_nan = False
                        elif isinstance(self._data[cname][0], str):
                            entry_is_nan = self._data[cname][0].lower() == 'nan'
                        else:
                            entry_is_nan = np.isnan(self._data[cname][0])
                        if entry_is_nan:
                            if isinstance(cat[cname][ind_closest], bool):
                                self._data[cname][0] = float(cat[cname][ind_closest])
                            else:
                                self._data[cname][0] = cat[cname][ind_closest]

        # Reorder columns
        pstarr_cols = [col for col in self._data.colnames if col.startswith('PSTARR')]
        ztf_cols = [col for col in self._data.colnames if col.startswith('ZTF')]
        other_cols = [col for col in self._data.colnames if not (col.startswith('PSTARR') or col.startswith('ZTF'))]
        ordered_cols = ['ra', 'dec'] + pstarr_cols + ztf_cols + [col for col in other_cols if col not in ['ra', 'dec']]
        self._data = self._data[ordered_cols]

        return self._data

    @property
    def postage_stamps(self) -> Dict[str, Postage_Stamp]:
        if self._postage_stamps is None:
            self._postage_stamps = {
                'ZTF': ZTF_Postage_Stamp(self.ra, self.dec, bands=self.cutout_bands, image_metadata=self.image_metadata, ztf_data_dir=self.ztf_data_dir),
                'PSTARR': PSTARR_Postage_Stamp(self.ra, self.dec, bands=self.cutout_bands),
            }

        return self._postage_stamps

    def _add_lc_mag_columns(self, lc: pd.DataFrame):
        # Recommended SNT and SNU from https://web.ipac.caltech.edu/staff/fmasci/ztf/forcedphot.pdf
        snt = 3
        snu = 5

        # Make arrays and mask for upper lims
        mag = np.zeros(len(lc)) * np.nan
        sigma_mag = np.zeros(len(lc)) * np.nan
        upper_lim_mask = (lc['forcediffimflux'] / lc['forcediffimfluxunc']) < snt

        # Fill in arrays
        mag[~upper_lim_mask] = (lc['zpdiff'] - 2.5 * np.log10(lc['forcediffimflux']))[~upper_lim_mask]
        sigma_mag[~upper_lim_mask] = (1.0857 * lc['forcediffimfluxunc'] / \
                                     lc['forcediffimflux'])[~upper_lim_mask]
        mag[upper_lim_mask] = (lc['zpdiff'] - \
                              2.5 * np.log10(snu * lc['forcediffimfluxunc']))[upper_lim_mask]

        # Add to lc dataframe
        lc['mag'] = mag
        lc['magerr'] = sigma_mag
        lc['upperlim'] = upper_lim_mask.astype(int)

        return lc

    @property
    def ztf_lightcurve(self) -> pd.DataFrame:
        if self._ztf_lightcurve is None:

            # Get the lightcurve filename if it exists
            photo_map = Forced_Photo_Map()
            lc_fname = photo_map.get_lightcurve_fname(self.ra, self.dec)

            # If it doesn't exist, submit a query
            if len(lc_fname) == 0:
                print(f'No lightcurve found for source with ra, dec = ({self.ra}, {self.dec}).')
                return self._ztf_lightcurve

            # Path to the data
            data_path = get_data_path()

            # If it exists, load it and return
            data = pd.read_csv(
                os.path.join(data_path, 'ztf_forced_photometry', lc_fname[0]),
                sep=r'\s+',
                comment='#',
                header=0,
            )
            data.columns = [c.replace(',', '') for c in data.columns]
            self._ztf_lightcurve = data

            # Add magnitudes to the lc dataframe
            self._ztf_lightcurve = self._add_lc_mag_columns(self._ztf_lightcurve)

        return self._ztf_lightcurve

    def plot_postage_stamps(self, band: str, axes: Optional[Axes] = None, add_labels: bool = True, **kwargs) -> Axes:
        # Make axes if not given
        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(15, 7.5))

        # Plot
        if self.catch_plotting_exceptions:  # if the user wants to catch plotting exceptions
            try:
                self.postage_stamps['PSTARR'].plot_cutout(band=band, ax=axes[0], **kwargs)
            except Exception as e:
                    tb = traceback.format_exc()
                    print(f'Warning: Experienced error plotting the PanSTARR {band} band:\n{e}\nTraceback:\n{tb}\nSkipping...')
        else:
            self.postage_stamps['PSTARR'].plot_cutout(band=band, ax=axes[0], **kwargs)
        if self.catch_plotting_exceptions:
            try:
                self.postage_stamps['ZTF'].plot_cutout(band=band, ax=axes[1], **kwargs)
            except Exception as e:
                tb = traceback.format_exc()
                print(f'Warning: Experienced error plotting the ZTF {band} band:\n{e}\nTraceback:\n{tb}\nSkipping...')
        else:
            self.postage_stamps['ZTF'].plot_cutout(band=band, ax=axes[1], **kwargs)

        # Annotate with the mags
        axes[0].text(
            0.01,
            0.99,
            rf'{band} mag = {self.data[f"PSTARR_{band}PSFMag"][0]:.2f}',
            transform=axes[0].transAxes,
            ha='left',
            va='top',
            fontsize=15,
            color='red'
        )
        axes[1].text(
            0.01,
            0.99,
            rf'{band} mag = {self.data[f"ZTF_{band}PSFMag"][0]:.2f}',
            transform=axes[1].transAxes,
            ha='left',
            va='top',
            fontsize=15,
            color='red'
        )

        # Formatting
        if add_labels:
            axes[0].set_title('Pan-STARRS', fontsize=15)
            axes[1].set_title('ZTF', fontsize=15)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        return axes

    def _filter_lc_proc_status(
            self,
            lc: pd.DataFrame,
            acceptable_proc_status: List[int] = ACCEPTABLE_PROC_STATUS
        ) -> np.ndarray:
        # Make mask for acceptable processing statuses
        proc_statuses = lc['procstatus'].to_numpy()
        proc_statuses = [np.array(p.split(',')).astype(int) for p in proc_statuses]
        mask = np.array([np.all(np.isin(proc_status, acceptable_proc_status)) for proc_status in proc_statuses])

        return lc[mask]

    def plot_ztf_lightcurve(
            self,
            bands: Optional[str] = None,
            ax: Optional[Axes] = None,
            colors: Dict[str, str] = {'g': 'forestgreen', 'r': 'lightcoral', 'i': 'darkorchid'},
            include_upper_lim: bool = True,
            time_offset: Union[str, float] = 'first',
            acceptable_proc_status: List[int] = ACCEPTABLE_PROC_STATUS,
            y_units: str = 'mag',
            **kwargs,
        ) -> Axes:
        """Plot the lightcurve."""
        # Make an axis if not given
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 5))

        # Update kwargs with default parameters if not already provided
        default_params = {
            'markersize': 5,
            'capsize': 2,
            'fmt': 'o'
        }
        for key, value in default_params.items():
            kwargs.setdefault(key, value)

        # Handle the time offset
        if time_offset == 'first':
            # NOTE: This is not band specific so that we can plot source bands on same axis
            time_offset = np.nanmin(self.ztf_lightcurve['jd'])

        # Adapt the kwargs for scatters
        scatter_kwargs = kwargs.copy()
        scatter_kwargs['marker'] = 'v'
        if scatter_kwargs.get('markersize'):
            scatter_kwargs['s'] = scatter_kwargs.get('markersize') * 4
        else:
            scatter_kwargs['s'] = None
        for bad_kwarg in ['markersize', 'capsize', 'fmt', 'markershape']:
            scatter_kwargs.pop(bad_kwarg, None)

        if bands is None:
            bands = self.bands
        for band in self.bands:

            # Get the right band
            lc = self.ztf_lightcurve[self.ztf_lightcurve['filter'] == f'ZTF_{band}']

            # Only get the acceptable processing status
            lc = self._filter_lc_proc_status(lc, acceptable_proc_status=acceptable_proc_status)

            # Get the key based on what the y units are
            if y_units == 'mag':
                y_key = 'mag'
                yerr_key = 'magerr'
            elif y_units == 'flux':
                y_key = 'forcediffimflux'
                yerr_key = 'forcediffimfluxunc'
            else:
                raise ValueError(f'Invalid y_units: {y_units}')

            # Plot
            not_upper_lim_mask = np.logical_not(lc['upperlim']).to_numpy()
            ax.errorbar(
                x=lc['jd'][not_upper_lim_mask] - time_offset,
                y=lc[y_key][not_upper_lim_mask],
                yerr=lc[yerr_key][not_upper_lim_mask],
                color=colors[band],
                **kwargs,
            )
            if include_upper_lim:

                # Plot upper limits
                ax.scatter(
                    x=lc['jd'][~not_upper_lim_mask] - time_offset,
                    y=lc[y_key][~not_upper_lim_mask],
                    color=colors[band],
                    **scatter_kwargs,
                )

        # Formatting
        if y_units == 'mag':
            ax.invert_yaxis()

        return ax

    def plot_all_cutouts(self, axes: Optional[Iterable[Axes]] = None, **kwargs) -> np.ndarray[Axes]:
        n_bands = len(self.bands)

        # Make axes if not given
        if axes is None:
            _, axes = plt.subplots(2, n_bands, figsize=(15, 3 * n_bands))
        if not isinstance(axes, np.ndarray):
            axes = np.array(axes)

        # Plot
        for band, ax_col in zip(self.bands, axes.T):
            self.plot_postage_stamps(band=band, axes=ax_col, add_labels=False, **kwargs)
            ax_col[1].set_xlabel(band, fontsize=15)

        # Formatting
        axes[0, 0].set_ylabel('Pan-STARRS', fontsize=15)
        axes[1, 0].set_ylabel('ZTF', fontsize=15)

        return axes

    def plot_cutouts_and_light_curves(
            self,
            ax_pstarr_cutout: Optional[Axes] = None,
            ax_ztf_cutout: Optional[Axes] = None,
            ax_light_curves: Optional[Axes] = None,
            acceptable_proc_status: List[int] = ACCEPTABLE_PROC_STATUS,
            y_units: str = 'mag',
        ) -> Tuple[Axes, Axes, Axes]:
        # If any of the axes are not given, make new axes
        if None in [ax_pstarr_cutout, ax_ztf_cutout, ax_light_curves]:

            # Make plot grid
            fig = plt.figure(figsize=(12, 10), layout="constrained")
            spec = fig.add_gridspec(2, 2)

            # Pick axes
            ax_pstarr_cutout = fig.add_subplot(spec[0, 0])
            ax_ztf_cutout = fig.add_subplot(spec[0, 1])
            ax_light_curves = fig.add_subplot(spec[1, :])

        # Plot
        self.plot_postage_stamps(band=self.bands[0], axes=[ax_pstarr_cutout, ax_ztf_cutout])
        self.plot_ztf_lightcurve(
            bands=self.bands,
            ax=ax_light_curves,
            acceptable_proc_status=acceptable_proc_status,
            y_units=y_units,
        )

        # Formatting
        ax_pstarr_cutout.set_title('Pan-STARRS', fontsize=15)
        ax_ztf_cutout.set_title('ZTF', fontsize=15)
        ax_light_curves.grid(ls=':', lw=0.5)
        ax_light_curves.set_xlabel('Time [day]')
        ax_light_curves.set_ylabel('Mag' if y_units == 'mag' else 'Fluxdiff')

        return ax_pstarr_cutout, ax_ztf_cutout, ax_light_curves

    def plot_lc(
            self,
            bands: Optional[List[str]] = None,
            ax: Optional[Axes] = None,
            fig: Optional[Axes] = None,
            time_as_str: bool = True,
            xlab_kwags: dict = {'rotation': 45, 'ha': 'right'},
            include_wise: bool = True,
            **kwargs,
        ) -> Axes:
        """Plot lightcurve for all bands specified in 'bands', or all bands if bands is None."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        elif fig is None:
            fig = plt.gcf()

        # Annotate if no light curve present
        if self.light_curve.lc is None:
            ax.text(0.5, 0.5, 'Source has no light curve.', horizontalalignment='center', verticalalignment='center')
            return ax

        # If not given, get the bands from the lightcurve data itself
        # don't include wise data
        if bands is None:
            bands = [
                b for b in self.light_curve.lc.colnames if (b[-4:] == '_mag') and (b[:2] not in ('w1', 'w2', 'w3', 'w4')) and ('Psf_mag' not in b)
            ]

        # Time vector to mjd
        time = Time(self.light_curve.lc['mjd'], format='mjd')

        # Update kwargs with default parameters if not already provided
        default_params = {
            'ecolor': 'k',
            'lw': 0.5,
            'capsize': 2.0,
            'fmt': 'o'
        }
        for key, value in default_params.items():
            kwargs.setdefault(key, value)

        # Iterate through bands and plot
        for band in bands:
            if not np.all(self.light_curve.lc[band].mask):  # make sure everything is not nan
                pos_err_mask = self.light_curve.lc[f'{band}err'].filled(fill_value=np.nan) > 0
                ax.errorbar(
                    x=time.mjd[pos_err_mask],
                    y=self.light_curve.lc[band].filled(fill_value=np.nan)[pos_err_mask],
                    yerr=self.light_curve.lc[f'{band}err'].filled(fill_value=np.nan)[pos_err_mask],
                    marker=LC_MARKER_INFO[ALL_BAND_DF.loc['survey', band]],
                    color=LC_COLOR_INFO[ALL_BAND_DF.loc['band', band]],
                    markeredgewidth=2 if ALL_BAND_DF.loc['survey', band] == 'panstarrs' else 1,
                    zorder=1,
                    **kwargs,
                )

        # Format
        ax.invert_yaxis()
        ax.set_ylabel('Mag')
        ax.set_xlabel('Time [mjd]')

        # Add WISE mags if requested
        if include_wise:

            # Create axis and label it
            wise_ax = ax.twinx()
            wise_ax.set_ylabel('WISE Mag')

            # Plot W1 and W2
            if 'w1_mag' in self.light_curve.lc.columns:
                wise_ax.errorbar(
                    x=time.mjd,
                    y=self.light_curve.lc['w1_mag'].filled(fill_value=np.nan),
                    yerr=self.light_curve.lc[f'w1_magerr'].filled(fill_value=np.nan),
                    marker='*',
                    color='saddlebrown',
                    zorder=1,
                    **kwargs,
                )
            if 'w2_mag' in self.light_curve.lc.columns:
                wise_ax.errorbar(
                    x=time.mjd,
                    y=self.light_curve.lc['w2_mag'].filled(fill_value=np.nan),
                    yerr=self.light_curve.lc['w2_magerr'].filled(fill_value=np.nan),
                    marker='*',
                    color='sandybrown',
                    zorder=1,
                    **kwargs,
                )

            # Make the handles for a legend
            w1_handle = mlines.Line2D(
                [],
                [],
                marker='*',
                color='saddlebrown',
                linestyle='None',
                markersize=8,
                label='W1'
            )
            w2_handle = mlines.Line2D(
                [],
                [],
                marker='*',
                color='sandybrown',
                linestyle='None',
                markersize=8,
                label='W2'
            )
            wise_ax.invert_yaxis()

        # Create legend handles for markers
        marker_handles = []
        for label, marker in LC_MARKER_INFO.items():
            # We use a dummy black marker (or any color you prefer) to represent the marker type.
            handle = mlines.Line2D([], [], marker=marker, color='gray', linestyle='None', label=label,
                                   markeredgewidth=2 if marker == '3' else 1)
            marker_handles.append(handle)

        # Decide on which axes to add the legends for markers and colors.
        legend_ax = wise_ax if include_wise else ax

        # Create the first legend for markers and add it to the axis
        legend_markers = legend_ax.legend(
            handles=marker_handles,
            loc='upper right',
            framealpha=0.8,
            handletextpad=0.3,
            columnspacing=0.85,
            ncols=3,
        )
        legend_ax.add_artist(legend_markers).set_zorder(10)

        # Create legend handles for colors
        color_handles = []
        for label, color in LC_COLOR_INFO.items():
            # Use a patch to show the color
            handle = mpatches.Patch(color=color, label=label)
            color_handles.append(handle)

        # Create the second legend for colors and add it to the axis
        legend_colors = legend_ax.legend(
            handles=color_handles,
            ncols=4,
            loc='upper left',
            framealpha=0.8,
            columnspacing=0.85,
            handlelength=0.8,
            handletextpad=0.3,
        )
        legend_ax.add_artist(legend_colors).set_zorder(10)

        # If requested, make time into date strings
        if time_as_str:
            ticks_as_time = Time(ax.get_xticks(), format='mjd')
            ax.set_xticks(
                ticks_as_time.mjd,
                ticks_as_time.strftime('%m-%d-%Y'),
                **xlab_kwags,
            )
    
        # Add wise legend and adjust y bounds a little
        if include_wise:
            # Get x anchor
            renderer = fig.canvas.get_renderer()
            bbox_disp = legend_colors.get_window_extent(renderer=renderer)
            bbox_axes = legend_ax.transAxes.inverted().transform(bbox_disp)
            x_anchor = bbox_axes[1, 0]  # the right edge (x1) of the first legend in axes coordinates

            # Make the legend and add it to the axes
            wise_legend = legend_ax.legend(
                handles=[w1_handle, w2_handle],
                loc='upper left',
                bbox_to_anchor=(x_anchor, 1),
                framealpha=0.8,
                handletextpad=0.3,
            )
            legend_ax.add_artist(wise_legend).set_zorder(10)

            # Increase ylim a little for the wise legend
            wise_ylim = wise_ax.get_ylim()
            wise_ax.set_ylim((
                wise_ylim[0],
                wise_ylim[0] - 1.1 * (wise_ylim[0] - wise_ylim[1]),
            ))

        # Increase ylim a little for the legends
        ylim = ax.get_ylim()
        ax.set_ylim((
            ylim[0],
            ylim[0] - 1.1 * (ylim[0] - ylim[1]),
        ))

        if include_wise:
            return ax, wise_ax
        return ax

    def plot_wise_mag_hist(self, ax: Optional[Axes] = None, snr_thresh: float = 0.0, **kwargs) -> Axes:
        """Plot the distribution of W1 - W2 magnitudes. Sources with W1 - W2 > 0.8 are AGN according to
        https://iopscience.iop.org/article/10.1088/0004-637X/753/1/30/pdf
        """
        if ax is None:
            _, ax = plt.subplots()

        # Annotate if no data
        if 'w1_snr' not in self.light_curve.lc.columns or 'w2_snr' not in self.light_curve.lc.columns:
            ax.text(0.5, 0.5, 'Source has no WISE data.', verticalalignment='center', horizontalalignment='center')
            return ax

        # Mask for SNR minimum
        snr_mask = np.logical_and(self.light_curve.lc['w1_snr'] > snr_thresh, self.light_curve.lc['w2_snr'] > snr_thresh)

        # Annotate if no sufficient SNR
        if np.sum(snr_mask) == 0:
            ax.text(
                0.5,
                0.5,
                f'Source has no WISE data\nwith SNR $>$ {snr_thresh}.',
                verticalalignment='center',
                horizontalalignment='center',
            )
            return ax

        # Get the delta mags
        delta_mag = self.light_curve.lc['w1_mag'] - self.light_curve.lc['w2_mag']
        delta_mag = delta_mag[snr_mask]

        # Plot
        ax.hist(delta_mag, color='k', bins=10)

        # Plot summary stats and the 0.8 criterion
        ax.axvline(0.8, color='red', lw=0.75, label=r'$\rm{W}1 - \rm{W}2 = 0.8$')
        mean_dmag, median_dmag = np.mean(delta_mag), np.median(delta_mag)
        ax.axvline(mean_dmag, label=f'Mean ({mean_dmag:.2f})', color='green')
        ax.axvline(median_dmag, label=f'Median ({median_dmag:.2f})', color='green', linestyle='--')

        # Format
        ax.set_xlabel('W1 - W2')
        ax.set_ylabel('Number')
        ax.legend()

        return ax

    def plot_spectrum(self, ax: Optional[Axes] = None) -> Axes:
        """Plot the spectrum from SDSS."""
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 5))
        
        # Annotate if no spectrum
        if self.spectrum is None:
            ax.text(0.5, 0.5, 'Source has no spectrum from SDSS.', horizontalalignment='center', verticalalignment='center')
            return ax

        # Label axes
        ax.set_xlabel(r'log(Wavelength [$\rm{\AA}$])')
        ax.set_ylabel(r'$F_\lambda \ [10^{-17} \ \rm{erg} \ \rm{cm}^{-2} \ \rm{s}^{-1} \ \rm{\AA}^{-1}]$')

        # If no spectrum just return ax
        if self.spectrum is None:
            print(f'No spectrum for source at ({self.coord.ra.deg}, {self.coord.dec.deg}). Skipping plotting...')
            return ax

        # Plot
        ax.plot(self.spectrum[0][1].data['loglam'], self.spectrum[0][1].data['flux'], color='gray', label='Data')
        ax.plot(self.spectrum[0][1].data['loglam'], self.spectrum[0][1].data['model'], color='k', label='Model')

        # Add legend
        ax.legend(loc='upper right')

        return ax

    def get_TNS_info(self, tns_df: Optional[pd.DataFrame] = None, tns_coords: Optional[SkyCoord] = None) -> bool:
        # Load TNS if it is not given
        if tns_df is None:
            tns_df = pd.read_csv(os.path.join(get_data_path(), 'tns_public_objects.csv'))
        if tns_coords is None:
            tns_coords = SkyCoord(tns_df['ra'], tns_df['declination'], unit='deg')

        # Get info from TNS
        idx, sep2d, _ = match_coordinates_sky(self.coord, tns_coords)
        if sep2d.arcsec > self.max_arcsec:
            return None
        return tns_df.iloc[[idx]]

    def _get_GAIA_info(self, max_arcsec: float):
        return Gaia.query_object_async(
            coordinate=self.coord,
            radius=max_arcsec * u.arcsec,
        )

    @property
    def GAIA_info(self) -> Table:
        if self._GAIA_info is None:
            self._GAIA_info = self._get_GAIA_info(self.gaia_max_arcsec)

        return self._GAIA_info

    def get_info_string(self, wise_snr_thresh: float = 3.0) -> str:
        """Get string with all the necessary source information."""
        info_string = r'\textbf{Source Information:}' f'\nCoordinates: ({self.ra:.5f}, {self.dec:.5f})'
        tns_info = self.get_TNS_info()
        if tns_info is None:
            info_string += '\nSource not in TNS.'
        else:
            tns_info = self.get_TNS_info().iloc[0]
            info_string += (
                f'\nTNS Name: {tns_info["name_prefix"]} {tns_info["name"]}'
                f'\nTNS Discovery date: {tns_info["discoverydate"]}'
                f'\nTNS Reporter: {tns_info["reporting_group"]}'
                f'\nTNS Type: {tns_info["type"]}'
            )
        if self.spectrum is None:
            info_string += '\nNo SDSS Source Classification.'
        else:
            info_string += f'\nSDSS Class: {self.spectrum[0][2].data["CLASS"]}'
        if 'w1_snr' in self.light_curve.lc.columns and 'w2_snr' in self.light_curve.lc.columns:
            snr_mask = np.logical_and(
                self.light_curve.lc['w1_snr'] > wise_snr_thresh,
                self.light_curve.lc['w2_snr'] > wise_snr_thresh,
            )
            if np.sum(snr_mask) > 0:
                delta_mag = self.light_curve.lc['w1_mag'] - self.light_curve.lc['w2_mag']
                delta_mag = delta_mag[snr_mask]
                info_string += (
                    f'\nWISE W1-W2 Mean (SNR $>$ {wise_snr_thresh}) $=$ {np.mean(delta_mag):.2f}'
                    f'\nWISE W1-W2 Median (SNR $>$ {wise_snr_thresh}) $=$ {np.median(delta_mag):.2f}'
                )
        else:
            info_string += '\nNo WISE W1-W2 data.'
        
        return info_string

    def plot_everything(self) -> Axes:
        """Function that plots everything on one page!"""
        # Set up the layout
        fig = plt.figure(figsize=(12, 18))
        ax0 = plt.subplot2grid((5, 3), (0, 0))
        ax1 = plt.subplot2grid((5, 3), (0, 1))
        ax2 = plt.subplot2grid((5, 3), (0, 2))
        ax3 = plt.subplot2grid((5, 3), (1, 0))
        ax4 = plt.subplot2grid((5, 3), (1, 1))
        ax5 = plt.subplot2grid((5, 3), (1, 2))
        lc_ax = plt.subplot2grid((5, 3), (2, 0), colspan=3)
        spec_ax = plt.subplot2grid((5, 3), (3, 0), colspan=3)
        wise_ax = plt.subplot2grid((5, 3), (4, 2))
        cutout_axes = np.array([[ax0, ax1, ax2], [ax3, ax4, ax5]])
        axes = np.array([cutout_axes, lc_ax, spec_ax, wise_ax], dtype=object)

        # Axis for text stuff
        text_ax = plt.subplot2grid((5, 3), (4, 0), colspan=2)

        # Plot
        snr_thresh = 3.0  # snr min for our WISE observations
        self.plot_all_cutouts(axes=cutout_axes)
        self.plot_lc(ax=lc_ax, fig=fig, xlab_kwags={})
        self.plot_spectrum(ax=spec_ax)
        self.plot_wise_mag_hist(ax=wise_ax, snr_thresh=snr_thresh)

        # Annotate text info at the bottom
        info_string = self.get_info_string(wise_snr_thresh=snr_thresh)
        text_ax.text(0, 0.95, s=info_string, verticalalignment='top', fontsize='large')
        text_ax.axis('off')

        return axes


class Sources:
    """Collection of the Source class."""
    def __init__(
            self,
            ras: Optional[Iterable[float]] = None,
            decs: Optional[Iterable[float]] = None,
            sources: Optional[List[Source]] = None,
            catch_plotting_exceptions: bool = True,
            **kwargs,
        ):
        if sources is not None:
            # Initialize from an existing list of Source objects
            self.sources = sources
            self.ras = np.array([src.ra for src in self.sources], dtype=float)
            self.decs = np.array([src.dec for src in self.sources], dtype=float)
        else:
            # Initialize from ras and decs
            self.ras = np.array(ras, dtype=float)
            self.decs = np.array(decs, dtype=float)
            self.sources = [
                Source(ra, dec, **kwargs) for ra, dec in zip(self.ras, self.decs)
            ]

        self._data = None
        self._coords = None

    @property
    def coords(self) -> SkyCoord:
        if self._coords is None or len(self._coords) != self.__len__():
            self._coords = SkyCoord([s.coord for s in self.sources])
        return self._coords

    def __iter__(self):
        return iter(self.sources)

    def __len__(self):
        return len(self.sources)

    def __str__(self):
        return str(self.data)

    def __add__(self, other):
        if not isinstance(other, Sources):
            raise TypeError("Can only add Sources to Sources.")
        return Sources(sources=self.sources + other.sources)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.sources[index]

        if isinstance(index, slice):
            # Return a new Sources object containing a slice of the existing sources
            return Sources(sources=self.sources[index])

        if isinstance(index, (list, np.ndarray)):
            index = np.asarray(index)
            if index.dtype == bool or np.issubdtype(index.dtype, np.integer):
                # Return a new Sources object containing a subset of the existing sources
                return Sources(sources=np.array(self.sources)[index].tolist())

            raise TypeError("Array-based indexing must be boolean or integer indices.")

        raise TypeError("Invalid index type. Must be int, slice, or boolean/integer array.")

    @property
    def data(self) -> Table:
        if self._data is None:
            if len(self.sources) == 0:
                return Table()
            self._data = vstack([src.data for src in self.sources])
            self._data = Table(self._data, masked=False)

        return self._data

    def save(self, fname: str, overwrite: bool = True):
        to_save = self.data.copy()
        to_save['filter_info'] = [str(src.filter_info) for src in self.sources]
        if len(to_save) == 0:
            to_save['ra'] = []
            to_save['dec'] = []
        to_save.write(fname, format='ascii.ecsv', overwrite=overwrite)

    def submit_forced_photometry_batch(self) -> int:
        # Load important objects
        ztf_fp_service = ZTFFP_Service()
        ztf_fp_map = Forced_Photo_Map()

        # Boolean masks for filtering
        already_downloaded = ztf_fp_map.contains(self.ras, self.decs)
        recently_queried = ztf_fp_service.recently_queried(self.ras, self.decs)
        currently_pending = ztf_fp_service.currently_pending(self.ras, self.decs)

        to_submit_mask = (not already_downloaded) & (not recently_queried) & (not currently_pending)
        ras_to_submit, decs_to_submit = self.ras[to_submit_mask], self.decs[to_submit_mask]

        print(f'Submitting forced photometry request on {len(ras_to_submit)} source(s). '
              f'{np.sum(not to_submit_mask)} of the given coordinates were already downloaded or requested.')

        return ztf_fp_service.submit(ras_to_submit, decs_to_submit)

    def inTNS(self):
        # Load TNS if it is not given
        tns_df = pd.read_csv(os.path.join(get_data_path(), 'tns_public_objects.csv'))
        tns_coords = SkyCoord(tns_df['ra'], tns_df['declination'], unit='deg')

        return np.array(
            [src.get_TNS_info(tns_df=tns_df, tns_coords=tns_coords) is not None for src in self.sources],
            dtype=bool,
        )
