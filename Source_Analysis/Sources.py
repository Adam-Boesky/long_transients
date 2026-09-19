import io
import os
import sys
import ast
import json
import requests
from glob import glob
import warnings
import traceback
import time
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from concurrent import futures
from functools import lru_cache
from typing import Any, List, Tuple, Union, Optional, Iterable, Dict
from matplotlib.axes._axes import Axes

from astropy.wcs import WCS
from astropy.time import Time
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.visualization import simple_norm, time_support
from astropy.io import fits
from astropy.io.fits import HDUList
from astroquery.gaia import Gaia
from sparcl.client import SparclClient

sys.path.append('/Users/adamboesky/Research/long_transients')

from Extracting.utils import get_data_path, load_cached_table, load_ecsv, get_snr_from_mag, prepare_table_for_write, _INT64_COLUMNS
from Extracting.Catalogs import ZTF_Catalog, ZTF_CUTOUT_HALFWIDTH, get_ztf_metadata_from_coords, get_ztf_metadata_from_metadata, get_pstarr_cutout
from ztf_fp_query.forced_photometry import ForcedPhotometry
try:
    from Light_Curve import Light_Curve, LC_MARKER_INFO, LC_COLOR_INFO, ALL_BAND_DF
except ModuleNotFoundError:
    from .Light_Curve import Light_Curve, LC_MARKER_INFO, LC_COLOR_INFO, ALL_BAND_DF
from Source_Analysis.catalogs.agn import cone_search as agn_cone_search, CATALOG_ID_TO_NAME
from Source_Analysis.catalogs import sdss as sdss_catalog, simbad as simbad_catalog, tns as tns_catalog
time_support()

_get_sparcl_client = lru_cache(maxsize=1)(lambda: SparclClient(connect_timeout=10))

ACCEPTABLE_PROC_STATUS = [0]
MANDATORY_SOURCE_COLUMNS = [
    'ra', 'dec', 'PSTARR_rPSFMag', 'PSTARR_iKronMagErr', 'PSTARR_primaryDetection',
    'PSTARR_rinfoFlag2', 'PSTARR_gPSFMagErr',
    'PSTARR_gKronMag', 'PSTARR_gKronMagErr', 'PSTARR_iinfoFlag2', 'PSTARR_dec', 'PSTARR_rKronMagErr',
    'PSTARR_ginfoFlag2', 'PSTARR_rKronMag', 'PSTARR_iPSFMagErr',
    'PSTARR_ra', 'PSTARR_iPSFMag', 'PSTARR_PanSTARR_ID', 'PSTARR_gPSFMag', 'PSTARR_rPSFMagErr',
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
    'ZTF_g_xcpeak', 'ZTF_g_ymax', 'ZTF_i_tnpix', 'ZTF_gKronFlag',
    'g_x', 'g_y', 'r_x', 'r_y', 'i_x', 'i_y',
    'g_association_separation_arcsec', 'r_association_separation_arcsec', 'i_association_separation_arcsec',
    'g_Catalog_Flag', 'r_Catalog_Flag', 'i_Catalog_Flag',
    'g_Catalog', 'r_Catalog', 'i_Catalog',
    'filter_info', 'ZTF_g_field', 'ZTF_g_ccdid', 'ZTF_g_qid', 'ZTF_r_field', 'ZTF_r_ccdid',
    'ZTF_r_qid', 'ZTF_i_field', 'ZTF_i_ccdid', 'ZTF_i_qid',
]
Gaia.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'
CATALOG_INT_MAP = {'in_both': 0, 'in_ztf': 1, 'in_pstarr': 2}


def set_mpl_params(font_size: int = 12):
    """Set the matplotlib parameters."""
    plt.rc('text', usetex=True)
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'cmr10'
    mpl.rcParams['font.size'] = font_size
    mpl.rcParams['axes.formatter.use_mathtext'] = True


def latex_escape(s: str) -> str:
    """Escape a plain string for literal display under matplotlib's usetex renderer.

    LaTeX treats `\\`, `_`, `{`, `}`, `&`, `%`, and `#` as special (e.g. `_` triggers a
    subscript, `{`/`}` are grouping delimiters that render invisibly) -- needed for any
    data-derived string (catalog names, class labels, ...) embedded in a plotted label.
    """
    s = s.replace('\\', r'\textbackslash{}')
    for char in ('_', '{', '}', '&', '%', '#'):
        s = s.replace(char, f'\\{char}')
    return s


def format_magerr(err: float) -> str:
    """Format a magnitude error for a LaTeX ``\\pm`` label.

    Uses ``.2f`` for errors that round to something nonzero; switches to
    scientific notation (e.g. ``2.0\\times10^{-3}``) for errors small enough
    that ``.2f`` would otherwise display as a misleading ``0.00``.
    """
    if not np.isfinite(err) or err <= 0 or round(err, 2) != 0:
        return f'{err:.2f}'
    exponent = int(np.floor(np.log10(err)))
    mantissa = err / 10 ** exponent
    return rf'{mantissa:.1f}\times10^{{{exponent}}}'


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
            scale_bar_arcsec: float = 10.0,
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

        # Draw scale bar in the top-right corner
        if self.arcsec_per_pixel is not None:
            img_h, img_w = self.images[band].shape
            bar_pixels = scale_bar_arcsec / self.arcsec_per_pixel
            pad_x = img_w * 0.03
            pad_y = img_h * 0.01
            x_bar = img_w - 1 - pad_x
            y_top = img_h - 1 - pad_y
            y_bottom = y_top - bar_pixels
            ax.plot([x_bar, x_bar], [y_bottom, y_top], color='black', lw=2, solid_capstyle='butt')
            label = f'{scale_bar_arcsec:.0f}"' if scale_bar_arcsec >= 1 else f'{scale_bar_arcsec * 60:.0f}\''
            ax.text(
                x_bar - img_w * 0.02,
                (y_bottom + y_top) / 2,
                label,
                color='black',
                ha='right',
                va='center',
                fontsize=9,
            )

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

                # Set empty cutouts to None
                if self._images[band].size == 0:
                    self._images[band] = None

        return self._images, self._WCSs


class PSTARR_Postage_Stamp(Postage_Stamp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arcsec_per_pixel = 0.25

    def get_images(self) -> Tuple[Dict[str, np.ndarray], Dict[str, WCS]]:

        # Get the images from the ZTF cutouts
        images = {}
        wcss = {}
        def get_cutout_with_retries(band, max_attempts=3):
            for attempt in range(max_attempts):
                try:
                    return band, get_pstarr_cutout(
                        self.ra,
                        self.dec,
                        size=self.stamp_width_arcsec / self.arcsec_per_pixel,
                        filter=band
                    )
                except Exception as e:
                    print(f"Attempt {attempt+1} failed for band {band} in get_cutout_with_retries:")
                    traceback.print_exc()
                    if attempt == max_attempts - 1:
                        print(f"All {max_attempts} attempts failed for band {band}. Raising exception.")
                        raise
                    else:
                        continue

        with futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(
                get_cutout_with_retries,
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
            # 1.5" matches ZTF's own internal match radius for grouping per-epoch
            # detections into its Objects/lightcurve catalog (ZTF Explanatory
            # Supplement), and Light_Curve's own query_rad_arcsec default.
            max_arcsec: float = 1.5,
            gaia_max_arcsec: float = 5.0,
            verbose: int = 1,
            catch_plotting_exceptions: bool = True,
            lc_catalogs: List[str] = ['ztf', 'wise', 'neowise', 'ptf', 'sdss', 'panstarrs', 'gaia', 'custom'],
            ztf_lc_dir: Optional[str] = None,
            detected_bands: Optional[tuple] = None,
            filtering_dirpath: Optional[str] = None,
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
        self.ztf_lc_dir = ztf_lc_dir  # directory of local ZTF parquet LC files (if None, uses API)
        self.lc_catalogs = lc_catalogs

        # The maximum distance for an object to be considered a match
        # We will use a different value for our GAIA queries because sources are likely in motion or have a considerable
        # parallax
        self.max_arcsec = max_arcsec
        self.gaia_max_arcsec = gaia_max_arcsec
        self.filtering_dirpath = filtering_dirpath

        # Properties
        self._field_catalogs = field_catalogs
        self._precomputed_cat_indices = None  # {band: ind_closest} pre-set by Sources for bulk efficiency
        self._data = None
        self._postage_stamps = None
        self._ztf_lightcurve = None
        self._paddedfield = None
        self._GAIA_info = None
        self.filter_info = {}
        self._image_metadata = None
        self._spectrum = None
        self._has_spectrum = True
        self._desi_spectrum = None
        self._has_desi_spectrum = True
        self._agn_match = None
        self._has_agn_match = True
        self._tns_match = None
        self._has_tns_match = True
        self._simbad_match = None
        self._has_simbad_match = True
        self._light_curve = None

        # If detected_bands is provided at construction time, pre-populate the in_* cache
        # so the lazy data lookup is bypassed entirely.
        if detected_bands is not None:
            self._in_bands = list(detected_bands)
            self._in_g = 'g' in detected_bands
            self._in_r = 'r' in detected_bands
            self._in_i = 'i' in detected_bands
        else:
            self._in_bands = None

    @property
    def light_curve(self) -> Light_Curve:
        """The lightcurve class for the source"""
        if self._light_curve is None:
            self._light_curve = Light_Curve(
                self.ra,
                self.dec,
                query_rad_arcsec=self.max_arcsec,
                catalogs=self.lc_catalogs,
                pstarr_objid=self.data['PSTARR_PanSTARR_ID'][0],
                pstarr_coord=(
                    self.data['PSTARR_ra'][0],
                    self.data['PSTARR_dec'][0],
                ) if not np.any(
                    np.isnan((
                        self.data['PSTARR_ra'][0],
                        self.data['PSTARR_dec'][0],
                    ))
                ) else None,
                ztf_local_dir=self.ztf_lc_dir,
            )

        return self._light_curve

    @property
    def spectrum(self) -> Union[List[HDUList], None]:
        """The SDSS DR17 spectrum as a one-element list of HDULists, or None.

        The match is found in the local specObj catalog and the spectrum is
        pulled straight from the Science Archive Server, rather than going
        through astroquery's SkyServer endpoint. Same data release and the same
        file -- verified bit-identical in HDU1 (loglam/flux/model) and HDU2
        (CLASS/SUBCLASS/Z) -- but SkyServer's query interface is frequently
        unreachable while the SAS is not.
        """
        if self._has_spectrum and self._spectrum is None:
            match = sdss_catalog.cone_search(self.ra, self.dec, radius_arcsec=self.max_arcsec)
            if match.empty:
                print(f'Source at ({self.ra}, {self.dec}) has no spectrum in SDSS.')
                self._has_spectrum = False
            else:
                url = sdss_catalog.spectrum_url(match.iloc[0])
                print(f'Getting source spectrum from SDSS ({url.rsplit("/", 1)[-1]})...')
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                self._spectrum = [fits.open(io.BytesIO(resp.content))]

        return self._spectrum

    @property
    def desi_spectrum(self) -> Optional[Any]:
        """The source spectrum record from DESI DR1 (via SPARCL), or None if there's no match."""
        if self._has_desi_spectrum and self._desi_spectrum is None:
            print('Getting source spectrum from DESI DR1 (SPARCL)...')
            client = _get_sparcl_client()

            # SPARCL's find() only supports range constraints (no radius/cone search), so
            # box around the source and post-filter by true angular separation.
            dra = (self.max_arcsec * u.arcsec).to(u.deg).value / np.cos(np.radians(self.dec))
            ddec = (self.max_arcsec * u.arcsec).to(u.deg).value
            cons = {
                'ra': [self.ra - dra, self.ra + dra],
                'dec': [self.dec - ddec, self.dec + ddec],
                'data_release': ['DESI-DR1'],
            }
            found = client.find(outfields=['sparcl_id', 'ra', 'dec', 'spectype', 'redshift'], constraints=cons)

            if len(found.records) == 0:
                print(f'Source at ({self.ra}, {self.dec}) has no spectrum in DESI DR1.')
                self._has_desi_spectrum = False
            else:
                cat_coord = SkyCoord(
                    ra=[r.ra for r in found.records],
                    dec=[r.dec for r in found.records],
                    unit='deg',
                )
                seps = self.coord.separation(cat_coord)
                best = int(np.argmin(seps.arcsec))
                if seps.arcsec[best] > self.max_arcsec:
                    print(f'Source at ({self.ra}, {self.dec}) has no spectrum in DESI DR1.')
                    self._has_desi_spectrum = False
                else:
                    retrieved = client.retrieve(
                        uuid_list=[found.records[best].sparcl_id],
                        include=['wavelength', 'flux', 'model', 'spectype', 'redshift'],
                    )
                    self._desi_spectrum = retrieved.records[0]

        return self._desi_spectrum

    @property
    def agn_match(self) -> Optional[pd.Series]:
        """Closest AGN-DB catalog match within max_arcsec, or None."""
        if self._has_agn_match and self._agn_match is None:
            matches = agn_cone_search(
                self.ra, self.dec, radius_arcsec=self.max_arcsec,
                columns=[
                    'best_class_all', 'best_class_origin', 'best_class_sub_all',
                    'best_Z_merged', 'star_flag',
                ],
            )
            if matches.empty:
                self._has_agn_match = False
            else:
                self._agn_match = matches.iloc[0]

        return self._agn_match

    @property
    def simbad_match(self) -> Optional[pd.Series]:
        """Closest SIMBAD object within max_arcsec, or None.

        Columns keep the prefixed names `catalogs.simbad` returns
        (`simbad_main_id`, `simbad_otype`, ...), matching the crossmatch columns
        `enrich_combined_tabs.py` writes for the same catalog.
        """
        if self._has_simbad_match and self._simbad_match is None:
            matches = simbad_catalog.cone_search(
                self.ra, self.dec, radius_arcsec=self.max_arcsec,
                columns=[
                    'simbad_main_id', 'simbad_otype', 'simbad_otype_label',
                    'simbad_otype_path', 'simbad_is_candidate', 'simbad_z',
                    'simbad_nbref',
                ],
            )
            if matches.empty:
                self._has_simbad_match = False
            else:
                self._simbad_match = matches.iloc[0]

        return self._simbad_match

    @property
    def image_metadata(self) -> Dict[str, Dict]:
        if self._image_metadata is None and self._data is not None:

            # Iterate through bands and get the first metadata that works
            metadata = {}
            field_vals = np.array([self.data[k][0] for k in self.data.columns if 'field' in k], dtype=float)
            ccd_vals = np.array([self.data[k][0] for k in self.data.columns if 'ccd' in k], dtype=float)
            qid_vals = np.array([self.data[k][0] for k in self.data.columns if 'qid' in k], dtype=float)

            # Drop nans
            field_vals = field_vals[~np.isnan(field_vals)]
            ccd_vals = ccd_vals[~np.isnan(ccd_vals)]
            qid_vals = qid_vals[~np.isnan(qid_vals)]

            # If there are more than one, just grab the first
            if np.sum(~np.isnan(field_vals)) > 0:
                metadata['fieldid'] = field_vals[0]
            if np.sum(~np.isnan(ccd_vals)) > 0:
                metadata['ccdid'] = ccd_vals[0]
            if np.sum(~np.isnan(qid_vals)) > 0:
                metadata['qid'] = qid_vals[0]

            # Cast to ints
            metadata['fieldid'] = int(metadata['fieldid'])
            metadata['ccdid'] = int(metadata['ccdid'])
            metadata['qid'] = int(metadata['qid'])

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
            # The IRSA coordinate query returns 'field'; the metadata query returns 'fieldid'.
            field_col = 'fieldid' if 'fieldid' in self._image_metadata.columns else 'field'
            inds_extracted = []
            test_paths = []
            for ind, field in self._image_metadata[field_col].items():
                field_id = str(field).zfill(6)
                if os.path.exists(os.path.join(self.merged_field_basedir, f'{field_id}_g.hdf5')):
                    inds_extracted.append(ind)
                    test_paths.append(os.path.join(self.merged_field_basedir, f'{field_id}_g.hdf5'))
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

            # Rename field to fieldid
            if 'field' in self._image_metadata:
                self._image_metadata['fieldid'] = self._image_metadata['field']
                del self._image_metadata['field']

        return self._image_metadata

    @property
    def field_catalogs(self) -> dict[str, Table]:
        """Dictionary containing the catalogs for each band in the ZTF field that contains the current source."""
        if self._field_catalogs is None:
            print('Loading catalogs!')
            self._field_catalogs = {}

            def load_catalog(band):
                padded_field = str(int(self.image_metadata["fieldid"])).zfill(6)
                print(f'Loading {band} catalog from locally stored catalog {padded_field}_{band}...')
                return band, load_cached_table(os.path.join(self.merged_field_basedir, f'{padded_field}_{band}.hdf5')).copy()

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
                    elif cname in ('Catalog_Flag', 'Catalog', 'association_separation_arcsec', 'x', 'y') \
                            and not cname.startswith(f'{band}_'):
                        # Per-band columns that need a band prefix to avoid collision across bands
                        new_cname = f'{band}_{cname}'
                        self.field_catalogs[band].rename_column(cname, new_cname)
                        unique_colnames.append(new_cname)
                    else:
                        unique_colnames.append(cname)

            # Add the mandatory columns and make sure they're unique
            unique_colnames += MANDATORY_SOURCE_COLUMNS
            unique_colnames = set(unique_colnames)

            # Make the empty table to fill in
            data_dict = {k: [np.nan] for k in unique_colnames}
            str_cols = ['g_Catalog', 'r_Catalog', 'i_Catalog']  # need to have types align
            for col in str_cols:
                data_dict[col] = [str(data_dict[col][0])]
            self._data = Table(data_dict)

            # Cast id to string
            self._data['PSTARR_PanSTARR_ID'] = self._data['PSTARR_PanSTARR_ID'].astype(object)

            # Make sure the per-band catalog columns are long enough strings
            for _cat_col in ('g_Catalog', 'r_Catalog', 'i_Catalog'):
                if _cat_col in self._data.colnames:
                    self._data[_cat_col] = self._data[_cat_col].astype('S10')

            for band, cat in self.field_catalogs.items():

                # Get the source from each catalog
                if self._precomputed_cat_indices is not None and band in self._precomputed_cat_indices:
                    ind_closest = self._precomputed_cat_indices[band]
                else:
                    if self.verbose > 0: print(f'Searching {band} catalog for source...')
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
                        elif self._data[cname][0] is None:
                            entry_is_nan = True
                        elif isinstance(self._data[cname][0], str):
                            entry_is_nan = self._data[cname][0].lower() == 'nan'
                        else:
                            entry_is_nan = np.isnan(self._data[cname][0])
                        if entry_is_nan:
                            if isinstance(cat[cname][ind_closest], bool):
                                self._data[cname][0] = float(cat[cname][ind_closest])
                            else:
                                self._data[cname][0] = cat[cname][ind_closest]

            # Reorder columns once after building
            pstarr_cols = [col for col in self._data.colnames if col.startswith('PSTARR')]
            ztf_cols = [col for col in self._data.colnames if col.startswith('ZTF')]
            other_cols = [col for col in self._data.colnames if not (col.startswith('PSTARR') or col.startswith('ZTF'))]
            ordered_cols = ['ra', 'dec'] + pstarr_cols + ztf_cols + [col for col in other_cols if col not in ['ra', 'dec']]
            self._data = self._data[ordered_cols]

        return self._data

    @data.setter 
    def data(self, data_tab: Table) -> None:
        """Set the data table for this source."""
        if not isinstance(data_tab, Table):
            raise TypeError("data must be an astropy Table")
        if len(data_tab) != 1:
            raise ValueError("data must have exactly one row")
        if len(np.intersect1d(data_tab.colnames, MANDATORY_SOURCE_COLUMNS)) < len(MANDATORY_SOURCE_COLUMNS):
            # Add the mandatory columns if they're not in the table
            warnings.warn("data table does not have all the mandatory columns, adding them with nans.")
            for col in [c for c in MANDATORY_SOURCE_COLUMNS if c not in data_tab.columns]:
                data_tab[col] = [np.nan]

        self._data = Table(data_tab, masked=False)

        # Table(masked=False) fills masked ints with the column fill_value (e.g. -1) rather
        # than None, so integer ID columns that were masked on load need a second pass.
        for colname in _INT64_COLUMNS:
            if colname in self._data.colnames:
                if np.ma.is_masked(data_tab[colname][0]) or data_tab[colname][0] is None:
                    self._data[colname] = np.array([None], dtype=object)

    @property
    def in_bands(self) -> List[str]:
        """The bands that the source was extracted/found in."""
        if self._in_bands is None:
            self._in_bands = [band for band in self.bands if ~np.isnan(self.data[f'ZTF_{band}_ra'])]

        return self._in_bands

    @property
    def in_g(self) -> bool:
        if not hasattr(self, '_in_g'):
            self._in_g = ~np.isnan(self.data['ZTF_g_ra'])[0]
        return self._in_g

    @property
    def in_r(self) -> bool:
        if not hasattr(self, '_in_r'):
            self._in_r = ~np.isnan(self.data['ZTF_r_ra'])[0]
        return self._in_r

    @property
    def in_i(self) -> bool:
        if not hasattr(self, '_in_i'):
            self._in_i = ~np.isnan(self.data['ZTF_i_ra'])[0]
        return self._in_i

    @property
    def postage_stamps(self) -> Dict[str, Postage_Stamp]:
        if self._postage_stamps is None:
            self._postage_stamps = {
                'ZTF': ZTF_Postage_Stamp(self.ra, self.dec, bands=self.cutout_bands, image_metadata=self.image_metadata, ztf_data_dir=self.ztf_data_dir),
                'PSTARR': PSTARR_Postage_Stamp(self.ra, self.dec, bands=self.cutout_bands),
            }

        return self._postage_stamps

    @property
    def ztf_fp(self) -> Optional['ForcedPhotometry']:
        """The ForcedPhotometry object for this source, or None if not yet synced."""
        if self._ztf_lightcurve is None:
            try:
                self._ztf_lightcurve = ForcedPhotometry(self.ra, self.dec)
            except FileNotFoundError:
                print(f'No lightcurve found for source with ra, dec = ({self.ra}, {self.dec}).')
        return self._ztf_lightcurve

    @property
    def ztf_lightcurve(self) -> Optional[pd.DataFrame]:
        """The forced-photometry DataFrame (mag, magerr, upperlim already included)."""
        fp = self.ztf_fp
        return fp.df if fp is not None else None

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

        # Get the mag strings based on catalog flag
        if np.isnan(self.data[f'PSTARR_{band}PSFMag'][0]):
            pstarr_mag_str = 'ND'
            pstarr_kron_mag_str = ''
        else:
            pstarr_mag_str = rf'PSF: ${self.data[f"PSTARR_{band}PSFMag"][0]:.2f} \pm {format_magerr(self.data[f"PSTARR_{band}PSFMagErr"][0])}$'
            pstarr_snr = get_snr_from_mag(self.data[f'PSTARR_{band}PSFMag'][0], self.data[f'PSTARR_{band}PSFMagErr'][0], zp=25)
            axes[0].text(
                0.99,
                0.10,
                f'PSF SNR$={pstarr_snr:.2f}$',
                transform=axes[0].transAxes,
                ha='right',
                va='bottom',
                fontsize=15,
                color='red'
            )
            pstarr_kron_mag_str = rf'Kron: ${self.data[f"PSTARR_{band}KronMag"][0]:.2f} \pm {format_magerr(self.data[f"PSTARR_{band}KronMagErr"][0])}$'
            pstarr_kron_snr = get_snr_from_mag(self.data[f'PSTARR_{band}KronMag'][0], self.data[f'PSTARR_{band}KronMagErr'][0], zp=25)
            axes[0].text(
                0.99,
                0.01,
                f'Kron SNR$={pstarr_kron_snr:.2f}$',
                transform=axes[0].transAxes,
                ha='right',
                va='bottom',
                fontsize=15,
                color='red'
            )
        if np.isnan(self.data[f'ZTF_{band}PSFMag'][0]):
            ztf_mag_str = 'ND'
            ztf_kron_mag_str = ''
        else:
            ztf_mag_str = rf'PSF: ${self.data[f"ZTF_{band}PSFMag"][0]:.2f} \pm {format_magerr(self.data[f"ZTF_{band}PSFMagErr"][0])}$'
            ztf_snr = get_snr_from_mag(self.data[f'ZTF_{band}PSFMag'][0], self.data[f'ZTF_{band}PSFMagErr'][0], zp=np.nan_to_num(self.data[f'ZTF_{band}_zero_pt_mag'][0], nan=25))
            axes[1].text(
                0.99,
                0.01,
                f'PSF SNR$={ztf_snr:.2f}$',
                transform=axes[1].transAxes,
                ha='right',
                va='bottom',
                fontsize=15,
                color='red'
            )
            ztf_kron_mag_str = rf'Kron: ${self.data[f"ZTF_{band}KronMag"][0]:.2f} \pm {format_magerr(self.data[f"ZTF_{band}KronMagErr"][0])}$'
            ztf_kron_snr = get_snr_from_mag(self.data[f'ZTF_{band}KronMag'][0], self.data[f'ZTF_{band}KronMagErr'][0], zp=np.nan_to_num(self.data[f'ZTF_{band}_zero_pt_mag'][0], nan=25))
            axes[1].text(
                0.99,
                0.10,
                f'Kron SNR$={ztf_kron_snr:.2f}$',
                transform=axes[1].transAxes,
                ha='right',
                va='bottom',
                fontsize=15,
                color='red'
            )

        # Annotate with the mags
        axes[0].text(
            0.01,
            0.99,
            pstarr_mag_str,
            transform=axes[0].transAxes,
            ha='left',
            va='top',
            fontsize=15,
            color='red'
        )
        axes[0].text(
            0.01,
            0.90,
            pstarr_kron_mag_str,
            transform=axes[0].transAxes,
            ha='left',
            va='top',
            fontsize=15,
            color='red'
        )
        axes[1].text(
            0.01,
            0.99,
            ztf_mag_str,
            transform=axes[1].transAxes,
            ha='left',
            va='top',
            fontsize=15,
            color='red'
        )
        axes[1].text(
            0.01,
            0.90,
            ztf_kron_mag_str,
            transform=axes[1].transAxes,
            ha='left',
            va='top',
            fontsize=15,
            color='red'
        )
        for ax in axes:
            ax.text(
                0.02,
                0.02,
                rf'\textbf{{{band}}}',
                transform=ax.transAxes,
                ha='left',
                va='bottom',
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
        for band in bands:

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
            ax_col[1].set_xlabel(rf'\textbf{{{band}}}', fontsize=15)

        # Formatting
        axes[0, 0].set_ylabel(r'\textbf{Pan-STARRS}', fontsize=15)
        axes[1, 0].set_ylabel(r'\textbf{ZTF}', fontsize=15)

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
            include_legend: bool = True,
            time_since_peak: bool = False,
            **kwargs,
        ) -> Axes:
        """Plot lightcurve for all bands specified in 'bands', or all bands if bands is None.

        WISE (W1/W2) is not included here -- see plot_wise_lc for a dedicated panel.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        elif fig is None:
            fig = plt.gcf()

        # Label the axes up front so they're present even if there's no data to plot below.
        ax.set_ylabel(r'\textbf{Mag}')
        ax.set_xlabel(r'\textbf{Time [mjd]}', labelpad=2)

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

        # Handle the time offset
        time = time.mjd
        if time_since_peak:
            all_mags = np.array(self.light_curve.lc[bands])
            all_mags = all_mags.view(np.float64).reshape(all_mags.shape[0], -1)
            peak_time = time[np.nanargmin(np.nanmin(all_mags, axis=1))]
            time = time - peak_time

        # Iterate through bands and plot
        for band in bands:
            if not np.all(self.light_curve.lc[band].mask):  # make sure everything is not nan
                pos_err_mask = self.light_curve.lc[f'{band}err'].filled(fill_value=np.nan) > 0
                ax.errorbar(
                    x=time[pos_err_mask],
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

        # Create legend handles for markers
        lc_colnames = set(self.light_curve.lc.colnames)
        marker_handles = []
        for label, marker in LC_MARKER_INFO.items():
            if label in ('ztforce', 'zubercal'):
                has_data = any(
                    col in lc_colnames and not np.all(self.light_curve.lc[col].mask)
                    for col in ALL_BAND_DF.columns
                    if ALL_BAND_DF.loc['survey', col] == label
                )
                if not has_data:
                    continue
            handle = mlines.Line2D([], [], marker=marker, color='gray', linestyle='None', label=label,
                                   markeredgewidth=2 if marker == '3' else 1)
            marker_handles.append(handle)

        # Create the first legend for markers and add it to the axis
        if include_legend:
            legend_markers = ax.legend(
                handles=marker_handles,
                loc='upper right',
                framealpha=0.8,
                handletextpad=0.3,
                columnspacing=0.85,
                ncols=len(marker_handles),
            )
            ax.add_artist(legend_markers).set_zorder(10)

        # Create legend handles for colors
        color_handles = []
        for label, color in LC_COLOR_INFO.items():
            # Use a patch to show the color
            handle = mpatches.Patch(color=color, label=label)
            color_handles.append(handle)

        # Create the second legend for colors and add it to the axis
        if include_legend:
            legend_colors = ax.legend(
                handles=color_handles,
                ncols=4,
                loc='upper left',
                framealpha=0.8,
                columnspacing=0.85,
                handlelength=0.8,
                handletextpad=0.3,
            )
            ax.add_artist(legend_colors).set_zorder(10)

        # If requested, make time into date strings
        if time_as_str:
            ticks_as_time = Time(ax.get_xticks(), format='mjd')
            ax.set_xticks(
                ticks_as_time.mjd,
                ticks_as_time.strftime('%m-%d-%Y'),
                **xlab_kwags,
            )

        if include_legend:
            # Increase ylim a little for the legends
            ylim = ax.get_ylim()
            ax.set_ylim((
                ylim[0],
                ylim[0] - 1.1 * (ylim[0] - ylim[1]),
            ))

        return ax

    def plot_wise_lc(
            self,
            ax: Optional[Axes] = None,
            fig: Optional[Axes] = None,
            time_as_str: bool = True,
            xlab_kwags: dict = {'rotation': 45, 'ha': 'right'},
            include_legend: bool = True,
            **kwargs,
        ) -> Axes:
        """Plot the WISE (W1/W2) light curve in its own dedicated panel.

        A null magerr means SNR<2 for that epoch, so the quoted magnitude is a
        95%-confidence flux upper limit rather than a real detection (NEOWISE
        Explanatory Supplement, sec2_1c) -- plot those as deemphasized, upside-down
        triangles behind the real detections.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        elif fig is None:
            fig = plt.gcf()

        # Label the axes up front so they're present even if there's no data to plot below.
        ax.set_ylabel(r'\textbf{Mag}', labelpad=2)
        ax.set_xlabel(r'\textbf{Time [mjd]}', labelpad=2)

        if self.light_curve.lc is None:
            ax.text(0.5, 0.5, 'Source has no light curve.', horizontalalignment='center', verticalalignment='center')
            return ax

        if 'w1_mag' not in self.light_curve.lc.columns and 'w2_mag' not in self.light_curve.lc.columns:
            ax.text(0.5, 0.5, 'Source has no WISE data.', horizontalalignment='center', verticalalignment='center')
            return ax

        time = Time(self.light_curve.lc['mjd'], format='mjd').mjd

        default_params = {
            'ecolor': 'k',
            'lw': 0.5,
            'capsize': 2.0,
            'fmt': 'o'
        }
        for key, value in default_params.items():
            kwargs.setdefault(key, value)

        for band, color in (('w1', 'saddlebrown'), ('w2', 'sandybrown')):
            mag_col, magerr_col = f'{band}_mag', f'{band}_magerr'
            if mag_col not in self.light_curve.lc.columns:
                continue
            mag = self.light_curve.lc[mag_col].filled(fill_value=np.nan)
            magerr = self.light_curve.lc[magerr_col].filled(fill_value=np.nan)
            has_mag = ~np.isnan(mag)
            detected_mask = has_mag & (magerr > 0)
            upperlim_mask = has_mag & ~(magerr > 0)

            ax.errorbar(
                x=time[detected_mask],
                y=mag[detected_mask],
                yerr=magerr[detected_mask],
                marker='*',
                color=color,
                zorder=1,
                **kwargs,
            )
            ax.scatter(
                x=time[upperlim_mask],
                y=mag[upperlim_mask],
                marker='v',
                color=color,
                alpha=0.3,
                zorder=0,
            )

        w1_handle = mlines.Line2D([], [], marker='*', color='saddlebrown', linestyle='None', markersize=8, label='W1')
        w2_handle = mlines.Line2D([], [], marker='*', color='sandybrown', linestyle='None', markersize=8, label='W2')

        ax.invert_yaxis()

        if include_legend:
            legend = ax.legend(handles=[w1_handle, w2_handle], loc='upper right', framealpha=0.8, handletextpad=0.3)
            ax.add_artist(legend).set_zorder(10)
            ylim = ax.get_ylim()
            ax.set_ylim((ylim[0], ylim[0] - 1.1 * (ylim[0] - ylim[1])))

        if time_as_str:
            ticks_as_time = Time(ax.get_xticks(), format='mjd')
            ax.set_xticks(
                ticks_as_time.mjd,
                ticks_as_time.strftime('%m-%d-%Y'),
                **xlab_kwags,
            )

        return ax

    def plot_wise_mag_hist(self, ax: Optional[Axes] = None, color_err_thresh: float = 0.5, **kwargs) -> Axes:
        """Plot the distribution of W1 - W2 magnitudes. Sources with W1 - W2 > 0.8 are AGN according to
        https://iopscience.iop.org/article/10.1088/0004-637X/753/1/30/pdf
        """
        if ax is None:
            _, ax = plt.subplots()

        # Label the axes up front so they're present even if there's no data to plot below.
        ax.set_xlabel(r'\textbf{W1 - W2}', labelpad=2)
        ax.set_ylabel(r'\textbf{Number}', labelpad=2)

        # Annotate if no data
        if 'w1_magerr' not in self.light_curve.lc.columns or 'w2_magerr' not in self.light_curve.lc.columns:
            ax.text(0.5, 0.5, 'Source has no WISE data.', verticalalignment='center', horizontalalignment='center')
            return ax

        # Mask on propagated color uncertainty: sigma(W1-W2) = sqrt(sigma_W1^2 + sigma_W2^2)
        w1_err = self.light_curve.lc['w1_magerr'].filled(fill_value=np.nan)
        w2_err = self.light_curve.lc['w2_magerr'].filled(fill_value=np.nan)
        color_err = np.sqrt(w1_err**2 + w2_err**2)
        color_mask = color_err < color_err_thresh

        # Annotate if nothing passes the cut
        if np.sum(color_mask) == 0:
            ax.text(
                0.5,
                0.5,
                f'Source has no WISE data\nwith $\\sigma_{{W1-W2}} < {color_err_thresh}$.',
                verticalalignment='center',
                horizontalalignment='center',
            )
            return ax

        # Get the delta mags
        delta_mag = self.light_curve.lc['w1_mag'] - self.light_curve.lc['w2_mag']
        delta_mag = delta_mag[color_mask]

        # Plot
        ax.hist(delta_mag, color='k', bins=10)

        # Plot summary stats and the 0.8 criterion
        ax.axvline(0.8, color='red', lw=0.75, label=r'$\rm{W}1 - \rm{W}2 = 0.8$')
        mean_dmag, median_dmag = np.mean(delta_mag), np.median(delta_mag)
        ax.axvline(mean_dmag, label=f'Mean ({mean_dmag:.2f})', color='green')
        ax.axvline(median_dmag, label=f'Median ({median_dmag:.2f})', color='green', linestyle='--')

        # Format
        ax.legend()

        return ax

    def plot_spectrum(self, ax: Optional[Axes] = None) -> Axes:
        """Plot the SDSS and DESI DR1 spectra overlaid on the same panel."""
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 5))

        # Label axes
        ax.set_xlabel(r'\textbf{Wavelength [\AA]}', labelpad=2)
        ax.set_ylabel(r'\boldmath$F_\lambda \ [10^{-17} \ \rm{erg} \ \rm{cm}^{-2} \ \rm{s}^{-1} \ \rm{\AA}^{-1}]$')

        has_sdss = self.spectrum is not None
        has_desi = self.desi_spectrum is not None

        # Annotate if there's no spectrum from either catalog
        if not has_sdss and not has_desi:
            ax.text(
                0.5, 0.5, 'Source has no spectrum from SDSS or DESI DR1.',
                horizontalalignment='center', verticalalignment='center',
            )
            return ax

        # SDSS stores wavelength as log10(Angstroms); convert to linear Angstroms to
        # match DESI so both spectra share the same x-axis.
        if has_sdss:
            sdss_wave = 10 ** self.spectrum[0][1].data['loglam']
            ax.plot(sdss_wave, self.spectrum[0][1].data['model'], color='tab:blue', label='SDSS', zorder=10)
        if has_desi:
            ax.plot(self.desi_spectrum.wavelength, self.desi_spectrum.model, color='tab:red', label='DESI DR1', zorder=10)

        # Set ylim off the (smoother) model curves before plotting the noisier data,
        # so spikes in the raw data don't blow out the y-axis scale. The raw data
        # is unlabeled context for the model line, not its own legend entry.
        ylims = ax.get_ylim()
        if has_sdss:
            ax.plot(sdss_wave, self.spectrum[0][1].data['flux'], color='tab:blue', zorder=-1, alpha=0.3)
        if has_desi:
            ax.plot(self.desi_spectrum.wavelength, self.desi_spectrum.flux, color='tab:red', zorder=-1, alpha=0.3)
        ax.set_ylim(ylims)

        # Add legend
        ax.legend(loc='upper right', fontsize='small')

        return ax

    def get_TNS_info(self) -> Optional[pd.DataFrame]:
        """Closest TNS object within max_arcsec as a one-row frame, or None.

        Columns keep TNS's own names (`name`, `type`, ...) rather than the
        prefixed forms `catalogs.tns` returns, since callers here predate it.
        """
        if self._has_tns_match and self._tns_match is None:
            match = tns_catalog.cone_search(self.ra, self.dec, radius_arcsec=self.max_arcsec)
            if match.empty:
                self._has_tns_match = False
            else:
                self._tns_match = match.head(1).rename(columns=tns_catalog.NATIVE_NAMES)

        return self._tns_match

    def get_filtered_out_info(
        self, filtering_dirpath: Optional[str] = None
    ) -> Dict[str, Dict[str, str]]:
        """Get filtering info as {catalog: {band: reason}}.

        Each cell contains the filter reason from the corresponding
        {cat}_{band}_filtered_out.ecsv, '-' if the source isn't there,
        or 'N/A' if the file doesn't exist.
        """
        result = {cat: {band: '-' for band in self.bands} for cat in CATALOG_INT_MAP}

        for band in self.bands:
            for cat_name, cat_idx in CATALOG_INT_MAP.items():
                field = (
                    str(self.image_metadata['fieldid']).zfill(6)
                    if isinstance(self.image_metadata['fieldid'], int)
                    else self.image_metadata['fieldid']
                )
                base = filtering_dirpath or os.path.join(get_data_path(), 'filter_results')
                fname = f'{cat_idx}_{band}_filtered_out.ecsv'
                path = os.path.join(base, field, fname)
                if not os.path.exists(path):
                    path = os.path.join(base, fname)
                if not os.path.exists(path):
                    result[cat_name][band] = 'N/A'
                    continue

                table = load_cached_table(path)
                if len(table) == 0:
                    continue

                coords = SkyCoord(table['ra'], table['dec'], unit='deg')
                seps = self.coord.separation(coords)
                if np.min(seps.arcsec) <= self.max_arcsec:
                    result[cat_name][band] = table[np.argmin(seps.arcsec)]['filter']

        return result

    @property
    def filtered_out_info(self) -> Dict[str, Dict[str, str]]:
        """Get the information about the source being filtered out."""
        if not hasattr(self, '_filtered_out_info'):
            self._filtered_out_info = self.get_filtered_out_info(self.filtering_dirpath)
        return self._filtered_out_info

    def _get_GAIA_info(self, max_arcsec: float):
        for i in range(3):
            try:
                return Gaia.query_object_async(
                    coordinate=self.coord,
                    radius=max_arcsec * u.arcsec,
                )
            except Exception as e:
                print(f'Error getting GAIA info on attempt {i}: {e}')
                time.sleep(1)
        return Table()

    @property
    def GAIA_info(self) -> Table:
        if self._GAIA_info is None:
            self._GAIA_info = self._get_GAIA_info(self.gaia_max_arcsec)

        return self._GAIA_info

    def get_info_string(self, wise_color_err_thresh: float = 0.5) -> str:
        """Get string with all the necessary source information."""
        info_string = r'\textbf{Source Information:}' f'\nCoordinates: ({self.ra:.5f}, {self.dec:.5f})'
        info_string += f'\nZTF location: {int(self.data["fieldid"])} {int(self.data["ccdid"])} {int(self.data["qid"])}'
        tns_info = self.get_TNS_info()
        if tns_info is None:
            info_string += '\nSource not in TNS.'
        else:
            tns_info = tns_info.iloc[0]
            info_string += (
                f'\nTNS Name: {tns_info["name_prefix"]} {tns_info["name"]}'
                f'\nTNS Discovery date: {tns_info["discoverydate"]}'
                f'\nTNS Reporter: {tns_info["reporting_group"]}'
                f'\nTNS Type: {tns_info["type"]}'
            )
        if self.spectrum is None:
            info_string += '\nNo SDSS Source Classification.'
        else:
            sdss_class = self.spectrum[0][2].data["CLASS"]
            # SUBCLASS carries the emission-line-ratio-based AGN/Seyfert/LINER call that
            # CLASS alone doesn't: a source can be CLASS='GALAXY' and SUBCLASS='AGN'.
            sdss_subclass = self.spectrum[0][2].data["SUBCLASS"]
            info_string += f'\nSDSS Class: {sdss_class}' + (f' ({sdss_subclass})' if sdss_subclass else '')
        if self.desi_spectrum is None:
            info_string += '\nNo DESI DR1 Source Classification.'
        else:
            info_string += f'\nDESI Class: {self.desi_spectrum.spectype}'
        if self.agn_match is None:
            # AGN-DB is a merge of pre-selected AGN/quasar catalogs, not a complete
            # spectroscopic galaxy census (Peca et al. 2026, arXiv:2609.04322) — this is
            # not a confirmed non-AGN determination.
            info_string += '\nSource not in AGN-DB.'
        else:
            # best_class_all is a JSON array of one entry per contributing catalog at the
            # winning tier (spec > SED > xray > image > gen) -- these can genuinely disagree
            # (e.g. one catalog says type1, another says type2), so show all of them keyed
            # by catalog name (via best_class_origin) rather than silently picking index 0.
            classes = json.loads(self.agn_match['best_class_all'])
            origins = json.loads(self.agn_match['best_class_origin'])
            class_by_catalog = {}
            for origin_id, cls in zip(origins, classes):
                catalog_name = CATALOG_ID_TO_NAME.get(str(origin_id), f'cat{origin_id}')
                class_by_catalog[catalog_name] = cls
            agn_class_str = ', '.join(
                f'{latex_escape(name)}: {latex_escape(cls)}' for name, cls in class_by_catalog.items()
            )
            info_string += '\nAGN-DB Class: \\{' + agn_class_str + '\\}'
            if pd.notna(self.agn_match['best_Z_merged']):
                info_string += f'\nAGN-DB Redshift: {self.agn_match["best_Z_merged"]:.4f}'
        if self.simbad_match is None:
            info_string += '\nSource not in SIMBAD.'
        else:
            simbad_info = self.simbad_match
            # main_id routinely contains '_' and '&' (e.g. survey designations), which
            # LaTeX would otherwise read as a subscript and an alignment tab.
            info_string += f'\nSIMBAD Name: {latex_escape(str(simbad_info["simbad_main_id"]))}'
            # otype_label already spells out candidacy ('Active Galaxy Nucleus
            # Candidate'), so the terse code is shown alongside only for reference.
            if pd.notna(simbad_info['simbad_otype']):
                otype = str(simbad_info['simbad_otype'])
                label = (str(simbad_info['simbad_otype_label'])
                         if pd.notna(simbad_info['simbad_otype_label']) else otype)
                info_string += (f'\nSIMBAD Type: {latex_escape(label)} '
                                f'({latex_escape(otype)})')
            if pd.notna(simbad_info['simbad_z']):
                info_string += f'\nSIMBAD Redshift: {simbad_info["simbad_z"]:.4f}'
            if pd.notna(simbad_info['simbad_nbref']):
                info_string += f'\nSIMBAD References: {int(simbad_info["simbad_nbref"])}'
        if 'w1_magerr' in self.light_curve.lc.columns and 'w2_magerr' in self.light_curve.lc.columns:
            w1_err = self.light_curve.lc['w1_magerr'].filled(fill_value=np.nan)
            w2_err = self.light_curve.lc['w2_magerr'].filled(fill_value=np.nan)
            color_err = np.sqrt(w1_err**2 + w2_err**2)
            color_mask = color_err < wise_color_err_thresh
            if np.sum(color_mask) > 0:
                delta_mag = self.light_curve.lc['w1_mag'] - self.light_curve.lc['w2_mag']
                delta_mag = delta_mag[color_mask]
                info_string += (
                    f'\nWISE W1-W2 Mean ($\\sigma_{{W1-W2}} < {wise_color_err_thresh}$) $=$ {np.mean(delta_mag):.2f}'
                    f'\nWISE W1-W2 Median ($\\sigma_{{W1-W2}} < {wise_color_err_thresh}$) $=$ {np.median(delta_mag):.2f}'
                )
        else:
            info_string += '\nNo WISE W1-W2 data.'

        return info_string

    def plot_filtered_out_table(self, ax: Axes) -> None:
        """Render filtered_out_info as a matplotlib table (rows=catalogs, cols=bands)."""
        catalog_names = list(CATALOG_INT_MAP.keys())
        info = self.filtered_out_info
        cell_text = [[band] + [info[cat][band] for cat in catalog_names] for band in self.bands]
        col_widths = [0.08] + [0.3] * len(catalog_names)

        tbl = ax.table(
            cellText=cell_text,
            colLabels=[''] + catalog_names,
            colWidths=col_widths,
            # Explicit bbox at the table's own natural height (measured empirically for
            # this content: loc='upper center' renders it at y=[0.74, 0.98]), just shifted
            # down -- bbox stretches a table to fill whatever height it's given, so it must
            # match the natural height, not some arbitrary fraction, or the table balloons.
            bbox=[0.0, 0.55, 1.0, 0.24],
            cellLoc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        # ax.set_title() sat inconsistently relative to the table above; use ax.text (same
        # approach as the "Source Information:" panel) for a title anchored directly to a
        # known axes-fraction position instead.
        ax.text(0.5, 0.80, r'\textbf{Filtering}', ha='center', va='bottom', fontsize='large')
        ax.axis('off')

    def plot_everything(self) -> Axes:
        """Function that plots everything on one page!"""
        # Set up the layout. Left half (cols 0-2): cutouts + main light curve, stacked as
        # before. Right half (cols 3-5): what used to be the next three full-width rows
        # (spectrum, WISE lc + WISE color, info text + filtering table), folded sideways
        # into the same 3 rows instead of stacking further down the page.
        set_mpl_params()
        # 26 columns: the left block (cutouts + main lc) is 12 columns, then a 1-column
        # gutter, then a 13-column right block. Within the right block, the spectrum spans
        # the full 13 columns, while the row below (WISE lc + WISE color) and the row below
        # that (info text + filter table) each reserve their own 1-column gutter between
        # their two sub-panels, so none of their y-axis labels crowd their neighbor.
        fig = plt.figure(figsize=(26, 10.8))
        ax0 = plt.subplot2grid((3, 26), (0, 0), colspan=4)
        ax1 = plt.subplot2grid((3, 26), (0, 4), colspan=4)
        ax2 = plt.subplot2grid((3, 26), (0, 8), colspan=4)
        ax3 = plt.subplot2grid((3, 26), (1, 0), colspan=4)
        ax4 = plt.subplot2grid((3, 26), (1, 4), colspan=4)
        ax5 = plt.subplot2grid((3, 26), (1, 8), colspan=4)
        lc_ax = plt.subplot2grid((3, 26), (2, 0), colspan=12)
        # column 12 intentionally left empty as the left/right block gutter
        spec_ax = plt.subplot2grid((3, 26), (0, 13), colspan=13)
        wise_lc_ax = plt.subplot2grid((3, 26), (1, 13), colspan=8)
        # column 21 intentionally left empty as the WISE lc / WISE color gutter
        wise_hist_ax = plt.subplot2grid((3, 26), (1, 22), colspan=4)
        text_ax = plt.subplot2grid((3, 26), (2, 13), colspan=5)
        # column 18 intentionally left empty as the text / filter-table gutter
        filter_table_ax = plt.subplot2grid((3, 26), (2, 19), colspan=7)
        cutout_axes = np.array([[ax0, ax1, ax2], [ax3, ax4, ax5]])
        axes = np.array([cutout_axes, lc_ax, spec_ax, wise_lc_ax, wise_hist_ax], dtype=object)

        # Plot
        self.plot_all_cutouts(axes=cutout_axes)
        self.plot_lc(ax=lc_ax, fig=fig, xlab_kwags={})
        self.plot_spectrum(ax=spec_ax)
        self.plot_wise_lc(ax=wise_lc_ax, fig=fig, xlab_kwags={})
        self.plot_wise_mag_hist(ax=wise_hist_ax)
        self.plot_filtered_out_table(ax=filter_table_ax)

        # Annotate text info at the bottom
        info_string = self.get_info_string()
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
            bands_per_source: Optional[Iterable[tuple]] = None,
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
            if bands_per_source is None:
                bands_per_source = [None] * len(self.ras)
            self.sources = [
                Source(ra, dec, detected_bands=db, **kwargs)
                for ra, dec, db in zip(self.ras, self.decs, bands_per_source)
            ]

        self._data = None
        self._coords = None

    @classmethod
    def from_table(cls, table: Table, **kwargs) -> 'Sources':
        """Load Sources from a table.

        Args:
            table: Table to load from
            **kwargs: Keyword arguments to pass to the Source constructor

        Returns:
            A new Sources instance loaded from the table
        """

        # Add the mandatory columns if they're not in the table
        if len(np.intersect1d(table.colnames, MANDATORY_SOURCE_COLUMNS)) < len(MANDATORY_SOURCE_COLUMNS):
            warnings.warn("data table does not have all the mandatory columns, adding them with nans.")
            for col in [c for c in MANDATORY_SOURCE_COLUMNS if c not in table.columns]:
                table[col] = [np.nan] * len(table)

        # Create Source objects for each row
        sources = [Source(row['ra'], row['dec'], **kwargs) for row in table]
        for i, row in enumerate(table):
            sources[i].data = Table(row)

        return cls(sources=sources, **kwargs)

    @classmethod
    def from_file(cls, fname: str, **kwargs) -> 'Sources':
        """Load Sources from a file that was saved using Sources.save().

        Args:
            fname: Path to the file to load from

        Returns:
            A new Sources instance loaded from the file
        """
        # Derive filtering_dirpath from the file location.
        # If filtered_out ecsvs live alongside the source file (single-field dir), use the
        # parent dir directly. Otherwise go two levels up (multi-field {filter_dir}/{field}/).
        if 'filtering_dirpath' not in kwargs:
            parent = os.path.dirname(os.path.abspath(fname))
            if glob(os.path.join(parent, '*_filtered_out.ecsv')):
                kwargs['filtering_dirpath'] = parent
            else:
                kwargs['filtering_dirpath'] = os.path.dirname(parent)

        # Read the table from file
        table = load_ecsv(fname)

        return cls.from_table(table, **kwargs)

    @property
    def coords(self) -> SkyCoord:
        if self._coords is None or len(self._coords) != self.__len__():
            if len(self.sources) == 0:
                return []
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
                return Table(data={k: [] for k in MANDATORY_SOURCE_COLUMNS}, masked=False)

            # Pre-compute closest catalog index per band for all sources at once (one KD-tree
            # query per band instead of one O(K) linear scan per source per band). Only the
            # sources that still have to build their row need this, and reaching for
            # field_catalogs loads a per-band HDF5 catalog off disk -- so skip it entirely
            # when every source already carries its data (e.g. loaded via from_file).
            if any(src._data is None for src in self.sources):
                source_coords = self.coords
                ref = self.sources[0]
                for band, cat in ref.field_catalogs.items():
                    cat_coords = SkyCoord(ra=cat['ra'], dec=cat['dec'], unit='deg')
                    idx, sep2d, _ = match_coordinates_sky(source_coords, cat_coords)
                    for i, src in enumerate(self.sources):
                        if src._precomputed_cat_indices is None:
                            src._precomputed_cat_indices = {}
                        src._precomputed_cat_indices[band] = (
                            idx[i] if sep2d[i].arcsecond <= src.max_arcsec else None
                        )

            self._data = vstack([src.data for src in self.sources])
            self._data = Table(self._data, masked=False)

        return self._data

    @data.setter
    def data(self, data_tab: Table) -> None:
        """Set the data table for this Sources object."""
        if not isinstance(data_tab, Table):
            raise TypeError("data must be an astropy Table")
        if len(data_tab) != len(self.sources):
            raise ValueError(f"data must have exactly {len(self.sources)} rows")
        if len(np.intersect1d(data_tab.colnames, MANDATORY_SOURCE_COLUMNS)) < len(MANDATORY_SOURCE_COLUMNS):
            # Add the mandatory columns if they're not in the table
            warnings.warn("data table does not have all the mandatory columns, adding them with nans.")
            for col in [c for c in MANDATORY_SOURCE_COLUMNS if c not in data_tab.columns]:
                data_tab[col] = [np.nan] * len(data_tab)

        self._data = data_tab

    @property
    def in_bands(self) -> List[List[str]]:
        """The bands that each source was extracted/found in."""
        return [src.in_bands for src in self.sources]

    @property
    def in_g(self) -> np.ndarray:
        if not hasattr(self, '_in_g'):
            self._in_g = np.array([s.in_g for s in self.sources], dtype=bool)
        return self._in_g

    @property
    def in_r(self) -> np.ndarray:
        if not hasattr(self, '_in_r'):
            self._in_r = np.array([s.in_r for s in self.sources], dtype=bool)
        return self._in_r

    @property
    def in_i(self) -> np.ndarray:
        if not hasattr(self, '_in_i'):
            self._in_i = np.array([s.in_i for s in self.sources], dtype=bool)
        return self._in_i

    def save(self, fname: str, overwrite: bool = True):
        """Save the Sources data to a hdf5 file."""
        to_save = self.data.copy()
        to_save['filter_info'] = [str(src.filter_info) for src in self.sources]
        if len(to_save) == 0:
            to_save['ra'] = []
            to_save['dec'] = []
        to_save = prepare_table_for_write(to_save)
        if fname.endswith('.hdf5'):
            to_save.write(
                fname,
                path='data',
                serialize_meta=True,
                overwrite=overwrite,
            )
        elif fname.endswith('.ecsv'):
            to_save.write(fname, format='ascii.ecsv', overwrite=overwrite)
        else:
            raise ValueError(f'{fname} must end with .hdf5 or .ecsv')


    def inTNS(self):
        """Boolean array: does each source have a TNS object within its match radius?"""
        if len(self.sources) == 0:
            return np.array([], dtype=bool)

        # One vectorised query at the widest radius any source uses, then cut
        # each source back to its own -- they can differ per source.
        per_source = np.array([src.max_arcsec for src in self.sources], dtype=float)
        matches = tns_catalog.cone_search_many(
            [src.ra for src in self.sources],
            [src.dec for src in self.sources],
            radius_arcsec=float(per_source.max()),
        )
        sep = matches['tns_sep_arcsec'].to_numpy(dtype=float)
        return np.isfinite(sep) & (sep <= per_source)
