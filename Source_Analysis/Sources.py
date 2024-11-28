import os
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt

from concurrent import futures
from typing import List, Tuple, Union, Optional, Iterable, Dict
from matplotlib.axes._axes import Axes

from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.visualization import simple_norm

from Extracting.utils import get_data_path
from Extracting.Catalogs import ZTF_Catalog, ZTF_CUTOUT_HALFWIDTH, get_ztf_metadata, get_pstarr_cutout
from ztf_fp_query.Forced_Photo_Map import Forced_Photo_Map
from ztf_fp_query.query import ZTFFP_Service

ACCEPTABLE_PROC_STATUS = [0]


def closest_within_radius(coord: SkyCoord, coords: SkyCoord, max_arcsec: float = 1.0) -> Tuple[int, SkyCoord]:
    """Finds the closest coordinate in 'coords' to 'coord' that is less than a given distance away."""
    # Calculate separations between coord and each coordinate in coords
    seps = coord.separation(coords)
    within_one_arcsecond = seps < max_arcsec * u.arcsec

    print(f'min sep = {np.nanmin(seps)}')

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
        ):
        self.ra = ra
        self.dec = dec
        self.coord = SkyCoord(ra=ra, dec=dec, unit='deg')
        self.bands = bands
        self.stamp_width_arcsec = stamp_width_arcsec
        self.arcsec_per_pixel = None

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arcsec_per_pixel = 1.01

    def transform_pix_coords(
            self,
            x: Union[float, np.ndarray],
            y: Union[float, np.ndarray],
            band: str,
        ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        return self.images[band].shape[1] - x - 1, self.images[band].shape[0] - y - 1


    def get_images(self) -> Tuple[Dict[str, np.ndarray], Dict[str, WCS]]:

        # Get the images from the ZTF cutouts
        ztf_catalog = ZTF_Catalog(self.ra, self.dec, catalog_bands=self.bands)
        self._images, self._WCSs = {}, {}
        for band in self.bands:
            im = ztf_catalog.sextractors[band].image_sub
            self._WCSs[band] = ztf_catalog.sextractors[band].wcs

            # Get the pixel location of the center of the image
            x, y = ztf_catalog.sextractors[band].ra_dec_to_pix(self.ra, self.dec)

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
        for band in self.bands:

            # Query the image
            width_pixels = self.stamp_width_arcsec / self.arcsec_per_pixel
            images[band], wcss[band] = get_pstarr_cutout(self.ra, self.dec, size=width_pixels, filter=band)

        return images, wcss


class Source():
    def __init__(
            self,
            ra: float,
            dec: float,
            bands: list = ['g', 'r', 'i'],
            cutout_bands: Optional[List[str]] = None,
            merged_field_basedir: str = '/Users/adamboesky/Research/long_transients/Data/catalog_results/field_results',
        ):
        self.ra = ra
        self.dec = dec
        self.coord = SkyCoord(ra=ra, dec=dec, unit='deg')

        # Get the field for the given RA and DEC
        self.bands = bands
        self.cutout_bands = bands if cutout_bands is None else cutout_bands
        self.merged_field_basedir = merged_field_basedir

        # Properties
        self._field_catalogs = None
        self._data = None
        self._postage_stamps = None
        self._ztf_lightcurve = None
        self._paddedfield = None

    @property
    def paddedfield(self) -> str:
        if self._paddedfield is None:
            metadata_table = get_ztf_metadata(
                ra_range=(self.ra - ZTF_CUTOUT_HALFWIDTH, self.ra + ZTF_CUTOUT_HALFWIDTH),
                dec_range=(self.dec - ZTF_CUTOUT_HALFWIDTH, self.dec + ZTF_CUTOUT_HALFWIDTH),
                filter='g',
            )
            self._paddedfield = str(metadata_table['field'].iloc[0]).zfill(6)
        return self._paddedfield

    @property
    def field_catalogs(self) -> dict[str, Table]:
        print('Loading catalogs!')
        if self._field_catalogs is None:
            self._field_catalogs = {}

            def load_catalog(band):
                print(f'Loading {band} catalog from locally stored catalogs...')
                return band, Table.read(
                    os.path.join(self.merged_field_basedir, f'{self.paddedfield}_{band}.ecsv')
                )

            with futures.ThreadPoolExecutor() as executor:
                future_to_band = {executor.submit(load_catalog, band): band for band in self.bands}
                for future in futures.as_completed(future_to_band):
                    band, catalog = future.result()
                    self._field_catalogs[band] = catalog
        return self._field_catalogs

    @property
    def data(self) -> Table:
        if self._data is None:

            # TODO: URGENT SOMEHOW THIS IS BROKEN **** DOESN'T GET THE RIGHT MAGS
            # Set up an empty table to fill in
            unique_colnames = []
            for tab in self.field_catalogs.values():
                unique_colnames.extend([cname for cname in tab.colnames])
            unique_colnames = set(unique_colnames)
            data_dict = {k: [np.nan] for k in unique_colnames}
            str_cols = ['Catalog']  # need to have types align
            for col in str_cols:
                data_dict[col] = [str(data_dict[col][0])]
            self._data = Table(data_dict)

            print('Searching for source in the catalogs!')
            for band, cat in self.field_catalogs.items():
                print(f'Searching {band}...')

                # Get the source from each catalog
                coords = SkyCoord(ra=cat['ra'], dec=cat['dec'], unit='deg')
                ind_closest, _ = closest_within_radius(self.coord, coords, max_arcsec=1.0)

                # If a coord was found, join the tables
                if ind_closest is not None:

                    # Fill in table
                    for cname in cat.colnames:
                        if isinstance(self._data[cname][0], str):
                            entry_is_nan = self._data[cname][0].lower() == 'nan'
                        else:
                            entry_is_nan = np.isnan(self._data[cname][0])
                        if entry_is_nan:
                            if isinstance(cat[cname][ind_closest], bool):
                                self._data[cname][0] = float(cat[cname][ind_closest])
                            else:
                                self._data[cname][0] = cat[cname][ind_closest]

        return self._data

    @property
    def postage_stamps(self) -> Dict[str, Postage_Stamp]:
        if self._postage_stamps is None:
            self._postage_stamps = {
                'ZTF': ZTF_Postage_Stamp(self.ra, self.dec, bands=self.cutout_bands),
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

    def plot_postage_stamps(self, band: str, axes: Optional[Axes] = None, **kwargs) -> Axes:
        # Make axes if not given
        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(15, 7.5))

        # Plot
        self.postage_stamps['PSTARR'].plot_cutout(band='g', ax=axes[0], **kwargs)
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
            color='white'
        )
        axes[1].text(
            0.01,
            0.99,
            rf'{band} mag = {self.data[f"ZTF_{band}PSFMag"][0]:.2f}',
            transform=axes[1].transAxes,
            ha='left',
            va='top',
            fontsize=15,
            color='white'
        )

        # Formatting
        axes[0].set_title('Pan-STARRS', fontsize=15)
        axes[1].set_title('ZTF', fontsize=15)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        plt.tight_layout()

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
        scatter_kwargs['marker'] = '^'
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

            # Plot
            not_upper_lim_mask = np.logical_not(lc['upperlim']).to_numpy()
            ax.errorbar(
                x=lc['jd'][not_upper_lim_mask] - time_offset,
                y=lc['mag'][not_upper_lim_mask],
                yerr=lc['magerr'][not_upper_lim_mask],
                color=colors[band],
                **kwargs,
            )
            if include_upper_lim:

                # Plot upper limits
                ax.scatter(
                    x=lc['jd'][~not_upper_lim_mask] - time_offset,
                    y=lc['mag'][~not_upper_lim_mask],
                    color=colors[band],
                    **scatter_kwargs,
                )

        # Formatting)
        ax.invert_yaxis()

        return ax

    def plot_cutouts_and_light_curves(
            self,
            ax_pstarr_cutout: Optional[Axes] = None,
            ax_ztf_cutout: Optional[Axes] = None,
            ax_light_curves: Optional[Axes] = None,
            acceptable_proc_status: List[int] = ACCEPTABLE_PROC_STATUS,
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
        self.plot_ztf_lightcurve(bands=self.bands, ax=ax_light_curves, acceptable_proc_status=acceptable_proc_status)

        # Formatting
        ax_pstarr_cutout.set_title('Pan-STARRS', fontsize=15)
        ax_ztf_cutout.set_title('ZTF', fontsize=15)
        ax_light_curves.grid(ls=':', lw=0.5)
        ax_light_curves.set_xlabel('Time [day]')
        ax_light_curves.set_ylabel('Mag')
        plt.tight_layout()

        return ax_pstarr_cutout, ax_ztf_cutout, ax_light_curves

    def get_TNS_info(self, tns_df: Optional[pd.DataFrame] = None, tns_coords: Optional[SkyCoord] = None) -> bool:
        # Load TNS if it is not given
        if tns_df is None:
            tns_df = pd.read_csv(os.path.join(get_data_path(), 'tns_public_objects.csv'))
        if tns_coords is None:
            tns_coords = SkyCoord(tns_df['ra'], tns_df['declination'], unit='deg')

        # Get info from TNS
        idx, sep2d, _ = match_coordinates_sky(self.coord, tns_coords)
        if sep2d.arcsec > 1:
            return None
        return tns_df.iloc[[idx]]


class Sources:
    """Collection of the Source class."""
    def __init__(
            self,
            ras: Iterable[float],
            decs: Iterable[float],
            **kwargs,
        ):
        self.sources = [
            Source(ra, dec, **kwargs) for ra, dec in zip(ras, decs)
        ]

    def __iter__(self):
        return iter(self.sources)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        return self.sources[index]

    def submit_forced_photometry_batch(self):
        # Load important objects
        ztf_fp_service = ZTFFP_Service()
        ztf_fp_map = Forced_Photo_Map()

        # Make sure that the ras and decs aren't already downloaded, pending, or have been recently queried
        already_downloaded = ztf_fp_map.contains(self.ras, self.decs)
        recently_queried = ztf_fp_service.recently_queried(self.ras, self.decs)
        currently_pending = ztf_fp_service.currently_pending(self.ras, self.decs)
        to_submit_mask = (not already_downloaded) & (not recently_queried) & (not currently_pending)
        ras_to_submit, decs_to_submit = self.ras[to_submit_mask], self.decs[to_submit_mask]

        # Submit forced photometry job
        print(f'Submitting forced photometry request on {len(ras_to_submit)} source. \n{np.sum(not to_submit_mask)} of'
              'the given coordinates were already downloaded or requested.')
        ztf_fp_service.submit(ras_to_submit, decs_to_submit)

    def inTNS(self):
        # Load TNS if it is not given
        tns_df = pd.read_csv(os.path.join(get_data_path(), 'tns_public_objects.csv'))
        tns_coords = SkyCoord(tns_df['ra'], tns_df['declination'], unit='deg')

        return np.array(
            [src.get_TNS_info(tns_df=tns_df, tns_coords=tns_coords) is not None for src in self.sources],
            dtype=bool,
        )
