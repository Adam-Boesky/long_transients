import os
import numpy as np
import astropy.units as u

from concurrent import futures
from typing import Tuple, Union, Optional, Iterable, Dict
from matplotlib.axes._axes import Axes

from Extracting.Catalogs import ZTF_Catalog, ZTF_CUTOUT_HALFWIDTH, get_ztf_metadata, get_pstarr_cutout
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm


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
            import matplotlib.pyplot as plt
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
            merged_field_basedir: str = '/Users/adamboesky/Research/long_transients/Data/catalog_results/field_results',
        ):
        self.ra = ra
        self.dec = dec
        self.coord = SkyCoord(ra=ra, dec=dec, unit='deg')

        # Get the field for the given RA and DEC
        metadata_table = get_ztf_metadata(
            ra_range=(self.ra - ZTF_CUTOUT_HALFWIDTH, self.ra + ZTF_CUTOUT_HALFWIDTH),
            dec_range=(self.dec - ZTF_CUTOUT_HALFWIDTH, self.dec + ZTF_CUTOUT_HALFWIDTH),
            filter='g',
        )
        self.paddedfield = str(metadata_table['field'].iloc[0]).zfill(6)
        self.bands = bands
        self.merged_field_basedir = merged_field_basedir

        # Properties
        self._field_catalogs = None
        self._data = None
        self._images = None

    @property
    def field_catalogs(self) -> dict[str, Table]:
        print('Loading catalogs!')
        if self._field_catalogs is None:
            self._field_catalogs = {}

            def load_catalog(band):
                print(f'Loading {band}...')
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
    def images(self) -> Table:
        # Make a ZTF catalog for the object. This will be used largely for the image access
        ztf_catalogs = ZTF_Catalog(self.ra, self.dec, catalog_bands=bands)
