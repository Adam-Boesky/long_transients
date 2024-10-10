import numpy as np

from typing import Iterable, Optional, Tuple
from astropy.table import Table
from concurrent.futures import ThreadPoolExecutor

from Extracting.Catalogs import (PSTARR_Catalog, ZTF_Catalog,
                                 associate_tables_by_coordinates)


class Tile():
    def __init__(
            self,
            ra: float,
            dec: float,
            bands: Iterable[str] = ['g', 'r', 'i'],
            data_dir: Optional[str] = None,
            parallel: bool = False,
        ):

        # Get the ZTF catalog and coordinate range for the tile
        self.bands = bands
        self.parallel = parallel
        if self.parallel:
            with ThreadPoolExecutor() as executor:
                self.ztf_catalogs = dict(executor.map(lambda band: (band, ZTF_Catalog(ra, dec, catalog_bands=band, data_dir=data_dir)), bands))
        else:
            self.ztf_catalogs = {band: ZTF_Catalog(ra, dec, catalog_bands=band, data_dir=data_dir) for band in bands}
        self.ra_range, self.dec_range = self.ztf_catalogs[self.bands[0]].get_coordinate_range()

        # Get the PanSTARRS catalog for the tile
        self.pstar_catalog = PSTARR_Catalog(self.ra_range, self.dec_range, prefetch=parallel)

        # The crux of this class will be this massive Astropy table
        self._data_dicts = None

    # @property
    # def data(self) -> dict:
    #     if self._data is None:
    #         self._data = associate_tables_by_coordinates(self.ztf_catalog.data, self.pstar_catalog.data, prefix1='ZTF', prefix2='PSTARR')

    #     return self._data
    @property
    def data_dicts(self) -> dict:
        if self._data_dicts is None:

            # Function to process each band
            def process_band(band: str) -> Tuple[str, Table]:
                self.ztf_catalogs[band].sextractors[band].set_sources_for_psf(
                    self.pstar_catalog.data[['ra', 'dec', f'{band}KronMag', f'{band}KronMagErr', f'{band}PSFMag']]
                )
                return band, associate_tables_by_coordinates(
                    self.ztf_catalogs[band].data,
                    self.pstar_catalog.data,
                    prefix1='ZTF', prefix2='PSTARR'
                )

            if self.parallel:
                with ThreadPoolExecutor() as executor:
                    results = executor.map(process_band, self.bands)
                    self._data_dicts = {band: data for band, data in results}
            else:
                self._data_dicts = {band: process_band(band)[1] for band in self.bands}

        return self._data_dicts

    def store_unassociated(self, fpath: str):
        for band in self.bands:

            # ZTF
            only_ztf_sources = self.data_dicts[band][self.data_dicts[band]['Catalog'] == 'ZTF']
            with open(f'ZTF{band}_{fpath}', 'w') as f:
                for row in only_ztf_sources:
                    f.write(f"{row['ZTF_x']} {row['ZTF_y']}\n")

            # PanSTARRS
            only_pstarr_sources = self.data_dicts[band][self.data_dicts[band]['Catalog'] == 'PSTARR']
            with open(f'PSTARR{band}_{fpath}', 'w') as f:
                for row in only_pstarr_sources:
                    x, y = self.ztf_catalogs[band].sextractors[band].ra_dec_to_pix(row['PSTARR_ra'], row['PSTARR_dec'])
                    f.write(f"{x} {y}\n")
