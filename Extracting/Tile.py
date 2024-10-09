import numpy as np

from typing import Iterable, Optional

from Extracting.Catalogs import (PSTARR_Catalog, ZTF_Catalog,
                                 associate_tables_by_coordinates)


class Tile():
    def __init__(self, ra: float, dec: float, bands: Iterable[str] = ['g', 'r', 'i'], data_dir: Optional[str] = None):

        # Get the ZTF catalog and coordinate range for the tile
        self.bands = bands
        self.ztf_catalogs = {band: ZTF_Catalog(ra, dec, catalog_bands=band, data_dir=data_dir) for band in bands}
        self.ra_range, self.dec_range = self.ztf_catalogs[self.bands[0]].get_coordinate_range()

        # Get the PanSTARRS catalog for the tile
        self.pstar_catalog = PSTARR_Catalog(self.ra_range, self.dec_range)

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
            # TODO: Parallelize?

            # Set the point source scores for the PSF construction
            for band in self.bands:
                self.ztf_catalogs[band].sextractors[band].set_sources_for_psf(
                    self.pstar_catalog.data[['ra', 'dec', f'{band}KronMag', f'{band}KronMagErr', f'{band}PSFMag']]
                )

            self._data_dicts = {
                band: associate_tables_by_coordinates(
                    self.ztf_catalogs[band].data,
                    self.pstar_catalog.data,
                    prefix1='ZTF', prefix2='PSTARR'
                ) for band in self.bands
            }

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
