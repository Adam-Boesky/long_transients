import os
import shutil
import pickle
import numpy as np

from typing import Dict, Iterable, Optional, Tuple, Union
from astropy.table import Table
from concurrent.futures import ThreadPoolExecutor

try:
    from Catalogs import (PSTARR_Catalog, ZTF_Catalog, associate_tables_by_coordinates)
except ModuleNotFoundError:
    from .Catalogs import (PSTARR_Catalog, ZTF_Catalog, associate_tables_by_coordinates)


class Tile():
    def __init__(
            self,
            ra: float = None,
            dec: float = None,
            ztf_metadata: Dict[str, Union[str, int]] = None,
            bands: Union[Iterable[str], str] = ['g', 'r', 'i'],
            data_dir: Optional[str] = None,
            parallel: bool = False,
            overwrite_mydb: bool = False,
        ):
        """A class to represent a tile of the sky. This corresponds to one quadrant of a ZTF field."""
        if isinstance(bands, str):
            bands = [bands]

        if not isinstance(ztf_metadata, dict) and not isinstance(ra, float) and not isinstance(dec, float):
            raise ValueError('Tile must be given either (ra, dec) or a ztf_metadata dictionary.')

        # Get the ZTF catalog and coordinate range for the tile
        def _create_ztf_catalog(band):
            """Function to help create catalogs."""
            return ZTF_Catalog(ra, dec, band=band, data_dir=data_dir, image_metadata=ztf_metadata)

        self.parallel = parallel
        if self.parallel:
            with ThreadPoolExecutor() as executor:
                results = executor.map(_create_ztf_catalog, bands)
                self.ztf_catalogs = {band: catalog for band, catalog in results if catalog is not None}
        else:
            self.ztf_catalogs = {band: _create_ztf_catalog(band) for band in bands}

        # Drop the bands that aren't available
        new_cats = self.ztf_catalogs.copy()
        for b, c in self.ztf_catalogs.items():
            if c is None:
                bands.remove(b)
                new_cats.pop(b)
        self.ztf_catalogs = new_cats
        self.bands = bands

        # Load PSTARR catalog
        if self.n_bands > 0:
            self.ra_range, self.dec_range = self.ztf_catalogs[self.bands[0]].get_coordinate_range()

            # Get the PanSTARRS catalog for the tile
            self.pstar_catalog = PSTARR_Catalog(self.ra_range, self.dec_range, prefetch=parallel, catalog_bands=self.bands, overwrite_mydb=overwrite_mydb)

            # The crux of this class will be this massive Astropy table
            self._data_dicts = None

    @property
    def n_bands(self) -> int:
        return len(self.bands)

    @property
    def data_dicts(self) -> dict[str, Table]:
        if self._data_dicts is None:

            # Function to process each band
            def process_band(band: str) -> Tuple[str, Table]:
                self.ztf_catalogs[band].sextractor.set_sources_for_psf(
                    self.pstar_catalog.data[
                        ['ra', 'dec', f'{band}KronMag', f'{band}KronMagErr', f'{band}PSFMag', f'{band}PSFMagErr']
                    ]
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
                    x, y = self.ztf_catalogs[band].sextractor.ra_dec_to_pix(row['PSTARR_ra'], row['PSTARR_dec'])
                    f.write(f"{x} {y}\n")

    def prefecth_catalogs(self):
        # Prefetch PanSTARRS
        self.pstar_catalog.prefetch()

        # Set the PSF sources
        def _set_sources_for_psf(band: str):
            self.ztf_catalogs[band].sextractor.set_sources_for_psf(
                self.pstar_catalog.data[
                    ['ra', 'dec', f'{band}KronMag', f'{band}KronMagErr', f'{band}PSFMag', f'{band}PSFMagErr']
                ]
            )

        def _prefetch_ztf(band: str):
            self.ztf_catalogs[band].prefetch()

        if self.parallel:
            with ThreadPoolExecutor() as executor:
                executor.map(_set_sources_for_psf, self.bands)
                executor.map(_prefetch_ztf, self.bands)
        else:
            for band in self.bands:
                _set_sources_for_psf(band)
                _prefetch_ztf(band)

    def store_catalogs(self, out_parent_dir: str, overwrite: bool = False) -> str:
        """Store the ZTF and PanSTARRS catalogs."""
        # Get the output directory
        out_subdir = f"{self.ztf_catalogs[self.bands[0]].image_metadata['fieldid']}_{self.ztf_catalogs[self.bands[0]].image_metadata['ccdid']}_{self.ztf_catalogs[self.bands[0]].image_metadata['qid']}"
        outdir = os.path.join(out_parent_dir, out_subdir)

        if not overwrite and os.path.exists(outdir):
            print(f"Directory {outdir} already exists. Skipping.")
            return outdir

        # Prefetch catalogs
        self.prefecth_catalogs()

        # Make a subdir... overwrite if told
        if os.path.exists(outdir):
            if not overwrite:
                raise FileExistsError(f"Directory {outdir} already exists.")
            shutil.rmtree(outdir)
        os.makedirs(outdir)
        os.makedirs(os.path.join(outdir, 'nan_masks'))
        os.makedirs(os.path.join(outdir, 'WCSs'))
        os.makedirs(os.path.join(outdir, 'EPSFs'))

        # Store the PanSTARR catalog
        self.pstar_catalog.data.write(os.path.join(outdir, f'PSTARR.ecsv'))

        # Store the ZTF catalogs, ZTF nan masks, and WCSs
        for band in self.bands:

            # Save the ztf catalog
            self.ztf_catalogs[band].data.write(os.path.join(outdir, f'ZTF_{band}.ecsv'))

            # Save the ZTF catalog's nan mask, WCS, and EPSF fit
            np.save(os.path.join(outdir, 'nan_masks', f'ZTF_{band}_nan_mask.npy'), self.ztf_catalogs[band].sextractor.nan_mask)
            with open(os.path.join(outdir, 'WCSs', f'ZTF_{band}_wcs.pkl'), 'wb') as f:
                pickle.dump(self.ztf_catalogs[band].sextractor.wcs, f)
            np.save(os.path.join(outdir, 'EPSFs', f'ZTF_{band}_EPSF.npy'), self.ztf_catalogs[band].sextractor.epsf.data)

        return outdir
