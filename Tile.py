from Catalogs import ZTF_Catalog, PSTARR_Catalog, associate_tables_by_coordinates


class Tile():
    def __init__(self, ra: float, dec: float):

        # Get the ZTF catalog and coordinate range for the tile
        self.ztf_catalog = ZTF_Catalog(ra, dec)
        self.ra_range, self.dec_range = self.ztf_catalog.get_coordinate_range()

        # Get the PanSTARRS catalog for the tile
        self.pstar_catalog = PSTARR_Catalog(self.ra_range, self.dec_range)

        # The crux of this class will be this massive Astropy table
        self._data = None

    @property
    def data(self) -> dict:
        if self._data is None:
            self._data = associate_tables_by_coordinates(self.ztf_catalog.data, self.pstar_catalog.data, prefix1='ZTF', prefix2='PSTARR')

        return self._data
