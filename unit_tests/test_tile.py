import numpy as np
import pytest
from astropy.table import Table
from unittest.mock import Mock, patch
from Catalogs import PSTARR_Catalog, ZTF_Catalog, associate_tables_by_coordinates
from Extracting.Tile import Tile

@pytest.fixture
def mock_catalogs(mocker):
    mock_ztf_catalog = mocker.Mock(spec=ZTF_Catalog)
    mock_pstar_catalog = mocker.Mock(spec=PSTARR_Catalog)
    mock_ztf_catalog.get_coordinate_range.return_value = ((10.0, 20.0), (-10.0, 10.0))
    mock_ztf_catalog.data = Table({'ra': [11.0, 14.0, 16.0], 'dec': [1.0, 4.0, 6.0]})
    mock_pstar_catalog.data = Table({'ra': [11.0, 15.0, 19.0], 'dec': [1.0, 5.0, 9.0]})
    mocker.patch('Tile.ZTF_Catalog', return_value=mock_ztf_catalog)
    mocker.patch('Tile.PSTARR_Catalog', return_value=mock_pstar_catalog)
    return mock_ztf_catalog, mock_pstar_catalog

def test_tile_init(mock_catalogs):
    ra, dec = 15.0, 5.0
    tile = Tile(ra, dec)
    assert tile.ztf_catalogs is not None
    assert tile.pstar_catalog is not None
    assert tile.ra_range == (10.0, 20.0)
    assert tile.dec_range == (-10.0, 10.0)

def test_tile_data_dicts(mock_catalogs):
    ra, dec = 15.0, 5.0
    tile = Tile(ra, dec)
    data_dicts = tile.data_dicts
    assert isinstance(data_dicts, dict)
    for band, data in data_dicts.items():
        assert isinstance(data, Table)
        assert len(data) == 5
        assert np.sum(np.isnan(data['association_separation_arcsec'])) == 4