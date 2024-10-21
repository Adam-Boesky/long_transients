from unittest.mock import Mock, patch

from astropy.table import Table
from Extracting.Catalogs import Catalog, PSTARR_Catalog, ZTF_Catalog, associate_tables_by_coordinates


def test_catalog_init():
    catalog = Catalog()
    assert catalog.catalog_name is None
    assert catalog.column_map is None
    assert catalog.ra_range is None
    assert catalog.dec_range is None
    assert catalog._data is None

def test_catalog_data():
    catalog = Catalog()
    catalog.get_data = Mock(return_value=Table({'col1': [1, 2], 'col2': [3, 4]}))
    catalog.rename_columns = Mock()
    data = catalog.data
    assert isinstance(data, Table)
    assert catalog.rename_columns.called

def test_catalog_rename_columns():
    catalog = Catalog()
    catalog._data = Table({'old_col': [1, 2]})
    catalog.column_map = {'old_col': 'new_col'}
    catalog.rename_columns()
    assert 'new_col' in catalog._data.colnames

def test_catalog_get_coordinate_range():
    catalog = Catalog()
    catalog.ra_range = (10.0, 20.0)
    catalog.dec_range = (-10.0, 10.0)
    ra_range, dec_range = catalog.get_coordinate_range()
    assert ra_range == (10.0, 20.0)
    assert dec_range == (-10.0, 10.0)

@patch('Catalogs.get_credentials', return_value=('wsid', 'password'))
@patch('Catalogs.MastCasJobs')
def test_pstarr_catalog_get_data(mock_mastcasjobs, mock_get_credentials):
    mock_jobs = Mock()
    mock_mastcasjobs.return_value = mock_jobs
    mock_jobs.quick.return_value = Table({'objID': [1], 'raMean': [15.0], 'decMean': [0.0]})

    ra_range = (10.0, 20.0)
    dec_range = (-10.0, 10.0)
    pstarr_catalog = PSTARR_Catalog(ra_range, dec_range)
    data = pstarr_catalog.get_data()

    assert isinstance(data, Table)
    mock_jobs.quick.assert_called_once()

@patch('Catalogs.Source_Extractor')
def test_ztf_catalog_init(mock_source_extractor):
    mock_se_instance = Mock()
    mock_source_extractor.return_value = mock_se_instance
    mock_se_instance.get_coord_range.return_value = ((10.0, 20.0), (-10.0, 10.0))

    ztf_catalog = ZTF_Catalog(ra=15.0, dec=0.0)
    assert ztf_catalog.catalog_name == 'ZTF'
    assert 'g' in ztf_catalog.sextractors
    assert ztf_catalog.ra_range == (10.0, 20.0)
    assert ztf_catalog.dec_range == (-10.0, 10.0)

@patch('Catalogs.get_credentials', return_value=('wsid', 'password'))
@patch('Catalogs.MastCasJobs')
def test_get_pstar_sources(mock_mastcasjobs, mock_get_credentials):
    mock_jobs = Mock()
    mock_mastcasjobs.return_value = mock_jobs
    mock_jobs.quick.return_value = Table({'objID': [1], 'raMean': [15.0], 'decMean': [0.0]})

    ra_range = (10.0, 20.0)
    dec_range = (-10.0, 10.0)
    sources = get_pstar_sources(ra_range, dec_range)

    assert isinstance(sources, Table)
    mock_jobs.quick.assert_called_once()

def test_associate_tables_by_coordinates():
    table1 = Table({'ra': [10.0, 15.0], 'dec': [0.0, 5.0]})
    table2 = Table({'ra': [10.000000001, 15.000000001], 'dec': [0.000000001, 5.000000001], 'mag': [20.0, 21.0]})

    associated_table = associate_tables_by_coordinates(table1, table2, max_sep=1.0)

    assert isinstance(associated_table, Table)
    assert 'mag' in associated_table.colnames
    assert len(associated_table) == 2
