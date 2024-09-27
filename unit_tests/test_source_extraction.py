import os
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from Source_Extractor import Source_Extractor, img_flux_to_ab_mag, get_pstar_sources, make_nan, query_cone_ps1


# Sample FITS file for testing
@pytest.fixture
def sample_fits_file(tmp_path):
    data = np.random.rand(600, 600)
    hdu = fits.PrimaryHDU(data)
    hdu.header = fits.Header([
        ('SIMPLE', True),
        ('BITPIX', -64),
        ('NAXIS', 2),
        ('NAXIS1', data.shape[1]),
        ('NAXIS2', data.shape[0]),
        ('NFRAMES', 100),
        ('MEDFWHM', 2.0),
        ('MAGZP', 25.0),
        ('RADESYS', 'ICRS'),
        ('CTYPE1', 'RA---TAN'),
        ('CRVAL1', 1.585656199000E+02),
        ('CRPIX1', -7.765000000000000E+02),
        ('CD1_1', 2.777777372621E-04),
        ('CD1_2', 1.500289800209E-07),
        ('CTYPE2', 'DEC--TAN'),
        ('CRVAL2', 5.958793500000E+00),
        ('CRPIX2', 4.605000000000000E+02),
        ('CD2_1', 1.500289800209E-07),
        ('CD2_2', -2.777777372621E-04)
    ])
    hdul = fits.HDUList([hdu])
    fits_path = tmp_path / "sample.fits"
    hdul.writeto(fits_path)
    return str(fits_path)

def test_img_flux_to_ab_mag():
    flux = np.array([10, 20, 30])
    zero_point = 25.0
    mag = img_flux_to_ab_mag(flux, zero_point)
    assert np.allclose(mag, [22.5, 21.75, 21.31], atol=0.01)

def test_source_extractor_init(sample_fits_file):
    se = Source_Extractor(sample_fits_file)
    assert se.image_data is not None
    assert isinstance(se.wcs, WCS)

def test_source_extractor_bkg(sample_fits_file):
    se = Source_Extractor(sample_fits_file)
    bkg = se.bkg
    assert bkg is not None

def test_source_extractor_get_sources(sample_fits_file):
    se = Source_Extractor(sample_fits_file)
    sources = se.get_sources()
    assert sources is not None

def test_source_extractor_get_kron_mags(sample_fits_file):
    se = Source_Extractor(sample_fits_file)
    se.get_sources()
    photometry = se.get_kron_mags()
    assert 'kronmag' in photometry.dtype.names

def test_source_extractor_get_sources_ra_dec(sample_fits_file):
    se = Source_Extractor(sample_fits_file)
    se.get_sources()
    coords = se.get_sources_ra_dec()
    assert coords.shape[1] == 2

def test_source_extractor_store_coords(sample_fits_file, tmp_path):
    se = Source_Extractor(sample_fits_file)
    se.get_sources()
    coord_file = tmp_path / "coords.txt"
    se.store_coords(str(coord_file))
    assert os.path.exists(coord_file)

def test_source_extractor_plot_segmap(sample_fits_file, tmp_path):
    se = Source_Extractor(sample_fits_file)
    se.get_sources()
    segmap_file = tmp_path / "segmap.png"
    se.plot_segmap(fpath=str(segmap_file))
    assert os.path.exists(segmap_file)

def test_get_pstar_sources(mocker):
    ra_range = (10.0, 20.0)
    dec_range = (-10.0, 10.0)
    mock_jobs = mocker.Mock()
    mocker.patch('source_extraction.MastCasJobs', return_value=mock_jobs)
    mock_jobs.quick.return_value = Table({'ra': [15.0], 'dec': [0.0]})    
    sources = get_pstar_sources(ra_range, dec_range)

    assert isinstance(sources, Table)
    mock_jobs.quick.assert_called_once_with(mocker.ANY, task_name="PanSTARRS_DR2_RA_DEC_Query")

def test_query_cone_ps1():
    ra_deg = 10.0
    dec_deg = 10.0
    search_radius_arcmin = 1.0
    result = query_cone_ps1(ra_deg, dec_deg, search_radius_arcmin)
    assert isinstance(result, Table) or result is None
