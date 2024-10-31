import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from astropy.table import Table
from astropy.io import fits

from Extracting.Source_Extractor import Source_Extractor, make_nan, query_cone_ps1


class TestSourceExtractor(unittest.TestCase):

    @patch('source_extractor.fits.open')
    def setUp(self, mock_fits_open):
        # Mock the FITS file data
        mock_hdul = MagicMock()
        mock_hdul[0].data = np.random.rand(100, 100)
        mock_hdul[0].header = {'NFRAMES': 1, 'MEDFWHM': 2.5, 'MAGZP': 25.0}
        mock_fits_open.return_value = mock_hdul

        self.source_extractor = Source_Extractor('dummy.fits')

    def test_init(self):
        self.assertIsNotNone(self.source_extractor.image_data)
        self.assertIsNotNone(self.source_extractor.header)
        self.assertIsNotNone(self.source_extractor.wcs)

    def test_get_sources(self):
        with patch('source_extractor.sep.extract') as mock_extract:
            mock_extract.return_value = np.array([{'x': 50, 'y': 50}])
            sources = self.source_extractor.get_sources()
            self.assertIsNotNone(sources)

    def test_get_kron_mags(self):
        with patch('source_extractor.sep.kron_radius') as mock_kron_radius, \
             patch('source_extractor.sep.sum_ellipse') as mock_sum_ellipse, \
             patch('source_extractor.sep.sum_circle') as mock_sum_circle:
            mock_kron_radius.return_value = (np.array([3.0]), None)
            mock_sum_ellipse.return_value = (np.array([1000.0]), np.array([10.0]), None)
            mock_sum_circle.return_value = (np.array([1000.0]), np.array([10.0]), None)
            mag, magerr, circle_flag = self.source_extractor.get_kron_mags()
            self.assertIsNotNone(mag)
            self.assertIsNotNone(magerr)
            self.assertIsNotNone(circle_flag)

    def test_get_psf_mags(self):
        kron_mags = np.array([20.0])
        with patch('source_extractor.NDData') as mock_nddata, \
             patch('source_extractor.extract_stars') as mock_extract_stars, \
             patch('source_extractor.EPSFBuilder') as mock_epsf_builder, \
             patch('source_extractor.PSFPhotometry') as mock_psf_photometry:
            mock_epsf = MagicMock()
            mock_epsf_builder.return_value = (mock_epsf, None)
            mock_psf_photometry_instance = MagicMock()
            mock_psf_photometry.return_value = mock_psf_photometry_instance
            mock_psf_photometry_instance.return_value = Table({
                'flux_fit': [1000.0],
                'flux_err': [10.0],
                'flags': [0]
            })
            mag, magerr, flags = self.source_extractor.get_psf_mags(kron_mags)
            self.assertIsNotNone(mag)
            self.assertIsNotNone(magerr)
            self.assertIsNotNone(flags)

    def test_pix_to_ra_dec(self):
        x, y = np.array([50]), np.array([50])
        with patch.object(self.source_extractor.wcs, 'pixel_to_world') as mock_pixel_to_world:
            mock_coord = MagicMock()
            mock_coord.ra.deg = np.array([150.0])
            mock_coord.dec.deg = np.array([2.0])
            mock_pixel_to_world.return_value = mock_coord
            ra, dec = self.source_extractor.pix_to_ra_dec(x, y)
            self.assertEqual(ra[0], 150.0)
            self.assertEqual(dec[0], 2.0)

    def test_ra_dec_to_pix(self):
        ra, dec = np.array([150.0]), np.array([2.0])
        with patch.object(self.source_extractor.wcs, 'world_to_pixel') as mock_world_to_pixel:
            mock_world_to_pixel.return_value = (np.array([50]), np.array([50]))
            x, y = self.source_extractor.ra_dec_to_pix(ra, dec)
            self.assertEqual(x[0], 50)
            self.assertEqual(y[0], 50)

    def test_get_coord_range(self):
        with patch.object(self.source_extractor.wcs, 'pixel_to_world') as mock_pixel_to_world:
            mock_coord_start = MagicMock()
            mock_coord_end = MagicMock()
            mock_coord_start.ra.deg = 149.0
            mock_coord_start.dec.deg = 1.0
            mock_coord_end.ra.deg = 151.0
            mock_coord_end.dec.deg = 3.0
            mock_pixel_to_world.side_effect = [mock_coord_start, mock_coord_end]
            ra_range, dec_range = self.source_extractor.get_coord_range()
            self.assertEqual(ra_range, (149.0, 151.0))
            self.assertEqual(dec_range, (1.0, 3.0))

    def test_get_data_table(self):
        with patch.object(self.source_extractor, 'get_sources') as mock_get_sources, \
             patch.object(self.source_extractor, 'get_kron_mags') as mock_get_kron_mags, \
             patch.object(self.source_extractor, 'get_psf_mags') as mock_get_psf_mags, \
             patch.object(self.source_extractor, 'get_sources_ra_dec') as mock_get_sources_ra_dec:
            mock_get_sources.return_value = None
            mock_get_kron_mags.return_value = (np.array([20.0]), np.array([0.1]), np.array([0]))
            mock_get_psf_mags.return_value = (np.array([19.5]), np.array([0.1]), np.array([0]))
            mock_get_sources_ra_dec.return_value = np.array([[150.0, 2.0]])
            data_table = self.source_extractor.get_data_table()
            self.assertIsInstance(data_table, Table)
            self.assertIn('ra', data_table.colnames)
            self.assertIn('dec', data_table.colnames)

class TestMakeNan(unittest.TestCase):

    def test_make_nan(self):
        catalog = Table({
            'col1': ['-999', '10', 'nan'],
            'col2': ['--', '5', 'n']
        })
        cleaned_catalog = make_nan(catalog)
        self.assertTrue(np.isnan(cleaned_catalog['col1'][0]))
        self.assertEqual(cleaned_catalog['col1'][1], '10')
        self.assertTrue(np.isnan(cleaned_catalog['col1'][2]))
        self.assertTrue(np.isnan(cleaned_catalog['col2'][0]))
        self.assertEqual(cleaned_catalog['col2'][1], '5')
        self.assertTrue(np.isnan(cleaned_catalog['col2'][2]))

class TestQueryConePS1(unittest.TestCase):

    @patch('source_extractor.MastCasJobs')
    def test_query_cone_ps1(self, mock_mast_cas_jobs):
        mock_jobs_instance = MagicMock()
        mock_mast_cas_jobs.return_value = mock_jobs_instance
        mock_jobs_instance.quick.return_value = {
            'objID': [1, 2],
            'raStack': [150.0, 151.0],
            'decStack': [2.0, 3.0],
            'primaryDetection': [1, 1],
            'ps_score': [0.95, 0.85]
        }
        ra_deg, dec_deg, search_radius_arcmin = 150.0, 2.0, 1.0
        result_table = query_cone_ps1(ra_deg, dec_deg, search_radius_arcmin)
        self.assertIsInstance(result_table, Table)
        self.assertEqual(len(result_table), 2)
        self.assertIn('objID_3pi', result_table.colnames)

if __name__ == '__main__':
    unittest.main()
