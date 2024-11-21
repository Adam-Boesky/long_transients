import unittest
import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from Source_Analysis.Sources import Postage_Stamp, ZTF_Postage_Stamp

class TestPostageStamp(unittest.TestCase):

    def setUp(self):
        self.ra = 150.0
        self.dec = 2.0
        self.stamp_width_arcsec = 50
        self.bands = ['g', 'r', 'i']
        self.postage_stamp = Postage_Stamp(self.ra, self.dec, self.stamp_width_arcsec, self.bands)
        self.ztf_postage_stamp = ZTF_Postage_Stamp(self.ra, self.dec, self.stamp_width_arcsec, self.bands)

        # Mock WCS
        self.mock_wcs = WCS(naxis=2)
        self.mock_wcs.wcs.crpix = [0, 0]
        self.mock_wcs.wcs.cdelt = np.array([-0.0002777777778, 0.0002777777778])
        self.mock_wcs.wcs.crval = [self.ra, self.dec]
        self.mock_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        self.postage_stamp._WCSs = {band: self.mock_wcs for band in self.bands}

    def test_pix_to_ra_dec(self):
        x, y = np.array([50]), np.array([50])
        ra, dec = self.postage_stamp.pix_to_ra_dec(x, y, 'g')
        self.assertIsInstance(ra, np.ndarray)
        self.assertIsInstance(dec, np.ndarray)
        self.assertEqual(ra.shape, (1,))
        self.assertEqual(dec.shape, (1,))

    def test_ra_dec_to_pix(self):
        ra, dec = np.array([150.0]), np.array([2.0])
        x, y = self.postage_stamp.ra_dec_to_pix(ra, dec, 'g')
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(x.shape, (1,))
        self.assertEqual(y.shape, (1,))

    def test_transform_pix_coords(self):
        x, y = 10, 20
        x_transformed, y_transformed = self.postage_stamp.transform_pix_coords(x, y, 'g')
        self.assertEqual(x, x_transformed)
        self.assertEqual(y, y_transformed)

    def test_origpix_to_current(self):
        x_orig, y_orig = 10, 20
        x_current, y_current = self.postage_stamp.origpix_to_current(x_orig, y_orig, 'g')
        self.assertEqual(x_orig, x_current)
        self.assertEqual(y_orig, y_current)

    def test_currentpix_to_orig(self):
        x_current, y_current = 10, 20
        x_orig, y_orig = self.postage_stamp.currentpix_to_orig(x_current, y_current, 'g')
        self.assertEqual(x_current, x_orig)
        self.assertEqual(y_current, y_orig)

    def test_ztf_postage_stamp(self):
        x_current, y_current = 10, 20
        ra, dec = self.ztf_postage_stamp.pix_to_ra_dec(x_current, y_current, band='g')
        x_trans, y_trans = self.ztf_postage_stamp.ra_dec_to_pix(ra, dec, band='g')
        self.assertAlmostEqual(x_current, x_trans)
        self.assertAlmostEqual(y_current, y_trans)


if __name__ == '__main__':
    unittest.main()
