#!/usr/bin/env python3
""" 
Test functions of spectrum_fits_utils.py
"""

import unittest
import numpy.testing as npt
from spectrum_fits_utils import *

class TestAdaptiveSmoothing(unittest.TestCase):

    def testGaussianWeightedIvar(self):
        # Test window size of 15
        ivar = np.zeros(15)
        ivar[0] = 0.5  # Leave the rest as 0
        npt.assert_array_equal([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                gaussianWeightedIvar(15, ivar, 7))

        ivar[14] = 0.5  # Now the first and last are 0.5
        npt.assert_array_equal([0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5],
                gaussianWeightedIvar(15, ivar, 7))

        ivar[1] = 0.5
        ivar[13] = 0.5
        gw = gaussianWeightedIvar(15, ivar, 7)
        self.assertEqual(gw[0], gw[14])
        self.assertEqual(gw[1], gw[13])
        self.assertTrue(gw[1] > gw[0])
        self.assertTrue(gw[13] > gw[14])

    def testConvolutionFlux(self):
        flux = np.zeros(15)
        flux[0] = 0.5
        flux[14] = 0.5
        ivar = np.zeros(15)
        ivar[0] = 0.5
        ivar[14] = 0.5
        self.assertEqual(0.5, convolutionFlux(flux, 15, ivar, 7))

    def testConvolutionFlux0ivar(self):
        flux = np.zeros(15)
        flux[0] = 0.5
        flux[14] = 0.5
        ivar = np.zeros(15)
        self.assertEqual(0.0, convolutionFlux(flux, 15, ivar, 7))

    def testConvolutionFlux0flux(self):
        flux = np.zeros(15)
        ivar = np.zeros(15)
        ivar[0] = 0.5
        ivar[14] = 0.5
        self.assertEqual(0.0, convolutionFlux(flux, 15, ivar, 7))

    def testAdaptiveSmoothing(self):
        flux = np.zeros(10) + 1
        flux[3:7] = 0
        ivar = np.zeros(10) + 1
        ivar[3:7] = 0
        result = adaptiveSmoothing(flux, ivar)
        expected_result = np.zeros(10) + 1
        npt.assert_array_almost_equal(expected_result, result)

    def testLogBinPixels(self):
        flux = np.zeros(1000) + 1
        # Assuming 100 bins:
        loglam = np.arange(10, 10.995, 0.01)  # 10 to 10.9
        self.assertEqual(100, loglam.size)
        self.assertAlmostEqual(10, loglam[0])
        self.assertAlmostEqual(10.99, loglam[99])
        # In linear lambda space:
        lam = np.linspace(math.exp(10-0.01), math.exp(10.99), 1000)
        self.assertEqual(1000, lam.size)
        bin_flux, bin_lam = logBinPixels(100, lam, flux, min_lam=math.exp(10-0.01), max_lam=math.exp(10.99))
        self.assertEqual(100, bin_lam.size)
        npt.assert_array_almost_equal(loglam, bin_lam)
        self.assertEqual(100, bin_flux.size)
        # All flux was 1.0, mean of each bin should be 1.0
        npt.assert_array_almost_equal(np.zeros(100)+1, bin_flux)

if __name__ == '__main__':
    unittest.main()
