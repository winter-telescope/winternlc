"""
Test for schedule
"""

import logging
import unittest

import numpy as np
from astropy.io import fits

from winternlc.config import (
    EXAMPLE_CORRECTED_IMG_PATH,
    EXAMPLE_IMG_PATH,
    EXAMPLE_MASKED_IMG_PATH,
)
from winternlc.mask import apply_mask_single
from winternlc.non_linear_correction import apply_nlc_single

logger = logging.getLogger(__name__)


class TestNLC(unittest.TestCase):
    """
    Class for testing API
    """

    def test_nlc_correction(self):
        """
        Test nlc correction on test image

        :return: None
        """
        logger.info("Testing nlc")

        with (
            fits.open(EXAMPLE_IMG_PATH) as hdul,
            fits.open(EXAMPLE_CORRECTED_IMG_PATH) as hdul_corrected,
        ):
            for ext in range(1, len(hdul)):
                header = hdul[ext].header
                image = hdul[ext].data
                logger.info(f"Processing extension {ext}")
                corrected_image = apply_nlc_single(image, header)

                comparison_image = hdul_corrected[ext].data

                ratio = corrected_image / comparison_image

                mask = np.isnan(corrected_image) | np.isnan(comparison_image)
                corrected_image[mask] = np.nan
                comparison_image[mask] = np.nan

                self.assertAlmostEqual(float(np.nanmin(ratio)), 1.0, delta=0.1)
                self.assertAlmostEqual(float(np.nanmax(ratio)), 1.0, delta=0.1)
                self.assertAlmostEqual(float(np.nanmean(ratio)), 1.0, delta=0.001)
                self.assertAlmostEqual(float(np.nanmedian(ratio)), 1.0, delta=0.001)
                self.assertAlmostEqual(float(np.nanstd(ratio)), 0.0, delta=0.01)

    def test_mask(self):
        """
        Test mask application on test image
        """

        logger.info("Testing mask")
        with (
            fits.open(EXAMPLE_IMG_PATH) as hdul,
            fits.open(EXAMPLE_MASKED_IMG_PATH) as hdul_corrected,
        ):
            for ext in range(1, len(hdul)):
                header = hdul[ext].header
                image = hdul[ext].data
                corrected_image = apply_mask_single(image, header)

                comparison_image = hdul_corrected[ext].data

                self.assertTrue(
                    np.allclose(corrected_image, comparison_image, equal_nan=True)
                )
