import logging
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

from winternlc.config import (
    DEFAULT_CUTOFF,
    EXAMPLE_CORRECTED_IMG_PATH,
    EXAMPLE_IMG_PATH,
    EXAMPLE_MASKED_IMG_PATH,
    get_correction_dir,
)
from winternlc.mask import apply_mask_single
from winternlc.non_linear_correction import apply_nlc_single

logger = logging.getLogger(__name__)


def test_apply_nlc_single(
    EXAMPLE_IMG_PATH,
    CORRECTED_IMG_PATH,
    MASKED_IMG_PATH,
):
    """
    Creates two output files:
    1. Non-linearity corrected FITS file (CORRECTED_IMG_PATH)
    2. Masked (but not corrected) FITS file (MASKED_IMG_PATH)
    """
    with fits.open(EXAMPLE_IMG_PATH) as hdul:
        corrected_hdul = fits.HDUList()
        masked_hdul = fits.HDUList()

        for ext in range(len(hdul)):
            hdu = hdul[ext]
            if hdu.data is None or "BOARD_ID" not in hdu.header:
                corrected_hdul.append(hdu)
                masked_hdul.append(hdu)
                continue

            header = hdu.header.copy()
            image = hdu.data
            board_id = header["BOARD_ID"]
            logger.info(f"Processing extension {ext}, BOARD_ID {board_id}")

            # Apply correction
            corrected_image = apply_nlc_single(image, header)

            # Apply mask only
            masked_image = apply_mask_single(image, header)

            # Clean up headers
            for key in ["BSCALE", "BZERO", "BUNIT"]:
                header.pop(key, None)

            # Add corrected image
            if ext == 0:
                corrected_hdu = fits.PrimaryHDU(corrected_image, header)
                masked_hdu = fits.PrimaryHDU(masked_image, header)
            else:
                corrected_hdu = fits.ImageHDU(corrected_image, header)
                masked_hdu = fits.ImageHDU(masked_image, header)

            corrected_hdul.append(corrected_hdu)
            masked_hdul.append(masked_hdu)

        os.makedirs(os.path.dirname(CORRECTED_IMG_PATH), exist_ok=True)
        corrected_hdul.writeto(CORRECTED_IMG_PATH, overwrite=True)
        masked_hdul.writeto(MASKED_IMG_PATH, overwrite=True)

        print(f"✔ Corrected image saved to {CORRECTED_IMG_PATH}")
        print(f"✔ Masked image saved to {MASKED_IMG_PATH}")


if __name__ == "__main__":
    test_apply_nlc_single(
        EXAMPLE_IMG_PATH, EXAMPLE_CORRECTED_IMG_PATH, EXAMPLE_MASKED_IMG_PATH
    )
