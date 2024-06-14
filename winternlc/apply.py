import os
import time
from pathlib import Path

from astropy.io import fits

from winternlc.config import (
    corrections_dir,
    DEFAULT_CUTOFF,
    EXAMPLE_IMG_PATH,
    example_data_dir,
)
from winternlc.mask import mask_single
from winternlc.non_linear_correction import nlc_single


def apply_nlc_mef(
    fits_file: str | Path, cor_dir: str | Path, save_dir: str | Path, cutoff: float
):
    """
    Process a multi-extension FITS file, applying nonlinearity correction to each
    extension, and write the corrected FITS file to disk.

    :param fits_file: Path to the FITS file
    :param cor_dir: Directory containing the correction files
    :param save_dir: Directory to save the corrected FITS file
    :param cutoff: Cutoff value for the image

    :return: None
    """
    with fits.open(fits_file) as hdul:
        for ext in range(1, len(hdul)):
            header = hdul[ext].header
            image = hdul[ext].data
            board_id = header.get("BOARD_ID", None)
            if board_id is not None:
                print(f"Processing extension {ext} with BOARD_ID {board_id}")
                start = time.time()
                corrected_image = nlc_single(image, board_id, cor_dir, cutoff)
                end = time.time()
                print(f"took {end-start} s to execute")
                hdul[ext].data = corrected_image
            else:
                print(f"Skipping extension {ext} as it does not have a BOARD_ID")

        corrected_fits_file = os.path.join(
            save_dir, "corrected_" + os.path.basename(fits_file)
        )
        hdul.writeto(corrected_fits_file, overwrite=True)
        print(f"Corrected FITS file saved to {corrected_fits_file}")


def apply_mask_mef(fits_file: str | Path, cor_dir: str | Path, save_dir: str | Path):
    """
    Process a multi-extension FITS file, applying nonlinearity correction to each
    extension, and write the corrected FITS file to disk.

    :param fits_file: Path to the FITS file
    :param cor_dir: Directory containing the correction files
    :param save_dir: Directory to save the corrected FITS file

    :return: None
    """
    with fits.open(fits_file) as hdul:
        for ext in range(1, len(hdul)):
            header = hdul[ext].header
            image = hdul[ext].data
            board_id = header.get("BOARD_ID", None)
            if board_id is not None:
                print(f"Masking extension {ext} with BOARD_ID {board_id}")
                start = time.time()
                corrected_image = mask_single(image, board_id, cor_dir)
                end = time.time()
                print(f"took {end-start} s to execute")
                hdul[ext].data = corrected_image
            else:
                print(f"Skipping extension {ext} as it does not have a BOARD_ID")

        corrected_fits_file = os.path.join(
            save_dir, "masked_" + os.path.basename(fits_file)
        )
        hdul.writeto(corrected_fits_file, overwrite=True)
        print(f"Corrected FITS file saved to {corrected_fits_file}")


if __name__ == "__main__":
    apply_nlc_mef(
        EXAMPLE_IMG_PATH, corrections_dir, example_data_dir, DEFAULT_CUTOFF
    )
    apply_mask_mef(EXAMPLE_IMG_PATH, corrections_dir, example_data_dir)
