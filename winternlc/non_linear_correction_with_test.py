"""
Module for applying nonlinearity correction and bad pixel masking to images.
"""

import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from numpy.polynomial.polynomial import polyval
from packaging.version import parse

from winternlc.config import (
    CROSSOVER_POINT,
    DEFAULT_CUTOFF,
    ORDER_LOWER,
    ORDER_UPPER,
    get_correction_dir,
)
from winternlc.get_corrections import check_for_files
from winternlc.rational import rational_func
from winternlc.versions import get_nlc_version

logger = logging.getLogger(__name__)


def get_coeffs_path(board_id: int, version: str) -> Path:
    """
    Returns the path to the coefficients file for a given board ID.

    :param board_id: Board ID
    :param version: Version of the WinterNLC corrections

    :return: Path to the appropriate coefficients file
    """
    corrections_dir = get_correction_dir(version)

    v = parse(version)

    fname = (
        f"corr_coeffs_board_{board_id}.npy"
        if v > parse("v1.1")
        else f"rat_coeffs_board_{board_id}.npy"
    )

    return corrections_dir / fname


def load_coeffs(board_id: int, version: str) -> np.ndarray:
    """
    Loads the coefficients for a given board ID.

    :param board_id: Board ID
    :param version: Version of the WinterNLC corrections

    :return: Coefficients
    """

    coeffs_path = get_coeffs_path(board_id=board_id, version=version)

    if not coeffs_path.exists():
        raise FileNotFoundError(
            f"Coefficients file not found at {coeffs_path} "
            f"for board_id {board_id} and version {version}"
        )

    return np.load(str(coeffs_path))


def pixelwise_polyval(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    result = np.zeros_like(x, dtype=np.float64)
    for j in range(coeffs.shape[1]):
        result = result * x + coeffs[:, j]
    return result


def polyval_rows(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    result = np.empty_like(x, dtype=np.float64)
    result[:] = 0.0
    for c in coeffs.T:
        result *= x
        result += c
    return result


def apply_nonlinearity_two_segment(
    image: np.ndarray,
    coeffs: np.ndarray,
    crossover_point: float,
    ceiling: float | None = None,
) -> np.ndarray:
    y, x = image.shape
    flat_image = image.ravel()
    flat_coeffs = coeffs.reshape(-1, ORDER_LOWER + 1 + ORDER_UPPER + 1)

    split_idx = ORDER_LOWER + 1
    coeffs_lower = flat_coeffs[:, :split_idx]
    coeffs_upper = flat_coeffs[:, split_idx:]  # no second reshape

    corrected = np.empty_like(flat_image, dtype=np.float64)

    mask_lower = flat_image < crossover_point
    mask_upper = ~mask_lower

    if np.any(mask_lower):
        corrected[mask_lower] = polyval_rows(
            flat_image[mask_lower], coeffs_lower[mask_lower]
        )
    if np.any(mask_upper):
        corrected[mask_upper] = polyval_rows(
            flat_image[mask_upper], coeffs_upper[mask_upper]
        )

    if ceiling is not None:
        clip = (corrected > ceiling) | (flat_image > ceiling)
        corrected[clip] = flat_image[clip]

    return corrected.reshape(y, x)


def apply_nonlinearity_correction(
    image: np.ndarray,
    coefficients: np.ndarray,
    cutoff: float = DEFAULT_CUTOFF,
    version: str = "v1.0",
    crossover_point: float = CROSSOVER_POINT,
    ceiling: float | None = None,
) -> np.ndarray:
    """
    Applies appropriate nonlinearity correction based on coefficient format and version.
    """
    v = parse(version)

    if v <= parse("v1.1"):
        image = np.clip(image, None, cutoff)
        image_norm = image / cutoff
        coeffs = coefficients.reshape(-1, 8).T
        corrected = rational_func(image_norm.flatten(), *coeffs).reshape(image.shape)
        return cutoff * corrected

    image = image.astype(np.float64, copy=False)

    if coefficients.ndim == 2:
        flat_image = image.flatten()
        flat_coeffs = coefficients.reshape(-1, coefficients.shape[-1])
        return pixelwise_polyval(flat_image, flat_coeffs).reshape(image.shape)

    elif coefficients.ndim == 3:
        return apply_nonlinearity_two_segment(
            image, coefficients, crossover_point, ceiling
        )

    else:
        raise ValueError(f"Unsupported coefficient shape: {coefficients.shape}")


def nlc_single(
    image: np.ndarray,
    board_id: int,
    version: str,
    cutoff: float = DEFAULT_CUTOFF,
) -> np.ndarray:
    """
    Applies non-linearity correction to an image using precomputed coefficients.

    :param image: Image to correct
    :param board_id: Board ID of the image
    :param version: Version of the WinterNLC corrections
    :param cutoff: Cutoff value for the image

    :return: Corrected image
    """
    coeffs = load_coeffs(board_id=board_id, version=version)
    return apply_nonlinearity_correction(image, coeffs, cutoff, version)


def apply_nlc_single(
    image: np.ndarray,
    header: fits.header,
    version: str | None = None,
    cutoff: float = DEFAULT_CUTOFF,
) -> np.ndarray:
    """
    Finds appropriate non-linearity correction for an image, and applies them.

    :param image: Image to correct
    :param header: Image header
    :param version: Version of the WinterNLC corrections (default is None)
    :param cutoff: Cutoff value for the image

    :return: Corrected image
    """
    board_id = header.get("BOARD_ID", None)
    if version is None:
        version = get_nlc_version(header)

    try:
        return nlc_single(image, board_id, version, cutoff)
    except FileNotFoundError as e:
        logger.warning(
            f"Error applying NLC to image: {e} \n " f"Checking that files are present"
        )
        check_for_files()
        return nlc_single(image, board_id, version, cutoff)


# # # test
# # # TODO DELETE

# def test_apply_nlc_single(EXAMPLE_IMG_PATH, CORRECTED_IMG_PATH):
#     """
#     Test function for apply_nlc_single with multi-extension FITS files.
#     """
#     # Open the input and output FITS files
#     with fits.open(EXAMPLE_IMG_PATH) as hdul:
#         # Create a new HDU list for the corrected image
#         corrected_hdul = fits.HDUList()

#         # Process each extension
#         for ext in range(len(hdul)):
#             if hdul[ext].data is None or "BOARD_ID" not in hdul[ext].header:
#                 corrected_hdul.append(hdul[ext])
#                 continue
#             header = hdul[ext].header
#             image = hdul[ext].data
#             logger.info(f"Processing extension {ext}")

#             # Apply non-linearity correction
#             corrected_image = apply_nlc_single(image, header)

#             # Apply bad pixel mask if available
#             board_id = header["BOARD_ID"]
#             mask_path = Path(get_correction_dir("v2.0")) / f"bad_pixel_mask_board_{board_id}.npy"
#             # if mask_path.exists():
#             #     mask = np.load(mask_path)
#             #     if mask.shape == corrected_image.shape:
#             #         corrected_image[mask] = np.nan
#             #     else:
#             #         logger.warning(f"Mask shape mismatch for ext {ext}")
#             # else:
#             #     logger.warning(f"No mask found for ext {ext} (expected {mask_path})")

#             # Strip scaling-related header keywords to avoid display issues in DS9
#             header = header.copy()
#             for key in ['BSCALE', 'BZERO', 'BUNIT']:
#                 header.pop(key, None)
#             if ext == 0:
#                 corrected_hdu = fits.PrimaryHDU(corrected_image, header)
#             else:
#                 corrected_hdu = fits.ImageHDU(corrected_image, header)
#             corrected_hdul.append(corrected_hdu)

#         # Save the corrected HDU list to the output file
#         corrected_hdul.writeto(CORRECTED_IMG_PATH, overwrite=True)

# if __name__ == "__main__":
#     EXAMPLE_IMG_PATH = "/Users/frostig/Downloads/example_science_image_mef.fits"
#     CORRECTED_IMG_PATH = "/Users/frostig/Downloads/corrected_example_science_image_mef.fits"
#     test_apply_nlc_single(EXAMPLE_IMG_PATH, CORRECTED_IMG_PATH)
#     print(f"✔ Corrected image saved to {CORRECTED_IMG_PATH}")


import os


def apply_mask_single(image: np.ndarray, header, version: str = "v2.0") -> np.ndarray:
    """
    Applies the bad pixel mask to the image by setting bad pixels to NaN.

    :param image: The input image (2D array)
    :param header: FITS header with BOARD_ID
    :param version: Correction version
    :return: Masked image (same shape with bad pixels set to NaN)
    """
    board_id = header.get("BOARD_ID", None)
    if board_id is None:
        raise ValueError("BOARD_ID not found in FITS header.")

    mask_path = (
        Path(get_correction_dir(version)) / f"bad_pixel_mask_board_{board_id}.npy"
    )

    if not mask_path.exists():
        raise FileNotFoundError(f"Bad pixel mask not found: {mask_path}")

    mask = np.load(mask_path)
    if mask.shape != image.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {image.shape}."
        )

    masked_image = image.astype(np.float64).copy()
    masked_image[mask] = np.nan
    return masked_image


def test_apply_nlc_single(
    EXAMPLE_IMG_PATH, CORRECTED_IMG_PATH, MASKED_IMG_PATH, version="v2.0"
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
            import time

            start = time.perf_counter()
            corrected_image = apply_nlc_single(image, header, version=version)
            elapsed = time.perf_counter() - start
            print(f"NLC correction took {elapsed:.3f} seconds for ext {ext}")
            # corrected_image = apply_nlc_single(image, header, version=version)

            # Apply mask only
            masked_image = apply_mask_single(image, header, version=version)

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
    EXAMPLE_IMG_PATH = "/Users/frostig/Downloads/example_science_image_mef.fits"
    CORRECTED_IMG_PATH = (
        "/Users/frostig/Downloads/corrected_example_science_image_mef.fits"
    )
    MASKED_IMG_PATH = "/Users/frostig/Downloads/masked_example_science_image_mef.fits"

    test_apply_nlc_single(EXAMPLE_IMG_PATH, CORRECTED_IMG_PATH, MASKED_IMG_PATH)
