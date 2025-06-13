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
from winternlc.zenodo import LATEST_ZENODO_VERSION

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
    version: str = LATEST_ZENODO_VERSION,
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
