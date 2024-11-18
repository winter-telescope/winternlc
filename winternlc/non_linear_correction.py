"""
Module for applying nonlinearity correction and bad pixel masking to images.
"""

from pathlib import Path

import numpy as np
from astropy.io import fits

from winternlc.config import DEFAULT_CUTOFF, get_correction_dir
from winternlc.rational import rational_func
from winternlc.versions import get_nlc_version


def get_coeffs_path(board_id: int, version: str) -> Path:
    """
    Returns the path to the rational coefficients file for a given board ID.

    :param board_id: Board ID
    :param str version: Version of the WinterNLC corrections

    :return: Path to the rational coefficients file
    """

    corrections_dir = get_correction_dir(version)

    return corrections_dir / f"rat_coeffs_board_{board_id}.npy"


def load_rational_coeffs(board_id: int, version: str) -> np.ndarray:
    """
    Loads the rational coefficients for a given board ID.

    :param board_id: Board ID
    :param version: Version of the WinterNLC corrections

    :return: Rational coefficients
    """

    rat_coeffs_path = get_coeffs_path(board_id=board_id, version=version)

    if not rat_coeffs_path.exists():
        raise FileNotFoundError(
            f"Rational coefficients file not found at {rat_coeffs_path} "
            f"for board_id {board_id} and version {version}"
        )

    return np.load(str(rat_coeffs_path))


def apply_nonlinearity_correction(
    image: np.ndarray, coefficients: np.ndarray, cutoff: float = DEFAULT_CUTOFF
) -> np.ndarray:
    """
    Applies non-linearity correction to an image using precomputed rational coefficients.

    :param image: Image to correct
    :param coefficients: Rational coefficients for the correction
    :param cutoff: Cutoff value for the image
    """

    # Apply cutoff
    image = np.clip(image, None, cutoff)

    # Normalize image by cutoff
    image = image / cutoff

    # Vectorized application of the fitted function
    coefficients = coefficients.reshape(-1, 8)
    image = rational_func(image.flatten(), *coefficients.T).reshape(image.shape)

    # Scale back by cutoff
    image = cutoff * image
    return image


def nlc_single(
    image: np.ndarray,
    board_id: int,
    version: str,
    cutoff: float = DEFAULT_CUTOFF,
) -> np.ndarray:
    """
    Applies non-linearity correction to an image using precomputed rational coefficients.

    :param image: Image to correct
    :param board_id: Board ID of the image
    :param version: Version of the WinterNLC corrections
    :param cutoff: Cutoff value for the image

    :return: Corrected image
    """
    rat_coeffs = load_rational_coeffs(board_id=board_id, version=version)
    return apply_nonlinearity_correction(image, rat_coeffs, cutoff)


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

    return nlc_single(image, board_id, version, cutoff)
