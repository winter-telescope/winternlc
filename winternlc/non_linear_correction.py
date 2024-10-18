"""
Module for applying nonlinearity correction and bad pixel masking to images.
"""

from pathlib import Path

import numpy as np

from winternlc.config import DEFAULT_CUTOFF, corrections_dir
from winternlc.rational import rational_func


def get_coeffs_path(board_id: int, cor_dir: Path = corrections_dir) -> Path:
    """
    Returns the path to the rational coefficients file for a given board ID.

    :param board_id: Board ID
    :param cor_dir: Directory containing the correction files

    :return: Path to the rational coefficients file
    """
    return cor_dir / f"rat_coeffs_board_{board_id}.npy"


def load_rational_coeffs(
    board_id: int, cor_dir: Path | str = corrections_dir
) -> np.ndarray:
    """
    Loads the rational coefficients for a given board ID.

    :param board_id: Board ID
    :param cor_dir: Directory containing the correction files

    :return: Rational coefficients
    """
    rat_coeffs_path = get_coeffs_path(board_id=board_id, cor_dir=cor_dir)

    if not rat_coeffs_path.exists():
        raise FileNotFoundError(
            f"Rational coefficients file not found at {rat_coeffs_path} "
            f"for board_id {board_id}"
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
    cor_dir: str | Path = corrections_dir,
    cutoff: float = DEFAULT_CUTOFF,
) -> np.ndarray:
    """
    Applies non-linearity correction to an image using precomputed rational coefficients.

    :param image: Image to correct
    :param board_id: Board ID of the image
    :param cor_dir: Directory containing the correction files
    :param cutoff: Cutoff value for the image

    :return: Corrected image
    """
    rat_coeffs = load_rational_coeffs(board_id=board_id, cor_dir=cor_dir)
    return apply_nonlinearity_correction(image, rat_coeffs, cutoff)
