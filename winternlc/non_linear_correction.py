"""
Module for applying nonlinearity correction and bad pixel masking to images.
"""

import os

import numpy as np

from pathlib import Path
from winternlc.config import DEFAULT_CUTOFF
from winternlc.rational import rational_func


def get_coeffs_path(cor_dir: Path, board_id: int) -> Path:
    """
    Returns the path to the rational coefficients file for a given board ID.

    :param cor_dir: Directory containing the correction files
    :param board_id: Board ID

    :return: Path to the rational coefficients file
    """
    return cor_dir / f"rat_coeffs_board_{board_id}.npy"


def load_rational_coeffs(cor_dir: Path | str, board_id: int) -> np.ndarray:
    """
    Loads the rational coefficients for a given board ID.

    :param cor_dir: Directory containing the correction files
    :param board_id: Board ID
    """
    rat_coeffs_path = get_coeffs_path(cor_dir, board_id)

    if not rat_coeffs_path.exists():
        raise FileNotFoundError(
            f"Rational coefficients file not found at {rat_coeffs_path} "
            f"for board_id {board_id}"
        )

    return np.load(str(rat_coeffs_path))


def apply_nonlinearity_correction(
        image: np.ndarray, coeffs: np.ndarray, cutoff: float = DEFAULT_CUTOFF
) -> np.ndarray:
    """
    Applies nonlinearity correction to an image using precomputed rational coefficients.

    :param image: Image to correct
    :param coeffs: Rational coefficients for the correction
    :param cutoff: Cutoff value for the image
    """

    # Apply cutoff
    image = np.clip(image, None, cutoff)

    # Normalize image by cutoff
    image = image / cutoff

    # Vectorized application of the fitted function
    coeffs = coeffs.reshape(-1, 8)
    image = rational_func(image.flatten(), *coeffs.T).reshape(image.shape)

    # Scale back by cutoff
    image = cutoff * image
    return image


def nlc_single(
    image: np.ndarray, board_id: int, cor_dir: str, cutoff: float
) -> np.ndarray:
    """
    Applies nonlinearity correction to an image using precomputed rational coefficients.

    :param image: Image to correct
    :param board_id: Board ID of the image
    :param cor_dir: Directory containing the correction files
    :param cutoff: Cutoff value for the image

    :return: Corrected image
    """
    rat_coeffs = load_rational_coeffs(cor_dir, board_id)
    return apply_nonlinearity_correction(image, rat_coeffs, cutoff)
