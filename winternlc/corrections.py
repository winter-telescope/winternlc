"""
Module for applying nonlinearity correction and bad pixel masking to images.
"""

import os

import numpy as np


def rational_func(
    x,
    a0: float,
    a1: float,
    a2: float,
    a3: float,
    b0: float,
    b1: float,
    b2: float,
    b3: float,
):
    """
    A rational function with 4th order polynomials in the numerator and denominator.

    :param x: Input value
    :param a0: Coefficient for the constant term in the numerator
    :param a1: Coefficient for the linear term in the numerator
    :param a2: Coefficient for the quadratic term in the numerator
    :param a3: Coefficient for the cubic term in the numerator
    :param b0: Coefficient for the constant term in the denominator
    :param b1: Coefficient for the linear term in the denominator
    :param b2: Coefficient for the quadratic term in the denominator
    :param b3: Coefficient for the cubic term in the denominator

    :return: Rational function value
    """
    return (a0 + a1 * x + a2 * x**2 + a3 * x**3) / (
        1 + b0 * x + b1 * x**2 + b2 * x**3 + b3 * x**4
    )


def fitted_func(x, *coeffs):
    """
    Returns the value of the fitted rational function at the given input

    :param x: Input value
    :param coeffs: Coefficients for the rational function

    :return: Rational function value
    """
    return rational_func(x, *coeffs)


def nonlinearity_correction(
    image: np.ndarray, board_id: int, ext: int, cor_dir: str, cutoff: float
) -> np.ndarray:
    """
    Applies nonlinearity correction to an image using precomputed rational coefficients.

    :param image: Image to correct
    :param board_id: Board ID of the image
    :param ext: Extension number of the image
    :param cor_dir: Directory containing the correction files
    :param cutoff: Cutoff value for the image

    :return: Corrected image
    """
    rat_coeffs_path = os.path.join(
        cor_dir, f"rat_coeffs_board_{board_id}_ext_{ext-1}.npy"
    )

    if os.path.exists(rat_coeffs_path):
        rat_coeffs = np.load(rat_coeffs_path)

        # Apply cutoff
        image = np.clip(image, None, cutoff)

        # Normalize image by cutoff
        image = image / cutoff

        # Vectorized application of the fitted function
        rat_coeffs = rat_coeffs.reshape(-1, 8)
        image = fitted_func(image.flatten(), rat_coeffs.T).reshape(image.shape)

        # Scale back by cutoff
        image = cutoff * image
        return image
    else:
        raise FileNotFoundError(
            f"Rational coefficients file not found for board_id {board_id}"
        )


def mask_bad_pixels(
    image: np.ndarray, board_id: int, ext: int, cor_dir: str
) -> np.ndarray:
    """
    Applies a bad pixel mask to an image.

    :param image: Image to mask
    :param board_id: Board ID of the image
    :param ext: Extension number of the image
    :param cor_dir: Directory containing the correction files

    :return: Masked image (masked pixels set to NaN)
    """
    mask_path = os.path.join(
        cor_dir, f"bad_pixel_mask_board_{board_id}_ext_{ext-1}.npy"
    )

    if os.path.exists(mask_path):
        mask = np.load(mask_path)
        image[mask] = np.nan  # Set bad pixels to NaN
        return image
    else:
        raise FileNotFoundError(
            f"Bad pixel mask file not found for board_id {board_id}"
        )
