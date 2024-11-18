from pathlib import Path

import numpy as np
from astropy.io import fits

from winternlc.config import get_correction_dir
from winternlc.versions import get_nlc_version


def get_mask_path(board_id: int, version: str) -> Path:
    """
    Returns the path to the rational coefficients file for a given board ID.

    :param board_id: Board ID
    :param version: Version of the WinterNLC corrections

    :return: Path to the rational coefficients file
    """
    corrections_dir = get_correction_dir(version)
    return corrections_dir / f"bad_pixel_mask_board_{board_id}.npy"


def load_mask(
    board_id: int,
    version: str,
) -> np.ndarray:
    """
    Loads the rational coefficients for a given board ID.

    :param board_id: Board ID
    :param version: Version of the WinterNLC corrections

    :return: Rational coefficients
    """
    mask_path = get_mask_path(board_id=board_id, version=version)

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Bad pixel mask file not found at {mask_path} "
            f"for board_id {board_id} and version {version}"
        )

    return np.load(str(mask_path))


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies a bad pixel mask to an image.

    :param image: Image to mask
    :param mask: Bad pixel mask (boolean array)

    :return: Masked image (masked pixels set to NaN)
    """
    image[mask] = np.nan  # Set bad pixels to NaN
    return image


def mask_single(image: np.ndarray, board_id: int, version: str) -> np.ndarray:
    """
    Applies a bad pixel mask to an image.

    :param image: Image to mask
    :param board_id: Board ID of the image
    :param version: Version of the WinterNLC corrections

    :return: Masked image (masked pixels set to NaN)
    """
    mask = load_mask(board_id=board_id, version=version)
    return apply_mask(image, mask)


def apply_mask_single(
    image: np.ndarray,
    header: fits.header,
    version: str | None = None,
) -> np.ndarray:
    """
    Finds and applies the appropriate bad pixel mask.

    :param image: Image to mask
    :param header: Header of the image
    :param version: Version of the WinterNLC corrections

    :return: Masked image (masked pixels set to NaN)
    """
    board_id = header["BOARD_ID"]

    if version is None:
        version = get_nlc_version(header)

    return mask_single(image, board_id, version)
