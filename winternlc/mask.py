import numpy as np
from pathlib import Path


def get_mask_path(cor_dir: Path, board_id: int) -> Path:
    """
    Returns the path to the rational coefficients file for a given board ID.

    :param cor_dir: Directory containing the correction files
    :param board_id: Board ID

    :return: Path to the rational coefficients file
    """
    return cor_dir / f"bad_pixel_mask_board_{board_id}.npy"


def load_mask(cor_dir: Path | str, board_id: int) -> np.ndarray:
    """
    Loads the rational coefficients for a given board ID.

    :param cor_dir: Directory containing the correction files
    :param board_id: Board ID
    """
    mask_path = get_mask_path(cor_dir, board_id)

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Bad pixel mask file not found at {mask_path} "
            f"for board_id {board_id}"
        )

    return np.load(str(mask_path))


def apply_mask(
    image: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """
    Applies a bad pixel mask to an image.

    :param image: Image to mask

    :return: Masked image (masked pixels set to NaN)
    """
    image[mask] = np.nan  # Set bad pixels to NaN
    return image


def mask_single(
    image: np.ndarray, board_id: int, cor_dir: str
) -> np.ndarray:
    """
    Applies a bad pixel mask to an image.

    :param image: Image to mask
    :param board_id: Board ID of the image
    :param cor_dir: Directory containing the correction files

    :return: Masked image (masked pixels set to NaN)
    """
    mask = load_mask(cor_dir, board_id)
    return apply_mask(image, mask)
