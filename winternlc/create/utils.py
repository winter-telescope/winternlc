"""
Utility functions for the WinterNLC project.
"""

import os
from pathlib import Path

import numpy as np
from astropy.io import fits


def get_exposure_time(filename: str | Path):
    """
    Extracts the exposure time from the filename.
    Assumes the exposure time is between '_exp_' and the next '_' character.

    :param filename: Filename to extract the exposure time from
    :return: Exposure time as a float
    """
    parts = str(filename).split("_")
    for i, part in enumerate(parts):
        if part == "exp":
            exposure_time_str = parts[i + 1].split(".fits")[
                0
            ]  # Split by '.' to handle file extension
            return float(exposure_time_str)


def find_median_files(directory: str | Path) -> list[str]:
    """
    Finds all median FITS files in the specified directory.

    :param directory: Directory to search for median files

    :return: List of median FITS files
    """
    median_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith("median_exp_") and f.endswith(".fits")
    ]
    return median_files


def extract_pixel_values(file: str | Path) -> tuple[list[np.ndarray], list[int]]:
    """
    Extracts the pixel values for each extension in the FITS file.

    :param file: FITS file to extract pixel values from

    :return: List of pixel values for each extension and list of board IDs
    """
    pixel_values = []
    board_ids = []
    with fits.open(file) as hdul:
        for hdu in hdul:
            data = hdu.data
            if data is not None:
                pixel_values.append(data)
                board_ids.append(hdu.header["BOARD_ID"])
    return pixel_values, board_ids
