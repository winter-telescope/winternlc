#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:25:43 2024

@author: winter
"""
import os
from collections import defaultdict

import numpy as np
from astropy.io import fits

from winternlc.create.utils import get_exposure_time


def find_files(directory: str) -> list[str]:
    """
    Finds all FITS files in the specified directory.

    :param directory: The directory to search for FITS files
    :return: A list of FITS files
    """
    fits_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith("mef.fits")
    ]
    return fits_files


def median_average_files(files: list[str]) -> tuple[list[dict], list[np.ndarray]]:
    """
    Calculates the median average of the provided FITS files.
    Assumes all files have the same number of extensions.

    :param files: A list of FITS files to process

    :return: A tuple containing the headers and median data for each extension
    """
    # Open the first file to get the number of extensions and their headers
    with fits.open(files[0]) as hdul:
        n_extensions = len(hdul)
        headers = [hdul[i].header for i in range(n_extensions)]

    # Read all data
    all_data = defaultdict(list)
    for file in files:
        with fits.open(file) as hdul:
            for i in range(n_extensions):
                data = hdul[i].data
                if data is not None:
                    all_data[i].append(data)

    # Calculate the median
    median_data = []
    for i in range(n_extensions):
        if all_data[i]:  # Check if there is data for this extension
            stacked_data = np.stack(all_data[i])
            median_data.append(np.median(stacked_data, axis=0))
        else:
            median_data.append(
                None
            )  # Append None if no valid data exists for this extension

    return headers, median_data


def save_median_file(
    headers: list[dict], median_data: list[np.ndarray], output_filename: str
):
    """
    Saves the median data to a new FITS file.
    """
    hdul = fits.HDUList()
    for i, data in enumerate(median_data):
        if data is not None:
            hdu = fits.ImageHDU(data, header=headers[i])
        else:
            hdu = fits.ImageHDU(
                header=headers[i]
            )  # Create an empty extension if data is None
        hdul.append(hdu)
    hdul.writeto(output_filename, overwrite=True)


def process_directory(directory: str):
    """
    Processes all FITS files in the specified directory.
    Groups them by exposure time and saves median averaged files.

    :param directory: The directory to process

    :return: None
    """
    files = find_files(directory)
    files_by_exposure = defaultdict(list)

    for file in files:
        exposure_time = get_exposure_time(os.path.basename(file))
        files_by_exposure[exposure_time].append(file)

    for exposure_time, files in files_by_exposure.items():
        headers, median_data = median_average_files(files)
        output_filename = os.path.join(directory, f"median_exp_{exposure_time}.fits")
        save_median_file(headers, median_data, output_filename)
        print(
            f"Saved median file for exposure time {exposure_time} to {output_filename}"
        )


if __name__ == "__main__":
    # Example usage
    process_directory("/data/flats_iwr/20240610")
