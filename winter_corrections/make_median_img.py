#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:25:43 2024

@author: winter
"""
import os
import numpy as np
from astropy.io import fits
from collections import defaultdict

def get_exposure_time(filename):
    """
    Extracts the exposure time from the filename.
    Assumes the exposure time is between '_exp_' and the next '_' character.
    """
    parts = filename.split('_')
    for i, part in enumerate(parts):
        if part == 'exp':
            return float(parts[i + 1])

def find_files(directory):
    """
    Finds all FITS files in the specified directory.
    """
    fits_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('mef.fits')]
    return fits_files

def median_average_files(files):
    """
    Calculates the median average of the provided FITS files.
    Assumes all files have the same number of extensions.
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
            median_data.append(None)  # Append None if no valid data exists for this extension
    
    return headers, median_data

def save_median_file(headers, median_data, output_filename):
    """
    Saves the median data to a new FITS file.
    """
    hdul = fits.HDUList()
    for i, data in enumerate(median_data):
        if data is not None:
            hdu = fits.ImageHDU(data, header=headers[i])
        else:
            hdu = fits.ImageHDU(header=headers[i])  # Create an empty extension if data is None
        hdul.append(hdu)
    hdul.writeto(output_filename, overwrite=True)

def process_directory(directory):
    """
    Processes all FITS files in the specified directory.
    Groups them by exposure time and saves median averaged files.
    """
    files = find_files(directory)
    files_by_exposure = defaultdict(list)
    
    for file in files:
        exposure_time = get_exposure_time(os.path.basename(file))
        files_by_exposure[exposure_time].append(file)
    
    for exposure_time, files in files_by_exposure.items():
        headers, median_data = median_average_files(files)
        output_filename = os.path.join(directory, f'median_exp_{exposure_time}.fits')
        save_median_file(headers, median_data, output_filename)
        print(f'Saved median file for exposure time {exposure_time} to {output_filename}')

if __name__ == "__main__":
    # Example usage
    process_directory('/data/flats_iwr/20240610')
