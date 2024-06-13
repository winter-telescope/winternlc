
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 08:04:11 2024

@author: winter
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from collections import defaultdict
from scipy.optimize import curve_fit
from config import cutoff, test_directory, output_directory

def get_exposure_time(filename):
    """
    Extracts the exposure time from the filename.
    Assumes the exposure time is between '_exp_' and the next '_' character.
    """
    parts = filename.split('_')
    for i, part in enumerate(parts):
        if part == 'exp':
            exposure_time_str = parts[i + 1].split('.fits')[0]  # Split by '.' to handle file extension
            return float(exposure_time_str)

def find_median_files(directory):
    """
    Finds all median FITS files in the specified directory.
    """
    median_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith('median_exp_') and f.endswith('.fits')]
    return median_files

def extract_pixel_values(file):
    """
    Extracts the pixel values for each extension in the FITS file.
    """
    pixel_values = []
    board_ids = []
    with fits.open(file) as hdul:
        for hdu in hdul:
            data = hdu.data
            if data is not None:
                pixel_values.append(data)
                board_ids.append(hdu.header['BOARD_ID'])
    return pixel_values, board_ids

def fit_polynomial_to_pixels(exposure_times, pixel_values, order, cutoff, test=False, test_pixel=(0, 0)):
    """
    Fits an nth order polynomial to the given pixel values for each pixel.
    Only uses pixel values below the cutoff counts.
    If test flag is True, only fits the polynomial for the specified test pixel.
    """
    if test:
        coeffs = np.zeros((order + 1,))
        i, j = test_pixel
        print("test", pixel_values)
        pixel_series = [frame[i, j] for frame in pixel_values if frame[i, j] < cutoff]
        valid_exposure_times = [exposure_times[k] for k in range(len(pixel_values)) if pixel_values[k][i, j] < cutoff]

        if len(pixel_series) >= order + 1:
            try:
                
                x = np.array(pixel_series)/cutoff
                #x = x#-x[0]
                y = valid_exposure_times/np.max(valid_exposure_times)
                #print("params", cutoff, order, x, y)
                poly_coeffs = np.polyfit(x, y, order)
                coeffs[:] = poly_coeffs

                
                # plt.figure()
                
                # p = np.poly1d(poly_coeffs)
                # plt.plot(x,y, label="data")
                # plt.plot(x, p(np.linspace(0,1, len(x))), label="poly fit")
                # plt.plot(x, p((x)), label="fit data")
                # plt.plot(x, sigmoid(x, *popt_sigmoid), 'r--', label='Sigmoid fit')
                # plt.plot(x, exponential(x, *popt_exp), 'g--', label='Exponential fit')
                # plt.legend()
                # plt.show()

            except np.linalg.LinAlgError:
                coeffs[:] = np.nan
        else:
            coeffs[:] = np.nan  # Handle cases where no valid data points are present
    else:
        coeffs = np.zeros(pixel_values[0].shape + (order + 1,))
        for i in range(pixel_values[0].shape[0]):
            for j in range(pixel_values[0].shape[1]):
                pixel_series = [frame[i, j] for frame in pixel_values if frame[i, j] < cutoff]
                valid_exposure_times = [exposure_times[k] for k in range(len(pixel_values)) if pixel_values[k][i, j] < cutoff]
                
                if len(pixel_series) >= order + 1:
                    try:
                        print("cutoff", cutoff)
                        poly_coeffs = np.polyfit(np.array(pixel_series)/cutoff, valid_exposure_times, order)

                    except np.linalg.LinAlgError:
                        coeffs[i, j, :] = np.nan
                else:
                    coeffs[i, j, :] = np.nan  # Handle cases where no valid data points are present
    
    return coeffs


# def fit_polynomial_to_pixels(exposure_times, pixel_values, order, cutoff, test=False, test_pixel=(0, 0)):
#     """
#     Fits an nth order polynomial to the given pixel values for each pixel.
#     Only uses pixel values below the cutoff counts.
#     If test flag is True, only fits the polynomial for the specified test pixel.
#     """
#     if test:
#         coeffs = np.zeros((order + 1,))
#         i, j = test_pixel
#         print("frame", pixel_values)
#         if test:
#             pixel_series = [frame for frame in pixel_values if frame < cutoff]
#         else:
#             pixel_series = [frame[i, j] for frame in pixel_values if frame[i, j] < cutoff]
#         if len(pixel_series) >= order + 1:
#             if test:
#                 valid_exposure_times = [exposure_times[k] for k in range(len(pixel_values)) if pixel_values[k] < cutoff]
#             else:
#                 valid_exposure_times = [exposure_times[k] for k in range(len(pixel_values)) if pixel_values[k][i, j] < cutoff]
#             try:
#                 print(np.array(pixel_series)/cutoff)
#                 print(np.shape(valid_exposure_times))
#                 poly_coeffs = np.polyfit(np.array(pixel_series)/cutoff, valid_exposure_times, order)
#                 coeffs[:] = poly_coeffs
#             except np.linalg.LinAlgError:
#                 coeffs[:] = np.nan
#         else:
#             coeffs[:] = np.nan  # Handle cases where no valid data points are present
#     else:
#         coeffs = np.zeros(pixel_values[0].shape + (order + 1,))
#         for i in range(pixel_values[0].shape[0]):
#             for j in range(pixel_values[0].shape[1]):
#                 pixel_series = [frame[i, j] for frame in pixel_values if frame[i, j] < cutoff]
#                 if len(pixel_series) >= order + 1:
#                     valid_exposure_times = [exposure_times[k] for k in range(len(pixel_values)) if pixel_values[k][i, j] < cutoff]
#                     try:
#                         poly_coeffs = np.polyfit(np.array(pixel_series)/cutoff, valid_exposure_times, order)
#                         coeffs[i, j, :] = poly_coeffs
#                     except np.linalg.LinAlgError:
#                         coeffs[i, j, :] = np.nan
#                 else:
#                     coeffs[i, j, :] = np.nan  # Handle cases where no valid data points are present
    
#     return coeffs


def save_polynomial_coefficients(median_files, poly_order, output_dir, cutoff, test=False, test_pixel=(0, 0)):
    """
    Fits an nth order polynomial to the pixel values and saves the coefficients as .npy files.
    Only uses pixel values below the cutoff counts.
    """
    exposure_times = []
    pixel_values_by_extension = defaultdict(list)
    board_ids_by_extension = {}

    for file in median_files:
        exposure_time = get_exposure_time(os.path.basename(file))
        exposure_times.append(exposure_time)
        pixel_values, board_ids = extract_pixel_values(file)
        
        for i, value in enumerate(pixel_values):
            pixel_values_by_extension[i].append(value)
            board_ids_by_extension[i] = board_ids[i]
    
    # Sort exposure times and corresponding pixel values
    sorted_indices = np.argsort(exposure_times)
    sorted_exposure_times = np.array(exposure_times)[sorted_indices]

    for ext, values in pixel_values_by_extension.items():
        sorted_values = [values[i] for i in sorted_indices]
        
        # Fit polynomial to each pixel and save coefficients
        poly_coeffs = fit_polynomial_to_pixels(sorted_exposure_times, sorted_values,  poly_order, cutoff, test=test, test_pixel=test_pixel)
        board_id = board_ids_by_extension[ext]
        if test:
            np.save(os.path.join(output_dir, f'poly_coeffs_board_{board_id}_ext_{ext}_test.npy'), poly_coeffs)
        else:
            np.save(os.path.join(output_dir, f'poly_coeffs_board_{board_id}_ext_{ext}.npy'), poly_coeffs)


def plot_pixel_signal(median_files, cutoff, test_pixel):
    """
    Plots the signal value versus exposure time for the specified test pixel of each extension.
    Only uses pixel values below the cutoff counts.
    """
    exposure_times = []
    pixel_values_by_extension = defaultdict(list)
    board_ids_by_extension = {}
    valid_exposure_times_by_extension = defaultdict(list)

    for file in median_files:
        exposure_time = get_exposure_time(os.path.basename(file))
        pixel_values, board_ids = extract_pixel_values(file)
        
        for i, values in enumerate(pixel_values):
            ny, nx = values.shape
            pixel_value = values[test_pixel[0], test_pixel[1]]
            if pixel_value < cutoff:
                pixel_values_by_extension[i].append(pixel_value)
                valid_exposure_times_by_extension[i].append(exposure_time)
                board_ids_by_extension[i] = board_ids[i]
    
    for ext, values in pixel_values_by_extension.items():
        sorted_indices = np.argsort(valid_exposure_times_by_extension[ext])
        sorted_exposure_times = np.array(valid_exposure_times_by_extension[ext])[sorted_indices]
        sorted_values = np.array(values)[sorted_indices]
        plt.plot(sorted_exposure_times, sorted_values, label=f'BOARD_ID: {board_ids_by_extension[ext]}')
    
    plt.xlabel('Exposure Time')
    plt.ylabel('Signal Value')
    plt.title(f'Signal Value vs Exposure Time for Pixel {test_pixel}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()

def load_and_plot_polynomials(median_files, output_dir, cutoff, poly_order, test_pixel, test=False):
    """
    Loads .npy coefficient files and plots the signal for the specified test pixel based on the coefficients.
    """
    exposure_times = []
    pixel_values_by_extension = defaultdict(list)
    board_ids_by_extension = {}
    valid_exposure_times_by_extension = defaultdict(list)

    for file in median_files:
        exposure_time = get_exposure_time(os.path.basename(file))
        pixel_values, board_ids = extract_pixel_values(file)
        
        for i, values in enumerate(pixel_values):
            ny, nx = values.shape
            pixel_value = values[test_pixel[0], test_pixel[1]]
            if pixel_value < cutoff:
                pixel_values_by_extension[i].append(pixel_value)
                valid_exposure_times_by_extension[i].append(exposure_time)
                board_ids_by_extension[i] = board_ids[i]
    colors = ['k', 'gray', 'blue', 'green', 'purple', 'cyan']
    for ext, values in pixel_values_by_extension.items():
        sorted_indices = np.argsort(valid_exposure_times_by_extension[ext])
        sorted_exposure_times = np.array(valid_exposure_times_by_extension[ext])[sorted_indices]
        sorted_values = np.array(values)[sorted_indices]

        if test:
            poly_coeffs_path = os.path.join(output_dir, f'poly_coeffs_board_{board_ids_by_extension[ext]}_ext_{ext}_test.npy')
        else:
            poly_coeffs_path = os.path.join(output_dir, f'poly_coeffs_board_{board_ids_by_extension[ext]}_ext_{ext}.npy')

        if os.path.exists(poly_coeffs_path):
            poly_coeffs = np.load(poly_coeffs_path)
            if test:
                pixel_coeffs = poly_coeffs
            else:
                pixel_coeffs = poly_coeffs[test_pixel[0], test_pixel[1]]

            p = np.poly1d(pixel_coeffs)
            print("cutoff", cutoff)
            
            plt.plot(sorted_exposure_times, sorted_values, color=colors[ext], label=f'Signal BOARD_ID: {board_ids_by_extension[ext]}')
            #plt.plot(sorted_exposure_times, polynomial(sorted_exposure_times), linestyle='--', label=f'Fit BOARD_ID: {board_ids_by_extension[ext]}')
            plt.plot(sorted_exposure_times, cutoff*p(np.array(sorted_values)/cutoff),  color=colors[ext],linestyle='--', label=f'Fit BOARD_ID: {board_ids_by_extension[ext]}')

    plt.xlabel('Exposure Time [s]')
    plt.ylabel('Signal Value [counts]')
    plt.title(f'Signal vs Polynomial Fit for Pixel {test_pixel}, with Polynomial Order {poly_order}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.ylim((0, cutoff))
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage
    median_files = find_median_files(test_directory)
    poly_order = 7
    test_pixel = (519, 519)
    test = True
    
    if test:
        # Plot pixel signal for the test pixel
        print("plot pix sig")
        plot_pixel_signal(median_files, cutoff=cutoff, test_pixel=test_pixel)
        
        print("save poly coeff")
        # Save polynomial coefficients for the test pixel
        save_polynomial_coefficients(median_files, poly_order, output_dir=output_directory, cutoff=cutoff, test=test, test_pixel=test_pixel)
        
        print("load poly coeff")
        # Load and plot the fitted polynomials for the test pixel
        load_and_plot_polynomials(median_files, output_directory, cutoff, poly_order, test_pixel=test_pixel, test=test)
    else:
        # Plot central pixel signal for all pixels
        plot_pixel_signal(median_files, cutoff=cutoff, test_pixel=test_pixel)
    
        # Save polynomial coefficients for all pixels
        save_polynomial_coefficients(median_files, poly_order, output_dir=output_directory, cutoff=cutoff, test=test)
    
        # Load and plot the fitted polynomials for the central pixel
        load_and_plot_polynomials(median_files, output_directory, cutoff, poly_order, test_pixel=test_pixel)