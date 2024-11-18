#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:19:27 2024

@author: winter
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from winternlc.config import DEFAULT_CUTOFF, output_directory, test_directory
from winternlc.create.utils import (
    extract_pixel_values,
    find_median_files,
    get_exposure_time,
)
from winternlc.mask import get_mask_path
from winternlc.non_linear_correction import get_coeffs_path


def create_rational_func(num_params: int) -> Callable[[float], float]:
    """
    Creates a rational function with the specified number of parameters.

    :param num_params: Number of parameters for the rational function

    :return: Rational function
    """

    if num_params % 2 != 0:
        raise ValueError("Number of parameters must be even.")

    def rational_func(x, *params):
        """
        Rational function with the specified number of parameters.

        :param x: Input value
        :param params: Parameters for the rational function
        """
        n = num_params // 2
        numerator = sum([params[i] * x**i for i in range(n)])
        denominator = 1 + sum([params[n + i] * x ** (i + 1) for i in range(n)])
        return numerator / denominator

    return rational_func


def fit_rational_to_pixels(
    exposure_times: list[float],
    pixel_values: list[np.ndarray],
    num_params: int,
    cutoff: float,
    test: bool = False,
    test_pixel: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits a rational function to the pixel values.

    :param exposure_times: List of exposure times
    :param pixel_values: List of pixel values
    :param num_params: Number of parameters for the rational function
    :param cutoff: Cutoff value for the pixel values
    :param test: Flag to fit a single test pixel
    :param test_pixel: Test pixel coordinates

    :return: Rational function coefficients and bad pixel mask
    """
    rational_func = create_rational_func(num_params)

    if test:
        coeffs = np.zeros((num_params,))
        i, j = test_pixel
        pixel_series = [frame[i, j] for frame in pixel_values if frame[i, j] < cutoff]
        valid_exposure_times = [
            exposure_times[k]
            for k in range(len(pixel_values))
            if pixel_values[k][i, j] < cutoff
        ]
        bad = 0
        if len(pixel_series) >= num_params:
            try:
                x = np.array(pixel_series) / cutoff
                y = np.array(valid_exposure_times) / np.max(valid_exposure_times)
                popt_rat, _ = curve_fit(
                    rational_func, x, y, p0=np.ones(num_params), maxfev=10000
                )
                coeffs[:] = popt_rat
            except RuntimeError:
                coeffs[:] = np.nan
        else:
            coeffs[:] = np.nan  # Handle cases where no valid data points are present
    else:
        coeffs = np.zeros(pixel_values[0].shape + (num_params,))
        bad = np.zeros(pixel_values[0].shape)
        for i in range(pixel_values[0].shape[0]):
            for j in range(pixel_values[0].shape[1]):
                pixel_series = [
                    frame[i, j] for frame in pixel_values if frame[i, j] < cutoff
                ]
                valid_exposure_times = [
                    exposure_times[k]
                    for k in range(len(pixel_values))
                    if pixel_values[k][i, j] < cutoff
                ]

                if len(pixel_series) >= num_params:
                    try:
                        x = np.array(pixel_series) / cutoff
                        y = np.array(valid_exposure_times) / np.max(
                            valid_exposure_times
                        )
                        popt_rat, _ = curve_fit(
                            rational_func, x, y, p0=np.ones(num_params), maxfev=10000
                        )
                        coeffs[i, j, :] = popt_rat
                    except RuntimeError:
                        coeffs[i, j, :] = np.nan
                        bad[i, j] = 1
                else:
                    coeffs[i, j, :] = (
                        np.nan
                    )  # Handle cases where no valid data points are present
                    bad[i, j] = 1
    return coeffs, bad


def save_rational_coefficients(
    median_files: list[str],
    num_params: int,
    output_dir: str | Path,
    cutoff: float,
    test: bool = False,
    test_pixel: tuple[int, int] = (0, 0),
):
    """
    Fits a rational function to the pixel values and saves the coefficients as .npy files.
    Only uses pixel values below the cutoff counts.

    :param median_files: List of median FITS files
    :param num_params: Number of parameters for the rational function
    :param output_dir: Directory to save the coefficients
    :param cutoff: Cutoff value for the pixel values
    :param test: Flag to fit a single test pixel
    :param test_pixel: Test pixel coordinates

    :return: None
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

        # Fit rational function to each pixel and save coefficients
        rat_coeffs, bad_pix = fit_rational_to_pixels(
            sorted_exposure_times,
            sorted_values,
            num_params,
            cutoff,
            test=test,
            test_pixel=test_pixel,
        )
        board_id = board_ids_by_extension[ext]
        if test:
            np.save(
                os.path.join(
                    output_dir, f"rat_coeffs_board_{board_id}_ext_{ext}_test.npy"
                ),
                rat_coeffs,
            )
        else:
            np.save(
                str(get_coeffs_path(board_id, output_dir)),
                rat_coeffs,
            )
            np.save(
                str(get_mask_path(board_id, output_dir)),
                bad_pix,
            )


def plot_pixel_signal(
    median_files: list[str], cutoff: float, test_pixel: tuple[int, int]
):
    """
    Plots the signal value versus exposure time for the specified test pixel of each extension.
    Only uses pixel values below the cutoff counts.

    :param median_files: List of median FITS files
    :param cutoff: Cutoff value for the pixel values
    :param test_pixel: Test pixel coordinates

    :return: None
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
        sorted_exposure_times = np.array(valid_exposure_times_by_extension[ext])[
            sorted_indices
        ]
        sorted_values = np.array(values)[sorted_indices]
        plt.plot(
            sorted_exposure_times,
            sorted_values,
            label=f"BOARD_ID: {board_ids_by_extension[ext]}",
        )

    plt.xlabel("Exposure Time")
    plt.ylabel("Signal Value")
    plt.title(f"Signal Value vs Exposure Time for Pixel {test_pixel}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.show()


def load_and_plot_rational(
    median_files: list[str],
    output_dir: str | Path,
    cutoff: float,
    num_params: int,
    test_pixel: tuple[int, int] = (0, 0),
    test: bool = False,
):
    """
    Fits a rational function to the pixel values and plots
    the fitted function for the specified test pixel.

    :param median_files: List of median FITS files
    :param output_dir: Directory containing the coefficients
    :param cutoff: Cutoff value for the pixel values
    :param num_params: Number of parameters for the rational function
    :param test_pixel: Test pixel coordinates
    :param test: Flag to fit a single test pixel

    :return: None
    """
    rational_func = create_rational_func(num_params)

    pixel_values_by_extension = defaultdict(list)
    board_ids_by_extension = {}
    valid_exposure_times_by_extension = defaultdict(list)

    for file in median_files:
        exposure_time = get_exposure_time(os.path.basename(file))
        pixel_values, board_ids = extract_pixel_values(file)

        for i, values in enumerate(pixel_values):
            ny, nx = values.shape
            pixel_value = values[test_pixel[0], test_pixel[1]]
            if pixel_value >= cutoff:
                pixel_value = cutoff
            pixel_values_by_extension[i].append(pixel_value)
            valid_exposure_times_by_extension[i].append(exposure_time)
            board_ids_by_extension[i] = board_ids[i]

    colors = ["k", "gray", "blue", "green", "purple", "cyan"]
    for ext, values in pixel_values_by_extension.items():
        sorted_indices = np.argsort(valid_exposure_times_by_extension[ext])
        sorted_exposure_times = np.array(valid_exposure_times_by_extension[ext])[
            sorted_indices
        ]
        sorted_values = np.array(values)[sorted_indices]

        if test:
            rat_coeffs_path = os.path.join(
                output_dir,
                f"rat_coeffs_board_{board_ids_by_extension[ext]}_ext_{ext}_test.npy",
            )
        else:
            rat_coeffs_path = get_coeffs_path(
                board_id=board_ids_by_extension[ext], cor_dir=output_dir
            )

        if os.path.exists(rat_coeffs_path):
            rat_coeffs = np.load(rat_coeffs_path)
            if test:
                pixel_coeffs = rat_coeffs
            else:
                pixel_coeffs = rat_coeffs[test_pixel[0], test_pixel[1]]

            def fitted_func(x):
                return rational_func(x, *pixel_coeffs)

            plt.plot(
                sorted_exposure_times,
                sorted_values,
                color=colors[ext],
                label=f"Signal BOARD_ID: {board_ids_by_extension[ext]}",
            )
            plt.plot(
                sorted_exposure_times,
                cutoff * fitted_func(np.array(sorted_values) / cutoff),
                color=colors[ext],
                linestyle="--",
                label=f"Fit BOARD_ID: {board_ids_by_extension[ext]}",
            )

    plt.xlabel("Exposure Time [s]")
    plt.ylabel("Signal Value [counts]")
    plt.title(
        f"Signal vs Rational Fit for Pixel {test_pixel} with {num_params} Parameters"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Example usage
    median_files = find_median_files(test_directory)
    num_params = 8  # Set the number of parameters for the rational function
    test_pixel = (519, 519)
    test = True

    if test:
        # Plot pixel signal for the test pixel
        plot_pixel_signal(median_files, cutoff=DEFAULT_CUTOFF, test_pixel=test_pixel)

        # Save rational coefficients for the test pixel
        save_rational_coefficients(
            median_files,
            num_params,
            output_dir=output_directory,
            cutoff=DEFAULT_CUTOFF,
            test=test,
            test_pixel=test_pixel,
        )

        # Load and plot the fitted rational functions for the test pixel
        load_and_plot_rational(
            median_files,
            output_directory,
            DEFAULT_CUTOFF,
            num_params,
            test_pixel=test_pixel,
            test=test,
        )
    else:
        # Plot central pixel signal for all pixels
        plot_pixel_signal(median_files, cutoff=DEFAULT_CUTOFF, test_pixel=test_pixel)

        # Save rational coefficients for all pixels
        save_rational_coefficients(
            median_files,
            num_params,
            output_dir=output_directory,
            cutoff=DEFAULT_CUTOFF,
            test=test,
        )

        # Load and plot the fitted rational functions for the central pixel
        load_and_plot_rational(
            median_files,
            output_directory,
            DEFAULT_CUTOFF,
            num_params,
            test_pixel=test_pixel,
        )
