import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count

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
from winternlc.rational import rational_func


def fit_rational_to_pixel(args):
    i, j, pixel_values, exposure_times, cutoff = args
    coeffs = np.zeros((8,))
    pixel_series = [frame[i, j] for frame in pixel_values if frame[i, j] < cutoff]
    valid_exposure_times = [
        exposure_times[k]
        for k in range(len(pixel_values))
        if pixel_values[k][i, j] < cutoff
    ]
    if len(pixel_series) >= 8:
        try:
            x = np.array(pixel_series) / cutoff
            y = np.array(valid_exposure_times) / np.max(valid_exposure_times)
            popt_rat, _ = curve_fit(rational_func, x, y, p0=np.ones(8), maxfev=10000)
            coeffs[:] = popt_rat
        except RuntimeError:
            coeffs[:] = np.nan
    else:
        coeffs[:] = np.nan  # Handle cases where no valid data points are present
    return i, j, coeffs


def process_chunk(chunk, pixel_values, exposure_times, cutoff):
    print("in a chunk")
    chunk_results = []
    for i, j in chunk:
        chunk_results.append(
            fit_rational_to_pixel((i, j, pixel_values, exposure_times, cutoff))
        )
    return chunk_results


def fit_rational_to_pixels(
    exposure_times, pixel_values, cutoff, test=False, test_pixel=(0, 0)
):
    """
    Fits a rational function to the given pixel values for each pixel.
    Only uses pixel values below the cutoff counts.
    If test flag is True, only fits the rational function for the specified test pixel.
    """
    if test:
        coeffs = np.zeros((8,))
        i, j = test_pixel
        pixel_series = [frame[i, j] for frame in pixel_values if frame[i, j] < cutoff]
        valid_exposure_times = [
            exposure_times[k]
            for k in range(len(pixel_values))
            if pixel_values[k][i, j] < cutoff
        ]

        if len(pixel_series) >= 8:
            try:
                x = np.array(pixel_series) / cutoff
                y = np.array(valid_exposure_times) / np.max(valid_exposure_times)
                popt_rat, _ = curve_fit(rational_func, x, y, maxfev=10000)
                coeffs[:] = popt_rat
            except RuntimeError:
                coeffs[:] = np.nan
        else:
            coeffs[:] = np.nan  # Handle cases where no valid data points are present
        return coeffs, None  # No bad pixel mask for test case
    else:
        ny, nx = pixel_values[0].shape
        num_chunks = cpu_count() - 1
        indices = [(i, j) for i in range(ny) for j in range(nx)]
        chunk_size = len(indices) // num_chunks
        chunks = [
            indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)
        ]

        coeffs = np.zeros((ny, nx, 8))
        bad_pixel_mask = np.zeros((ny, nx), dtype=bool)

        print("starting pool")
        # with Pool(int(cpu_count()/2)) as pool:
        with Pool(10) as pool:
            results = pool.starmap(
                process_chunk,
                [(chunk, pixel_values, exposure_times, cutoff) for chunk in chunks],
            )
        print("ending pool")
        for chunk_results in results:
            for i, j, res in chunk_results:
                coeffs[i, j, :] = res
                if np.any(np.isnan(res)):
                    bad_pixel_mask[i, j] = True

        return coeffs, bad_pixel_mask


def save_rational_coefficients(
    median_files, output_dir, cutoff, test=False, test_pixel=(0, 0)
):
    """
    Fits a rational function to the pixel values and saves the coefficients as .npy files.
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
        print(f"trying corrections for board {ext}")
        # Fit rational function to each pixel and save coefficients
        rat_coeffs, bad_pixel_mask = fit_rational_to_pixels(
            sorted_exposure_times,
            sorted_values,
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
            if bad_pixel_mask is not None:
                np.save(
                    os.path.join(
                        output_dir,
                        f"bad_pixel_mask_board_{board_id}_ext_{ext}_test.npy",
                    ),
                    bad_pixel_mask,
                )
        else:
            np.save(
                str(get_coeffs_path(board_id, output_dir)),
                rat_coeffs,
            )
            np.save(
                str(get_mask_path(board_id, output_dir)),
                bad_pixel_mask,
            )


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
            if pixel_value >= cutoff:
                pixel_value = cutoff
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

    plt.xlabel("Exposure Time [s]")
    plt.ylabel("Signal Value [counts]")
    plt.title(f"Signal Value vs Exposure Time for Pixel {test_pixel}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.show()


def load_and_plot_rational(median_files, output_dir, cutoff, test_pixel, test=False):
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
            if pixel_value >= cutoff:
                pixel_value = cutoff
            pixel_values_by_extension[i].append(pixel_value)
            valid_exposure_times_by_extension[i].append(exposure_time)
            board_ids_by_extension[i] = board_ids[i]

    # colors = ['k', 'gray', 'blue', 'green', 'purple', 'cyan']
    boards = {
        1: "Port A",
        3: "Port B",
        4: "Port C",
        2: "Star A",
        6: "Star B",
        5: "Star C",
    }
    colors = {
        1: "limegreen",
        3: "darkturquoise",
        4: "royalblue",
        2: "darkviolet",
        6: "darkorange",
        5: "red",
    }
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
            rat_coeffs_path = os.path.join(
                output_dir,
                f"rat_coeffs_board_{board_ids_by_extension[ext]}_ext_{ext}.npy",
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
                color=colors[board_ids_by_extension[ext]],
                label=f"{boards[board_ids_by_extension[ext]]}",
            )
            plt.plot(
                sorted_exposure_times,
                cutoff * fitted_func(np.array(sorted_values) / cutoff),
                color=colors[board_ids_by_extension[ext]],
                linestyle="--",
            )

    plt.xlabel("Exposure time [s]")
    plt.ylabel("Raw counts [DN]")
    plt.title(f"Signal vs Rational Fit for Pixel {test_pixel} with 8 Parameters")
    # plt.ylim(0, cutoff)
    plt.legend()
    plt.tight_layout()
    # plt.savefig('/home/winter/Downloads/rational_fit.png', dpi=300)
    # plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Example usage
    median_files = find_median_files(test_directory)
    test_pixel = (519, 519)
    test = True
    if test:
        # Plot pixel signal for the test pixel
        # plot_pixel_signal(median_files, cutoff=cutoff, test_pixel=test_pixel)

        # Save rational coefficients for the test pixel
        # save_rational_coefficients(median_files, output_dir=output_directory, cutoff=cutoff, test=test, test_pixel=test_pixel)

        # Load and plot the fitted rational functions for the test pixel
        load_and_plot_rational(
            median_files,
            output_directory,
            DEFAULT_CUTOFF,
            test_pixel=test_pixel,
            test=test,
        )
    else:
        # Plot central pixel signal for all pixels
        # plot_pixel_signal(median_files, cutoff=cutoff, test_pixel=test_pixel)

        # Save rational coefficients for all pixels
        save_rational_coefficients(
            median_files, output_dir=output_directory, cutoff=DEFAULT_CUTOFF, test=test
        )

        # Load and plot the fitted rational functions for the central pixel
        # load_and_plot_rational(median_files, output_directory, cutoff, test_pixel=test_pixel)
