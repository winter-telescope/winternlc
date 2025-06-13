import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev
from utils import extract_pixel_values, find_median_files, get_exposure_time

from winternlc.config import DEFAULT_CUTOFF, output_directory, test_directory

LINEAR_REGION_MINCOUNTS = 20000
LINEAR_REGION_MAXCOUNTS = 50000
SATURATION = 56000


def compute_correction_poly(exposure_times, pixel_series, order, test=True, PLOT=False):
    counts = np.array(pixel_series, dtype=np.float64)
    exptime = np.array(exposure_times, dtype=np.float64)

    linear_mask = (counts > LINEAR_REGION_MINCOUNTS) & (
        counts < LINEAR_REGION_MAXCOUNTS
    )
    if linear_mask.sum() < 2:
        return np.full(order + 1, np.nan)

    linear_fit = np.polyfit(exptime[linear_mask], counts[linear_mask], 1)

    if test and PLOT:
        plt.figure()
        plt.plot(exptime, counts, "o", label="Data")
        plt.plot(
            exptime, np.polyval(linear_fit, exptime), label="Linear Fit", color="red"
        )
        plt.title("Step 2: Fit a line")
        plt.legend()

    mask = counts < SATURATION
    mu_raw = counts[mask]
    mu_cal = np.polyval(linear_fit, exptime[mask])

    if test and PLOT:
        plt.figure()
        plt.plot(counts, np.polyval(linear_fit, exptime), label="Full data set")
        plt.plot(mu_raw, mu_cal, label="Data with saturation cut")
        plt.xlabel(r"$\mu_{\mathrm{raw}}$")
        plt.ylabel(r"$\mu_{\mathrm{cal}}$")
        plt.title("Step 4: $\mu_{\mathrm{cal}}$ vs $\mu_{\mathrm{raw}}$")
        plt.legend()

    poly_coeffs = np.polyfit(mu_raw, mu_cal, order)

    if test and PLOT:
        plt.figure()
        mu_new = np.linspace(np.min(mu_raw), np.max(mu_raw), 1000)
        plt.plot(
            mu_new,
            np.polyval(poly_coeffs, mu_new),
            label="Polynomial Fit",
            color="green",
        )
        plt.xlabel(r"$\mu_{\mathrm{raw}}$")
        plt.ylabel(r"$\mu_{\mathrm{cal}}$")
        plt.title(
            "Step 5: Polynomial Fit to $\mu_{\mathrm{cal}}$ vs $\mu_{\mathrm{raw}}$"
        )
        plt.legend()

        plt.figure()
        mu_new = np.linspace(np.min(mu_raw), np.max(mu_raw), 1000)
        plt.plot(
            mu_new,
            np.polyval(poly_coeffs, mu_new),
            label="Polynomial Fit",
            color="green",
        )
        plt.plot(mu_raw, mu_cal, "o", label="Data with saturation cut")
        plt.xlabel(r"$\mu_{\mathrm{raw}}$")
        plt.ylabel(r"$\mu_{\mathrm{cal}}$")
        plt.title("Combined: $\mu_{\mathrm{cal}}$ vs $\mu_{\mathrm{raw}}$")
        plt.legend()

        #### 6. Save polynomial coefficents to a file
        np.save(output_directory + "polynomial_coefficients.npy", poly_coeffs)

        #### 7. Apply correction to the data
        # Load coefficients from file
        loaded_poly_coeffs = np.load("polynomial_coefficients.npy")
        corr = np.where(
            counts > SATURATION,
            np.polyval(loaded_poly_coeffs, SATURATION),
            np.polyval(loaded_poly_coeffs, counts),
        )

        plt.figure()
        plt.plot(exptime, counts, "o", color="black", label="Raw data")
        plt.plot(exptime, corr, ".", label="Corrected nominal data", color="purple")
        plt.legend()
        plt.xlabel("Exposure Time (s)")
        plt.ylabel("Corrected Counts")
        plt.title("Step 7: Apply correction")

    return poly_coeffs


def fit_correction_to_pixels(
    exposure_times, pixel_values, order, test=False, test_pixel=(0, 0)
):
    if test:
        i, j = test_pixel
        pixel_series = [frame[i, j] for frame in pixel_values]
        return compute_correction_poly(exposure_times, pixel_series, order, test=test)

    ny, nx = pixel_values[0].shape
    coeffs = np.zeros((ny, nx, order + 1))
    for i in range(ny):
        for j in range(nx):
            pixel_series = [frame[i, j] for frame in pixel_values]
            coeffs[i, j, :] = compute_correction_poly(
                exposure_times, pixel_series, order
            )
    return coeffs


def save_correction_coefficients(
    median_files, poly_order, output_dir, test=False, test_pixel=(0, 0)
):
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

    sorted_indices = np.argsort(exposure_times)
    sorted_exptimes = np.array(exposure_times)[sorted_indices]

    for ext, values in pixel_values_by_extension.items():
        sorted_values = [values[i] for i in sorted_indices]
        coeffs = fit_correction_to_pixels(
            sorted_exptimes, sorted_values, poly_order, test=test, test_pixel=test_pixel
        )
        board_id = board_ids_by_extension[ext]
        suffix = f"_test" if test else ""
        os.makedirs(output_dir, exist_ok=True)
        np.save(
            os.path.join(
                output_dir, f"corr_coeffs_board_{board_id}_ext_{ext}{suffix}.npy"
            ),
            coeffs.astype(np.float32),
        )


def plot_pixel_signal(median_files, cutoff, test_pixel):
    exposure_times = []
    pixel_values_by_extension = defaultdict(list)
    board_ids_by_extension = {}
    valid_exposure_times_by_extension = defaultdict(list)

    for file in median_files:
        exposure_time = get_exposure_time(os.path.basename(file))
        pixel_values, board_ids = extract_pixel_values(file)

        for i, values in enumerate(pixel_values):
            val = values[test_pixel[0], test_pixel[1]]
            if val < cutoff:
                pixel_values_by_extension[i].append(val)
                valid_exposure_times_by_extension[i].append(exposure_time)
                board_ids_by_extension[i] = board_ids[i]

    for ext, values in pixel_values_by_extension.items():
        expt = np.array(valid_exposure_times_by_extension[ext])
        vals = np.array(values)
        sidx = np.argsort(expt)
        plt.plot(
            expt[sidx], vals[sidx], label=f"BOARD_ID: {board_ids_by_extension[ext]}"
        )

    plt.xlabel("Exposure Time")
    plt.ylabel("Signal Value")
    plt.title(f"Signal vs Exposure Time for Pixel {test_pixel}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)


def load_and_plot_correction(
    median_files, output_dir, poly_order, test_pixel, test=False
):
    exposure_times = []
    pixel_values_by_extension = defaultdict(list)
    board_ids_by_extension = {}
    valid_exposure_times_by_extension = defaultdict(list)
    plt.figure()
    for file in median_files:
        exposure_time = get_exposure_time(os.path.basename(file))
        pixel_values, board_ids = extract_pixel_values(file)

        for i, values in enumerate(pixel_values):
            val = values[test_pixel[0], test_pixel[1]]
            if val < SATURATION:
                pixel_values_by_extension[i].append(val)
                valid_exposure_times_by_extension[i].append(exposure_time)
                board_ids_by_extension[i] = board_ids[i]

    for ext, values in pixel_values_by_extension.items():
        expt = np.array(valid_exposure_times_by_extension[ext])
        vals = np.array(values, dtype=np.float64)
        # Sort by exposure time
        sidx = np.argsort(expt)
        expt = expt[sidx]
        vals = vals[sidx]

        suffix = "_test" if test else ""
        path = os.path.join(
            output_dir,
            f"corr_coeffs_board_{board_ids_by_extension[ext]}_ext_{ext}{suffix}.npy",
        )
        if not os.path.exists(path):
            continue

        coeffs = np.load(path).astype(np.float64)
        pcoeff = coeffs if test else coeffs[test_pixel[0], test_pixel[1]]

        def apply_correction(count):
            return np.where(
                count > SATURATION,
                np.polyval(pcoeff, SATURATION),
                np.polyval(pcoeff, count),
            )

        corrected = apply_correction(vals)
        color = f"C{board_ids_by_extension[ext] % 10}"  # Cycle through 10 colors
        plt.plot(
            expt,
            vals,
            ".",
            label=f"Raw BOARD_ID: {board_ids_by_extension[ext]}",
            color=color,
        )
        plt.plot(
            expt,
            corrected,
            "--",
            label=f"Corrected BOARD_ID: {board_ids_by_extension[ext]}",
            color=color,
        )

    plt.xlabel("Exposure Time")
    plt.ylabel("Counts")
    plt.ylim(0, 1.2 * SATURATION)
    plt.xlim(0, 1.2 * max(expt))
    plt.title(f"Corrected Signal vs Exposure Time for Pixel {test_pixel}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    median_files = find_median_files(test_directory)
    poly_order = 12
    test_pixel = (523, 523)
    test = True

    if test:
        plot_pixel_signal(median_files, cutoff=DEFAULT_CUTOFF, test_pixel=test_pixel)
        save_correction_coefficients(
            median_files, poly_order, output_directory, test=test, test_pixel=test_pixel
        )
        load_and_plot_correction(
            median_files, output_directory, poly_order, test_pixel, test=test
        )
