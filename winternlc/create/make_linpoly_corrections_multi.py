import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from utils import extract_pixel_values, find_median_files, get_exposure_time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import matplotlib.pyplot as plt

# from config import output_directory, test_directory
from joblib import Parallel, delayed

LINEAR_REGION_MINCOUNTS = 20000
LINEAR_REGION_MAXCOUNTS = 50000
SATURATION = 54000
EXTRA = 1000
BAD_PIXEL_STD_THRESHOLD = 500


def compute_correction_poly(expt, pixel_series, order):
    counts = pixel_series.astype(np.float64)
    mask_linear = (counts > LINEAR_REGION_MINCOUNTS) & (
        counts < LINEAR_REGION_MAXCOUNTS
    )
    if mask_linear.sum() < 2:
        return np.full(order + 1, np.nan), True

    linear_fit = np.polyfit(expt[mask_linear], counts[mask_linear], 1)
    mask_valid = counts < (SATURATION + EXTRA)
    mu_raw = counts[mask_valid]
    mu_cal = np.polyval(linear_fit, expt[mask_valid])

    if mu_raw.size < order + 1:
        return np.full(order + 1, np.nan), True

    fit = np.polyfit(mu_raw, mu_cal, order)
    residuals = mu_cal - np.polyval(fit, mu_raw)
    is_bad = np.isnan(fit).any() or residuals.std() > BAD_PIXEL_STD_THRESHOLD

    return fit, is_bad


def fit_pixel_block(yx_block, exptimes, cube, order):
    results = []
    for y, x in yx_block:
        pixel_series = cube[:, y, x]
        coeffs, is_bad = compute_correction_poly(exptimes, pixel_series, order)
        results.append((y, x, coeffs, is_bad))
    return results


def fit_correction_to_pixels_parallel(
    exptimes, pixel_values, order, n_jobs=8, block_size=500
):
    ny, nx = pixel_values.shape[1:]
    yx_coords = [(i, j) for i in range(ny) for j in range(nx)]
    blocks = [
        yx_coords[i : i + block_size] for i in range(0, len(yx_coords), block_size)
    ]

    coeffs = np.zeros((ny, nx, order + 1))
    bad_pixel_mask = np.zeros((ny, nx), dtype=bool)

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(fit_pixel_block)(block, exptimes, pixel_values, order)
        for block in blocks
    )

    for block_result in results:
        for y, x, c, is_bad in block_result:
            coeffs[y, x] = c
            if is_bad:
                bad_pixel_mask[y, x] = True

    return coeffs, bad_pixel_mask


def save_correction_coefficients_parallel(
    median_files, poly_order, output_dir, n_jobs=8
):
    exposure_times = []
    pixel_values_by_ext = defaultdict(list)
    board_ids_by_ext = {}

    for file in median_files:
        exposure_time = get_exposure_time(os.path.basename(file))
        exposure_times.append(exposure_time)
        pixel_values, board_ids = extract_pixel_values(file)
        for i, value in enumerate(pixel_values):
            pixel_values_by_ext[i].append(value)
            board_ids_by_ext[i] = board_ids[i]

    idx_sorted = np.argsort(exposure_times)
    sorted_expt = np.array(exposure_times)[idx_sorted]

    for ext, values in pixel_values_by_ext.items():
        if ext == 1:
            cube = np.array([values[i] for i in idx_sorted])  # shape: (n, y, x)
            coeffs, bad_pixel_mask = fit_correction_to_pixels_parallel(
                sorted_expt, cube, poly_order, n_jobs=n_jobs
            )
            board_id = board_ids_by_ext[ext]
            os.makedirs(output_dir, exist_ok=True)
            np.save(
                os.path.join(output_dir, f"corr_coeffs_board_{board_id}_ext_{ext}.npy"),
                coeffs,
            )
            np.save(
                os.path.join(
                    output_dir, f"bad_pixel_mask_board_{board_id}_ext_{ext}.npy"
                ),
                bad_pixel_mask,
            )


def plot_roi_correction(median_files, coeffs_dir, roi=(520, 530, 520, 530)):
    y0, y1, x0, x1 = roi
    exposure_times = []
    raw_data_by_ext = defaultdict(list)
    board_ids_by_ext = {}

    for file in median_files:
        exptime = get_exposure_time(os.path.basename(file))
        exposure_times.append(exptime)
        pixel_values, board_ids = extract_pixel_values(file)

        for ext, frame in enumerate(pixel_values):
            roi_values = frame[y0:y1, x0:x1]
            raw_data_by_ext[ext].append(roi_values)
            board_ids_by_ext[ext] = board_ids[ext]

    exposure_times = np.array(exposure_times)
    sort_idx = np.argsort(exposure_times)
    exposure_times = exposure_times[sort_idx]

    for ext, raw_stack in raw_data_by_ext.items():
        if ext == 1:
            raw_stack = np.array(raw_stack)[sort_idx]  # shape: (n_times, y, x)
            board_id = board_ids_by_ext[ext]
            coeff_path = os.path.join(
                coeffs_dir, f"corr_coeffs_board_{board_id}_ext_{ext}.npy"
            )
            mask_path = os.path.join(
                coeffs_dir, f"bad_pixel_mask_board_{board_id}_ext_{ext}.npy"
            )

            if not os.path.exists(coeff_path):
                print(f"Missing coeffs for ext {ext}")
                continue

            coeffs = np.load(coeff_path)
            bad_pixel_mask = np.load(mask_path) if os.path.exists(mask_path) else None

            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
            ax_raw, ax_corr = axes
            ax_raw.set_title(f"Raw Counts — Board {board_id}, Ext {ext}")
            ax_corr.set_title(f"Corrected Counts — Board {board_id}, Ext {ext}")

            for dy in range(y1 - y0):
                for dx in range(x1 - x0):
                    y, x = y0 + dy, x0 + dx
                    counts = raw_stack[:, dy, dx]
                    pcoeff = coeffs[y, x]

                    ax_raw.plot(exposure_times, counts, alpha=0.3, linewidth=0.8)

                    if np.isnan(pcoeff).any():
                        continue

                    corrected = np.where(
                        counts > SATURATION,
                        np.polyval(pcoeff, SATURATION),
                        np.polyval(pcoeff, counts),
                    )
                    # ax_corr.plot(exposure_times, corrected, alpha=0.3, linewidth=0.8)
                    color = (
                        "black"
                        if bad_pixel_mask is not None and bad_pixel_mask[y, x]
                        else None
                    )
                    ax_corr.plot(
                        exposure_times, corrected, alpha=0.3, linewidth=0.8, color=color
                    )

            for ax in axes:
                ax.set_xlabel("Exposure Time (s)")
                ax.set_ylabel("Counts")
                ax.set_ylim(0, 65000)

            fig.suptitle(
                f"ROI {roi} Pixel Correction: Board {board_id}, Ext {ext}", fontsize=14
            )
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    coeffs_dir, f"roi_correction_board_{board_id}_ext_{ext}.png"
                ),
                dpi=300,
            )
            # fig.show()
            plt.close(fig)

            roi_mask = bad_pixel_mask[y0:y1, x0:x1]
            roi_coeffs = coeffs[y0:y1, x0:x1]

            nan_mask = np.isnan(roi_coeffs).any(axis=-1)
            bad_nan = np.sum(roi_mask & nan_mask)
            bad_std = np.sum(roi_mask & ~nan_mask)
            total = roi_mask.size
            bad_total = bad_nan + bad_std

            print(f"ROI ({y0}:{y1}, {x0}:{x1}) — Total pixels: {total}")
            print(f"  Bad due to NaNs in fit: {bad_nan}")
            print(f"  Bad due to large residuals: {bad_std}")
            print(f"  Bad overall: {bad_total}")

            fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
            ax_raw_bad, ax_corr_bad = axes
            ax_raw_bad.set_title("Raw Bad Pixels")
            ax_corr_bad.set_title("Corrected Bad Pixels")

            if bad_pixel_mask is not None:
                for dy in range(y1 - y0):
                    for dx in range(x1 - x0):
                        y, x = y0 + dy, x0 + dx
                        if bad_pixel_mask[y, x]:
                            # Plot raw bad pixels
                            ax_raw_bad.plot(
                                exposure_times,
                                raw_stack[:, dy, dx],
                                alpha=0.3,
                                linewidth=0.8,
                                color="black",
                            )
                            # Plot corrected bad pixels
                            pcoeff = coeffs[y, x]
                            if not np.isnan(pcoeff).any():
                                corrected = np.where(
                                    raw_stack[:, dy, dx] > SATURATION,
                                    np.polyval(pcoeff, SATURATION),
                                    np.polyval(pcoeff, raw_stack[:, dy, dx]),
                                )
                                ax_corr_bad.plot(
                                    exposure_times,
                                    corrected,
                                    alpha=0.3,
                                    linewidth=0.8,
                                    color="black",
                                )

            for ax in axes:
                ax.set_xlabel("Exposure Time (s)")
                ax.set_ylabel("Counts")
                # ax.set_ylim(0, 65000)

            fig.tight_layout()
            plt.savefig(
                os.path.join(coeffs_dir, f"bad_pixels_board_{board_id}_ext_{ext}.png"),
                dpi=300,
            )
            plt.close()

            # Plot full-frame bad pixel mask as a scatter plot
            if bad_pixel_mask is not None:
                y_bad, x_bad = np.where(bad_pixel_mask)
                plt.figure(figsize=(6, 6))
                plt.scatter(x_bad, y_bad, s=0.1, color="red")
                plt.gca().add_patch(
                    plt.Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        edgecolor="blue",
                        facecolor="none",
                        linewidth=2,
                    )
                )
                plt.xlim(0, bad_pixel_mask.shape[1])
                plt.ylim(bad_pixel_mask.shape[0], 0)
                plt.gca().set_aspect("equal")
                plt.title(f"Bad Pixel Mask — Board {board_id}, Ext {ext}")
                plt.xlabel("X Pixel")
                plt.ylabel("Y Pixel")
                plt.grid(True, linestyle="--", alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        coeffs_dir, f"bad_pixel_mask_board_{board_id}_ext_{ext}.png"
                    ),
                    dpi=300,
                )
                # plt.show()
                plt.close()


if __name__ == "__main__":
    median_files = find_median_files("/Users/frostig/Downloads/flats_20250522")
    poly_order = 12
    save_correction_coefficients_parallel(
        median_files, poly_order, "/Users/frostig/Downloads/flats_20250522", n_jobs=10
    )

    roi = (500, 600, 1700, 1800)
    plot_roi_correction(median_files, "/Users/frostig/Downloads/flats_20250522", roi)
