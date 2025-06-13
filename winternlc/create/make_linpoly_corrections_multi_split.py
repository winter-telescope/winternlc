import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from split_linpoly_corrections import apply_nlc_correction, compute_correction_poly
from utils import extract_pixel_values, find_median_files, get_exposure_time

CORRECTION_CEILING = 55_000  # counts
CROSSOVER_POINT = 25_000  # counts
LINEAR_REGION_MIN_FRACTION = 0.33
LINEAR_REGION_MAX_FRACTION = 0.85
OVERLAP = 10_000
ORDER_LOWER = 11
ORDER_UPPER = 5
BAD_PIXEL_STD_THRESHOLD = 500
RESAMPLE_POINTS = 2500


def fit_pixel_block(yx_block, exptimes, cube, order):
    results = []
    for y, x in yx_block:
        pixel_series = cube[:, y, x]

        # compute the correction polynomial
        coeffs_lower, coeffs_upper, is_bad = compute_correction_poly(
            expt=exptimes,
            counts=pixel_series,
            linear_region_min_fraction=LINEAR_REGION_MIN_FRACTION,
            linear_region_max_fraction=LINEAR_REGION_MAX_FRACTION,
            order_lower=ORDER_LOWER,
            order_upper=ORDER_UPPER,
            crossover_point=CROSSOVER_POINT,
            overlap=OVERLAP,
            ceiling=CORRECTION_CEILING,
            resample_interpolation_kind="linear",
            resample_points=RESAMPLE_POINTS,
            residual_std_threshold=BAD_PIXEL_STD_THRESHOLD,
            verbose=False,
        )
        coeffs = np.concatenate([coeffs_lower, coeffs_upper])
        results.append((y, x, coeffs, is_bad))
    return results


def fit_correction_to_pixels_parallel(
    exptimes, pixel_values, order=(11, 5), n_jobs=8, block_size=500
):
    total_order = sum(order) + 1

    ny, nx = pixel_values.shape[1:]
    yx_coords = [(i, j) for i in range(ny) for j in range(nx)]
    blocks = [
        yx_coords[i : i + block_size] for i in range(0, len(yx_coords), block_size)
    ]

    coeffs = np.zeros((ny, nx, total_order + 1))
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
    median_files, output_dir, n_jobs=8, exts=None
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
        if ext in exts or exts is None:
            cube = np.array([values[i] for i in idx_sorted])  # shape: (n, y, x)
            coeffs, bad_pixel_mask = fit_correction_to_pixels_parallel(
                sorted_expt, cube, (11, 5), n_jobs=n_jobs
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


def plot_roi_correction(
    median_files,
    coeffs_dir,
    roi=(520, 530, 520, 530),
    selection="roi",
    n_random_pixels=100,
):

    exposure_times = []
    raw_data_by_ext = defaultdict(list)
    board_ids_by_ext = {}

    for file in median_files:
        exptime = get_exposure_time(os.path.basename(file))
        exposure_times.append(exptime)

        pixel_values, board_ids = extract_pixel_values(file)

        for ext, frame in enumerate(pixel_values):
            if selection == "random":
                # Select random pixels from the frame
                y_indices = np.random.choice(
                    frame.shape[0], n_random_pixels, replace=False
                )
                x_indices = np.random.choice(
                    frame.shape[1], n_random_pixels, replace=False
                )
                # roi_values = frame[y_indices, x_indices]
            elif selection == "roi":
                # Use the specified ROI
                y0, y1, x0, x1 = roi
                # roi_values = frame[y0:y1, x0:x1]
                x_indices = np.arange(x0, x1)
                y_indices = np.arange(y0, y1)

            else:
                raise ValueError(
                    f"Unknown selection type: {selection}, must be 'roi' or 'random'"
                )

            raw_data_by_ext[ext].append(frame)
            board_ids_by_ext[ext] = board_ids[ext]

    exposure_times = np.array(exposure_times)
    print(f"number of exptimes = {len(exposure_times)}")
    sort_idx = np.argsort(exposure_times)
    exposure_times = exposure_times[sort_idx]

    for ext, raw_stack in raw_data_by_ext.items():
        raw_stack = np.array(raw_stack)[sort_idx]  # shape: (n_times, y, x)
        print(f"Raw stack shape for ext {ext}: {raw_stack.shape}")
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
        print(f"Loaded coeffs for ext {ext}: {coeffs.shape}")
        # coeffs shape: (y, x, order_lower + 1 + order_upper + 1)
        coeffs_lower = coeffs[:, :, : ORDER_LOWER + 1]
        coeffs_upper = coeffs[:, :, ORDER_LOWER + 1 :]
        # we need to split the coeffs back into lower and upper parts
        # the first half is the lower order polynomial, the second half is the upper order polynomial

        print(
            f"Check: lower order = {coeffs_lower.shape}, upper order = {coeffs_upper.shape}"
        )

        bad_pixel_mask = np.load(mask_path) if os.path.exists(mask_path) else None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        ax_raw, ax_corr = axes
        ax_raw.set_title(f"Raw Counts — Board {board_id}, Ext {ext}")
        ax_corr.set_title(f"Corrected Counts — Board {board_id}, Ext {ext}")

        for y in y_indices:
            for x in x_indices:

                counts = raw_stack[:, y, x]
                pixel_is_bad = bad_pixel_mask[y, x]
                # pcoeff = coeffs[y, x]
                coeffs_lower_pixel = coeffs_lower[y, x]
                coeffs_upper_pixel = coeffs_upper[y, x]

                ax_raw.plot(exposure_times, counts, alpha=0.3, linewidth=0.8)

                if np.isnan(coeffs_lower_pixel).any():
                    continue
                if np.isnan(coeffs_upper_pixel).any():
                    continue
                print(
                    f"Plotting pixel ({y}, {x}) lower coeffs shape: {coeffs_lower_pixel.shape}, upper coeffs shape: {coeffs_upper_pixel.shape}, counts shape: {counts.shape}"
                )
                # apply the correction
                corrected = apply_nlc_correction(
                    counts=counts,
                    coeffs_lower=coeffs_lower_pixel,
                    coeffs_upper=coeffs_upper_pixel,
                    crossover_point=CROSSOVER_POINT,
                    ceiling=None,
                )
                # ax_corr.plot(exposure_times, corrected, alpha=0.3, linewidth=0.8)
                # color = (
                #    "black"
                #    if bad_pixel_mask is not None and bad_pixel_mask[y, x]
                #    else None
                # )
                if not pixel_is_bad:
                    # Only plot if the pixel is not bad
                    ax_corr.plot(
                        exposure_times,
                        corrected,
                        alpha=0.3,
                        linewidth=0.8,
                        color=None,
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
            os.path.join(coeffs_dir, f"roi_correction_board_{board_id}_ext_{ext}.png"),
            dpi=300,
        )
        # fig.show()
        plt.close(fig)

        plt.figure(figsize=(6, 6))
        if bad_pixel_mask is not None:
            for dy in range(y1 - y0):
                for dx in range(x1 - x0):
                    y, x = y0 + dy, x0 + dx
                    if bad_pixel_mask[y, x]:
                        plt.plot(
                            exposure_times,
                            raw_stack[:, dy, dx],
                            alpha=0.3,
                            linewidth=0.8,
                            color="black",
                        )
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
            # plot the locations of the ROI points
            for y in y_indices:
                for x in x_indices:
                    plt.scatter(x, y, s=10, color="blue", alpha=0.5)

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
    # median_files = find_median_files("/Users/frostig/Downloads/flats_20250522")
    # data_dir = "/Users/nlourie/data/winternlc"
    data_dir = "/Users/frostig/Downloads/flats_20250513"
    median_files = find_median_files(data_dir)
    output_dir = os.path.join(data_dir, "coeffs")
    # save_correction_coefficients_parallel(
    # median_files, output_dir, n_jobs=10, exts=[2, 5, 4, 3, 1]
    #   median_files, output_dir, n_jobs=8, exts=[0]
    # )

    roi = (200, 300, 200, 300)
    plot_roi_correction(
        median_files, output_dir, roi, selection="random", n_random_pixels=10
    )
    print("doing the SB data now")
    data_dir = "/Users/frostig/Downloads/flats_20250522"
    median_files = find_median_files(data_dir)
    output_dir = os.path.join(data_dir, "coeffs")
    save_correction_coefficients_parallel(median_files, output_dir, n_jobs=8, exts=[1])

    roi = (200, 300, 200, 300)
    plot_roi_correction(
        median_files, output_dir, roi, selection="random", n_random_pixels=10
    )
