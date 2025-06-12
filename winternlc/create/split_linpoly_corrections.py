from __future__ import annotations

from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

# Convenient alias so the signature stays readable
ArrayLike = npt.ArrayLike


def get_min_linear_slope(
    max_exposure_time=12,
    bias_level=12000,
    saturation_level=58000,
) -> float:
    """
    Approximate the minimum linear slope for a pixel series based on
    nominal values for the bias and saturation, and an estimate of the
    time by which the pixel should saturate.

    Parameters:
        max_exposure_time : float
            The maximum exposure time in seconds.
        bias_level : float
            The bias level in counts.
        saturation_level : float
            The saturation level in counts.
    Returns:
        float
            The minimum linear slope in counts per second.
    """
    return (saturation_level - bias_level) / max_exposure_time


def compute_correction_poly(
    expt: ArrayLike,
    counts: ArrayLike,
    linear_region_min_fraction: float = 0.33,
    linear_region_max_fraction: float = 0.85,
    order_lower: int = 11,
    order_upper: int = 5,
    crossover_point: int | float = 20_000,
    overlap: int | float = 10_000,
    ceiling: int | float | None = 55_000,
    resample_points: int | None = 250,
    resample_interpolation_kind: str = "log",
    residual_std_threshold: float = 500.0,
    min_linear_slope: float | None = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Compute the polynomial correction coefficients for a pixel series.
    """
    try:
        if min_linear_slope is None:
            min_linear_slope = get_min_linear_slope()

        counts = np.asarray(counts, dtype=float)
        expt = np.asarray(expt, dtype=float)
        if counts.shape != expt.shape:
            raise ValueError(
                f"Counts and exposure times must have the same length, "
                f"got {counts.shape} and {expt.shape}."
            )
        if ceiling is None:
            ceiling = counts.max()
        max_counts = counts.max()

        linear_region_min_counts = max_counts * linear_region_min_fraction
        linear_region_max_counts = max_counts * linear_region_max_fraction
        mask_lin = (counts > linear_region_min_counts) & (
            counts < linear_region_max_counts
        )
        linear_fit = np.polyfit(expt[mask_lin], counts[mask_lin], 1)
        mu_cal_all = np.polyval(linear_fit, expt)

        if linear_fit[0] < min_linear_slope or np.isnan(linear_fit).any():
            raise ValueError(
                f"Linear fit slope {linear_fit[0]:.2f} is below the minimum "
                f"slope {min_linear_slope:.2f} or contains NaN values."
            )

        if resample_points is not None:
            grid = np.geomspace if resample_interpolation_kind == "log" else np.linspace
            mu_raw = grid(counts.min(), ceiling, resample_points)
            mu_cal = interp1d(
                counts,
                mu_cal_all,
                kind=resample_interpolation_kind,
                bounds_error=False,
                fill_value="extrapolate",
            )(mu_raw)
        else:
            valid = counts < ceiling
            mu_raw = counts[valid]
            mu_cal = mu_cal_all[valid]

        lower_fit_mask = mu_raw < (crossover_point + overlap / 2)
        upper_fit_mask = mu_raw >= (crossover_point - overlap / 2)

        fit_lower = np.polyfit(
            mu_raw[lower_fit_mask], mu_cal[lower_fit_mask], order_lower
        )
        fit_upper = np.polyfit(
            mu_raw[upper_fit_mask], mu_cal[upper_fit_mask], order_upper
        )

        if verbose:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(mu_raw, mu_cal, "o", label="Linear Interpolation")
            plt.plot(counts, mu_cal_all, "o", label="Data")

            x_fit_lower = np.linspace(
                mu_raw[lower_fit_mask].min(), crossover_point, 1000
            )
            x_fit_upper = np.linspace(
                crossover_point, mu_raw[upper_fit_mask].max(), 1000
            )
            plt.plot(
                x_fit_lower,
                np.polyval(fit_lower, x_fit_lower),
                label=f"Lower Fit (order {order_lower})",
            )
            plt.plot(
                x_fit_upper,
                np.polyval(fit_upper, x_fit_upper),
                label=f"Upper Fit (order {order_upper})",
            )
            plt.axvline(crossover_point, color="red", linestyle="--", label="Crossover")
            plt.xlabel("Counts (mu_raw)")
            plt.ylabel("Linear Fit (mu_cal)")
            plt.xlim(0, ceiling)
            plt.ylim(0, ceiling)
            plt.legend()
            plt.title("Polynomial Fits to Nonlinearity Correction")
            plt.show()

        corrected_counts = apply_nlc_correction(
            counts,
            fit_lower,
            fit_upper,
            crossover_point,
            ceiling=ceiling,
        )

        residuals = mu_cal_all - corrected_counts
        std_mask = counts < 40_000
        residuals_std = np.std(residuals[std_mask])
        is_bad = (
            np.isnan(fit_lower).any()
            or np.isnan(fit_upper).any()
            or residuals_std > residual_std_threshold
        )

        if verbose:
            plt.figure(figsize=(10, 6))
            plt.plot(mu_cal_all, residuals, "o", label="Residuals")
            plt.axhline(
                residuals_std, color="red", linestyle="--", label="Residuals Std Dev"
            )
            plt.axhline(-residuals_std, color="red", linestyle="--")
            plt.xlabel("Counts (mu_raw)")
            plt.ylabel("Residuals (mu_cal - corrected_counts)")
            plt.title("Residuals of Nonlinearity Correction")
            ylim = np.max(np.abs(residuals[std_mask])) * 1.1
            plt.ylim(-ylim, ylim)
            plt.legend()
            plt.show()

        return fit_lower, fit_upper, is_bad

    except Exception as e:
        if verbose:
            print(f"Error during polynomial fitting: {e}")
        return np.full(order_lower + 1, np.nan), np.full(order_upper + 1, np.nan), True


def apply_nlc_correction(
    counts: ArrayLike,
    coeffs_lower: ArrayLike,
    coeffs_upper: ArrayLike,
    crossover_point: float,
    ceiling: float | None = None,
) -> ArrayLike:
    counts = np.asarray(counts, dtype=float)
    corrected = np.where(
        counts < crossover_point,
        np.polyval(coeffs_lower, counts),
        np.polyval(coeffs_upper, counts),
    )
    if ceiling is not None:
        above = (corrected > ceiling) | (counts > ceiling)
        corrected[above] = counts[above]
    return corrected
