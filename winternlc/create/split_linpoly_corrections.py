from __future__ import annotations

from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

# Convenient alias so the signature stays readable
ArrayLike = npt.ArrayLike


# minimum linear slope
def calculate_min_linear_slope(
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
    # Calculate the minimum linear slope
    min_linear_slope = (saturation_level - bias_level) / max_exposure_time
    return min_linear_slope


MIN_LINEAR_SLOPE = calculate_min_linear_slope(
    max_exposure_time=12,  # seconds
    bias_level=12000,  # counts
    saturation_level=58000,  # counts
)
print(f"Minimum linear slope: {MIN_LINEAR_SLOPE:.2f} counts/s")


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
    min_linear_slope: float = MIN_LINEAR_SLOPE,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Compute the polynomial correction coefficients for a pixel series.

    Parameters
    ----------
    expt : ArrayLike
        Exposure times corresponding to `pixel_series`.
    counts : ArrayLike
        Pixel values over time (same length as `expt`).
    linear_region_min_fraction : float
        Minimum fraction of counts for the linear‑response region.
    linear_region_max_fraction : float
        Maximum fraction of counts for the linear‑response region.
    order_lower : int
        Polynomial order for the lower segment.
    order_upper : int
        Polynomial order for the upper segment.
    crossover_point : int | float
        Count value where the lower and upper fits meet.
    overlap : int | float
        Half‑width of the overlap window around `crossover_point`.
    ceiling : int | float | None
        Saturation ceiling (if ``None``, infer from data).
    resample_points : int | None
        Number of points for the resampled grid.
    resample_interpolation_kind : str
        Interpolation method passed to ``scipy.interpolate.interp1d`` (e.g. "linear").
    residual_std_threshold : float
        Standard deviation threshold for residuals to determine if the fit is bad.
    min_linear_slope : float
        Minimum slope for the linear fit to be considered valid.
    verbose : bool
        If `True`, print additional information during processing.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, bool]
        ``(coeffs_lower, coeffs_upper, is_bad)`` where each coefficients
        array is the NumPy polyfit output for its segment, and ``is_bad`` is
        a flag indicating an unusable or failed fit.
    """
    try:
        # if the whole process fails, then the correction is bad for this pixel

        # ------------------------------------------------------------------
        # Prepare the input data
        # ------------------------------------------------------------------
        # Make sure the input counts are treated as floats
        counts = np.asarray(counts, dtype=float)
        expt = np.asarray(expt, dtype=float)
        # Check that counts and expt have the same length
        if counts.shape != expt.shape:
            raise ValueError(
                f"Counts and exposure times must have the same length, "
                f"got {counts.shape} and {expt.shape}."
            )
        # Set up a ceiling: the max counts that the correction will be applied to
        if ceiling is None:
            ceiling = counts.max()
        max_counts = counts.max()

        # ------------------------------------------------------------------
        # Fit the linear region
        # ------------------------------------------------------------------
        linear_region_min_counts = max_counts * linear_region_min_fraction
        linear_region_max_counts = max_counts * linear_region_max_fraction
        mask_lin = (counts > linear_region_min_counts) & (
            counts < linear_region_max_counts
        )
        linear_fit = np.polyfit(expt[mask_lin], counts[mask_lin], 1)
        mu_cal_all = np.polyval(linear_fit, expt)  # same length as counts

        # Some quality cuts on the linear fit
        # if the fit is negative or below a certain threshold, we consider it bad
        if linear_fit[0] < min_linear_slope or np.isnan(linear_fit).any():
            raise ValueError(
                f"Linear fit slope {linear_fit[0]:.2f} is below the minimum "
                f"slope {min_linear_slope:.2f} or contains NaN values."
            )
        # ------------------------------------------------------------------
        # Resample the data if requested
        # ------------------------------------------------------------------
        # The nonlinearity will be fit to mu_raw (the raw counts) vs. mu_cal
        # (the same points extrapolated to the linear fit). If resampling is
        # requested, mu_raw will be a uniform grid of counts, and mu_cal will
        # be the corresponding linear fit values, otherwise they will be the
        # original counts and their linear fit values. The resampling puts the
        # fits on equal footing regardless of the illumination at each pixel
        # which can lead to very different sampling of the different parts of
        # the ramp.

        # Note: only use points with mu_raw *below the ceiling* for the resampling.
        if resample_points is not None:
            # Should the resampling be logarithmic or linear?
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

        # ------------------------------------------------------------------
        # Fit the lower and upper segments of the ramp with separate polynomials
        # ------------------------------------------------------------------
        # Overlap the fitted regions to incentivize continuity
        lower_fit_mask = mu_raw < (crossover_point + overlap / 2)
        upper_fit_mask = mu_raw >= (crossover_point - overlap / 2)

        fit_lower = np.polyfit(
            mu_raw[lower_fit_mask], mu_cal[lower_fit_mask], order_lower
        )
        fit_upper = np.polyfit(
            mu_raw[upper_fit_mask], mu_cal[upper_fit_mask], order_upper
        )

        # diagnostic plot
        if verbose:

            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))

            plt.plot(
                mu_raw,
                mu_cal,
                "o",
                label="Linear Interpolation",
            )
            plt.plot(
                counts,
                mu_cal_all,
                "o",
                label="Data",
            )

            x_fit_lower = np.linspace(
                mu_raw[lower_fit_mask].min(),
                crossover_point,
                # mu_raw[lower_fit_mask].max(),
                1000,
            )
            x_fit_upper = np.linspace(
                # mu_raw[upper_fit_mask].min(),
                crossover_point,
                mu_raw[upper_fit_mask].max(),
                1000,
            )
            plt.plot(
                x_fit_lower,
                np.polyval(fit_lower, x_fit_lower),
                label=f"Lower Fit (order {order_lower})",
                # color="magenta",
            )
            plt.plot(
                x_fit_upper,
                np.polyval(fit_upper, x_fit_upper),
                label=f"Upper Fit (order {order_upper})",
                # color="grereden",
            )
            plt.axvline(crossover_point, color="red", linestyle="--", label="Crossover")
            plt.xlabel("Counts (mu_raw)")
            plt.ylabel("Linear Fit (mu_cal)")
            plt.xlim(0, ceiling)
            plt.ylim(0, ceiling)
            plt.legend()
            plt.title("Polynomial Fits to Nonlinearity Correction")
            plt.show()
        # ------------------------------------------------------------------
        # Apply correction and calculate residuals
        # ------------------------------------------------------------------
        corrected_counts = apply_nlc_correction(
            counts,
            fit_lower,
            fit_upper,
            crossover_point,
            ceiling=ceiling,
        )

        residuals = mu_cal_all - corrected_counts
        # calculate the standard deviation of the residuals
        std_mask = counts < 40_000  # or make this tunable
        residuals_std = np.std(residuals[std_mask])
        # If the residuals are too large, we consider the fit bad
        is_bad = (
            np.isnan(fit_lower).any()
            or np.isnan(fit_upper).any()
            or residuals_std > residual_std_threshold
        )
        if verbose:
            # plot the residuals
            plt.figure(figsize=(10, 6))
            plt.plot(
                mu_cal_all,
                residuals,
                "o",
                label="Residuals",
            )
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
        # -------------------------------------------------------------------
        # Return the results
        # -------------------------------------------------------------------
        return fit_lower, fit_upper, is_bad

    except Exception as e:
        # If any error occurs, we consider the fit bad
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
    """
    Apply the nonlinearity correction (NLC) to the counts using the polynomial coefficients.

    Parameters
    ----------
    counts : ArrayLike
        The raw pixel values.
    coeffs_lower : ArrayLike
        Polynomial coefficients for the lower segment of the ramp.
    coeffs_upper : ArrayLike
        Polynomial coefficients for the upper segment of the ramp.
    crossover_point : float
        The count value where the lower and upper fits meet.

    Returns
    -------
    ArrayLike
        The corrected pixel values.
    """
    # make sure the input counts are treated as floats
    counts = np.asarray(counts, dtype=float)
    # Calculate the corrected values using the polynomial coefficients
    corrected = np.where(
        counts < crossover_point,
        np.polyval(coeffs_lower, counts),
        np.polyval(coeffs_upper, counts),
    )
    # Apply the ceiling to the corrected values:
    # above the ceiling, the corrected counts are just the raw counts
    if ceiling is not None:
        above = (corrected > ceiling) | (counts > ceiling)
        corrected[above] = counts[above]

    return corrected
