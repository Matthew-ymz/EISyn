"""Screen the number of non-random SLP PCA modes with climate surrogates.

The screening follows the dimension-reduction logic used by Vejmelka et al.
(2015) and Runge et al. (2015): each grid-point time series keeps its temporal
power spectrum, independent random phases destroy spatial dependence, and the
ordered data eigenvalues are compared with the corresponding surrogate nulls.

This fast Fourier screen is intended to locate the component-count boundary.
The original confirmatory test used at least 20,000 AR surrogates so that
Bonferroni-adjusted empirical p-values were sufficiently resolved.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.linalg import eigvalsh
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reproduce_runge2015_gateways import (  # noqa: E402
    detrend_time_axis,
    latitude_area_weights,
    load_daily_slp,
    standardize_daily_anomalies,
)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


@dataclass(frozen=True)
class SpectrumScreen:
    label: str
    start_year: int
    end_year: int
    n_months: int
    last_month: str
    data_eigenvalues: np.ndarray
    surrogate_eigenvalues: np.ndarray
    exceedance_counts: np.ndarray
    zero_exceedance_prefix: int
    median_crossing_prefix: int


@dataclass(frozen=True)
class ArNullModel:
    coefficients: np.ndarray
    innovation_scale: np.ndarray
    orders: np.ndarray


def top_covariance_eigenvalues(matrix: np.ndarray, n_eigenvalues: int) -> np.ndarray:
    """Return descending non-zero covariance eigenvalues via the time Gram matrix."""

    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("matrix must have shape [time, space].")
    if values.shape[0] < 2:
        raise ValueError("matrix must contain at least two time samples.")
    count = min(int(n_eigenvalues), values.shape[0] - 1)
    if count < 1:
        raise ValueError("n_eigenvalues must be positive.")
    centered = values - values.mean(axis=0, keepdims=True)
    gram = centered @ centered.T / float(values.shape[0] - 1)
    eigenvalues = eigvalsh(
        gram,
        subset_by_index=[values.shape[0] - count, values.shape[0] - 1],
        check_finite=False,
    )
    return np.asarray(eigenvalues[::-1], dtype=np.float64)


def phase_randomized_surrogate(matrix: np.ndarray, seed: int) -> np.ndarray:
    """Independently randomize Fourier phases while retaining each power spectrum."""

    values = np.asarray(matrix, dtype=np.float32)
    spectrum = np.fft.rfft(values, axis=0)
    rng = np.random.default_rng(int(seed))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape)
    phases[0, :] = 0.0
    if values.shape[0] % 2 == 0:
        phases[-1, :] = 0.0
    surrogate = np.fft.irfft(
        spectrum * np.exp(1j * phases),
        n=values.shape[0],
        axis=0,
    )
    return np.asarray(surrogate, dtype=np.float32)


def fit_independent_ar_bic(matrix: np.ndarray, max_order: int = 30) -> ArNullModel:
    """Fit independent AR orders by vectorized Yule-Walker BIC selection."""

    values = np.asarray(matrix, dtype=np.float64)
    values = values - values.mean(axis=0, keepdims=True)
    n_time, n_series = values.shape
    order_limit = min(int(max_order), max(0, n_time // 4))
    autocov = np.empty((order_limit + 1, n_series), dtype=np.float64)
    for lag in range(order_limit + 1):
        autocov[lag] = np.mean(values[lag:] * values[: n_time - lag], axis=0)
    autocov[0] = np.maximum(autocov[0], 1.0e-12)

    all_coefficients = np.zeros((order_limit, order_limit, n_series), dtype=np.float32)
    residual_variances = np.empty((order_limit + 1, n_series), dtype=np.float64)
    residual_variances[0] = autocov[0]
    previous = np.empty((0, n_series), dtype=np.float64)
    variance = autocov[0].copy()
    for order in range(1, order_limit + 1):
        if order == 1:
            numerator = autocov[1]
        else:
            numerator = autocov[order] - np.sum(
                previous * autocov[1:order][::-1],
                axis=0,
            )
        reflection = numerator / np.maximum(variance, 1.0e-12)
        reflection = np.clip(reflection, -0.999, 0.999)
        current = np.empty((order, n_series), dtype=np.float64)
        if order > 1:
            current[:-1] = previous - reflection[None, :] * previous[::-1]
        current[-1] = reflection
        variance = np.maximum(variance * (1.0 - reflection**2), 1.0e-12)
        residual_variances[order] = variance
        all_coefficients[order - 1, :order] = current.astype(np.float32)
        previous = current

    penalties = (
        np.arange(order_limit + 1, dtype=np.float64)[:, None]
        * np.log(float(n_time))
        / float(n_time)
    )
    bic = np.log(residual_variances) + penalties
    orders = np.argmin(bic, axis=0).astype(np.int16)
    selected = np.zeros((order_limit, n_series), dtype=np.float32)
    for order in range(1, order_limit + 1):
        mask = orders == order
        if np.any(mask):
            indices = np.flatnonzero(mask)
            selected[np.ix_(np.arange(order), indices)] = all_coefficients[
                order - 1, :order, :
            ][:, indices]
    selected_variance = residual_variances[orders, np.arange(n_series)]
    return ArNullModel(
        coefficients=selected,
        innovation_scale=np.sqrt(selected_variance).astype(np.float32),
        orders=orders,
    )


def ar_randomized_surrogate(
    model: ArNullModel,
    *,
    n_time: int,
    seed: int,
    burn_in: int = 100,
) -> np.ndarray:
    """Simulate mutually independent AR processes with a short spin-up."""

    coefficients = np.asarray(model.coefficients, dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    n_series = coefficients.shape[1]
    history = np.zeros((coefficients.shape[0], n_series), dtype=np.float32)
    output = np.empty((int(n_time), n_series), dtype=np.float32)
    for step in range(int(burn_in) + int(n_time)):
        innovation = rng.normal(size=n_series).astype(np.float32)
        current = np.sum(coefficients * history, axis=0)
        current += innovation * model.innovation_scale
        if step >= int(burn_in):
            output[step - int(burn_in)] = current
        if len(history):
            history[1:] = history[:-1]
            history[0] = current
    return output


def leading_true_prefix(mask: np.ndarray) -> int:
    """Count consecutive true values from the leading eigenvalue."""

    values = np.asarray(mask, dtype=bool)
    false_indices = np.flatnonzero(~values)
    return int(false_indices[0]) if len(false_indices) else int(len(values))


def standardize_monthly_anomalies(slp: xr.DataArray) -> xr.DataArray:
    """Remove calendar-month mean and variance as in the published SLP screen."""

    field = slp.assign_coords(calendar_month=("time", np.asarray(slp.time.dt.month, dtype=int)))
    climatology = field.groupby("calendar_month").mean("time")
    anomaly = field.groupby("calendar_month") - climatology
    scale = anomaly.groupby("calendar_month").std("time")
    scale = scale.where(np.isfinite(scale) & (scale > 0.0), 1.0)
    return (anomaly.groupby("calendar_month") / scale).fillna(0.0).drop_vars("calendar_month")


def prepare_monthly_matrix(
    data_dir: Path,
    start_year: int,
    end_year: int,
    *,
    preprocessing: str,
) -> tuple[np.ndarray, str]:
    """Prepare the monthly field under the paper or legacy experiment pipeline."""

    daily = load_daily_slp(data_dir, int(start_year), int(end_year))
    if preprocessing == "paper":
        last_timestamp = np.datetime64(daily["time"].values[-1], "D")
        last_month = last_timestamp.astype("datetime64[M]")
        next_month = last_month + np.timedelta64(1, "M")
        if last_timestamp + np.timedelta64(1, "D") < next_month.astype("datetime64[D]"):
            daily = daily.sel(time=daily.time < last_month)
        monthly = daily.resample(time="MS").mean()
        monthly = detrend_time_axis(standardize_monthly_anomalies(monthly))
        monthly = monthly.sel(lat=(monthly.lat > -90.0) & (monthly.lat < 90.0))
    elif preprocessing == "legacy":
        standardized = detrend_time_axis(standardize_daily_anomalies(daily))
        monthly = standardized.resample(time="MS").mean()
    else:
        raise ValueError("preprocessing must be 'paper' or 'legacy'.")
    weights = latitude_area_weights(monthly["lat"].values)
    weighted = np.asarray(monthly.values, dtype=np.float32) * weights[None, :, None]
    matrix = weighted.reshape(weighted.shape[0], -1)
    matrix -= matrix.mean(axis=0, keepdims=True)
    last_month = str(np.datetime_as_string(monthly["time"].values[-1], unit="M"))
    return matrix, last_month


def screen_matrix(
    matrix: np.ndarray,
    *,
    label: str,
    start_year: int,
    end_year: int,
    last_month: str,
    n_surrogates: int,
    n_eigenvalues: int,
    seed: int,
    workers: int,
    surrogate_kind: str,
    max_ar_order: int,
) -> SpectrumScreen:
    """Compute the observed and phase-randomized null eigenspectra."""

    data_eigenvalues = top_covariance_eigenvalues(matrix, n_eigenvalues)
    surrogate_eigenvalues = np.empty(
        (int(n_surrogates), len(data_eigenvalues)),
        dtype=np.float32,
    )
    ar_model = (
        fit_independent_ar_bic(matrix, max_order=max_ar_order)
        if surrogate_kind == "ar"
        else None
    )
    if ar_model is not None:
        counts = np.bincount(ar_model.orders, minlength=max_ar_order + 1)
        print(f"[{label}] AR order counts: {counts.tolist()}", flush=True)

    def run_one(index: int) -> tuple[int, np.ndarray]:
        if surrogate_kind == "ar":
            assert ar_model is not None
            surrogate = ar_randomized_surrogate(
                ar_model,
                n_time=len(matrix),
                seed=int(seed) + index,
            )
        elif surrogate_kind == "fourier":
            surrogate = phase_randomized_surrogate(matrix, int(seed) + index)
        else:
            raise ValueError("surrogate_kind must be 'ar' or 'fourier'.")
        values = top_covariance_eigenvalues(surrogate, len(data_eigenvalues))
        return index, values.astype(np.float32)

    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            futures = [executor.submit(run_one, index) for index in range(int(n_surrogates))]
            completed = 0
            for future in as_completed(futures):
                index, values = future.result()
                surrogate_eigenvalues[index] = values
                completed += 1
                if completed == 1 or completed % max(1, int(n_surrogates) // 10) == 0:
                    print(f"[{label}] surrogates {completed}/{n_surrogates}", flush=True)

    exceedances = np.sum(
        surrogate_eigenvalues >= data_eigenvalues[None, :],
        axis=0,
    )
    zero_prefix = leading_true_prefix(exceedances == 0)
    surrogate_median = np.median(surrogate_eigenvalues, axis=0)
    median_prefix = leading_true_prefix(data_eigenvalues > surrogate_median)
    return SpectrumScreen(
        label=label,
        start_year=int(start_year),
        end_year=int(end_year),
        n_months=int(matrix.shape[0]),
        last_month=last_month,
        data_eigenvalues=data_eigenvalues,
        surrogate_eigenvalues=surrogate_eigenvalues,
        exceedance_counts=exceedances,
        zero_exceedance_prefix=zero_prefix,
        median_crossing_prefix=median_prefix,
    )


def screen_to_summary(screen: SpectrumScreen, alpha: float) -> dict[str, object]:
    n_hypotheses = int(screen.n_months)
    boundary = int(screen.zero_exceedance_prefix)
    lo = max(0, boundary - 3)
    hi = min(len(screen.data_eigenvalues), boundary + 4)
    null_median = np.median(screen.surrogate_eigenvalues, axis=0)
    null_max = np.max(screen.surrogate_eigenvalues, axis=0)
    rows = []
    for index in range(lo, hi):
        rows.append(
            {
                "component_number_1based": index + 1,
                "data_eigenvalue": float(screen.data_eigenvalues[index]),
                "null_median": float(null_median[index]),
                "null_max": float(null_max[index]),
                "data_to_null_median_ratio": float(
                    screen.data_eigenvalues[index] / null_median[index]
                ),
                "exceedances": int(screen.exceedance_counts[index]),
            }
        )
    return {
        "label": screen.label,
        "start_year": screen.start_year,
        "end_year_argument": screen.end_year,
        "n_months": screen.n_months,
        "last_month_included": screen.last_month,
        "n_hypotheses": n_hypotheses,
        "bonferroni_alpha": float(alpha / n_hypotheses),
        "minimum_surrogates_for_empirical_bonferroni_resolution": int(
            math.ceil(n_hypotheses / alpha)
        ),
        "zero_exceedance_prefix": boundary,
        "median_crossing_prefix": int(screen.median_crossing_prefix),
        "boundary_rows": rows,
    }


def plot_screens(screens: Sequence[SpectrumScreen], output_base: Path) -> None:
    """Plot the calibration and extended-period eigenspectrum boundaries."""

    fig, axes = plt.subplots(
        1,
        len(screens),
        figsize=(7.2, 2.75),
        sharey=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.19, top=0.78, wspace=0.28)
    axes_array = np.atleast_1d(axes)
    for panel, (axis, screen) in enumerate(zip(axes_array, screens, strict=True)):
        boundary = int(screen.zero_exceedance_prefix)
        start = max(0, boundary - 12)
        stop = min(len(screen.data_eigenvalues), boundary + 13)
        x = np.arange(start + 1, stop + 1)
        null_median = np.median(screen.surrogate_eigenvalues, axis=0)
        null_low = np.quantile(screen.surrogate_eigenvalues, 0.01, axis=0)
        null_high = np.quantile(screen.surrogate_eigenvalues, 0.99, axis=0)

        axis.fill_between(
            x,
            null_low[start:stop],
            null_high[start:stop],
            color="#AFC5D8",
            alpha=0.55,
            linewidth=0.0,
            label="Surrogate 1–99%",
        )
        axis.plot(
            x,
            null_median[start:stop],
            color="#5B7C99",
            linewidth=1.2,
            label="Surrogate median",
        )
        axis.plot(
            x,
            screen.data_eigenvalues[start:stop],
            color="#D97724",
            marker="o",
            markersize=2.8,
            linewidth=1.35,
            label="SLP data",
        )
        axis.axvline(boundary + 0.5, color="#30343B", linestyle=":", linewidth=1.0)
        axis.text(
            0.98,
            0.96,
            f"screened count = {boundary}",
            ha="right",
            va="top",
            transform=axis.transAxes,
            fontsize=7,
        )
        axis.set_xlabel("Ordered principal component")
        axis.set_ylabel("Covariance eigenvalue")
        axis.text(
            -0.12,
            1.03,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=8,
        )
        axis.text(
            0.02,
            0.96,
            f"{screen.start_year}–{screen.last_month}",
            ha="left",
            va="top",
            transform=axis.transAxes,
            fontsize=7,
        )

    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".png", {"dpi": 600}),
        (".svg", {}),
        (".pdf", {}),
    ):
        fig.savefig(output_base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def parse_period(value: str) -> tuple[str, int, int]:
    pieces = value.split(":")
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("period must be LABEL:START_YEAR:END_YEAR")
    label, start, end = pieces
    return label, int(start), int(end)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--period",
        action="append",
        type=parse_period,
        default=None,
        help="LABEL:START_YEAR:END_YEAR; repeat for paired screens.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/ncep_reanalysis_slp"))
    parser.add_argument("--n-surrogates", type=int, default=256)
    parser.add_argument("--n-eigenvalues", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--workers", type=int, default=min(5, os.cpu_count() or 1))
    parser.add_argument("--preprocessing", choices=["paper", "legacy"], default="paper")
    parser.add_argument("--surrogate", choices=["ar", "fourier"], default="ar")
    parser.add_argument("--max-ar-order", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/runge/slp_pca_significance"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_surrogates < 1 or args.n_eigenvalues < 1:
        raise ValueError("n-surrogates and n-eigenvalues must be positive.")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")

    periods = args.period or [
        ("Runge calibration", 1948, 2011),
        ("Current SLP", 1948, 2026),
    ]
    screens: list[SpectrumScreen] = []
    for position, (label, start_year, end_year) in enumerate(periods):
        matrix, last_month = prepare_monthly_matrix(
            args.data_dir,
            start_year,
            end_year,
            preprocessing=args.preprocessing,
        )
        screens.append(
            screen_matrix(
                matrix,
                label=label,
                start_year=start_year,
                end_year=end_year,
                last_month=last_month,
                n_surrogates=args.n_surrogates,
                n_eigenvalues=args.n_eigenvalues,
                seed=args.seed + position * args.n_surrogates,
                workers=args.workers,
                surrogate_kind=args.surrogate,
                max_ar_order=args.max_ar_order,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": {
            "surrogate": (
                f"independent AR(0-{args.max_ar_order}) processes with BIC order selection"
                if args.surrogate == "ar"
                else "independent Fourier phase randomization per grid point"
            ),
            "preprocessing": (
                "monthly calendar-month mean/variance anomaly, linear detrending, "
                "pole exclusion, sqrt(cos(latitude)) weighting, incomplete final "
                "month excluded"
                if args.preprocessing == "paper"
                else "daily calendar-day mean/variance anomaly, linear detrending, "
                "monthly averaging, sqrt(cos(latitude)) weighting"
            ),
            "n_surrogates": int(args.n_surrogates),
            "n_eigenvalues": int(args.n_eigenvalues),
            "alpha": float(args.alpha),
            "seed": int(args.seed),
            "interpretation": (
                "Fast boundary screen calibrated against the published 60-component "
                "period; not a replacement for the >=20,000 AR-surrogate Bonferroni test."
            ),
        },
        "periods": [screen_to_summary(screen, args.alpha) for screen in screens],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output_dir / "spectra.npz",
        **{
            f"data_{index}": screen.data_eigenvalues
            for index, screen in enumerate(screens)
        },
        **{
            f"surrogates_{index}": screen.surrogate_eigenvalues
            for index, screen in enumerate(screens)
        },
    )
    plot_screens(screens, args.output_dir / "slp_pca_significance_screen")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
