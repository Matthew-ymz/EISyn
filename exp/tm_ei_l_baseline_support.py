from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from yrd.transport_map import estimate_mutual_information_transport_map, lift_transport_source_features


DEFAULT_DYNAMICS_SPECS: tuple[dict[str, Any], ...] = (
    {"dynamics": "identity", "source_dim": 1, "target_dim": 1, "label": "identity"},
    {"dynamics": "linear_gain", "source_dim": 1, "target_dim": 1, "gain": 0.5, "label": "gain=0.5"},
    {"dynamics": "linear_gain", "source_dim": 1, "target_dim": 1, "gain": 2.0, "label": "gain=2.0"},
    {"dynamics": "product", "source_dim": 2, "target_dim": 1, "label": "product"},
    {"dynamics": "tanh", "source_dim": 1, "target_dim": 1, "beta": 1.0, "label": "tanh"},
)

DEFAULT_BASELINES: tuple[str, ...] = (
    "null",
    "identity",
    "gain_matched",
    "variance_matched",
    "shuffled_target",
    "analytic_volume",
)


@dataclass(frozen=True)
class BaselineTarget:
    target: np.ndarray
    metadata: dict[str, Any]


def sample_uniform_source(
    *,
    l_value: float,
    n_samples: int,
    source_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    half_width = float(l_value) / 2.0
    return rng.uniform(-half_width, half_width, size=(int(n_samples), int(source_dim)))


def resolve_l_sample_count(
    *,
    l_value: float,
    source_dim: int,
    sample_count_mode: str,
    reference_l: float,
    reference_n_samples: int,
    min_n_samples: int,
    max_n_samples: int | None = None,
) -> int:
    """Resolve the sample count for fixed-count or fixed-density L sweeps."""

    if int(reference_n_samples) <= 0:
        raise ValueError("reference_n_samples must be positive.")
    if int(min_n_samples) <= 0:
        raise ValueError("min_n_samples must be positive.")
    if float(reference_l) <= 0.0:
        raise ValueError("reference_l must be positive.")
    if int(source_dim) <= 0:
        raise ValueError("source_dim must be positive.")

    if sample_count_mode == "fixed":
        resolved = int(reference_n_samples)
    elif sample_count_mode == "fixed_density":
        volume_ratio = (float(l_value) / float(reference_l)) ** int(source_dim)
        resolved = int(np.ceil(float(reference_n_samples) * volume_ratio))
    else:
        raise ValueError(f"Unknown sample_count_mode: {sample_count_mode}")

    resolved = max(int(min_n_samples), resolved)
    if max_n_samples is not None:
        resolved = min(int(max_n_samples), resolved)
    return resolved


def simulate_known_dynamics(
    dynamics: str,
    source: np.ndarray,
    *,
    noise_std: float,
    rng: np.random.Generator,
    gain: float = 1.0,
    beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    source_array = np.asarray(source, dtype=float)
    if source_array.ndim != 2:
        raise ValueError("source must be a 2D array.")

    if dynamics == "identity":
        signal = source_array[:, [0]]
    elif dynamics == "linear_gain":
        signal = float(gain) * source_array[:, [0]]
    elif dynamics == "product":
        if source_array.shape[1] < 2:
            raise ValueError("product dynamics requires at least two source dimensions.")
        signal = source_array[:, [0]] * source_array[:, [1]]
    elif dynamics == "tanh":
        signal = np.tanh(float(beta) * source_array[:, [0]])
    else:
        raise ValueError(f"Unknown dynamics: {dynamics}")

    target = signal + float(noise_std) * rng.normal(size=signal.shape)
    return signal, target


def transport_source_features(source: np.ndarray) -> np.ndarray:
    source_array = np.asarray(source, dtype=float)
    if source_array.ndim != 2:
        raise ValueError("source must be a 2D array.")
    if source_array.shape[1] in (1, 2):
        return lift_transport_source_features(source_array)
    return source_array


def estimate_transport_ei(source: np.ndarray, target: np.ndarray) -> float:
    summary = estimate_mutual_information_transport_map(
        transport_source_features(source),
        np.asarray(target, dtype=float),
    )
    return float(summary["mi_hat"])


def _identity_projection(source_dim: int, target_dim: int) -> np.ndarray:
    matrix = np.zeros((int(source_dim), int(target_dim)), dtype=float)
    for index in range(min(int(source_dim), int(target_dim))):
        matrix[index, index] = 1.0
    if int(target_dim) > int(source_dim):
        matrix[0, int(source_dim) :] = 1.0
    return matrix


def _linear_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_centered = source - source.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    coefficient, *_ = np.linalg.lstsq(source_centered, target_centered, rcond=None)
    return coefficient


def build_baseline_target(
    *,
    baseline: str,
    source: np.ndarray,
    target_signal: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> BaselineTarget:
    source_array = np.asarray(source, dtype=float)
    signal_array = np.asarray(target_signal, dtype=float)
    if source_array.ndim != 2 or signal_array.ndim != 2:
        raise ValueError("source and target_signal must be 2D arrays.")
    if source_array.shape[0] != signal_array.shape[0]:
        raise ValueError("source and target_signal must share the sample axis.")

    target_dim = signal_array.shape[1]
    noise = float(noise_std) * rng.normal(size=signal_array.shape)
    metadata: dict[str, Any] = {}

    if baseline == "null":
        base_signal = np.zeros_like(signal_array)
    elif baseline == "identity":
        projection = _identity_projection(source_array.shape[1], target_dim)
        base_signal = source_array @ projection
        metadata["projection"] = projection.tolist()
    elif baseline == "gain_matched":
        coefficient = _linear_fit(source_array, signal_array)
        base_signal = (source_array - source_array.mean(axis=0, keepdims=True)) @ coefficient
        base_signal += signal_array.mean(axis=0, keepdims=True)
        metadata["gain_matrix"] = coefficient.tolist()
    elif baseline == "variance_matched":
        direction = np.ones((source_array.shape[1], 1), dtype=float)
        direction /= float(np.linalg.norm(direction))
        projection = source_array @ direction
        projection = projection - projection.mean(axis=0, keepdims=True)
        projection_std = np.maximum(projection.std(axis=0, ddof=1), 1e-12)
        signal_std = np.maximum(signal_array.std(axis=0, ddof=1, keepdims=True), 1e-12)
        base_signal = projection / projection_std * signal_std
        base_signal += signal_array.mean(axis=0, keepdims=True)
        metadata["direction"] = direction[:, 0].tolist()
    elif baseline == "shuffled_target":
        permutation = rng.permutation(signal_array.shape[0])
        base_signal = signal_array[permutation]
    else:
        raise ValueError(f"Unsupported simulated baseline: {baseline}")

    return BaselineTarget(target=base_signal + noise, metadata=metadata)


def fit_log_l_slope(frame: pd.DataFrame, *, value_column: str) -> dict[str, float]:
    if frame.empty:
        return {"slope_log_l": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    x = np.log(frame["L"].to_numpy(dtype=float))
    y = frame[value_column].to_numpy(dtype=float)
    if len(np.unique(x)) < 2:
        return {"slope_log_l": 0.0, "intercept": float(y.mean()), "r2": float("nan")}
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return {"slope_log_l": float(slope), "intercept": float(intercept), "r2": float(r2)}


def corrected_ei_table(runs: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["dynamics", "label", "L", "seed", "noise_mode", "noise_std"]
    available_keys = [column for column in key_columns if column in runs.columns and column in baselines.columns]
    merged = runs.merge(baselines, on=available_keys, how="inner")
    merged["ei_corrected"] = merged["ei_tm"] - merged["baseline_ei_tm"]
    return merged


def summarize_l_sensitivity(corrected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["dynamics", "label", "noise_mode", "baseline"]
    for keys, group in corrected.groupby(group_columns, sort=False):
        dynamics, label, noise_mode, baseline = keys
        raw_by_l = group.groupby("L", as_index=False)["ei_tm"].mean()
        corrected_by_l = group.groupby("L", as_index=False)["ei_corrected"].mean()
        raw_slope = fit_log_l_slope(raw_by_l, value_column="ei_tm")
        corrected_slope = fit_log_l_slope(corrected_by_l, value_column="ei_corrected")
        raw_values = raw_by_l["ei_tm"].to_numpy(dtype=float)
        corrected_values = corrected_by_l["ei_corrected"].to_numpy(dtype=float)
        raw_cv = float(raw_values.std(ddof=0) / max(abs(raw_values.mean()), 1e-12))
        corrected_cv = float(corrected_values.std(ddof=0) / max(abs(corrected_values.mean()), 1e-12))
        rows.append(
            {
                "dynamics": dynamics,
                "label": label,
                "noise_mode": noise_mode,
                "baseline": baseline,
                "raw_slope_log_l": raw_slope["slope_log_l"],
                "corrected_slope_log_l": corrected_slope["slope_log_l"],
                "slope_abs_reduction": abs(raw_slope["slope_log_l"]) - abs(corrected_slope["slope_log_l"]),
                "raw_cv_l": raw_cv,
                "corrected_cv_l": corrected_cv,
                "cv_reduction": raw_cv - corrected_cv,
                "corrected_range_l": float(corrected_values.max() - corrected_values.min()),
            }
        )
    return pd.DataFrame(rows).sort_values(["noise_mode", "label", "corrected_slope_log_l"])


def select_best_corrected_baselines(
    summary: pd.DataFrame,
    *,
    noise_mode: str = "fixed",
    max_abs_slope: float | None = None,
    max_corrected_range: float | None = None,
) -> pd.DataFrame:
    """Select the baseline with the flattest corrected EI-vs-L slope per dynamics label."""

    filtered = summary[summary["noise_mode"].eq(noise_mode)].copy()
    if filtered.empty:
        return pd.DataFrame()
    filtered["abs_corrected_slope"] = filtered["corrected_slope_log_l"].abs()
    if max_abs_slope is not None:
        filtered = filtered[filtered["abs_corrected_slope"] <= float(max_abs_slope)]
    if max_corrected_range is not None and "corrected_range_l" in filtered.columns:
        filtered = filtered[filtered["corrected_range_l"] <= float(max_corrected_range)]
    if filtered.empty:
        return pd.DataFrame()
    sort_columns = ["label", "abs_corrected_slope"]
    if "corrected_cv_l" in filtered.columns:
        sort_columns.append("corrected_cv_l")
    best = (
        filtered.sort_values(sort_columns)
        .groupby("label", as_index=False)
        .first()
        .set_index("label")
    )
    return best


def run_ei_l_baseline_sweep(
    *,
    l_values: Iterable[float],
    dynamics_specs: Iterable[dict[str, Any]] = DEFAULT_DYNAMICS_SPECS,
    baselines: Iterable[str] = DEFAULT_BASELINES,
    n_samples: int = 4096,
    sample_count_mode: str = "fixed",
    reference_l: float = 4.0,
    min_n_samples: int = 512,
    max_n_samples: int | None = None,
    repeats: int = 10,
    base_noise_std: float = 0.05,
    noise_modes: Iterable[str] = ("fixed", "scaled_with_l"),
    seed: int = 20260429,
) -> dict[str, pd.DataFrame]:
    l_values_tuple = tuple(float(value) for value in l_values)
    baseline_names = tuple(str(name) for name in baselines)
    run_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    for spec_index, spec in enumerate(dynamics_specs):
        source_dim = int(spec["source_dim"])
        for noise_mode in noise_modes:
            for l_index, l_value in enumerate(l_values_tuple):
                noise_std = float(base_noise_std) * (float(l_value) if noise_mode == "scaled_with_l" else 1.0)
                analytic_volume_ei = source_dim * np.log(max(float(l_value), 1e-12))
                resolved_n_samples = resolve_l_sample_count(
                    l_value=l_value,
                    source_dim=source_dim,
                    sample_count_mode=str(sample_count_mode),
                    reference_l=float(reference_l),
                    reference_n_samples=int(n_samples),
                    min_n_samples=int(min_n_samples),
                    max_n_samples=max_n_samples,
                )
                for repeat in range(int(repeats)):
                    run_seed = int(seed + 100000 * spec_index + 1000 * l_index + 17 * repeat)
                    rng = np.random.default_rng(run_seed)
                    source = sample_uniform_source(
                        l_value=l_value,
                        n_samples=resolved_n_samples,
                        source_dim=source_dim,
                        rng=rng,
                    )
                    signal, target = simulate_known_dynamics(
                        str(spec["dynamics"]),
                        source,
                        noise_std=noise_std,
                        rng=rng,
                        gain=float(spec.get("gain", 1.0)),
                        beta=float(spec.get("beta", 1.0)),
                    )
                    ei_tm = estimate_transport_ei(source, target)
                    common = {
                        "dynamics": str(spec["dynamics"]),
                        "label": str(spec.get("label", spec["dynamics"])),
                        "L": float(l_value),
                        "seed": run_seed,
                        "repeat": int(repeat),
                        "noise_mode": str(noise_mode),
                        "noise_std": float(noise_std),
                        "n_samples": resolved_n_samples,
                        "sample_count_mode": str(sample_count_mode),
                        "reference_l": float(reference_l),
                        "source_dim": source_dim,
                        "target_dim": int(spec.get("target_dim", 1)),
                    }
                    run_rows.append({**common, "ei_tm": ei_tm})
                    for baseline in baseline_names:
                        if baseline == "analytic_volume":
                            baseline_rows.append(
                                {
                                    **common,
                                    "baseline": baseline,
                                    "baseline_ei_tm": float(analytic_volume_ei),
                                    "baseline_metadata": json.dumps({"source_dim": source_dim}),
                                }
                            )
                            continue
                        baseline_seed_offset = sum((index + 1) * ord(char) for index, char in enumerate(baseline))
                        baseline_target = build_baseline_target(
                            baseline=baseline,
                            source=source,
                            target_signal=signal,
                            noise_std=noise_std,
                            rng=np.random.default_rng(run_seed + 10000000 + baseline_seed_offset),
                        )
                        baseline_rows.append(
                            {
                                **common,
                                "baseline": baseline,
                                "baseline_ei_tm": estimate_transport_ei(source, baseline_target.target),
                                "baseline_metadata": json.dumps(baseline_target.metadata),
                            }
                        )

    runs = pd.DataFrame(run_rows)
    baseline_frame = pd.DataFrame(baseline_rows)
    corrected = corrected_ei_table(runs, baseline_frame)
    summary = summarize_l_sensitivity(corrected)
    return {"runs": runs, "baselines": baseline_frame, "corrected": corrected, "summary": summary}


def write_experiment_artifacts(
    *,
    results: dict[str, pd.DataFrame],
    output_dir: Path,
    notebook_path: str,
    config: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "runs_csv": output_dir / "tm_ei_l_baseline_runs.csv",
        "baselines_csv": output_dir / "tm_ei_l_baseline_baselines.csv",
        "corrected_csv": output_dir / "tm_ei_l_baseline_corrected.csv",
        "summary_csv": output_dir / "tm_ei_l_baseline_summary.csv",
        "manifest_json": output_dir / "tm_ei_l_baseline_manifest.json",
    }
    for key, frame_key in [
        ("runs_csv", "runs"),
        ("baselines_csv", "baselines"),
        ("corrected_csv", "corrected"),
        ("summary_csv", "summary"),
    ]:
        results[frame_key].to_csv(paths[key], index=False)
    manifest = {
        "notebook": notebook_path,
        "config": config,
        "artifacts": {key: path.name for key, path in paths.items() if key != "manifest_json"},
    }
    paths["manifest_json"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return {key: str(path) for key, path in paths.items()}


def raw_and_baseline_curve_table(
    runs: pd.DataFrame,
    baselines: pd.DataFrame,
    *,
    noise_mode: str = "fixed",
    dynamics_labels: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Build a long-form table for raw EI and baseline EI curves."""

    runs_filtered = runs[runs["noise_mode"].eq(noise_mode)].copy()
    baselines_filtered = baselines[baselines["noise_mode"].eq(noise_mode)].copy()
    if dynamics_labels is not None:
        labels = set(dynamics_labels)
        runs_filtered = runs_filtered[runs_filtered["label"].isin(labels)]
        baselines_filtered = baselines_filtered[baselines_filtered["label"].isin(labels)]

    raw = (
        runs_filtered.groupby(["label", "noise_mode", "L"], as_index=False)
        .agg(ei_mean=("ei_tm", "mean"), ei_std=("ei_tm", "std"))
        .assign(series="raw EI", series_kind="raw")
    )
    baseline_curves = (
        baselines_filtered.groupby(["label", "noise_mode", "baseline", "L"], as_index=False)
        .agg(ei_mean=("baseline_ei_tm", "mean"), ei_std=("baseline_ei_tm", "std"))
        .rename(columns={"baseline": "series"})
    )
    baseline_curves["series_kind"] = "baseline"
    raw["baseline"] = ""
    baseline_curves["baseline"] = baseline_curves["series"]
    columns = ["label", "noise_mode", "L", "series", "series_kind", "baseline", "ei_mean", "ei_std"]
    return pd.concat([raw[columns], baseline_curves[columns]], ignore_index=True).sort_values(
        ["label", "series_kind", "series", "L"]
    )


def plot_raw_and_baseline_ei_by_l(
    runs: pd.DataFrame,
    baselines: pd.DataFrame,
    *,
    output_dir: Path,
    noise_mode: str = "fixed",
    dynamics_labels: tuple[str, ...] | None = None,
    max_labels: int | None = None,
) -> tuple[Path, Path, pd.DataFrame]:
    """Plot raw EI and every baseline EI as functions of L."""

    output_dir.mkdir(parents=True, exist_ok=True)
    curve_table = raw_and_baseline_curve_table(
        runs,
        baselines,
        noise_mode=noise_mode,
        dynamics_labels=dynamics_labels,
    )
    if curve_table.empty:
        raise ValueError("No curve rows are available for the requested filters.")
    labels = list(dict.fromkeys(curve_table["label"].tolist()))
    if max_labels is not None:
        labels = labels[: int(max_labels)]
        curve_table = curve_table[curve_table["label"].isin(labels)]

    fig, axes = plt.subplots(len(labels), 1, figsize=(9.4, max(3.0, 2.7 * len(labels))), constrained_layout=True)
    if len(labels) == 1:
        axes = [axes]
    color_map = {
        "raw EI": "#202124",
        "null": "#7f8c8d",
        "identity": "#1f77b4",
        "gain_matched": "#2ca02c",
        "variance_matched": "#d62728",
        "shuffled_target": "#9467bd",
        "analytic_volume": "#8c564b",
    }
    for axis, label in zip(axes, labels):
        label_group = curve_table[curve_table["label"].eq(label)]
        series_order = ["raw EI"] + [
            name for name in DEFAULT_BASELINES if name in set(label_group["series"].tolist())
        ]
        for series in series_order:
            series_group = label_group[label_group["series"].eq(series)].sort_values("L")
            if series_group.empty:
                continue
            is_raw = series == "raw EI"
            axis.plot(
                series_group["L"],
                series_group["ei_mean"],
                marker="o" if is_raw else ".",
                markersize=5.5 if is_raw else 4.0,
                linewidth=2.2 if is_raw else 1.35,
                linestyle="-" if is_raw else "--",
                color=color_map.get(series, "#4b5563"),
                label=series,
            )
        axis.set_xscale("log", base=2)
        axis.set_xlabel("L")
        axis.set_ylabel("EI (nats)")
        axis.set_title(label)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    png_path = output_dir / "tm_ei_l_baseline_raw_and_baseline_curves.png"
    pdf_path = output_dir / "tm_ei_l_baseline_raw_and_baseline_curves.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path, curve_table


def plot_raw_vs_corrected_by_l(
    corrected: pd.DataFrame,
    *,
    output_dir: Path,
    noise_mode: str = "fixed",
    dynamics_labels: tuple[str, ...] = ("identity", "product", "tanh"),
    baselines: tuple[str, ...] = ("identity", "gain_matched", "variance_matched", "analytic_volume"),
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered = corrected[
        corrected["noise_mode"].eq(noise_mode)
        & corrected["label"].isin(dynamics_labels)
        & corrected["baseline"].isin(baselines)
    ]
    labels = list(dict.fromkeys(filtered["label"].tolist()))
    fig, axes = plt.subplots(len(labels), 1, figsize=(8.2, max(3.0, 2.6 * len(labels))), constrained_layout=True)
    if len(labels) == 1:
        axes = [axes]
    for axis, label in zip(axes, labels):
        group = filtered[filtered["label"].eq(label)]
        raw = group.groupby("L", as_index=False)["ei_tm"].mean()
        axis.plot(raw["L"], raw["ei_tm"], marker="o", linewidth=1.8, color="#202124", label="raw EI")
        for baseline in baselines:
            baseline_group = group[group["baseline"].eq(baseline)]
            if baseline_group.empty:
                continue
            summarized = baseline_group.groupby("L", as_index=False)["ei_corrected"].mean()
            axis.plot(summarized["L"], summarized["ei_corrected"], marker="o", linewidth=1.5, label=baseline)
        axis.set_xscale("log", base=2)
        axis.set_xlabel("L")
        axis.set_ylabel("EI (nats)")
        axis.set_title(label)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    png_path = output_dir / "tm_ei_l_baseline_raw_vs_corrected.png"
    pdf_path = output_dir / "tm_ei_l_baseline_raw_vs_corrected.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_baseline_slope_reduction(
    summary: pd.DataFrame,
    *,
    output_dir: Path,
    noise_mode: str = "fixed",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered = summary[summary["noise_mode"].eq(noise_mode)].copy()
    labels = list(dict.fromkeys(filtered["label"].tolist()))
    baselines = list(dict.fromkeys(filtered["baseline"].tolist()))
    fig, axes = plt.subplots(len(labels), 1, figsize=(8.5, max(3.0, 2.4 * len(labels))), constrained_layout=True)
    if len(labels) == 1:
        axes = [axes]
    color_map = {
        "null": "#7f8c8d",
        "identity": "#1f77b4",
        "gain_matched": "#2ca02c",
        "variance_matched": "#d62728",
        "shuffled_target": "#9467bd",
        "analytic_volume": "#8c564b",
    }
    for axis, label in zip(axes, labels):
        group = filtered[filtered["label"].eq(label)].set_index("baseline")
        values = [group.loc[baseline, "corrected_slope_log_l"] if baseline in group.index else np.nan for baseline in baselines]
        colors = [color_map.get(baseline, "#4b5563") for baseline in baselines]
        axis.bar(baselines, values, color=colors, alpha=0.85)
        axis.axhline(0.0, color="#202124", linewidth=1.0)
        axis.set_ylabel("slope vs log L")
        axis.set_title(label)
        axis.grid(True, axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=25)
    png_path = output_dir / "tm_ei_l_baseline_slope_reduction.png"
    pdf_path = output_dir / "tm_ei_l_baseline_slope_reduction.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_best_corrected_ei_by_l(
    corrected: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    output_dir: Path,
    noise_mode: str = "fixed",
    max_abs_slope: float | None = 0.08,
    max_corrected_range: float | None = 0.2,
    max_curves: int = 5,
) -> tuple[Path, Path, pd.DataFrame]:
    """Plot corrected EI-vs-L curves for the flattest successful baseline per dynamics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    best = select_best_corrected_baselines(
        summary,
        noise_mode=noise_mode,
        max_abs_slope=max_abs_slope,
        max_corrected_range=max_corrected_range,
    )
    if best.empty:
        best = select_best_corrected_baselines(summary, noise_mode=noise_mode)
    best = best.sort_values("abs_corrected_slope").head(int(max_curves))

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]
    plotted_rows: list[dict[str, Any]] = []
    for color_index, (label, row) in enumerate(best.iterrows()):
        baseline = str(row["baseline"])
        group = corrected[
            corrected["noise_mode"].eq(noise_mode)
            & corrected["label"].eq(label)
            & corrected["baseline"].eq(baseline)
        ]
        if group.empty:
            continue
        curve = (
            group.groupby("L", as_index=False)
            .agg(ei_corrected_mean=("ei_corrected", "mean"), ei_corrected_std=("ei_corrected", "std"))
            .sort_values("L")
        )
        label_text = f"{label} / {baseline} (slope={row['corrected_slope_log_l']:.3f})"
        ax.plot(
            curve["L"],
            curve["ei_corrected_mean"],
            marker="o",
            linewidth=1.8,
            color=palette[color_index % len(palette)],
            label=label_text,
        )
        if curve["ei_corrected_std"].notna().any():
            y = curve["ei_corrected_mean"].to_numpy(dtype=float)
            spread = curve["ei_corrected_std"].fillna(0.0).to_numpy(dtype=float)
            ax.fill_between(
                curve["L"].to_numpy(dtype=float),
                y - spread,
                y + spread,
                color=palette[color_index % len(palette)],
                alpha=0.12,
                linewidth=0,
            )
        plotted_rows.append({"label": label, **row.to_dict()})

    ax.axhline(0.0, color="#202124", linewidth=1.0, alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("L")
    ax.set_ylabel("corrected EI (nats)")
    ax.set_title("Flattest corrected EI curves")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    png_path = output_dir / "tm_ei_l_baseline_best_corrected_curves.png"
    pdf_path = output_dir / "tm_ei_l_baseline_best_corrected_curves.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path, pd.DataFrame(plotted_rows)
