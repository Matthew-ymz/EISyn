#!/usr/bin/env python3
"""Recover task-specific Schaefer-500 spatial maps from paired HCP task series."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_ROOT = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_30"
DEFAULT_REST_ROOT = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30"
DEFAULT_LABEL_FILE = (
    ROOT
    / "data"
    / "hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30"
    / "_atlas_labels"
    / "Schaefer2018_500Parcels_7Networks_order.txt"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "hcp_schaefer500_task_specific_regions"

TASKS = ("EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")
TASK_LABELS = ("Emotion", "Gambling", "Language", "Motor", "Relational", "Social", "WM")
STATES = ("REST", *TASKS)
STATE_LABELS = ("REST", *TASK_LABELS)
NETWORK_ORDER = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
NETWORK_LABELS = ("Vis", "SomMot", "DorsAttn", "Sal/VentAttn", "Limbic", "Control", "Default")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def discover_inputs(task_root: Path) -> tuple[list[str], dict[str, dict[str, Path]]]:
    by_task = {
        task: {path.parent.name: path for path in Path(task_root).glob(f"sub-*/{task}_LR.mat")}
        for task in TASKS
    }
    common = set.intersection(*(set(paths) for paths in by_task.values()))
    subjects = sorted(common)
    if not subjects:
        raise FileNotFoundError(f"No subjects with all seven LR tasks under {task_root}.")
    return subjects, {
        subject: {task: by_task[task][subject] for task in TASKS} for subject in subjects
    }


def load_task_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = loadmat(path, variable_names=["Schaefer500_taskRetained", "Schaefer500_taskRegressed"])
    try:
        retained = np.asarray(payload["Schaefer500_taskRetained"], dtype=float)
        regressed = np.asarray(payload["Schaefer500_taskRegressed"], dtype=float)
    except KeyError as exc:
        raise ValueError(f"Missing paired Schaefer-500 task arrays in {path}.") from exc
    if retained.shape != regressed.shape or retained.ndim != 2 or retained.shape[1] != 500:
        raise ValueError(f"Expected paired [time, 500] arrays in {path}, got {retained.shape} and {regressed.shape}.")
    if not np.isfinite(retained).all() or not np.isfinite(regressed).all():
        raise ValueError(f"Non-finite values found in {path}.")
    return retained, regressed


def discover_rest_inputs(rest_root: Path) -> dict[str, Path]:
    paths = {
        path.parent.name: path
        for path in Path(rest_root).glob("sub-*/*REST1_LR*schaefer500-1000_yeo7.mat")
    }
    if not paths:
        raise FileNotFoundError(f"No REST1_LR Schaefer-500 inputs under {rest_root}.")
    return paths


def load_rest_series(path: Path) -> np.ndarray:
    payload = loadmat(path, variable_names=["Schaefer500"])
    if "Schaefer500" not in payload:
        raise ValueError(f"Missing Schaefer500 REST array in {path}.")
    values = np.asarray(payload["Schaefer500"], dtype=float)
    if values.ndim != 2 or values.shape[1] != 500 or not np.isfinite(values).all():
        raise ValueError(f"Expected finite [time, 500] REST data in {path}, got {values.shape}.")
    return values


def task_evoked_variance_fraction(
    retained: np.ndarray,
    regressed: np.ndarray,
    *,
    eps: float = 1.0e-12,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return parcel-wise task-component energy divided by retained-series energy.

    The intercept is removed independently from both arrays.  With an OLS task
    regression, retained = residual + fitted task component and the two latter
    terms are orthogonal, so the ratio is an explained-variance fraction.
    """
    retained = np.asarray(retained, dtype=float)
    regressed = np.asarray(regressed, dtype=float)
    if retained.shape != regressed.shape or retained.ndim != 2:
        raise ValueError("retained and regressed must be equally shaped 2-D arrays.")
    centered_retained = retained - retained.mean(axis=0, keepdims=True)
    centered_regressed = regressed - regressed.mean(axis=0, keepdims=True)
    task_component = centered_retained - centered_regressed
    retained_energy = np.einsum("ti,ti->i", centered_retained, centered_retained)
    task_energy = np.einsum("ti,ti->i", task_component, task_component)
    residual_energy = np.einsum("ti,ti->i", centered_regressed, centered_regressed)
    cross = np.einsum("ti,ti->i", centered_regressed, task_component)
    raw_fraction = np.divide(
        task_energy,
        retained_energy,
        out=np.zeros_like(task_energy),
        where=retained_energy > eps,
    )
    denom = np.sqrt(np.maximum(residual_energy * task_energy, eps))
    orthogonality = np.divide(np.abs(cross), denom, out=np.zeros_like(cross), where=denom > eps)
    reconstruction_error = np.max(
        np.abs(centered_retained - centered_regressed - task_component)
    )
    diagnostics = {
        "fraction_min_raw": float(raw_fraction.min()),
        "fraction_max_raw": float(raw_fraction.max()),
        "median_abs_residual_task_correlation": float(np.median(orthogonality)),
        "max_abs_residual_task_correlation": float(np.max(orthogonality)),
        "max_reconstruction_error": float(reconstruction_error),
    }
    return np.clip(raw_fraction, 0.0, 1.0), diagnostics


def parse_schaefer_labels(path: Path) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        index = int(fields[0]) - 1
        name = fields[1]
        parts = name.split("_")
        if len(parts) < 4 or parts[1] not in {"LH", "RH"} or parts[2] not in NETWORK_ORDER:
            raise ValueError(f"Unexpected Schaefer label {name!r} in {path}.")
        labels.append(
            {
                "index": index,
                "name": name,
                "short_name": "_".join(parts[1:]),
                "hemisphere": parts[1],
                "network": parts[2],
            }
        )
    if len(labels) != 500 or [item["index"] for item in labels] != list(range(500)):
        raise ValueError(f"Expected ordered 500-parcel labels in {path}, found {len(labels)}.")
    return labels


def spatial_enrichment(fractions: np.ndarray, *, eps: float = 1.0e-12) -> np.ndarray:
    """Normalize every parcel map to mean one, retaining spatial shape only."""
    fractions = np.asarray(fractions, dtype=float)
    mean = fractions.mean(axis=-1, keepdims=True)
    if np.any(mean <= eps):
        raise ValueError("Cannot normalize a parcel map with zero mean task fraction.")
    return fractions / mean


def temporal_variance_enrichment(series: np.ndarray, *, eps: float = 1.0e-12) -> np.ndarray:
    """Return parcel temporal variance divided by the run's mean parcel variance."""
    series = np.asarray(series, dtype=float)
    if series.ndim != 2 or series.shape[1] != 500 or not np.isfinite(series).all():
        raise ValueError(f"Expected finite [time, 500] series, got {series.shape}.")
    variance = np.var(series, axis=0, ddof=1)
    mean_variance = float(np.mean(variance))
    if mean_variance <= eps:
        raise ValueError("Cannot normalize a run with zero mean parcel variance.")
    return variance / mean_variance


def task_specific_contrast(enrichment: np.ndarray) -> np.ndarray:
    """Subtract the mean of all other tasks for each subject and parcel."""
    enrichment = np.asarray(enrichment, dtype=float)
    if enrichment.ndim != 3 or enrichment.shape[1] < 2:
        raise ValueError("Expected [subject, task, parcel] enrichment maps.")
    n_tasks = enrichment.shape[1]
    other_mean = (enrichment.sum(axis=1, keepdims=True) - enrichment) / (n_tasks - 1)
    return enrichment - other_mean


def unit_spatial_features(maps: np.ndarray, *, eps: float = 1.0e-12) -> np.ndarray:
    maps = np.asarray(maps, dtype=float)
    centered = maps - maps.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(centered, axis=-1, keepdims=True)
    if np.any(norm <= eps):
        raise ValueError("At least one spatial map is constant.")
    return centered / norm


def loso_nearest_centroid(features: np.ndarray) -> dict[str, Any]:
    """Leave one subject out, classifying each task by cosine similarity."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 3:
        raise ValueError("Expected [subject, task, feature] input.")
    n_subjects, n_tasks, _ = features.shape
    if n_subjects < 2:
        raise ValueError("LOSO classification requires at least two subjects.")
    summed = features.sum(axis=0)
    scores = np.empty((n_subjects, n_tasks, n_tasks), dtype=float)
    predictions = np.empty((n_subjects, n_tasks), dtype=int)
    confusion = np.zeros((n_tasks, n_tasks), dtype=int)
    for subject in range(n_subjects):
        centroid = (summed - features[subject]) / (n_subjects - 1)
        centroid = unit_spatial_features(centroid)
        subject_scores = features[subject] @ centroid.T
        subject_predictions = np.argmax(subject_scores, axis=1)
        scores[subject] = subject_scores
        predictions[subject] = subject_predictions
        for truth, prediction in enumerate(subject_predictions):
            confusion[truth, prediction] += 1
    accuracy = float(np.mean(predictions == np.arange(n_tasks)[None, :]))
    subject_accuracy = np.mean(predictions == np.arange(n_tasks)[None, :], axis=1)
    return {
        "accuracy": accuracy,
        "subject_accuracy": subject_accuracy,
        "confusion": confusion,
        "scores": scores,
        "predictions": predictions,
    }


def permutation_accuracy_pvalue(
    features: np.ndarray,
    observed_accuracy: float,
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=float)
    for repeat in tqdm(range(permutations), desc="Within-subject label permutations", unit="perm"):
        permuted = np.empty_like(features)
        for subject in range(features.shape[0]):
            permuted[subject] = features[subject, rng.permutation(features.shape[1])]
        null[repeat] = loso_nearest_centroid(permuted)["accuracy"]
    pvalue = float((1 + np.count_nonzero(null >= observed_accuracy)) / (permutations + 1))
    return {
        "pvalue": pvalue,
        "null_mean": float(null.mean()),
        "null_ci95": [float(value) for value in np.quantile(null, [0.025, 0.975])],
        "null": null,
    }


def aggregate_networks(maps: np.ndarray, network_indices: Sequence[np.ndarray]) -> np.ndarray:
    maps = np.asarray(maps, dtype=float)
    return np.stack([maps[..., indices].mean(axis=-1) for indices in network_indices], axis=-1)


def atlas_blocks(labels: Sequence[dict[str, Any]]) -> list[tuple[int, int, str]]:
    blocks: list[tuple[int, int, str]] = []
    start = 0
    current = (labels[0]["hemisphere"], labels[0]["network"])
    for index, label in enumerate(labels[1:], start=1):
        key = (label["hemisphere"], label["network"])
        if key != current:
            blocks.append((start, index, f"{current[0]} {current[1]}"))
            start = index
            current = key
    blocks.append((start, len(labels), f"{current[0]} {current[1]}"))
    return blocks


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def annotate_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    fmt: str,
    threshold: float | None = None,
    dark_region: str = "high",
) -> None:
    if threshold is None:
        threshold = float((np.nanmin(matrix) + np.nanmax(matrix)) / 2)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if dark_region == "high":
                white = value > threshold
            elif dark_region == "low":
                white = value < threshold
            elif dark_region == "ends":
                white = abs(value) > threshold
            else:
                raise ValueError(f"Unknown dark_region {dark_region!r}.")
            ax.text(
                col,
                row,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if white else "black",
            )


def format_matrix_axes(ax: plt.Axes, xlabels: Sequence[str], ylabels: Sequence[str]) -> None:
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=35, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.tick_params(length=0)


def plot_region_profiles(
    group_fraction: np.ndarray,
    group_specificity: np.ndarray,
    network_fraction: np.ndarray,
    network_specificity: np.ndarray,
    blocks: Sequence[tuple[int, int, str]],
    output_dir: Path,
) -> None:
    configure_style()
    figure = plt.figure(figsize=(12.6, 7.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=[2.55, 1.0], height_ratios=[1, 1])
    axes = [figure.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]

    ax = axes[0]
    vmax = float(np.quantile(group_fraction, 0.995))
    image = ax.imshow(group_fraction, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(TASK_LABELS)), TASK_LABELS)
    centers = [(start + end - 1) / 2 for start, end, _ in blocks]
    ax.set_xticks(centers, [name for _, _, name in blocks], rotation=45, ha="right")
    for _, end, _ in blocks[:-1]:
        ax.axvline(end - 0.5, color="white", linewidth=0.45, alpha=0.75)
    ax.set_ylabel("Task")
    ax.set_xlabel("Schaefer-500 parcel (atlas order)")
    colorbar = figure.colorbar(image, ax=ax, location="right", shrink=0.82, pad=0.015)
    colorbar.set_label("Task-evoked variance fraction")
    add_panel_label(ax, "a")

    ax = axes[1]
    image = ax.imshow(network_fraction, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
    format_matrix_axes(ax, NETWORK_LABELS, TASK_LABELS)
    annotate_matrix(
        ax,
        100.0 * network_fraction,
        fmt=".1f",
        threshold=100.0 * vmax * 0.72,
        dark_region="low",
    )
    ax.set_xlabel("Yeo7 network")
    ax.set_ylabel("Task")
    colorbar = figure.colorbar(image, ax=ax, location="right", shrink=0.82, pad=0.025)
    colorbar.set_label("Task-evoked fraction")
    ax.text(1.01, -0.17, "cells: %", transform=ax.transAxes, ha="right", va="top", fontsize=7)
    add_panel_label(ax, "b")

    ax = axes[2]
    limit = float(np.quantile(np.abs(group_specificity), 0.995))
    image = ax.imshow(
        group_specificity,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(TASK_LABELS)), TASK_LABELS)
    ax.set_xticks(centers, [name for _, _, name in blocks], rotation=45, ha="right")
    for _, end, _ in blocks[:-1]:
        ax.axvline(end - 0.5, color="black", linewidth=0.4, alpha=0.45)
    ax.set_ylabel("Task")
    ax.set_xlabel("Schaefer-500 parcel (atlas order)")
    colorbar = figure.colorbar(image, ax=ax, location="right", shrink=0.82, pad=0.015)
    colorbar.set_label("Task-specific spatial enrichment")
    add_panel_label(ax, "c")

    ax = axes[3]
    network_limit = float(np.max(np.abs(network_specificity)))
    image = ax.imshow(
        network_specificity,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-network_limit,
        vmax=network_limit,
    )
    format_matrix_axes(ax, NETWORK_LABELS, TASK_LABELS)
    annotate_matrix(
        ax,
        network_specificity,
        fmt="+.2f",
        threshold=0.58 * network_limit,
        dark_region="ends",
    )
    ax.set_xlabel("Yeo7 network")
    ax.set_ylabel("Task")
    colorbar = figure.colorbar(image, ax=ax, location="right", shrink=0.82, pad=0.025)
    colorbar.set_label("Task-specific enrichment")
    add_panel_label(ax, "d")

    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output_dir / f"task_evoked_region_profiles.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(figure)


def plot_rest_task_state_profiles(
    group_enrichment: np.ndarray,
    group_specificity: np.ndarray,
    network_enrichment: np.ndarray,
    network_specificity: np.ndarray,
    blocks: Sequence[tuple[int, int, str]],
    output_dir: Path,
) -> None:
    """Plot REST and all tasks using the common temporal-variance metric."""
    configure_style()
    figure = plt.figure(figsize=(12.6, 7.35), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=[2.55, 1.0], height_ratios=[1, 1])
    axes = [figure.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    centers = [(start + end - 1) / 2 for start, end, _ in blocks]

    ax = axes[0]
    vmax = float(np.quantile(group_enrichment, 0.995))
    image = ax.imshow(
        group_enrichment,
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(STATE_LABELS)), STATE_LABELS)
    ax.set_xticks(centers, [name for _, _, name in blocks], rotation=45, ha="right")
    for _, end, _ in blocks[:-1]:
        ax.axvline(end - 0.5, color="white", linewidth=0.45, alpha=0.75)
    ax.axhline(0.5, color="white", linewidth=1.0)
    ax.set_ylabel("State")
    ax.set_xlabel("Schaefer-500 parcel (atlas order)")
    colorbar = figure.colorbar(image, ax=ax, location="right", shrink=0.82, pad=0.015)
    colorbar.set_label("Temporal-variance enrichment")
    add_panel_label(ax, "a")

    ax = axes[1]
    image = ax.imshow(network_enrichment, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
    format_matrix_axes(ax, NETWORK_LABELS, STATE_LABELS)
    annotate_matrix(
        ax,
        network_enrichment,
        fmt=".2f",
        threshold=0.72 * vmax,
        dark_region="low",
    )
    ax.axhline(0.5, color="white", linewidth=1.0)
    ax.set_xlabel("Yeo7 network")
    ax.set_ylabel("State")
    colorbar = figure.colorbar(image, ax=ax, location="right", shrink=0.82, pad=0.025)
    colorbar.set_label("Temporal-variance enrichment")
    ax.text(
        1.01,
        -0.17,
        "parcel mean = 1",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )
    add_panel_label(ax, "b")

    ax = axes[2]
    limit = float(np.quantile(np.abs(group_specificity), 0.995))
    image = ax.imshow(
        group_specificity,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(STATE_LABELS)), STATE_LABELS)
    ax.set_xticks(centers, [name for _, _, name in blocks], rotation=45, ha="right")
    for _, end, _ in blocks[:-1]:
        ax.axvline(end - 0.5, color="black", linewidth=0.4, alpha=0.45)
    ax.axhline(0.5, color="black", linewidth=1.0)
    ax.set_ylabel("State")
    ax.set_xlabel("Schaefer-500 parcel (atlas order)")
    colorbar = figure.colorbar(image, ax=ax, location="right", shrink=0.82, pad=0.015)
    colorbar.set_label("State-specific variance enrichment")
    add_panel_label(ax, "c")

    ax = axes[3]
    network_limit = float(np.max(np.abs(network_specificity)))
    image = ax.imshow(
        network_specificity,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-network_limit,
        vmax=network_limit,
    )
    format_matrix_axes(ax, NETWORK_LABELS, STATE_LABELS)
    annotate_matrix(
        ax,
        network_specificity,
        fmt="+.2f",
        threshold=0.58 * network_limit,
        dark_region="ends",
    )
    ax.axhline(0.5, color="black", linewidth=1.0)
    ax.set_xlabel("Yeo7 network")
    ax.set_ylabel("State")
    colorbar = figure.colorbar(image, ax=ax, location="right", shrink=0.82, pad=0.025)
    colorbar.set_label("State-specific variance enrichment")
    add_panel_label(ax, "d")

    for suffix in ("png", "svg", "pdf"):
        figure.savefig(
            output_dir / f"rest_all_tasks_variance_profiles.{suffix}",
            dpi=400,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_discriminability(
    centroid_correlation: np.ndarray,
    parcel_result: dict[str, Any],
    network_result: dict[str, Any],
    parcel_permutation: dict[str, Any],
    network_permutation: dict[str, Any],
    output_dir: Path,
) -> None:
    configure_style()
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.45), constrained_layout=True)

    ax = axes[0]
    lower = min(0.0, float(np.min(centroid_correlation)))
    image = ax.imshow(centroid_correlation, cmap="viridis", vmin=lower, vmax=1.0)
    format_matrix_axes(ax, TASK_LABELS, TASK_LABELS)
    annotate_matrix(
        ax,
        centroid_correlation,
        fmt=".2f",
        threshold=(1.0 + lower) / 2,
        dark_region="low",
    )
    ax.set_xlabel("Task centroid")
    ax.set_ylabel("Task centroid")
    colorbar = figure.colorbar(image, ax=ax, location="right", shrink=0.78, pad=0.025)
    colorbar.set_label("Spatial correlation")
    add_panel_label(ax, "a")

    for panel, (ax, result, permutation, title) in enumerate(
        zip(
            axes[1:],
            (
                parcel_result,
                network_result,
            ),
            (
                parcel_permutation,
                network_permutation,
            ),
            ("Schaefer-500 parcels", "Yeo7 network means"),
        ),
        start=1,
    ):
        confusion = result["confusion"] / result["confusion"].sum(axis=1, keepdims=True)
        image = ax.imshow(confusion, cmap="Blues", vmin=0.0, vmax=1.0)
        format_matrix_axes(ax, TASK_LABELS, TASK_LABELS)
        annotate_matrix(ax, 100.0 * confusion, fmt=".0f", threshold=55.0)
        ax.set_xlabel("Predicted task")
        ax.set_ylabel("True task")
        ax.set_title(title, pad=5)
        ax.text(
            0.5,
            -0.31,
            f"LOSO accuracy = {100 * result['accuracy']:.1f}%\n"
            f"within-subject permutation $p$ = {permutation['pvalue']:.4f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
        )
        colorbar = figure.colorbar(image, ax=ax, location="right", shrink=0.78, pad=0.025)
        colorbar.set_label("Row percentage")
        add_panel_label(ax, chr(ord("a") + panel))

    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output_dir / f"task_map_discriminability.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(figure)


def summarize_top_parcels(
    group_specificity: np.ndarray,
    group_fraction: np.ndarray,
    labels: Sequence[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for task_index, task in enumerate(TASKS):
        ranking = np.argsort(group_specificity[task_index])[::-1][:top_k]
        result[task] = [
            {
                **labels[int(parcel)],
                "task_evoked_fraction": float(group_fraction[task_index, parcel]),
                "task_specific_enrichment": float(group_specificity[task_index, parcel]),
            }
            for parcel in ranking
        ]
    return result


def write_report(summary: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# HCP 七任务 Schaefer-500 任务特异脑区分布",
        "",
        "## 方法",
        "",
        "对每个被试、任务和 parcel，分别去除 `taskRetained` 与 `taskRegressed` 的时间均值。",
        "令保留任务时序为 $r(t)$，回归任务后的残差为 $e(t)$，任务模型成分为",
        "$u(t)=r(t)-e(t)$。定义任务诱发方差比例",
        "",
        "$$\\mathrm{TEVF}=\\frac{\\sum_t u(t)^2}{\\sum_t r(t)^2}.$$",
        "在 OLS 任务回归下，$e$ 与 $u$ 正交，因此该量是 parcel 级任务成分解释比例。",
        "为分离整体效应强度与空间形状，将每张 500-parcel 图除以其 parcel 均值，得到均值为 1 的空间富集图 $q$；任务 $c$ 的特异性为",
        "",
        "$$d_{c,i}=q_{c,i}-\\frac{1}{6}\\sum_{c'\\ne c}q_{c',i}.$$",
        "因此每个任务的 $d$ 在 500 个 parcel 上严格和为 0，不会把任务整体更强误当成空间特异。",
        "",
        "个体级验证采用 leave-one-subject-out 最近质心分类。每张 $q$ 图先在 parcel 维去均值并作 $L_2$ 归一化；训练质心只由其余被试形成，以余弦相似度预测留出被试的七张任务图。显著性通过在每个被试内部独立置换七个任务标签并完整重跑 LOSO 得到。Yeo7 对照把同一 TEVF 图先聚合为七个网络均值，再执行完全相同的流程。",
        "",
        "## 结果",
        "",
        f"纳入 {summary['n_subjects']} 名具有全部七个 LR 任务的被试。",
        f"Schaefer-500 空间图的 LOSO 准确率为 {100 * summary['classification']['parcel']['accuracy']:.1f}%（chance = 14.3%，置换 $p={summary['classification']['parcel']['permutation_pvalue']:.4f}$）；",
        f"Yeo7 网络均值图准确率为 {100 * summary['classification']['network']['accuracy']:.1f}%（置换 $p={summary['classification']['network']['permutation_pvalue']:.4f}$）。",
        "",
        "![任务诱发 parcel 分布](task_evoked_region_profiles.png)",
        "",
        "![任务空间图可辨识性](task_map_discriminability.png)",
        "",
        "## REST 与七任务的共同口径比较",
        "",
        "REST 没有 task GLM，因此不能定义 TEVF。为避免把 REST 人为设为零，八状态比较改用每个状态都可计算的 parcel 时间方差，并将每张图除以其 500-parcel 平均方差。该量只比较空间形状，不比较不同 run 的绝对 BOLD 方差。状态特异性定义为本状态的方差富集减去其余七状态的平均方差富集。",
        "",
        f"REST 与全部七任务共同的被试数为 {summary['rest_comparison']['n_subjects']}。500-parcel 方差富集图的八状态 LOSO 准确率为 {100 * summary['rest_comparison']['classification']['parcel']['accuracy']:.1f}%（chance = 12.5%，置换 $p={summary['rest_comparison']['classification']['parcel']['permutation_pvalue']:.4f}$）；Yeo7 网络均值准确率为 {100 * summary['rest_comparison']['classification']['network']['accuracy']:.1f}%（$p={summary['rest_comparison']['classification']['network']['permutation_pvalue']:.4f}$）。",
        "",
        "![REST 与七任务的共同方差空间分布](rest_all_tasks_variance_profiles.png)",
        "",
        "## 每个任务最特异的 parcel",
        "",
        "| Task | Top parcels (task-specific enrichment) |",
        "|---|---|",
    ]
    for task in TASKS:
        text = "; ".join(
            f"{item['short_name']} ({item['task_specific_enrichment']:+.3f})"
            for item in summary["top_parcels"][task][:5]
        )
        lines.append(f"| {task} | {text} |")
    lines.extend(
        [
            "",
            "## 与 EI/Phi 的关系",
            "",
            "TEVF 回答的是“任务设计解释了哪个 parcel 的多少时间变异”，不是系统动力学的有效信息或整合信息。现有 Yeo7-Phi 分析把 500 个 parcel 压缩为七个 PC1、逐状态标准化并在整段时序上拟合平稳一步动力学，因此对任务平均激活、parcel 内部异质性和事件结构不敏感。两类指标应并列使用：Phi 描述预测动力学整合，TEVF 与 $d$ 描述任务相关脑区分布；不能把 TEVF 称为 Phi 的节点归因。",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--rest-root", type=Path, default=DEFAULT_REST_ROOT)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects, paths = discover_inputs(args.task_root)
    labels = parse_schaefer_labels(args.label_file)
    fractions = np.empty((len(subjects), len(TASKS), 500), dtype=float)
    diagnostics: list[dict[str, Any]] = []
    for subject_index, subject in enumerate(tqdm(subjects, desc="Subjects", unit="subject")):
        for task_index, task in enumerate(TASKS):
            retained, regressed = load_task_pair(paths[subject][task])
            fraction, diagnostic = task_evoked_variance_fraction(retained, regressed)
            fractions[subject_index, task_index] = fraction
            diagnostics.append({"subject": subject, "task": task, **diagnostic})

    enrichment = spatial_enrichment(fractions)
    specificity = task_specific_contrast(enrichment)
    group_fraction = fractions.mean(axis=0)
    group_specificity = specificity.mean(axis=0)

    network_indices = [
        np.asarray([item["index"] for item in labels if item["network"] == network], dtype=int)
        for network in NETWORK_ORDER
    ]
    network_fraction_subject = aggregate_networks(fractions, network_indices)
    network_enrichment_subject = aggregate_networks(enrichment, network_indices)
    network_specificity_subject = aggregate_networks(specificity, network_indices)

    parcel_features = unit_spatial_features(enrichment)
    network_features = unit_spatial_features(network_enrichment_subject)
    parcel_result = loso_nearest_centroid(parcel_features)
    network_result = loso_nearest_centroid(network_features)
    parcel_permutation = permutation_accuracy_pvalue(
        parcel_features,
        parcel_result["accuracy"],
        permutations=args.permutations,
        seed=args.seed,
    )
    network_permutation = permutation_accuracy_pvalue(
        network_features,
        network_result["accuracy"],
        permutations=args.permutations,
        seed=args.seed + 1,
    )

    group_features = unit_spatial_features(group_fraction)
    centroid_correlation = group_features @ group_features.T
    top_parcels = summarize_top_parcels(
        group_specificity, group_fraction, labels, top_k=args.top_k
    )

    plot_region_profiles(
        group_fraction,
        group_specificity,
        network_fraction_subject.mean(axis=0),
        network_specificity_subject.mean(axis=0),
        atlas_blocks(labels),
        args.output_dir,
    )
    plot_discriminability(
        centroid_correlation,
        parcel_result,
        network_result,
        parcel_permutation,
        network_permutation,
        args.output_dir,
    )

    rest_paths = discover_rest_inputs(args.rest_root)
    state_subjects = [subject for subject in subjects if subject in rest_paths]
    if not state_subjects:
        raise ValueError("No subjects are shared by REST and all seven task states.")
    state_enrichment = np.empty((len(state_subjects), len(STATES), 500), dtype=float)
    state_timepoints: dict[str, int] = {}
    for subject_index, subject in enumerate(
        tqdm(state_subjects, desc="REST + task variance maps", unit="subject")
    ):
        rest_series = load_rest_series(rest_paths[subject])
        state_enrichment[subject_index, 0] = temporal_variance_enrichment(rest_series)
        state_timepoints.setdefault("REST", int(len(rest_series)))
        for task_index, task in enumerate(TASKS, start=1):
            retained, _ = load_task_pair(paths[subject][task])
            state_enrichment[subject_index, task_index] = temporal_variance_enrichment(retained)
            state_timepoints.setdefault(task, int(len(retained)))
    state_specificity = task_specific_contrast(state_enrichment)
    network_state_enrichment = aggregate_networks(state_enrichment, network_indices)
    network_state_specificity = aggregate_networks(state_specificity, network_indices)
    state_parcel_features = unit_spatial_features(state_enrichment)
    state_network_features = unit_spatial_features(network_state_enrichment)
    state_parcel_result = loso_nearest_centroid(state_parcel_features)
    state_network_result = loso_nearest_centroid(state_network_features)
    state_parcel_permutation = permutation_accuracy_pvalue(
        state_parcel_features,
        state_parcel_result["accuracy"],
        permutations=args.permutations,
        seed=args.seed + 2,
    )
    state_network_permutation = permutation_accuracy_pvalue(
        state_network_features,
        state_network_result["accuracy"],
        permutations=args.permutations,
        seed=args.seed + 3,
    )
    plot_rest_task_state_profiles(
        state_enrichment.mean(axis=0),
        state_specificity.mean(axis=0),
        network_state_enrichment.mean(axis=0),
        network_state_specificity.mean(axis=0),
        atlas_blocks(labels),
        args.output_dir,
    )

    summary = {
        "analysis": "Schaefer-500 task-evoked variance fraction and task-specific spatial enrichment",
        "n_subjects": len(subjects),
        "subjects": subjects,
        "tasks": list(TASKS),
        "task_timepoints": {
            task: int(load_task_pair(paths[subjects[0]][task])[0].shape[0]) for task in TASKS
        },
        "metric": {
            "task_evoked_fraction": "sum_t ((retained-centered) - (regressed-centered))^2 / sum_t (retained-centered)^2",
            "spatial_enrichment": "task_evoked_fraction / parcel_mean(task_evoked_fraction)",
            "task_specificity": "spatial_enrichment(task) - mean spatial_enrichment(other six tasks)",
        },
        "mean_task_evoked_fraction": {
            task: float(fractions[:, index].mean()) for index, task in enumerate(TASKS)
        },
        "median_subject_task_evoked_fraction": {
            task: float(np.median(fractions[:, index].mean(axis=1))) for index, task in enumerate(TASKS)
        },
        "network_mean_task_evoked_fraction": {
            task: {
                network: float(network_fraction_subject[:, task_index, network_index].mean())
                for network_index, network in enumerate(NETWORK_ORDER)
            }
            for task_index, task in enumerate(TASKS)
        },
        "network_task_specific_enrichment": {
            task: {
                network: float(network_specificity_subject[:, task_index, network_index].mean())
                for network_index, network in enumerate(NETWORK_ORDER)
            }
            for task_index, task in enumerate(TASKS)
        },
        "classification": {
            "parcel": {
                "accuracy": parcel_result["accuracy"],
                "subject_accuracy": parcel_result["subject_accuracy"].tolist(),
                "confusion": parcel_result["confusion"].tolist(),
                "permutation_pvalue": parcel_permutation["pvalue"],
                "permutation_null_mean": parcel_permutation["null_mean"],
                "permutation_null_ci95": parcel_permutation["null_ci95"],
            },
            "network": {
                "accuracy": network_result["accuracy"],
                "subject_accuracy": network_result["subject_accuracy"].tolist(),
                "confusion": network_result["confusion"].tolist(),
                "permutation_pvalue": network_permutation["pvalue"],
                "permutation_null_mean": network_permutation["null_mean"],
                "permutation_null_ci95": network_permutation["null_ci95"],
            },
        },
        "rest_comparison": {
            "n_subjects": len(state_subjects),
            "subjects": state_subjects,
            "states": list(STATES),
            "timepoints": state_timepoints,
            "metric": {
                "temporal_variance_enrichment": "parcel temporal variance / mean temporal variance across 500 parcels",
                "state_specificity": "variance enrichment(state) - mean variance enrichment(other seven states)",
            },
            "network_temporal_variance_enrichment": {
                state: {
                    network: float(
                        network_state_enrichment[:, state_index, network_index].mean()
                    )
                    for network_index, network in enumerate(NETWORK_ORDER)
                }
                for state_index, state in enumerate(STATES)
            },
            "network_state_specific_enrichment": {
                state: {
                    network: float(
                        network_state_specificity[:, state_index, network_index].mean()
                    )
                    for network_index, network in enumerate(NETWORK_ORDER)
                }
                for state_index, state in enumerate(STATES)
            },
            "classification": {
                "parcel": {
                    "accuracy": state_parcel_result["accuracy"],
                    "subject_accuracy": state_parcel_result["subject_accuracy"].tolist(),
                    "confusion": state_parcel_result["confusion"].tolist(),
                    "permutation_pvalue": state_parcel_permutation["pvalue"],
                    "permutation_null_mean": state_parcel_permutation["null_mean"],
                    "permutation_null_ci95": state_parcel_permutation["null_ci95"],
                },
                "network": {
                    "accuracy": state_network_result["accuracy"],
                    "subject_accuracy": state_network_result["subject_accuracy"].tolist(),
                    "confusion": state_network_result["confusion"].tolist(),
                    "permutation_pvalue": state_network_permutation["pvalue"],
                    "permutation_null_mean": state_network_permutation["null_mean"],
                    "permutation_null_ci95": state_network_permutation["null_ci95"],
                },
            },
            "diagnostics": {
                "max_abs_parcel_mean_error": float(
                    np.max(np.abs(state_enrichment.mean(axis=-1) - 1.0))
                ),
                "max_abs_specificity_parcel_sum": float(
                    np.max(np.abs(state_specificity.sum(axis=-1)))
                ),
                "max_abs_specificity_state_sum": float(
                    np.max(np.abs(state_specificity.sum(axis=1)))
                ),
            },
        },
        "group_centroid_spatial_correlation": centroid_correlation.tolist(),
        "diagnostics": {
            "max_raw_fraction": float(max(item["fraction_max_raw"] for item in diagnostics)),
            "min_raw_fraction": float(min(item["fraction_min_raw"] for item in diagnostics)),
            "median_abs_residual_task_correlation": float(
                np.median([item["median_abs_residual_task_correlation"] for item in diagnostics])
            ),
            "max_abs_residual_task_correlation": float(
                max(item["max_abs_residual_task_correlation"] for item in diagnostics)
            ),
            "max_reconstruction_error": float(
                max(item["max_reconstruction_error"] for item in diagnostics)
            ),
            "max_abs_group_specificity_sum": float(np.max(np.abs(group_specificity.sum(axis=1)))),
        },
        "top_parcels": top_parcels,
    }
    np.savez_compressed(
        args.output_dir / "task_evoked_region_maps.npz",
        subjects=np.asarray(subjects),
        tasks=np.asarray(TASKS),
        fractions=fractions,
        enrichment=enrichment,
        specificity=specificity,
        parcel_permutation_null=parcel_permutation["null"],
        network_permutation_null=network_permutation["null"],
        state_subjects=np.asarray(state_subjects),
        states=np.asarray(STATES),
        state_variance_enrichment=state_enrichment,
        state_specificity=state_specificity,
        state_parcel_permutation_null=state_parcel_permutation["null"],
        state_network_permutation_null=state_network_permutation["null"],
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary, args.output_dir)

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "task_classification": summary["classification"],
                "rest_task_classification": summary["rest_comparison"]["classification"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
