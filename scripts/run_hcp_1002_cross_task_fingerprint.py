#!/usr/bin/env python3
"""Exploratory cross-task brain fingerprinting on the cached HCP 1002 cohort."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/hcp_s1200_1002_complete_scores_task_lr_mmp360_yeo7_pc1_timeseries"
CACHE = ROOT / "results/hcp_mmp360_same_task_score_screen_1002/coalition_synergy_all7_1002.npz"
OUT = ROOT / "results/hcp_1002_cross_task_fingerprint"
UPPER = np.triu_indices(7, 1)


def load_cached_features(limit: int | None) -> tuple[list[str], list[str], dict[str, np.ndarray], dict]:
    with np.load(CACHE, allow_pickle=False) as archive:
        subjects = archive["subjects"].astype(str).tolist()
        tasks = archive["tasks"].astype(str).tolist()
        sizes = archive["coalition_sizes"]
        syn = archive["synergy_bits"].astype(np.float64)
        tol = float(archive["syn_tolerance_bits"])
    if limit is not None:
        subjects = subjects[:limit]
        syn = syn[:, :limit]
    if len(set(subjects)) != len(subjects) or syn.shape != (len(tasks), len(subjects), 120):
        raise ValueError("Cache has inconsistent subject, task, or coalition dimensions")
    if int(np.sum(sizes == 2)) != 21 or not np.isfinite(syn).all():
        raise ValueError("Expected 21 finite pair coalitions and a finite full cache")
    violations = syn < -tol
    if np.any(violations):
        raise ValueError(
            f"Syn nonnegativity violation: min={float(syn.min()):.12g} bits, "
            f"threshold={-tol:.12g} bits, count={int(violations.sum())}"
        )
    quality = {
        "syn_tolerance_bits": tol,
        "minimum_syn_bits": float(syn.min()),
        "negative_within_tolerance_count": int(np.sum((syn < 0) & ~violations)),
        "significant_negative_count": int(violations.sum()),
        "syn_value_count": int(syn.size),
    }
    return subjects, tasks, {
        "Syn 21": syn[:, :, sizes == 2].astype(np.float32),
        "Syn 120": syn.astype(np.float32),
    }, quality


def compute_fc(subjects: list[str], tasks: list[str]) -> dict[str, np.ndarray]:
    signed = np.empty((len(tasks), len(subjects), 21), dtype=np.float32)
    for task_index, task in enumerate(tasks):
        for subject_index, subject in enumerate(subjects):
            path = DATA / subject / f"{task}_LR.mat"
            series = np.asarray(
                loadmat(path, variable_names=["Yeo7_taskRetainedPC1"])["Yeo7_taskRetainedPC1"],
                dtype=np.float64,
            )
            if series.ndim != 2 or series.shape[1] != 7 or not np.isfinite(series).all():
                raise ValueError(f"Invalid seven-network time series: {path}")
            fc = np.corrcoef(series, rowvar=False)[UPPER]
            if not np.isfinite(fc).all():
                raise ValueError(f"Undefined FC: {path}")
            signed[task_index, subject_index] = fc
        print(f"FC computed: {task_index + 1}/{len(tasks)} {task}", flush=True)
    return {"|FC| 21": np.abs(signed), "FC 21": signed}


def unit_rows(features: np.ndarray) -> np.ndarray:
    centered = features.astype(np.float64) - features.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("A feature vector is constant across coordinates")
    return (centered / norms).astype(np.float32)


def fingerprint(features: np.ndarray, tasks: list[str], task_centered: bool = False) -> list[dict]:
    normalized = [
        unit_rows(task_features - task_features.mean(axis=0, keepdims=True)
                  if task_centered else task_features)
        for task_features in features
    ]
    identities = np.arange(features.shape[1])
    pairs = []
    for left, right in itertools.combinations(range(len(tasks)), 2):
        similarity = normalized[left] @ normalized[right].T
        forward = float(np.mean(np.argmax(similarity, axis=1) == identities))
        reverse = float(np.mean(np.argmax(similarity, axis=0) == identities))
        diagonal = float(np.trace(similarity) / len(identities))
        off_diagonal = float((similarity.sum(dtype=np.float64) - np.trace(similarity)) /
                             (len(identities) * (len(identities) - 1)))
        pairs.append({
            "tasks": [tasks[left], tasks[right]],
            "accuracy": (forward + reverse) / 2,
            "forward_accuracy": forward,
            "reverse_accuracy": reverse,
            "idiff_100": 100 * (diagonal - off_diagonal),
        })
    return pairs


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json",
                                 dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def plot(results: dict[str, list[dict]], path: Path, n_subjects: int) -> None:
    labels = ["|FC| 21", "FC 21", "Syn 21", "Syn 120"]
    colors = {"|FC| 21": "#506e8e", "FC 21": "#9eaebd", "Syn 21": "#c66842", "Syn 120": "#793f6f"}
    fig, (ax, delta_ax) = plt.subplots(1, 2, figsize=(10.2, 3.9),
                                       gridspec_kw={"width_ratios": [1.12, 1]},
                                       constrained_layout=True)
    rng = np.random.default_rng(17)
    baseline = np.asarray([row["accuracy"] for row in results["|FC| 21"]]) * 100
    for index, label in enumerate(labels):
        accuracy = np.asarray([row["accuracy"] for row in results[label]]) * 100
        y = index + rng.uniform(-0.13, 0.13, size=len(accuracy))
        ax.scatter(accuracy, y, s=18, color=colors[label], alpha=0.72, linewidth=0)
        ax.scatter([accuracy.mean()], [index], marker="D", s=65,
                   facecolor="white", edgecolor=colors[label], linewidth=1.7, zorder=3)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Top-1 identification (%)")
    ax.axvline(100 / n_subjects, color="#666666", lw=0.8, ls="--")
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    ax.text(0.02, 1.04, "a  Across 21 task pairs", transform=ax.transAxes,
            fontweight="bold", va="bottom")

    for index, label in enumerate(["Syn 21", "Syn 120"]):
        delta = np.asarray([row["accuracy"] for row in results[label]]) * 100 - baseline
        y = index + rng.uniform(-0.13, 0.13, size=len(delta))
        delta_ax.scatter(delta, y, s=18, color=colors[label], alpha=0.72, linewidth=0)
        delta_ax.scatter([delta.mean()], [index], marker="D", s=65,
                         facecolor="white", edgecolor=colors[label], linewidth=1.7, zorder=3)
    delta_ax.set_yticks([0, 1], ["Syn 21 − |FC| 21", "Syn 120 − |FC| 21"])
    delta_ax.invert_yaxis()
    delta_ax.set_xlabel("Paired accuracy difference (percentage points)")
    delta_ax.axvline(0, color="#666666", lw=0.8)
    delta_ax.grid(axis="x", alpha=0.18)
    delta_ax.set_axisbelow(True)
    delta_ax.text(0.02, 1.04, "b  Within each task pair", transform=delta_ax.transAxes,
                  fontweight="bold", va="bottom")
    for axis in (ax, delta_ax):
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=int, metavar="N", help="Run only the first N subjects without saving")
    args = parser.parse_args()
    subjects, tasks, features, quality = load_cached_features(args.smoke)
    features.update(compute_fc(subjects, tasks))
    results = {name: fingerprint(matrix, tasks) for name, matrix in features.items()}
    centered_results = {
        name: fingerprint(matrix, tasks, task_centered=True)
        for name, matrix in features.items()
    }
    aggregate = {
        name: {
            "mean_accuracy": float(np.mean([row["accuracy"] for row in pairs])),
            "median_accuracy": float(np.median([row["accuracy"] for row in pairs])),
            "mean_idiff_100": float(np.mean([row["idiff_100"] for row in pairs])),
        }
        for name, pairs in results.items()
    }
    baseline = np.asarray([row["accuracy"] for row in results["|FC| 21"]])
    differences = {
        name: {
            "mean_percentage_points": float(100 * np.mean(
                np.asarray([row["accuracy"] for row in results[name]]) - baseline)),
            "pairs_above_baseline": int(np.sum(
                np.asarray([row["accuracy"] for row in results[name]]) > baseline)),
            "pairs_below_baseline": int(np.sum(
                np.asarray([row["accuracy"] for row in results[name]]) < baseline)),
        }
        for name in ("Syn 21", "Syn 120")
    }
    summary = {
        "cohort_size": len(subjects),
        "tasks": tasks,
        "task_pair_count": 21,
        "gallery_size": len(subjects),
        "chance_accuracy": 1 / len(subjects),
        "feature_definition": {
            "Syn 21": "Cached PEID coalition synergy for all 21 two-network coalitions, bits",
            "Syn 120": "Cached PEID coalition synergy for all 120 coalitions of sizes 2 to 7, bits",
            "|FC| 21": "Absolute Pearson correlation of seven network PC1 time series, 21 edges",
            "FC 21": "Signed Pearson correlation of seven network PC1 time series, 21 edges",
        },
        "matching": "Row-wise Pearson correlation across features, highest similarity, both task directions",
        "quality": quality,
        "aggregate": aggregate,
        "differences_from_absolute_fc": differences,
        "task_pairs": results,
        "task_centered_sensitivity": {
            "description": "Subtract per-task cohort mean for each feature before row-wise Pearson matching; unsupervised and transductive",
            "aggregate": {
                name: {
                    "mean_accuracy": float(np.mean([row["accuracy"] for row in pairs])),
                    "mean_idiff_100": float(np.mean([row["idiff_100"] for row in pairs])),
                }
                for name, pairs in centered_results.items()
            },
            "task_pairs": centered_results,
        },
    }
    print(json.dumps({"cohort_size": len(subjects), "quality": quality,
                      "aggregate": aggregate, "differences_from_absolute_fc": differences,
                      "task_centered_aggregate": summary["task_centered_sensitivity"]["aggregate"]}, indent=2))
    if args.smoke is None:
        OUT.mkdir(parents=True, exist_ok=True)
        save_json(OUT / "summary.json", summary)
        plot(results, OUT / "cross_task_fingerprint.png", len(subjects))


if __name__ == "__main__":
    main()
