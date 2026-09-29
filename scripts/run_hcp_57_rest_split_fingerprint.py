#!/usr/bin/env python3
"""Refit each half of HCP REST1_LR and compare Syn with FC fingerprinting."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_task_evoked_pc2_xi_hierarchy import fit_project_network_pca
from scripts.phi_hierarchy import subset_phi_raw
from scripts.run_hcp_schaefer500_yeo7_module_phi_decomposition import (
    module_ei_table,
    network_history_indices,
)
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import load_yeo7_groups


REST_ROOT = ROOT / "data/hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_57_brain"
LABELS = ROOT / "data/hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30/_atlas_labels/Schaefer2018_1000Parcels_7Networks_order.txt"
REFERENCE = ROOT / "results/hcp_schaefer1000_task_evoked_xi_57/full/k1_p3_a1/arrays.npz"
OUTPUT = ROOT / "results/hcp_57_rest_split_fingerprint"
CACHE = OUTPUT / "features.npz"
SUMMARY = OUTPUT / "summary.json"
FIGURE = OUTPUT / "rest_split_fingerprint.png"
HALF_LENGTH = 600
FIT_LENGTH = 450
ORDER = 3
ALPHA = 1.0
SYN_TOLERANCE_BITS = 1e-9
BOOTSTRAPS = 5000
SEED = 20260929
UPPER = np.triu_indices(7, 1)
FEATURE_KEYS = ("fc_abs21", "fc_signed21", "syn21", "syn120")
DISPLAY = {
    "fc_abs21": "|FC| 21",
    "fc_signed21": "FC 21",
    "syn21": "Syn 21",
    "syn120": "Syn 120",
}


def discover_subjects() -> tuple[list[str], dict[str, Path]]:
    paths = {
        path.parent.name: path
        for path in REST_ROOT.glob("sub-*/*REST1_LR*schaefer500-1000_yeo7.mat")
    }
    subjects = sorted(paths)
    with np.load(REFERENCE, allow_pickle=False) as archive:
        reference = archive["subjects"].astype(str).tolist()
    if len(subjects) != 57 or subjects != reference:
        raise ValueError("REST cohort does not match the frozen 57-subject reference")
    return subjects, paths


def fit_one_half(half: np.ndarray, groups: dict[str, list[int]]) -> dict[str, np.ndarray | float]:
    if half.shape != (HALF_LENGTH, 1000) or not np.isfinite(half).all():
        raise ValueError(f"Expected finite REST half [600,1000], got {half.shape}")
    reduced, explained = fit_project_network_pca(
        half, half, groups, development_end=FIT_LENGTH, n_components=1
    )
    if reduced.shape != (HALF_LENGTH, 7) or not np.isfinite(reduced).all():
        raise ValueError("Network PCA produced an invalid seven-network representation")
    fitted = fit_delta_history_phi(
        reduced, alpha=ALPHA, order=ORDER, development_end=FIT_LENGTH
    )
    names = tuple(groups)
    table = module_ei_table(
        fitted["transition"], fitted["noise_covariance"],
        network_history_indices(names, order=ORDER), ridge=1e-6,
    )
    singleton = {name: float(table[(name,)]) for name in names}
    coalitions = tuple(
        combination
        for size in range(2, 8)
        for combination in itertools.combinations(names, size)
    )
    syn = np.asarray(
        [subset_phi_raw(combination, table, singleton) for combination in coalitions],
        dtype=np.float64,
    )
    if syn.shape != (120,) or not np.isfinite(syn).all():
        raise ValueError("Coalition synergy is incomplete or non-finite")
    violations = syn < -SYN_TOLERANCE_BITS
    if np.any(violations):
        raise ValueError(
            f"Syn nonnegativity violation: min={syn.min():.12g} bits, "
            f"threshold={-SYN_TOLERANCE_BITS:.12g} bits, count={int(violations.sum())}"
        )
    fc = np.corrcoef(reduced[:FIT_LENGTH], rowvar=False)[UPPER]
    if not np.isfinite(fc).all():
        raise ValueError("FC contains non-finite values")
    return {
        "pc1_fit": reduced[:FIT_LENGTH].copy(),
        "syn21": syn[:21],
        "syn120": syn,
        "fc_signed21": fc,
        "fc_abs21": np.abs(fc),
        "heldout_skill_ratio": float(fitted["heldout"]["skill_ratio"]),
        "mean_pc1_explained": float(np.mean([values[0] for values in explained.values()])),
    }


def fit_cohort(subjects: list[str], paths: dict[str, Path]) -> dict[str, np.ndarray]:
    groups = load_yeo7_groups(LABELS, expected_parcels=1000)
    records: dict[str, list[list[np.ndarray | float]]] = {
        key: [[], []] for key in (*FEATURE_KEYS, "pc1_fit", "heldout_skill_ratio", "mean_pc1_explained")
    }
    for index, subject in enumerate(subjects, start=1):
        raw = np.asarray(
            loadmat(paths[subject], variable_names=["Schaefer1000"])["Schaefer1000"],
            dtype=np.float64,
        )
        if raw.shape != (1200, 1000) or not np.isfinite(raw).all():
            raise ValueError(f"Invalid Schaefer-1000 REST scan for {subject}: {raw.shape}")
        for half_index in range(2):
            start = half_index * HALF_LENGTH
            result = fit_one_half(raw[start : start + HALF_LENGTH], groups)
            for key, value in result.items():
                records[key][half_index].append(value)
        print(f"Refitted REST halves: {index}/{len(subjects)} {subject}", flush=True)
    arrays = {key: np.asarray(values, dtype=np.float64) for key, values in records.items()}
    arrays["subjects"] = np.asarray(subjects)
    arrays["half_frame_starts"] = np.asarray([0, 600])
    arrays["fit_length"] = np.asarray(FIT_LENGTH)
    arrays["order"] = np.asarray(ORDER)
    arrays["ridge_alpha"] = np.asarray(ALPHA)
    arrays["syn_tolerance_bits"] = np.asarray(SYN_TOLERANCE_BITS)
    validate_features(arrays, subjects)
    return arrays


def validate_features(arrays: dict[str, np.ndarray], subjects: list[str]) -> None:
    if arrays["subjects"].astype(str).tolist() != subjects:
        raise ValueError("Feature cache subject order differs from REST cohort")
    if not np.array_equal(arrays["half_frame_starts"], np.asarray([0, 600])):
        raise ValueError("Feature cache does not use the frozen 600-frame halves")
    dimensions = {"fc_abs21": 21, "fc_signed21": 21, "syn21": 21, "syn120": 120}
    for key, dimension in dimensions.items():
        value = arrays[key]
        if value.shape != (2, len(subjects), dimension) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key} feature matrix: {value.shape}")
    for key in ("heldout_skill_ratio", "mean_pc1_explained"):
        value = arrays[key]
        if value.shape != (2, len(subjects)) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key} diagnostic matrix")
    if "pc1_fit" in arrays:
        if (arrays["pc1_fit"].shape != (2, len(subjects), FIT_LENGTH, 7)
                or not np.isfinite(arrays["pc1_fit"]).all()):
            raise ValueError("Invalid cached network PC1 fitting windows")
        from_pc1 = np.asarray([
            [np.corrcoef(series, rowvar=False)[UPPER] for series in half]
            for half in arrays["pc1_fit"]
        ])
        if not np.allclose(from_pc1, arrays["fc_signed21"], atol=1e-10):
            raise ValueError("Cached PC1 windows do not reproduce FC")
    if (int(arrays["fit_length"]) != FIT_LENGTH or int(arrays["order"]) != ORDER
            or float(arrays["ridge_alpha"]) != ALPHA
            or float(arrays["syn_tolerance_bits"]) != SYN_TOLERANCE_BITS):
        raise ValueError("Feature cache configuration differs from frozen experiment")
    if not np.array_equal(arrays["syn21"], arrays["syn120"][:, :, :21]):
        raise ValueError("Syn 21 does not match the first 21 full-coalition values")
    if not np.allclose(arrays["fc_abs21"], np.abs(arrays["fc_signed21"])):
        raise ValueError("Absolute FC differs from the signed FC magnitude")
    syn = arrays["syn120"]
    violations = syn < -SYN_TOLERANCE_BITS
    if np.any(violations):
        raise ValueError(
            f"Syn nonnegativity violation: min={syn.min():.12g} bits, "
            f"threshold={-SYN_TOLERANCE_BITS:.12g} bits, count={int(violations.sum())}"
        )


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".json", mode="w",
                                 encoding="utf-8", delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def unit_rows(value: np.ndarray) -> np.ndarray:
    centered = value - value.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("A feature vector is constant across coordinates")
    return centered / norm


def match(value: np.ndarray, *, group_centered: bool) -> tuple[dict, np.ndarray]:
    if group_centered:
        value = value - value.mean(axis=1, keepdims=True)
    left, right = (unit_rows(half) for half in value)
    similarity = left @ right.T
    identities = np.arange(len(left))
    forward = np.argmax(similarity, axis=1) == identities
    reverse = np.argmax(similarity, axis=0) == identities
    diagonal = float(np.trace(similarity) / len(left))
    off_diagonal = float((similarity.sum() - np.trace(similarity)) /
                         (len(left) * (len(left) - 1)))
    return {
        "forward_correct": int(forward.sum()),
        "reverse_correct": int(reverse.sum()),
        "forward_accuracy": float(forward.mean()),
        "reverse_accuracy": float(reverse.mean()),
        "mean_accuracy": float((forward.mean() + reverse.mean()) / 2),
        "mean_same_subject_similarity": diagonal,
        "mean_other_subject_similarity": off_diagonal,
        "idiff_100": 100 * (diagonal - off_diagonal),
    }, (forward.astype(float) + reverse.astype(float)) / 2


def paired_bootstrap(scores: dict[str, np.ndarray], n_subjects: int) -> dict[str, dict]:
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, n_subjects, size=(BOOTSTRAPS, n_subjects))
    output = {}
    for key in ("syn21", "syn120"):
        delta = (scores[key] - scores["fc_abs21"]) * 100
        bootstrap_means = delta[draws].mean(axis=1)
        output[key] = {
            "mean_percentage_points": float(delta.mean()),
            "conditional_95pct_interval_percentage_points":
                np.quantile(bootstrap_means, [0.025, 0.975]).astype(float).tolist(),
            "subjects_positive_delta": int(np.sum(delta > 0)),
            "subjects_negative_delta": int(np.sum(delta < 0)),
            "subjects_tied": int(np.sum(delta == 0)),
        }
    return output


def plot(summary: dict, path: Path) -> None:
    keys = ("fc_abs21", "fc_signed21", "syn21", "syn120")
    colors = {"fc_abs21": "#53718f", "fc_signed21": "#a9b8c5",
              "syn21": "#c76845", "syn120": "#7d487b"}
    fig, (ax, delta_ax) = plt.subplots(
        1, 2, figsize=(10.5, 4.0), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.05, 1]},
    )
    for index, key in enumerate(keys):
        raw = 100 * summary["raw"][key]["mean_accuracy"]
        centered = 100 * summary["group_centered_sensitivity"][key]["mean_accuracy"]
        ax.scatter(raw, index - 0.12, color=colors[key], s=58, zorder=3,
                   label="Raw" if index == 0 else None)
        ax.scatter(centered, index + 0.12, facecolors="white", edgecolors=colors[key],
                   linewidth=1.7, s=58, zorder=3,
                   label="Group centered" if index == 0 else None)
    ax.set_yticks(range(len(keys)), [DISPLAY[key] for key in keys])
    ax.invert_yaxis()
    ax.axvline(100 / summary["n_subjects"], color="#696969", ls="--", lw=0.9)
    ax.set_xlabel("Top-1 identification (%)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    ax.text(0, 1.19, "a  Within-scan split-half matching", transform=ax.transAxes,
            fontweight="bold", va="bottom")

    for index, key in enumerate(("syn21", "syn120")):
        row = summary["paired_difference_from_abs_fc"][key]
        point = row["mean_percentage_points"]
        low, high = row["conditional_95pct_interval_percentage_points"]
        delta_ax.errorbar(point, index, xerr=[[point - low], [high - point]],
                          fmt="o", color=colors[key], capsize=3, markersize=7, lw=1.5)
    delta_ax.set_yticks((0, 1), ("Syn 21 − |FC| 21", "Syn 120 − |FC| 21"))
    delta_ax.invert_yaxis()
    delta_ax.axvline(0, color="#696969", lw=0.9)
    delta_ax.set_xlabel("Raw accuracy difference (percentage points)")
    delta_ax.grid(axis="x", alpha=0.2)
    delta_ax.set_axisbelow(True)
    delta_ax.text(0, 1.19, "b  Paired difference (95% interval)",
                  transform=delta_ax.transAxes, fontweight="bold", va="bottom")
    for axis in (ax, delta_ax):
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=int, metavar="N", help="Fit only the first N subjects without saving")
    parser.add_argument("--recompute", action="store_true", help="Refit even if the full feature cache exists")
    args = parser.parse_args()
    subjects, paths = discover_subjects()
    if args.smoke is not None:
        if args.smoke < 2 or args.smoke > len(subjects):
            parser.error("--smoke must be between 2 and 57")
        subjects = subjects[:args.smoke]
    if args.smoke is None and CACHE.exists() and not args.recompute:
        with np.load(CACHE, allow_pickle=False) as archive:
            features = {key: archive[key] for key in archive.files}
        validate_features(features, subjects)
        print("Loaded reusable REST split-half feature cache", flush=True)
    else:
        features = fit_cohort(subjects, paths)
        if args.smoke is None:
            atomic_npz(CACHE, features)
    results = {}
    scores = {}
    centered = {}
    for key in FEATURE_KEYS:
        results[key], scores[key] = match(features[key], group_centered=False)
        centered[key], _ = match(features[key], group_centered=True)
    syn = features["syn120"]
    violations = syn < -SYN_TOLERANCE_BITS
    summary = {
        "n_subjects": len(subjects),
        "scan": "REST1_LR Schaefer-1000, one 1200-frame scan per subject",
        "split_frames": [[0, 600], [600, 1200]],
        "fit_frames_per_half": FIT_LENGTH,
        "model": {"network_pc": 1, "history_order": ORDER, "ridge_alpha": ALPHA,
                  "ei_estimator": "repository Gaussian EI on independently fitted linear transition"},
        "gallery_size": len(subjects),
        "chance_accuracy": 1 / len(subjects),
        "matching": "Pearson across feature coordinates, top-1, both directions",
        "raw": results,
        "group_centered_sensitivity": centered,
        "paired_difference_from_abs_fc": paired_bootstrap(scores, len(subjects)),
        "uncertainty": {"bootstrap_replicates": BOOTSTRAPS, "seed": SEED,
                        "interpretation": "Paired probe-subject bootstrap with the full gallery fixed"},
        "quality": {
            "syn_tolerance_bits": SYN_TOLERANCE_BITS,
            "minimum_syn_bits": float(syn.min()),
            "negative_within_tolerance_count": int(np.sum((syn < 0) & ~violations)),
            "significant_negative_count": int(violations.sum()),
            "syn_value_count": int(syn.size),
            "heldout_skill_ratio_mean": float(features["heldout_skill_ratio"].mean()),
            "heldout_better_than_persistence_count":
                int(np.sum(features["heldout_skill_ratio"] < 1)),
            "fit_count": 2 * len(subjects),
            "mean_pc1_explained": float(features["mean_pc1_explained"].mean()),
        },
    }
    print(json.dumps({"n_subjects": len(subjects), "raw": results,
                      "paired_difference_from_abs_fc": summary["paired_difference_from_abs_fc"],
                      "quality": summary["quality"]}, indent=2), flush=True)
    if args.smoke is None:
        atomic_json(SUMMARY, summary)
        plot(summary, FIGURE)


if __name__ == "__main__":
    main()
