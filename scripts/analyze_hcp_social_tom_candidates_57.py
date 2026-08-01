#!/usr/bin/env python3
"""Test prespecified SOCIAL TOM coalitions in the expanded 57-subject cohort.

The three candidates operationalize the preregistered verbal family
"Default-Limbic-Control/Salience".  Inference is cohort-blocked and controls
Random-condition performance, age, sex, and recruitment cohort.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TASK_ROOT = ROOT / "data/hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_57_brain"
LABELS = ROOT / "data/hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30/_atlas_labels/Schaefer2018_1000Parcels_7Networks_order.txt"
BEHAVIOR = ROOT / "data/unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
SUBJECT_SOURCE = ROOT / "results/hcp_schaefer1000_panels_e_i_57/panel_values_57.npz"
OUTPUT = ROOT / "results/hcp_social_tom_candidates_57"
CACHE = OUTPUT / "social_candidate_synergy_57.npz"

NETWORK_ORDER = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
CANDIDATES = (
    ("Limbic", "Cont", "Default"),
    ("SalVentAttn", "Limbic", "Default"),
    ("SalVentAttn", "Limbic", "Cont", "Default"),
)
DISPLAY_NAMES = ("Limbic + Control + Default", "Salience + Limbic + Default", "Salience + Limbic + Control + Default")
SEED = 20260802
PERMUTATIONS = 100_000
BOOTSTRAPS = 20_000
SYN_TOLERANCE_BITS = 1.0e-9
ORDER = 3
ALPHA = 1.0


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def age_midpoint(value: str) -> float:
    if value == "36+":
        return 38.0
    low, high = value.split("-")
    return 0.5 * (float(low) + float(high))


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def standardized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centered = values - values.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.sum(centered**2, axis=0, keepdims=True))
    if np.any(norm <= 1.0e-12):
        raise ValueError("A tested variable is constant after residualization.")
    return centered / norm


def candidate_names() -> np.ndarray:
    return np.asarray(["+".join(value) for value in CANDIDATES])


def load_subjects() -> np.ndarray:
    with np.load(SUBJECT_SOURCE, allow_pickle=False) as archive:
        subjects = archive["subjects"].astype(str)
    if subjects.shape != (57,) or len(set(subjects.tolist())) != 57:
        raise ValueError("Expected 57 unique subjects in the frozen merged order.")
    return subjects


def compute_metrics(subjects: np.ndarray, recompute: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if CACHE.is_file() and not recompute:
        with np.load(CACHE, allow_pickle=False) as archive:
            if np.array_equal(archive["subjects"].astype(str), subjects) and np.array_equal(archive["coalitions"].astype(str), candidate_names()):
                values = archive["synergy_bits"].astype(float)
                if values.shape == (57, 3) and np.isfinite(values).all():
                    return values, archive["heldout_skill_ratio"].astype(float), archive["mean_pc1_explained"].astype(float)

    from scripts.analyze_hcp_task_evoked_pc2_xi_hierarchy import network_module_indices
    from scripts.phi_hierarchy import subset_phi_raw
    from scripts.run_hcp_schaefer500_yeo7_module_phi_decomposition import module_ei_table
    from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import load_yeo7_groups
    from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi
    from scripts.tune_hcp_task_evoked_xi_hierarchy import prepare_projection

    groups = load_yeo7_groups(LABELS, expected_parcels=1000)
    indices = network_module_indices(NETWORK_ORDER, n_components=1, order=ORDER)
    values = np.empty((57, 3), dtype=float)
    heldout = np.empty(57, dtype=float)
    explained = np.empty(57, dtype=float)
    for index, subject in enumerate(subjects):
        path = TASK_ROOT / str(subject) / "SOCIAL_LR.mat"
        projections, variance, development_end = prepare_projection(
            path, groups, state="SOCIAL", max_components=1,
            task_retained_key="Schaefer1000_taskRetained",
            task_regressed_key="Schaefer1000_taskRegressed", expected_parcels=1000,
        )
        fitted = fit_delta_history_phi(projections[1], alpha=ALPHA, order=ORDER, development_end=development_end)
        table = module_ei_table(fitted["transition"], fitted["noise_covariance"], indices, ridge=1.0e-6)
        singleton = {name: float(table[(name,)]) for name in NETWORK_ORDER}
        values[index] = [subset_phi_raw(candidate, table, singleton) for candidate in CANDIDATES]
        heldout[index] = float(fitted["heldout"]["skill_ratio"])
        explained[index] = float(np.mean(list(variance[1].values())))
        print(f"[{index + 1:02d}/57] {subject}", flush=True)

    violation = values < -SYN_TOLERANCE_BITS
    if np.any(violation):
        raise ValueError(
            "PEID Syn nonnegativity violation: "
            f"minimum={values.min():.12g}, threshold={-SYN_TOLERANCE_BITS:.12g}, "
            f"count={int(violation.sum())}"
        )
    atomic_npz(
        CACHE, subjects=subjects, coalitions=candidate_names(), synergy_bits=values,
        heldout_skill_ratio=heldout, mean_pc1_explained=explained,
        syn_tolerance_bits=np.asarray(SYN_TOLERANCE_BITS),
    )
    return values, heldout, explained


def load_behavior(subjects: np.ndarray) -> dict[str, np.ndarray]:
    with BEHAVIOR.open(newline="", encoding="utf-8-sig") as handle:
        rows = {str(row["Subject"]): row for row in csv.DictReader(handle)}
    keys = [str(value).removeprefix("sub-") for value in subjects]
    if any(key not in rows for key in keys):
        raise ValueError("Behavior table is missing one or more merged subjects.")
    return {
        "tom": np.asarray([float(rows[key]["Social_Task_TOM_Perc_TOM"]) for key in keys]),
        "random": np.asarray([float(rows[key]["Social_Task_Random_Perc_Random"]) for key in keys]),
        "age": np.asarray([age_midpoint(rows[key]["Age"]) for key in keys]),
        "sex": np.asarray([rows[key]["Gender"] == "M" for key in keys], dtype=float),
        "cohort": np.r_[np.zeros(29, dtype=int), np.ones(28, dtype=int)],
    }


def primary_test(brain: np.ndarray, behavior: Mapping[str, np.ndarray], permutations: int, seed: int) -> dict[str, Any]:
    cohort = behavior["cohort"].astype(int)
    nuisance = np.column_stack([
        np.ones(len(cohort)), rankdata(behavior["random"]), behavior["age"], behavior["sex"], cohort,
    ])
    brain_rank = rankdata(brain, axis=0)
    brain_residual = standardized(residualize(brain_rank, nuisance))
    tom_rank = rankdata(behavior["tom"])
    fitted = nuisance @ np.linalg.lstsq(nuisance, tom_rank, rcond=None)[0]
    tom_residual = residualize(tom_rank, nuisance)
    tom_unit = standardized(tom_residual[:, None])[:, 0]
    observed = brain_residual.T @ tom_unit

    counts = np.zeros(brain.shape[1], dtype=np.int64)
    max_counts = np.zeros(brain.shape[1], dtype=np.int64)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(cohort == value) for value in np.unique(cohort)]
    chunk = 1_000
    for start in range(0, permutations, chunk):
        size = min(chunk, permutations - start)
        permuted = np.tile(np.arange(len(cohort)), (size, 1))
        for group in groups:
            order = np.argsort(rng.random((size, len(group))), axis=1)
            permuted[:, group] = group[order]
        pseudo = fitted[None, :] + tom_residual[permuted]
        coefficients = np.linalg.lstsq(nuisance, pseudo.T, rcond=None)[0]
        residual = pseudo - (nuisance @ coefficients).T
        residual /= np.linalg.norm(residual, axis=1, keepdims=True)
        null = residual @ brain_residual
        absolute = np.abs(null)
        counts += np.sum(absolute >= np.abs(observed)[None, :], axis=0)
        maximum = absolute.max(axis=1)
        max_counts += np.sum(maximum[:, None] >= np.abs(observed)[None, :], axis=0)
    return {
        "partial_rho": observed,
        "p_pointwise": (counts + 1) / (permutations + 1),
        "p_max_t_three_candidates": (max_counts + 1) / (permutations + 1),
    }


def stratified_bootstrap(brain: np.ndarray, behavior: Mapping[str, np.ndarray], repeats: int, seed: int) -> np.ndarray:
    cohort = behavior["cohort"].astype(int)
    groups = [np.flatnonzero(cohort == value) for value in np.unique(cohort)]
    rng = np.random.default_rng(seed)
    output = np.full((repeats, brain.shape[1]), np.nan)
    for index in range(repeats):
        sample = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        design = np.column_stack([
            np.ones(len(sample)), rankdata(behavior["random"][sample]), behavior["age"][sample],
            behavior["sex"][sample], behavior["cohort"][sample],
        ])
        y = residualize(rankdata(behavior["tom"][sample]), design)
        x = residualize(rankdata(brain[sample], axis=0), design)
        y_norm = np.linalg.norm(y)
        x_norm = np.linalg.norm(x, axis=0)
        valid = (y_norm > 1e-12) & (x_norm > 1e-12)
        output[index, valid] = (x[:, valid] / x_norm[valid]).T @ (y / y_norm)
    return np.nanquantile(output, [0.025, 0.5, 0.975], axis=0)


def behavior_summary(values: np.ndarray) -> dict[str, Any]:
    unique, counts = np.unique(values, return_counts=True)
    return {
        "n": int(len(values)), "unique_values": int(len(unique)),
        "range": [float(values.min()), float(values.max())],
        "mode": float(unique[np.argmax(counts)]),
        "mode_share": float(counts.max() / len(values)),
        "sd": float(np.std(values, ddof=1)),
    }


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def plot(subjects: np.ndarray, brain: np.ndarray, behavior: Mapping[str, np.ndarray], summary: Mapping[str, Any]) -> None:
    configure_style()
    colors = np.asarray(["#7A8DA6", "#D07A55"])[behavior["cohort"].astype(int)]
    figure = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[0.9, 1.1])
    ax_a, ax_b, ax_c, ax_d = [figure.add_subplot(grid[i, j]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]

    rng = np.random.default_rng(SEED)
    for cohort_value, label, color in ((0, "Original 29", "#7A8DA6"), (1, "Supplement 28", "#D07A55")):
        mask = behavior["cohort"] == cohort_value
        x = np.full(mask.sum(), cohort_value) + rng.uniform(-0.10, 0.10, mask.sum())
        ax_a.scatter(x, behavior["tom"][mask], s=18, color=color, alpha=0.82, edgecolor="white", linewidth=0.35)
    ax_a.set_xticks([0, 1], ["Original 29", "Supplement 28"])
    ax_a.set_ylabel("TOM accuracy (%)")
    ax_a.set_title("a  Ceiling relief", loc="left", fontweight="bold")
    ax_a.text(0.02, 0.04, "Unique values: 3 → 7\nCeiling share: 75.9% → 35.7%", transform=ax_a.transAxes, va="bottom")

    candidates = summary["candidates"]
    y = np.arange(3)[::-1]
    estimates = np.asarray([item["partial_rho"] for item in candidates])
    intervals = np.asarray([item["bootstrap_ci95"] for item in candidates])
    ax_b.axvline(0, color="#B9C0C8", lw=0.8)
    ax_b.errorbar(estimates, y, xerr=np.vstack([estimates - intervals[:, 0], intervals[:, 1] - estimates]), fmt="o", color="#384E6B", ecolor="#6E8098", capsize=2.5, ms=4)
    ax_b.set_yticks(y, DISPLAY_NAMES)
    ax_b.set_xlabel(r"Partial Spearman $\rho$")
    ax_b.set_title("b  Prespecified candidate family", loc="left", fontweight="bold")
    for yi, item in zip(y, candidates):
        ax_b.text(0.98, yi, f"max-T p={item['p_max_t_three_candidates']:.3g}", transform=ax_b.get_yaxis_transform(), ha="right", va="bottom", fontsize=6)

    best = int(np.argmin([item["p_max_t_three_candidates"] for item in candidates]))
    jitter = rng.uniform(-1.2, 1.2, len(subjects))
    for axis, endpoint, letter, xlabel in ((ax_c, "tom", "c", "TOM accuracy (%)"), (ax_d, "random", "d", "Random accuracy (%)")):
        x = behavior[endpoint]
        axis.scatter(x + jitter, brain[:, best], c=colors, s=19, alpha=0.84, edgecolor="white", linewidth=0.35)
        order = np.argsort(x)
        coefficient = np.polyfit(x, brain[:, best], 1)
        axis.plot(x[order], np.polyval(coefficient, x[order]), color="#465563", ls="--", lw=0.9)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Coalition synergy (bits)")
        axis.set_title(f"{letter}  {DISPLAY_NAMES[best]}", loc="left", fontweight="bold")
    best_item = candidates[best]
    ax_c.text(0.02, 0.98, rf"partial $\rho$={best_item['partial_rho']:+.3f}" + f"\nmax-T $p$={best_item['p_max_t_three_candidates']:.4f}\n$n$=57", transform=ax_c.transAxes, va="top")
    raw_random = float(spearmanr(brain[:, best], behavior["random"]).statistic)
    ax_d.text(0.02, 0.98, rf"raw $\rho$={raw_random:+.3f}" + "\ncontrol endpoint", transform=ax_d.transAxes, va="top")
    handles = [mpl.lines.Line2D([], [], marker="o", ls="none", color=color, label=label, markersize=4) for label, color in (("Original 29", "#7A8DA6"), ("Supplement 28", "#D07A55"))]
    ax_d.legend(handles=handles, loc="lower right")

    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        figure.savefig(OUTPUT / f"social_tom_candidates_57.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(figure)


def write_report(summary: Mapping[str, Any]) -> None:
    rows = []
    for item in summary["candidates"]:
        rows.append(
            f"| {item['display_name']} | {item['old_29_rho']:+.3f} | {item['supplement_28_rho']:+.3f} | "
            f"{item['partial_rho']:+.3f} | [{item['bootstrap_ci95'][0]:+.3f}, {item['bootstrap_ci95'][1]:+.3f}] | "
            f"{item['p_pointwise']:.5f} | {item['p_max_t_three_candidates']:.5f} |"
        )
    report = [
        "# SOCIAL TOM candidate validation in 57 subjects", "",
        "![SOCIAL TOM candidates](social_tom_candidates_57.png)", "",
        "| Candidate | Original 29 raw rho | Supplement 28 raw rho | Adjusted partial rho | Stratified bootstrap 95% CI | Pointwise p | Three-candidate max-T p |",
        "|---|---:|---:|---:|---:|---:|---:|", *rows, "",
        "Primary inference controls Random-condition accuracy, age, sex, and cohort. Permutations are restricted within the original and supplementary cohorts. The three named coalitions form one prespecified family.", "",
        "The SOCIAL neural metric is estimated from the full SOCIAL run, so behavioral specificity after controlling Random does not imply block-specific neural estimation.", "",
    ]
    (OUTPUT / "report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    subjects = load_subjects()
    brain, heldout, explained = compute_metrics(subjects, args.recompute)
    behavior = load_behavior(subjects)
    test = primary_test(brain, behavior, args.permutations, SEED)
    intervals = stratified_bootstrap(brain, behavior, args.bootstraps, SEED + 1)
    candidates = []
    for index, name in enumerate(candidate_names()):
        candidates.append({
            "coalition": str(name), "display_name": DISPLAY_NAMES[index],
            "partial_rho": float(test["partial_rho"][index]),
            "p_pointwise": float(test["p_pointwise"][index]),
            "p_max_t_three_candidates": float(test["p_max_t_three_candidates"][index]),
            "bootstrap_ci95": [float(intervals[0, index]), float(intervals[2, index])],
            "bootstrap_median": float(intervals[1, index]),
            "pooled_raw_rho": float(spearmanr(brain[:, index], behavior["tom"]).statistic),
            "old_29_rho": float(spearmanr(brain[:29, index], behavior["tom"][:29]).statistic),
            "supplement_28_rho": float(spearmanr(brain[29:, index], behavior["tom"][29:]).statistic),
            "random_raw_rho": float(spearmanr(brain[:, index], behavior["random"]).statistic),
        })
    summary = {
        "experiment": "SOCIAL TOM association for prespecified Default-Limbic-Control/Salience coalitions",
        "n": 57, "parcellation": "Schaefer-1000 Yeo7", "state": "SOCIAL",
        "brain_measure_is_condition_specific": False,
        "configuration": {"history_order": ORDER, "ridge_alpha": ALPHA, "direction": "LR", "syn_tolerance_bits": SYN_TOLERANCE_BITS},
        "inference": {"permutations": args.permutations, "bootstraps": args.bootstraps, "permutation_blocks": [29, 28], "covariates": ["Random accuracy", "age", "sex", "cohort"], "multiplicity": "max-T across three prespecified coalitions"},
        "behavior": {"original_29_tom": behavior_summary(behavior["tom"][:29]), "supplement_28_tom": behavior_summary(behavior["tom"][29:]), "combined_57_tom": behavior_summary(behavior["tom"]), "combined_tom_random_rho": float(spearmanr(behavior["tom"], behavior["random"]).statistic)},
        "model_diagnostics": {"mean_heldout_skill_ratio": float(heldout.mean()), "models_better_than_persistence": int(np.sum(heldout < 1)), "mean_pc1_explained": float(explained.mean()), "minimum_synergy_bits": float(brain.min()), "negative_within_tolerance_count": int(np.sum((brain < 0) & (brain >= -SYN_TOLERANCE_BITS)))},
        "candidates": candidates,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(subjects, brain, behavior, summary)
    write_report(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
