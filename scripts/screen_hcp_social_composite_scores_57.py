#!/usr/bin/env python3
"""Screen SOCIAL Yeo-7 coalitions against balanced accuracy and corrected d'."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_task_evoked_pc2_xi_hierarchy import network_module_indices
from scripts.phi_hierarchy import subset_phi_raw
from scripts.run_hcp_schaefer500_yeo7_module_phi_decomposition import module_ei_table
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import load_yeo7_groups
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi
from scripts.tune_hcp_task_evoked_xi_hierarchy import prepare_projection


TASK_ROOT = ROOT / "data/hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_57_brain"
LABELS = ROOT / "data/hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30/_atlas_labels/Schaefer2018_1000Parcels_7Networks_order.txt"
BEHAVIOR = ROOT / "data/unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
SUBJECT_SOURCE = ROOT / "results/hcp_schaefer1000_panels_e_i_57/panel_values_57.npz"
CANDIDATE_CACHE = ROOT / "results/hcp_social_tom_candidates_57/social_candidate_synergy_57.npz"
OUTPUT = ROOT / "results/hcp_social_composite_scores_57"
CACHE = OUTPUT / "social_coalition_synergy_57.npz"
PARTIAL_CACHE = OUTPUT / "social_coalition_synergy_57.partial.npz"

NETWORK_ORDER = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
SHORT = {"Vis": "Vis", "SomMot": "Som", "DorsAttn": "DAN", "SalVentAttn": "SVAN", "Limbic": "Lim", "Cont": "Cont", "Default": "DMN"}
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


def coalitions() -> tuple[tuple[str, ...], ...]:
    return tuple(c for size in range(2, 8) for c in itertools.combinations(NETWORK_ORDER, size))


def coalition_names(values: Sequence[Sequence[str]]) -> np.ndarray:
    return np.asarray(["+".join(value) for value in values])


def age_midpoint(value: str) -> float:
    if value == "36+":
        return 38.0
    low, high = value.split("-")
    return 0.5 * (float(low) + float(high))


def load_subjects() -> np.ndarray:
    with np.load(SUBJECT_SOURCE, allow_pickle=False) as archive:
        subjects = archive["subjects"].astype(str)
    if subjects.shape != (57,) or len(set(subjects.tolist())) != 57:
        raise ValueError("Expected the frozen 57-subject order.")
    return subjects


def compute_matrix(subjects: np.ndarray, combos: Sequence[tuple[str, ...]], recompute: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = coalition_names(combos)
    if CACHE.is_file() and not recompute:
        with np.load(CACHE, allow_pickle=False) as archive:
            if np.array_equal(archive["subjects"].astype(str), subjects) and np.array_equal(archive["coalitions"].astype(str), names):
                matrix = archive["synergy_bits"].astype(float)
                if matrix.shape == (57, 120) and np.isfinite(matrix).all():
                    return matrix, archive["heldout_skill_ratio"].astype(float), archive["mean_pc1_explained"].astype(float)

    matrix = np.full((57, 120), np.nan)
    heldout = np.full(57, np.nan)
    explained = np.full(57, np.nan)
    if PARTIAL_CACHE.is_file() and not recompute:
        with np.load(PARTIAL_CACHE, allow_pickle=False) as archive:
            if np.array_equal(archive["subjects"].astype(str), subjects) and np.array_equal(archive["coalitions"].astype(str), names):
                matrix = archive["synergy_bits"].astype(float)
                heldout = archive["heldout_skill_ratio"].astype(float)
                explained = archive["mean_pc1_explained"].astype(float)

    groups = load_yeo7_groups(LABELS, expected_parcels=1000)
    indices = network_module_indices(NETWORK_ORDER, n_components=1, order=ORDER)
    for index in np.flatnonzero(~np.isfinite(matrix).all(axis=1)):
        subject = str(subjects[index])
        projections, variance, development_end = prepare_projection(
            TASK_ROOT / subject / "SOCIAL_LR.mat", groups, state="SOCIAL", max_components=1,
            task_retained_key="Schaefer1000_taskRetained", task_regressed_key="Schaefer1000_taskRegressed",
            expected_parcels=1000,
        )
        fitted = fit_delta_history_phi(projections[1], alpha=ALPHA, order=ORDER, development_end=development_end)
        table = module_ei_table(fitted["transition"], fitted["noise_covariance"], indices, ridge=1.0e-6)
        singleton = {name: float(table[(name,)]) for name in NETWORK_ORDER}
        matrix[index] = [subset_phi_raw(combo, table, singleton) for combo in combos]
        heldout[index] = float(fitted["heldout"]["skill_ratio"])
        explained[index] = float(np.mean(list(variance[1].values())))
        atomic_npz(PARTIAL_CACHE, subjects=subjects, coalitions=names, synergy_bits=matrix, heldout_skill_ratio=heldout, mean_pc1_explained=explained)
        print(f"[{index + 1:02d}/57] {subject}", flush=True)

    violation = matrix < -SYN_TOLERANCE_BITS
    if np.any(violation):
        raise ValueError(f"PEID Syn violation: min={matrix.min():.12g}, count={int(violation.sum())}")
    if CANDIDATE_CACHE.is_file():
        with np.load(CANDIDATE_CACHE, allow_pickle=False) as archive:
            candidate_names = archive["coalitions"].astype(str)
            candidate_values = archive["synergy_bits"].astype(float)
        maximum = max(float(np.max(np.abs(matrix[:, names.tolist().index(name)] - candidate_values[:, i]))) for i, name in enumerate(candidate_names))
        if maximum > 1.0e-10:
            raise ValueError(f"Candidate-cache smoke check failed: max abs difference={maximum:.3g}")
    atomic_npz(CACHE, subjects=subjects, coalitions=names, coalition_sizes=np.asarray([len(x) for x in combos]), synergy_bits=matrix, heldout_skill_ratio=heldout, mean_pc1_explained=explained, syn_tolerance_bits=np.asarray(SYN_TOLERANCE_BITS))
    if PARTIAL_CACHE.exists():
        PARTIAL_CACHE.unlink()
    return matrix, heldout, explained


def infer_effective_trials(row: Mapping[str, str]) -> int:
    tom_fields = ["Social_Task_TOM_Perc_Random", "Social_Task_TOM_Perc_TOM", "Social_Task_TOM_Perc_Unsure", "Social_Task_TOM_Perc_NLR"]
    random_fields = ["Social_Task_Random_Perc_Random", "Social_Task_Random_Perc_TOM", "Social_Task_Random_Perc_Unsure", "Social_Task_Random_Perc_NLR"]
    overall_fields = ["Social_Task_Perc_Random", "Social_Task_Perc_TOM", "Social_Task_Perc_Unsure", "Social_Task_Perc_NLR"]
    condition = np.asarray([float(row[x]) for x in tom_fields + random_fields]) / 100.0
    overall = np.asarray([float(row[x]) for x in overall_fields]) / 100.0
    for trials in range(5, 101, 5):
        error = max(float(np.max(np.abs(condition * trials - np.round(condition * trials)))), float(np.max(np.abs(overall * 2 * trials - np.round(overall * 2 * trials)))))
        if error < 0.011:
            return trials
    raise ValueError(f"Could not infer effective SOCIAL trial count for subject {row['Subject']}")


def load_scores(subjects: np.ndarray) -> dict[str, np.ndarray]:
    with BEHAVIOR.open(newline="", encoding="utf-8-sig") as handle:
        table = {str(row["Subject"]): row for row in csv.DictReader(handle)}
    keys = [str(x).removeprefix("sub-") for x in subjects]
    rows = [table[key] for key in keys]
    hit_percent = np.asarray([float(row["Social_Task_TOM_Perc_TOM"]) for row in rows])
    correct_reject_percent = np.asarray([float(row["Social_Task_Random_Perc_Random"]) for row in rows])
    trials = np.asarray([infer_effective_trials(row) for row in rows], dtype=int)
    hits = np.rint(hit_percent * trials / 100.0)
    false_alarms = np.rint((100.0 - correct_reject_percent) * trials / 100.0)
    corrected_hit = (hits + 0.5) / (trials + 1.0)
    corrected_false_alarm = (false_alarms + 0.5) / (trials + 1.0)
    z_hit, z_false_alarm = norm.ppf(corrected_hit), norm.ppf(corrected_false_alarm)
    return {
        "hit_percent": hit_percent,
        "correct_reject_percent": correct_reject_percent,
        "balanced_accuracy": 0.5 * (hit_percent + correct_reject_percent),
        "dprime": z_hit - z_false_alarm,
        "criterion": -0.5 * (z_hit + z_false_alarm),
        "effective_trials": trials,
        "age": np.asarray([age_midpoint(row["Age"]) for row in rows]),
        "sex": np.asarray([row["Gender"] == "M" for row in rows], dtype=float),
        "cohort": np.r_[np.zeros(29, dtype=int), np.ones(28, dtype=int)],
    }


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def unit_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    values = values - values.mean(axis=0, keepdims=True)
    norm_value = np.linalg.norm(values, axis=0, keepdims=True)
    if np.any(norm_value <= 1.0e-12):
        raise ValueError("Constant residualized variable.")
    return values / norm_value


def bh(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=float).ravel()
    order = np.argsort(flat)
    ranked = flat[order]
    adjusted_ranked = np.minimum.accumulate((ranked * len(flat) / np.arange(1, len(flat) + 1))[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(1.0, adjusted_ranked)
    return adjusted.reshape(np.asarray(values).shape)


def screen(matrix: np.ndarray, scores: Mapping[str, np.ndarray], permutations: int, seed: int) -> dict[str, np.ndarray]:
    cohort = scores["cohort"].astype(int)
    design = np.column_stack([np.ones(len(cohort)), scores["age"], scores["sex"], cohort])
    brain = unit_columns(residualize(rankdata(matrix, axis=0), design))
    endpoints = np.column_stack([rankdata(scores["balanced_accuracy"]), rankdata(scores["dprime"])])
    fitted = design @ np.linalg.lstsq(design, endpoints, rcond=None)[0]
    endpoint_residual = residualize(endpoints, design)
    endpoint_unit = unit_columns(endpoint_residual)
    observed = brain.T @ endpoint_unit
    counts = np.zeros_like(observed, dtype=np.int64)
    endpoint_counts = np.zeros_like(observed, dtype=np.int64)
    global_counts = np.zeros_like(observed, dtype=np.int64)
    groups = [np.flatnonzero(cohort == value) for value in np.unique(cohort)]
    rng = np.random.default_rng(seed)
    chunk = 500
    for start in range(0, permutations, chunk):
        size = min(chunk, permutations - start)
        permuted = np.tile(np.arange(len(cohort)), (size, 1))
        for group in groups:
            order = np.argsort(rng.random((size, len(group))), axis=1)
            permuted[:, group] = group[order]
        pseudo = fitted[None, :, :] + endpoint_residual[permuted]
        coefficient = np.linalg.lstsq(design, pseudo.transpose(1, 0, 2).reshape(len(cohort), -1), rcond=None)[0]
        residual = pseudo - (design @ coefficient).reshape(len(cohort), size, 2).transpose(1, 0, 2)
        residual /= np.linalg.norm(residual, axis=1, keepdims=True)
        null = np.einsum("nc,pne->pce", brain, residual, optimize=True)
        absolute = np.abs(null)
        counts += np.sum(absolute >= np.abs(observed)[None, :, :], axis=0)
        endpoint_max = absolute.max(axis=1)
        global_max = absolute.max(axis=(1, 2))
        endpoint_counts += np.sum(endpoint_max[:, None, :] >= np.abs(observed)[None, :, :], axis=0)
        global_counts += np.sum(global_max[:, None, None] >= np.abs(observed)[None, :, :], axis=0)
    p_raw = (counts + 1) / (permutations + 1)
    return {"rho_adjusted": observed, "p_raw": p_raw, "q_global_240": bh(p_raw), "p_endpoint_max_t_120": (endpoint_counts + 1) / (permutations + 1), "p_global_max_t_240": (global_counts + 1) / (permutations + 1)}


def bootstrap_candidate(brain: np.ndarray, endpoint: np.ndarray, scores: Mapping[str, np.ndarray], repeats: int, seed: int) -> list[float]:
    cohort = scores["cohort"].astype(int)
    groups = [np.flatnonzero(cohort == value) for value in np.unique(cohort)]
    rng = np.random.default_rng(seed)
    values = np.full(repeats, np.nan)
    for index in range(repeats):
        sample = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        design = np.column_stack([np.ones(len(sample)), scores["age"][sample], scores["sex"][sample], scores["cohort"][sample]])
        x = residualize(rankdata(brain[sample]), design)
        y = residualize(rankdata(endpoint[sample]), design)
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        if denominator > 1e-12:
            values[index] = float(x @ y / denominator)
    return np.nanquantile(values, [0.025, 0.5, 0.975]).tolist()


def make_rows(names: np.ndarray, sizes: np.ndarray, matrix: np.ndarray, scores: Mapping[str, np.ndarray], result: Mapping[str, np.ndarray], bootstraps: int) -> tuple[list[dict[str, Any]], list[int]]:
    rows = []
    endpoints = ("balanced_accuracy", "dprime")
    for index, name in enumerate(names):
        row: dict[str, Any] = {"index": index, "coalition": str(name), "short_coalition": "+".join(SHORT[x] for x in str(name).split("+")), "coalition_size": int(sizes[index])}
        for endpoint_index, endpoint in enumerate(endpoints):
            row[endpoint] = {
                "rho_adjusted": float(result["rho_adjusted"][index, endpoint_index]),
                "rho_pooled_raw": float(spearmanr(matrix[:, index], scores[endpoint]).statistic),
                "rho_original_29": float(spearmanr(matrix[:29, index], scores[endpoint][:29]).statistic),
                "rho_supplement_28": float(spearmanr(matrix[29:, index], scores[endpoint][29:]).statistic),
                "p_raw": float(result["p_raw"][index, endpoint_index]),
                "q_global_240": float(result["q_global_240"][index, endpoint_index]),
                "p_endpoint_max_t_120": float(result["p_endpoint_max_t_120"][index, endpoint_index]),
                "p_global_max_t_240": float(result["p_global_max_t_240"][index, endpoint_index]),
            }
        rows.append(row)
    winners = [int(np.argmax(np.abs(result["rho_adjusted"][:, index]))) for index in range(2)]
    for endpoint_index, (endpoint, winner) in enumerate(zip(endpoints, winners)):
        rows[winner][endpoint]["stratified_bootstrap_quantiles"] = bootstrap_candidate(matrix[:, winner], scores[endpoint], scores, bootstraps, SEED + 10 + endpoint_index)
        criterion_design = np.column_stack([np.ones(57), scores["age"], scores["sex"], scores["cohort"], rankdata(scores["criterion"])])
        x = residualize(rankdata(matrix[:, winner]), criterion_design)
        y = residualize(rankdata(scores[endpoint]), criterion_design)
        rows[winner][endpoint]["partial_rho_controlling_criterion"] = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    return rows, winners


def configure_style() -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42})


def scatter_panel(axis: plt.Axes, x: np.ndarray, y: np.ndarray, cohort: np.ndarray, xlabel: str, title: str, annotation: str, seed: int) -> None:
    colors = np.asarray(["#7A8DA6", "#D07A55"])[cohort.astype(int)]
    jitter = np.random.default_rng(seed).uniform(-0.008, 0.008, len(x)) * max(float(np.ptp(x)), 1.0)
    axis.scatter(x + jitter, y, c=colors, s=19, alpha=0.84, edgecolor="white", linewidth=0.35)
    order = np.argsort(x)
    coefficient = np.polyfit(x, y, 1)
    axis.plot(x[order], np.polyval(coefficient, x[order]), color="#465563", ls="--", lw=0.9)
    axis.set_xlabel(xlabel); axis.set_ylabel("Coalition synergy (bits)")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.text(0.02, 0.98, annotation, transform=axis.transAxes, va="top")


def plot(names: np.ndarray, sizes: np.ndarray, matrix: np.ndarray, scores: Mapping[str, np.ndarray], result: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], winners: Sequence[int]) -> None:
    configure_style()
    figure = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=[1.06, 1])
    axes = [figure.add_subplot(grid[i, j]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
    axes[0].scatter(scores["balanced_accuracy"], scores["dprime"], c=np.asarray(["#7A8DA6", "#D07A55"])[scores["cohort"].astype(int)], s=20, alpha=0.84, edgecolor="white", linewidth=0.35)
    axes[0].set(xlabel="Balanced accuracy (%)", ylabel=r"Corrected sensitivity $d'$", title="")
    axes[0].set_title("a  Composite behavioral endpoints", loc="left", fontweight="bold")
    axes[0].text(0.02, 0.98, rf"Spearman $\rho$={spearmanr(scores['balanced_accuracy'], scores['dprime']).statistic:+.3f}" + "\n$n$=57", transform=axes[0].transAxes, va="top")

    cloud = axes[1].scatter(result["rho_adjusted"][:, 1], result["rho_adjusted"][:, 0], c=sizes, cmap="viridis", s=24, alpha=0.78, edgecolor="white", linewidth=0.3)
    axes[1].axhline(0, color="#D2D6DC", lw=0.6); axes[1].axvline(0, color="#D2D6DC", lw=0.6)
    axes[1].set(xlabel=r"Association with $d'$ ($\rho$)", ylabel=r"Association with balanced accuracy ($\rho$)")
    axes[1].set_title("b  All 120 SOCIAL coalitions", loc="left", fontweight="bold")
    if winners[0] == winners[1]:
        winner = winners[0]
        axes[1].scatter(result["rho_adjusted"][winner, 1], result["rho_adjusted"][winner, 0], s=82, marker="D", facecolor="none", edgecolor="#335C81", lw=1.6)
        axes[1].annotate("+".join(SHORT[x] for x in names[winner].split("+")) + " (both)", result["rho_adjusted"][winner, ::-1], xytext=(4, 5), textcoords="offset points", fontsize=6.3, color="#335C81")
    else:
        for winner, marker, color in zip(winners, ("o", "s"), ("#C65D3A", "#335C81")):
            axes[1].scatter(result["rho_adjusted"][winner, 1], result["rho_adjusted"][winner, 0], s=75, marker=marker, facecolor="none", edgecolor=color, lw=1.5)
            axes[1].annotate("+".join(SHORT[x] for x in names[winner].split("+")), result["rho_adjusted"][winner, ::-1], xytext=(4, 5), textcoords="offset points", fontsize=6.3, color=color)
    colorbar = figure.colorbar(cloud, ax=axes[1], shrink=0.65, pad=0.02); colorbar.set_label("Coalition size")

    endpoint_info = (("balanced_accuracy", "Balanced accuracy (%)", "c", winners[0]), ("dprime", r"Corrected sensitivity $d'$", "d", winners[1]))
    for axis, (endpoint, xlabel, letter, winner) in zip(axes[2:], endpoint_info):
        item = rows[winner][endpoint]
        annotation = rf"adjusted $\rho$={item['rho_adjusted']:+.3f}" + f"\n120-coalition max-T $p$={item['p_endpoint_max_t_120']:.4f}\n240-test max-T $p$={item['p_global_max_t_240']:.4f}\n$n$=57"
        scatter_panel(axis, scores[endpoint], matrix[:, winner], scores["cohort"], xlabel, f"{letter}  {rows[winner]['short_coalition']}", annotation, SEED + winner)
    handles = [mpl.lines.Line2D([], [], marker="o", ls="none", color=color, label=label, markersize=4) for label, color in (("Original 29", "#7A8DA6"), ("Supplement 28", "#D07A55"))]
    axes[3].legend(handles=handles, loc="lower right")
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        figure.savefig(OUTPUT / f"social_composite_coalition_screen_57.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    subjects = load_subjects(); combos = coalitions(); names = coalition_names(combos); sizes = np.asarray([len(x) for x in combos])
    matrix, heldout, explained = compute_matrix(subjects, combos, args.recompute)
    scores = load_scores(subjects)
    result = screen(matrix, scores, args.permutations, SEED)
    rows, winners = make_rows(names, sizes, matrix, scores, result, args.bootstraps)
    winner_index = winners[1]
    winner_brain = matrix[:, winner_index]
    base_design = np.column_stack([np.ones(57), scores["age"], scores["sex"], scores["cohort"]])

    def adjusted_rho(endpoint: np.ndarray, extra: Sequence[np.ndarray] = ()) -> float:
        design = np.column_stack([base_design, *extra]) if extra else base_design
        x = residualize(rankdata(winner_brain), design)
        y = residualize(rankdata(endpoint), design)
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

    fixed_trials = 5
    fixed_hit = (np.rint(scores["hit_percent"] * fixed_trials / 100.0) + 0.5) / (fixed_trials + 1.0)
    fixed_false_alarm = (np.rint((100.0 - scores["correct_reject_percent"]) * fixed_trials / 100.0) + 0.5) / (fixed_trials + 1.0)
    fixed_n_dprime = norm.ppf(fixed_hit) - norm.ppf(fixed_false_alarm)
    standard_mask = scores["effective_trials"] == 5
    standard_design = np.column_stack([np.ones(int(standard_mask.sum())), scores["age"][standard_mask], scores["sex"][standard_mask], scores["cohort"][standard_mask]])
    standard_x = residualize(rankdata(winner_brain[standard_mask]), standard_design)
    standard_y = residualize(rankdata(scores["dprime"][standard_mask]), standard_design)
    winner_diagnostics = {
        "adjusted_rho_tom_hit": adjusted_rho(scores["hit_percent"]),
        "adjusted_rho_random_correct_rejection": adjusted_rho(scores["correct_reject_percent"]),
        "adjusted_rho_criterion": adjusted_rho(scores["criterion"]),
        "dprime_partial_rho_controlling_criterion": adjusted_rho(scores["dprime"], [rankdata(scores["criterion"])]),
        "dprime_partial_rho_controlling_criterion_and_effective_trials": adjusted_rho(scores["dprime"], [rankdata(scores["criterion"]), rankdata(scores["effective_trials"])]),
        "dprime_rho_fixed_n5_correction": adjusted_rho(fixed_n_dprime),
        "dprime_rho_standard_n5_subjects_only": float(standard_x @ standard_y / (np.linalg.norm(standard_x) * np.linalg.norm(standard_y))),
        "standard_n5_subjects": int(standard_mask.sum()),
    }
    summary = {
        "experiment": "Expanded SOCIAL coalition screen for balanced accuracy and corrected d-prime",
        "subjects": 57,
        "score_definition": {
            "balanced_accuracy": "0.5 * (TOM hit percentage + Random correct-rejection percentage)",
            "dprime": "Phi^-1((hits+0.5)/(n+1)) - Phi^-1((false_alarms+0.5)/(n+1))",
            "criterion": "-0.5 * [Phi^-1(corrected_hit_rate) + Phi^-1(corrected_false_alarm_rate)]",
            "effective_trials": "Smallest multiple of five matching condition-specific and overall percentages; minimum five by HCP design.",
        },
        "trial_count_distribution": {str(value): int(np.sum(scores["effective_trials"] == value)) for value in np.unique(scores["effective_trials"])},
        "behavior": {key: {"minimum": float(np.min(scores[key])), "maximum": float(np.max(scores[key])), "mean": float(np.mean(scores[key])), "sd": float(np.std(scores[key], ddof=1)), "unique_values": int(len(np.unique(scores[key])))} for key in ("balanced_accuracy", "dprime", "criterion")},
        "behavior_spearman_balanced_vs_dprime": float(spearmanr(scores["balanced_accuracy"], scores["dprime"]).statistic),
        "inference": {"permutations": args.permutations, "bootstraps": args.bootstraps, "covariates": ["age", "sex", "cohort"], "permutation": "Freedman-Lane residual permutation within original/supplement cohort", "families": ["120 coalitions within endpoint", "240 coalition-endpoint tests globally"]},
        "model": {"parcellation": "Schaefer-1000 Yeo7", "state": "full SOCIAL LR run", "history_order": ORDER, "ridge_alpha": ALPHA, "mean_heldout_skill_ratio": float(np.mean(heldout)), "models_better_than_persistence": int(np.sum(heldout < 1)), "mean_pc1_explained": float(np.mean(explained)), "minimum_synergy_bits": float(np.min(matrix)), "negative_within_tolerance_count": int(np.sum((matrix < 0) & (matrix >= -SYN_TOLERANCE_BITS)))},
        "winners": {"balanced_accuracy": rows[winners[0]], "dprime": rows[winners[1]]},
        "winner_diagnostics": winner_diagnostics,
        "significance_counts": {"raw_p_below_0_05": int(np.sum(result["p_raw"] < 0.05)), "global_q_below_0_05": int(np.sum(result["q_global_240"] < 0.05)), "endpoint_max_t_below_0_05": int(np.sum(result["p_endpoint_max_t_120"] < 0.05)), "global_max_t_below_0_05": int(np.sum(result["p_global_max_t_240"] < 0.05))},
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUTPUT / "all_associations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    plot(names, sizes, matrix, scores, result, rows, winners)
    lines = ["# SOCIAL composite-score coalition screen", "", "![Screen](social_composite_coalition_screen_57.png)", "", "| Endpoint | Top coalition | Adjusted rho | Original 29 rho | Supplement 28 rho | Raw p | Endpoint max-T p | Global max-T p |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for endpoint in ("balanced_accuracy", "dprime"):
        winner = summary["winners"][endpoint]; value = winner[endpoint]
        lines.append(f"| {endpoint} | {winner['short_coalition']} | {value['rho_adjusted']:+.3f} | {value['rho_original_29']:+.3f} | {value['rho_supplement_28']:+.3f} | {value['p_raw']:.5f} | {value['p_endpoint_max_t_120']:.5f} | {value['p_global_max_t_240']:.5f} |")
    lines += [
        "",
        "Adjusted rho residualizes age, sex, and cohort in rank space. Permutations preserve the original/supplement cohort blocks. The screen is exploratory; selection and effect estimation use the same 57 subjects.",
        "",
        f"The winning d-prime association is driven mainly by Random correct rejection (adjusted rho={winner_diagnostics['adjusted_rho_random_correct_rejection']:+.3f}) rather than TOM hits (rho={winner_diagnostics['adjusted_rho_tom_hit']:+.3f}). It remains {winner_diagnostics['dprime_partial_rho_controlling_criterion']:+.3f} after controlling response criterion.",
        "",
        f"Sensitivity checks: fixed-n=5 d-prime rho={winner_diagnostics['dprime_rho_fixed_n5_correction']:+.3f}; standard-n=5 subjects only (n={winner_diagnostics['standard_n5_subjects']}) rho={winner_diagnostics['dprime_rho_standard_n5_subjects_only']:+.3f}.",
        "",
        "The neural metric comes from the full SOCIAL LR run and is not TOM-block-specific.", "",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
