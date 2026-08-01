#!/usr/bin/env python3
"""Screen all LANGUAGE Yeo-7 coalitions against corrected Story/Math difficulty.

The costly 57 x 120 Schaefer-1000 synergy matrix is cached incrementally.  The
behavioral rows are permuted jointly within the original/supplementary cohorts.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_language_story_math_synergy import holm_adjust, williams_test
from scripts.analyze_hcp_schaefer500_yeo7_network_attribution import NETWORK_ORDER
from scripts.analyze_hcp_task_evoked_pc2_xi_hierarchy import network_module_indices
from scripts.phi_hierarchy import subset_phi_raw
from scripts.run_hcp_schaefer500_yeo7_module_phi_decomposition import module_ei_table
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import load_yeo7_groups
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi
from scripts.tune_hcp_task_evoked_xi_hierarchy import prepare_projection


TASK_ROOT = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_57_brain"
LABELS = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30" / "_atlas_labels" / "Schaefer2018_1000Parcels_7Networks_order.txt"
BEHAVIOR = ROOT / "data" / "unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
SUBJECT_SOURCE = ROOT / "results" / "hcp_schaefer1000_panels_e_i_57" / "panel_values_57.npz"
OLD_METRICS = ROOT / "results" / "hcp_cognition_exhaustive_targeted_greedy" / "metrics.npz"
OUTPUT = ROOT / "results" / "hcp_language_story_math_coalitions_57"
LOG = ROOT / "docs" / "log" / "hcp_language_story_math_coalition_progress.json"
CACHE = OUTPUT / "language_coalition_synergy_57.npz"
PARTIAL = OUTPUT / "language_coalition_synergy_57.partial.npz"

SEED = 20260801
PERMUTATIONS = 100_000
BOOTSTRAPS = 20_000
SYN_TOLERANCE_BITS = 1.0e-9
ORDER = 3
ALPHA = 1.0
SHORT = {"Vis": "Vis", "SomMot": "Som", "DorsAttn": "DAN", "SalVentAttn": "VAN", "Limbic": "Lim", "Cont": "Cont", "Default": "DMN"}


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


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


def status(phase: str, completed: int, total: int, message: str, *, state: str = "running") -> None:
    atomic_json(LOG, {"state": state, "phase": phase, "completed": completed, "total": total, "message": message, "updated_unix": time.time()})


def coalition_tuples() -> tuple[tuple[str, ...], ...]:
    return tuple(combo for size in range(2, 8) for combo in itertools.combinations(NETWORK_ORDER, size))


def coalition_names(coalitions: Sequence[Sequence[str]]) -> np.ndarray:
    return np.asarray(["+".join(value) for value in coalitions])


def load_subjects_and_coalitions() -> tuple[np.ndarray, tuple[tuple[str, ...], ...]]:
    with np.load(SUBJECT_SOURCE, allow_pickle=False) as archive:
        subjects = archive["subjects"].astype(str)
    if subjects.shape != (57,) or len(set(subjects.tolist())) != 57:
        raise ValueError("Expected exactly 57 unique subjects.")
    coalitions = coalition_tuples()
    with np.load(OLD_METRICS, allow_pickle=False) as archive:
        old_names = archive["coalitions"].astype(str)
        old_subjects = archive["subjects"].astype(str)
    if not np.array_equal(coalition_names(coalitions), old_names):
        raise ValueError("Coalition order differs from the frozen 29-subject screen.")
    if not np.array_equal(subjects[:29], old_subjects):
        raise ValueError("The original 29-subject prefix differs from the frozen screen.")
    return subjects, coalitions


def compute_subject(subject: str, groups: Mapping[str, Sequence[int]], coalitions: Sequence[tuple[str, ...]]) -> tuple[np.ndarray, dict[str, float]]:
    path = TASK_ROOT / subject / "LANGUAGE_LR.mat"
    projections, explained, development_end = prepare_projection(
        path, groups, state="LANGUAGE", max_components=1,
        task_retained_key="Schaefer1000_taskRetained",
        task_regressed_key="Schaefer1000_taskRegressed", expected_parcels=1000,
    )
    fitted = fit_delta_history_phi(projections[1], alpha=ALPHA, order=ORDER, development_end=development_end)
    indices = network_module_indices(NETWORK_ORDER, n_components=1, order=ORDER)
    table = module_ei_table(fitted["transition"], fitted["noise_covariance"], indices, ridge=1.0e-6)
    singleton = {name: float(table[(name,)]) for name in NETWORK_ORDER}
    values = np.asarray([subset_phi_raw(c, table, singleton) for c in coalitions], dtype=float)
    diagnostics = {
        "development_end": int(development_end),
        "heldout_skill_ratio": float(fitted["heldout"]["skill_ratio"]),
        "mean_pc1_explained": float(np.mean(list(explained[1].values()))),
    }
    return values, diagnostics


def compute_matrix(subjects: np.ndarray, coalitions: Sequence[tuple[str, ...]], *, recompute: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = coalition_names(coalitions)
    if CACHE.is_file() and not recompute:
        with np.load(CACHE, allow_pickle=False) as archive:
            if np.array_equal(archive["subjects"].astype(str), subjects) and np.array_equal(archive["coalitions"].astype(str), names):
                values = archive["synergy_bits"].astype(float)
                if values.shape == (57, 120) and np.isfinite(values).all():
                    return values, archive["heldout_skill_ratio"].astype(float), archive["mean_pc1_explained"].astype(float)

    values = np.full((57, 120), np.nan)
    heldout = np.full(57, np.nan)
    explained = np.full(57, np.nan)
    if PARTIAL.is_file() and not recompute:
        with np.load(PARTIAL, allow_pickle=False) as archive:
            if np.array_equal(archive["subjects"].astype(str), subjects) and np.array_equal(archive["coalitions"].astype(str), names):
                values = archive["synergy_bits"].astype(float)
                heldout = archive["heldout_skill_ratio"].astype(float)
                explained = archive["mean_pc1_explained"].astype(float)

    groups = load_yeo7_groups(LABELS, expected_parcels=1000)
    pending = np.flatnonzero(~np.isfinite(values).all(axis=1))
    progress = tqdm(pending, desc="LANGUAGE coalition EI", unit="subject")
    for count, index in enumerate(progress, start=1):
        subject = str(subjects[index])
        row, diagnostics = compute_subject(subject, groups, coalitions)
        values[index] = row
        heldout[index] = diagnostics["heldout_skill_ratio"]
        explained[index] = diagnostics["mean_pc1_explained"]
        atomic_npz(PARTIAL, subjects=subjects, coalitions=names, synergy_bits=values, heldout_skill_ratio=heldout, mean_pc1_explained=explained)
        done = int(np.isfinite(values).all(axis=1).sum())
        status("metric_recomputation", done, 57, f"Completed {subject}; {len(pending)-count} subjects pending")
    if not np.isfinite(values).all():
        raise ValueError("Metric cache remains incomplete after recomputation.")
    atomic_npz(CACHE, subjects=subjects, coalitions=names, coalition_sizes=np.asarray([len(c) for c in coalitions]), synergy_bits=values, heldout_skill_ratio=heldout, mean_pc1_explained=explained, syn_tolerance_bits=np.asarray(SYN_TOLERANCE_BITS))
    if PARTIAL.exists():
        PARTIAL.unlink()
    return values, heldout, explained


def normalized_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    ranks = rankdata(array, axis=0, method="average")
    ranks -= ranks.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.sum(ranks**2, axis=0, keepdims=True))
    if np.any(norms <= 1e-12):
        raise ValueError("A correlation variable is constant.")
    return ranks / norms


def load_behavior(subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    behavior = pd.read_csv(BEHAVIOR, dtype={"Subject": str})
    behavior["Subject"] = behavior["Subject"].str.removeprefix("sub-")
    ordered = behavior.set_index("Subject").loc[[s.removeprefix("sub-") for s in subjects]].copy()
    # Confirmed HCP S1200 derived-CSV swap: the corrected scientific aliases are intentional.
    story = ordered["Language_Task_Math_Avg_Difficulty_Level"].to_numpy(float)
    math = ordered["Language_Task_Story_Avg_Difficulty_Level"].to_numpy(float)
    if not np.isfinite(np.column_stack([story, math])).all():
        raise ValueError("Corrected Story/Math difficulty contains non-finite values.")
    return story, math, ordered


def blocked_max_t(brain: np.ndarray, endpoints: np.ndarray, cohorts: np.ndarray, *, permutations: int, seed: int) -> dict[str, np.ndarray]:
    brain_ranks = normalized_ranks(brain)
    endpoint_ranks = normalized_ranks(endpoints)
    observed = brain_ranks.T @ endpoint_ranks
    raw_counts = np.zeros_like(observed, dtype=np.int64)
    endpoint_counts = np.zeros_like(observed, dtype=np.int64)
    global_counts = np.zeros_like(observed, dtype=np.int64)
    cohort_indices = [np.flatnonzero(cohorts == value) for value in np.unique(cohorts)]
    rng = np.random.default_rng(seed)
    chunk = 500
    iterator = tqdm(range(0, permutations, chunk), desc="Cohort-blocked max-T", unit="chunk")
    for start in iterator:
        size = min(chunk, permutations - start)
        perm = np.tile(np.arange(len(cohorts)), (size, 1))
        for indices in cohort_indices:
            order = np.argsort(rng.random((size, len(indices))), axis=1)
            perm[:, indices] = indices[order]
        null = np.einsum("nc,pne->pce", brain_ranks, endpoint_ranks[perm], optimize=True)
        absolute = np.abs(null)
        raw_counts += np.sum(absolute >= np.abs(observed)[None, :, :], axis=0)
        endpoint_max = absolute.max(axis=1)
        global_max = absolute.max(axis=(1, 2))
        endpoint_counts += np.sum(endpoint_max[:, None, :] >= np.abs(observed)[None, :, :], axis=0)
        global_counts += np.sum(global_max[:, None, None] >= np.abs(observed)[None, :, :], axis=0)
        status("permutation_screen", min(start + size, permutations), permutations, "Joint Story/Math rows permuted within recruitment cohort")
    return {
        "rho": observed,
        "p_raw": (raw_counts + 1) / (permutations + 1),
        "p_endpoint_max_t": (endpoint_counts + 1) / (permutations + 1),
        "p_global_max_t": (global_counts + 1) / (permutations + 1),
    }


def partial_spearman(brain: np.ndarray, endpoint: np.ndarray, other: np.ndarray) -> float:
    x, y, z = rankdata(brain), rankdata(endpoint), rankdata(other)
    design = np.column_stack([np.ones(len(z)), z])
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_candidate(brain: np.ndarray, story: np.ndarray, math: np.ndarray, cohorts: np.ndarray, *, bootstraps: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(cohorts == value) for value in np.unique(cohorts)]
    output = np.empty((bootstraps, 3), dtype=float)
    chunk = 1000
    for start in range(0, bootstraps, chunk):
        size = min(chunk, bootstraps - start)
        sampled = np.concatenate([rng.choice(group, size=(size, len(group)), replace=True) for group in groups], axis=1)
        bx = rankdata(brain[sampled], axis=1)
        sy = rankdata(story[sampled], axis=1)
        my = rankdata(math[sampled], axis=1)
        bx -= bx.mean(axis=1, keepdims=True)
        sy -= sy.mean(axis=1, keepdims=True)
        my -= my.mean(axis=1, keepdims=True)
        rs = np.sum(bx * sy, axis=1) / np.sqrt(np.sum(bx**2, axis=1) * np.sum(sy**2, axis=1))
        rm = np.sum(bx * my, axis=1) / np.sqrt(np.sum(bx**2, axis=1) * np.sum(my**2, axis=1))
        output[start:start + size] = np.column_stack([rs, rm, rs - rm])
    quantiles = np.quantile(output, [0.025, 0.5, 0.975], axis=0)
    return {"story_ci95": quantiles[[0, 2], 0].tolist(), "math_ci95": quantiles[[0, 2], 1].tolist(), "delta_story_minus_math_ci95": quantiles[[0, 2], 2].tolist(), "delta_median": float(quantiles[1, 2])}


def candidate_diagnostics(index: int, name: str, brain: np.ndarray, story: np.ndarray, math: np.ndarray, cohorts: np.ndarray, *, seed: int) -> dict[str, Any]:
    rho_story = float(spearmanr(brain, story).statistic)
    rho_math = float(spearmanr(brain, math).statistic)
    score_rho = float(spearmanr(story, math).statistic)
    statistic, p_value = williams_test(score_rho, rho_story, rho_math, len(brain))
    loo = []
    for leave in range(len(brain)):
        keep = np.arange(len(brain)) != leave
        rs = float(spearmanr(brain[keep], story[keep]).statistic)
        rm = float(spearmanr(brain[keep], math[keep]).statistic)
        loo.append((rs, rm, rs - rm))
    loo_array = np.asarray(loo)
    return {
        "index": index, "coalition": name, "rho_story": rho_story, "rho_math": rho_math,
        "delta_story_minus_math": rho_story - rho_math,
        "williams_t": statistic, "williams_p": p_value,
        "partial_story_controlling_math": partial_spearman(brain, story, math),
        "partial_math_controlling_story": partial_spearman(brain, math, story),
        "leave_one_out_story_range": [float(loo_array[:, 0].min()), float(loo_array[:, 0].max())],
        "leave_one_out_math_range": [float(loo_array[:, 1].min()), float(loo_array[:, 1].max())],
        "leave_one_out_delta_range": [float(loo_array[:, 2].min()), float(loo_array[:, 2].max())],
        **bootstrap_candidate(brain, story, math, cohorts, bootstraps=BOOTSTRAPS, seed=seed),
    }


def configure_style() -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "svg.fonttype": "none", "pdf.fonttype": 42})


def scatter_panel(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str, xlabel: str, title: str, rho: float, p: float) -> None:
    ax.scatter(x, y, s=18, color=color, alpha=0.82, edgecolor="white", linewidth=0.35)
    order = np.argsort(x)
    coefficient = np.polyfit(x, y, 1)
    ax.plot(x[order], np.polyval(coefficient, x[order]), color="#384454", lw=1.0, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Coalition synergy (bits)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.text(
        0.02,
        0.98,
        f"Spearman $\\rho$={rho:+.3f}\nendpoint max-T $p$={p:.3g}",
        transform=ax.transAxes,
        va="top",
    )


def make_figure(names: np.ndarray, sizes: np.ndarray, synergy: np.ndarray, story: np.ndarray, math: np.ndarray, screen: Mapping[str, np.ndarray], selected: Sequence[int]) -> None:
    configure_style()
    story_color, math_color = "#C96B4B", "#4C78A8"
    fig = plt.figure(figsize=(7.2, 5.35), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.32, 1, 1])
    ax0 = fig.add_subplot(grid[:, 0])
    scatter = ax0.scatter(screen["rho"][:, 1], screen["rho"][:, 0], c=sizes, cmap="viridis", s=24, alpha=0.78, edgecolor="white", linewidth=0.3)
    limits = [-0.55, 0.65]
    ax0.plot(limits, limits, color="#7B8490", lw=0.8, ls="--")
    ax0.axhline(0, color="#D2D6DC", lw=0.6); ax0.axvline(0, color="#D2D6DC", lw=0.6)
    ax0.set(xlim=limits, ylim=limits, xlabel=r"Association with corrected Math difficulty ($\rho$)", ylabel=r"Association with corrected Story difficulty ($\rho$)")
    ax0.set_title("a  All 120 LANGUAGE coalitions", loc="left", fontweight="bold")
    cbar = fig.colorbar(scatter, ax=ax0, location="bottom", shrink=0.72, pad=0.08)
    cbar.set_label("Coalition size (networks)")
    for idx, marker, color in zip(selected, ["o", "s"], [story_color, math_color]):
        ax0.scatter(screen["rho"][idx, 1], screen["rho"][idx, 0], s=70, facecolor="none", edgecolor=color, marker=marker, lw=1.6, zorder=5)
        label = "+".join(SHORT[x] for x in names[idx].split("+"))
        ax0.annotate(label, (screen["rho"][idx, 1], screen["rho"][idx, 0]), xytext=(4, 5), textcoords="offset points", color=color, fontsize=6.5)

    panels = [(selected[0], story, story_color, "Corrected Story difficulty", "b  Shared endpoint winner", 0), (selected[0], math, math_color, "Corrected Math difficulty", "c  Same coalition, Math", 1), (selected[1], story, story_color, "Corrected Story difficulty", "d  Distinct Math runner-up", 0), (selected[1], math, math_color, "Corrected Math difficulty", "e  Distinct Math runner-up", 1)]
    axes = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[1, 2])]
    for ax, (idx, endpoint, color, xlabel, prefix, endpoint_index) in zip(axes, panels):
        short = "+".join(SHORT[x] for x in names[idx].split("+"))
        scatter_panel(ax, endpoint, synergy[:, idx], color, xlabel, f"{prefix}: {short}", float(screen["rho"][idx, endpoint_index]), float(screen["p_endpoint_max_t"][idx, endpoint_index]))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 600}), (".svg", {}), (".pdf", {})):
        fig.savefig(OUTPUT / f"story_math_coalition_screen{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def make_math_supplement_figure(
    names: np.ndarray,
    synergy: np.ndarray,
    math: np.ndarray,
    screen: Mapping[str, np.ndarray],
    index: int,
) -> None:
    """Render the descriptive, non-significant Math runner-up for the appendix."""
    configure_style()
    color = "#4C78A8"
    x = np.asarray(math, dtype=float)
    y = np.asarray(synergy[:, index], dtype=float)
    figure, axis = plt.subplots(figsize=(3.5, 3.05), constrained_layout=True)
    axis.scatter(
        x,
        y,
        s=24,
        color=color,
        alpha=0.84,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    order = np.argsort(x)
    slope, intercept = np.polyfit(x, y, 1)
    axis.plot(
        x[order],
        slope * x[order] + intercept,
        color="#384454",
        linewidth=1.0,
        linestyle="--",
        zorder=2,
    )
    axis.set(
        xlabel="Corrected Math average difficulty",
        ylabel="DorsAttn+Control synergy (bits)",
    )
    axis.text(
        0.03,
        0.97,
        f"Spearman $\\rho$={float(screen['rho'][index, 1]):+.3f}\n"
        f"cohort-blocked $p$={float(screen['p_raw'][index, 1]):.4f}\n"
        f"120-coalition max-T $p$={float(screen['p_endpoint_max_t'][index, 1]):.3f}\n"
        "n=57",
        transform=axis.transAxes,
        ha="left",
        va="top",
    )
    axis.grid(color="#E8EBED", linewidth=0.5, zorder=0)
    for suffix, kwargs in ((".png", {"dpi": 600}), (".svg", {}), (".pdf", {})):
        figure.savefig(
            OUTPUT / f"supplement_dorsattn_cont_math_difficulty{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(figure)


def write_outputs(subjects: np.ndarray, names: np.ndarray, sizes: np.ndarray, synergy: np.ndarray, story: np.ndarray, math: np.ndarray, cohorts: np.ndarray, heldout: np.ndarray, explained: np.ndarray, screen: Mapping[str, np.ndarray], *, permutations: int) -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    endpoint_names = ("corrected_story_difficulty", "corrected_math_difficulty")
    rows = []
    for index, name in enumerate(names):
        row: dict[str, Any] = {"index": index, "coalition": str(name), "coalition_size": int(sizes[index])}
        for endpoint_index, endpoint in enumerate(endpoint_names):
            row[endpoint] = {key: float(screen[key][index, endpoint_index]) for key in ("rho", "p_raw", "p_endpoint_max_t", "p_global_max_t")}
        rows.append(row)
    with (OUTPUT / "all_associations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    selected = [int(np.argmax(screen["rho"][:, 0])), int(np.argmax(screen["rho"][:, 1]))]
    math_runner_up = next(
        int(index)
        for index in np.argsort(screen["rho"][:, 1])[::-1]
        if int(index) != selected[0]
    )
    unique = list(dict.fromkeys(selected))
    diagnostics = [candidate_diagnostics(index, str(names[index]), synergy[:, index], story, math, cohorts, seed=SEED + 100 + position) for position, index in enumerate(unique)]
    adjusted = holm_adjust([row["williams_p"] for row in diagnostics])
    for row, value in zip(diagnostics, adjusted):
        row["williams_p_holm_selected_winners"] = float(value)
    runner_up_diagnostic = candidate_diagnostics(
        math_runner_up,
        str(names[math_runner_up]),
        synergy[:, math_runner_up],
        story,
        math,
        cohorts,
        seed=SEED + 500,
    )
    runner_up_diagnostic["role"] = (
        "descriptive best Math-associated coalition distinct from the shared winner; "
        "not a replacement confirmatory winner"
    )
    selected_frame = pd.DataFrame({"subject": subjects, "cohort": np.where(cohorts == 0, "original_29", "supplementary_28"), "corrected_story_difficulty": story, "corrected_math_difficulty": math})
    for index in list(dict.fromkeys([*unique, math_runner_up])):
        selected_frame[f"synergy_bits__{names[index]}"] = synergy[:, index]
    selected_frame.to_csv(OUTPUT / "selected_candidate_source_data.tsv", sep="\t", index=False)

    known_name = "SomMot+Limbic+Cont"
    known_index = names.tolist().index(known_name)
    known_story = float(screen["rho"][known_index, 0])
    if abs(known_story - 0.348960974) > 1e-6:
        raise ValueError(f"Known corrected-Story association failed validation: {known_story}")
    summary = {
        "exploratory": True, "n_subjects": 57, "cohorts": {"original": 29, "supplementary": 28},
        "correction": {"corrected_story_difficulty_source_column": "Language_Task_Math_Avg_Difficulty_Level", "corrected_math_difficulty_source_column": "Language_Task_Story_Avg_Difficulty_Level"},
        "model": {"atlas": "Schaefer1000/Yeo7", "state": "LANGUAGE", "network_pc": 1, "history_order": ORDER, "ridge_alpha": ALPHA, "coalitions": 120},
        "multiplicity": {"permutations": permutations, "scheme": "joint endpoint-row permutation within recruitment cohort", "endpoint_family": 120, "global_family": 240},
        "syn_nonnegativity": {"tolerance_bits": SYN_TOLERANCE_BITS, "numerical_zero_count": int(np.count_nonzero((synergy < 0) & (synergy >= -SYN_TOLERANCE_BITS))), "minimum_bits": float(synergy.min())},
        "quality": {"heldout_skill_ratio_range": [float(heldout.min()), float(heldout.max())], "mean_pc1_explained_range": [float(explained.min()), float(explained.max())]},
        "corrected_story_math_spearman": float(spearmanr(story, math).statistic),
        "selected_indices": {"story": selected[0], "math": selected[1]}, "selected_candidates": diagnostics,
        "best_distinct_math_runner_up": {
            **runner_up_diagnostic,
            "math_p_raw_blocked": float(screen["p_raw"][math_runner_up, 1]),
            "math_p_endpoint_max_t": float(screen["p_endpoint_max_t"][math_runner_up, 1]),
            "math_p_global_max_t": float(screen["p_global_max_t"][math_runner_up, 1]),
        },
        "known_story_coalition_validation": {"coalition": known_name, "rho_corrected_story": known_story, "rho_corrected_math": float(screen["rho"][known_index, 1]), "p_story_endpoint_max_t": float(screen["p_endpoint_max_t"][known_index, 0])},
    }
    atomic_json(OUTPUT / "summary.json", summary)
    make_figure(names, sizes, synergy, story, math, screen, [selected[0], math_runner_up])
    make_math_supplement_figure(
        names, synergy, math, screen, math_runner_up
    )
    notes = ["# Figure notes", "", "## Intended conclusion", "", "This exploratory screen compares all 120 non-singleton Yeo-7 network coalitions in the LANGUAGE state against corrected Story and Math adaptive difficulty in the pooled 57-subject sample.", "", "## Multiplicity and interpretation", "", "Endpoint-wise max-T controls selection across 120 coalitions; global max-T additionally controls both endpoints (240 tests). A coalition is endpoint-specific only if its Story-minus-Math correlation contrast is supported by the selected-winner diagnostic.", "", "## Selected coalitions", ""]
    notes.extend([f"- {row['coalition']}: Story rho={row['rho_story']:+.3f}, Math rho={row['rho_math']:+.3f}, Williams Holm p={row['williams_p_holm_selected_winners']:.4g}." for row in diagnostics])
    notes.append(
        f"- Descriptive distinct Math runner-up {runner_up_diagnostic['coalition']}: "
        f"Story rho={runner_up_diagnostic['rho_story']:+.3f}, "
        f"Math rho={runner_up_diagnostic['rho_math']:+.3f}; its Math association does "
        "not survive the 120-coalition max-T family."
    )
    (OUTPUT / "figure_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Compute one subject only and print timing.")
    parser.add_argument("--recompute", action="store_true", help="Ignore existing metric caches.")
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    args = parser.parse_args()
    subjects, coalitions = load_subjects_and_coalitions()
    if args.smoke:
        start = time.perf_counter()
        groups = load_yeo7_groups(LABELS, expected_parcels=1000)
        row, diagnostics = compute_subject(str(subjects[0]), groups, coalitions)
        violations = row < -SYN_TOLERANCE_BITS
        if np.any(violations):
            raise ValueError(f"Syn nonnegativity violation: min={row.min():.12g}, threshold={-SYN_TOLERANCE_BITS:.12g}, count={violations.sum()}")
        print(json.dumps({"subject": str(subjects[0]), "seconds": time.perf_counter() - start, "shape": row.shape, "min_bits": float(row.min()), "max_bits": float(row.max()), **diagnostics}, indent=2))
        return
    try:
        status("metric_recomputation", 0, 57, "Loading or computing 57 x 120 LANGUAGE synergy cache")
        synergy, heldout, explained = compute_matrix(subjects, coalitions, recompute=args.recompute)
        violations = synergy < -SYN_TOLERANCE_BITS
        if np.any(violations):
            raise ValueError(f"Syn nonnegativity violation: min={synergy.min():.12g}, threshold={-SYN_TOLERANCE_BITS:.12g}, count={violations.sum()}")
        numerical_zero = (synergy < 0) & ~violations
        analysis_synergy = synergy.copy()
        analysis_synergy[numerical_zero] = 0.0
        story, math, _ = load_behavior(subjects)
        cohorts = np.r_[np.zeros(29, dtype=int), np.ones(28, dtype=int)]
        screen = blocked_max_t(analysis_synergy, np.column_stack([story, math]), cohorts, permutations=args.permutations, seed=SEED)
        names = coalition_names(coalitions)
        sizes = np.asarray([len(c) for c in coalitions])
        summary = write_outputs(subjects, names, sizes, analysis_synergy, story, math, cohorts, heldout, explained, screen, permutations=args.permutations)
        status("complete", 1, 1, f"Selected Story={names[summary['selected_indices']['story']]}; Math={names[summary['selected_indices']['math']]}", state="complete")
        print(json.dumps(summary, indent=2))
    except Exception as error:
        status("failed", 0, 1, f"{type(error).__name__}: {error}", state="failed")
        raise


if __name__ == "__main__":
    main()
