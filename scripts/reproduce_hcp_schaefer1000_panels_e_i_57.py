#!/usr/bin/env python3
"""Reproduce task-only Schaefer-1000 main-figure panels E–I with 57 subjects."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_schaefer500_yeo7_network_attribution import NETWORK_ORDER
from scripts.analyze_hcp_task_evoked_pc2_xi_hierarchy import (
    decompose_transition,
    network_module_indices,
)
from scripts.analyze_hcp_wm_back_condition_correlations import (
    bootstrap_statistics as wm_bootstrap_statistics,
    leave_one_out as wm_leave_one_out,
    paired_difference_test as wm_paired_difference_test,
    permutation_correlations as wm_permutation_correlations,
)
from scripts.phi_hierarchy import subset_phi_raw
from scripts.run_hcp_schaefer500_yeo7_module_phi_decomposition import module_ei_table
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import load_yeo7_groups
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi
from scripts.tune_hcp_task_evoked_xi_hierarchy import prepare_projection


TASK_ROOT = (
    ROOT
    / "data"
    / "hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_57_brain"
)
NEW_SUBJECTS = (
    ROOT
    / "data"
    / "hcp_task_lr_schaefer500_1000_28_complete_subjects"
    / "subjects_28_complete.txt"
)
LABELS_1000 = (
    ROOT
    / "data"
    / "hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30"
    / "_atlas_labels"
    / "Schaefer2018_1000Parcels_7Networks_order.txt"
)
OLD_MAIN = (
    ROOT
    / "results"
    / "hcp_schaefer1000_task_evoked_xi_replication"
    / "full"
    / "k1_p3_a1"
    / "arrays.npz"
)
OLD_LANGUAGE = (
    ROOT
    / "results"
    / "hcp_language_story_math_candidates_schaefer1000_replication"
    / "fixed_candidates_schaefer1000.npz"
)
OLD_WM = (
    ROOT
    / "results"
    / "hcp_task_score_synergy_schaefer1000_validation"
    / "fixed_candidates_schaefer1000.npz"
)
COGNITION = (
    ROOT / "results" / "hcp_single_group_sem_full_1206" / "factor_scores_all_subjects.csv"
)
BEHAVIOR = ROOT / "data" / "unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
OUTPUT = ROOT / "results" / "hcp_schaefer1000_panels_e_i_57"
CACHE = OUTPUT / "supplementary_28_metrics.npz"

STATES = ("LANGUAGE", "MOTOR", "WM")
LANGUAGE_COALITION = ("SomMot", "Limbic", "Cont")
WM_COALITION = ("Cont", "Default")
ALL_NETWORK_ATOM = "+".join(NETWORK_ORDER)
PERMUTATIONS = 100_000
BOOTSTRAPS = 20_000
SEED = 20260731


def normalized_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=float)
    centered = ranks - ranks.mean()
    norm = float(np.linalg.norm(centered))
    if norm <= 1.0e-12:
        raise ValueError("Cannot correlate a constant vector.")
    return centered / norm


def permutation_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> tuple[float, float]:
    x_rank = normalized_ranks(x)
    y_rank = normalized_ranks(y)
    observed = float(x_rank @ y_rank)
    rng = np.random.default_rng(seed)
    exceed = 0
    chunk_size = 2_000
    for start in range(0, permutations, chunk_size):
        size = min(chunk_size, permutations - start)
        indices = np.argsort(rng.random((size, len(x_rank))), axis=1)
        null = x_rank[indices] @ y_rank
        exceed += int(np.count_nonzero(np.abs(null) >= abs(observed)))
    return observed, float((1 + exceed) / (permutations + 1))


def compute_subject_metrics(
    subject: str,
    groups: dict[str, list[int]],
    module_indices: dict[str, list[int]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for state in STATES:
        path = TASK_ROOT / subject / f"{state}_LR.mat"
        if not path.is_file():
            raise FileNotFoundError(path)
        projections, _, development_end = prepare_projection(
            path,
            groups,
            state=state,
            max_components=1,
            task_retained_key="Schaefer1000_taskRetained",
            task_regressed_key="Schaefer1000_taskRegressed",
            expected_parcels=1000,
        )
        fitted = fit_delta_history_phi(
            projections[1],
            alpha=1.0,
            order=3,
            development_end=development_end,
        )
        if state in {"LANGUAGE", "MOTOR"}:
            decomposition = decompose_transition(
                fitted["transition"],
                fitted["noise_covariance"],
                NETWORK_ORDER,
                n_components=1,
                order=3,
            )
            matching = [
                row
                for row in decomposition["atoms"]
                if "+".join(row["sources"]) == ALL_NETWORK_ATOM
            ]
            if len(matching) != 1:
                raise ValueError(
                    f"{subject} {state}: expected one all-network atom, found {len(matching)}."
                )
            result[f"{state.lower()}_all_network_atom"] = float(matching[0]["value"])
            result[f"{state.lower()}_heldout_skill"] = float(
                fitted["heldout"]["skill_ratio"]
            )
            max_identity_error = max(
                abs(float(value)) for value in decomposition["identity_errors"].values()
            )
            result[f"{state.lower()}_max_identity_error"] = max_identity_error
        if state in {"LANGUAGE", "WM"}:
            table = module_ei_table(
                fitted["transition"],
                fitted["noise_covariance"],
                module_indices,
                ridge=1.0e-6,
            )
            singleton = {name: float(table[(name,)]) for name in NETWORK_ORDER}
            coalition = LANGUAGE_COALITION if state == "LANGUAGE" else WM_COALITION
            result[f"{state.lower()}_fixed_coalition"] = float(
                subset_phi_raw(coalition, table, singleton)
            )
    return result


def load_old_metrics() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    main = np.load(OLD_MAIN)
    subjects = main["subjects"].astype(str)
    states = main["states"].astype(str).tolist()
    atom_names = main["atom_names"].astype(str).tolist()
    atom_index = atom_names.index(ALL_NETWORK_ATOM)

    language = np.load(OLD_LANGUAGE)
    if not np.array_equal(language["subjects"].astype(str), subjects):
        raise ValueError("Old Language and main subject orders differ.")
    language_index = language["coalitions"].astype(str).tolist().index(
        "+".join(LANGUAGE_COALITION)
    )

    wm = np.load(OLD_WM)
    if not np.array_equal(wm["subjects"].astype(str), subjects):
        raise ValueError("Old WM and main subject orders differ.")
    wm_state_index = wm["states"].astype(str).tolist().index("WM")
    if wm["coalitions"].astype(str).tolist()[wm_state_index] != "+".join(WM_COALITION):
        raise ValueError("Unexpected frozen WM coalition.")

    metrics = {
        "language_all_network_atom": main["atom_value"][
            states.index("LANGUAGE"), :, atom_index
        ],
        "motor_all_network_atom": main["atom_value"][
            states.index("MOTOR"), :, atom_index
        ],
        "language_fixed_coalition": language["values"][language_index],
        "wm_fixed_coalition": wm["values"][wm_state_index],
    }
    return subjects, {key: np.asarray(value, dtype=float) for key, value in metrics.items()}


def compute_supplementary(
    subjects: np.ndarray,
    *,
    recompute: bool,
) -> dict[str, np.ndarray]:
    keys = (
        "language_all_network_atom",
        "motor_all_network_atom",
        "language_fixed_coalition",
        "wm_fixed_coalition",
    )
    if CACHE.is_file() and not recompute:
        cache = np.load(CACHE)
        if np.array_equal(cache["subjects"].astype(str), subjects):
            values = {key: np.asarray(cache[key], dtype=float) for key in keys}
            if all(value.shape == (28,) and np.isfinite(value).all() for value in values.values()):
                return values

    groups = load_yeo7_groups(LABELS_1000, expected_parcels=1000)
    module_indices = network_module_indices(NETWORK_ORDER, n_components=1, order=3)
    rows = []
    for subject in tqdm(subjects, desc="Supplementary Schaefer-1000", unit="subject"):
        rows.append(compute_subject_metrics(str(subject), groups, module_indices))

    values = {
        key: np.asarray([row[key] for row in rows], dtype=float)
        for key in keys
    }
    diagnostics = {
        key: np.asarray([row[key] for row in rows], dtype=float)
        for key in (
            "language_heldout_skill",
            "motor_heldout_skill",
            "language_max_identity_error",
            "motor_max_identity_error",
        )
    }
    np.savez_compressed(CACHE, subjects=subjects, **values, **diagnostics)
    return values


def smoke_audit(
    subject: str,
    old_subjects: np.ndarray,
    old_metrics: dict[str, np.ndarray],
) -> dict[str, float]:
    groups = load_yeo7_groups(LABELS_1000, expected_parcels=1000)
    module_indices = network_module_indices(NETWORK_ORDER, n_components=1, order=3)
    recomputed = compute_subject_metrics(subject, groups, module_indices)
    index = old_subjects.astype(str).tolist().index(subject)
    differences = {
        key: abs(float(recomputed[key]) - float(old_metrics[key][index]))
        for key in old_metrics
    }
    if max(differences.values()) > 1.0e-9:
        raise AssertionError(f"Smoke audit failed for {subject}: {differences}")
    return differences


def load_aligned_tables(subjects: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = [subject.removeprefix("sub-") for subject in subjects.astype(str)]
    cognition = pd.read_csv(COGNITION, dtype={"Subject": str}).set_index("Subject")
    behavior = pd.read_csv(BEHAVIOR, dtype={"Subject": str}).set_index("Subject")
    missing_cognition = sorted(set(clean).difference(cognition.index))
    missing_behavior = sorted(set(clean).difference(behavior.index))
    if missing_cognition or missing_behavior:
        raise ValueError(
            f"Missing rows: cognition={missing_cognition}, behavior={missing_behavior}"
        )
    return cognition.loc[clean], behavior.loc[clean]


def association_summary(
    subjects: np.ndarray,
    metrics: dict[str, np.ndarray],
    cognition: pd.DataFrame,
    behavior: pd.DataFrame,
    old_metrics: dict[str, np.ndarray],
) -> dict[str, Any]:
    g_score = cognition["g_score"].to_numpy(dtype=float)
    story = behavior["Language_Task_Story_Acc"].to_numpy(dtype=float)
    math = behavior["Language_Task_Math_Acc"].to_numpy(dtype=float)
    score_2back = behavior["WM_Task_2bk_Acc"].to_numpy(dtype=float)
    score_0back = behavior["WM_Task_0bk_Acc"].to_numpy(dtype=float)

    panels: dict[str, Any] = {}
    for letter, state, x, y_key in (
        ("e", "LANGUAGE", g_score, "language_all_network_atom"),
        ("f", "MOTOR", g_score, "motor_all_network_atom"),
    ):
        association = spearmanr(x, metrics[y_key])
        old_association = spearmanr(g_score[:29], old_metrics[y_key])
        panels[letter] = {
            "state": state,
            "x": "g_score",
            "y": "all-network hierarchy-atom contribution (bits)",
            "n": 57,
            "rho": float(association.statistic),
            "p_two_sided": float(association.pvalue),
            "original_29_rho": float(old_association.statistic),
            "supplementary_28_rho": float(
                spearmanr(x[29:], metrics[y_key][29:]).statistic
            ),
            "direction_preserved": bool(
                np.sign(association.statistic) == np.sign(old_association.statistic)
            ),
        }

    for letter, label, x, seed_offset in (
        ("g", "Story", story, 10),
        ("h", "Math", math, 11),
    ):
        rho, pvalue = permutation_pvalue(
            x,
            metrics["language_fixed_coalition"],
            permutations=PERMUTATIONS,
            seed=SEED + seed_offset,
        )
        old_rho = float(
            spearmanr(x[:29], old_metrics["language_fixed_coalition"]).statistic
        )
        panels[letter] = {
            "state": "LANGUAGE",
            "condition": label,
            "x": f"Language_Task_{label}_Acc",
            "y": "SomMot+Limbic+Cont fixed-coalition synergy (bits)",
            "n": 57,
            "rho": rho,
            "p_pointwise_permutation": pvalue,
            "original_29_rho": old_rho,
            "supplementary_28_rho": float(
                spearmanr(x[29:], metrics["language_fixed_coalition"][29:]).statistic
            ),
            "direction_preserved": bool(np.sign(rho) == np.sign(old_rho)),
        }

    complete = np.isfinite(score_2back) & np.isfinite(score_0back)
    missing = subjects[~complete].astype(str).tolist()
    wm_brain = metrics["wm_fixed_coalition"][complete]
    wm_2back = score_2back[complete]
    wm_0back = score_0back[complete]
    wm_test = wm_permutation_correlations(
        wm_brain,
        (wm_2back, wm_0back),
        design=None,
        rng=np.random.default_rng(SEED),
    )
    wm_delta, wm_delta_p = wm_paired_difference_test(
        wm_brain,
        wm_2back,
        wm_0back,
        design=None,
        rng=np.random.default_rng(SEED + 1),
    )
    wm_bootstrap = wm_bootstrap_statistics(
        wm_brain,
        wm_2back,
        wm_0back,
        np.random.default_rng(SEED + 4),
    )
    wm_loo = wm_leave_one_out(wm_brain, wm_2back, wm_0back)
    old_complete = complete[:29]
    old_0back_rho = float(
        spearmanr(
            old_metrics["wm_fixed_coalition"][old_complete],
            score_0back[:29][old_complete],
        ).statistic
    )
    panels["i"] = {
        "state": "WM",
        "condition": "0-back",
        "x": "WM_Task_0bk_Acc",
        "y": "Cont+Default fixed-coalition synergy (bits)",
        "n": int(complete.sum()),
        "missing_subjects": missing,
        "rho_0back": float(wm_test["rho"][1]),
        "p_permutation_0back": float(wm_test["p_permutation"][1]),
        "p_max_t_0back": float(wm_test["p_max_t"][1]),
        "rho_2back": float(wm_test["rho"][0]),
        "p_permutation_2back": float(wm_test["p_permutation"][0]),
        "p_max_t_2back": float(wm_test["p_max_t"][0]),
        "delta_rho_2back_minus_0back": wm_delta,
        "p_condition_difference": wm_delta_p,
        "original_29_complete_rho_0back": old_0back_rho,
        "supplementary_28_rho_0back": float(
            spearmanr(
                metrics["wm_fixed_coalition"][29:],
                score_0back[29:],
            ).statistic
        ),
        "direction_preserved": bool(
            np.sign(wm_test["rho"][1]) == np.sign(old_0back_rho)
        ),
        **wm_bootstrap,
        **wm_loo,
    }
    return {
        "panels": panels,
        "wm_complete_mask": complete,
        "plot_values": {
            "g_score": g_score,
            "story": story,
            "math": math,
            "score_0back": score_0back,
        },
    }


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def scatter_panel(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    xlabel: str,
    ylabel: str,
    title: str,
    annotation: str,
    jitter: float = 0.0,
    seed: int = 0,
) -> None:
    display_x = np.asarray(x, dtype=float).copy()
    if jitter > 0:
        display_x += np.random.default_rng(seed).uniform(
            -jitter, jitter, size=len(display_x)
        )
    axis.scatter(
        display_x,
        y,
        s=24,
        color=color,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.84,
        zorder=3,
    )
    guide = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    slope, intercept = np.polyfit(x, y, deg=1)
    axis.plot(
        guide,
        slope * guide + intercept,
        color="#4B5563",
        linewidth=1.0,
        linestyle="--",
        zorder=2,
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="right", fontsize=8, fontweight="bold", color="#454545")
    axis.text(
        0.03,
        0.97,
        annotation,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
    )
    axis.grid(color="#E8EBED", linewidth=0.5, zorder=0)


def plot_panels(
    subjects: np.ndarray,
    metrics: dict[str, np.ndarray],
    analysis: dict[str, Any],
) -> None:
    configure_style()
    panels = analysis["panels"]
    values = analysis["plot_values"]
    complete = analysis["wm_complete_mask"]
    figure = plt.figure(figsize=(10.6, 6.0), constrained_layout=True)
    outer = figure.add_gridspec(2, 1, height_ratios=(1.0, 1.0), hspace=0.20)
    top = outer[0].subgridspec(1, 2, wspace=0.17)
    bottom = outer[1].subgridspec(1, 3, wspace=0.24)
    axes = [
        figure.add_subplot(top[0, 0]),
        figure.add_subplot(top[0, 1]),
        figure.add_subplot(bottom[0, 0]),
        figure.add_subplot(bottom[0, 1]),
        figure.add_subplot(bottom[0, 2]),
    ]

    for axis, letter, state, y_key in (
        (axes[0], "e", "LANGUAGE", "language_all_network_atom"),
        (axes[1], "f", "MOTOR", "motor_all_network_atom"),
    ):
        panel = panels[letter]
        scatter_panel(
            axis,
            values["g_score"],
            metrics[y_key],
            color="#B65F3C",
            xlabel="General cognition score",
            ylabel="All-network hierarchy-atom contribution (bits)",
            title=state,
            annotation=(
                rf"Spearman $\rho$ = {panel['rho']:+.3f}"
                f"\nraw two-sided $p$ = {panel['p_two_sided']:.5f}"
                f"\n$n$ = {panel['n']}"
            ),
        )
    shared_top = (
        min(metrics["language_all_network_atom"].min(), metrics["motor_all_network_atom"].min()),
        max(metrics["language_all_network_atom"].max(), metrics["motor_all_network_atom"].max()),
    )
    top_pad = 0.06 * max(shared_top[1] - shared_top[0], 0.1)
    for axis in axes[:2]:
        axis.set_ylim(
            shared_top[0] - top_pad,
            shared_top[1] + 0.28 * max(shared_top[1] - shared_top[0], 0.1),
        )

    for axis, letter, condition, x_key, color, jitter in (
        (axes[2], "g", "Story", "story", "#D66A4E", 0.45),
        (axes[3], "h", "Math", "math", "#4C78A8", 0.20),
    ):
        panel = panels[letter]
        scatter_panel(
            axis,
            values[x_key],
            metrics["language_fixed_coalition"],
            color=color,
            xlabel=f"{condition} accuracy (%)",
            ylabel="Fixed-coalition synergy (bits)",
            title=f"LANGUAGE · Som+Lim+Cont · {condition}",
            annotation=(
                rf"Spearman $\rho$ = {panel['rho']:+.3f}"
                f"\npointwise permutation $p$ = {panel['p_pointwise_permutation']:.4f}"
                f"\n$n$ = {panel['n']}"
            ),
            jitter=jitter,
            seed=SEED + (30 if letter == "g" else 31),
        )
    shared_bottom = (
        float(metrics["language_fixed_coalition"].min()),
        float(metrics["language_fixed_coalition"].max()),
    )
    bottom_pad = 0.07 * max(shared_bottom[1] - shared_bottom[0], 0.1)
    for axis in axes[2:4]:
        axis.set_ylim(
            shared_bottom[0] - bottom_pad,
            shared_bottom[1]
            + 0.24 * max(shared_bottom[1] - shared_bottom[0], 0.1),
        )

    panel_i = panels["i"]
    scatter_panel(
        axes[4],
        values["score_0back"][complete],
        metrics["wm_fixed_coalition"][complete],
        color="#8B6BAE",
        xlabel="0-back accuracy (%)",
        ylabel="Fixed-coalition synergy (bits)",
        title="WM · Cont+Default · 0-back",
        annotation=(
            rf"Spearman $\rho$ = {panel_i['rho_0back']:+.3f}"
            f"\ntwo-condition max-T $p$ = {panel_i['p_max_t_0back']:.4f}"
            f"\n$n$ = {panel_i['n']}"
        ),
        jitter=0.24,
        seed=SEED + 32,
    )
    wm_values = metrics["wm_fixed_coalition"][complete]
    wm_span = max(float(np.ptp(wm_values)), 0.1)
    axes[4].set_ylim(
        float(wm_values.min()) - 0.07 * wm_span,
        float(wm_values.max()) + 0.24 * wm_span,
    )

    for letter, axis in zip(("e", "f", "g", "h", "i"), axes, strict=True):
        axis.text(
            -0.13,
            1.04,
            letter,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(
            OUTPUT / f"hcp_schaefer1000_panels_e_i_57.{suffix}",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def write_report(summary: dict[str, Any]) -> None:
    panels = summary["panels"]
    rows = []
    for letter in ("e", "f", "g", "h"):
        panel = panels[letter]
        p_key = (
            "p_two_sided" if letter in {"e", "f"} else "p_pointwise_permutation"
        )
        rows.append(
            f"| {letter.upper()} | {panel.get('state')} "
            f"{panel.get('condition', '')} | {panel['n']} | "
            f"{panel['original_29_rho']:+.3f} | "
            f"{panel['supplementary_28_rho']:+.3f} | {panel['rho']:+.3f} | "
            f"{panel[p_key]:.5f} |"
        )
    panel_i = panels["i"]
    rows.append(
        f"| I | WM 0-back | {panel_i['n']} | "
        f"{panel_i['original_29_complete_rho_0back']:+.3f} | "
        f"{panel_i['supplementary_28_rho_0back']:+.3f} | "
        f"{panel_i['rho_0back']:+.3f} | {panel_i['p_max_t_0back']:.5f} |"
    )
    lines = [
        "# HCP Schaefer-1000 panels E–I: 57-subject extension",
        "",
        "![Panels E–I](hcp_schaefer1000_panels_e_i_57.png)",
        "",
        "| Panel | Association | n | Original 29 rho | Supplement 28 rho | Expanded rho | Displayed p |",
        "|---|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "The 57-subject analysis retains the frozen representation and coalitions from the original figure.",
        "Panels E–F use raw two-sided Spearman p-values; G–H use pointwise rank-permutation p-values;",
        "panel I uses the max-|rho| p-value across the prespecified 0-back and 2-back pair.",
        "Panel I has 56 complete cases because `sub-104012` lacks a 2-back score.",
        "",
        "All five pooled coefficients retain the direction observed in the original 29 subjects, "
        "but none reaches the prespecified two-sided 0.05 threshold in the expanded sample. "
        "Within the supplementary 28 subjects alone, panels F and G retain the original direction, "
        "panel E is near zero, and panels H and I reverse direction, explaining the pooled attenuation.",
        "",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    old_subjects, old_metrics = load_old_metrics()
    smoke = smoke_audit(str(old_subjects[0]), old_subjects, old_metrics)
    if args.smoke:
        print(json.dumps({"subject": str(old_subjects[0]), "absolute_differences": smoke}, indent=2))
        return 0

    new_subjects = np.asarray(
        [
            f"sub-{subject}"
            for subject in NEW_SUBJECTS.read_text(encoding="utf-8").split()
        ]
    )
    if len(new_subjects) != 28 or len(set(new_subjects)) != 28:
        raise ValueError("Expected 28 unique supplementary subjects.")
    if set(old_subjects) & set(new_subjects):
        raise ValueError("Original and supplementary subject sets overlap.")
    new_metrics = compute_supplementary(new_subjects, recompute=args.recompute)
    subjects = np.concatenate([old_subjects, new_subjects])
    metrics = {
        key: np.concatenate([old_metrics[key], new_metrics[key]])
        for key in old_metrics
    }
    if len(subjects) != 57 or any(len(value) != 57 for value in metrics.values()):
        raise ValueError("Incomplete 57-subject merge.")

    cognition, behavior = load_aligned_tables(subjects)
    analysis = association_summary(
        subjects, metrics, cognition, behavior, old_metrics
    )
    summary = {
        "analysis": "Schaefer-1000 task-only panels E-I expanded from 29 to 57 subjects",
        "subjects_total": 57,
        "subjects": subjects.tolist(),
        "configuration": {
            "parcellation": "Schaefer-1000 Yeo7",
            "k": 1,
            "history_order": 3,
            "ridge_alpha": 1.0,
            "direction": "LR",
            "language_fixed_coalition": "+".join(LANGUAGE_COALITION),
            "wm_fixed_coalition": "+".join(WM_COALITION),
        },
        "statistics": {
            "rank_permutations": PERMUTATIONS,
            "bootstraps": BOOTSTRAPS,
            "seed": SEED,
        },
        "smoke_cache_absolute_differences": smoke,
        "panels": analysis["panels"],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        OUTPUT / "panel_values_57.npz",
        subjects=subjects,
        g_score=analysis["plot_values"]["g_score"],
        language_story_accuracy=analysis["plot_values"]["story"],
        language_math_accuracy=analysis["plot_values"]["math"],
        wm_0back_accuracy=analysis["plot_values"]["score_0back"],
        wm_complete_mask=analysis["wm_complete_mask"],
        **metrics,
    )
    plot_panels(subjects, metrics, analysis)
    write_report(summary)
    print(json.dumps({"output": str(OUTPUT), "panels": summary["panels"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
