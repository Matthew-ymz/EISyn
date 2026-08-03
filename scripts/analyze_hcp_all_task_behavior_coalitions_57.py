#!/usr/bin/env python3
"""Pooled-57 behavior screens for all seven HCP task-state coalition matrices."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.screen_hcp_gambling_reward_valuation_57 as gambling
import scripts.screen_hcp_motor_composite_scores_57 as common
import scripts.screen_hcp_social_composite_scores_57 as social


BEHAVIOR = ROOT / "data/unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
OUTPUT = ROOT / "results/hcp_all_task_behavior_coalitions_57"
RELATIONAL_OUTPUT = ROOT / "results/hcp_relational_performance_coalitions_57"
WM_OUTPUT = ROOT / "results/hcp_wm_performance_coalitions_57"
PERMUTATIONS = 100_000
BOOTSTRAPS = 20_000
SEED = 20260803
SYN_TOLERANCE_BITS = 1.0e-9

TASK_ORDER = ("LANGUAGE", "SOCIAL", "EMOTION", "MOTOR", "GAMBLING", "RELATIONAL", "WM")
TASK_LABELS = {
    "LANGUAGE": "Language",
    "SOCIAL": "Social",
    "EMOTION": "Emotion",
    "MOTOR": "Motor",
    "GAMBLING": "Gambling",
    "RELATIONAL": "Relational",
    "WM": "Working memory",
}
CACHE_PATHS = {
    "LANGUAGE": ROOT / "results/hcp_language_story_math_coalitions_57/language_coalition_synergy_57.npz",
    "SOCIAL": ROOT / "results/hcp_social_composite_scores_57/social_coalition_synergy_57.npz",
    "EMOTION": ROOT / "results/hcp_emotion_performance_coalitions_57/emotion_rest_coalition_synergy_57.npz",
    "MOTOR": ROOT / "results/hcp_motor_composite_scores_57/motor_coalition_synergy_57.npz",
    "GAMBLING": ROOT / "results/hcp_gambling_reward_valuation_57/gambling_coalition_synergy_57.npz",
    "RELATIONAL": RELATIONAL_OUTPUT / "relational_coalition_synergy_57.npz",
    "WM": WM_OUTPUT / "wm_coalition_synergy_57.npz",
}
SHORT = common.SHORT


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def compact_name(value: str) -> str:
    return "+".join(SHORT[item] for item in value.split("+"))


def load_table() -> dict[str, dict[str, str]]:
    with BEHAVIOR.open(newline="", encoding="utf-8-sig") as handle:
        return {str(row["Subject"]): row for row in csv.DictReader(handle)}


def behavior_column(
    table: Mapping[str, Mapping[str, str]], subjects: np.ndarray, field: str
) -> np.ndarray:
    values = np.asarray(
        [float(table[str(subject).removeprefix("sub-")][field]) for subject in subjects],
        dtype=float,
    )
    if values.shape != (57,) or not np.isfinite(values).all():
        raise ValueError(f"Expected 57 finite values for {field}.")
    return values


def compute_new_matrices(subjects: np.ndarray, recompute: bool) -> None:
    combinations = common.coalitions()
    for state, output in (("RELATIONAL", RELATIONAL_OUTPUT), ("WM", WM_OUTPUT)):
        output.mkdir(parents=True, exist_ok=True)
        common.CACHE = CACHE_PATHS[state]
        common.PARTIAL_CACHE = output / f"{state.lower()}_coalition_synergy_57.partial.npz"
        common.compute_matrix(subjects, combinations, recompute, state=state)


def load_matrix(state: str, frozen_subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(CACHE_PATHS[state], allow_pickle=False) as archive:
        subjects = archive["subjects"].astype(str)
        names = archive["coalitions"].astype(str)
        sizes = archive["coalition_sizes"].astype(int)
        values = archive["synergy_bits"].astype(float)
        if state == "EMOTION":
            states = archive["states"].astype(str).tolist()
            values = values[states.index("EMOTION")]
    if not np.array_equal(subjects, frozen_subjects):
        if set(subjects.tolist()) != set(frozen_subjects.tolist()):
            raise ValueError(f"Subject set mismatch for {state}.")
        index = {subject: position for position, subject in enumerate(subjects)}
        reorder = np.asarray([index[subject] for subject in frozen_subjects], dtype=int)
        subjects = subjects[reorder]
        values = values[reorder]
    if values.shape != (57, 120) or not np.isfinite(values).all():
        raise ValueError(f"Invalid coalition matrix for {state}: {values.shape}.")
    violations = values < -SYN_TOLERANCE_BITS
    if np.any(violations):
        raise ValueError(
            f"{state} Syn nonnegativity violation: min={values.min():.12g}, "
            f"threshold={-SYN_TOLERANCE_BITS:.1e}, count={int(violations.sum())}"
        )
    return subjects, names, sizes, values


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def unit_columns(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    array -= array.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(array, axis=0, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError("Constant residualized variable.")
    return array / norms


def bh(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array)
    ranked = array[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(array) / np.arange(1, len(array) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(1.0, adjusted_ranked)
    return adjusted


def base_design(age: np.ndarray, sex: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(age)), rankdata(age), sex])


def screen(
    matrix: np.ndarray,
    endpoint: np.ndarray,
    design: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, np.ndarray]:
    brain = unit_columns(residualize(rankdata(matrix, axis=0), design))
    endpoint_rank = rankdata(endpoint)
    fitted = design @ np.linalg.lstsq(design, endpoint_rank, rcond=None)[0]
    endpoint_residual = residualize(endpoint_rank, design)
    endpoint_unit = unit_columns(endpoint_residual).ravel()
    observed = brain.T @ endpoint_unit
    point_counts = np.zeros(120, dtype=np.int64)
    family_counts = np.zeros(120, dtype=np.int64)
    rng = np.random.default_rng(seed)
    chunk = 1000
    n = len(endpoint)
    for start in range(0, permutations, chunk):
        size = min(chunk, permutations - start)
        indices = np.argsort(rng.random((size, n)), axis=1)
        pseudo = fitted[None, :] + endpoint_residual[indices]
        coefficients = np.linalg.lstsq(design, pseudo.T, rcond=None)[0]
        null_endpoint = pseudo - (design @ coefficients).T
        null_endpoint /= np.linalg.norm(null_endpoint, axis=1, keepdims=True)
        absolute = np.abs(null_endpoint @ brain)
        point_counts += np.sum(absolute >= np.abs(observed)[None, :], axis=0)
        maxima = absolute.max(axis=1)
        family_counts += np.sum(maxima[:, None] >= np.abs(observed)[None, :], axis=0)
    denominator = permutations + 1.0
    p_raw = (point_counts + 1.0) / denominator
    return {
        "rho": observed,
        "p_raw": p_raw,
        "q_bh": bh(p_raw),
        "p_max_t": (family_counts + 1.0) / denominator,
    }


def adjusted_rho(brain: np.ndarray, endpoint: np.ndarray, design: np.ndarray) -> float:
    x = residualize(rankdata(brain), design)
    y = residualize(rankdata(endpoint), design)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 1.0e-12 else float("nan")


def bootstrap_rho(
    brain: np.ndarray,
    endpoint: np.ndarray,
    nuisance: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    repeats: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    estimates = np.full(repeats, np.nan)
    n = len(endpoint)
    for index in range(repeats):
        sample = rng.integers(0, n, size=n)
        design_parts = [np.ones(n), rankdata(age[sample]), sex[sample]]
        if nuisance.size:
            design_parts.extend(rankdata(nuisance[sample], axis=0).T)
        design = np.column_stack(design_parts)
        estimates[index] = adjusted_rho(brain[sample], endpoint[sample], design)
    return np.nanquantile(estimates, [0.025, 0.5, 0.975]).tolist()


def leave_one_out(
    brain: np.ndarray,
    endpoint: np.ndarray,
    nuisance: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
) -> dict[str, float]:
    values = []
    n = len(endpoint)
    for removed in range(n):
        keep = np.arange(n) != removed
        parts = [np.ones(keep.sum()), rankdata(age[keep]), sex[keep]]
        if nuisance.size:
            parts.extend(rankdata(nuisance[keep], axis=0).T)
        values.append(adjusted_rho(brain[keep], endpoint[keep], np.column_stack(parts)))
    array = np.asarray(values)
    return {
        "minimum": float(np.nanmin(array)),
        "median": float(np.nanmedian(array)),
        "maximum": float(np.nanmax(array)),
        "same_sign_fraction": float(np.mean(np.sign(array) == np.sign(np.nanmedian(array)))),
    }


def make_endpoint_contracts(
    subjects: np.ndarray, table: Mapping[str, Mapping[str, str]]
) -> dict[str, dict[str, Any]]:
    age = np.asarray(
        [common.age_midpoint(table[str(s).removeprefix("sub-")]["Age"]) for s in subjects]
    )
    sex = np.asarray(
        [table[str(s).removeprefix("sub-")]["Gender"] == "M" for s in subjects], dtype=float
    )
    social_scores = social.load_scores(subjects)
    motor_scores = common.load_scores(subjects)
    gambling_scores = gambling.load_scores(subjects)
    face_speed = -np.log(behavior_column(table, subjects, "Emotion_Task_Face_Median_RT"))
    shape_speed = -np.log(behavior_column(table, subjects, "Emotion_Task_Shape_Median_RT"))
    contracts = {
        "LANGUAGE": {
            "endpoint": behavior_column(table, subjects, "Language_Task_Math_Avg_Difficulty_Level"),
            "nuisance": np.empty((57, 0)),
            "label": "Corrected Story difficulty",
            "definition": "Language_Task_Math_Avg_Difficulty_Level after the confirmed HCP Story/Math alias correction",
            "in_scanner": True,
            "condition_specific": True,
        },
        "SOCIAL": {
            "endpoint": social_scores["dprime"],
            "nuisance": np.empty((57, 0)),
            "label": "Corrected social $d'$",
            "definition": "finite-trial corrected d-prime for TOM hits versus Random false alarms",
            "in_scanner": True,
            "condition_specific": True,
        },
        "EMOTION": {
            "endpoint": face_speed,
            "nuisance": shape_speed[:, None],
            "label": "Face speed | Shape speed",
            "definition": "negative log Face median RT controlling negative log Shape median RT",
            "in_scanner": True,
            "condition_specific": True,
        },
        "MOTOR": {
            "endpoint": motor_scores["composite"],
            "nuisance": np.empty((57, 0)),
            "label": "Broad motor score",
            "definition": "mean z score of age-adjusted endurance, dexterity, and grip strength",
            "in_scanner": False,
            "condition_specific": False,
        },
        "GAMBLING": {
            "endpoint": gambling_scores["composite"],
            "nuisance": np.empty((57, 0)),
            "label": "Delayed-reward valuation",
            "definition": "mean z score of DDisc_AUC_200 and DDisc_AUC_40K",
            "in_scanner": False,
            "condition_specific": False,
        },
        "RELATIONAL": {
            "endpoint": behavior_column(table, subjects, "Relational_Task_Acc"),
            "nuisance": np.empty((57, 0)),
            "label": "Overall relational accuracy",
            "definition": "Relational_Task_Acc",
            "in_scanner": True,
            "condition_specific": False,
        },
        "WM": {
            "endpoint": behavior_column(table, subjects, "WM_Task_Acc"),
            "nuisance": np.empty((57, 0)),
            "label": "Overall WM accuracy",
            "definition": "WM_Task_Acc",
            "in_scanner": True,
            "condition_specific": False,
        },
    }
    for contract in contracts.values():
        contract["age"] = age
        contract["sex"] = sex
        parts = [np.ones(57), rankdata(age), sex]
        if contract["nuisance"].size:
            parts.extend(rankdata(contract["nuisance"], axis=0).T)
        contract["design"] = np.column_stack(parts)
    return contracts


def condition_diagnostics(
    state: str,
    winner_brain: np.ndarray,
    subjects: np.ndarray,
    table: Mapping[str, Mapping[str, str]],
    age: np.ndarray,
    sex: np.ndarray,
) -> dict[str, float]:
    base = base_design(age, sex)
    fields: dict[str, str]
    if state == "RELATIONAL":
        fields = {
            "overall_accuracy": "Relational_Task_Acc",
            "relational_accuracy": "Relational_Task_Rel_Acc",
            "match_accuracy": "Relational_Task_Match_Acc",
        }
    elif state == "WM":
        fields = {
            "overall_accuracy": "WM_Task_Acc",
            "zero_back_accuracy": "WM_Task_0bk_Acc",
        }
    else:
        return {}
    output = {
        label: adjusted_rho(winner_brain, behavior_column(table, subjects, field), base)
        for label, field in fields.items()
    }
    if state == "RELATIONAL":
        match = behavior_column(table, subjects, "Relational_Task_Match_Acc")
        relational = behavior_column(table, subjects, "Relational_Task_Rel_Acc")
        output["relational_specific_controlling_match"] = adjusted_rho(
            winner_brain, relational, np.column_stack([base, rankdata(match)])
        )
    if state == "WM":
        two_back = np.asarray(
            [
                float(table[str(s).removeprefix("sub-")]["WM_Task_2bk_Acc"])
                if table[str(s).removeprefix("sub-")]["WM_Task_2bk_Acc"] not in ("", "NA")
                else np.nan
                for s in subjects
            ]
        )
        keep = np.isfinite(two_back)
        zero_back = behavior_column(table, subjects, "WM_Task_0bk_Acc")
        base_complete = base_design(age[keep], sex[keep])
        output["two_back_accuracy_n56"] = adjusted_rho(
            winner_brain[keep], two_back[keep], base_complete
        )
        output["two_back_specific_controlling_zero_back_n56"] = adjusted_rho(
            winner_brain[keep],
            two_back[keep],
            np.column_stack([base_complete, rankdata(zero_back[keep])]),
        )
    return output


def evidence_tier(row: Mapping[str, Any]) -> str:
    interval = row["bootstrap_95_ci"]
    excludes_zero = interval[0] > 0 or interval[1] < 0
    if row["p_max_t_120"] < 0.05:
        return "A"
    if row["p_max_t_120"] < 0.10 and excludes_zero:
        return "B"
    if row["p_raw"] < 0.05 and excludes_zero:
        return "C"
    return "D"


def plot_task(
    state: str,
    names: np.ndarray,
    sizes: np.ndarray,
    matrix: np.ndarray,
    contract: Mapping[str, Any],
    result: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    table: Mapping[str, Mapping[str, str]],
    subjects: np.ndarray,
) -> None:
    configure_style()
    output = RELATIONAL_OUTPUT if state == "RELATIONAL" else WM_OUTPUT
    winner = int(np.argmax(np.abs(result["rho"])))
    top = np.argsort(-np.abs(result["rho"]))[:10]
    figure = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=[0.95, 1.15])
    axes = [figure.add_subplot(grid[r, c]) for r, c in ((0, 0), (0, 1), (1, 0), (1, 1))]

    if state == "RELATIONAL":
        x = behavior_column(table, subjects, "Relational_Task_Match_Acc")
        y = behavior_column(table, subjects, "Relational_Task_Rel_Acc")
        axes[0].set(xlabel="Match accuracy (%)", ylabel="Relational accuracy (%)")
    else:
        x = behavior_column(table, subjects, "WM_Task_0bk_Acc")
        raw_two = [table[str(s).removeprefix("sub-")]["WM_Task_2bk_Acc"] for s in subjects]
        y = np.asarray([float(value) if value not in ("", "NA") else np.nan for value in raw_two])
        axes[0].set(xlabel="0-back accuracy (%)", ylabel="2-back accuracy (%)")
    keep = np.isfinite(x) & np.isfinite(y)
    axes[0].scatter(x[keep], y[keep], s=20, color="#7890A8", alpha=0.82, edgecolor="white", linewidth=0.35)
    axes[0].text(
        0.04,
        0.96,
        rf"Spearman $\rho$={spearmanr(x[keep], y[keep]).statistic:+.3f}"
        + f"\n$n$={keep.sum()}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
    )
    axes[0].set_title("a  Condition-score structure", loc="left", fontweight="bold")

    rng = np.random.default_rng(SEED)
    axes[1].axhline(0, color="#D6DADF", linewidth=0.7)
    axes[1].scatter(sizes + rng.uniform(-0.10, 0.10, len(sizes)), result["rho"], s=19, color="#7890A8", alpha=0.78, edgecolor="white", linewidth=0.3)
    axes[1].scatter(sizes[winner], result["rho"][winner], s=62, marker="D", color="#C56F4D", edgecolor="white", linewidth=0.5, zorder=4)
    axes[1].annotate(compact_name(str(names[winner])), (sizes[winner], result["rho"][winner]), xytext=(5, 5), textcoords="offset points", fontsize=6.3, color="#91452E")
    axes[1].set(xlabel="Coalition size (Yeo7 networks)", ylabel=r"Age/sex-adjusted $\rho$")
    axes[1].set_xticks(np.arange(2, 8))
    axes[1].set_title(f"b  All 120 {state} coalitions", loc="left", fontweight="bold")

    ordered = top[::-1]
    axes[2].axvline(0, color="#D6DADF", linewidth=0.7)
    axes[2].scatter(result["rho"][ordered], np.arange(10), s=24, color="#526D82")
    axes[2].scatter(result["rho"][winner], int(np.flatnonzero(ordered == winner)[0]), s=42, marker="D", color="#C56F4D", zorder=4)
    axes[2].set_yticks(np.arange(10), [compact_name(str(names[index])) for index in ordered])
    axes[2].set_xlabel(r"Adjusted $\rho$")
    axes[2].set_title("c  Ten strongest associations", loc="left", fontweight="bold")

    brain_rank = residualize(rankdata(matrix[:, winner]), contract["design"])
    score_rank = residualize(rankdata(contract["endpoint"]), contract["design"])
    axes[3].scatter(score_rank, brain_rank, s=20, color="#7890A8", alpha=0.82, edgecolor="white", linewidth=0.35)
    order = np.argsort(score_rank)
    coefficient = np.polyfit(score_rank, brain_rank, 1)
    axes[3].plot(score_rank[order], np.polyval(coefficient, score_rank[order]), color="#465563", linestyle="--", linewidth=0.9)
    winner_row = rows[winner]
    axes[3].text(1.04, 0.98, rf"adjusted $\rho$={winner_row['rho']:+.3f}" + f"\nmax-T $p$={winner_row['p_max_t_120']:.4f}\n$n$=57", transform=axes[3].transAxes, ha="left", va="top", clip_on=False)
    axes[3].set(xlabel=f"{contract['label']} (adjusted rank residual)", ylabel="Coalition Syn (adjusted rank residual)")
    axes[3].set_title(f"d  {compact_name(str(names[winner]))}", loc="left", fontweight="bold")

    stem = "relational_performance_coalition_screen_57" if state == "RELATIONAL" else "wm_performance_coalition_screen_57"
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        figure.savefig(output / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(figure)


def plot_ranking(ranking: Sequence[Mapping[str, Any]]) -> None:
    configure_style()
    ordered = list(reversed(ranking))
    labels = [TASK_LABELS[row["task"]] for row in ordered]
    y = np.arange(len(ordered))
    rho = np.asarray([row["rho"] for row in ordered])
    intervals = np.asarray([row["bootstrap_95_ci"] for row in ordered])
    pmax = np.asarray([row["p_max_t_120"] for row in ordered])
    colors = {"A": "#3F7F6F", "B": "#C17A3F", "C": "#7890A8", "D": "#AEB7C1"}
    point_colors = [colors[row["evidence_tier"]] for row in ordered]
    figure = plt.figure(figsize=(7.2, 4.45), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=[1.28, 0.92, 1.08])
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]

    axes[0].axvline(0, color="#D6DADF", linewidth=0.8)
    for index in range(len(ordered)):
        axes[0].errorbar(rho[index], y[index], xerr=np.asarray([[rho[index] - intervals[index, 0]], [intervals[index, 1] - rho[index]]]), fmt="o", color=point_colors[index], ecolor="#9AA8B2", elinewidth=0.9, capsize=2, markersize=4.5)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel(r"Winner adjusted $\rho$ (bootstrap 95% CI)")
    axes[0].set_title("a  Strongest primary association", loc="left", fontweight="bold")

    axes[1].axvline(0.05, color="#8A4F3D", linestyle=":", linewidth=0.9)
    axes[1].scatter(pmax, y, s=30, color=point_colors, edgecolor="white", linewidth=0.4)
    axes[1].set_xscale("log")
    axes[1].set_xlim(1.0e-4, 1.1)
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel(r"120-coalition max-$T$ $p$")
    axes[1].set_title("b  Selection-aware evidence", loc="left", fontweight="bold")

    matrix = np.asarray(
        [
            [row["p_max_t_120"] < 0.05, row["p_raw"] < 0.05, row["ci_excludes_zero"], row["in_scanner"], row["condition_specific"]]
            for row in ordered
        ],
        dtype=float,
    )
    axes[2].imshow(matrix, vmin=0, vmax=1, cmap=mpl.colors.ListedColormap(["#E3E7EA", "#526D82"]), aspect="auto")
    axes[2].set_xticks(np.arange(5), ["max-T\n< .05", "raw p\n< .05", "CI excludes\n0", "in-scanner\nscore", "condition-\nspecific"], rotation=35, ha="right")
    axes[2].set_yticks(y, [])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[2].text(column, row, "Y" if matrix[row, column] else "-", ha="center", va="center", color="white" if matrix[row, column] else "#7E8893", fontsize=6.5, fontweight="bold" if matrix[row, column] else "normal")
    axes[2].set_title("c  Manuscript-priority checks", loc="left", fontweight="bold")

    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        figure.savefig(OUTPUT / f"hcp_all_task_behavior_evidence_ranking_57.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(common.SUBJECT_SOURCE, allow_pickle=False) as archive:
        subjects = archive["subjects"].astype(str)
    if subjects.shape != (57,):
        raise ValueError("Expected the frozen 57-subject order.")
    compute_new_matrices(subjects, args.recompute)
    table = load_table()
    contracts = make_endpoint_contracts(subjects, table)

    task_results: dict[str, Any] = {}
    association_rows: list[dict[str, Any]] = []
    frozen_names: np.ndarray | None = None
    frozen_sizes: np.ndarray | None = None
    for task_index, state in enumerate(TASK_ORDER):
        _, names, sizes, matrix = load_matrix(state, subjects)
        if frozen_names is None:
            frozen_names, frozen_sizes = names, sizes
        elif not np.array_equal(names, frozen_names):
            raise ValueError(f"Coalition order mismatch for {state}.")
        contract = contracts[state]
        result = screen(matrix, contract["endpoint"], contract["design"], args.permutations, SEED + task_index)
        winner = int(np.argmax(np.abs(result["rho"])))
        ranking = np.argsort(-np.abs(result["rho"]))
        rows = []
        for index in range(120):
            row = {
                "task": state,
                "index": index,
                "coalition": str(names[index]),
                "short_coalition": compact_name(str(names[index])),
                "coalition_size": int(sizes[index]),
                "rho": float(result["rho"][index]),
                "p_raw": float(result["p_raw"][index]),
                "q_bh_120": float(result["q_bh"][index]),
                "p_max_t_120": float(result["p_max_t"][index]),
            }
            rows.append(row)
            association_rows.append(row)
        nuisance = np.asarray(contract["nuisance"], dtype=float)
        winner_ci = bootstrap_rho(matrix[:, winner], contract["endpoint"], nuisance, contract["age"], contract["sex"], args.bootstraps, SEED + 100 + task_index)
        winner_row = rows[winner]
        winner_row["bootstrap_95_ci"] = [winner_ci[0], winner_ci[2]]
        winner_row["bootstrap_median"] = winner_ci[1]
        winner_row["leave_one_out"] = leave_one_out(matrix[:, winner], contract["endpoint"], nuisance, contract["age"], contract["sex"])
        winner_row["condition_diagnostics"] = condition_diagnostics(state, matrix[:, winner], subjects, table, contract["age"], contract["sex"])
        winner_row["in_scanner"] = bool(contract["in_scanner"])
        winner_row["condition_specific"] = bool(contract["condition_specific"])
        winner_row["ci_excludes_zero"] = bool(winner_ci[0] > 0 or winner_ci[2] < 0)
        winner_row["evidence_tier"] = evidence_tier(winner_row)
        task_results[state] = {
            "endpoint": {key: contract[key] for key in ("label", "definition", "in_scanner", "condition_specific")},
            "winner": winner_row,
            "top_ten": [rows[index] for index in ranking[:10]],
            "significance_counts": {
                "raw_p_below_0_05": int(np.sum(result["p_raw"] < 0.05)),
                "bh_q_below_0_05": int(np.sum(result["q_bh"] < 0.05)),
                "max_t_p_below_0_05": int(np.sum(result["p_max_t"] < 0.05)),
            },
            "model": {
                "minimum_synergy_bits": float(matrix.min()),
                "checked_syn_count": int(matrix.size),
                "significant_nonnegativity_violation_count": int(np.sum(matrix < -SYN_TOLERANCE_BITS)),
            },
        }
        if state in ("RELATIONAL", "WM"):
            plot_task(state, names, sizes, matrix, contract, result, rows, table, subjects)

    tier_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    ranking = sorted(
        [
            {
                "rank": 0,
                "task": state,
                **task_results[state]["winner"],
            }
            for state in TASK_ORDER
        ],
        key=lambda row: (tier_order[row["evidence_tier"]], row["p_max_t_120"], row["p_raw"], -abs(row["rho"])),
    )
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
    plot_ranking(ranking)

    summary = {
        "experiment": "Pooled-57 primary behavior association ranking across all seven HCP task states",
        "subjects": 57,
        "cohort_handling": "All 57 subjects analyzed as one pooled sample; no original/supplement cohort covariate, blocking, coloring, or subgroup estimate.",
        "common_contract": {
            "coalitions_per_task": 120,
            "primary_endpoints_per_task": 1,
            "covariates": ["age rank", "sex"],
            "estimator": "Schaefer-1000/Yeo7 network PC1, order-3 Ridge alpha=1, affine Gaussian TM fixed-coalition Syn",
            "permutations": args.permutations,
            "permutation_scheme": "unrestricted pooled-sample Freedman-Lane residual permutation",
            "bootstraps": args.bootstraps,
            "ranking_rule": "lexicographic evidence tier (A max-T<.05; B max-T<.10 and CI excludes 0; C raw p<.05 and CI excludes 0; D otherwise), then max-T p, raw p, and absolute rho",
        },
        "tasks": task_results,
        "ranking": ranking,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUTPUT / "all_associations.jsonl").open("w", encoding="utf-8") as handle:
        for row in association_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    contract = {
        "scientific_question": "Which task state has the strongest selection-aware pooled-57 association between its primary behavior endpoint and a fixed Yeo7 coalition?",
        "treatment_factor": "task state (seven levels) after an identical 120-coalition screen per task",
        "frozen_variables": summary["common_contract"],
        "primary_endpoints": {state: task_results[state]["endpoint"] for state in TASK_ORDER},
        "figure_contract": {
            "core_conclusion": "Rank seven task states by corrected evidence rather than by uncorrected effect size alone.",
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib",
            "final_width_mm": 183,
            "panels": ["winner effect and CI", "120-coalition max-T p", "manuscript-priority checks"],
            "exports": ["PNG 600 dpi", "editable SVG", "PDF"],
        },
    }
    (OUTPUT / "experiment_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Pooled-57 HCP task-behavior coalition ranking",
        "",
        "![Seven-task evidence ranking](hcp_all_task_behavior_evidence_ranking_57.png)",
        "",
        "All 57 subjects are analyzed as a single pooled sample. Age and sex are covariates; recruitment cohort is not modeled, blocked, colored, or reported. Each task contributes one frozen primary endpoint and exactly 120 coalition tests.",
        "",
        "| Rank | Task | Primary endpoint | Coalition | Adjusted rho | Bootstrap 95% CI | Raw p | BH q | max-T p | Tier |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in ranking:
        endpoint = task_results[row["task"]]["endpoint"]["label"]
        lines.append(f"| {row['rank']} | {TASK_LABELS[row['task']]} | {endpoint} | {row['short_coalition']} | {row['rho']:+.3f} | [{row['bootstrap_95_ci'][0]:+.3f}, {row['bootstrap_95_ci'][1]:+.3f}] | {row['p_raw']:.5f} | {row['q_bh_120']:.5f} | {row['p_max_t_120']:.5f} | {row['evidence_tier']} |")
    lines.extend(["", "Tiers are evidence grades, not biological importance scores. The ranking is selection-aware within each task but remains exploratory because the same 57 subjects are used for coalition selection and effect estimation.", ""])
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")

    for state, output, stem in (("RELATIONAL", RELATIONAL_OUTPUT, "relational_performance_coalition_screen_57"), ("WM", WM_OUTPUT, "wm_performance_coalition_screen_57")):
        task = task_results[state]
        winner = task["winner"]
        report = [
            f"# HCP {state} pooled-57 coalition screen",
            "",
            f"![{state} screen]({stem}.png)",
            "",
            "All 57 subjects are analyzed as one pooled sample with age and sex covariates. No recruitment-cohort split or blocking is used.",
            "",
            f"Primary endpoint: **{task['endpoint']['label']}**. Winner: **{winner['short_coalition']}**, adjusted rho={winner['rho']:+.3f}, raw p={winner['p_raw']:.5f}, BH q={winner['q_bh_120']:.5f}, 120-coalition max-T p={winner['p_max_t_120']:.5f}.",
            "",
            "Condition diagnostics: " + "; ".join(f"{key}={value:+.3f}" for key, value in winner["condition_diagnostics"].items()) + ".",
            "",
        ]
        (output / "report.md").write_text("\n".join(report), encoding="utf-8")
        (output / "summary.json").write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
