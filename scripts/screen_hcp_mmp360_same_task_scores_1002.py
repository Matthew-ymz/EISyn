#!/usr/bin/env python3
"""Screen same-task HCP scores against all Yeo7 coalition orders in MMP360 data."""

from __future__ import annotations

import argparse
import csv
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
from scipy.io import loadmat
from scipy.stats import norm, rankdata


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phi_hierarchy import subset_phi_raw
from scripts.run_hcp_schaefer500_yeo7_module_phi_decomposition import (
    module_ei_table,
    network_history_indices,
)
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi
from scripts.screen_hcp_motor_composite_scores_57 import age_midpoint
from scripts.screen_hcp_social_composite_scores_57 import infer_effective_trials


DATA_ROOT = ROOT / "data/hcp_s1200_1002_complete_scores_task_lr_mmp360_yeo7_pc1_timeseries"
BEHAVIOR = ROOT / "data/unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
OLD_CACHE = ROOT / "results/hcp_mmp360_behavior_main_validation_1002/coalition_synergy_1002.npz"
OUTPUT = ROOT / "results/hcp_mmp360_same_task_score_screen_1002"
CACHE = OUTPUT / "coalition_synergy_all7_1002.npz"
SUMMARY = OUTPUT / "summary.json"
FIGURE = OUTPUT / "same_task_score_coalition_screen_1002.png"

TASKS = ("LANGUAGE", "SOCIAL", "EMOTION", "MOTOR", "GAMBLING", "RELATIONAL", "WM")
NETWORKS = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
SHORT = {
    "Vis": "Vis",
    "SomMot": "SM",
    "DorsAttn": "DAN",
    "SalVentAttn": "SVAN",
    "Limbic": "Lim",
    "Cont": "Cont",
    "Default": "DMN",
}
ORDER = 3
ALPHA = 1.0
SYN_TOLERANCE_BITS = 1.0e-9
SEED = 20260914


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".json", mode="w", encoding="utf-8", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def coalition_names() -> tuple[np.ndarray, np.ndarray]:
    combinations = tuple(
        combination
        for size in range(2, 8)
        for combination in itertools.combinations(NETWORKS, size)
    )
    return (
        np.asarray(["+".join(combination) for combination in combinations]),
        np.asarray([len(combination) for combination in combinations], dtype=np.int8),
    )


def load_subjects() -> np.ndarray:
    subjects = np.asarray(sorted(path.name for path in DATA_ROOT.glob("sub-*")))
    if subjects.shape != (1002,) or len(set(subjects.tolist())) != 1002:
        raise ValueError(f"Expected 1002 unique subjects, found {subjects.shape}.")
    return subjects


def save_cache(
    subjects: np.ndarray,
    names: np.ndarray,
    sizes: np.ndarray,
    values: np.ndarray,
    heldout: np.ndarray,
) -> None:
    atomic_npz(
        CACHE,
        subjects=subjects,
        tasks=np.asarray(TASKS),
        coalitions=names,
        coalition_sizes=sizes,
        synergy_bits=values,
        heldout_skill_ratio=heldout,
        order=np.asarray(ORDER),
        alpha=np.asarray(ALPHA),
        syn_tolerance_bits=np.asarray(SYN_TOLERANCE_BITS),
    )


def compute_matrices(
    subjects: np.ndarray,
    names: np.ndarray,
    sizes: np.ndarray,
    recompute: bool,
    required_tasks: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((len(TASKS), len(subjects), len(names)), np.nan, dtype=np.float64)
    heldout = np.full((len(TASKS), len(subjects)), np.nan, dtype=np.float64)
    if CACHE.is_file() and not recompute:
        with np.load(CACHE, allow_pickle=False) as archive:
            if (
                np.array_equal(archive["subjects"].astype(str), subjects)
                and np.array_equal(archive["tasks"].astype(str), np.asarray(TASKS))
                and np.array_equal(archive["coalitions"].astype(str), names)
            ):
                candidate = archive["synergy_bits"].astype(float)
                candidate_heldout = archive["heldout_skill_ratio"].astype(float)
                if candidate.shape == values.shape and candidate_heldout.shape == heldout.shape:
                    values, heldout = candidate, candidate_heldout
    elif OLD_CACHE.is_file() and not recompute:
        with np.load(OLD_CACHE, allow_pickle=False) as archive:
            if (
                np.array_equal(archive["subjects"].astype(str), subjects)
                and np.array_equal(archive["coalitions"].astype(str), names)
            ):
                for source_index, task in enumerate(archive["tasks"].astype(str)):
                    target_index = TASKS.index(task)
                    values[target_index] = archive["synergy_bits"][source_index]
                    heldout[target_index] = archive["heldout_skill_ratio"][source_index]

    combinations = tuple(tuple(name.split("+")) for name in names)
    indices = network_history_indices(NETWORKS, order=ORDER)
    started = time.perf_counter()
    for task_index, task in enumerate(TASKS):
        if task not in required_tasks:
            continue
        missing = np.flatnonzero(~np.isfinite(values[task_index]).all(axis=1))
        if not len(missing):
            continue
        for completed, subject_index in enumerate(missing, start=1):
            subject = str(subjects[subject_index])
            with_mat = loadmat(DATA_ROOT / subject / f"{task}_LR.mat")
            series = np.asarray(with_mat["Yeo7_taskRetainedPC1"], dtype=float)
            development_end = int(np.asarray(with_mat["development_end"]).squeeze())
            fitted = fit_delta_history_phi(
                series, alpha=ALPHA, order=ORDER, development_end=development_end
            )
            table = module_ei_table(
                fitted["transition"], fitted["noise_covariance"], indices, ridge=1.0e-6
            )
            singleton = {network: float(table[(network,)]) for network in NETWORKS}
            values[task_index, subject_index] = [
                subset_phi_raw(combination, table, singleton) for combination in combinations
            ]
            heldout[task_index, subject_index] = float(fitted["heldout"]["skill_ratio"])
            if completed % 100 == 0 or completed == len(missing):
                save_cache(subjects, names, sizes, values, heldout)
                print(
                    f"coalitions {task}: {completed}/{len(missing)} missing rows completed; "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    violations = values < -SYN_TOLERANCE_BITS
    if np.any(violations):
        raise ValueError(
            f"Syn nonnegativity violation: min={np.nanmin(values):.12g}, "
            f"threshold={-SYN_TOLERANCE_BITS:.1e}, count={int(violations.sum())}"
        )
    required_indices = [TASKS.index(task) for task in required_tasks]
    if not np.isfinite(values[required_indices]).all() or not np.isfinite(heldout[required_indices]).all():
        raise ValueError("Every requested task matrix must be complete and finite.")
    save_cache(subjects, names, sizes, values, heldout)
    return values, heldout


def as_float(rows: Sequence[Mapping[str, str]], field: str) -> np.ndarray:
    output = np.full(len(rows), np.nan)
    for index, row in enumerate(rows):
        try:
            output[index] = float(row[field])
        except (KeyError, TypeError, ValueError):
            pass
    return output


def speed(rows: Sequence[Mapping[str, str]], field: str) -> np.ndarray:
    values = as_float(rows, field)
    values[values <= 0] = np.nan
    return -np.log(values)


def mean_speed(rows: Sequence[Mapping[str, str]], fields: Sequence[str]) -> np.ndarray:
    values = np.column_stack([as_float(rows, field) for field in fields])
    values[values <= 0] = np.nan
    return -np.log(np.mean(values, axis=1))


def endpoint(label: str, values: np.ndarray, nuisance: Sequence[np.ndarray] = (), **meta: Any) -> dict[str, Any]:
    return {"label": label, "values": values, "nuisance": list(nuisance), **meta}


def load_endpoints(subjects: np.ndarray) -> tuple[dict[str, list[dict[str, Any]]], np.ndarray, np.ndarray]:
    with BEHAVIOR.open(newline="", encoding="utf-8-sig") as handle:
        table = {str(row["Subject"]): row for row in csv.DictReader(handle)}
    rows = [table[str(subject).removeprefix("sub-")] for subject in subjects]
    age = np.asarray([age_midpoint(row["Age"]) for row in rows], dtype=float)
    sex = np.asarray([row["Gender"] == "M" for row in rows], dtype=float)

    trials = np.asarray([infer_effective_trials(row) for row in rows], dtype=int)
    social_hit = as_float(rows, "Social_Task_TOM_Perc_TOM")
    social_cr = as_float(rows, "Social_Task_Random_Perc_Random")
    hits = np.rint(social_hit * trials / 100.0)
    false_alarms = np.rint((100.0 - social_cr) * trials / 100.0)
    z_hit = norm.ppf((hits + 0.5) / (trials + 1.0))
    z_false_alarm = norm.ppf((false_alarms + 0.5) / (trials + 1.0))

    motor_raw = np.column_stack(
        [
            as_float(rows, "Endurance_AgeAdj"),
            as_float(rows, "Dexterity_AgeAdj"),
            as_float(rows, "Strength_AgeAdj"),
        ]
    )
    motor_composite = ((motor_raw - motor_raw.mean(axis=0)) / motor_raw.std(axis=0, ddof=1)).mean(axis=1)

    face_accuracy = as_float(rows, "Emotion_Task_Face_Acc")
    shape_accuracy = as_float(rows, "Emotion_Task_Shape_Acc")
    face_speed = speed(rows, "Emotion_Task_Face_Median_RT")
    shape_speed = speed(rows, "Emotion_Task_Shape_Median_RT")
    relational_accuracy = as_float(rows, "Relational_Task_Rel_Acc")
    match_accuracy = as_float(rows, "Relational_Task_Match_Acc")
    relational_speed = speed(rows, "Relational_Task_Rel_Median_RT")
    match_speed = speed(rows, "Relational_Task_Match_Median_RT")
    wm_zero_accuracy = as_float(rows, "WM_Task_0bk_Acc")
    wm_two_accuracy = as_float(rows, "WM_Task_2bk_Acc")
    wm_zero_speed = speed(rows, "WM_Task_0bk_Median_RT")
    wm_two_speed = speed(rows, "WM_Task_2bk_Median_RT")

    endpoints: dict[str, list[dict[str, Any]]] = {
        "LANGUAGE": [
            endpoint("Overall accuracy", as_float(rows, "Language_Task_Acc")),
            endpoint("Story accuracy", as_float(rows, "Language_Task_Story_Acc")),
            endpoint("Math accuracy", as_float(rows, "Language_Task_Math_Acc")),
            endpoint("Overall speed", speed(rows, "Language_Task_Median_RT")),
            endpoint("Story speed", speed(rows, "Language_Task_Story_Median_RT")),
            endpoint("Math speed", speed(rows, "Language_Task_Math_Median_RT")),
        ],
        "SOCIAL": [
            endpoint("Corrected d-prime", z_hit - z_false_alarm),
            endpoint("Balanced accuracy", 0.5 * (social_hit + social_cr)),
            endpoint("TOM hit rate", social_hit),
            endpoint("Random correct rejection", social_cr),
            endpoint("TOM correct-response speed", speed(rows, "Social_Task_TOM_Median_RT_TOM")),
            endpoint("Random correct-response speed", speed(rows, "Social_Task_Random_Median_RT_Random")),
        ],
        "EMOTION": [
            endpoint("Overall accuracy", as_float(rows, "Emotion_Task_Acc")),
            endpoint("Shape accuracy", shape_accuracy),
            endpoint("Face accuracy | Shape accuracy", face_accuracy, [shape_accuracy]),
            endpoint("Overall speed", speed(rows, "Emotion_Task_Median_RT")),
            endpoint("Shape speed", shape_speed),
            endpoint("Face speed | Shape speed", face_speed, [shape_speed]),
        ],
        "MOTOR": [
            endpoint("Broad motor composite", motor_composite, source="out_of_scanner"),
            endpoint("Endurance", motor_raw[:, 0], source="out_of_scanner"),
            endpoint("Dexterity", motor_raw[:, 1], source="out_of_scanner"),
            endpoint("Strength", motor_raw[:, 2], source="out_of_scanner"),
        ],
        "GAMBLING": [
            endpoint("Overall larger-choice fraction", as_float(rows, "Gambling_Task_Perc_Larger"), interpretation="choice_tendency"),
            endpoint("Reward larger-choice fraction", as_float(rows, "Gambling_Task_Reward_Perc_Larger"), interpretation="choice_tendency"),
            endpoint("Punish larger-choice fraction", as_float(rows, "Gambling_Task_Punish_Perc_Larger"), interpretation="choice_tendency"),
            endpoint(
                "Overall choice speed",
                mean_speed(rows, ("Gambling_Task_Median_RT_Larger", "Gambling_Task_Median_RT_Smaller")),
                interpretation="response_speed",
            ),
            endpoint(
                "Reward choice speed",
                mean_speed(rows, ("Gambling_Task_Reward_Median_RT_Larger", "Gambling_Task_Reward_Median_RT_Smaller")),
                interpretation="response_speed",
            ),
            endpoint(
                "Punish choice speed",
                mean_speed(rows, ("Gambling_Task_Punish_Median_RT_Larger", "Gambling_Task_Punish_Median_RT_Smaller")),
                interpretation="response_speed",
            ),
        ],
        "RELATIONAL": [
            endpoint("Overall accuracy", as_float(rows, "Relational_Task_Acc")),
            endpoint("Match accuracy", match_accuracy),
            endpoint("Relational accuracy | Match accuracy", relational_accuracy, [match_accuracy]),
            endpoint("Overall speed", speed(rows, "Relational_Task_Median_RT")),
            endpoint("Match speed", match_speed),
            endpoint("Relational speed | Match speed", relational_speed, [match_speed]),
        ],
        "WM": [
            endpoint("Overall accuracy", as_float(rows, "WM_Task_Acc")),
            endpoint("0-back accuracy", wm_zero_accuracy),
            endpoint("2-back accuracy | 0-back accuracy", wm_two_accuracy, [wm_zero_accuracy]),
            endpoint("Overall speed", speed(rows, "WM_Task_Median_RT")),
            endpoint("0-back speed", wm_zero_speed),
            endpoint("2-back speed | 0-back speed", wm_two_speed, [wm_zero_speed]),
        ],
    }
    for stimulus in ("Body", "Face", "Place", "Tool"):
        zero_accuracy = as_float(rows, f"WM_Task_0bk_{stimulus}_Acc")
        two_accuracy = as_float(rows, f"WM_Task_2bk_{stimulus}_Acc")
        zero_speed = speed(rows, f"WM_Task_0bk_{stimulus}_Median_RT")
        two_speed = speed(rows, f"WM_Task_2bk_{stimulus}_Median_RT")
        endpoints["WM"].extend(
            [
                endpoint(f"2-back {stimulus.lower()} accuracy | 0-back", two_accuracy, [zero_accuracy]),
                endpoint(f"2-back {stimulus.lower()} speed | 0-back", two_speed, [zero_speed]),
            ]
        )
    if sum(len(value) for value in endpoints.values()) != 48:
        raise AssertionError("Endpoint contract must contain exactly 48 endpoints.")
    return endpoints, age, sex


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def unit_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    values = values - values.mean(axis=0, keepdims=True)
    lengths = np.linalg.norm(values, axis=0, keepdims=True)
    if np.any(lengths <= 1.0e-12):
        raise ValueError("Constant residualized variable.")
    return values / lengths


def bh(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=float).ravel()
    order = np.argsort(flat)
    ranked = flat[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(flat) / np.arange(1, len(flat) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(1.0, adjusted_ranked)
    return adjusted.reshape(np.asarray(values).shape)


def holm(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.maximum.accumulate(ranked * np.arange(len(values), 0, -1))
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1.0, adjusted_ranked)
    return adjusted


def task_mask(specs: Sequence[Mapping[str, Any]], age: np.ndarray, sex: np.ndarray) -> np.ndarray:
    mask = np.isfinite(age) & np.isfinite(sex)
    for spec in specs:
        mask &= np.isfinite(spec["values"])
        for nuisance in spec["nuisance"]:
            mask &= np.isfinite(nuisance)
    return mask


def design_for(spec: Mapping[str, Any], age: np.ndarray, sex: np.ndarray, mask: np.ndarray) -> np.ndarray:
    parts = [np.ones(mask.sum()), rankdata(age[mask]), sex[mask]]
    parts.extend(rankdata(nuisance[mask]) for nuisance in spec["nuisance"])
    return np.column_stack(parts)


def screen_task(
    matrix: np.ndarray,
    specs: Sequence[Mapping[str, Any]],
    age: np.ndarray,
    sex: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    mask = task_mask(specs, age, sex)
    matrix = matrix[mask]
    endpoint_count = len(specs)
    observed = np.empty((endpoint_count, matrix.shape[1]))
    raw_counts = np.zeros_like(observed, dtype=np.int64)
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for endpoint_index, spec in enumerate(specs):
        design = design_for(spec, age, sex, mask)
        brain = unit_columns(residualize(rankdata(matrix, axis=0), design))
        endpoint_rank = rankdata(spec["values"][mask])
        fitted = design @ np.linalg.lstsq(design, endpoint_rank, rcond=None)[0]
        endpoint_residual = residualize(endpoint_rank, design)
        endpoint_unit = unit_columns(endpoint_residual).ravel()
        observed[endpoint_index] = brain.T @ endpoint_unit
        prepared.append((design, brain, fitted, endpoint_residual))

    family_counts = np.zeros_like(observed, dtype=np.int64)
    rng = np.random.default_rng(seed)
    indices_base = np.arange(mask.sum())
    chunk = 500
    for start in range(0, permutations, chunk):
        batch = min(chunk, permutations - start)
        indices = np.asarray([rng.permutation(indices_base) for _ in range(batch)])
        family_maximum = np.zeros(batch)
        for endpoint_index, (design, brain, fitted, endpoint_residual) in enumerate(prepared):
            pseudo = fitted[None, :] + endpoint_residual[indices]
            coefficients = np.linalg.lstsq(design, pseudo.T, rcond=None)[0]
            null_endpoint = pseudo - (design @ coefficients).T
            null_endpoint /= np.linalg.norm(null_endpoint, axis=1, keepdims=True)
            absolute = np.abs(null_endpoint @ brain)
            raw_counts[endpoint_index] += np.sum(
                absolute >= np.abs(observed[endpoint_index])[None, :], axis=0
            )
            family_maximum = np.maximum(family_maximum, absolute.max(axis=1))
        flat_observed = np.abs(observed).ravel()
        family_counts += np.sum(
            family_maximum[:, None] >= flat_observed[None, :], axis=0
        ).reshape(observed.shape)

    denominator = permutations + 1.0
    p_raw = (raw_counts + 1.0) / denominator
    p_max_t = (family_counts + 1.0) / denominator
    q_bh = bh(p_raw)
    rows: list[dict[str, Any]] = []
    for endpoint_index, spec in enumerate(specs):
        for coalition_index in range(matrix.shape[1]):
            rows.append(
                {
                    "endpoint_index": endpoint_index,
                    "coalition_index": coalition_index,
                    "endpoint": spec["label"],
                    "rho": float(observed[endpoint_index, coalition_index]),
                    "p_raw": float(p_raw[endpoint_index, coalition_index]),
                    "q_bh_task_family": float(q_bh[endpoint_index, coalition_index]),
                    "p_max_t_task_family": float(p_max_t[endpoint_index, coalition_index]),
                }
            )
    return rows, mask


def bootstrap_ci(
    brain: np.ndarray,
    spec: Mapping[str, Any],
    age: np.ndarray,
    sex: np.ndarray,
    mask: np.ndarray,
    repeats: int,
    seed: int,
) -> list[float]:
    brain, endpoint_values = brain[mask], spec["values"][mask]
    nuisance = [value[mask] for value in spec["nuisance"]]
    age, sex = age[mask], sex[mask]
    rng = np.random.default_rng(seed)
    estimates = np.full(repeats, np.nan)
    for repeat in range(repeats):
        take = rng.integers(0, len(brain), len(brain))
        parts = [np.ones(len(take)), rankdata(age[take]), sex[take]]
        parts.extend(rankdata(value[take]) for value in nuisance)
        design = np.column_stack(parts)
        x = residualize(rankdata(brain[take]), design)
        y = residualize(rankdata(endpoint_values[take]), design)
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        if denominator > 1.0e-12:
            estimates[repeat] = float(x @ y / denominator)
    return np.nanquantile(estimates, [0.025, 0.5, 0.975]).tolist()


def compact(name: str) -> str:
    return "+".join(SHORT[value] for value in name.split("+"))


def plot_summary(payload: Mapping[str, Any]) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    winners = payload["task_winners"]
    task_colors = {
        "LANGUAGE": "#4C78A8",
        "SOCIAL": "#3A8A7A",
        "EMOTION": "#8C6BB1",
        "MOTOR": "#D17A52",
        "GAMBLING": "#B69B35",
        "RELATIONAL": "#687A8C",
        "WM": "#B55D6E",
    }
    rows = [row for task in payload["tasks"].values() for row in task["significant_rows"]]
    endpoint_best: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (row["task"], row["endpoint"])
        if key not in endpoint_best or row["p_max_t_task_family"] < endpoint_best[key]["p_max_t_task_family"]:
            endpoint_best[key] = row
    display = sorted(
        endpoint_best.values() if endpoint_best else winners,
        key=lambda row: (row["p_max_t_task_family"], -abs(row["rho"])),
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), layout="constrained", gridspec_kw={"width_ratios": [0.92, 1.25]})
    y = np.arange(len(winners))[::-1]
    axes[0].axvline(0, color="#D9DEE2", linewidth=0.7)
    for position, row in zip(y, winners, strict=True):
        lo, _, hi = row["bootstrap_95_ci"]
        axes[0].errorbar(
            row["rho"], position, xerr=[[row["rho"] - lo], [hi - row["rho"]]],
            fmt="o", color=task_colors[row["task"]], ecolor=task_colors[row["task"]],
            markersize=4.0, elinewidth=0.9, capsize=1.8,
        )
    axes[0].set_yticks(y, ["WM" if row["task"] == "WM" else row["task"].title() for row in winners])
    axes[0].set_xlabel(r"Adjusted $\rho$ (bootstrap 95% CI)")
    axes[0].set_title("a  Strongest association per task", loc="left", fontweight="bold")
    axes[0].grid(axis="x", color="#E9ECEF", linewidth=0.5)

    y = np.arange(len(display))[::-1]
    axes[1].axvline(0, color="#D9DEE2", linewidth=0.7)
    for position, row in zip(y, display, strict=True):
        axes[1].scatter(row["rho"], position, s=23, color=task_colors[row["task"]])
    labels = [
        f"{'WM' if row['task'] == 'WM' else row['task'].title()} · {row['endpoint']} · {compact(row['coalition'])}"
        for row in display
    ]
    axes[1].set_yticks(y, labels)
    for position, row in zip(y, display, strict=True):
        axes[1].text(
            0.02 if row["rho"] > 0 else 0.98,
            position,
            f"p={row['p_max_t_task_family']:.3g}",
            transform=axes[1].get_yaxis_transform(),
            ha="left" if row["rho"] > 0 else "right",
            va="center",
            fontsize=6.5,
            color="#4B535A",
        )
    axes[1].set_xlabel(r"Adjusted $\rho$")
    axes[1].set_title(
        "b  Strongest significant coalition per endpoint"
        if rows
        else "b  No task-family max-$T<0.05$",
        loc="left",
        fontweight="bold",
    )
    axes[1].grid(axis="x", color="#E9ECEF", linewidth=0.5)
    fig.savefig(FIGURE, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--bootstraps", type=int, default=5_000)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot_summary(json.loads(SUMMARY.read_text(encoding="utf-8")))
        print(FIGURE)
        return
    started = time.perf_counter()
    subjects = load_subjects()
    names, sizes = coalition_names()
    matrices, heldout = compute_matrices(subjects, names, sizes, args.recompute, args.tasks)
    endpoint_contracts, age, sex = load_endpoints(subjects)

    task_results: dict[str, Any] = {}
    winners: list[dict[str, Any]] = []
    for task_index, task in enumerate(args.tasks):
        specs = endpoint_contracts[task]
        rows, mask = screen_task(
            matrices[TASKS.index(task)], specs, age, sex,
            args.permutations, SEED + task_index,
        )
        for row in rows:
            row["coalition"] = str(names[row.pop("coalition_index")])
            row["order"] = int(row["coalition"].count("+") + 1)
            row["task"] = task
        winner = min(rows, key=lambda row: (row["p_max_t_task_family"], -abs(row["rho"])))
        winner_spec = specs[winner["endpoint_index"]]
        coalition_index = int(np.flatnonzero(names == winner["coalition"])[0])
        winner["bootstrap_95_ci"] = bootstrap_ci(
            matrices[TASKS.index(task), :, coalition_index], winner_spec,
            age, sex, mask, args.bootstraps, SEED + 100 + task_index,
        )
        winner["n"] = int(mask.sum())
        winner["endpoint_metadata"] = {
            key: value for key, value in winner_spec.items() if key not in ("values", "nuisance")
        }
        significant = [row for row in rows if row["p_max_t_task_family"] < 0.05]
        task_results[task] = {
            "n": int(mask.sum()),
            "endpoint_count": len(specs),
            "test_count": len(rows),
            "winner": winner,
            "significant_count": len(significant),
            "significant_rows": sorted(significant, key=lambda row: row["p_max_t_task_family"]),
            "heldout_skill_ratio_median": float(np.median(heldout[TASKS.index(task)])),
        }
        winners.append(winner)
        print(
            f"statistics {task}: n={mask.sum()}, endpoints={len(specs)}, "
            f"winner={winner['endpoint']} × {winner['coalition']}, "
            f"rho={winner['rho']:.4f}, max-T={winner['p_max_t_task_family']:.5g}",
            flush=True,
        )

    adjusted = holm([row["p_max_t_task_family"] for row in winners])
    for row, value in zip(winners, adjusted, strict=True):
        row["p_holm_across_task_winners"] = float(value)

    payload = {
        "experiment_contract": {
            "scientific_question": "Within each task, which same-task behavioral score and Yeo7 coalition order show the strongest association in the MMP360 1002-subject sample?",
            "subjects": 1002,
            "tasks": list(args.tasks),
            "endpoint_count": int(sum(len(endpoint_contracts[task]) for task in args.tasks)),
            "coalitions_per_endpoint": 120,
            "orders": [2, 3, 4, 5, 6, 7],
            "model": {"history_order": ORDER, "ridge_alpha": ALPHA, "estimator": "Gaussian linear TM/log-det"},
            "statistics": {
                "permutations": args.permutations,
                "bootstraps_for_task_winners": args.bootstraps,
                "within_task_family": "all declared endpoints x 120 coalitions",
                "across_tasks": "Holm correction of seven task-family winner p-values",
            },
            "scope_notes": {
                "same_task_only": True,
                "motor": "Out-of-scanner endurance, dexterity, strength, and their composite; no in-scanner MOTOR accuracy exists.",
                "gambling": "Choice fractions and response speeds, not correctness, because HCP outcomes are predetermined.",
            },
        },
        "data_checks": {
            "minimum_synergy_bits": float(np.nanmin(matrices)),
            "significant_nonnegativity_violation_count": int(np.sum(matrices < -SYN_TOLERANCE_BITS)),
            "syn_tolerance_bits": SYN_TOLERANCE_BITS,
        },
        "task_winners": winners,
        "tasks": task_results,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    atomic_json(SUMMARY, payload)
    plot_summary(payload)
    print(json.dumps({"summary": str(SUMMARY), "figure": str(FIGURE), "runtime_seconds": payload["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
