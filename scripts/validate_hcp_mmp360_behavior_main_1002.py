#!/usr/bin/env python3
"""Validate the behavior panels of the HCP main figure with MMP360 Yeo7 PC1 data."""

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
from scipy.stats import rankdata, spearmanr


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
FROZEN_57 = ROOT / "results/hcp_schaefer1000_task_evoked_xi_57/full/k1_p3_a1/arrays.npz"
OLD_MOTOR = ROOT / "results/hcp_motor_composite_scores_57/summary.json"
OUTPUT = ROOT / "results/hcp_mmp360_behavior_main_validation_1002"
CACHE = OUTPUT / "coalition_synergy_1002.npz"
SUMMARY = OUTPUT / "summary.json"
FIGURE = OUTPUT / "hcp_mmp360_behavior_main_validation_1002.png"

NETWORKS = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
SHORT = {
    "Vis": "Vis", "SomMot": "SM", "DorsAttn": "DAN", "SalVentAttn": "SVAN",
    "Limbic": "Lim", "Cont": "Cont", "Default": "DMN",
}
TASKS = ("LANGUAGE", "SOCIAL", "EMOTION", "MOTOR")
ORDER = 3
ALPHA = 1.0
SYN_TOLERANCE_BITS = 1.0e-9
SEED = 20260913

FIXED = (
    ("language_story", "LANGUAGE", "Vis+SomMot+Limbic+Cont", "story_accuracy", +0.258),
    ("language_math", "LANGUAGE", "Vis+DorsAttn+Cont", "math_accuracy", +0.281),
    ("social_dprime", "SOCIAL", "Vis+Limbic+Cont", "social_dprime", -0.464),
    ("emotion_face_speed", "EMOTION", "Limbic+Cont+Default", "face_speed", -0.323),
    ("motor_composite", "MOTOR", "Vis+SomMot+Limbic+Cont+Default", "motor_composite", -0.419),
)


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
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".json", mode="w", encoding="utf-8", delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def coalitions() -> tuple[tuple[str, ...], ...]:
    return tuple(c for size in range(2, 8) for c in itertools.combinations(NETWORKS, size))


def compact(value: str) -> str:
    return "+".join(SHORT[x] for x in value.split("+"))


def load_subjects() -> tuple[np.ndarray, np.ndarray]:
    subjects = np.asarray(sorted(path.name for path in DATA_ROOT.glob("sub-*")))
    if subjects.shape != (1002,) or len(set(subjects.tolist())) != 1002:
        raise ValueError(f"Expected 1002 unique subjects, found {subjects.shape}.")
    with np.load(FROZEN_57, allow_pickle=False) as archive:
        frozen = archive["subjects"].astype(str)
    missing = sorted(set(frozen) - set(subjects))
    if frozen.shape != (57,) or missing:
        raise ValueError(f"Frozen-57 mismatch; missing={missing}.")
    return subjects, frozen


def load_behavior(subjects: np.ndarray) -> dict[str, np.ndarray]:
    with BEHAVIOR.open(newline="", encoding="utf-8-sig") as handle:
        table = {str(row["Subject"]): row for row in csv.DictReader(handle)}
    rows = [table[str(s).removeprefix("sub-")] for s in subjects]
    value = lambda field: np.asarray([float(row[field]) for row in rows], dtype=float)
    age = np.asarray([age_midpoint(row["Age"]) for row in rows], dtype=float)
    sex = np.asarray([row["Gender"] == "M" for row in rows], dtype=float)

    trials = np.asarray([infer_effective_trials(row) for row in rows], dtype=int)
    hits = np.rint(value("Social_Task_TOM_Perc_TOM") * trials / 100.0)
    false_alarms = np.rint((100.0 - value("Social_Task_Random_Perc_Random")) * trials / 100.0)
    from scipy.stats import norm
    social_dprime = norm.ppf((hits + 0.5) / (trials + 1.0)) - norm.ppf(
        (false_alarms + 0.5) / (trials + 1.0)
    )

    motor_raw = np.column_stack([value("Endurance_AgeAdj"), value("Dexterity_AgeAdj"), value("Strength_AgeAdj")])
    motor_z = (motor_raw - motor_raw.mean(axis=0)) / motor_raw.std(axis=0, ddof=1)
    result = {
        "age": age,
        "sex": sex,
        "story_accuracy": value("Language_Task_Story_Acc"),
        "math_accuracy": value("Language_Task_Math_Acc"),
        "social_dprime": social_dprime,
        "face_speed": -np.log(value("Emotion_Task_Face_Median_RT")),
        "shape_speed": -np.log(value("Emotion_Task_Shape_Median_RT")),
        "motor_raw": motor_raw,
        "motor_composite": motor_z.mean(axis=1),
    }
    invalid = {key: int(np.sum(~np.isfinite(values))) for key, values in result.items()}
    if any(invalid.values()):
        raise ValueError(f"Non-finite behavior values: {invalid}")
    return result


def compute_matrices(subjects: np.ndarray, recompute: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    combos = coalitions()
    names = np.asarray(["+".join(c) for c in combos])
    sizes = np.asarray([len(c) for c in combos], dtype=np.int8)
    if CACHE.is_file() and not recompute:
        with np.load(CACHE, allow_pickle=False) as archive:
            if np.array_equal(archive["subjects"].astype(str), subjects) and np.array_equal(archive["coalitions"].astype(str), names):
                values = archive["synergy_bits"].astype(float)
                if values.shape == (4, 1002, 120) and np.isfinite(values).all():
                    return names, sizes, values, archive["heldout_skill_ratio"].astype(float)

    values = np.full((len(TASKS), len(subjects), len(combos)), np.nan, dtype=np.float64)
    heldout = np.full((len(TASKS), len(subjects)), np.nan, dtype=np.float64)
    indices = network_history_indices(NETWORKS, order=ORDER)
    started = time.perf_counter()
    for task_index, task in enumerate(TASKS):
        for subject_index, subject in enumerate(subjects):
            with_mat = loadmat(DATA_ROOT / subject / f"{task}_LR.mat")
            series = np.asarray(with_mat["Yeo7_taskRetainedPC1"], dtype=float)
            development_end = int(np.asarray(with_mat["development_end"]).squeeze())
            fitted = fit_delta_history_phi(series, alpha=ALPHA, order=ORDER, development_end=development_end)
            table = module_ei_table(fitted["transition"], fitted["noise_covariance"], indices, ridge=1.0e-6)
            singleton = {name: float(table[(name,)]) for name in NETWORKS}
            values[task_index, subject_index] = [subset_phi_raw(combo, table, singleton) for combo in combos]
            heldout[task_index, subject_index] = float(fitted["heldout"]["skill_ratio"])
        print(f"coalitions {task_index + 1}/{len(TASKS)} {task}: {time.perf_counter() - started:.1f}s", flush=True)
    violations = values < -SYN_TOLERANCE_BITS
    if np.any(violations):
        raise ValueError(
            f"Syn nonnegativity violation: min={values.min():.12g}, "
            f"threshold={-SYN_TOLERANCE_BITS:.1e}, count={int(violations.sum())}"
        )
    atomic_npz(
        CACHE, subjects=subjects, tasks=np.asarray(TASKS), coalitions=names,
        coalition_sizes=sizes, synergy_bits=values, heldout_skill_ratio=heldout,
        order=np.asarray(ORDER), alpha=np.asarray(ALPHA), syn_tolerance_bits=np.asarray(SYN_TOLERANCE_BITS),
    )
    return names, sizes, values, heldout


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def unit_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    values = values - values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(values, axis=0, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError("Constant residualized variable.")
    return values / norms


def bh(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(1.0, adjusted_ranked)
    return adjusted


def screen(
    matrix: np.ndarray,
    endpoint: np.ndarray,
    design: np.ndarray,
    permutations: int,
    seed: int,
    strata: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    brain = unit_columns(residualize(rankdata(matrix, axis=0), design))
    endpoint_rank = rankdata(endpoint)
    fitted = design @ np.linalg.lstsq(design, endpoint_rank, rcond=None)[0]
    endpoint_residual = residualize(endpoint_rank, design)
    endpoint_unit = unit_columns(endpoint_residual).ravel()
    observed = brain.T @ endpoint_unit
    point_counts = np.zeros(matrix.shape[1], dtype=np.int64)
    family_counts = np.zeros(matrix.shape[1], dtype=np.int64)
    groups = [np.arange(len(endpoint))] if strata is None else [np.flatnonzero(strata == x) for x in np.unique(strata)]
    rng = np.random.default_rng(seed)
    for start in range(0, permutations, 1000):
        size = min(1000, permutations - start)
        indices = np.tile(np.arange(len(endpoint)), (size, 1))
        for group in groups:
            indices[:, group] = group[np.argsort(rng.random((size, len(group))), axis=1)]
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
        "q_bh_120": bh(p_raw),
        "p_max_t_120": (family_counts + 1.0) / denominator,
    }


def bootstrap(
    brain: np.ndarray,
    endpoint: np.ndarray,
    covariates: Sequence[tuple[np.ndarray, bool]],
    repeats: int,
    seed: int,
    strata: np.ndarray | None = None,
) -> list[float]:
    rng = np.random.default_rng(seed)
    groups = [np.arange(len(endpoint))] if strata is None else [np.flatnonzero(strata == x) for x in np.unique(strata)]
    estimates = np.full(repeats, np.nan)
    for repeat in range(repeats):
        sample = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        parts = [np.ones(len(sample))] + [rankdata(value[sample]) if rerank else value[sample] for value, rerank in covariates]
        design = np.column_stack(parts)
        x = residualize(rankdata(brain[sample]), design)
        y = residualize(rankdata(endpoint[sample]), design)
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        if denominator > 1.0e-12:
            estimates[repeat] = float(x @ y / denominator)
    return np.nanquantile(estimates, [0.025, 0.5, 0.975]).tolist()


def design_for(
    endpoint_name: str,
    behavior: Mapping[str, np.ndarray],
    take: np.ndarray,
    cohort: np.ndarray | None,
) -> tuple[np.ndarray, list[tuple[np.ndarray, bool]]]:
    age = behavior["age"][take]
    sex = behavior["sex"][take]
    if endpoint_name in ("story_accuracy", "math_accuracy"):
        columns: list[np.ndarray] = []
        bootstrap_columns: list[tuple[np.ndarray, bool]] = []
    elif endpoint_name == "social_dprime":
        columns = [rankdata(age), sex]
        bootstrap_columns = [(age, True), (sex, False)]
    elif endpoint_name == "face_speed":
        shape = behavior["shape_speed"][take]
        columns = [rankdata(age), sex]
        bootstrap_columns = [(rankdata(age), False), (sex, False)]
        if cohort is not None:
            columns.append(cohort.astype(float))
            bootstrap_columns.append((cohort.astype(float), False))
        columns.append(rankdata(shape))
        bootstrap_columns.append((rankdata(shape), False))
    elif endpoint_name == "motor_composite":
        columns = [age, sex]
        bootstrap_columns = [(age, False), (sex, False)]
        if cohort is not None:
            columns.append(cohort.astype(float))
            bootstrap_columns.append((cohort.astype(float), False))
    else:
        raise KeyError(endpoint_name)
    return np.column_stack([np.ones(len(take))] + columns), bootstrap_columns


def endpoint_for(endpoint_name: str, behavior: Mapping[str, np.ndarray], take: np.ndarray) -> np.ndarray:
    if endpoint_name != "motor_composite":
        return np.asarray(behavior[endpoint_name][take], dtype=float)
    raw = np.asarray(behavior["motor_raw"][take], dtype=float)
    standardized = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=1)
    return standardized.mean(axis=1)


def analyze_sample(
    label: str,
    subjects: np.ndarray,
    take: np.ndarray,
    names: np.ndarray,
    matrices: np.ndarray,
    behavior: Mapping[str, np.ndarray],
    permutations: int,
    bootstraps: int,
    cohort: np.ndarray | None,
) -> dict[str, Any]:
    task_index = {task: index for index, task in enumerate(TASKS)}
    screens: dict[str, dict[str, np.ndarray]] = {}
    endpoint_specs = (
        ("language_story", "LANGUAGE", "story_accuracy"),
        ("language_math", "LANGUAGE", "math_accuracy"),
        ("social", "SOCIAL", "social_dprime"),
        ("emotion", "EMOTION", "face_speed"),
        ("motor", "MOTOR", "motor_composite"),
    )
    for screen_index, (key, task, endpoint_name) in enumerate(endpoint_specs):
        design, _ = design_for(endpoint_name, behavior, take, cohort)
        endpoint = endpoint_for(endpoint_name, behavior, take)
        screens[key] = screen(
            matrices[task_index[task], take], endpoint, design,
            permutations, SEED + screen_index + (0 if label == "paired_57_mmp360" else 1000), cohort,
        )
        print(f"statistics {label} {key}", flush=True)

    fixed_rows: dict[str, Any] = {}
    for fixed_index, (key, task, coalition, endpoint_name, old_rho) in enumerate(FIXED):
        screen_key = {
            "story_accuracy": "language_story", "math_accuracy": "language_math",
            "social_dprime": "social", "face_speed": "emotion", "motor_composite": "motor",
        }[endpoint_name]
        index = int(np.flatnonzero(names == coalition)[0])
        result = screens[screen_key]
        _, covariates = design_for(endpoint_name, behavior, take, cohort)
        endpoint = endpoint_for(endpoint_name, behavior, take)
        interval = bootstrap(
            matrices[task_index[task], take, index], endpoint, covariates,
            bootstraps, SEED + 100 + fixed_index + (0 if label == "paired_57_mmp360" else 1000), cohort,
        )
        rho = float(result["rho"][index])
        fixed_rows[key] = {
            "task": task, "endpoint": endpoint_name, "coalition": coalition,
            "rho": rho, "bootstrap_quantiles": interval,
            "p_raw": float(result["p_raw"][index]),
            "q_bh_120": float(result["q_bh_120"][index]),
            "p_max_t_120": float(result["p_max_t_120"][index]),
            "same_direction_as_original": bool(np.sign(rho) == np.sign(old_rho)),
            "nominally_reproduced": bool(np.sign(rho) == np.sign(old_rho) and result["p_raw"][index] < 0.05),
            "selection_corrected_reproduced": bool(np.sign(rho) == np.sign(old_rho) and result["p_max_t_120"][index] < 0.05),
        }

    winners = {}
    for key, task, _ in endpoint_specs:
        result = screens[key]
        winner = int(np.argmax(np.abs(result["rho"])))
        winners[key] = {
            "coalition": str(names[winner]), "rho": float(result["rho"][winner]),
            "p_raw": float(result["p_raw"][winner]), "p_max_t_120": float(result["p_max_t_120"][winner]),
        }
    return {"n": int(len(take)), "fixed_candidates": fixed_rows, "winners": winners, "screens": screens}


def motor_rows(
    old_summary: Mapping[str, Any], sample: Mapping[str, Any], names: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    result = sample["screens"]["motor"]
    for old in old_summary["top_ten"]:
        if float(old["p_raw"]) >= 0.05:
            continue
        index = int(np.flatnonzero(names == old["coalition"])[0])
        rows.append({
            "coalition": old["coalition"], "old_rho": float(old["rho_adjusted"]),
            "rho": float(result["rho"][index]), "p_raw": float(result["p_raw"][index]),
            "p_max_t_120": float(result["p_max_t_120"][index]),
        })
    return rows


def plot_validation(summary: Mapping[str, Any]) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "legend.frameon": False,
    })
    labels = ["Language: Story", "Language: Math", "Social: d'", "Emotion: face speed", "Motor: composite"]
    keys = [row[0] for row in FIXED]
    old = np.asarray([row[4] for row in FIXED])
    paired = summary["samples"]["paired_57_mmp360"]["fixed_candidates"]
    full = summary["samples"]["full_1002_mmp360"]["fixed_candidates"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.65), layout="constrained", gridspec_kw={"width_ratios": [1.03, 1.25]})
    ax = axes[0]
    y = np.arange(len(keys))[::-1]
    ax.axvline(0, color="#D8DDE1", linewidth=0.8)
    ax.scatter(old, y + 0.20, marker="s", s=25, color="#A6AFB7", label="Original 57 · Schaefer1000")
    for offset, data, color, marker, legend in (
        (0.00, paired, "#D17A52", "D", "Paired 57 · MMP360"),
        (-0.20, full, "#397A70", "o", "Full 1002 · MMP360"),
    ):
        centers = np.asarray([data[key]["rho"] for key in keys])
        intervals = np.asarray([data[key]["bootstrap_quantiles"] for key in keys])
        ax.errorbar(centers, y + offset, xerr=np.vstack([centers - intervals[:, 0], intervals[:, 2] - centers]),
                    fmt=marker, color=color, ecolor=color, alpha=0.95, elinewidth=0.85,
                    capsize=1.8, markersize=4.2, label=legend)
    ax.set_yticks(y, labels)
    ax.set_xlabel(r"Rank association $\rho$ (bootstrap 95% CI)")
    ax.set_title("a  Fixed main-figure conclusions", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E8EBED", linewidth=0.55)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=1)

    ax = axes[1]
    motor = summary["motor_original_top9"]
    y = np.arange(len(motor))[::-1]
    ax.axvline(0, color="#D8DDE1", linewidth=0.8)
    ax.scatter([row["old_rho"] for row in motor], y + 0.14, marker="s", s=23, color="#A6AFB7", label="Original 57 · Schaefer1000")
    ax.scatter([row["paired_rho"] for row in motor], y, marker="D", s=22, color="#D17A52", label="Paired 57 · MMP360")
    ax.scatter([row["full_rho"] for row in motor], y - 0.14, marker="o", s=23, color="#397A70", label="Full 1002 · MMP360")
    ax.set_yticks(y, [compact(row["coalition"]) for row in motor])
    ax.set_xlabel(r"Age/sex-adjusted $\rho$")
    ax.set_title("b  Original MOTOR forest candidates", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E8EBED", linewidth=0.55)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=1)
    fig.savefig(FIGURE, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--bootstraps", type=int, default=20_000)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    subjects, frozen = load_subjects()
    behavior = load_behavior(subjects)
    names, sizes, matrices, heldout = compute_matrices(subjects, args.recompute)
    position = {subject: index for index, subject in enumerate(subjects)}
    take57 = np.asarray([position[subject] for subject in frozen], dtype=int)
    cohort57 = np.r_[np.zeros(29, dtype=int), np.ones(28, dtype=int)]
    paired = analyze_sample("paired_57_mmp360", subjects, take57, names, matrices, behavior, args.permutations, args.bootstraps, cohort57)
    full = analyze_sample("full_1002_mmp360", subjects, np.arange(len(subjects)), names, matrices, behavior, args.permutations, args.bootstraps, None)

    old_motor = json.loads(OLD_MOTOR.read_text(encoding="utf-8"))
    paired_motor = {row["coalition"]: row for row in motor_rows(old_motor, paired, names)}
    full_motor = {row["coalition"]: row for row in motor_rows(old_motor, full, names)}
    motor = [{
        "coalition": name, "old_rho": paired_motor[name]["old_rho"],
        "paired_rho": paired_motor[name]["rho"], "paired_p_raw": paired_motor[name]["p_raw"],
        "full_rho": full_motor[name]["rho"], "full_p_raw": full_motor[name]["p_raw"],
        "full_p_max_t_120": full_motor[name]["p_max_t_120"],
    } for name in paired_motor]

    def clean(sample: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in sample.items() if key != "screens"}

    payload = {
        "experiment_contract": {
            "scientific_question": "Do the fixed main-figure behavior associations retain direction and evidence under MMP360, first in the paired 57 and then in all 1002 score-complete subjects?",
            "comparison_levels": ["original_57_schaefer1000", "paired_57_mmp360", "full_1002_mmp360"],
            "model": {"order": ORDER, "ridge_alpha": ALPHA, "development_fraction": 0.75, "estimator": "Gaussian linear TM/log-det"},
            "statistics": {"permutations": args.permutations, "bootstraps": args.bootstraps, "family": "120 fixed Yeo7 coalitions", "syn_tolerance_bits": SYN_TOLERANCE_BITS},
            "limitation": "The 57-to-1002 contrast changes sample composition and size; it is an external-validation contrast, not a single-factor causal comparison.",
        },
        "data_checks": {
            "subjects": int(len(subjects)), "tasks": list(TASKS), "mat_files": int(len(subjects) * len(TASKS)),
            "minimum_synergy_bits": float(matrices.min()),
            "significant_nonnegativity_violation_count": int(np.sum(matrices < -SYN_TOLERANCE_BITS)),
            "heldout_skill_ratio_median_by_task": {task: float(np.nanmedian(heldout[index])) for index, task in enumerate(TASKS)},
        },
        "samples": {"paired_57_mmp360": clean(paired), "full_1002_mmp360": clean(full)},
        "motor_original_top9": motor,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    atomic_json(SUMMARY, payload)
    plot_validation(payload)
    print(json.dumps({"summary": str(SUMMARY), "figure": str(FIGURE), "runtime_seconds": payload["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
