#!/usr/bin/env python3
"""Screen fixed Yeo-7 EMOTION coalitions against task performance in 57 HCP subjects."""

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

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TASK_ROOT = ROOT / "data/hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_57_brain"
REST_ROOT = ROOT / "data/hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_57_brain"
LABELS = ROOT / "data/hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30/_atlas_labels/Schaefer2018_1000Parcels_7Networks_order.txt"
BEHAVIOR = ROOT / "data/unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
SUBJECT_SOURCE = ROOT / "results/hcp_schaefer1000_task_evoked_xi_57/full/k1_p3_a1/arrays.npz"
OUTPUT = ROOT / "results/hcp_emotion_performance_coalitions_57"
CACHE = OUTPUT / "emotion_rest_coalition_synergy_57.npz"
PARTIAL = OUTPUT / "emotion_rest_coalition_synergy_57.partial.npz"
SENSITIVITY_CACHE = OUTPUT / "emotion_rest_coalition_synergy_57_p5_a10.npz"
SENSITIVITY_PARTIAL = OUTPUT / "emotion_rest_coalition_synergy_57_p5_a10.partial.npz"
STATUS = OUTPUT / "live_progress.json"

NETWORK_ORDER = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
PREDEFINED = (
    "Vis+Limbic",
    "Vis+SalVentAttn+Limbic",
    "Vis+Limbic+Cont",
    "Vis+DorsAttn+Limbic",
    "SalVentAttn+Limbic+Cont",
    "Vis+DorsAttn+Cont",
)
ORDER = 3
ALPHA = 1.0
SYN_TOLERANCE_BITS = 1.0e-9
SEED = 20260802


def import_project_functions() -> tuple[Any, ...]:
    from scripts.analyze_hcp_task_evoked_pc2_xi_hierarchy import network_module_indices
    from scripts.phi_hierarchy import subset_phi_raw
    from scripts.run_hcp_schaefer500_yeo7_module_phi_decomposition import module_ei_table
    from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import load_yeo7_groups
    from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi
    from scripts.tune_hcp_task_evoked_xi_hierarchy import prepare_projection

    return (
        network_module_indices,
        subset_phi_raw,
        module_ei_table,
        load_yeo7_groups,
        fit_delta_history_phi,
        prepare_projection,
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
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def status(phase: str, completed: int, total: int, message: str, state: str = "running") -> None:
    atomic_json(
        STATUS,
        {
            "state": state,
            "phase": phase,
            "completed": int(completed),
            "total": int(total),
            "message": message,
            "updated_unix": time.time(),
        },
    )


def coalitions() -> tuple[tuple[str, ...], ...]:
    return tuple(combo for size in range(2, 8) for combo in itertools.combinations(NETWORK_ORDER, size))


def coalition_names(values: Sequence[Sequence[str]]) -> np.ndarray:
    return np.asarray(["+".join(value) for value in values])


def load_subjects() -> np.ndarray:
    with np.load(SUBJECT_SOURCE, allow_pickle=False) as archive:
        subjects = archive["subjects"].astype(str)
    if subjects.shape != (57,) or len(set(subjects.tolist())) != 57:
        raise ValueError("Expected the frozen 57-subject order.")
    return subjects


def rest_path(subject: str) -> Path:
    matches = sorted((REST_ROOT / subject).glob("*REST1_LR*.mat"))
    if len(matches) != 1:
        raise ValueError(f"Expected one REST1_LR file for {subject}, found {len(matches)}.")
    return matches[0]


def compute_subject_state(
    subject: str,
    state_name: str,
    groups: Mapping[str, Sequence[int]],
    combos: Sequence[tuple[str, ...]],
    functions: Sequence[Any],
    order: int = ORDER,
    alpha: float = ALPHA,
) -> tuple[np.ndarray, float, float]:
    network_module_indices, subset_phi_raw, module_ei_table, _, fit_delta_history_phi, prepare_projection = functions
    if state_name == "EMOTION":
        projections, explained, development_end = prepare_projection(
            TASK_ROOT / subject / "EMOTION_LR.mat",
            groups,
            state="EMOTION",
            max_components=1,
            task_retained_key="Schaefer1000_taskRetained",
            task_regressed_key="Schaefer1000_taskRegressed",
            expected_parcels=1000,
        )
    elif state_name == "REST":
        projections, explained, development_end = prepare_projection(
            rest_path(subject),
            groups,
            state="REST",
            max_components=1,
            rest_data_key="Schaefer1000",
            expected_parcels=1000,
        )
    else:
        raise ValueError(state_name)
    fitted = fit_delta_history_phi(projections[1], alpha=alpha, order=order, development_end=development_end)
    indices = network_module_indices(NETWORK_ORDER, n_components=1, order=order)
    table = module_ei_table(fitted["transition"], fitted["noise_covariance"], indices, ridge=1.0e-6)
    singleton = {name: float(table[(name,)]) for name in NETWORK_ORDER}
    values = np.asarray([subset_phi_raw(combo, table, singleton) for combo in combos], dtype=float)
    return values, float(fitted["heldout"]["skill_ratio"]), float(np.mean(list(explained[1].values())))


def smoke_test(subjects: np.ndarray, combos: Sequence[tuple[str, ...]]) -> None:
    functions = import_project_functions()
    groups = functions[3](LABELS, expected_parcels=1000)
    for state_name in ("EMOTION", "REST"):
        values, heldout, explained = compute_subject_state(str(subjects[0]), state_name, groups, combos, functions)
        if values.shape != (120,) or not np.isfinite(values).all():
            raise ValueError(f"{state_name} smoke test returned invalid coalition values.")
        violations = values < -SYN_TOLERANCE_BITS
        if np.any(violations):
            raise ValueError(
                f"{state_name} smoke-test Syn violation: min={values.min():.12g}, "
                f"threshold={-SYN_TOLERANCE_BITS:.12g}, count={int(violations.sum())}"
            )
        print(
            f"smoke {state_name}: subject={subjects[0]}, min={values.min():.6g}, "
            f"max={values.max():.6g}, heldout={heldout:.4f}, pca={explained:.4f}",
            flush=True,
        )


def compute_matrices(
    subjects: np.ndarray,
    combos: Sequence[tuple[str, ...]],
    recompute: bool,
    *,
    order: int = ORDER,
    alpha: float = ALPHA,
    cache_path: Path = CACHE,
    partial_path: Path = PARTIAL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    names = coalition_names(combos)
    states = np.asarray(["EMOTION", "REST"])
    if cache_path.is_file() and not recompute:
        with np.load(cache_path, allow_pickle=False) as archive:
            if np.array_equal(archive["subjects"].astype(str), subjects) and np.array_equal(archive["coalitions"].astype(str), names):
                values = archive["synergy_bits"].astype(float)
                if values.shape == (2, 57, 120) and np.isfinite(values).all():
                    audit = json.loads(str(archive["nonnegativity_audit_json"].item()))
                    return values, archive["heldout_skill_ratio"].astype(float), archive["mean_pc1_explained"].astype(float), audit

    values = np.full((2, 57, 120), np.nan)
    heldout = np.full((2, 57), np.nan)
    explained = np.full((2, 57), np.nan)
    if partial_path.is_file() and not recompute:
        with np.load(partial_path, allow_pickle=False) as archive:
            if np.array_equal(archive["subjects"].astype(str), subjects) and np.array_equal(archive["coalitions"].astype(str), names):
                values = archive["synergy_bits"].astype(float)
                heldout = archive["heldout_skill_ratio"].astype(float)
                explained = archive["mean_pc1_explained"].astype(float)

    functions = import_project_functions()
    groups = functions[3](LABELS, expected_parcels=1000)
    total = 2 * len(subjects)
    for state_index, state_name in enumerate(states):
        for subject_index, subject in enumerate(subjects):
            if np.isfinite(values[state_index, subject_index]).all():
                continue
            row, skill, pca = compute_subject_state(
                str(subject), str(state_name), groups, combos, functions, order=order, alpha=alpha
            )
            values[state_index, subject_index] = row
            heldout[state_index, subject_index] = skill
            explained[state_index, subject_index] = pca
            atomic_npz(
                partial_path,
                states=states,
                subjects=subjects,
                coalitions=names,
                synergy_bits=values,
                heldout_skill_ratio=heldout,
                mean_pc1_explained=explained,
            )
            done = int(np.isfinite(values).all(axis=2).sum())
            status("coalition_recomputation", done, total, f"Completed {state_name} {subject}")
            print(f"[{done:03d}/{total}] {state_name} {subject}", flush=True)
    if not np.isfinite(values).all():
        raise ValueError("Coalition cache is incomplete after recomputation.")

    violations = values < -SYN_TOLERANCE_BITS
    if np.any(violations):
        raise ValueError(
            f"PEID Syn nonnegativity violation: min={values.min():.12g}, "
            f"threshold={-SYN_TOLERANCE_BITS:.12g}, count={int(violations.sum())}"
        )
    tolerance_mask = (values < 0.0) & ~violations
    audit = {
        "tolerance_bits": SYN_TOLERANCE_BITS,
        "checked_count": int(values.size),
        "minimum_bits": float(values.min()),
        "within_tolerance_negative_count": int(tolerance_mask.sum()),
        "significant_violation_count": 0,
    }
    if np.any(tolerance_mask):
        values = values.copy()
        values[tolerance_mask] = 0.0
    atomic_npz(
        cache_path,
        states=states,
        subjects=subjects,
        coalitions=names,
        coalition_sizes=np.asarray([len(combo) for combo in combos]),
        synergy_bits=values,
        heldout_skill_ratio=heldout,
        mean_pc1_explained=explained,
        syn_tolerance_bits=np.asarray(SYN_TOLERANCE_BITS),
        nonnegativity_audit_json=np.asarray(json.dumps(audit)),
    )
    partial_path.unlink(missing_ok=True)
    return values, heldout, explained, audit


def age_midpoint(value: str) -> float:
    if value == "36+":
        return 38.0
    low, high = value.split("-")
    return 0.5 * (float(low) + float(high))


def load_behavior(subjects: np.ndarray) -> dict[str, np.ndarray]:
    with BEHAVIOR.open(newline="", encoding="utf-8-sig") as handle:
        table = {str(row["Subject"]): row for row in csv.DictReader(handle)}
    rows = [table[str(subject).removeprefix("sub-")] for subject in subjects]
    fields = (
        "Emotion_Task_Acc",
        "Emotion_Task_Median_RT",
        "Emotion_Task_Face_Acc",
        "Emotion_Task_Face_Median_RT",
        "Emotion_Task_Shape_Acc",
        "Emotion_Task_Shape_Median_RT",
    )
    result = {field: np.asarray([float(row[field]) for row in rows]) for field in fields}
    result.update(
        age=np.asarray([age_midpoint(row["Age"]) for row in rows]),
        sex=np.asarray([row["Gender"] == "M" for row in rows], dtype=float),
        cohort=np.r_[np.zeros(29, dtype=int), np.ones(28, dtype=int)],
    )
    if any(not np.isfinite(values).all() for values in result.values()):
        raise ValueError("Behavior or covariate data contain missing/nonfinite values.")
    return result


def projection_residual(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def unit_columns(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    array = array - array.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(array, axis=0, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError("A residualized variable is constant.")
    return array / norms


def bh(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=float).ravel()
    order = np.argsort(flat)
    ranked = flat[order]
    adjusted_ranked = np.minimum.accumulate((ranked * len(flat) / np.arange(1, len(flat) + 1))[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(1.0, adjusted_ranked)
    return adjusted.reshape(np.asarray(values).shape)


def prepare_analyses(synergy: np.ndarray, behavior: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    cohort = behavior["cohort"].astype(float)
    base = np.column_stack(
        [np.ones(len(cohort)), rankdata(behavior["age"]), behavior["sex"], cohort]
    )
    face_speed = -np.log(behavior["Emotion_Task_Face_Median_RT"])
    shape_speed = -np.log(behavior["Emotion_Task_Shape_Median_RT"])
    overall_speed = -np.log(behavior["Emotion_Task_Median_RT"])
    face_efficiency = -np.log(
        behavior["Emotion_Task_Face_Median_RT"] / (behavior["Emotion_Task_Face_Acc"] / 100.0)
    )
    shape_efficiency = -np.log(
        behavior["Emotion_Task_Shape_Median_RT"] / (behavior["Emotion_Task_Shape_Acc"] / 100.0)
    )
    primary_design = np.column_stack([base, rankdata(shape_speed)])
    efficiency_design = np.column_stack([base, rankdata(shape_efficiency)])
    task = synergy[0]
    delta = synergy[0] - synergy[1]
    definitions = (
        ("task_face_specific_speed", task, face_speed, primary_design, "EMOTION Syn vs face speed controlling shape speed"),
        ("delta_face_specific_speed", delta, face_speed, primary_design, "EMOTION-minus-REST Syn vs face speed controlling shape speed"),
        ("task_overall_speed", task, overall_speed, base, "EMOTION Syn vs overall speed"),
        ("task_overall_accuracy", task, behavior["Emotion_Task_Acc"], base, "EMOTION Syn vs overall accuracy"),
        ("task_face_efficiency", task, face_efficiency, efficiency_design, "EMOTION Syn vs face efficiency controlling shape efficiency"),
    )
    analyses: list[dict[str, Any]] = []
    for name, matrix, endpoint, design, label in definitions:
        brain_rank = rankdata(matrix, axis=0, method="average")
        endpoint_rank = rankdata(endpoint, method="average")
        brain_unit = unit_columns(projection_residual(brain_rank, design))
        fitted = design @ np.linalg.lstsq(design, endpoint_rank, rcond=None)[0]
        endpoint_residual = projection_residual(endpoint_rank, design)
        endpoint_unit = unit_columns(endpoint_residual).ravel()
        hat_residual = np.eye(len(endpoint_rank)) - design @ np.linalg.pinv(design)
        analyses.append(
            {
                "name": name,
                "label": label,
                "matrix": matrix,
                "endpoint": endpoint,
                "design": design,
                "brain_unit": brain_unit,
                "fitted": fitted,
                "endpoint_residual": endpoint_residual,
                "endpoint_unit": endpoint_unit,
                "residual_maker": hat_residual,
                "observed": brain_unit.T @ endpoint_unit,
            }
        )
    return analyses


def screen_analyses(
    analyses: Sequence[dict[str, Any]],
    cohort: np.ndarray,
    predefined_indices: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, np.ndarray]:
    n_analyses = len(analyses)
    n_coalitions = analyses[0]["brain_unit"].shape[1]
    observed = np.vstack([analysis["observed"] for analysis in analyses])
    raw_counts = np.zeros((n_analyses, n_coalitions), dtype=np.int64)
    family_counts = np.zeros_like(raw_counts)
    predefined_counts = np.zeros_like(raw_counts)
    global_counts = np.zeros_like(raw_counts)
    groups = [np.flatnonzero(cohort == value) for value in np.unique(cohort)]
    rng = np.random.default_rng(seed)
    chunk = 500
    for start in range(0, permutations, chunk):
        size = min(chunk, permutations - start)
        permuted = np.tile(np.arange(len(cohort)), (size, 1))
        for group in groups:
            order = np.argsort(rng.random((size, len(group))), axis=1)
            permuted[:, group] = group[order]
        null_blocks: list[np.ndarray] = []
        for analysis_index, analysis in enumerate(analyses):
            pseudo = analysis["fitted"][None, :] + analysis["endpoint_residual"][permuted]
            residual = pseudo @ analysis["residual_maker"].T
            residual /= np.linalg.norm(residual, axis=1, keepdims=True)
            absolute = np.abs(residual @ analysis["brain_unit"])
            null_blocks.append(absolute)
            raw_counts[analysis_index] += np.sum(
                absolute >= np.abs(observed[analysis_index])[None, :], axis=0
            )
            family_max = absolute.max(axis=1)
            predefined_max = absolute[:, predefined_indices].max(axis=1)
            family_counts[analysis_index] += np.sum(
                family_max[:, None] >= np.abs(observed[analysis_index])[None, :], axis=0
            )
            predefined_counts[analysis_index] += np.sum(
                predefined_max[:, None] >= np.abs(observed[analysis_index])[None, :], axis=0
            )
        global_max = np.maximum.reduce([block.max(axis=1) for block in null_blocks])
        for analysis_index in range(n_analyses):
            global_counts[analysis_index] += np.sum(
                global_max[:, None] >= np.abs(observed[analysis_index])[None, :], axis=0
            )
        status("permutation", min(start + size, permutations), permutations, "Cohort-blocked Freedman-Lane max-T")
    denominator = permutations + 1.0
    p_raw = (raw_counts + 1.0) / denominator
    return {
        "observed": observed,
        "p_raw": p_raw,
        "q_bh_within_analysis": np.vstack([bh(row) for row in p_raw]),
        "p_max_t_120": (family_counts + 1.0) / denominator,
        "p_max_t_predefined6": (predefined_counts + 1.0) / denominator,
        "p_max_t_global600": (global_counts + 1.0) / denominator,
    }


def partial_rank_correlation(x: np.ndarray, y: np.ndarray, design: np.ndarray) -> float:
    xr = projection_residual(rankdata(x), design)
    yr = projection_residual(rankdata(y), design)
    return float(xr @ yr / np.sqrt((xr @ xr) * (yr @ yr)))


def bootstrap_interval(
    matrix: np.ndarray,
    endpoint: np.ndarray,
    design: np.ndarray,
    cohort: np.ndarray,
    candidate_indices: Sequence[int],
    bootstraps: int,
    seed: int,
) -> dict[int, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(cohort == value) for value in np.unique(cohort)]
    samples = np.empty((bootstraps, len(candidate_indices)), dtype=float)
    for bootstrap in range(bootstraps):
        sample = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        local_design = design[sample]
        for position, candidate in enumerate(candidate_indices):
            samples[bootstrap, position] = partial_rank_correlation(
                matrix[sample, candidate], endpoint[sample], local_design
            )
    bounds = np.quantile(samples, [0.025, 0.975], axis=0)
    return {int(candidate): (float(bounds[0, index]), float(bounds[1, index])) for index, candidate in enumerate(candidate_indices)}


def cohort_correlations(analysis: Mapping[str, Any], candidate: int, cohort: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for cohort_value, label in ((0, "original_29"), (1, "supplementary_28")):
        mask = cohort == cohort_value
        local_design = analysis["design"][mask]
        keep = np.ptp(local_design, axis=0) > 1.0e-12
        keep[0] = True
        result[label] = partial_rank_correlation(
            analysis["matrix"][mask, candidate],
            analysis["endpoint"][mask],
            local_design[:, keep],
        )
    return result


def behavior_summary(behavior: Mapping[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in (
        "Emotion_Task_Acc",
        "Emotion_Task_Median_RT",
        "Emotion_Task_Face_Acc",
        "Emotion_Task_Face_Median_RT",
        "Emotion_Task_Shape_Acc",
        "Emotion_Task_Shape_Median_RT",
    ):
        values = behavior[field]
        result[field] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "maximum": float(values.max()),
            "unique_count": int(len(np.unique(values))),
            "maximum_count": int(np.sum(values == values.max())),
        }
    return result


def compact_name(name: str) -> str:
    replacements = {"SomMot": "Som", "DorsAttn": "DAN", "SalVentAttn": "SVAN", "Limbic": "Lim", "Default": "DMN"}
    return "+".join(replacements.get(part, part) for part in name.split("+"))


def make_figure(
    subjects: np.ndarray,
    names: np.ndarray,
    analyses: Sequence[dict[str, Any]],
    screen: Mapping[str, np.ndarray],
    behavior: Mapping[str, np.ndarray],
    intervals: Mapping[int, tuple[float, float]],
    winner: int,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    colors = np.asarray(["#617A9A", "#D17A55"])[behavior["cohort"].astype(int)]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), constrained_layout=True)
    ax = axes[0, 0]
    face_rt = behavior["Emotion_Task_Face_Median_RT"]
    shape_rt = behavior["Emotion_Task_Shape_Median_RT"]
    for cohort_value, label, color in ((0, "Original 29", "#617A9A"), (1, "Supplementary 28", "#D17A55")):
        mask = behavior["cohort"] == cohort_value
        ax.scatter(shape_rt[mask], face_rt[mask], s=19, alpha=0.84, color=color, edgecolor="white", linewidth=0.35, label=label)
    lower = min(float(shape_rt.min()), float(face_rt.min()))
    upper = max(float(shape_rt.max()), float(face_rt.max()))
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="#8A8A8A", linewidth=0.8)
    ax.set(xlabel="Shape median RT (ms)", ylabel="Face median RT (ms)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.20), ncol=2)

    ax = axes[0, 1]
    primary = screen["observed"][0]
    top = np.argsort(-np.abs(primary))[:10][::-1]
    y = np.arange(len(top))
    low = np.asarray([intervals[int(index)][0] for index in top])
    high = np.asarray([intervals[int(index)][1] for index in top])
    ax.errorbar(primary[top], y, xerr=np.vstack([primary[top] - low, high - primary[top]]), fmt="o", color="#345E7D", ecolor="#9EB5C8", capsize=2.2, markersize=4.0)
    ax.axvline(0.0, color="#8A8A8A", linewidth=0.8)
    ax.set_yticks(y, [compact_name(str(names[index])) for index in top])
    ax.set_xlabel("Adjusted rank correlation, ρ")

    ax = axes[1, 0]
    delta = screen["observed"][1]
    sizes = np.asarray([name.count("+") + 1 for name in names])
    scatter = ax.scatter(primary, delta, c=sizes, cmap="viridis", s=20, alpha=0.78, edgecolor="none")
    ax.scatter(primary[winner], delta[winner], s=54, facecolor="none", edgecolor="#B33A3A", linewidth=1.2)
    ax.axhline(0.0, color="#AAAAAA", linewidth=0.7)
    ax.axvline(0.0, color="#AAAAAA", linewidth=0.7)
    ax.set(xlabel="EMOTION Syn: adjusted ρ", ylabel="EMOTION − REST Syn: adjusted ρ")
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.047, pad=0.03)
    colorbar.set_label("Coalition size")

    ax = axes[1, 1]
    analysis = analyses[0]
    x = projection_residual(rankdata(analysis["matrix"][:, winner]), analysis["design"])
    y_residual = projection_residual(rankdata(analysis["endpoint"]), analysis["design"])
    for cohort_value, color in ((0, "#617A9A"), (1, "#D17A55")):
        mask = behavior["cohort"] == cohort_value
        ax.scatter(x[mask], y_residual[mask], s=20, alpha=0.84, color=color, edgecolor="white", linewidth=0.35)
    coefficient = np.polyfit(x, y_residual, 1)
    grid = np.linspace(float(x.min()), float(x.max()), 100)
    ax.plot(grid, np.polyval(coefficient, grid), color="#333333", linewidth=1.0)
    p120 = screen["p_max_t_120"][0, winner]
    ax.text(
        0.0,
        1.03,
        f"{compact_name(str(names[winner]))}: ρ={primary[winner]:+.3f}, 120-combination max-T p={p120:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        clip_on=False,
    )
    ax.set(xlabel="Residualized rank Syn", ylabel="Residualized face-speed rank")

    for letter, axis in zip("abcd", axes.ravel()):
        axis.text(-0.15, 1.07, letter, transform=axis.transAxes, fontweight="bold", fontsize=8, va="top")
    stem = OUTPUT / "emotion_performance_coalition_screen_57"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def serializable_candidate(
    analysis_index: int,
    coalition_index: int,
    names: np.ndarray,
    screen: Mapping[str, np.ndarray],
    intervals: Mapping[tuple[int, int], tuple[float, float]],
) -> dict[str, Any]:
    result = {
        "coalition": str(names[coalition_index]),
        "rho": float(screen["observed"][analysis_index, coalition_index]),
        "p_raw": float(screen["p_raw"][analysis_index, coalition_index]),
        "q_bh_within_analysis": float(screen["q_bh_within_analysis"][analysis_index, coalition_index]),
        "p_max_t_120": float(screen["p_max_t_120"][analysis_index, coalition_index]),
        "p_max_t_global600": float(screen["p_max_t_global600"][analysis_index, coalition_index]),
    }
    if (analysis_index, coalition_index) in intervals:
        result["bootstrap_95_ci"] = list(intervals[(analysis_index, coalition_index)])
    return result


def write_source_data(
    names: np.ndarray,
    analyses: Sequence[dict[str, Any]],
    screen: Mapping[str, np.ndarray],
    path: Path = OUTPUT / "emotion_performance_coalition_source_data.tsv",
    global_column_name: str = "p_max_t_global600",
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["analysis", "coalition", "size", "rho", "p_raw", "q_bh_within_analysis", "p_max_t_120", "p_max_t_predefined6", global_column_name])
        for analysis_index, analysis in enumerate(analyses):
            for coalition_index, name in enumerate(names):
                writer.writerow(
                    [
                        analysis["name"],
                        name,
                        str(name).count("+") + 1,
                        f"{screen['observed'][analysis_index, coalition_index]:.12g}",
                        f"{screen['p_raw'][analysis_index, coalition_index]:.12g}",
                        f"{screen['q_bh_within_analysis'][analysis_index, coalition_index]:.12g}",
                        f"{screen['p_max_t_120'][analysis_index, coalition_index]:.12g}",
                        f"{screen['p_max_t_predefined6'][analysis_index, coalition_index]:.12g}" if str(name) in PREDEFINED else "",
                        f"{screen['p_max_t_global600'][analysis_index, coalition_index]:.12g}",
                    ]
                )


def write_report(summary: Mapping[str, Any]) -> None:
    primary = summary["analyses"]["task_face_specific_speed"]
    delta = summary["analyses"]["delta_face_specific_speed"]
    sensitivity = summary["hyperparameter_sensitivity_p5_a10"]["analyses"]
    lines = [
        "# HCP EMOTION performance–coalition synergy screen",
        "",
        "## Analysis contract",
        "",
        "The primary behavioral endpoint is faster face matching after rank-space adjustment for shape-matching speed, age, sex, and recruitment cohort. Because condition EV files are unavailable, coalition synergy is estimated from the complete EMOTION run rather than separate face and shape blocks. The primary brain family contains all 120 fixed Yeo-7 coalitions; EMOTION-minus-REST and three alternative behavioral endpoints are sensitivity analyses.",
        "",
        "## Primary result",
        "",
        f"The largest absolute adjusted association was **{primary['winner']['coalition']}** (rho={primary['winner']['rho']:+.3f}, raw p={primary['winner']['p_raw']:.4f}, 120-combination max-T p={primary['winner']['p_max_t_120']:.4f}, global 600-test max-T p={primary['winner']['p_max_t_global600']:.4f}).",
        "",
        "## EMOTION-minus-REST sensitivity",
        "",
        f"The largest absolute adjusted association was **{delta['winner']['coalition']}** (rho={delta['winner']['rho']:+.3f}, raw p={delta['winner']['p_raw']:.4f}, 120-combination max-T p={delta['winner']['p_max_t_120']:.4f}, global 600-test max-T p={delta['winner']['p_max_t_global600']:.4f}).",
        "",
        "## Hyperparameter sensitivity",
        "",
        f"At (p, alpha)=(5,10), the full-run winner was **{sensitivity['task_face_specific_speed']['winner']['coalition']}** (rho={sensitivity['task_face_specific_speed']['winner']['rho']:+.3f}, 120-combination max-T p={sensitivity['task_face_specific_speed']['winner']['p_max_t_120']:.4f}); the EMOTION-minus-REST winner was **{sensitivity['delta_face_specific_speed']['winner']['coalition']}** (rho={sensitivity['delta_face_specific_speed']['winner']['rho']:+.3f}, max-T p={sensitivity['delta_face_specific_speed']['winner']['p_max_t_120']:.4f}).",
        "",
        "## Prespecified combinations",
        "",
        "| Coalition | rho | raw p | six-combination max-T p | 120-combination max-T p |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in primary["predefined"]:
        lines.append(f"| {row['coalition']} | {row['rho']:+.3f} | {row['p_raw']:.4f} | {row['p_max_t_predefined6']:.4f} | {row['p_max_t_120']:.4f} |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Selection and effect estimation use the same 57 subjects. Only permutation max-T results support family-wise inference; raw p-values are exploratory. The supplementary cohort was selected for behavioral diversity, so age, sex, and cohort are adjusted and permutations are restricted within the original and supplementary cohorts. Family identifiers, head-motion summaries, and face/shape block EVs are unavailable, and the Schaefer atlas excludes the amygdala.",
            "",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    subjects = load_subjects()
    combos = coalitions()
    names = coalition_names(combos)
    predefined_indices = np.asarray([names.tolist().index(name) for name in PREDEFINED], dtype=int)
    if args.smoke:
        smoke_test(subjects, combos)
        return
    status("start", 0, 1, "Loading or recomputing fixed coalition synergy")
    synergy, heldout, explained, nonnegativity = compute_matrices(subjects, combos, args.recompute)
    behavior = load_behavior(subjects)
    analyses = prepare_analyses(synergy, behavior)
    screen = screen_analyses(analyses, behavior["cohort"].astype(int), predefined_indices, args.permutations, SEED)

    status("sensitivity_recomputation", 0, 114, "Computing p=5, alpha=10 sensitivity matrices")
    sensitivity_synergy, sensitivity_heldout, sensitivity_explained, sensitivity_nonnegativity = compute_matrices(
        subjects,
        combos,
        args.recompute,
        order=5,
        alpha=10.0,
        cache_path=SENSITIVITY_CACHE,
        partial_path=SENSITIVITY_PARTIAL,
    )
    sensitivity_analyses = prepare_analyses(sensitivity_synergy, behavior)[:2]
    sensitivity_screen = screen_analyses(
        sensitivity_analyses,
        behavior["cohort"].astype(int),
        predefined_indices,
        args.permutations,
        SEED + 50,
    )

    selected: set[tuple[int, int]] = set()
    for analysis_index in range(len(analyses)):
        selected.add((analysis_index, int(np.argmax(np.abs(screen["observed"][analysis_index])))))
    for coalition_index in predefined_indices:
        selected.add((0, int(coalition_index)))
    for coalition_index in np.argsort(-np.abs(screen["observed"][0]))[:10]:
        selected.add((0, int(coalition_index)))
    intervals: dict[tuple[int, int], tuple[float, float]] = {}
    for analysis_index in range(len(analyses)):
        local_candidates = sorted(candidate for local_analysis, candidate in selected if local_analysis == analysis_index)
        if not local_candidates:
            continue
        local = bootstrap_interval(
            analyses[analysis_index]["matrix"],
            analyses[analysis_index]["endpoint"],
            analyses[analysis_index]["design"],
            behavior["cohort"].astype(int),
            local_candidates,
            args.bootstraps,
            SEED + 100 + analysis_index,
        )
        intervals.update({(analysis_index, candidate): bounds for candidate, bounds in local.items()})

    summary_analyses: dict[str, Any] = {}
    for analysis_index, analysis in enumerate(analyses):
        winner = int(np.argmax(np.abs(screen["observed"][analysis_index])))
        predefined_rows = []
        for candidate in predefined_indices:
            row = serializable_candidate(analysis_index, int(candidate), names, screen, intervals)
            row["p_max_t_predefined6"] = float(screen["p_max_t_predefined6"][analysis_index, candidate])
            predefined_rows.append(row)
        summary_analyses[analysis["name"]] = {
            "label": analysis["label"],
            "winner": serializable_candidate(analysis_index, winner, names, screen, intervals),
            "predefined": predefined_rows,
            "corrected_120_count": int(np.sum(screen["p_max_t_120"][analysis_index] < 0.05)),
            "corrected_global600_count": int(np.sum(screen["p_max_t_global600"][analysis_index] < 0.05)),
            "winner_cohort_rho": cohort_correlations(
                analysis, winner, behavior["cohort"].astype(int)
            ),
        }

    primary_winner = int(np.argmax(np.abs(screen["observed"][0])))
    sensitivity_summary: dict[str, Any] = {}
    for analysis_index, analysis in enumerate(sensitivity_analyses):
        winner = int(np.argmax(np.abs(sensitivity_screen["observed"][analysis_index])))
        primary_config_winner = int(np.argmax(np.abs(screen["observed"][analysis_index])))
        signed_profile_rank_correlation = float(
            np.corrcoef(
                rankdata(screen["observed"][analysis_index]),
                rankdata(sensitivity_screen["observed"][analysis_index]),
            )[0, 1]
        )
        sensitivity_summary[analysis["name"]] = {
            "winner": {
                "coalition": str(names[winner]),
                "rho": float(sensitivity_screen["observed"][analysis_index, winner]),
                "p_raw": float(sensitivity_screen["p_raw"][analysis_index, winner]),
                "p_max_t_120": float(sensitivity_screen["p_max_t_120"][analysis_index, winner]),
            },
            "primary_config_winner_under_sensitivity": {
                "coalition": str(names[primary_config_winner]),
                "rho": float(sensitivity_screen["observed"][analysis_index, primary_config_winner]),
                "p_raw": float(sensitivity_screen["p_raw"][analysis_index, primary_config_winner]),
                "p_max_t_120": float(sensitivity_screen["p_max_t_120"][analysis_index, primary_config_winner]),
            },
            "signed_association_profile_spearman": signed_profile_rank_correlation,
            "winner_cohort_rho": cohort_correlations(
                analysis, winner, behavior["cohort"].astype(int)
            ),
        }
    summary = {
        "exploratory": True,
        "n_subjects": 57,
        "cohorts": {"original": 29, "supplementary": 28},
        "configuration": {
            "parcellation": "Schaefer-1000 / Yeo-7 cortex",
            "representation": "network PC1; task PCA fitted to taskRetained-taskRegressed and projected onto taskRetained",
            "history_order": ORDER,
            "ridge_alpha": ALPHA,
            "estimator": "affine Gaussian TM fixed-coalition Syn",
            "syn_nonnegative_tolerance_bits": SYN_TOLERANCE_BITS,
        },
        "behavior": behavior_summary(behavior),
        "inference": {
            "permutations": args.permutations,
            "bootstraps": args.bootstraps,
            "scheme": "Freedman-Lane residual permutation within original/supplementary cohort",
            "covariates": ["age", "sex", "cohort", "shape speed for primary endpoint"],
            "families": ["six prespecified coalitions", "120 coalitions per analysis", "600 tests globally"],
        },
        "nonnegativity_audit": nonnegativity,
        "diagnostics": {
            "emotion_heldout_skill_ratio_mean": float(heldout[0].mean()),
            "rest_heldout_skill_ratio_mean": float(heldout[1].mean()),
            "emotion_mean_pc1_explained": float(explained[0].mean()),
            "rest_mean_pc1_explained": float(explained[1].mean()),
            "sensitivity_emotion_heldout_skill_ratio_mean": float(sensitivity_heldout[0].mean()),
            "sensitivity_rest_heldout_skill_ratio_mean": float(sensitivity_heldout[1].mean()),
            "sensitivity_emotion_mean_pc1_explained": float(sensitivity_explained[0].mean()),
            "sensitivity_rest_mean_pc1_explained": float(sensitivity_explained[1].mean()),
        },
        "analyses": summary_analyses,
        "hyperparameter_sensitivity_p5_a10": {
            "configuration": {"history_order": 5, "ridge_alpha": 10.0},
            "nonnegativity_audit": sensitivity_nonnegativity,
            "analyses": sensitivity_summary,
        },
        "limitations": [
            "Face/shape condition EV files are unavailable, so Syn is estimated from the complete EMOTION run.",
            "Family identifiers and complete task-motion summaries are unavailable.",
            "The Schaefer cortical atlas excludes the amygdala and other subcortical regions.",
            "Selection and effect estimation use the same 57-subject cohort.",
        ],
    }
    atomic_json(OUTPUT / "summary.json", summary)
    atomic_json(
        OUTPUT / "experiment_contract.json",
        {
            "scientific_question": "Which fixed Yeo-7 coalition is most strongly associated with EMOTION face-matching performance when only coalition membership changes?",
            "pairing_unit": "subject",
            "primary_behavior": "negative log face median RT, rank-adjusted for shape median RT",
            "primary_brain_metric": "full-run EMOTION fixed-coalition Syn",
            "sensitivity_brain_metric": "EMOTION-minus-REST fixed-coalition Syn",
            "frozen_variables": summary["configuration"],
            "statistics": summary["inference"],
            "figure_contract": {
                "core_conclusion": "Quantify whether any fixed cortical network coalition tracks face-specific EMOTION performance after cohort-aware multiplicity correction.",
                "archetype": "quantitative grid",
                "backend": "Python/matplotlib",
                "final_size": "double-column, 183 mm wide",
                "panels": {
                    "a": "face versus shape reaction-time support",
                    "b": "top primary correlations with stratified bootstrap intervals",
                    "c": "full-run versus EMOTION-minus-REST sensitivity",
                    "d": "primary winning coalition residual association",
                },
                "exports": ["PNG", "SVG", "PDF"],
                "reviewer_risks": summary["limitations"],
            },
        },
    )
    write_source_data(names, analyses, screen)
    write_source_data(
        names,
        sensitivity_analyses,
        sensitivity_screen,
        OUTPUT / "emotion_performance_coalition_sensitivity_p5_a10_source_data.tsv",
        "p_max_t_global240",
    )
    plot_intervals = {candidate: intervals[(0, candidate)] for candidate in np.argsort(-np.abs(screen["observed"][0]))[:10]}
    make_figure(subjects, names, analyses, screen, behavior, plot_intervals, primary_winner)
    write_report(summary)
    status("complete", 1, 1, "Analysis, figure, source data, and report complete", state="complete")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--bootstraps", type=int, default=20_000)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
