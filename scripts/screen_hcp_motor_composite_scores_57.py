#!/usr/bin/env python3
"""Screen MOTOR Yeo-7 coalition Syn against a broad motor-capacity score."""

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
from scipy.stats import rankdata, spearmanr


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
SUBJECT_SOURCE = ROOT / "results/hcp_schaefer1000_task_evoked_xi_57/full/k1_p3_a1/arrays.npz"
OUTPUT = ROOT / "results/hcp_motor_composite_scores_57"
CACHE = OUTPUT / "motor_coalition_synergy_57.npz"
PARTIAL_CACHE = OUTPUT / "motor_coalition_synergy_57.partial.npz"

NETWORK_ORDER = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
SHORT = {
    "Vis": "Vis",
    "SomMot": "Som",
    "DorsAttn": "DAN",
    "SalVentAttn": "SVAN",
    "Limbic": "Lim",
    "Cont": "Cont",
    "Default": "DMN",
}
COMPONENT_FIELDS = {
    "Endurance": "Endurance_AgeAdj",
    "Dexterity": "Dexterity_AgeAdj",
    "Strength": "Strength_AgeAdj",
}
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
    return tuple(
        combination
        for size in range(2, 8)
        for combination in itertools.combinations(NETWORK_ORDER, size)
    )


def coalition_names(values: Sequence[Sequence[str]]) -> np.ndarray:
    return np.asarray(["+".join(value) for value in values])


def compact_name(value: str) -> str:
    return "+".join(SHORT[item] for item in value.split("+"))


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


def compute_matrix(
    subjects: np.ndarray,
    combinations: Sequence[tuple[str, ...]],
    recompute: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = coalition_names(combinations)
    if CACHE.is_file() and not recompute:
        with np.load(CACHE, allow_pickle=False) as archive:
            if (
                np.array_equal(archive["subjects"].astype(str), subjects)
                and np.array_equal(archive["coalitions"].astype(str), names)
            ):
                matrix = archive["synergy_bits"].astype(float)
                if matrix.shape == (57, 120) and np.isfinite(matrix).all():
                    return (
                        matrix,
                        archive["heldout_skill_ratio"].astype(float),
                        archive["mean_pc1_explained"].astype(float),
                    )

    matrix = np.full((57, 120), np.nan)
    heldout = np.full(57, np.nan)
    explained = np.full(57, np.nan)
    if PARTIAL_CACHE.is_file() and not recompute:
        with np.load(PARTIAL_CACHE, allow_pickle=False) as archive:
            if (
                np.array_equal(archive["subjects"].astype(str), subjects)
                and np.array_equal(archive["coalitions"].astype(str), names)
            ):
                matrix = archive["synergy_bits"].astype(float)
                heldout = archive["heldout_skill_ratio"].astype(float)
                explained = archive["mean_pc1_explained"].astype(float)

    groups = load_yeo7_groups(LABELS, expected_parcels=1000)
    indices = network_module_indices(NETWORK_ORDER, n_components=1, order=ORDER)
    for index in np.flatnonzero(~np.isfinite(matrix).all(axis=1)):
        subject = str(subjects[index])
        projections, variance, development_end = prepare_projection(
            TASK_ROOT / subject / "MOTOR_LR.mat",
            groups,
            state="MOTOR",
            max_components=1,
            task_retained_key="Schaefer1000_taskRetained",
            task_regressed_key="Schaefer1000_taskRegressed",
            expected_parcels=1000,
        )
        fitted = fit_delta_history_phi(
            projections[1], alpha=ALPHA, order=ORDER, development_end=development_end
        )
        table = module_ei_table(
            fitted["transition"], fitted["noise_covariance"], indices, ridge=1.0e-6
        )
        singleton = {name: float(table[(name,)]) for name in NETWORK_ORDER}
        matrix[index] = [
            subset_phi_raw(combination, table, singleton)
            for combination in combinations
        ]
        heldout[index] = float(fitted["heldout"]["skill_ratio"])
        explained[index] = float(np.mean(list(variance[1].values())))
        atomic_npz(
            PARTIAL_CACHE,
            subjects=subjects,
            coalitions=names,
            synergy_bits=matrix,
            heldout_skill_ratio=heldout,
            mean_pc1_explained=explained,
        )
        print(f"[{index + 1:02d}/57] {subject}", flush=True)

    violation = matrix < -SYN_TOLERANCE_BITS
    if np.any(violation):
        raise ValueError(
            "PEID Syn nonnegativity violation: "
            f"minimum={matrix.min():.12g} bits, threshold={-SYN_TOLERANCE_BITS:.1e}, "
            f"count={int(violation.sum())}"
        )
    atomic_npz(
        CACHE,
        subjects=subjects,
        coalitions=names,
        coalition_sizes=np.asarray([len(value) for value in combinations]),
        synergy_bits=matrix,
        heldout_skill_ratio=heldout,
        mean_pc1_explained=explained,
        syn_tolerance_bits=np.asarray(SYN_TOLERANCE_BITS),
    )
    if PARTIAL_CACHE.exists():
        PARTIAL_CACHE.unlink()
    return matrix, heldout, explained


def load_scores(subjects: np.ndarray) -> dict[str, np.ndarray]:
    with BEHAVIOR.open(newline="", encoding="utf-8-sig") as handle:
        table = {str(row["Subject"]): row for row in csv.DictReader(handle)}
    keys = [str(value).removeprefix("sub-") for value in subjects]
    rows = [table[key] for key in keys]
    raw = np.column_stack(
        [
            np.asarray([float(row[field]) for row in rows], dtype=float)
            for field in COMPONENT_FIELDS.values()
        ]
    )
    if not np.isfinite(raw).all():
        raise ValueError("The frozen 57-subject sample has missing motor component scores.")
    means = raw.mean(axis=0)
    standard_deviations = raw.std(axis=0, ddof=1)
    standardized = (raw - means) / standard_deviations
    composite = standardized.mean(axis=1)
    return {
        "components_raw": raw,
        "components_z": standardized,
        "component_means": means,
        "component_sds": standard_deviations,
        "composite": composite,
        "age": np.asarray([age_midpoint(row["Age"]) for row in rows]),
        "sex": np.asarray([row["Gender"] == "M" for row in rows], dtype=float),
        "cohort": np.r_[np.zeros(29, dtype=int), np.ones(28, dtype=int)],
    }


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def unit_columns(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    array = array - array.mean(axis=0, keepdims=True)
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


def base_design(scores: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(scores["cohort"])),
            scores["age"],
            scores["sex"],
            scores["cohort"],
        ]
    )


def adjusted_rho(
    x: np.ndarray, y: np.ndarray, scores: Mapping[str, np.ndarray]
) -> float:
    design = base_design(scores)
    x_residual = residualize(rankdata(x), design)
    y_residual = residualize(rankdata(y), design)
    denominator = np.linalg.norm(x_residual) * np.linalg.norm(y_residual)
    if denominator <= 1.0e-12:
        return float("nan")
    return float(x_residual @ y_residual / denominator)


def screen(
    matrix: np.ndarray,
    scores: Mapping[str, np.ndarray],
    permutations: int,
    seed: int,
) -> dict[str, np.ndarray]:
    cohort = scores["cohort"].astype(int)
    design = base_design(scores)
    brain = unit_columns(residualize(rankdata(matrix, axis=0), design))
    endpoint = rankdata(scores["composite"])
    fitted = design @ np.linalg.lstsq(design, endpoint, rcond=None)[0]
    endpoint_residual = residualize(endpoint, design)
    endpoint_unit = unit_columns(endpoint_residual)[:, 0]
    observed = brain.T @ endpoint_unit
    point_counts = np.zeros(120, dtype=np.int64)
    family_counts = np.zeros(120, dtype=np.int64)
    groups = [np.flatnonzero(cohort == value) for value in np.unique(cohort)]
    rng = np.random.default_rng(seed)
    chunk = 1000
    for start in range(0, permutations, chunk):
        size = min(chunk, permutations - start)
        permutation_indices = np.tile(np.arange(len(cohort)), (size, 1))
        for group in groups:
            order = np.argsort(rng.random((size, len(group))), axis=1)
            permutation_indices[:, group] = group[order]
        pseudo = fitted[None, :] + endpoint_residual[permutation_indices]
        coefficients = np.linalg.lstsq(design, pseudo.T, rcond=None)[0]
        residual = pseudo - (design @ coefficients).T
        residual /= np.linalg.norm(residual, axis=1, keepdims=True)
        null = residual @ brain
        absolute = np.abs(null)
        point_counts += np.sum(absolute >= np.abs(observed)[None, :], axis=0)
        family_maximum = absolute.max(axis=1)
        family_counts += np.sum(
            family_maximum[:, None] >= np.abs(observed)[None, :], axis=0
        )
    denominator = permutations + 1.0
    p_raw = (point_counts + 1.0) / denominator
    return {
        "rho_adjusted": observed,
        "p_raw": p_raw,
        "q_bh_120": bh(p_raw),
        "p_max_t_120": (family_counts + 1.0) / denominator,
    }


def bootstrap_rho(
    brain: np.ndarray,
    endpoint: np.ndarray,
    scores: Mapping[str, np.ndarray],
    repeats: int,
    seed: int,
) -> list[float]:
    cohort = scores["cohort"].astype(int)
    groups = [np.flatnonzero(cohort == value) for value in np.unique(cohort)]
    rng = np.random.default_rng(seed)
    values = np.full(repeats, np.nan)
    for index in range(repeats):
        sample = np.concatenate(
            [rng.choice(group, len(group), replace=True) for group in groups]
        )
        design = np.column_stack(
            [
                np.ones(len(sample)),
                scores["age"][sample],
                scores["sex"][sample],
                scores["cohort"][sample],
            ]
        )
        x_residual = residualize(rankdata(brain[sample]), design)
        y_residual = residualize(rankdata(endpoint[sample]), design)
        denominator = np.linalg.norm(x_residual) * np.linalg.norm(y_residual)
        if denominator > 1.0e-12:
            values[index] = float(x_residual @ y_residual / denominator)
    return np.nanquantile(values, [0.025, 0.5, 0.975]).tolist()


def make_rows(
    names: np.ndarray,
    sizes: np.ndarray,
    matrix: np.ndarray,
    scores: Mapping[str, np.ndarray],
    result: Mapping[str, np.ndarray],
    bootstraps: int,
) -> tuple[list[dict[str, Any]], int, list[int]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        rows.append(
            {
                "index": index,
                "coalition": str(name),
                "short_coalition": compact_name(str(name)),
                "coalition_size": int(sizes[index]),
                "rho_adjusted": float(result["rho_adjusted"][index]),
                "rho_pooled_raw": float(
                    spearmanr(matrix[:, index], scores["composite"]).statistic
                ),
                "rho_original_29": float(
                    spearmanr(matrix[:29, index], scores["composite"][:29]).statistic
                ),
                "rho_supplement_28": float(
                    spearmanr(matrix[29:, index], scores["composite"][29:]).statistic
                ),
                "p_raw": float(result["p_raw"][index]),
                "q_bh_120": float(result["q_bh_120"][index]),
                "p_max_t_120": float(result["p_max_t_120"][index]),
            }
        )
    ranking = np.argsort(-np.abs(result["rho_adjusted"]))
    winner = int(ranking[0])
    top = ranking[:10].astype(int).tolist()
    for rank_index, coalition_index in enumerate(top):
        rows[coalition_index]["stratified_bootstrap_quantiles"] = bootstrap_rho(
            matrix[:, coalition_index],
            scores["composite"],
            scores,
            bootstraps,
            SEED + 100 + rank_index,
        )
    return rows, winner, top


def cronbach_alpha(standardized_components: np.ndarray) -> float:
    items = np.asarray(standardized_components, dtype=float)
    item_variances = items.var(axis=0, ddof=1).sum()
    total_variance = items.sum(axis=1).var(ddof=1)
    count = items.shape[1]
    return float(count / (count - 1) * (1.0 - item_variances / total_variance))


def leave_one_out(
    brain: np.ndarray, endpoint: np.ndarray, scores: Mapping[str, np.ndarray]
) -> dict[str, float]:
    values = []
    for index in range(len(brain)):
        keep = np.arange(len(brain)) != index
        subset_scores = {key: np.asarray(value)[keep] for key, value in scores.items() if key in {"age", "sex", "cohort"}}
        values.append(adjusted_rho(brain[keep], endpoint[keep], subset_scores))
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.nanmin(array)),
        "median": float(np.nanmedian(array)),
        "maximum": float(np.nanmax(array)),
    }


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


def plot(
    names: np.ndarray,
    sizes: np.ndarray,
    matrix: np.ndarray,
    scores: Mapping[str, np.ndarray],
    result: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    winner: int,
    top: Sequence[int],
) -> None:
    configure_style()
    figure = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=[0.95, 1.15])
    axes = [figure.add_subplot(grid[row, column]) for row, column in ((0, 0), (0, 1), (1, 0), (1, 1))]

    labels = [*COMPONENT_FIELDS.keys(), "Composite"]
    behavior = np.column_stack([scores["components_raw"], scores["composite"]])
    correlations = spearmanr(behavior, axis=0).statistic
    image = axes[0].imshow(correlations, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_xticks(np.arange(4), labels, rotation=35, ha="right")
    axes[0].set_yticks(np.arange(4), labels)
    for row in range(4):
        for column in range(4):
            axes[0].text(
                column,
                row,
                f"{correlations[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(correlations[row, column]) > 0.58 else "#30343B",
                fontsize=6.1,
            )
    axes[0].set_title("a  Motor-score coherence", loc="left", fontweight="bold")
    colorbar = figure.colorbar(image, ax=axes[0], shrink=0.70, pad=0.02)
    colorbar.set_label(r"Spearman $\rho$")

    rng = np.random.default_rng(SEED)
    x = sizes + rng.uniform(-0.10, 0.10, len(sizes))
    axes[1].axhline(0, color="#D6DADF", linewidth=0.7)
    axes[1].scatter(
        x,
        result["rho_adjusted"],
        s=20,
        color="#7A8DA6",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.3,
    )
    axes[1].scatter(
        sizes[winner],
        result["rho_adjusted"][winner],
        s=66,
        marker="D",
        facecolor="#D07A55",
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
    )
    axes[1].annotate(
        compact_name(str(names[winner])),
        (sizes[winner], result["rho_adjusted"][winner]),
        xytext=(5, 5),
        textcoords="offset points",
        color="#9A4D32",
        fontsize=6.4,
    )
    axes[1].set(
        xlabel="Coalition size (Yeo7 networks)",
        ylabel=r"Adjusted association with motor score ($\rho$)",
    )
    axes[1].set_xticks(np.arange(2, 8))
    axes[1].set_title("b  All 120 MOTOR coalitions", loc="left", fontweight="bold")

    retained_top = [index for index in top if rows[index]["p_raw"] < 0.05]
    if len(retained_top) != 9:
        raise ValueError(
            "Expected nine pointwise-significant negative associations after "
            f"removing the p=0.0501 result; found {len(retained_top)}."
        )
    ordered = list(reversed(retained_top))
    y = np.arange(len(ordered))
    centers = np.asarray([rows[index]["rho_adjusted"] for index in ordered])
    intervals = np.asarray(
        [rows[index]["stratified_bootstrap_quantiles"] for index in ordered]
    )
    axes[2].axvline(0, color="#D6DADF", linewidth=0.7)
    axes[2].errorbar(
        centers,
        y,
        xerr=np.vstack([centers - intervals[:, 0], intervals[:, 2] - centers]),
        fmt="o",
        markersize=3.8,
        color="#526D82",
        ecolor="#9AA8B2",
        elinewidth=0.8,
        capsize=1.8,
    )
    axes[2].scatter(centers[-1], y[-1], s=34, marker="D", color="#D07A55", zorder=4)
    axes[2].set_xlim(-0.64, 0.52)
    for position, coalition_index in zip(y, ordered, strict=True):
        row = rows[coalition_index]
        label = rf"$\rho$={row['rho_adjusted']:+.3f}; $p$={row['p_raw']:.4f}"
        axes[2].text(
            0.035,
            position,
            label,
            ha="left",
            va="center",
            fontsize=5.7,
            color="#9A4D32",
            fontweight="bold",
        )
    axes[2].set_yticks(y, [rows[index]["short_coalition"] for index in ordered])
    axes[2].set_xlabel(r"Adjusted $\rho$ (stratified bootstrap 95% CI)")
    axes[2].set_title("c  Nine retained negative associations", loc="left", fontweight="bold")

    cohort = scores["cohort"].astype(int)
    colors = np.asarray(["#7A8DA6", "#D07A55"])
    for cohort_value, label in ((0, "Original 29"), (1, "Supplement 28")):
        mask = cohort == cohort_value
        axes[3].scatter(
            scores["composite"][mask],
            matrix[mask, winner],
            s=21,
            color=colors[cohort_value],
            alpha=0.84,
            edgecolor="white",
            linewidth=0.35,
            label=label,
        )
    order = np.argsort(scores["composite"])
    coefficient = np.polyfit(scores["composite"], matrix[:, winner], 1)
    axes[3].plot(
        scores["composite"][order],
        np.polyval(coefficient, scores["composite"][order]),
        color="#465563",
        linestyle="--",
        linewidth=0.9,
    )
    selected = rows[winner]
    axes[3].text(
        1.05,
        0.98,
        rf"adjusted $\rho$={selected['rho_adjusted']:+.3f}"
        + f"\n120-coalition max-T $p$={selected['p_max_t_120']:.4f}\n$n$=57",
        transform=axes[3].transAxes,
        ha="left",
        va="top",
        clip_on=False,
    )
    axes[3].set(
        xlabel="Composite motor-capacity score (z units)",
        ylabel="Coalition Syn (bits)",
    )
    axes[3].set_title(
        f"d  {compact_name(str(names[winner]))}", loc="left", fontweight="bold"
    )
    axes[3].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        figure.savefig(
            OUTPUT / f"motor_composite_coalition_screen_57.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    subjects = load_subjects()
    combinations = coalitions()
    names = coalition_names(combinations)
    sizes = np.asarray([len(value) for value in combinations])
    matrix, heldout, explained = compute_matrix(subjects, combinations, args.recompute)
    scores = load_scores(subjects)
    result = screen(matrix, scores, args.permutations, SEED)
    rows, winner, top = make_rows(
        names, sizes, matrix, scores, result, args.bootstraps
    )
    winner_brain = matrix[:, winner]
    component_correlations = {
        name: adjusted_rho(winner_brain, scores["components_raw"][:, index], scores)
        for index, name in enumerate(COMPONENT_FIELDS)
    }
    score_component_correlations = {
        name: float(spearmanr(scores["composite"], scores["components_raw"][:, index]).statistic)
        for index, name in enumerate(COMPONENT_FIELDS)
    }
    winner_row = rows[winner]
    winner_row["leave_one_out_adjusted_rho"] = leave_one_out(
        winner_brain, scores["composite"], scores
    )
    winner_row["component_adjusted_rho"] = component_correlations

    summary = {
        "experiment": "MOTOR fixed-coalition Syn screen against broad motor capacity",
        "subjects": 57,
        "score_definition": {
            "formula": "mean[z(Endurance_AgeAdj), z(Dexterity_AgeAdj), z(Strength_AgeAdj)]",
            "standardization_reference": "Frozen 57-subject imaging sample; sample SD with ddof=1",
            "direction": "Higher values indicate better broad motor capacity.",
            "missing_values": 0,
            "component_means": {
                name: float(scores["component_means"][index])
                for index, name in enumerate(COMPONENT_FIELDS)
            },
            "component_sds": {
                name: float(scores["component_sds"][index])
                for index, name in enumerate(COMPONENT_FIELDS)
            },
            "composite_minimum": float(scores["composite"].min()),
            "composite_maximum": float(scores["composite"].max()),
            "composite_sd": float(scores["composite"].std(ddof=1)),
            "cronbach_alpha": cronbach_alpha(scores["components_z"]),
            "composite_component_spearman": score_component_correlations,
        },
        "inference": {
            "permutations": args.permutations,
            "bootstraps": args.bootstraps,
            "covariates": ["age", "sex", "original/supplement cohort"],
            "scheme": "Freedman-Lane residual permutation within original/supplement cohort",
            "primary_family": "120 fixed Yeo7 coalitions for one composite motor endpoint",
            "selection_status": "Exploratory: coalition selection and effect estimation use the same 57 subjects.",
        },
        "model": {
            "parcellation": "Schaefer-1000 / Yeo-7 cortex",
            "state": "full MOTOR LR run",
            "representation": "network PC1 fitted to taskRetained-taskRegressed and projected onto taskRetained",
            "history_order": ORDER,
            "ridge_alpha": ALPHA,
            "estimator": "affine Gaussian TM fixed-coalition Syn",
            "mean_heldout_skill_ratio": float(heldout.mean()),
            "models_better_than_persistence": int(np.sum(heldout < 1)),
            "mean_pc1_explained": float(explained.mean()),
            "syn_nonnegative_tolerance_bits": SYN_TOLERANCE_BITS,
            "minimum_synergy_bits": float(matrix.min()),
            "negative_within_tolerance_count": int(
                np.sum((matrix < 0) & (matrix >= -SYN_TOLERANCE_BITS))
            ),
            "significant_negative_count": int(
                np.sum(matrix < -SYN_TOLERANCE_BITS)
            ),
        },
        "winner": winner_row,
        "top_ten": [rows[index] for index in top],
        "significance_counts": {
            "raw_p_below_0_05": int(np.sum(result["p_raw"] < 0.05)),
            "bh_q_below_0_05": int(np.sum(result["q_bh_120"] < 0.05)),
            "max_t_p_below_0_05": int(np.sum(result["p_max_t_120"] < 0.05)),
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT / "all_associations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUTPUT / "experiment_contract.json").write_text(
        json.dumps(
            {
                "scientific_question": "Which fixed Yeo-7 coalition changes most strongly with broad motor capacity when only coalition membership changes?",
                "pairing_unit": "subject",
                "primary_behavior": summary["score_definition"]["formula"],
                "primary_brain_metric": "full-run MOTOR fixed-coalition TM Syn",
                "treatment_factor": "Yeo-7 coalition membership (120 levels)",
                "frozen_variables": summary["model"],
                "statistics": summary["inference"],
                "figure_contract": {
                    "core_conclusion": "Identify the strongest MOTOR coalition association and show whether it survives family-wise correction.",
                    "evidence_chain": ["score coherence", "all-coalition screen", "top-ten uncertainty", "winner association"],
                    "archetype": "quantitative grid",
                    "backend": "Python/matplotlib",
                    "exports": ["PNG 600 dpi", "editable SVG", "PDF"],
                    "review_risks": ["broad phenotype is not in-scanner task accuracy", "post-selection effect inflation", "cortical Yeo7 atlas excludes subcortex and cerebellum"],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot(names, sizes, matrix, scores, result, rows, winner, top)

    lines = [
        "# HCP MOTOR composite-score coalition screen",
        "",
        "![MOTOR screen](motor_composite_coalition_screen_57.png)",
        "",
        "The primary motor score is the equal-weight mean of sample-standardized age-adjusted endurance, dexterity, and grip-strength scores. Higher values indicate better broad motor capacity; this is an out-of-scanner phenotype, not MOTOR-run accuracy.",
        "",
        "| Coalition | Adjusted rho | Original 29 rho | Supplement 28 rho | Raw p | BH q | 120-coalition max-T p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index in top[:10]:
        row = rows[index]
        lines.append(
            f"| {row['short_coalition']} | {row['rho_adjusted']:+.3f} | "
            f"{row['rho_original_29']:+.3f} | {row['rho_supplement_28']:+.3f} | "
            f"{row['p_raw']:.5f} | {row['q_bh_120']:.5f} | {row['p_max_t_120']:.5f} |"
        )
    lines.extend(
        [
            "",
            "Adjusted rho residualizes age, sex, and recruitment cohort in rank space. Permutations preserve the original/supplement cohort blocks. The screen is exploratory because coalition selection and effect estimation use the same 57 subjects.",
            "",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
