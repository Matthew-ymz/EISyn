#!/usr/bin/env python3
"""Post-hoc MOTOR test of mean Syn across DMN-containing Yeo7 coalitions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.screen_hcp_motor_composite_scores_57 import (  # noqa: E402
    BEHAVIOR,
    NETWORK_ORDER,
    age_midpoint,
    load_scores,
)


INPUT = ROOT / "results/hcp_motor_composite_scores_57/motor_coalition_synergy_57.npz"
OUTPUT = ROOT / "results/hcp_motor_dmn_coalition_composite_57"
PERMUTATIONS = 100_000
BOOTSTRAPS = 20_000
SEED = 20260804
DMN = "Default"
SYN_TOLERANCE_BITS = 1.0e-9
SHORT = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal attention",
    "SalVentAttn": "Salience/ventral attention",
    "Limbic": "Limbic",
    "Cont": "Control",
    "Default": "Default mode",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 6.8,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array - design @ np.linalg.lstsq(design, array, rcond=None)[0]


def unit_vector(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-12:
        raise ValueError("Cannot normalize a constant residualized variable.")
    return array / norm


def design_matrix(age: np.ndarray, sex: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(age)), rankdata(age), sex])


def partial_spearman(
    brain: np.ndarray, behavior: np.ndarray, age: np.ndarray, sex: np.ndarray
) -> float:
    design = design_matrix(age, sex)
    brain_residual = residualize(rankdata(brain), design)
    behavior_residual = residualize(rankdata(behavior), design)
    return float(unit_vector(brain_residual) @ unit_vector(behavior_residual))


def bh_adjust(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array)
    ranked = array[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(array) / np.arange(1, len(array) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def bootstrap_interval(
    brain: np.ndarray,
    behavior: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    repeats: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    estimates = np.full(repeats, np.nan)
    n = len(brain)
    for index in range(repeats):
        sample = rng.integers(0, n, size=n)
        estimates[index] = partial_spearman(
            brain[sample], behavior[sample], age[sample], sex[sample]
        )
    return np.nanquantile(estimates, [0.025, 0.5, 0.975]).tolist()


def leave_one_out(
    brain: np.ndarray, behavior: np.ndarray, age: np.ndarray, sex: np.ndarray
) -> dict[str, float]:
    estimates = []
    for removed in range(len(brain)):
        keep = np.arange(len(brain)) != removed
        estimates.append(
            partial_spearman(brain[keep], behavior[keep], age[keep], sex[keep])
        )
    array = np.asarray(estimates)
    return {
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
        "same_negative_direction_fraction": float(np.mean(array < 0)),
    }


def load_inputs() -> dict[str, Any]:
    with np.load(INPUT, allow_pickle=False) as archive:
        subjects = archive["subjects"].astype(str)
        names = archive["coalitions"].astype(str)
        sizes = archive["coalition_sizes"].astype(int)
        synergy = archive["synergy_bits"].astype(float)
    if subjects.shape != (57,) or synergy.shape != (57, 120):
        raise ValueError("Expected the frozen 57-subject by 120-coalition MOTOR matrix.")
    violations = synergy < -SYN_TOLERANCE_BITS
    if np.any(violations):
        raise ValueError(
            "PEID Syn nonnegativity violation: "
            f"minimum={synergy.min():.12g} bits, threshold={-SYN_TOLERANCE_BITS:.1e}, "
            f"count={int(violations.sum())}"
        )
    with BEHAVIOR.open(newline="", encoding="utf-8-sig") as handle:
        table = {str(row["Subject"]): row for row in csv.DictReader(handle)}
    age = np.asarray(
        [age_midpoint(table[subject.removeprefix("sub-")]["Age"]) for subject in subjects]
    )
    sex = np.asarray(
        [table[subject.removeprefix("sub-")]["Gender"] == "M" for subject in subjects],
        dtype=float,
    )
    behavior = load_scores(subjects)["composite"]
    return {
        "subjects": subjects,
        "names": names,
        "sizes": sizes,
        "synergy": synergy,
        "age": age,
        "sex": sex,
        "behavior": behavior,
        "nonnegativity": {
            "tolerance_bits": SYN_TOLERANCE_BITS,
            "checked_count": int(synergy.size),
            "minimum_bits": float(synergy.min()),
            "numerical_zero_count": int(
                np.sum((synergy < 0) & (synergy >= -SYN_TOLERANCE_BITS))
            ),
            "significant_violation_count": int(violations.sum()),
        },
    }


def build_composites(
    names: np.ndarray, sizes: np.ndarray, synergy: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    members = [set(name.split("+")) for name in names]
    anchor_raw: dict[str, np.ndarray] = {}
    anchor_balanced: dict[str, np.ndarray] = {}
    for network in NETWORK_ORDER:
        contains = np.asarray([network in coalition for coalition in members])
        anchor_raw[network] = synergy[:, contains].mean(axis=1)
        size_means = [
            synergy[:, contains & (sizes == size)].mean(axis=1) for size in range(2, 8)
        ]
        anchor_balanced[network] = np.column_stack(size_means).mean(axis=1)

    contains_dmn = np.asarray([DMN in coalition for coalition in members])
    dmn_balanced_2_6 = np.column_stack(
        [
            synergy[:, contains_dmn & (sizes == size)].mean(axis=1)
            for size in range(2, 7)
        ]
    ).mean(axis=1)
    no_dmn_balanced_2_6 = np.column_stack(
        [
            synergy[:, (~contains_dmn) & (sizes == size)].mean(axis=1)
            for size in range(2, 7)
        ]
    ).mean(axis=1)

    index = {name: position for position, name in enumerate(names)}
    increments = []
    pairs = []
    for name, position in index.items():
        coalition = name.split("+")
        if DMN not in coalition and 2 <= len(coalition) <= 6:
            with_dmn = name + "+Default"
            increments.append(synergy[:, index[with_dmn]] - synergy[:, position])
            pairs.append((name, with_dmn))
    paired_increment = np.column_stack(increments).mean(axis=1)

    controls = {
        "dmn_raw_mean": anchor_raw[DMN],
        "dmn_size_balanced_mean": anchor_balanced[DMN],
        "dmn_size_balanced_2_6": dmn_balanced_2_6,
        "no_dmn_raw_mean": synergy[:, ~contains_dmn].mean(axis=1),
        "no_dmn_size_balanced_2_6": no_dmn_balanced_2_6,
        "dmn_paired_increment_mean": paired_increment,
    }
    metadata = {
        "dmn_coalition_count": int(contains_dmn.sum()),
        "no_dmn_coalition_count": int((~contains_dmn).sum()),
        "paired_increment_count": len(pairs),
        "paired_increment_definition": "mean over Syn(S union Default) - Syn(S) for all 57 non-DMN coalitions S of size 2-6",
        "paired_increment_minimum_bits": float(np.column_stack(increments).min()),
        "paired_increment_negative_count": int(np.sum(np.column_stack(increments) < 0)),
    }
    return anchor_raw, controls, metadata


def permutation_statistics(
    composites: Mapping[str, np.ndarray],
    behavior: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    permutations: int,
    max_t_keys: tuple[str, ...] = (),
) -> dict[str, dict[str, float]]:
    keys = tuple(composites)
    design = design_matrix(age, sex)
    brain_units = np.column_stack(
        [unit_vector(residualize(rankdata(composites[key]), design)) for key in keys]
    )
    behavior_rank = rankdata(behavior)
    fitted = design @ np.linalg.lstsq(design, behavior_rank, rcond=None)[0]
    behavior_residual = behavior_rank - fitted
    observed = unit_vector(behavior_residual) @ brain_units
    two_sided_counts = np.zeros(len(keys), dtype=np.int64)
    negative_counts = np.zeros(len(keys), dtype=np.int64)
    max_counts = np.zeros(len(max_t_keys), dtype=np.int64)
    max_indices = np.asarray([keys.index(key) for key in max_t_keys], dtype=int)
    rng = np.random.default_rng(SEED)
    chunk = 1000
    for start in range(0, permutations, chunk):
        size = min(chunk, permutations - start)
        indices = np.argsort(rng.random((size, len(behavior))), axis=1)
        pseudo = fitted[None, :] + behavior_residual[indices]
        coefficients = np.linalg.lstsq(design, pseudo.T, rcond=None)[0]
        residuals = pseudo - (design @ coefficients).T
        residuals /= np.linalg.norm(residuals, axis=1, keepdims=True)
        null = residuals @ brain_units
        two_sided_counts += np.sum(np.abs(null) >= np.abs(observed)[None, :], axis=0)
        negative_counts += np.sum(null <= observed[None, :], axis=0)
        if len(max_t_keys):
            maxima = np.max(np.abs(null[:, max_indices]), axis=1)
            max_counts += np.sum(
                maxima[:, None] >= np.abs(observed[max_indices])[None, :], axis=0
            )
    denominator = permutations + 1.0
    results = {}
    for position, key in enumerate(keys):
        results[key] = {
            "rho": float(observed[position]),
            "p_two_sided": float((two_sided_counts[position] + 1.0) / denominator),
            "p_negative_one_sided": float((negative_counts[position] + 1.0) / denominator),
        }
    if len(max_t_keys):
        raw_p = np.asarray([results[key]["p_two_sided"] for key in max_t_keys])
        adjusted = bh_adjust(raw_p)
        for position, key in enumerate(max_t_keys):
            results[key]["q_bh_across_7_anchors"] = float(adjusted[position])
            results[key]["p_max_t_across_7_anchors"] = float(
                (max_counts[position] + 1.0) / denominator
            )
    return results


def add_intervals(
    results: dict[str, dict[str, Any]],
    composites: Mapping[str, np.ndarray],
    behavior: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    bootstraps: int,
    offset: int,
) -> None:
    for position, key in enumerate(composites):
        interval = bootstrap_interval(
            composites[key], behavior, age, sex, bootstraps, SEED + offset + position
        )
        results[key]["bootstrap_95_ci"] = [float(interval[0]), float(interval[2])]
        results[key]["bootstrap_median"] = float(interval[1])


def plot_results(
    anchor_results: Mapping[str, Mapping[str, Any]],
    control_results: Mapping[str, Mapping[str, Any]],
    anchor_raw: Mapping[str, np.ndarray],
    controls: Mapping[str, np.ndarray],
    behavior: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
) -> None:
    configure_style()
    figure = plt.figure(figsize=(7.2, 3.45), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.12, 1.18])
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]

    design = design_matrix(age, sex)
    x = residualize(rankdata(behavior), design)
    y = residualize(rankdata(controls["dmn_raw_mean"]), design)
    axes[0].scatter(
        x,
        y,
        s=18,
        color="#C8764F",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.35,
    )
    order = np.argsort(x)
    coefficient = np.polyfit(x, y, 1)
    axes[0].plot(x[order], np.polyval(coefficient, x[order]), color="#9F4F32", linewidth=1.0)
    dmn = control_results["dmn_raw_mean"]
    axes[0].text(
        0.04,
        0.96,
        rf"$\rho$={dmn['rho']:+.3f}" + "\n" + rf"$p_{{neg}}$={dmn['p_negative_one_sided']:.3f}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
    )
    axes[0].set(
        xlabel="Broad motor score\n(age/sex-adjusted rank residual)",
        ylabel="Mean DMN-containing Syn\n(age/sex-adjusted rank residual)",
    )
    axes[0].set_title("a  Mean of 63 DMN coalitions", loc="left", fontweight="bold")

    networks = list(NETWORK_ORDER)
    positions = np.arange(len(networks))[::-1]
    axes[1].axvline(0, color="#B9C0C6", linewidth=0.75)
    for index, network in enumerate(networks):
        row = anchor_results[network]
        low, high = row["bootstrap_95_ci"]
        estimate = row["rho"]
        color = "#C8764F" if network == DMN else "#6E8798"
        axes[1].errorbar(
            estimate,
            positions[index],
            xerr=np.asarray([[estimate - low], [high - estimate]]),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=4.0,
            capsize=2.0,
            elinewidth=0.95,
        )
    axes[1].set_yticks(positions, [SHORT[network] for network in networks])
    axes[1].set_xlim(-0.52, 0.42)
    axes[1].set_xlabel(r"Partial Spearman $\rho$ (95% CI)")
    axes[1].set_title("b  Same average for each anchor", loc="left", fontweight="bold")

    control_keys = (
        "dmn_raw_mean",
        "dmn_size_balanced_mean",
        "no_dmn_size_balanced_2_6",
        "dmn_paired_increment_mean",
    )
    control_labels = (
        "DMN: raw mean",
        "DMN: size-balanced",
        "No DMN: size-balanced",
        "Paired DMN addition",
    )
    control_positions = np.arange(len(control_keys))[::-1]
    axes[2].axvline(0, color="#B9C0C6", linewidth=0.75)
    for index, key in enumerate(control_keys):
        row = control_results[key]
        low, high = row["bootstrap_95_ci"]
        estimate = row["rho"]
        color = "#87939C" if key.startswith("no_dmn") else "#C8764F"
        axes[2].errorbar(
            estimate,
            control_positions[index],
            xerr=np.asarray([[estimate - low], [high - estimate]]),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=4.0,
            capsize=2.0,
            elinewidth=0.95,
        )
        axes[2].text(
            0.40,
            control_positions[index],
            f"p={row['p_negative_one_sided']:.3f}",
            ha="right",
            va="center",
            fontsize=5.8,
            color="#56616A",
        )
    axes[2].set_yticks(control_positions, control_labels)
    axes[2].set_xlim(-0.52, 0.42)
    axes[2].set_xlabel(r"Partial Spearman $\rho$ (95% CI)")
    axes[2].set_title("c  Aggregation controls", loc="left", fontweight="bold")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "hcp_motor_dmn_coalition_composite_57"
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_outputs(
    anchor_results: Mapping[str, Mapping[str, Any]],
    control_results: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Any],
    inputs: Mapping[str, Any],
    top_ten: list[dict[str, Any]],
    permutations: int,
    bootstraps: int,
) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dmn_anchor = anchor_results[DMN]
    summary = {
        "experiment": "Post-hoc MOTOR association of mean Syn across DMN-containing coalitions",
        "subjects": 57,
        "selection_status": "post-hoc hypothesis motivated by the observed top MOTOR coalitions",
        "behavior": "broad motor score: mean z score of age-adjusted endurance, dexterity, and grip strength",
        "covariates": ["age rank", "sex"],
        "permutations": permutations,
        "bootstraps": bootstraps,
        "primary_dmn_result": dmn_anchor,
        "anchor_results": anchor_results,
        "aggregation_controls": control_results,
        "coalition_metadata": metadata,
        "top_ten_original_coalitions": top_ten,
        "top_ten_containing_dmn_count": int(sum(row["contains_dmn"] for row in top_ten)),
        "nonnegativity_audit": inputs["nonnegativity"],
        "interpretation": "The DMN-containing mean is negative but not significant. Similar negative anchor averages occur for all seven networks, so the pattern is not DMN-specific.",
    }
    contract = {
        "scientific_question": "What changes when only the anchor network used to select coalition Syn values changes?",
        "treatment_factor": "anchor network included in a coalition",
        "levels": list(NETWORK_ORDER),
        "paired_unit": "same 57 subjects and same 120 MOTOR coalition Syn estimates",
        "primary_metric": "partial Spearman correlation between the subject-wise mean of 63 anchor-containing coalition Syn values and broad motor score",
        "secondary_diagnostics": [
            "equal weighting across coalition sizes 2-7",
            "size-matched no-DMN average over sizes 2-6",
            "mean paired increment Syn(S union Default)-Syn(S) over 57 base coalitions",
        ],
        "statistics": "100,000 pooled Freedman-Lane permutations; one-sided negative p for the post-hoc directional question; BH and max-T across seven anchor networks; 20,000 subject bootstraps",
        "controlled_variables": [
            "subjects",
            "MOTOR state",
            "behavior endpoint",
            "age and sex adjustment",
            "Schaefer-1000/Yeo7 representation",
            "order-3 Ridge alpha=1 affine Gaussian TM estimator",
            "coalition cache",
            "permutation indices",
        ],
        "figure_contract": {
            "core_conclusion": "Test whether the negative MOTOR association survives aggregation across all DMN-containing coalitions and whether it is DMN-specific.",
            "evidence_chain": [
                "DMN composite scatter",
                "seven identically constructed anchor-network composites",
                "coalition-size and paired-addition controls",
            ],
            "archetype": "asymmetric mixed-modality figure",
            "role": "validation and controlled comparison",
            "backend": "Python/matplotlib",
            "exports": ["PNG 600 dpi", "editable SVG", "PDF"],
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "experiment_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dmn = anchor_results[DMN]
    paired = control_results["dmn_paired_increment_mean"]
    no_dmn = control_results["no_dmn_size_balanced_2_6"]
    lines = [
        "# MOTOR：包含默认模式网络的组合平均",
        "",
        "对每名被试先平均全部 63 个包含 Default mode network（DMN）的 MOTOR coalition Syn，再与广义运动指数作控制年龄和性别的偏 Spearman 相关。由于该假设来自已观察到的最高相关组合，方向性检验属于事后分析。",
        "",
        f"DMN 平均量与运动表现为负相关（$\\rho={dmn['rho']:+.3f}$，95% CI [{dmn['bootstrap_95_ci'][0]:+.3f}, {dmn['bootstrap_95_ci'][1]:+.3f}]），但双侧置换 $p={dmn['p_two_sided']:.4f}$，负向单侧 $p={dmn['p_negative_one_sided']:.4f}$，七网络 max-$T$ $p={dmn['p_max_t_across_7_anchors']:.4f}$，没有达到 0.05。",
        "",
        "七个锚定网络用完全相同的方法聚合后均为负相关，DMN 不是效应最强的锚点。这说明原组合榜单中的 DMN 富集不能单独证明 DMN 特异机制。",
        "",
        f"组合大小等权的 no-DMN 对照为 $\\rho={no_dmn['rho']:+.3f}$。对 57 个基础组合逐一计算加入 DMN 后的 Syn 增量，再跨组合平均，得到 $\\rho={paired['rho']:+.3f}$、负向单侧 $p={paired['p_negative_one_sided']:.4f}$；该敏感性结果同样是事后且未对多种聚合定义校正。",
        "",
        "![MOTOR DMN coalition composite](hcp_motor_dmn_coalition_composite_57.png)",
        "",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = load_inputs()
    anchor_raw, controls, metadata = build_composites(
        inputs["names"], inputs["sizes"], inputs["synergy"]
    )
    anchor_results = permutation_statistics(
        anchor_raw,
        inputs["behavior"],
        inputs["age"],
        inputs["sex"],
        args.permutations,
        max_t_keys=tuple(NETWORK_ORDER),
    )
    control_results = permutation_statistics(
        controls,
        inputs["behavior"],
        inputs["age"],
        inputs["sex"],
        args.permutations,
    )
    add_intervals(
        anchor_results,
        anchor_raw,
        inputs["behavior"],
        inputs["age"],
        inputs["sex"],
        args.bootstraps,
        100,
    )
    add_intervals(
        control_results,
        controls,
        inputs["behavior"],
        inputs["age"],
        inputs["sex"],
        args.bootstraps,
        200,
    )
    anchor_results[DMN]["leave_one_out"] = leave_one_out(
        anchor_raw[DMN], inputs["behavior"], inputs["age"], inputs["sex"]
    )
    coalition_results = permutation_statistics(
        {
            str(name): inputs["synergy"][:, index]
            for index, name in enumerate(inputs["names"])
        },
        inputs["behavior"],
        inputs["age"],
        inputs["sex"],
        args.permutations,
    )
    ranking = sorted(
        range(len(inputs["names"])),
        key=lambda index: -abs(coalition_results[str(inputs["names"][index])]["rho"]),
    )[:10]
    top_ten = [
        {
            "coalition": str(inputs["names"][index]),
            "size": int(inputs["sizes"][index]),
            "rho": coalition_results[str(inputs["names"][index])]["rho"],
            "contains_dmn": bool(DMN in str(inputs["names"][index]).split("+")),
        }
        for index in ranking
    ]
    plot_results(
        anchor_results,
        control_results,
        anchor_raw,
        controls,
        inputs["behavior"],
        inputs["age"],
        inputs["sex"],
    )
    write_outputs(
        anchor_results,
        control_results,
        metadata,
        inputs,
        top_ten,
        args.permutations,
        args.bootstraps,
    )
    print(
        json.dumps(
            {"anchors": anchor_results, "controls": control_results},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
