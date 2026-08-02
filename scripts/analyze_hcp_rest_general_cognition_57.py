#!/usr/bin/env python3
"""Associate 57-subject REST fixed-coalition synergy with general cognition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
CACHE = (
    ROOT
    / "results/hcp_emotion_performance_coalitions_57"
    / "emotion_rest_coalition_synergy_57.npz"
)
COGNITION = ROOT / "results/hcp_single_group_sem_full_1206/factor_scores_all_subjects.csv"
ORIGINAL = ROOT / "results/hcp_single_group_sem_full_1206/selected_29_sem_results.csv"
BEHAVIOR = ROOT / "data/unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
OUTPUT = ROOT / "results/hcp_rest_general_cognition_57"
SEED = 2026080201
SYN_TOLERANCE_BITS = 1.0e-9
NETWORK_SHORT = {
    "Vis": "Vis",
    "SomMot": "Som",
    "DorsAttn": "DAN",
    "SalVentAttn": "SVAN",
    "Limbic": "Lim",
    "Cont": "Cont",
    "Default": "DMN",
}


def age_midpoint(value: str) -> float:
    if value == "36+":
        return 38.0
    low, high = value.split("-")
    return 0.5 * (float(low) + float(high))


def compact_name(name: str) -> str:
    return "+".join(NETWORK_SHORT.get(part, part) for part in name.split("+"))


def projection_residual(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def unit_columns(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    array = array - array.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(array, axis=0, keepdims=True)
    if np.any(norm <= 1.0e-12):
        raise ValueError("A residualized variable is constant.")
    return array / norm


def adjusted_rank_correlation(x: np.ndarray, y: np.ndarray, design: np.ndarray) -> float:
    x_residual = projection_residual(rankdata(x), design)
    y_residual = projection_residual(rankdata(y), design)
    return float(
        x_residual @ y_residual
        / np.sqrt((x_residual @ x_residual) * (y_residual @ y_residual))
    )


def bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def load_inputs() -> dict[str, Any]:
    with np.load(CACHE, allow_pickle=False) as archive:
        states = archive["states"].astype(str).tolist()
        subjects = archive["subjects"].astype(str)
        names = archive["coalitions"].astype(str)
        sizes = archive["coalition_sizes"].astype(int)
        synergy = archive["synergy_bits"].astype(float)[states.index("REST")]
        cached_tolerance = float(archive["syn_tolerance_bits"])
    if subjects.shape != (57,) or synergy.shape != (57, 120):
        raise ValueError("Expected a frozen 57 x 120 REST coalition matrix.")
    if not np.isfinite(synergy).all():
        raise ValueError("REST coalition matrix contains non-finite values.")
    if not np.isclose(cached_tolerance, SYN_TOLERANCE_BITS):
        raise ValueError(f"Unexpected Syn tolerance: {cached_tolerance:g} bits.")
    violation = synergy < -SYN_TOLERANCE_BITS
    if np.any(violation):
        raise ValueError(
            "PEID Syn nonnegativity violation: "
            f"min={synergy.min():.12g}, threshold={-SYN_TOLERANCE_BITS:.12g}, "
            f"count={int(violation.sum())}"
        )

    cognition = pd.read_csv(COGNITION, dtype={"Subject": str}).set_index("Subject")
    original = set(
        pd.read_csv(ORIGINAL, dtype={"Subject": str})["Subject"].astype(str).tolist()
    )
    behavior = pd.read_csv(BEHAVIOR, dtype={"Subject": str}).set_index("Subject")
    ids = np.asarray([subject.removeprefix("sub-") for subject in subjects])
    score = cognition.loc[ids, "g_score"].to_numpy(dtype=float)
    age = np.asarray([age_midpoint(value) for value in behavior.loc[ids, "Age"]])
    sex = (behavior.loc[ids, "Gender"].to_numpy(dtype=str) == "M").astype(float)
    cohort = np.asarray([subject in original for subject in ids], dtype=int)
    # Use 0=original 29 and 1=supplementary 28 in all outputs and plots.
    cohort = 1 - cohort
    if int(np.sum(cohort == 0)) != 29 or int(np.sum(cohort == 1)) != 28:
        raise ValueError("Expected original/supplementary cohort sizes of 29/28.")
    if not all(np.isfinite(values).all() for values in (score, age, sex, cohort)):
        raise ValueError("Cognition or covariate data contain non-finite values.")
    return {
        "subjects": subjects,
        "ids": ids,
        "names": names,
        "sizes": sizes,
        "synergy": synergy,
        "score": score,
        "age": age,
        "sex": sex,
        "cohort": cohort,
    }


def permutation_screen(
    synergy: np.ndarray,
    score: np.ndarray,
    design: np.ndarray,
    cohort: np.ndarray,
    permutations: int,
) -> dict[str, np.ndarray]:
    brain_rank = rankdata(synergy, axis=0, method="average")
    brain_unit = unit_columns(projection_residual(brain_rank, design))
    score_rank = rankdata(score, method="average")
    fitted = design @ np.linalg.lstsq(design, score_rank, rcond=None)[0]
    score_residual = projection_residual(score_rank, design)
    score_unit = unit_columns(score_residual).ravel()
    residual_maker = np.eye(len(score)) - design @ np.linalg.pinv(design)
    observed = brain_unit.T @ score_unit

    groups = [np.flatnonzero(cohort == value) for value in np.unique(cohort)]
    rng = np.random.default_rng(SEED)
    point_count = np.zeros(synergy.shape[1], dtype=np.int64)
    max_count = np.zeros_like(point_count)
    for start in range(0, permutations, 500):
        count = min(500, permutations - start)
        indices = np.tile(np.arange(len(score)), (count, 1))
        for group in groups:
            order = np.argsort(rng.random((count, len(group))), axis=1)
            indices[:, group] = group[order]
        pseudo = fitted[None, :] + score_residual[indices]
        permuted_residual = pseudo @ residual_maker.T
        permuted_residual /= np.linalg.norm(permuted_residual, axis=1, keepdims=True)
        absolute = np.abs(permuted_residual @ brain_unit)
        point_count += np.sum(absolute >= np.abs(observed)[None, :] - 1.0e-15, axis=0)
        maximum = absolute.max(axis=1)
        max_count += np.sum(
            maximum[:, None] >= np.abs(observed)[None, :] - 1.0e-15, axis=0
        )
    denominator = permutations + 1.0
    pointwise = (point_count + 1.0) / denominator
    return {
        "rho": observed,
        "p_pointwise": pointwise,
        "q_bh_120": bh(pointwise),
        "p_max_t_120": (max_count + 1.0) / denominator,
    }


def bootstrap_intervals(
    synergy: np.ndarray,
    score: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    cohort: np.ndarray,
    indices: np.ndarray,
    bootstraps: int,
) -> dict[int, tuple[float, float]]:
    rng = np.random.default_rng(SEED + 1)
    groups = [np.flatnonzero(cohort == value) for value in np.unique(cohort)]
    samples = np.empty((bootstraps, len(indices)), dtype=float)
    for bootstrap in range(bootstraps):
        sample = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        local_design = np.column_stack(
            [
                np.ones(len(sample)),
                rankdata(age[sample]),
                sex[sample],
                cohort[sample],
            ]
        )
        for position, coalition in enumerate(indices):
            samples[bootstrap, position] = adjusted_rank_correlation(
                synergy[sample, coalition], score[sample], local_design
            )
    bounds = np.quantile(samples, [0.025, 0.975], axis=0)
    return {
        int(coalition): (float(bounds[0, position]), float(bounds[1, position]))
        for position, coalition in enumerate(indices)
    }


def cohort_correlations(
    synergy: np.ndarray,
    score: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    cohort: np.ndarray,
    coalition: int,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for value, label in ((0, "original_29"), (1, "supplementary_28")):
        mask = cohort == value
        design = np.column_stack(
            [np.ones(int(mask.sum())), rankdata(age[mask]), sex[mask]]
        )
        result[label] = adjusted_rank_correlation(
            synergy[mask, coalition], score[mask], design
        )
    return result


def configure_style() -> None:
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


def make_figure(
    data: dict[str, Any],
    screen: dict[str, np.ndarray],
    top: np.ndarray,
    intervals: dict[int, tuple[float, float]],
    winner: int,
) -> None:
    configure_style()
    names = data["names"]
    sizes = data["sizes"]
    rho = screen["rho"]
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.65),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.95, 1.25, 1.05]},
    )

    axis = axes[0]
    rng = np.random.default_rng(SEED + 2)
    jitter = rng.uniform(-0.14, 0.14, len(sizes))
    axis.scatter(
        sizes + jitter,
        rho,
        s=14,
        color="#7895AA",
        alpha=0.64,
        edgecolor="none",
    )
    axis.scatter(
        sizes[winner],
        rho[winner],
        s=42,
        color="#B65F3C",
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    axis.axhline(0, color="#969696", linewidth=0.75)
    axis.set(
        xlabel="Coalition size",
        ylabel="Adjusted rank correlation, ρ",
        xticks=np.arange(2, 8),
        xlim=(1.6, 7.4),
    )
    axis.text(
        0.97,
        0.04,
        f"120 fixed coalitions\nmax |ρ| = {abs(rho[winner]):.3f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
    )

    axis = axes[1]
    display = top[::-1]
    y = np.arange(len(display))
    low = np.asarray([intervals[int(index)][0] for index in display])
    high = np.asarray([intervals[int(index)][1] for index in display])
    colors = np.where(screen["p_max_t_120"][display] < 0.05, "#B65F3C", "#466B83")
    for position, index in enumerate(display):
        axis.errorbar(
            rho[index],
            position,
            xerr=np.asarray([[rho[index] - low[position]], [high[position] - rho[index]]]),
            fmt="o",
            color=colors[position],
            ecolor="#A9BAC5",
            capsize=1.8,
            markersize=3.7,
            linewidth=0.8,
        )
    axis.axvline(0, color="#969696", linewidth=0.75)
    axis.set_yticks(y, [compact_name(str(names[index])) for index in display])
    axis.set_xlabel("Adjusted ρ (95% stratified bootstrap CI)")
    axis.tick_params(axis="y", length=0, labelsize=5.7)
    axis.text(
        0.98,
        0.03,
        "Top 10 by |ρ|",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color="#666666",
        fontsize=6.1,
    )

    axis = axes[2]
    design = np.column_stack(
        [
            np.ones(len(data["score"])),
            rankdata(data["age"]),
            data["sex"],
            data["cohort"],
        ]
    )
    x = projection_residual(rankdata(data["score"]), design)
    y_value = projection_residual(rankdata(data["synergy"][:, winner]), design)
    palette = ((0, "Original 29", "#617A9A"), (1, "Supplementary 28", "#D17A55"))
    for cohort_value, label, color in palette:
        mask = data["cohort"] == cohort_value
        axis.scatter(
            x[mask],
            y_value[mask],
            s=19,
            alpha=0.84,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            label=label,
        )
    coefficient = np.polyfit(x, y_value, 1)
    grid = np.linspace(float(x.min()), float(x.max()), 100)
    axis.plot(grid, np.polyval(coefficient, grid), color="#333333", linewidth=0.9)
    axis.set(
        xlabel="Residualized general-cognition rank",
        ylabel="Residualized REST Syn rank",
    )
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        fontsize=5.8,
        handletextpad=0.4,
    )
    axis.text(
        0.03,
        0.97,
        f"{compact_name(str(names[winner]))}\n"
        f"ρ={rho[winner]:+.3f}\n"
        f"120-coalition max-T p={screen['p_max_t_120'][winner]:.3f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
    )

    for letter, axis in zip("abc", axes, strict=True):
        axis.text(
            -0.17,
            1.08,
            letter,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=8,
            va="top",
        )
    stem = OUTPUT / "rest_general_cognition_coalition_correlations_57"
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_outputs(
    data: dict[str, Any],
    screen: dict[str, np.ndarray],
    top: np.ndarray,
    intervals: dict[int, tuple[float, float]],
    winner: int,
    permutations: int,
    bootstraps: int,
) -> dict[str, Any]:
    names = data["names"]
    rho = screen["rho"]
    raw_rho = np.asarray(
        [spearmanr(data["score"], data["synergy"][:, index]).statistic for index in range(120)]
    )
    rows = []
    for index, name in enumerate(names):
        row = {
            "coalition": str(name),
            "coalition_size": int(data["sizes"][index]),
            "rho_adjusted": float(rho[index]),
            "rho_unadjusted": float(raw_rho[index]),
            "p_permutation_pointwise": float(screen["p_pointwise"][index]),
            "q_bh_120": float(screen["q_bh_120"][index]),
            "p_max_t_120": float(screen["p_max_t_120"][index]),
            "mean_syn_bits": float(data["synergy"][:, index].mean()),
            "minimum_syn_bits": float(data["synergy"][:, index].min()),
        }
        if index in intervals:
            row["bootstrap_95_ci"] = list(intervals[index])
        rows.append(row)
    winner_row = dict(rows[winner])
    winner_row["cohort_rho_age_sex_adjusted"] = cohort_correlations(
        data["synergy"],
        data["score"],
        data["age"],
        data["sex"],
        data["cohort"],
        winner,
    )
    loo = []
    for omitted in range(57):
        keep = np.arange(57) != omitted
        local_design = np.column_stack(
            [
                np.ones(int(keep.sum())),
                rankdata(data["age"][keep]),
                data["sex"][keep],
                data["cohort"][keep],
            ]
        )
        loo.append(
            adjusted_rank_correlation(
                data["synergy"][keep, winner],
                data["score"][keep],
                local_design,
            )
    )
    winner_row["leave_one_out_rho_range"] = [float(np.min(loo)), float(np.max(loo))]

    summary = {
        "experiment": "REST fixed-coalition synergy and general cognition in 57 HCP subjects",
        "exploratory": True,
        "n_subjects": 57,
        "cohorts": {"original": 29, "supplementary": 28},
        "configuration": {
            "parcellation": "Schaefer-1000 / Yeo-7 cortex",
            "state": "REST1_LR",
            "representation": "one PC per Yeo-7 network",
            "history_order": 3,
            "ridge_alpha": 1.0,
            "brain_metric": "affine Gaussian TM fixed-coalition Syn",
            "coalitions": 120,
            "syn_nonnegative_tolerance_bits": SYN_TOLERANCE_BITS,
        },
        "general_cognition": {
            "source": "frozen single-group SEM factor scores estimated in the 1,206-subject sample",
            "minimum": float(data["score"].min()),
            "mean": float(data["score"].mean()),
            "maximum": float(data["score"].max()),
        },
        "inference": {
            "statistic": "partial Spearman correlation in rank space",
            "covariates": ["age", "sex", "recruitment cohort"],
            "permutation_scheme": "Freedman-Lane residual permutation within original/supplementary cohort",
            "permutations": permutations,
            "bootstraps": bootstraps,
            "bootstrap_scheme": "stratified resampling within original/supplementary cohort",
            "multiplicity": ["BH across 120 coalitions", "max-T across 120 coalitions"],
        },
        "nonnegativity_audit": {
            "tolerance_bits": SYN_TOLERANCE_BITS,
            "checked_count": int(data["synergy"].size),
            "minimum_bits": float(data["synergy"].min()),
            "within_tolerance_negative_count": int(
                np.sum((data["synergy"] < 0) & (data["synergy"] >= -SYN_TOLERANCE_BITS))
            ),
            "significant_violation_count": int(
                np.sum(data["synergy"] < -SYN_TOLERANCE_BITS)
            ),
        },
        "counts": {
            "pointwise_p_below_0_05": int(np.sum(screen["p_pointwise"] < 0.05)),
            "bh_q_below_0_05": int(np.sum(screen["q_bh_120"] < 0.05)),
            "max_t_p_below_0_05": int(np.sum(screen["p_max_t_120"] < 0.05)),
        },
        "winner": winner_row,
        "top_10": [rows[int(index)] for index in top],
        "limitations": [
            "Selection and effect estimation use the same 57 subjects.",
            "HCP family identifiers, head-motion summaries, and physiological covariates are not modeled.",
            "The supplementary 28 subjects were selected for behavioral diversity, so cohort is adjusted and permutations are cohort-blocked.",
            "Only the frozen p=3, alpha=1 primary configuration and REST1_LR run are tested.",
        ],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(rows)[
        [
            "coalition",
            "coalition_size",
            "rho_adjusted",
            "rho_unadjusted",
            "p_permutation_pointwise",
            "q_bh_120",
            "p_max_t_120",
            "mean_syn_bits",
            "minimum_syn_bits",
        ]
    ].to_csv(
        OUTPUT / "rest_general_cognition_all_coalitions.tsv", sep="\t", index=False
    )
    selected = pd.DataFrame(
        {
            "subject": data["subjects"],
            "cohort": np.where(data["cohort"] == 0, "original_29", "supplementary_28"),
            "age_midpoint": data["age"],
            "sex": np.where(data["sex"] == 1, "M", "F"),
            "general_cognition_score": data["score"],
            "winner_coalition": str(names[winner]),
            "winner_rest_syn_bits": data["synergy"][:, winner],
        }
    )
    selected.to_csv(
        OUTPUT / "rest_general_cognition_winner_source_data.tsv", sep="\t", index=False
    )
    contract = {
        "scientific_question": "Does any fixed Yeo-7 REST coalition covary with the frozen general-cognition factor in the 57-subject cohort?",
        "pairing_unit": "subject",
        "primary_brain_family": "all 120 fixed Yeo-7 coalitions",
        "primary_outcome": "general cognition SEM factor score",
        "statistics": summary["inference"],
        "figure_contract": {
            "core_conclusion": "Show whether the largest REST coalition-cognition association survives cohort-aware 120-coalition correction.",
            "evidence_chain": {
                "a": "all 120 adjusted correlations by coalition size",
                "b": "top ten adjusted correlations with stratified bootstrap intervals",
                "c": "residual association and cohort support for the selected coalition",
            },
            "archetype": "quantitative grid",
            "role": "exploratory biological relevance",
            "backend": "Python/matplotlib",
            "final_size": "double-column, 183 mm wide",
            "exports": ["PNG", "SVG", "PDF"],
            "reviewer_risks": summary["limitations"],
        },
    }
    (OUTPUT / "experiment_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# REST 脑网络组合协同与一般认知（57 人）",
        "",
        "![REST 固定网络组合协同与一般认知](rest_general_cognition_coalition_correlations_57.png)",
        "",
        "主分析沿用 Schaefer-1000、Yeo-7 网络 PC1、三阶历史和 Ridge $\\alpha=1$。对 120 个固定网络组合的 REST Syn 与冻结的 SEM 一般认知因子做秩相关，并控制年龄、性别和原 29 人/新增 28 人批次。置换采用批次内 Freedman--Lane 方案。",
        "",
        "## 结果",
        "",
        f"绝对相关最大的组合为 **{winner_row['coalition']}**：调整后 $\\rho={winner_row['rho_adjusted']:+.3f}$，点对点置换 $p={winner_row['p_permutation_pointwise']:.4f}$，120 组合 BH $q={winner_row['q_bh_120']:.4f}$，max-$T$ $p={winner_row['p_max_t_120']:.4f}$，分层 bootstrap 95% CI $[{winner_row['bootstrap_95_ci'][0]:+.3f},{winner_row['bootstrap_95_ci'][1]:+.3f}]$。",
        "",
        f"120 个组合中，点对点 $p<0.05$ 的有 {summary['counts']['pointwise_p_below_0_05']} 个；BH 与 max-$T$ 校正后分别有 {summary['counts']['bh_q_below_0_05']} 和 {summary['counts']['max_t_p_below_0_05']} 个。",
        "",
        "| 排名 | 网络组合 | 调整后 $\\rho$ | 95% CI | 点对点 $p$ | BH $q$ | max-$T$ $p$ |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, index in enumerate(top, start=1):
        row = rows[int(index)]
        low, high = intervals[int(index)]
        report.append(
            f"| {rank} | {row['coalition']} | {row['rho_adjusted']:+.3f} | "
            f"[{low:+.3f}, {high:+.3f}] | {row['p_permutation_pointwise']:.4f} | "
            f"{row['q_bh_120']:.4f} | {row['p_max_t_120']:.4f} |"
        )
    report.extend(
        [
            "",
            "## 解释边界",
            "",
            "这是同一 57 人样本内的探索性筛查。只有 120 组合 max-$T$ 可支持组合家族层面的推断；若最高关联未通过校正，就只能作为候选，不能据此声称 REST 某一网络组合稳定解释一般认知。分析没有控制 HCP 家系、头动或生理变量，也没有检验 RL run 与超参数稳定性。",
            "",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--bootstraps", type=int, default=20_000)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    design = np.column_stack(
        [np.ones(57), rankdata(data["age"]), data["sex"], data["cohort"]]
    )
    screen = permutation_screen(
        data["synergy"], data["score"], design, data["cohort"], args.permutations
    )
    top = np.argsort(-np.abs(screen["rho"]))[:10]
    winner = int(top[0])
    intervals = bootstrap_intervals(
        data["synergy"],
        data["score"],
        data["age"],
        data["sex"],
        data["cohort"],
        top,
        args.bootstraps,
    )
    summary = write_outputs(
        data, screen, top, intervals, winner, args.permutations, args.bootstraps
    )
    make_figure(data, screen, top, intervals, winner)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
