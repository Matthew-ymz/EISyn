#!/usr/bin/env python3
"""Screen GAMBLING coalition Syn against an intertemporal reward-valuation score."""

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

import scripts.screen_hcp_motor_composite_scores_57 as common


BEHAVIOR = ROOT / "data/unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
OUTPUT = ROOT / "results/hcp_gambling_reward_valuation_57"
CACHE = OUTPUT / "gambling_coalition_synergy_57.npz"
PARTIAL_CACHE = OUTPUT / "gambling_coalition_synergy_57.partial.npz"

COMPONENT_FIELDS = {
    "AUC $200": "DDisc_AUC_200",
    "AUC $40,000": "DDisc_AUC_40K",
}
SEED = 20260803
PERMUTATIONS = 100_000
BOOTSTRAPS = 20_000


def load_scores(subjects: np.ndarray) -> dict[str, np.ndarray]:
    with BEHAVIOR.open(newline="", encoding="utf-8-sig") as handle:
        table = {str(row["Subject"]): row for row in csv.DictReader(handle)}
    rows = [table[str(subject).removeprefix("sub-")] for subject in subjects]
    raw = np.column_stack(
        [
            np.asarray([float(row[field]) for row in rows], dtype=float)
            for field in COMPONENT_FIELDS.values()
        ]
    )
    if raw.shape != (57, 2) or not np.isfinite(raw).all():
        raise ValueError("The frozen 57-subject sample must have both delay-discounting AUCs.")
    means = raw.mean(axis=0)
    standard_deviations = raw.std(axis=0, ddof=1)
    standardized = (raw - means) / standard_deviations
    composite = standardized.mean(axis=1)
    reward_larger = np.asarray(
        [float(row["Gambling_Task_Reward_Perc_Larger"]) for row in rows]
    )
    punish_larger = np.asarray(
        [float(row["Gambling_Task_Punish_Perc_Larger"]) for row in rows]
    )
    return {
        "components_raw": raw,
        "components_z": standardized,
        "component_means": means,
        "component_sds": standard_deviations,
        "composite": composite,
        "reward_minus_punish_larger_pp": reward_larger - punish_larger,
        "age": np.asarray([common.age_midpoint(row["Age"]) for row in rows]),
        "sex": np.asarray([row["Gender"] == "M" for row in rows], dtype=float),
        "cohort": np.r_[np.zeros(29, dtype=int), np.ones(28, dtype=int)],
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


def residual_rank_values(
    brain: np.ndarray, endpoint: np.ndarray, scores: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    design = common.base_design(scores)
    return (
        common.residualize(rankdata(brain), design),
        common.residualize(rankdata(endpoint), design),
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
    figure = plt.figure(figsize=(7.2, 5.15), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=[0.96, 1.16])
    axes = [
        figure.add_subplot(grid[row, column])
        for row, column in ((0, 0), (0, 1), (1, 0), (1, 1))
    ]

    behavior = np.column_stack([scores["components_raw"], scores["composite"]])
    labels = [*COMPONENT_FIELDS.keys(), "Composite"]
    correlations = np.asarray(spearmanr(behavior, axis=0).statistic)
    image = axes[0].imshow(correlations, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_xticks(np.arange(3), labels, rotation=30, ha="right")
    axes[0].set_yticks(np.arange(3), labels)
    for row in range(3):
        for column in range(3):
            value = correlations[row, column]
            axes[0].text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.62 else "#30343B",
                fontsize=6.2,
            )
    axes[0].set_title("a  Reward-valuation score", loc="left", fontweight="bold")
    colorbar = figure.colorbar(image, ax=axes[0], shrink=0.72, pad=0.02)
    colorbar.set_label(r"Spearman $\rho$")

    rng = np.random.default_rng(SEED)
    x = sizes + rng.uniform(-0.10, 0.10, len(sizes))
    axes[1].axhline(0, color="#D6DADF", linewidth=0.7)
    axes[1].scatter(
        x,
        result["rho_adjusted"],
        s=20,
        color="#7890A8",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.3,
    )
    axes[1].scatter(
        sizes[winner],
        result["rho_adjusted"][winner],
        s=66,
        marker="D",
        facecolor="#C56F4D",
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
    )
    axes[1].annotate(
        common.compact_name(str(names[winner])),
        (sizes[winner], result["rho_adjusted"][winner]),
        xytext=(5, 5),
        textcoords="offset points",
        color="#91452E",
        fontsize=6.4,
    )
    axes[1].set(
        xlabel="Coalition size (Yeo7 networks)",
        ylabel=r"Adjusted association with reward valuation ($\rho$)",
    )
    axes[1].set_xticks(np.arange(2, 8))
    axes[1].set_title("b  All 120 GAMBLING coalitions", loc="left", fontweight="bold")

    ordered = list(reversed(top))
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
    winner_position = ordered.index(winner)
    axes[2].scatter(
        centers[winner_position],
        winner_position,
        s=34,
        marker="D",
        color="#C56F4D",
        zorder=4,
    )
    axes[2].set_yticks(y, [rows[index]["short_coalition"] for index in ordered])
    axes[2].set_xlabel(r"Adjusted $\rho$ (stratified bootstrap 95% CI)")
    axes[2].set_title("c  Ten strongest associations", loc="left", fontweight="bold")

    brain_residual, score_residual = residual_rank_values(
        matrix[:, winner], scores["composite"], scores
    )
    cohort = scores["cohort"].astype(int)
    colors = ("#7890A8", "#C56F4D")
    for cohort_value, label in ((0, "Original 29"), (1, "Supplement 28")):
        mask = cohort == cohort_value
        axes[3].scatter(
            score_residual[mask],
            brain_residual[mask],
            s=21,
            color=colors[cohort_value],
            alpha=0.84,
            edgecolor="white",
            linewidth=0.35,
            label=label,
        )
    order = np.argsort(score_residual)
    coefficient = np.polyfit(score_residual, brain_residual, 1)
    axes[3].plot(
        score_residual[order],
        np.polyval(coefficient, score_residual[order]),
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
        xlabel="Reward-valuation score (adjusted rank residual)",
        ylabel="Coalition Syn (adjusted rank residual)",
    )
    axes[3].set_title(
        f"d  {common.compact_name(str(names[winner]))}",
        loc="left",
        fontweight="bold",
    )
    axes[3].legend(loc="center left", bbox_to_anchor=(1.02, 0.50), frameon=False)

    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        figure.savefig(
            OUTPUT / f"gambling_reward_valuation_coalition_screen_57.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(figure)


def write_source_data(
    subjects: np.ndarray,
    scores: Mapping[str, np.ndarray],
    matrix: np.ndarray,
    winner: int,
) -> None:
    fields = [
        "subject",
        "cohort",
        "age_midpoint",
        "sex_male",
        "ddisc_auc_200",
        "ddisc_auc_40k",
        "reward_valuation_score_z",
        "reward_minus_punish_larger_percentage_points",
        "winner_syn_bits",
    ]
    with (OUTPUT / "winner_source_data.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(fields)
        for index, subject in enumerate(subjects):
            writer.writerow(
                [
                    subject,
                    int(scores["cohort"][index]),
                    scores["age"][index],
                    int(scores["sex"][index]),
                    *scores["components_raw"][index].tolist(),
                    scores["composite"][index],
                    scores["reward_minus_punish_larger_pp"][index],
                    matrix[index, winner],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    common.CACHE = CACHE
    common.PARTIAL_CACHE = PARTIAL_CACHE
    common.SEED = SEED
    subjects = common.load_subjects()
    combinations = common.coalitions()
    names = common.coalition_names(combinations)
    sizes = np.asarray([len(value) for value in combinations])
    matrix, heldout, explained = common.compute_matrix(
        subjects, combinations, args.recompute, state="GAMBLING"
    )
    scores = load_scores(subjects)
    result = common.screen(matrix, scores, args.permutations, SEED)
    rows, winner, top = common.make_rows(
        names, sizes, matrix, scores, result, args.bootstraps
    )
    winner_brain = matrix[:, winner]
    component_correlations = {
        name: common.adjusted_rho(
            winner_brain, scores["components_raw"][:, index], scores
        )
        for index, name in enumerate(COMPONENT_FIELDS)
    }
    winner_row = rows[winner]
    winner_row["leave_one_out_adjusted_rho"] = common.leave_one_out(
        winner_brain, scores["composite"], scores
    )
    winner_row["component_adjusted_rho"] = component_correlations
    winner_row["gambling_choice_shift_adjusted_rho"] = common.adjusted_rho(
        winner_brain, scores["reward_minus_punish_larger_pp"], scores
    )

    score_component_correlations = {
        name: float(
            spearmanr(
                scores["composite"], scores["components_raw"][:, index]
            ).statistic
        )
        for index, name in enumerate(COMPONENT_FIELDS)
    }
    summary = {
        "experiment": "GAMBLING fixed-coalition Syn screen against intertemporal reward valuation",
        "subjects": 57,
        "score_definition": {
            "formula": "mean[z(DDisc_AUC_200), z(DDisc_AUC_40K)]",
            "standardization_reference": "Frozen 57-subject imaging sample; sample SD with ddof=1",
            "direction": "Higher values indicate weaker temporal discounting / greater delayed-reward valuation.",
            "task_score_boundary": "The HCP GAMBLING outcomes were predetermined, so choice proportions are not accuracy scores. The primary endpoint is an out-of-scanner reward-decision phenotype.",
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
            "cronbach_alpha": common.cronbach_alpha(scores["components_z"]),
            "composite_component_spearman": score_component_correlations,
        },
        "inference": {
            "permutations": args.permutations,
            "bootstraps": args.bootstraps,
            "permutation_seed": SEED,
            "bootstrap_seed_base": SEED,
            "covariates": ["age", "sex", "original/supplement cohort"],
            "scheme": "Freedman-Lane residual permutation within original/supplement cohort",
            "primary_family": "120 fixed Yeo7 coalitions for one frozen reward-valuation endpoint",
            "selection_status": "Exploratory: coalition selection and effect estimation use the same 57 subjects.",
        },
        "model": {
            "parcellation": "Schaefer-1000 / Yeo-7 cortex",
            "state": "full GAMBLING LR run",
            "representation": "network PC1 fitted to taskRetained-taskRegressed and projected onto taskRetained",
            "history_order": common.ORDER,
            "ridge_alpha": common.ALPHA,
            "estimator": "affine Gaussian TM fixed-coalition Syn",
            "mean_heldout_skill_ratio": float(heldout.mean()),
            "models_better_than_persistence": int(np.sum(heldout < 1)),
            "mean_pc1_explained": float(explained.mean()),
            "syn_nonnegative_tolerance_bits": common.SYN_TOLERANCE_BITS,
            "minimum_synergy_bits": float(matrix.min()),
            "negative_within_tolerance_count": int(
                np.sum(
                    (matrix < 0)
                    & (matrix >= -common.SYN_TOLERANCE_BITS)
                )
            ),
            "significant_negative_count": int(
                np.sum(matrix < -common.SYN_TOLERANCE_BITS)
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
    contract = {
        "scientific_question": "Which fixed Yeo-7 coalition changes most strongly with delayed-reward valuation when only coalition membership changes?",
        "pairing_unit": "subject",
        "primary_behavior": summary["score_definition"]["formula"],
        "primary_brain_metric": "full-run GAMBLING fixed-coalition TM Syn",
        "treatment_factor": "Yeo-7 coalition membership (120 levels)",
        "frozen_variables": summary["model"],
        "statistics": summary["inference"],
        "figure_contract": {
            "core_conclusion": "Show whether any GAMBLING coalition association with delayed-reward valuation survives 120-coalition correction.",
            "evidence_chain": [
                "score coherence",
                "all-coalition screen",
                "top-ten uncertainty",
                "winner adjusted association",
            ],
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib",
            "exports": ["PNG 600 dpi", "editable SVG", "PDF"],
            "review_risks": [
                "out-of-scanner phenotype rather than task accuracy",
                "post-selection effect inflation",
                "cortical atlas excludes reward-relevant subcortex",
            ],
        },
    }
    (OUTPUT / "experiment_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_source_data(subjects, scores, matrix, winner)
    plot(names, sizes, matrix, scores, result, rows, winner, top)

    lines = [
        "# HCP GAMBLING reward-valuation coalition screen",
        "",
        "![GAMBLING reward-valuation screen](gambling_reward_valuation_coalition_screen_57.png)",
        "",
        "The HCP gambling outcomes were predetermined, so larger/smaller choice percentages are not accuracy. The primary score is the equal-weight mean of sample-standardized delay-discounting AUCs at the $200 and $40,000 magnitudes; higher values indicate greater delayed-reward valuation.",
        "",
        "| Coalition | Adjusted rho | Original 29 rho | Supplement 28 rho | Raw p | BH q | 120-coalition max-T p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index in top:
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


if __name__ == "__main__":
    main()
