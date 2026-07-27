#!/usr/bin/env python3
"""Plot a compact, interpretable subset of the exhaustive HCP cognition screen."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = (
    ROOT / "results/hcp_cognition_exhaustive_targeted_greedy/metrics.npz"
)
ASSOCIATIONS_PATH = (
    ROOT
    / "results/hcp_cognition_exhaustive_targeted_greedy/all_associations.jsonl"
)
COGNITION_PATH = (
    ROOT / "results/hcp_single_group_sem_full_1206/selected_29_sem_results.csv"
)
OUTPUT_DIR = ROOT / "results/hcp_cognition_exhaustive_targeted_greedy"
OUTPUT_STEM = OUTPUT_DIR / "extended_task_aligned_correlations"
SUMMARY_PATH = OUTPUT_DIR / "extended_interpretable_candidates.json"

METRIC = "targeted_first_residual"
SCORE_LABELS = {
    "cry_score": "Crystallized cognition",
    "mem_score": "Memory",
    "spd_score": "Processing speed",
}
SCORE_COLORS = {
    "cry_score": "#B86A4B",
    "mem_score": "#4F8A78",
    "spd_score": "#587DA5",
}
NETWORK_SHORT = {
    "Vis": "Vis",
    "SomMot": "Som",
    "DorsAttn": "DAN",
    "SalVentAttn": "SVAN",
    "Limbic": "Lim",
    "Cont": "FPN",
    "Default": "DMN",
}

# These candidates are intentionally selected as four non-identical, task-interpretable
# examples per construct. They are not treated as twelve independent discoveries.
SELECTED = {
    "cry_score": (
        ("LANGUAGE", "Vis+SomMot"),
        ("EMOTION", "DorsAttn+Limbic"),
        ("REST", "DorsAttn+SalVentAttn+Cont"),
        ("WM", "DorsAttn+SalVentAttn+Cont"),
    ),
    "mem_score": (
        ("SOCIAL", "SalVentAttn+Limbic+Default"),
        ("SOCIAL", "Limbic+Default"),
        ("RELATIONAL", "Vis+SalVentAttn"),
        ("EMOTION", "Vis+DorsAttn+SalVentAttn+Default"),
    ),
    "spd_score": (
        ("RELATIONAL", "Vis+Limbic+Cont"),
        ("LANGUAGE", "SomMot+Limbic"),
        ("SOCIAL", "DorsAttn+Cont"),
        ("MOTOR", "DorsAttn+SalVentAttn+Limbic+Cont+Default"),
    ),
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def short_coalition(coalition: str) -> str:
    return "+".join(NETWORK_SHORT[name] for name in coalition.split("+"))


def load_aligned_inputs() -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    with np.load(METRICS_PATH) as archive:
        metrics = {key: np.asarray(archive[key]) for key in archive.files}
    subjects = np.asarray(
        [str(subject).removeprefix("sub-") for subject in metrics["subjects"]]
    )
    cognition = pd.read_csv(COGNITION_PATH, dtype={"Subject": str})
    cognition["Subject"] = cognition["Subject"].str.removeprefix("sub-")
    cognition = cognition.set_index("Subject")
    if len(subjects) != 29 or set(subjects) != set(cognition.index):
        raise ValueError("Expected exact alignment of the same 29 cognition-imaging subjects")
    cognition = cognition.loc[subjects, list(SCORE_LABELS)]
    if not np.isfinite(cognition.to_numpy(dtype=float)).all():
        raise ValueError("Cognition scores contain non-finite values")
    return metrics, cognition


def association_lookup() -> dict[tuple[str, str, str], dict[str, object]]:
    rows = [
        json.loads(line)
        for line in ASSOCIATIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        (str(row["score"]), str(row["state"]), str(row["coalition"])): row
        for row in rows
        if row["metric"] == METRIC
    }


def leave_one_out(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    full = float(spearmanr(x, y).statistic)
    estimates = []
    for index in range(len(x)):
        keep = np.arange(len(x)) != index
        estimates.append(float(spearmanr(x[keep], y[keep]).statistic))
    values = np.asarray(estimates)
    return {
        "minimum_rho": float(values.min()),
        "median_rho": float(np.median(values)),
        "maximum_rho": float(values.max()),
        "same_direction_fraction": float(np.mean(np.sign(values) == np.sign(full))),
    }


def pointwise_permutation_p(
    x: np.ndarray, y: np.ndarray, *, repeats: int = 20_000, seed: int = 2026072701
) -> float:
    observed = abs(float(spearmanr(x, y).statistic))
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(repeats):
        estimate = abs(float(spearmanr(rng.permutation(x), y).statistic))
        exceedances += estimate >= observed - 1.0e-15
    return float((exceedances + 1) / (repeats + 1))


def main() -> None:
    configure_style()
    metrics, cognition = load_aligned_inputs()
    lookup = association_lookup()
    states = metrics["states"].astype(str).tolist()
    coalitions = metrics["coalitions"].astype(str).tolist()
    values = np.asarray(metrics[METRIC], dtype=float)

    figure, axes = plt.subplots(
        3,
        4,
        figsize=(7.2, 6.45),
        constrained_layout=True,
        gridspec_kw={"hspace": 0.22, "wspace": 0.16},
    )
    summary_rows: list[dict[str, object]] = []
    panel_index = 0

    for row_index, score in enumerate(SCORE_LABELS):
        x = cognition[score].to_numpy(dtype=float)
        color = SCORE_COLORS[score]
        for column_index, (state, coalition) in enumerate(SELECTED[score]):
            axis = axes[row_index, column_index]
            state_index = states.index(state)
            coalition_index = coalitions.index(coalition)
            y = values[state_index, :, coalition_index]
            association = lookup[(score, state, coalition)]
            if (
                int(association["positive_subjects"]) != 29
                or int(association["negative_subjects"]) != 0
                or int(association["near_zero_subjects"]) != 0
                or float(association["p_raw_two_sided"]) >= 0.05
                or float(association["p_permutation_pointwise"]) >= 0.05
            ):
                raise AssertionError(f"Candidate does not satisfy the declared screen: {association}")

            axis.scatter(
                x,
                y,
                s=19,
                color=color,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
            slope, intercept = np.polyfit(x, y, 1)
            guide_x = np.linspace(float(x.min()), float(x.max()), 100)
            axis.plot(
                guide_x,
                slope * guide_x + intercept,
                color=color,
                alpha=0.62,
                linewidth=0.8,
                linestyle=(0, (2.2, 2.2)),
                zorder=2,
            )
            axis.set_title(
                f"{state} · {short_coalition(coalition)}\n"
                rf"$\rho$={float(association['rho']):+.3f}; "
                rf"$p$={float(association['p_raw_two_sided']):.3g}; "
                rf"$p_{{perm}}$={float(association['p_permutation_pointwise']):.3g}",
                fontsize=6.25,
                pad=4.0,
            )
            axis.text(
                -0.20,
                1.16,
                chr(ord("a") + panel_index),
                transform=axis.transAxes,
                fontweight="bold",
                fontsize=8.2,
                ha="left",
                va="top",
            )
            if column_index == 0:
                axis.set_ylabel("Targeted residual (bits)")
            if row_index == 2:
                axis.set_xlabel("Factor score")
            axis.margins(x=0.08, y=0.12)
            panel_index += 1

            row = dict(association)
            row["leave_one_out"] = leave_one_out(x, y)
            summary_rows.append(row)

        axes[row_index, 0].text(
            -0.47,
            0.5,
            SCORE_LABELS[score],
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            fontweight="bold",
            fontsize=7.2,
            ha="center",
            va="center",
            color=color,
        )

    figure.text(
        0.5,
        -0.012,
        "All panels: n=29; Spearman tests are exploratory. "
        "Every candidate has raw and pointwise permutation p<0.05, "
        "but none survives the 2,872-feature BH/maxT correction.",
        ha="center",
        va="top",
        fontsize=5.8,
        color="#4D4D4D",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    crystal_x = cognition["cry_score"].to_numpy(dtype=float)
    crystal_coalition = coalitions.index("DorsAttn+SalVentAttn+Cont")
    rest_minus_wm = (
        values[states.index("REST"), :, crystal_coalition]
        - values[states.index("WM"), :, crystal_coalition]
    )
    contrast_result = spearmanr(crystal_x, rest_minus_wm)

    payload = {
        "figure_contract": {
            "core_conclusion": (
                "Fully observed exploratory correlations organize into a "
                "task-dependent crystallized-cognition pattern, a consistently positive "
                "memory pattern, and a predominantly negative processing-speed pattern."
            ),
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib",
            "subjects": 29,
            "metric": METRIC,
            "selection": (
                "Four task-interpretable, non-identical examples per construct; "
                "raw and pointwise permutation p<0.05; 29/29 positive metric values."
            ),
            "review_risks": [
                "post hoc selection in the same cohort",
                "correlated and nested coalitions",
                "no feature survives within-score BH or maxT correction",
                "unrestricted rather than family-blocked permutations",
                "no covariate adjustment",
                "the three original top candidates weaken at Schaefer-1000",
            ],
        },
        "task_contrasts": [
            {
                "score": "cry_score",
                "contrast": "REST_minus_WM",
                "coalition": "DorsAttn+SalVentAttn+Cont",
                "rho": float(contrast_result.statistic),
                "p_raw_two_sided": float(contrast_result.pvalue),
                "p_permutation_pointwise": pointwise_permutation_p(
                    crystal_x, rest_minus_wm
                ),
                "interpretation_boundary": (
                    "Post hoc task-contrast support; not corrected for candidate selection."
                ),
            }
        ],
        "candidates": summary_rows,
    }
    SUMMARY_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
