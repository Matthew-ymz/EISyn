#!/usr/bin/env python3
"""Robustness matrix for the five frozen task-score synergy candidates."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import NormalDist

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, zscore


ROOT = Path(__file__).resolve().parents[1]
METRICS_500 = ROOT / "results" / "hcp_cognition_exhaustive_targeted_greedy" / "metrics.npz"
VALIDATION_DIR = ROOT / "results" / "hcp_task_score_synergy_schaefer1000_validation"
METRICS_1000 = VALIDATION_DIR / "fixed_candidates_schaefer1000.npz"
VALIDATION_SUMMARY = VALIDATION_DIR / "summary.json"
BEHAVIOR = ROOT / "Data" / "unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
OUTPUT_DIR = ROOT / "results" / "hcp_task_score_synergy_robustness"

PERMUTATIONS = 50_000
SEED = 20260730
SHORT = {
    "Vis": "Vis", "SomMot": "Som", "DorsAttn": "DAN",
    "SalVentAttn": "SVAN", "Limbic": "Lim", "Cont": "Cont",
    "Default": "Def",
}


def short_name(value: str) -> str:
    return "+".join(SHORT[item] for item in value.split("+"))


def holm(values: list[float]) -> np.ndarray:
    p = np.asarray(values)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate(
        np.minimum(1.0, ranked * (len(p) - np.arange(len(p))))
    )
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def residualize(values: np.ndarray, design: np.ndarray | None) -> np.ndarray:
    ranks = rankdata(values, method="average")
    if design is not None:
        ranks = ranks - design @ np.linalg.lstsq(design, ranks, rcond=None)[0]
    centered = ranks - ranks.mean()
    return centered / np.sqrt(np.sum(centered**2))


def permutation_association(
    x: np.ndarray,
    y: np.ndarray,
    permutations: np.ndarray,
    design: np.ndarray | None,
) -> tuple[float, float]:
    x_residual = residualize(x, design)
    y_residual = residualize(y, design)
    rho = float(x_residual @ y_residual)
    null = x_residual[permutations] @ y_residual
    p = float(
        (1 + np.count_nonzero(np.abs(null) >= abs(rho))) / (len(null) + 1)
    )
    return rho, p


def age_midpoint(value: str) -> float:
    if value == "36+":
        return 38.0
    low, high = value.split("-")
    return 0.5 * (float(low) + float(high))


def required_n(rho: float, alpha: float = 0.01, power: float = 0.80) -> int:
    """Fisher-z approximation for a two-sided correlation test."""
    normal = NormalDist()
    critical = normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)
    return int(np.ceil((critical / np.arctanh(abs(rho))) ** 2 + 3))


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = np.load(METRICS_500)
    target = np.load(METRICS_1000)
    validation = json.loads(VALIDATION_SUMMARY.read_text(encoding="utf-8"))
    subjects = target["subjects"].astype(str).tolist()
    if source["subjects"].astype(str).tolist() != subjects:
        raise ValueError("Subject order mismatch")
    with BEHAVIOR.open(newline="", encoding="utf-8-sig") as handle:
        table = {str(row["Subject"]): row for row in csv.DictReader(handle)}
    clean_subjects = [subject.removeprefix("sub-") for subject in subjects]
    design = np.column_stack([
        np.ones(29),
        [age_midpoint(table[subject]["Age"]) for subject in clean_subjects],
        [table[subject]["Gender"] == "M" for subject in clean_subjects],
    ]).astype(float)
    rng = np.random.default_rng(SEED)
    permutations = np.vstack([rng.permutation(29) for _ in range(PERMUTATIONS)])
    source_states = source["states"].astype(str).tolist()
    source_coalitions = source["coalitions"].astype(str).tolist()

    methods = (
        ("s500_unadjusted", "S500 unadjusted", None, "500"),
        ("s500_age_sex", "S500 age/sex-adjusted", design, "500"),
        ("s1000_unadjusted", "S1000 unadjusted", None, "1000"),
        ("s1000_age_sex", "S1000 age/sex-adjusted", design, "1000"),
        ("consensus_age_sex", "Cross-atlas consensus, adjusted", design, "consensus"),
    )
    results: dict[str, list[dict[str, float | str | int | bool]]] = {
        key: [] for key, _, _, _ in methods
    }
    for task_index, item in enumerate(validation["results"]):
        state = item["state"]
        coalition = item["coalition"]
        if state == "SOCIAL":
            score = np.asarray([
                0.5 * (
                    float(table[subject]["Social_Task_Random_Perc_Random"])
                    + float(table[subject]["Social_Task_TOM_Perc_TOM"])
                )
                for subject in clean_subjects
            ])
        else:
            score = np.asarray([
                float(table[subject][item["field"]]) for subject in clean_subjects
            ])
        y500 = source["fixed_block_synergy"][
            source_states.index(state), :, source_coalitions.index(coalition)
        ]
        y1000 = target["values"][task_index]
        values = {
            "500": y500,
            "1000": y1000,
            "consensus": 0.5 * (zscore(y500) + zscore(y1000)),
        }
        for method, _, covariates, value_key in methods:
            rho, p = permutation_association(
                score, values[value_key], permutations, covariates
            )
            results[method].append({
                "state": state,
                "task_label": item["label"],
                "coalition": coalition,
                "n_subjects": 29,
                "rho": rho,
                "p_permutation": p,
                "holm_p": 0.0,
            })

    for method, _, _, _ in methods:
        adjusted = holm([
            float(item["p_permutation"]) for item in results[method]
        ])
        for item, value in zip(results[method], adjusted, strict=True):
            item["holm_p"] = float(value)

    power = []
    for item in validation["results"]:
        power.append({
            "state": item["state"],
            "task_label": item["label"],
            "coalition": item["coalition"],
            "discovery_rho": item["rho_500"],
            "required_n_80pct_bonferroni_five": required_n(
                float(item["rho_500"]), alpha=0.01, power=0.80
            ),
        })
    payload = {
        "experiment": "Five-candidate robustness matrix",
        "subjects": 29,
        "permutations": PERMUTATIONS,
        "covariates": ["age-bin midpoint", "gender"],
        "warning": (
            "S500 and consensus analyses remain selection-aware sensitivities because "
            "candidate coalitions were selected using S500 behavior. S1000 is the "
            "fixed cross-atlas validation."
        ),
        "methods": results,
        "power_projection": power,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Five-candidate robustness matrix",
        "",
        "All analyses use the same 29 subjects. Holm correction is applied across "
        "the five tasks separately within each analysis family.",
        "",
        "| Method | Emotion | Language | Relational | Social | Working memory |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, label, _, _ in methods:
        cells = [
            f"{item['rho']:+.3f} (Holm {item['holm_p']:.3f})"
            for item in results[method]
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "The Schaefer-1000 rows are the valid fixed-candidate cross-atlas tests. "
        "Schaefer-500 adjusted and consensus rows are robustness sensitivities, not "
        "independent confirmation, because candidate selection used Schaefer-500.",
        "",
        "## Approximate sample size for 80% power",
        "",
        "| Task | Discovery rho | Required n (two-sided alpha=0.01) |",
        "|---|---:|---:|",
    ]
    for item in power:
        lines.append(
            f"| {item['task_label']} | {item['discovery_rho']:+.3f} | "
            f"{item['required_n_80pct_bonferroni_five']} |"
        )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    configure_style()
    matrix = np.asarray([
        [float(item["rho"]) for item in results[method]]
        for method, _, _, _ in methods
    ])
    figure, axis = plt.subplots(figsize=(7.3, 3.4), constrained_layout=True)
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-0.55, vmax=0.55, aspect="auto")
    axis.set_xticks(range(5), [
        f"{item['label']}\n{short_name(item['coalition'])}"
        for item in validation["results"]
    ])
    axis.xaxis.tick_top()
    axis.set_yticks(range(len(methods)), [label for _, label, _, _ in methods])
    axis.tick_params(length=0)
    for row, (method, _, _, _) in enumerate(methods):
        for column, item in enumerate(results[method]):
            rho = float(item["rho"])
            marker = "*" if float(item["holm_p"]) < 0.05 else ""
            axis.text(
                column, row - 0.10, f"{rho:+.2f}{marker}",
                ha="center", va="center", fontsize=7, weight="bold",
                color="white" if abs(rho) > 0.34 else "#263238",
            )
            axis.text(
                column, row + 0.18, f"Holm {float(item['holm_p']):.3f}",
                ha="center", va="center", fontsize=5.1,
                color="white" if abs(rho) > 0.34 else "#263238",
            )
    axis.set_title(
        "Five task-matched synergy candidates across robustness analyses",
        loc="left", weight="bold", pad=28,
    )
    colorbar = figure.colorbar(image, ax=axis, orientation="horizontal",
                              fraction=0.08, pad=0.10)
    colorbar.set_label(r"Partial or unadjusted Spearman $\rho$")
    for extension in ("png", "svg", "pdf"):
        figure.savefig(
            OUTPUT_DIR / f"robustness_matrix.{extension}",
            dpi=300, bbox_inches="tight", facecolor="white",
        )
    plt.close(figure)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
