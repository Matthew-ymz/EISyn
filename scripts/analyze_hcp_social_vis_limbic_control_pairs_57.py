#!/usr/bin/env python3
"""Post-hoc pair decomposition of the SOCIAL Vis-Limbic-Control winner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.screen_hcp_social_composite_scores_57 import (
    BOOTSTRAPS,
    PERMUTATIONS,
    SEED,
    bootstrap_candidate,
    configure_style,
    load_scores,
    load_subjects,
    screen,
)


SOURCE = ROOT / "results/hcp_social_composite_scores_57/social_coalition_synergy_57.npz"
FULL_ROWS = ROOT / "results/hcp_social_composite_scores_57/all_associations.jsonl"
OUTPUT = ROOT / "results/hcp_social_vis_limbic_control_pairs_57"
PAIRS = ("Vis+Limbic", "Vis+Cont", "Limbic+Cont")
DISPLAY = ("Visual + Limbic", "Visual + Control", "Limbic + Control")
ENDPOINTS = ("balanced_accuracy", "dprime")


def load_pair_data() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, dict]]:
    with np.load(SOURCE, allow_pickle=False) as archive:
        names = archive["coalitions"].astype(str)
        matrix = archive["synergy_bits"].astype(float)
    indices = [names.tolist().index(pair) for pair in PAIRS]
    subjects = load_subjects()
    if matrix.shape != (57, 120):
        raise ValueError("Expected the frozen 57 x 120 SOCIAL synergy matrix.")
    full_rows = {}
    with FULL_ROWS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["coalition"] in PAIRS:
                full_rows[row["coalition"]] = row
    if set(full_rows) != set(PAIRS):
        raise ValueError("Full-screen association rows are incomplete.")
    return subjects, matrix[:, indices], load_scores(subjects), full_rows


def make_summary(matrix: np.ndarray, scores: dict[str, np.ndarray], full_rows: dict[str, dict]) -> dict:
    local = screen(matrix, scores, PERMUTATIONS, SEED + 101)
    rows = []
    for pair_index, pair in enumerate(PAIRS):
        row = {"coalition": pair, "display_name": DISPLAY[pair_index]}
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            full = full_rows[pair][endpoint]
            quantiles = bootstrap_candidate(matrix[:, pair_index], scores[endpoint], scores, BOOTSTRAPS, SEED + 200 + 10 * pair_index + endpoint_index)
            row[endpoint] = {
                "rho_adjusted": float(local["rho_adjusted"][pair_index, endpoint_index]),
                "rho_original_29": float(full["rho_original_29"]),
                "rho_supplement_28": float(full["rho_supplement_28"]),
                "p_pointwise": float(local["p_raw"][pair_index, endpoint_index]),
                "p_local_endpoint_max_t_three_pairs": float(local["p_endpoint_max_t_120"][pair_index, endpoint_index]),
                "p_local_global_max_t_six_tests": float(local["p_global_max_t_240"][pair_index, endpoint_index]),
                "p_full_endpoint_max_t_120": float(full["p_endpoint_max_t_120"]),
                "p_full_global_max_t_240": float(full["p_global_max_t_240"]),
                "bootstrap_ci95": [float(quantiles[0]), float(quantiles[2])],
                "bootstrap_median": float(quantiles[1]),
            }
        rows.append(row)
    return {
        "experiment": "Post-hoc pair decomposition of SOCIAL Visual-Limbic-Control synergy",
        "n": 57,
        "pairs": list(PAIRS),
        "endpoints": list(ENDPOINTS),
        "statistics": {
            "permutations": PERMUTATIONS,
            "bootstraps": BOOTSTRAPS,
            "covariates": ["age", "sex", "cohort"],
            "local_family": "three pairs x two endpoints",
            "selection_aware_family": "all 120 coalitions x two endpoints from the parent screen",
        },
        "results": rows,
        "interpretation_boundary": "The three pairs were chosen after the Visual-Limbic-Control winner was observed; full-screen max-T values are the selection-aware boundary.",
    }


def make_figure(matrix: np.ndarray, scores: dict[str, np.ndarray], summary: dict) -> None:
    configure_style()
    colors = np.asarray(["#7A8DA6", "#D07A55"])[scores["cohort"].astype(int)]
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.75), constrained_layout=True, sharey=True)
    rng = np.random.default_rng(SEED + 300)
    for row_index, endpoint in enumerate(ENDPOINTS):
        x = scores[endpoint]
        xlabel = "Balanced accuracy (%)" if endpoint == "balanced_accuracy" else r"Corrected sensitivity $d'$"
        for column_index, pair in enumerate(PAIRS):
            axis = axes[row_index, column_index]
            y = matrix[:, column_index]
            jitter = rng.uniform(-0.008, 0.008, len(x)) * max(float(np.ptp(x)), 1.0)
            axis.scatter(x + jitter, y, c=colors, s=18, alpha=0.84, edgecolor="white", linewidth=0.35)
            order = np.argsort(x)
            coefficient = np.polyfit(x, y, 1)
            axis.plot(x[order], np.polyval(coefficient, x[order]), color="#465563", ls="--", lw=0.9)
            value = summary["results"][column_index][endpoint]
            letter = chr(ord("a") + row_index * 3 + column_index)
            axis.set_title(f"{letter}  {DISPLAY[column_index]}", loc="left", fontweight="bold")
            axis.set_xlabel(xlabel)
            if column_index == 0:
                axis.set_ylabel("Pair synergy (bits)")
            axis.text(
                0.02, 0.98,
                rf"adjusted $\rho$={value['rho_adjusted']:+.3f}"
                f"\nlocal 6-test max-T $p$={value['p_local_global_max_t_six_tests']:.4f}"
                f"\nfull 240-test max-T $p$={value['p_full_global_max_t_240']:.4f}",
                transform=axis.transAxes, va="top", fontsize=6.2,
            )
    handles = [mpl.lines.Line2D([], [], marker="o", ls="none", color=color, label=label, markersize=4) for label, color in (("Original 29", "#7A8DA6"), ("Supplement 28", "#D07A55"))]
    figure.legend(handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5))
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        figure.savefig(OUTPUT / f"social_vis_limbic_control_pair_correlations_57.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(figure)


def write_report(summary: dict) -> None:
    lines = [
        "# SOCIAL Visual-Limbic-Control pair decomposition", "",
        "![Pair correlations](social_vis_limbic_control_pair_correlations_57.png)", "",
        "| Endpoint | Pair | Adjusted rho | Bootstrap 95% CI | Original 29 rho | Supplement 28 rho | Local 6-test max-T p | Full 240-test max-T p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for endpoint in ENDPOINTS:
        for row in summary["results"]:
            value = row[endpoint]
            lines.append(
                f"| {endpoint} | {row['display_name']} | {value['rho_adjusted']:+.3f} | "
                f"[{value['bootstrap_ci95'][0]:+.3f}, {value['bootstrap_ci95'][1]:+.3f}] | "
                f"{value['rho_original_29']:+.3f} | {value['rho_supplement_28']:+.3f} | "
                f"{value['p_local_global_max_t_six_tests']:.5f} | {value['p_full_global_max_t_240']:.5f} |"
            )
    lines += [
        "",
        "The local six-test family describes the requested post-hoc decomposition. Because the three pairs were chosen after the three-network winner was observed, the full 240-test max-T value is the appropriate selection-aware boundary.", "",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _, matrix, scores, full_rows = load_pair_data()
    summary = make_summary(matrix, scores, full_rows)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figure(matrix, scores, summary)
    write_report(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
