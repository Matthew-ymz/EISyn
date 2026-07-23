#!/usr/bin/env python3
"""Exploratory raw-p screen of HCP cognition against task-state Xi hierarchy atoms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARRAYS = ROOT / "results/hcp_schaefer500_task_evoked_xi_tuning/full/k1_p3_a1/arrays.npz"
DEFAULT_COGNITION = ROOT / "results/hcp_single_group_sem_full_1206/selected_29_sem_results.csv"
DEFAULT_OUTPUT = ROOT / "results/hcp_cognition_atom_exploration"
SCORES = ("g_score", "cry_score", "mem_score", "spd_score")
SCORE_LABELS = {
    "g_score": "General cognition",
    "cry_score": "Crystallized cognition",
    "mem_score": "Memory",
    "spd_score": "Processing speed",
}
NETWORK_SHORT = {
    "Vis": "Vis",
    "SomMot": "Som",
    "DorsAttn": "DAN",
    "SalVentAttn": "SVAN",
    "Limbic": "Lim",
    "Cont": "Cont",
    "Default": "Def",
}
PRIMARY_MIN_SUPPORT = 5
P_THRESHOLD = 0.05
ZERO_TOLERANCE = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument("--cognition", type=Path, default=DEFAULT_COGNITION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def short_atom(name: str) -> str:
    return "+".join(NETWORK_SHORT[item] for item in name.split("+"))


def load_inputs(arrays_path: Path, cognition_path: Path) -> tuple[np.lib.npyio.NpzFile, pd.DataFrame]:
    archive = np.load(arrays_path)
    subjects = np.asarray([str(value).removeprefix("sub-") for value in archive["subjects"]])
    cognition = pd.read_csv(cognition_path, dtype={"Subject": str})
    missing = {"Subject", *SCORES}.difference(cognition.columns)
    if missing:
        raise ValueError(f"Cognition table is missing columns: {sorted(missing)}")
    cognition["Subject"] = cognition["Subject"].str.removeprefix("sub-")
    if cognition["Subject"].duplicated().any():
        raise ValueError("Cognition table contains duplicate Subject values")
    cognition = cognition.set_index("Subject")
    if set(subjects) != set(cognition.index):
        raise ValueError("Cognition and Xi Subject sets do not match exactly")
    cognition = cognition.loc[subjects, list(SCORES)]
    if len(subjects) != 29 or not np.isfinite(cognition.to_numpy(dtype=float)).all():
        raise ValueError("Expected 29 aligned subjects with finite cognition scores")
    return archive, cognition


def compute_rows(archive: np.lib.npyio.NpzFile, cognition: pd.DataFrame) -> list[dict[str, object]]:
    atom_value = np.asarray(archive["atom_share"], dtype=float) * np.asarray(
        archive["cross_xi"], dtype=float
    )[:, :, None]
    states = archive["states"].astype(str)
    atom_names = archive["atom_names"].astype(str)
    rows: list[dict[str, object]] = []
    for score in SCORES:
        cognition_values = cognition[score].to_numpy(dtype=float)
        for state_index, state in enumerate(states):
            for atom_index, atom_name in enumerate(atom_names):
                values = atom_value[state_index, :, atom_index]
                support = int(np.count_nonzero(values > ZERO_TOLERANCE))
                if np.allclose(values, values[0]):
                    rho = p_value = float("nan")
                else:
                    result = spearmanr(cognition_values, values)
                    rho = float(result.statistic)
                    p_value = float(result.pvalue)
                rows.append(
                    {
                        "score": score,
                        "score_label": SCORE_LABELS[score],
                        "state": str(state),
                        "atom_index": int(atom_index),
                        "atom": str(atom_name),
                        "atom_short": short_atom(str(atom_name)),
                        "rho": rho,
                        "abs_rho": abs(rho) if np.isfinite(rho) else float("nan"),
                        "p_raw_two_sided": p_value,
                        "nonzero_subjects": support,
                        "zero_subjects": int(len(values) - support),
                        "mean_bits": float(values.mean()),
                        "max_bits": float(values.max()),
                    }
                )
    return rows


def ranked_candidates(rows: list[dict[str, object]], score: str, min_support: int) -> list[dict[str, object]]:
    candidates = [
        row
        for row in rows
        if row["score"] == score
        and int(row["nonzero_subjects"]) >= min_support
        and np.isfinite(float(row["p_raw_two_sided"]))
        and float(row["p_raw_two_sided"]) < P_THRESHOLD
    ]
    return sorted(
        candidates,
        key=lambda row: (-float(row["abs_rho"]), float(row["p_raw_two_sided"])),
    )


def save_figure(figure: plt.Figure, stem: Path) -> None:
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_ranked_candidates(primary: dict[str, list[dict[str, object]]], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), constrained_layout=True)
    for axis, score in zip(axes.ravel(), SCORES, strict=True):
        rows = primary[score]
        labels = [
            f"{row['state']} | {row['atom_short']} | n+={row['nonzero_subjects']}"
            for row in rows
        ]
        y = np.arange(len(rows))
        rho = np.asarray([float(row["rho"]) for row in rows])
        colors = np.where(rho >= 0, "#B65F3C", "#5B8FB9")
        axis.axvline(0.0, color="#777777", linewidth=0.75, linestyle="--")
        axis.scatter(rho, y, s=42, color=colors, edgecolor="white", linewidth=0.45, zorder=3)
        for index, row in enumerate(rows):
            x_offset = 0.018 if float(row["rho"]) >= 0 else -0.018
            axis.text(
                float(row["rho"]) + x_offset,
                index,
                f"p={float(row['p_raw_two_sided']):.3g}",
                ha="left" if x_offset > 0 else "right",
                va="center",
                fontsize=5.7,
            )
        axis.set(
            yticks=y,
            yticklabels=labels,
            xlim=(-0.68, 0.68),
            xlabel="Spearman rho",
            title=SCORE_LABELS[score],
        )
        axis.invert_yaxis()
        axis.tick_params(axis="y", length=0)
        axis.text(
            0.99,
            0.02,
            f"raw two-sided p<{P_THRESHOLD:g}; n+≥{PRIMARY_MIN_SUPPORT}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.8,
            color="#555555",
        )
    for label, axis in zip("abcd", axes.ravel(), strict=True):
        axis.text(-0.16, 1.06, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    save_figure(figure, output / "top_atom_associations")


def plot_top_scatter(
    primary: dict[str, list[dict[str, object]]],
    archive: np.lib.npyio.NpzFile,
    cognition: pd.DataFrame,
    output: Path,
) -> None:
    atom_value = np.asarray(archive["atom_share"], dtype=float) * np.asarray(
        archive["cross_xi"], dtype=float
    )[:, :, None]
    states = archive["states"].astype(str).tolist()
    figure, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), constrained_layout=True)
    for axis, score in zip(axes.ravel(), SCORES, strict=True):
        row = primary[score][0]
        state_index = states.index(str(row["state"]))
        atom_index = int(row["atom_index"])
        x = cognition[score].to_numpy(dtype=float)
        y = atom_value[state_index, :, atom_index]
        present = y > ZERO_TOLERANCE
        axis.scatter(
            x[~present],
            y[~present],
            s=20,
            color="#C4CAD1",
            alpha=0.8,
            linewidth=0,
            label="zero/not selected",
        )
        axis.scatter(
            x[present],
            y[present],
            s=28,
            color="#B65F3C",
            edgecolor="white",
            linewidth=0.45,
            label="positive atom",
        )
        axis.set(
            xlabel=SCORE_LABELS[score],
            ylabel="Hierarchy-atom contribution (bits)",
        )
        axis.text(
            0.02,
            0.98,
            f"{row['state']} | {row['atom_short']}\n"
            f"rho={float(row['rho']):.3f}, p={float(row['p_raw_two_sided']):.3g}, "
            f"n+={row['nonzero_subjects']}/29",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
        )
    axes[0, 0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    for label, axis in zip("abcd", axes.ravel(), strict=True):
        axis.text(-0.14, 1.06, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    save_figure(figure, output / "top_atom_scatter")


def plot_general_language_scatter(
    primary: dict[str, list[dict[str, object]]],
    archive: np.lib.npyio.NpzFile,
    cognition: pd.DataFrame,
    output: Path,
) -> None:
    """Plot the fully supported general-cognition LANGUAGE association."""
    row = next(
        candidate
        for candidate in primary["g_score"]
        if candidate["state"] == "LANGUAGE"
        and int(candidate["nonzero_subjects"]) == len(cognition)
    )
    states = archive["states"].astype(str).tolist()
    atom_value = np.asarray(archive["atom_share"], dtype=float) * np.asarray(
        archive["cross_xi"], dtype=float
    )[:, :, None]
    x = cognition["g_score"].to_numpy(dtype=float)
    y = atom_value[states.index("LANGUAGE"), :, int(row["atom_index"])]

    figure, axis = plt.subplots(figsize=(4.25, 3.35), constrained_layout=True)
    axis.scatter(
        x,
        y,
        s=34,
        color="#B65F3C",
        edgecolor="white",
        linewidth=0.55,
        alpha=0.92,
        zorder=3,
    )
    slope, intercept = np.polyfit(x, y, deg=1)
    x_line = np.linspace(float(x.min()), float(x.max()), 200)
    axis.plot(
        x_line,
        slope * x_line + intercept,
        color="#4B5563",
        linewidth=1.1,
        linestyle="--",
        zorder=2,
    )
    axis.text(
        0.03,
        0.97,
        f"Spearman rho = {float(row['rho']):.3f}\n"
        f"raw two-sided p = {float(row['p_raw_two_sided']):.5f}\n"
        f"n = {len(x)}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    axis.text(
        0.98,
        0.03,
        "Dashed line: linear visual guide",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#666666",
    )
    axis.set(
        xlabel="General cognition score",
        ylabel="LANGUAGE all-network atom contribution (bits)",
    )
    save_figure(figure, output / "general_language_all_network_scatter")


def markdown_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Rank | State | Network combination | rho | Raw p | Nonzero subjects |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['state']} | {row['atom']} | {float(row['rho']):+.3f} | "
            f"{float(row['p_raw_two_sided']):.4g} | {row['nonzero_subjects']}/29 |"
        )
    return lines


def write_report(
    primary: dict[str, list[dict[str, object]]],
    fragile: dict[str, list[dict[str, object]]],
    output: Path,
) -> None:
    lines = [
        "# HCP cognition–hierarchy atom exploratory screen",
        "",
        "This exploratory screen evaluates all 120 greedy hierarchy combinations in REST and seven task states against four cognition factors. Candidates are filtered by raw two-sided Spearman p<0.05 and ranked by descending absolute rho. The primary tables require the atom contribution to be nonzero in at least 5 of 29 subjects; associations supported by only 2–4 nonzero subjects are retained in summary.json as fragile candidates.",
        "",
        "A zero contribution means that the combination did not receive a positive contribution on the subject's current greedy path; it is not evidence that the corresponding networks were biologically inactive.",
    ]
    for score in SCORES:
        lines.extend(["", f"## {SCORE_LABELS[score]}", ""])
        lines.extend(markdown_table(primary[score]))
        lines.extend(
            [
                "",
                f"Fragile raw-p candidates with only 2–4 nonzero subjects: {len(fragile[score])}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Raw p values are used only to prioritize hypotheses in this exploratory phase. They are not adjusted for the number of searched task–atom combinations, demographics, motion, signal quality, or family structure. Sparse candidates can be driven by a few subjects and should be interpreted from the scatter plots before receiving a mechanistic explanation.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive, cognition = load_inputs(args.arrays, args.cognition)
    rows = compute_rows(archive, cognition)
    primary = {score: ranked_candidates(rows, score, PRIMARY_MIN_SUPPORT) for score in SCORES}
    fragile = {
        score: [
            row
            for row in ranked_candidates(rows, score, 2)
            if int(row["nonzero_subjects"]) < PRIMARY_MIN_SUPPORT
        ]
        for score in SCORES
    }
    if any(not primary[score] for score in SCORES):
        raise RuntimeError("At least one cognition factor has no primary raw-p candidate")

    plot_ranked_candidates(primary, args.output_dir)
    plot_top_scatter(primary, archive, cognition, args.output_dir)
    plot_general_language_scatter(primary, archive, cognition, args.output_dir)
    write_report(primary, fragile, args.output_dir)
    with (args.output_dir / "all_associations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "analysis": "Exploratory raw-p task-state hierarchy-atom screen",
        "subject_count": 29,
        "cognition_scores": list(SCORES),
        "candidate_space": {
            "states": archive["states"].astype(str).tolist(),
            "atom_count": int(len(archive["atom_names"])),
            "tests_per_score": int(len(archive["states"]) * len(archive["atom_names"])),
            "ranking": "raw two-sided Spearman p < 0.05, then descending absolute rho",
            "primary_min_nonzero_subjects": PRIMARY_MIN_SUPPORT,
            "multiple_comparison_correction_used_for_selection": False,
        },
        "primary_candidates": primary,
        "fragile_candidates_nonzero_2_to_4": fragile,
        "outputs": {
            "ranked_figure": "top_atom_associations.png",
            "top_scatter": "top_atom_scatter.png",
            "general_language_scatter": "general_language_all_network_scatter.png",
            "report": "report.md",
            "all_rows": "all_associations.jsonl",
        },
        "interpretation_boundary": (
            "Exploratory raw-p ranking only; sparse greedy atoms, unmodeled covariates, and the full search "
            "space must be considered before confirmation or mechanistic claims."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    run(args)
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
