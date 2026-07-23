#!/usr/bin/env python3
"""Visualize frozen SEM factor scores for the 29 HCP imaging subjects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTED = ROOT / "results/hcp_single_group_sem_full_1206/selected_29_sem_results.csv"
DEFAULT_REFERENCE = ROOT / "results/hcp_single_group_sem_full_1206/factor_scores_all_subjects.csv"
DEFAULT_XI_RECORDS = ROOT / "results/hcp_schaefer500_task_evoked_xi_tuning/full/records.jsonl"
DEFAULT_OUTPUT = ROOT / "results/hcp_cognition_score_overview"
XI_CONFIG_ID = "k1_p3_a1"

SCORES = ("g_score", "cry_score", "mem_score", "spd_score")
STATES = ("REST", "EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")
LABELS = {
    "g_score": "General cognition",
    "cry_score": "Crystallized cognition",
    "mem_score": "Memory",
    "spd_score": "Processing speed",
}
COLORS = {
    "g_score": "#3B6FB6",
    "cry_score": "#D28A45",
    "mem_score": "#3D9270",
    "spd_score": "#8B6BAE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--xi-records", type=Path, default=DEFAULT_XI_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_scores(selected_path: Path, reference_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(selected_path, dtype={"Subject": str})
    reference = pd.read_csv(reference_path, dtype={"Subject": str})
    required = {"Subject", *SCORES}
    for name, frame in (("selected", selected), ("reference", reference)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} data are missing columns: {sorted(missing)}")
        if not np.isfinite(frame[list(SCORES)].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} data contain non-finite factor scores")
    if len(selected) != 29 or selected["Subject"].nunique() != 29:
        raise ValueError("Expected exactly 29 unique selected subjects")
    return selected, reference


def strip_subject(value: object) -> str:
    return str(value).removeprefix("sub-")


def load_system_xi(records_path: Path, subjects: list[str]) -> pd.DataFrame:
    rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    subject_set = set(subjects)
    record_map: dict[tuple[str, str], float] = {}
    for row in rows:
        if str(row.get("config_id")) != XI_CONFIG_ID:
            continue
        state = str(row["state"])
        subject = strip_subject(row["subject"])
        if subject in subject_set:
            key = (subject, state)
            if key in record_map:
                raise ValueError(f"Duplicate system-level Xi record for {key} under {XI_CONFIG_ID}")
            record_map[key] = float(row["system_xi"])

    missing = [(subject, state) for subject in subjects for state in STATES if (subject, state) not in record_map]
    if missing:
        raise ValueError(f"Missing system-level Xi records: {missing[:5]}")
    system_xi = pd.DataFrame(
        [[record_map[(subject, state)] for state in STATES] for subject in subjects],
        index=subjects,
        columns=STATES,
        dtype=float,
    )
    if not np.isfinite(system_xi.to_numpy()).all():
        raise ValueError("System-level Xi data contain non-finite values")
    return system_xi


def build_summary(
    selected: pd.DataFrame, reference: pd.DataFrame, system_xi: pd.DataFrame
) -> dict[str, object]:
    summary: dict[str, object] = {
        "analysis": "Distribution of frozen SEM factor scores for 29 HCP imaging subjects",
        "selected_n": int(len(selected)),
        "reference_n": int(len(reference)),
        "scores": {},
        "pearson_correlations_selected_29": selected[list(SCORES)].corr().to_dict(),
    }
    score_summary = summary["scores"]
    assert isinstance(score_summary, dict)
    for score in SCORES:
        ks = ks_2samp(selected[score], reference[score])
        score_summary[score] = {
            "selected_mean": float(selected[score].mean()),
            "selected_sd": float(selected[score].std(ddof=1)),
            "selected_median": float(selected[score].median()),
            "selected_min": float(selected[score].min()),
            "selected_max": float(selected[score].max()),
            "selected_skew": float(selected[score].skew()),
            "reference_mean": float(reference[score].mean()),
            "reference_sd": float(reference[score].std(ddof=1)),
            "ks_statistic_descriptive": float(ks.statistic),
            "ks_p_descriptive": float(ks.pvalue),
        }
    g_score = selected.set_index(selected["Subject"].map(strip_subject)).loc[system_xi.index, "g_score"]
    state_z = (system_xi - system_xi.mean(axis=0)) / system_xi.std(axis=0, ddof=1)
    mean_raw = system_xi.mean(axis=1)
    mean_state_z = state_z.mean(axis=1)
    raw_spearman = spearmanr(g_score, mean_raw)
    z_spearman = spearmanr(g_score, mean_state_z)
    z_pearson = pearsonr(g_score, mean_state_z)
    summary["system_xi_alignment"] = {
        "model_config_id": XI_CONFIG_ID,
        "states": list(STATES),
        "across_state_mean_raw_bits_spearman_rho": float(raw_spearman.statistic),
        "across_state_mean_raw_bits_spearman_p_descriptive": float(raw_spearman.pvalue),
        "mean_within_state_z_spearman_rho": float(z_spearman.statistic),
        "mean_within_state_z_spearman_p_descriptive": float(z_spearman.pvalue),
        "mean_within_state_z_pearson_r": float(z_pearson.statistic),
        "mean_within_state_z_pearson_p_descriptive": float(z_pearson.pvalue),
        "state_spearman": {
            state: {
                "rho": float(spearmanr(g_score, system_xi[state]).statistic),
                "p_descriptive": float(spearmanr(g_score, system_xi[state]).pvalue),
            }
            for state in STATES
        },
    }
    return summary


def plot_scores(selected: pd.DataFrame, output_dir: Path) -> None:
    ordered = selected.sort_values("g_score", ascending=True).reset_index(drop=True)
    within_z = (ordered[list(SCORES)] - ordered[list(SCORES)].mean()) / ordered[list(SCORES)].std(ddof=1)
    # Matplotlib places the first y value at the bottom in panel a, whereas
    # seaborn places the first heatmap row at the top. Reverse the heatmap rows
    # so both panels read from highest cognition at the top to lowest at the bottom.
    profile_top_to_bottom = within_z.iloc[::-1]
    panel_a_top_to_bottom = ordered.iloc[::-1]["Subject"].tolist()
    panel_b_top_to_bottom = ordered.loc[profile_top_to_bottom.index, "Subject"].tolist()
    if panel_a_top_to_bottom != panel_b_top_to_bottom:
        raise RuntimeError("Panel a and panel b subject orders are not aligned")

    fig = plt.figure(figsize=(7.2, 5.45), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.08, 1.0))

    ax_rank = fig.add_subplot(grid[0])
    y = np.arange(len(ordered))
    values = ordered["g_score"].to_numpy()
    ax_rank.hlines(y, 0, values, color="#C7CED8", linewidth=0.8, zorder=1)
    ax_rank.scatter(values, y, s=18, color=COLORS["g_score"], edgecolor="white", linewidth=0.35, zorder=2)
    ax_rank.axvline(0, color="#666666", linewidth=0.7, linestyle="--")
    ax_rank.set_yticks(y, ordered["Subject"])
    ax_rank.set_xlabel("General cognition factor score")
    ax_rank.set_ylabel("Subject (ordered by general cognition)")
    ax_rank.set_ylim(-0.8, len(ordered) - 0.2)
    ax_rank.text(-0.12, 1.02, "a", transform=ax_rank.transAxes, fontweight="bold", fontsize=8)

    ax_profile = fig.add_subplot(grid[1])
    sns.heatmap(
        profile_top_to_bottom,
        ax=ax_profile,
        cmap=sns.diverging_palette(240, 25, as_cmap=True),
        center=0,
        vmin=-2.5,
        vmax=2.5,
        cbar_kws={"label": "Within-29 score z", "shrink": 0.72},
        yticklabels=False,
        xticklabels=["General", "Crystallized", "Memory", "Speed"],
        linewidths=0.15,
        linecolor="white",
    )
    ax_profile.set_ylabel("Same top-to-bottom subject order as panel a")
    ax_profile.tick_params(axis="x", rotation=35)
    ax_profile.text(-0.12, 1.02, "b", transform=ax_profile.transAxes, fontweight="bold", fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "selected_29_cognition_scores"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_system_xi_alignment(
    selected: pd.DataFrame, system_xi: pd.DataFrame, output_dir: Path
) -> None:
    selected_indexed = selected.assign(Subject=selected["Subject"].map(strip_subject)).set_index("Subject")
    ordered_subjects = selected_indexed.sort_values("g_score", ascending=True).index.tolist()
    ordered_scores = selected_indexed.loc[ordered_subjects, "g_score"].to_numpy(dtype=float)
    ordered_xi = system_xi.loc[ordered_subjects]
    state_values = ordered_xi.to_numpy(dtype=float)
    mean_xi = state_values.mean(axis=1)
    min_xi = state_values.min(axis=1)
    max_xi = state_values.max(axis=1)
    rho, p_value = spearmanr(ordered_scores, mean_xi)

    fig, ax = plt.subplots(figsize=(6.9, 5.6), constrained_layout=True)
    y = np.arange(len(ordered_subjects))
    ax.hlines(y, min_xi, max_xi, color="#C9CED6", linewidth=0.9, zorder=1)
    state_offsets = np.linspace(-0.16, 0.16, len(STATES))
    for state_idx, offset in enumerate(state_offsets):
        ax.scatter(
            state_values[:, state_idx],
            y + offset,
            s=7,
            color="#8B95A3",
            alpha=0.55,
            linewidth=0,
            zorder=2,
        )
    ax.scatter(
        mean_xi,
        y,
        s=24,
        marker="D",
        color="#D28A45",
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    ax.axvline(float(mean_xi.mean()), color="#666666", linewidth=0.7, linestyle="--")
    ax.set_yticks(y, ordered_subjects)
    ax.set_ylim(-0.8, len(ordered_subjects) - 0.2)
    ax.set_xlabel(r"System-level $\Xi$ (bits)")
    ax.set_ylabel("Subject (ordered by general cognition)")
    ax.text(
        0.0,
        1.045,
        rf"General cognition vs across-state mean: Spearman $\rho$={rho:.2f}, descriptive $p$={p_value:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
    )
    ax.text(
        1.0,
        1.015,
        "Gray: state values/range   Orange diamond: across-state mean",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#555555",
    )

    stem = output_dir / "selected_29_system_xi_by_cognition_rank"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()
    selected, reference = load_scores(args.selected, args.reference)
    selected_subjects = selected["Subject"].map(strip_subject).tolist()
    system_xi = load_system_xi(args.xi_records, selected_subjects)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ordered = selected.sort_values("g_score", ascending=False).reset_index(drop=True).copy()
    ordered.insert(1, "g_rank_descending", np.arange(1, len(ordered) + 1))
    ordered[["Subject", "g_rank_descending", *SCORES]].to_csv(
        args.output_dir / "selected_29_cognition_scores.tsv", sep="\t", index=False
    )

    xi_table = selected.assign(Subject=selected["Subject"].map(strip_subject)).set_index("Subject")[["g_score"]]
    xi_table = xi_table.join(system_xi)
    xi_table["system_xi_mean_bits"] = system_xi.mean(axis=1)
    xi_table["system_xi_mean_within_state_z"] = (
        (system_xi - system_xi.mean(axis=0)) / system_xi.std(axis=0, ddof=1)
    ).mean(axis=1)
    xi_table.sort_values("g_score", ascending=False).to_csv(
        args.output_dir / "selected_29_system_xi_by_cognition_rank.tsv", sep="\t"
    )

    summary = build_summary(selected, reference, system_xi)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_scores(selected, args.output_dir)
    plot_system_xi_alignment(selected, system_xi, args.output_dir)
    print(f"Wrote cognition score overview to {args.output_dir}")


if __name__ == "__main__":
    main()
