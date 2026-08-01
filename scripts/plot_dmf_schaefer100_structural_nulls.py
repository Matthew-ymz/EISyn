#!/usr/bin/env python3
"""Plot the Schaefer100 empirical-SC versus structural-null comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    ROOT / "results" / "dmf_schaefer100" / "structural_nulls" / "full" / "summary.json"
)
DEFAULT_OUTPUT = (
    ROOT / "fig" / "dmf_schaefer100" / "dmf_schaefer100_structural_nulls"
)

ORDER = ("empirical", "weight_shuffle", "degree_strength", "yeo_block")
LABELS = {
    "empirical": "Empirical SC",
    "weight_shuffle": "Weight shuffle",
    "degree_strength": "Degree + strength",
    "yeo_block": "Yeo-block",
}
COLORS = {
    "empirical": "#252525",
    "weight_shuffle": "#4C78A8",
    "degree_strength": "#E07A2D",
    "yeo_block": "#2A9D78",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(axis: plt.Axes, label: str, *, x: float = -0.14) -> None:
    axis.text(
        x,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )


def condition(summary: dict[str, object], name: str) -> dict[str, object]:
    if name == "empirical":
        return summary["empirical"]
    return summary["nulls"][name]


def seed_sd(values: object) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def save(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot(summary: dict[str, object], output: Path) -> None:
    configure()
    figure = plt.figure(figsize=(7.2, 5.25), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.12, 1.0), hspace=0.34, wspace=0.28)
    ax_a = figure.add_subplot(grid[0, 0])
    peak_grid = grid[0, 1].subgridspec(2, 1, hspace=0.46)
    ax_b1 = figure.add_subplot(peak_grid[0, 0])
    ax_b2 = figure.add_subplot(peak_grid[1, 0])
    ax_c = figure.add_subplot(grid[1, 0])
    ax_d = figure.add_subplot(grid[1, 1])

    for name in ORDER:
        item = condition(summary, name)
        g = np.asarray(item["G"], dtype=float)
        mean = np.asarray(item["phi_mean"], dtype=float)
        error = np.asarray(item["phi_sd"], dtype=float)
        ax_a.plot(g, mean, color=COLORS[name], lw=1.45, label=LABELS[name])
        ax_a.fill_between(g, mean - error, mean + error, color=COLORS[name], alpha=0.14, lw=0)
    ax_a.set_xlabel("Global coupling $G$")
    ax_a.set_ylabel(r"Full-system $\Xi$ (bits)")
    ax_a.grid(True, color="0.90", lw=0.5)
    ax_a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=2, columnspacing=1.2)
    panel_label(ax_a, "a")

    y = np.arange(len(ORDER))
    peak_phi = np.asarray(
        [condition(summary, name)["peak_phi_mean_bits"] for name in ORDER], dtype=float
    )
    peak_phi_sd = np.asarray(
        [seed_sd(condition(summary, name)["peak_phi_by_seed_bits"]) for name in ORDER]
    )
    peak_g = np.asarray([condition(summary, name)["peak_G_mean"] for name in ORDER])
    peak_g_sd = np.asarray(
        [seed_sd(condition(summary, name)["peak_G_by_seed"]) for name in ORDER]
    )
    for axis, values, errors, xlabel in (
        (ax_b1, peak_phi, peak_phi_sd, r"Peak $\Xi$ (bits)"),
        (ax_b2, peak_g, peak_g_sd, r"Peak $G$"),
    ):
        for index, name in enumerate(ORDER):
            axis.errorbar(
                values[index],
                y[index],
                xerr=errors[index],
                fmt="o",
                color=COLORS[name],
                ms=4.6,
                elinewidth=1.0,
                capsize=2.2,
                zorder=2,
            )
        axis.set_yticks(y, ["Empirical", "Weight shuffle", "Degree + strength", "Yeo-block"])
        axis.invert_yaxis()
        axis.set_xlabel(xlabel)
        axis.grid(True, axis="x", color="0.90", lw=0.5)
    panel_label(ax_b1, "b", x=-0.18)

    fractions = np.asarray(
        [
            [
                100.0 * float(condition(summary, name)["cross_roi_fraction_mean"]),
                100.0 * float(condition(summary, name)["between_network_fraction_mean"]),
            ]
            for name in ORDER
        ]
    )
    fraction_sd = np.asarray(
        [
            [
                100.0 * seed_sd(condition(summary, name)["cross_roi_fraction_by_seed"]),
                100.0 * seed_sd(
                    condition(summary, name)["between_network_fraction_by_seed"]
                ),
            ]
            for name in ORDER
        ]
    )
    offsets = np.linspace(-0.24, 0.24, len(ORDER))
    for index, name in enumerate(ORDER):
        positions = np.arange(2) + offsets[index]
        ax_c.errorbar(
            positions,
            fractions[index],
            yerr=fraction_sd[index],
            color=COLORS[name],
            marker="o",
            ms=4.3,
            lw=1.1,
            capsize=2,
            label=LABELS[name],
        )
    ax_c.set_xticks([0, 1], ["Cross ROI / total", "Between network / total"])
    ax_c.set_ylabel(r"Fraction of $\Xi$ (%)")
    ax_c.grid(True, axis="y", color="0.90", lw=0.5)
    ax_c.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=2, columnspacing=1.2)
    panel_label(ax_c, "c")

    audits = summary["structural_audits"]
    audit_rows = ("weight_shuffle", "degree_strength", "yeo_block")
    audit_columns = (
        "global_weight_multiset_equal",
        "nodewise_degree_equal",
        "strength_preserved",
        "yeo_block_weight_multisets_equal",
    )
    audit_values = np.zeros((len(audit_rows), len(audit_columns)), dtype=float)
    for row, name in enumerate(audit_rows):
        item = audits[name]
        audit_values[row] = (
            float(bool(item["global_weight_multiset_equal"])),
            float(bool(item["nodewise_degree_equal"])),
            float(float(item["maximum_relative_strength_error"]) < 1.0e-8),
            float(bool(item["yeo_block_weight_multisets_equal"])),
        )
    ax_d.imshow(audit_values, cmap=mpl.colors.ListedColormap(["#ECECEC", "#4C78A8"]), vmin=0, vmax=1, aspect="auto")
    ax_d.set_xticks(
        np.arange(4),
        ["Weight\nmultiset", "Nodewise\ndegree", "Node\nstrength", "Yeo-block\nmultisets"],
    )
    ax_d.set_yticks(np.arange(3), [LABELS[name] for name in audit_rows])
    for row in range(audit_values.shape[0]):
        for column in range(audit_values.shape[1]):
            ax_d.text(
                column,
                row,
                "preserved" if audit_values[row, column] else "destroyed",
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if audit_values[row, column] else "0.35",
            )
    for spine in ax_d.spines.values():
        spine.set_visible(False)
    ax_d.tick_params(length=0)
    panel_label(ax_d, "d")

    save(figure, output)


def main() -> None:
    args = parse_args()
    summary_path = args.summary if args.summary.is_absolute() else ROOT / args.summary
    output = args.output if args.output.is_absolute() else ROOT / args.output
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    plot(summary, output)


if __name__ == "__main__":
    main()
