"""Plot the fixed-RMS pairwise-asymmetry MLP-EI sweep."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT / "results" / "pairwise_asymmetry_kuramoto_mlp" / "summary.json"
)
OUTPUT = (
    ROOT
    / "fig"
    / "part1_synergy_comparison"
    / "pairwise_asymmetry_kuramoto_mlp"
)


def build_figure(payload: dict[str, object]) -> plt.Figure:
    """Show composition and paired contrast without redundant titles."""
    pair_color = "#4477AA"
    triple_color = "#D9922E"
    rho_levels = (1.0, 0.5, 0.25)
    rows = payload["rows"]
    summary = payload["summary"]["by_rho"]

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 7.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig, (composition_ax, contrast_ax) = plt.subplots(
        1,
        2,
        figsize=(150 / 25.4, 65 / 25.4),
        gridspec_kw={"width_ratios": (1.35, 1.0)},
        constrained_layout=True,
    )

    pair_fraction = np.asarray(
        [
            float(summary[f"{rho:g}"]["pair_atom_fraction_123"]["mean"])
            for rho in rho_levels
        ]
    )
    triple_fraction = 1.0 - pair_fraction
    triadic_triple = float(
        summary["1"]["triple_residual_fraction_456"]["mean"]
    )
    fractions_pair = np.r_[pair_fraction, 1.0 - triadic_triple]
    fractions_triple = np.r_[triple_fraction, triadic_triple]
    y = np.arange(4)[::-1]
    composition_ax.barh(
        y,
        fractions_pair,
        height=0.58,
        color=pair_color,
        edgecolor="white",
        lw=0.8,
    )
    composition_ax.barh(
        y,
        fractions_triple,
        left=fractions_pair,
        height=0.58,
        color=triple_color,
        edgecolor="white",
        lw=0.8,
    )
    for y_value, pair_value, triple_value in zip(
        y,
        fractions_pair,
        fractions_triple,
    ):
        if pair_value >= 0.10:
            composition_ax.text(
                pair_value / 2.0,
                y_value,
                f"{100.0 * pair_value:.1f}",
                color="white",
                fontweight="bold",
                ha="center",
                va="center",
            )
        else:
            composition_ax.text(
                pair_value + 0.012,
                y_value + 0.37,
                f"{100.0 * pair_value:.1f}",
                color=pair_color,
                fontweight="bold",
                ha="left",
                va="center",
            )
        composition_ax.text(
            pair_value + triple_value / 2.0,
            y_value,
            f"{100.0 * triple_value:.1f}",
            color="white",
            fontweight="bold",
            ha="center",
            va="center",
        )
    composition_ax.text(
        0.23,
        3.55,
        r"$\mathcal{A}_2$",
        color=pair_color,
        fontweight="bold",
        ha="center",
    )
    composition_ax.text(
        0.76,
        3.55,
        r"$\mathcal{A}_3$",
        color=triple_color,
        fontweight="bold",
        ha="center",
    )
    composition_ax.set(
        xlim=(0.0, 1.0),
        ylim=(-0.55, 3.75),
        yticks=y,
        yticklabels=(
            r"$\rho=1$",
            r"$\rho=0.5$",
            r"$\rho=0.25$",
            r"$\{4,5,6\}$",
        ),
        xticks=(0.0, 0.25, 0.5, 0.75, 1.0),
        xticklabels=("0", "25", "50", "75", "100%"),
    )
    composition_ax.spines["left"].set_visible(False)
    composition_ax.grid(axis="x", color="0.92", lw=0.5, zorder=0)

    x = np.arange(len(rho_levels))
    for seed in sorted({int(row["seed"]) for row in rows}):
        values = [
            100.0
            * float(
                next(
                    row["triple_fraction_contrast_456_minus_123"]
                    for row in rows
                    if int(row["seed"]) == seed
                    and np.isclose(float(row["rho"]), rho)
                )
            )
            for rho in rho_levels
        ]
        contrast_ax.plot(
            x,
            values,
            color="0.70",
            lw=0.8,
            marker="o",
            ms=2.7,
            zorder=1,
        )
    contrast_mean = np.asarray(
        [
            100.0
            * float(
                summary[f"{rho:g}"][
                    "triple_fraction_contrast_456_minus_123"
                ]["mean"]
            )
            for rho in rho_levels
        ]
    )
    contrast_sem = np.asarray(
        [
            100.0
            * float(
                summary[f"{rho:g}"][
                    "triple_fraction_contrast_456_minus_123"
                ]["sem"]
            )
            for rho in rho_levels
        ]
    )
    contrast_ax.errorbar(
        x,
        contrast_mean,
        yerr=contrast_sem,
        color=triple_color,
        lw=1.8,
        marker="o",
        ms=4.5,
        capsize=2.2,
        zorder=3,
    )
    for x_value, mean_value, sem_value in zip(
        x,
        contrast_mean,
        contrast_sem,
    ):
        contrast_ax.text(
            x_value,
            mean_value + 2.8,
            rf"${mean_value:.1f}\pm{sem_value:.1f}$",
            color=triple_color,
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
        )
    contrast_ax.set(
        xlim=(-0.25, 2.25),
        ylim=(20.0, 60.0),
        xticks=x,
        xticklabels=(r"$1$", r"$0.5$", r"$0.25$"),
        xlabel=r"$\rho$",
        ylabel=r"$100\,\Delta\mathcal{A}_3$",
    )
    contrast_ax.grid(axis="y", color="0.92", lw=0.5, zorder=0)
    contrast_ax.text(
        0.98,
        0.04,
        r"$n=3$",
        transform=contrast_ax.transAxes,
        ha="right",
        va="bottom",
        color="0.35",
        fontsize=6.2,
    )
    fig.text(0.004, 0.985, "a", fontsize=9, fontweight="bold", va="top")
    fig.text(0.565, 0.985, "b", fontsize=9, fontweight="bold", va="top")
    return fig


def main() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(payload)
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
