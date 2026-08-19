#!/usr/bin/env python3
"""Create an editable schematic of the reproduced MGSTN and EI readout."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fig/nyc_taxi_mgstn_architecture"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7.0,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def box(ax, x, y, w, h, text, face, edge="#52616B", size=6.6, lw=0.8):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", facecolor=face,
                           edgecolor=edge, linewidth=lw)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.2)


def arrow(ax, start, end, color="#667680", dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, linewidth=0.8,
                                 color=color, linestyle="--" if dashed else "-"))


def main():
    fig, ax = plt.subplots(figsize=(7.25, 4.35))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.02, 0.965, "Predictive model", fontsize=8.5, fontweight="bold", color="#26343C", va="top")

    ys = [0.72, 0.50, 0.28]
    names = ["Recent\n7 consecutive hours", "Daily\n5 matched hours", "Weekly\n7 matched hours"]
    colors = ["#DCE8F1", "#E6E1F0", "#F2E5D5"]
    for y, name, color in zip(ys, names, colors):
        box(ax, 0.025, y, 0.135, 0.13, name, color, size=6.7)
        box(ax, 0.22, y, 0.455, 0.13,
            "STN branch\nspatial: residual GCN ×2 + typed RGCN ×2\n"
            "temporal: non-stationary Transformer ×2  ·  weather/date context",
            "#E8F0F0", size=5.9)
        box(ax, 0.735, y, 0.085, 0.13, "Branch\nstate", color, size=6.5)
        arrow(ax, (0.16, y + 0.065), (0.22, y + 0.065))
        arrow(ax, (0.675, y + 0.065), (0.735, y + 0.065))

    box(ax, 0.865, 0.42, 0.115, 0.30,
        "Multi-granularity\nfusion\n\n+ target attributes\n\n66 zones ×\n(inflow, outflow)", "#DCE7F0", size=6.7)
    for y in ys:
        arrow(ax, (0.82, y + 0.065), (0.865, 0.57))

    ax.plot([0.02, 0.98], [0.205, 0.205], color="#CBD1D5", lw=0.7)
    ax.text(0.02, 0.18, "Post-hoc information readout (not a training layer)", fontsize=7.4,
            fontweight="bold", color="#4F5A60", va="top")
    box(ax, 0.18, 0.035, 0.18, 0.105, "Fixed checkpoint\nlocal Jacobian J*", "#F4F4F4", edge="#808080")
    box(ax, 0.43, 0.035, 0.17, 0.105, "Validation residual\ncovariance Σe", "#F4F4F4", edge="#808080")
    box(ax, 0.68, 0.025, 0.285, 0.125,
        "Affine triangular TM\nEI → across-region Ξ\n+ within-region multi-scale Ξ", "#E8F0EC", edge="#4F8C85")
    arrow(ax, (0.36, 0.087), (0.68, 0.087), color="#4F8C85")
    arrow(ax, (0.60, 0.087), (0.68, 0.087), color="#4F8C85")
    ax.text(0.98, 0.222, "Hidden width 96 · 4 heads · 1,796,376 parameters", ha="right",
            fontsize=6.2, color="#666666")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(OUTPUT.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
