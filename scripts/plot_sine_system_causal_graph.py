#!/usr/bin/env python3
"""Render the causal diagram for the simplified four-variable sine system."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "fig" / "granger_peid_mlp_comparison" / "causal_graph3.png"

BLUE = "#0826a8"
RED = "#d6281f"
BLACK = "#111111"


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.8,
    mutation_scale: float = 18.0,
    connectionstyle: str = "arc3",
    shrink_a: float = 0.0,
    shrink_b: float = 0.0,
    zorder: int = 2,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        mutation_scale=mutation_scale,
        connectionstyle=connectionstyle,
        shrinkA=shrink_a,
        shrinkB=shrink_b,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _line(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linewidth: float = 1.6,
    zorder: int = 1,
) -> None:
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=color,
        linewidth=linewidth,
        solid_capstyle="round",
        zorder=zorder,
    )


def render(output: Path) -> list[Path]:
    """Draw the diagram and save PNG, SVG, and PDF variants."""
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.72, 10.0), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    positions = {
        "w": (0.17, 0.50),
        "x": (0.49, 0.78),
        "y": (0.49, 0.22),
        "z": (0.82, 0.50),
    }
    # Shared-driver paths.  Unit coefficients are written as beta, not 1 beta.
    _arrow(
        ax,
        positions["w"],
        positions["x"],
        color=BLUE,
        linestyle=(0, (4, 4)),
        linewidth=2.0,
        shrink_a=30,
        shrink_b=32,
    )
    _arrow(
        ax,
        positions["w"],
        positions["y"],
        color=BLUE,
        linestyle=(0, (4, 4)),
        linewidth=2.0,
        shrink_a=30,
        shrink_b=32,
    )
    _arrow(
        ax,
        (0.21, 0.51),
        positions["z"],
        color=BLUE,
        linestyle=(0, (4, 4)),
        linewidth=2.0,
        connectionstyle="arc3,rad=-0.42",
        shrink_a=24,
        shrink_b=32,
    )

    ax.text(0.31, 0.67, r"$\beta$", color=BLUE, fontsize=27, ha="center", va="center")
    ax.text(0.31, 0.33, r"$\beta$", color=BLUE, fontsize=27, ha="center", va="center")
    ax.text(0.70, 0.72, r"$0.5\beta$", color=BLUE, fontsize=27, ha="center", va="center")

    # Nonlinear causal hyperedge {x, y} -> z.
    hub = (0.61, 0.50)
    hub_width, hub_height = 0.13, 0.27
    _line(ax, (0.525, 0.735), (0.573, 0.625), color=RED, linewidth=1.8)
    _line(ax, (0.525, 0.265), (0.573, 0.375), color=RED, linewidth=1.8)
    hyperedge = Ellipse(
        hub,
        width=hub_width,
        height=hub_height,
        facecolor="#fffaf7",
        edgecolor=RED,
        linewidth=1.8,
        linestyle=(0, (5, 4)),
        zorder=2,
    )
    ax.add_patch(hyperedge)
    _arrow(
        ax,
        (hub[0] + hub_width / 2, hub[1]),
        positions["z"],
        color=RED,
        linewidth=1.8,
        mutation_scale=18,
        shrink_b=32,
        zorder=3,
    )
    ax.text(hub[0], hub[1], r"$\sin(xy)$", color=RED, fontsize=23, ha="center", va="center")

    # State variables.
    for name, position in positions.items():
        edge_color = BLUE if name == "w" else BLACK
        text_color = BLUE if name == "w" else BLACK
        ax.scatter(
            [position[0]],
            [position[1]],
            s=3600,
            marker="o",
            facecolor="white",
            edgecolor=edge_color,
            linewidth=1.9,
            zorder=5,
        )
        ax.text(
            *position,
            rf"${name}$",
            color=text_color,
            fontsize=35,
            ha="center",
            va="center",
            zorder=6,
        )

    # Independent innovations.
    noise_specs = {
        "w": ((0.035, 0.61), (0.125, 0.54), (0.030, 0.63)),
        "x": ((0.34, 0.89), (0.445, 0.82), (0.33, 0.92)),
        "y": ((0.34, 0.11), (0.445, 0.18), (0.33, 0.08)),
        "z": ((0.96, 0.61), (0.875, 0.54), (0.97, 0.63)),
    }
    for name, (start, end, label_pos) in noise_specs.items():
        _arrow(
            ax,
            start,
            end,
            color=BLACK,
            linewidth=1.7,
            mutation_scale=15,
            shrink_b=5,
            zorder=4,
        )
        ax.text(
            *label_pos,
            rf"$\eta^{{{name}}}$",
            color=BLACK,
            fontsize=29,
            ha="center",
            va="center",
        )

    legend_handles = [
        Line2D([0], [0], color=BLUE, lw=2.0, linestyle=(0, (4, 4)), label="Confounder"),
        Line2D([0], [0], color=RED, lw=1.8, label="Causal hyperedge"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.04),
        frameon=False,
        fontsize=20,
        handlelength=3.0,
        borderaxespad=0.0,
    )

    outputs: list[Path] = []
    for suffix in (".png", ".svg", ".pdf"):
        path = output.with_suffix(suffix)
        save_kwargs: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == ".png":
            save_kwargs["dpi"] = 180
        fig.savefig(path, **save_kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in render(args.output):
        print(path)


if __name__ == "__main__":
    main()
