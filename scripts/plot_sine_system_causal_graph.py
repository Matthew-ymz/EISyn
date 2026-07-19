#!/usr/bin/env python3
"""Render a publication-quality causal diagram for the four-variable sine system."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import Ellipse, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "fig"
    / "granger_peid_mlp_comparison"
    / "causal_graph_original_neighborhood_one_decimal.png"
)

# Restrained, colorblind-friendly accents on a neutral academic canvas.
BLUE = "#2F5AA8"
RED = "#C7443E"
INK = "#20262E"
MUTED = "#4B5563"
BLUE_FILL = "#F4F7FC"
RED_FILL = "#FFF7F3"


def _connector(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    patch_a: Ellipse | None = None,
    patch_b: Ellipse | None = None,
    arrowstyle: str = "-|>",
    linestyle: str | tuple[int, tuple[int, ...]] = "-",
    linewidth: float = 1.65,
    connectionstyle: str = "arc3,rad=0",
    mutation_scale: float = 14.0,
    shrink_a: float = 2.0,
    shrink_b: float = 2.0,
    zorder: int = 2,
) -> FancyArrowPatch:
    """Add a smooth connector clipped exactly to its endpoint patches."""
    connector = FancyArrowPatch(
        start,
        end,
        patchA=patch_a,
        patchB=patch_b,
        arrowstyle=arrowstyle,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        mutation_scale=mutation_scale,
        connectionstyle=connectionstyle,
        shrinkA=shrink_a,
        shrinkB=shrink_b,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(connector)
    return connector


def _node(
    ax: plt.Axes,
    center: tuple[float, float],
    label: str,
    *,
    edge_color: str = INK,
    text_color: str = INK,
    face_color: str = "white",
) -> Ellipse:
    """Draw a circular state node in display coordinates."""
    patch = Ellipse(
        center,
        width=0.066,
        height=0.108,
        facecolor=face_color,
        edgecolor=edge_color,
        linewidth=1.65,
        zorder=4,
    )
    ax.add_patch(patch)
    ax.text(
        *center,
        rf"${label}$",
        color=text_color,
        fontsize=25,
        ha="center",
        va="center",
        zorder=5,
    )
    return patch


def render(
    output: Path,
    *,
    source_loading: float = 0.8,
    target_loading: float = 0.1,
) -> list[Path]:
    """Draw the diagram and save PNG, SVG, and PDF variants."""
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    style = {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.0,
        "lines.dash_capstyle": "round",
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    with mpl.rc_context(style):
        fig, ax = plt.subplots(figsize=(12.0, 7.4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")
        fig.subplots_adjust(left=0.025, right=0.985, bottom=0.035, top=0.98)

        positions = {
            "w": (0.15, 0.50),
            "x": (0.43, 0.74),
            "y": (0.43, 0.26),
            "z": (0.84, 0.50),
        }
        nodes = {
            name: _node(
                ax,
                center,
                name,
                edge_color=BLUE if name == "w" else INK,
                text_color=BLUE if name == "w" else INK,
                face_color=BLUE_FILL if name == "w" else "white",
            )
            for name, center in positions.items()
        }

        confounder_style = (0, (4.5, 4.0))
        _connector(
            ax,
            positions["w"],
            positions["x"],
            color=BLUE,
            patch_a=nodes["w"],
            patch_b=nodes["x"],
            linestyle=confounder_style,
            connectionstyle="arc3,rad=-0.035",
            linewidth=1.8,
            zorder=1,
        )
        _connector(
            ax,
            positions["w"],
            positions["y"],
            color=BLUE,
            patch_a=nodes["w"],
            patch_b=nodes["y"],
            linestyle=confounder_style,
            connectionstyle="arc3,rad=0.035",
            linewidth=1.8,
            zorder=1,
        )
        direct_path = MplPath(
            [(0.179, 0.530), (0.32, 0.58), (0.66, 0.68), (0.807, 0.545)],
            [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
        )
        direct_arrow = FancyArrowPatch(
            path=direct_path,
            arrowstyle="-|>",
            color=BLUE,
            linestyle=confounder_style,
            linewidth=1.8,
            mutation_scale=14.0,
            capstyle="round",
            joinstyle="round",
            zorder=1,
        )
        ax.add_patch(direct_arrow)

        source_label = rf"${source_loading:.1f}\beta$"
        target_label = rf"${target_loading:.1f}\beta$"
        ax.text(0.285, 0.665, source_label, color=BLUE, fontsize=19, ha="center", va="center")
        ax.text(0.285, 0.335, source_label, color=BLUE, fontsize=19, ha="center", va="center")
        ax.text(0.675, 0.715, target_label, color=BLUE, fontsize=19, ha="center", va="center")

        # Nonlinear causal hyperedge {x, y} -> z.
        hub_center = (0.635, 0.50)
        hub = Ellipse(
            hub_center,
            width=0.115,
            height=0.230,
            facecolor=RED_FILL,
            edgecolor=RED,
            linewidth=1.65,
            linestyle=(0, (5.0, 4.0)),
            zorder=3,
        )
        ax.add_patch(hub)
        _connector(
            ax,
            positions["x"],
            hub_center,
            color=RED,
            patch_a=nodes["x"],
            patch_b=hub,
            arrowstyle="-",
            connectionstyle="arc3,rad=-0.055",
            linewidth=1.7,
            shrink_a=0.5,
            shrink_b=0.0,
            zorder=2,
        )
        _connector(
            ax,
            positions["y"],
            hub_center,
            color=RED,
            patch_a=nodes["y"],
            patch_b=hub,
            arrowstyle="-",
            connectionstyle="arc3,rad=0.055",
            linewidth=1.7,
            shrink_a=0.5,
            shrink_b=0.0,
            zorder=2,
        )
        _connector(
            ax,
            hub_center,
            positions["z"],
            color=RED,
            patch_a=hub,
            patch_b=nodes["z"],
            linewidth=1.8,
            mutation_scale=14.5,
            zorder=2,
        )
        ax.text(
            *hub_center,
            r"$\sin(xy)$",
            color=RED,
            fontsize=19,
            ha="center",
            va="center",
            zorder=5,
        )

        # Independent innovations use a consistent radial geometry and muted ink.
        noise_specs = {
            "w": ((0.038, 0.615), (0.032, 0.645)),
            "x": ((0.292, 0.895), (0.285, 0.930)),
            "y": ((0.292, 0.105), (0.285, 0.070)),
            "z": ((0.955, 0.615), (0.967, 0.645)),
        }
        for name, (start, label_position) in noise_specs.items():
            _connector(
                ax,
                start,
                positions[name],
                color=MUTED,
                patch_b=nodes[name],
                linewidth=1.35,
                mutation_scale=12.5,
                zorder=3,
            )
            ax.text(
                *label_position,
                rf"$\eta^{{{name}}}$",
                color=INK,
                fontsize=20,
                ha="center",
                va="center",
                zorder=5,
            )

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=BLUE,
                lw=1.8,
                linestyle=confounder_style,
                label="Confounder",
            ),
            Line2D([0], [0], color=RED, lw=1.8, label="Causal hyperedge"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            bbox_to_anchor=(0.972, 0.055),
            frameon=False,
            fontsize=14.5,
            handlelength=3.2,
            handletextpad=0.9,
            labelspacing=0.55,
            borderaxespad=0.0,
        )

        outputs: list[Path] = []
        for suffix in (".png", ".svg", ".pdf"):
            path = output.with_suffix(suffix)
            save_kwargs: dict[str, object] = {
                "bbox_inches": "tight",
                "pad_inches": 0.08,
                "facecolor": "white",
            }
            if suffix == ".png":
                save_kwargs["dpi"] = 300
            fig.savefig(path, **save_kwargs)
            outputs.append(path)
        plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-loading", type=float, default=0.8)
    parser.add_argument("--target-loading", type=float, default=0.1)
    args = parser.parse_args()
    for path in render(
        args.output,
        source_loading=args.source_loading,
        target_loading=args.target_loading,
    ):
        print(path)


if __name__ == "__main__":
    main()
