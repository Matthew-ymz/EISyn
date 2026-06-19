from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Ellipse, FancyArrowPatch, PathPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "reports" / "assets"
STEM = "part3_pair_ignition_mechanism"


COLORS = {
    "ink": "#1F2933",
    "muted": "#8A9099",
    "light_gray": "#D8DADD",
    "blue": "#00A6D6",
    "red": "#C93A2F",
    "red_fill": "#F5CBA7",
    "teal": "#19A7A8",
    "teal_fill": "#DDEDD5",
    "gold": "#D99A2B",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "dejavusans",
        }
    )


def closed_bezier(points: list[tuple[float, float]]) -> MplPath:
    codes = [MplPath.MOVETO]
    vertices = [points[0]]
    for index in range(1, len(points), 3):
        vertices.extend(points[index : index + 3])
        codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    vertices.append(points[0])
    codes.append(MplPath.CLOSEPOLY)
    return MplPath(vertices, codes)


def add_state_axes(ax: plt.Axes) -> None:
    ax.set_xlim(-0.08, 1.05)
    ax.set_ylim(-0.08, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    arrow_kwargs = dict(arrowstyle="-|>", mutation_scale=11, lw=1.35, color="black")
    ax.add_patch(FancyArrowPatch((0.02, 0.02), (0.98, 0.02), **arrow_kwargs))
    ax.add_patch(FancyArrowPatch((0.02, 0.02), (0.02, 0.96), **arrow_kwargs))
    ax.text(0.98, -0.015, r"$z_1$", ha="right", va="top", color=COLORS["ink"], fontsize=8)
    ax.text(-0.015, 0.96, r"$z_2$", ha="right", va="top", color=COLORS["ink"], fontsize=8)


def add_basins(ax: plt.Axes) -> None:
    low = closed_bezier(
        [
            (0.08, 0.10),
            (0.02, 0.25),
            (0.03, 0.70),
            (0.15, 0.82),
            (0.26, 0.93),
            (0.34, 0.67),
            (0.30, 0.53),
            (0.48, 0.48),
            (0.70, 0.43),
            (0.76, 0.23),
            (0.84, 0.05),
            (0.45, 0.09),
            (0.08, 0.10),
        ]
    )
    high = closed_bezier(
        [
            (0.33, 0.78),
            (0.38, 0.93),
            (0.58, 0.86),
            (0.72, 0.84),
            (0.93, 0.88),
            (0.94, 0.60),
            (0.92, 0.28),
            (0.90, 0.05),
            (0.74, 0.08),
            (0.70, 0.30),
            (0.61, 0.42),
            (0.42, 0.45),
            (0.33, 0.49),
            (0.25, 0.57),
            (0.26, 0.71),
            (0.33, 0.78),
        ]
    )
    ax.add_patch(PathPatch(low, facecolor=COLORS["red_fill"], edgecolor=COLORS["red"], lw=2.2, zorder=1))
    ax.add_patch(PathPatch(high, facecolor=COLORS["teal_fill"], edgecolor=COLORS["teal"], lw=2.2, zorder=0))
    ax.text(0.11, 0.61, r"$\mathcal{B}_0$", color=COLORS["red"], fontsize=10, weight="bold")
    ax.text(0.72, 0.70, r"$\mathcal{B}_1$", color=COLORS["teal"], fontsize=10, weight="bold")


def add_panel_header(ax: plt.Axes, letter: str, title: str, subtitle: str, color: str) -> None:
    ax.text(-0.06, 1.05, letter, transform=ax.transAxes, fontsize=9, weight="bold", color="black")
    ax.text(0.50, 1.05, title, transform=ax.transAxes, ha="center", color=color, fontsize=9.5)
    ax.text(0.50, 0.985, subtitle, transform=ax.transAxes, ha="center", color=COLORS["ink"], fontsize=7.6)


def add_failed_singletons(ax: plt.Axes) -> None:
    xi = (0.29, 0.24)
    xj = (0.63, 0.29)
    end_i = (0.54, 0.25)
    end_j = (0.43, 0.36)
    ax.scatter(*xi, s=27, color="red", zorder=4)
    ax.scatter(*xj, s=27, color=COLORS["gold"], edgecolor="black", linewidth=0.3, zorder=4)
    ax.scatter(*end_i, s=14, color=COLORS["red"], alpha=0.72, zorder=4)
    ax.scatter(*end_j, s=14, color=COLORS["gold"], alpha=0.78, zorder=4)
    ax.text(xi[0] - 0.09, xi[1] + 0.03, r"$x_i$", color="red", fontsize=9, weight="bold")
    ax.text(xj[0] + 0.02, xj[1] + 0.00, r"$x_j$", color=COLORS["gold"], fontsize=8.5, weight="bold")
    ax.add_patch(
        FancyArrowPatch(
            xi,
            end_i,
            connectionstyle="arc3,rad=0.06",
            arrowstyle="-|>",
            mutation_scale=9,
            lw=1.05,
            color=COLORS["red"],
            zorder=3,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            xj,
            end_j,
            connectionstyle="arc3,rad=0.34",
            arrowstyle="-|>",
            mutation_scale=9,
            lw=1.05,
            linestyle="--",
            color=COLORS["gold"],
            zorder=3,
        )
    )
    ax.text(0.38, 0.17, r"$\Delta_i$", color=COLORS["red"], fontsize=7.8, ha="center")
    ax.text(0.46, 0.42, r"$\Delta_j$", color=COLORS["gold"], fontsize=7.8, ha="center")
    ax.text(0.22, 0.11, r"$Y_i=0$", color=COLORS["red"], fontsize=8.2, weight="bold")
    ax.text(0.50, 0.11, r"$Y_j=0$", color=COLORS["gold"], fontsize=8.2, weight="bold")


def add_recovered_pair(ax: plt.Axes) -> None:
    xi = (0.25, 0.22)
    xj = (0.47, 0.27)
    merge = (0.70, 0.52)
    high = (0.43, 0.68)
    ax.add_patch(
        Ellipse(
            (0.36, 0.245),
            width=0.32,
            height=0.12,
            angle=12,
            facecolor="none",
            edgecolor=COLORS["blue"],
            lw=1.1,
            linestyle=":",
            zorder=2,
        )
    )
    ax.scatter(*xi, s=27, color="red", zorder=4)
    ax.scatter(*xj, s=27, color=COLORS["gold"], edgecolor="black", linewidth=0.3, zorder=4)
    ax.scatter(*merge, s=16, color="black", zorder=4)
    ax.scatter(*high, s=24, color="#00B8B8", zorder=4)
    ax.text(xi[0] - 0.09, xi[1] + 0.03, r"$x_i$", color="red", fontsize=9, weight="bold")
    ax.text(xj[0] + 0.02, xj[1] - 0.01, r"$x_j$", color=COLORS["gold"], fontsize=8.5, weight="bold")
    ax.text(high[0] - 0.12, high[1] + 0.03, r"$\mathbf{x}(T)$", color=COLORS["teal"], fontsize=8.5, weight="bold")
    ax.add_patch(
        FancyArrowPatch(
            xi,
            merge,
            connectionstyle="arc3,rad=0.30",
            arrowstyle="-|>",
            mutation_scale=11,
            lw=1.35,
            color="black",
            zorder=3,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            merge,
            high,
            connectionstyle="arc3,rad=0.22",
            arrowstyle="-|>",
            mutation_scale=11,
            lw=1.35,
            linestyle="--",
            color="black",
            zorder=3,
        )
    )
    ax.text(0.60, 0.45, r"$\Delta_i+\Delta_j$", color=COLORS["ink"], fontsize=8.1, ha="center", weight="bold")
    ax.text(0.62, 0.76, "high PEID Syn", color=COLORS["blue"], fontsize=7.8, ha="center")
    ax.text(0.79, 0.11, r"$Y=1$", color=COLORS["teal"], fontsize=8.5, weight="bold")


def add_top_rule(fig: plt.Figure) -> None:
    overlay = fig.add_axes([0, 0, 1, 1], frameon=False)
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")
    overlay.text(
        0.50,
        0.93,
        "Joint-required basin transition",
        ha="center",
        va="center",
        color=COLORS["blue"],
        fontsize=11,
    )
    overlay.text(
        0.50,
        0.885,
        r"screen pairs by $S_{ij}=I(\Delta_i,\Delta_j;Y)-I(\Delta_i;Y)-I(\Delta_j;Y)$",
        ha="center",
        va="center",
        color=COLORS["ink"],
        fontsize=7.8,
    )
    for start, end, rad in [((0.50, 0.85), (0.27, 0.735), -0.34), ((0.50, 0.85), (0.73, 0.735), 0.34)]:
        overlay.add_patch(
            FancyArrowPatch(
                start,
                end,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=12,
                lw=1.5,
                color=COLORS["light_gray"],
            )
        )
    overlay.text(0.29, 0.755, "singleton tests fail", color=COLORS["red"], fontsize=7.6, ha="center")
    overlay.text(0.71, 0.755, "pair test succeeds", color=COLORS["teal"], fontsize=7.6, ha="center")


def build_figure() -> plt.Figure:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.09, top=0.64, wspace=0.22)
    add_top_rule(fig)

    add_panel_header(
        axes[0],
        "a",
        "Unrecovered",
        r"each source alone stays in $\mathcal{B}_0$",
        COLORS["red"],
    )
    add_panel_header(
        axes[1],
        "b",
        "Recovered",
        r"joint source pair crosses to $\mathcal{B}_1$",
        COLORS["blue"],
    )
    for axis in axes:
        add_state_axes(axis)
        add_basins(axis)
    add_failed_singletons(axes[0])
    add_recovered_pair(axes[1])
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    for suffix, kwargs in {
        "png": {"dpi": 600},
        "pdf": {},
        "svg": {},
    }.items():
        fig.savefig(OUT_DIR / f"{STEM}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(OUT_DIR / f"{STEM}.png")


if __name__ == "__main__":
    main()
