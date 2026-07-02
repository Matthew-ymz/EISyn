from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


ROWS = ["WMS/MI", "MMI-PID", "MLP+SHAP", "SURD", "MLP+PEID"]
COLUMNS = ["a", "b", "c", "d", "e", "f"]
VALUES = [
    [1, 1, 0, 1, 1, 1],
    [1, 1, 0, 0, 1, 1],
    [1, 0, 1, 1, "1*", 0],
    [0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1],
]


def draw_check(ax, x, y, starred=False):
    size = 0.18
    box = FancyBboxPatch(
        (x - size / 2, y - size / 2),
        size,
        size,
        boxstyle="round,pad=0.008,rounding_size=0.038",
        linewidth=0,
        facecolor="#2FB344",
        zorder=3,
    )
    ax.add_patch(box)
    ax.add_line(
        Line2D(
            [x - 0.052, x - 0.014, x + 0.064],
            [y - 0.004, y - 0.050, y + 0.058],
            color="white",
            linewidth=2.5,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=4,
        )
    )
    if starred:
        ax.text(
            x + 0.125,
            y + 0.030,
            "*",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#111111",
        )


def draw_cross(ax, x, y):
    half = 0.075
    for x0, x1, y0, y1 in [
        (x - half, x + half, y - half, y + half),
        (x - half, x + half, y + half, y - half),
    ]:
        ax.add_line(
            Line2D(
                [x0, x1],
                [y0, y1],
                color="#D7191C",
                linewidth=3.2,
                solid_capstyle="round",
                zorder=4,
            )
        )


def main():
    out_base = Path("docs/reports/assets/method_capability_matrix_table")
    out_base.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(ROWS)
    n_cols = len(COLUMNS)
    label_w = 1.35
    cell_w = 1.05
    header_h = 0.88
    cell_h = 0.86
    width = label_w + n_cols * cell_w
    height = header_h + n_rows * cell_h

    fig_w = 7.2
    fig_h = fig_w * height / width
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    header_y0 = height - header_h
    ax.add_patch(Rectangle((0, header_y0), width, header_h, facecolor="#F4F6F8", edgecolor="none"))

    for i in range(n_rows):
        y0 = height - header_h - (i + 1) * cell_h
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, y0), width, cell_h, facecolor="#FAFBFC", edgecolor="none"))

    outer = "#111111"
    inner = "#D7DCE1"
    ax.plot([0, width], [height, height], color=outer, lw=1.2, clip_on=False)
    ax.plot([0, width], [header_y0, header_y0], color=outer, lw=1.0, clip_on=False)
    ax.plot([0, width], [0, 0], color=outer, lw=1.2, clip_on=False)

    for i in range(1, n_rows):
        y = header_y0 - i * cell_h
        ax.plot([0, width], [y, y], color=inner, lw=0.55, clip_on=False)

    for j in range(n_cols + 1):
        x = label_w + j * cell_w
        ax.plot([x, x], [0, height], color=inner, lw=0.55, clip_on=False)
    ax.plot([0, 0], [0, height], color=inner, lw=0.55, clip_on=False)

    for j, col in enumerate(COLUMNS):
        x = label_w + (j + 0.5) * cell_w
        ax.text(x, header_y0 + header_h / 2, col, ha="center", va="center", fontsize=13, fontweight="bold")

    for i, row in enumerate(ROWS):
        y = header_y0 - (i + 0.5) * cell_h
        ax.text(
            label_w * 0.50,
            y,
            row,
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold" if row == "MLP+PEID" else "normal",
            color="#111111",
        )
        for j, value in enumerate(VALUES[i]):
            x = label_w + (j + 0.5) * cell_w
            if value == 0:
                draw_cross(ax, x, y)
            else:
                draw_check(ax, x, y, starred=value == "1*")

    for ext, kwargs in {
        "png": {"dpi": 600},
        "pdf": {},
        "svg": {},
    }.items():
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", facecolor="white", **kwargs)


if __name__ == "__main__":
    main()
