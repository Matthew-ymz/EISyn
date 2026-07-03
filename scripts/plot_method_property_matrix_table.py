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


NOTES = [
    "A. Non-negativity;",
    "B. Accurate identification of\n   unique causal effects;",
    "C. Accurate identification of\n   synergistic causal effects;",
]
ROWS = ["WMS/MI", "PCMCI", "Neural\nGranger", "MLP+SHAP", "SURD", "IF", "MMI-PID", "MLP+PEID"]
COLUMNS = ["A", "B", "C"]
VALUES = [
    [0, 0, 0],
    [1, 0, "-"],
    [1, 0, "-"],
    ["-", 0, 0],
    [1, 1, 0],
    ["-", 1, "-"],
    [1, 1, 0],
    [1, 1, 1],
]


def draw_check(ax, x, y):
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


def draw_table(ax, x0, y0, width, height):
    n_rows = len(ROWS)
    n_cols = len(COLUMNS)
    label_w = 1.55
    cell_w = (width - label_w) / n_cols
    header_h = 0.78
    cell_h = (height - header_h) / n_rows

    header_y0 = y0 + height - header_h
    ax.add_patch(Rectangle((x0, header_y0), width, header_h, facecolor="#F4F6F8", edgecolor="none"))

    for i in range(n_rows):
        row_y0 = header_y0 - (i + 1) * cell_h
        if i % 2 == 1:
            ax.add_patch(Rectangle((x0, row_y0), width, cell_h, facecolor="#FAFBFC", edgecolor="none"))

    outer = "#111111"
    inner = "#D7DCE1"
    ax.plot([x0, x0 + width], [y0 + height, y0 + height], color=outer, lw=1.2, clip_on=False)
    ax.plot([x0, x0 + width], [header_y0, header_y0], color=outer, lw=1.0, clip_on=False)
    ax.plot([x0, x0 + width], [y0, y0], color=outer, lw=1.2, clip_on=False)

    for i in range(1, n_rows):
        y = header_y0 - i * cell_h
        ax.plot([x0, x0 + width], [y, y], color=inner, lw=0.55, clip_on=False)

    ax.plot([x0, x0], [y0, y0 + height], color=inner, lw=0.55, clip_on=False)
    for j in range(n_cols + 1):
        x = x0 + label_w + j * cell_w
        ax.plot([x, x], [y0, y0 + height], color=inner, lw=0.55, clip_on=False)

    for j, col in enumerate(COLUMNS):
        x = x0 + label_w + (j + 0.5) * cell_w
        ax.text(x, header_y0 + header_h / 2, col, ha="center", va="center", fontsize=13, fontweight="bold")

    for i, row in enumerate(ROWS):
        y = header_y0 - (i + 0.5) * cell_h
        ax.text(
            x0 + label_w * 0.50,
            y,
            row,
            ha="center",
            va="center",
            fontsize=11.8,
            linespacing=0.92,
            fontweight="bold" if row == "MLP+PEID" else "normal",
            color="#111111",
        )
        for j, value in enumerate(VALUES[i]):
            x = x0 + label_w + (j + 0.5) * cell_w
            if value == 1:
                draw_check(ax, x, y)
            elif value == 0:
                draw_cross(ax, x, y)
            else:
                ax.text(x, y, "-", ha="center", va="center", fontsize=13, color="#111111")


def main():
    out_base = Path("docs/reports/assets/method_property_matrix_table")
    out_base.parent.mkdir(parents=True, exist_ok=True)

    canvas_w = 9.6
    canvas_h = 4.6
    fig, ax = plt.subplots(figsize=(canvas_w, canvas_h), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, canvas_h)
    ax.axis("off")

    note_x = 0.25
    note_ys = [3.15, 2.30, 1.35]
    for note, y in zip(NOTES, note_ys):
        ax.text(
            note_x,
            y,
            note,
            ha="left",
            va="center",
            fontsize=18,
            linespacing=1.18,
            color="#111111",
        )

    draw_table(ax, x0=4.55, y0=0.34, width=4.75, height=3.92)

    for ext, kwargs in {
        "png": {"dpi": 600},
        "pdf": {},
        "svg": {},
    }.items():
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", facecolor="white", **kwargs)


if __name__ == "__main__":
    main()
