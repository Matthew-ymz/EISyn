from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from PIL import Image, ImageChops, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FIGURE_DIR = ROOT / "fig" / "part1_synergy_comparison"
SOURCE_DIR = FIGURE_DIR / "figure1_sources"
OUTPUT_STEM = FIGURE_DIR / "figure1_integrated_hierarchy_draft"

INTERVENTION_DIAGRAM = SOURCE_DIR / "interventional_peid_decomposition.png"
HYPEREDGE_DIAGRAM = SOURCE_DIR / "confounded_hyperedge_system.png"
SYSTEM_BENCHMARK = FIGURE_DIR / "six_system_five_method_synergy_panels.png"
CONFOUNDER_BENCHMARK = (
    ROOT
    / "fig"
    / "granger_peid_mlp_comparison"
    / "sine_beta_original_neighborhood_one_decimal_all_methods.png"
)
SYSTEM_BENCHMARK_LARGE_TEXT = SOURCE_DIR / "six_system_large_text.png"
CONFOUNDER_BENCHMARK_LARGE_TEXT = SOURCE_DIR / "confounder_large_text.png"
CONFOUNDER_RESULT = (
    ROOT
    / "results"
    / "granger_peid_mlp_comparison"
    / "sine_beta_original_neighborhood_one_decimal.json"
)
KURAMOTO_HIERARCHY_RESULT = (
    ROOT / "results" / "mixed_order_kuramoto_kout_main" / "summary.json"
)


COMPOSITE_STYLE = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.2,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
mpl.rcParams.update(COMPOSITE_STYLE)


def trim_white(image: Image.Image, *, tolerance: int = 12, pad: int = 10) -> Image.Image:
    """Remove only near-white outer margins; do not alter image pixels."""
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, (255, 255, 255))
    difference = ImageChops.difference(rgb, background).convert("L")
    mask = difference.point(lambda value: 255 if value > tolerance else 0)
    box = mask.getbbox()
    if box is None:
        return rgb
    left, upper, right, lower = box
    return rgb.crop(
        (
            max(0, left - pad),
            max(0, upper - pad),
            min(rgb.width, right + pad),
            min(rgb.height, lower + pad),
        )
    )


def load_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def remove_internal_panel_letters(image: Image.Image) -> Image.Image:
    """Remove the source figure's two outer a/b labels without touching axes."""
    cleaned = image.copy()
    width, height = cleaned.size
    draw = ImageDraw.Draw(cleaned)
    x_right = round(0.030 * width)
    draw.rectangle(
        (0, round(0.035 * height), x_right, round(0.090 * height)),
        fill="white",
    )
    draw.rectangle(
        (0, round(0.485 * height), x_right, round(0.555 * height)),
        fill="white",
    )
    return cleaned


def remove_six_grid_panel_letters(image: Image.Image) -> Image.Image:
    """Remove the source grid's a-f prefixes while preserving system titles."""
    cleaned = image.copy()
    width, height = cleaned.size
    draw = ImageDraw.Draw(cleaned)
    x_positions = (0.038, 0.322, 0.610)
    y_positions = (0.009, 0.500)
    for y_fraction in y_positions:
        for x_fraction in x_positions:
            left = round(x_fraction * width)
            upper = round(y_fraction * height)
            draw.rectangle(
                (
                    left,
                    upper,
                    left + round(0.007 * width),
                    upper + round(0.030 * height),
                ),
                fill="white",
            )
    return cleaned


def image_panel(
    fig: plt.Figure,
    bounds: tuple[float, float, float, float],
    image: Image.Image,
) -> plt.Axes:
    ax = fig.add_axes(bounds)
    ax.imshow(image, interpolation="lanczos")
    ax.set_axis_off()
    return ax


def panel_letter(
    fig: plt.Figure,
    *,
    x: float,
    y: float,
    letter: str,
) -> None:
    fig.text(x, y, letter, fontsize=9.2, fontweight="bold", va="top", ha="left")


def prepare_large_text_sources() -> None:
    """Re-render cached numerical results with fonts sized for the final panel."""
    from scripts.classic_network_dynamics_benchmark import run_part1_combined_synergy_figure
    from scripts.compare_granger_peid_mlp import _plot_sine_beta_combined_readout_sweep

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    run_part1_combined_synergy_figure(
        figure_path=SYSTEM_BENCHMARK_LARGE_TEXT,
        font_size=18.0,
        title_font_size=19.0,
        figure_size=(12.8, 7.0),
        include_panel_letters=False,
        legend_font_size=15.5,
        compact_xlabels=True,
        legend_position="top",
    )
    payload = json.loads(CONFOUNDER_RESULT.read_text(encoding="utf-8"))
    _plot_sine_beta_combined_readout_sweep(
        payload["full_result"],
        SOURCE_DIR,
        liang_result=payload["liang_result"],
        stem=CONFOUNDER_BENCHMARK_LARGE_TEXT.stem,
        include_oracle=False,
        font_scale=2.65,
        figure_size=(9.2, 6.1),
        include_panel_labels=False,
        legend_columns=3,
        compact_text=True,
        reserve_legend_band=True,
    )


def draw_kuramoto_hierarchy_panel(fig: plt.Figure, *, payload: dict) -> None:
    """Draw all three cached networks and complete trees at a common text size."""
    from scripts.run_mixed_order_kuramoto_kout_main import (
        NAMES, NETWORK_POSITIONS, WITHIN_COUPLING, pairwise_ring_weights,
        tree_from_record,
    )
    from scripts.synergy_hierarchy_tree_plot import _layout, _blend_with_white

    records = payload["conditions"]
    tolerance = float(payload["experiment_contract"]["syn_nonnegative_tolerance_bits"])
    values = np.array([atom["value_bits"] for row in records for atom in row["atoms"]])
    violations = values < -tolerance
    if violations.any():
        raise ValueError(f"Syn minimum={values.min():.6g} bits; threshold={-tolerance:g}; "
                         f"affected count={violations.sum()}")
    print(f"SPT Syn tolerance={tolerance:g} bits; numerical-zero-band count="
          f"{((values < 0) & (values >= -tolerance)).sum()}; values displayed without clipping")
    scale = float(values.max())
    positions = np.array([NETWORK_POSITIONS[name] for name in NAMES])
    weights = pairwise_ring_weights(float(payload["experiment_contract"]["pairwise_asymmetry"]))
    fig.text(.066, .382, "Mixed-order Kuramoto: complete SPT", fontsize=7, weight="bold", va="top")
    # Label the two planted communities once, on opposite sides of the first
    # reference network, without repeating the annotations across conditions.
    fig.text(.097, .292, "pairwise\ntriangle", ha="right", va="center",
             fontsize=5.2, linespacing=1.05, color="#355F7A")
    fig.text(.269, .292, "triadic\nhyperedge", ha="left", va="center",
             fontsize=5.2, linespacing=1.05, color="#A95E24")
    for index, record in enumerate(records):
        left = .035 + index * .323
        width = .295
        k = float(record["k_out"])
        fig.text(left + width / 2, .356, rf"$K_{{\mathrm{{out}}}}={k:g}$",
                 ha="center", va="top", fontsize=7, weight="bold")
        network = fig.add_axes((left + .015, .237, width - .030, .116))
        if k > 0:
            for i in range(3):
                for j in range(3, 6):
                    network.plot(*zip(positions[i], positions[j]), color="#B8BDC5",
                                 lw=.23 + .85*k/WITHIN_COUPLING, alpha=.50, zorder=0)
        network.add_patch(Polygon(positions[3:], closed=True, facecolor="#D9903D",
                                  edgecolor="#B86722", alpha=.28, lw=.9))
        for (i, j), weight in weights.items():
            network.plot(*zip(positions[i], positions[j]), color="#4477A8",
                         lw=.4 + 1.2*weight/max(weights.values()), zorder=2)
        for i, (x, y) in enumerate(positions):
            network.scatter(x, y, s=58, facecolor="#DCEAF3" if i < 3 else "#F4DDC5",
                            edgecolor="#355F7A" if i < 3 else "#A95E24", lw=.6, zorder=3)
            network.text(x, y, str(i+1), ha="center", va="center", fontsize=5.4, zorder=4)
        network.set(xlim=(-1.42, 1.42), ylim=(-.95, .92), aspect="equal")
        network.axis("off")
        tree = tree_from_record(record)
        axis = fig.add_axes((left, .022, width, .198))
        layout = _layout(tree)
        max_depth = max(-y for x, y in layout.values())
        # Depth-specific spacing retains every node while giving each tree equal area.
        def draw(node):
            x, y = layout[node.sources]
            for child in node.children:
                cx, cy = layout[child.sources]
                axis.plot([x, cx], [y, cy], color="#AFBAC2", lw=.65, zorder=1)
                draw(child)
            internal = bool(node.children)
            label = ",".join(name.removeprefix("theta") for name in node.sources)
            if internal:
                label = "{" + label + "}" + f"\nSyn {node.residual:.2f}"
            strength = abs(float(node.residual))/scale if internal else 0
            axis.text(x, y, label, ha="center", va="center", fontsize=5.1,
                      linespacing=1.12, color="#24313C", zorder=3,
                      bbox=dict(boxstyle="round,pad=.24", lw=.55+.6*strength,
                                facecolor=_blend_with_white("#267A70", .10+.52*strength) if internal else "#F4F6F8",
                                edgecolor="#267A70" if internal else "#8B96A1"))
        draw(tree)
        axis.text(.5, 1.055, rf"$\Xi={record['root_xi_bits']:.2f}$ bits",
                  transform=axis.transAxes, ha="center", fontsize=6.1)
        axis.set(xlim=(-.7, 5.7), ylim=(-max_depth-.32, .42))
        axis.axis("off")


def build_figure() -> plt.Figure:
    # Explicit three-band layout: mechanism, benchmark evidence, complete SPTs.
    fig = plt.figure(figsize=(183 / 25.4, 165 / 25.4), facecolor="white")

    intervention = trim_white(load_rgb(INTERVENTION_DIAGRAM), tolerance=10, pad=4)
    hyperedge = trim_white(load_rgb(HYPEREDGE_DIAGRAM), tolerance=10, pad=8)
    systems = trim_white(load_rgb(SYSTEM_BENCHMARK_LARGE_TEXT), tolerance=8, pad=4)

    # Retain both original panels: pairwise readouts above and interaction /
    # synergy readouts below.
    confounder = trim_white(load_rgb(CONFOUNDER_BENCHMARK_LARGE_TEXT), tolerance=8, pad=4)
    hierarchy_sweep = json.loads(
        KURAMOTO_HIERARCHY_RESULT.read_text(encoding="utf-8")
    )
    panel_letter(fig, x=0.022, y=0.985, letter="a")
    panel_letter(fig, x=0.589, y=0.985, letter="b")
    image_panel(fig, (0.022, 0.752, 0.535, 0.233), intervention)
    image_panel(fig, (0.589, 0.750, 0.390, 0.233), hyperedge)

    panel_letter(fig, x=0.022, y=0.738, letter="c")
    image_panel(fig, (0.022, 0.409, 0.543, 0.320), systems)
    image_panel(fig, (0.589, 0.409, 0.390, 0.320), confounder)

    panel_letter(fig, x=0.022, y=0.382, letter="d")
    draw_kuramoto_hierarchy_panel(fig, payload=hierarchy_sweep)

    return fig


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description="Compose the cached Part 1 main figure.")
    parser.add_argument("--refresh-sources", action="store_true",
                        help="Re-render benchmark source panels from their cached results.")
    args = parser.parse_args()
    if args.refresh_sources or not all(path.exists() for path in
                                      (SYSTEM_BENCHMARK_LARGE_TEXT, CONFOUNDER_BENCHMARK_LARGE_TEXT)):
        prepare_large_text_sources()
    # Upstream cached-figure renderers use enlarged source fonts. Restore the
    # final composite style before adding native axes to the integrated figure.
    mpl.rcParams.update(COMPOSITE_STYLE)
    fig = build_figure()
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":
    main()
