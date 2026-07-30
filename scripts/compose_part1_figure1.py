from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FIGURE_DIR = ROOT / "fig" / "part1_synergy_comparison"
SOURCE_DIR = FIGURE_DIR / "figure1_sources"
OUTPUT_STEM = FIGURE_DIR / "figure1_integrated_draft"

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


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.2,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


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
        figure_size=(11.8, 7.0),
        include_panel_letters=False,
        legend_font_size=15.5,
        compact_xlabels=True,
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


def build_figure() -> plt.Figure:
    # 183 mm × 100 mm: compact double-column mixed-modality figure.
    fig = plt.figure(figsize=(183 / 25.4, 100 / 25.4), facecolor="white")

    intervention = trim_white(load_rgb(INTERVENTION_DIAGRAM), tolerance=10, pad=4)
    hyperedge = trim_white(load_rgb(HYPEREDGE_DIAGRAM), tolerance=10, pad=8)
    systems = trim_white(load_rgb(SYSTEM_BENCHMARK_LARGE_TEXT), tolerance=8, pad=4)

    # Retain both original panels: pairwise readouts above and interaction /
    # synergy readouts below.
    confounder = trim_white(load_rgb(CONFOUNDER_BENCHMARK_LARGE_TEXT), tolerance=8, pad=4)

    panel_letter(
        fig,
        x=0.025,
        y=0.982,
        letter="a",
    )
    panel_letter(fig, x=0.585, y=0.982, letter="b")
    image_panel(fig, (0.025, 0.567, 0.525, 0.383), intervention)
    image_panel(fig, (0.585, 0.555, 0.390, 0.395), hyperedge)

    panel_letter(
        fig,
        x=0.025,
        y=0.548,
        letter="c",
    )
    image_panel(fig, (0.025, 0.030, 0.545, 0.482), systems)

    # The generative diagram and the beta-sweep readouts are one experiment.
    # Align them as a single stacked panel b without a second outer heading.
    image_panel(fig, (0.585, 0.025, 0.390, 0.530), confounder)

    return fig


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    prepare_large_text_sources()
    fig = build_figure()
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches=None)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches=None)
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":
    main()
