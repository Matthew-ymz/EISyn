from __future__ import annotations

import itertools
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
    ROOT / "results" / "pairwise_asymmetry_kuramoto_mlp" / "summary.json"
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


def select_asymmetric_kuramoto_condition(
    payload: dict[str, object],
    *,
    rho: float,
) -> dict[str, object]:
    """Convert one sweep level to the panel's module-summary schema."""
    rows = [
        row
        for row in payload["rows"]
        if np.isclose(float(row["rho"]), float(rho))
    ]
    if not rows:
        raise ValueError(f"No Kuramoto asymmetry rows found for rho={rho}.")
    summary: dict[str, object] = {}
    for mechanism in ("pairwise", "triadic"):
        mechanism_summary: dict[str, float] = {}
        for metric in (
            "pair_atom_bits",
            "triple_residual_bits",
            "positive_atom_bits",
            "pair_atom_fraction",
            "triple_residual_fraction",
        ):
            values = np.asarray(
                [
                    float(row["modules"][mechanism][metric])
                    for row in rows
                ]
            )
            mechanism_summary[f"{metric}_mean"] = float(values.mean())
            mechanism_summary[f"{metric}_sem"] = float(
                values.std(ddof=1) / np.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )
        summary[mechanism] = mechanism_summary
    mae = np.asarray(
        [
            float(row["dynamics_fit"]["heldout_circular_mae_rad"])
            for row in rows
        ]
    )
    summary["heldout_circular_mae_rad_mean"] = float(mae.mean())
    delta = np.asarray(
        [
            float(row["modules"]["triadic"]["triple_residual_fraction"])
            - float(row["modules"]["pairwise"]["triple_residual_fraction"])
            for row in rows
        ]
    )
    summary["paired_delta_triple_fraction"] = {
        "mean": float(delta.mean()),
        "sem": float(
            delta.std(ddof=1) / np.sqrt(len(delta))
            if len(delta) > 1
            else 0.0
        ),
        "positive_count": int(np.sum(delta > 0.0)),
        "n_seeds": len(delta),
    }
    return {
        "rows": rows,
        "summary": summary,
        "selected_rho": float(rho),
        "pairwise_edge_weights": rows[0]["pairwise_edge_weights"],
    }


def draw_kuramoto_hierarchy_panel(
    fig: plt.Figure,
    *,
    payload: dict[str, object],
) -> None:
    """Show the mechanism and atom contrast for equal-size mixed-order modules."""
    pair_color = "#4477AA"
    triadic_color = "#D9922E"
    cross_color = "#B8BDC5"
    pairwise_weights = payload.get("pairwise_edge_weights", {})
    edge_keys = (
        "theta1-theta2",
        "theta1-theta3",
        "theta2-theta3",
    )
    edge_values = np.asarray(
        [float(pairwise_weights.get(key, 0.75)) for key in edge_keys]
    )
    edge_widths = 0.75 + 1.75 * edge_values / edge_values.max()

    network_ax = fig.add_axes((0.055, 0.050, 0.205, 0.185))
    atom_ax = fig.add_axes((0.325, 0.063, 0.250, 0.158))
    mass_ax = fig.add_axes((0.655, 0.063, 0.305, 0.158))

    positions = np.array(
        [
            [0.0, 0.62],
            [-0.55, -0.35],
            [0.55, -0.35],
            [2.0, 0.62],
            [1.45, -0.35],
            [2.55, -0.35],
        ]
    )
    for (left, right), edge_width in zip(
        itertools.combinations((0, 1, 2), 2),
        edge_widths,
    ):
        network_ax.plot(
            *zip(positions[left], positions[right]),
            color=pair_color,
            lw=float(edge_width),
            zorder=1,
        )
    network_ax.add_patch(
        Polygon(
            positions[[3, 4, 5]],
            closed=True,
            facecolor=triadic_color,
            edgecolor=triadic_color,
            alpha=0.18,
            lw=1.8,
            zorder=1,
        )
    )
    for left in (0, 1, 2):
        for right in (3, 4, 5):
            network_ax.plot(
                *zip(positions[left], positions[right]),
                color=cross_color,
                lw=0.42,
                alpha=0.38,
                linestyle=(0, (2.0, 2.0)),
                zorder=0,
            )
    network_ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=105,
        color="white",
        edgecolor="0.15",
        lw=0.75,
        zorder=2,
    )
    for index, (x_value, y_value) in enumerate(positions):
        network_ax.text(
            x_value,
            y_value,
            str(index + 1),
            ha="center",
            va="center",
            fontsize=5.8,
            zorder=3,
        )
    network_ax.text(
        0.0,
        -0.72,
        r"$K_2$",
        color=pair_color,
        ha="center",
        fontsize=5.8,
    )
    network_ax.text(
        2.0,
        -0.72,
        r"$K_3$",
        color=triadic_color,
        ha="center",
        fontsize=5.8,
    )
    network_ax.set(xlim=(-0.95, 2.95), ylim=(-0.87, 1.02))
    network_ax.axis("off")

    pair_bits = np.asarray(
        [
            float(payload["summary"]["pairwise"]["pair_atom_bits_mean"]),
            float(payload["summary"]["triadic"]["pair_atom_bits_mean"]),
        ]
    )
    triple_bits = np.asarray(
        [
            float(payload["summary"]["pairwise"]["triple_residual_bits_mean"]),
            float(payload["summary"]["triadic"]["triple_residual_bits_mean"]),
        ]
    )
    atom_positions = np.asarray([1.0, 0.0])
    atom_ax.barh(
        atom_positions,
        pair_bits,
        color=pair_color,
        height=0.56,
        edgecolor="white",
        lw=0.8,
    )
    atom_ax.barh(
        atom_positions,
        triple_bits,
        left=pair_bits,
        color=triadic_color,
        height=0.56,
        edgecolor="white",
        lw=0.8,
    )
    for y_value, pair_value, triple_value in zip(
        atom_positions,
        pair_bits,
        triple_bits,
    ):
        if pair_value > 0.25:
            atom_ax.text(
                pair_value / 2.0,
                y_value,
                f"{pair_value:.2f}",
                color="white",
                fontsize=5.2,
                fontweight="bold",
                ha="center",
                va="center",
            )
        else:
            atom_ax.text(
                pair_value + 0.04,
                y_value + 0.34,
                f"{pair_value:.2f}",
                color=pair_color,
                fontsize=5.0,
                fontweight="bold",
                ha="left",
                va="center",
            )
        atom_ax.text(
            pair_value + triple_value / 2.0,
            y_value,
            f"{triple_value:.2f}",
            color="white",
            fontsize=5.2,
            fontweight="bold",
            ha="center",
            va="center",
        )
    atom_ax.set_yticks(atom_positions, ("{1,2,3}", "{4,5,6}"))
    atom_ax.set_xlabel("bits", fontsize=5.8)
    atom_ax.tick_params(axis="both", labelsize=5.5, length=2)
    atom_ax.set(xlim=(0.0, 3.55), ylim=(-0.55, 1.55))
    atom_ax.grid(axis="x", color="0.92", lw=0.45, zorder=0)
    atom_ax.spines["left"].set_visible(False)
    atom_ax.spines["top"].set_visible(False)
    atom_ax.spines["right"].set_visible(False)

    pair_fractions = np.asarray(
        [
            float(row["modules"]["pairwise"]["triple_residual_fraction"])
            for row in payload["rows"]
        ]
    )
    triadic_fractions = np.asarray(
        [
            float(row["modules"]["triadic"]["triple_residual_fraction"])
            for row in payload["rows"]
        ]
    )
    triple_means = np.asarray(
        [pair_fractions.mean(), triadic_fractions.mean()]
    )
    triple_sems = np.asarray(
        [
            float(payload["summary"]["pairwise"]["triple_residual_fraction_sem"]),
            float(payload["summary"]["triadic"]["triple_residual_fraction_sem"]),
        ]
    )
    pair_means = 1.0 - triple_means
    y_positions = np.asarray([1.0, 0.0])
    bar_height = 0.56
    mass_ax.barh(
        y_positions,
        pair_means,
        height=bar_height,
        color=pair_color,
        edgecolor="white",
        lw=0.8,
        zorder=1,
    )
    mass_ax.barh(
        y_positions,
        triple_means,
        left=pair_means,
        height=bar_height,
        color=triadic_color,
        edgecolor="white",
        lw=0.8,
        zorder=1,
    )
    for y_value, pair_mean, triple_mean, triple_sem in zip(
        y_positions,
        pair_means,
        triple_means,
        triple_sems,
    ):
        if pair_mean >= 0.12:
            mass_ax.text(
                pair_mean / 2.0,
                y_value,
                f"{100.0 * pair_mean:.1f}%",
                color="white",
                fontsize=5.3,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=3,
            )
        else:
            mass_ax.text(
                pair_mean + 0.008,
                y_value + 0.34,
                f"{100.0 * pair_mean:.1f}%",
                color=pair_color,
                fontsize=5.0,
                fontweight="bold",
                ha="left",
                va="center",
                zorder=3,
            )
        mass_ax.text(
            pair_mean + triple_mean / 2.0,
            y_value,
            f"{100.0 * triple_mean:.1f}%"
            f"\n± {100.0 * triple_sem:.1f}",
            color="white",
            fontsize=5.1,
            fontweight="bold",
            ha="center",
            va="center",
            linespacing=0.88,
            zorder=3,
        )
    pair_boundaries = 1.0 - pair_fractions
    triadic_boundaries = 1.0 - triadic_fractions
    for y_value, boundaries in zip(
        y_positions,
        (pair_boundaries, triadic_boundaries),
    ):
        for offset, boundary in zip((-0.12, 0.0, 0.12), boundaries):
            mass_ax.plot(
                boundary,
                y_value + offset,
                marker="|",
                color="0.12",
                ms=4.0,
                mew=0.65,
                zorder=4,
            )
    mass_ax.text(
        0.18,
        1.47,
        r"$\mathcal{A}_2$",
        color=pair_color,
        fontsize=5.3,
        fontweight="bold",
        ha="center",
        va="center",
    )
    mass_ax.text(
        0.72,
        1.47,
        r"$\mathcal{A}_3$",
        color=triadic_color,
        fontsize=5.3,
        fontweight="bold",
        ha="center",
        va="center",
    )
    mass_ax.set(
        xlim=(0.0, 1.0),
        ylim=(-0.45, 1.63),
        yticks=y_positions,
        yticklabels=("{1,2,3}", "{4,5,6}"),
        xticks=(0.0, 0.25, 0.50, 0.75, 1.0),
        xticklabels=("0", "25", "50", "75", "100%"),
    )
    mass_ax.tick_params(axis="both", labelsize=5.5, length=2)
    mass_ax.grid(axis="x", color="0.92", lw=0.45, zorder=0)
    mass_ax.spines["left"].set_visible(False)
    mass_ax.spines["top"].set_visible(False)
    mass_ax.spines["right"].set_visible(False)


def build_figure() -> plt.Figure:
    # 183 mm × 138 mm: full-width mixed-modality figure with a hierarchy
    # validation band beneath the original three-panel argument.
    fig = plt.figure(figsize=(183 / 25.4, 138 / 25.4), facecolor="white")

    intervention = trim_white(load_rgb(INTERVENTION_DIAGRAM), tolerance=10, pad=4)
    hyperedge = trim_white(load_rgb(HYPEREDGE_DIAGRAM), tolerance=10, pad=8)
    systems = trim_white(load_rgb(SYSTEM_BENCHMARK_LARGE_TEXT), tolerance=8, pad=4)

    # Retain both original panels: pairwise readouts above and interaction /
    # synergy readouts below.
    confounder = trim_white(load_rgb(CONFOUNDER_BENCHMARK_LARGE_TEXT), tolerance=8, pad=4)
    hierarchy_sweep = json.loads(
        KURAMOTO_HIERARCHY_RESULT.read_text(encoding="utf-8")
    )
    hierarchy_payload = select_asymmetric_kuramoto_condition(
        hierarchy_sweep,
        rho=0.25,
    )

    panel_letter(
        fig,
        x=0.025,
        y=0.982,
        letter="a",
    )
    panel_letter(fig, x=0.585, y=0.982, letter="b")
    image_panel(fig, (0.025, 0.658, 0.525, 0.292), intervention)
    image_panel(fig, (0.585, 0.650, 0.390, 0.300), hyperedge)

    panel_letter(
        fig,
        x=0.025,
        y=0.638,
        letter="c",
    )
    image_panel(fig, (0.025, 0.275, 0.545, 0.340), systems)

    # The generative diagram and the beta-sweep readouts are one experiment.
    # Align them as a single stacked panel b without a second outer heading.
    image_panel(fig, (0.585, 0.265, 0.390, 0.370), confounder)

    panel_letter(fig, x=0.025, y=0.250, letter="d")
    draw_kuramoto_hierarchy_panel(fig, payload=hierarchy_payload)

    return fig


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    prepare_large_text_sources()
    # Upstream cached-figure renderers use enlarged source fonts. Restore the
    # final composite style before adding native axes to the integrated figure.
    mpl.rcParams.update(COMPOSITE_STYLE)
    fig = build_figure()
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches=None)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches=None)
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":
    main()
