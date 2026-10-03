from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle
from PIL import Image, ImageChops, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FIGURE_DIR = ROOT / "fig" / "part1_synergy_comparison"
SOURCE_DIR = FIGURE_DIR / "figure1_sources"
OUTPUT_STEM = ROOT / "paper_assets" / "figure1_integrated_hierarchy_spt_clean"
if os.environ.get("EISYN_NATS_REVIEW_DIR"):
    OUTPUT_STEM = Path(os.environ["EISYN_NATS_REVIEW_DIR"]) / "figure1_integrated_hierarchy_spt_clean"
NATS_PER_BIT = np.log(2.0)

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
CONFOUNDER_BENCHMARK_LARGE_TEXT = SOURCE_DIR / "confounder_mlp_peid_focus.png"
CONFOUNDER_RESULT = (
    ROOT
    / "results"
    / "granger_peid_mlp_comparison"
    / "sine_beta_original_neighborhood_one_decimal.json"
)
CONFOUNDER_READOUT_RESULT = (
    ROOT
    / "results"
    / "granger_peid_mlp_comparison"
    / "sine_beta_intervention_sample_robustness.json"
)
SYSTEM_RESULT_PATHS = {
    "standard_result_path": ROOT / "results/coupled_standard_map_method_comparison/part1_four_method_synergy.json",
    "wilson_cowan_refractory_result_path": ROOT / "results/discrete_iteration_dynamics_benchmark/wilson_cowan_refractory_synergy_sweep.json",
    "kuramoto_result_path": ROOT / "results/classic_network_dynamics_benchmark/kuramoto_coupling_synergy_sweep.json",
    "controlled_henon_result_path": ROOT / "results/henon_unique_five_method_synergy/summary.json",
    "ikeda_result_path": ROOT / "results/discrete_iteration_dynamics_benchmark/ikeda_y_tau_synergy_sweep.json",
    "nicholson_bailey_result_path": ROOT / "results/discrete_iteration_dynamics_benchmark/nicholson_bailey_synergy_sweep.json",
}
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


def panel_border(fig: plt.Figure, bounds: tuple[float, float, float, float]) -> None:
    fig.add_artist(Rectangle(
        (bounds[0], bounds[1]), bounds[2], bounds[3],
        transform=fig.transFigure, fill=False, edgecolor="#D9DEE3", lw=0.45,
    ))


def draw_hyperedge_panel(fig: plt.Figure, bounds: tuple[float, float, float, float]) -> None:
    """Redraw the existing causal schematic at the size used in the composite."""
    ax = fig.add_axes(bounds)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    blue, red, ink = "#315F9E", "#C9574A", "#303841"

    def arrow(start, end, color, *, dashed=False, rad=0, width=1.35, zorder=1):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=7.3,
            connectionstyle=f"arc3,rad={rad}", lw=width, color=color,
            linestyle=(0, (3, 2)) if dashed else "solid", zorder=zorder,
        ))

    w, x, y, h, z = (0.12, .50), (.38, .79), (.38, .21), (.68, .50), (.91, .50)
    # The three dashed paths encode the original common-driver coefficients.
    arrow((.155, .54), (.349, .765), blue, dashed=True)
    arrow((.155, .46), (.349, .235), blue, dashed=True)
    # Route w -> z above x; it must not pass through x or its noise arrow.
    confounder_path = MplPath(
        [( .12, .60), (.16, .94), (.39, .97), (.53, .96),
         (.76, .94), (.85, .73), (.88, .57)],
        [MplPath.MOVETO] + [MplPath.CURVE4] * 6,
    )
    ax.add_patch(FancyArrowPatch(
        path=confounder_path, arrowstyle="-|>", mutation_scale=7.3,
        lw=1.35, color=blue, linestyle=(0, (3, 2)), zorder=1,
    ))
    ax.text(.245, .54, r"$0.8\beta$", color=blue, fontsize=7.2, ha="center")
    ax.text(.245, .18, r"$0.8\beta$", color=blue, fontsize=7.2, ha="center")
    ax.text(.70, .82, r"$0.1\beta$", color=blue, fontsize=7.2, ha="center")

    # The red links retain the joint x,y -> sin(xy) -> z hyperedge semantics.
    arrow((.410, .77), (.628, .55), red, rad=-.28, width=1.65)
    arrow((.410, .23), (.628, .45), red, rad=.28, width=1.65)
    arrow((.742, .50), (.877, .50), red, width=1.65)
    for start, end in [((.49, .88), (.43, .845)), ((.26, .02), (.345, .135)),
                       ((.995, .88), (.94, .585)), ((.005, .91), (.085, .59))]:
        arrow(start, end, ink, width=1.05, zorder=4)

    for point, label, face, edge in [
        (w, r"$w$", "#DCE8F7", blue),
        (x, r"$x$", "#E6EEF8", ink),
        (y, r"$y$", "#E6EEF8", ink),
        (z, r"$z$", "#EFF1F3", ink),
    ]:
        ax.scatter(*point, s=540, marker="o", facecolor=face, edgecolor=edge,
                   linewidth=1.45, transform=ax.transAxes, zorder=3)
        ax.text(*point, label, ha="center", va="center", fontsize=9.3,
                weight="bold", color=ink, transform=ax.transAxes, zorder=4)
    ax.add_patch(Ellipse(h, .135, .25, transform=ax.transAxes,
                         facecolor="#FBE7E4", edgecolor=red, lw=1.5,
                         linestyle=(0, (3, 2)), zorder=3))
    ax.text(*h, r"$\sin(xy)$", ha="center", va="center", color="#9D382F",
            fontsize=8.1, weight="bold", transform=ax.transAxes, zorder=4)
    for px, py, label in [(.54, .84, r"$\eta^x$"), (.23, .035, r"$\eta^y$"),
                          (.965, .93, r"$\eta^z$"), (.015, .94, r"$\eta^w$")]:
        ax.text(px, py, label, fontsize=7.1, color=ink, ha="center", va="center",
                transform=ax.transAxes)
    ax.plot([.55, .61], [.09, .09], color=blue, lw=1.35, ls=(0, (3, 2)))
    ax.text(.625, .09, "Confounder", fontsize=6.8, va="center", color=ink)
    ax.plot([.55, .61], [.025, .025], color=red, lw=1.65)
    ax.text(.625, .025, "Causal hyperedge", fontsize=6.8, va="center", color=ink)


def confounder_result_in_nats() -> tuple[dict, dict]:
    """Use the final 5,120-sample readout and convert only information scores."""
    base = json.loads(CONFOUNDER_RESULT.read_text(encoding="utf-8"))
    readout = json.loads(CONFOUNDER_READOUT_RESULT.read_text(encoding="utf-8"))
    result = copy.deepcopy(readout["updated_full_result"])
    if result["config"]["intervention_samples"] != 5120:
        raise ValueError("Figure 1b requires the documented 5,120-sample readout.")
    # This display audit declares a strict tolerance; it does not retrospectively
    # change the estimator or silently project cached values onto zero.
    tolerance_nats = 0.0
    values = NATS_PER_BIT * np.asarray(
        [row["mlp_peid_xy_synergy"] for row in result["runs"]], dtype=float
    )
    violations = values < -tolerance_nats
    if violations.any():
        raise ValueError(f"Syn minimum={values.min():.6g} nats; threshold="
                         f"{-tolerance_nats:g}; affected count={violations.sum()}")
    print(f"Figure 1b Syn display audit: tolerance={tolerance_nats:g} nats; "
          f"minimum={values.min():.6g}; numerical-zero-band count=0; "
          "significant violation count=0; no clipping")
    information_prefixes = (
        "observational_", "mmi_pid_", "mlp_peid_", "oracle_peid_",
        "surd_", "peid_", "tm_peid_",
    )
    for row in result["summary"]:
        for key, value in row.items():
            if key.startswith(information_prefixes) and key.endswith(("_mean", "_std")):
                row[key] = NATS_PER_BIT * float(value)
    # SHAP, PCMCI, Neural Granger and Liang flow retain their native scores.
    return result, base["liang_result"]


def prepare_large_text_sources() -> None:
    """Re-render cached numerical results with fonts sized for the final panel."""
    from scripts.classic_network_dynamics_benchmark import run_part1_combined_synergy_figure
    from scripts.compare_granger_peid_mlp import _plot_sine_beta_combined_readout_sweep

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    # Display-only audit of existing caches; no retrospective experiment tolerance.
    tolerance_bits = 0.0
    for name, path in SYSTEM_RESULT_PATHS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("runs", payload.get("rows", []))
        values = np.asarray([
            row.get("raw_peid_synergy", row["peid_synergy"]) for row in rows
        ], dtype=float)
        violations = values < -tolerance_bits
        if not values.size or not np.isfinite(values).all():
            raise ValueError(f"{name}: missing or non-finite cached Syn values")
        if violations.any():
            raise ValueError(f"{name}: Syn minimum={values.min():.6g} bits; "
                             f"threshold={-tolerance_bits:g}; affected count={violations.sum()}")
        print(f"Figure 1c {name}: tolerance={tolerance_bits:g} bits; "
              f"minimum={values.min():.6g}; count={values.size}; "
              "numerical-zero-band count=0; significant violation count=0; no clipping")
    run_part1_combined_synergy_figure(
        **SYSTEM_RESULT_PATHS,
        figure_path=SYSTEM_BENCHMARK_LARGE_TEXT,
        font_size=18.0,
        title_font_size=19.0,
        figure_size=(12.8, 7.0),
        include_panel_letters=False,
        legend_font_size=15.5,
        compact_xlabels=True,
        legend_position="top",
    )
    confounder, liang = confounder_result_in_nats()
    _plot_sine_beta_combined_readout_sweep(
        confounder,
        SOURCE_DIR,
        liang_result=liang,
        stem=CONFOUNDER_BENCHMARK_LARGE_TEXT.stem,
        include_oracle=False,
        font_scale=2.65,
        figure_size=(10.4, 5.8),
        include_panel_labels=False,
        legend_columns=4,
        compact_text=True,
        reserve_legend_band=True,
        highlight_mlp_peid=True,
        png_only=True,
    )


def draw_kuramoto_hierarchy_panel(fig: plt.Figure, *, payload: dict) -> None:
    """Draw cached complete trees with local Syn as a share of each root Xi."""
    from scripts.run_mixed_order_kuramoto_kout_main import (
        NAMES, NETWORK_POSITIONS, WITHIN_COUPLING, pairwise_ring_weights,
        tree_from_record,
    )
    from scripts.synergy_hierarchy_tree_plot import _layout, _blend_with_white

    records = payload["conditions"]
    tolerance = float(payload["experiment_contract"]["syn_nonnegative_tolerance_bits"])
    values = np.array([atom["value_bits"] for row in records for atom in row["atoms"]])
    if not values.size or not np.isfinite(values).all():
        raise ValueError("SPT requires finite cached Syn values")
    violations = values < -tolerance
    if violations.any():
        raise ValueError(f"Syn minimum={values.min():.6g} bits; threshold={-tolerance:g}; "
                         f"affected count={violations.sum()}")
    print(f"SPT Syn tolerance={tolerance:g} bits; numerical-zero-band count="
          f"{((values < 0) & (values >= -tolerance)).sum()}; values displayed without clipping")
    shares = []
    for record in records:
        root_xi = float(record["root_xi_bits"])
        if not np.isfinite(root_xi) or root_xi <= 0:
            raise ValueError("SPT percentage display requires finite, positive root Xi")
        closure = sum(float(atom["value_bits"]) for atom in record["atoms"]) - root_xi
        if abs(closure) > tolerance:
            raise ValueError(f"SPT share closure failed: error={closure:.12g} bits; "
                             f"tolerance={tolerance:g} bits")
        # Bits cancel: normalize the unrounded local residual by its own tree's
        # total Xi, not by the cumulative Xi of the node or the node maximum.
        shares.extend(100.0 * float(atom["value_bits"]) / root_xi
                      for atom in record["atoms"])
    scale = max(shares)
    positions = np.array([NETWORK_POSITIONS[name] for name in NAMES])
    weights = pairwise_ring_weights(float(payload["experiment_contract"]["pairwise_asymmetry"]))
    fig.text(.066, .382, "Mixed-order Kuramoto: complete SPT", fontsize=7, weight="bold", va="top")
    fig.text(.955, .382, r"Node values: $\mathrm{Syn}/\Xi_{\mathrm{total}}$ (%)",
             fontsize=5.8, ha="right", va="top")
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
        node_scale = 100.0 / float(record["root_xi_bits"])
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
                label = "{" + label + "}" + f"\nSyn {node_scale * node.residual:.1f}%"
            strength = abs(node_scale * float(node.residual))/scale if internal else 0
            axis.text(x, y, label, ha="center", va="center", fontsize=5.1,
                      linespacing=1.12, color="#24313C", zorder=3,
                      bbox=dict(boxstyle="round,pad=.24", lw=.55+.6*strength,
                                facecolor=_blend_with_white("#267A70", .10+.52*strength) if internal else "#F4F6F8",
                                edgecolor="#267A70" if internal else "#8B96A1"))
        draw(tree)
        axis.text(.5, 1.055, rf"$\Xi={NATS_PER_BIT * record['root_xi_bits']:.2f}$ nats",
                  transform=axis.transAxes, ha="center", fontsize=6.1)
        axis.set(xlim=(-.7, 5.7), ylim=(-max_depth-.32, .42))
        axis.axis("off")


def build_figure() -> plt.Figure:
    # The common-driver schematic and its beta sweep share panel b.
    fig = plt.figure(figsize=(183 / 25.4, 180 / 25.4), facecolor="white")

    intervention = trim_white(load_rgb(INTERVENTION_DIAGRAM), tolerance=10, pad=4)
    systems = trim_white(load_rgb(SYSTEM_BENCHMARK_LARGE_TEXT), tolerance=8, pad=4)

    # Retain both original panels: pairwise readouts above and interaction /
    # synergy readouts below.
    confounder = trim_white(load_rgb(CONFOUNDER_BENCHMARK_LARGE_TEXT), tolerance=8, pad=4)
    hierarchy_sweep = json.loads(
        KURAMOTO_HIERARCHY_RESULT.read_text(encoding="utf-8")
    )
    for bounds in ((.022, .727, .490, .259), (.532, .727, .448, .259),
                   (.022, .425, .958, .289), (.022, .014, .958, .401)):
        panel_border(fig, bounds)
    panel_letter(fig, x=.030, y=.980, letter="a")
    image_panel(fig, (.037, .744, .461, .222), intervention)

    panel_letter(fig, x=.030, y=.707, letter="b")
    draw_hyperedge_panel(fig, (.047, .446, .435, .250))
    image_panel(fig, (.514, .436, .452, .267), confounder)

    panel_letter(fig, x=.540, y=.980, letter="c")
    image_panel(fig, (.546, .741, .422, .226), systems)

    panel_letter(fig, x=.030, y=.406, letter="d")
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
