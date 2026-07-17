#!/usr/bin/env python3
"""Compose all requested DMF critical-window panels into one figure."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_dmf_critical_phi_hierarchy_topology import configure_matplotlib
from scripts.analyze_dmf_critical_phi_yeo7_hierarchy import COMPONENT_COLORS, NETWORK_COLORS
from scripts.brain_surface_plot import (
    draw_brain_map_four_views,
    load_fsaverage_meshes,
    parcel_values_to_vertices,
)
from scripts.plot_dmf_fixed_uniform_multihorizon import safe_sem
from scripts.plot_dmf_kuramoto_aligned_fullstate import (
    CRITICAL_COUPLING,
    CRITICAL_HIGH,
    CRITICAL_LOW,
    load_dmf,
    load_mean_rate,
)


DEFAULT_TOPOLOGY = ROOT / "results" / "dmf_phi_eid_hierarchical_topology" / "critical_hierarchy.npz"
DEFAULT_YEO7 = ROOT / "results" / "dmf_phi_eid_yeo7_hierarchy" / "critical_yeo7_hierarchy.npz"
DEFAULT_WMS = ROOT / "results" / "dmf_83_whole_system_wms" / "aligned_observational_tau300_n2048_seeds3_10_dense_g01.npz"
DEFAULT_OUTPUT = ROOT / "fig" / "dmf_roi_yeo7_critical_summary_wms.png"
DEFAULT_PANEL_C_OUTPUT = ROOT / "fig" / "dmf_roi_yeo7_panel_c_observational_whole_system_wms.png"
DEFAULT_BASE_FIGURE = ROOT / "fig" / "dmf_roi_yeo7_critical_summary_phi_axis_8_13.png"


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def load_pair_curve(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        if not np.all(np.asarray(archive["completed"], dtype=bool)):
            raise RuntimeError("The ROI-pair PhiEID sweep is incomplete.")
        return {
            "G": np.asarray(archive["G"], dtype=float),
            "pair_mean_phi_eid": np.asarray(archive["pair_mean_phi_eid"], dtype=float),
            "pair_positive_fraction": np.asarray(archive["pair_positive_fraction"], dtype=float),
            "pair_count": np.asarray(archive["pair_count"], dtype=int),
        }


def load_whole_system_wms(path: Path) -> dict[str, np.ndarray]:
    """Load ordinary-MI WMS computed from natural 83-ROI rate trajectories."""
    with np.load(path) as archive:
        if str(np.asarray(archive["source_state"]).item()) != "se_si":
            raise ValueError("Aligned WMS cache must use the complete E/I source state.")
        g_values = np.asarray(archive["G"], dtype=float)
        return {
            "G": g_values,
            "phi_wms": np.asarray(archive["phi_wms"], dtype=float),
        }


def plot_wms_panel(wms: dict[str, np.ndarray], output: Path) -> None:
    """Render the panel-C WMS replacement without the surface-map dependencies."""
    configure_matplotlib()
    figure, axis = plt.subplots(figsize=(5.7, 3.8), constrained_layout=True)
    wms_axis = add_rate_phi_dual_axis(
        axis,
        load_mean_rate(),
        np.asarray(wms["G"], dtype=float),
        np.asarray(wms["phi_wms"], dtype=float),
        phi_color="#1B9E77",
        phi_label=r"Aligned observational $\Phi^{WMS}$",
        phi_ylabel=r"Observational $\Phi^{WMS}$ (bits)",
        phi_ylim=(-360.0, 20.0),
    )
    wms_axis.axhline(0.0, color="#1B9E77", lw=0.7, ls="--", alpha=0.55)
    axis.text(-0.16, 1.06, "C", transform=axis.transAxes, fontsize=11, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=450, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def replace_panel_c(base_figure: Path, panel_c: Path, output: Path) -> None:
    """Replace the top-right panel in the existing composite PNG."""
    from PIL import Image

    def on_white(path: Path) -> Image.Image:
        image = Image.open(path).convert("RGBA")
        white = Image.new("RGBA", image.size, "white")
        return Image.alpha_composite(white, image).convert("RGB")

    base = on_white(base_figure)
    panel = on_white(panel_c)
    left = int(round(base.width * 0.650))
    height = int(round(base.height * 0.516))
    panel = panel.resize((base.width - left, height), Image.Resampling.LANCZOS)
    base.paste(panel, (left, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    base.save(output, dpi=(450, 450))


def format_critical_axis(axis: plt.Axes, ylabel: str) -> None:
    axis.axvspan(CRITICAL_LOW, CRITICAL_HIGH, color="0.92", zorder=0)
    axis.axvline(CRITICAL_COUPLING, color="0.35", lw=1.0, ls=":")
    axis.grid(True, color="0.88", lw=0.8)
    axis.set_xlabel("Global coupling")
    axis.set_ylabel(ylabel)


def add_mean_sem(axis: plt.Axes, x: np.ndarray, values: np.ndarray, *, color: str, label: str) -> None:
    mean = np.mean(values, axis=0)
    error = safe_sem(values)
    axis.plot(x, mean, color=color, lw=1.9, label=label)
    axis.fill_between(x, mean - error, mean + error, color=color, alpha=0.16, lw=0)


def add_rate_phi_dual_axis(
    rate_axis: plt.Axes,
    mean_rate: dict[str, np.ndarray],
    phi_g: np.ndarray,
    phi_values: np.ndarray,
    *,
    phi_color: str,
    phi_label: str,
    phi_ylabel: str,
    phi_ylim: tuple[float, float] | None = None,
) -> plt.Axes:
    """Align a Phi curve with the common firing-rate reference on a twin axis."""

    rate_line, = rate_axis.plot(
        mean_rate["G"], mean_rate["mean_rate_hz"], color="0.20", lw=1.2,
        marker="o", ms=3.0, label="Mean firing rate", zorder=2,
    )
    format_critical_axis(rate_axis, "Mean firing rate (Hz)")
    rate_axis.tick_params(axis="y", colors="0.20")
    rate_axis.yaxis.label.set_color("0.20")

    phi_axis = rate_axis.twinx()
    phi_mean = np.mean(phi_values, axis=0)
    phi_error = safe_sem(phi_values)
    phi_line, = phi_axis.plot(
        phi_g, phi_mean, color=phi_color, lw=1.9, marker="o", ms=3.4,
        label=phi_label, zorder=3,
    )
    phi_axis.fill_between(
        phi_g, phi_mean - phi_error, phi_mean + phi_error,
        color=phi_color, alpha=0.16, lw=0,
    )
    phi_axis.set_ylabel(phi_ylabel, color=phi_color)
    if phi_ylim is not None:
        phi_axis.set_ylim(*phi_ylim)
    phi_axis.tick_params(axis="y", colors=phi_color)
    phi_axis.spines["right"].set_visible(True)
    rate_axis.legend(
        handles=(rate_line, phi_line), loc="upper center", bbox_to_anchor=(0.5, 1.19),
        ncol=2, frameon=False,
    )
    return phi_axis


def map_cross_roi_to_desikan_vertices(
    region_labels: np.ndarray,
    cross_roi_values: np.ndarray,
    vertex_labels: np.ndarray,
    atlas_label_names: dict[int, str],
    *,
    hemisphere: str,
) -> np.ndarray:
    """Map the 34 cortical ROI values for one hemisphere to fsaverage vertices."""

    prefix = f"ctx-{hemisphere}-"
    roi_lookup = {
        str(region).removeprefix(prefix): float(value)
        for region, value in zip(region_labels, cross_roi_values)
        if str(region).startswith(prefix)
    }
    atlas_lookup = {
        int(label): roi_lookup[name]
        for label, name in atlas_label_names.items()
        if name in roi_lookup
    }
    mapped_names = {atlas_label_names[label] for label in atlas_lookup}
    missing = sorted(set(roi_lookup) - mapped_names)
    if missing:
        raise ValueError(f"Desikan surface is missing cortical ROIs for {hemisphere}: {missing}")
    if len(roi_lookup) != 34:
        raise ValueError(f"Expected 34 {hemisphere} cortical ROIs, got {len(roi_lookup)}.")
    return parcel_values_to_vertices(vertex_labels, atlas_lookup, background_labels=(0,))


def load_desikan2006_surface_map(
    region_labels: np.ndarray,
    cross_roi_values: np.ndarray,
):
    """Load fsaverage5 plus TemplateFlow's matching 10k Desikan annotations."""

    try:
        import nibabel as nib
        from templateflow.api import get
    except ImportError as error:
        raise ImportError(
            "The cortical panel requires nilearn, nibabel, and templateflow. "
            "Install them with `pip install nilearn nibabel templateflow`."
        ) from error

    left_mesh, right_mesh = load_fsaverage_meshes("fsaverage5")
    vertex_values = []
    for hemisphere, templateflow_hemi in (("lh", "L"), ("rh", "R")):
        annotation_path = get(
            "fsaverage",
            hemi=templateflow_hemi,
            density="10k",
            atlas="Desikan2006",
            segmentation="aparc",
            suffix="dseg",
            extension="label.gii",
            raise_empty=True,
        )
        annotation = nib.load(str(annotation_path))
        vertex_labels = np.asarray(annotation.darrays[0].data, dtype=int)
        label_names = {int(label.key): str(label.label) for label in annotation.labeltable.labels}
        vertex_values.append(
            map_cross_roi_to_desikan_vertices(
                region_labels,
                cross_roi_values,
                vertex_labels,
                label_names,
                hemisphere=hemisphere,
            )
        )
    return left_mesh, right_mesh, vertex_values[0], vertex_values[1]


def plot(
    topology: dict[str, np.ndarray],
    yeo7: dict[str, np.ndarray],
    wms: dict[str, np.ndarray],
    output: Path,
) -> None:
    configure_matplotlib()
    dmf = load_dmf()
    mean_rate = load_mean_rate()
    figure = plt.figure(figsize=(16.5, 7.8))
    grid = figure.add_gridspec(
        2, 20, height_ratios=(1.0, 1.04), left=0.052, right=0.975,
        bottom=0.10, top=0.91, wspace=0.92, hspace=0.22,
    )
    axes = [
        figure.add_subplot(grid[0, 0:6]),
        figure.add_subplot(grid[0, 7:13]),
        figure.add_subplot(grid[0, 14:20]),
        figure.add_subplot(grid[1, 0:5]),
        figure.add_subplot(grid[1, 6:9]),
        figure.add_subplot(grid[1, 10:14]),
    ]
    rate_axis, ei_axis, pair_axis, local_axis, fraction_axis, network_axis = axes

    full_phi_axis = add_rate_phi_dual_axis(
        rate_axis,
        mean_rate,
        np.asarray(dmf["G"], dtype=float),
        np.asarray(dmf["phi_eid"], dtype=float),
        phi_color="#6A3D9A",
        phi_label=r"Full-system $\Phi^{EID}$",
        phi_ylabel=r"Full-system $\Phi^{EID}$ (bits)",
        phi_ylim=(8.0, 13.0),
    )
    full_phi_axis.axhline(0.0, color="#6A3D9A", lw=0.7, ls="--", alpha=0.55)

    add_mean_sem(ei_axis, dmf["G"], dmf["whole_ei"], color="#0072B2", label="Whole EI")
    add_mean_sem(
        ei_axis, dmf["G"], dmf["singleton_ei_sum"], color="#D55E00",
        label="Sum of E/I-source EI",
    )
    format_critical_axis(ei_axis, "Effective information (bits)")
    ei_axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=2, frameon=False)

    wms_axis = add_rate_phi_dual_axis(
        pair_axis,
        mean_rate,
        np.asarray(wms["G"], dtype=float),
        np.asarray(wms["phi_wms"], dtype=float),
        phi_color="#1B9E77",
        phi_label=r"Aligned observational $\Phi^{WMS}$",
        phi_ylabel=r"Observational $\Phi^{WMS}$ (bits)",
        phi_ylim=(-360.0, 20.0),
    )
    wms_axis.axhline(0.0, color="#1B9E77", lw=0.7, ls="--", alpha=0.55)

    strength = np.asarray(topology["strength"], dtype=float)
    local = np.asarray(topology["within_roi"], dtype=float).mean(axis=(0, 1))
    cross = np.asarray(topology["roi_cross_leverage"], dtype=float).mean(axis=(0, 1))
    local_axis.scatter(
        strength, local, s=25, color="#4C78A8", alpha=0.82,
        label="Local E/I coupling",
    )
    local_axis.set_xlabel("Weighted structural strength")
    local_axis.set_ylabel("Local E/I conditional coupling (bits)", color="#4C78A8")
    local_axis.tick_params(axis="y", colors="#4C78A8")
    local_axis.grid(True, color="0.90", lw=0.6)
    cross_axis = local_axis.twinx()
    cross_axis.scatter(
        strength, cross, s=25, color="#D55E00", alpha=0.72,
        marker="D", label="Cross-ROI coupling",
    )
    cross_axis.set_ylabel("Cross-ROI conditional coupling (bits)", color="#D55E00")
    cross_axis.tick_params(axis="y", colors="#D55E00")
    cross_axis.spines["right"].set_visible(True)

    fine = np.asarray(yeo7["fine_phi"], dtype=float)
    roi_fraction = 100.0 * np.asarray(yeo7["within_roi"], dtype=float) / fine
    cross_fraction = 100.0 - roi_fraction
    fractions = (roi_fraction, cross_fraction)
    means = [float(values.mean()) for values in fractions]
    sems = [float(values.std(ddof=1) / np.sqrt(values.size)) for values in fractions]
    fraction_axis.bar([0, 1], means, yerr=sems, color=(COMPONENT_COLORS[0], "#775DA6"), capsize=2, width=0.68)
    fraction_axis.set_xticks([0, 1], ["Within\nROI", "Cross\nROI"])
    fraction_axis.set_ylabel(r"Fraction of $\Phi^{EID}$ (%)")
    fraction_axis.set_ylim(0.0, 80.0)
    fraction_axis.grid(True, axis="y", color="0.90", lw=0.6)
    for index, value in enumerate(means):
        fraction_axis.text(index, value + 2.0, f"{value:.1f}%", ha="center", va="bottom")

    network_values = np.asarray(yeo7["within_group_by_network"], dtype=float).mean(axis=(0, 1))
    order = np.argsort(network_values)
    y = np.arange(len(order))
    network_axis.barh(y, network_values[order], color=[NETWORK_COLORS[index] for index in order], height=0.68)
    network_axis.set_yticks(y, [str(yeo7["network_names"][index]) + f" (n={int(yeo7['network_sizes'][index])})" for index in order])
    network_axis.yaxis.tick_right()
    network_axis.tick_params(axis="y", labelsize=6.0)
    network_axis.set_xlabel(r"Within-group cross-ROI $\Phi^{EID}$ (bits)")
    network_axis.grid(True, axis="x", color="0.90", lw=0.6)

    cross_roi_values = np.asarray(topology["roi_cross_leverage"], dtype=float).mean(axis=(0, 1))
    left_mesh, right_mesh, left_values, right_values = load_desikan2006_surface_map(
        np.asarray(topology["region_labels"]), cross_roi_values
    )
    brain_grid = grid[1, 15:20].subgridspec(
        3, 2, height_ratios=(1.0, 1.0, 0.12), wspace=-0.24, hspace=-0.30
    )
    brain_axes = [
        figure.add_subplot(brain_grid[row, column], projection="3d")
        for row in range(2)
        for column in range(2)
    ]
    brain_colorbar_axis = figure.add_subplot(brain_grid[2, :])
    draw_brain_map_four_views(
        brain_axes,
        brain_colorbar_axis,
        left_mesh,
        right_mesh,
        left_values,
        right_values,
        cmap="viridis",
        colorbar_label="Cross-ROI conditional coupling (bits)",
        colorbar_label_size=6.2,
    )
    brain_axes[0].text2D(
        -0.16, 1.08, "G", transform=brain_axes[0].transAxes, fontsize=11, fontweight="bold"
    )
    brain_axes[1].text2D(
        1.0,
        1.02,
        "68 cortical ROIs shown; 15 non-cortical ROIs not projected",
        transform=brain_axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="0.35",
    )

    for panel, axis in zip(("A", "B", "C", "D", "E", "F"), axes):
        axis.text(-0.16, 1.06, panel, transform=axis.transAxes, fontsize=11, fontweight="bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=450, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--yeo7", type=Path, default=DEFAULT_YEO7)
    parser.add_argument("--wms", type=Path, default=DEFAULT_WMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel-c-only", action="store_true")
    parser.add_argument("--replace-panel-c", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.panel_c_only:
        plot_wms_panel(load_whole_system_wms(args.wms), DEFAULT_PANEL_C_OUTPUT)
        print(f"Saved figure: {DEFAULT_PANEL_C_OUTPUT}")
        return
    if args.replace_panel_c:
        plot_wms_panel(load_whole_system_wms(args.wms), DEFAULT_PANEL_C_OUTPUT)
        replace_panel_c(DEFAULT_BASE_FIGURE, DEFAULT_PANEL_C_OUTPUT, args.output)
        print(f"Saved figure: {args.output}")
        return
    plot(
        load_archive(args.topology),
        load_archive(args.yeo7),
        load_whole_system_wms(args.wms),
        args.output,
    )
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
