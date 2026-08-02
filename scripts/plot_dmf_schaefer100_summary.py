#!/usr/bin/env python3
"""Plot the Schaefer100 DMF A--G summary and the 83-vs-100 comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.brain_surface_plot import (
    SurfaceMesh,
    draw_brain_map_four_views,
    parcel_values_to_vertices,
)


OLD_SOURCE = ROOT / "exp" / "brain" / "result_lausanne_fig6" / "count_00_fig6b_mean_rate.npz"
OLD_MAIN = ROOT / "results" / "dmf_fullstate_uniform_support" / "confirm_c050_h020_tau300_n2048_no_clip_seeds3_10.npz"
OLD_TOPOLOGY = ROOT / "results" / "dmf_phi_eid_hierarchical_topology" / "critical_hierarchy.npz"
DEFAULT_SURFACE_ASSET = (
    ROOT / "results" / "dmf_schaefer100" / "schaefer100_fsaverage5_surface.npz"
)
DEFAULT_HORIZON = (
    ROOT / "results" / "dmf_schaefer100" / "multihorizon" / "full" / "results.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--wms", type=Path, required=True)
    parser.add_argument("--phi-r", type=Path)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--yeo7", type=Path, required=True)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--horizon", type=Path, default=DEFAULT_HORIZON)
    parser.add_argument("--surface-asset", type=Path, default=DEFAULT_SURFACE_ASSET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path)
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape[axis] <= 1:
        return np.zeros_like(np.mean(array, axis=axis))
    return np.std(array, axis=axis, ddof=1) / np.sqrt(array.shape[axis])


def sd(values: np.ndarray, axis: int = 0) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape[axis] <= 1:
        return np.zeros_like(np.mean(array, axis=axis))
    return np.std(array, axis=axis, ddof=1)


def save(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def panel_label(axis: plt.Axes, label: str, *, y: float = 1.06) -> None:
    axis.text(-0.10, y, label, transform=axis.transAxes, fontsize=10, fontweight="bold")


def critical_format(axis: plt.Axes, _critical_g: np.ndarray) -> None:
    axis.grid(True, color="0.90", lw=0.5, zorder=0)
    axis.set_xlabel("Global coupling $G$")


def direct_values(main: dict[str, np.ndarray], name: str) -> np.ndarray:
    modes = [str(value) for value in np.asarray(main["modes"])]
    return np.asarray(main[name], dtype=float)[modes.index("direct")]


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    centers = np.asarray(values, dtype=float)
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(
        (
            [centers[0] - (midpoints[0] - centers[0])],
            midpoints,
            [centers[-1] + (centers[-1] - midpoints[-1])],
        )
    )


def add_rate_information_panel(
    axis: plt.Axes,
    *,
    source: dict[str, np.ndarray],
    g: np.ndarray,
    values: np.ndarray,
    critical_g: np.ndarray,
    color: str,
    label: str,
    ylabel: str,
) -> plt.Axes:
    rate_line, = axis.plot(
        source["G"], source["mean_rate_hz"], color="0.20", lw=1.0, marker="o", ms=2.2,
        label="Mean firing rate", zorder=3,
    )
    critical_format(axis, critical_g)
    axis.set_ylabel("Mean firing rate (Hz)")
    second = axis.twinx()
    mean = np.mean(values, axis=0)
    error = sd(values)
    info_line, = second.plot(g, mean, color=color, lw=1.55, marker="o", ms=2.4, label=label, zorder=4)
    second.fill_between(g, mean - error, mean + error, color=color, alpha=0.24, lw=0)
    second.set_ylabel(ylabel, color=color)
    second.tick_params(axis="y", colors=color)
    second.spines["right"].set_visible(True)
    axis.legend(
        handles=(rate_line, info_line), loc="lower center", bbox_to_anchor=(0.5, 1.08),
        ncol=2, fontsize=6.3,
    )
    return second


def network_color(name: str) -> str:
    colors = {
        "Visual": "#6A3D9A",
        "Somatomotor": "#1F78B4",
        "Dorsal attention": "#33A02C",
        "Salience / ventral attention": "#E31A1C",
        "Limbic": "#B15928",
        "Frontoparietal control": "#FF7F00",
        "Default mode": "#66A61E",
        "Non-cortical": "#8C8C8C",
    }
    return colors[str(name)]


def load_schaefer100_surface_map(
    asset_path: Path,
    region_labels: np.ndarray,
    roi_values: np.ndarray,
) -> tuple[SurfaceMesh, SurfaceMesh, np.ndarray, np.ndarray]:
    """Map exact Schaefer100 parcel values onto the official fsaverage5 labels."""

    asset = load(asset_path)
    value_by_name = {
        str(name): float(value) for name, value in zip(region_labels, roi_values)
    }
    meshes: list[SurfaceMesh] = []
    vertex_values: list[np.ndarray] = []
    mapped_names: set[str] = set()
    for hemisphere in ("left", "right"):
        label_names = [str(name) for name in asset[f"{hemisphere}_label_names"]]
        label_to_value = {
            label: value_by_name[name]
            for label, name in enumerate(label_names)
            if name in value_by_name
        }
        mapped_names.update(label_names[label] for label in label_to_value)
        meshes.append(
            SurfaceMesh(
                asset[f"{hemisphere}_coordinates"],
                asset[f"{hemisphere}_faces"],
                asset[f"{hemisphere}_sulc"],
            )
        )
        vertex_values.append(
            parcel_values_to_vertices(
                asset[f"{hemisphere}_vertex_labels"],
                label_to_value,
                background_labels=(0,),
            )
        )
    missing = sorted(set(value_by_name) - mapped_names)
    if missing:
        raise ValueError(f"Surface annotation is missing Schaefer100 parcels: {missing}")
    if len(mapped_names) != 100:
        raise ValueError(f"Expected 100 mapped Schaefer parcels, got {len(mapped_names)}.")
    return meshes[0], meshes[1], vertex_values[0], vertex_values[1]


def plot_summary(args: argparse.Namespace) -> None:
    source, main, wms = load(args.source), load(args.main), load(args.wms)
    phi_r = load(args.phi_r) if args.phi_r is not None else None
    topology, yeo = load(args.topology), load(args.yeo7)
    horizon = load(args.horizon)
    critical_g = np.asarray(topology["G"], dtype=float)
    phi = direct_values(main, "phi_eid")
    whole = direct_values(main, "whole_ei")
    singleton = direct_values(main, "singleton_ei_sum")

    figure = plt.figure(figsize=(15.8, 7.2))
    outer_grid = GridSpec(
        1,
        3,
        figure=figure,
        width_ratios=(5.0, 1.05, 9.65),
        wspace=0.0,
    )
    panel_a_grid = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=outer_grid[0, 0],
        height_ratios=(3.1, 1.25),
        hspace=0.05,
    )
    right_grid = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=outer_grid[0, 2],
        height_ratios=(1.0, 0.95),
        hspace=0.32,
    )
    top_grid = GridSpecFromSubplotSpec(
        1,
        5,
        subplot_spec=right_grid[0, 0],
        width_ratios=(2.55, 0.80, 3.20, 0.55, 1.75),
        wspace=0.0,
    )
    bottom_grid = GridSpecFromSubplotSpec(
        1,
        5,
        subplot_spec=right_grid[1, 0],
        width_ratios=(2.75, 0.45, 2.75, 0.45, 3.35),
        wspace=0.0,
    )
    ax_a = figure.add_subplot(panel_a_grid[0, 0])
    ax_a_wms = figure.add_subplot(panel_a_grid[1, 0], sharex=ax_a)
    ax_b = figure.add_subplot(top_grid[0, 0])
    ax_c = figure.add_subplot(top_grid[0, 2])
    ax_d = figure.add_subplot(top_grid[0, 4])
    ax_e = figure.add_subplot(bottom_grid[0, 0])
    ax_f = figure.add_subplot(bottom_grid[0, 2])
    panel_g = GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=bottom_grid[0, 4],
        height_ratios=(1.0, 1.0, 0.10),
        hspace=-0.38,
        wspace=-0.32,
    )
    ax_g = [
        figure.add_subplot(panel_g[row, column], projection="3d")
        for row in range(2)
        for column in range(2)
    ]
    ax_g_cb = figure.add_subplot(panel_g[2, :])

    ax_a_info = add_rate_information_panel(
        ax_a, source=source, g=np.asarray(main["G"], dtype=float), values=phi,
        critical_g=critical_g, color="#6A3D9A", label=r"Full-system $\Xi$",
        ylabel=r"$\Xi$ (bits)",
    )
    ax_a_info.set_ylabel(r"$\Xi$ (bits)", color="#6A3D9A", labelpad=7)
    ax_a.set_xlabel("")
    ax_a.tick_params(axis="x", labelbottom=False)
    wms_values = np.asarray(wms["phi_wms"], dtype=float)
    wms_mean, wms_error = np.mean(wms_values, axis=0), sd(wms_values)
    ax_a_wms.plot(
        np.asarray(wms["G"], dtype=float),
        wms_mean,
        color="#1B9E77",
        lw=1.35,
        marker="o",
        ms=2.0,
        zorder=3,
    )
    ax_a_wms.fill_between(
        np.asarray(wms["G"], dtype=float),
        wms_mean - wms_error,
        wms_mean + wms_error,
        color="#1B9E77",
        alpha=0.22,
        lw=0,
    )
    ax_a_wms.grid(True, color="0.90", lw=0.5, zorder=0)
    ax_a_wms.set_xlabel("Global coupling $G$")
    ax_a_wms.set_ylabel("Observational\n" + r"$\Phi^{WMS}$", color="#1B9E77")
    ax_a_wms.tick_params(axis="y", colors="#1B9E77")
    if phi_r is not None:
        phi_r_values = np.asarray(phi_r["phi_r_mean"], dtype=float)
        phi_r_mean, phi_r_error = np.mean(phi_r_values, axis=0), sd(phi_r_values)
        phi_r_g = np.asarray(phi_r["G"], dtype=float)
        ax_a_phi_r = ax_a_wms.twinx()
        ax_a_phi_r.plot(
            phi_r_g,
            phi_r_mean,
            color="#D55E00",
            lw=1.35,
            marker="s",
            ms=1.9,
            zorder=4,
        )
        ax_a_phi_r.fill_between(
            phi_r_g,
            phi_r_mean - phi_r_error,
            phi_r_mean + phi_r_error,
            color="#D55E00",
            alpha=0.18,
            lw=0,
            zorder=2,
        )
        ax_a_phi_r.set_ylabel(r"Pairwise BOLD-like $\Phi^R$ (bits)", color="#D55E00")
        ax_a_phi_r.tick_params(axis="y", colors="#D55E00")
        ax_a_phi_r.spines["right"].set_visible(True)
    panel_label(ax_a, "A", y=1.04)

    horizon_g = np.asarray(horizon["G"], dtype=float)
    horizons = np.asarray(horizon["horizons"], dtype=float)
    horizon_phi = np.asarray(horizon["phi_eid"], dtype=float)
    mean_horizon_phi = np.mean(horizon_phi, axis=0)
    peak_g_by_seed = horizon_g[np.argmax(horizon_phi, axis=1)]
    peak_g_mean = np.mean(peak_g_by_seed, axis=0)
    image = ax_b.pcolormesh(
        centers_to_edges(horizon_g),
        centers_to_edges(horizons),
        mean_horizon_phi.T,
        shading="flat",
        cmap="magma",
    )
    ax_b.plot(
        peak_g_mean,
        horizons,
        color="white",
        lw=1.15,
        marker="o",
        ms=2.3,
        zorder=3,
    )
    ax_b.set_xlabel("Global coupling $G$")
    ax_b.set_ylabel("Target horizon (steps)")
    ax_b.set_yticks(horizons)
    colorbar = figure.colorbar(image, ax=ax_b, fraction=0.055, pad=0.035)
    colorbar.set_label(r"Mean $\Xi$ (bits)")
    panel_label(ax_b, "B")

    strength = np.asarray(topology["strength"], dtype=float)
    local = np.asarray(topology["within_roi"], dtype=float).mean(axis=(0, 1))
    cross = np.asarray(topology["roi_cross_leverage"], dtype=float).mean(axis=(0, 1))
    ax_c.scatter(strength, local, s=9, color="#4C78A8", alpha=0.82, label="Within ROI")
    ax_c.set_xlabel("Weighted structural strength")
    ax_c.set_ylabel("Within-ROI coupling (bits)", color="#4C78A8", labelpad=1)
    ax_c.tick_params(axis="y", colors="#4C78A8")
    ax_c.grid(True, color="0.90", lw=0.5)
    ax_c2 = ax_c.twinx()
    ax_c2.scatter(strength, cross, s=9, marker="D", color="#D55E00", alpha=0.78, label="Cross-ROI")
    ax_c2.set_ylabel("")
    ax_c2.tick_params(axis="y", colors="#D55E00")
    ax_c2.spines["right"].set_visible(True)
    handles = ax_c.get_legend_handles_labels()[0] + ax_c2.get_legend_handles_labels()[0]
    labels = ax_c.get_legend_handles_labels()[1] + ax_c2.get_legend_handles_labels()[1]
    ax_c.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=6.0)
    panel_label(ax_c, "C")

    fine = np.asarray(topology["fine_phi"], dtype=float)
    within_total = np.asarray(topology["within_roi_total"], dtype=float)
    seed_fractions = np.stack(
        (within_total / fine, (fine - within_total) / fine),
        axis=-1,
    ).mean(axis=1) * 100.0
    means, errors = seed_fractions.mean(axis=0), sd(seed_fractions, axis=0)
    ax_d.bar([0, 1], means, yerr=errors, color=("#66CCEE", "#775DA6"), width=0.68, capsize=2)
    ax_d.set_xticks([0, 1], ["Within\nROI", "Cross\nROI"])
    ax_d.set_ylabel(r"Fraction of $\Xi$ (%)", labelpad=7)
    ax_d.grid(True, axis="y", color="0.90", lw=0.5)
    for index, value in enumerate(means):
        ax_d.text(index, value + max(1.0, 0.03 * means.max()), f"{value:.1f}%", ha="center", va="bottom")
    panel_label(ax_d, "D")

    within_network = np.asarray(yeo["within_group_by_network"], dtype=float)
    within_network_by_seed = within_network.mean(axis=1)
    network_values = within_network_by_seed.mean(axis=0)
    network_errors = sd(within_network_by_seed, axis=0)
    network_names = [str(value) for value in yeo["network_names"]]
    network_sizes = np.asarray(yeo["network_sizes"], dtype=int)
    order = np.argsort(network_values)
    ypos = np.arange(len(order))
    ax_e.barh(
        ypos, network_values[order], xerr=network_errors[order],
        color=[network_color(network_names[index]) for index in order], capsize=2,
    )
    short_network = {
        "Visual": "Vis",
        "Somatomotor": "SomMot",
        "Dorsal attention": "DorsAttn",
        "Salience / ventral attention": "Sal/Vent",
        "Frontoparietal control": "Control",
        "Default mode": "Default",
    }
    ax_e.set_yticks(
        ypos,
        [f"{short_network.get(network_names[index], network_names[index])}/{network_sizes[index]}" for index in order],
        fontsize=5.8,
    )
    ax_e.tick_params(axis="y", pad=1)
    ax_e.set_xlabel(r"Within-network cross-ROI $\Xi$ (bits)")
    ax_e.grid(True, axis="x", color="0.90", lw=0.5)
    panel_label(ax_e, "E")

    between_shapley = np.asarray(yeo["between_group_shapley"], dtype=float)
    between_shapley_by_seed = between_shapley.mean(axis=1)
    shapley_values = between_shapley_by_seed.mean(axis=0)
    shapley_errors = sd(between_shapley_by_seed, axis=0)
    ax_f.barh(
        ypos, shapley_values[order], xerr=shapley_errors[order],
        color=[network_color(network_names[index]) for index in order], capsize=2,
    )
    ax_f.set_yticks(
        ypos,
        [f"{short_network.get(network_names[index], network_names[index])}/{network_sizes[index]}" for index in order],
        fontsize=5.8,
    )
    ax_f.tick_params(axis="y", pad=1)
    ax_f.set_xlabel(r"Between-network Shapley $\Xi$ (bits)")
    ax_f.grid(True, axis="x", color="0.90", lw=0.5)
    panel_label(ax_f, "F")

    left_mesh, right_mesh, left_values, right_values = load_schaefer100_surface_map(
        args.surface_asset,
        np.asarray(topology["region_labels"]),
        cross,
    )
    draw_brain_map_four_views(
        ax_g,
        ax_g_cb,
        left_mesh,
        right_mesh,
        left_values,
        right_values,
        cmap="viridis",
        colorbar_label="Cross-ROI leverage (bits)",
        colorbar_label_size=6.2,
        zoom=1.10,
    )
    for axis, label in zip(ax_g, ("LH lateral", "RH lateral", "LH medial", "RH medial")):
        axis.text2D(0.03, 0.90, label, transform=axis.transAxes, fontsize=5.4, color="0.25")
    ax_g[0].text2D(
        -0.10, 1.00, "G", transform=ax_g[0].transAxes, fontsize=10, fontweight="bold"
    )
    figure.subplots_adjust(left=0.052, right=0.987, top=0.91, bottom=0.10, hspace=0.32)
    save(figure, args.output)

    appendix, appendix_axis = plt.subplots(
        1, 1, figsize=(3.55, 2.55), constrained_layout=True
    )
    for values, color, label in (
        (whole, "#0072B2", "Whole EI"),
        (singleton, "#D55E00", "Sum of singleton EI"),
    ):
        mean, error = np.mean(values, axis=0), sd(values)
        appendix_axis.plot(main["G"], mean, color=color, lw=1.45, label=label)
        appendix_axis.fill_between(
            main["G"], mean - error, mean + error, color=color, alpha=0.22, lw=0
        )
    critical_format(appendix_axis, critical_g)
    appendix_axis.set_ylabel("EI (bits)")
    appendix_axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=2, fontsize=6.3
    )
    save(
        appendix,
        args.output.parent / "dmf_schaefer100_ei_components_appendix",
    )


def spectral_radius(matrix: np.ndarray) -> float:
    symmetric = 0.5 * (matrix + matrix.T)
    return float(np.max(np.abs(np.linalg.eigvalsh(symmetric))))


def plot_comparison(args: argparse.Namespace) -> None:
    if args.comparison_output is None:
        return
    old_source, new_source = load(OLD_SOURCE), load(args.source)
    old_main, new_main = load(OLD_MAIN), load(args.main)
    old_topology, new_topology = load(OLD_TOPOLOGY), load(args.topology)
    old_phi, new_phi = direct_values(old_main, "phi_eid"), direct_values(new_main, "phi_eid")
    old_rho = spectral_radius(np.asarray(old_source["connectivity"], dtype=float))
    new_rho = spectral_radius(np.asarray(new_source["connectivity"], dtype=float))
    old_rate_x = np.asarray(old_source["G"], dtype=float) * old_rho
    new_rate_x = np.asarray(new_source["G"], dtype=float) * new_rho
    old_phi_x = np.asarray(old_main["G"], dtype=float) * old_rho
    new_phi_x = np.asarray(new_main["G"], dtype=float) * new_rho
    old_phi_normalized = old_phi / float(old_main["source_count"])
    new_phi_normalized = new_phi / float(new_main["source_count"])

    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    colors = {"83 ROI": "#7F7F7F", "100 ROI": "#6A3D9A"}
    axes[0].plot(old_rate_x, old_source["mean_rate_hz"], color=colors["83 ROI"], lw=1.35, label="83 ROI")
    axes[0].plot(new_rate_x, new_source["mean_rate_hz"], color=colors["100 ROI"], lw=1.55, label="100 ROI")
    axes[0].set_xlabel(r"Effective coupling $G\rho(\mathbf{C})$")
    axes[0].set_ylabel("Mean firing rate (Hz)")
    axes[0].grid(True, color="0.90", lw=0.5)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=2)

    for x, values, label in (
        (old_phi_x, old_phi_normalized, "83 ROI"),
        (new_phi_x, new_phi_normalized, "100 ROI"),
    ):
        mean, error = values.mean(axis=0), sem(values)
        axes[1].plot(x, mean, color=colors[label], lw=1.5, label=label)
        axes[1].fill_between(x, mean - error, mean + error, color=colors[label], alpha=0.16, lw=0)
    axes[1].set_xlabel(r"Effective coupling $G\rho(\mathbf{C})$")
    axes[1].set_ylabel(r"$\Xi$ per source (bits)")
    axes[1].grid(True, color="0.90", lw=0.5)
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=2)

    old_fraction = 100.0 * np.asarray(old_topology["between_roi"], dtype=float) / np.asarray(old_topology["fine_phi"], dtype=float)
    new_fraction = 100.0 * np.asarray(new_topology["between_roi"], dtype=float) / np.asarray(new_topology["fine_phi"], dtype=float)
    fraction_values = (old_fraction.ravel(), new_fraction.ravel())
    axes[2].bar(
        [0, 1], [values.mean() for values in fraction_values],
        yerr=[sem(values, axis=0) for values in fraction_values],
        color=(colors["83 ROI"], colors["100 ROI"]), width=0.68, capsize=2,
    )
    axes[2].set_xticks([0, 1], ["83 ROI", "100 ROI"])
    axes[2].set_ylabel(r"Cross-ROI fraction of $\Xi$ (%)")
    axes[2].grid(True, axis="y", color="0.90", lw=0.5)
    for panel, axis in zip("ABC", axes):
        panel_label(axis, panel)
    save(figure, args.comparison_output)

    old_peak = old_phi_x[np.argmax(old_phi_normalized, axis=1)]
    new_peak = new_phi_x[np.argmax(new_phi_normalized, axis=1)]
    summary = {
        "comparison_type": "cross-connectome replication; not a single-factor causal contrast",
        "normalization": {
            "x_axis": "G multiplied by the spectral radius of each connectivity matrix",
            "phi_axis": "Xi divided by the number of scalar source variables",
        },
        "spectral_radius": {"old83": old_rho, "new100": new_rho},
        "paired_phi_peak_effective_coupling": {
            "old83": old_peak.tolist(),
            "new100": new_peak.tolist(),
            "paired_delta_mean": float(np.mean(new_peak - old_peak)),
            "paired_delta_sem": float(sem(new_peak - old_peak, axis=0)),
            "new_exceeds_old_seed_count": int(np.sum(new_peak > old_peak)),
        },
        "cross_roi_fraction_percent": {
            "old83_mean": float(old_fraction.mean()),
            "new100_mean": float(new_fraction.mean()),
        },
    }
    args.comparison_output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    configure()
    plot_summary(args)
    plot_comparison(args)


if __name__ == "__main__":
    main()
