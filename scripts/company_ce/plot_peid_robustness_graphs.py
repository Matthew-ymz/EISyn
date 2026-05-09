#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd


DEFAULT_RESULTS_ROOT = Path("results/company_ce/csv/peid_sensitivity_main")
DEFAULT_FIGURE_DIR = Path("fig/company_ce/peid_sensitivity_main")
VARIABLES = ("inf_at", "inf_revt", "emp", "inf_lt", "inf_ch", "inf_ni", "inf_cogs")
RUN_ORDER = (
    ("baseline", "baseline"),
    ("bins_3", "bins=3"),
    ("bins_4", "bins=4"),
    ("bins_6", "bins=6"),
    ("min_source_count_10", "support=10"),
    ("min_source_count_50", "support=50"),
    ("alpha_0.1", "alpha=0.1"),
    ("alpha_1", "alpha=1.0"),
    ("winsor_0p005_0p995", "winsor 0.5%"),
    ("winsor_0p025_0p975", "winsor 2.5%"),
    ("years_early", "early years"),
    ("years_late", "late years"),
)
LABELS = {
    "inf_at": "Assets",
    "inf_revt": "Revenue",
    "emp": "Employees",
    "inf_lt": "Liabilities",
    "inf_ch": "Cash",
    "inf_ni": "Net income",
    "inf_cogs": "COGS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw PEID robustness graphs for each sensitivity parameter setting.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.unicode_minus": False,
        }
    )


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 450
        fig.savefig(figure_dir / f"{stem}.{suffix}", **kwargs)


def draw_directed_edge(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    width: float,
    alpha: float,
    baseline: bool,
    loop_side: int = 1,
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.hypot(dx, dy)
    shrink = 0.06
    start2 = (start[0] + shrink * dx / distance, start[1] + shrink * dy / distance)
    end2 = (end[0] - shrink * dx / distance, end[1] - shrink * dy / distance)
    rad = 0.08 * np.sign(dy) if abs(dy) > 0.02 else 0.0
    arrow = FancyArrowPatch(
        start2,
        end2,
        arrowstyle="-|>",
        mutation_scale=6.2,
        linewidth=width,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0,
        shrinkB=0,
        zorder=1 if baseline else 4,
    )
    ax.add_patch(arrow)


def read_pairwise(results_root: Path, run_id: str) -> pd.DataFrame:
    return pd.read_csv(results_root / run_id / "peid_pairwise_edges.csv")


def read_synergy(results_root: Path, run_id: str) -> pd.DataFrame:
    return pd.read_csv(results_root / run_id / "peid_synergy_hyperedges.csv")


def setup_panel(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=7.4, fontweight="bold", pad=2.5)
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(0.03, 0.98)
    ax.set_aspect("equal")
    ax.set_axis_off()


def heatmap_panel(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    xlabels: list[str],
    ylabels: list[str],
    show_xlabels: bool,
    show_ylabels: bool,
) -> mpl.image.AxesImage:
    image = ax.imshow(values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=7.2, fontweight="bold", pad=2.0)
    ax.set_xticks(range(len(xlabels)))
    ax.set_yticks(range(len(ylabels)))
    if show_xlabels:
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=4.9)
    else:
        ax.set_xticklabels([])
    if show_ylabels:
        ax.set_yticklabels(ylabels, fontsize=4.9)
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0, pad=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    return image


def plot_pairwise_heatmap_grid(results_root: Path, figure_dir: Path) -> None:
    matrices: dict[str, np.ndarray] = {}
    labels = [LABELS.get(variable, variable) for variable in VARIABLES]
    for run_id, _ in RUN_ORDER:
        frame = read_pairwise(results_root, run_id)
        matrix = frame.pivot(index="source", columns="target", values="ei").reindex(index=VARIABLES, columns=VARIABLES)
        matrices[run_id] = matrix.to_numpy(dtype=float)
    vmax = max(float(np.nanmax(values)) for values in matrices.values())

    fig, axes = plt.subplots(3, 4, figsize=(183 / 25.4, 128 / 25.4), constrained_layout=True)
    last_image = None
    for idx, (ax, (run_id, title)) in enumerate(zip(axes.flat, RUN_ORDER)):
        row, col = divmod(idx, 4)
        last_image = heatmap_panel(
            ax,
            matrices[run_id],
            title=title,
            vmin=0.0,
            vmax=vmax,
            cmap="Blues",
            xlabels=labels,
            ylabels=labels,
            show_xlabels=row == 2,
            show_ylabels=col == 0,
        )
    fig.suptitle("Pairwise EI heatmaps across robustness settings", fontsize=8.2, fontweight="bold")
    cax = inset_axes(axes[0, -1], width="4%", height="92%", loc="center right", bbox_to_anchor=(0.12, 0, 1, 1), bbox_transform=axes[0, -1].transAxes, borderpad=0)
    cbar = fig.colorbar(last_image, cax=cax)
    cbar.ax.set_ylabel("EI (bits)", fontsize=5.5)
    cbar.ax.tick_params(labelsize=5.0, length=2)
    save_figure(fig, figure_dir, "peid_robustness_pairwise_heatmap_grid")
    plt.close(fig)


def synergy_matrix(frame: pd.DataFrame, source_sets: list[str]) -> np.ndarray:
    matrix = frame.pivot(index="sources", columns="target", values="synergy_raw").reindex(index=source_sets, columns=VARIABLES)
    return matrix.fillna(0.0).to_numpy(dtype=float)


def ordered_synergy_source_sets(results_root: Path, top_k: int) -> list[str]:
    baseline = read_synergy(results_root, "baseline").sort_values("synergy_raw", ascending=False)
    ordered = list(dict.fromkeys(baseline.head(top_k)["sources"]))
    for run_id, _ in RUN_ORDER:
        frame = read_synergy(results_root, run_id).sort_values("synergy_raw", ascending=False).head(top_k)
        for source in frame["sources"]:
            if source not in ordered:
                ordered.append(source)
    return ordered


def plot_synergy_heatmap_grid(results_root: Path, figure_dir: Path, top_k: int) -> None:
    source_sets = ordered_synergy_source_sets(results_root, top_k)
    ylabels = [source_set_label(source) for source in source_sets]
    xlabels = [LABELS.get(variable, variable) for variable in VARIABLES]
    matrices = {run_id: synergy_matrix(read_synergy(results_root, run_id), source_sets) for run_id, _ in RUN_ORDER}
    vmax = max(float(np.nanmax(values)) for values in matrices.values())

    fig, axes = plt.subplots(3, 4, figsize=(183 / 25.4, 150 / 25.4), constrained_layout=True)
    last_image = None
    for idx, (ax, (run_id, title)) in enumerate(zip(axes.flat, RUN_ORDER)):
        row, col = divmod(idx, 4)
        last_image = heatmap_panel(
            ax,
            matrices[run_id],
            title=title,
            vmin=0.0,
            vmax=vmax,
            cmap="Oranges",
            xlabels=xlabels,
            ylabels=ylabels,
            show_xlabels=row == 2,
            show_ylabels=col == 0,
        )
    fig.suptitle("PEID synergy heatmaps across robustness settings", fontsize=8.2, fontweight="bold")
    cax = inset_axes(axes[0, -1], width="4%", height="92%", loc="center right", bbox_to_anchor=(0.12, 0, 1, 1), bbox_transform=axes[0, -1].transAxes, borderpad=0)
    cbar = fig.colorbar(last_image, cax=cax)
    cbar.ax.set_ylabel("synergy_raw (bits)", fontsize=5.5)
    cbar.ax.tick_params(labelsize=5.0, length=2)
    save_figure(fig, figure_dir, "peid_robustness_synergy_heatmap_grid")
    plt.close(fig)


def plot_pairwise_grid(results_root: Path, figure_dir: Path, top_k: int) -> None:
    baseline = read_pairwise(results_root, "baseline").sort_values("ei", ascending=False).head(top_k)
    max_value = max(read_pairwise(results_root, run_id)["ei"].head(top_k).max() for run_id, _ in RUN_ORDER if (results_root / run_id).exists())
    y_positions = {variable: 0.86 - idx * (0.72 / (len(VARIABLES) - 1)) for idx, variable in enumerate(VARIABLES)}

    fig, axes = plt.subplots(3, 4, figsize=(183 / 25.4, 128 / 25.4), constrained_layout=True)
    for ax, (run_id, title) in zip(axes.flat, RUN_ORDER):
        setup_panel(ax, title)
        for variable, y in y_positions.items():
            ax.text(
                0.06,
                y,
                LABELS.get(variable, variable),
                ha="left",
                va="center",
                fontsize=4.7,
                color="#2f3a4a",
                bbox=dict(boxstyle="round,pad=0.12,rounding_size=0.02", fc="#f8fafc", ec="#8a98a6", lw=0.36),
                zorder=5,
            )
            ax.text(
                0.88,
                y,
                LABELS.get(variable, variable),
                ha="center",
                va="center",
                fontsize=4.7,
                color="#3f2a12",
                bbox=dict(boxstyle="round,pad=0.12,rounding_size=0.02", fc="#fff8ef", ec="#9a6b2f", lw=0.36),
                zorder=5,
            )
        for row in baseline.itertuples():
            width = 0.35 + 1.2 * float(row.ei) / max_value
            draw_directed_edge(
                ax,
                (0.28, y_positions[row.source]),
                (0.73, y_positions[row.target]),
                color="#c8cdd4",
                width=width,
                alpha=0.34,
                baseline=True,
            )
        frame = read_pairwise(results_root, run_id).sort_values("ei", ascending=False).head(top_k)
        base_keys = set(zip(baseline["source"], baseline["target"]))
        for row in frame.itertuples():
            key = (row.source, row.target)
            color = "#34699A" if key in base_keys else "#C46A32"
            width = 0.55 + 1.55 * float(row.ei) / max_value
            draw_directed_edge(
                ax,
                (0.28, y_positions[row.source]),
                (0.73, y_positions[row.target]),
                color=color,
                width=width,
                alpha=0.86,
                baseline=False,
            )
    fig.suptitle("Pairwise EI causal graphs across robustness settings", fontsize=8.2, fontweight="bold")
    save_figure(fig, figure_dir, "peid_robustness_pairwise_graph_grid")
    plt.close(fig)


def source_set_label(source_set: str) -> str:
    return " + ".join(LABELS.get(part, part) for part in source_set.split("+"))


def draw_hyperedge_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    baseline_keys: set[tuple[str, str]],
    max_value: float,
) -> None:
    top = frame.sort_values("synergy_raw", ascending=False).head(12).copy()
    source_sets = list(dict.fromkeys(top["sources"]))
    targets = list(dict.fromkeys(top["target"]))
    source_y = {source: 0.88 - idx * (0.76 / max(1, len(source_sets) - 1)) for idx, source in enumerate(source_sets)}
    target_y = {target: 0.84 - idx * (0.64 / max(1, len(targets) - 1)) for idx, target in enumerate(targets)}

    for source in source_sets:
        y = source_y[source]
        ax.text(
            0.07,
            y,
            source_set_label(source),
            ha="left",
            va="center",
            fontsize=4.7,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.03", fc="#f7f9fb", ec="#7f8c99", lw=0.42),
            zorder=5,
        )
    for target in targets:
        y = target_y[target]
        ax.text(
            0.88,
            y,
            LABELS.get(target, target),
            ha="center",
            va="center",
            fontsize=4.9,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.03", fc="#fff7ed", ec="#9a6b2f", lw=0.42),
            zorder=5,
        )
    for row in top.itertuples():
        key = (row.sources, row.target)
        color = "#D27730" if key in baseline_keys else "#8E5B9A"
        y0 = source_y[row.sources]
        y1 = target_y[row.target]
        width = 0.45 + 1.7 * float(row.synergy_raw) / max_value
        arrow = FancyArrowPatch(
            (0.36, y0),
            (0.75, y1),
            arrowstyle="-|>",
            mutation_scale=5.8,
            linewidth=width,
            color=color,
            alpha=0.84,
            connectionstyle=f"arc3,rad={0.12 if y1 > y0 else -0.12}",
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
        ax.add_patch(arrow)


def plot_synergy_grid(results_root: Path, figure_dir: Path, top_k: int) -> None:
    baseline = read_synergy(results_root, "baseline").sort_values("synergy_raw", ascending=False).head(top_k)
    baseline_keys = set(zip(baseline["sources"], baseline["target"]))
    max_value = max(read_synergy(results_root, run_id)["synergy_raw"].head(top_k).max() for run_id, _ in RUN_ORDER if (results_root / run_id).exists())

    fig, axes = plt.subplots(3, 4, figsize=(183 / 25.4, 146 / 25.4), constrained_layout=True)
    for ax, (run_id, title) in zip(axes.flat, RUN_ORDER):
        setup_panel(ax, title)
        frame = read_synergy(results_root, run_id)
        draw_hyperedge_panel(ax, frame, baseline_keys, max_value)
    fig.suptitle("PEID synergy hypergraphs across robustness settings", fontsize=8.2, fontweight="bold")
    save_figure(fig, figure_dir, "peid_robustness_synergy_hypergraph_grid")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()
    plot_pairwise_heatmap_grid(args.results_root, args.figure_dir)
    plot_synergy_heatmap_grid(args.results_root, args.figure_dir, args.top_k)
    plot_pairwise_grid(args.results_root, args.figure_dir, args.top_k)
    plot_synergy_grid(args.results_root, args.figure_dir, args.top_k)


if __name__ == "__main__":
    main()
