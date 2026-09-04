#!/usr/bin/env python3
"""Combine Earth SLP hyperedge maps and compressed SPTs by forecast horizon."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_runge_slp_pc60_xi_hierarchy import (
    SPLIT_COLOR,
    _blend_with_white,
    flatten_nodes,
)
from scripts.compare_runge_slp_pc60_xi_horizons import _node_from_record
from scripts.plot_earth_spt_chain_compressed_horizons import (
    RESULT_ROOT,
    _display_syn,
    _draw_panel,
)
from scripts.plot_earth_system_main_figures import (
    RUNGE_RESULT_DIR,
    configure_matplotlib,
    draw_compact_runge_map,
)
from scripts.plot_runge_exhaustive_tm_maps import (
    DEFAULT_COMPONENT_MAPS,
    load_exhaustive_top10,
    load_nodes,
)
from scripts.plot_runge_gateway_mediator_map import (
    COASTLINE_URL,
    LAND_URL,
    extract_lines,
    extract_polygons,
    load_geojson,
)


HORIZONS = (1, 10, 60)
OUTPUT = ROOT / "fig/earth_slp_hyperedge_spt_horizons.png"
INK = "#172033"


def _load_trees() -> tuple[list[object], float]:
    roots = []
    for horizon in HORIZONS:
        payload = json.loads(
            (RESULT_ROOT / f"H{horizon:03d}/summary.json").read_text(encoding="utf-8")
        )
        roots.append(_node_from_record(payload["tree"]))
    maximum = max(
        _display_syn(node.syn_bits)
        for root in roots
        for node in flatten_nodes(root)
        if node.children
    )
    return roots, maximum


def main() -> None:
    configure_matplotlib()
    mpl.rcParams.update({"font.size": 7, "savefig.facecolor": "white"})

    nodes = load_nodes(DEFAULT_COMPONENT_MAPS)
    frames = {
        horizon: load_exhaustive_top10(RUNGE_RESULT_DIR, horizon=horizon)
        for horizon in HORIZONS
    }
    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))
    if not land or not coastlines:
        raise RuntimeError("Natural Earth map outlines could not be loaded.")
    roots, maximum = _load_trees()

    figure = plt.figure(figsize=(13.2, 5.9), layout="constrained", facecolor="white")
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.045),
        height_ratios=(1.25, 2.05),
        wspace=0.025,
        hspace=0.015,
    )
    map_axes = [
        figure.add_subplot(grid[0, column], projection="mollweide")
        for column in range(3)
    ]
    tree_axes = [figure.add_subplot(grid[1, column]) for column in range(3)]
    colorbar_axis = figure.add_subplot(grid[1, 3])
    figure.add_subplot(grid[0, 3]).axis("off")

    for column, (horizon, frame, axis) in enumerate(
        zip(HORIZONS, frames.values(), map_axes, strict=True)
    ):
        draw_compact_runge_map(
            axis,
            nodes,
            frame,
            land,
            coastlines,
            horizon,
            scale=1.32,
        )
        axis.text(
            -0.035,
            1.04,
            "abc"[column],
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#111111",
            clip_on=False,
        )

    for panel, horizon, root, axis in zip(
        "def", HORIZONS, roots, tree_axes, strict=True
    ):
        _draw_panel(
            axis,
            root,
            horizon,
            maximum,
            panel,
            show_horizon=False,
        )

    color_map = mpl.colors.LinearSegmentedColormap.from_list(
        "syn_strength",
        (
            _blend_with_white(SPLIT_COLOR, 0.14),
            _blend_with_white(SPLIT_COLOR, 0.86),
        ),
    )
    color_norm = mpl.colors.Normalize(vmin=0.0, vmax=maximum)
    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=color_norm, cmap=color_map),
        cax=colorbar_axis,
        ticks=(0.0, 0.1, 0.2, maximum),
    )
    colorbar.set_label("Syn (bits)", fontsize=7.2, color=INK, labelpad=7)
    colorbar.ax.tick_params(labelsize=6.5, width=0.6, length=2.5, colors=INK)
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.set_yticklabels(("0.00", "0.10", "0.20", f"{maximum:.2f}"))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
