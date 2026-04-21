from __future__ import annotations

import html
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
    if (candidate / "utils.py").exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        break

from utils import (
    _normalize_topology_connection_style,
    _svg_marker_def,
    build_probabilistic_boolean_tpm,
    joint_ei_decomposition,
    render_topology_mechanism_svg,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

N_NODES = 8
PANEL_ORDER = ("A", "B", "C", "D", "E", "F")
PANEL_POSITIONS = {
    "A": (20, 20),
    "B": (460, 20),
    "C": (900, 20),
    "D": (20, 340),
    "E": (460, 340),
    "F": (900, 340),
}
PANEL_LOWERCASE = dict(zip(PANEL_ORDER, "abcdef", strict=True))


def fully_connected() -> np.ndarray:
    return np.ones((N_NODES, N_NODES), dtype=float) - np.eye(N_NODES)


def bidirectional_ring() -> np.ndarray:
    adjacency = np.zeros((N_NODES, N_NODES), dtype=float)
    for i in range(N_NODES):
        adjacency[(i - 1) % N_NODES, i] = 1.0
        adjacency[(i + 1) % N_NODES, i] = 1.0
    return adjacency


def bidirectional_skip_ring() -> np.ndarray:
    adjacency = np.zeros((N_NODES, N_NODES), dtype=float)
    for i in range(N_NODES):
        adjacency[(i - 2) % N_NODES, i] = 1.0
        adjacency[(i + 2) % N_NODES, i] = 1.0
    return adjacency


def unidirectional_ring() -> np.ndarray:
    adjacency = np.zeros((N_NODES, N_NODES), dtype=float)
    for i in range(N_NODES):
        adjacency[(i - 1) % N_NODES, i] = 1.0
    return adjacency


def empty_topology() -> np.ndarray:
    return np.zeros((N_NODES, N_NODES), dtype=float)


def dense_two_community_weak_bridge() -> np.ndarray:
    adjacency = np.zeros((N_NODES, N_NODES), dtype=float)
    weighted_edges = {
        (0, 1): 1.0,
        (1, 2): 1.0,
        (2, 3): 1.0,
        (0, 2): 0.9,
        (1, 3): 0.9,
        (4, 5): 1.0,
        (5, 6): 1.0,
        (6, 7): 1.0,
        (4, 6): 0.9,
        (5, 7): 0.9,
        (3, 4): 0.25,
        (7, 0): 0.25,
    }
    for (src, dst), weight in weighted_edges.items():
        adjacency[src, dst] = weight
    return adjacency


def phi_optimal_binary_like() -> np.ndarray:
    adjacency = np.zeros((N_NODES, N_NODES), dtype=float)
    edges = [
        (0, 1),
        (0, 3),
        (1, 2),
        (1, 4),
        (2, 5),
        (3, 4),
        (4, 6),
        (5, 7),
        (6, 0),
        (7, 2),
        (6, 3),
        (2, 6),
    ]
    for src, dst in edges:
        adjacency[src, dst] = 1.0
    return adjacency


def phi_optimal_weighted_like() -> np.ndarray:
    adjacency = np.zeros((N_NODES, N_NODES), dtype=float)
    weights = {
        (0, 1): 1.2,
        (0, 3): 0.9,
        (1, 2): 1.1,
        (1, 4): 0.8,
        (2, 5): 1.4,
        (3, 4): 1.0,
        (4, 6): 1.3,
        (5, 7): 1.1,
        (6, 0): 0.7,
        (7, 2): 1.2,
        (6, 3): 1.5,
        (2, 6): 1.0,
    }
    for (src, dst), weight in weights.items():
        adjacency[src, dst] = weight
    return adjacency


def benchmark_node_specs(name: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "bias": 0.0,
            "alpha": 0.6,
            "beta": 0.0,
            "gamma": 0.0,
            "coop_sources": [],
            "parity_sources": [],
        }
        for _ in range(N_NODES)
    ]

    if name == "A":
        for spec in specs:
            spec["alpha"] = 0.16
            spec["beta"] = 0.05
    elif name == "B":
        for spec in specs:
            spec["alpha"] = 0.40
            spec["beta"] = 0.08
        specs[2]["coop_sources"] = ["ALL_INPUTS"]
        specs[4]["coop_sources"] = ["ALL_INPUTS"]
    elif name == "C":
        for spec in specs:
            spec["alpha"] = 0.62
            spec["beta"] = 1.05
            spec["gamma"] = 0.95
        specs[2]["coop_sources"] = ["ALL_INPUTS"]
        specs[6]["coop_sources"] = ["ALL_INPUTS"]
        specs[6]["beta"] = 1.26
        specs[3]["parity_sources"] = [0, 6]
        specs[7]["parity_sources"] = [2, 5]
    elif name == "D":
        for spec in specs:
            spec["alpha"] = 0.65
            spec["beta"] = 0.12
        for j in range(N_NODES):
            specs[j]["coop_sources"] = ["ALL_INPUTS"]
    elif name == "E":
        for spec in specs:
            spec["alpha"] = 0.8
            spec["beta"] = 0.3
            spec["gamma"] = 0.2
        specs[0]["parity_sources"] = [1, 3]
        specs[2]["coop_sources"] = ["ALL_INPUTS"]
        specs[3]["coop_sources"] = ["ALL_INPUTS"]
        specs[6]["coop_sources"] = ["ALL_INPUTS"]
        specs[7]["coop_sources"] = ["ALL_INPUTS"]
        specs[5]["parity_sources"] = [4, 7]
        specs[3]["parity_sources"] = [1, 2]
        specs[7]["parity_sources"] = [5, 6]
    elif name == "F":
        for spec in specs:
            spec["alpha"] = 0.0
            spec["beta"] = 0.0
            spec["gamma"] = 0.85
        for j in range(N_NODES):
            specs[j]["parity_sources"] = [(j - 2) % N_NODES, (j - 3) % N_NODES]
    else:
        raise ValueError(f"Unknown topology: {name}")

    return specs


def build_benchmark_networks() -> dict[str, dict[str, Any]]:
    topologies = {
        "A": ("fully connected", fully_connected()),
        "B": ("binary-like optimal", phi_optimal_binary_like()),
        "C": ("weighted-like optimal", phi_optimal_weighted_like()),
        "D": ("bidirectional skip ring", bidirectional_skip_ring()),
        "E": ("dense two-community weak bridge", dense_two_community_weak_bridge()),
        "F": ("parity-only cycle", empty_topology()),
    }
    return {
        name: {
            "label": label,
            "adjacency": adjacency,
            "node_specs": _finalize_benchmark_node_specs(adjacency, benchmark_node_specs(name)),
        }
        for name, (label, adjacency) in topologies.items()
    }


def _finalize_benchmark_node_specs(
    adjacency: np.ndarray,
    node_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    specs = [dict(spec) for spec in node_specs]
    for node, spec in enumerate(specs):
        coop_sources = spec.get("coop_sources", [])
        if coop_sources == ["ALL_INPUTS"]:
            spec["coop_sources"] = [int(src) for src in np.flatnonzero(adjacency[:, node] > 0)]
        for key in ("coop_sources", "parity_sources"):
            sources = tuple(int(src) for src in spec.get(key, ()))
            spec[key] = list(sources) if len(sources) >= 2 else []
    return specs


BENCHMARK_NETWORKS = build_benchmark_networks()


def compute_benchmark_results(
    benchmark_networks: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, float | str]]:
    networks = benchmark_networks or BENCHMARK_NETWORKS
    results: dict[str, dict[str, float | str]] = {}
    for name in PANEL_ORDER:
        network = networks[name]
        summary = joint_ei_decomposition(
            build_probabilistic_boolean_tpm(network["adjacency"], network["node_specs"]),
            n_nodes=N_NODES,
        )
        results[name] = {
            "label": str(network["label"]),
            "ei": float(summary["ei_full"]),
            "phi_eid": float(summary["syn_high"]),
        }
    return results


def _strip_svg_wrapper(svg_text: str) -> str:
    match = re.match(r"<svg[^>]*>(.*)</svg>\s*$", svg_text, re.DOTALL)
    if match is None:
        raise ValueError("Unexpected SVG wrapper format")
    return match.group(1)


def _svg_chart_title(title: str, center_x: float, baseline_y: float) -> str:
    normalized = title.strip()
    if "Phi" in normalized and "EID" in normalized or normalized == "Φ^EID":
        return (
            f"<text x='{center_x:.1f}' y='{baseline_y:.1f}' text-anchor='middle' font-size='16'>"
            "<tspan>Φ</tspan>"
            "<tspan baseline-shift='super' font-size='70%'>EID</tspan>"
            "</text>"
        )
    return (
        f"<text x='{center_x:.1f}' y='{baseline_y:.1f}' text-anchor='middle' font-size='16'>"
        f"{html.escape(title)}</text>"
    )


def render_benchmark_topology_overview_svg(
    benchmark_networks: dict[str, dict[str, Any]] | None = None,
    *,
    panel_width: int = 260,
    panel_height: int = 260,
    horizontal_gap: int = 18,
    vertical_gap: int = 22,
    outer_margin_x: int = 20,
    outer_margin_y: int = 20,
    show_panel_letters: bool = True,
    panel_letter_dx: float = 28.0,
    panel_letter_dy: float = 46.0,
    panel_letter_font_size: int = 40,
    panel_letter_fill: str = "#222",
    panel_letter_font_weight: str = "700",
    show_mechanism_legend: bool = True,
    mechanism_legend_x: float | None = None,
    mechanism_legend_y: float = 16.0,
    mechanism_legend_width: float = 180.0,
    mechanism_legend_height: float = 48.0,
    mechanism_legend_title: str | None = "Mechanism legend",
    mechanism_legend_title_font_size: float = 11.0,
    mechanism_legend_item_font_size: float = 9.5,
    mechanism_legend_item_spacing: float = 15.0,
    arrow_marker_width: float = 8.0,
    arrow_marker_height: float = 6.0,
    arrow_marker_ref_x: float = 7.0,
    arrow_marker_ref_y: float = 3.0,
    arrow_marker_fill: str = "#7d8a97",
    connection_style: dict[str, dict[str, Any]] | None = None,
    connection_filter_style: dict[str, Any] | None = None,
) -> str:
    networks = benchmark_networks or BENCHMARK_NETWORKS
    styles = _normalize_topology_connection_style(
        connection_style,
        fallback_marker_width=arrow_marker_width,
        fallback_marker_height=arrow_marker_height,
        fallback_marker_ref_x=arrow_marker_ref_x,
        fallback_marker_ref_y=arrow_marker_ref_y,
        fallback_marker_fill=arrow_marker_fill,
    )
    copy_style = styles["copy"]
    cooperation_style = styles["cooperation"]
    parity_style = styles["parity"]
    base_width = outer_margin_x * 2 + panel_width * 3 + horizontal_gap * 2
    base_height = outer_margin_y * 2 + panel_height * 2 + vertical_gap
    legend_x = (
        mechanism_legend_x
        if mechanism_legend_x is not None
        else base_width - mechanism_legend_width - 30.0
    )
    overview_width = max(base_width, int(math.ceil(legend_x + mechanism_legend_width + outer_margin_x)))
    overview_height = max(
        base_height,
        int(math.ceil(mechanism_legend_y + mechanism_legend_height + outer_margin_y)),
    )
    pieces = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{overview_width}' height='{overview_height}' viewBox='0 0 {overview_width} {overview_height}'>",
        "<defs>"
        + _svg_marker_def("copy_arrowhead", copy_style)
        + _svg_marker_def("cooperation_arrowhead", cooperation_style)
        + _svg_marker_def("parity_arrowhead", parity_style)
        + "</defs>",
        f"<rect x='0' y='0' width='{overview_width}' height='{overview_height}' fill='white'/>",
    ]
    if show_mechanism_legend:
        legend_title = (
            mechanism_legend_title.strip()
            if isinstance(mechanism_legend_title, str)
            else ""
        )
        has_legend_title = bool(legend_title)
        item_baseline_start = mechanism_legend_y + (30.0 if has_legend_title else 26.0)
        legend_item_y = {
            "copy": item_baseline_start,
            "cooperation": item_baseline_start + mechanism_legend_item_spacing,
            "parity": item_baseline_start + 2.0 * mechanism_legend_item_spacing,
        }
        pieces.append(
            f"<rect x='{legend_x:.1f}' y='{mechanism_legend_y:.1f}' width='{mechanism_legend_width:.1f}' height='{mechanism_legend_height:.1f}' fill='white' stroke='#d4dbe3' rx='8' ry='8'/>"
        )
        if has_legend_title:
            pieces.append(
                f"<text x='{legend_x + 12:.1f}' y='{mechanism_legend_y + 16:.1f}' font-size='{mechanism_legend_title_font_size:.1f}' font-weight='700' fill='#223'>{html.escape(legend_title)}</text>"
            )
        pieces.append(
            f"<line x1='{legend_x + 12:.1f}' y1='{legend_item_y['copy'] - 3.0:.1f}' x2='{legend_x + 56:.1f}' y2='{legend_item_y['copy'] - 3.0:.1f}' "
            f"stroke='{html.escape(str(copy_style['stroke']))}' stroke-width='{float(copy_style['stroke_width_base']) + float(copy_style['stroke_width_scale']):.2f}' "
            f"opacity='{float(copy_style['opacity']):.2f}' marker-end='url(#copy_arrowhead)'/>"
        )
        pieces.append(
            f"<text x='{legend_x + 66:.1f}' y='{legend_item_y['copy']:.1f}' font-size='{mechanism_legend_item_font_size:.1f}' fill='#223'>copy</text>"
        )
        pieces.append(
            f"<path d='M {legend_x + 12:.1f},{legend_item_y['cooperation'] - 3.0:.1f} Q {legend_x + 34:.1f},{legend_item_y['cooperation'] - 12.0:.1f} {legend_x + 56:.1f},{legend_item_y['cooperation'] - 3.0:.1f}' "
            f"fill='none' stroke='{html.escape(str(cooperation_style['stroke']))}' stroke-width='{float(cooperation_style['stroke_width']):.2f}' "
            f"stroke-dasharray='{html.escape(str(cooperation_style['dash']))}' marker-end='url(#cooperation_arrowhead)' opacity='{float(cooperation_style['opacity']):.2f}'/>"
        )
        pieces.append(
            f"<text x='{legend_x + 66:.1f}' y='{legend_item_y['cooperation']:.1f}' font-size='{mechanism_legend_item_font_size:.1f}' fill='#223'>cooperation</text>"
        )
        pieces.append(
            f"<path d='M {legend_x + 12:.1f},{legend_item_y['parity'] - 3.0:.1f} Q {legend_x + 34:.1f},{legend_item_y['parity'] + 6.0:.1f} {legend_x + 56:.1f},{legend_item_y['parity'] - 3.0:.1f}' "
            f"fill='none' stroke='{html.escape(str(parity_style['stroke']))}' stroke-width='{float(parity_style['stroke_width']):.2f}' "
            f"stroke-dasharray='{html.escape(str(parity_style['dash']))}' stroke-linecap='{html.escape(str(parity_style.get('linecap', 'round')))}' "
            f"marker-end='url(#parity_arrowhead)' opacity='{float(parity_style['opacity']):.2f}'/>"
        )
        pieces.append(
            f"<text x='{legend_x + 66:.1f}' y='{legend_item_y['parity']:.1f}' font-size='{mechanism_legend_item_font_size:.1f}' fill='#223'>parity</text>"
        )
    for name in PANEL_ORDER:
        index = PANEL_ORDER.index(name)
        row = index // 3
        col = index % 3
        x = outer_margin_x + col * (panel_width + horizontal_gap)
        y = outer_margin_y + row * (panel_height + vertical_gap)
        panel_svg = render_topology_mechanism_svg(
            name,
            str(networks[name]["label"]),
            networks[name]["adjacency"],
            networks[name]["node_specs"],
            width=panel_width,
            height=panel_height,
            show_summary_box=False,
            show_panel_title=False,
            show_mechanism_hint=False,
            show_mechanism_text_labels=False,
            arrow_marker_width=arrow_marker_width,
            arrow_marker_height=arrow_marker_height,
            arrow_marker_ref_x=arrow_marker_ref_x,
            arrow_marker_ref_y=arrow_marker_ref_y,
            arrow_marker_fill=arrow_marker_fill,
            uniform_node_fill="#ecd8cf",
            uniform_node_stroke="#2f3e4e",
            uniform_node_stroke_width=1.7,
            connection_style=connection_style,
            connection_filter_style=connection_filter_style,
        )
        if show_panel_letters:
            pieces.append(
                f"<text x='{x + panel_letter_dx:.0f}' y='{y + panel_letter_dy:.0f}' "
                f"font-size='{panel_letter_font_size}' font-weight='{html.escape(panel_letter_font_weight)}' "
                f"fill='{html.escape(panel_letter_fill)}'>{PANEL_LOWERCASE[name]}</text>"
            )
        pieces.append(f"<g transform='translate({x},{y})'>{_strip_svg_wrapper(panel_svg)}</g>")
    pieces.append("</svg>")
    return "".join(pieces)


def render_metric_bar_chart_svg(
    metric_key: str,
    title: str,
    results: dict[str, dict[str, float | str]] | None = None,
    color: str = "#4C78A8",
    y_label: str = "bit",
) -> str:
    benchmark_results = results or compute_benchmark_results()
    values = {name: float(benchmark_results[name][metric_key]) for name in PANEL_ORDER}
    width = 620
    height = 252
    margin_left = 68
    margin_bottom = 40
    margin_top = 42
    chart_width = width - margin_left - 20
    chart_height = height - margin_top - margin_bottom
    bar_width = chart_width / max(len(values), 1)
    max_value = max(max(values.values()), 1e-9)
    pieces = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>",
        f"<line x1='{margin_left}' y1='{margin_top + chart_height}' x2='{width - 15}' y2='{margin_top + chart_height}' stroke='black'/>",
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + chart_height}' stroke='black'/>",
        _svg_chart_title(title, width / 2, 24.0),
        f"<text x='34' y='{margin_top + chart_height / 2:.1f}' text-anchor='middle' font-size='12' transform='rotate(-90 34 {margin_top + chart_height / 2:.1f})'>{html.escape(y_label)}</text>",
    ]
    for index, (name, value) in enumerate(values.items()):
        bar_h = chart_height * (value / max_value)
        x = margin_left + index * bar_width + 8
        y = margin_top + chart_height - bar_h
        pieces.append(
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width - 16:.1f}' height='{bar_h:.1f}' fill='{color}' opacity='0.85'/>"
        )
        pieces.append(
            f"<text x='{x + (bar_width - 16) / 2:.1f}' y='{margin_top + chart_height + 16}' text-anchor='middle' font-size='12'>{PANEL_LOWERCASE[name]}</text>"
        )
        pieces.append(
            f"<text x='{x + (bar_width - 16) / 2:.1f}' y='{y - 6:.1f}' text-anchor='middle' font-size='11'>{value:.3f}</text>"
        )
    pieces.append("</svg>")
    return "".join(pieces)


def render_benchmark_ei_phi_summary_svg(
    results: dict[str, dict[str, float | str]] | None = None,
) -> str:
    benchmark_results = results or compute_benchmark_results()
    ei_svg = render_metric_bar_chart_svg("ei", "Total EI", benchmark_results, color="#1F77B4")
    phi_svg = render_metric_bar_chart_svg("phi_eid", "Φ^EID", benchmark_results, color="#E15759")
    width = 1240
    height = 280
    pieces = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>",
        "<g transform='translate(0,20)'>",
        _strip_svg_wrapper(ei_svg),
        "</g>",
        "<g transform='translate(620,20)'>",
        _strip_svg_wrapper(phi_svg),
        "</g>",
        "</svg>",
    ]
    return "".join(pieces)


def export_benchmark_artifacts(
    fig_dir: Path,
    benchmark_networks: dict[str, dict[str, Any]] | None = None,
    results: dict[str, dict[str, float | str]] | None = None,
    topology_overview_kwargs: dict[str, Any] | None = None,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    networks = benchmark_networks or BENCHMARK_NETWORKS
    benchmark_results = results or compute_benchmark_results(networks)
    topology_kwargs = topology_overview_kwargs or {}

    topology_path = fig_dir / "eight_node_topology_panels_clean_labeled.svg"
    ei_path = fig_dir / "joint_state_ei.svg"
    phi_path = fig_dir / "highest_order_total_synergy.svg"
    legacy_summary_path = fig_dir / "ei_phi_summary.svg"

    topology_path.write_text(render_benchmark_topology_overview_svg(networks, **topology_kwargs))
    ei_path.write_text(
        render_metric_bar_chart_svg("ei", "Total EI", benchmark_results, color="#1F77B4")
    )
    phi_path.write_text(
        render_metric_bar_chart_svg("phi_eid", "Φ^EID", benchmark_results, color="#E15759")
    )
    legacy_summary_path.write_text(render_benchmark_ei_phi_summary_svg(benchmark_results))

    manifest = [
        {
            "index": 0,
            "label": "Six benchmark topologies",
            "file": str(topology_path),
        },
        {
            "index": 1,
            "label": "Benchmark total EI",
            "file": str(ei_path),
        },
        {
            "index": 2,
            "label": "Benchmark Phi^EID",
            "file": str(phi_path),
        },
    ]
    (fig_dir / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
