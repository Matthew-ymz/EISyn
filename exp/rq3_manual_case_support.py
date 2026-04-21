from __future__ import annotations

import html
import io
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = Path(tempfile.gettempdir()) / "eisyn-matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "DejaVu Serif"

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from utils import (
    coarse_grain_binary_or_tpm,
    discrete_causal_graph,
    enumerate_binary_states,
    render_coarse_graining_comparison_png,
    render_coarse_graining_comparison_svg,
    search_binary_or_fixed_macro_dim_coarse_grainings,
)

MICRO_LABELS = ("a1", "a2", "b1", "b2", "c1", "c2")
MACRO_LABELS = ("A", "B", "C")
OPTIMAL_GROUPS = ((0, 1), (2, 3), (4, 5))
NONOPTIMAL_GROUPS = ((0, 1), (2, 4), (3, 5))
PAIRWISE_DISPLAY_THRESHOLD = 0.03
HYPEREDGE_DISPLAY_THRESHOLD = 0.10
DEFAULT_CACHE_VERSION = "rq3_manual_case_v4"

DEFAULT_CACHE_DIR = REPO_ROOT / "exp" / "cache" / "rq3_manual_case"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "rq3_manual_case"
DEFAULT_FIGURE_DIR = REPO_ROOT / "fig" / "rq3_boolean_causal_emergence"
DEFAULT_CACHE_PATH = DEFAULT_CACHE_DIR / "manual_case_summary.json"
DEFAULT_RESULTS_SUMMARY_PATH = DEFAULT_RESULTS_DIR / "manual_case_summary.json"
DEFAULT_RESULTS_METRICS_PATH = DEFAULT_RESULTS_DIR / "metrics_table.html"
FIGURE_PATHS = {
    "topology_svg": DEFAULT_FIGURE_DIR / "micro_topology_overview.svg",
    "topology_png": DEFAULT_FIGURE_DIR / "micro_topology_overview.png",
    "optimal_svg": DEFAULT_FIGURE_DIR / "micro_macro_coarse_graining_comparison.svg",
    "optimal_png": DEFAULT_FIGURE_DIR / "micro_macro_coarse_graining_comparison.png",
    "nonoptimal_svg": DEFAULT_FIGURE_DIR / "non_optimal_macro_comparison.svg",
    "nonoptimal_png": DEFAULT_FIGURE_DIR / "non_optimal_macro_comparison.png",
}


def clean_svg_text(svg_text: str) -> str:
    return re.sub(r"<metadata>.*?</metadata>\s*", "", svg_text, flags=re.DOTALL)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _tuple_groups(groups: list[list[int]] | tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(index) for index in block) for block in groups)


def _load_cached_summary(cache_path: Path) -> dict[str, Any]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    summary = dict(payload["summary"])
    summary["optimal_groups"] = _tuple_groups(summary["optimal_groups"])
    summary["nonoptimal_groups"] = _tuple_groups(summary["nonoptimal_groups"])
    return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_manual_micro_tpm() -> np.ndarray:
    states = enumerate_binary_states(len(MICRO_LABELS))
    state_to_index = {
        tuple(int(bit) for bit in state.tolist()): index
        for index, state in enumerate(states)
    }
    tpm = np.zeros((len(states), len(states)), dtype=float)

    for row_index, state in enumerate(states):
        a1, a2, b1, b2, c1, c2 = (int(value) for value in state.tolist())
        macro_a = int(a1 or a2)
        macro_b = int(b1 or b2)
        macro_c = int(c1 or c2)
        next_state = (
            int(macro_b and macro_c),
            int(macro_b and (not macro_c)),
            int(macro_c and macro_a),
            int(macro_c and (not macro_a)),
            int(macro_a and macro_b),
            int(macro_a and (not macro_b)),
        )
        tpm[row_index, state_to_index[next_state]] = 1.0

    return tpm


def _threshold_pairwise(pairwise_matrix: np.ndarray, threshold: float = PAIRWISE_DISPLAY_THRESHOLD) -> np.ndarray:
    pairwise = np.asarray(pairwise_matrix, dtype=float).copy()
    pairwise[np.abs(pairwise) <= threshold] = 0.0
    return pairwise


def _select_display_hyperedges(
    hyperedges: list[dict[str, Any]],
    *,
    min_value: float = HYPEREDGE_DISPLAY_THRESHOLD,
    max_per_target: int = 1,
) -> list[dict[str, Any]]:
    strongest_by_target: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in hyperedges:
        target = int(edge["target"])
        value = float(edge.get("value", edge.get("synergy", 0.0)))
        if value <= min_value:
            continue
        strongest_by_target[target].append(
            {
                "sources": tuple(int(source) for source in edge["sources"]),
                "target": target,
                "value": value,
                "label": "syn",
                "opacity": 0.90,
            }
        )

    selected: list[dict[str, Any]] = []
    for target in sorted(strongest_by_target):
        ranked = sorted(
            strongest_by_target[target],
            key=lambda row: (-float(row["value"]), tuple(row["sources"])),
        )
        selected.extend(ranked[:max_per_target])
    return selected


def _max_positive_hyperedge(hyperedges: list[dict[str, Any]]) -> float:
    if not hyperedges:
        return 0.0
    return float(
        max(max(float(edge.get("value", edge.get("synergy", 0.0))), 0.0) for edge in hyperedges)
    )


def _shorten_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    start_offset: float,
    end_offset: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1e-9)
    ux = dx / length
    uy = dy / length
    return (
        (start[0] + start_offset * ux, start[1] + start_offset * uy),
        (end[0] - end_offset * ux, end[1] - end_offset * uy),
    )


def _manual_topology_positions() -> dict[int, tuple[float, float]]:
    return {
        0: (235.2, 85.0),
        1: (326.5, 137.7),
        2: (326.5, 243.1),
        3: (235.2, 295.8),
        4: (143.9, 243.1),
        5: (143.9, 137.7),
    }


def _manual_topology_copy_edges() -> list[tuple[int, int]]:
    return [
        (2, 0),
        (3, 0),
        (2, 1),
        (3, 1),
        (4, 2),
        (5, 2),
        (4, 3),
        (5, 3),
        (0, 4),
        (1, 4),
        (0, 5),
        (1, 5),
    ]


def _manual_topology_hyperedges() -> list[tuple[tuple[int, int], int]]:
    return [
        ((4, 5), 0),
        ((4, 5), 1),
        ((0, 1), 2),
        ((0, 1), 3),
        ((2, 3), 4),
        ((2, 3), 5),
    ]


def build_manual_micro_mechanism_hyperedges() -> list[dict[str, Any]]:
    return [
        {"sources": (2, 3, 4, 5), "target": 0, "value": 1.0, "label": "syn", "opacity": 0.90},
        {"sources": (2, 3, 4, 5), "target": 1, "value": 0.5, "label": "syn", "opacity": 0.90},
        {"sources": (0, 1, 4, 5), "target": 2, "value": 1.0, "label": "syn", "opacity": 0.90},
        {"sources": (0, 1, 4, 5), "target": 3, "value": 0.5, "label": "syn", "opacity": 0.90},
        {"sources": (0, 1, 2, 3), "target": 4, "value": 1.0, "label": "syn", "opacity": 0.90},
        {"sources": (0, 1, 2, 3), "target": 5, "value": 0.5, "label": "syn", "opacity": 0.90},
    ]


def _svg_line(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    stroke: str,
    stroke_width: float,
    dash: str = "",
    opacity: float = 1.0,
    marker_end: str | None = None,
) -> str:
    marker_attr = f" marker-end='{marker_end}'" if marker_end else ""
    dash_attr = f" stroke-dasharray='{dash}'" if dash else ""
    return (
        f"<line x1='{start[0]:.1f}' y1='{start[1]:.1f}' "
        f"x2='{end[0]:.1f}' y2='{end[1]:.1f}' "
        f"stroke='{stroke}' stroke-width='{stroke_width:.1f}'{dash_attr} opacity='{opacity:.2f}'{marker_attr}/>"
    )


def render_manual_micro_topology_assets(
    *,
    svg_path: Path,
    png_path: Path,
    title: str = "",
    subtitle: str = "",
    width: int = 560,
    height: int = 340,
    dpi: int = 180,
) -> str:
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    positions = _manual_topology_positions()
    copy_edges = _manual_topology_copy_edges()
    hyperedges = _manual_topology_hyperedges()
    node_radius = 18.0

    pieces = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<defs><marker id='static-arrow' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto' markerUnits='strokeWidth'>"
        "<path d='M 0 0 L 8 3 L 0 6 z' fill='#6b7785'/></marker></defs>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>",
    ]
    if title:
        pieces.append(
            f"<text x='14' y='22' font-size='16' font-weight='700' fill='#111'>{html.escape(title)}</text>"
        )
    if subtitle:
        pieces.append(
            f"<text x='14' y='40' font-size='11' fill='#555'>{html.escape(subtitle)}</text>"
        )

    for source, target in copy_edges:
        start, end = _shorten_segment(
            positions[source],
            positions[target],
            start_offset=node_radius,
            end_offset=node_radius + 2.0,
        )
        pieces.append(
            _svg_line(
                start,
                end,
                stroke="#adb7c2",
                stroke_width=1.8,
                marker_end="url(#static-arrow)",
            )
        )

    for sources, target in hyperedges:
        source_points = [positions[index] for index in sources]
        target_point = positions[target]
        centroid = (
            sum(point[0] for point in source_points) / len(source_points),
            sum(point[1] for point in source_points) / len(source_points),
        )
        junction = (
            target_point[0] - 0.42 * (target_point[0] - centroid[0]),
            target_point[1] - 0.42 * (target_point[1] - centroid[1]),
        )
        for source_point in source_points:
            start, end = _shorten_segment(
                source_point,
                junction,
                start_offset=node_radius,
                end_offset=6.5,
            )
            pieces.append(
                _svg_line(
                    start,
                    end,
                    stroke="#e17c05",
                    stroke_width=2.0,
                    dash="5,3",
                    opacity=0.95,
                )
            )
        start, end = _shorten_segment(
            junction,
            target_point,
            start_offset=6.5,
            end_offset=node_radius + 2.0,
        )
        pieces.append(
            _svg_line(
                start,
                end,
                stroke="#e17c05",
                stroke_width=2.2,
                dash="5,3",
                opacity=0.95,
                marker_end="url(#static-arrow)",
            )
        )
        pieces.append(
            f"<circle cx='{junction[0]:.1f}' cy='{junction[1]:.1f}' r='6.5' fill='white' stroke='#e17c05' stroke-width='2.0'/>"
        )

    for index, (x, y) in positions.items():
        pieces.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{node_radius:.0f}' fill='#eef3f7' stroke='#66727f' stroke-width='2.0'/>"
        )
        pieces.append(
            f"<text x='{x:.1f}' y='{y + 4.0:.1f}' text-anchor='middle' font-size='12' font-weight='700' fill='#111'>{MICRO_LABELS[index]}</text>"
        )

    pieces.append(
        "<rect x='413.0' y='60.0' width='126' height='46.0' rx='8' ry='8' fill='#fbfcfd' stroke='#d7dee6' stroke-width='1'/>"
    )
    pieces.append(
        _svg_line(
            (425.0, 76.0),
            (443.0, 76.0),
            stroke="#adb7c2",
            stroke_width=2.0,
            marker_end="url(#static-arrow)",
        )
    )
    pieces.append("<text x='449.0' y='80.0' font-size='12.5' fill='#333'>copy</text>")
    pieces.append(
        _svg_line(
            (425.0, 94.0),
            (443.0, 94.0),
            stroke="#e17c05",
            stroke_width=2.0,
            dash="5,3",
        )
    )
    pieces.append("<text x='449.0' y='98.0' font-size='12.5' fill='#333'>AND/OR</text>")
    pieces.append("</svg>")

    svg_text = "".join(pieces)
    svg_path.write_text(svg_text, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_position([0.0, 0.0, 1.0, 1.0])

    if title:
        ax.text(14, 22, title, fontsize=16, fontweight="bold", color="#111", va="center")
    if subtitle:
        ax.text(14, 40, subtitle, fontsize=11, color="#555", va="center")

    for source, target in copy_edges:
        start, end = _shorten_segment(
            positions[source],
            positions[target],
            start_offset=node_radius,
            end_offset=node_radius + 2.0,
        )
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.8,
            color="#adb7c2",
            connectionstyle="arc3,rad=0.0",
            shrinkA=0.0,
            shrinkB=0.0,
        )
        ax.add_patch(arrow)

    for sources, target in hyperedges:
        source_points = [positions[index] for index in sources]
        target_point = positions[target]
        centroid = (
            sum(point[0] for point in source_points) / len(source_points),
            sum(point[1] for point in source_points) / len(source_points),
        )
        junction = (
            target_point[0] - 0.42 * (target_point[0] - centroid[0]),
            target_point[1] - 0.42 * (target_point[1] - centroid[1]),
        )
        for source_point in source_points:
            start, end = _shorten_segment(
                source_point,
                junction,
                start_offset=node_radius,
                end_offset=6.5,
            )
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="#e17c05",
                linewidth=2.0,
                linestyle=(0, (5, 3)),
                alpha=0.95,
            )
        start, end = _shorten_segment(
            junction,
            target_point,
            start_offset=6.5,
            end_offset=node_radius + 2.0,
        )
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=2.2,
            color="#e17c05",
            linestyle=(0, (5, 3)),
            connectionstyle="arc3,rad=0.0",
            shrinkA=0.0,
            shrinkB=0.0,
            alpha=0.95,
        )
        ax.add_patch(arrow)
        ax.add_patch(
            Circle(
                junction,
                radius=6.5,
                facecolor="white",
                edgecolor="#e17c05",
                linewidth=2.0,
            )
        )

    for index, (x, y) in positions.items():
        ax.add_patch(
            Circle(
                (x, y),
                radius=node_radius,
                facecolor="#eef3f7",
                edgecolor="#66727f",
                linewidth=2.0,
            )
        )
        ax.text(
            x,
            y,
            MICRO_LABELS[index],
            fontsize=12,
            fontweight="bold",
            color="#111",
            ha="center",
            va="center",
        )

    legend_box = FancyBboxPatch(
        (413.0, 60.0),
        126.0,
        46.0,
        boxstyle="round,pad=0.35,rounding_size=8",
        facecolor="#fbfcfd",
        edgecolor="#d7dee6",
        linewidth=1.0,
    )
    ax.add_patch(legend_box)
    copy_arrow = FancyArrowPatch(
        (425.0, 76.0),
        (443.0, 76.0),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=2.0,
        color="#adb7c2",
    )
    ax.add_patch(copy_arrow)
    ax.text(449.0, 80.0, "copy", fontsize=12.5, color="#333", va="center")
    ax.plot(
        [425.0, 443.0],
        [94.0, 94.0],
        color="#e17c05",
        linewidth=2.0,
        linestyle=(0, (5, 3)),
    )
    ax.text(449.0, 98.0, "AND/OR", fontsize=12.5, color="#333", va="center")

    fig.savefig(png_path, format="png", dpi=dpi)
    plt.close(fig)
    return svg_text


def _groups_to_label(groups: tuple[tuple[int, ...], ...]) -> str:
    parts = []
    for block in groups:
        parts.append("{" + ",".join(MICRO_LABELS[index] for index in block) + "}")
    return " | ".join(parts)


def _render_metrics_html(summary: dict[str, Any]) -> str:
    rows = [
        (
            "Optimal OR coarse-graining",
            _groups_to_label(summary["optimal_groups"]),
            f"{summary['optimal_ei']:.3f}",
            f"{summary['optimal_syn']:.3f}",
            "0.000",
        ),
        (
            "Representative non-optimal grouping",
            _groups_to_label(summary["nonoptimal_groups"]),
            f"{summary['nonoptimal_ei']:.3f}",
            f"{summary['nonoptimal_syn']:.3f}",
            f"{summary['nonoptimal_max_hyperedge']:.3f}",
        ),
        (
            "Mean over all 90 OR partitions",
            "all 3-block partitions with OR pooling",
            f"{summary['or_partition_mean_ei']:.3f}",
            f"{summary['or_partition_mean_syn']:.3f}",
            "-",
        ),
    ]
    header = "".join(
        f"<th style='border-bottom:1px solid #c8d0da;padding:6px 10px;text-align:left;'>{html.escape(column)}</th>"
        for column in ("Case", "Groups", "EI", "Syn", "Max binary synergy")
    )
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(
                f"<td style='border-bottom:1px solid #e7ebef;padding:6px 10px;'>{html.escape(str(value))}</td>"
                for value in row
            )
            + "</tr>"
        )
    return (
        "<h3 style='margin:0 0 8px 0;'>关键指标</h3>"
        "<p style='margin:0 0 10px 0;color:#444;font-size:13px;'>"
        "这里把最优自然分组、附录使用的代表性非最优分组，以及当前 support 模块下"
        "对全部 90 个 OR 分区的平均值并列展示。"
        "</p>"
        "<table style='border-collapse:collapse;font-size:13px;'>"
        "<thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _compute_analysis(*, figure_dir: Path) -> dict[str, Any]:
    micro_tpm = build_manual_micro_tpm()
    search_rows = search_binary_or_fixed_macro_dim_coarse_grainings(
        micro_tpm,
        n_nodes=len(MICRO_LABELS),
        n_macro=len(MACRO_LABELS),
    )

    optimal_row = search_rows[0]
    nonoptimal_row = next(
        row for row in search_rows if tuple(row["groups"]) == NONOPTIMAL_GROUPS
    )

    micro_causal = discrete_causal_graph(
        micro_tpm,
        n_nodes=len(MICRO_LABELS),
        synergy_order=2,
    )
    optimal_macro_tpm = coarse_grain_binary_or_tpm(
        micro_tpm,
        n_nodes=len(MICRO_LABELS),
        groups=OPTIMAL_GROUPS,
    )
    optimal_causal = discrete_causal_graph(
        optimal_macro_tpm,
        n_nodes=len(MACRO_LABELS),
        synergy_order=2,
    )
    nonoptimal_causal = discrete_causal_graph(
        nonoptimal_row["macro_tpm"],
        n_nodes=len(MACRO_LABELS),
        synergy_order=2,
    )
    micro_mechanism_hyperedges = build_manual_micro_mechanism_hyperedges()

    summary = {
        "cache_version": DEFAULT_CACHE_VERSION,
        "n_micro_nodes": len(MICRO_LABELS),
        "n_macro_nodes": len(MACRO_LABELS),
        "candidate_count": len(search_rows),
        "optimal_groups": OPTIMAL_GROUPS,
        "optimal_ei": float(optimal_row["ei"]),
        "optimal_syn": float(optimal_row["syn"]),
        "nonoptimal_groups": NONOPTIMAL_GROUPS,
        "nonoptimal_ei": float(nonoptimal_row["ei"]),
        "nonoptimal_syn": float(nonoptimal_row["syn"]),
        "nonoptimal_max_hyperedge": _max_positive_hyperedge(nonoptimal_causal["hyperedges"]),
        "or_partition_mean_ei": float(np.mean([row["ei"] for row in search_rows])),
        "or_partition_mean_syn": float(np.mean([row["syn"] for row in search_rows])),
        "figure_dir": str(figure_dir),
    }

    return {
        "micro_tpm": micro_tpm,
        "summary": summary,
        "micro_pairwise_display": _threshold_pairwise(micro_causal["pairwise_ei"]),
        "micro_hyperedges_display": micro_mechanism_hyperedges,
        "micro_mechanism_hyperedges": micro_mechanism_hyperedges,
        "optimal_pairwise_display": _threshold_pairwise(optimal_causal["pairwise_ei"]),
        "optimal_hyperedges_display": _select_display_hyperedges(optimal_causal["hyperedges"]),
        "nonoptimal_pairwise_display": _threshold_pairwise(nonoptimal_causal["pairwise_ei"]),
        "nonoptimal_hyperedges_display": _select_display_hyperedges(nonoptimal_causal["hyperedges"]),
    }


def run_manual_case_analysis(
    *,
    refresh_cache: bool = False,
    export_figures: bool = True,
    cache_dir: Path | None = None,
    results_dir: Path | None = None,
    figure_dir: Path | None = None,
) -> dict[str, Any]:
    cache_root = Path(cache_dir or DEFAULT_CACHE_DIR)
    results_root = Path(results_dir or DEFAULT_RESULTS_DIR)
    figures_root = Path(figure_dir or DEFAULT_FIGURE_DIR)
    cache_path = cache_root / DEFAULT_CACHE_PATH.name
    results_summary_path = results_root / DEFAULT_RESULTS_SUMMARY_PATH.name
    metrics_path = results_root / DEFAULT_RESULTS_METRICS_PATH.name
    figure_paths = {
        key: figures_root / path.name
        for key, path in FIGURE_PATHS.items()
    }

    required_paths = (
        cache_path,
        figure_paths["topology_svg"],
        figure_paths["optimal_svg"],
        figure_paths["nonoptimal_svg"],
        figure_paths["topology_png"],
        figure_paths["optimal_png"],
        figure_paths["nonoptimal_png"],
        metrics_path,
        results_summary_path,
    )

    if not refresh_cache and all(path.exists() for path in required_paths):
        summary = _load_cached_summary(cache_path)
        if summary.get("cache_version") == DEFAULT_CACHE_VERSION:
            metrics_html = metrics_path.read_text(encoding="utf-8")
            return {
                "MANUAL_CASE_RESULTS": summary,
                "topology_svg": figure_paths["topology_svg"].read_text(encoding="utf-8"),
                "optimal_comparison_svg": figure_paths["optimal_svg"].read_text(encoding="utf-8"),
                "nonoptimal_comparison_svg": figure_paths["nonoptimal_svg"].read_text(encoding="utf-8"),
                "metrics_html": metrics_html,
                "manual_case_cache_path": str(cache_path),
                "micro_mechanism_hyperedges": build_manual_micro_mechanism_hyperedges(),
            }

    analysis = _compute_analysis(figure_dir=figures_root)
    summary = analysis["summary"]
    metrics_html = _render_metrics_html(summary)

    if export_figures:
        topology_svg = render_manual_micro_topology_assets(
            svg_path=figure_paths["topology_svg"],
            png_path=figure_paths["topology_png"],
        )
        optimal_svg = render_coarse_graining_comparison_svg(
            "",
            micro_pairwise=analysis["micro_pairwise_display"],
            micro_labels=MICRO_LABELS,
            macro_labels=MACRO_LABELS,
            groups=OPTIMAL_GROUPS,
            macro_pairwise=analysis["optimal_pairwise_display"],
            macro_hyperedges=analysis["optimal_hyperedges_display"],
            micro_hyperedges=analysis["micro_hyperedges_display"],
            width=980,
            height=760,
        )
        figure_paths["optimal_svg"].write_text(optimal_svg, encoding="utf-8")
        render_coarse_graining_comparison_png(
            figure_paths["optimal_png"],
            "",
            micro_pairwise=analysis["micro_pairwise_display"],
            micro_labels=MICRO_LABELS,
            macro_labels=MACRO_LABELS,
            groups=OPTIMAL_GROUPS,
            macro_pairwise=analysis["optimal_pairwise_display"],
            macro_hyperedges=analysis["optimal_hyperedges_display"],
            micro_hyperedges=analysis["micro_hyperedges_display"],
            width=980,
            height=760,
        )

        nonoptimal_svg = render_coarse_graining_comparison_svg(
            "",
            micro_pairwise=analysis["micro_pairwise_display"],
            micro_labels=MICRO_LABELS,
            macro_labels=MACRO_LABELS,
            groups=NONOPTIMAL_GROUPS,
            macro_pairwise=analysis["nonoptimal_pairwise_display"],
            macro_hyperedges=analysis["nonoptimal_hyperedges_display"],
            micro_hyperedges=analysis["micro_hyperedges_display"],
            width=980,
            height=760,
        )
        figure_paths["nonoptimal_svg"].write_text(nonoptimal_svg, encoding="utf-8")
        render_coarse_graining_comparison_png(
            figure_paths["nonoptimal_png"],
            "",
            micro_pairwise=analysis["micro_pairwise_display"],
            micro_labels=MICRO_LABELS,
            macro_labels=MACRO_LABELS,
            groups=NONOPTIMAL_GROUPS,
            macro_pairwise=analysis["nonoptimal_pairwise_display"],
            macro_hyperedges=analysis["nonoptimal_hyperedges_display"],
            micro_hyperedges=analysis["micro_hyperedges_display"],
            width=980,
            height=760,
        )
    else:
        topology_svg = render_manual_micro_topology_assets(
            svg_path=figure_paths["topology_svg"],
            png_path=figure_paths["topology_png"],
        )
        optimal_svg = render_coarse_graining_comparison_svg(
            "",
            micro_pairwise=analysis["micro_pairwise_display"],
            micro_labels=MICRO_LABELS,
            macro_labels=MACRO_LABELS,
            groups=OPTIMAL_GROUPS,
            macro_pairwise=analysis["optimal_pairwise_display"],
            macro_hyperedges=analysis["optimal_hyperedges_display"],
            micro_hyperedges=analysis["micro_hyperedges_display"],
            width=980,
            height=760,
        )
        nonoptimal_svg = render_coarse_graining_comparison_svg(
            "",
            micro_pairwise=analysis["micro_pairwise_display"],
            micro_labels=MICRO_LABELS,
            macro_labels=MACRO_LABELS,
            groups=NONOPTIMAL_GROUPS,
            macro_pairwise=analysis["nonoptimal_pairwise_display"],
            macro_hyperedges=analysis["nonoptimal_hyperedges_display"],
            micro_hyperedges=analysis["micro_hyperedges_display"],
            width=980,
            height=760,
        )

    _write_json(cache_path, {"summary": summary})
    _write_json(results_summary_path, {"summary": summary})
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(metrics_html, encoding="utf-8")

    return {
        "MANUAL_CASE_RESULTS": summary,
        "topology_svg": topology_svg,
        "optimal_comparison_svg": optimal_svg,
        "nonoptimal_comparison_svg": nonoptimal_svg,
        "metrics_html": metrics_html,
        "manual_case_cache_path": str(cache_path),
        "micro_mechanism_hyperedges": analysis["micro_mechanism_hyperedges"],
    }
