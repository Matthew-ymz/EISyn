from __future__ import annotations

import html
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_horizon_comparison_plot(frame: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    ax.plot(frame["horizon"], frame["syn_nis"], marker="o", label="Syn_nis")
    ax.set_xlabel("Horizon (hours)")
    ax.set_ylabel("Coupling strength")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _edge_strength(row: pd.Series) -> float:
    for key in ("abs_mean", "mean", "value"):
        if key in row and pd.notna(row[key]):
            return abs(float(row[key]))
    return 0.0


def _normalize_source_pair(value: object) -> tuple[str, str]:
    if isinstance(value, str):
        if "|" in value:
            left, right = value.split("|", maxsplit=1)
            return left.strip(), right.strip()
        if "," in value:
            left, right = value.split(",", maxsplit=1)
            return left.strip(), right.strip()
        raise ValueError("source_station_ids must contain two station ids.")

    pair = tuple(value)  # type: ignore[arg-type]
    if len(pair) != 2:
        raise ValueError("source_station_ids must contain exactly two station ids.")
    return str(pair[0]), str(pair[1])


def _scale_positions(frame: pd.DataFrame, *, width: float, height: float) -> dict[str, tuple[float, float]]:
    required = {"station_id", "lon", "lat"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"station_positions is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("station_positions must be non-empty.")

    left_margin = 72.0
    right_margin = 72.0
    top_margin = 82.0
    bottom_margin = 68.0
    plot_width = width - left_margin - right_margin - 180.0
    plot_height = height - top_margin - bottom_margin

    lon_min = float(frame["lon"].min())
    lon_max = float(frame["lon"].max())
    lat_min = float(frame["lat"].min())
    lat_max = float(frame["lat"].max())
    lon_span = max(lon_max - lon_min, 1e-9)
    lat_span = max(lat_max - lat_min, 1e-9)

    positions: dict[str, tuple[float, float]] = {}
    for _, row in frame.iterrows():
        x = left_margin + (float(row["lon"]) - lon_min) / lon_span * plot_width
        y = top_margin + (lat_max - float(row["lat"])) / lat_span * plot_height
        positions[str(row["station_id"])] = (x, y)
    return positions


def _quadratic_path(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    curvature: float,
    stroke: str,
    stroke_width: float,
    opacity: float,
    marker_end: str = "",
    dasharray: str = "",
) -> str:
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    distance = max(math.hypot(dx, dy), 1e-9)
    px = -dy / distance
    py = dx / distance
    cx = 0.5 * (x1 + x2) + curvature * distance * px
    cy = 0.5 * (y1 + y2) + curvature * distance * py
    marker = f" marker-end='{marker_end}'" if marker_end else ""
    dash = f" stroke-dasharray='{dasharray}'" if dasharray else ""
    return (
        f"<path d='M {x1:.1f} {y1:.1f} Q {cx:.1f} {cy:.1f} {x2:.1f} {y2:.1f}' "
        f"fill='none' stroke='{stroke}' stroke-width='{stroke_width:.2f}' opacity='{opacity:.2f}'{marker}{dash}/>"
    )


def render_station_causal_graph_svg(
    *,
    station_positions: pd.DataFrame,
    pairwise_edges: pd.DataFrame,
    binary_hyperedges: pd.DataFrame,
    horizon_label: str,
) -> str:
    width = 1120.0
    height = 800.0
    node_radius = 8.0
    positions = _scale_positions(station_positions, width=width, height=height)

    pair_strengths = [
        _edge_strength(row)
        for _, row in pairwise_edges.iterrows()
    ] if not pairwise_edges.empty else [0.0]
    hyper_strengths = [
        _edge_strength(row)
        for _, row in binary_hyperedges.iterrows()
    ] if not binary_hyperedges.empty else [0.0]
    max_pair_strength = max(max(pair_strengths), 1e-9)
    max_hyper_strength = max(max(hyper_strengths), 1e-9)

    pieces = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{int(width)}' height='{int(height)}' viewBox='0 0 {int(width)} {int(height)}'>",
        "<defs><marker id='station-arrow' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto' markerUnits='userSpaceOnUse'>"
        "<path d='M 0 0 L 8 3 L 0 6 z' fill='#345995'/></marker></defs>",
        f"<rect x='0' y='0' width='{int(width)}' height='{int(height)}' fill='white'/>",
        "<rect x='48' y='60' width='860' height='660' rx='18' ry='18' fill='#fbfcfe' stroke='#d7dee6' stroke-width='1.0'/>",
        f"<text x='72' y='100' font-size='18' font-weight='700' fill='#111'>Shanghai station-level causal graph ({html.escape(horizon_label)})</text>",
        "<text x='72' y='122' font-size='11' fill='#5b6572'>Directed pairwise edges and binary hyperedge junctions</text>",
    ]

    for index, (_, row) in enumerate(pairwise_edges.iterrows()):
        source_id = str(row["source_station_id"])
        target_id = str(row["target_station_id"])
        if source_id not in positions or target_id not in positions:
            continue
        strength = _edge_strength(row)
        x0, y0 = positions[source_id]
        x1, y1 = positions[target_id]
        dx = x1 - x0
        dy = y1 - y0
        distance = max(math.hypot(dx, dy), 1e-9)
        ux = dx / distance
        uy = dy / distance
        start = (x0 + node_radius * ux, y0 + node_radius * uy)
        end = (x1 - node_radius * ux, y1 - node_radius * uy)
        curvature = 0.10 * (1 if index % 2 == 0 else -1)
        stroke_width = 1.2 + 2.6 * strength / max_pair_strength
        pieces.append(
            _quadratic_path(
                start,
                end,
                curvature=curvature,
                stroke="#345995",
                stroke_width=stroke_width,
                opacity=0.82,
                marker_end="url(#station-arrow)",
            )
        )

    for _, row in binary_hyperedges.iterrows():
        source_left, source_right = _normalize_source_pair(row["source_station_ids"])
        target_id = str(row["target_station_id"])
        if source_left not in positions or source_right not in positions or target_id not in positions:
            continue
        strength = _edge_strength(row)
        left_x, left_y = positions[source_left]
        right_x, right_y = positions[source_right]
        target_x, target_y = positions[target_id]
        midpoint_x = 0.5 * (left_x + right_x)
        midpoint_y = 0.5 * (left_y + right_y)
        junction_x = 0.58 * midpoint_x + 0.42 * target_x
        junction_y = 0.58 * midpoint_y + 0.42 * target_y
        stroke_width = 1.3 + 2.3 * strength / max_hyper_strength
        color = "#2F7D63"

        for x0, y0 in ((left_x, left_y), (right_x, right_y)):
            dx = junction_x - x0
            dy = junction_y - y0
            distance = max(math.hypot(dx, dy), 1e-9)
            ux = dx / distance
            uy = dy / distance
            pieces.append(
                f"<line x1='{x0 + node_radius * ux:.1f}' y1='{y0 + node_radius * uy:.1f}' "
                f"x2='{junction_x - 6.0 * ux:.1f}' y2='{junction_y - 6.0 * uy:.1f}' "
                f"stroke='{color}' stroke-width='{stroke_width:.2f}' stroke-dasharray='5,3' opacity='0.92'/>"
            )

        dx = target_x - junction_x
        dy = target_y - junction_y
        distance = max(math.hypot(dx, dy), 1e-9)
        ux = dx / distance
        uy = dy / distance
        pieces.append(
            f"<line x1='{junction_x + 6.0 * ux:.1f}' y1='{junction_y + 6.0 * uy:.1f}' "
            f"x2='{target_x - node_radius * ux:.1f}' y2='{target_y - node_radius * uy:.1f}' "
            f"stroke='{color}' stroke-width='{stroke_width + 0.15:.2f}' stroke-dasharray='5,3' opacity='0.95' marker-end='url(#station-arrow)'/>"
        )
        pieces.append(
            f"<circle cx='{junction_x:.1f}' cy='{junction_y:.1f}' r='5.6' fill='white' stroke='{color}' stroke-width='1.0' opacity='0.95'/>"
        )

    for _, row in station_positions.iterrows():
        station_id = str(row["station_id"])
        x, y = positions[station_id]
        pieces.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='8' fill='#D8DEE9' stroke='#51606f' stroke-width='1.0'/>"
        )
        pieces.append(
            f"<text x='{x + 10.0:.1f}' y='{y + 3.0:.1f}' font-size='10' fill='#233142'>{html.escape(station_id)}</text>"
        )

    legend_x = 960.0
    legend_y = 130.0
    pieces.append(
        f"<rect x='{legend_x - 18.0:.1f}' y='{legend_y - 38.0:.1f}' width='160' height='108' rx='10' ry='10' fill='#fbfcfe' stroke='#d7dee6' stroke-width='1.0'/>"
    )
    pieces.append(f"<text x='{legend_x:.1f}' y='{legend_y:.1f}' font-size='11.5' font-weight='700' fill='#333'>Legend</text>")
    pieces.append(
        f"<line x1='{legend_x:.1f}' y1='{legend_y + 20.0:.1f}' x2='{legend_x + 28.0:.1f}' y2='{legend_y + 20.0:.1f}' "
        "stroke='#345995' stroke-width='2.4' marker-end='url(#station-arrow)'/>"
    )
    pieces.append(
        f"<text x='{legend_x + 38.0:.1f}' y='{legend_y + 24.0:.1f}' font-size='10.5' fill='#333'>Pairwise edge</text>"
    )
    pieces.append(
        f"<line x1='{legend_x:.1f}' y1='{legend_y + 46.0:.1f}' x2='{legend_x + 28.0:.1f}' y2='{legend_y + 46.0:.1f}' "
        "stroke='#2F7D63' stroke-width='2.2' stroke-dasharray='5,3' marker-end='url(#station-arrow)'/>"
    )
    pieces.append(
        f"<circle cx='{legend_x + 14.0:.1f}' cy='{legend_y + 70.0:.1f}' r='5.2' fill='white' stroke='#2F7D63' stroke-width='1.0'/>"
    )
    pieces.append(
        f"<text x='{legend_x + 38.0:.1f}' y='{legend_y + 50.0:.1f}' font-size='10.5' fill='#333'>Binary hyperedge</text>"
    )
    pieces.append(
        f"<text x='{legend_x + 38.0:.1f}' y='{legend_y + 74.0:.1f}' font-size='10.5' fill='#333'>junction</text>"
    )
    pieces.append("</svg>")
    return "".join(pieces)
