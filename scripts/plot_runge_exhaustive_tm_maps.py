#!/usr/bin/env python3
"""Render verified global top-10 exhaustive degree-3 TM Runge hyperedges."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scripts.plot_runge_gateway_mediator_map import (
        COASTLINE_URL,
        DEFAULT_COMPONENT_MAPS,
        LAND_URL,
        add_geographic_ticks,
        component_center,
        draw_world,
        extract_lines,
        extract_polygons,
        load_geojson,
        local_to_paper,
    )
except ModuleNotFoundError:
    from plot_runge_gateway_mediator_map import (
        COASTLINE_URL,
        DEFAULT_COMPONENT_MAPS,
        LAND_URL,
        add_geographic_ticks,
        component_center,
        draw_world,
        extract_lines,
        extract_polygons,
        load_geojson,
        local_to_paper,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = (
    ROOT
    / "results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/"
    "multistep_conditioned_ei_tm_exhaustive"
)
DEFAULT_OUTPUT_DIR = ROOT / "fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive"
N_COMPONENTS = 60
CANDIDATE_COUNT = 102660
TOP_N = 10
RANKING_COLUMNS = (
    "source_a",
    "source_b",
    "target",
    "raw_ei_a",
    "raw_ei_b",
    "raw_joint_ei",
    "ei_a",
    "ei_b",
    "joint_ei",
    "delta2_tm",
    "tm_rank",
)
METADATA_FIELDS = (
    "schema_version",
    "input_fingerprint",
    "estimator_fingerprint",
    "horizon",
    "candidate_count",
    "candidate_order_hash",
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.65,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def fingerprint_array(values: np.ndarray) -> str:
    """Return the ranking-format fingerprint without depending on the scorer."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def canonical_candidates() -> np.ndarray:
    """Build the degree-3 cross-target candidate universe in canonical order."""

    return np.asarray(
        [
            (source_a, source_b, target)
            for source_a in range(N_COMPONENTS)
            for source_b in range(source_a + 1, N_COMPONENTS)
            for target in range(N_COMPONENTS)
            if target not in (source_a, source_b)
        ],
        dtype=np.int16,
    )


def parse_horizons(text: str) -> list[int]:
    values: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first, last = item.split("-", 1)
            values.extend(range(int(first), int(last) + 1))
        else:
            values.append(int(item))
    if not values or any(value < 1 for value in values):
        raise ValueError("--horizons must contain positive integer horizons.")
    return sorted(dict.fromkeys(values))


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON metadata: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON metadata must be an object: {path}")
    return payload


def _load_ranking_arrays(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {column: np.asarray(archive[column]) for column in RANKING_COLUMNS}
            metadata = json.loads(str(archive["metadata_json"].item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read full ranking: {path}") from error
    if not isinstance(metadata, dict):
        raise ValueError("Ranking metadata must be an object.")
    return arrays, metadata


def _validate_metadata(summary: dict[str, object], metadata: dict[str, object], horizon: int) -> None:
    if any(field not in metadata for field in METADATA_FIELDS):
        raise ValueError("Ranking metadata is incomplete.")
    if int(metadata["schema_version"]) != 1:
        raise ValueError("Ranking metadata schema_version is unsupported.")
    ranking_metadata = summary.get("ranking_metadata")
    if not isinstance(ranking_metadata, dict) or ranking_metadata != metadata:
        raise ValueError("summary.json ranking_metadata does not match full_ranking.npz.")
    for field in ("horizon", "input_fingerprint", "estimator_fingerprint", "candidate_count"):
        if summary.get(field) != metadata.get(field):
            raise ValueError(f"summary.json {field} does not match ranking metadata.")
    if int(metadata["horizon"]) != int(horizon):
        raise ValueError("Ranking horizon does not match requested horizon.")
    if int(metadata["candidate_count"]) != CANDIDATE_COUNT:
        raise ValueError(f"candidate_count must equal {CANDIDATE_COUNT}.")
    if int(summary.get("candidate_count", -1)) != CANDIDATE_COUNT or not bool(summary.get("finite")):
        raise ValueError("summary.json does not certify a finite exhaustive candidate ranking.")


def _validate_ranking(arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> pd.DataFrame:
    lengths = {len(values) for values in arrays.values()}
    if lengths != {CANDIDATE_COUNT}:
        raise ValueError(f"full_ranking.npz must contain exactly {CANDIDATE_COUNT} rows.")
    for column, values in arrays.items():
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError(f"Ranking column {column} must contain finite numeric values.")
    triples = np.column_stack([arrays["source_a"], arrays["source_b"], arrays["target"]])
    if not np.all(np.equal(triples, np.floor(triples))):
        raise ValueError("Candidate indices must be integers.")
    triples = triples.astype(np.int16)
    if np.any(triples < 0) or np.any(triples >= N_COMPONENTS):
        raise ValueError("Candidate indices must be in [0, 59].")
    if not np.all(triples[:, 0] < triples[:, 1]):
        raise ValueError("Candidates require source_a < source_b.")
    if np.any(triples[:, 0] == triples[:, 2]) or np.any(triples[:, 1] == triples[:, 2]):
        raise ValueError("Candidate sources must be distinct from target.")
    if len(np.unique(triples, axis=0)) != CANDIDATE_COUNT:
        raise ValueError("Candidate universe contains duplicate triples.")
    canonical = canonical_candidates()
    if canonical.shape != (CANDIDATE_COUNT, 3) or not np.array_equal(
        triples[np.lexsort((triples[:, 2], triples[:, 1], triples[:, 0]))], canonical
    ):
        raise ValueError("Candidate universe is not the exact canonical degree-3 cross-target universe.")
    if fingerprint_array(triples) != str(metadata["candidate_order_hash"]):
        raise ValueError("Candidate order hash does not match ranking metadata.")
    expected_ranks = np.arange(1, CANDIDATE_COUNT + 1, dtype=arrays["tm_rank"].dtype)
    if not np.array_equal(arrays["tm_rank"], expected_ranks):
        raise ValueError("tm_rank must be consecutive ranks 1..N.")
    if np.any(arrays["delta2_tm"][:-1] < arrays["delta2_tm"][1:]):
        raise ValueError("delta2_tm must be sorted in descending order.")
    return pd.DataFrame(arrays)


def load_exhaustive_top10(result_dir: str | Path, *, horizon: int, top_n: int = TOP_N) -> pd.DataFrame:
    """Load a verified global TM ranking and return top rows with paper labels."""

    if int(top_n) < 1 or int(top_n) > CANDIDATE_COUNT:
        raise ValueError(f"top_n must be between 1 and {CANDIDATE_COUNT}.")
    horizon_dir = Path(result_dir).expanduser() / f"H{int(horizon):03d}"
    summary = _load_json(horizon_dir / "summary.json")
    arrays, metadata = _load_ranking_arrays(horizon_dir / "full_ranking.npz")
    _validate_metadata(summary, metadata, int(horizon))
    ranking = _validate_ranking(arrays, metadata)
    frame = ranking.head(int(top_n)).copy()
    frame["source_a_local"] = frame["source_a"].astype(int)
    frame["source_b_local"] = frame["source_b"].astype(int)
    frame["target_local"] = frame["target"].astype(int)
    frame["source_a_paper"] = frame["source_a_local"].map(local_to_paper)
    frame["source_b_paper"] = frame["source_b_local"].map(local_to_paper)
    frame["target_paper"] = frame["target_local"].map(local_to_paper)
    frame["input_fingerprint"] = str(metadata["input_fingerprint"])
    return frame[
        [
            "tm_rank",
            "source_a_local",
            "source_b_local",
            "target_local",
            "source_a_paper",
            "source_b_paper",
            "target_paper",
            "delta2_tm",
            "joint_ei",
            "ei_a",
            "ei_b",
            "input_fingerprint",
        ]
    ]


def load_nodes(component_maps_path: str | Path) -> pd.DataFrame:
    component_maps = np.load(Path(component_maps_path), allow_pickle=False)["component_maps"]
    if component_maps.ndim != 3 or component_maps.shape[2] != N_COMPONENTS:
        raise ValueError("component_maps must have 60 components on a [lat, lon, component] grid.")
    lat = np.linspace(-90.0, 90.0, component_maps.shape[0])
    lon = ((np.linspace(0.0, 360.0, component_maps.shape[1], endpoint=False) + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    rows = []
    for local in range(N_COMPONENTS):
        center_lon, center_lat = component_center(component_maps[:, order, local], lat, lon[order])
        rows.append({"local": local, "paper": local_to_paper(local), "lon": center_lon, "lat": center_lat})
    return pd.DataFrame(rows)


def _to_axes_xy(ax: plt.Axes, lon: float, lat: float) -> np.ndarray:
    display = ax.transData.transform((np.radians(lon), np.radians(lat)))
    return ax.transAxes.inverted().transform(display)


def _offset_hub(source_mid: np.ndarray, target_xy: np.ndarray, index: int) -> np.ndarray:
    direction = target_xy - source_mid
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        direction, length = np.array([1.0, 0.0]), 1.0
    direction = direction / length
    perpendicular = np.array([-direction[1], direction[0]])
    side = -1.0 if index % 2 else 1.0
    hub = 0.58 * source_mid + 0.42 * target_xy + side * 0.012 * (1 + index // 2) * perpendicular
    return np.clip(hub, np.array([0.04, 0.08]), np.array([0.96, 0.92]))


def _draw_hyperedges(ax: plt.Axes, nodes: pd.DataFrame, frame: pd.DataFrame) -> None:
    lookup = nodes.set_index("local")
    active = set(frame[["source_a_local", "source_b_local", "target_local"]].to_numpy().ravel().astype(int))
    inactive = nodes[~nodes["local"].isin(active)]
    targets = set(frame["target_local"].astype(int)) - set(frame[["source_a_local", "source_b_local"]].to_numpy().ravel().astype(int))
    sources = set(frame[["source_a_local", "source_b_local"]].to_numpy().ravel().astype(int))
    ax.scatter(np.radians(inactive.lon), np.radians(inactive.lat), s=48, color="#9aa0a6", edgecolors="white", linewidths=0.22, alpha=0.28, zorder=3)
    for subset, color, size, edgecolor, zorder in (
        (targets, "#2a9d8f", 210, "#18343c", 5),
        (sources, "#1f78b4", 350, "#112f43", 6),
    ):
        selected = nodes[nodes["local"].isin(subset)]
        ax.scatter(np.radians(selected.lon), np.radians(selected.lat), s=size, color=color, edgecolors=edgecolor, linewidths=0.72, alpha=0.96, zorder=zorder)
    values = frame["delta2_tm"].to_numpy(dtype=float)
    denominator = max(float(values.max() - values.min()), 1e-12)
    for index, row in enumerate(frame.itertuples(index=False)):
        source_xy = [_to_axes_xy(ax, float(lookup.loc[source].lon), float(lookup.loc[source].lat)) for source in (row.source_a_local, row.source_b_local)]
        target_xy = _to_axes_xy(ax, float(lookup.loc[row.target_local].lon), float(lookup.loc[row.target_local].lat))
        hub = _offset_hub(0.5 * (source_xy[0] + source_xy[1]), target_xy, index)
        strength = (float(row.delta2_tm) - float(values.min())) / denominator
        linewidth, alpha = 0.55 + 2.35 * strength, 0.22 + 0.50 * strength
        for start_xy in source_xy:
            ax.plot([start_xy[0], hub[0]], [start_xy[1], hub[1]], transform=ax.transAxes, color="#7c2d6c", linewidth=max(0.45, linewidth * 0.72), alpha=max(0.14, alpha * 0.55), solid_capstyle="round", zorder=4.2)
        ax.add_patch(mpatches.FancyArrowPatch(posA=hub, posB=target_xy, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=5.5 + 3.0 * strength, linewidth=linewidth, color="#7c2d6c", alpha=alpha, shrinkA=1.0, shrinkB=9.0, connectionstyle=f"arc3,rad={0.075 if index % 2 == 0 else -0.075}", clip_on=True, zorder=4.4))
        ax.scatter([hub[0]], [hub[1]], transform=ax.transAxes, s=10 + 13 * strength, color="#7c2d6c", edgecolors="white", linewidths=0.28, alpha=min(0.84, alpha + 0.22), zorder=7)
    for row in nodes[nodes["local"].isin(active)].itertuples(index=False):
        ax.text(np.radians(row.lon), np.radians(row.lat), str(int(row.paper)), ha="center", va="center", fontsize=7.2, weight="bold", color="white", path_effects=[pe.withStroke(linewidth=1.45, foreground="#1b1b1b")], zorder=8)


def render_top10_map(nodes: pd.DataFrame, frame: pd.DataFrame, output_base: str | Path, *, horizon: int) -> list[Path]:
    """Render one horizon in the established Mollweide source-pair map style."""

    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))
    fig = plt.figure(figsize=(7.7, 4.35), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="mollweide")
    draw_world(ax, land, coastlines)
    add_geographic_ticks(ax)
    _draw_hyperedges(ax, nodes, frame)
    ax.set_title(f"Global TM top {len(frame)} second-order hyperedges (H={int(horizon)})", fontsize=8.4, fontweight="bold", pad=8)
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, options in ((".png", {"dpi": 350}), (".svg", {}), (".pdf", {})):
        path = base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **options)
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_horizons(result_dir: str | Path, horizons: Iterable[int], output_dir: str | Path, component_maps: str | Path) -> dict[int, list[Path]]:
    """Write CSV and PNG/SVG/PDF artifacts for each requested verified horizon."""

    nodes = load_nodes(component_maps)
    root = Path(output_dir).expanduser()
    outputs: dict[int, list[Path]] = {}
    for horizon in horizons:
        frame = load_exhaustive_top10(result_dir, horizon=int(horizon))
        base = root / f"top10_order2_hyperedges_H{int(horizon):03d}_tm_exhaustive"
        csv_path = base.with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv_path, index=False)
        outputs[int(horizon)] = [*render_top10_map(nodes, frame, base, horizon=int(horizon)), csv_path]
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--horizons", default="1,10,60")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--component-maps", type=Path, default=DEFAULT_COMPONENT_MAPS)
    args = parser.parse_args()
    outputs = plot_horizons(args.result_dir, parse_horizons(args.horizons), args.output_dir, args.component_maps)
    print(json.dumps({str(horizon): [str(path) for path in paths] for horizon, paths in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
