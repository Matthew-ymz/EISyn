#!/usr/bin/env python3
"""Compute gateway phi and broadcast redundancy for the 1948-2026 Runge SLP MLP."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_runge_scheme_a_full_order2 as full_order2  # noqa: E402
from scripts.plot_runge_gateway_mediator_map import (  # noqa: E402
    COASTLINE_URL,
    LAND_URL,
    add_geographic_ticks,
    add_labels,
    component_center,
    draw_world,
    extract_lines,
    extract_polygons,
    load_geojson,
    local_to_paper,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

DEFAULT_RUN_ROOT = Path("results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04")
DEFAULT_COMPONENT_SCORES = Path(
    "results/runge_slp_daily_1948_2026_20260628/results/runge/2015_gateways/component_weekly_scores.csv"
)
DEFAULT_COMPONENT_MAPS = Path(
    "results/runge_slp_daily_1948_2026_20260628/results/runge/2015_gateways/component_maps.npz"
)
DEFAULT_RESULT_DIR = DEFAULT_RUN_ROOT / "results/runge/gateway_broadcast_metrics"
DEFAULT_FIG_DIR = DEFAULT_RUN_ROOT / "fig/runge/gateway_broadcast_metrics"

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


def _coerce_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("expected a one- or two-dimensional array")
    return array


def _safe_logdet(covariance: np.ndarray, *, ridge: float) -> float:
    matrix = np.asarray(covariance, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    matrix = matrix + float(ridge) * np.eye(matrix.shape[0], dtype=float)
    sign, logdet = np.linalg.slogdet(matrix)
    if sign > 0 and np.isfinite(logdet):
        return float(logdet)
    eigenvalues = np.linalg.eigvalsh(matrix)
    floor = max(float(ridge), 1.0e-12)
    return float(np.log(np.clip(eigenvalues, floor, None)).sum())


def gaussian_mutual_information(x: np.ndarray, y: np.ndarray, *, ridge: float = 1.0e-6) -> float:
    """Gaussian log-det MI in bits for empirical intervention samples."""

    x_array = _coerce_2d(x)
    y_array = _coerce_2d(y)
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError("x and y must have matching sample counts")
    if x_array.shape[0] < 3:
        raise ValueError("at least three samples are required")
    joint = np.concatenate([x_array, y_array], axis=1)
    cov_x = np.cov(x_array, rowvar=False, bias=False)
    cov_y = np.cov(y_array, rowvar=False, bias=False)
    cov_joint = np.cov(joint, rowvar=False, bias=False)
    mi_nats = 0.5 * (
        _safe_logdet(np.atleast_2d(cov_x), ridge=ridge)
        + _safe_logdet(np.atleast_2d(cov_y), ridge=ridge)
        - _safe_logdet(np.atleast_2d(cov_joint), ridge=ridge)
    )
    return max(0.0, float(mi_nats / np.log(2.0)))


def compute_gaussian_ei_terms(
    source_samples: np.ndarray,
    target_samples: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = _coerce_2d(source_samples)
    target = _coerce_2d(target_samples)
    if source.shape != target.shape:
        raise ValueError("source and target samples must have shape [samples, n_components]")
    n_components = source.shape[1]
    pairwise = np.zeros((n_components, n_components), dtype=float)
    full_source_to_target = np.zeros(n_components, dtype=float)
    source_to_joint_target = np.zeros(n_components, dtype=float)
    for i in range(n_components):
        source_i = source[:, [i]]
        source_to_joint_target[i] = gaussian_mutual_information(source_i, target, ridge=ridge)
        for j in range(n_components):
            pairwise[i, j] = gaussian_mutual_information(source_i, target[:, [j]], ridge=ridge)
    for j in range(n_components):
        full_source_to_target[j] = gaussian_mutual_information(source, target[:, [j]], ridge=ridge)
    return pairwise, full_source_to_target, source_to_joint_target


def compute_gateway_broadcast_scores(
    *,
    pairwise_ei: np.ndarray,
    full_source_to_target_ei: np.ndarray,
    source_to_joint_target_ei: np.ndarray,
    names: Sequence[str],
    include_self: bool = False,
) -> pd.DataFrame:
    pairwise = np.asarray(pairwise_ei, dtype=float)
    full_source = np.asarray(full_source_to_target_ei, dtype=float)
    source_joint = np.asarray(source_to_joint_target_ei, dtype=float)
    if pairwise.ndim != 2 or pairwise.shape[0] != pairwise.shape[1]:
        raise ValueError("pairwise_ei must be square")
    n_components = pairwise.shape[0]
    if full_source.shape != (n_components,) or source_joint.shape != (n_components,):
        raise ValueError("joint EI vectors must have length n_components")
    if len(names) != n_components:
        raise ValueError("names length must match pairwise_ei")
    working = pairwise.copy()
    if not include_self:
        np.fill_diagonal(working, 0.0)
    single_source_sum_to_target = working.sum(axis=0)
    single_target_sum_from_source = working.sum(axis=1)
    gateway_phi = full_source - single_source_sum_to_target
    broadcast_redundancy = single_target_sum_from_source - source_joint
    frame = pd.DataFrame(
        {
            "component": list(names),
            "component_index": np.arange(n_components),
            "paper_component": [local_to_paper(i) for i in range(n_components)],
            "gateway_phi_eid": gateway_phi,
            "gateway_phi_eid_pos": np.maximum(gateway_phi, 0.0),
            "full_source_to_target_ei": full_source,
            "single_source_sum_to_target": single_source_sum_to_target,
            "broadcast_redundancy": broadcast_redundancy,
            "broadcast_redundancy_pos": np.maximum(broadcast_redundancy, 0.0),
            "broadcast_complementarity_pos": np.maximum(-broadcast_redundancy, 0.0),
            "source_to_joint_target_ei": source_joint,
            "single_target_sum_from_source": single_target_sum_from_source,
        }
    )
    frame["gateway_phi_rank"] = frame["gateway_phi_eid"].rank(ascending=False, method="min").astype(int)
    frame["broadcast_rank"] = frame["broadcast_redundancy"].rank(ascending=False, method="min").astype(int)
    return frame


def _latest_source_block(features: np.ndarray, *, n_components: int, lag: int) -> np.ndarray:
    start = (int(lag) - 1) * int(n_components)
    stop = start + int(n_components)
    return np.asarray(features[:, start:stop], dtype=float)


def load_or_compute_metrics(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    scores_path = result_dir / "gateway_broadcast_scores.csv"
    pairwise_path = result_dir / "gaussian_pairwise_ei_matrix.csv"
    full_source_path = result_dir / "full_source_to_target_ei.csv"
    source_joint_path = result_dir / "source_to_joint_target_ei.csv"
    manifest_path = result_dir / "manifest.json"

    if scores_path.exists() and pairwise_path.exists() and not args.force:
        scores = pd.read_csv(scores_path)
        pairwise = pd.read_csv(pairwise_path, index_col=0).to_numpy(dtype=float)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        return scores, pairwise, manifest

    config = full_order2.load_peid_config(args)
    intervention_features, predictions, names, n_components, member_summaries = full_order2.prepare_intervention_predictions(config)
    source_samples = _latest_source_block(intervention_features, n_components=n_components, lag=int(args.lag))
    pairwise, full_source, source_joint = compute_gaussian_ei_terms(source_samples, predictions, ridge=float(args.gaussian_ridge))
    scores = compute_gateway_broadcast_scores(
        pairwise_ei=pairwise,
        full_source_to_target_ei=full_source,
        source_to_joint_target_ei=source_joint,
        names=names,
        include_self=bool(args.include_self),
    )

    pd.DataFrame(pairwise, index=names, columns=names).to_csv(pairwise_path)
    pd.DataFrame({"component": names, "full_source_to_target_ei": full_source}).to_csv(full_source_path, index=False)
    pd.DataFrame({"component": names, "source_to_joint_target_ei": source_joint}).to_csv(source_joint_path, index=False)
    scores.to_csv(scores_path, index=False)

    manifest = {
        "scheme": "gateway_broadcast_metrics",
        "estimator": "gaussian_logdet",
        "gaussian_ridge": float(args.gaussian_ridge),
        "include_self": bool(args.include_self),
        "n_components": int(n_components),
        "intervention_samples": int(args.intervention_samples),
        "lag": int(args.lag),
        "source_mode": "latest",
        "component_scores": str(args.component_scores),
        "run_root": str(args.run_root),
        "model_members": member_summaries,
        "outputs": {
            "scores": str(scores_path),
            "pairwise_matrix": str(pairwise_path),
            "full_source_to_target": str(full_source_path),
            "source_to_joint_target": str(source_joint_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return scores, pairwise, manifest


def build_node_frame(component_maps_path: Path, scores: pd.DataFrame) -> pd.DataFrame:
    payload = np.load(component_maps_path)
    maps = np.asarray(payload["component_maps"], dtype=float)
    lat = np.asarray(payload["lat"], dtype=float) if "lat" in payload else np.linspace(-90.0, 90.0, maps.shape[0])
    lon = np.asarray(payload["lon"], dtype=float) if "lon" in payload else np.linspace(0.0, 360.0, maps.shape[1], endpoint=False)
    lon = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    lon = lon[order]
    maps = maps[:, order, :]
    rows: list[dict[str, float | int | str]] = []
    for row in scores.itertuples(index=False):
        local = int(row.component_index)
        center_lon, center_lat = component_center(maps[..., local], lat, lon)
        rows.append(
            {
                "component": str(row.component),
                "local": local,
                "paper": int(row.paper_component),
                "lon": center_lon,
                "lat": center_lat,
                "gateway_phi_eid": float(row.gateway_phi_eid),
                "gateway_phi_eid_pos": float(row.gateway_phi_eid_pos),
                "broadcast_redundancy": float(row.broadcast_redundancy),
                "broadcast_redundancy_pos": float(row.broadcast_redundancy_pos),
            }
        )
    return pd.DataFrame(rows)


def _label_subset(nodes: pd.DataFrame) -> pd.DataFrame:
    ids = set(nodes.nlargest(10, "gateway_phi_eid")["local"].astype(int))
    ids |= set(nodes.nlargest(10, "broadcast_redundancy")["local"].astype(int))
    return nodes[nodes["local"].astype(int).isin(ids)].copy()


def _signed_norm(values: np.ndarray) -> mpl.colors.Normalize:
    vmax = max(float(np.nanmax(np.abs(values))), 1.0e-12)
    return mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def plot_metric_map(nodes: pd.DataFrame, output: Path, *, save_svg: bool = True) -> dict[str, float]:
    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))
    fig = plt.figure(figsize=(10.8, 4.35), constrained_layout=True)
    axes = [fig.add_subplot(1, 2, 1, projection="mollweide"), fig.add_subplot(1, 2, 2, projection="mollweide")]
    for ax in axes:
        draw_world(ax, land, coastlines)
        add_geographic_ticks(ax)

    labels = _label_subset(nodes)
    phi_norm = _signed_norm(nodes["gateway_phi_eid"].to_numpy(dtype=float))
    br_norm = _signed_norm(nodes["broadcast_redundancy"].to_numpy(dtype=float))
    cmap = mpl.colormaps["PRGn"]
    for ax, column, norm, title in (
        (axes[0], "gateway_phi_eid", phi_norm, "Gateway phi"),
        (axes[1], "broadcast_redundancy", br_norm, "Broadcast redundancy"),
    ):
        values = nodes[column].to_numpy(dtype=float)
        sizes = 95.0 + 310.0 * np.clip(np.abs(values) / max(float(np.nanmax(np.abs(values))), 1.0e-12), 0.0, 1.0)
        ax.scatter(
            np.radians(nodes["lon"].to_numpy()),
            np.radians(nodes["lat"].to_numpy()),
            s=sizes,
            c=values,
            cmap=cmap,
            norm=norm,
            edgecolors="#222222",
            linewidths=0.26,
            alpha=0.95,
            zorder=4,
        )
        add_labels(ax, labels)
        ax.text(0.5, 1.06, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")
    axes[0].text(-0.08, 1.06, "a", transform=axes[0].transAxes, fontsize=16, fontweight="bold")
    axes[1].text(-0.08, 1.06, "b", transform=axes[1].transAxes, fontsize=16, fontweight="bold")
    for ax, norm, label in (
        (axes[0], phi_norm, r"$\phi^{EID}(T)$"),
        (axes[1], br_norm, "BR(S)"),
    ):
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax, location="bottom", shrink=0.62, pad=0.08, aspect=24)
        cbar.set_label(label)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if save_svg:
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "gateway_phi_abs_vmax": max(float(np.nanmax(np.abs(nodes["gateway_phi_eid"].to_numpy(dtype=float)))), 1.0e-12),
        "broadcast_abs_vmax": max(float(np.nanmax(np.abs(nodes["broadcast_redundancy"].to_numpy(dtype=float)))), 1.0e-12),
        "labeled_nodes": int(len(_label_subset(nodes))),
    }


def write_summary(scores: pd.DataFrame, manifest: dict[str, object], output: Path) -> None:
    top_phi = scores.sort_values("gateway_phi_eid", ascending=False).head(12)
    top_br = scores.sort_values("broadcast_redundancy", ascending=False).head(12)
    neg_phi = int((scores["gateway_phi_eid"] < 0.0).sum())
    neg_br = int((scores["broadcast_redundancy"] < 0.0).sum())

    def table(frame: pd.DataFrame, value_col: str) -> str:
        lines = ["| rank | component | paper | value |", "|---:|---|---:|---:|"]
        for rank, row in enumerate(frame.itertuples(index=False), start=1):
            lines.append(f"| {rank} | {row.component} | {int(row.paper_component)} | {float(getattr(row, value_col)):.6g} |")
        return "\n".join(lines)

    text = "\n\n".join(
        [
            "# Runge 1948-2026 gateway phi and broadcast redundancy",
            "All terms use the same Gaussian log-det estimator on the intervention predictions from the cached MLP ensemble. Self source-target terms are excluded unless `include_self=true` in the manifest.",
            f"- estimator: `{manifest.get('estimator', 'unknown')}`",
            f"- intervention samples: `{manifest.get('intervention_samples', 'unknown')}`",
            f"- nodes with negative gateway phi: `{neg_phi}` / `{len(scores)}`",
            f"- nodes with negative broadcast redundancy: `{neg_br}` / `{len(scores)}`",
            "## Top gateway phi",
            table(top_phi, "gateway_phi_eid"),
            "## Top broadcast redundancy",
            table(top_br, "broadcast_redundancy"),
            "",
        ]
    )
    output.write_text(text, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--component-scores", default=str(DEFAULT_COMPONENT_SCORES))
    parser.add_argument("--component-maps", default=str(DEFAULT_COMPONENT_MAPS))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--fig-dir", default=str(DEFAULT_FIG_DIR))
    parser.add_argument("--pairwise-matrix", default=str(DEFAULT_RUN_ROOT / "results/runge/peid_hypergraph/pairwise_ei_matrix.csv"))
    parser.add_argument("--pairwise-gateway", default=str(DEFAULT_RUN_ROOT / "results/runge/pairwise_mlp_tm_ei_path_effects/gateway_scores.csv"))
    parser.add_argument("--pairwise-mediator", default=str(DEFAULT_RUN_ROOT / "results/runge/pairwise_mlp_tm_ei_path_effects/mediator_scores.csv"))
    parser.add_argument("--lag", type=int, default=4)
    parser.add_argument("--intervention-samples", type=int, default=4096)
    parser.add_argument("--gaussian-ridge", type=float, default=1.0e-6)
    parser.add_argument("--include-self", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scores, pairwise, manifest = load_or_compute_metrics(args)
    nodes = build_node_frame(Path(args.component_maps), scores)
    result_dir = Path(args.result_dir)
    fig_dir = Path(args.fig_dir)
    nodes.to_csv(result_dir / "gateway_broadcast_map_nodes.csv", index=False)
    figure_path = fig_dir / "gateway_phi_broadcast_redundancy_map.png"
    plot_limits = plot_metric_map(nodes, figure_path, save_svg=True)
    manifest = {**manifest, "figure_png": str(figure_path), "figure_svg": str(figure_path.with_suffix(".svg")), "figure_pdf": str(figure_path.with_suffix(".pdf")), "plot_limits": plot_limits}
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(scores, manifest, result_dir / "summary.md")
    print(json.dumps({"scores": str(result_dir / "gateway_broadcast_scores.csv"), "figure": str(figure_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
