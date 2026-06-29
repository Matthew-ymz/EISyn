#!/usr/bin/env python3
"""Compute original EI+Syn hyper metrics for the 1948-2026 Runge SLP run."""

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
DEFAULT_FULL_ORDER2 = DEFAULT_RUN_ROOT / "results/runge/scheme_a_full_order2"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_ROOT / "results/runge/original_hyper_metric_full_order2"
DEFAULT_FIG_DIR = DEFAULT_RUN_ROOT / "fig/runge/original_hyper_metric_full_order2"
DEFAULT_COMPONENT_MAPS = Path("results/runge_slp_daily_1948_2026_20260628/results/runge/2015_gateways/component_maps.npz")
DEFAULT_PAIRWISE_MATRIX = DEFAULT_RUN_ROOT / "results/runge/peid_hypergraph/pairwise_ei_matrix.csv"
DEFAULT_PAIRWISE_GATEWAY = DEFAULT_RUN_ROOT / "results/runge/pairwise_mlp_tm_ei_path_effects/gateway_scores.csv"
DEFAULT_PAIRWISE_MEDIATOR = DEFAULT_RUN_ROOT / "results/runge/pairwise_mlp_tm_ei_path_effects/mediator_scores.csv"


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


def load_component_maps(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(path)
    maps = np.asarray(payload["component_maps"], dtype=float)
    lat = np.asarray(payload["lat"], dtype=float) if "lat" in payload else np.linspace(-90.0, 90.0, maps.shape[0])
    lon = np.asarray(payload["lon"], dtype=float) if "lon" in payload else np.linspace(0.0, 360.0, maps.shape[1], endpoint=False)
    lon = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    return maps[:, order, :], lat, lon[order]


def compute_scores(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairwise_frame = pd.read_csv(args.pairwise_matrix, index_col=0)
    names = [str(name) for name in pairwise_frame.index.tolist()]
    pairwise = pairwise_frame.to_numpy(dtype=float)
    np.fill_diagonal(pairwise, 0.0)
    n = pairwise.shape[0]
    denom = max(1, n - 1)

    pairwise_gateway = pd.read_csv(args.pairwise_gateway)
    path_ace_lookup = {int(row.component_index): float(row.ace) for row in pairwise_gateway.itertuples(index=False)}
    path_acs_lookup = {int(row.component_index): float(row.acs) for row in pairwise_gateway.itertuples(index=False)}

    use_hyperedges = bool(args.peid_hyperedges)
    joint = pd.read_csv(args.peid_hyperedges if use_hyperedges else args.joint_order2)
    if use_hyperedges:
        joint = joint[joint["order"].astype(int) == 2].copy()
        total_order2 = int(len(joint))
        if args.significance_z is not None:
            joint = joint[joint["z"].abs() >= float(args.significance_z)].copy()
        joint["source_a_index_calc"] = joint["subset_str"].astype(str).str.split(",").str[0].astype(int)
        joint["source_b_index_calc"] = joint["subset_str"].astype(str).str.split(",").str[1].astype(int)
        joint["delta2_calc"] = joint["delta_K"].astype(float)
    else:
        total_order2 = int(len(joint))
    source_syn = np.zeros(n, dtype=float)
    target_syn = np.zeros(n, dtype=float)
    delta_values = []
    rows = []
    for row in joint.itertuples(index=False):
        if use_hyperedges:
            a = int(row.source_a_index_calc)
            b = int(row.source_b_index_calc)
            target = int(row.target_index)
            delta = float(row.delta2_calc)
        else:
            a = int(row.source_a_index)
            b = int(row.source_b_index)
            target = int(row.target_index)
            delta = float(row.ei_joint) - float(pairwise[a, target]) - float(pairwise[b, target])
        syn_abs = abs(delta)
        source_share = syn_abs / 2.0
        source_syn[a] += source_share
        source_syn[b] += source_share
        target_syn[target] += syn_abs
        delta_values.append(delta)
        rows.append(
            {
                "order": 2,
                "source_a_index": a,
                "source_b_index": b,
                "target_index": target,
                "target": names[target],
                "ei_joint": float(row.ei_joint),
                "ei_source_a": float(pairwise[a, target]),
                "ei_source_b": float(pairwise[b, target]),
                "delta2_signed": delta,
                "syn_eid_abs": syn_abs,
                "source_member_share": source_share,
                "z": float(getattr(row, "z", np.nan)),
                "p_empirical": float(getattr(row, "p_empirical", np.nan)),
            }
        )
    hyperedges = pd.DataFrame(rows)
    hyperedges.to_csv(output_dir / "original_hyperedges_full_order2.csv", index=False)

    pairwise_ace = np.sum(np.abs(pairwise), axis=1) / denom
    pairwise_acs = np.sum(np.abs(pairwise), axis=0) / denom
    source_syn_score = source_syn / denom
    target_syn_score = target_syn / denom

    pairwise_mediator = pd.read_csv(args.pairwise_mediator)
    pair_amce_lookup = {
        int(row.component_index): float(row.amce) for row in pairwise_mediator.itertuples(index=False)
    }
    pairwise_amce = np.asarray([pair_amce_lookup.get(i, 0.0) for i in range(n)], dtype=float)

    baseline_gateway = pd.DataFrame(
        {
            "component": names,
            "component_index": np.arange(n),
            "ace_ei": np.asarray([path_ace_lookup.get(i, pairwise_ace[i]) for i in range(n)], dtype=float),
            "acs_ei": np.asarray([path_acs_lookup.get(i, pairwise_acs[i]) for i in range(n)], dtype=float),
            "ace_total": np.asarray([path_ace_lookup.get(i, pairwise_ace[i]) for i in range(n)], dtype=float),
            "acs_total": np.asarray([path_acs_lookup.get(i, pairwise_acs[i]) for i in range(n)], dtype=float),
        }
    )
    baseline_mediator = pd.DataFrame(
        {
            "component": names,
            "component_index": np.arange(n),
            "amce_pairwise": pairwise_amce,
            "amce_syn_order2": np.zeros(n, dtype=float),
            "amce_total": pairwise_amce,
        }
    )
    hyper_gateway = pd.DataFrame(
        {
            "component": names,
            "component_index": np.arange(n),
            "ace_ei": pairwise_ace,
            "ace_syn_order2": source_syn_score,
            "ace_total": pairwise_ace + source_syn_score,
            "acs_ei": pairwise_acs,
            "acs_syn_order2": target_syn_score,
            "acs_total": pairwise_acs + target_syn_score,
        }
    )
    hyper_mediator = pd.DataFrame(
        {
            "component": names,
            "component_index": np.arange(n),
            "amce_pairwise": pairwise_amce,
            "amce_syn_order2": source_syn_score,
            "amce_total": pairwise_amce + source_syn_score,
        }
    )
    for frame, col in (
        (baseline_gateway, "ace_total"),
        (baseline_mediator, "amce_total"),
        (hyper_gateway, "ace_total"),
        (hyper_mediator, "amce_total"),
    ):
        rank_col = col.replace("_total", "_rank")
        frame[rank_col] = frame[col].rank(ascending=False, method="min").astype(int)

    baseline_gateway.sort_values("ace_total", ascending=False).to_csv(output_dir / "pairwise_direct_gateway_scores.csv", index=False)
    baseline_mediator.sort_values("amce_total", ascending=False).to_csv(output_dir / "pairwise_path_mediator_scores.csv", index=False)
    hyper_gateway.sort_values("ace_total", ascending=False).to_csv(output_dir / "original_hyper_gateway_scores.csv", index=False)
    hyper_mediator.sort_values("amce_total", ascending=False).to_csv(output_dir / "original_hyper_mediator_scores.csv", index=False)

    delta_arr = np.asarray(delta_values, dtype=float)
    manifest = {
        "metric": "original_hyper_ei_plus_syn",
        "n_components": n,
        "order2_hyperedge_count": int(len(hyperedges)),
        "order2_hyperedge_total_before_gate": total_order2,
        "expected_order2_hyperedge_count": int(n * (n - 1) * (n - 2) / 2) if not use_hyperedges else None,
        "significance_gate": f"abs(z)>={float(args.significance_z)}" if use_hyperedges and args.significance_z is not None else "none_full_scan_no_null_z",
        "syn_definition": "abs(delta2), delta2=EI({a,b}->j)-EI(a->j)-EI(b->j)",
        "source_syn_divisor_per_hyperedge": 2,
        "target_syn_divisor_per_hyperedge": 1,
        "node_denominator": denom,
        "delta2_signed": {
            "min": float(delta_arr.min()),
            "max": float(delta_arr.max()),
            "mean": float(delta_arr.mean()),
            "negative_count": int(np.count_nonzero(delta_arr < 0.0)),
            "positive_count": int(np.count_nonzero(delta_arr > 0.0)),
        },
        "top_original_hyper_gateway": hyper_gateway.sort_values("ace_total", ascending=False).head(10).to_dict("records"),
        "top_original_hyper_mediator": hyper_mediator.sort_values("amce_total", ascending=False).head(10).to_dict("records"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "names": names,
        "baseline_gateway": baseline_gateway,
        "baseline_mediator": baseline_mediator,
        "hyper_gateway": hyper_gateway,
        "hyper_mediator": hyper_mediator,
        "manifest": manifest,
    }


def build_nodes(component_maps_path: Path, gateway: pd.DataFrame, mediator: pd.DataFrame) -> pd.DataFrame:
    maps, lat, lon = load_component_maps(component_maps_path)
    merged = gateway.merge(mediator[["component_index", "amce_total"]], on="component_index", how="left")
    rows = []
    for row in merged.itertuples(index=False):
        idx = int(row.component_index)
        center_lon, center_lat = component_center(maps[..., idx], lat, lon)
        rows.append(
            {
                "local": idx,
                "paper": local_to_paper(idx),
                "lon": center_lon,
                "lat": center_lat,
                "ace": float(row.ace_total),
                "acs": float(row.acs_total),
                "amce": float(row.amce_total),
            }
        )
    return pd.DataFrame(rows)


def draw_ace_acs(ax: plt.Axes, nodes: pd.DataFrame, norm: mpl.colors.Normalize, label_nodes: pd.DataFrame) -> None:
    lon = np.radians(nodes["lon"].to_numpy())
    lat = np.radians(nodes["lat"].to_numpy())
    ax.scatter(lon, lat, s=330, c=nodes["ace"], cmap="OrRd", norm=norm, edgecolors="#3d1d0d", linewidths=0.32, alpha=0.96, zorder=4)
    ax.scatter(lon, lat, s=175, c=nodes["acs"], cmap="OrRd", norm=norm, edgecolors="none", alpha=0.98, zorder=5)
    add_labels(ax, label_nodes)


def draw_amce(ax: plt.Axes, nodes: pd.DataFrame, norm: mpl.colors.Normalize, label_nodes: pd.DataFrame) -> None:
    values = nodes["amce"].to_numpy(dtype=float)
    vmax = max(float(np.nanmax(np.abs(values))), 1.0e-12)
    sizes = 160.0 + 350.0 * np.clip(np.abs(values) / vmax, 0.0, 1.0)
    ax.scatter(
        np.radians(nodes["lon"].to_numpy()),
        np.radians(nodes["lat"].to_numpy()),
        s=sizes,
        c=values,
        cmap="YlGnBu",
        norm=norm,
        edgecolors="#222222",
        linewidths=0.28,
        alpha=0.96,
        zorder=4,
    )
    add_labels(ax, label_nodes)


def plot_map(baseline_nodes: pd.DataFrame, hyper_nodes: pd.DataFrame, output: Path) -> dict[str, float]:
    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))
    fig = plt.figure(figsize=(10.8, 8.15), constrained_layout=True)
    axes = [
        fig.add_subplot(2, 2, 1, projection="mollweide"),
        fig.add_subplot(2, 2, 2, projection="mollweide"),
        fig.add_subplot(2, 2, 3, projection="mollweide"),
        fig.add_subplot(2, 2, 4, projection="mollweide"),
    ]
    for ax in axes:
        draw_world(ax, land, coastlines)
        add_geographic_ticks(ax)
    base_label_ids = set(baseline_nodes.nlargest(8, "ace")["local"].astype(int)) | set(baseline_nodes.nlargest(8, "acs")["local"].astype(int)) | set(baseline_nodes.nlargest(8, "amce")["local"].astype(int))
    hyper_label_ids = set(hyper_nodes.nlargest(8, "ace")["local"].astype(int)) | set(hyper_nodes.nlargest(8, "acs")["local"].astype(int)) | set(hyper_nodes.nlargest(8, "amce")["local"].astype(int))
    base_labels = baseline_nodes[baseline_nodes["local"].astype(int).isin(base_label_ids)]
    hyper_labels = hyper_nodes[hyper_nodes["local"].astype(int).isin(hyper_label_ids)]
    base_ace_vmax = max(float(baseline_nodes[["ace", "acs"]].to_numpy().max()), 1.0e-12)
    hyper_ace_vmax = max(float(hyper_nodes[["ace", "acs"]].to_numpy().max()), 1.0e-12)
    base_amce_vmax = max(float(np.nanmax(np.abs(baseline_nodes["amce"].to_numpy(dtype=float)))), 1.0e-12)
    hyper_amce_vmax = max(float(np.nanmax(np.abs(hyper_nodes["amce"].to_numpy(dtype=float)))), 1.0e-12)
    base_ace_norm = mpl.colors.Normalize(vmin=0.0, vmax=base_ace_vmax)
    hyper_ace_norm = mpl.colors.Normalize(vmin=0.0, vmax=hyper_ace_vmax)
    base_amce_norm = mpl.colors.Normalize(vmin=0.0, vmax=base_amce_vmax)
    hyper_amce_norm = mpl.colors.Normalize(vmin=0.0, vmax=hyper_amce_vmax)
    draw_ace_acs(axes[0], baseline_nodes, base_ace_norm, base_labels)
    draw_amce(axes[1], baseline_nodes, base_amce_norm, base_labels)
    draw_ace_acs(axes[2], hyper_nodes, hyper_ace_norm, hyper_labels)
    draw_amce(axes[3], hyper_nodes, hyper_amce_norm, hyper_labels)
    labels = ["a", "b", "c", "d"]
    titles = ["Pairwise EI baseline", "Pairwise path AMCE", "Original EI + Syn", "Original Hyper-AMCE"]
    for label, title, ax in zip(labels, titles, axes, strict=True):
        ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=16, fontweight="bold")
        ax.text(0.5, 1.06, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")
    for ax, norm, cmap, label in [
        (axes[0], base_ace_norm, "OrRd", "Pairwise ACS (inner) and ACE (outer)"),
        (axes[1], base_amce_norm, "YlGnBu", "Pairwise path AMCE"),
        (axes[2], hyper_ace_norm, "OrRd", "Original ACS/ACE: EI + Syn"),
        (axes[3], hyper_amce_norm, "YlGnBu", "Original Hyper-AMCE: AMCE + Syn"),
    ]:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colormaps[cmap])
        cbar = fig.colorbar(sm, ax=ax, location="bottom", shrink=0.68, pad=0.07, aspect=24)
        cbar.set_label(label)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "baseline_ace_acs_vmax": base_ace_vmax,
        "hyper_ace_acs_vmax": hyper_ace_vmax,
        "baseline_amce_vmax": base_amce_vmax,
        "hyper_amce_vmax": hyper_amce_vmax,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise-matrix", default=str(DEFAULT_PAIRWISE_MATRIX))
    parser.add_argument("--pairwise-gateway", default=str(DEFAULT_PAIRWISE_GATEWAY))
    parser.add_argument("--joint-order2", default=str(DEFAULT_FULL_ORDER2 / "joint_order2_full.csv"))
    parser.add_argument("--peid-hyperedges", default="")
    parser.add_argument("--significance-z", type=float, default=None)
    parser.add_argument("--pairwise-mediator", default=str(DEFAULT_PAIRWISE_MEDIATOR))
    parser.add_argument("--component-maps", default=str(DEFAULT_COMPONENT_MAPS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fig-dir", default=str(DEFAULT_FIG_DIR))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = compute_scores(args)
    baseline_nodes = build_nodes(Path(args.component_maps), result["baseline_gateway"], result["baseline_mediator"])
    hyper_nodes = build_nodes(Path(args.component_maps), result["hyper_gateway"], result["hyper_mediator"])
    output_dir = Path(args.output_dir)
    baseline_nodes.to_csv(output_dir / "pairwise_baseline_map_nodes.csv", index=False)
    hyper_nodes.to_csv(output_dir / "original_hyper_map_nodes.csv", index=False)
    fig_path = Path(args.fig_dir) / "original_hyper_metric_full_order2_map.png"
    plot_limits = plot_map(baseline_nodes, hyper_nodes, fig_path)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "figure_png": str(fig_path),
            "figure_svg": str(fig_path.with_suffix(".svg")),
            "figure_pdf": str(fig_path.with_suffix(".pdf")),
            "plot_limits": plot_limits,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
