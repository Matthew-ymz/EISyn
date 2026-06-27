#!/usr/bin/env python3
"""Runge scheme A: conditioned one-step EI with original walk-sum path effects."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_runge_gateway_mediator_map import (  # noqa: E402
    COASTLINE_URL,
    DEFAULT_COMPONENT_MAPS,
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
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from scripts.run_runge_multistep_conditioned_ei import (  # noqa: E402
    overlap_summary,
    rank_correlations,
)
from scripts.run_runge_pairwise_mlp_ei import (  # noqa: E402
    compute_ei_path_effects,
    sparsify_ei_graph,
)


DEFAULT_INPUT = ROOT / "results" / "runge" / "multistep_conditioned_ei" / "horizon_001" / "conditioned_ei_signed.npy"
DEFAULT_NAMES = ROOT / "results" / "runge" / "multistep_conditioned_ei" / "horizon_001" / "pairwise_ei_matrix.csv"
DEFAULT_RESULT_DIR = ROOT / "results" / "runge" / "scheme_a_conditioned_path_effects"
DEFAULT_REFERENCE_GATEWAY = ROOT / "results" / "runge" / "peid_hypergraph" / "hyper_gateway_scores.csv"
DEFAULT_REFERENCE_MEDIATOR = ROOT / "results" / "runge" / "peid_hypergraph" / "hyper_mediator_scores.csv"
DEFAULT_REFERENCE_HYPEREDGES = ROOT / "results" / "runge" / "peid_hypergraph" / "peid_hyperedges.csv"
DEFAULT_FIGURE = ROOT / "docs" / "reports" / "assets" / "runge_scheme_a_conditioned_path_map.png"
DEFAULT_COMPARISON_FIGURE = ROOT / "docs" / "reports" / "assets" / "runge_scheme_a_vs_peid_hyper_map.png"
DEFAULT_REPORT = ROOT / "docs" / "reports" / "Runge_Multistep_EI_Path_Design.md"


def load_names(path: Path, n_components: int) -> list[str]:
    frame = pd.read_csv(path, index_col=0)
    names = [str(value) for value in frame.index.tolist()]
    if len(names) != int(n_components):
        raise ValueError("names matrix index length does not match conditioned EI matrix.")
    return names


def save_matrix_csv(matrix: np.ndarray, names: list[str], path: Path) -> None:
    pd.DataFrame(np.asarray(matrix, dtype=float), index=names, columns=names).to_csv(path)


def markdown_top_table(gateway: pd.DataFrame, mediator: pd.DataFrame, k: int = 10) -> str:
    rows = ["| Rank | ACE | ACS | AMCE |", "|---:|---:|---:|---:|"]
    for rank in range(min(int(k), len(gateway), len(mediator))):
        ace = gateway.sort_values("ace", ascending=False).iloc[rank]
        acs = gateway.sort_values("acs", ascending=False).iloc[rank]
        amce = mediator.sort_values("amce", ascending=False).iloc[rank]
        ace_id = int(ace.get("paper_component", ace["component_index"]))
        acs_id = int(acs.get("paper_component", acs["component_index"]))
        amce_id = int(amce.get("paper_component", amce["component_index"]))
        rows.append(
            f"| {rank + 1} | {ace_id} ({float(ace['ace']):.4g}) | {acs_id} ({float(acs['acs']):.4g}) | {amce_id} ({float(amce['amce']):.4g}) |"
        )
    return "\n".join(rows)


def append_report_section(
    report_path: Path,
    *,
    figure_path: Path,
    gateway: pd.DataFrame,
    mediator: pd.DataFrame,
    correlations: dict[str, dict[str, float]] | None,
    overlaps: dict[str, dict[str, int]] | None,
    manifest: dict[str, Any],
) -> None:
    start = "<!-- scheme-a-conditioned-path-results:start -->"
    end = "<!-- scheme-a-conditioned-path-results:end -->"
    rel_figure = figure_path.relative_to(report_path.parent)
    comparison_figure = manifest.get("comparison_figure")
    comparison_image = ""
    if comparison_figure:
        rel_comparison = Path(str(comparison_figure)).relative_to(report_path.parent)
        comparison_image = f"\nScheme A 与旧二阶 PEID Hyper 的直接对比如下：\n\n![Runge scheme A vs PEID hyper map]({rel_comparison.as_posix()})\n"
    comparison = ""
    if correlations is not None and overlaps is not None:
        lines = ["| Metric | Spearman | Kendall | top-5 overlap | top-10 overlap |", "|---|---:|---:|---:|---:|"]
        for metric in ("ace", "acs", "amce"):
            lines.append(
                f"| {metric.upper()} | {correlations[metric]['spearman']:.3f} | {correlations[metric]['kendall']:.3f} | "
                f"{overlaps[metric]['top5_overlap']}/5 | {overlaps[metric]['top10_overlap']}/10 |"
            )
        comparison = "\n与当前 `peid_hypergraph` 的二阶 PEID Hyper 排名相比：\n\n" + "\n".join(lines) + "\n"
    section = f"""
{start}

### 实验结果：方案 A 平均条件 EI 构图

本次结果只改变直接构图边权：用 `horizon_001` 已估计的 signed 平均条件 EI 矩阵作为 \\(\\bar E^{{(2)}}\\)，再按原 Runge path-effect 流程做正部截断、source-top-{manifest['graph_topk']} 稀疏化、谱缩放和 walk-sum 路径汇总。该结果仍是 one-step conditioned EI graph-walk path score，不是方案 B 的 MLP 自迭代多步 EI。

![Runge scheme A conditioned path map]({rel_figure.as_posix()})
{comparison_image}

Top 节点如下，括号内为对应指标值：

{markdown_top_table(gateway, mediator)}
{comparison}
谱缩放因子为 `{manifest['scale_factor']:.6g}`；正部稀疏直接边数量为 `{manifest['n_direct_edges']}`。配色使用本图数据自适应上限：ACE/ACS 色标上限 `{manifest['plot_color_limits']['ace_acs_vmax']:.4g}`，AMCE 色标上限 `{manifest['plot_color_limits']['amce_vmax']:.4g}`，避免沿用旧图固定上限后颜色分布被压平。另输出 Scheme A 与旧二阶 PEID Hyper 的 2x2 对比图：`{Path(manifest['comparison_figure']).name if manifest.get('comparison_figure') else '未生成'}`。

{end}
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    old = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if start in old and end in old:
        prefix = old.split(start)[0].rstrip()
        suffix = old.split(end, 1)[1].lstrip()
        new_text = f"{prefix}\n\n{section.strip()}\n\n{suffix}".rstrip() + "\n"
    else:
        new_text = old.rstrip() + "\n\n" + section.strip() + "\n"
    report_path.write_text(new_text, encoding="utf-8")


def add_paper_component(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "component" in out.columns:
        out["component_name"] = out["component"]
    out["component"] = out["component_index"].astype(int)
    out["paper_component"] = out["component_index"].map(lambda value: int(local_to_paper(int(value))))
    return out


def build_scheme_node_frame(component_maps: np.ndarray, gateway: pd.DataFrame, mediator: pd.DataFrame) -> pd.DataFrame:
    lat = np.linspace(-90.0, 90.0, component_maps.shape[0])
    lon = np.linspace(0.0, 360.0, component_maps.shape[1], endpoint=False)
    lon = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    maps = component_maps[:, order, :]
    lon = lon[order]
    mediator_frame = mediator.copy()
    if "mediated_fraction" not in mediator_frame.columns:
        total_amce = float(mediator_frame["amce"].sum()) if "amce" in mediator_frame.columns else 0.0
        mediator_frame["mediated_fraction"] = mediator_frame["amce"] / total_amce if total_amce > 1.0e-12 else 0.0
    frame = gateway.merge(mediator_frame[["component_index", "amce", "mediated_fraction"]], on="component_index", how="left")
    rows: list[dict[str, float | int]] = []
    for row in frame.itertuples(index=False):
        local = int(row.component_index)
        center_lon, center_lat = component_center(maps[..., local], lat, lon)
        rows.append(
            {
                "local": local,
                "paper": local_to_paper(local),
                "lon": center_lon,
                "lat": center_lat,
                "ace": float(row.ace),
                "acs": float(row.acs),
                "amce": float(row.amce),
                "mediated_fraction": float(row.mediated_fraction),
            }
        )
    return pd.DataFrame(rows)


def build_hyper_reference_frames(
    gateway_path: Path,
    mediator_path: Path,
    hyperedges_path: Path,
    *,
    n_components: int,
    significance_z: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scripts.plot_runge_peid_synergy_map import aggregate_hyper_acs

    gateway = pd.read_csv(gateway_path)
    mediator = pd.read_csv(mediator_path)
    hyperedges = pd.read_csv(hyperedges_path)
    acs = aggregate_hyper_acs(hyperedges, n_components=int(n_components), significance_z=float(significance_z))
    ref_gateway = gateway.merge(acs, on="component_index", how="left")
    ref_gateway = ref_gateway.rename(columns={"hyper_ace_total": "ace", "hyper_acs_total": "acs"})
    ref_mediator = mediator.rename(columns={"hyper_amce_total": "amce"})
    return ref_gateway, ref_mediator


def build_hyper_node_frame(
    component_maps: np.ndarray,
    gateway_path: Path,
    mediator_path: Path,
    hyperedges_path: Path,
    *,
    significance_z: float = 2.0,
) -> pd.DataFrame:
    ref_gateway, ref_mediator = build_hyper_reference_frames(
        gateway_path,
        mediator_path,
        hyperedges_path,
        n_components=component_maps.shape[2],
        significance_z=float(significance_z),
    )
    ref_gateway = add_paper_component(ref_gateway)
    ref_mediator = add_paper_component(ref_mediator)
    return build_scheme_node_frame(component_maps, ref_gateway, ref_mediator)


def color_limits(nodes: pd.DataFrame) -> dict[str, float]:
    return {
        "ace_acs_vmax": max(float(nodes[["ace", "acs"]].to_numpy(dtype=float).max()), 1.0e-12),
        "amce_vmax": max(float(nodes["amce"].to_numpy(dtype=float).max()), 1.0e-12),
    }


def draw_ace_acs(ax: plt.Axes, nodes: pd.DataFrame, norm: mpl.colors.Normalize) -> None:
    cmap = mpl.colormaps["OrRd"]
    lon = np.radians(nodes["lon"].to_numpy())
    lat = np.radians(nodes["lat"].to_numpy())
    ax.scatter(lon, lat, s=360, c=nodes["ace"], cmap=cmap, norm=norm, edgecolors="#3d1d0d", linewidths=0.32, alpha=0.96, zorder=4)
    ax.scatter(lon, lat, s=190, c=nodes["acs"], cmap=cmap, norm=norm, edgecolors="none", alpha=0.98, zorder=5)
    add_labels(ax, nodes)


def draw_amce(ax: plt.Axes, nodes: pd.DataFrame, norm: mpl.colors.Normalize) -> None:
    cmap = mpl.colormaps["Greens"]
    values = nodes["amce"].to_numpy(dtype=float)
    sizes = 185.0 + 360.0 * np.clip(values / max(float(np.nanmax(values)), 1.0e-12), 0.0, 1.0)
    ax.scatter(np.radians(nodes["lon"].to_numpy()), np.radians(nodes["lat"].to_numpy()), s=sizes, c=values, cmap=cmap, norm=norm, edgecolors="#173317", linewidths=0.32, alpha=0.96, zorder=4)
    add_labels(ax, nodes)


def add_colorbar(fig: plt.Figure, ax: plt.Axes, norm: mpl.colors.Normalize, cmap_name: str, label: str) -> None:
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colormaps[cmap_name])
    cbar = fig.colorbar(sm, ax=ax, location="bottom", shrink=0.62, pad=0.08, aspect=22)
    cbar.set_label(label)


def plot_scheme_a_map(nodes: pd.DataFrame, output: Path, *, save_svg: bool = True) -> dict[str, float]:
    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))
    fig = plt.figure(figsize=(10.4, 4.25), constrained_layout=True)
    axes = [fig.add_subplot(1, 2, 1, projection="mollweide"), fig.add_subplot(1, 2, 2, projection="mollweide")]
    for ax in axes:
        draw_world(ax, land, coastlines)
        add_geographic_ticks(ax)
    limits = color_limits(nodes)
    ace_norm = mpl.colors.Normalize(vmin=0.0, vmax=limits["ace_acs_vmax"])
    amce_norm = mpl.colors.Normalize(vmin=0.0, vmax=limits["amce_vmax"])
    draw_ace_acs(axes[0], nodes, ace_norm)
    draw_amce(axes[1], nodes, amce_norm)
    axes[0].text(-0.08, 1.06, "a", transform=axes[0].transAxes, fontsize=16, fontweight="bold")
    axes[1].text(-0.08, 1.06, "b", transform=axes[1].transAxes, fontsize=16, fontweight="bold")
    axes[0].text(0.5, 1.06, "Scheme A conditioned path", transform=axes[0].transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")
    axes[1].text(0.5, 1.06, "Scheme A conditioned path", transform=axes[1].transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")
    add_colorbar(fig, axes[0], ace_norm, "OrRd", "Scheme-A ACS (inner node) and ACE (outer ring)")
    add_colorbar(fig, axes[1], amce_norm, "Greens", "Scheme-A AMCE")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if save_svg:
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return limits


def plot_scheme_a_vs_hyper_map(scheme_nodes: pd.DataFrame, hyper_nodes: pd.DataFrame, output: Path, *, save_svg: bool = True) -> None:
    land = extract_polygons(load_geojson(LAND_URL))
    coastlines = extract_lines(load_geojson(COASTLINE_URL))
    fig = plt.figure(figsize=(10.6, 8.0), constrained_layout=True)
    axes = [
        fig.add_subplot(2, 2, 1, projection="mollweide"),
        fig.add_subplot(2, 2, 2, projection="mollweide"),
        fig.add_subplot(2, 2, 3, projection="mollweide"),
        fig.add_subplot(2, 2, 4, projection="mollweide"),
    ]
    for ax in axes:
        draw_world(ax, land, coastlines)
        add_geographic_ticks(ax)
    scheme_limits = color_limits(scheme_nodes)
    hyper_limits = color_limits(hyper_nodes)
    scheme_ace_norm = mpl.colors.Normalize(vmin=0.0, vmax=scheme_limits["ace_acs_vmax"])
    scheme_amce_norm = mpl.colors.Normalize(vmin=0.0, vmax=scheme_limits["amce_vmax"])
    hyper_ace_norm = mpl.colors.Normalize(vmin=0.0, vmax=hyper_limits["ace_acs_vmax"])
    hyper_amce_norm = mpl.colors.Normalize(vmin=0.0, vmax=hyper_limits["amce_vmax"])
    draw_ace_acs(axes[0], scheme_nodes, scheme_ace_norm)
    draw_amce(axes[1], scheme_nodes, scheme_amce_norm)
    draw_ace_acs(axes[2], hyper_nodes, hyper_ace_norm)
    draw_amce(axes[3], hyper_nodes, hyper_amce_norm)
    for label, ax in zip(["a", "b", "c", "d"], axes, strict=True):
        ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=16, fontweight="bold")
    for ax in axes[:2]:
        ax.text(0.5, 1.06, "Scheme A conditioned path", transform=ax.transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")
    for ax in axes[2:]:
        ax.text(0.5, 1.06, "Previous second-order PEID", transform=ax.transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")
    add_colorbar(fig, axes[0], scheme_ace_norm, "OrRd", "Scheme-A ACS (inner) and ACE (outer)")
    add_colorbar(fig, axes[1], scheme_amce_norm, "Greens", "Scheme-A AMCE")
    add_colorbar(fig, axes[2], hyper_ace_norm, "OrRd", "Hyper-ACS (inner) and Hyper-ACE (outer)")
    add_colorbar(fig, axes[3], hyper_amce_norm, "Greens", "Hyper-AMCE")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if save_svg:
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def run_scheme_a(
    *,
    input_matrix: Path,
    names_matrix: Path,
    output_dir: Path,
    graph_topk: int,
    graph_sparsify: str = "source_topk",
    path_alpha: float,
    max_path_length: int,
    reference_gateway: Path | None,
    reference_mediator: Path | None,
    reference_hyperedges: Path | None,
    component_maps: Path | None,
    figure_output: Path | None,
    comparison_figure_output: Path | None,
    report: Path | None,
) -> dict[str, Any]:
    conditioned = np.asarray(np.load(input_matrix), dtype=float)
    if conditioned.ndim != 2 or conditioned.shape[0] != conditioned.shape[1]:
        raise ValueError("input_matrix must contain a square 2D matrix.")
    names = load_names(names_matrix, conditioned.shape[0])
    output_dir.mkdir(parents=True, exist_ok=True)

    if graph_sparsify not in {"none", "source_topk"}:
        raise ValueError("graph_sparsify must be 'none' or 'source_topk'.")
    direct = sparsify_ei_graph(conditioned, mode=str(graph_sparsify), topk=int(graph_topk))
    path_effects = compute_ei_path_effects(direct, names, path_alpha=float(path_alpha), max_path_length=int(max_path_length))
    gateway = add_paper_component(path_effects.gateway_scores)
    mediator = add_paper_component(path_effects.mediator_scores)

    save_matrix_csv(conditioned, names, output_dir / "conditioned_ei_matrix.csv")
    save_matrix_csv(direct, names, output_dir / "direct_ei_matrix.csv")
    save_matrix_csv(path_effects.scaled_direct_matrix, names, output_dir / "scaled_direct_effect_matrix.csv")
    save_matrix_csv(path_effects.total_matrix, names, output_dir / "total_effect_matrix.csv")
    path_effects.direct_effects.to_csv(output_dir / "direct_effects.csv", index=False)
    path_effects.total_effects.to_csv(output_dir / "total_effects.csv", index=False)
    path_effects.path_effects.to_csv(output_dir / "mediated_path_effects.csv", index=False)
    gateway.to_csv(output_dir / "gateway_scores.csv", index=False)
    mediator.to_csv(output_dir / "mediator_scores.csv", index=False)

    correlations = None
    overlaps = None
    reference_kind = "none"
    if reference_gateway is not None and reference_mediator is not None:
        if reference_hyperedges is not None:
            ref_gateway, ref_mediator = build_hyper_reference_frames(
                reference_gateway,
                reference_mediator,
                reference_hyperedges,
                n_components=conditioned.shape[0],
            )
            reference_kind = "peid_hypergraph"
        else:
            ref_gateway = pd.read_csv(reference_gateway)
            ref_mediator = pd.read_csv(reference_mediator)
            reference_kind = "path_effect"
        correlations = rank_correlations(gateway, mediator, ref_gateway, ref_mediator)
        overlaps = overlap_summary(gateway, mediator, ref_gateway, ref_mediator)

    figure_path = None
    comparison_figure_path = None
    plot_limits = {"ace_acs_vmax": None, "amce_vmax": None}
    if component_maps is not None and figure_output is not None:
        maps = np.load(component_maps)["component_maps"]
        nodes = build_scheme_node_frame(maps, gateway, mediator)
        nodes.to_csv(output_dir / "map_nodes.csv", index=False)
        plot_limits = plot_scheme_a_map(nodes, figure_output, save_svg=True)
        figure_path = figure_output
        if comparison_figure_output is not None and reference_gateway is not None and reference_mediator is not None and reference_hyperedges is not None:
            hyper_nodes = build_hyper_node_frame(maps, reference_gateway, reference_mediator, reference_hyperedges)
            hyper_nodes.to_csv(output_dir / "reference_hyper_map_nodes.csv", index=False)
            plot_scheme_a_vs_hyper_map(nodes, hyper_nodes, comparison_figure_output, save_svg=True)
            comparison_figure_path = comparison_figure_output

    manifest: dict[str, Any] = {
        "input_matrix": str(input_matrix),
        "names_matrix": str(names_matrix),
        "n_components": int(conditioned.shape[0]),
        "graph_sparsify": str(graph_sparsify),
        "graph_topk": int(graph_topk),
        "path_alpha": float(path_alpha),
        "max_path_length": int(max_path_length),
        "scale_factor": float(path_effects.scale_factor),
        "n_direct_edges": int(np.count_nonzero(direct)),
        "reference_gateway": None if reference_gateway is None else str(reference_gateway),
        "reference_mediator": None if reference_mediator is None else str(reference_mediator),
        "reference_hyperedges": None if reference_hyperedges is None else str(reference_hyperedges),
        "reference_kind": reference_kind,
        "figure": None if figure_path is None else str(figure_path),
        "comparison_figure": None if comparison_figure_path is None else str(comparison_figure_path),
        "plot_color_limits": plot_limits,
        "rank_correlations": correlations,
        "overlap_summary": overlaps,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    if report is not None and figure_path is not None:
        append_report_section(
            report,
            figure_path=figure_path,
            gateway=gateway,
            mediator=mediator,
            correlations=correlations,
            overlaps=overlaps,
            manifest=manifest,
        )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-matrix", default=str(DEFAULT_INPUT))
    parser.add_argument("--names-matrix", default=str(DEFAULT_NAMES))
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--graph-sparsify", choices=["none", "source_topk"], default="source_topk")
    parser.add_argument("--graph-topk", type=int, default=5)
    parser.add_argument("--path-alpha", type=float, default=0.8)
    parser.add_argument("--max-path-length", type=int, default=60)
    parser.add_argument("--reference-gateway", default=str(DEFAULT_REFERENCE_GATEWAY))
    parser.add_argument("--reference-mediator", default=str(DEFAULT_REFERENCE_MEDIATOR))
    parser.add_argument("--reference-hyperedges", default=str(DEFAULT_REFERENCE_HYPEREDGES))
    parser.add_argument("--component-maps", default=str(DEFAULT_COMPONENT_MAPS))
    parser.add_argument("--figure-output", default=str(DEFAULT_FIGURE))
    parser.add_argument("--comparison-figure-output", default=str(DEFAULT_COMPARISON_FIGURE))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest = run_scheme_a(
        input_matrix=Path(args.input_matrix).expanduser(),
        names_matrix=Path(args.names_matrix).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        graph_topk=int(args.graph_topk),
        graph_sparsify=str(args.graph_sparsify),
        path_alpha=float(args.path_alpha),
        max_path_length=int(args.max_path_length),
        reference_gateway=Path(args.reference_gateway).expanduser() if args.reference_gateway else None,
        reference_mediator=Path(args.reference_mediator).expanduser() if args.reference_mediator else None,
        reference_hyperedges=Path(args.reference_hyperedges).expanduser() if args.reference_hyperedges else None,
        component_maps=None if args.no_figure else Path(args.component_maps).expanduser(),
        figure_output=None if args.no_figure else Path(args.figure_output).expanduser(),
        comparison_figure_output=None if args.no_figure else Path(args.comparison_figure_output).expanduser(),
        report=None if args.no_report else Path(args.report).expanduser(),
    )
    print(json.dumps({"result_dir": str(Path(args.output_dir).expanduser()), "figure": manifest.get("figure")}, indent=2))


if __name__ == "__main__":
    main()
