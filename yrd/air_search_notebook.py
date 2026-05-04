from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from .air_search import (
    build_air_search_artifact_paths,
    build_air_search_config,
    compute_air_search_tm_summary,
    load_json,
    prepare_air_search_bundle,
    run_or_load_air_search_predictions,
    summarize_edge_distribution,
)
from .shanghai_notebook import (
    build_global_edge_ranking,
    build_self_loop_node_strengths,
    draw_station_causal_graph,
    find_project_root,
    metric_rows_for_scope,
    style_pairwise_edge,
)


def build_tm_run_tag(
    *,
    gamma: float,
    sample_count: int,
    seed: int,
    use_smoke: bool,
    box_mode: str = "per_variable",
    global_box_size_override: float | None = None,
) -> str:
    gamma_label = str(f"{float(gamma):.2f}").replace(".", "p")
    prefix = "refine_smoke" if use_smoke else "refine"
    run_tag = f"{prefix}_tm_g{gamma_label}_m{int(sample_count)}_seed{int(seed)}"
    if box_mode == "per_variable":
        return run_tag
    if box_mode == "global_max":
        if global_box_size_override is None:
            return f"{run_tag}_lmax"
        override_label = f"{float(global_box_size_override):.4f}".replace(".", "p")
        return f"{run_tag}_l{override_label}"
    raise ValueError(f"Unsupported box_mode={box_mode!r}.")


def _to_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False) + "\n")


def filter_station_edge_frame(
    frame: pd.DataFrame,
    *,
    top_k_edges: int,
    min_abs_strength: float,
    include_self_loops: bool,
    positive_only: bool,
    sort_by_abs: bool,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    filtered = frame.copy()
    if not include_self_loops:
        filtered = filtered[filtered["source_station_id"] != filtered["target_station_id"]].copy()
    filtered["abs_mean"] = filtered["mean"].astype(float).abs()
    filtered = filtered[filtered["abs_mean"] >= float(min_abs_strength)].copy()
    if positive_only:
        filtered = filtered[filtered["mean"].astype(float) > 0.0].copy()
    if filtered.empty:
        return filtered.reset_index(drop=True)
    sort_col = "abs_mean" if sort_by_abs else "mean"
    return (
        filtered.sort_values(sort_col, ascending=False)
        .head(max(1, int(top_k_edges)))
        .reset_index(drop=True)
    )


def build_support_cover_profile_table(profile: dict[str, object]) -> pd.DataFrame:
    variables = list(profile.get("input_variables", []))
    rows: list[dict[str, object]] = []
    lower_bound_by_variable = dict(profile.get("lower_bound_by_variable", {}))
    nonnegative_variables = set(profile.get("nonnegative_variables", []))
    for variable in variables:
        rows.append(
            {
                "variable": variable,
                "center": float(dict(profile.get("center_by_variable", {})).get(variable, 0.0)),
                "train_min": float(dict(profile.get("train_min_by_variable", {})).get(variable, 0.0)),
                "train_max": float(dict(profile.get("train_max_by_variable", {})).get(variable, 0.0)),
                "cover_radius": float(dict(profile.get("cover_radius_by_variable", {})).get(variable, 0.0)),
                "box_size_Lv": float(dict(profile.get("box_size_by_variable", {})).get(variable, 0.0)),
                "lower_bound": (
                    float(lower_bound_by_variable[variable])
                    if variable in lower_bound_by_variable
                    else None
                ),
                "nonnegative_clipped": bool(variable in nonnegative_variables),
            }
        )
    return pd.DataFrame(rows)


def _horizon_label(horizon: int) -> str:
    return f"{int(horizon)}h"


def _metric_summary_table(
    *,
    bundle: dict[str, object],
    predictions: dict[str, object],
    tm_summary: dict[str, object],
) -> pd.DataFrame:
    horizon = int(bundle["cfg"].horizons[0])
    metrics_frame = pd.DataFrame(
        metric_rows_for_scope(
            predictions["y_test_original"],
            {
                "joint_model": predictions["joint_original_predictions"],
                "persistence": predictions["baseline_original_predictions"],
            },
            target_names=bundle["target_names"],
        )
    )
    metrics_scope = metrics_frame[
        (metrics_frame["horizon"] == _horizon_label(horizon))
        & (metrics_frame["scope"].isin(["overall", "O3", "PM2.5"]))
    ].copy()
    syn_stats = summarize_edge_distribution(tm_summary["conditional_synergy"]["conditional_synergy_edges"])
    pm25_stats = summarize_edge_distribution(tm_summary["single_pollutant_pairwise"]["pairwise_edges"])
    extra_rows = pd.DataFrame(
        [
            {"metric": "Syn mean", "value": float(syn_stats["mean"])},
            {"metric": "Syn negative ratio", "value": float(syn_stats["negative_ratio"])},
            {"metric": "PM2.5 -> O3 mean", "value": float(pm25_stats["mean"])},
            {"metric": "PM2.5 -> O3 negative ratio", "value": float(pm25_stats["negative_ratio"])},
        ]
    )
    if metrics_scope.empty:
        return extra_rows
    metric_rows = []
    for _, row in metrics_scope.iterrows():
        metric_rows.append(
            {
                "metric": f"{row['model']} {row['scope']} RMSE",
                "value": float(row["rmse"]),
            }
        )
        metric_rows.append(
            {
                "metric": f"{row['model']} {row['scope']} Corr",
                "value": float(row["corr"]),
            }
        )
    return pd.concat([pd.DataFrame(metric_rows), extra_rows], ignore_index=True)


def _build_notebook_graph_paths(
    *,
    root_dir: Path,
    city_en: str,
    horizon: int,
    run_tag: str,
    use_smoke: bool,
    top_k_edges: int,
    min_abs_strength: float,
    show_negative_synergy_edges: bool,
) -> dict[str, Path]:
    base_paths = build_air_search_artifact_paths(
        root_dir=root_dir,
        city_en=city_en,
        horizon=horizon,
        run_tag=run_tag,
        use_smoke=use_smoke,
    )
    view_tag = (
        f"notebook_top{int(top_k_edges)}_"
        f"min{float(min_abs_strength):.4f}".replace(".", "p")
        + f"_neg{int(bool(show_negative_synergy_edges))}"
    )
    results_dir = Path(base_paths["results_dir"]) / view_tag
    results_dir.mkdir(parents=True, exist_ok=True)
    return {
        "results_dir": results_dir,
        "o3_pairwise": results_dir / "o3_pairwise_graph.png",
        "pm25_to_o3_pairwise": results_dir / "pm25_to_o3_pairwise_graph.png",
        "o3_pm25_synergy": results_dir / "o3_pm25_synergy_graph.png",
        "combined_panel": results_dir / "combined_tm_graph_panel_labeled.png",
    }


def _draw_combined_panel_edges(
    ax: plt.Axes,
    *,
    position_map: dict[str, tuple[float, float]],
    edges: pd.DataFrame,
    color: str,
    negative_color: str | None = None,
    strength_col: str = "mean",
) -> None:
    if edges.empty:
        return
    render_edges = edges.copy()
    if strength_col in render_edges.columns:
        render_edges = render_edges[render_edges[strength_col] > 0].copy()
    if render_edges.empty:
        return
    max_edge = max(float(render_edges[strength_col].max()), 1e-6)
    for _, row in render_edges.iterrows():
        x0, y0 = position_map[row["source_station_id"]]
        x1, y1 = position_map[row["target_station_id"]]
        edge_color = color if negative_color is None or float(row["mean"]) >= 0 else negative_color
        width, alpha = style_pairwise_edge(float(row[strength_col]) / max_edge, alpha_min=0.08, alpha_max=0.82)
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": edge_color,
                "linewidth": width,
                "alpha": alpha,
                "connectionstyle": f"arc3,rad={0.12 if x0 <= x1 else -0.12}",
                "shrinkA": 10,
                "shrinkB": 8,
                "mutation_scale": 16,
            },
            zorder=2,
        )


def save_combined_tm_graph_panel(
    results: dict[str, object],
    out_path: Path | str | None = None,
    *,
    panel_labels: tuple[str, str, str] | list[str] = ("(a)", "(b)", "(c)"),
    panel_label_fontsize: float = 15.0,
    panel_label_x: float = 0.5,
    panel_label_y: float = -0.17,
    panel_label_fontweight: str = "bold",
) -> Path:
    station_positions = results["station_positions_df"]
    station_ids = list(results["station_ids"])
    graph_paths = {key: Path(value) for key, value in results["graph_paths"].items()}
    output_path = Path(out_path) if out_path is not None else graph_paths["combined_panel"]
    position_map = {row["station_id"]: (float(row["lon"]), float(row["lat"])) for _, row in station_positions.iterrows()}

    panels = [
        ("O3 -> O3", results["o3_pairwise_display_df"], results["o3_pairwise_df"], "mean", "#345995", None),
        ("PM2.5 -> O3", results["pm25_to_o3_display_df"], results["pm25_to_o3_df"], "mean", "#345995", None),
        ("O3+PM2.5 -> O3", results["synergy_display_df"], results["synergy_df"], "abs_mean", "#2F7D63", "#B04A5A"),
    ]
    node_strengths = [build_self_loop_node_strengths(frame, station_ids=station_ids) for _, _, frame, _, _, _ in panels]
    node_values = np.asarray([float(strengths.get(station_id, 0.0)) for strengths in node_strengths for station_id in station_ids])
    max_abs_node = max(float(np.max(np.abs(node_values))) if len(node_values) else 0.0, 1e-9)
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs_node, vcenter=0.0, vmax=max_abs_node)

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.6), sharex=True, sharey=True, constrained_layout=True)
    panel_tags = list(panel_labels)
    for ax, tag, (label, edges, _, strength_col, color, negative_color), strengths in zip(axes, panel_tags, panels, node_strengths):
        node_colors = matplotlib.colormaps["RdBu_r"]([norm(float(strengths.get(station_id, 0.0))) for station_id in station_ids])
        ax.scatter(station_positions["lon"], station_positions["lat"], color=node_colors, s=74, edgecolors="#4C566A", linewidths=0.6, zorder=5)
        for _, row in station_positions.iterrows():
            ax.text(row["lon"] + 0.003, row["lat"] + 0.002, row["station_id"], fontsize=6.7, color="#233142", zorder=6)
        _draw_combined_panel_edges(ax, position_map=position_map, edges=edges, color=color, negative_color=negative_color, strength_col=strength_col)
        ax.text(
            float(panel_label_x),
            float(panel_label_y),
            tag,
            transform=ax.transAxes,
            fontsize=float(panel_label_fontsize),
            fontweight=panel_label_fontweight,
            va="top",
            ha="center",
            clip_on=False,
        )
        ax.set_title(label, fontsize=10.5)
        ax.set_xlabel("Longitude")
        ax.grid(True, alpha=0.16, linewidth=0.6)
    axes[0].set_ylabel("Latitude")
    colorbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="RdBu_r"), ax=axes, fraction=0.025, pad=0.015)
    colorbar.set_label("Self-loop strength")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _render_graphs(
    *,
    bundle: dict[str, object],
    o3_pairwise_display_df: pd.DataFrame,
    pm25_to_o3_display_df: pd.DataFrame,
    synergy_display_df: pd.DataFrame,
    graph_paths: dict[str, Path],
) -> None:
    station_positions = bundle["city_metadata"][["station_id", "lon", "lat"]]
    station_ids = bundle["station_ids"]
    horizon_label = _horizon_label(int(bundle["cfg"].horizons[0]))

    o3_pairwise_df = pd.DataFrame(bundle["tm_summary"]["o3_pairwise"]["pairwise_edges"])
    pm25_pairwise_df = pd.DataFrame(bundle["tm_summary"]["single_pollutant_pairwise"]["pairwise_edges"])
    synergy_df = pd.DataFrame(bundle["tm_summary"]["conditional_synergy"]["conditional_synergy_edges"])

    draw_station_causal_graph(
        station_positions=station_positions,
        pairwise_edges=o3_pairwise_display_df,
        horizon_label=horizon_label,
        out_path=graph_paths["o3_pairwise"],
        title=f"{bundle['run_context']['city_en'].title()} O3 -> O3 pairwise graph ({horizon_label})",
        strength_col="mean",
        node_self_strengths=build_self_loop_node_strengths(o3_pairwise_df, station_ids=station_ids),
        show_title=False,
        show_edge_legend=False,
    )
    draw_station_causal_graph(
        station_positions=station_positions,
        pairwise_edges=pm25_to_o3_display_df,
        horizon_label=horizon_label,
        out_path=graph_paths["pm25_to_o3_pairwise"],
        title=f"{bundle['run_context']['city_en'].title()} PM2.5 -> O3 pairwise graph ({horizon_label})",
        strength_col="mean",
        node_self_strengths=build_self_loop_node_strengths(pm25_pairwise_df, station_ids=station_ids),
        legend_label="PM2.5 -> O3 edge",
        show_title=False,
        show_edge_legend=False,
    )
    synergy_render_df = synergy_display_df.copy()
    if not synergy_render_df.empty:
        synergy_render_df["abs_mean"] = synergy_render_df["mean"].astype(float).abs()
    draw_station_causal_graph(
        station_positions=station_positions,
        pairwise_edges=synergy_render_df,
        horizon_label=horizon_label,
        out_path=graph_paths["o3_pm25_synergy"],
        title=f"{bundle['run_context']['city_en'].title()} O3+PM2.5 -> O3 synergy graph ({horizon_label})",
        strength_col="abs_mean",
        positive_color="#2F7D63",
        negative_color="#B04A5A",
        legend_label="Synergy edge",
        node_self_strengths=build_self_loop_node_strengths(synergy_df, station_ids=station_ids),
        node_colorbar_label="Self Syn",
        show_title=False,
        show_edge_legend=False,
    )


def run_air_tm_notebook_case(
    *,
    root_dir: Path,
    city_en: str,
    horizon: int,
    tm_sample_count: int,
    sampling_seed: int,
    gamma: float,
    top_k_edges: int,
    min_abs_strength: float,
    show_negative_synergy_edges: bool,
    force_retrain: bool,
    force_recompute_tm: bool,
    use_smoke: bool,
    box_mode: str = "per_variable",
    global_box_size_override: float | None = None,
) -> dict[str, object]:
    resolved_root = find_project_root(Path(root_dir))
    cfg = build_air_search_config(
        resolved_root,
        city_en=city_en,
        horizon=int(horizon),
        test_mode=use_smoke,
    )
    run_tag = build_tm_run_tag(
        gamma=float(gamma),
        sample_count=int(tm_sample_count),
        seed=int(sampling_seed),
        use_smoke=use_smoke,
        box_mode=box_mode,
        global_box_size_override=global_box_size_override,
    )
    bundle = prepare_air_search_bundle(
        cfg=cfg,
        city_en=city_en,
        run_tag=run_tag,
        use_smoke=use_smoke,
    )
    predictions = run_or_load_air_search_predictions(bundle, force_retrain=force_retrain)
    refine_summary_path = Path(bundle["artifact_paths"]["refine_summary"])
    used_cached_results = refine_summary_path.exists() and not force_recompute_tm
    if used_cached_results:
        tm_summary = load_json(refine_summary_path)
    else:
        tm_summary = compute_air_search_tm_summary(
            bundle,
            predictions,
            sample_count=int(tm_sample_count),
            sampling_seed=int(sampling_seed),
            gamma=float(gamma),
            box_mode=box_mode,
            global_box_size_override=global_box_size_override,
        )
        _save_json(refine_summary_path, tm_summary)
    bundle["tm_summary"] = tm_summary

    o3_pairwise_df = pd.DataFrame(tm_summary["o3_pairwise"]["pairwise_edges"])
    pm25_to_o3_df = pd.DataFrame(tm_summary["single_pollutant_pairwise"]["pairwise_edges"])
    synergy_df = pd.DataFrame(tm_summary["conditional_synergy"]["conditional_synergy_edges"])

    o3_pairwise_display_df = filter_station_edge_frame(
        o3_pairwise_df,
        top_k_edges=int(top_k_edges),
        min_abs_strength=float(min_abs_strength),
        include_self_loops=False,
        positive_only=True,
        sort_by_abs=False,
    )
    pm25_to_o3_display_df = filter_station_edge_frame(
        pm25_to_o3_df,
        top_k_edges=int(top_k_edges),
        min_abs_strength=float(min_abs_strength),
        include_self_loops=False,
        positive_only=True,
        sort_by_abs=False,
    )
    synergy_display_df = filter_station_edge_frame(
        synergy_df,
        top_k_edges=int(top_k_edges),
        min_abs_strength=float(min_abs_strength),
        include_self_loops=False,
        positive_only=not bool(show_negative_synergy_edges),
        sort_by_abs=True,
    )
    o3_pairwise_ranked_df = build_global_edge_ranking(o3_pairwise_display_df, sort_col="mean")
    pm25_to_o3_ranked_df = build_global_edge_ranking(pm25_to_o3_display_df, sort_col="mean")
    synergy_ranked_df = build_global_edge_ranking(synergy_display_df, sort_col="abs_mean")
    profile_variable_df = build_support_cover_profile_table(tm_summary["profile"])
    summary_metrics_df = _metric_summary_table(
        bundle=bundle,
        predictions=predictions,
        tm_summary=tm_summary,
    )
    graph_paths = _build_notebook_graph_paths(
        root_dir=resolved_root,
        city_en=city_en,
        horizon=int(horizon),
        run_tag=run_tag,
        use_smoke=use_smoke,
        top_k_edges=int(top_k_edges),
        min_abs_strength=float(min_abs_strength),
        show_negative_synergy_edges=bool(show_negative_synergy_edges),
    )
    _render_graphs(
        bundle=bundle,
        o3_pairwise_display_df=o3_pairwise_display_df,
        pm25_to_o3_display_df=pm25_to_o3_display_df,
        synergy_display_df=synergy_display_df,
        graph_paths=graph_paths,
    )
    profile = tm_summary["profile"]
    if box_mode == "global_max":
        global_box_size = float(profile["global_box_size"])
        global_box_size_override_value = profile.get("global_box_size_override")
        box_description = f"scalar L=max_v L_v={global_box_size:.4f}"
    else:
        global_box_size = None
        global_box_size_override_value = None
        box_description = "support-cover L_v"
    final_conclusion_text = (
        f"{city_en.title()} {int(horizon)}h uses {box_description} with gamma={float(gamma):.2f}. "
        f"The notebook defaults to cached TM results, exposes O3 -> O3, PM2.5 -> O3, and "
        f"O3 + PM2.5 -> O3 synergy graphs, and lets you compare top-{int(top_k_edges)} edges "
        f"under different sample_count/seed settings."
    )
    return {
        "run_context": {
            "city_en": str(city_en),
            "horizon": int(horizon),
            "tm_sample_count": int(tm_sample_count),
            "sampling_seed": int(sampling_seed),
            "gamma": float(gamma),
            "box_mode": str(profile.get("box_mode", box_mode)),
            **({"global_box_size": global_box_size} if global_box_size is not None else {}),
            **(
                {"global_box_size_override": float(global_box_size_override_value)}
                if global_box_size_override_value is not None
                else {}
            ),
            "run_tag": run_tag,
            "used_cached_results": bool(used_cached_results),
            "results_dir": str(graph_paths["results_dir"]),
        },
        "summary_metrics_df": summary_metrics_df,
        "profile_variable_df": profile_variable_df,
        "o3_pairwise_df": o3_pairwise_df,
        "pm25_to_o3_df": pm25_to_o3_df,
        "synergy_df": synergy_df,
        "o3_pairwise_display_df": o3_pairwise_display_df,
        "pm25_to_o3_display_df": pm25_to_o3_display_df,
        "synergy_display_df": synergy_display_df,
        "o3_pairwise_ranked_df": o3_pairwise_ranked_df,
        "pm25_to_o3_ranked_df": pm25_to_o3_ranked_df,
        "synergy_ranked_df": synergy_ranked_df,
        "graph_paths": {key: str(value) for key, value in graph_paths.items() if key != "results_dir"},
        "station_positions_df": bundle["city_metadata"][["station_id", "lon", "lat"]].copy(),
        "station_ids": list(bundle["station_ids"]),
        "profile": tm_summary["profile"],
        "tm_summary": tm_summary,
        "final_conclusion_text": final_conclusion_text,
    }
