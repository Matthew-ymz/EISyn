from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

from .config import YRDExperimentConfig
from .coupling import (
    build_one_step_station_pollutant_feature_groups,
    build_one_step_station_source_groups,
    compute_station_pollutant_pair_synergy_summary,
    compute_station_level_ei_summary,
    compute_station_level_nis_summary,
    estimate_residual_covariance,
    jacobian_for_target_subset,
    summarize_global_station_pollutant_synergy,
    summarize_global_station_coupling,
)
from .data import build_one_step_samples, load_dataset, load_station_metadata, select_station_metadata
from .intervention_sampling import (
    compute_training_input_center as _compute_training_input_center,
    estimate_support_cover_box_profile,
    resolve_causal_graph_box_label as _resolve_causal_graph_box_label,
    resolve_nonnegative_lower_bounds_by_feature as _resolve_nonnegative_lower_bounds_by_feature,
    resolve_variable_box_size_by_feature as _resolve_variable_box_size_by_feature,
    sample_uniform_box_inputs as _sample_uniform_box_inputs,
)
from .models import PersistenceBaseline
from .train import _compute_metrics, _predict_numpy, rebuild_joint_model_from_checkpoint, set_seed, train_joint_model_with_history

PAIRWISE_EDGE_COLOR = "#345995"
HYPEREDGE_COLOR = "#2F7D63"
SYNERGY_POSITIVE_EDGE_COLOR = "#2F7D63"
SYNERGY_NEGATIVE_EDGE_COLOR = "#B04A5A"
SYNERGY_RATIO_EDGE_COLOR = "#C47C2E"


def configure_notebook_runtime() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def find_project_root(start: Path | None = None) -> Path:
    candidates = [start.resolve() if start is not None else Path.cwd().resolve()]
    candidates.extend(candidates[0].parents)
    fallback_candidate: Path | None = None
    for candidate in candidates:
        if not (candidate / "yrd").is_dir() or not (candidate / "data").is_dir():
            continue
        if fallback_candidate is None:
            fallback_candidate = candidate
        if (
            ((candidate / "data" / "dataset_yrd.nc").exists() and (candidate / "data" / "stations_yrd.csv").exists())
            or (
                (candidate / "data" / "dataset_bthsa.nc").exists()
                and (candidate / "data" / "stations_bthsa.csv").exists()
            )
        ):
            return candidate
    if fallback_candidate is not None:
        return fallback_candidate
    raise RuntimeError("Could not locate project root containing yrd/ and data/")


def build_default_shanghai_one_step_config(root: Path, *, test_mode: bool) -> YRDExperimentConfig:
    return replace(
        YRDExperimentConfig(root_dir=root),
        sample_mode="one_step",
        history_hours=1,
        horizons=(1,),
        model_name="resmlp",
        hidden_dim=16 if test_mode else 96,
        num_layers=2 if test_mode else 3,
        dropout=0.0 if test_mode else 0.05,
        norm_type="layernorm",
        activation="silu",
        learning_rate=5e-4,
        weight_decay=0.0 if test_mode else 1e-5,
        batch_size=8 if test_mode else 64,
        epochs=2 if test_mode else 12,
        max_epochs=4 if test_mode else 60,
        early_stopping_patience=2 if test_mode else 8,
        seed=0,
    )


def build_run_tag(*, test_mode: bool) -> str:
    return "shanghai_one_step_o3_station_graph_smoke" if test_mode else "shanghai_one_step_o3_station_graph"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False))


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def inverse_transform_targets(
    array: np.ndarray,
    target_names: list[str],
    stats: dict[str, dict[str, float]],
) -> np.ndarray:
    restored = np.asarray(array, dtype=np.float32).copy()
    for index, name in enumerate(target_names):
        variable = name.split("__")[-1]
        restored[:, index] = restored[:, index] * stats[variable]["std"] + stats[variable]["mean"]
    return restored


def metric_rows_for_scope(
    y_true_by_horizon: dict[int, np.ndarray],
    prediction_sets: dict[str, dict[int, np.ndarray]],
    *,
    target_names: list[str],
) -> list[dict[str, object]]:
    scope_indices = {
        "overall": list(range(len(target_names))),
        "O3": [index for index, name in enumerate(target_names) if name.endswith("__O3")],
        "PM2.5": [index for index, name in enumerate(target_names) if name.endswith("__PM2.5")],
    }
    rows: list[dict[str, object]] = []
    for model_name, predictions_by_horizon in prediction_sets.items():
        for horizon, y_true in y_true_by_horizon.items():
            for scope_name, indices in scope_indices.items():
                summary = _compute_metrics(y_true[:, indices], predictions_by_horizon[horizon][:, indices])
                rows.append(
                    {
                        "model": model_name,
                        "horizon": f"{horizon}h",
                        "scope": scope_name,
                        "rmse": summary["rmse"],
                        "mae": summary["mae"],
                        "corr": summary["corr"],
                    }
                )
    return rows


def station_metric_rows(
    y_true_by_horizon: dict[int, np.ndarray],
    prediction_sets: dict[str, dict[int, np.ndarray]],
    *,
    station_ids: list[str],
    target_width: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name, predictions_by_horizon in prediction_sets.items():
        for horizon, y_true in y_true_by_horizon.items():
            for station_index, station_id in enumerate(station_ids):
                start = station_index * target_width
                indices = list(range(start, start + target_width))
                summary = _compute_metrics(y_true[:, indices], predictions_by_horizon[horizon][:, indices])
                rows.append(
                    {
                        "model": model_name,
                        "horizon": f"{horizon}h",
                        "station_id": station_id,
                        "rmse": summary["rmse"],
                        "mae": summary["mae"],
                        "corr": summary["corr"],
                    }
                )
    return rows


def build_station_variable_index_map(target_names: list[str], variable: str) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    suffix = f"__{variable}"
    for index, name in enumerate(target_names):
        if not name.endswith(suffix):
            continue
        _, station_id, _ = name.split("__")
        mapping.setdefault(station_id, []).append(index)
    return mapping


def compute_training_input_center(x_train: np.ndarray) -> np.ndarray:
    return _compute_training_input_center(x_train)


def sample_uniform_box_inputs(
    *,
    center: np.ndarray,
    box_size: float | np.ndarray | list[float],
    sample_count: int,
    seed: int,
    lower_bounds: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    return _sample_uniform_box_inputs(
        center=center,
        box_size=box_size,
        sample_count=sample_count,
        seed=seed,
        lower_bounds=lower_bounds,
    )


def causal_graph_cache_is_valid(
    *,
    cache_meta: dict[str, object],
    requested_box_size: float | None = None,
    requested_box_size_by_variable: dict[str, float] | None = None,
    nonnegative_variables: tuple[str, ...] = (),
    coupling_sample_count: int,
    sampling_mode: str,
    sampling_seed: int,
) -> bool:
    cached_sample_count = int(cache_meta.get("coupling_sample_count", -1))
    cached_sampling_mode = str(cache_meta.get("sampling_mode", ""))
    cached_sampling_seed = int(cache_meta.get("sampling_seed", -1))
    if (
        cached_sample_count != coupling_sample_count
        or cached_sampling_mode != sampling_mode
        or cached_sampling_seed != sampling_seed
    ):
        return False
    if requested_box_size_by_variable is not None:
        cached_requested_box_sizes = cache_meta.get("requested_causal_graph_box_size_by_variable")
        cached_box_sizes = cache_meta.get("causal_graph_box_size_by_variable")
        cached_nonnegative_variables = tuple(cache_meta.get("causal_graph_nonnegative_variables", ()))
        return bool(
            isinstance(cached_requested_box_sizes, dict)
            and isinstance(cached_box_sizes, dict)
            and cached_requested_box_sizes == requested_box_size_by_variable
            and cached_box_sizes == requested_box_size_by_variable
            and cached_nonnegative_variables == tuple(nonnegative_variables)
        )

    cached_requested_box_size = float(cache_meta.get("requested_causal_graph_box_size", -1.0))
    cached_box_size = float(cache_meta.get("causal_graph_box_size", -1.0))
    return bool(cached_box_size > 0.0 and np.isclose(cached_requested_box_size, float(requested_box_size)))


def resolve_variable_box_size_by_feature(
    *,
    input_variables: tuple[str, ...],
    box_size_by_variable: dict[str, float],
    n_stations: int,
) -> np.ndarray:
    return _resolve_variable_box_size_by_feature(
        input_variables=input_variables,
        box_size_by_variable=box_size_by_variable,
        n_stations=n_stations,
    )


def resolve_nonnegative_lower_bounds_by_feature(
    *,
    input_variables: tuple[str, ...],
    stats: dict[str, dict[str, float]],
    nonnegative_variables: tuple[str, ...],
    n_stations: int,
) -> np.ndarray | None:
    return _resolve_nonnegative_lower_bounds_by_feature(
        input_variables=input_variables,
        stats=stats,
        nonnegative_variables=nonnegative_variables,
        n_stations=n_stations,
    )


def resolve_causal_graph_box_label(
    *,
    box_size: float | None = None,
    box_size_by_variable: dict[str, float] | None = None,
) -> str | float | None:
    return _resolve_causal_graph_box_label(
        box_size=box_size,
        box_size_by_variable=box_size_by_variable,
    )


def select_display_rows(
    frame: pd.DataFrame,
    *,
    per_target: int,
    source_col: str,
    target_col: str,
    ranking_col: str = "mean",
    positive_only: bool = False,
    include_self_loops: bool = True,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    filtered = frame.copy()
    if not include_self_loops:
        filtered = filtered[filtered[source_col] != filtered[target_col]].copy()
    if positive_only:
        filtered = filtered[filtered["mean"] > 0].copy()
    if filtered.empty:
        return filtered.reset_index(drop=True)
    filtered = filtered.sort_values([target_col, ranking_col], ascending=[True, False])
    return filtered.groupby(target_col, group_keys=False).head(per_target).reset_index(drop=True)


def select_display_hyperedge_rows(frame: pd.DataFrame, *, per_target: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    display_frame = frame.copy()
    display_frame["source_pair_label"] = display_frame["source_station_ids"].apply(
        lambda value: " + ".join(map(str, value))
    )
    display_frame["_abs_strength"] = display_frame["mean"].astype(float).abs()
    display_frame = display_frame.sort_values(["target_station_id", "_abs_strength"], ascending=[True, False])
    display_frame = display_frame.drop(columns="_abs_strength")
    return display_frame.groupby("target_station_id", group_keys=False).head(per_target).reset_index(drop=True)


def build_global_hyperedge_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ranked = frame.copy()
    ranked["source_pair_label"] = ranked["source_station_ids"].apply(lambda value: " + ".join(map(str, value)))
    ranked["hyperedge_label"] = ranked.apply(
        lambda row: f"{row['source_pair_label']} -> {row['target_station_id']}",
        axis=1,
    )
    ranked["_abs_strength"] = ranked["mean"].astype(float).abs()
    ranked = ranked.sort_values("_abs_strength", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    total_abs_mean = float(ranked["_abs_strength"].sum())
    ranked["cumulative_abs_mean"] = ranked["_abs_strength"].cumsum()
    ranked["cumulative_share"] = (
        ranked["cumulative_abs_mean"] / total_abs_mean if total_abs_mean > 0 else 0.0
    )
    ranked = ranked.drop(columns="_abs_strength")
    return ranked


def build_global_edge_ranking(frame: pd.DataFrame, *, sort_col: str = "mean") -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ranked = frame.copy()
    ranked["edge_label"] = ranked.apply(
        lambda row: f"{row['source_station_id']} -> {row['target_station_id']}",
        axis=1,
    )
    ranked["_abs_strength"] = ranked["mean"].astype(float).abs()
    if sort_col == "abs_mean":
        ranked = ranked.sort_values("_abs_strength", ascending=False).reset_index(drop=True)
    else:
        ranked = ranked.sort_values(sort_col, ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    total_abs_mean = float(ranked["_abs_strength"].sum())
    ranked["cumulative_abs_mean"] = ranked["_abs_strength"].cumsum()
    ranked["cumulative_share"] = (
        ranked["cumulative_abs_mean"] / total_abs_mean if total_abs_mean > 0 else 0.0
    )
    ranked = ranked.drop(columns="_abs_strength")
    return ranked


def summarize_hyperedge_cutoff(frame: pd.DataFrame, *, top_k_values: tuple[int, ...] = (5, 10, 15, 20)) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    if frame.empty:
        return pd.DataFrame(columns=["top_k", "edge_count", "cumulative_share", "cutoff_abs_mean"])
    for top_k in top_k_values:
        selected = frame.head(top_k)
        if selected.empty:
            continue
        rows.append(
            {
                "top_k": int(top_k),
                "edge_count": int(len(selected)),
                "cumulative_share": float(selected["cumulative_share"].iloc[-1]),
                "cutoff_abs_mean": float(abs(float(selected["mean"].iloc[-1]))),
            }
        )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _format_graph_title(base_title: str, *, box_size: float | str | None = None) -> str:
    if box_size is None:
        return base_title
    if isinstance(box_size, str):
        return f"{base_title}, {box_size}"
    return f"{base_title}, L = {float(box_size):.3f}"


def build_self_loop_node_strengths(
    frame: pd.DataFrame,
    *,
    station_ids: list[str],
    source_col: str = "source_station_id",
    target_col: str = "target_station_id",
    strength_col: str = "mean",
) -> dict[str, float]:
    strengths = {str(station_id): 0.0 for station_id in station_ids}
    if frame.empty:
        return strengths
    diagonal_rows = frame[frame[source_col] == frame[target_col]].copy()
    if diagonal_rows.empty:
        return strengths
    for _, row in diagonal_rows.iterrows():
        station_id = str(row[source_col])
        if station_id in strengths:
            strengths[station_id] = float(row[strength_col])
    return strengths


def draw_station_causal_graph(
    *,
    station_positions: pd.DataFrame,
    pairwise_edges: pd.DataFrame,
    horizon_label: str,
    out_path: Path,
    title: str | None = None,
    positive_color: str = PAIRWISE_EDGE_COLOR,
    negative_color: str | None = None,
    legend_label: str | None = None,
    strength_col: str = "abs_mean",
    box_size: float | None = None,
    alpha_min: float = 0.08,
    alpha_max: float = 0.82,
    arrow_mutation_scale: float = 18.0,
    arrow_shrink_target: float = 8.0,
    node_self_strengths: dict[str, float] | None = None,
    node_colorbar_label: str | None = None,
    show_title: bool = True,
    show_edge_legend: bool = True,
) -> None:
    position_map = {
        row["station_id"]: (float(row["lon"]), float(row["lat"]))
        for _, row in station_positions.iterrows()
    }
    fig, ax = plt.subplots(figsize=(10.8, 7.6), constrained_layout=True)
    node_colors = "#D8DEE9"
    node_norm: mcolors.Normalize | None = None
    if node_self_strengths is not None:
        node_values = np.asarray(
            [float(node_self_strengths.get(str(station_id), 0.0)) for station_id in station_positions["station_id"]],
            dtype=float,
        )
        max_abs_value = float(np.max(np.abs(node_values))) if len(node_values) else 0.0
        if max_abs_value > 0:
            node_norm = mcolors.TwoSlopeNorm(vmin=-max_abs_value, vcenter=0.0, vmax=max_abs_value)
            node_colors = matplotlib.colormaps["RdBu_r"](node_norm(node_values))
    ax.scatter(
        station_positions["lon"],
        station_positions["lat"],
        color=node_colors,
        s=110,
        edgecolors="#4C566A",
        linewidths=0.7,
        zorder=5,
    )
    for _, row in station_positions.iterrows():
        ax.text(
            row["lon"] + 0.004,
            row["lat"] + 0.002,
            row["station_id"],
            fontsize=7.4,
            color="#233142",
            zorder=6,
        )

    render_edges = pairwise_edges.copy()
    if strength_col in render_edges.columns:
        render_edges = render_edges[render_edges[strength_col] > 0].copy()

    if not render_edges.empty:
        max_edge = max(float(render_edges[strength_col].max()), 1e-6)
        for _, row in render_edges.iterrows():
            x0, y0 = position_map[row["source_station_id"]]
            x1, y1 = position_map[row["target_station_id"]]
            strength = float(row[strength_col])
            normalized_strength = max(0.0, min(1.0, strength / max_edge))
            width, alpha = style_pairwise_edge(normalized_strength, alpha_min=alpha_min, alpha_max=alpha_max)
            color = positive_color if negative_color is None or float(row["mean"]) >= 0 else negative_color
            rad = 0.12 if x0 <= x1 else -0.12
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "linewidth": width,
                    "alpha": alpha,
                    "connectionstyle": f"arc3,rad={rad}",
                    "shrinkA": 10,
                    "shrinkB": arrow_shrink_target,
                    "mutation_scale": arrow_mutation_scale,
                },
                zorder=2,
            )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.18, linewidth=0.6)
    if show_title:
        base_title = title or f"Shanghai O3 station-level causal graph ({horizon_label})"
        ax.set_title(_format_graph_title(base_title, box_size=box_size), fontsize=13)
    has_negative_display = negative_color is not None and not render_edges.empty and bool((render_edges["mean"] < 0).any())
    if negative_color is None or not has_negative_display:
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=positive_color,
                linewidth=2.4,
                label=legend_label or "O3 pairwise edge",
            )
        ]
    else:
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=positive_color,
                linewidth=2.4,
                label=f"{legend_label or 'Synergy edge'} (+)",
            ),
            Line2D(
                [0],
                [0],
                color=negative_color,
                linewidth=2.4,
                label=f"{legend_label or 'Synergy edge'} (-)",
            ),
        ]
    if show_edge_legend:
        ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    if node_norm is not None:
        colorbar = fig.colorbar(
            cm.ScalarMappable(norm=node_norm, cmap="RdBu_r"),
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
        colorbar.set_label(node_colorbar_label or "Self-loop strength")
    _save_figure(fig, out_path)


def build_pairwise_edge_render_frame(
    pairwise_edges: pd.DataFrame,
    *,
    strength_col: str = "mean",
    alpha_min: float = 0.08,
    alpha_max: float = 0.82,
) -> pd.DataFrame:
    render_edges = pairwise_edges.copy()
    if strength_col in render_edges.columns:
        render_edges = render_edges[render_edges[strength_col] > 0].copy()
    if render_edges.empty:
        return render_edges

    max_edge = max(float(render_edges[strength_col].max()), 1e-6)
    render_edges["normalized_strength"] = render_edges[strength_col].astype(float) / max_edge
    render_edges["normalized_strength"] = render_edges["normalized_strength"].clip(lower=0.0, upper=1.0)
    styles = render_edges["normalized_strength"].apply(
        lambda value: style_pairwise_edge(value, alpha_min=alpha_min, alpha_max=alpha_max)
    )
    render_edges["linewidth"] = styles.str[0].astype(float)
    render_edges["alpha"] = styles.str[1].astype(float)
    return render_edges.reset_index(drop=True)


def select_top_fraction_edges(
    pairwise_edges: pd.DataFrame,
    *,
    strength_col: str = "mean",
    keep_fraction: float = 0.5,
) -> pd.DataFrame:
    if pairwise_edges.empty:
        return pairwise_edges.copy()

    keep_count = max(1, int(np.ceil(len(pairwise_edges) * float(keep_fraction))))
    return (
        pairwise_edges.sort_values(strength_col, ascending=False)
        .head(keep_count)
        .reset_index(drop=True)
    )


def style_pairwise_edge(
    normalized_strength: float,
    *,
    alpha_min: float = 0.08,
    alpha_max: float = 0.82,
) -> tuple[float, float]:
    contrast_strength = max(0.0, min(1.0, float(normalized_strength))) ** 1.9
    linewidth = 0.18 + 5.4 * contrast_strength
    alpha = alpha_min + (alpha_max - alpha_min) * contrast_strength
    return linewidth, alpha


def build_pairwise_degree_summary(
    pairwise_edges: pd.DataFrame,
    *,
    station_ids: list[str],
    strength_col: str = "mean",
) -> pd.DataFrame:
    edge_frame = build_pairwise_edge_render_frame(pairwise_edges, strength_col=strength_col)
    if edge_frame.empty:
        return pd.DataFrame(
            {
                "station_id": station_ids,
                "in_degree": np.zeros(len(station_ids), dtype=int),
                "out_degree": np.zeros(len(station_ids), dtype=int),
                "in_strength": np.zeros(len(station_ids), dtype=float),
                "out_strength": np.zeros(len(station_ids), dtype=float),
            }
        )

    in_degree = edge_frame.groupby("target_station_id").size().reindex(station_ids, fill_value=0)
    out_degree = edge_frame.groupby("source_station_id").size().reindex(station_ids, fill_value=0)
    in_strength = (
        edge_frame.groupby("target_station_id")[strength_col].sum().reindex(station_ids, fill_value=0.0)
    )
    out_strength = (
        edge_frame.groupby("source_station_id")[strength_col].sum().reindex(station_ids, fill_value=0.0)
    )
    return pd.DataFrame(
        {
            "station_id": station_ids,
            "in_degree": in_degree.astype(int).to_numpy(),
            "out_degree": out_degree.astype(int).to_numpy(),
            "in_strength": in_strength.astype(float).to_numpy(),
            "out_strength": out_strength.astype(float).to_numpy(),
        }
    )


def draw_pairwise_ei_degree_diagnostics(
    *,
    render_edges: pd.DataFrame,
    node_degree_summary: pd.DataFrame,
    out_path: Path,
    horizon_label: str,
    box_size: float | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), constrained_layout=True)
    bins = min(18, max(8, int(np.sqrt(max(len(render_edges), 1)))))

    axes[0, 0].hist(
        render_edges["mean"] if not render_edges.empty else [0.0],
        bins=bins,
        color=PAIRWISE_EDGE_COLOR,
        alpha=0.86,
        edgecolor="white",
    )
    axes[0, 0].set_title("All positive pairwise EI distribution", fontsize=12)
    axes[0, 0].set_xlabel("Mean pairwise EI (NIS)")
    axes[0, 0].set_ylabel("Edge count")
    axes[0, 0].grid(True, axis="y", alpha=0.18, linewidth=0.6)

    for ax, column, title in (
        (axes[0, 1], "in_degree", "In-degree distribution"),
        (axes[1, 0], "out_degree", "Out-degree distribution"),
    ):
        values = node_degree_summary[column].to_numpy(dtype=int)
        if values.size == 0:
            values = np.array([0], dtype=int)
        bins_edges = np.arange(values.min() - 0.5, values.max() + 1.5, 1.0)
        ax.hist(values, bins=bins_edges, color=PAIRWISE_EDGE_COLOR, alpha=0.86, edgecolor="white")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Node count")
        ax.grid(True, axis="y", alpha=0.18, linewidth=0.6)

    axes[1, 1].hist(
        [node_degree_summary["in_strength"].to_numpy(dtype=float), node_degree_summary["out_strength"].to_numpy(dtype=float)],
        bins=min(14, max(6, len(node_degree_summary) // 2)),
        color=["#4C78A8", "#F28E2B"],
        alpha=0.72,
        edgecolor="white",
        label=["Weighted in-strength", "Weighted out-strength"],
    )
    axes[1, 1].set_title("Weighted degree-strength distribution", fontsize=12)
    axes[1, 1].set_xlabel("Sum of mean pairwise EI")
    axes[1, 1].set_ylabel("Node count")
    axes[1, 1].grid(True, axis="y", alpha=0.18, linewidth=0.6)
    axes[1, 1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig.suptitle(
        _format_graph_title(f"Shanghai O3 pairwise-edge diagnostics ({horizon_label})", box_size=box_size),
        fontsize=13,
    )
    _save_figure(fig, out_path)


def build_causal_summary_records(
    *,
    raw_records: list[dict[str, object]],
    station_source_groups: dict[str, list[int]],
    station_pollutant_feature_groups: dict[str, dict[str, list[int]]],
    box_size: float,
    causal_graph_variable: str,
) -> list[dict[str, object]]:
    sample_records: list[dict[str, object]] = []
    for raw_record in raw_records:
        jacobian = np.asarray(raw_record["jacobian"], dtype=float)
        target_indices = list(raw_record["target_indices"])
        sigma_eps = np.asarray(raw_record["sigma_eps"], dtype=float)
        summary = compute_station_level_nis_summary(
            jacobian=jacobian,
            sigma_eps=sigma_eps,
            station_source_groups=station_source_groups,
            target_indices=target_indices,
            box_size=box_size,
        )
        pollutant_pair_summary = compute_station_pollutant_pair_synergy_summary(
            jacobian=jacobian,
            sigma_eps=sigma_eps,
            station_pollutant_feature_groups=station_pollutant_feature_groups,
            target_indices=target_indices,
            box_size=box_size,
        )
        sample_records.append(
            {
                "sample_id": int(raw_record["sample_id"]),
                "sampling_mode": str(raw_record["sampling_mode"]),
                "horizon": "1h",
                "target_station_id": raw_record["target_station_id"],
                "target_variable": causal_graph_variable,
                "ei_nis": float(summary["ei_nis"]),
                "pairwise_station_ei_nis": summary["pairwise_station_ei_nis"],
                "binary_station_synergy_nis": summary["binary_station_synergy_nis"],
                "joint_station_pair_ei_nis": pollutant_pair_summary["joint_station_pair_ei_nis"],
                "single_pollutant_ei_nis": pollutant_pair_summary["single_pollutant_ei_nis"],
                "station_pair_synergy_nis": pollutant_pair_summary["station_pair_synergy_nis"],
            }
        )
    return sample_records


def build_transport_map_global_causal_summary(
    *,
    source_samples: np.ndarray | list[object],
    predicted_next_o3: np.ndarray | list[object],
    station_ids: list[str],
    o3_feature_index: int,
    pm25_feature_index: int,
) -> dict[str, object]:
    source_array = np.asarray(source_samples, dtype=float)
    target_array = np.asarray(predicted_next_o3, dtype=float)
    if source_array.ndim != 3:
        raise ValueError("source_samples must have shape [samples, stations, features].")
    if target_array.ndim != 2:
        raise ValueError("predicted_next_o3 must have shape [samples, stations].")
    if source_array.shape[0] != target_array.shape[0] or source_array.shape[1] != len(station_ids):
        raise ValueError("source_samples and predicted_next_o3 must align with station_ids.")

    current_o3 = source_array[:, :, int(o3_feature_index)]
    current_o3_pm25 = np.empty((source_array.shape[0], len(station_ids) * 2), dtype=float)
    station_o3_groups: dict[str, list[int]] = {}
    station_pollutant_feature_groups: dict[str, dict[str, list[int]]] = {}
    for station_index, station_id in enumerate(station_ids):
        current_o3_pm25[:, 2 * station_index] = source_array[:, station_index, int(o3_feature_index)]
        current_o3_pm25[:, 2 * station_index + 1] = source_array[:, station_index, int(pm25_feature_index)]
        station_o3_groups[str(station_id)] = [station_index]
        station_pollutant_feature_groups[str(station_id)] = {
            "O3": [2 * station_index],
            "PM2.5": [2 * station_index + 1],
        }

    summary_records: list[dict[str, object]] = []
    for target_index, target_station_id in enumerate(station_ids):
        target_matrix = target_array[:, [target_index]]
        pairwise_summary = compute_station_level_ei_summary(
            method="tm",
            source_samples=current_o3,
            target_samples=target_matrix,
            station_source_groups=station_o3_groups,
        )
        pollutant_summary = compute_station_pollutant_pair_synergy_summary(
            method="tm",
            source_samples=current_o3_pm25,
            target_samples=target_matrix,
            station_pollutant_feature_groups=station_pollutant_feature_groups,
        )
        summary_records.append(
            {
                "target_station_id": str(target_station_id),
                "pairwise_station_ei": pairwise_summary["pairwise_station_ei"],
                "binary_station_synergy": pairwise_summary["binary_station_synergy"],
                "joint_station_pair_ei": pollutant_summary["joint_station_pair_ei"],
                "single_pollutant_ei": pollutant_summary["single_pollutant_ei"],
                "station_pair_synergy": pollutant_summary["station_pair_synergy"],
            }
        )

    station_coupling_summary = summarize_global_station_coupling(summary_records, station_ids=station_ids)
    pollutant_synergy_summary = summarize_global_station_pollutant_synergy(summary_records, station_ids=station_ids)
    return {
        "coupling_method": "tm",
        "1h": {
            **station_coupling_summary,
            "conditional_synergy_edges": pollutant_synergy_summary["conditional_synergy_edges"],
            "conditional_synergy_ratio_edges": pollutant_synergy_summary["conditional_synergy_ratio_edges"],
            "conditional_synergy_per_target_station": pollutant_synergy_summary["per_target_station"],
        },
    }


def resolve_causal_graph_box_size(
    *,
    raw_records: list[dict[str, object]],
    station_ids: list[str],
    station_source_groups: dict[str, list[int]],
    station_pollutant_feature_groups: dict[str, dict[str, list[int]]],
    requested_box_size: float,
    graph_edges_per_target: int,
    causal_graph_variable: str,
) -> tuple[float, list[dict[str, object]], dict[str, object]]:
    candidate_box_sizes = [float(requested_box_size) * (2.0**power) for power in range(10)]
    fallback_payload: tuple[float, list[dict[str, object]], dict[str, object]] | None = None

    for box_size in candidate_box_sizes:
        sample_records = build_causal_summary_records(
            raw_records=raw_records,
            station_source_groups=station_source_groups,
            station_pollutant_feature_groups=station_pollutant_feature_groups,
            box_size=box_size,
            causal_graph_variable=causal_graph_variable,
        )
        station_coupling_summary = summarize_global_station_coupling(sample_records, station_ids=station_ids)
        pollutant_synergy_summary = summarize_global_station_pollutant_synergy(sample_records, station_ids=station_ids)
        global_summary_by_horizon = {
            "1h": {
                **station_coupling_summary,
                "conditional_synergy_edges": pollutant_synergy_summary["conditional_synergy_edges"],
                "conditional_synergy_ratio_edges": pollutant_synergy_summary["conditional_synergy_ratio_edges"],
                "conditional_synergy_per_target_station": pollutant_synergy_summary["per_target_station"],
            }
        }
        pairwise_edges_df = pd.DataFrame(global_summary_by_horizon["1h"]["pairwise_edges"])
        non_self_pairwise = (
            pairwise_edges_df[pairwise_edges_df["source_station_id"] != pairwise_edges_df["target_station_id"]].copy()
            if not pairwise_edges_df.empty
            else pd.DataFrame()
        )
        payload = (box_size, sample_records, global_summary_by_horizon)
        if fallback_payload is None:
            fallback_payload = payload
        if (
            not non_self_pairwise.empty
            and len(non_self_pairwise) == len(station_ids) * max(len(station_ids) - 1, 0)
            and bool((non_self_pairwise["mean"] > 0).all())
        ):
            return payload

    assert fallback_payload is not None
    return fallback_payload


def draw_station_hyperedge_graph(
    *,
    station_positions: pd.DataFrame,
    binary_hyperedges: pd.DataFrame,
    horizon_label: str,
    out_path: Path,
    box_size: float | None = None,
    arrow_mutation_scale: float = 18.0,
    arrow_shrink_target: float = 8.0,
) -> None:
    position_map = {
        row["station_id"]: (float(row["lon"]), float(row["lat"]))
        for _, row in station_positions.iterrows()
    }
    fig, ax = plt.subplots(figsize=(12.4, 7.6), constrained_layout=True)
    ax.scatter(station_positions["lon"], station_positions["lat"], color="#D8DEE9", s=92, zorder=5)
    for _, row in station_positions.iterrows():
        ax.text(
            row["lon"] + 0.004,
            row["lat"] + 0.002,
            row["station_id"],
            fontsize=7.4,
            color="#233142",
            zorder=6,
        )

    if not binary_hyperedges.empty:
        max_edge = max(float(binary_hyperedges["mean"].abs().max()), 1e-6)
        for _, row in binary_hyperedges.iterrows():
            source_left, source_right = tuple(row["source_station_ids"])
            target_id = row["target_station_id"]
            left_x, left_y = position_map[source_left]
            right_x, right_y = position_map[source_right]
            target_x, target_y = position_map[target_id]
            midpoint_x = 0.5 * (left_x + right_x)
            midpoint_y = 0.5 * (left_y + right_y)
            junction_x = 0.58 * midpoint_x + 0.42 * target_x
            junction_y = 0.58 * midpoint_y + 0.42 * target_y
            width = 1.0 + 3.0 * abs(float(row["mean"])) / max_edge

            for source_x, source_y in ((left_x, left_y), (right_x, right_y)):
                ax.plot(
                    [source_x, junction_x],
                    [source_y, junction_y],
                    color=HYPEREDGE_COLOR,
                    linewidth=width,
                    linestyle=(0, (5, 3)),
                    alpha=0.8,
                    zorder=2,
                )
            ax.scatter(junction_x, junction_y, s=34, facecolors="white", edgecolors=HYPEREDGE_COLOR, zorder=4)
            ax.annotate(
                "",
                xy=(target_x, target_y),
                xytext=(junction_x, junction_y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": HYPEREDGE_COLOR,
                    "linewidth": width,
                    "alpha": 0.88,
                    "linestyle": (0, (5, 3)),
                    "shrinkA": 8,
                    "shrinkB": arrow_shrink_target,
                    "mutation_scale": arrow_mutation_scale,
                },
                zorder=3,
            )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.set_title(
        _format_graph_title(f"Shanghai O3 station-level binary hyperedges ({horizon_label})", box_size=box_size),
        fontsize=13,
    )
    legend_handles = [
        Line2D([0], [0], color=HYPEREDGE_COLOR, linewidth=2.4, linestyle=(0, (5, 3)), label="O3 binary hyperedge"),
        Line2D([0], [0], marker="o", color=HYPEREDGE_COLOR, markerfacecolor="white", linewidth=0, label="Hyperedge junction"),
    ]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_figure(fig, out_path)


def draw_global_hyperedge_topk_plot(
    *,
    ranked_hyperedges: pd.DataFrame,
    out_path: Path,
    top_k: int = 20,
    box_size: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12.6, 5.8), constrained_layout=True)
    plot_frame = ranked_hyperedges.head(top_k).copy()
    if plot_frame.empty:
        ax.text(0.5, 0.5, "No binary hyperedges available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        _save_figure(fig, out_path)
        return
    colors = [HYPEREDGE_COLOR if float(value) >= 0 else "#B04A5A" for value in plot_frame["mean"]]
    ax.bar(range(len(plot_frame)), plot_frame["mean"].abs(), color=colors, alpha=0.86)
    ax.set_xticks(range(len(plot_frame)))
    ax.set_xticklabels(plot_frame["hyperedge_label"], rotation=60, ha="right", fontsize=8.5)
    ax.set_xlabel("Global hyperedge rank")
    ax.set_ylabel("Absolute mean synergy")
    ax.set_title(
        _format_graph_title(f"Global top {len(plot_frame)} station-level binary hyperedges (1h)", box_size=box_size),
        fontsize=13,
    )
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.6)
    legend_handles = [
        Line2D([0], [0], color=HYPEREDGE_COLOR, linewidth=6, label="Positive synergy"),
        Line2D([0], [0], color="#B04A5A", linewidth=6, label="Negative synergy"),
    ]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_figure(fig, out_path)


def draw_global_hyperedge_cumulative_plot(
    *,
    ranked_hyperedges: pd.DataFrame,
    out_path: Path,
    box_size: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    if ranked_hyperedges.empty:
        ax.text(0.5, 0.5, "No binary hyperedges available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        _save_figure(fig, out_path)
        return
    ax.plot(
        ranked_hyperedges["rank"],
        ranked_hyperedges["cumulative_share"],
        color=HYPEREDGE_COLOR,
        linewidth=2.4,
        marker="o",
        markersize=4.6,
    )
    for share in (0.5, 0.75, 0.9):
        ax.axhline(share, color="#8793A0", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xlabel("Top-k hyperedges")
    ax.set_ylabel("Cumulative absolute-strength share")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(
        _format_graph_title("Cumulative coverage of global binary hyperedge strength (1h)", box_size=box_size),
        fontsize=13,
    )
    ax.grid(True, alpha=0.18, linewidth=0.6)
    _save_figure(fig, out_path)


def prepare_shanghai_one_step_bundle(
    *,
    cfg: YRDExperimentConfig,
    city_en: str,
    run_tag: str,
    use_smoke: bool,
    coupling_sample_count: int,
    graph_edges_per_target: int,
    causal_graph_edge_keep_fraction: float,
    causal_graph_arrow_mutation_scale: float,
    causal_graph_arrow_shrink_target: float,
    causal_graph_variable: str = "O3",
) -> dict[str, object]:
    if use_smoke:
        smoke_root = Path(tempfile.gettempdir()) / "eisyn_yrd_smoke"
        cache_dir = smoke_root / "exp" / "cache" / "yrd_coupling" / run_tag
        results_dir = smoke_root / "fig" / "yrd_shanghai" / "artifacts" / run_tag
    else:
        cache_dir = cfg.cache_dir / run_tag
        results_dir = cfg.results_dir / run_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {
        "config": cache_dir / "config.json",
        "checkpoint": cache_dir / "joint_model_checkpoint.pt",
        "loss_history": cache_dir / "loss_history.json",
        "predictions": cache_dir / "test_predictions.npz",
        "metrics": cache_dir / "metrics_summary.json",
        "global_coupling_samples": cache_dir / "o3_station_coupling_samples.json",
        "global_coupling_summary": cache_dir / "o3_station_causal_graph_summary.json",
        "run_manifest": results_dir / "run_manifest.json",
        "global_graph_1h": results_dir / "o3_station_causal_graph_1h.png",
        "global_graph_edge_weights_1h": results_dir / "o3_station_causal_graph_edge_weights_1h.csv",
        "global_graph_degree_summary_1h": results_dir / "o3_station_causal_graph_degree_summary_1h.csv",
        "global_graph_diagnostics_1h": results_dir / "o3_station_causal_graph_diagnostics_1h.png",
        "global_hypergraph_1h": results_dir / "o3_station_binary_hypergraph_1h.png",
        "global_hyperedge_topk_1h": results_dir / "o3_station_binary_hyperedge_topk_1h.png",
        "global_hyperedge_cumulative_1h": results_dir / "o3_station_binary_hyperedge_cumulative_1h.png",
        "conditional_synergy_graph_1h": results_dir / "o3_pm25_conditional_synergy_graph_1h.png",
        "conditional_synergy_ratio_graph_1h": results_dir / "o3_pm25_conditional_synergy_ratio_graph_1h.png",
        "conclusion": results_dir / "shanghai_one_step_conclusion.md",
    }

    ds, metadata = load_dataset(cfg, smoke=use_smoke, city_en=city_en)
    full_ds, _ = load_dataset(cfg, smoke=False, city_en=city_en)
    full_city_metadata = select_station_metadata(
        load_station_metadata(cfg),
        available_station_ids=full_ds["station"].values.tolist(),
        city_en=city_en,
    )
    sample_bundle = build_one_step_samples(ds, metadata, cfg, smoke=use_smoke)
    splits = sample_bundle["splits"]
    stats = sample_bundle["stats"]
    target_names = sample_bundle["target_names"]
    station_ids = sample_bundle["station_ids"]
    target_width = len(cfg.target_variables)
    station_target_indices = build_station_variable_index_map(target_names, causal_graph_variable)

    x_train = splits["train"]["X"]
    x_val = splits["val"]["X"]
    x_test = splits["test"]["X"]
    y_train_scaled = splits["train"]["targets"]
    y_val_scaled = splits["val"]["targets"]
    y_test_scaled = splits["test"]["targets"]
    target_dim = y_train_scaled[1].shape[1]
    effective_input_dim = sample_bundle["n_stations"] * sample_bundle["n_features"]
    full_effective_input_dim = len(full_city_metadata) * sample_bundle["n_features"]
    requested_box_size_by_variable = (
        {name: float(value) for name, value in cfg.causal_graph_box_size_by_variable.items()}
        if cfg.causal_graph_box_size_by_variable is not None
        else None
    )
    nonnegative_variables = tuple(str(name) for name in cfg.causal_graph_nonnegative_variables)
    box_label = resolve_causal_graph_box_label(
        box_size=None if requested_box_size_by_variable is not None else cfg.box_size,
        box_size_by_variable=requested_box_size_by_variable,
    )

    run_context = {
        "run_tag": run_tag,
        "city_en": city_en,
        "test_mode": use_smoke,
        "use_smoke": use_smoke,
        "sample_mode": cfg.sample_mode,
        "data_resolution_hours": 1,
        "history_hours": cfg.history_hours,
        "horizons": list(cfg.horizons),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "hidden_dim": cfg.hidden_dim,
        "model_name": cfg.model_name,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "norm_type": cfg.norm_type,
        "activation": cfg.activation,
        "station_count": len(station_ids),
        "input_shape": [sample_bundle["n_stations"], sample_bundle["n_features"]],
        "effective_input_dim": int(effective_input_dim),
        "full_effective_input_dim": int(full_effective_input_dim),
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "coupling_sample_count": coupling_sample_count,
        "graph_edges_per_target": graph_edges_per_target,
        "causal_graph_variable": causal_graph_variable,
        "causal_graph_sampling_mode": "uniform_box_centered_at_train_mean",
        "causal_graph_center_source": "train_input_mean",
        "causal_graph_sampling_seed": int(cfg.seed),
        "causal_graph_edge_keep_fraction": float(causal_graph_edge_keep_fraction),
        "causal_graph_arrow_mutation_scale": float(causal_graph_arrow_mutation_scale),
        "causal_graph_arrow_shrink_target": float(causal_graph_arrow_shrink_target),
    }
    if requested_box_size_by_variable is None:
        run_context["requested_causal_graph_box_size"] = float(cfg.box_size)
        run_context["causal_graph_box_size"] = float(cfg.box_size)
    else:
        run_context["requested_causal_graph_box_size_by_variable"] = requested_box_size_by_variable
        run_context["causal_graph_box_size_by_variable"] = dict(requested_box_size_by_variable)
        run_context["causal_graph_nonnegative_variables"] = list(nonnegative_variables)
        run_context["causal_graph_box_label"] = str(box_label)
    save_json(artifact_paths["config"], run_context)
    return {
        "cfg": cfg,
        "artifact_paths": artifact_paths,
        "sample_bundle": sample_bundle,
        "splits": splits,
        "stats": stats,
        "target_names": target_names,
        "station_ids": station_ids,
        "target_width": target_width,
        "station_target_indices": station_target_indices,
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train_scaled": y_train_scaled,
        "y_val_scaled": y_val_scaled,
        "y_test_scaled": y_test_scaled,
        "target_dim": target_dim,
        "effective_input_dim": int(effective_input_dim),
        "full_effective_input_dim": int(full_effective_input_dim),
        "run_context": run_context,
        "full_city_metadata": full_city_metadata,
        "coupling_sample_count": coupling_sample_count,
        "graph_edges_per_target": graph_edges_per_target,
        "causal_graph_edge_keep_fraction": float(causal_graph_edge_keep_fraction),
        "causal_graph_arrow_mutation_scale": float(causal_graph_arrow_mutation_scale),
        "causal_graph_arrow_shrink_target": float(causal_graph_arrow_shrink_target),
        "causal_graph_variable": causal_graph_variable,
    }


def run_or_load_one_step_predictions(
    bundle: dict[str, object],
    *,
    force_retrain: bool,
) -> dict[str, object]:
    cfg = bundle["cfg"]
    artifact_paths = bundle["artifact_paths"]
    target_dim = bundle["target_dim"]
    target_names = bundle["target_names"]
    stats = bundle["stats"]
    x_test = bundle["x_test"]
    y_test_scaled = bundle["y_test_scaled"]

    assert isinstance(cfg, YRDExperimentConfig)
    assert isinstance(artifact_paths, dict)
    assert isinstance(target_names, list)
    assert isinstance(stats, dict)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_test_scaled, dict)

    set_seed(cfg.seed)
    baseline_model = PersistenceBaseline(target_dim=int(target_dim), horizons=cfg.horizons)

    if (
        Path(artifact_paths["checkpoint"]).exists()
        and Path(artifact_paths["loss_history"]).exists()
        and not force_retrain
    ):
        checkpoint_payload = torch.load(artifact_paths["checkpoint"], map_location="cpu")
        if "model_kwargs" not in checkpoint_payload:
            raise RuntimeError(
                "Checkpoint is missing model_kwargs metadata. "
                "Set FORCE_RETRAIN = True to regenerate caches with the one-step format."
            )
        joint_model = rebuild_joint_model_from_checkpoint(checkpoint_payload)
        loss_history_payload = load_json(artifact_paths["loss_history"])
    else:
        training_result = train_joint_model_with_history(
            n_stations=bundle["sample_bundle"]["n_stations"],
            n_features=bundle["sample_bundle"]["n_features"],
            history_hours=cfg.history_hours,
            target_dim=int(target_dim),
            hidden_dim=cfg.hidden_dim,
            horizons=cfg.horizons,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            max_epochs=cfg.max_epochs,
            early_stopping_patience=cfg.early_stopping_patience,
            seed=cfg.seed,
            x_train=bundle["x_train"],
            y_train=bundle["y_train_scaled"],
            x_val=bundle["x_val"],
            y_val=bundle["y_val_scaled"],
            model_name=cfg.model_name,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            norm_type=cfg.norm_type,
            activation=cfg.activation,
        )
        joint_model = training_result["model"]
        checkpoint_payload = {
            "state_dict": joint_model.state_dict(),
            "best_epoch": training_result["best_epoch"],
            "best_val_loss": training_result["best_val_loss"],
            "train_loss_history": training_result["train_loss_history"],
            "val_loss_history": training_result["val_loss_history"],
            "model_kwargs": training_result["model_kwargs"],
            "run_context": bundle["run_context"],
        }
        torch.save(checkpoint_payload, artifact_paths["checkpoint"])
        loss_history_payload = {
            "best_epoch": training_result["best_epoch"],
            "best_val_loss": training_result["best_val_loss"],
            "train_loss_history": training_result["train_loss_history"],
            "val_loss_history": training_result["val_loss_history"],
        }
        save_json(artifact_paths["loss_history"], loss_history_payload)

    joint_model.eval()
    baseline_scaled_predictions = _predict_numpy(baseline_model, x_test, cfg.horizons)
    joint_scaled_predictions = _predict_numpy(joint_model, x_test, cfg.horizons)

    y_test_original = {1: inverse_transform_targets(y_test_scaled[1], target_names, stats)}
    baseline_original_predictions = {
        1: inverse_transform_targets(baseline_scaled_predictions[1], target_names, stats)
    }
    joint_original_predictions = {
        1: inverse_transform_targets(joint_scaled_predictions[1], target_names, stats)
    }

    np.savez(
        artifact_paths["predictions"],
        y_test_scaled_1h=y_test_scaled[1],
        baseline_scaled_1h=baseline_scaled_predictions[1],
        joint_scaled_1h=joint_scaled_predictions[1],
        y_test_original_1h=y_test_original[1],
        baseline_original_1h=baseline_original_predictions[1],
        joint_original_1h=joint_original_predictions[1],
    )

    run_manifest = {
        "cache_dir": str(Path(artifact_paths["config"]).parent),
        "results_dir": str(Path(artifact_paths["run_manifest"]).parent),
    }
    for key in ("config", "checkpoint", "loss_history", "predictions"):
        path = Path(artifact_paths[key])
        if path.exists():
            run_manifest[key] = str(path)
    save_json(artifact_paths["run_manifest"], run_manifest)
    return {
        "baseline_model": baseline_model,
        "joint_model": joint_model,
        "checkpoint_payload": checkpoint_payload,
        "loss_history_payload": loss_history_payload,
        "baseline_scaled_predictions": baseline_scaled_predictions,
        "joint_scaled_predictions": joint_scaled_predictions,
        "y_test_original": y_test_original,
        "baseline_original_predictions": baseline_original_predictions,
        "joint_original_predictions": joint_original_predictions,
        "run_manifest": run_manifest,
    }


def build_prediction_tables(
    bundle: dict[str, object],
    predictions: dict[str, object],
) -> dict[str, object]:
    cfg = bundle["cfg"]
    target_names = bundle["target_names"]
    station_ids = bundle["station_ids"]
    target_width = bundle["target_width"]

    assert isinstance(cfg, YRDExperimentConfig)
    assert isinstance(target_names, list)
    assert isinstance(station_ids, list)
    assert isinstance(target_width, int)

    prediction_sets = {
        "PersistenceBaseline": predictions["baseline_original_predictions"],
        "JointStationMLP": predictions["joint_original_predictions"],
    }
    metrics_frame = pd.DataFrame(
        metric_rows_for_scope(
            predictions["y_test_original"],
            prediction_sets,
            target_names=target_names,
        )
    )
    metrics_overall_df = metrics_frame[metrics_frame["scope"] == "overall"].reset_index(drop=True)
    metrics_pollutant_df = metrics_frame[metrics_frame["scope"].isin(cfg.target_variables)].reset_index(drop=True)
    station_metrics_df = pd.DataFrame(
        station_metric_rows(
            predictions["y_test_original"],
            prediction_sets,
            station_ids=station_ids,
            target_width=target_width,
        )
    )

    save_json(
        bundle["artifact_paths"]["metrics"],
        {
            "overall": metrics_overall_df.to_dict(orient="records"),
            "by_pollutant": metrics_pollutant_df.to_dict(orient="records"),
            "by_station": station_metrics_df.to_dict(orient="records"),
        },
    )
    return {
        "prediction_sets": prediction_sets,
        "metrics_frame": metrics_frame,
        "metrics_overall_df": metrics_overall_df,
        "metrics_pollutant_df": metrics_pollutant_df,
        "station_metrics_df": station_metrics_df,
    }


def compute_station_causal_graph_results(
    bundle: dict[str, object],
    predictions: dict[str, object],
    *,
    force_recompute: bool,
    coupling_method: str = "nis",
) -> dict[str, object]:
    cfg = bundle["cfg"]
    x_train = bundle["x_train"]
    station_ids = bundle["station_ids"]
    sample_bundle = bundle["sample_bundle"]
    causal_graph_variable = bundle["causal_graph_variable"]
    graph_edges_per_target = bundle["graph_edges_per_target"]
    causal_graph_edge_keep_fraction = float(bundle["causal_graph_edge_keep_fraction"])
    causal_graph_arrow_mutation_scale = float(bundle["causal_graph_arrow_mutation_scale"])
    causal_graph_arrow_shrink_target = float(bundle["causal_graph_arrow_shrink_target"])
    run_context = bundle["run_context"]
    requested_box_size_by_variable = run_context.get("requested_causal_graph_box_size_by_variable")
    if requested_box_size_by_variable is not None:
        requested_box_size_by_variable = {
            str(name): float(value)
            for name, value in dict(requested_box_size_by_variable).items()
        }
    requested_box_size = (
        float(run_context.get("requested_causal_graph_box_size", cfg.box_size))
        if requested_box_size_by_variable is None
        else None
    )
    nonnegative_variables = tuple(str(name) for name in run_context.get("causal_graph_nonnegative_variables", ()))
    sampling_mode = str(run_context.get("causal_graph_sampling_mode", "uniform_box_centered_at_train_mean"))
    sampling_seed = int(run_context.get("causal_graph_sampling_seed", cfg.seed))
    coupling_sample_count = int(bundle["coupling_sample_count"])
    coupling_method = str(coupling_method).lower()

    assert isinstance(cfg, YRDExperimentConfig)
    assert isinstance(x_train, np.ndarray)
    assert isinstance(station_ids, list)
    assert isinstance(sample_bundle, dict)
    assert isinstance(causal_graph_variable, str)
    assert isinstance(graph_edges_per_target, int)

    station_source_groups = build_one_step_station_source_groups(
        n_stations=sample_bundle["n_stations"],
        n_features=sample_bundle["n_features"],
        station_ids=station_ids,
    )
    station_pollutant_feature_groups = build_one_step_station_pollutant_feature_groups(
        n_stations=sample_bundle["n_stations"],
        n_features=sample_bundle["n_features"],
        pollutant_feature_indices={
            "O3": cfg.input_variables.index("O3"),
            "PM2.5": cfg.input_variables.index("PM2.5"),
        },
        station_ids=station_ids,
    )
    sigma_eps_by_horizon = {
        1: estimate_residual_covariance(
            bundle["y_test_scaled"][1],
            predictions["joint_scaled_predictions"][1],
        )
    }
    artifact_paths = bundle["artifact_paths"]

    sample_cache_path = Path(artifact_paths["global_coupling_samples"])
    summary_cache_path = Path(artifact_paths["global_coupling_summary"])
    has_cache = sample_cache_path.exists() and summary_cache_path.exists() and not force_recompute
    if has_cache:
        global_coupling_sample_records = load_json(artifact_paths["global_coupling_samples"])["records"]
        global_coupling_summary_by_horizon = load_json(artifact_paths["global_coupling_summary"])
        horizon_cache = dict(global_coupling_summary_by_horizon.get("1h", {}))
        has_binary_cache = "binary_hyperedges" in horizon_cache
        has_synergy_cache = "conditional_synergy_edges" in horizon_cache
        has_ratio_cache = "conditional_synergy_ratio_edges" in horizon_cache
        cache_meta = dict(global_coupling_summary_by_horizon.get("_meta", {}))
        if (
            not has_binary_cache
            or not has_synergy_cache
            or not has_ratio_cache
            or str(cache_meta.get("coupling_method", "nis")) != coupling_method
            or not causal_graph_cache_is_valid(
                cache_meta=cache_meta,
                requested_box_size=requested_box_size,
                requested_box_size_by_variable=requested_box_size_by_variable,
                nonnegative_variables=nonnegative_variables,
                coupling_sample_count=coupling_sample_count,
                sampling_mode=sampling_mode,
                sampling_seed=sampling_seed,
            )
        ):
            has_cache = False
        else:
            if requested_box_size_by_variable is None:
                run_context["causal_graph_box_size"] = float(cache_meta["causal_graph_box_size"])
            else:
                run_context["causal_graph_box_size_by_variable"] = dict(cache_meta["causal_graph_box_size_by_variable"])
                run_context["causal_graph_nonnegative_variables"] = list(cache_meta["causal_graph_nonnegative_variables"])
            cached_pairwise_edges = pd.DataFrame(global_coupling_summary_by_horizon["1h"]["pairwise_edges"])
            cached_non_self_pairwise = cached_pairwise_edges[
                cached_pairwise_edges["source_station_id"] != cached_pairwise_edges["target_station_id"]
            ].copy()
            if cached_non_self_pairwise.empty or not bool((cached_non_self_pairwise["mean"] > 0).all()):
                has_cache = False
        if has_cache:
            run_context["coupling_method"] = coupling_method
            save_json(artifact_paths["config"], run_context)
    if not has_cache:
        intervention_center = compute_training_input_center(x_train)
        sampling_box_size: float | np.ndarray
        sampling_lower_bounds: np.ndarray | None = None
        if requested_box_size_by_variable is None:
            sampling_box_size = float(requested_box_size)
        else:
            sampling_box_size = resolve_variable_box_size_by_feature(
                input_variables=cfg.input_variables,
                box_size_by_variable=requested_box_size_by_variable,
                n_stations=sample_bundle["n_stations"],
            )
            sampling_lower_bounds = resolve_nonnegative_lower_bounds_by_feature(
                input_variables=cfg.input_variables,
                stats=bundle["stats"],
                nonnegative_variables=nonnegative_variables,
                n_stations=sample_bundle["n_stations"],
            )
        synthetic_inputs = sample_uniform_box_inputs(
            center=intervention_center,
            box_size=sampling_box_size,
            sample_count=coupling_sample_count,
            seed=sampling_seed,
            lower_bounds=sampling_lower_bounds,
        )
        joint_model = predictions["joint_model"]
        joint_model.eval()
        if coupling_method == "nis":
            if requested_box_size_by_variable is not None:
                raise ValueError("Variable-specific causal-graph box sizes are currently supported only for TM coupling.")
            raw_coupling_records: list[dict[str, object]] = []
            for sample_id, synthetic_input in enumerate(synthetic_inputs):
                sample_x = torch.from_numpy(synthetic_input[None, ...]).to(dtype=torch.float32)
                flat_sample = sample_x.reshape(-1).detach().clone().requires_grad_(True)

                def horizon_model(tensor: torch.Tensor) -> torch.Tensor:
                    shaped = tensor.reshape(1, sample_bundle["n_stations"], sample_bundle["n_features"])
                    return joint_model(shaped)[1].reshape(-1)

                for target_station_id, target_indices in bundle["station_target_indices"].items():
                    jacobian = jacobian_for_target_subset(
                        horizon_model,
                        flat_sample,
                        target_indices=target_indices,
                    ).detach().cpu().numpy()
                    raw_coupling_records.append(
                        {
                            "sample_id": int(sample_id),
                            "sampling_mode": sampling_mode,
                            "target_station_id": target_station_id,
                            "target_indices": list(target_indices),
                            "jacobian": jacobian.tolist(),
                            "sigma_eps": sigma_eps_by_horizon[1].tolist(),
                        }
                    )

            selected_box_size, global_coupling_sample_records, global_coupling_summary_by_horizon = resolve_causal_graph_box_size(
                raw_records=raw_coupling_records,
                station_ids=station_ids,
                station_source_groups=station_source_groups,
                station_pollutant_feature_groups=station_pollutant_feature_groups,
                requested_box_size=float(requested_box_size),
                graph_edges_per_target=graph_edges_per_target,
                causal_graph_variable=causal_graph_variable,
            )
        elif coupling_method == "tm":
            with torch.no_grad():
                predicted_next = joint_model(torch.from_numpy(synthetic_inputs).to(dtype=torch.float32))[1].detach().cpu().numpy()
            predicted_next_o3 = np.stack(
                [
                    predicted_next[:, bundle["station_target_indices"][station_id][0]]
                    for station_id in station_ids
                ],
                axis=1,
            )
            selected_box_size = requested_box_size
            global_coupling_sample_records = [
                {
                    "sample_id": int(sample_id),
                    "sampling_mode": sampling_mode,
                    "source_sample": synthetic_inputs[sample_id].tolist(),
                    "predicted_next_o3": predicted_next_o3[sample_id].tolist(),
                }
                for sample_id in range(synthetic_inputs.shape[0])
            ]
            global_coupling_summary_by_horizon = build_transport_map_global_causal_summary(
                source_samples=synthetic_inputs,
                predicted_next_o3=predicted_next_o3,
                station_ids=station_ids,
                o3_feature_index=cfg.input_variables.index("O3"),
                pm25_feature_index=cfg.input_variables.index("PM2.5"),
            )
        else:
            raise ValueError(f"Unsupported coupling method: {coupling_method}")
        meta: dict[str, object] = {
            "coupling_method": coupling_method,
            "coupling_sample_count": coupling_sample_count,
            "sampling_mode": sampling_mode,
            "sampling_seed": sampling_seed,
        }
        if requested_box_size_by_variable is None:
            meta["requested_causal_graph_box_size"] = float(requested_box_size)
            meta["causal_graph_box_size"] = float(selected_box_size)
            run_context["causal_graph_box_size"] = float(selected_box_size)
        else:
            meta["requested_causal_graph_box_size_by_variable"] = dict(requested_box_size_by_variable)
            meta["causal_graph_box_size_by_variable"] = dict(requested_box_size_by_variable)
            meta["causal_graph_nonnegative_variables"] = list(nonnegative_variables)
            run_context["causal_graph_box_size_by_variable"] = dict(requested_box_size_by_variable)
            run_context["causal_graph_nonnegative_variables"] = list(nonnegative_variables)
        global_coupling_summary_by_horizon["_meta"] = meta
        run_context["coupling_method"] = coupling_method
        save_json(artifact_paths["config"], run_context)
        save_json(artifact_paths["global_coupling_samples"], {"records": global_coupling_sample_records})
        save_json(artifact_paths["global_coupling_summary"], global_coupling_summary_by_horizon)

    pairwise_edges_df = pd.DataFrame(global_coupling_summary_by_horizon["1h"]["pairwise_edges"])
    binary_hyperedges_df = pd.DataFrame(global_coupling_summary_by_horizon["1h"]["binary_hyperedges"])
    conditional_synergy_edges_df = pd.DataFrame(global_coupling_summary_by_horizon["1h"]["conditional_synergy_edges"])
    conditional_synergy_ratio_edges_df = pd.DataFrame(
        global_coupling_summary_by_horizon["1h"]["conditional_synergy_ratio_edges"]
    )
    pairwise_self_loop_strengths_1h = build_self_loop_node_strengths(pairwise_edges_df, station_ids=station_ids)
    conditional_synergy_self_loop_strengths_1h = build_self_loop_node_strengths(
        conditional_synergy_edges_df,
        station_ids=station_ids,
    )
    conditional_synergy_ratio_self_loop_strengths_1h = build_self_loop_node_strengths(
        conditional_synergy_ratio_edges_df,
        station_ids=station_ids,
    )
    pairwise_display_1h = (
        pairwise_edges_df[
            (pairwise_edges_df["source_station_id"] != pairwise_edges_df["target_station_id"])
            & (pairwise_edges_df["mean"] > 0)
        ]
        .sort_values(["target_station_id", "mean"], ascending=[True, False])
        .reset_index(drop=True)
        if not pairwise_edges_df.empty
        else pd.DataFrame()
    )
    binary_hyperedges_display_1h = (
        select_display_hyperedge_rows(binary_hyperedges_df, per_target=graph_edges_per_target)
        if not binary_hyperedges_df.empty
        else pd.DataFrame()
    )
    conditional_synergy_edges_display_1h = (
        select_display_rows(
            conditional_synergy_edges_df,
            per_target=graph_edges_per_target,
            source_col="source_station_id",
            target_col="target_station_id",
            ranking_col="mean",
            positive_only=True,
            include_self_loops=False,
        )
        if not conditional_synergy_edges_df.empty
        else pd.DataFrame()
    )
    conditional_synergy_ratio_edges_display_1h = (
        select_display_rows(
            conditional_synergy_ratio_edges_df,
            per_target=graph_edges_per_target,
            source_col="source_station_id",
            target_col="target_station_id",
            ranking_col="mean",
            positive_only=True,
            include_self_loops=False,
        )
        if not conditional_synergy_ratio_edges_df.empty
        else pd.DataFrame()
    )
    conditional_synergy_edges_global_ranked_1h = build_global_edge_ranking(
        conditional_synergy_edges_df,
        sort_col="mean",
    )
    conditional_synergy_ratio_edges_global_ranked_1h = build_global_edge_ranking(
        conditional_synergy_ratio_edges_df,
        sort_col="mean",
    )
    binary_hyperedges_global_ranked_1h = build_global_hyperedge_ranking(binary_hyperedges_df)
    binary_hyperedge_cutoff_summary_1h = summarize_hyperedge_cutoff(binary_hyperedges_global_ranked_1h)
    box_annotation = run_context.get("causal_graph_box_label")
    if box_annotation is None:
        box_annotation = run_context.get("causal_graph_box_size")
    pairwise_visual_display_1h = select_top_fraction_edges(
        pairwise_display_1h,
        strength_col="mean",
        keep_fraction=causal_graph_edge_keep_fraction,
    )
    pairwise_render_edges_1h = build_pairwise_edge_render_frame(pairwise_visual_display_1h, strength_col="mean")
    pairwise_degree_summary_1h = build_pairwise_degree_summary(
        pairwise_display_1h,
        station_ids=station_ids,
        strength_col="mean",
    )
    pairwise_diagnostics_edges_1h = build_pairwise_edge_render_frame(pairwise_display_1h, strength_col="mean")
    conditional_synergy_visual_display_1h = select_top_fraction_edges(
        conditional_synergy_edges_display_1h,
        strength_col="mean",
        keep_fraction=causal_graph_edge_keep_fraction,
    )
    conditional_synergy_ratio_visual_display_1h = select_top_fraction_edges(
        conditional_synergy_ratio_edges_display_1h,
        strength_col="mean",
        keep_fraction=causal_graph_edge_keep_fraction,
    )
    pairwise_render_edges_1h.to_csv(artifact_paths["global_graph_edge_weights_1h"], index=False)
    pairwise_degree_summary_1h.to_csv(artifact_paths["global_graph_degree_summary_1h"], index=False)
    draw_station_causal_graph(
        station_positions=bundle["full_city_metadata"][["station_id", "lon", "lat"]],
        pairwise_edges=pairwise_visual_display_1h,
        horizon_label="1h",
        out_path=artifact_paths["global_graph_1h"],
        strength_col="mean",
        box_size=box_annotation,
        arrow_mutation_scale=causal_graph_arrow_mutation_scale,
        arrow_shrink_target=causal_graph_arrow_shrink_target,
        node_self_strengths=pairwise_self_loop_strengths_1h,
        node_colorbar_label="Self EI^tm",
    )
    draw_pairwise_ei_degree_diagnostics(
        render_edges=pairwise_diagnostics_edges_1h,
        node_degree_summary=pairwise_degree_summary_1h,
        out_path=artifact_paths["global_graph_diagnostics_1h"],
        horizon_label="1h",
        box_size=box_annotation,
    )
    draw_station_hyperedge_graph(
        station_positions=bundle["full_city_metadata"][["station_id", "lon", "lat"]],
        binary_hyperedges=binary_hyperedges_display_1h,
        horizon_label="1h",
        out_path=artifact_paths["global_hypergraph_1h"],
        box_size=box_annotation,
        arrow_mutation_scale=causal_graph_arrow_mutation_scale,
        arrow_shrink_target=causal_graph_arrow_shrink_target,
    )
    draw_global_hyperedge_topk_plot(
        ranked_hyperedges=binary_hyperedges_global_ranked_1h,
        out_path=artifact_paths["global_hyperedge_topk_1h"],
        box_size=box_annotation,
    )
    draw_global_hyperedge_cumulative_plot(
        ranked_hyperedges=binary_hyperedges_global_ranked_1h,
        out_path=artifact_paths["global_hyperedge_cumulative_1h"],
        box_size=box_annotation,
    )
    draw_station_causal_graph(
        station_positions=bundle["full_city_metadata"][["station_id", "lon", "lat"]],
        pairwise_edges=conditional_synergy_visual_display_1h,
        horizon_label="1h",
        out_path=artifact_paths["conditional_synergy_graph_1h"],
        title="Shanghai O3 + PM2.5 conditional synergy graph (1h)",
        positive_color=SYNERGY_POSITIVE_EDGE_COLOR,
        legend_label="O3 + PM2.5 conditional synergy",
        strength_col="mean",
        box_size=box_annotation,
        arrow_mutation_scale=causal_graph_arrow_mutation_scale,
        arrow_shrink_target=causal_graph_arrow_shrink_target,
        node_self_strengths=conditional_synergy_self_loop_strengths_1h,
        node_colorbar_label="Self Syn^tm",
    )
    draw_station_causal_graph(
        station_positions=bundle["full_city_metadata"][["station_id", "lon", "lat"]],
        pairwise_edges=conditional_synergy_ratio_visual_display_1h,
        horizon_label="1h",
        out_path=artifact_paths["conditional_synergy_ratio_graph_1h"],
        title="Shanghai O3 + PM2.5 synergy-to-EI ratio graph (1h)",
        positive_color=SYNERGY_RATIO_EDGE_COLOR,
        legend_label="Syn / EI_joint",
        strength_col="mean",
        box_size=box_annotation,
        arrow_mutation_scale=causal_graph_arrow_mutation_scale,
        arrow_shrink_target=causal_graph_arrow_shrink_target,
        node_self_strengths=conditional_synergy_ratio_self_loop_strengths_1h,
        node_colorbar_label="Self Syn / EI_joint",
    )
    return {
        "binary_hyperedges_df": binary_hyperedges_df,
        "binary_hyperedges_display_1h": binary_hyperedges_display_1h,
        "binary_hyperedges_global_ranked_1h": binary_hyperedges_global_ranked_1h,
        "binary_hyperedge_cutoff_summary_1h": binary_hyperedge_cutoff_summary_1h,
        "conditional_synergy_edges_df": conditional_synergy_edges_df,
        "conditional_synergy_edges_display_1h": conditional_synergy_edges_display_1h,
        "conditional_synergy_edges_global_ranked_1h": conditional_synergy_edges_global_ranked_1h,
        "conditional_synergy_self_loop_strengths_1h": conditional_synergy_self_loop_strengths_1h,
        "conditional_synergy_ratio_edges_df": conditional_synergy_ratio_edges_df,
        "conditional_synergy_ratio_edges_display_1h": conditional_synergy_ratio_edges_display_1h,
        "conditional_synergy_ratio_edges_global_ranked_1h": conditional_synergy_ratio_edges_global_ranked_1h,
        "conditional_synergy_ratio_self_loop_strengths_1h": conditional_synergy_ratio_self_loop_strengths_1h,
        "pairwise_display_1h": pairwise_display_1h,
        "pairwise_self_loop_strengths_1h": pairwise_self_loop_strengths_1h,
        "pairwise_visual_display_1h": pairwise_visual_display_1h,
        "pairwise_render_edges_1h": pairwise_render_edges_1h,
        "pairwise_degree_summary_1h": pairwise_degree_summary_1h,
        "global_coupling_summary_by_horizon": global_coupling_summary_by_horizon,
    }


def metric_value(
    frame: pd.DataFrame,
    *,
    model: str,
    horizon: str,
    scope: str,
    metric: str,
) -> float:
    row = frame[
        frame["model"].eq(model)
        & frame["horizon"].eq(horizon)
        & frame["scope"].eq(scope)
    ].iloc[0]
    return float(row[metric])


def write_shanghai_one_step_conclusion(
    bundle: dict[str, object],
    *,
    metrics_frame: pd.DataFrame,
    conditional_synergy_edges_display_1h: pd.DataFrame,
) -> str:
    sample_bundle = bundle["sample_bundle"]
    station_ids = bundle["station_ids"]
    effective_input_dim = bundle["effective_input_dim"]
    full_effective_input_dim = bundle["full_effective_input_dim"]

    assert isinstance(sample_bundle, dict)
    assert isinstance(station_ids, list)
    assert isinstance(effective_input_dim, int)
    assert isinstance(full_effective_input_dim, int)

    joint_1h_rmse = metric_value(
        metrics_frame,
        model="JointStationMLP",
        horizon="1h",
        scope="overall",
        metric="rmse",
    )
    baseline_1h_rmse = metric_value(
        metrics_frame,
        model="PersistenceBaseline",
        horizon="1h",
        scope="overall",
        metric="rmse",
    )
    strongest_conditional_synergy_edge_1h = (
        conditional_synergy_edges_display_1h.sort_values("mean", ascending=False).iloc[0]
        if not conditional_synergy_edges_display_1h.empty
        else None
    )
    run_context = bundle["run_context"]
    if "causal_graph_box_size_by_variable" in run_context:
        box_description = "按变量设置的 train-q99 干预盒 L_v，并对非负变量施加原始量纲下的 0 下界裁剪"
    else:
        box_description = f"边长 L = {float(run_context['causal_graph_box_size']):.3f} 的输入盒"

    conclusion_lines = [
        f"本 notebook 使用原始 1h 分辨率数据，在上海单城 {len(station_ids)} 个站点上构造当前时刻联合快照输入。",
        (
            f"每个样本的输入形状为 {sample_bundle['n_stations']} x {sample_bundle['n_features']}；"
            f"当前运行的有效输入维度为 {effective_input_dim}，"
            f"full Shanghai 设置下对应 {full_effective_input_dim}（即 190 维一步输入）。"
        ),
        (
            f"当前实验只做一步 1h 预测；在整体 1h 预测上，"
            f"JointStationMLP 的 RMSE 为 {joint_1h_rmse:.3f}，"
            f"PersistenceBaseline 的 RMSE 为 {baseline_1h_rmse:.3f}。"
        ),
        (
            "因果图平均不是跨真实测试时刻取样，而是对训练集输入均值居中的均匀干预样本做 Monte Carlo 平均；"
            f"采样分布为{box_description}。"
        ),
        (
            "在条件协同图分析中，source side 表示每个源站点当前时刻的 O3 + PM2.5，"
            "target side 表示下一小时目标站点的 O3，其余输入维度保留为条件背景。"
        ),
    ]
    if strongest_conditional_synergy_edge_1h is not None:
        conclusion_lines.append(
            "在当前 1h O3 + PM2.5 条件协同图里，最强的显示边是 "
            f"{strongest_conditional_synergy_edge_1h['source_station_id']} -> "
            f"{strongest_conditional_synergy_edge_1h['target_station_id']}，"
            f"其平均协同强度为 {strongest_conditional_synergy_edge_1h['mean']:.3f}。"
        )
    final_conclusion_text = "\n".join(conclusion_lines)
    Path(bundle["artifact_paths"]["conclusion"]).write_text(final_conclusion_text)
    return final_conclusion_text
