#!/usr/bin/env python3
"""Compute multi-step target-wise PhiEID-ACS with a linear Ridge surrogate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPONENT_SCORES = ROOT / "results" / "runge_slp_daily_1948_2026_20260628" / "results" / "runge" / "2015_gateways" / "component_weekly_scores.csv"
DEFAULT_COMPONENT_MAPS = ROOT / "results" / "runge_slp_daily_1948_2026_20260628" / "results" / "runge" / "2015_gateways" / "component_maps.npz"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "runge_slp_daily_1948_2026_oldstyle_ace_acs" / "ridge_multistep_phi_acs_lag04"
DEFAULT_FIGURE = ROOT / "fig" / "runge_ridge_multistep_phi_acs_1948_2026.png"


@dataclass(frozen=True)
class Config:
    component_scores: Path = DEFAULT_COMPONENT_SCORES
    component_maps: Path = DEFAULT_COMPONENT_MAPS
    output_dir: Path = DEFAULT_OUTPUT_DIR
    figure: Path = DEFAULT_FIGURE
    lag: int = 4
    horizon_max: int = 8
    ridge_alpha: float = 1000.0
    intervention_samples: int = 8192
    quantile_low: float = 0.05
    quantile_high: float = 0.95
    seed: int = 42
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    jitter: float = 1.0e-8
    exclude_self: bool = True


def _ensure_scripts_on_path() -> None:
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def _load_pairwise_module():
    _ensure_scripts_on_path()
    import run_runge_pairwise_mlp_ei as pairwise  # type: ignore

    return pairwise


def _load_map_helpers():
    _ensure_scripts_on_path()
    from plot_runge_ace_acs_oldstyle_alignment import add_panel_colorbar, select_label_nodes  # type: ignore
    from plot_runge_gateway_mediator_map import (  # type: ignore
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

    return {
        "add_panel_colorbar": add_panel_colorbar,
        "select_label_nodes": select_label_nodes,
        "COASTLINE_URL": COASTLINE_URL,
        "LAND_URL": LAND_URL,
        "add_geographic_ticks": add_geographic_ticks,
        "add_labels": add_labels,
        "component_center": component_center,
        "draw_world": draw_world,
        "extract_lines": extract_lines,
        "extract_polygons": extract_polygons,
        "load_geojson": load_geojson,
        "local_to_paper": local_to_paper,
    }


def fit_original_space_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    pairwise = _load_pairwise_module()
    x_scaled, x_mean, x_std = pairwise._standardize(train_x, train_x)
    y_scaled, y_mean, y_std = pairwise._standardize(train_y, train_y)
    weight_scaled, bias_scaled = pairwise.fit_ridge_linear_map(x_scaled, y_scaled, alpha=float(ridge_alpha))
    x_mean_1d = x_mean.reshape(-1).astype(float)
    x_std_1d = x_std.reshape(-1).astype(float)
    y_mean_1d = y_mean.reshape(-1).astype(float)
    y_std_1d = y_std.reshape(-1).astype(float)
    transition = np.asarray(weight_scaled, dtype=float) * (y_std_1d[:, None] / x_std_1d[None, :])
    intercept = y_mean_1d + y_std_1d * np.asarray(bias_scaled, dtype=float) - transition @ x_mean_1d
    scalers = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return transition, intercept, scalers


def rollout_linear(
    initial_features: np.ndarray,
    transition: np.ndarray,
    intercept: np.ndarray,
    *,
    n_components: int,
    lag: int,
    horizon_max: int,
) -> list[np.ndarray]:
    state = np.asarray(initial_features, dtype=float).copy()
    predictions: list[np.ndarray] = []
    for _ in range(int(horizon_max)):
        next_values = state @ transition.T + intercept
        predictions.append(next_values)
        state = np.concatenate([state[:, n_components:], next_values], axis=1)
    return predictions


def gaussian_mi_scalar(source: np.ndarray, target: np.ndarray, *, jitter: float) -> float:
    x = np.asarray(source, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    y = np.asarray(target, dtype=float).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("source and target must have the same number of rows.")
    if x.shape[1] == 0 or x.shape[0] < 4:
        return 0.0
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - float(y.mean())
    n = max(1, x.shape[0] - 1)
    cov_xx = (x_centered.T @ x_centered) / float(n)
    cov_xy = (x_centered.T @ y_centered) / float(n)
    var_y = float((y_centered @ y_centered) / float(n))
    if not np.isfinite(var_y) or var_y <= 0.0:
        return 0.0
    scale = max(float(np.trace(cov_xx) / max(1, cov_xx.shape[0])), var_y, 1.0)
    ridge = float(jitter) * scale
    cov_xx = cov_xx + ridge * np.eye(cov_xx.shape[0], dtype=float)
    try:
        explained = float(cov_xy @ np.linalg.solve(cov_xx, cov_xy))
    except np.linalg.LinAlgError:
        explained = float(cov_xy @ np.linalg.pinv(cov_xx) @ cov_xy)
    conditional_var = max(var_y - explained, ridge)
    return max(0.0, 0.5 * math.log(var_y / conditional_var, 2.0))


def latest_source_block(features: np.ndarray, source: int, *, n_components: int, lag: int) -> np.ndarray:
    return features[:, (int(lag) - 1) * int(n_components) + int(source)]


def latest_joint_block(features: np.ndarray, sources: Sequence[int], *, n_components: int, lag: int) -> np.ndarray:
    cols = [(int(lag) - 1) * int(n_components) + int(source) for source in sources]
    return features[:, cols]


def compute_scores(config: Config) -> tuple[pd.DataFrame, dict[str, object]]:
    pairwise = _load_pairwise_module()
    frame = pairwise.load_component_scores(config.component_scores)
    names = list(frame.columns)
    n_components = len(names)
    features, targets = pairwise.build_lagged_dataset(frame, lag=int(config.lag), horizon=1)
    splits = pairwise.split_temporal_arrays(
        features,
        targets,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
    )
    transition, intercept, _ = fit_original_space_ridge(
        splits["train"][0],
        splits["train"][1],
        ridge_alpha=float(config.ridge_alpha),
    )
    intervention_features = pairwise.sample_max_entropy_features(
        splits["train"][0],
        n_components=n_components,
        lag=int(config.lag),
        samples=int(config.intervention_samples),
        low_q=float(config.quantile_low),
        high_q=float(config.quantile_high),
        seed=int(config.seed),
    )
    predictions_by_h = rollout_linear(
        intervention_features,
        transition,
        intercept,
        n_components=n_components,
        lag=int(config.lag),
        horizon_max=int(config.horizon_max),
    )
    weights = np.ones(int(config.horizon_max), dtype=float) / float(config.horizon_max)
    rows: list[dict[str, object]] = []
    for target_index, name in enumerate(names):
        source_indices = [idx for idx in range(n_components) if (idx != target_index or not config.exclude_self)]
        joint_block = latest_joint_block(
            intervention_features,
            source_indices,
            n_components=n_components,
            lag=int(config.lag),
        )
        acs_total = 0.0
        joint_total = 0.0
        raw_phi_total = 0.0
        phi_positive_total = 0.0
        singleton_by_h: list[float] = []
        joint_by_h: list[float] = []
        phi_by_h: list[float] = []
        for h_index, predictions in enumerate(predictions_by_h, start=1):
            target_values = predictions[:, target_index]
            singleton_sum = 0.0
            for source_index in source_indices:
                source_values = latest_source_block(
                    intervention_features,
                    source_index,
                    n_components=n_components,
                    lag=int(config.lag),
                )
                singleton_sum += gaussian_mi_scalar(source_values, target_values, jitter=float(config.jitter))
            joint_ei = gaussian_mi_scalar(joint_block, target_values, jitter=float(config.jitter))
            raw_phi = joint_ei - singleton_sum
            weight = float(weights[h_index - 1])
            acs_total += weight * singleton_sum
            joint_total += weight * joint_ei
            raw_phi_total += weight * raw_phi
            phi_positive_total += weight * max(0.0, raw_phi)
            singleton_by_h.append(float(singleton_sum))
            joint_by_h.append(float(joint_ei))
            phi_by_h.append(float(raw_phi))
        rows.append(
            {
                "component": name,
                "component_index": int(target_index),
                "acs_multistep": float(acs_total),
                "joint_ei_multistep": float(joint_total),
                "phi_acs_raw": float(raw_phi_total),
                "phi_acs_positive": float(phi_positive_total),
                "phi_share_of_acs": float(phi_positive_total / acs_total) if acs_total > 0.0 else 0.0,
                "singleton_sum_by_h": json.dumps(singleton_by_h),
                "joint_ei_by_h": json.dumps(joint_by_h),
                "phi_raw_by_h": json.dumps(phi_by_h),
            }
        )
    scores = pd.DataFrame(rows)
    scores["acs_rank"] = scores["acs_multistep"].rank(ascending=False, method="min").astype(int)
    scores["phi_acs_rank"] = scores["phi_acs_positive"].rank(ascending=False, method="min").astype(int)
    scores["rank_gain_phi_vs_acs"] = scores["acs_rank"] - scores["phi_acs_rank"]
    manifest = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "n_components": int(n_components),
        "n_rows": int(len(frame)),
        "n_lagged_samples": int(len(features)),
        "splits": {key: int(len(value[0])) for key, value in splits.items()},
        "source_definition": "latest component value; target self excluded from source set" if config.exclude_self else "latest component value; target self included",
        "horizon_weights": weights.tolist(),
    }
    return scores, manifest


def load_component_nodes(component_maps_path: Path, scores: pd.DataFrame) -> pd.DataFrame:
    helpers = _load_map_helpers()
    maps = np.load(component_maps_path)["component_maps"]
    lat = np.linspace(-90.0, 90.0, maps.shape[0])
    lon = np.linspace(0.0, 360.0, maps.shape[1], endpoint=False)
    lon = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    maps = maps[:, order, :]
    lon = lon[order]
    rows = []
    for row in scores.itertuples(index=False):
        local = int(row.component_index)
        center_lon, center_lat = helpers["component_center"](maps[..., local], lat, lon)
        rows.append(
            {
                "local": local,
                "paper": int(helpers["local_to_paper"](local)),
                "lon": float(center_lon),
                "lat": float(center_lat),
                "acs": float(row.acs_multistep),
                "phi_acs": float(row.phi_acs_positive),
                "rank_gain": float(row.rank_gain_phi_vs_acs),
            }
        )
    return pd.DataFrame(rows)


def draw_score_panel(
    ax: plt.Axes,
    nodes: pd.DataFrame,
    value_col: str,
    label_nodes: pd.DataFrame,
    norm: mpl.colors.Normalize,
    cmap: mpl.colors.Colormap,
    land: list,
    coastlines: list,
) -> None:
    helpers = _load_map_helpers()
    helpers["draw_world"](ax, land, coastlines)
    helpers["add_geographic_ticks"](ax)
    values = nodes[value_col].astype(float).to_numpy()
    vmax = max(float(np.nanmax(values)), 1.0e-12)
    sizes = 170.0 + 260.0 * np.sqrt(np.clip(values / vmax, 0.0, 1.0))
    ax.scatter(
        np.radians(nodes["lon"].astype(float).to_numpy()),
        np.radians(nodes["lat"].astype(float).to_numpy()),
        s=sizes,
        c=values,
        cmap=cmap,
        norm=norm,
        edgecolors="#7f3b1d",
        linewidths=0.55,
        alpha=0.96,
        zorder=4,
    )
    helpers["add_labels"](ax, label_nodes)


def plot_maps(config: Config, scores: pd.DataFrame, manifest: dict[str, object]) -> None:
    helpers = _load_map_helpers()
    nodes = load_component_nodes(config.component_maps, scores)
    labels = nodes.copy()
    cmap = mpl.colormaps["OrRd"]
    acs_norm = mpl.colors.Normalize(vmin=0.0, vmax=float(nodes["acs"].max()))
    phi_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(float(nodes["phi_acs"].max()), 1.0e-12))
    land = helpers["extract_polygons"](helpers["load_geojson"](helpers["LAND_URL"]))
    coastlines = helpers["extract_lines"](helpers["load_geojson"](helpers["COASTLINE_URL"]))
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
        }
    )
    fig = plt.figure(figsize=(13.2, 5.1), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.06)
    axes = [fig.add_subplot(grid[0, 0], projection="mollweide"), fig.add_subplot(grid[0, 1], projection="mollweide")]
    draw_score_panel(axes[0], nodes, "acs", labels, acs_norm, cmap, land, coastlines)
    draw_score_panel(axes[1], nodes, "phi_acs", labels, phi_norm, cmap, land, coastlines)
    axes[0].set_title("Multi-step ACS: sum of singleton EI", fontsize=9, fontweight="bold", pad=8)
    axes[1].set_title("Multi-step PhiACS: positive joint surplus", fontsize=9, fontweight="bold", pad=8)
    axes[0].text(-0.06, 1.03, "a", transform=axes[0].transAxes, fontsize=16, fontweight="bold")
    axes[1].text(-0.06, 1.03, "b", transform=axes[1].transAxes, fontsize=16, fontweight="bold")
    helpers["add_panel_colorbar"](fig, axes[0], acs_norm, cmap, label="ACS, averaged over horizons")
    helpers["add_panel_colorbar"](fig, axes[1], phi_norm, cmap, label="PhiACS, averaged positive surplus")
    config.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.figure, dpi=500, bbox_inches="tight")
    fig.savefig(config.figure.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(config.figure.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    manifest["figure_png"] = str(config.figure.relative_to(ROOT) if config.figure.is_absolute() else config.figure)
    manifest["figure_svg"] = str(config.figure.with_suffix(".svg").relative_to(ROOT) if config.figure.with_suffix(".svg").is_absolute() else config.figure.with_suffix(".svg"))
    manifest["figure_pdf"] = str(config.figure.with_suffix(".pdf").relative_to(ROOT) if config.figure.with_suffix(".pdf").is_absolute() else config.figure.with_suffix(".pdf"))


def parse_args(argv: Sequence[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-scores", type=Path, default=DEFAULT_COMPONENT_SCORES)
    parser.add_argument("--component-maps", type=Path, default=DEFAULT_COMPONENT_MAPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--lag", type=int, default=4)
    parser.add_argument("--horizon-max", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=1000.0)
    parser.add_argument("--intervention-samples", type=int, default=8192)
    parser.add_argument("--quantile-low", type=float, default=0.05)
    parser.add_argument("--quantile-high", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--jitter", type=float, default=1.0e-8)
    parser.add_argument("--include-self", dest="exclude_self", action="store_false")
    parser.set_defaults(exclude_self=True)
    return Config(**vars(parser.parse_args(argv)))


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    scores, manifest = compute_scores(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = config.output_dir / "multistep_phi_acs_scores.csv"
    manifest_path = config.output_dir / "manifest.json"
    scores.to_csv(scores_path, index=False)
    manifest["scores_csv"] = str(scores_path.relative_to(ROOT) if scores_path.is_absolute() else scores_path)
    manifest["top_acs"] = scores.sort_values("acs_multistep", ascending=False).head(10).to_dict("records")
    manifest["top_phi_acs"] = scores.sort_values("phi_acs_positive", ascending=False).head(10).to_dict("records")
    plot_maps(config, scores, manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"scores": str(scores_path), "manifest": str(manifest_path), "figure": str(config.figure)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
