#!/usr/bin/env python3
"""Run full order-2 Scheme A conditioned EI on the 1948-2026 Runge SLP run.

This script estimates every joint EI term EI({i, r} -> j) needed by Scheme A:
for each target j, all unordered source pairs that exclude j are evaluated.
It writes one CSV per target so the run can resume without recomputing
completed targets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_runge_pairwise_mlp_ei as pairwise  # noqa: E402
from scripts import run_runge_peid_hypergraph as peid  # noqa: E402
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
from exp.TM.transport_map_density import estimate_mutual_information_transport_map  # noqa: E402

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

DEFAULT_RUN_ROOT = Path("results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04")
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_ROOT / "results/runge/scheme_a_full_order2"
DEFAULT_FIG_DIR = DEFAULT_RUN_ROOT / "fig/runge/scheme_a_full_order2"
DEFAULT_COMPONENT_MAPS = Path("results/runge_slp_daily_1948_2026_20260628/results/runge/2015_gateways/component_maps.npz")
DEFAULT_COMPONENT_SCORES = Path("results/runge_slp_daily_1948_2026_20260628/results/runge/2015_gateways/component_weekly_scores.csv")
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


def parse_int_list(text: str | None) -> list[int] | None:
    if text is None or not str(text).strip():
        return None
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def load_peid_config(args: argparse.Namespace) -> peid.PeidHypergraphConfig:
    return peid.PeidHypergraphConfig(
        component_scores=Path(args.component_scores),
        output_dir=Path(args.run_root),
        lag=int(args.lag),
        horizon=1,
        hidden_dim=128,
        num_layers=1,
        dropout=0.5,
        epochs=120,
        learning_rate=0.001,
        batch_size=256,
        weight_decay=0.001,
        ridge_alpha=1000.0,
        ensemble_ridge_alphas=(10.0, 100.0, 1000.0, 3000.0),
        linear_blend_grid_steps=101,
        freeze_linear_skip=True,
        residual_shrinkage=True,
        residual_gamma_min=-0.5,
        residual_gamma_max=0.5,
        residual_gamma_steps=101,
        early_stopping_patience=80,
        scheduler_patience=20,
        gradient_clip_norm=1.0,
        intervention_samples=int(args.intervention_samples),
        quantile_low=0.05,
        quantile_high=0.95,
        source_mode="latest",
        seed=42,
        train_fraction=0.70,
        val_fraction=0.15,
        force_retrain=False,
        order_max=2,
        null_reps=0,
        pairwise_matrix_path=Path(args.pairwise_matrix),
        pairwise_gateway_path=Path(args.pairwise_gateway),
        pairwise_mediator_path=Path(args.pairwise_mediator),
    )


def prepare_intervention_predictions(config: peid.PeidHypergraphConfig) -> tuple[np.ndarray, np.ndarray, list[str], int, object]:
    component_scores_path = peid._resolve_path(config.component_scores)
    frame = pairwise.load_component_scores(component_scores_path)
    data_hash = pairwise._frame_content_hash(frame)
    names = list(frame.columns)
    n_components = len(names)
    features, targets = pairwise.build_lagged_dataset(frame, lag=int(config.lag), horizon=int(config.horizon))
    splits = pairwise.split_temporal_arrays(
        features,
        targets,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
    )
    proxy = pairwise.PairwiseMlpEiConfig(
        component_scores=config.component_scores,
        output_dir=config.output_dir,
        lag=config.lag,
        horizon=config.horizon,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        ridge_alpha=config.ridge_alpha,
        ensemble_ridge_alphas=config.ensemble_ridge_alphas,
        linear_blend_grid_steps=config.linear_blend_grid_steps,
        freeze_linear_skip=config.freeze_linear_skip,
        residual_shrinkage=config.residual_shrinkage,
        residual_gamma_min=config.residual_gamma_min,
        residual_gamma_max=config.residual_gamma_max,
        residual_gamma_steps=config.residual_gamma_steps,
        early_stopping_patience=config.early_stopping_patience,
        min_delta=config.min_delta,
        scheduler_patience=config.scheduler_patience,
        gradient_clip_norm=config.gradient_clip_norm,
        intervention_samples=config.intervention_samples,
        bins=8,
        ei_estimator="tm",
        gateway_mode="pairwise",
        graph_sparsify="source_topk",
        graph_topk=5,
        graph_quantile=0.95,
        path_alpha=0.8,
        quantile_low=config.quantile_low,
        quantile_high=config.quantile_high,
        source_mode=config.source_mode,
        seed=config.seed,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        force_retrain=config.force_retrain,
    )
    ensemble_alphas = tuple(config.ensemble_ridge_alphas) if config.ensemble_ridge_alphas else (float(config.ridge_alpha),)
    model_dir = Path(config.output_dir) / "results/runge/pairwise_mlp_ei"
    models = []
    scalers = None
    member_summaries = []
    for alpha in ensemble_alphas:
        member_config = replace(proxy, ridge_alpha=float(alpha), ensemble_ridge_alphas=())
        member_hash = pairwise._model_config_hash(
            member_config,
            n_components=n_components,
            n_rows=len(frame),
            data_hash=data_hash,
        )
        alpha_label = str(float(alpha)).replace("-", "m").replace(".", "p")
        member_path = model_dir / f"mlp_transition_alpha{alpha_label}_{member_hash}.pt"
        member_model, member_scalers, _, cache_reused = pairwise.train_or_load_mlp(
            splits,
            member_config,
            member_path,
            config_hash=member_hash,
        )
        models.append(member_model)
        scalers = member_scalers if scalers is None else scalers
        member_summaries.append({"ridge_alpha": float(alpha), "model_cache": str(member_path), "cache_reused": bool(cache_reused)})
    if scalers is None:
        raise RuntimeError("failed to load MLP scalers")
    model = models[0] if len(models) == 1 else pairwise.AveragedTransition(models)
    if int(config.linear_blend_grid_steps) > 1:
        ridge_transition = pairwise.build_scaled_ridge_transition(
            splits,
            scalers,
            ridge_alpha=float(config.ridge_alpha),
        )
        blend = pairwise.select_linear_blend_weight(
            model,
            ridge_transition,
            scalers,
            splits,
            names,
            grid_steps=int(config.linear_blend_grid_steps),
        )
        model = pairwise.WeightedAveragedTransition(
            [model, ridge_transition],
            [float(blend["mlp_weight"]), float(blend["ridge_weight"])],
            training_summary={"type": "validation_linear_blend", **blend},
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
    predictions = pairwise.predict_mlp(model, scalers, intervention_features)
    return intervention_features, predictions, names, n_components, member_summaries


def latest_source_block(features: np.ndarray, indices: Sequence[int], *, n_components: int, lag: int) -> np.ndarray:
    cols = [(int(lag) - 1) * int(n_components) + int(index) for index in indices]
    return features[:, cols]


def estimate_full_order2(
    *,
    intervention_features: np.ndarray,
    predictions: np.ndarray,
    names: list[str],
    n_components: int,
    lag: int,
    output_dir: Path,
    targets: Sequence[int] | None,
    max_targets: int | None,
    force: bool,
    jitter: float,
    status_path: Path,
) -> None:
    joint_dir = output_dir / "joint_order2_by_target"
    joint_dir.mkdir(parents=True, exist_ok=True)
    target_list = list(range(n_components)) if targets is None else [int(target) for target in targets]
    if max_targets is not None:
        target_list = target_list[: int(max_targets)]
    for target_pos, target in enumerate(target_list, start=1):
        out_path = joint_dir / f"target_{int(target):02d}.csv"
        if out_path.exists() and not force:
            continue
        start = time.time()
        rows = []
        for i in range(n_components):
            if i == target:
                continue
            for r in range(i + 1, n_components):
                if r == target:
                    continue
                source = latest_source_block(intervention_features, (i, r), n_components=n_components, lag=lag)
                summary = estimate_mutual_information_transport_map(
                    source,
                    predictions[:, [int(target)]],
                    jitter=float(jitter),
                )
                raw = float(summary["mi_hat"])
                rows.append(
                    {
                        "order": 2,
                        "source_a_index": int(i),
                        "source_b_index": int(r),
                        "subset_str": f"{int(i)},{int(r)}",
                        "target_index": int(target),
                        "target": names[int(target)],
                        "ei_joint": raw,
                        "ei_joint_raw": raw,
                        "bias_correction": float(summary.get("bias_correction", 0.0)),
                    }
                )
        frame = pd.DataFrame(rows)
        tmp_path = out_path.with_name(f"{out_path.name}.tmp")
        frame.to_csv(tmp_path, index=False)
        tmp_path.replace(out_path)
        elapsed = time.time() - start
        done_targets = len(list(joint_dir.glob("target_*.csv")))
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            "\n".join(
                [
                    "# 当前状态",
                    f"- 已完成 target 文件: {done_targets}/{n_components}",
                    f"- 最新完成 target: {target} ({names[int(target)]})",
                    f"- 最新 target 耗时: {elapsed:.1f}s",
                    f"- 当前批次进度: {target_pos}/{len(target_list)}",
                    f"- 输出目录: `{output_dir}`",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[full-order2] target {target:02d} done in {elapsed:.1f}s ({done_targets}/{n_components})", flush=True)


def load_pairwise_matrix(path: Path, n_components: int) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(path, index_col=0)
    matrix = frame.to_numpy(dtype=float)
    if matrix.shape != (n_components, n_components):
        raise ValueError(f"pairwise matrix shape {matrix.shape} does not match {n_components}")
    return matrix, [str(name) for name in frame.index.tolist()]


def build_conditioned_matrix(output_dir: Path, pairwise_matrix: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    n = pairwise_matrix.shape[0]
    joint_dir = output_dir / "joint_order2_by_target"
    files = sorted(joint_dir.glob("target_*.csv"))
    if len(files) != n:
        raise RuntimeError(f"need {n} target files for full Scheme A, found {len(files)}")
    cond_sum = np.zeros((n, n), dtype=float)
    cond_count = np.zeros((n, n), dtype=int)
    all_rows = []
    for path in files:
        frame = pd.read_csv(path)
        all_rows.append(frame)
        for row in frame.itertuples(index=False):
            a = int(row.source_a_index)
            b = int(row.source_b_index)
            target = int(row.target_index)
            joint = float(row.ei_joint)
            cond_sum[a, target] += joint - float(pairwise_matrix[b, target])
            cond_count[a, target] += 1
            cond_sum[b, target] += joint - float(pairwise_matrix[a, target])
            cond_count[b, target] += 1
    conditioned = np.divide(cond_sum, cond_count, out=np.zeros_like(cond_sum), where=cond_count > 0)
    np.fill_diagonal(conditioned, 0.0)
    pd.concat(all_rows, ignore_index=True).to_csv(output_dir / "joint_order2_full.csv", index=False)
    pd.DataFrame(conditioned, index=names, columns=names).to_csv(output_dir / "conditioned_ei_matrix.csv")
    pd.DataFrame(cond_count, index=names, columns=names).to_csv(output_dir / "conditioned_background_counts.csv")
    np.save(output_dir / "conditioned_ei_matrix.npy", conditioned)
    return conditioned, cond_count


def signed_path_scores(matrix: np.ndarray, names: list[str], *, max_path_length: int) -> dict[str, object]:
    direct = np.asarray(matrix, dtype=float).copy()
    np.fill_diagonal(direct, 0.0)
    total = np.zeros_like(direct)
    power = direct.copy()
    for _ in range(int(max_path_length)):
        total += power
        power = power @ direct
    n = direct.shape[0]
    denom = max(1, n - 1)
    med_denom = max(1, (n - 1) * (n - 2))
    ace = (total.sum(axis=1) - np.diag(total)) / denom
    acs = (total.sum(axis=0) - np.diag(total)) / denom
    amce = np.zeros(n, dtype=float)
    for mediator in range(n):
        value = 0.0
        for source in range(n):
            if source == mediator:
                continue
            for target in range(n):
                if target == mediator or target == source:
                    continue
                value += direct[source, mediator] * total[mediator, target]
        amce[mediator] = value / med_denom
    gateway = pd.DataFrame(
        {
            "component": names,
            "component_index": np.arange(n),
            "ace": ace,
            "acs": acs,
            "direct_out_strength": direct.sum(axis=1),
            "direct_in_strength": direct.sum(axis=0),
        }
    )
    gateway["out_rank"] = gateway["ace"].rank(ascending=False, method="min").astype(int)
    gateway["in_rank"] = gateway["acs"].rank(ascending=False, method="min").astype(int)
    mediator = pd.DataFrame({"component": names, "component_index": np.arange(n), "amce": amce})
    total_abs_amce = float(np.sum(np.abs(amce)))
    mediator["amce_abs_fraction"] = np.abs(amce) / total_abs_amce if total_abs_amce > 1.0e-12 else 0.0
    radius = float(np.max(np.abs(np.linalg.eigvals(direct)))) if np.any(direct) else 0.0
    return {
        "direct": direct,
        "total": total,
        "gateway": gateway.sort_values("ace", ascending=False).reset_index(drop=True),
        "mediator": mediator.sort_values("amce", ascending=False).reset_index(drop=True),
        "spectral_radius": radius,
        "max_abs_direct_edge": float(np.max(np.abs(direct))) if direct.size else 0.0,
        "negative_edges": int(np.count_nonzero(direct < 0.0)),
        "positive_edges": int(np.count_nonzero(direct > 0.0)),
    }


def component_nodes(component_maps_path: Path, gateway: pd.DataFrame, mediator: pd.DataFrame) -> pd.DataFrame:
    payload = np.load(component_maps_path)
    maps = np.asarray(payload["component_maps"], dtype=float)
    lat = np.asarray(payload["lat"], dtype=float) if "lat" in payload else np.linspace(-90.0, 90.0, maps.shape[0])
    lon = np.asarray(payload["lon"], dtype=float) if "lon" in payload else np.linspace(0.0, 360.0, maps.shape[1], endpoint=False)
    lon = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    lon = lon[order]
    maps = maps[:, order, :]
    merged = gateway.merge(mediator[["component_index", "amce", "amce_abs_fraction"]], on="component_index", how="left")
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
                "ace": float(row.ace),
                "acs": float(row.acs),
                "amce": float(row.amce),
                "amce_abs_fraction": float(row.amce_abs_fraction),
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
        cmap="PRGn",
        norm=norm,
        edgecolors="#222222",
        linewidths=0.28,
        alpha=0.96,
        zorder=4,
    )
    add_labels(ax, label_nodes)


def plot_comparison(
    baseline_nodes: pd.DataFrame,
    scheme_nodes: pd.DataFrame,
    output: Path,
    *,
    save_svg: bool = True,
) -> dict[str, float]:
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
    scheme_label_ids = set(scheme_nodes.nlargest(8, "ace")["local"].astype(int)) | set(scheme_nodes.nlargest(8, "acs")["local"].astype(int)) | set(scheme_nodes.nlargest(8, "amce")["local"].astype(int))
    base_labels = baseline_nodes[baseline_nodes["local"].astype(int).isin(base_label_ids)]
    scheme_labels = scheme_nodes[scheme_nodes["local"].astype(int).isin(scheme_label_ids)]
    base_ace_vmax = max(float(baseline_nodes[["ace", "acs"]].to_numpy().max()), 1.0e-12)
    scheme_ace_vmax = max(float(scheme_nodes[["ace", "acs"]].to_numpy().max()), 1.0e-12)
    base_amce_abs = max(float(np.nanmax(np.abs(baseline_nodes["amce"].to_numpy(dtype=float)))), 1.0e-12)
    scheme_amce_abs = max(float(np.nanmax(np.abs(scheme_nodes["amce"].to_numpy(dtype=float)))), 1.0e-12)
    base_ace_norm = mpl.colors.Normalize(vmin=0.0, vmax=base_ace_vmax)
    scheme_ace_norm = mpl.colors.Normalize(vmin=0.0, vmax=scheme_ace_vmax)
    base_amce_norm = mpl.colors.TwoSlopeNorm(vmin=-base_amce_abs, vcenter=0.0, vmax=base_amce_abs)
    scheme_amce_norm = mpl.colors.TwoSlopeNorm(vmin=-scheme_amce_abs, vcenter=0.0, vmax=scheme_amce_abs)
    draw_ace_acs(axes[0], baseline_nodes, base_ace_norm, base_labels)
    draw_amce(axes[1], baseline_nodes, base_amce_norm, base_labels)
    draw_ace_acs(axes[2], scheme_nodes, scheme_ace_norm, scheme_labels)
    draw_amce(axes[3], scheme_nodes, scheme_amce_norm, scheme_labels)
    labels = ["a", "b", "c", "d"]
    titles = ["Pairwise dense path", "Pairwise dense path", "Full Scheme A", "Full Scheme A"]
    for label, title, ax in zip(labels, titles, axes, strict=True):
        ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=16, fontweight="bold")
        ax.text(0.5, 1.06, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")
    colorbars = [
        (axes[0], base_ace_norm, "OrRd", "Pairwise ACS (inner) and ACE (outer)"),
        (axes[1], base_amce_norm, "PRGn", "Pairwise AMCE"),
        (axes[2], scheme_ace_norm, "OrRd", "Scheme A ACS (inner) and ACE (outer)"),
        (axes[3], scheme_amce_norm, "PRGn", "Scheme A AMCE"),
    ]
    for ax, norm, cmap, label in colorbars:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colormaps[cmap])
        cbar = fig.colorbar(sm, ax=ax, location="bottom", shrink=0.68, pad=0.07, aspect=24)
        cbar.set_label(label)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if save_svg:
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "baseline_ace_acs_vmax": base_ace_vmax,
        "scheme_ace_acs_vmax": scheme_ace_vmax,
        "baseline_amce_abs_vmax": base_amce_abs,
        "scheme_amce_abs_vmax": scheme_amce_abs,
        "baseline_labeled_nodes": len(base_labels),
        "scheme_labeled_nodes": len(scheme_labels),
    }


def finalize_outputs(args: argparse.Namespace, names: list[str] | None = None) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if names is None:
        frame = pd.read_csv(args.pairwise_matrix, index_col=0)
        names = [str(name) for name in frame.index.tolist()]
        pairwise_matrix = frame.to_numpy(dtype=float)
    else:
        pairwise_matrix, _ = load_pairwise_matrix(Path(args.pairwise_matrix), len(names))
    n_components = len(names)
    conditioned, counts = build_conditioned_matrix(output_dir, pairwise_matrix, names)
    pairwise_direct = pairwise_matrix.copy()
    np.fill_diagonal(pairwise_direct, 0.0)
    baseline = signed_path_scores(pairwise_direct, names, max_path_length=int(args.max_path_length))
    scheme = signed_path_scores(conditioned, names, max_path_length=int(args.max_path_length))
    for prefix, result in (("pairwise_dense", baseline), ("scheme_a_full_order2", scheme)):
        pd.DataFrame(result["direct"], index=names, columns=names).to_csv(output_dir / f"{prefix}_direct_matrix.csv")
        pd.DataFrame(result["total"], index=names, columns=names).to_csv(output_dir / f"{prefix}_total_effect_matrix.csv")
        result["gateway"].to_csv(output_dir / f"{prefix}_gateway_scores.csv", index=False)
        result["mediator"].to_csv(output_dir / f"{prefix}_mediator_scores.csv", index=False)
    baseline_nodes = component_nodes(Path(args.component_maps), baseline["gateway"], baseline["mediator"])
    scheme_nodes = component_nodes(Path(args.component_maps), scheme["gateway"], scheme["mediator"])
    baseline_nodes.to_csv(output_dir / "pairwise_dense_map_nodes.csv", index=False)
    scheme_nodes.to_csv(output_dir / "scheme_a_full_order2_map_nodes.csv", index=False)
    fig_dir = Path(args.fig_dir)
    fig_path = fig_dir / "pairwise_dense_vs_scheme_a_full_order2_map.png"
    plot_limits = plot_comparison(baseline_nodes, scheme_nodes, fig_path, save_svg=True)
    if not np.all(counts[~np.eye(n_components, dtype=bool)] == (n_components - 2)):
        raise RuntimeError("conditioned counts are not full for every non-diagonal edge")
    manifest = {
        "scheme": "scheme_a_full_order2",
        "n_components": n_components,
        "max_path_length": int(args.max_path_length),
        "order2_joint_ei_count": int(sum(len(pd.read_csv(path)) for path in sorted((output_dir / "joint_order2_by_target").glob("target_*.csv")))),
        "expected_order2_joint_ei_count": int(n_components * (n_components - 1) * (n_components - 2) / 2),
        "conditioned_entries": int(np.count_nonzero(counts)),
        "expected_conditioned_entries": int(n_components * (n_components - 1)),
        "background_count_per_edge": int(n_components - 2),
        "missing_conditioned_entries": 0,
        "pairwise": {
            "spectral_radius": baseline["spectral_radius"],
            "max_abs_direct_edge": baseline["max_abs_direct_edge"],
            "positive_edges": baseline["positive_edges"],
            "negative_edges": baseline["negative_edges"],
        },
        "scheme_a": {
            "spectral_radius": scheme["spectral_radius"],
            "max_abs_direct_edge": scheme["max_abs_direct_edge"],
            "positive_edges": scheme["positive_edges"],
            "negative_edges": scheme["negative_edges"],
        },
        "figure_png": str(fig_path),
        "figure_svg": str(fig_path.with_suffix(".svg")),
        "figure_pdf": str(fig_path.with_suffix(".pdf")),
        "plot_limits": plot_limits,
        "input_pairwise_matrix": str(args.pairwise_matrix),
        "component_maps": str(args.component_maps),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--component-scores", default=str(DEFAULT_COMPONENT_SCORES))
    parser.add_argument("--pairwise-matrix", default=str(DEFAULT_PAIRWISE_MATRIX))
    parser.add_argument("--pairwise-gateway", default=str(DEFAULT_PAIRWISE_GATEWAY))
    parser.add_argument("--pairwise-mediator", default=str(DEFAULT_PAIRWISE_MEDIATOR))
    parser.add_argument("--component-maps", default=str(DEFAULT_COMPONENT_MAPS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fig-dir", default=str(DEFAULT_FIG_DIR))
    parser.add_argument("--lag", type=int, default=4)
    parser.add_argument("--intervention-samples", type=int, default=4096)
    parser.add_argument("--tm-jitter", type=float, default=1.0e-6)
    parser.add_argument("--max-path-length", type=int, default=60)
    parser.add_argument("--targets", default="")
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--force-targets", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--status-path", default="docs/log/runge_slp_full_scheme_a_status.md")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if not args.finalize_only:
        config = load_peid_config(args)
        intervention_features, predictions, names, n_components, member_summaries = prepare_intervention_predictions(config)
        prep_manifest = {
            "n_components": n_components,
            "intervention_samples": int(args.intervention_samples),
            "member_summaries": member_summaries,
        }
        (Path(args.output_dir) / "preparation_manifest.json").write_text(json.dumps(prep_manifest, indent=2), encoding="utf-8")
        estimate_full_order2(
            intervention_features=intervention_features,
            predictions=predictions,
            names=names,
            n_components=n_components,
            lag=int(args.lag),
            output_dir=Path(args.output_dir),
            targets=parse_int_list(args.targets),
            max_targets=args.max_targets,
            force=bool(args.force_targets),
            jitter=float(args.tm_jitter),
            status_path=Path(args.status_path),
        )
    else:
        names = None
    if args.estimate_only:
        print(json.dumps({"output_dir": str(args.output_dir), "estimate_only": True}, indent=2))
        return 0
    manifest = finalize_outputs(args, names=names)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
