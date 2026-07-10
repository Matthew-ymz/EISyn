#!/usr/bin/env python3
"""Exhaustive second-order PEID synergy scan for the selected Runge Transformer."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import digamma


DEFAULT_SELECTOR = Path("results/runge_transformer_forecast_sweep/horizon_selector_selection.csv")
DEFAULT_COMPONENT_MAPS = Path("results/runge_slp_daily_1948_2026_20260628/results/runge/2015_gateways/component_maps.npz")
DEFAULT_LINEAR_EDGES = Path("results/runge/2015_gateways/causal_edges.csv")
RESULT_SUBDIR = Path("results/runge/transformer_full_pair_synergy")
FIG_SUBDIR = Path("fig/runge/transformer_full_pair_synergy")
PAPER_LABEL_MAP = {7: 18, 18: 7, 8: 26, 26: 8, 21: 48, 48: 21}
KNOWN_COMPONENTS = {
    0: "印度尼西亚与东印度洋上空的 ENSO/Walker 环流西侧上升支",
    1: "东太平洋 ENSO 区域，正常态对应 Walker 环流下沉支",
    2: "热带大西洋强对流与上升运动区域",
    5: "北大西洋涛动（NAO）偶极模态",
    18: "西太平洋强对流与 Walker 环流相关区域",
    26: "热带大西洋及西非季风相关区域",
    33: "阿拉伯海高压及印度季风相关区域",
    53: "跨南大西洋与喜马拉雅区域的非局地模态",
    59: "Runge Fig.3 中影响 ENSO 与阿拉伯海关系的共同驱动诊断分量",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def _gaussian_logdet_bias_correction(dimension: int, sample_size: int) -> float:
    if int(sample_size) <= int(dimension):
        return 0.0
    nu = int(sample_size) - 1
    return float(
        sum(digamma((nu + 1 - index) / 2.0) for index in range(1, int(dimension) + 1))
        + int(dimension) * np.log(2.0)
        - int(dimension) * np.log(nu)
    )


def _stable_logdet(matrix: np.ndarray, *, jitter: float = 1.0e-6) -> float:
    values = np.asarray(matrix, dtype=float)
    sign, logdet = np.linalg.slogdet(values + float(jitter) * np.eye(values.shape[0], dtype=float))
    if sign <= 0:
        raise np.linalg.LinAlgError("Covariance submatrix is not positive definite.")
    return float(logdet)


def _mean_fitted_gaussian_log_prob(covariance: np.ndarray, *, sample_size: int, jitter: float) -> float:
    values = np.asarray(covariance, dtype=float)
    fitted = values + float(jitter) * np.eye(values.shape[0], dtype=float)
    mean_quadratic = ((int(sample_size) - 1) / int(sample_size)) * float(np.trace(np.linalg.solve(fitted, values)))
    return -0.5 * (
        values.shape[0] * np.log(2.0 * np.pi)
        + _stable_logdet(values, jitter=jitter)
        + mean_quadratic
    )


def analytic_affine_tm_mi(
    covariance: np.ndarray,
    *,
    source_indices: Sequence[int],
    target_index: int,
    sample_size: int,
    jitter: float = 1.0e-6,
) -> float:
    """Match the repository affine transport-map MI estimator from covariance."""

    source = tuple(int(index) for index in source_indices)
    target = int(target_index)
    joint = source + (target,)
    source_cov = np.asarray(covariance)[np.ix_(source, source)]
    target_cov = np.asarray(covariance)[np.ix_((target,), (target,))]
    joint_cov = np.asarray(covariance)[np.ix_(joint, joint)]
    raw_mi = (
        _mean_fitted_gaussian_log_prob(joint_cov, sample_size=sample_size, jitter=jitter)
        - _mean_fitted_gaussian_log_prob(source_cov, sample_size=sample_size, jitter=jitter)
        - _mean_fitted_gaussian_log_prob(target_cov, sample_size=sample_size, jitter=jitter)
    )
    bias = 0.5 * (
        _gaussian_logdet_bias_correction(len(source), int(sample_size))
        + _gaussian_logdet_bias_correction(1, int(sample_size))
        - _gaussian_logdet_bias_correction(len(joint), int(sample_size))
    )
    return float(raw_mi - bias)


def enumerate_all_pair_targets(n_components: int) -> np.ndarray:
    """Return every distinct source pair crossed with every target."""

    n = int(n_components)
    rows = [
        (int(source_a), int(source_b), int(target))
        for source_a, source_b in itertools.combinations(range(n), 2)
        for target in range(n)
    ]
    return np.asarray(rows, dtype=np.int16 if n < np.iinfo(np.int16).max else np.int32)


def aggregate_seed_tables(seed_tables: Sequence[pd.DataFrame], *, seeds: Sequence[int]) -> pd.DataFrame:
    """Aggregate exhaustive tables while preserving per-seed ranking stability."""

    if len(seed_tables) != len(seeds) or not seed_tables:
        raise ValueError("seed_tables and seeds must be non-empty and have matching lengths.")
    keys = ["source_a", "source_b", "target"]
    value_columns = ["joint_ei", "source_a_ei", "source_b_ei", "delta2", "delta2_over_joint"]
    frames: list[pd.DataFrame] = []
    for frame, seed in zip(seed_tables, seeds):
        seeded = frame[keys + value_columns].copy()
        seeded["seed"] = int(seed)
        seeded["rank"] = seeded["delta2"].rank(method="min", ascending=False)
        frames.append(seeded)
    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby(keys, sort=False)
    result = grouped.agg(
        mean_joint_ei=("joint_ei", "mean"),
        mean_source_a_ei=("source_a_ei", "mean"),
        mean_source_b_ei=("source_b_ei", "mean"),
        mean_delta2=("delta2", "mean"),
        std_delta2=("delta2", "std"),
        mean_delta2_over_joint=("delta2_over_joint", "mean"),
        median_rank=("rank", "median"),
    ).reset_index()
    positive_rate = grouped["delta2"].apply(lambda values: float(np.mean(np.asarray(values) > 0.0))).reset_index(name="positive_rate")
    result = result.merge(positive_rate, on=keys, how="left")
    result["std_delta2"] = result["std_delta2"].fillna(0.0)
    return result.sort_values(["mean_delta2", "positive_rate"], ascending=[False, False], kind="mergesort").reset_index(drop=True)


def resolve_h1_transformer_candidate(selector_path: str | Path = DEFAULT_SELECTOR) -> dict[str, object]:
    """Resolve the validation-selected h=1 Transformer candidate and checkpoint."""

    selector = Path(selector_path).expanduser().resolve()
    selection = pd.read_csv(selector)
    rows = selection[selection["horizon"].astype(int) == 1]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one h=1 selector row, found {len(rows)}.")
    candidate = str(rows.iloc[0]["candidate"])
    manifest = selector.parent / "candidates" / candidate / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    checkpoint = Path(str(payload["rnn_cache"])).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (manifest.parent / checkpoint).resolve()
    return {
        "candidate": candidate,
        "selector": selector,
        "manifest": manifest.resolve(),
        "checkpoint": checkpoint.resolve(),
        "manifest_payload": payload,
        "selection": rows.iloc[0].to_dict(),
    }


def paper_label(local_index: int) -> int:
    return int(PAPER_LABEL_MAP.get(int(local_index), int(local_index)))


def exhaustive_pair_synergy_table(
    sources: np.ndarray,
    predictions: np.ndarray,
    *,
    jitter: float = 1.0e-6,
) -> pd.DataFrame:
    """Compute every two-source-to-one-target affine-TM PEID interaction."""

    source_values = np.asarray(sources, dtype=float)
    target_values = np.asarray(predictions, dtype=float)
    if source_values.ndim != 2 or target_values.ndim != 2 or source_values.shape != target_values.shape:
        raise ValueError("sources and predictions must be matching 2D arrays.")
    n_samples, n_components = source_values.shape
    covariance = np.cov(np.column_stack([source_values, target_values]), rowvar=False, bias=False)
    single = np.empty((n_components, n_components), dtype=float)
    for source in range(n_components):
        for target in range(n_components):
            single[source, target] = max(
                0.0,
                analytic_affine_tm_mi(
                    covariance,
                    source_indices=(source,),
                    target_index=n_components + target,
                    sample_size=n_samples,
                    jitter=jitter,
                ),
            )
    triplets = enumerate_all_pair_targets(n_components)
    joint = np.empty(len(triplets), dtype=float)
    for index, (source_a, source_b, target) in enumerate(triplets):
        joint[index] = max(
            0.0,
            analytic_affine_tm_mi(
                covariance,
                source_indices=(int(source_a), int(source_b)),
                target_index=n_components + int(target),
                sample_size=n_samples,
                jitter=jitter,
            ),
        )
    source_a_ei = single[triplets[:, 0], triplets[:, 2]]
    source_b_ei = single[triplets[:, 1], triplets[:, 2]]
    delta2 = joint - source_a_ei - source_b_ei
    ratio = np.divide(delta2, joint, out=np.zeros_like(delta2), where=joint > 1.0e-12)
    return pd.DataFrame(
        {
            "source_a": triplets[:, 0].astype(int),
            "source_b": triplets[:, 1].astype(int),
            "target": triplets[:, 2].astype(int),
            "joint_ei": joint,
            "source_a_ei": source_a_ei,
            "source_b_ei": source_b_ei,
            "delta2": delta2,
            "delta2_over_joint": ratio,
        }
    )


def _relation_delta2(
    sources: np.ndarray,
    predictions: np.ndarray,
    source_a: int,
    source_b: int,
    target: int,
    *,
    jitter: float,
) -> float:
    values = np.column_stack([sources[:, source_a], sources[:, source_b], predictions[:, target]])
    covariance = np.cov(values, rowvar=False, bias=False)
    sample_size = len(values)
    left = max(0.0, analytic_affine_tm_mi(covariance, source_indices=(0,), target_index=2, sample_size=sample_size, jitter=jitter))
    right = max(0.0, analytic_affine_tm_mi(covariance, source_indices=(1,), target_index=2, sample_size=sample_size, jitter=jitter))
    joint = max(0.0, analytic_affine_tm_mi(covariance, source_indices=(0, 1), target_index=2, sample_size=sample_size, jitter=jitter))
    return float(joint - left - right)


def permutation_review(
    ranked: pd.DataFrame,
    batches: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    top_k: int,
    reps: int,
    seed: int = 9701,
    jitter: float = 1.0e-6,
) -> pd.DataFrame:
    """Attach a descriptive IID-permutation null to the strongest relations."""

    reviewed = ranked.head(int(top_k)).copy()
    rng = np.random.default_rng(int(seed))
    null_means = np.empty((len(reviewed), int(reps)), dtype=float)
    for row_index, row in enumerate(reviewed.itertuples(index=False)):
        for rep in range(int(reps)):
            values = []
            for sources, predictions in batches:
                permutation = rng.permutation(len(predictions))
                values.append(
                    _relation_delta2(
                        sources,
                        predictions[permutation],
                        int(row.source_a),
                        int(row.source_b),
                        int(row.target),
                        jitter=jitter,
                    )
                )
            null_means[row_index, rep] = float(np.mean(values))
    reviewed["permutation_null_mean"] = null_means.mean(axis=1)
    reviewed["permutation_null_std"] = null_means.std(axis=1, ddof=1) if int(reps) > 1 else 0.0
    reviewed["permutation_z"] = np.divide(
        reviewed["mean_delta2"] - reviewed["permutation_null_mean"],
        reviewed["permutation_null_std"],
        out=np.full(len(reviewed), np.nan),
        where=reviewed["permutation_null_std"].to_numpy() > 1.0e-12,
    )
    reviewed["permutation_p_empirical"] = [
        float((1 + np.sum(null_means[index] >= float(row.mean_delta2))) / (int(reps) + 1))
        for index, row in enumerate(reviewed.itertuples(index=False))
    ]
    return reviewed


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    clean = frame.replace({np.nan: None, np.inf: None, -np.inf: None})
    return clean.to_dict("records")


def _decorate_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("source_a", "source_b", "target"):
        result[f"{column}_paper"] = result[column].map(paper_label)
    result["relation"] = result.apply(
        lambda row: f"{{No.{int(row.source_a_paper)}, No.{int(row.source_b_paper)}}} -> No.{int(row.target_paper)}",
        axis=1,
    )
    return result


def _pair_summary(ranked: pd.DataFrame) -> pd.DataFrame:
    result = (
        ranked.groupby(["source_a", "source_b"], as_index=False)
        .agg(
            total_positive_delta2=("mean_delta2", lambda values: float(np.maximum(np.asarray(values), 0.0).sum())),
            max_delta2=("mean_delta2", "max"),
            mean_delta2=("mean_delta2", "mean"),
            positive_target_count=("mean_delta2", lambda values: int(np.sum(np.asarray(values) > 0.0))),
        )
        .sort_values("total_positive_delta2", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    result["source_a_paper"] = result["source_a"].map(paper_label)
    result["source_b_paper"] = result["source_b"].map(paper_label)
    return result


def _component_peak(component_map: np.ndarray) -> tuple[float, float]:
    lat_count, lon_count = component_map.shape
    row, col = np.unravel_index(int(np.nanargmax(np.abs(component_map))), component_map.shape)
    latitudes = np.linspace(90.0, -90.0, lat_count)
    longitudes = np.linspace(0.0, 360.0, lon_count, endpoint=False)
    longitude = float(longitudes[col])
    if longitude > 180.0:
        longitude -= 360.0
    return float(latitudes[row]), longitude


def _linear_support(source_a: int, source_b: int, target: int, linear_edges: pd.DataFrame | None) -> str:
    if linear_edges is None or linear_edges.empty:
        return "未加载线性 Runge 路径支持。"
    rows = linear_edges[
        linear_edges["source"].astype(int).isin([int(source_a), int(source_b)])
        & (linear_edges["target"].astype(int) == int(target))
    ].copy()
    if rows.empty:
        return "旧 Runge 线性因果图中未发现两个源到该目标的直接边；该关系更可能反映非线性或间接模型内结构。"
    descriptions = [
        f"No.{paper_label(int(row.source))}->No.{paper_label(int(row.target))} (lag={int(row.lag)}周, coef={float(row.coefficient):.3g})"
        for row in rows.sort_values(["source", "lag"]).itertuples(index=False)
    ]
    return "旧 Runge 线性因果图提供直接边支持：" + "；".join(descriptions) + "。"


def _physical_note(
    source_a: int,
    source_b: int,
    target: int,
    component_maps: np.ndarray | None,
    linear_edges: pd.DataFrame | None = None,
) -> tuple[str, str]:
    labels = [paper_label(source_a), paper_label(source_b), paper_label(target)]
    known = list(dict.fromkeys(KNOWN_COMPONENTS[label] for label in labels if label in KNOWN_COMPONENTS))
    if len(known) >= 2:
        confidence = "中"
        note = "该关系连接多个已有气候学标注模态：" + "；".join(known) + "。可作为跨区域协同传播或共同调制的候选机制。"
    elif len(known) == 1:
        confidence = "低-中"
        note = "其中一个模态具有已有气候学解释：" + known[0] + "；其余模态需要结合空间载荷和季节分层进一步验证。"
    else:
        confidence = "低"
        note = "当前没有仓库内已校准的气候过程标签，物理含义主要来自空间载荷位置，属于待验证假说。"
    if component_maps is not None:
        peaks = [_component_peak(component_maps[..., index]) for index in (source_a, source_b, target)]
        note += " 三个模态绝对载荷峰值约位于 " + "、".join(f"({lat:.1f}°, {lon:.1f}°)" for lat, lon in peaks) + "。"
    note += " " + _linear_support(source_a, source_b, target, linear_edges)
    return confidence, note


def _save_ranking_figure(top: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    plot = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 6.0), constrained_layout=True)
    xerr = plot["std_delta2"].fillna(0.0).to_numpy()
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(plot)))
    ax.barh(np.arange(len(plot)), plot["mean_delta2"], xerr=xerr, color=colors, alpha=0.9, ecolor="#4b5563", capsize=2)
    ax.set_yticks(np.arange(len(plot)))
    ax.set_yticklabels(plot["relation"])
    ax.set_xlabel("Mean positive $\\Delta_2$ across intervention seeds (nats)")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.6)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def _save_hyperedge_figure(top: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    nodes = sorted(set(top["source_a_paper"]) | set(top["source_b_paper"]) | set(top["target_paper"]))
    angles = np.linspace(0.0, 2.0 * np.pi, len(nodes), endpoint=False)
    positions = {node: np.array([np.cos(angle), np.sin(angle)]) for node, angle in zip(nodes, angles)}
    fig, ax = plt.subplots(figsize=(8.0, 7.0), constrained_layout=True)
    ax.axis("off")
    maximum = max(float(top["mean_delta2"].max()), 1.0e-12)
    for row in top.itertuples(index=False):
        a = positions[int(row.source_a_paper)]
        b = positions[int(row.source_b_paper)]
        target = positions[int(row.target_paper)]
        midpoint = (a + b) / 2.0
        width = 0.6 + 4.0 * float(row.mean_delta2) / maximum
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#7c3aed", linewidth=width, alpha=0.25)
        ax.plot([midpoint[0], target[0]], [midpoint[1], target[1]], color="#0f766e", linewidth=width, alpha=0.55)
        ax.scatter(target[0], target[1], marker=">", s=12.0 + 5.0 * width, color="#0f766e", alpha=0.75, zorder=3)
    for node, position in positions.items():
        ax.scatter(position[0], position[1], s=180, color="#f8fafc", edgecolor="#334155", linewidth=0.8, zorder=4)
        ax.text(position[0], position[1], f"No.{node}", ha="center", va="center", fontsize=6.5, zorder=5)
    ax.plot([], [], color="#7c3aed", linewidth=3, alpha=0.35, label="Source pair")
    ax.plot([], [], color="#0f766e", linewidth=3, alpha=0.7, label="Joint influence to target")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def _save_spatial_figures(top: pd.DataFrame, component_maps: np.ndarray, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for page_start in range(0, len(top), 5):
        page = top.iloc[page_start : page_start + 5]
        fig, axes = plt.subplots(len(page), 3, figsize=(10.0, 2.0 * len(page)), constrained_layout=True, squeeze=False)
        indices = np.concatenate([page["source_a"].to_numpy(), page["source_b"].to_numpy(), page["target"].to_numpy()])
        vlim = max(float(np.nanpercentile(np.abs(component_maps[..., indices]), 99.0)), 1.0e-12)
        last_image = None
        for row_index, row in enumerate(page.itertuples(index=False)):
            for col_index, (kind, local) in enumerate((("Source A", row.source_a), ("Source B", row.source_b), ("Target", row.target))):
                ax = axes[row_index, col_index]
                last_image = ax.imshow(component_maps[..., int(local)], cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
                ax.set_title(f"#{page_start + row_index + 1} {kind}: No.{paper_label(int(local))}", fontsize=7.5)
                ax.set_xticks([])
                ax.set_yticks([])
        if last_image is not None:
            fig.colorbar(last_image, ax=axes.ravel().tolist(), location="right", shrink=0.72, label="Rotated loading")
        path = output_dir / f"top20_spatial_modes_{page_start + 1:02d}_{page_start + len(page):02d}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _write_report(
    path: Path,
    top: pd.DataFrame,
    cross_target_top: pd.DataFrame,
    pair_summary: pd.DataFrame,
    *,
    n_relations: int,
    seeds: Sequence[int],
    sample_size: int,
    component_maps: np.ndarray | None,
    linear_edges: pd.DataFrame | None,
    model_context: dict[str, object] | None,
) -> Path:
    lines = [
        "# Runge Transformer 全量二阶协同分析",
        "",
        "## 结论摘要",
        "",
        f"- 基于当前 validation-selected h=1 Transformer，对最新周 60 个分量完整扫描了 `{n_relations:,}` 个二源到单目标关系。",
        f"- 主排序使用 `{len(seeds)}` 个独立最大熵干预批次（每批 `{sample_size}` 个样本）的平均二阶协同 `Delta2`。",
        "- 正值表示训练模型中联合源对提供了超出两个单源 EI 之和的非加性信息；负值仅作连续估计与反协同诊断。",
        "- 全量主榜允许目标同时出现在源对中，因而会包含强自调制协同；另给出排除该情况的纯跨目标榜，用于跨区域物理解读。",
        "- 置换 p 值是对 Top 候选的描述性 IID 空分布复核，不是全局多重检验结论。",
        "",
        "## Top 20 正协同关系",
        "",
        "| rank | relation | mean Delta2 | std | positive rate | median rank | permutation p | physical confidence |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        confidence, _ = _physical_note(int(row.source_a), int(row.source_b), int(row.target), component_maps, linear_edges)
        lines.append(
            f"| {rank} | {row.relation} | {float(row.mean_delta2):.6g} | {float(row.std_delta2):.3g} | "
            f"{float(row.positive_rate):.2f} | {float(row.median_rank):.1f} | "
            f"{float(getattr(row, 'permutation_p_empirical', float('nan'))):.4g} | {confidence} |"
        )
    lines.extend(["", "## 逐条物理解读", ""])
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        confidence, note = _physical_note(int(row.source_a), int(row.source_b), int(row.target), component_maps, linear_edges)
        lines.extend(
            [
                f"### {rank}. {row.relation}",
                "",
                f"- 模型内协同：平均 `Delta2={float(row.mean_delta2):.6g}`，五批正值率 `{float(row.positive_rate):.2f}`，物理解释可信度 `{confidence}`。",
                f"- 解释：{note}",
                "- 限制：该读数描述固定 Transformer 在独立最大熵干预下的响应；未观测海温、季节性和其他共同驱动可能改变真实气候因果解释。",
                "",
            ]
        )
    lines.extend(
        [
            "## Top 20 纯跨目标协同关系",
            "",
            "该榜排除 `target in {source_a, source_b}`，更适合提出跨区域传播或共同调制的物理假说。",
            "",
            "| rank | relation | mean Delta2 | std | positive rate | permutation p | physical confidence |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for rank, row in enumerate(cross_target_top.itertuples(index=False), start=1):
        confidence, _ = _physical_note(int(row.source_a), int(row.source_b), int(row.target), component_maps, linear_edges)
        lines.append(
            f"| {rank} | {row.relation} | {float(row.mean_delta2):.6g} | {float(row.std_delta2):.3g} | "
            f"{float(row.positive_rate):.2f} | {float(getattr(row, 'permutation_p_empirical', float('nan'))):.4g} | {confidence} |"
        )
    lines.extend(["", "### 跨目标榜物理解读", ""])
    for rank, row in enumerate(cross_target_top.itertuples(index=False), start=1):
        confidence, note = _physical_note(int(row.source_a), int(row.source_b), int(row.target), component_maps, linear_edges)
        lines.extend([f"- **{rank}. {row.relation}（{confidence}）**：{note}", ""])
    lines.extend(
        [
            "## 最强全局源对",
            "",
            "| rank | source pair | summed positive Delta2 | strongest target Delta2 | positive targets |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for rank, row in enumerate(pair_summary.head(20).itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | No.{int(row.source_a_paper)} + No.{int(row.source_b_paper)} | "
            f"{float(row.total_positive_delta2):.6g} | {float(row.max_delta2):.6g} | {int(row.positive_target_count)} |"
        )
    lines.extend(
        [
            "",
            "## 方法与解释边界",
            "",
            "- 计算使用与仓库 affine transport-map estimator 数值一致的解析协方差形式，避免对 106,200 个关系重复拟合密度模型。",
            "- 源变量只取最新周 60 个分量；更早一周仍作为 Transformer 预测上下文，但不作为本次二阶源节点。",
            "- PEID 连续变量二阶项不保证非负，因此主榜只解释稳定正值，负值单独保存为诊断。",
            "- 物理解释依据 Varimax 空间载荷、Runge 原文已标注过程及当前模型读出，应视为机制假说而非真实系统因果定论。",
        ]
    )
    if model_context:
        lines.extend(["", "## 模型来源", "", f"- Candidate: `{model_context.get('candidate', 'unknown')}`", f"- Checkpoint: `{model_context.get('checkpoint', 'unknown')}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_analysis_from_batches(
    batches: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    seeds: Sequence[int],
    output_dir: str | Path,
    component_maps: np.ndarray | None = None,
    linear_edges: pd.DataFrame | None = None,
    top_k: int = 20,
    permutation_top_k: int = 200,
    permutation_reps: int = 200,
    jitter: float = 1.0e-6,
    model_context: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run exhaustive scoring and write reusable analysis artifacts."""

    if len(batches) != len(seeds) or not batches:
        raise ValueError("batches and seeds must be non-empty and have matching lengths.")
    root = Path(output_dir).expanduser().resolve()
    result_dir = root / RESULT_SUBDIR
    fig_dir = root / FIG_SUBDIR
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    seed_tables = [exhaustive_pair_synergy_table(sources, predictions, jitter=jitter) for sources, predictions in batches]
    ranked = aggregate_seed_tables(seed_tables, seeds=seeds)
    ranked = _decorate_labels(ranked)
    positive = ranked[ranked["mean_delta2"] > 0.0].copy().reset_index(drop=True)
    reviewed = permutation_review(
        positive,
        batches,
        top_k=min(int(permutation_top_k), len(positive)),
        reps=int(permutation_reps),
        jitter=jitter,
    )
    positive = positive.merge(
        reviewed[["source_a", "source_b", "target", "permutation_null_mean", "permutation_null_std", "permutation_z", "permutation_p_empirical"]],
        on=["source_a", "source_b", "target"],
        how="left",
    )
    top = positive.head(int(top_k)).copy()
    cross_target = positive[
        (positive["target"] != positive["source_a"]) & (positive["target"] != positive["source_b"])
    ].head(int(top_k)).copy()
    negative = ranked.sort_values("mean_delta2", ascending=True, kind="mergesort").head(max(int(top_k), 20)).copy()
    pair_summary = _pair_summary(ranked)

    full_cache = result_dir / "full_pair_synergy_ranked.npz"
    np.savez_compressed(full_cache, **{column: ranked[column].to_numpy() for column in ranked.columns if ranked[column].dtype != object})
    top_path = result_dir / "top_positive_relations.json"
    top_path.write_text(json.dumps(_records(positive.head(max(int(permutation_top_k), int(top_k)))), ensure_ascii=False, indent=2), encoding="utf-8")
    cross_target_path = result_dir / "top_cross_target_relations.json"
    cross_target_path.write_text(json.dumps(_records(cross_target), ensure_ascii=False, indent=2), encoding="utf-8")
    negative_path = result_dir / "negative_relations.json"
    negative_path.write_text(json.dumps(_records(negative), ensure_ascii=False, indent=2), encoding="utf-8")
    pair_path = result_dir / "pair_summary.json"
    pair_path.write_text(json.dumps(_records(pair_summary), ensure_ascii=False, indent=2), encoding="utf-8")
    ranking_figure = _save_ranking_figure(top, fig_dir / "top20_positive_synergy_ranking.png")
    cross_target_ranking_figure = _save_ranking_figure(cross_target, fig_dir / "top20_cross_target_synergy_ranking.png")
    hyperedge_figure = _save_hyperedge_figure(cross_target.head(8), fig_dir / "top20_synergy_hypergraph.png")
    spatial_figures = _save_spatial_figures(top, component_maps, fig_dir) if component_maps is not None else []
    report = _write_report(
        result_dir / "report_zh.md",
        top,
        cross_target,
        pair_summary,
        n_relations=len(ranked),
        seeds=seeds,
        sample_size=len(batches[0][0]),
        component_maps=component_maps,
        linear_edges=linear_edges,
        model_context=model_context,
    )
    manifest_payload = {
        "n_components": int(batches[0][0].shape[1]),
        "n_relations": int(len(ranked)),
        "seeds": [int(seed) for seed in seeds],
        "intervention_samples": int(len(batches[0][0])),
        "top_k": int(top_k),
        "permutation_top_k": int(min(permutation_top_k, len(positive))),
        "permutation_reps": int(permutation_reps),
        "estimator": "analytic affine transport-map MI from shared covariance",
        "model_context": model_context or {},
        "outputs": {
            "full_cache": str(full_cache),
            "top_relations": str(top_path),
            "top_cross_target_relations": str(cross_target_path),
            "negative_relations": str(negative_path),
            "pair_summary": str(pair_path),
            "report": str(report),
            "ranking_figure": str(ranking_figure),
            "cross_target_ranking_figure": str(cross_target_ranking_figure),
            "hyperedge_figure": str(hyperedge_figure),
            "spatial_figures": [str(path) for path in spatial_figures],
        },
    }
    manifest = result_dir / "manifest.json"
    manifest.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "full_cache": full_cache,
        "top_relations": top_path,
        "top_cross_target_relations": cross_target_path,
        "negative_relations": negative_path,
        "pair_summary": pair_path,
        "manifest": manifest,
        "report": report,
        "ranking_figure": ranking_figure,
        "cross_target_ranking_figure": cross_target_ranking_figure,
        "hyperedge_figure": hyperedge_figure,
        "spatial_figures": spatial_figures,
    }


def load_selected_transformer_batches(
    selector_path: str | Path = DEFAULT_SELECTOR,
    *,
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    intervention_samples: int = 4096,
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
    device: str = "cpu",
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    """Load the selected h=1 Transformer and generate shared intervention batches."""

    import torch

    try:
        from scripts import run_runge_pairwise_mlp_ei as pairwise
        from scripts import run_runge_rnn_forecast_comparison as forecast
    except ImportError:
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        from scripts import run_runge_pairwise_mlp_ei as pairwise
        from scripts import run_runge_rnn_forecast_comparison as forecast

    resolved = resolve_h1_transformer_candidate(selector_path)
    manifest_payload = dict(resolved["manifest_payload"])
    raw_config = dict(manifest_payload["config"])
    config_fields = {field.name for field in fields(forecast.RungeRnnForecastConfig)}
    kwargs = {key: value for key, value in raw_config.items() if key in config_fields}
    for key in ("component_scores", "output_dir", "result_subdir", "fig_subdir"):
        if key in kwargs:
            kwargs[key] = Path(kwargs[key])
    config = forecast.RungeRnnForecastConfig(**kwargs)
    config = replace(config, device=str(device))
    component_scores = pairwise._resolve_path(manifest_payload["component_scores"])
    frame = pairwise.load_component_scores(component_scores)
    names = list(frame.columns)
    horizons = forecast.parse_int_tuple(config.horizons)
    features, targets = forecast.build_multistep_lagged_dataset(frame, lag=int(config.lag), horizons=horizons)
    splits = forecast.split_multistep_arrays(
        features,
        targets,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
    )
    checkpoint_payload = torch.load(resolved["checkpoint"], map_location="cpu", weights_only=False)
    output_dim = int(len(names)) if str(config.rnn_objective) == "rollout_multistep" else int(len(names) * len(horizons))
    model_config = replace(
        config,
        hidden_dim=int(checkpoint_payload["hidden_dim"]),
        num_layers=int(checkpoint_payload["num_layers"]),
        dropout=float(checkpoint_payload["dropout"]),
        rnn_type=str(checkpoint_payload["rnn_type"]),
        use_linear_skip=bool(checkpoint_payload["use_linear_skip"]),
        transformer_nhead=int(checkpoint_payload.get("transformer_nhead", config.transformer_nhead)),
        transformer_dim_feedforward=int(checkpoint_payload.get("transformer_dim_feedforward", config.transformer_dim_feedforward)),
        transformer_pooling=str(checkpoint_payload.get("transformer_pooling", config.transformer_pooling)),
        transformer_positional_encoding=str(checkpoint_payload.get("transformer_positional_encoding", config.transformer_positional_encoding)),
    )
    model = forecast.build_transition_model(config=model_config, input_size=len(names), output_dim=output_dim)
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.residual_scale = float(checkpoint_payload.get("residual_scale", 1.0))
    model.to(forecast.resolve_torch_device(str(device)))
    scalers = {
        "x_mean": np.asarray(checkpoint_payload["x_mean"], dtype=np.float32),
        "x_std": np.asarray(checkpoint_payload["x_std"], dtype=np.float32),
        "y_mean": np.asarray(checkpoint_payload["y_mean"], dtype=np.float32),
        "y_std": np.asarray(checkpoint_payload["y_std"], dtype=np.float32),
    }
    batches: list[tuple[np.ndarray, np.ndarray]] = []
    for seed in seeds:
        intervention_features = pairwise.sample_max_entropy_features(
            splits["train"][0],
            n_components=len(names),
            lag=int(config.lag),
            samples=int(intervention_samples),
            low_q=float(quantile_low),
            high_q=float(quantile_high),
            seed=int(seed),
        )
        raw_predictions = forecast.predict_torch_model(model, scalers, intervention_features)
        if str(config.rnn_objective) == "rollout_multistep":
            h1_predictions = raw_predictions[:, : len(names)]
        else:
            h1_position = list(horizons).index(1)
            h1_predictions = raw_predictions[:, h1_position * len(names) : (h1_position + 1) * len(names)]
        latest_sources = intervention_features[:, -len(names) :]
        batches.append((np.asarray(latest_sources, dtype=float), np.asarray(h1_predictions, dtype=float)))
    context = {
        "candidate": resolved["candidate"],
        "selector": str(resolved["selector"]),
        "manifest": str(resolved["manifest"]),
        "checkpoint": str(resolved["checkpoint"]),
        "component_scores": str(component_scores),
        "lag": int(config.lag),
        "horizons": [int(value) for value in horizons],
        "objective": str(config.rnn_objective),
        "n_components": len(names),
        "device": str(device),
        "quantile_low": float(quantile_low),
        "quantile_high": float(quantile_high),
    }
    return batches, context


def parse_int_tuple(text: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(text, str):
        values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    else:
        values = tuple(int(value) for value in text)
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--component-maps", type=Path, default=DEFAULT_COMPONENT_MAPS)
    parser.add_argument("--linear-edges", type=Path, default=DEFAULT_LINEAR_EDGES)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--intervention-samples", type=int, default=4096)
    parser.add_argument("--quantile-low", type=float, default=0.05)
    parser.add_argument("--quantile-high", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--permutation-top-k", type=int, default=200)
    parser.add_argument("--permutation-reps", type=int, default=200)
    parser.add_argument("--tm-jitter", type=float, default=1.0e-6)
    parser.add_argument("--device", choices=["cpu", "auto", "mps", "cuda"], default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = parse_int_tuple(args.seeds)
    start = time.time()
    batches, context = load_selected_transformer_batches(
        args.selector,
        seeds=seeds,
        intervention_samples=int(args.intervention_samples),
        quantile_low=float(args.quantile_low),
        quantile_high=float(args.quantile_high),
        device=str(args.device),
    )
    maps_path = Path(args.component_maps).expanduser()
    component_maps = np.load(maps_path)["component_maps"] if maps_path.exists() else None
    linear_edges_path = Path(args.linear_edges).expanduser()
    linear_edges = pd.read_csv(linear_edges_path) if linear_edges_path.exists() else None
    artifacts = run_analysis_from_batches(
        batches,
        seeds=seeds,
        output_dir=args.output_dir,
        component_maps=component_maps,
        linear_edges=linear_edges,
        top_k=int(args.top_k),
        permutation_top_k=int(args.permutation_top_k),
        permutation_reps=int(args.permutation_reps),
        jitter=float(args.tm_jitter),
        model_context=context,
    )
    print(json.dumps({key: [str(item) for item in value] if isinstance(value, list) else str(value) for key, value in artifacts.items()}, indent=2))
    print(f"[runge-full-synergy] completed in {time.time() - start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
