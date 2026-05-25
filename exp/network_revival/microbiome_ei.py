from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from .microbiome import (
    MicrobiomeParameters,
    RESULT_SUBDIR as MICROBIOME_RESULT_SUBDIR,
    solve_ecological_steady_state,
)

try:
    from exp.TM.transport_map_density import estimate_mutual_information_transport_map
except ModuleNotFoundError:
    from TM.transport_map_density import estimate_mutual_information_transport_map


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "network_revival_microbiome_ei"
DEFAULT_MICROBIOME_RESULTS_DIR = REPO_ROOT / MICROBIOME_RESULT_SUBDIR


@dataclass(frozen=True)
class MicrobiomeEIConfig:
    delta_max: float = 10.0
    n_delta: int = 32
    seed: int = 42
    target_noise_fraction: float = 0.01
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    node_indices: tuple[int, ...] | None = None
    params: MicrobiomeParameters = field(default_factory=MicrobiomeParameters)

    def __post_init__(self) -> None:
        if self.delta_max <= 0.0:
            raise ValueError("delta_max must be positive.")
        if self.n_delta < 2:
            raise ValueError("n_delta must be at least 2.")
        if self.target_noise_fraction < 0.0:
            raise ValueError("target_noise_fraction must be nonnegative.")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.node_indices is not None:
            object.__setattr__(self, "node_indices", tuple(int(index) for index in self.node_indices))


def load_microbiome_ei_inputs(
    results_dir: str | Path = DEFAULT_MICROBIOME_RESULTS_DIR,
) -> dict[str, object]:
    """Load active network and paper recovery ranking from the reproduction cache."""

    root = Path(results_dir)
    active = np.load(root / "active_network.npz", allow_pickle=False)
    ranked = pd.read_csv(root / "node_ignition_ranked.csv")
    return {
        "adjacency": active["active_adjacency"],
        "active_indices": active["active_indices"],
        "recovery_ranked": ranked,
        "source_dir": root,
    }


def simulate_single_node_mean_response_samples(
    adjacency: np.ndarray,
    params: MicrobiomeParameters,
    *,
    node_indices: Sequence[int],
    deltas: np.ndarray,
) -> np.ndarray:
    """Return final mean activation samples with shape [node, delta]."""

    nodes = [int(node) for node in node_indices]
    delta_values = np.asarray(deltas, dtype=float).reshape(-1)
    samples = np.empty((len(nodes), delta_values.size), dtype=float)

    for node_pos, node in enumerate(nodes):
        for delta_pos, delta in enumerate(delta_values):
            x_ss = solve_ecological_steady_state(
                adjacency,
                params,
                free_value=params.collapsed_free_value,
                fixed_node=node,
                fixed_value=float(delta),
            )
            samples[node_pos, delta_pos] = float(np.mean(x_ss))
    return samples


def estimate_single_node_mean_response_ei(
    deltas: np.ndarray,
    mean_response_samples: np.ndarray,
    *,
    target_noise_fraction: float = 0.01,
    seed: int = 0,
    clip_negative: bool = True,
) -> dict[str, object]:
    """Estimate I(Delta_i; final mean activation) for each ignition node."""

    delta_array = np.asarray(deltas, dtype=float).reshape(-1, 1)
    samples = np.asarray(mean_response_samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("mean_response_samples must have shape [node, delta].")
    if samples.shape[1] != delta_array.shape[0]:
        raise ValueError("deltas and mean_response_samples disagree on sample count.")

    rng = np.random.default_rng(seed)
    node_ei = np.empty(samples.shape[0], dtype=float)
    target_noise_sigma = np.empty(samples.shape[0], dtype=float)
    bias_correction = np.empty(samples.shape[0], dtype=float)
    backend = "affine_triangular_transport_map"

    for node_pos in range(samples.shape[0]):
        target = samples[node_pos].reshape(-1, 1)
        sigma = max(1.0e-6, float(target_noise_fraction) * float(np.std(target, ddof=1)))
        noisy_target = target + sigma * rng.normal(size=target.shape)
        summary = estimate_mutual_information_transport_map(delta_array, noisy_target)
        value = float(summary["mi_hat"])
        if clip_negative:
            value = max(0.0, value)
        node_ei[node_pos] = value
        target_noise_sigma[node_pos] = sigma
        bias_correction[node_pos] = float(summary["bias_correction"])
        backend = str(summary["backend"])

    return {
        "node_ei": node_ei,
        "target_noise_sigma": target_noise_sigma,
        "bias_correction": bias_correction,
        "backend": backend,
    }


def run_microbiome_single_node_ei(
    config: MicrobiomeEIConfig,
    *,
    adjacency: np.ndarray | None = None,
    active_indices: np.ndarray | None = None,
    recovery_ranked: pd.DataFrame | None = None,
    force_recompute: bool = False,
) -> dict[str, object]:
    """Run or load single-node microbiome EI and comparison artifacts."""

    output_dir = Path(config.output_dir)
    cache_paths = _cache_paths(output_dir)
    if not force_recompute and _all_cache_paths_exist(cache_paths):
        return _load_cached_result(cache_paths)

    if adjacency is None or active_indices is None or recovery_ranked is None:
        loaded = load_microbiome_ei_inputs()
        adjacency = np.asarray(loaded["adjacency"], dtype=float)
        active_indices = np.asarray(loaded["active_indices"], dtype=int)
        recovery_ranked = loaded["recovery_ranked"]
    else:
        adjacency = np.asarray(adjacency, dtype=float)
        active_indices = np.asarray(active_indices, dtype=int)
        recovery_ranked = recovery_ranked.copy()

    nodes = tuple(range(adjacency.shape[0])) if config.node_indices is None else tuple(config.node_indices)
    rng = np.random.default_rng(config.seed)
    deltas = rng.uniform(0.0, float(config.delta_max), size=int(config.n_delta))
    deltas.sort()

    output_dir.mkdir(parents=True, exist_ok=True)
    mean_response_samples = simulate_single_node_mean_response_samples(
        adjacency,
        config.params,
        node_indices=nodes,
        deltas=deltas,
    )
    ei_summary = estimate_single_node_mean_response_ei(
        deltas,
        mean_response_samples,
        target_noise_fraction=config.target_noise_fraction,
        seed=config.seed + 1009,
    )
    node_ei = np.asarray(ei_summary["node_ei"], dtype=float)
    summary_rows = _build_summary_rows(nodes, active_indices, node_ei, ei_summary)
    summary_df = pd.DataFrame(summary_rows).sort_values("ei_mean_response", ascending=False).reset_index(drop=True)
    summary_df["ei_rank"] = np.arange(1, len(summary_df) + 1)
    comparison_df, metrics = compare_ei_to_recovery(summary_df, recovery_ranked)
    manifest = _build_manifest(config, adjacency, nodes, ei_summary["backend"], metrics)

    np.savez_compressed(
        cache_paths["samples_npz"],
        deltas=deltas,
        node_indices=np.asarray(nodes, dtype=int),
        active_indices=active_indices,
        mean_response_samples=mean_response_samples,
        node_ei=node_ei,
        target_noise_sigma=np.asarray(ei_summary["target_noise_sigma"], dtype=float),
        bias_correction=np.asarray(ei_summary["bias_correction"], dtype=float),
    )
    summary_df.to_csv(cache_paths["summary_csv"], index=False)
    comparison_df.to_csv(cache_paths["comparison_csv"], index=False)
    cache_paths["manifest_json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "deltas": deltas,
        "node_indices": np.asarray(nodes, dtype=int),
        "mean_response_samples": mean_response_samples,
        "node_ei": node_ei,
        "summary": summary_df,
        "comparison": comparison_df,
        "metrics": metrics,
        "cache_paths": cache_paths,
        "manifest": manifest,
    }


def compare_ei_to_recovery(
    summary: pd.DataFrame,
    recovery_ranked: pd.DataFrame,
    *,
    k_values: Sequence[int] = (10, 20, 50),
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Join EI and paper recovery metrics, then compute ranking comparison metrics."""

    left = summary.copy()
    if "ei_rank" not in left.columns:
        left = left.sort_values("ei_mean_response", ascending=False).reset_index(drop=True)
        left["ei_rank"] = np.arange(1, len(left) + 1)

    recovery = recovery_ranked.copy()
    recovery["tree_size_rank"] = recovery["tree_size"].rank(method="first", ascending=False).astype(int)
    recovery["state_rank"] = recovery["state"].rank(method="first", ascending=False).astype(int)
    merged = left.merge(recovery, on="active_node", how="inner", suffixes=("", "_paper"))
    merged["success"] = merged["success"].astype(bool)
    merged["ei_residual_vs_tree_size"] = _linear_residual(
        merged["ei_mean_response"].to_numpy(dtype=float),
        merged["tree_size"].to_numpy(dtype=float),
    )

    metrics: dict[str, float] = {}
    metrics.update(_correlation_metrics(merged))
    success = merged["success"].to_numpy(dtype=bool)
    metrics.update(_score_metrics("ei", merged["ei_mean_response"].to_numpy(dtype=float), success, k_values))
    metrics.update(_score_metrics("tree_size", merged["tree_size"].to_numpy(dtype=float), success, k_values))
    metrics.update(_score_metrics("state", merged["state"].to_numpy(dtype=float), success, k_values))
    metrics.update(_logistic_comparison_metrics(merged))
    return merged, metrics


def plot_microbiome_ei_comparison(
    comparison: pd.DataFrame,
    metrics: dict[str, float],
    output_dir: str | Path,
    *,
    k_values: Sequence[int] = (10, 20, 50),
) -> dict[str, Path]:
    """Write the main notebook figures and return their paths."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ei_ranking_curve": out_dir / "microbiome_ei_ranking_curve.png",
        "ei_vs_recovery": out_dir / "microbiome_ei_vs_recovery.png",
        "success_enrichment": out_dir / "microbiome_success_enrichment.png",
        "success_prediction": out_dir / "microbiome_success_prediction_metrics.png",
        "ei_specific_residual": out_dir / "microbiome_ei_specific_residual.png",
    }

    frame = comparison.copy()
    frame["success"] = frame["success"].astype(bool)
    success_colors = np.where(frame["success"], "#D55E00", "#0072B2")

    ranked = frame.sort_values("ei_rank")
    fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    ax.plot(ranked["ei_rank"], ranked["ei_mean_response"], color="#0072B2", lw=1.6)
    ax.scatter(
        ranked.loc[ranked["success"], "ei_rank"],
        ranked.loc[ranked["success"], "ei_mean_response"],
        color="#D55E00",
        s=20,
        label="Successful ignition",
        zorder=3,
    )
    ax.set_xlabel("EI rank")
    ax.set_ylabel(r"$I(\Delta_i; \bar{x}_{final})$")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.grid(color="0.9", linewidth=0.8)
    fig.savefig(paths["ei_ranking_curve"], dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), constrained_layout=True)
    axes[0].scatter(frame["tree_size"], frame["ei_mean_response"], c=success_colors, s=18, edgecolors="none")
    axes[0].set_xlabel("Paper tree size")
    axes[0].set_ylabel(r"$I(\Delta_i; \bar{x}_{final})$")
    axes[0].grid(color="0.9", linewidth=0.8)
    axes[1].scatter(frame["state"], frame["ei_mean_response"], c=success_colors, s=18, edgecolors="none")
    axes[1].set_xlabel("Final mean state")
    axes[1].set_ylabel(r"$I(\Delta_i; \bar{x}_{final})$")
    axes[1].grid(color="0.9", linewidth=0.8)
    fig.savefig(paths["ei_vs_recovery"], dpi=220, bbox_inches="tight")
    plt.close(fig)

    k_available = [int(k) for k in k_values if f"ei_precision_at_{int(k)}" in metrics]
    if not k_available:
        k_available = [10, 20, 50]
    x = np.arange(len(k_available))
    ei_precision = [float(metrics.get(f"ei_precision_at_{k}", np.nan)) for k in k_available]
    tree_precision = [float(metrics.get(f"tree_size_precision_at_{k}", np.nan)) for k in k_available]
    fig, ax = plt.subplots(figsize=(5.4, 3.4), constrained_layout=True)
    width = 0.36
    ax.bar(x - width / 2, ei_precision, width=width, label="EI rank", color="#0072B2")
    ax.bar(x + width / 2, tree_precision, width=width, label="Paper tree rank", color="#999999")
    ax.set_xticks(x, [f"Top {k}" for k in k_available])
    ax.set_ylabel("Precision among top-ranked nodes")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    fig.savefig(paths["success_enrichment"], dpi=220, bbox_inches="tight")
    plt.close(fig)

    metric_names = ["success_auroc", "success_average_precision"]
    methods = ["ei", "tree_size", "state"]
    values = np.array(
        [[float(metrics.get(f"{method}_{metric}", np.nan)) for method in methods] for metric in metric_names],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.2), constrained_layout=True)
    im = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(methods)), ["EI", "Tree size", "State"])
    ax.set_yticks(np.arange(len(metric_names)), ["AUROC", "Average precision"])
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            text = "nan" if not np.isfinite(values[row, col]) else f"{values[row, col]:.2f}"
            ax.text(col, row, text, ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    fig.savefig(paths["success_prediction"], dpi=220, bbox_inches="tight")
    plt.close(fig)

    residual_ranked = frame.sort_values("ei_residual_vs_tree_size", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    ax.scatter(
        np.arange(1, len(residual_ranked) + 1),
        residual_ranked["ei_residual_vs_tree_size"],
        c=np.where(residual_ranked["success"], "#D55E00", "#0072B2"),
        s=16,
        edgecolors="none",
    )
    ax.axhline(0.0, color="0.35", lw=1.0, linestyle="--")
    ax.set_xlabel("Rank by EI residual over tree size")
    ax.set_ylabel("EI residual")
    ax.grid(color="0.9", linewidth=0.8)
    fig.savefig(paths["ei_specific_residual"], dpi=220, bbox_inches="tight")
    plt.close(fig)

    return paths


def _build_summary_rows(
    nodes: Sequence[int],
    active_indices: np.ndarray,
    node_ei: np.ndarray,
    ei_summary: dict[str, object],
) -> list[dict[str, object]]:
    target_noise_sigma = np.asarray(ei_summary["target_noise_sigma"], dtype=float)
    bias_correction = np.asarray(ei_summary["bias_correction"], dtype=float)
    rows = []
    for pos, active_node in enumerate(nodes):
        rows.append(
            {
                "active_node": int(active_node),
                "species_index": int(active_indices[int(active_node)]),
                "ei_mean_response": float(node_ei[pos]),
                "target_noise_sigma": float(target_noise_sigma[pos]),
                "bias_correction": float(bias_correction[pos]),
            }
        )
    return rows


def _correlation_metrics(frame: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    targets = {
        "tree_size": frame["tree_size"].to_numpy(dtype=float),
        "state": frame["state"].to_numpy(dtype=float),
        "paper_rank": -frame["rank"].to_numpy(dtype=float),
    }
    ei = frame["ei_mean_response"].to_numpy(dtype=float)
    for name, values in targets.items():
        metrics[f"spearman_ei_{name}"] = _safe_corr(stats.spearmanr, ei, values)
        metrics[f"kendall_ei_{name}"] = _safe_corr(stats.kendalltau, ei, values)
    return metrics


def _safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.nanstd(x) == 0.0 or np.nanstd(y) == 0.0:
        return float("nan")
    result = fn(x, y)
    return float(result.statistic)


def _score_metrics(prefix: str, scores: np.ndarray, success: np.ndarray, k_values: Sequence[int]) -> dict[str, float]:
    metrics = {
        f"{prefix}_success_auroc": _binary_auroc(scores, success),
        f"{prefix}_success_average_precision": _average_precision(scores, success),
    }
    for k in k_values:
        precision, recall = _precision_recall_at_k(scores, success, int(k))
        metrics[f"{prefix}_precision_at_{int(k)}"] = precision
        metrics[f"{prefix}_recall_at_{int(k)}"] = recall
    return metrics


def _binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores, method="average")
    auc = (float(ranks[labels].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    if not labels.any():
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    precision = cumulative / (np.arange(len(sorted_labels)) + 1.0)
    return float(np.sum(precision[sorted_labels]) / labels.sum())


def _precision_recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=bool)
    if k <= 0:
        raise ValueError("k must be positive.")
    k_eff = min(int(k), len(labels))
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")[:k_eff]
    hits = int(labels[order].sum())
    precision = hits / k_eff if k_eff else float("nan")
    recall = hits / int(labels.sum()) if labels.any() else float("nan")
    return float(precision), float(recall)


def _linear_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(y) < 2 or np.nanstd(x) == 0.0:
        return y - np.nanmean(y)
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def _logistic_comparison_metrics(frame: pd.DataFrame) -> dict[str, float]:
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return {
            "logistic_tree_size_auc": float("nan"),
            "logistic_ei_auc": float("nan"),
            "logistic_tree_size_plus_ei_auc": float("nan"),
        }

    labels = frame["success"].to_numpy(dtype=bool)
    specs = {
        "tree_size": ["tree_size"],
        "ei": ["ei_mean_response"],
        "tree_size_plus_ei": ["tree_size", "ei_mean_response"],
    }
    metrics: dict[str, float] = {}
    for name, cols in specs.items():
        values = frame[cols].to_numpy(dtype=float)
        if len(np.unique(labels)) < 2 or np.any(np.nanstd(values, axis=0) == 0.0):
            metrics[f"logistic_{name}_auc"] = float("nan")
            continue
        values = (values - values.mean(axis=0)) / values.std(axis=0)
        model = LogisticRegression(solver="lbfgs", class_weight="balanced", random_state=0)
        model.fit(values, labels.astype(int))
        prob = model.predict_proba(values)[:, 1]
        metrics[f"logistic_{name}_auc"] = _binary_auroc(prob, labels)
    return metrics


def _cache_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "samples_npz": output_dir / "microbiome_single_node_ei_samples.npz",
        "summary_csv": output_dir / "microbiome_single_node_ei_summary.csv",
        "comparison_csv": output_dir / "microbiome_ei_vs_recovery_comparison.csv",
        "manifest_json": output_dir / "manifest.json",
    }


def _all_cache_paths_exist(cache_paths: dict[str, Path]) -> bool:
    return all(path.exists() for path in cache_paths.values())


def _load_cached_result(cache_paths: dict[str, Path]) -> dict[str, object]:
    arrays = np.load(cache_paths["samples_npz"], allow_pickle=False)
    summary = pd.read_csv(cache_paths["summary_csv"])
    comparison = pd.read_csv(cache_paths["comparison_csv"])
    manifest = json.loads(cache_paths["manifest_json"].read_text(encoding="utf-8"))
    return {
        "deltas": arrays["deltas"],
        "node_indices": arrays["node_indices"],
        "mean_response_samples": arrays["mean_response_samples"],
        "node_ei": arrays["node_ei"],
        "summary": summary,
        "comparison": comparison,
        "metrics": dict(manifest.get("comparison_metrics", {})),
        "cache_paths": cache_paths,
        "manifest": manifest,
    }


def _build_manifest(
    config: MicrobiomeEIConfig,
    adjacency: np.ndarray,
    nodes: Sequence[int],
    backend: str,
    metrics: dict[str, float],
) -> dict[str, object]:
    params_dict = asdict(config.params)
    return {
        "experiment": "network_revival_microbiome_single_node_ei",
        "source_variable": "point_ignition_strength_delta_i",
        "target_variable": "whole_system_final_mean_activation",
        "release_after_forcing": False,
        "transport_backend": str(backend),
        "delta_max": float(config.delta_max),
        "n_delta": int(config.n_delta),
        "seed": int(config.seed),
        "target_noise_fraction": float(config.target_noise_fraction),
        "microbiome_parameters": params_dict,
        "node_count": int(adjacency.shape[0]),
        "evaluated_node_count": int(len(nodes)),
        "comparison_metrics": {key: float(value) for key, value in metrics.items()},
    }
