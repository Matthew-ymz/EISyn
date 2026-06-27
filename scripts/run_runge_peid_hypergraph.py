#!/usr/bin/env python3
"""PEID causal hypergraph (Möbius higher-order EI interaction) on Runge component dynamics.

Pipeline:

1. Reuse the cached MLP transition model trained by ``run_runge_pairwise_mlp_ei.py``
   on the 60 weekly Varimax components (Runge et al. 2015 reproduction). No
   retraining is performed when the cache matches.
2. Draw a single batch of independent maximum-entropy interventions over the
   full lagged input ``[X_{t-3}, X_{t-2}, X_{t-1}, X_t]`` and obtain one fixed
   set of MLP predictions ``\\hat X_{t+1}``. This batch is shared across all
   orders so that Möbius increments are numerically consistent.
3. For each candidate source subset ``A`` of order in ``{1, 2, 3}`` and target
   component ``j`` we estimate the joint effective information

       EI(X_A -> X_j^{t+1})

   with the transport-map continuous MI estimator (clipped at zero). The
   estimator already supports arbitrary-dimensional ``x``.
4. The exact ``k``-order EI interaction follows the Möbius formula (see
   ``docs/高阶PEID协同定义.md`` equations 3, 6, 7):

       Delta_K(T) = sum_{A subset K} (-1)^{|K|-|A|} EI(X_A -> T)

   Delta_K is signed; at k>=3 it is not guaranteed non-negative. The script
   keeps the sign and reports both polarities as "high-order coordination"
   versus "anti-coordination" so the language stays honest.
5. A block-bootstrap (circular block shift, default ~26 weeks) null
   distribution is built by shuffling the target time series before refitting
   the joint EI estimator on the same intervention batch. Empirical p-values
   and z-scores are attached to every candidate hyperedge.
6. Hyper-ACE / Hyper-AMCE scores aggregate the direct hyperedge magnitudes into
   gateway and mediator readouts.

Outputs land in ``results/runge/peid_hypergraph/`` and ``fig/runge/peid_hypergraph/``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

RESULT_SUBDIR = Path("results/runge/peid_hypergraph")
FIG_SUBDIR = Path("fig/runge/peid_hypergraph")
MLP_CACHE_DIR = Path("results/runge/pairwise_mlp_ei")
PAIRWISE_PATH_DIR = Path("results/runge/pairwise_mlp_tm_ei_path_effects")
DEFAULT_COMPONENT_SCORES = Path("results/runge/2015_gateways/component_weekly_scores.csv")


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


@dataclass(frozen=True)
class PeidHypergraphConfig:
    """Frozen configuration for the higher-order PEID hypergraph run."""

    component_scores: Path = DEFAULT_COMPONENT_SCORES
    output_dir: Path = Path(".")
    lag: int = 4
    horizon: int = 1
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.05
    epochs: int = 40
    learning_rate: float = 1.0e-3
    batch_size: int = 256
    weight_decay: float = 1.0e-4
    ridge_alpha: float = 1.0
    ensemble_ridge_alphas: tuple[float, ...] = ()
    linear_blend_grid_steps: int = 0
    freeze_linear_skip: bool = True
    residual_shrinkage: bool = True
    residual_gamma_min: float = -0.5
    residual_gamma_max: float = 0.5
    residual_gamma_steps: int = 101
    early_stopping_patience: int = 40
    min_delta: float = 1.0e-5
    scheduler_patience: int = 12
    gradient_clip_norm: float = 5.0
    intervention_samples: int = 4096
    quantile_low: float = 0.05
    quantile_high: float = 0.95
    source_mode: str = "latest"
    seed: int = 42
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    force_retrain: bool = False
    order_max: int = 3
    candidate_top_sources: int = 14
    candidate_target_topk: int = 10
    candidate_third_source_topk: int = 15
    delta2_significance_z: float = 2.0
    null_reps: int = 20
    block_size: int = 26
    tm_jitter: float = 1.0e-6
    pairwise_matrix_path: Path = PAIRWISE_PATH_DIR / "pairwise_ei_matrix.csv"
    pairwise_gateway_path: Path = PAIRWISE_PATH_DIR / "gateway_scores.csv"
    pairwise_mediator_path: Path = PAIRWISE_PATH_DIR / "mediator_scores.csv"
    self_check_samples: int = 5


# ---------------------------------------------------------------------------
# Shared utilities re-exported from the pairwise pipeline.
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    root_candidate = _repo_root() / candidate
    return root_candidate if root_candidate.exists() else candidate.resolve()


def _import_pairwise():
    """Import the pairwise module without re-running its CLI."""

    _ensure_repo_on_path()
    from scripts import run_runge_pairwise_mlp_ei as pairwise  # type: ignore

    return pairwise


def _import_transport_map():
    _ensure_repo_on_path()
    from exp.TM.transport_map_density import estimate_mutual_information_transport_map  # type: ignore

    return estimate_mutual_information_transport_map


def _jsonable_config(config: PeidHypergraphConfig) -> dict[str, object]:
    data = asdict(config)
    for key in ("component_scores", "output_dir", "pairwise_matrix_path", "pairwise_gateway_path", "pairwise_mediator_path"):
        data[key] = str(getattr(config, key))
    return data


def parse_float_tuple(text: str | Sequence[float] | None) -> tuple[float, ...]:
    if text is None:
        return ()
    if isinstance(text, str):
        if not text.strip():
            return ()
        return tuple(float(part.strip()) for part in text.split(",") if part.strip())
    return tuple(float(value) for value in text)


def _model_config_hash_for_cache(pairwise_module, config: PeidHypergraphConfig, *, n_components: int, n_rows: int) -> str:
    """Compute the cache key the pairwise pipeline uses for the MLP cache.

    The pairwise module already encodes the cache hash from a dataclass-level
    config; we mirror its required fields so the cache file is reused when the
    training hyperparameters line up with the previous run.
    """

    proxy = pairwise_module.PairwiseMlpEiConfig(
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
    frame = pairwise_module.load_component_scores(_resolve_path(config.component_scores))
    data_hash = pairwise_module._frame_content_hash(frame)
    return pairwise_module._model_config_hash(proxy, n_components=n_components, n_rows=n_rows, data_hash=data_hash)


# ---------------------------------------------------------------------------
# Higher-order joint EI estimation.
# ---------------------------------------------------------------------------


def _source_block(
    intervention_features: np.ndarray,
    *,
    indices: Sequence[int],
    n_components: int,
    lag: int,
    source_mode: str,
) -> np.ndarray:
    """Extract a multi-source intervention block from the full lagged matrix."""

    if source_mode == "latest":
        cols = [(lag - 1) * int(n_components) + int(i) for i in indices]
    elif source_mode == "history":
        cols = [
            lag_idx * int(n_components) + int(i)
            for i in indices
            for lag_idx in range(int(lag))
        ]
    else:
        raise ValueError("source_mode must be 'latest' or 'history'.")
    return intervention_features[:, cols]


def _clipped_joint_ei(
    source_block: np.ndarray,
    target_column: np.ndarray,
    *,
    estimator,
    jitter: float,
) -> dict[str, float]:
    """Wrap the transport-map estimator and clip MI at zero."""

    summary = estimator(source_block, target_column.reshape(-1, 1), jitter=jitter)
    raw = float(summary["mi_hat"])
    return {
        "ei": max(0.0, raw),
        "ei_raw": raw,
        "bias_correction": float(summary["bias_correction"]),
    }


def joint_ei_table(
    intervention_features: np.ndarray,
    predictions: np.ndarray,
    *,
    subset_target_pairs: Iterable[tuple[tuple[int, ...], int]],
    n_components: int,
    lag: int,
    source_mode: str,
    estimator,
    jitter: float,
) -> pd.DataFrame:
    """Estimate EI(X_A -> X_target) for every requested (A, target) pair."""

    rows: list[dict[str, object]] = []
    for subset, target in subset_target_pairs:
        block = _source_block(
            intervention_features,
            indices=subset,
            n_components=n_components,
            lag=lag,
            source_mode=source_mode,
        )
        result = _clipped_joint_ei(
            block,
            predictions[:, int(target)],
            estimator=estimator,
            jitter=jitter,
        )
        rows.append(
            {
                "subset": tuple(sorted(int(i) for i in subset)),
                "subset_str": ",".join(str(int(i)) for i in sorted(subset)),
                "order": int(len(subset)),
                "target_index": int(target),
                "ei": float(result["ei"]),
                "ei_raw": float(result["ei_raw"]),
                "bias_correction": float(result["bias_correction"]),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Möbius higher-order EI interaction.
# ---------------------------------------------------------------------------


def _subset_key(indices: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(int(i) for i in indices))


def mobius_delta(
    subset: Sequence[int],
    target: int,
    ei_lookup: dict[tuple[tuple[int, ...], int], float],
) -> float:
    """Compute Delta_K(T) = sum_{A subset K} (-1)^{|K|-|A|} EI(X_A -> T)."""

    k = len(subset)
    if k == 0:
        return 0.0
    total = 0.0
    for size in range(1, k + 1):
        for combo in itertools.combinations(sorted(int(i) for i in subset), size):
            sign = (-1) ** (k - size)
            ei_value = ei_lookup.get((tuple(combo), int(target)))
            if ei_value is None:
                raise KeyError(f"Missing joint EI for subset {combo} -> target {target}")
            total += float(sign) * float(ei_value)
    return float(total)


def _mobius_recursive(
    subset: Sequence[int],
    target: int,
    ei_lookup: dict[tuple[tuple[int, ...], int], float],
) -> float:
    """Recursive Delta_K(T) = EI(X_K->T) - sum_{A subset K, |A|<|K|} Delta_A(T).

    Used only by the self-check to cross-validate the closed-form computation.
    """

    k = len(subset)
    if k == 0:
        return 0.0
    sorted_subset = tuple(sorted(int(i) for i in subset))
    own_ei = float(ei_lookup[(sorted_subset, int(target))])
    contribution = own_ei
    for size in range(1, k):
        for combo in itertools.combinations(sorted_subset, size):
            contribution -= _mobius_recursive(combo, target, ei_lookup)
    return contribution


# ---------------------------------------------------------------------------
# Candidate selection.
# ---------------------------------------------------------------------------


def _candidate_source_pool(matrix: np.ndarray, top_sources: int) -> np.ndarray:
    """Pick the top sources by outgoing pairwise EI mass."""

    out_strength = matrix.sum(axis=1) - np.diag(matrix)
    top = np.argsort(out_strength)[::-1][: int(top_sources)]
    return np.sort(top)


def _target_pool_per_source(matrix: np.ndarray, top_targets: int) -> dict[int, set[int]]:
    """For each source, retain only its strongest pairwise targets."""

    n = matrix.shape[0]
    pool: dict[int, set[int]] = {}
    for source in range(n):
        row = matrix[source].copy()
        row[source] = -np.inf
        order = np.argsort(row)[::-1][: int(top_targets)]
        pool[source] = {int(t) for t in order if matrix[source, int(t)] > 0.0}
    return pool


def enumerate_subset_targets(
    matrix: np.ndarray,
    *,
    n_components: int,
    candidate_top_sources: int,
    candidate_target_topk: int,
    third_source_topk: int,
    order_max: int,
) -> dict[str, list[tuple[tuple[int, ...], int]]]:
    """Enumerate the (subset, target) pairs we need joint EI for, per order.

    The 1st-order requirement is the full N x N matrix (we still re-estimate it
    in the shared intervention batch so all orders share a numerical baseline).
    Higher orders are pruned by combining a source pool with per-source target
    pools.
    """

    n = int(n_components)
    pool = _candidate_source_pool(matrix, candidate_top_sources)
    target_pool = _target_pool_per_source(matrix, candidate_target_topk)

    order_pairs: dict[str, list[tuple[tuple[int, ...], int]]] = {"1": [], "2": [], "3": []}
    for i in range(n):
        for j in range(n):
            order_pairs["1"].append(((int(i),), int(j)))

    if order_max >= 2:
        for i, j in itertools.combinations(pool.tolist(), 2):
            shared_targets = target_pool[int(i)] | target_pool[int(j)]
            shared_targets = {t for t in shared_targets if t not in (int(i), int(j))}
            for target in sorted(shared_targets):
                order_pairs["2"].append(((int(i), int(j)), int(target)))

    if order_max >= 3:
        # third source restricted to a small high-gateway set
        out_strength = matrix.sum(axis=1) - np.diag(matrix)
        third_pool = np.argsort(out_strength)[::-1][: int(third_source_topk)].tolist()
        third_pool = sorted({int(x) for x in third_pool})
        seen: set[tuple[tuple[int, ...], int]] = set()
        for i, j in itertools.combinations(pool.tolist(), 2):
            for ell in third_pool:
                if ell in (int(i), int(j)):
                    continue
                subset = tuple(sorted((int(i), int(j), int(ell))))
                shared_targets = target_pool[int(i)] | target_pool[int(j)] | target_pool[int(ell)]
                shared_targets = {t for t in shared_targets if t not in subset}
                for target in sorted(shared_targets):
                    key = (subset, int(target))
                    if key in seen:
                        continue
                    seen.add(key)
                    order_pairs["3"].append((subset, int(target)))
    return order_pairs


# ---------------------------------------------------------------------------
# Block bootstrap null.
# ---------------------------------------------------------------------------


def _shift_rows(values: np.ndarray, offset: int) -> np.ndarray:
    """Circular shift along the sample axis (used for block-bootstrap surrogates)."""

    n = values.shape[0]
    offset = int(offset) % max(1, n)
    if offset == 0:
        return values.copy()
    return np.concatenate([values[offset:], values[:offset]], axis=0)


def build_null_for_hyperedges(
    candidates: pd.DataFrame,
    *,
    intervention_features: np.ndarray,
    predictions: np.ndarray,
    n_components: int,
    lag: int,
    source_mode: str,
    estimator,
    jitter: float,
    null_reps: int,
    block_size: int,
    seed: int,
) -> pd.DataFrame:
    """Block-bootstrap shifts the target time series to break the causal link.

    For each surrogate we re-estimate every (subset, target) joint EI and the
    corresponding Möbius Delta. The empirical distribution of surrogate Delta
    values produces null_mean, null_std, z, p_empirical columns. The reps live
    in long form so subsets of any order share the same code path.
    """

    if candidates.empty:
        return candidates.copy()
    rng = np.random.default_rng(int(seed) + 8081)
    n_samples = int(predictions.shape[0])
    block = max(1, int(block_size))
    null_records: dict[tuple[tuple[int, ...], int], list[float]] = {
        (tuple(row.subset), int(row.target_index)): []
        for row in candidates.itertuples()
    }
    ei_lookup_template: dict[tuple[tuple[int, ...], int], float] = {}
    needed_subset_targets: set[tuple[tuple[int, ...], int]] = set()
    # Collect every subset that participates in any candidate (including
    # proper subsets), since Möbius needs them too.
    for row in candidates.itertuples():
        subset = tuple(int(i) for i in row.subset)
        target = int(row.target_index)
        for size in range(1, len(subset) + 1):
            for combo in itertools.combinations(subset, size):
                needed_subset_targets.add((tuple(sorted(combo)), target))

    for rep in range(int(null_reps)):
        # circular shift by a multiple of block_size to preserve season-scale
        # autocorrelation. We pick the offset uniformly from [block, n-block].
        max_offset = max(1, n_samples - block - 1)
        offset_blocks = int(rng.integers(low=1, high=max(2, max_offset // block + 1)))
        offset = (offset_blocks * block) % max(1, n_samples)
        shifted_pred = _shift_rows(predictions, offset)
        ei_lookup: dict[tuple[tuple[int, ...], int], float] = {}
        for subset_key, target in needed_subset_targets:
            block_x = _source_block(
                intervention_features,
                indices=subset_key,
                n_components=n_components,
                lag=lag,
                source_mode=source_mode,
            )
            ei = _clipped_joint_ei(
                block_x,
                shifted_pred[:, int(target)],
                estimator=estimator,
                jitter=jitter,
            )["ei"]
            ei_lookup[(subset_key, target)] = float(ei)
        for row in candidates.itertuples():
            subset = tuple(int(i) for i in row.subset)
            target = int(row.target_index)
            delta = mobius_delta(subset, target, ei_lookup)
            null_records[(subset, target)].append(float(delta))

    enriched = candidates.copy()
    null_means = []
    null_stds = []
    zs = []
    p_emps = []
    for row in enriched.itertuples():
        samples = null_records[(tuple(row.subset), int(row.target_index))]
        if not samples:
            null_means.append(np.nan)
            null_stds.append(np.nan)
            zs.append(np.nan)
            p_emps.append(np.nan)
            continue
        arr = np.asarray(samples, dtype=float)
        mu = float(arr.mean())
        sigma = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        null_means.append(mu)
        null_stds.append(sigma)
        if sigma > 1.0e-12:
            zs.append((float(row.delta_K) - mu) / sigma)
        else:
            zs.append(np.nan)
        # two-sided empirical p-value: fraction of |null| >= |observed|
        p_emps.append(float((np.abs(arr) >= abs(float(row.delta_K))).sum() + 1) / float(len(arr) + 1))
    enriched["null_mean"] = null_means
    enriched["null_std"] = null_stds
    enriched["z"] = zs
    enriched["p_empirical"] = p_emps
    return enriched


# ---------------------------------------------------------------------------
# Hypergraph aggregation: Hyper-ACE, Hyper-AMCE.
# ---------------------------------------------------------------------------


def aggregate_hyper_gateway(
    hyperedges: pd.DataFrame,
    *,
    n_components: int,
    component_names: Sequence[str],
    significance_z: float,
) -> pd.DataFrame:
    """Per-component gateway scores split by order and an aggregate score."""

    n = int(n_components)
    by_order = {1: np.zeros(n), 2: np.zeros(n), 3: np.zeros(n)}
    by_order_signed = {1: np.zeros(n), 2: np.zeros(n), 3: np.zeros(n)}
    for row in hyperedges.itertuples():
        order = int(row.order)
        if order not in by_order:
            continue
        # Significance gating only for k>=2 (order 1 already standard pairwise).
        if order >= 2 and (np.isnan(getattr(row, "z", np.nan)) or abs(float(row.z)) < float(significance_z)):
            continue
        share = float(row.delta_K) / float(order)
        share_signed = share  # keep sign for higher-order anti-coordination
        share_mag = abs(share)
        for src in row.subset:
            by_order[order][int(src)] += share_mag
            by_order_signed[order][int(src)] += share_signed

    denom_per_node = max(1, n - 1)
    rows = []
    for idx, name in enumerate(component_names):
        rows.append(
            {
                "component": name,
                "component_index": idx,
                "hyper_ace_order1": by_order[1][idx] / denom_per_node,
                "hyper_ace_order2": by_order[2][idx] / denom_per_node,
                "hyper_ace_order3": by_order[3][idx] / denom_per_node,
                "hyper_ace_order2_signed": by_order_signed[2][idx] / denom_per_node,
                "hyper_ace_order3_signed": by_order_signed[3][idx] / denom_per_node,
                "hyper_ace_total": (
                    by_order[1][idx] + by_order[2][idx] + by_order[3][idx]
                ) / denom_per_node,
            }
        )
    frame = pd.DataFrame(rows)
    frame["hyper_ace_rank"] = frame["hyper_ace_total"].rank(ascending=False, method="min").astype(int)
    return frame.sort_values("hyper_ace_total", ascending=False).reset_index(drop=True)


def aggregate_hyper_mediator(
    hyperedges: pd.DataFrame,
    *,
    n_components: int,
    component_names: Sequence[str],
    significance_z: float,
    pairwise_mediator: pd.DataFrame,
) -> pd.DataFrame:
    """Hyper-AMCE: direct synergetic source-membership component.

    For each node ``m`` we accumulate:
    - ``mediator_path_amce``: the existing path-effect AMCE from the pairwise
      pipeline, kept only as a diagnostic baseline.
    - ``mediator_synergy_order2``: |delta_2| share when ``m`` co-appears with
      another source on a target subset, normalized by the order.
    - ``mediator_synergy_order3``: same idea for k=3 hyperedges. We split into
      positive and signed components so anti-coordination contributions are
      visible.

    The total intentionally excludes ``mediator_path_amce`` so this score does
    not average over multi-step pairwise mediated paths.
    """

    n = int(n_components)
    syn2 = np.zeros(n)
    syn3 = np.zeros(n)
    syn3_signed = np.zeros(n)
    for row in hyperedges.itertuples():
        order = int(row.order)
        if order < 2:
            continue
        if np.isnan(getattr(row, "z", np.nan)) or abs(float(row.z)) < float(significance_z):
            continue
        contribution = float(row.delta_K) / float(order)
        contribution_mag = abs(contribution)
        for src in row.subset:
            if order == 2:
                syn2[int(src)] += contribution_mag
            elif order == 3:
                syn3[int(src)] += contribution_mag
                syn3_signed[int(src)] += contribution

    pair_amce_lookup = {
        int(row.component_index): float(row.amce) for row in pairwise_mediator.itertuples()
    }
    denom_per_node = max(1, n - 1)
    rows = []
    for idx, name in enumerate(component_names):
        path_amce = pair_amce_lookup.get(int(idx), 0.0)
        rows.append(
            {
                "component": name,
                "component_index": idx,
                "mediator_path_amce": path_amce,
                "mediator_synergy_order2": syn2[idx] / denom_per_node,
                "mediator_synergy_order3": syn3[idx] / denom_per_node,
                "mediator_synergy_order3_signed": syn3_signed[idx] / denom_per_node,
                "hyper_amce_total": syn2[idx] / denom_per_node + syn3[idx] / denom_per_node,
            }
        )
    frame = pd.DataFrame(rows)
    frame["hyper_amce_rank"] = frame["hyper_amce_total"].rank(ascending=False, method="min").astype(int)
    return frame.sort_values("hyper_amce_total", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Comparison against the pairwise baseline.
# ---------------------------------------------------------------------------


def _topk_overlap(ranked_a: Sequence[int], ranked_b: Sequence[int], k: int) -> tuple[int, list[int]]:
    head_a = list(ranked_a)[: int(k)]
    head_b = list(ranked_b)[: int(k)]
    common = sorted(set(head_a) & set(head_b))
    return len(common), common


def _spearman_kendall(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, float]:
    if len(values_a) < 2 or len(values_b) < 2:
        return float("nan"), float("nan")
    rank_a = pd.Series(values_a).rank().to_numpy()
    rank_b = pd.Series(values_b).rank().to_numpy()
    spearman = float(np.corrcoef(rank_a, rank_b)[0, 1])
    # Kendall tau via pairwise concordance count
    n = len(values_a)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = values_a[i] - values_a[j]
            db = values_b[i] - values_b[j]
            if da == 0 or db == 0:
                continue
            if (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1
        if (i + 1) * n > 50000:  # safety stop for very large n
            break
    pairs = concordant + discordant
    kendall = float((concordant - discordant) / pairs) if pairs > 0 else float("nan")
    return spearman, kendall


def compare_rankings(
    hyper_gateway: pd.DataFrame,
    hyper_mediator: pd.DataFrame,
    pairwise_gateway: pd.DataFrame,
    pairwise_mediator: pd.DataFrame,
) -> dict[str, object]:
    """Top-k overlap and rank correlations against the pairwise readouts."""

    def _ordered_indices(frame: pd.DataFrame, score_col: str) -> list[int]:
        sorted_frame = frame.sort_values(score_col, ascending=False)
        return [int(i) for i in sorted_frame["component_index"].tolist()]

    pair_gw_order = _ordered_indices(pairwise_gateway, "ace")
    hyper_gw_order = _ordered_indices(hyper_gateway, "hyper_ace_total")
    pair_md_order = _ordered_indices(pairwise_mediator, "amce")
    hyper_md_order = _ordered_indices(hyper_mediator, "hyper_amce_total")

    overlaps = {}
    for k in (5, 10, 15, 20):
        gw_overlap, gw_common = _topk_overlap(hyper_gw_order, pair_gw_order, k)
        md_overlap, md_common = _topk_overlap(hyper_md_order, pair_md_order, k)
        overlaps[str(k)] = {
            "gateway": {"overlap": gw_overlap, "common": gw_common},
            "mediator": {"overlap": md_overlap, "common": md_common},
        }

    pair_gw_score = pairwise_gateway.set_index("component_index")["ace"].reindex(range(len(pair_gw_order))).to_numpy()
    hyper_gw_score = hyper_gateway.set_index("component_index")["hyper_ace_total"].reindex(range(len(pair_gw_order))).to_numpy()
    pair_md_score = pairwise_mediator.set_index("component_index")["amce"].reindex(range(len(pair_md_order))).to_numpy()
    hyper_md_score = hyper_mediator.set_index("component_index")["hyper_amce_total"].reindex(range(len(pair_md_order))).to_numpy()
    gw_spearman, gw_kendall = _spearman_kendall(pair_gw_score, hyper_gw_score)
    md_spearman, md_kendall = _spearman_kendall(pair_md_score, hyper_md_score)
    return {
        "topk_overlap": overlaps,
        "gateway_rank_correlation": {"spearman": gw_spearman, "kendall": gw_kendall},
        "mediator_rank_correlation": {"spearman": md_spearman, "kendall": md_kendall},
    }


# ---------------------------------------------------------------------------
# Plot helpers.
# ---------------------------------------------------------------------------


PAPER_COMPONENT_LABEL_MAP: dict[int, int] = {
    7: 18,
    18: 7,
    8: 26,
    26: 8,
    21: 48,
    48: 21,
}


def paper_component_label(index: object) -> str:
    paper_index = PAPER_COMPONENT_LABEL_MAP.get(int(index), int(index))
    return f"No.{paper_index}"


def save_delta2_heatmap(hyperedges: pd.DataFrame, n_components: int, output_path: Path, *, title: str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Aggregate Delta_2 magnitudes onto a (i,j) source-pair grid by summing
    # over targets so we can see which pairs are coordinated overall.
    matrix = np.zeros((int(n_components), int(n_components)), dtype=float)
    for row in hyperedges[hyperedges["order"] == 2].itertuples():
        i, j = sorted(int(s) for s in row.subset)
        matrix[i, j] += float(row.delta_K)
        matrix[j, i] += float(row.delta_K)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    vmax = float(np.percentile(np.abs(matrix[matrix != 0]), 99)) if np.any(matrix != 0) else 1.0
    image = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("source j")
    ax.set_ylabel("source i")
    step = max(1, int(n_components) // 12)
    ticks = np.arange(0, int(n_components), step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(int(t)) for t in ticks], rotation=45, ha="right")
    ax.set_yticklabels([str(int(t)) for t in ticks])
    fig.colorbar(image, ax=ax, label="sum_target Delta_2 (bits)")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_hyper_gateway_ranking(scores: pd.DataFrame, output_path: Path, *, title: str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_frame = scores.head(20).copy()
    x = np.arange(len(plot_frame))
    fig, ax = plt.subplots(figsize=(8.6, 4.4), constrained_layout=True)
    ax.bar(x, plot_frame["hyper_ace_order1"], width=0.7, label="order 1 (pairwise)", color="#4c78a8")
    bottom = plot_frame["hyper_ace_order1"].to_numpy()
    ax.bar(x, plot_frame["hyper_ace_order2"], width=0.7, bottom=bottom, label="order 2 |Delta_2| share", color="#f58518")
    bottom = bottom + plot_frame["hyper_ace_order2"].to_numpy()
    if "hyper_ace_order3" in plot_frame and float(plot_frame["hyper_ace_order3"].abs().sum()) > 0.0:
        ax.bar(x, plot_frame["hyper_ace_order3"], width=0.7, bottom=bottom, label="order 3 |Delta_3| share", color="#54a24b")
    ax.set_xticks(x)
    ax.set_xticklabels([paper_component_label(index) for index in plot_frame["component_index"]], rotation=50, ha="right")
    ax.set_ylabel("hyper-ACE (bits)")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_hyper_mediator_ranking(scores: pd.DataFrame, output_path: Path, *, title: str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_frame = scores.head(20).copy()
    x = np.arange(len(plot_frame))
    fig, ax = plt.subplots(figsize=(8.6, 4.4), constrained_layout=True)
    ax.bar(x, plot_frame["mediator_synergy_order2"], width=0.7, label="order 2 synergy share", color="#f58518")
    bottom = plot_frame["mediator_synergy_order2"].to_numpy()
    if "mediator_synergy_order3" in plot_frame and float(plot_frame["mediator_synergy_order3"].abs().sum()) > 0.0:
        ax.bar(x, plot_frame["mediator_synergy_order3"], width=0.7, bottom=bottom, label="order 3 synergy share", color="#b279a2")
    ax.set_xticks(x)
    ax.set_xticklabels([paper_component_label(index) for index in plot_frame["component_index"]], rotation=50, ha="right")
    ax.set_ylabel("hyper-AMCE (bits)")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_delta3_top_table(hyperedges: pd.DataFrame, output_path: Path, *, title: str, top_n: int = 15) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    subset = hyperedges[hyperedges["order"] == 3].copy()
    if subset.empty:
        # Save an empty placeholder so the report links resolve.
        fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
        ax.text(0.5, 0.5, "no order-3 hyperedges in this run", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output
    subset["abs_delta"] = subset["delta_K"].abs()
    subset = subset.sort_values("abs_delta", ascending=False).head(int(top_n))
    text_rows = []
    for row in subset.itertuples():
        sources = "+".join(f"No.{int(s)}" for s in row.subset)
        sign = "+" if float(row.delta_K) >= 0 else "-"
        z_text = f"{float(row.z):+.2f}" if not np.isnan(float(getattr(row, "z", np.nan))) else "n/a"
        text_rows.append(f"{sources} -> No.{int(row.target_index)}  Delta_3={sign}{abs(float(row.delta_K)):.4f}  z={z_text}")
    fig, ax = plt.subplots(figsize=(7.6, 0.32 * len(text_rows) + 1.0), constrained_layout=True)
    ax.set_axis_off()
    ax.set_title(title, loc="left")
    for i, text in enumerate(text_rows):
        color = "#1f77b4" if "Delta_3=+" in text else "#d62728"
        ax.text(0.02, 1.0 - (i + 0.5) / max(1, len(text_rows)), text, color=color, family="monospace")
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def save_enso_hyperedge_subgraph(
    hyperedges: pd.DataFrame,
    pairwise_matrix: np.ndarray,
    output_path: Path,
    *,
    title: str,
    enso_nodes: Sequence[int] = (0, 1, 9, 16, 18, 19, 57),
) -> Path:
    """Show pairwise edges among ENSO nodes plus any significant 2nd-order hyperedge incident on them."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    max_nodes = min(int(pairwise_matrix.shape[0]), int(pairwise_matrix.shape[1]))
    nodes = [int(node) for node in enso_nodes if 0 <= int(node) < max_nodes]
    if len(nodes) < 2:
        ax.text(
            0.5,
            0.5,
            "ENSO subgraph requires at least two available ENSO nodes.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        ax.set_axis_off()
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output
    n = len(nodes)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radius = 1.0
    positions = {int(node): (radius * float(np.cos(a)), radius * float(np.sin(a))) for node, a in zip(nodes, angles)}
    # Pairwise edges (gray)
    for src in nodes:
        for tgt in nodes:
            if src == tgt:
                continue
            ei_value = float(pairwise_matrix[int(src), int(tgt)])
            if ei_value <= 0.0:
                continue
            x0, y0 = positions[int(src)]
            x1, y1 = positions[int(tgt)]
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->",
                    color="gray",
                    alpha=min(1.0, max(0.15, ei_value * 60.0)),
                    lw=min(2.4, max(0.6, ei_value * 80.0)),
                ),
            )
    # Hyperedges of order 2 incident on ENSO nodes: draw a triangle (i, j, target) with light fill
    drawn_hyper = 0
    for row in hyperedges[hyperedges["order"] == 2].sort_values("delta_K", ascending=False).itertuples():
        subset = tuple(int(s) for s in row.subset)
        target = int(row.target_index)
        if target not in positions:
            continue
        if not all(s in positions for s in subset):
            continue
        if np.isnan(float(getattr(row, "z", np.nan))) or abs(float(row.z)) < 2.0:
            continue
        if drawn_hyper >= 6:
            break
        xs = [positions[s][0] for s in subset] + [positions[target][0]]
        ys = [positions[s][1] for s in subset] + [positions[target][1]]
        color = "#1f77b4" if float(row.delta_K) >= 0 else "#d62728"
        ax.fill(xs[:3] + [positions[target][0]], ys[:3] + [positions[target][1]], color=color, alpha=0.10)
        for s in subset:
            sx, sy = positions[s]
            tx, ty = positions[target]
            ax.plot([sx, tx], [sy, ty], color=color, alpha=0.6, lw=1.2, linestyle="--")
        ax.text(positions[target][0] * 1.18, positions[target][1] * 1.18,
                f"{'+' if float(row.delta_K)>=0 else '-'}{abs(float(row.delta_K)):.3f}",
                color=color, fontsize=7)
        drawn_hyper += 1

    for node, (x, y) in positions.items():
        ax.scatter([x], [y], s=300, color="#4c78a8", edgecolor="black", zorder=5)
        ax.text(x * 1.12, y * 1.12, f"No.{int(node)}", ha="center", va="center", fontsize=9, zorder=6)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------
# Run / CLI.
# ---------------------------------------------------------------------------


def dependency_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "matplotlib", "torch"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def run(config: PeidHypergraphConfig) -> dict[str, Path]:
    pairwise = _import_pairwise()
    estimator = _import_transport_map()

    component_scores_path = _resolve_path(config.component_scores)
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

    proxy_pairwise_config = pairwise.PairwiseMlpEiConfig(
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
    config_hash = pairwise._model_config_hash(
        proxy_pairwise_config,
        n_components=n_components,
        n_rows=len(frame),
        data_hash=data_hash,
    )
    ensemble_alphas = tuple(config.ensemble_ridge_alphas) if config.ensemble_ridge_alphas else (float(config.ridge_alpha),)
    model_dir = Path(config.output_dir) / MLP_CACHE_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    model_paths: list[Path] = []
    models: list[object] = []
    member_summaries: list[dict[str, object]] = []
    scalers = None
    cache_reused = True
    loss_history: list[float] = []
    for member_index, alpha in enumerate(ensemble_alphas):
        member_config = replace(proxy_pairwise_config, ridge_alpha=float(alpha), ensemble_ridge_alphas=())
        member_hash = pairwise._model_config_hash(
            member_config,
            n_components=n_components,
            n_rows=len(frame),
            data_hash=data_hash,
        )
        if len(ensemble_alphas) == 1:
            member_path = model_dir / "mlp_transition.pt"
        else:
            alpha_label = str(float(alpha)).replace("-", "m").replace(".", "p")
            member_path = model_dir / f"mlp_transition_alpha{alpha_label}_{member_hash}.pt"
        member_model, member_scalers, member_loss_history, member_cache_reused = pairwise.train_or_load_mlp(
            splits,
            member_config,
            member_path,
            config_hash=member_hash,
        )
        model_paths.append(member_path)
        models.append(member_model)
        scalers = member_scalers if scalers is None else scalers
        cache_reused = cache_reused and bool(member_cache_reused)
        if member_index == 0:
            loss_history = member_loss_history
        member_summaries.append(
            {
                "ridge_alpha": float(alpha),
                "model_cache": str(member_path),
                "model_config_hash": member_hash,
                "cache_reused": bool(member_cache_reused),
                "training": getattr(member_model, "training_summary", {}),
            }
        )
    model = models[0] if len(models) == 1 else pairwise.AveragedTransition(models)
    assert scalers is not None
    linear_blend: dict[str, object] = {"enabled": False}
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
            training_summary={
                "type": "validation_linear_blend",
                "base_model": getattr(model, "training_summary", {}),
                "ridge_model": getattr(ridge_transition, "training_summary", {}),
                **blend,
            },
        )
        linear_blend = {
            "enabled": True,
            "ridge_alpha": float(config.ridge_alpha),
            "grid_steps": int(config.linear_blend_grid_steps),
            **blend,
        }

    result_dir = Path(config.output_dir) / RESULT_SUBDIR
    fig_dir = Path(config.output_dir) / FIG_SUBDIR
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Shared intervention batch and predictions.
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

    # Pairwise matrix as a candidate prior (we re-estimate it inside the shared
    # batch for consistency with higher orders).
    print(f"[peid] estimating order-1 joint EI ({n_components * n_components} pairs)...", flush=True)
    start = time.time()
    order1_pairs = [((i,), j) for i in range(n_components) for j in range(n_components)]
    order1_table = joint_ei_table(
        intervention_features,
        predictions,
        subset_target_pairs=order1_pairs,
        n_components=n_components,
        lag=int(config.lag),
        source_mode=str(config.source_mode),
        estimator=estimator,
        jitter=float(config.tm_jitter),
    )
    print(f"[peid] order-1 EI done in {time.time() - start:.1f}s", flush=True)

    matrix = np.zeros((n_components, n_components), dtype=float)
    for row in order1_table.itertuples():
        i = int(row.subset[0])
        matrix[i, int(row.target_index)] = float(row.ei)
    pd.DataFrame(matrix, index=names, columns=names).to_csv(result_dir / "pairwise_ei_matrix.csv")

    # Higher-order candidate enumeration.
    pair_lists = enumerate_subset_targets(
        matrix,
        n_components=n_components,
        candidate_top_sources=int(config.candidate_top_sources),
        candidate_target_topk=int(config.candidate_target_topk),
        third_source_topk=int(config.candidate_third_source_topk),
        order_max=int(config.order_max),
    )
    print(
        "[peid] candidate counts: order2={n2}, order3={n3}".format(
            n2=len(pair_lists["2"]), n3=len(pair_lists["3"])
        ),
        flush=True,
    )

    order2_table = pd.DataFrame()
    order3_table = pd.DataFrame()
    if int(config.order_max) >= 2 and pair_lists["2"]:
        start = time.time()
        order2_table = joint_ei_table(
            intervention_features,
            predictions,
            subset_target_pairs=pair_lists["2"],
            n_components=n_components,
            lag=int(config.lag),
            source_mode=str(config.source_mode),
            estimator=estimator,
            jitter=float(config.tm_jitter),
        )
        print(f"[peid] order-2 EI done in {time.time() - start:.1f}s", flush=True)
    if int(config.order_max) >= 3 and pair_lists["3"]:
        start = time.time()
        order3_table = joint_ei_table(
            intervention_features,
            predictions,
            subset_target_pairs=pair_lists["3"],
            n_components=n_components,
            lag=int(config.lag),
            source_mode=str(config.source_mode),
            estimator=estimator,
            jitter=float(config.tm_jitter),
        )
        print(f"[peid] order-3 EI done in {time.time() - start:.1f}s", flush=True)

    all_ei = pd.concat([order1_table, order2_table, order3_table], ignore_index=True)
    ei_lookup: dict[tuple[tuple[int, ...], int], float] = {
        (tuple(int(s) for s in row.subset), int(row.target_index)): float(row.ei)
        for row in all_ei.itertuples()
    }

    # Möbius deltas. We compute one row per subset-target candidate of order
    # in {1, 2, 3}. For order 1 the delta is trivially equal to EI.
    delta_rows: list[dict[str, object]] = []
    for row in all_ei.itertuples():
        subset = tuple(int(s) for s in row.subset)
        target = int(row.target_index)
        try:
            delta = mobius_delta(subset, target, ei_lookup)
        except KeyError:
            # Skip candidate if some required sub-EI isn't in the lookup;
            # only happens when a higher-order entry references a sub-pair
            # not enumerated for that target.
            continue
        delta_rows.append(
            {
                "order": int(row.order),
                "subset": subset,
                "subset_str": ",".join(str(s) for s in subset),
                "target_index": target,
                "target": names[target],
                "ei_joint": float(row.ei),
                "ei_joint_raw": float(row.ei_raw),
                "bias_correction": float(row.bias_correction),
                "delta_K": float(delta),
            }
        )
    hyperedges = pd.DataFrame(delta_rows)

    # Möbius self-check.
    self_check = _run_self_check(hyperedges, ei_lookup, n_samples=int(config.self_check_samples))

    # Block-bootstrap null only for order>=2 (order-1 is the pairwise baseline).
    if not hyperedges.empty:
        higher = hyperedges[hyperedges["order"] >= 2].copy()
        if not higher.empty:
            print(
                f"[peid] null reps={config.null_reps}, candidates={len(higher)}",
                flush=True,
            )
            start = time.time()
            higher_with_null = build_null_for_hyperedges(
                higher,
                intervention_features=intervention_features,
                predictions=predictions,
                n_components=n_components,
                lag=int(config.lag),
                source_mode=str(config.source_mode),
                estimator=estimator,
                jitter=float(config.tm_jitter),
                null_reps=int(config.null_reps),
                block_size=int(config.block_size),
                seed=int(config.seed),
            )
            print(f"[peid] null done in {time.time() - start:.1f}s", flush=True)
            order1_rows = hyperedges[hyperedges["order"] == 1].copy()
            order1_rows["null_mean"] = np.nan
            order1_rows["null_std"] = np.nan
            order1_rows["z"] = np.nan
            order1_rows["p_empirical"] = np.nan
            hyperedges = pd.concat([order1_rows, higher_with_null], ignore_index=True)
        else:
            for col in ("null_mean", "null_std", "z", "p_empirical"):
                hyperedges[col] = np.nan

    # Persist the hyperedge table.
    hyperedges_to_save = hyperedges.copy()
    hyperedges_to_save["subset"] = hyperedges_to_save["subset"].apply(list)
    hyperedges_to_save.to_csv(result_dir / "peid_hyperedges.csv", index=False)

    # Pairwise baseline mediator/gateway (for the aggregation + comparison).
    pairwise_gateway = pd.read_csv(_resolve_path(config.pairwise_gateway_path))
    pairwise_mediator = pd.read_csv(_resolve_path(config.pairwise_mediator_path))

    hyper_gateway = aggregate_hyper_gateway(
        hyperedges,
        n_components=n_components,
        component_names=names,
        significance_z=float(config.delta2_significance_z),
    )
    hyper_mediator = aggregate_hyper_mediator(
        hyperedges,
        n_components=n_components,
        component_names=names,
        significance_z=float(config.delta2_significance_z),
        pairwise_mediator=pairwise_mediator,
    )
    hyper_gateway.to_csv(result_dir / "hyper_gateway_scores.csv", index=False)
    hyper_mediator.to_csv(result_dir / "hyper_mediator_scores.csv", index=False)

    # Comparison metrics.
    comparison = compare_rankings(hyper_gateway, hyper_mediator, pairwise_gateway, pairwise_mediator)
    (result_dir / "ranking_comparison.json").write_text(json.dumps(comparison, indent=2))

    # Plots.
    order_title = f"orders 1-{int(config.order_max)}"
    save_delta2_heatmap(hyperedges, n_components, fig_dir / "delta2_heatmap.png", title="Order-2 Delta_2 (signed, summed over targets)")
    delta3_path = fig_dir / "delta3_top_table.png"
    if int(config.order_max) >= 3:
        save_delta3_top_table(hyperedges, delta3_path, title="Top |Delta_3| order-3 hyperedges")
    elif delta3_path.exists():
        delta3_path.unlink()
    save_hyper_gateway_ranking(hyper_gateway, fig_dir / "hyper_gateway_ranking.png", title=f"PEID hyper-ACE ranking ({order_title})")
    save_hyper_mediator_ranking(hyper_mediator, fig_dir / "hyper_mediator_ranking.png", title=f"PEID hyper-AMCE ranking ({order_title})")
    save_enso_hyperedge_subgraph(
        hyperedges,
        matrix,
        fig_dir / "enso_hyperedge_subgraph.png",
        title="ENSO subgraph with significant order-2 hyperedges",
    )

    manifest = {
        "config": _jsonable_config(config),
        "config_hash": config_hash,
        "model_config_hash": config_hash,
        "component_scores": str(component_scores_path),
        "n_rows": int(len(frame)),
        "n_components": int(n_components),
        "n_lagged_samples": int(len(features)),
        "splits": {key: int(len(value[0])) for key, value in splits.items()},
        "model_cache": str(model_paths[0]),
        "model_caches": [str(path) for path in model_paths],
        "model_cache_reused": bool(cache_reused),
        "ensemble_members": member_summaries,
        "linear_blend": linear_blend,
        "final_train_loss": float(loss_history[-1]) if loss_history else None,
        "dependency_versions": dependency_versions(),
        "candidate_counts": {
            "order_1": int(len(order1_table)),
            "order_2": int(len(order2_table)),
            "order_3": int(len(order3_table)),
        },
        "self_check": self_check,
        "ranking_comparison": comparison,
        "top_hyper_gateways": hyper_gateway.head(10).to_dict("records"),
        "top_hyper_mediators": hyper_mediator.head(10).to_dict("records"),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    summary_lines = _build_summary(manifest, hyper_gateway, hyper_mediator, hyperedges)
    (result_dir / "summary.md").write_text("\n".join(summary_lines) + "\n")

    return {
        "result_dir": result_dir,
        "fig_dir": fig_dir,
        "model_path": model_paths[0],
        "manifest": result_dir / "manifest.json",
    }


def _run_self_check(
    hyperedges: pd.DataFrame,
    ei_lookup: dict[tuple[tuple[int, ...], int], float],
    *,
    n_samples: int,
) -> dict[str, object]:
    """Cross-check closed-form vs recursive Möbius Delta_K on a few candidates."""

    rng = np.random.default_rng(0)
    higher = hyperedges[hyperedges["order"] >= 2].reset_index(drop=True)
    if higher.empty:
        return {"checked": 0, "max_abs_diff": 0.0}
    sample_size = min(int(n_samples), len(higher))
    sample_idx = rng.choice(len(higher), size=sample_size, replace=False)
    diffs: list[float] = []
    for idx in sample_idx:
        row = higher.loc[int(idx)]
        subset = tuple(int(s) for s in row["subset"])
        target = int(row["target_index"])
        try:
            recursive = _mobius_recursive(subset, target, ei_lookup)
        except KeyError:
            continue
        closed = float(row["delta_K"])
        diffs.append(abs(closed - recursive))
    return {"checked": len(diffs), "max_abs_diff": float(max(diffs) if diffs else 0.0)}


def _build_summary(
    manifest: dict[str, object],
    gateway: pd.DataFrame,
    mediator: pd.DataFrame,
    hyperedges: pd.DataFrame,
) -> list[str]:
    config_dict = manifest.get("config", {})
    order_max = int(config_dict.get("order_max", 3))
    lines = [
        "# Runge PEID causal hypergraph summary",
        "",
        "Higher-order PEID extension of the pairwise MLP-TM-EI gateway readout.",
        "Delta_K is computed by Mobius inversion on the joint EI lattice.",
    ]
    if order_max >= 3:
        lines.append("Values at k>=3 are signed and may be negative (high-order anti-coordination).")
    else:
        lines.append("This run is restricted to order 1 and order 2; no order-3 hyperedges are estimated.")
    lines.extend(
        [
            "",
            "## Run",
            "",
            f"- Components: {manifest['n_components']}",
            f"- Lagged samples: {manifest['n_lagged_samples']}",
            f"- Intervention samples: {config_dict.get('intervention_samples')}",
            f"- Order max: {config_dict.get('order_max')}",
            f"- Null reps: {config_dict.get('null_reps')} (block size {config_dict.get('block_size')})",
            f"- MLP cache reused: {manifest['model_cache_reused']}",
            f"- Self-check max |closed - recursive|: {manifest['self_check']['max_abs_diff']:.3e} ({manifest['self_check']['checked']} samples)",
            "",
            "## Hyperedge counts",
            "",
            f"- Order 1: {manifest['candidate_counts']['order_1']}",
            f"- Order 2: {manifest['candidate_counts']['order_2']}",
            f"- Order 3: {manifest['candidate_counts']['order_3']}",
            "",
            "## Top hyper-gateways",
            "",
        ]
    )
    if order_max >= 3:
        lines.extend(
            [
                "| paper_component | hyper_ace_order1 | hyper_ace_order2 | hyper_ace_order3 | hyper_ace_total |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                "| paper_component | hyper_ace_order1 | hyper_ace_order2 | hyper_ace_total |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
    for row in gateway.head(15).itertuples():
        if order_max >= 3:
            lines.append(
                f"| {paper_component_label(row.component_index)} | {float(row.hyper_ace_order1):.6g} | {float(row.hyper_ace_order2):.6g} | {float(row.hyper_ace_order3):.6g} | {float(row.hyper_ace_total):.6g} |"
            )
        else:
            lines.append(
                f"| {paper_component_label(row.component_index)} | {float(row.hyper_ace_order1):.6g} | {float(row.hyper_ace_order2):.6g} | {float(row.hyper_ace_total):.6g} |"
            )
    lines.extend(
        [
            "",
            "## Top hyper-mediators",
            "",
        ]
    )
    if order_max >= 3:
        lines.extend(
            [
                "| paper_component | pairwise_path_amce_diagnostic | synergy_order2 | synergy_order3 | hyper_amce_total |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                "| paper_component | pairwise_path_amce_diagnostic | synergy_order2 | hyper_amce_total |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
    for row in mediator.head(15).itertuples():
        if order_max >= 3:
            lines.append(
                f"| {paper_component_label(row.component_index)} | {float(row.mediator_path_amce):.6g} | {float(row.mediator_synergy_order2):.6g} | {float(row.mediator_synergy_order3):.6g} | {float(row.hyper_amce_total):.6g} |"
            )
        else:
            lines.append(
                f"| {paper_component_label(row.component_index)} | {float(row.mediator_path_amce):.6g} | {float(row.mediator_synergy_order2):.6g} | {float(row.hyper_amce_total):.6g} |"
            )
    lines.extend(
        [
            "",
            "## Ranking comparison vs pairwise baseline",
            "",
            f"- Gateway Spearman: {manifest['ranking_comparison']['gateway_rank_correlation']['spearman']:.4f}",
            f"- Gateway Kendall: {manifest['ranking_comparison']['gateway_rank_correlation']['kendall']:.4f}",
            f"- Mediator Spearman: {manifest['ranking_comparison']['mediator_rank_correlation']['spearman']:.4f}",
            f"- Mediator Kendall: {manifest['ranking_comparison']['mediator_rank_correlation']['kendall']:.4f}",
        ]
    )
    return lines


def parse_args(argv: Sequence[str] | None = None) -> PeidHypergraphConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-scores", type=Path, default=DEFAULT_COMPONENT_SCORES)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--lag", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ensemble-ridge-alphas", default="")
    parser.add_argument("--linear-blend-grid-steps", type=int, default=0)
    parser.add_argument("--train-linear-skip", dest="freeze_linear_skip", action="store_false")
    parser.set_defaults(freeze_linear_skip=True)
    parser.add_argument("--disable-residual-shrinkage", dest="residual_shrinkage", action="store_false")
    parser.set_defaults(residual_shrinkage=True)
    parser.add_argument("--residual-gamma-min", type=float, default=-0.5)
    parser.add_argument("--residual-gamma-max", type=float, default=0.5)
    parser.add_argument("--residual-gamma-steps", type=int, default=101)
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument("--min-delta", type=float, default=1.0e-5)
    parser.add_argument("--scheduler-patience", type=int, default=12)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--intervention-samples", type=int, default=4096)
    parser.add_argument("--quantile-low", type=float, default=0.05)
    parser.add_argument("--quantile-high", type=float, default=0.95)
    parser.add_argument("--source-mode", choices=["latest", "history"], default="latest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--order-max", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--candidate-top-sources", type=int, default=14)
    parser.add_argument("--candidate-target-topk", type=int, default=10)
    parser.add_argument("--candidate-third-source-topk", type=int, default=15)
    parser.add_argument("--delta2-significance-z", type=float, default=2.0)
    parser.add_argument("--null-reps", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=26)
    parser.add_argument("--tm-jitter", type=float, default=1.0e-6)
    parser.add_argument("--pairwise-matrix-path", type=Path, default=PAIRWISE_PATH_DIR / "pairwise_ei_matrix.csv")
    parser.add_argument("--pairwise-gateway-path", type=Path, default=PAIRWISE_PATH_DIR / "gateway_scores.csv")
    parser.add_argument("--pairwise-mediator-path", type=Path, default=PAIRWISE_PATH_DIR / "mediator_scores.csv")
    parser.add_argument("--self-check-samples", type=int, default=5)
    args = parser.parse_args(argv)
    args.ensemble_ridge_alphas = parse_float_tuple(args.ensemble_ridge_alphas)
    return PeidHypergraphConfig(**vars(args))


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    artifacts = run(config)
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
