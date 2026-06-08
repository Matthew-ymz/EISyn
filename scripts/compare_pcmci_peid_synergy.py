#!/usr/bin/env python3
"""Compare pairwise PCMCI graphs with MLP + PEID typed causal graphs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
VARIABLE_NAMES = ("x", "y", "p", "s", "c")
RESULT_DIR = ROOT / "results" / "pcmci_peid_synergy_comparison"
FIGURE_DIR = ROOT / "fig" / "pcmci_peid_synergy_comparison"


@dataclass(frozen=True)
class BenchmarkConfig:
    n_samples: int = 2000
    burn_in: int = 300
    source_noise: float = 0.35
    target_noise: float = 0.08
    seed: int = 0
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    mlp_hidden_dims: tuple[int, ...] = (64, 64)
    mlp_epochs: int = 200
    mlp_patience: int = 25
    mlp_batch_size: int = 128
    mlp_learning_rate: float = 2.0e-3
    mlp_weight_decay: float = 1.0e-5
    intervention_quantile_low: float = 0.05
    intervention_quantile_high: float = 0.95
    intervention_samples: int = 4096
    intervention_batches: int = 5
    peid_permutations: int = 100
    peid_min_score: float = 0.05
    pcmci_tau_max: int = 1
    pcmci_pc_alpha: float = 0.05
    pcmci_cmiknn_knn: float = 0.10
    pcmci_cmiknn_sig_samples: int = 100
    pcmci_cmiknn_workers: int = -1
    q_threshold: float = 0.05

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class FittedTransitionMLP:
    model: Any
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    train_end: int
    val_end: int
    metrics: dict[str, float]
    loss_history: list[float]

    def predict(self, states: np.ndarray) -> np.ndarray:
        import torch

        values = np.asarray(states, dtype=np.float32)
        scaled = (values - self.x_mean) / self.x_std
        self.model.eval()
        with torch.no_grad():
            output = np.asarray(
                self.model(torch.tensor(scaled.tolist(), dtype=torch.float32)).cpu().tolist(),
                dtype=np.float32,
            )
        return output * self.y_std + self.y_mean


def simulate_known_synergy_system(config: BenchmarkConfig) -> tuple[pd.DataFrame, dict[str, list[tuple[str, ...]]]]:
    """Simulate the controlled five-variable first-order dynamical system."""

    if config.n_samples < 20:
        raise ValueError("n_samples must be at least 20.")
    rng = np.random.default_rng(int(config.seed))
    total = int(config.burn_in) + int(config.n_samples)
    values = np.zeros((total, len(VARIABLE_NAMES)), dtype=float)
    values[0] = rng.normal(0.0, 0.2, size=len(VARIABLE_NAMES))
    for t in range(total - 1):
        x, y, p, s, c = values[t]
        values[t + 1, 4] = 0.72 * c + rng.normal(0.0, config.source_noise)
        values[t + 1, 0] = 0.38 * x + 0.68 * c + rng.normal(0.0, config.source_noise)
        values[t + 1, 1] = 0.34 * y + 0.62 * c + rng.normal(0.0, config.source_noise)
        values[t + 1, 2] = 0.20 * p + 0.95 * x + rng.normal(0.0, config.target_noise)
        values[t + 1, 3] = (
            0.20 * s
            + 1.15 * np.tanh(1.60 * x * y)
            + rng.normal(0.0, config.target_noise)
        )
    frame = pd.DataFrame(values[-int(config.n_samples) :], columns=VARIABLE_NAMES)
    truth: dict[str, list[tuple[str, ...]]] = {
        "pairwise": [("c", "x"), ("c", "y"), ("x", "p")],
        "hyperedges": [("x", "y", "s")],
        "self_edges": [(name, name) for name in VARIABLE_NAMES],
    }
    return frame, truth


def oracle_transition(states: np.ndarray) -> np.ndarray:
    """Return the deterministic conditional mean of the known transition."""

    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(VARIABLE_NAMES):
        raise ValueError(f"states must have shape (n, {len(VARIABLE_NAMES)}).")
    x, y, p, s, c = values.T
    output = np.empty_like(values, dtype=float)
    output[:, 4] = 0.72 * c
    output[:, 0] = 0.38 * x + 0.68 * c
    output[:, 1] = 0.34 * y + 0.62 * c
    output[:, 2] = 0.20 * p + 0.95 * x
    output[:, 3] = 0.20 * s + 1.15 * np.tanh(1.60 * x * y)
    return output


def _temporal_split(n_rows: int, config: BenchmarkConfig) -> tuple[int, int]:
    train_end = max(1, min(n_rows - 2, int(round(n_rows * config.train_fraction))))
    val_end = max(train_end + 1, min(n_rows - 1, train_end + int(round(n_rows * config.val_fraction))))
    return train_end, val_end


def fit_transition_mlp(series: pd.DataFrame, config: BenchmarkConfig) -> FittedTransitionMLP:
    """Fit a compact two-layer transition MLP with temporal validation."""

    import torch

    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    torch.set_num_threads(1)
    values = series[list(VARIABLE_NAMES)].to_numpy(dtype=np.float32)
    x = values[:-1]
    y = values[1:]
    train_end, val_end = _temporal_split(len(x), config)
    x_mean = x[:train_end].mean(axis=0, keepdims=True)
    x_std = np.maximum(x[:train_end].std(axis=0, keepdims=True), 1.0e-6)
    y_mean = y[:train_end].mean(axis=0, keepdims=True)
    y_std = np.maximum(y[:train_end].std(axis=0, keepdims=True), 1.0e-6)
    xn = (x - x_mean) / x_std
    yn = (y - y_mean) / y_std

    layers: list[Any] = []
    current = x.shape[1]
    for hidden in config.mlp_hidden_dims:
        layers.extend([torch.nn.Linear(current, int(hidden)), torch.nn.SiLU()])
        current = int(hidden)
    layers.append(torch.nn.Linear(current, y.shape[1]))
    model = torch.nn.Sequential(*layers)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.mlp_learning_rate),
        weight_decay=float(config.mlp_weight_decay),
    )
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(int(config.seed) + 701)
    best_state: dict[str, Any] | None = None
    best_val = float("inf")
    patience_left = int(config.mlp_patience)
    history: list[float] = []

    for _ in range(int(config.mlp_epochs)):
        model.train()
        order = rng.permutation(train_end)
        losses: list[float] = []
        for start in range(0, train_end, int(config.mlp_batch_size)):
            idx = order[start : start + int(config.mlp_batch_size)]
            xb = torch.tensor(xn[idx], dtype=torch.float32)
            yb = torch.tensor(yn[idx], dtype=torch.float32)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        model.eval()
        with torch.no_grad():
            validation = loss_fn(
                model(torch.tensor(xn[train_end:val_end], dtype=torch.float32)),
                torch.tensor(yn[train_end:val_end], dtype=torch.float32),
            )
        val_loss = float(validation)
        if val_loss < best_val - 1.0e-6:
            best_val = val_loss
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            patience_left = int(config.mlp_patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    fitted = FittedTransitionMLP(
        model=model,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        train_end=train_end,
        val_end=val_end,
        metrics={},
        loss_history=history,
    )
    prediction = fitted.predict(x[val_end:])
    persistence = x[val_end:]
    test_target = y[val_end:]
    metrics: dict[str, float] = {}
    for idx, name in enumerate(VARIABLE_NAMES):
        metrics[f"{name}_test_rmse"] = float(np.sqrt(np.mean((prediction[:, idx] - test_target[:, idx]) ** 2)))
        metrics[f"{name}_persistence_rmse"] = float(
            np.sqrt(np.mean((persistence[:, idx] - test_target[:, idx]) ** 2))
        )
    metrics["overall_test_rmse"] = float(np.sqrt(np.mean((prediction - test_target) ** 2)))
    metrics["overall_persistence_rmse"] = float(np.sqrt(np.mean((persistence - test_target) ** 2)))
    fitted.metrics = metrics
    return fitted


def adjust_fdr_bh(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg correction preserving the original order."""

    values = np.asarray(list(p_values), dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=float)
    values = np.where(np.isfinite(values), np.clip(values, 0.0, 1.0), 1.0)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def run_pcmci_variant(series: pd.DataFrame, config: BenchmarkConfig, *, variant: str) -> pd.DataFrame:
    """Run PCMCI with ParCorr or nonlinear CMIknn and return cross-variable lagged edges."""

    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI

    if variant == "parcorr":
        from tigramite.independence_tests.parcorr import ParCorr

        test = ParCorr(significance="analytic")
        method = "PCMCI-ParCorr"
    elif variant == "cmiknn":
        from tigramite.independence_tests.cmiknn import CMIknn

        test = CMIknn(
            knn=float(config.pcmci_cmiknn_knn),
            significance="shuffle_test",
            sig_samples=int(config.pcmci_cmiknn_sig_samples),
            workers=int(config.pcmci_cmiknn_workers),
        )
        method = "PCMCI-CMIknn"
    else:
        raise ValueError("variant must be 'parcorr' or 'cmiknn'.")

    names = list(VARIABLE_NAMES)
    pcmci = PCMCI(
        dataframe=pp.DataFrame(series[names].to_numpy(dtype=float), var_names=names),
        cond_ind_test=test,
        verbosity=0,
    )
    result = pcmci.run_pcmci(
        tau_min=1,
        tau_max=int(config.pcmci_tau_max),
        pc_alpha=float(config.pcmci_pc_alpha),
        alpha_level=float(config.q_threshold),
    )
    p_matrix = np.asarray(result["p_matrix"], dtype=float)
    val_matrix = np.asarray(result["val_matrix"], dtype=float)
    rows: list[dict[str, Any]] = []
    for source_idx, source in enumerate(names):
        for target_idx, target in enumerate(names):
            if source == target:
                continue
            for lag in range(1, int(config.pcmci_tau_max) + 1):
                rows.append(
                    {
                        "method": method,
                        "relation_type": "pairwise",
                        "sources": source,
                        "target": target,
                        "lag": lag,
                        "score": float(abs(val_matrix[source_idx, target_idx, lag])),
                        "signed_score": float(val_matrix[source_idx, target_idx, lag]),
                        "p_value": float(p_matrix[source_idx, target_idx, lag]),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["q_value"] = adjust_fdr_bh(frame["p_value"])
    return frame.sort_values(["q_value", "score"], ascending=[True, False]).reset_index(drop=True)


def _sample_interventions(series: pd.DataFrame, config: BenchmarkConfig, *, batch_index: int) -> np.ndarray:
    rng = np.random.default_rng(int(config.seed) + 1009 + 97 * int(batch_index))
    values = series[list(VARIABLE_NAMES)].to_numpy(dtype=float)
    samples = np.empty((int(config.intervention_samples), len(VARIABLE_NAMES)), dtype=float)
    for index in range(len(VARIABLE_NAMES)):
        low = float(np.quantile(values[:, index], config.intervention_quantile_low))
        high = float(np.quantile(values[:, index], config.intervention_quantile_high))
        samples[:, index] = rng.uniform(low, high, size=int(config.intervention_samples))
    return samples


def _single_tm_ei(source: np.ndarray, target: np.ndarray) -> float:
    from yrd import clip_nonnegative_ei, estimate_mutual_information_transport_map, lift_transport_source_features

    return clip_nonnegative_ei(
        float(estimate_mutual_information_transport_map(lift_transport_source_features(source), target)["mi_hat"])
    )


def _pair_tm_synergy(left: np.ndarray, right: np.ndarray, target: np.ndarray) -> dict[str, float]:
    from yrd import summarize_two_source_synergy_transport_map

    return summarize_two_source_synergy_transport_map(left, right, target)


def estimate_tm_peid_graph(
    predictor: Callable[[np.ndarray], np.ndarray],
    reference_series: pd.DataFrame,
    config: BenchmarkConfig,
    *,
    method: str,
    permutations: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Estimate pairwise EI and all second-order PEID hyperedges."""

    permutation_count = int(config.peid_permutations if permutations is None else permutations)
    batches: list[tuple[np.ndarray, np.ndarray]] = []
    for batch in range(int(config.intervention_batches)):
        source = _sample_interventions(reference_series, config, batch_index=batch)
        batches.append((source, np.asarray(predictor(source), dtype=float)))

    pair_rows: list[dict[str, Any]] = []
    pair_lookup: dict[tuple[int, int], float] = {}
    for source_idx, source_name in enumerate(VARIABLE_NAMES):
        for target_idx, target_name in enumerate(VARIABLE_NAMES):
            estimates = [
                _single_tm_ei(source[:, [source_idx]], target[:, [target_idx]])
                for source, target in batches
            ]
            score = float(np.mean(estimates))
            pair_lookup[(source_idx, target_idx)] = score
            pair_rows.append(
                {
                    "method": method,
                    "relation_type": "pairwise",
                    "sources": source_name,
                    "target": target_name,
                    "lag": 1,
                    "score": score,
                    "batch_std": float(np.std(estimates)),
                }
            )

    hyper_rows: list[dict[str, Any]] = []
    for source_i, source_j in itertools.combinations(range(len(VARIABLE_NAMES)), 2):
        for target_idx, target_name in enumerate(VARIABLE_NAMES):
            estimates = [
                _pair_tm_synergy(
                    source[:, [source_i]],
                    source[:, [source_j]],
                    target[:, [target_idx]],
                )
                for source, target in batches
            ]
            hyper_rows.append(
                {
                    "method": method,
                    "relation_type": "hyperedge",
                    "sources": f"{VARIABLE_NAMES[source_i]}+{VARIABLE_NAMES[source_j]}",
                    "target": target_name,
                    "lag": 1,
                    "score": float(np.mean([row["syn"] for row in estimates])),
                    "joint_ei": float(np.mean([row["joint_ei"] for row in estimates])),
                    "single_ei_sum": float(
                        pair_lookup[(source_i, target_idx)] + pair_lookup[(source_j, target_idx)]
                    ),
                    "batch_std": float(np.std([row["syn"] for row in estimates])),
                    "source_i": int(source_i),
                    "source_j": int(source_j),
                    "target_index": int(target_idx),
                }
            )

    pair_frame = pd.DataFrame(pair_rows)
    hyper_frame = pd.DataFrame(hyper_rows)
    pair_frame["p_value"] = 1.0
    pair_frame["empirical_p_value"] = 1.0
    hyper_frame["p_value"] = 1.0
    hyper_frame["empirical_p_value"] = 1.0
    if permutation_count > 0:
        from scipy.stats import norm

        pooled_source = np.concatenate([source for source, _ in batches], axis=0)
        pooled_target = np.concatenate([target for _, target in batches], axis=0)
        rng = np.random.default_rng(int(config.seed) + 4049)

        def null_p_values(observed: float, null_values: Sequence[float]) -> tuple[float, float]:
            null = np.asarray(null_values, dtype=float)
            empirical = float((1 + np.sum(null >= observed)) / (1 + len(null)))
            std = float(np.std(null, ddof=1)) if len(null) > 1 else 0.0
            if std <= 1.0e-12:
                parametric = 0.0 if observed > float(np.mean(null)) else 1.0
            else:
                parametric = float(norm.sf((observed - float(np.mean(null))) / std))
            return parametric, empirical

        for row_idx, row in pair_frame.iterrows():
            if float(row["score"]) < float(config.peid_min_score) or str(row["sources"]) == str(row["target"]):
                continue
            source_idx = VARIABLE_NAMES.index(str(row["sources"]))
            target_idx = VARIABLE_NAMES.index(str(row["target"]))
            observed = float(row["score"])
            null = [
                _single_tm_ei(
                    pooled_source[:, [source_idx]],
                    pooled_target[rng.permutation(len(pooled_target))][:, [target_idx]],
                )
                for _ in range(permutation_count)
            ]
            p_value, empirical = null_p_values(observed, null)
            pair_frame.loc[row_idx, ["p_value", "empirical_p_value"]] = [p_value, empirical]
        for row_idx, row in hyper_frame.iterrows():
            if (
                float(row["score"]) < float(config.peid_min_score)
                or str(row["target"]) in str(row["sources"]).split("+")
            ):
                continue
            source_i = int(row["source_i"])
            source_j = int(row["source_j"])
            target_idx = int(row["target_index"])
            observed = float(row["score"])
            null = [
                _pair_tm_synergy(
                    pooled_source[:, [source_i]],
                    pooled_source[:, [source_j]],
                    pooled_target[rng.permutation(len(pooled_target))][:, [target_idx]],
                )["syn"]
                for _ in range(permutation_count)
            ]
            p_value, empirical = null_p_values(observed, null)
            hyper_frame.loc[row_idx, ["p_value", "empirical_p_value"]] = [p_value, empirical]
    combined_q = adjust_fdr_bh(pd.concat([pair_frame["p_value"], hyper_frame["p_value"]], ignore_index=True))
    pair_frame["q_value"] = combined_q[: len(pair_frame)]
    hyper_frame["q_value"] = combined_q[len(pair_frame) :]
    return {
        "pairwise": pair_frame.sort_values("score", ascending=False).reset_index(drop=True),
        "hyperedges": hyper_frame.sort_values("score", ascending=False).reset_index(drop=True),
    }


def _relation_key(row: dict[str, Any] | pd.Series) -> tuple[str, tuple[str, ...], str]:
    relation_type = str(row["relation_type"])
    sources = tuple(sorted(str(row["sources"]).split("+")))
    return relation_type, sources, str(row["target"])


def score_typed_graph(
    predicted_relations: Sequence[dict[str, Any]] | pd.DataFrame,
    truth: dict[str, list[tuple[str, ...]]],
) -> dict[str, float]:
    """Score ordinary edges and hyperedges in one typed relation space."""

    records = predicted_relations.to_dict("records") if isinstance(predicted_relations, pd.DataFrame) else list(predicted_relations)
    predicted = {_relation_key(row) for row in records if str(row["sources"]) != str(row["target"])}
    truth_pairwise = {("pairwise", (source,), target) for source, target in truth["pairwise"]}
    truth_hyper = {
        ("hyperedge", tuple(sorted((source_i, source_j))), target)
        for source_i, source_j, target in truth["hyperedges"]
    }
    truth_all = truth_pairwise | truth_hyper

    def metrics(pred: set[tuple[str, tuple[str, ...], str]], actual: set[tuple[str, tuple[str, ...], str]]) -> tuple[float, float, float]:
        tp = len(pred & actual)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(actual) if actual else 0.0
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        return float(precision), float(recall), float(f1)

    pair_pred = {key for key in predicted if key[0] == "pairwise"}
    hyper_pred = {key for key in predicted if key[0] == "hyperedge"}
    pair_precision, pair_recall, pair_f1 = metrics(pair_pred, truth_pairwise)
    hyper_precision, hyper_recall, hyper_f1 = metrics(hyper_pred, truth_hyper)
    typed_precision, typed_recall, typed_f1 = metrics(predicted, truth_all)
    return {
        "pairwise_precision": pair_precision,
        "pairwise_recall": pair_recall,
        "pairwise_f1": pair_f1,
        "hyperedge_precision": hyper_precision,
        "hyperedge_recall": hyper_recall,
        "hyperedge_f1": hyper_f1,
        "typed_precision": typed_precision,
        "typed_recall": typed_recall,
        "typed_f1": typed_f1,
    }


def save_cached_json(path: Path | str, payload: dict[str, Any], config: BenchmarkConfig) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"config_hash": config.config_hash(), "payload": payload}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def load_cached_json(path: Path | str, config: BenchmarkConfig) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    raw = json.loads(candidate.read_text(encoding="utf-8"))
    if raw.get("config_hash") != config.config_hash():
        return None
    return dict(raw["payload"])


def select_significant_relations(
    relations: Sequence[dict[str, Any]] | pd.DataFrame,
    config: BenchmarkConfig,
) -> list[dict[str, Any]]:
    """Select cross-variable relations used by the main typed-graph score."""

    records = relations.to_dict("records") if isinstance(relations, pd.DataFrame) else list(relations)
    selected: list[dict[str, Any]] = []
    for row in records:
        source_parts = str(row["sources"]).split("+")
        target = str(row["target"])
        if target in source_parts:
            continue
        if float(row.get("q_value", 1.0)) > float(config.q_threshold):
            continue
        if "PEID" in str(row.get("method", "")) and float(row.get("score", 0.0)) < float(config.peid_min_score):
            continue
        selected.append(dict(row))
    return selected


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _relation_records(graph: pd.DataFrame | dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    if isinstance(graph, pd.DataFrame):
        return graph.to_dict("records")
    return pd.concat([graph["pairwise"], graph["hyperedges"]], ignore_index=True).to_dict("records")


def _decomposition_rows(method: str, graph: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    pair = graph["pairwise"].set_index(["sources", "target"])
    hyper = graph["hyperedges"].set_index(["sources", "target"])
    rows: list[dict[str, Any]] = []
    for target in ("p", "s"):
        relation = hyper.loc[("x+y", target)]
        rows.append(
            {
                "method": method,
                "target": target,
                "x_ei": float(pair.loc[("x", target), "score"]),
                "y_ei": float(pair.loc[("y", target), "score"]),
                "joint_ei": float(relation["joint_ei"]),
                "synergy": float(relation["score"]),
                "q_value": float(relation["q_value"]),
            }
        )
    return rows


def run_single_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    """Run all four estimators for one simulation configuration."""

    series, truth = simulate_known_synergy_system(config)
    fitted = fit_transition_mlp(series, config)
    training_series = series.iloc[: fitted.train_end + 1]
    pcmci_parcorr = run_pcmci_variant(series, config, variant="parcorr")
    pcmci_cmiknn = run_pcmci_variant(series, config, variant="cmiknn")
    mlp_peid = estimate_tm_peid_graph(fitted.predict, training_series, config, method="MLP + PEID")
    oracle_peid = estimate_tm_peid_graph(oracle_transition, training_series, config, method="Oracle + PEID")

    graphs: dict[str, pd.DataFrame | dict[str, pd.DataFrame]] = {
        "PCMCI-ParCorr": pcmci_parcorr,
        "PCMCI-CMIknn": pcmci_cmiknn,
        "MLP + PEID": mlp_peid,
        "Oracle + PEID": oracle_peid,
    }
    relation_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for method, graph in graphs.items():
        records = _relation_records(graph)
        selected = select_significant_relations(records, config)
        selected_keys = {_relation_key(row) for row in selected}
        for row in records:
            relation_rows.append({**row, "selected": _relation_key(row) in selected_keys})
        metric_rows.append({"method": method, **score_typed_graph(selected, truth)})

    return {
        "config": asdict(config),
        "truth": truth,
        "model_metrics": fitted.metrics,
        "relations": relation_rows,
        "metrics": metric_rows,
        "decomposition": _decomposition_rows("MLP + PEID", mlp_peid)
        + _decomposition_rows("Oracle + PEID", oracle_peid),
    }


def _run_cache_path(result_dir: Path, config: BenchmarkConfig) -> Path:
    return result_dir / "runs" / f"run_{config.config_hash()}.json"


def _load_or_run_single(result_dir: Path, config: BenchmarkConfig, *, force: bool) -> dict[str, Any]:
    cache_path = _run_cache_path(result_dir, config)
    if not force:
        cached = load_cached_json(cache_path, config)
        if cached is not None:
            return cached
    payload = _jsonable(run_single_benchmark(config))
    save_cached_json(cache_path, payload, config)
    return payload


def paired_bootstrap_ci(values: Sequence[float], *, seed: int = 7301, reps: int = 2000) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(int(seed))
    means = np.asarray([np.mean(rng.choice(array, size=len(array), replace=True)) for _ in range(int(reps))])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _acceptance_summary(
    primary_metrics: list[dict[str, Any]],
    primary_decomposition: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = pd.DataFrame(primary_metrics)
    decomposition = pd.DataFrame(primary_decomposition)
    mlp = metrics[metrics["method"] == "MLP + PEID"].sort_values("seed")
    cmi = metrics[metrics["method"] == "PCMCI-CMIknn"].sort_values("seed")
    paired = mlp.merge(cmi, on="seed", suffixes=("_mlp", "_cmi"))
    differences = (paired["typed_f1_mlp"] - paired["typed_f1_cmi"]).to_numpy(dtype=float)
    ci_low, ci_high = paired_bootstrap_ci(differences)
    mlp_decomp = decomposition[decomposition["method"] == "MLP + PEID"]
    pivot = mlp_decomp.pivot(index="seed", columns="target", values="synergy")
    ratios = pivot["s"] / np.maximum(np.abs(pivot["p"]), 1.0e-6) if {"p", "s"}.issubset(pivot.columns) else pd.Series(dtype=float)
    detected = int(np.sum(mlp["hyperedge_recall"].to_numpy(dtype=float) >= 1.0))
    return {
        "mlp_true_hyperedge_detected_count": detected,
        "mlp_primary_seed_count": int(len(mlp)),
        "mlp_detected_at_least_4_of_5": bool(detected >= 4) if len(mlp) >= 5 else None,
        "mean_synergy_ratio_s_over_p": float(np.mean(ratios)) if len(ratios) else float("nan"),
        "synergy_ratio_exceeds_10": bool(np.mean(ratios) > 10.0) if len(ratios) else False,
        "pcmci_parcorr_mean_pairwise_recall": float(
            metrics.loc[metrics["method"] == "PCMCI-ParCorr", "pairwise_recall"].mean()
        ),
        "typed_f1_mlp_minus_cmiknn_mean": float(np.mean(differences)) if len(differences) else float("nan"),
        "typed_f1_mlp_minus_cmiknn_bootstrap_ci95": [ci_low, ci_high],
        "typed_f1_advantage_ci_above_zero": bool(ci_low > 0.0) if np.isfinite(ci_low) else False,
    }


def _set_plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _draw_typed_graph(ax: Any, relations: Sequence[dict[str, Any]], *, title: str) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, FancyArrowPatch

    positions = {
        "c": (-1.7, 0.0),
        "x": (-0.5, 0.9),
        "y": (-0.5, -0.9),
        "p": (1.3, 0.9),
        "s": (1.3, -0.9),
    }
    for name, (x_pos, y_pos) in positions.items():
        ax.add_patch(Circle((x_pos, y_pos), 0.20, facecolor="#f5f5f2", edgecolor="#333333", lw=1.0))
        ax.text(x_pos, y_pos, name, ha="center", va="center", fontsize=9)
    for row in relations:
        sources = str(row["sources"]).split("+")
        target = str(row["target"])
        if len(sources) == 1:
            ax.add_patch(
                FancyArrowPatch(
                    positions[sources[0]],
                    positions[target],
                    arrowstyle="-|>",
                    mutation_scale=9,
                    color="#4c78a8",
                    linewidth=1.4,
                    shrinkA=16,
                    shrinkB=16,
                    connectionstyle="arc3,rad=0.05",
                )
            )
        else:
            hub = (0.40, -0.35 if target == "s" else 0.35)
            for source in sources:
                ax.plot(
                    [positions[source][0], hub[0]],
                    [positions[source][1], hub[1]],
                    color="#d97732",
                    lw=1.5,
                )
            ax.add_patch(
                FancyArrowPatch(
                    hub,
                    positions[target],
                    arrowstyle="-|>",
                    mutation_scale=9,
                    color="#d97732",
                    linewidth=1.6,
                    shrinkB=16,
                )
            )
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-2.15, 1.75)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax._typed_graph_legend = [
        Line2D([0], [0], color="#4c78a8", lw=1.5, label="Pairwise edge"),
        Line2D([0], [0], color="#d97732", lw=1.5, label="Synergy hyperedge"),
    ]


def plot_graph_comparison(summary: dict[str, Any], output_path: Path) -> Path:
    import matplotlib.pyplot as plt

    _set_plot_style()
    representative = summary["representative"]
    truth = representative["truth"]
    truth_rows = [
        {"relation_type": "pairwise", "sources": source, "target": target}
        for source, target in truth["pairwise"]
    ] + [
        {"relation_type": "hyperedge", "sources": f"{left}+{right}", "target": target}
        for left, right, target in truth["hyperedges"]
    ]
    relation_frame = pd.DataFrame(representative["relations"])
    panels: list[tuple[str, list[dict[str, Any]]]] = [("Ground truth", truth_rows)]
    for method in ("PCMCI-ParCorr", "PCMCI-CMIknn", "MLP + PEID"):
        rows = relation_frame[(relation_frame["method"] == method) & relation_frame["selected"].astype(bool)]
        panels.append((method, rows.to_dict("records")))
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 2.7), constrained_layout=True)
    for ax, (title, rows) in zip(axes, panels):
        _draw_typed_graph(ax, rows, title=title)
    axes[-1].legend(
        handles=axes[-1]._typed_graph_legend,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_decomposition(summary: dict[str, Any], output_path: Path) -> Path:
    import matplotlib.pyplot as plt

    _set_plot_style()
    frame = pd.DataFrame(summary["primary_decomposition"])
    frame = frame.groupby(["method", "target"], as_index=False)[["x_ei", "y_ei", "joint_ei", "synergy"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), constrained_layout=True, sharey=True)
    colors = {"x_ei": "#4c78a8", "y_ei": "#72a0c1", "joint_ei": "#d9a441", "synergy": "#d97732"}
    labels = {"x_ei": "EI(x)", "y_ei": "EI(y)", "joint_ei": "Joint EI", "synergy": "Synergy"}
    for ax, target in zip(axes, ("p", "s")):
        target_frame = frame[frame["target"] == target]
        methods = list(target_frame["method"])
        x_pos = np.arange(len(methods))
        width = 0.18
        for offset, column in enumerate(("x_ei", "y_ei", "joint_ei", "synergy")):
            ax.bar(
                x_pos + (offset - 1.5) * width,
                target_frame[column],
                width,
                color=colors[column],
                label=labels[column],
            )
        ax.set_xticks(x_pos, methods, rotation=20, ha="right")
        ax.set_title(f"Target {target}")
        ax.set_ylabel("Information (nats)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_recovery_metrics(summary: dict[str, Any], output_path: Path) -> Path:
    import matplotlib.pyplot as plt

    _set_plot_style()
    frame = pd.DataFrame(summary["primary_metrics"])
    methods = ["PCMCI-ParCorr", "PCMCI-CMIknn", "MLP + PEID", "Oracle + PEID"]
    columns = [
        ("pairwise_f1", "Pairwise F1"),
        ("hyperedge_recall", "Hyperedge recall"),
        ("typed_f1", "Typed structure F1"),
    ]
    colors = ["#4c78a8", "#72a0c1", "#d97732", "#5c9e6e"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), constrained_layout=True, sharey=True)
    for ax, (column, label) in zip(axes, columns):
        means = [float(frame.loc[frame["method"] == method, column].mean()) for method in methods]
        stds = [float(frame.loc[frame["method"] == method, column].std(ddof=0)) for method in methods]
        x_pos = np.arange(len(methods))
        ax.errorbar(x_pos, means, yerr=stds, fmt="none", ecolor="#555555", capsize=3, lw=0.8)
        ax.scatter(x_pos, means, c=colors, s=32, zorder=3)
        ax.set_xticks(x_pos, methods, rotation=28, ha="right")
        ax.set_title(label)
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("Score")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_robustness(summary: dict[str, Any], output_path: Path) -> Path:
    import matplotlib.pyplot as plt

    _set_plot_style()
    frame = pd.DataFrame(summary["robustness_metrics"])
    methods = ["PCMCI-ParCorr", "PCMCI-CMIknn", "MLP + PEID"]
    colors = {"PCMCI-ParCorr": "#4c78a8", "PCMCI-CMIknn": "#72a0c1", "MLP + PEID": "#d97732"}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), constrained_layout=True, sharey=True)
    specifications = [("sample_size", "n_samples", "Sample size"), ("target_noise", "target_noise", "Target noise")]
    if frame.empty:
        for ax, (_, _, x_label) in zip(axes, specifications):
            ax.text(0.5, 0.5, "No sweep configured", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel(x_label)
            ax.set_ylim(-0.03, 1.05)
        axes[0].set_ylabel("Typed structure F1")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return output_path
    for ax, (setting, x_column, x_label) in zip(axes, specifications):
        subset = frame[frame["setting_kind"] == setting]
        for method in methods:
            values = subset[subset["method"] == method]
            if values.empty:
                continue
            grouped = values.groupby(x_column)["typed_f1"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                grouped[x_column],
                grouped["mean"],
                yerr=grouped["std"].fillna(0.0),
                marker="o",
                color=colors[method],
                label=method,
                capsize=3,
            )
        ax.set_xlabel(x_label)
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("Typed structure F1")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_benchmark(
    *,
    mode: str = "smoke",
    base_config: BenchmarkConfig | None = None,
    primary_seeds: Sequence[int] | None = None,
    sample_sizes: Sequence[int] | None = None,
    noise_values: Sequence[float] | None = None,
    scan_seeds: Sequence[int] | None = None,
    result_dir: Path | str = RESULT_DIR,
    figure_dir: Path | str = FIGURE_DIR,
    force: bool = False,
) -> dict[str, Any]:
    """Run the primary comparison, robustness sweeps, caching, and figure export."""

    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    if base_config is None:
        base_config = BenchmarkConfig() if mode == "full" else BenchmarkConfig(
            n_samples=700,
            burn_in=150,
            mlp_epochs=80,
            mlp_patience=12,
            intervention_samples=800,
            intervention_batches=2,
            peid_permutations=20,
            pcmci_cmiknn_sig_samples=20,
        )
    primary_seeds = tuple(primary_seeds if primary_seeds is not None else (range(5) if mode == "full" else (0,)))
    sample_sizes = tuple(sample_sizes if sample_sizes is not None else ((500, 1000, 2000) if mode == "full" else (500, 1000)))
    noise_values = tuple(noise_values if noise_values is not None else ((0.04, 0.08, 0.16) if mode == "full" else (0.04, 0.16)))
    scan_seeds = tuple(scan_seeds if scan_seeds is not None else (range(3) if mode == "full" else (0,)))
    result_path = Path(result_dir)
    figure_path = Path(figure_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []

    def execute(config: BenchmarkConfig, *, setting_kind: str, setting_value: float) -> None:
        payload = _load_or_run_single(result_path, config, force=force)
        payload = dict(payload)
        payload["setting_kind"] = setting_kind
        payload["setting_value"] = setting_value
        payload["seed"] = int(config.seed)
        runs.append(payload)

    for seed in primary_seeds:
        execute(
            BenchmarkConfig(**{**asdict(base_config), "seed": int(seed)}),
            setting_kind="primary",
            setting_value=float(base_config.n_samples),
        )
    for sample_size in sample_sizes:
        for seed in scan_seeds:
            execute(
                BenchmarkConfig(**{**asdict(base_config), "n_samples": int(sample_size), "seed": int(seed)}),
                setting_kind="sample_size",
                setting_value=float(sample_size),
            )
    for noise in noise_values:
        for seed in scan_seeds:
            execute(
                BenchmarkConfig(**{**asdict(base_config), "target_noise": float(noise), "seed": int(seed)}),
                setting_kind="target_noise",
                setting_value=float(noise),
            )

    metric_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    for run in runs:
        common = {
            "setting_kind": run["setting_kind"],
            "setting_value": run["setting_value"],
            "seed": run["seed"],
            "n_samples": int(run["config"]["n_samples"]),
            "target_noise": float(run["config"]["target_noise"]),
        }
        metric_rows.extend([{**row, **common} for row in run["metrics"]])
        decomposition_rows.extend([{**row, **common} for row in run["decomposition"]])
        relation_rows.extend([{**row, **common} for row in run["relations"]])

    primary_metrics = [row for row in metric_rows if row["setting_kind"] == "primary"]
    primary_decomposition = [row for row in decomposition_rows if row["setting_kind"] == "primary"]
    summary = {
        "mode": mode,
        "base_config": asdict(base_config),
        "primary_metrics": primary_metrics,
        "primary_decomposition": primary_decomposition,
        "robustness_metrics": [row for row in metric_rows if row["setting_kind"] != "primary"],
        "acceptance": _acceptance_summary(primary_metrics, primary_decomposition),
        "representative": next(run for run in runs if run["setting_kind"] == "primary"),
    }
    summary_path = result_path / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    relation_path = result_path / "relations.jsonl"
    relation_path.write_text(
        "".join(json.dumps(_jsonable(row), ensure_ascii=False) + "\n" for row in relation_rows),
        encoding="utf-8",
    )
    metric_path = result_path / "metrics.jsonl"
    metric_path.write_text(
        "".join(json.dumps(_jsonable(row), ensure_ascii=False) + "\n" for row in metric_rows),
        encoding="utf-8",
    )
    figure_paths = {
        "graph_comparison": str(plot_graph_comparison(summary, figure_path / "typed_graph_comparison.png")),
        "decomposition": str(plot_decomposition(summary, figure_path / "peid_decomposition.png")),
        "recovery_metrics": str(plot_recovery_metrics(summary, figure_path / "recovery_metrics.png")),
        "robustness": str(plot_robustness(summary, figure_path / "robustness_curves.png")),
    }
    return {
        "summary_path": str(summary_path),
        "relations_path": str(relation_path),
        "metrics_path": str(metric_path),
        "figure_paths": figure_paths,
        "summary": summary,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = run_benchmark(mode=args.mode, force=bool(args.force))
    print(json.dumps(_jsonable(output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
