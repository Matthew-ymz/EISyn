#!/usr/bin/env python3
"""Benchmark causal readouts on classical network dynamical systems."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_RESULT_DIR = ROOT / "results" / "classic_network_dynamics_benchmark"
DEFAULT_FIGURE_DIR = ROOT / "fig" / "classic_network_dynamics_benchmark"
DEFAULT_REPORT_PATH = ROOT / "docs" / "reports" / "granger_peid_mlp_comparison.md"
BENCHMARK_MODEL_NAMES = ("kuramoto", "coupled_rossler", "sis", "wilson_cowan")
LEGACY_MARKER = "## 附录：原共同驱动 sine 基准"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    state_names: tuple[str, ...]
    target_names: tuple[str, ...]
    equation: str
    dt: float
    warmup_steps: int
    intervention_bounds: np.ndarray
    truth_pairwise: tuple[tuple[str, str], ...]
    truth_hyperedges: tuple[tuple[str, str, str], ...]
    _vector_field: Callable[[np.ndarray], np.ndarray]
    _initial_state: Callable[[np.random.Generator], np.ndarray]
    clip_bounds: tuple[float, float] | None = None

    def vector_field(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(self.state_names):
            raise ValueError(f"states must have shape (n, {len(self.state_names)}).")
        result = np.asarray(self._vector_field(values), dtype=float)
        if result.shape != values.shape:
            raise ValueError("vector_field returned an invalid shape.")
        return result

    def simulate(self, *, seed: int, samples: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
        if samples < 2:
            raise ValueError("samples must be at least 2.")
        if noise < 0.0:
            raise ValueError("noise must be nonnegative.")
        rng = np.random.default_rng(int(seed))
        state = np.asarray(self._initial_state(rng), dtype=float).reshape(-1)
        rows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        total = int(self.warmup_steps) + int(samples)
        for step in range(total):
            if step >= self.warmup_steps:
                rows.append(state.copy())
                derivative = self.vector_field(state)[0]
                targets.append(derivative + rng.normal(0.0, noise, size=len(state)))
            state = _rk4_step(self.vector_field, state, self.dt)
            if self.name == "kuramoto":
                state = (state + np.pi) % (2.0 * np.pi) - np.pi
            if self.clip_bounds is not None:
                state = np.clip(state, self.clip_bounds[0], self.clip_bounds[1])
            if noise > 0.0:
                state = state + rng.normal(0.0, noise * math.sqrt(self.dt), size=len(state))
                if self.clip_bounds is not None:
                    state = np.clip(state, self.clip_bounds[0], self.clip_bounds[1])
            if not np.isfinite(state).all():
                raise FloatingPointError(f"{self.name} simulation diverged.")
        return np.asarray(rows), np.asarray(targets)


def _rk4_step(field: Callable[[np.ndarray], np.ndarray], state: np.ndarray, dt: float) -> np.ndarray:
    y = np.asarray(state, dtype=float)
    k1 = field(y)[0]
    k2 = field(y + 0.5 * dt * k1)[0]
    k3 = field(y + 0.5 * dt * k2)[0]
    k4 = field(y + dt * k3)[0]
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def build_model_specs() -> dict[str, ModelSpec]:
    def kuramoto_field(values: np.ndarray) -> np.ndarray:
        x, y, w = values.T
        return np.column_stack(
            [
                1.0 + 0.2 * np.sin(w - x),
                1.1 + 0.2 * np.sin(w - y),
                np.full(len(values), 0.9),
            ]
        )

    def rossler_field(values: np.ndarray) -> np.ndarray:
        x0, y0, z0, x1, y1, z1 = values.T
        return np.column_stack(
            [
                -y0 - z0 + 0.5 * np.sin(x1 - x0),
                x0 + 0.165 * y0,
                2.0 + z0 * (x0 - 5.5),
                -y1 - z1 + 0.5 * np.sin(x0 - x1),
                x1 + 0.165 * y1,
                2.0 + z1 * (x1 - 5.5),
            ]
        )

    def sis_field(values: np.ndarray) -> np.ndarray:
        w, x, y = values.T
        return np.column_stack(
            [
                -0.8 * w + w * (1.0 - w),
                -1.0 * x + w * (1.0 - x),
                -1.2 * y + w * (1.0 - y),
            ]
        )

    def sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-5.1 * (values - 1.0)))

    def wilson_field(values: np.ndarray) -> np.ndarray:
        w, x, y = values.T
        drive = sigmoid(w)
        return np.column_stack([-w + drive, -x + drive, -y + drive])

    return {
        "kuramoto": ModelSpec(
            name="kuramoto",
            display_name="Kuramoto",
            state_names=("x", "y", "w"),
            target_names=("dx", "dy", "dw"),
            equation=r"\dot{x}_i=\omega_i+0.2\sum_j A_{ij}\sin(x_j-x_i)",
            dt=0.02,
            warmup_steps=400,
            intervention_bounds=np.array([[-np.pi, np.pi]] * 3),
            truth_pairwise=(("w", "dx"), ("w", "dy")),
            truth_hyperedges=(("w", "x", "dx"), ("w", "y", "dy")),
            _vector_field=kuramoto_field,
            _initial_state=lambda rng: rng.uniform(-np.pi, np.pi, size=3),
        ),
        "coupled_rossler": ModelSpec(
            name="coupled_rossler",
            display_name="Coupled Rössler",
            state_names=("x0", "y0", "z0", "x1", "y1", "z1"),
            target_names=("dx0", "dy0", "dz0", "dx1", "dy1", "dz1"),
            equation=(
                r"\dot{x}_i=-y_i-z_i+0.5\sum_jA_{ij}\sin(x_j-x_i),\;"
                r"\dot{y}_i=x_i+0.165y_i,\;\dot{z}_i=2+z_i(x_i-5.5)"
            ),
            dt=0.01,
            warmup_steps=2000,
            intervention_bounds=np.array(
                [[-7.0, 7.0], [-7.0, 7.0], [0.1, 9.0], [-7.0, 7.0], [-7.0, 7.0], [0.1, 9.0]]
            ),
            truth_pairwise=(("x0", "dy0"), ("x0", "dz0"), ("x1", "dy1"), ("x1", "dz1")),
            truth_hyperedges=(
                ("x0", "z0", "dz0"),
                ("x1", "z1", "dz1"),
                ("x0", "x1", "dx0"),
                ("x0", "x1", "dx1"),
            ),
            _vector_field=rossler_field,
            _initial_state=lambda rng: np.array(
                [rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0.1, 0.9),
                 rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0.1, 0.9)]
            ),
        ),
        "sis": ModelSpec(
            name="sis",
            display_name="SIS",
            state_names=("w", "x", "y"),
            target_names=("dw", "dx", "dy"),
            equation=r"\dot{x}_i=-\delta_i x_i+\sum_jA_{ij}x_j(1-x_i)",
            dt=0.02,
            warmup_steps=300,
            intervention_bounds=np.array([[0.02, 0.98]] * 3),
            truth_pairwise=(("w", "dx"), ("w", "dy")),
            truth_hyperedges=(("w", "x", "dx"), ("w", "y", "dy")),
            _vector_field=sis_field,
            _initial_state=lambda rng: rng.uniform(0.15, 0.75, size=3),
            clip_bounds=(0.0, 1.0),
        ),
        "wilson_cowan": ModelSpec(
            name="wilson_cowan",
            display_name="Wilson–Cowan",
            state_names=("w", "x", "y"),
            target_names=("dw", "dx", "dy"),
            equation=r"\dot{x}_i=-x_i+\sum_jA_{ij}[1+e^{-5.1(x_j-1)}]^{-1}",
            dt=0.02,
            warmup_steps=300,
            intervention_bounds=np.array([[0.0, 2.0]] * 3),
            truth_pairwise=(("w", "dx"), ("w", "dy")),
            truth_hyperedges=(),
            _vector_field=wilson_field,
            _initial_state=lambda rng: rng.uniform(0.0, 1.8, size=3),
            clip_bounds=(0.0, 5.0),
        ),
    }


def _entropy(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float)
    probs = probs[probs > 0.0]
    return float(-(probs * np.log2(probs)).sum()) if len(probs) else 0.0


def _discretize(values: np.ndarray, bins: int = 6) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    edges = np.unique(np.quantile(vector, np.linspace(0.0, 1.0, int(bins) + 1)))
    if len(edges) <= 2:
        return np.zeros(len(vector), dtype=int)
    return np.digitize(vector, edges[1:-1], right=False).astype(int)


def _mi_discrete(sources: np.ndarray, target: np.ndarray) -> float:
    source = np.asarray(sources, dtype=int)
    if source.ndim == 1:
        source = source.reshape(-1, 1)
    _, source_codes = np.unique(source, axis=0, return_inverse=True)
    target_codes = np.asarray(target, dtype=int).reshape(-1)
    _, target_codes = np.unique(target_codes, return_inverse=True)
    _, joint_codes = np.unique(np.column_stack([source_codes, target_codes]), axis=0, return_inverse=True)

    def entropy_codes(codes: np.ndarray) -> float:
        counts = np.bincount(codes)
        return _entropy(counts / counts.sum())

    return entropy_codes(source_codes) + entropy_codes(target_codes) - entropy_codes(joint_codes)


def _histogram_synergy(left: np.ndarray, right: np.ndarray, target: np.ndarray, bins: int) -> dict[str, float]:
    a = _discretize(left, bins)
    b = _discretize(right, bins)
    t = _discretize(target, bins)
    left_ei = _mi_discrete(a, t)
    right_ei = _mi_discrete(b, t)
    joint_ei = _mi_discrete(np.column_stack([a, b]), t)
    return {"left_ei": left_ei, "right_ei": right_ei, "joint_ei": joint_ei, "syn": joint_ei - left_ei - right_ei}


def _transport_synergy(left: np.ndarray, right: np.ndarray, target: np.ndarray) -> dict[str, float]:
    from yrd import summarize_two_source_synergy_transport_map

    result = summarize_two_source_synergy_transport_map(left, right, target)
    return {key: float(result[key]) for key in ("left_ei", "right_ei", "joint_ei", "syn") if key in result}


def _transport_single(source: np.ndarray, target: np.ndarray) -> float:
    from yrd import clip_nonnegative_ei, estimate_mutual_information_transport_map, lift_transport_source_features

    estimate = estimate_mutual_information_transport_map(lift_transport_source_features(source), target)
    return float(clip_nonnegative_ei(float(estimate["mi_hat"])))


def estimate_peid(
    spec: ModelSpec,
    predictor: Callable[[np.ndarray], np.ndarray],
    *,
    samples: int,
    seed: int,
    estimator: str,
    bins: int = 6,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    interventions = np.column_stack(
        [rng.uniform(low, high, size=int(samples)) for low, high in spec.intervention_bounds]
    )
    targets = np.asarray(predictor(interventions), dtype=float)
    pair_rows: list[dict[str, object]] = []
    pair_lookup: dict[tuple[int, int], float] = {}
    for source_idx, source in enumerate(spec.state_names):
        for target_idx, target in enumerate(spec.target_names):
            if estimator == "transport":
                score = _transport_single(interventions[:, [source_idx]], targets[:, [target_idx]])
            elif estimator == "histogram":
                score = _mi_discrete(_discretize(interventions[:, source_idx], bins), _discretize(targets[:, target_idx], bins))
            else:
                raise ValueError("estimator must be 'histogram' or 'transport'.")
            pair_lookup[(source_idx, target_idx)] = float(score)
            pair_rows.append({"source": source, "target": target, "score": float(score)})

    hyper_rows: list[dict[str, object]] = []
    for left_idx, right_idx in combinations(range(len(spec.state_names)), 2):
        for target_idx, target in enumerate(spec.target_names):
            if estimator == "transport":
                values = _transport_synergy(
                    interventions[:, [left_idx]], interventions[:, [right_idx]], targets[:, [target_idx]]
                )
            else:
                values = _histogram_synergy(
                    interventions[:, left_idx], interventions[:, right_idx], targets[:, target_idx], bins
                )
            hyper_rows.append(
                {
                    "sources": "+".join(sorted((spec.state_names[left_idx], spec.state_names[right_idx]))),
                    "target": target,
                    "score": float(values["syn"]),
                    "joint_ei": float(values["joint_ei"]),
                    "single_ei_sum": float(pair_lookup[(left_idx, target_idx)] + pair_lookup[(right_idx, target_idx)]),
                }
            )
    return {"pairwise": pd.DataFrame(pair_rows), "hyperedges": pd.DataFrame(hyper_rows)}


def estimate_oracle_peid(
    spec: ModelSpec, *, samples: int = 2048, seed: int = 0, estimator: str = "transport"
) -> dict[str, pd.DataFrame]:
    return estimate_peid(spec, spec.vector_field, samples=samples, seed=seed, estimator=estimator)


@dataclass
class FittedMLP:
    net: object
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    train_mse: float
    baseline_mse: float

    def predict(self, states: np.ndarray) -> np.ndarray:
        import torch

        values = np.asarray(states, dtype=np.float32)
        scaled = (values - self.x_mean) / self.x_std
        self.net.eval()
        with torch.no_grad():
            pred = np.asarray(
                self.net(torch.tensor(scaled.tolist(), dtype=torch.float32)).cpu().tolist(),
                dtype=float,
            )
        return pred * self.y_std + self.y_mean


def fit_mlp(states: np.ndarray, targets: np.ndarray, *, seed: int, epochs: int) -> FittedMLP:
    import torch

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    x = np.asarray(states, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    split = max(32, int(0.8 * len(x)))
    x_mean = x[:split].mean(axis=0, keepdims=True)
    x_std = np.maximum(x[:split].std(axis=0, keepdims=True), 1e-6)
    y_mean = y[:split].mean(axis=0, keepdims=True)
    y_std = np.maximum(y[:split].std(axis=0, keepdims=True), 1e-6)
    xn = (x - x_mean) / x_std
    yn = (y - y_mean) / y_std
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*", category=UserWarning)
        net = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], 48),
            torch.nn.SiLU(),
            torch.nn.Linear(48, 48),
            torch.nn.SiLU(),
            torch.nn.Linear(48, y.shape[1]),
        )
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-5)
    xt = torch.tensor(xn[:split], dtype=torch.float32)
    yt = torch.tensor(yn[:split], dtype=torch.float32)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((net(xt) - yt) ** 2)
        loss.backward()
        optimizer.step()
    fitted = FittedMLP(net, x_mean, x_std, y_mean, y_std, 0.0, 0.0)
    prediction = fitted.predict(x[split:])
    fitted.train_mse = float(np.mean((prediction - y[split:]) ** 2))
    fitted.baseline_mse = float(np.mean((y[split:] - y[:split].mean(axis=0, keepdims=True)) ** 2))
    return fitted


def estimate_granger_ablation(
    model: FittedMLP, states: np.ndarray, targets: np.ndarray, spec: ModelSpec
) -> pd.DataFrame:
    base = model.predict(states)
    rows: list[dict[str, object]] = []
    for source_idx, source in enumerate(spec.state_names):
        ablated = np.asarray(states, dtype=float).copy()
        ablated[:, source_idx] = np.mean(ablated[:, source_idx])
        prediction = model.predict(ablated)
        for target_idx, target in enumerate(spec.target_names):
            base_mse = np.mean((targets[:, target_idx] - base[:, target_idx]) ** 2)
            ablated_mse = np.mean((targets[:, target_idx] - prediction[:, target_idx]) ** 2)
            rows.append({"source": source, "target": target, "score": float(max(0.0, ablated_mse - base_mse))})
    return pd.DataFrame(rows)


def estimate_neural_granger(
    states: np.ndarray, targets: np.ndarray, spec: ModelSpec, *, seed: int, epochs: int
) -> pd.DataFrame:
    import torch

    x = np.asarray(states, dtype=np.float32)
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = np.maximum(x.std(axis=0, keepdims=True), 1e-6)
    xn = (x - x_mean) / x_std
    rows: list[dict[str, object]] = []
    for target_idx, target in enumerate(spec.target_names):
        torch.manual_seed(int(seed) + target_idx)
        y = np.asarray(targets[:, target_idx], dtype=np.float32)
        yn = (y - y.mean()) / max(float(y.std()), 1e-6)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*", category=UserWarning)
            net = torch.nn.Sequential(
                torch.nn.Linear(x.shape[1], 24), torch.nn.Tanh(), torch.nn.Linear(24, 1)
            )
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        xt = torch.tensor(xn, dtype=torch.float32)
        yt = torch.tensor(yn[:, None], dtype=torch.float32)
        for _ in range(int(epochs)):
            optimizer.zero_grad(set_to_none=True)
            weights = net[0].weight
            group_penalty = torch.linalg.vector_norm(weights, dim=0).sum()
            loss = torch.mean((net(xt) - yt) ** 2) + 0.01 * group_penalty
            loss.backward()
            optimizer.step()
        norms = np.linalg.norm(np.asarray(net[0].weight.detach().tolist(), dtype=float), axis=0)
        for source, score in zip(spec.state_names, norms):
            rows.append({"source": source, "target": target, "score": float(score)})
    return pd.DataFrame(rows)


def estimate_shap_readout(
    model: FittedMLP, states: np.ndarray, spec: ModelSpec, *, samples: int, seed: int
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    count = min(int(samples), len(states))
    foreground = np.asarray(states, dtype=float)[rng.choice(len(states), size=count, replace=False)]
    background = np.mean(states, axis=0)
    baseline_rows = np.repeat(background[None, :], count, axis=0)
    baseline = model.predict(baseline_rows)
    feature_rows: list[dict[str, object]] = []
    for source_idx, source in enumerate(spec.state_names):
        modified = baseline_rows.copy()
        modified[:, source_idx] = foreground[:, source_idx]
        contribution = model.predict(modified) - baseline
        for target_idx, target in enumerate(spec.target_names):
            feature_rows.append(
                {"source": source, "target": target, "score": float(np.mean(np.abs(contribution[:, target_idx])))}
            )
    interaction_rows: list[dict[str, object]] = []
    for left_idx, right_idx in combinations(range(len(spec.state_names)), 2):
        both = baseline_rows.copy()
        left = baseline_rows.copy()
        right = baseline_rows.copy()
        both[:, [left_idx, right_idx]] = foreground[:, [left_idx, right_idx]]
        left[:, left_idx] = foreground[:, left_idx]
        right[:, right_idx] = foreground[:, right_idx]
        interaction = model.predict(both) - model.predict(left) - model.predict(right) + baseline
        for target_idx, target in enumerate(spec.target_names):
            interaction_rows.append(
                {
                    "sources": "+".join(sorted((spec.state_names[left_idx], spec.state_names[right_idx]))),
                    "target": target,
                    "score": float(np.mean(np.abs(interaction[:, target_idx]))),
                }
            )
    return {"pairwise": pd.DataFrame(feature_rows), "interactions": pd.DataFrame(interaction_rows)}


def observational_wms_surd(
    states: np.ndarray, targets: np.ndarray, spec: ModelSpec, *, bins: int = 6
) -> pd.DataFrame:
    source_index = {name: idx for idx, name in enumerate(spec.state_names)}
    target_index = {name: idx for idx, name in enumerate(spec.target_names)}
    rows: list[dict[str, object]] = []
    relations = list(spec.truth_hyperedges)
    if spec.name == "wilson_cowan":
        relations = [("w", "x", "dx"), ("w", "y", "dy")]
    for left, right, target in relations:
        a = _discretize(states[:, source_index[left]], bins)
        b = _discretize(states[:, source_index[right]], bins)
        t = _discretize(targets[:, target_index[target]], bins)
        values = _histogram_synergy(a, b, t, bins)
        surd = _specific_information_surd(a, b, t)
        rows.append(
            {
                "sources": "+".join(sorted((left, right))),
                "target": target,
                "wms": float(values["syn"]),
                **surd,
            }
        )
    return pd.DataFrame(rows)


def _specific_information_surd(a: np.ndarray, b: np.ndarray, target: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    t = np.asarray(target, dtype=int)
    n = len(t)
    totals = {"redundancy": 0.0, "unique_left": 0.0, "unique_right": 0.0, "synergy": 0.0}
    for target_value in np.unique(t):
        mask = t == target_value
        pt = float(mask.mean())
        if pt <= 0.0:
            continue

        def specific(source: np.ndarray) -> float:
            value = 0.0
            for source_value in np.unique(source[mask]):
                joint_mask = mask & (source == source_value)
                p_source_given_t = float(joint_mask.sum() / mask.sum())
                p_t_given_source = float(joint_mask.sum() / max((source == source_value).sum(), 1))
                if p_source_given_t > 0.0 and p_t_given_source > 0.0:
                    value += p_source_given_t * math.log2(p_t_given_source / pt)
            return value

        ia = specific(a)
        ib = specific(b)
        joint_codes = a * (int(b.max()) + 1) + b
        iab = specific(joint_codes)
        redundancy = min(ia, ib)
        totals["redundancy"] += pt * redundancy
        totals["unique_left"] += pt * (ia - redundancy)
        totals["unique_right"] += pt * (ib - redundancy)
        totals["synergy"] += pt * (iab - max(ia, ib))
    return {key: float(value) for key, value in totals.items()}


def _frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def _mean_records(
    seed_payloads: list[dict[str, object]], path: tuple[str, ...], keys: Sequence[str]
) -> list[dict[str, object]]:
    frames: list[pd.DataFrame] = []
    for payload in seed_payloads:
        value: object = payload
        for part in path:
            value = value[part]  # type: ignore[index]
        frames.append(pd.DataFrame(value))
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return []
    numeric = [column for column in combined.select_dtypes(include=[np.number]).columns if column not in keys]
    return _frame_records(combined.groupby(list(keys), as_index=False)[numeric].mean())


def _aggregate_seed_payloads(seed_payloads: list[dict[str, object]]) -> dict[str, object]:
    metrics = pd.DataFrame([payload["mlp_metrics"] for payload in seed_payloads])
    return {
        "seed": "mean",
        "mlp_metrics": {key: float(value) for key, value in metrics.mean().items()},
        "oracle_peid": {
            "pairwise": _mean_records(seed_payloads, ("oracle_peid", "pairwise"), ("source", "target")),
            "hyperedges": _mean_records(seed_payloads, ("oracle_peid", "hyperedges"), ("sources", "target")),
        },
        "mlp_peid": {
            "pairwise": _mean_records(seed_payloads, ("mlp_peid", "pairwise"), ("source", "target")),
            "hyperedges": _mean_records(seed_payloads, ("mlp_peid", "hyperedges"), ("sources", "target")),
        },
        "granger_ablation": _mean_records(seed_payloads, ("granger_ablation",), ("source", "target")),
        "neural_granger": _mean_records(seed_payloads, ("neural_granger",), ("source", "target")),
        "shap": {
            "pairwise": _mean_records(seed_payloads, ("shap", "pairwise"), ("source", "target")),
            "interactions": _mean_records(seed_payloads, ("shap", "interactions"), ("sources", "target")),
        },
        "observational": _mean_records(seed_payloads, ("observational",), ("sources", "target")),
        "all_seeds": seed_payloads,
    }


def _truth_synergy_scores(frame: pd.DataFrame, spec: ModelSpec) -> list[float]:
    scores: list[float] = []
    for left, right, target in spec.truth_hyperedges:
        key = "+".join(sorted((left, right)))
        selected = frame[(frame["sources"] == key) & (frame["target"] == target)]
        if not selected.empty:
            scores.append(float(selected.iloc[0]["score"]))
    return scores


def _plot_model_result(spec: ModelSpec, payload: dict[str, object], path: Path) -> None:
    import matplotlib.pyplot as plt

    oracle = pd.DataFrame(payload["oracle_peid"]["hyperedges"])
    learned = pd.DataFrame(payload["mlp_peid"]["hyperedges"])
    relations = list(spec.truth_hyperedges)
    if not relations:
        relations = [("w", "x", "dx"), ("w", "y", "dy")]
    labels = [f"{'+'.join(sorted((left, right)))}→{target}" for left, right, target in relations]
    positions = np.arange(len(labels))
    oracle_lookup = oracle.assign(label=oracle["sources"] + "→" + oracle["target"]).set_index("label")["score"]
    learned_lookup = learned.assign(label=learned["sources"] + "→" + learned["target"]).set_index("label")["score"]
    oracle_values = np.asarray([float(oracle_lookup.get(label, 0.0)) for label in labels])
    learned_values = np.asarray([float(learned_lookup.get(label, 0.0)) for label in labels])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, max(3.5, 0.55 * len(labels) + 2.0)), constrained_layout=True)
    axes[0].barh(positions - 0.16, oracle_values, height=0.28, label="Oracle + PEID", color="#4C78A8")
    axes[0].barh(positions + 0.16, learned_values, height=0.28, label="MLP + PEID", color="#E45756")
    axes[0].set_yticks(positions, labels, fontsize=7)
    axes[0].set_xlabel("Synergy (bits)")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)
    pair_labels = [f"{source}→{target}" for source, target in spec.truth_pairwise]
    methods = {
        "Granger": pd.DataFrame(payload["granger_ablation"]),
        "Neural Granger": pd.DataFrame(payload["neural_granger"]),
        "SHAP": pd.DataFrame(payload["shap"]["pairwise"]),
        "MLP PEID": pd.DataFrame(payload["mlp_peid"]["pairwise"]),
    }
    matrix = np.zeros((len(methods), len(pair_labels)), dtype=float)
    raw = np.zeros_like(matrix)
    for method_idx, frame in enumerate(methods.values()):
        lookup = frame.set_index(["source", "target"])["score"]
        raw[method_idx] = [float(lookup.get(edge, 0.0)) for edge in spec.truth_pairwise]
        scale = float(np.max(np.abs(raw[method_idx])))
        matrix[method_idx] = raw[method_idx] / scale if scale > 0.0 else raw[method_idx]
    axes[1].imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    axes[1].set_xticks(np.arange(len(pair_labels)), pair_labels, rotation=25, ha="right", fontsize=7)
    axes[1].set_yticks(np.arange(len(methods)), list(methods), fontsize=7)
    for row in range(raw.shape[0]):
        for col in range(raw.shape[1]):
            axes[1].text(col, row, f"{raw[row, col]:.3g}", ha="center", va="center", fontsize=6.5)
    axes[1].set_xlabel("Within-method normalized color; cells show raw scores")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(payloads: dict[str, dict[str, object]], specs: dict[str, ModelSpec], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows: list[list[float]] = []
    labels: list[str] = []
    for name, spec in specs.items():
        payload = payloads[name]
        oracle = pd.DataFrame(payload["oracle_peid"]["hyperedges"])
        learned = pd.DataFrame(payload["mlp_peid"]["hyperedges"])
        oracle_truth = _truth_synergy_scores(oracle, spec)
        learned_truth = _truth_synergy_scores(learned, spec)
        rows.append(
            [
                float(np.mean(oracle_truth)) if oracle_truth else 0.0,
                float(np.mean(learned_truth)) if learned_truth else 0.0,
                float(oracle["score"].abs().max()) if spec.name == "wilson_cowan" else 0.0,
                float(payload["mlp_metrics"]["skill_ratio"]),
            ]
        )
        labels.append(spec.display_name)
    matrix = np.asarray(rows, dtype=float)
    normalized = matrix.copy()
    for col in range(matrix.shape[1] - 1):
        scale = np.max(np.abs(matrix[:, col]))
        normalized[:, col] = matrix[:, col] / scale if scale > 0 else matrix[:, col]
    normalized[:, -1] = 1.0 - np.clip(matrix[:, -1], 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
    image = ax.imshow(normalized, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(
        np.arange(4),
        ["Oracle truth synergy", "MLP truth synergy", "WC additive-control max |syn|", "MLP MSE / baseline"],
        rotation=18,
        ha="right",
        fontsize=7,
    )
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03, label="Column-normalized intensity")
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _legacy_appendix(report_path: Path) -> str:
    if not report_path.exists():
        return "原 sine 基准未在此输出目录中提供。"
    current = report_path.read_text(encoding="utf-8")
    if LEGACY_MARKER in current:
        legacy = current.split(LEGACY_MARKER, maxsplit=1)[1].lstrip()
        intro = "以下内容保留原人工系统，用于校准两个外部源到第三目标的纯协同语义。它不再作为主实验。"
        while legacy.startswith(intro):
            legacy = legacy[len(intro) :].lstrip()
        return legacy
    return current


def _write_report(
    report_path: Path,
    payloads: dict[str, dict[str, object]],
    specs: dict[str, ModelSpec],
    summary_figure_path: Path,
    model_figure_paths: dict[str, Path],
) -> None:
    legacy = _legacy_appendix(report_path)

    def relative(path: Path) -> str:
        return os.path.relpath(path, report_path.parent).replace(os.sep, "/")

    sections: list[str] = []
    for name, spec in specs.items():
        payload = payloads[name]
        oracle = pd.DataFrame(payload["oracle_peid"]["hyperedges"])
        learned = pd.DataFrame(payload["mlp_peid"]["hyperedges"])
        truth_oracle = _truth_synergy_scores(oracle, spec)
        truth_learned = _truth_synergy_scores(learned, spec)
        truth_text = ", ".join(
            f"`{{{left},{right}}}->{target}`" for left, right, target in spec.truth_hyperedges
        ) or "无显式乘积或相位差交互（结构可加对照）"
        hyper_relations = list(spec.truth_hyperedges)
        if not hyper_relations:
            hyper_relations = [("w", "x", "dx"), ("w", "y", "dy")]
        oracle_lookup = oracle.set_index(["sources", "target"])["score"]
        learned_lookup = learned.set_index(["sources", "target"])["score"]
        shap_lookup = pd.DataFrame(payload["shap"]["interactions"]).set_index(["sources", "target"])["score"]
        observational = pd.DataFrame(payload["observational"])
        observational_lookup = (
            observational.set_index(["sources", "target"])
            if not observational.empty
            else pd.DataFrame(columns=["wms", "synergy"])
        )
        hyper_lines = [
            "| source set -> target | Oracle PEID | MLP PEID | SHAP interaction | observational WMS | SURD synergy |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for left, right, target in hyper_relations:
            key = "+".join(sorted((left, right)))
            obs_key = (key, target)
            wms = float(observational_lookup.loc[obs_key, "wms"]) if obs_key in observational_lookup.index else float("nan")
            surd = float(observational_lookup.loc[obs_key, "synergy"]) if obs_key in observational_lookup.index else float("nan")
            hyper_lines.append(
                f"| `{{{left},{right}}}->{target}` | {float(oracle_lookup.get((key, target), 0.0)):.4f} | "
                f"{float(learned_lookup.get((key, target), 0.0)):.4f} | "
                f"{float(shap_lookup.get((key, target), 0.0)):.4f} | "
                f"{wms:.4f} | {surd:.4f} |"
            )
        pair_frames = {
            "Granger": pd.DataFrame(payload["granger_ablation"]).set_index(["source", "target"])["score"],
            "Neural Granger": pd.DataFrame(payload["neural_granger"]).set_index(["source", "target"])["score"],
            "SHAP": pd.DataFrame(payload["shap"]["pairwise"]).set_index(["source", "target"])["score"],
            "MLP PEID": pd.DataFrame(payload["mlp_peid"]["pairwise"]).set_index(["source", "target"])["score"],
        }
        pair_lines = [
            "| pairwise truth | Granger | Neural Granger | SHAP | MLP PEID |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for source, target in spec.truth_pairwise:
            pair_lines.append(
                f"| `{source}->{target}` | {float(pair_frames['Granger'].get((source, target), 0.0)):.4f} | "
                f"{float(pair_frames['Neural Granger'].get((source, target), 0.0)):.4f} | "
                f"{float(pair_frames['SHAP'].get((source, target), 0.0)):.4f} | "
                f"{float(pair_frames['MLP PEID'].get((source, target), 0.0)):.4f} |"
            )
        interpretation = (
            "该模型没有显式二源乘积或相位差项，但 PEID 仍可能为 `状态 + 外部驱动` 给出正联合信息残差；"
            "因此它是结构交互负对照，不是 PEID 数值零对照。"
            if spec.name == "wilson_cowan"
            else "真值协同采用包含目标当前状态的状态依赖门控口径。"
        )
        hyper_table = "\n".join(hyper_lines)
        pair_table = "\n".join(pair_lines)
        sections.append(
            f"""## {spec.display_name}

论文方程：

$$
{spec.equation}
$$

- 结构真值：{truth_text}。
- Oracle 真值协同均值：{float(np.mean(truth_oracle)) if truth_oracle else 0.0:.4f} bits。
- MLP 真值协同均值：{float(np.mean(truth_learned)) if truth_learned else 0.0:.4f} bits。
- MLP 测试 MSE / 常数基线：{float(payload['mlp_metrics']['skill_ratio']):.4f}。
- 解释：{interpretation}

{hyper_table}

{pair_table}

![{spec.display_name} 方法读出]({relative(model_figure_paths[name])})
"""
        )
    model_sections = "\n".join(sections)
    text = rf"""# 经典网络动力学中的共同驱动与状态依赖协同

完整实验设计、数值协议和结果讨论见 [经典网络动力学 benchmark 报告](classic_network_dynamics_benchmark.md)。

主比较使用论文 *Discovering network dynamics with neural symbolic regression* 的原始动力学方程。网络只缩减为可解释 motif；预测目标统一为当前状态到向量场 $\dot{{\mathbf{{x}}}}$，避免小步长下一状态中的恒等映射掩盖耦合机制。

协同源集合允许包含目标变量的当前状态。它表示状态依赖门控，例如 SIS 中感染源 $w$ 的作用受到目标当前易感比例 $1-x$ 调制，并不等价于两个外部源共同指向第三变量的 collider。

主表比较 Granger ablation、Neural Granger、SHAP、观测 WMS/SURD、MLP+PEID 与 Oracle+PEID。PCMCI 保留在附录，因为它检验下一状态时间序列中的滞后关系，而不是当前状态到向量场的监督映射。

![跨模型汇总]({relative(summary_figure_path)})

图中前三列在各列内部归一化，格内数字是原始值。最后一列越低表示 MLP 相对常数基线越好。Wilson–Cowan 是结构可加对照，不应被误写成 PEID 数值零对照。

{model_sections}

## 方法口径

- Granger ablation：在固定 MLP 上把单个当前状态替换为均值，读取目标导数预测误差增量。
- Neural Granger：逐目标 cMLP 第一层 source-group norm，仍是 pairwise 预测结构。
- SHAP：独立背景替换下的单源贡献和二源 inclusion–exclusion interaction。
- Observational WMS/SURD：直接基于自然轨迹的状态与导数经验分布。
- PEID：对源状态做独立最大熵干预，再比较联合 EI 与单源 EI；主结果使用 transport-map 估计，smoke 测试使用离散估计。

{LEGACY_MARKER}

以下内容保留原人工系统，用于校准两个外部源到第三目标的纯协同语义。它不再作为主实验。

{legacy.rstrip()}
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run_benchmark(
    *,
    mode: str = "full",
    result_dir: Path = DEFAULT_RESULT_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    seeds: Sequence[int] = (0, 1, 2),
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    result_dir = Path(result_dir)
    figure_dir = Path(figure_dir)
    report_path = Path(report_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    specs = build_model_specs()
    trajectory_samples = 260 if mode == "smoke" else 1600
    intervention_samples = 700 if mode == "smoke" else 1800
    epochs = 35 if mode == "smoke" else 180
    neural_epochs = 20 if mode == "smoke" else 100
    estimator = "histogram" if mode == "smoke" else "transport"
    payloads: dict[str, dict[str, object]] = {}
    model_figure_paths: dict[str, Path] = {}

    for model_name, spec in specs.items():
        seed_payloads: list[dict[str, object]] = []
        for seed in seeds:
            states, targets = spec.simulate(seed=int(seed), samples=trajectory_samples, noise=0.01)
            fitted = fit_mlp(states, targets, seed=int(seed) + 101, epochs=epochs)
            oracle = estimate_oracle_peid(
                spec, samples=intervention_samples, seed=int(seed) + 1001, estimator=estimator
            )
            learned = estimate_peid(
                spec,
                fitted.predict,
                samples=intervention_samples,
                seed=int(seed) + 1001,
                estimator=estimator,
            )
            shap = estimate_shap_readout(fitted, states, spec, samples=28 if mode == "smoke" else 72, seed=seed)
            seed_payloads.append(
                {
                    "seed": int(seed),
                    "mlp_metrics": {
                        "test_mse": fitted.train_mse,
                        "baseline_mse": fitted.baseline_mse,
                        "skill_ratio": fitted.train_mse / max(fitted.baseline_mse, 1e-12),
                    },
                    "oracle_peid": {key: _frame_records(value) for key, value in oracle.items()},
                    "mlp_peid": {key: _frame_records(value) for key, value in learned.items()},
                    "granger_ablation": _frame_records(estimate_granger_ablation(fitted, states, targets, spec)),
                    "neural_granger": _frame_records(
                        estimate_neural_granger(states, targets, spec, seed=seed + 4001, epochs=neural_epochs)
                    ),
                    "shap": {key: _frame_records(value) for key, value in shap.items()},
                    "observational": _frame_records(observational_wms_surd(states, targets, spec)),
                }
            )
        payload = _aggregate_seed_payloads(seed_payloads)
        payloads[model_name] = payload
        model_path = figure_dir / f"{model_name}_readout.png"
        _plot_model_result(spec, payload, model_path)
        model_figure_paths[model_name] = model_path
        (result_dir / f"{model_name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    summary_figure_path = figure_dir / "classic_dynamics_summary.png"
    _plot_summary(payloads, specs, summary_figure_path)
    _write_report(report_path, payloads, specs, summary_figure_path, model_figure_paths)
    summary = {
        "mode": mode,
        "models": list(specs),
        "seeds": [int(seed) for seed in seeds],
        "estimator": estimator,
        "summary_figure_path": str(summary_figure_path),
        "model_figure_paths": {key: str(value) for key, value in model_figure_paths.items()},
        "report_path": str(report_path),
    }
    summary_path = result_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**summary, "summary_path": str(summary_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(
        mode=args.mode,
        result_dir=args.result_dir,
        figure_dir=args.figure_dir,
        report_path=args.report_path,
        seeds=tuple(args.seeds),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
