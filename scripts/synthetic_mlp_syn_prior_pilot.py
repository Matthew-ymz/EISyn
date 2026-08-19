#!/usr/bin/env python3
"""Controlled pilot: can a known Syn prior improve a finite-sample MLP?"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "synthetic_mlp_syn_prior_pilot"
METHODS = ("uniform", "oracle_syn", "shuffled_syn", "oracle_univariate")
COLORS = {
    "uniform": "#6B7280",
    "oracle_syn": "#D97706",
    "shuffled_syn": "#9CA3AF",
    "oracle_univariate": "#4C78A8",
}
LABELS = {
    "uniform": "Uniform",
    "oracle_syn": "Oracle Syn",
    "shuffled_syn": "Shuffled Syn",
    "oracle_univariate": "Oracle univariate",
}


@dataclass(frozen=True)
class Config:
    input_dim: int = 10
    validation_size: int = 512
    test_size: int = 4096
    noise_sd: float = 0.35
    hidden_widths: tuple[int, ...] = (32, 16)
    learning_rate: float = 1e-2
    max_epochs: int = 160
    patience: int = 20
    alpha_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1)
    gamma_grid: tuple[float, ...] = (0.25, 0.5, 1.0)
    syn_tolerance_bits: float = 0.005


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_widths: Iterable[int]) -> None:
        super().__init__()
        widths = (input_dim, *tuple(hidden_widths), 1)
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(widths) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    @property
    def first_linear(self) -> nn.Linear:
        return self.net[0]  # type: ignore[return-value]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def generate_split(size: int, eta: float, config: Config, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = rng.choice((-1.0, 1.0), size=(size, config.input_dim)).astype(np.float32)
    unique = x[:, 0]
    synergy = x[:, 1] * x[:, 2]
    noise = config.noise_sd * rng.normal(size=size)
    y = math.sqrt(1.0 - eta) * unique + math.sqrt(eta) * synergy + noise
    return x, y.astype(np.float32)


def make_data(train_size: int, eta: float, seed: int, config: Config) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(10_000 + seed)
    train_x, train_y = generate_split(train_size, eta, config, rng)
    val_x, val_y = generate_split(config.validation_size, eta, config, rng)
    test_x, test_y = generate_split(config.test_size, eta, config, rng)
    x_mean = train_x.mean(axis=0)
    x_sd = train_x.std(axis=0)
    y_mean = float(train_y.mean())
    y_sd = float(train_y.std())
    if y_sd <= 0.0 or np.any(x_sd <= 0.0):
        raise RuntimeError("Degenerate training scale.")
    return {
        "train_x": (train_x - x_mean) / x_sd,
        "val_x": (val_x - x_mean) / x_sd,
        "test_x": (test_x - x_mean) / x_sd,
        "train_y": (train_y - y_mean) / y_sd,
        "val_y": (val_y - y_mean) / y_sd,
        "test_y": (test_y - y_mean) / y_sd,
        "train_y_sd": np.asarray(y_sd),
    }


def prior_scores(method: str, eta: float, input_dim: int, seed: int) -> np.ndarray:
    score = np.zeros(input_dim, dtype=np.float64)
    if method in {"oracle_syn", "shuffled_syn"} and eta > 0.0:
        score[1:3] = eta
        if method == "shuffled_syn":
            score = np.random.default_rng(80_000 + seed).permutation(score)
    elif method == "oracle_univariate" and eta < 1.0:
        score[0] = 1.0 - eta
    elif method != "uniform":
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
    return score


def penalty_weights(scores: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 0.0 or float(scores.max()) <= 0.0:
        return np.ones_like(scores)
    epsilon = 0.05 * float(scores.mean())
    raw = np.power(scores + epsilon, -gamma)
    return raw / raw.mean()


def regularization(model: MLP, feature_weights: torch.Tensor) -> torch.Tensor:
    first = model.first_linear.weight
    penalty = torch.mean(feature_weights * torch.mean(first.square(), dim=0))
    for module in model.net[1:]:
        if isinstance(module, nn.Linear):
            penalty = penalty + torch.mean(module.weight.square())
    return penalty


def fit_candidate(
    data: dict[str, np.ndarray], config: Config, *, alpha: float, weights: np.ndarray, init_seed: int
) -> tuple[MLP, float, int]:
    set_seed(init_seed)
    model = MLP(config.input_dim, config.hidden_widths)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_x = torch.as_tensor(data["train_x"], dtype=torch.float32)
    train_y = torch.as_tensor(data["train_y"], dtype=torch.float32)
    val_x = torch.as_tensor(data["val_x"], dtype=torch.float32)
    val_y = torch.as_tensor(data["val_y"], dtype=torch.float32)
    tensor_weights = torch.as_tensor(weights, dtype=torch.float32)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    for epoch in range(config.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        mse = torch.mean((model(train_x) - train_y).square())
        loss = mse + float(alpha) * regularization(model, tensor_weights)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(torch.mean((model(val_x) - val_y).square()).item())
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("Training failed to produce a finite validation loss.")
    model.load_state_dict(best_state)
    return model, best_loss, best_epoch


def evaluate(model: MLP, data: dict[str, np.ndarray]) -> float:
    model.eval()
    with torch.no_grad():
        prediction = np.asarray(
            model(torch.as_tensor(data["test_x"], dtype=torch.float32)).tolist(),
            dtype=np.float32,
        )
    return float(np.sqrt(np.mean((prediction - data["test_y"]) ** 2)))


def tune_method(method: str, data: dict[str, np.ndarray], eta: float, seed: int, config: Config) -> dict[str, object]:
    scores = prior_scores(method, eta, config.input_dim, seed)
    gammas = (0.0,) if method == "uniform" or float(scores.max()) == 0.0 else config.gamma_grid
    candidates: list[tuple[float, float, float, int, MLP]] = []
    for alpha in config.alpha_grid:
        for gamma in gammas:
            weights = penalty_weights(scores, gamma)
            model, val_mse, epoch = fit_candidate(
                data, config, alpha=alpha, weights=weights, init_seed=100_000 + seed
            )
            candidates.append((val_mse, alpha, gamma, epoch, model))
    val_mse, alpha, gamma, epoch, model = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return {
        "method": method,
        "test_nrmse": evaluate(model, data),
        "validation_mse": float(val_mse),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "best_epoch": int(epoch),
        "prior_scores": scores.tolist(),
        "penalty_weights": penalty_weights(scores, gamma).tolist(),
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    scenarios = sorted({(int(row["train_size"]), float(row["eta"])) for row in rows})
    for train_size, eta in scenarios:
        scenario = [row for row in rows if int(row["train_size"]) == train_size and float(row["eta"]) == eta]
        uniform = {int(row["seed"]): float(row["test_nrmse"]) for row in scenario if row["method"] == "uniform"}
        for method in METHODS:
            selected = [row for row in scenario if row["method"] == method]
            values = np.asarray([float(row["test_nrmse"]) for row in selected])
            deltas = np.asarray([uniform[int(row["seed"])] - float(row["test_nrmse"]) for row in selected])
            summary.append({
                "train_size": train_size,
                "eta": eta,
                "method": method,
                "n_seeds": len(selected),
                "test_nrmse_mean": float(values.mean()),
                "test_nrmse_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "gain_over_uniform_mean": float(deltas.mean()),
                "gain_over_uniform_sd": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
                "gain_positive_fraction": float(np.mean(deltas > 0.0)),
            })
    return summary


def plot_results(rows: list[dict[str, object]], output: Path) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    })
    scenarios = sorted({(int(row["train_size"]), float(row["eta"])) for row in rows})
    x = np.arange(len(scenarios), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), constrained_layout=True)
    for method in METHODS:
        means, errors = [], []
        for train_size, eta in scenarios:
            values = np.asarray([
                float(row["test_nrmse"]) for row in rows
                if row["method"] == method and int(row["train_size"]) == train_size and float(row["eta"]) == eta
            ])
            means.append(values.mean())
            errors.append(values.std(ddof=1) if len(values) > 1 else 0.0)
        axes[0].errorbar(x, means, yerr=errors, marker="o", ms=3.5, lw=1.1, capsize=2,
                         color=COLORS[method], label=LABELS[method])
    labels = [f"N={size}\nη={eta:g}" for size, eta in scenarios]
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Test nRMSE (lower is better)")
    axes[0].text(-0.14, 1.03, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=9)
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    jitter = np.linspace(-0.07, 0.07, len({int(row["seed"]) for row in rows}))
    for scenario_index, (train_size, eta) in enumerate(scenarios):
        uniform = {int(row["seed"]): float(row["test_nrmse"]) for row in rows
                   if row["method"] == "uniform" and int(row["train_size"]) == train_size and float(row["eta"]) == eta}
        oracle = sorted((int(row["seed"]), float(row["test_nrmse"])) for row in rows
                        if row["method"] == "oracle_syn" and int(row["train_size"]) == train_size and float(row["eta"]) == eta)
        gains = np.asarray([uniform[seed] - value for seed, value in oracle])
        axes[1].scatter(scenario_index + jitter[:len(gains)], gains, s=20, color="#D97706", alpha=0.8, zorder=3)
        axes[1].plot([scenario_index - 0.12, scenario_index + 0.12], [gains.mean(), gains.mean()], color="#111827", lw=1.2)
    axes[1].axhline(0.0, color="#9CA3AF", lw=0.8, ls="--")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Oracle-Syn gain over uniform nRMSE")
    axes[1].text(-0.14, 1.03, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=9)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def run(config: Config, train_sizes: tuple[int, ...], etas: tuple[float, ...], seeds: tuple[int, ...], output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for train_size in train_sizes:
        for eta in etas:
            for seed in seeds:
                data = make_data(train_size, eta, seed, config)
                for method in METHODS:
                    result = tune_method(method, data, eta, seed, config)
                    row = {"train_size": train_size, "eta": eta, "seed": seed, **result}
                    rows.append(row)
                    print(json.dumps({key: row[key] for key in ("train_size", "eta", "seed", "method", "test_nrmse", "alpha", "gamma")}), flush=True)
    payload = {
        "question": "What changes when only the assignment of known Syn centrality to MLP regularization changes?",
        "mechanism": "independent_binary_unique_plus_xor_product",
        "status": "exploratory_pilot",
        "config": asdict(config),
        "train_sizes": list(train_sizes),
        "etas": list(etas),
        "seeds": list(seeds),
        "rows": rows,
        "summary": summarize(rows),
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_results(rows, output_dir / "synthetic_mlp_syn_prior_pilot")
    return payload


def parse_tuple(text: str, cast) -> tuple:
    return tuple(cast(value) for value in text.split(",") if value.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sizes", default="256,1024")
    parser.add_argument("--etas", default="0,1")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-epochs", type=int, default=Config.max_epochs)
    args = parser.parse_args()
    config = Config(max_epochs=args.max_epochs)
    run(config, parse_tuple(args.train_sizes, int), parse_tuple(args.etas, float), parse_tuple(args.seeds, int), args.output_dir)


if __name__ == "__main__":
    main()
