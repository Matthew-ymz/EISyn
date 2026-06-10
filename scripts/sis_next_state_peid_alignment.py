#!/usr/bin/env python3
"""Align learned and known-dynamics PEID on a stochastic SIS transition."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_network_dynamics_benchmark import (
    _rk4_step,
    build_model_specs,
    estimate_peid,
)

DEFAULT_RESULT_PATH = ROOT / "results" / "classic_network_dynamics_benchmark" / "sis_next_state_alignment.json"
DEFAULT_FIGURE_PATH = ROOT / "fig" / "classic_network_dynamics_benchmark" / "sis_next_state_alignment.png"
DEFAULT_REPORT_PATH = ROOT / "docs" / "reports" / "sis_next_state_peid_alignment.md"


@dataclass(frozen=True)
class SisAlignmentConfig:
    tau: float = 1.0
    process_noise: float = 0.05
    training_samples_per_source: int = 2400
    transition_replicates: int = 3
    warmup_steps: int = 400
    acceptance_relative_error: float = 0.20

    @property
    def dt(self) -> float:
        return float(build_model_specs()["sis"].dt)

    @property
    def integration_steps(self) -> int:
        steps = int(round(float(self.tau) / self.dt))
        if steps <= 0 or not np.isclose(steps * self.dt, self.tau):
            raise ValueError("tau must be a positive integer multiple of the SIS integration step.")
        return steps


@dataclass(frozen=True)
class TrainingPairs:
    inputs: np.ndarray
    targets: np.ndarray
    source_labels: np.ndarray
    target_names: tuple[str, ...] = ("w_tau", "x_tau", "y_tau")


@dataclass
class FittedProbabilisticMLP:
    net: object
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    test_nll: float

    def predict_distribution(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch

        values = np.asarray(states, dtype=float)
        scaled = (values - self.x_mean) / self.x_std
        self.net.eval()
        with torch.no_grad():
            raw = np.asarray(
                self.net(torch.tensor(scaled.tolist(), dtype=torch.float32)).cpu().tolist(),
                dtype=float,
            )
        mean = raw[:, :3] * self.y_std + self.y_mean
        log_std = np.clip(raw[:, 3:], -5.0, 2.0)
        std = np.exp(log_std) * self.y_std
        return mean, std

    def sample(self, states: np.ndarray, *, seed: int) -> np.ndarray:
        mean, std = self.predict_distribution(states)
        rng = np.random.default_rng(int(seed))
        return np.clip(mean + std * rng.normal(size=mean.shape), 0.0, 1.0)


def _advance(states: np.ndarray, *, steps: int, process_noise: float, rng: np.random.Generator) -> np.ndarray:
    spec = build_model_specs()["sis"]
    result = np.asarray(states, dtype=float).copy()
    for _ in range(int(steps)):
        result = np.asarray([_rk4_step(spec.vector_field, row, spec.dt) for row in result])
        result += rng.normal(0.0, float(process_noise) * np.sqrt(spec.dt), size=result.shape)
        result = np.clip(result, 0.0, 1.0)
    return result


def simulate_sis_transition(
    states: np.ndarray, *, config: SisAlignmentConfig, seed: int
) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("states must have shape (n, 3).")
    if config.process_noise < 0.0:
        raise ValueError("process_noise must be nonnegative.")
    return _advance(
        values,
        steps=config.integration_steps,
        process_noise=config.process_noise,
        rng=np.random.default_rng(int(seed)),
    )


def _natural_states(*, config: SisAlignmentConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    state = rng.uniform(0.15, 0.75, size=(1, 3))
    rows: list[np.ndarray] = []
    total = int(config.warmup_steps) + int(config.training_samples_per_source)
    for step in range(total):
        state = _advance(state, steps=1, process_noise=config.process_noise, rng=rng)
        if step >= config.warmup_steps:
            rows.append(state[0].copy())
    return np.asarray(rows)


def build_training_pairs(*, config: SisAlignmentConfig, seed: int) -> TrainingPairs:
    if config.training_samples_per_source <= 0:
        raise ValueError("training_samples_per_source must be positive.")
    if config.transition_replicates <= 0:
        raise ValueError("transition_replicates must be positive.")
    natural = _natural_states(config=config, seed=int(seed) + 11)
    rng = np.random.default_rng(int(seed) + 23)
    intervention = rng.uniform(0.02, 0.98, size=(config.training_samples_per_source, 3))
    base_inputs = np.vstack([natural, intervention])
    base_labels = np.concatenate(
        [
            np.repeat("natural", len(natural)),
            np.repeat("intervention", len(intervention)),
        ]
    )
    inputs = np.vstack([base_inputs] * config.transition_replicates)
    labels = np.concatenate([base_labels] * config.transition_replicates)
    targets = np.vstack(
        [
            simulate_sis_transition(base_inputs, config=config, seed=int(seed) + 101 + replicate)
            for replicate in range(config.transition_replicates)
        ]
    )
    return TrainingPairs(inputs=inputs, targets=targets, source_labels=labels)


def fit_probabilistic_mlp(
    states: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    epochs: int,
    hidden_units: int = 64,
) -> FittedProbabilisticMLP:
    import torch

    x = np.asarray(states, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("states and targets must both have shape (n, 3).")
    if len(x) < 10:
        raise ValueError("at least 10 training pairs are required.")
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(x))
    split = max(8, int(0.8 * len(x)))
    train_idx, test_idx = order[:split], order[split:]
    x_mean = x[train_idx].mean(axis=0, keepdims=True)
    x_std = np.maximum(x[train_idx].std(axis=0, keepdims=True), 1e-6)
    y_mean = y[train_idx].mean(axis=0, keepdims=True)
    y_std = np.maximum(y[train_idx].std(axis=0, keepdims=True), 1e-6)
    xn = (x - x_mean) / x_std
    yn = (y - y_mean) / y_std
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*", category=UserWarning)
        net = torch.nn.Sequential(
            torch.nn.Linear(3, int(hidden_units)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_units), int(hidden_units)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_units), 6),
        )
    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    xt = torch.tensor(xn[train_idx].tolist(), dtype=torch.float32)
    yt = torch.tensor(yn[train_idx].tolist(), dtype=torch.float32)

    def gaussian_nll(raw, expected):
        mean = raw[:, :3]
        log_std = torch.clamp(raw[:, 3:], -5.0, 2.0)
        return torch.mean(0.5 * ((expected - mean) * torch.exp(-log_std)) ** 2 + log_std)

    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = gaussian_nll(net(xt), yt)
        loss.backward()
        optimizer.step()
    model = FittedProbabilisticMLP(net, x_mean, x_std, y_mean, y_std, 0.0)
    test_x = torch.tensor(xn[test_idx].tolist(), dtype=torch.float32)
    test_y = torch.tensor(yn[test_idx].tolist(), dtype=torch.float32)
    net.eval()
    with torch.no_grad():
        model.test_nll = float(gaussian_nll(net(test_x), test_y).item())
    return model


def run_alignment(
    *,
    config: SisAlignmentConfig,
    seed: int,
    estimator: str,
    peid_samples: int,
    epochs: int,
) -> dict[str, object]:
    dataset = build_training_pairs(config=config, seed=int(seed))
    model = fit_probabilistic_mlp(
        dataset.inputs,
        dataset.targets,
        seed=int(seed) + 1001,
        epochs=int(epochs),
    )
    base_spec = build_model_specs()["sis"]
    transition_spec = replace(
        base_spec,
        target_names=dataset.target_names,
        truth_pairwise=(("w", "x_tau"), ("w", "y_tau")),
        truth_hyperedges=(("w", "x", "x_tau"), ("w", "y", "y_tau")),
    )
    channel_seed = int(seed) + 5001
    oracle = estimate_peid(
        transition_spec,
        lambda states: simulate_sis_transition(states, config=config, seed=channel_seed),
        samples=int(peid_samples),
        seed=int(seed) + 4001,
        estimator=estimator,
    )
    learned = estimate_peid(
        transition_spec,
        lambda states: model.sample(states, seed=channel_seed),
        samples=int(peid_samples),
        seed=int(seed) + 4001,
        estimator=estimator,
    )
    oracle_scores = oracle["hyperedges"].set_index(["sources", "target"])["score"]
    learned_scores = learned["hyperedges"].set_index(["sources", "target"])["score"]
    relations: dict[str, dict[str, float]] = {}
    for sources, target in (("w+x", "x_tau"), ("w+y", "y_tau")):
        oracle_value = float(oracle_scores.loc[(sources, target)])
        learned_value = float(learned_scores.loc[(sources, target)])
        relative_error = abs(learned_value - oracle_value) / max(abs(oracle_value), 1e-12)
        relations[f"{sources}->{target}"] = {
            "oracle_synergy": oracle_value,
            "mlp_synergy": learned_value,
            "relative_error": float(relative_error),
        }
    passed = all(
        relation["relative_error"] <= config.acceptance_relative_error
        for relation in relations.values()
    )
    return {
        "protocol": {
            "system": "sis",
            "tau": float(config.tau),
            "dt": float(config.dt),
            "integration_steps": int(config.integration_steps),
            "process_noise": float(config.process_noise),
            "noise_location": "sis_dynamics",
            "training_distribution": "equal_natural_and_uniform_intervention",
            "transition_replicates": int(config.transition_replicates),
            "target_names": list(dataset.target_names),
            "estimator": estimator,
            "peid_samples": int(peid_samples),
            "acceptance_relative_error": float(config.acceptance_relative_error),
        },
        "model": {"test_nll": float(model.test_nll), "epochs": int(epochs)},
        "relations": relations,
        "passed": bool(passed),
    }


def _plot_alignment(summary: dict[str, object], path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    relations = summary["relations"]
    labels = list(relations)
    oracle = np.asarray([relations[label]["oracle_synergy"] for label in labels], dtype=float)
    learned = np.asarray([relations[label]["mlp_synergy"] for label in labels], dtype=float)
    errors = np.asarray([relations[label]["relative_error"] for label in labels], dtype=float)
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
    ax.bar(positions - 0.18, oracle, width=0.34, label="Known dynamics + PEID", color="#4C78A8")
    ax.bar(positions + 0.18, learned, width=0.34, label="Probabilistic MLP + PEID", color="#E45756")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Synergistic effective information (bits)")
    ax.set_title("Stochastic SIS next-state PEID alignment")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    for index, (value, error) in enumerate(zip(learned, errors)):
        ax.text(index + 0.18, value, f"  err={100.0 * error:.1f}%", ha="center", va="bottom", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _write_report(summary: dict[str, object], figure_path: Path, report_path: Path) -> None:
    protocol = summary["protocol"]
    relations = summary["relations"]
    relative_figure = os.path.relpath(figure_path, report_path.parent).replace(os.sep, "/")
    rows = [
        "| 关系 | 已知动力学 PEID | MLP+PEID | 相对误差 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in relations.items():
        rows.append(
            f"| `{name}` | {values['oracle_synergy']:.4f} | {values['mlp_synergy']:.4f} | "
            f"{100.0 * values['relative_error']:.2f}% |"
        )
    status = "通过" if summary["passed"] else "未通过"
    text = f"""# SIS 下一状态 PEID 对齐实验

本实验比较已知随机 SIS 动力学与概率 MLP 所定义的有限时间转移通道。输入输出间隔为 `tau={protocol['tau']}`，即 {protocol['integration_steps']} 个 RK4 步。

随机性来自 SIS 动力学过程噪声：每个积分步加入标准差为 `{protocol['process_noise']} * sqrt(dt)` 的 Wiener 增量。PEID 估计阶段不额外添加噪声。概率 MLP 使用条件高斯负对数似然学习完整的未来状态分布，而不是只学习条件均值。

训练输入由等量自然轨迹状态和独立均匀干预域状态组成。Oracle 与 MLP 均以当前完整状态为输入，以 $\\mathbf{{x}}(t+\\tau)$ 为目标。

![SIS 下一状态 PEID 对齐]({relative_figure})

{chr(10).join(rows)}

验收阈值为每条目标关系相对误差不超过 {100.0 * protocol['acceptance_relative_error']:.0f}%。本次结果：**{status}**。

该结果支持一个有限结论：当训练分布覆盖 PEID 干预域、MLP 学习随机有限时间转移分布时，MLP+PEID 可以在 SIS 系统上逼近已知动力学 PEID。它不代表所有经典系统都自动满足这一性质。
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")


def run_experiment(
    *,
    config: SisAlignmentConfig,
    seed: int,
    estimator: str,
    peid_samples: int,
    epochs: int,
    result_path: Path = DEFAULT_RESULT_PATH,
    figure_path: Path = DEFAULT_FIGURE_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, object]:
    summary = run_alignment(
        config=config,
        seed=seed,
        estimator=estimator,
        peid_samples=peid_samples,
        epochs=epochs,
    )
    result_path = Path(result_path)
    figure_path = Path(figure_path)
    report_path = Path(report_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot_alignment(summary, figure_path)
    _write_report(summary, figure_path, report_path)
    return {
        **summary,
        "result_path": str(result_path),
        "figure_path": str(figure_path),
        "report_path": str(report_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        config = SisAlignmentConfig(training_samples_per_source=64, transition_replicates=2, warmup_steps=20)
        estimator, peid_samples, epochs = "histogram", 500, 50
    else:
        config = SisAlignmentConfig()
        estimator, peid_samples, epochs = "transport", 1800, 450
    result = run_experiment(
        config=config,
        seed=args.seed,
        estimator=estimator,
        peid_samples=peid_samples,
        epochs=epochs,
        result_path=args.result_path,
        figure_path=args.figure_path,
        report_path=args.report_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
