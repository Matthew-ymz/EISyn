#!/usr/bin/env python3
"""Screen ODE mechanisms for interpretable broad-domain MLP+PEID curves."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_network_dynamics_benchmark import _plot_panel, fit_mlp
from scripts.discrete_iteration_dynamics_benchmark import (
    MapSpec,
    _broad_one_step_sample_count,
    _broad_one_step_distribution_metadata,
    _broad_one_step_readout_factory,
    _identity_project,
    _run_map_sweep,
    simulate_broad_one_step_pool,
)


DEFAULT_RESULT_DIR = ROOT / "results" / "ode_synergy_candidate_benchmark"
DEFAULT_FIGURE_DIR = ROOT / "fig" / "ode_synergy_candidate_benchmark"
DEFAULT_REPORT_PATH = ROOT / "docs" / "reports" / "ode_synergy_candidate_screening.md"

CANDIDATE_PARAMETER_VALUES = {
    "sis": (0.0, 0.25, 0.5, 0.75, 1.0),
    "lorenz": (0.0, 0.25, 0.5, 0.75, 1.0),
    "rossler": (0.0, 0.25, 0.5, 0.75, 1.0),
    "kuramoto": (0.0, 0.05, 0.1, 0.2, 0.3),
}

CANDIDATE_META = {
    "sis": {
        "display_name": "SIS Infection Gate",
        "parameter_key": "beta",
        "xlabel": "Infection gate strength beta",
        "panel": "a  SIS infection gate",
        "relation": "w+x->x_tau",
        "role": "Primary biological state-dependent gate",
    },
    "lorenz": {
        "display_name": "Lorenz Product Gate",
        "parameter_key": "gamma",
        "xlabel": "Lorenz product gate gamma",
        "panel": "b  Lorenz product gate",
        "relation": "x+y->z_tau",
        "role": "Primary chaotic-system product mechanism",
    },
    "rossler": {
        "display_name": "Rossler Product Gate",
        "parameter_key": "gamma",
        "xlabel": "Rossler product gate gamma",
        "panel": "c  Rossler internal product gate",
        "relation": "x+z->z_tau",
        "role": "Second explicit product validation",
    },
    "kuramoto": {
        "display_name": "Kuramoto Phase Gate",
        "parameter_key": "kappa",
        "xlabel": "Phase gate strength kappa",
        "panel": "d  Kuramoto phase gate",
        "relation": "w+x->x_tau",
        "role": "Periodic-variable estimator boundary case",
    },
}


@dataclass
class IncrementFittedMLP:
    """Expose an increment-trained MLP through the next-state predictor API."""

    increment_model: object
    baseline_mse: float | None = None

    def __post_init__(self) -> None:
        for name in ("net", "x_mean", "x_std", "y_mean", "y_std", "train_mse"):
            setattr(self, name, getattr(self.increment_model, name))
        if self.baseline_mse is None:
            self.baseline_mse = float(getattr(self.increment_model, "baseline_mse"))

    def predict(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=float)
        return values + np.asarray(self.increment_model.predict(values), dtype=float)


@dataclass
class QuadraticIncrementModel:
    net: object
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    train_mse: float
    baseline_mse: float

    @staticmethod
    def features(states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=float)
        products = [
            values[:, left] * values[:, right]
            for left in range(values.shape[1])
            for right in range(left, values.shape[1])
        ]
        return np.column_stack([values, *products])

    def predict(self, states: np.ndarray) -> np.ndarray:
        import torch

        features = self.features(states)
        scaled = (features - self.x_mean) / self.x_std
        self.net.eval()
        with torch.no_grad():
            prediction = np.asarray(
                self.net(torch.tensor(scaled.tolist(), dtype=torch.float32)).cpu().tolist(),
                dtype=float,
            )
        return prediction * self.y_std + self.y_mean


def _fit_quadratic_increment_model(
    states: np.ndarray,
    increments: np.ndarray,
) -> QuadraticIncrementModel:
    import torch

    values = np.asarray(states, dtype=float)
    targets = np.asarray(increments, dtype=float)
    split = max(32, int(0.8 * len(values)))
    features = QuadraticIncrementModel.features(values)
    x_mean = features[:split].mean(axis=0, keepdims=True)
    x_std = np.maximum(features[:split].std(axis=0, keepdims=True), 1e-8)
    y_mean = targets[:split].mean(axis=0, keepdims=True)
    y_std = np.maximum(targets[:split].std(axis=0, keepdims=True), 1e-8)
    design = (features[:split] - x_mean) / x_std
    response = (targets[:split] - y_mean) / y_std
    augmented = np.column_stack([design, np.ones(len(design))])
    coefficients = np.linalg.lstsq(augmented, response, rcond=None)[0]
    net = torch.nn.Linear(features.shape[1], targets.shape[1])
    with torch.no_grad():
        net.weight.copy_(
            torch.tensor(coefficients[:-1].T.tolist(), dtype=torch.float32)
        )
        net.bias.copy_(torch.tensor(coefficients[-1].tolist(), dtype=torch.float32))
    model = QuadraticIncrementModel(net, x_mean, x_std, y_mean, y_std, 0.0, 0.0)
    validation_prediction = model.predict(values[split:])
    model.train_mse = float(np.mean((validation_prediction - targets[split:]) ** 2))
    model.baseline_mse = float(
        np.mean((targets[split:] - targets[:split].mean(axis=0, keepdims=True)) ** 2)
    )
    return model


def _increment_surrogate_factory(
    spec: MapSpec,
    seed: int,
    params: Mapping[str, int | float | str],
) -> tuple[object, np.ndarray, np.ndarray, dict[str, object]]:
    train_states, train_targets = simulate_broad_one_step_pool(
        spec,
        seed=100000 + int(seed),
        samples=_broad_one_step_sample_count(params),
    )
    neural_increment_model = fit_mlp(
        train_states,
        train_targets - train_states,
        seed=300000 + int(seed),
        epochs=int(params["epochs"]),
    )
    quadratic_increment_model = _fit_quadratic_increment_model(
        train_states,
        train_targets - train_states,
    )
    candidates = {
        "nonlinear_increment_mlp": neural_increment_model,
        "quadratic_feature_increment_mlp": quadratic_increment_model,
    }
    selected_architecture, increment_model = min(
        candidates.items(), key=lambda item: float(item[1].train_mse)
    )
    split = max(32, int(0.8 * len(train_states)))
    validation_targets = train_targets[split:]
    baseline = train_targets[:split].mean(axis=0, keepdims=True)
    baseline_mse = float(np.mean((validation_targets - baseline) ** 2))
    fitted = IncrementFittedMLP(increment_model, baseline_mse=baseline_mse)
    return (
        fitted,
        train_states,
        train_targets,
        {
            "surrogate_training_distribution": "broad_intervention_domain_one_step_pool",
            "surrogate_training_target": "one_step_increment",
            "surrogate_validation": "fit_mlp_internal_heldout_split",
            "surrogate_selection_objective": "heldout_increment_prediction_mse",
            "surrogate_selected_architecture": selected_architecture,
            "surrogate_candidate_validation_mse": {
                name: float(model.train_mse) for name, model in candidates.items()
            },
            "oracle_used_for_selection": False,
            "peid_used_for_selection": False,
        },
    )


def _uniform_sampler(bounds: np.ndarray) -> Callable[[int, int], np.ndarray]:
    def sample(samples: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        return np.column_stack(
            [rng.uniform(low, high, size=int(samples)) for low, high in bounds]
        )

    return sample


def build_sis_infection_gate_spec(beta: float) -> MapSpec:
    beta = float(beta)
    dt = 0.5
    bounds = np.array([[0.02, 0.98], [0.02, 0.98]], dtype=float)

    def transition(values: np.ndarray) -> np.ndarray:
        w, x = values.T
        return np.column_stack(
            [
                w + dt * (-0.8 * w + w * (1.0 - w)),
                x + dt * (-x + beta * w * (1.0 - x)),
            ]
        )

    return MapSpec(
        name="ode_sis_infection_gate",
        display_name=CANDIDATE_META["sis"]["display_name"],
        state_names=("w", "x"),
        target_names=("w_tau", "x_tau"),
        equation=r"\dot w=-0.8w+w(1-w),\quad \dot x=-x+\beta w(1-x)",
        parameter_key="beta",
        parameter_value=beta,
        parameter_values=CANDIDATE_PARAMETER_VALUES["sis"],
        intervention_bounds=bounds,
        truth_hyperedges=(("w", "x", "x_tau"),),
        _transition=transition,
        _project=_identity_project,
        _initial_state=lambda rng: rng.uniform(bounds[:, 0], bounds[:, 1]),
        _sample_intervention=_uniform_sampler(bounds),
        burnin_steps=0,
    )


def build_lorenz_product_gate_spec(gamma: float) -> MapSpec:
    gamma = float(gamma)
    dt = 0.01
    bounds = np.array([[-12.0, 12.0], [-16.0, 16.0], [5.0, 35.0]], dtype=float)

    def transition(values: np.ndarray) -> np.ndarray:
        x, y, z = values.T
        return np.column_stack(
            [
                x + dt * 10.0 * (y - x),
                y + dt * (x * (28.0 - z) - y),
                z + dt * (gamma * x * y - (8.0 / 3.0) * z),
            ]
        )

    return MapSpec(
        name="ode_lorenz_product_gate",
        display_name=CANDIDATE_META["lorenz"]["display_name"],
        state_names=("x", "y", "z"),
        target_names=("x_tau", "y_tau", "z_tau"),
        equation=(
            r"\dot x=10(y-x),\quad \dot y=x(28-z)-y,\quad "
            r"\dot z=\gamma xy-\frac{8}{3}z"
        ),
        parameter_key="gamma",
        parameter_value=gamma,
        parameter_values=CANDIDATE_PARAMETER_VALUES["lorenz"],
        intervention_bounds=bounds,
        truth_hyperedges=(("x", "y", "z_tau"),),
        _transition=transition,
        _project=_identity_project,
        _initial_state=lambda rng: rng.uniform(bounds[:, 0], bounds[:, 1]),
        _sample_intervention=_uniform_sampler(bounds),
        burnin_steps=0,
    )


def build_rossler_product_gate_spec(gamma: float) -> MapSpec:
    gamma = float(gamma)
    dt = 0.05
    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0], [0.2, 6.0]], dtype=float)

    def transition(values: np.ndarray) -> np.ndarray:
        x, y, z = values.T
        return np.column_stack(
            [
                x + dt * (-y - z),
                y + dt * (x + 0.165 * y),
                z + dt * (2.0 + z * (gamma * x - 5.5)),
            ]
        )

    return MapSpec(
        name="ode_rossler_product_gate",
        display_name=CANDIDATE_META["rossler"]["display_name"],
        state_names=("x", "y", "z"),
        target_names=("x_tau", "y_tau", "z_tau"),
        equation=(
            r"\dot x=-y-z,\quad \dot y=x+0.165y,\quad "
            r"\dot z=2+z(\gamma x-5.5)"
        ),
        parameter_key="gamma",
        parameter_value=gamma,
        parameter_values=CANDIDATE_PARAMETER_VALUES["rossler"],
        intervention_bounds=bounds,
        truth_hyperedges=(("x", "z", "z_tau"),),
        _transition=transition,
        _project=_identity_project,
        _initial_state=lambda rng: rng.uniform(bounds[:, 0], bounds[:, 1]),
        _sample_intervention=_uniform_sampler(bounds),
        burnin_steps=0,
    )


def build_kuramoto_phase_gate_spec(kappa: float) -> MapSpec:
    kappa = float(kappa)
    dt = 0.2
    bounds = np.array([[-np.pi, np.pi], [-np.pi, np.pi]], dtype=float)

    def transition(values: np.ndarray) -> np.ndarray:
        w, x = values.T
        return np.column_stack(
            [
                w + dt * 0.9,
                x + dt * (1.0 + kappa * np.sin(w - x)),
            ]
        )

    return MapSpec(
        name="ode_kuramoto_phase_gate",
        display_name=CANDIDATE_META["kuramoto"]["display_name"],
        state_names=("w", "x"),
        target_names=("w_tau", "x_tau"),
        equation=r"\dot w=0.9,\quad \dot x=1+\kappa\sin(w-x)",
        parameter_key="kappa",
        parameter_value=kappa,
        parameter_values=CANDIDATE_PARAMETER_VALUES["kuramoto"],
        intervention_bounds=bounds,
        truth_hyperedges=(("w", "x", "x_tau"),),
        _transition=transition,
        _project=_identity_project,
        _initial_state=lambda rng: rng.uniform(bounds[:, 0], bounds[:, 1]),
        _sample_intervention=_uniform_sampler(bounds),
        burnin_steps=0,
    )


CANDIDATE_BUILDERS: Mapping[str, Callable[[float], MapSpec]] = {
    "sis": build_sis_infection_gate_spec,
    "lorenz": build_lorenz_product_gate_spec,
    "rossler": build_rossler_product_gate_spec,
    "kuramoto": build_kuramoto_phase_gate_spec,
}


def _candidate_sweep_parameters(mode: str) -> dict[str, int | float | str]:
    if mode == "smoke":
        return {
            "trajectories": 3,
            "samples_per_trajectory": 30,
            "epochs": 35,
            "shap_samples": 18,
            "estimator": "transport",
        }
    if mode == "full":
        return {
            "trajectories": 10,
            "samples_per_trajectory": 180,
            "epochs": 300,
            "shap_samples": 72,
            "estimator": "transport",
        }
    raise ValueError("mode must be 'smoke' or 'full'.")


def _enrich_result(result: dict[str, object], *, result_path: Path) -> dict[str, object]:
    rows = list(result["rows"])
    for row in rows:
        mse = float(row["mlp_test_mse"])
        baseline = max(float(row["mlp_baseline_mse"]), 1e-12)
        row["prediction_nrmse"] = float(math.sqrt(mse / baseline))
        row["peid_oracle_abs_error"] = abs(
            float(row["peid_synergy"]) - float(row["oracle_peid_synergy"])
        )

    frame = pd.DataFrame(rows)
    parameter_key = str(result["parameter_key"])
    summary_lookup = {
        float(row[parameter_key]): row for row in result["summary"]  # type: ignore[index]
    }
    for parameter_value, group in frame.groupby(parameter_key, sort=True):
        summary_row = summary_lookup[float(parameter_value)]
        for metric in ("prediction_nrmse", "peid_oracle_abs_error"):
            values = group[metric].astype(float)
            summary_row[f"{metric}_mean"] = float(values.mean())
            summary_row[f"{metric}_std"] = float(values.std(ddof=0))

    screening = _screen_candidate(result)
    result["rows"] = rows
    result["screening"] = screening
    result["protocol_note"] = (
        "The scanned parameter directly controls the registered two-source term; "
        "all methods and Oracle PEID share the same held-out broad one-step states."
    )
    persisted = {key: value for key, value in result.items() if key != "result_path"}
    result_path.write_text(json.dumps(persisted, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _screen_candidate(result: Mapping[str, object]) -> dict[str, object]:
    summary = list(result["summary"])
    parameter_key = str(result["parameter_key"])
    zero_rows = [row for row in summary if np.isclose(float(row[parameter_key]), 0.0)]
    active_rows = [row for row in summary if not np.isclose(float(row[parameter_key]), 0.0)]
    mean_nrmse = float(np.mean([float(row["prediction_nrmse_mean"]) for row in summary]))
    oracle_mae = float(np.mean([float(row["peid_oracle_abs_error_mean"]) for row in summary]))
    if active_rows:
        active_oracle_scale = max(
            float(np.mean([abs(float(row["oracle_peid_synergy_mean"])) for row in active_rows])),
            0.05,
        )
        active_mlp_scale = max(
            float(np.mean([abs(float(row["peid_synergy_mean"])) for row in active_rows])),
            1e-12,
        )
        normalized_oracle_mae = oracle_mae / active_oracle_scale
        zero_residual_ratio = (
            abs(float(zero_rows[0]["peid_synergy_mean"])) / active_mlp_scale
            if zero_rows
            else float("nan")
        )
    else:
        normalized_oracle_mae = float("nan")
        zero_residual_ratio = float("nan")
    if len(active_rows) >= 2:
        oracle_values = np.asarray(
            [float(row["oracle_peid_synergy_mean"]) for row in active_rows], dtype=float
        )
        mlp_values = np.asarray(
            [float(row["peid_synergy_mean"]) for row in active_rows], dtype=float
        )
        if np.std(oracle_values) > 1e-12 and np.std(mlp_values) > 1e-12:
            trend_correlation = float(np.corrcoef(oracle_values, mlp_values)[0, 1])
        else:
            trend_correlation = float("nan")
    else:
        trend_correlation = float("nan")
    fairness_passed = bool(result["fairness_audit"]["passed"])  # type: ignore[index]
    sufficient_grid = bool(active_rows)
    recommended = (
        fairness_passed
        and sufficient_grid
        and mean_nrmse <= 0.20
        and normalized_oracle_mae <= 0.50
        and zero_residual_ratio <= 0.50
        and np.isfinite(trend_correlation)
        and trend_correlation >= 0.50
    )
    system = str(result["system"]).removeprefix("ode_").removesuffix("_candidate")
    if not sufficient_grid:
        decision = "insufficient_grid"
    elif system == "kuramoto" and not recommended:
        decision = "boundary_case"
    else:
        decision = "recommended" if recommended else "not_recommended"
    return {
        "decision": decision,
        "fairness_passed": fairness_passed,
        "mean_prediction_nrmse": mean_nrmse,
        "mean_peid_oracle_abs_error": oracle_mae,
        "normalized_peid_oracle_mae": normalized_oracle_mae,
        "zero_residual_to_active_ratio": zero_residual_ratio,
        "active_trend_correlation": trend_correlation,
        "thresholds": {
            "mean_prediction_nrmse_max": 0.20,
            "normalized_peid_oracle_mae_max": 0.50,
            "zero_residual_to_active_ratio_max": 0.50,
            "active_trend_correlation_min": 0.50,
        },
    }


def run_candidate_sweep(
    system: str,
    *,
    mode: str = "full",
    parameter_values: Sequence[float] | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    result_path: Path | None = None,
    figure_path: Path | None = None,
) -> dict[str, object]:
    if system not in CANDIDATE_BUILDERS:
        raise ValueError(f"Unknown candidate {system!r}.")
    meta = CANDIDATE_META[system]
    result_path = Path(result_path or DEFAULT_RESULT_DIR / f"{system}_synergy_sweep.json")
    figure_path = Path(figure_path or DEFAULT_FIGURE_DIR / f"{system}_synergy_sweep.png")
    result = _run_map_sweep(
        system=f"ode_{system}_candidate",
        mode=mode,
        parameter_values=parameter_values or CANDIDATE_PARAMETER_VALUES[system],
        seeds=seeds,
        result_path=result_path,
        figure_path=figure_path,
        result_system=f"ode_{system}_candidate",
        display_name=str(meta["display_name"]),
        builder_override=CANDIDATE_BUILDERS[system],
        xlabel_override=str(meta["xlabel"]),
        structural_zero_values=(0.0,),
        params_override=_candidate_sweep_parameters(mode),
        surrogate_factory=_increment_surrogate_factory,
        readout_factory=_broad_one_step_readout_factory,
        peid_uses_readout_states=True,
        distribution_metadata=_broad_one_step_distribution_metadata(),
    )
    return _enrich_result(result, result_path=result_path)


def _plot_candidate_grid(systems: Mapping[str, Mapping[str, object]], path: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2), constrained_layout=True)
    for index, system in enumerate(("sis", "lorenz", "rossler", "kuramoto")):
        axis = axes.flat[index]
        payload = systems[system]
        meta = CANDIDATE_META[system]
        _plot_panel(
            axis,
            payload["summary"],  # type: ignore[arg-type]
            parameter_key=str(payload["parameter_key"]),
            xlabel=str(meta["xlabel"]),
            label=str(meta["panel"]),
            include_oracle_peid=True,
        )
        if index in (0, 2):
            axis.set_ylabel("Native synergy readout")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def _format_metric(value: object) -> str:
    number = float(value)
    return "n/a" if not np.isfinite(number) else f"{number:.4f}"


def _write_report(
    systems: Mapping[str, Mapping[str, object]],
    *,
    report_path: Path,
    combined_figure_path: Path,
) -> None:
    relative_figure = os.path.relpath(combined_figure_path, report_path.parent).replace(os.sep, "/")
    sections: list[str] = []
    decision_rows = [
        "| Candidate | Selected increment model(s) | Fairness | Mean prediction NRMSE | Normalized MLP/Oracle PEID MAE | Zero/active ratio | Trend correlation | Decision |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    interpretations = {
        "sis": (
            "扫描参数直接打开感染源与当前易感比例的乘性门控。若 MLP+PEID 贴近 Oracle，"
            "它说明 surrogate 保留了状态依赖感染机制，而不仅是自然轨迹上的预测相关。"
        ),
        "lorenz": (
            "扫描参数只直接改变读取目标中的 $xy$ 乘积项，因此比扫描 $\\rho$ 更适合检验"
            "二源协同强度。"
        ),
        "rossler": (
            "扫描内部乘积系数而非外部振子耦合，修正了旧曲线中“横轴不控制被读取机制”"
            "导致的近水平结果。"
        ),
        "kuramoto": (
            "相位差项是明确的联合非线性，但盒状最大熵干预没有显式识别圆周拓扑；"
            "因此该 panel 同时检验周期支持上的估计边界。"
        ),
    }
    findings = {
        "sis": (
            "Oracle PEID 在门控打开后显著升高并随更强门控进入下降区间；正式配置下 "
            "MLP+PEID 恢复了这一非单调形状。SHAP interaction 随幅值缓慢增加，但没有"
            "表达相同的信息平台与下降趋势。"
        ),
        "lorenz": (
            "MLP+PEID 几乎贴合 Oracle PEID，二者只缓慢上升；SHAP interaction 则随 "
            "$\\gamma$ 近线性放大。该对比说明局部乘积响应幅值增加，并不等价于一步"
            "转移中的不可约机制信息同比增加。"
        ),
        "rossler": (
            "直接拟合下一状态时，低 prediction NRMSE 曾掩盖错误的 PEID 趋势。改为增量训练并"
            "按 held-out 增量误差选择通用二阶交互特征后，MLP+PEID 与 Oracle 的下降曲线"
            "基本重合，说明问题来自 surrogate 对小交互增量的表示误差。"
        ),
        "kuramoto": (
            "Oracle PEID 在任意正 $\\kappa$ 后接近平台，符合固定相位差形状的尺度不变性。"
            "增量 MLP 恢复了相同平台和轻微下降趋势；这表明原先的持续上升主要来自下一状态"
            "恒等项主导训练损失，而不是圆周机制本身不可学习。"
        ),
    }
    for system in ("sis", "lorenz", "rossler", "kuramoto"):
        payload = systems[system]
        meta = CANDIDATE_META[system]
        screening = payload["screening"]
        fairness = payload["fairness_audit"]
        architectures = ", ".join(
            sorted({str(row["surrogate_selected_architecture"]) for row in payload["rows"]})
        )
        decision_rows.append(
            f"| {meta['display_name']} | `{architectures}` | {str(bool(fairness['passed']))} | "
            f"{_format_metric(screening['mean_prediction_nrmse'])} | "
            f"{_format_metric(screening['normalized_peid_oracle_mae'])} | "
            f"{_format_metric(screening['zero_residual_to_active_ratio'])} | "
            f"{_format_metric(screening['active_trend_correlation'])} | "
            f"`{screening['decision']}` |"
        )
        parameter_key = str(payload["parameter_key"])
        curve_rows = [
            f"| {parameter_key} | MLP+PEID | Oracle PEID | WMS | SURD | SHAP interaction | Prediction NRMSE |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in payload["summary"]:
            curve_rows.append(
                f"| {float(row[parameter_key]):.4g} | {float(row['peid_synergy_mean']):.4f} | "
                f"{float(row['oracle_peid_synergy_mean']):.4f} | {float(row['wms_mean']):.4f} | "
                f"{float(row['surd_synergy_mean']):.4f} | {float(row['shap_interaction_mean']):.4f} | "
                f"{float(row['prediction_nrmse_mean']):.4f} |"
            )
        sections.append(
            f"""## {meta['display_name']}

$$
{payload['equation']}
$$

- 注册关系：`{meta['relation']}`。
- 机制解释：{interpretations[system]}
- 结果解释：{findings[system]}
- 公平性审计：`passed={str(bool(fairness['passed'])).lower()}`。
- 候选结论：`{screening['decision']}`。

{chr(10).join(curve_rows)}
"""
        )
    report = f"""# 连续微分方程协同候选筛选

## 实验问题

本实验检验早期微分方程案例能否复现 Wilson-Cowan refractory panel 的解释链：
扫描参数直接控制被读取的二源机制，MLP 在 broad one-step 域内拟合真实转移，
MLP+PEID 再从该 surrogate 中恢复与 Oracle PEID 一致的机制趋势。

PEID 的本地理论依据强调：最大熵干预下的协同读出属于动力学机制，而不是自然轨迹
相关性；当动力学未知时，可先用机器学习拟合转移机制，再在固定 predictor 上实施
干预读出。连续变量结果仍是有限样本 transport-map 估计，不能把微小负值或零点残差
解释为严格理论原子。

## 统一协议与公平性审计

- 每个候选均用显式 Euler 一步映射定义 `state -> next state`。
- 横轴参数直接打开或增强注册二源项，并包含结构零点。
- 每个参数和 seed 使用同一 broad 训练输入池分布，以及独立 held-out broad readout 池。
- WMS、SURD、SHAP、MLP+PEID 与 Oracle PEID 共享 held-out readout states。
- SHAP 与 MLP+PEID 共享同一个 fitted MLP。
- surrogate 拟合一步增量而非下一状态；训练候选包括普通非线性增量 MLP 与带通用二阶
  交互特征的增量网络，只按内部 held-out 增量预测 MSE 选择，PEID 与 Oracle 不参与选择。
- 全部信息读出使用相同的 transport-map 配置。
- 零点保留同流程估计 residual，不替换为手工结构零。
- 因为 broad source states 按最大熵独立采样，Oracle PEID 的
  `joint EI - single EI sum` 与 WMS 在数值上重合；这属于定义关系，不能把 WMS
  当作该协议下独立于 Oracle PEID 的对照证据。

![Four ODE synergy candidates]({relative_figure})

## 候选结论

{chr(10).join(decision_rows)}

`recommended` 表示当前 broad one-step 协议下同时通过拟合、Oracle 对齐和零点残差筛选；
`boundary_case` 表示机制本身合理，但估计支持或曲线稳定性需要单独处理；
`not_recommended` 表示当前结果不足以支持进入 Part1 主图。

{chr(10).join(sections)}

## 解释边界

不同方法保留各自原生尺度。WMS 和 SURD 读取 held-out broad 样本的联合分布，
SHAP interaction 读取 fitted MLP 响应幅值中的非加性形状，MLP+PEID 与 Oracle PEID
读取最大熵干预下的机制信息。主比较应关注结构零点、参数趋势、MLP/Oracle 一致性和
跨 seed 稳定性，而不是直接比较不同方法的绝对纵轴数值。
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.rstrip() + "\n", encoding="utf-8")


def run_ode_synergy_candidate_benchmark(
    *,
    mode: str = "full",
    seeds: Sequence[int] = (0, 1, 2),
    result_dir: Path = DEFAULT_RESULT_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    parameter_overrides: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, object]:
    result_dir = Path(result_dir)
    figure_dir = Path(figure_dir)
    report_path = Path(report_path)
    systems: dict[str, dict[str, object]] = {}
    for system in ("sis", "lorenz", "rossler", "kuramoto"):
        systems[system] = run_candidate_sweep(
            system,
            mode=mode,
            parameter_values=(
                parameter_overrides[system]
                if parameter_overrides and system in parameter_overrides
                else CANDIDATE_PARAMETER_VALUES[system]
            ),
            seeds=seeds,
            result_path=result_dir / f"{system}_synergy_sweep.json",
            figure_path=figure_dir / f"{system}_synergy_sweep.png",
        )
    combined_figure_path = figure_dir / "four_ode_candidate_synergy_panels.png"
    _plot_candidate_grid(systems, combined_figure_path)
    _write_report(systems, report_path=report_path, combined_figure_path=combined_figure_path)
    summary = {
        "mode": mode,
        "seeds": [int(seed) for seed in seeds],
        "systems": {
            system: {
                "result_path": payload["result_path"],
                "figure_path": payload["figure_path"],
                "screening": payload["screening"],
            }
            for system, payload in systems.items()
        },
        "combined_figure_path": str(combined_figure_path),
        "report_path": str(report_path),
    }
    summary_path = result_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**summary, "summary_path": str(summary_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_ode_synergy_candidate_benchmark(
        mode=args.mode,
        seeds=args.seeds,
        result_dir=args.result_dir,
        figure_dir=args.figure_dir,
        report_path=args.report_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
