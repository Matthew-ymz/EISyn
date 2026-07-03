#!/usr/bin/env python3
"""Run a no-confounder sine-frequency MLP+PEID experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_granger_peid_mlp import (  # noqa: E402
    DEFAULT_FIGURE_DIR,
    DEFAULT_RESULT_DIR,
    SimConfig,
    _intervention_features,
    _sample_intervention_sources,
    make_lagged_dataset,
    train_mlp_transition_model,
)

DEFAULT_RESULT_PATH = DEFAULT_RESULT_DIR / "sine_frequency_mlp_peid_sweep.json"
DEFAULT_FIGURE_PATH = DEFAULT_FIGURE_DIR / "sine_frequency_mlp_peid_sweep.png"
DEFAULT_REPORT_PATH = ROOT / "docs" / "reports" / "Sine_Frequency_MLP_PEID.md"
VARIABLES = ("x", "y", "z")


def simulate_no_confounder_sine_system(
    *,
    n_samples: int,
    seed: int,
    alpha: float,
    sine_frequency: float,
    noise: float,
) -> pd.DataFrame:
    """Generate x/y/z dynamics with only the sine hyperedge into z."""

    rng = np.random.default_rng(int(seed))
    data = np.zeros((int(n_samples), len(VARIABLES)), dtype=float)
    data[0, 0] = rng.normal(0.0, 0.4)
    data[0, 1] = rng.normal(0.0, 0.4)
    data[0, 2] = rng.normal(0.0, 0.2)
    for t in range(int(n_samples) - 1):
        data[t + 1, 0] = 0.42 * data[t, 0] + rng.normal(0.0, 0.55)
        data[t + 1, 1] = 0.38 * data[t, 1] + rng.normal(0.0, 0.55)
        data[t + 1, 2] = (
            0.22 * data[t, 2]
            + float(alpha) * np.sin(float(sine_frequency) * data[t, 0] * data[t, 1])
            + rng.normal(0.0, float(noise))
        )
    return pd.DataFrame(data, columns=VARIABLES)


def _fixed_oracle_xy_z(
    *,
    alpha: float,
    sine_frequency: float,
    samples: int,
    seed: int,
    support: Mapping[str, tuple[float, float]],
) -> dict[str, float]:
    from yrd.transport_map import summarize_two_source_synergy_transport_map

    for name in VARIABLES:
        low, high = support[name]
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"invalid oracle support for {name!r}")
    if tuple(support["x"]) != tuple(support["y"]):
        raise ValueError("oracle support for x and y must match")

    rng = np.random.default_rng(int(seed))
    pair_count = max(1, int(math.ceil(int(samples) / 2)))
    x_base = rng.uniform(*support["x"], size=pair_count)
    y_base = rng.uniform(*support["y"], size=pair_count)
    z_base = rng.uniform(*support["z"], size=pair_count)
    x = np.concatenate([x_base, y_base])[: int(samples)].reshape(-1, 1)
    y = np.concatenate([y_base, x_base])[: int(samples)].reshape(-1, 1)
    z_state = np.concatenate([z_base, z_base])[: int(samples)]
    target = (
        0.22 * z_state
        + float(alpha) * np.sin(float(sine_frequency) * x[:, 0] * y[:, 0])
    ).reshape(-1, 1)
    result = summarize_two_source_synergy_transport_map(x, y, target)
    return {
        "sine_frequency": float(sine_frequency),
        **{key: float(result[key]) for key in ("left_ei", "right_ei", "joint_ei", "syn")},
    }


def _slope(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    if len(frame) < 2:
        return float("nan")
    x = frame[x_col].to_numpy(dtype=float)
    y = frame[y_col].to_numpy(dtype=float)
    if len(np.unique(x)) < 2:
        return float("nan")
    return float(np.polyfit(x, y, deg=1)[0])


def _trend(summary: pd.DataFrame) -> dict[str, object]:
    alpha_slope_by_k = []
    for k, group in summary.groupby("k"):
        ordered = group.sort_values("alpha")
        alpha_slope_by_k.append(
            {
                "k": float(k),
                "mlp_peid_synergy_slope_per_alpha": _slope(ordered, "alpha", "mlp_peid_xy_synergy_mean"),
                "oracle_peid_synergy_slope_per_alpha": _slope(ordered, "alpha", "oracle_peid_xy_synergy_mean"),
                "z_train_r2_slope_per_alpha": _slope(ordered, "alpha", "z_train_r2_mean"),
            }
        )

    k_slope_by_alpha = []
    for alpha, group in summary.groupby("alpha"):
        ordered = group.sort_values("k")
        k_slope_by_alpha.append(
            {
                "alpha": float(alpha),
                "mlp_peid_synergy_slope_per_k": _slope(ordered, "k", "mlp_peid_xy_synergy_mean"),
                "oracle_peid_synergy_slope_per_k": _slope(ordered, "k", "oracle_peid_xy_synergy_mean"),
                "z_train_r2_slope_per_k": _slope(ordered, "k", "z_train_r2_mean"),
            }
        )

    return {
        "alpha_slope_by_k": alpha_slope_by_k,
        "k_slope_by_alpha": k_slope_by_alpha,
    }


def run_sine_frequency_mlp_peid_sweep(
    *,
    alpha_values: Sequence[float] = (0.25, 0.5, 1.0, 1.5, 2.0),
    k_values: Sequence[float] = (1.0, 2.0, 4.0, 6.0, 8.0, 10.0),
    seeds: Sequence[int] = (0, 1, 2, 3),
    n_samples: int = 1100,
    noise: float = 0.05,
    mlp_epochs: int = 90,
    intervention_samples: int = 640,
    bins: int = 4,
    oracle_intervention_support: Mapping[str, tuple[float, float]] | None = None,
    oracle_intervention_seed: int = 17021,
) -> dict[str, object]:
    from yrd.transport_map import summarize_two_source_synergy_transport_map

    oracle_support = dict(
        oracle_intervention_support
        or {
            "x": (-1.8, 1.8),
            "y": (-1.8, 1.8),
            "z": (-1.25, 1.25),
        }
    )

    oracle_by_alpha_k = {
        (float(alpha), float(k)): _fixed_oracle_xy_z(
            alpha=float(alpha),
            sine_frequency=float(k),
            samples=int(intervention_samples),
            seed=int(oracle_intervention_seed),
            support=oracle_support,
        )
        for alpha in alpha_values
        for k in k_values
    }

    rows: list[dict[str, float]] = []
    for alpha in alpha_values:
        for k in k_values:
            for seed in seeds:
                series = simulate_no_confounder_sine_system(
                    n_samples=int(n_samples),
                    seed=int(seed),
                    alpha=float(alpha),
                    sine_frequency=float(k),
                    noise=float(noise),
                )
                config = SimConfig(
                    mechanism="common_driver_sine_synergy",
                    n_samples=int(n_samples),
                    noise=float(noise),
                    seed=int(seed),
                    synergy_strength=float(alpha),
                    common_driver_strength=0.0,
                    mlp_epochs=int(mlp_epochs),
                    intervention_samples=int(intervention_samples),
                    bins=int(bins),
                    variable_names=VARIABLES,
                )
                features, targets = make_lagged_dataset(series, lag=config.lag)
                model = train_mlp_transition_model(features, targets, config)
                samples = _sample_intervention_sources(series, config)
                predictions = model.predict(_intervention_features(samples, config))
                tm_peid = summarize_two_source_synergy_transport_map(
                    samples[["x"]].to_numpy(dtype=float),
                    samples[["y"]].to_numpy(dtype=float),
                    predictions[:, [VARIABLES.index("z")]],
                )
                train_pred = model.predict(features)
                z_target = targets[:, VARIABLES.index("z")]
                z_pred = train_pred[:, VARIABLES.index("z")]
                z_mse = float(np.mean((z_target - z_pred) ** 2))
                z_baseline_mse = float(np.mean((z_target - float(np.mean(z_target))) ** 2))
                oracle = oracle_by_alpha_k[(float(alpha), float(k))]
                rows.append(
                    {
                        "run_id": f"alpha={float(alpha):g}|sine_frequency={float(k):g}|seed={int(seed)}",
                        "alpha": float(alpha),
                        "k": float(k),
                        "seed": float(seed),
                        "final_train_loss": float(model.loss_history[-1]) if model.loss_history else float("nan"),
                        "z_train_mse": z_mse,
                        "z_train_r2": float(1.0 - z_mse / (z_baseline_mse + 1e-12)),
                        "mlp_peid_unique_x": float(tm_peid["left_ei"]),
                        "mlp_peid_unique_y": float(tm_peid["right_ei"]),
                        "mlp_peid_xy_joint": float(tm_peid["joint_ei"]),
                        "mlp_peid_xy_synergy": float(tm_peid["syn"]),
                        "oracle_peid_unique_x": float(oracle["left_ei"]),
                        "oracle_peid_unique_y": float(oracle["right_ei"]),
                        "oracle_peid_xy_joint": float(oracle["joint_ei"]),
                        "oracle_peid_xy_synergy": float(oracle["syn"]),
                    }
                )

    frame = pd.DataFrame(rows)
    aggregations = dict(
        final_train_loss_mean=("final_train_loss", "mean"),
        final_train_loss_std=("final_train_loss", "std"),
        z_train_r2_mean=("z_train_r2", "mean"),
        z_train_r2_std=("z_train_r2", "std"),
        mlp_peid_unique_x_mean=("mlp_peid_unique_x", "mean"),
        mlp_peid_unique_x_std=("mlp_peid_unique_x", "std"),
        mlp_peid_unique_y_mean=("mlp_peid_unique_y", "mean"),
        mlp_peid_unique_y_std=("mlp_peid_unique_y", "std"),
        mlp_peid_xy_joint_mean=("mlp_peid_xy_joint", "mean"),
        mlp_peid_xy_joint_std=("mlp_peid_xy_joint", "std"),
        mlp_peid_xy_synergy_mean=("mlp_peid_xy_synergy", "mean"),
        mlp_peid_xy_synergy_std=("mlp_peid_xy_synergy", "std"),
        oracle_peid_unique_x_mean=("oracle_peid_unique_x", "mean"),
        oracle_peid_unique_x_std=("oracle_peid_unique_x", "std"),
        oracle_peid_unique_y_mean=("oracle_peid_unique_y", "mean"),
        oracle_peid_unique_y_std=("oracle_peid_unique_y", "std"),
        oracle_peid_xy_joint_mean=("oracle_peid_xy_joint", "mean"),
        oracle_peid_xy_joint_std=("oracle_peid_xy_joint", "std"),
        oracle_peid_xy_synergy_mean=("oracle_peid_xy_synergy", "mean"),
        oracle_peid_xy_synergy_std=("oracle_peid_xy_synergy", "std"),
    )
    summary = (
        frame.groupby(["alpha", "k"], as_index=False)
        .agg(**aggregations)
        .sort_values(["alpha", "k"])
        .reset_index(drop=True)
    )
    summary_by_alpha = (
        summary.groupby("alpha", as_index=False)
        .agg(
            mlp_peid_xy_synergy_mean=("mlp_peid_xy_synergy_mean", "mean"),
            mlp_peid_xy_synergy_std=("mlp_peid_xy_synergy_mean", "std"),
            oracle_peid_xy_synergy_mean=("oracle_peid_xy_synergy_mean", "mean"),
            oracle_peid_xy_synergy_std=("oracle_peid_xy_synergy_mean", "std"),
            z_train_r2_mean=("z_train_r2_mean", "mean"),
            z_train_r2_std=("z_train_r2_mean", "std"),
        )
        .sort_values("alpha")
        .reset_index(drop=True)
    )
    summary_by_k = (
        summary.groupby("k", as_index=False)
        .agg(
            mlp_peid_xy_synergy_mean=("mlp_peid_xy_synergy_mean", "mean"),
            mlp_peid_xy_synergy_std=("mlp_peid_xy_synergy_mean", "std"),
            oracle_peid_xy_synergy_mean=("oracle_peid_xy_synergy_mean", "mean"),
            oracle_peid_xy_synergy_std=("oracle_peid_xy_synergy_mean", "std"),
            z_train_r2_mean=("z_train_r2_mean", "mean"),
            z_train_r2_std=("z_train_r2_mean", "std"),
        )
        .sort_values("k")
        .reset_index(drop=True)
    )
    return {
        "config": {
            "alpha_values": [float(value) for value in alpha_values],
            "k_values": [float(value) for value in k_values],
            "seeds": [int(value) for value in seeds],
            "n_samples": int(n_samples),
            "noise": float(noise),
            "mlp_epochs": int(mlp_epochs),
            "intervention_samples": int(intervention_samples),
            "bins": int(bins),
            "variables": list(VARIABLES),
            "confounders": [],
            "oracle_intervention_support": {
                name: [float(bound) for bound in oracle_support[name]]
                for name in VARIABLES
            },
            "oracle_intervention_seed": int(oracle_intervention_seed),
        },
        "units": {"mlp_peid": "bits", "oracle_peid": "bits"},
        "runs": rows,
        "summary": summary.to_dict("records"),
        "summary_by_alpha": summary_by_alpha.to_dict("records"),
        "summary_by_k": summary_by_k.to_dict("records"),
        "trend": _trend(summary),
    }


def plot_sine_frequency_sweep(result: dict[str, object], figure_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(result["summary"]).sort_values(["alpha", "k"])
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)
    ax_alpha, ax_k, ax_fit = axes
    colors = ["#4C78A8", "#1B9E77", "#E68613", "#6F5AA7", "#D65F8F", "#8C8C8C"]
    markers = ["o", "D", "s", "^", "v", "P"]

    def error_line(
        ax,
        x: np.ndarray,
        y: np.ndarray,
        std: np.ndarray,
        *,
        label: str,
        color: str,
        marker: str,
        linestyle: str = "-",
    ) -> None:
        ax.plot(
            x,
            y,
            marker=marker,
            color=color,
            linewidth=2.0,
            linestyle=linestyle,
            markersize=5.8,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=label,
        )
        if np.any(std > 0.0):
            ax.errorbar(
                x,
                y,
                yerr=std,
                fmt="none",
                ecolor=color,
                elinewidth=0.8,
                capsize=2.2,
                capthick=0.8,
                alpha=0.55,
            )

    for idx, (k, group) in enumerate(frame.groupby("k")):
        ordered = group.sort_values("alpha")
        error_line(
            ax_alpha,
            ordered["alpha"].to_numpy(dtype=float),
            ordered["mlp_peid_xy_synergy_mean"].to_numpy(dtype=float),
            ordered["mlp_peid_xy_synergy_std"].fillna(0.0).to_numpy(dtype=float),
            label=f"k={float(k):g}",
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
        )
    ax_alpha.axhline(0.0, color="#6b7280", linestyle="--", linewidth=0.9)
    ax_alpha.set_xlabel("alpha in alpha sin(k x y)")
    ax_alpha.set_ylabel("MLP+PEID Syn(x,y) (bits)")
    ax_alpha.grid(alpha=0.18, linewidth=0.5)
    ax_alpha.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    for idx, (alpha, group) in enumerate(frame.groupby("alpha")):
        ordered = group.sort_values("k")
        error_line(
            ax_k,
            ordered["k"].to_numpy(dtype=float),
            ordered["mlp_peid_xy_synergy_mean"].to_numpy(dtype=float),
            ordered["mlp_peid_xy_synergy_std"].fillna(0.0).to_numpy(dtype=float),
            label=f"alpha={float(alpha):g}",
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
        )
    ax_k.axhline(0.0, color="#6b7280", linestyle="--", linewidth=0.9)
    ax_k.set_xlabel("frequency k in sin(k x y)")
    ax_k.set_ylabel("MLP+PEID Syn(x,y) (bits)")
    ax_k.grid(alpha=0.18, linewidth=0.5)
    ax_k.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    for idx, (alpha, group) in enumerate(frame.groupby("alpha")):
        ordered = group.sort_values("k")
        error_line(
            ax_fit,
            ordered["k"].to_numpy(dtype=float),
            ordered["z_train_r2_mean"].to_numpy(dtype=float),
            ordered["z_train_r2_std"].fillna(0.0).to_numpy(dtype=float),
            label=f"alpha={float(alpha):g}",
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
        )
    ax_fit.axhline(0.0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax_fit.set_xlabel("frequency k in sin(k x y)")
    ax_fit.set_ylabel("MLP z train R2")
    ax_fit.grid(alpha=0.18, linewidth=0.5)
    ax_fit.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def _fmt(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.4g}"


def write_sine_frequency_report(
    result: dict[str, object],
    figure_path: Path,
    report_path: Path,
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fig_rel = os.path.relpath(figure_path, start=report_path.parent)
    config = dict(result["config"])
    summary = list(result["summary"])
    trend = dict(result.get("trend", {}))
    table_lines = [
        "| alpha | k | z train R2 | Ux->z | Uy->z | Joint EI | MLP Syn | Oracle Syn |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        table_lines.append(
            "| {alpha:.2f} | {k:.1f} | {r2} | {ux} | {uy} | {joint} | {syn} | {oracle} |".format(
                alpha=float(row["alpha"]),
                k=float(row["k"]),
                r2=_fmt(row["z_train_r2_mean"]),
                ux=_fmt(row["mlp_peid_unique_x_mean"]),
                uy=_fmt(row["mlp_peid_unique_y_mean"]),
                joint=_fmt(row["mlp_peid_xy_joint_mean"]),
                syn=_fmt(row["mlp_peid_xy_synergy_mean"]),
                oracle=_fmt(row["oracle_peid_xy_synergy_mean"]),
            )
        )
    alpha_trend_lines = [
        "| fixed k | MLP Syn slope / alpha | Oracle Syn slope / alpha | R2 slope / alpha |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in trend.get("alpha_slope_by_k", []):
        alpha_trend_lines.append(
            "| {k:.1f} | {mlp} | {oracle} | {r2} |".format(
                k=float(row["k"]),
                mlp=_fmt(row.get("mlp_peid_synergy_slope_per_alpha")),
                oracle=_fmt(row.get("oracle_peid_synergy_slope_per_alpha")),
                r2=_fmt(row.get("z_train_r2_slope_per_alpha")),
            )
        )
    k_trend_lines = [
        "| fixed alpha | MLP Syn slope / k | Oracle Syn slope / k | R2 slope / k |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in trend.get("k_slope_by_alpha", []):
        k_trend_lines.append(
            "| {alpha:.2f} | {mlp} | {oracle} | {r2} |".format(
                alpha=float(row["alpha"]),
                mlp=_fmt(row.get("mlp_peid_synergy_slope_per_k")),
                oracle=_fmt(row.get("oracle_peid_synergy_slope_per_k")),
                r2=_fmt(row.get("z_train_r2_slope_per_k")),
            )
        )
    text = f"""# 无 Confounder 高频 Sine 的 MLP+PEID 识别实验

本实验单独检验没有共同驱动、没有隐藏变量时，`alpha` 与 `k` 分别改变后，MLP+PEID 的 `{{x,y}} -> z` 协同读数如何变化。

$$
\\begin{{aligned}}
x_{{t+1}} &= 0.42x_t + \\eta^x_t,\\\\
y_{{t+1}} &= 0.38y_t + \\eta^y_t,\\\\
z_{{t+1}} &= 0.22z_t + \\alpha\\sin(kx_ty_t) + \\eta^z_t.
\\end{{aligned}}
$$

这里 `alpha` 控制 sine 项幅度，`k` 控制 sine 项频率。拟合时只使用 `[x_t,y_t,z_t] -> [x_{{t+1}},y_{{t+1}},z_{{t+1}}]`。

## 实验设置

- alpha 网格：`{config["alpha_values"]}`
- k 网格：`{config["k_values"]}`
- seeds：`{config["seeds"]}`
- 每条轨迹样本数：`{config["n_samples"]}`
- noise：`{config["noise"]}`
- MLP epochs：`{config["mlp_epochs"]}`
- PEID 干预样本数：`{config["intervention_samples"]}`
- 变量：`{config["variables"]}`；confounders：`{config["confounders"]}`

![无 confounder sine frequency sweep]({fig_rel})

## 数值结果

{chr(10).join(table_lines)}

## 趋势斜率

固定 `k` 时看 `alpha` 方向：

{chr(10).join(alpha_trend_lines)}

固定 `alpha` 时看 `k` 方向：

{chr(10).join(k_trend_lines)}

## 读数

总体上，`alpha` 方向主要改变 sine 项幅度，因此更接近协同强度扫描；`k` 方向改变响应面的振荡频率，因此更像可学习性/采样分辨率压力测试。解释高频条件下 Syn 的下降时，需要同时看右侧 R2 面板：若 R2 仍高而 Syn 下降，说明 PEID 读数对高频映射本身更敏感；若 R2 同时下降，则说明 learned surrogate 已经开始难以拟合短周期响应面。
"""
    report_path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return report_path


def _parse_float_values(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("value list must contain at least one numeric value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--alpha-values", type=_parse_float_values, default=(0.25, 0.5, 1.0, 1.5, 2.0))
    parser.add_argument("--k-values", type=_parse_float_values, default=(1.0, 2.0, 4.0, 6.0, 8.0, 10.0))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_sine_frequency_mlp_peid_sweep(alpha_values=args.alpha_values, k_values=args.k_values)
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    figure_path = plot_sine_frequency_sweep(result, args.figure_path)
    report_path = write_sine_frequency_report(result, figure_path, args.report_path)
    print(
        json.dumps(
            {
                "result_path": str(args.result_path),
                "figure_path": str(figure_path),
                "report_path": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
