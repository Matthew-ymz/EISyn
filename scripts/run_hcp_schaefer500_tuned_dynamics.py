#!/usr/bin/env python3
"""Tune state and residual Ridge VAR forecasts for Schaefer-500 time series.

This runner is deliberately prediction-only.  It does not recompute or reinterpret
the existing 500-dimensional one-step PhiEID baseline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import solve
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30"
DEFAULT_OUTPUT = ROOT / "results" / "hcp_schaefer500_tuned_dynamics" / "summary.json"
DEFAULT_LOG_DIR = ROOT / "docs" / "log" / "hcp_schaefer500_tuned_dynamics"
DEFAULT_REPORT = DEFAULT_LOG_DIR / "run_report.md"
DEFAULT_ALPHAS = tuple(float(10.0**power) for power in range(-4, 7))
DEFAULT_ORDERS = (1, 2, 3, 5, 8)
# With p=8 the design has 4,000 predictors but fewer than 900 training rows.
# The dual solution factors an n_samples × n_samples matrix once and solves all
# 500 targets together, avoiding both a 4,000 × 4,000 primal Gram matrix and
# LSQR's separate iterative solve for every target.
RIDGE_BACKEND = "dual"


@dataclass(frozen=True)
class TuningConfig:
    development_end: int = 900
    fold_validation_size: int = 100
    fold_train_ends: tuple[int, ...] = (600, 700, 800)
    alphas: tuple[float, ...] = DEFAULT_ALPHAS
    orders: tuple[int, ...] = DEFAULT_ORDERS


def load_subject_series(path: Path, *, expected_parcels: int = 500) -> np.ndarray:
    values = np.asarray(loadmat(path)["Schaefer500"], dtype=float)
    if values.ndim != 2 or values.shape[1] != expected_parcels:
        raise ValueError(f"Expected [time, {expected_parcels}] Schaefer500 data, got {values.shape} in {path}.")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite Schaefer500 values in {path}.")
    return values


def make_history_samples(series: np.ndarray, *, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return [x_t, ..., x_{t-order+1}] rows paired with x_{t+1}."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 2:
        raise ValueError("series must have shape [time, parcel].")
    if order < 1:
        raise ValueError("order must be positive.")
    if values.shape[0] <= order:
        raise ValueError("series is too short for requested history order.")
    history = np.concatenate(
        [values[order - 1 - lag : values.shape[0] - 1 - lag] for lag in range(order)], axis=1
    )
    return history, values[order:]


def reconstruct_next_state(current: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return np.asarray(current, dtype=float) + np.asarray(delta, dtype=float)


def _dual_ridge_predict(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, *, alpha: float) -> np.ndarray:
    """Multi-target Ridge prediction using the sample-space dual formulation."""
    x_mean = train_x.mean(axis=0, keepdims=True)
    y_mean = train_y.mean(axis=0, keepdims=True)
    centered_x = train_x - x_mean
    centered_y = train_y - y_mean
    kernel = centered_x @ centered_x.T
    kernel.flat[:: kernel.shape[0] + 1] += float(alpha)
    dual_weights = solve(kernel, centered_y, assume_a="pos", check_finite=False)
    return (eval_x - x_mean) @ centered_x.T @ dual_weights + y_mean


def _state_scale(current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(current, dtype=float).mean(axis=0, keepdims=True)
    scale = np.asarray(current, dtype=float).std(axis=0, ddof=1, keepdims=True)
    return mean, np.where(scale > 1.0e-12, scale, 1.0)


def _standardize_history(history: np.ndarray, *, state_mean: np.ndarray, state_scale: np.ndarray, n_parcels: int) -> np.ndarray:
    blocks = [
        (history[:, start : start + n_parcels] - state_mean) / state_scale
        for start in range(0, history.shape[1], n_parcels)
    ]
    return np.concatenate(blocks, axis=1)


def _metrics(true_state: np.ndarray, predicted_state: np.ndarray, current_state: np.ndarray, *, state_mean: np.ndarray, state_scale: np.ndarray) -> dict[str, float]:
    true_z = (np.asarray(true_state, dtype=float) - state_mean) / state_scale
    predicted_z = (np.asarray(predicted_state, dtype=float) - state_mean) / state_scale
    persistence_z = (np.asarray(current_state, dtype=float) - state_mean) / state_scale
    error = true_z - predicted_z
    persistence_error = true_z - persistence_z
    rmse = float(np.sqrt(np.mean(error**2)))
    persistence_rmse = float(np.sqrt(np.mean(persistence_error**2)))
    corr = float(np.corrcoef(true_z.reshape(-1), predicted_z.reshape(-1))[0, 1])
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(error))),
        "persistence_rmse": persistence_rmse,
        "skill_ratio": float(rmse / max(persistence_rmse, 1.0e-12)),
        "corr": corr if np.isfinite(corr) else 0.0,
    }


def _fit_and_predict(
    train_history: np.ndarray,
    train_next: np.ndarray,
    eval_history: np.ndarray,
    *,
    target_mode: str,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target_mode not in {"state", "delta"}:
        raise ValueError("target_mode must be 'state' or 'delta'.")
    n_parcels = train_next.shape[1]
    train_current = train_history[:, :n_parcels]
    eval_current = eval_history[:, :n_parcels]
    state_mean, state_scale = _state_scale(train_current)
    train_x = _standardize_history(
        train_history, state_mean=state_mean, state_scale=state_scale, n_parcels=n_parcels
    )
    eval_x = _standardize_history(eval_history, state_mean=state_mean, state_scale=state_scale, n_parcels=n_parcels)
    if target_mode == "state":
        target = train_next
        target_mean, target_scale = state_mean, state_scale
    else:
        target = train_next - train_current
        target_mean, target_scale = _state_scale(target)
    train_y = (target - target_mean) / target_scale
    prediction = _dual_ridge_predict(train_x, train_y, eval_x, alpha=alpha) * target_scale + target_mean
    if target_mode == "delta":
        prediction = reconstruct_next_state(eval_current, prediction)
    return prediction, state_mean, state_scale


def evaluate_split(
    series: np.ndarray,
    *,
    target_mode: str,
    alpha: float,
    order: int,
    train_end: int,
    evaluation_start: int,
    evaluation_end: int,
) -> dict[str, float]:
    history, next_state = make_history_samples(series, order=order)
    target_indices = np.arange(order, np.asarray(series).shape[0])
    train_mask = target_indices < int(train_end)
    evaluation_mask = (target_indices >= int(evaluation_start)) & (target_indices < int(evaluation_end))
    if not train_mask.any() or not evaluation_mask.any():
        raise ValueError("Requested split has no train or evaluation rows.")
    prediction, state_mean, state_scale = _fit_and_predict(
        history[train_mask], next_state[train_mask], history[evaluation_mask], target_mode=target_mode, alpha=alpha
    )
    return _metrics(
        next_state[evaluation_mask],
        prediction,
        history[evaluation_mask, : next_state.shape[1]],
        state_mean=state_mean,
        state_scale=state_scale,
    )


def _candidate_sort_key(candidate: dict[str, object]) -> tuple[float, int, float]:
    selected = candidate["selected"]
    return (
        float(candidate["mean_validation_skill_ratio"]),
        int(selected["order"]),
        -float(selected["alpha"]),
    )


def select_configuration(series: np.ndarray, *, target_mode: str, config: TuningConfig) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for order in config.orders:
        for alpha in config.alphas:
            folds = [
                evaluate_split(
                    series,
                    target_mode=target_mode,
                    alpha=alpha,
                    order=order,
                    train_end=train_end,
                    evaluation_start=train_end,
                    evaluation_end=train_end + config.fold_validation_size,
                )
                for train_end in config.fold_train_ends
            ]
            candidates.append(
                {
                    "selected": {"alpha": float(alpha), "order": int(order)},
                    "mean_validation_skill_ratio": float(np.mean([fold["skill_ratio"] for fold in folds])),
                    "folds": folds,
                }
            )
    candidates.sort(key=_candidate_sort_key)
    return {"selected": candidates[0]["selected"], "candidates": candidates}


def _test_selected(series: np.ndarray, *, target_mode: str, selected: dict[str, object], config: TuningConfig) -> dict[str, float]:
    return evaluate_split(
        series,
        target_mode=target_mode,
        alpha=float(selected["alpha"]),
        order=int(selected["order"]),
        train_end=config.development_end,
        evaluation_start=config.development_end,
        evaluation_end=np.asarray(series).shape[0],
    )


def _selection_config(config: TuningConfig, *, orders: Iterable[int]) -> TuningConfig:
    return replace(config, orders=tuple(int(order) for order in orders))


def analyze_subject(series: np.ndarray, *, subject: str, config: TuningConfig) -> dict[str, object]:
    baseline_selected = {"alpha": 1.0, "order": 1}
    direct_selection = select_configuration(series, target_mode="state", config=_selection_config(config, orders=(1,)))
    delta_p1_selection = select_configuration(series, target_mode="delta", config=_selection_config(config, orders=(1,)))
    delta_history_selection = select_configuration(series, target_mode="delta", config=config)
    models = {
        "fixed_state_ridge_p1_alpha1": {
            "selected": baseline_selected,
            "test": _test_selected(series, target_mode="state", selected=baseline_selected, config=config),
        },
        "tuned_state_ridge_p1": {
            **direct_selection,
            "test": _test_selected(series, target_mode="state", selected=direct_selection["selected"], config=config),
        },
        "tuned_delta_ridge_p1": {
            **delta_p1_selection,
            "test": _test_selected(series, target_mode="delta", selected=delta_p1_selection["selected"], config=config),
        },
        "tuned_delta_ridge_history": {
            **delta_history_selection,
            "test": _test_selected(series, target_mode="delta", selected=delta_history_selection["selected"], config=config),
        },
    }
    baseline_skill = float(models["fixed_state_ridge_p1_alpha1"]["test"]["skill_ratio"])
    for name, payload in models.items():
        payload["test"]["skill_improvement_vs_fixed_baseline"] = baseline_skill - float(payload["test"]["skill_ratio"])
    return {
        "subject": subject,
        "shape": list(np.asarray(series).shape),
        "explicit_preprocessing": "none",
        "phi_eid": "unchanged_existing_500d_onestep_baseline_not_recomputed",
        "models": models,
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    names = list(rows[0]["models"].keys())
    models = {}
    for name in names:
        skills = np.asarray([row["models"][name]["test"]["skill_ratio"] for row in rows], dtype=float)
        models[name] = {
            "test_skill_ratio_mean": float(skills.mean()),
            "test_skill_ratio_median": float(np.median(skills)),
            "subjects_better_than_persistence": int(np.sum(skills < 1.0)),
        }
        selected = [row["models"][name]["selected"] for row in rows if "selected" in row["models"][name]]
        if selected:
            alpha_counts: dict[str, int] = {}
            order_counts: dict[str, int] = {}
            for choice in selected:
                alpha_key = str(float(choice["alpha"]))
                order_key = str(int(choice["order"]))
                alpha_counts[alpha_key] = alpha_counts.get(alpha_key, 0) + 1
                order_counts[order_key] = order_counts.get(order_key, 0) + 1
            models[name]["selected_alpha_counts"] = alpha_counts
            models[name]["selected_order_counts"] = order_counts
        if name != "fixed_state_ridge_p1_alpha1":
            improvements = np.asarray(
                [row["models"][name]["test"]["skill_improvement_vs_fixed_baseline"] for row in rows], dtype=float
            )
            models[name]["skill_improvement_vs_fixed_baseline_mean"] = float(improvements.mean())
            models[name]["skill_improvement_vs_fixed_baseline_median"] = float(np.median(improvements))
    return {"models": models}


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_live_status(log_dir: Path, *, status: str, running: str, recent: str, next_step: str) -> Path:
    return _write(
        log_dir / "live_status.md",
        "\n".join(
            [
                "# 实时状态",
                "",
                "## 当前状态",
                f"- {status}",
                "",
                "## 正在运行",
                f"- {running}",
                "",
                "## 最近结果",
                f"- {recent}",
                "",
                "## 下一步",
                f"- {next_step}",
                "",
                "## 监控文件",
                "- run history: `run_history.jsonl`",
            ]
        )
        + "\n",
    )


def write_tuning_artifacts(summary: dict[str, object], *, log_dir: Path) -> dict[str, Path]:
    """Persist concise, resumable tuning metadata after a completed batch."""
    log_dir = Path(log_dir)
    aggregate_models = summary["aggregate"]["models"]
    ranking = sorted(
        aggregate_models.items(), key=lambda item: float(item[1]["test_skill_ratio_median"]))
    audit = _write(
        log_dir / "repo_audit.md",
        "# 仓库审计\n\n"
        "- Entry point: `scripts/run_hcp_schaefer500_tuned_dynamics.py`。\n"
        "- 数据: 每名 HCP Schaefer-500 序列为 1200 × 500。\n"
        "- 优化目标: 测试段 `RMSE / persistence_RMSE` 最小化。\n"
        "- PhiEID: 保留既有 500D one-step baseline，本实验不重算。\n",
    )
    search_plan = _write(
        log_dir / "search_plan.md",
        "# 搜索计划\n\n"
        "- 比较 fixed state Ridge、tuned state Ridge、delta Ridge(p=1) 与 delta Ridge(history)。\n"
        "- 每名被试在开发段 expanding-window validation 中独立选择参数。\n"
        "- 成功阈值：median skill ratio < 1 且至少 24/30 名被试优于 persistence。\n",
    )
    search_space = log_dir / "search_space.json"
    _write(search_space, json.dumps(summary["config"], indent=2, sort_keys=True) + "\n")
    leaderboard_lines = ["# Leaderboard", "", "| Model | Median test skill ratio | Better than persistence |", "|---|---:|---:|"]
    for name, metrics in ranking:
        leaderboard_lines.append(
            f"| {name} | {float(metrics['test_skill_ratio_median']):.6f} | {int(metrics['subjects_better_than_persistence'])}/{summary['n_subjects']} |"
        )
    leaderboard = _write(log_dir / "leaderboard.md", "\n".join(leaderboard_lines) + "\n")
    best_name, best_metrics = ranking[0]
    notes = _write(
        log_dir / "notes.md",
        "# 观察\n\n"
        f"- 当前最优模型为 `{best_name}`，median test skill ratio={float(best_metrics['test_skill_ratio_median']):.6f}。\n"
        "- 结果只描述预测泛化；不能据此改变或强化现有 PhiEID 结论。\n",
    )
    next_steps = _write(
        log_dir / "next_steps.md",
        "# 下一步\n\n"
        "- 若预先成功阈值达到，再单独设计与多阶历史状态可比的 PhiEID 估计。\n"
        "- 否则将此结果作为原始时序中 persistence 主导的证据，并在独立敏感性分析中考察去趋势和混杂控制。\n",
    )
    report = _write(
        log_dir / "tuning_report.md",
        "# Schaefer-500 动力学调优报告\n\n"
        f"本批次完成 {summary['n_subjects']} 名被试的时间嵌套 Ridge 调优。模型按测试集 median skill ratio 排序：\n\n"
        + "\n".join(leaderboard_lines[2:])
        + "\n\nPhiEID 维持既有 one-step 500D baseline，未由本预测实验重新计算。\n",
    )
    status = _write_live_status(
        log_dir,
        status="completed",
        running="无。",
        recent=f"最优 {best_name}，median skill ratio={float(best_metrics['test_skill_ratio_median']):.6f}。",
        next_step="阅读 `tuning_report.md` 与 `leaderboard.md` 后决定是否进入 PhiEID 敏感性分析。",
    )
    history = log_dir / "run_history.jsonl"
    with history.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "run_name": "hcp_schaefer500_tuned_dynamics",
                    "status": "completed",
                    "objective": "minimize test RMSE / persistence_RMSE",
                    "metrics": {name: metrics for name, metrics in aggregate_models.items()},
                },
                sort_keys=True,
            )
            + "\n"
        )
    return {
        "audit": audit,
        "search_plan": search_plan,
        "search_space": search_space,
        "status": status,
        "leaderboard": leaderboard,
        "notes": notes,
        "next_steps": next_steps,
        "report": report,
        "history": history,
    }


def write_experiment_report(summary: dict[str, object], *, report_path: Path) -> Path:
    models = summary["aggregate"]["models"]
    lines = [
        "# HCP Schaefer-500 动力学调优报告",
        "",
        "本实验仅调优下一状态预测；既有 500D one-step $\\Phi^{EID}$ 基线未重算或重新解释。",
        "",
        "| 模型 | Test skill ratio（中位数） | 优于 persistence |",
        "|---|---:|---:|",
    ]
    for name, metrics in sorted(models.items(), key=lambda item: float(item[1]["test_skill_ratio_median"])):
        lines.append(
            f"| `{name}` | {float(metrics['test_skill_ratio_median']):.6f} | {int(metrics['subjects_better_than_persistence'])}/{summary['n_subjects']} |"
        )
    lines.extend(
        [
            "",
            "预先成功阈值为 median skill ratio < 1，且至少 24/30 名被试优于 persistence。",
            "所有参数选择均在每名被试前 900 点的 expanding-window validation 中完成，后 300 点仅用于最终测试。",
        ]
    )
    return _write(Path(report_path), "\n".join(lines) + "\n")


def run(
    data_root: Path,
    output: Path,
    *,
    config: TuningConfig,
    subjects: Sequence[str] | None = None,
    log_dir: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, object]:
    files = sorted(data_root.glob("sub-*/*.mat"))
    wanted = None if subjects is None else set(subjects)
    if wanted is not None:
        files = [path for path in files if path.parent.name in wanted]
    if not files:
        raise FileNotFoundError(f"No subject MAT files found under {data_root}.")
    if log_dir is not None:
        _write_live_status(
            Path(log_dir),
            status="running",
            running=f"Schaefer-500 nested temporal tuning: 0/{len(files)} subjects。",
            recent="尚无结果。",
            next_step="完成当前被试后记录 run history，并继续剩余被试。",
        )
    rows = []
    for index, path in enumerate(files, start=1):
        subject = path.parent.name
        print(f"[{index}/{len(files)}] {subject}", flush=True)
        rows.append(analyze_subject(load_subject_series(path), subject=subject, config=config))
        if log_dir is not None:
            with (Path(log_dir) / "run_history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"run_name": subject, "status": "completed", "models": rows[-1]["models"]}, sort_keys=True) + "\n")
            _write_live_status(
                Path(log_dir),
                status="running",
                running=f"Schaefer-500 nested temporal tuning: {index}/{len(files)} subjects。",
                recent=f"{subject} 已完成。",
                next_step="继续下一个被试。",
            )
    summary = {
        "data_root": str(data_root),
        "n_subjects": len(rows),
        "config": {
            "explicit_preprocessing": "none",
            "phi_eid": "unchanged_existing_500d_onestep_baseline_not_recomputed",
            "development_end": config.development_end,
            "test_start": config.development_end,
            "fold_train_ends": list(config.fold_train_ends),
            "fold_validation_size": config.fold_validation_size,
            "alphas": list(config.alphas),
            "orders": list(config.orders),
        },
        "aggregate": _aggregate(rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if report_path is not None:
        write_experiment_report(summary, report_path=Path(report_path))
    if log_dir is not None:
        write_tuning_artifacts(summary, log_dir=Path(log_dir))
    return summary


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated sub-<id> list.")
    parser.add_argument("--development-end", type=int, default=900)
    parser.add_argument("--fold-validation-size", type=int, default=100)
    parser.add_argument("--fold-train-ends", type=_parse_int_list, default=(600, 700, 800))
    parser.add_argument("--alphas", type=_parse_float_list, default=DEFAULT_ALPHAS)
    parser.add_argument("--orders", type=_parse_int_list, default=DEFAULT_ORDERS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TuningConfig(
        development_end=int(args.development_end),
        fold_validation_size=int(args.fold_validation_size),
        fold_train_ends=tuple(args.fold_train_ends),
        alphas=tuple(args.alphas),
        orders=tuple(args.orders),
    )
    subjects = None if args.subjects is None else tuple(part.strip() for part in args.subjects.split(",") if part.strip())
    summary = run(args.data_root, args.output, config=config, subjects=subjects, log_dir=args.log_dir, report_path=args.report)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
