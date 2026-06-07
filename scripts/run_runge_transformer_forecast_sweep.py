#!/usr/bin/env python3
"""Stage and resume Transformer forecast sweeps for Runge component dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_runge_rnn_forecast_comparison import (  # noqa: E402
    RungeRnnForecastConfig,
    _json_sanitize,
    add_best_baseline_rows,
    compute_prediction_significance,
    dependency_versions,
    evaluate_multistep,
    parse_float_tuple,
    parse_history_grid,
    parse_int_tuple,
    rank_leaderboard,
    run,
    save_metric_plot,
)
from scripts import run_runge_pairwise_mlp_ei as pairwise  # noqa: E402


RESULT_SUBDIR = Path("results/runge_transformer_forecast_sweep")
FIG_SUBDIR = Path("fig/runge/transformer_forecast_sweep")
REPORT_PATH = Path("docs/runge_transformer_forecast_sweep_report.md")

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


def _slug_float(value: float) -> str:
    text = f"{float(value):.6g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "").replace("e", "e")


def candidate_name(spec: dict[str, object]) -> str:
    keys = [
        ("stage", str(spec["stage"])),
        ("h", int(spec["history"])),
        ("d", int(spec["hidden_dim"])),
        ("nh", int(spec["transformer_nhead"])),
        ("l", int(spec["num_layers"])),
        ("ff", int(spec["transformer_dim_feedforward"])),
        ("pool", str(spec["transformer_pooling"])),
        ("pos", str(spec["transformer_positional_encoding"])),
        ("obj", str(spec["rnn_objective"])),
        ("skip", str(spec["skip_mode"])),
        ("lr", _slug_float(float(spec["learning_rate"]))),
        ("do", _slug_float(float(spec["dropout"]))),
        ("wd", _slug_float(float(spec["weight_decay"]))),
        ("seed", int(spec["seed"])),
    ]
    raw = "_".join(f"{key}{value}" for key, value in keys)
    digest = hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:8]
    return f"{raw}_{digest}"


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_sanitize(payload), sort_keys=True, allow_nan=False) + "\n")


def _read_existing_rows(log_path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    if not log_path.exists():
        return rows
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("status") == "completed":
            rows[str(payload["candidate"])] = payload
    return rows


def _overall_summary(metrics: pd.DataFrame, *, model: str, prefix: str) -> dict[str, float]:
    overall = metrics[(metrics["component"] == "overall") & (metrics["model"] == model)].copy()
    values = {
        f"{prefix}_avg_rmse": float(overall["rmse"].mean()),
        f"{prefix}_avg_corr": float(overall["corr"].mean()),
    }
    for horizon in sorted(metrics["horizon"].dropna().unique()):
        rows = overall[overall["horizon"] == horizon]
        values[f"{prefix}_h{int(horizon)}_rmse"] = float(rows.iloc[0]["rmse"]) if not rows.empty else float("nan")
    return values


def _candidate_row(candidate: str, spec: dict[str, object], artifacts: dict[str, Path], elapsed_s: float) -> dict[str, object]:
    validation = pd.read_csv(artifacts["validation_metrics"])
    test = pd.read_csv(artifacts["metrics"])
    manifest = json.loads(Path(artifacts["manifest"]).read_text(encoding="utf-8"))
    model = str(manifest.get("primary_model", "Transformer"))
    row: dict[str, object] = {
        "candidate": candidate,
        "status": "completed",
        "elapsed_s": float(elapsed_s),
        "result_dir": str(artifacts["result_dir"]),
        "fig_dir": str(artifacts["fig_dir"]),
        "manifest": str(artifacts["manifest"]),
        "metrics": str(artifacts["metrics"]),
        "validation_metrics": str(artifacts["validation_metrics"]),
        "forecast_arrays": str(artifacts["forecast_arrays"]),
        "primary_model": model,
        "best_linear_alpha": float(manifest["best_linear_alpha"]),
        **spec,
    }
    row.update(_overall_summary(validation, model=model, prefix="val"))
    row.update(_overall_summary(test, model=model, prefix="test"))
    return row


def _base_spec(args: argparse.Namespace) -> dict[str, object]:
    return {
        "history": int(args.lag),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "weight_decay": float(args.weight_decay),
        "learning_rate": float(args.learning_rate),
        "transformer_nhead": int(args.transformer_nhead),
        "transformer_dim_feedforward": int(args.transformer_dim_feedforward),
        "transformer_pooling": str(args.transformer_pooling),
        "transformer_positional_encoding": str(args.transformer_positional_encoding),
        "rnn_objective": str(args.rnn_objective),
        "skip_mode": "frozen_linear_skip" if args.use_linear_skip and args.freeze_linear_skip else "trainable_linear_skip"
        if args.use_linear_skip
        else "no_skip",
        "seed": int(args.seed),
    }


def _valid_heads(hidden_dim: int, heads: Iterable[int]) -> list[int]:
    return [int(head) for head in heads if int(head) > 0 and int(hidden_dim) % int(head) == 0]


def build_candidate_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    histories = parse_history_grid(args.history_grid)
    base = _base_spec(args)
    specs: list[dict[str, object]] = []

    for history in histories:
        spec = dict(base, stage="A_history", history=int(history))
        specs.append(spec)

    for history in histories[: min(3, len(histories))]:
        for hidden_dim, default_head in [(64, 4), (128, 4), (192, 4), (256, 8)]:
            for layers in [1, 2]:
                for ff_mult in [2, 4]:
                    head = default_head if hidden_dim % default_head == 0 else _valid_heads(hidden_dim, [2, 4, 8])[0]
                    specs.append(
                        dict(
                            base,
                            stage="B_capacity",
                            history=int(history),
                            hidden_dim=int(hidden_dim),
                            transformer_nhead=int(head),
                            num_layers=int(layers),
                            transformer_dim_feedforward=int(hidden_dim * ff_mult),
                        )
                    )

    refine_seed = specs[:8]
    for parent in refine_seed:
        for head in _valid_heads(int(parent["hidden_dim"]), [2, 4, 8]):
            for pooling in ["last", "mean"]:
                for pos in ["learned", "sinusoidal"]:
                    specs.append(
                        dict(
                            parent,
                            stage="C_pool_pos",
                            transformer_nhead=int(head),
                            transformer_pooling=pooling,
                            transformer_positional_encoding=pos,
                        )
                    )

    reg_seed = specs[:8]
    for parent in reg_seed:
        for lr in [1e-4, 3e-4, 8e-4, 1.5e-3]:
            specs.append(dict(parent, stage="D_lr", learning_rate=float(lr)))
        for dropout in [0.0, 0.05, 0.1, 0.2]:
            specs.append(dict(parent, stage="D_dropout", dropout=float(dropout)))
        for wd in [0.0, 1e-6, 1e-5, 1e-4, 1e-3]:
            specs.append(dict(parent, stage="D_weight_decay", weight_decay=float(wd)))

    objective_seed = specs[:6]
    for parent in objective_seed:
        for objective in ["direct_multihorizon", "rollout_multistep"]:
            for skip_mode in ["frozen_linear_skip", "trainable_linear_skip", "no_skip"]:
                specs.append(dict(parent, stage="E_objective_skip", rnn_objective=objective, skip_mode=skip_mode))

    final_seed = specs[:3]
    for parent in final_seed:
        for seed in parse_history_grid(args.final_seeds):
            specs.append(dict(parent, stage="F_seed", seed=int(seed)))

    unique: dict[str, dict[str, object]] = {}
    for spec in specs:
        unique.setdefault(candidate_name(spec), spec)
    selected = list(unique.values())
    if args.max_candidates is not None:
        selected = selected[: max(0, int(args.max_candidates))]
    return selected


def _spec_from_row(row: dict[str, object] | pd.Series, *, stage: str | None = None) -> dict[str, object]:
    data = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    spec = {
        "stage": str(stage if stage is not None else data["stage"]),
        "history": int(data["history"]),
        "hidden_dim": int(data["hidden_dim"]),
        "num_layers": int(data["num_layers"]),
        "dropout": float(data["dropout"]),
        "weight_decay": float(data["weight_decay"]),
        "learning_rate": float(data["learning_rate"]),
        "transformer_nhead": int(data["transformer_nhead"]),
        "transformer_dim_feedforward": int(data["transformer_dim_feedforward"]),
        "transformer_pooling": str(data["transformer_pooling"]),
        "transformer_positional_encoding": str(data["transformer_positional_encoding"]),
        "rnn_objective": str(data["rnn_objective"]),
        "skip_mode": str(data["skip_mode"]),
        "seed": int(data["seed"]),
    }
    return spec


def _config_for_candidate(args: argparse.Namespace, candidate: str, spec: dict[str, object]) -> RungeRnnForecastConfig:
    use_skip = str(spec["skip_mode"]) != "no_skip"
    freeze_skip = str(spec["skip_mode"]) == "frozen_linear_skip"
    return RungeRnnForecastConfig(
        component_scores=args.component_scores,
        output_dir=args.output_dir,
        result_subdir=RESULT_SUBDIR / "candidates" / candidate,
        fig_subdir=FIG_SUBDIR / "candidates" / candidate,
        lag=int(spec["history"]),
        horizons=parse_int_tuple(args.horizons),
        rnn_type="transformer",
        rnn_objective=str(spec["rnn_objective"]),
        device=str(args.device),
        hidden_dim=int(spec["hidden_dim"]),
        num_layers=int(spec["num_layers"]),
        dropout=float(spec["dropout"]),
        transformer_nhead=int(spec["transformer_nhead"]),
        transformer_dim_feedforward=int(spec["transformer_dim_feedforward"]),
        transformer_pooling=str(spec["transformer_pooling"]),
        transformer_positional_encoding=str(spec["transformer_positional_encoding"]),
        epochs=int(args.epochs),
        learning_rate=float(spec["learning_rate"]),
        batch_size=int(args.batch_size),
        weight_decay=float(spec["weight_decay"]),
        ridge_alphas=parse_float_tuple(args.ridge_alphas),
        use_linear_skip=bool(use_skip),
        freeze_linear_skip=bool(freeze_skip),
        residual_shrinkage=bool(args.residual_shrinkage),
        residual_gamma_min=float(args.residual_gamma_min),
        residual_gamma_max=float(args.residual_gamma_max),
        residual_gamma_steps=int(args.residual_gamma_steps),
        rnn_linear_blend_grid_steps=int(args.rnn_linear_blend_grid_steps),
        early_stopping_patience=int(args.early_stopping_patience),
        min_delta=float(args.min_delta),
        scheduler_patience=int(args.scheduler_patience),
        gradient_clip_norm=float(args.gradient_clip_norm),
        mlp_hidden_dim=int(args.mlp_hidden_dim),
        mlp_num_layers=int(args.mlp_num_layers),
        mlp_dropout=float(args.mlp_dropout),
        mlp_epochs=int(args.mlp_epochs),
        bootstrap_reps=int(args.bootstrap_reps),
        bootstrap_block_size=int(args.bootstrap_block_size),
        train_fraction=float(args.train_fraction),
        val_fraction=float(args.val_fraction),
        seed=int(spec["seed"]),
        force_retrain=bool(args.force_retrain),
    )


def _save_leaderboard_plot(leaderboard: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    for stage, frame in leaderboard.groupby("stage", sort=False):
        ax.scatter(frame["history"], frame["val_avg_rmse"], s=30, alpha=0.86, label=str(stage))
    ax.set_xlabel("History length (weeks)")
    ax.set_ylabel("Validation average RMSE")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_report(path: Path, leaderboard: pd.DataFrame, manifest: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Runge N=60 Transformer 预测调参报告",
        "",
        "## 当前结论",
        "",
    ]
    if leaderboard.empty:
        lines.append("尚无完成的 Transformer 候选。")
    else:
        best = leaderboard.iloc[0]
        lines.extend(
            [
                f"- 验证集选择的最佳候选：`{best['candidate']}`。",
                f"- 选择规则：只按 `{manifest['selection_rule']['rank_metric']}` 排序，不使用测试集选择。",
                f"- Validation average RMSE: {float(best['val_avg_rmse']):.6g}。",
                f"- Test average RMSE: {float(best['test_avg_rmse']):.6g}。",
                "- 是否显著优于基线需要查看 `final_prediction_significance.json`；本报告不把未验证提升表述为结论。",
            ]
        )
        lines.extend(
            [
                "",
                "## Top candidates",
                "",
                "| rank | candidate | stage | history | d_model | nhead | layers | val_avg_rmse | test_avg_rmse |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in leaderboard.head(10).to_dict("records"):
            lines.append(
                f"| {int(row['rank'])} | `{row['candidate']}` | {row['stage']} | {int(row['history'])} | "
                f"{int(row['hidden_dim'])} | {int(row['transformer_nhead'])} | {int(row['num_layers'])} | "
                f"{float(row['val_avg_rmse']):.6g} | {float(row['test_avg_rmse']):.6g} |"
            )
    lines.extend(
        [
            "",
            "## 产物",
            "",
            f"- Leaderboard: `{manifest['artifacts'].get('leaderboard', '')}`",
            f"- Final metrics: `{manifest['artifacts'].get('final_test_metrics', '')}`",
            f"- Significance: `{manifest['artifacts'].get('final_prediction_significance', '')}`",
            f"- Figure directory: `{manifest['artifacts'].get('fig_dir', '')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _array_forecasts(array_path: str | Path, *, split: str, model: str, horizons: Sequence[int]) -> dict[int, np.ndarray]:
    arrays = np.load(array_path)
    return {int(horizon): arrays[f"{split}_{model}_h{int(horizon)}"] for horizon in horizons}


def _array_targets(array_path: str | Path, *, split: str, horizons: Sequence[int]) -> dict[int, np.ndarray]:
    arrays = np.load(array_path)
    return {int(horizon): arrays[f"{split}_target_h{int(horizon)}"] for horizon in horizons}


def _average_forecasts(rows: pd.DataFrame, *, split: str, horizons: Sequence[int]) -> dict[int, np.ndarray]:
    averaged: dict[int, np.ndarray] = {}
    for horizon in horizons:
        preds = [
            np.load(row["forecast_arrays"])[f"{split}_Transformer_h{int(horizon)}"]
            for row in rows.to_dict("records")
        ]
        averaged[int(horizon)] = np.mean(preds, axis=0)
    return averaged


def _avg_rmse(forecasts: dict[int, np.ndarray], targets: dict[int, np.ndarray]) -> float:
    return float(np.mean([np.sqrt(np.mean((targets[horizon] - forecasts[horizon]) ** 2)) for horizon in sorted(targets)]))


def evaluate_ensembles(
    leaderboard: pd.DataFrame,
    *,
    result_dir: Path,
    horizons: Sequence[int],
    bootstrap_reps: int,
    block_size: int,
    seed: int,
) -> dict[str, object]:
    if leaderboard.empty:
        return {"selected": None, "leaderboard": None}
    selected_single = leaderboard.iloc[0]
    pool = leaderboard[leaderboard["rnn_objective"] == selected_single["rnn_objective"]].copy()
    if pool.empty:
        pool = leaderboard.copy()
    first_arrays = str(pool.iloc[0]["forecast_arrays"])
    targets = {
        "val": _array_targets(first_arrays, split="val", horizons=horizons),
        "test": _array_targets(first_arrays, split="test", horizons=horizons),
    }
    names = [f"component_{idx + 1:02d}" for idx in range(next(iter(targets["test"].values())).shape[1])]
    rows: list[dict[str, object]] = []
    forecast_payload: dict[str, np.ndarray] = {}
    for k in [2, 3, 5]:
        members = pool.head(k)
        if len(members) < k:
            continue
        val_forecasts = _average_forecasts(members, split="val", horizons=horizons)
        test_forecasts = _average_forecasts(members, split="test", horizons=horizons)
        row = {
            "system": f"TransformerEnsembleTop{k}",
            "k": int(k),
            "pool_objective": str(selected_single["rnn_objective"]),
            "member_candidates": ";".join(members["candidate"].astype(str).tolist()),
            "val_avg_rmse": _avg_rmse(val_forecasts, targets["val"]),
            "test_avg_rmse": _avg_rmse(test_forecasts, targets["test"]),
        }
        rows.append(row)
        for split, split_forecasts in [("val", val_forecasts), ("test", test_forecasts)]:
            for horizon, values in split_forecasts.items():
                forecast_payload[f"{split}_{row['system']}_h{int(horizon)}"] = np.asarray(values, dtype=np.float32)
                forecast_payload[f"{split}_target_h{int(horizon)}"] = np.asarray(targets[split][horizon], dtype=np.float32)
    if not rows:
        return {"selected": None, "leaderboard": None}
    ensemble_leaderboard = pd.DataFrame(rows).sort_values("val_avg_rmse", kind="mergesort").reset_index(drop=True)
    ensemble_leaderboard.insert(0, "rank", np.arange(1, len(ensemble_leaderboard) + 1, dtype=int))
    ensemble_path = result_dir / "ensemble_leaderboard.csv"
    ensemble_leaderboard.to_csv(ensemble_path, index=False)
    np.savez_compressed(result_dir / "ensemble_forecast_arrays.npz", **forecast_payload)
    selected = ensemble_leaderboard.iloc[0].to_dict()
    system = str(selected["system"])
    selected_members = pool.head(int(selected["k"]))
    test_forecasts = _average_forecasts(selected_members, split="test", horizons=horizons)
    baseline_forecasts = {
        "MLP": _array_forecasts(first_arrays, split="test", model="MLP", horizons=horizons),
        "TunedRidge": _array_forecasts(first_arrays, split="test", model="TunedRidge", horizons=horizons),
    }
    metric_frames = [
        evaluate_multistep(system, test_forecasts, targets["test"], names),
        evaluate_multistep("MLP", baseline_forecasts["MLP"], targets["test"], names),
        evaluate_multistep("TunedRidge", baseline_forecasts["TunedRidge"], targets["test"], names),
    ]
    metrics = pd.concat(metric_frames, ignore_index=True)
    significance = compute_prediction_significance(
        {system: test_forecasts, **baseline_forecasts},
        targets["test"],
        reps=int(bootstrap_reps),
        block_size=int(block_size),
        seed=int(seed),
        primary_model=system,
    )
    return {
        "selected": selected,
        "leaderboard": ensemble_path,
        "forecast_arrays": result_dir / "ensemble_forecast_arrays.npz",
        "metrics": metrics,
        "significance": significance,
    }


def evaluate_horizon_selector(
    leaderboard: pd.DataFrame,
    *,
    result_dir: Path,
    horizons: Sequence[int],
    bootstrap_reps: int,
    block_size: int,
    seed: int,
) -> dict[str, object]:
    if leaderboard.empty:
        return {"selected": None}
    selected_rows: list[dict[str, object]] = []
    forecast_payload: dict[str, np.ndarray] = {}
    selector_forecasts = {"val": {}, "test": {}}
    targets = {"val": {}, "test": {}}
    baseline_forecasts = {"MLP": {}, "TunedRidge": {}}
    for horizon in horizons:
        best: tuple[float, pd.Series] | None = None
        for _, row in leaderboard.iterrows():
            arrays = np.load(row["forecast_arrays"])
            pred = arrays[f"val_Transformer_h{int(horizon)}"]
            target = arrays[f"val_target_h{int(horizon)}"]
            val_rmse = float(np.sqrt(np.mean((target - pred) ** 2)))
            if best is None or val_rmse < best[0]:
                best = (val_rmse, row)
        assert best is not None
        row = best[1]
        arrays = np.load(row["forecast_arrays"])
        for split in ["val", "test"]:
            targets[split][int(horizon)] = arrays[f"{split}_target_h{int(horizon)}"]
            selector_forecasts[split][int(horizon)] = arrays[f"{split}_Transformer_h{int(horizon)}"]
            forecast_payload[f"{split}_TransformerHorizonSelector_h{int(horizon)}"] = np.asarray(
                selector_forecasts[split][int(horizon)],
                dtype=np.float32,
            )
            forecast_payload[f"{split}_target_h{int(horizon)}"] = np.asarray(targets[split][int(horizon)], dtype=np.float32)
        baseline_forecasts["MLP"][int(horizon)] = arrays[f"test_MLP_h{int(horizon)}"]
        baseline_forecasts["TunedRidge"][int(horizon)] = arrays[f"test_TunedRidge_h{int(horizon)}"]
        selected_rows.append(
            {
                "horizon": int(horizon),
                "candidate": str(row["candidate"]),
                "stage": str(row["stage"]),
                "history": int(row["history"]),
                "hidden_dim": int(row["hidden_dim"]),
                "val_rmse": float(best[0]),
                "test_rmse": float(
                    np.sqrt(
                        np.mean(
                            (targets["test"][int(horizon)] - selector_forecasts["test"][int(horizon)]) ** 2
                        )
                    )
                ),
            }
        )
    selection = pd.DataFrame(selected_rows)
    selection_path = result_dir / "horizon_selector_selection.csv"
    selection.to_csv(selection_path, index=False)
    forecast_path = result_dir / "horizon_selector_forecast_arrays.npz"
    np.savez_compressed(forecast_path, **forecast_payload)
    names = [f"component_{idx + 1:02d}" for idx in range(next(iter(targets["test"].values())).shape[1])]
    metric_frames = [
        evaluate_multistep("TransformerHorizonSelector", selector_forecasts["test"], targets["test"], names),
        evaluate_multistep("MLP", baseline_forecasts["MLP"], targets["test"], names),
        evaluate_multistep("TunedRidge", baseline_forecasts["TunedRidge"], targets["test"], names),
    ]
    metrics = pd.concat(metric_frames, ignore_index=True)
    significance = compute_prediction_significance(
        {"TransformerHorizonSelector": selector_forecasts["test"], **baseline_forecasts},
        targets["test"],
        reps=int(bootstrap_reps),
        block_size=int(block_size),
        seed=int(seed),
        primary_model="TransformerHorizonSelector",
    )
    selected = {
        "system": "TransformerHorizonSelector",
        "val_avg_rmse": float(selection["val_rmse"].mean()),
        "test_avg_rmse": float(selection["test_rmse"].mean()),
        "selection": str(selection_path),
    }
    return {
        "selected": selected,
        "selection": selection_path,
        "forecast_arrays": forecast_path,
        "metrics": metrics,
        "significance": significance,
    }


def _jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    data: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            data[key] = str(value)
        else:
            data[key] = value
    return data


def run_sweep(args: argparse.Namespace) -> dict[str, Path]:
    result_dir = (Path(args.output_dir) / RESULT_SUBDIR).resolve()
    fig_dir = (Path(args.output_dir) / FIG_SUBDIR).resolve()
    docs_dir = (Path(args.output_dir) / "docs").resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "run_log.jsonl"
    existing = _read_existing_rows(log_path)
    rows: list[dict[str, object]] = list(existing.values())

    start = time.monotonic()
    completed_count = 0

    def should_stop() -> bool:
        if args.max_candidates is not None and completed_count >= int(args.max_candidates):
            return True
        if args.wall_clock_hours is not None and (time.monotonic() - start) / 3600.0 > float(args.wall_clock_hours):
            return True
        return False

    def run_specs(specs: Iterable[dict[str, object]]) -> None:
        nonlocal completed_count
        for spec in specs:
            if should_stop():
                break
            candidate = candidate_name(spec)
            if candidate in existing:
                continue
            _append_jsonl(log_path, {"candidate": candidate, "status": "started", "started_at": time.time(), **spec})
            t0 = time.monotonic()
            try:
                artifacts = run(_config_for_candidate(args, candidate, spec))
                row = _candidate_row(candidate, spec, artifacts, time.monotonic() - t0)
                rows.append(row)
                existing[candidate] = row
                completed_count += 1
                _append_jsonl(log_path, row)
            except Exception as exc:
                _append_jsonl(log_path, {"candidate": candidate, "status": "failed", "error": repr(exc), **spec})
                if not args.keep_going:
                    raise

    base = _base_spec(args)
    histories = parse_history_grid(args.history_grid)
    run_specs(dict(base, stage="A_history", history=int(history)) for history in histories)

    if rows and not should_stop() and not args.final_seeds_only:
        ranked_rows = rank_leaderboard(pd.DataFrame(rows), rank_metric=str(args.rank_metric))
        stage_a = ranked_rows[ranked_rows["stage"] == "A_history"].head(3)
        capacity_specs: list[dict[str, object]] = []
        for history in stage_a["history"].astype(int).tolist():
            for hidden_dim, default_head in [(64, 4), (128, 4), (192, 4), (256, 8)]:
                for layers in [1, 2]:
                    for ff_mult in [2, 4]:
                        head = default_head if hidden_dim % default_head == 0 else _valid_heads(hidden_dim, [2, 4, 8])[0]
                        capacity_specs.append(
                            dict(
                                base,
                                stage="B_capacity",
                                history=int(history),
                                hidden_dim=int(hidden_dim),
                                transformer_nhead=int(head),
                                num_layers=int(layers),
                                transformer_dim_feedforward=int(hidden_dim * ff_mult),
                            )
                        )
        run_specs(capacity_specs)

    if rows and not should_stop() and not args.final_seeds_only:
        ranked_rows = rank_leaderboard(pd.DataFrame(rows), rank_metric=str(args.rank_metric))
        top8 = ranked_rows[ranked_rows["stage"].isin(["A_history", "B_capacity"])].head(8)
        refine_specs: list[dict[str, object]] = []
        for _, row in top8.iterrows():
            parent = _spec_from_row(row, stage="C_pool_pos")
            for head in _valid_heads(int(parent["hidden_dim"]), [2, 4, 8]):
                for pooling in ["last", "mean"]:
                    for pos in ["learned", "sinusoidal"]:
                        refine_specs.append(
                            dict(
                                parent,
                                transformer_nhead=int(head),
                                transformer_pooling=pooling,
                                transformer_positional_encoding=pos,
                            )
                        )
        run_specs(refine_specs)

    if rows and not should_stop() and not args.final_seeds_only:
        ranked_rows = rank_leaderboard(pd.DataFrame(rows), rank_metric=str(args.rank_metric))
        top8 = ranked_rows[ranked_rows["stage"].isin(["A_history", "B_capacity", "C_pool_pos"])].head(8)
        reg_specs: list[dict[str, object]] = []
        for _, row in top8.iterrows():
            parent = _spec_from_row(row)
            for lr in [1e-4, 3e-4, 8e-4, 1.5e-3]:
                reg_specs.append(dict(parent, stage="D_lr", learning_rate=float(lr)))
            for dropout in [0.0, 0.05, 0.1, 0.2]:
                reg_specs.append(dict(parent, stage="D_dropout", dropout=float(dropout)))
            for wd in [0.0, 1e-6, 1e-5, 1e-4, 1e-3]:
                reg_specs.append(dict(parent, stage="D_weight_decay", weight_decay=float(wd)))
        run_specs(reg_specs)

    if rows and not should_stop() and not args.final_seeds_only:
        ranked_rows = rank_leaderboard(pd.DataFrame(rows), rank_metric=str(args.rank_metric))
        top6 = ranked_rows[
            ranked_rows["stage"].isin(["A_history", "B_capacity", "C_pool_pos", "D_lr", "D_dropout", "D_weight_decay"])
        ].head(6)
        objective_specs: list[dict[str, object]] = []
        for _, row in top6.iterrows():
            parent = _spec_from_row(row)
            for objective in ["direct_multihorizon", "rollout_multistep"]:
                for skip_mode in ["frozen_linear_skip", "trainable_linear_skip", "no_skip"]:
                    objective_specs.append(dict(parent, stage="E_objective_skip", rnn_objective=objective, skip_mode=skip_mode))
        run_specs(objective_specs)

    if rows and not should_stop():
        ranked_rows = rank_leaderboard(pd.DataFrame(rows), rank_metric=str(args.rank_metric))
        top3 = ranked_rows[
            ranked_rows["stage"].isin(
                ["A_history", "B_capacity", "C_pool_pos", "D_lr", "D_dropout", "D_weight_decay", "E_objective_skip"]
            )
        ].head(3)
        seed_specs: list[dict[str, object]] = []
        for _, row in top3.iterrows():
            parent = _spec_from_row(row, stage="F_seed")
            for seed in parse_history_grid(args.final_seeds):
                seed_specs.append(dict(parent, seed=int(seed)))
        run_specs(seed_specs)

    if not rows:
        raise RuntimeError("No Transformer candidates completed.")
    leaderboard = rank_leaderboard(pd.DataFrame(rows), rank_metric=str(args.rank_metric))
    leaderboard_path = result_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    plot_path = _save_leaderboard_plot(leaderboard, fig_dir / "leaderboard_val_rmse.png")

    best = leaderboard.iloc[0]
    best_result_dir = Path(str(best["result_dir"]))
    final_metrics = add_best_baseline_rows(pd.read_csv(best_result_dir / "multistep_metrics.csv"))
    ensemble_artifacts = evaluate_ensembles(
        leaderboard,
        result_dir=result_dir,
        horizons=parse_int_tuple(args.horizons),
        bootstrap_reps=int(args.bootstrap_reps),
        block_size=int(args.bootstrap_block_size),
        seed=int(args.seed),
    )
    horizon_selector_artifacts = evaluate_horizon_selector(
        leaderboard,
        result_dir=result_dir,
        horizons=parse_int_tuple(args.horizons),
        bootstrap_reps=int(args.bootstrap_reps),
        block_size=int(args.bootstrap_block_size),
        seed=int(args.seed),
    )
    selected_system: dict[str, object] = {
        "kind": "single_candidate",
        "candidate": str(best["candidate"]),
        "val_avg_rmse": float(best["val_avg_rmse"]),
        "test_avg_rmse": float(best["test_avg_rmse"]),
    }
    if ensemble_artifacts.get("selected") is not None:
        selected_ensemble = dict(ensemble_artifacts["selected"])
        if float(selected_ensemble["val_avg_rmse"]) < float(best["val_avg_rmse"]):
            selected_system = {
                "kind": "ensemble",
                "system": str(selected_ensemble["system"]),
                "k": int(selected_ensemble["k"]),
                "member_candidates": str(selected_ensemble["member_candidates"]),
                "val_avg_rmse": float(selected_ensemble["val_avg_rmse"]),
                "test_avg_rmse": float(selected_ensemble["test_avg_rmse"]),
            }
            final_metrics = add_best_baseline_rows(ensemble_artifacts["metrics"])
    if horizon_selector_artifacts.get("selected") is not None:
        selected_horizon = dict(horizon_selector_artifacts["selected"])
        if float(selected_horizon["val_avg_rmse"]) < float(selected_system["val_avg_rmse"]):
            selected_system = {
                "kind": "horizon_selector",
                "system": str(selected_horizon["system"]),
                "selection": str(selected_horizon["selection"]),
                "val_avg_rmse": float(selected_horizon["val_avg_rmse"]),
                "test_avg_rmse": float(selected_horizon["test_avg_rmse"]),
            }
            final_metrics = add_best_baseline_rows(horizon_selector_artifacts["metrics"])
    final_metrics_path = result_dir / "final_test_metrics.csv"
    final_metrics.to_csv(final_metrics_path, index=False)
    final_plot = save_metric_plot(final_metrics, fig_dir / "final_multistep_rmse.png")
    final_sig_path = result_dir / "final_prediction_significance.json"
    if selected_system["kind"] == "ensemble":
        final_sig_path.write_text(
            json.dumps(_json_sanitize(ensemble_artifacts["significance"]), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
    elif selected_system["kind"] == "horizon_selector":
        final_sig_path.write_text(
            json.dumps(
                _json_sanitize(horizon_selector_artifacts["significance"]),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    else:
        final_sig_path.write_text(
            (best_result_dir / "prediction_significance.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    manifest = {
        "selection_rule": {
            "rank_metric": str(args.rank_metric),
            "selected_candidate": str(best["candidate"]),
            "uses_test_for_selection": False,
        },
        "selected_system": selected_system,
        "n_completed_candidates": int(len(leaderboard)),
        "skip_gru_reference": bool(args.skip_gru_reference),
        "candidate_plan_size": int(len(build_candidate_specs(args))),
        "artifacts": {
            "result_dir": str(result_dir),
            "fig_dir": str(fig_dir),
            "leaderboard": str(leaderboard_path),
            "ensemble_leaderboard": str(ensemble_artifacts.get("leaderboard") or ""),
            "ensemble_forecast_arrays": str(ensemble_artifacts.get("forecast_arrays") or ""),
            "horizon_selector_selection": str(horizon_selector_artifacts.get("selection") or ""),
            "horizon_selector_forecast_arrays": str(horizon_selector_artifacts.get("forecast_arrays") or ""),
            "final_test_metrics": str(final_metrics_path),
            "final_prediction_significance": str(final_sig_path),
            "leaderboard_plot": str(plot_path),
            "final_metric_plot": str(final_plot),
        },
        "config": _jsonable_args(args),
        "dependency_versions": dependency_versions(),
    }
    manifest_path = result_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_json_sanitize(manifest), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    report_path = (Path(args.output_dir) / REPORT_PATH).resolve()
    _write_report(report_path, leaderboard, manifest)
    return {
        "result_dir": result_dir,
        "fig_dir": fig_dir,
        "manifest": manifest_path,
        "leaderboard": leaderboard_path,
        "final_test_metrics": final_metrics_path,
        "final_prediction_significance": final_sig_path,
        "report": report_path,
        "leaderboard_plot": plot_path,
        "final_metric_plot": final_plot,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-scores", type=Path, default=pairwise.DEFAULT_COMPONENT_SCORES)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--history-grid", default="1,2,4,8,12,16,24,32")
    parser.add_argument("--horizons", default="1,2,4,8")
    parser.add_argument("--rank-metric", default="val_avg_rmse")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--lag", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--transformer-nhead", type=int, default=4)
    parser.add_argument("--transformer-dim-feedforward", type=int, default=256)
    parser.add_argument("--transformer-pooling", choices=["last", "mean"], default="last")
    parser.add_argument("--transformer-positional-encoding", choices=["learned", "sinusoidal"], default="learned")
    parser.add_argument(
        "--rnn-objective",
        choices=["direct_multihorizon", "one_step_recursive", "rollout_multistep"],
        default="direct_multihorizon",
    )
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ridge-alphas", default="10,100,1000,3000")
    parser.add_argument("--disable-linear-skip", dest="use_linear_skip", action="store_false")
    parser.set_defaults(use_linear_skip=True)
    parser.add_argument("--train-linear-skip", dest="freeze_linear_skip", action="store_false")
    parser.set_defaults(freeze_linear_skip=True)
    parser.add_argument("--disable-residual-shrinkage", dest="residual_shrinkage", action="store_false")
    parser.set_defaults(residual_shrinkage=True)
    parser.add_argument("--residual-gamma-min", type=float, default=-0.5)
    parser.add_argument("--residual-gamma-max", type=float, default=0.5)
    parser.add_argument("--residual-gamma-steps", type=int, default=101)
    parser.add_argument("--rnn-linear-blend-grid-steps", type=int, default=101)
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--scheduler-patience", type=int, default=12)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-num-layers", type=int, default=1)
    parser.add_argument("--mlp-dropout", type=float, default=0.5)
    parser.add_argument("--mlp-epochs", type=int, default=120)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-block-size", type=int, default=26)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final-seeds", default="42,43,44,45,46")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--wall-clock-hours", type=float, default=12.0)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--skip-gru-reference", action="store_true")
    parser.add_argument("--final-seeds-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    artifacts = run_sweep(parse_args(argv))
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
