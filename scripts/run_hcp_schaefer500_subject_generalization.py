#!/usr/bin/env python3
"""Nested subject-generalization benchmark for Schaefer-500 dynamics.

The runner treats complete subjects as the experimental unit.  A held-out subject
may calibrate only on its first ``calibration_end`` samples, then forecasts its
future samples with a frozen shared model and an adapted low-dimensional module.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hcp_schaefer500_subject_models import (
    DeltaRidge,
    HierarchicalLowRankDeltaVAR,
    NeuralDeltaModel,
    SubjectNormalizer,
    make_history_samples,
    make_subject_folds,
    recursive_rollout,
)


DEFAULT_DATA_ROOT = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30"
DEFAULT_OUTPUT = ROOT / "results" / "hcp_schaefer500_subject_generalization"
DEFAULT_LOG_DIR = ROOT / "docs" / "log" / "hcp_schaefer500_subject_generalization"
CALIBRATION_END = 900
HORIZONS = (1, 5, 10, 20)
NEURAL_SEEDS = (101, 202, 303)


@dataclass(frozen=True)
class Candidate:
    model: str
    kind: str
    order: int = 1
    alpha: float = 100.0
    rank: int = 8
    adapter_ridge: float = 1.0
    width: int = 64
    adapter_dim: int = 8
    learning_rate: float = 3.0e-3
    epochs: int = 180
    personalized: bool = False
    seed: int = 202

    @property
    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def load_subject_series(data_root: Path) -> dict[str, np.ndarray]:
    result = {}
    for path in sorted(Path(data_root).glob("sub-*/*.mat")):
        values = np.asarray(loadmat(path)["Schaefer500"], dtype=float)
        if values.shape != (1200, 500) or not np.isfinite(values).all():
            raise ValueError(f"Expected finite [1200, 500] Schaefer500 data in {path}, got {values.shape}.")
        result[path.parent.name] = values
    if len(result) != 30:
        raise ValueError(f"Expected exactly 30 subject MAT files, found {len(result)} under {data_root}.")
    return result


def normalize_subjects(raw: Mapping[str, np.ndarray], *, calibration_end: int) -> dict[str, np.ndarray]:
    return {
        subject: SubjectNormalizer.fit(series, calibration_end=calibration_end).transform(series)
        for subject, series in raw.items()
    }


def _relative_fc_error(truth: np.ndarray, prediction: np.ndarray) -> float:
    true_fc = np.corrcoef(np.asarray(truth, dtype=float), rowvar=False)
    predicted_fc = np.corrcoef(np.asarray(prediction, dtype=float), rowvar=False)
    numerator = np.linalg.norm(np.nan_to_num(predicted_fc - true_fc), ord="fro")
    denominator = max(np.linalg.norm(np.nan_to_num(true_fc), ord="fro"), 1.0e-12)
    return float(numerator / denominator)


def _log_psd_mae(truth: np.ndarray, prediction: np.ndarray) -> float:
    segment = min(128, len(truth))
    _, true_psd = welch(np.asarray(truth, dtype=float), axis=0, nperseg=segment)
    _, predicted_psd = welch(np.asarray(prediction, dtype=float), axis=0, nperseg=segment)
    return float(np.mean(np.abs(np.log(true_psd + 1.0e-8) - np.log(predicted_psd + 1.0e-8))))


def evaluate_rollout_metrics(
    series: np.ndarray,
    *,
    order: int,
    test_start: int,
    horizons: Sequence[int],
    predict_delta: Callable[[np.ndarray], np.ndarray],
) -> dict[str, object]:
    """Score a delta predictor on teacher-reset recursive rollouts.

    Each origin is an observed test history; recursive predictions are therefore
    local dynamical forecasts rather than a single noise-free 300-step simulation.
    """
    values = np.asarray(series, dtype=float)
    max_horizon = max(int(horizon) for horizon in horizons)
    origins = range(int(test_start), len(values) - max_horizon + 1, max_horizon)
    squared_error = {int(horizon): [] for horizon in horizons}
    persistence_error = {int(horizon): [] for horizon in horizons}
    one_step_prediction, one_step_truth = [], []
    horizon_twenty_prediction, horizon_twenty_truth = [], []
    all_finite = True
    for origin in origins:
        history = np.stack([values[origin - 1 - lag] for lag in range(order)])
        prediction = recursive_rollout(history, horizon=max_horizon, predict_delta=predict_delta)
        all_finite &= bool(np.isfinite(prediction).all())
        for horizon in horizons:
            truth = values[origin + int(horizon) - 1]
            squared_error[int(horizon)].append(float(np.mean((prediction[int(horizon) - 1] - truth) ** 2)))
            persistence_error[int(horizon)].append(float(np.mean((history[0] - truth) ** 2)))
        one_step_prediction.append(prediction[0])
        one_step_truth.append(values[origin])
        horizon_twenty_prediction.append(prediction[max_horizon - 1])
        horizon_twenty_truth.append(values[origin + max_horizon - 1])
    ratios = {
        str(horizon): float(
            np.sqrt(np.mean(squared_error[horizon])) / max(np.sqrt(np.mean(persistence_error[horizon])), 1.0e-12)
        )
        for horizon in squared_error
    }
    one_step_prediction_array = np.asarray(one_step_prediction)
    one_step_truth_array = np.asarray(one_step_truth)
    long_prediction = np.asarray(horizon_twenty_prediction)
    long_truth = np.asarray(horizon_twenty_truth)
    return {
        "n_rollout_origins": int(len(range(int(test_start), len(values) - max_horizon + 1, max_horizon))),
        "skill_ratio_by_horizon": ratios,
        "mean_skill_ratio": float(np.mean(list(ratios.values()))),
        "fc_error": _relative_fc_error(one_step_truth_array, one_step_prediction_array),
        "psd_error": _log_psd_mae(one_step_truth_array, one_step_prediction_array),
        "rollout_finite": float(all_finite),
        "rollout_variance_ratio": float(np.mean(np.std(long_prediction, axis=0) / np.maximum(np.std(long_truth, axis=0), 1.0e-8))),
    }


def choose_champion(rows: Sequence[Mapping[str, object]], *, practical_margin: float = 0.01) -> str:
    """Choose by primary score, then FC and PSD only inside a practical tie."""
    by_model: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    primary = {name: float(np.median([float(row["mean_skill_ratio"]) for row in values])) for name, values in by_model.items()}
    best = min(primary.values())
    tied = [name for name, value in primary.items() if value <= best + practical_margin]
    return min(
        tied,
        key=lambda name: (
            float(np.mean([float(row["fc_error"]) for row in by_model[name]])),
            float(np.mean([float(row["psd_error"]) for row in by_model[name]])),
            name,
        ),
    )


def paired_subject_bootstrap(
    rows: Sequence[Mapping[str, object]], *, reference: str, replicates: int = 10_000, seed: int = 90210
) -> dict[str, dict[str, float]]:
    """Paired subject-level bootstrap of model minus reference primary score."""
    by_model_subject: dict[str, dict[str, float]] = {}
    for row in rows:
        by_model_subject.setdefault(str(row["model"]), {})[str(row["subject"])] = float(row["mean_skill_ratio"])
    subject_order = tuple(sorted(by_model_subject[reference]))
    rng = np.random.default_rng(seed)
    output = {}
    reference_values = np.asarray([by_model_subject[reference][subject] for subject in subject_order])
    for model, values in by_model_subject.items():
        delta = np.asarray([values[subject] for subject in subject_order]) - reference_values
        draws = np.mean(delta[rng.integers(0, len(delta), size=(replicates, len(delta)))], axis=1)
        output[model] = {
            "mean_delta_vs_reference": float(np.mean(delta)),
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "subjects_worse_than_reference": int(np.sum(delta > 0.0)),
        }
    return output


def candidate_grid(*, smoke: bool) -> dict[str, tuple[Candidate, ...]]:
    neural_epochs = 20 if smoke else 100
    grids: dict[str, tuple[Candidate, ...]] = {
        "individual_ridge": tuple(Candidate("individual_ridge", "individual_ridge", order=1, alpha=alpha) for alpha in (100.0, 1000.0)),
        "pooled_ridge": tuple(Candidate("pooled_ridge", "pooled_ridge", order=1, alpha=alpha) for alpha in (100.0, 1000.0)),
        "hierarchical_var": tuple(Candidate("hierarchical_var", "hierarchical", order=1, alpha=100.0, rank=rank, adapter_ridge=1.0) for rank in (8, 16)),
    }
    for model, kind, personalized in (
        ("residual_mlp_shared", "mlp", False),
        ("residual_mlp_film", "mlp", True),
        ("factorized_tcn_shared", "tcn", False),
        ("factorized_tcn_film", "tcn", True),
    ):
        grids[model] = tuple(
            Candidate(model, kind, order=order, width=64, adapter_dim=8, learning_rate=lr, epochs=neural_epochs, personalized=personalized, seed=seed)
            for order in (5, 10)
            for lr in (1.0e-3,)
            for seed in (202,)
        )
    if smoke:
        return {name: values[:1] for name, values in grids.items()}
    return grids


def _seed_variants(candidate: Candidate, *, smoke: bool) -> tuple[Candidate, ...]:
    if candidate.kind not in {"mlp", "tcn"} or smoke:
        return (candidate,)
    return tuple(replace(candidate, seed=seed) for seed in NEURAL_SEEDS)


def _mean_seed_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Average repeated neural seeds before comparing them with deterministic models."""
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault((row["outer_fold"], row["subject"], row["model"]), []).append(row)
    averaged = []
    for key, values in grouped.items():
        first = values[0]
        horizons = sorted(first["skill_ratio_by_horizon"], key=int)
        averaged.append(
            {
                "outer_fold": key[0],
                "subject": key[1],
                "model": key[2],
                "candidate": first["candidate"],
                "seeds": [int(value["candidate"]["seed"]) for value in values],
                "skill_ratio_by_horizon": {
                    horizon: float(np.mean([value["skill_ratio_by_horizon"][horizon] for value in values])) for horizon in horizons
                },
                "mean_skill_ratio": float(np.mean([value["mean_skill_ratio"] for value in values])),
                "fc_error": float(np.mean([value["fc_error"] for value in values])),
                "psd_error": float(np.mean([value["psd_error"] for value in values])),
                "rollout_finite": float(np.min([value["rollout_finite"] for value in values])),
                "rollout_variance_ratio": float(np.mean([value["rollout_variance_ratio"] for value in values])),
                "n_rollout_origins": int(first["n_rollout_origins"]),
            }
        )
    return averaged


def _fit_predictor(candidate: Candidate, sequences: Mapping[str, np.ndarray], *, calibration_end: int):
    if candidate.kind == "pooled_ridge":
        return DeltaRidge(order=candidate.order, alpha=candidate.alpha).fit(list(sequences.values()), calibration_end=calibration_end)
    if candidate.kind == "hierarchical":
        return HierarchicalLowRankDeltaVAR(
            order=candidate.order, alpha=candidate.alpha, rank=candidate.rank, adapter_ridge=candidate.adapter_ridge
        ).fit(sequences, calibration_end=calibration_end)
    if candidate.kind in {"mlp", "tcn"}:
        return NeuralDeltaModel(
            kind=candidate.kind,
            order=candidate.order,
            width=candidate.width,
            adapter_dim=candidate.adapter_dim,
            learning_rate=candidate.learning_rate,
            epochs=candidate.epochs,
            seed=candidate.seed,
            personalized=candidate.personalized,
            # The benchmark uses many small calibration batches; CPU avoids the
            # host-device transfer overhead that dominates this workload on MPS.
            device="cpu",
        ).fit(sequences, calibration_end=calibration_end)
    if candidate.kind == "individual_ridge":
        return None
    raise ValueError(f"Unknown candidate kind {candidate.kind}.")


def _subject_predict_delta(candidate: Candidate, predictor, sequence: np.ndarray, *, calibration_end: int):
    if candidate.kind == "individual_ridge":
        local = DeltaRidge(order=candidate.order, alpha=candidate.alpha).fit([sequence], calibration_end=calibration_end)
        return local.predict_delta, local
    if candidate.kind == "hierarchical":
        adapter = predictor.fit_adapter(sequence, calibration_end=calibration_end)
        return lambda history: predictor.predict_delta(history, adapter), adapter
    if candidate.kind in {"mlp", "tcn"}:
        adapter = predictor.fit_adapter(sequence, calibration_end=calibration_end)
        return lambda history: predictor.predict_delta(history, adapter), adapter
    return predictor.predict_delta, None


def evaluate_candidate(
    candidate: Candidate,
    *,
    normalized: Mapping[str, np.ndarray],
    train_subjects: Sequence[str],
    test_subjects: Sequence[str],
    calibration_end: int,
) -> list[dict[str, object]]:
    predictor = _fit_predictor(candidate, {subject: normalized[subject] for subject in train_subjects}, calibration_end=calibration_end)
    rows = []
    for subject in test_subjects:
        predict_delta, _ = _subject_predict_delta(candidate, predictor, normalized[subject], calibration_end=calibration_end)
        metrics = evaluate_rollout_metrics(
            normalized[subject], order=candidate.order, test_start=calibration_end, horizons=HORIZONS, predict_delta=predict_delta
        )
        rows.append({"subject": subject, "model": candidate.model, "candidate": asdict(candidate), **metrics})
    return rows


def _select_candidate(
    candidates: Sequence[Candidate],
    *,
    normalized: Mapping[str, np.ndarray],
    train_subjects: Sequence[str],
    calibration_end: int,
    inner_seed: int,
    smoke: bool,
) -> Candidate:
    inner_folds = make_subject_folds(train_subjects, n_splits=4, seed=inner_seed)
    scores: list[tuple[float, Candidate]] = []
    for candidate in candidates:
        rows = []
        for fold in inner_folds:
            for variant in _seed_variants(candidate, smoke=smoke):
                rows.extend(
                    evaluate_candidate(
                        variant,
                        normalized=normalized,
                        train_subjects=fold.train_subjects,
                        test_subjects=fold.test_subjects,
                        calibration_end=calibration_end,
                    )
                )
        scores.append((float(np.mean([float(row["mean_skill_ratio"]) for row in rows])), candidate))
    return min(scores, key=lambda item: (item[0], item[1].order, item[1].alpha, item[1].seed))[1]


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON serialize {type(value)!r}.")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_status(log_dir: Path, *, status: str, running: str, recent: str, next_step: str) -> None:
    _write(
        log_dir / "live_status.md",
        "# 实时状态\n\n## 当前状态\n- " + status + "\n\n## 正在运行\n- " + running + "\n\n## 最近结果\n- " + recent + "\n\n## 下一步\n- " + next_step + "\n\n## 监控文件\n- run history: `run_history.jsonl`\n",
    )


def _plot(rows: Sequence[Mapping[str, object]], destination: Path) -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8, "axes.spines.right": False, "axes.spines.top": False})
    by_model: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    names = sorted(by_model, key=lambda name: np.median([float(row["mean_skill_ratio"]) for row in by_model[name]]))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.2), constrained_layout=True)
    rng = np.random.default_rng(18)
    for position, name in enumerate(names):
        values = np.asarray([float(row["mean_skill_ratio"]) for row in by_model[name]])
        axes[0].scatter(position + rng.normal(0, 0.045, len(values)), values, s=12, alpha=0.75)
        axes[0].plot([position - 0.2, position + 0.2], [np.median(values)] * 2, color="black", linewidth=1.2)
    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)
    axes[0].set_xticks(range(len(names)), names, rotation=50, ha="right")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean rollout skill ratio (log; lower is better)")
    for name in names:
        horizon_values = {str(horizon): [] for horizon in HORIZONS}
        for row in by_model[name]:
            for horizon, value in row["skill_ratio_by_horizon"].items():
                horizon_values[str(horizon)].append(float(value))
        axes[1].plot(HORIZONS, [np.median(horizon_values[str(horizon)]) for horizon in HORIZONS], marker="o", label=name)
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)
    axes[1].set_yscale("log")
    axes[1].set(xlabel="Forecast horizon", ylabel="Median skill ratio (log)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=6)
    fc = [np.median([float(row["fc_error"]) for row in by_model[name]]) for name in names]
    psd = [np.median([float(row["psd_error"]) for row in by_model[name]]) for name in names]
    axes[2].scatter(fc, psd, s=36)
    for name, x_value, y_value in zip(names, fc, psd):
        axes[2].annotate(name, (x_value, y_value), xytext=(3, 2), textcoords="offset points", fontsize=6)
    axes[2].set(xlabel="FC relative error", ylabel="Log-PSD MAE")
    for suffix, kwargs in ((".png", {"dpi": 300}), (".svg", {})):
        fig.savefig(destination.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def _refit_full_cohort(
    candidate: Candidate,
    *,
    normalized: Mapping[str, np.ndarray],
    calibration_end: int,
    output_dir: Path,
) -> dict[str, object]:
    """Fit the selected architecture on every subject after CV selection."""
    predictor = _fit_predictor(candidate, normalized, calibration_end=calibration_end)
    adapters: dict[str, np.ndarray] = {}
    individual_models: dict[str, object] = {}
    for subject, sequence in normalized.items():
        _, adapted = _subject_predict_delta(candidate, predictor, sequence, calibration_end=calibration_end)
        if candidate.kind == "individual_ridge":
            individual_models[subject] = adapted
        elif adapted is not None:
            adapters[subject] = np.asarray(adapted, dtype=float)
    model_path = output_dir / "final_model.pkl"
    if candidate.kind in {"mlp", "tcn"}:
        import torch

        torch.save({"candidate": asdict(candidate), "state_dict": predictor.net.state_dict()}, output_dir / "final_model.pt")
        model_path = output_dir / "final_model.pt"
    else:
        with model_path.open("wb") as handle:
            pickle.dump(individual_models if candidate.kind == "individual_ridge" else predictor, handle)
    np.savez_compressed(output_dir / "final_subject_adapters.npz", **adapters)
    return {"candidate": asdict(candidate), "model_path": str(model_path), "adapter_path": str(output_dir / "final_subject_adapters.npz"), "n_subject_adapters": len(adapters), "n_individual_models": len(individual_models), "note": "Refit on all 30 subjects after nested-CV selection; this artifact has no independent generalization score."}


def run_benchmark(
    *,
    data_root: Path,
    output_dir: Path,
    log_dir: Path,
    smoke: bool,
    calibration_end: int = CALIBRATION_END,
) -> dict[str, object]:
    raw = load_subject_series(data_root)
    normalized = normalize_subjects(raw, calibration_end=calibration_end)
    outer_folds = make_subject_folds(tuple(raw), n_splits=5, seed=17)
    grids = candidate_grid(smoke=smoke)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "experiment_contract.json", json.dumps({"data": "HCP REST1_LR Schaefer500", "n_subjects": 30, "calibration_end": calibration_end, "horizons": HORIZONS, "outer_folds": [asdict(fold) for fold in outer_folds], "smoke": smoke}, indent=2, default=_json_default) + "\n")
    _write(log_dir / "repo_audit.md", "# 仓库审计\n\n- 输入：30 名 HCP `REST1_LR` Schaefer-500 序列，每人 1200 × 500。\n- 主入口：`scripts/run_hcp_schaefer500_subject_generalization.py`。\n- 评价：嵌套五折被试级泛化；前 900 点只用于校准，后 300 点只用于评分。\n")
    _write(log_dir / "search_plan.md", "# 搜索计划\n\n- 以被试为单位执行嵌套五折共享—个体化动力学比较。\n- 训练及个体适配只读取每名被试前 900 点；后 300 点只评分。\n")
    _write(log_dir / "search_space.json", json.dumps({name: [asdict(candidate) for candidate in candidates] for name, candidates in grids.items()}, indent=2, default=_json_default) + "\n")
    _write_status(log_dir, status="running", running="初始化嵌套被试级 benchmark。", recent="尚无结果。", next_step="完成第一个外层折。")
    all_rows: list[dict[str, object]] = []
    selected: dict[str, list[Candidate]] = {name: [] for name in grids}
    active_folds = outer_folds[:1] if smoke else outer_folds
    for fold_index, outer in enumerate(active_folds, start=1):
        for model, candidates in grids.items():
            choice = _select_candidate(candidates, normalized=normalized, train_subjects=outer.train_subjects, calibration_end=calibration_end, inner_seed=100 + fold_index, smoke=smoke)
            selected[model].append(choice)
            seed_rows = []
            for variant in _seed_variants(choice, smoke=smoke):
                rows = evaluate_candidate(variant, normalized=normalized, train_subjects=outer.train_subjects, test_subjects=outer.test_subjects, calibration_end=calibration_end)
                for row in rows:
                    row["outer_fold"] = fold_index
                seed_rows.extend(rows)
            rows = _mean_seed_rows(seed_rows)
            all_rows.extend(rows)
            with (log_dir / "run_history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"outer_fold": fold_index, "model": model, "candidate": asdict(choice), "n_test_subjects": len(rows), "mean_skill_ratio": float(np.mean([row["mean_skill_ratio"] for row in rows]))}, default=_json_default) + "\n")
        _write_status(log_dir, status="running", running=f"完成外层折 {fold_index}/{len(active_folds)}。", recent=f"已记录 {len(all_rows)} 个被试级 out-of-fold 结果。", next_step="继续下一外层折。")
    champion = choose_champion(all_rows)
    summary_by_model = {}
    for model in grids:
        rows = [row for row in all_rows if row["model"] == model]
        summary_by_model[model] = {
            "n_subject_rows": len(rows),
            "median_mean_skill_ratio": float(np.median([row["mean_skill_ratio"] for row in rows])),
            "mean_fc_error": float(np.mean([row["fc_error"] for row in rows])),
            "mean_psd_error": float(np.mean([row["psd_error"] for row in rows])),
            "finite_rollouts": int(np.sum([row["rollout_finite"] == 1.0 for row in rows])),
        }
    final_refit = None
    if not smoke:
        winner_choices = selected[champion]
        chosen_key = Counter(choice.key for choice in winner_choices).most_common(1)[0][0]
        final_choice = next(choice for choice in winner_choices if choice.key == chosen_key)
        final_refit = _refit_full_cohort(final_choice, normalized=normalized, calibration_end=calibration_end, output_dir=output_dir)
    paired_bootstrap = paired_subject_bootstrap(all_rows, reference=champion)
    summary = {"smoke": smoke, "n_outer_folds": len(active_folds), "n_subjects_evaluated": len({row["subject"] for row in all_rows}), "winner": champion, "models": summary_by_model, "paired_subject_bootstrap": paired_bootstrap, "selected_candidates": {model: [asdict(value) for value in values] for model, values in selected.items()}, "final_refit": final_refit, "rows": all_rows}
    _write(output_dir / "summary.json", json.dumps(summary, indent=2, default=_json_default) + "\n")
    _write(output_dir / "paired_subject_bootstrap.json", json.dumps(paired_bootstrap, indent=2, default=_json_default) + "\n")
    _plot(all_rows, output_dir / "subject_generalization_benchmark")
    leaderboard = ["# Leaderboard", "", "| Model | Median rollout skill ratio | FC error | Log-PSD MAE |", "|---|---:|---:|---:|"]
    for model, metrics in sorted(summary_by_model.items(), key=lambda item: item[1]["median_mean_skill_ratio"]):
        leaderboard.append(f"| {model} | {metrics['median_mean_skill_ratio']:.6f} | {metrics['mean_fc_error']:.6f} | {metrics['mean_psd_error']:.6f} |")
    _write(log_dir / "leaderboard.md", "\n".join(leaderboard) + "\n")
    note = "smoke 仅验证流程，不用于科学结论。" if smoke else "完整结果按被试级 out-of-fold 指标汇总。"
    _write(log_dir / "notes.md", f"# 观察\n\n- 当前嵌套 benchmark 的胜者：`{champion}`。\n- {note}\n")
    _write(log_dir / "next_steps.md", "# 下一步\n\n- 仅在完整 benchmark 完成后，将胜出模型用于后续机制分析。\n- 不将本预测基准直接解释为 raw $\\Phi^{EID}$ 证据。\n")
    _write(log_dir / "tuning_report.md", "# Schaefer-500 跨被试共享—个体化动力学报告\n\n## 目标\n\n比较共享、个体化与非线性预测器在留出被试未来 300 点上的多步预测。\n\n## 当前结果\n\n" + "\n".join(leaderboard) + f"\n\n- Winner: `{champion}`。\n- {note}\n")
    _write_status(log_dir, status="completed", running="无。", recent=f"完成 {len(active_folds)} 个外层折；当前胜者为 {champion}。", next_step="阅读 leaderboard 与 summary，再决定是否进行后续机制分析。")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--smoke", action="store_true", help="Run one outer fold with one configuration per model.")
    parser.add_argument("--calibration-end", type=int, default=CALIBRATION_END)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_benchmark(data_root=args.data_root, output_dir=args.output_dir, log_dir=args.log_dir, smoke=bool(args.smoke), calibration_end=int(args.calibration_end))
    print(json.dumps({"winner": summary["winner"], "models": summary["models"]}, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
