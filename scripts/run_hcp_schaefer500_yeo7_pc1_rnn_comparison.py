#!/usr/bin/env python3
"""Controlled RNN comparison for train-only Yeo7-PC1 dynamics."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import (
    DEFAULT_DATA,
    DEFAULT_LABELS,
    fit_yeo7_pc1,
    load_yeo7_groups,
)

DEFAULT_DATA_ROOT = DEFAULT_DATA.parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "hcp_schaefer500_yeo7_pc1_rnn_comparison"
FOLDS = (600, 700, 800)
DEVELOPMENT_END = 900
TEST_END = 1200
SELECTION_SEEDS = (101, 202, 303)
FINAL_SEEDS = (1001, 1002, 1003)


@dataclass(frozen=True)
class RidgeCandidate:
    history: int
    alpha: float


@dataclass(frozen=True)
class RnnCandidate:
    history: int
    hidden: int
    learning_rate: float
    epochs: int


@dataclass(frozen=True)
class ExperimentConfig:
    subject_limit: int | None
    ridge_candidates: tuple[RidgeCandidate, ...]
    rnn_candidates: tuple[RnnCandidate, ...]
    selection_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]


def build_config(*, smoke: bool) -> ExperimentConfig:
    """Return the pre-registered candidate grid for smoke or full execution."""
    if smoke:
        return ExperimentConfig(
            subject_limit=1,
            ridge_candidates=(RidgeCandidate(history=3, alpha=10.0),),
            rnn_candidates=(RnnCandidate(history=3, hidden=4, learning_rate=3.0e-3, epochs=10),),
            selection_seeds=(101,),
            final_seeds=(1001,),
        )
    return ExperimentConfig(
        subject_limit=None,
        ridge_candidates=tuple(
            RidgeCandidate(history=history, alpha=alpha)
            for history in (1, 2, 3, 5, 8)
            for alpha in (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6)
        ),
        rnn_candidates=tuple(
            RnnCandidate(history=history, hidden=hidden, learning_rate=learning_rate, epochs=300)
            for history in (3, 5, 8)
            for hidden in (8, 16)
            for learning_rate in (1.0e-3, 3.0e-3)
        ),
        selection_seeds=SELECTION_SEEDS,
        final_seeds=FINAL_SEEDS,
    )


def reduce_train_only(
    raw_series: np.ndarray,
    groups: Mapping[str, Sequence[int]],
    *,
    train_end: int,
) -> np.ndarray:
    """Return Yeo-network PC1 states after fitting every PC only on the train prefix."""
    reducer = fit_yeo7_pc1(np.asarray(raw_series, dtype=float)[: int(train_end)], groups)
    return reducer.transform(np.asarray(raw_series, dtype=float))


def make_sequence_samples(
    series: np.ndarray,
    *,
    history: int,
    target_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build chronological history windows with targets strictly before ``target_end``."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 2 or history < 1 or target_end > len(values) or target_end <= history:
        raise ValueError("series, history, and target_end do not define any valid sequence samples.")
    starts = range(0, int(target_end) - int(history))
    return (
        np.stack([values[start : start + history] for start in starts]),
        np.stack([values[start + history] for start in starts]),
    )


def fit_rnn_predict(
    train_x: np.ndarray,
    train_delta: np.ndarray,
    eval_x: np.ndarray,
    *,
    hidden: int,
    learning_rate: float,
    epochs: int,
    seed: int,
) -> np.ndarray:
    """Fit a deterministic vanilla RNN and predict one standardized delta per window."""
    import torch

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    train_windows = np.asarray(train_x, dtype=np.float32)
    train_targets = np.asarray(train_delta, dtype=np.float32)
    eval_windows = np.asarray(eval_x, dtype=np.float32)
    rnn = torch.nn.RNN(
        input_size=train_windows.shape[-1],
        hidden_size=int(hidden),
        batch_first=True,
        nonlinearity="tanh",
    )
    head = torch.nn.Linear(int(hidden), train_targets.shape[-1])
    parameters = list(rnn.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate), weight_decay=1.0e-4)
    x = torch.tensor(train_windows)
    y = torch.tensor(train_targets)
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        encoded, _ = rnn(x)
        loss = torch.mean((head(encoded[:, -1]) - y) ** 2)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        encoded, _ = rnn(torch.tensor(eval_windows))
        return head(encoded[:, -1]).cpu().numpy().astype(float)


def _paired_bootstrap(delta: np.ndarray, *, replicates: int, seed: int) -> dict[str, float]:
    values = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(int(seed))
    draws = values[rng.integers(0, len(values), size=(int(replicates), len(values)))].mean(axis=1)
    return {
        "mean_delta_rmse": float(values.mean()),
        "median_delta_rmse": float(np.median(values)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "subjects_lower_rmse": int(np.sum(values < 0.0)),
        "n_subjects": int(len(values)),
    }


def summarize_subject_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_replicates: int = 10_000,
    seed: int = 90210,
) -> dict[str, dict[str, float | int]]:
    """Summarize paired RNN-minus-Ridge test RMSE across the same subjects."""
    by_model = {
        str(model): {str(row["subject"]): float(row["rmse"]) for row in rows if str(row["model"]) == str(model)}
        for model in {str(row["model"]) for row in rows}
    }
    reference = by_model["ridge"]
    summary = {}
    for index, model in enumerate(("individual_rnn", "shared_rnn")):
        subjects = tuple(sorted(reference))
        if set(by_model[model]) != set(subjects):
            raise ValueError(f"{model} must have exactly the same subject rows as ridge.")
        delta = np.asarray([by_model[model][subject] - reference[subject] for subject in subjects])
        summary[f"{model}_minus_ridge"] = _paired_bootstrap(
            delta,
            replicates=bootstrap_replicates,
            seed=int(seed) + index,
        )
    return summary


def _standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(train, dtype=float).mean(axis=0, keepdims=True)
    scale = np.asarray(train, dtype=float).std(axis=0, ddof=1, keepdims=True)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return (np.asarray(values, dtype=float) - mean) / scale, mean, scale


def _prepared_split(
    raw_series: np.ndarray,
    groups: Mapping[str, Sequence[int]],
    *,
    train_end: int,
    evaluation_start: int,
    evaluation_end: int,
    history: int,
) -> dict[str, np.ndarray]:
    """Prepare one leakage-free, standardized delta-prediction split."""
    reduced = reduce_train_only(raw_series, groups, train_end=train_end)
    windows, targets = make_sequence_samples(reduced, history=history, target_end=evaluation_end)
    target_indices = np.arange(history, evaluation_end)
    train_mask = target_indices < train_end
    eval_mask = (target_indices >= evaluation_start) & (target_indices < evaluation_end)
    if not train_mask.any() or not eval_mask.any():
        raise ValueError("Split has no train or evaluation sequence windows.")
    train_windows = windows[train_mask]
    eval_windows = windows[eval_mask]
    train_next = targets[train_mask]
    eval_next = targets[eval_mask]
    train_x, x_mean, x_scale = _standardize(train_windows, train_windows)
    eval_x = (eval_windows - x_mean) / x_scale
    train_delta = train_next - train_windows[:, -1]
    train_delta_z, delta_mean, delta_scale = _standardize(train_delta, train_delta)
    state_train = train_windows[:, -1]
    _, state_mean, state_scale = _standardize(state_train, eval_next)
    return {
        "train_x": train_x,
        "train_delta_z": train_delta_z,
        "eval_x": eval_x,
        "eval_last": eval_windows[:, -1],
        "eval_next": eval_next,
        "delta_mean": delta_mean,
        "delta_scale": delta_scale,
        "state_mean": state_mean,
        "state_scale": state_scale,
    }


def _evaluate_delta_prediction(split: Mapping[str, np.ndarray], delta_z: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(split["eval_last"]) + np.asarray(delta_z) * np.asarray(split["delta_scale"]) + np.asarray(split["delta_mean"])
    truth_z = (np.asarray(split["eval_next"]) - np.asarray(split["state_mean"])) / np.asarray(split["state_scale"])
    prediction_z = (prediction - np.asarray(split["state_mean"])) / np.asarray(split["state_scale"])
    persistence_z = (np.asarray(split["eval_last"]) - np.asarray(split["state_mean"])) / np.asarray(split["state_scale"])
    time_mse = np.mean((truth_z - prediction_z) ** 2, axis=1)
    persistence_mse = np.mean((truth_z - persistence_z) ** 2, axis=1)
    return {
        "rmse": float(np.sqrt(time_mse.mean())),
        "skill_ratio": float(np.sqrt(time_mse.mean()) / max(np.sqrt(persistence_mse.mean()), 1.0e-12)),
        "time_mse": time_mse,
    }


def _ridge_prediction(split: Mapping[str, np.ndarray], candidate: RidgeCandidate) -> np.ndarray:
    predictor = Ridge(alpha=float(candidate.alpha), fit_intercept=True).fit(
        np.asarray(split["train_x"]).reshape(len(split["train_x"]), -1),
        np.asarray(split["train_delta_z"]),
    )
    return predictor.predict(np.asarray(split["eval_x"]).reshape(len(split["eval_x"]), -1))


def _rnn_prediction(split: Mapping[str, np.ndarray], candidate: RnnCandidate, *, seed: int) -> np.ndarray:
    return fit_rnn_predict(
        np.asarray(split["train_x"]),
        np.asarray(split["train_delta_z"]),
        np.asarray(split["eval_x"]),
        hidden=candidate.hidden,
        learning_rate=candidate.learning_rate,
        epochs=candidate.epochs,
        seed=seed,
    )


def _mean_metrics(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rmse": float(np.mean([float(item["rmse"]) for item in metrics])),
        "skill_ratio": float(np.mean([float(item["skill_ratio"]) for item in metrics])),
        "seed_metrics": [{"rmse": float(item["rmse"]), "skill_ratio": float(item["skill_ratio"])} for item in metrics],
    }


def select_individual_ridge(
    raw_series: np.ndarray,
    groups: Mapping[str, Sequence[int]],
    candidates: Sequence[RidgeCandidate],
) -> RidgeCandidate:
    scores = []
    for candidate in candidates:
        fold_scores = []
        for train_end in FOLDS:
            split = _prepared_split(raw_series, groups, train_end=train_end, evaluation_start=train_end, evaluation_end=train_end + 100, history=candidate.history)
            fold_scores.append(float(_evaluate_delta_prediction(split, _ridge_prediction(split, candidate))["skill_ratio"]))
        scores.append((float(np.mean(fold_scores)), candidate))
    return min(scores, key=lambda item: (item[0], item[1].history, -item[1].alpha))[1]


def select_individual_rnn(
    raw_series: np.ndarray,
    groups: Mapping[str, Sequence[int]],
    candidates: Sequence[RnnCandidate],
    *,
    seeds: Sequence[int],
) -> RnnCandidate:
    scores = []
    for candidate in candidates:
        fold_scores = []
        for fold_index, train_end in enumerate(FOLDS):
            split = _prepared_split(raw_series, groups, train_end=train_end, evaluation_start=train_end, evaluation_end=train_end + 100, history=candidate.history)
            for seed in seeds:
                fold_scores.append(float(_evaluate_delta_prediction(split, _rnn_prediction(split, candidate, seed=int(seed) + fold_index))["skill_ratio"]))
        scores.append((float(np.mean(fold_scores)), candidate))
    return min(scores, key=lambda item: (item[0], item[1].history, item[1].hidden, item[1].learning_rate))[1]


def _shared_predictions(
    splits: Mapping[str, Mapping[str, np.ndarray]],
    candidate: RnnCandidate,
    *,
    seed: int,
) -> dict[str, dict[str, Any]]:
    train_x = np.concatenate([np.asarray(split["train_x"]) for split in splits.values()], axis=0)
    train_delta = np.concatenate([np.asarray(split["train_delta_z"]) for split in splits.values()], axis=0)
    eval_x = np.concatenate([np.asarray(split["eval_x"]) for split in splits.values()], axis=0)
    all_delta_z = fit_rnn_predict(train_x, train_delta, eval_x, hidden=candidate.hidden, learning_rate=candidate.learning_rate, epochs=candidate.epochs, seed=seed)
    offset = 0
    output = {}
    for subject, split in splits.items():
        count = len(split["eval_x"])
        output[subject] = _evaluate_delta_prediction(split, all_delta_z[offset : offset + count])
        offset += count
    return output


def select_shared_rnn(
    raw_by_subject: Mapping[str, np.ndarray],
    groups: Mapping[str, Sequence[int]],
    candidates: Sequence[RnnCandidate],
    *,
    seeds: Sequence[int],
) -> RnnCandidate:
    scores = []
    for candidate in candidates:
        candidate_scores = []
        for fold_index, train_end in enumerate(FOLDS):
            splits = {
                subject: _prepared_split(raw, groups, train_end=train_end, evaluation_start=train_end, evaluation_end=train_end + 100, history=candidate.history)
                for subject, raw in raw_by_subject.items()
            }
            for seed in seeds:
                metrics = _shared_predictions(splits, candidate, seed=int(seed) + fold_index)
                candidate_scores.append(float(np.mean([float(item["skill_ratio"]) for item in metrics.values()])))
        scores.append((float(np.mean(candidate_scores)), candidate))
    return min(scores, key=lambda item: (item[0], item[1].history, item[1].hidden, item[1].learning_rate))[1]


def _load_subject_series(data_root: Path, *, limit: int | None) -> dict[str, np.ndarray]:
    output = {}
    for path in sorted(Path(data_root).glob("sub-*/*.mat")):
        values = np.asarray(loadmat(path)["Schaefer500"], dtype=float)
        if values.shape != (1200, 500) or not np.isfinite(values).all():
            raise ValueError(f"Expected finite [1200, 500] data in {path}, got {values.shape}.")
        output[path.parent.name] = values
        if limit is not None and len(output) >= int(limit):
            break
    if not output:
        raise FileNotFoundError(f"No Schaefer500 MAT files found under {data_root}.")
    return output


def _plot_subject_results(rows: Sequence[Mapping[str, Any]], paired: Mapping[str, Mapping[str, float | int]], destination: Path) -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8, "axes.spines.right": False, "axes.spines.top": False})
    colors = {"ridge": "#4C78A8", "individual_rnn": "#E17C05", "shared_rnn": "#59A14F"}
    display = {"ridge": "Tuned Ridge", "individual_rnn": "Individual RNN", "shared_rnn": "Shared RNN"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    rng = np.random.default_rng(19)
    for index, model in enumerate(("ridge", "individual_rnn", "shared_rnn")):
        values = np.asarray([float(row["rmse"]) for row in rows if str(row["model"]) == model])
        axes[0].scatter(index + rng.normal(0.0, 0.045, len(values)), values, color=colors[model], s=14, alpha=0.84)
        axes[0].plot([index - 0.18, index + 0.18], [values.mean(), values.mean()], color=colors[model], linewidth=1.5)
    axes[0].set(xticks=[0, 1, 2], xticklabels=[display[item] for item in ("ridge", "individual_rnn", "shared_rnn")], ylabel="Held-out RMSE (standardized)")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].text(0.02, 0.97, "a", transform=axes[0].transAxes, va="top", fontweight="bold")
    positions = np.arange(2)
    labels = ("Individual RNN − Ridge", "Shared RNN − Ridge")
    for position, key in zip(positions, ("individual_rnn_minus_ridge", "shared_rnn_minus_ridge")):
        values = np.asarray([float(row["rmse"]) for row in rows if str(row["model"]) == key.removesuffix("_minus_ridge")])
        ridge = np.asarray([float(row["rmse"]) for row in rows if str(row["model"]) == "ridge"])
        axes[1].scatter(values - ridge, position + rng.normal(0.0, 0.045, len(values)), color=colors[key.removesuffix("_minus_ridge")], s=14, alpha=0.84)
        statistic = paired[key]
        axes[1].errorbar(float(statistic["mean_delta_rmse"]), position, xerr=np.array([[float(statistic["mean_delta_rmse"]) - float(statistic["ci95_low"])], [float(statistic["ci95_high"]) - float(statistic["mean_delta_rmse"])]]) , fmt="o", color="black", capsize=2, markersize=3.5)
    axes[1].axvline(0.0, color="#5A5A5A", linewidth=0.8, linestyle="--")
    axes[1].set(yticks=positions, yticklabels=labels, xlabel="RNN − Ridge RMSE")
    axes[1].text(0.02, 0.97, "b", transform=axes[1].transAxes, va="top", fontweight="bold")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 300}), (".svg", {}), (".pdf", {})):
        fig.savefig(destination.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def _write_report(destination: Path, payload: Mapping[str, Any]) -> None:
    paired = payload["paired_rmse"]
    rows = payload["subject_rows"]
    models = ("ridge", "individual_rnn", "shared_rnn")
    means = {model: float(np.mean([float(row["rmse"]) for row in rows if row["model"] == model])) for model in models}
    lines = [
        "# Yeo7-PC1 RNN versus tuned Δ-Ridge",
        "",
        "## Experiment contract",
        "",
        "The treatment is predictor family: individually fitted vanilla RNN or a parameter-shared vanilla RNN, each compared with a tuned per-subject Δ-Ridge. All conditions use train-only Yeo7 PC1, chronological validation endpoints 600/700/800, a 900:1200 held-out test, one-step delta targets, and train-row standardization. The shared RNN pools sequence windows only; no hidden state crosses a subject boundary.",
        "",
        "## Held-out results",
        "",
        f"- Tuned per-subject Ridge mean RMSE: {means['ridge']:.6f}.",
        f"- Individual RNN mean RMSE: {means['individual_rnn']:.6f}; RNN − Ridge mean delta: {paired['individual_rnn_minus_ridge']['mean_delta_rmse']:.6f}, 95% paired-subject bootstrap [{paired['individual_rnn_minus_ridge']['ci95_low']:.6f}, {paired['individual_rnn_minus_ridge']['ci95_high']:.6f}], lower-RMSE subjects: {paired['individual_rnn_minus_ridge']['subjects_lower_rmse']}/{paired['individual_rnn_minus_ridge']['n_subjects']}.",
        f"- Shared RNN mean RMSE: {means['shared_rnn']:.6f}; RNN − Ridge mean delta: {paired['shared_rnn_minus_ridge']['mean_delta_rmse']:.6f}, 95% paired-subject bootstrap [{paired['shared_rnn_minus_ridge']['ci95_low']:.6f}, {paired['shared_rnn_minus_ridge']['ci95_high']:.6f}], lower-RMSE subjects: {paired['shared_rnn_minus_ridge']['subjects_lower_rmse']}/{paired['shared_rnn_minus_ridge']['n_subjects']}.",
        "",
        "Negative RNN − Ridge deltas favor RNN. This predictive comparison does not identify a neural mechanism and does not establish that the true dynamics are linear or nonlinear outside the specified data, representation, training budget, and test interval.",
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    data_root: Path,
    labels_path: Path,
    output_dir: Path,
    *,
    smoke: bool,
) -> dict[str, Any]:
    config = build_config(smoke=smoke)
    groups = load_yeo7_groups(labels_path)
    raw_by_subject = _load_subject_series(data_root, limit=config.subject_limit)
    subject_rows: list[dict[str, Any]] = []
    individual_rnn_choices = {}
    ridge_choices = {}
    for subject, raw in raw_by_subject.items():
        ridge = select_individual_ridge(raw, groups, config.ridge_candidates)
        rnn = select_individual_rnn(raw, groups, config.rnn_candidates, seeds=config.selection_seeds)
        ridge_choices[subject] = asdict(ridge)
        individual_rnn_choices[subject] = asdict(rnn)
        ridge_split = _prepared_split(raw, groups, train_end=DEVELOPMENT_END, evaluation_start=DEVELOPMENT_END, evaluation_end=TEST_END, history=ridge.history)
        ridge_metrics = _evaluate_delta_prediction(ridge_split, _ridge_prediction(ridge_split, ridge))
        subject_rows.append({"subject": subject, "model": "ridge", "candidate": asdict(ridge), **_mean_metrics([ridge_metrics])})
        rnn_split = _prepared_split(raw, groups, train_end=DEVELOPMENT_END, evaluation_start=DEVELOPMENT_END, evaluation_end=TEST_END, history=rnn.history)
        rnn_metrics = [_evaluate_delta_prediction(rnn_split, _rnn_prediction(rnn_split, rnn, seed=seed)) for seed in config.final_seeds]
        subject_rows.append({"subject": subject, "model": "individual_rnn", "candidate": asdict(rnn), **_mean_metrics(rnn_metrics)})
    shared_rnn = select_shared_rnn(raw_by_subject, groups, config.rnn_candidates, seeds=config.selection_seeds)
    shared_splits = {subject: _prepared_split(raw, groups, train_end=DEVELOPMENT_END, evaluation_start=DEVELOPMENT_END, evaluation_end=TEST_END, history=shared_rnn.history) for subject, raw in raw_by_subject.items()}
    shared_by_seed = [_shared_predictions(shared_splits, shared_rnn, seed=seed) for seed in config.final_seeds]
    for subject in raw_by_subject:
        subject_rows.append({"subject": subject, "model": "shared_rnn", "candidate": asdict(shared_rnn), **_mean_metrics([items[subject] for items in shared_by_seed])})
    paired = summarize_subject_rows(subject_rows)
    payload: dict[str, Any] = {
        "contract": {
            "scientific_question": "What changes when only the predictor family changes from tuned Δ-Ridge to a vanilla RNN?",
            "treatment_levels": ["per-subject tuned Δ-Ridge", "per-subject tuned vanilla RNN", "shared vanilla RNN"],
            "unit_of_pairing": "subject",
            "primary_metric": "held-out 900:1200 standardized one-step RMSE",
            "fixed": ["raw Schaefer500 source", "Yeo7 network PC1 fitted only on each train prefix", "chronological validation endpoints 600/700/800", "delta target", "train-row standardization", "three final RNN seeds", "held-out time interval"],
            "shared_model_note": "The shared RNN pools standardized subject windows but never passes a hidden state across subject boundaries.",
        },
        "config": {"smoke": bool(smoke), "n_subjects": len(raw_by_subject), "ridge_candidates": [asdict(item) for item in config.ridge_candidates], "rnn_candidates": [asdict(item) for item in config.rnn_candidates], "selection_seeds": list(config.selection_seeds), "final_seeds": list(config.final_seeds)},
        "selected_ridge_by_subject": ridge_choices,
        "selected_individual_rnn_by_subject": individual_rnn_choices,
        "selected_shared_rnn": asdict(shared_rnn),
        "subject_rows": subject_rows,
        "paired_rmse": paired,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _plot_subject_results(subject_rows, paired, output_dir / "heldout_subject_comparison")
    _write_report(output_dir / "report.md", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    payload = run(args.data_root, args.labels, args.output_dir, smoke=bool(args.smoke))
    print(json.dumps(payload["paired_rmse"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
