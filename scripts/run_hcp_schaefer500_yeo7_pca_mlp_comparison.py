#!/usr/bin/env python3
"""Controlled one-step Ridge-vs-MLP comparison on train-only Yeo7-PC1 states."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30/sub-100206/sub-100206_hcp_s1200_rfMRI_REST1_LR_schaefer500-1000_yeo7.mat"
DEFAULT_LABELS = ROOT / "data" / "hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30/_atlas_labels/Schaefer2018_500Parcels_7Networks_order.txt"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "hcp_schaefer500_yeo7_pca_mlp_comparison"
DEFAULT_LOG_DIR = ROOT / "docs" / "log" / "hcp_schaefer500_yeo7_pca_mlp_comparison"
NETWORK_ORDER = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")


def default_yeo7_labels(parcel_count: int) -> Path:
    """Return the bundled Yeo7 label file for a supported Schaefer resolution."""
    count = int(parcel_count)
    if count not in {500, 1000}:
        raise ValueError("parcel_count must be 500 or 1000.")
    return DEFAULT_LABELS.with_name(f"Schaefer2018_{count}Parcels_7Networks_order.txt")


def default_data_key(parcel_count: int) -> str:
    count = int(parcel_count)
    if count not in {500, 1000}:
        raise ValueError("parcel_count must be 500 or 1000.")
    return f"Schaefer{count}"


def load_hcp_series(data_path: Path, *, parcel_count: int = 500, data_key: str | None = None) -> np.ndarray:
    """Load and validate one 1200-point Schaefer time series from a MAT file."""
    count = int(parcel_count)
    key = data_key or default_data_key(count)
    payload = loadmat(data_path)
    if key not in payload:
        raise ValueError(f"MAT file {data_path} does not contain {key!r}.")
    values = np.asarray(payload[key], dtype=float)
    if values.shape != (1200, count) or not np.isfinite(values).all():
        raise ValueError(f"Expected finite [1200, {count}] {key} data in {data_path}, got {values.shape}.")
    return values


@dataclass(frozen=True)
class Yeo7Pc1:
    network_order: tuple[str, ...]
    parcel_indices: tuple[np.ndarray, ...]
    models: tuple[PCA, ...]

    def transform(self, series: np.ndarray) -> np.ndarray:
        values = np.asarray(series, dtype=float)
        return np.column_stack(
            [model.transform(values[:, indices])[:, 0] for indices, model in zip(self.parcel_indices, self.models)]
        )


def load_yeo7_groups(path: Path, *, expected_parcels: int = 500) -> dict[str, list[int]]:
    groups = {name: [] for name in NETWORK_ORDER}
    for expected_index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"Malformed atlas-label row: {line!r}")
        index = int(fields[0]) - 1
        if index != expected_index:
            raise ValueError("Atlas labels must be ordered consecutively from 1.")
        tokens = fields[1].split("_")
        if len(tokens) < 3 or tokens[0] != "7Networks":
            raise ValueError(f"Unexpected Yeo7 label: {fields[1]!r}")
        network = tokens[2]
        if network not in groups:
            raise ValueError(f"Unknown Yeo7 network: {network!r}")
        groups[network].append(index)
    if sum(map(len, groups.values())) != int(expected_parcels) or any(not groups[name] for name in NETWORK_ORDER):
        raise ValueError(f"Expected exactly {expected_parcels} parcels with every Yeo7 network represented.")
    return groups


def fit_yeo7_pc1(train_series: np.ndarray, groups: Mapping[str, Sequence[int]]) -> Yeo7Pc1:
    values = np.asarray(train_series, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("train_series must contain at least three time points.")
    order = tuple(groups)
    indices = tuple(np.asarray(groups[name], dtype=int) for name in order)
    models = tuple(PCA(n_components=1, svd_solver="full").fit(values[:, item]) for item in indices)
    return Yeo7Pc1(order, indices, models)


def make_history_samples(series: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(series, dtype=float)
    if order < 1 or len(values) <= order:
        raise ValueError("order must be positive and shorter than the series.")
    history = np.concatenate([values[order - 1 - lag : -1 - lag] for lag in range(order)], axis=1)
    return history, values[order:], np.arange(order, len(values))


def standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(train, dtype=float).mean(axis=0, keepdims=True)
    scale = np.asarray(train, dtype=float).std(axis=0, ddof=1, keepdims=True)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (np.asarray(values, dtype=float) - mean) / scale, mean, scale


def _fit_mlp(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, *, hidden: int, learning_rate: float, epochs: int, seed: int) -> np.ndarray:
    import torch

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    net = torch.nn.Sequential(
        torch.nn.Linear(train_x.shape[1], int(hidden)), torch.nn.SiLU(),
        torch.nn.Linear(int(hidden), int(hidden)), torch.nn.SiLU(),
        torch.nn.Linear(int(hidden), train_y.shape[1]),
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    x = torch.tensor(train_x, dtype=torch.float32)
    y = torch.tensor(train_y, dtype=torch.float32)
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        loss = torch.mean((net(x) - y) ** 2)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return np.asarray(net(torch.tensor(eval_x, dtype=torch.float32)).cpu().tolist(), dtype=float)


def predict_split(
    raw_series: np.ndarray,
    groups: Mapping[str, Sequence[int]],
    *,
    train_end: int,
    evaluation_start: int,
    evaluation_end: int,
    order: int,
    model: str,
    alpha: float | None = None,
    hidden: int | None = None,
    learning_rate: float | None = None,
    epochs: int | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray | float]:
    reducer = fit_yeo7_pc1(np.asarray(raw_series)[:train_end], groups)
    series = reducer.transform(raw_series)
    history, next_state, target_indices = make_history_samples(series, order)
    train_mask = target_indices < train_end
    eval_mask = (target_indices >= evaluation_start) & (target_indices < evaluation_end)
    if not train_mask.any() or not eval_mask.any():
        raise ValueError("Split has no train or evaluation rows.")
    train_history, eval_history = history[train_mask], history[eval_mask]
    train_next, eval_next = next_state[train_mask], next_state[eval_mask]
    width = series.shape[1]
    train_x, x_mean, x_scale = standardize(train_history, train_history)
    eval_x = (eval_history - x_mean) / x_scale
    train_delta = train_next - train_history[:, :width]
    train_y, y_mean, y_scale = standardize(train_delta, train_delta)
    if model == "ridge":
        if alpha is None:
            raise ValueError("Ridge requires alpha.")
        predictor = Ridge(alpha=float(alpha), fit_intercept=True).fit(train_x, train_y)
        delta_z = predictor.predict(eval_x)
    elif model == "mlp":
        if None in {hidden, learning_rate, epochs}:
            raise ValueError("MLP requires hidden, learning_rate, and epochs.")
        delta_z = _fit_mlp(train_x, train_y, eval_x, hidden=int(hidden), learning_rate=float(learning_rate), epochs=int(epochs), seed=seed)
    else:
        raise ValueError("model must be 'ridge' or 'mlp'.")
    prediction = eval_history[:, :width] + delta_z * y_scale + y_mean
    _, state_mean, state_scale = standardize(train_history[:, :width], eval_next)
    truth_z = (eval_next - state_mean) / state_scale
    prediction_z = (prediction - state_mean) / state_scale
    persistence_z = (eval_history[:, :width] - state_mean) / state_scale
    time_mse = np.mean((truth_z - prediction_z) ** 2, axis=1)
    return {
        "rmse": float(np.sqrt(time_mse.mean())),
        "skill_ratio": float(np.sqrt(time_mse.mean()) / max(np.sqrt(np.mean((truth_z - persistence_z) ** 2)), 1e-12)),
        "time_mse": time_mse,
        "prediction": prediction_z,
        "truth": truth_z,
    }


def select_ridge(raw_series: np.ndarray, groups: Mapping[str, Sequence[int]], *, folds: Sequence[int]) -> dict[str, object]:
    candidates = []
    for order in (1, 2, 3, 5, 8):
        for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6):
            scores = [predict_split(raw_series, groups, train_end=end, evaluation_start=end, evaluation_end=end + 100, order=order, model="ridge", alpha=alpha)["skill_ratio"] for end in folds]
            candidates.append({"order": order, "alpha": alpha, "mean_validation_skill_ratio": float(np.mean(scores))})
    return min(candidates, key=lambda item: (item["mean_validation_skill_ratio"], item["order"], -item["alpha"]))


def select_mlp(raw_series: np.ndarray, groups: Mapping[str, Sequence[int]], *, folds: Sequence[int]) -> dict[str, object]:
    candidates = []
    for order in (1, 2, 3, 5, 8):
        for hidden in (4, 8, 16):
            for learning_rate in (1e-3, 3e-3):
                for epochs in (100, 300):
                    scores = []
                    for fold_index, end in enumerate(folds):
                        for seed in (101, 202, 303):
                            scores.append(float(predict_split(raw_series, groups, train_end=end, evaluation_start=end, evaluation_end=end + 100, order=order, model="mlp", hidden=hidden, learning_rate=learning_rate, epochs=epochs, seed=seed + fold_index)["skill_ratio"]))
                    candidates.append({"order": order, "hidden": hidden, "learning_rate": learning_rate, "epochs": epochs, "mean_validation_skill_ratio": float(np.mean(scores))})
    return min(candidates, key=lambda item: (item["mean_validation_skill_ratio"], item["order"], item["hidden"]))


def block_bootstrap(delta: np.ndarray, *, block: int = 12, replicates: int = 5000, seed: int = 90210) -> dict[str, float]:
    values = np.asarray(delta, dtype=float)
    starts = np.arange(0, len(values) - block + 1)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    blocks_needed = int(np.ceil(len(values) / block))
    for index in range(replicates):
        sample = np.concatenate([values[start : start + block] for start in rng.choice(starts, size=blocks_needed)])[: len(values)]
        draws[index] = sample.mean()
    return {"mean_delta": float(values.mean()), "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975)), "one_sided_p_mlp_better": float(np.mean(draws >= 0.0))}


def plot_result(path: Path, ridge_rmse: float, mlp_rmses: np.ndarray, bootstrap: Sequence[Mapping[str, float]]) -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8, "axes.spines.right": False, "axes.spines.top": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    rng = np.random.default_rng(7)
    axes[0].scatter(rng.normal(0, 0.035, len(mlp_rmses)), np.repeat(ridge_rmse, len(mlp_rmses)), color="#4C78A8", s=18)
    axes[0].scatter(rng.normal(1, 0.035, len(mlp_rmses)), mlp_rmses, color="#D17A22", s=18)
    axes[0].plot([-0.18, 0.18], [ridge_rmse, ridge_rmse], color="#4C78A8", linewidth=1.4)
    axes[0].plot([0.82, 1.18], [np.mean(mlp_rmses), np.mean(mlp_rmses)], color="#D17A22", linewidth=1.4)
    axes[0].set_xticks([0, 1], ["Tuned Ridge", "Tuned MLP"])
    axes[0].set_ylabel("Held-out RMSE (standardized)")
    axes[0].text(0.03, 0.96, "a", transform=axes[0].transAxes, va="top", fontweight="bold")
    means = np.asarray([item["mean_delta"] for item in bootstrap])
    low = np.asarray([item["ci95_low"] for item in bootstrap])
    high = np.asarray([item["ci95_high"] for item in bootstrap])
    positions = np.arange(1, len(means) + 1)
    axes[1].errorbar(means, positions, xerr=np.vstack((means - low, high - means)), fmt="o", color="#D17A22", ecolor="#D17A22", capsize=2, markersize=3.5)
    axes[1].axvline(0, color="#5A5A5A", linewidth=0.8, linestyle="--")
    axes[1].set(yticks=positions, yticklabels=[f"seed {seed}" for seed in range(1, len(means) + 1)], xlabel="MLP − Ridge test MSE", ylabel="MLP run")
    axes[1].text(0.03, 0.96, "b", transform=axes[1].transAxes, va="top", fontweight="bold")
    for suffix, kwargs in ((".png", {"dpi": 300}), (".svg", {}), (".pdf", {})):
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight", **kwargs)


def run(data_path: Path, labels_path: Path, output_dir: Path, log_dir: Path) -> dict[str, object]:
    raw = np.asarray(loadmat(data_path)["Schaefer500"], dtype=float)
    groups = load_yeo7_groups(labels_path)
    if raw.shape != (1200, 500) or not np.isfinite(raw).all():
        raise ValueError(f"Expected finite 1200 x 500 Schaefer500 series, got {raw.shape}.")
    folds = (600, 700, 800)
    ridge = select_ridge(raw, groups, folds=folds)
    mlp = select_mlp(raw, groups, folds=folds)
    ridge_test = predict_split(raw, groups, train_end=900, evaluation_start=900, evaluation_end=1200, model="ridge", alpha=float(ridge["alpha"]), order=int(ridge["order"]))
    final_seeds = tuple(range(1001, 1011))
    mlp_tests = [predict_split(raw, groups, train_end=900, evaluation_start=900, evaluation_end=1200, model="mlp", order=int(mlp["order"]), hidden=int(mlp["hidden"]), learning_rate=float(mlp["learning_rate"]), epochs=int(mlp["epochs"]), seed=seed) for seed in final_seeds]
    bootstrap = [block_bootstrap(np.asarray(result["time_mse"]) - np.asarray(ridge_test["time_mse"]), seed=seed) for seed, result in zip(final_seeds, mlp_tests)]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = {"contract": {"treatment": "predictor family (delta Ridge vs delta MLP)", "fixed": ["subject sub-100206", "raw Schaefer500 input", "Yeo7 PC1 fitted only on each train segment", "one-step target", "chronological folds 600/700/800", "development end 900", "held-out test 900:1200", "standardization fitted on training rows"], "primary_metric": "held-out standardized RMSE", "significance": "moving-block bootstrap (block=12) on paired per-time MSE, one-sided MLP-better test"}, "representation": {"network_order": list(groups), "network_sizes": {name: len(indices) for name, indices in groups.items()}}, "ridge_selection": ridge, "mlp_selection": mlp, "ridge_test": {"rmse": ridge_test["rmse"], "skill_ratio": ridge_test["skill_ratio"]}, "mlp_test": [{"seed": seed, "rmse": result["rmse"], "skill_ratio": result["skill_ratio"], "bootstrap": statistic} for seed, result, statistic in zip(final_seeds, mlp_tests, bootstrap)]}
    payload["summary"] = {"ridge_test_rmse": ridge_test["rmse"], "mlp_test_rmse_mean": float(np.mean([result["rmse"] for result in mlp_tests])), "mlp_test_rmse_min": float(np.min([result["rmse"] for result in mlp_tests])), "mlp_lower_rmse_runs": int(np.sum(np.asarray([result["rmse"] for result in mlp_tests]) < float(ridge_test["rmse"]))), "mlp_significantly_lower_mse_runs": int(np.sum([item["ci95_high"] < 0.0 for item in bootstrap]))}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report = ["# HCP Schaefer-500 Yeo7-PC1: tuned Ridge vs tuned MLP", "", "## Contract", "", "Only predictor family differs: both models forecast the one-step 7D PC1 delta using the same chronological protocol. PCA is refit using only the relevant training segment, preventing test-segment leakage.", "", "## Held-out result", "", f"- Tuned Ridge: order={ridge['order']}, alpha={ridge['alpha']}, RMSE={ridge_test['rmse']:.6f}, skill ratio={ridge_test['skill_ratio']:.6f}.", f"- Tuned MLP: order={mlp['order']}, hidden={mlp['hidden']}, lr={mlp['learning_rate']}, epochs={mlp['epochs']}; mean RMSE={payload['summary']['mlp_test_rmse_mean']:.6f} across 10 final seeds.", f"- MLP lower RMSE: {payload['summary']['mlp_lower_rmse_runs']} / 10; bootstrap 95% CI entirely below zero: {payload['summary']['mlp_significantly_lower_mse_runs']} / 10.", "", "A negative MLP−Ridge MSE interval favors MLP. This single-subject test establishes predictive evidence only; it does not identify a neural mechanism."]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    plot_result(output_dir / "heldout_comparison", float(ridge_test["rmse"]), np.asarray([result["rmse"] for result in mlp_tests]), bootstrap)
    (log_dir / "live_status.md").write_text("# 实时状态\n\n## 当前状态\n- completed\n\n## 最近结果\n- 完成 sub-100206 的 Yeo7-PC1 tuned Ridge vs tuned MLP 对照。\n\n## 下一步\n- 读取 results 目录中的 report.md 与 heldout_comparison.png。\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args(argv)
    payload = run(args.data, args.labels, args.output_dir, args.log_dir)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
