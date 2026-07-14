#!/usr/bin/env python3
"""Select train-only Yeo7-PC1 delta-Ridge hyperparameters for a Schaefer resolution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import (
    DEFAULT_DATA,
    default_data_key,
    default_yeo7_labels,
    load_hcp_series,
    load_yeo7_groups,
    fit_yeo7_pc1,
    make_history_samples,
    standardize,
)


DEFAULT_OUTPUT_DIR = ROOT / "results" / "hcp_schaefer1000_yeo7_ridge_selection"


def select_delta_ridge(values: np.ndarray, groups: dict[str, list[int]], *, folds: Sequence[int]) -> dict[str, float | int]:
    """Select the existing Ridge grid while fitting each fold's PC1 representation once."""
    candidates: dict[tuple[int, float], list[float]] = {(order, alpha): [] for order in (1, 2, 3, 5, 8) for alpha in (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6)}
    for end in folds:
        reducer = fit_yeo7_pc1(values[:end], groups)
        series = reducer.transform(values[: end + 100])
        for order in (1, 2, 3, 5, 8):
            history, next_state, target_indices = make_history_samples(series, order)
            train_mask = target_indices < end
            eval_mask = (target_indices >= end) & (target_indices < end + 100)
            train_history, eval_history = history[train_mask], history[eval_mask]
            train_next, eval_next = next_state[train_mask], next_state[eval_mask]
            width = series.shape[1]
            train_x, x_mean, x_scale = standardize(train_history, train_history)
            eval_x = (eval_history - x_mean) / x_scale
            train_delta = train_next - train_history[:, :width]
            train_y, y_mean, y_scale = standardize(train_delta, train_delta)
            _, state_mean, state_scale = standardize(train_history[:, :width], eval_next)
            truth_z = (eval_next - state_mean) / state_scale
            persistence_z = (eval_history[:, :width] - state_mean) / state_scale
            baseline_rmse = max(float(np.sqrt(np.mean((truth_z - persistence_z) ** 2))), 1.0e-12)
            for alpha in (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6):
                delta_z = Ridge(alpha=alpha, fit_intercept=True).fit(train_x, train_y).predict(eval_x)
                prediction = eval_history[:, :width] + delta_z * y_scale + y_mean
                prediction_z = (prediction - state_mean) / state_scale
                candidates[(order, alpha)].append(float(np.sqrt(np.mean((truth_z - prediction_z) ** 2)) / baseline_rmse))
    rows = [{"order": order, "alpha": alpha, "mean_validation_skill_ratio": float(np.mean(scores))} for (order, alpha), scores in candidates.items()]
    return min(rows, key=lambda item: (item["mean_validation_skill_ratio"], item["order"], -item["alpha"]))


def run(
    data_path: Path,
    labels_path: Path,
    output_dir: Path,
    *,
    parcel_count: int = 1000,
    data_key: str | None = None,
    development_end: int = 900,
    folds: Sequence[int] = (600, 700, 800),
) -> dict[str, object]:
    """Choose p and alpha using chronological folds wholly inside development data."""
    count = int(parcel_count)
    key = data_key or default_data_key(count)
    raw = load_hcp_series(data_path, parcel_count=count, data_key=key)
    if not all(0 < int(end) < int(development_end) for end in folds):
        raise ValueError("Every validation-fold endpoint must lie inside the development segment.")
    groups = load_yeo7_groups(labels_path, expected_parcels=count)
    selection = select_delta_ridge(raw[:development_end], groups, folds=tuple(int(end) for end in folds))
    payload = {
        "subject": data_path.parent.name,
        "config": {
            "parcel_count": count,
            "data_key": key,
            "labels": str(labels_path),
            "network_sizes": {name: len(indices) for name, indices in groups.items()},
            "development_end": int(development_end),
            "fold_endpoints": [int(end) for end in folds],
            "candidate_orders": [1, 2, 3, 5, 8],
            "candidate_alphas": [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6],
            "selection_metric": "mean chronological validation skill ratio",
            "test_segment_used": False,
        },
        "selected_delta_ridge": selection,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        "# HCP Yeo7-PC1 Δ-Ridge 时间验证\n\n"
        f"- Subject: `{payload['subject']}`\n"
        f"- Resolution: Schaefer-{count}; source: `{key}`\n"
        f"- Development segment: first {development_end} points; folds: {', '.join(map(str, folds))}.\n"
        f"- Selected: `p={selection['order']}`, `alpha={selection['alpha']}`; mean validation skill ratio={selection['mean_validation_skill_ratio']:.6f}.\n"
        "- The last 300 points are excluded from selection and subsequent Phi/null fitting.\n",
        encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--parcel-count", type=int, choices=(500, 1000), default=1000)
    parser.add_argument("--data-key", default="", help="MAT variable name; defaults to Schaefer<parcel-count>.")
    parser.add_argument("--development-end", type=int, default=900)
    parser.add_argument("--folds", default="600,700,800")
    args = parser.parse_args(argv)
    folds = tuple(int(value.strip()) for value in args.folds.split(",") if value.strip())
    payload = run(args.data, args.labels or default_yeo7_labels(args.parcel_count), args.output_dir, parcel_count=args.parcel_count, data_key=args.data_key or None, development_end=args.development_end, folds=folds)
    print(json.dumps(payload["selected_delta_ridge"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
