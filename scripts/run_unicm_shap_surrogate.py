from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.unicm_peid_syn_analysis import (  # noqa: E402
    HISTORY_LENGTH,
    MODE_NAMES,
    overall_prediction_cache_path,
    sample_full_history_mode_inputs,
)


MODE_ORDER = tuple(MODE_NAMES.keys())
PREDICTION_LENGTH = 24
DEFAULT_CACHE_DIR = ROOT / "results" / "unicm_overall_ei_cpu_bound4_n8192" / "cache"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "unicm_shap_mode_attribution"
DEFAULT_REPORT_PATH = ROOT / "docs" / "reports" / "log" / "unicm_shap_vs_peid_enso_iod_summary.md"
DEFAULT_PEID_SUMMARIES = (
    ROOT / "results" / "unicm_full_history_mode_pair_syn_cpu_bound4_n8192" / "full_history_mode_pair_syn_summary.csv",
    ROOT / "results" / "unicm_full_history_mode_pair_syn_cpu_bound4_n8192_iod" / "full_history_mode_pair_syn_summary.csv",
)
DEFAULT_PEID_ROWS = (
    ROOT / "results" / "unicm_full_history_mode_pair_syn_cpu_bound4_n8192" / "full_history_mode_pair_syn_rows.jsonl",
    ROOT / "results" / "unicm_full_history_mode_pair_syn_cpu_bound4_n8192_iod" / "full_history_mode_pair_syn_rows.jsonl",
)
TARGET_DISPLAY = {"nino": "ENSO", "IOD": "IOD"}


def ordered_targets(values: Iterable[str]) -> list[str]:
    seen = list(dict.fromkeys(str(value) for value in values))
    preferred = [target for target in ("nino", "IOD") if target in seen]
    return preferred + [target for target in seen if target not in preferred]


def feature_names() -> list[str]:
    names: list[str] = []
    for month in range(HISTORY_LENGTH):
        lag = HISTORY_LENGTH - month
        for mode in MODE_ORDER:
            names.append(f"t-{lag}:{mode}")
    return names


def _mode_feature_indices(mode: str) -> list[int]:
    mode_index = MODE_ORDER.index(str(mode))
    return [month * len(MODE_ORDER) + mode_index for month in range(HISTORY_LENGTH)]


def mode_feature_groups() -> list[list[int]]:
    return [_mode_feature_indices(mode) for mode in MODE_ORDER]


def aggregate_feature_shap_to_modes(
    shap_values: np.ndarray,
    *,
    target: str,
    seed: int,
    lead: int,
    surrogate_r2: float,
) -> pd.DataFrame:
    values = np.asarray(shap_values, dtype=float)
    if values.ndim != 2 or values.shape[1] != HISTORY_LENGTH * len(MODE_ORDER):
        raise ValueError(
            "shap_values must have shape "
            f"(n_samples, {HISTORY_LENGTH * len(MODE_ORDER)}), got {tuple(values.shape)}."
        )
    cube = values.reshape(values.shape[0], HISTORY_LENGTH, len(MODE_ORDER))
    rows = []
    for mode_index, mode in enumerate(MODE_ORDER):
        mode_values = cube[:, :, mode_index]
        rows.append(
            {
                "target": str(target),
                "seed": int(seed),
                "lead": int(lead),
                "mode": str(mode),
                "mean_abs_shap": float(np.mean(np.abs(mode_values))),
                "mean_signed_shap": float(np.mean(mode_values)),
                "surrogate_r2": float(surrogate_r2),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["mean_abs_shap", "mode"], ascending=[False, True]).reset_index(drop=True)
    frame["rank_shap"] = np.arange(1, len(frame) + 1, dtype=int)
    return frame


def aggregate_feature_interactions_to_pairs(
    interaction_values: np.ndarray,
    *,
    target: str,
    seed: int,
    lead: int,
    surrogate_r2: float,
) -> pd.DataFrame:
    values = np.asarray(interaction_values, dtype=float)
    expected = HISTORY_LENGTH * len(MODE_ORDER)
    if values.ndim != 3 or values.shape[1:] != (expected, expected):
        raise ValueError(
            "interaction_values must have shape "
            f"(n_samples, {expected}, {expected}), got {tuple(values.shape)}."
        )

    rows = []
    for left_index, left_mode in enumerate(MODE_ORDER):
        left_features = _mode_feature_indices(left_mode)
        for right_index, right_mode in enumerate(MODE_ORDER[left_index:], start=left_index):
            right_features = _mode_feature_indices(right_mode)
            block = values[:, left_features][:, :, right_features]
            rows.append(
                {
                    "target": str(target),
                    "seed": int(seed),
                    "lead": int(lead),
                    "pair": f"{left_mode}|{right_mode}",
                    "left_source": str(left_mode),
                    "right_source": str(right_mode),
                    "is_diagonal": bool(left_index == right_index),
                    "mean_abs_interaction": float(np.mean(np.abs(block))),
                    "mean_signed_interaction": float(np.mean(block)),
                    "surrogate_r2": float(surrogate_r2),
                }
            )
    return pd.DataFrame(rows)


def exact_group_shapley_interactions(
    X: np.ndarray,
    baseline: np.ndarray,
    predict,
    *,
    group_names: Sequence[str],
    group_indices: Sequence[Sequence[int]] | None = None,
    coalition_batch_size: int = 128,
) -> np.ndarray:
    samples = np.asarray(X, dtype=float)
    baseline_array = np.asarray(baseline, dtype=float)
    if samples.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if baseline_array.ndim != 1 or baseline_array.shape[0] != samples.shape[1]:
        raise ValueError("baseline must be a 1D vector with the same feature count as X.")

    names = [str(name) for name in group_names]
    group_count = len(names)
    if group_indices is None:
        if samples.shape[1] != group_count:
            raise ValueError("group_indices is required when feature count differs from group count.")
        groups = [[index] for index in range(group_count)]
    else:
        groups = [[int(index) for index in indices] for indices in group_indices]
        if len(groups) != group_count:
            raise ValueError("group_indices length must match group_names length.")

    coalition_count = 1 << group_count
    predictions = np.empty((coalition_count, samples.shape[0]), dtype=float)
    batch_size = max(1, int(coalition_batch_size))
    for batch_start in range(0, coalition_count, batch_size):
        batch_masks = list(range(batch_start, min(coalition_count, batch_start + batch_size)))
        masked_batch = np.tile(baseline_array[None, None, :], (len(batch_masks), samples.shape[0], 1))
        for local_index, mask in enumerate(batch_masks):
            masked = masked_batch[local_index]
            for group_index, feature_indices in enumerate(groups):
                if mask & (1 << group_index):
                    masked[:, feature_indices] = samples[:, feature_indices]
        predicted = np.asarray(
            predict(masked_batch.reshape(len(batch_masks) * samples.shape[0], samples.shape[1])),
            dtype=float,
        ).reshape(len(batch_masks), samples.shape[0])
        predictions[batch_start : batch_start + len(batch_masks)] = predicted

    interactions = np.zeros((samples.shape[0], group_count, group_count), dtype=float)
    denominator = math.factorial(max(group_count - 1, 1))
    for left in range(group_count):
        for right in range(left + 1, group_count):
            total = np.zeros(samples.shape[0], dtype=float)
            excluded = (1 << left) | (1 << right)
            for mask in range(coalition_count):
                if mask & excluded:
                    continue
                subset_size = int(mask.bit_count())
                weight = (
                    math.factorial(subset_size)
                    * math.factorial(group_count - subset_size - 2)
                    / denominator
                )
                both = mask | excluded
                left_only = mask | (1 << left)
                right_only = mask | (1 << right)
                total += weight * (
                    predictions[both]
                    - predictions[left_only]
                    - predictions[right_only]
                    + predictions[mask]
                )
            interactions[:, left, right] = total
            interactions[:, right, left] = total
    return interactions


def aggregate_group_interactions_to_pairs(
    interaction_values: np.ndarray,
    *,
    target: str,
    seed: int,
    lead: int,
    surrogate_r2: float,
) -> pd.DataFrame:
    values = np.asarray(interaction_values, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (len(MODE_ORDER), len(MODE_ORDER)):
        raise ValueError(
            "interaction_values must have shape "
            f"(n_samples, {len(MODE_ORDER)}, {len(MODE_ORDER)}), got {tuple(values.shape)}."
        )
    rows = []
    for left_index, left_mode in enumerate(MODE_ORDER):
        for right_index, right_mode in enumerate(MODE_ORDER[left_index:], start=left_index):
            values_ij = values[:, left_index, right_index]
            rows.append(
                {
                    "target": str(target),
                    "seed": int(seed),
                    "lead": int(lead),
                    "pair": f"{left_mode}|{right_mode}",
                    "left_source": str(left_mode),
                    "right_source": str(right_mode),
                    "is_diagonal": bool(left_index == right_index),
                    "mean_abs_interaction": float(np.mean(np.abs(values_ij))),
                    "mean_signed_interaction": float(np.mean(values_ij)),
                    "surrogate_r2": float(surrogate_r2),
                    "interaction_backend": "exact_group_shapley",
                }
            )
    return pd.DataFrame(rows)


def summarize_modes(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(["target", "mode"], as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            mean_signed_shap=("mean_signed_shap", "mean"),
            min_surrogate_r2=("surrogate_r2", "min"),
        )
        .sort_values(["target", "mean_abs_shap", "mode"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    summary["rank_shap"] = summary.groupby("target").cumcount() + 1
    return summary


def summarize_mode_leads(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(["target", "mode", "lead"], as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            std_abs_shap=("mean_abs_shap", "std"),
            mean_signed_shap=("mean_signed_shap", "mean"),
            min_surrogate_r2=("surrogate_r2", "min"),
        )
        .sort_values(["target", "mode", "lead"])
        .reset_index(drop=True)
    )
    summary["std_abs_shap"] = summary["std_abs_shap"].fillna(0.0)
    return summary


def build_peid_source_ei_lead_summary(pair_rows: pd.DataFrame) -> pd.DataFrame:
    required = {"target", "seed", "lead", "left_source", "right_source", "left_ei", "right_ei"}
    missing = required - set(pair_rows.columns)
    if missing:
        raise ValueError(f"PEID pair rows missing required columns: {', '.join(sorted(missing))}")
    left = pair_rows[["target", "seed", "lead", "left_source", "left_ei"]].rename(
        columns={"left_source": "mode", "left_ei": "source_ei"}
    )
    right = pair_rows[["target", "seed", "lead", "right_source", "right_ei"]].rename(
        columns={"right_source": "mode", "right_ei": "source_ei"}
    )
    per_pair = pd.concat([left, right], ignore_index=True)
    per_seed = (
        per_pair.groupby(["target", "mode", "seed", "lead"], as_index=False)["source_ei"]
        .mean()
        .sort_values(["target", "mode", "seed", "lead"])
    )
    summary = (
        per_seed.groupby(["target", "mode", "lead"], as_index=False)["source_ei"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_peid_source_ei", "std": "std_peid_source_ei"})
        .sort_values(["target", "mode", "lead"])
        .reset_index(drop=True)
    )
    summary["std_peid_source_ei"] = summary["std_peid_source_ei"].fillna(0.0)
    return summary


def summarize_pair_interactions(rows: pd.DataFrame, *, top_k: int | None = None) -> pd.DataFrame:
    pairs = rows[~rows["is_diagonal"].astype(bool)].copy()
    summary = (
        pairs.groupby(["target", "pair", "left_source", "right_source"], as_index=False)
        .agg(
            mean_abs_interaction=("mean_abs_interaction", "mean"),
            mean_signed_interaction=("mean_signed_interaction", "mean"),
            min_surrogate_r2=("surrogate_r2", "min"),
        )
        .sort_values(["target", "mean_abs_interaction", "pair"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    summary["rank_shap_interaction"] = summary.groupby("target").cumcount() + 1
    if top_k is not None:
        summary = summary[summary["rank_shap_interaction"] <= int(top_k)].reset_index(drop=True)
    return summary


def summarize_pair_interaction_leads(rows: pd.DataFrame) -> pd.DataFrame:
    pairs = rows[~rows["is_diagonal"].astype(bool)].copy()
    summary = (
        pairs.groupby(["target", "pair", "left_source", "right_source", "lead"], as_index=False)["mean_abs_interaction"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_abs_interaction", "std": "std_abs_interaction"})
        .sort_values(["target", "pair", "lead"])
        .reset_index(drop=True)
    )
    summary["std_abs_interaction"] = summary["std_abs_interaction"].fillna(0.0)
    return summary


def summarize_peid_syn_leads(pair_rows: pd.DataFrame) -> pd.DataFrame:
    required = {"target", "pair", "left_source", "right_source", "lead", "seed", "syn"}
    missing = required - set(pair_rows.columns)
    if missing:
        raise ValueError(f"PEID pair rows missing required columns: {', '.join(sorted(missing))}")
    summary = (
        pair_rows.groupby(["target", "pair", "left_source", "right_source", "lead"], as_index=False)["syn"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_peid_syn", "std": "std_peid_syn"})
        .sort_values(["target", "pair", "lead"])
        .reset_index(drop=True)
    )
    summary["std_peid_syn"] = summary["std_peid_syn"].fillna(0.0)
    return summary


def load_peid_summaries(paths: Sequence[Path], *, targets: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    warnings: list[str] = []
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            frames.append(pd.read_csv(candidate))
        else:
            warnings.append(f"PEID pair summary missing: {candidate}")
    if frames:
        peid = pd.concat(frames, ignore_index=True)
        peid = peid.drop_duplicates(["target", "pair"], keep="first").reset_index(drop=True)
    else:
        peid = pd.DataFrame()
    for target in targets:
        if peid.empty or str(target) not in set(peid["target"].astype(str)):
            warnings.append(f"No PEID pair summary rows found for target={target}")
    return peid, warnings


def load_peid_pair_rows(paths: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            frames.append(pd.read_json(candidate, lines=True))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(["target", "pair", "seed", "lead"]).reset_index(drop=True)


def _peid_source_summary(peid: pd.DataFrame) -> pd.DataFrame:
    if peid.empty:
        return pd.DataFrame(columns=["target", "mode", "mean_peid_source_ei", "rank_peid_source_ei"])
    rows = []
    for row in peid.itertuples(index=False):
        rows.append({"target": row.target, "mode": row.left_source, "source_ei": float(row.mean_left_ei)})
        rows.append({"target": row.target, "mode": row.right_source, "source_ei": float(row.mean_right_ei)})
    frame = (
        pd.DataFrame(rows)
        .groupby(["target", "mode"], as_index=False)["source_ei"]
        .mean()
        .rename(columns={"source_ei": "mean_peid_source_ei"})
        .sort_values(["target", "mean_peid_source_ei", "mode"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    frame["rank_peid_source_ei"] = frame.groupby("target").cumcount() + 1
    return frame


def compare_shap_modes_with_peid(mode_summary: pd.DataFrame, peid: pd.DataFrame) -> pd.DataFrame:
    peid_modes = _peid_source_summary(peid)
    comparison = mode_summary.merge(peid_modes, on=["target", "mode"], how="left")
    comparison["peid_available"] = comparison["mean_peid_source_ei"].notna()
    return comparison.sort_values(["target", "rank_shap", "mode"]).reset_index(drop=True)


def compare_pair_interactions_with_peid(pair_summary: pd.DataFrame, peid: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "target",
        "pair",
        "mean_syn",
        "rank_within_target",
        "mean_joint_ei",
        "mean_left_ei",
        "mean_right_ei",
    ]
    if peid.empty:
        peid_pairs = pd.DataFrame(columns=columns)
    else:
        peid_pairs = peid[[column for column in columns if column in peid.columns]].copy()
        peid_pairs = peid_pairs.rename(
            columns={
                "rank_within_target": "rank_peid_syn",
                "mean_syn": "mean_peid_syn",
            }
        )
    comparison = pair_summary.merge(peid_pairs, on=["target", "pair"], how="left")
    comparison["peid_available"] = comparison["mean_peid_syn"].notna()
    return comparison.sort_values(["target", "rank_shap_interaction", "pair"]).reset_index(drop=True)


def _cache_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        n_samples=int(args.n_samples),
        sampling_seed=int(args.sampling_seed),
        intervention_bound=float(args.intervention_bound),
        start_month=int(args.start_month),
        device=str(args.device),
    )


def load_prediction_cache(cache_dir: Path, *, seed: int, args: argparse.Namespace) -> np.ndarray:
    path = overall_prediction_cache_path(Path(cache_dir), seed=int(seed), args=_cache_args(args))
    if not path.exists():
        raise FileNotFoundError(f"Missing UniCM prediction cache: {path}")
    with np.load(path, allow_pickle=False) as payload:
        targets = np.asarray(payload["all_mode_targets"], dtype=np.float32)
    expected_shape = (int(args.n_samples), PREDICTION_LENGTH, len(MODE_ORDER))
    if targets.shape != expected_shape:
        raise ValueError(f"Expected cache shape {expected_shape}, got {targets.shape} in {path}")
    return targets


def write_jsonl(rows: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _fit_surrogate_and_explain(
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int,
    n_estimators: int,
    shap_samples: int,
) -> tuple[pd.Series, np.ndarray, np.ndarray, object]:
    import shap
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=min(6144, max(2, X.shape[0] - 1024)),
        test_size=min(1024, max(1, X.shape[0] // 8)),
        random_state=int(random_state),
    )
    model = ExtraTreesRegressor(
        n_estimators=int(n_estimators),
        max_features=0.5,
        min_samples_leaf=2,
        random_state=int(random_state),
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    quality = pd.Series(
        {
            "surrogate_r2": float(r2_score(y_test, predicted)),
            "surrogate_mae": float(mean_absolute_error(y_test, predicted)),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        }
    )
    rng = np.random.default_rng(int(random_state))
    sample_count = min(int(shap_samples), len(X_test))
    sample_indices = rng.choice(len(X_test), size=sample_count, replace=False)
    X_shap = X_test[sample_indices]
    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(X_shap, check_additivity=False), dtype=float)
    return quality, shap_values, X_shap, model


def plot_outputs(mode_summary: pd.DataFrame, pair_summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    fig_dir = Path(output_dir) / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    targets = ordered_targets(mode_summary["target"].astype(str))
    fig, axes = plt.subplots(1, len(targets), figsize=(4.0 * len(targets), 3.0), constrained_layout=True, sharey=False)
    axes = np.atleast_1d(axes)
    for axis, target in zip(axes, targets):
        rows = mode_summary[mode_summary["target"].astype(str) == target].sort_values("rank_shap").head(10)
        axis.barh(rows["mode"], rows["mean_abs_shap"], color="#4C78A8", alpha=0.86)
        axis.invert_yaxis()
        axis.set_title(TARGET_DISPLAY.get(target, target), fontsize=8)
        axis.set_xlabel("Mean |SHAP|")
    axes[0].set_ylabel("Source mode")
    for suffix in (".png", ".svg"):
        path = fig_dir / f"unicm_enso_iod_shap_mode_ranking{suffix}"
        fig.savefig(path, dpi=600 if suffix == ".png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(targets), figsize=(4.4 * len(targets), 3.2), constrained_layout=True, sharey=False)
    axes = np.atleast_1d(axes)
    for axis, target in zip(axes, targets):
        rows = pair_summary[pair_summary["target"].astype(str) == target].sort_values("rank_shap_interaction").head(10)
        labels = rows["pair"].astype(str).str.replace("|", " + ", regex=False)
        axis.barh(labels, rows["mean_abs_interaction"], color="#B279A2", alpha=0.86)
        axis.invert_yaxis()
        axis.set_title(TARGET_DISPLAY.get(target, target), fontsize=8)
        axis.set_xlabel("Mean |SHAP interaction|")
    axes[0].set_ylabel("Source mode pair")
    for suffix in (".png", ".svg"):
        path = fig_dir / f"unicm_enso_iod_shap_pair_interactions{suffix}"
        fig.savefig(path, dpi=600 if suffix == ".png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _line_palette(labels: Sequence[str]) -> dict[str, object]:
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab20").colors
    return {str(label): colors[index % len(colors)] for index, label in enumerate(labels)}


def _plot_curve(
    ax,
    rows: pd.DataFrame,
    *,
    key: str,
    mean_col: str,
    std_col: str,
    labels: Sequence[str],
    palette: dict[str, object],
    label_map: dict[str, str] | None = None,
) -> list[object]:
    handles = []
    for label in labels:
        subset = rows[rows[key].astype(str) == str(label)].sort_values("lead")
        if subset.empty:
            continue
        x = subset["lead"].to_numpy(dtype=float)
        mean = subset[mean_col].to_numpy(dtype=float)
        std = subset[std_col].fillna(0.0).to_numpy(dtype=float)
        display = label_map.get(str(label), str(label)) if label_map else str(label)
        (line,) = ax.plot(
            x,
            mean,
            marker="o",
            markersize=2.0,
            linewidth=1.2,
            color=palette[str(label)],
            label=display,
        )
        ax.fill_between(x, mean - std, mean + std, color=palette[str(label)], alpha=0.08, linewidth=0)
        handles.append(line)
    ax.axhline(0.0, color="#888888", linewidth=0.7, linestyle=":")
    ax.set_xlim(0.5, PREDICTION_LENGTH + 0.5)
    ax.set_xlabel("Lead (months)")
    return handles


def plot_lead_curve_outputs(
    *,
    shap_mode_leads: pd.DataFrame,
    shap_pair_leads: pd.DataFrame,
    peid_source_leads: pd.DataFrame,
    peid_syn_leads: pd.DataFrame,
    pair_summary: pd.DataFrame,
    peid_summary: pd.DataFrame,
    output_dir: Path,
    top_k_pairs: int = 10,
) -> list[Path]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    fig_dir = Path(output_dir) / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    targets = ordered_targets(shap_mode_leads["target"].astype(str))
    mode_palette = _line_palette(MODE_ORDER)

    fig, axes = plt.subplots(
        len(targets),
        2,
        figsize=(9.2, 3.1 * len(targets)),
        constrained_layout=True,
        sharex=True,
    )
    axes = np.asarray(axes).reshape(len(targets), 2)
    mode_handles: list[object] = []
    for row_index, target in enumerate(targets):
        shap_rows = shap_mode_leads[shap_mode_leads["target"].astype(str) == target]
        peid_rows = peid_source_leads[peid_source_leads["target"].astype(str) == target]
        mode_handles = _plot_curve(
            axes[row_index, 0],
            shap_rows,
            key="mode",
            mean_col="mean_abs_shap",
            std_col="std_abs_shap",
            labels=MODE_ORDER,
            palette=mode_palette,
        )
        _plot_curve(
            axes[row_index, 1],
            peid_rows,
            key="mode",
            mean_col="mean_peid_source_ei",
            std_col="std_peid_source_ei",
            labels=MODE_ORDER,
            palette=mode_palette,
        )
        axes[row_index, 0].set_ylabel(f"{TARGET_DISPLAY.get(target, target)}\nMean |SHAP|")
        axes[row_index, 1].set_ylabel("Source EI (bits)")
        axes[row_index, 0].set_title(f"{TARGET_DISPLAY.get(target, target)}: SHAP attribution")
        axes[row_index, 1].set_title(f"{TARGET_DISPLAY.get(target, target)}: PEID source EI")
    fig.legend(mode_handles, [handle.get_label() for handle in mode_handles], loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    source_paths = []
    for suffix in (".png", ".svg"):
        path = fig_dir / f"unicm_enso_iod_shap_vs_peid_source_leads{suffix}"
        fig.savefig(path, dpi=600 if suffix == ".png" else None, bbox_inches="tight")
        source_paths.append(path)
    plt.close(fig)

    pair_paths = []
    fig, axes = plt.subplots(
        len(targets),
        2,
        figsize=(9.8, 3.2 * len(targets)),
        constrained_layout=True,
        sharex=True,
    )
    axes = np.asarray(axes).reshape(len(targets), 2)
    for row_index, target in enumerate(targets):
        top_shap = (
            pair_summary[pair_summary["target"].astype(str) == target]
            .sort_values("rank_shap_interaction")
            .head(int(top_k_pairs))["pair"]
            .astype(str)
            .tolist()
        )
        top_peid = (
            peid_summary[peid_summary["target"].astype(str) == target]
            .sort_values("rank_within_target")
            .head(int(top_k_pairs))["pair"]
            .astype(str)
            .tolist()
            if not peid_summary.empty
            else []
        )
        pairs = list(dict.fromkeys(top_shap + top_peid))[: max(int(top_k_pairs), len(top_shap))]
        palette = _line_palette(pairs)
        label_map = {pair: pair.replace("|", " + ") for pair in pairs}
        shap_rows = shap_pair_leads[shap_pair_leads["target"].astype(str) == target]
        peid_rows = peid_syn_leads[peid_syn_leads["target"].astype(str) == target]
        pair_handles = _plot_curve(
            axes[row_index, 0],
            shap_rows,
            key="pair",
            mean_col="mean_abs_interaction",
            std_col="std_abs_interaction",
            labels=pairs,
            palette=palette,
            label_map=label_map,
        )
        _plot_curve(
            axes[row_index, 1],
            peid_rows,
            key="pair",
            mean_col="mean_peid_syn",
            std_col="std_peid_syn",
            labels=pairs,
            palette=palette,
            label_map=label_map,
        )
        axes[row_index, 0].set_ylabel(f"{TARGET_DISPLAY.get(target, target)}\nMean |interaction|")
        axes[row_index, 1].set_ylabel("Syn (bits)")
        axes[row_index, 0].set_title(f"{TARGET_DISPLAY.get(target, target)}: SHAP interaction")
        axes[row_index, 1].set_title(f"{TARGET_DISPLAY.get(target, target)}: PEID pair Syn")
        axes[row_index, 1].legend(
            pair_handles,
            [handle.get_label() for handle in pair_handles],
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )
    for suffix in (".png", ".svg"):
        path = fig_dir / f"unicm_enso_iod_shap_vs_peid_pair_leads{suffix}"
        fig.savefig(path, dpi=600 if suffix == ".png" else None, bbox_inches="tight")
        pair_paths.append(path)
    plt.close(fig)
    return source_paths + pair_paths


def _fmt(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    return "nan" if not np.isfinite(number) else f"{number:.6f}"


def write_report(
    *,
    report_path: Path,
    mode_comparison: pd.DataFrame,
    pair_comparison: pd.DataFrame,
    quality: pd.DataFrame,
    warnings: Sequence[str],
    output_dir: Path,
    lead_figures: Sequence[Path] | None = None,
) -> None:
    lines = [
        "# UniCM SHAP vs PEID: ENSO and IOD",
        "",
        "本轮使用 frozen UniCM Modeformer 的 full-history prediction cache 训练 tree surrogate，",
        "再用 TreeSHAP 估计 mode-level 单源归因，并用 exact group Shapley interaction 估计 11 个 mode group 的二阶交互。没有重训 UniCM，也没有重新执行 checkpoint forward。",
        "",
        "SHAP/SHAP interaction 是 surrogate prediction attribution；PEID single-source EI 与 pair Syn 是干预分布下的信息分解。二者用于对照机制线索，不能直接等同。",
        "",
        "## Surrogate quality",
        "",
        "| Target | min R2 | mean R2 | low-quality fits (R2 < 0.95) |",
        "|---|---:|---:|---:|",
    ]
    for target, rows in quality.groupby("target"):
        low = int((rows["surrogate_r2"] < 0.95).sum())
        lines.append(f"| {TARGET_DISPLAY.get(str(target), str(target))} | {_fmt(rows['surrogate_r2'].min())} | {_fmt(rows['surrogate_r2'].mean())} | {low} |")

    low_total = int((quality["surrogate_r2"] < 0.95).sum())
    if low_total:
        lines.extend(
            [
                "",
                f"注意：`{low_total}/{len(quality)}` 个 seed-target-lead surrogate 的 R2 低于 `0.95`，主要影响长 lead 的归因强度解释；",
                "下表排名仍可作为 frozen UniCM prediction cache 的筛查读数，但不应被解释为高保真 surrogate 下的最终机制结论。",
            ]
        )

    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.extend(["", "## Single-source SHAP vs PEID EI", ""])
    for target in ordered_targets(mode_comparison["target"].astype(str)):
        rows = mode_comparison[mode_comparison["target"].astype(str) == target].sort_values("rank_shap").head(8)
        lines.extend(
            [
                f"### {TARGET_DISPLAY.get(target, target)}",
                "",
                "| SHAP rank | Mode | mean \\|SHAP\\| | PEID source EI | PEID rank |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for row in rows.itertuples(index=False):
            peid_rank = "" if pd.isna(getattr(row, "rank_peid_source_ei", np.nan)) else str(int(row.rank_peid_source_ei))
            lines.append(
                f"| {int(row.rank_shap)} | {row.mode} | {_fmt(row.mean_abs_shap)} | "
                f"{_fmt(getattr(row, 'mean_peid_source_ei', np.nan))} | {peid_rank} |"
            )
        lines.append("")

    lines.extend(["## Second-order SHAP interaction vs PEID Syn", ""])
    for target in ordered_targets(pair_comparison["target"].astype(str)):
        rows = pair_comparison[pair_comparison["target"].astype(str) == target].sort_values("rank_shap_interaction").head(10)
        lines.extend(
            [
                f"### {TARGET_DISPLAY.get(target, target)}",
                "",
                "| SHAP int. rank | Pair | mean \\|interaction\\| | PEID Syn | PEID rank |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for row in rows.itertuples(index=False):
            peid_rank = "" if pd.isna(getattr(row, "rank_peid_syn", np.nan)) else str(int(row.rank_peid_syn))
            lines.append(
                f"| {int(row.rank_shap_interaction)} | {str(row.pair).replace('|', ' + ')} | "
                f"{_fmt(row.mean_abs_interaction)} | {_fmt(getattr(row, 'mean_peid_syn', np.nan))} | {peid_rank} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Outputs",
            "",
            f"- output directory: `{Path(output_dir).relative_to(ROOT) if Path(output_dir).is_relative_to(ROOT) else output_dir}`",
            "- figures: `fig/unicm_enso_iod_shap_mode_ranking.*`, `fig/unicm_enso_iod_shap_pair_interactions.*`",
        ]
    )
    if lead_figures:
        names = ", ".join(f"`fig/{Path(path).name}`" for path in lead_figures if Path(path).suffix == ".png")
        lines.append(f"- lead curves: {names}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = [str(target) for target in args.targets]
    unknown = [target for target in targets if target not in MODE_NAMES]
    if unknown:
        raise ValueError(f"Unknown target mode(s): {', '.join(unknown)}")
    leads = list(range(int(args.lead_start), int(args.lead_end) + 1))
    if min(leads) < 1 or max(leads) > PREDICTION_LENGTH:
        raise ValueError("Lead range must stay within [1, 24].")

    X = sample_full_history_mode_inputs(
        n_samples=int(args.n_samples),
        intervention_bound=float(args.intervention_bound),
        seed=int(args.sampling_seed),
    ).reshape(int(args.n_samples), -1)

    mode_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []

    for seed in [int(seed) for seed in args.seeds]:
        predictions = load_prediction_cache(Path(args.cache_dir), seed=seed, args=args)
        for target in targets:
            target_index = MODE_NAMES[target]
            for lead in leads:
                random_state = int(seed * 10000 + target_index * 100 + lead)
                y = predictions[:, int(lead) - 1, target_index]
                quality, shap_values, X_shap, model = _fit_surrogate_and_explain(
                    X,
                    y,
                    random_state=random_state,
                    n_estimators=int(args.n_estimators),
                    shap_samples=int(args.shap_samples),
                )
                quality_dict = {
                    "target": target,
                    "seed": seed,
                    "lead": int(lead),
                    **quality.to_dict(),
                    "low_quality": bool(float(quality["surrogate_r2"]) < 0.95),
                }
                quality_rows.append(quality_dict)
                mode_rows.extend(
                    aggregate_feature_shap_to_modes(
                        shap_values,
                        target=target,
                        seed=seed,
                        lead=int(lead),
                        surrogate_r2=float(quality["surrogate_r2"]),
                    ).to_dict("records")
                )
                baseline = np.mean(X, axis=0)
                X_interaction = X_shap[: min(int(args.interaction_samples), len(X_shap))]
                interaction_values = exact_group_shapley_interactions(
                    X_interaction,
                    baseline,
                    model.predict,
                    group_names=MODE_ORDER,
                    group_indices=mode_feature_groups(),
                    coalition_batch_size=int(args.coalition_batch_size),
                )
                pair_rows.extend(
                    aggregate_group_interactions_to_pairs(
                        interaction_values,
                        target=target,
                        seed=seed,
                        lead=int(lead),
                        surrogate_r2=float(quality["surrogate_r2"]),
                    ).to_dict("records")
                )

    quality_frame = pd.DataFrame(quality_rows)
    mode_frame = pd.DataFrame(mode_rows)
    pair_frame = pd.DataFrame(pair_rows)
    mode_summary = summarize_modes(mode_frame)
    mode_lead_summary = summarize_mode_leads(mode_frame)
    pair_summary = summarize_pair_interactions(pair_frame)

    peid, warnings = load_peid_summaries([Path(path) for path in args.peid_summaries], targets=targets)
    peid_pair_rows = load_peid_pair_rows([Path(path) for path in args.peid_rows])
    mode_comparison = compare_shap_modes_with_peid(mode_summary, peid)
    pair_comparison = compare_pair_interactions_with_peid(pair_summary, peid)
    pair_lead_summary = summarize_pair_interaction_leads(pair_frame)
    if peid_pair_rows.empty:
        peid_source_leads = pd.DataFrame()
        peid_syn_leads = pd.DataFrame()
        lead_figure_paths: list[Path] = []
    else:
        peid_source_leads = build_peid_source_ei_lead_summary(peid_pair_rows)
        peid_syn_leads = summarize_peid_syn_leads(peid_pair_rows)
        lead_figure_paths = plot_lead_curve_outputs(
            shap_mode_leads=mode_lead_summary,
            shap_pair_leads=pair_lead_summary,
            peid_source_leads=peid_source_leads,
            peid_syn_leads=peid_syn_leads,
            pair_summary=pair_summary,
            peid_summary=peid,
            output_dir=output_dir,
        )

    quality_path = output_dir / "surrogate_quality.csv"
    mode_rows_path = output_dir / "shap_mode_rows.jsonl"
    pair_rows_path = output_dir / "shap_mode_pair_interaction_rows.jsonl"
    mode_summary_path = output_dir / "shap_mode_summary.csv"
    lead_summary_path = output_dir / "shap_lead_summary.csv"
    pair_summary_path = output_dir / "shap_mode_pair_interaction_summary.csv"
    pair_lead_summary_path = output_dir / "shap_pair_interaction_lead_summary.csv"
    peid_source_leads_path = output_dir / "peid_source_ei_lead_summary.csv"
    peid_syn_leads_path = output_dir / "peid_pair_syn_lead_summary.csv"
    mode_comparison_path = output_dir / "peid_shap_mode_comparison.csv"
    pair_comparison_path = output_dir / "peid_shap_pair_interaction_comparison.csv"
    warnings_path = output_dir / "warnings.json"

    quality_frame.to_csv(quality_path, index=False)
    write_jsonl(mode_rows, mode_rows_path)
    write_jsonl(pair_rows, pair_rows_path)
    mode_summary.to_csv(mode_summary_path, index=False)
    mode_lead_summary.to_csv(lead_summary_path, index=False)
    pair_summary.to_csv(pair_summary_path, index=False)
    pair_lead_summary.to_csv(pair_lead_summary_path, index=False)
    peid_source_leads.to_csv(peid_source_leads_path, index=False)
    peid_syn_leads.to_csv(peid_syn_leads_path, index=False)
    mode_comparison.to_csv(mode_comparison_path, index=False)
    pair_comparison.to_csv(pair_comparison_path, index=False)
    warnings_path.write_text(json.dumps(warnings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    figure_paths = plot_outputs(mode_summary, pair_summary, output_dir)
    if args.report_path:
        write_report(
            report_path=Path(args.report_path),
            mode_comparison=mode_comparison,
            pair_comparison=pair_comparison,
            quality=quality_frame,
            warnings=warnings,
            output_dir=output_dir,
            lead_figures=lead_figure_paths,
        )

    return {
        "quality": str(quality_path),
        "mode_summary": str(mode_summary_path),
        "pair_summary": str(pair_summary_path),
        "pair_lead_summary": str(pair_lead_summary_path),
        "peid_source_leads": str(peid_source_leads_path),
        "peid_syn_leads": str(peid_syn_leads_path),
        "mode_comparison": str(mode_comparison_path),
        "pair_comparison": str(pair_comparison_path),
        "warnings": str(warnings_path),
        "figures": [str(path) for path in figure_paths + lead_figure_paths],
        "report": str(args.report_path) if args.report_path else None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run UniCM prediction-cache surrogate SHAP analysis.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--peid-summaries", nargs="+", type=Path, default=list(DEFAULT_PEID_SUMMARIES))
    parser.add_argument("--peid-rows", nargs="+", type=Path, default=list(DEFAULT_PEID_ROWS))
    parser.add_argument("--targets", nargs="+", default=["nino", "IOD"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--lead-start", type=int, default=1)
    parser.add_argument("--lead-end", type=int, default=24)
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument("--sampling-seed", type=int, default=20260619)
    parser.add_argument("--intervention-bound", type=float, default=4.0)
    parser.add_argument("--start-month", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-estimators", type=int, default=64)
    parser.add_argument("--shap-samples", type=int, default=512)
    parser.add_argument("--interaction-samples", type=int, default=128)
    parser.add_argument("--coalition-batch-size", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    outputs = run_analysis(args)
    print(json.dumps(outputs, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
