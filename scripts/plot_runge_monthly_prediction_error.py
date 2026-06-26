#!/usr/bin/env python3
"""Summarize monthly Runge prediction errors across climate variables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_runge_pairwise_mlp_ei import (  # noqa: E402
    build_lagged_dataset,
    load_component_scores,
    regression_metrics,
    split_temporal_arrays,
)


DATASETS = [
    ("slp_monthly", "SLP"),
    ("t2m_monthly", "2m air temperature"),
    ("air1000_monthly", "1000hPa air temperature"),
    ("sst_monthly", "SST"),
]

MODEL_FILES = [
    ("selected", "Selected MLP/blend", "mlp_metrics.csv"),
    ("ridge_current", "Ridge current", "linear_baseline_metrics.csv"),
    ("ridge_alpha1", "Ridge alpha=1", "reference_linear_alpha1_metrics.csv"),
]

COLORS = {
    "Persistence": "#7f7f7f",
    "Selected MLP/blend": "#4c78a8",
    "Ridge current": "#f58518",
    "Ridge alpha=1": "#54a24b",
}


def _overall_metric(frame: pd.DataFrame, split: str = "test") -> pd.Series:
    if "split" in frame.columns:
        return frame[(frame["split"] == split) & (frame["component"] == "overall")].iloc[0]
    return frame[frame["component"] == "overall"].iloc[0]


def collect_prediction_error(result_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    for dataset, label in DATASETS:
        dataset_root = result_root / dataset
        result_dir = dataset_root / "mlp_tm_ei/results/runge/pairwise_mlp_tm_ei_path_effects"
        manifest = json.loads((result_dir / "manifest.json").read_text())
        config = manifest["config"]
        frame = load_component_scores(dataset_root / "component_monthly_scores.csv")
        features, targets = build_lagged_dataset(
            frame,
            lag=int(config["lag"]),
            horizon=int(config["horizon"]),
        )
        splits = split_temporal_arrays(
            features,
            targets,
            train_fraction=float(config["train_fraction"]),
            val_fraction=float(config["val_fraction"]),
        )
        _, y_test = splits["test"]
        zero_rmse = float(np.sqrt(np.mean(y_test**2)))
        zero_component_rmse = np.sqrt(np.mean(y_test**2, axis=0))

        persistence_pred = splits["test"][0][:, -y_test.shape[1] :]
        persistence_metrics = regression_metrics(persistence_pred, y_test, list(frame.columns))
        persistence_overall = _overall_metric(persistence_metrics)
        rows.append(
            {
                "dataset": dataset,
                "label": label,
                "model": "Persistence",
                "rmse": float(persistence_overall["rmse"]),
                "mae": float(persistence_overall["mae"]),
                "corr": float(persistence_overall["corr"]),
                "relative_rmse": float(persistence_overall["rmse"] / zero_rmse),
                "zero_rmse": zero_rmse,
                "mlp_weight": np.nan,
            }
        )
        for idx, row in persistence_metrics[persistence_metrics["component"] != "overall"].reset_index(drop=True).iterrows():
            component_rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "component": row["component"],
                    "model": "Persistence",
                    "rmse": float(row["rmse"]),
                    "relative_rmse": float(row["rmse"] / zero_component_rmse[int(idx)]),
                }
            )

        blend = manifest.get("linear_blend", {})
        mlp_weight = float(blend.get("mlp_weight", np.nan)) if blend.get("enabled", False) else np.nan
        for _, model_label, filename in MODEL_FILES:
            metrics = pd.read_csv(result_dir / filename)
            overall = _overall_metric(metrics)
            rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "model": model_label,
                    "rmse": float(overall["rmse"]),
                    "mae": float(overall["mae"]),
                    "corr": float(overall["corr"]),
                    "relative_rmse": float(overall["rmse"] / zero_rmse),
                    "zero_rmse": zero_rmse,
                    "mlp_weight": mlp_weight if model_label == "Selected MLP/blend" else np.nan,
                }
            )
            component_metrics = metrics[(metrics["split"] == "test") & (metrics["component"] != "overall")].reset_index(drop=True)
            for idx, row in component_metrics.iterrows():
                component_rows.append(
                    {
                        "dataset": dataset,
                        "label": label,
                        "component": row["component"],
                        "model": model_label,
                        "rmse": float(row["rmse"]),
                        "relative_rmse": float(row["rmse"] / zero_component_rmse[int(idx)]),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(component_rows)


def plot_prediction_error(summary: pd.DataFrame, output: Path, save_svg: bool = False) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
        }
    )
    dataset_labels = [label for _, label in DATASETS]
    model_labels = ["Persistence", "Selected MLP/blend", "Ridge current", "Ridge alpha=1"]
    x = np.arange(len(dataset_labels), dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(model_labels))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.0), constrained_layout=True)
    for model_label, offset in zip(model_labels, offsets):
        data = summary[summary["model"] == model_label].set_index("label").loc[dataset_labels]
        axes[0].scatter(
            x + offset,
            data["relative_rmse"],
            s=54,
            color=COLORS[model_label],
            edgecolor="white",
            linewidth=0.6,
            label=model_label,
            zorder=3,
        )
        axes[1].scatter(
            x + offset,
            data["corr"],
            s=54,
            color=COLORS[model_label],
            edgecolor="white",
            linewidth=0.6,
            label=model_label,
            zorder=3,
        )

    axes[0].axhline(1.0, color="#6e6e6e", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Relative RMSE vs zero predictor")
    axes[0].set_ylim(0.50, 1.25)
    axes[1].set_ylabel("Test correlation")
    axes[1].set_ylim(0.15, 0.86)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels, rotation=18, ha="right")
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if save_svg:
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_markdown(summary: pd.DataFrame, component_summary: pd.DataFrame, output: Path) -> None:
    selected = summary[summary["model"] == "Selected MLP/blend"].copy()
    persistence = component_summary[component_summary["model"] == "Persistence"].set_index(["dataset", "component"])
    selected_components = component_summary[component_summary["model"] == "Selected MLP/blend"].set_index(["dataset", "component"])
    beats = []
    for dataset, label in DATASETS:
        common = selected_components.loc[dataset].join(
            persistence.loc[dataset],
            lsuffix="_selected",
            rsuffix="_persistence",
        )
        beats.append(
            {
                "dataset": dataset,
                "label": label,
                "selected_beats_persistence": int((common["rmse_selected"] < common["rmse_persistence"]).sum()),
            }
        )
    beats_frame = pd.DataFrame(beats).set_index("dataset")
    selected = selected.set_index("dataset").join(beats_frame[["selected_beats_persistence"]]).reset_index()

    lines = [
        "# Monthly Runge Prediction Error Summary",
        "",
        "| dataset | selected rel RMSE | selected corr | MLP blend weight | components beating persistence | best overall rel RMSE model |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for dataset, label in DATASETS:
        ds_summary = summary[summary["dataset"] == dataset].copy()
        best = ds_summary.sort_values("relative_rmse", ascending=True).iloc[0]
        row = selected[selected["dataset"] == dataset].iloc[0]
        lines.append(
            f"| {label} | {row['relative_rmse']:.3f} | {row['corr']:.3f} | "
            f"{row['mlp_weight']:.2f} | {int(row['selected_beats_persistence'])}/60 | "
            f"{best['model']} ({best['relative_rmse']:.3f}) |"
        )
    lines.extend(
        [
            "",
            "Relative RMSE is normalized by the test-set zero-predictor RMSE for the same dataset.",
            "The selected model is the saved validation-selected MLP/ridge blend used for downstream EI.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=Path("results/runge_monthly_variable_comparison"))
    parser.add_argument("--output", type=Path, default=Path("fig/runge_monthly_variable_comparison/prediction_error.png"))
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("results/runge_monthly_variable_comparison/prediction_error_summary.md"),
    )
    parser.add_argument("--save-svg", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, component_summary = collect_prediction_error(args.result_root)
    plot_prediction_error(summary, args.output, save_svg=bool(args.save_svg))
    write_markdown(summary, component_summary, args.summary_md)
    print(args.output)
    print(args.summary_md)


if __name__ == "__main__":
    main()
