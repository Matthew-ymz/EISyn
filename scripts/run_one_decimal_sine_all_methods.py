#!/usr/bin/env python3
"""Run and plot every comparable method on the one-decimal sine DGP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_granger_peid_mlp import (
    BETA_COMMON_DRIVER_SWEEP_VALUES,
    DEFAULT_FIGURE_DIR,
    DEFAULT_RESULT_DIR,
    _plot_sine_beta_combined_readout_sweep,
    run_sine_beta_common_driver_sweep,
)
from scripts.run_sine_beta_liang_information_flow import run_experiment as run_liang
DEFAULT_BASE_RESULT = DEFAULT_RESULT_DIR / "sine_beta_one_decimal_all_methods.json"
DEFAULT_LIANG_RESULT = DEFAULT_RESULT_DIR / "sine_beta_one_decimal_all_methods_liang.json"
DEFAULT_COMBINED_RESULT = (
    DEFAULT_RESULT_DIR / "sine_beta_one_decimal_all_methods_combined.json"
)
DEFAULT_FIGURE_STEM = "sine_beta_one_decimal_all_methods_comparison"
DEFAULT_TV_FIGURE_STEM = "sine_beta_one_decimal_direct_readout_absolute_tv"
DEFAULT_CHECKPOINT = (
    DEFAULT_RESULT_DIR / "sine_beta_one_decimal_all_methods_runs.jsonl"
)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected object in {path}")
    return payload


def _absolute_tv(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.abs(np.diff(array)).sum())


def _plot_direct_readout_absolute_tv(
    result: Mapping[str, object],
    *,
    stem: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(result["summary"]).sort_values("beta")
    panels = [
        (
            "Synergy",
            [
                ("MLP+PEID", "mlp_peid_xy_synergy_mean", "#009E73"),
                ("MMI-PID", "mmi_pid_xy_synergy_mean", "#B07AA1"),
                ("SURD", "surd_xy_synergy_mean", "#8F8F8F"),
                ("Observational WMS", "observational_wms_mean", "#4E79A7"),
            ],
        ),
        (
            r"$U_x$",
            [
                ("MLP+PEID", "mlp_peid_unique_x_mean", "#009E73"),
                ("MMI-PID", "mmi_pid_unique_x_mean", "#B07AA1"),
                ("SURD", "surd_unique_x_mean", "#8F8F8F"),
            ],
        ),
        (
            r"$U_y$",
            [
                ("MLP+PEID", "mlp_peid_unique_y_mean", "#009E73"),
                ("MMI-PID", "mmi_pid_unique_y_mean", "#B07AA1"),
                ("SURD", "surd_unique_y_mean", "#8F8F8F"),
            ],
        ),
    ]
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
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    for ax, (title, specs) in zip(axes, panels):
        labels = [label for label, _, _ in specs]
        values = [_absolute_tv(frame[column]) for _, column, _ in specs]
        colors = [color for _, _, color in specs]
        bars = ax.bar(np.arange(len(specs)), values, color=colors, width=0.68)
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=2, fontsize=6.5)
        ax.set_title(title, fontsize=8)
        ax.set_ylabel("Absolute TV (bits)")
        ax.set_xticks(np.arange(len(specs)), labels, rotation=27, ha="right")
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.margins(y=0.12)
    DEFAULT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = DEFAULT_FIGURE_DIR / stem
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return output.with_suffix(".png")


def run_all(*, smoke: bool = False) -> dict[str, object]:
    beta_values = (0.0, 1.0) if smoke else BETA_COMMON_DRIVER_SWEEP_VALUES
    seeds = (0,) if smoke else (0, 1, 2, 3)
    base_result = run_sine_beta_common_driver_sweep(
        beta_values=beta_values,
        seeds=seeds,
        n_samples=320 if smoke else 1100,
        alpha=1.0,
        noise=0.1,
        mlp_epochs=5 if smoke else 90,
        intervention_samples=64 if smoke else 640,
        neural_granger_epochs=5 if smoke else 120,
        pcmci_cmiknn_sig_samples=5 if smoke else 30,
        checkpoint_path=None if smoke else DEFAULT_CHECKPOINT,
        show_progress=True,
    )
    liang_result = run_liang(
        beta_values=tuple(float(value) for value in beta_values),
        seeds=tuple(int(seed) for seed in seeds),
        n_samples=320 if smoke else 1100,
        alpha=1.0,
        noise=0.1,
        show_progress=True,
    )
    suffix = "_smoke" if smoke else ""
    base_path = DEFAULT_BASE_RESULT.with_name(DEFAULT_BASE_RESULT.stem + suffix + ".json")
    liang_path = DEFAULT_LIANG_RESULT.with_name(DEFAULT_LIANG_RESULT.stem + suffix + ".json")
    combined_path = DEFAULT_COMBINED_RESULT.with_name(
        DEFAULT_COMBINED_RESULT.stem + suffix + ".json"
    )
    _write_json(base_path, base_result)
    _write_json(liang_path, liang_result)
    combined = dict(base_result)
    combined["mlp_peid_readout"] = {
        "observed_variables": ["w", "x", "y", "z"],
        "hidden_variables": [],
        "seeds": [int(seed) for seed in seeds],
        "readout": "direct TM-PEID of the full fitted MLP output; no functional ANOVA",
        "source_support": [-1.8, 1.8],
        "context_sampling": "random intervention samples from each beta-specific empirical support",
    }
    combined["comparison_contract"] = {
        "question": "How do native readouts respond when only beta changes in the one-decimal DGP?",
        "paired": [
            "dynamics",
            "beta grid",
            "seeds",
            "trajectory length",
            "MLP architecture and training budget",
        ],
        "method_native_scales": True,
        "mlp_peid_function_anova": False,
        "oracle_plotted": False,
        "oracle_used_for_robustness_ranking": False,
    }
    _write_json(combined_path, combined)
    figure = _plot_sine_beta_combined_readout_sweep(
        combined,
        DEFAULT_FIGURE_DIR,
        liang_result=liang_result,
        stem=DEFAULT_FIGURE_STEM + suffix,
        include_oracle=False,
    )
    tv_figure = _plot_direct_readout_absolute_tv(
        combined,
        stem=DEFAULT_TV_FIGURE_STEM + suffix,
    )
    required_columns = {
        "observational_x_to_z_mi_mean",
        "mmi_pid_unique_x_mean",
        "mlp_peid_unique_x_mean",
        "oracle_peid_unique_x_mean",
        "surd_unique_x_mean",
        "shap_x_to_z_mean_abs_mean",
        "pcmci_cmiknn_x_to_z_mean",
        "neural_granger_x_to_z_mean",
        "observational_wms_mean",
        "mmi_pid_xy_synergy_mean",
        "surd_xy_synergy_mean",
        "mlp_peid_xy_synergy_mean",
        "oracle_peid_xy_synergy_mean",
        "shap_xy_mean_abs_interaction_mean",
    }
    available = set(pd.DataFrame(combined["summary"]).columns)
    missing = sorted(required_columns - available)
    if missing or not liang_result.get("summary"):
        raise RuntimeError(f"Missing method outputs: {missing}")
    return {
        "base_result": str(base_path),
        "liang_result": str(liang_path),
        "combined_result": str(combined_path),
        "figure": str(figure),
        "absolute_tv_figure": str(tv_figure),
        "methods": [
            "Obs. MI/WMS",
            "MMI-PID",
            "MLP+PEID",
            "SURD",
            "SHAP",
            "PCMCI-CMIknn",
            "Neural Granger",
            "Liang IF",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run_all(smoke=parse_args().smoke)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
