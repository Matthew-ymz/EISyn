#!/usr/bin/env python3
"""Round the original sine DGP to one decimal while preserving qualitative trends."""

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


ORIGINAL_SUMMARY = DEFAULT_RESULT_DIR / "summary.json"
PILOT_BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
PILOT_SEEDS = (0, 1)
FULL_SEEDS = (0, 1, 2, 3)
BASE_ROUNDED = {
    "w_memory": 0.8,
    "x_memory": 0.4,
    "y_memory": 0.4,
    "w_to_x": 0.8,
    "w_to_y": 0.8,
    "z_memory": 0.2,
}
CANDIDATES = {
    "wz_0p1": {**BASE_ROUNDED, "w_to_z": 0.1},
    "wz_0p2": {**BASE_ROUNDED, "w_to_z": 0.2},
}
METRICS = {
    "xy_corr": "xy_observed_corr",
    "observational_wms": "observational_wms",
    "mmi_pid_synergy": "mmi_pid_xy_synergy",
    "surd_synergy": "surd_xy_synergy",
    "shap_interaction": "shap_xy_mean_abs_interaction",
    "mlp_peid_synergy": "tm_peid_xy_synergy",
}
EXPECTED_SIGNS = {
    "xy_corr": 1,
    "observational_wms": -1,
    "mmi_pid_synergy": -1,
    "surd_synergy": -1,
    "shap_interaction": 1,
    "mlp_peid_synergy": -1,
}
DEFAULT_RESULT = DEFAULT_RESULT_DIR / "sine_beta_original_neighborhood_one_decimal.json"
DEFAULT_FIGURE_STEM = "sine_beta_original_neighborhood_one_decimal_all_methods"
DEFAULT_AUDIT_STEM = "sine_beta_original_vs_one_decimal_slope_audit"


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _slopes(
    runs: Sequence[Mapping[str, object]],
    *,
    beta_values: Sequence[float] | None = None,
    seeds: Sequence[int] | None = None,
) -> dict[str, float]:
    frame = pd.DataFrame(runs)
    if beta_values is not None:
        allowed = {round(float(value), 8) for value in beta_values}
        frame = frame[frame["beta"].round(8).isin(allowed)]
    if seeds is not None:
        frame = frame[frame["seed"].astype(int).isin([int(seed) for seed in seeds])]
    mean = frame.groupby("beta", as_index=False).mean(numeric_only=True).sort_values("beta")
    beta = mean["beta"].to_numpy(dtype=float)
    return {
        name: float(np.polyfit(beta, mean[column].to_numpy(dtype=float), 1)[0])
        for name, column in METRICS.items()
    }


def _per_seed_signs(runs: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    frame = pd.DataFrame(runs)
    output: dict[str, dict[str, object]] = {}
    for name, column in METRICS.items():
        expected = EXPECTED_SIGNS[name]
        slopes: list[float] = []
        for _seed, seed_frame in frame.groupby(frame["seed"].astype(int)):
            seed_frame = seed_frame.sort_values("beta")
            slopes.append(
                float(
                    np.polyfit(
                        seed_frame["beta"].to_numpy(dtype=float),
                        seed_frame[column].to_numpy(dtype=float),
                        1,
                    )[0]
                )
            )
        output[name] = {
            "slopes": slopes,
            "expected_sign": expected,
            "matching_seeds": int(sum(np.sign(value) == expected for value in slopes)),
            "total_seeds": len(slopes),
        }
    return output


def _candidate_score(candidate: Mapping[str, float], reference: Mapping[str, float]) -> dict[str, object]:
    sign_matches = {
        name: bool(np.sign(float(candidate[name])) == EXPECTED_SIGNS[name])
        for name in METRICS
    }
    normalized_error = float(
        sum(
            ((float(candidate[name]) - float(reference[name])) / max(abs(float(reference[name])), 0.05)) ** 2
            for name in METRICS
        )
    )
    mismatch_count = int(sum(not value for value in sign_matches.values()))
    return {
        "sign_matches": sign_matches,
        "mismatch_count": mismatch_count,
        "normalized_slope_error": normalized_error,
        "selection_score": float(100.0 * mismatch_count + normalized_error),
    }


def _plot_slope_audit(
    original_slopes: Mapping[str, float],
    rounded_slopes: Mapping[str, float],
    *,
    stem: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    panels = [
        ("Information readouts", ["observational_wms", "mmi_pid_synergy", "surd_synergy", "mlp_peid_synergy"]),
        ("Native readouts", ["xy_corr", "shap_interaction"]),
    ]
    labels = {
        "xy_corr": r"corr$(x,y)$",
        "observational_wms": "Obs. WMS",
        "mmi_pid_synergy": "MMI-PID synergy",
        "surd_synergy": "SURD synergy",
        "shap_interaction": "SHAP interaction",
        "mlp_peid_synergy": "MLP+PEID synergy",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), constrained_layout=True)
    width = 0.34
    for ax, (title, names) in zip(axes, panels):
        y = np.arange(len(names))
        ax.barh(y - width / 2, [original_slopes[name] for name in names], height=width, color="#8F8F8F", label="Original")
        ax.barh(y + width / 2, [rounded_slopes[name] for name in names], height=width, color="#009E73", label="One-decimal")
        ax.axvline(0.0, color="#374151", linewidth=0.8)
        ax.set_yticks(y, [labels[name] for name in names])
        ax.set_xlabel(r"Slope per unit $\beta$")
        ax.set_title(title, fontsize=8)
        ax.grid(axis="x", alpha=0.18, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.05, 1.02), ncol=2)
    DEFAULT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = DEFAULT_FIGURE_DIR / stem
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return output.with_suffix(".png")


def run(*, smoke: bool = False) -> dict[str, object]:
    stored = _read_json(ORIGINAL_SUMMARY)["sine_beta_common_driver_sweep"]
    original_runs = stored["runs"]
    if smoke:
        beta_values = (0.0, 1.0)
        seeds = (0,)
        n_samples, epochs, intervention_samples = 320, 5, 64
        candidates = {"wz_0p2": CANDIDATES["wz_0p2"]}
        suffix = "_smoke"
    else:
        beta_values = PILOT_BETAS
        seeds = PILOT_SEEDS
        n_samples, epochs, intervention_samples = 1100, 90, 640
        candidates = CANDIDATES
        suffix = ""

    pilot_results: dict[str, object] = {}
    reference_pilot_slopes = _slopes(original_runs, beta_values=beta_values, seeds=seeds)
    for candidate_id, dynamics in candidates.items():
        checkpoint = DEFAULT_RESULT_DIR / f"sine_beta_rounded_{candidate_id}_runs{suffix}.jsonl"
        result = run_sine_beta_common_driver_sweep(
            beta_values=beta_values,
            seeds=seeds,
            n_samples=n_samples,
            alpha=1.0,
            noise=0.05,
            mlp_epochs=epochs,
            intervention_samples=intervention_samples,
            neural_granger_epochs=5 if smoke else 120,
            pcmci_cmiknn_sig_samples=5 if smoke else 30,
            dynamics_coefficients=dynamics,
            checkpoint_path=checkpoint,
            show_progress=True,
        )
        slopes = _slopes(result["runs"])
        pilot_results[candidate_id] = {
            "dynamics": dynamics,
            "slopes": slopes,
            "score": _candidate_score(slopes, reference_pilot_slopes),
            "result": result,
        }

    if smoke:
        payload = {
            "mode": "smoke",
            "reference_slopes": reference_pilot_slopes,
            "pilot": pilot_results,
        }
        _write_json(DEFAULT_RESULT.with_name(DEFAULT_RESULT.stem + suffix + ".json"), payload)
        return payload

    selected_id = min(
        pilot_results,
        key=lambda key: pilot_results[key]["score"]["selection_score"],
    )
    selected_dynamics = CANDIDATES[selected_id]
    full_checkpoint = DEFAULT_RESULT_DIR / f"sine_beta_rounded_{selected_id}_runs.jsonl"
    full = run_sine_beta_common_driver_sweep(
        beta_values=BETA_COMMON_DRIVER_SWEEP_VALUES,
        seeds=FULL_SEEDS,
        n_samples=1100,
        alpha=1.0,
        noise=0.05,
        mlp_epochs=90,
        intervention_samples=640,
        neural_granger_epochs=120,
        pcmci_cmiknn_sig_samples=30,
        dynamics_coefficients=selected_dynamics,
        checkpoint_path=full_checkpoint,
        show_progress=True,
    )
    liang = run_liang(
        beta_values=BETA_COMMON_DRIVER_SWEEP_VALUES,
        seeds=FULL_SEEDS,
        n_samples=1100,
        alpha=1.0,
        noise=0.05,
        dynamics_coefficients=selected_dynamics,
        show_progress=True,
    )
    all_methods_figure = _plot_sine_beta_combined_readout_sweep(
        full,
        DEFAULT_FIGURE_DIR,
        liang_result=liang,
        stem=DEFAULT_FIGURE_STEM,
        include_oracle=False,
    )
    original_full_slopes = _slopes(original_runs)
    rounded_full_slopes = _slopes(full["runs"])
    audit_figure = _plot_slope_audit(
        original_full_slopes,
        rounded_full_slopes,
        stem=DEFAULT_AUDIT_STEM,
    )
    final_score = _candidate_score(rounded_full_slopes, original_full_slopes)
    payload = {
        "scientific_question": "Does one-decimal rounding preserve the original qualitative beta trends without ANOVA?",
        "function_anova": False,
        "original_dynamics": {
            "w_memory": 0.78,
            "x_memory": 0.42,
            "y_memory": 0.38,
            "w_to_x": 0.82,
            "w_to_y": 0.76,
            "z_memory": 0.22,
            "w_to_z": 0.15,
        },
        "pilot_reference_slopes": reference_pilot_slopes,
        "pilot": pilot_results,
        "selected_candidate": selected_id,
        "selected_dynamics": selected_dynamics,
        "full_result": full,
        "liang_result": liang,
        "original_full_slopes": original_full_slopes,
        "rounded_full_slopes": rounded_full_slopes,
        "full_score": final_score,
        "per_seed_signs": _per_seed_signs(full["runs"]),
        "all_methods_figure": str(all_methods_figure),
        "audit_figure": str(audit_figure),
    }
    _write_json(DEFAULT_RESULT, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(smoke=args.smoke)
    summary = {
        "mode": result.get("mode", "full"),
        "selected_candidate": result.get("selected_candidate"),
        "selected_dynamics": result.get("selected_dynamics"),
        "full_score": result.get("full_score"),
        "all_methods_figure": result.get("all_methods_figure"),
        "audit_figure": result.get("audit_figure"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
