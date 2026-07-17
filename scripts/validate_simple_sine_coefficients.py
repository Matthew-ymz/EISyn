#!/usr/bin/env python3
"""Re-run the Part 1 sine-beta experiment with simple 1/0.5 coefficients."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_granger_peid_mlp import (
    DEFAULT_FIGURE_DIR,
    DEFAULT_RESULT_DIR,
    _plot_sine_beta_combined_readout_sweep,
    run_sine_beta_common_driver_sweep,
)
from scripts.run_hidden_w_sine_beta_mlp_peid import (
    compare_hidden_and_full_state,
    plot_combined_with_hidden_w_mlp_readouts,
    plot_hidden_vs_observed_w_syn,
    plot_hidden_w_sweep,
    run_hidden_w_sine_beta_mlp_peid_sweep,
)
from scripts.run_sine_beta_liang_information_flow import (
    run_experiment as run_liang_experiment,
)


DEFAULT_LEGACY_SUMMARY = DEFAULT_RESULT_DIR / "summary.json"
DEFAULT_FULL_RESULT = (
    DEFAULT_RESULT_DIR / "sine_beta_simple_coefficients_full_state.json"
)
DEFAULT_HIDDEN_RESULT = (
    DEFAULT_RESULT_DIR / "sine_beta_simple_coefficients_hidden_w.json"
)
DEFAULT_LIANG_RESULT = DEFAULT_RESULT_DIR / "sine_beta_simple_coefficients_liang.json"
DEFAULT_VALIDATION_RESULT = (
    DEFAULT_RESULT_DIR / "sine_beta_simple_coefficients_validation.json"
)

FROZEN_CONFIG_KEYS = (
    "beta_values",
    "seeds",
    "n_samples",
    "alpha",
    "noise",
    "mlp_epochs",
    "intervention_samples",
    "neural_granger_epochs",
    "pcmci_cmiknn_sig_samples",
    "pcmci_cmiknn_knn",
    "peid_source_support",
    "oracle_intervention_support",
    "oracle_intervention_seed",
)

PRIMARY_DIRECTIONS = {
    "xy_observed_corr": 1,
    "observational_wms": -1,
    "mmi_pid_xy_synergy": -1,
}


def _read_legacy_beta_result(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("sine_beta_common_driver_sweep")
    if not isinstance(result, dict) or not result.get("runs"):
        raise ValueError(f"No legacy sine beta sweep in {path}")
    return result


def _seed_slopes(result: Mapping[str, object], metric: str) -> dict[int, float]:
    frame = pd.DataFrame(result["runs"])
    slopes: dict[int, float] = {}
    for seed, group in frame.groupby("seed"):
        ordered = group.sort_values("beta")
        slope, _ = np.polyfit(
            ordered["beta"].to_numpy(dtype=float),
            ordered[metric].to_numpy(dtype=float),
            deg=1,
        )
        slopes[int(seed)] = float(slope)
    return slopes


def _trend_value(result: Mapping[str, object], key: str) -> float:
    return float(dict(result.get("trend", {})).get(key, float("nan")))


def build_validation(
    legacy: Mapping[str, object],
    simple: Mapping[str, object],
    hidden: Mapping[str, object],
) -> dict[str, object]:
    legacy_config = dict(legacy["config"])
    simple_config = dict(simple["config"])
    frozen_checks = {
        key: legacy_config.get(key) == simple_config.get(key)
        for key in FROZEN_CONFIG_KEYS
    }

    metric_checks: dict[str, object] = {}
    for metric, expected_sign in PRIMARY_DIRECTIONS.items():
        legacy_slopes = _seed_slopes(legacy, metric)
        simple_slopes = _seed_slopes(simple, metric)
        metric_checks[metric] = {
            "expected_sign": int(expected_sign),
            "legacy_seed_slopes": legacy_slopes,
            "simple_seed_slopes": simple_slopes,
            "legacy_sign_consistency": int(
                sum(np.sign(value) == expected_sign for value in legacy_slopes.values())
            ),
            "simple_sign_consistency": int(
                sum(np.sign(value) == expected_sign for value in simple_slopes.values())
            ),
            "n_seeds": int(len(simple_slopes)),
            "direction_preserved": bool(
                all(np.sign(value) == expected_sign for value in legacy_slopes.values())
                and all(
                    np.sign(value) == expected_sign for value in simple_slopes.values()
                )
            ),
        }

    oracle_legacy = abs(_trend_value(legacy, "oracle_peid_synergy_slope"))
    oracle_simple = abs(_trend_value(simple, "oracle_peid_synergy_slope"))
    learned_metrics = {
        "observational_wms_slope": (
            _trend_value(legacy, "observational_wms_slope"),
            _trend_value(simple, "observational_wms_slope"),
        ),
        "mmi_pid_synergy_slope": (
            _trend_value(legacy, "mmi_pid_synergy_slope"),
            _trend_value(simple, "mmi_pid_synergy_slope"),
        ),
        "tm_peid_synergy_slope": (
            _trend_value(legacy, "tm_peid_synergy_slope"),
            _trend_value(simple, "tm_peid_synergy_slope"),
        ),
        "hidden_w_tm_peid_synergy_slope": (
            float("nan"),
            _trend_value(hidden, "tm_peid_synergy_slope"),
        ),
    }
    primary_conclusion_unchanged = bool(
        all(frozen_checks.values())
        and all(bool(item["direction_preserved"]) for item in metric_checks.values())
        and oracle_legacy < 1e-10
        and oracle_simple < 1e-10
    )
    return {
        "scientific_question": "What changes when only the decimal dynamics coefficients are replaced by 1/0.5 coefficients?",
        "treatment": {
            "legacy": {
                "w_memory": 0.78,
                "x_memory": 0.42,
                "y_memory": 0.38,
                "w_to_x": 0.82,
                "w_to_y": 0.76,
                "z_memory": 0.22,
                "w_to_z": 0.15,
                "sin_xy": 1.0,
            },
            "simple": dict(simple_config["dynamics_coefficients"]),
        },
        "frozen_config_checks": frozen_checks,
        "metric_checks": metric_checks,
        "oracle_fixed_structure": {
            "legacy_abs_slope": float(oracle_legacy),
            "simple_abs_slope": float(oracle_simple),
            "passed": bool(oracle_legacy < 1e-10 and oracle_simple < 1e-10),
        },
        "aggregate_slopes_legacy_vs_simple": {
            name: {"legacy": float(values[0]), "simple": float(values[1])}
            for name, values in learned_metrics.items()
        },
        "primary_conclusion_unchanged": primary_conclusion_unchanged,
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-summary", type=Path, default=DEFAULT_LEGACY_SUMMARY)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legacy = _read_legacy_beta_result(args.legacy_summary)

    simple = run_sine_beta_common_driver_sweep(show_progress=True)
    full_path = args.result_dir / DEFAULT_FULL_RESULT.name
    _write_json(full_path, simple)

    liang = run_liang_experiment(show_progress=True)
    liang_path = args.result_dir / DEFAULT_LIANG_RESULT.name
    _write_json(liang_path, liang)

    hidden = run_hidden_w_sine_beta_mlp_peid_sweep(show_progress=True)
    hidden_path = args.result_dir / DEFAULT_HIDDEN_RESULT.name
    _write_json(hidden_path, hidden)

    wxyz_figure = _plot_sine_beta_combined_readout_sweep(
        simple,
        args.figure_dir,
        liang_result=liang,
        stem="sine_beta_simple_coefficients_wxyz_mlp",
    )
    xyz_figure = plot_combined_with_hidden_w_mlp_readouts(
        hidden,
        simple,
        args.figure_dir,
        liang_result=liang,
        stem="sine_beta_simple_coefficients_xyz_mlp_fixed_support",
    )
    hidden_detail_figure = plot_hidden_w_sweep(
        hidden,
        args.figure_dir / "sine_beta_simple_coefficients_hidden_w_detail.png",
    )
    hidden_comparison = compare_hidden_and_full_state(hidden, simple)
    hidden_comparison_figure = plot_hidden_vs_observed_w_syn(
        hidden,
        simple,
        args.figure_dir / "sine_beta_simple_coefficients_hidden_vs_observed_w.png",
    )

    validation = build_validation(legacy, simple, hidden)
    validation["outputs"] = {
        "full_result": str(full_path),
        "hidden_result": str(hidden_path),
        "liang_result": str(liang_path),
        "wxyz_figure": str(wxyz_figure),
        "xyz_figure": str(xyz_figure),
        "hidden_detail_figure": str(hidden_detail_figure),
        "hidden_comparison_figure": str(hidden_comparison_figure),
    }
    validation["hidden_vs_observed_w"] = hidden_comparison
    validation_path = args.result_dir / DEFAULT_VALIDATION_RESULT.name
    _write_json(validation_path, validation)

    print(
        json.dumps(
            {
                "validation_path": str(validation_path),
                "primary_conclusion_unchanged": validation[
                    "primary_conclusion_unchanged"
                ],
                "wxyz_figure": str(wxyz_figure),
                "xyz_figure": str(xyz_figure),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
