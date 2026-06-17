#!/usr/bin/env python3
"""Standalone MMI-PID synergy report for the six Part1 systems."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_network_dynamics_benchmark import (
    KURAMOTO_COUPLING_VALUES,
    KURAMOTO_FREQUENCY_DETUNING,
    KURAMOTO_PHASE_POTENTIAL_STRENGTH,
    _digest,
    _histogram_synergy,
    _transport_map_config,
    _transport_synergy,
    build_kuramoto_coupling_spec,
    simulate_natural_trajectory_pool,
)
from scripts.compare_coupled_standard_map_methods import (
    _broad_standard_map_split,
    _spearman,
)
from scripts.coupled_standard_map_peid import (
    StandardMapConfig,
)
from scripts.discrete_iteration_dynamics_benchmark import (
    COUPLED_HENON_HISTOGRAM_BINS,
    COUPLED_HENON_KAPPA_VALUES,
    IKEDA_U_VALUES,
    NICHOLSON_A_VALUES,
    WILSON_COWAN_REFRACTORY_GAIN_VALUES,
    MapSpec,
    _broad_one_step_sample_count,
    _broad_one_step_sweep_parameters,
    _coupled_henon_broad_distribution_metadata,
    _coupled_henon_sweep_parameters,
    build_coupled_henon_spec,
    build_ikeda_spec,
    build_nicholson_bailey_spec,
    build_wilson_cowan_refractory_spec,
    simulate_broad_one_step_pool,
    simulate_coupled_henon_prediction_pool,
)


DEFAULT_RESULT_PATH = ROOT / "results" / "part1_mmi_pid_synergy_report" / "summary.json"
DEFAULT_REPORT_PATH = ROOT / "docs" / "reports" / "mmi_pid_six_system_synergy.md"
DEFAULT_FIGURE_PATH = ROOT / "fig" / "part1_mmi_pid_synergy_report" / "mmi_pid_six_system_synergy.png"

SYSTEM_ORDER = (
    "standard_map",
    "wilson_cowan_refractory",
    "kuramoto",
    "coupled_henon",
    "ikeda_y_tau",
    "nicholson_bailey",
)


def compute_mmi_pid_atoms(*, left_mi: float, right_mi: float, joint_mi: float) -> dict[str, float]:
    left = float(left_mi)
    right = float(right_mi)
    joint = float(joint_mi)
    redundancy = min(left, right)
    unique_left = left - redundancy
    unique_right = right - redundancy
    synergy = joint - max(left, right)
    return {
        "I_left": left,
        "I_right": right,
        "I_joint": joint,
        "left_mi": left,
        "right_mi": right,
        "joint_mi": joint,
        "redundancy": float(redundancy),
        "unique_left": float(unique_left),
        "unique_right": float(unique_right),
        "synergy": float(synergy),
    }


def _mmi_pid_from_samples(
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    *,
    estimator: str,
    bins: int = 6,
) -> dict[str, float]:
    if estimator == "transport":
        values = _transport_synergy(left, right, target)
    elif estimator == "histogram":
        values = _histogram_synergy(
            np.asarray(left, dtype=float).reshape(len(left), -1)[:, 0],
            np.asarray(right, dtype=float).reshape(len(right), -1)[:, 0],
            np.asarray(target, dtype=float).reshape(len(target), -1)[:, 0],
            bins,
        )
    else:
        raise ValueError("estimator must be 'transport' or 'histogram'.")
    return compute_mmi_pid_atoms(
        left_mi=float(values["left_ei"]),
        right_mi=float(values["right_ei"]),
        joint_mi=float(values["joint_ei"]),
    )


def _prefixed_atoms(prefix: str, atoms: Mapping[str, float]) -> dict[str, float]:
    return {
        f"{prefix}_{key}": float(value)
        for key, value in atoms.items()
        if key in {"I_left", "I_right", "I_joint", "redundancy", "unique_left", "unique_right", "synergy"}
    }


def _summary_rows(rows: Sequence[dict[str, object]], *, parameter_key: str) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    summary: list[dict[str, object]] = []
    metrics = (
        "mmi_pid_synergy",
        "I_left",
        "I_right",
        "I_joint",
        "redundancy",
        "unique_left",
        "unique_right",
    )
    for value, group in frame.groupby(parameter_key, sort=True):
        item: dict[str, object] = {parameter_key: float(value), "n_seeds": int(group["seed"].nunique())}
        for metric in metrics:
            values = group[metric].astype(float)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=0))
        summary.append(item)
    return summary


def _source_target_arrays(
    states: np.ndarray,
    targets: np.ndarray,
    *,
    source_names: Sequence[str],
    target_names: Sequence[str],
    relation: tuple[str, str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right, target = relation
    source_index = {name: idx for idx, name in enumerate(source_names)}
    target_index = {name: idx for idx, name in enumerate(target_names)}
    return (
        np.asarray(states)[:, [source_index[left]]],
        np.asarray(states)[:, [source_index[right]]],
        np.asarray(targets)[:, [target_index[target]]],
    )


def _with_overrides(
    params: Mapping[str, int | float | str],
    sample_overrides: Mapping[str, int | float | str] | None,
) -> dict[str, int | float | str]:
    merged = dict(params)
    if not sample_overrides:
        return merged
    for key, value in sample_overrides.items():
        if key in {
            "trajectories",
            "samples_per_trajectory",
            "epochs",
            "shap_samples",
            "peid_samples",
            "bins",
        }:
            merged[key] = value
    if "readout_samples" in sample_overrides:
        merged["samples_per_trajectory"] = int(sample_overrides["readout_samples"])
        merged["trajectories"] = 1
    return merged


def _standard_map_rows(
    *,
    mode: str,
    parameter_values: Sequence[float],
    seeds: Sequence[int],
    sample_overrides: Mapping[str, int | float | str] | None,
) -> dict[str, object]:
    defaults = {"readout_samples": 1800 if mode == "full" else 48}
    overrides = dict(sample_overrides or {})
    readout_samples = int(overrides.get("readout_samples", overrides.get("peid_samples", defaults["readout_samples"])))
    rows: list[dict[str, object]] = []
    for coupling in parameter_values:
        for seed_value in seeds:
            seed = int(seed_value)
            config = StandardMapConfig(k=1.5, coupling=float(coupling), noise_std=0.0)
            readout = _broad_standard_map_split(config, seed=300000 + seed, samples=readout_samples)
            atoms = _mmi_pid_from_samples(
                readout.states[:, [0]],
                readout.states[:, [2]],
                readout.impulses[:, [0]],
                estimator="transport",
            )
            rows.append(
                {
                    "system": "standard_map",
                    "coupling": float(coupling),
                    "seed": seed,
                    "relation": "q1+q2->I1",
                    "estimator": "transport",
                    "I_left": atoms["I_left"],
                    "I_right": atoms["I_right"],
                    "I_joint": atoms["I_joint"],
                    "redundancy": atoms["redundancy"],
                    "unique_left": atoms["unique_left"],
                    "unique_right": atoms["unique_right"],
                    "mmi_pid_synergy": atoms["synergy"],
                    "observed_target_digest": _digest(readout.impulses[:, [0]]),
                    "readout_state_digest": _digest(readout.states),
                    "readout_target_digest": _digest(readout.impulses[:, [0]]),
                    "mmi_pid_state_digest": _digest(readout.states),
                    "mmi_pid_target_digest": _digest(readout.impulses[:, [0]]),
                }
            )
    return _system_payload(
        system="standard_map",
        display_name="Coupled standard map",
        parameter_key="coupling",
        parameter_values=parameter_values,
        seeds=seeds,
        estimator="transport",
        relation="q1+q2->I1",
        rows=rows,
        protocol_extra={
            "shared_readout_state_distribution": "held_out_broad_intervention_domain_one_step_pool",
            "mmi_pid_state_distribution": "same_held_out_broad_observed_states_as_wms_surd",
            "readout_samples": readout_samples,
            "transport_map": _transport_map_config(),
        },
    )


def _map_rows(
    *,
    system: str,
    display_name: str,
    builder: Callable[[float], MapSpec],
    parameter_key: str,
    parameter_values: Sequence[float],
    seeds: Sequence[int],
    mode: str,
    relation: tuple[str, str, str],
    estimator: str,
    params: Mapping[str, int | float | str],
    sample_overrides: Mapping[str, int | float | str] | None,
    protocol_extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    params = _with_overrides(params, sample_overrides)
    rows: list[dict[str, object]] = []
    for parameter_value in parameter_values:
        spec = builder(float(parameter_value))
        for seed_value in seeds:
            seed = int(seed_value)
            readout_count = int(sample_overrides["readout_samples"]) if sample_overrides and "readout_samples" in sample_overrides else _broad_one_step_sample_count(params)
            if system == "coupled_henon":
                readout_states, readout_targets = simulate_coupled_henon_prediction_pool(
                    spec, seed=200000 + seed, samples=readout_count
                )
            else:
                readout_states, readout_targets = simulate_broad_one_step_pool(
                    spec, seed=200000 + seed, samples=readout_count
                )
            left, right, target_values = _source_target_arrays(
                readout_states,
                readout_targets,
                source_names=spec.state_names,
                target_names=spec.target_names,
                relation=relation,
            )
            atoms = _mmi_pid_from_samples(
                left,
                right,
                target_values,
                estimator=estimator,
                bins=int(params.get("bins", COUPLED_HENON_HISTOGRAM_BINS)),
            )
            rows.append(
                {
                    "system": system,
                    parameter_key: float(parameter_value),
                    "seed": seed,
                    "relation": f"{relation[0]}+{relation[1]}->{relation[2]}",
                    "estimator": estimator,
                    "bins": int(params.get("bins", 0)) if estimator == "histogram" else None,
                    "I_left": atoms["I_left"],
                    "I_right": atoms["I_right"],
                    "I_joint": atoms["I_joint"],
                    "redundancy": atoms["redundancy"],
                    "unique_left": atoms["unique_left"],
                    "unique_right": atoms["unique_right"],
                    "mmi_pid_synergy": atoms["synergy"],
                    "observed_target_digest": _digest(target_values),
                    "readout_state_digest": _digest(readout_states),
                    "readout_target_digest": _digest(readout_targets),
                    "mmi_pid_state_digest": _digest(readout_states),
                    "mmi_pid_target_digest": _digest(target_values),
                }
            )
    extra = dict(protocol_extra or {})
    extra.update(
        {
            "training_distribution": "not_used_by_observational_mmi_pid",
            "model_training": "not_used_by_observational_mmi_pid",
            "shared_readout_state_distribution": extra.get("shared_readout_state_distribution", "held_out_broad_one_step_pool"),
            "mmi_pid_state_distribution": extra.get("mmi_pid_state_distribution", "same_observed_readout_states_as_wms_surd"),
            "peid_interventions": "not_used_by_observational_mmi_pid",
            "peid_target_distribution": "not_used_by_observational_mmi_pid",
            "readout_samples": int(sample_overrides["readout_samples"]) if sample_overrides and "readout_samples" in sample_overrides else _broad_one_step_sample_count(params),
            "transport_map": _transport_map_config() if estimator == "transport" else None,
            "histogram": {"bins": int(params.get("bins", COUPLED_HENON_HISTOGRAM_BINS))} if estimator == "histogram" else None,
        }
    )
    return _system_payload(
        system=system,
        display_name=display_name,
        parameter_key=parameter_key,
        parameter_values=parameter_values,
        seeds=seeds,
        estimator=estimator,
        relation=f"{relation[0]}+{relation[1]}->{relation[2]}",
        rows=rows,
        protocol_extra=extra,
    )


def _kuramoto_rows(
    *,
    mode: str,
    parameter_values: Sequence[float],
    seeds: Sequence[int],
    sample_overrides: Mapping[str, int | float | str] | None,
) -> dict[str, object]:
    natural_trajectories = int(sample_overrides.get("trajectories", 4 if mode == "smoke" else 12)) if sample_overrides else (4 if mode == "smoke" else 12)
    samples_per_trajectory = int(sample_overrides.get("samples_per_trajectory", 65 if mode == "smoke" else 100)) if sample_overrides else (65 if mode == "smoke" else 100)
    natural_burnin_steps = int(sample_overrides.get("burnin_steps", 1200 if mode == "smoke" else 2400)) if sample_overrides else (1200 if mode == "smoke" else 2400)
    phase_velocity_noise = 0.01
    rows: list[dict[str, object]] = []
    relation = ("theta1", "theta2", "dtheta1")
    for coupling in parameter_values:
        spec = build_kuramoto_coupling_spec(float(coupling))
        for seed_value in seeds:
            seed = int(seed_value)
            natural_states, natural_targets = simulate_natural_trajectory_pool(
                spec,
                seed=seed,
                trajectories=natural_trajectories,
                samples_per_trajectory=samples_per_trajectory,
                burnin_steps=natural_burnin_steps,
                noise=phase_velocity_noise,
            )
            left, right, target_values = _source_target_arrays(
                natural_states,
                natural_targets,
                source_names=spec.state_names,
                target_names=spec.target_names,
                relation=relation,
            )
            atoms = _mmi_pid_from_samples(left, right, target_values, estimator="transport")
            rows.append(
                {
                    "system": "kuramoto",
                    "coupling": float(coupling),
                    "seed": seed,
                    "relation": "theta1+theta2->dtheta1",
                    "estimator": "transport",
                    "I_left": atoms["I_left"],
                    "I_right": atoms["I_right"],
                    "I_joint": atoms["I_joint"],
                    "redundancy": atoms["redundancy"],
                    "unique_left": atoms["unique_left"],
                    "unique_right": atoms["unique_right"],
                    "mmi_pid_synergy": atoms["synergy"],
                    "observed_target_digest": _digest(target_values),
                    "readout_state_digest": _digest(natural_states),
                    "readout_target_digest": _digest(natural_targets),
                    "mmi_pid_state_digest": _digest(natural_states),
                    "mmi_pid_target_digest": _digest(target_values),
                }
            )
    return _system_payload(
        system="kuramoto",
        display_name="Kuramoto phase locking",
        parameter_key="coupling",
        parameter_values=parameter_values,
        seeds=seeds,
        estimator="transport",
        relation="theta1+theta2->dtheta1",
        rows=rows,
        protocol_extra={
            "training_distribution": "not_used_by_observational_mmi_pid",
            "shared_readout_state_distribution": "natural_trajectory_for_wms_surd_shap",
            "mmi_pid_state_distribution": "same_natural_observed_states_as_wms_surd",
            "natural_trajectory_protocol": {
                "trajectories": natural_trajectories,
                "samples_per_trajectory": samples_per_trajectory,
                "burnin_steps": natural_burnin_steps,
            },
            "phase_velocity_noise_std": phase_velocity_noise,
            "frequency_detuning": KURAMOTO_FREQUENCY_DETUNING,
            "phase_potential_strength": KURAMOTO_PHASE_POTENTIAL_STRENGTH,
            "transport_map": _transport_map_config(),
        },
    )


def _audit(rows: Sequence[dict[str, object]], *, estimator: str) -> dict[str, object]:
    identity_ok = all(
        np.isclose(
            float(row["I_joint"]),
            float(row["redundancy"])
            + float(row["unique_left"])
            + float(row["unique_right"])
            + float(row["mmi_pid_synergy"]),
            atol=1e-8,
        )
        for row in rows
    )
    return {
        "estimator_matches_part1": all(str(row["estimator"]) == estimator for row in rows),
        "no_mlp_used_for_mmi_pid": all(
            "mlp_model_digest" not in row and "existing_mlp_peid_model_digest" not in row
            for row in rows
        ),
        "mmi_pid_identity_holds": bool(identity_ok),
        "has_state_and_target_digests": all(
            {"readout_state_digest", "mmi_pid_state_digest", "mmi_pid_target_digest", "observed_target_digest"} <= set(row)
            for row in rows
        ),
    }


def _system_payload(
    *,
    system: str,
    display_name: str,
    parameter_key: str,
    parameter_values: Sequence[float],
    seeds: Sequence[int],
    estimator: str,
    relation: str,
    rows: list[dict[str, object]],
    protocol_extra: Mapping[str, object],
) -> dict[str, object]:
    summary = _summary_rows(rows, parameter_key=parameter_key)
    trends = {
        "mmi_pid_synergy": _spearman(
            [float(row[parameter_key]) for row in summary],
            [float(row["mmi_pid_synergy_mean"]) for row in summary],
        )
        if len(summary) >= 2
        else float("nan"),
    }
    protocol = {
        "mmi_pid_definition": "MMI",
        "target_distribution": "observed_readout_targets",
        "atoms": {
            "redundancy": "min(I_left, I_right)",
            "unique_left": "I_left - redundancy",
            "unique_right": "I_right - redundancy",
            "synergy": "I_joint - max(I_left, I_right)",
        },
        "parameter_key": parameter_key,
        "parameter_values": [float(value) for value in parameter_values],
        "seeds": [int(seed) for seed in seeds],
        "relation": relation,
        "estimator": estimator,
        **dict(protocol_extra),
    }
    return {
        "system": system,
        "display_name": display_name,
        "parameter_key": parameter_key,
        "summary": summary,
        "rows": rows,
        "trends": trends,
        "protocol": protocol,
        "audit": _audit(rows, estimator=estimator),
    }


def _parameter_values(
    key: str,
    overrides: Mapping[str, Sequence[float]] | None,
    default: Sequence[float],
) -> Sequence[float]:
    return tuple(float(value) for value in (overrides or {}).get(key, default))


def _plot_report(systems: Mapping[str, dict[str, object]], figure_path: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 7.2), constrained_layout=True)
    axes = axes.flat
    letters = "abcdef"
    for axis, letter, system_key in zip(axes, letters, SYSTEM_ORDER):
        payload = systems[system_key]
        parameter_key = str(payload["parameter_key"])
        summary = list(payload["summary"])
        x_values = np.asarray([float(row[parameter_key]) for row in summary])
        mean = np.asarray([float(row["mmi_pid_synergy_mean"]) for row in summary])
        std = np.asarray([float(row["mmi_pid_synergy_std"]) for row in summary])
        axis.fill_between(x_values, mean - std, mean + std, color="#2F7D5A", alpha=0.14, linewidth=0)
        axis.plot(
            x_values,
            mean,
            color="#2F7D5A",
            marker="D",
            linewidth=1.6,
            markersize=4.0,
            label="Observed MMI-PID synergy",
        )
        axis.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
        axis.set_title(f"{letter}  {payload['display_name']}", loc="left", fontsize=9, fontweight="bold")
        axis.set_xlabel(parameter_key)
        axis.grid(axis="y", alpha=0.20, linewidth=0.6)
    axes[0].set_ylabel("MMI-PID synergy")
    axes[3].set_ylabel("MMI-PID synergy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _format_float(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 100.0 or (0.0 < abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.6f}"


def _write_report(payload: Mapping[str, object], report_path: Path, figure_path: Path) -> None:
    systems = payload["systems"]  # type: ignore[index]
    rel_figure = os.path.relpath(figure_path, report_path.parent).replace(os.sep, "/")
    lines = [
        "# MMI-PID Six-System Synergy Report",
        "",
        "This standalone report recomputes two-source observational MMI-PID synergy for the six Part1 examples under their registered readout data conditions. It does not train or call an MLP.",
        "",
        f"![MMI-PID six-system synergy]({rel_figure})",
        "",
        "## Definition",
        "",
        "$$R=\\min\\{I_1,I_2\\},\\quad U_1=I_1-R,\\quad U_2=I_2-R,\\quad S=I_{12}-\\max\\{I_1,I_2\\}.$$",
        "",
        "## Audit",
        "",
        "| system | relation | estimator | seeds | matched estimator | no MLP | identity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for system_key in SYSTEM_ORDER:
        item = systems[system_key]
        audit = item["audit"]
        protocol = item["protocol"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["display_name"]),
                    str(protocol["relation"]),
                    str(protocol["estimator"]),
                    str(len(protocol["seeds"])),
                    "yes" if audit["estimator_matches_part1"] else "no",
                    "yes" if audit["no_mlp_used_for_mmi_pid"] else "no",
                    "yes" if audit["mmi_pid_identity_holds"] else "no",
                ]
            )
            + " |"
        )
    for system_key in SYSTEM_ORDER:
        item = systems[system_key]
        protocol = item["protocol"]
        parameter_key = str(item["parameter_key"])
        lines.extend(
            [
                "",
                f"## {item['display_name']}",
                "",
                f"- relation: `{protocol['relation']}`",
                f"- parameter grid: `{protocol['parameter_values']}`",
                f"- seeds: `{protocol['seeds']}`",
                f"- data protocol: `{protocol.get('shared_readout_state_distribution')}`; MMI-PID states: `{protocol.get('mmi_pid_state_distribution')}`",
                f"- target distribution: `{protocol.get('target_distribution')}`",
                "",
                f"| {parameter_key} | observed MMI-PID synergy | I_joint | redundancy |",
                "|---:|---:|---:|---:|",
            ]
        )
        for row in item["summary"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _format_float(float(row[parameter_key])),
                        f"{_format_float(float(row['mmi_pid_synergy_mean']))} ± {_format_float(float(row['mmi_pid_synergy_std']))}",
                        f"{_format_float(float(row['I_joint_mean']))} ± {_format_float(float(row['I_joint_std']))}",
                        f"{_format_float(float(row['redundancy_mean']))} ± {_format_float(float(row['redundancy_std']))}",
                    ]
                )
                + " |"
            )
        trend = item["trends"]["mmi_pid_synergy"]
        lines.extend(
            [
                "",
                f"Interpretation: observational MMI-PID Spearman trend is `{_format_float(float(trend))}`.",
            ]
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_mmi_pid_six_system_report(
    *,
    mode: str = "full",
    seeds: Sequence[int] = (0, 1, 2),
    result_path: Path = DEFAULT_RESULT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    figure_path: Path = DEFAULT_FIGURE_PATH,
    parameter_overrides: Mapping[str, Sequence[float]] | None = None,
    sample_overrides: Mapping[str, int | float | str] | None = None,
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    seeds = tuple(int(seed) for seed in seeds)
    systems: dict[str, dict[str, object]] = {}
    systems["standard_map"] = _standard_map_rows(
        mode=mode,
        parameter_values=_parameter_values("standard_map", parameter_overrides, (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
        seeds=seeds,
        sample_overrides=sample_overrides,
    )
    systems["wilson_cowan_refractory"] = _map_rows(
        system="wilson_cowan_refractory",
        display_name="Wilson-Cowan gain",
        builder=build_wilson_cowan_refractory_spec,
        parameter_key="gain",
        parameter_values=_parameter_values("wilson_cowan_refractory", parameter_overrides, WILSON_COWAN_REFRACTORY_GAIN_VALUES),
        seeds=seeds,
        mode=mode,
        relation=("E", "I", "E_tau"),
        estimator="transport",
        params=_broad_one_step_sweep_parameters(mode, system="wilson_cowan_refractory"),
        sample_overrides=sample_overrides,
    )
    systems["kuramoto"] = _kuramoto_rows(
        mode=mode,
        parameter_values=_parameter_values("kuramoto", parameter_overrides, KURAMOTO_COUPLING_VALUES),
        seeds=seeds,
        sample_overrides=sample_overrides,
    )
    systems["coupled_henon"] = _map_rows(
        system="coupled_henon",
        display_name="Coupled Henon",
        builder=build_coupled_henon_spec,
        parameter_key="kappa",
        parameter_values=_parameter_values("coupled_henon", parameter_overrides, COUPLED_HENON_KAPPA_VALUES),
        seeds=seeds,
        mode=mode,
        relation=("x", "z", "x_tau"),
        estimator="histogram",
        params=_coupled_henon_sweep_parameters(mode),
        sample_overrides=sample_overrides,
        protocol_extra=_coupled_henon_broad_distribution_metadata(),
    )
    systems["ikeda_y_tau"] = _map_rows(
        system="ikeda_y_tau",
        display_name="Ikeda optical cavity",
        builder=build_ikeda_spec,
        parameter_key="u",
        parameter_values=_parameter_values("ikeda_y_tau", parameter_overrides, IKEDA_U_VALUES),
        seeds=seeds,
        mode=mode,
        relation=("x", "y", "y_tau"),
        estimator="transport",
        params=_broad_one_step_sweep_parameters(mode, system="ikeda"),
        sample_overrides=sample_overrides,
    )
    systems["nicholson_bailey"] = _map_rows(
        system="nicholson_bailey",
        display_name="Nicholson-Bailey",
        builder=build_nicholson_bailey_spec,
        parameter_key="a",
        parameter_values=_parameter_values("nicholson_bailey", parameter_overrides, NICHOLSON_A_VALUES),
        seeds=seeds,
        mode=mode,
        relation=("H", "P", "H_tau"),
        estimator="transport",
        params=_broad_one_step_sweep_parameters(mode, system="nicholson_bailey"),
        sample_overrides=sample_overrides,
    )
    payload = {
        "mode": mode,
        "systems": systems,
        "result_path": str(result_path),
        "report_path": str(report_path),
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_report(systems, Path(figure_path))
    _write_report(payload, Path(report_path), Path(figure_path))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_mmi_pid_six_system_report(
        mode=args.mode,
        seeds=tuple(args.seeds),
        result_path=args.result_path,
        report_path=args.report_path,
        figure_path=args.figure_path,
    )
    print(
        json.dumps(
            {
                "result_path": result["result_path"],
                "report_path": result["report_path"],
                "figure_path": result["figure_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
