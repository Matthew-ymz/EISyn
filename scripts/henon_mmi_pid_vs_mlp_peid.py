#!/usr/bin/env python3
"""Contrast observational MMI-PID with MLP+PEID on the classic Hénon map."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_network_dynamics_benchmark import (
    _digest,
    _discretize,
    _entropy_discrete,
    _histogram_synergy,
    _specific_information_surd,
    fit_mlp,
)


DEFAULT_RESULT_PATH = ROOT / "results" / "henon_mmi_pid_vs_mlp_peid" / "summary.json"
DEFAULT_REPORT_PATH = ROOT / "docs" / "reports" / "henon_mmi_pid_vs_mlp_peid.md"
DEFAULT_FIGURE_PATH = ROOT / "fig" / "henon_mmi_pid_vs_mlp_peid" / "henon_mmi_pid_vs_mlp_peid.png"
DEFAULT_SWEEP_RESULT_PATH = ROOT / "results" / "henon_unique_sweep_mmi_vs_mlp_peid" / "summary.json"
DEFAULT_SWEEP_REPORT_PATH = ROOT / "docs" / "reports" / "henon_unique_sweep_mmi_vs_mlp_peid.md"
DEFAULT_SWEEP_FIGURE_PATH = (
    ROOT / "fig" / "henon_unique_sweep_mmi_vs_mlp_peid" / "henon_unique_sweep_mmi_vs_mlp_peid.png"
)
DEFAULT_FIVE_METHOD_RESULT_PATH = ROOT / "results" / "henon_unique_five_method_synergy" / "summary.json"
DEFAULT_FIVE_METHOD_FIGURE_PATH = ROOT / "fig" / "henon_unique_five_method_synergy" / "henon_unique_five_method_synergy.png"


def decompose_mi_triplet(*, left_mi: float, right_mi: float, joint_mi: float) -> dict[str, float]:
    left = float(left_mi)
    right = float(right_mi)
    joint = float(joint_mi)
    weaker = min(left, right)
    peid_residual = joint - left - right
    mmi_pid_synergy = joint - max(left, right)
    return {
        "left_mi": left,
        "right_mi": right,
        "joint_mi": joint,
        "weaker_single_source_mi": weaker,
        "mmi_pid_synergy": float(mmi_pid_synergy),
        "peid_residual": float(peid_residual),
        "mmi_minus_peid": float(mmi_pid_synergy - peid_residual),
    }


def henon_next_x(states: np.ndarray, *, a: float = 1.4) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    x = values[:, 0]
    y = values[:, 1]
    return (1.0 - float(a) * x * x + y).reshape(-1, 1)


def henon_fixed_interaction_next_x(
    states: np.ndarray,
    *,
    gamma: float,
    kappa: float = 0.35,
    a: float = 1.4,
) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    x = values[:, 0]
    y = values[:, 1]
    return (1.0 - float(a) * x * x + float(gamma) * y + float(kappa) * x * y).reshape(-1, 1)


def henon_fixed_interaction_with_unique_observation(
    states: np.ndarray,
    *,
    gamma: float,
    noise: np.ndarray,
    kappa: float = 0.35,
    a: float = 1.4,
) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    x = values[:, 0]
    y = values[:, 1]
    interaction_readout = 1.0 - float(a) * x * x + float(kappa) * x * y
    unique_readout = float(gamma) * y + np.asarray(noise, dtype=float).reshape(-1)
    return np.column_stack([interaction_readout, unique_readout])


def sample_henon_states(*, samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    x = rng.uniform(-1.5, 1.5, size=int(samples))
    y = rng.uniform(-0.5, 0.5, size=int(samples))
    return np.column_stack([x, y])


def _histogram_triplet(left: np.ndarray, right: np.ndarray, target: np.ndarray, *, bins: int) -> dict[str, float]:
    values = _histogram_synergy(left, right, target, bins=int(bins))
    return decompose_mi_triplet(
        left_mi=float(values["left_ei"]),
        right_mi=float(values["right_ei"]),
        joint_mi=float(values["joint_ei"]),
    )


def _mi_discrete_any_target(sources: np.ndarray, target: np.ndarray) -> float:
    source = np.asarray(sources, dtype=int)
    if source.ndim == 1:
        source = source.reshape(-1, 1)
    target_codes = np.asarray(target, dtype=int)
    if target_codes.ndim == 1:
        target_codes = target_codes.reshape(-1, 1)
    return float(
        _entropy_discrete(source)
        + _entropy_discrete(target_codes)
        - _entropy_discrete(np.column_stack([source, target_codes]))
    )


def _histogram_triplet_any_target(
    left: np.ndarray, right: np.ndarray, target: np.ndarray, *, bins: int
) -> dict[str, float]:
    a = _discretize(left, bins)
    b = _discretize(right, bins)
    target_values = np.asarray(target, dtype=float)
    if target_values.ndim == 1 or target_values.shape[1] == 1:
        t = _discretize(target_values.reshape(-1), bins).reshape(-1, 1)
    else:
        t = np.column_stack([_discretize(target_values[:, column], bins) for column in range(target_values.shape[1])])
    return decompose_mi_triplet(
        left_mi=_mi_discrete_any_target(a, t),
        right_mi=_mi_discrete_any_target(b, t),
        joint_mi=_mi_discrete_any_target(np.column_stack([a, b]), t),
    )


def _joint_target_codes(target: np.ndarray, *, bins: int) -> np.ndarray:
    target_values = np.asarray(target, dtype=float)
    if target_values.ndim == 1 or target_values.shape[1] == 1:
        return _discretize(target_values.reshape(-1), bins)
    columns = [_discretize(target_values[:, column], bins) for column in range(target_values.shape[1])]
    _, codes = np.unique(np.column_stack(columns), axis=0, return_inverse=True)
    return np.asarray(codes, dtype=int)


def _histogram_wms_surd_mmi_any_target(
    left: np.ndarray, right: np.ndarray, target: np.ndarray, *, bins: int
) -> dict[str, float]:
    a = _discretize(left, bins)
    b = _discretize(right, bins)
    t = _joint_target_codes(target, bins=bins)
    atoms = decompose_mi_triplet(
        left_mi=_mi_discrete_any_target(a, t),
        right_mi=_mi_discrete_any_target(b, t),
        joint_mi=_mi_discrete_any_target(np.column_stack([a, b]), t),
    )
    surd = _specific_information_surd(a, b, t)
    return {
        "wms": atoms["peid_residual"],
        "surd_synergy": float(surd["synergy"]),
        "mmi_pid_synergy": atoms["mmi_pid_synergy"],
        "left_mi": atoms["left_mi"],
        "right_mi": atoms["right_mi"],
        "joint_mi": atoms["joint_mi"],
        "weaker_single_source_mi": atoms["weaker_single_source_mi"],
    }


def _two_source_vector_shap_interaction(model: object, states: np.ndarray, *, samples: int, seed: int) -> float:
    rng = np.random.default_rng(int(seed))
    values = np.asarray(states, dtype=float)
    count = min(int(samples), len(values))
    foreground = values[rng.choice(len(values), size=count, replace=False)]
    background = np.mean(values, axis=0)
    baseline_rows = np.repeat(background[None, :], count, axis=0)
    both = baseline_rows.copy()
    left = baseline_rows.copy()
    right = baseline_rows.copy()
    both[:, [0, 1]] = foreground[:, [0, 1]]
    left[:, 0] = foreground[:, 0]
    right[:, 1] = foreground[:, 1]
    baseline = model.predict(baseline_rows)
    interaction = model.predict(both) - model.predict(left) - model.predict(right) + baseline
    return float(np.mean(np.abs(interaction)))


def _mean_std(rows: Sequence[dict[str, object]], key: str) -> tuple[float, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(values.mean()), float(values.std(ddof=0))


def _dynamic_range(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(array.max() - array.min())


def _plot(rows: Sequence[dict[str, object]], summary: dict[str, float], figure_path: Path) -> None:
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
    specs = [
        ("observed_mmi_pid_synergy", "Observed\nMMI-PID", "#2F7D5A"),
        ("oracle_peid_residual", "True-map\nPEID residual", "#3D3D3D"),
        ("mlp_peid_residual", "MLP+PEID\nresidual", "#7068A8"),
        ("observed_weaker_single_source_mi", "Weaker\nsingle-source MI", "#D99A48"),
    ]
    means = [summary[f"{key}_mean"] for key, _, _ in specs]
    stds = [summary[f"{key}_std"] for key, _, _ in specs]
    colors = [color for _, _, color in specs]
    labels = [label for _, label, _ in specs]
    x = np.arange(len(specs))
    fig, ax = plt.subplots(figsize=(5.4, 3.4), constrained_layout=True)
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.82, error_kw={"linewidth": 0.9, "capsize": 3})
    for index, (key, _, _) in enumerate(specs):
        values = [float(row[key]) for row in rows]
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else [0.0]
        ax.scatter(np.asarray(jitter) + index, values, color="black", s=12, zorder=3, linewidths=0)
    ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Information readout (bits)")
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_report(payload: dict[str, object], report_path: Path, figure_path: Path) -> None:
    rel_figure = os.path.relpath(figure_path, report_path.parent).replace(os.sep, "/")
    summary = payload["summary"]
    lines = [
        "# Hénon MMI-PID vs MLP+PEID",
        "",
        "This report uses the classic Hénon one-step equation",
        "",
        "$$x_{t+1}=1-1.4x_t^2+y_t,$$",
        "",
        "with sources `x_t` and `y_t` and target `x_{t+1}`. The example is deliberately chosen because both sources carry strong single-source information about the target. Under MMI-PID, the synergy term is",
        "",
        "$$S_{MMI}=I(x,y;x_{t+1})-\\max\\{I(x;x_{t+1}), I(y;x_{t+1})\\},$$",
        "",
        "so it exceeds the PEID residual by the weaker single-source information term.",
        "",
        f"![Hénon comparison]({rel_figure})",
        "",
        "## Summary",
        "",
        "| readout | mean ± std |",
        "|---|---:|",
    ]
    for key, label in [
        ("observed_mmi_pid_synergy", "Observed MMI-PID synergy"),
        ("oracle_peid_residual", "True-map PEID residual"),
        ("mlp_peid_residual", "MLP+PEID residual"),
        ("observed_weaker_single_source_mi", "Weaker single-source MI"),
        ("observed_mmi_minus_oracle_peid", "MMI minus true-map PEID"),
    ]:
        lines.append(
            f"| {label} | {float(summary[f'{key}_mean']):.6f} ± {float(summary[f'{key}_std']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The observed MMI-PID synergy is not isolating a learned mechanism. It is high because the weaker source has its own information about the target, and MMI redundancy uses the minimum single-source information. The MLP+PEID residual remains tied to the fitted mechanism's joint surplus after subtracting both single-source terms, so the two quantities separate on this classic additive Hénon readout.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _summarize_by_parameter(rows: Sequence[dict[str, object]], metric_keys: Sequence[str]) -> list[dict[str, object]]:
    parameters = sorted({float(row["lambda"]) for row in rows})
    summaries: list[dict[str, object]] = []
    for parameter in parameters:
        selected = [row for row in rows if np.isclose(float(row["lambda"]), parameter)]
        item: dict[str, object] = {
            "lambda": parameter,
            "gamma": float(np.mean([float(row["gamma"]) for row in selected])),
            "kappa": float(np.mean([float(row["kappa"]) for row in selected])),
            "n": len(selected),
        }
        for key in metric_keys:
            mean, std = _mean_std(selected, key)
            item[f"{key}_mean"] = mean
            item[f"{key}_std"] = std
        summaries.append(item)
    return summaries


def _summarize_by_gamma(rows: Sequence[dict[str, object]], metric_keys: Sequence[str]) -> list[dict[str, object]]:
    gammas = sorted({float(row["gamma"]) for row in rows})
    summaries: list[dict[str, object]] = []
    for gamma in gammas:
        selected = [row for row in rows if np.isclose(float(row["gamma"]), gamma)]
        item: dict[str, object] = {"gamma": gamma, "n_seeds": int(len(selected))}
        for key in metric_keys:
            mean, std = _mean_std(selected, key)
            item[f"{key}_mean"] = mean
            item[f"{key}_std"] = std
        summaries.append(item)
    return summaries


def _plot_five_method_sweep(summary: Sequence[dict[str, object]], figure_path: Path) -> None:
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
    parameter_key = "lambda" if summary and "lambda" in summary[0] else "gamma"
    x = np.asarray([float(item[parameter_key]) for item in summary], dtype=float)
    specs = [
        ("wms", "WMS", "#9C6B5A", "o"),
        ("surd_synergy", "SURD synergy", "#E3A13D", "s"),
        ("shap_interaction", "MLP+SHAP interaction", "#7068A8", "^"),
        ("peid_synergy", "MLP+PEID synergy", "#2F7D5A", "D"),
        ("mmi_pid_synergy", "MMI-PID synergy", "#4C78A8", "P"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    for key, label, color, marker in specs:
        mean = np.asarray([float(item[f"{key}_mean"]) for item in summary], dtype=float)
        std = np.asarray([float(item[f"{key}_std"]) for item in summary], dtype=float)
        ax.plot(x, mean, color=color, marker=marker, markersize=4.2, linewidth=1.5, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    xlabel = "Control parameter lambda" if parameter_key == "lambda" else "Additive single-source coefficient gamma"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Synergy / Interaction")
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_unique_sweep(parameter_summary: Sequence[dict[str, object]], figure_path: Path) -> None:
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
    x = np.asarray([float(item["lambda"]) for item in parameter_summary], dtype=float)
    specs = [
        ("observed_mmi_pid_synergy", "Observed MMI-PID", "#2F7D5A", "o"),
        ("mlp_peid_residual", "MLP+PEID residual", "#7068A8", "^"),
        ("oracle_peid_residual", "True-map PEID residual", "#3D3D3D", "s"),
        ("observed_weaker_single_source_mi", "Weaker source MI", "#D99A48", "D"),
    ]
    fig, ax = plt.subplots(figsize=(5.9, 3.35), constrained_layout=True)
    for key, label, color, marker in specs:
        mean = np.asarray([float(item[f"{key}_mean"]) for item in parameter_summary], dtype=float)
        std = np.asarray([float(item[f"{key}_std"]) for item in parameter_summary], dtype=float)
        ax.plot(x, mean, color=color, marker=marker, markersize=4.2, linewidth=1.5, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Control parameter lambda")
    ax.set_ylabel("Information readout (bits)")
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_unique_sweep_report(payload: dict[str, object], report_path: Path, figure_path: Path) -> None:
    rel_figure = os.path.relpath(figure_path, report_path.parent).replace(os.sep, "/")
    diagnostics = payload["diagnostics"]
    summaries = payload["parameter_summary"]
    lines = [
        "# Hénon Unique-Information Sweep: MMI-PID vs MLP+PEID",
        "",
        "This controlled Hénon-style readout increases a separate single-source observation channel while decreasing the explicit interaction coefficient:",
        "",
        "$$\\mathbf{z}_{t+1}=\\left[1-1.4x_t^2+\\kappa(\\lambda)x_ty_t,\\;\\gamma(\\lambda)y_t+\\epsilon_t\\right],\\quad \\sigma_\\epsilon=%.3f.$$"
        % float(payload["unique_noise_sigma"]),
        "",
        "`lambda` maps linearly to an increasing `gamma(lambda)` and a decreasing `kappa(lambda)`. Here `gamma` runs from %.3f to %.3f and `kappa` runs from %.3f to %.3f. Thus the single-source channel strengthens while the explicit interaction term weakens. The same noise draw is reused across the sweep for each seed."
        % (
            float(payload["gamma_range"][0]),
            float(payload["gamma_range"][1]),
            float(payload["kappa_range"][0]),
            float(payload["kappa_range"][1]),
        ),
        "",
        f"![Hénon unique sweep]({rel_figure})",
        "",
        "## Dynamic Range",
        "",
        "| quantity | max-min |",
        "|---|---:|",
        f"| Observed MMI-PID synergy | {float(diagnostics['observed_mmi_pid_synergy_dynamic_range']):.6f} |",
        f"| MLP+PEID residual | {float(diagnostics['mlp_peid_residual_dynamic_range']):.6f} |",
        f"| True-map PEID residual | {float(diagnostics['oracle_peid_residual_dynamic_range']):.6f} |",
        f"| Weaker source MI | {float(diagnostics['observed_weaker_single_source_mi_dynamic_range']):.6f} |",
        "",
        "## Parameter Sweep",
        "",
        "| lambda | gamma | kappa | MMI-PID synergy | MLP+PEID residual | true-map PEID residual | weaker source MI |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| %.3f | %.3f | %.3f | %.6f ± %.6f | %.6f ± %.6f | %.6f ± %.6f | %.6f ± %.6f |"
            % (
                float(item["lambda"]),
                float(item["gamma"]),
                float(item["kappa"]),
                float(item["observed_mmi_pid_synergy_mean"]),
                float(item["observed_mmi_pid_synergy_std"]),
                float(item["mlp_peid_residual_mean"]),
                float(item["mlp_peid_residual_std"]),
                float(item["oracle_peid_residual_mean"]),
                float(item["oracle_peid_residual_std"]),
                float(item["observed_weaker_single_source_mi_mean"]),
                float(item["observed_weaker_single_source_mi_std"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "For every row, `MMI-PID synergy - true-map PEID residual` equals the weaker single-source MI by construction of two-source MMI-PID. The curve therefore exposes the qualitative mismatch: PEID can fall as the explicit interaction weakens, while MMI-PID can still rise when the weaker single-source information grows faster.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_henon_unique_sweep_mmi_vs_mlp_peid(
    *,
    mode: str = "full",
    lambdas: Sequence[float] = (0.0, 0.1666667, 0.3333333, 0.5, 0.6666667, 0.8333333, 1.0),
    gamma_range: tuple[float, float] = (0.3, 2.0),
    kappa_range: tuple[float, float] = (0.5, 0.1),
    unique_noise_sigma: float = 0.5,
    seeds: Sequence[int] = (0, 1, 2),
    result_path: Path = DEFAULT_SWEEP_RESULT_PATH,
    report_path: Path = DEFAULT_SWEEP_REPORT_PATH,
    figure_path: Path = DEFAULT_SWEEP_FIGURE_PATH,
    samples: int | None = None,
    epochs: int | None = None,
    bins: int = 10,
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    readout_samples = int(samples if samples is not None else (700 if mode == "smoke" else 5000))
    train_samples = max(readout_samples * 2, 1000)
    mlp_epochs = int(epochs if epochs is not None else (45 if mode == "smoke" else 260))
    rows: list[dict[str, object]] = []
    gamma_start, gamma_stop = float(gamma_range[0]), float(gamma_range[1])
    kappa_start, kappa_stop = float(kappa_range[0]), float(kappa_range[1])
    for lambda_value in lambdas:
        parameter = float(lambda_value)
        gamma = gamma_start + parameter * (gamma_stop - gamma_start)
        kappa = kappa_start + parameter * (kappa_stop - kappa_start)
        for seed_value in seeds:
            seed = int(seed_value)
            train_states = sample_henon_states(samples=train_samples, seed=110000 + seed)
            readout_states = sample_henon_states(samples=readout_samples, seed=210000 + seed)
            train_noise = np.random.default_rng(410000 + seed).normal(0.0, float(unique_noise_sigma), size=train_samples)
            readout_noise = np.random.default_rng(510000 + seed).normal(
                0.0, float(unique_noise_sigma), size=readout_samples
            )
            train_targets = henon_fixed_interaction_with_unique_observation(
                train_states,
                gamma=gamma,
                noise=train_noise,
                kappa=kappa,
            )
            readout_targets = henon_fixed_interaction_with_unique_observation(
                readout_states,
                gamma=gamma,
                noise=readout_noise,
                kappa=kappa,
            )
            fitted = fit_mlp(train_states, train_targets, seed=310000 + seed, epochs=mlp_epochs)
            predicted_targets = fitted.predict(readout_states)
            left = readout_states[:, 0]
            right = readout_states[:, 1]
            observed = _histogram_triplet_any_target(left, right, readout_targets, bins=bins)
            mlp = _histogram_triplet_any_target(left, right, predicted_targets, bins=bins)
            rows.append(
                {
                    "lambda": parameter,
                    "gamma": gamma,
                    "kappa": float(kappa),
                    "seed": seed,
                    "observed_mmi_pid_synergy": observed["mmi_pid_synergy"],
                    "oracle_peid_residual": observed["peid_residual"],
                    "mlp_peid_residual": mlp["peid_residual"],
                    "observed_weaker_single_source_mi": observed["weaker_single_source_mi"],
                    "observed_mmi_minus_oracle_peid": observed["mmi_minus_peid"],
                    "observed_left_mi": observed["left_mi"],
                    "observed_right_mi": observed["right_mi"],
                    "observed_joint_mi": observed["joint_mi"],
                    "mlp_left_mi": mlp["left_mi"],
                    "mlp_right_mi": mlp["right_mi"],
                    "mlp_joint_mi": mlp["joint_mi"],
                    "mlp_train_mse": float(fitted.train_mse),
                    "mlp_baseline_mse": float(fitted.baseline_mse),
                    "train_state_digest": _digest(train_states),
                    "readout_state_digest": _digest(readout_states),
                    "observed_target_digest": _digest(readout_targets),
                    "mlp_target_digest": _digest(predicted_targets),
                }
            )
    metric_keys = [
        "observed_mmi_pid_synergy",
        "oracle_peid_residual",
        "mlp_peid_residual",
        "observed_weaker_single_source_mi",
        "observed_mmi_minus_oracle_peid",
        "observed_left_mi",
        "observed_right_mi",
        "observed_joint_mi",
    ]
    parameter_summary = _summarize_by_parameter(rows, metric_keys)
    diagnostics = {
        f"{key}_dynamic_range": _dynamic_range([float(item[f"{key}_mean"]) for item in parameter_summary])
        for key in metric_keys
    }
    payload = {
        "mode": mode,
        "system": "controlled_henon_unique_information_sweep",
        "equation": "z_tau = [1 - 1.4*x^2 + kappa(lambda)*x*y, gamma(lambda)*y + epsilon]",
        "relation": "x+y->z_tau",
        "parameter_key": "lambda",
        "lambdas": [float(parameter) for parameter in lambdas],
        "gamma_range": [gamma_start, gamma_stop],
        "kappa_range": [kappa_start, kappa_stop],
        "unique_noise_sigma": float(unique_noise_sigma),
        "estimator": "histogram",
        "bins": int(bins),
        "seeds": [int(seed) for seed in seeds],
        "samples": {"train": int(train_samples), "readout": int(readout_samples)},
        "mlp_epochs": int(mlp_epochs),
        "claim": "Scanning lambda increases additive single-source information while decreasing the explicit interaction coefficient.",
        "diagnostics": diagnostics,
        "parameter_summary": parameter_summary,
        "rows": rows,
        "result_path": str(result_path),
        "report_path": str(report_path),
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_unique_sweep(parameter_summary, Path(figure_path))
    _write_unique_sweep_report(payload, Path(report_path), Path(figure_path))
    return payload


def run_henon_unique_five_method_sweep(
    *,
    mode: str = "full",
    lambdas: Sequence[float] = (0.0, 0.1666667, 0.3333333, 0.5, 0.6666667, 0.8333333, 1.0),
    gamma_range: tuple[float, float] = (0.3, 2.0),
    kappa_range: tuple[float, float] = (0.5, 0.1),
    gammas: Sequence[float] | None = None,
    kappa: float | None = None,
    unique_noise_sigma: float = 0.5,
    seeds: Sequence[int] = (0, 1, 2),
    result_path: Path = DEFAULT_FIVE_METHOD_RESULT_PATH,
    figure_path: Path = DEFAULT_FIVE_METHOD_FIGURE_PATH,
    samples: int | None = None,
    epochs: int | None = None,
    bins: int = 10,
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    readout_samples = int(samples if samples is not None else (700 if mode == "smoke" else 5000))
    train_samples = max(readout_samples * 2, 1000)
    mlp_epochs = int(epochs if epochs is not None else (45 if mode == "smoke" else 260))
    rows: list[dict[str, object]] = []
    if gammas is not None:
        parameter_rows = [(float(gamma), float(gamma), float(0.35 if kappa is None else kappa)) for gamma in gammas]
        parameter_key = "gamma"
    else:
        gamma_start, gamma_stop = float(gamma_range[0]), float(gamma_range[1])
        kappa_start, kappa_stop = float(kappa_range[0]), float(kappa_range[1])
        parameter_rows = [
            (
                float(lambda_value),
                gamma_start + float(lambda_value) * (gamma_stop - gamma_start),
                kappa_start + float(lambda_value) * (kappa_stop - kappa_start),
            )
            for lambda_value in lambdas
        ]
        parameter_key = "lambda"
    for parameter_value, gamma, kappa_value in parameter_rows:
        for seed_value in seeds:
            seed = int(seed_value)
            train_states = sample_henon_states(samples=train_samples, seed=110000 + seed)
            readout_states = sample_henon_states(samples=readout_samples, seed=210000 + seed)
            train_noise = np.random.default_rng(410000 + seed).normal(0.0, float(unique_noise_sigma), size=train_samples)
            readout_noise = np.random.default_rng(510000 + seed).normal(
                0.0, float(unique_noise_sigma), size=readout_samples
            )
            train_targets = henon_fixed_interaction_with_unique_observation(
                train_states,
                gamma=gamma,
                noise=train_noise,
                kappa=kappa_value,
            )
            readout_targets = henon_fixed_interaction_with_unique_observation(
                readout_states,
                gamma=gamma,
                noise=readout_noise,
                kappa=kappa_value,
            )
            fitted = fit_mlp(train_states, train_targets, seed=310000 + seed, epochs=mlp_epochs)
            predicted_targets = fitted.predict(readout_states)
            left = readout_states[:, 0]
            right = readout_states[:, 1]
            observed = _histogram_wms_surd_mmi_any_target(left, right, readout_targets, bins=bins)
            learned = _histogram_triplet_any_target(left, right, predicted_targets, bins=bins)
            rows.append(
                {
                    parameter_key: parameter_value,
                    "gamma": gamma,
                    "kappa": float(kappa_value),
                    "seed": seed,
                    "wms": observed["wms"],
                    "surd_synergy": observed["surd_synergy"],
                    "shap_interaction": _two_source_vector_shap_interaction(
                        fitted,
                        readout_states,
                        samples=min(160, readout_samples),
                        seed=610000 + seed,
                    ),
                    "peid_synergy": learned["peid_residual"],
                    "mmi_pid_synergy": observed["mmi_pid_synergy"],
                    "observed_weaker_single_source_mi": observed["weaker_single_source_mi"],
                    "observed_left_mi": observed["left_mi"],
                    "observed_right_mi": observed["right_mi"],
                    "observed_joint_mi": observed["joint_mi"],
                    "mlp_train_mse": float(fitted.train_mse),
                    "mlp_baseline_mse": float(fitted.baseline_mse),
                    "train_state_digest": _digest(train_states),
                    "readout_state_digest": _digest(readout_states),
                    "observed_target_digest": _digest(readout_targets),
                    "mlp_target_digest": _digest(predicted_targets),
                }
            )
    plotted_metric_keys = ("wms", "surd_synergy", "shap_interaction", "peid_synergy", "mmi_pid_synergy")
    summary = (
        _summarize_by_parameter(rows, plotted_metric_keys)
        if parameter_key == "lambda"
        else _summarize_by_gamma(rows, plotted_metric_keys)
    )
    payload = {
        "mode": mode,
        "system": "controlled_henon_unique_information_five_method",
        "equation": "z_tau = [1 - 1.4*x^2 + kappa(lambda)*x*y, gamma(lambda)*y + epsilon]",
        "relation": "x+y->z_tau",
        "parameter_key": parameter_key,
        "lambdas": [float(value) for value in lambdas] if parameter_key == "lambda" else None,
        "gammas": [float(gamma) for gamma in gammas] if gammas is not None else [row[1] for row in parameter_rows],
        "gamma_range": [float(gamma_range[0]), float(gamma_range[1])] if parameter_key == "lambda" else None,
        "kappa_range": [float(kappa_range[0]), float(kappa_range[1])] if parameter_key == "lambda" else None,
        "kappa": float(kappa) if kappa is not None else None,
        "unique_noise_sigma": float(unique_noise_sigma),
        "estimator": "histogram",
        "bins": int(bins),
        "seeds": [int(seed) for seed in seeds],
        "samples": {"train": int(train_samples), "readout": int(readout_samples)},
        "mlp_epochs": int(mlp_epochs),
        "plotted_methods": list(plotted_metric_keys),
        "claim": "Scanning lambda increases additive single-source information while decreasing the explicit interaction coefficient.",
        "summary": summary,
        "rows": rows,
        "result_path": str(result_path),
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_five_method_sweep(summary, Path(figure_path))
    return payload


def run_henon_mmi_pid_vs_mlp_peid(
    *,
    mode: str = "full",
    seeds: Sequence[int] = (0, 1, 2),
    result_path: Path = DEFAULT_RESULT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    figure_path: Path = DEFAULT_FIGURE_PATH,
    samples: int | None = None,
    epochs: int | None = None,
    bins: int = 10,
) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    readout_samples = int(samples if samples is not None else (600 if mode == "smoke" else 5000))
    train_samples = max(readout_samples * 2, 800)
    mlp_epochs = int(epochs if epochs is not None else (40 if mode == "smoke" else 240))
    rows: list[dict[str, object]] = []
    for seed_value in seeds:
        seed = int(seed_value)
        train_states = sample_henon_states(samples=train_samples, seed=100000 + seed)
        train_targets = henon_next_x(train_states)
        readout_states = sample_henon_states(samples=readout_samples, seed=200000 + seed)
        readout_targets = henon_next_x(readout_states)
        fitted = fit_mlp(train_states, train_targets, seed=300000 + seed, epochs=mlp_epochs)
        predicted_targets = fitted.predict(readout_states)
        left = readout_states[:, 0]
        right = readout_states[:, 1]
        observed = _histogram_triplet(left, right, readout_targets[:, 0], bins=bins)
        mlp = _histogram_triplet(left, right, predicted_targets[:, 0], bins=bins)
        rows.append(
            {
                "seed": seed,
                "observed_mmi_pid_synergy": observed["mmi_pid_synergy"],
                "oracle_peid_residual": observed["peid_residual"],
                "mlp_peid_residual": mlp["peid_residual"],
                "observed_weaker_single_source_mi": observed["weaker_single_source_mi"],
                "observed_mmi_minus_oracle_peid": observed["mmi_minus_peid"],
                "observed_left_mi": observed["left_mi"],
                "observed_right_mi": observed["right_mi"],
                "observed_joint_mi": observed["joint_mi"],
                "mlp_left_mi": mlp["left_mi"],
                "mlp_right_mi": mlp["right_mi"],
                "mlp_joint_mi": mlp["joint_mi"],
                "mlp_train_mse": float(fitted.train_mse),
                "mlp_baseline_mse": float(fitted.baseline_mse),
                "train_state_digest": _digest(train_states),
                "readout_state_digest": _digest(readout_states),
                "observed_target_digest": _digest(readout_targets),
                "mlp_target_digest": _digest(predicted_targets),
            }
        )
    metric_keys = [
        "observed_mmi_pid_synergy",
        "oracle_peid_residual",
        "mlp_peid_residual",
        "observed_weaker_single_source_mi",
        "observed_mmi_minus_oracle_peid",
        "observed_left_mi",
        "observed_right_mi",
        "observed_joint_mi",
    ]
    summary: dict[str, float] = {}
    for key in metric_keys:
        mean, std = _mean_std(rows, key)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
    payload = {
        "mode": mode,
        "system": "classic_henon_map",
        "equation": "x_tau = 1 - 1.4*x^2 + y",
        "relation": "x+y->x_tau",
        "estimator": "histogram",
        "bins": int(bins),
        "seeds": [int(seed) for seed in seeds],
        "samples": {"train": int(train_samples), "readout": int(readout_samples)},
        "mlp_epochs": int(mlp_epochs),
        "claim": "MMI-PID synergy exceeds the PEID residual by the weaker single-source information term on an additive Hénon readout.",
        "summary": summary,
        "rows": rows,
        "result_path": str(result_path),
        "report_path": str(report_path),
        "figure_path": str(figure_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(rows, summary, Path(figure_path))
    _write_report(payload, Path(report_path), Path(figure_path))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unique-sweep", action="store_true", help="Run the gamma sweep instead of the static Hénon bar report.")
    parser.add_argument("--five-method", action="store_true", help="Run the Part1 five-method gamma sweep.")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.1666667, 0.3333333, 0.5, 0.6666667, 0.8333333, 1.0],
    )
    parser.add_argument("--gamma-range", type=float, nargs=2, default=[0.3, 2.0])
    parser.add_argument("--kappa-range", type=float, nargs=2, default=[0.5, 0.1])
    parser.add_argument("--unique-noise-sigma", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--result-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--figure-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.unique_sweep and args.five_method:
        result = run_henon_unique_five_method_sweep(
            mode=args.mode,
            lambdas=tuple(args.lambdas),
            gamma_range=(float(args.gamma_range[0]), float(args.gamma_range[1])),
            kappa_range=(float(args.kappa_range[0]), float(args.kappa_range[1])),
            unique_noise_sigma=args.unique_noise_sigma,
            seeds=tuple(args.seeds),
            result_path=args.result_path or DEFAULT_FIVE_METHOD_RESULT_PATH,
            figure_path=args.figure_path or DEFAULT_FIVE_METHOD_FIGURE_PATH,
            samples=args.samples,
            epochs=args.epochs,
            bins=args.bins,
        )
    elif args.unique_sweep:
        result = run_henon_unique_sweep_mmi_vs_mlp_peid(
            mode=args.mode,
            lambdas=tuple(args.lambdas),
            gamma_range=(float(args.gamma_range[0]), float(args.gamma_range[1])),
            kappa_range=(float(args.kappa_range[0]), float(args.kappa_range[1])),
            unique_noise_sigma=args.unique_noise_sigma,
            seeds=tuple(args.seeds),
            result_path=args.result_path or DEFAULT_SWEEP_RESULT_PATH,
            report_path=args.report_path or DEFAULT_SWEEP_REPORT_PATH,
            figure_path=args.figure_path or DEFAULT_SWEEP_FIGURE_PATH,
            samples=args.samples,
            epochs=args.epochs,
            bins=args.bins,
        )
    else:
        result = run_henon_mmi_pid_vs_mlp_peid(
            mode=args.mode,
            seeds=tuple(args.seeds),
            result_path=args.result_path or DEFAULT_RESULT_PATH,
            report_path=args.report_path or DEFAULT_REPORT_PATH,
            figure_path=args.figure_path or DEFAULT_FIGURE_PATH,
            samples=args.samples,
            epochs=args.epochs,
            bins=args.bins,
        )
    print(
        json.dumps(
            {
                "result_path": result["result_path"],
                "report_path": result.get("report_path"),
                "figure_path": result["figure_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
