from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DMF_MODULE_PATH = ROOT / "exp" / "brain" / "dmf_fig6.py"
DEFAULT_SOURCE_RESULTS = ROOT / "exp" / "brain" / "result_lausanne_fig6" / "count_00_fig6b_mean_rate.npz"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "dmf_83_region_oracle_phi_eid"
DEFAULT_FIGURE_BASE = ROOT / "fig" / "dmf_83_region_oracle_phi_eid_robustness"
DEFAULT_REPORT = ROOT / "docs" / "log" / "dmf_83_region_oracle_phi_eid_robustness.md"
CRITICAL_LOW = 1.7
CRITICAL_HIGH = 1.9


def load_dmf_module():
    spec = importlib.util.spec_from_file_location("dmf_fig6_exp_brain", DMF_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def standardize(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(samples, dtype=float)
    mean = array.mean(axis=0, keepdims=True)
    scale = array.std(axis=0, ddof=1, keepdims=True)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return (array - mean) / scale, mean.reshape(-1), scale.reshape(-1)


def safe_logdet_psd(matrix: np.ndarray, *, floor: float = 1.0e-12) -> float:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    return float(np.log(np.maximum(eigenvalues, float(floor))).sum())


def gaussian_singleton_source_phi(
    source: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float,
) -> dict[str, object]:
    source_array = np.asarray(source, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if source_array.ndim != 2 or target_array.ndim != 2:
        raise ValueError("source and target must be 2D arrays.")
    if source_array.shape[0] != target_array.shape[0]:
        raise ValueError("source and target must have matching sample counts.")

    sample_count, source_dim = source_array.shape
    target_dim = target_array.shape[1]
    empirical_source_cov = np.cov(source_array, rowvar=False, bias=False)
    empirical_source_cov = np.atleast_2d(empirical_source_cov)
    source_cov = np.diag(np.diag(empirical_source_cov))
    source_cov += float(ridge) * np.eye(source_dim)

    coefficient, *_ = np.linalg.lstsq(source_array, target_array, rcond=None)
    transition = coefficient.T
    residual = target_array - source_array @ coefficient
    noise_cov = np.cov(residual, rowvar=False, bias=False)
    noise_cov = np.atleast_2d(noise_cov) + float(ridge) * np.eye(target_dim)

    target_cov = transition @ source_cov @ transition.T + noise_cov
    target_cov = 0.5 * (target_cov + target_cov.T) + float(ridge) * np.eye(target_dim)
    source_target_cov = source_cov @ transition.T
    target_cov_pinv = np.linalg.pinv(target_cov)
    conditional_source_cov = source_cov - source_target_cov @ target_cov_pinv @ source_target_cov.T
    conditional_source_cov = (
        0.5 * (conditional_source_cov + conditional_source_cov.T)
        + float(ridge) * np.eye(source_dim)
    )

    joint_ei = 0.5 * (safe_logdet_psd(source_cov) - safe_logdet_psd(conditional_source_cov)) / math.log(2.0)
    singleton_ei = np.empty(source_dim, dtype=float)
    gaussian_entropy_constant = target_dim * math.log(2.0 * math.pi * math.e)
    target_entropy = 0.5 * (gaussian_entropy_constant + safe_logdet_psd(target_cov)) / math.log(2.0)
    joint_conditional_entropy = 0.5 * (
        gaussian_entropy_constant + safe_logdet_psd(noise_cov)
    ) / math.log(2.0)
    singleton_conditional_entropy = np.empty(source_dim, dtype=float)
    singleton_target_base_cov = target_cov + float(ridge) * np.eye(target_dim)
    singleton_target_base_logdet = safe_logdet_psd(singleton_target_base_cov)
    singleton_target_base_pinv = np.linalg.pinv(singleton_target_base_cov)
    singleton_conditional_logdet_sum = 0.0
    for index in range(source_dim):
        prior = source_cov[index : index + 1, index : index + 1]
        conditional = conditional_source_cov[index : index + 1, index : index + 1]
        singleton_ei[index] = 0.5 * (safe_logdet_psd(prior) - safe_logdet_psd(conditional)) / math.log(2.0)
        singleton_conditional_logdet_sum += safe_logdet_psd(conditional)
        singleton_source_target_cov = source_target_cov[index : index + 1, :]
        explained_fraction = float(
            (singleton_source_target_cov @ singleton_target_base_pinv @ singleton_source_target_cov.T)[0, 0]
            / float(prior[0, 0])
        )
        target_given_singleton_logdet = singleton_target_base_logdet + math.log(
            max(1.0 - explained_fraction, 1.0e-12)
        )
        singleton_conditional_entropy[index] = 0.5 * (
            gaussian_entropy_constant + target_given_singleton_logdet
        ) / math.log(2.0)
    singleton_sum = float(np.sum(singleton_ei))
    phi = float(joint_ei - singleton_sum)
    conditional_total_correlation = float(
        0.5 * (singleton_conditional_logdet_sum - safe_logdet_psd(conditional_source_cov)) / math.log(2.0)
    )
    return {
        "sample_count": int(sample_count),
        "source_dim": int(source_dim),
        "target_dim": int(target_dim),
        "phi": phi,
        "raw_phi": phi,
        "conditional_total_correlation": conditional_total_correlation,
        "joint_ei": float(joint_ei),
        "singleton_ei_sum": singleton_sum,
        "singleton_ei": singleton_ei,
        "target_entropy": float(target_entropy),
        "joint_conditional_entropy": float(joint_conditional_entropy),
        "singleton_conditional_entropy_sum": float(np.sum(singleton_conditional_entropy)),
        "noise_condition_number": float(np.linalg.cond(noise_cov)),
        "source_condition_number": float(np.linalg.cond(source_cov)),
    }


def dmf_step_batch(
    dmf,
    se: np.ndarray,
    si: np.ndarray,
    *,
    connectivity: np.ndarray,
    coupling_g: float,
    j_fic: np.ndarray,
    parameters,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    dt = float(parameters.dt)
    input_e = (
        parameters.w_e * parameters.i0
        + parameters.w_plus * parameters.j_nmda * se
        + float(coupling_g) * parameters.j_nmda * (se @ connectivity.T)
        - j_fic.reshape(1, -1) * si
    )
    input_i = parameters.w_i * parameters.i0 + parameters.j_nmda * se - si
    rate_e = dmf.transfer_function(
        input_e,
        gain=parameters.gain_e,
        threshold=parameters.threshold_e,
        shape=parameters.shape_e,
    )
    rate_i = dmf.transfer_function(
        input_i,
        gain=parameters.gain_i,
        threshold=parameters.threshold_i,
        shape=parameters.shape_i,
    )
    noise_e = parameters.sigma * np.sqrt(dt) * rng.standard_normal(se.shape)
    noise_i = parameters.sigma * np.sqrt(dt) * rng.standard_normal(si.shape)
    dse = dt * (-se / parameters.tau_e + (1.0 - se) * parameters.gamma_e * rate_e) + noise_e
    dsi = dt * (-si / parameters.tau_i + rate_i) + noise_i
    return np.clip(se + dse, 0.0, 1.0), np.clip(si + dsi, 0.0, 1.0)


def build_intervention_samples(
    rng: np.random.Generator,
    *,
    sample_count: int,
    dimension: int,
) -> dict[str, np.ndarray]:
    return {
        "gaussian": rng.standard_normal((sample_count, dimension)),
        "uniform": rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=(sample_count, dimension)),
        "fixed_uniform": rng.uniform(0.0, 1.0, size=(sample_count, dimension)),
    }


def run_single_g(
    dmf,
    *,
    simulation: Mapping[str, np.ndarray | float],
    connectivity: np.ndarray,
    coupling_g: float,
    j_fic: np.ndarray,
    parameters,
    sample_count: int,
    tau: int,
    ridge: float,
    seed: int,
) -> dict[str, dict[str, object]]:
    start_step = int(float(simulation["stabilization_start_step"]))
    se_trace = np.asarray(simulation["state_se_trace"], dtype=float)[start_step:]
    si_trace = np.asarray(simulation["state_si_trace"], dtype=float)[start_step:]
    if se_trace.shape[0] <= tau + 2:
        raise ValueError("Stable state trace is too short for the requested tau.")

    source_trace = se_trace[:-tau]
    _, source_mean, source_scale = standardize(source_trace)
    rng = np.random.default_rng(seed)
    background_indices = rng.integers(0, se_trace.shape[0] - tau, size=sample_count)
    background_si = si_trace[background_indices]
    out: dict[str, dict[str, object]] = {}
    for distribution, intervention_values in build_intervention_samples(
        rng,
        sample_count=sample_count,
        dimension=connectivity.shape[0],
    ).items():
        if distribution == "fixed_uniform":
            source_se = intervention_values
            out_of_bounds = np.zeros_like(source_se, dtype=bool)
        else:
            physical_source = source_mean.reshape(1, -1) + intervention_values * source_scale.reshape(1, -1)
            out_of_bounds = (physical_source < 0.0) | (physical_source > 1.0)
            source_se = np.clip(physical_source, 0.0, 1.0)
        actual_source_z = (source_se - source_mean.reshape(1, -1)) / source_scale.reshape(1, -1)
        target_se = source_se
        target_si = background_si.copy()
        for _ in range(tau):
            target_se, target_si = dmf_step_batch(
                dmf,
                target_se,
                target_si,
                connectivity=connectivity,
                coupling_g=coupling_g,
                j_fic=j_fic,
                parameters=parameters,
                rng=rng,
            )
        target_z, _, _ = standardize(target_se)
        metrics = gaussian_singleton_source_phi(actual_source_z, target_z, ridge=ridge)
        metrics["clip_fraction"] = float(np.mean(out_of_bounds))
        out[distribution] = metrics
    return out


def summarize_peaks(
    g_values: np.ndarray,
    phi: np.ndarray,
    *,
    distribution_names: Sequence[str],
    seed_values: Sequence[int],
) -> dict[str, object]:
    critical_mask = (g_values >= CRITICAL_LOW) & (g_values <= CRITICAL_HIGH)
    analysis_mask = g_values > 1.0
    rows: list[dict[str, object]] = []
    distributions: dict[str, object] = {}
    for d_index, distribution in enumerate(distribution_names):
        peak_values = []
        peak_hits = []
        top2_hits = []
        for s_index, seed in enumerate(seed_values):
            values = phi[d_index, s_index]
            finite_indices = np.flatnonzero(np.isfinite(values) & analysis_mask)
            if finite_indices.size == 0:
                continue
            order = finite_indices[np.argsort(values[finite_indices])]
            peak_index = int(order[-1])
            top2 = order[-2:] if order.size >= 2 else order
            peak_g = float(g_values[peak_index])
            peak_values.append(peak_g)
            peak_hit = bool(critical_mask[peak_index])
            top2_hit = bool(np.any(critical_mask[top2]))
            peak_hits.append(peak_hit)
            top2_hits.append(top2_hit)
            rows.append(
                {
                    "distribution": distribution,
                    "seed": int(seed),
                    "peak_g": peak_g,
                    "peak_phi_eid": float(values[peak_index]),
                    "peak_in_critical": peak_hit,
                    "top2_hits_critical": top2_hit,
                }
            )
        distributions[distribution] = {
            "n_seeds": int(len(peak_values)),
            "peak_hit_count": int(np.sum(peak_hits)),
            "top2_hit_count": int(np.sum(top2_hits)),
            "median_peak_g": float(np.median(peak_values)) if peak_values else None,
            "robust": bool(
                len(peak_values) > 0
                and np.sum(peak_hits) >= min(6, len(peak_values))
                and np.sum(top2_hits) >= min(7, len(peak_values))
                and CRITICAL_LOW <= float(np.median(peak_values)) <= CRITICAL_HIGH
            ),
        }
    return {"critical_range": [CRITICAL_LOW, CRITICAL_HIGH], "rows": rows, "distributions": distributions}


def safe_sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    count = np.sum(np.isfinite(values), axis=axis)
    std = np.nanstd(values, axis=axis, ddof=1 if np.nanmax(count) > 1 else 0)
    return std / np.sqrt(np.maximum(count, 1))


def configure_matplotlib() -> None:
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
            "legend.frameon": False,
        }
    )


def plot_results(
    *,
    figure_base: Path,
    g_values: np.ndarray,
    mean_rate_hz: np.ndarray,
    phi: np.ndarray,
    distribution_names: Sequence[str],
    seed_values: Sequence[int],
) -> None:
    configure_matplotlib()
    colors = {"gaussian": "#0072B2", "uniform": "#D55E00"}
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1), constrained_layout=True)
    ax_rate, ax_phi, ax_peak = axes
    ax_rate.plot(g_values, mean_rate_hz, color="0.25", lw=1.4)
    ax_rate.scatter(g_values, mean_rate_hz, color="black", s=14)
    ax_rate.axvspan(CRITICAL_LOW, CRITICAL_HIGH, color="0.88", zorder=0)
    ax_rate.set_xlabel("Global coupling G")
    ax_rate.set_ylabel("Mean firing rate (Hz)")
    ax_rate.grid(True, color="0.88", lw=0.8)

    for d_index, distribution in enumerate(distribution_names):
        values = phi[d_index]
        mean = np.nanmean(values, axis=0)
        sem = safe_sem(values, axis=0)
        color = colors.get(distribution)
        ax_phi.plot(g_values, mean, color=color, lw=1.6, label=f"{distribution} intervention")
        ax_phi.fill_between(g_values, mean - sem, mean + sem, color=color, alpha=0.18, lw=0.0)
    ax_phi.axhline(0.0, color="0.55", lw=0.9, ls="--")
    ax_phi.axvspan(CRITICAL_LOW, CRITICAL_HIGH, color="0.90", zorder=0)
    ax_phi.set_xlabel("Global coupling G")
    ax_phi.set_ylabel(r"83-region signed $\Phi^{EID}$ (bits)")
    ax_phi.grid(True, color="0.88", lw=0.8)
    ax_phi.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    y_positions = np.arange(len(distribution_names), dtype=float)
    for d_index, distribution in enumerate(distribution_names):
        values = phi[d_index]
        peak_g = []
        for s_index in range(len(seed_values)):
            finite = np.flatnonzero(np.isfinite(values[s_index]) & (g_values > 1.0))
            if finite.size:
                peak_g.append(float(g_values[finite[np.argmax(values[s_index, finite])]]))
        if peak_g:
            ax_peak.scatter(
                peak_g,
                np.full(len(peak_g), y_positions[d_index]),
                color=colors.get(distribution, "0.25"),
                s=26,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.4,
            )
    ax_peak.axvspan(CRITICAL_LOW, CRITICAL_HIGH, color="0.90", zorder=0)
    ax_peak.set_yticks(y_positions)
    ax_peak.set_yticklabels(distribution_names)
    ax_peak.set_xlabel("Peak G")
    ax_peak.set_xlim(float(g_values.min()) - 0.05, float(g_values.max()) + 0.05)
    ax_peak.grid(True, axis="x", color="0.88", lw=0.8)

    for label, axis in zip(("A", "B", "C"), axes):
        axis.text(-0.18, 1.05, label, transform=axis.transAxes, fontsize=12, fontweight="bold")
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_base.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(figure_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(figure_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_report(
    *,
    report_path: Path,
    summary: Mapping[str, object],
    figure_base: Path,
    sample_count: int,
    ridge: float,
    clip_fraction: np.ndarray,
) -> None:
    distributions = summary["distributions"]
    robust_all = all(item["robust"] for item in distributions.values())
    conclusion = (
        "支持当前 Kuramoto-aligned 83-region whole-state 口径下对临界区的鲁棒识别。"
        if robust_all
        else "当前 Kuramoto-aligned 83-region whole-state 口径未通过鲁棒识别标准。"
    )
    lines = [
        "# DMF 83-region oracle PhiEID robustness",
        "",
        f"结论：{conclusion}",
        "",
        "- Source: 83 singleton region `sE(t)` values.",
        "- Target: whole-system 83D future `sE(t+tau)` state.",
        "- Estimator: Kuramoto-aligned Gaussian block conditional total correlation, signed raw PhiEID, no nonnegative clipping.",
        f"- Intervention samples per G/seed/distribution: {sample_count}",
        f"- Ridge: {ridge:g}",
        f"- Critical range: G={CRITICAL_LOW}-{CRITICAL_HIGH}",
        f"- Figure: `{figure_base.with_suffix('.png').relative_to(ROOT)}`",
        "",
        "| Distribution | Peak hits | Top-2 hits | Median peak G | Robust | Mean clip fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    names = list(distributions.keys())
    for d_index, name in enumerate(names):
        item = distributions[name]
        lines.append(
            "| {name} | {peak}/{n} | {top2}/{n} | {median} | {robust} | {clip:.2%} |".format(
                name=name,
                peak=item["peak_hit_count"],
                top2=item["top2_hit_count"],
                n=item["n_seeds"],
                median="NA" if item["median_peak_g"] is None else f"{item['median_peak_g']:.2f}",
                robust="yes" if item["robust"] else "no",
                clip=float(np.nanmean(clip_fraction[d_index])),
            )
        )
    lines.extend(
        [
            "",
            "Peak rows:",
            "",
            "| Distribution | Seed | Peak G | Peak PhiEID | Peak in critical | Top-2 hits critical |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            "| {distribution} | {seed} | {peak_g:.2f} | {peak_phi_eid:.6g} | {peak_in_critical} | {top2_hits_critical} |".format(
                **row
            )
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())


def parse_distribution_list(raw: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    allowed = {"gaussian", "uniform", "fixed_uniform"}
    if not names or any(name not in allowed for name in names):
        raise argparse.ArgumentTypeError("distributions must be drawn from gaussian, uniform, fixed_uniform.")
    return names


def format_progress_message(
    *,
    seed: int,
    coupling_g: float,
    distribution_names: Sequence[str],
    phi_values: np.ndarray,
    clip_values: np.ndarray,
) -> str:
    scores = " ".join(
        f"{name}={float(value):.4g}" for name, value in zip(distribution_names, phi_values, strict=True)
    )
    return (
        f"seed={seed} G={coupling_g:.1f} {scores} "
        f"clip={float(np.nanmean(clip_values)):.2%}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Kuramoto-aligned 83-region DMF oracle PhiEID.")
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-base", type=Path, default=DEFAULT_FIGURE_BASE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seeds", type=parse_int_list, default=(0, 1, 2, 3, 4, 5, 6, 7))
    parser.add_argument("--distributions", type=parse_distribution_list, default=("gaussian", "uniform"))
    parser.add_argument("--g-stride", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--t-total", type=float, default=1.05)
    parser.add_argument("--burn-in", type=float, default=0.3)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--stabilization-window", type=float, default=0.05)
    parser.add_argument("--stabilization-tolerance", type=float, default=0.15)
    parser.add_argument("--stabilization-confirm-windows", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dmf = load_dmf_module()
    archive = np.load(resolve_path(args.source_results))
    all_g = np.asarray(archive["G"], dtype=float)
    selected = np.arange(0, all_g.size, max(1, int(args.g_stride)))
    g_values = all_g[selected]
    connectivity = np.asarray(archive["connectivity"], dtype=float)
    j_fic = np.asarray(archive["j_fic"], dtype=float)[selected]
    mean_rate_hz = np.asarray(archive["mean_rate_hz"], dtype=float)[selected]
    parameters = dmf.DMFParameters(
        t_total=float(args.t_total),
        burn_in=float(args.burn_in),
        dt=float(args.dt),
        sigma=float(args.sigma),
    )
    stabilization = dmf.StabilizationParameters(
        window=float(args.stabilization_window),
        tolerance_hz=float(args.stabilization_tolerance),
        confirm_windows=int(args.stabilization_confirm_windows),
    )
    distribution_names = tuple(args.distributions)
    seed_values = tuple(args.seeds)
    shape = (len(distribution_names), len(seed_values), len(g_values))
    phi = np.full(shape, np.nan, dtype=float)
    joint_ei = np.full(shape, np.nan, dtype=float)
    singleton_sum = np.full(shape, np.nan, dtype=float)
    ctc = np.full(shape, np.nan, dtype=float)
    clip_fraction = np.full(shape, np.nan, dtype=float)
    noise_condition = np.full(shape, np.nan, dtype=float)
    target_entropy = np.full(shape, np.nan, dtype=float)
    joint_conditional_entropy = np.full(shape, np.nan, dtype=float)

    for s_index, seed in enumerate(seed_values):
        initial_se = None
        initial_si = None
        for g_index, coupling_g in enumerate(g_values):
            simulation = dmf.simulate_dmf(
                connectivity,
                float(coupling_g),
                np.asarray(j_fic[g_index], dtype=float),
                parameters=parameters,
                stabilization_parameters=stabilization,
                seed=int(seed) + int(selected[g_index]),
                initial_se=initial_se,
                initial_si=initial_si,
                record_rate_trace=False,
                record_state_trace=True,
            )
            metrics = run_single_g(
                dmf,
                simulation=simulation,
                connectivity=connectivity,
                coupling_g=float(coupling_g),
                j_fic=np.asarray(j_fic[g_index], dtype=float),
                parameters=parameters,
                sample_count=int(args.sample_count),
                tau=int(args.tau),
                ridge=float(args.ridge),
                seed=int(seed) * 100000 + g_index * 1000,
            )
            for d_index, distribution in enumerate(distribution_names):
                item = metrics[distribution]
                phi[d_index, s_index, g_index] = float(item["raw_phi"])
                joint_ei[d_index, s_index, g_index] = float(item["joint_ei"])
                singleton_sum[d_index, s_index, g_index] = float(item["singleton_ei_sum"])
                ctc[d_index, s_index, g_index] = float(item["conditional_total_correlation"])
                clip_fraction[d_index, s_index, g_index] = float(item["clip_fraction"])
                noise_condition[d_index, s_index, g_index] = float(item["noise_condition_number"])
                target_entropy[d_index, s_index, g_index] = float(item["target_entropy"])
                joint_conditional_entropy[d_index, s_index, g_index] = float(item["joint_conditional_entropy"])
            initial_se = np.asarray(simulation["final_se"], dtype=float)
            initial_si = np.asarray(simulation["final_si"], dtype=float)
            print(
                format_progress_message(
                    seed=int(seed),
                    coupling_g=float(coupling_g),
                    distribution_names=distribution_names,
                    phi_values=phi[:, s_index, g_index],
                    clip_values=clip_fraction[:, s_index, g_index],
                ),
                flush=True,
            )

    summary = summarize_peaks(g_values, phi, distribution_names=distribution_names, seed_values=seed_values)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "dmf_83_region_oracle_phi_eid_curves.npz",
        G=g_values,
        selected_g_indices=selected,
        seeds=np.asarray(seed_values, dtype=int),
        distributions=np.asarray(distribution_names, dtype=object),
        mean_rate_hz=mean_rate_hz,
        phi_eid=phi,
        raw_phi_eid=phi,
        joint_ei=joint_ei,
        singleton_sum=singleton_sum,
        conditional_total_correlation=ctc,
        clip_fraction=clip_fraction,
        noise_condition_number=noise_condition,
        target_entropy=target_entropy,
        joint_conditional_entropy=joint_conditional_entropy,
        sample_count=int(args.sample_count),
        tau=int(args.tau),
        ridge=float(args.ridge),
        t_total=float(args.t_total),
        burn_in=float(args.burn_in),
        dt=float(args.dt),
        sigma=float(args.sigma),
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    figure_base = resolve_path(args.figure_base)
    plot_distribution_index = distribution_names.index("uniform")
    plot_results(
        figure_base=figure_base,
        g_values=g_values,
        mean_rate_hz=mean_rate_hz,
        phi=phi[plot_distribution_index : plot_distribution_index + 1],
        distribution_names=("uniform",),
        seed_values=seed_values,
    )
    write_report(
        report_path=resolve_path(args.report),
        summary=summary,
        figure_base=figure_base,
        sample_count=int(args.sample_count),
        ridge=float(args.ridge),
        clip_fraction=clip_fraction,
    )
    print(f"Saved cache to: {output_dir / 'dmf_83_region_oracle_phi_eid_curves.npz'}")
    print(f"Saved summary to: {output_dir / 'summary.json'}")
    print(f"Saved figure to: {figure_base.with_suffix('.png')}")
    print(f"Saved report to: {resolve_path(args.report)}")


if __name__ == "__main__":
    main()
