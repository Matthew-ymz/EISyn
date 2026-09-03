from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
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
DEFAULT_CONNECTIVITY_LABELS = ROOT / "exp" / "brain" / "result_lausanne_fig6" / "count_00_connectivity.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "dmf_module_tm_phi_eid"
DEFAULT_FIGURE_BASE = ROOT / "fig" / "dmf_module_tm_phi_eid_robustness"
DEFAULT_REPORT = ROOT / "docs" / "log" / "dmf_module_tm_phi_eid_robustness.md"
CRITICAL_LOW = 1.7
CRITICAL_HIGH = 1.9


@dataclass(frozen=True)
class InterventionConfig:
    name: str
    samples: np.ndarray


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


def module_matrix(module_indices: Mapping[str, Sequence[int]], n_regions: int) -> tuple[list[str], np.ndarray]:
    from scripts.plot_dmf_phi_eid_greedy_decomposition import MODULE_ORDER

    names = [name for name in MODULE_ORDER if name in module_indices]
    matrix = np.zeros((n_regions, len(names)), dtype=float)
    for col, name in enumerate(names):
        indices = np.asarray(module_indices[name], dtype=int)
        if indices.size == 0:
            raise ValueError(f"Module {name!r} has no regions.")
        matrix[indices, col] = 1.0 / float(indices.size)
    return names, matrix


def load_module_indices(label_path: Path, n_regions: int) -> dict[str, list[int]]:
    from scripts.plot_dmf_phi_eid_greedy_decomposition import (
        load_region_labels,
        module_indices_from_labels,
    )

    labels = load_region_labels(label_path, n_regions)
    return module_indices_from_labels(labels)


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


def sample_interventions(
    rng: np.random.Generator,
    *,
    sample_count: int,
    dimension: int,
) -> list[InterventionConfig]:
    gaussian = rng.standard_normal((sample_count, dimension))
    uniform = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=(sample_count, dimension))
    return [
        InterventionConfig("gaussian", gaussian),
        InterventionConfig("uniform", uniform),
    ]


def embed_module_intervention(
    background_se: np.ndarray,
    intervention_z: np.ndarray,
    *,
    module_names: Sequence[str],
    module_indices: Mapping[str, Sequence[int]],
    module_mean: np.ndarray,
    module_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    se = np.asarray(background_se, dtype=float).copy()
    desired_module_mean = module_mean.reshape(1, -1) + intervention_z * module_scale.reshape(1, -1)
    for col, name in enumerate(module_names):
        indices = np.asarray(module_indices[name], dtype=int)
        current = se[:, indices].mean(axis=1, keepdims=True)
        se[:, indices] = se[:, indices] + desired_module_mean[:, [col]] - current
    out_of_bounds = (se < 0.0) | (se > 1.0)
    clip_fraction = float(np.mean(out_of_bounds))
    clipped = np.clip(se, 0.0, 1.0)
    actual_module = np.column_stack(
        [clipped[:, np.asarray(module_indices[name], dtype=int)].mean(axis=1) for name in module_names]
    )
    actual_z = (actual_module - module_mean.reshape(1, -1)) / module_scale.reshape(1, -1)
    return clipped, actual_z, clip_fraction


def estimate_tm_phi_eid(
    source_z: np.ndarray,
    target: np.ndarray,
    *,
    degree: int,
    jitter: float,
) -> dict[str, float]:
    from exp.TM.transport_map_density import estimate_mutual_information_transport_map

    target_z, _, _ = standardize(target)
    whole = estimate_mutual_information_transport_map(source_z, target_z, degree=degree, jitter=jitter)
    whole_ei = max(0.0, float(whole["mi_hat"]))
    singleton_values = []
    for col in range(source_z.shape[1]):
        summary = estimate_mutual_information_transport_map(
            source_z[:, [col]],
            target_z,
            degree=degree,
            jitter=jitter,
        )
        singleton_values.append(max(0.0, float(summary["mi_hat"])))
    singleton_sum = float(np.sum(singleton_values))
    raw_phi = float(whole_ei - singleton_sum)
    return {
        "whole_ei": whole_ei,
        "singleton_sum": singleton_sum,
        "raw_phi_eid": raw_phi,
        "phi_eid": raw_phi,
    }


def run_single_g(
    dmf,
    *,
    simulation: Mapping[str, np.ndarray | float],
    connectivity: np.ndarray,
    coupling_g: float,
    j_fic: np.ndarray,
    parameters,
    module_names: Sequence[str],
    module_indices: Mapping[str, Sequence[int]],
    module_projection: np.ndarray,
    sample_count: int,
    tau: int,
    degree: int,
    jitter: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    start_step = int(float(simulation["stabilization_start_step"]))
    se_trace = np.asarray(simulation["state_se_trace"], dtype=float)[start_step:]
    si_trace = np.asarray(simulation["state_si_trace"], dtype=float)[start_step:]
    if se_trace.shape[0] <= tau + 2:
        raise ValueError("Stable state trace is too short for the requested tau.")

    source_module = se_trace[:-tau] @ module_projection
    _, module_mean, module_scale = standardize(source_module)
    rng = np.random.default_rng(seed)
    background_indices = rng.integers(0, se_trace.shape[0] - tau, size=sample_count)
    background_se = se_trace[background_indices]
    background_si = si_trace[background_indices]

    out: dict[str, dict[str, float]] = {}
    for intervention in sample_interventions(
        rng,
        sample_count=sample_count,
        dimension=len(module_names),
    ):
        intervened_se, actual_source_z, clip_fraction = embed_module_intervention(
            background_se,
            intervention.samples,
            module_names=module_names,
            module_indices=module_indices,
            module_mean=module_mean,
            module_scale=module_scale,
        )
        target_se = intervened_se
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
        target_module = target_se @ module_projection
        metrics = estimate_tm_phi_eid(
            actual_source_z,
            target_module,
            degree=degree,
            jitter=jitter,
        )
        metrics["clip_fraction"] = clip_fraction
        out[intervention.name] = metrics
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


def safe_sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    count = np.sum(np.isfinite(values), axis=axis)
    std = np.nanstd(values, axis=axis, ddof=1 if np.nanmax(count) > 1 else 0)
    return std / np.sqrt(np.maximum(count, 1))


def plot_results(
    *,
    figure_base: Path,
    g_values: np.ndarray,
    mean_rate_hz: np.ndarray,
    phi: np.ndarray,
    clip_fraction: np.ndarray,
    distribution_names: Sequence[str],
    seed_values: Sequence[int],
) -> None:
    configure_matplotlib()
    colors = {"gaussian": "#0072B2", "uniform": "#D55E00"}
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9.2, 3.1),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.35, 1.0]},
    )
    ax_rate, ax_phi, ax_peak = axes
    ax_rate.plot(g_values, mean_rate_hz, color="0.25", lw=1.4)
    ax_rate.scatter(g_values, mean_rate_hz, color="black", s=14)
    ax_rate.axvspan(CRITICAL_LOW, CRITICAL_HIGH, color="0.86", zorder=0)
    ax_rate.set_xlabel("Global coupling G")
    ax_rate.set_ylabel("Mean firing rate (Hz)")
    ax_rate.grid(True, color="0.88", lw=0.8)

    for d_index, distribution in enumerate(distribution_names):
        values = phi[d_index]
        mean = np.nanmean(values, axis=0)
        sem = safe_sem(values, axis=0)
        color = colors.get(distribution, None)
        label = f"{distribution} intervention"
        ax_phi.plot(g_values, mean, color=color, lw=1.6, label=label)
        ax_phi.fill_between(g_values, mean - sem, mean + sem, color=color, alpha=0.18, lw=0.0)
    ax_phi.axvspan(CRITICAL_LOW, CRITICAL_HIGH, color="0.90", zorder=0)
    ax_phi.set_xlabel("Global coupling G")
    ax_phi.set_ylabel(r"Module $\Phi^{EID}_{TM}$ (bits)")
    ax_phi.set_ylim(bottom=0.0)
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

    mean_clip = np.nanmean(clip_fraction, axis=(1, 2))
    for d_index, value in enumerate(mean_clip):
        ax_peak.text(
            1.02,
            y_positions[d_index],
            f"clip {value:.1%}",
            transform=ax_peak.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=7,
            color="0.35",
        )

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
    degree: int,
    module_names: Sequence[str],
    clip_fraction: np.ndarray,
) -> None:
    distributions = summary["distributions"]
    robust_all = all(item["robust"] for item in distributions.values())
    conclusion = (
        "支持当前 module-level TM-PhiEID 口径下对临界区的鲁棒识别。"
        if robust_all
        else "PhiEID 在当前 module-level TM 干预协议下不鲁棒或仅部分鲁棒，不能强行声称准确识别临界点。"
    )
    lines = [
        "# DMF module-level TM-PhiEID robustness",
        "",
        f"结论：{conclusion}",
        "",
        f"- Modules: {', '.join(module_names)}",
        f"- Intervention samples per G/seed/distribution: {sample_count}",
        f"- TM degree: {degree}",
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
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate module-level DMF TM-PhiEID critical-point robustness.")
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--connectivity-labels", type=Path, default=DEFAULT_CONNECTIVITY_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-base", type=Path, default=DEFAULT_FIGURE_BASE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seeds", type=parse_int_list, default=(0, 1, 2, 3, 4, 5, 6, 7))
    parser.add_argument("--g-stride", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--jitter", type=float, default=1.0e-6)
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
    module_indices = load_module_indices(resolve_path(args.connectivity_labels), connectivity.shape[0])
    module_names, module_projection = module_matrix(module_indices, connectivity.shape[0])
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
    distribution_names = ("gaussian", "uniform")
    seed_values = tuple(args.seeds)
    shape = (len(distribution_names), len(seed_values), len(g_values))
    phi = np.full(shape, np.nan, dtype=float)
    raw_phi = np.full(shape, np.nan, dtype=float)
    whole_ei = np.full(shape, np.nan, dtype=float)
    singleton_sum = np.full(shape, np.nan, dtype=float)
    clip_fraction = np.full(shape, np.nan, dtype=float)

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
                module_names=module_names,
                module_indices=module_indices,
                module_projection=module_projection,
                sample_count=int(args.sample_count),
                tau=int(args.tau),
                degree=int(args.degree),
                jitter=float(args.jitter),
                seed=int(seed) * 100000 + g_index * 1000 + int(args.degree),
            )
            for d_index, distribution in enumerate(distribution_names):
                item = metrics[distribution]
                phi[d_index, s_index, g_index] = item["phi_eid"]
                raw_phi[d_index, s_index, g_index] = item["raw_phi_eid"]
                whole_ei[d_index, s_index, g_index] = item["whole_ei"]
                singleton_sum[d_index, s_index, g_index] = item["singleton_sum"]
                clip_fraction[d_index, s_index, g_index] = item["clip_fraction"]
            initial_se = np.asarray(simulation["final_se"], dtype=float)
            initial_si = np.asarray(simulation["final_si"], dtype=float)
            print(
                f"seed={seed} G={coupling_g:.1f} "
                f"gaussian={phi[0, s_index, g_index]:.4g} "
                f"uniform={phi[1, s_index, g_index]:.4g} "
                f"clip={np.nanmean(clip_fraction[:, s_index, g_index]):.2%}",
                flush=True,
            )

    summary = summarize_peaks(
        g_values,
        phi,
        distribution_names=distribution_names,
        seed_values=seed_values,
    )
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "module_tm_phi_eid_curves.npz",
        G=g_values,
        selected_g_indices=selected,
        seeds=np.asarray(seed_values, dtype=int),
        distributions=np.asarray(distribution_names, dtype=object),
        module_names=np.asarray(module_names, dtype=object),
        mean_rate_hz=mean_rate_hz,
        phi_eid=phi,
        raw_phi_eid=raw_phi,
        whole_ei=whole_ei,
        singleton_sum=singleton_sum,
        clip_fraction=clip_fraction,
        sample_count=int(args.sample_count),
        tau=int(args.tau),
        degree=int(args.degree),
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
        clip_fraction=clip_fraction[plot_distribution_index : plot_distribution_index + 1],
        distribution_names=("uniform",),
        seed_values=seed_values,
    )
    write_report(
        report_path=resolve_path(args.report),
        summary=summary,
        figure_base=figure_base,
        sample_count=int(args.sample_count),
        degree=int(args.degree),
        module_names=module_names,
        clip_fraction=clip_fraction,
    )
    print(f"Saved cache to: {output_dir / 'module_tm_phi_eid_curves.npz'}")
    print(f"Saved summary to: {output_dir / 'summary.json'}")
    print(f"Saved figure to: {figure_base.with_suffix('.png')}")
    print(f"Saved report to: {resolve_path(args.report)}")


if __name__ == "__main__":
    main()
