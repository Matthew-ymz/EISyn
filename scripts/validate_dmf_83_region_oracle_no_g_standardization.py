from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_dmf_83_region_oracle_phi_eid import (
    CRITICAL_HIGH,
    CRITICAL_LOW,
    DEFAULT_SOURCE_RESULTS,
    build_intervention_samples,
    configure_matplotlib,
    dmf_step_batch,
    gaussian_singleton_source_phi,
    load_dmf_module,
    plot_results,
    resolve_path,
    safe_sem,
    standardize,
    summarize_peaks,
)


DEFAULT_OUTPUT_DIR = ROOT / "results" / "dmf_83_region_oracle_no_g_standardization"
DEFAULT_FIGURE_BASE = ROOT / "fig" / "dmf_83_region_oracle_no_g_standardization"
DEFAULT_DETDEG_BASE = ROOT / "fig" / "dmf_83_region_oracle_no_g_standardization_detdeg"
DEFAULT_REPORT = ROOT / "docs" / "log" / "dmf_83_region_oracle_no_g_standardization.md"
SOURCE_COUNT = 83


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())


def collect_seed_simulations(
    dmf,
    *,
    connectivity: np.ndarray,
    j_fic: np.ndarray,
    g_values: np.ndarray,
    selected: np.ndarray,
    parameters,
    stabilization,
    seed: int,
) -> list[dict[str, np.ndarray | float]]:
    simulations: list[dict[str, np.ndarray | float]] = []
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
        simulations.append(simulation)
        initial_se = np.asarray(simulation["final_se"], dtype=float)
        initial_si = np.asarray(simulation["final_si"], dtype=float)
    return simulations


def seed_global_source_scale(
    simulations: list[dict[str, np.ndarray | float]],
    *,
    tau: int,
) -> tuple[np.ndarray, np.ndarray]:
    traces = []
    for simulation in simulations:
        start_step = int(float(simulation["stabilization_start_step"]))
        se_trace = np.asarray(simulation["state_se_trace"], dtype=float)[start_step:]
        traces.append(se_trace[:-tau])
    _, mean, scale = standardize(np.vstack(traces))
    return mean, scale


def run_single_g_no_g_standardization(
    dmf,
    *,
    simulation: dict[str, np.ndarray | float],
    connectivity: np.ndarray,
    coupling_g: float,
    j_fic: np.ndarray,
    parameters,
    source_mean: np.ndarray,
    source_scale: np.ndarray,
    sample_count: int,
    tau: int,
    ridge: float,
    seed: int,
) -> dict[str, dict[str, object]]:
    start_step = int(float(simulation["stabilization_start_step"]))
    se_trace = np.asarray(simulation["state_se_trace"], dtype=float)[start_step:]
    si_trace = np.asarray(simulation["state_si_trace"], dtype=float)[start_step:]
    rng = np.random.default_rng(seed)
    background_indices = rng.integers(0, se_trace.shape[0] - tau, size=sample_count)
    background_si = si_trace[background_indices]
    out: dict[str, dict[str, object]] = {}
    for distribution, intervention_z in build_intervention_samples(
        rng,
        sample_count=sample_count,
        dimension=connectivity.shape[0],
    ).items():
        physical_source = source_mean.reshape(1, -1) + intervention_z * source_scale.reshape(1, -1)
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
        metrics = gaussian_singleton_source_phi(actual_source_z, target_se, ridge=ridge)
        metrics["clip_fraction"] = float(np.mean(out_of_bounds))
        out[distribution] = metrics
    return out


def save_detdeg_figure(
    *,
    figure_base: Path,
    g_values: np.ndarray,
    joint_ei: np.ndarray,
    singleton_sum: np.ndarray,
    target_entropy: np.ndarray,
    distribution_names: tuple[str, ...],
    distribution: str,
) -> dict[str, np.ndarray | float]:
    configure_matplotlib()
    d_index = distribution_names.index(distribution)
    joint = np.asarray(joint_ei[d_index], dtype=float)
    singleton = np.asarray(singleton_sum[d_index], dtype=float)
    entropy = np.asarray(target_entropy[d_index], dtype=float)
    plot_mask = g_values >= 1.1 - 1.0e-9
    x = g_values[plot_mask]
    joint = joint[:, plot_mask]
    singleton = singleton[:, plot_mask]
    entropy = entropy[:, plot_mask]
    h0 = float(np.nanmax(entropy))
    joint_degeneracy = h0 - entropy
    joint_determinism = joint + joint_degeneracy
    singleton_degeneracy_sum = float(SOURCE_COUNT) * joint_degeneracy
    singleton_determinism_sum = singleton + singleton_degeneracy_sum

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.3), constrained_layout=True, sharex=True)
    panels = (
        (axes[0, 0], joint_determinism, r"$Det(all;Y)$", "#4C78A8", "a  Whole EI determinism"),
        (axes[0, 1], joint_degeneracy, r"$Deg(all;Y)$", "#4C78A8", "b  Whole EI degeneracy"),
        (axes[1, 0], singleton_determinism_sum, r"$\sum_i Det(i;Y)$", "#D98C2F", "c  Singleton-sum determinism"),
        (axes[1, 1], singleton_degeneracy_sum, r"$\sum_i Deg(i;Y)$", "#D98C2F", "d  Singleton-sum degeneracy"),
    )
    for axis, values, ylabel, color, title in panels:
        mean = np.nanmean(values, axis=0)
        err = safe_sem(values, axis=0)
        axis.plot(x, mean, color=color, marker="o", linewidth=1.8, markersize=4.0, label=ylabel)
        axis.fill_between(x, mean - err, mean + err, color=color, alpha=0.16, linewidth=0.0)
        axis.axvspan(CRITICAL_LOW, CRITICAL_HIGH, color="0.90", zorder=0)
        axis.grid(True, color="0.88", lw=0.8)
        axis.set_xlabel("Global coupling G")
        axis.set_ylabel("Information component (bits)")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(
        f"83-region oracle determinism / degeneracy without per-G standardization ({distribution}, "
        + rf"$H_0={h0:.3f}$ bits)",
        fontsize=9,
        fontweight="bold",
    )
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_base.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(figure_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(figure_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "target_reference_entropy": h0,
        "joint_determinism": joint_determinism,
        "joint_degeneracy": joint_degeneracy,
        "singleton_determinism_sum": singleton_determinism_sum,
        "singleton_degeneracy_sum": singleton_degeneracy_sum,
    }


def write_report(
    *,
    report_path: Path,
    summary: dict[str, object],
    figure_base: Path,
    detdeg_base: Path,
    sample_count: int,
    ridge: float,
    clip_fraction: np.ndarray,
    components: dict[str, np.ndarray | float],
) -> None:
    uniform = summary["distributions"]["uniform"]
    conclusion = (
        "不支持用 Kuramoto 式 determinism/degeneracy 曲线解释 DMF 临界峰。"
        if not uniform["robust"]
        else "PhiEID 峰仍落在临界区，但 determinism/degeneracy 解释需看分量曲线是否同步支持。"
    )
    lines = [
        "# DMF 83-region oracle without per-G standardization",
        "",
        f"结论：{conclusion}",
        "",
        "- Variant: source intervention scale is shared across the whole G sweep within each seed; target is kept in physical `sE` units.",
        "- This removes the previous per-G target z-scoring and source-scale matching.",
        f"- Intervention samples per G/seed/distribution: {sample_count}",
        f"- Ridge: {ridge:g}",
        f"- Critical range: G={CRITICAL_LOW}-{CRITICAL_HIGH}",
        f"- PhiEID figure: `{figure_base.with_suffix('.png').relative_to(ROOT)}`",
        f"- Determinism/degeneracy figure: `{detdeg_base.with_suffix('.png').relative_to(ROOT)}`",
        f"- Uniform mean clip fraction: {float(np.nanmean(clip_fraction[1])):.2%}",
        f"- Uniform H0: {float(components['target_reference_entropy']):.6g} bits",
        "",
        "| Distribution | Peak hits | Top-2 hits | Median peak G | Robust | Mean clip fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for d_index, name in enumerate(summary["distributions"].keys()):
        item = summary["distributions"][name]
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
    lines.extend(["", "Peak rows:", "", "| Distribution | Seed | Peak G | Peak PhiEID | Peak in critical | Top-2 hits critical |", "|---|---:|---:|---:|---|---|"])
    for row in summary["rows"]:
        lines.append(
            "| {distribution} | {seed} | {peak_g:.2f} | {peak_phi_eid:.6g} | {peak_in_critical} | {top2_hits_critical} |".format(
                **row
            )
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DMF PhiEID without per-G standardization.")
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-base", type=Path, default=DEFAULT_FIGURE_BASE)
    parser.add_argument("--detdeg-base", type=Path, default=DEFAULT_DETDEG_BASE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seeds", type=parse_int_list, default=(0, 1, 2, 3, 4, 5, 6, 7))
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
    parameters = dmf.DMFParameters(t_total=args.t_total, burn_in=args.burn_in, dt=args.dt, sigma=args.sigma)
    stabilization = dmf.StabilizationParameters(
        window=args.stabilization_window,
        tolerance_hz=args.stabilization_tolerance,
        confirm_windows=args.stabilization_confirm_windows,
    )
    distribution_names = ("gaussian", "uniform")
    seed_values = tuple(args.seeds)
    shape = (len(distribution_names), len(seed_values), len(g_values))
    phi = np.full(shape, np.nan)
    joint_ei = np.full(shape, np.nan)
    singleton_sum = np.full(shape, np.nan)
    ctc = np.full(shape, np.nan)
    clip_fraction = np.full(shape, np.nan)
    target_entropy = np.full(shape, np.nan)
    joint_conditional_entropy = np.full(shape, np.nan)

    for s_index, seed in enumerate(seed_values):
        simulations = collect_seed_simulations(
            dmf,
            connectivity=connectivity,
            j_fic=j_fic,
            g_values=g_values,
            selected=selected,
            parameters=parameters,
            stabilization=stabilization,
            seed=int(seed),
        )
        source_mean, source_scale = seed_global_source_scale(simulations, tau=int(args.tau))
        for g_index, coupling_g in enumerate(g_values):
            metrics = run_single_g_no_g_standardization(
                dmf,
                simulation=simulations[g_index],
                connectivity=connectivity,
                coupling_g=float(coupling_g),
                j_fic=np.asarray(j_fic[g_index], dtype=float),
                parameters=parameters,
                source_mean=source_mean,
                source_scale=source_scale,
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
                target_entropy[d_index, s_index, g_index] = float(item["target_entropy"])
                joint_conditional_entropy[d_index, s_index, g_index] = float(item["joint_conditional_entropy"])
            print(
                f"seed={seed} G={coupling_g:.1f} "
                f"gaussian={phi[0, s_index, g_index]:.4g} "
                f"uniform={phi[1, s_index, g_index]:.4g} "
                f"clip={np.nanmean(clip_fraction[:, s_index, g_index]):.2%}",
                flush=True,
            )

    summary = summarize_peaks(g_values, phi, distribution_names=distribution_names, seed_values=seed_values)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "dmf_83_region_oracle_no_g_standardization_curves.npz",
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
        target_entropy=target_entropy,
        joint_conditional_entropy=joint_conditional_entropy,
        sample_count=int(args.sample_count),
        tau=int(args.tau),
        ridge=float(args.ridge),
        normalization="seed-global source scale and physical target",
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
    detdeg_base = resolve_path(args.detdeg_base)
    components = save_detdeg_figure(
        figure_base=detdeg_base,
        g_values=g_values,
        joint_ei=joint_ei,
        singleton_sum=singleton_sum,
        target_entropy=target_entropy,
        distribution_names=distribution_names,
        distribution="uniform",
    )
    write_report(
        report_path=resolve_path(args.report),
        summary=summary,
        figure_base=figure_base,
        detdeg_base=detdeg_base,
        sample_count=int(args.sample_count),
        ridge=float(args.ridge),
        clip_fraction=clip_fraction,
        components=components,
    )
    print(f"Saved cache to: {output_dir / 'dmf_83_region_oracle_no_g_standardization_curves.npz'}")
    print(f"Saved summary to: {output_dir / 'summary.json'}")
    print(f"Saved figure to: {figure_base.with_suffix('.png')}")
    print(f"Saved det/deg figure to: {detdeg_base.with_suffix('.png')}")
    print(f"Saved report to: {resolve_path(args.report)}")


if __name__ == "__main__":
    main()
