from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DMF_MODULE_PATH = ROOT / "exp" / "brain" / "dmf_fig6.py"
DEFAULT_SOURCE_RESULTS = ROOT / "exp" / "brain" / "result_lausanne_fig6" / "count_00_fig6b_mean_rate.npz"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "dmf_phi_eid_robustness"
DEFAULT_FIGURE_BASE = ROOT / "fig" / "dmf_phi_eid_robustness"
DEFAULT_REPORT = ROOT / "docs" / "log" / "dmf_phi_eid_robustness.md"


@dataclass(frozen=True)
class RunMode:
    label: str
    independent_restarts: bool


def load_dmf_module():
    spec = importlib.util.spec_from_file_location("dmf_fig6_exp_brain", DMF_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def lagged_samples(series: np.ndarray, tau: int) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(series, dtype=float)
    if array.ndim != 2 or array.shape[0] <= tau + 2:
        raise ValueError("series must have shape [time, region] with enough lagged samples.")
    return array[:-tau], array[tau:]


def safe_sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    count = np.sum(np.isfinite(array), axis=axis)
    ddof = 1 if np.nanmax(count) > 1 else 0
    std = np.nanstd(array, axis=axis, ddof=ddof)
    with np.errstate(invalid="ignore", divide="ignore"):
        return std / np.sqrt(np.maximum(count, 1))


def bootstrap_phi_eid(
    dmf,
    source: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float,
    count: int,
    seed: int,
    fraction: float,
) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=float)
    rng = np.random.default_rng(seed)
    n_samples = source.shape[0]
    draw_count = max(source.shape[1] + 3, int(round(n_samples * fraction)))
    values = np.empty(count, dtype=float)
    for index in range(count):
        sample_index = rng.integers(0, n_samples, size=draw_count)
        metrics = dmf.estimate_whole_system_phi_eid_from_lagged_samples(
            source[sample_index],
            target[sample_index],
            ridge=ridge,
        )
        values[index] = float(metrics["phi_eid"])
    return values


def run_single_curve(
    dmf,
    *,
    archive: np.lib.npyio.NpzFile,
    seed: int,
    mode: RunMode,
    tau: int,
    ridge: float,
    t_total: float,
    burn_in: float,
    dt: float,
    sigma: float,
    use_bold: bool,
    bootstrap_count: int,
    bootstrap_fraction: float,
    stabilization_window: float,
    stabilization_tolerance: float,
    stabilization_confirm_windows: int,
) -> dict[str, np.ndarray | str | int | bool]:
    g_values = np.asarray(archive["G"], dtype=float)
    connectivity = np.asarray(archive["connectivity"], dtype=float)
    j_fic = np.asarray(archive["j_fic"], dtype=float)
    parameters = dmf.DMFParameters(t_total=t_total, burn_in=burn_in, dt=dt, sigma=sigma)
    stabilization = dmf.StabilizationParameters(
        window=stabilization_window,
        tolerance_hz=stabilization_tolerance,
        confirm_windows=stabilization_confirm_windows,
    )

    phi_eid = np.empty(g_values.size, dtype=float)
    raw_phi_eid = np.empty(g_values.size, dtype=float)
    whole_ei = np.empty(g_values.size, dtype=float)
    singleton_sum = np.empty(g_values.size, dtype=float)
    sample_count = np.empty(g_values.size, dtype=float)
    mean_rate = np.empty(g_values.size, dtype=float)
    condition = np.empty(g_values.size, dtype=float)
    bootstrap = np.full((g_values.size, bootstrap_count), np.nan, dtype=float)

    initial_se = None
    initial_si = None
    for g_index, coupling_g in enumerate(g_values):
        simulation = dmf.simulate_dmf(
            connectivity,
            float(coupling_g),
            np.asarray(j_fic[g_index], dtype=float),
            parameters=parameters,
            stabilization_parameters=stabilization,
            seed=seed + g_index,
            initial_se=None if mode.independent_restarts else initial_se,
            initial_si=None if mode.independent_restarts else initial_si,
            record_rate_trace=True,
        )
        start_step = int(float(simulation["stabilization_start_step"]))
        rates = np.asarray(simulation["region_rate_trace_hz"], dtype=float)[start_step:]
        state_series = dmf.transform_rates_to_bold(rates, dt=dt) if use_bold else rates
        source, target = lagged_samples(state_series, tau)
        metrics = dmf.estimate_whole_system_phi_eid_from_lagged_samples(source, target, ridge=ridge)

        phi_eid[g_index] = float(metrics["phi_eid"])
        raw_phi_eid[g_index] = float(metrics["raw_phi_eid"])
        whole_ei[g_index] = float(metrics["whole_ei"])
        singleton_sum[g_index] = float(metrics["singleton_ei_sum"])
        sample_count[g_index] = float(metrics["sample_count"])
        mean_rate[g_index] = float(np.asarray(simulation["mean_rate_trace_hz"], dtype=float).mean())
        condition[g_index] = float(metrics["noise_condition_number"])
        bootstrap[g_index] = bootstrap_phi_eid(
            dmf,
            source,
            target,
            ridge=ridge,
            count=bootstrap_count,
            seed=seed * 1000 + g_index,
            fraction=bootstrap_fraction,
        )

        initial_se = np.asarray(simulation["final_se"], dtype=float)
        initial_si = np.asarray(simulation["final_si"], dtype=float)
        print(
            f"{mode.label} seed={seed} G={coupling_g:.1f} "
            f"PhiEID={phi_eid[g_index]:.4g} samples={sample_count[g_index]:.0f}",
            flush=True,
        )

    return {
        "seed": int(seed),
        "mode": mode.label,
        "independent_restarts": bool(mode.independent_restarts),
        "G": g_values,
        "mean_rate_hz": mean_rate,
        "phi_eid": phi_eid,
        "raw_phi_eid": raw_phi_eid,
        "whole_ei": whole_ei,
        "singleton_ei_sum": singleton_sum,
        "sample_count": sample_count,
        "noise_condition_number": condition,
        "bootstrap_phi_eid": bootstrap,
    }


def summarize_curves(
    curves: list[dict[str, np.ndarray | str | int | bool]],
    *,
    critical_low: float,
    critical_high: float,
) -> dict[str, np.ndarray | str]:
    if not curves:
        raise ValueError("No curves to summarize.")
    g_values = np.asarray(curves[0]["G"], dtype=float)
    analysis_mask = g_values > 1.0
    critical_mask = (g_values >= critical_low) & (g_values <= critical_high)
    rows = []
    for curve in curves:
        phi = np.asarray(curve["phi_eid"], dtype=float)
        finite = np.isfinite(phi) & analysis_mask
        peak_index = int(np.flatnonzero(finite)[np.nanargmax(phi[finite])])
        top2_indices = np.flatnonzero(finite)[np.argsort(phi[finite])[-2:]]
        top3_indices = np.flatnonzero(finite)[np.argsort(phi[finite])[-3:]]
        rows.append(
            {
                "mode": str(curve["mode"]),
                "seed": int(curve["seed"]),
                "peak_g": float(g_values[peak_index]),
                "peak_phi_eid": float(phi[peak_index]),
                "peak_in_critical": bool(critical_mask[peak_index]),
                "top2_hits_critical": int(np.sum(critical_mask[top2_indices])),
                "top3_hits_critical": int(np.sum(critical_mask[top3_indices])),
                "critical_max_phi_eid": float(np.nanmax(phi[critical_mask])),
                "noncritical_max_phi_eid": float(np.nanmax(phi[analysis_mask & ~critical_mask])),
            }
        )
    return {
        "G": g_values,
        "rows_json": json.dumps(rows, ensure_ascii=False),
    }


def write_curve_csv(curves: list[dict[str, np.ndarray | str | int | bool]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "seed",
        "G",
        "mean_rate_hz",
        "phi_eid",
        "raw_phi_eid",
        "whole_ei",
        "singleton_ei_sum",
        "sample_count",
        "noise_condition_number",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for curve in curves:
            g_values = np.asarray(curve["G"], dtype=float)
            for index, coupling_g in enumerate(g_values):
                writer.writerow(
                    {
                        "mode": str(curve["mode"]),
                        "seed": int(curve["seed"]),
                        "G": float(coupling_g),
                        "mean_rate_hz": float(np.asarray(curve["mean_rate_hz"], dtype=float)[index]),
                        "phi_eid": float(np.asarray(curve["phi_eid"], dtype=float)[index]),
                        "raw_phi_eid": float(np.asarray(curve["raw_phi_eid"], dtype=float)[index]),
                        "whole_ei": float(np.asarray(curve["whole_ei"], dtype=float)[index]),
                        "singleton_ei_sum": float(np.asarray(curve["singleton_ei_sum"], dtype=float)[index]),
                        "sample_count": float(np.asarray(curve["sample_count"], dtype=float)[index]),
                        "noise_condition_number": float(
                            np.asarray(curve["noise_condition_number"], dtype=float)[index]
                        ),
                    }
                )


def write_summary_csv(summary_rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "seed",
        "peak_g",
        "peak_phi_eid",
        "peak_in_critical",
        "top2_hits_critical",
        "top3_hits_critical",
        "critical_max_phi_eid",
        "noncritical_max_phi_eid",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def plot_robustness(
    curves: list[dict[str, np.ndarray | str | int | bool]],
    summary_rows: list[dict[str, object]],
    *,
    archive_mean_rate: np.ndarray,
    figure_base: Path,
    critical_low: float,
    critical_high: float,
) -> None:
    g_values = np.asarray(curves[0]["G"], dtype=float)
    modes = list(dict.fromkeys(str(curve["mode"]) for curve in curves))
    colors = {
        "continuation": "#0072B2",
        "independent": "#D55E00",
    }
    with plt.rc_context(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.55), constrained_layout=True)
        ax_rate, ax_curve, ax_peak = axes

        for axis in axes:
            axis.axvspan(critical_low, critical_high, color="0.90", zorder=0)
            axis.grid(True, color="0.88", lw=0.6, zorder=0)

        ax_rate.plot(g_values, archive_mean_rate, color="black", lw=1.5)
        ax_rate.scatter(g_values, archive_mean_rate, color="black", s=12, zorder=3)
        gradient = np.gradient(archive_mean_rate, g_values)
        gradient_scaled = gradient / np.nanmax(gradient) * np.nanmax(archive_mean_rate)
        ax_rate.plot(g_values, gradient_scaled, color="0.45", lw=1.0, ls="--", label="scaled slope")
        ax_rate.set_xlabel("Global coupling G")
        ax_rate.set_ylabel("Mean firing rate (Hz)")
        ax_rate.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

        for mode in modes:
            mode_curves = [curve for curve in curves if str(curve["mode"]) == mode]
            matrix = np.vstack([np.asarray(curve["phi_eid"], dtype=float) for curve in mode_curves])
            for row in matrix:
                ax_curve.plot(g_values, row, color=colors.get(mode, "0.3"), lw=0.7, alpha=0.22)
            mean = np.nanmean(matrix, axis=0)
            sem = safe_sem(matrix, axis=0)
            ax_curve.plot(g_values, mean, color=colors.get(mode, "0.3"), lw=1.7, label=mode)
            ax_curve.fill_between(
                g_values,
                mean - 1.96 * sem,
                mean + 1.96 * sem,
                color=colors.get(mode, "0.3"),
                alpha=0.14,
                lw=0,
            )
        ax_curve.set_xlabel("Global coupling G")
        ax_curve.set_ylabel(r"Whole-system $\Phi^{EID}$")
        ax_curve.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

        y_positions = {mode: index for index, mode in enumerate(modes)}
        for row in summary_rows:
            mode = str(row["mode"])
            ax_peak.scatter(
                float(row["peak_g"]),
                y_positions[mode] + 0.035 * (int(row["seed"]) % 5 - 2),
                color=colors.get(mode, "0.3"),
                s=20,
                alpha=0.75,
            )
        ax_peak.set_yticks(list(y_positions.values()))
        ax_peak.set_yticklabels(list(y_positions.keys()))
        ax_peak.set_xlabel(r"Peak $G^*$ from $\Phi^{EID}$")
        ax_peak.set_xlim(float(np.nanmin(g_values)) - 0.05, float(np.nanmax(g_values)) + 0.05)

        for label, axis in zip(("A", "B", "C"), axes):
            axis.text(-0.16, 1.05, label, transform=axis.transAxes, fontsize=10, fontweight="bold")

        figure_base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{figure_base}.png", dpi=600, bbox_inches="tight")
        fig.savefig(f"{figure_base}.svg", bbox_inches="tight")
        fig.savefig(f"{figure_base}.pdf", bbox_inches="tight")
        plt.close(fig)


def write_report(
    *,
    path: Path,
    summary_rows: list[dict[str, object]],
    curves: list[dict[str, np.ndarray | str | int | bool]],
    figure_base: Path,
    critical_low: float,
    critical_high: float,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_figure_base = figure_base.resolve() if not figure_base.is_absolute() else figure_base
    figure_ref = Path("../../") / resolved_figure_base.relative_to(ROOT)
    modes = list(dict.fromkeys(str(curve["mode"]) for curve in curves))
    lines = [
        "# DMF PhiEID 鲁棒性验证",
        "",
        f"目标：验证 whole-system $\\Phi^{{EID}}$ 的高值区是否稳定落入 $G={critical_low:.1f}\\text{{-}}{critical_high:.1f}$ 的 firing-rate 快速转变区。",
        "",
        f"![DMF PhiEID robustness]({figure_ref}.png)",
        "",
        "## 判据",
        "",
        "- 强判据：每条 seed 曲线的全局峰值 $G^*$ 落在临界区。",
        "- 弱判据：每条 seed 曲线的 top-2 或 top-3 高值至少有一个落在临界区。",
        "- 边界点 $G=1.0$ 不计入峰值判定。",
        "",
        "## 汇总",
        "",
        "| Mode | Seeds | Peak in band | Top-2 hit | Top-3 hit | Median peak G |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        rows = [row for row in summary_rows if str(row["mode"]) == mode]
        n_rows = len(rows)
        peak_hit = sum(bool(row["peak_in_critical"]) for row in rows)
        top2_hit = sum(int(row["top2_hits_critical"]) > 0 for row in rows)
        top3_hit = sum(int(row["top3_hits_critical"]) > 0 for row in rows)
        median_peak = float(np.median([float(row["peak_g"]) for row in rows]))
        lines.append(
            f"| `{mode}` | {n_rows} | {peak_hit}/{n_rows} | {top2_hit}/{n_rows} | {top3_hit}/{n_rows} | {median_peak:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 解释",
            "",
            "如果 strong 判据低但 weak 判据高，结论应写成“$\\Phi^{EID}$ 在临界邻域增强”，而不是“单点峰值鲁棒”。",
            "如果 strong 和 weak 判据都高，才适合写成“$\\Phi^{EID}$ 鲁棒识别临界区”。",
            "",
            "## Reproducibility metadata",
            "",
            "```json",
            json.dumps(metadata, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate robustness of DMF whole-system PhiEID critical detection.")
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-base", type=Path, default=DEFAULT_FIGURE_BASE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--modes", nargs="+", default=["continuation", "independent"])
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--t-total", type=float, default=0.55)
    parser.add_argument("--burn-in", type=float, default=0.15)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--use-bold", action="store_true")
    parser.add_argument("--bootstrap-count", type=int, default=8)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.65)
    parser.add_argument("--critical-low", type=float, default=1.7)
    parser.add_argument("--critical-high", type=float, default=1.9)
    parser.add_argument("--stabilization-window", type=float, default=0.05)
    parser.add_argument("--stabilization-tolerance", type=float, default=0.05)
    parser.add_argument("--stabilization-confirm-windows", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dmf = load_dmf_module()
    archive = np.load(args.source_results)
    mode_lookup = {
        "continuation": RunMode("continuation", independent_restarts=False),
        "independent": RunMode("independent", independent_restarts=True),
    }
    modes = [mode_lookup[name] for name in args.modes]
    curves: list[dict[str, np.ndarray | str | int | bool]] = []
    for mode in modes:
        for seed in args.seeds:
            curves.append(
                run_single_curve(
                    dmf,
                    archive=archive,
                    seed=int(seed),
                    mode=mode,
                    tau=args.tau,
                    ridge=args.ridge,
                    t_total=args.t_total,
                    burn_in=args.burn_in,
                    dt=args.dt,
                    sigma=args.sigma,
                    use_bold=args.use_bold,
                    bootstrap_count=args.bootstrap_count,
                    bootstrap_fraction=args.bootstrap_fraction,
                    stabilization_window=args.stabilization_window,
                    stabilization_tolerance=args.stabilization_tolerance,
                    stabilization_confirm_windows=args.stabilization_confirm_windows,
                )
            )

    summary = summarize_curves(curves, critical_low=args.critical_low, critical_high=args.critical_high)
    summary_rows = json.loads(str(summary["rows_json"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "dmf_phi_eid_robustness.npz",
        curves=np.asarray(curves, dtype=object),
        summary_rows=np.asarray(summary_rows, dtype=object),
        metadata=json.dumps(vars(args), default=str),
    )
    write_curve_csv(curves, args.output_dir / "dmf_phi_eid_robustness_curves.csv")
    write_summary_csv(summary_rows, args.output_dir / "dmf_phi_eid_robustness_summary.csv")
    plot_robustness(
        curves,
        summary_rows,
        archive_mean_rate=np.asarray(archive["mean_rate_hz"], dtype=float),
        figure_base=args.figure_base,
        critical_low=args.critical_low,
        critical_high=args.critical_high,
    )
    metadata = {
        "source_results": str(args.source_results),
        "seeds": [int(seed) for seed in args.seeds],
        "modes": [mode.label for mode in modes],
        "tau": int(args.tau),
        "ridge": float(args.ridge),
        "t_total": float(args.t_total),
        "burn_in": float(args.burn_in),
        "dt": float(args.dt),
        "sigma": float(args.sigma),
        "bootstrap_count": int(args.bootstrap_count),
        "bootstrap_fraction": float(args.bootstrap_fraction),
        "critical_band": [float(args.critical_low), float(args.critical_high)],
        "caveat": "Lausanne-33 count-derived 83-region approximation, not exact paper HCP Lausanne-83 SC.",
    }
    write_report(
        path=args.report,
        summary_rows=summary_rows,
        curves=curves,
        figure_base=args.figure_base,
        critical_low=args.critical_low,
        critical_high=args.critical_high,
        metadata=metadata,
    )
    print(f"Saved curves: {args.output_dir / 'dmf_phi_eid_robustness_curves.csv'}")
    print(f"Saved summary: {args.output_dir / 'dmf_phi_eid_robustness_summary.csv'}")
    print(f"Saved figure base: {args.figure_base}")
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
