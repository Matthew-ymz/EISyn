from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_unicm_all_mode_target_pair_syn import extract_all_mode_target, parse_leads  # noqa: E402
from scripts.unicm_peid_syn_analysis import (  # noqa: E402
    HISTORY_LENGTH,
    MODE_NAMES,
    create_ei_estimator,
    estimate_gaussian_mutual_information,
    load_full_history_prediction_cache,
    overall_prediction_cache_path,
    sample_full_history_mode_inputs,
)

DEFAULT_CACHE_DIR = ROOT / "results" / "unicm_overall_ei_cpu_bound4_n8192" / "cache"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "unicm_month_resolved_phi_eid_tm_degree1_signed_n8192"
DEFAULT_ASSET_BASE = ROOT / "fig" / "unicm_month_resolved_phi_eid_tm_degree1_signed"

Estimator = Callable[[np.ndarray, np.ndarray], float]


def history_index_from_tau(tau: int) -> int:
    """Map tau in {-11, ..., 0} to the zero-based UniCM history index."""

    value = int(tau)
    if value < -(HISTORY_LENGTH - 1) or value > 0:
        raise ValueError(f"tau must be in [-{HISTORY_LENGTH - 1}, 0], got {tau}.")
    return value + HISTORY_LENGTH - 1


def parse_taus(values: Sequence[str | int] | None) -> list[int]:
    if not values:
        return list(range(-(HISTORY_LENGTH - 1), 1))
    taus = [int(value) for value in values]
    for tau in taus:
        history_index_from_tau(tau)
    return taus


def compute_month_resolved_phi_eid_for_target(
    history_modes: np.ndarray,
    target: np.ndarray,
    *,
    tau: int,
    mode_names: Mapping[str, int] = MODE_NAMES,
    estimator: Estimator = estimate_gaussian_mutual_information,
    clip_phi: bool = True,
) -> dict[str, float | dict[str, float]]:
    history = np.asarray(history_modes, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if history.ndim != 3:
        raise ValueError("history_modes must have shape (n_samples, history_length, n_modes).")
    if history.shape[1] != HISTORY_LENGTH or history.shape[2] != len(mode_names):
        raise ValueError(
            "history_modes must have shape "
            f"(n_samples, {HISTORY_LENGTH}, {len(mode_names)}), got {tuple(history.shape)}."
        )
    if target_array.ndim == 1:
        target_array = target_array.reshape(-1, 1)
    if target_array.ndim != 2 or target_array.shape[0] != history.shape[0]:
        raise ValueError("target must be 2D and share the sample axis with history_modes.")

    month_index = history_index_from_tau(int(tau))
    whole_source = history[:, month_index, :]
    whole_ei = float(estimator(whole_source, target_array))
    singleton_ei: dict[str, float] = {}
    for mode_name, mode_index in mode_names.items():
        source = history[:, month_index, [int(mode_index)]]
        singleton_ei[str(mode_name)] = float(estimator(source, target_array))
    singleton_sum = float(sum(singleton_ei.values()))
    raw_phi = float(whole_ei - singleton_sum)
    return {
        "whole_ei": whole_ei,
        "singleton_ei_sum": singleton_sum,
        "singleton_ei_mean": singleton_sum / float(len(mode_names)),
        "raw_phi_eid": raw_phi,
        "phi_eid": max(0.0, raw_phi) if bool(clip_phi) else raw_phi,
        "singleton_ei": singleton_ei,
    }


def compute_month_resolved_rows(
    history_modes: np.ndarray,
    targets_by_seed: Mapping[int, np.ndarray],
    *,
    taus: Sequence[int],
    leads: Sequence[int],
    mode_names: Mapping[str, int] = MODE_NAMES,
    estimator: Estimator = estimate_gaussian_mutual_information,
    clip_phi: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    singleton_rows: list[dict[str, object]] = []
    for seed in sorted(int(seed) for seed in targets_by_seed):
        predictions = np.asarray(targets_by_seed[int(seed)], dtype=float)
        for tau in [int(value) for value in taus]:
            for lead in [int(value) for value in leads]:
                target = extract_all_mode_target(predictions, lead=lead)
                metrics = compute_month_resolved_phi_eid_for_target(
                    history_modes,
                    target,
                    tau=tau,
                    mode_names=mode_names,
                    estimator=estimator,
                    clip_phi=clip_phi,
                )
                singleton_sum = float(metrics["singleton_ei_sum"])
                singleton_mean = float(metrics["singleton_ei_mean"])
                phi = float(metrics["phi_eid"])
                whole = float(metrics["whole_ei"])
                rows.append(
                    {
                        "seed": seed,
                        "target": "all_modes",
                        "target_dim": int(target.shape[1]),
                        "tau": tau,
                        "history_index": history_index_from_tau(tau),
                        "lead": lead,
                        "whole_ei": whole,
                        "singleton_ei_sum": singleton_sum,
                        "singleton_ei_mean": singleton_mean,
                        "raw_phi_eid": float(metrics["raw_phi_eid"]),
                        "phi_eid": phi,
                        "phi_to_whole": phi / whole if abs(whole) > 1.0e-12 else np.nan,
                        "phi_to_singleton_sum": phi / singleton_sum if abs(singleton_sum) > 1.0e-12 else np.nan,
                        "phi_to_singleton_mean": phi / singleton_mean if abs(singleton_mean) > 1.0e-12 else np.nan,
                    }
                )
                for mode_name, value in dict(metrics["singleton_ei"]).items():
                    singleton_rows.append(
                        {
                            "seed": seed,
                            "target": "all_modes",
                            "target_dim": int(target.shape[1]),
                            "tau": tau,
                            "history_index": history_index_from_tau(tau),
                            "lead": lead,
                            "source": mode_name,
                            "singleton_ei": float(value),
                        }
                    )
    return (
        pd.DataFrame(rows).sort_values(["seed", "tau", "lead"]).reset_index(drop=True),
        pd.DataFrame(singleton_rows).sort_values(["source", "seed", "tau", "lead"]).reset_index(drop=True),
    )


def summarize_month_resolved(rows: pd.DataFrame) -> pd.DataFrame:
    summary = rows.groupby(["tau", "history_index", "lead"], as_index=False).agg(
        phi_eid_mean=("phi_eid", "mean"),
        phi_eid_std=("phi_eid", "std"),
        raw_phi_eid_mean=("raw_phi_eid", "mean"),
        raw_phi_eid_std=("raw_phi_eid", "std"),
        whole_ei_mean=("whole_ei", "mean"),
        whole_ei_std=("whole_ei", "std"),
        singleton_ei_sum_mean=("singleton_ei_sum", "mean"),
        singleton_ei_sum_std=("singleton_ei_sum", "std"),
        singleton_ei_mean_mean=("singleton_ei_mean", "mean"),
        singleton_ei_mean_std=("singleton_ei_mean", "std"),
        phi_to_whole_mean=("phi_to_whole", "mean"),
        phi_to_singleton_sum_mean=("phi_to_singleton_sum", "mean"),
        phi_to_singleton_mean_mean=("phi_to_singleton_mean", "mean"),
    )
    for column in summary.columns:
        if column.endswith("_std"):
            summary[column] = summary[column].fillna(0.0)
    return summary.sort_values(["tau", "lead"]).reset_index(drop=True)


def summarize_tau(summary: pd.DataFrame) -> pd.DataFrame:
    tau_summary = summary.groupby(["tau", "history_index"], as_index=False).agg(
        phi_eid_mean_1_24=("phi_eid_mean", "mean"),
        phi_eid_max_1_24=("phi_eid_mean", "max"),
        whole_ei_mean_1_24=("whole_ei_mean", "mean"),
        singleton_ei_mean_1_24=("singleton_ei_mean_mean", "mean"),
        phi_to_singleton_mean_1_24=("phi_to_singleton_mean_mean", "mean"),
        phi_to_whole_mean_1_24=("phi_to_whole_mean", "mean"),
    )
    return tau_summary.sort_values("tau").reset_index(drop=True)


def configure_matplotlib() -> None:
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
            "legend.frameon": False,
        }
    )


def _matrix(summary: pd.DataFrame, column: str, taus: Sequence[int], leads: Sequence[int]) -> np.ndarray:
    pivot = summary.pivot(index="tau", columns="lead", values=column)
    return pivot.reindex(index=list(taus), columns=list(leads)).to_numpy(dtype=float)


def _imshow_with_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    values: np.ndarray,
    *,
    title: str,
    cmap: str,
    norm: mcolors.Normalize | None = None,
) -> None:
    image = ax.imshow(values, aspect="auto", origin="lower", cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=7, fontweight="bold", pad=4)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=6)


def plot_month_resolved_heatmaps(summary: pd.DataFrame, output_base: Path) -> list[Path]:
    configure_matplotlib()
    taus = sorted(int(value) for value in summary["tau"].unique())
    leads = sorted(int(value) for value in summary["lead"].unique())
    phi = _matrix(summary, "phi_eid_mean", taus, leads)
    singleton_mean = _matrix(summary, "singleton_ei_mean_mean", taus, leads)
    ratio = _matrix(summary, "phi_to_singleton_mean_mean", taus, leads)
    whole = _matrix(summary, "whole_ei_mean", taus, leads)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), constrained_layout=True, sharex=True, sharey=True)
    phi_abs = float(np.nanmax(np.abs(phi))) if np.isfinite(phi).any() else 1.0
    phi_norm = mcolors.TwoSlopeNorm(vmin=-phi_abs, vcenter=0.0, vmax=phi_abs) if np.nanmin(phi) < 0 else None
    _imshow_with_colorbar(fig, axes[0, 0], phi, title=r"$\Phi^{EID}_{\tau,\ell}$ (bits)", cmap="RdBu_r" if phi_norm else "Blues", norm=phi_norm)
    _imshow_with_colorbar(fig, axes[0, 1], singleton_mean, title="Mean singleton EI (bits)", cmap="Purples")
    ratio_cap = np.nanpercentile(np.abs(ratio[np.isfinite(ratio)]), 95) if np.isfinite(ratio).any() else 1.0
    ratio_cap = max(float(ratio_cap), 1.0e-6)
    ratio_norm = mcolors.TwoSlopeNorm(vmin=-ratio_cap, vcenter=0.0, vmax=ratio_cap)
    _imshow_with_colorbar(fig, axes[1, 0], ratio, title=r"$\Phi$ / mean singleton EI", cmap="RdBu_r", norm=ratio_norm)
    _imshow_with_colorbar(fig, axes[1, 1], whole, title="Whole EI (bits)", cmap="Greys")

    for ax in axes.flat:
        ax.set_xticks(np.arange(0, len(leads), 4))
        ax.set_xticklabels([str(leads[index]) for index in range(0, len(leads), 4)])
        ax.set_yticks(np.arange(len(taus)))
        ax.set_yticklabels([str(tau) for tau in taus])
        ax.tick_params(length=0)
    for ax in axes[1, :]:
        ax.set_xlabel("Lead (months)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"History lag $\tau$")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    svg_path = output_base.with_suffix(".svg")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, svg_path]


def _cache_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        n_samples=int(args.n_samples),
        sampling_seed=int(args.sampling_seed),
        intervention_bound=float(args.intervention_bound),
        start_month=int(args.start_month),
        device=str(args.device),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    leads = parse_leads(args.leads)
    taus = parse_taus(args.taus)
    seeds = [int(seed) for seed in args.seeds]
    history_modes = sample_full_history_mode_inputs(
        n_samples=int(args.n_samples),
        intervention_bound=float(args.intervention_bound),
        seed=int(args.sampling_seed),
    )
    cache_args = _cache_args(args)
    targets_by_seed = {
        seed: load_full_history_prediction_cache(
            overall_prediction_cache_path(Path(args.cache_dir), seed=seed, args=cache_args),
            n_samples=int(args.n_samples),
        )
        for seed in seeds
    }
    estimator, estimator_metadata = create_ei_estimator(
        str(args.estimator),
        tm_degree=int(args.tm_degree),
        tm_jitter=float(args.tm_jitter),
        clip_negative=not bool(args.no_mi_clip),
    )
    rows, singleton_rows = compute_month_resolved_rows(
        history_modes,
        targets_by_seed,
        taus=taus,
        leads=leads,
        mode_names=MODE_NAMES,
        estimator=estimator,
        clip_phi=not bool(args.no_phi_clip),
    )
    summary = summarize_month_resolved(rows)
    tau_summary = summarize_tau(summary)

    paths = {
        "rows": output_dir / "month_resolved_phi_eid_rows.csv",
        "singleton_rows": output_dir / "month_resolved_phi_eid_singleton_rows.csv",
        "summary": output_dir / "month_resolved_phi_eid_summary.csv",
        "tau_summary": output_dir / "month_resolved_phi_eid_tau_summary.csv",
    }
    rows.to_csv(paths["rows"], index=False)
    singleton_rows.to_csv(paths["singleton_rows"], index=False)
    summary.to_csv(paths["summary"], index=False)
    tau_summary.to_csv(paths["tau_summary"], index=False)
    figures = plot_month_resolved_heatmaps(summary, Path(args.asset_base))
    manifest = {
        "target_definition": "all 11 predicted UniCM modes at each lead as a multivariate target",
        "source_definition": "month-resolved singleton partition; each source is one mode at one history lag tau",
        "phi_eid_definition": (
            "I(all modes at tau; all-mode target) - sum_i I(mode_i at tau; all-mode target)"
            if bool(args.no_phi_clip)
            else "max(0, I(all modes at tau; all-mode target) - sum_i I(mode_i at tau; all-mode target))"
        ),
        "clip_phi": not bool(args.no_phi_clip),
        "seeds": seeds,
        "taus": taus,
        "leads": leads,
        "n_samples": int(args.n_samples),
        "sampling_seed": int(args.sampling_seed),
        "intervention_bound": float(args.intervention_bound),
        "target_dim": int(len(MODE_NAMES)),
        "tables": {key: str(path) for key, path in paths.items()},
        "figures": [str(path) for path in figures],
        "cache_dir": str(args.cache_dir),
        "estimator": estimator_metadata,
    }
    manifest_path = output_dir / "month_resolved_phi_eid_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute month-resolved UniCM all-mode target Phi^EID heatmaps.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--asset-base", type=Path, default=DEFAULT_ASSET_BASE)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--taus", nargs="*", default=None)
    parser.add_argument("--leads", nargs="*", default=None)
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument("--sampling-seed", type=int, default=20260619)
    parser.add_argument("--intervention-bound", type=float, default=4.0)
    parser.add_argument("--start-month", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--estimator", choices=["gaussian_logdet", "transport_map"], default="transport_map")
    parser.add_argument("--tm-degree", type=int, default=1)
    parser.add_argument("--tm-jitter", type=float, default=1.0e-6)
    parser.add_argument("--no-phi-clip", action="store_true", help="Plot and summarize signed raw PhiEID values.")
    parser.add_argument("--no-mi-clip", action="store_true", help="Do not clip individual MI estimates at zero.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    outputs = run(build_arg_parser().parse_args(argv))
    print(json.dumps(outputs, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
