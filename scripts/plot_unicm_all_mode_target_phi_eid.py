from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_unicm_all_mode_target_pair_syn import extract_all_mode_target, parse_leads  # noqa: E402
from scripts.unicm_peid_syn_analysis import (  # noqa: E402
    MODE_NAMES,
    create_ei_estimator,
    estimate_gaussian_mutual_information,
    load_full_history_prediction_cache,
    overall_prediction_cache_path,
    sample_full_history_mode_inputs,
)

DEFAULT_CACHE_DIR = ROOT / "results" / "unicm_overall_ei_cpu_bound4_n8192" / "cache"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "unicm_all_mode_target_phi_eid_cpu_bound4_n8192"
DEFAULT_ASSET_BASE = ROOT / "fig" / "unicm_all_mode_target_phi_eid_leads"

Estimator = Callable[[np.ndarray, np.ndarray], float]


def compute_phi_eid_for_target(
    history_modes: np.ndarray,
    target: np.ndarray,
    *,
    mode_names: Mapping[str, int] = MODE_NAMES,
    estimator: Estimator = estimate_gaussian_mutual_information,
) -> dict[str, float | dict[str, float]]:
    history = np.asarray(history_modes, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if history.ndim != 3:
        raise ValueError("history_modes must have shape (n_samples, history_length, n_modes).")
    if target_array.ndim == 1:
        target_array = target_array.reshape(-1, 1)
    if target_array.ndim != 2 or target_array.shape[0] != history.shape[0]:
        raise ValueError("target must be 2D and share the sample axis with history_modes.")

    whole_source = history.reshape(history.shape[0], history.shape[1] * history.shape[2])
    whole_ei = float(estimator(whole_source, target_array))
    singleton_ei: dict[str, float] = {}
    for mode_name, mode_index in mode_names.items():
        source = history[:, :, int(mode_index)]
        singleton_ei[str(mode_name)] = float(estimator(source, target_array))
    singleton_sum = float(sum(singleton_ei.values()))
    raw_phi = float(whole_ei - singleton_sum)
    return {
        "whole_ei": whole_ei,
        "singleton_ei_sum": singleton_sum,
        "raw_phi_eid": raw_phi,
        "phi_eid": raw_phi,
        "singleton_ei": singleton_ei,
    }


def compute_phi_eid_rows(
    history_modes: np.ndarray,
    targets_by_seed: Mapping[int, np.ndarray],
    *,
    mode_names: Mapping[str, int] = MODE_NAMES,
    leads: Sequence[int] | None = None,
    estimator: Estimator = estimate_gaussian_mutual_information,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lead_values = list(range(1, 25)) if leads is None else [int(lead) for lead in leads]
    rows: list[dict[str, object]] = []
    singleton_rows: list[dict[str, object]] = []
    for seed in sorted(int(seed) for seed in targets_by_seed):
        predictions = np.asarray(targets_by_seed[int(seed)], dtype=float)
        for lead in lead_values:
            target = extract_all_mode_target(predictions, lead=int(lead))
            metrics = compute_phi_eid_for_target(
                history_modes,
                target,
                mode_names=mode_names,
                estimator=estimator,
            )
            rows.append(
                {
                    "seed": int(seed),
                    "target": "all_modes",
                    "target_dim": int(target.shape[1]),
                    "lead": int(lead),
                    "whole_ei": float(metrics["whole_ei"]),
                    "singleton_ei_sum": float(metrics["singleton_ei_sum"]),
                    "raw_phi_eid": float(metrics["raw_phi_eid"]),
                    "phi_eid": float(metrics["phi_eid"]),
                }
            )
            for mode_name, value in dict(metrics["singleton_ei"]).items():
                singleton_rows.append(
                    {
                        "seed": int(seed),
                        "target": "all_modes",
                        "target_dim": int(target.shape[1]),
                        "lead": int(lead),
                        "source": mode_name,
                        "singleton_ei": float(value),
                    }
                )
    return (
        pd.DataFrame(rows).sort_values(["seed", "lead"]).reset_index(drop=True),
        pd.DataFrame(singleton_rows).sort_values(["source", "seed", "lead"]).reset_index(drop=True),
    )


def summarize_phi_eid_leads(rows: pd.DataFrame) -> pd.DataFrame:
    summary = rows.groupby("lead", as_index=False).agg(
        phi_eid_mean=("phi_eid", "mean"),
        phi_eid_std=("phi_eid", "std"),
        raw_phi_eid_mean=("raw_phi_eid", "mean"),
        raw_phi_eid_std=("raw_phi_eid", "std"),
        whole_ei_mean=("whole_ei", "mean"),
        whole_ei_std=("whole_ei", "std"),
        singleton_ei_sum_mean=("singleton_ei_sum", "mean"),
        singleton_ei_sum_std=("singleton_ei_sum", "std"),
    )
    for column in summary.columns:
        if column.endswith("_std"):
            summary[column] = summary[column].fillna(0.0)
    return summary.sort_values("lead").reset_index(drop=True)


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


def plot_phi_eid_leads(summary: pd.DataFrame, output_base: Path) -> list[Path]:
    configure_matplotlib()
    fig, (ax_phi, ax_ei) = plt.subplots(
        2,
        1,
        figsize=(6.8, 5.2),
        constrained_layout=True,
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )
    x = summary["lead"].to_numpy(dtype=float)
    phi = summary["phi_eid_mean"].to_numpy(dtype=float)
    phi_std = summary["phi_eid_std"].to_numpy(dtype=float)
    whole = summary["whole_ei_mean"].to_numpy(dtype=float)
    whole_std = summary["whole_ei_std"].to_numpy(dtype=float)
    singleton = summary["singleton_ei_sum_mean"].to_numpy(dtype=float)
    singleton_std = summary["singleton_ei_sum_std"].to_numpy(dtype=float)
    ax_phi.plot(x, phi, color="#4C78A8", marker="o", markersize=2.3, linewidth=1.6, label=r"$\Xi$")
    ax_phi.fill_between(x, phi - phi_std, phi + phi_std, color="#4C78A8", alpha=0.16, linewidth=0)
    ax_phi.axhline(0.0, color="#888888", linewidth=0.7, linestyle=":")
    ax_phi.set_ylabel(r"$\Xi$ (bits)")
    ax_phi.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax_ei.plot(x, whole, color="#777777", linewidth=1.25, linestyle="--", label="Whole EI")
    ax_ei.fill_between(x, whole - whole_std, whole + whole_std, color="#777777", alpha=0.08, linewidth=0)
    ax_ei.plot(x, singleton, color="#B279A2", linewidth=1.25, linestyle=":", label="Singleton EI sum")
    ax_ei.fill_between(x, singleton - singleton_std, singleton + singleton_std, color="#B279A2", alpha=0.08, linewidth=0)
    ax_ei.set_xlabel("Lead (months)")
    ax_ei.set_ylabel("EI terms (bits)")
    ax_ei.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
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
        clip_negative=False,
    )
    rows, singleton_rows = compute_phi_eid_rows(
        history_modes,
        targets_by_seed,
        mode_names=MODE_NAMES,
        leads=leads,
        estimator=estimator,
    )
    summary = summarize_phi_eid_leads(rows)

    paths = {
        "rows": output_dir / "all_mode_target_phi_eid_rows.csv",
        "singleton_rows": output_dir / "all_mode_target_phi_eid_singleton_rows.csv",
        "lead_summary": output_dir / "all_mode_target_phi_eid_lead_summary.csv",
    }
    rows.to_csv(paths["rows"], index=False)
    singleton_rows.to_csv(paths["singleton_rows"], index=False)
    summary.to_csv(paths["lead_summary"], index=False)
    figures = plot_phi_eid_leads(summary, Path(args.asset_base))
    manifest = {
        "target_definition": "all 11 predicted UniCM modes at each lead as a multivariate target",
        "source_definition": "singleton partition over 11 mode histories; each source is one mode's 12-month history",
        "phi_eid_definition": "I(all mode histories; all-mode target) - sum_i I(mode_i history; all-mode target)",
        "signed_outputs": True,
        "seeds": seeds,
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
    manifest_path = output_dir / "all_mode_target_phi_eid_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute UniCM all-mode target Xi lead curves.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--asset-base", type=Path, default=DEFAULT_ASSET_BASE)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--leads", nargs="*", default=None)
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument("--sampling-seed", type=int, default=20260619)
    parser.add_argument("--intervention-bound", type=float, default=4.0)
    parser.add_argument("--start-month", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--estimator", choices=["gaussian_logdet", "transport_map"], default="gaussian_logdet")
    parser.add_argument("--tm-degree", type=int, default=3)
    parser.add_argument("--tm-jitter", type=float, default=1.0e-6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    outputs = run(build_arg_parser().parse_args(argv))
    print(json.dumps(outputs, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
