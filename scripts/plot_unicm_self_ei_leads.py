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

from scripts.unicm_peid_syn_analysis import (  # noqa: E402
    MODE_NAMES,
    estimate_gaussian_mutual_information,
    load_full_history_prediction_cache,
    overall_prediction_cache_path,
    sample_full_history_mode_inputs,
)

DEFAULT_CACHE_DIR = ROOT / "results" / "unicm_overall_ei_cpu_bound4_n8192" / "cache"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "unicm_self_ei_cpu_bound4_n8192"
DEFAULT_ASSET_BASE = ROOT / "docs" / "reports" / "assets" / "unicm_all_modes_self_ei_leads"

Estimator = Callable[[np.ndarray, np.ndarray], float]


def parse_leads(values: Sequence[str] | None) -> list[int]:
    if not values:
        return list(range(1, 25))
    leads: list[int] = []
    for raw in values:
        text = str(raw)
        if ".." in text:
            start, end = text.split("..", 1)
            leads.extend(range(int(start), int(end) + 1))
        else:
            leads.append(int(text))
    unique = sorted(set(leads))
    invalid = [lead for lead in unique if lead < 1 or lead > 24]
    if invalid:
        raise ValueError(f"Lead must be in [1, 24], got {invalid[0]}.")
    return unique


def compute_self_ei_rows(
    history_modes: np.ndarray,
    targets_by_seed: Mapping[int, np.ndarray],
    *,
    mode_names: Mapping[str, int] = MODE_NAMES,
    leads: Sequence[int] | None = None,
    estimator: Estimator = estimate_gaussian_mutual_information,
) -> pd.DataFrame:
    history = np.asarray(history_modes, dtype=float)
    if history.ndim != 3:
        raise ValueError("history_modes must have shape (n_samples, history_length, n_modes).")
    lead_values = parse_leads([str(lead) for lead in leads]) if leads is not None else list(range(1, 25))
    rows: list[dict[str, object]] = []
    for seed in sorted(int(seed) for seed in targets_by_seed):
        targets = np.asarray(targets_by_seed[int(seed)], dtype=float)
        if targets.ndim != 3:
            raise ValueError("Each prediction cache must have shape (n_samples, prediction_length, n_modes).")
        if targets.shape[0] != history.shape[0] or targets.shape[2] != history.shape[2]:
            raise ValueError("Prediction cache shape is incompatible with history_modes.")
        for mode_name, mode_index in mode_names.items():
            source = history[:, :, int(mode_index)]
            for lead in lead_values:
                target = targets[:, [int(lead) - 1], int(mode_index)]
                rows.append(
                    {
                        "seed": int(seed),
                        "target": str(mode_name),
                        "source": str(mode_name),
                        "lead": int(lead),
                        "self_ei": float(estimator(source, target)),
                    }
                )
    return pd.DataFrame(rows).sort_values(["target", "seed", "lead"]).reset_index(drop=True)


def summarize_self_ei_leads(rows: pd.DataFrame) -> pd.DataFrame:
    summary = rows.groupby(["target", "source", "lead"], as_index=False)["self_ei"].agg(["mean", "std"]).reset_index()
    summary["std"] = summary["std"].fillna(0.0)
    summary["cv"] = summary["std"] / summary["mean"].abs().clip(lower=1e-12)
    return summary.sort_values(["target", "lead"]).reset_index(drop=True)


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


def plot_self_ei_leads(summary: pd.DataFrame, output_base: Path) -> list[Path]:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
    palette = plt.get_cmap("tab20").colors
    for index, target in enumerate(MODE_NAMES):
        subset = summary[summary["target"].astype(str) == target].sort_values("lead")
        if subset.empty:
            continue
        x = subset["lead"].to_numpy(dtype=float)
        mean = subset["mean"].to_numpy(dtype=float)
        std = subset["std"].to_numpy(dtype=float)
        color = palette[index % len(palette)]
        ax.plot(x, mean, marker="o", markersize=2.0, linewidth=1.25, color=color, label=target)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10, linewidth=0)
    ax.axhline(0, color="#888888", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Lead (months)")
    ax.set_ylabel("Seed mean self-EI (bits)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
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
    rows = compute_self_ei_rows(history_modes, targets_by_seed, leads=leads)
    summary = summarize_self_ei_leads(rows)

    rows_path = output_dir / "self_ei_rows.csv"
    summary_path = output_dir / "self_ei_lead_summary.csv"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    figure_paths = plot_self_ei_leads(summary, Path(args.asset_base))

    manifest = {
        "rows": str(rows_path),
        "lead_summary": str(summary_path),
        "figures": [str(path) for path in figure_paths],
        "n_rows": int(len(rows)),
        "n_modes": int(len(MODE_NAMES)),
        "n_seeds": int(len(seeds)),
        "leads": leads,
        "source_definition": "same target mode 12-month history",
        "n_samples": int(args.n_samples),
        "sampling_seed": int(args.sampling_seed),
        "intervention_bound": float(args.intervention_bound),
        "cache_dir": str(args.cache_dir),
    }
    manifest_path = output_dir / "self_ei_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot UniCM self-EI lead curves for all modes.")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    outputs = run(build_arg_parser().parse_args(argv))
    print(json.dumps(outputs, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
