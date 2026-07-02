from __future__ import annotations

import argparse
import itertools
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
    summarize_full_history_mode_pair_syn,
)

DEFAULT_CACHE_DIR = ROOT / "results" / "unicm_overall_ei_cpu_bound4_n8192" / "cache"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "unicm_delta_target_ei_syn_cpu_bound4_n8192"
DEFAULT_ASSET_DIR = ROOT / "docs" / "reports" / "assets"
DEFAULT_TARGETS = ("ENSO", "IOD")
TARGET_ALIASES = {"ENSO": "nino"}

Estimator = Callable[[np.ndarray, np.ndarray], float]
PairEstimator = Callable[[np.ndarray, str, str, np.ndarray], Mapping[str, float]]


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


def resolve_mode_name(name: str, mode_names: Mapping[str, int]) -> str:
    text = str(name)
    if text in mode_names:
        return text
    canonical = TARGET_ALIASES.get(text, text)
    if canonical not in mode_names:
        raise ValueError(f"Unknown mode: {name}")
    return canonical


def delta_target_for_mode(
    history_modes: np.ndarray,
    predictions: np.ndarray,
    *,
    target_index: int,
    lead: int,
) -> np.ndarray:
    history = np.asarray(history_modes, dtype=float)
    future = np.asarray(predictions, dtype=float)
    if history.ndim != 3 or future.ndim != 3:
        raise ValueError("history_modes and predictions must both be 3D arrays.")
    if future.shape[0] != history.shape[0] or future.shape[2] != history.shape[2]:
        raise ValueError("Prediction cache shape is incompatible with history_modes.")
    current = history[:, -1, int(target_index)]
    target = future[:, int(lead) - 1, int(target_index)] - current
    return target.reshape(-1, 1)


def compute_delta_single_source_ei_rows(
    history_modes: np.ndarray,
    targets_by_seed: Mapping[int, np.ndarray],
    *,
    targets: Sequence[str] = DEFAULT_TARGETS,
    mode_names: Mapping[str, int] = MODE_NAMES,
    leads: Sequence[int] | None = None,
    estimator: Estimator = estimate_gaussian_mutual_information,
) -> pd.DataFrame:
    history = np.asarray(history_modes, dtype=float)
    lead_values = list(range(1, 25)) if leads is None else [int(lead) for lead in leads]
    rows: list[dict[str, object]] = []
    for seed in sorted(int(seed) for seed in targets_by_seed):
        predictions = np.asarray(targets_by_seed[int(seed)], dtype=float)
        for target in targets:
            target_mode = resolve_mode_name(str(target), mode_names)
            target_index = int(mode_names[target_mode])
            for lead in lead_values:
                delta_target = delta_target_for_mode(history, predictions, target_index=target_index, lead=int(lead))
                for source_name, source_index in mode_names.items():
                    source = history[:, :, int(source_index)]
                    rows.append(
                        {
                            "seed": int(seed),
                            "target": str(target),
                            "target_mode": target_mode,
                            "source": str(source_name),
                            "lead": int(lead),
                            "single_ei": float(estimator(source, delta_target)),
                        }
                    )
    return pd.DataFrame(rows).sort_values(["target", "source", "seed", "lead"]).reset_index(drop=True)


def _mode_pairs(mode_names: Mapping[str, int]) -> list[tuple[str, str]]:
    names = list(mode_names)
    return [(left, right) for index, left in enumerate(names) for right in names[index + 1 :]]


def compute_delta_pair_syn_rows(
    history_modes: np.ndarray,
    targets_by_seed: Mapping[int, np.ndarray],
    *,
    targets: Sequence[str] = DEFAULT_TARGETS,
    mode_names: Mapping[str, int] = MODE_NAMES,
    leads: Sequence[int] | None = None,
    pair_estimator: PairEstimator = summarize_full_history_mode_pair_syn,
) -> pd.DataFrame:
    history = np.asarray(history_modes, dtype=float)
    lead_values = list(range(1, 25)) if leads is None else [int(lead) for lead in leads]
    rows: list[dict[str, object]] = []
    for seed in sorted(int(seed) for seed in targets_by_seed):
        predictions = np.asarray(targets_by_seed[int(seed)], dtype=float)
        for target in targets:
            target_mode = resolve_mode_name(str(target), mode_names)
            target_index = int(mode_names[target_mode])
            for lead in lead_values:
                delta_target = delta_target_for_mode(history, predictions, target_index=target_index, lead=int(lead))
                for left_name, right_name in _mode_pairs(mode_names):
                    summary = pair_estimator(history, left_name, right_name, delta_target)
                    rows.append(
                        {
                            "seed": int(seed),
                            "target": str(target),
                            "target_mode": target_mode,
                            "pair": f"{left_name}|{right_name}",
                            "left_source": left_name,
                            "right_source": right_name,
                            "lead": int(lead),
                            **{key: float(value) for key, value in summary.items() if key != "backend"},
                        }
                    )
    return pd.DataFrame(rows).sort_values(["target", "pair", "seed", "lead"]).reset_index(drop=True)


def summarize_single_source(rows: pd.DataFrame) -> pd.DataFrame:
    summary = rows.groupby(["target", "target_mode", "source", "lead"], as_index=False)["single_ei"].agg(
        ["mean", "std"]
    )
    summary = summary.reset_index()
    summary["std"] = summary["std"].fillna(0.0)
    summary["mean_over_leads"] = summary.groupby(["target", "source"])["mean"].transform("mean")
    return summary.sort_values(["target", "mean_over_leads", "source", "lead"], ascending=[True, False, True, True])


def summarize_pair_syn(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lead = rows.groupby(
        ["target", "target_mode", "pair", "left_source", "right_source", "lead"],
        as_index=False,
    )["syn"].agg(["mean", "std"])
    lead = lead.reset_index()
    lead["std"] = lead["std"].fillna(0.0)
    pair = lead.groupby(["target", "target_mode", "pair", "left_source", "right_source"], as_index=False)[
        "mean"
    ].mean()
    pair = pair.rename(columns={"mean": "mean_syn"}).sort_values(["target", "mean_syn", "pair"], ascending=[True, False, True])
    pair["rank_within_target"] = pair.groupby("target")["mean_syn"].rank(method="first", ascending=False).astype(int)
    return pair.reset_index(drop=True), lead.sort_values(["target", "pair", "lead"]).reset_index(drop=True)


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


def display_label(name: object) -> str:
    text = str(name)
    return "ENSO" if text == "nino" else text


def display_pair(pair: object) -> str:
    return " + ".join(display_label(part) for part in str(pair).split("|"))


def _save(fig: plt.Figure, output_base: Path) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix(".png")
    svg = output_base.with_suffix(".svg")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, svg]


def plot_delta_single_source(summary: pd.DataFrame, output_base: Path, targets: Sequence[str]) -> list[Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(
        2,
        len(targets),
        figsize=(4.4 * len(targets), 5.2),
        constrained_layout=True,
        sharex=True,
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(2, 1)
    palette = plt.get_cmap("tab20").colors
    source_order = list(MODE_NAMES)
    legend_by_label: dict[str, object] = {}
    for col, target in enumerate(targets):
        target_rows = summary[summary["target"].astype(str) == str(target)]
        if target_rows.empty:
            continue
        target_mode = str(target_rows["target_mode"].iloc[0])
        self_axis = axes[0, col]
        nonself_axis = axes[1, col]
        for index, source in enumerate(source_order):
            rows = target_rows[target_rows["source"].astype(str) == source].sort_values("lead")
            if rows.empty:
                continue
            x = rows["lead"].to_numpy(dtype=float)
            mean = rows["mean"].to_numpy(dtype=float)
            std = rows["std"].to_numpy(dtype=float)
            color = palette[index % len(palette)]
            axis = self_axis if source == target_mode else nonself_axis
            axis.plot(x, mean, marker="o", markersize=2.0, linewidth=1.15, color=color, label=display_label(source))
            axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.08, linewidth=0)
        self_axis.axhline(0, color="#888888", linewidth=0.7, linestyle=":")
        nonself_axis.axhline(0, color="#888888", linewidth=0.7, linestyle=":")
        self_axis.set_title(f"{target} - Self source", fontsize=8)
        nonself_axis.set_title(f"{target} - Non-self sources", fontsize=8)
        nonself_axis.set_xlabel("Lead (months)")
        self_axis.set_ylabel("Self EI to delta target (bits)")
        nonself_axis.set_ylabel("Non-self EI to delta target (bits)")
        handles, labels = nonself_axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_by_label.setdefault(str(label), handle)
    if legend_by_label:
        fig.legend(
            list(legend_by_label.values()),
            list(legend_by_label.keys()),
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
        )
    return _save(fig, output_base)


def plot_delta_pair_syn(
    pair_summary: pd.DataFrame,
    lead_summary: pd.DataFrame,
    output_base: Path,
    *,
    target: str,
    top_k: int = 12,
    required_pairs: Sequence[str] = ("NPMM|TNA", "nino|NPMM", "nino|TNA", "NPMM|nino3", "TNA|nino3", "IOD|SIOD", "nino|IOD"),
) -> list[Path]:
    configure_matplotlib()
    target_pairs = pair_summary[pair_summary["target"].astype(str) == str(target)].sort_values("rank_within_target")
    selected = target_pairs.head(int(top_k))["pair"].astype(str).tolist()
    available = set(target_pairs["pair"].astype(str))
    for pair in required_pairs:
        if pair in available and pair not in selected:
            selected.append(pair)

    fig, ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
    palette = itertools.cycle(plt.get_cmap("tab20").colors)
    for pair in selected:
        rows = lead_summary[
            (lead_summary["target"].astype(str) == str(target)) & (lead_summary["pair"].astype(str) == pair)
        ].sort_values("lead")
        if rows.empty:
            continue
        color = next(palette)
        x = rows["lead"].to_numpy(dtype=float)
        mean = rows["mean"].to_numpy(dtype=float)
        ax.plot(x, mean, marker="o", markersize=2.0, linewidth=1.15, color=color, label=display_pair(pair))
        ax.axhline(float(np.mean(mean)), color=color, linewidth=0.8, linestyle=":", alpha=0.8)
    ax.axhline(0, color="#777777", linewidth=0.7, linestyle="-")
    ax.set_title(str(target), fontsize=8)
    ax.set_xlabel("Lead (months)")
    ax.set_ylabel("Seed mean Syn to delta target (bits)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return _save(fig, output_base)


def _cache_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        n_samples=int(args.n_samples),
        sampling_seed=int(args.sampling_seed),
        intervention_bound=float(args.intervention_bound),
        start_month=int(args.start_month),
        device=str(args.device),
    )


def _asset_slug(target: str) -> str:
    return "enso" if str(target) == "ENSO" else str(target).lower()


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    asset_dir = Path(args.asset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    leads = parse_leads(args.leads)
    seeds = [int(seed) for seed in args.seeds]
    targets = tuple(str(target) for target in args.targets)
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

    single_rows = compute_delta_single_source_ei_rows(history_modes, targets_by_seed, targets=targets, leads=leads)
    pair_rows = compute_delta_pair_syn_rows(history_modes, targets_by_seed, targets=targets, leads=leads)
    single_summary = summarize_single_source(single_rows)
    pair_summary, pair_lead_summary = summarize_pair_syn(pair_rows)

    paths = {
        "single_rows": output_dir / "delta_single_source_ei_rows.csv",
        "single_summary": output_dir / "delta_single_source_ei_lead_summary.csv",
        "pair_rows": output_dir / "delta_mode_pair_syn_rows.csv",
        "pair_summary": output_dir / "delta_mode_pair_syn_summary.csv",
        "pair_lead_summary": output_dir / "delta_mode_pair_syn_lead_summary.csv",
    }
    single_rows.to_csv(paths["single_rows"], index=False)
    single_summary.to_csv(paths["single_summary"], index=False)
    pair_rows.to_csv(paths["pair_rows"], index=False)
    pair_summary.to_csv(paths["pair_summary"], index=False)
    pair_lead_summary.to_csv(paths["pair_lead_summary"], index=False)

    figure_paths = plot_delta_single_source(
        single_summary,
        asset_dir / "unicm_delta_target_single_source_ei_enso_iod",
        targets,
    )
    for target in targets:
        figure_paths.extend(
            plot_delta_pair_syn(
                pair_summary,
                pair_lead_summary,
                asset_dir / f"unicm_delta_target_mode_pair_syn_{_asset_slug(target)}",
                target=target,
                top_k=int(args.top_k),
            )
        )

    manifest = {
        "target_definition": "delta = predicted future target state - current target state at history month 12",
        "source_definition": "each source mode contributes its own 12-month history",
        "targets": list(targets),
        "leads": leads,
        "seeds": seeds,
        "n_samples": int(args.n_samples),
        "sampling_seed": int(args.sampling_seed),
        "intervention_bound": float(args.intervention_bound),
        "cache_dir": str(args.cache_dir),
        "tables": {key: str(path) for key, path in paths.items()},
        "figures": [str(path) for path in figure_paths],
        "n_single_rows": int(len(single_rows)),
        "n_pair_rows": int(len(pair_rows)),
    }
    manifest_path = output_dir / "delta_target_ei_syn_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute UniCM EI/Syn curves for delta targets.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--leads", nargs="*", default=None)
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument("--sampling-seed", type=int, default=20260619)
    parser.add_argument("--intervention-bound", type=float, default=4.0)
    parser.add_argument("--start-month", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    outputs = run(build_arg_parser().parse_args(argv))
    print(json.dumps(outputs, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
