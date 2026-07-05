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
)

DEFAULT_CACHE_DIR = ROOT / "results" / "unicm_overall_ei_cpu_bound4_n8192" / "cache"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "unicm_all_mode_target_pair_syn_cpu_bound4_n8192"
DEFAULT_ASSET_BASE = ROOT / "fig" / "unicm_all_mode_target_mode_pair_syn_leads"

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


def mode_pairs(mode_names: Mapping[str, int] = MODE_NAMES) -> list[tuple[str, str]]:
    names = list(mode_names)
    return [(left, right) for index, left in enumerate(names) for right in names[index + 1 :]]


def extract_all_mode_target(predictions: np.ndarray, *, lead: int) -> np.ndarray:
    array = np.asarray(predictions, dtype=float)
    if array.ndim != 3:
        raise ValueError("predictions must have shape (n_samples, n_leads, n_modes).")
    if int(lead) < 1 or int(lead) > array.shape[1]:
        raise ValueError(f"lead must be in [1, {array.shape[1]}], got {lead}.")
    return array[:, int(lead) - 1, :]


def summarize_multivariate_mode_pair_syn(
    history_modes: np.ndarray,
    left_name: str,
    right_name: str,
    target: np.ndarray,
    *,
    mode_names: Mapping[str, int] = MODE_NAMES,
    estimator: Estimator = estimate_gaussian_mutual_information,
) -> dict[str, float]:
    history = np.asarray(history_modes, dtype=float)
    if history.ndim != 3:
        raise ValueError("history_modes must have shape (n_samples, history_length, n_modes).")
    if left_name not in mode_names or right_name not in mode_names:
        raise ValueError("Unknown source mode.")
    target_array = np.asarray(target, dtype=float)
    if target_array.ndim == 1:
        target_array = target_array.reshape(-1, 1)
    if target_array.ndim != 2:
        raise ValueError("target must be one-dimensional or two-dimensional.")
    if target_array.shape[0] != history.shape[0]:
        raise ValueError("history_modes and target must share the sample axis.")

    left = history[:, :, int(mode_names[left_name])]
    right = history[:, :, int(mode_names[right_name])]
    joint = np.concatenate([left, right], axis=1)
    left_ei = float(estimator(left, target_array))
    right_ei = float(estimator(right, target_array))
    joint_ei = float(estimator(joint, target_array))
    return {
        "left_ei": left_ei,
        "right_ei": right_ei,
        "joint_ei": joint_ei,
        "syn": joint_ei - left_ei - right_ei,
    }


def compute_all_mode_target_pair_syn_rows(
    history_modes: np.ndarray,
    targets_by_seed: Mapping[int, np.ndarray],
    *,
    mode_names: Mapping[str, int] = MODE_NAMES,
    leads: Sequence[int] | None = None,
    estimator: Estimator = estimate_gaussian_mutual_information,
) -> pd.DataFrame:
    lead_values = list(range(1, 25)) if leads is None else [int(lead) for lead in leads]
    rows: list[dict[str, object]] = []
    pairs = mode_pairs(mode_names)
    for seed in sorted(int(seed) for seed in targets_by_seed):
        predictions = np.asarray(targets_by_seed[int(seed)], dtype=float)
        for lead in lead_values:
            target = extract_all_mode_target(predictions, lead=int(lead))
            for left_name, right_name in pairs:
                summary = summarize_multivariate_mode_pair_syn(
                    history_modes,
                    left_name,
                    right_name,
                    target,
                    mode_names=mode_names,
                    estimator=estimator,
                )
                rows.append(
                    {
                        "seed": int(seed),
                        "target": "all_modes",
                        "target_dim": int(target.shape[1]),
                        "pair": f"{left_name}|{right_name}",
                        "left_source": left_name,
                        "right_source": right_name,
                        "lead": int(lead),
                        **summary,
                    }
                )
    return pd.DataFrame(rows).sort_values(["pair", "seed", "lead"]).reset_index(drop=True)


def summarize_pair_syn(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lead = rows.groupby(["pair", "left_source", "right_source", "lead"], as_index=False)["syn"].agg(["mean", "std"])
    lead = lead.reset_index()
    lead["std"] = lead["std"].fillna(0.0)
    pair = (
        rows.groupby(["pair", "left_source", "right_source"], as_index=False)
        .agg(
            mean_left_ei=("left_ei", "mean"),
            mean_right_ei=("right_ei", "mean"),
            mean_joint_ei=("joint_ei", "mean"),
            mean_syn=("syn", "mean"),
            std_syn=("syn", "std"),
            min_syn=("syn", "min"),
            max_syn=("syn", "max"),
        )
        .sort_values(["mean_syn", "pair"], ascending=[False, True])
        .reset_index(drop=True)
    )
    pair["std_syn"] = pair["std_syn"].fillna(0.0)
    pair["rank"] = np.arange(1, len(pair) + 1)
    seed_rank = (
        rows.groupby(["seed", "pair"], as_index=False)["syn"]
        .mean()
        .rename(columns={"syn": "seed_mean_syn"})
        .sort_values(["seed", "seed_mean_syn", "pair"], ascending=[True, False, True])
    )
    seed_rank["seed_rank"] = seed_rank.groupby("seed")["seed_mean_syn"].rank(method="first", ascending=False).astype(int)
    seed_stats = seed_rank.groupby("pair", as_index=False).agg(
        seed_mean_syn=("seed_mean_syn", "mean"),
        seed_sd_syn=("seed_mean_syn", "std"),
        positive_seed_count=("seed_mean_syn", lambda values: int((values > 0.0).sum())),
        seed_rank_min=("seed_rank", "min"),
        seed_rank_max=("seed_rank", "max"),
    )
    seed_stats["seed_sd_syn"] = seed_stats["seed_sd_syn"].fillna(0.0)
    pair = pair.merge(seed_stats, on="pair", how="left", validate="one_to_one")
    return pair, lead.sort_values(["pair", "lead"]).reset_index(drop=True)


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


def plot_all_mode_pair_syn(pair_summary: pd.DataFrame, lead_summary: pd.DataFrame, output_base: Path, *, top_k: int) -> list[Path]:
    configure_matplotlib()
    selected = pair_summary.sort_values(["mean_syn", "pair"], ascending=[False, True]).head(int(top_k))
    selected_pairs = selected["pair"].astype(str).tolist()
    selected_set = set(selected_pairs)
    palette = itertools.cycle(
        [
            "#4C78A8",
            "#F58518",
            "#54A24B",
            "#B279A2",
            "#72B7B2",
            "#E45756",
            "#79706E",
            "#9D755D",
            "#A0CBE8",
            "#FFBE7D",
            "#8CD17D",
            "#D4A6C8",
        ]
    )
    color_by_pair = {pair: next(palette) for pair in selected_pairs}
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    for pair, rows in lead_summary.groupby("pair", sort=False):
        if str(pair) in selected_set:
            continue
        rows = rows.sort_values("lead")
        ax.plot(rows["lead"], rows["mean"], color="#B8B8B8", linewidth=0.45, alpha=0.35, zorder=1)
    for pair in selected_pairs:
        rows = lead_summary[lead_summary["pair"].astype(str) == pair].sort_values("lead")
        summary_row = pair_summary[pair_summary["pair"].astype(str) == pair].iloc[0]
        x = rows["lead"].to_numpy(dtype=float)
        mean = rows["mean"].to_numpy(dtype=float)
        std = rows["std"].to_numpy(dtype=float)
        color = color_by_pair[pair]
        ax.plot(x, mean, marker="o", markersize=2.0, linewidth=1.25, color=color, label=display_pair(pair), zorder=3)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.08, linewidth=0, zorder=2)
        ax.axhline(float(summary_row["mean_syn"]), color=color, linewidth=0.55, linestyle=":", alpha=0.45, zorder=1)
    ax.axhline(0.0, color="#777777", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Lead (months)")
    ax.set_ylabel("Syn to all-mode target (bits)")
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
    rows = compute_all_mode_target_pair_syn_rows(history_modes, targets_by_seed, leads=leads)
    pair_summary, lead_summary = summarize_pair_syn(rows)

    paths = {
        "rows": output_dir / "all_mode_target_pair_syn_rows.csv",
        "summary": output_dir / "all_mode_target_pair_syn_summary.csv",
        "lead_summary": output_dir / "all_mode_target_pair_syn_lead_summary.csv",
    }
    rows.to_csv(paths["rows"], index=False)
    pair_summary.to_csv(paths["summary"], index=False)
    lead_summary.to_csv(paths["lead_summary"], index=False)
    figures = plot_all_mode_pair_syn(pair_summary, lead_summary, Path(args.asset_base), top_k=int(args.top_k))
    manifest = {
        "target_definition": "all 11 predicted UniCM modes at each lead as a multivariate target",
        "source_definition": "each source mode contributes its own 12-month history",
        "seeds": seeds,
        "leads": leads,
        "n_samples": int(args.n_samples),
        "sampling_seed": int(args.sampling_seed),
        "intervention_bound": float(args.intervention_bound),
        "target_dim": int(len(MODE_NAMES)),
        "n_pairs": int(len(pair_summary)),
        "tables": {key: str(path) for key, path in paths.items()},
        "figures": [str(path) for path in figures],
        "cache_dir": str(args.cache_dir),
    }
    manifest_path = output_dir / "all_mode_target_pair_syn_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute UniCM source-pair Syn curves for all-mode multivariate targets.")
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
    parser.add_argument("--top-k", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    outputs = run(build_arg_parser().parse_args(argv))
    print(json.dumps(outputs, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
