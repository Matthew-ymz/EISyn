#!/usr/bin/env python3
"""Force TM estimates for selected Runge hyperedges across horizons."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_runge_gateway_mediator_map import PAPER_TO_LOCAL, local_to_paper
from run_runge_multistep_conditioned_ei import (
    DEFAULT_PAIRWISE_MANIFEST,
    config_from_manifest,
    estimate_mi,
    load_cached_pairwise_model,
    rollout_mlp_closed_loop,
    source_state_matrix,
)
from run_runge_pairwise_mlp_ei import sample_max_entropy_features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TM_DIR = (
    ROOT
    / "results"
    / "runge_slp_daily_1948_2026_20260628"
    / "mlp_tm_ei_lag04"
    / "results"
    / "runge"
    / "multistep_conditioned_ei_tm_forced_edges"
)
DEFAULT_FIG_DIR = ROOT / "fig" / "runge_slp_daily_1948_2026_20260628" / "multistep_conditioned_ei_tm_targeted"
DEFAULT_HORIZONS = "1-10,15,20,30,40,50,60"
DEFAULT_EDGES = "0+6->32,0+1->28,0+1->50,0+1->46"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def parse_horizons(text: str) -> list[int]:
    values: list[int] = []
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            values.extend(range(int(left), int(right) + 1))
        else:
            values.append(int(item))
    if not values:
        raise ValueError("At least one horizon is required.")
    return sorted(dict.fromkeys(values))


def parse_edge(text: str) -> tuple[int, int, int]:
    sources, target = str(text).split("->", 1)
    left, right = sources.split("+", 1)
    return int(left), int(right), int(target)


def paper_to_local(index: int) -> int:
    return int(PAPER_TO_LOCAL.get(int(index), int(index)))


def normalized_edge_label(source_a_local: int, source_b_local: int, target_local: int) -> str:
    left = local_to_paper(int(source_a_local))
    right = local_to_paper(int(source_b_local))
    target = local_to_paper(int(target_local))
    return f"{left}+{right}->{target}"


def build_rollout_inputs(args: argparse.Namespace, max_horizon: int) -> tuple[list[np.ndarray], np.ndarray, list[str], dict[str, object]]:
    config = config_from_manifest(Path(args.pairwise_manifest).expanduser())
    config = replace(
        config,
        intervention_samples=int(args.intervention_samples),
        ei_estimator="tm",
        source_mode=str(args.source_mode),
        force_retrain=False,
    )
    model, scalers, splits, names, model_info = load_cached_pairwise_model(config)
    n_components = len(names)
    features = sample_max_entropy_features(
        splits["train"][0],
        n_components=n_components,
        lag=int(config.lag),
        samples=int(args.intervention_samples),
        low_q=float(config.quantile_low),
        high_q=float(config.quantile_high),
        seed=int(config.seed),
    )
    source_states = source_state_matrix(
        features,
        n_components=n_components,
        lag=int(config.lag),
        source_mode=str(args.source_mode),
    )
    result_dir = Path(args.result_dir).expanduser()
    result_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = result_dir / f"rollout_predictions_H{int(max_horizon):03d}_n{int(args.intervention_samples)}.npy"
    if rollout_path.exists() and bool(args.resume):
        predictions = np.load(rollout_path)
    else:
        predictions = rollout_mlp_closed_loop(
            model,
            scalers,
            features,
            n_components=n_components,
            lag=int(config.lag),
            horizons=int(max_horizon),
        )
        np.save(rollout_path, predictions)
    manifest = {
        "pairwise_manifest": str(Path(args.pairwise_manifest).expanduser()),
        "intervention_samples": int(args.intervention_samples),
        "source_mode": str(args.source_mode),
        "max_horizon": int(max_horizon),
        **model_info,
    }
    return source_states, predictions, names, manifest


def estimate_forced_edges(args: argparse.Namespace) -> pd.DataFrame:
    horizons = parse_horizons(args.horizons)
    edge_labels = [part.strip() for part in str(args.edges).split(",") if part.strip()]
    edges_paper = [parse_edge(label) for label in edge_labels]
    edges_local = [(paper_to_local(a), paper_to_local(b), paper_to_local(t)) for a, b, t in edges_paper]
    result_dir = Path(args.result_dir).expanduser()
    result_dir.mkdir(parents=True, exist_ok=True)
    source_states, predictions, names, manifest = build_rollout_inputs(args, max(horizons))
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        target_values = predictions[:, int(horizon) - 1, :]
        for source_a, source_b, target in edges_local:
            cache_path = result_dir / f"H{horizon:03d}_{normalized_edge_label(source_a, source_b, target).replace('->', '_to_')}.json"
            if cache_path.exists() and bool(args.resume):
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                source_a_state = source_states[int(source_a)]
                source_b_state = source_states[int(source_b)]
                joint_source = np.concatenate([source_a_state, source_b_state], axis=1)
                target_state = target_values[:, [int(target)]]
                ei_a, bias_a = estimate_mi(source_a_state, target_state, estimator="tm", bins=int(args.bins))
                ei_b, bias_b = estimate_mi(source_b_state, target_state, estimator="tm", bins=int(args.bins))
                joint_ei, joint_bias = estimate_mi(joint_source, target_state, estimator="tm", bins=int(args.bins))
                payload = {
                    "horizon": int(horizon),
                    "source_a": int(source_a),
                    "source_b": int(source_b),
                    "target_index": int(target),
                    "source_a_paper": local_to_paper(int(source_a)),
                    "source_b_paper": local_to_paper(int(source_b)),
                    "target_paper": local_to_paper(int(target)),
                    "edge_label_paper": normalized_edge_label(source_a, source_b, target),
                    "ei_a_tm": float(ei_a),
                    "ei_b_tm": float(ei_b),
                    "joint_ei_tm": float(joint_ei),
                    "delta2_tm": float(joint_ei - ei_a - ei_b),
                    "bias_a": bias_a,
                    "bias_b": bias_b,
                    "joint_bias": joint_bias,
                }
                cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            rows.append(payload)
            print(
                f"H={horizon:03d} {payload['edge_label_paper']} "
                f"delta2_tm={float(payload['delta2_tm']):.6f}",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(result_dir / "forced_tm_edge_trends.csv", index=False)
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return frame


def plot_forced_trends(frame: pd.DataFrame, output: Path) -> None:
    horizons = sorted(frame["horizon"].astype(int).unique().tolist())
    fig, ax = plt.subplots(figsize=(5.6, 3.35), constrained_layout=True)
    colors = {
        "0+6->32": "#4c78a8",
        "0+1->28": "#f58518",
        "0+1->50": "#54a24b",
        "0+1->46": "#b279a2",
    }
    for edge, subset in frame.groupby("edge_label_paper", sort=False):
        subset = subset.sort_values("horizon")
        ax.plot(
            subset["horizon"],
            subset["delta2_tm"],
            marker="o",
            linewidth=1.65,
            markersize=3.5,
            color=colors.get(str(edge), "#666666"),
            label=str(edge),
        )
    ax.axhline(0.0, color="#555555", linewidth=0.75)
    ax.set_xlabel("Horizon H")
    ax.set_ylabel(r"$\Delta_{2,\mathrm{TM}}$")
    ax.set_xticks([h for h in [1, 5, 10, 20, 40, 60] if h in set(horizons)])
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.45)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Hyperedge")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=450, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise-manifest", default=str(DEFAULT_PAIRWISE_MANIFEST))
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_TM_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIG_DIR / "forced_tm_edge_trends_H001_H060.png")
    parser.add_argument("--horizons", default=DEFAULT_HORIZONS)
    parser.add_argument("--edges", default=DEFAULT_EDGES)
    parser.add_argument("--intervention-samples", type=int, default=4096)
    parser.add_argument("--source-mode", choices=["latest", "history"], default="latest")
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    frame = estimate_forced_edges(args)
    output = Path(args.output).expanduser()
    plot_forced_trends(frame, output)
    frame.to_csv(output.with_suffix(".csv"), index=False)
    print(output)


if __name__ == "__main__":
    main()
