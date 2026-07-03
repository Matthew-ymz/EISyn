#!/usr/bin/env python3
"""Recompute Runge ACE/ACS from existing weekly scores using PC-stable parents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from reproduce_runge2015_gateways import apply_paper_component_labels, compute_sem_effects, discover_causal_edges


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-weekly-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-lag", type=int, default=4)
    parser.add_argument("--pc-alpha", type=float, default=0.001)
    parser.add_argument("--link-density", type=float, default=0.2)
    args = parser.parse_args()

    scores = pd.read_csv(args.component_weekly_scores, index_col=0)
    edges = discover_causal_edges(
        scores,
        max_lag=int(args.max_lag),
        pc_alpha=float(args.pc_alpha),
        link_density=float(args.link_density),
        backend="tigramite",
    )
    effects = compute_sem_effects(edges, n_components=scores.shape[1], max_lag=int(args.max_lag))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    edge_rows = [
        {
            "source": edge.source,
            "target": edge.target,
            "lag": edge.lag,
            "coefficient": edge.coefficient,
            "p_value": edge.p_value,
        }
        for edge in edges
    ]
    pd.DataFrame(edge_rows).to_csv(output / "causal_edges.csv", index=False)
    apply_paper_component_labels(effects.gateway_scores).to_csv(output / "gateway_scores.csv", index=False)
    apply_paper_component_labels(effects.mediator_scores).to_csv(output / "mediator_scores.csv", index=False)
    effects.total_effects.to_csv(output / "total_effects.csv", index=False)
    effects.path_effects.to_csv(output / "mediated_path_effects.csv", index=False)

    manifest = {
        "component_weekly_scores": str(args.component_weekly_scores),
        "method": "Runge 2015 SEM ACE/ACS using Tigramite PC-stable parents, sparse OLS, and link-density threshold",
        "fix": "uses run_pc_stable parents for sparse regression instead of thresholding final run_pcmci MCI p_matrix",
        "max_lag": int(args.max_lag),
        "pc_alpha": float(args.pc_alpha),
        "link_density": float(args.link_density),
        "n_components": int(scores.shape[1]),
        "n_weekly_samples": int(scores.shape[0]),
        "n_edges": int(len(edges)),
        "top_gateways": apply_paper_component_labels(effects.gateway_scores).head(10).to_dict("records"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
