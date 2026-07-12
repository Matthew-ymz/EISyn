"""Run the N=20 natural-network pre-control Phi--cost experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.network_revival.network_basin_phi_cost import (  # noqa: E402
    NaturalNetworkPhiCostConfig,
    plot_natural_network_phi_cost,
    run_natural_network_phi_cost_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "network_basin_phi_cost")
    parser.add_argument("--figure-base", type=Path, default=ROOT / "fig" / "part3_natural_network_phi_cost")
    parser.add_argument("--instances-per-group", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        config = NaturalNetworkPhiCostConfig(
            output_dir=args.output_dir,
            model_names=("Neural",), network_kinds=("ER",), instances_per_group=1,
            candidate_seed_count=20, candidate_counts_by_order=((2, 190), (3, 8), (4, 8)),
            precontrol_sample_count=64, binary_steps=5, permutations=99, bootstrap_reps=100,
            seed=args.seed,
        )
    else:
        config = NaturalNetworkPhiCostConfig(output_dir=args.output_dir, instances_per_group=args.instances_per_group, seed=args.seed)
    result = run_natural_network_phi_cost_experiment(config, force=args.force)
    paths = plot_natural_network_phi_cost(result, args.figure_base)
    print(json.dumps({"summary": result["summary"], "figures": [str(path) for path in paths]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
