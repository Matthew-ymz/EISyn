"""Run the continuous hierarchical multistable control example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.network_revival.hierarchical_multistable_control import (  # noqa: E402
    HierarchicalControlConfig,
    plot_hierarchical_control_result,
    run_hierarchical_control_example,
    write_example_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "hierarchical_multistable_control",
    )
    parser.add_argument(
        "--figure-base",
        type=Path,
        default=ROOT / "fig" / "part3_hierarchical_multistable_control",
    )
    parser.add_argument("--sample-count", type=int, default=768)
    parser.add_argument("--control-samples", type=int, default=24)
    parser.add_argument("--tm-degree", type=int, default=3)
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    max_order = min(int(args.max_order), 4) if args.smoke else int(args.max_order)
    config = HierarchicalControlConfig(dt=0.08 if args.smoke else 0.04)
    result = run_hierarchical_control_example(
        config,
        sample_count=int(args.sample_count),
        control_samples=int(args.control_samples),
        tm_degree=int(args.tm_degree),
        max_order=max_order,
        seed=int(args.seed),
    )
    write_example_results(result, args.output_dir)
    figure_paths = plot_hierarchical_control_result(result, args.figure_base)
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "output_dir": str(args.output_dir),
                "figures": [str(path) for path in figure_paths],
            },
            ensure_ascii=False,
            default=list,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

