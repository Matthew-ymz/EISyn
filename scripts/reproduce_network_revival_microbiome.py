from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.network_revival.microbiome import (
    DEFAULT_ARTICLE_ZIP,
    MicrobiomeParameters,
    run_microbiome_reproduction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the Nature Physics 2021 microbiome point-ignition experiment."
    )
    parser.add_argument("--article-zip", type=Path, default=DEFAULT_ARTICLE_ZIP)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--node-indices", type=str, default=None, help="Comma-separated active-network node indices.")
    parser.add_argument("--dt", type=float, default=MicrobiomeParameters.dt)
    parser.add_argument("--t-max", type=float, default=MicrobiomeParameters.t_max)
    parser.add_argument("--steady-tol", type=float, default=MicrobiomeParameters.steady_tol)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _parse_node_indices(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    params = replace(
        MicrobiomeParameters(),
        dt=float(args.dt),
        t_max=float(args.t_max),
        steady_tol=float(args.steady_tol),
    )
    result = run_microbiome_reproduction(
        article_zip=args.article_zip,
        output_dir=args.output_dir,
        params=params,
        max_nodes=args.max_nodes,
        node_indices=_parse_node_indices(args.node_indices),
        force=bool(args.force),
    )
    metadata = result["metadata"]
    print(
        "microbiome reproduction wrote "
        f"{metadata['evaluated_node_count']} node evaluations to {result['results_dir']}"
    )


if __name__ == "__main__":
    main()
