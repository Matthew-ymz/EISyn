from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yrd.air_search import (
    parse_csv,
    parse_float_csv,
    parse_int_csv,
    run_coarse_stage,
    run_refine_stage,
    run_report_stage,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-city air causal search experiments.")
    parser.add_argument("--stage", choices=("coarse", "refine", "report"), default="coarse")
    parser.add_argument("--cities", default="shanghai,nanjing,hangzhou,beijing")
    parser.add_argument("--horizons", default="1,3,6,12,24")
    parser.add_argument("--smoke", action="store_true", help="Run reduced smoke settings.")
    parser.add_argument("--force-retrain", action="store_true", help="Ignore cached model checkpoints.")
    parser.add_argument(
        "--force-recompute-coupling",
        action="store_true",
        help="Ignore cached coarse/refine coupling summaries.",
    )
    parser.add_argument("--top-k", type=int, default=2, help="Refine the top-k coarse runs.")
    parser.add_argument("--coarse-sample-count", type=int, default=None)
    parser.add_argument("--tm-sample-counts", default="64")
    parser.add_argument("--tm-seeds", default="0")
    parser.add_argument("--tm-gammas", default="1.0,1.1,1.2")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cities = parse_csv(args.cities)
    horizons = parse_int_csv(args.horizons)

    if args.stage == "coarse":
        result = run_coarse_stage(
            root_dir=ROOT,
            cities=cities,
            horizons=horizons,
            smoke=args.smoke,
            force_retrain=args.force_retrain,
            force_recompute_coupling=args.force_recompute_coupling,
            coupling_sample_count=args.coarse_sample_count,
        )
        print(f"coarse rows: {len(result['rows'])}")
        return 0

    if args.stage == "refine":
        result = run_refine_stage(
            root_dir=ROOT,
            cities=cities,
            horizons=horizons,
            smoke=args.smoke,
            top_k=args.top_k,
            force_retrain=args.force_retrain,
            force_recompute_coupling=args.force_recompute_coupling,
            tm_sample_counts=parse_int_csv(args.tm_sample_counts),
            tm_seeds=parse_int_csv(args.tm_seeds),
            tm_gammas=parse_float_csv(args.tm_gammas),
        )
        print(f"refine rows: {len(result['rows'])}")
        return 0

    result = run_report_stage(root_dir=ROOT)
    print(f"report manifests: {len(result['reports'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
