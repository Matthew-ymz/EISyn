from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yrd.analysis import save_json, write_markdown_summary
from yrd.config import YRDExperimentConfig
from yrd.plotting import save_horizon_comparison_plot
from yrd.train import run_smoke_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YRD forecasting and coupling experiments.")
    parser.add_argument("stage", choices=("train", "analyze", "plot", "run-all"), nargs="?")
    parser.add_argument("--smoke", action="store_true", help="Run a reduced smoke pipeline.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cfg = YRDExperimentConfig(root_dir=Path("."))

    if args.stage is None:
        parser.print_help()
        return 0

    if not args.smoke:
        raise SystemExit("Only --smoke execution is implemented in this first implementation slice.")

    result = run_smoke_pipeline(cfg)

    if args.stage in ("analyze", "plot", "run-all"):
        plot_frame = pd.DataFrame(
            {
                "horizon": [1, 24],
                "syn_nis": [
                    result["coupling_summary"]["1"]["syn_nis"],
                    result["coupling_summary"]["24"]["syn_nis"],
                ],
            }
        )
        plot_path = cfg.results_dir / "smoke_horizon_coupling.png"
        plot_note_path = cfg.results_dir / "smoke_horizon_coupling.md"
        save_horizon_comparison_plot(plot_frame, plot_path)
        write_markdown_summary(
            plot_note_path,
            title="YRD Smoke Horizon Coupling Plot Note",
            intro="这一文件解释 `smoke_horizon_coupling.png` 中 1h 与 24h 协同强度对比图的含义。",
            bullets=[
                f"图中 1h 的 `Syn_p^{{nis}}` 为 {result['coupling_summary']['1']['syn_nis']:.4f}。",
                f"图中 24h 的 `Syn_p^{{nis}}` 为 {result['coupling_summary']['24']['syn_nis']:.4f}。",
            ],
        )
        save_json(
            cfg.results_dir / "smoke_run_manifest.json",
            {
                "metrics_markdown": str(result["metrics_md_path"]),
                "coupling_markdown": str(result["coupling_md_path"]),
                "plot": str(plot_path),
                "plot_note": str(plot_note_path),
            },
        )

    print("completed smoke yrd pipeline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
