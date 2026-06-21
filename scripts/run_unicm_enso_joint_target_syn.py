from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.unicm_peid_syn_analysis import (
    MODE_NAMES,
    enumerate_full_history_mode_pairs,
    load_full_history_prediction_cache,
    overall_prediction_cache_path,
    sample_full_history_mode_inputs,
    summarize_full_history_mode_pair_syn,
    summarize_full_history_pair_syn,
    write_jsonl,
)


DEFAULT_OVERALL_CACHE_DIR = ROOT / "results" / "unicm_overall_ei_cpu_bound4_n1024" / "cache"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "unicm_enso_joint_target_pair_syn_cpu_bound4_n1024"
DEFAULT_REPORT_PATH = ROOT / "docs" / "reports" / "Part2.md"

JOINT_TARGET_LABEL = "nino+nino3"
JOINT_TARGET_NAMES = ("nino", "nino3")
SECTION_TITLE = "### Joint ENSO target Syn ranking"
BOUNDARY_TITLE = "## 解释边界"


def _fmt_bits(value: object) -> str:
    number = float(value)
    return "nan" if not np.isfinite(number) else f"{number:.6f}"


def _display_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _insert_or_replace_section(markdown: str, section: str) -> str:
    start = markdown.find(SECTION_TITLE)
    if start >= 0:
        boundary = markdown.find(BOUNDARY_TITLE, start)
        if boundary < 0:
            raise ValueError(f"Could not find boundary heading after {SECTION_TITLE!r}.")
        return markdown[:start].rstrip() + "\n\n" + section.rstrip() + "\n\n" + markdown[boundary:].lstrip()

    boundary = markdown.find(BOUNDARY_TITLE)
    if boundary < 0:
        raise ValueError(f"Could not find insertion boundary heading {BOUNDARY_TITLE!r}.")
    return markdown[:boundary].rstrip() + "\n\n" + section.rstrip() + "\n\n" + markdown[boundary:].lstrip()


def build_joint_target_report_section(summary: pd.DataFrame, *, output_dir: Path, top_k: int) -> str:
    top_rows = summary.sort_values(["mean_syn", "pair"], ascending=[False, True]).head(int(top_k))
    lines = [
        SECTION_TITLE,
        "",
        "这里把同一 lead 上的 `nino` 和 `nino3` 未来输出合并为二维 target `{nino, nino3}`，",
        "重新计算每个 source-mode pair 的 full-history Gaussian log-det Syn。数值为 checkpoint seeds `1,2,3` 与 lead `1..24` 的均值；",
        "source pair 保留全部 11 个 mode 的两两组合，包括含 `nino` 或 `nino3` 历史的 pair。",
        "",
        "| rank | Source pair | mean Syn | joint EI | left EI | right EI |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for _, row in top_rows.iterrows():
        lines.append(
            f"| {int(row['rank_within_target'])} | {row['left_source']} + {row['right_source']} | "
            f"{_fmt_bits(row['mean_syn'])} | {_fmt_bits(row['mean_joint_ei'])} | "
            f"{_fmt_bits(row['mean_left_ei'])} | {_fmt_bits(row['mean_right_ei'])} |"
        )
    lines.extend(
        [
            "",
            "完整 55 个 source pair 排名见：",
            f"`{_display_path(output_dir / 'joint_target_mode_pair_syn_summary.csv')}`；"
            f"逐 seed / lead 原始结果见 `{_display_path(output_dir / 'joint_target_mode_pair_syn_rows.jsonl')}`。",
        ]
    )
    return "\n".join(lines)


def run_joint_target_syn(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in args.seeds]
    leads = list(range(int(args.lead_start), int(args.lead_end) + 1))
    if min(leads) < 1 or max(leads) > 24:
        raise ValueError("lead range must stay within [1, 24].")

    source_modes = list(MODE_NAMES)
    source_pairs = enumerate_full_history_mode_pairs(source_modes)
    history_modes = sample_full_history_mode_inputs(
        n_samples=int(args.n_samples),
        intervention_bound=float(args.intervention_bound),
        seed=int(args.sampling_seed),
    )
    cache_args = SimpleNamespace(
        n_samples=int(args.n_samples),
        sampling_seed=int(args.sampling_seed),
        intervention_bound=float(args.intervention_bound),
        start_month=int(args.start_month),
        device=str(args.device),
    )
    target_indices = [MODE_NAMES[name] for name in JOINT_TARGET_NAMES]

    rows: list[dict[str, object]] = []
    for seed in seeds:
        cache_path = overall_prediction_cache_path(Path(args.overall_cache_dir), seed=seed, args=cache_args)
        all_mode_targets = load_full_history_prediction_cache(cache_path, n_samples=int(args.n_samples))
        for left_name, right_name in source_pairs:
            pair_key = f"{left_name}|{right_name}"
            for lead in leads:
                target = all_mode_targets[:, int(lead) - 1, target_indices]
                summary = summarize_full_history_mode_pair_syn(
                    history_modes,
                    left_name,
                    right_name,
                    target,
                    bootstrap_indices=None,
                )
                rows.append(
                    {
                        "seed": int(seed),
                        "pair": pair_key,
                        "left_source": left_name,
                        "right_source": right_name,
                        "target": JOINT_TARGET_LABEL,
                        "target_modes": "|".join(JOINT_TARGET_NAMES),
                        "target_dim": len(JOINT_TARGET_NAMES),
                        "lead": int(lead),
                        "n_samples": int(args.n_samples),
                        "intervention_bound": float(args.intervention_bound),
                        "sampling_seed": int(args.sampling_seed),
                        "sampling_mode": "full_history_max_entropy",
                        **summary,
                    }
                )

    rows_path = output_dir / "joint_target_mode_pair_syn_rows.jsonl"
    write_jsonl(rows, rows_path)
    frame = pd.DataFrame(rows)
    summary = summarize_full_history_pair_syn(frame, window=(int(args.lead_start), int(args.lead_end)))
    summary_path = output_dir / "joint_target_mode_pair_syn_summary.csv"
    summary.to_csv(summary_path, index=False)
    top_pairs = summary[summary["rank_within_target"] <= int(args.top_k)]
    top_pairs_path = output_dir / "joint_target_mode_pair_syn_top_pairs.csv"
    top_pairs.to_csv(top_pairs_path, index=False)

    if args.report_path:
        report_path = Path(args.report_path)
        section = build_joint_target_report_section(summary, output_dir=output_dir, top_k=int(args.top_k))
        markdown = report_path.read_text(encoding="utf-8")
        report_path.write_text(_insert_or_replace_section(markdown, section), encoding="utf-8")

    return {
        "rows": str(rows_path),
        "summary": str(summary_path),
        "top_pairs": str(top_pairs_path),
        "n_rows": int(len(frame)),
        "n_pairs": int(len(summary)),
        "report": str(args.report_path) if args.report_path else None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank full-history source-pair Syn for joint ENSO target {nino,nino3}.")
    parser.add_argument("--overall-cache-dir", type=Path, default=DEFAULT_OVERALL_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--sampling-seed", type=int, default=20260619)
    parser.add_argument("--intervention-bound", type=float, default=4.0)
    parser.add_argument("--start-month", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lead-start", type=int, default=1)
    parser.add_argument("--lead-end", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    outputs = run_joint_target_syn(args)
    print(json.dumps(outputs, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
