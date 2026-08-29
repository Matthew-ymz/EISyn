#!/usr/bin/env python3
"""Render pure PEID hierarchy trees across modular-Kuramoto structures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phi_hierarchy import greedy_phi_tree
from scripts.synergy_hierarchy_tree_plot import _max_abs_syn, plot_synergy_hierarchy_tree


DEFAULT_SUMMARY = ROOT / "results/greedy_hierarchy_kuramoto/summary.json"
DEFAULT_OUTPUT_DIR = ROOT / "results/greedy_hierarchy_kuramoto/xi_hierarchy_trees"
DEFAULT_CROSS_COUPLINGS = (0.0, 0.25, 0.75, 1.5)


def _ei_table(row: dict[str, object]) -> dict[tuple[str, ...], float]:
    return {
        tuple(str(key).split("+")): float(value)
        for key, value in dict(row["ei_bits"]).items()
    }


def _condition_name(cross_coupling: float) -> str:
    return f"kout_{float(cross_coupling):.2f}".replace(".", "p")


def _condition_rows(summary: dict[str, object], *, seed: int) -> list[dict[str, object]]:
    rows = [
        dict(row)
        for row in summary["rows"]
        if int(row["seed"]) == int(seed)
        and not bool(row["shuffle_target"])
        and float(row["cross_coupling"]) in DEFAULT_CROSS_COUPLINGS
    ]
    return sorted(rows, key=lambda row: float(row["cross_coupling"]))


def _tree(row: dict[str, object]):
    table = _ei_table(row)
    sources = max(table, key=len)
    return greedy_phi_tree(sources, table, eps=1.0e-5, split_tolerance=0.10)


def render_condition(
    row: dict[str, object],
    output_path: Path,
    *,
    tree=None,
    syn_scale_max: float | None = None,
    dpi: int = 600,
) -> Path:
    hierarchy = _tree(row) if tree is None else tree
    labels = {f"theta{index}": rf"$\theta_{{{index}}}$" for index in range(1, 7)}
    return plot_synergy_hierarchy_tree(
        hierarchy,
        output_path,
        source_labels=labels,
        syn_scale_max=syn_scale_max,
        dpi=dpi,
    )


def render_all(summary_path: Path, output_dir: Path, *, seed: int = 0, dpi: int = 600) -> list[Path]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _condition_rows(summary, seed=seed)
    if len(rows) != len(DEFAULT_CROSS_COUPLINGS):
        found = [float(row["cross_coupling"]) for row in rows]
        raise ValueError(f"Expected conditions {DEFAULT_CROSS_COUPLINGS}, found {found}")
    trees = [_tree(row) for row in rows]
    shared_syn_scale = max(_max_abs_syn(tree) for tree in trees)
    return [
        render_condition(
            row,
            output_dir / f"kuramoto_xi_tree_{_condition_name(float(row['cross_coupling']))}.png",
            tree=tree,
            syn_scale_max=shared_syn_scale,
            dpi=dpi,
        )
        for row, tree in zip(rows, trees, strict=True)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_all(args.summary, args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
