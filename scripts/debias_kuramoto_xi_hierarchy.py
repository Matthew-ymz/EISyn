#!/usr/bin/env python3
"""Apply one uniform split-sample jackknife to every Kuramoto EI subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phi_hierarchy import flatten_phi_tree
from scripts.plot_kuramoto_xi_hierarchy_trees import RAW_SUMMARY, _ei_table, _tree
from scripts.validate_greedy_hierarchy_kuramoto import paired_data, transport_map_ei_table


DEFAULT_OUTPUT = ROOT / "results/greedy_hierarchy_kuramoto/jackknife_summary.json"
HALF_SPLIT_SEED = 2203
SYN_NONNEGATIVE_TOLERANCE_BITS = 0.10


def _key(subset: tuple[str, ...]) -> str:
    return "+".join(subset)


def debias_row(row: dict[str, object], half_indices: tuple[np.ndarray, np.ndarray]) -> dict[str, object]:
    phases, target = paired_data(
        sample_count=int(row["sample_count"]),
        seed=int(row["seed"]),
        within_coupling=float(row["within_coupling"]),
        cross_coupling=float(row["cross_coupling"]),
        noise_scale=float(row["noise_scale"]),
    )
    half_tables = [
        transport_map_ei_table(phases[index], target[index], degree=int(row["transport_degree"]))
        for index in half_indices
    ]
    full_table = {
        tuple(str(name) for name in key.split("+")): float(value)
        for key, value in dict(row["ei_bits"]).items()
    }
    jackknife = {
        subset: float(2.0 * full_table[subset] - 0.5 * (half_tables[0][subset] + half_tables[1][subset]))
        for subset in full_table
    }
    output = dict(row)
    output["jackknife_ei_bits"] = {_key(key): value for key, value in jackknife.items()}
    output["half_sample_ei_bits"] = [
        {_key(key): float(value) for key, value in table.items()} for table in half_tables
    ]
    tree = _tree(output)
    leaves = []

    def visit(node) -> None:
        if not node.children:
            leaves.append(node.sources)
        for child in node.children:
            visit(child)

    visit(tree)
    atoms = flatten_phi_tree(tree)
    output["jackknife_tree"] = {
        "root_phi_bits": float(tree.phi_value),
        "root_syn_bits": float(tree.residual),
        "root_split": [list(child.sources) for child in tree.children],
        "leaf_count": len(leaves),
        "all_singleton_leaves": bool(len(leaves) == 6 and all(len(leaf) == 1 for leaf in leaves)),
        "minimum_atom_bits": float(min(atom.value for atom in atoms)),
        "negative_within_tolerance_count": int(
            sum(-SYN_NONNEGATIVE_TOLERANCE_BITS <= atom.value < 0.0 for atom in atoms)
        ),
        "below_negative_tolerance_count": int(
            sum(atom.value < -SYN_NONNEGATIVE_TOLERANCE_BITS for atom in atoms)
        ),
        "closure_error_bits": float(sum(atom.value for atom in atoms) - tree.phi_value),
    }
    return output


def build(raw_summary: Path, output: Path, *, seed: int) -> dict[str, object]:
    source = json.loads(raw_summary.read_text(encoding="utf-8"))
    rows = [
        dict(row)
        for row in source["rows"]
        if int(row["seed"]) == int(seed) and not bool(row["shuffle_target"])
    ]
    if not rows:
        raise ValueError(f"No unshuffled rows found for seed {seed}.")
    sample_count = int(rows[0]["sample_count"])
    order = np.random.default_rng(HALF_SPLIT_SEED).permutation(sample_count)
    half_indices = tuple(np.asarray(part, dtype=int) for part in np.array_split(order, 2))
    corrected = []
    for index, row in enumerate(sorted(rows, key=lambda item: float(item["cross_coupling"])), start=1):
        corrected.append(debias_row(row, half_indices))
        print(f"[{index}/{len(rows)}] K_out={float(row['cross_coupling']):.2f}", flush=True)
    payload = {
        "method": {
            "name": "uniform two-half delete-group jackknife",
            "formula": "EI_jackknife = 2*EI_full - mean(EI_half_1, EI_half_2)",
            "uses_network_truth": False,
            "condition_specific_rules": False,
            "half_split_seed": HALF_SPLIT_SEED,
            "syn_nonnegative_tolerance_bits": SYN_NONNEGATIVE_TOLERANCE_BITS,
            "tree_completion": "recurse until every leaf is a singleton",
        },
        "source_summary": str(raw_summary),
        "rows": corrected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-summary", type=Path, default=RAW_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.raw_summary, args.output, seed=args.seed)
