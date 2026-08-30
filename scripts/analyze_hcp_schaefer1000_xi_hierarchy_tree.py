#!/usr/bin/env python3
"""Render a representative REST Yeo-7 Xi hierarchy from the frozen 57-subject HCP run."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = ROOT / "results/hcp_schaefer1000_task_evoked_xi_57/full/records.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "results/hcp_schaefer1000_xi_hierarchy_tree"
DEFAULT_FIGURE = ROOT / "fig/brain_hcp_schaefer1000_xi_hierarchy_rest_representative.png"
TOLERANCE_BITS = 1.0e-10
NETWORK_ORDER = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
NETWORK_COLORS = {
    "Vis": "#6A51A3",
    "SomMot": "#3182BD",
    "DorsAttn": "#31A354",
    "SalVentAttn": "#E6550D",
    "Limbic": "#9C6B30",
    "Cont": "#E6AB02",
    "Default": "#66A61E",
}
EDGE = "#AEB8C2"
INK = "#25313C"
SYN = "#267A70"


def topology(row: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    atoms = sorted(row["atoms"], key=lambda atom: (int(atom["depth"]), -len(atom["sources"])))
    return tuple(tuple(atom["sources"]) for atom in atoms)


def blend(color: str, strength: float) -> str:
    base = to_rgb(color)
    amount = min(1.0, max(0.0, float(strength)))
    return to_hex(tuple(1.0 - amount * (1.0 - channel) for channel in base))


def choose_representative(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    rest = [row for row in rows if row["state"] == "REST"]
    if len(rest) != 57:
        raise ValueError(f"Expected 57 REST records, found {len(rest)}")
    counts = collections.Counter(topology(row) for row in rest)
    modal_signature, modal_count = counts.most_common(1)[0]
    modal_rows = [row for row in rest if topology(row) == modal_signature]
    mean_cross = float(np.mean([float(row["cross_network_xi"]) for row in rest]))
    selected = min(modal_rows, key=lambda row: abs(float(row["cross_network_xi"]) - mean_cross))
    metadata = {
        "rest_subject_count": len(rest),
        "unique_complete_topology_count": len(counts),
        "modal_topology_frequency": int(modal_count),
        "modal_topology_fraction": float(modal_count / len(rest)),
        "rest_mean_cross_network_xi_bits": mean_cross,
        "selection_rule": "modal complete topology; within it, nearest to REST mean cross-network Xi",
    }
    return selected, metadata


def validate_atoms(row: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    atoms = sorted(row["atoms"], key=lambda atom: int(atom["depth"]))
    if [len(atom["sources"]) for atom in atoms] != list(range(len(NETWORK_ORDER), 1, -1)):
        raise ValueError("Expected a complete seven-network nested hierarchy")
    tolerance_zero = 0
    for atom in atoms:
        value = float(atom["value"])
        if value < -TOLERANCE_BITS:
            raise ValueError(f"Significant Syn nonnegativity violation: {value:.12g} < {-TOLERANCE_BITS:.12g}")
        if value < 0.0:
            tolerance_zero += 1
            atom["value"] = 0.0
    closure = float(sum(float(atom["value"]) for atom in atoms) - float(row["cross_network_xi"]))
    if abs(closure) > 1.0e-8:
        raise ValueError(f"Atom closure failed: {closure:.12g} bits")
    return atoms, tolerance_zero


def render(row: dict[str, Any], metadata: dict[str, Any], atoms: Sequence[dict[str, Any]], output: Path, dpi: int) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.0,
        "savefig.facecolor": "white",
    })
    signature = [tuple(atom["sources"]) for atom in atoms]
    removed = [next(name for name in signature[index] if name not in signature[index + 1]) for index in range(len(signature) - 1)]
    terminal = list(signature[-1])
    leaf_order = removed + terminal
    x = {name: float(index) for index, name in enumerate(leaf_order)}
    levels = {coalition: float(len(coalition) - 1) for coalition in signature}
    node_x = {coalition: float(np.mean([x[name] for name in coalition])) for coalition in signature}
    maximum = max(float(atom["value"]) for atom in atoms)

    figure, axis = plt.subplots(figsize=(10.8, 7.2), constrained_layout=True)
    figure.patch.set_facecolor("white")
    for index, atom in enumerate(atoms[:-1]):
        parent = signature[index]
        child = signature[index + 1]
        side = removed[index]
        py, cy = levels[parent], levels[child]
        px, cx, sx = node_x[parent], node_x[child], x[side]
        axis.plot([cx, sx], [py, py], color=EDGE, linewidth=1.1, zorder=1)
        axis.plot([cx, cx], [cy, py], color=EDGE, linewidth=1.1, zorder=1)
        axis.plot([sx, sx], [0.0, py], color=EDGE, linewidth=1.1, zorder=1)
    last = signature[-1]
    lx = [x[name] for name in last]
    axis.plot(lx, [levels[last], levels[last]], color=EDGE, linewidth=1.1, zorder=1)
    for value in lx:
        axis.plot([value, value], [0.0, levels[last]], color=EDGE, linewidth=1.1, zorder=1)

    for atom, coalition in zip(atoms, signature, strict=True):
        value = float(atom["value"])
        relative = value / maximum if maximum else 0.0
        label = (" + ".join(coalition) if len(coalition) <= 3 else f"{len(coalition)} networks") + f"\nSyn {value:.3f}"
        axis.text(
            node_x[coalition], levels[coalition], label,
            ha="center", va="center", fontsize=7.2, color=INK, linespacing=1.15,
            bbox={"boxstyle": "round,pad=0.30", "facecolor": blend(SYN, 0.14 + 0.60 * relative),
                  "edgecolor": blend(SYN, 0.58 + 0.42 * relative), "linewidth": 0.9 + 1.4 * relative},
            zorder=4,
        )
    for name in leaf_order:
        axis.scatter([x[name]], [-0.02], s=42, color=NETWORK_COLORS[name], zorder=4, clip_on=False)
        axis.text(x[name], -0.17, name, rotation=35, ha="right", va="top", fontsize=8.0, color=INK, clip_on=False)

    axis.text(
        0.02, 0.98,
        f"REST representative: {row['subject']}\n"
        rf"system $\Xi$ = {float(row['system_xi']):.3f} bits" "\n"
        rf"cross-network $\Xi$ = {float(row['cross_network_xi']):.3f} bits",
        transform=axis.transAxes, ha="left", va="top", fontsize=8.2, color=INK, linespacing=1.4,
    )
    axis.text(
        1.02, 0.72,
        f"Modal topology: {metadata['modal_topology_frequency']}/{metadata['rest_subject_count']} subjects\n"
        f"Unique complete topologies: {metadata['unique_complete_topology_count']}\n"
        f"Selection: modal topology, then nearest\nREST mean cross-network Xi",
        transform=axis.transAxes, ha="left", va="top", fontsize=7.5, color=INK, linespacing=1.45,
    )
    axis.text(
        0.5, 1.04, "HCP Schaefer-1000: representative REST Yeo-7 hierarchy",
        transform=axis.transAxes, ha="center", va="bottom", fontsize=10.0, color=INK,
    )
    axis.set_xlim(-0.7, len(leaf_order) - 0.3)
    axis.set_ylim(-0.65, 6.75)
    axis.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    selected, metadata = choose_representative(rows)
    atoms, tolerance_zero = validate_atoms(selected)
    render(selected, metadata, atoms, args.figure, args.dpi)
    summary = {
        "input": str(args.records),
        "figure": str(args.figure),
        "nonnegative_tolerance_bits": TOLERANCE_BITS,
        "tolerance_zero_atom_count": tolerance_zero,
        "representative": {
            "subject": selected["subject"],
            "system_xi_bits": float(selected["system_xi"]),
            "cross_network_xi_bits": float(selected["cross_network_xi"]),
            "within_network_xi_bits": float(selected["within_network_xi_sum"]),
            "atoms": atoms,
        },
        **metadata,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
