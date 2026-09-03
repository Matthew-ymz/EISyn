#!/usr/bin/env python3
"""Render four state-specific Yeo-7 SPTs for one frozen HCP subject."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = (
    ROOT / "results/hcp_schaefer1000_task_evoked_xi_57/full/records.jsonl"
)
DEFAULT_FIGURE = (
    ROOT / "fig/brain_hcp_schaefer1000_xi_hierarchy_sub101915_states.png"
)
TOLERANCE_BITS = 1.0e-10
NETWORK_ORDER = (
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
)
NETWORK_LABELS = {
    "Vis": "Visual",
    "SomMot": "SomMot",
    "DorsAttn": "DorsAttn",
    "SalVentAttn": "SalVentAttn",
    "Limbic": "Limbic",
    "Cont": "Control",
    "Default": "Default",
}
NETWORK_SHORT = {
    "Vis": "V",
    "SomMot": "SM",
    "DorsAttn": "DAN",
    "SalVentAttn": "VAN",
    "Limbic": "Lim",
    "Cont": "FPN",
    "Default": "DMN",
}
NETWORK_COLORS = {
    "Vis": "#6A51A3",
    "SomMot": "#3182BD",
    "DorsAttn": "#31A354",
    "SalVentAttn": "#E6550D",
    "Limbic": "#9C6B30",
    "Cont": "#E6AB02",
    "Default": "#66A61E",
}
SUBJECT = "sub-101915"
SELECTIONS = (
    ("REST", "sub-101915"),
    ("LANGUAGE", "sub-101915"),
    ("MOTOR", "sub-101915"),
    ("SOCIAL", "sub-101915"),
)
EDGE = "#B7C0CA"
INK = "#25313C"
SYN_COLOR = "#267A70"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_selected(path: Path) -> list[dict[str, Any]]:
    wanted = set(SELECTIONS)
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["state"]), str(row["subject"]))
        if key in wanted:
            found[key] = row
    missing = [key for key in SELECTIONS if key not in found]
    if missing:
        raise ValueError(f"Missing selected state-subject records: {missing}")
    return [found[key] for key in SELECTIONS]


def validate_atoms(row: dict[str, Any]) -> list[dict[str, Any]]:
    atoms = [dict(atom) for atom in sorted(row["atoms"], key=lambda item: int(item["depth"]))]
    sizes = [len(atom["sources"]) for atom in atoms]
    if sizes != list(range(len(NETWORK_ORDER), 1, -1)):
        raise ValueError(
            f"Expected a complete seven-network nested hierarchy for "
            f"{row['state']} {row['subject']}, got sizes {sizes}"
        )
    for atom in atoms:
        value = float(atom["value"])
        if value < -TOLERANCE_BITS:
            raise ValueError(
                "Significant Syn nonnegativity violation: "
                f"state={row['state']}, subject={row['subject']}, "
                f"minimum={value:.12g} bits, threshold={-TOLERANCE_BITS:.12g} bits"
            )
        if value < 0.0:
            atom["value"] = 0.0
    closure = sum(float(atom["value"]) for atom in atoms) - float(row["cross_network_xi"])
    if abs(closure) > 1.0e-8:
        raise ValueError(
            f"Atom closure failed for {row['state']} {row['subject']}: "
            f"{closure:.12g} bits"
        )
    return atoms


def text_color(facecolor: tuple[float, float, float, float]) -> str:
    red, green, blue = facecolor[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.53 else INK


def blend_with_white(color: str, strength: float) -> str:
    base = to_rgb(color)
    amount = min(1.0, max(0.0, float(strength)))
    return to_hex(tuple(1.0 - amount * (1.0 - channel) for channel in base))


def atom_label(sources: Sequence[str], value: float) -> str:
    if len(sources) >= 4:
        coalition = f"{len(sources)} networks"
    else:
        coalition = "+".join(NETWORK_SHORT[name] for name in sources)
    return f"{coalition}\nSyn {value:.3f}"


def draw_tree(
    axis: plt.Axes,
    row: dict[str, Any],
    atoms: Sequence[dict[str, Any]],
    norm: mpl.colors.Normalize,
    cmap: mpl.colors.Colormap,
    panel: str,
) -> None:
    signature = [tuple(atom["sources"]) for atom in atoms]
    removed = [
        next(name for name in signature[index] if name not in signature[index + 1])
        for index in range(len(signature) - 1)
    ]
    leaf_order = removed + list(signature[-1])
    x = {name: float(index) for index, name in enumerate(leaf_order)}
    level = {coalition: float(len(coalition) - 1) for coalition in signature}
    node_x = {
        coalition: float(np.mean([x[name] for name in coalition]))
        for coalition in signature
    }

    for index in range(len(atoms) - 1):
        parent = signature[index]
        child = signature[index + 1]
        side = removed[index]
        parent_y, child_y = level[parent], level[child]
        child_x, side_x = node_x[child], x[side]
        axis.plot([child_x, side_x], [parent_y, parent_y], color=EDGE, lw=0.95, zorder=1)
        axis.plot([child_x, child_x], [child_y, parent_y], color=EDGE, lw=0.95, zorder=1)
        axis.plot([side_x, side_x], [0.0, parent_y], color=EDGE, lw=0.95, zorder=1)
    terminal = signature[-1]
    terminal_x = [x[name] for name in terminal]
    axis.plot(
        terminal_x,
        [level[terminal], level[terminal]],
        color=EDGE,
        lw=0.95,
        zorder=1,
    )
    for value in terminal_x:
        axis.plot([value, value], [0.0, level[terminal]], color=EDGE, lw=0.95, zorder=1)

    for atom, coalition in zip(atoms, signature, strict=True):
        value = float(atom["value"])
        facecolor = cmap(norm(value))
        relative = float(norm(value))
        axis.text(
            node_x[coalition],
            level[coalition],
            atom_label(coalition, value),
            ha="center",
            va="center",
            fontsize=6.5,
            color=text_color(facecolor),
            linespacing=1.10,
            bbox={
                "boxstyle": "round,pad=0.27",
                "facecolor": facecolor,
                "edgecolor": blend_with_white(SYN_COLOR, 0.58 + 0.42 * relative),
                "linewidth": 0.9 + 1.5 * relative,
            },
            zorder=4,
        )

    for name in leaf_order:
        color = NETWORK_COLORS[name]
        axis.text(
            x[name],
            -0.16,
            NETWORK_LABELS[name],
            rotation=55,
            ha="right",
            va="top",
            fontsize=7.0,
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": blend_with_white(color, 0.16),
                "edgecolor": color,
                "linewidth": 0.85,
            },
            clip_on=False,
        )

    axis.text(
        0.0,
        1.03,
        f"{panel}  {row['state'].title()}",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        0.0,
        0.995,
        f"{row['subject']}\n"
        rf"system $\Xi$ {float(row['system_xi']):.3f} · "
        rf"cross-network $\Xi$ {float(row['cross_network_xi']):.3f} bits",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color="#4F5B66",
        linespacing=1.35,
    )
    axis.set_xlim(-0.55, len(leaf_order) - 0.45)
    axis.set_ylim(-1.05, 6.85)
    axis.axis("off")


def render(
    rows: Sequence[dict[str, Any]],
    atoms_by_row: Sequence[Sequence[dict[str, Any]]],
    output: Path,
    dpi: int,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "savefig.facecolor": "white",
        }
    )
    all_values = np.asarray(
        [float(atom["value"]) for atoms in atoms_by_row for atom in atoms],
        dtype=float,
    )
    norm = mpl.colors.Normalize(vmin=0.0, vmax=float(all_values.max()))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "syn", ["#F1F7F5", SYN_COLOR]
    )

    figure = plt.figure(figsize=(13.2, 9.4), layout="constrained")
    grid = figure.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 0.045), wspace=0.06, hspace=0.12)
    axes = (
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
    )
    color_axis = figure.add_subplot(grid[:, 2])
    for panel, axis, row, atoms in zip(
        "abcd", axes, rows, atoms_by_row, strict=True
    ):
        draw_tree(
            axis,
            row,
            atoms,
            norm,
            cmap,
            panel,
        )
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = figure.colorbar(scalar, cax=color_axis)
    colorbar.set_label("Local hierarchy Syn (bits)", labelpad=7)
    colorbar.ax.tick_params(labelsize=6.5, length=2.5)
    figure.text(
        0.5,
        -0.01,
        f"Same subject ({SUBJECT}) · frozen model configuration · node fill shows local Syn",
        ha="center",
        va="top",
        fontsize=7.2,
        color=INK,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    rows = load_selected(args.records)
    atoms_by_row = [validate_atoms(row) for row in rows]
    render(rows, atoms_by_row, args.figure, args.dpi)
    print(f"Saved {args.figure}")
    for row, atoms in zip(rows, atoms_by_row, strict=True):
        print(
            f"{row['state']} {row['subject']}: "
            f"system_Xi={float(row['system_xi']):.6f}, "
            f"cross_Xi={float(row['cross_network_xi']):.6f}, "
            f"closure_error={sum(float(atom['value']) for atom in atoms) - float(row['cross_network_xi']):.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
