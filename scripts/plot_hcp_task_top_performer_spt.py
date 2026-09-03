#!/usr/bin/env python3
"""Plot the state-matched Yeo-7 SPT for each task's top-scoring subject."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgb


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.analyze_hcp_all_task_behavior_coalitions_57 as behavior


DEFAULT_RECORDS = ROOT / "results/hcp_schaefer1000_task_evoked_xi_57/full/records.jsonl"
DEFAULT_FIGURE = ROOT / "fig/brain_hcp_schaefer1000_task_top_performer_spt.png"
SYN_TOLERANCE_BITS = 1.0e-10
TASK_ORDER = ("LANGUAGE", "SOCIAL", "EMOTION", "MOTOR", "GAMBLING", "RELATIONAL", "WM")
TASK_LABELS = {
    "LANGUAGE": "Language",
    "SOCIAL": "Social",
    "EMOTION": "Emotion",
    "MOTOR": "Motor",
    "GAMBLING": "Gambling",
    "RELATIONAL": "Relational",
    "WM": "Working memory",
}
NETWORK_LABELS = {
    "Vis": "Visual",
    "SomMot": "SomMot",
    "DorsAttn": "DorsAttn",
    "SalVentAttn": "Sal/VentAttn",
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
EDGE = "#B7C0CA"
INK = "#25313C"
MUTED = "#56616C"
SYN_COLOR = "#267A70"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["state"]), str(row["subject"]))
        records[key] = row
    return records


def validate_atoms(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    atoms = [dict(atom) for atom in sorted(row["atoms"], key=lambda item: int(item["depth"]))]
    sizes = [len(atom["sources"]) for atom in atoms]
    if sizes != list(range(7, 1, -1)):
        raise ValueError(
            f"Expected a complete seven-network hierarchy for {row['state']} "
            f"{row['subject']}, got coalition sizes {sizes}."
        )
    values = np.asarray([float(atom["value"]) for atom in atoms], dtype=float)
    violations = values < -SYN_TOLERANCE_BITS
    if np.any(violations):
        raise ValueError(
            "Significant Syn nonnegativity violation: "
            f"state={row['state']}, subject={row['subject']}, "
            f"minimum={values.min():.12g} bits, "
            f"threshold={-SYN_TOLERANCE_BITS:.12g} bits, "
            f"count={int(violations.sum())}."
        )
    closure = float(values.sum() - float(row["cross_network_xi"]))
    if abs(closure) > 1.0e-8:
        raise ValueError(
            f"Atom closure failed for {row['state']} {row['subject']}: "
            f"error={closure:.12g} bits."
        )
    return atoms


def select_rows(
    records: Mapping[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subject_sets = [
        {subject for state, subject in records if state == task}
        for task in TASK_ORDER
    ]
    subjects = np.asarray(sorted(set.intersection(*subject_sets)), dtype=str)
    if subjects.shape != (57,):
        raise ValueError(f"Expected 57 common task subjects, got {subjects.shape}.")
    contracts = behavior.make_endpoint_contracts(subjects, behavior.load_table())
    rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        scores = np.asarray(contracts[task]["endpoint"], dtype=float)
        maximum = float(scores.max())
        tied = np.flatnonzero(np.isclose(scores, maximum, rtol=0.0, atol=1.0e-12))
        # A subject-ID rule keeps ceiling ties independent of the SPT outcome.
        selected_index = int(tied[0])
        selected_subject = str(subjects[selected_index])
        key = (task, selected_subject)
        if key not in records:
            raise ValueError(f"Missing selected task-state record: {key}.")
        rows.append(records[key])
        selections.append(
            {
                "task": task,
                "subject": selected_subject,
                "score": maximum,
                "score_label": str(contracts[task]["label"]),
                "tie_count": int(len(tied)),
            }
        )
    return rows, selections


def blend_with_white(color: str, strength: float) -> str:
    base = to_rgb(color)
    amount = min(1.0, max(0.0, float(strength)))
    return to_hex(tuple(1.0 - amount * (1.0 - channel) for channel in base))


def text_color(facecolor: tuple[float, float, float, float]) -> str:
    red, green, blue = facecolor[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.53 else INK


def score_text(selection: Mapping[str, Any]) -> str:
    score = float(selection["score"])
    label = str(selection["score_label"]).replace("$", "")
    tie = int(selection["tie_count"])
    suffix = f" · {tie}-way ceiling tie" if tie > 1 else ""
    return f"{selection['subject']} · {label}: {score:.3f}{suffix}"


def atom_label(sources: Sequence[str], value: float) -> str:
    coalition = (
        f"{len(sources)} networks"
        if len(sources) >= 4
        else "+".join(NETWORK_SHORT[name] for name in sources)
    )
    return f"{coalition}\n{value:.3f}"


def draw_tree(
    axis: plt.Axes,
    row: Mapping[str, Any],
    atoms: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
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
        axis.plot([child_x, side_x], [parent_y, parent_y], color=EDGE, lw=0.8, zorder=1)
        axis.plot([child_x, child_x], [child_y, parent_y], color=EDGE, lw=0.8, zorder=1)
        axis.plot([side_x, side_x], [0.0, parent_y], color=EDGE, lw=0.8, zorder=1)
    terminal = signature[-1]
    terminal_x = [x[name] for name in terminal]
    axis.plot(terminal_x, [level[terminal]] * 2, color=EDGE, lw=0.8, zorder=1)
    for value in terminal_x:
        axis.plot([value, value], [0.0, level[terminal]], color=EDGE, lw=0.8, zorder=1)

    for atom, coalition in zip(atoms, signature, strict=True):
        value = float(atom["value"])
        facecolor = cmap(norm(value))
        relative = float(np.clip(norm(value), 0.0, 1.0))
        axis.text(
            node_x[coalition],
            level[coalition],
            atom_label(coalition, value),
            ha="center",
            va="center",
            fontsize=5.25,
            color=text_color(facecolor),
            linespacing=1.02,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": facecolor,
                "edgecolor": blend_with_white(SYN_COLOR, 0.55 + 0.45 * relative),
                "linewidth": 0.7 + 1.0 * relative,
            },
            zorder=4,
        )

    for name in leaf_order:
        color = NETWORK_COLORS[name]
        axis.text(
            x[name],
            -0.15,
            NETWORK_LABELS[name],
            rotation=58,
            ha="right",
            va="top",
            fontsize=5.4,
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": blend_with_white(color, 0.14),
                "edgecolor": color,
                "linewidth": 0.7,
            },
            clip_on=False,
        )

    axis.text(
        0.0,
        1.025,
        f"{panel}  {TASK_LABELS[str(row['state'])]}",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.3,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        0.0,
        0.985,
        score_text(selection) + "\n"
        + rf"system $\Xi$ {float(row['system_xi']):.3f} · "
        + rf"cross-network $\Xi$ {float(row['cross_network_xi']):.3f} bits",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.35,
        color=MUTED,
        linespacing=1.22,
    )
    axis.set_xlim(-0.55, len(leaf_order) - 0.45)
    axis.set_ylim(-1.10, 6.90)
    axis.axis("off")


def render(
    rows: Sequence[Mapping[str, Any]],
    atoms_by_row: Sequence[Sequence[Mapping[str, Any]]],
    selections: Sequence[Mapping[str, Any]],
    output: Path,
    dpi: int,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.0,
            "savefig.facecolor": "white",
        }
    )
    all_values = np.asarray(
        [float(atom["value"]) for atoms in atoms_by_row for atom in atoms], dtype=float
    )
    norm = mpl.colors.Normalize(vmin=0.0, vmax=float(all_values.max()))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("syn", ["#F1F7F5", SYN_COLOR])

    figure = plt.figure(figsize=(13.8, 7.5), layout="constrained")
    outer = figure.add_gridspec(2, 2, width_ratios=(1.0, 0.025), hspace=0.10, wspace=0.025)
    top = outer[0, 0].subgridspec(1, 3, wspace=0.04)
    bottom = outer[1, 0].subgridspec(1, 4, wspace=0.04)
    axes = [figure.add_subplot(top[0, index]) for index in range(3)]
    axes.extend(figure.add_subplot(bottom[0, index]) for index in range(4))
    color_axis = figure.add_subplot(outer[:, 1])

    for panel, axis, row, atoms, selection in zip(
        "abcdefg", axes, rows, atoms_by_row, selections, strict=True
    ):
        draw_tree(axis, row, atoms, selection, norm, cmap, panel)

    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = figure.colorbar(scalar, cax=color_axis)
    colorbar.set_label("Local hierarchy Syn (bits)", labelpad=6)
    colorbar.ax.tick_params(labelsize=5.5, length=2.2)
    figure.text(
        0.49,
        -0.012,
        "One state-matched tree per frozen behavioral endpoint · ceiling ties use the lowest subject ID",
        ha="center",
        va="top",
        fontsize=6.0,
        color=INK,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    rows, selections = select_rows(load_records(args.records))
    atoms_by_row = [validate_atoms(row) for row in rows]
    render(rows, atoms_by_row, selections, args.figure, args.dpi)
    print(f"Saved {args.figure}")
    for row, atoms, selection in zip(rows, atoms_by_row, selections, strict=True):
        values = np.asarray([float(atom["value"]) for atom in atoms], dtype=float)
        dominant = int(np.argmax(values))
        print(
            f"{row['state']} {row['subject']}: score={float(selection['score']):.6g}, "
            f"ties={int(selection['tie_count'])}, system_Xi={float(row['system_xi']):.6f}, "
            f"cross_Xi={float(row['cross_network_xi']):.6f}, "
            f"largest_atom={'+'.join(atoms[dominant]['sources'])} "
            f"({values[dominant]:.6f} bits), tolerance_negative_count="
            f"{int(np.sum((values < 0.0) & (values >= -SYN_TOLERANCE_BITS)))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
