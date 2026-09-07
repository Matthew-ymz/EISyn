#!/usr/bin/env python3
"""Validate HCP order trends using the Earth-style tree-wise SPT aggregation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = (
    ROOT / "results/hcp_schaefer1000_task_evoked_xi_57/full/records.jsonl"
)
DEFAULT_ALL_COALITIONS = (
    ROOT
    / "results/hcp_min_bipartition_synergy_57/minimum_bipartition_synergy_57.npz"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/hcp_treewise_order_mass_57"
STATES = (
    "REST",
    "EMOTION",
    "GAMBLING",
    "LANGUAGE",
    "MOTOR",
    "RELATIONAL",
    "SOCIAL",
    "WM",
)
STATE_LABELS = (
    "REST",
    "Emotion",
    "Gambling",
    "Language",
    "Motor",
    "Relational",
    "Social",
    "WM",
)
ORDERS = np.arange(2, 8, dtype=int)
SYN_TOLERANCE_BITS = 1.0e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--all-coalitions", type=Path, default=DEFAULT_ALL_COALITIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_tree_order_mass(path: Path) -> tuple[np.ndarray, list[str], dict[str, object]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keys = [(str(row["state"]), str(row["subject"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate state-subject trees in the HCP records")

    subjects = sorted({str(row["subject"]) for row in rows})
    expected = {(state, subject) for state in STATES for subject in subjects}
    if set(keys) != expected:
        missing = sorted(expected - set(keys))
        extra = sorted(set(keys) - expected)
        raise ValueError(f"Incomplete state-subject tree grid: missing={missing}, extra={extra}")

    mass = np.zeros((len(STATES), len(subjects), len(ORDERS)), dtype=float)
    node_counts = np.zeros_like(mass, dtype=int)
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    tolerance_zero_count = 0
    minimum_atom = float("inf")
    maximum_closure_error = 0.0
    order_count_patterns: Counter[tuple[tuple[int, int], ...]] = Counter()

    for row in rows:
        state_index = STATES.index(str(row["state"]))
        tree_index = subject_index[str(row["subject"])]
        tree_counts: Counter[int] = Counter()
        for atom in row["atoms"]:
            order = len(atom["sources"])
            if order not in ORDERS:
                raise ValueError(f"Unexpected SPT atom order {order}")
            value = float(atom["value"])
            minimum_atom = min(minimum_atom, value)
            if value < -SYN_TOLERANCE_BITS:
                raise ValueError(
                    "SPT Syn nonnegativity violation: "
                    f"minimum={value:.12g}, threshold={-SYN_TOLERANCE_BITS:.12g}, "
                    "affected_count=1"
                )
            if value < 0.0:
                tolerance_zero_count += 1
                value = 0.0
            order_index = order - int(ORDERS[0])
            mass[state_index, tree_index, order_index] += value
            node_counts[state_index, tree_index, order_index] += 1
            tree_counts[order] += 1
        order_count_patterns[tuple(sorted(tree_counts.items()))] += 1
        closure_error = float(mass[state_index, tree_index].sum() - row["cross_network_xi"])
        maximum_closure_error = max(maximum_closure_error, abs(closure_error))
        if abs(closure_error) > SYN_TOLERANCE_BITS:
            raise ValueError(
                f"Tree order-mass closure failed: error={closure_error:.12g}, "
                f"threshold={SYN_TOLERANCE_BITS:.12g}"
            )

    audit = {
        "record_count": len(rows),
        "subject_count": len(subjects),
        "minimum_selected_syn_bits": minimum_atom,
        "syn_tolerance_bits": SYN_TOLERANCE_BITS,
        "tolerance_zero_count": tolerance_zero_count,
        "maximum_tree_closure_error_bits": maximum_closure_error,
        "order_count_patterns": [
            {
                "order_node_counts": {str(order): count for order, count in pattern},
                "tree_count": frequency,
            }
            for pattern, frequency in order_count_patterns.most_common()
        ],
    }
    return mass, subjects, {"node_counts": node_counts, **audit}


def load_all_coalition_order_means(path: Path, subjects: list[str]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        values = np.asarray(archive["subject_values_bits"], dtype=float)
        sizes = np.asarray(archive["coalition_sizes"], dtype=int)
        cached_subjects = archive["subjects"].astype(str).tolist()
        cached_states = archive["states"].astype(str).tolist()
    if cached_subjects != subjects or cached_states != list(STATES):
        raise ValueError("All-coalition cache does not match the tree subject/state grid")
    return np.stack(
        [values[:, :, sizes == order].mean(axis=2).mean(axis=1) for order in ORDERS],
        axis=1,
    )


def text_color(value: float, image: mpl.image.AxesImage) -> str:
    red, green, blue, _ = image.cmap(image.norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#111111" if luminance > 0.56 else "#FFFFFF"


def plot_heatmap(values: np.ndarray, subject_count: int, output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.linewidth": 0.7,
            "savefig.facecolor": "white",
        }
    )
    figure, axis = plt.subplots(figsize=(6.8, 3.05), layout="constrained")
    display = values.T
    image = axis.imshow(
        display,
        cmap="viridis",
        vmin=0.0,
        vmax=float(np.ceil(display.max() * 10.0) / 10.0),
        aspect="auto",
        interpolation="nearest",
    )
    axis.set(
        xticks=np.arange(len(STATES)),
        xticklabels=STATE_LABELS,
        yticks=np.arange(len(ORDERS)),
        yticklabels=[str(order) for order in ORDERS],
        xlabel="State",
        ylabel="SPT node order",
    )
    axis.xaxis.tick_top()
    axis.xaxis.set_label_position("top")
    axis.tick_params(axis="x", labelrotation=30, length=0, labelsize=7.2, pad=3)
    axis.tick_params(axis="y", length=0, labelsize=7.2, pad=3)
    axis.set_xticks(np.arange(-0.5, len(STATES), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(ORDERS), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.45, alpha=0.55)
    axis.tick_params(which="minor", bottom=False, left=False)
    for row in range(display.shape[0]):
        for column in range(display.shape[1]):
            value = float(display[row, column])
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=6.4,
                fontweight="medium",
                color=text_color(value, image),
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.036, pad=0.025, aspect=22)
    colorbar.set_label("SPT order mass (bits)")
    colorbar.ax.tick_params(labelsize=6.5, length=2)
    axis.text(
        1.0,
        -0.135,
        f"Tree-wise order sum, then {subject_count}-subject mean; missing orders = 0",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color="#4A4A4A",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    tree_mass, subjects, audit = load_tree_order_mass(args.records)
    tree_means = tree_mass.mean(axis=1)
    old_means = load_all_coalition_order_means(args.all_coalitions, subjects)
    tree_deltas = np.diff(tree_means, axis=1)
    old_deltas = np.diff(old_means, axis=1)
    subject_strict = np.all(np.diff(tree_mass, axis=2) > 0.0, axis=2).sum(axis=1)
    node_counts = np.asarray(audit.pop("node_counts"), dtype=int)

    summary = {
        "status": "complete",
        "scientific_question": (
            "Does the Brain order trend remain strictly increasing when only the aggregation "
            "changes from all-coalition means to Earth-style tree-wise SPT order mass?"
        ),
        "treatment_factor": "order aggregation",
        "treatment_levels": ["all-coalition mean", "tree-wise SPT order sum then subject mean"],
        "frozen": {
            "records": "same 57-subject, eight-state k1_p3_a1 decomposition",
            "state_order": list(STATES),
            "orders": ORDERS.tolist(),
            "missing_order_rule": "zero mass within that tree",
            "estimator_and_tree": "unchanged cached affine/Gaussian Xi and selected SPT atoms",
        },
        "audit": audit,
        "treewise_order_mass_mean_bits": {
            state: tree_means[index].tolist() for index, state in enumerate(STATES)
        },
        "all_coalition_mean_bits": {
            state: old_means[index].tolist() for index, state in enumerate(STATES)
        },
        "treewise_adjacent_differences_bits": {
            state: tree_deltas[index].tolist() for index, state in enumerate(STATES)
        },
        "strictly_increasing_by_state": {
            state: bool(np.all(tree_deltas[index] > 0.0))
            for index, state in enumerate(STATES)
        },
        "old_strictly_increasing_by_state": {
            state: bool(np.all(old_deltas[index] > 0.0))
            for index, state in enumerate(STATES)
        },
        "strictly_increasing_tree_count_by_state": {
            state: int(subject_strict[index]) for index, state in enumerate(STATES)
        },
        "mean_node_count_by_order_and_state": {
            state: node_counts[index].mean(axis=0).tolist()
            for index, state in enumerate(STATES)
        },
        "peak_order_by_state": {
            state: int(ORDERS[np.argmax(tree_means[index])])
            for index, state in enumerate(STATES)
        },
        "conclusion": (
            "The original strict increase does not survive the aggregation change: all eight "
            "states decrease from order 6 to 7; MOTOR and RELATIONAL also decrease from order 2 "
            "to 3. The stable tree-wise pattern is a rise toward a mid/high-order peak, usually "
            "order 6, rather than a strict increase through order 7."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    figure_path = args.output_dir / "treewise_order_mass_heatmap_57.png"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_heatmap(tree_means, len(subjects), figure_path)

    print("tree-wise order mass means (rows=states, columns=orders 2..7)")
    print(np.array2string(tree_means, precision=6, suppress_small=False))
    print("strictly increasing:", summary["strictly_increasing_by_state"])
    print("peak orders:", summary["peak_order_by_state"])
    print(summary_path)
    print(figure_path)


if __name__ == "__main__":
    main()
