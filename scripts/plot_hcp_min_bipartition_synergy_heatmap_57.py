#!/usr/bin/env python3
"""Plot mean minimum-bipartition synergy for all HCP Yeo-7 coalitions."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/hcp_min_bipartition_synergy_57"

NETWORKS = (
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
)
SHORT = {
    "Vis": "V",
    "SomMot": "SM",
    "DorsAttn": "DAN",
    "SalVentAttn": "VAN",
    "Limbic": "Lim",
    "Cont": "FPN",
    "Default": "DMN",
}
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
SYN_TOLERANCE_BITS = 1.0e-9

CACHE_SPECS = {
    "REST": (
        ROOT
        / "results/hcp_emotion_performance_coalitions_57/emotion_rest_coalition_synergy_57.npz",
        1,
    ),
    "EMOTION": (
        ROOT
        / "results/hcp_emotion_performance_coalitions_57/emotion_rest_coalition_synergy_57.npz",
        0,
    ),
    "GAMBLING": (
        ROOT
        / "results/hcp_gambling_reward_valuation_57/gambling_coalition_synergy_57.npz",
        None,
    ),
    "LANGUAGE": (
        ROOT
        / "results/hcp_language_story_math_coalitions_57/language_coalition_synergy_57.npz",
        None,
    ),
    "MOTOR": (
        ROOT / "results/hcp_motor_composite_scores_57/motor_coalition_synergy_57.npz",
        None,
    ),
    "RELATIONAL": (
        ROOT
        / "results/hcp_relational_performance_coalitions_57/relational_coalition_synergy_57.npz",
        None,
    ),
    "SOCIAL": (
        ROOT
        / "results/hcp_social_composite_scores_57/social_coalition_synergy_57.npz",
        None,
    ),
    "WM": (
        ROOT / "results/hcp_wm_performance_coalitions_57/wm_coalition_synergy_57.npz",
        None,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def coalitions() -> tuple[tuple[str, ...], ...]:
    return tuple(
        coalition
        for size in range(2, len(NETWORKS) + 1)
        for coalition in itertools.combinations(NETWORKS, size)
    )


def canonical_bipartitions(coalition: tuple[str, ...]):
    """Yield each unordered nontrivial bipartition exactly once."""
    first, rest = coalition[0], coalition[1:]
    for mask in range(1 << len(rest)):
        left = (first,) + tuple(
            rest[index] for index in range(len(rest)) if mask & (1 << index)
        )
        if len(left) == len(coalition):
            continue
        left_set = set(left)
        right = tuple(source for source in coalition if source not in left_set)
        yield left, right


def load_state_matrix(
    state: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path, state_index = CACHE_SPECS[state]
    with np.load(path, allow_pickle=False) as archive:
        values = np.asarray(archive["synergy_bits"], dtype=float)
        subjects = archive["subjects"].astype(str)
        names = archive["coalitions"].astype(str)
    if state_index is not None:
        values = values[int(state_index)]
    if values.shape != (57, 120):
        raise ValueError(f"{state}: expected (57, 120), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{state}: coalition cache contains non-finite values")
    return values, subjects, names


def minimum_bipartition_values(
    coalition_values: np.ndarray,
    coalition_names: np.ndarray,
    ordered_coalitions: tuple[tuple[str, ...], ...],
) -> tuple[np.ndarray, int]:
    lookup = {
        tuple(str(name).split("+")): index
        for index, name in enumerate(coalition_names)
    }
    expected = set(ordered_coalitions)
    if set(lookup) != expected:
        missing = sorted(expected - set(lookup))
        extra = sorted(set(lookup) - expected)
        raise ValueError(f"Coalition cache mismatch: missing={missing}, extra={extra}")

    result = np.empty_like(coalition_values)
    candidate_count = 0
    for coalition_index, coalition in enumerate(ordered_coalitions):
        candidates = []
        for left, right in canonical_bipartitions(coalition):
            left_xi = (
                0.0
                if len(left) == 1
                else coalition_values[:, lookup[left]]
            )
            right_xi = (
                0.0
                if len(right) == 1
                else coalition_values[:, lookup[right]]
            )
            candidates.append(
                coalition_values[:, lookup[coalition]] - left_xi - right_xi
            )
            candidate_count += 1
        result[:, coalition_index] = np.min(np.stack(candidates, axis=1), axis=1)
    return result, candidate_count


def compute_all_states() -> tuple[np.ndarray, np.ndarray, tuple[tuple[str, ...], ...], int]:
    ordered_coalitions = coalitions()
    state_values = []
    reference_subjects = None
    candidates_per_subject_state = None
    for state in STATES:
        raw_values, subjects, names = load_state_matrix(state)
        if reference_subjects is None:
            reference_subjects = subjects
        elif not np.array_equal(subjects, reference_subjects):
            if set(subjects.tolist()) != set(reference_subjects.tolist()):
                raise ValueError(f"{state}: subject set differs from the reference state")
            subject_lookup = {subject: index for index, subject in enumerate(subjects)}
            reorder = [subject_lookup[subject] for subject in reference_subjects]
            raw_values = raw_values[reorder]
            subjects = subjects[reorder]
        minima, candidate_count = minimum_bipartition_values(
            raw_values, names, ordered_coalitions
        )
        if candidates_per_subject_state is None:
            candidates_per_subject_state = candidate_count
        elif candidate_count != candidates_per_subject_state:
            raise RuntimeError("Bipartition candidate count changed across states")
        state_values.append(minima)

    values = np.stack(state_values, axis=0)
    violation = values < -SYN_TOLERANCE_BITS
    if np.any(violation):
        raise ValueError(
            "Minimum-bipartition Syn nonnegativity violation: "
            f"minimum={values.min():.12g}, threshold={-SYN_TOLERANCE_BITS:.12g}, "
            f"affected_count={int(violation.sum())}"
        )
    assert reference_subjects is not None
    assert candidates_per_subject_state is not None
    return values, reference_subjects, ordered_coalitions, candidates_per_subject_state


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.linewidth": 0.7,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
        }
    )


def compact_label(coalition: tuple[str, ...]) -> str:
    return "+".join(SHORT[source] for source in coalition)


def plot_heatmap(
    mean_values: np.ndarray,
    ordered_coalitions: tuple[tuple[str, ...], ...],
    output_path: Path,
) -> None:
    configure_style()
    grand_mean = mean_values.mean(axis=0)
    display_order = np.asarray(
        [
            index
            for size in range(2, 8)
            for index in sorted(
                (i for i, coalition in enumerate(ordered_coalitions) if len(coalition) == size),
                key=lambda i: (-float(grand_mean[i]), ordered_coalitions[i]),
            )
        ],
        dtype=int,
    )
    panel = mean_values[:, display_order].T
    displayed = [ordered_coalitions[index] for index in display_order]

    figure, axis = plt.subplots(figsize=(9.2, 21.0), layout="constrained")
    color_max = float(np.ceil(panel.max() * 20.0) / 20.0)
    image = axis.imshow(
        panel,
        cmap="viridis",
        vmin=0.0,
        vmax=color_max,
        aspect="auto",
        interpolation="nearest",
    )
    axis.set(
        xticks=np.arange(len(STATES)),
        xticklabels=STATE_LABELS,
        yticks=np.arange(len(displayed)),
        yticklabels=[compact_label(coalition) for coalition in displayed],
        xlabel="State",
        ylabel="Network coalition",
    )
    axis.xaxis.tick_top()
    axis.xaxis.set_label_position("top")
    axis.tick_params(axis="x", labelrotation=35, labelsize=7.0, pad=3)
    axis.tick_params(axis="y", labelsize=4.2, pad=1.5)
    axis.axvline(0.5, color="white", linewidth=1.0)

    boundaries = []
    start = 0
    for size in range(2, 8):
        count = sum(len(coalition) == size for coalition in displayed)
        midpoint = start + (count - 1) / 2.0
        axis.text(
            -0.20,
            midpoint,
            f"{size}-network",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=5.8,
            fontweight="bold",
            color="#303030",
        )
        start += count
        if start < len(displayed):
            boundaries.append(start - 0.5)
    for boundary in boundaries:
        axis.axhline(boundary, color="white", linewidth=1.2)

    for row in range(panel.shape[0]):
        for column in range(panel.shape[1]):
            axis.text(
                column,
                row,
                f"{panel[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=3.15,
                color="white",
                path_effects=[
                    path_effects.withStroke(linewidth=0.65, foreground="#171717")
                ],
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.024, pad=0.018, aspect=55)
    colorbar.set_label("Mean minimum-bipartition Syn (bits)", fontsize=7.0)
    colorbar.ax.tick_params(labelsize=6.2, length=2)
    axis.text(
        1.0,
        1.016,
        "n = 57 subjects; cells show subject means",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#4A4A4A",
    )
    figure.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    values, subjects, ordered_coalitions, candidate_count = compute_all_states()
    mean_values = values.mean(axis=1)
    sd_values = values.std(axis=1, ddof=1)

    np.savez_compressed(
        args.output_dir / "minimum_bipartition_synergy_57.npz",
        states=np.asarray(STATES),
        subjects=subjects,
        coalitions=np.asarray(["+".join(value) for value in ordered_coalitions]),
        coalition_sizes=np.asarray([len(value) for value in ordered_coalitions]),
        subject_values_bits=values,
        mean_bits=mean_values,
        sd_bits=sd_values,
        syn_tolerance_bits=np.asarray(SYN_TOLERANCE_BITS),
    )
    output_path = args.output_dir / "minimum_bipartition_synergy_heatmap_57.png"
    plot_heatmap(mean_values, ordered_coalitions, output_path)

    grand_mean = mean_values.mean(axis=0)
    top = np.argsort(grand_mean)[::-1][:10]
    print(f"states={len(STATES)} subjects={len(subjects)} coalitions={len(ordered_coalitions)}")
    print(f"bipartitions_per_subject_state={candidate_count}")
    print(
        f"subject_value_min={values.min():.9f} subject_value_max={values.max():.9f} "
        f"mean_cell_min={mean_values.min():.9f} mean_cell_max={mean_values.max():.9f}"
    )
    for rank, index in enumerate(top, start=1):
        print(
            f"top{rank} {'+'.join(ordered_coalitions[index])}: "
            f"cross_state_mean={grand_mean[index]:.6f} bits"
        )
    print(output_path)


if __name__ == "__main__":
    main()
