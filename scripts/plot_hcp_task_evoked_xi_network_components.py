#!/usr/bin/env python3
"""Plot within-network Xi and cross-network Shapley components for HCP states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = (
    ROOT / "results" / "hcp_schaefer500_task_evoked_xi_tuning" / "full" / "records.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "hcp_schaefer500_task_evoked_xi_tuning"
    / "final"
    / "network_xi_within_shapley_decomposition"
)
CONFIG_ID = "k1_p3_a1"
STATES = ("REST", "EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")
STATE_LABELS = ("REST", "Emotion", "Gambling", "Language", "Motor", "Relational", "Social", "WM")
NETWORKS = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
NETWORK_LABELS = ("Visual", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Control", "Default")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_components(
    path: Path, config_id: str
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["config_id"] == config_id:
                selected[(row["state"], row["subject"])] = row

    subjects = sorted({subject for _, subject in selected})
    expected = len(STATES) * len(subjects)
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} state-subject rows, found {len(selected)}")

    shape = (len(STATES), len(subjects), len(NETWORKS))
    within = np.empty(shape, dtype=float)
    shapley = np.empty(shape, dtype=float)
    maximum_closure_error = 0.0
    for state_index, state in enumerate(STATES):
        for subject_index, subject in enumerate(subjects):
            row = selected[(state, subject)]
            for network_index, network in enumerate(NETWORKS):
                within_value = float(row["within_network_xi"][network])
                shapley_value = float(row["cross_network_shapley"][network])
                total_value = float(row["network_attribution"][network])
                within[state_index, subject_index, network_index] = within_value
                shapley[state_index, subject_index, network_index] = shapley_value
                maximum_closure_error = max(
                    maximum_closure_error,
                    abs(within_value + shapley_value - total_value),
                )
    if np.any(within < 0.0) or np.any(shapley < 0.0):
        raise ValueError("The selected decomposition contains negative components")
    if maximum_closure_error > 1.0e-12:
        raise ValueError(f"Component closure error is {maximum_closure_error:.3e} bits")
    return within, shapley, subjects, maximum_closure_error


def annotate_heatmap(axis: Any, values: np.ndarray, *, threshold: float, percent: bool) -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            label = f"{value:.1f}" if percent else f"{value:.3f}"
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=5.2,
                color="white" if value > threshold else "black",
            )


def format_heatmap(axis: Any, title: str) -> None:
    axis.set(
        xticks=np.arange(len(STATES)),
        xticklabels=STATE_LABELS,
        yticks=np.arange(len(NETWORKS)),
        yticklabels=NETWORK_LABELS,
        title=title,
    )
    axis.tick_params(axis="x", labelrotation=35, length=0)
    axis.tick_params(axis="y", length=0)
    axis.set_aspect("auto")


def plot_components(within: np.ndarray, shapley: np.ndarray, n_subjects: int, output: Path) -> None:
    within_mean = within.mean(axis=1).T
    shapley_mean = shapley.mean(axis=1).T
    within_fraction_mean = (within / (within + shapley)).mean(axis=1).T * 100.0

    absolute_maximum = float(max(within_mean.max(), shapley_mean.max()))
    fraction_maximum = max(25.0, float(np.ceil(within_fraction_mean.max() / 5.0) * 5.0))
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), constrained_layout=True)

    images = []
    for axis, values, title in zip(
        axes[:2],
        (within_mean, shapley_mean),
        (r"a   Within-network integration  $\Xi_g^{\mathrm{within}}$", r"b   Cross-network allocation  $\mathrm{Shapley}_g$"),
    ):
        image = axis.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=absolute_maximum)
        images.append(image)
        format_heatmap(axis, title)
        annotate_heatmap(axis, values, threshold=0.58 * absolute_maximum, percent=False)

    fraction_image = axes[2].imshow(
        within_fraction_mean,
        cmap="YlOrBr",
        vmin=0.0,
        vmax=fraction_maximum,
    )
    format_heatmap(axes[2], r"c   Within-network fraction of total attribution  $C_g$")
    axes[2].set_xlabel("State")
    annotate_heatmap(
        axes[2],
        within_fraction_mean,
        threshold=0.58 * fraction_maximum,
        percent=True,
    )

    absolute_bar = figure.colorbar(images[0], ax=axes[:2], shrink=0.80, pad=0.025)
    absolute_bar.set_label(f"Mean contribution (bits; n={n_subjects})")
    fraction_bar = figure.colorbar(fraction_image, ax=axes[2], shrink=0.80, pad=0.025)
    fraction_bar.set_label(f"Mean within fraction (%; n={n_subjects})")

    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--config-id", default=CONFIG_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    configure_style()
    within, shapley, subjects, maximum_closure_error = load_components(
        args.records, args.config_id
    )
    plot_components(within, shapley, len(subjects), args.output)
    print(
        json.dumps(
            {
                "config_id": args.config_id,
                "n_subjects": len(subjects),
                "output": str(args.output),
                "maximum_closure_error_bits": maximum_closure_error,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
