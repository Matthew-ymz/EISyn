#!/usr/bin/env python3
"""Plot subject-level HCP Xi attribution, hierarchy atoms, and cognition profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARRAYS = (
    ROOT
    / "results/hcp_schaefer500_task_evoked_xi_tuning/full/k1_p3_a1/arrays.npz"
)
DEFAULT_COGNITION = (
    ROOT / "results/hcp_single_group_sem_full_1206/selected_29_sem_results.csv"
)
DEFAULT_OUTPUT = ROOT / "results/hcp_cognition_individual_xi_profiles"

STATES = ("REST", "EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")
STATE_LABELS = ("REST", "Emotion", "Gambling", "Language", "Motor", "Relational", "Social", "WM")
NETWORKS = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
NETWORK_LABELS = ("Visual", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Control", "Default")
NETWORK_SHORT = ("Vis", "Som", "DAN", "SVAN", "Lim", "Cont", "Def")
SCORES = ("g_score", "cry_score", "mem_score", "spd_score")
SCORE_LABELS = ("General", "Crystallized", "Memory", "Speed")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument("--cognition", type=Path, default=DEFAULT_COGNITION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-individual",
        action="store_true",
        help="Reuse the existing 29 subject PNGs and multipage PDF while redrawing cohort overviews.",
    )
    return parser.parse_args()


def strip_subject(value: object) -> str:
    return str(value).removeprefix("sub-")


def short_atom(name: str) -> str:
    mapping = {
        "Vis": "Vis",
        "SomMot": "Som",
        "DorsAttn": "DAN",
        "SalVentAttn": "SVAN",
        "Limbic": "Lim",
        "Cont": "Cont",
        "Default": "Def",
    }
    return "+".join(mapping[item] for item in name.split("+"))


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return adjusted
    flat = values[valid]
    order = np.argsort(flat)
    ranked = flat[order]
    q_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q = np.empty_like(flat)
    q[order] = np.clip(q_ranked, 0.0, 1.0)
    adjusted[valid] = q
    return adjusted


def load_data(arrays_path: Path, cognition_path: Path) -> dict[str, object]:
    archive = np.load(arrays_path)
    states = tuple(archive["states"].astype(str))
    networks = tuple(archive["networks"].astype(str))
    if states != STATES or networks != NETWORKS:
        raise ValueError(f"Unexpected state/network order: {states}; {networks}")

    subjects = np.asarray([strip_subject(value) for value in archive["subjects"]], dtype=str)
    if len(subjects) != 29 or len(set(subjects)) != 29:
        raise ValueError("Expected 29 unique Xi subjects")

    cognition = pd.read_csv(cognition_path, dtype={"Subject": str})
    required = {"Subject", *SCORES}
    missing = required.difference(cognition.columns)
    if missing:
        raise ValueError(f"Cognition table is missing columns: {sorted(missing)}")
    cognition["Subject"] = cognition["Subject"].map(strip_subject)
    cognition = cognition.set_index("Subject")
    if set(cognition.index) != set(subjects):
        raise ValueError(
            "Xi and cognition subject sets differ: "
            f"missing={sorted(set(subjects) - set(cognition.index))}; "
            f"extra={sorted(set(cognition.index) - set(subjects))}"
        )
    cognition = cognition.loc[subjects, list(SCORES)]
    score_values = cognition.to_numpy(dtype=float)
    if not np.isfinite(score_values).all():
        raise ValueError("Cognition scores contain non-finite values")
    score_z = (score_values - score_values.mean(axis=0)) / score_values.std(axis=0, ddof=1)

    network_share = np.asarray(archive["network_share"], dtype=float) * 100.0
    atom_share = np.asarray(archive["atom_share"], dtype=float)
    cross_xi = np.asarray(archive["cross_xi"], dtype=float)
    atom_value = atom_share * cross_xi[:, :, None]
    atom_names = archive["atom_names"].astype(str)
    selected = np.argsort(atom_share.mean(axis=1).mean(axis=0))[::-1][:12]

    expected_network_shape = (len(STATES), len(subjects), len(NETWORKS))
    if network_share.shape != expected_network_shape:
        raise ValueError(f"Unexpected network_share shape: {network_share.shape}")
    if atom_value.shape[:2] != (len(STATES), len(subjects)):
        raise ValueError(f"Unexpected atom_value shape: {atom_value.shape}")

    return {
        "subjects": subjects,
        "cognition": cognition,
        "score_z": score_z,
        "network_share": network_share,
        "atom_value": atom_value[:, :, selected],
        "atom_names": atom_names[selected],
        "selected_atom_indices": selected,
    }


def cognition_cmap() -> mpl.colors.Colormap:
    return mpl.colors.LinearSegmentedColormap.from_list(
        "cognition_diverging", ("#5B8FB9", "#F4F1EF", "#B65F3C")
    )


def annotate_heatmap(
    axis: mpl.axes.Axes,
    values: np.ndarray,
    *,
    formatter: str,
    threshold: float,
    fontsize: float,
) -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            axis.text(
                column,
                row,
                format(value, formatter),
                ha="center",
                va="center",
                fontsize=fontsize,
                color="white" if value >= threshold else "black",
            )


def individual_figure(
    subject: str,
    network_panel: np.ndarray,
    atom_panel: np.ndarray,
    atom_labels: list[str],
    score_panel: np.ndarray,
    *,
    network_vmax: float,
    atom_vmax: float,
) -> plt.Figure:
    figure = plt.figure(figsize=(15.2, 5.2), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.0, 1.55, 0.34), wspace=0.08)
    network_axis = figure.add_subplot(grid[0, 0])
    atom_axis = figure.add_subplot(grid[0, 1])
    cognition_axis = figure.add_subplot(grid[0, 2])

    network_image = network_axis.imshow(
        network_panel,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=network_vmax,
        aspect="auto",
        interpolation="nearest",
    )
    network_axis.set(
        xticks=np.arange(len(STATES)),
        xticklabels=STATE_LABELS,
        yticks=np.arange(len(NETWORKS)),
        yticklabels=NETWORK_LABELS,
        xlabel="State (each column sums to 100%)",
        ylabel="Yeo7 network",
    )
    network_axis.tick_params(axis="x", rotation=35, length=0)
    network_axis.tick_params(axis="y", length=0)
    network_axis.axvline(0.5, color="#333333", linewidth=0.8)
    annotate_heatmap(
        network_axis,
        network_panel,
        formatter=".1f",
        threshold=0.58 * network_vmax,
        fontsize=4.2,
    )
    figure.colorbar(network_image, ax=network_axis, shrink=0.76, pad=0.02).set_label(
        r"Subject-level share of system-level $\Xi$ (%)"
    )

    atom_image = atom_axis.imshow(
        atom_panel,
        cmap="magma_r",
        vmin=0.0,
        vmax=atom_vmax,
        aspect="auto",
        interpolation="nearest",
    )
    atom_axis.set(
        xticks=np.arange(len(STATES)),
        xticklabels=STATE_LABELS,
        yticks=np.arange(len(atom_labels)),
        yticklabels=atom_labels,
        xlabel="State",
        ylabel="Greedy hierarchy atom",
    )
    atom_axis.tick_params(axis="x", rotation=35, length=0)
    atom_axis.tick_params(axis="y", length=0)
    atom_axis.axvline(0.5, color="#F0F0F0", linewidth=0.8)
    annotate_heatmap(
        atom_axis,
        atom_panel,
        formatter=".3f",
        threshold=0.38 * atom_vmax,
        fontsize=3.8,
    )
    figure.colorbar(atom_image, ax=atom_axis, shrink=0.76, pad=0.02, extend="max").set_label(
        "Subject-level hierarchy-atom contribution (bits)"
    )

    cognition_image = cognition_axis.imshow(
        score_panel[:, None],
        cmap=cognition_cmap(),
        vmin=-2.5,
        vmax=2.5,
        aspect="auto",
        interpolation="nearest",
    )
    cognition_axis.set(
        xticks=[],
        yticks=np.arange(len(SCORES)),
        yticklabels=SCORE_LABELS,
        ylabel="Cognition",
    )
    cognition_axis.tick_params(axis="y", length=0)
    for row, value in enumerate(score_panel):
        cognition_axis.text(
            0,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=6,
            color="white" if abs(value) > 1.35 else "black",
        )
    figure.colorbar(cognition_image, ax=cognition_axis, shrink=0.55, pad=0.08).set_label(
        "Within-29 score z"
    )

    for label, axis in zip("abc", (network_axis, atom_axis, cognition_axis)):
        axis.text(-0.12, 1.035, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    figure.text(0.01, 1.015, f"Subject {subject}", ha="left", va="bottom", fontsize=8)
    return figure


def add_group_boundaries(axis: mpl.axes.Axes, group_size: int, group_count: int) -> None:
    for boundary in range(1, group_count):
        axis.axvline(boundary * group_size - 0.5, color="white", linewidth=1.1)


def ranked_overview(
    cognition: np.ndarray,
    brain: np.ndarray,
    subjects: np.ndarray,
    *,
    ordering_label: str,
    ordering_score_index: int,
    group_size: int,
    cmap: str | mpl.colors.Colormap,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    output_stem: Path,
) -> None:
    figure = plt.figure(figsize=(16.2, 8.5), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=(0.34, 2.5), wspace=0.04)
    cognition_axis = figure.add_subplot(grid[0, 0])
    brain_axis = figure.add_subplot(grid[0, 1])

    cognition_image = cognition_axis.imshow(
        cognition,
        cmap=cognition_cmap(),
        vmin=-2.5,
        vmax=2.5,
        aspect="auto",
        interpolation="nearest",
    )
    cognition_axis.set(
        xticks=np.arange(len(SCORES)),
        xticklabels=SCORE_LABELS,
        yticks=np.arange(len(subjects)),
        yticklabels=subjects,
        ylabel=f"Subjects ordered by {ordering_label.lower()} (high to low)",
    )
    cognition_axis.tick_params(axis="x", rotation=35, length=0)
    cognition_axis.tick_params(axis="y", length=0)
    cognition_axis.add_patch(
        mpl.patches.Rectangle(
            (ordering_score_index - 0.5, -0.5),
            1.0,
            len(subjects),
            fill=False,
            edgecolor="#202020",
            linewidth=1.2,
            clip_on=False,
        )
    )
    figure.colorbar(cognition_image, ax=cognition_axis, shrink=0.52, pad=0.04).set_label(
        "Within-29 score z"
    )

    brain_image = brain_axis.imshow(
        brain,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    centers = np.arange(len(STATES)) * group_size + (group_size - 1) / 2
    brain_axis.set(
        xticks=centers,
        xticklabels=STATE_LABELS,
        yticks=[],
        xlabel="State blocks",
    )
    brain_axis.tick_params(axis="x", rotation=25, length=0)
    add_group_boundaries(brain_axis, group_size, len(STATES))
    figure.colorbar(brain_image, ax=brain_axis, shrink=0.68, pad=0.012, extend="max").set_label(
        colorbar_label
    )
    for label, axis in zip("ab", (cognition_axis, brain_axis)):
        axis.text(-0.10, 1.025, label, transform=axis.transAxes, fontweight="bold", fontsize=9)

    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def correlation_family(scores: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    correlations = np.full((scores.shape[1], features.shape[1]), np.nan, dtype=float)
    p_values = np.full_like(correlations, np.nan)
    q_values = np.full_like(correlations, np.nan)
    for score_index in range(scores.shape[1]):
        for feature_index in range(features.shape[1]):
            x = features[:, feature_index]
            if np.allclose(x, x[0]):
                continue
            result = spearmanr(scores[:, score_index], x)
            correlations[score_index, feature_index] = float(result.statistic)
            p_values[score_index, feature_index] = float(result.pvalue)
        q_values[score_index] = bh_adjust(p_values[score_index])
    return correlations, p_values, q_values


def plot_correlations(
    network_r: np.ndarray,
    atom_r: np.ndarray,
    *,
    output_stem: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(17.2, 5.8), constrained_layout=True)
    panels = (
        (axes[0], network_r, len(NETWORKS), "Network attribution share"),
        (axes[1], atom_r, atom_r.shape[1] // len(STATES), "Hierarchy-atom contribution"),
    )
    image = None
    for axis, values, group_size, ylabel in panels:
        image = axis.imshow(values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
        centers = np.arange(len(STATES)) * group_size + (group_size - 1) / 2
        axis.set(
            xticks=centers,
            xticklabels=STATE_LABELS,
            yticks=np.arange(len(SCORES)),
            yticklabels=SCORE_LABELS,
            ylabel=ylabel,
        )
        axis.tick_params(axis="x", rotation=25, length=0)
        axis.tick_params(axis="y", length=0)
        add_group_boundaries(axis, group_size, len(STATES))
        for row, column in zip(*np.where(np.abs(values) >= 0.40)):
            axis.text(
                column,
                row,
                f"{values[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=4.1,
                color="white" if abs(values[row, column]) >= 0.48 else "black",
            )
    axes[1].set_xlabel("State blocks; annotations show descriptive |Spearman rho| ≥ 0.40")
    for label, axis in zip("ab", axes):
        axis.text(-0.055, 1.05, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    assert image is not None
    figure.colorbar(image, ax=axes, shrink=0.78, pad=0.012).set_label("Descriptive Spearman rho")
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def strongest_result(
    correlations: np.ndarray,
    p_values: np.ndarray,
    q_values: np.ndarray,
    feature_labels: list[str],
) -> dict[str, object]:
    flat_index = int(np.nanargmax(np.abs(correlations)))
    score_index, feature_index = np.unravel_index(flat_index, correlations.shape)
    return {
        "score": SCORES[score_index],
        "feature": feature_labels[feature_index],
        "rho": float(correlations[score_index, feature_index]),
        "p_descriptive": float(p_values[score_index, feature_index]),
        "q_within_score_family": float(q_values[score_index, feature_index]),
    }


def run(args: argparse.Namespace) -> None:
    configure_style()
    data = load_data(args.arrays, args.cognition)
    output = args.output_dir
    subject_dir = output / "subjects"
    subject_dir.mkdir(parents=True, exist_ok=True)

    subjects = np.asarray(data["subjects"])
    cognition = data["cognition"]
    assert isinstance(cognition, pd.DataFrame)
    score_values = cognition.to_numpy(dtype=float)
    score_z = np.asarray(data["score_z"], dtype=float)
    network_share = np.asarray(data["network_share"], dtype=float)
    atom_value = np.asarray(data["atom_value"], dtype=float)
    atom_names = np.asarray(data["atom_names"]).astype(str)
    atom_labels = [short_atom(name) for name in atom_names]

    network_vmax = float(np.ceil(network_share.max()))
    atom_vmax = float(np.quantile(atom_value, 0.995))
    individual_paths = [
        str(Path("subjects") / f"sub-{subject}_xi_cognition_profile.png")
        for subject in subjects
    ]
    multipage_path = output / "individual_xi_cognition_profiles.pdf"
    if args.skip_individual:
        missing_individual = [path for path in individual_paths if not (output / path).exists()]
        if missing_individual or not multipage_path.exists():
            raise FileNotFoundError(
                f"Cannot reuse individual outputs; missing PNGs={missing_individual[:3]}, "
                f"multipage_pdf_exists={multipage_path.exists()}"
            )
    else:
        with PdfPages(multipage_path) as pdf:
            for subject_index, subject in enumerate(subjects):
                figure = individual_figure(
                    subject,
                    network_share[:, subject_index, :].T,
                    atom_value[:, subject_index, :].T,
                    atom_labels,
                    score_z[subject_index],
                    network_vmax=network_vmax,
                    atom_vmax=atom_vmax,
                )
                path = subject_dir / f"sub-{subject}_xi_cognition_profile.png"
                figure.savefig(path, dpi=400, bbox_inches="tight")
                pdf.savefig(figure, bbox_inches="tight")
                plt.close(figure)

    network_flat = network_share.transpose(1, 0, 2).reshape(len(subjects), -1)
    atom_flat = atom_value.transpose(1, 0, 2).reshape(len(subjects), -1)
    ranked_outputs: dict[str, dict[str, object]] = {}
    for score_index, (score, score_label) in enumerate(zip(SCORES, SCORE_LABELS, strict=True)):
        order = np.argsort(-score_values[:, score_index], kind="stable")
        if np.any(np.diff(score_values[order, score_index]) > 0):
            raise RuntimeError(f"Descending subject order failed for {score}")
        ranked_subjects = subjects[order]
        prefix = score.removesuffix("_score")
        network_stem = output / f"{prefix}_ranked_network_attribution"
        atom_stem = output / f"{prefix}_ranked_atom_contributions"
        ranked_overview(
            score_z[order],
            network_flat[order],
            ranked_subjects,
            ordering_label=score_label,
            ordering_score_index=score_index,
            group_size=len(NETWORKS),
            cmap="YlGnBu",
            vmin=0.0,
            vmax=network_vmax,
            colorbar_label=r"Subject-level share of system-level $\Xi$ (%)",
            output_stem=network_stem,
        )
        ranked_overview(
            score_z[order],
            atom_flat[order],
            ranked_subjects,
            ordering_label=score_label,
            ordering_score_index=score_index,
            group_size=len(atom_names),
            cmap="magma_r",
            vmin=0.0,
            vmax=atom_vmax,
            colorbar_label="Subject-level hierarchy-atom contribution (bits)",
            output_stem=atom_stem,
        )
        ranked_outputs[score] = {
            "ordering": "descending",
            "subjects": ranked_subjects.tolist(),
            "network_attribution": network_stem.with_suffix(".png").name,
            "atom_contributions": atom_stem.with_suffix(".png").name,
        }

    network_r, network_p, network_q = correlation_family(score_values, network_flat)
    atom_r, atom_p, atom_q = correlation_family(score_values, atom_flat)
    plot_correlations(
        network_r,
        atom_r,
        output_stem=output / "descriptive_brain_cognition_spearman",
    )

    network_feature_labels = [
        f"{state}:{network}"
        for state in STATES
        for network in NETWORKS
    ]
    atom_feature_labels = [
        f"{state}:{atom}"
        for state in STATES
        for atom in atom_names
    ]
    summary = {
        "analysis": "Subject-level Xi attribution and cognition visual audit",
        "subjects": subjects.tolist(),
        "subject_count": int(len(subjects)),
        "states": list(STATES),
        "networks": list(NETWORKS),
        "selected_atom_indices": np.asarray(data["selected_atom_indices"]).astype(int).tolist(),
        "selected_atom_names": atom_names.tolist(),
        "shared_scales": {
            "network_share_percent_vmin": 0.0,
            "network_share_percent_vmax": network_vmax,
            "atom_bits_vmin": 0.0,
            "atom_bits_vmax_99_5_percentile": atom_vmax,
        },
        "multiple_testing": {
            "procedure": "BH separately within each cognition score and feature family",
            "network_tests_per_score": int(network_flat.shape[1]),
            "atom_valid_tests_per_score": [int(np.isfinite(atom_p[index]).sum()) for index in range(len(SCORES))],
            "network_q_lt_0_05_count": int(np.nansum(network_q < 0.05)),
            "atom_q_lt_0_05_count": int(np.nansum(atom_q < 0.05)),
        },
        "strongest_descriptive_network_association": strongest_result(
            network_r, network_p, network_q, network_feature_labels
        ),
        "strongest_descriptive_atom_association": strongest_result(
            atom_r, atom_p, atom_q, atom_feature_labels
        ),
        "outputs": {
            "individual_multipage_pdf": str(multipage_path.relative_to(output)),
            "individual_pngs": individual_paths,
            "ranked_overviews": ranked_outputs,
            "descriptive_correlations": "descriptive_brain_cognition_spearman.png",
        },
        "interpretation_boundary": (
            "Correlations are unadjusted for demographics, motion, signal quality, and family structure; "
            "the heatmaps are descriptive and do not establish a stable brain-cognition association."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    run(args)
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
