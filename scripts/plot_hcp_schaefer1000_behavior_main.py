#!/usr/bin/env python3
"""Build the eight-panel 57-subject HCP main figure and attribution supplement."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reproduce_hcp_schaefer1000_panels_a_c_57 import (
    NETWORK_LABELS,
    STATE_LABELS,
    compact_atom,
    significance_stars,
)


MODEL_ROOT = ROOT / "results/hcp_schaefer1000_task_evoked_xi_57/full/k1_p3_a1"
BEHAVIOR_ROOT = ROOT / "results/hcp_language_story_math_coalitions_57"
BEHAVIOR_CSV = ROOT / "data/unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
COALITION_CACHE = BEHAVIOR_ROOT / "language_coalition_synergy_57.npz"
EMOTION_ROOT = ROOT / "results/hcp_emotion_performance_coalitions_57"
EMOTION_CACHE = EMOTION_ROOT / "emotion_rest_coalition_synergy_57.npz"
EMOTION_STATS = EMOTION_ROOT / "emotion_performance_coalition_source_data.tsv"
MOTOR_ROOT = ROOT / "results/hcp_motor_composite_scores_57"
MOTOR_SUMMARY = MOTOR_ROOT / "summary.json"
OUTPUT = ROOT / "results/hcp_schaefer1000_task_evoked_xi_57/final"
MAIN_STEM = "hcp_schaefer1000_behavior_main_57"
EMOTION_PREVIEW_STEM = "hcp_schaefer1000_behavior_main_57_emotion_preview"
EMOTION_TWO_PANEL_STEM = "hcp_emotion_behavior_two_panel_preview_57"
SUPPLEMENT_STEM = "hcp_schaefer1000_attribution_supplement_57"
SCATTER_SPECS = (
    {
        "coalition": "Vis+SomMot+Limbic+Cont",
        "endpoint": "Language_Task_Story_Acc",
        "xlabel": "Story accuracy (%)",
        "title": "LANGUAGE · Story",
        "color": "#D06B4F",
        "seed": 20260802,
    },
    {
        "coalition": "Vis+DorsAttn+Cont",
        "endpoint": "Language_Task_Math_Acc",
        "xlabel": "Math accuracy (%)",
        "title": "LANGUAGE · Math",
        "color": "#D06B4F",
        "seed": 20260802,
    },
)
PERMUTATIONS = 100_000
EMOTION_SCATTER_SPECS = (
    {
        "coalition": "Limbic+Cont",
        "title": "EMOTION · Lim + Cont",
        "color": "#4C78A8",
    },
    {
        "coalition": "Limbic+Cont+Default",
        "title": "EMOTION · Lim + Cont + DMN",
        "color": "#4C78A8",
    },
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(figure: mpl.figure.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(
            OUTPUT / f"{stem}.{suffix}",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def load_inputs() -> tuple[dict, dict[str, np.ndarray], pd.DataFrame]:
    summary = json.loads((MODEL_ROOT / "summary.json").read_text(encoding="utf-8"))
    archive = np.load(MODEL_ROOT / "arrays.npz")
    arrays = {key: np.asarray(archive[key]) for key in archive.files}
    cohort_table = pd.read_csv(
        BEHAVIOR_ROOT / "selected_candidate_source_data.tsv", sep="\t"
    ).set_index("subject")
    coalition_archive = np.load(COALITION_CACHE)
    subjects = coalition_archive["subjects"].astype(str)
    coalition_names = coalition_archive["coalitions"].astype(str).tolist()
    if set(subjects) != set(arrays["subjects"].astype(str)):
        raise ValueError("Coalition cache and frozen model use different subjects")
    raw_behavior = pd.read_csv(BEHAVIOR_CSV, dtype={"Subject": str}).set_index("Subject")
    subject_ids = [subject.removeprefix("sub-") for subject in subjects]
    behavior = raw_behavior.loc[subject_ids].copy()
    behavior.insert(0, "subject", subjects)
    behavior["cohort"] = cohort_table.loc[subjects, "cohort"].to_numpy()
    for spec in SCATTER_SPECS:
        coalition = str(spec["coalition"])
        index = coalition_names.index(coalition)
        behavior[f"synergy_bits__{coalition}"] = coalition_archive[
            "synergy_bits"
        ][:, index]
    if len(behavior) != 57 or arrays["system_xi"].shape != (8, 57):
        raise ValueError("Expected the frozen 57-subject inputs")
    tolerance = 1.0e-9
    for spec in SCATTER_SPECS:
        coalition = str(spec["coalition"])
        synergy = behavior[f"synergy_bits__{coalition}"].to_numpy(dtype=float)
        violating = synergy < -tolerance
        if np.any(violating):
            raise ValueError(
                "PEID Syn nonnegativity violation: "
                f"coalition={coalition}, minimum={synergy.min():.6g}, "
                f"tolerance={tolerance:.1e}, count={int(violating.sum())}"
            )
    source_columns = ["subject", "cohort"]
    for spec in SCATTER_SPECS:
        source_columns.extend(
            [str(spec["endpoint"]), f"synergy_bits__{spec['coalition']}"]
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    behavior[source_columns].to_csv(
        OUTPUT / f"{MAIN_STEM}_source_data.tsv", sep="\t", index=False
    )
    return summary, arrays, behavior


def age_midpoint(value: str) -> float:
    if value == "36+":
        return 38.0
    low, high = value.split("-")
    return 0.5 * (float(low) + float(high))


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def load_emotion_behavior(arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    with np.load(EMOTION_CACHE, allow_pickle=False) as archive:
        subjects = archive["subjects"].astype(str)
        coalitions = archive["coalitions"].astype(str).tolist()
        states = archive["states"].astype(str).tolist()
        synergy = archive["synergy_bits"].astype(float)
    frozen_subjects = arrays["subjects"].astype(str)
    if not np.array_equal(subjects, frozen_subjects):
        raise ValueError("EMOTION coalition cache and main figure use different subject order")
    emotion_index = states.index("EMOTION")
    raw_behavior = pd.read_csv(BEHAVIOR_CSV, dtype={"Subject": str}).set_index("Subject")
    subject_ids = [subject.removeprefix("sub-") for subject in subjects]
    frame = raw_behavior.loc[subject_ids].copy()
    frame.insert(0, "subject", subjects)
    frame["cohort"] = np.where(np.arange(len(subjects)) < 29, "Original 29", "Supplementary 28")
    cohort_code = (np.arange(len(subjects)) >= 29).astype(float)
    age = np.asarray([age_midpoint(value) for value in frame["Age"].astype(str)])
    sex = (frame["Gender"].astype(str).to_numpy() == "M").astype(float)
    face_speed = -np.log(frame["Emotion_Task_Face_Median_RT"].to_numpy(dtype=float))
    shape_speed = -np.log(frame["Emotion_Task_Shape_Median_RT"].to_numpy(dtype=float))
    design = np.column_stack(
        [np.ones(len(frame)), rankdata(age), sex, cohort_code, rankdata(shape_speed)]
    )
    frame["face_speed_residual_rank"] = residualize(rankdata(face_speed), design)
    statistics = pd.read_csv(EMOTION_STATS, sep="\t")
    statistics = statistics.loc[statistics["analysis"] == "task_face_specific_speed"].set_index("coalition")
    for spec in EMOTION_SCATTER_SPECS:
        coalition = str(spec["coalition"])
        coalition_index = coalitions.index(coalition)
        raw = synergy[emotion_index, :, coalition_index]
        if np.any(raw < -1.0e-9):
            raise ValueError(
                f"PEID Syn nonnegativity violation for {coalition}: min={raw.min():.6g}"
            )
        frame[f"synergy_bits__{coalition}"] = raw
        frame[f"synergy_residual_rank__{coalition}"] = residualize(
            rankdata(raw), design
        )
        frame.attrs[f"rho__{coalition}"] = float(statistics.loc[coalition, "rho"])
        frame.attrs[f"p_raw__{coalition}"] = float(statistics.loc[coalition, "p_raw"])
    source_columns = [
        "subject",
        "cohort",
        "Emotion_Task_Face_Median_RT",
        "Emotion_Task_Shape_Median_RT",
        "face_speed_residual_rank",
    ]
    for spec in EMOTION_SCATTER_SPECS:
        source_columns.extend(
            [
                f"synergy_bits__{spec['coalition']}",
                f"synergy_residual_rank__{spec['coalition']}",
            ]
        )
    frame[source_columns].to_csv(
        OUTPUT / f"{EMOTION_PREVIEW_STEM}_source_data.tsv", sep="\t", index=False
    )
    return frame


def load_motor_associations() -> list[dict[str, float | int | str | list[float]]]:
    summary = json.loads(MOTOR_SUMMARY.read_text(encoding="utf-8"))
    retained = [
        row
        for row in summary["top_ten"]
        if float(row["rho_adjusted"]) < 0 and float(row["p_raw"]) < 0.05
    ]
    if len(retained) != 9:
        raise ValueError(
            "Expected nine negative pointwise-significant MOTOR associations "
            f"after removing the p=0.0501 result; found {len(retained)}."
        )
    source = pd.DataFrame(
        {
            "coalition": [row["coalition"] for row in retained],
            "short_coalition": [row["short_coalition"] for row in retained],
            "rho_adjusted": [float(row["rho_adjusted"]) for row in retained],
            "p_pointwise": [float(row["p_raw"]) for row in retained],
            "bootstrap_95_ci_low": [
                float(row["stratified_bootstrap_quantiles"][0]) for row in retained
            ],
            "bootstrap_95_ci_high": [
                float(row["stratified_bootstrap_quantiles"][2]) for row in retained
            ],
        }
    )
    source.to_csv(
        OUTPUT / f"{MAIN_STEM}_motor_panel_source_data.tsv", sep="\t", index=False
    )
    return retained


def blocked_pointwise_spearman(
    x: np.ndarray,
    y: np.ndarray,
    cohorts: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float]:
    def normalized_ranks(values: np.ndarray) -> np.ndarray:
        ranks = rankdata(values, method="average").astype(float)
        ranks -= ranks.mean()
        return ranks / np.sqrt(np.sum(ranks**2))

    x_rank = normalized_ranks(x)
    y_rank = normalized_ranks(y)
    observed = float(x_rank @ y_rank)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(cohorts == value) for value in np.unique(cohorts)]
    exceedances = 0
    chunk = 1_000
    for start in range(0, PERMUTATIONS, chunk):
        size = min(chunk, PERMUTATIONS - start)
        permutations = np.tile(np.arange(len(x)), (size, 1))
        for indices in groups:
            order = np.argsort(rng.random((size, len(indices))), axis=1)
            permutations[:, indices] = indices[order]
        null = y_rank[permutations] @ x_rank
        exceedances += int(np.sum(np.abs(null) >= abs(observed)))
    return observed, (exceedances + 1) / (PERMUTATIONS + 1)


def plot_system_xi(axis: mpl.axes.Axes, summary: dict, arrays: dict[str, np.ndarray]) -> None:
    states = arrays["states"].astype(str).tolist()
    values = arrays["system_xi"].T
    colors = ["#4C78A8"] + ["#D07A3A"] * 7
    positions = np.arange(8, dtype=float)
    boxes = axis.boxplot(
        values,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#303030", "linewidth": 1.0},
        whiskerprops={"color": "#7B8490", "linewidth": 0.75},
        capprops={"color": "#7B8490", "linewidth": 0.75},
    )
    for patch, color in zip(boxes["boxes"], colors, strict=True):
        patch.set(facecolor=color, alpha=0.18, edgecolor=color, linewidth=0.9)
    rng = np.random.default_rng(20260719)
    for index, color in enumerate(colors):
        jitter = rng.uniform(-0.13, 0.13, size=values.shape[0])
        axis.scatter(
            positions[index] + jitter,
            values[:, index],
            s=10,
            color=color,
            alpha=0.70,
            linewidths=0,
            zorder=3,
        )
    axis.scatter(
        positions,
        values.mean(axis=0),
        marker="D",
        s=18,
        facecolor="white",
        edgecolor="#303030",
        linewidth=0.7,
        zorder=4,
    )
    tests = {str(row["task"]): row for row in summary["rest_system_tests"]}
    data_min, data_max = float(values.min()), float(values.max())
    span = max(data_max - data_min, 1.0)
    star_y = data_max + 0.075 * span
    for index, state in enumerate(states[1:], start=1):
        axis.text(
            index,
            star_y,
            significance_stars(float(tests[state]["q"])),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.axvline(0.5, color="#A7ADB5", linewidth=0.7, linestyle="--", zorder=0)
    axis.set(
        xticks=positions,
        xticklabels=STATE_LABELS,
        xlim=(-0.55, 7.45),
        ylim=(data_min - 0.08 * span, star_y + 0.11 * span),
        ylabel=r"System-level $\Xi$ (bits)",
        xlabel="State",
    )
    axis.tick_params(axis="x", labelrotation=20)
    axis.text(
        0.99,
        1.025,
        "paired n=57 · versus REST: Wilcoxon, BH-corrected · diamond: mean",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.6,
        color="#454545",
        clip_on=False,
    )
    axis.text(
        0.01,
        0.02,
        "*** q<0.001   ** q<0.01   * q<0.05",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color="#454545",
    )


def plot_behavior_scatter(
    axis: mpl.axes.Axes,
    behavior: pd.DataFrame,
    *,
    coalition: str,
    endpoint: str,
    xlabel: str,
    title: str,
    color: str,
    seed: int,
) -> None:
    x = behavior[endpoint].to_numpy(dtype=float)
    y = behavior[f"synergy_bits__{coalition}"].to_numpy(dtype=float)
    jitter_rng = np.random.default_rng(2026080101 if "story" in endpoint else 2026080102)
    jitter = jitter_rng.uniform(-0.015, 0.015, size=len(x)) * max(np.ptp(x), 1.0)
    axis.scatter(
        x + jitter,
        y,
        s=19,
        marker="o",
        color=color,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.86,
        zorder=3,
    )
    guide_x = np.linspace(float(x.min()), float(x.max()), 200)
    slope, intercept = np.polyfit(x, y, deg=1)
    axis.plot(guide_x, slope * guide_x + intercept, color=color, linewidth=1.1, zorder=2)
    x_pad = 0.05 * max(float(np.ptp(x)), 1.0)
    y_pad = 0.08 * max(float(np.ptp(y)), 0.1)
    axis.set(
        xlabel=xlabel,
        xlim=(float(x.min()) - x_pad, float(x.max()) + x_pad),
        ylim=(float(y.min()) - y_pad, float(y.max()) + y_pad),
    )
    axis.grid(color="#E7EAED", linewidth=0.55, zorder=0)
    cohort_codes = pd.Categorical(behavior["cohort"]).codes
    rho, p_value = blocked_pointwise_spearman(
        x, y, cohort_codes, seed=seed
    )
    axis.set_title(
        title,
        loc="left",
        fontsize=7.2,
        fontweight="bold",
        pad=13,
    )
    axis.text(
        0.0,
        1.015,
        rf"$\rho$={rho:+.3f}  ·  $p$={p_value:.3f}",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.6,
        color="#3F4852",
        clip_on=False,
    )


def plot_emotion_scatter(
    axis: mpl.axes.Axes,
    behavior: pd.DataFrame,
    *,
    coalition: str,
    title: str,
    color: str,
) -> None:
    x = behavior["face_speed_residual_rank"].to_numpy(dtype=float)
    y = behavior[f"synergy_residual_rank__{coalition}"].to_numpy(dtype=float)
    axis.scatter(
        x,
        y,
        s=19,
        color=color,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.86,
        zorder=3,
    )
    guide_x = np.linspace(float(x.min()), float(x.max()), 200)
    slope, intercept = np.polyfit(x, y, deg=1)
    axis.plot(guide_x, slope * guide_x + intercept, color=color, linewidth=1.1, zorder=2)
    x_pad = 0.06 * max(float(np.ptp(x)), 1.0)
    y_pad = 0.08 * max(float(np.ptp(y)), 1.0)
    axis.set(
        xlabel="Face-specific speed\n(higher = faster)",
        ylabel="Coalition synergy (residualized rank)",
        xlim=(float(x.min()) - x_pad, float(x.max()) + x_pad),
        ylim=(float(y.min()) - y_pad, float(y.max()) + y_pad),
    )
    axis.grid(color="#E7EAED", linewidth=0.55, zorder=0)
    rho = float(behavior.attrs[f"rho__{coalition}"])
    p_value = float(behavior.attrs[f"p_raw__{coalition}"])
    axis.set_title(title, loc="left", fontsize=7.2, fontweight="bold", pad=13)
    axis.text(
        0.0,
        1.015,
        rf"$\rho$={rho:+.3f}  ·  $p$={p_value:.3f}",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.6,
        color="#3F4852",
        clip_on=False,
    )


def plot_motor_associations(
    axis: mpl.axes.Axes,
    rows: list[dict[str, float | int | str | list[float]]],
) -> None:
    ordered = list(reversed(rows))
    positions = np.arange(len(ordered))
    centers = np.asarray([float(row["rho_adjusted"]) for row in ordered])
    intervals = np.asarray(
        [row["stratified_bootstrap_quantiles"] for row in ordered], dtype=float
    )
    axis.axvline(0, color="#D6DADF", linewidth=0.7)
    axis.errorbar(
        centers,
        positions,
        xerr=np.vstack([centers - intervals[:, 0], intervals[:, 2] - centers]),
        fmt="o",
        markersize=4.0,
        color="#526D82",
        ecolor="#9AA8B2",
        elinewidth=0.85,
        capsize=1.9,
    )
    axis.scatter(
        centers[-1], positions[-1], s=38, marker="D", color="#D07A55", zorder=4
    )
    for position, row in zip(positions, ordered, strict=True):
        axis.text(
            0.025,
            position,
            rf"$\rho$={float(row['rho_adjusted']):+.3f}; "
            rf"$p$={float(row['p_raw']):.4f}",
            ha="left",
            va="center",
            fontsize=6.1,
            color="#9A4D32",
            fontweight="bold",
        )
    main_abbreviation = {
        "Vis": "V",
        "SomMot": "SM",
        "DorsAttn": "DAN",
        "SalVentAttn": "VAN",
        "Limbic": "L",
        "Cont": "C",
        "Default": "D",
    }
    axis.set(
        yticks=positions,
        yticklabels=[
            "+".join(main_abbreviation[item] for item in str(row["coalition"]).split("+"))
            for row in ordered
        ],
        xlim=(-0.64, 0.30),
        xlabel=r"Adjusted $\rho$ (stratified bootstrap 95% CI)",
    )
    axis.tick_params(axis="y", labelsize=6.2)
    axis.set_title(
        "MOTOR",
        loc="left",
        fontsize=7.2,
        fontweight="bold",
        pad=7,
    )


def plot_main(
    summary: dict,
    arrays: dict[str, np.ndarray],
    behavior: pd.DataFrame,
    *,
    emotion_behavior: pd.DataFrame | None = None,
    motor_associations: list[dict[str, float | int | str | list[float]]] | None = None,
    stem: str = MAIN_STEM,
) -> None:
    atom_names = arrays["atom_names"].astype(str)
    selected = np.argsort(arrays["atom_share"].mean(axis=1).mean(axis=0))[::-1][:12]
    atom_panel = arrays["atom_value"].mean(axis=1)[:, selected].T
    network_panel = arrays["network_share"].mean(axis=1).T * 100.0

    include_emotion = emotion_behavior is not None
    include_motor = motor_associations is not None
    if include_motor and not include_emotion:
        raise ValueError("The main layout expects EMOTION panels when MOTOR is included.")
    figure = plt.figure(
        figsize=(7.6, 11.4 if include_motor else (9.35 if include_emotion else 7.6)),
        constrained_layout=True,
    )
    row_count = 3 if include_motor else (4 if include_emotion else 3)
    height_ratios = (
        (0.62, 1.05, 2.05)
        if include_motor
        else ((0.72, 1.05, 0.90, 0.90) if include_emotion else (0.72, 1.05, 0.90))
    )
    outer_grid = figure.add_gridspec(
        row_count,
        1,
        height_ratios=height_ratios,
        hspace=0.34 if include_motor else 0.16,
    )
    middle_grid = outer_grid[1, 0].subgridspec(
        1, 2, width_ratios=(1.34, 1.0), wspace=0.28
    )
    axis_a = figure.add_subplot(outer_grid[0, 0])
    axis_b = figure.add_subplot(middle_grid[0, 0])
    axis_c = figure.add_subplot(middle_grid[0, 1])
    if include_motor:
        behavior_grid = outer_grid[2, 0].subgridspec(
            2,
            3,
            width_ratios=(1.0, 1.0, 1.28),
            height_ratios=(1.0, 1.0),
            wspace=0.58,
            hspace=0.42,
        )
        axis_d = figure.add_subplot(behavior_grid[0, 0])
        axis_e = figure.add_subplot(behavior_grid[0, 1])
        axis_f = figure.add_subplot(behavior_grid[1, 0])
        axis_g = figure.add_subplot(behavior_grid[1, 1])
        axis_h = figure.add_subplot(behavior_grid[:, 2])
    else:
        bottom_grid = outer_grid[2, 0].subgridspec(1, 2, wspace=0.22)
        emotion_grid = (
            outer_grid[3, 0].subgridspec(1, 2, wspace=0.22)
            if include_emotion
            else None
        )
        axis_d = figure.add_subplot(bottom_grid[0, 0])
        axis_e = figure.add_subplot(bottom_grid[0, 1])
        axis_f = (
            figure.add_subplot(emotion_grid[0, 0]) if emotion_grid is not None else None
        )
        axis_g = (
            figure.add_subplot(emotion_grid[0, 1]) if emotion_grid is not None else None
        )
        axis_h = None
    plot_system_xi(axis_a, summary, arrays)

    atom_upper = max(float(np.quantile(atom_panel, 0.995)), 0.1)
    atom_image = axis_b.imshow(
        atom_panel,
        cmap="magma_r",
        vmin=0.0,
        vmax=atom_upper,
        aspect="auto",
        interpolation="nearest",
    )
    axis_b.set(
        xticks=np.arange(8),
        xticklabels=STATE_LABELS,
        yticks=np.arange(len(selected)),
        yticklabels=[compact_atom(atom_names[index]) for index in selected],
        xlabel="",
        ylabel="Greedy hierarchy atom",
    )
    axis_b.tick_params(axis="x", labelrotation=34, length=0, labelsize=6.6)
    axis_b.tick_params(axis="y", length=0, labelsize=5.5)
    axis_b.axvline(0.5, color="#F0F0F0", linewidth=0.9)
    for row in range(atom_panel.shape[0]):
        for column in range(atom_panel.shape[1]):
            value = atom_panel[row, column]
            axis_b.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=4.1,
                color="white" if value > 0.38 * atom_upper else "black",
            )
    atom_colorbar = figure.colorbar(
        atom_image, ax=axis_b, fraction=0.032, pad=0.022, aspect=32
    )
    atom_colorbar.set_label("")
    atom_colorbar.ax.tick_params(labelsize=6.0)

    network_lower = float(np.floor(network_panel.min()))
    network_upper = float(np.ceil(network_panel.max()))
    axis_c.imshow(
        network_panel,
        cmap="YlGnBu",
        vmin=network_lower,
        vmax=network_upper,
        aspect="auto",
        interpolation="nearest",
    )
    axis_c.set(
        xticks=np.arange(8),
        xticklabels=STATE_LABELS,
        yticks=np.arange(7),
        yticklabels=NETWORK_LABELS,
        xlabel="",
        ylabel="",
    )
    axis_c.tick_params(axis="x", labelrotation=38, length=0, labelsize=6.1)
    axis_c.tick_params(axis="y", length=0, labelsize=6.1)
    axis_c.axvline(0.5, color="#333333", linewidth=0.9)
    for row in range(network_panel.shape[0]):
        for column in range(network_panel.shape[1]):
            value = network_panel[row, column]
            normalized = (value - network_lower) / max(
                network_upper - network_lower, 1.0e-12
            )
            axis_c.text(
                column,
                row,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=4.7,
                color="white" if normalized > 0.6 else "black",
            )

    plot_behavior_scatter(axis_d, behavior, **SCATTER_SPECS[0])
    plot_behavior_scatter(axis_e, behavior, **SCATTER_SPECS[1])
    axis_d.set_ylabel("Syn (bits)")
    axis_e.set_ylabel("")
    panel_labels: list[tuple[str, mpl.axes.Axes, float, float]] = [
        ("a", axis_a, -0.06, 1.04),
        ("b", axis_b, -0.11, 1.03),
        ("c", axis_c, -0.12, 1.03),
        ("d", axis_d, -0.18 if include_motor else -0.11, 1.14),
        ("e", axis_e, -0.14 if include_motor else -0.08, 1.14),
    ]
    if include_emotion:
        assert emotion_behavior is not None and axis_f is not None and axis_g is not None
        plot_emotion_scatter(axis_f, emotion_behavior, **EMOTION_SCATTER_SPECS[0])
        plot_emotion_scatter(axis_g, emotion_behavior, **EMOTION_SCATTER_SPECS[1])
        axis_f.set_ylabel("Syn (residualized rank)")
        axis_g.set_ylabel("")
        panel_labels.extend(
            [
                ("f", axis_f, -0.18 if include_motor else -0.11, 1.14),
                ("g", axis_g, -0.14 if include_motor else -0.08, 1.14),
            ]
        )
    if include_motor:
        assert motor_associations is not None and axis_h is not None
        plot_motor_associations(axis_h, motor_associations)
        panel_labels.append(("h", axis_h, -0.06, 1.045))
    for label, axis, x_position, y_position in panel_labels:
        axis.text(
            x_position,
            y_position,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9.5,
        )
    save_figure(figure, stem)


def plot_emotion_two_panel(behavior: pd.DataFrame) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.15), constrained_layout=True
    )
    for axis, spec in zip(axes, EMOTION_SCATTER_SPECS, strict=True):
        plot_emotion_scatter(axis, behavior, **spec)
    for label, axis in zip(("f", "g"), axes, strict=True):
        axis.text(
            -0.11,
            1.14,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9.5,
        )
    save_figure(figure, EMOTION_TWO_PANEL_STEM)


def plot_supplement(arrays: dict[str, np.ndarray]) -> None:
    atom_names = arrays["atom_names"].astype(str)
    selected = np.argsort(arrays["atom_share"].mean(axis=1).mean(axis=0))[::-1][:12]
    atom_panel = arrays["atom_value"].mean(axis=1)[:, selected].T
    network_panel = arrays["network_share"].mean(axis=1).T * 100.0
    figure, (axis_a, axis_b) = plt.subplots(
        1, 2, figsize=(7.2, 3.25), constrained_layout=True, gridspec_kw={"width_ratios": (1.32, 1.0)}
    )
    atom_upper = max(float(np.quantile(atom_panel, 0.995)), 0.1)
    image = axis_a.imshow(atom_panel, cmap="magma_r", vmin=0.0, vmax=atom_upper, aspect="auto")
    axis_a.set(
        xticks=np.arange(8),
        xticklabels=STATE_LABELS,
        yticks=np.arange(len(selected)),
        yticklabels=[compact_atom(atom_names[index]) for index in selected],
        xlabel="State",
        ylabel="Greedy hierarchy atom",
    )
    axis_a.tick_params(axis="x", labelrotation=34, length=0)
    axis_a.tick_params(axis="y", length=0, labelsize=5.9)
    axis_a.axvline(0.5, color="#F0F0F0", linewidth=0.9)
    for row in range(atom_panel.shape[0]):
        for column in range(atom_panel.shape[1]):
            value = atom_panel[row, column]
            axis_a.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=4.5, color="white" if value > 0.38 * atom_upper else "black")
    colorbar = figure.colorbar(image, ax=axis_a, fraction=0.035, pad=0.025, aspect=30)
    colorbar.set_label("Contribution (bits)", fontsize=6.7)
    lower, upper = float(np.floor(network_panel.min())), float(np.ceil(network_panel.max()))
    axis_b.imshow(network_panel, cmap="YlGnBu", vmin=lower, vmax=upper, aspect="auto")
    axis_b.set(
        xticks=np.arange(8),
        xticklabels=STATE_LABELS,
        yticks=np.arange(7),
        yticklabels=NETWORK_LABELS,
        xlabel="State (columns sum to 100%)",
        ylabel="Yeo7 network",
    )
    axis_b.tick_params(axis="x", labelrotation=34, length=0)
    axis_b.tick_params(axis="y", length=0, labelsize=6.3)
    axis_b.axvline(0.5, color="#333333", linewidth=0.9)
    for row in range(network_panel.shape[0]):
        for column in range(network_panel.shape[1]):
            value = network_panel[row, column]
            normalized = (value - lower) / max(upper - lower, 1.0e-12)
            axis_b.text(column, row, f"{value:.1f}%", ha="center", va="center", fontsize=5.1, color="white" if normalized > 0.6 else "black")
    for label, axis, x_position in (("a", axis_a, -0.12), ("b", axis_b, -0.12)):
        axis.text(x_position, 1.03, label, transform=axis.transAxes, fontweight="bold", fontsize=9.5)
    save_figure(figure, SUPPLEMENT_STEM)


def main() -> int:
    configure_style()
    summary, arrays, behavior = load_inputs()
    emotion_behavior = load_emotion_behavior(arrays)
    motor_associations = load_motor_associations()
    # The formal Figure 2 contains state/attribution evidence (a-c), compact
    # LANGUAGE and EMOTION scatter panels (d-g), and MOTOR associations (h).
    plot_main(
        summary,
        arrays,
        behavior,
        emotion_behavior=emotion_behavior,
        motor_associations=motor_associations,
        stem=MAIN_STEM,
    )
    plot_main(
        summary,
        arrays,
        behavior,
        emotion_behavior=emotion_behavior,
        motor_associations=motor_associations,
        stem=EMOTION_PREVIEW_STEM,
    )
    plot_emotion_two_panel(emotion_behavior)
    plot_supplement(arrays)
    print(
        json.dumps(
            {
                "main": str(OUTPUT / f"{MAIN_STEM}.png"),
                "emotion_preview": str(OUTPUT / f"{EMOTION_PREVIEW_STEM}.png"),
                "emotion_two_panel_preview": str(OUTPUT / f"{EMOTION_TWO_PANEL_STEM}.png"),
                "supplement": str(OUTPUT / f"{SUPPLEMENT_STEM}.png"),
                "behavior_panels": [
                    {
                        "coalition": spec["coalition"],
                        "endpoint": spec["endpoint"],
                    }
                    for spec in SCATTER_SPECS
                ],
                "syn_nonnegativity_tolerance_bits": 1.0e-9,
                "emotion_panels": [
                    {
                        "coalition": spec["coalition"],
                        "rho": emotion_behavior.attrs[f"rho__{spec['coalition']}"],
                        "pointwise_p": emotion_behavior.attrs[f"p_raw__{spec['coalition']}"],
                    }
                    for spec in EMOTION_SCATTER_SPECS
                ],
                "motor_panel": {
                    "retained_associations": len(motor_associations),
                    "criterion": "negative rho and uncorrected pointwise permutation p<0.05",
                    "removed": "SalVentAttn+Limbic (rho=-0.277, p=0.0501)",
                },
                "syn_minimum_bits": {
                    spec["coalition"]: float(
                        behavior[f"synergy_bits__{spec['coalition']}"] .min()
                    )
                    for spec in SCATTER_SPECS
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
