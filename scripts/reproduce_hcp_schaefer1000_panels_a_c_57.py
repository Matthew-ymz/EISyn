#!/usr/bin/env python3
"""Reproduce HCP Schaefer-1000 main-figure panels A-C with 57 subjects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT / "results/hcp_schaefer1000_task_evoked_xi_replication/full/k1_p3_a1"
NEW_ROOT = ROOT / "results/hcp_schaefer1000_task_evoked_xi_57/full/k1_p3_a1"
DEFAULT_OUTPUT = ROOT / "results/hcp_schaefer1000_task_evoked_xi_57/final"
DEFAULT_RECORDS = ROOT / "results/hcp_schaefer1000_task_evoked_xi_57/full/records.jsonl"

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
NETWORK_LABELS = (
    "Visual",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Control",
    "Default",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    parser.add_argument("--new-root", type=Path, default=NEW_ROOT)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def compact_atom(name: str) -> str:
    mapping = {
        "Vis": "V",
        "SomMot": "SM",
        "DorsAttn": "DAN",
        "SalVentAttn": "VAN",
        "Limbic": "Lim",
        "Cont": "FPN",
        "Default": "DMN",
    }
    return "+".join(mapping[item] for item in name.split("+"))


def significance_stars(q_value: float) -> str:
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return "ns"


def load_result(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    archive = np.load(root / "arrays.npz")
    arrays = {key: np.asarray(archive[key]) for key in archive.files}
    return summary, arrays


def validate_arrays(arrays: dict[str, np.ndarray], expected_subjects: int) -> None:
    expected = {
        "system_xi": (8, expected_subjects),
        "network_share": (8, expected_subjects, 7),
        "atom_value": (8, expected_subjects, 120),
        "atom_share": (8, expected_subjects, 120),
    }
    for key, shape in expected.items():
        if key not in arrays or arrays[key].shape != shape:
            raise ValueError(f"Expected {key} shape {shape}, got {arrays.get(key, np.array([])).shape}")
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    closure = np.max(np.abs(arrays["network_share"].sum(axis=2) - 1.0))
    if closure > 1.0e-10:
        raise ValueError(f"Network-share closure error {closure:.3e} exceeds tolerance")


def bh_adjust(values: list[float]) -> list[float]:
    p_values = np.asarray(values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted.tolist()


def max_condition_sensitivity(
    records_path: Path, arrays: dict[str, np.ndarray]
) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    worst = max(
        records,
        key=lambda row: float(row["quality_diagnostics"]["noise_covariance_condition"]),
    )
    subject = str(worst["subject"])
    subjects = arrays["subjects"].astype(str)
    matches = np.flatnonzero(subjects == subject)
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely locate maximum-condition subject {subject}")
    keep = np.arange(len(subjects)) != int(matches[0])
    rows = []
    p_values = []
    for state_index, state in enumerate(arrays["states"].astype(str)[1:], start=1):
        difference = arrays["system_xi"][0, keep] - arrays["system_xi"][state_index, keep]
        p_value = float(wilcoxon(difference).pvalue)
        p_values.append(p_value)
        rows.append(
            {
                "task": state,
                "rest_minus_task_mean_bits": float(difference.mean()),
                "rest_greater_fraction": float(np.mean(difference > 0.0)),
                "p": p_value,
            }
        )
    for row, q_value in zip(rows, bh_adjust(p_values), strict=True):
        row["q"] = float(q_value)
    return {
        "excluded_subject": subject,
        "trigger_state": str(worst["state"]),
        "maximum_noise_covariance_condition": float(
            worst["quality_diagnostics"]["noise_covariance_condition"]
        ),
        "trigger_state_system_xi_bits": float(worst["system_xi"]),
        "all_rest_task_differences_positive": bool(
            all(row["rest_minus_task_mean_bits"] > 0 for row in rows)
        ),
        "all_rest_task_tests_bh_significant": bool(all(row["q"] < 0.05 for row in rows)),
        "rest_system_tests": rows,
    }


def comparison_summary(
    old_summary: dict[str, Any],
    old: dict[str, np.ndarray],
    new_summary: dict[str, Any],
    new: dict[str, np.ndarray],
    selected_old: np.ndarray,
    selected_new: np.ndarray,
    sensitivity: dict[str, Any],
) -> dict[str, Any]:
    states = new["states"].astype(str)
    old_system_mean = old["system_xi"].mean(axis=1)
    new_system_mean = new["system_xi"].mean(axis=1)
    old_atom_mean = old["atom_value"].mean(axis=1)
    new_atom_mean = new["atom_value"].mean(axis=1)
    old_network_mean = old["network_share"].mean(axis=1)
    new_network_mean = new["network_share"].mean(axis=1)
    names = new["atom_names"].astype(str)

    panel_a_rows = []
    old_tests = {row["task"]: row for row in old_summary["rest_system_tests"]}
    new_tests = {row["task"]: row for row in new_summary["rest_system_tests"]}
    for index, state in enumerate(states):
        row: dict[str, Any] = {
            "state": state,
            "mean_bits_n29": float(old_system_mean[index]),
            "mean_bits_n57": float(new_system_mean[index]),
            "mean_change_bits": float(new_system_mean[index] - old_system_mean[index]),
        }
        if state != "REST":
            row.update(
                {
                    "rest_minus_task_mean_bits_n29": float(
                        old_tests[state]["rest_minus_task_mean_bits"]
                    ),
                    "rest_minus_task_mean_bits_n57": float(
                        new_tests[state]["rest_minus_task_mean_bits"]
                    ),
                    "rest_greater_fraction_n57": float(
                        new_tests[state]["rest_greater_fraction"]
                    ),
                    "bh_q_n29": float(old_tests[state]["q"]),
                    "bh_q_n57": float(new_tests[state]["q"]),
                }
            )
        panel_a_rows.append(row)

    return {
        "experiment": {
            "old_n": int(old_summary["n_subjects"]),
            "new_n": int(new_summary["n_subjects"]),
            "model": new_summary["params"],
            "states": states.tolist(),
        },
        "panel_a": {
            "all_rest_task_differences_positive_n57": bool(
                all(row["rest_minus_task_mean_bits"] > 0 for row in new_tests.values())
            ),
            "all_rest_task_tests_bh_significant_n57": bool(
                all(row["q"] < 0.05 for row in new_tests.values())
            ),
            "state_mean_spearman_n29_vs_n57": float(
                spearmanr(old_system_mean, new_system_mean).statistic
            ),
            "states": panel_a_rows,
        },
        "panel_b": {
            "full_atom_matrix_spearman_n29_vs_n57": float(
                spearmanr(old_atom_mean.ravel(), new_atom_mean.ravel()).statistic
            ),
            "mean_absolute_change_bits": float(
                np.mean(np.abs(new_atom_mean - old_atom_mean))
            ),
            "top12_overlap_count": int(len(set(selected_old) & set(selected_new))),
            "top12_n29": names[selected_old].tolist(),
            "top12_n57": names[selected_new].tolist(),
            "significant_atom_features_across_tasks_bh_n29": int(
                old_summary["significance"]["atom_features_tasks"]
            ),
            "significant_atom_features_across_tasks_bh_n57": int(
                new_summary["significance"]["atom_features_tasks"]
            ),
        },
        "panel_c": {
            "network_matrix_spearman_n29_vs_n57": float(
                spearmanr(old_network_mean.ravel(), new_network_mean.ravel()).statistic
            ),
            "mean_absolute_change_percentage_points": float(
                np.mean(np.abs(new_network_mean - old_network_mean)) * 100.0
            ),
            "maximum_absolute_change_percentage_points": float(
                np.max(np.abs(new_network_mean - old_network_mean)) * 100.0
            ),
            "significant_networks_across_tasks_bh_n29": int(
                old_summary["significance"]["network_features_tasks"]
            ),
            "significant_networks_across_tasks_bh_n57": int(
                new_summary["significance"]["network_features_tasks"]
            ),
            "maximum_subject_level_closure_error": float(
                np.max(np.abs(new["network_share"].sum(axis=2) - 1.0))
            ),
        },
        "diagnostics_n57": new_summary["diagnostics"],
        "predictive_diagnostics_n57": {
            "models_better_than_persistence": int(
                new_summary["models_better_than_persistence"]
            ),
            "n_models": int(new_summary["n_models"]),
            "heldout_skill_ratio_mean": float(new_summary["heldout_skill_ratio_mean"]),
        },
        "sensitivity_max_condition_subject_excluded": sensitivity,
    }


def plot_panels(
    summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
    selected: np.ndarray,
    output_dir: Path,
) -> None:
    configure_style()
    states = arrays["states"].astype(str).tolist()
    values = arrays["system_xi"].T
    atom_names = arrays["atom_names"].astype(str)
    atom_panel = arrays["atom_value"].mean(axis=1)[:, selected].T
    network_panel = arrays["network_share"].mean(axis=1).T * 100.0

    figure = plt.figure(figsize=(14.4, 7.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(0.82, 1.18), width_ratios=(1.42, 1.0))
    axis_a = figure.add_subplot(grid[0, :])
    axis_b = figure.add_subplot(grid[1, 0])
    axis_c = figure.add_subplot(grid[1, 1])

    rest_color, task_color = "#4C78A8", "#D07A3A"
    colors = [rest_color] + [task_color] * 7
    positions = np.arange(8, dtype=float)
    boxes = axis_a.boxplot(
        values,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#303030", "linewidth": 1.1},
        whiskerprops={"color": "#7B8490", "linewidth": 0.75},
        capprops={"color": "#7B8490", "linewidth": 0.75},
    )
    for patch, color in zip(boxes["boxes"], colors, strict=True):
        patch.set(facecolor=color, alpha=0.18, edgecolor=color, linewidth=1.0)
    rng = np.random.default_rng(20260719)
    for index, color in enumerate(colors):
        jitter = rng.uniform(-0.13, 0.13, size=values.shape[0])
        axis_a.scatter(
            positions[index] + jitter,
            values[:, index],
            s=13,
            color=color,
            alpha=0.72,
            linewidths=0,
            zorder=3,
        )
    axis_a.scatter(
        positions,
        values.mean(axis=0),
        marker="D",
        s=21,
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
        axis_a.text(
            index,
            star_y,
            significance_stars(float(tests[state]["q"])),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    axis_a.axvline(0.5, color="#A7ADB5", linewidth=0.75, linestyle="--", zorder=0)
    axis_a.set(
        xticks=positions,
        xticklabels=STATE_LABELS,
        xlim=(-0.55, 7.45),
        ylim=(data_min - 0.08 * span, star_y + 0.11 * span),
        ylabel=r"System-level $\Xi$ (bits)",
        xlabel="State",
    )
    axis_a.tick_params(axis="x", labelrotation=22, labelsize=8)
    axis_a.text(
        0.99,
        1.025,
        "paired n=57 · vs REST: Wilcoxon, BH-corrected · white diamond: mean",
        transform=axis_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#454545",
        clip_on=False,
    )
    axis_a.text(
        0.01,
        0.02,
        "*** q<0.001   ** q<0.01   * q<0.05",
        transform=axis_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.1,
        color="#454545",
    )

    atom_upper = max(float(np.quantile(atom_panel, 0.995)), 0.1)
    image_b = axis_b.imshow(
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
        xlabel="State",
        ylabel="Greedy hierarchy atom",
    )
    axis_b.tick_params(axis="x", labelrotation=34, length=0)
    axis_b.tick_params(axis="y", length=0, labelsize=6.5)
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
                fontsize=5.3,
                color="white" if value > 0.38 * atom_upper else "black",
            )
    colorbar_b = figure.colorbar(image_b, ax=axis_b, fraction=0.035, pad=0.025, aspect=32)
    colorbar_b.set_label("Contribution (bits)")

    lower = float(np.floor(network_panel.min()))
    upper = float(np.ceil(network_panel.max()))
    axis_c.imshow(
        network_panel,
        cmap="YlGnBu",
        vmin=lower,
        vmax=upper,
        aspect="auto",
        interpolation="nearest",
    )
    axis_c.set(
        xticks=np.arange(8),
        xticklabels=STATE_LABELS,
        yticks=np.arange(7),
        yticklabels=NETWORK_LABELS,
        xlabel="State (each column sums to 100%)",
        ylabel="Yeo7 network",
    )
    axis_c.tick_params(axis="x", labelrotation=34, length=0)
    axis_c.tick_params(axis="y", length=0)
    axis_c.axvline(0.5, color="#333333", linewidth=0.9)
    for row in range(network_panel.shape[0]):
        for column in range(network_panel.shape[1]):
            value = network_panel[row, column]
            normalized = (value - lower) / max(upper - lower, 1.0e-12)
            axis_c.text(
                column,
                row,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=6.1,
                color="white" if normalized > 0.6 else "black",
            )

    for label, axis, x_position in (
        ("a", axis_a, -0.05),
        ("b", axis_b, -0.08),
        ("c", axis_c, -0.09),
    ):
        axis.text(
            x_position,
            1.025,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10.5,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(
            output_dir / f"hcp_schaefer1000_panels_a_c_57.{suffix}",
            dpi=600,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> int:
    args = parse_args()
    old_summary, old = load_result(args.old_root)
    new_summary, new = load_result(args.new_root)
    validate_arrays(old, 29)
    validate_arrays(new, 57)
    for key in ("states", "networks", "atom_names"):
        if not np.array_equal(old[key].astype(str), new[key].astype(str)):
            raise ValueError(f"Old and new {key} do not align")

    selected_old = np.argsort(old["atom_share"].mean(axis=1).mean(axis=0))[::-1][:12]
    selected_new = np.argsort(new["atom_share"].mean(axis=1).mean(axis=0))[::-1][:12]
    sensitivity = max_condition_sensitivity(args.records, new)
    comparison = comparison_summary(
        old_summary,
        old,
        new_summary,
        new,
        selected_old,
        selected_new,
        sensitivity,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "panels_a_c_comparison_summary.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_panels(new_summary, new, selected_new, args.output_dir)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
