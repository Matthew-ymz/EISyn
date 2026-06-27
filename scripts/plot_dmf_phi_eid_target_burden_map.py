from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DMF_MODULE_PATH = ROOT / "exp" / "brain" / "dmf_fig6.py"
DEFAULT_SOURCE_RESULTS = ROOT / "exp" / "brain" / "result_lausanne_fig6" / "count_00_fig6b_mean_rate.npz"
DEFAULT_CONNECTIVITY_CSV = ROOT / "exp" / "brain" / "result_lausanne_fig6" / "count_00_connectivity.csv"
DEFAULT_OUTPUT_BASE = ROOT / "docs" / "reports" / "assets" / "part2_dmf_phi_eid_target_burden_map"
DEFAULT_TABLE = ROOT / "docs" / "reports" / "assets" / "part2_dmf_phi_eid_target_burden.csv"
DEFAULT_RUNGE_PATH_TABLE = ROOT / "docs" / "reports" / "assets" / "part2_dmf_runge_path_scores_g17_g18.csv"
DEFAULT_RUNGE_PATH_OUTPUT_BASE = ROOT / "docs" / "reports" / "assets" / "part2_dmf_runge_path_scores"
DEFAULT_SINGLETON_MATRIX = ROOT / "docs" / "reports" / "assets" / "part2_dmf_phi_eid_singleton_ei_matrix.csv"

DISPLAY_MODULE_ORDER = ["DMN", "Som", "Vis", "VAN", "DAN", "FPN", "Lim", "Sub"]
DISPLAY_MODULE_COLORS = {
    "DMN": "#E53935",
    "Som": "#F5A400",
    "Vis": "#77B255",
    "VAN": "#00A9C8",
    "DAN": "#2F55D4",
    "FPN": "#E056B5",
    "Lim": "#F2C300",
    "Sub": "#B6D63A",
}


def load_dmf_module():
    spec = importlib.util.spec_from_file_location("dmf_fig6_exp_brain", DMF_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_connectivity_labels(path: Path = DEFAULT_CONNECTIVITY_CSV) -> list[str]:
    rows = pd.read_csv(path, header=None)
    return [str(value) for value in rows.iloc[:, 0].tolist()]


def lagged_samples(series: np.ndarray, tau: int) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(series, dtype=float)
    if array.ndim != 2 or array.shape[0] <= tau + 2:
        raise ValueError("series must have shape [time, region] with enough lagged samples.")
    return array[:-tau], array[tau:]


def compute_target_burden_scores(
    transition_matrix: np.ndarray,
    noise_covariance: np.ndarray,
    *,
    labels: list[str] | None = None,
    ridge: float = 1.0e-12,
    log_base: float = np.e,
) -> pd.DataFrame:
    """Compute target-specific PhiEID burden for a linear Gaussian transition.

    Sources are assumed to be independent standardized interventions, matching the
    whole-system PhiEID estimator in exp/brain/dmf_fig6.py.
    """

    transition = np.asarray(transition_matrix, dtype=float)
    noise = np.asarray(noise_covariance, dtype=float)
    if transition.ndim != 2:
        raise ValueError("transition_matrix must be 2D.")
    target_dim, source_dim = transition.shape
    if noise.shape != (target_dim, target_dim):
        raise ValueError(f"noise_covariance must have shape ({target_dim}, {target_dim}).")
    if labels is None:
        labels = [f"region_{index:03d}" for index in range(target_dim)]
    if len(labels) != target_dim:
        raise ValueError(f"Expected {target_dim} labels, got {len(labels)}.")

    rows: list[dict[str, float | int | str]] = []
    noise_var = np.maximum(np.diag(noise), ridge)
    for target_index, row in enumerate(transition):
        target_var = max(float(row @ row + noise_var[target_index]), ridge)
        whole = 0.5 * np.log(target_var / noise_var[target_index]) / np.log(log_base)
        singleton_values = []
        for source_index in range(source_dim):
            conditional_var = max(target_var - float(row[source_index] * row[source_index]), ridge)
            value = 0.5 * np.log(target_var / conditional_var) / np.log(log_base)
            singleton_values.append(max(0.0, float(value)))
        raw = float(whole - np.sum(singleton_values))
        rows.append(
            {
                "region_index": target_index,
                "region": labels[target_index],
                "whole_ei_to_target": max(0.0, float(whole)),
                "singleton_ei_sum_to_target": float(np.sum(singleton_values)),
                "raw_target_burden": raw,
                "target_burden": max(0.0, raw),
                "target_noise_variance": float(noise_var[target_index]),
                "target_signal_variance": float(max(target_var - noise_var[target_index], 0.0)),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("target_burden", ascending=False).reset_index(drop=True)


def compute_singleton_ei_matrix(
    transition_matrix: np.ndarray,
    noise_covariance: np.ndarray,
    *,
    ridge: float = 1.0e-12,
    log_base: float = np.e,
    zero_self_loops: bool = True,
) -> np.ndarray:
    """Return source-to-target singleton Gaussian EI values.

    The returned matrix is indexed as ``[source, target]`` so it can be used as
    a directed EI graph for Runge-style path aggregation.
    """

    transition = np.asarray(transition_matrix, dtype=float)
    noise = np.asarray(noise_covariance, dtype=float)
    if transition.ndim != 2:
        raise ValueError("transition_matrix must be 2D.")
    target_dim, source_dim = transition.shape
    if noise.shape != (target_dim, target_dim):
        raise ValueError(f"noise_covariance must have shape ({target_dim}, {target_dim}).")

    direct = np.zeros((source_dim, target_dim), dtype=float)
    noise_var = np.maximum(np.diag(noise), ridge)
    for target_index, row in enumerate(transition):
        target_var = max(float(row @ row + noise_var[target_index]), ridge)
        for source_index, weight in enumerate(row):
            conditional_var = max(target_var - float(weight * weight), ridge)
            value = 0.5 * np.log(target_var / conditional_var) / np.log(log_base)
            direct[source_index, target_index] = max(0.0, float(value))
    if zero_self_loops and source_dim == target_dim:
        np.fill_diagonal(direct, 0.0)
    return direct


def infer_display_module(label: str) -> str:
    """Map Lausanne/FreeSurfer-style labels to coarse Luppi-style modules."""

    lower = label.lower()
    if any(token in lower for token in ("thalamus", "pallidum", "putamen", "hippocampus", "caudate", "accumbens", "amygdala", "stem")):
        return "Sub"
    if any(token in lower for token in ("cuneus", "lingual", "pericalcarine", "occipital")):
        return "Vis"
    if any(token in lower for token in ("precentral", "postcentral", "paracentral", "transversetemporal")):
        return "Som"
    if any(token in lower for token in ("supramarginal", "superiorparietal", "inferiorparietal", "bankssts")):
        return "VAN"
    if "precuneus" in lower:
        return "DAN"
    if any(token in lower for token in ("superiorfrontal", "middlefrontal", "parsopercularis", "parstriangularis", "caudalmiddlefrontal", "rostralmiddlefrontal")):
        return "FPN"
    if any(token in lower for token in ("entorhinal", "parahippocampal", "temporalpole", "orbitofrontal", "insula")):
        return "Lim"
    if any(token in lower for token in ("cingulate", "medialorbitofrontal", "frontalpole", "middletemporal", "inferiortemporal", "superiortemporal", "fusiform")):
        return "DMN"
    return "FPN"


def order_matrix_by_display_module(matrix: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square.")
    if len(labels) != array.shape[0]:
        raise ValueError(f"Expected {array.shape[0]} labels, got {len(labels)}.")
    module_rank = {module: index for index, module in enumerate(DISPLAY_MODULE_ORDER)}
    order = sorted(
        range(len(labels)),
        key=lambda index: (module_rank.get(infer_display_module(labels[index]), len(module_rank)), labels[index]),
    )
    ordered_labels = [labels[index] for index in order]
    ordered_modules = [infer_display_module(label) for label in ordered_labels]
    return array[np.ix_(order, order)], ordered_labels, ordered_modules


def compute_runge_style_path_scores(
    direct_ei: np.ndarray,
    labels: list[str],
    *,
    path_alpha: float = 1.0,
    max_path_length: int = 60,
) -> dict[str, pd.DataFrame | np.ndarray | float]:
    """Aggregate a directed EI graph into Runge-style ACE, ACS, and AMCE."""

    direct = np.asarray(direct_ei, dtype=float)
    if direct.ndim != 2 or direct.shape[0] != direct.shape[1]:
        raise ValueError("direct_ei must be a square source-by-target matrix.")
    n = direct.shape[0]
    if len(labels) != n:
        raise ValueError(f"Expected {n} labels, got {len(labels)}.")
    if not 0.0 < float(path_alpha) <= 1.0:
        raise ValueError("path_alpha must be in (0, 1].")
    if int(max_path_length) < 1:
        raise ValueError("max_path_length must be at least 1.")

    direct = np.nan_to_num(direct, nan=0.0, posinf=0.0, neginf=0.0).copy()
    direct = np.maximum(direct, 0.0)
    np.fill_diagonal(direct, 0.0)
    if np.any(direct):
        radius = float(np.max(np.abs(np.linalg.eigvals(direct))))
    else:
        radius = 0.0
    scale_factor = float(path_alpha) / radius if radius > 1.0e-12 and radius > float(path_alpha) else 1.0
    scaled = direct * scale_factor

    total = np.zeros_like(scaled)
    power = scaled.copy()
    for _ in range(int(max_path_length)):
        total += power
        power = power @ scaled

    denom = max(1, n - 1)
    mediator_denom = max(1, (n - 1) * (n - 2))
    rows: list[dict[str, float | int | str]] = []
    for index, label in enumerate(labels):
        mediated = 0.0
        for source in range(n):
            if source == index:
                continue
            for target in range(n):
                if target == index or target == source:
                    continue
                mediated += float(scaled[source, index] * total[index, target])
        rows.append(
            {
                "region_index": index,
                "region": label,
                "ace": float(np.sum(total[index, :]) / denom),
                "acs": float(np.sum(total[:, index]) / denom),
                "amce": float(mediated / mediator_denom),
                "direct_out_strength": float(np.sum(scaled[index, :])),
                "direct_in_strength": float(np.sum(scaled[:, index])),
            }
        )
    frame = pd.DataFrame(rows)
    frame["ace_rank"] = frame["ace"].rank(ascending=False, method="min").astype(int)
    frame["acs_rank"] = frame["acs"].rank(ascending=False, method="min").astype(int)
    frame["amce_rank"] = frame["amce"].rank(ascending=False, method="min").astype(int)
    return {
        "region_scores": frame,
        "scaled_direct_matrix": scaled,
        "total_effect_matrix": total,
        "scale_factor": float(scale_factor),
        "spectral_radius": float(radius),
    }


def append_runge_style_scores(
    frame: pd.DataFrame,
    transition_matrix: np.ndarray,
    noise_covariance: np.ndarray,
    labels: list[str],
    *,
    path_alpha: float,
    max_path_length: int,
    ridge: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    direct = compute_singleton_ei_matrix(transition_matrix, noise_covariance, ridge=ridge)
    scores = compute_runge_style_path_scores(
        direct,
        labels,
        path_alpha=path_alpha,
        max_path_length=max_path_length,
    )
    score_frame = scores["region_scores"]
    if not isinstance(score_frame, pd.DataFrame):
        raise TypeError("region_scores must be a DataFrame.")
    merged = frame.merge(score_frame, on=["region_index", "region"], how="left")
    merged.attrs.update(frame.attrs)
    metadata = {
        "runge_path_scale_factor": float(scores["scale_factor"]),
        "runge_path_spectral_radius": float(scores["spectral_radius"]),
        "runge_path_alpha": float(path_alpha),
        "runge_path_max_length": float(max_path_length),
    }
    return merged, metadata


def _region_family(label: str) -> tuple[str, str]:
    lower = label.lower()
    if lower.startswith("ctx-lh-"):
        hemisphere = "left"
    elif lower.startswith("ctx-rh-"):
        hemisphere = "right"
    elif lower.startswith("left-"):
        hemisphere = "left"
    elif lower.startswith("right-"):
        hemisphere = "right"
    else:
        hemisphere = "midline"

    if any(token in lower for token in ("cuneus", "lingual", "pericalcarine", "occipital")):
        family = "occipital"
    elif any(token in lower for token in ("temporal", "fusiform", "entorhinal", "parahippocampal")):
        family = "temporal"
    elif any(token in lower for token in ("parietal", "precuneus", "supramarginal", "postcentral")):
        family = "parietal"
    elif any(token in lower for token in ("frontal", "precentral", "pars", "orbitofrontal", "frontalpole")):
        family = "frontal"
    elif "cingulate" in lower:
        family = "cingulate"
    elif "insula" in lower:
        family = "insula"
    elif any(token in lower for token in ("thalamus", "pallidum", "putamen", "hippocampus", "caudate", "accumbens", "amygdala")):
        family = "subcortical"
    elif "stem" in lower:
        family = "brainstem"
    else:
        family = "other"
    return hemisphere, family


def make_schematic_brain_layout(labels: list[str]) -> pd.DataFrame:
    base = {
        "frontal": (0.55, 0.20),
        "parietal": (0.05, 0.48),
        "temporal": (-0.15, -0.42),
        "occipital": (-0.70, 0.05),
        "cingulate": (0.05, 0.10),
        "insula": (0.05, -0.02),
        "subcortical": (0.00, -0.18),
        "brainstem": (0.00, -0.74),
        "other": (0.00, 0.00),
    }
    rows = []
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, label in enumerate(labels):
        key = _region_family(label)
        grouped.setdefault(key, []).append(index)

    for (hemisphere, family), indices in grouped.items():
        n = len(indices)
        for local_index, index in enumerate(indices):
            angle = 2.0 * np.pi * local_index / max(n, 1)
            radius = 0.055 + 0.015 * (n > 4)
            x0, y0 = base[family]
            x = x0 + radius * np.cos(angle)
            y = y0 + radius * np.sin(angle)
            panel_shift = {"left": -1.15, "right": 1.15, "midline": 0.0}[hemisphere]
            rows.append(
                {
                    "region_index": index,
                    "region": labels[index],
                    "hemisphere": hemisphere,
                    "family": family,
                    "x": x + panel_shift,
                    "y": y,
                }
            )
    return pd.DataFrame(rows).sort_values("region_index").reset_index(drop=True)


def compute_from_cached_dmf(
    source_results: Path,
    *,
    coupling_g: float,
    tau: int,
    seed: int,
    t_total: float,
    burn_in: float,
    dt: float,
    sigma: float,
    ridge: float,
    runge_path_scores: bool = False,
    path_alpha: float = 1.0,
    max_path_length: int = 60,
) -> tuple[pd.DataFrame, dict[str, float]]:
    dmf = load_dmf_module()
    archive = np.load(source_results)
    g_values = np.asarray(archive["G"], dtype=float)
    selected = int(np.argmin(np.abs(g_values - coupling_g)))
    actual_g = float(g_values[selected])
    connectivity = np.asarray(archive["connectivity"], dtype=float)
    labels = load_connectivity_labels()

    parameters = dmf.DMFParameters(t_total=t_total, burn_in=burn_in, dt=dt, sigma=sigma)
    stabilization = dmf.StabilizationParameters(window=0.05, tolerance_hz=0.05, confirm_windows=3)
    simulation = None
    initial_se = None
    initial_si = None
    for index in range(selected + 1):
        simulation = dmf.simulate_dmf(
            connectivity,
            float(g_values[index]),
            np.asarray(archive["j_fic"], dtype=float)[index],
            parameters=parameters,
            stabilization_parameters=stabilization,
            seed=seed + index,
            initial_se=initial_se,
            initial_si=initial_si,
            record_rate_trace=index == selected,
        )
        initial_se = np.asarray(simulation["final_se"], dtype=float)
        initial_si = np.asarray(simulation["final_si"], dtype=float)
    if simulation is None:
        raise RuntimeError("No DMF simulation was run.")
    start_step = int(float(simulation["stabilization_start_step"]))
    rates = np.asarray(simulation["region_rate_trace_hz"], dtype=float)[start_step:]
    source, target = lagged_samples(rates, tau)
    eid_metrics = dmf.estimate_whole_system_phi_eid_from_lagged_samples(source, target, ridge=ridge)
    transition_matrix = np.asarray(eid_metrics["transition_matrix"], dtype=float)
    noise_covariance = np.asarray(eid_metrics["noise_covariance"], dtype=float)
    singleton_ei_matrix = compute_singleton_ei_matrix(transition_matrix, noise_covariance, ridge=ridge)
    frame = compute_target_burden_scores(
        transition_matrix,
        noise_covariance,
        labels=labels,
        ridge=ridge,
    )
    frame.attrs["singleton_ei_matrix"] = singleton_ei_matrix
    frame.attrs["labels"] = labels
    metadata = {
        "requested_g": float(coupling_g),
        "actual_g": actual_g,
        "tau": float(tau),
        "sample_count": float(eid_metrics["sample_count"]),
        "whole_system_phi_eid": float(eid_metrics["phi_eid"]),
        "target_burden_sum": float(frame["target_burden"].sum()),
        "mean_rate_hz": float(simulation["mean_rate_hz"]),
        "stabilization_start_s": float(simulation["stabilization_start_time_s"]),
    }
    if runge_path_scores:
        frame, runge_metadata = append_runge_style_scores(
            frame,
            transition_matrix,
            noise_covariance,
            labels,
            path_alpha=path_alpha,
            max_path_length=max_path_length,
            ridge=ridge,
        )
        metadata.update(runge_metadata)
    return frame, metadata


def plot_target_burden_map(frame: pd.DataFrame, output_base: Path, *, top_k: int = 12) -> None:
    labels = [str(label) for label in frame.sort_values("region_index")["region"].tolist()]
    layout = make_schematic_brain_layout(labels)
    plot_frame = frame.merge(layout, on=["region_index", "region"], how="left")
    top_desc = plot_frame.nlargest(top_k, "target_burden")
    top = top_desc.iloc[::-1]
    singleton = frame.attrs.get("singleton_ei_matrix")
    if singleton is None:
        singleton = np.zeros((len(labels), len(labels)), dtype=float)
    ordered_matrix, ordered_labels, ordered_modules = order_matrix_by_display_module(np.asarray(singleton, dtype=float), labels)
    module_color_row = np.asarray(
        [
            mpl.colors.to_rgba(DISPLAY_MODULE_COLORS.get(module, "#BBBBBB"))
            for module in ordered_modules
        ]
    )[np.newaxis, :, :]

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
        }
    )
    burden_cmap = LinearSegmentedColormap.from_list("burden", ["#F2F2F2", "#F2C300", "#E056B5"])
    fig = plt.figure(figsize=(7.4, 5.8), constrained_layout=True)
    outer = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.22, 0.80, 1.05],
        height_ratios=[0.82, 1.65],
    )
    left_top = outer[0, 0].subgridspec(2, 1, height_ratios=[0.075, 1.0], hspace=0.02)
    ax_module = fig.add_subplot(left_top[0, 0])
    ax_matrix = fig.add_subplot(left_top[1, 0])
    ax_legend = fig.add_subplot(outer[0, 1])
    brain_grid = outer[1, 0:2].subgridspec(2, 2, wspace=0.02, hspace=0.05)
    brain_axes = [
        fig.add_subplot(brain_grid[0, 0]),
        fig.add_subplot(brain_grid[0, 1]),
        fig.add_subplot(brain_grid[1, 0]),
        fig.add_subplot(brain_grid[1, 1]),
    ]
    ax_bar = fig.add_subplot(outer[:, 2])

    matrix_vmax = float(np.nanpercentile(ordered_matrix, 99.0)) if np.isfinite(ordered_matrix).any() else 1.0
    matrix_vmax = max(matrix_vmax, 1.0e-12)
    ax_module.imshow(module_color_row, aspect="auto")
    ax_module.set_axis_off()
    im = ax_matrix.imshow(
        ordered_matrix,
        cmap="hot",
        vmin=0.0,
        vmax=matrix_vmax,
        interpolation="nearest",
        aspect="equal",
    )
    ax_matrix.set_xlabel("Targets")
    ax_matrix.set_ylabel("Sources")
    ax_matrix.set_xticks([])
    ax_matrix.set_yticks([])
    matrix_cbar = fig.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.02)
    matrix_cbar.set_label("Singleton EI")
    ax_matrix.text(-0.17, 1.08, "(A)", transform=ax_matrix.transAxes, fontsize=9)

    ax_legend.axis("off")
    y = 0.95
    for module in DISPLAY_MODULE_ORDER:
        ax_legend.scatter(0.05, y, s=32, marker="s", color=DISPLAY_MODULE_COLORS[module], edgecolor="none")
        ax_legend.text(0.13, y, module, va="center", fontsize=7)
        y -= 0.105
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)

    values = plot_frame["target_burden"].to_numpy(dtype=float)
    if np.isfinite(values).any():
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    else:
        vmin, vmax = 0.0, 1.0
    if np.isclose(vmin, vmax):
        vmin = 0.0
    top_indices = set(top["region_index"].astype(int).tolist())

    def draw_brain_view(ax: plt.Axes, hemisphere: str, view: str) -> mpl.collections.PathCollection:
        ax.add_patch(
            Ellipse((0.0, 0.0), width=1.88, height=1.08, facecolor="#F4F4F4", edgecolor="#BDBDBD", lw=0.8)
        )
        ax.add_patch(
            Ellipse((0.12 if view == "lateral" else -0.18, -0.08), width=1.12, height=0.58, facecolor="white", edgecolor="none", alpha=0.55)
        )
        subset = plot_frame[plot_frame["hemisphere"].eq(hemisphere)].copy()
        if subset.empty:
            subset = plot_frame[plot_frame["hemisphere"].eq("midline")].copy()
        sx = subset["x"].to_numpy(dtype=float)
        if hemisphere == "left":
            sx = sx + 1.15
        elif hemisphere == "right":
            sx = sx - 1.15
        sy = subset["y"].to_numpy(dtype=float)
        if view == "medial":
            sx = -0.82 * sx
            sy = 0.92 * sy - 0.03
        else:
            sx = 0.92 * sx
            sy = 0.95 * sy
        points = ax.scatter(
            sx,
            sy,
            c=subset["target_burden"].to_numpy(dtype=float),
            s=46,
            cmap=burden_cmap,
            vmin=vmin,
            vmax=max(vmax, 1.0e-12),
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
        top_subset = subset[subset["region_index"].isin(top_indices)]
        if not top_subset.empty:
            tx = top_subset["x"].to_numpy(dtype=float)
            if hemisphere == "left":
                tx = tx + 1.15
            elif hemisphere == "right":
                tx = tx - 1.15
            ty = top_subset["y"].to_numpy(dtype=float)
            if view == "medial":
                tx = -0.82 * tx
                ty = 0.92 * ty - 0.03
            else:
                tx = 0.92 * tx
                ty = 0.95 * ty
            ax.scatter(tx, ty, facecolors="none", edgecolors="black", s=78, linewidths=0.65, zorder=4)
        ax.text(0.0, 0.70, f"{hemisphere.capitalize()} {view}", ha="center", va="center", fontsize=7, color="0.28")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.72, 0.78)
        ax.set_aspect("equal")
        ax.axis("off")
        return points

    last_points = draw_brain_view(brain_axes[0], "left", "lateral")
    draw_brain_view(brain_axes[1], "right", "lateral")
    draw_brain_view(brain_axes[2], "left", "medial")
    draw_brain_view(brain_axes[3], "right", "medial")
    brain_axes[0].text(-0.17, 1.05, "(B)", transform=brain_axes[0].transAxes, fontsize=9)
    colorbar = fig.colorbar(last_points, ax=brain_axes, fraction=0.035, pad=0.015)
    colorbar.set_label(r"Target burden $\Phi^{EID}_{\to j}$")

    colors = burden_cmap(
        (top["target_burden"].to_numpy(dtype=float) - vmin) / max(vmax - vmin, 1.0e-12)
    )
    ax_bar.barh(top["region"], top["target_burden"], color=colors, edgecolor="none")
    ax_bar.set_xlabel(r"Target burden $\Phi^{EID}_{\to j}$")
    ax_bar.tick_params(axis="y", labelsize=6)
    ax_bar.grid(True, axis="x", color="0.88", lw=0.6)
    ax_bar.text(-0.18, 1.02, "(C)", transform=ax_bar.transAxes, fontsize=9)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_runge_path_comparison(
    source_results: Path,
    *,
    coupling_values: tuple[float, ...] = (1.7, 1.8),
    tau: int,
    seed: int,
    t_total: float,
    burn_in: float,
    dt: float,
    sigma: float,
    ridge: float,
    path_alpha: float,
    max_path_length: int,
) -> pd.DataFrame:
    frames = []
    for coupling_g in coupling_values:
        frame, metadata = compute_from_cached_dmf(
            source_results,
            coupling_g=coupling_g,
            tau=tau,
            seed=seed,
            t_total=t_total,
            burn_in=burn_in,
            dt=dt,
            sigma=sigma,
            ridge=ridge,
            runge_path_scores=True,
            path_alpha=path_alpha,
            max_path_length=max_path_length,
        )
        columns = [
            "region_index",
            "region",
            "target_burden",
            "ace",
            "acs",
            "amce",
            "direct_out_strength",
            "direct_in_strength",
            "ace_rank",
            "acs_rank",
            "amce_rank",
        ]
        out = frame.loc[:, columns].copy()
        for key, value in metadata.items():
            out[key] = value
        out.attrs = {}
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def plot_runge_path_score_distributions(frame: pd.DataFrame, output_base: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    metrics = [
        ("ace", "ACE"),
        ("acs", "ACS"),
        ("amce", "AMCE"),
    ]
    actual_g_values = sorted(float(value) for value in frame["actual_g"].drop_duplicates())
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    rng = np.random.default_rng(17)
    legend_handles = []
    legend_labels = []
    for ax, (metric, label) in zip(axes, metrics):
        positions = np.arange(len(actual_g_values), dtype=float) + 1.0
        data = [
            frame.loc[np.isclose(frame["actual_g"].astype(float), actual_g), metric].to_numpy(dtype=float)
            for actual_g in actual_g_values
        ]
        ax.boxplot(
            data,
            positions=positions,
            widths=0.46,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 0.8},
            boxprops={"facecolor": "#EFEFEF", "edgecolor": "#777777", "linewidth": 0.7},
            whiskerprops={"color": "#777777", "linewidth": 0.7},
            capprops={"color": "#777777", "linewidth": 0.7},
        )
        for index, (actual_g, values) in enumerate(zip(actual_g_values, data)):
            jitter = rng.normal(0.0, 0.035, size=len(values))
            scatter = ax.scatter(
                np.full(len(values), positions[index]) + jitter,
                values,
                s=8,
                color=colors[index % len(colors)],
                alpha=0.58,
                linewidths=0.0,
                zorder=3,
            )
            if metric == "ace":
                legend_handles.append(scatter)
                legend_labels.append(f"G={actual_g:.1f}")
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{value:.1f}" for value in actual_g_values])
        ax.set_xlabel("Global coupling G")
        ax.set_ylabel(label)
        ax.grid(True, axis="y", color="0.88", linewidth=0.6)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DMF PhiEID target-burden brain distribution.")
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--g", type=float, default=1.7)
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t-total", type=float, default=0.55)
    parser.add_argument("--burn-in", type=float, default=0.15)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--runge-path-scores", action="store_true", help="Append Runge-style ACS/ACE/AMCE path scores.")
    parser.add_argument("--path-alpha", type=float, default=1.0)
    parser.add_argument("--max-path-length", type=int, default=60)
    parser.add_argument("--runge-path-table", type=Path, default=DEFAULT_RUNGE_PATH_TABLE)
    parser.add_argument("--runge-path-output-base", type=Path, default=DEFAULT_RUNGE_PATH_OUTPUT_BASE)
    parser.add_argument("--singleton-matrix", type=Path, default=DEFAULT_SINGLETON_MATRIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, metadata = compute_from_cached_dmf(
        args.source_results,
        coupling_g=args.g,
        tau=args.tau,
        seed=args.seed,
        t_total=args.t_total,
        burn_in=args.burn_in,
        dt=args.dt,
        sigma=args.sigma,
        ridge=args.ridge,
        runge_path_scores=args.runge_path_scores,
        path_alpha=args.path_alpha,
        max_path_length=args.max_path_length,
    )
    for key, value in metadata.items():
        frame[key] = value
    args.table.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.table, index=False)
    singleton = frame.attrs.get("singleton_ei_matrix")
    labels = frame.attrs.get("labels")
    if singleton is not None and labels is not None:
        ordered_matrix, ordered_labels, ordered_modules = order_matrix_by_display_module(
            np.asarray(singleton, dtype=float),
            [str(label) for label in labels],
        )
        matrix_frame = pd.DataFrame(ordered_matrix, index=ordered_labels, columns=ordered_labels)
        matrix_frame.insert(0, "module", ordered_modules)
        args.singleton_matrix.parent.mkdir(parents=True, exist_ok=True)
        matrix_frame.to_csv(args.singleton_matrix, index_label="source_region")
    plot_target_burden_map(frame, args.output_base, top_k=args.top_k)
    print(f"Saved table: {args.table}")
    if singleton is not None and labels is not None:
        print(f"Saved singleton EI matrix: {args.singleton_matrix}")
    print(f"Saved figure: {args.output_base.with_suffix('.png')}")
    print(f"actual_g={metadata['actual_g']:.3g} phi_eid={metadata['whole_system_phi_eid']:.6g}")
    if args.runge_path_scores:
        comparison = build_runge_path_comparison(
            args.source_results,
            tau=args.tau,
            seed=args.seed,
            t_total=args.t_total,
            burn_in=args.burn_in,
            dt=args.dt,
            sigma=args.sigma,
            ridge=args.ridge,
            path_alpha=args.path_alpha,
            max_path_length=args.max_path_length,
        )
        args.runge_path_table.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(args.runge_path_table, index=False)
        plot_runge_path_score_distributions(comparison, args.runge_path_output_base)
        print(f"Saved Runge-style score table: {args.runge_path_table}")
        print(f"Saved Runge-style score figure: {args.runge_path_output_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
