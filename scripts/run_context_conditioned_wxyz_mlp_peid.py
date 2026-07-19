#!/usr/bin/env python3
"""Compare marginal and context-conditioned PEID readouts on the same 4D MLP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_granger_peid_mlp import (
    BETA_COMMON_DRIVER_SWEEP_VALUES,
    DEFAULT_FIGURE_DIR,
    DEFAULT_RESULT_DIR,
    SimConfig,
    _intervention_features,
    _mmi_pid_from_mi_triplet,
    _observational_wms,
    make_lagged_dataset,
    simulate_system,
    train_mlp_transition_model,
)
from scripts.run_fixed_support_wxyz_sine_beta_mlp_peid import (
    DEFAULT_SUPPORTS,
    _r2_score,
    _tv_metrics,
)


DEFAULT_RESULT_PATH = (
    DEFAULT_RESULT_DIR / "sine_beta_simple_coefficients_wxyz_context_conditioned.json"
)
DEFAULT_FIGURE_STEM = "sine_beta_wxyz_context_conditioning_ablation"
READOUT_MODES = (
    "marginal_fixed_support",
    "zero_context",
    "conditional_context_mean",
    "centered_context_pool",
    "anchored_interaction_surface",
    "anova_interaction_surface",
    "anova_interaction_common_support",
)
MODE_LABELS = {
    "marginal_fixed_support": "Marginal fixed 4D support",
    "zero_context": r"Fixed context $w=z=0$",
    "conditional_context_mean": "Mean of context-conditional PEID",
    "centered_context_pool": "Context-centered pooled PEID",
    "anchored_interaction_surface": "Anchored interaction surface",
    "anova_interaction_surface": "Functional-ANOVA interaction surface",
    "anova_interaction_common_support": "ANOVA interaction on common support",
}
PLOT_MODES = (
    "marginal_fixed_support",
    "conditional_context_mean",
    "anova_interaction_surface",
    "anova_interaction_common_support",
)
COMMON_SOURCE_SUPPORT = (-1.0, 1.0)
COMMON_METHODS = ("mlp_peid", "mmi_pid", "surd", "wms")
COMMON_METHOD_LABELS = {
    "mlp_peid": "MLP+PEID",
    "mmi_pid": "MMI-PID",
    "surd": "SURD",
    "wms": "WMS",
}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _source_samples(seed: int, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed) + 1009)
    x = rng.uniform(*DEFAULT_SUPPORTS["x"], size=int(n_samples))
    y = rng.uniform(*DEFAULT_SUPPORTS["y"], size=int(n_samples))
    return x, y


def _context_anchors(seed: int, n_contexts: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed) + 7009)
    w = rng.uniform(*DEFAULT_SUPPORTS["w"], size=int(n_contexts))
    z = rng.uniform(*DEFAULT_SUPPORTS["z"], size=int(n_contexts))
    return w, z


def _state_frame(
    variable_names: Sequence[str],
    *,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | float,
    z: np.ndarray | float,
) -> pd.DataFrame:
    n = len(x)
    values = {
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y, dtype=float),
        "w": np.broadcast_to(np.asarray(w, dtype=float), (n,)),
        "z": np.broadcast_to(np.asarray(z, dtype=float), (n,)),
    }
    return pd.DataFrame({name: values[name] for name in variable_names})


def _summarize(left: np.ndarray, right: np.ndarray, target: np.ndarray) -> dict[str, float]:
    from yrd.transport_map import summarize_two_source_synergy_transport_map

    result = summarize_two_source_synergy_transport_map(
        np.asarray(left, dtype=float).reshape(-1, 1),
        np.asarray(right, dtype=float).reshape(-1, 1),
        np.asarray(target, dtype=float).reshape(-1, 1),
    )
    return {
        "unique_x": float(result["left_ei"]),
        "unique_y": float(result["right_ei"]),
        "synergy": float(result["syn"]),
        "joint": float(result["joint_ei"]),
    }


def evaluate_readout_modes(
    model,
    config: SimConfig,
    *,
    source_samples: int,
    context_anchors: int,
    interaction_grid_levels: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Evaluate four paired readout definitions without changing the fitted MLP."""

    x, y = _source_samples(config.seed, source_samples)
    context_w, context_z = _context_anchors(config.seed, context_anchors)
    target_index = config.variable_names.index("z")

    zero_states = _state_frame(
        config.variable_names,
        x=x,
        y=y,
        w=0.0,
        z=0.0,
    )
    zero_target = model.predict(_intervention_features(zero_states, config))[:, target_index]
    zero_result = _summarize(x, y, zero_target)

    context_predictions: list[np.ndarray] = []
    conditional_results: list[dict[str, float]] = []
    for w_value, z_value in zip(context_w, context_z, strict=True):
        states = _state_frame(
            config.variable_names,
            x=x,
            y=y,
            w=float(w_value),
            z=float(z_value),
        )
        target = model.predict(_intervention_features(states, config))[:, target_index]
        context_predictions.append(target)
        conditional_results.append(_summarize(x, y, target))

    conditional_result = {
        key: float(np.mean([result[key] for result in conditional_results]))
        for key in ("unique_x", "unique_y", "synergy", "joint")
    }
    conditional_result.update(
        {
            f"{key}_context_std": float(
                np.std([result[key] for result in conditional_results], ddof=1)
            )
            for key in ("unique_x", "unique_y", "synergy", "joint")
        }
    )

    repeated_x = np.tile(x, int(context_anchors))
    repeated_y = np.tile(y, int(context_anchors))
    pooled_target = np.concatenate(context_predictions)
    marginal_result = _summarize(repeated_x, repeated_y, pooled_target)
    centered_target = np.concatenate(
        [prediction - float(np.mean(prediction)) for prediction in context_predictions]
    )
    centered_result = _summarize(repeated_x, repeated_y, centered_target)

    levels_x = np.linspace(*DEFAULT_SUPPORTS["x"], int(interaction_grid_levels))
    levels_y = np.linspace(*DEFAULT_SUPPORTS["y"], int(interaction_grid_levels))
    grid_x, grid_y = np.meshgrid(levels_x, levels_y, indexing="ij")
    flat_x = grid_x.reshape(-1)
    flat_y = grid_y.reshape(-1)
    interaction_targets: list[np.ndarray] = []
    for w_value, z_value in zip(context_w, context_z, strict=True):
        states = _state_frame(
            config.variable_names,
            x=flat_x,
            y=flat_y,
            w=float(w_value),
            z=float(z_value),
        )
        target = model.predict(_intervention_features(states, config))[:, target_index]
        interaction_targets.append(target.reshape(len(levels_x), len(levels_y)))
    mean_surface = np.mean(interaction_targets, axis=0)
    anchor_x = int(np.argmin(np.abs(levels_x)))
    anchor_y = int(np.argmin(np.abs(levels_y)))
    anchored_surface = (
        mean_surface
        - mean_surface[:, [anchor_y]]
        - mean_surface[[anchor_x], :]
        + mean_surface[anchor_x, anchor_y]
    )
    anova_surface = (
        mean_surface
        - np.mean(mean_surface, axis=1, keepdims=True)
        - np.mean(mean_surface, axis=0, keepdims=True)
        + float(np.mean(mean_surface))
    )
    anchored_result = _summarize(flat_x, flat_y, anchored_surface.reshape(-1))
    anova_result = _summarize(flat_x, flat_y, anova_surface.reshape(-1))

    common_levels_x = np.linspace(*COMMON_SOURCE_SUPPORT, int(interaction_grid_levels))
    common_levels_y = np.linspace(*COMMON_SOURCE_SUPPORT, int(interaction_grid_levels))
    common_grid_x, common_grid_y = np.meshgrid(common_levels_x, common_levels_y, indexing="ij")
    common_flat_x = common_grid_x.reshape(-1)
    common_flat_y = common_grid_y.reshape(-1)
    common_targets: list[np.ndarray] = []
    for w_value, z_value in zip(context_w, context_z, strict=True):
        states = _state_frame(
            config.variable_names,
            x=common_flat_x,
            y=common_flat_y,
            w=float(w_value),
            z=float(z_value),
        )
        target = model.predict(_intervention_features(states, config))[:, target_index]
        common_targets.append(
            target.reshape(len(common_levels_x), len(common_levels_y))
        )
    common_mean_surface = np.mean(common_targets, axis=0)
    common_anova_surface = (
        common_mean_surface
        - np.mean(common_mean_surface, axis=1, keepdims=True)
        - np.mean(common_mean_surface, axis=0, keepdims=True)
        + float(np.mean(common_mean_surface))
    )
    common_anova_result = _summarize(
        common_flat_x,
        common_flat_y,
        common_anova_surface.reshape(-1),
    )
    observational = _observational_wms(
        common_flat_x,
        common_flat_y,
        common_anova_surface.reshape(-1),
        bins=int(config.bins),
    )
    mmi = _mmi_pid_from_mi_triplet(
        left_mi=float(observational["x_mi"]),
        right_mi=float(observational["y_mi"]),
        joint_mi=float(observational["joint_mi"]),
    )
    from scripts.reproduce_surd_synergistic_collider import decompose_surd_2source_transport_map

    surd = decompose_surd_2source_transport_map(
        common_flat_x,
        common_flat_y,
        common_anova_surface.reshape(-1),
        degree=3,
        target_anchors=128,
        conditional_samples=64,
        seed=int(config.seed),
    )
    modes = {
        "marginal_fixed_support": marginal_result,
        "zero_context": zero_result,
        "conditional_context_mean": conditional_result,
        "centered_context_pool": centered_result,
        "anchored_interaction_surface": anchored_result,
        "anova_interaction_surface": anova_result,
        "anova_interaction_common_support": common_anova_result,
    }
    common_methods = {
        "mlp_peid": common_anova_result,
        "mmi_pid": {
            "unique_x": float(mmi["unique_x"]),
            "unique_y": float(mmi["unique_y"]),
            "synergy": float(mmi["synergy"]),
            "joint": float(mmi["joint"]),
        },
        "surd": {
            "unique_x": float(surd["unique_x"]),
            "unique_y": float(surd["unique_y"]),
            "synergy": float(surd["synergy"]),
            "joint": float(surd["joint_ei"]),
        },
        "wms": {"synergy": float(observational["wms"])},
    }
    return modes, common_methods


def run_context_conditioning_ablation(
    *,
    beta_values: Sequence[float] = BETA_COMMON_DRIVER_SWEEP_VALUES,
    seeds: Sequence[int] = (0, 1, 2, 3),
    n_samples: int = 1100,
    noise: float = 0.05,
    mlp_epochs: int = 90,
    source_samples: int = 640,
    context_anchors: int = 16,
    interaction_grid_levels: int = 25,
    show_progress: bool = True,
) -> dict[str, object]:
    pairs: Sequence[tuple[float, int]] = [
        (float(beta), int(seed)) for beta in beta_values for seed in seeds
    ]
    if show_progress:
        from tqdm.auto import tqdm

        pairs = tqdm(pairs, desc="4D context-conditioned PEID", unit="run", mininterval=1.0)

    rows: list[dict[str, float | str]] = []
    for beta, seed in pairs:
        config = SimConfig(
            mechanism="common_driver_sine_synergy",
            n_samples=int(n_samples),
            noise=float(noise),
            seed=int(seed),
            synergy_strength=1.0,
            common_driver_strength=float(beta),
            mlp_epochs=int(mlp_epochs),
            intervention_samples=int(source_samples),
            bins=4,
        )
        series, _ = simulate_system(config)
        features, targets = make_lagged_dataset(series, lag=config.lag)
        model = train_mlp_transition_model(features, targets, config)
        mode_results, common_methods = evaluate_readout_modes(
            model,
            config,
            source_samples=int(source_samples),
            context_anchors=int(context_anchors),
            interaction_grid_levels=int(interaction_grid_levels),
        )
        target_index = config.variable_names.index("z")
        prediction = model.predict(features)[:, target_index]
        row: dict[str, float | str] = {
            "run_id": f"beta={beta:.2f}|seed={seed}",
            "beta": beta,
            "seed": float(seed),
            "z_train_r2": _r2_score(targets[:, target_index], prediction),
        }
        for mode, result in mode_results.items():
            for metric, value in result.items():
                row[f"{mode}__{metric}"] = float(value)
        for method, result in common_methods.items():
            for metric, value in result.items():
                row[f"common_anova__{method}__{metric}"] = float(value)
        rows.append(row)

    frame = pd.DataFrame(rows)
    aggregations: dict[str, tuple[str, str]] = {}
    for mode in READOUT_MODES:
        for metric in ("unique_x", "unique_y", "synergy", "joint"):
            column = f"{mode}__{metric}"
            aggregations[f"{column}_mean"] = (column, "mean")
            aggregations[f"{column}_std"] = (column, "std")
    aggregations["z_train_r2_mean"] = ("z_train_r2", "mean")
    aggregations["z_train_r2_std"] = ("z_train_r2", "std")
    for method in COMMON_METHODS:
        metrics = ("synergy",) if method == "wms" else ("unique_x", "unique_y", "synergy", "joint")
        for metric in metrics:
            column = f"common_anova__{method}__{metric}"
            aggregations[f"{column}_mean"] = (column, "mean")
            aggregations[f"{column}_std"] = (column, "std")
    summary = frame.groupby("beta", as_index=False).agg(**aggregations).sort_values("beta")

    beta = summary["beta"].to_numpy(dtype=float)
    sensitivity: dict[str, dict[str, dict[str, float]]] = {}
    trend: dict[str, float] = {}
    for mode in READOUT_MODES:
        sensitivity[mode] = {}
        for metric in ("synergy", "unique_x", "unique_y"):
            values = summary[f"{mode}__{metric}_mean"].to_numpy(dtype=float)
            sensitivity[mode][metric] = _tv_metrics(beta, values)
            trend[f"{mode}__{metric}_slope"] = float(np.polyfit(beta, values, 1)[0])

    common_sensitivity: dict[str, dict[str, dict[str, float]]] = {}
    for method in COMMON_METHODS:
        common_sensitivity[method] = {}
        metrics = ("synergy",) if method == "wms" else ("synergy", "unique_x", "unique_y")
        for metric in metrics:
            values = summary[f"common_anova__{method}__{metric}_mean"].to_numpy(dtype=float)
            common_sensitivity[method][metric] = _tv_metrics(beta, values)

    return {
        "contract": {
            "question": "What changes when only the PEID treatment of w,z context changes on the same fitted 4D MLP?",
            "treatment": list(READOUT_MODES),
            "controlled": [
                "dynamics",
                "beta grid",
                "seeds",
                "trajectories",
                "MLP architecture and training budget",
                "x/y source support and paired samples",
                "w/z context support and anchors",
                "TM estimator",
            ],
        },
        "config": {
            "beta_values": [float(value) for value in beta_values],
            "seeds": [int(value) for value in seeds],
            "n_samples": int(n_samples),
            "noise": float(noise),
            "mlp_epochs": int(mlp_epochs),
            "source_samples_per_context": int(source_samples),
            "context_anchors": int(context_anchors),
            "interaction_grid_levels": int(interaction_grid_levels),
            "common_source_support": [float(bound) for bound in COMMON_SOURCE_SUPPORT],
            "common_source_support_rationale": (
                "Symmetric [-1,1] lies inside the cross-beta, cross-seed 5%-95% "
                "support intersection for both x and y; it is fixed before method comparison."
            ),
            "fixed_supports": {
                name: [float(bound) for bound in DEFAULT_SUPPORTS[name]]
                for name in ("w", "x", "y", "z")
            },
        },
        "runs": frame.to_dict("records"),
        "summary": summary.to_dict("records"),
        "sensitivity": sensitivity,
        "common_anova_comparison": {
            "estimand": "All methods decompose the identical functional-ANOVA interaction surface from the fitted MLP.",
            "oracle_used": False,
            "methods": list(COMMON_METHODS),
            "sensitivity": common_sensitivity,
        },
        "trend": trend,
    }


def plot_ablation(
    result: Mapping[str, object],
    figure_dir: Path,
    *,
    stem: str = DEFAULT_FIGURE_STEM,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    frame = pd.DataFrame(result["summary"]).sort_values("beta")
    styles = {
        "marginal_fixed_support": ("#8C8C8C", "o", 1.35),
        "zero_context": ("#4C78A8", "D", 1.45),
        "conditional_context_mean": ("#009E73", "s", 1.9),
        "centered_context_pool": ("#E68613", "^", 1.45),
        "anchored_interaction_surface": ("#7E57C2", "P", 1.55),
        "anova_interaction_surface": ("#C44E52", "X", 1.75),
        "anova_interaction_common_support": ("#2A9D8F", "*", 2.0),
    }
    panels = (("synergy", "Synergy (bits)"), ("unique_x", r"$U_x$ (bits)"), ("unique_y", r"$U_y$ (bits)"))
    fig, axes = plt.subplots(1, 3, figsize=(8.7, 2.8), sharex=True, constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, panels, strict=True):
        for mode in PLOT_MODES:
            color, marker, linewidth = styles[mode]
            x = frame["beta"].to_numpy(dtype=float)
            y = frame[f"{mode}__{metric}_mean"].to_numpy(dtype=float)
            sd = frame[f"{mode}__{metric}_std"].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                color=color,
                marker=marker,
                markevery=max(1, (len(x) - 1) // 5),
                linewidth=linewidth,
                markersize=3.0,
                markeredgecolor="white",
                markeredgewidth=0.3,
                label=MODE_LABELS[mode],
            )
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.10, linewidth=0)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.01, 1.01)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.grid(axis="y", alpha=0.17, linewidth=0.5)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3)
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / f"{stem}.png"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    return png


def plot_common_method_comparison(
    result: Mapping[str, object],
    figure_dir: Path,
    *,
    stem: str = "sine_beta_wxyz_anova_common_readout_comparison",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    frame = pd.DataFrame(result["summary"]).sort_values("beta")
    styles = {
        "mlp_peid": ("#009E73", "s", 1.9),
        "mmi_pid": ("#B07AA1", "P", 1.4),
        "surd": ("#8C8C8C", "o", 1.4),
        "wms": ("#4C78A8", "D", 1.4),
    }
    panels = (("synergy", "Synergy (bits)"), ("unique_x", r"$U_x$ (bits)"), ("unique_y", r"$U_y$ (bits)"))
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.75), sharex=True, constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, panels, strict=True):
        for method in COMMON_METHODS:
            if method == "wms" and metric != "synergy":
                continue
            color, marker, linewidth = styles[method]
            x = frame["beta"].to_numpy(dtype=float)
            y = frame[f"common_anova__{method}__{metric}_mean"].to_numpy(dtype=float)
            sd = frame[f"common_anova__{method}__{metric}_std"].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                color=color,
                marker=marker,
                markevery=max(1, (len(x) - 1) // 5),
                linewidth=linewidth,
                markersize=3.1,
                markeredgecolor="white",
                markeredgewidth=0.3,
                label=COMMON_METHOD_LABELS[method],
            )
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.11, linewidth=0)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.01, 1.01)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.grid(axis="y", alpha=0.17, linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4)
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / f"{stem}.png"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    return png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs: dict[str, object] = {}
    if args.smoke:
        kwargs.update(
            beta_values=(0.0, 1.0),
            seeds=(0,),
            n_samples=320,
            mlp_epochs=5,
            source_samples=64,
            context_anchors=4,
            interaction_grid_levels=9,
        )
    result = run_context_conditioning_ablation(**kwargs)
    _write_json(args.result_path, result)
    figure = plot_ablation(
        result,
        args.figure_dir,
        stem=DEFAULT_FIGURE_STEM + ("_smoke" if args.smoke else ""),
    )
    common_figure = plot_common_method_comparison(
        result,
        args.figure_dir,
        stem="sine_beta_wxyz_anova_common_readout_comparison" + ("_smoke" if args.smoke else ""),
    )
    print(
        json.dumps(
            {
                "result": str(args.result_path),
                "figure": str(figure),
                "common_method_figure": str(common_figure),
                "sensitivity": result["sensitivity"],
                "common_anova_comparison": result["common_anova_comparison"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
