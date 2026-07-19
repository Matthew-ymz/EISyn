#!/usr/bin/env python3
"""Search and validate one-decimal sine dynamics for end-to-end robustness.

The selection split uses sparse beta values and seeds 0--1.  The selected
dynamics are then checked on untouched seeds 2--3 and the full beta grid.
No Oracle quantity enters either the search score or the reported metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_granger_peid_mlp import (
    DEFAULT_FIGURE_DIR,
    DEFAULT_RESULT_DIR,
    SimConfig,
    _intervention_features,
    _mmi_pid_from_mi_triplet,
    _observational_wms,
    make_lagged_dataset,
    train_mlp_transition_model,
)
from scripts.run_context_conditioned_wxyz_mlp_peid import (
    COMMON_SOURCE_SUPPORT,
    _context_anchors,
    _state_frame,
    _summarize,
)
from scripts.run_fixed_support_wxyz_sine_beta_mlp_peid import _r2_score, _tv_metrics


LOG_DIR = ROOT / "docs" / "log" / "sine_beta_robust_dynamics"
RESULT_PATH = DEFAULT_RESULT_DIR / "sine_beta_one_decimal_robust_dynamics.json"
SEARCH_PATH = DEFAULT_RESULT_DIR / "sine_beta_one_decimal_dynamics_search.json"
FIGURE_STEM = "sine_beta_one_decimal_end_to_end_robustness"
METHODS = ("mlp_peid", "mmi_pid", "surd", "wms")
METRICS = ("synergy", "unique_x", "unique_y")
LABELS = {
    "mlp_peid": "MLP+PEID",
    "mmi_pid": "MMI-PID",
    "surd": "SURD",
    "wms": "Observational WMS",
}


@dataclass(frozen=True)
class Dynamics:
    driver_memory: float = 0.5
    source_memory: float = 0.5
    driver_to_sources: float = 1.0
    target_memory: float = 0.5
    synergy_loading: float = 1.0
    driver_to_target: float = 0.5
    driver_noise_sd: float = 0.4
    private_source_noise_sd: float = 0.6
    source_noise_sd: float = 0.3
    target_noise_sd: float = 0.1

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isclose(float(value), round(float(value), 1)):
                raise ValueError(f"{name}={value} is not a one-decimal coefficient")
        for name in ("driver_memory", "source_memory", "target_memory"):
            if abs(float(getattr(self, name))) >= 1.0:
                raise ValueError(f"{name} must have absolute value below one")

    @property
    def candidate_id(self) -> str:
        return (
            f"src{self.driver_to_sources:.1f}_tgt{self.driver_to_target:.1f}"
            f"_sm{self.source_memory:.1f}_tm{self.target_memory:.1f}"
        )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def simulate_dynamics(config: SimConfig, dynamics: Dynamics) -> pd.DataFrame:
    """Simulate the shared four-dimensional DGP with one-decimal constants."""

    rng = np.random.default_rng(int(config.seed))
    n = int(config.n_samples)
    beta = float(config.common_driver_strength)
    private_scale = float(np.sqrt(max(0.0, 1.0 - beta**2)))
    data = np.zeros((n, 4), dtype=float)
    data[0] = rng.normal(0.0, (0.4, 0.4, 0.2, 0.8))
    for t in range(n - 1):
        x, y, z, w = data[t]
        data[t + 1, 3] = (
            dynamics.driver_memory * w
            + rng.normal(0.0, dynamics.driver_noise_sd)
        )
        data[t + 1, 0] = (
            dynamics.source_memory * x
            + dynamics.driver_to_sources
            * (
                beta * w
                + private_scale * rng.normal(0.0, dynamics.private_source_noise_sd)
            )
            + rng.normal(0.0, dynamics.source_noise_sd)
        )
        data[t + 1, 1] = (
            dynamics.source_memory * y
            + dynamics.driver_to_sources
            * (
                beta * w
                + private_scale * rng.normal(0.0, dynamics.private_source_noise_sd)
            )
            + rng.normal(0.0, dynamics.source_noise_sd)
        )
        data[t + 1, 2] = (
            dynamics.target_memory * z
            + dynamics.synergy_loading * np.sin(x * y)
            + dynamics.driver_to_target * beta * w
            + rng.normal(0.0, dynamics.target_noise_sd)
        )
    if not np.all(np.isfinite(data)):
        raise FloatingPointError(f"Non-finite state for {dynamics.candidate_id}")
    return pd.DataFrame(data, columns=config.variable_names)


def mlp_anova_readout(
    model,
    config: SimConfig,
    *,
    context_anchors: int,
    grid_levels: int,
) -> dict[str, float]:
    """PEID of the MLP's beta-invariant, context-averaged x-y interaction."""

    levels = np.linspace(*COMMON_SOURCE_SUPPORT, int(grid_levels))
    grid_x, grid_y = np.meshgrid(levels, levels, indexing="ij")
    flat_x, flat_y = grid_x.reshape(-1), grid_y.reshape(-1)
    context_w, context_z = _context_anchors(int(config.seed), int(context_anchors))
    target_index = config.variable_names.index("z")
    surfaces: list[np.ndarray] = []
    for w_value, z_value in zip(context_w, context_z, strict=True):
        states = _state_frame(
            config.variable_names,
            x=flat_x,
            y=flat_y,
            w=float(w_value),
            z=float(z_value),
        )
        prediction = model.predict(_intervention_features(states, config))[:, target_index]
        surfaces.append(prediction.reshape(len(levels), len(levels)))
    surface = np.mean(surfaces, axis=0)
    interaction = (
        surface
        - np.mean(surface, axis=1, keepdims=True)
        - np.mean(surface, axis=0, keepdims=True)
        + float(np.mean(surface))
    )
    return _summarize(flat_x, flat_y, interaction.reshape(-1))


def evaluate_run(
    beta: float,
    seed: int,
    dynamics: Dynamics,
    *,
    n_samples: int,
    mlp_epochs: int,
    context_anchors: int,
    grid_levels: int,
    surd_target_anchors: int,
    surd_conditional_samples: int,
) -> dict[str, float | str]:
    from scripts.reproduce_surd_synergistic_collider import (
        decompose_surd_2source_transport_map,
    )

    config = SimConfig(
        mechanism="common_driver_sine_synergy",
        n_samples=int(n_samples),
        noise=float(dynamics.target_noise_sd),
        seed=int(seed),
        synergy_strength=float(dynamics.synergy_loading),
        common_driver_strength=float(beta),
        mlp_epochs=int(mlp_epochs),
        intervention_samples=max(64, int(grid_levels) ** 2),
        bins=4,
    )
    series = simulate_dynamics(config, dynamics)
    features, targets = make_lagged_dataset(series, lag=config.lag)
    model = train_mlp_transition_model(features, targets, config)
    mlp = mlp_anova_readout(
        model,
        config,
        context_anchors=int(context_anchors),
        grid_levels=int(grid_levels),
    )
    x = series["x"].to_numpy(dtype=float)[:-1]
    y = series["y"].to_numpy(dtype=float)[:-1]
    target = series["z"].to_numpy(dtype=float)[1:]
    observational = _observational_wms(x, y, target, bins=4)
    mmi = _mmi_pid_from_mi_triplet(
        left_mi=float(observational["x_mi"]),
        right_mi=float(observational["y_mi"]),
        joint_mi=float(observational["joint_mi"]),
    )
    surd = decompose_surd_2source_transport_map(
        x,
        y,
        target,
        degree=3,
        target_anchors=int(surd_target_anchors),
        conditional_samples=int(surd_conditional_samples),
        seed=int(seed),
    )
    z_index = config.variable_names.index("z")
    prediction = model.predict(features)[:, z_index]
    return {
        "run_id": f"{dynamics.candidate_id}|beta={beta:.2f}|seed={seed}",
        "candidate_id": dynamics.candidate_id,
        "beta": float(beta),
        "seed": int(seed),
        "z_train_r2": _r2_score(targets[:, z_index], prediction),
        "state_abs_max": float(np.max(np.abs(series.to_numpy(dtype=float)))),
        "mlp_peid__synergy": float(mlp["synergy"]),
        "mlp_peid__unique_x": float(mlp["unique_x"]),
        "mlp_peid__unique_y": float(mlp["unique_y"]),
        "mmi_pid__synergy": float(mmi["synergy"]),
        "mmi_pid__unique_x": float(mmi["unique_x"]),
        "mmi_pid__unique_y": float(mmi["unique_y"]),
        "surd__synergy": float(surd["synergy"]),
        "surd__unique_x": float(surd["unique_x"]),
        "surd__unique_y": float(surd["unique_y"]),
        "wms__synergy": float(observational["wms"]),
    }


def summarize_runs(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    value_columns = [
        f"{method}__{metric}"
        for method in METHODS
        for metric in METRICS
        if not (method == "wms" and metric != "synergy")
    ]
    aggregations: dict[str, tuple[str, str]] = {
        f"{column}_mean": (column, "mean") for column in value_columns
    }
    aggregations.update(
        {f"{column}_std": (column, "std") for column in value_columns}
    )
    aggregations["z_train_r2_mean"] = ("z_train_r2", "mean")
    aggregations["z_train_r2_std"] = ("z_train_r2", "std")
    summary = frame.groupby("beta", as_index=False).agg(**aggregations).sort_values("beta")
    beta_values = summary["beta"].to_numpy(dtype=float)
    sensitivity: dict[str, dict[str, dict[str, float]]] = {}
    for method in METHODS:
        sensitivity[method] = {}
        for metric in METRICS:
            if method == "wms" and metric != "synergy":
                continue
            values = summary[f"{method}__{metric}_mean"].to_numpy(dtype=float)
            metrics = _tv_metrics(beta_values, values)
            metrics["range_over_mean"] = float(
                (np.max(values) - np.min(values)) / max(np.mean(np.abs(values)), 1e-12)
            )
            sensitivity[method][metric] = metrics
    gaps: dict[str, float] = {}
    for metric in METRICS:
        baselines = ("mmi_pid", "surd", "wms") if metric == "synergy" else ("mmi_pid", "surd")
        mlp_tv = float(sensitivity["mlp_peid"][metric]["absolute_tv"])
        gaps[metric] = float(
            min(float(sensitivity[method][metric]["absolute_tv"]) for method in baselines)
            - mlp_tv
        )
    return {
        "summary": summary.to_dict("records"),
        "sensitivity": sensitivity,
        "robustness_gaps": gaps,
        "minimum_gap": float(min(gaps.values())),
        "mean_z_train_r2": float(frame["z_train_r2"].mean()),
    }


def run_candidate(
    dynamics: Dynamics,
    *,
    beta_values: Sequence[float],
    seeds: Sequence[int],
    n_samples: int,
    mlp_epochs: int,
    context_anchors: int,
    grid_levels: int,
    surd_target_anchors: int = 128,
    surd_conditional_samples: int = 64,
    show_progress: bool = True,
) -> dict[str, object]:
    pairs: Sequence[tuple[float, int]] = [
        (float(beta), int(seed)) for beta in beta_values for seed in seeds
    ]
    if show_progress:
        from tqdm.auto import tqdm

        pairs = tqdm(pairs, desc=dynamics.candidate_id, unit="run", mininterval=1.0)
    rows = [
        evaluate_run(
            beta,
            seed,
            dynamics,
            n_samples=int(n_samples),
            mlp_epochs=int(mlp_epochs),
            context_anchors=int(context_anchors),
            grid_levels=int(grid_levels),
            surd_target_anchors=int(surd_target_anchors),
            surd_conditional_samples=int(surd_conditional_samples),
        )
        for beta, seed in pairs
    ]
    result = summarize_runs(rows)
    result.update(
        {
            "dynamics": asdict(dynamics),
            "candidate_id": dynamics.candidate_id,
            "beta_values": [float(value) for value in beta_values],
            "seeds": [int(value) for value in seeds],
            "runs": rows,
        }
    )
    return result


def candidate_grid() -> list[Dynamics]:
    return [
        Dynamics(driver_to_sources=float(source), driver_to_target=float(target))
        for source in (0.8, 1.2, 1.6, 2.0)
        for target in (0.8, 1.2, 1.6, 2.0)
    ]


def _append_history(payload: Mapping[str, object]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "run_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_leaderboard(results: Sequence[Mapping[str, object]]) -> None:
    ranked = sorted(results, key=lambda row: float(row["minimum_gap"]), reverse=True)
    lines = [
        "# 候选排行榜",
        "",
        "| Rank | Candidate | min gap | synergy gap | Ux gap | Uy gap | mean R2 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, start=1):
        gaps = row["robustness_gaps"]
        lines.append(
            f"| {rank} | {row['candidate_id']} | {float(row['minimum_gap']):.5f} | "
            f"{float(gaps['synergy']):.5f} | {float(gaps['unique_x']):.5f} | "
            f"{float(gaps['unique_y']):.5f} | {float(row['mean_z_train_r2']):.4f} |"
        )
    (LOG_DIR / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_search(*, smoke: bool = False) -> dict[str, object]:
    candidates = candidate_grid()[:1] if smoke else candidate_grid()
    beta_values = (0.0, 1.0) if smoke else (0.0, 0.25, 0.5, 0.75, 1.0)
    seeds = (0,) if smoke else (0, 1)
    settings = {
        "n_samples": 320 if smoke else 1100,
        "mlp_epochs": 5 if smoke else 90,
        "context_anchors": 4 if smoke else 16,
        "grid_levels": 9 if smoke else 25,
        "surd_target_anchors": 32 if smoke else 128,
        "surd_conditional_samples": 16 if smoke else 64,
    }
    results: list[dict[str, object]] = []
    for dynamics in candidates:
        result = run_candidate(
            dynamics,
            beta_values=beta_values,
            seeds=seeds,
            **settings,
        )
        compact = {key: value for key, value in result.items() if key not in {"runs", "summary"}}
        results.append(compact)
        _append_history(
            {"phase": "smoke" if smoke else "search_final_fidelity", **compact}
        )
        payload = {
            "contract": {
                "selection_only": True,
                "oracle_used": False,
                "beta_values": list(beta_values),
                "seeds": list(seeds),
                "settings": settings,
            },
            "candidates": results,
        }
        _write_json(SEARCH_PATH.with_name(SEARCH_PATH.stem + ("_smoke.json" if smoke else ".json")), payload)
        _write_leaderboard(results)
    ranked = sorted(results, key=lambda row: float(row["minimum_gap"]), reverse=True)
    return {"candidates": ranked, "selected": ranked[0]}


def plot_curves(result: Mapping[str, object], figure_dir: Path) -> Path:
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
    styles = {
        "mlp_peid": ("#009E73", "s", 2.0),
        "mmi_pid": ("#B07AA1", "P", 1.35),
        "surd": ("#8C8C8C", "o", 1.35),
        "wms": ("#4C78A8", "D", 1.35),
    }
    frame = pd.DataFrame(result["summary"]).sort_values("beta")
    panels = (("synergy", "Synergy (bits)"), ("unique_x", r"$U_x$ (bits)"), ("unique_y", r"$U_y$ (bits)"))
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.75), sharex=True, constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, panels, strict=True):
        for method in METHODS:
            if method == "wms" and metric != "synergy":
                continue
            color, marker, linewidth = styles[method]
            x = frame["beta"].to_numpy(dtype=float)
            y = frame[f"{method}__{metric}_mean"].to_numpy(dtype=float)
            sd = frame[f"{method}__{metric}_std"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                color=color,
                marker=marker,
                markevery=max(1, (len(x) - 1) // 5),
                markersize=3.0,
                linewidth=linewidth,
                markeredgecolor="white",
                markeredgewidth=0.3,
                label=LABELS[method],
            )
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.10, linewidth=0)
        ax.set(xlabel=r"Common-driver strength $\beta$", ylabel=ylabel, xlim=(-0.01, 1.01))
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.grid(axis="y", alpha=0.17, linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=4)
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / f"{FIGURE_STEM}.png"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(figure_dir / f"{FIGURE_STEM}.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / f"{FIGURE_STEM}.svg", bbox_inches="tight")
    plt.close(fig)
    return png


def plot_sensitivity(result: Mapping[str, object], figure_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"mlp_peid": "#009E73", "mmi_pid": "#B07AA1", "surd": "#8C8C8C", "wms": "#4C78A8"}
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.65), constrained_layout=True)
    for ax, metric in zip(axes, METRICS, strict=True):
        methods = list(METHODS if metric == "synergy" else METHODS[:-1])
        values = [float(result["sensitivity"][method][metric]["absolute_tv"]) for method in methods]
        bars = ax.bar(range(len(methods)), values, color=[colors[method] for method in methods], width=0.68)
        ax.set_xticks(range(len(methods)), [LABELS[method] for method in methods], rotation=24, ha="right")
        ax.set_ylabel("Absolute TV (bits)")
        ax.set_title({"synergy": "Synergy", "unique_x": r"$U_x$", "unique_y": r"$U_y$"}[metric])
        ax.grid(axis="y", alpha=0.17, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=6)
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / f"{FIGURE_STEM}_absolute_tv.png"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(figure_dir / f"{FIGURE_STEM}_absolute_tv.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / f"{FIGURE_STEM}_absolute_tv.svg", bbox_inches="tight")
    plt.close(fig)
    return png


def validate_selected(dynamics: Dynamics, *, quick: bool = False) -> dict[str, object]:
    beta_values = tuple(float(value) for value in np.linspace(0.0, 1.0, 5 if quick else 21))
    settings = {
        "n_samples": 500 if quick else 1100,
        "mlp_epochs": 12 if quick else 90,
        "context_anchors": 6 if quick else 16,
        "grid_levels": 13 if quick else 25,
        "surd_target_anchors": 64 if quick else 128,
        "surd_conditional_samples": 32 if quick else 64,
    }
    held_out = run_candidate(
        dynamics,
        beta_values=beta_values,
        seeds=(2, 3),
        **settings,
    )
    passed = bool(float(held_out["minimum_gap"]) > 0.0)
    _append_history(
        {
            "phase": "held_out_quick" if quick else "held_out",
            "candidate_id": dynamics.candidate_id,
            "passed": passed,
            "minimum_gap": held_out["minimum_gap"],
            "robustness_gaps": held_out["robustness_gaps"],
            "mean_z_train_r2": held_out["mean_z_train_r2"],
        }
    )
    result: dict[str, object] = {
        "contract": {
            "comparison": "end-to-end robustness under a shared one-decimal DGP",
            "oracle_used": False,
            "selection_seeds": [0, 1],
            "held_out_seeds": [2, 3],
            "confirmatory_seeds": [4, 5],
            "beta_values": list(beta_values),
            "fixed_source_support": list(COMMON_SOURCE_SUPPORT),
            "criterion": "MLP+PEID absolute TV is strictly lowest for synergy, Ux, and Uy",
        },
        "equations": {
            "w": "w[t+1] = a_w w[t] + eta_w",
            "x": "x[t+1] = a_x x[t] + c_s(beta w[t] + sqrt(1-beta^2) xi_x[t]) + eta_x",
            "y": "y[t+1] = a_x y[t] + c_s(beta w[t] + sqrt(1-beta^2) xi_y[t]) + eta_y",
            "z": "z[t+1] = a_z z[t] + c_xy sin(x[t] y[t]) + c_z beta w[t] + eta_z",
        },
        "dynamics": asdict(dynamics),
        "held_out": held_out,
        "held_out_passed": passed,
    }
    if not passed or quick:
        return result
    confirmatory = run_candidate(
        dynamics,
        beta_values=beta_values,
        seeds=(4, 5),
        **settings,
    )
    confirmatory_passed = bool(float(confirmatory["minimum_gap"]) > 0.0)
    result["confirmatory"] = confirmatory
    result["confirmatory_passed"] = confirmatory_passed
    _append_history(
        {
            "phase": "confirmatory",
            "candidate_id": dynamics.candidate_id,
            "passed": confirmatory_passed,
            "minimum_gap": confirmatory["minimum_gap"],
            "robustness_gaps": confirmatory["robustness_gaps"],
            "mean_z_train_r2": confirmatory["mean_z_train_r2"],
        }
    )
    if not confirmatory_passed:
        return result
    final = run_candidate(
        dynamics,
        beta_values=beta_values,
        seeds=(0, 1, 2, 3, 4, 5),
        **settings,
    )
    result["final_all_seeds"] = final
    _write_json(RESULT_PATH, result)
    plot_curves(final, DEFAULT_FIGURE_DIR)
    plot_sensitivity(final, DEFAULT_FIGURE_DIR)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "search", "validate"), required=True)
    parser.add_argument("--source-loading", type=float)
    parser.add_argument("--target-loading", type=float)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"smoke", "search"}:
        result = run_search(smoke=args.mode == "smoke")
    else:
        if args.source_loading is None or args.target_loading is None:
            raise ValueError("validate mode requires --source-loading and --target-loading")
        dynamics = Dynamics(
            driver_to_sources=float(args.source_loading),
            driver_to_target=float(args.target_loading),
        )
        result = validate_selected(dynamics, quick=bool(args.quick))
    print(
        json.dumps(
            {
                "mode": args.mode,
                "selected": result.get("selected", result.get("dynamics")),
                "held_out_passed": result.get("held_out_passed"),
                "result_path": str(RESULT_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
