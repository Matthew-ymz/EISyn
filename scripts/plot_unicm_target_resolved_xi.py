from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_unicm_all_mode_target_pair_syn import parse_leads  # noqa: E402
from scripts.plot_unicm_all_mode_target_phi_eid import compute_phi_eid_for_target  # noqa: E402
from scripts.plot_unicm_fixed_module_xi import precompute_selected_source_logdets  # noqa: E402
from scripts.plot_unicm_phi_eid_greedy_decomposition import (  # noqa: E402
    compute_subset_ei_table_from_covariance,
)
from scripts.unicm_peid_syn_analysis import (  # noqa: E402
    MODE_NAMES,
    create_ei_estimator,
    load_full_history_prediction_cache,
    overall_prediction_cache_path,
    sample_full_history_mode_inputs,
)

DEFAULT_CACHE_DIR = ROOT / "results" / "unicm_overall_ei_cpu_bound4_n8192" / "cache"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "unicm_target_resolved_xi_tm_degree1_signed_n8192"
DEFAULT_ASSET_BASE = ROOT / "fig" / "unicm_target_resolved_xi"

Estimator = Callable[[np.ndarray, np.ndarray], float]

DISPLAY_NAME = {
    "nino": "ENSO",
    "NPMM": "NPMM",
    "SPMM": "SPMM",
    "IOB": "IOB",
    "IOD": "IOD",
    "SIOD": "SIOD",
    "TNA": "TNA",
    "nino12": "nino12",
    "nino3": "nino3",
    "nino4": "nino4",
    "WWV": "WWV",
}

TARGET_DISPLAY_ORDER = (
    "nino",
    "nino12",
    "nino3",
    "nino4",
    "WWV",
    "NPMM",
    "SPMM",
    "IOB",
    "IOD",
    "SIOD",
    "TNA",
)


def extract_target_mode(
    predictions: np.ndarray,
    *,
    lead: int,
    target_index: int,
) -> np.ndarray:
    array = np.asarray(predictions, dtype=float)
    if array.ndim != 3:
        raise ValueError("predictions must have shape (n_samples, n_leads, n_modes).")
    if int(lead) < 1 or int(lead) > array.shape[1]:
        raise ValueError(f"lead must be in [1, {array.shape[1]}], got {lead}.")
    if int(target_index) < 0 or int(target_index) >= array.shape[2]:
        raise ValueError(f"target_index must be in [0, {array.shape[2] - 1}], got {target_index}.")
    return array[:, int(lead) - 1, int(target_index) : int(target_index) + 1]


def compute_target_resolved_rows(
    history_modes: np.ndarray,
    targets_by_seed: Mapping[int, np.ndarray],
    *,
    mode_names: Mapping[str, int] = MODE_NAMES,
    leads: Sequence[int] | None = None,
    estimator_name: str = "transport_map",
    tm_degree: int = 1,
    tm_jitter: float = 1.0e-6,
    estimator: Estimator | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    history = np.asarray(history_modes, dtype=float)
    if history.ndim != 3:
        raise ValueError("history_modes must have shape (n_samples, history_length, n_modes).")
    if history.shape[2] != len(mode_names):
        raise ValueError("history_modes mode axis must match mode_names.")

    lead_values = list(range(1, 25)) if leads is None else [int(value) for value in leads]
    ordered_targets = sorted(mode_names.items(), key=lambda item: int(item[1]))
    all_sources = tuple(name for name, _ in ordered_targets)
    required_subsets = tuple((name,) for name in all_sources) + (all_sources,)

    if estimator is None:
        estimator, estimator_metadata = create_ei_estimator(
            str(estimator_name),
            tm_degree=int(tm_degree),
            tm_jitter=float(tm_jitter),
            clip_negative=False,
        )
    else:
        estimator_metadata = {
            "estimator": str(estimator_name),
            "backend": "caller_supplied",
            "tm_degree": int(tm_degree),
            "tm_jitter": float(tm_jitter),
            "clip_negative": False,
        }

    covariance_inputs = None
    if (
        estimator is not None
        and str(estimator_name) == "transport_map"
        and int(tm_degree) == 1
        and mode_names == MODE_NAMES
    ):
        covariance_inputs = precompute_selected_source_logdets(
            history,
            required_subsets,
            jitter=float(tm_jitter),
        )
        estimator_metadata = {
            "estimator": "transport_map",
            "backend": "affine_triangular_transport_map_degree_1_fast_logdet_equivalent",
            "tm_degree": 1,
            "tm_jitter": float(tm_jitter),
            "clip_negative": False,
            "note": "Uses the covariance/log-det closed form equivalent to an affine triangular TM.",
        }

    rows: list[dict[str, object]] = []
    for seed in sorted(int(value) for value in targets_by_seed):
        predictions = np.asarray(targets_by_seed[seed], dtype=float)
        for lead in lead_values:
            for target_name, target_index in ordered_targets:
                target = extract_target_mode(
                    predictions,
                    lead=int(lead),
                    target_index=int(target_index),
                )
                if covariance_inputs is not None:
                    history_flat, subset_columns, source_logdets = covariance_inputs
                    ei_lookup = compute_subset_ei_table_from_covariance(
                        history_flat,
                        target,
                        subset_columns,
                        source_logdets,
                        jitter=float(tm_jitter),
                    )
                    whole_ei = float(ei_lookup[all_sources])
                    singleton_sum = float(sum(ei_lookup[(source,)] for source in all_sources))
                    singleton_values = {source: float(ei_lookup[(source,)]) for source in all_sources}
                else:
                    metrics = compute_phi_eid_for_target(
                        history,
                        target,
                        mode_names=mode_names,
                        estimator=estimator,
                    )
                    whole_ei = float(metrics["whole_ei"])
                    singleton_sum = float(metrics["singleton_ei_sum"])
                    singleton_values = {
                        str(source): float(value)
                        for source, value in dict(metrics["singleton_ei"]).items()
                    }

                xi = float(whole_ei - singleton_sum)
                dominant_source = max(singleton_values, key=singleton_values.get)
                rows.append(
                    {
                        "seed": int(seed),
                        "lead": int(lead),
                        "target": str(target_name),
                        "display_target": DISPLAY_NAME.get(str(target_name), str(target_name)),
                        "target_index": int(target_index),
                        "target_dim": 1,
                        "whole_ei": whole_ei,
                        "singleton_ei_sum": singleton_sum,
                        "xi_target": xi,
                        "dominant_singleton_source": dominant_source,
                        "dominant_singleton_ei": float(singleton_values[dominant_source]),
                    }
                )

    frame = pd.DataFrame(rows)
    return (
        frame.sort_values(["seed", "target_index", "lead"]).reset_index(drop=True),
        estimator_metadata,
    )


def summarize_target_resolved_rows(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lead_summary = (
        rows.groupby(
            ["target", "display_target", "target_index", "lead"],
            as_index=False,
        )
        .agg(
            xi_mean=("xi_target", "mean"),
            xi_std=("xi_target", "std"),
            xi_min=("xi_target", "min"),
            xi_max=("xi_target", "max"),
            positive_seed_count=("xi_target", lambda values: int((np.asarray(values) > 0).sum())),
            whole_ei_mean=("whole_ei", "mean"),
            singleton_ei_sum_mean=("singleton_ei_sum", "mean"),
        )
        .sort_values(["target_index", "lead"])
        .reset_index(drop=True)
    )
    lead_summary["xi_std"] = lead_summary["xi_std"].fillna(0.0)

    window_rows = rows[rows["lead"].between(7, 10)].copy()
    seed_window = (
        window_rows.groupby(
            ["seed", "target", "display_target", "target_index"],
            as_index=False,
        )["xi_target"]
        .mean()
        .rename(columns={"xi_target": "window_mean_xi"})
    )
    window_summary = (
        seed_window.groupby(
            ["target", "display_target", "target_index"],
            as_index=False,
        )
        .agg(
            window_mean_xi=("window_mean_xi", "mean"),
            window_std_xi=("window_mean_xi", "std"),
            positive_seed_count=("window_mean_xi", lambda values: int((np.asarray(values) > 0).sum())),
        )
        .sort_values("target_index")
        .reset_index(drop=True)
    )
    window_summary["window_std_xi"] = window_summary["window_std_xi"].fillna(0.0)

    peak_summary = (
        lead_summary.loc[lead_summary.groupby("target")["xi_mean"].idxmax()]
        .sort_values(["xi_mean", "target_index"], ascending=[False, True])
        .reset_index(drop=True)
    )
    peak_summary.insert(0, "rank", np.arange(1, len(peak_summary) + 1))
    return lead_summary, seed_window, window_summary.merge(
        peak_summary[
            [
                "target",
                "lead",
                "xi_mean",
                "xi_std",
                "positive_seed_count",
            ]
        ].rename(
            columns={
                "lead": "peak_lead",
                "xi_mean": "peak_xi_mean",
                "xi_std": "peak_xi_std",
                "positive_seed_count": "peak_positive_seed_count",
            }
        ),
        on="target",
        how="left",
        validate="one_to_one",
    )


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def plot_target_resolved_xi(
    rows: pd.DataFrame,
    lead_summary: pd.DataFrame,
    seed_window: pd.DataFrame,
    window_summary: pd.DataFrame,
    output_base: Path,
) -> list[Path]:
    configure_matplotlib()
    available = (
        lead_summary[["target", "display_target", "target_index"]]
        .drop_duplicates()
        .set_index("target")
    )
    requested = [target for target in TARGET_DISPLAY_ORDER if target in available.index]
    remaining = [target for target in available.index if target not in requested]
    target_order = available.loc[[*requested, *remaining]].reset_index()
    targets = target_order["target"].tolist()
    labels = target_order["display_target"].tolist()
    heat = (
        lead_summary.pivot(index="target", columns="lead", values="xi_mean")
        .reindex(targets)
        .sort_index(axis=1)
    )

    values = heat.to_numpy(dtype=float)
    value_min = float(np.nanmin(values))
    value_max = float(np.nanmax(values))
    if not np.isfinite(value_max) or value_max <= 0:
        value_max = 1.0
    if value_min < 0.0 < value_max:
        absolute_limit = max(abs(value_min), abs(value_max))
        color_norm = TwoSlopeNorm(vmin=-absolute_limit, vcenter=0.0, vmax=absolute_limit)
        color_map = "RdBu_r"
    else:
        color_norm = mpl.colors.Normalize(vmin=min(0.0, value_min), vmax=value_max)
        color_map = "YlOrRd"

    fig = plt.figure(figsize=(7.2, 3.45), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[4.6, 1.55])
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_window = fig.add_subplot(grid[0, 1], sharey=ax_heat)

    image = ax_heat.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
        norm=color_norm,
    )
    ax_heat.set_yticks(np.arange(len(labels)))
    ax_heat.set_yticklabels(labels)
    ax_heat.set_xticks(np.arange(0, len(heat.columns), 2))
    ax_heat.set_xticklabels([str(int(value)) for value in heat.columns[::2]])
    ax_heat.set_xlabel("Prediction lead (months)")
    ax_heat.set_ylabel("Predicted target mode")
    ax_heat.axvline(5.5, color="#2B2B2B", linewidth=0.55, linestyle=":")
    ax_heat.axvline(9.5, color="#2B2B2B", linewidth=0.55, linestyle=":")
    for boundary in (4.5, 6.5, 9.5):
        ax_heat.axhline(boundary, color="white", linewidth=0.8)
    ax_heat.text(
        7.5,
        -0.86,
        "lead 7–10",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#444444",
        clip_on=False,
    )
    ax_heat.text(-0.11, 1.025, "a", transform=ax_heat.transAxes, fontweight="bold", fontsize=8)
    colorbar = fig.colorbar(
        image,
        ax=ax_heat,
        orientation="horizontal",
        fraction=0.055,
        pad=0.11,
        aspect=35,
    )
    colorbar.set_label(r"Target-resolved $\Xi_j$ (bits)")

    summary_lookup = window_summary.set_index("target")
    seed_lookup = seed_window.copy()
    y_positions = np.arange(len(targets))
    for y, target in enumerate(targets):
        seed_values = seed_lookup.loc[
            seed_lookup["target"].eq(target),
            "window_mean_xi",
        ].to_numpy(dtype=float)
        ax_window.scatter(
            seed_values,
            np.full(seed_values.shape, y),
            s=10,
            color="#9AA0A6",
            alpha=0.8,
            zorder=2,
        )
        current = summary_lookup.loc[target]
        ax_window.errorbar(
            float(current["window_mean_xi"]),
            y,
            xerr=float(current["window_std_xi"]),
            fmt="D",
            markersize=3.0,
            color="#D07A55",
            ecolor="#D07A55",
            elinewidth=0.8,
            capsize=1.6,
            zorder=3,
        )
    ax_window.axvline(0.0, color="#777777", linewidth=0.65, linestyle=":")
    ax_window.set_ylim(len(targets) - 0.5, -0.5)
    ax_window.tick_params(axis="y", left=False, labelleft=False)
    ax_window.set_xlabel(r"Lead 7–10 mean $\Xi_j$ (bits)")
    ax_window.grid(axis="x", color="#E4E7EA", linewidth=0.45)
    ax_window.scatter([], [], s=10, color="#9AA0A6", label="checkpoint")
    ax_window.errorbar(
        [],
        [],
        xerr=[],
        fmt="D",
        markersize=3.0,
        color="#D07A55",
        label="mean ± s.d.",
    )
    ax_window.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        borderaxespad=0.0,
        fontsize=6.1,
    )
    ax_window.text(-0.18, 1.025, "b", transform=ax_window.transAxes, fontweight="bold", fontsize=8)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_base.with_suffix(".png"),
        output_base.with_suffix(".svg"),
        output_base.with_suffix(".pdf"),
    ]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], bbox_inches="tight")
    plt.close(fig)
    return outputs


def write_experiment_contract(
    path: Path,
    args: argparse.Namespace,
    estimator_metadata: Mapping[str, object],
) -> None:
    lines = [
        "# UniCM target-resolved controlled-comparison contract",
        "",
        "| Field | Frozen value |",
        "|---|---|",
        "| Scientific question | What changes when only the scalar predicted target mode changes? |",
        "| Treatment factor | Target readout $Y_j$ |",
        "| Treatment levels | ENSO, NPMM, SPMM, IOB, IOD, SIOD, TNA, nino12, nino3, nino4, WWV |",
        "| Unit of pairing | checkpoint seed × prediction lead |",
        "| Primary metric | signed $\\Xi_j=EI(\\mathbf{X}_{1:11}\\to Y_j)-\\sum_m EI(X_m\\to Y_j)$ |",
        f"| Seeds | {', '.join(str(int(seed)) for seed in args.seeds)} |",
        f"| Leads | {', '.join(str(value) for value in parse_leads(args.leads))} |",
        f"| Samples | {int(args.n_samples)} shared maximum-entropy intervention samples |",
        f"| Intervention support | independent uniform $[-{float(args.intervention_bound):g},{float(args.intervention_bound):g}]$ for every mode-month |",
        f"| Sampling seed | {int(args.sampling_seed)} |",
        f"| Estimator | `{json.dumps(dict(estimator_metadata), ensure_ascii=False)}` |",
        "| Model and predictions | frozen UniCM checkpoints; shared cached predictions; no new forward pass |",
        "| Source partition | all 11 mode histories, each containing the same 12 months |",
        "| Postprocessing | signed values; no clipping; identical heatmap scale for all targets and leads |",
        "| Main limitation | $\\sum_j\\Xi_j$ is not equal to the multivariate-target $\\Xi$ because target dependence is retained only in the latter |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    path: Path,
    lead_summary: pd.DataFrame,
    window_summary: pd.DataFrame,
) -> None:
    ranked = window_summary.sort_values("window_mean_xi", ascending=False)
    strongest = ranked.iloc[0]
    weakest = ranked.iloc[-1]
    peak = lead_summary.loc[lead_summary["xi_mean"].idxmax()]
    lines = [
        "# UniCM target-resolved Xi report",
        "",
        "## Stable findings",
        "",
        (
            f"- The largest target-resolved value occurs for {peak.display_target} at lead "
            f"{int(peak.lead)}: {float(peak.xi_mean):.6f} ± {float(peak.xi_std):.6f} bits "
            "(checkpoint mean ± s.d.)."
        ),
        (
            f"- In the lead 7–10 window, {strongest.display_target} has the largest mean "
            f"$\\Xi_j$ ({float(strongest.window_mean_xi):.6f} ± "
            f"{float(strongest.window_std_xi):.6f} bits), whereas "
            f"{weakest.display_target} has the smallest "
            f"({float(weakest.window_mean_xi):.6f} ± "
            f"{float(weakest.window_std_xi):.6f} bits)."
        ),
        "",
        "## Lead 7–10 target ranking",
        "",
        "| Rank | Target | Mean Xi_j (bits) | s.d. | Positive checkpoints | Peak lead | Peak Xi_j (bits) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked.itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | {row.display_target} | {float(row.window_mean_xi):.6f} | "
            f"{float(row.window_std_xi):.6f} | {int(row.positive_seed_count)}/3 | "
            f"{int(row.peak_lead)} | {float(row.peak_xi_mean):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Positive Xi_j means the frozen model reads the 11 histories jointly for that scalar target beyond the sum of singleton EIs under the selected signed affine-TM definition.",
            "- Negative Xi_j is retained and indicates redundancy-dominated accounting under this definition; it is not a failed estimate and is not clipped.",
            "- Target-resolved Xi_j values are not additive components of the multivariate-target Xi. They localize receiving targets but cannot be summed into the joint-target result.",
            "- Three checkpoints quantify model-seed variability only. Degree-2/3 TM checks remain necessary before treating small target differences as nonlinear-mechanism evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _cache_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        n_samples=int(args.n_samples),
        sampling_seed=int(args.sampling_seed),
        intervention_bound=float(args.intervention_bound),
        start_month=int(args.start_month),
        device=str(args.device),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    leads = parse_leads(args.leads)
    seeds = [int(seed) for seed in args.seeds]
    history_modes = sample_full_history_mode_inputs(
        n_samples=int(args.n_samples),
        intervention_bound=float(args.intervention_bound),
        seed=int(args.sampling_seed),
    )
    cache_args = _cache_args(args)
    targets_by_seed = {
        seed: load_full_history_prediction_cache(
            overall_prediction_cache_path(Path(args.cache_dir), seed=seed, args=cache_args),
            n_samples=int(args.n_samples),
        )
        for seed in seeds
    }
    rows, estimator_metadata = compute_target_resolved_rows(
        history_modes,
        targets_by_seed,
        leads=leads,
        estimator_name=str(args.estimator),
        tm_degree=int(args.tm_degree),
        tm_jitter=float(args.tm_jitter),
    )
    lead_summary, seed_window, window_summary = summarize_target_resolved_rows(rows)

    paths = {
        "rows": output_dir / "target_resolved_xi_rows.csv",
        "lead_summary": output_dir / "target_resolved_xi_lead_summary.csv",
        "seed_window": output_dir / "target_resolved_xi_lead7_10_by_seed.csv",
        "window_summary": output_dir / "target_resolved_xi_target_summary.csv",
        "experiment_contract": output_dir / "experiment_contract.md",
        "report": output_dir / "report.md",
    }
    rows.to_csv(paths["rows"], index=False)
    lead_summary.to_csv(paths["lead_summary"], index=False)
    seed_window.to_csv(paths["seed_window"], index=False)
    window_summary.to_csv(paths["window_summary"], index=False)
    write_experiment_contract(paths["experiment_contract"], args, estimator_metadata)
    write_report(paths["report"], lead_summary, window_summary)
    figures = plot_target_resolved_xi(
        rows,
        lead_summary,
        seed_window,
        window_summary,
        Path(args.asset_base),
    )

    manifest = {
        "analysis": "UniCM target-resolved integrated effective-information increment",
        "definition": "Xi_j = EI(all 11 mode histories -> target mode j) - sum_m EI(mode m history -> target mode j)",
        "target_modes": list(MODE_NAMES),
        "seeds": seeds,
        "leads": leads,
        "n_samples": int(args.n_samples),
        "sampling_seed": int(args.sampling_seed),
        "intervention_bound": float(args.intervention_bound),
        "signed_outputs": True,
        "estimator": estimator_metadata,
        "cache_dir": str(args.cache_dir),
        "tables": {key: str(value) for key, value in paths.items()},
        "figures": [str(path) for path in figures],
        "nonadditivity_warning": "Sum_j Xi_j is not the multivariate-target Xi.",
    }
    manifest_path = output_dir / "target_resolved_xi_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute target-resolved Xi for frozen UniCM checkpoints.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--asset-base", type=Path, default=DEFAULT_ASSET_BASE)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--leads", nargs="*", default=None)
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument("--sampling-seed", type=int, default=20260619)
    parser.add_argument("--intervention-bound", type=float, default=4.0)
    parser.add_argument("--start-month", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--estimator", choices=["gaussian_logdet", "transport_map"], default="transport_map")
    parser.add_argument("--tm-degree", type=int, default=1)
    parser.add_argument("--tm-jitter", type=float, default=1.0e-6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    outputs = run(build_arg_parser().parse_args(argv))
    print(json.dumps(outputs, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
