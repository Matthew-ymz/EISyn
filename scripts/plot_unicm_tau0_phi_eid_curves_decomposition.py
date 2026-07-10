from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_unicm_all_mode_target_pair_syn import extract_all_mode_target, parse_leads  # noqa: E402
from scripts.plot_unicm_month_resolved_phi_eid import history_index_from_tau  # noqa: E402
from scripts.plot_unicm_phi_eid_greedy_decomposition import (  # noqa: E402
    _all_mode_subsets,
    _regularized_logdet,
    compute_subset_ei_table_from_covariance,
    greedy_phi_atoms,
    summarize_atom_rows,
    subset_phi_raw,
)
from scripts.unicm_peid_syn_analysis import (  # noqa: E402
    MODE_NAMES,
    load_full_history_prediction_cache,
    overall_prediction_cache_path,
    sample_full_history_mode_inputs,
)

DEFAULT_CACHE_DIR = ROOT / "results" / "unicm_overall_ei_cpu_bound4_n8192" / "cache"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "unicm_tau0_phi_eid_curves_decomposition_tm_degree1_signed_n8192"
DEFAULT_ASSET_BASE = ROOT / "fig" / "unicm_tau0_phi_eid_curves_decomposition_tm_degree1_signed"


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
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def precompute_tau_mode_subset_logdets(
    history_modes: np.ndarray,
    *,
    tau: int = 0,
    mode_names: Mapping[str, int] = MODE_NAMES,
    jitter: float = 1.0e-6,
) -> tuple[np.ndarray, dict[tuple[str, ...], list[int]], dict[tuple[str, ...], float]]:
    history = np.asarray(history_modes, dtype=float)
    if history.ndim != 3:
        raise ValueError("history_modes must have shape (n_samples, history_length, n_modes).")
    month_index = history_index_from_tau(int(tau))
    source = history[:, month_index, :]
    source_cov = np.cov(source, rowvar=False, bias=False)
    subsets = _all_mode_subsets(tuple(mode_names))
    columns = {subset: [int(mode_names[name]) for name in subset] for subset in subsets}
    logdets = {
        subset: _regularized_logdet(source_cov[np.ix_(cols, cols)], jitter=jitter)
        for subset, cols in columns.items()
    }
    return source, columns, logdets


def select_top_phi_leads(rows: Sequence[Mapping[str, object]] | pd.DataFrame, *, top_k: int) -> list[int]:
    frame = pd.DataFrame(rows)
    if "lead" not in frame or "phi_eid_mean" not in frame:
        raise ValueError("rows must contain lead and phi_eid_mean columns.")
    ordered = frame.sort_values(["phi_eid_mean", "lead"], ascending=[False, True])
    return [int(value) for value in ordered["lead"].head(int(top_k)).tolist()]


def summarize_curves(rows: pd.DataFrame, singleton_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_summary = rows.groupby("lead", as_index=False).agg(
        whole_ei_mean=("whole_ei", "mean"),
        whole_ei_std=("whole_ei", "std"),
        singleton_ei_sum_mean=("singleton_ei_sum", "mean"),
        singleton_ei_sum_std=("singleton_ei_sum", "std"),
        raw_phi_eid_mean=("raw_phi_eid", "mean"),
        raw_phi_eid_std=("raw_phi_eid", "std"),
        phi_eid_mean=("phi_eid", "mean"),
        phi_eid_std=("phi_eid", "std"),
    )
    singleton_summary = singleton_rows.groupby(["lead", "source"], as_index=False).agg(
        singleton_ei_mean=("singleton_ei", "mean"),
        singleton_ei_std=("singleton_ei", "std"),
    )
    for frame in (curve_summary, singleton_summary):
        for column in frame.columns:
            if column.endswith("_std"):
                frame[column] = frame[column].fillna(0.0)
    return curve_summary.sort_values("lead").reset_index(drop=True), singleton_summary.sort_values(["source", "lead"]).reset_index(drop=True)


def compute_tau0_rows(
    source: np.ndarray,
    subset_columns: Mapping[tuple[str, ...], list[int]],
    source_logdets: Mapping[tuple[str, ...], float],
    targets_by_seed: Mapping[int, np.ndarray],
    *,
    leads: Sequence[int],
    mode_names: Mapping[str, int] = MODE_NAMES,
    jitter: float = 1.0e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, int], dict[tuple[str, ...], float]]]:
    full_subset = tuple(mode_names)
    rows: list[dict[str, object]] = []
    singleton_rows: list[dict[str, object]] = []
    ei_tables: dict[tuple[int, int], dict[tuple[str, ...], float]] = {}
    for seed in sorted(int(seed) for seed in targets_by_seed):
        predictions = np.asarray(targets_by_seed[seed], dtype=float)
        for lead in [int(value) for value in leads]:
            target = extract_all_mode_target(predictions, lead=lead)
            ei_table = compute_subset_ei_table_from_covariance(
                source,
                target,
                subset_columns,
                source_logdets,
                jitter=float(jitter),
            )
            ei_tables[(seed, lead)] = ei_table
            singleton_ei = {name: float(ei_table[(name,)]) for name in mode_names}
            singleton_sum = float(sum(singleton_ei.values()))
            raw_phi = subset_phi_raw(full_subset, ei_table, singleton_ei)
            rows.append(
                {
                    "seed": seed,
                    "lead": lead,
                    "whole_ei": float(ei_table[full_subset]),
                    "singleton_ei_sum": singleton_sum,
                    "raw_phi_eid": float(raw_phi),
                    "phi_eid": float(raw_phi),
                    "phi_to_whole": float(raw_phi / ei_table[full_subset]) if abs(float(ei_table[full_subset])) > 1.0e-12 else np.nan,
                    "phi_to_singleton_sum": float(raw_phi / singleton_sum) if abs(singleton_sum) > 1.0e-12 else np.nan,
                }
            )
            for mode_name, value in singleton_ei.items():
                singleton_rows.append(
                    {
                        "seed": seed,
                        "lead": lead,
                        "source": mode_name,
                        "singleton_ei": float(value),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(singleton_rows), ei_tables


def compute_tau0_greedy_atoms(
    ei_tables: Mapping[tuple[int, int], Mapping[tuple[str, ...], float]],
    *,
    selected_leads: Sequence[int],
    seeds: Sequence[int],
    mode_names: Mapping[str, int] = MODE_NAMES,
    eps: float = 1.0e-5,
    split_tolerance: float = 1.0e-4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_subset = tuple(mode_names)
    atom_rows: list[dict[str, object]] = []
    total_rows: list[dict[str, object]] = []
    for seed in [int(value) for value in seeds]:
        for lead in [int(value) for value in selected_leads]:
            ei_table = ei_tables[(seed, lead)]
            singleton_ei = {name: float(ei_table[(name,)]) for name in mode_names}
            raw_phi = subset_phi_raw(full_subset, ei_table, singleton_ei)
            atoms = greedy_phi_atoms(
                full_subset,
                ei_table,
                eps=float(eps),
                split_tolerance=float(split_tolerance),
                singleton_ei=singleton_ei,
            )
            atom_sum = float(sum(atom.value for atom in atoms))
            total_rows.append(
                {
                    "seed": seed,
                    "lead": lead,
                    "whole_ei": float(ei_table[full_subset]),
                    "singleton_ei_sum": float(sum(singleton_ei.values())),
                    "raw_phi_eid": float(raw_phi),
                    "phi_eid": float(max(0.0, raw_phi)),
                    "phi_atom_sum": atom_sum,
                    "residual_to_phi": float(max(0.0, raw_phi) - atom_sum),
                    "n_atoms": int(len(atoms)),
                }
            )
            for atom in atoms:
                atom_rows.append(
                    {
                        "seed": seed,
                        "lead": lead,
                        "sources": "|".join(atom.sources),
                        "order": int(len(atom.sources)),
                        "value": float(atom.value),
                        "fraction_of_phi": float(atom.value / raw_phi) if raw_phi > 0 else 0.0,
                        "kind": atom.kind,
                        "depth": int(atom.depth),
                    }
                )
    atoms = pd.DataFrame(atom_rows)
    totals = pd.DataFrame(total_rows).sort_values(["seed", "lead"]).reset_index(drop=True)
    total_summary = totals.groupby("lead", as_index=False).agg(
        phi_eid_mean=("phi_eid", "mean"),
        phi_atom_sum_mean=("phi_atom_sum", "mean"),
        n_atoms_mean=("n_atoms", "mean"),
    )
    lead_order, module_summary, module_lead = summarize_atom_rows(atoms, totals)
    return atoms, totals, total_summary, lead_order, module_summary, module_lead


def plot_tau0_curves(
    curve_summary: pd.DataFrame,
    singleton_summary: pd.DataFrame,
    selected_leads: Sequence[int],
    output_base: Path,
) -> list[Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.0), constrained_layout=True, sharex=True)
    x = curve_summary["lead"].to_numpy(dtype=float)
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 0.95, len(MODE_NAMES)))
    for color, mode_name in zip(colors, MODE_NAMES):
        part = singleton_summary[singleton_summary["source"].eq(mode_name)].sort_values("lead")
        axes[0].plot(
            part["lead"],
            part["singleton_ei_mean"],
            linewidth=1.05,
            color=color,
            label=str(mode_name),
        )
    axes[0].set_ylabel("Singleton EI (bits)")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1)

    axes[1].plot(x, curve_summary["whole_ei_mean"], color="#555555", linewidth=1.45, label="Whole EI")
    axes[1].fill_between(
        x,
        curve_summary["whole_ei_mean"] - curve_summary["whole_ei_std"],
        curve_summary["whole_ei_mean"] + curve_summary["whole_ei_std"],
        color="#555555",
        alpha=0.10,
        linewidth=0,
    )
    axes[1].plot(x, curve_summary["singleton_ei_sum_mean"], color="#B279A2", linestyle=":", linewidth=1.35, label="Singleton EI sum")
    axes[1].set_ylabel("EI terms (bits)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    axes[2].plot(x, curve_summary["phi_eid_mean"], color="#4C78A8", marker="o", markersize=2.4, linewidth=1.5, label=r"$\Phi^{EID}$")
    axes[2].fill_between(
        x,
        curve_summary["phi_eid_mean"] - curve_summary["phi_eid_std"],
        curve_summary["phi_eid_mean"] + curve_summary["phi_eid_std"],
        color="#4C78A8",
        alpha=0.16,
        linewidth=0,
    )
    for lead in selected_leads:
        axes[2].axvline(int(lead), color="#9A3412", linewidth=0.75, linestyle="--", alpha=0.55)
    axes[2].axhline(0.0, color="#888888", linewidth=0.7, linestyle=":")
    axes[2].set_xlabel("Lead (months)")
    axes[2].set_ylabel(r"$\Phi^{EID}$ (bits)")
    axes[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_name(output_base.name + "_curves").with_suffix(".png"), output_base.with_name(output_base.name + "_curves").with_suffix(".svg")]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_tau0_decomposition(
    lead_order: pd.DataFrame,
    module_summary: pd.DataFrame,
    module_lead: pd.DataFrame,
    output_base: Path,
    *,
    top_k: int = 12,
) -> list[Path]:
    configure_matplotlib()
    fig, (ax_order, ax_heat) = plt.subplots(1, 2, figsize=(9.2, 4.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.0, 1.55]})
    order_pivot = lead_order.pivot(index="lead", columns="order", values="mean").fillna(0.0).sort_index()
    bottom = np.zeros(order_pivot.shape[0], dtype=float)
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, max(1, len(order_pivot.columns))))
    x = np.arange(order_pivot.shape[0])
    for color, order in zip(colors, order_pivot.columns):
        values = order_pivot[order].to_numpy(dtype=float)
        ax_order.bar(x, values, bottom=bottom, color=color, width=0.72, label=f"order {int(order)}")
        bottom += values
    ax_order.set_xticks(x)
    ax_order.set_xticklabels([str(int(value)) for value in order_pivot.index])
    ax_order.set_xlabel("Selected lead")
    ax_order.set_ylabel(r"Greedy atoms (bits)")
    ax_order.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    top_modules = module_summary.head(int(top_k))["sources"].astype(str).tolist()
    heat = (
        module_lead[module_lead["sources"].astype(str).isin(top_modules)]
        .pivot(index="sources", columns="lead", values="mean")
        .reindex(top_modules)
        .fillna(0.0)
    )
    image = ax_heat.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="magma", interpolation="nearest")
    ax_heat.set_yticks(np.arange(len(heat.index)))
    ax_heat.set_yticklabels([label.replace("|", " + ") for label in heat.index])
    ax_heat.set_xticks(np.arange(len(heat.columns)))
    ax_heat.set_xticklabels([str(int(value)) for value in heat.columns])
    ax_heat.set_xlabel("Selected lead")
    ax_heat.set_ylabel("Top atoms")
    fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.02, label="bits")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        output_base.with_name(output_base.name + "_decomposition").with_suffix(".png"),
        output_base.with_name(output_base.name + "_decomposition").with_suffix(".svg"),
    ]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


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
    source, subset_columns, source_logdets = precompute_tau_mode_subset_logdets(
        history_modes,
        tau=int(args.tau),
        jitter=float(args.jitter),
    )
    cache_args = _cache_args(args)
    targets_by_seed = {
        seed: load_full_history_prediction_cache(
            overall_prediction_cache_path(Path(args.cache_dir), seed=seed, args=cache_args),
            n_samples=int(args.n_samples),
        )
        for seed in seeds
    }
    rows, singleton_rows, ei_tables = compute_tau0_rows(
        source,
        subset_columns,
        source_logdets,
        targets_by_seed,
        leads=leads,
        jitter=float(args.jitter),
    )
    curve_summary, singleton_summary = summarize_curves(rows, singleton_rows)
    selected_leads = parse_leads(args.decomposition_leads) if args.decomposition_leads else select_top_phi_leads(curve_summary, top_k=int(args.top_leads))
    atoms, totals, total_summary, lead_order, module_summary, module_lead = compute_tau0_greedy_atoms(
        ei_tables,
        selected_leads=selected_leads,
        seeds=seeds,
        eps=float(args.eps),
        split_tolerance=float(args.split_tolerance),
    )

    paths = {
        "curve_rows": output_dir / "tau0_phi_eid_curve_rows.csv",
        "curve_summary": output_dir / "tau0_phi_eid_curve_summary.csv",
        "singleton_rows": output_dir / "tau0_singleton_ei_rows.csv",
        "singleton_summary": output_dir / "tau0_singleton_ei_summary.csv",
        "atoms": output_dir / "tau0_phi_eid_greedy_atoms.csv",
        "totals": output_dir / "tau0_phi_eid_greedy_totals.csv",
        "total_summary": output_dir / "tau0_phi_eid_greedy_total_summary.csv",
        "order_summary": output_dir / "tau0_phi_eid_greedy_order_summary.csv",
        "module_summary": output_dir / "tau0_phi_eid_greedy_module_summary.csv",
        "module_lead": output_dir / "tau0_phi_eid_greedy_module_lead_summary.csv",
    }
    rows.to_csv(paths["curve_rows"], index=False)
    curve_summary.to_csv(paths["curve_summary"], index=False)
    singleton_rows.to_csv(paths["singleton_rows"], index=False)
    singleton_summary.to_csv(paths["singleton_summary"], index=False)
    atoms.to_csv(paths["atoms"], index=False)
    totals.to_csv(paths["totals"], index=False)
    total_summary.to_csv(paths["total_summary"], index=False)
    lead_order.to_csv(paths["order_summary"], index=False)
    module_summary.to_csv(paths["module_summary"], index=False)
    module_lead.to_csv(paths["module_lead"], index=False)
    asset_base = Path(args.asset_base)
    figures = [
        *plot_tau0_curves(curve_summary, singleton_summary, selected_leads, asset_base),
        *plot_tau0_decomposition(lead_order, module_summary, module_lead, asset_base, top_k=int(args.top_atoms)),
    ]
    manifest = {
        "target_definition": "all 11 predicted UniCM modes at each lead as a multivariate target",
        "source_definition": f"only mode variables at history lag tau={int(args.tau)}; other history variables enter the cached UniCM forward pass as uniform background",
        "phi_eid_definition": "I(all modes at tau; all-mode target) - sum_i I(mode_i at tau; all-mode target)",
        "decomposition": "top-down greedy bipartition using hierarchical additivity residual EI(C)-EI(L)-EI(R)",
        "selected_leads": selected_leads,
        "seeds": seeds,
        "leads": leads,
        "n_samples": int(args.n_samples),
        "sampling_seed": int(args.sampling_seed),
        "intervention_bound": float(args.intervention_bound),
        "tables": {key: str(path) for key, path in paths.items()},
        "figures": [str(path) for path in figures],
        "estimator": {
            "estimator": "transport_map",
            "backend": "affine_triangular_transport_map_degree_1_fast_logdet_equivalent",
            "clip_negative": False,
            "tm_degree": 1,
            "tm_jitter": float(args.jitter),
        },
    }
    manifest_path = output_dir / "tau0_phi_eid_curves_decomposition_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot tau=0 UniCM EI/Phi curves and greedy Phi decomposition.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--asset-base", type=Path, default=DEFAULT_ASSET_BASE)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--leads", nargs="*", default=None)
    parser.add_argument("--decomposition-leads", nargs="*", default=None)
    parser.add_argument("--top-leads", type=int, default=3)
    parser.add_argument("--top-atoms", type=int, default=12)
    parser.add_argument("--tau", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument("--sampling-seed", type=int, default=20260619)
    parser.add_argument("--intervention-bound", type=float, default=4.0)
    parser.add_argument("--start-month", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jitter", type=float, default=1.0e-6)
    parser.add_argument("--eps", type=float, default=1.0e-5)
    parser.add_argument("--split-tolerance", type=float, default=1.0e-4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    outputs = run(build_arg_parser().parse_args(argv))
    print(json.dumps(outputs, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
