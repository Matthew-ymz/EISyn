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
from scripts.phi_hierarchy import (  # noqa: E402
    PhiAtom as GreedyAtom,
    all_nonempty_subsets,
    greedy_phi_atoms,
    nontrivial_bipartitions,
    subset_phi_raw,
)
from scripts.unicm_peid_syn_analysis import (  # noqa: E402
    MODE_NAMES,
    load_full_history_prediction_cache,
    overall_prediction_cache_path,
    sample_full_history_mode_inputs,
)

DEFAULT_CACHE_DIR = ROOT / "results" / "unicm_overall_ei_cpu_bound4_n8192" / "cache"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "unicm_phi_eid_greedy_decomposition_cpu_bound4_n8192"
DEFAULT_ASSET_BASE = ROOT / "fig" / "unicm_phi_eid_greedy_decomposition"


def _safe_logdet(matrix: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(np.asarray(matrix, dtype=float))
    if sign <= 0 or not np.isfinite(logdet):
        raise ValueError("Regularized covariance matrix is not positive definite.")
    return float(logdet)


def _regularized_logdet(covariance: np.ndarray, *, jitter: float = 1.0e-6) -> float:
    cov = np.atleast_2d(np.asarray(covariance, dtype=float))
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be square.")
    scale = float(np.trace(cov) / cov.shape[0]) if cov.size else 1.0
    ridge = float(jitter) * max(scale, 1.0)
    return _safe_logdet(cov + ridge * np.eye(cov.shape[0], dtype=float))


def _all_mode_subsets(mode_names: Sequence[str]) -> list[tuple[str, ...]]:
    return all_nonempty_subsets(mode_names)


def _source_columns_for_subset(
    subset: Sequence[str],
    *,
    mode_names: Mapping[str, int],
    history_length: int,
) -> list[int]:
    columns: list[int] = []
    for month in range(int(history_length)):
        for mode in subset:
            columns.append(month * len(mode_names) + int(mode_names[str(mode)]))
    return columns


def precompute_source_logdets(
    history_modes: np.ndarray,
    *,
    mode_names: Mapping[str, int] = MODE_NAMES,
    jitter: float = 1.0e-6,
) -> tuple[np.ndarray, dict[tuple[str, ...], list[int]], dict[tuple[str, ...], float]]:
    history = np.asarray(history_modes, dtype=float)
    if history.ndim != 3:
        raise ValueError("history_modes must have shape (n_samples, history_length, n_modes).")
    flat = history.reshape(history.shape[0], history.shape[1] * history.shape[2])
    source_cov = np.cov(flat, rowvar=False, bias=False)
    subsets = _all_mode_subsets(tuple(mode_names))
    columns = {
        subset: _source_columns_for_subset(subset, mode_names=mode_names, history_length=history.shape[1])
        for subset in subsets
    }
    logdets = {subset: _regularized_logdet(source_cov[np.ix_(cols, cols)], jitter=jitter) for subset, cols in columns.items()}
    return flat, columns, logdets


def compute_subset_ei_table_from_covariance(
    history_flat: np.ndarray,
    target: np.ndarray,
    subset_columns: Mapping[tuple[str, ...], list[int]],
    source_logdets: Mapping[tuple[str, ...], float],
    *,
    jitter: float = 1.0e-6,
) -> dict[tuple[str, ...], float]:
    source = np.asarray(history_flat, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if target_array.ndim == 1:
        target_array = target_array.reshape(-1, 1)
    if target_array.ndim != 2 or source.shape[0] != target_array.shape[0]:
        raise ValueError("target must be 2D and share the sample axis with history_flat.")

    target_cov = np.cov(target_array, rowvar=False, bias=False)
    target_logdet = _regularized_logdet(target_cov, jitter=jitter)
    joint = np.concatenate([source, target_array], axis=1)
    joint_cov = np.cov(joint, rowvar=False, bias=False)
    target_cols = list(range(source.shape[1], source.shape[1] + target_array.shape[1]))
    table: dict[tuple[str, ...], float] = {}
    for subset, cols in subset_columns.items():
        selected = [*cols, *target_cols]
        joint_logdet = _regularized_logdet(joint_cov[np.ix_(selected, selected)], jitter=jitter)
        value = 0.5 * (float(source_logdets[subset]) + target_logdet - joint_logdet) / np.log(2.0)
        table[subset] = float(value)
    return table


def summarize_atom_rows(rows: pd.DataFrame, totals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = rows.groupby(["seed", "lead", "sources", "order"], as_index=False)["value"].sum()
    context = totals[["seed", "lead"]].drop_duplicates()
    modules = grouped[["sources", "order"]].drop_duplicates()
    full_index = context.merge(modules, how="cross")
    complete = full_index.merge(grouped, on=["seed", "lead", "sources", "order"], how="left")
    complete["value"] = complete["value"].fillna(0.0)

    order_seed = complete.groupby(["seed", "lead", "order"], as_index=False)["value"].sum()
    lead_order = order_seed.groupby(["lead", "order"], as_index=False)["value"].agg(["mean", "std"]).reset_index()
    lead_order["std"] = lead_order["std"].fillna(0.0)

    module = (
        complete.groupby(["sources", "order"], as_index=False)
        .agg(
            mean_value=("value", "mean"),
            max_value=("value", "max"),
            context_count=("value", "count"),
            nonzero_count=("value", lambda values: int((np.asarray(values, dtype=float) > 0.0).sum())),
        )
        .sort_values(["mean_value", "sources"], ascending=[False, True])
        .reset_index(drop=True)
    )
    module["rank"] = np.arange(1, len(module) + 1)

    module_lead = complete.groupby(["sources", "order", "lead"], as_index=False)["value"].agg(["mean", "std"]).reset_index()
    module_lead["std"] = module_lead["std"].fillna(0.0)
    return lead_order, module, module_lead


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


def plot_decomposition(
    total_summary: pd.DataFrame,
    lead_order: pd.DataFrame,
    module: pd.DataFrame,
    module_lead: pd.DataFrame,
    output_base: Path,
    *,
    mode_names: Sequence[str] = tuple(MODE_NAMES),
    top_k: int = 10,
) -> list[Path]:
    configure_matplotlib()
    fig = plt.figure(figsize=(9.6, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.65])
    ax_order = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])

    order_pivot = lead_order.pivot(index="lead", columns="order", values="mean").fillna(0.0).sort_index()
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, max(1, len(order_pivot.columns))))
    ax_order.stackplot(order_pivot.index.to_numpy(dtype=float), [order_pivot[col].to_numpy(dtype=float) for col in order_pivot.columns], labels=[f"order {col}" for col in order_pivot.columns], colors=colors, alpha=0.86)
    total = total_summary.sort_values("lead")
    ax_order.plot(total["lead"], total["phi_atom_sum_mean"], color="#111111", linewidth=1.3, label="atom sum")
    ax_order.set_xlabel("Lead (months)")
    ax_order.set_ylabel(r"Greedy $\mathrm{Syn}^{\mathrm{EID}}$ atoms (bits)")
    ax_order.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    top_modules = module.head(int(top_k))["sources"].astype(str).tolist()
    heat = (
        module_lead[module_lead["sources"].astype(str).isin(top_modules)]
        .pivot(index="sources", columns="lead", values="mean")
        .reindex(top_modules)
        .fillna(0.0)
    )
    image = ax_heat.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="magma", interpolation="nearest")
    ax_heat.set_yticks(np.arange(len(heat.index)))
    ax_heat.set_yticklabels([label.replace("|", " + ") for label in heat.index])
    ax_heat.set_xticks(np.arange(len(heat.columns))[::3])
    ax_heat.set_xticklabels([str(int(value)) for value in heat.columns[::3]])
    ax_heat.set_xlabel("Lead (months)")
    ax_heat.set_ylabel("Top modules")
    fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.02, label="bits")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".svg")]
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
    mode_names = tuple(MODE_NAMES)
    full_subset = tuple(mode_names)
    estimator_metadata: dict[str, object] = {
        "estimator": str(args.estimator),
        "backend": "gaussian_logdet",
        "clip_negative": False,
    }
    if str(args.estimator) == "transport_map":
        if int(args.tm_degree) != 1:
            raise ValueError(
                "Greedy all-subset Xi decomposition only supports --tm-degree 1. "
                "Higher-degree polynomial TM over all 2047 source subsets is computationally prohibitive; "
                "run selected subset probes instead."
            )
        estimator_metadata = {
            "estimator": "transport_map",
            "backend": "affine_triangular_transport_map_degree_1_fast_logdet_equivalent",
            "tm_degree": 1,
            "tm_jitter": float(args.jitter),
            "clip_negative": False,
            "note": "Uses the covariance/log-det closed form equivalent to an affine triangular TM.",
        }
    history_modes = sample_full_history_mode_inputs(
        n_samples=int(args.n_samples),
        intervention_bound=float(args.intervention_bound),
        seed=int(args.sampling_seed),
    )
    history_flat, subset_columns, source_logdets = precompute_source_logdets(history_modes)
    cache_args = _cache_args(args)

    atom_rows: list[dict[str, object]] = []
    total_rows: list[dict[str, object]] = []
    for seed in seeds:
        predictions = load_full_history_prediction_cache(
            overall_prediction_cache_path(Path(args.cache_dir), seed=seed, args=cache_args),
            n_samples=int(args.n_samples),
        )
        for lead in leads:
            target = extract_all_mode_target(predictions, lead=int(lead))
            ei_table = compute_subset_ei_table_from_covariance(
                history_flat,
                target,
                subset_columns,
                source_logdets,
                jitter=float(args.jitter),
            )
            singleton_ei = {name: ei_table[(name,)] for name in mode_names}
            raw_phi = subset_phi_raw(full_subset, ei_table, singleton_ei)
            atoms = greedy_phi_atoms(
                full_subset,
                ei_table,
                eps=float(args.eps),
                split_tolerance=float(args.split_tolerance),
                singleton_ei=singleton_ei,
            )
            atom_sum = float(sum(atom.value for atom in atoms))
            total_rows.append(
                {
                    "seed": int(seed),
                    "lead": int(lead),
                    "whole_ei": float(ei_table[full_subset]),
                    "singleton_ei_sum": float(sum(singleton_ei.values())),
                    "raw_phi_eid": float(raw_phi),
                    "phi_eid": float(raw_phi),
                    "phi_atom_sum": atom_sum,
                    "residual_to_phi": float(raw_phi - atom_sum),
                    "n_atoms": int(len(atoms)),
                }
            )
            for atom in atoms:
                atom_rows.append(
                    {
                        "seed": int(seed),
                        "lead": int(lead),
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
        phi_eid_std=("phi_eid", "std"),
        phi_atom_sum_mean=("phi_atom_sum", "mean"),
        phi_atom_sum_std=("phi_atom_sum", "std"),
        n_atoms_mean=("n_atoms", "mean"),
    )
    total_summary[["phi_eid_std", "phi_atom_sum_std"]] = total_summary[["phi_eid_std", "phi_atom_sum_std"]].fillna(0.0)
    lead_order, module_summary, module_lead = summarize_atom_rows(atoms, totals)

    paths = {
        "atoms": output_dir / "unicm_phi_eid_greedy_atoms.csv",
        "totals": output_dir / "unicm_phi_eid_greedy_totals.csv",
        "total_summary": output_dir / "unicm_phi_eid_greedy_total_summary.csv",
        "order_summary": output_dir / "unicm_phi_eid_greedy_order_summary.csv",
        "module_summary": output_dir / "unicm_phi_eid_greedy_module_summary.csv",
        "module_lead": output_dir / "unicm_phi_eid_greedy_module_lead_summary.csv",
    }
    atoms.to_csv(paths["atoms"], index=False)
    totals.to_csv(paths["totals"], index=False)
    total_summary.to_csv(paths["total_summary"], index=False)
    lead_order.to_csv(paths["order_summary"], index=False)
    module_summary.to_csv(paths["module_summary"], index=False)
    module_lead.to_csv(paths["module_lead"], index=False)
    figures = plot_decomposition(total_summary, lead_order, module_summary, module_lead, Path(args.asset_base))

    manifest = {
        "target_definition": "all 11 predicted UniCM modes at each lead as a multivariate target",
        "source_definition": "singleton partition over 11 mode histories; each source is one mode's 12-month history",
        "decomposition": "top-down greedy bipartition using hierarchical additivity residual EI(C)-EI(L)-EI(R)",
        "signed_outputs": True,
        "seeds": seeds,
        "leads": leads,
        "n_samples": int(args.n_samples),
        "sampling_seed": int(args.sampling_seed),
        "intervention_bound": float(args.intervention_bound),
        "eps": float(args.eps),
        "split_tolerance": float(args.split_tolerance),
        "tables": {key: str(path) for key, path in paths.items()},
        "figures": [str(path) for path in figures],
        "estimator": estimator_metadata,
    }
    manifest_path = output_dir / "unicm_phi_eid_greedy_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Greedy hierarchical Xi decomposition for UniCM all-mode targets.")
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
    parser.add_argument("--jitter", type=float, default=1.0e-6)
    parser.add_argument("--estimator", choices=["gaussian_logdet", "transport_map"], default="gaussian_logdet")
    parser.add_argument("--tm-degree", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1.0e-5)
    parser.add_argument("--split-tolerance", type=float, default=1.0e-4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    outputs = run(build_arg_parser().parse_args(argv))
    print(json.dumps(outputs, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
