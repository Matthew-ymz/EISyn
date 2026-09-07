#!/usr/bin/env python3
"""Compute SPT-selected order mass for every UniCM lead and checkpoint."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_unicm_order_syn import (  # noqa: E402
    ESTIMATOR_VERSION,
    OUTPUT as ALL_COALITION_CACHE,
    independent_source_ei_table,
    load_order_means,
)
from scripts.spt import (  # noqa: E402
    NONNEGATIVE_TOLERANT,
    RAW_RESIDUAL,
    SPTConfig,
    build_spt_from_ei_table,
    flatten_nodes,
)
from scripts.unicm_peid_syn_analysis import MODE_NAMES, sample_full_history_mode_inputs  # noqa: E402

OUTPUT_DIR = ROOT / "results" / "unicm_spt_order_mass"
OUTPUT = OUTPUT_DIR / "order_mass.npz"
SUMMARY = OUTPUT_DIR / "summary.json"
SHARE_FIGURE = ROOT / "fig" / "earth_unicm_spt_order_mass_share.png"
ALL_COALITION_FIGURE = ROOT / "fig" / "earth_unicm_all_coalition_order_mean.png"
CHECKPOINTS = (1, 2, 3)
LEADS = tuple(range(1, 25))
ORDERS = tuple(range(2, 12))
N_SAMPLES = 16384
SAMPLING_SEED = 20260901
INTERVENTION_BOUND = 4.0
JITTER = 1.0e-6
SYN_TOLERANCE = 1.0e-8
EPS = 1.0e-8


def _prediction_cache(checkpoint: int) -> Path:
    return ROOT / "results" / "unicm_xi_hierarchy_uniform_n16384" / "cache" / (
        f"checkpoint{checkpoint}_samples16384_sampling20260901_"
        "bound4_fullhist12_start0_cpu.npz"
    )


def _topology_signature(root) -> str:
    def topology(node):
        return {
            "sources": list(node.sources),
            "children": [topology(child) for child in node.children],
        }

    encoded = json.dumps(topology(root), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _aggregate_tree(root) -> tuple[np.ndarray, np.ndarray, int]:
    mass = np.zeros(len(ORDERS), dtype=float)
    counts = np.zeros(len(ORDERS), dtype=int)
    tolerance_zero_count = 0
    internal_nodes = [node for node in flatten_nodes(root) if node.children]
    for node in internal_nodes:
        value = float(node.syn_value)
        if value < -SYN_TOLERANCE:
            raise ValueError(
                f"SPT Syn violation: minimum={value:.12g}, "
                f"threshold={-SYN_TOLERANCE:.12g}, affected_count=1"
            )
        if value < 0.0:
            tolerance_zero_count += 1
            value = 0.0
        mass[int(node.order) - 2] += value
        counts[int(node.order) - 2] += 1
    closure_error = float(mass.sum() - float(root.xi_value))
    if abs(closure_error) > max(SYN_TOLERANCE, 1.0e-10):
        raise ValueError(
            f"Order-mass closure failed: error={closure_error:.12g}, "
            f"threshold={max(SYN_TOLERANCE, 1.0e-10):.12g}"
        )
    return mass, counts, tolerance_zero_count


def load_spt_order_mass(path: Path = OUTPUT) -> dict[str, np.ndarray | dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        mass = np.asarray(data["mass_bits"], dtype=float)
        share = np.asarray(data["share"], dtype=float)
        counts = np.asarray(data["node_counts"], dtype=int)
        xi = np.asarray(data["xi_bits"], dtype=float)
        if mass.shape != (3, 24, 10) or share.shape != mass.shape or counts.shape != mass.shape:
            raise ValueError(f"Incomplete SPT order-mass cache: {path}")
        if xi.shape != (3, 24) or not np.isfinite(mass).all() or not np.isfinite(share).all():
            raise ValueError(f"Invalid SPT order-mass cache: {path}")
        if not np.allclose(mass.sum(axis=2), xi, atol=SYN_TOLERANCE, rtol=0.0):
            raise ValueError("Cached SPT order masses do not close to Xi")
        if not np.allclose(share.sum(axis=2), 1.0, atol=1.0e-10, rtol=0.0):
            raise ValueError("Cached within-tree shares do not sum to one")
        return {
            "mass_bits": mass,
            "share": share,
            "node_counts": counts,
            "xi_bits": xi,
            "mean_mass_bits": mass.mean(axis=0).T,
            "mean_share": share.mean(axis=0).T,
            "metadata": metadata,
        }


def _plot_heatmap(values: np.ndarray, output: Path, *, label: str, percent: bool = False) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "savefig.facecolor": "white",
        }
    )
    display = 100.0 * values if percent else values
    fig, ax = plt.subplots(figsize=(6.8, 3.15), layout="constrained")
    image = ax.imshow(
        display,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        norm=mpl.colors.Normalize(vmin=0.0, vmax=float(np.max(display))),
    )
    ax.set_yticks(np.arange(len(ORDERS)), [str(order) for order in ORDERS])
    tick_indices = np.asarray((0, 3, 7, 11, 15, 19, 23))
    ax.set_xticks(tick_indices, [str(index + 1) for index in tick_indices])
    ax.set_xlabel(r"Prediction lead, $\ell$ (months)")
    ax.set_ylabel("SPT node order")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label(label)
    ax.text(
        1.0,
        1.025,
        "Tree-wise order sum, then 3-checkpoint mean",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color="#444444",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_all_coalition_appendix(output: Path) -> None:
    values = load_order_means(ALL_COALITION_CACHE)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "savefig.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(6.8, 3.15), layout="constrained")
    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        norm=mpl.colors.Normalize(vmin=0.0, vmax=float(values.max())),
    )
    ax.set_yticks(np.arange(len(ORDERS)), [str(order) for order in ORDERS])
    tick_indices = np.asarray((0, 3, 7, 11, 15, 19, 23))
    ax.set_xticks(tick_indices, [str(index + 1) for index in tick_indices])
    ax.set_xlabel(r"Prediction lead, $\ell$ (months)")
    ax.set_ylabel("Coalition order")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("All-coalition mean Syn (bits)")
    ax.text(
        1.0,
        1.025,
        "All same-order coalitions, then 3-checkpoint mean",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color="#444444",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    histories = sample_full_history_mode_inputs(
        n_samples=N_SAMPLES,
        intervention_bound=INTERVENTION_BOUND,
        seed=SAMPLING_SEED,
    )
    mass_rows = []
    count_rows = []
    xi_rows = []
    records = []
    total_tolerance_zeros = 0
    minimum_selected_syn = float("inf")
    for checkpoint in CHECKPOINTS:
        cache = _prediction_cache(checkpoint)
        with np.load(cache, allow_pickle=False) as data:
            predictions = np.asarray(data["all_mode_targets"], dtype=float)
        if predictions.shape != (N_SAMPLES, len(LEADS), len(MODE_NAMES)):
            raise ValueError(f"Invalid prediction cache: {cache}")
        checkpoint_mass = []
        checkpoint_counts = []
        checkpoint_xi = []
        for lead in LEADS:
            ei_table, estimator_audit = independent_source_ei_table(
                histories,
                predictions[:, lead - 1, :],
                intervention_bound=INTERVENTION_BOUND,
                jitter=JITTER,
            )
            singleton_ei = {name: float(ei_table[(name,)]) for name in MODE_NAMES}
            result = build_spt_from_ei_table(
                tuple(MODE_NAMES),
                ei_table,
                singleton_ei=singleton_ei,
                config=SPTConfig(
                    policy=NONNEGATIVE_TOLERANT,
                    split_objective=RAW_RESIDUAL,
                    syn_tolerance=SYN_TOLERANCE,
                    eps=EPS,
                    complete_to_singletons=True,
                ),
            )
            mass, counts, tolerance_zeros = _aggregate_tree(result.root)
            selected = [float(node.syn_value) for node in flatten_nodes(result.root) if node.children]
            minimum_selected_syn = min(minimum_selected_syn, min(selected))
            total_tolerance_zeros += tolerance_zeros
            checkpoint_mass.append(mass)
            checkpoint_counts.append(counts)
            checkpoint_xi.append(float(result.root.xi_value))
            records.append(
                {
                    "checkpoint": checkpoint,
                    "lead": lead,
                    "xi_bits": float(result.root.xi_value),
                    "order_mass_bits": {str(order): float(mass[order - 2]) for order in ORDERS},
                    "node_counts": {str(order): int(counts[order - 2]) for order in ORDERS},
                    "internal_nodes": [
                        {
                            "order": int(node.order),
                            "sources": list(node.sources),
                            "syn_bits": float(node.syn_value),
                        }
                        for node in flatten_nodes(result.root)
                        if node.children
                    ],
                    "topology_sha256": _topology_signature(result.root),
                    "closure_error_bits": float(result.closure_error),
                    "tolerance_zero_count": tolerance_zeros,
                    "estimator_audit": estimator_audit,
                }
            )
            print(
                f"checkpoint={checkpoint} lead={lead} Xi={result.root.xi_value:.6f} "
                f"closure={result.closure_error:.3g}",
                flush=True,
            )
        mass_rows.append(checkpoint_mass)
        count_rows.append(checkpoint_counts)
        xi_rows.append(checkpoint_xi)

    mass = np.asarray(mass_rows, dtype=float)
    counts = np.asarray(count_rows, dtype=int)
    xi = np.asarray(xi_rows, dtype=float)
    share = mass / xi[:, :, None]
    metadata = {
        "estimator": ESTIMATOR_VERSION,
        "aggregation": "sum selected internal-node Syn within each tree and order, then equal checkpoint mean",
        "share_aggregation": "normalize order mass by Xi within each tree, then equal checkpoint mean",
        "missing_order_rule": "zero mass",
        "checkpoints": list(CHECKPOINTS),
        "leads": list(LEADS),
        "orders": list(ORDERS),
        "n_samples": N_SAMPLES,
        "sampling_seed": SAMPLING_SEED,
        "intervention_bound": INTERVENTION_BOUND,
        "start_month": 0,
        "jitter": JITTER,
        "syn_tolerance_bits": SYN_TOLERANCE,
        "eps_bits": EPS,
        "tolerance_negative_rule": "values in [-tolerance, 0) are recorded and treated as numerical zero",
        "tolerance_zero_count": total_tolerance_zeros,
        "minimum_selected_syn_bits": minimum_selected_syn,
        "projection": "none beyond the declared tolerance-zero rule",
    }
    np.savez_compressed(
        OUTPUT,
        mass_bits=mass,
        share=share,
        node_counts=counts,
        xi_bits=xi,
        checkpoints=np.asarray(CHECKPOINTS),
        leads=np.asarray(LEADS),
        orders=np.asarray(ORDERS),
        metadata=json.dumps(metadata, sort_keys=True),
    )
    loaded = load_spt_order_mass(OUTPUT)
    summary = {
        "status": "complete",
        "method": metadata,
        "mean_mass_bits": {
            str(order): loaded["mean_mass_bits"][order - 2].tolist() for order in ORDERS
        },
        "mean_share_percent": {
            str(order): (100.0 * loaded["mean_share"][order - 2]).tolist() for order in ORDERS
        },
        "records": records,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot_heatmap(
        loaded["mean_share"],
        SHARE_FIGURE,
        label="Mean within-tree order mass (%)",
        percent=True,
    )
    _plot_all_coalition_appendix(ALL_COALITION_FIGURE)
    print(OUTPUT)
    print(SHARE_FIGURE)
    print(ALL_COALITION_FIGURE)


if __name__ == "__main__":
    main()
