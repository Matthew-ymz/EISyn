#!/usr/bin/env python3
"""Relate five-network MOTOR system Xi to the frozen broad motor score."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_all_task_behavior_coalitions_57 import (  # noqa: E402
    load_table,
    make_endpoint_contracts,
)
from scripts.analyze_hcp_system_xi_task_behavior_57 import (  # noqa: E402
    leave_one_out,
    partial_pearson,
    partial_spearman,
    rank_design,
    raw_design,
    residualize,
    unit_vector,
)
from scripts.run_hcp_schaefer500_all_tasks_phi import (  # noqa: E402
    development_end_for_length,
)
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import (  # noqa: E402
    load_yeo7_groups,
)
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import (  # noqa: E402
    fit_delta_history_phi,
)
from scripts.tune_hcp_task_evoked_xi_hierarchy import prepare_projection  # noqa: E402


TASK_ROOT = ROOT / "data/hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_57_brain"
LABELS = ROOT / "data/hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30/_atlas_labels/Schaefer2018_1000Parcels_7Networks_order.txt"
FULL_ARRAYS = ROOT / "results/hcp_schaefer1000_task_evoked_xi_57/full/k1_p3_a1/arrays.npz"
OUTPUT = ROOT / "results/hcp_motor_five_network_xi_57"
CACHE = OUTPUT / "motor_five_network_xi_57.npz"

NETWORK_ORDER = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
SELECTED_NETWORKS = ("Vis", "SomMot", "Limbic", "Cont", "Default")
SELECTED_INDICES = np.asarray([NETWORK_ORDER.index(name) for name in SELECTED_NETWORKS])
ORDER = 3
ALPHA = 1.0
PERMUTATIONS = 100_000
BOOTSTRAPS = 20_000
SEED = 20260804
XI_TOLERANCE_BITS = 1.0e-10


def load_frozen_inputs() -> tuple[np.ndarray, np.ndarray]:
    with np.load(FULL_ARRAYS, allow_pickle=False) as archive:
        subjects = archive["subjects"].astype(str)
        states = archive["states"].astype(str).tolist()
        full_xi = archive["system_xi"].astype(float)[states.index("MOTOR")]
    if subjects.shape != (57,) or full_xi.shape != (57,):
        raise ValueError("Expected the frozen 57-subject MOTOR system-Xi inputs.")
    return subjects, full_xi


def compute_five_network_xi(
    subjects: np.ndarray, *, recompute: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if CACHE.is_file() and not recompute:
        with np.load(CACHE, allow_pickle=False) as archive:
            if np.array_equal(archive["subjects"].astype(str), subjects):
                xi = archive["system_xi_bits"].astype(float)
                heldout = archive["heldout_skill_ratio"].astype(float)
                explained = archive["mean_pc1_explained"].astype(float)
                if xi.shape == heldout.shape == explained.shape == (57,):
                    return xi, heldout, explained

    groups = load_yeo7_groups(LABELS, expected_parcels=1000)
    xi = np.full(57, np.nan)
    heldout = np.full(57, np.nan)
    explained = np.full(57, np.nan)
    for index, subject in enumerate(subjects):
        projections, variance, development_end = prepare_projection(
            TASK_ROOT / str(subject) / "MOTOR_LR.mat",
            groups,
            state="MOTOR",
            max_components=1,
            task_retained_key="Schaefer1000_taskRetained",
            task_regressed_key="Schaefer1000_taskRegressed",
            expected_parcels=1000,
        )
        reduced = np.asarray(projections[1][:, SELECTED_INDICES], dtype=float)
        expected_end = development_end_for_length(len(reduced))
        if development_end != expected_end:
            raise ValueError("Mismatched development split.")
        fitted = fit_delta_history_phi(
            reduced,
            alpha=ALPHA,
            order=ORDER,
            development_end=development_end,
        )
        xi[index] = float(fitted["phi"]["raw_phi"])
        heldout[index] = float(fitted["heldout"]["skill_ratio"])
        explained[index] = float(
            np.mean([variance[1][network] for network in SELECTED_NETWORKS])
        )
        print(f"[{index + 1:02d}/57] {subject} Xi={xi[index]:.6f}", flush=True)

    violations = xi < -XI_TOLERANCE_BITS
    if np.any(violations):
        raise ValueError(
            "Five-network system Xi nonnegativity violation: "
            f"minimum={xi.min():.12g} bits, threshold={-XI_TOLERANCE_BITS:.1e}, "
            f"count={int(violations.sum())}"
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        subjects=subjects,
        networks=np.asarray(SELECTED_NETWORKS),
        system_xi_bits=xi,
        heldout_skill_ratio=heldout,
        mean_pc1_explained=explained,
        order=np.asarray(ORDER),
        alpha=np.asarray(ALPHA),
        xi_tolerance_bits=np.asarray(XI_TOLERANCE_BITS),
    )
    return xi, heldout, explained


def freedman_lane_test(
    brain: np.ndarray,
    endpoint: np.ndarray,
    contract: Mapping[str, Any],
    *,
    permutations: int,
) -> dict[str, float]:
    design = rank_design(contract)
    brain_unit = unit_vector(residualize(rankdata(brain), design))
    endpoint_rank = rankdata(endpoint)
    endpoint_fitted = design @ np.linalg.lstsq(design, endpoint_rank, rcond=None)[0]
    endpoint_residual = residualize(endpoint_rank, design)
    observed = float(brain_unit @ unit_vector(endpoint_residual))
    two_sided_count = 0
    negative_count = 0
    rng = np.random.default_rng(SEED)
    n = len(brain)
    chunk = 1000
    for start in range(0, permutations, chunk):
        size = min(chunk, permutations - start)
        indices = np.argsort(rng.random((size, n)), axis=1)
        pseudo = endpoint_fitted[None, :] + endpoint_residual[indices]
        coefficients = np.linalg.lstsq(design, pseudo.T, rcond=None)[0]
        null_residual = pseudo - (design @ coefficients).T
        null_residual /= np.linalg.norm(null_residual, axis=1, keepdims=True)
        null_rho = null_residual @ brain_unit
        two_sided_count += int(np.sum(np.abs(null_rho) >= abs(observed)))
        negative_count += int(np.sum(null_rho <= observed))
    denominator = permutations + 1.0
    return {
        "rho": observed,
        "p_two_sided": float((two_sided_count + 1.0) / denominator),
        "p_negative_one_sided": float((negative_count + 1.0) / denominator),
    }


def bootstrap_comparison(
    five_xi: np.ndarray,
    full_xi: np.ndarray,
    endpoint: np.ndarray,
    contract: Mapping[str, Any],
    *,
    repeats: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(SEED + 100)
    five = np.full(repeats, np.nan)
    full = np.full(repeats, np.nan)
    difference = np.full(repeats, np.nan)
    n = len(endpoint)
    for index in range(repeats):
        sample = rng.integers(0, n, size=n)
        design = rank_design(contract, sample)
        five[index] = partial_spearman(
            five_xi[sample], endpoint[sample], design
        )
        full[index] = partial_spearman(
            full_xi[sample], endpoint[sample], design
        )
        difference[index] = five[index] - full[index]
    return {
        "five_network_rho_quantiles": np.nanquantile(five, [0.025, 0.5, 0.975]).tolist(),
        "seven_network_rho_quantiles": np.nanquantile(full, [0.025, 0.5, 0.975]).tolist(),
        "rho_difference_five_minus_seven_quantiles": np.nanquantile(
            difference, [0.025, 0.5, 0.975]
        ).tolist(),
    }


def write_outputs(
    subjects: np.ndarray,
    five_xi: np.ndarray,
    full_xi: np.ndarray,
    heldout: np.ndarray,
    explained: np.ndarray,
    contract: Mapping[str, Any],
    *,
    permutations: int,
    bootstraps: int,
) -> dict[str, Any]:
    endpoint = np.asarray(contract["endpoint"], dtype=float)
    test = freedman_lane_test(
        five_xi, endpoint, contract, permutations=permutations
    )
    comparison = bootstrap_comparison(
        five_xi,
        full_xi,
        endpoint,
        contract,
        repeats=bootstraps,
    )
    design = rank_design(contract)
    five_ci = comparison["five_network_rho_quantiles"]
    seven_rho = partial_spearman(full_xi, endpoint, design)
    summary = {
        "experiment": "Five-network MOTOR system Xi versus broad motor score",
        "scientific_question": "Is MOTOR system-level Xi within Visual, Somatomotor, Limbic, Control, and Default networks negatively associated with broad motor performance?",
        "status": "post-hoc single-combination analysis motivated by the observed coalition-Syn screen",
        "subjects": subjects.tolist(),
        "n_subjects": int(len(subjects)),
        "network_system": list(SELECTED_NETWORKS),
        "estimator": {
            "representation": "Schaefer-1000/Yeo7 network PC1; task-evoked PCA fitted on taskRetained-taskRegressed and projected onto taskRetained",
            "dynamics": "order-3 delta Ridge alpha=1",
            "effective_information": "affine Gaussian TM/log-det",
            "source_dimension": int(len(SELECTED_NETWORKS) * ORDER),
            "target_dimension": int(len(SELECTED_NETWORKS)),
            "xi_definition": "joint EI of all lagged source variables to the five-network future minus the sum of individual lag-source EIs",
        },
        "behavior": {
            "endpoint": contract["definition"],
            "covariates": "age rank and sex",
        },
        "five_network_result": {
            "partial_spearman_rho": test["rho"],
            "bootstrap_95_ci": [float(five_ci[0]), float(five_ci[2])],
            "bootstrap_median": float(five_ci[1]),
            "permutation_p_two_sided": test["p_two_sided"],
            "permutation_p_negative_one_sided": test["p_negative_one_sided"],
            "partial_pearson_sensitivity": partial_pearson(
                five_xi, endpoint, raw_design(contract)
            ),
            "leave_one_out": leave_one_out(five_xi, endpoint, contract),
        },
        "paired_seven_network_reference": {
            "partial_spearman_rho": float(seven_rho),
            "bootstrap_95_ci": [
                float(comparison["seven_network_rho_quantiles"][0]),
                float(comparison["seven_network_rho_quantiles"][2]),
            ],
            "rho_difference_five_minus_seven_bootstrap_95_ci": [
                float(comparison["rho_difference_five_minus_seven_quantiles"][0]),
                float(comparison["rho_difference_five_minus_seven_quantiles"][2]),
            ],
            "five_seven_xi_spearman": float(spearmanr(five_xi, full_xi).statistic),
            "note": "Absolute Xi values are not directly compared because source and target dimensionality change with system membership.",
        },
        "quality": {
            "five_network_xi_mean_bits": float(five_xi.mean()),
            "five_network_xi_sd_bits": float(five_xi.std(ddof=1)),
            "five_network_xi_minimum_bits": float(five_xi.min()),
            "five_network_xi_maximum_bits": float(five_xi.max()),
            "mean_heldout_skill_ratio": float(heldout.mean()),
            "models_better_than_persistence": int(np.sum(heldout < 1.0)),
            "mean_pc1_explained_variance": float(explained.mean()),
            "xi_nonnegativity_tolerance_bits": XI_TOLERANCE_BITS,
            "xi_nonnegativity_violation_count": int(
                np.sum(five_xi < -XI_TOLERANCE_BITS)
            ),
        },
        "inference_boundary": "The pointwise p-values do not correct the post-hoc choice of these five networks. A confirmatory claim requires a frozen test in an independent sample.",
        "permutations": int(permutations),
        "bootstraps": int(bootstraps),
    }
    experiment_contract = {
        "question": "What changes when only the modeled MOTOR network system changes from all Yeo7 networks to the specified five-network subsystem?",
        "treatment_factor": "system membership and corresponding source/target dimensionality",
        "treatment_levels": ["all seven Yeo networks", "+".join(SELECTED_NETWORKS)],
        "paired_unit": "same 57 subjects",
        "frozen_controls": [
            "MOTOR_LR input and taskRetained/taskRegressed preprocessing",
            "network-wise PC1 representation",
            "75% development split",
            "order-3 delta Ridge alpha=1",
            "affine Gaussian TM/log-det Xi estimator",
            "broad motor endpoint",
            "age-rank and sex covariates",
        ],
        "primary_metric": "partial Spearman rho between five-network system Xi and broad motor score",
        "primary_test": "two-sided Freedman-Lane permutation",
        "directional_supplement": "negative one-sided Freedman-Lane permutation",
        "multiplicity": "none; one post-hoc subsystem definition",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "experiment_contract.json").write_text(
        json.dumps(experiment_contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    low, high = summary["five_network_result"]["bootstrap_95_ci"]
    lines = [
        "# MOTOR 五网络子系统 Xi 与运动表现",
        "",
        "五网络子系统由 Visual、Somatomotor、Limbic、Control 和 Default 组成。这里重新拟合五维 MOTOR 动力学并计算该子系统自身的 system-level $\\Xi$；它不是五网络 coalition Syn。",
        "",
        f"- 偏 Spearman $\\rho={test['rho']:+.3f}$，bootstrap 95% CI $[{low:+.3f}, {high:+.3f}]$。",
        f"- 双侧 Freedman--Lane 置换 $p={test['p_two_sided']:.4f}$；事后负向单侧 $p={test['p_negative_one_sided']:.4f}$。",
        f"- 七网络 system-level $\\Xi$ 的同口径相关为 $\\rho={seven_rho:+.3f}$。",
        "",
        "该五网络定义来自已观察到的 coalition Syn 榜单，因此当前单项 $p$ 没有校正这一事后选择，结果只能作为探索性分析。",
        "",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects, full_xi = load_frozen_inputs()
    five_xi, heldout, explained = compute_five_network_xi(
        subjects, recompute=args.recompute
    )
    contracts = make_endpoint_contracts(subjects, load_table())
    summary = write_outputs(
        subjects,
        five_xi,
        full_xi,
        heldout,
        explained,
        contracts["MOTOR"],
        permutations=args.permutations,
        bootstraps=args.bootstraps,
    )
    print(json.dumps(summary["five_network_result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
