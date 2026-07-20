#!/usr/bin/env python3
"""Screen k/p/alpha for task-evoked-PCA Xi state separation without LOSO."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_task_evoked_pc2_xi_hierarchy import (
    STATES,
    TASKS,
    all_atom_subsets,
    bh_adjust,
    decompose_transition,
    fit_project_network_pca,
    pairwise_distribution_tests,
)
from scripts.analyze_hcp_schaefer500_yeo7_network_attribution import (
    DEFAULT_REST_ROOT,
    DEFAULT_TASK_ROOT,
    NETWORK_ORDER,
    discover_inputs,
    load_rest_series,
)
from scripts.analyze_hcp_task_evoked_pc1_phi_attribution import load_task_pair
from scripts.run_hcp_schaefer500_all_tasks_phi import development_end_for_length
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import (
    default_yeo7_labels,
    load_yeo7_groups,
)
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi


DEFAULT_OUTPUT = ROOT / "results" / "hcp_schaefer500_task_evoked_xi_tuning"
DEFAULT_LOG_DIR = ROOT / "docs" / "log" / "hcp_task_evoked_xi_tuning"
DEFAULT_KP = ((1, 5), (2, 5), (2, 3), (2, 2), (3, 3), (3, 2), (4, 2))
DEFAULT_ALPHAS = (1.0, 10.0, 100.0)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def config_id(k: int, p: int, alpha: float) -> str:
    return f"k{int(k)}_p{int(p)}_a{float(alpha):g}"


def parse_config(value: str) -> tuple[int, int, float]:
    parts = value.replace(",", ":").split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("config must be k:p:alpha")
    k, p, alpha = int(parts[0]), int(parts[1]), float(parts[2])
    if k < 1 or p < 1 or alpha <= 0:
        raise argparse.ArgumentTypeError("k and p must be positive integers; alpha must be positive")
    return k, p, alpha


def default_configs() -> list[tuple[int, int, float]]:
    return [(k, p, alpha) for k, p in DEFAULT_KP for alpha in DEFAULT_ALPHAS]


def update_live_status(
    log_dir: Path, *, status: str, running: str, recent: str, next_step: str, log_name: str
) -> None:
    text = "\n".join(
        [
            "# 实时状态",
            "",
            "## 当前状态",
            f"- {status}",
            "",
            "## 正在运行",
            f"- {running}",
            "",
            "## 最近结果",
            f"- {recent}",
            "",
            "## 下一步",
            f"- {next_step}",
            "",
            "## 监控文件",
            f"- stdout/stderr: `logs/{log_name}`",
            "- run history: `run_history.jsonl`",
            "- leaderboard: `leaderboard.md`",
            "",
        ]
    )
    (log_dir / "live_status.md").write_text(text, encoding="utf-8")


def load_cache(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[(str(row["config_id"]), str(row["subject"]), str(row["state"]))] = row
    return rows


def prepare_projection(
    path: Path,
    groups: Mapping[str, Sequence[int]],
    *,
    state: str,
    max_components: int,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, float]], int]:
    """Fit PCA once at max k and reuse its ordered leading components for every smaller k."""
    if state == "REST":
        projection_signal = load_rest_series(path)
        fitting_signal = projection_signal
    else:
        retained, regressed = load_task_pair(path)
        projection_signal = retained
        fitting_signal = retained - regressed
    development_end = development_end_for_length(len(projection_signal))
    reduced_max, explained = fit_project_network_pca(
        fitting_signal,
        projection_signal,
        groups,
        development_end=development_end,
        n_components=max_components,
    )
    cube = reduced_max.reshape(len(reduced_max), len(NETWORK_ORDER), max_components)
    projections = {
        k: np.asarray(cube[:, :, :k].reshape(len(reduced_max), len(NETWORK_ORDER) * k), dtype=float)
        for k in range(1, max_components + 1)
    }
    cumulative = {
        k: {
            network: float(sum(explained[network][:k]))
            for network in NETWORK_ORDER
        }
        for k in range(1, max_components + 1)
    }
    return projections, cumulative, development_end


def analyze_reduced(
    reduced: np.ndarray,
    pca_cumulative: Mapping[str, float],
    *,
    subject: str,
    state: str,
    development_end: int,
    k: int,
    p: int,
    alpha: float,
) -> dict[str, Any]:
    fitted = fit_delta_history_phi(
        reduced,
        alpha=alpha,
        order=p,
        development_end=development_end,
    )
    decomposition = decompose_transition(
        fitted["transition"],
        fitted["noise_covariance"],
        tuple(NETWORK_ORDER),
        n_components=k,
        order=p,
    )
    development = reduced[:development_end]
    scale = development.std(axis=0, ddof=1)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return {
        "config_id": config_id(k, p, alpha),
        "params": {"k": int(k), "p": int(p), "alpha": float(alpha)},
        "subject": subject,
        "state": state,
        "n_timepoints": int(len(reduced)),
        "development_end": int(development_end),
        "pca_cumulative_explained_variance": dict(pca_cumulative),
        "heldout_skill_ratio": float(fitted["heldout"]["skill_ratio"]),
        "quality_diagnostics": {
            "max_abs_development_pc_zscore": float(
                np.max(np.abs((development - development.mean(axis=0)) / scale))
            ),
            "noise_covariance_condition": float(np.linalg.cond(fitted["noise_covariance"])),
        },
        **decomposition,
    }


def distribution_metrics(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    centroids = array.mean(axis=1)
    pairwise = np.asarray(
        [
            0.5 * np.abs(centroids[left] - centroids[right]).sum()
            for left in range(len(centroids))
            for right in range(left + 1, len(centroids))
        ],
        dtype=float,
    )
    within = float(
        np.mean(
            [
                0.5 * np.abs(array[state] - centroids[state]).sum(axis=1).mean()
                for state in range(array.shape[0])
            ]
        )
    )
    return {
        "between_tv_mean": float(pairwise.mean()),
        "between_tv_min": float(pairwise.min()),
        "between_tv_max": float(pairwise.max()),
        "within_tv_mean": within,
        "between_within_ratio": float(pairwise.mean() / max(within, 1.0e-12)),
    }


def friedman_counts(values: np.ndarray) -> tuple[int, list[dict[str, float]]]:
    rows = []
    for feature in range(values.shape[2]):
        selected = np.asarray(values[:, :, feature], dtype=float)
        if np.allclose(selected, selected[0, 0]):
            statistic, p_value = 0.0, 1.0
        else:
            result = friedmanchisquare(*[selected[state] for state in range(selected.shape[0])])
            statistic = float(result.statistic) if np.isfinite(result.statistic) else 0.0
            p_value = float(result.pvalue) if np.isfinite(result.pvalue) else 1.0
        rows.append({"feature": feature, "statistic": statistic, "p": p_value})
    adjusted = bh_adjust([row["p"] for row in rows])
    for row, q_value in zip(rows, adjusted):
        row["q"] = float(q_value)
    return int(sum(row["q"] < 0.05 for row in rows)), rows


def rest_system_tests(system_xi: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for index, task in enumerate(TASKS, start=1):
        difference = system_xi[0] - system_xi[index]
        p_value = 1.0 if np.allclose(difference, 0.0) else float(wilcoxon(difference).pvalue)
        rows.append(
            {
                "task": task,
                "rest_minus_task_mean_bits": float(difference.mean()),
                "rest_greater_fraction": float(np.mean(difference > 0.0)),
                "p": p_value,
            }
        )
    adjusted = bh_adjust([row["p"] for row in rows])
    for row, q_value in zip(rows, adjusted):
        row["q"] = float(q_value)
    return rows


def summarize_config(
    records: Mapping[tuple[str, str, str], Mapping[str, Any]],
    subjects: Sequence[str],
    params: tuple[int, int, float],
    *,
    permutation_repeats: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    k, p, alpha = params
    identifier = config_id(k, p, alpha)
    n_states, n_subjects, n_networks = len(STATES), len(subjects), len(NETWORK_ORDER)
    atom_subsets = all_atom_subsets()
    atom_lookup = {sources: index for index, sources in enumerate(atom_subsets)}
    system_xi = np.empty((n_states, n_subjects), dtype=float)
    cross_xi = np.empty_like(system_xi)
    network_attribution = np.empty((n_states, n_subjects, n_networks), dtype=float)
    atom_value = np.zeros((n_states, n_subjects, len(atom_subsets)), dtype=float)
    skill = np.empty_like(system_xi)
    pca_variance = np.empty((n_states, n_subjects, n_networks), dtype=float)
    identity_errors = []
    conditions = []
    z_extrema = []
    for state_index, state in enumerate(STATES):
        for subject_index, subject in enumerate(subjects):
            row = records[(identifier, subject, state)]
            system_xi[state_index, subject_index] = float(row["system_xi"])
            cross_xi[state_index, subject_index] = float(row["cross_network_xi"])
            network_attribution[state_index, subject_index] = [
                float(row["network_attribution"][network]) for network in NETWORK_ORDER
            ]
            skill[state_index, subject_index] = float(row["heldout_skill_ratio"])
            pca_variance[state_index, subject_index] = [
                float(row["pca_cumulative_explained_variance"][network])
                for network in NETWORK_ORDER
            ]
            identity_errors.extend(abs(float(value)) for value in row["identity_errors"].values())
            conditions.append(float(row["quality_diagnostics"]["noise_covariance_condition"]))
            z_extrema.append(float(row["quality_diagnostics"]["max_abs_development_pc_zscore"]))
            for atom in row["atoms"]:
                atom_value[state_index, subject_index, atom_lookup[tuple(atom["sources"])]] = float(
                    atom["value"]
                )
    if np.any(system_xi <= 1.0e-10) or np.any(cross_xi <= 1.0e-10):
        raise ValueError(f"{identifier} has non-positive system/cross Xi")
    network_share = network_attribution / system_xi[:, :, None]
    atom_share = atom_value / cross_xi[:, :, None]
    network_sig_all, network_friedman_all = friedman_counts(network_share)
    network_sig_task, network_friedman_task = friedman_counts(network_share[1:])
    atom_sig_all, atom_friedman_all = friedman_counts(atom_share)
    atom_sig_task, atom_friedman_task = friedman_counts(atom_share[1:])
    network_pairs = pairwise_distribution_tests(
        network_share, repeats=permutation_repeats, seed=2026075101
    )
    atom_pairs = pairwise_distribution_tests(
        atom_share, repeats=permutation_repeats, seed=2026075102
    )
    rest_tests = rest_system_tests(system_xi)
    summary = {
        "config_id": identifier,
        "params": {"k": k, "p": p, "alpha": alpha, "source_dimension": 7 * k * p, "target_dimension": 7 * k},
        "n_subjects": n_subjects,
        "network_all": distribution_metrics(network_share),
        "network_tasks": distribution_metrics(network_share[1:]),
        "atom_all": distribution_metrics(atom_share),
        "atom_tasks": distribution_metrics(atom_share[1:]),
        "significance": {
            "network_features_all": network_sig_all,
            "network_features_tasks": network_sig_task,
            "atom_features_all": atom_sig_all,
            "atom_features_tasks": atom_sig_task,
            "network_pairs_all": int(sum(row["q"] < 0.05 for row in network_pairs)),
            "network_pairs_tasks": int(sum(row["q"] < 0.05 for row in network_pairs if row["left"] != "REST")),
            "atom_pairs_all": int(sum(row["q"] < 0.05 for row in atom_pairs)),
            "atom_pairs_tasks": int(sum(row["q"] < 0.05 for row in atom_pairs if row["left"] != "REST")),
        },
        "system_xi_mean_bits": {state: float(system_xi[index].mean()) for index, state in enumerate(STATES)},
        "rest_system_tests": rest_tests,
        "rest_min_mean_margin_bits": float(
            min(system_xi[0].mean() - system_xi[index].mean() for index in range(1, len(STATES)))
        ),
        "rest_significant_higher_tasks": int(
            sum(row["rest_minus_task_mean_bits"] > 0.0 and row["q"] < 0.05 for row in rest_tests)
        ),
        "heldout_skill_ratio_mean": float(skill.mean()),
        "heldout_skill_ratio_median": float(np.median(skill)),
        "models_better_than_persistence": int(np.sum(skill < 1.0)),
        "n_models": int(skill.size),
        "task_pca_variance_mean": float(pca_variance[1:].mean()),
        "rest_pca_variance_mean": float(pca_variance[0].mean()),
        "diagnostics": {
            "max_identity_error_bits": float(max(identity_errors)),
            "max_network_share_closure_error": float(np.max(np.abs(network_share.sum(axis=2) - 1.0))),
            "max_atom_share_closure_error": float(np.max(np.abs(atom_share.sum(axis=2) - 1.0))),
            "max_noise_condition": float(max(conditions)),
            "max_abs_pc_zscore": float(max(z_extrema)),
        },
        "statistics": {
            "network_friedman_all": network_friedman_all,
            "network_friedman_tasks": network_friedman_task,
            "atom_friedman_all": atom_friedman_all,
            "atom_friedman_tasks": atom_friedman_task,
            "network_pairwise": network_pairs,
            "atom_pairwise": atom_pairs,
        },
    }
    arrays = {
        "system_xi": system_xi,
        "cross_xi": cross_xi,
        "network_share": network_share,
        "atom_value": atom_value,
        "atom_share": atom_share,
        "heldout_skill_ratio": skill,
        "pca_variance": pca_variance,
    }
    return summary, arrays


def ranking_key(summary: Mapping[str, Any], objective: str) -> tuple[float, ...]:
    feasible = (
        float(summary["heldout_skill_ratio_mean"]) < 1.0
        and float(summary["diagnostics"]["max_identity_error_bits"]) < 1.0e-8
    )
    rest_ok = float(summary["rest_min_mean_margin_bits"]) > 0.0
    if objective == "network":
        metric = float(summary["network_all"]["between_within_ratio"])
    elif objective == "atom":
        metric = float(summary["atom_all"]["between_within_ratio"])
    else:
        metric = math.sqrt(
            max(float(summary["network_all"]["between_within_ratio"]), 0.0)
            * max(float(summary["atom_all"]["between_within_ratio"]), 0.0)
        )
    return (float(feasible), float(rest_ok), metric, -float(summary["heldout_skill_ratio_mean"]))


def write_leaderboard(summaries: Sequence[Mapping[str, Any]], log_dir: Path) -> None:
    ranked = sorted(summaries, key=lambda row: ranking_key(row, "balanced"), reverse=True)
    lines = [
        "# Leaderboard",
        "",
        "| Config | Network ratio | Atom ratio | Network/atom significant pairs | REST min margin | Skill ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['config_id']} | {row['network_all']['between_within_ratio']:.4f} | "
            f"{row['atom_all']['between_within_ratio']:.4f} | "
            f"{row['significance']['network_pairs_all']}/{row['significance']['atom_pairs_all']} | "
            f"{row['rest_min_mean_margin_bits']:+.4f} | {row['heldout_skill_ratio_mean']:.4f} |"
        )
    (log_dir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    output_dir = Path(args.output_dir) / args.stage
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    (log_dir / "logs").mkdir(parents=True, exist_ok=True)
    discovered = discover_inputs(args.rest_root, args.task_root)
    all_subjects = sorted(discovered)
    start = int(args.subject_offset)
    stop = start + int(args.max_subjects) if args.max_subjects else None
    subjects = all_subjects[start:stop]
    groups = load_yeo7_groups(args.labels, expected_parcels=500)
    configs = list(args.config) if args.config else default_configs()
    cache_path = output_dir / "records.jsonl"
    records = load_cache(cache_path)
    required_k = sorted({int(params[0]) for params in configs})
    # Full-SVD PCA computes the same leading PC regardless of requesting one or two
    # components.  Request at least two so k=1-only confirmation runs reuse the same
    # stable PCA path as the mixed-k screening batches.
    max_k = max(2, max(required_k))
    projection_cache: dict[tuple[str, str, int], tuple[np.ndarray, dict[str, float], int]] = {}
    projection_jobs = [(subject, state, Path(discovered[subject][state])) for subject in subjects for state in STATES]
    projection_progress = tqdm(projection_jobs, desc=f"PCA cache {args.stage}", unit="state", mininterval=1.0)
    for subject, state, path in projection_progress:
        projections, cumulative, development_end = prepare_projection(
            path, groups, state=state, max_components=max_k
        )
        for k in required_k:
            projection_cache[(subject, state, k)] = (
                projections[k], cumulative[k], development_end
            )
    jobs = [
        (params, subject, state)
        for params in configs
        for subject in subjects
        for state in STATES
        if (config_id(*params), subject, state) not in records
    ]
    update_live_status(
        log_dir,
        status="running",
        running=f"{args.stage}: {len(configs)} configs × {len(subjects)} subjects × 8 states",
        recent="已完成实验合同与理论口径核对。",
        next_step="当前批次完成后更新 leaderboard 并选择下一批候选。",
        log_name=f"{args.stage}.log",
    )
    progress = tqdm(jobs, desc=f"Xi tuning {args.stage}", unit="model", mininterval=1.0)
    for completed, (params, subject, state) in enumerate(progress, start=1):
        k, p, alpha = params
        reduced, cumulative, development_end = projection_cache[(subject, state, k)]
        row = analyze_reduced(
            reduced,
            cumulative,
            subject=subject,
            state=state,
            development_end=development_end,
            k=k,
            p=p,
            alpha=alpha,
        )
        records[(row["config_id"], subject, state)] = row
        append_jsonl(cache_path, row)
        elapsed = time.monotonic() - started
        rate = completed / elapsed if elapsed > 0 else 0.0
        atomic_json(
            output_dir / "live_progress.json",
            {
                "phase": "compute",
                "current": completed,
                "total": len(jobs),
                "elapsed_seconds": elapsed,
                "eta_seconds": (len(jobs) - completed) / rate if rate > 0 else None,
                "config_id": row["config_id"],
                "subject": subject,
                "state": state,
                "system_xi": row["system_xi"],
            },
        )
        progress.set_postfix(config=row["config_id"], state=state)

    summaries = []
    for params in configs:
        summary, arrays = summarize_config(
            records,
            subjects,
            params,
            permutation_repeats=int(args.permutation_repeats),
        )
        identifier = config_id(*params)
        config_dir = output_dir / identifier
        config_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(config_dir / "summary.json", summary)
        np.savez_compressed(
            config_dir / "arrays.npz",
            states=np.asarray(STATES),
            subjects=np.asarray(subjects),
            networks=np.asarray(NETWORK_ORDER),
            atom_names=np.asarray(["+".join(item) for item in all_atom_subsets()]),
            **arrays,
        )
        summaries.append(summary)
        append_jsonl(
            log_dir / "run_history.jsonl",
            {
                "run_name": f"{args.stage}_{identifier}",
                "status": "completed",
                "objective": "maximize network/atom state separation",
                "params": summary["params"],
                "metrics": {
                    "network_ratio": summary["network_all"]["between_within_ratio"],
                    "atom_ratio": summary["atom_all"]["between_within_ratio"],
                    "rest_min_margin_bits": summary["rest_min_mean_margin_bits"],
                    "skill_ratio": summary["heldout_skill_ratio_mean"],
                },
            },
        )
    atomic_json(
        output_dir / "batch_summary.json",
        {"stage": args.stage, "n_subjects": len(subjects), "subjects": subjects, "configs": summaries},
    )
    write_leaderboard(summaries, log_dir)
    best_network = max(summaries, key=lambda row: ranking_key(row, "network"))
    best_atom = max(summaries, key=lambda row: ranking_key(row, "atom"))
    best_balanced = max(summaries, key=lambda row: ranking_key(row, "balanced"))
    recommendation = {
        "best_network": best_network["config_id"],
        "best_atom": best_atom["config_id"],
        "best_balanced": best_balanced["config_id"],
    }
    atomic_json(output_dir / "recommendation.json", recommendation)
    update_live_status(
        log_dir,
        status="completed",
        running="无。",
        recent=(
            f"{args.stage} 完成；network={recommendation['best_network']}，"
            f"atom={recommendation['best_atom']}，balanced={recommendation['best_balanced']}。"
        ),
        next_step="读取 recommendation.json，启动全样本候选确认或局部细化。",
        log_name=f"{args.stage}.log",
    )
    return {"stage": args.stage, "n_subjects": len(subjects), **recommendation}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "screen", "refine", "confirm", "full"), default="screen")
    parser.add_argument("--config", type=parse_config, action="append", default=[])
    parser.add_argument("--max-subjects", type=int, default=8)
    parser.add_argument("--subject-offset", type=int, default=0)
    parser.add_argument("--permutation-repeats", type=int, default=500)
    parser.add_argument("--rest-root", type=Path, default=DEFAULT_REST_ROOT)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--labels", type=Path, default=default_yeo7_labels(500))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
