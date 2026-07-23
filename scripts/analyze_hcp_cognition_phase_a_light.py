#!/usr/bin/env python3
"""Lightweight Phase-A cognition prediction from existing HCP information features.

The script does not recompute brain dynamics. It pairs the frozen 29-subject
cognitive factor scores with existing Xi/Gaussian and TM-PEID results, compares
feature families under identical nested leave-one-subject-out Ridge prediction,
and maps age/sex-adjusted cognition associations by state, network, and atom.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COGNITION = ROOT / "results/hcp_single_group_sem_full_1206/selected_29_sem_results.csv"
DEFAULT_BEHAVIOR = ROOT / "data/hcp_unrestricted_selected_29.csv"
DEFAULT_XI_RECORDS = ROOT / "results/hcp_schaefer500_task_evoked_xi_tuning/full/records.jsonl"
XI_CONFIG_ID = "k1_p3_a1"
DEFAULT_TM_RECORDS = ROOT / "results/hcp_schaefer500_fixed_hierarchy_tm_peid/records.jsonl"
DEFAULT_TM_ARRAYS = (
    ROOT / "results/hcp_schaefer500_fixed_hierarchy_tm_peid/fixed_hierarchy_tm_peid.npz"
)
DEFAULT_OUTPUT = ROOT / "results/hcp_cognition_phase_a_light"

STATES = ("REST", "EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")
NETWORKS = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
TARGETS = ("g_score", "cry_score", "mem_score", "spd_score")
ALPHAS = np.logspace(-6, 6, 25)
SEED = 20260722

MODEL_ORDER = (
    "demographics",
    "xi_global",
    "xi_network_abs",
    "xi_network_share",
    "tm_whole_ei",
    "tm_phi_eid",
    "tm_network_abs",
    "tm_network_share",
    "tm_atoms_abs",
    "tm_atoms_share",
)

MODEL_LABELS = {
    "demographics": "Demographics",
    "xi_global": r"Final $\Xi$ global",
    "xi_network_abs": r"Final $\Xi$ network (abs)",
    "xi_network_share": r"Final $\Xi$ network (share)",
    "tm_whole_ei": "TM whole EI",
    "tm_phi_eid": r"TM $\Phi^{EID}$",
    "tm_network_abs": "TM network (abs)",
    "tm_network_share": "TM network (share)",
    "tm_atoms_abs": "TM atoms (abs)",
    "tm_atoms_share": "TM atoms (share)",
}

MODEL_COLORS = {
    "demographics": "#8A8A8A",
    "xi_global": "#4477AA",
    "xi_network_abs": "#6699CC",
    "xi_network_share": "#88BBDD",
    "tm_whole_ei": "#CC8844",
    "tm_phi_eid": "#DDAA66",
    "tm_network_abs": "#228866",
    "tm_network_share": "#66AA88",
    "tm_atoms_abs": "#8866AA",
    "tm_atoms_share": "#AA88CC",
}


@dataclass(frozen=True)
class Inputs:
    subjects: np.ndarray
    targets: dict[str, np.ndarray]
    demographics: np.ndarray
    demographic_names: list[str]
    xi_global: np.ndarray
    xi_network_abs: np.ndarray
    xi_network_share: np.ndarray
    tm_whole_ei: np.ndarray
    tm_phi_eid: np.ndarray
    tm_network_abs: np.ndarray
    tm_network_share: np.ndarray
    tm_atoms_abs: np.ndarray
    tm_atoms_share: np.ndarray
    atom_names: list[str]
    identity_checks: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cognition", type=Path, default=DEFAULT_COGNITION)
    parser.add_argument("--behavior", type=Path, default=DEFAULT_BEHAVIOR)
    parser.add_argument("--xi-records", type=Path, default=DEFAULT_XI_RECORDS)
    parser.add_argument("--tm-records", type=Path, default=DEFAULT_TM_RECORDS)
    parser.add_argument("--tm-arrays", type=Path, default=DEFAULT_TM_ARRAYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prediction-permutations", type=int, default=1000)
    parser.add_argument("--association-permutations", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def strip_subject(value: Any) -> str:
    return str(value).removeprefix("sub-")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_inputs(args: argparse.Namespace) -> Inputs:
    cognition = pd.read_csv(args.cognition, dtype={"Subject": str})
    behavior = pd.read_csv(args.behavior, dtype={"Subject": str})
    for column in ("Subject", *TARGETS):
        if column not in cognition:
            raise ValueError(f"Missing cognitive column: {column}")
    for column in ("Subject", "Age", "Gender"):
        if column not in behavior:
            raise ValueError(f"Missing demographic column: {column}")

    subjects = cognition["Subject"].map(strip_subject).to_numpy(dtype=str)
    if len(subjects) != 29 or len(set(subjects)) != 29:
        raise ValueError(f"Expected 29 unique cognitive subjects, found {len(set(subjects))}.")
    targets = {name: cognition[name].to_numpy(dtype=float) for name in TARGETS}
    if not all(np.isfinite(values).all() for values in targets.values()):
        raise ValueError("Non-finite cognitive factor score found.")

    behavior = behavior.assign(Subject=behavior["Subject"].map(strip_subject)).set_index("Subject")
    demo_frame = behavior.loc[subjects, ["Age", "Gender"]].copy()
    demo_frame["Age"] = pd.Categorical(
        demo_frame["Age"], categories=["22-25", "26-30", "31-35", "36+"]
    )
    demo_frame["Gender"] = pd.Categorical(demo_frame["Gender"], categories=["F", "M"])
    demo_frame = pd.get_dummies(demo_frame, drop_first=True, dtype=float)
    demographics = demo_frame.to_numpy(dtype=float)

    xi_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for line in args.xi_records.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("config_id")) != XI_CONFIG_ID:
            continue
        key = (strip_subject(row["subject"]), str(row["state"]))
        if key in xi_rows:
            raise ValueError(f"Duplicate Xi record for {key} under {XI_CONFIG_ID}")
        xi_rows[key] = row
    xi_global = np.empty((len(subjects), len(STATES)), dtype=float)
    xi_network_abs_3d = np.empty((len(subjects), len(STATES), len(NETWORKS)), dtype=float)
    for i, subject in enumerate(subjects):
        for s, state in enumerate(STATES):
            row = xi_rows[(subject, state)]
            xi_global[i, s] = float(row["system_xi"])
            xi_network_abs_3d[i, s] = [float(row["network_attribution"][n]) for n in NETWORKS]
    xi_network_share_3d = xi_network_abs_3d / xi_global[:, :, None]

    tm = np.load(args.tm_arrays, allow_pickle=True)
    tm_states = tuple(str(x) for x in tm["states"])
    tm_subjects = np.asarray([strip_subject(x) for x in tm["common_subjects"]], dtype=str)
    atom_names = [str(x) for x in tm["atom_names"]]
    if tm_states != STATES:
        raise ValueError(f"TM state order mismatch: {tm_states}")
    if not np.array_equal(tm_subjects, subjects):
        raise ValueError("Cognition and TM subject order do not match exactly.")
    tm_phi_state_subject = np.asarray(tm["absolute_phi"], dtype=float)
    tm_network_state_subject = np.asarray(tm["absolute_contribution"], dtype=float)
    tm_atoms_state_subject = np.asarray(tm["absolute_atoms"], dtype=float)
    tm_phi_eid = tm_phi_state_subject.T
    tm_network_abs_3d = np.transpose(tm_network_state_subject, (1, 0, 2))
    tm_atoms_abs_3d = np.transpose(tm_atoms_state_subject, (1, 0, 2))
    tm_network_share_3d = tm_network_abs_3d / tm_phi_eid[:, :, None]
    tm_atoms_share_3d = tm_atoms_abs_3d / tm_phi_eid[:, :, None]

    record_rows = [json.loads(line) for line in args.tm_records.read_text().splitlines() if line.strip()]
    tm_record_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in record_rows:
        state = str(row["condition"])
        variant = str(row["variant"])
        if (state == "REST" and variant == "rest") or (state != "REST" and variant == "retained"):
            subject = strip_subject(row["subject"])
            if subject in set(subjects):
                tm_record_map[(subject, state)] = row
    tm_whole_ei = np.empty((len(subjects), len(STATES)), dtype=float)
    for i, subject in enumerate(subjects):
        for s, state in enumerate(STATES):
            tm_whole_ei[i, s] = float(tm_record_map[(subject, state)]["whole_ei"])

    xi_sum_error = float(np.max(np.abs(xi_network_abs_3d.sum(axis=2) - xi_global)))
    tm_network_sum_error = float(np.max(np.abs(tm_network_abs_3d.sum(axis=2) - tm_phi_eid)))
    tm_atom_sum_error = float(np.max(np.abs(tm_atoms_abs_3d.sum(axis=2) - tm_phi_eid)))
    identity_checks = {
        "xi_network_sum_max_abs_error": xi_sum_error,
        "tm_network_sum_max_abs_error": tm_network_sum_error,
        "tm_atom_sum_max_abs_error": tm_atom_sum_error,
    }
    if max(identity_checks.values()) > 1e-10:
        raise AssertionError(f"Attribution identity failed: {identity_checks}")

    arrays = (
        demographics,
        xi_global,
        xi_network_abs_3d,
        xi_network_share_3d,
        tm_whole_ei,
        tm_phi_eid,
        tm_network_abs_3d,
        tm_network_share_3d,
        tm_atoms_abs_3d,
        tm_atoms_share_3d,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Non-finite value found in a feature array.")

    return Inputs(
        subjects=subjects,
        targets=targets,
        demographics=demographics,
        demographic_names=list(demo_frame.columns),
        xi_global=xi_global,
        xi_network_abs=xi_network_abs_3d,
        xi_network_share=xi_network_share_3d,
        tm_whole_ei=tm_whole_ei,
        tm_phi_eid=tm_phi_eid,
        tm_network_abs=tm_network_abs_3d,
        tm_network_share=tm_network_share_3d,
        tm_atoms_abs=tm_atoms_abs_3d,
        tm_atoms_share=tm_atoms_share_3d,
        atom_names=atom_names,
        identity_checks=identity_checks,
    )


def flatten_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values.reshape(values.shape[0], -1)


def build_feature_sets(inputs: Inputs) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    demo = inputs.demographics
    brain_sets = {
        "demographics": np.empty((len(inputs.subjects), 0), dtype=float),
        "xi_global": inputs.xi_global,
        "xi_network_abs": flatten_features(inputs.xi_network_abs),
        "xi_network_share": flatten_features(inputs.xi_network_share),
        "tm_whole_ei": inputs.tm_whole_ei,
        "tm_phi_eid": inputs.tm_phi_eid,
        "tm_network_abs": flatten_features(inputs.tm_network_abs),
        "tm_network_share": flatten_features(inputs.tm_network_share),
        "tm_atoms_abs": flatten_features(inputs.tm_atoms_abs),
        "tm_atoms_share": flatten_features(inputs.tm_atoms_share),
    }
    brain_names = {
        "demographics": [],
        "xi_global": [f"{state}__xi" for state in STATES],
        "xi_network_abs": [f"{state}__{network}" for state in STATES for network in NETWORKS],
        "xi_network_share": [f"{state}__{network}" for state in STATES for network in NETWORKS],
        "tm_whole_ei": [f"{state}__whole_ei" for state in STATES],
        "tm_phi_eid": [f"{state}__phi_eid" for state in STATES],
        "tm_network_abs": [f"{state}__{network}" for state in STATES for network in NETWORKS],
        "tm_network_share": [f"{state}__{network}" for state in STATES for network in NETWORKS],
        "tm_atoms_abs": [f"{state}__{atom}" for state in STATES for atom in inputs.atom_names],
        "tm_atoms_share": [f"{state}__{atom}" for state in STATES for atom in inputs.atom_names],
    }
    feature_sets: dict[str, np.ndarray] = {}
    feature_names: dict[str, list[str]] = {}
    for model in MODEL_ORDER:
        feature_sets[model] = np.column_stack([demo, brain_sets[model]])
        feature_names[model] = [*inputs.demographic_names, *brain_names[model]]
        if feature_sets[model].shape[1] != len(feature_names[model]):
            raise AssertionError(f"Feature name mismatch for {model}.")
    return feature_sets, feature_names


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(pearsonr(y_true, y_pred).statistic)


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson_r": safe_pearson(y_true, y_pred),
        "spearman_rho": safe_spearman(y_true, y_pred),
    }


def nested_loo_predictions(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    predictions = np.empty(n, dtype=float)
    selected_alphas = np.empty(n, dtype=float)
    for test_index in range(n):
        train = np.arange(n) != test_index
        model = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=ALPHAS, cv=None, scoring="neg_mean_squared_error"),
        )
        model.fit(x[train], y[train])
        predictions[test_index] = float(model.predict(x[[test_index]])[0])
        selected_alphas[test_index] = float(model.named_steps["ridgecv"].alpha_)
    return predictions, selected_alphas


def run_observed_predictions(
    feature_sets: dict[str, np.ndarray], targets: dict[str, np.ndarray]
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], np.ndarray]]:
    results: dict[str, dict[str, Any]] = {}
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for target_name, y in targets.items():
        results[target_name] = {}
        for model_name in MODEL_ORDER:
            pred, alphas = nested_loo_predictions(feature_sets[model_name], y)
            predictions[(target_name, model_name)] = pred
            results[target_name][model_name] = {
                **prediction_metrics(y, pred),
                "n_features_total": int(feature_sets[model_name].shape[1]),
                "median_selected_alpha": float(np.median(alphas)),
                "selected_alpha_min": float(np.min(alphas)),
                "selected_alpha_max": float(np.max(alphas)),
            }
    return results, predictions


def run_prediction_permutations(
    feature_sets: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    permutations = np.stack([rng.permutation(len(y)) for _ in range(n_permutations)]).astype(
        np.int16
    )

    def evaluate(order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r2_values = np.empty(len(MODEL_ORDER), dtype=float)
        mae_values = np.empty(len(MODEL_ORDER), dtype=float)
        y_perm = y[order]
        for m, model_name in enumerate(MODEL_ORDER):
            pred, _ = nested_loo_predictions(feature_sets[model_name], y_perm)
            metrics = prediction_metrics(y_perm, pred)
            r2_values[m] = metrics["r2"]
            mae_values[m] = metrics["mae"]
        return r2_values, mae_values

    evaluated = Parallel(n_jobs=max(1, workers), prefer="processes", batch_size=1)(
        delayed(evaluate)(order) for order in permutations
    )
    r2_null = np.stack([item[0] for item in evaluated])
    mae_null = np.stack([item[1] for item in evaluated])
    return r2_null, mae_null, permutations


def add_permutation_statistics(
    primary_results: dict[str, Any], r2_null: np.ndarray, mae_null: np.ndarray
) -> None:
    baseline_index = MODEL_ORDER.index("demographics")
    observed_baseline_r2 = float(primary_results["demographics"]["r2"])
    observed_baseline_mae = float(primary_results["demographics"]["mae"])
    for m, model_name in enumerate(MODEL_ORDER):
        observed_r2 = float(primary_results[model_name]["r2"])
        observed_mae = float(primary_results[model_name]["mae"])
        delta_r2 = observed_r2 - observed_baseline_r2
        mae_improvement = observed_baseline_mae - observed_mae
        null_delta_r2 = r2_null[:, m] - r2_null[:, baseline_index]
        null_mae_improvement = mae_null[:, baseline_index] - mae_null[:, m]
        primary_results[model_name].update(
            {
                "permutation_p_r2": float(
                    (1 + np.count_nonzero(r2_null[:, m] >= observed_r2)) / (len(r2_null) + 1)
                ),
                "delta_r2_vs_demographics": float(delta_r2),
                "permutation_p_delta_r2": float(
                    (1 + np.count_nonzero(null_delta_r2 >= delta_r2)) / (len(r2_null) + 1)
                ),
                "mae_improvement_vs_demographics": float(mae_improvement),
                "permutation_p_mae_improvement": float(
                    (1 + np.count_nonzero(null_mae_improvement >= mae_improvement))
                    / (len(r2_null) + 1)
                ),
                "null_r2_median": float(np.median(r2_null[:, m])),
                "null_r2_ci95": np.quantile(r2_null[:, m], [0.025, 0.975]).tolist(),
                "null_mae_improvement_ci95": np.quantile(
                    null_mae_improvement, [0.025, 0.975]
                ).tolist(),
            }
        )


def residual_maker(covariates: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(covariates)), np.asarray(covariates, dtype=float)])
    return np.eye(len(design)) - design @ np.linalg.pinv(design)


def normalize_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=0, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def association_permutation_test(
    y: np.ndarray,
    x: np.ndarray,
    covariates: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
) -> dict[str, np.ndarray]:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    residualizer = residual_maker(covariates)
    y_fitted = y - residualizer @ y
    y_residual = residualizer @ y
    x_residual = residualizer @ x
    x_normalized = normalize_columns(x_residual)
    y_observed = normalize_columns(y_residual[:, None])[:, 0]
    observed_r = y_observed @ x_normalized

    rng = np.random.default_rng(seed)
    permutations = np.stack([rng.permutation(len(y)) for _ in range(n_permutations)])
    y_star = y_fitted[None, :] + y_residual[permutations]
    permuted_residuals = (residualizer @ y_star.T).T
    norms = np.linalg.norm(permuted_residuals, axis=1, keepdims=True)
    permuted_normalized = np.divide(
        permuted_residuals,
        norms,
        out=np.zeros_like(permuted_residuals),
        where=norms > 0,
    )
    permuted_r = permuted_normalized @ x_normalized
    abs_permuted = np.abs(permuted_r)
    abs_observed = np.abs(observed_r)
    p_uncorrected = (1 + np.sum(abs_permuted >= abs_observed[None, :], axis=0)) / (
        n_permutations + 1
    )
    max_abs = np.max(abs_permuted, axis=1)
    p_max_t = (1 + np.sum(max_abs[:, None] >= abs_observed[None, :], axis=0)) / (
        n_permutations + 1
    )
    q_fdr = multipletests(p_uncorrected, method="fdr_bh")[1]
    return {
        "partial_r": observed_r,
        "p_uncorrected": p_uncorrected,
        "p_max_t": p_max_t,
        "q_fdr": q_fdr,
        "null_max_abs_r": max_abs,
    }


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_prediction_comparison(
    results: dict[str, Any],
    predictions: dict[tuple[str, str], np.ndarray],
    y: np.ndarray,
    output_dir: Path,
) -> str:
    configure_style()
    primary = results["g_score"]
    y_positions = np.arange(len(MODEL_ORDER))
    fig = plt.figure(figsize=(7.2, 3.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.1, 1.0])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    for y_pos, model_name in zip(y_positions, MODEL_ORDER):
        stats = primary[model_name]
        low, high = stats["null_r2_ci95"]
        ax_a.plot([low, high], [y_pos, y_pos], color="#C8C8C8", lw=2.2, solid_capstyle="round")
        ax_a.scatter(
            stats["r2"],
            y_pos,
            s=28,
            color=MODEL_COLORS[model_name],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    ax_a.axvline(0, color="#555555", lw=0.7, ls="--")
    ax_a.set_yticks(y_positions, [MODEL_LABELS[m] for m in MODEL_ORDER])
    ax_a.invert_yaxis()
    ax_a.set_xlabel(r"Out-of-fold $R^2$")
    ax_a.text(-0.12, 1.03, "a", transform=ax_a.transAxes, fontweight="bold", fontsize=8)
    ax_a.text(0.02, 0.02, "grey: permutation 95% interval", transform=ax_a.transAxes, fontsize=5.8)

    baseline_mae = primary["demographics"]["mae"]
    for y_pos, model_name in zip(y_positions, MODEL_ORDER):
        stats = primary[model_name]
        low, high = stats["null_mae_improvement_ci95"]
        ax_b.plot([low, high], [y_pos, y_pos], color="#C8C8C8", lw=2.2, solid_capstyle="round")
        ax_b.scatter(
            stats["mae_improvement_vs_demographics"],
            y_pos,
            s=28,
            color=MODEL_COLORS[model_name],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    ax_b.axvline(0, color="#555555", lw=0.7, ls="--")
    ax_b.set_yticks([])
    ax_b.set_ylim(ax_a.get_ylim())
    ax_b.set_xlabel("MAE improvement vs demographics")
    ax_b.text(-0.12, 1.03, "b", transform=ax_b.transAxes, fontweight="bold", fontsize=8)
    ax_b.text(
        0.02,
        0.02,
        f"baseline MAE = {baseline_mae:.3f}",
        transform=ax_b.transAxes,
        fontsize=5.8,
    )

    information_models = [m for m in MODEL_ORDER if m != "demographics"]
    best_model = max(information_models, key=lambda name: primary[name]["r2"])
    displayed_model = "xi_global"
    pred = predictions[("g_score", displayed_model)]
    lo = float(min(y.min(), pred.min()))
    hi = float(max(y.max(), pred.max()))
    ax_c.plot([lo, hi], [lo, hi], color="#777777", lw=0.8, ls="--", zorder=1)
    ax_c.scatter(
        y,
        pred,
        s=22,
        color=MODEL_COLORS[displayed_model],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
    )
    ax_c.set_xlabel("Observed g score")
    ax_c.set_ylabel("Out-of-fold prediction")
    ax_c.text(-0.18, 1.03, "c", transform=ax_c.transAxes, fontweight="bold", fontsize=8)
    ax_c.text(
        0.03,
        0.97,
        f"{MODEL_LABELS[displayed_model]} ({XI_CONFIG_ID})\n"
        f"$R^2$={primary[displayed_model]['r2']:.3f}, "
        f"$p_{{perm}}$={primary[displayed_model]['permutation_p_r2']:.3f}\n"
        f"$\Delta R^2$={primary[displayed_model]['delta_r2_vs_demographics']:.3f}, "
        f"$p_{{\Delta}}$={primary[displayed_model]['permutation_p_delta_r2']:.3f}\n"
        "same model as REST-task experiment",
        transform=ax_c.transAxes,
        va="top",
        fontsize=5.8,
    )
    sns.despine(fig=fig)
    stem = output_dir / "phase_a_light_prediction_comparison"
    save_figure(fig, stem)
    return best_model


def heatmap_panel(
    ax: plt.Axes,
    values: np.ndarray,
    corrected_p: np.ndarray,
    row_labels: list[str],
    *,
    vmax: float,
    show_y: bool,
) -> None:
    sns.heatmap(
        values,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        cbar=False,
        linewidths=0.25,
        linecolor="white",
        xticklabels=[state.title() if state != "WM" else "WM" for state in STATES],
        yticklabels=row_labels if show_y else False,
    )
    for row, col in np.argwhere(corrected_p < 0.05):
        ax.text(col + 0.5, row + 0.52, "*", ha="center", va="center", fontsize=7, color="black")
    ax.tick_params(axis="x", rotation=45, length=0)
    ax.tick_params(axis="y", rotation=0, length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_association_maps(
    association: dict[str, np.ndarray], inputs: Inputs, output_dir: Path
) -> None:
    configure_style()
    n_xi = len(STATES) * len(NETWORKS)
    n_tm_network = n_xi
    xi_slice = slice(0, n_xi)
    tm_network_slice = slice(n_xi, n_xi + n_tm_network)
    tm_atom_slice = slice(n_xi + n_tm_network, None)
    xi_r = association["partial_r"][xi_slice].reshape(len(STATES), len(NETWORKS)).T
    tm_network_r = association["partial_r"][tm_network_slice].reshape(
        len(STATES), len(NETWORKS)
    ).T
    tm_atom_r = association["partial_r"][tm_atom_slice].reshape(
        len(STATES), len(inputs.atom_names)
    ).T
    xi_p = association["p_max_t"][xi_slice].reshape(len(STATES), len(NETWORKS)).T
    tm_network_p = association["p_max_t"][tm_network_slice].reshape(
        len(STATES), len(NETWORKS)
    ).T
    tm_atom_p = association["p_max_t"][tm_atom_slice].reshape(
        len(STATES), len(inputs.atom_names)
    ).T
    vmax = max(float(np.max(np.abs(xi_r))), float(np.max(np.abs(tm_network_r))), float(np.max(np.abs(tm_atom_r))))
    vmax = max(vmax, 0.1)

    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.55], width_ratios=[1, 1])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    heatmap_panel(ax_a, xi_r, xi_p, list(NETWORKS), vmax=vmax, show_y=True)
    heatmap_panel(ax_b, tm_network_r, tm_network_p, list(NETWORKS), vmax=vmax, show_y=True)
    heatmap_panel(ax_c, tm_atom_r, tm_atom_p, inputs.atom_names, vmax=vmax, show_y=True)
    ax_a.text(-0.12, 1.06, "a", transform=ax_a.transAxes, fontweight="bold", fontsize=8)
    ax_b.text(-0.12, 1.06, "b", transform=ax_b.transAxes, fontweight="bold", fontsize=8)
    ax_c.text(-0.06, 1.04, "c", transform=ax_c.transAxes, fontweight="bold", fontsize=8)
    ax_a.text(0.0, 1.03, r"$\Xi$ network attribution", transform=ax_a.transAxes, fontsize=6.5)
    ax_b.text(0.0, 1.03, "TM network contribution", transform=ax_b.transAxes, fontsize=6.5)
    ax_c.text(0.0, 1.02, "TM hierarchy atoms", transform=ax_c.transAxes, fontsize=6.5)
    scalar = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-vmax, vmax=vmax), cmap="vlag")
    cbar = fig.colorbar(scalar, ax=[ax_a, ax_b, ax_c], location="right", shrink=0.68, pad=0.02)
    cbar.set_label("Partial r with g score\n(adjusted for age and sex)")
    fig.text(0.5, -0.005, "* family-wise max-T p < 0.05 across all displayed cells", ha="center", fontsize=5.8)
    stem = output_dir / "phase_a_light_attribution_associations"
    save_figure(fig, stem)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_report(
    output_dir: Path,
    inputs: Inputs,
    results: dict[str, Any],
    best_model: str,
    association: dict[str, np.ndarray],
    association_names: list[str],
    *,
    prediction_permutations: int,
    association_permutations: int,
    smoke: bool,
) -> None:
    primary = results["g_score"]
    overall_success = [
        model
        for model in MODEL_ORDER
        if model != "demographics"
        and primary[model]["r2"] > 0
        and primary[model]["permutation_p_r2"] < 0.05
    ]
    top_indices = np.argsort(np.abs(association["partial_r"]))[::-1][:10]
    max_t_hits = np.flatnonzero(association["p_max_t"] < 0.05)

    lines = [
        "# HCP 认知预测阶段 A：轻量结果",
        "",
        "## 结论",
        "",
        f"与 REST—任务主实验统一的 `{XI_CONFIG_ID}` system-level $\\Xi$ 模型为："
        f"$R^2={primary['xi_global']['r2']:.4f}$，MAE={primary['xi_global']['mae']:.4f}，"
        f"置换 $p={primary['xi_global']['permutation_p_r2']:.4f}$；"
        f"相对人口学基线 $\\Delta R^2={primary['xi_global']['delta_r2_vs_demographics']:.4f}$。"
        "该结果没有支持稳定的样本外认知预测。",
        "",
    ]
    if overall_success:
        lines.append(
            "跨模型探索比较中，至少一个“人口学 + 脑信息”模型获得正的样本外预测性能并超过共享认知标签置换 null："
            + "、".join(f"`{name}`" for name in overall_success)
            + "。它不是与 REST—任务主实验同源的模型，不能替换冻结的主指标。"
        )
    else:
        lines.append(
            "本轮没有信息特征家族同时满足正的样本外 $R^2$ 和置换 $p<0.05$。"
            "因此现有 29 人结果不能支持可靠的个体认知预测；脑区关联只能用于提出后续假设。"
        )
    lines.extend(
        [
            "",
            f"探索性比较中表现最高的信息模型是 `{best_model}`：$R^2={primary[best_model]['r2']:.4f}$，"
            f"MAE={primary[best_model]['mae']:.4f}，置换 $p={primary[best_model]['permutation_p_r2']:.4f}$。",
            "",
            f"相对人口学基线，该模型的 $\Delta R^2={primary[best_model]['delta_r2_vs_demographics']:.4f}$，"
            f"置换 $p={primary[best_model]['permutation_p_delta_r2']:.4f}$；"
            f"MAE 改善为 {primary[best_model]['mae_improvement_vs_demographics']:.4f}，"
            f"置换 $p={primary[best_model]['permutation_p_mae_improvement']:.4f}$。"
            "因此主指标 $\Delta R^2$ 尚未证明独立增量，次要 MAE 指标出现候选增量信号。",
            "",
            "![预测模型比较](phase_a_light_prediction_comparison.png)",
            "",
            "## 受控比较",
            "",
            "所有模型使用相同的 29 名被试、同一认知目标、相同外层 LOOCV、训练折内标准化、"
            "相同 Ridge alpha 网格和共享标签置换。唯一改变的是脑信息特征家族。",
            "",
            f"主目标使用 {prediction_permutations:,} 次标签置换；归因定位使用 "
            f"{association_permutations:,} 次 Freedman–Lane 残差置换。"
            + (" 本文件来自 smoke test，不用于正式解释。" if smoke else ""),
            "",
            "## 一般认知预测",
            "",
            "| 模型 | 特征数 | OOF R² | MAE | p(R²) | ΔR² | p(ΔR²) | MAE改善 | p(MAE改善) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODEL_ORDER:
        item = primary[model]
        lines.append(
            f"| {MODEL_LABELS[model]} | {item['n_features_total']} | {item['r2']:.4f} | "
            f"{item['mae']:.4f} | {item['permutation_p_r2']:.4f} | "
            f"{item['delta_r2_vs_demographics']:.4f} | {item['permutation_p_delta_r2']:.4f} | "
            f"{item['mae_improvement_vs_demographics']:.4f} | "
            f"{item['permutation_p_mae_improvement']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 次要认知目标（仅探索性点估计）",
            "",
            "| 目标 | 表现最高的信息模型 | OOF R² | MAE |",
            "|---|---|---:|---:|",
        ]
    )
    for target in TARGETS[1:]:
        target_best = max(
            (model for model in MODEL_ORDER if model != "demographics"),
            key=lambda model: results[target][model]["r2"],
        )
        item = results[target][target_best]
        lines.append(f"| {target} | {MODEL_LABELS[target_best]} | {item['r2']:.4f} | {item['mae']:.4f} |")

    lines.extend(
        [
            "",
            "## 网络与层级定位",
            "",
            "热图显示控制年龄和性别后的偏相关。星号要求在全部 216 个展示单元中通过 max-T 家族校正。",
            "",
            "![网络与层级关联](phase_a_light_attribution_associations.png)",
            "",
        ]
    )
    if len(max_t_hits):
        lines.append("通过 max-T 校正的单元：")
        lines.append("")
        for index in max_t_hits:
            lines.append(
                f"- `{association_names[index]}`：partial r={association['partial_r'][index]:.4f}，"
                f"p_maxT={association['p_max_t'][index]:.4f}。"
            )
    else:
        lines.append("没有网络或 atom 单元通过全局 max-T 校正。绝对偏相关最大的十个单元仅作为候选：")
        lines.append("")
        for index in top_indices:
            lines.append(
                f"- `{association_names[index]}`：partial r={association['partial_r'][index]:.4f}，"
                f"p_unc={association['p_uncorrected'][index]:.4f}，"
                f"p_maxT={association['p_max_t'][index]:.4f}。"
            )

    lines.extend(
        [
            "",
            "## 质量检查",
            "",
            f"- 认知、Xi 和 TM-PEID 的共同被试数：{len(inputs.subjects)}。",
            f"- Xi 网络贡献加和最大误差：{inputs.identity_checks['xi_network_sum_max_abs_error']:.3e}。",
            f"- TM 网络贡献加和最大误差：{inputs.identity_checks['tm_network_sum_max_abs_error']:.3e}。",
            f"- TM atom 加和最大误差：{inputs.identity_checks['tm_atom_sum_max_abs_error']:.3e}。",
            "",
            "## 解释边界",
            "",
            "- 本轮没有 FC 和 $\\Phi^R$，不能回答 Xi 是否优于这两类基线。",
            "- 轻量方案只控制年龄组和性别，没有完整头动、DVARS、tSNR 和家庭结构。",
            "- 高维网络和 atom 特征多于全局特征；嵌套 Ridge 降低过拟合，但不能消除维数差异的解释问题。",
            "- 29 人的跨模型最佳结果来自多个候选家族，存在选择偏差；预测图 panel c 固定展示同源的最终 $\\Xi$，而不是事后最佳模型。",
            "- 结果来自已有单次 LR 指标，尚未检验 LR/RL 重测信度。",
            "- 预测或偏相关均不构成因果证据。",
            "",
            "## 下一步",
            "",
            f"最小的下一步是优先为 `{XI_CONFIG_ID}` system-level $\\Xi$ 补齐头动/QC、静态 FC 和 $\\Phi^R$，"
            "再在完全相同的 29 折划分中检验其 $\\Delta R^2$ 和 MAE 改善是否成立。"
            "当前 LANGUAGE 网络候选和其他网络/atom 定位均未通过全局校正，不应据此选择脑区。",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.prediction_permutations = min(args.prediction_permutations, 10)
        args.association_permutations = min(args.association_permutations, 100)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(args)
    feature_sets, feature_names = build_feature_sets(inputs)
    results, predictions = run_observed_predictions(feature_sets, inputs.targets)

    r2_null, mae_null, prediction_permutation_indices = run_prediction_permutations(
        feature_sets,
        inputs.targets["g_score"],
        n_permutations=args.prediction_permutations,
        seed=SEED,
        workers=args.workers,
    )
    add_permutation_statistics(results["g_score"], r2_null, mae_null)

    association_blocks = [
        flatten_features(inputs.xi_network_abs),
        flatten_features(inputs.tm_network_abs),
        flatten_features(inputs.tm_atoms_abs),
    ]
    association_names = [
        *[f"xi_network__{state}__{network}" for state in STATES for network in NETWORKS],
        *[f"tm_network__{state}__{network}" for state in STATES for network in NETWORKS],
        *[f"tm_atom__{state}__{atom}" for state in STATES for atom in inputs.atom_names],
    ]
    association_x = np.column_stack(association_blocks)
    association = association_permutation_test(
        inputs.targets["g_score"],
        association_x,
        inputs.demographics,
        n_permutations=args.association_permutations,
        seed=SEED + 1,
    )

    best_model = plot_prediction_comparison(
        results, predictions, inputs.targets["g_score"], args.output_dir
    )
    plot_association_maps(association, inputs, args.output_dir)

    summary = {
        "analysis": "HCP cognition Phase-A lightweight prediction from cached Xi and TM-PEID features",
        "status": "smoke" if args.smoke else "full_lightweight_phase_a",
        "config": {
            "subjects": inputs.subjects.tolist(),
            "n_subjects": len(inputs.subjects),
            "states": list(STATES),
            "networks": list(NETWORKS),
            "atom_names": inputs.atom_names,
            "targets": list(TARGETS),
            "model_order": list(MODEL_ORDER),
            "primary_brain_model": {
                "feature": "system_xi",
                "config_id": XI_CONFIG_ID,
                "representation": "Yeo7 PC1 with task-evoked PCA",
                "history_order": 3,
                "ridge_alpha": 1.0,
            },
            "alpha_grid": ALPHAS.tolist(),
            "outer_cv": "leave-one-subject-out",
            "inner_selection": "RidgeCV efficient leave-one-out on outer training fold",
            "prediction_permutations": args.prediction_permutations,
            "association_permutations": args.association_permutations,
            "seed": SEED,
            "workers": args.workers,
            "controlled_covariates": ["Age group", "Gender"],
            "not_in_lightweight_phase": ["static FC", "Phi-R", "complete motion/QC", "family blocks"],
        },
        "identity_checks": inputs.identity_checks,
        "feature_names": feature_names,
        "prediction_results": results,
        "best_information_model_for_g": best_model,
        "association": {
            "feature_names": association_names,
            "n_max_t_significant": int(np.count_nonzero(association["p_max_t"] < 0.05)),
            "top_by_absolute_partial_r": [
                {
                    "feature": association_names[index],
                    "partial_r": float(association["partial_r"][index]),
                    "p_uncorrected": float(association["p_uncorrected"][index]),
                    "p_max_t": float(association["p_max_t"][index]),
                    "q_fdr": float(association["q_fdr"][index]),
                }
                for index in np.argsort(np.abs(association["partial_r"]))[::-1][:20]
            ],
        },
        "conclusion_flags": {
            "positive_r2_and_absolute_permutation_p_lt_0_05": [
                model
                for model in MODEL_ORDER
                if model != "demographics"
                and results["g_score"][model]["r2"] > 0
                and results["g_score"][model]["permutation_p_r2"] < 0.05
            ],
            "incremental_r2_permutation_p_lt_0_05": [
                model
                for model in MODEL_ORDER
                if model != "demographics"
                and results["g_score"][model]["delta_r2_vs_demographics"] > 0
                and results["g_score"][model]["permutation_p_delta_r2"] < 0.05
            ],
            "incremental_mae_permutation_p_lt_0_05": [
                model
                for model in MODEL_ORDER
                if model != "demographics"
                and results["g_score"][model]["mae_improvement_vs_demographics"] > 0
                and results["g_score"][model]["permutation_p_mae_improvement"] < 0.05
            ],
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n"
    )
    np.savez_compressed(
        args.output_dir / "oof_predictions.npz",
        subjects=inputs.subjects,
        model_names=np.asarray(MODEL_ORDER),
        target_names=np.asarray(TARGETS),
        observed_targets=np.stack([inputs.targets[target] for target in TARGETS]),
        predictions=np.stack(
            [
                np.stack([predictions[(target, model)] for model in MODEL_ORDER])
                for target in TARGETS
            ]
        ),
    )
    np.savez_compressed(
        args.output_dir / "null_distributions.npz",
        model_names=np.asarray(MODEL_ORDER),
        r2_null=r2_null,
        mae_null=mae_null,
        prediction_permutation_indices=prediction_permutation_indices,
    )
    np.savez_compressed(
        args.output_dir / "association_results.npz",
        feature_names=np.asarray(association_names),
        **association,
    )
    write_report(
        args.output_dir,
        inputs,
        results,
        best_model,
        association,
        association_names,
        prediction_permutations=args.prediction_permutations,
        association_permutations=args.association_permutations,
        smoke=args.smoke,
    )
    print(
        json.dumps(
            {
                "status": "smoke" if args.smoke else "complete",
                "output_dir": str(args.output_dir),
                "best_information_model_for_g": best_model,
                "best_r2": results["g_score"][best_model]["r2"],
                "best_permutation_p": results["g_score"][best_model]["permutation_p_r2"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
