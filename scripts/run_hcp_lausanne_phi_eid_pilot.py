from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "hcp_lausanne_phi_eid_pilot"
DEFAULT_FIG_NULL = ROOT / "fig" / "hcp_lausanne_phi_eid_null_comparison"
DEFAULT_FIG_DECOMP = ROOT / "fig" / "hcp_lausanne_phi_eid_decomposition"
DEFAULT_REPORT = ROOT / "docs" / "log" / "hcp_lausanne_phi_eid_pilot.md"
DEFAULT_HCP_ROOT = ROOT / "data" / "hcp_s1200"
DEFAULT_SUBJECTS = (
    "100307",
    "103414",
    "105115",
    "110411",
    "111312",
    "113619",
    "115320",
    "117122",
    "118528",
    "118730",
)
DEFAULT_RUNS = ("REST1_LR",)
HCP_S3_ROOT = "s3://hcp-openaccess/HCP_1200"
MODULE_ORDER = ("DMN", "Som", "Vis", "VAN", "DAN", "FPN", "Lim", "Sub")
MODULE_COLORS = {
    "DMN": "#D55E00",
    "Som": "#E69F00",
    "Vis": "#009E73",
    "VAN": "#56B4E9",
    "DAN": "#0072B2",
    "FPN": "#CC79A7",
    "Lim": "#F0E442",
    "Sub": "#7F7F7F",
}


@dataclass(frozen=True)
class HcpRunPaths:
    functional: Path
    aparc_aseg: Path
    motion: Path | None


@dataclass(frozen=True)
class GreedyAtom:
    sources: tuple[str, ...]
    value: float
    kind: str
    depth: int


def build_freesurfer_lausanne83_mapping() -> dict[int, str]:
    cortical = {
        1: "bankssts",
        2: "caudalanteriorcingulate",
        3: "caudalmiddlefrontal",
        5: "cuneus",
        6: "entorhinal",
        7: "fusiform",
        8: "inferiorparietal",
        9: "inferiortemporal",
        10: "isthmuscingulate",
        11: "lateraloccipital",
        12: "lateralorbitofrontal",
        13: "lingual",
        14: "medialorbitofrontal",
        15: "middletemporal",
        16: "parahippocampal",
        17: "paracentral",
        18: "parsopercularis",
        19: "parsorbitalis",
        20: "parstriangularis",
        21: "pericalcarine",
        22: "postcentral",
        23: "posteriorcingulate",
        24: "precentral",
        25: "precuneus",
        26: "rostralanteriorcingulate",
        27: "rostralmiddlefrontal",
        28: "superiorfrontal",
        29: "superiorparietal",
        30: "superiortemporal",
        31: "supramarginal",
        32: "frontalpole",
        33: "temporalpole",
        34: "transversetemporal",
        35: "insula",
    }
    mapping: dict[int, str] = {}
    for code, name in cortical.items():
        mapping[1000 + code] = f"ctx-lh-{name}"
        mapping[2000 + code] = f"ctx-rh-{name}"
    mapping.update(
        {
            16: "Brain-Stem",
            26: "Left-Accumbens-area",
            18: "Left-Amygdala",
            11: "Left-Caudate",
            17: "Left-Hippocampus",
            13: "Left-Pallidum",
            12: "Left-Putamen",
            10: "Left-Thalamus-Proper",
            58: "Right-Accumbens-area",
            54: "Right-Amygdala",
            50: "Right-Caudate",
            53: "Right-Hippocampus",
            52: "Right-Pallidum",
            51: "Right-Putamen",
            49: "Right-Thalamus-Proper",
        }
    )
    return mapping


def ordered_roi_labels(mapping: Mapping[int, str] | None = None) -> list[str]:
    labels = list((mapping or build_freesurfer_lausanne83_mapping()).values())
    return sorted(labels, key=lambda name: (name.startswith("Right-"), name.startswith("Left-"), name))


def standardize_columns(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(array, dtype=float)
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, ddof=1, keepdims=True)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return (values - mean) / scale, mean.reshape(-1), scale.reshape(-1)


def make_lagged_samples(series: np.ndarray, *, tau: int = 1) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(series, dtype=float)
    if values.ndim != 2:
        raise ValueError("series must have shape [time, roi].")
    if tau < 1:
        raise ValueError("tau must be positive.")
    if values.shape[0] <= tau + 1:
        raise ValueError("series is too short for requested tau.")
    return values[:-tau], values[tau:]


def circular_shift_null(series: np.ndarray, *, seed: int) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    if values.ndim != 2:
        raise ValueError("series must have shape [time, roi].")
    rng = np.random.default_rng(seed)
    shifted = np.empty_like(values)
    n_time = values.shape[0]
    for column in range(values.shape[1]):
        offset = int(rng.integers(1, n_time))
        shifted[:, column] = np.roll(values[:, column], offset)
    return shifted


def safe_logdet_psd(matrix: np.ndarray, *, floor: float = 1.0e-12) -> float:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    return float(np.log(np.maximum(eigenvalues, float(floor))).sum())


def gaussian_singleton_source_phi(
    source: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float = 1.0e-6,
) -> dict[str, object]:
    source_array = np.asarray(source, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if source_array.ndim != 2 or target_array.ndim != 2:
        raise ValueError("source and target must be 2D arrays.")
    if source_array.shape[0] != target_array.shape[0]:
        raise ValueError("source and target must share sample axis.")

    source_z, _, _ = standardize_columns(source_array)
    target_z, _, _ = standardize_columns(target_array)
    coefficient, *_ = np.linalg.lstsq(source_z, target_z, rcond=None)
    residual = target_z - source_z @ coefficient
    transition = coefficient.T
    noise_cov = np.cov(residual, rowvar=False, bias=False)
    noise_cov = np.atleast_2d(noise_cov) + float(ridge) * np.eye(target_z.shape[1])
    return gaussian_phi_from_linear_transition(transition, noise_cov, ridge=ridge)


def gaussian_phi_from_linear_transition(
    transition: np.ndarray,
    noise_covariance: np.ndarray,
    *,
    ridge: float = 1.0e-6,
    source_indices: Sequence[int] | None = None,
) -> dict[str, object]:
    transition_array = np.asarray(transition, dtype=float)
    noise = np.asarray(noise_covariance, dtype=float)
    if source_indices is None:
        selected = list(range(transition_array.shape[1]))
    else:
        selected = sorted(set(int(index) for index in source_indices))
    if not selected:
        return {
            "joint_ei": 0.0,
            "singleton_ei_sum": 0.0,
            "raw_phi": 0.0,
            "phi_eid_clipped": 0.0,
            "singleton_ei": np.zeros(0, dtype=float),
        }

    selected_transition = transition_array[:, selected]
    source_cov = np.eye(len(selected), dtype=float)
    target_cov = selected_transition @ source_cov @ selected_transition.T + noise
    target_cov = 0.5 * (target_cov + target_cov.T) + float(ridge) * np.eye(target_cov.shape[0])
    source_target_cov = source_cov @ selected_transition.T
    conditional_source_cov = source_cov - source_target_cov @ np.linalg.pinv(target_cov) @ source_target_cov.T
    conditional_source_cov = (
        0.5 * (conditional_source_cov + conditional_source_cov.T)
        + float(ridge) * np.eye(len(selected))
    )
    joint_ei = 0.5 * (
        safe_logdet_psd(source_cov, floor=ridge)
        - safe_logdet_psd(conditional_source_cov, floor=ridge)
    ) / math.log(2.0)
    singleton_values = []
    for local_index in range(len(selected)):
        prior = source_cov[local_index : local_index + 1, local_index : local_index + 1]
        conditional = conditional_source_cov[local_index : local_index + 1, local_index : local_index + 1]
        singleton_values.append(
            0.5 * (safe_logdet_psd(prior, floor=ridge) - safe_logdet_psd(conditional, floor=ridge)) / math.log(2.0)
        )
    singleton = np.asarray(singleton_values, dtype=float)
    singleton_sum = float(singleton.sum())
    raw_phi = float(joint_ei - singleton_sum)
    return {
        "joint_ei": float(joint_ei),
        "singleton_ei_sum": singleton_sum,
        "raw_phi": raw_phi,
        "phi_eid_clipped": max(0.0, raw_phi),
        "singleton_ei": singleton,
    }


def split_train_validation(
    source: np.ndarray,
    target: np.ndarray,
    *,
    train_fraction: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = int(round(source.shape[0] * float(train_fraction)))
    split = max(2, min(split, source.shape[0] - 2))
    return source[:split], target[:split], source[split:], target[split:]


def fit_ridge_transition(
    source: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float = 1.0,
    ridge: float = 1.0e-6,
) -> dict[str, object]:
    train_x, train_y, val_x, val_y = split_train_validation(source, target)
    train_x_z, x_mean, x_scale = standardize_columns(train_x)
    train_y_z, y_mean, y_scale = standardize_columns(train_y)
    val_x_z = (val_x - x_mean.reshape(1, -1)) / x_scale.reshape(1, -1)
    val_y_z = (val_y - y_mean.reshape(1, -1)) / y_scale.reshape(1, -1)

    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(train_x_z, train_y_z)
    train_pred = model.predict(train_x_z)
    val_pred = model.predict(val_x_z)
    residual = train_y_z - train_pred
    noise_cov = np.cov(residual, rowvar=False, bias=False)
    noise_cov = np.atleast_2d(noise_cov) + float(ridge) * np.eye(train_y_z.shape[1])
    metrics = prediction_metrics(val_y_z, val_pred, val_x_z)
    phi = gaussian_phi_from_linear_transition(np.asarray(model.coef_, dtype=float), noise_cov, ridge=ridge)
    return {
        "model": model,
        "transition": np.asarray(model.coef_, dtype=float),
        "noise_covariance": noise_cov,
        "metrics": metrics,
        "phi": phi,
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
    }


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray, persistence: np.ndarray) -> dict[str, float]:
    diff = np.asarray(y_true) - np.asarray(y_pred)
    persist_diff = np.asarray(y_true) - np.asarray(persistence)
    rmse = float(np.sqrt(np.mean(diff**2)))
    persistence_rmse = float(np.sqrt(np.mean(persist_diff**2)))
    corr = float(np.corrcoef(np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1))[0, 1])
    if not np.isfinite(corr):
        corr = 0.0
    return {
        "rmse": rmse,
        "persistence_rmse": persistence_rmse,
        "skill_ratio": float(rmse / max(persistence_rmse, 1.0e-12)),
        "corr": corr,
    }


def fit_mlp_transition(
    source: np.ndarray,
    target: np.ndarray,
    *,
    hidden_dim: int = 128,
    epochs: int = 80,
    seed: int = 0,
    learning_rate: float = 1.0e-3,
) -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on local runtime
        raise RuntimeError("torch is required for --fit-mlp.") from exc

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    train_x, train_y, val_x, val_y = split_train_validation(source, target)
    train_x_z, x_mean, x_scale = standardize_columns(train_x)
    train_y_z, y_mean, y_scale = standardize_columns(train_y)
    val_x_z = (val_x - x_mean.reshape(1, -1)) / x_scale.reshape(1, -1)
    val_y_z = (val_y - y_mean.reshape(1, -1)) / y_scale.reshape(1, -1)

    net = torch.nn.Sequential(
        torch.nn.Linear(train_x_z.shape[1], int(hidden_dim)),
        torch.nn.SiLU(),
        torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
        torch.nn.SiLU(),
        torch.nn.Linear(int(hidden_dim), train_y_z.shape[1]),
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=float(learning_rate), weight_decay=1.0e-5)
    x_tensor = torch.tensor(train_x_z, dtype=torch.float32)
    y_tensor = torch.tensor(train_y_z, dtype=torch.float32)
    vx_tensor = torch.tensor(val_x_z, dtype=torch.float32)
    vy_tensor = torch.tensor(val_y_z, dtype=torch.float32)
    best_state = None
    best_val = float("inf")
    patience = 12
    stale = 0
    for _ in range(int(epochs)):
        net.train()
        optimizer.zero_grad()
        loss = torch.mean((net(x_tensor) - y_tensor) ** 2)
        loss.backward()
        optimizer.step()
        net.eval()
        with torch.no_grad():
            val_loss = float(torch.mean((net(vx_tensor) - vy_tensor) ** 2).cpu())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().clone() for key, value in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        val_pred = net(vx_tensor).cpu().numpy()
    return {
        "metrics": prediction_metrics(val_y_z, val_pred, val_x_z),
        "best_val_loss": best_val,
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
    }


def infer_display_module(label: str) -> str:
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


def module_indices_from_labels(labels: Sequence[str]) -> dict[str, list[int]]:
    grouped = {name: [] for name in MODULE_ORDER}
    for index, label in enumerate(labels):
        grouped[infer_display_module(str(label))].append(index)
    return {name: indices for name, indices in grouped.items() if indices}


def subset_phi_raw(
    subset: tuple[str, ...],
    ei_table: Mapping[tuple[str, ...], float],
    singleton_ei: Mapping[str, float],
) -> float:
    return float(ei_table[tuple(subset)] - sum(float(singleton_ei[name]) for name in subset))


def nontrivial_bipartitions(subset: Sequence[str]) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    ordered = tuple(str(name) for name in subset)
    if len(ordered) <= 1:
        return []
    first = ordered[0]
    rest = ordered[1:]
    full = set(ordered)
    splits: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for mask in range(1 << len(rest)):
        left = {first}
        for index, name in enumerate(rest):
            if mask & (1 << index):
                left.add(name)
        if len(left) == len(ordered):
            continue
        right = full - left
        splits.append((tuple(name for name in ordered if name in left), tuple(name for name in ordered if name in right)))
    return splits


def greedy_phi_atoms(
    subset: tuple[str, ...],
    ei_table: Mapping[tuple[str, ...], float],
    *,
    eps: float = 1.0e-6,
    split_tolerance: float = 1.0e-4,
    depth: int = 0,
    singleton_ei: Mapping[str, float] | None = None,
) -> list[GreedyAtom]:
    if singleton_ei is None:
        singleton_ei = {name: float(ei_table[(name,)]) for name in subset}
    block_phi = subset_phi_raw(subset, ei_table, singleton_ei)
    if len(subset) <= 1 or block_phi <= float(eps):
        return []

    best: tuple[float, float, tuple[str, ...], tuple[str, ...]] | None = None
    for left, right in nontrivial_bipartitions(subset):
        left_phi = subset_phi_raw(left, ei_table, singleton_ei)
        right_phi = subset_phi_raw(right, ei_table, singleton_ei)
        residual = block_phi - left_phi - right_phi
        if residual < -float(split_tolerance):
            continue
        captured = left_phi + right_phi
        if best is None or captured > best[0] or (np.isclose(captured, best[0]) and residual < best[1]):
            best = (captured, residual, left, right)

    if best is None or best[0] <= float(eps):
        return [GreedyAtom(subset, max(0.0, block_phi), "terminal", int(depth))]

    _, residual, left, right = best
    atoms: list[GreedyAtom] = []
    if residual > float(eps):
        atoms.append(GreedyAtom(subset, max(0.0, residual), "split_residual", int(depth)))
    atoms.extend(greedy_phi_atoms(left, ei_table, eps=eps, split_tolerance=split_tolerance, depth=depth + 1, singleton_ei=singleton_ei))
    atoms.extend(greedy_phi_atoms(right, ei_table, eps=eps, split_tolerance=split_tolerance, depth=depth + 1, singleton_ei=singleton_ei))
    return atoms


def ei_for_source_indices(transition: np.ndarray, noise: np.ndarray, selected: Sequence[int], *, ridge: float) -> float:
    trans = np.asarray(transition, dtype=float)
    selected_set = set(int(index) for index in selected)
    complement = [index for index in range(trans.shape[1]) if index not in selected_set]
    full_cov = trans @ trans.T + noise
    conditional = noise.copy()
    if complement:
        conditional = conditional + trans[:, complement] @ trans[:, complement].T
    return float(0.5 * (safe_logdet_psd(full_cov, floor=ridge) - safe_logdet_psd(conditional, floor=ridge)) / math.log(2.0))


def module_ei_table(
    transition: np.ndarray,
    noise: np.ndarray,
    module_indices: Mapping[str, Sequence[int]],
    *,
    ridge: float,
) -> dict[tuple[str, ...], float]:
    import itertools

    names = tuple(name for name in MODULE_ORDER if name in module_indices)
    table: dict[tuple[str, ...], float] = {}
    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(names, size):
            indices = sorted({idx for name in subset for idx in module_indices[name]})
            table[tuple(subset)] = max(0.0, ei_for_source_indices(transition, noise, indices, ridge=ridge))
    return table


def roi_leave_one_out_burden(
    transition: np.ndarray,
    noise: np.ndarray,
    labels: Sequence[str],
    *,
    ridge: float,
) -> list[dict[str, object]]:
    all_indices = list(range(len(labels)))
    singleton_ei = {
        index: ei_for_source_indices(transition, noise, [index], ridge=ridge)
        for index in all_indices
    }
    full_phi = (
        ei_for_source_indices(transition, noise, all_indices, ridge=ridge)
        - sum(singleton_ei.values())
    )
    rows = []
    for index, label in enumerate(labels):
        selected = [other for other in all_indices if other != index]
        phi_without = (
            ei_for_source_indices(transition, noise, selected, ridge=ridge)
            - sum(singleton_ei[other] for other in selected)
        )
        rows.append(
            {
                "roi": str(label),
                "index": int(index),
                "module": infer_display_module(str(label)),
                "burden": float(max(0.0, full_phi - phi_without)),
                "phi_without": float(phi_without),
            }
        )
    return sorted(rows, key=lambda row: float(row["burden"]), reverse=True)


def check_neuro_dependencies() -> dict[str, str]:
    missing: dict[str, str] = {}
    for name in ("nibabel", "nilearn"):
        try:
            __import__(name)
        except Exception as exc:  # pragma: no cover - depends on local runtime
            missing[name] = f"{type(exc).__name__}: {exc}"
    return missing


def find_hcp_run_paths(hcp_root: Path, subject: str, run: str) -> HcpRunPaths:
    subject_roots = [hcp_root / "HCP_1200" / subject, hcp_root / subject]
    functional_names = [
        f"MNINonLinear/Results/rfMRI_{run}/rfMRI_{run}.nii.gz",
        f"MNINonLinear/Results/rfMRI_{run}/rfMRI_{run}_hp2000_clean.nii.gz",
    ]
    motion_names = [
        f"MNINonLinear/Results/rfMRI_{run}/Movement_RelativeRMS.txt",
        f"MNINonLinear/Results/rfMRI_{run}/Movement_Regressors.txt",
    ]
    for root in subject_roots:
        aparc = root / "MNINonLinear" / "aparc+aseg.nii.gz"
        for functional_name in functional_names:
            functional = root / functional_name
            if functional.exists() and aparc.exists():
                motion = next((root / name for name in motion_names if (root / name).exists()), None)
                return HcpRunPaths(functional=functional, aparc_aseg=aparc, motion=motion)
    raise FileNotFoundError(
        f"Missing HCP files for subject={subject} run={run} under {hcp_root}. "
        "Expected MNINonLinear/Results/rfMRI_<RUN>/rfMRI_<RUN>.nii.gz and MNINonLinear/aparc+aseg.nii.gz."
    )


def download_hcp_subject_run(hcp_root: Path, subject: str, run: str) -> None:
    aws = shutil.which("aws")
    if aws is None:
        raise RuntimeError("aws CLI is required for --download but was not found on PATH.")
    targets = [
        (f"{HCP_S3_ROOT}/{subject}/MNINonLinear/aparc+aseg.nii.gz", hcp_root / "HCP_1200" / subject / "MNINonLinear" / "aparc+aseg.nii.gz"),
        (f"{HCP_S3_ROOT}/{subject}/MNINonLinear/Results/rfMRI_{run}/rfMRI_{run}.nii.gz", hcp_root / "HCP_1200" / subject / "MNINonLinear" / "Results" / f"rfMRI_{run}" / f"rfMRI_{run}.nii.gz"),
        (f"{HCP_S3_ROOT}/{subject}/MNINonLinear/Results/rfMRI_{run}/Movement_RelativeRMS.txt", hcp_root / "HCP_1200" / subject / "MNINonLinear" / "Results" / f"rfMRI_{run}" / "Movement_RelativeRMS.txt"),
    ]
    for source, destination in targets:
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [aws, "s3", "cp", source, str(destination)]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"HCP download failed: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )


def extract_roi_timeseries(paths: HcpRunPaths, *, labels: Sequence[str]) -> tuple[np.ndarray, dict[str, object]]:
    try:
        import nibabel as nib
        from nibabel.processing import resample_from_to
    except Exception as exc:  # pragma: no cover - depends on local runtime
        raise RuntimeError("nibabel is required to extract ROI time series from HCP NIfTI files.") from exc

    mapping = build_freesurfer_lausanne83_mapping()
    code_by_label = {label: code for code, label in mapping.items()}
    functional_img = nib.load(str(paths.functional))
    aparc_img = nib.load(str(paths.aparc_aseg))
    if functional_img.ndim != 4:
        raise ValueError(f"Functional image must be 4D: {paths.functional}")
    target_grid = (functional_img.shape[:3], functional_img.affine)
    if aparc_img.shape[:3] != functional_img.shape[:3] or not np.allclose(aparc_img.affine, functional_img.affine):
        aparc_img = resample_from_to(aparc_img, target_grid, order=0)
    roi_labels = np.rint(aparc_img.get_fdata(dtype=np.float32)).astype(np.int32)
    data = functional_img.get_fdata(dtype=np.float32)
    time_count = int(data.shape[3])
    flat_data = data.reshape(-1, time_count)
    flat_labels = roi_labels.reshape(-1)

    series = np.empty((time_count, len(labels)), dtype=float)
    roi_qc = []
    for column, label in enumerate(labels):
        code = code_by_label[str(label)]
        voxels = np.flatnonzero(flat_labels == code)
        if voxels.size == 0:
            series[:, column] = np.nan
        else:
            series[:, column] = flat_data[voxels].mean(axis=0)
        roi_qc.append({"roi": str(label), "freesurfer_code": int(code), "voxel_count": int(voxels.size)})
    missing = [row["roi"] for row in roi_qc if int(row["voxel_count"]) == 0]
    if missing:
        raise ValueError(f"Empty ROIs after aparc+aseg extraction: {missing}")
    series, _, _ = standardize_columns(series)
    motion_summary = summarize_motion(paths.motion)
    return series, {
        "time_count": time_count,
        "roi_count": len(labels),
        "roi_qc": roi_qc,
        "motion": motion_summary,
        "functional": str(paths.functional),
        "aparc_aseg": str(paths.aparc_aseg),
    }


def summarize_motion(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {"available": False}
    values = np.loadtxt(path)
    return {
        "available": True,
        "path": str(path),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def save_roi_timeseries(
    output_dir: Path,
    *,
    subject: str,
    run: str,
    series: np.ndarray,
    labels: Sequence[str],
    metadata: Mapping[str, object],
) -> Path:
    target = output_dir / "roi_timeseries" / f"sub-{subject}_{run}_lausanne83_timeseries.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        series=np.asarray(series, dtype=float),
        labels=np.asarray(labels, dtype=object),
        metadata=json.dumps(dict(metadata), sort_keys=True),
    )
    return target


def synthetic_timeseries(*, n_time: int, n_roi: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=0.35, size=(n_time, n_roi))
    series = np.zeros((n_time, n_roi), dtype=float)
    transition = 0.35 * np.eye(n_roi)
    for index in range(n_roi - 1):
        transition[index + 1, index] += 0.16
    transition[:8, 20:28] += 0.035
    transition[20:28, :8] += 0.035
    for t in range(1, n_time):
        series[t] = series[t - 1] @ transition.T + noise[t]
    return standardize_columns(series)[0]


def analyze_series(
    series: np.ndarray,
    *,
    labels: Sequence[str],
    subject: str,
    run: str,
    null_reps: int,
    seed: int,
    ridge_alpha: float,
    ridge: float,
    fit_mlp: bool,
) -> dict[str, object]:
    source, target = make_lagged_samples(series, tau=1)
    ridge_result = fit_ridge_transition(source, target, alpha=ridge_alpha, ridge=ridge)
    null_rows = []
    for rep in range(int(null_reps)):
        null_series = circular_shift_null(series, seed=int(seed) + rep + 1000)
        null_source, null_target = make_lagged_samples(null_series, tau=1)
        null_fit = fit_ridge_transition(null_source, null_target, alpha=ridge_alpha, ridge=ridge)
        null_rows.append(
            {
                "rep": rep,
                "raw_phi": float(null_fit["phi"]["raw_phi"]),
                "phi_eid_clipped": float(null_fit["phi"]["phi_eid_clipped"]),
                "whole_ei": float(null_fit["phi"]["joint_ei"]),
                "singleton_ei_sum": float(null_fit["phi"]["singleton_ei_sum"]),
            }
        )
    observed_phi = float(ridge_result["phi"]["raw_phi"])
    p_value = (1.0 + sum(1 for row in null_rows if float(row["raw_phi"]) >= observed_phi)) / (1.0 + len(null_rows))

    modules = module_indices_from_labels(labels)
    table = module_ei_table(
        np.asarray(ridge_result["transition"], dtype=float),
        np.asarray(ridge_result["noise_covariance"], dtype=float),
        modules,
        ridge=ridge,
    )
    module_names = tuple(name for name in MODULE_ORDER if name in modules)
    singleton = {name: float(table[(name,)]) for name in module_names}
    atoms = greedy_phi_atoms(module_names, table, singleton_ei=singleton)
    burden = roi_leave_one_out_burden(
        np.asarray(ridge_result["transition"], dtype=float),
        np.asarray(ridge_result["noise_covariance"], dtype=float),
        labels,
        ridge=ridge,
    )
    mlp_result = None
    if fit_mlp:
        mlp_result = fit_mlp_transition(source, target, seed=seed)

    return {
        "subject": str(subject),
        "run": str(run),
        "sample_count": int(source.shape[0]),
        "ridge_metrics": ridge_result["metrics"],
        "ridge_phi": {
            "whole_ei": float(ridge_result["phi"]["joint_ei"]),
            "singleton_ei_sum": float(ridge_result["phi"]["singleton_ei_sum"]),
            "raw_phi": observed_phi,
            "phi_eid_clipped": float(ridge_result["phi"]["phi_eid_clipped"]),
        },
        "null": null_rows,
        "empirical_p_value": float(p_value),
        "module_atoms": [
            {
                "sources": list(atom.sources),
                "label": "+".join(atom.sources),
                "value": float(atom.value),
                "kind": atom.kind,
                "depth": int(atom.depth),
                "order": len(atom.sources),
            }
            for atom in atoms
        ],
        "roi_burden": burden,
        "module_indices": {name: list(map(int, indices)) for name, indices in modules.items()},
        "mlp_metrics": None if mlp_result is None else mlp_result["metrics"],
    }


def plot_null_comparison(rows: Sequence[Mapping[str, object]], figure_base: Path) -> None:
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8})
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    observed = np.asarray([float(row["ridge_phi"]["raw_phi"]) for row in rows], dtype=float)
    null_mean = np.asarray([float(np.mean([n["raw_phi"] for n in row["null"]])) for row in rows], dtype=float)
    p_values = np.asarray([float(row["empirical_p_value"]) for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), constrained_layout=True)
    x = np.arange(len(rows))
    axes[0].plot(x, null_mean, "o", color="#7F7F7F", label="Null mean")
    axes[0].plot(x, observed, "o", color="#2F7D5A", label="Observed")
    for idx in x:
        axes[0].plot([idx, idx], [null_mean[idx], observed[idx]], color="#B8B8B8", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(row["subject"]) for row in rows], rotation=45, ha="right")
    axes[0].set_ylabel("Raw PhiEID (bits)")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    axes[1].hist(p_values, bins=np.linspace(0, 1, 11), color="#5B8DB8", edgecolor="white")
    axes[1].axvline(0.05, color="#D55E00", linewidth=1.0)
    axes[1].set_xlabel("Empirical p-value")
    axes[1].set_ylabel("Runs")
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(str(figure_base) + suffix, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_decomposition(rows: Sequence[Mapping[str, object]], figure_base: Path) -> None:
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8})
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    atom_totals: dict[str, float] = {}
    roi_totals: dict[str, tuple[str, float]] = {}
    for row in rows:
        for atom in row["module_atoms"]:
            label = str(atom["label"])
            atom_totals[label] = atom_totals.get(label, 0.0) + float(atom["value"])
        for roi_row in row["roi_burden"][:20]:
            roi = str(roi_row["roi"])
            module = str(roi_row["module"])
            previous = roi_totals.get(roi, (module, 0.0))[1]
            roi_totals[roi] = (module, previous + float(roi_row["burden"]))
    top_atoms = sorted(atom_totals.items(), key=lambda item: item[1], reverse=True)[:12]
    top_rois = sorted(roi_totals.items(), key=lambda item: item[1][1], reverse=True)[:15]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    atom_labels = [item[0] for item in top_atoms]
    atom_values = [item[1] / max(len(rows), 1) for item in top_atoms]
    axes[0].barh(np.arange(len(atom_labels)), atom_values, color="#2F7D5A")
    axes[0].set_yticks(np.arange(len(atom_labels)))
    axes[0].set_yticklabels(atom_labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean atom PhiEID")
    roi_labels = [item[0] for item in top_rois]
    roi_values = [item[1][1] / max(len(rows), 1) for item in top_rois]
    roi_colors = [MODULE_COLORS.get(item[1][0], "#7F7F7F") for item in top_rois]
    axes[1].barh(np.arange(len(roi_labels)), roi_values, color=roi_colors)
    axes[1].set_yticks(np.arange(len(roi_labels)))
    axes[1].set_yticklabels(roi_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean leave-one-out burden")
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(str(figure_base) + suffix, dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(
    report_path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    synthetic: bool,
    null_reps: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    observed = np.asarray([float(row["ridge_phi"]["raw_phi"]) for row in rows], dtype=float)
    null_means = np.asarray([float(np.mean([n["raw_phi"] for n in row["null"]])) for row in rows], dtype=float)
    p_values = np.asarray([float(row["empirical_p_value"]) for row in rows], dtype=float)
    top_roi: dict[str, float] = {}
    top_atom: dict[str, float] = {}
    for row in rows:
        for roi_row in row["roi_burden"][:10]:
            top_roi[str(roi_row["roi"])] = top_roi.get(str(roi_row["roi"]), 0.0) + float(roi_row["burden"])
        for atom in row["module_atoms"]:
            top_atom[str(atom["label"])] = top_atom.get(str(atom["label"]), 0.0) + float(atom["value"])
    roi_lines = "\n".join(
        f"- {name}: {value / max(len(rows), 1):.4f}"
        for name, value in sorted(top_roi.items(), key=lambda item: item[1], reverse=True)[:10]
    )
    atom_lines = "\n".join(
        f"- {name}: {value / max(len(rows), 1):.4f}"
        for name, value in sorted(top_atom.items(), key=lambda item: item[1], reverse=True)[:10]
    )
    report_path.write_text(
        "\n".join(
            [
                "# HCP Lausanne-83 PhiEID Pilot",
                "",
                f"- Synthetic smoke mode: `{synthetic}`",
                f"- Runs analyzed: `{len(rows)}`",
                f"- Null repetitions per run: `{null_reps}`",
                "- Main estimator: Gaussian log-det whole-state screening.",
                "- TM full-dimensional estimation is computationally prohibitive for this pilot; TM should be limited to low-dimensional module checks.",
                "",
                "## Null comparison",
                "",
                f"- Mean observed raw PhiEID: `{float(np.mean(observed)):.6f}`",
                f"- Mean null raw PhiEID: `{float(np.mean(null_means)):.6f}`",
                f"- Median empirical p-value: `{float(np.median(p_values)):.6f}`",
                "",
                "## Top module atoms",
                "",
                atom_lines or "- No positive module atoms.",
                "",
                "## Top ROI burden candidates",
                "",
                roi_lines or "- No positive ROI burden.",
                "",
                "## Limits",
                "",
                "- This is a pilot workflow, not a full S1200 inference.",
                "- ROI burden is a leave-one-out candidate score, not an exact exhaustive 83D high-order atom.",
                "- 2017 S1200 paths are not mixed with 2025 BALSA outputs.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    labels = ordered_roi_labels()
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows = []
    roi_files = []
    subjects = [part.strip() for part in str(args.subjects).split(",") if part.strip()]
    runs = [part.strip() for part in str(args.runs).split(",") if part.strip()]
    if not args.synthetic:
        missing = check_neuro_dependencies()
        if missing:
            details = "; ".join(f"{name} ({reason})" for name, reason in sorted(missing.items()))
            raise RuntimeError(f"Missing neuroimaging dependencies for real HCP extraction: {details}.")
    for subject_index, subject in enumerate(subjects):
        for run in runs:
            if args.synthetic:
                series = synthetic_timeseries(n_time=int(args.synthetic_timepoints), n_roi=len(labels), seed=int(args.seed) + subject_index)
                metadata = {"synthetic": True, "time_count": int(series.shape[0]), "roi_count": int(series.shape[1])}
            else:
                if args.download:
                    download_hcp_subject_run(Path(args.hcp_root), subject, run)
                paths = find_hcp_run_paths(Path(args.hcp_root), subject, run)
                series, metadata = extract_roi_timeseries(paths, labels=labels)
            roi_path = save_roi_timeseries(output_dir, subject=subject, run=run, series=series, labels=labels, metadata=metadata)
            roi_files.append(str(roi_path))
            rows.append(
                analyze_series(
                    series,
                    labels=labels,
                    subject=subject,
                    run=run,
                    null_reps=int(args.null_reps),
                    seed=int(args.seed) + len(rows) * 100,
                    ridge_alpha=float(args.ridge_alpha),
                    ridge=float(args.ridge),
                    fit_mlp=bool(args.fit_mlp),
                )
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary = {
        "synthetic": bool(args.synthetic),
        "subjects": subjects,
        "runs": runs,
        "labels": labels,
        "roi_files": roi_files,
        "rows": rows,
        "config": {
            "null_reps": int(args.null_reps),
            "ridge_alpha": float(args.ridge_alpha),
            "ridge": float(args.ridge),
            "fit_mlp": bool(args.fit_mlp),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    plot_null_comparison(rows, Path(args.null_figure_base))
    plot_decomposition(rows, Path(args.decomposition_figure_base))
    write_report(Path(args.report), rows, synthetic=bool(args.synthetic), null_reps=int(args.null_reps))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a lightweight HCP-YA Lausanne-83 PhiEID pilot.")
    parser.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    parser.add_argument("--hcp-root", default=str(DEFAULT_HCP_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--null-figure-base", default=str(DEFAULT_FIG_NULL))
    parser.add_argument("--decomposition-figure-base", default=str(DEFAULT_FIG_DECOMP))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-timepoints", type=int, default=220)
    parser.add_argument("--null-reps", type=int, default=20)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--fit-mlp", action="store_true")
    parser.add_argument("--seed", type=int, default=20260707)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_pipeline(args)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
