from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNICM_SRC = ROOT / "data" / "UniCM-checkpoint" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODE_NAMES = {
    "nino": 0,
    "NPMM": 1,
    "SPMM": 2,
    "IOB": 3,
    "IOD": 4,
    "SIOD": 5,
    "TNA": 6,
    "nino12": 7,
    "nino3": 8,
    "nino4": 9,
    "WWV": 10,
}

HISTORY_LENGTH = 12
PREDICTION_LENGTH = 24


@dataclass(frozen=True)
class Relation:
    key: str
    left: str
    right: str
    target: str
    label: str

    @property
    def left_index(self) -> int:
        return MODE_NAMES[self.left]

    @property
    def right_index(self) -> int:
        return MODE_NAMES[self.right]

    @property
    def target_index(self) -> int:
        return MODE_NAMES[self.target]


RELATIONS: dict[str, Relation] = {
    "NPMM_WWV_to_nino": Relation("NPMM_WWV_to_nino", "NPMM", "WWV", "nino", "NPMM + WWV -> nino"),
    "NPMM_WWV_to_nino3": Relation("NPMM_WWV_to_nino3", "NPMM", "WWV", "nino3", "NPMM + WWV -> nino3"),
    "TNA_nino_to_nino": Relation("TNA_nino_to_nino", "TNA", "nino", "nino", "TNA + nino -> nino"),
    "IOD_IOB_to_nino": Relation("IOD_IOB_to_nino", "IOD", "IOB", "nino", "IOD + IOB -> nino"),
    "SIOD_IOB_to_IOB": Relation("SIOD_IOB_to_IOB", "SIOD", "IOB", "IOB", "SIOD + IOB -> IOB"),
    "SIOD_IOB_to_IOD": Relation("SIOD_IOB_to_IOD", "SIOD", "IOB", "IOD", "SIOD + IOB -> IOD"),
}


def group_relations_by_sources(relations: Sequence[Relation]) -> dict[tuple[str, str], list[Relation]]:
    groups: dict[tuple[str, str], list[Relation]] = {}
    for relation in relations:
        groups.setdefault((relation.left, relation.right), []).append(relation)
    return groups


def parse_leads(values: Sequence[str] | None) -> list[int]:
    if not values:
        return list(range(1, 25))
    leads: list[int] = []
    for raw in values:
        if ".." in raw:
            start, end = raw.split("..", 1)
            leads.extend(range(int(start), int(end) + 1))
        else:
            leads.append(int(raw))
    unique = sorted(set(leads))
    for lead in unique:
        if lead < 1 or lead > 24:
            raise ValueError(f"Lead must be in [1, 24], got {lead}.")
    return unique


def resolve_checkpoint_paths(checkpoint_root: Path, seeds: Sequence[int]) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for seed in seeds:
        matches = sorted(checkpoint_root.glob(f"*Seed{int(seed)}/model_save/model_best.pkl"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one checkpoint for seed {seed} under {checkpoint_root}, found {len(matches)}."
            )
        paths[int(seed)] = matches[0]
    return paths


def make_unicm_args(device: str, *, dropout: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        his_len=12,
        pred_len=24,
        input_channal=5,
        patch_size=[2, 2],
        emb_spatial_size=216,
        d_size=256,
        nheads=4,
        dim_feedforward=512,
        dropout=float(dropout),
        num_encoder_layers=4,
        num_decoder_layers=4,
        mode_interaction="1",
        t20d_mode=1,
        val_relative=[(0, 1, 0, 1)] * 10,
        device=device,
        autoregressive=0,
    )


def install_unicm_import_shims() -> None:
    """Install optional dependency shims required by UniCM imports in lean environments."""

    if "pynvml" in sys.modules:
        return
    module = types.ModuleType("pynvml")
    module.nvmlInit = lambda: None
    module.nvmlShutdown = lambda: None
    module.nvmlDeviceGetCount = lambda: 0
    module.nvmlDeviceGetHandleByIndex = lambda index: None
    module.nvmlDeviceGetMemoryInfo = lambda handle: SimpleNamespace(total=0, used=0, free=0)
    sys.modules["pynvml"] = module


def load_unicm_model(checkpoint_path: Path, device: str):
    if str(UNICM_SRC) not in sys.path:
        sys.path.insert(0, str(UNICM_SRC))
    install_unicm_import_shims()
    import torch
    from models import UniCM

    model = UniCM(make_unicm_args(device)).to(device)
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def sample_full_history_mode_inputs(
    *,
    n_samples: int,
    intervention_bound: float,
    seed: int,
) -> np.ndarray:
    """Sample the full UniCM mode-history input from a bounded maximum-entropy box."""

    if int(n_samples) < 1:
        raise ValueError("n_samples must be positive.")
    if float(intervention_bound) <= 0:
        raise ValueError("intervention_bound must be positive.")
    rng = np.random.default_rng(int(seed))
    return rng.uniform(
        -float(intervention_bound),
        float(intervention_bound),
        size=(int(n_samples), HISTORY_LENGTH, len(MODE_NAMES)),
    ).astype(np.float32)


def _validate_full_history_mode_inputs(history_modes: np.ndarray) -> np.ndarray:
    array = np.asarray(history_modes, dtype=np.float32)
    expected_shape_tail = (HISTORY_LENGTH, len(MODE_NAMES))
    if array.ndim != 3 or tuple(array.shape[1:]) != expected_shape_tail:
        raise ValueError(
            "history_modes must have shape "
            f"(n_samples, {HISTORY_LENGTH}, {len(MODE_NAMES)}), got {tuple(array.shape)}."
        )
    if not np.isfinite(array).all():
        raise ValueError("history_modes contains non-finite values.")
    return array


def make_full_history_mode_tensor(history_modes: np.ndarray) -> np.ndarray:
    history = _validate_full_history_mode_inputs(history_modes)
    n_samples = int(history.shape[0])
    tensor = np.zeros(
        (n_samples, HISTORY_LENGTH + PREDICTION_LENGTH, 1, len(MODE_NAMES), 1),
        dtype=np.float32,
    )
    tensor[:, :HISTORY_LENGTH, 0, :, 0] = history
    return tensor


def extract_relation_history_sources(history_modes: np.ndarray, relation: Relation) -> tuple[np.ndarray, np.ndarray]:
    history = _validate_full_history_mode_inputs(history_modes)
    left = history[:, :, relation.left_index].astype(float)
    right = history[:, :, relation.right_index].astype(float)
    return left, right


def make_mode_tensor(
    left_source: np.ndarray,
    right_source: np.ndarray,
    relation: Relation,
    *,
    history_months: int,
) -> np.ndarray:
    if left_source.shape != right_source.shape:
        raise ValueError("left_source and right_source must have matching shapes.")
    if left_source.ndim != 2 or left_source.shape[1] != 1:
        raise ValueError("source arrays must have shape (n_samples, 1).")
    if history_months < 1 or history_months > 12:
        raise ValueError("history_months must be in [1, 12].")

    n_samples = left_source.shape[0]
    tensor = np.zeros((n_samples, 36, 1, len(MODE_NAMES), 1), dtype=np.float32)
    start = 12 - int(history_months)
    tensor[:, start:12, 0, relation.left_index, 0] = left_source.astype(np.float32)
    tensor[:, start:12, 0, relation.right_index, 0] = right_source.astype(np.float32)
    return tensor


def make_timestamps(n_samples: int, *, start_month: int = 0) -> np.ndarray:
    months = (np.arange(HISTORY_LENGTH + PREDICTION_LENGTH, dtype=np.int64) + int(start_month)) % 12
    return np.tile(months[None, :], (int(n_samples), 1))


def predict_modeformer_all_modes(
    model,
    left_source: np.ndarray,
    right_source: np.ndarray,
    relation: Relation,
    *,
    device: str,
    batch_size: int,
    history_months: int,
    start_month: int,
) -> np.ndarray:
    import torch

    predictions: list[np.ndarray] = []
    n_samples = int(left_source.shape[0])
    source_start = 12 - int(history_months)

    with torch.no_grad():
        for start in range(0, n_samples, int(batch_size)):
            end = min(n_samples, start + int(batch_size))
            current_batch = end - start
            predictor = torch.zeros((current_batch, 36, 1, len(MODE_NAMES), 1), device=device, dtype=torch.float32)
            left_values = torch.tensor(left_source[start:end, 0].tolist(), device=device, dtype=torch.float32)
            right_values = torch.tensor(right_source[start:end, 0].tolist(), device=device, dtype=torch.float32)
            predictor[:, source_start:12, 0, relation.left_index, 0] = left_values[:, None]
            predictor[:, source_start:12, 0, relation.right_index, 0] = right_values[:, None]
            months = (torch.arange(36, device=device, dtype=torch.int64) + int(start_month)) % 12
            time = months.unsqueeze(0).repeat(current_batch, 1)
            out, _, _ = model.forward_sep(
                predictor,
                time,
                model.encoder_mode,
                model.decoder_mode,
                model.linear_output_mode,
                model.predictor_emb_mode,
                model.predictand_emb_mode,
                1,
                [1, 1],
                train=False,
            )
            mode_tensor = out.squeeze(-1).squeeze(2).detach().cpu()
            predictions.append(np.asarray(mode_tensor.tolist(), dtype=float))
    return np.concatenate(predictions, axis=0)


def predict_modeformer_all_modes_from_history(
    model,
    history_modes: np.ndarray,
    *,
    device: str,
    batch_size: int,
    start_month: int,
) -> np.ndarray:
    import torch

    history = _validate_full_history_mode_inputs(history_modes)
    predictions: list[np.ndarray] = []
    n_samples = int(history.shape[0])

    with torch.no_grad():
        for start in range(0, n_samples, int(batch_size)):
            end = min(n_samples, start + int(batch_size))
            predictor_np = make_full_history_mode_tensor(history[start:end])
            predictor = torch.tensor(predictor_np, device=device, dtype=torch.float32)
            months = (torch.arange(HISTORY_LENGTH + PREDICTION_LENGTH, device=device, dtype=torch.int64) + int(start_month)) % 12
            time = months.unsqueeze(0).repeat(end - start, 1)
            out, _, _ = model.forward_sep(
                predictor,
                time,
                model.encoder_mode,
                model.decoder_mode,
                model.linear_output_mode,
                model.predictor_emb_mode,
                model.predictand_emb_mode,
                1,
                [1, 1],
                train=False,
            )
            mode_tensor = out.squeeze(-1).squeeze(2).detach().cpu()
            predictions.append(np.asarray(mode_tensor.tolist(), dtype=float))
    return np.concatenate(predictions, axis=0)


def predict_modeformer_targets(
    model,
    left_source: np.ndarray,
    right_source: np.ndarray,
    relation: Relation,
    *,
    device: str,
    batch_size: int,
    history_months: int,
    start_month: int,
) -> np.ndarray:
    predictions = predict_modeformer_all_modes(
        model,
        left_source,
        right_source,
        relation,
        device=device,
        batch_size=batch_size,
        history_months=history_months,
        start_month=start_month,
    )
    return predictions[:, :, relation.target_index]


def _gaussian_logdet_bias_correction(dimension: int, sample_size: int) -> float:
    if sample_size <= dimension:
        return 0.0
    from scipy.special import digamma

    nu = sample_size - 1
    return float(
        sum(digamma((nu + 1 - index) / 2.0) for index in range(1, dimension + 1))
        + dimension * np.log(2.0)
        - dimension * np.log(nu)
    )


def _mean_gaussian_log_prob_terms(samples: np.ndarray, *, jitter: float = 1e-6) -> tuple[float, np.ndarray, np.ndarray]:
    array = np.asarray(samples, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError("samples must be a 2D array with at least two rows.")
    centered = array - array.mean(axis=0, keepdims=True)
    sample_size, dimension = centered.shape
    covariance = np.cov(array, rowvar=False, bias=False)
    covariance = np.atleast_2d(covariance)
    covariance += float(jitter) * np.eye(covariance.shape[0], dtype=float)
    biased_covariance = centered.T @ centered / float(sample_size)
    cholesky = np.linalg.cholesky(covariance)
    log_det_cholesky = float(np.log(np.diag(cholesky)).sum())
    mean_quadratic = float(np.trace(np.linalg.solve(covariance, biased_covariance)))
    mean_log_prob = -0.5 * (dimension * np.log(2.0 * np.pi) + mean_quadratic) - log_det_cholesky
    return mean_log_prob, covariance, biased_covariance


def _mean_gaussian_log_prob_from_covariance(covariance: np.ndarray, biased_covariance: np.ndarray) -> float:
    covariance = np.asarray(covariance, dtype=float)
    biased_covariance = np.asarray(biased_covariance, dtype=float)
    dimension = int(covariance.shape[0])
    cholesky = np.linalg.cholesky(covariance)
    log_det_cholesky = float(np.log(np.diag(cholesky)).sum())
    mean_quadratic = float(np.trace(np.linalg.solve(covariance, biased_covariance)))
    return -0.5 * (dimension * np.log(2.0 * np.pi) + mean_quadratic) - log_det_cholesky


def _regularized_covariance(samples: np.ndarray, *, jitter: float) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    covariance = np.cov(array, rowvar=False, bias=False)
    covariance = np.atleast_2d(covariance)
    scale = float(np.trace(covariance) / covariance.shape[0]) if covariance.size else 1.0
    ridge = float(jitter) * max(scale, 1.0)
    return covariance + ridge * np.eye(covariance.shape[0], dtype=float)


def _safe_logdet(matrix: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(np.asarray(matrix, dtype=float))
    if sign <= 0 or not math.isfinite(float(logdet)):
        raise ValueError("Covariance matrix must be positive definite after regularization.")
    return float(logdet)


def estimate_gaussian_mutual_information(
    source: np.ndarray,
    target: np.ndarray,
    *,
    jitter: float = 1e-6,
) -> float:
    source_array = np.asarray(source, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if source_array.ndim == 1:
        source_array = source_array.reshape(-1, 1)
    if target_array.ndim == 1:
        target_array = target_array.reshape(-1, 1)
    if source_array.ndim != 2 or target_array.ndim != 2:
        raise ValueError("source and target must be one-dimensional or two-dimensional arrays.")
    if source_array.shape[0] != target_array.shape[0]:
        raise ValueError("source and target must share the sample axis.")
    if source_array.shape[0] < 3:
        raise ValueError("At least three samples are required.")
    if not np.isfinite(source_array).all() or not np.isfinite(target_array).all():
        raise ValueError("source and target must contain finite values.")

    source_cov = _regularized_covariance(source_array, jitter=float(jitter))
    target_cov = _regularized_covariance(target_array, jitter=float(jitter))
    joint_cov = _regularized_covariance(np.concatenate([source_array, target_array], axis=1), jitter=float(jitter))
    mi = 0.5 * (_safe_logdet(source_cov) + _safe_logdet(target_cov) - _safe_logdet(joint_cov)) / np.log(2.0)
    return max(0.0, float(mi))


def flatten_full_history_modes(history_modes: np.ndarray) -> np.ndarray:
    history = _validate_full_history_mode_inputs(history_modes)
    return history.reshape(history.shape[0], HISTORY_LENGTH * len(MODE_NAMES)).astype(float)


def enumerate_full_history_mode_pairs(mode_names: Sequence[str] | None = None) -> list[tuple[str, str]]:
    names = list(MODE_NAMES) if mode_names is None else [str(name) for name in mode_names]
    unknown = [name for name in names if name not in MODE_NAMES]
    if unknown:
        raise ValueError(f"Unknown source mode(s): {', '.join(unknown)}")
    return [(left, right) for left_index, left in enumerate(names) for right in names[left_index + 1 :]]


def _extract_full_history_mode_source(history_modes: np.ndarray, mode_name: str) -> np.ndarray:
    history = _validate_full_history_mode_inputs(history_modes)
    if mode_name not in MODE_NAMES:
        raise ValueError(f"Unknown mode: {mode_name}")
    return history[:, :, MODE_NAMES[mode_name]].astype(float)


def summarize_full_history_mode_pair_syn(
    history_modes: np.ndarray,
    left_name: str,
    right_name: str,
    target: np.ndarray,
    *,
    bootstrap_indices: np.ndarray | None = None,
) -> dict[str, float | str]:
    left_source = _extract_full_history_mode_source(history_modes, left_name)
    right_source = _extract_full_history_mode_source(history_modes, right_name)
    target_array = np.asarray(target, dtype=float)
    if target_array.ndim == 1:
        target_array = target_array.reshape(-1, 1)
    if target_array.ndim != 2 or target_array.shape[1] != 1:
        raise ValueError("target must be one-dimensional or a single-column 2D array.")
    if left_source.shape[0] != target_array.shape[0]:
        raise ValueError("history_modes and target must share the sample axis.")

    joint_source = np.concatenate([left_source, right_source], axis=1)
    left_ei = estimate_gaussian_mutual_information(left_source, target_array)
    right_ei = estimate_gaussian_mutual_information(right_source, target_array)
    joint_ei = estimate_gaussian_mutual_information(joint_source, target_array)
    result: dict[str, float | str] = {
        "backend": "gaussian_logdet_full_history_pair",
        "left_ei": float(left_ei),
        "right_ei": float(right_ei),
        "joint_ei": float(joint_ei),
        "syn": float(joint_ei - left_ei - right_ei),
    }
    if bootstrap_indices is not None and len(bootstrap_indices) > 0:
        boot_syn = []
        for indices in bootstrap_indices:
            boot_left = left_source[indices]
            boot_right = right_source[indices]
            boot_target = target_array[indices]
            boot_joint = np.concatenate([boot_left, boot_right], axis=1)
            boot_syn.append(
                estimate_gaussian_mutual_information(boot_joint, boot_target)
                - estimate_gaussian_mutual_information(boot_left, boot_target)
                - estimate_gaussian_mutual_information(boot_right, boot_target)
            )
        boot_array = np.asarray(boot_syn, dtype=float)
        result["syn_ci_low"] = float(np.nanpercentile(boot_array, 2.5))
        result["syn_ci_high"] = float(np.nanpercentile(boot_array, 97.5))
        result["syn_bootstrap_std"] = float(np.nanstd(boot_array, ddof=1)) if len(boot_array) > 1 else 0.0
    else:
        result["syn_ci_low"] = float("nan")
        result["syn_ci_high"] = float("nan")
        result["syn_bootstrap_std"] = float("nan")
    return result


def summarize_overall_ei_for_target(
    history_modes: np.ndarray,
    target: np.ndarray,
    *,
    bootstrap_indices: np.ndarray | None,
) -> dict[str, float]:
    source = flatten_full_history_modes(history_modes)
    target_array = np.asarray(target, dtype=float)
    if target_array.ndim == 1:
        target_array = target_array.reshape(-1, 1)
    overall_ei = estimate_gaussian_mutual_information(source, target_array)
    result = {"overall_ei": float(overall_ei), "backend": "gaussian_logdet_full_history"}
    if bootstrap_indices is not None and len(bootstrap_indices) > 0:
        boot_values = [
            estimate_gaussian_mutual_information(source[indices], target_array[indices])
            for indices in bootstrap_indices
        ]
        boot_array = np.asarray(boot_values, dtype=float)
        result["overall_ei_ci_low"] = float(np.nanpercentile(boot_array, 2.5))
        result["overall_ei_ci_high"] = float(np.nanpercentile(boot_array, 97.5))
        result["overall_ei_bootstrap_std"] = (
            float(np.nanstd(boot_array, ddof=1)) if len(boot_array) > 1 else 0.0
        )
    else:
        result["overall_ei_ci_low"] = float("nan")
        result["overall_ei_ci_high"] = float("nan")
        result["overall_ei_bootstrap_std"] = float("nan")
    return result


def _estimate_mutual_information_affine(x: np.ndarray, y: np.ndarray) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.ndim != 2 or y_array.ndim != 2 or x_array.shape[0] != y_array.shape[0]:
        raise ValueError("x and y must be 2D arrays with matching sample counts.")

    joint = np.concatenate([y_array, x_array], axis=1)
    mean_log_pxy = _mean_polynomial_tm_log_prob(joint)
    mean_log_px = _mean_polynomial_tm_log_prob(x_array)
    mean_log_py = _mean_polynomial_tm_log_prob(y_array)
    return float((mean_log_pxy - mean_log_px - mean_log_py) / np.log(2.0))


@lru_cache(maxsize=None)
def _polynomial_exponents_cached(predictor_dimension: int, degree: int) -> tuple[tuple[int, ...], ...]:
    if predictor_dimension == 0:
        return ((),)
    rows: list[tuple[int, ...]] = [tuple(0 for _ in range(predictor_dimension))]
    for total_degree in range(1, int(degree) + 1):
        rows.extend(
            tuple(int(value) for value in exponent)
            for exponent in np.ndindex(*([total_degree + 1] * predictor_dimension))
            if sum(exponent) == total_degree
        )
    return tuple(rows)


def _polynomial_design_fast(
    predictors: np.ndarray,
    *,
    degree: int,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values = np.asarray(predictors, dtype=float)
    if values.ndim != 2:
        raise ValueError("predictors must be a 2D array.")
    if values.shape[1] == 0:
        return np.ones((values.shape[0], 1), dtype=float)
    exponents = np.asarray(_polynomial_exponents_cached(values.shape[1], int(degree)), dtype=int)
    standardized = (values - np.asarray(mean, dtype=float)) / np.asarray(scale, dtype=float)
    return np.prod(standardized[:, None, :] ** exponents[None, :, :], axis=2)


def _mean_polynomial_tm_log_prob(
    samples: np.ndarray,
    *,
    degree: int = 3,
    ridge: float = 1e-6,
    min_scale: float = 1e-8,
) -> float:
    array = np.asarray(samples, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("samples must be a one-dimensional or two-dimensional array.")
    if array.shape[0] < 4:
        raise ValueError("samples must contain at least four rows.")
    if degree < 1:
        raise ValueError("degree must be positive.")

    quadratic_mean = 0.0
    log_scale_sum = 0.0
    for dimension in range(array.shape[1]):
        previous = array[:, :dimension]
        mean = previous.mean(axis=0) if dimension else np.empty(0, dtype=float)
        scale = previous.std(axis=0, ddof=1) if dimension else np.empty(0, dtype=float)
        scale = np.where(scale > min_scale, scale, 1.0)
        design = _polynomial_design_fast(previous, degree=int(degree), mean=mean, scale=scale)
        gram = design.T @ design + float(ridge) * np.eye(design.shape[1], dtype=float)
        gram[0, 0] -= float(ridge)
        coefficient = np.linalg.solve(gram, design.T @ array[:, dimension])
        residual = array[:, dimension] - design @ coefficient
        residual_scale = max(float(np.std(residual, ddof=1)), float(min_scale))
        reference = residual / residual_scale
        quadratic_mean += float(np.mean(reference**2))
        log_scale_sum += float(np.log(residual_scale))

    return -0.5 * (array.shape[1] * np.log(2.0 * np.pi) + quadratic_mean) - log_scale_sum


def summarize_two_source_syn_affine(
    left_source: np.ndarray,
    right_source: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    left_array = np.asarray(left_source, dtype=float)
    right_array = np.asarray(right_source, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if left_array.ndim != 2 or right_array.ndim != 2 or target_array.ndim != 2:
        raise ValueError("left_source, right_source, and target must be 2D arrays.")
    if left_array.shape[0] != right_array.shape[0] or left_array.shape[0] != target_array.shape[0]:
        raise ValueError("left_source, right_source, and target must share the sample axis.")
    if left_array.shape[1] != 1 or right_array.shape[1] != 1:
        raise ValueError("left_source and right_source must each contain exactly one source dimension.")

    left_ei = max(0.0, _estimate_mutual_information_affine(left_array, target_array))
    right_ei = max(0.0, _estimate_mutual_information_affine(right_array, target_array))
    joint_ei = max(0.0, _estimate_mutual_information_affine(np.concatenate([left_array, right_array], axis=1), target_array))
    return {
        "backend": "polynomial_triangular_transport_map_degree_3",
        "left_ei": float(left_ei),
        "right_ei": float(right_ei),
        "joint_ei": float(joint_ei),
        "syn": float(joint_ei - left_ei - right_ei),
    }


def _source_log_probs(left_source: np.ndarray, right_source: np.ndarray) -> dict[str, float]:
    return {
        "left": _mean_polynomial_tm_log_prob(left_source),
        "right": _mean_polynomial_tm_log_prob(right_source),
        "joint": _mean_polynomial_tm_log_prob(np.concatenate([left_source, right_source], axis=1)),
    }


def build_source_log_cache(
    left_source: np.ndarray,
    right_source: np.ndarray,
    *,
    bootstrap_indices: np.ndarray | None,
) -> dict[str, object]:
    cache: dict[str, object] = {"base": _source_log_probs(left_source, right_source)}
    if bootstrap_indices is not None and len(bootstrap_indices) > 0:
        cache["bootstrap"] = [
            _source_log_probs(left_source[indices], right_source[indices]) for indices in bootstrap_indices
        ]
    else:
        cache["bootstrap"] = []
    return cache


def _summarize_syn_with_source_logs(
    left_source: np.ndarray,
    right_source: np.ndarray,
    target: np.ndarray,
    source_logs: dict[str, float],
) -> dict[str, float]:
    target_log = _mean_polynomial_tm_log_prob(target)
    left_joint_log = _mean_polynomial_tm_log_prob(np.concatenate([target, left_source], axis=1))
    right_joint_log = _mean_polynomial_tm_log_prob(np.concatenate([target, right_source], axis=1))
    joint_log = _mean_polynomial_tm_log_prob(np.concatenate([target, left_source, right_source], axis=1))
    log_2 = float(np.log(2.0))
    left_ei = max(0.0, float((left_joint_log - source_logs["left"] - target_log) / log_2))
    right_ei = max(0.0, float((right_joint_log - source_logs["right"] - target_log) / log_2))
    joint_ei = max(0.0, float((joint_log - source_logs["joint"] - target_log) / log_2))
    return {
        "left_ei": float(left_ei),
        "right_ei": float(right_ei),
        "joint_ei": float(joint_ei),
        "syn": float(joint_ei - left_ei - right_ei),
    }


def summarize_syn_for_target(
    left_source: np.ndarray,
    right_source: np.ndarray,
    target: np.ndarray,
    *,
    bootstrap_indices: np.ndarray | None,
    source_log_cache: dict[str, object] | None = None,
) -> dict[str, float]:
    cache = source_log_cache or build_source_log_cache(left_source, right_source, bootstrap_indices=bootstrap_indices)
    summary = _summarize_syn_with_source_logs(
        left_source,
        right_source,
        target,
        cache["base"],
    )
    result = {
        "left_ei": float(summary["left_ei"]),
        "right_ei": float(summary["right_ei"]),
        "joint_ei": float(summary["joint_ei"]),
        "syn": float(summary["syn"]),
    }
    if bootstrap_indices is not None and len(bootstrap_indices) > 0:
        boot_syn = []
        boot_logs = list(cache.get("bootstrap", []))
        for boot_index, indices in enumerate(bootstrap_indices):
            source_logs = boot_logs[boot_index] if boot_index < len(boot_logs) else _source_log_probs(
                left_source[indices],
                right_source[indices],
            )
            boot = _summarize_syn_with_source_logs(
                left_source[indices],
                right_source[indices],
                target[indices],
                source_logs,
            )
            boot_syn.append(float(boot["syn"]))
        boot_array = np.asarray(boot_syn, dtype=float)
        result["syn_ci_low"] = float(np.nanpercentile(boot_array, 2.5))
        result["syn_ci_high"] = float(np.nanpercentile(boot_array, 97.5))
        result["syn_bootstrap_std"] = float(np.nanstd(boot_array, ddof=1)) if len(boot_array) > 1 else 0.0
    else:
        result["syn_ci_low"] = float("nan")
        result["syn_ci_high"] = float("nan")
        result["syn_bootstrap_std"] = float("nan")
    return result


def _safe_corr(left: Sequence[float], right: Sequence[float], *, method: str) -> float:
    a = pd.Series(left, dtype=float)
    b = pd.Series(right, dtype=float)
    if method == "spearman":
        a = a.rank(method="average")
        b = b.rank(method="average")
    if np.isclose(float(a.std(ddof=0)), 0.0) or np.isclose(float(b.std(ddof=0)), 0.0):
        return 1.0 if np.allclose(a.to_numpy(), b.to_numpy()) else 0.0
    corr = float(np.corrcoef(a.to_numpy(dtype=float), b.to_numpy(dtype=float))[0, 1])
    return corr if math.isfinite(corr) else 0.0


def _top_leads(frame: pd.DataFrame, k: int = 3) -> set[int]:
    return set(frame.sort_values(["syn", "lead"], ascending=[False, True]).head(k)["lead"].astype(int).tolist())


def _top_metric_leads(frame: pd.DataFrame, metric: str, k: int = 3) -> set[int]:
    return set(frame.sort_values([metric, "lead"], ascending=[False, True]).head(k)["lead"].astype(int).tolist())


def compute_relation_robustness(
    rows: pd.DataFrame,
    *,
    full_window: tuple[int, int] = (1, 24),
    climate_window: tuple[int, int] = (6, 18),
    pearson_threshold: float = 0.80,
    spearman_threshold: float = 0.75,
    top3_overlap_threshold: int = 2,
    sign_consistency_threshold: float = 0.80,
) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    for relation, relation_rows in rows.groupby("relation", sort=False):
        full = relation_rows[
            (relation_rows["lead"].astype(int) >= full_window[0])
            & (relation_rows["lead"].astype(int) <= full_window[1])
        ].copy()
        climate = relation_rows[
            (relation_rows["lead"].astype(int) >= climate_window[0])
            & (relation_rows["lead"].astype(int) <= climate_window[1])
        ].copy()
        seeds = sorted(int(seed) for seed in full["seed"].unique())
        pearsons: list[float] = []
        spearmans: list[float] = []
        top_overlaps: list[int] = []
        for seed_a, seed_b in itertools.combinations(seeds, 2):
            left = full[full["seed"] == seed_a].sort_values("lead")
            right = full[full["seed"] == seed_b].sort_values("lead")
            merged = left[["lead", "syn"]].merge(right[["lead", "syn"]], on="lead", suffixes=("_a", "_b"))
            pearsons.append(_safe_corr(merged["syn_a"], merged["syn_b"], method="pearson"))
            spearmans.append(_safe_corr(merged["syn_a"], merged["syn_b"], method="spearman"))
            top_overlaps.append(len(_top_leads(left) & _top_leads(right)))

        lead_stats = full.groupby("lead")["syn"].agg(["mean", "std"]).reset_index()
        lead_stats["std"] = lead_stats["std"].fillna(0.0)
        lead_stats["cv"] = lead_stats["std"] / lead_stats["mean"].abs().clip(lower=1e-12)
        positive_leads = lead_stats[lead_stats["mean"] > 0]["lead"].astype(int).tolist()
        if positive_leads:
            positive_rows = full[full["lead"].astype(int).isin(positive_leads)]
            positive_sign_consistency = float((positive_rows["syn"].astype(float) > 0).mean())
        else:
            positive_sign_consistency = 0.0

        pearson_min = float(np.min(pearsons)) if pearsons else 1.0
        spearman_min = float(np.min(spearmans)) if spearmans else 1.0
        top3_overlap_min = int(np.min(top_overlaps)) if top_overlaps else 3
        passes = (
            pearson_min >= pearson_threshold
            and spearman_min >= spearman_threshold
            and top3_overlap_min >= top3_overlap_threshold
            and positive_sign_consistency >= sign_consistency_threshold
        )
        relation_label = RELATIONS[relation].label if relation in RELATIONS else str(relation)
        summaries.append(
            {
                "relation": relation,
                "relation_label": relation_label,
                "n_seeds": len(seeds),
                "pearson_min": pearson_min,
                "pearson_mean": float(np.mean(pearsons)) if pearsons else 1.0,
                "spearman_min": spearman_min,
                "spearman_mean": float(np.mean(spearmans)) if spearmans else 1.0,
                "top3_overlap_min": top3_overlap_min,
                "top3_overlap_mean": float(np.mean(top_overlaps)) if top_overlaps else 3.0,
                "positive_sign_consistency": positive_sign_consistency,
                "mean_left_ei_full": float(full["left_ei"].astype(float).mean()),
                "mean_right_ei_full": float(full["right_ei"].astype(float).mean()),
                "mean_joint_ei_full": float(full["joint_ei"].astype(float).mean()),
                "mean_syn_full": float(full["syn"].astype(float).mean()),
                "mean_left_ei_climate": float(climate["left_ei"].astype(float).mean())
                if not climate.empty
                else float("nan"),
                "mean_right_ei_climate": float(climate["right_ei"].astype(float).mean())
                if not climate.empty
                else float("nan"),
                "mean_joint_ei_climate": float(climate["joint_ei"].astype(float).mean())
                if not climate.empty
                else float("nan"),
                "mean_syn_climate": float(climate["syn"].astype(float).mean()) if not climate.empty else float("nan"),
                "max_seed_std_by_lead": float(lead_stats["std"].max()) if not lead_stats.empty else 0.0,
                "max_seed_cv_by_lead": float(lead_stats["cv"].replace([np.inf, -np.inf], np.nan).max())
                if not lead_stats.empty
                else 0.0,
                "passes_robustness": bool(passes),
            }
        )
    return pd.DataFrame(summaries).sort_values("mean_syn_full", ascending=False).reset_index(drop=True)


def compute_overall_ei_robustness(
    rows: pd.DataFrame,
    *,
    full_window: tuple[int, int] = (1, 24),
    climate_window: tuple[int, int] = (6, 18),
    pearson_threshold: float = 0.80,
    spearman_threshold: float = 0.75,
    top3_overlap_threshold: int = 2,
) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    for target, target_rows in rows.groupby("target", sort=False):
        full = target_rows[
            (target_rows["lead"].astype(int) >= full_window[0])
            & (target_rows["lead"].astype(int) <= full_window[1])
        ].copy()
        climate = target_rows[
            (target_rows["lead"].astype(int) >= climate_window[0])
            & (target_rows["lead"].astype(int) <= climate_window[1])
        ].copy()
        seeds = sorted(int(seed) for seed in full["seed"].unique())
        pearsons: list[float] = []
        spearmans: list[float] = []
        top_overlaps: list[int] = []
        for seed_a, seed_b in itertools.combinations(seeds, 2):
            left = full[full["seed"] == seed_a].sort_values("lead")
            right = full[full["seed"] == seed_b].sort_values("lead")
            merged = left[["lead", "overall_ei"]].merge(
                right[["lead", "overall_ei"]],
                on="lead",
                suffixes=("_a", "_b"),
            )
            pearsons.append(_safe_corr(merged["overall_ei_a"], merged["overall_ei_b"], method="pearson"))
            spearmans.append(_safe_corr(merged["overall_ei_a"], merged["overall_ei_b"], method="spearman"))
            top_overlaps.append(len(_top_metric_leads(left, "overall_ei") & _top_metric_leads(right, "overall_ei")))

        lead_stats = full.groupby("lead")["overall_ei"].agg(["mean", "std"]).reset_index()
        lead_stats["std"] = lead_stats["std"].fillna(0.0)
        lead_stats["cv"] = lead_stats["std"] / lead_stats["mean"].abs().clip(lower=1e-12)
        pearson_min = float(np.min(pearsons)) if pearsons else 1.0
        spearman_min = float(np.min(spearmans)) if spearmans else 1.0
        top3_overlap_min = int(np.min(top_overlaps)) if top_overlaps else 3
        passes = (
            pearson_min >= pearson_threshold
            and spearman_min >= spearman_threshold
            and top3_overlap_min >= top3_overlap_threshold
        )
        summaries.append(
            {
                "target": str(target),
                "n_seeds": len(seeds),
                "pearson_min": pearson_min,
                "pearson_mean": float(np.mean(pearsons)) if pearsons else 1.0,
                "spearman_min": spearman_min,
                "spearman_mean": float(np.mean(spearmans)) if spearmans else 1.0,
                "top3_overlap_min": top3_overlap_min,
                "top3_overlap_mean": float(np.mean(top_overlaps)) if top_overlaps else 3.0,
                "mean_overall_ei_full": float(full["overall_ei"].astype(float).mean()),
                "mean_overall_ei_climate": float(climate["overall_ei"].astype(float).mean())
                if not climate.empty
                else float("nan"),
                "max_seed_std_by_lead": float(lead_stats["std"].max()) if not lead_stats.empty else 0.0,
                "max_seed_cv_by_lead": float(lead_stats["cv"].replace([np.inf, -np.inf], np.nan).max())
                if not lead_stats.empty
                else 0.0,
                "passes_robustness": bool(passes),
            }
        )
    return pd.DataFrame(summaries).sort_values("mean_overall_ei_full", ascending=False).reset_index(drop=True)


def summarize_relation_ranks(
    rows: pd.DataFrame,
    *,
    window: tuple[int, int] = (1, 24),
) -> tuple[pd.DataFrame, dict[str, float]]:
    filtered = rows[(rows["lead"].astype(int) >= window[0]) & (rows["lead"].astype(int) <= window[1])]
    rank_rows = (
        filtered.groupby(["seed", "relation"], as_index=False)["syn"]
        .mean()
        .rename(columns={"syn": "mean_syn"})
    )
    rank_rows["relation_label"] = rank_rows["relation"].map(lambda key: RELATIONS[key].label if key in RELATIONS else key)
    rank_rows = rank_rows.sort_values(["seed", "mean_syn", "relation"], ascending=[True, False, True])
    rank_rows["rank"] = rank_rows.groupby("seed")["mean_syn"].rank(ascending=False, method="first").astype(int)

    correlations: list[float] = []
    for seed_a, seed_b in itertools.combinations(sorted(rank_rows["seed"].unique()), 2):
        left = rank_rows[rank_rows["seed"] == seed_a][["relation", "rank"]]
        right = rank_rows[rank_rows["seed"] == seed_b][["relation", "rank"]]
        merged = left.merge(right, on="relation", suffixes=("_a", "_b"))
        correlations.append(_safe_corr(merged["rank_a"], merged["rank_b"], method="spearman"))
    summary = {
        "mean_seed_pair_spearman": float(np.mean(correlations)) if correlations else 1.0,
        "min_seed_pair_spearman": float(np.min(correlations)) if correlations else 1.0,
    }
    return rank_rows.reset_index(drop=True), summary


def summarize_seed_leads(rows: pd.DataFrame) -> pd.DataFrame:
    summary = rows.groupby(["relation", "lead"], as_index=False)["syn"].agg(["mean", "std"]).reset_index()
    summary["std"] = summary["std"].fillna(0.0)
    summary["cv"] = summary["std"] / summary["mean"].abs().clip(lower=1e-12)
    summary["relation_label"] = summary["relation"].map(lambda key: RELATIONS[key].label if key in RELATIONS else key)
    return summary


def plot_seed_overlay(rows: pd.DataFrame, output_base: Path) -> list[Path]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

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
    relation_keys = [key for key in RELATIONS if key in set(rows["relation"])]
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 4.6), constrained_layout=True, sharex=True)
    axes = axes.ravel()
    colors = {1: "#4C78A8", 2: "#F58518", 3: "#54A24B"}
    for axis, relation in zip(axes, relation_keys):
        relation_rows = rows[rows["relation"] == relation]
        for seed, seed_rows in relation_rows.groupby("seed"):
            seed_rows = seed_rows.sort_values("lead")
            axis.plot(
                seed_rows["lead"],
                seed_rows["syn"],
                marker="o",
                markersize=2.2,
                linewidth=1.2,
                color=colors.get(int(seed), None),
                label=f"Seed {int(seed)}",
            )
        axis.axhline(0, color="#999999", linewidth=0.7, linestyle="--")
        axis.set_title(RELATIONS[relation].label, fontsize=7.5)
        axis.set_xlabel("Lead (months)")
        axis.set_ylabel("Syn (bits)")
    for axis in axes[len(relation_keys) :]:
        axis.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".svg")]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_seed_mean_std(rows: pd.DataFrame, output_base: Path) -> list[Path]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7, "legend.frameon": False})
    summary = summarize_seed_leads(rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#72B7B2", "#E45756"]
    for color, relation in zip(palette, [key for key in RELATIONS if key in set(summary["relation"])]):
        subset = summary[summary["relation"] == relation].sort_values("lead")
        x = subset["lead"].to_numpy(dtype=float)
        mean = subset["mean"].to_numpy(dtype=float)
        std = subset["std"].to_numpy(dtype=float)
        ax.plot(x, mean, color=color, linewidth=1.4, label=RELATIONS[relation].label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
    ax.axhline(0, color="#999999", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Lead (months)")
    ax.set_ylabel("Seed mean Syn (bits)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".svg")]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_rank_heatmap(rank_frame: pd.DataFrame, output_base: Path) -> list[Path]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7})
    pivot = rank_frame.pivot(index="relation_label", columns="seed", values="rank")
    ordered = rank_frame.groupby("relation_label")["mean_syn"].mean().sort_values(ascending=False).index.tolist()
    pivot = pivot.loc[ordered]
    fig, ax = plt.subplots(figsize=(4.8, 3.6), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]), [f"Seed {int(seed)}" for seed in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0]), pivot.index)
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            ax.text(col, row, int(pivot.iloc[row, col]), ha="center", va="center", color="white", fontsize=7)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Rank")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".svg")]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_overall_ei_seed_overlay(rows: pd.DataFrame, output_base: Path) -> list[Path]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

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
    targets = sorted(str(target) for target in rows["target"].unique())
    n_cols = min(2, max(1, len(targets)))
    n_rows = int(math.ceil(len(targets) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 2.7 * n_rows), constrained_layout=True, sharex=True)
    axes_array = np.atleast_1d(axes).ravel()
    colors = {1: "#4C78A8", 2: "#F58518", 3: "#54A24B"}
    for axis, target in zip(axes_array, targets):
        target_rows = rows[rows["target"].astype(str) == target]
        for seed, seed_rows in target_rows.groupby("seed"):
            seed_rows = seed_rows.sort_values("lead")
            axis.plot(
                seed_rows["lead"],
                seed_rows["overall_ei"],
                marker="o",
                markersize=2.2,
                linewidth=1.2,
                color=colors.get(int(seed), None),
                label=f"Seed {int(seed)}",
            )
        axis.set_title(str(target), fontsize=7.5)
        axis.set_xlabel("Lead (months)")
        axis.set_ylabel("Overall EI (bits)")
    for axis in axes_array[len(targets) :]:
        axis.axis("off")
    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".svg")]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_full_history_pair_syn_top(summary: pd.DataFrame, lead_summary: pd.DataFrame, output_base: Path, *, top_k: int = 5) -> list[Path]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

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
    targets = sorted(str(target) for target in summary["target"].unique())
    n_cols = min(2, max(1, len(targets)))
    n_rows = int(math.ceil(len(targets) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 2.8 * n_rows), constrained_layout=True, sharex=True)
    axes_array = np.atleast_1d(axes).ravel()
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#72B7B2", "#E45756", "#79706E"]
    for axis, target in zip(axes_array, targets):
        top_pairs = (
            summary[summary["target"].astype(str) == target]
            .sort_values(["mean_syn", "pair"], ascending=[False, True])
            .head(int(top_k))["pair"]
            .tolist()
        )
        for color, pair in zip(palette, top_pairs):
            subset = lead_summary[
                (lead_summary["target"].astype(str) == target) & (lead_summary["pair"].astype(str) == str(pair))
            ].sort_values("lead")
            axis.plot(
                subset["lead"],
                subset["mean"],
                linewidth=1.2,
                marker="o",
                markersize=2.0,
                color=color,
                label=str(pair).replace("|", " + "),
            )
        axis.axhline(0, color="#999999", linewidth=0.7, linestyle="--")
        axis.set_title(str(target), fontsize=7.5)
        axis.set_xlabel("Lead (months)")
        axis.set_ylabel("Seed mean Syn (bits)")
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    for axis in axes_array[len(targets) :]:
        axis.axis("off")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".svg")]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def build_report_markdown(
    robustness_summary: pd.DataFrame,
    rank_summary: dict[str, float],
    *,
    output_dir: Path,
    figure_paths: Sequence[Path],
    n_samples: int,
    n_bootstrap: int,
    seeds: Sequence[int],
    intervention_bound: float,
    history_months: int,
    sampling_seed: int,
    start_month: int,
) -> str:
    passed = robustness_summary[robustness_summary["passes_robustness"].astype(bool)]
    failed = robustness_summary[~robustness_summary["passes_robustness"].astype(bool)]
    overall_pass = len(failed) == 0 and float(rank_summary["min_seed_pair_spearman"]) >= 0.70
    if overall_pass:
        headline = "三个 seed 的 Syn 曲线整体一致，当前 Modeformer PEID/Syn 读数对初始化 seed 具有鲁棒性。"
    elif len(passed) > 0:
        headline = "三个 seed 的 Syn 曲线在部分 relation 上一致；不稳定 relation 不进入科学解释。"
    else:
        headline = "三个 seed 的 Syn 曲线未通过鲁棒性标准，当前结果只能作为诊断。"

    def _fmt_float(value: object) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "nan"
        return "nan" if not math.isfinite(number) else f"{number:.3f}"

    def _fmt_bits(value: object) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "nan"
        return "nan" if not math.isfinite(number) else f"{number:.6f}"

    def _display_path(path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(ROOT))
        except ValueError:
            return str(path)

    lines = [
        "# UniCM Modeformer PEID/Syn seed 鲁棒性分析",
        "",
        "## 结论",
        "",
        headline,
        "",
        f"- checkpoint seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"- intervention samples: `{int(n_samples)}`",
        f"- intervention support: each source mode is sampled independently from `[-{float(intervention_bound):g}, {float(intervention_bound):g}]`",
        f"- bootstrap repeats: `{int(n_bootstrap)}`",
        "- 主窗口：lead `1..24`；climate-relevant 补充窗口：lead `6..18`",
        f"- relation ranking min Spearman: `{_fmt_float(rank_summary['min_seed_pair_spearman'])}`",
        f"- relation ranking mean Spearman: `{_fmt_float(rank_summary['mean_seed_pair_spearman'])}`",
        "- 通过标准：seed-pair Pearson >= `0.80`，Spearman >= `0.75`，top-3 Syn lead overlap >= `2`，"
        "positive-sign consistency >= `0.80`，relation ranking 跨 seed Spearman >= `0.70`。",
        "",
        "## 干预与时空处理",
        "",
        "本实验不是把真实 reanalysis 时间序列中的某些月份取出来做扰动，而是在 frozen UniCM Modeformer 上构造合成的有界干预集合。"
        f"对每个 source pair，脚本使用 sampling seed `{int(sampling_seed)}` 生成 `{int(n_samples)}` 个样本；"
        f"左源和右源两个 climate-mode index 分别独立采样自均匀分布 `U(-{float(intervention_bound):g}, {float(intervention_bound):g})`。"
        "同一批 source samples 在所有 checkpoint seed、relation 和 lead 上复用，因此 seed 差异主要来自 checkpoint learned mechanism，而不是干预样本随机性。",
        "",
        f"每个干预样本被写入 Modeformer 的 mode 分支输入张量，形状为 `(B, 36, 1, 11, 1)`：`36 = 12` 个月历史窗口 + `24` 个月预测窗口，`11` 是 10 个 SST climate modes 加 WWV-like t20d mode。"
        f"在当前实现的默认 `history_months={int(history_months)}` 下，两个源模态的干预值只填入历史窗口最后 `{int(history_months)}` 个月，即时间位置 `12-history_months..11`；"
        "历史窗口更早月份、未来 24 个月占位、以及非 source 模态都置零。"
        "因此这里的实际口径只是“零背景下的局部机制探针”：把两个指定 climate-mode index 在最近若干历史月固定到同一个合成幅值，然后读取模型自回归未来响应。",
        "",
        "这个口径不满足更严格的 bounded maximum-entropy intervention 定义。"
        "若把整个历史 mode 输入视为机制输入，那么所有输入维度都应从指定支持上的最大熵分布采样；"
        "在有界盒支持下，即对所有历史月份和所有 mode 维度使用相应的均匀分布。"
        "对二源 PEID 来说，left/right 源维度是待分解变量，其余历史 mode 维度不应固定为 0，而应作为 nuisance intervention variables 同步采样并在估计中边缘化。"
        "否则，读数依赖于人为零背景，不能解释为完整输入空间上的 EI 或 PEID。",
        "",
        f"月份时间戳仍然提供给模型：36 个位置使用从 `start_month={int(start_month)}` 开始的月序号并按 12 个月循环。"
        "推理时 `train=False`，模型先编码 12 个月历史 mode 状态，再从第 12 个月末状态开始自回归产生 lead `1..24` 的所有 mode 输出。"
        "报告中某个 lead 的 Syn 是同一批 `(left_source, right_source)` 干预样本与该 lead 上目标 mode 输出之间的信息分解读数。",
        "",
        "“只对历史窗口最早月份做最大熵采样，其余历史月份由模型推出”在概念上可以作为另一种 closed-loop roll-in 机制，但不是当前 UniCM `forward_sep(train=False)` 原生支持的操作。"
        "当前模型的 encoder 需要一次性接收完整 12 个月历史状态；自回归循环只在 decoder 的未来 24 个月内发生。"
        "如果要采用这种方案，需要额外定义一个历史填充/roll-in 过程，例如先用外部动力学模型或滑动窗口递推生成第 2 到第 12 个历史月，再把生成后的 12 个月历史输入 UniCM。"
        "否则仅采样第 1 个月而把后 11 个月交给当前 forward 推出，在实现上没有明确入口，在理论上也会把“历史状态的干预分布”改成“单月干预加模型诱导历史分布”，与全历史最大熵干预不是同一个 estimand。",
        "",
        "因此，后续更合理的重跑方案应优先采用全历史输入最大熵采样：对 12 个历史月份、11 个 mode 维度同时采样 bounded uniform box；"
        "按 relation 指定的两个源变量读取 PEID 的 left/right 分量；其他历史输入作为背景干预变量进入模型但不进入分解源集合。"
        "若研究问题明确限定为“最早月份冲击经过模型内部传播后的效应”，再单独设计 closed-loop roll-in 实验，而不把它与全历史最大熵 EI 混用。",
        "",
        "空间维度在这里按 UniCM 的 mode 分支处理，而不是按物理网格处理。"
        "11 个 climate modes 被模型当作伪空间维度；本实验没有向 5 通道全球物理场分支注入 SST、风应力或热含量网格，也没有对具体经纬度格点做局部遮挡。"
        "所以 relation 里的 `NPMM + WWV -> nino` 等关系表示“两个指数模态作为源、目标指数模态作为输出”的 mode-level 机制读出；它不能直接解释为某两个地理格点对另一个格点的干预效应。",
        "",
        "PEID/Syn 估计时，对每个 lead 分别计算 `left_source -> target`、`right_source -> target` 和 `{left_source, right_source} -> target` 的信息量；"
        "密度项由三阶 polynomial triangular transport map 估计，Syn 取 `joint_ei - left_ei - right_ei`。"
        f"`{int(n_bootstrap)}` 次 bootstrap 只在这 `{int(n_samples)}` 个干预样本上重采样，用于估计 Syn 的不确定性；不重新训练模型，也不重新抽取新的时间序列。",
        "",
        "## 鲁棒 relation",
        "",
    ]
    if passed.empty:
        lines.append("- 无。")
    else:
        for _, row in passed.iterrows():
            lines.append(
                "- {label}: mean Syn 1..24 `{mean_syn}`, mean Syn 6..18 `{climate_syn}`, "
                "Pearson min `{pearson}`, Spearman min `{spearman}`, top-3 overlap min `{overlap}`, "
                "sign consistency `{sign}`。".format(
                    label=row["relation_label"],
                    mean_syn=_fmt_float(row["mean_syn_full"]),
                    climate_syn=_fmt_float(row["mean_syn_climate"]),
                    pearson=_fmt_float(row["pearson_min"]),
                    spearman=_fmt_float(row["spearman_min"]),
                    overlap=int(row["top3_overlap_min"]),
                    sign=_fmt_float(row["positive_sign_consistency"]),
                )
            )
    lines.extend(["", "## Syn coupling 排名", ""])
    for rank, (_, row) in enumerate(robustness_summary.sort_values("mean_syn_full", ascending=False).iterrows(), start=1):
        status = "通过" if bool(row["passes_robustness"]) else "不稳定"
        lines.append(
            f"{rank}. {row['relation_label']}: mean Syn 1..24 `{_fmt_float(row['mean_syn_full'])}`, "
            f"mean Syn 6..18 `{_fmt_float(row['mean_syn_climate'])}`，`{status}`。"
        )
    lines.extend(
        [
            "",
            "## EI 分量结果",
            "",
            "下表报告与 Syn 同一组干预样本上的窗口均值，单位为 bits。"
            "`left EI` 和 `right EI` 是两个源模态单独对目标 lead 输出的信息读数；"
            "`joint EI` 是两个源模态联合输入时对同一目标的读数。",
            "",
            "| Relation | left EI 1..24 | right EI 1..24 | joint EI 1..24 | Syn 1..24 | left EI 6..18 | right EI 6..18 | joint EI 6..18 | Syn 6..18 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in robustness_summary.sort_values("mean_joint_ei_full", ascending=False).iterrows():
        lines.append(
            f"| {row['relation_label']} | "
            f"{_fmt_bits(row['mean_left_ei_full'])} | "
            f"{_fmt_bits(row['mean_right_ei_full'])} | "
            f"{_fmt_bits(row['mean_joint_ei_full'])} | "
            f"{_fmt_bits(row['mean_syn_full'])} | "
            f"{_fmt_bits(row['mean_left_ei_climate'])} | "
            f"{_fmt_bits(row['mean_right_ei_climate'])} | "
            f"{_fmt_bits(row['mean_joint_ei_climate'])} | "
            f"{_fmt_bits(row['mean_syn_climate'])} |"
        )
    lines.extend(
        [
            "",
            "这些 EI 均值显示，联合读数普遍高于两个单源读数，但绝对量级仍很小；"
            "同时由于上面的 seed 鲁棒性标准未通过，这些 EI/Syn 数值应作为 Modeformer 机制诊断，而不是稳定气候因果强度排序。",
        ]
    )
    lines.extend(["", "## 不稳定 relation", ""])
    if failed.empty:
        lines.append("- 无。")
    else:
        for _, row in failed.iterrows():
            lines.append(
                "- {label}: Pearson min `{pearson}`, Spearman min `{spearman}`, top-3 overlap min `{overlap}`, "
                "sign consistency `{sign}`。".format(
                    label=row["relation_label"],
                    pearson=_fmt_float(row["pearson_min"]),
                    spearman=_fmt_float(row["spearman_min"]),
                    overlap=int(row["top3_overlap_min"]),
                    sign=_fmt_float(row["positive_sign_consistency"]),
                )
            )
    lines.extend(
        [
            "",
            "## 图表与数据",
            "",
            f"- 逐 seed / relation / lead 原始结果：`{_display_path(output_dir / 'seed_robustness_rows.jsonl')}`",
            f"- relation 鲁棒性汇总：`{_display_path(output_dir / 'seed_robustness_summary.csv')}`",
            f"- relation 排序：`{_display_path(output_dir / 'relation_rank_by_seed.csv')}`",
            f"- lead-level seed mean/std：`{_display_path(output_dir / 'seed_lead_summary.csv')}`",
        ]
    )
    for path in figure_paths:
        lines.append(f"- 图：`{_display_path(path)}`")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 本报告只分析 frozen UniCM checkpoint 的 Modeformer learned mechanism，不使用 reanalysis 数据做预测复现。",
            "- 当前结果使用零背景局部探针，不满足全历史输入最大熵干预定义；在按全输入最大熵采样重跑前，EI/Syn 数值只能作为实现诊断。",
            "- Syn 是 bounded-intervention 下的信息分解读数；即使用修正后的最大熵干预，也不能直接声明为真实气候系统因果强度。",
            "- 当前只有 3 个发布 checkpoint，因此 seed 鲁棒性结论限定在这 3 个初始化上。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_overall_ei_report_markdown(
    robustness_summary: pd.DataFrame,
    *,
    output_dir: Path,
    figure_paths: Sequence[Path],
    n_samples: int,
    n_bootstrap: int,
    seeds: Sequence[int],
    intervention_bound: float,
    sampling_seed: int,
    start_month: int,
) -> str:
    passed = robustness_summary[robustness_summary["passes_robustness"].astype(bool)]
    failed = robustness_summary[~robustness_summary["passes_robustness"].astype(bool)]
    if failed.empty:
        headline = "全历史最大熵采样下，整体 EI lead 曲线在当前 checkpoint seeds 间通过鲁棒性标准。"
    elif not passed.empty:
        headline = "全历史最大熵采样下，部分目标 mode 的整体 EI lead 曲线通过鲁棒性标准。"
    else:
        headline = "全历史最大熵采样下，整体 EI lead 曲线未通过 seed 鲁棒性标准。"

    def _fmt_float(value: object) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "nan"
        return "nan" if not math.isfinite(number) else f"{number:.3f}"

    def _fmt_bits(value: object) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "nan"
        return "nan" if not math.isfinite(number) else f"{number:.6f}"

    def _display_path(path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(ROOT))
        except ValueError:
            return str(path)

    mean_ei = float(robustness_summary["mean_overall_ei_full"].mean()) if not robustness_summary.empty else float("nan")
    lines = [
        "# UniCM Modeformer 全历史最大熵整体 EI seed 鲁棒性分析",
        "",
        "## 结论",
        "",
        headline,
        "",
        f"- checkpoint seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"- intervention samples: `{int(n_samples)}`",
        f"- intervention support: all 12 historical months x 11 mode dimensions sampled independently from `[-{float(intervention_bound):g}, {float(intervention_bound):g}]`",
        f"- sampling seed: `{int(sampling_seed)}`",
        f"- bootstrap repeats: `{int(n_bootstrap)}`",
        f"- start month: `{int(start_month)}`",
        f"- target-mean overall EI 1..24: `{_fmt_bits(mean_ei)}` bits",
        "- 主窗口：lead `1..24`；climate-relevant 补充窗口：lead `6..18`",
        "- 通过标准：seed-pair Pearson >= `0.80`，Spearman >= `0.75`，top-3 EI lead overlap >= `2`。",
        "",
        "## 干预口径",
        "",
        "这里把 UniCM mode 分支的完整历史输入视为机制输入。每个样本同时采样 12 个历史月份和 11 个 mode 维度，形成 `(B, 12, 11)` 的 bounded uniform 最大熵输入；该历史张量写入模型 encoder 的 12 个月历史段，未来 24 个月仍由 decoder 在 `train=False` 下自回归生成。",
        "",
        "整体 EI 读数使用 flattened full-history source，即 132 维历史 mode 输入，对每个目标 mode 和 lead 分别估计 `I(history_{1:12,1:11}; target_lead)`。高维整体读数采用 Gaussian log-det MI 作为快速筛查口径；它用于检查绝对量级和 seed 稳定性，不等同于二源 PEID/Syn 分解。",
        "",
        "## Overall EI target 排名",
        "",
        "| Target | mean EI 1..24 | mean EI 6..18 | Pearson min | Spearman min | top-3 overlap min | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in robustness_summary.sort_values("mean_overall_ei_full", ascending=False).iterrows():
        status = "通过" if bool(row["passes_robustness"]) else "不稳定"
        lines.append(
            f"| {row['target']} | {_fmt_bits(row['mean_overall_ei_full'])} | "
            f"{_fmt_bits(row['mean_overall_ei_climate'])} | {_fmt_float(row['pearson_min'])} | "
            f"{_fmt_float(row['spearman_min'])} | {int(row['top3_overlap_min'])} | {status} |"
        )
    if figure_paths:
        first_figure = _display_path(figure_paths[0])
        lines.extend(
            [
                "",
                f"![Full-history overall EI seed overlay](../../{first_figure})",
                "",
                "*图 1. Full-history overall EI lead curves under the selected bounded maximum-entropy intervention. "
                "Each panel is one target mode and each curve is one checkpoint seed; stable targets should show both similar curve shape and similar lead ordering across seeds.*",
            ]
        )
    lines.extend(["", "## 不稳定 target", ""])
    if failed.empty:
        lines.append("- 无。")
    else:
        for _, row in failed.iterrows():
            lines.append(
                f"- {row['target']}: Pearson min `{_fmt_float(row['pearson_min'])}`, "
                f"Spearman min `{_fmt_float(row['spearman_min'])}`, top-3 overlap min `{int(row['top3_overlap_min'])}`。"
            )
    lines.extend(
        [
            "",
            "## 图表与数据",
            "",
            f"- 逐 seed / target / lead 原始结果：`{_display_path(output_dir / 'overall_ei_rows.jsonl')}`",
            f"- target 鲁棒性汇总：`{_display_path(output_dir / 'overall_ei_seed_robustness_summary.csv')}`",
            f"- lead-level seed mean/std：`{_display_path(output_dir / 'overall_ei_seed_lead_summary.csv')}`",
        ]
    )
    for path in figure_paths:
        lines.append(f"- 图：`{_display_path(path)}`")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 本报告只分析 frozen UniCM checkpoint 的 Modeformer learned mechanism，不使用 reanalysis 数据做预测复现。",
            "- 这里的整体 EI 是全历史输入到单目标 lead 输出的 Gaussian log-det 读数，用于量级和 seed 稳定性筛查。",
            "- 若整体 EI 量级和 seed 稳定性足够，再继续对指定 source pair 做二源 PEID/Syn 分解更合理。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_full_history_pair_syn_report_markdown(
    summary: pd.DataFrame,
    *,
    output_dir: Path,
    overall_cache_dir: Path,
    figure_paths: Sequence[Path],
    n_samples: int,
    n_bootstrap: int,
    seeds: Sequence[int],
    intervention_bound: float,
    sampling_seed: int,
    start_month: int,
    source_modes: Sequence[str],
    target_names: Sequence[str],
) -> str:
    def _fmt_bits(value: object) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "nan"
        return "nan" if not math.isfinite(number) else f"{number:.6f}"

    def _display_path(path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(ROOT))
        except ValueError:
            return str(path)

    top_rows = summary.sort_values(["target", "mean_syn", "pair"], ascending=[True, False, True])
    lines = [
        "# UniCM Modeformer 全历史 mode-pair Syn 分析",
        "",
        "## 结论",
        "",
        "本轮使用与 full-history overall EI 图完全相同的干预集合和 checkpoint 输出缓存，计算 source mode pair 到目标 mode lead 输出的 Gaussian log-det Syn。"
        "结果是 mode-level 二源筛查读数，不重新运行 UniCM forward。",
        "",
        f"- checkpoint seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"- intervention samples: `{int(n_samples)}`",
        f"- intervention support: all 12 historical months x 11 mode dimensions sampled independently from `[-{float(intervention_bound):g}, {float(intervention_bound):g}]`",
        f"- sampling seed: `{int(sampling_seed)}`",
        f"- start month: `{int(start_month)}`",
        f"- bootstrap repeats: `{int(n_bootstrap)}`",
        f"- source modes: `{', '.join(source_modes)}`",
        f"- target modes: `{', '.join(target_names)}`",
        f"- reused prediction cache: `{_display_path(overall_cache_dir)}`",
        "",
        "## 估计口径",
        "",
        "每个 source mode 使用该 mode 的 12 个月历史向量作为一个多维源；target 是对应 lead 的单个目标 mode 输出。"
        "对每个 `(left, right, target, lead, checkpoint seed)` 估计 `I(left; target)`、`I(right; target)` 和 `I(left,right; target)`，"
        "Syn 定义为 `joint EI - left EI - right EI`。"
        "由于其他历史 mode 同步来自同一 full-history maximum-entropy intervention ensemble，但不进入 source 集合，读数是在这些 nuisance intervention variables 上边缘化后的 mode-pair Syn。",
        "",
        "## Top source pairs",
        "",
    ]
    for target in target_names:
        target_rows = top_rows[top_rows["target"].astype(str) == str(target)].head(10)
        lines.extend(
            [
                f"### {target}",
                "",
                "| Rank | Source pair | mean Syn 1..24 | joint EI | left EI | right EI |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in target_rows.iterrows():
            lines.append(
                f"| {int(row['rank_within_target'])} | {row['left_source']} + {row['right_source']} | "
                f"{_fmt_bits(row['mean_syn'])} | {_fmt_bits(row['mean_joint_ei'])} | "
                f"{_fmt_bits(row['mean_left_ei'])} | {_fmt_bits(row['mean_right_ei'])} |"
            )
        lines.append("")
    if figure_paths:
        first_figure = _display_path(figure_paths[0])
        lines.extend(
            [
                f"![Top mode-pair Syn curves](../../{first_figure})",
                "",
                "*图 1. 每个 target 按 1..24 lead 平均 Syn 排名前五的 source-mode pair 曲线；曲线为 checkpoint seed 均值。*",
                "",
            ]
        )
    lines.extend(
        [
            "## 图表与数据",
            "",
            f"- raw rows: `{_display_path(output_dir / 'full_history_mode_pair_syn_rows.jsonl')}`",
            f"- pair summary: `{_display_path(output_dir / 'full_history_mode_pair_syn_summary.csv')}`",
            f"- lead summary: `{_display_path(output_dir / 'full_history_mode_pair_syn_lead_summary.csv')}`",
            f"- top pairs: `{_display_path(output_dir / 'full_history_mode_pair_syn_top_pairs.csv')}`",
        ]
    )
    for path in figure_paths:
        lines.append(f"- figure: `{_display_path(path)}`")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 后端与 overall EI 图一致，使用 Gaussian log-det MI；这适合快速筛查，不等同于 transport-map PEID 的最终非线性分解。",
            "- Syn 可以为负，表示 pair 的联合读数低于两个单源读数之和；这里不做非负截断。",
            "- 结果只对应 frozen UniCM Modeformer learned mechanism，不是 reanalysis 预测技能评估。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_jsonl(rows: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prediction_cache_path(
    cache_dir: Path,
    *,
    seed: int,
    left_name: str,
    right_name: str,
    args: argparse.Namespace,
) -> Path:
    stem = (
        f"pred_seed{int(seed)}_{left_name}_{right_name}"
        f"_samples{int(args.n_samples)}_sampling{int(args.sampling_seed)}"
        f"_bound{float(args.intervention_bound):g}_hist{int(args.history_months)}"
        f"_start{int(args.start_month)}_{args.device}"
    )
    safe_stem = "".join(char if char.isalnum() or char in "._-" else "_" for char in stem)
    return cache_dir / f"{safe_stem}.npz"


def overall_prediction_cache_path(
    cache_dir: Path,
    *,
    seed: int,
    args: argparse.Namespace,
) -> Path:
    stem = (
        f"overall_pred_seed{int(seed)}"
        f"_samples{int(args.n_samples)}_sampling{int(args.sampling_seed)}"
        f"_bound{float(args.intervention_bound):g}_fullhist12"
        f"_start{int(args.start_month)}_{args.device}"
    )
    safe_stem = "".join(char if char.isalnum() or char in "._-" else "_" for char in stem)
    return cache_dir / f"{safe_stem}.npz"


def _resolve_overall_targets(args: argparse.Namespace, relations: Sequence[Relation]) -> list[str]:
    raw_targets = getattr(args, "overall_targets", None)
    if raw_targets:
        targets = [str(target) for target in raw_targets]
    else:
        targets = []
        for relation in relations:
            if relation.target not in targets:
                targets.append(relation.target)
    unknown = [target for target in targets if target not in MODE_NAMES]
    if unknown:
        raise ValueError(f"Unknown target mode(s): {', '.join(unknown)}")
    return targets


def _resolve_source_modes(args: argparse.Namespace) -> list[str]:
    raw_sources = getattr(args, "source_modes", None)
    source_modes = list(MODE_NAMES) if not raw_sources else [str(source) for source in raw_sources]
    unknown = [source for source in source_modes if source not in MODE_NAMES]
    if unknown:
        raise ValueError(f"Unknown source mode(s): {', '.join(unknown)}")
    return source_modes


def summarize_overall_seed_leads(rows: pd.DataFrame) -> pd.DataFrame:
    summary = rows.groupby(["target", "lead"], as_index=False)["overall_ei"].agg(["mean", "std"]).reset_index()
    summary["std"] = summary["std"].fillna(0.0)
    summary["cv"] = summary["std"] / summary["mean"].abs().clip(lower=1e-12)
    return summary


def summarize_full_history_pair_seed_leads(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["pair", "left_source", "right_source", "target", "lead"]
    summary = rows.groupby(group_cols, as_index=False)["syn"].agg(["mean", "std"]).reset_index()
    summary["std"] = summary["std"].fillna(0.0)
    summary["cv"] = summary["std"] / summary["mean"].abs().clip(lower=1e-12)
    return summary


def summarize_full_history_pair_syn(rows: pd.DataFrame, *, window: tuple[int, int] = (1, 24)) -> pd.DataFrame:
    filtered = rows[(rows["lead"].astype(int) >= window[0]) & (rows["lead"].astype(int) <= window[1])].copy()
    group_cols = ["pair", "left_source", "right_source", "target"]
    summary = (
        filtered.groupby(group_cols, as_index=False)
        .agg(
            mean_left_ei=("left_ei", "mean"),
            mean_right_ei=("right_ei", "mean"),
            mean_joint_ei=("joint_ei", "mean"),
            mean_syn=("syn", "mean"),
            std_syn=("syn", "std"),
            min_syn=("syn", "min"),
            max_syn=("syn", "max"),
        )
        .sort_values(["target", "mean_syn", "pair"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    summary["std_syn"] = summary["std_syn"].fillna(0.0)
    summary["rank_within_target"] = summary.groupby("target")["mean_syn"].rank(ascending=False, method="first").astype(int)
    return summary


def load_full_history_prediction_cache(cache_path: Path, *, n_samples: int) -> np.ndarray:
    with np.load(cache_path) as payload:
        if "all_mode_targets" not in payload:
            raise ValueError(f"Prediction cache is missing all_mode_targets: {cache_path}")
        all_mode_targets = payload["all_mode_targets"].astype(float)
    expected_shape = (int(n_samples), PREDICTION_LENGTH, len(MODE_NAMES))
    if tuple(all_mode_targets.shape) != expected_shape:
        raise ValueError(
            f"Prediction cache has shape {tuple(all_mode_targets.shape)}, expected {expected_shape}: {cache_path}"
        )
    if not np.isfinite(all_mode_targets).all():
        raise ValueError(f"Prediction cache contains non-finite values: {cache_path}")
    return all_mode_targets


def run_overall_ei_analysis(args: argparse.Namespace) -> dict[str, object]:
    import torch

    if args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))

    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "fig"
    cache_dir = output_dir / "cache"
    report_path = Path(args.report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = [int(seed) for seed in args.seeds]
    relations = [RELATIONS[key] for key in args.relations]
    target_names = _resolve_overall_targets(args, relations)
    leads = parse_leads(args.leads)
    checkpoint_paths = resolve_checkpoint_paths(Path(args.checkpoint_root), seeds)
    history_modes = sample_full_history_mode_inputs(
        n_samples=int(args.n_samples),
        intervention_bound=float(args.intervention_bound),
        seed=int(args.sampling_seed),
    )
    rng = np.random.default_rng(int(args.sampling_seed) + 7919)
    bootstrap_indices = (
        rng.integers(0, int(args.n_samples), size=(int(args.n_bootstrap), int(args.n_samples)))
        if int(args.n_bootstrap) > 0
        else None
    )

    rows: list[dict[str, object]] = []
    for seed in seeds:
        print(f"[overall seed {seed}] loading {checkpoint_paths[seed]}", file=sys.stderr, flush=True)
        model = load_unicm_model(checkpoint_paths[seed], args.device)
        cache_path = overall_prediction_cache_path(cache_dir, seed=seed, args=args)
        if bool(args.prediction_cache) and cache_path.exists():
            print(f"  loading prediction cache {cache_path}", file=sys.stderr, flush=True)
            with np.load(cache_path) as payload:
                all_mode_targets = payload["all_mode_targets"].astype(float)
            if not np.isfinite(all_mode_targets).all():
                raise ValueError(f"Prediction cache contains non-finite values: {cache_path}")
        else:
            all_mode_targets = predict_modeformer_all_modes_from_history(
                model,
                history_modes,
                device=args.device,
                batch_size=args.batch_size,
                start_month=args.start_month,
            )
            if not np.isfinite(all_mode_targets).all():
                raise ValueError(f"Model prediction contains non-finite values for seed {seed}.")
            if bool(args.prediction_cache):
                np.savez(
                    cache_path,
                    all_mode_targets=all_mode_targets.astype(np.float32),
                    metadata=json.dumps(
                        {
                            "seed": int(seed),
                            "n_samples": int(args.n_samples),
                            "sampling_seed": int(args.sampling_seed),
                            "intervention_bound": float(args.intervention_bound),
                            "sampling_mode": "full_history_max_entropy",
                            "history_shape": list(history_modes.shape[1:]),
                            "start_month": int(args.start_month),
                            "device": str(args.device),
                        },
                        sort_keys=True,
                    ),
                )
        for target_name in target_names:
            target_index = MODE_NAMES[target_name]
            targets = all_mode_targets[:, :, target_index]
            print(f"  overall EI target {target_name}", file=sys.stderr, flush=True)
            for lead in leads:
                summary = summarize_overall_ei_for_target(
                    history_modes,
                    targets[:, [int(lead) - 1]],
                    bootstrap_indices=bootstrap_indices,
                )
                rows.append(
                    {
                        "seed": int(seed),
                        "checkpoint": str(checkpoint_paths[seed]),
                        "target": target_name,
                        "lead": int(lead),
                        "n_samples": int(args.n_samples),
                        "n_bootstrap": int(args.n_bootstrap),
                        "intervention_bound": float(args.intervention_bound),
                        "sampling_seed": int(args.sampling_seed),
                        "sampling_mode": "full_history_max_entropy",
                        **summary,
                    }
                )
        del model

    row_path = output_dir / "overall_ei_rows.jsonl"
    write_jsonl(rows, row_path)
    frame = pd.DataFrame(rows)
    robustness_summary = compute_overall_ei_robustness(frame)
    robustness_path = output_dir / "overall_ei_seed_robustness_summary.csv"
    robustness_summary.to_csv(robustness_path, index=False)
    lead_summary_path = output_dir / "overall_ei_seed_lead_summary.csv"
    summarize_overall_seed_leads(frame).to_csv(lead_summary_path, index=False)

    figure_paths: list[Path] = []
    figure_paths.extend(plot_overall_ei_seed_overlay(frame, fig_dir / "overall_ei_seed_overlay"))
    markdown = build_overall_ei_report_markdown(
        robustness_summary,
        output_dir=output_dir,
        figure_paths=[path for path in figure_paths if path.suffix == ".png"],
        n_samples=int(args.n_samples),
        n_bootstrap=int(args.n_bootstrap),
        seeds=seeds,
        intervention_bound=float(args.intervention_bound),
        sampling_seed=int(args.sampling_seed),
        start_month=int(args.start_month),
    )
    report_path.write_text(markdown, encoding="utf-8")
    return {
        "rows": str(row_path),
        "robustness_summary": str(robustness_path),
        "lead_summary": str(lead_summary_path),
        "report": str(report_path),
        "figures": [str(path) for path in figure_paths],
    }


def run_full_history_mode_pair_syn_analysis(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "fig"
    report_path = Path(args.report_path)
    overall_cache_dir = Path(args.overall_cache_dir) if args.overall_cache_dir else output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    overall_cache_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if args.torch_threads > 0:
        try:
            import torch

            torch.set_num_threads(int(args.torch_threads))
        except ImportError:
            pass

    seeds = [int(seed) for seed in args.seeds]
    relations = [RELATIONS[key] for key in args.relations]
    target_names = _resolve_overall_targets(args, relations)
    source_modes = _resolve_source_modes(args)
    source_pairs = enumerate_full_history_mode_pairs(source_modes)
    leads = parse_leads(args.leads)
    checkpoint_paths = resolve_checkpoint_paths(Path(args.checkpoint_root), seeds)
    history_modes = sample_full_history_mode_inputs(
        n_samples=int(args.n_samples),
        intervention_bound=float(args.intervention_bound),
        seed=int(args.sampling_seed),
    )
    rng = np.random.default_rng(int(args.sampling_seed) + 7919)
    bootstrap_indices = (
        rng.integers(0, int(args.n_samples), size=(int(args.n_bootstrap), int(args.n_samples)))
        if int(args.n_bootstrap) > 0
        else None
    )

    rows: list[dict[str, object]] = []
    for seed in seeds:
        cache_path = overall_prediction_cache_path(overall_cache_dir, seed=seed, args=args)
        if bool(args.prediction_cache) and cache_path.exists():
            print(f"[full-history pair syn seed {seed}] loading prediction cache {cache_path}", file=sys.stderr, flush=True)
            all_mode_targets = load_full_history_prediction_cache(cache_path, n_samples=int(args.n_samples))
        else:
            print(f"[full-history pair syn seed {seed}] cache missing; loading {checkpoint_paths[seed]}", file=sys.stderr, flush=True)
            model = load_unicm_model(checkpoint_paths[seed], args.device)
            all_mode_targets = predict_modeformer_all_modes_from_history(
                model,
                history_modes,
                device=args.device,
                batch_size=args.batch_size,
                start_month=args.start_month,
            )
            if not np.isfinite(all_mode_targets).all():
                raise ValueError(f"Model prediction contains non-finite values for seed {seed}.")
            if bool(args.prediction_cache):
                np.savez(
                    cache_path,
                    all_mode_targets=all_mode_targets.astype(np.float32),
                    metadata=json.dumps(
                        {
                            "seed": int(seed),
                            "n_samples": int(args.n_samples),
                            "sampling_seed": int(args.sampling_seed),
                            "intervention_bound": float(args.intervention_bound),
                            "sampling_mode": "full_history_max_entropy",
                            "history_shape": list(history_modes.shape[1:]),
                            "start_month": int(args.start_month),
                            "device": str(args.device),
                        },
                        sort_keys=True,
                    ),
                )
            del model

        for target_name in target_names:
            print(f"  target {target_name}: {len(source_pairs)} source pairs", file=sys.stderr, flush=True)
            target_index = MODE_NAMES[target_name]
            targets = all_mode_targets[:, :, target_index]
            for left_name, right_name in source_pairs:
                pair_key = f"{left_name}|{right_name}"
                for lead in leads:
                    summary = summarize_full_history_mode_pair_syn(
                        history_modes,
                        left_name,
                        right_name,
                        targets[:, [int(lead) - 1]],
                        bootstrap_indices=bootstrap_indices,
                    )
                    rows.append(
                        {
                            "seed": int(seed),
                            "checkpoint": str(checkpoint_paths[seed]),
                            "pair": pair_key,
                            "left_source": left_name,
                            "right_source": right_name,
                            "target": target_name,
                            "lead": int(lead),
                            "n_samples": int(args.n_samples),
                            "n_bootstrap": int(args.n_bootstrap),
                            "intervention_bound": float(args.intervention_bound),
                            "sampling_seed": int(args.sampling_seed),
                            "sampling_mode": "full_history_max_entropy",
                            **summary,
                        }
                    )

    row_path = output_dir / "full_history_mode_pair_syn_rows.jsonl"
    write_jsonl(rows, row_path)
    frame = pd.DataFrame(rows)
    pair_summary = summarize_full_history_pair_syn(frame)
    summary_path = output_dir / "full_history_mode_pair_syn_summary.csv"
    pair_summary.to_csv(summary_path, index=False)
    lead_summary = summarize_full_history_pair_seed_leads(frame)
    lead_summary_path = output_dir / "full_history_mode_pair_syn_lead_summary.csv"
    lead_summary.to_csv(lead_summary_path, index=False)
    top_pairs = pair_summary[pair_summary["rank_within_target"] <= int(args.top_k)]
    top_pairs_path = output_dir / "full_history_mode_pair_syn_top_pairs.csv"
    top_pairs.to_csv(top_pairs_path, index=False)

    figure_paths = plot_full_history_pair_syn_top(
        pair_summary,
        lead_summary,
        fig_dir / "full_history_mode_pair_syn_top",
        top_k=int(args.top_k),
    )
    markdown = build_full_history_pair_syn_report_markdown(
        pair_summary,
        output_dir=output_dir,
        overall_cache_dir=overall_cache_dir,
        figure_paths=[path for path in figure_paths if path.suffix == ".png"],
        n_samples=int(args.n_samples),
        n_bootstrap=int(args.n_bootstrap),
        seeds=seeds,
        intervention_bound=float(args.intervention_bound),
        sampling_seed=int(args.sampling_seed),
        start_month=int(args.start_month),
        source_modes=source_modes,
        target_names=target_names,
    )
    report_path.write_text(markdown, encoding="utf-8")
    return {
        "rows": str(row_path),
        "summary": str(summary_path),
        "lead_summary": str(lead_summary_path),
        "top_pairs": str(top_pairs_path),
        "report": str(report_path),
        "figures": [str(path) for path in figure_paths],
    }


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    import torch

    if args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))

    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "fig"
    cache_dir = output_dir / "cache"
    report_path = Path(args.report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = [int(seed) for seed in args.seeds]
    relations = [RELATIONS[key] for key in args.relations]
    leads = parse_leads(args.leads)
    checkpoint_paths = resolve_checkpoint_paths(Path(args.checkpoint_root), seeds)
    rng = np.random.default_rng(int(args.sampling_seed))
    left_source = rng.uniform(-float(args.intervention_bound), float(args.intervention_bound), size=(args.n_samples, 1))
    right_source = rng.uniform(-float(args.intervention_bound), float(args.intervention_bound), size=(args.n_samples, 1))
    bootstrap_indices = (
        rng.integers(0, int(args.n_samples), size=(int(args.n_bootstrap), int(args.n_samples)))
        if int(args.n_bootstrap) > 0
        else None
    )
    print("[setup] building source density cache", file=sys.stderr, flush=True)
    source_log_cache = build_source_log_cache(left_source, right_source, bootstrap_indices=bootstrap_indices)
    relation_groups = group_relations_by_sources(relations)

    rows: list[dict[str, object]] = []
    total_jobs = len(seeds) * len(relation_groups)
    job_index = 0
    for seed in seeds:
        print(f"[seed {seed}] loading {checkpoint_paths[seed]}", file=sys.stderr, flush=True)
        model = load_unicm_model(checkpoint_paths[seed], args.device)
        for (left_name, right_name), group in relation_groups.items():
            job_index += 1
            print(
                f"[{job_index}/{total_jobs}] seed {seed}: forward source pair {left_name}+{right_name}",
                file=sys.stderr,
                flush=True,
            )
            source_relation = group[0]
            cache_path = prediction_cache_path(
                cache_dir,
                seed=seed,
                left_name=left_name,
                right_name=right_name,
                args=args,
            )
            if bool(args.prediction_cache) and cache_path.exists():
                print(f"  loading prediction cache {cache_path}", file=sys.stderr, flush=True)
                with np.load(cache_path) as payload:
                    all_mode_targets = payload["all_mode_targets"].astype(float)
                if not np.isfinite(all_mode_targets).all():
                    raise ValueError(f"Prediction cache contains non-finite values: {cache_path}")
            else:
                all_mode_targets = predict_modeformer_all_modes(
                    model,
                    left_source,
                    right_source,
                    source_relation,
                    device=args.device,
                    batch_size=args.batch_size,
                    history_months=args.history_months,
                    start_month=args.start_month,
                )
                if not np.isfinite(all_mode_targets).all():
                    raise ValueError(
                        f"Model prediction contains non-finite values for seed {seed}, "
                        f"source pair {left_name}+{right_name}."
                    )
                if bool(args.prediction_cache):
                    np.savez(
                        cache_path,
                        all_mode_targets=all_mode_targets.astype(np.float32),
                        metadata=json.dumps(
                            {
                                "seed": int(seed),
                                "left": left_name,
                                "right": right_name,
                                "n_samples": int(args.n_samples),
                                "sampling_seed": int(args.sampling_seed),
                                "intervention_bound": float(args.intervention_bound),
                                "history_months": int(args.history_months),
                                "start_month": int(args.start_month),
                                "device": str(args.device),
                            },
                            sort_keys=True,
                        ),
                    )
            for relation in group:
                print(f"  PEID/Syn {relation.label}", file=sys.stderr, flush=True)
                targets = all_mode_targets[:, :, relation.target_index]
                for lead in leads:
                    summary = summarize_syn_for_target(
                        left_source,
                        right_source,
                        targets[:, [int(lead) - 1]],
                        bootstrap_indices=bootstrap_indices,
                        source_log_cache=source_log_cache,
                    )
                    rows.append(
                        {
                            "seed": int(seed),
                            "checkpoint": str(checkpoint_paths[seed]),
                            "relation": relation.key,
                            "relation_label": relation.label,
                            "left_source": relation.left,
                            "right_source": relation.right,
                            "target": relation.target,
                            "lead": int(lead),
                            "n_samples": int(args.n_samples),
                            "n_bootstrap": int(args.n_bootstrap),
                            "intervention_bound": float(args.intervention_bound),
                            "history_months": int(args.history_months),
                            "sampling_seed": int(args.sampling_seed),
                            **summary,
                        }
                    )
        del model

    row_path = output_dir / "seed_robustness_rows.jsonl"
    write_jsonl(rows, row_path)
    frame = pd.DataFrame(rows)
    robustness_summary = compute_relation_robustness(frame)
    robustness_path = output_dir / "seed_robustness_summary.csv"
    robustness_summary.to_csv(robustness_path, index=False)
    lead_summary_path = output_dir / "seed_lead_summary.csv"
    summarize_seed_leads(frame).to_csv(lead_summary_path, index=False)
    rank_frame, rank_summary = summarize_relation_ranks(frame)
    rank_path = output_dir / "relation_rank_by_seed.csv"
    rank_frame.to_csv(rank_path, index=False)
    (output_dir / "relation_rank_summary.json").write_text(
        json.dumps(rank_summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    figure_paths: list[Path] = []
    figure_paths.extend(plot_seed_overlay(frame, fig_dir / "seed_overlay_syn_leads"))
    figure_paths.extend(plot_seed_mean_std(frame, fig_dir / "seed_mean_std_syn_leads"))
    figure_paths.extend(plot_rank_heatmap(rank_frame, fig_dir / "seed_rank_consistency_heatmap"))
    markdown = build_report_markdown(
        robustness_summary,
        rank_summary,
        output_dir=output_dir,
        figure_paths=[path for path in figure_paths if path.suffix == ".png"],
        n_samples=int(args.n_samples),
        n_bootstrap=int(args.n_bootstrap),
        seeds=seeds,
        intervention_bound=float(args.intervention_bound),
        history_months=int(args.history_months),
        sampling_seed=int(args.sampling_seed),
        start_month=int(args.start_month),
    )
    report_path.write_text(markdown, encoding="utf-8")
    return {
        "rows": str(row_path),
        "robustness_summary": str(robustness_path),
        "rank_by_seed": str(rank_path),
        "report": str(report_path),
        "figures": [str(path) for path in figure_paths],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run UniCM Modeformer PEID/Syn seed robustness analysis.")
    parser.add_argument("--analysis-mode", choices=["pair-peid", "overall-ei", "full-history-pair-syn"], default="pair-peid")
    parser.add_argument("--robustness", choices=["seed"], default="seed")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "data" / "UniCM-checkpoint" / "src" / "experiments")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "unicm_peid_syn")
    parser.add_argument("--report-path", type=Path, default=ROOT / "docs" / "reports" / "unicm_peid_seed_robustness.md")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--relations", nargs="+", default=list(RELATIONS))
    parser.add_argument("--overall-targets", nargs="*", default=None)
    parser.add_argument("--source-modes", nargs="*", default=None)
    parser.add_argument("--leads", nargs="*", default=None)
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--sampling-seed", type=int, default=20260619)
    parser.add_argument("--intervention-bound", type=float, default=2.0)
    parser.add_argument("--history-months", type=int, default=3)
    parser.add_argument("--start-month", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--overall-cache-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-prediction-cache", action="store_false", dest="prediction_cache")
    parser.set_defaults(prediction_cache=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    unknown = [key for key in args.relations if key not in RELATIONS]
    if unknown:
        parser.error(f"Unknown relation key(s): {', '.join(unknown)}")
    if args.analysis_mode == "overall-ei":
        outputs = run_overall_ei_analysis(args)
    elif args.analysis_mode == "full-history-pair-syn":
        outputs = run_full_history_mode_pair_syn_analysis(args)
    else:
        outputs = run_analysis(args)
    print(json.dumps(outputs, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
