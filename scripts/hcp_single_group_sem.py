#!/usr/bin/env python3
"""Python translation of the HCP single-group latent-mean SEM.

This model-specific implementation keeps the supplied lavaan factor structure
but treats all subjects as one sample. It uses raw-data full-information
maximum likelihood (FIML), so rows with partial indicator data are retained.
No external group file or group-equality test is required.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import OptimizeResult, minimize
from scipy.stats import chi2


ZSCORE_COLUMNS = (
    "PMAT24_A_CRz",
    "VSPLOT_TCz",
    "ListSort_Unadjz",
    "PicSeq_Unadjz",
    "IWRD_TOTz",
    "PicVocab_Unadjz",
    "ReadEng_Unadjz",
    "CardSort_Unadjz",
    "Flanker_Unadjz",
    "ProcSpeed_Unadjz",
    "Language_Task_Story_Accz",
    "Language_Task_Math_Accz",
    "WM_Task_2bk_Place_Accz",
    "WM_Task_2bk_Tool_Accz",
    "WM_Task_2bk_Body_Accz",
    "Relational_Task_Match_Accz",
    "Relational_Task_Rel_Accz",
    "Loneliness_Unadjz",
)

RAW_SCORE_COLUMNS = (
    "PMAT24_A_CR",
    "VSPLOT_TC",
    "ListSort_Unadj",
    "PicSeq_Unadj",
    "IWRD_TOT",
    "PicVocab_Unadj",
    "ReadEng_Unadj",
    "CardSort_Unadj",
    "Flanker_Unadj",
    "ProcSpeed_Unadj",
    "Language_Task_Story_Acc",
    "Language_Task_Math_Acc",
    "WM_Task_2bk_Place_Acc",
    "WM_Task_2bk_Tool_Acc",
    "WM_Task_2bk_Body_Acc",
    "Relational_Task_Match_Acc",
    "Relational_Task_Rel_Acc",
    "Loneliness_Unadj",
)

# Only these nine standardized indicators enter ModelMG in the R code.
INDICATORS = (
    "PMAT24_A_CRz",
    "VSPLOT_TCz",
    "PicVocab_Unadjz",
    "ReadEng_Unadjz",
    "PicSeq_Unadjz",
    "IWRD_TOTz",
    "CardSort_Unadjz",
    "Flanker_Unadjz",
    "ProcSpeed_Unadjz",
)
FACTORS = ("g", "cry", "mem", "spd")

# These observed intercepts are fixed at zero in the lavaan syntax.
ANCHOR_INDICES = (0, 2, 4, 6)
FREE_INTERCEPT_INDICES = tuple(i for i in range(len(INDICATORS)) if i not in ANCHOR_INDICES)


@dataclass(frozen=True)
class MissingPattern:
    indices: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class GroupData:
    label: str
    n_rows: int
    patterns: tuple[MissingPattern, ...]
    observed_means: np.ndarray


@dataclass(frozen=True)
class ModelSpec:
    groups: tuple[GroupData, ...]
    equal_latent_means: bool

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def group_block_size(self) -> int:
        # g loadings + spd loadings + factor variances + residual variances + free intercepts
        return 8 + 2 + 4 + 9 + 5

    @property
    def n_parameters(self) -> int:
        mean_parameters = 4 if self.equal_latent_means else 4 * self.n_groups
        return 2 + self.group_block_size * self.n_groups + mean_parameters


@dataclass
class FittedModel:
    name: str
    spec: ModelSpec
    result: OptimizeResult
    log_likelihood: float
    aic: float
    bic: float
    parameters: pd.DataFrame


def zscore_like_r(frame: pd.DataFrame) -> pd.DataFrame:
    """Match R's scale(): subtract column mean and divide by sample SD (n - 1)."""
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    means = numeric.mean(axis=0, skipna=True)
    standard_deviations = numeric.std(axis=0, skipna=True, ddof=1)
    invalid = standard_deviations.index[(~np.isfinite(standard_deviations)) | (standard_deviations <= 0)]
    if len(invalid):
        raise ValueError(f"Cannot z-score constant or empty columns: {', '.join(map(str, invalid))}")
    return (numeric - means) / standard_deviations


def read_input_table(path: Path, *, sheet_name: str | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    return pd.read_csv(path, low_memory=False)


def load_and_prepare_data(
    behavioral_path: Path,
    brain_path: Path,
    *,
    group_column: str,
    behavioral_sheet: str | None = None,
    brain_sheet: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reproduce read.table, scale, cbind, merge, and Group assignment from R."""
    behavioral = read_input_table(behavioral_path, sheet_name=behavioral_sheet)
    if "Subject" not in behavioral.columns:
        raise ValueError(f"{behavioral_path} has no Subject column.")
    if behavioral.shape[1] < 21:
        raise ValueError("Behavioral input needs at least 21 columns because R scales columns 4:21.")
    if behavioral["Subject"].astype(str).duplicated().any():
        raise ValueError("Behavioral input contains duplicate Subject values.")
    duplicated_output_names = set(ZSCORE_COLUMNS) & set(behavioral.columns)
    if duplicated_output_names:
        raise ValueError(
            "Behavioral input already contains standardized output columns: "
            + ", ".join(sorted(duplicated_output_names))
        )

    if set(RAW_SCORE_COLUMNS).issubset(behavioral.columns):
        zscore_source_columns = list(RAW_SCORE_COLUMNS)
        selection_method = "column names"
    else:
        # R's 4:21 is inclusive and one-based; pandas iloc[:, 3:21] is equivalent.
        zscore_source_columns = list(behavioral.columns[3:21])
        selection_method = "legacy positions 4:21"
    standardized = zscore_like_r(behavioral.loc[:, zscore_source_columns].copy())
    standardized.columns = list(ZSCORE_COLUMNS)
    behavioral_augmented = pd.concat(
        [behavioral.reset_index(drop=True), standardized.reset_index(drop=True)], axis=1
    )

    brain = read_input_table(brain_path, sheet_name=brain_sheet)
    required_brain = {"Subject", group_column}
    missing_brain = required_brain - set(brain.columns)
    if missing_brain:
        raise ValueError(f"{brain_path} is missing: {', '.join(sorted(missing_brain))}")
    if brain["Subject"].astype(str).duplicated().any():
        raise ValueError("Brain-measure input contains duplicate Subject values.")

    # Normalizing Subject to string prevents 100206 and "100206" from failing to merge.
    behavioral_augmented["Subject"] = behavioral_augmented["Subject"].astype(str)
    brain["Subject"] = brain["Subject"].astype(str)
    merged = behavioral_augmented.merge(brain, on="Subject", how="inner", validate="one_to_one")
    merged["Group"] = merged[group_column]

    missing_indicators = set(INDICATORS) - set(merged.columns)
    if missing_indicators:
        raise ValueError(f"Prepared data is missing indicators: {', '.join(sorted(missing_indicators))}")
    indicator_columns = list(INDICATORS)
    merged.loc[:, indicator_columns] = merged.loc[:, indicator_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    before_group_filter = len(merged)
    merged = merged.loc[merged["Group"].notna()].copy()
    merged["Group"] = merged["Group"].astype(str)
    if merged["Group"].nunique() < 2:
        raise ValueError("Multigroup SEM requires at least two nonempty Group values.")

    all_missing = merged.loc[:, indicator_columns].isna().all(axis=1)
    removed_all_missing = int(all_missing.sum())
    merged = merged.loc[~all_missing].copy()
    group_sizes = merged["Group"].value_counts().sort_index()
    if (group_sizes < 5).any():
        too_small = group_sizes[group_sizes < 5].to_dict()
        raise ValueError(f"Every group needs at least five usable rows; too small: {too_small}")

    audit = {
        "behavioral_rows": int(len(behavioral)),
        "brain_rows": int(len(brain)),
        "zscore_selection_method": selection_method,
        "zscore_source_columns": zscore_source_columns,
        "zscore_output_columns": list(ZSCORE_COLUMNS),
        "inner_join_rows": int(before_group_filter),
        "rows_without_group_removed": int(before_group_filter - len(merged) - removed_all_missing),
        "rows_with_all_sem_indicators_missing_removed": removed_all_missing,
        "analysis_rows": int(len(merged)),
        "group_sizes": {str(key): int(value) for key, value in group_sizes.items()},
        "indicator_missing_fraction": {
            column: float(merged[column].isna().mean()) for column in INDICATORS
        },
    }
    return merged, audit


def load_and_prepare_single_group_data(
    behavioral_path: Path,
    *,
    behavioral_sheet: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, standardize, and validate one behavioral sample for SEM."""
    behavioral = read_input_table(behavioral_path, sheet_name=behavioral_sheet)
    if "Subject" not in behavioral.columns:
        raise ValueError(f"{behavioral_path} has no Subject column.")
    if behavioral.shape[1] < 21:
        raise ValueError("Behavioral input needs at least 21 columns.")
    behavioral["Subject"] = behavioral["Subject"].astype(str)
    if behavioral["Subject"].duplicated().any():
        raise ValueError("Behavioral input contains duplicate Subject values.")
    duplicated_output_names = set(ZSCORE_COLUMNS) & set(behavioral.columns)
    if duplicated_output_names:
        raise ValueError(
            "Behavioral input already contains standardized output columns: "
            + ", ".join(sorted(duplicated_output_names))
        )

    if set(RAW_SCORE_COLUMNS).issubset(behavioral.columns):
        zscore_source_columns = list(RAW_SCORE_COLUMNS)
        selection_method = "column names"
    else:
        zscore_source_columns = list(behavioral.columns[3:21])
        selection_method = "legacy positions 4:21"
    standardized = zscore_like_r(behavioral.loc[:, zscore_source_columns].copy())
    standardized.columns = list(ZSCORE_COLUMNS)
    prepared = pd.concat(
        [behavioral.reset_index(drop=True), standardized.reset_index(drop=True)], axis=1
    )
    prepared["Group"] = "all"
    indicator_columns = list(INDICATORS)
    prepared.loc[:, indicator_columns] = prepared.loc[:, indicator_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    all_missing = prepared.loc[:, indicator_columns].isna().all(axis=1)
    removed_all_missing = int(all_missing.sum())
    prepared = prepared.loc[~all_missing].copy()
    if len(prepared) < 10:
        raise ValueError("Single-group SEM needs at least 10 usable rows.")
    entirely_missing = [column for column in INDICATORS if prepared[column].isna().all()]
    if entirely_missing:
        raise ValueError(
            "Indicators with no observations: " + ", ".join(entirely_missing)
        )

    audit = {
        "behavioral_rows": int(len(behavioral)),
        "analysis_rows": int(len(prepared)),
        "rows_with_all_sem_indicators_missing_removed": removed_all_missing,
        "zscore_selection_method": selection_method,
        "zscore_source_columns": zscore_source_columns,
        "zscore_output_columns": list(ZSCORE_COLUMNS),
        "indicator_missing_fraction": {
            column: float(prepared[column].isna().mean()) for column in INDICATORS
        },
    }
    return prepared, audit


def _missing_patterns(values: np.ndarray) -> tuple[MissingPattern, ...]:
    masks: dict[tuple[bool, ...], list[np.ndarray]] = {}
    for row in values:
        key = tuple(bool(value) for value in np.isfinite(row))
        if any(key):
            masks.setdefault(key, []).append(row[np.asarray(key, dtype=bool)])
    return tuple(
        MissingPattern(
            indices=np.flatnonzero(np.asarray(mask, dtype=bool)),
            values=np.asarray(rows, dtype=float),
        )
        for mask, rows in masks.items()
    )


def build_group_data(frame: pd.DataFrame) -> tuple[GroupData, ...]:
    groups: list[GroupData] = []
    for label, subset in frame.groupby("Group", sort=True):
        values = subset.loc[:, list(INDICATORS)].to_numpy(dtype=float)
        entirely_missing = [
            indicator for indicator, missing in zip(INDICATORS, np.isnan(values).all(axis=0)) if missing
        ]
        if entirely_missing:
            raise ValueError(
                f"Group {label} has indicators with no observations: {', '.join(entirely_missing)}"
            )
        groups.append(
            GroupData(
                label=str(label),
                n_rows=len(subset),
                patterns=_missing_patterns(values),
                observed_means=np.nanmean(values, axis=0),
            )
        )
    return tuple(groups)


def initial_parameters(spec: ModelSpec) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Construct a stable starting point and L-BFGS-B bounds."""
    parameters: list[float] = [0.7, 0.7]  # shared b and c loadings
    bounds: list[tuple[float, float]] = [(-5.0, 5.0), (-5.0, 5.0)]
    for group in spec.groups:
        parameters.extend([0.7] * 8)  # free g loadings; PMAT loading is fixed to 1
        parameters.extend([0.7, 0.7])  # Flanker and ProcSpeed loadings on spd
        parameters.extend([math.log(0.5)] * 4)
        parameters.extend([math.log(0.5)] * 9)
        parameters.extend(group.observed_means[list(FREE_INTERCEPT_INDICES)])
        bounds.extend([(-5.0, 5.0)] * 10)
        bounds.extend([(-10.0, 5.0)] * 13)
        bounds.extend([(-10.0, 10.0)] * 5)
    n_mean_blocks = 1 if spec.equal_latent_means else spec.n_groups
    parameters.extend([0.0] * (4 * n_mean_blocks))
    bounds.extend([(-10.0, 10.0)] * (4 * n_mean_blocks))
    vector = np.asarray(parameters, dtype=float)
    if len(vector) != spec.n_parameters:
        raise RuntimeError("Internal parameter-count mismatch.")
    return vector, bounds


def unpack_parameters(
    vector: np.ndarray, spec: ModelSpec
) -> tuple[float, float, list[dict[str, np.ndarray]], list[np.ndarray]]:
    cursor = 0
    b = float(vector[cursor])
    c = float(vector[cursor + 1])
    cursor += 2
    blocks: list[dict[str, np.ndarray]] = []
    for _ in spec.groups:
        g_loadings = vector[cursor : cursor + 8]
        cursor += 8
        spd_loadings = vector[cursor : cursor + 2]
        cursor += 2
        factor_variances = np.exp(vector[cursor : cursor + 4])
        cursor += 4
        residual_variances = np.exp(vector[cursor : cursor + 9])
        cursor += 9
        free_intercepts = vector[cursor : cursor + 5]
        cursor += 5
        blocks.append(
            {
                "g_loadings": g_loadings,
                "spd_loadings": spd_loadings,
                "factor_variances": factor_variances,
                "residual_variances": residual_variances,
                "free_intercepts": free_intercepts,
            }
        )
    if spec.equal_latent_means:
        common = np.asarray(vector[cursor : cursor + 4], dtype=float)
        latent_means = [common for _ in spec.groups]
        cursor += 4
    else:
        latent_means = []
        for _ in spec.groups:
            latent_means.append(np.asarray(vector[cursor : cursor + 4], dtype=float))
            cursor += 4
    if cursor != len(vector):
        raise RuntimeError("Internal parameter unpacking mismatch.")
    return b, c, blocks, latent_means


def implied_moments(
    b: float,
    c: float,
    block: dict[str, np.ndarray],
    latent_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loadings = np.zeros((len(INDICATORS), len(FACTORS)), dtype=float)
    loadings[0, 0] = 1.0
    loadings[1:, 0] = block["g_loadings"]
    loadings[2, 1] = b
    loadings[3, 1] = b
    loadings[4, 2] = c
    loadings[5, 2] = c
    loadings[6, 3] = 1.0
    loadings[7:9, 3] = block["spd_loadings"]

    intercepts = np.zeros(len(INDICATORS), dtype=float)
    intercepts[list(FREE_INTERCEPT_INDICES)] = block["free_intercepts"]
    factor_covariance = np.diag(block["factor_variances"])
    residual_covariance = np.diag(block["residual_variances"])
    covariance = loadings @ factor_covariance @ loadings.T + residual_covariance
    mean = intercepts + loadings @ latent_means
    return mean, covariance, loadings


def negative_log_likelihood(vector: np.ndarray, spec: ModelSpec) -> float:
    b, c, blocks, latent_means = unpack_parameters(vector, spec)
    total = 0.0
    log_two_pi = math.log(2.0 * math.pi)
    for group, block, alpha in zip(spec.groups, blocks, latent_means):
        mean, covariance, _ = implied_moments(b, c, block, alpha)
        for pattern in group.patterns:
            indices = pattern.indices
            observed_covariance = covariance[np.ix_(indices, indices)]
            try:
                cholesky = np.linalg.cholesky(observed_covariance)
            except np.linalg.LinAlgError:
                return 1.0e100
            differences = pattern.values - mean[indices]
            solved = solve_triangular(cholesky, differences.T, lower=True, check_finite=False)
            quadratic = float(np.square(solved).sum())
            log_determinant = 2.0 * float(np.log(np.diag(cholesky)).sum())
            n_rows, n_variables = pattern.values.shape
            total += 0.5 * (
                n_rows * (n_variables * log_two_pi + log_determinant) + quadratic
            )
    return float(total) if np.isfinite(total) else 1.0e100


def parameter_table(vector: np.ndarray, spec: ModelSpec) -> pd.DataFrame:
    b, c, blocks, latent_means = unpack_parameters(vector, spec)
    rows: list[dict[str, Any]] = []
    for group, block, alpha in zip(spec.groups, blocks, latent_means):
        mean, covariance, loadings = implied_moments(b, c, block, alpha)
        observed_sd = np.sqrt(np.diag(covariance))
        factor_sd = np.sqrt(block["factor_variances"])
        for indicator_index, indicator in enumerate(INDICATORS):
            for factor_index, factor in enumerate(FACTORS):
                estimate = float(loadings[indicator_index, factor_index])
                if estimate != 0.0:
                    standardized = estimate * factor_sd[factor_index] / observed_sd[indicator_index]
                    rows.append(
                        {
                            "group": group.label,
                            "parameter_type": "loading",
                            "lhs": factor,
                            "operator": "=~",
                            "rhs": indicator,
                            "estimate": estimate,
                            "standardized_estimate": float(standardized),
                        }
                    )
        intercepts = mean - loadings @ alpha
        for index, indicator in enumerate(INDICATORS):
            rows.append(
                {
                    "group": group.label,
                    "parameter_type": "observed_intercept",
                    "lhs": indicator,
                    "operator": "~",
                    "rhs": "1",
                    "estimate": float(intercepts[index]),
                    "standardized_estimate": float(intercepts[index] / observed_sd[index]),
                }
            )
            rows.append(
                {
                    "group": group.label,
                    "parameter_type": "residual_variance",
                    "lhs": indicator,
                    "operator": "~~",
                    "rhs": indicator,
                    "estimate": float(block["residual_variances"][index]),
                    "standardized_estimate": float(
                        block["residual_variances"][index] / covariance[index, index]
                    ),
                }
            )
        for index, factor in enumerate(FACTORS):
            rows.append(
                {
                    "group": group.label,
                    "parameter_type": "latent_mean",
                    "lhs": factor,
                    "operator": "~",
                    "rhs": "1",
                    "estimate": float(alpha[index]),
                    "standardized_estimate": float(alpha[index] / factor_sd[index]),
                }
            )
            rows.append(
                {
                    "group": group.label,
                    "parameter_type": "factor_variance",
                    "lhs": factor,
                    "operator": "~~",
                    "rhs": factor,
                    "estimate": float(block["factor_variances"][index]),
                    "standardized_estimate": 1.0,
                }
            )
    return pd.DataFrame(rows)


def factor_scores(model: FittedModel, data: pd.DataFrame) -> pd.DataFrame:
    """Return empirical-Bayes regression scores E[eta | observed indicators]."""
    if len(model.spec.groups) != 1:
        raise ValueError("factor_scores currently supports one fitted group.")
    b, c, blocks, latent_means = unpack_parameters(
        np.asarray(model.result.x, dtype=float), model.spec
    )
    block = blocks[0]
    alpha = latent_means[0]
    mean, covariance, loadings = implied_moments(b, c, block, alpha)
    factor_covariance = np.diag(block["factor_variances"])
    values = data.loc[:, list(INDICATORS)].to_numpy(dtype=float)
    rows: list[list[Any]] = []
    for subject, row in zip(data["Subject"].astype(str), values):
        observed = np.isfinite(row)
        indices = np.flatnonzero(observed)
        if len(indices) == 0:
            scores = np.full(len(FACTORS), np.nan)
        else:
            observed_covariance = covariance[np.ix_(indices, indices)]
            cross_covariance = factor_covariance @ loadings[indices, :].T
            residual = row[indices] - mean[indices]
            scores = alpha + cross_covariance @ np.linalg.solve(observed_covariance, residual)
        rows.append([subject, *[float(value) for value in scores]])
    return pd.DataFrame(rows, columns=["Subject", *[f"{factor}_score" for factor in FACTORS]])


def fit_model(
    name: str,
    spec: ModelSpec,
    *,
    starts: int,
    seed: int,
    maxiter: int,
) -> FittedModel:
    initial, bounds = initial_parameters(spec)
    rng = np.random.default_rng(seed)
    candidates: list[OptimizeResult] = []
    for start in range(starts):
        candidate = initial.copy()
        if start:
            candidate += rng.normal(0.0, 0.08, size=len(candidate))
            candidate = np.asarray(
                [np.clip(value, lower, upper) for value, (lower, upper) in zip(candidate, bounds)]
            )
        result = minimize(
            negative_log_likelihood,
            candidate,
            args=(spec,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": 1.0e-10, "gtol": 1.0e-6, "maxls": 50},
        )
        candidates.append(result)
    result = min(candidates, key=lambda item: float(item.fun))
    if not np.isfinite(result.fun):
        raise RuntimeError(f"{name} failed: non-finite objective.")
    log_likelihood = -float(result.fun)
    n_rows = sum(group.n_rows for group in spec.groups)
    n_parameters = spec.n_parameters
    return FittedModel(
        name=name,
        spec=spec,
        result=result,
        log_likelihood=log_likelihood,
        aic=float(2 * n_parameters - 2 * log_likelihood),
        bic=float(math.log(n_rows) * n_parameters - 2 * log_likelihood),
        parameters=parameter_table(np.asarray(result.x, dtype=float), spec),
    )


def model_summary(model: FittedModel) -> dict[str, Any]:
    return {
        "name": model.name,
        "equal_latent_means": model.spec.equal_latent_means,
        "converged": bool(model.result.success),
        "optimizer_message": str(model.result.message),
        "iterations": int(getattr(model.result, "nit", -1)),
        "n_parameters": int(model.spec.n_parameters),
        "log_likelihood": model.log_likelihood,
        "aic": model.aic,
        "bic": model.bic,
    }


def likelihood_ratio_test(free: FittedModel, equal: FittedModel) -> dict[str, Any]:
    degrees_of_freedom = free.spec.n_parameters - equal.spec.n_parameters
    statistic = max(0.0, 2.0 * (free.log_likelihood - equal.log_likelihood))
    return {
        "comparison": "free latent means vs equal latent means",
        "chi_square": float(statistic),
        "df": int(degrees_of_freedom),
        "p_value": float(chi2.sf(statistic, degrees_of_freedom)),
        "interpretation": (
            "Rejecting the equal-means model indicates that at least one latent factor mean differs "
            "between groups."
        ),
    }


def plot_sem_paths(model: FittedModel, output_dir: Path) -> list[str]:
    """Create one uncluttered path diagram per group using standardized loadings."""
    import matplotlib.pyplot as plt

    paths: list[str] = []
    loading_rows = model.parameters.loc[model.parameters["parameter_type"] == "loading"]
    factor_x = {factor: index for index, factor in enumerate(FACTORS)}
    indicator_x = {indicator: index for index, indicator in enumerate(INDICATORS)}
    for group in (item.label for item in model.spec.groups):
        group_rows = loading_rows.loc[loading_rows["group"] == group]
        figure, axis = plt.subplots(figsize=(15, 6), constrained_layout=True)
        axis.set_xlim(-0.7, len(INDICATORS) - 0.3)
        axis.set_ylim(-0.35, 1.35)
        axis.axis("off")
        factor_positions = {
            factor: (np.linspace(0.8, len(INDICATORS) - 1.2, len(FACTORS))[factor_x[factor]], 1.0)
            for factor in FACTORS
        }
        indicator_positions = {indicator: (indicator_x[indicator], 0.0) for indicator in INDICATORS}
        for factor, (x, y) in factor_positions.items():
            axis.text(
                x,
                y,
                factor,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                bbox={"boxstyle": "circle,pad=0.55", "facecolor": "#D9EAF7", "edgecolor": "#17365D"},
            )
        for indicator, (x, y) in indicator_positions.items():
            display = indicator.removesuffix("z").replace("_", "\n")
            axis.text(
                x,
                y,
                display,
                ha="center",
                va="center",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F2F2F2", "edgecolor": "#666666"},
            )
        for _, row in group_rows.iterrows():
            start = factor_positions[str(row["lhs"])]
            end = indicator_positions[str(row["rhs"])]
            axis.annotate(
                "",
                xy=(end[0], end[1] + 0.12),
                xytext=(start[0], start[1] - 0.12),
                arrowprops={"arrowstyle": "->", "color": "#4F6D7A", "lw": 1.0, "alpha": 0.8},
            )
            midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
            axis.text(
                midpoint[0],
                midpoint[1],
                f"{float(row['standardized_estimate']):.2f}",
                fontsize=7,
                color="#17365D",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.5},
            )
        title = "Single-group SEM standardized loadings" if len(model.spec.groups) == 1 else (
            f"Multigroup SEM standardized loadings — Group {group}"
        )
        axis.set_title(title, fontsize=14)
        safe_group = "".join(character if character.isalnum() or character in "-_" else "_" for character in group)
        destination = output_dir / f"sem_paths_group_{safe_group}.png"
        figure.savefig(destination, dpi=220, bbox_inches="tight")
        plt.close(figure)
        paths.append(str(destination))
    return paths


def write_outputs(
    output_dir: Path,
    data: pd.DataFrame,
    audit: dict[str, Any],
    free: FittedModel,
    equal: FittedModel,
    comparison: dict[str, Any],
    *,
    make_plots: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    free_parameter_path = output_dir / "parameters_free_latent_means.csv"
    equal_parameter_path = output_dir / "parameters_equal_latent_means.csv"
    free.parameters.to_csv(free_parameter_path, index=False)
    equal.parameters.to_csv(equal_parameter_path, index=False)
    analysis_columns = ["Subject", "Group", *INDICATORS]
    data.loc[:, analysis_columns].to_csv(output_dir / "analysis_data.csv", index=False)
    figures = plot_sem_paths(free, output_dir) if make_plots else []
    payload = {
        "data_audit": audit,
        "model_definition": {
            "indicators": list(INDICATORS),
            "factors": list(FACTORS),
            "orthogonal_factors": True,
            "fixed_zero_observed_intercepts": [INDICATORS[index] for index in ANCHOR_INDICES],
            "shared_loading_b": ["PicVocab_Unadjz", "ReadEng_Unadjz"],
            "shared_loading_c": ["PicSeq_Unadjz", "IWRD_TOTz"],
            "missing_data": "raw-data FIML under multivariate normality",
        },
        "free_latent_means_model": model_summary(free),
        "equal_latent_means_model": model_summary(equal),
        "likelihood_ratio_test": comparison,
        "figures": figures,
        "limitations": [
            "This model-specific implementation reports likelihood, AIC, BIC, standardized loadings, and the nested LRT.",
            "It does not reproduce lavaan's full robust fit-index and standard-error table.",
            "The original labels b and c are treated as equality constraints within and across groups, matching lavaan multigroup label behavior.",
        ],
    }
    result_path = output_dir / "sem_results.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_single_group_outputs(
    output_dir: Path,
    data: pd.DataFrame,
    audit: dict[str, Any],
    model: FittedModel,
    *,
    make_plots: bool,
    selected_subjects: Sequence[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.parameters.to_csv(output_dir / "parameters.csv", index=False)
    analysis_columns = ["Subject", *INDICATORS]
    analysis_data = data.loc[:, analysis_columns].copy()
    analysis_data.to_csv(output_dir / "analysis_data.csv", index=False)
    scores = factor_scores(model, data)
    scores.to_csv(output_dir / "factor_scores_all_subjects.csv", index=False)
    selected_audit: dict[str, Any] | None = None
    if selected_subjects is not None:
        wanted = [str(subject) for subject in selected_subjects]
        if len(wanted) != len(set(wanted)):
            raise ValueError("Selected-subject list contains duplicate Subject values.")
        available = set(scores["Subject"])
        missing = [subject for subject in wanted if subject not in available]
        if missing:
            raise ValueError("Selected subjects missing after SEM preprocessing: " + ", ".join(missing))
        selected_order = pd.DataFrame({"Subject": wanted, "selection_order": range(1, len(wanted) + 1)})
        selected_results = (
            selected_order.merge(analysis_data, on="Subject", how="left", validate="one_to_one")
            .merge(scores, on="Subject", how="left", validate="one_to_one")
            .sort_values("selection_order")
            .drop(columns="selection_order")
        )
        selected_results.to_csv(output_dir / "selected_29_sem_results.csv", index=False)
        selected_audit = {
            "requested_subjects": len(wanted),
            "matched_subjects": int(len(selected_results)),
            "output": str(output_dir / "selected_29_sem_results.csv"),
        }
    figures = plot_sem_paths(model, output_dir) if make_plots else []
    payload = {
        "data_audit": audit,
        "model_definition": {
            "analysis_type": "single-group SEM",
            "indicators": list(INDICATORS),
            "factors": list(FACTORS),
            "orthogonal_factors": True,
            "fixed_zero_observed_intercepts": [INDICATORS[index] for index in ANCHOR_INDICES],
            "equal_loading_b": ["PicVocab_Unadjz", "ReadEng_Unadjz"],
            "equal_loading_c": ["PicSeq_Unadjz", "IWRD_TOTz"],
            "missing_data": "raw-data FIML under multivariate normality",
        },
        "model": model_summary(model),
        "factor_scores": {
            "method": "empirical-Bayes regression score E[latent factors | observed indicators]",
            "all_subjects_output": str(output_dir / "factor_scores_all_subjects.csv"),
            "selected_subjects": selected_audit,
        },
        "figures": figures,
        "limitations": [
            "With 29 subjects and 34 free parameters, estimates can be unstable and should be treated as exploratory.",
            "This implementation reports likelihood, AIC, BIC, and standardized loadings, but not lavaan's full standard-error and fit-index table.",
        ],
    }
    (output_dir / "sem_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the HCP orthogonal single-group SEM."
    )
    parser.add_argument(
        "--behavior",
        type=Path,
        default=Path("HCPS1200_behavioral_gfactor_modeling.csv"),
        help="Behavioral CSV/Excel file used as dat1 in the R script.",
    )
    parser.add_argument(
        "--behavior-sheet",
        default=None,
        help="Excel sheet for --behavior, for example Selected_29. Ignored for CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/hcp_single_group_sem"),
        help="Directory for JSON, parameter tables, analysis data, and path diagrams.",
    )
    parser.add_argument(
        "--selected-subjects",
        type=Path,
        default=None,
        help="Optional CSV/Excel containing Subject IDs to extract after fitting.",
    )
    parser.add_argument(
        "--selected-sheet",
        default=None,
        help="Excel sheet for --selected-subjects. Ignored for CSV.",
    )
    parser.add_argument("--starts", type=int, default=3, help="Optimizer restarts.")
    parser.add_argument("--maxiter", type=int, default=3000, help="Maximum L-BFGS-B iterations.")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--no-plots", action="store_true", help="Skip SEM path diagrams.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.starts < 1:
        raise ValueError("--starts must be at least 1.")
    data, audit = load_and_prepare_single_group_data(
        args.behavior.resolve(),
        behavioral_sheet=args.behavior_sheet,
    )
    groups = build_group_data(data)
    print(f"Prepared one sample with {len(data)} subjects.")
    selected_subjects: list[str] | None = None
    if args.selected_subjects is not None:
        selected_table = read_input_table(
            args.selected_subjects.resolve(), sheet_name=args.selected_sheet
        )
        if "Subject" not in selected_table.columns:
            raise ValueError("--selected-subjects input has no Subject column.")
        selected_subjects = selected_table["Subject"].astype(str).tolist()

    spec = ModelSpec(groups=groups, equal_latent_means=False)
    model = fit_model(
        "single_group_sem",
        spec,
        starts=args.starts,
        seed=args.seed,
        maxiter=args.maxiter,
    )
    payload = write_single_group_outputs(
        args.output_dir.resolve(),
        data,
        audit,
        model,
        make_plots=not args.no_plots,
        selected_subjects=selected_subjects,
    )
    print(json.dumps(payload["model"], ensure_ascii=False, indent=2))
    if not model.result.success:
        print("WARNING: optimizer did not report convergence; inspect sem_results.json.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
