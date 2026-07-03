from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Literal, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from sklearn.decomposition import PCA


DEFAULT_DATA_DIR = Path("data/ncep_reanalysis_slp")
RESULT_SUBDIR = Path("results/runge/2015_gateways")
FIG_SUBDIR = Path("fig/runge/2015_gateways")
# The local orthomax implementation follows the paper ordering convention, but
# a few paper-discussed modes are permuted relative to the published labels.
# These visual calibrations are based on the published Fig. 2/Fig. 4 component
# locations and keep the mapping bijective.
DEFAULT_PAPER_COMPONENT_LABEL_MAP: dict[int, int] = {
    7: 18,
    18: 7,
    8: 26,
    26: 8,
    21: 48,
    48: 21,
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


@dataclass(frozen=True)
class RungeConfig:
    mode: Literal["smoke", "full"] = "full"
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = Path(".")
    start_year: int = 1948
    end_year: int = 2012
    n_components: int = 60
    max_lag: int = 4
    pc_alpha: float = 0.001
    link_density: float = 0.2
    seed: int = 42
    causal_backend: Literal["tigramite", "regression"] = "tigramite"


@dataclass(frozen=True)
class LaggedEdge:
    source: int
    target: int
    lag: int
    coefficient: float
    p_value: float


@dataclass(frozen=True)
class SemEffects:
    direct_effects: pd.DataFrame
    total_effects: pd.DataFrame
    path_effects: pd.DataFrame
    gateway_scores: pd.DataFrame
    mediator_scores: pd.DataFrame
    coefficient_matrices: np.ndarray
    causal_effect_matrices: np.ndarray
    linear_coefficient_matrix: np.ndarray


def apply_paper_component_labels(
    frame: pd.DataFrame,
    label_map: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Attach paper-aligned component labels while preserving internal indices."""

    if "component" not in frame.columns:
        raise ValueError("frame must contain a 'component' column.")
    mapping = label_map if label_map is not None else DEFAULT_PAPER_COMPONENT_LABEL_MAP
    labelled = frame.copy()
    labelled["paper_component"] = labelled["component"].map(lambda value: int(mapping.get(int(value), int(value))))
    return labelled


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    root_candidate = _repo_root() / candidate
    return root_candidate if root_candidate.exists() else candidate.resolve()


def varimax(loadings: np.ndarray, *, gamma: float = 1.0, max_iter: int = 500, tol: float = np.finfo(np.float32).eps**0.5) -> tuple[np.ndarray, np.ndarray]:
    """Rotate columns using the orthomax/Varimax convention from Runge et al."""

    phi = np.asarray(loadings, dtype=float)
    if phi.ndim != 2:
        raise ValueError("loadings must be a two-dimensional array.")
    n_rows, n_cols = phi.shape
    if n_cols == 0:
        raise ValueError("loadings must contain at least one component.")

    rotation = np.eye(n_cols)
    rotated = phi.copy(order="C")
    previous = 0.0
    for _ in range(max_iter):
        old_previous = previous
        column_norms = np.sum(rotated**2, axis=0, keepdims=True)
        target = n_rows * rotated**3 - gamma * rotated * column_norms
        u, singular_values, vh = np.linalg.svd(rotated.T @ target, full_matrices=False)
        rotation = u @ vh
        previous = float(np.sum(singular_values))
        rotated = phi @ rotation
        if old_previous and abs(previous - old_previous) / max(previous, 1.0e-12) < tol:
            break

    for index in range(n_cols):
        if float(np.max(rotated[:, index])) < -float(np.min(rotated[:, index])):
            rotated[:, index] *= -1.0
            rotation[index, :] *= -1.0
    return rotated, rotation


def rotated_component_order(rotation: np.ndarray, eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot = np.asarray(rotation, dtype=float)
    values = np.asarray(eigenvalues, dtype=float)
    if rot.ndim != 2 or rot.shape[0] != rot.shape[1]:
        raise ValueError("rotation must be a square matrix.")
    if values.ndim != 1 or len(values) != rot.shape[0]:
        raise ValueError("eigenvalues length must match rotation dimensions.")
    rotated_covariance = rot.T @ np.diag(values) @ rot
    diagonal = np.diag(rotated_covariance)
    order = np.argsort(diagonal)[::-1]
    return order.astype(int), diagonal


def weekly_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()
    n_weeks = len(frame) // 7
    if n_weeks == 0:
        return frame.iloc[0:0].copy()
    trimmed = frame.iloc[: n_weeks * 7]
    values = trimmed.to_numpy(dtype=float).reshape(n_weeks, 7, frame.shape[1]).mean(axis=1)
    index = trimmed.index[::7][:n_weeks]
    weekly = pd.DataFrame(values, index=index, columns=frame.columns)
    weekly.index.name = "time"
    return weekly


def load_daily_slp(data_dir: str | Path, start_year: int, end_year: int) -> xr.DataArray:
    resolved = _resolve_path(data_dir)
    daily_dir = resolved / "daily"
    paths = [daily_dir / f"slp.{year}.nc" for year in range(int(start_year), int(end_year) + 1)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(f"Missing NCEP daily SLP file(s): {preview}")

    arrays: list[xr.DataArray] = []
    for path in paths:
        with xr.open_dataset(path) as ds:
            if "slp" not in ds:
                raise ValueError(f"{path} does not contain variable 'slp'.")
            arrays.append(ds["slp"].load())
    slp = xr.concat(arrays, dim="time").sortby("time")
    return slp.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))


def daily_slp_paths(data_dir: str | Path, start_year: int, end_year: int) -> list[Path]:
    resolved = _resolve_path(data_dir)
    daily_dir = resolved / "daily"
    return [daily_dir / f"slp.{year}.nc" for year in range(int(start_year), int(end_year) + 1)]


def standardize_daily_anomalies(slp: xr.DataArray) -> xr.DataArray:
    slp = drop_feb29(slp)
    day = calendar_day_365_index(slp["time"])
    field = slp.assign_coords(calendar_day=("time", day))
    counts = field.groupby("calendar_day").count("time")
    if int(counts.max()) <= 1:
        anomaly = field - field.mean("time")
        scale = field.std("time").where(lambda value: value > 0.0, 1.0)
        return (anomaly / scale).fillna(0.0)
    climatology = field.groupby("calendar_day").mean("time")
    anomaly = field.groupby("calendar_day") - climatology
    scale = anomaly.groupby("calendar_day").std("time")
    scale = scale.where(np.isfinite(scale) & (scale > 0.0), 1.0)
    standardized = anomaly.groupby("calendar_day") / scale
    return standardized.fillna(0.0).drop_vars("calendar_day")


def drop_feb29(field: xr.DataArray) -> xr.DataArray:
    time = field["time"]
    keep = ~((time.dt.month == 2) & (time.dt.day == 29))
    return field.sel(time=keep)


def count_feb29(field: xr.DataArray) -> int:
    time = field["time"]
    return int(((time.dt.month == 2) & (time.dt.day == 29)).sum())


def calendar_day_365_index(time: xr.DataArray) -> np.ndarray:
    month_lengths = np.asarray([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=int)
    month_offsets = np.concatenate([[0], np.cumsum(month_lengths[:-1])])
    months = np.asarray(time.dt.month, dtype=int)
    days = np.asarray(time.dt.day, dtype=int)
    if np.any((months == 2) & (days == 29)):
        raise ValueError("Feb 29 must be removed before building a 365-day calendar index.")
    return month_offsets[months - 1] + days


def detrend_time_axis(field: xr.DataArray) -> xr.DataArray:
    values = np.asarray(field.values, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("field must have a time axis and at least one feature axis.")
    n_time = values.shape[0]
    if n_time < 2:
        return field.copy()
    matrix = values.reshape(n_time, -1)
    x = np.arange(n_time, dtype=np.float64)
    x_centered = x - x.mean()
    denom = float(np.sum(x_centered**2))
    slopes = (x_centered[:, None] * matrix).sum(axis=0) / denom if denom > 0.0 else np.zeros(matrix.shape[1])
    intercepts = matrix.mean(axis=0) - slopes * x.mean()
    trend = intercepts[None, :] + slopes[None, :] * x[:, None]
    detrended = (matrix - trend).reshape(values.shape)
    return xr.DataArray(detrended, coords=field.coords, dims=field.dims, attrs=field.attrs, name=field.name)


def latitude_area_weights(latitudes: xr.DataArray | np.ndarray) -> np.ndarray:
    lat = np.asarray(latitudes, dtype=float)
    weights = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, None))
    weights[~np.isfinite(weights)] = 0.0
    return weights


def fit_varimax_components(
    standardized_slp: xr.DataArray,
    *,
    n_components: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    return fit_projected_varimax_components(
        standardized_slp,
        standardized_slp,
        n_components=n_components,
        seed=seed,
    )


def fit_projected_varimax_components(
    fit_slp: xr.DataArray,
    score_slp: xr.DataArray,
    *,
    n_components: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    values = np.asarray(fit_slp.values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("SLP array must have shape [time, lat, lon].")
    n_time, n_lat, n_lon = values.shape
    if n_components < 1 or n_components > min(n_time, n_lat * n_lon):
        raise ValueError("n_components must be between 1 and min(time, grid_size).")

    if tuple(fit_slp["lat"].values) != tuple(score_slp["lat"].values) or tuple(fit_slp["lon"].values) != tuple(score_slp["lon"].values):
        raise ValueError("fit_slp and score_slp must share the same spatial grid.")

    weights = latitude_area_weights(fit_slp["lat"].values)
    weighted = values * weights[None, :, None]
    matrix = weighted.reshape(n_time, n_lat * n_lon)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    pca = PCA(n_components=int(n_components), svd_solver="full")
    pca_scores = pca.fit_transform(matrix)
    loadings = pca.components_.T
    rotated_loadings, rotation = varimax(loadings)
    order, rotated_diagonal = rotated_component_order(rotation, np.asarray(pca.explained_variance_, dtype=float))
    rotated_loadings = rotated_loadings[:, order]
    score_values = np.asarray(score_slp.values, dtype=np.float64)
    score_weighted = score_values * weights[None, :, None]
    score_matrix = score_weighted.reshape(score_values.shape[0], n_lat * n_lon)
    score_matrix = np.nan_to_num(score_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    rotated_scores = (pca.transform(score_matrix) @ rotation)[:, order]
    rotated_scores = (rotated_scores - rotated_scores.mean(axis=0, keepdims=True)) / np.maximum(
        rotated_scores.std(axis=0, ddof=1, keepdims=True), 1.0e-12
    )

    columns = [f"component_{index + 1:02d}" for index in range(int(n_components))]
    dates = pd.to_datetime(score_slp["time"].values)
    scores = pd.DataFrame(rotated_scores, index=dates, columns=columns)
    maps = rotated_loadings.reshape(n_lat, n_lon, int(n_components))
    total_variance = float(np.sum(pca.explained_variance_))
    rotated_ratio = rotated_diagonal[order] / total_variance if total_variance > 0.0 else np.zeros_like(rotated_diagonal[order])
    return scores, maps, np.asarray(rotated_ratio, dtype=float)


def discover_causal_edges(
    weekly_scores: pd.DataFrame,
    *,
    max_lag: int,
    pc_alpha: float,
    link_density: float,
    backend: Literal["tigramite", "regression"],
) -> list[LaggedEdge]:
    if backend == "tigramite":
        return _discover_causal_edges_tigramite(
            weekly_scores,
            max_lag=max_lag,
            pc_alpha=pc_alpha,
            link_density=link_density,
        )
    if backend == "regression":
        return _discover_causal_edges_regression(weekly_scores, max_lag=max_lag, pc_alpha=pc_alpha)
    raise ValueError(f"Unsupported causal backend: {backend}")


def ensure_causal_backend_available(backend: Literal["tigramite", "regression"]) -> None:
    if backend == "regression":
        return
    if backend != "tigramite":
        raise ValueError(f"Unsupported causal backend: {backend}")
    if importlib.util.find_spec("tigramite") is None:
        raise RuntimeError(
            "tigramite is required for the paper-aligned causal reconstruction. "
            "Install it with `pip install tigramite` or run smoke tests with "
            "`--causal-backend regression`."
        )


def _discover_causal_edges_tigramite(
    weekly_scores: pd.DataFrame,
    *,
    max_lag: int,
    pc_alpha: float,
    link_density: float,
) -> list[LaggedEdge]:
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except ImportError as exc:
        raise RuntimeError(
            "tigramite is required for the paper-aligned causal reconstruction. "
            "Install it with `pip install tigramite` or run smoke tests with "
            "`--causal-backend regression`."
        ) from exc

    data = pp.DataFrame(weekly_scores.to_numpy(dtype=float), var_names=list(weekly_scores.columns))
    pcmci = PCMCI(dataframe=data, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    # Runge et al. use the PC step as variable selection for the subsequent
    # sparse causal regression. The final PCMCI/MCI p-matrix is not the same
    # object as the parent set listed in Supplementary Tables 2/3.
    parents = pcmci.run_pc_stable(tau_min=1, tau_max=int(max_lag), pc_alpha=float(pc_alpha))
    n_components = weekly_scores.shape[1]
    parent_candidates: dict[int, list[tuple[int, int, float]]] = {target: [] for target in range(n_components)}
    for target, target_parents in parents.items():
        for source, negative_lag in target_parents:
            lag = abs(int(negative_lag))
            if 1 <= lag <= int(max_lag):
                parent_candidates[int(target)].append((int(source), int(lag), float("nan")))

    candidates = _fit_sparse_standardized_regressions(weekly_scores, parent_candidates, max_lag=max_lag)
    return _threshold_edges_by_link_density(candidates, n_components=n_components, link_density=link_density)


def _fit_sparse_standardized_regressions(
    weekly_scores: pd.DataFrame,
    parent_candidates: dict[int, list[tuple[int, int, float]]],
    *,
    max_lag: int,
) -> list[LaggedEdge]:
    values = weekly_scores.to_numpy(dtype=float)
    n_time, n_components = values.shape
    start = int(max_lag)
    edges: list[LaggedEdge] = []
    if n_time <= start + 2:
        return edges
    for target in range(n_components):
        parents = sorted(set(parent_candidates.get(target, [])))
        if not parents:
            continue
        y = values[start:, target]
        x_columns = [values[start - lag : n_time - lag, source] for source, lag, _ in parents]
        x = np.column_stack(x_columns)
        x_std = x.std(axis=0, ddof=1)
        keep = x_std > 1.0e-12
        if not np.any(keep) or y.std(ddof=1) <= 1.0e-12:
            continue
        kept_parents = [parent for parent, ok in zip(parents, keep) if bool(ok)]
        x = x[:, keep]
        x = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, ddof=1, keepdims=True)
        y_scaled = (y - y.mean()) / y.std(ddof=1)
        design = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(design, y_scaled, rcond=None)
        coefficients = beta[1:]
        residual = y_scaled - design @ beta
        df = max(1, len(y_scaled) - design.shape[1])
        sigma2 = float(np.sum(residual**2) / df)
        try:
            cov = sigma2 * np.linalg.pinv(design.T @ design)
            se = np.sqrt(np.maximum(np.diag(cov)[1:], 0.0))
            t_values = np.divide(coefficients, se, out=np.zeros_like(coefficients), where=se > 1.0e-12)
            p_values = 2.0 * stats.t.sf(np.abs(t_values), df=df)
        except np.linalg.LinAlgError:
            p_values = np.ones_like(coefficients, dtype=float)
        for (source, lag, selection_p), coefficient, p_value in zip(kept_parents, coefficients, p_values):
            if abs(float(coefficient)) <= 1.0e-12:
                continue
            edges.append(
                LaggedEdge(
                    source=int(source),
                    target=int(target),
                    lag=int(lag),
                    coefficient=float(coefficient),
                    p_value=float(p_value) if np.isfinite(p_value) else float(selection_p),
                )
            )
    return edges


def _threshold_edges_by_link_density(
    candidates: Sequence[LaggedEdge],
    *,
    n_components: int,
    link_density: float,
) -> list[LaggedEdge]:
    if not candidates:
        return []
    density = min(max(float(link_density), 0.0), 1.0)
    target_pairs = max(1, int(round(density * int(n_components) * (int(n_components) - 1))))
    cross_pair_strength: dict[tuple[int, int], float] = {}
    for edge in candidates:
        if int(edge.source) == int(edge.target):
            continue
        key = (int(edge.source), int(edge.target))
        cross_pair_strength[key] = max(cross_pair_strength.get(key, 0.0), abs(float(edge.coefficient)))
    if not cross_pair_strength:
        return list(candidates)
    ordered = sorted(cross_pair_strength.values(), reverse=True)
    threshold = ordered[min(target_pairs, len(ordered)) - 1]
    return [edge for edge in candidates if abs(float(edge.coefficient)) >= threshold]


def _discover_causal_edges_regression(weekly_scores: pd.DataFrame, *, max_lag: int, pc_alpha: float) -> list[LaggedEdge]:
    values = weekly_scores.to_numpy(dtype=float)
    n_time, n_components = values.shape
    edges: list[LaggedEdge] = []
    candidates: list[LaggedEdge] = []
    for source in range(n_components):
        for target in range(n_components):
            for lag in range(1, int(max_lag) + 1):
                if n_time <= lag + 2:
                    continue
                x = values[:-lag, source]
                y = values[lag:, target]
                if np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
                    continue
                fit = stats.linregress(x, y)
                edge = LaggedEdge(
                    source=source,
                    target=target,
                    lag=lag,
                    coefficient=float(fit.slope),
                    p_value=float(fit.pvalue) if np.isfinite(fit.pvalue) else 1.0,
                )
                candidates.append(edge)
                if edge.p_value <= pc_alpha:
                    edges.append(edge)
    if edges or not candidates:
        return edges
    candidates.sort(key=lambda edge: (edge.p_value, -abs(edge.coefficient)))
    return candidates[: min(3, len(candidates))]


def compute_sem_effects(edges: Sequence[LaggedEdge], *, n_components: int, max_lag: int) -> SemEffects:
    n = int(n_components)
    tau_max = int(max_lag)
    coefficient_matrices = np.zeros((tau_max + 1, n, n), dtype=float)
    direct_rows: list[dict[str, float | int]] = []
    for edge in edges:
        if not 1 <= int(edge.lag) <= tau_max:
            continue
        coefficient_matrices[int(edge.lag), int(edge.target), int(edge.source)] += float(edge.coefficient)
        direct_rows.append(
            {
                "source": int(edge.source),
                "target": int(edge.target),
                "lag": int(edge.lag),
                "coefficient": float(edge.coefficient),
                "p_value": float(edge.p_value),
            }
        )

    causal_effect_matrices = _compute_causal_effect_matrices(coefficient_matrices)
    ce_source_target = np.transpose(causal_effect_matrices[:, :, :], (0, 2, 1))
    ce_max_abs = np.max(np.abs(ce_source_target[1:]), axis=0) if tau_max > 0 else np.zeros((n, n), dtype=float)
    coefficient_source_target = np.transpose(coefficient_matrices, (0, 2, 1))
    linear_coefficient_matrix = _collapse_lagged_matrix_by_max_abs(coefficient_source_target)

    total_rows = [
        {"source": source, "target": target, "lag": lag, "total_effect": float(ce_source_target[lag, source, target])}
        for lag in range(1, tau_max + 1)
        for source in range(n)
        for target in range(n)
        if source != target and abs(float(ce_source_target[lag, source, target])) > 1.0e-12
    ]
    path_rows: list[dict[str, float | int]] = []
    cmax = max(1, (n - 1) * (n - 2))
    for mediator in range(n):
        blocked = coefficient_matrices.copy()
        blocked[:, mediator, :] = 0.0
        blocked_ce = np.transpose(_compute_causal_effect_matrices(blocked), (0, 2, 1))
        mce = ce_source_target - blocked_ce
        for source in range(n):
            if source == mediator:
                continue
            for target in range(n):
                if target in (source, mediator):
                    continue
                lag_values = mce[1:, source, target]
                if len(lag_values) == 0:
                    continue
                best_index = int(np.argmax(np.abs(lag_values)))
                mediated = float(lag_values[best_index])
                if abs(mediated) <= 1.0e-12:
                    continue
                path_rows.append(
                    {
                        "source": source,
                        "mediator": mediator,
                        "target": target,
                        "lag": best_index + 1,
                        "mce": mediated,
                        "mce_max_abs": abs(mediated),
                    }
                )

    gateway_rows = []
    mediator_rows = []
    mediated_total = float(sum(float(row["mce_max_abs"]) for row in path_rows))
    denom = max(1, n - 1)
    direct_abs_max = np.max(np.abs(coefficient_source_target[1:]), axis=0) if tau_max > 0 else np.zeros((n, n), dtype=float)
    for component in range(n):
        outgoing = float(np.sum(ce_max_abs[component, :]) / denom)
        incoming = float(np.sum(ce_max_abs[:, component]) / denom)
        gateway_rows.append(
            {
                "component": component,
                "ace": outgoing,
                "acs": incoming,
                "incoming_total_effect": incoming,
                "direct_out_strength": float(np.sum(direct_abs_max[component, :]) / denom),
                "direct_in_strength": float(np.sum(direct_abs_max[:, component]) / denom),
            }
        )
        mediated_values = [float(row["mce_max_abs"]) for row in path_rows if int(row["mediator"]) == component]
        mediated = float(np.mean(mediated_values)) if mediated_values else 0.0
        mediator_rows.append(
            {
                "component": component,
                "amce": mediated,
                "mediated_fraction": float(len(mediated_values) / cmax),
                "mediated_strength_fraction": float(sum(mediated_values) / mediated_total) if mediated_total > 0.0 else 0.0,
            }
        )

    return SemEffects(
        direct_effects=pd.DataFrame(direct_rows),
        total_effects=pd.DataFrame(total_rows),
        path_effects=pd.DataFrame(path_rows),
        gateway_scores=pd.DataFrame(gateway_rows).sort_values("ace", ascending=False).reset_index(drop=True),
        mediator_scores=pd.DataFrame(mediator_rows).sort_values("amce", ascending=False).reset_index(drop=True),
        coefficient_matrices=coefficient_matrices,
        causal_effect_matrices=causal_effect_matrices,
        linear_coefficient_matrix=linear_coefficient_matrix,
    )


def _compute_causal_effect_matrices(coefficient_matrices: np.ndarray) -> np.ndarray:
    matrices = np.asarray(coefficient_matrices, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("coefficient_matrices must have shape [lag, target, source].")
    tau_max = matrices.shape[0] - 1
    n = matrices.shape[1]
    effects = np.zeros_like(matrices)
    effects[0] = np.eye(n)
    for tau in range(1, tau_max + 1):
        total = np.zeros((n, n), dtype=float)
        for lag in range(1, tau + 1):
            total += matrices[lag] @ effects[tau - lag]
        effects[tau] = total
    return effects


def _collapse_lagged_matrix_by_max_abs(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices, dtype=float)
    if values.ndim != 3:
        raise ValueError("matrices must have shape [lag, source, target].")
    if values.shape[0] <= 1:
        return np.zeros(values.shape[1:], dtype=float)
    lagged = values[1:]
    best = np.argmax(np.abs(lagged), axis=0)
    rows, cols = np.indices(best.shape)
    return lagged[best, rows, cols]


def save_ranking_figure(frame: pd.DataFrame, output_path: str | Path, *, title: str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        plot_frame = pd.DataFrame({"component": [], "score": []})
    else:
        score_columns = [column for column in ("ace", "acs", "amce") if column in frame.columns]
        if not score_columns:
            raise ValueError("ranking frame must contain one of ace, acs, or amce.")
        primary = score_columns[0]
        plot_frame = frame.sort_values(primary, ascending=False).head(15).copy()

    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    x = np.arange(len(plot_frame))
    if len(plot_frame) == 0:
        ax.text(0.5, 0.5, "No significant links", ha="center", va="center", transform=ax.transAxes)
    elif "ace" in plot_frame.columns and "acs" in plot_frame.columns:
        ax.bar(x - 0.18, plot_frame["ace"], width=0.36, label="ACE", color="#4c78a8")
        ax.bar(x + 0.18, plot_frame["acs"], width=0.36, label="ACS", color="#f58518")
    else:
        score_column = "amce" if "amce" in plot_frame.columns else "ace"
        ax.bar(x, plot_frame[score_column], width=0.64, label=score_column.upper(), color="#54a24b")
    label_column = "paper_component" if "paper_component" in plot_frame.columns else "component"
    labels = [f"No.{int(component)}" for component in plot_frame.get(label_column, [])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("effect score")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_component_map_figure(component_maps: np.ndarray, output_path: str | Path, *, n_show: int = 6) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    n_components = component_maps.shape[-1]
    count = min(int(n_show), n_components)
    fig, axes = plt.subplots(2, int(np.ceil(count / 2)), figsize=(8.0, 4.2), constrained_layout=True)
    flat_axes = np.ravel(axes)
    vlim = float(np.nanpercentile(np.abs(component_maps[..., :count]), 98))
    vlim = max(vlim, 1.0e-9)
    for index, ax in enumerate(flat_axes):
        if index >= count:
            ax.axis("off")
            continue
        image = ax.imshow(component_maps[..., index], cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
        ax.set_title(f"C{index + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=list(flat_axes[:count]), location="right", shrink=0.72, label="rotated loading")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_causal_network_figure(edges: Sequence[LaggedEdge], output_path: str | Path, *, n_components: int) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    ax.set_aspect("equal")
    ax.axis("off")

    angles = np.linspace(0.0, 2.0 * np.pi, int(n_components), endpoint=False)
    positions = {
        index: np.array([np.cos(angle), np.sin(angle)], dtype=float)
        for index, angle in enumerate(angles)
    }
    strengths = [abs(float(edge.coefficient)) for edge in edges]
    max_strength = max(strengths) if strengths else 1.0
    sorted_edges = sorted(edges, key=lambda edge: abs(float(edge.coefficient)), reverse=True)
    for edge in sorted_edges[: min(80, len(sorted_edges))]:
        start = positions[int(edge.source)]
        end = positions[int(edge.target)]
        if int(edge.source) == int(edge.target):
            loop_center = start * 1.05
            circle = plt.Circle(loop_center, 0.12, fill=False, color="#6f6f6f", lw=0.8 + 2.0 * abs(edge.coefficient) / max_strength, alpha=0.65)
            ax.add_patch(circle)
            continue
        delta = end - start
        start2 = start + 0.11 * delta
        end2 = end - 0.11 * delta
        ax.annotate(
            "",
            xy=end2,
            xytext=start2,
            arrowprops={
                "arrowstyle": "->",
                "color": "#4f6f8f",
                "lw": 0.6 + 2.4 * abs(edge.coefficient) / max_strength,
                "alpha": 0.38 + 0.55 * abs(edge.coefficient) / max_strength,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )

    for index, position in positions.items():
        ax.scatter([position[0]], [position[1]], s=280, color="#f2f5f8", edgecolor="#36454f", zorder=3)
        ax.text(position[0], position[1], f"C{index + 1}", ha="center", va="center", fontsize=8, zorder=4)
    ax.set_title("Lagged causal network")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def dependency_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "xarray", "scipy", "scikit-learn", "tigramite"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _paper_component_label(value: object) -> str:
    return f"No.{int(value)}"


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], *, n: int = 10) -> str:
    subset = frame.loc[:, list(columns)].head(n).copy()
    display_columns = list(columns)
    lines = ["| " + " | ".join(display_columns) + " |", "| " + " | ".join(["---"] * len(display_columns)) + " |"]
    for row in subset.to_dict("records"):
        cells = []
        for column in columns:
            value = row[column]
            if column in {"component", "paper_component"}:
                cells.append(_paper_component_label(value))
            elif isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary_report(
    path: str | Path,
    *,
    manifest: dict[str, object],
    gateway_scores: pd.DataFrame,
    mediator_scores: pd.DataFrame,
) -> Path:
    output = Path(path)
    lines = [
        "# Runge 2015 causal gateways and mediators reproduction",
        "",
        "This report reproduces the core workflow from `Identifying causal gateways and mediators in complex spatio-temporal systems` on the local NCEP/NCAR sea-level-pressure data.",
        "",
        "## Method",
        "",
        "- Daily SLP fields are restricted to the configured year range; Feb 29 is removed; each gridpoint is transformed to standardized 365-day calendar-day anomalies; the anomalies are linearly detrended and latitude-area weighted.",
        "- Varimax-rotated PCA components are fitted on monthly fields when enough monthly samples are available, then projected back to daily fields.",
        "- Component scores are aggregated to weekly resolution.",
        "- Tigramite/ParCorr selects candidate parents; sparse standardized OLS estimates the lagged causal regression coefficients.",
        "- Lagged links are thresholded by the configured aggregated cross-link density.",
        "- Causal effects use the lag-resolved Runge recursion, and gateways/mediators are ranked by ACE, ACS, and AMCE.",
        "",
        "## Run",
        "",
        f"- Years: {manifest['config']['start_year']}-{manifest['config']['end_year']}",
        f"- Components: {manifest['config']['n_components']}",
        f"- Weekly lag maximum: {manifest['config']['max_lag']}",
        f"- Link density target: {manifest['config']['link_density']}",
        f"- Backend: {manifest['config']['causal_backend']}",
        f"- Daily samples: {manifest['n_daily_samples']}",
        f"- Removed leap days: {manifest['preprocessing']['n_removed_leap_days']}",
        f"- Weekly samples: {manifest['n_weekly_samples']}",
        f"- Causal links: {manifest['n_edges']}",
        "",
        "## Top causal gateways",
        "",
        _markdown_table(gateway_scores, ["paper_component", "component", "ace", "acs", "direct_out_strength", "direct_in_strength"]),
        "",
        "## Top causal mediators",
        "",
        _markdown_table(mediator_scores, ["paper_component", "component", "amce", "mediated_fraction"]),
        "",
        "## Artifacts",
        "",
        "- `causal_edges.csv`: lagged directed links.",
        "- `gateway_scores.csv`: component ACE/ACS rankings.",
        "- `mediator_scores.csv`: component AMCE rankings.",
        "- `mediated_path_effects.csv`: source-mediator-target path effects.",
        "- `component_weekly_scores.csv`: weekly rotated component scores.",
        "- `fig/runge/2015_gateways/*.png`: component maps, network, gateway ranking, and mediator ranking.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def save_outputs(
    config: RungeConfig,
    *,
    daily_scores: pd.DataFrame,
    weekly_scores: pd.DataFrame,
    component_maps: np.ndarray,
    explained_variance_ratio: np.ndarray,
    edges: Sequence[LaggedEdge],
    effects: SemEffects,
    preprocessing: dict[str, object],
) -> dict[str, object]:
    result_dir = config.output_dir / RESULT_SUBDIR
    fig_dir = config.output_dir / FIG_SUBDIR
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    daily_scores.to_csv(result_dir / "component_daily_scores.csv", index_label="time")
    weekly_scores.to_csv(result_dir / "component_weekly_scores.csv", index_label="time")
    np.savez_compressed(
        result_dir / "component_maps.npz",
        component_maps=component_maps,
        explained_variance_ratio=explained_variance_ratio,
    )

    edge_rows = [asdict(edge) for edge in edges]
    edge_columns = ["source", "target", "lag", "coefficient", "p_value"]
    with (result_dir / "causal_edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=edge_columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(edge_rows)

    gateway_scores = apply_paper_component_labels(effects.gateway_scores)
    mediator_scores = apply_paper_component_labels(effects.mediator_scores)

    effects.direct_effects.to_csv(result_dir / "direct_effects.csv", index=False)
    effects.total_effects.to_csv(result_dir / "total_effects.csv", index=False)
    effects.path_effects.to_csv(result_dir / "mediated_path_effects.csv", index=False)
    gateway_scores.to_csv(result_dir / "gateway_scores.csv", index=False)
    mediator_scores.to_csv(result_dir / "mediator_scores.csv", index=False)
    pd.DataFrame(
        effects.linear_coefficient_matrix,
        index=weekly_scores.columns,
        columns=weekly_scores.columns,
    ).to_csv(result_dir / "linear_coefficient_matrix.csv")
    np.savez_compressed(
        result_dir / "lagged_linear_matrices.npz",
        coefficient_matrices=effects.coefficient_matrices,
        causal_effect_matrices=effects.causal_effect_matrices,
        component_names=np.asarray(list(weekly_scores.columns), dtype=object),
    )

    save_component_map_figure(component_maps, fig_dir / "component_maps.png")
    save_causal_network_figure(edges, fig_dir / "causal_network.png", n_components=config.n_components)
    save_ranking_figure(gateway_scores, fig_dir / "gateway_ranking.png", title="Causal gateway ranking")
    save_ranking_figure(mediator_scores, fig_dir / "mediator_ranking.png", title="Causal mediator ranking")

    manifest: dict[str, object] = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "data_files": [str(path) for path in daily_slp_paths(config.data_dir, config.start_year, config.end_year)],
        "dependency_versions": dependency_versions(),
        "preprocessing": preprocessing,
        "n_daily_samples": int(len(daily_scores)),
        "n_weekly_samples": int(len(weekly_scores)),
        "n_edges": int(len(edges)),
        "n_linear_matrix_elements": int(effects.linear_coefficient_matrix.size),
        "paper_component_label_map": DEFAULT_PAPER_COMPONENT_LABEL_MAP,
        "top_gateways": gateway_scores.head(10).to_dict("records"),
        "top_mediators": mediator_scores.head(10).to_dict("records"),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary_report(
        result_dir / "summary.md",
        manifest=manifest,
        gateway_scores=gateway_scores,
        mediator_scores=mediator_scores,
    )
    return manifest


def run_pipeline(config: RungeConfig) -> dict[str, object]:
    ensure_causal_backend_available(config.causal_backend)
    slp = load_daily_slp(config.data_dir, config.start_year, config.end_year)
    removed_leap_days = count_feb29(slp)
    standardized = detrend_time_axis(standardize_daily_anomalies(slp))
    monthly = standardized.resample(time="MS").mean()
    fit_field = monthly if int(monthly.sizes["time"]) >= int(config.n_components) else standardized
    daily_scores, component_maps, explained = fit_projected_varimax_components(
        fit_field,
        standardized,
        n_components=config.n_components,
        seed=config.seed,
    )
    weekly_scores = weekly_aggregate(daily_scores)
    edges = discover_causal_edges(
        weekly_scores,
        max_lag=config.max_lag,
        pc_alpha=config.pc_alpha,
        link_density=config.link_density,
        backend=config.causal_backend,
    )
    effects = compute_sem_effects(edges, n_components=config.n_components, max_lag=config.max_lag)
    return save_outputs(
        config,
        daily_scores=daily_scores,
        weekly_scores=weekly_scores,
        component_maps=component_maps,
        explained_variance_ratio=explained,
        edges=edges,
        effects=effects,
        preprocessing={
            "calendar_policy": "drop_feb29_365_day",
            "standardization": "gridpoint_calendar_day_mean_std",
            "detrending": "gridpoint_linear_time_axis",
            "n_removed_leap_days": int(removed_leap_days),
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--start-year", type=int, default=1948)
    parser.add_argument("--end-year", type=int, default=2012)
    parser.add_argument("--n-components", type=int, default=60)
    parser.add_argument("--max-lag", type=int, default=RungeConfig.max_lag)
    parser.add_argument("--pc-alpha", type=float, default=RungeConfig.pc_alpha)
    parser.add_argument("--link-density", type=float, default=RungeConfig.link_density)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--causal-backend", choices=["tigramite", "regression"], default="tigramite")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = RungeConfig(
        mode=args.mode,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        n_components=args.n_components,
        max_lag=args.max_lag,
        pc_alpha=args.pc_alpha,
        link_density=args.link_density,
        seed=args.seed,
        causal_backend=args.causal_backend,
    )
    try:
        manifest = run_pipeline(config)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"n_edges": manifest["n_edges"], "result_dir": str(config.output_dir / RESULT_SUBDIR)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
