from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Callable, Sequence
import time
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.spatial import cKDTree
from scipy.special import ndtr, ndtri
from scipy.stats import t as student_t

from exp.TM.transport_map_density import estimate_mutual_information_transport_map


LOG_2 = float(np.log(2.0))
METHOD_ORDER = (
    "observational_tm",
    "oracle_tm",
    "mlp_tm",
    "reweight_true_tm",
    "reweight_knn_tm",
)
DISPLAY_METHOD_ORDER = (
    "observational_tm",
    "oracle_tm",
    "mlp_tm",
    "reweight_knn_tm",
)
METHOD_LABELS = {
    "observational_tm": "Ordinary MI (observed + TM)",
    "oracle_tm": "Oracle samples + TM",
    "mlp_tm": "MLP intervention + TM",
    "reweight_true_tm": "Reweight (true weights) + TM",
    "reweight_knn_tm": "RW-EI",
}
METHOD_COLORS = {
    "observational_tm": "#8c6d9c",
    "oracle_tm": "#4d4d4d",
    "mlp_tm": "#4477aa",
    "reweight_true_tm": "#66a61e",
    "reweight_knn_tm": "#cc6677",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Frozen controls for the paired EI comparison."""

    n_samples: int = 2_000
    rho: float = 0.5
    noise_sd: float = 0.3
    seeds: tuple[int, ...] = (0, 1, 2)
    intervention_samples: int = 2_000
    mlp_hidden_dim: int = 32
    mlp_epochs: int = 80
    mlp_learning_rate: float = 0.01
    mlp_weight_decay: float = 1e-4
    mlp_batch_size: int = 256
    calibration_fraction: float = 0.2
    tm_degree: int = 5
    knn_k: int = 20
    oracle_nodes: int = 64
    oracle_y_points: int = 2_000
    equivalence_margin_bits: float = 0.05

    def __post_init__(self) -> None:
        if self.n_samples < 200:
            raise ValueError("n_samples must be at least 200.")
        if not -0.95 < self.rho < 0.95:
            raise ValueError("rho must lie strictly between -0.95 and 0.95.")
        if self.noise_sd <= 0.0:
            raise ValueError("noise_sd must be positive.")
        if not self.seeds:
            raise ValueError("seeds must not be empty.")
        if self.intervention_samples < 200:
            raise ValueError("intervention_samples must be at least 200.")
        if not 0.05 <= self.calibration_fraction <= 0.4:
            raise ValueError("calibration_fraction must be between 0.05 and 0.4.")
        if self.knn_k < 3 or self.knn_k >= self.n_samples:
            raise ValueError("knn_k must be at least 3 and below n_samples.")
        if self.oracle_nodes < 16 or self.oracle_y_points < 500:
            raise ValueError("oracle quadrature resolution is too small.")


@dataclass
class MLPFit:
    net: object
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float
    residual_sd: float
    calibration_rmse: float
    loss_history: list[float]

    def predict(self, source: np.ndarray) -> np.ndarray:
        import torch

        values = np.asarray(source, dtype=np.float32)
        standardized = (values - self.x_mean) / self.x_scale
        self.net.eval()
        with torch.no_grad():
            prediction = np.asarray(
                self.net(torch.tensor(standardized, dtype=torch.float32)).cpu().tolist(),
                dtype=float,
            )
        return prediction[:, 0].astype(float) * self.y_scale + self.y_mean


def known_dynamics(source: np.ndarray) -> np.ndarray:
    """Smooth nonlinear one-step mechanism used by every experiment arm."""

    values = np.asarray(source, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("source must have shape [sample, 2].")
    x1 = values[:, 0]
    x2 = values[:, 1]
    return 0.8 * x1 - 0.4 * x2 + 1.1 * np.sin(np.pi * x1 * x2) + 0.35 * x1**2


def simulate_observational_data(
    config: ExperimentConfig,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample correlated uniform-margin sources and their noisy future target."""

    rng = np.random.default_rng(int(seed))
    covariance = np.array([[1.0, config.rho], [config.rho, 1.0]], dtype=float)
    latent = rng.multivariate_normal(np.zeros(2, dtype=float), covariance, size=config.n_samples)
    source = 2.0 * ndtr(latent) - 1.0
    target = known_dynamics(source) + config.noise_sd * rng.normal(size=config.n_samples)
    return source, target.reshape(-1, 1)


def sample_intervention_channel(
    config: ExperimentConfig,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw paired maximum-entropy intervention states and common target noise."""

    source_rng = np.random.default_rng(int(seed) + 10_009)
    noise_rng = np.random.default_rng(int(seed) + 20_011)
    source = source_rng.uniform(-1.0, 1.0, size=(config.intervention_samples, 2))
    standard_noise = noise_rng.normal(size=config.intervention_samples)
    target = known_dynamics(source) + config.noise_sd * standard_noise
    return source, target.reshape(-1, 1), standard_noise


def oracle_ei_quadrature(
    *,
    noise_sd: float,
    nodes: int = 64,
    y_points: int = 2_000,
) -> float:
    """Compute I(X;Y) from H(Y)-H(noise) under independent uniform intervention."""

    source_nodes, source_weights = leggauss(int(nodes))
    x1, x2 = np.meshgrid(source_nodes, source_nodes, indexing="ij")
    source = np.column_stack([x1.ravel(), x2.ravel()])
    mechanism = known_dynamics(source)
    mixture_weight = np.outer(source_weights, source_weights).ravel() / 4.0
    lower = float(mechanism.min() - 8.0 * noise_sd)
    upper = float(mechanism.max() + 8.0 * noise_sd)
    y_grid = np.linspace(lower, upper, int(y_points), dtype=float)
    density = np.empty_like(y_grid)
    normalizer = float(noise_sd * np.sqrt(2.0 * np.pi))
    for start in range(0, len(y_grid), 256):
        stop = min(start + 256, len(y_grid))
        residual = (y_grid[start:stop, None] - mechanism[None, :]) / noise_sd
        kernels = np.exp(-0.5 * residual**2) / normalizer
        density[start:stop] = kernels @ mixture_weight
    mass = float(np.trapz(density, y_grid))
    density = density / mass
    output_entropy = float(-np.trapz(density * np.log2(np.maximum(density, 1e-300)), y_grid))
    noise_entropy = 0.5 * float(np.log2(2.0 * np.pi * np.e * noise_sd**2))
    return output_entropy - noise_entropy


def observational_mi_quadrature(
    *,
    noise_sd: float,
    rho: float,
    nodes: int = 64,
    y_points: int = 2_000,
) -> float:
    """Compute observational I(X;Y) under the known Gaussian-copula source law."""

    latent_nodes, latent_weights = hermgauss(int(nodes))
    independent_z1, independent_z2 = np.meshgrid(
        np.sqrt(2.0) * latent_nodes,
        np.sqrt(2.0) * latent_nodes,
        indexing="ij",
    )
    correlated_z2 = rho * independent_z1 + np.sqrt(1.0 - rho**2) * independent_z2
    latent = np.column_stack([independent_z1.ravel(), correlated_z2.ravel()])
    source = 2.0 * ndtr(latent) - 1.0
    mechanism = known_dynamics(source)
    mixture_weight = np.outer(latent_weights, latent_weights).ravel() / np.pi
    lower = float(mechanism.min() - 8.0 * noise_sd)
    upper = float(mechanism.max() + 8.0 * noise_sd)
    y_grid = np.linspace(lower, upper, int(y_points), dtype=float)
    density = np.empty_like(y_grid)
    normalizer = float(noise_sd * np.sqrt(2.0 * np.pi))
    for start in range(0, len(y_grid), 256):
        stop = min(start + 256, len(y_grid))
        residual = (y_grid[start:stop, None] - mechanism[None, :]) / noise_sd
        kernels = np.exp(-0.5 * residual**2) / normalizer
        density[start:stop] = kernels @ mixture_weight
    density = density / float(np.trapz(density, y_grid))
    output_entropy = float(-np.trapz(density * np.log2(np.maximum(density, 1e-300)), y_grid))
    noise_entropy = 0.5 * float(np.log2(2.0 * np.pi * np.e * noise_sd**2))
    return output_entropy - noise_entropy


def oracle_convergence_check(config: ExperimentConfig) -> pd.DataFrame:
    """Verify that deterministic quadrature is stable to a resolution increase."""

    resolutions = tuple(dict.fromkeys((max(16, config.oracle_nodes // 2), config.oracle_nodes)))
    rows = []
    for nodes in resolutions:
        rows.append(
            {
                "nodes_per_axis": int(nodes),
                "y_points": int(config.oracle_y_points),
                "ei_bits": oracle_ei_quadrature(
                    noise_sd=config.noise_sd,
                    nodes=int(nodes),
                    y_points=config.oracle_y_points,
                ),
                "observational_mi_bits": observational_mi_quadrature(
                    noise_sd=config.noise_sd,
                    rho=config.rho,
                    nodes=int(nodes),
                    y_points=config.oracle_y_points,
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame["change_bits"] = frame["ei_bits"].diff().abs()
    frame["observational_change_bits"] = frame["observational_mi_bits"].diff().abs()
    return frame


def _fit_mlp(
    source: np.ndarray,
    target: np.ndarray,
    config: ExperimentConfig,
    *,
    seed: int,
) -> MLPFit:
    import torch

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    source_array = np.asarray(source, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.float32).reshape(-1, 1)
    rng = np.random.default_rng(int(seed) + 30_013)
    order = rng.permutation(len(source_array))
    calibration_size = max(32, int(round(config.calibration_fraction * len(order))))
    calibration_index = order[:calibration_size]
    train_index = order[calibration_size:]

    x_train = source_array[train_index]
    y_train = target_array[train_index]
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_scale = x_train.std(axis=0, keepdims=True)
    x_scale = np.where(x_scale > 1e-8, x_scale, 1.0)
    y_mean = float(y_train.mean())
    y_scale = max(float(y_train.std()), 1e-8)
    x_standardized = (x_train - x_mean) / x_scale
    y_standardized = (y_train - y_mean) / y_scale

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*", category=UserWarning)
        net = torch.nn.Sequential(
            torch.nn.Linear(2, int(config.mlp_hidden_dim)),
            torch.nn.Tanh(),
            torch.nn.Linear(int(config.mlp_hidden_dim), int(config.mlp_hidden_dim)),
            torch.nn.Tanh(),
            torch.nn.Linear(int(config.mlp_hidden_dim), 1),
        )
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=float(config.mlp_learning_rate),
        weight_decay=float(config.mlp_weight_decay),
    )
    x_tensor = torch.tensor(x_standardized, dtype=torch.float32)
    y_tensor = torch.tensor(y_standardized, dtype=torch.float32)
    generator = torch.Generator().manual_seed(int(seed) + 40_009)
    batch_size = min(int(config.mlp_batch_size), len(x_tensor))
    loss_history: list[float] = []
    for _ in range(int(config.mlp_epochs)):
        permutation = torch.randperm(len(x_tensor), generator=generator)
        batch_losses: list[float] = []
        for start in range(0, len(permutation), batch_size):
            batch = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = net(x_tensor[batch])
            loss = torch.mean((prediction - y_tensor[batch]) ** 2)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().item()))
        loss_history.append(float(np.mean(batch_losses)))

    fit = MLPFit(
        net=net,
        x_mean=x_mean.astype(np.float32),
        x_scale=x_scale.astype(np.float32),
        y_mean=y_mean,
        y_scale=y_scale,
        residual_sd=float("nan"),
        calibration_rmse=float("nan"),
        loss_history=loss_history,
    )
    calibration_prediction = fit.predict(source_array[calibration_index])
    calibration_residual = target_array[calibration_index, 0] - calibration_prediction
    residual_sd = max(float(np.std(calibration_residual, ddof=1)), 1e-8)
    calibration_rmse = float(np.sqrt(np.mean(calibration_residual**2)))
    fit.residual_sd = residual_sd
    fit.calibration_rmse = calibration_rmse
    return fit


def gaussian_copula_true_weights(source: np.ndarray, *, rho: float) -> np.ndarray:
    """Return q_product(x)/p_observed(x) for the known Gaussian copula."""

    values = np.asarray(source, dtype=float)
    u = np.clip((values + 1.0) / 2.0, 1e-8, 1.0 - 1e-8)
    z = ndtri(u)
    correlation = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    inverse_gap = np.linalg.inv(correlation) - np.eye(2, dtype=float)
    log_copula = -0.5 * float(np.log(np.linalg.det(correlation))) - 0.5 * np.einsum(
        "ni,ij,nj->n", z, inverse_gap, z
    )
    log_weight = np.clip(-log_copula, -30.0, 30.0)
    return np.exp(log_weight)


def estimate_product_density_ratio_knn(
    source: np.ndarray,
    *,
    k: int,
    seed: int,
) -> np.ndarray:
    """Estimate product-marginal / observed density ratio without a transition model."""

    values = np.asarray(source, dtype=float)
    if values.ndim != 2:
        raise ValueError("source must be a two-dimensional array.")
    if not 2 <= int(k) < len(values):
        raise ValueError("k must be at least 2 and smaller than the sample count.")
    rng = np.random.default_rng(int(seed) + 50_021)
    product = np.column_stack(
        [values[rng.permutation(len(values)), column] for column in range(values.shape[1])]
    )
    product = product + rng.normal(scale=1e-10, size=product.shape)
    observed_tree = cKDTree(values)
    product_tree = cKDTree(product)
    observed_distance = observed_tree.query(values, k=int(k) + 1)[0][:, -1]
    product_distance = product_tree.query(values, k=int(k))[0][:, -1]
    observed_distance = np.maximum(observed_distance, 1e-12)
    product_distance = np.maximum(product_distance, 1e-12)
    log_ratio = values.shape[1] * (np.log(observed_distance) - np.log(product_distance))
    log_ratio += np.log((len(values) - 1.0) / len(product))
    return np.exp(np.clip(log_ratio, -20.0, 20.0))


def effective_sample_size(weight: np.ndarray) -> float:
    values = np.asarray(weight, dtype=float).reshape(-1)
    return float(values.sum() ** 2 / np.sum(values**2))


def _tm_ei(
    source: np.ndarray,
    target: np.ndarray,
    *,
    degree: int,
    sample_weight: np.ndarray | None = None,
) -> dict[str, object]:
    return estimate_mutual_information_transport_map(
        np.asarray(source, dtype=float),
        np.asarray(target, dtype=float),
        degree=int(degree),
        sample_weight=sample_weight,
        joint_order="source_first",
    )


def _run_seed(
    config: ExperimentConfig,
    *,
    seed: int,
    oracle_true: float,
    observational_true: float,
) -> list[dict[str, object]]:
    source_obs, target_obs = simulate_observational_data(config, seed=seed)
    source_do, target_oracle, standard_noise = sample_intervention_channel(config, seed=seed)
    fit = _fit_mlp(source_obs, target_obs, config, seed=seed)
    target_mlp = fit.predict(source_do) + fit.residual_sd * standard_noise
    target_mlp = target_mlp.reshape(-1, 1)

    true_weight = gaussian_copula_true_weights(source_obs, rho=config.rho)
    estimated_weight = estimate_product_density_ratio_knn(
        source_obs,
        k=config.knn_k,
        seed=seed,
    )
    estimates = {
        "observational_tm": _tm_ei(source_obs, target_obs, degree=config.tm_degree),
        "oracle_tm": _tm_ei(source_do, target_oracle, degree=config.tm_degree),
        "mlp_tm": _tm_ei(source_do, target_mlp, degree=config.tm_degree),
        "reweight_true_tm": _tm_ei(
            source_obs,
            target_obs,
            degree=config.tm_degree,
            sample_weight=true_weight,
        ),
        "reweight_knn_tm": _tm_ei(
            source_obs,
            target_obs,
            degree=config.tm_degree,
            sample_weight=estimated_weight,
        ),
    }
    mechanism_rmse = float(
        np.sqrt(np.mean((fit.predict(source_do) - known_dynamics(source_do)) ** 2))
    )
    weight_log_rmse = float(
        np.sqrt(
            np.mean(
                (
                    np.log(np.maximum(estimated_weight, 1e-300))
                    - np.log(np.maximum(true_weight, 1e-300))
                )
                ** 2
            )
        )
    )
    common = {
        "seed": int(seed),
        "n_samples": int(config.n_samples),
        "rho": float(config.rho),
        "noise_sd": float(config.noise_sd),
        "oracle_true_bits": float(oracle_true),
        "observational_mi_true_bits": float(observational_true),
        "mlp_calibration_rmse": float(fit.calibration_rmse),
        "mlp_mechanism_rmse_on_do": mechanism_rmse,
        "estimated_noise_sd": float(fit.residual_sd),
        "true_weight_ess": effective_sample_size(true_weight),
        "estimated_weight_ess": effective_sample_size(estimated_weight),
        "weight_log_rmse": weight_log_rmse,
    }
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        estimate = float(estimates[method]["mi_hat"])
        reference_truth = observational_true if method == "observational_tm" else oracle_true
        rows.append(
            {
                **common,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "ei_bits": estimate,
                "reference_truth_bits": float(reference_truth),
                "error_bits": estimate - reference_truth,
                "absolute_error_bits": abs(estimate - reference_truth),
                "tm_effective_sample_size": float(estimates[method]["effective_sample_size"]),
            }
        )
    return rows


def run_experiment(
    config: ExperimentConfig,
    *,
    cache_path: str | Path | None = None,
    force: bool = False,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> pd.DataFrame:
    """Run or resume a paired comparison and return one row per method and seed."""

    resolved_cache = Path(cache_path) if cache_path is not None else None
    records: list[dict[str, object]] = []
    completed_seeds: set[int] = set()
    if resolved_cache is not None and resolved_cache.exists() and not force:
        payload = json.loads(resolved_cache.read_text(encoding="utf-8"))
        if payload.get("schema_version") == 3 and payload.get("config") == _jsonable_config(config):
            records = list(payload.get("records", []))
            completed_seeds = {int(seed) for seed in payload.get("completed_seeds", [])}
            if completed_seeds == {int(seed) for seed in config.seeds}:
                return pd.DataFrame(records)
    oracle_true = oracle_ei_quadrature(
        noise_sd=config.noise_sd,
        nodes=config.oracle_nodes,
        y_points=config.oracle_y_points,
    )
    observational_true = observational_mi_quadrature(
        noise_sd=config.noise_sd,
        rho=config.rho,
        nodes=config.oracle_nodes,
        y_points=config.oracle_y_points,
    )
    for seed in config.seeds:
        if int(seed) in completed_seeds:
            continue
        records.extend(
            _run_seed(
                config,
                seed=int(seed),
                oracle_true=oracle_true,
                observational_true=observational_true,
            )
        )
        completed_seeds.add(int(seed))
        if resolved_cache is not None:
            _write_experiment_cache(
                resolved_cache,
                config=config,
                completed_seeds=completed_seeds,
                records=records,
            )
        if progress_callback is not None:
            progress_callback(int(seed), len(completed_seeds), len(config.seeds))
    frame = pd.DataFrame(records)
    if resolved_cache is not None and not resolved_cache.exists():
        _write_experiment_cache(
            resolved_cache,
            config=config,
            completed_seeds=completed_seeds,
            records=records,
        )
    return frame


def _write_experiment_cache(
    path: Path,
    *,
    config: ExperimentConfig,
    completed_seeds: set[int],
    records: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 3,
        "config": _jsonable_config(config),
        "completed_seeds": sorted(completed_seeds),
        "records": records,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _jsonable_config(config: ExperimentConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["seeds"] = list(config.seeds)
    return payload


def summarize_results(records: pd.DataFrame, config: ExperimentConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = records.copy()
    frame["method_label"] = frame["method"].map(METHOD_LABELS).fillna(frame["method_label"])
    summary = (
        frame.groupby(["method", "method_label"], sort=False)
        .agg(
            mean_ei_bits=("ei_bits", "mean"),
            sd_ei_bits=("ei_bits", "std"),
            bias_bits=("error_bits", "mean"),
            mae_bits=("absolute_error_bits", "mean"),
            rmse_bits=("error_bits", lambda values: float(np.sqrt(np.mean(np.asarray(values) ** 2)))),
        )
        .reset_index()
    )
    summary["method_order"] = summary["method"].map({name: i for i, name in enumerate(METHOD_ORDER)})
    summary = summary.sort_values("method_order").drop(columns="method_order").reset_index(drop=True)

    pivot = frame.pivot(index="seed", columns="method", values="ei_bits")
    paired = pivot["mlp_tm"] - pivot["reweight_knn_tm"]
    mean_difference = float(paired.mean())
    if len(paired) > 1:
        standard_error = float(paired.std(ddof=1) / np.sqrt(len(paired)))
        critical = float(student_t.ppf(0.95, df=len(paired) - 1))
        ci_low = mean_difference - critical * standard_error
        ci_high = mean_difference + critical * standard_error
    else:
        ci_low = ci_high = mean_difference
    agreement = {
        "paired_mean_difference_bits": mean_difference,
        "paired_90ci_low_bits": float(ci_low),
        "paired_90ci_high_bits": float(ci_high),
        "equivalence_margin_bits": float(config.equivalence_margin_bits),
        "equivalent_within_margin": bool(
            ci_low > -config.equivalence_margin_bits and ci_high < config.equivalence_margin_bits
        ),
    }
    return summary, agreement


def evaluate_between_relation(records: pd.DataFrame) -> pd.DataFrame:
    """Check whether estimated reweighted EI lies between the two numerical truths."""

    reweighted = records.loc[records["method"] == "reweight_knn_tm"].copy()
    lower = np.minimum(
        reweighted["observational_mi_true_bits"].to_numpy(dtype=float),
        reweighted["oracle_true_bits"].to_numpy(dtype=float),
    )
    upper = np.maximum(
        reweighted["observational_mi_true_bits"].to_numpy(dtype=float),
        reweighted["oracle_true_bits"].to_numpy(dtype=float),
    )
    estimate = reweighted["ei_bits"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "seed": reweighted["seed"].to_numpy(dtype=int),
            "ordinary_mi_truth_bits": reweighted["observational_mi_true_bits"].to_numpy(dtype=float),
            "reweighted_ei_bits": estimate,
            "ei_truth_bits": reweighted["oracle_true_bits"].to_numpy(dtype=float),
            "between_truths": (estimate >= lower) & (estimate <= upper),
            "distance_above_interval_bits": np.maximum(estimate - upper, 0.0),
            "distance_below_interval_bits": np.maximum(lower - estimate, 0.0),
        }
    )


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def plot_comparison(
    records: pd.DataFrame,
    config: ExperimentConfig,
    *,
    output_dir: str | Path | None = None,
) -> plt.Figure:
    """Render the primary comparison, paired errors, and overlap diagnostic."""

    _configure_matplotlib()
    frame = records.copy()
    truth = float(frame["oracle_true_bits"].iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.2), constrained_layout=True)

    x_positions = np.arange(len(DISPLAY_METHOD_ORDER), dtype=float)
    for position, method in enumerate(DISPLAY_METHOD_ORDER):
        values = frame.loc[frame["method"] == method, "ei_bits"].to_numpy(dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.zeros(1)
        axes[0].scatter(
            position + jitter,
            values,
            s=18,
            color=METHOD_COLORS[method],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        axes[0].errorbar(
            position,
            mean,
            yerr=1.96 * sem,
            fmt="o",
            color=METHOD_COLORS[method],
            markersize=5,
            capsize=3,
            linewidth=1.2,
            zorder=3,
        )
    axes[0].axhline(truth, color="black", linestyle="--", linewidth=1.0, label="EI truth")
    axes[0].set_xticks(
        x_positions,
        ["Ordinary\nMI", "Oracle\nTM", "MLP", "RW-EI"],
    )
    axes[0].set_ylabel("Information (bits)")
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )

    main_methods = ("mlp_tm", "reweight_knn_tm")
    pivot = frame.pivot(index="seed", columns="method", values="error_bits")
    for seed, row in pivot.iterrows():
        axes[1].plot(
            [0, 1],
            [row[main_methods[0]], row[main_methods[1]]],
            color="#b7b7b7",
            linewidth=0.8,
            alpha=0.7,
            zorder=1,
        )
        axes[1].scatter(
            [0, 1],
            [row[main_methods[0]], row[main_methods[1]]],
            color=[METHOD_COLORS[name] for name in main_methods],
            s=20,
            zorder=2,
        )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    axes[1].set_xticks([0, 1], ["MLP", "RW-EI"])
    axes[1].set_ylabel("Error relative to truth (bits)")

    rw = frame[frame["method"] == "reweight_knn_tm"].copy()
    ess_ratio = rw["estimated_weight_ess"].to_numpy(dtype=float) / rw["n_samples"].to_numpy(dtype=float)
    axes[2].scatter(
        ess_ratio,
        rw["absolute_error_bits"],
        color=METHOD_COLORS["reweight_knn_tm"],
        s=25,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.35,
    )
    axes[2].set_xlabel("Weight ESS / N")
    axes[2].set_ylabel("RW-EI absolute error (bits)")

    for label, axis in zip("abc", axes):
        axis.text(-0.14, 1.03, label, transform=axis.transAxes, fontsize=8, fontweight="bold")
        axis.grid(False)

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for suffix, kwargs in (("png", {"dpi": 300}), ("svg", {}), ("pdf", {})):
            fig.savefig(
                destination / f"mlp_vs_reweighted_ei.{suffix}",
                bbox_inches="tight",
                **kwargs,
            )
    return fig


def build_interpretation(records: pd.DataFrame, config: ExperimentConfig) -> str:
    summary, agreement = summarize_results(records, config)
    lookup = summary.set_index("method")
    mlp_mae = float(lookup.loc["mlp_tm", "mae_bits"])
    rw_mae = float(lookup.loc["reweight_knn_tm", "mae_bits"])
    ordinary_estimate = float(lookup.loc["observational_tm", "mean_ei_bits"])
    ordinary_truth = float(records["observational_mi_true_bits"].iloc[0])
    ei_truth = float(records["oracle_true_bits"].iloc[0])
    between = evaluate_between_relation(records)
    between_count = int(between["between_truths"].sum())
    ess_ratio = float(
        records.loc[records["method"] == "reweight_knn_tm", "estimated_weight_ess"].mean()
        / config.n_samples
    )
    equivalence_text = (
        "落在预注册等价界限内"
        if agreement["equivalent_within_margin"]
        else "尚未完全落入预注册等价界限"
    )
    return (
        f"**Smoke 结果解读。** MLP 与 RW-EI 的配对差为 "
        f"{agreement['paired_mean_difference_bits']:.3f} bit，90% CI "
        f"[{agreement['paired_90ci_low_bits']:.3f}, {agreement['paired_90ci_high_bits']:.3f}]，"
        f"{equivalence_text}（±{config.equivalence_margin_bits:.2f} bit）。"
        f"相对数值真值，MLP 的 MAE 为 {mlp_mae:.3f} bit，RW-EI 为 {rw_mae:.3f} bit；"
        f"普通 MI 的数值真值为 {ordinary_truth:.3f} bit，未加权 TM 均值为 "
        f"{ordinary_estimate:.3f} bit。RW-EI 在 {between_count}/{len(between)} 个 seed 中"
        f"位于普通 MI 真值 {ordinary_truth:.3f} bit 与 EI 真值 {ei_truth:.3f} bit 之间。"
        f"重加权的平均 ESS/N 为 {ess_ratio:.3f}。Smoke 结果只验证代码链路，"
        "正式结论仍应使用 30 个配对 seed 和样本量/相关强度扫描。"
    )


def aggregate_full_validation(records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the full factorial validation with seed-level uncertainty."""

    required = {"sweep", "grid_id", "method", "method_label"}
    records = records.copy()
    records["method_label"] = records["method"].map(METHOD_LABELS).fillna(records["method_label"])
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"full-validation records are missing columns: {sorted(missing)}")
    summary = (
        records.groupby(
            ["sweep", "grid_id", "n_samples", "rho", "method", "method_label"],
            sort=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            mean_information_bits=("ei_bits", "mean"),
            sd_information_bits=("ei_bits", "std"),
            mean_error_bits=("error_bits", "mean"),
            mae_bits=("absolute_error_bits", "mean"),
            rmse_bits=("error_bits", lambda values: float(np.sqrt(np.mean(np.asarray(values) ** 2)))),
            mean_estimated_weight_ess=("estimated_weight_ess", "mean"),
            mean_true_weight_ess=("true_weight_ess", "mean"),
            ordinary_mi_truth_bits=("observational_mi_true_bits", "first"),
            ei_truth_bits=("oracle_true_bits", "first"),
        )
        .reset_index()
    )
    summary["ci95_mae_bits"] = (
        1.96
        * records.groupby(
            ["sweep", "grid_id", "n_samples", "rho", "method", "method_label"],
            sort=False,
        )["absolute_error_bits"]
        .sem()
        .to_numpy()
    )
    summary["estimated_ess_ratio"] = summary["mean_estimated_weight_ess"] / summary["n_samples"]
    return summary


def summarize_full_agreement(
    records: pd.DataFrame,
    *,
    equivalence_margin_bits: float = 0.05,
) -> pd.DataFrame:
    """Summarize paired MLP-minus-reweighting agreement for every full-grid condition."""

    rows: list[dict[str, object]] = []
    group_columns = ["sweep", "grid_id", "n_samples", "rho"]
    for keys, block in records.groupby(group_columns, sort=True):
        pivot = block.pivot(index="seed", columns="method", values="ei_bits")
        paired = pivot["mlp_tm"] - pivot["reweight_knn_tm"]
        mean = float(paired.mean())
        standard_error = float(paired.std(ddof=1) / np.sqrt(len(paired)))
        critical = float(student_t.ppf(0.95, df=len(paired) - 1))
        low = mean - critical * standard_error
        high = mean + critical * standard_error
        relation = evaluate_between_relation(block)
        rw = block[block["method"] == "reweight_knn_tm"]
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "seeds": int(len(paired)),
                "paired_mean_difference_bits": mean,
                "paired_90ci_low_bits": float(low),
                "paired_90ci_high_bits": float(high),
                "equivalent_within_margin": bool(
                    low > -equivalence_margin_bits and high < equivalence_margin_bits
                ),
                "between_truths_rate": float(relation["between_truths"].mean()),
                "reweighted_mean_error_bits": float(rw["error_bits"].mean()),
                "estimated_ess_ratio": float(
                    (rw["estimated_weight_ess"] / rw["n_samples"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_full_interpretation(records: pd.DataFrame) -> str:
    """Create a compact Chinese interpretation of the completed full grid."""

    summary = aggregate_full_validation(records)
    agreement = summarize_full_agreement(records)
    sample = summary[
        (summary["sweep"] == "sample_size") & (summary["method"] == "reweight_knn_tm")
    ].sort_values("n_samples")
    rho = summary[
        (summary["sweep"] == "rho") & (summary["method"] == "reweight_knn_tm")
    ].sort_values("rho")
    equivalent_count = int(agreement["equivalent_within_margin"].sum())
    total_conditions = int(len(agreement))
    return (
        f"**全量结果解读。** 在样本量扫描中，RW-EI 的 MAE 从 "
        f"N={int(sample.iloc[0]['n_samples']):,} 时的 {sample.iloc[0]['mae_bits']:.3f} bit "
        f"变化到 N={int(sample.iloc[-1]['n_samples']):,} 时的 {sample.iloc[-1]['mae_bits']:.3f} bit。"
        f"在相关强度扫描中，其 MAE 从 ρ={rho.iloc[0]['rho']:.1f} 时的 "
        f"{rho.iloc[0]['mae_bits']:.3f} bit 变化到 ρ={rho.iloc[-1]['rho']:.1f} 时的 "
        f"{rho.iloc[-1]['mae_bits']:.3f} bit。MLP 与 RW-EI 在 "
        f"{equivalent_count}/{total_conditions} 个条件下通过 ±0.05 bit 的配对等价判据。"
        f"逐条件的 ESS/N、配对区间及重加权结果落在两个真值之间的比例见正式汇总表。"
    )


def plot_full_validation(
    records: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
) -> plt.Figure:
    """Plot sample-size convergence, correlation robustness, truth tracking, and ESS."""

    _configure_matplotlib()
    summary = aggregate_full_validation(records)
    methods = ("oracle_tm", "mlp_tm", "reweight_knn_tm")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    axes = axes.ravel()

    sample_summary = summary[summary["sweep"] == "sample_size"]
    for method in methods:
        block = sample_summary[sample_summary["method"] == method].sort_values("n_samples")
        axes[0].errorbar(
            block["n_samples"],
            block["mae_bits"],
            yerr=block["ci95_mae_bits"],
            marker="o",
            markersize=3.8,
            linewidth=1.1,
            capsize=2.5,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Observational sample size")
    axes[0].set_ylabel("MAE to corresponding truth (bits)")

    rho_summary = summary[summary["sweep"] == "rho"]
    for method in methods:
        block = rho_summary[rho_summary["method"] == method].sort_values("rho")
        axes[1].errorbar(
            block["rho"],
            block["mae_bits"],
            yerr=block["ci95_mae_bits"],
            marker="o",
            markersize=3.8,
            linewidth=1.1,
            capsize=2.5,
            color=METHOD_COLORS[method],
        )
    axes[1].set_xlabel("Observational source correlation, ρ")
    axes[1].set_ylabel("MAE to corresponding truth (bits)")

    for method in ("observational_tm", "mlp_tm", "reweight_knn_tm"):
        block = rho_summary[rho_summary["method"] == method].sort_values("rho")
        axes[2].errorbar(
            block["rho"],
            block["mean_information_bits"],
            yerr=1.96 * block["sd_information_bits"] / np.sqrt(block["seeds"]),
            marker="o",
            markersize=3.8,
            linewidth=1.1,
            capsize=2.5,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    truth_block = rho_summary[rho_summary["method"] == "reweight_knn_tm"].sort_values("rho")
    axes[2].plot(
        truth_block["rho"],
        truth_block["ei_truth_bits"],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="EI truth",
    )
    axes[2].set_xlabel("Observational source correlation, ρ")
    axes[2].set_ylabel("Information (bits)")

    rw = (
        records[records["method"] == "reweight_knn_tm"]
        .drop_duplicates(subset=["n_samples", "rho", "seed"])
        .copy()
    )
    ess_ratio = rw["estimated_weight_ess"].to_numpy(dtype=float) / rw["n_samples"].to_numpy(dtype=float)
    signed_error = rw["error_bits"].to_numpy(dtype=float)
    axes[3].scatter(
        ess_ratio,
        signed_error,
        s=10,
        color=METHOD_COLORS["reweight_knn_tm"],
        alpha=0.45,
        edgecolor="none",
    )
    if len(rw) > 1:
        coefficient = np.polyfit(ess_ratio, signed_error, deg=1)
        x_line = np.linspace(float(ess_ratio.min()), float(ess_ratio.max()), 100)
        axes[3].plot(x_line, np.polyval(coefficient, x_line), color="#7f1d3a", linewidth=1.1)
    axes[3].axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    axes[3].set_xlabel("Estimated weight ESS / N")
    axes[3].set_ylabel("RW-EI error (bits)")

    handles, labels = axes[0].get_legend_handles_labels()
    information_handles, information_labels = axes[2].get_legend_handles_labels()
    unique_handles: dict[str, object] = {}
    for handle, label in zip(handles + information_handles, labels + information_labels):
        unique_handles.setdefault(label, handle)
    fig.legend(
        list(unique_handles.values()),
        list(unique_handles.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        frameon=False,
    )
    for label, axis in zip("abcd", axes):
        axis.text(-0.16, 1.04, label, transform=axis.transAxes, fontsize=8, fontweight="bold")
        axis.grid(False)

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for suffix, kwargs in (("png", {"dpi": 300}), ("svg", {}), ("pdf", {})):
            fig.savefig(destination / f"full_validation.{suffix}", bbox_inches="tight", **kwargs)
    return fig


def benchmark_method_runtimes(
    config: ExperimentConfig,
    *,
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Measure warm-process method and component wall time on paired data."""

    import torch

    torch.set_num_threads(1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*", category=UserWarning)
        warmup = torch.nn.Linear(2, 2)
        with torch.no_grad():
            warmup(torch.zeros((8, 2), dtype=torch.float32))

    rows: list[dict[str, object]] = []
    method_names = (
        "observational_tm",
        "oracle_tm",
        "mlp_tm",
        "reweight_true_tm",
        "reweight_knn_tm",
    )
    for seed in seeds:
        preparation_start = time.perf_counter()
        source_obs, target_obs = simulate_observational_data(config, seed=int(seed))
        source_do, target_oracle, standard_noise = sample_intervention_channel(config, seed=int(seed))
        preparation_seconds = time.perf_counter() - preparation_start
        rows.append(
            {
                "n_samples": int(config.n_samples),
                "intervention_samples": int(config.intervention_samples),
                "rho": float(config.rho),
                "seed": int(seed),
                "method": "common_data",
                "method_label": "Common data preparation",
                "stage": "data_generation",
                "seconds": float(preparation_seconds),
                "ei_bits": float("nan"),
            }
        )

        order_rng = np.random.default_rng(int(seed) + 91_003)
        for method in order_rng.permutation(method_names):
            stage_times: list[tuple[str, float]] = []
            estimate = float("nan")
            if method == "observational_tm":
                started = time.perf_counter()
                result = _tm_ei(source_obs, target_obs, degree=config.tm_degree)
                stage_times.append(("tm", time.perf_counter() - started))
                estimate = float(result["mi_hat"])
            elif method == "oracle_tm":
                started = time.perf_counter()
                result = _tm_ei(source_do, target_oracle, degree=config.tm_degree)
                stage_times.append(("tm", time.perf_counter() - started))
                estimate = float(result["mi_hat"])
            elif method == "mlp_tm":
                started = time.perf_counter()
                fit = _fit_mlp(source_obs, target_obs, config, seed=int(seed))
                stage_times.append(("mlp_fit", time.perf_counter() - started))
                started = time.perf_counter()
                target_mlp = (
                    fit.predict(source_do) + fit.residual_sd * standard_noise
                ).reshape(-1, 1)
                stage_times.append(("intervention_generation", time.perf_counter() - started))
                started = time.perf_counter()
                result = _tm_ei(source_do, target_mlp, degree=config.tm_degree)
                stage_times.append(("tm", time.perf_counter() - started))
                estimate = float(result["mi_hat"])
            elif method == "reweight_true_tm":
                started = time.perf_counter()
                weight = gaussian_copula_true_weights(source_obs, rho=config.rho)
                stage_times.append(("weight", time.perf_counter() - started))
                started = time.perf_counter()
                result = _tm_ei(
                    source_obs,
                    target_obs,
                    degree=config.tm_degree,
                    sample_weight=weight,
                )
                stage_times.append(("tm", time.perf_counter() - started))
                estimate = float(result["mi_hat"])
            elif method == "reweight_knn_tm":
                started = time.perf_counter()
                weight = estimate_product_density_ratio_knn(
                    source_obs,
                    k=config.knn_k,
                    seed=int(seed),
                )
                stage_times.append(("density_ratio", time.perf_counter() - started))
                started = time.perf_counter()
                result = _tm_ei(
                    source_obs,
                    target_obs,
                    degree=config.tm_degree,
                    sample_weight=weight,
                )
                stage_times.append(("tm", time.perf_counter() - started))
                estimate = float(result["mi_hat"])
            else:
                raise RuntimeError(f"Unknown benchmark method: {method}")

            common = {
                "n_samples": int(config.n_samples),
                "intervention_samples": int(config.intervention_samples),
                "rho": float(config.rho),
                "seed": int(seed),
                "method": str(method),
                "method_label": METHOD_LABELS[str(method)],
            }
            for stage, seconds in stage_times:
                rows.append(
                    {
                        **common,
                        "stage": stage,
                        "seconds": float(seconds),
                        "ei_bits": float("nan"),
                    }
                )
            rows.append(
                {
                    **common,
                    "stage": "total",
                    "seconds": float(sum(seconds for _, seconds in stage_times)),
                    "ei_bits": estimate,
                }
            )
    return pd.DataFrame(rows)


def aggregate_runtime_benchmark(records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paired runtime replicates and estimate empirical scaling exponents."""

    total = records[(records["stage"] == "total") & (records["n_samples"] > 0)].copy()
    total["method_label"] = total["method"].map(METHOD_LABELS).fillna(total["method_label"])
    summary = (
        total.groupby(["n_samples", "method", "method_label"], sort=False)
        .agg(
            seeds=("seed", "nunique"),
            median_seconds=("seconds", "median"),
            q25_seconds=("seconds", lambda values: float(np.quantile(values, 0.25))),
            q75_seconds=("seconds", lambda values: float(np.quantile(values, 0.75))),
            mean_seconds=("seconds", "mean"),
            sd_seconds=("seconds", "std"),
        )
        .reset_index()
    )
    exponents: dict[str, float] = {}
    for method, block in summary.groupby("method"):
        coefficient = np.polyfit(
            np.log(block["n_samples"].to_numpy(dtype=float)),
            np.log(block["median_seconds"].to_numpy(dtype=float)),
            deg=1,
        )
        exponents[str(method)] = float(coefficient[0])
    summary["empirical_scaling_exponent"] = summary["method"].map(exponents)
    return summary


def plot_runtime_benchmark(
    records: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
) -> plt.Figure:
    """Plot method runtime scaling and largest-sample absolute wall time."""

    _configure_matplotlib()
    summary = aggregate_runtime_benchmark(records)
    methods = DISPLAY_METHOD_ORDER
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for method in methods:
        block = summary[summary["method"] == method].sort_values("n_samples")
        yerr = np.vstack(
            [
                block["median_seconds"] - block["q25_seconds"],
                block["q75_seconds"] - block["median_seconds"],
            ]
        )
        axes[0].errorbar(
            block["n_samples"],
            block["median_seconds"],
            yerr=yerr,
            marker="o",
            markersize=3.8,
            linewidth=1.1,
            capsize=2.5,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Sample size")
    axes[0].set_ylabel("Median wall time per seed (s)")
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )

    largest_n = int(summary["n_samples"].max())
    largest = summary[summary["n_samples"] == largest_n].set_index("method").loc[list(methods)]
    positions = np.arange(len(methods))
    axes[1].barh(
        positions,
        largest["median_seconds"],
        color=[METHOD_COLORS[method] for method in methods],
        height=0.62,
    )
    axes[1].set_xscale("log")
    axes[1].set_yticks(
        positions,
        ["Ordinary MI", "Oracle TM", "MLP", "RW-EI"],
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel(f"Median wall time at N={largest_n:,} (s)")
    for position, method in enumerate(methods):
        value = float(largest.loc[method, "median_seconds"])
        axes[1].text(value * 1.05, position, f"{value:.3g}", va="center", fontsize=6.5)

    for label, axis in zip("ab", axes):
        axis.text(-0.16, 1.04, label, transform=axis.transAxes, fontsize=8, fontweight="bold")
        axis.grid(False)
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for suffix, kwargs in (("png", {"dpi": 300}), ("svg", {}), ("pdf", {})):
            fig.savefig(destination / f"runtime_benchmark.{suffix}", bbox_inches="tight", **kwargs)
    return fig


def plot_combined_validation_and_runtime(
    full_records: pd.DataFrame,
    runtime_records: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
) -> plt.Figure:
    """Combine accuracy, robustness, ESS, and runtime evidence in one figure."""

    _configure_matplotlib()
    validation_summary = aggregate_full_validation(full_records)
    runtime_summary = aggregate_runtime_benchmark(runtime_records)
    validation_methods = ("oracle_tm", "mlp_tm", "reweight_knn_tm")
    runtime_methods = DISPLAY_METHOD_ORDER

    fig = plt.figure(figsize=(12.2, 6.5), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 1.14),
        height_ratios=(1.0, 1.0),
    )
    axes = {
        "a": fig.add_subplot(grid[0, 0]),
        "b": fig.add_subplot(grid[0, 1]),
        "d": fig.add_subplot(grid[0, 2]),
    }
    axes["e"] = fig.add_subplot(grid[1, 0], sharex=axes["a"])
    axes["c"] = fig.add_subplot(grid[1, 1], sharex=axes["b"])
    axes["f"] = fig.add_subplot(grid[1, 2])

    sample_summary = validation_summary[validation_summary["sweep"] == "sample_size"]
    for method in validation_methods:
        block = sample_summary[sample_summary["method"] == method].sort_values("n_samples")
        axes["a"].errorbar(
            block["n_samples"],
            block["mae_bits"],
            yerr=block["ci95_mae_bits"],
            marker="o",
            markersize=3.8,
            linewidth=1.1,
            capsize=2.5,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes["a"].set_xscale("log")
    axes["a"].set_ylabel("MAE to corresponding truth (bits)")
    axes["a"].tick_params(axis="x", labelbottom=False)

    rho_summary = validation_summary[validation_summary["sweep"] == "rho"]
    for method in validation_methods:
        block = rho_summary[rho_summary["method"] == method].sort_values("rho")
        axes["b"].errorbar(
            block["rho"],
            block["mae_bits"],
            yerr=block["ci95_mae_bits"],
            marker="o",
            markersize=3.8,
            linewidth=1.1,
            capsize=2.5,
            color=METHOD_COLORS[method],
        )
    axes["b"].set_ylabel("MAE to corresponding truth (bits)")
    axes["b"].tick_params(axis="x", labelbottom=False)

    for method in runtime_methods:
        block = runtime_summary[runtime_summary["method"] == method].sort_values("n_samples")
        yerr = np.vstack(
            [
                block["median_seconds"] - block["q25_seconds"],
                block["q75_seconds"] - block["median_seconds"],
            ]
        )
        axes["e"].errorbar(
            block["n_samples"],
            block["median_seconds"],
            yerr=yerr,
            marker="o",
            markersize=3.8,
            linewidth=1.1,
            capsize=2.5,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes["e"].set_xscale("log")
    axes["e"].set_yscale("log")
    sample_sizes = sample_summary["n_samples"].drop_duplicates().to_numpy(dtype=float)
    axes["e"].set_xlim(float(sample_sizes.min()) / 1.25, float(sample_sizes.max()) * 1.25)
    axes["e"].set_xlabel("Observational sample size")
    axes["e"].set_ylabel("Median wall time per seed (s)")

    rw = (
        full_records[full_records["method"] == "reweight_knn_tm"]
        .drop_duplicates(subset=["n_samples", "rho", "seed"])
        .copy()
    )
    ess_ratio = (
        rw["estimated_weight_ess"].to_numpy(dtype=float)
        / rw["n_samples"].to_numpy(dtype=float)
    )
    signed_error = rw["error_bits"].to_numpy(dtype=float)
    axes["d"].scatter(
        ess_ratio,
        signed_error,
        s=10,
        color=METHOD_COLORS["reweight_knn_tm"],
        alpha=0.45,
        edgecolor="none",
    )
    if len(rw) > 1:
        coefficient = np.polyfit(ess_ratio, signed_error, deg=1)
        x_line = np.linspace(float(ess_ratio.min()), float(ess_ratio.max()), 100)
        axes["d"].plot(
            x_line,
            np.polyval(coefficient, x_line),
            color="#7f1d3a",
            linewidth=1.1,
        )
    axes["d"].axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    axes["d"].set_xlabel("Estimated weight ESS / N")
    axes["d"].set_ylabel("RW-EI error (bits)")

    for method in ("observational_tm", "mlp_tm", "reweight_knn_tm"):
        block = rho_summary[rho_summary["method"] == method].sort_values("rho")
        axes["c"].errorbar(
            block["rho"],
            block["mean_information_bits"],
            yerr=1.96 * block["sd_information_bits"] / np.sqrt(block["seeds"]),
            marker="o",
            markersize=3.8,
            linewidth=1.1,
            capsize=2.5,
            color=METHOD_COLORS[method],
        )
    truth_block = rho_summary[rho_summary["method"] == "reweight_knn_tm"].sort_values("rho")
    axes["c"].plot(
        truth_block["rho"],
        truth_block["ei_truth_bits"],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="EI truth",
    )
    axes["c"].set_xlabel("Observational source correlation, ρ")
    axes["c"].set_ylabel("Information (bits)")

    largest_n = int(runtime_summary["n_samples"].max())
    largest = (
        runtime_summary[runtime_summary["n_samples"] == largest_n]
        .set_index("method")
        .loc[list(runtime_methods)]
    )
    positions = np.arange(len(runtime_methods))
    axes["f"].barh(
        positions,
        largest["median_seconds"],
        color=[METHOD_COLORS[method] for method in runtime_methods],
        height=0.62,
    )
    axes["f"].set_xscale("log")
    axes["f"].set_yticks(
        positions,
        ["Ordinary MI", "Oracle TM", "MLP", "RW-EI"],
    )
    axes["f"].invert_yaxis()
    axes["f"].set_xlabel(f"Median wall time at N={largest_n:,} (s)")
    for position, method in enumerate(runtime_methods):
        value = float(largest.loc[method, "median_seconds"])
        axes["f"].text(value * 1.05, position, f"{value:.3g}", va="center", fontsize=6.5)

    method_handles, method_labels = axes["e"].get_legend_handles_labels()
    truth_handles, truth_labels = axes["c"].get_legend_handles_labels()
    fig.legend(
        method_handles + truth_handles,
        method_labels + truth_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        frameon=False,
    )
    for axis_key, label in {key: key for key in "abcdef"}.items():
        axis = axes[axis_key]
        axis.text(
            -0.16,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
        )
        axis.grid(False)

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for suffix, kwargs in (("png", {"dpi": 300}), ("svg", {}), ("pdf", {})):
            fig.savefig(
                destination / f"rw_ei_combined_results.{suffix}",
                bbox_inches="tight",
                **kwargs,
            )
    return fig


def full_grid_configs() -> Sequence[ExperimentConfig]:
    """Dense robustness grid with legacy conditions first for cache reuse."""

    seeds = tuple(range(30))
    configs: list[ExperimentConfig] = []
    # Keep the original seven conditions in their original order so existing
    # grid-indexed caches remain valid.
    for rho in (0.0, 0.3, 0.6, 0.8):
        configs.append(
            ExperimentConfig(
                n_samples=8_000,
                rho=rho,
                seeds=seeds,
                intervention_samples=8_000,
                mlp_epochs=120,
                oracle_nodes=96,
                oracle_y_points=3_000,
            )
        )
    for sample_size in (2_000, 8_000, 32_000):
        configs.append(
            ExperimentConfig(
                n_samples=sample_size,
                rho=0.5,
                seeds=seeds,
                intervention_samples=sample_size,
                mlp_epochs=120,
                oracle_nodes=96,
                oracle_y_points=3_000,
            )
        )
    for rho in (0.1, 0.2, 0.4, 0.5, 0.7, 0.9):
        configs.append(
            ExperimentConfig(
                n_samples=8_000,
                rho=rho,
                seeds=seeds,
                intervention_samples=8_000,
                mlp_epochs=120,
                oracle_nodes=96,
                oracle_y_points=3_000,
            )
        )
    for sample_size in (1_000, 4_000, 16_000, 64_000):
        configs.append(
            ExperimentConfig(
                n_samples=sample_size,
                rho=0.5,
                seeds=seeds,
                intervention_samples=sample_size,
                mlp_epochs=120,
                oracle_nodes=96,
                oracle_y_points=3_000,
            )
        )
    return configs


def full_grid_sweeps() -> Sequence[str]:
    """Return the one-factor sweep label paired with each full-grid config."""

    return (
        ("rho",) * 4
        + ("sample_size",) * 3
        + ("rho",) * 6
        + ("sample_size",) * 4
    )
