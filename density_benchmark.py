from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd().resolve() / "tmp" / "matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from exp.TM.transport_map_density import fit_affine_transport_map_density

DENSITY_BENCHMARK_VERSION = "2026-05-13-balanced-repeats"

DENSITY_FAMILY_SPECS = {
    "gaussian": {"title": "Gaussian"},
    "gmm": {"title": "Gaussian mixture"},
    "banana": {"title": "Banana"},
}
DENSITY_METHOD_ORDER = ("transport_map", "kde", "knn")
DENSITY_METHOD_LABELS = {
    "transport_map": "Transport map",
    "kde": "KDE",
    "knn": "kNN",
}
DENSITY_COLORS = {
    "transport_map": "#2f6f9f",
    "kde": "#d8892d",
    "knn": "#3ca090",
}
DENSITY_MARKERS = {
    "transport_map": "o",
    "kde": "s",
    "knn": "^",
}


def resolve_project_root() -> Path:
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / "utils.py").exists():
            return candidate
    return Path.cwd().resolve()


PROJECT_ROOT = resolve_project_root()


@dataclass(frozen=True)
class DensityBenchmarkConfig:
    cache_dir: Path = PROJECT_ROOT / "exp" / "cache" / "density_benchmark"
    fig_dir: Path = PROJECT_ROOT / "fig" / "transport_map_mutual_information"
    families: tuple[str, ...] = ("gaussian", "gmm", "banana")
    methods: tuple[str, ...] = DENSITY_METHOD_ORDER
    accuracy_repeats: int = 10
    scan_repeats: int = 8
    accuracy_dim: int = 8
    scan_dim: int = 8
    n_train: int = 2000
    n_test: int = 2000
    total_dims: tuple[int, ...] = (2, 4, 8, 16)
    sample_sizes: tuple[int, ...] = (200, 500, 1000, 2000)
    seed: int = 7
    knn_neighbors: int = 10

    def signature(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cache_dir"] = str(Path(self.cache_dir))
        payload["fig_dir"] = str(Path(self.fig_dir))
        for key in ("families", "methods", "total_dims", "sample_sizes"):
            payload[key] = list(payload[key])
        payload["version"] = DENSITY_BENCHMARK_VERSION
        return payload


@dataclass(frozen=True)
class DensityPlotConfig:
    fig_dir: Path = PROJECT_ROOT / "fig" / "transport_map_mutual_information"
    basename: str = "density_benchmark_composite"
    dpi: int = 300
    formats: tuple[str, ...] = ("png", "pdf")
    width_in: float = 7.2
    height_in: float = 5.6
    confidence_label: str = "mean +/- s.e.m."


class KNNDensityEstimator:
    def __init__(self, samples: np.ndarray, *, n_neighbors: int = 10) -> None:
        array = np.asarray(samples, dtype=float)
        if array.ndim != 2 or array.shape[0] < 2:
            raise ValueError("samples must be a 2D array with at least two rows.")
        self.samples = array
        self.n_samples, self.dimension = array.shape
        self.n_neighbors = min(max(1, int(n_neighbors)), self.n_samples)
        self.log_unit_ball = 0.5 * self.dimension * np.log(np.pi) - gammaln(0.5 * self.dimension + 1.0)

    def log_prob(self, samples: np.ndarray) -> np.ndarray:
        query = np.asarray(samples, dtype=float)
        distances = np.linalg.norm(query[:, None, :] - self.samples[None, :, :], axis=2)
        kth_distances = np.partition(distances, self.n_neighbors - 1, axis=1)[:, self.n_neighbors - 1]
        radii = np.maximum(kth_distances, 1e-12)
        return (
            np.log(self.n_neighbors)
            - np.log(self.n_samples)
            - self.log_unit_ball
            - self.dimension * np.log(radii)
        )


class GaussianKDEDensityEstimator:
    def __init__(self, samples: np.ndarray, *, bandwidth: float) -> None:
        array = np.asarray(samples, dtype=float)
        if array.ndim != 2 or array.shape[0] < 2:
            raise ValueError("samples must be a 2D array with at least two rows.")
        self.samples = array
        self.n_samples, self.dimension = array.shape
        self.bandwidth = max(float(bandwidth), 1e-3)
        self.log_norm = -0.5 * self.dimension * np.log(2.0 * np.pi) - self.dimension * np.log(self.bandwidth)

    def log_prob(self, samples: np.ndarray) -> np.ndarray:
        query = np.asarray(samples, dtype=float)
        diff = (query[:, None, :] - self.samples[None, :, :]) / self.bandwidth
        log_kernel = -0.5 * np.sum(diff**2, axis=2)
        return self.log_norm + logsumexp(log_kernel, axis=1) - np.log(self.n_samples)


def build_density_covariance(total_dim: int, *, decay: float = 0.65, latent_strength: float = 0.35) -> np.ndarray:
    if total_dim < 2:
        raise ValueError("total_dim must be at least 2.")
    indices = np.arange(total_dim)
    toeplitz = decay ** np.abs(indices[:, None] - indices[None, :])
    latent = np.stack(
        [
            np.linspace(0.4, 1.0, total_dim),
            np.cos(np.linspace(0.0, np.pi, total_dim)),
        ],
        axis=1,
    )
    covariance = 0.45 * toeplitz + latent_strength * (latent @ latent.T) + 0.55 * np.eye(total_dim)
    return 0.5 * (covariance + covariance.T)


def make_banana_transform(base_samples: np.ndarray, *, beta: float = 0.35) -> np.ndarray:
    warped = np.asarray(base_samples, dtype=float).copy()
    warped[:, 1] = warped[:, 1] + beta * (warped[:, 0] ** 2 - 1.0)
    return warped


def banana_inverse(samples: np.ndarray, *, beta: float = 0.35) -> np.ndarray:
    array = np.asarray(samples, dtype=float).copy()
    array[:, 1] = array[:, 1] - beta * (array[:, 0] ** 2 - 1.0)
    return array


def standard_gaussian_logpdf(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    dim = array.shape[1]
    return -0.5 * (dim * np.log(2.0 * np.pi) + np.sum(array**2, axis=1))


def multivariate_gaussian_logpdf(samples: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    mean_array = np.asarray(mean, dtype=float)
    covariance_array = np.asarray(covariance, dtype=float)
    cholesky = np.linalg.cholesky(covariance_array)
    centered = array - mean_array
    whitened = np.linalg.solve(cholesky, centered.T).T
    quadratic = np.sum(whitened**2, axis=1)
    log_det = 2.0 * float(np.log(np.diag(cholesky)).sum())
    dim = int(mean_array.shape[0])
    return -0.5 * (dim * np.log(2.0 * np.pi) + log_det + quadratic)


def build_density_family_dataset(
    family: str,
    *,
    total_dim: int,
    n_train: int,
    n_test: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if family == "gaussian":
        covariance = build_density_covariance(total_dim)
        mean = np.zeros(total_dim, dtype=float)
        train = rng.multivariate_normal(mean, covariance, size=n_train)
        test = rng.multivariate_normal(mean, covariance, size=n_test)
        return {
            "family": family,
            "train": train,
            "test": test,
            "log_true_test": multivariate_gaussian_logpdf(test, mean, covariance),
        }

    if family == "gmm":
        covariance_a = build_density_covariance(total_dim, decay=0.6, latent_strength=0.25)
        covariance_b = build_density_covariance(total_dim, decay=0.45, latent_strength=0.20) + 0.12 * np.eye(total_dim)
        mean_template = np.linspace(1.0, 0.2, total_dim)
        mean_a = 0.85 * mean_template
        mean_b = -0.85 * mean_template

        def sample(size: int) -> np.ndarray:
            component = rng.binomial(1, 0.5, size=size)
            samples = np.empty((size, total_dim), dtype=float)
            count_a = int(np.sum(component == 0))
            count_b = size - count_a
            if count_a > 0:
                samples[component == 0] = rng.multivariate_normal(mean_a, covariance_a, size=count_a)
            if count_b > 0:
                samples[component == 1] = rng.multivariate_normal(mean_b, covariance_b, size=count_b)
            return samples

        train = sample(n_train)
        test = sample(n_test)
        log_true = logsumexp(
            np.stack(
                [
                    np.log(0.5) + multivariate_gaussian_logpdf(test, mean_a, covariance_a),
                    np.log(0.5) + multivariate_gaussian_logpdf(test, mean_b, covariance_b),
                ],
                axis=0,
            ),
            axis=0,
        )
        return {"family": family, "train": train, "test": test, "log_true_test": np.asarray(log_true)}

    if family == "banana":
        if total_dim < 2:
            raise ValueError("banana family requires total_dim >= 2.")
        train_base = rng.normal(size=(n_train, total_dim))
        test_base = rng.normal(size=(n_test, total_dim))
        train = make_banana_transform(train_base)
        test = make_banana_transform(test_base)
        return {"family": family, "train": train, "test": test, "log_true_test": standard_gaussian_logpdf(banana_inverse(test))}

    raise KeyError(f"Unknown density family: {family}")


def estimate_scott_bandwidth(samples: np.ndarray) -> float:
    array = np.asarray(samples, dtype=float)
    n_samples, dim = array.shape
    scale = float(np.mean(np.std(array, axis=0, ddof=1)))
    return float(max(scale, 1e-3) * n_samples ** (-1.0 / (dim + 4.0)))


def fit_kde_density_estimator(samples: np.ndarray, *, bandwidth: float | None = None) -> GaussianKDEDensityEstimator:
    array = np.asarray(samples, dtype=float)
    bandwidth_value = estimate_scott_bandwidth(array) if bandwidth is None else float(bandwidth)
    return GaussianKDEDensityEstimator(array, bandwidth=bandwidth_value)


def fit_density_estimator(method: str, samples: np.ndarray, *, seed: int, knn_neighbors: int) -> Any:
    if method == "transport_map":
        del seed
        return fit_affine_transport_map_density(samples)
    if method == "kde":
        del seed, knn_neighbors
        return fit_kde_density_estimator(samples)
    if method == "knn":
        del seed
        return KNNDensityEstimator(samples, n_neighbors=knn_neighbors)
    raise KeyError(f"Unknown density method: {method}")


def score_density_estimator(method: str, estimator: Any, samples: np.ndarray) -> np.ndarray:
    if method in ("transport_map", "knn"):
        return np.asarray(estimator.log_prob(samples), dtype=float)
    if method == "kde":
        return np.asarray(estimator.log_prob(samples), dtype=float)
    raise KeyError(f"Unknown density method: {method}")


def evaluate_density_setting(
    *,
    family: str,
    total_dim: int,
    n_train: int,
    n_test: int,
    repeats: int,
    seed: int,
    methods: tuple[str, ...],
    knn_neighbors: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for repeat in range(repeats):
        dataset = build_density_family_dataset(
            family,
            total_dim=total_dim,
            n_train=n_train,
            n_test=n_test,
            seed=seed + 1000 * repeat,
        )
        train = dataset["train"]
        test = dataset["test"]
        log_true = dataset["log_true_test"]

        for method in methods:
            fit_start = time.perf_counter()
            estimator = fit_density_estimator(method, train, seed=seed + repeat, knn_neighbors=knn_neighbors)
            fit_time = time.perf_counter() - fit_start

            score_start = time.perf_counter()
            log_hat = score_density_estimator(method, estimator, test)
            score_time = time.perf_counter() - score_start

            rows.append(
                {
                    "family": family,
                    "family_title": DENSITY_FAMILY_SPECS[family]["title"],
                    "method": method,
                    "repeat": int(repeat),
                    "total_dim": int(total_dim),
                    "n_train": int(n_train),
                    "n_test": int(n_test),
                    "heldout_nll": float(-np.mean(log_hat)),
                    "rmse_log_density": float(np.sqrt(np.mean((log_hat - log_true) ** 2))),
                    "kl_p_phat": float(np.mean(log_true - log_hat)),
                    "fit_time": float(fit_time),
                    "score_time": float(score_time),
                    "total_time": float(fit_time + score_time),
                }
            )
    return pd.DataFrame(rows)


def summarize_density_results(results: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    grouped = results.groupby(group_keys + ["method"], as_index=False)
    summary = grouped.agg(
        n_repeats=("repeat", "nunique"),
        heldout_nll_mean=("heldout_nll", "mean"),
        heldout_nll_std=("heldout_nll", "std"),
        heldout_nll_sem=("heldout_nll", "sem"),
        rmse_log_density_mean=("rmse_log_density", "mean"),
        rmse_log_density_std=("rmse_log_density", "std"),
        rmse_log_density_sem=("rmse_log_density", "sem"),
        kl_p_phat_mean=("kl_p_phat", "mean"),
        kl_p_phat_std=("kl_p_phat", "std"),
        kl_p_phat_sem=("kl_p_phat", "sem"),
        fit_time_mean=("fit_time", "mean"),
        fit_time_std=("fit_time", "std"),
        fit_time_sem=("fit_time", "sem"),
        score_time_mean=("score_time", "mean"),
        score_time_std=("score_time", "std"),
        score_time_sem=("score_time", "sem"),
        total_time_mean=("total_time", "mean"),
        total_time_std=("total_time", "std"),
        total_time_sem=("total_time", "sem"),
    )
    return summary.sort_values(group_keys + ["method"]).reset_index(drop=True).fillna(0.0)


def run_density_family_benchmark(config: DensityBenchmarkConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.concat(
        [
            evaluate_density_setting(
                family=family,
                total_dim=config.accuracy_dim,
                n_train=config.n_train,
                n_test=config.n_test,
                repeats=config.accuracy_repeats,
                seed=config.seed + 100 * index,
                methods=config.methods,
                knn_neighbors=config.knn_neighbors,
            )
            for index, family in enumerate(config.families)
        ],
        ignore_index=True,
    )
    return raw, summarize_density_results(raw, ["family", "family_title"])


def run_density_dimension_benchmark(config: DensityBenchmarkConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for family_index, family in enumerate(config.families):
        for dim_index, total_dim in enumerate(config.total_dims):
            frames.append(
                evaluate_density_setting(
                    family=family,
                    total_dim=total_dim,
                    n_train=config.n_train,
                    n_test=config.n_test,
                    repeats=config.scan_repeats,
                    seed=config.seed + 10_000 + 1000 * family_index + 100 * dim_index,
                    methods=config.methods,
                    knn_neighbors=config.knn_neighbors,
                )
            )
    raw = pd.concat(frames, ignore_index=True)
    return raw, summarize_density_results(raw, ["family", "family_title", "total_dim"])


def run_density_sample_size_benchmark(config: DensityBenchmarkConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for family_index, family in enumerate(config.families):
        for sample_index, n_train in enumerate(config.sample_sizes):
            frames.append(
                evaluate_density_setting(
                    family=family,
                    total_dim=config.scan_dim,
                    n_train=n_train,
                    n_test=config.n_test,
                    repeats=config.scan_repeats,
                    seed=config.seed + 20_000 + 1000 * family_index + 100 * sample_index,
                    methods=config.methods,
                    knn_neighbors=config.knn_neighbors,
                )
            )
    raw = pd.concat(frames, ignore_index=True)
    return raw, summarize_density_results(raw, ["family", "family_title", "n_train"])


def build_density_timing_summary(
    dimension_summary: pd.DataFrame,
    sample_size_summary: pd.DataFrame,
) -> pd.DataFrame:
    dim_df = (
        dimension_summary.groupby(["total_dim", "method"], as_index=False)
        .agg(total_time_mean=("total_time_mean", "mean"), total_time_sem=("total_time_sem", "mean"))
        .assign(scan="dimension")
        .rename(columns={"total_dim": "x_value"})
    )
    sample_df = (
        sample_size_summary.groupby(["n_train", "method"], as_index=False)
        .agg(total_time_mean=("total_time_mean", "mean"), total_time_sem=("total_time_sem", "mean"))
        .assign(scan="sample_size")
        .rename(columns={"n_train": "x_value"})
    )
    return pd.concat([dim_df, sample_df], ignore_index=True)


def density_pickle_path(path: Path) -> Path:
    return path.with_suffix(".pkl")


def can_use_parquet() -> bool:
    for module_name in ("pyarrow", "fastparquet"):
        try:
            __import__(module_name)
            return True
        except ImportError:
            continue
    return False


def density_raw_cache_available(path: Path) -> bool:
    return density_pickle_path(path).exists() or (path.exists() and can_use_parquet())


def read_density_frame(path: Path) -> pd.DataFrame:
    pickle_path = density_pickle_path(path)
    if pickle_path.exists():
        return pd.read_pickle(pickle_path)
    if path.exists() and can_use_parquet():
        return pd.read_parquet(path)
    raise FileNotFoundError(
        f"No readable cache found for {path.name}. Expected {pickle_path.name} or a usable parquet engine."
    )


def write_density_frame(frame: pd.DataFrame, path: Path) -> list[str]:
    written = []
    pickle_path = density_pickle_path(path)
    frame.to_pickle(pickle_path)
    written.append(pickle_path.name)
    if can_use_parquet():
        frame.to_parquet(path, index=False)
        written.insert(0, path.name)
    return written


def cache_manifest_path(config: DensityBenchmarkConfig) -> Path:
    return Path(config.cache_dir) / "density_benchmark_manifest.json"


def cache_signature_matches(config: DensityBenchmarkConfig) -> bool:
    path = cache_manifest_path(config)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("config") == config.signature()


def load_or_run_density_benchmark(
    *,
    config: DensityBenchmarkConfig | None = None,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    config = DensityBenchmarkConfig() if config is None else config
    cache_dir = Path(config.cache_dir)
    fig_dir = Path(config.fig_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    accuracy_raw_path = cache_dir / "accuracy_raw.parquet"
    accuracy_summary_path = cache_dir / "accuracy_summary.csv"
    dimension_raw_path = cache_dir / "dimension_raw.parquet"
    dimension_summary_path = cache_dir / "dimension_summary.csv"
    sample_raw_path = cache_dir / "sample_size_raw.parquet"
    sample_summary_path = cache_dir / "sample_size_summary.csv"
    timing_summary_path = cache_dir / "timing_summary.csv"

    raw_cache_ready = all(density_raw_cache_available(path) for path in (accuracy_raw_path, dimension_raw_path, sample_raw_path))
    csv_cache_ready = all(
        path.exists()
        for path in (
            accuracy_summary_path,
            dimension_summary_path,
            sample_summary_path,
            timing_summary_path,
        )
    )
    if (not force) and raw_cache_ready and csv_cache_ready and cache_signature_matches(config):
        return {
            "accuracy_raw": read_density_frame(accuracy_raw_path),
            "accuracy_summary": pd.read_csv(accuracy_summary_path),
            "dimension_raw": read_density_frame(dimension_raw_path),
            "dimension_summary": pd.read_csv(dimension_summary_path),
            "sample_size_raw": read_density_frame(sample_raw_path),
            "sample_size_summary": pd.read_csv(sample_summary_path),
            "timing_summary": pd.read_csv(timing_summary_path),
        }

    accuracy_raw, accuracy_summary = run_density_family_benchmark(config)
    dimension_raw, dimension_summary = run_density_dimension_benchmark(config)
    sample_raw, sample_summary = run_density_sample_size_benchmark(config)
    timing_summary = build_density_timing_summary(dimension_summary, sample_summary)

    cache_files = []
    cache_files.extend(write_density_frame(accuracy_raw, accuracy_raw_path))
    accuracy_summary.to_csv(accuracy_summary_path, index=False)
    cache_files.append(accuracy_summary_path.name)
    cache_files.extend(write_density_frame(dimension_raw, dimension_raw_path))
    dimension_summary.to_csv(dimension_summary_path, index=False)
    cache_files.append(dimension_summary_path.name)
    cache_files.extend(write_density_frame(sample_raw, sample_raw_path))
    sample_summary.to_csv(sample_summary_path, index=False)
    timing_summary.to_csv(timing_summary_path, index=False)
    cache_files.extend([sample_summary_path.name, timing_summary_path.name])

    manifest = {
        "notebook": "exp/density_estimation_benchmark.ipynb",
        "benchmark": "density_benchmark",
        "config": config.signature(),
        "cache_files": cache_files,
    }
    cache_manifest_path(config).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "accuracy_raw": accuracy_raw,
        "accuracy_summary": accuracy_summary,
        "dimension_raw": dimension_raw,
        "dimension_summary": dimension_summary,
        "sample_size_raw": sample_raw,
        "sample_size_summary": sample_summary,
        "timing_summary": timing_summary,
    }


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _ordered_families(frame: pd.DataFrame) -> list[str]:
    return [family for family in DENSITY_FAMILY_SPECS if family in set(frame["family"])]


def _plot_grouped_bars(
    ax: Any,
    summary_df: pd.DataFrame,
    *,
    metric: str,
    sem: str,
    ylabel: str,
    panel_label: str,
) -> None:
    families = _ordered_families(summary_df)
    x = np.arange(len(families), dtype=float)
    width = 0.23
    for method_index, method in enumerate(DENSITY_METHOD_ORDER):
        if method not in set(summary_df["method"]):
            continue
        frame = summary_df.loc[summary_df["method"] == method].set_index("family").reindex(families)
        offsets = x + (method_index - 1) * width
        ax.bar(
            offsets,
            frame[metric],
            width=width,
            color=DENSITY_COLORS[method],
            alpha=0.92,
            yerr=frame[sem],
            capsize=2.0,
            linewidth=0.0,
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
            label=DENSITY_METHOD_LABELS[method],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([DENSITY_FAMILY_SPECS[key]["title"] for key in families], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#d5d8dc", linewidth=0.45, alpha=0.8)
    ax.text(-0.12, 1.03, panel_label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="bottom")


def _plot_family_curves(
    ax: Any,
    summary_df: pd.DataFrame,
    *,
    family: str,
    x_key: str,
    metric: str,
    sem: str,
    xlabel: str,
    ylabel: str,
    panel_label: str,
    log_y: bool = False,
) -> None:
    frame = summary_df.loc[summary_df["family"] == family].sort_values(x_key)
    for method in DENSITY_METHOD_ORDER:
        method_df = frame.loc[frame["method"] == method].sort_values(x_key)
        if method_df.empty:
            continue
        x = method_df[x_key].to_numpy(dtype=float)
        y = method_df[metric].to_numpy(dtype=float)
        yerr = method_df[sem].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker=DENSITY_MARKERS[method],
            linewidth=1.45,
            markersize=3.8,
            color=DENSITY_COLORS[method],
            label=DENSITY_METHOD_LABELS[method],
        )
        lower = np.maximum(y - yerr, 1e-12 if log_y else -np.inf)
        ax.fill_between(x, lower, y + yerr, color=DENSITY_COLORS[method], alpha=0.14, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(color="#d5d8dc", linewidth=0.45, alpha=0.75)
    ax.text(-0.12, 1.03, panel_label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="bottom")


def _plot_timing(ax: Any, timing_summary: pd.DataFrame, *, panel_label: str) -> None:
    frame = timing_summary.loc[timing_summary["scan"] == "sample_size"].sort_values("x_value")
    for method in DENSITY_METHOD_ORDER:
        method_df = frame.loc[frame["method"] == method].sort_values("x_value")
        if method_df.empty:
            continue
        ax.plot(
            method_df["x_value"],
            method_df["total_time_mean"],
            marker=DENSITY_MARKERS[method],
            linewidth=1.45,
            markersize=3.8,
            color=DENSITY_COLORS[method],
            label=DENSITY_METHOD_LABELS[method],
        )
    ax.set_yscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Total time (s)")
    ax.grid(color="#d5d8dc", linewidth=0.45, alpha=0.75)
    ax.text(-0.12, 1.03, panel_label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="bottom")


def plot_density_benchmark_composite(
    results: dict[str, pd.DataFrame],
    *,
    plot_config: DensityPlotConfig | None = None,
) -> list[Path]:
    plot_config = DensityPlotConfig() if plot_config is None else plot_config
    plot_config.fig_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    import matplotlib.pyplot as plt

    accuracy_summary = results["accuracy_summary"]
    sample_summary = results["sample_size_summary"]
    dimension_summary = results["dimension_summary"]
    timing_summary = results["timing_summary"]
    curve_family = "banana" if "banana" in set(sample_summary["family"]) else _ordered_families(sample_summary)[-1]

    fig = plt.figure(figsize=(plot_config.width_in, plot_config.height_in), constrained_layout=True)
    spec = fig.add_gridspec(2, 6, width_ratios=[1.25, 1.25, 0.95, 1.05, 1.05, 1.05], height_ratios=[1.05, 1.0])
    ax_rmse = fig.add_subplot(spec[0, 0:3])
    ax_kl = fig.add_subplot(spec[0, 3:5])
    ax_time = fig.add_subplot(spec[0, 5])
    ax_sample = fig.add_subplot(spec[1, 0:3])
    ax_dim = fig.add_subplot(spec[1, 3:6])

    _plot_grouped_bars(
        ax_rmse,
        accuracy_summary,
        metric="rmse_log_density_mean",
        sem="rmse_log_density_sem",
        ylabel=r"RMSE($\log p$)",
        panel_label="a",
    )
    _plot_grouped_bars(
        ax_kl,
        accuracy_summary,
        metric="kl_p_phat_mean",
        sem="kl_p_phat_sem",
        ylabel=r"$KL(p\|\hat p)$",
        panel_label="b",
    )
    _plot_timing(ax_time, timing_summary, panel_label="c")
    _plot_family_curves(
        ax_sample,
        sample_summary,
        family=curve_family,
        x_key="n_train",
        metric="rmse_log_density_mean",
        sem="rmse_log_density_sem",
        xlabel="Training samples",
        ylabel=r"RMSE($\log p$)",
        panel_label="d",
    )
    _plot_family_curves(
        ax_dim,
        dimension_summary,
        family=curve_family,
        x_key="total_dim",
        metric="rmse_log_density_mean",
        sem="rmse_log_density_sem",
        xlabel="Total dimension",
        ylabel=r"RMSE($\log p$)",
        panel_label="e",
    )

    for ax in (ax_rmse, ax_kl, ax_time, ax_sample, ax_dim):
        ax.tick_params(width=0.7, length=2.5)

    handles, labels = ax_rmse.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 1.02), ncol=3, frameon=False)

    paths: list[Path] = []
    for fmt in plot_config.formats:
        path = plot_config.fig_dir / f"{plot_config.basename}.{fmt}"
        if fmt.lower() == "png":
            fig.savefig(path, dpi=plot_config.dpi, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)

    manifest = {
        "notebook": "exp/density_estimation_benchmark.ipynb",
        "benchmark": "density_benchmark",
        "figure": [path.name for path in paths],
        "claim": "Accuracy is assessed by log-density RMSE, convergence by repeated-run uncertainty and sample-size scans, and cost by total fit plus score time.",
        "interval": plot_config.confidence_label,
    }
    (plot_config.fig_dir / "density_figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths


def compact_accuracy_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "family_title",
        "method",
        "n_repeats",
        "rmse_log_density_mean",
        "rmse_log_density_sem",
        "kl_p_phat_mean",
        "kl_p_phat_sem",
        "total_time_mean",
    ]
    table = summary.loc[:, columns].copy()
    table["method"] = table["method"].map(DENSITY_METHOD_LABELS)
    return table
