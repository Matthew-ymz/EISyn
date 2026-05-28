"""
Time-resolved effective information for network revival ignition experiments.

The source variable is the forced ignition strength Delta_i. For state-space
node EI, the target is the whole-system state after free evolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import io
import json
from pathlib import Path
import zipfile
from typing import Iterable, Sequence

import numpy as np

from .dynamics import get_model
from .network import largest_connected_component
from .simulate import _rk4_step, solve_odes

try:  # pytest/import from repository root
    from exp.TM.transport_map_density import estimate_mutual_information_transport_map
except ModuleNotFoundError:  # notebook import with exp/ on sys.path
    from TM.transport_map_density import estimate_mutual_information_transport_map


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "network_revival_node_ei" / "fig5l_wout5"
DEFAULT_STATE_SPACE_OUTPUT_DIR = (
    REPO_ROOT / "results" / "network_revival_state_space_ei" / "fig5l_wout5_tau20"
)
DEFAULT_IGNITION_THRESHOLD_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "network_revival_ei_ignition_threshold"
    / "fig5l_state_wout5p0_win20p0_tau5p0_n10000_seed42"
)
DEFAULT_PAIR_SYNERGY_OUTPUT_ROOT = REPO_ROOT / "results" / "network_revival_state_space_pair_synergy"
DEFAULT_PAIR_IGNITION_OUTPUT_ROOT = REPO_ROOT / "results" / "network_revival_pair_ignition_cost"
DEFAULT_ARTICLE_ZIP = Path("/Users/yangmingzhe/Downloads/NaturePhys2021-main.zip")
DEFAULT_LOCAL_BRAIN_MAT = Path(__file__).resolve().parent / "data" / "Brain.mat"

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover - scipy is available in the project env
    sp = None


@dataclass(frozen=True)
class DynamicEIConfig:
    delta_max: float = 20.0
    n_delta: int = 96
    tau_grid: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0)
    t_ignite: float = 12.0
    dt: float = 0.08
    seed: int = 42
    chunk_size: int = 32
    target_noise_fraction: float = 0.0
    show_progress: bool = True
    win: float = 20.0
    wout: float = 5.0
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    node_indices: tuple[int, ...] | None = None
    free_init: float = 0.0

    def __post_init__(self) -> None:
        if self.delta_max <= 0.0:
            raise ValueError("delta_max must be positive.")
        if self.n_delta < 2:
            raise ValueError("n_delta must be at least 2.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.t_ignite < 0.0:
            raise ValueError("t_ignite must be nonnegative.")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        if self.target_noise_fraction < 0.0:
            raise ValueError("target_noise_fraction must be nonnegative.")
        tau = tuple(float(value) for value in self.tau_grid)
        if not tau:
            raise ValueError("tau_grid must contain at least one value.")
        if min(tau) < 0.0:
            raise ValueError("tau_grid values must be nonnegative.")
        if tuple(sorted(tau)) != tau:
            raise ValueError("tau_grid must be sorted in ascending order.")
        object.__setattr__(self, "tau_grid", tau)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.node_indices is not None:
            object.__setattr__(self, "node_indices", tuple(int(index) for index in self.node_indices))


@dataclass(frozen=True)
class StateSpaceEIConfig:
    sample_count: int = 10000
    state_low: float = 0.0
    state_high: float = 30.0
    tau: float = 20.0
    dt: float = 0.08
    seed: int = 42
    batch_size: int = 512
    target_noise_fraction: float = 0.0
    show_progress: bool = True
    win: float = 20.0
    wout: float = 5.0
    output_dir: Path = field(default_factory=lambda: DEFAULT_STATE_SPACE_OUTPUT_DIR)

    def __post_init__(self) -> None:
        if self.sample_count < 2:
            raise ValueError("sample_count must be at least 2.")
        if self.state_high <= self.state_low:
            raise ValueError("state_high must be greater than state_low.")
        if self.tau < 0.0:
            raise ValueError("tau must be nonnegative.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.target_noise_fraction < 0.0:
            raise ValueError("target_noise_fraction must be nonnegative.")
        object.__setattr__(self, "output_dir", Path(self.output_dir))


@dataclass(frozen=True)
class EIIgnitionThresholdConfig:
    state_space_run_id: str = "fig5l_state_wout5p0_win20p0_tau5p0_n10000_seed42"
    per_stratum: int = 30
    delta_low: float = 0.0
    delta_high: float = 30.0
    binary_steps: int = 10
    success_threshold: float = 5.0
    t_force: float = 12.0
    dt: float = 0.08
    tol_ss: float = 2e-3
    win: float = 20.0
    wout: float = 5.0
    show_progress: bool = True
    output_dir: Path = field(default_factory=lambda: DEFAULT_IGNITION_THRESHOLD_OUTPUT_DIR)

    def __post_init__(self) -> None:
        if self.per_stratum < 1:
            raise ValueError("per_stratum must be positive.")
        if self.delta_high <= self.delta_low:
            raise ValueError("delta_high must be greater than delta_low.")
        if self.binary_steps < 1:
            raise ValueError("binary_steps must be positive.")
        if self.success_threshold < 0.0:
            raise ValueError("success_threshold must be nonnegative.")
        if self.t_force < 0.0:
            raise ValueError("t_force must be nonnegative.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.tol_ss < 0.0:
            raise ValueError("tol_ss must be nonnegative.")
        object.__setattr__(self, "output_dir", Path(self.output_dir))


@dataclass(frozen=True)
class StateSpacePairSynergyConfig:
    state_space_run_id: str = "fig5l_state_wout5p0_win20p0_tau20p0_n10000_seed42"
    pair_count: int = 200
    pair_seed: int = 42
    target_noise_fraction: float = 0.0
    seed: int = 42
    win: float = 20.0
    wout: float = 5.0
    show_progress: bool = True
    output_dir: Path | None = None
    state_space_output_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.pair_count < 1:
            raise ValueError("pair_count must be positive.")
        if self.target_noise_fraction < 0.0:
            raise ValueError("target_noise_fraction must be nonnegative.")
        run_id = f"{self.state_space_run_id}_pairs{int(self.pair_count)}_seed{int(self.pair_seed)}"
        output_dir = (
            DEFAULT_PAIR_SYNERGY_OUTPUT_ROOT / run_id
            if self.output_dir is None
            else Path(self.output_dir)
        )
        state_space_output_dir = (
            REPO_ROOT / "results" / "network_revival_state_space_ei" / self.state_space_run_id
            if self.state_space_output_dir is None
            else Path(self.state_space_output_dir)
        )
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "state_space_output_dir", state_space_output_dir)


@dataclass(frozen=True)
class PairIgnitionCostConfig:
    pair_synergy_run_id: str = "fig5l_state_wout5p0_win20p0_tau20p0_n10000_seed42_pairs200_seed42"
    cost_low: float = 0.0
    cost_high: float = 60.0
    single_delta_low: float = 0.0
    single_delta_high: float = 30.0
    binary_steps: int = 10
    success_threshold: float = 5.0
    t_force: float = 12.0
    dt: float = 0.08
    tol_ss: float = 2e-3
    win: float = 20.0
    wout: float = 5.0
    show_progress: bool = True
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.cost_high <= self.cost_low:
            raise ValueError("cost_high must be greater than cost_low.")
        if self.single_delta_high <= self.single_delta_low:
            raise ValueError("single_delta_high must be greater than single_delta_low.")
        if self.binary_steps < 1:
            raise ValueError("binary_steps must be positive.")
        if self.success_threshold < 0.0:
            raise ValueError("success_threshold must be nonnegative.")
        if self.t_force < 0.0:
            raise ValueError("t_force must be nonnegative.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.tol_ss < 0.0:
            raise ValueError("tol_ss must be nonnegative.")
        output_dir = (
            DEFAULT_PAIR_IGNITION_OUTPUT_ROOT / self.pair_synergy_run_id
            if self.output_dir is None
            else Path(self.output_dir)
        )
        object.__setattr__(self, "output_dir", output_dir)


def load_fig5_brain_modular_adjacency(
    *,
    win: float = 20.0,
    wout: float = 5.0,
    local_brain_mat: Path | None = None,
    article_zip: Path | None = None,
) -> dict[str, object]:
    """Load the Fig. 5 brain GCC and return the block-ordered weighted adjacency."""

    from scipy.io import loadmat

    local_path = Path(local_brain_mat) if local_brain_mat is not None else DEFAULT_LOCAL_BRAIN_MAT
    zip_path = Path(article_zip) if article_zip is not None else DEFAULT_ARTICLE_ZIP

    if local_path.exists():
        brain_a = loadmat(local_path)["A"]
        source = str(local_path)
    elif zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open("NaturePhys2021-main/data/Brain.mat") as handle:
                brain_a = loadmat(io.BytesIO(handle.read()))["A"]
        source = str(zip_path)
    else:
        raise FileNotFoundError(
            "Need Brain.mat at exp/network_revival/data/Brain.mat or "
            "/Users/yangmingzhe/Downloads/NaturePhys2021-main.zip."
        )

    anw, original_idx = largest_connected_component((brain_a > 0.03).astype(float))
    comm1_original_order = original_idx < 500
    comm2_original_order = ~comm1_original_order

    a1 = anw[np.ix_(comm1_original_order, comm1_original_order)]
    a2 = anw[np.ix_(comm2_original_order, comm2_original_order)]
    a12 = anw[np.ix_(comm1_original_order, comm2_original_order)]
    a21 = anw[np.ix_(comm2_original_order, comm1_original_order)]
    adjacency = np.block([[float(win) * a1, float(wout) * a12], [float(wout) * a21, float(win) * a2]])
    adjacency_sparse = sp.csr_matrix(adjacency) if sp is not None else None

    comm1_mask = np.r_[np.ones(a1.shape[0], dtype=bool), np.zeros(a2.shape[0], dtype=bool)]
    comm2_mask = ~comm1_mask

    return {
        "adjacency": adjacency.astype(float, copy=False),
        "adjacency_sparse": adjacency_sparse,
        "comm1_mask": comm1_mask,
        "comm2_mask": comm2_mask,
        "original_indices": original_idx,
        "source": source,
        "win": float(win),
        "wout": float(wout),
    }


def sample_uniform_state_space(
    *,
    node_count: int,
    sample_count: int,
    state_low: float,
    state_high: float,
    seed: int,
) -> np.ndarray:
    """Sample full-network initial states independently from a uniform box."""

    if int(node_count) < 1:
        raise ValueError("node_count must be positive.")
    if int(sample_count) < 2:
        raise ValueError("sample_count must be at least 2.")
    if float(state_high) <= float(state_low):
        raise ValueError("state_high must be greater than state_low.")

    rng = np.random.default_rng(int(seed))
    return rng.uniform(
        float(state_low),
        float(state_high),
        size=(int(sample_count), int(node_count)),
    )


def simulate_state_space_final_mean_activity(
    adjacency: np.ndarray,
    model: dict,
    *,
    initial_states: np.ndarray,
    tau: float,
    dt: float,
    batch_size: int = 512,
) -> np.ndarray:
    """Simulate free neural dynamics from full-state samples to one final time."""

    return simulate_state_space_final_states(
        adjacency,
        model,
        initial_states=initial_states,
        tau=tau,
        dt=dt,
        batch_size=batch_size,
    ).mean(axis=1)


def simulate_state_space_final_states(
    adjacency: np.ndarray,
    model: dict,
    *,
    initial_states: np.ndarray,
    tau: float,
    dt: float,
    batch_size: int = 512,
) -> np.ndarray:
    """Simulate free neural dynamics from full-state samples to final states."""

    states = np.asarray(initial_states, dtype=float)
    if states.ndim != 2:
        raise ValueError("initial_states must have shape [sample, node].")
    if states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError("initial_states must be nonempty.")
    if float(tau) < 0.0:
        raise ValueError("tau must be nonnegative.")
    if float(dt) <= 0.0:
        raise ValueError("dt must be positive.")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive.")

    use_sparse = sp is not None and sp.issparse(adjacency)
    adjacency_op = adjacency if use_sparse else np.asarray(adjacency, dtype=float)
    if adjacency_op.shape[0] != states.shape[1] or adjacency_op.shape[1] != states.shape[1]:
        raise ValueError("adjacency and initial_states disagree on node count.")

    m0 = model["M0"]
    m1 = model["M1"]
    m2 = model["M2"]
    final_states = np.empty_like(states, dtype=float)

    def rhs_batch(_, x_batch: np.ndarray) -> np.ndarray:
        m2_x = m2(x_batch)
        if use_sparse:
            interaction = adjacency_op.dot(m2_x.T).T
        else:
            interaction = m2_x @ adjacency_op.T
        return m0(x_batch) + m1(x_batch) * interaction

    for start in range(0, states.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), states.shape[0])
        x = states[start:stop].copy()
        t = 0.0
        while t < float(tau) - 1e-12:
            dt_use = min(float(dt), float(tau) - t)
            x = _rk4_step(rhs_batch, t, x, dt_use)
            x = np.maximum(x, 0.0)
            t += dt_use
        final_states[start:stop] = x

    return final_states


def estimate_state_space_node_ei(
    initial_states: np.ndarray,
    final_states: np.ndarray,
    *,
    target_noise_fraction: float = 0.0,
    seed: int = 0,
    clip_negative: bool = True,
) -> dict[str, object]:
    """Estimate I(X_i(0); X(tau)) for every source node."""

    states = np.asarray(initial_states, dtype=float)
    target = np.asarray(final_states, dtype=float)
    if states.ndim != 2:
        raise ValueError("initial_states must have shape [sample, node].")
    if target.ndim != 2:
        raise ValueError("final_states must have shape [sample, node].")
    if states.shape[0] != target.shape[0]:
        raise ValueError("initial_states and final_states disagree on sample count.")
    if float(target_noise_fraction) < 0.0:
        raise ValueError("target_noise_fraction must be nonnegative.")

    noise_fraction = float(target_noise_fraction)
    if noise_fraction == 0.0:
        sigma = np.zeros(target.shape[1], dtype=float)
        noisy_target = target
    else:
        rng = np.random.default_rng(seed)
        sigma = np.maximum(1e-6, noise_fraction * np.std(target, axis=0, ddof=1))
        noisy_target = target + sigma * rng.normal(size=target.shape)

    node_ei, bias_correction = _estimate_scalar_to_multivariate_gaussian_mi(
        states,
        noisy_target,
        clip_negative=clip_negative,
    )
    backend = "affine_triangular_transport_map"

    return {
        "node_ei": node_ei,
        "target_noise_sigma": sigma,
        "bias_correction": bias_correction,
        "backend": backend,
    }


def sample_random_node_pairs(*, node_count: int, pair_count: int, seed: int) -> list[tuple[int, int]]:
    """Sample deterministic unordered node pairs without replacement."""

    n_nodes = int(node_count)
    n_pairs = int(pair_count)
    if n_nodes < 2:
        raise ValueError("node_count must be at least 2.")
    if n_pairs < 1:
        raise ValueError("pair_count must be positive.")
    all_pairs = [(left, right) for left in range(n_nodes) for right in range(left + 1, n_nodes)]
    if n_pairs > len(all_pairs):
        raise ValueError("pair_count exceeds the number of available unordered node pairs.")
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(len(all_pairs), size=n_pairs, replace=False)
    return [all_pairs[int(index)] for index in selected]


def lift_state_space_source_features(source: np.ndarray) -> np.ndarray:
    """Lift one-node or two-node source blocks before affine TM MI estimation."""

    array = np.asarray(source, dtype=float)
    if array.ndim != 2:
        raise ValueError("source must be a two-dimensional array.")
    if array.shape[1] == 1:
        x = array[:, [0]]
        return np.concatenate([x, x**2, x**3], axis=1)
    if array.shape[1] == 2:
        x = array[:, [0]]
        y = array[:, [1]]
        return np.concatenate([x, y, x * y, x**2, y**2], axis=1)
    raise ValueError("source must contain one or two columns.")


def estimate_state_space_pair_synergy(
    initial_states: np.ndarray,
    final_states: np.ndarray,
    *,
    pairs: Sequence[tuple[int, int]],
    target_noise_fraction: float = 0.0,
    seed: int = 0,
    clip_negative_ei: bool = True,
) -> dict[str, object]:
    """Estimate lifted source-side pair synergy against the full final state."""

    states = np.asarray(initial_states, dtype=float)
    target = np.asarray(final_states, dtype=float)
    if states.ndim != 2:
        raise ValueError("initial_states must have shape [sample, node].")
    if target.ndim != 2:
        raise ValueError("final_states must have shape [sample, target_dim].")
    if states.shape[0] != target.shape[0]:
        raise ValueError("initial_states and final_states disagree on sample count.")
    if float(target_noise_fraction) < 0.0:
        raise ValueError("target_noise_fraction must be nonnegative.")

    normalized_pairs = [(int(left), int(right)) for left, right in pairs]
    if not normalized_pairs:
        raise ValueError("pairs must contain at least one pair.")
    node_count = states.shape[1]
    for left, right in normalized_pairs:
        if left < 0 or right < 0 or left >= node_count or right >= node_count:
            raise ValueError("pairs contain a node outside initial_states.")
        if left == right:
            raise ValueError("pairs must not contain self-pairs.")

    noise_fraction = float(target_noise_fraction)
    if noise_fraction == 0.0:
        sigma = np.zeros(target.shape[1], dtype=float)
        noisy_target = target
    else:
        rng = np.random.default_rng(seed)
        sigma = np.maximum(1e-6, noise_fraction * np.std(target, axis=0, ddof=1))
        noisy_target = target + sigma * rng.normal(size=target.shape)

    singleton_cache: dict[int, tuple[float, float]] = {}
    pair_rows: list[dict[str, object]] = []
    backend = "affine_triangular_transport_map"

    def _single_ei(node: int) -> float:
        if node not in singleton_cache:
            summary = estimate_mutual_information_transport_map(
                lift_state_space_source_features(states[:, [node]]),
                noisy_target,
            )
            value = float(summary["mi_hat"])
            if clip_negative_ei:
                value = max(0.0, value)
            singleton_cache[node] = (value, float(summary["bias_correction"]))
        return singleton_cache[node][0]

    for left, right in normalized_pairs:
        ordered_left, ordered_right = sorted((left, right))
        left_ei = _single_ei(ordered_left)
        right_ei = _single_ei(ordered_right)
        joint_summary = estimate_mutual_information_transport_map(
            lift_state_space_source_features(states[:, [ordered_left, ordered_right]]),
            noisy_target,
        )
        backend = str(joint_summary["backend"])
        joint_ei = float(joint_summary["mi_hat"])
        if clip_negative_ei:
            joint_ei = max(0.0, joint_ei)
        synergy = float(joint_ei - left_ei - right_ei)
        pair_rows.append(
            {
                "pair_i": int(ordered_left),
                "pair_j": int(ordered_right),
                "left_ei": float(left_ei),
                "right_ei": float(right_ei),
                "joint_ei": float(joint_ei),
                "synergy": synergy,
                "synergy_ratio": float(synergy / joint_ei) if abs(joint_ei) > 1e-12 else 0.0,
            }
        )

    pair_rows.sort(key=lambda row: (-float(row["synergy"]), int(row["pair_i"]), int(row["pair_j"])))
    for rank, row in enumerate(pair_rows, start=1):
        row["rank_synergy"] = int(rank)

    return {
        "pair_rows": pair_rows,
        "pairs": np.asarray([(row["pair_i"], row["pair_j"]) for row in pair_rows], dtype=int),
        "target_noise_sigma": sigma,
        "backend": backend,
    }


def _estimate_scalar_to_multivariate_gaussian_mi(
    source_matrix: np.ndarray,
    target_matrix: np.ndarray,
    *,
    jitter: float = 1e-6,
    clip_negative: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate I(source_i; target_vector) for all scalar source columns."""

    source = np.asarray(source_matrix, dtype=float)
    target = np.asarray(target_matrix, dtype=float)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source_matrix and target_matrix must be two-dimensional.")
    if source.shape[0] != target.shape[0]:
        raise ValueError("source_matrix and target_matrix must have matching sample counts.")
    if source.shape[0] < 2:
        raise ValueError("at least two samples are required.")

    sample_size = int(source.shape[0])
    target_dim = int(target.shape[1])
    source_centered = source - source.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    denom = float(sample_size - 1)

    source_var = np.sum(source_centered**2, axis=0) / denom
    source_var = np.maximum(source_var, float(jitter))
    target_cov = (target_centered.T @ target_centered) / denom
    target_cov = np.atleast_2d(target_cov) + float(jitter) * np.eye(target_dim, dtype=float)
    source_target_cov = (source_centered.T @ target_centered) / denom

    solved = np.linalg.solve(target_cov, source_target_cov.T)
    explained_var = np.sum(source_target_cov.T * solved, axis=0)
    r_squared = np.clip(explained_var / source_var, 0.0, 1.0 - 1e-12)
    raw_mi = -0.5 * np.log1p(-r_squared)
    bias = 0.5 * (
        _gaussian_logdet_bias_correction_local(1, sample_size)
        + _gaussian_logdet_bias_correction_local(target_dim, sample_size)
        - _gaussian_logdet_bias_correction_local(target_dim + 1, sample_size)
    )
    node_ei = raw_mi - bias
    if clip_negative:
        node_ei = np.maximum(node_ei, 0.0)
    return node_ei.astype(float, copy=False), np.full(source.shape[1], float(bias), dtype=float)


def _gaussian_logdet_bias_correction_local(dimension: int, sample_size: int) -> float:
    if sample_size <= dimension:
        return 0.0
    from scipy.special import digamma

    nu = sample_size - 1
    return float(
        sum(digamma((nu + 1 - index) / 2.0) for index in range(1, dimension + 1))
        + dimension * np.log(2.0)
        - dimension * np.log(nu)
    )


def run_state_space_node_ei(config: StateSpaceEIConfig, *, force_recompute: bool = False) -> dict[str, object]:
    """Run or load the Fig.5l state-space node EI experiment."""

    output_dir = Path(config.output_dir)
    cache_paths = _state_space_cache_paths(output_dir)
    if not force_recompute and _state_space_cache_matches_config(cache_paths, config):
        return _load_cached_state_space_result(cache_paths)

    output_dir.mkdir(parents=True, exist_ok=True)
    brain = load_fig5_brain_modular_adjacency(win=config.win, wout=config.wout)
    adjacency_dense = np.asarray(brain["adjacency"], dtype=float)
    adjacency = brain["adjacency_sparse"] if brain.get("adjacency_sparse") is not None else adjacency_dense
    model = get_model("Neural", mu=10.0, delta=1.0)
    node_count = int(adjacency.shape[0])

    initial_states = sample_uniform_state_space(
        node_count=node_count,
        sample_count=int(config.sample_count),
        state_low=float(config.state_low),
        state_high=float(config.state_high),
        seed=int(config.seed),
    )
    if config.show_progress:
        print(
            f"State-space EI simulation samples 1-{config.sample_count}/{config.sample_count} "
            f"(n={node_count}, tau={config.tau:.1f}, batch_size={config.batch_size})",
            flush=True,
        )
    final_states = simulate_state_space_final_states(
        adjacency,
        model,
        initial_states=initial_states,
        tau=float(config.tau),
        dt=float(config.dt),
        batch_size=int(config.batch_size),
    )
    ei_summary = estimate_state_space_node_ei(
        initial_states,
        final_states,
        target_noise_fraction=float(config.target_noise_fraction),
        seed=int(config.seed) + 1009,
    )
    final_mean = final_states.mean(axis=1)
    node_ei = np.asarray(ei_summary["node_ei"], dtype=float)
    nodes = tuple(range(node_count))
    node_summary = _build_state_space_node_summary(
        nodes,
        node_ei,
        np.asarray(brain["comm1_mask"], dtype=bool),
    )
    manifest = _build_state_space_manifest(config, brain, ei_summary["backend"])

    np.savez_compressed(
        cache_paths["samples_npz"],
        initial_states=initial_states,
        final_states=final_states,
        final_mean_activity=final_mean,
        node_ei=node_ei,
        target_noise_sigma=np.asarray(ei_summary["target_noise_sigma"], dtype=float),
        bias_correction=np.asarray(ei_summary["bias_correction"], dtype=float),
    )
    _write_state_space_summary_csv(cache_paths["summary_csv"], node_summary)
    cache_paths["manifest_json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "initial_states": initial_states,
        "final_states": final_states,
        "final_mean_activity": final_mean,
        "node_ei": node_ei,
        "node_summary": node_summary,
        "target_noise_sigma": np.asarray(ei_summary["target_noise_sigma"], dtype=float),
        "cache_paths": cache_paths,
        "manifest": manifest,
    }


def run_state_space_pair_synergy(
    config: StateSpacePairSynergyConfig,
    *,
    force_recompute: bool = False,
) -> dict[str, object]:
    """Run or load state-space pair synergy from cached state-space samples."""

    output_dir = Path(config.output_dir)
    cache_paths = _pair_synergy_cache_paths(output_dir)
    if not force_recompute and _pair_synergy_cache_matches_config(cache_paths, config):
        return _load_cached_pair_synergy_result(cache_paths)

    output_dir.mkdir(parents=True, exist_ok=True)
    state_space_npz = Path(config.state_space_output_dir) / "state_space_ei_samples.npz"
    if not state_space_npz.exists():
        raise FileNotFoundError(f"Missing state-space samples: {state_space_npz}")
    arrays = np.load(state_space_npz, allow_pickle=False)
    initial_states = np.asarray(arrays["initial_states"], dtype=float)
    final_states = np.asarray(arrays["final_states"], dtype=float)

    brain = load_fig5_brain_modular_adjacency(win=config.win, wout=config.wout)
    comm1_mask = np.asarray(brain["comm1_mask"], dtype=bool)
    if comm1_mask.shape[0] != initial_states.shape[1]:
        comm1_mask = np.zeros(initial_states.shape[1], dtype=bool)

    pairs = sample_random_node_pairs(
        node_count=initial_states.shape[1],
        pair_count=int(config.pair_count),
        seed=int(config.pair_seed),
    )
    if config.show_progress:
        print(
            f"State-space pair synergy: {len(pairs)} pairs from n={initial_states.shape[1]}",
            flush=True,
        )
    synergy = estimate_state_space_pair_synergy(
        initial_states,
        final_states,
        pairs=pairs,
        target_noise_fraction=float(config.target_noise_fraction),
        seed=int(config.seed) + 2003,
    )
    pair_rows = _add_pair_communities(synergy["pair_rows"], comm1_mask)
    manifest = _build_pair_synergy_manifest(config, brain, state_space_npz, synergy["backend"])

    np.savez_compressed(
        cache_paths["samples_npz"],
        initial_state_shape=np.asarray(initial_states.shape, dtype=int),
        final_state_shape=np.asarray(final_states.shape, dtype=int),
        pairs=np.asarray([(row["pair_i"], row["pair_j"]) for row in pair_rows], dtype=int),
        target_noise_sigma=np.asarray(synergy["target_noise_sigma"], dtype=float),
        left_ei=np.asarray([float(row["left_ei"]) for row in pair_rows], dtype=float),
        right_ei=np.asarray([float(row["right_ei"]) for row in pair_rows], dtype=float),
        joint_ei=np.asarray([float(row["joint_ei"]) for row in pair_rows], dtype=float),
        synergy=np.asarray([float(row["synergy"]) for row in pair_rows], dtype=float),
        synergy_ratio=np.asarray([float(row["synergy_ratio"]) for row in pair_rows], dtype=float),
    )
    _write_pair_synergy_csv(cache_paths["summary_csv"], pair_rows)
    cache_paths["manifest_json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "pair_rows": pair_rows,
        "pairs": np.asarray([(row["pair_i"], row["pair_j"]) for row in pair_rows], dtype=int),
        "target_noise_sigma": np.asarray(synergy["target_noise_sigma"], dtype=float),
        "cache_paths": cache_paths,
        "manifest": manifest,
    }


def select_ei_stratified_nodes(
    ei_rows: Sequence[dict[str, object]],
    *,
    per_stratum: int = 30,
) -> list[dict[str, object]]:
    """Select top, middle, and bottom EI-ranked nodes without duplicates."""

    if int(per_stratum) < 1:
        raise ValueError("per_stratum must be positive.")
    rows = []
    for row in ei_rows:
        if "ei_final_state" in row:
            ei_value = float(row["ei_final_state"])
        elif "ei_final_mean" in row:
            ei_value = float(row["ei_final_mean"])
        else:
            raise ValueError("EI rows must contain ei_final_state or ei_final_mean.")
        rank_value = int(row.get("rank_ei_final_state", row.get("rank_ei_final_mean", 0)) or 0)
        rows.append(
            {
                "node": int(row["node"]),
                "community": str(row.get("community", "")),
                "ei_final_state": ei_value,
                "rank_ei_final_state": rank_value,
            }
        )
    rows.sort(key=lambda item: (-item["ei_final_state"], item["node"]))
    for index, row in enumerate(rows, start=1):
        if row["rank_ei_final_state"] <= 0:
            row["rank_ei_final_state"] = index

    count = int(per_stratum)
    n_rows = len(rows)
    middle_start = max(0, (n_rows - count) // 2)
    groups = [
        ("top", rows[:count]),
        ("middle", rows[middle_start : middle_start + count]),
        ("bottom", rows[max(0, n_rows - count) :]),
    ]

    selected: list[dict[str, object]] = []
    seen: set[int] = set()
    for stratum, group in groups:
        for row in group:
            node = int(row["node"])
            if node in seen:
                continue
            item = dict(row)
            item["ei_stratum"] = stratum
            selected.append(item)
            seen.add(node)
    return selected


def evaluate_fixed_node_ignition(
    adjacency,
    model: dict,
    *,
    source_node: int,
    delta: float,
    comm1_mask: np.ndarray,
    comm2_mask: np.ndarray,
    success_threshold: float = 5.0,
    t_force: float = 12.0,
    dt: float = 0.08,
    tol_ss: float = 2e-3,
) -> dict[str, object]:
    """Evaluate one Fig.5g-style fixed single-node ignition at a given Delta."""

    node_count = int(adjacency.shape[0])
    source = int(source_node)
    if source < 0 or source >= node_count:
        raise ValueError("source_node is outside adjacency.")

    fixed_mask = np.zeros(node_count, dtype=bool)
    fixed_mask[source] = True
    x0 = np.zeros(node_count, dtype=float)
    x0[source] = float(delta)
    res = solve_odes(
        x0,
        adjacency,
        model,
        mode="BC",
        fixed_mask=fixed_mask,
        free_init=0.0,
        release=False,
        T_force=float(t_force),
        T_free=0.0,
        tol_ss=float(tol_ss),
        dt=float(dt),
    )
    x_ss = np.asarray(res["x_ss"], dtype=float)
    module1_mean = float(x_ss[np.asarray(comm1_mask, dtype=bool)].mean())
    module2_mean = float(x_ss[np.asarray(comm2_mask, dtype=bool)].mean())
    recovered_modules = int(module1_mean > float(success_threshold)) + int(
        module2_mean > float(success_threshold)
    )
    return {
        "success": bool(recovered_modules == 2),
        "recovered_modules": int(recovered_modules),
        "module1_mean": module1_mean,
        "module2_mean": module2_mean,
    }


def evaluate_fixed_pair_ignition(
    adjacency,
    model: dict,
    *,
    source_pair: tuple[int, int],
    total_cost: float,
    comm1_mask: np.ndarray,
    comm2_mask: np.ndarray,
    success_threshold: float = 5.0,
    t_force: float = 12.0,
    dt: float = 0.08,
    tol_ss: float = 2e-3,
) -> dict[str, object]:
    """Evaluate equal-split two-node fixed ignition at a total cost."""

    node_count = int(adjacency.shape[0])
    left, right = (int(source_pair[0]), int(source_pair[1]))
    if left == right:
        raise ValueError("source_pair must contain two distinct nodes.")
    if min(left, right) < 0 or max(left, right) >= node_count:
        raise ValueError("source_pair contains a node outside adjacency.")
    delta_each = float(total_cost) / 2.0

    fixed_mask = np.zeros(node_count, dtype=bool)
    fixed_mask[[left, right]] = True
    x0 = np.zeros(node_count, dtype=float)
    x0[[left, right]] = delta_each
    res = solve_odes(
        x0,
        adjacency,
        model,
        mode="BC",
        fixed_mask=fixed_mask,
        free_init=0.0,
        release=False,
        T_force=float(t_force),
        T_free=0.0,
        tol_ss=float(tol_ss),
        dt=float(dt),
    )
    x_ss = np.asarray(res["x_ss"], dtype=float)
    module1_mean = float(x_ss[np.asarray(comm1_mask, dtype=bool)].mean())
    module2_mean = float(x_ss[np.asarray(comm2_mask, dtype=bool)].mean())
    recovered_modules = int(module1_mean > float(success_threshold)) + int(
        module2_mean > float(success_threshold)
    )
    return {
        "success": bool(recovered_modules == 2),
        "recovered_modules": int(recovered_modules),
        "module1_mean": module1_mean,
        "module2_mean": module2_mean,
        "delta_i": delta_each,
        "delta_j": delta_each,
    }


def binary_search_critical_delta(
    recovery_fn,
    *,
    delta_low: float = 0.0,
    delta_high: float = 30.0,
    binary_steps: int = 10,
) -> dict[str, object]:
    """Find the smallest Delta that satisfies a monotone recovery predicate."""

    low = float(delta_low)
    high = float(delta_high)
    samples: list[dict[str, object]] = []

    high_eval = dict(recovery_fn(high))
    high_eval["delta"] = high
    samples.append(high_eval)
    if not bool(high_eval["success"]):
        return {
            "critical_delta": float("nan"),
            "threshold_status": "censored_above_delta_max",
            "recovered_modules_at_delta_max": int(high_eval.get("recovered_modules", 0)),
            "module1_mean": float(high_eval.get("module1_mean", np.nan)),
            "module2_mean": float(high_eval.get("module2_mean", np.nan)),
            "samples": samples,
        }

    best_eval = high_eval
    for _ in range(int(binary_steps)):
        mid = 0.5 * (low + high)
        mid_eval = dict(recovery_fn(mid))
        mid_eval["delta"] = mid
        samples.append(mid_eval)
        if bool(mid_eval["success"]):
            high = mid
            best_eval = mid_eval
        else:
            low = mid

    return {
        "critical_delta": float(high),
        "threshold_status": "finite",
        "recovered_modules_at_delta_max": int(high_eval.get("recovered_modules", 0)),
        "module1_mean": float(best_eval.get("module1_mean", np.nan)),
        "module2_mean": float(best_eval.get("module2_mean", np.nan)),
        "samples": samples,
    }


def run_ei_ignition_threshold_experiment(
    config: EIIgnitionThresholdConfig,
    *,
    ei_summary_csv: str | Path | None = None,
    force_recompute: bool = False,
) -> dict[str, object]:
    """Compare EI-ranked nodes against fixed-ignition recovery thresholds."""

    output_dir = Path(config.output_dir)
    cache_paths = _ignition_threshold_cache_paths(output_dir)
    if not force_recompute and _ignition_threshold_cache_matches_config(cache_paths, config):
        return _load_cached_ignition_threshold_result(cache_paths)

    output_dir.mkdir(parents=True, exist_ok=True)
    if ei_summary_csv is None:
        ei_summary_path = (
            REPO_ROOT
            / "results"
            / "network_revival_state_space_ei"
            / config.state_space_run_id
            / "state_space_node_ei_summary.csv"
        )
    else:
        ei_summary_path = Path(ei_summary_csv)
    ei_rows = _read_state_space_summary_csv(ei_summary_path)
    selected_nodes = select_ei_stratified_nodes(ei_rows, per_stratum=config.per_stratum)

    brain = load_fig5_brain_modular_adjacency(win=config.win, wout=config.wout)
    adjacency_dense = np.asarray(brain["adjacency"], dtype=float)
    adjacency = brain["adjacency_sparse"] if brain.get("adjacency_sparse") is not None else adjacency_dense
    comm1_mask = np.asarray(brain["comm1_mask"], dtype=bool)
    comm2_mask = np.asarray(brain["comm2_mask"], dtype=bool)
    model = get_model("Neural", mu=10.0, delta=1.0)

    threshold_rows: list[dict[str, object]] = []
    sample_records: list[dict[str, object]] = []
    total = len(selected_nodes)
    for index, node_row in enumerate(selected_nodes, start=1):
        node = int(node_row["node"])
        if config.show_progress:
            print(f"Ignition threshold node {index}/{total}: {node}", flush=True)

        def _recover(delta: float, node: int = node) -> dict[str, object]:
            return evaluate_fixed_node_ignition(
                adjacency,
                model,
                source_node=node,
                delta=delta,
                comm1_mask=comm1_mask,
                comm2_mask=comm2_mask,
                success_threshold=config.success_threshold,
                t_force=config.t_force,
                dt=config.dt,
                tol_ss=config.tol_ss,
            )

        search = binary_search_critical_delta(
            _recover,
            delta_low=config.delta_low,
            delta_high=config.delta_high,
            binary_steps=config.binary_steps,
        )
        for sample_index, sample in enumerate(search["samples"]):
            sample_records.append(
                {
                    "node": node,
                    "sample_index": int(sample_index),
                    "delta": float(sample["delta"]),
                    "success": bool(sample["success"]),
                    "recovered_modules": int(sample.get("recovered_modules", 0)),
                    "module1_mean": float(sample.get("module1_mean", np.nan)),
                    "module2_mean": float(sample.get("module2_mean", np.nan)),
                }
            )
        threshold_rows.append(
            {
                "node": node,
                "community": node_row["community"],
                "ei_final_state": float(node_row["ei_final_state"]),
                "ei_rank": int(node_row["rank_ei_final_state"]),
                "ei_stratum": node_row["ei_stratum"],
                "critical_delta": float(search["critical_delta"]),
                "threshold_status": str(search["threshold_status"]),
                "recovered_modules_at_delta_max": int(search["recovered_modules_at_delta_max"]),
                "module1_mean": float(search["module1_mean"]),
                "module2_mean": float(search["module2_mean"]),
            }
        )

    manifest = _build_ignition_threshold_manifest(config, brain, ei_summary_path, len(threshold_rows))
    _write_ignition_threshold_csv(cache_paths["threshold_csv"], threshold_rows)
    _write_ignition_threshold_samples(cache_paths["samples_npz"], threshold_rows, sample_records)
    cache_paths["manifest_json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "threshold_rows": threshold_rows,
        "sample_records": sample_records,
        "cache_paths": cache_paths,
        "manifest": manifest,
    }


def run_pair_ignition_cost_experiment(
    config: PairIgnitionCostConfig,
    *,
    pair_synergy_csv: str | Path | None = None,
    force_recompute: bool = False,
) -> dict[str, object]:
    """Compare state-space pair synergy against equal-split co-ignition total cost."""

    output_dir = Path(config.output_dir)
    cache_paths = _pair_ignition_cache_paths(output_dir)
    if not force_recompute and _pair_ignition_cache_matches_config(cache_paths, config):
        return _load_cached_pair_ignition_result(cache_paths)

    output_dir.mkdir(parents=True, exist_ok=True)
    if pair_synergy_csv is None:
        pair_synergy_path = (
            DEFAULT_PAIR_SYNERGY_OUTPUT_ROOT
            / config.pair_synergy_run_id
            / "state_space_pair_synergy.csv"
        )
    else:
        pair_synergy_path = Path(pair_synergy_csv)
    pair_rows = _read_pair_synergy_csv(pair_synergy_path)

    brain = load_fig5_brain_modular_adjacency(win=config.win, wout=config.wout)
    adjacency_dense = np.asarray(brain["adjacency"], dtype=float)
    adjacency = brain["adjacency_sparse"] if brain.get("adjacency_sparse") is not None else adjacency_dense
    comm1_mask = np.asarray(brain["comm1_mask"], dtype=bool)
    comm2_mask = np.asarray(brain["comm2_mask"], dtype=bool)
    model = get_model("Neural", mu=10.0, delta=1.0)

    cost_rows: list[dict[str, object]] = []
    sample_records: list[dict[str, object]] = []
    single_cache: dict[int, dict[str, object]] = {}

    def _single_cost(node: int) -> dict[str, object]:
        if node not in single_cache:
            def _recover_single(delta: float, node: int = node) -> dict[str, object]:
                return evaluate_fixed_node_ignition(
                    adjacency,
                    model,
                    source_node=node,
                    delta=delta,
                    comm1_mask=comm1_mask,
                    comm2_mask=comm2_mask,
                    success_threshold=config.success_threshold,
                    t_force=config.t_force,
                    dt=config.dt,
                    tol_ss=config.tol_ss,
                )

            single_cache[node] = binary_search_critical_delta(
                _recover_single,
                delta_low=config.single_delta_low,
                delta_high=config.single_delta_high,
                binary_steps=config.binary_steps,
            )
        return single_cache[node]

    for index, pair_row in enumerate(pair_rows, start=1):
        left = int(pair_row["pair_i"])
        right = int(pair_row["pair_j"])
        if config.show_progress:
            print(f"Pair ignition cost {index}/{len(pair_rows)}: ({left}, {right})", flush=True)

        def _recover_pair(total_cost: float, left: int = left, right: int = right) -> dict[str, object]:
            return evaluate_fixed_pair_ignition(
                adjacency,
                model,
                source_pair=(left, right),
                total_cost=total_cost,
                comm1_mask=comm1_mask,
                comm2_mask=comm2_mask,
                success_threshold=config.success_threshold,
                t_force=config.t_force,
                dt=config.dt,
                tol_ss=config.tol_ss,
            )

        pair_search = binary_search_critical_delta(
            _recover_pair,
            delta_low=config.cost_low,
            delta_high=config.cost_high,
            binary_steps=config.binary_steps,
        )
        single_left = _single_cost(left)
        single_right = _single_cost(right)
        single_i_cost = float(single_left["critical_delta"])
        single_j_cost = float(single_right["critical_delta"])
        single_i_effective = _effective_threshold_cost(single_left, config.single_delta_high)
        single_j_effective = _effective_threshold_cost(single_right, config.single_delta_high)
        single_min_cost = min(single_i_effective, single_j_effective)
        pair_effective_cost = _effective_threshold_cost(pair_search, config.cost_high)
        cost_saving = float(single_min_cost - pair_effective_cost)
        cost_saving_ratio = (
            float(cost_saving / single_min_cost)
            if np.isfinite(single_min_cost) and abs(single_min_cost) > 1e-12
            else float("nan")
        )

        for sample_index, sample in enumerate(pair_search["samples"]):
            sample_records.append(
                {
                    "pair_i": left,
                    "pair_j": right,
                    "sample_index": int(sample_index),
                    "total_cost": float(sample["delta"]),
                    "success": bool(sample["success"]),
                    "recovered_modules": int(sample.get("recovered_modules", 0)),
                    "module1_mean": float(sample.get("module1_mean", np.nan)),
                    "module2_mean": float(sample.get("module2_mean", np.nan)),
                    "delta_i": float(sample.get("delta_i", sample["delta"] / 2.0)),
                    "delta_j": float(sample.get("delta_j", sample["delta"] / 2.0)),
                }
            )

        cost_rows.append(
            {
                **pair_row,
                "critical_total_cost": float(pair_search["critical_delta"]),
                "threshold_status": str(pair_search["threshold_status"]),
                "recovered_modules_at_cost_max": int(pair_search["recovered_modules_at_delta_max"]),
                "module1_mean": float(pair_search["module1_mean"]),
                "module2_mean": float(pair_search["module2_mean"]),
                "single_i_cost": single_i_cost,
                "single_j_cost": single_j_cost,
                "single_min_cost": float(single_min_cost),
                "cost_saving": cost_saving,
                "cost_saving_ratio": cost_saving_ratio,
            }
        )

    manifest = _build_pair_ignition_manifest(config, brain, pair_synergy_path, len(cost_rows))
    _write_pair_ignition_csv(cache_paths["cost_csv"], cost_rows)
    _write_pair_ignition_samples(cache_paths["samples_npz"], cost_rows, sample_records)
    cache_paths["manifest_json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "cost_rows": cost_rows,
        "sample_records": sample_records,
        "cache_paths": cache_paths,
        "manifest": manifest,
    }


def simulate_single_ignition_trajectory(
    adjacency: np.ndarray,
    model: dict,
    *,
    source_node: int,
    delta: float,
    tau_grid: Sequence[float],
    t_ignite: float,
    dt: float,
    free_init: float = 0.0,
) -> dict[str, np.ndarray]:
    """Simulate one forced ignition and record states at post-release tau values."""

    use_sparse = sp is not None and sp.issparse(adjacency)
    adjacency_op = adjacency if use_sparse else np.asarray(adjacency, dtype=float)
    node_count = int(adjacency.shape[0])
    source = int(source_node)
    if source < 0 or source >= node_count:
        raise ValueError("source_node is outside adjacency.")

    tau = np.asarray(tuple(float(value) for value in tau_grid), dtype=float)
    if tau.ndim != 1 or tau.size == 0:
        raise ValueError("tau_grid must be a nonempty one-dimensional sequence.")
    if np.any(tau < 0.0) or np.any(np.diff(tau) < 0.0):
        raise ValueError("tau_grid must be sorted and nonnegative.")

    m0 = model["M0"]
    m1 = model["M1"]
    m2 = model["M2"]

    def matvec(x: np.ndarray) -> np.ndarray:
        if use_sparse:
            return adjacency_op.dot(m2(x))
        return adjacency_op @ m2(x)

    def rhs_free(_, x):
        return m0(x) + m1(x) * matvec(x)

    def rhs_forced(t, x):
        x_forced = x.copy()
        x_forced[source] = float(delta)
        dx = rhs_free(t, x_forced)
        dx[source] = 0.0
        return dx

    x = np.full(node_count, float(free_init), dtype=float)
    x[source] = float(delta)
    t = 0.0
    forced_values = [float(x[source])]
    while t < float(t_ignite) - 1e-12:
        dt_use = min(float(dt), float(t_ignite) - t)
        x[source] = float(delta)
        x = _rk4_step(rhs_forced, t, x, dt_use)
        x = np.maximum(x, 0.0)
        x[source] = float(delta)
        t += dt_use
        forced_values.append(float(x[source]))

    states = np.empty((tau.size, node_count), dtype=float)
    release_t = float(t_ignite)
    current_tau = 0.0
    for index, target_tau in enumerate(tau):
        while current_tau < float(target_tau) - 1e-12:
            dt_use = min(float(dt), float(target_tau) - current_tau)
            x = _rk4_step(rhs_free, release_t + current_tau, x, dt_use)
            x = np.maximum(x, 0.0)
            current_tau += dt_use
        states[index] = x

    return {
        "tau_grid": tau,
        "states_by_tau": states,
        "mean_activity_by_tau": states.mean(axis=1),
        "forced_source_values": np.asarray(forced_values, dtype=float),
    }


def simulate_node_mean_activity_samples(
    adjacency: np.ndarray,
    model: dict,
    *,
    node_indices: Sequence[int],
    deltas: np.ndarray,
    tau_grid: Sequence[float],
    t_ignite: float,
    dt: float,
    free_init: float = 0.0,
) -> np.ndarray:
    """Return mean activity samples with shape [node, delta, tau]."""

    nodes = [int(index) for index in node_indices]
    delta_values = np.asarray(deltas, dtype=float).reshape(-1)
    tau = tuple(float(value) for value in tau_grid)
    samples = np.empty((len(nodes), delta_values.size, len(tau)), dtype=float)

    for node_pos, node in enumerate(nodes):
        for delta_pos, delta in enumerate(delta_values):
            trajectory = simulate_single_ignition_trajectory(
                adjacency,
                model,
                source_node=node,
                delta=float(delta),
                tau_grid=tau,
                t_ignite=t_ignite,
                dt=dt,
                free_init=free_init,
            )
            samples[node_pos, delta_pos, :] = trajectory["mean_activity_by_tau"]
    return samples


def estimate_node_mean_activity_ei(
    deltas: np.ndarray,
    mean_activity_samples: np.ndarray,
    *,
    tau_grid: Sequence[float],
    target_noise_fraction: float = 0.0,
    seed: int = 0,
    clip_negative: bool = True,
) -> dict[str, object]:
    """Estimate I(Delta_i; mean_activity_tau) for each node and tau."""

    delta_array = np.asarray(deltas, dtype=float).reshape(-1, 1)
    samples = np.asarray(mean_activity_samples, dtype=float)
    if samples.ndim != 3:
        raise ValueError("mean_activity_samples must have shape [node, delta, tau].")
    if samples.shape[1] != delta_array.shape[0]:
        raise ValueError("deltas and mean_activity_samples disagree on sample count.")
    tau = tuple(float(value) for value in tau_grid)
    if samples.shape[2] != len(tau):
        raise ValueError("tau_grid length must match the sample tau axis.")

    node_ei = np.empty((samples.shape[0], samples.shape[2]), dtype=float)
    target_noise_sigma = np.empty_like(node_ei)
    backend = "affine_triangular_transport_map"
    bias_correction = np.empty_like(node_ei)
    noise_fraction = float(target_noise_fraction)
    rng = np.random.default_rng(seed) if noise_fraction > 0.0 else None

    for node_index in range(samples.shape[0]):
        for tau_index in range(samples.shape[2]):
            target = samples[node_index, :, tau_index].reshape(-1, 1)
            if noise_fraction == 0.0:
                sigma = 0.0
                noisy_target = target
            else:
                sigma = max(1e-6, noise_fraction * float(np.std(target, ddof=1)))
                noisy_target = target + sigma * rng.normal(size=target.shape)
            summary = estimate_mutual_information_transport_map(delta_array, noisy_target)
            value = float(summary["mi_hat"])
            if clip_negative:
                value = max(0.0, value)
            node_ei[node_index, tau_index] = value
            target_noise_sigma[node_index, tau_index] = sigma
            backend = str(summary["backend"])
            bias_correction[node_index, tau_index] = float(summary["bias_correction"])

    return {
        "node_ei_by_tau": node_ei,
        "target_noise_sigma": target_noise_sigma,
        "bias_correction": bias_correction,
        "backend": backend,
        "tau_grid": np.asarray(tau, dtype=float),
    }


def run_node_mean_activity_ei(config: DynamicEIConfig, *, force_recompute: bool = False) -> dict[str, object]:
    """Run or load the Fig.5l node ignition EI experiment."""

    output_dir = Path(config.output_dir)
    cache_paths = _cache_paths(output_dir)
    if not force_recompute and _all_cache_paths_exist(cache_paths):
        return _load_cached_result(cache_paths)

    output_dir.mkdir(parents=True, exist_ok=True)
    brain = load_fig5_brain_modular_adjacency(win=config.win, wout=config.wout)
    adjacency_dense = np.asarray(brain["adjacency"], dtype=float)
    adjacency = brain["adjacency_sparse"] if brain.get("adjacency_sparse") is not None else adjacency_dense
    model = get_model("Neural", mu=10.0, delta=1.0)
    nodes = (
        tuple(range(adjacency.shape[0]))
        if config.node_indices is None
        else tuple(int(index) for index in config.node_indices)
    )

    rng = np.random.default_rng(config.seed)
    deltas = rng.uniform(0.0, float(config.delta_max), size=int(config.n_delta))
    deltas.sort()

    mean_activity = np.empty((len(nodes), int(config.n_delta), len(config.tau_grid)), dtype=float)
    for start in range(0, len(nodes), int(config.chunk_size)):
        stop = min(start + int(config.chunk_size), len(nodes))
        if config.show_progress:
            print(
                f"Dynamic EI simulation nodes {start + 1}-{stop}/{len(nodes)} "
                f"(n_delta={config.n_delta}, tau_max={max(config.tau_grid):.1f})",
                flush=True,
            )
        mean_activity[start:stop] = simulate_node_mean_activity_samples(
            adjacency,
            model,
            node_indices=nodes[start:stop],
            deltas=deltas,
            tau_grid=config.tau_grid,
            t_ignite=config.t_ignite,
            dt=config.dt,
            free_init=config.free_init,
        )

    ei_summary = estimate_node_mean_activity_ei(
        deltas,
        mean_activity,
        tau_grid=config.tau_grid,
        target_noise_fraction=config.target_noise_fraction,
        seed=config.seed + 1009,
    )
    node_ei = np.asarray(ei_summary["node_ei_by_tau"], dtype=float)
    tau_array = np.asarray(config.tau_grid, dtype=float)
    node_summary = _build_node_summary(nodes, node_ei, tau_array, np.asarray(brain["comm1_mask"], dtype=bool))
    manifest = _build_manifest(config, brain, nodes, ei_summary["backend"])

    np.savez_compressed(
        cache_paths["samples_npz"],
        deltas=deltas,
        tau_grid=tau_array,
        node_indices=np.asarray(nodes, dtype=int),
        mean_activity_samples=mean_activity,
        node_ei_by_tau=node_ei,
        target_noise_sigma=np.asarray(ei_summary["target_noise_sigma"], dtype=float),
        bias_correction=np.asarray(ei_summary["bias_correction"], dtype=float),
    )
    _write_long_csv(cache_paths["long_csv"], nodes, deltas, tau_array, mean_activity, node_ei)
    _write_summary_csv(cache_paths["summary_csv"], node_summary)
    cache_paths["manifest_json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "deltas": deltas,
        "tau_grid": tau_array,
        "mean_activity_samples": mean_activity,
        "node_ei_by_tau": node_ei,
        "node_summary": node_summary,
        "target_noise_sigma": np.asarray(ei_summary["target_noise_sigma"], dtype=float),
        "cache_paths": cache_paths,
        "manifest": manifest,
    }


def _cache_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "samples_npz": output_dir / "node_dynamic_ei_samples.npz",
        "long_csv": output_dir / "node_dynamic_ei_long.csv",
        "summary_csv": output_dir / "node_dynamic_ei_summary.csv",
        "manifest_json": output_dir / "manifest.json",
    }


def _state_space_cache_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "samples_npz": output_dir / "state_space_ei_samples.npz",
        "summary_csv": output_dir / "state_space_node_ei_summary.csv",
        "manifest_json": output_dir / "manifest.json",
    }


def _ignition_threshold_cache_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "threshold_csv": output_dir / "node_ignition_thresholds.csv",
        "samples_npz": output_dir / "node_ignition_threshold_samples.npz",
        "manifest_json": output_dir / "manifest.json",
    }


def _pair_synergy_cache_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary_csv": output_dir / "state_space_pair_synergy.csv",
        "samples_npz": output_dir / "state_space_pair_synergy_samples.npz",
        "manifest_json": output_dir / "manifest.json",
    }


def _pair_ignition_cache_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "cost_csv": output_dir / "pair_ignition_costs.csv",
        "samples_npz": output_dir / "pair_ignition_cost_samples.npz",
        "manifest_json": output_dir / "manifest.json",
    }


def _all_cache_paths_exist(cache_paths: dict[str, Path]) -> bool:
    return all(path.exists() for path in cache_paths.values())


def _state_space_cache_matches_config(cache_paths: dict[str, Path], config: StateSpaceEIConfig) -> bool:
    if not _all_cache_paths_exist(cache_paths):
        return False
    try:
        manifest = json.loads(cache_paths["manifest_json"].read_text(encoding="utf-8"))
        arrays = np.load(cache_paths["samples_npz"], allow_pickle=False)
        keys = set(arrays.files)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False

    required_array_keys = {
        "initial_states",
        "final_states",
        "final_mean_activity",
        "node_ei",
        "target_noise_sigma",
        "bias_correction",
    }
    if not required_array_keys.issubset(keys):
        return False

    expected = {
        "experiment": "network_revival_state_space_node_ei",
        "source_variable": "initial_node_state_x_i_0",
        "target_variable": "whole_system_state_at_tau",
        "sampling_mode": "independent_uniform_state_space",
        "sample_count": int(config.sample_count),
        "state_low": float(config.state_low),
        "state_high": float(config.state_high),
        "tau": float(config.tau),
        "dt": float(config.dt),
        "seed": int(config.seed),
        "batch_size": int(config.batch_size),
        "target_noise_fraction": float(config.target_noise_fraction),
        "show_progress": bool(config.show_progress),
        "win": float(config.win),
        "wout": float(config.wout),
    }
    for key, expected_value in expected.items():
        if key not in manifest:
            return False
        actual_value = manifest[key]
        if isinstance(expected_value, float):
            if not np.isclose(float(actual_value), expected_value, rtol=0.0, atol=1e-12):
                return False
        else:
            if actual_value != expected_value:
                return False
    return True


def _ignition_threshold_cache_matches_config(
    cache_paths: dict[str, Path],
    config: EIIgnitionThresholdConfig,
) -> bool:
    if not _all_cache_paths_exist(cache_paths):
        return False
    try:
        manifest = json.loads(cache_paths["manifest_json"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "experiment": "network_revival_ei_ignition_threshold",
        "state_space_run_id": str(config.state_space_run_id),
        "per_stratum": int(config.per_stratum),
        "delta_low": float(config.delta_low),
        "delta_high": float(config.delta_high),
        "binary_steps": int(config.binary_steps),
        "success_threshold": float(config.success_threshold),
        "t_force": float(config.t_force),
        "dt": float(config.dt),
        "tol_ss": float(config.tol_ss),
        "win": float(config.win),
        "wout": float(config.wout),
    }
    for key, expected_value in expected.items():
        if key not in manifest:
            return False
        actual_value = manifest[key]
        if isinstance(expected_value, float):
            if not np.isclose(float(actual_value), expected_value, rtol=0.0, atol=1e-12):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _pair_synergy_cache_matches_config(
    cache_paths: dict[str, Path],
    config: StateSpacePairSynergyConfig,
) -> bool:
    if not _all_cache_paths_exist(cache_paths):
        return False
    try:
        manifest = json.loads(cache_paths["manifest_json"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "experiment": "network_revival_state_space_pair_synergy",
        "state_space_run_id": str(config.state_space_run_id),
        "pair_count": int(config.pair_count),
        "pair_seed": int(config.pair_seed),
        "target_noise_fraction": float(config.target_noise_fraction),
        "seed": int(config.seed),
        "win": float(config.win),
        "wout": float(config.wout),
    }
    for key, expected_value in expected.items():
        if key not in manifest:
            return False
        actual_value = manifest[key]
        if isinstance(expected_value, float):
            if not np.isclose(float(actual_value), expected_value, rtol=0.0, atol=1e-12):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _pair_ignition_cache_matches_config(
    cache_paths: dict[str, Path],
    config: PairIgnitionCostConfig,
) -> bool:
    if not _all_cache_paths_exist(cache_paths):
        return False
    try:
        manifest = json.loads(cache_paths["manifest_json"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "experiment": "network_revival_pair_ignition_cost",
        "pair_synergy_run_id": str(config.pair_synergy_run_id),
        "cost_low": float(config.cost_low),
        "cost_high": float(config.cost_high),
        "single_delta_low": float(config.single_delta_low),
        "single_delta_high": float(config.single_delta_high),
        "binary_steps": int(config.binary_steps),
        "success_threshold": float(config.success_threshold),
        "t_force": float(config.t_force),
        "dt": float(config.dt),
        "tol_ss": float(config.tol_ss),
        "win": float(config.win),
        "wout": float(config.wout),
    }
    for key, expected_value in expected.items():
        if key not in manifest:
            return False
        actual_value = manifest[key]
        if isinstance(expected_value, float):
            if not np.isclose(float(actual_value), expected_value, rtol=0.0, atol=1e-12):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _load_cached_result(cache_paths: dict[str, Path]) -> dict[str, object]:
    arrays = np.load(cache_paths["samples_npz"], allow_pickle=False)
    manifest = json.loads(cache_paths["manifest_json"].read_text(encoding="utf-8"))
    node_summary = _read_summary_csv(cache_paths["summary_csv"])
    return {
        "deltas": arrays["deltas"],
        "tau_grid": arrays["tau_grid"],
        "mean_activity_samples": arrays["mean_activity_samples"],
        "node_ei_by_tau": arrays["node_ei_by_tau"],
        "node_summary": node_summary,
        "target_noise_sigma": arrays["target_noise_sigma"],
        "cache_paths": cache_paths,
        "manifest": manifest,
    }


def _load_cached_state_space_result(cache_paths: dict[str, Path]) -> dict[str, object]:
    arrays = np.load(cache_paths["samples_npz"], allow_pickle=False)
    manifest = json.loads(cache_paths["manifest_json"].read_text(encoding="utf-8"))
    node_summary = _read_state_space_summary_csv(cache_paths["summary_csv"])
    return {
        "initial_states": arrays["initial_states"],
        "final_states": arrays["final_states"],
        "final_mean_activity": arrays["final_mean_activity"],
        "node_ei": arrays["node_ei"],
        "node_summary": node_summary,
        "target_noise_sigma": arrays["target_noise_sigma"],
        "cache_paths": cache_paths,
        "manifest": manifest,
    }


def _load_cached_ignition_threshold_result(cache_paths: dict[str, Path]) -> dict[str, object]:
    manifest = json.loads(cache_paths["manifest_json"].read_text(encoding="utf-8"))
    threshold_rows = _read_ignition_threshold_csv(cache_paths["threshold_csv"])
    arrays = np.load(cache_paths["samples_npz"], allow_pickle=False)
    sample_records = []
    for index in range(arrays["node"].shape[0]):
        sample_records.append(
            {
                "node": int(arrays["node"][index]),
                "sample_index": int(arrays["sample_index"][index]),
                "delta": float(arrays["delta"][index]),
                "success": bool(arrays["success"][index]),
                "recovered_modules": int(arrays["recovered_modules"][index]),
                "module1_mean": float(arrays["module1_mean"][index]),
                "module2_mean": float(arrays["module2_mean"][index]),
            }
        )
    return {
        "threshold_rows": threshold_rows,
        "sample_records": sample_records,
        "cache_paths": cache_paths,
        "manifest": manifest,
    }


def _load_cached_pair_synergy_result(cache_paths: dict[str, Path]) -> dict[str, object]:
    arrays = np.load(cache_paths["samples_npz"], allow_pickle=False)
    manifest = json.loads(cache_paths["manifest_json"].read_text(encoding="utf-8"))
    pair_rows = _read_pair_synergy_csv(cache_paths["summary_csv"])
    return {
        "pair_rows": pair_rows,
        "pairs": arrays["pairs"],
        "target_noise_sigma": arrays["target_noise_sigma"],
        "cache_paths": cache_paths,
        "manifest": manifest,
    }


def _load_cached_pair_ignition_result(cache_paths: dict[str, Path]) -> dict[str, object]:
    manifest = json.loads(cache_paths["manifest_json"].read_text(encoding="utf-8"))
    cost_rows = _read_pair_ignition_csv(cache_paths["cost_csv"])
    arrays = np.load(cache_paths["samples_npz"], allow_pickle=False)
    sample_records = []
    for index in range(arrays["pair_i"].shape[0]):
        sample_records.append(
            {
                "pair_i": int(arrays["pair_i"][index]),
                "pair_j": int(arrays["pair_j"][index]),
                "sample_index": int(arrays["sample_index"][index]),
                "total_cost": float(arrays["total_cost"][index]),
                "success": bool(arrays["success"][index]),
                "recovered_modules": int(arrays["recovered_modules"][index]),
                "module1_mean": float(arrays["module1_mean"][index]),
                "module2_mean": float(arrays["module2_mean"][index]),
                "delta_i": float(arrays["delta_i"][index]),
                "delta_j": float(arrays["delta_j"][index]),
            }
        )
    return {
        "cost_rows": cost_rows,
        "sample_records": sample_records,
        "cache_paths": cache_paths,
        "manifest": manifest,
    }


def _build_node_summary(
    nodes: Sequence[int],
    node_ei: np.ndarray,
    tau_grid: np.ndarray,
    comm1_mask: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_index, node in enumerate(nodes):
        ei_curve = node_ei[row_index]
        rows.append(
            {
                "node": int(node),
                "community": "M1" if bool(comm1_mask[int(node)]) else "M2",
                "max_tau_ei": float(np.max(ei_curve)),
                "argmax_tau": float(tau_grid[int(np.argmax(ei_curve))]),
                "integral_tau_ei": float(np.trapz(ei_curve, tau_grid)),
                "final_tau_ei": float(ei_curve[-1]),
            }
        )
    rows.sort(key=lambda row: row["max_tau_ei"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank_max_tau_ei"] = rank
    return rows


def _build_state_space_node_summary(
    nodes: Sequence[int],
    node_ei: np.ndarray,
    comm1_mask: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_index, node in enumerate(nodes):
        rows.append(
            {
                "node": int(node),
                "community": "M1" if bool(comm1_mask[int(node)]) else "M2",
                "ei_final_state": float(node_ei[row_index]),
            }
        )
    rows.sort(key=lambda row: row["ei_final_state"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank_ei_final_state"] = rank
    return rows


def _target_noise_policy(target_noise_fraction: float) -> str:
    return "none" if float(target_noise_fraction) == 0.0 else "gaussian_target_noise_fraction"


def _build_manifest(
    config: DynamicEIConfig,
    brain: dict[str, object],
    nodes: Sequence[int],
    backend: str,
) -> dict[str, object]:
    adjacency = np.asarray(brain["adjacency"], dtype=float)
    return {
        "experiment": "network_revival_node_dynamic_ei",
        "source_variable": "point_ignition_strength_delta_i",
        "target_variable": "whole_system_mean_activity_after_release",
        "tau_reference": "post_release",
        "noise_policy": _target_noise_policy(config.target_noise_fraction),
        "transport_backend": str(backend),
        "delta_max": float(config.delta_max),
        "n_delta": int(config.n_delta),
        "tau_grid": [float(value) for value in config.tau_grid],
        "t_ignite": float(config.t_ignite),
        "dt": float(config.dt),
        "seed": int(config.seed),
        "chunk_size": int(config.chunk_size),
        "target_noise_fraction": float(config.target_noise_fraction),
        "show_progress": bool(config.show_progress),
        "win": float(config.win),
        "wout": float(config.wout),
        "node_count": int(adjacency.shape[0]),
        "evaluated_node_count": int(len(nodes)),
        "brain_source": str(brain["source"]),
    }


def _build_state_space_manifest(
    config: StateSpaceEIConfig,
    brain: dict[str, object],
    backend: str,
) -> dict[str, object]:
    adjacency = np.asarray(brain["adjacency"], dtype=float)
    return {
        "experiment": "network_revival_state_space_node_ei",
        "source_variable": "initial_node_state_x_i_0",
        "target_variable": "whole_system_state_at_tau",
        "sampling_mode": "independent_uniform_state_space",
        "noise_policy": _target_noise_policy(config.target_noise_fraction),
        "transport_backend": str(backend),
        "sample_count": int(config.sample_count),
        "state_low": float(config.state_low),
        "state_high": float(config.state_high),
        "tau": float(config.tau),
        "dt": float(config.dt),
        "seed": int(config.seed),
        "batch_size": int(config.batch_size),
        "target_noise_fraction": float(config.target_noise_fraction),
        "show_progress": bool(config.show_progress),
        "win": float(config.win),
        "wout": float(config.wout),
        "node_count": int(adjacency.shape[0]),
        "brain_source": str(brain["source"]),
    }


def _build_ignition_threshold_manifest(
    config: EIIgnitionThresholdConfig,
    brain: dict[str, object],
    ei_summary_path: Path,
    evaluated_node_count: int,
) -> dict[str, object]:
    adjacency = np.asarray(brain["adjacency"], dtype=float)
    return {
        "experiment": "network_revival_ei_ignition_threshold",
        "state_space_run_id": str(config.state_space_run_id),
        "source_variable": "fixed_single_node_state_delta",
        "target_variable": "two_module_recovery_success",
        "recovery_protocol": "fig5g_fixed_no_release",
        "ei_summary_csv": str(ei_summary_path),
        "per_stratum": int(config.per_stratum),
        "delta_low": float(config.delta_low),
        "delta_high": float(config.delta_high),
        "binary_steps": int(config.binary_steps),
        "success_threshold": float(config.success_threshold),
        "t_force": float(config.t_force),
        "dt": float(config.dt),
        "tol_ss": float(config.tol_ss),
        "win": float(config.win),
        "wout": float(config.wout),
        "node_count": int(adjacency.shape[0]),
        "evaluated_node_count": int(evaluated_node_count),
        "brain_source": str(brain["source"]),
    }


def _build_pair_synergy_manifest(
    config: StateSpacePairSynergyConfig,
    brain: dict[str, object],
    state_space_npz: Path,
    backend: str,
) -> dict[str, object]:
    adjacency = np.asarray(brain["adjacency"], dtype=float)
    return {
        "experiment": "network_revival_state_space_pair_synergy",
        "state_space_run_id": str(config.state_space_run_id),
        "source_variable": "initial_node_pair_state_x_i_x_j_0",
        "target_variable": "whole_system_state_at_tau",
        "synergy_definition": "joint_ei_minus_singleton_ei_sum",
        "source_lift": "single:x,x2,x3;pair:x_i,x_j,x_i*x_j,x_i2,x_j2",
        "state_space_samples_npz": str(state_space_npz),
        "noise_policy": _target_noise_policy(config.target_noise_fraction),
        "transport_backend": str(backend),
        "pair_count": int(config.pair_count),
        "pair_seed": int(config.pair_seed),
        "target_noise_fraction": float(config.target_noise_fraction),
        "seed": int(config.seed),
        "win": float(config.win),
        "wout": float(config.wout),
        "node_count": int(adjacency.shape[0]),
        "brain_source": str(brain["source"]),
    }


def _build_pair_ignition_manifest(
    config: PairIgnitionCostConfig,
    brain: dict[str, object],
    pair_synergy_path: Path,
    evaluated_pair_count: int,
) -> dict[str, object]:
    adjacency = np.asarray(brain["adjacency"], dtype=float)
    return {
        "experiment": "network_revival_pair_ignition_cost",
        "pair_synergy_run_id": str(config.pair_synergy_run_id),
        "source_variable": "fixed_two_node_equal_split_total_cost",
        "target_variable": "two_module_recovery_success",
        "recovery_protocol": "fig5g_fixed_no_release_equal_split_pair",
        "pair_synergy_csv": str(pair_synergy_path),
        "cost_low": float(config.cost_low),
        "cost_high": float(config.cost_high),
        "single_delta_low": float(config.single_delta_low),
        "single_delta_high": float(config.single_delta_high),
        "binary_steps": int(config.binary_steps),
        "success_threshold": float(config.success_threshold),
        "t_force": float(config.t_force),
        "dt": float(config.dt),
        "tol_ss": float(config.tol_ss),
        "win": float(config.win),
        "wout": float(config.wout),
        "node_count": int(adjacency.shape[0]),
        "evaluated_pair_count": int(evaluated_pair_count),
        "brain_source": str(brain["source"]),
    }


def _write_long_csv(
    path: Path,
    nodes: Sequence[int],
    deltas: np.ndarray,
    tau_grid: np.ndarray,
    mean_activity: np.ndarray,
    node_ei: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["node", "delta_index", "delta", "tau", "mean_activity", "ei_tau"],
        )
        writer.writeheader()
        for node_pos, node in enumerate(nodes):
            for delta_index, delta in enumerate(deltas):
                for tau_index, tau in enumerate(tau_grid):
                    writer.writerow(
                        {
                            "node": int(node),
                            "delta_index": int(delta_index),
                            "delta": float(delta),
                            "tau": float(tau),
                            "mean_activity": float(mean_activity[node_pos, delta_index, tau_index]),
                            "ei_tau": float(node_ei[node_pos, tau_index]),
                        }
            )


def _write_state_space_summary_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _add_pair_communities(rows: Iterable[dict[str, object]], comm1_mask: np.ndarray) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        left = int(item["pair_i"])
        right = int(item["pair_j"])
        item["community_i"] = "M1" if bool(comm1_mask[left]) else "M2"
        item["community_j"] = "M1" if bool(comm1_mask[right]) else "M2"
        result.append(item)
    return result


def _write_pair_synergy_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "pair_i",
        "pair_j",
        "community_i",
        "community_j",
        "left_ei",
        "right_ei",
        "joint_ei",
        "synergy",
        "synergy_ratio",
        "rank_synergy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_ignition_threshold_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "node",
        "community",
        "ei_final_state",
        "ei_rank",
        "ei_stratum",
        "critical_delta",
        "threshold_status",
        "recovered_modules_at_delta_max",
        "module1_mean",
        "module2_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_pair_ignition_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "pair_i",
        "pair_j",
        "community_i",
        "community_j",
        "left_ei",
        "right_ei",
        "joint_ei",
        "synergy",
        "synergy_ratio",
        "rank_synergy",
        "critical_total_cost",
        "threshold_status",
        "recovered_modules_at_cost_max",
        "module1_mean",
        "module2_mean",
        "single_i_cost",
        "single_j_cost",
        "single_min_cost",
        "cost_saving",
        "cost_saving_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_ignition_threshold_samples(
    path: Path,
    threshold_rows: Sequence[dict[str, object]],
    sample_records: Sequence[dict[str, object]],
) -> None:
    np.savez_compressed(
        path,
        threshold_node=np.asarray([int(row["node"]) for row in threshold_rows], dtype=int),
        critical_delta=np.asarray([float(row["critical_delta"]) for row in threshold_rows], dtype=float),
        ei_final_state=np.asarray([float(row["ei_final_state"]) for row in threshold_rows], dtype=float),
        node=np.asarray([int(row["node"]) for row in sample_records], dtype=int),
        sample_index=np.asarray([int(row["sample_index"]) for row in sample_records], dtype=int),
        delta=np.asarray([float(row["delta"]) for row in sample_records], dtype=float),
        success=np.asarray([bool(row["success"]) for row in sample_records], dtype=bool),
        recovered_modules=np.asarray(
            [int(row["recovered_modules"]) for row in sample_records],
            dtype=int,
        ),
        module1_mean=np.asarray([float(row["module1_mean"]) for row in sample_records], dtype=float),
        module2_mean=np.asarray([float(row["module2_mean"]) for row in sample_records], dtype=float),
    )


def _write_pair_ignition_samples(
    path: Path,
    cost_rows: Sequence[dict[str, object]],
    sample_records: Sequence[dict[str, object]],
) -> None:
    np.savez_compressed(
        path,
        cost_pair_i=np.asarray([int(row["pair_i"]) for row in cost_rows], dtype=int),
        cost_pair_j=np.asarray([int(row["pair_j"]) for row in cost_rows], dtype=int),
        critical_total_cost=np.asarray([float(row["critical_total_cost"]) for row in cost_rows], dtype=float),
        synergy=np.asarray([float(row["synergy"]) for row in cost_rows], dtype=float),
        cost_saving=np.asarray([float(row["cost_saving"]) for row in cost_rows], dtype=float),
        pair_i=np.asarray([int(row["pair_i"]) for row in sample_records], dtype=int),
        pair_j=np.asarray([int(row["pair_j"]) for row in sample_records], dtype=int),
        sample_index=np.asarray([int(row["sample_index"]) for row in sample_records], dtype=int),
        total_cost=np.asarray([float(row["total_cost"]) for row in sample_records], dtype=float),
        success=np.asarray([bool(row["success"]) for row in sample_records], dtype=bool),
        recovered_modules=np.asarray([int(row["recovered_modules"]) for row in sample_records], dtype=int),
        module1_mean=np.asarray([float(row["module1_mean"]) for row in sample_records], dtype=float),
        module2_mean=np.asarray([float(row["module2_mean"]) for row in sample_records], dtype=float),
        delta_i=np.asarray([float(row["delta_i"]) for row in sample_records], dtype=float),
        delta_j=np.asarray([float(row["delta_j"]) for row in sample_records], dtype=float),
    )


def _write_summary_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_summary_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed.append(
            {
                "node": int(row["node"]),
                "community": row["community"],
                "max_tau_ei": float(row["max_tau_ei"]),
                "argmax_tau": float(row["argmax_tau"]),
                "integral_tau_ei": float(row["integral_tau_ei"]),
                "final_tau_ei": float(row["final_tau_ei"]),
                "rank_max_tau_ei": int(row["rank_max_tau_ei"]),
            }
        )
    return parsed


def _read_ignition_threshold_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed.append(
            {
                "node": int(row["node"]),
                "community": row["community"],
                "ei_final_state": float(row["ei_final_state"]),
                "ei_rank": int(row["ei_rank"]),
                "ei_stratum": row["ei_stratum"],
                "critical_delta": float(row["critical_delta"]),
                "threshold_status": row["threshold_status"],
                "recovered_modules_at_delta_max": int(row["recovered_modules_at_delta_max"]),
                "module1_mean": float(row["module1_mean"]),
                "module2_mean": float(row["module2_mean"]),
            }
        )
    return parsed


def _read_pair_synergy_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed.append(
            {
                "pair_i": int(row["pair_i"]),
                "pair_j": int(row["pair_j"]),
                "community_i": row["community_i"],
                "community_j": row["community_j"],
                "left_ei": float(row["left_ei"]),
                "right_ei": float(row["right_ei"]),
                "joint_ei": float(row["joint_ei"]),
                "synergy": float(row["synergy"]),
                "synergy_ratio": float(row["synergy_ratio"]),
                "rank_synergy": int(row["rank_synergy"]),
            }
        )
    return parsed


def _read_pair_ignition_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed.append(
            {
                "pair_i": int(row["pair_i"]),
                "pair_j": int(row["pair_j"]),
                "community_i": row["community_i"],
                "community_j": row["community_j"],
                "left_ei": float(row["left_ei"]),
                "right_ei": float(row["right_ei"]),
                "joint_ei": float(row["joint_ei"]),
                "synergy": float(row["synergy"]),
                "synergy_ratio": float(row["synergy_ratio"]),
                "rank_synergy": int(row["rank_synergy"]),
                "critical_total_cost": float(row["critical_total_cost"]),
                "threshold_status": row["threshold_status"],
                "recovered_modules_at_cost_max": int(row["recovered_modules_at_cost_max"]),
                "module1_mean": float(row["module1_mean"]),
                "module2_mean": float(row["module2_mean"]),
                "single_i_cost": float(row["single_i_cost"]),
                "single_j_cost": float(row["single_j_cost"]),
                "single_min_cost": float(row["single_min_cost"]),
                "cost_saving": float(row["cost_saving"]),
                "cost_saving_ratio": float(row["cost_saving_ratio"]),
            }
        )
    return parsed


def _read_state_space_summary_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, object]] = []
    for row in rows:
        ei_key = "ei_final_state" if "ei_final_state" in row else "ei_final_mean"
        rank_key = "rank_ei_final_state" if "rank_ei_final_state" in row else "rank_ei_final_mean"
        parsed.append(
            {
                "node": int(row["node"]),
                "community": row["community"],
                "ei_final_state": float(row[ei_key]),
                "rank_ei_final_state": int(row[rank_key]),
            }
        )
    return parsed


def _effective_threshold_cost(search: dict[str, object], censored_value: float) -> float:
    if str(search.get("threshold_status", "")) == "finite":
        return float(search["critical_delta"])
    return float(censored_value)
