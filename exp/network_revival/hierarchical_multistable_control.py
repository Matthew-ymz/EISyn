"""Pre-control hierarchical PEID for a continuous multistable latch."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from exp.TM.transport_map_density import estimate_mutual_information_transport_map


@dataclass(frozen=True)
class HierarchicalControlConfig:
    node_count: int = 6
    module_supports: tuple[tuple[int, ...], ...] = ((0, 1), (2, 3))
    module_input_weights: tuple[tuple[float, ...], ...] = (
        (0.75, 0.75, 0.15, 0.15, 0.40, 0.25),
        (0.15, 0.15, 0.75, 0.75, 0.25, 0.40),
    )
    module_gate_half: float = 1.1
    module_gate_power: float = 6.0
    hill_power: float = 4.0
    hill_half: float = 0.5
    module_theta: float = 0.5
    global_theta: float = 0.5
    module_gain: float = 0.12
    global_gain: float = 0.12
    module_rate: float = 1.0
    global_rate: float = 1.0
    amplitude_max: float = 2.0
    basin_threshold: float = 0.5
    initial_state: float = 0.02
    dt: float = 0.04
    t_force: float = 10.0
    t_free: float = 10.0

    def __post_init__(self) -> None:
        if self.node_count < 4:
            raise ValueError("node_count must be at least four.")
        if self.dt <= 0.0 or self.t_force < 0.0 or self.t_free < 0.0:
            raise ValueError("time parameters are invalid.")
        if self.amplitude_max <= 0.0:
            raise ValueError("amplitude_max must be positive.")
        weights = np.asarray(self.module_input_weights, dtype=float)
        if weights.shape != (len(self.module_supports), self.node_count):
            raise ValueError("module_input_weights has an invalid shape.")


@dataclass(frozen=True)
class ControlOutcome:
    support: tuple[int, ...]
    amplitudes: tuple[float, ...]
    duration: float
    final_modules: np.ndarray
    final_global: float
    switched: bool


@dataclass(frozen=True)
class SignedHierarchyAtom:
    support: tuple[int, ...]
    value: float
    kind: str
    depth: int


def enumerate_supports(node_count: int, max_order: int, min_order: int = 1) -> list[tuple[int, ...]]:
    if not 1 <= min_order <= max_order <= node_count:
        raise ValueError("support orders must satisfy 1 <= min_order <= max_order <= node_count.")
    return [
        tuple(combo)
        for order in range(int(min_order), int(max_order) + 1)
        for combo in combinations(range(int(node_count)), order)
    ]


def _subset_phi(
    support: tuple[int, ...],
    ei_table: Mapping[tuple[int, ...], float],
    singleton_ei: Mapping[int, float],
) -> float:
    return float(ei_table[support] - sum(float(singleton_ei[node]) for node in support))


def _bipartitions(support: tuple[int, ...]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    if len(support) <= 1:
        return []
    first, rest = support[0], support[1:]
    splits: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for mask in range(1 << len(rest)):
        left_set = {first, *(node for idx, node in enumerate(rest) if mask & (1 << idx))}
        if len(left_set) == len(support):
            continue
        left = tuple(node for node in support if node in left_set)
        right = tuple(node for node in support if node not in left_set)
        splits.append((left, right))
    return splits


def signed_greedy_atoms_from_ei(
    support: Sequence[int],
    ei_table: Mapping[tuple[int, ...], float],
    *,
    eps: float = 1.0e-8,
    depth: int = 0,
    singleton_ei: Mapping[int, float] | None = None,
) -> list[SignedHierarchyAtom]:
    """Decompose raw subset Phi while retaining signed terminal and split atoms."""

    block = tuple(int(node) for node in support)
    singles = (
        {node: float(ei_table[(node,)]) for node in block}
        if singleton_ei is None
        else singleton_ei
    )
    block_phi = _subset_phi(block, ei_table, singles)
    if len(block) <= 1:
        return []

    splits = _bipartitions(block)
    scored = []
    for left, right in splits:
        left_phi = _subset_phi(left, ei_table, singles)
        right_phi = _subset_phi(right, ei_table, singles)
        captured = left_phi + right_phi
        residual = block_phi - captured
        scored.append((captured, abs(residual), left, right, residual))
    best = max(scored, key=lambda row: (row[0], -row[1]))
    captured, _, left, right, residual = best

    if captured <= float(eps):
        return [SignedHierarchyAtom(block, float(block_phi), "terminal", int(depth))]

    atoms: list[SignedHierarchyAtom] = []
    if abs(float(residual)) > float(eps):
        atoms.append(SignedHierarchyAtom(block, float(residual), "split_residual", int(depth)))
    atoms.extend(
        signed_greedy_atoms_from_ei(
            left,
            ei_table,
            eps=eps,
            depth=depth + 1,
            singleton_ei=singles,
        )
    )
    atoms.extend(
        signed_greedy_atoms_from_ei(
            right,
            ei_table,
            eps=eps,
            depth=depth + 1,
            singleton_ei=singles,
        )
    )
    return atoms


def estimate_subset_ei_table(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_order: int,
    degree: int = 3,
    estimator: Callable[..., Mapping[str, Any]] | None = None,
) -> tuple[dict[tuple[int, ...], float], str]:
    """Estimate raw subset EI values with one shared continuous TM protocol."""

    source_array = np.asarray(source, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if source_array.ndim != 2 or target_array.ndim != 2:
        raise ValueError("source and target must be two-dimensional.")
    if source_array.shape[0] != target_array.shape[0]:
        raise ValueError("source and target must share the sample axis.")
    if not 1 <= int(max_order) <= source_array.shape[1]:
        raise ValueError("max_order is outside the source dimension.")
    tm_estimator = estimate_mutual_information_transport_map if estimator is None else estimator
    table: dict[tuple[int, ...], float] = {}
    backends: set[str] = set()
    for support in enumerate_supports(source_array.shape[1], int(max_order)):
        summary = tm_estimator(source_array[:, support], target_array, degree=int(degree))
        table[support] = float(summary["mi_hat"])
        backends.add(str(summary.get("backend", "transport_map")))
    backend = next(iter(backends)) if len(backends) == 1 else "+".join(sorted(backends))
    return table, backend


def random_control_search(
    config: HierarchicalControlConfig,
    *,
    max_order: int,
    samples_per_support: int,
    duration_bounds: tuple[float, float],
    seed: int,
    fixed_node_cost: float = 1.0,
    energy_weight: float = 0.05,
) -> list[dict[str, Any]]:
    """Search controls independently of the precomputed Phi ranking."""

    if samples_per_support < 1:
        raise ValueError("samples_per_support must be positive.")
    duration_low, duration_high = (float(value) for value in duration_bounds)
    if not 0.0 < duration_low <= duration_high:
        raise ValueError("duration bounds are invalid.")
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for support in enumerate_supports(config.node_count, int(max_order)):
        order = len(support)
        amplitude_draws = rng.uniform(
            0.1 * float(config.amplitude_max),
            float(config.amplitude_max),
            size=(int(samples_per_support), order),
        )
        duration_draws = rng.uniform(duration_low, duration_high, size=int(samples_per_support))
        amplitude_draws[0, :] = float(config.amplitude_max)
        duration_draws[0] = duration_high

        best_cost = float("inf")
        best_amplitudes: tuple[float, ...] | None = None
        best_duration: float | None = None
        success_count = 0
        for amplitudes, duration in zip(amplitude_draws, duration_draws, strict=True):
            outcome = simulate_control(support, amplitudes, float(duration), config)
            if not outcome.switched:
                continue
            success_count += 1
            normalized_energy = float(duration) * float(
                np.sum((np.asarray(amplitudes, dtype=float) / float(config.amplitude_max)) ** 2)
            )
            cost = float(fixed_node_cost) * order + float(energy_weight) * normalized_energy
            if cost < best_cost:
                best_cost = cost
                best_amplitudes = tuple(float(value) for value in amplitudes)
                best_duration = float(duration)
        rows.append(
            {
                "support": support,
                "order": order,
                "switched": bool(success_count > 0),
                "success_count": int(success_count),
                "success_rate": float(success_count / int(samples_per_support)),
                "min_cost": float(best_cost),
                "best_amplitudes": best_amplitudes,
                "best_duration": best_duration,
            }
        )
    return rows


def refine_uniform_amplitude_cost(
    support: Sequence[int],
    *,
    config: HierarchicalControlConfig,
    duration: float,
    binary_steps: int = 12,
) -> dict[str, Any]:
    """Refine the minimum common amplitude and report per-node and total energy."""

    support_tuple = tuple(int(node) for node in support)
    high = float(config.amplitude_max)
    anchor = simulate_control(support_tuple, (high,) * len(support_tuple), float(duration), config)
    if not anchor.switched:
        return {
            "switched": False,
            "threshold_amplitude": float("inf"),
            "per_node_cost": float("inf"),
            "total_energy": float("inf"),
        }
    low = 0.0
    for _ in range(int(binary_steps)):
        middle = 0.5 * (low + high)
        outcome = simulate_control(
            support_tuple,
            (middle,) * len(support_tuple),
            float(duration),
            config,
        )
        if outcome.switched:
            high = middle
        else:
            low = middle
    per_node_cost = float(duration) * (high / float(config.amplitude_max)) ** 2
    return {
        "switched": True,
        "threshold_amplitude": high,
        "per_node_cost": per_node_cost,
        "total_energy": float(len(support_tuple)) * per_node_cost,
    }


def spearman_negative_summary(
    score: Sequence[float],
    cost: Sequence[float],
    *,
    permutations: int = 9999,
    bootstrap_reps: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """One-sided negative-association test plus paired bootstrap interval."""

    x = np.asarray(score, dtype=float)
    y = np.asarray(cost, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 4:
        raise ValueError("score and cost must be equal-length vectors with at least four values.")
    observed = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(int(seed))
    permuted = np.empty(int(permutations), dtype=float)
    for index in range(int(permutations)):
        permuted[index] = float(spearmanr(x, rng.permutation(y)).statistic)
    p_one_sided = float((1 + np.sum(permuted <= observed)) / (int(permutations) + 1))
    boot: list[float] = []
    for _ in range(int(bootstrap_reps)):
        indices = rng.integers(0, x.size, size=x.size)
        value = float(spearmanr(x[indices], y[indices]).statistic)
        if np.isfinite(value):
            boot.append(value)
    ci_low, ci_high = np.quantile(np.asarray(boot, dtype=float), [0.025, 0.975])
    return {
        "rho": observed,
        "p_one_sided": p_one_sided,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n": int(x.size),
    }


def sample_precontrol_transitions(
    config: HierarchicalControlConfig,
    *,
    sample_count: int,
    horizon: float,
    seed: int,
    target_noise: float = 0.015,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one-step mechanism samples without running released-basin trials."""

    if sample_count < 16 or horizon <= 0.0 or target_noise < 0.0:
        raise ValueError("invalid pre-control sampling configuration.")
    rng = np.random.default_rng(int(seed))
    source = rng.uniform(0.0, float(config.amplitude_max), size=(int(sample_count), config.node_count))
    targets = np.empty((int(sample_count), len(config.module_supports) + 1), dtype=float)
    for index in range(int(sample_count)):
        initial = rng.uniform(0.005, 0.04, size=len(config.module_supports) + 1)
        targets[index] = _integrate(initial, source[index], float(horizon), config)
    if target_noise > 0.0:
        scale = np.maximum(targets.std(axis=0, ddof=1), 1.0e-4)
        targets += rng.normal(0.0, float(target_noise) * scale, size=targets.shape)
    source = (source - source.mean(axis=0)) / np.maximum(source.std(axis=0, ddof=1), 1.0e-8)
    targets = (targets - targets.mean(axis=0)) / np.maximum(targets.std(axis=0, ddof=1), 1.0e-8)
    return source, targets


def _support_label(support: Sequence[int]) -> str:
    return "{" + ",".join(str(int(node) + 1) for node in support) + "}"


def _root_atom_value(
    support: tuple[int, ...],
    ei_table: Mapping[tuple[int, ...], float],
) -> float:
    if len(support) == 1:
        return 0.0
    atoms = signed_greedy_atoms_from_ei(support, ei_table)
    for atom in atoms:
        if atom.support == support:
            return float(atom.value)
    return 0.0


def normalize_atom_by_support_size(atom_value: float, support: Sequence[int]) -> float:
    """Return the hierarchical atom on a per-intervened-node basis."""

    size = len(tuple(support))
    if size == 0:
        raise ValueError("support must be nonempty.")
    return float(atom_value) / size


def run_hierarchical_control_example(
    config: HierarchicalControlConfig,
    *,
    sample_count: int,
    control_samples: int,
    tm_degree: int,
    max_order: int,
    seed: int,
    precontrol_horizon: float = 4.0,
) -> dict[str, Any]:
    source, target = sample_precontrol_transitions(
        config,
        sample_count=int(sample_count),
        horizon=float(precontrol_horizon),
        seed=int(seed),
    )
    macro_target = target[:, [-1]]
    ei_table, backend = estimate_subset_ei_table(
        source,
        macro_target,
        max_order=int(max_order),
        degree=int(tm_degree),
    )
    controls = random_control_search(
        config,
        max_order=int(max_order),
        samples_per_support=int(control_samples),
        duration_bounds=(6.0, float(config.t_force)),
        seed=int(seed) + 1,
    )
    rows: list[dict[str, Any]] = []
    for control in controls:
        support = tuple(control["support"])
        refined = refine_uniform_amplitude_cost(
            support,
            config=config,
            duration=float(config.t_force),
            binary_steps=12,
        )
        singleton_sum = sum(float(ei_table[(node,)]) for node in support)
        phi_raw = float(ei_table[support] - singleton_sum)
        rows.append(
            {
                **control,
                "switched": bool(refined["switched"]),
                "min_cost": float(refined["per_node_cost"]),
                "threshold_amplitude": float(refined["threshold_amplitude"]),
                "total_energy": float(refined["total_energy"]),
                "support_label": _support_label(support),
                "joint_ei": float(ei_table[support]),
                "phi_raw": phi_raw,
                "phi_per_node": normalize_atom_by_support_size(phi_raw, support),
            }
        )
    ranked = sorted(rows, key=lambda row: (-float(row["phi_per_node"]), int(row["order"]), row["support"]))
    for rank, row in enumerate(ranked, start=1):
        row["phi_rank"] = int(rank)
    finite = [row for row in rows if np.isfinite(float(row["min_cost"]))]
    phi_per_node_association = (
        spearman_negative_summary(
            [float(row["phi_per_node"]) for row in finite],
            [float(row["total_energy"]) for row in finite],
            permutations=4999,
            bootstrap_reps=5000,
            seed=int(seed) + 2,
        )
        if len(finite) >= 4
        else {"rho": float("nan"), "p_one_sided": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": len(finite)}
    )
    global_best = min((float(row["total_energy"]) for row in finite), default=float("inf"))
    optimal_supports = [row["support"] for row in finite if np.isclose(float(row["total_energy"]), global_best)]
    top_support = ranked[0]["support"] if ranked else ()
    best_so_far = float("inf")
    regret_curve: list[dict[str, float]] = []
    for rank, row in enumerate(ranked, start=1):
        best_so_far = min(best_so_far, float(row["total_energy"]))
        regret = (
            float((best_so_far - global_best) / global_best)
            if np.isfinite(best_so_far) and np.isfinite(global_best) and global_best > 0.0
            else float("inf")
        )
        regret_curve.append({"k": int(rank), "regret": regret})
    return {
        "summary": {
            "estimator": "transport_map",
            "backend": backend,
            "tm_degree": int(tm_degree),
            "sample_count": int(sample_count),
            "control_samples_per_support": int(control_samples),
            "candidate_count": len(rows),
            "successful_support_count": len(finite),
            "global_empirical_min_total_energy": global_best,
            "optimal_supports": optimal_supports,
            "top_phi_per_node_support": top_support,
            "top_phi_per_node_hits_empirical_optimum": bool(top_support in optimal_supports),
            "target": "one-step global latch state z(t+tau)",
            "primary_cost": "minimum fixed-duration total normalized energy",
            "spearman_phi_per_node_vs_total_energy": phi_per_node_association,
        },
        "rows": ranked,
        "regret_curve": regret_curve,
        "source": source,
        "target": macro_target,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_example_results(result: Mapping[str, Any], output_dir: Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(_json_safe(result["summary"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "support_rows.json").write_text(
        json.dumps(_json_safe(result["rows"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        output / "precontrol_samples.npz",
        source=np.asarray(result["source"], dtype=float),
        target=np.asarray(result["target"], dtype=float),
    )


def plot_hierarchical_control_result(result: Mapping[str, Any], figure_base: Path) -> list[Path]:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )
    rows = list(result["rows"])
    top = rows[: min(8, len(rows))]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.4, 3.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.9, 1.35, 1.0]},
    )

    y = np.arange(len(top))
    values = np.asarray([float(row["phi_per_node"]) for row in top], dtype=float)
    axes[0].barh(y, values, color="#4477AA", alpha=0.9)
    axes[0].set_yticks(y, [row["support_label"] for row in top])
    axes[0].invert_yaxis()
    axes[0].axvline(0.0, color="#333333", linewidth=0.7)
    axes[0].set_xlabel(r"Dimension-normalized $\widetilde{\Phi}(C)/|C|$ (bits)")
    axes[0].set_ylabel("Candidate support")

    finite = [
        row
        for row in rows
        if row.get("min_cost") is not None and np.isfinite(float(row["min_cost"]))
    ]
    normalized_values = np.asarray([float(row["phi_per_node"]) for row in finite], dtype=float)
    total_energy_values = np.asarray([float(row["total_energy"]) for row in finite], dtype=float)
    axes[1].scatter(
        normalized_values,
        total_energy_values,
        color="#4477AA",
        s=34,
        edgecolor="white",
        linewidth=0.5,
    )
    if normalized_values.size >= 2:
        slope, intercept = np.polyfit(normalized_values, total_energy_values, deg=1)
        line_x = np.linspace(float(normalized_values.min()), float(normalized_values.max()), 100)
        axes[1].plot(line_x, intercept + slope * line_x, color="#222222", linewidth=1.0)
    phi_per_node_association = result["summary"].get("spearman_phi_per_node_vs_total_energy", {})
    if phi_per_node_association:
        axes[1].text(
            0.98,
            0.96,
            rf"$\rho_s={float(phi_per_node_association['rho']):.2f}$"
            + "\n"
            + rf"$p_{{-}}={float(phi_per_node_association['p_one_sided']):.3f}$, $n={int(phi_per_node_association['n'])}$",
            transform=axes[1].transAxes,
            ha="right",
            va="top",
        )
    axes[1].set_xlabel(r"Dimension-normalized $\widetilde{\Phi}(C)/|C|$ (bits)")
    axes[1].set_ylabel("Minimum total normalized energy")

    trace_config = HierarchicalControlConfig()
    trace_specs = (
        ((0,), "Single node", "#999999"),
        ((0, 1), "One local module", "#EE7733"),
        (tuple(result["summary"]["top_phi_per_node_support"]), "Top normalized Phi", "#228833"),
    )
    for support, label, color in trace_specs:
        time, states = simulate_control_trace(
            support,
            amplitudes=(trace_config.amplitude_max,) * len(support),
            duration=trace_config.t_force,
            config=trace_config,
        )
        axes[2].plot(time, states[:, -1], color=color, linewidth=1.2, label=label)
    axes[2].axvline(trace_config.t_force, color="#333333", linewidth=0.7, linestyle="--")
    axes[2].axhline(trace_config.basin_threshold, color="#777777", linewidth=0.7, linestyle=":")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Global latch state $z$")
    axes[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    for label, axis in zip("abc", axes, strict=True):
        axis.text(-0.16, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=9)

    base = Path(figure_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix, kwargs in (("png", {"dpi": 400}), ("svg", {}), ("pdf", {})):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def _hill(values: np.ndarray, config: HierarchicalControlConfig) -> np.ndarray:
    positive = np.maximum(np.asarray(values, dtype=float), 0.0)
    powered = positive ** float(config.hill_power)
    return powered / (float(config.hill_half) ** float(config.hill_power) + powered)


def module_gate_values(source: np.ndarray, config: HierarchicalControlConfig) -> np.ndarray:
    source_hill = _hill(np.asarray(source, dtype=float), config)
    activation = np.asarray(config.module_input_weights, dtype=float) @ source_hill
    powered = activation ** float(config.module_gate_power)
    half = float(config.module_gate_half) ** float(config.module_gate_power)
    return powered / (half + powered)


def _derivative(state: np.ndarray, source: np.ndarray, config: HierarchicalControlConfig) -> np.ndarray:
    modules = np.asarray(state[:-1], dtype=float)
    global_state = float(state[-1])
    gates = module_gate_values(source, config)
    module_drive = float(config.module_gain) * gates
    module_drift = modules * (1.0 - modules) * (modules - float(config.module_theta))
    module_dot = float(config.module_rate) * (module_drift + module_drive)

    global_gate = float(np.prod(_hill(modules, config)))
    global_drift = global_state * (1.0 - global_state) * (global_state - float(config.global_theta))
    global_dot = float(config.global_rate) * (global_drift + float(config.global_gain) * global_gate)
    return np.concatenate([module_dot, np.asarray([global_dot])])


def _integrate(
    state: np.ndarray,
    source: np.ndarray,
    duration: float,
    config: HierarchicalControlConfig,
) -> np.ndarray:
    values = np.asarray(state, dtype=float).copy()
    steps = int(np.ceil(float(duration) / float(config.dt)))
    elapsed = 0.0
    for _ in range(steps):
        dt = min(float(config.dt), float(duration) - elapsed)
        k1 = _derivative(values, source, config)
        k2 = _derivative(values + 0.5 * dt * k1, source, config)
        k3 = _derivative(values + 0.5 * dt * k2, source, config)
        k4 = _derivative(values + dt * k3, source, config)
        values += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        values = np.clip(values, -0.25, 1.5)
        elapsed += dt
    return values


def simulate_control(
    support: Sequence[int],
    amplitudes: Sequence[float],
    duration: float,
    config: HierarchicalControlConfig | None = None,
) -> ControlOutcome:
    cfg = HierarchicalControlConfig() if config is None else config
    support_tuple = tuple(int(node) for node in support)
    amplitude_tuple = tuple(float(value) for value in amplitudes)
    if len(support_tuple) != len(amplitude_tuple):
        raise ValueError("support and amplitudes must have equal length.")
    if len(set(support_tuple)) != len(support_tuple):
        raise ValueError("support nodes must be unique.")
    if any(node < 0 or node >= cfg.node_count for node in support_tuple):
        raise ValueError("support contains an invalid node.")

    source = np.zeros(cfg.node_count, dtype=float)
    source[list(support_tuple)] = np.asarray(amplitude_tuple, dtype=float)
    initial = np.full(len(cfg.module_supports) + 1, float(cfg.initial_state), dtype=float)
    forced = _integrate(initial, source, float(duration), cfg)
    released = _integrate(forced, np.zeros_like(source), float(cfg.t_free), cfg)
    final_global = float(released[-1])
    return ControlOutcome(
        support=support_tuple,
        amplitudes=amplitude_tuple,
        duration=float(duration),
        final_modules=released[:-1].copy(),
        final_global=final_global,
        switched=bool(final_global > float(cfg.basin_threshold)),
    )


def simulate_control_trace(
    support: Sequence[int],
    amplitudes: Sequence[float],
    duration: float,
    config: HierarchicalControlConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = HierarchicalControlConfig() if config is None else config
    support_tuple = tuple(int(node) for node in support)
    amplitude_array = np.asarray(tuple(float(value) for value in amplitudes), dtype=float)
    if len(support_tuple) != amplitude_array.size:
        raise ValueError("support and amplitudes must have equal length.")
    source = np.zeros(cfg.node_count, dtype=float)
    source[list(support_tuple)] = amplitude_array
    state = np.full(len(cfg.module_supports) + 1, float(cfg.initial_state), dtype=float)
    time_values = [0.0]
    state_values = [state.copy()]
    elapsed = 0.0
    total = float(duration) + float(cfg.t_free)
    while elapsed < total - 1.0e-12:
        dt = min(float(cfg.dt), total - elapsed)
        active_source = source if elapsed < float(duration) - 1.0e-12 else np.zeros_like(source)
        k1 = _derivative(state, active_source, cfg)
        k2 = _derivative(state + 0.5 * dt * k1, active_source, cfg)
        k3 = _derivative(state + 0.5 * dt * k2, active_source, cfg)
        k4 = _derivative(state + dt * k3, active_source, cfg)
        state += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        state = np.clip(state, -0.25, 1.5)
        elapsed += dt
        time_values.append(elapsed)
        state_values.append(state.copy())
    return np.asarray(time_values, dtype=float), np.asarray(state_values, dtype=float)
