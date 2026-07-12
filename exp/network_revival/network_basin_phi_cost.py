"""Pre-control continuous Phi screening for natural whole-network basin control."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from exp.network_revival.effective_information import estimate_mutual_information_transport_map
from exp.network_revival.network_basin_pair_ignition import (
    NetworkBasinPairIgnitionConfig,
    _fixed_conditions,
    _model_for_name,
    _screen_candidate_instance,
    simulate_released_ignition_batch,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "network_basin_phi_cost"


@dataclass(frozen=True)
class NaturalNetworkPhiCostConfig:
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    node_count: int = 20
    model_names: tuple[str, ...] = ("Neural", "Eco")
    network_kinds: tuple[str, ...] = ("ER", "WS")
    instances_per_group: int = 5
    candidate_seed_count: int = 100
    candidate_counts_by_order: tuple[tuple[int, int], ...] = ((2, 190), (3, 300), (4, 300))
    precontrol_sample_count: int = 768
    precontrol_horizon: float = 4.0
    target_mode: str = "macro_state"
    tm_degree: int = 3
    t_force: float = 10.0
    t_free: float = 10.0
    dt: float = 0.08
    binary_steps: int = 12
    state_clip: float = 50.0
    min_switchable_per_order: int = 10
    permutations: int = 999
    bootstrap_reps: int = 1000
    seed: int = 20260712
    er_avg_degree: float = 3.5
    ws_degree: int = 4
    ws_rewire_probability: float = 0.2
    coupling_scales: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.8, 2.6, 3.8, 5.5, 8.0)
    delta_max_values: tuple[float, ...] = (0.2, 0.5, 1.0, 1.5, 2.5, 4.0, 6.0, 8.0)
    neural_parameters: dict[str, float] = field(default_factory=lambda: {"mu": 3.0, "delta": 1.0})
    eco_parameters: dict[str, float] = field(default_factory=dict)
    high_initial_by_model: dict[str, float] = field(default_factory=lambda: {"Neural": 12.0, "Eco": 3.0})

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.node_count < 4 or self.instances_per_group < 1:
            raise ValueError("node_count must be at least four and instances_per_group positive.")
        if self.precontrol_sample_count < 32 or self.precontrol_horizon <= 0.0:
            raise ValueError("precontrol sampling parameters are invalid.")
        if self.tm_degree < 1 or self.binary_steps < 1:
            raise ValueError("estimation and refinement parameters are invalid.")
        if self.target_mode not in {"macro_state", "basin_oracle"}:
            raise ValueError("target_mode must be 'macro_state' or 'basin_oracle'.")
        if not self.model_names or not self.network_kinds:
            raise ValueError("model_names and network_kinds must be nonempty.")


def enumerate_candidate_supports(config: NaturalNetworkPhiCostConfig) -> dict[int, list[tuple[int, ...]]]:
    """Enumerate pairs and draw reproducible higher-order candidate supports."""

    rng = np.random.default_rng(int(config.seed))
    supports: dict[int, list[tuple[int, ...]]] = {}
    for order, requested in config.candidate_counts_by_order:
        universe = list(combinations(range(int(config.node_count)), int(order)))
        if int(order) == 2 or int(requested) >= len(universe):
            supports[int(order)] = universe
            continue
        selected = rng.choice(len(universe), size=int(requested), replace=False)
        supports[int(order)] = [universe[int(index)] for index in np.sort(selected)]
    return supports


def normalized_phi_cost_summary(
    rows: Sequence[dict[str, Any]],
    *,
    order: int,
    permutations: int,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, float | int]:
    """Summarize the predeclared Phi--minimum-total-energy association."""

    subset = [row for row in rows if int(row["order"]) == int(order)]
    finite = [row for row in subset if bool(row["switched"]) and np.isfinite(float(row["total_energy"]))]
    base: dict[str, float | int] = {
        "candidate_count": len(subset),
        "switchable_count": len(finite),
        "rho": float("nan"),
        "p_one_sided": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
    }
    if len(finite) < 2:
        return base
    phi = np.asarray([float(row["phi_per_node"]) for row in finite], dtype=float)
    cost = np.asarray([float(row["total_energy"]) for row in finite], dtype=float)
    if np.unique(phi).size < 2 or np.unique(cost).size < 2:
        return base
    rho = float(spearmanr(phi, cost).statistic)
    rng = np.random.default_rng(int(seed) + int(order))
    null = np.asarray([spearmanr(phi, rng.permutation(cost)).statistic for _ in range(int(permutations))], dtype=float)
    boot = []
    for _ in range(int(bootstrap_reps)):
        index = rng.integers(0, len(phi), len(phi))
        sample_phi, sample_cost = phi[index], cost[index]
        if np.unique(sample_phi).size > 1 and np.unique(sample_cost).size > 1:
            value = float(spearmanr(sample_phi, sample_cost).statistic)
            if np.isfinite(value):
                boot.append(value)
    ci = np.quantile(np.asarray(boot, dtype=float), [0.025, 0.975]) if boot else np.asarray([np.nan, np.nan])
    return {
        **base,
        "rho": rho,
        "p_one_sided": float((1 + np.sum(null <= rho)) / (len(null) + 1)),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
    }


def _pair_screen_config(config: NaturalNetworkPhiCostConfig) -> NetworkBasinPairIgnitionConfig:
    return NetworkBasinPairIgnitionConfig(
        node_count=int(config.node_count),
        model_names=tuple(config.model_names), network_kinds=tuple(config.network_kinds),
        accepted_instances_per_group=1, candidate_seed_count=int(config.candidate_seed_count),
        er_avg_degree=float(config.er_avg_degree), ws_degree=int(config.ws_degree),
        ws_rewire_probability=float(config.ws_rewire_probability), coupling_scales=tuple(config.coupling_scales),
        delta_max_values=tuple(config.delta_max_values), t_force=float(config.t_force), t_free=float(config.t_free),
        dt=float(config.dt), state_clip=float(config.state_clip), seed=int(config.seed),
        neural_parameters=dict(config.neural_parameters), eco_parameters=dict(config.eco_parameters),
        high_initial_by_model=dict(config.high_initial_by_model),
    )


def _short_response_phi_rows(
    adjacency: np.ndarray,
    model: dict[str, Any],
    supports: dict[int, list[tuple[int, ...]]],
    *,
    delta_max: float,
    basin_low: float,
    basin_high: float,
    config: NaturalNetworkPhiCostConfig,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    denominator = max(float(basin_high) - float(basin_low), 1.0e-8)
    source_all = rng.uniform(0.0, float(delta_max), size=(int(config.precontrol_sample_count), int(config.node_count)))
    masks = np.zeros_like(source_all, dtype=bool)
    fixed = np.zeros_like(source_all, dtype=float)
    response = simulate_released_ignition_batch(
        adjacency, model, masks, fixed, t_force=0.0, t_free=float(config.precontrol_horizon),
        dt=float(config.dt), state_clip=float(config.state_clip), initial_states=source_all,
    )
    target_all = ((np.asarray(response["final_mean"], dtype=float) - float(basin_low)) / denominator)[:, None]
    valid_all = np.asarray(response["valid"], dtype=bool) & np.isfinite(target_all[:, 0])
    if config.target_mode == "basin_oracle":
        released = simulate_released_ignition_batch(
            adjacency, model, masks, fixed, t_force=0.0, t_free=float(config.t_free),
            dt=float(config.dt), state_clip=float(config.state_clip), initial_states=np.asarray(response["final_states"], dtype=float),
        )
        threshold = 0.5 * (float(basin_low) + float(basin_high))
        target_all = (np.asarray(released["valid"], dtype=bool) & (np.asarray(released["final_mean"], dtype=float) > threshold)).astype(float)[:, None]
        valid_all = valid_all & np.asarray(released["valid"], dtype=bool)
    for order, order_supports in supports.items():
        for support in order_supports:
            source = source_all[valid_all][:, support]
            target = target_all[valid_all]
            if len(source) < 32 or np.nanstd(target) <= 1.0e-10:
                phi = float("nan")
            else:
                joint = float(estimate_mutual_information_transport_map(source, target, degree=int(config.tm_degree))["mi_hat"])
                singles = sum(float(estimate_mutual_information_transport_map(source[:, [index]], target, degree=int(config.tm_degree))["mi_hat"]) for index in range(source.shape[1]))
                phi = joint - singles
            rows.append({"support": support, "order": int(order), "phi_per_node": float(phi / len(support))})
    return rows


def _minimum_uniform_cost_rows(
    adjacency: np.ndarray,
    model: dict[str, Any],
    score_rows: list[dict[str, Any]],
    *,
    delta_max: float,
    basin_threshold: float,
    config: NaturalNetworkPhiCostConfig,
) -> list[dict[str, Any]]:
    supports = [tuple(row["support"]) for row in score_rows]
    node_count = int(config.node_count)
    amplitudes = np.full(len(supports), float(delta_max), dtype=float)

    def run(values: np.ndarray) -> np.ndarray:
        masks = np.zeros((len(supports), node_count), dtype=bool)
        fixed = np.zeros_like(masks, dtype=float)
        for index, support in enumerate(supports):
            masks[index, list(support)] = True
            fixed[index, list(support)] = float(values[index])
        outcome = simulate_released_ignition_batch(
            adjacency, model, masks, fixed, t_force=float(config.t_force), t_free=float(config.t_free),
            dt=float(config.dt), state_clip=float(config.state_clip),
        )
        return np.asarray(outcome["valid"], dtype=bool) & (np.asarray(outcome["final_mean"], dtype=float) > float(basin_threshold))

    switched = run(amplitudes)
    low = np.zeros_like(amplitudes)
    high = amplitudes.copy()
    for _ in range(int(config.binary_steps)):
        middle = 0.5 * (low + high)
        success = run(middle)
        high = np.where(switched & success, middle, high)
        low = np.where(switched & ~success, middle, low)
    rows: list[dict[str, Any]] = []
    for index, score in enumerate(score_rows):
        order = int(score["order"])
        total = float(order * float(config.t_force) * (high[index] / float(delta_max)) ** 2) if switched[index] else float("inf")
        rows.append({**score, "switched": bool(switched[index]), "threshold_amplitude": float(high[index]) if switched[index] else float("inf"), "total_energy": total})
    return rows


def _find_instances(config: NaturalNetworkPhiCostConfig) -> list[dict[str, Any]]:
    screen = _pair_screen_config(config)
    accepted: list[dict[str, Any]] = []
    for model_name in config.model_names:
        for network_kind in config.network_kinds:
            count = 0
            for offset in range(int(config.candidate_seed_count)):
                seed = int(config.seed) + 100_003 * (len(accepted) + 1) + offset
                candidate = _screen_candidate_instance(model_name, network_kind, seed, screen)
                if candidate is None:
                    continue
                accepted.append(candidate)
                count += 1
                if count >= int(config.instances_per_group):
                    break
            if count < int(config.instances_per_group):
                raise RuntimeError(f"Only found {count} valid {model_name}/{network_kind} instances.")
    return accepted


def run_natural_network_phi_cost_experiment(config: NaturalNetworkPhiCostConfig, *, force: bool = False) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "summary.json"
    rows_path = config.output_dir / "support_rows.jsonl"
    instance_path = config.output_dir / "instance_summaries.jsonl"
    if not force and summary_path.exists() and rows_path.exists() and instance_path.exists():
        return {"summary": json.loads(summary_path.read_text()), "rows_path": rows_path, "instance_path": instance_path}
    supports = enumerate_candidate_supports(config)
    rows: list[dict[str, Any]] = []
    instance_summaries: list[dict[str, Any]] = []
    for index, found in enumerate(_find_instances(config)):
        instance = found["instance"]
        model = _model_for_name(str(instance["model_name"]), _pair_screen_config(config))
        score_rows = _short_response_phi_rows(
            np.asarray(instance["adjacency"], dtype=float), model, supports, delta_max=float(instance["delta_max"]),
            basin_low=float(instance["low_mean"]), basin_high=float(instance["high_mean"]), config=config,
            seed=int(config.seed) + index,
        )
        evaluated = _minimum_uniform_cost_rows(
            np.asarray(instance["adjacency"], dtype=float), model, score_rows, delta_max=float(instance["delta_max"]),
            basin_threshold=float(instance["basin_threshold"]), config=config,
        )
        for row in evaluated:
            row.update({key: instance[key] for key in ("model_name", "network_kind", "seed", "coupling_scale", "delta_max")})
            row["instance_index"] = index
        rows.extend(evaluated)
        for order in sorted(supports):
            instance_summaries.append({
                "model_name": instance["model_name"], "network_kind": instance["network_kind"], "instance_index": index,
                "instance_seed": instance["seed"], "order": order,
                **normalized_phi_cost_summary(evaluated, order=order, permutations=config.permutations, bootstrap_reps=config.bootstrap_reps, seed=int(config.seed) + index),
            })
    summary = {
        "experiment": "natural_network_precontrol_phi_minimum_total_cost",
        "target": "oracle released-basin coordinate after the shared tau-horizon state" if config.target_mode == "basin_oracle" else "normalized whole-network mean after the shared tau-horizon state",
        "cost": "fixed-duration minimum common-amplitude total normalized energy",
        "node_count": int(config.node_count), "candidate_counts_by_order": {str(key): len(value) for key, value in supports.items()},
        "instance_count": len({row["instance_index"] for row in rows}),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    with rows_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), allow_nan=False) + "\n")
    with instance_path.open("w") as handle:
        for row in instance_summaries:
            handle.write(json.dumps(_jsonable(row), allow_nan=False) + "\n")
    return {"summary": summary, "rows": rows, "instance_summaries": instance_summaries, "rows_path": rows_path, "instance_path": instance_path}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def plot_natural_network_phi_cost(result: dict[str, Any], figure_base: Path) -> list[Path]:
    """Plot one Phi--cost scatter panel for every dynamics/topology/order condition."""

    rows = result.get("rows")
    if rows is None:
        rows = [json.loads(line) for line in Path(result["rows_path"]).read_text().splitlines()]
    groups = sorted({(row["model_name"], row["network_kind"]) for row in rows})
    orders = sorted({int(row["order"]) for row in rows})
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 7, "axes.spines.right": False, "axes.spines.top": False, "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(len(groups), len(orders), figsize=(8.4, 7.4), constrained_layout=True, squeeze=False)
    for row_index, group in enumerate(groups):
        for column_index, order in enumerate(orders):
            axis = axes[row_index, column_index]
            subset = [row for row in rows if (row["model_name"], row["network_kind"]) == group and int(row["order"]) == order and row.get("switched") and row.get("total_energy") is not None and row.get("phi_per_node") is not None]
            x = np.asarray([float(row["phi_per_node"]) for row in subset], dtype=float)
            y = np.asarray([float(row["total_energy"]) for row in subset], dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            x, y = x[finite], y[finite]
            axis.scatter(x, y, s=12, color="#4477AA", alpha=0.62, edgecolor="white", linewidth=0.25)
            if len(x) >= 3 and np.unique(x).size > 1:
                slope, intercept = np.polyfit(x, y, 1)
                line_x = np.linspace(float(x.min()), float(x.max()), 100)
                axis.plot(line_x, intercept + slope * line_x, color="#222222", linewidth=0.8)
                rho = float(spearmanr(x, y).statistic)
                axis.text(0.97, 0.96, f"$\\rho_s={rho:.2f}$\n$n={len(x)}$", transform=axis.transAxes, ha="right", va="top")
            axis.set_title(f"{group[0]} {group[1]}, |C|={order}")
            axis.set_xlabel(r"Normalized $\widetilde{\Phi}(C)/|C|$")
            axis.set_ylabel(r"Minimum total energy $J_{\mathrm{tot}}$")
    base = Path(figure_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in (("png", {"dpi": 400}), ("svg", {}), ("pdf", {})):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs
