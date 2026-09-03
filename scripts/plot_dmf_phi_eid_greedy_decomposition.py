from __future__ import annotations

import argparse
import importlib.util
import itertools
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phi_hierarchy import PhiAtom as GreedyAtom, greedy_phi_atoms, subset_phi_raw

DMF_MODULE_PATH = ROOT / "exp" / "brain" / "dmf_fig6.py"
DEFAULT_SOURCE_RESULTS = ROOT / "exp" / "brain" / "result_lausanne_fig6" / "count_00_fig6b_mean_rate.npz"
DEFAULT_CONNECTIVITY_CSV = ROOT / "exp" / "brain" / "result_lausanne_fig6" / "count_00_connectivity.csv"
DEFAULT_FIGURE = ROOT / "fig" / "dmf_phi_eid_greedy_decomposition.png"
DEFAULT_RESULTS = ROOT / "fig" / "dmf_phi_eid_greedy_decomposition.npz"
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


def load_dmf_module():
    spec = importlib.util.spec_from_file_location("dmf_fig6_exp_brain", DMF_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def safe_logdet(matrix: np.ndarray, *, ridge: float = 1.0e-9) -> float:
    array = np.asarray(matrix, dtype=float)
    array = 0.5 * (array + array.T)
    scale = max(float(np.nanmean(np.diag(array))), 1.0e-12)
    regularized = array + float(ridge) * scale * np.eye(array.shape[0], dtype=float)
    sign, logdet = np.linalg.slogdet(regularized)
    if sign <= 0 or not np.isfinite(logdet):
        raise ValueError("Covariance matrix is not positive definite after regularization.")
    return float(logdet)


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


def load_region_labels(path: Path, n_regions: int) -> list[str]:
    if not path.exists():
        return [f"region_{index:03d}" for index in range(n_regions)]
    labels: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            labels.append(stripped.split(",")[0])
    return labels[:n_regions] if len(labels) >= n_regions else [f"region_{index:03d}" for index in range(n_regions)]


def module_indices_from_labels(labels: Sequence[str]) -> dict[str, list[int]]:
    grouped = {name: [] for name in MODULE_ORDER}
    for index, label in enumerate(labels):
        grouped[infer_display_module(str(label))].append(index)
    return {name: indices for name, indices in grouped.items() if indices}


def lagged_samples(series: np.ndarray, tau: int) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(series, dtype=float)
    if array.ndim != 2 or array.shape[0] <= tau + 2:
        raise ValueError("series must have shape [time, region] with enough lagged samples.")
    return array[:-tau], array[tau:]


def fit_standardized_transition(source_samples: np.ndarray, target_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_samples, dtype=float)
    target = np.asarray(target_samples, dtype=float)
    source_z = (source - source.mean(axis=0, keepdims=True)) / np.maximum(source.std(axis=0, ddof=1, keepdims=True), 1.0e-12)
    target_z = (target - target.mean(axis=0, keepdims=True)) / np.maximum(target.std(axis=0, ddof=1, keepdims=True), 1.0e-12)
    coefficient, *_ = np.linalg.lstsq(source_z, target_z, rcond=None)
    transition = coefficient.T
    residual = target_z - source_z @ coefficient
    noise = np.cov(residual, rowvar=False, bias=False)
    return transition, np.asarray(noise, dtype=float)


def module_ei_table(
    transition_matrix: np.ndarray,
    noise_covariance: np.ndarray,
    module_indices: Mapping[str, Sequence[int]],
    *,
    ridge: float = 1.0e-9,
) -> dict[tuple[str, ...], float]:
    transition = np.asarray(transition_matrix, dtype=float)
    noise = np.asarray(noise_covariance, dtype=float)
    names = tuple(name for name in MODULE_ORDER if name in module_indices)
    module_columns = {name: list(map(int, module_indices[name])) for name in names}
    target_cov = transition @ transition.T + noise
    target_logdet = safe_logdet(target_cov, ridge=ridge)
    table: dict[tuple[str, ...], float] = {}
    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(names, size):
            selected = {column for name in subset for column in module_columns[name]}
            complement = [column for column in range(transition.shape[1]) if column not in selected]
            conditional = noise.copy()
            if complement:
                conditional = conditional + transition[:, complement] @ transition[:, complement].T
            value = 0.5 * (target_logdet - safe_logdet(conditional, ridge=ridge)) / np.log(2.0)
            table[tuple(subset)] = float(value)
    return table


def build_payload_for_transition(
    g_values: np.ndarray,
    transitions: Sequence[np.ndarray],
    noises: Sequence[np.ndarray],
    module_indices: Mapping[str, Sequence[int]],
    *,
    ridge: float,
    eps: float,
) -> dict[str, np.ndarray]:
    module_names = np.asarray([name for name in MODULE_ORDER if name in module_indices], dtype=object)
    phi_eid = np.zeros(len(g_values), dtype=float)
    rows: list[tuple[int, str, float, int, int, str]] = []
    for g_index, (transition, noise) in enumerate(zip(transitions, noises)):
        table = module_ei_table(transition, noise, module_indices, ridge=ridge)
        root = tuple(str(name) for name in module_names)
        singleton = {name: float(table[(name,)]) for name in root}
        phi_eid[g_index] = subset_phi_raw(root, table, singleton)
        atoms = greedy_phi_atoms(root, table, eps=eps, singleton_ei=singleton)
        for atom in atoms:
            rows.append((g_index, "+".join(atom.sources), float(atom.value), int(atom.depth), len(atom.sources), atom.kind))

    atom_g_indices = np.asarray([row[0] for row in rows], dtype=int)
    atom_labels = np.asarray([row[1] for row in rows], dtype=object)
    atom_values = np.asarray([row[2] for row in rows], dtype=float)
    atom_depths = np.asarray([row[3] for row in rows], dtype=int)
    atom_orders = np.asarray([row[4] for row in rows], dtype=int)
    atom_kinds = np.asarray([row[5] for row in rows], dtype=object)
    return {
        "G": np.asarray(g_values, dtype=float),
        "module_names": module_names,
        "phi_eid": phi_eid,
        "atom_g_indices": atom_g_indices,
        "atom_labels": atom_labels,
        "atom_values": atom_values,
        "atom_depths": atom_depths,
        "atom_orders": atom_orders,
        "atom_kinds": atom_kinds,
    }


def synthetic_payload() -> dict[str, np.ndarray]:
    g_values = np.asarray([1.0, 1.5, 2.0], dtype=float)
    module_indices = {"DMN": [0], "Som": [1], "Vis": [2], "FPN": [3]}
    transitions: list[np.ndarray] = []
    noises: list[np.ndarray] = []
    for coupling_g in g_values:
        shared = 0.18 + 0.18 * coupling_g
        transition = np.asarray(
            [
                [0.55, shared, 0.03, 0.00],
                [shared, 0.50, 0.00, 0.04],
                [0.02, 0.00, 0.52, 0.20 + 0.08 * coupling_g],
                [0.00, 0.02, 0.20 + 0.08 * coupling_g, 0.50],
            ],
            dtype=float,
        )
        transitions.append(transition)
        noises.append(np.diag([0.18, 0.18, 0.20, 0.20]))
    return build_payload_for_transition(g_values, transitions, noises, module_indices, ridge=1.0e-9, eps=1.0e-7)


def actual_payload(args: argparse.Namespace) -> dict[str, np.ndarray]:
    dmf = load_dmf_module()
    archive = np.load(resolve_path(args.source_results))
    g_all = np.asarray(archive["G"], dtype=float)
    selected = np.arange(0, g_all.size, max(1, int(args.g_stride)))
    g_values = g_all[selected]
    connectivity = np.asarray(archive["connectivity"], dtype=float)
    j_fic = np.asarray(archive["j_fic"], dtype=float)[selected]
    labels = load_region_labels(resolve_path(args.connectivity_labels), connectivity.shape[0])
    module_indices = module_indices_from_labels(labels)

    parameters = dmf.DMFParameters(t_total=args.t_total, burn_in=args.burn_in, dt=args.dt, sigma=args.sigma)
    stabilization = dmf.StabilizationParameters(
        window=args.stabilization_window,
        tolerance_hz=args.stabilization_tolerance,
        confirm_windows=args.stabilization_confirm_windows,
    )
    transitions: list[np.ndarray] = []
    noises: list[np.ndarray] = []
    initial_se = None
    initial_si = None
    for index, coupling_g in enumerate(g_values):
        simulation = dmf.simulate_dmf(
            connectivity,
            float(coupling_g),
            np.asarray(j_fic[index], dtype=float),
            parameters=parameters,
            stabilization_parameters=stabilization,
            seed=args.seed + int(selected[index]),
            initial_se=initial_se if not args.independent_restarts else None,
            initial_si=initial_si if not args.independent_restarts else None,
            record_rate_trace=True,
        )
        start_step = int(float(simulation["stabilization_start_step"]))
        rates = np.asarray(simulation["region_rate_trace_hz"], dtype=float)[start_step:]
        series = dmf.transform_rates_to_bold(rates, dt=args.dt) if args.use_bold else rates
        source, target = lagged_samples(series, args.tau)
        transition, noise = fit_standardized_transition(source, target)
        transitions.append(transition)
        noises.append(noise)
        initial_se = np.asarray(simulation["final_se"], dtype=float)
        initial_si = np.asarray(simulation["final_si"], dtype=float)
        print(f"[{index + 1}/{g_values.size}] fitted transition for G={coupling_g:.3f}")

    payload = build_payload_for_transition(g_values, transitions, noises, module_indices, ridge=args.ridge, eps=args.eps)
    payload["mean_rate_hz"] = np.asarray(archive["mean_rate_hz"], dtype=float)[selected]
    return payload


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def plot_payload(payload: Mapping[str, np.ndarray], output_path: Path) -> None:
    configure_matplotlib()
    g_values = np.asarray(payload["G"], dtype=float)
    phi_eid = np.asarray(payload["phi_eid"], dtype=float)
    labels = np.asarray(payload["atom_labels"], dtype=object)
    values = np.asarray(payload["atom_values"], dtype=float)
    orders = np.asarray(payload["atom_orders"], dtype=int)
    depths = np.asarray(payload["atom_depths"], dtype=int)
    g_indices = np.asarray(payload["atom_g_indices"], dtype=int)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9.2, 3.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.35]},
    )
    ax_curve, ax_order, ax_top = axes
    ax_curve.plot(g_values, phi_eid, color="#0072B2", lw=1.8)
    ax_curve.scatter(g_values, phi_eid, color="#0072B2", s=20)
    ax_curve.set_xlabel("Global coupling $G$")
    ax_curve.set_ylabel(r"Module-level $\Phi^{EID}$")
    ax_curve.set_ylim(bottom=0.0)
    ax_curve.grid(True, color="0.88", lw=0.8)

    order_values: dict[int, float] = {}
    for order in sorted(set(orders.tolist())):
        order_values[int(order)] = float(values[orders == order].sum())
    order_total = max(sum(order_values.values()), 1.0e-12)
    order_labels = [f"order {order}" for order in order_values]
    order_heights = [order_values[order] / order_total for order in order_values]
    ax_order.bar(order_labels, order_heights, color="#56B4E9", width=0.62)
    ax_order.set_ylabel("Fraction of greedy residual")
    ax_order.set_ylim(0.0, 1.0)
    ax_order.grid(True, axis="y", color="0.88", lw=0.8)
    ax_order.tick_params(axis="x", rotation=35)

    if values.size:
        aggregate: dict[str, float] = {}
        aggregate_depth: dict[str, int] = {}
        for label, value, depth in zip(labels, values, depths):
            aggregate[str(label)] = aggregate.get(str(label), 0.0) + float(value)
            aggregate_depth[str(label)] = min(aggregate_depth.get(str(label), int(depth)), int(depth))
        top = sorted(aggregate.items(), key=lambda item: item[1], reverse=True)[:8]
        top_labels = [item[0] for item in top][::-1]
        top_values = [item[1] for item in top][::-1]
        colors = ["#999999" if aggregate_depth[label] > 0 else "#D55E00" for label in top_labels]
        ax_top.barh(top_labels, top_values, color=colors, height=0.62)
    ax_top.set_xlabel(r"Accumulated $\Phi^{EID}$ residual")
    ax_top.grid(True, axis="x", color="0.88", lw=0.8)

    for label, axis in zip(("A", "B", "C"), axes):
        axis.text(-0.18, 1.05, label, transform=axis.transAxes, fontsize=13, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy decomposition of DMF module-level PhiEID.")
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--connectivity-labels", type=Path, default=DEFAULT_CONNECTIVITY_CSV)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--g-stride", type=int, default=2)
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1.0e-8)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t-total", type=float, default=1.2)
    parser.add_argument("--burn-in", type=float, default=0.3)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--use-bold", action="store_true")
    parser.add_argument("--independent-restarts", action="store_true")
    parser.add_argument("--stabilization-window", type=float, default=0.05)
    parser.add_argument("--stabilization-tolerance", type=float, default=0.15)
    parser.add_argument("--stabilization-confirm-windows", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = synthetic_payload() if args.synthetic_smoke else actual_payload(args)
    results_path = resolve_path(args.results)
    figure_path = resolve_path(args.figure)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(results_path, **payload)
    plot_payload(payload, figure_path)
    print(f"Saved PhiEID greedy figure to: {figure_path}")
    print(f"Saved PhiEID greedy results to: {results_path}")


if __name__ == "__main__":
    main()
