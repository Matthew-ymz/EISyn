from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DMF_MODULE_PATH = ROOT / "exp" / "brain" / "dmf_fig6.py"
DEFAULT_SOURCE_RESULTS = ROOT / "exp" / "brain" / "result_lausanne_fig6" / "count_00_fig6b_mean_rate.npz"
DEFAULT_CONNECTIVITY_CSV = ROOT / "exp" / "brain" / "result_lausanne_fig6" / "count_00_connectivity.csv"
DEFAULT_FIGURE = ROOT / "fig" / "dmf_phi_eid_region_peel_decomposition.png"
DEFAULT_RESULTS = ROOT / "fig" / "dmf_phi_eid_region_peel_decomposition.npz"


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_dmf_module():
    spec = importlib.util.spec_from_file_location("dmf_fig6_exp_brain", DMF_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_logdet(matrix: np.ndarray, *, ridge: float = 1.0e-8) -> float:
    array = np.asarray(matrix, dtype=float)
    array = 0.5 * (array + array.T)
    scale = max(float(np.nanmean(np.diag(array))), 1.0e-12)
    regularized = array + float(ridge) * scale * np.eye(array.shape[0], dtype=float)
    sign, logdet = np.linalg.slogdet(regularized)
    if sign <= 0 or not np.isfinite(logdet):
        raise ValueError("Covariance matrix is not positive definite after regularization.")
    return float(logdet)


def load_region_labels(path: Path, n_regions: int) -> list[str]:
    if not path.exists():
        return [f"region_{index:03d}" for index in range(n_regions)]
    labels: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                labels.append(stripped.split(",")[0])
    return labels[:n_regions] if len(labels) >= n_regions else [f"region_{index:03d}" for index in range(n_regions)]


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


def region_ei(
    selected: Sequence[int],
    *,
    transition: np.ndarray,
    noise: np.ndarray,
    target_logdet: float,
    ridge: float,
) -> float:
    selected_set = set(map(int, selected))
    complement = [index for index in range(transition.shape[1]) if index not in selected_set]
    conditional = np.asarray(noise, dtype=float).copy()
    if complement:
        conditional = conditional + transition[:, complement] @ transition[:, complement].T
    return float(0.5 * (target_logdet - safe_logdet(conditional, ridge=ridge)) / np.log(2.0))


def region_phi(
    selected: Sequence[int],
    *,
    transition: np.ndarray,
    noise: np.ndarray,
    target_logdet: float,
    singleton_ei: np.ndarray,
    ridge: float,
) -> float:
    selected_tuple = tuple(map(int, selected))
    if not selected_tuple:
        return 0.0
    return float(
        region_ei(selected_tuple, transition=transition, noise=noise, target_logdet=target_logdet, ridge=ridge)
        - float(np.sum(singleton_ei[list(selected_tuple)]))
    )


def single_peel_decomposition(
    transition: np.ndarray,
    noise: np.ndarray,
    labels: Sequence[str],
    *,
    ridge: float = 1.0e-8,
    eps: float = 1.0e-6,
    max_depth: int | None = None,
) -> dict[str, np.ndarray]:
    target_cov = transition @ transition.T + noise
    target_logdet = safe_logdet(target_cov, ridge=ridge)
    singleton_ei = np.asarray(
        [
            region_ei([index], transition=transition, noise=noise, target_logdet=target_logdet, ridge=ridge)
            for index in range(transition.shape[1])
        ],
        dtype=float,
    )
    remaining = tuple(range(transition.shape[1]))
    root_phi = region_phi(
        remaining,
        transition=transition,
        noise=noise,
        target_logdet=target_logdet,
        singleton_ei=singleton_ei,
        ridge=ridge,
    )

    rows: list[tuple[int, int, str, float, float, int]] = []
    depth = 0
    while len(remaining) > 1:
        if max_depth is not None and depth >= int(max_depth):
            break
        current_phi = region_phi(
            remaining,
            transition=transition,
            noise=noise,
            target_logdet=target_logdet,
            singleton_ei=singleton_ei,
            ridge=ridge,
        )
        if current_phi < -float(eps):
            raise RuntimeError(
                f"Syn nonnegativity violation for coalition {remaining}: "
                f"minimum={current_phi:.12g}, threshold={-float(eps):.12g}, affected_count=1."
            )
        if current_phi <= float(eps):
            break
        best_removed: int | None = None
        best_child_phi = -np.inf
        for candidate in remaining:
            child = tuple(index for index in remaining if index != candidate)
            child_phi = region_phi(
                child,
                transition=transition,
                noise=noise,
                target_logdet=target_logdet,
                singleton_ei=singleton_ei,
                ridge=ridge,
            )
            if child_phi > best_child_phi:
                best_child_phi = child_phi
                best_removed = int(candidate)
        if best_removed is None:
            break
        residual = float(current_phi - float(best_child_phi))
        if residual < -float(eps):
            raise RuntimeError(
                f"Syn nonnegativity violation at peel depth {depth}: "
                f"minimum={residual:.12g}, threshold={-float(eps):.12g}, affected_count=1."
            )
        rows.append((depth, best_removed, str(labels[best_removed]), residual, current_phi, len(remaining)))
        remaining = tuple(index for index in remaining if index != best_removed)
        depth += 1

    return {
        "root_phi": np.asarray([root_phi], dtype=float),
        "peel_depths": np.asarray([row[0] for row in rows], dtype=int),
        "peel_region_indices": np.asarray([row[1] for row in rows], dtype=int),
        "peel_region_labels": np.asarray([row[2] for row in rows], dtype=object),
        "peel_residual_values": np.asarray([row[3] for row in rows], dtype=float),
        "parent_phi_values": np.asarray([row[4] for row in rows], dtype=float),
        "remaining_orders": np.asarray([row[5] for row in rows], dtype=int),
        "terminal_region_indices": np.asarray(remaining, dtype=int),
        "terminal_region_labels": np.asarray([labels[index] for index in remaining], dtype=object),
    }


def budgeted_local_split_decomposition(
    transition: np.ndarray,
    noise: np.ndarray,
    labels: Sequence[str],
    *,
    ridge: float = 1.0e-8,
    eps: float = 1.0e-6,
    split_search_budget: int = 300,
    max_depth: int = 4,
    seed: int = 0,
    split_tolerance: float = 1.0e-4,
    min_split_size: int = 1,
) -> dict[str, np.ndarray]:
    target_cov = transition @ transition.T + noise
    target_logdet = safe_logdet(target_cov, ridge=ridge)
    singleton_ei = np.asarray(
        [
            region_ei([index], transition=transition, noise=noise, target_logdet=target_logdet, ridge=ridge)
            for index in range(transition.shape[1])
        ],
        dtype=float,
    )
    phi_cache: dict[tuple[int, ...], float] = {}

    def phi(selected: Sequence[int]) -> float:
        key = tuple(sorted(map(int, selected)))
        if key not in phi_cache:
            phi_cache[key] = region_phi(
                key,
                transition=transition,
                noise=noise,
                target_logdet=target_logdet,
                singleton_ei=singleton_ei,
                ridge=ridge,
            )
        return float(phi_cache[key])

    def best_budgeted_split(subset: tuple[int, ...], rng: np.random.Generator):
        current_phi = phi(subset)
        if current_phi < -float(split_tolerance):
            raise RuntimeError(
                f"Syn nonnegativity violation for coalition {subset}: "
                f"minimum={current_phi:.12g}, threshold={-float(split_tolerance):.12g}, "
                "affected_count=1."
            )
        min_size = max(1, int(min_split_size))
        if len(subset) < 2 * min_size or current_phi <= float(eps):
            return None
        budget = max(1, int(split_search_budget))
        eval_count = 0
        best: tuple[float, float, tuple[int, ...], tuple[int, ...]] | None = None

        def consider(left_values: Sequence[int]) -> None:
            nonlocal eval_count, best
            if eval_count >= budget:
                return
            left = tuple(sorted(set(map(int, left_values))))
            if len(left) < min_size or len(left) > len(subset) - min_size:
                return
            left_set = set(left)
            right = tuple(index for index in subset if index not in left_set)
            if not right:
                return
            captured = phi(left) + phi(right)
            residual = current_phi - captured
            eval_count += 1
            if residual < -float(split_tolerance):
                raise RuntimeError(
                    f"Syn nonnegativity violation for split {left} | {right}: "
                    f"minimum={residual:.12g}, threshold={-float(split_tolerance):.12g}, "
                    "affected_count=1."
                )
            if best is None or captured > best[0] or (np.isclose(captured, best[0]) and residual < best[1]):
                best = (captured, residual, left, right)

        # Deterministic anchor candidates: all one-vs-rest peels and a few prefix halves.
        if min_size <= 1:
            for candidate in subset:
                consider((candidate,))
                if eval_count >= budget:
                    break
        if eval_count < budget:
            ordered = list(subset)
            for size in (max(1, len(ordered) // 4), max(1, len(ordered) // 2), max(1, 3 * len(ordered) // 4)):
                consider(ordered[:size])

        # Random starts plus one-flip hill climbing under the same evaluation budget.
        while eval_count < budget:
            size = int(rng.integers(min_size, len(subset) - min_size + 1))
            left = set(rng.choice(np.asarray(subset, dtype=int), size=size, replace=False).tolist())
            before = eval_count
            consider(left)
            improved = True
            while improved and eval_count < budget:
                improved = False
                best_local_left = set(left)
                best_local_score = -np.inf
                for candidate in rng.permutation(np.asarray(subset, dtype=int)):
                    trial = set(left)
                    if int(candidate) in trial:
                        if len(trial) == min_size:
                            continue
                        trial.remove(int(candidate))
                    else:
                        if len(trial) == len(subset) - min_size:
                            continue
                        trial.add(int(candidate))
                    old_best = best
                    consider(trial)
                    if best is not None and best is not old_best and best[2] == tuple(sorted(trial)) and best[0] > best_local_score:
                        best_local_score = best[0]
                        best_local_left = trial
                        improved = True
                    if eval_count >= budget:
                        break
                left = best_local_left
            if eval_count == before:
                break
        if best is None:
            return None
        captured, residual, left, right = best
        return {
            "captured": float(captured),
            "residual": float(residual),
            "left": left,
            "right": right,
            "eval_count": int(eval_count),
            "parent_phi": float(current_phi),
        }

    root = tuple(range(transition.shape[1]))
    root_phi = phi(root)
    if root_phi < -float(split_tolerance):
        raise RuntimeError(
            f"Syn nonnegativity violation at root: minimum={root_phi:.12g}, "
            f"threshold={-float(split_tolerance):.12g}, affected_count=1."
        )
    rng = np.random.default_rng(seed)
    rows: list[tuple[int, tuple[int, ...], tuple[int, ...], float, float, float, int]] = []
    stack: list[tuple[int, tuple[int, ...]]] = [(0, root)]
    while stack:
        depth, subset = stack.pop()
        if depth >= int(max_depth) or len(subset) <= 1:
            continue
        split = best_budgeted_split(subset, rng)
        if split is None:
            continue
        left = tuple(split["left"])
        right = tuple(split["right"])
        rows.append(
            (
                int(depth),
                left,
                right,
                float(split["residual"]),
                float(split["captured"]),
                float(split["parent_phi"]),
                int(split["eval_count"]),
            )
        )
        # Recurse into larger blocks first so small interpretable blocks stay near the top of the stack.
        children = sorted((left, right), key=len)
        for child in children:
            if len(child) > 1:
                stack.append((depth + 1, child))

    def labels_for(indices: Sequence[int]) -> str:
        return "+".join(str(labels[index]) for index in indices)

    return {
        "root_phi": np.asarray([root_phi], dtype=float),
        "split_depths": np.asarray([row[0] for row in rows], dtype=int),
        "split_left_indices": np.asarray([",".join(map(str, row[1])) for row in rows], dtype=object),
        "split_right_indices": np.asarray([",".join(map(str, row[2])) for row in rows], dtype=object),
        "split_left_labels": np.asarray([labels_for(row[1]) for row in rows], dtype=object),
        "split_right_labels": np.asarray([labels_for(row[2]) for row in rows], dtype=object),
        "split_left_orders": np.asarray([len(row[1]) for row in rows], dtype=int),
        "split_right_orders": np.asarray([len(row[2]) for row in rows], dtype=int),
        "split_residual_values": np.asarray([row[3] for row in rows], dtype=float),
        "split_captured_values": np.asarray([row[4] for row in rows], dtype=float),
        "split_parent_phi_values": np.asarray([row[5] for row in rows], dtype=float),
        "split_eval_counts": np.asarray([row[6] for row in rows], dtype=int),
    }


def synthetic_payload(
    *,
    mode: str = "single-peel",
    split_search_budget: int = 300,
    max_depth: int = 30,
    seed: int = 0,
    min_split_size: int = 1,
) -> dict[str, np.ndarray]:
    g_values = np.asarray([1.0, 1.5, 2.0], dtype=float)
    labels = [f"R{index + 1}" for index in range(5)]
    row_keys = (
        ["peel_depths", "peel_region_indices", "peel_region_labels", "peel_residual_values", "parent_phi_values", "remaining_orders"]
        if mode == "single-peel"
        else [
            "split_depths",
            "split_left_indices",
            "split_right_indices",
            "split_left_labels",
            "split_right_labels",
            "split_left_orders",
            "split_right_orders",
            "split_residual_values",
            "split_captured_values",
            "split_parent_phi_values",
            "split_eval_counts",
        ]
    )
    all_rows: dict[str, list[np.ndarray]] = {key: [] for key in row_keys}
    phi = np.empty(g_values.shape, dtype=float)
    g_index_rows: list[np.ndarray] = []
    for g_index, coupling_g in enumerate(g_values):
        transition = np.asarray(
            [
                [0.52, 0.18 + 0.10 * coupling_g, 0.02, 0.00, 0.00],
                [0.18 + 0.10 * coupling_g, 0.50, 0.03, 0.00, 0.00],
                [0.00, 0.02, 0.48, 0.18, 0.05],
                [0.00, 0.00, 0.18, 0.50, 0.14],
                [0.00, 0.00, 0.03, 0.14, 0.46],
            ],
            dtype=float,
        )
        if mode == "single-peel":
            result = single_peel_decomposition(transition, np.diag([0.18] * 5), labels, max_depth=max_depth)
        else:
            result = budgeted_local_split_decomposition(
                transition,
                np.diag([0.18] * 5),
                labels,
                split_search_budget=split_search_budget,
                max_depth=max_depth,
                seed=seed + g_index,
                min_split_size=min_split_size,
            )
        phi[g_index] = float(result["root_phi"][0])
        index_shape = result["peel_depths"].shape if mode == "single-peel" else result["split_depths"].shape
        g_index_rows.append(np.full(index_shape, g_index, dtype=int))
        for key in all_rows:
            all_rows[key].append(result[key])

    payload = {"G": g_values, "phi_eid": phi, "region_labels": np.asarray(labels, dtype=object)}
    payload["mode"] = np.asarray(mode)
    payload["peel_g_indices" if mode == "single-peel" else "split_g_indices"] = np.concatenate(g_index_rows)
    for key, parts in all_rows.items():
        payload[key] = np.concatenate(parts)
    return payload


def actual_payload(args: argparse.Namespace) -> dict[str, np.ndarray]:
    dmf = load_dmf_module()
    archive = np.load(resolve_path(args.source_results))
    g_all = np.asarray(archive["G"], dtype=float)
    selected = np.arange(0, g_all.size, max(1, int(args.g_stride)))
    g_values = g_all[selected]
    connectivity = np.asarray(archive["connectivity"], dtype=float)
    j_fic = np.asarray(archive["j_fic"], dtype=float)[selected]
    labels = load_region_labels(resolve_path(args.connectivity_labels), connectivity.shape[0])
    parameters = dmf.DMFParameters(t_total=args.t_total, burn_in=args.burn_in, dt=args.dt, sigma=args.sigma)
    stabilization = dmf.StabilizationParameters(
        window=args.stabilization_window,
        tolerance_hz=args.stabilization_tolerance,
        confirm_windows=args.stabilization_confirm_windows,
    )

    row_keys = (
        ["peel_depths", "peel_region_indices", "peel_region_labels", "peel_residual_values", "parent_phi_values", "remaining_orders"]
        if args.mode == "single-peel"
        else [
            "split_depths",
            "split_left_indices",
            "split_right_indices",
            "split_left_labels",
            "split_right_labels",
            "split_left_orders",
            "split_right_orders",
            "split_residual_values",
            "split_captured_values",
            "split_parent_phi_values",
            "split_eval_counts",
        ]
    )
    all_rows: dict[str, list[np.ndarray]] = {key: [] for key in row_keys}
    phi = np.empty(g_values.shape, dtype=float)
    g_index_rows: list[np.ndarray] = []
    initial_se = None
    initial_si = None
    for g_index, coupling_g in enumerate(g_values):
        simulation = dmf.simulate_dmf(
            connectivity,
            float(coupling_g),
            np.asarray(j_fic[g_index], dtype=float),
            parameters=parameters,
            stabilization_parameters=stabilization,
            seed=args.seed + int(selected[g_index]),
            initial_se=initial_se if not args.independent_restarts else None,
            initial_si=initial_si if not args.independent_restarts else None,
            record_rate_trace=True,
        )
        start_step = int(float(simulation["stabilization_start_step"]))
        rates = np.asarray(simulation["region_rate_trace_hz"], dtype=float)[start_step:]
        series = dmf.transform_rates_to_bold(rates, dt=args.dt) if args.use_bold else rates
        source, target = lagged_samples(series, args.tau)
        transition, noise = fit_standardized_transition(source, target)
        if args.mode == "single-peel":
            result = single_peel_decomposition(
                transition,
                noise,
                labels,
                ridge=args.ridge,
                eps=args.eps,
                max_depth=args.max_depth,
            )
        else:
            result = budgeted_local_split_decomposition(
                transition,
                noise,
                labels,
                ridge=args.ridge,
                eps=args.eps,
                split_search_budget=args.split_search_budget,
                max_depth=args.max_depth,
                seed=args.seed + int(selected[g_index]),
                split_tolerance=args.split_tolerance,
                min_split_size=args.min_split_size,
            )
        phi[g_index] = float(result["root_phi"][0])
        index_shape = result["peel_depths"].shape if args.mode == "single-peel" else result["split_depths"].shape
        g_index_rows.append(np.full(index_shape, g_index, dtype=int))
        for key in all_rows:
            all_rows[key].append(result[key])
        initial_se = np.asarray(simulation["final_se"], dtype=float)
        initial_si = np.asarray(simulation["final_si"], dtype=float)
        print(f"[{g_index + 1}/{g_values.size}] {args.mode} PhiEID={phi[g_index]:.4f} at G={coupling_g:.3f}")

    payload = {
        "G": g_values,
        "phi_eid": phi,
        "mean_rate_hz": np.asarray(archive["mean_rate_hz"], dtype=float)[selected],
        "region_labels": np.asarray(labels, dtype=object),
        "mode": np.asarray(args.mode),
    }
    payload["peel_g_indices" if args.mode == "single-peel" else "split_g_indices"] = np.concatenate(g_index_rows)
    for key, parts in all_rows.items():
        payload[key] = np.concatenate(parts)
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


def plot_payload(payload: dict[str, np.ndarray], output_path: Path, *, top_n: int = 14) -> None:
    configure_matplotlib()
    g_values = np.asarray(payload["G"], dtype=float)
    phi = np.asarray(payload["phi_eid"], dtype=float)
    peak_index = int(np.nanargmax(phi))
    mode = str(np.asarray(payload.get("mode", np.asarray("single-peel"))))
    is_split = "split_g_indices" in payload
    g_indices = np.asarray(payload["split_g_indices" if is_split else "peel_g_indices"], dtype=int)
    peak_mask = g_indices == peak_index
    if is_split:
        left_labels = np.asarray(payload["split_left_labels"], dtype=object)[peak_mask]
        right_orders = np.asarray(payload["split_right_orders"], dtype=int)[peak_mask]
        left_orders = np.asarray(payload["split_left_orders"], dtype=int)[peak_mask]
        labels = np.asarray(
            [
                f"{left} | rest {right_order}" if left_order <= 3 else f"L{left_order} | R{right_order}"
                for left, left_order, right_order in zip(left_labels, left_orders, right_orders)
            ],
            dtype=object,
        )
        values = np.asarray(payload["split_residual_values"], dtype=float)[peak_mask]
        depths = np.asarray(payload["split_depths"], dtype=int)[peak_mask]
        orders = left_orders + right_orders
    else:
        labels = np.asarray(payload["peel_region_labels"], dtype=object)[peak_mask]
        values = np.asarray(payload["peel_residual_values"], dtype=float)[peak_mask]
        depths = np.asarray(payload["peel_depths"], dtype=int)[peak_mask]
        orders = np.asarray(payload["remaining_orders"], dtype=int)[peak_mask]

    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.5), constrained_layout=True, gridspec_kw={"width_ratios": [1.05, 1.25, 1.35]})
    ax_curve, ax_depth, ax_top = axes
    ax_curve.plot(g_values, phi, color="#0072B2", lw=1.7)
    ax_curve.scatter(g_values, phi, color="#0072B2", s=20)
    ax_curve.axvline(g_values[peak_index], color="0.35", lw=0.9, ls="--")
    ax_curve.set_xlabel("Global coupling $G$")
    ax_curve.set_ylabel(r"83-region $\Phi^{EID}$")
    ax_curve.set_ylim(bottom=0.0)
    ax_curve.grid(True, color="0.88", lw=0.8)

    shown = min(top_n, values.size)
    ax_depth.bar(depths[:shown], values[:shown], color="#56B4E9", width=0.8)
    ax_depth.set_xlabel(("Split" if is_split else "Peel") + " depth at peak $G$")
    ax_depth.set_ylabel("Residual")
    ax_depth.grid(True, axis="y", color="0.88", lw=0.8)
    if shown:
        unit = "splits" if is_split else "peels"
        ax_depth.set_title(f"Peak G = {g_values[peak_index]:.2f}; first {shown} {unit}", fontsize=8)

    if values.size:
        order = np.argsort(values)[::-1][:top_n]
        top_labels = [str(labels[index]) for index in order][::-1]
        top_values = [float(values[index]) for index in order][::-1]
        top_orders = [int(orders[index]) for index in order][::-1]
        colors = ["#D55E00" if (is_split and order_value > 20) or (not is_split and order_value > 70) else "#999999" for order_value in top_orders]
        ax_top.barh(top_labels, top_values, color=colors, height=0.62)
    ax_top.set_xlabel("Residual at split" if is_split else "Residual at removal")
    ax_top.grid(True, axis="x", color="0.88", lw=0.8)

    for label, axis in zip(("A", "B", "C"), axes):
        axis.text(-0.16, 1.05, label, transform=axis.transAxes, fontsize=13, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="83-region constrained single-peel PhiEID decomposition for DMF.")
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--connectivity-labels", type=Path, default=DEFAULT_CONNECTIVITY_CSV)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--mode", choices=("single-peel", "local-split"), default="single-peel")
    parser.add_argument("--g-stride", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=30)
    parser.add_argument("--split-search-budget", type=int, default=300)
    parser.add_argument("--split-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--min-split-size", type=int, default=1)
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
    payload = (
        synthetic_payload(
            mode=args.mode,
            split_search_budget=args.split_search_budget,
            max_depth=args.max_depth,
            seed=args.seed,
            min_split_size=args.min_split_size,
        )
        if args.synthetic_smoke
        else actual_payload(args)
    )
    results_path = resolve_path(args.results)
    figure_path = resolve_path(args.figure)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(results_path, **payload)
    plot_payload(payload, figure_path)
    print(f"Saved region {args.mode} PhiEID figure to: {figure_path}")
    print(f"Saved region {args.mode} PhiEID results to: {results_path}")


if __name__ == "__main__":
    main()
