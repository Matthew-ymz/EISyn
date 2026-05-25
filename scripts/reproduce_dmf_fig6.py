from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CONNECTOME = Path("data/connectome_brodmann82.npy")
RESULT_SUBDIR = Path("results/dmf_fig6_brodmann82")
FIG_SUBDIR = Path("fig/dmf_fig6_brodmann82")


@dataclass(frozen=True)
class BrodmannConnectome:
    matrix: np.ndarray
    labels: list[str]


@dataclass(frozen=True)
class DMFParameters:
    w_e: float = 1.0
    w_i: float = 0.7
    i0: float = 0.382
    w_plus: float = 1.4
    j_nmda: float = 0.15
    gain_e: float = 310.0
    threshold_e: float = 0.403
    shape_e: float = 0.16
    gain_i: float = 615.0
    threshold_i: float = 0.288
    shape_i: float = 0.087
    tau_e: float = 0.100
    tau_i: float = 0.010
    gamma_e: float = 0.641
    sigma: float = 0.01
    dt: float = 1.0e-4
    t_total: float = 1.5
    burn_in: float = 0.3
    init_se: float = 0.001
    init_si: float = 0.001


@dataclass(frozen=True)
class FICParameters:
    target_rate_hz: float = 3.0
    tolerance_hz: float = 0.05
    max_iterations: int = 12
    learning_rate: float = 0.025
    initial_j: float = 1.0
    min_j: float = 0.1
    max_j: float = 10.0
    calibration_sigma: float = 0.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    root_candidate = _repo_root() / candidate
    return root_candidate if root_candidate.exists() else candidate.resolve()


def load_brodmann_labels(path: str | Path | None = None, *, n_regions: int = 82) -> list[str]:
    node_path = _repo_root() / "data" / "Node_Brodmann82.node" if path is None else _resolve_path(path)
    if not node_path.exists():
        return [str(index + 1) for index in range(n_regions)]

    labels: list[str] = []
    with node_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 6:
                labels.append(parts[5])
    if len(labels) != n_regions:
        return [str(index + 1) for index in range(n_regions)]
    return labels


def load_brodmann_connectome(path: str | Path = DEFAULT_CONNECTOME) -> BrodmannConnectome:
    matrix = np.asarray(np.load(_resolve_path(path)), dtype=float)
    if matrix.shape != (82, 82):
        raise ValueError(f"Expected a Brodmann82 82x82 connectome, got {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("Connectome contains non-finite values.")
    if np.any(matrix < 0.0):
        raise ValueError("Connectome must be nonnegative.")

    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return BrodmannConnectome(matrix=matrix, labels=load_brodmann_labels(n_regions=matrix.shape[0]))


def transfer_function(current: np.ndarray, gain: float, threshold: float, shape: float) -> np.ndarray:
    y = gain * (current - threshold)
    denominator = 1.0 - np.exp(-shape * y)
    rate = np.empty_like(y)
    near_zero = np.abs(denominator) < 1.0e-12
    rate[~near_zero] = y[~near_zero] / denominator[~near_zero]
    rate[near_zero] = 1.0 / shape
    return rate


def simulate_dmf(
    connectivity: np.ndarray,
    coupling_g: float,
    j_fic: np.ndarray,
    *,
    parameters: DMFParameters = DMFParameters(),
    seed: int = 0,
    initial_se: np.ndarray | None = None,
    initial_si: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    n_regions = connectivity.shape[0]
    n_steps = int(round(parameters.t_total / parameters.dt))
    burn_steps = int(round(parameters.burn_in / parameters.dt))
    if n_steps < 3 or burn_steps >= n_steps - 2:
        raise ValueError("Simulation needs at least two post-burn lagged samples.")

    rng = np.random.default_rng(seed)
    se = np.full(n_regions, parameters.init_se) if initial_se is None else np.asarray(initial_se, dtype=float).copy()
    si = np.full(n_regions, parameters.init_si) if initial_si is None else np.asarray(initial_si, dtype=float).copy()
    j_fic = np.asarray(j_fic, dtype=float)
    rates = np.empty((n_steps, n_regions), dtype=float)

    for step in range(n_steps):
        input_e = (
            parameters.w_e * parameters.i0
            + parameters.w_plus * parameters.j_nmda * se
            + coupling_g * parameters.j_nmda * (connectivity @ se)
            - j_fic * si
        )
        input_i = parameters.w_i * parameters.i0 + parameters.j_nmda * se - si
        rate_e = transfer_function(input_e, parameters.gain_e, parameters.threshold_e, parameters.shape_e)
        rate_i = transfer_function(input_i, parameters.gain_i, parameters.threshold_i, parameters.shape_i)

        noise_e = parameters.sigma * np.sqrt(parameters.dt) * rng.standard_normal(n_regions)
        noise_i = parameters.sigma * np.sqrt(parameters.dt) * rng.standard_normal(n_regions)
        se += parameters.dt * (-se / parameters.tau_e + (1.0 - se) * parameters.gamma_e * rate_e) + noise_e
        si += parameters.dt * (-si / parameters.tau_i + rate_i) + noise_i
        se = np.clip(se, 0.0, 1.0)
        si = np.clip(si, 0.0, 1.0)
        rates[step] = rate_e

    post_burn_rates = rates[burn_steps:]
    return {
        "post_burn_rates_hz": post_burn_rates,
        "mean_rate_hz": float(post_burn_rates.mean()),
        "mean_region_rate_hz": post_burn_rates.mean(axis=0),
        "final_se": se,
        "final_si": si,
    }


def calibrate_j_fic(
    connectivity: np.ndarray,
    coupling_g: float,
    *,
    parameters: DMFParameters,
    fic: FICParameters,
    seed: int,
) -> dict[str, np.ndarray | float | bool]:
    calibration_parameters = replace(parameters, sigma=fic.calibration_sigma)
    n_regions = connectivity.shape[0]

    def evaluate_scalar_j(value: float) -> float:
        result = simulate_dmf(
            connectivity,
            coupling_g,
            np.full(n_regions, value, dtype=float),
            parameters=calibration_parameters,
            seed=seed,
        )
        return float(result["mean_rate_hz"])

    candidates: list[tuple[float, float]] = []
    low = float(fic.min_j)
    high = float(fic.max_j)
    low_rate = evaluate_scalar_j(low)
    high_rate = evaluate_scalar_j(high)
    candidates.extend([(low, low_rate), (high, high_rate)])

    iterations = 2
    for iteration in range(1, fic.max_iterations + 1):
        mid = 0.5 * (low + high)
        mid_rate = evaluate_scalar_j(mid)
        candidates.append((mid, mid_rate))
        iterations += 1
        if abs(mid_rate - fic.target_rate_hz) <= fic.tolerance_hz:
            j_fic = np.full(n_regions, mid, dtype=float)
            return {
                "j_fic": j_fic,
                "converged": True,
                "iterations": float(iterations),
                "max_abs_error_hz": abs(mid_rate - fic.target_rate_hz),
            }

        # Mean firing rate decreases sharply as inhibition increases in this model.
        if mid_rate > fic.target_rate_hz:
            low = mid
        else:
            high = mid

    best_j, best_rate = min(candidates, key=lambda item: abs(item[1] - fic.target_rate_hz))
    return {
        "j_fic": np.full(n_regions, best_j, dtype=float),
        "converged": abs(best_rate - fic.target_rate_hz) <= fic.tolerance_hz,
        "iterations": float(iterations),
        "max_abs_error_hz": float(abs(best_rate - fic.target_rate_hz)),
    }


def _safe_logdet(matrix: np.ndarray, *, atol: float = 1.0e-10) -> float:
    sym = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(sym)
    return float(np.log(np.clip(eigenvalues, atol, None)).sum())


def gaussian_mutual_information(
    covariance: np.ndarray,
    *,
    sources: Sequence[int],
    targets: Sequence[int],
    log_base: float = np.e,
    atol: float = 1.0e-10,
) -> float:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square.")
    source = list(sources)
    target = list(targets)
    if not source or not target:
        return 0.0
    joint = source + target
    value = 0.5 * (
        _safe_logdet(covariance[np.ix_(source, source)], atol=atol)
        + _safe_logdet(covariance[np.ix_(target, target)], atol=atol)
        - _safe_logdet(covariance[np.ix_(joint, joint)], atol=atol)
    ) / np.log(log_base)
    return max(0.0, float(value))


def compute_pairwise_phi_metrics(rates: np.ndarray, *, atol: float = 1.0e-10) -> dict[str, float | int]:
    rates = np.asarray(rates, dtype=float)
    if rates.ndim != 2 or rates.shape[0] < 3 or rates.shape[1] < 2:
        raise ValueError("rates must have shape (time, regions) with at least 3 time points and 2 regions.")

    phi_wms_values: list[float] = []
    phi_r_values: list[float] = []
    for left in range(rates.shape[1] - 1):
        for right in range(left + 1, rates.shape[1]):
            lagged = np.column_stack(
                [
                    rates[:-1, left],
                    rates[:-1, right],
                    rates[1:, left],
                    rates[1:, right],
                ]
            )
            covariance = np.cov(lagged, rowvar=False, bias=False)
            covariance = 0.5 * (covariance + covariance.T)

            tdmi = gaussian_mutual_information(covariance, sources=[0, 1], targets=[2, 3], atol=atol)
            self_left = gaussian_mutual_information(covariance, sources=[0], targets=[2], atol=atol)
            self_right = gaussian_mutual_information(covariance, sources=[1], targets=[3], atol=atol)
            phi_wms = tdmi - self_left - self_right
            double_redundancy = min(
                gaussian_mutual_information(covariance, sources=[source], targets=[target], atol=atol)
                for source in (0, 1)
                for target in (2, 3)
            )
            phi_wms_values.append(float(phi_wms))
            phi_r_values.append(float(phi_wms + double_redundancy))

    return {
        "pair_count": len(phi_wms_values),
        "phi_wms_mean": float(np.mean(phi_wms_values)),
        "phi_r_mean": float(np.mean(phi_r_values)),
        "phi_wms_std": float(np.std(phi_wms_values)),
        "phi_r_std": float(np.std(phi_r_values)),
    }


def run_sweep(
    connectome: BrodmannConnectome,
    *,
    g_values: np.ndarray,
    parameters: DMFParameters,
    fic: FICParameters,
    seed: int,
) -> dict[str, np.ndarray | float | bool]:
    calibration = calibrate_j_fic(
        connectome.matrix,
        float(g_values[0]),
        parameters=parameters,
        fic=fic,
        seed=seed,
    )
    j_fic = np.asarray(calibration["j_fic"], dtype=float)

    mean_rates = np.empty(g_values.shape[0], dtype=float)
    phi_wms = np.empty(g_values.shape[0], dtype=float)
    phi_r = np.empty(g_values.shape[0], dtype=float)
    pair_count = np.empty(g_values.shape[0], dtype=int)
    final_se = None
    final_si = None

    for index, coupling_g in enumerate(g_values):
        result = simulate_dmf(
            connectome.matrix,
            float(coupling_g),
            j_fic,
            parameters=parameters,
            seed=seed + index,
            initial_se=final_se,
            initial_si=final_si,
        )
        rates = np.asarray(result["post_burn_rates_hz"], dtype=float)
        metrics = compute_pairwise_phi_metrics(rates)
        mean_rates[index] = float(result["mean_rate_hz"])
        phi_wms[index] = float(metrics["phi_wms_mean"])
        phi_r[index] = float(metrics["phi_r_mean"])
        pair_count[index] = int(metrics["pair_count"])
        final_se = np.asarray(result["final_se"], dtype=float)
        final_si = np.asarray(result["final_si"], dtype=float)

    return {
        "G": g_values,
        "mean_rate_hz": mean_rates,
        "phi_wms": phi_wms,
        "phi_r": phi_r,
        "pair_count": pair_count,
        "j_fic": j_fic,
        "j_fic_converged": bool(calibration["converged"]),
        "j_fic_iterations": float(calibration["iterations"]),
        "j_fic_max_abs_error_hz": float(calibration["max_abs_error_hz"]),
    }


def save_outputs(
    sweep: dict[str, np.ndarray | float | bool],
    *,
    connectome: BrodmannConnectome,
    connectome_path: Path,
    output_dir: Path,
    parameters: DMFParameters,
    fic: FICParameters,
    seed: int,
) -> dict[str, Path]:
    results_dir = output_dir / RESULT_SUBDIR
    fig_dir = output_dir / FIG_SUBDIR
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    npz_path = results_dir / "sweep.npz"
    np.savez(
        npz_path,
        G=sweep["G"],
        mean_rate_hz=sweep["mean_rate_hz"],
        phi_wms=sweep["phi_wms"],
        phi_r=sweep["phi_r"],
        pair_count=sweep["pair_count"],
        j_fic=sweep["j_fic"],
    )

    summary_path = results_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["G", "mean_rate_hz", "phi_wms", "phi_r", "pair_count"])
        writer.writeheader()
        for row in zip(sweep["G"], sweep["mean_rate_hz"], sweep["phi_wms"], sweep["phi_r"], sweep["pair_count"]):
            writer.writerow(
                {
                    "G": f"{float(row[0]):.8g}",
                    "mean_rate_hz": f"{float(row[1]):.8g}",
                    "phi_wms": f"{float(row[2]):.8g}",
                    "phi_r": f"{float(row[3]):.8g}",
                    "pair_count": int(row[4]),
                }
            )

    figure_path = fig_dir / "fig6b_brodmann82_pilot.png"
    plot_fig6b_pilot(sweep, figure_path)

    metadata_path = results_dir / "metadata.json"
    metadata = {
        "source_paper": "Toward a unified taxonomy of information dynamics via Integrated Information Decomposition",
        "atlas_note": "Brodmann82 approximation; paper used HCP Lausanne-83.",
        "phi_metrics_source": "excitatory_rate_gaussian_proxy",
        "connectome": {
            "path": str(connectome_path),
            "n_regions": int(connectome.matrix.shape[0]),
            "labels": connectome.labels,
        },
        "dmf_parameters": asdict(parameters),
        "fic_parameters": asdict(fic),
        "seed": int(seed),
        "j_fic": {
            "converged": bool(sweep["j_fic_converged"]),
            "iterations": float(sweep["j_fic_iterations"]),
            "max_abs_error_hz": float(sweep["j_fic_max_abs_error_hz"]),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"npz": npz_path, "summary": summary_path, "metadata": metadata_path, "figure": figure_path}


def plot_fig6b_pilot(sweep: dict[str, np.ndarray | float | bool], output_path: Path) -> None:
    g_values = np.asarray(sweep["G"], dtype=float)
    mean_rates = np.asarray(sweep["mean_rate_hz"], dtype=float)
    phi_wms = np.asarray(sweep["phi_wms"], dtype=float)
    phi_r = np.asarray(sweep["phi_r"], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.2), sharex=True, constrained_layout=True)
    axes[0].scatter(g_values, mean_rates, s=12, color="black")
    axes[0].plot(g_values, mean_rates, lw=1.0, color="0.25")
    axes[0].set_ylabel("Mean firing rate (Hz)")
    axes[0].grid(True, color="0.82", lw=0.8)

    axes[1].scatter(g_values, phi_wms, s=12, color="#2f80c1", label=r"$\Phi^{WMS}$")
    axes[1].scatter(g_values, phi_r, s=12, color="#df2b2b", label=r"$\Phi^R$")
    axes[1].plot(g_values, phi_wms, lw=0.9, color="#2f80c1")
    axes[1].plot(g_values, phi_r, lw=0.9, color="#df2b2b")
    axes[1].set_xlabel(r"Global coupling $G$")
    axes[1].set_ylabel("Integrated information\n(rate-Gaussian proxy)")
    axes[1].grid(True, color="0.82", lw=0.8)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    for label, axis in zip(("A", "B"), axes):
        axis.text(-0.12, 1.02, label, transform=axis.transAxes, fontsize=14, fontweight="bold")

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brodmann82 pilot reproduction of Fig. 6B DMF dynamics.")
    parser.add_argument("--connectome", type=Path, default=DEFAULT_CONNECTOME)
    parser.add_argument("--g-min", type=float, default=1.0)
    parser.add_argument("--g-max", type=float, default=3.0)
    parser.add_argument("--g-count", type=int, default=101)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t-total", type=float, default=DMFParameters().t_total)
    parser.add_argument("--burn-in", type=float, default=DMFParameters().burn_in)
    parser.add_argument("--dt", type=float, default=DMFParameters().dt)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--j-fic-target-rate", type=float, default=FICParameters().target_rate_hz)
    parser.add_argument("--j-fic-max-iters", type=int, default=FICParameters().max_iterations)
    parser.add_argument("--j-fic-learning-rate", type=float, default=FICParameters().learning_rate)
    parser.add_argument("--j-fic-tolerance", type=float, default=FICParameters().tolerance_hz)
    parser.add_argument("--skip-trace", action="store_true", help="Accepted for smoke-test compatibility; no trace plot is emitted.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.g_count < 2:
        raise ValueError("--g-count must be at least 2.")

    connectome_path = _resolve_path(args.connectome)
    connectome = load_brodmann_connectome(connectome_path)
    parameters = replace(DMFParameters(), t_total=args.t_total, burn_in=args.burn_in, dt=args.dt)
    fic = replace(
        FICParameters(),
        target_rate_hz=args.j_fic_target_rate,
        max_iterations=args.j_fic_max_iters,
        learning_rate=args.j_fic_learning_rate,
        tolerance_hz=args.j_fic_tolerance,
    )
    g_values = np.linspace(args.g_min, args.g_max, args.g_count)
    sweep = run_sweep(connectome, g_values=g_values, parameters=parameters, fic=fic, seed=args.seed)
    paths = save_outputs(
        sweep,
        connectome=connectome,
        connectome_path=connectome_path,
        output_dir=args.output_dir,
        parameters=parameters,
        fic=fic,
        seed=args.seed,
    )
    print(f"Saved sweep: {paths['npz']}")
    print(f"Saved summary: {paths['summary']}")
    print(f"Saved metadata: {paths['metadata']}")
    print(f"Saved figure: {paths['figure']}")


if __name__ == "__main__":
    main()
