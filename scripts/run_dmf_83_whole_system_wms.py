#!/usr/bin/env python3
"""Compute observational whole-system WMS for the 83-ROI DMF rate process."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.brain.dmf_fig6 import DMFParameters, gaussian_mutual_information, simulate_dmf


DEFAULT_SOURCE = ROOT / "exp" / "brain" / "result_lausanne_fig6" / "count_00_fig6b_mean_rate.npz"
DEFAULT_OUTPUT = ROOT / "results" / "dmf_83_whole_system_wms" / "observational_gaussian_mi_seeds3_5.npz"


def whole_system_wms(rates: np.ndarray, *, lag: int = 1) -> tuple[float, float, float]:
    """Return I(R_t;R_t+lag) - sum_i I(R_i,t;R_i,t+lag), in bits."""
    if lag < 1 or rates.shape[0] <= lag + rates.shape[1]:
        raise ValueError("The rate trace is too short for the requested lag and dimension.")
    source = np.asarray(rates[:-lag], dtype=float)
    target = np.asarray(rates[lag:], dtype=float)
    joint = np.column_stack((source, target))
    covariance = np.cov(joint, rowvar=False, bias=False)
    covariance = 0.5 * (covariance + covariance.T)
    n_roi = source.shape[1]
    whole_mi = gaussian_mutual_information(
        covariance, sources=range(n_roi), targets=range(n_roi, 2 * n_roi), log_base=2.0
    )
    self_mi = sum(
        gaussian_mutual_information(
            covariance, sources=[roi], targets=[n_roi + roi], log_base=2.0
        )
        for roi in range(n_roi)
    )
    return float(whole_mi - self_mi), float(whole_mi), float(self_mi)


def run(source: Path, output: Path, seeds: tuple[int, ...], lag: int) -> None:
    with np.load(source) as archive:
        g_values = np.asarray(archive["G"], dtype=float)
        connectivity = np.asarray(archive["connectivity"], dtype=float)
        j_fic = np.asarray(archive["j_fic"], dtype=float)
        mean_rate_reference = np.asarray(archive["mean_rate_hz"], dtype=float)

    wms = np.empty((len(seeds), g_values.size), dtype=float)
    whole_mi = np.empty_like(wms)
    self_mi_sum = np.empty_like(wms)
    simulated_mean_rate = np.empty_like(wms)
    parameters = DMFParameters()

    for seed_index, seed in enumerate(seeds):
        initial_se = None
        initial_si = None
        for g_index, coupling in enumerate(g_values):
            result = simulate_dmf(
                connectivity,
                float(coupling),
                np.asarray(j_fic[g_index], dtype=float),
                parameters=parameters,
                seed=seed + g_index,
                initial_se=initial_se,
                initial_si=initial_si,
                record_rate_trace=True,
            )
            start = int(result["stabilization_start_step"])
            rates = np.asarray(result["region_rate_trace_hz"], dtype=float)[start:]
            wms_value, whole_value, self_value = whole_system_wms(rates, lag=lag)
            wms[seed_index, g_index] = wms_value
            whole_mi[seed_index, g_index] = whole_value
            self_mi_sum[seed_index, g_index] = self_value
            simulated_mean_rate[seed_index, g_index] = float(result["mean_rate_hz"])
            initial_se = np.asarray(result["final_se"], dtype=float)
            initial_si = np.asarray(result["final_si"], dtype=float)
            print(
                f"[{seed_index + 1}/{len(seeds)} seed, {g_index + 1}/{g_values.size} G] "
                f"G={coupling:.2f} WMS={wms_value:.5g} bits",
                flush=True,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        G=g_values,
        seeds=np.asarray(seeds, dtype=int),
        lag=np.asarray(lag, dtype=int),
        phi_wms=wms,
        whole_mi=whole_mi,
        self_mi_sum=self_mi_sum,
        simulated_mean_rate_hz=simulated_mean_rate,
        mean_rate_reference_hz=mean_rate_reference,
        roi_count=np.asarray(connectivity.shape[0], dtype=int),
        estimator=np.asarray("observational Gaussian mutual information"),
        state=np.asarray("natural excitatory-rate trajectory"),
    )
    print(f"Saved cache: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--lag", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.source, args.output, tuple(args.seeds), args.lag)


if __name__ == "__main__":
    main()
