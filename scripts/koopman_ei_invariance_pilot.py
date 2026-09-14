#!/usr/bin/env python3
"""Pilot: coordinate invariance of EI and the limits of a Gaussian formula.

Experiment contract
-------------------
The one-factor question is: what changes when the intervention distribution changes,
while the stochastic channel is held fixed?  For the paired Gaussian-prior condition,
the same latent samples and the same noise are used in both coordinate systems.  The
only change is the invertible encoding ``X=asinh(Z)`` (and ``Y=asinh(W)``), so the
mechanism, noise, sample pairing, and measurement/channel are shared.  This is a
Gaussian latent intervention pulled back to X; it is not called an X-space uniform EI.

The fresh uniform conditions change only the intervention factor and are therefore a
separate comparison: X~Uniform[-1,1] versus Z~Uniform[-sinh(1),sinh(1)].  Do not
compare diagonal independent-noise refits as if they were the same channel.  All TM
estimates use the repository degree-3 estimator; its return value is in bits and is
reported here in nats.  The analytic Gaussian reference is closed form, while uniform
references use quadrature over the known additive-noise channel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr, roots_legendre

# Running a script by path puts ``scripts/`` first on sys.path; add repository root
# so the estimator is imported from the checked-out experiment implementation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp.TM.transport_map_density import estimate_mutual_information_transport_map


A = 0.8
SIGMA = 0.25
GAUSSIAN_REFERENCE = 0.5 * np.log1p((A / SIGMA) ** 2)


def tm_mi_nats(x: np.ndarray, y: np.ndarray) -> float:
    """Repository TM MI estimate, converted from bits to nats."""
    result = estimate_mutual_information_transport_map(
        x[:, None], y[:, None], degree=3, joint_order="source_first"
    )
    return float(result["mi_hat"]) * np.log(2.0)


def uniform_channel_mi_quadrature(
    low: float,
    high: float,
    *,
    input_transform=lambda values: values,
    nodes: int = 256,
    grid_size: int = 50001,
) -> float:
    """Reference MI for W=A*h(Z)+epsilon, Z uniform on [low, high]."""
    nodes_x, weights = roots_legendre(nodes)
    z = 0.5 * (high - low) * nodes_x + 0.5 * (high + low)
    weights = 0.5 * weights  # expectation under Uniform[low, high]
    transformed = np.asarray(input_transform(z), dtype=float)
    # Include essentially all Gaussian tail mass while keeping the grid regular.
    span = float(np.max(np.abs(A * transformed)))
    y = np.linspace(-span - 8.0 * SIGMA, span + 8.0 * SIGMA, grid_size)
    density = np.exp(-0.5 * ((y[:, None] - A * transformed[None, :]) / SIGMA) ** 2)
    density /= SIGMA * np.sqrt(2.0 * np.pi)
    density = density @ weights
    entropy = -np.trapz(density * np.log(np.maximum(density, 1e-300)), y)
    noise_entropy = 0.5 * np.log(2.0 * np.pi * np.e * SIGMA**2)
    return float(entropy - noise_entropy)


def run(seed: int, n: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    z_gauss = rng.normal(size=n)
    noise_gauss = rng.normal(scale=SIGMA, size=n)
    w_gauss = A * z_gauss + noise_gauss
    x_gauss = np.arcsinh(z_gauss)
    y_gauss = np.arcsinh(w_gauss)

    # Same samples/noise under the invertible coordinate change.
    gaussian_latent_tm = tm_mi_nats(z_gauss, w_gauss)
    gaussian_original_tm = tm_mi_nats(x_gauss, y_gauss)

    # Fresh uniform interventions: the distribution, not the channel, changes.
    shared_uniform = rng.uniform(-1.0, 1.0, size=n)
    shared_eps = rng.normal(scale=SIGMA, size=n)
    u_x = shared_uniform
    eps_x = shared_eps
    w_x = A * np.sinh(u_x) + eps_x
    u_z = shared_uniform * np.sinh(1.0)
    eps_z = shared_eps
    w_z = A * u_z + eps_z
    uniform_x_tm = tm_mi_nats(u_x, w_x)
    uniform_z_tm = tm_mi_nats(u_z, w_z)
    return {
        "seed": int(seed),
        "gaussian_latent_tm_nats": gaussian_latent_tm,
        "gaussian_original_tm_nats": gaussian_original_tm,
        "gaussian_latent_error_nats": gaussian_latent_tm - GAUSSIAN_REFERENCE,
        "gaussian_original_error_nats": gaussian_original_tm - GAUSSIAN_REFERENCE,
        "uniform_x_tm_nats": uniform_x_tm,
        "uniform_z_tm_nats": uniform_z_tm,
    }


def make_figure(records: list[dict[str, float]], uniform_refs: dict[str, float], path: Path) -> None:
    seeds = np.arange(len(records))
    latent = np.array([r["gaussian_latent_tm_nats"] for r in records])
    original = np.array([r["gaussian_original_tm_nats"] for r in records])
    ux = np.array([r["uniform_x_tm_nats"] for r in records])
    uz = np.array([r["uniform_z_tm_nats"] for r in records])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.4, 3.1), layout="constrained")
    colors = {"latent": "#3b6ea8", "original": "#d47a3a"}
    for i in seeds:
        ax0.plot([i, i], [latent[i], original[i]], color="#b6b6b6", lw=0.8, zorder=1)
    ax0.scatter(seeds, latent, s=26, color=colors["latent"], label="latent Z,W")
    ax0.scatter(seeds, original, s=26, color=colors["original"], label="encoded X,Y")
    ax0.axhline(GAUSSIAN_REFERENCE, color="black", ls="--", lw=1.0, label="closed form")
    ax0.set_xlabel("paired seed")
    ax0.set_ylabel("EI / mutual information (nats)")
    ax0.text(-0.16, 1.03, "a", transform=ax0.transAxes, fontsize=11, fontweight="bold")
    ax0.set_xticks(seeds)
    ax0.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)

    for i in seeds:
        ax1.plot([i, i], [ux[i], uz[i]], color="#b6b6b6", lw=0.8, zorder=1)
    ax1.scatter(seeds, ux, s=26, color="#7a5195", label="X uniform")
    ax1.scatter(seeds, uz, s=26, color="#2a9d8f", label="Z uniform")
    ax1.axhline(uniform_refs["x"], color="#7a5195", ls="--", lw=1.0, label="X quadrature")
    ax1.axhline(uniform_refs["z"], color="#2a9d8f", ls=":", lw=1.0, label="Z quadrature")
    ax1.set_xlabel("paired seed")
    ax1.set_ylabel("EI / mutual information (nats)")
    ax1.text(-0.16, 1.03, "b", transform=ax1.transAxes, fontsize=11, fontweight="bold")
    ax1.set_xticks(seeds)
    ax1.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed0", type=int, default=11)
    parser.add_argument("--smoke", action="store_true", help="small one-seed run")
    parser.add_argument("--output", type=Path, default=Path("fig/koopman_ei_invariance_pilot.png"))
    args = parser.parse_args()
    n = 512 if args.smoke else args.n
    seed_count = 1 if args.smoke else min(max(args.seeds, 1), 5)
    seeds = [args.seed0 + i for i in range(seed_count)]
    records = [run(seed, n) for seed in seeds]
    uniform_refs = {}
    refinements = {}
    for nodes in (64, 128, 256):
        refinements[str(nodes)] = {
            "x": uniform_channel_mi_quadrature(-1.0, 1.0, input_transform=np.sinh, nodes=nodes),
            "z": uniform_channel_mi_quadrature(-np.sinh(1.0), np.sinh(1.0), nodes=nodes),
        }
    uniform_refs = refinements["256"]
    # Identity-transform self-check against the analytic uniform-Gaussian density.
    low_z, high_z = -np.sinh(1.0), np.sinh(1.0)
    y_check = np.linspace(-A * high_z - 8 * SIGMA, A * high_z + 8 * SIGMA, 50001)
    analytic_density = (
        ndtr((y_check - A * low_z) / SIGMA)
        - ndtr((y_check - A * high_z) / SIGMA)
    ) / (A * (high_z - low_z))
    analytic_entropy = -np.trapz(
        analytic_density * np.log(np.maximum(analytic_density, 1e-300)), y_check
    )
    analytic_mi = analytic_entropy - 0.5 * np.log(2 * np.pi * np.e * SIGMA**2)
    identity_check = {
        "quadrature_identity_nats": uniform_channel_mi_quadrature(
            low_z, high_z, input_transform=lambda values: values, nodes=256
        ),
        "analytic_cdf_difference_nats": float(analytic_mi),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_figure(records, uniform_refs, args.output)

    # Optional compact analytic linear counterexample requested in the brief.
    g1 = np.diag([100.0, 0.01])
    g2 = np.diag([1.0, 1.0])
    target = np.array([0.0, 1.0])
    c1 = float(target @ np.linalg.inv(g1) @ target)
    c2 = float(target @ np.linalg.inv(g2) @ target)
    i1 = 0.5 * np.log(np.linalg.det(np.eye(2) + g1))
    i2 = 0.5 * np.log(np.linalg.det(np.eye(2) + g2))
    summary = {
        "n": n,
        "seeds": seeds,
        "a": A,
        "sigma": SIGMA,
        "gaussian_reference_nats": GAUSSIAN_REFERENCE,
        "uniform_quadrature_refinements_nats": refinements,
        "uniform_reference_nats": uniform_refs,
        "uniform_identity_density_check_nats": identity_check,
        "records": records,
        "means_nats": {
            key: float(np.mean([r[key] for r in records]))
            for key in records[0]
            if key.endswith("tm_nats") or key.endswith("error_nats")
        },
        "limitations": [
            "TM coordinate-dependent finite-sample bias is not a change in true EI.",
            "Uniform interventions are fresh draws and deliberately change the intervention factor.",
            "Quadrature is a known-channel reference; the primary estimates are TM degree 3.",
        ],
        "analytic_linear_counterexample": {
            "I_G1_nats": float(i1), "I_G2_nats": float(i2),
            "target_cost_G1": c1, "target_cost_G2": c2,
            "claim": "EI(G1)>EI(G2) while target cost under G1 is greater than under G2.",
        },
        "figure": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
