"""Permutation Shapley of the fixed-target, 100-ROI conditional-Xi game.

Contract: reuse all 8 seeds x 3 G covariances and E/I-paired ROI blocks.
Change attribution only, from leave-one-out leverage to ordinary ROI Shapley;
do not constrain permutations by Yeo and do not include within-ROI Xi.
Each independent draw is a random permutation paired with its reverse. The
same draw is used across conditions, so aggregate MC errors retain covariance
across conditions. MC errors are distinct from simulation-seed variability.
Cholesky prefix determinants evaluate the existing coalition game without
refitting dynamics or introducing a new EI estimator. No Syn is clipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.linalg import cholesky
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOLERANCE_BITS = 1e-8
VERSION = "paired-permutation-roi-xi-v1"


def source_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_game(covariances: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = np.asarray(covariances, dtype=float)
    if c.ndim != 3 or c.shape[1] != c.shape[2] or c.shape[1] % 2:
        raise ValueError("Expected [condition, 2*ROI, 2*ROI] covariance")
    if not np.isfinite(c).all() or not np.allclose(c, c.swapaxes(-1, -2), atol=1e-12, rtol=0):
        raise ValueError("Covariances must be finite and symmetric")
    c = 0.5 * (c + c.swapaxes(-1, -2))
    # Match the existing oracle, which floors eigenvalues at 1e-12. Refuse to
    # replace that definition silently when its floor would be active.
    if np.linalg.eigvalsh(c).min() <= 1e-12:
        raise ValueError("Covariance floor is active; Cholesky game would not match the existing oracle")
    n = c.shape[1] // 2
    block_logdet = np.stack([
        np.linalg.slogdet(c[:, [i, i+n]][:, :, [i, i+n]])[1] for i in range(n)
    ], axis=1)
    totals = 0.5 * (block_logdet.sum(axis=1) - np.linalg.slogdet(c)[1]) / np.log(2)
    return c, block_logdet, totals


def permutation_contributions(covariances, block_logdet, permutation):
    p = np.asarray(permutation, dtype=int)
    n = block_logdet.shape[1]
    if p.shape != (n,) or not np.array_equal(np.sort(p), np.arange(n)):
        raise ValueError("Not a permutation of all ROI indices")
    indices = np.column_stack((p, p+n)).ravel()
    result = np.empty_like(block_logdet)
    for condition, c in enumerate(covariances):
        factor = cholesky(c[np.ix_(indices, indices)], lower=True, check_finite=False)
        increments = 2 * np.log(np.diag(factor)).reshape(n, 2).sum(axis=1)
        result[condition, p] = 0.5 * (block_logdet[condition, p] - increments) / np.log(2)
    return result


class Moments:
    def __init__(self, shape):
        self.count = 0
        self.mean = np.zeros(shape)
        self.m2 = np.zeros(shape)

    def add(self, value):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def se(self):
        if self.count < 2:
            return np.full_like(self.mean, np.inf)
        # Only floating-point variance roundoff is projected, never Syn values.
        if self.m2.min() < -1e-12:
            raise RuntimeError("Invalid negative sampling variance")
        return np.sqrt(np.maximum(self.m2, 0) / (self.count - 1) / self.count)


def estimate(covariances, *, rng_seed=20260831, min_pairs=512, max_pairs=16384,
             mean_se_target=0.001, condition_se_target=0.003, split_target=0.004):
    if min_pairs < 2 or max_pairs < min_pairs or max_pairs % min_pairs:
        raise ValueError("Require max_pairs to be a power-of-two multiple of min_pairs >= 2")
    ratio = max_pairs // min_pairs
    if ratio & (ratio - 1):
        raise ValueError("Require max_pairs to be a power-of-two multiple of min_pairs")
    if min(mean_se_target, condition_se_target, split_target) <= 0:
        raise ValueError("Precision targets must be positive")
    c, block_logdet, totals = prepare_game(covariances)
    n = block_logdet.shape[1]
    per_condition = Moments(block_logdet.shape)
    aggregate = Moments((n,))
    rng = np.random.default_rng(rng_seed)
    next_check, previous_mean = min_pairs, None
    checkpoints = []
    negative_count, minimum_marginal, maximum_closure = 0, float("inf"), 0.0
    start = perf_counter()
    converged = False
    for pair in range(1, max_pairs + 1):
        p = rng.permutation(n)
        forward = permutation_contributions(c, block_logdet, p)
        backward = permutation_contributions(c, block_logdet, p[::-1])
        for values in (forward, backward):
            affected = values < -TOLERANCE_BITS
            minimum_marginal = min(minimum_marginal, float(values.min()))
            if affected.any():
                raise RuntimeError(f"Syn nonnegativity violation: min={values.min():.12g}, "
                                   f"threshold={-TOLERANCE_BITS}, affected_count={affected.sum()}")
            negative_count += int((values < 0).sum())
            maximum_closure = max(maximum_closure, float(np.abs(values.sum(axis=1)-totals).max()))
        if maximum_closure > TOLERANCE_BITS:
            raise RuntimeError(f"Permutation closure failed: {maximum_closure:.12g} bits")
        paired = (forward + backward) / 2
        per_condition.add(paired)
        aggregate.add(paired.mean(axis=0))
        if pair % 256 == 0:
            print(f"[ROI Shapley] pairs={pair}/{max_pairs}, elapsed={perf_counter()-start:.1f}s, "
                  f"max mean MC SE={aggregate.se().max():.6f} bits", flush=True)
        if pair != next_check:
            continue
        if previous_mean is None:
            delta, rank, top_overlap = None, None, None
        else:
            second_half = 2 * aggregate.mean - previous_mean
            delta = float(np.abs(second_half - previous_mean).max())
            rank = float(spearmanr(previous_mean, second_half).statistic)
            k = min(10, n)
            top_overlap = len(set(np.argsort(previous_mean)[-k:]) & set(np.argsort(second_half)[-k:]))
            converged = bool(
                aggregate.se().max() <= mean_se_target and
                per_condition.se().max() <= condition_se_target and
                delta <= split_target and rank >= 0.995 and top_overlap >= max(1, k-1)
            )
        checkpoints.append({
            "independent_pairs": pair, "permutations_per_condition": pair*2,
            "max_mean_mc_se_bits": float(aggregate.se().max()),
            "max_condition_mc_se_bits": float(per_condition.se().max()),
            "split_half_max_delta_bits": delta, "split_half_rank_correlation": rank,
            "split_half_top10_overlap": top_overlap, "converged": converged,
        })
        print(json.dumps(checkpoints[-1]), flush=True)
        if converged:
            break
        previous_mean = aggregate.mean.copy()
        next_check *= 2
    summary = {
        "version": VERSION, "rng_seed": rng_seed, "independent_pairs": pair,
        "permutations_per_condition": 2*pair, "converged": converged,
        "convergence_targets": {"max_mean_mc_se_bits": mean_se_target,
                                "max_condition_mc_se_bits": condition_se_target,
                                "split_half_max_delta_bits": split_target,
                                "split_half_rank_correlation": 0.995,
                                "split_half_top10_overlap": 9},
        "syn_nonnegative_tolerance_bits": TOLERANCE_BITS,
        "significant_nonnegativity_violation_count": 0,
        "tolerance_negative_marginal_count": negative_count,
        "tolerance_negative_handling": "retained raw; no clipping or contribution renormalization",
        "minimum_marginal_bits": minimum_marginal,
        "maximum_permutation_closure_error_bits": maximum_closure,
        "maximum_estimate_closure_error_bits": float(np.abs(per_condition.mean.sum(axis=1)-totals).max()),
        "maximum_mean_mc_se_bits": float(aggregate.se().max()),
        "elapsed_seconds": perf_counter()-start, "checkpoints": checkpoints,
        "game": "conditional total correlation among E/I-paired ROI blocks, fixed full-system target",
        "interpretation": "ordinary ROI Shapley of cross-ROI Xi; not Yeo-constrained; not within-ROI Xi",
        "mc_error": "SE across independent antithetic pairs; aggregate SE accounts for shared permutations across conditions",
    }
    return per_condition.mean, per_condition.se(), aggregate.se(), totals, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/dmf_schaefer100/full/critical_yeo7.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/dmf_schaefer100/roi_shapley/results.npz")
    parser.add_argument("--max-pairs", type=int, default=16384)
    parser.add_argument("--min-pairs", type=int, default=512)
    parser.add_argument("--rng-seed", type=int, default=20260831)
    args = parser.parse_args()
    digest = source_digest(args.input)
    if args.output.exists():
        with np.load(args.output, allow_pickle=False) as archive:
            meta = json.loads(str(archive["summary_json"]))
            if meta["source_sha256"] == digest and meta["version"] == VERSION and meta["rng_seed"] == args.rng_seed and meta["converged"]:
                print("[reuse] " + json.dumps(meta), flush=True)
                return
        raise RuntimeError("Existing cache does not match or is unconverged; use a different output path")
    with np.load(args.input, allow_pickle=True) as z:
        cov = np.asarray(z["conditional_covariance"], dtype=float)
        expected = np.asarray(z["cross_roi"], dtype=float)
        labels, names = z["region_labels"].astype(str), z["network_names"].astype(str)
        membership, seeds, coupling = z["network_membership"], z["seeds"], z["G"]
    with threadpool_limits(limits=1):
        flat = cov.reshape(-1, cov.shape[-2], cov.shape[-1])
        _, _, exact_totals = prepare_game(flat)
        np.testing.assert_allclose(exact_totals.reshape(expected.shape), expected, atol=1e-8, rtol=0)
        values, se, mean_se, totals, summary = estimate(
            flat, rng_seed=args.rng_seed, min_pairs=args.min_pairs, max_pairs=args.max_pairs,
        )
    values, se = values.reshape(*expected.shape, -1), se.reshape(*expected.shape, -1)
    np.testing.assert_allclose(values.sum(axis=-1), expected, atol=1e-8, rtol=0)
    mean = values.mean(axis=(0, 1))
    summary.update(source_sha256=digest, seed_count=len(seeds), coupling_values=coupling.tolist(),
                   mean_cross_roi_xi_bits=float(expected.mean()),
                   top10=[{"roi": str(labels[i]), "shapley_bits": float(mean[i]), "mc_se_bits": float(mean_se[i])}
                          for i in np.argsort(mean)[-10:][::-1]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, roi_shapley_bits=values, condition_mc_se_bits=se, mean_mc_se_bits=mean_se,
        cross_roi=expected, seeds=seeds, G=coupling, region_labels=labels,
        network_names=names, network_membership=membership, summary_json=json.dumps(summary),
    )
    print("[done] " + json.dumps(summary), flush=True)
    if not summary["converged"]:
        raise SystemExit("Sampling budget reached before convergence; do not present as a converged map")


if __name__ == "__main__":
    main()
