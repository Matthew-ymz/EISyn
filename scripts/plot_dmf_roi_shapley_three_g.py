#!/usr/bin/env python3
"""Bounded three-state ordinary ROI Shapley comparison using one affine-TM protocol.

Missing extreme-G conditions can be prepared explicitly; existing conditions are
reused. Each ROI is one E/I-paired player, with the complete future E/I target.
Shared antithetic permutations preserve paired Monte Carlo difference errors.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from itertools import combinations
from pathlib import Path
import sys
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compute_dmf_roi_shapley import prepare_game, permutation_contributions, source_digest
from scripts.dmf_joint_readout import CommonTargetGame, audit_nonnegative
from scripts.dmf_response_benchmark import Diagnostics, fit_affine_joint, simulate_sources, step
from scripts.run_dmf_paired_spt_pilot import paired_sources, noise_seed
from scripts.run_dmf_subject_consistency import atomic_json
from scripts.run_dmf_subject_curve_baselines import atomic_savez
from scripts.brain_surface_plot import _draw_surface
from scripts.plot_dmf_schaefer100_summary import load_schaefer100_surface_map
from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module

G = np.array([0., 1.3, 3.])
SEEDS = np.array([3, 4, 5])
CONTRASTS = [(0, 1), (1, 2), (0, 2)]
TOL = 1e-8  # native nats; never clip information contributions
VERSION = "three-g-affine-tm-roi-shapley-v2"
PILOT = ROOT / "results/dmf_schaefer100/subject_consistency_pilot"
DENSE = ROOT / "results/dmf_schaefer100/subject_curves_93_dense"
RESULT = ROOT / "results/dmf_schaefer100/roi_shapley_three_G"
FIGURE = ROOT / "fig/dmf_schaefer100/roi_shapley_three_G"
SURFACE = ROOT / "results/dmf_schaefer100/schaefer100_fsaverage5_surface.npz"


def simulate_extreme_sources(dmf, x, sc, jf, g, p, seed):
    """Same native equations/RNG; record the 500-Hz diagnostic without stopping.

    Used only by explicit --allow-extreme-rates. Nonfinite state/rate failures
    still stop, and the caller rejects any state outside [0,1]. No state clipping.
    """
    se, si = np.split(x.copy(), 2, axis=1)
    rng = np.random.default_rng(seed)
    diagnostics = Diagnostics()
    for _ in range(300):
        ne = p.sigma * math.sqrt(p.dt) * rng.standard_normal(se.shape)
        ni = p.sigma * math.sqrt(p.dt) * rng.standard_normal(si.shape)
        se, si, re, ri = step(dmf, se, si, sc, jf, g, p, ne, ni)
        try:
            diagnostics.add(se, si, re, ri)
        except ArithmeticError as error:
            if not str(error).startswith("Abnormal DMF rate:"):
                raise
    record = diagnostics.record()
    record.update(rate_diagnostic_threshold_hz=500.,
        abnormal_rate_fraction=record["abnormal_rate_count"] / record["state_count"],
        rate_guard_policy="Record threshold exceedance; explicit extreme-model comparison, not physiological validation")
    return np.concatenate([se, si], axis=1), record


def prepare_condition(path, g, seed, sc, jf, contract, implementation, *, prepare=False,
                      allow_extreme_rates=False):
    """Extend the frozen protocol at one G, with separately identified caches."""
    x = np.concatenate(paired_sources(seed, 2048, 100), 1)
    dmf = load_dmf_module()
    identity = dict(protocol=contract, G=float(g), seed=int(seed),
        source_sha256=hashlib.sha256(x.tobytes()).hexdigest(), noise_seed=noise_seed(seed),
        sc_sha256=hashlib.sha256(sc.tobytes()).hexdigest(),
        jfic_sha256=hashlib.sha256(jf.tobytes()).hexdigest(),
        implementation_sha256=implementation, dmf_source_sha256=source_digest(Path(dmf.__file__)),
        allow_extreme_rates=allow_extreme_rates,
        density="fit_affine_joint: independent uniform-prior Gaussian moments, halfwidth=0.2, ridge=1e-6",
        density_extension_source_sha256=hashlib.sha256((inspect.getsource(prepare_condition) +
            inspect.getsource(simulate_extreme_sources)).encode()).hexdigest())
    if path.exists():
        with np.load(path) as a:
            if json.loads(str(a["extension_identity_json"])) != identity:
                raise ValueError(f"Extension cache mismatch: {path.name}")
        return
    if not prepare:
        raise FileNotFoundError(f"Missing {path}; use --prepare-missing-conditions")
    start = perf_counter()
    p = dmf.DMFParameters(t_total=1, burn_in=0, dt=.001, sigma=.01)
    simulator = simulate_extreme_sources if allow_extreme_rates else simulate_sources
    y, diagnostics = simulator(dmf, x, sc, jf, g, p, noise_seed(seed))
    if diagnostics["outside_state_count"]:
        raise ArithmeticError(f"DMF state violation: G={g}, seed={seed}: {diagnostics}")
    covariance = fit_affine_joint(x, y, .2, ridge=1e-6)
    game = CommonTargetGame(covariance)
    metrics = game.totals()
    audit = audit_nonnegative([*game.audited, *metrics.values()], TOL)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez(path, covariance=covariance, conditional=game.conditional,
        metrics_json=json.dumps(metrics), diagnostics_json=json.dumps(diagnostics),
        density_audit_json=json.dumps(audit), extension_identity_json=json.dumps(identity),
        source_sha256=identity["source_sha256"], noise_seed=noise_seed(seed),
        elapsed_seconds=perf_counter()-start)
    print(f"[prepare] G={g:g}, seed={seed}: Xi={metrics['xi_nats']:.6f} nats; {perf_counter()-start:.1f}s", flush=True)


def load_conditions(pilot=PILOT, dense=DENSE, *, extension_dir=None, prepare_missing=False,
                    allow_extreme_rates=False):
    contract = json.loads((pilot / "contract.json").read_text())
    dense_contract = json.loads((dense / "contract.json").read_text())
    if dense_contract["native_protocol"]["pilot_contract"] != contract:
        raise ValueError("Pilot and dense contracts differ")
    expected = dict(sample_count=2048, support=[.3, .7], horizon_steps=300,
                    dt_seconds=.001, sigma=.01, ridge=1e-6, state_boundary="none",
                    seeds=SEEDS.tolist(), paired_inputs_and_noise_across_G_and_subjects=True)
    for key, value in expected.items():
        if contract[key] != value:
            raise ValueError(f"Frozen protocol mismatch: {key}")
    # The generator consumes two equal-sized noise arrays on each of 300 steps;
    # neither G nor state-dependent branches alter its RNG consumption.
    files = ["scripts/dmf_response_benchmark.py", "scripts/dmf_joint_readout.py",
             "scripts/run_dmf_paired_spt_pilot.py", "exp/TM/transport_map_density.py"]
    for rel in files:
        if source_digest(ROOT / rel) != dense_contract["implementation_sha256"][rel]:
            raise ValueError(f"Authoritative implementation changed: {rel}")
    source = ROOT / "results/dmf_schaefer100/source/group_mean_native_mean_rate.npz"
    if source_digest(source) != contract["reference_source_sha256"]:
        raise ValueError("Mean SC/JFIC source changed")
    with np.load(pilot / "inputs.npz") as a:
        if json.loads(str(a["contract_json"])) != contract:
            raise ValueError("Frozen input contract mismatch")
        labels = a["labels"].astype(str)
        membership, networks = a["network_membership"].copy(), a["network_names"].astype(str)
        index = a["subject_ids"].tolist().index("group_mean_93")
        sc, jf = a["connectivity"][index].copy(), a["j_fic"].copy()
    with np.load(source) as a:
        np.testing.assert_allclose(sc, a["connectivity"], atol=0, rtol=0)
        if not np.array_equal(a["j_fic"], np.broadcast_to(jf, a["j_fic"].shape)):
            raise ValueError("JFIC differs from frozen reference")
    if len(labels) != 100 or len(set(labels)) != 100:
        raise ValueError("Expected 100 unique ROI labels")
    if labels.tolist() != (ROOT / "results/dmf_schaefer100/schaefer100_labels.txt").read_text().splitlines():
        raise ValueError("ROI order differs from original atlas preparation")
    sources = {int(s): hashlib.sha256(np.concatenate(paired_sources(s, 2048, 100), 1).tobytes()).hexdigest()
               for s in SEEDS}
    covariances, local, totals, provenance = [], [], [], []
    for g in G:
        for seed in SEEDS:
            path = pilot / "conditions" / f"group_mean_93_G{g:.2f}_seed{seed}.npz"
            comparison = dense / "conditions" / path.name
            if not path.exists():
                if extension_dir is None:
                    raise FileNotFoundError(f"No extension directory for G={g}")
                path = extension_dir / path.name
                prepare_condition(path, g, int(seed), sc, jf, contract,
                    {rel: source_digest(ROOT / rel) for rel in files}, prepare=prepare_missing,
                    allow_extreme_rates=allow_extreme_rates)
                with np.load(path) as a:
                    game = CommonTargetGame(a["covariance"])
                    np.testing.assert_allclose(game.conditional, a["conditional"], atol=1e-12, rtol=0)
                    exact = game.totals()
                    metrics = json.loads(str(a["metrics_json"]))
                    for key in exact:
                        np.testing.assert_allclose(exact[key], metrics[key], atol=TOL, rtol=0)
                    audit = audit_nonnegative([*game.audited, *exact.values()], TOL)
                    covariances.append(game.conditional)
                    local.append(game.local_xi)
                    totals.append([exact[k] for k in ["xi_nats", "roi_local_xi_nats", "cross_roi_nats"]])
                    provenance.append(dict(G=float(g), seed=int(seed), cache=str(path.relative_to(ROOT)),
                        cache_sha256=source_digest(path), dense_cache_sha256=None,
                        origin="Frozen protocol extension; not part of the old dense sweep",
                        source_sha256=sources[int(seed)], noise_seed=noise_seed(seed), density_audit=audit,
                        diagnostics=json.loads(str(a["diagnostics_json"]))))
                continue
            with np.load(path) as a, np.load(comparison) as b:
                if json.loads(str(a["contract_json"])) != contract:
                    raise ValueError(f"Condition contract mismatch: {path.name}")
                if str(b["contract_sha256"]) != source_digest(dense / "contract.json"):
                    raise ValueError("Dense condition contract mismatch")
                if str(a["source_sha256"]) != sources[int(seed)] or str(b["intervention_source_sha256"]) != sources[int(seed)]:
                    raise ValueError("Actual intervention fingerprint mismatch")
                if int(a["noise_seed"]) != noise_seed(seed) or not bool(b["pilot_cache_reused"]):
                    raise ValueError("Paired noise or dense-cache provenance mismatch")
                if str(b["subject"]) != "group_mean_93" or float(b["G"]) != g or int(b["seed"]) != seed:
                    raise ValueError("Dense condition identity mismatch")
                game = CommonTargetGame(a["covariance"])
                np.testing.assert_allclose(game.conditional, a["conditional"], atol=1e-12, rtol=0)
                metrics, dense_metrics = json.loads(str(a["metrics_json"])), json.loads(str(b["metrics_json"]))
                exact = game.totals()
                for key in exact:
                    np.testing.assert_allclose(exact[key], metrics[key], atol=TOL, rtol=0)
                    np.testing.assert_allclose(exact[key], dense_metrics[key], atol=TOL, rtol=0)
                audit = audit_nonnegative([*game.audited, *exact.values()], TOL)
                covariances.append(game.conditional)
                local.append(game.local_xi)
                totals.append([exact[k] for k in ["xi_nats", "roi_local_xi_nats", "cross_roi_nats"]])
                provenance.append(dict(G=float(g), seed=int(seed), cache=str(path.relative_to(ROOT)),
                    cache_sha256=source_digest(path), dense_cache_sha256=source_digest(comparison),
                    source_sha256=sources[int(seed)], noise_seed=noise_seed(seed), density_audit=audit))
    return (np.array(covariances), np.array(local).reshape(3, 3, 100),
            np.array(totals).reshape(3, 3, 3), labels, membership, networks,
            dict(protocol=contract, conditions=provenance,
                 pairing_evidence="Actual source SHA256 matches regenerated arrays; old conditions also match dense caches. Extensions use the same frozen SC/JFIC, source arrays, noise seed, and hash-matched 300-step simulation. Raw noise was not archived.",
                 roi_label_status=contract["roi_label_status"],
                 source_sha256=source_digest(source), input_sha256=source_digest(pilot / "inputs.npz")))


def sampling_diagnostics(samples_nats, xi):
    """SE across independent pairs, preserving shared-permutation covariance."""
    count = len(samples_nats)
    pp = 100 * samples_nats / xi[None, ..., None]
    means = pp.mean(axis=2)  # equal-weight seeds AFTER per-condition division
    condition_se = pp.std(axis=0, ddof=1) / np.sqrt(count)
    mean_se = means.std(axis=0, ddof=1) / np.sqrt(count)
    half = count // 2
    first, second = means[:half].mean(0), means[half:].mean(0)
    delta = np.abs(first - second)
    record = dict(independent_pairs=count, permutations_per_condition=2 * count,
                  maximum_mean_mc_se_pp=float(mean_se.max()),
                  maximum_condition_mc_se_pp=float(condition_se.max()),
                  split_half_maximum_mean_difference_pp=float(delta.max()),
                  by_G=[dict(G=float(g), maximum_mean_mc_se_pp=float(mean_se[i].max()),
                      maximum_condition_mc_se_pp=float(condition_se[i].max()),
                      split_half_maximum_mean_difference_pp=float(delta[i].max()),
                      split_half_spearman=(float(spearmanr(first[i], second[i]).statistic)
                          if np.ptp(first[i]) > 0 and np.ptp(second[i]) > 0 else None),
                      split_half_top10_overlap=len(set(np.argsort(first[i])[-10:]) & set(np.argsort(second[i])[-10:])))
                      for i, g in enumerate(G)])
    return pp, condition_se, mean_se, record


def estimate(covariances, xi, local, *, rng_seed=20261002, min_pairs=512, max_pairs=16384):
    if min_pairs < 2 or max_pairs < min_pairs or max_pairs % min_pairs:
        raise ValueError("Require power-of-two max/min ratio, min >= 2")
    ratio = max_pairs // min_pairs
    if ratio & (ratio - 1) or min_pairs % 2:
        raise ValueError("Require power-of-two max/min ratio and even min_pairs")
    c, scalar_logs, exact_bits = prepare_game(covariances)
    np.testing.assert_allclose(exact_bits.reshape(xi.shape) * np.log(2), xi, atol=TOL, rtol=0)
    audit_nonnegative(xi, TOL)
    audit_nonnegative(local, TOL)
    samples = np.empty((max_pairs, *local.shape))
    rng = np.random.default_rng(rng_seed)
    checkpoints, next_check = [], min_pairs
    minimum, cross_minimum, negatives, cross_negatives, closure = np.inf, np.inf, 0, 0, 0.
    start = perf_counter()
    for draw in range(max_pairs):
        order = rng.permutation(local.shape[-1])
        terms = []
        for p in (order, order[::-1]):
            value = (permutation_contributions(c, scalar_logs, p) * np.log(2)).reshape(local.shape)
            audit = audit_nonnegative(value, TOL)
            cross_audit = audit_nonnegative(value - local, TOL)
            minimum, cross_minimum = min(minimum, audit["minimum_nats"]), min(cross_minimum, cross_audit["minimum_nats"])
            negatives += audit["tolerance_negative_count"]
            cross_negatives += cross_audit["tolerance_negative_count"]
            closure = max(closure, float(np.abs(value.sum(-1) - xi).max()))
            if closure > TOL:
                raise ArithmeticError(f"Permutation closure failure: {closure} nats")
            terms.append(value)
        samples[draw] = .5 * (terms[0] + terms[1])
        count = draw + 1
        if count != next_check:
            continue
        _, _, _, record = sampling_diagnostics(samples[:count], xi)
        record["converged"] = bool(record["maximum_mean_mc_se_pp"] <= .03 and
                                   record["maximum_condition_mc_se_pp"] <= .05 and
                                   record["split_half_maximum_mean_difference_pp"] <= .10)
        record["elapsed_seconds"] = perf_counter() - start
        checkpoints.append(record)
        print(json.dumps(record), flush=True)
        if record["converged"]:
            break
        next_check *= 2
    samples = samples[:count]
    mean = samples.mean(0)
    audit_nonnegative(mean, TOL)
    audit_nonnegative(mean - local, TOL)
    estimate_closure = float(np.abs(mean.sum(-1) - xi).max())
    if estimate_closure > TOL:
        raise ArithmeticError(f"Estimate closure failure: {estimate_closure} nats")
    summary = dict(converged=checkpoints[-1]["converged"], independent_pairs=count,
        permutations_per_condition=2 * count, rng_seed=rng_seed, elapsed_seconds=perf_counter()-start,
        targets=dict(maximum_mean_mc_se_pp=.03, maximum_condition_mc_se_pp=.05,
                     split_half_maximum_mean_difference_pp=.10), checkpoints=checkpoints,
        tolerance_nats=TOL, minimum_marginal_nats=minimum, minimum_cross_marginal_nats=cross_minimum,
        tolerance_negative_marginal_count=negatives, tolerance_negative_cross_marginal_count=cross_negatives,
        significant_nonnegativity_violation_count=0,
        negative_handling="Retain all raw values; no clipping, projection or contribution renormalization",
        maximum_permutation_closure_error_nats=closure, maximum_estimate_closure_error_nats=estimate_closure,
        mc_error="Independent antithetic pairs; shared permutations across all nine conditions. Ranking is diagnostic only, not a stopping criterion.")
    return samples, summary


def comparison_statistics(pp, labels):
    mean = pp.mean(0)
    averaged = mean.mean(1)
    contrasts, within = [], []
    for a, b in CONTRASTS:
        distances = .5 * np.abs(mean[b] - mean[a]).sum(-1)
        ranks = [float(spearmanr(x, y).statistic) for x, y in zip(mean[a], mean[b])]
        delta = mean[b] - mean[a]
        agree = np.all(delta > 0, axis=0) | np.all(delta < 0, axis=0)
        overlap = [len(set(np.argsort(x)[-10:]) & set(np.argsort(y)[-10:])) for x, y in zip(mean[a], mean[b])]
        contrasts.append(dict(G_from=float(G[a]), G_to=float(G[b]),
            total_variation_pp_by_seed=distances.tolist(), total_variation_pp_mean=float(distances.mean()),
            total_variation_pp_range=[float(distances.min()), float(distances.max())],
            spearman_by_seed=ranks, mean_map_spearman=float(spearmanr(averaged[a], averaged[b]).statistic),
            mean_map_total_variation_pp=float(.5 * np.abs(averaged[b] - averaged[a]).sum()),
            top10_overlap_by_seed=overlap,
            same_direction_roi_count=int(agree.sum()),
            paired_mean_delta_mc_se_pp_max=float((pp[:, b].mean(1) - pp[:, a].mean(1)).std(0, ddof=1).max() / np.sqrt(len(pp)))))
    for i, g in enumerate(G):
        ds = [.5 * np.abs(mean[i, b] - mean[i, a]).sum() for a, b in combinations(range(3), 2)]
        rs = [float(spearmanr(mean[i, a], mean[i, b]).statistic) for a, b in combinations(range(3), 2)]
        within.append(dict(G=float(g), total_variation_pp_by_seed_pair=list(map(float, ds)),
            total_variation_pp_mean=float(np.mean(ds)), spearman_by_seed_pair=rs,
            spearman_mean=float(np.mean(rs)),
            share_pp_range=[float(averaged[i].min()), float(averaged[i].max())],
            top10=[dict(roi=str(labels[j]), share_percent=float(averaged[i, j]))
                   for j in np.argsort(averaged[i])[-10:][::-1]]))
    return dict(contrasts=contrasts, same_G_seed_variability=within,
        inference="Descriptive paired simulation-seed comparisons, n=3. ROI and MC draws are not independent biological replicates.")


def surface_plate(values, labels, row_labels, annotations, output, *, asset=SURFACE,
                  colorbar_label, cmap="viridis", diverging=False, footer="", reference_vmax=None,
                  observed_range=False):
    if observed_range and (diverging or reference_vmax is not None):
        raise ValueError("Observed range requires a sequential scale without a historical maximum")
    nrows = len(values)
    low, high = (0., float(values.max()))
    if diverging:
        high = float(np.abs(values).max())
        low = -high
    elif values.min() < 0:
        raise ArithmeticError("Negative raw attribution cannot use zero-based sequential scale")
    elif observed_range:
        # One range across all displayed seed-mean maps uses the full color band
        # without per-row rescaling, quantile clipping, or changing attribution.
        low = float(values.min())
    elif reference_vmax is not None:
        high = max(high, float(reference_vmax))
    norm = Normalize(low, high)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"}):
        fig = plt.figure(figsize=(12.4, .6 + 2.3 * nrows + .9), facecolor="white")
        # Explicit equal slots and fixed orthographic cameras; guides outside data.
        left, right, top, bottom = .145, .99, .93, .18
        width = (right - left) / 4
        height = (top - bottom) / nrows
        view_names = ["Left lateral", "Right lateral", "Left medial", "Right medial"]
        for j, name in enumerate(view_names):
            fig.text(left + (j + .5) * width, .965, name, ha="center", va="center", fontsize=10)
        for i, row in enumerate(values):
            y = top - (i + 1) * height
            lm, rm, lv, rv = load_schaefer100_surface_map(asset, labels, row)
            for j, (mesh, data, azim) in enumerate([(lm, lv, 180.), (rm, rv, 0.), (lm, lv, 0.), (rm, rv, 180.)]):
                ax = fig.add_axes([left + j * width, y, width, height], projection="3d")
                _draw_surface(ax, mesh, data, cmap=matplotlib.colormaps[cmap], norm=norm,
                              elev=0., azim=azim, background_color="#D7D7D7", zoom=1.50)
                # Rasterize only mesh faces in PDF; labels/colorbar remain vectors.
                for collection in ax.collections:
                    collection.set_rasterized(True)
            center = y + height / 2
            fig.text(.018, center + .040, row_labels[i], fontsize=12, weight="bold", va="center")
            fig.text(.018, center - .008, annotations[i], fontsize=9, va="top", linespacing=1.55)
        cax = fig.add_axes([.315, .12, .52, .018])
        bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        if observed_range:
            bar.set_ticks(np.linspace(low, high, 5))
            bar.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        bar.outline.set_linewidth(.6)
        bar.ax.tick_params(labelsize=9, length=3, width=.6)
        bar.set_label(colorbar_label, fontsize=10, labelpad=4)
        fig.text(.5, .025, footer, ha="center", fontsize=8.5, color=".35")
        output.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".pdf"):
            fig.savefig(output.with_suffix(suffix), dpi=300, bbox_inches=None)
        plt.close(fig)
    return dict(vmin=low, vmax=high, cmap=cmap, values_clipped=0,
                reference_vmax=reference_vmax,
                limits_policy=("Shared observed ROI range across all displayed rows" if observed_range else
                               "Shared symmetric range" if diverging else "Shared zero-based range"),
                normalization="linear",
                camera="LH lateral 180, RH lateral 0, LH medial 0, RH medial 180; elev=0; orthographic; zoom=1.50",
                mapping="Exact parcel-name match; 100 unique labels; medial wall gray; original SC row order remains inferred")


def main():
    global G
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strong-g", type=float, default=3.)
    p.add_argument("--prepare-missing-conditions", action="store_true")
    p.add_argument("--allow-extreme-rates", action="store_true",
                   help="Record rather than stop at the 500-Hz rate diagnostic in new extreme-model conditions")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--figure-dir", type=Path)
    p.add_argument("--scale-reference", type=Path,
                   help="Retain an earlier summary's absolute nats scale; percentages use the current observed ROI range")
    p.add_argument("--max-pairs", type=int, default=16384)
    p.add_argument("--min-pairs", type=int, default=512)
    p.add_argument("--rng-seed", type=int, default=20261002)
    args = p.parse_args()
    if not np.isfinite(args.strong_g) or args.strong_g <= 1.3:
        p.error("--strong-g must be finite and greater than 1.3")
    G = np.array([0., 1.3, args.strong_g])
    suffix = "" if args.strong_g == 3. else f"_G{round(100 * args.strong_g):04d}"
    args.output_dir = args.output_dir or RESULT.with_name(RESULT.name + suffix)
    args.figure_dir = args.figure_dir or FIGURE.with_name(FIGURE.name + suffix)
    args.output_dir, args.figure_dir = args.output_dir.resolve(), args.figure_dir.resolve()
    if args.scale_reference:
        args.scale_reference = args.scale_reference.resolve()
    reference = json.loads(args.scale_reference.read_text()) if args.scale_reference else None
    with threadpool_limits(limits=1):
        cov, local, totals, labels, membership, networks, provenance = load_conditions(
            extension_dir=args.output_dir / "conditions", prepare_missing=args.prepare_missing_conditions,
            allow_extreme_rates=args.allow_extreme_rates)
        identity = dict(version=VERSION, G=G.tolist(), seeds=SEEDS.tolist(), provenance=provenance,
                        rng_seed=args.rng_seed, min_pairs=args.min_pairs, max_pairs=args.max_pairs,
                        estimator_source_sha256=hashlib.sha256("\n".join(inspect.getsource(f)
                            for f in (simulate_extreme_sources, prepare_condition, load_conditions,
                                      estimate, sampling_diagnostics)).encode()).hexdigest(),
                        prefix_source_sha256=source_digest(ROOT / "scripts/compute_dmf_roi_shapley.py"))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        cache = args.output_dir / "roi_shapley.npz"
        if cache.exists():
            with np.load(cache) as a:
                if json.loads(str(a["identity_json"])) != identity:
                    raise ValueError("Existing attribution cache provenance/budget mismatch; use a different output directory")
                samples, sampling = a["paired_samples_nats"].copy(), json.loads(str(a["sampling_json"]))
                print(f"[reuse] {len(samples)} independent pairs", flush=True)
        else:
            samples, sampling = estimate(cov, totals[..., 0], local, rng_seed=args.rng_seed,
                                          min_pairs=args.min_pairs, max_pairs=args.max_pairs)
            atomic_savez(cache, paired_samples_nats=samples, roi_shapley_nats=samples.mean(0),
                xi_nats=totals[..., 0], roi_local_xi_nats=local, G=G, seeds=SEEDS, region_labels=labels,
                network_membership=membership, network_names=networks,
                sampling_json=json.dumps(sampling), identity_json=json.dumps(identity))
        pp, condition_se, mean_se, diagnostics = sampling_diagnostics(samples, totals[..., 0])
        shares = pp.mean(0)
        if np.abs(shares.sum(-1) - 100).max() > 1e-8:
            raise ArithmeticError("Percentage budget does not close")
        audit_nonnegative(samples.mean(0), TOL)
        cross = samples.mean(0) - local
        audit_nonnegative(cross, TOL)
        analysis = comparison_statistics(pp, labels)
        delta_samples = np.array([pp[:, b] - pp[:, a] for a, b in CONTRASTS])
        atomic_savez(args.output_dir / "roi_tables.npz", G=G, seeds=SEEDS, region_labels=labels,
            roi_shapley_nats=samples.mean(0), share_percent=shares, mean_share_percent=shares.mean(1),
            condition_mc_se_pp=condition_se, mean_mc_se_pp=mean_se,
            roi_local_xi_nats=local, roi_cross_shapley_nats=cross, xi_nats=totals[..., 0],
            condition_paired_difference_mc_se_pp=delta_samples.std(1, ddof=1) / np.sqrt(len(samples)),
            mean_paired_difference_mc_se_pp=delta_samples.mean(2).std(1, ddof=1) / np.sqrt(len(samples)),
            contrast_G=np.array([(G[a], G[b]) for a, b in CONTRASTS]))
        summary = dict(**identity, sampling=sampling, diagnostics=diagnostics, analysis=analysis,
            normalization="100 * raw phi_i,s(G) / Xi_s(G), then equal-weight mean across seeds; no renormalization",
            nonnegative_audits=dict(xi=audit_nonnegative(totals[..., 0], TOL), local=audit_nonnegative(local, TOL),
                attribution=audit_nonnegative(samples.mean(0), TOL), cross_attribution=audit_nonnegative(cross, TOL)),
            state_totals=[dict(G=float(g), xi_mean_nats=float(totals[i, :, 0].mean()),
                xi_seed_sd_nats=float(totals[i, :, 0].std(ddof=1)), local_mean_nats=float(totals[i, :, 1].mean()),
                cross_mean_nats=float(totals[i, :, 2].mean())) for i, g in enumerate(G)],
            manuscript_recheck=dict(parent="P6UJCVG8", attachment="DXGC7JEA", pages=19,
                supplementary_attachment="MWIWKSVG", supplementary_pages=28,
                checked_date="2026-10-06", version_date=None,
                locations="Brain/Fig.2 pp.6-7; Methods Eqs.5-12 pp.15-17; SI S1.2-S1.3 pp.3-5, S12.2.1 p.23",
                limitation="One body and one SI attachment; neither supplies an explicit manuscript date/revision. Metadata timestamps do not establish a version.",
                consistency="Fixed target, factorized source and scalar-fine Xi definitions checked. Affine Gaussian-moment protocol is an approximation to uniform EI; no feature lift or S14 finite-sample MI correction is applied here. Old Fig.2e averages 24 empirical-Gaussian conditions, not these nine affine-TM conditions."))
        nats_by_condition = samples.mean(0)
        nats_mean = nats_by_condition.mean(1)
        summary["absolute_contrasts"] = [dict(G_from=float(G[a]), G_to=float(G[b]),
            mean_increased_roi_count=int((nats_mean[b] > nats_mean[a]).sum()),
            mean_decreased_roi_count=int((nats_mean[b] < nats_mean[a]).sum()),
            all_seed_increased_roi_count=int(np.all(nats_by_condition[b] > nats_by_condition[a], axis=0).sum()),
            all_seed_decreased_roi_count=int(np.all(nats_by_condition[b] < nats_by_condition[a], axis=0).sum()))
            for a, b in CONTRASTS]
    state_labels = [rf"$G={g:g}$" for g in G]
    state_notes = [f"{name}\n$\\Xi$ = {row['xi_mean_nats']:.2f}\n$\\pm$ {row['xi_seed_sd_nats']:.2f} nats"
                   for name, row in zip(["No long-range\ncoupling", "Peak", "Extreme coupling" if args.strong_g >= 10 else "High coupling"], summary["state_totals"])]
    preview = "" if sampling["converged"] else "preview_"
    summary["surface_sha256"] = source_digest(SURFACE)
    summary["plot_source_sha256"] = source_digest(Path(__file__))
    summary["scale_reference"] = (dict(path=str(args.scale_reference),
        sha256=source_digest(args.scale_reference), applies_to="absolute_plot") if args.scale_reference else None)
    summary["plot"] = surface_plate(shares.mean(1), labels, state_labels, state_notes,
        args.figure_dir / f"{preview}roi_share_three_G", colorbar_label=r"ROI attribution to overall $\Xi$ (%)",
        observed_range=True,
        footer="Mean SC model · seeds 3, 4, 5 · each row sums to 100% · total Ξ: mean ± seed SD" +
               (" · MC budget exhausted: preview" if preview else ""))
    summary["absolute_plot"] = surface_plate(samples.mean((0, 2)), labels, state_labels, state_notes,
        args.figure_dir / f"{preview}roi_nats_three_G", colorbar_label=r"ROI attribution to overall $\Xi$ (nats)",
        reference_vmax=reference["absolute_plot"]["vmax"] if reference else None,
        footer="Mean SC model · seeds 3, 4, 5 · absolute Shapley attribution · shared scale")
    delta = np.array([shares[b].mean(0) - shares[a].mean(0) for a, b in CONTRASTS[:2]])
    summary["difference_plot"] = surface_plate(delta, labels,
        [rf"${G[b]:g} - {G[a]:g}$" for a, b in CONTRASTS[:2]], ["Change in share", "Change in share"],
        args.figure_dir / f"{preview}roi_share_difference", cmap="RdBu_r", diverging=True,
        colorbar_label=r"Change in ROI attribution to overall $\Xi$ (percentage points)",
        footer="Red: increased share · blue: decreased share · both contrasts use the same scale")
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(dict(converged=sampling["converged"], pairs=sampling["independent_pairs"],
        state_totals=summary["state_totals"], contrasts=analysis["contrasts"])), flush=True)
    if not sampling["converged"]:
        raise SystemExit("Budget exhausted; retained preview and diagnostics, no automatic expansion")


if __name__ == "__main__":
    main()
