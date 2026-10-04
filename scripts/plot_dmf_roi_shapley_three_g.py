#!/usr/bin/env python3
"""Bounded three-state ordinary ROI Shapley comparison from frozen affine-TM caches.

No dynamics, fitting, SPT search or population analysis is performed. Each ROI is
one E/I-paired player, with the complete future E/I state as a fixed target.
Shared antithetic permutations preserve paired Monte Carlo difference errors.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
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
from scripts.run_dmf_paired_spt_pilot import paired_sources, noise_seed
from scripts.run_dmf_subject_consistency import atomic_json
from scripts.run_dmf_subject_curve_baselines import atomic_savez
from scripts.brain_surface_plot import _draw_surface
from scripts.plot_dmf_schaefer100_summary import load_schaefer100_surface_map

G = np.array([0., 1.3, 3.])
SEEDS = np.array([3, 4, 5])
CONTRASTS = [(0, 1), (1, 2), (0, 2)]
TOL = 1e-8  # native nats; never clip information contributions
VERSION = "three-g-affine-tm-roi-shapley-v1"
PILOT = ROOT / "results/dmf_schaefer100/subject_consistency_pilot"
DENSE = ROOT / "results/dmf_schaefer100/subject_curves_93_dense"
RESULT = ROOT / "results/dmf_schaefer100/roi_shapley_three_G"
FIGURE = ROOT / "fig/dmf_schaefer100/roi_shapley_three_G"
SURFACE = ROOT / "results/dmf_schaefer100/schaefer100_fsaverage5_surface.npz"


def load_conditions(pilot=PILOT, dense=DENSE):
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
                 pairing_evidence="Actual source SHA256 matches regenerated arrays and dense caches. Noise seed, shapes and 300-step fixed RNG consumption verified in hash-matched simulation code; raw noise was not archived.",
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
                  colorbar_label, cmap="viridis", diverging=False, footer=""):
    nrows = len(values)
    low, high = (0., float(values.max()))
    if diverging:
        high = float(np.abs(values).max())
        low = -high
    elif values.min() < 0:
        raise ArithmeticError("Negative raw attribution cannot use zero-based sequential scale")
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
        bar.outline.set_linewidth(.6)
        bar.ax.tick_params(labelsize=9, length=3, width=.6)
        bar.set_label(colorbar_label, fontsize=10, labelpad=4)
        fig.text(.5, .025, footer, ha="center", fontsize=8.5, color=".35")
        output.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".pdf"):
            fig.savefig(output.with_suffix(suffix), dpi=300, bbox_inches=None)
        plt.close(fig)
    return dict(vmin=low, vmax=high, cmap=cmap, values_clipped=0,
                camera="LH lateral 180, RH lateral 0, LH medial 0, RH medial 180; elev=0; orthographic; zoom=1.50",
                mapping="Exact parcel-name match; 100 unique labels; medial wall gray; original SC row order remains inferred")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=RESULT)
    p.add_argument("--figure-dir", type=Path, default=FIGURE)
    p.add_argument("--max-pairs", type=int, default=16384)
    p.add_argument("--min-pairs", type=int, default=512)
    p.add_argument("--rng-seed", type=int, default=20261002)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        cov, local, totals, labels, membership, networks, provenance = load_conditions()
        identity = dict(version=VERSION, G=G.tolist(), seeds=SEEDS.tolist(), provenance=provenance,
                        rng_seed=args.rng_seed, min_pairs=args.min_pairs, max_pairs=args.max_pairs,
                        estimator_source_sha256=hashlib.sha256("\n".join(inspect.getsource(f)
                            for f in (load_conditions, estimate, sampling_diagnostics)).encode()).hexdigest(),
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
                checked_date="2026-10-04", version_date=None, locations="Brain/Fig.2 pp.6-7; Methods Eqs.5-12 pp.15-17",
                limitation="Only one indexed body attachment; no explicit manuscript date/version; supplementary appendices unavailable",
                consistency="Fixed target, factorized source and scalar-fine Xi definitions match visible text. Affine/Gaussian moment approximation differs from exact uniform EI; old Fig.2e averages 24 empirical-Gaussian conditions, not these nine affine-TM conditions."))
        nats_by_condition = samples.mean(0)
        nats_mean = nats_by_condition.mean(1)
        summary["absolute_contrasts"] = [dict(G_from=float(G[a]), G_to=float(G[b]),
            mean_increased_roi_count=int((nats_mean[b] > nats_mean[a]).sum()),
            mean_decreased_roi_count=int((nats_mean[b] < nats_mean[a]).sum()),
            all_seed_increased_roi_count=int(np.all(nats_by_condition[b] > nats_by_condition[a], axis=0).sum()),
            all_seed_decreased_roi_count=int(np.all(nats_by_condition[b] < nats_by_condition[a], axis=0).sum()))
            for a, b in CONTRASTS]
    state_labels = [r"$G=0$", r"$G=1.3$", r"$G=3$"]
    state_notes = [f"{name}\n$\\Xi$ = {row['xi_mean_nats']:.2f}\n$\\pm$ {row['xi_seed_sd_nats']:.2f} nats"
                   for name, row in zip(["No long-range\ncoupling", "Peak", "High coupling"], summary["state_totals"])]
    preview = "" if sampling["converged"] else "preview_"
    summary["surface_sha256"] = source_digest(SURFACE)
    summary["plot_source_sha256"] = source_digest(Path(__file__))
    summary["plot"] = surface_plate(shares.mean(1), labels, state_labels, state_notes,
        args.figure_dir / f"{preview}roi_share_three_G", colorbar_label=r"ROI attribution to overall $\Xi$ (%)",
        footer="Mean SC model · seeds 3, 4, 5 · each row sums to 100% · total Ξ: mean ± seed SD" +
               (" · MC budget exhausted: preview" if preview else ""))
    summary["absolute_plot"] = surface_plate(samples.mean((0, 2)), labels, state_labels, state_notes,
        args.figure_dir / f"{preview}roi_nats_three_G", colorbar_label=r"ROI attribution to overall $\Xi$ (nats)",
        footer="Mean SC model · seeds 3, 4, 5 · absolute Shapley attribution · shared scale")
    delta = np.array([shares[b].mean(0) - shares[a].mean(0) for a, b in CONTRASTS[:2]])
    summary["difference_plot"] = surface_plate(delta, labels,
        [r"$1.3 - 0$", r"$3 - 1.3$"], ["Change in share", "Change in share"],
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
