#!/usr/bin/env python3
"""Finish locked 2023 exploration: full SHAP, independently tuned nulls and T23.

Requires a passed, provenance-matching --stage core run. No 2024 confirmation.
All selection (including single checkpoints and nulls) precedes T23 prediction.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import train_nyc_taxi_mgstn as training
from scripts import compute_nyc_taxi_mgstn_ei as models
from scripts.nyc_taxi_information_priors import penalty_weights, VERSION
from scripts.run_nyc_taxi_information_prior_calibration import (
    OUTPUT, SEEDS, LAMBDAS, GAMMAS, array_hash, cached_arrays, file_hash, grouped_shap,
    split_indices, check_splits, make_design, fit_calibrator, metrics, predict, tune_method,
)
from scripts.run_unicm_synergy_guided_forecast import circular_block_indices

COMPETITORS = ("phi_r", "phi_wms", "surd", "o_shapley", "correlation", "grouped_shap")
PAIRING_KEYS = ("xi_shapley", "ei_shapley", "pair_only")
FULL_VERSION = "taxi-calibration-2023-exploration-v1"


def normalized_scores(raw):
    total = raw.sum(axis=-1, keepdims=True)
    return np.divide(raw, total, out=np.full_like(raw, 1 / 3), where=total > 0)


def shap_precision(initial, doubled):
    a, b = normalized_scores(initial), normalized_scores(doubled)
    gaps_a, gaps_b = np.diff(np.sort(a, axis=-1)[..., -2:], axis=-1)[..., 0], np.diff(np.sort(b, axis=-1)[..., -2:], axis=-1)[..., 0]
    untied = (gaps_a > 1e-8) & (gaps_b > 1e-8)
    changed = (a.argmax(axis=-1) != b.argmax(axis=-1)) & untied
    count = int(untied.sum())
    fraction = float(changed.sum() / count) if count else 0.
    maximum = float(np.max(np.abs(a - b)))
    return dict(maximum_absolute_weight_change=maximum, untied_zones=count,
                top_rank_changed_zones=int(changed.sum()), top_rank_change_fraction=fraction,
                converged=bool(maximum <= .05 and fraction <= .10),
                thresholds=dict(maximum_weight_change=.05, top_rank_change_fraction=.10, normalized_tie=1e-8))


def shap_draws(splits):
    rng = np.random.default_rng(SEEDS["shap"])
    fg = rng.choice(splits["F"], size=32, replace=False)
    bg = rng.choice(np.setdiff1d(splits["P"], fg), size=16, replace=False)
    # Nested precision samples, with all foreground/background records disjoint.
    extra_fg = rng.choice(np.setdiff1d(splits["F"], np.r_[fg, bg]), size=32, replace=False)
    doubled_fg = np.r_[fg, extra_fg]
    extra_bg = rng.choice(np.setdiff1d(splits["P"], np.r_[doubled_fg, bg]), size=16, replace=False)
    return (fg, bg), (doubled_fg, np.r_[bg, extra_bg])


def compute_shap_with_cache(model, prepared, fg, bg, zone, device, batch_size, path, provenance, origins):
    """Persist the actual costly branch representations, not just final scores."""
    def compute():
        captures = {m: [] for m in ("recent", "daily", "weekly")}
        target = []
        hooks = []
        for name in captures:
            def hook(_module, _input, output, name=name):
                captures[name].append(output[:, zone].detach().cpu().numpy())
            hooks.append(getattr(model, name).register_forward_hook(hook))
        hooks.append(model.target_attribute.register_forward_hook(
            lambda _module, _input, output: target.append(output.detach().cpu().numpy())))
        try:
            scores, _ = grouped_shap(model, prepared, fg, bg, [zone], device, batch_size, verify=False)
        finally:
            for hook in hooks:
                hook.remove()
        return dict(raw_scores=scores[0], foreground=fg, background=bg, target_representation=target[0],
                    **{f"branch_{m}_baseline_then_hidden": np.concatenate(parts) for m, parts in captures.items()})
    return cached_arrays(path, provenance, compute, origins)["raw_scores"]


def fit_selected(design, prediction, target, fit_n, raw, method, selection):
    weights, _ = penalty_weights(raw, method, selection["gamma"])
    return fit_calibrator(design, prediction, target, fit_n, weights,
                          selection["lambda_value"], method == "univariate")[0]


def holm(p_values):
    values = np.asarray(p_values, float)
    order = np.argsort(values, kind="stable")
    adjusted = np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    out = np.empty_like(adjusted)
    out[order] = np.minimum(1, adjusted)
    return out


def bootstrap_comparisons(predictions, target_z, mean, scale, primary):
    """Paired 7-day circular time blocks, 4000 draws; descriptive T23 only."""
    count, n = 4000, len(target_z)
    rng = np.random.default_rng(SEEDS["bootstrap"])
    weights = np.empty((count, n), dtype=np.int16)
    for b in range(count):
        indices = circular_block_indices(rng, n, 7 * 24)
        weights[b] = np.bincount(indices, minlength=n)
    samples, point = {}, {}
    for method, z in predictions.items():
        error = (np.maximum(z * scale + mean, 0.) - (target_z * scale + mean)) / scale
        squared = error.reshape(n, -1) ** 2
        point[method] = float(np.sqrt(squared.mean(axis=0)).mean())
        samples[method] = np.sqrt(weights.astype(float) @ squared / n).mean(axis=1)
    comparisons = {}
    xi = "xi_shapley"
    for method in predictions:
        if method == xi:
            continue
        delta = point[method] - point[xi]
        boot_delta = samples[method] - samples[xi]
        gains = 100 * boot_delta / samples[method]
        centered = boot_delta - delta
        p = float((1 + np.sum(np.abs(centered) >= abs(delta))) / (count + 1))
        comparisons[method] = dict(delta_nrmse=delta, delta_ci95=np.percentile(boot_delta, [2.5, 97.5]).tolist(),
                relative_gain_percent=100 * delta / point[method],
                relative_gain_ci95=np.percentile(gains, [2.5, 97.5]).tolist(),
                descriptive_two_sided_p=p, primary=method in primary)
    adjusted = holm([comparisons[m]["descriptive_two_sided_p"] for m in primary])
    for method, p in zip(primary, adjusted):
        comparisons[method]["descriptive_holm_adjusted_p"] = float(p)
    return comparisons, samples, weights


def strata_summary(predictions, target_z, mean, scale, volume, times):
    ranks = np.argsort(volume, kind="stable")
    quintiles = np.array_split(ranks, 5)
    cells, months = {}, {}
    dates = np.datetime64("2023-01-01T00") + times.astype("timedelta64[h]")
    month_names = dates.astype("datetime64[M]").astype(str)
    for method, z in predictions.items():
        error = (np.maximum(z * scale + mean, 0.) - (target_z * scale + mean)) / scale
        local = np.sqrt(np.mean(error ** 2, axis=0)).mean(axis=1)
        cells[method] = local
        months[method] = {month: dict(hours=int((month_names == month).sum()),
                         macro_nrmse=float(np.sqrt(np.mean(error[month_names == month] ** 2, axis=0)).mean()))
                         for month in np.unique(month_names)}
    rows = []
    for method in predictions:
        if method == "xi_shapley":
            continue
        delta = cells[method] - cells["xi_shapley"]
        rows.append(dict(method=method, zones_improved=int((delta > 1e-12).sum()),
                         zones_tied=int((np.abs(delta) <= 1e-12).sum()),
                         zone_improvement_fraction=float((delta > 1e-12).mean()),
                         quintile_gain_percent=[float(100 * (cells[method][q].mean() - cells['xi_shapley'][q].mean()) /
                                                      cells[method][q].mean()) for q in quintiles]))
    return dict(comparisons=rows, monthly=months, volume_quintile_zone_positions=[q.tolist() for q in quintiles],
                quintile_unit="D0 mean count over channels; rank quintiles, all 66 zones"), cells


def run(args):
    started = time.monotonic()
    output = args.output / "exploration2023"
    output.mkdir(parents=True, exist_ok=True)
    core = args.output / "core"
    manifest = dict(version=FULL_VERSION, status="running", T23_evaluated=False, T24_evaluated=False,
                    confirmation=False, core_version=VERSION, timing_seconds={}, cache_origins={})
    models.atomic_json(output / "manifest.json", manifest)
    audit = {}
    try:
        core_manifest = json.loads((core / "manifest.json").read_text())
        if core_manifest["status"] != "passed" or core_manifest["sample_count"] != 4096 or core_manifest["model_seeds"] != [0, 1, 2]:
            raise ValueError("Need passed full core, all three checkpoints")
        for path, expected in core_manifest["input_hashes"].items():
            if file_hash(ROOT / path) != expected:
                raise ValueError(f"Core provenance mismatch: {path}")
        provenance = dict(core_hash=file_hash(core / "manifest.json"), runner_hash=file_hash(Path(__file__)),
                          input_hashes=core_manifest["input_hashes"], stage=FULL_VERSION,
                          batch_size=args.batch_size, device=args.device)
        manifest["provenance"] = provenance
        data, prepared, original = training.load_data(False)
        splits = split_indices()
        check_splits(data, original, splits)
        with np.load(core / "inputs.npz", allow_pickle=False) as saved:
            fv_source, pool_source = saved["fv_source_scores"], saved["pool_source_scores"]
        with np.load(core / "priors_and_fv_predictions.npz", allow_pickle=False) as saved:
            priors = {k[6:]: saved[k] for k in saved.files if k.startswith("prior_")}
            conditional_priors = {m: saved[f"condition_prior_{m}"].reshape(3, 3, 66, 3).mean(axis=1) for m in PAIRING_KEYS}
            fv_prediction = saved["frozen_predictions"]
        fg_bg = shap_draws(splits)
        manifest["shap_draw_hashes"] = [{"foreground": array_hash(f), "background": array_hash(b)} for f, b in fg_bg]
        device = models.choose_device(args.device)
        manifest["device"] = str(device)
        shap_scores = np.zeros((2, 3, 66, 3))
        t = time.monotonic()
        cache_dir = output / "shap_branch_cache"
        cache_dir.mkdir(exist_ok=True)
        for seed in range(3):
            model, _ = models.load_model(training, data, prepared["attributes_z"].shape[1], seed, device)
            for budget, (fg, bg) in enumerate(fg_bg):
                print(f"SHAP seed {seed}: {len(fg)} foreground x {len(bg)} background, 66 zones", flush=True)
                for zone in range(66):
                    key = dict(**provenance, seed=seed, zone=zone, foreground=array_hash(fg), background=array_hash(bg))
                    path = cache_dir / f"seed{seed}_f{len(fg)}_b{len(bg)}_zone{zone}.npz"
                    shap_scores[budget, seed, zone] = compute_shap_with_cache(model, prepared, fg, bg, zone, device,
                                                                          args.batch_size, path, key, manifest["cache_origins"])
            del model
        manifest["timing_seconds"]["full_shap"] = time.monotonic() - t
        precision = shap_precision(shap_scores[0].mean(axis=0), shap_scores[1].mean(axis=0))
        precision["per_checkpoint"] = [shap_precision(shap_scores[0, s], shap_scores[1, s]) for s in range(3)]
        priors["grouped_shap"] = shap_scores[0].mean(axis=0)
        np.savez_compressed(output / "full_shap.npz", raw_scores=shap_scores,
                            initial_foreground=fg_bg[0][0], initial_background=fg_bg[0][1],
                            doubled_foreground=fg_bg[1][0], doubled_background=fg_bg[1][1])
        models.atomic_json(output / "shap_precision.json", precision)
        print(f"SHAP precision: max weight change {precision['maximum_absolute_weight_change']:.4f}; converged={precision['converged']}", flush=True)
        fv = np.r_[splits["F"], splits["V"]]
        target_fv, fit_n = prepared["targets"][fv], len(splits["F"])
        normal = original["normalization"]
        mean, scale = normal["flow_mean"], normal["flow_std"]
        ensemble_fv = fv_prediction.mean(axis=0)
        design_fv, _, _ = make_design(fv_source, ensemble_fv, fit_n)
        selections, trials, mappings = {}, {}, {}
        all_priors = {**priors, "univariate": np.ones((66, 3))}
        t = time.monotonic()
        for method, raw in all_priors.items():
            best, candidates, _, mapping = tune_method(design_fv, ensemble_fv, target_fv, fit_n, raw, method, mean, scale, audit)
            selections[method], trials[method], mappings[method] = best, candidates, mapping
        eligible = [m for m in COMPETITORS if m != "grouped_shap" or precision["converged"]]
        strongest = min(eligible, key=lambda m: (selections[m]["macro_nrmse"], COMPETITORS.index(m)))
        print(f"Validation locked competitor B*: {strongest}", flush=True)
        single_selections, single_trials, single_priors = [], [], []
        for seed in range(3):
            local_priors = {**all_priors, **{m: v[seed] for m, v in conditional_priors.items()}, "grouped_shap": shap_scores[0, seed]}
            local_design, _, _ = make_design(fv_source, fv_prediction[seed], fit_n)
            chosen, candidates = {}, {}
            for method, raw in local_priors.items():
                best, grid, _, _ = tune_method(local_design, fv_prediction[seed], target_fv, fit_n, raw, method, mean, scale, audit)
                chosen[method], candidates[method] = best, grid
            single_selections.append(chosen)
            single_trials.append(candidates)
            single_priors.append(local_priors)
        manifest["timing_seconds"]["main_and_single_selection"] = time.monotonic() - t
        rng = np.random.default_rng(SEEDS["placebo"])
        permutations = np.argsort(rng.random((200, 66, 3)), axis=-1)
        null_scores = np.take_along_axis(np.broadcast_to(priors["xi_shapley"], (200, 66, 3)), permutations, axis=-1)
        null_selection, null_trials = [], []
        t = time.monotonic()
        print("Selecting all 200 scale-permutation priors on V before opening T23", flush=True)
        for raw in null_scores:
            best, grid, _, _ = tune_method(design_fv, ensemble_fv, target_fv, fit_n, raw, "xi_shapley", mean, scale, audit)
            null_selection.append(best)
            null_trials.append(grid)
        manifest["timing_seconds"]["null_selection"] = time.monotonic() - t
        lock = dict(version=FULL_VERSION, provenance=provenance, T23_evaluated=False, B_star=strongest,
                    eligible_competitors=eligible, SHAP_converged=precision["converged"],
                    primary_comparisons=["uniform", "ei_shapley", strongest],
                    main_selection=selections, single_selection=single_selections, placebo_selection=null_selection,
                    lambda_grid=list(LAMBDAS), gamma_grid=list(GAMMAS), prior_hashes={m: array_hash(v) for m, v in all_priors.items()},
                    placebo_scores_hash=array_hash(null_scores), placebo_permutation_hash=array_hash(permutations),
                    random_seeds=SEEDS, inference="descriptive 2023 exploration; no independent confirmation")
        models.atomic_json(output / "configuration_lock.json", lock)
        models.atomic_json(output / "validation_trials.json", dict(main=trials, single=single_trials, placebo=null_trials, mappings=mappings))
        np.savez_compressed(output / "locked_priors.npz", **all_priors, null_scores=null_scores, permutations=permutations)
        lock_hash = file_hash(output / "configuration_lock.json")
        # T23 begins here, after all main/single/null selection and B* locking.
        print("Configuration lock written; opening the full 760-hour T23 exploration", flush=True)
        t = time.monotonic()
        test_predictions = []
        for seed in range(3):
            model, _ = models.load_model(training, data, prepared["attributes_z"].shape[1], seed, device)
            key = dict(**provenance, seed=seed, configuration_lock=lock_hash, times=array_hash(splits["T23"]))
            saved = cached_arrays(output / f"seed{seed}_T23.npz", key,
                     lambda: dict(prediction=predict(model, prepared, splits["T23"], device, args.batch_size)), manifest["cache_origins"])
            test_predictions.append(saved["prediction"])
            del model
        manifest["timing_seconds"]["T23_forward"] = time.monotonic() - t
        # Rebuild the frozen P transformation; no representation update using V/T.
        from scripts.nyc_taxi_information_priors import SourceRepresentation, BRANCHES
        with np.load(core / "inputs.npz", allow_pickle=False) as saved:
            rep = SourceRepresentation(tuple(saved[f"pca_mean_{m}"] for m in BRANCHES),
                    tuple(saved[f"pca_basis_{m}"] for m in BRANCHES), saved["pca_scales"], saved["pca_explained"], saved["pca_constant"])
        test_source = rep.transform(data["flow"], splits["T23"])
        source = np.concatenate([fv_source, test_source])
        target = prepared["targets"][np.r_[fv, splits["T23"]]]
        full_prediction = np.concatenate([fv_prediction, np.stack(test_predictions)], axis=1)
        ensemble = full_prediction.mean(axis=0)
        design, _, _ = make_design(source, ensemble, fit_n)
        np.testing.assert_allclose(design[:len(fv)], design_fv, rtol=0, atol=0)
        predictions = {m: fit_selected(design, ensemble, target, fit_n, raw, m, selections[m])[len(fv):]
                       for m, raw in all_priors.items()}
        predictions["frozen"] = ensemble[len(fv):]
        target_test = target[len(fv):]
        main_metrics = {m: metrics(p, target_test, mean, scale) for m, p in predictions.items()}
        single_results, single_predictions = [], {}
        for seed in range(3):
            local = full_prediction[seed]
            local_design, _, _ = make_design(source, local, fit_n)
            predicted = {m: fit_selected(local_design, local, target, fit_n, raw, m, single_selections[seed][m])[len(fv):]
                         for m, raw in single_priors[seed].items()}
            predicted["frozen"] = local[len(fv):]
            scores = {m: metrics(p, target_test, mean, scale) for m, p in predicted.items()}
            single_results.append(dict(seed=seed, metrics=scores, selection=single_selections[seed],
                    primary_gain_percent={m: 100*(scores[m]["macro_nrmse"]-scores['xi_shapley']["macro_nrmse"])/scores[m]["macro_nrmse"]
                                           for m in lock["primary_comparisons"]},
                    shap_converged=precision["per_checkpoint"][seed]["converged"]))
            single_predictions.update({f"seed{seed}_{m}": p for m, p in predicted.items()})
        t = time.monotonic()
        null_metrics = []
        null_predictions = []
        for raw, choice in zip(null_scores, null_selection):
            p = fit_selected(design, ensemble, target, fit_n, raw, "xi_shapley", choice)[len(fv):]
            null_metrics.append(metrics(p, target_test, mean, scale)["macro_nrmse"])
            null_predictions.append(p)
        null_metrics = np.asarray(null_metrics)
        real = main_metrics["xi_shapley"]["macro_nrmse"]
        at_least_as_good = int((null_metrics <= real + 1e-12).sum())
        placebo = dict(count=200, independently_selected=True, nrmse_mean=float(null_metrics.mean()),
                       nrmse_ci95=np.percentile(null_metrics, [2.5, 97.5]).tolist(),
                       better_count=int((null_metrics < real - 1e-12).sum()),
                       tied_count=int((np.abs(null_metrics - real) <= 1e-12).sum()),
                       equal_or_better_count=at_least_as_good, monte_carlo_value=(1+at_least_as_good)/201,
                       rank_min=1+int((null_metrics < real - 1e-12).sum()), rank_max=1+at_least_as_good,
                       caveat="scale labels are not strictly exchangeable; structural placebo, not exact unconditional test")
        manifest["timing_seconds"]["null_test"] = time.monotonic() - t
        t = time.monotonic()
        comparisons, bootstrap, bootstrap_weights = bootstrap_comparisons(predictions, target_test, mean, scale, lock["primary_comparisons"])
        stratification, cells = strata_summary(predictions, target_test, mean, scale,
                                              data["flow"][splits["D0"]].mean(axis=(0, 2)), splits["T23"])
        manifest["timing_seconds"]["bootstrap_and_strata"] = time.monotonic() - t
        summary = dict(version=FULL_VERSION, scope="2023 exploration only", confirmation=False,
                       metrics=main_metrics, comparison=comparisons, B_star=strongest, SHAP_precision=precision,
                       selections=selections, placebo=placebo, single_checkpoints=single_results,
                       stratification=stratification, bootstrap=dict(replicates=4000, circular_block_hours=168,
                            pairing="same actual sampled hour counts for every method, all zones/channels",
                            independent_replication="approximately 4.5 weeks; seeds/zones/hours/draws are not independent systems",
                            intervals="95% percentile; centered bootstrap errors for descriptive two-sided p; Holm for three primary contrasts"),
                       remaining=["new-year data and confirmation", "unresolved manuscript version",
                            "observed target-hour weather limits deployable forecasting interpretation"],
                       failure_or_exclusion=[] if precision["converged"] else ["SHAP initial budget failed prespecified precision audit; excluded from eligible B* only"])
        np.savez_compressed(output / "T23_predictions.npz", **predictions, target_z=target_test, times=splits["T23"],
                            flow_mean=mean, flow_std=scale, null_predictions=np.stack(null_predictions),
                            null_nrmse=null_metrics, **{f"bootstrap_{m}": v for m, v in bootstrap.items()},
                            bootstrap_hour_counts=bootstrap_weights, **{f"zone_nrmse_{m}": v for m, v in cells.items()})
        np.savez_compressed(output / "single_checkpoint_T23_predictions.npz", **single_predictions)
        models.atomic_json(output / "summary.json", summary)
        manifest.update(status="completed", T23_evaluated=True, configuration_lock_hash=lock_hash,
                        total_seconds=time.monotonic()-started, peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                        B_star=strongest, SHAP_converged=precision["converged"], summary_hash=file_hash(output / "summary.json"))
        print(f"Completed full 2023 exploration in {manifest['total_seconds']:.1f}s; Xi nRMSE={real:.6f}", flush=True)
    except Exception as error:
        manifest.update(status="failed", failure=str(error), total_seconds=time.monotonic()-started)
        raise
    finally:
        models.atomic_json(output / "audit.json", audit)
        models.atomic_json(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == "__main__":
    main()
