#!/usr/bin/env python3
"""Implement stages 0/1 of the Taxi protocol without opening T23 evaluation.

Default: one checkpoint/context, 512 donors, F/V checks, small SHAP benchmark.
--stage core: three checkpoints/contexts and 4096 donors, still F/V only.
Full SHAP, placebo, T23 and new-year confirmation require later stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import train_nyc_taxi_mgstn as training
from scripts import compute_nyc_taxi_mgstn_ei as models
from scripts.nyc_taxi_information_priors import (
    VERSION, BRANCHES, LAGS, RIDGE, TOLERANCE_BITS, SourceRepresentation,
    fit_intervention, observational_priors, penalty_weights, positive_covariance, shapley,
)

OUTPUT = ROOT / "results/nyc_taxi_information_prior_calibration"
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100., 1000.)
GAMMAS = (0., .5, 1., 2., 4.)
SEEDS = dict(donor=20261011, shap=20261012, placebo=20261013, bootstrap=20261014)
STATES = ("weekday_peak", "weekend_midday", "rainy_high_demand")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(str((array.shape, array.dtype)).encode() + np.ascontiguousarray(array).tobytes()).hexdigest()


def cached_arrays(path: Path, provenance: dict, compute, origins: dict):
    """Only reuse expensive forwards when the complete input provenance matches."""
    signature = json.dumps(provenance, sort_keys=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["provenance"]) == signature:
                origins[path.name] = "matched cache"
                return {k: saved[k] for k in saved.files if k != "provenance"}
    arrays = compute()
    np.savez_compressed(path, provenance=np.array(signature), **arrays)
    origins[path.name] = "computed (absent or mismatched provenance)"
    return arrays


def split_indices():
    return dict(D0=np.arange(1176, 6484), F=np.arange(6484, 7492), V=np.arange(7492, 8000),
                P=np.arange(1176, 7492), T23=np.arange(8000, 8760))


def check_splits(data: dict, original: dict, splits: dict):
    if data["flow"].shape != (8760, 66, 2):
        raise ValueError("Protocol requires 8760 hours and 66 two-channel zones")
    np.testing.assert_array_equal(original["indices"]["train"], splits["D0"])
    np.testing.assert_array_equal(original["indices"]["valid"], np.r_[splits["F"], splits["V"]])
    np.testing.assert_array_equal(original["indices"]["test"], splits["T23"])
    if any(i in data["zone_ids"] for i in (103, 104, 105)):
        raise ValueError("Inactive zones present")
    for lag in LAGS:
        if np.any(lag <= 0) or splits["P"][0] - lag.max() < 0:
            raise ValueError("History is not strictly before target")


@torch.inference_mode()
def predict(model, prepared: dict, times: np.ndarray, device, batch_size: int):
    loader = DataLoader(training.TaxiWindows(prepared["flow_z"], prepared["attributes_z"],
                                            prepared["targets"], times), batch_size=batch_size, shuffle=False)
    return np.concatenate([model(training.move(b, device)).cpu().numpy() for b in loader])


def donor_batch(prepared: dict, donors: np.ndarray, center: dict, device):
    count, nodes, _ = donors.shape
    batch = {k: v.unsqueeze(0).expand(count, *v.shape).to(device) for k, v in center.items() if k.endswith("attr")}
    for m, (name, lag) in enumerate(zip(BRANCHES, LAGS)):
        t = donors[:, :, m]
        raw = prepared["flow_z"][t[:, :, None] - lag, np.arange(nodes)[None, :, None]]
        batch[name] = torch.from_numpy(raw.transpose(0, 1, 3, 2).copy()).to(device)
    return batch


@torch.inference_mode()
def intervention_response(model, prepared, donors, center, device, batch_size):
    out = []
    for start in range(0, len(donors), batch_size):
        batch = donor_batch(prepared, donors[start:start + batch_size], center, device)
        out.append(model(batch).cpu().numpy())
    return np.concatenate(out)


def donor_scores(representation, flow, pool, donors):
    pool_scores = representation.transform(flow, pool)
    positions = np.searchsorted(pool, donors)
    return pool_scores[positions, np.arange(flow.shape[1])[None, :, None], np.arange(3)[None, None, :]], pool_scores


@torch.inference_mode()
def grouped_shap(model, prepared, foreground, background, zones, device, batch_size=64,
                 verify=False):
    """Cache independent branch representations with identical hidden donor rows.

    Graph mixing happens inside each branch before extracting the target zone.
    Do not substitute cached representations from a different foreground world.
    """
    dataset = training.TaxiWindows(prepared["flow_z"], prepared["attributes_z"], prepared["targets"], foreground)
    fg = {k: torch.stack([dataset[i][k] for i in range(len(dataset))]).to(device)
          for k in dataset[0] if k != "target"}
    bg_dataset = training.TaxiWindows(prepared["flow_z"], prepared["attributes_z"], prepared["targets"], background)
    bg = {m: torch.stack([bg_dataset[i][m] for i in range(len(background))]).to(device) for m in BRANCHES}
    nf, nb = len(foreground), len(background)
    baseline = [getattr(model, m)(fg[m], fg[m + "_attr"]) for m in BRANCHES]
    target = model.target_attribute(fg["target_attr"])[:, None].expand(nf, nb, -1).reshape(nf * nb, -1)
    result = np.zeros((len(zones), 3))
    max_error = 0.
    for zi, zone in enumerate(zones):
        hidden = []
        for m in BRANCHES:
            pieces = []
            for start in range(0, nf * nb, batch_size):
                rows = torch.arange(start, min(start + batch_size, nf * nb), device=device)
                a, b = rows // nb, rows % nb
                masked = fg[m][a].clone()
                masked[:, zone] = bg[m][b, zone]
                pieces.append(getattr(model, m)(masked, fg[m + "_attr"][a])[:, zone])
            hidden.append(torch.cat(pieces))
        games = []
        for mask in range(8):
            representations = [baseline[j][:, zone, None].expand(nf, nb, -1).reshape(nf * nb, -1)
                               if mask & (1 << j) else hidden[j] for j in range(3)]
            fused = model.granularity_fusion(torch.cat(representations, dim=-1))
            pred = model.output(torch.cat([fused, target], dim=-1))
            games.append(pred.reshape(nf, nb, 2).mean(dim=1).cpu().numpy())
            if verify:
                # Full masked forward on the same sample, all eight coalitions.
                direct = {k: v[:1].clone() for k, v in fg.items()}
                for j, m in enumerate(BRANCHES):
                    if not mask & (1 << j):
                        direct[m][:, zone] = bg[m][:1, zone]
                full = model(direct)[:, zone]
                max_error = max(max_error, float((pred[:1] - full).abs().max().cpu()))
        values = np.stack(games).astype(np.float64)
        # SHAP game has a nonzero baseline; subtract it before generic attribution.
        attribution = shapley(values - values[0])
        result[zi] = np.abs(attribution).mean(axis=(1, 2))
    if verify and max_error > 1e-5:
        raise ValueError(f"SHAP branch/full forward mismatch: {max_error:g} standardized units")
    return result, dict(branch_full_max_error_z=max_error, foreground_times=foreground.tolist(),
                        background_times=background.tolist(), zone_positions=list(map(int, zones)),
                        branch_samples=3 * nf + len(zones) * 3 * nf * nb,
                        complete_coalitions=8)


def make_design(source: np.ndarray, prediction: np.ndarray, fit_n: int):
    # [time,zone,channel,7]; both targets share the same six history columns.
    history = source.reshape(len(source), source.shape[1], 6)
    history = np.broadcast_to(history[:, :, None, :], prediction.shape + (6,))
    design = np.concatenate([prediction[..., None], history], axis=-1).astype(float)
    mean, scale = design[:fit_n].mean(axis=0), design[:fit_n].std(axis=0)
    constant = scale < 1e-12
    scale[constant] = 1.
    design = np.where(constant, 0., (design - mean) / scale)
    return design, mean, scale


def ridge_statistics(design, prediction, target, fit_n, univariate=False):
    xf = (design[..., :1] if univariate else design)[:fit_n]
    residual = target[:fit_n].astype(float) - prediction[:fit_n].astype(float)
    intercept = residual.mean(axis=0)
    # Designs are centered on F, so the unpenalized intercept separates exactly.
    gram = np.einsum("tzci,tzcj->zcij", xf, xf) / fit_n
    rhs = np.einsum("tzci,tzc->zci", xf, residual - intercept) / fit_n
    return gram, rhs, intercept


def fit_calibrator(design, prediction, target, fit_n, penalty, lam, univariate=False, statistics=None):
    x = design[..., :1] if univariate else design
    gram, rhs, intercept = (ridge_statistics(design, prediction, target, fit_n, univariate)
                            if statistics is None else statistics)
    weights = np.ones(gram.shape[:-1])
    if not univariate:
        weights[..., 1:] = np.repeat(penalty, 2, axis=-1)[:, None, :]
    regularized = gram + lam * weights[..., None, :] * np.eye(x.shape[-1])
    beta = np.linalg.solve(regularized, rhs[..., None])[..., 0]
    fitted = prediction + intercept + np.einsum("tzci,zci->tzc", x, beta)
    effective_df = 1 + np.trace(np.linalg.solve(regularized, gram), axis1=-2, axis2=-1)
    return fitted, effective_df


def metrics(prediction_z, target_z, mean, scale):
    raw = prediction_z * scale + mean
    target = target_z * scale + mean
    prediction = np.maximum(raw, 0.)  # Count forecast rule; never applied to information.
    error = prediction - target
    cells = np.sqrt(np.mean(error ** 2, axis=0)) / scale
    record = dict(macro_nrmse=float(cells.mean()), macro_nrmse_by_channel=cells.mean(axis=0).tolist(),
                  unclipped_macro_nrmse=float((np.sqrt(np.mean((raw - target) ** 2, axis=0)) / scale).mean()),
                  clipped_prediction_fraction=float((raw < 0).mean()))
    for c, name in enumerate(("inflow", "outflow")):
        record[name + "_rmse"] = float(np.sqrt(np.mean(error[..., c] ** 2)))
        record[name + "_mae"] = float(np.mean(np.abs(error[..., c])))
    return record


def tune_method(design, prediction, target, fit_n, raw, method, mean, scale, audit):
    trials, best, best_prediction = [], None, None
    gammas = (0.,) if method in ("uniform", "univariate") else GAMMAS
    statistics = ridge_statistics(design, prediction, target, fit_n, method == "univariate")
    for gamma in gammas:
        penalties, mapping = penalty_weights(raw, method, gamma, audit)
        for lam in LAMBDAS:
            fitted, df = fit_calibrator(design, prediction, target, fit_n, penalties, lam,
                                        method == "univariate", statistics)
            metric = metrics(fitted[fit_n:], target[fit_n:], mean, scale)
            entry = dict(lambda_value=lam, gamma=gamma, **metric,
                         mean_effective_df=float(df.mean()), penalty_std=mapping["penalty_std"])
            trials.append(entry)
            if best is None or entry["macro_nrmse"] < best["macro_nrmse"] - 1e-12 or (
                abs(entry["macro_nrmse"] - best["macro_nrmse"]) <= 1e-12 and
                (gamma, -lam) < (best["gamma"], -best["lambda_value"])):
                best, best_prediction = entry, fitted
    _, mapping = penalty_weights(raw, method, best["gamma"], audit)
    return best, trials, best_prediction, mapping


def run(args):
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    output = args.output / args.stage
    output.mkdir(parents=True, exist_ok=True)
    audit = dict(nonnegative_tolerance_bit=TOLERANCE_BITS, covariance_ridge=RIDGE)
    manifest = dict(version=VERSION, stage=args.stage, status="running", test_evaluated=False,
                    random_seeds=SEEDS, input_hashes={}, timing_seconds={}, cache_origins={})
    models.atomic_json(output / "manifest.json", manifest)
    try:
        data, prepared, original = training.load_data(False)
        splits = split_indices()
        check_splits(data, original, splits)
        seeds = [0] if args.stage == "smoke" else [0, 1, 2]
        states = STATES[:1] if args.stage == "smoke" else STATES
        samples = 512 if args.stage == "smoke" else 4096
        paths = [training.DATA_PATH, *(models.checkpoint_path(s) for s in seeds),
                 Path(__file__), ROOT / "scripts/nyc_taxi_information_priors.py",
                 ROOT / "scripts/train_nyc_taxi_mgstn.py", ROOT / "scripts/compute_nyc_taxi_mgstn_ei.py",
                 ROOT / "scripts/unicm_observational_information.py", ROOT / "exp/TM/transport_map_density.py",
                 ROOT / "scripts/run_unicm_information_prior_comparison.py"]
        manifest["input_hashes"] = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
        manifest.update(sample_count=samples, model_seeds=seeds, contexts=list(states),
                        information_unit="bit", split_indices={k: v.tolist() for k, v in splits.items()},
                        estimator="coherent independent-block affine triangular TM; observed correlated affine TM",
                        support="198 independent empirical P-donor history blocks; within-block channels retained",
                        fit_fraction=.7, threads=args.threads, batch_size=args.batch_size,
                        cache_reuse="expensive forward arrays require matching full provenance; small fits recomputed",
                        manuscript=dict(parent="P6UJCVG8", main="DXGC7JEA", supplement="MWIWKSVG",
                            read_date="2026-10-11", version="unresolved: no manuscript date/revision",
                            locations="main pp12-17 Eqs5-12; supplement S1.2/S1.3/S3.2/S12.3",
                            differences="bits vs manuscript nats; per-channel PC; three blocks; affine vs quadratic TM"))
        t = time.monotonic()
        representation = SourceRepresentation.fit(data["flow"], splits["P"])
        fv = np.r_[splits["F"], splits["V"]]
        fv_source = representation.transform(data["flow"], fv)
        rng = np.random.default_rng(SEEDS["donor"])
        donors = rng.choice(splits["P"], size=(samples, 66, 3), replace=True)
        source, pool_source = donor_scores(representation, data["flow"], splits["P"], donors)
        order = rng.permutation(samples)
        fit, evaluate = order[:int(.7 * samples)], order[int(.7 * samples):]
        standard_noise = rng.standard_normal((samples, 132))
        manifest["pairing_hashes"] = {k: array_hash(a) for k, a in dict(donors=donors, fit=fit,
                                             evaluate=evaluate, standard_noise=standard_noise).items()}
        manifest["timing_seconds"]["representation"] = time.monotonic() - t
        np.savez_compressed(output / "inputs.npz", donors=donors, fit=fit, evaluate=evaluate,
                            standard_noise=standard_noise, pca_scales=representation.scales,
                            pca_explained=representation.explained, pca_constant=representation.constant,
                            source_scores=source, pool_source_scores=pool_source, fv_source_scores=fv_source,
                            **{f"pca_mean_{m}": representation.means[j] for j, m in enumerate(BRANCHES)},
                            **{f"pca_basis_{m}": representation.bases[j] for j, m in enumerate(BRANCHES)})
        manifest["pca"] = dict(minimum_explained_variance=float(representation.explained.min()),
                               median_explained_variance=float(np.median(representation.explained)),
                               constant_columns=int(representation.constant.sum()))
        print("Source representations and paired donors ready", flush=True)
        observed = {m: [] for m in ("phi_r", "phi_wms", "surd", "o_shapley", "correlation")}
        native_observed = {}
        t = time.monotonic()
        for zone in range(66):
            scores, native = observational_priors(pool_source[:, zone], prepared["targets"][splits["P"], zone],
                                                   audit, f"observed_zone{zone}")
            for m, v in scores.items():
                observed[m].append(v)
            native_observed.update({f"zone{zone}_{k}": v for k, v in native.items()})
        manifest["timing_seconds"]["observational_priors"] = time.monotonic() - t
        np.savez_compressed(output / "observational_native.npz", **native_observed)
        predictions, interventions, diagnostics, noise_records = [], {m: [] for m in ("xi_shapley", "ei_shapley", "pair_only")}, [], []
        device = models.choose_device(args.device)
        manifest["device"] = str(device)
        local_split = dict(indices={"valid": splits["F"]})
        centers = models.select_centers(training, data, prepared, local_split, list(states))
        manifest["centers"] = [{k: c[k] for k in ("name", "timestamp_index", "timestamp")} for c in centers]
        for seed in seeds:
            model, _ = models.load_model(training, data, prepared["attributes_z"].shape[1], seed, device)
            t = time.monotonic()
            prediction = cached_arrays(output / f"seed{seed}_fv_forward.npz",
                dict(input_hashes=manifest["input_hashes"], times=array_hash(fv), seed=seed,
                     device=str(device), batch_size=args.batch_size),
                lambda: dict(prediction=predict(model, prepared, fv, device, args.batch_size)),
                manifest["cache_origins"])["prediction"]
            predictions.append(prediction)
            manifest["timing_seconds"][f"seed{seed}_fv_forward"] = time.monotonic() - t
            residual = (prepared["targets"][splits["F"]] - prediction[:len(splits["F"])]).reshape(-1, 132)
            lw = LedoitWolf().fit(residual.astype(float))
            cov = lw.covariance_
            positive_covariance(cov, f"seed{seed}_F_noise", audit, ridge=0.)
            paired_noise = (standard_noise @ np.linalg.cholesky(cov).T).reshape(samples, 66, 2)
            noise_records.append(dict(seed=seed, samples=len(residual), shrinkage=float(lw.shrinkage_),
                                      residual_mean_z=residual.mean(axis=0).tolist()))
            for center in centers:
                print(f"seed {seed}, {center['name']}: {samples} complete interventions", flush=True)
                t = time.monotonic()
                clean = cached_arrays(output / f"seed{seed}_{center['name']}_forward.npz",
                    dict(input_hashes=manifest["input_hashes"], donors=array_hash(donors), seed=seed,
                         context=center["name"], center_time=center["timestamp_index"],
                         device=str(device), batch_size=args.batch_size),
                    lambda: dict(clean=intervention_response(model, prepared, donors, center["batch"], device, args.batch_size)),
                    manifest["cache_origins"])["clean"]
                elapsed = time.monotonic() - t
                manifest["timing_seconds"][f"seed{seed}_{center['name']}_forward"] = elapsed
                noisy = clean + paired_noise
                np.savez_compressed(output / f"seed{seed}_{center['name']}_responses.npz",
                                    clean=clean, noisy=noisy, F_noise_covariance=cov)
                per_condition = {m: [] for m in interventions}
                native_condition = {}
                t = time.monotonic()
                for zone in range(66):
                    scores, native, diag = fit_intervention(source[:, zone], noisy[:, zone], pool_source[:, zone],
                                      fit, evaluate, audit, f"seed{seed}_{center['name']}_zone{zone}")
                    for m, v in scores.items():
                        per_condition[m].append(v)
                    native_condition.update({f"zone{zone}_{k}": v for k, v in native.items()})
                    diagnostics.append(dict(seed=seed, state=center["name"], zone_id=int(data["zone_ids"][zone]), **diag))
                manifest["timing_seconds"][f"seed{seed}_{center['name']}_oracles"] = time.monotonic() - t
                for m, v in per_condition.items():
                    interventions[m].append(np.stack(v))
                np.savez_compressed(output / f"seed{seed}_{center['name']}_native.npz", **native_condition)
            if seed == 0:
                shap_rng = np.random.default_rng(SEEDS["shap"])
                fg = shap_rng.choice(splits["F"], size=4, replace=False)
                bg = shap_rng.choice(np.setdiff1d(splits["P"], fg), size=4, replace=False)
                t = time.monotonic()
                probe, record = grouped_shap(model, prepared, fg, bg, [0, 33], device, args.batch_size, verify=True)
                elapsed = time.monotonic() - t
                record.update(seconds=elapsed, initial_full_branch_samples=3 * 32 + 66 * 3 * 32 * 16,
                              doubled_full_branch_samples=3 * 64 + 66 * 3 * 64 * 32,
                              estimate_caveat="small batch, includes direct verification; rough extrapolation only")
                record["three_checkpoint_initial_plus_doubled_seconds_estimate"] = elapsed / record["branch_samples"] * (
                    record["initial_full_branch_samples"] + record["doubled_full_branch_samples"]) * 3
                # Measure the actual 32/16 and 64/32 budgets on one zone. These are
                # speed probes, not the required all-zone SHAP convergence audit.
                fg_large = shap_rng.choice(splits["F"], size=64, replace=False)
                bg_large = shap_rng.choice(np.setdiff1d(splits["P"], fg_large), size=32, replace=False)
                budget_timings = []
                for nf, nb in ((32, 16), (64, 32)):
                    t = time.monotonic()
                    scores, detail = grouped_shap(model, prepared, fg_large[:nf], bg_large[:nb], [0], device,
                                                  args.batch_size, verify=False)
                    probe_seconds = time.monotonic() - t
                    budget_timings.append(dict(foreground=nf, background=nb, seconds=probe_seconds))
                    np.savez_compressed(output / f"shap_speed_{nf}_{nb}.npz", raw_scores=scores,
                                        foreground=fg_large[:nf], background=bg_large[:nb])
                record["actual_budget_one_zone_timings"] = budget_timings
                record["three_checkpoint_initial_plus_doubled_seconds_estimate"] = sum(
                    b["seconds"] for b in budget_timings) * 66 * 3
                record["estimate_caveat"] = "one zone/checkpoint at actual budgets; approximate wall-time, not convergence"
                manifest["shap_benchmark"] = record
                np.savez_compressed(output / "shap_probe.npz", raw_scores=probe, foreground=fg, background=bg)
                print(f"SHAP equivalence verified; probe {elapsed:.2f}s", flush=True)
            del model
        priors = {m: np.stack(v) for m, v in observed.items()}
        priors.update({m: np.mean(v, axis=0) for m, v in interventions.items()})
        priors["uniform"] = np.ones((66, 3))
        average_prediction = np.mean(predictions, axis=0)
        target = prepared["targets"][fv]
        fit_n = len(splits["F"])
        design, design_mean, design_scale = make_design(fv_source, average_prediction, fit_n)
        normal = original["normalization"]
        mean, scale = normal["flow_mean"], normal["flow_std"]
        manifest["normalization_hashes"] = {k: array_hash(v) for k, v in normal.items()}
        selection, trials, fitted, mappings = {}, {}, {}, {}
        t = time.monotonic()
        for method, raw in {**priors, "univariate": np.ones((66, 3))}.items():
            best, candidates, prediction, mapping = tune_method(design, average_prediction, target,
                                                               fit_n, raw, method, mean, scale, audit)
            selection[method], trials[method], fitted[method], mappings[method] = best, candidates, prediction, mapping
        uniform_penalty, _ = penalty_weights(priors["uniform"], "uniform", 0., audit)
        expected, _ = fit_calibrator(design, average_prediction, target, fit_n, uniform_penalty, .1)
        for method, raw in priors.items():
            p, _ = penalty_weights(raw, method, 0., audit)
            actual, _ = fit_calibrator(design, average_prediction, target, fit_n, p, .1)
            np.testing.assert_array_equal(actual, expected)
        audit["gamma_zero_uniform_prediction_max_error"] = 0.
        manifest["timing_seconds"]["calibration_search"] = time.monotonic() - t
        selection["frozen"] = metrics(average_prediction[fit_n:], target[fit_n:], mean, scale)
        manifest.update(status="passed", total_seconds=time.monotonic() - started,
                        peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                        stage0_checks="splits/history, native validity, PD, closure, gamma0, branch/full SHAP",
                        calibration_parameters_per_target=8, fitted_targets=132,
                        missing_before_test=["full SHAP and doubled precision check", "validation B* lock",
                          "200 independently tuned placebos", "three single-checkpoint robustness", "T23 paired statistics"],
                        model_noise=noise_records)
        if args.stage == "smoke" and manifest["cache_origins"]["seed0_weekday_peak_forward.npz"].startswith("computed"):
            intervention_time = manifest["timing_seconds"]["seed0_weekday_peak_forward"]
            manifest["core_forward_seconds_estimate"] = intervention_time * (4096 / 512) * 9
        models.atomic_json(output / "validation_selection.json", dict(selection=selection, trials=trials, mappings=mappings,
                              scope="F/V engineering check; incomplete method list; no B* or test ranking"))
        np.savez_compressed(output / "priors_and_fv_predictions.npz", **{f"prior_{m}": v for m, v in priors.items()},
                            **{f"condition_prior_{m}": np.stack(v) for m, v in interventions.items()},
                            frozen_predictions=np.stack(predictions), target_z=target, times=fv,
                            design_mean=design_mean, design_scale=design_scale,
                            **{f"calibrated_{m}": v for m, v in fitted.items()})
        models.atomic_json(output / "oracle_diagnostics.json", dict(records=diagnostics))
        print(f"Stage {args.stage} passed in {manifest['total_seconds']:.1f}s; T23 untouched", flush=True)
    except Exception as error:
        manifest.update(status="failed", failure=str(error), total_seconds=time.monotonic() - started)
        raise
    finally:
        models.atomic_json(output / "audit.json", audit)
        models.atomic_json(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "core"), default="smoke")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("threads and batch size must be positive")
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == "__main__":
    main()
