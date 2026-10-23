#!/usr/bin/env python3
"""Candidate-only seven-output calibration; retain eleven source histories.

See docs/log/earth_seven_target_candidate.md for the predeclared search.
No existing report, figure, prediction cache, or checkpoint is overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_unicm_information_prior_comparison import sha256, nonnegative, coalition_operators
from scripts.run_unicm_target_xi_shapley_prior import DEFAULT_INPUT, DEFAULT_CALIBRATION
from scripts.run_unicm_synergy_guided_forecast import (
    chronological_split, history_summaries, target_scaling, cell_rmse,
    mean_cell_nrmse, mean_cell_acc, ridge_univariate,
)
from scripts.run_unicm_synergy_regularized_calibration import (
    prepare_designs, predict_generalized_ridge, shuffled_centrality,
)
from scripts.unicm_observational_information import nonnegative_ridge_scores

TARGETS = ("ENSO", "NPMM", "SPMM", "IOB", "IOD", "SIOD", "TNA")
METHODS = ("xi_shapley", "shap", "ei_shapley", "phi_r", "phi_wms",
           "phi_si", "causal_density", "surd")
ALPHAS = (100., 300., 1000., 3000., 10000., 30000., 100000., 300000.)
GAMMAS = (.5, 1., 2., 3., 4., 6.)
REFERENCE = ROOT / "results/unicm_observational_prior_comparison_fit253_surd_allorders"
SHAP = ROOT / "results/unicm_frozen_shap_prior_fit253_pilot"
OUTPUT = ROOT / "results/unicm_seven_target_candidate/forecast"


def target_indices(names):
    if len(names) != len(set(names)) or any(name not in names for name in TARGETS):
        raise ValueError("Missing or duplicate source/target names.")
    return np.asarray([names.index(name) for name in TARGETS])


def load_priors(names, indices, *, prior_scope='joint7'):
    """Reuse exact scalar marginals; rebuild pair/CD aggregates on seven futures."""
    audit, priors = {}, {}
    reference = json.loads((REFERENCE / 'summary.json').read_text())
    if reference['mode_names'] != names:
        raise ValueError('Prior source/target mode order differs from forecast inputs.')
    if reference['prior_data_audit']['input_sha256'] != sha256(DEFAULT_INPUT / 'model_inputs.npz'):
        raise ValueError('Prior fit data differ from forecast fit data.')
    with np.load(REFERENCE / "information_priors.npz", allow_pickle=False) as cache:
        for key in ("xi_shapley", "ei_shapley", "surd"):
            priors[key] = nonnegative(cache[key][indices], key, audit)
        pairs = [(a, b) for a in range(len(names)) for b in range(a + 1, len(names))]
        selected = set(indices.tolist())
        for key in ("phi_r", "phi_wms", "phi_si"):
            source = np.zeros((24, len(names)))
            used = []
            for pair_index, (a, b) in enumerate(pairs):
                if a in selected and b in selected:
                    edge = cache[key + "_edges"][:, pair_index]
                    source[:, a] += edge / 2
                    source[:, b] += edge / 2
                    used.append([names[a], names[b]])
            raw = np.broadcast_to(source[None], (len(indices), *source.shape)).copy()
            priors[key], mapping = nonnegative_ridge_scores(raw, key)
            audit[key] = {"output_pairs": used, "pair_count": len(used),
                          "extra_source_raw_attribution": 0., "ridge_mapping": mapping}
        transfer = nonnegative(cache["causal_density_transfer"][:, :, indices],
                               "seven_target_conditional_transfer", audit)
        # Self-links were already removed in the native cache. There are
        # 7*(11-1)=70 admissible directed source-to-target links at every lead.
        source = transfer.sum(axis=2).T / (len(indices) * (len(names) - 1))
        priors["causal_density"] = np.broadcast_to(source[None], (len(indices), *source.shape)).copy()
        audit["causal_density"] = {"admissible_links_per_lead": 70,
                                   "future_modes": list(TARGETS), "source_modes": names}
    with np.load(SHAP / "shap_priors.npz", allow_pickle=False) as cache:
        priors["shap"] = cache["shap"][indices].copy()
    if prior_scope == 'joint7':
        path = OUTPUT.parent / 'structure/joint7/ei_tables.npz'
        with np.load(path, allow_pickle=False) as cache:
            ei, xi = cache['ei_bits'], cache['xi_bits']
            source_names = ['ENSO' if n == 'nino' else n for n in cache['source_names'].astype(str)]
            if source_names != names or ei.shape != (3,24,2048) or xi.shape != ei.shape:
                raise ValueError('Joint-target coalition table scope mismatch.')
            metadata = json.loads(str(cache['metadata']))
            if metadata['target_display_names'] != list(TARGETS):
                raise ValueError('Joint-target future modes mismatch.')
        _, operator = coalition_operators(11)
        for key, table in (('xi_shapley', xi), ('ei_shapley', ei)):
            values = nonnegative(np.einsum('ms,cls->clm', operator, table), 'joint7_'+key, audit)
            np.testing.assert_allclose(values.sum(axis=-1), table[..., -1], atol=1e-10, rtol=0)
            priors[key] = np.broadcast_to(values.mean(axis=0)[None], (7,24,11)).copy()
        audit['joint7'] = dict(coalition_table=str(path.relative_to(ROOT)), sha256=sha256(path),
            target_definition='seven-dimensional joint future at each lead',
            allocation='exact Shapley of joint EI/Xi, mean across checkpoints, broadcast to seven forecasts',
            estimator='native-coordinate independent-source degree-1 affine TM, same game as SPT and system attribution')
    elif prior_scope != 'scalar7':
        raise ValueError('Unknown calibration prior scope.')
    for key, prior in priors.items():
        if prior.shape != (7, 24, 11) or not np.isfinite(prior).all() or np.any(prior < 0):
            raise ValueError(f"Invalid seven-target prior: {key}.")
    return priors, audit


def validation_search(designs, prior, validation_target, scale, *, alphas=ALPHAS, gammas=GAMMAS):
    """Select strictly on validation; return predictions for that selected fit."""
    scores, best = {}, None
    for alpha in alphas:
        for gamma in gammas:
            val, test = predict_generalized_ridge(designs, prior, alpha=alpha,
                                                  gamma=gamma, floor_fraction=.05)
            score = mean_cell_nrmse(val, validation_target, scale)
            scores[f"alpha={alpha:g},gamma={gamma:g}"] = score
            if best is None or score < best[0]:
                best = (score, alpha, gamma, val, test)
    return best, scores


def bootstrap(predictions, target, scale, weights):
    aggregate, by_target, by_lead = {}, {}, {}
    for key, pred in predictions.items():
        squared = ((pred - target) / scale) ** 2
        cells = np.sqrt((weights @ squared.reshape(len(target), -1)).reshape(len(weights), 7, 24))
        aggregate[key] = cells.mean(axis=(1, 2))
        by_target[key] = cells.mean(axis=2)
        by_lead[key] = cells.mean(axis=1)
    return aggregate, by_target, by_lead


def shuffle_prior(prior, rng, prior_scope):
    if prior_scope == 'scalar7':
        return shuffled_centrality(prior, rng)
    if prior_scope != 'joint7':
        raise ValueError('Unknown calibration prior scope.')
    np.testing.assert_array_equal(prior, np.broadcast_to(prior[:1], prior.shape))
    # Preserve one shared source prior per lead. Independent target-wise shuffles
    # would introduce target-specific weights absent from the real joint game.
    one = np.stack([row[rng.permutation(prior.shape[2])] for row in prior[0]])
    return np.broadcast_to(one[None], prior.shape).copy()


def summarize(predictions, target, scale, weights, hyperparameters):
    aggregate, target_draws, lead_draws = bootstrap(predictions, target, scale, weights)
    scores = {key: mean_cell_nrmse(pred, target, scale) for key, pred in predictions.items()}
    result = {}
    for key, pred in predictions.items():
        cells = cell_rmse(pred, target, scale)
        entry = {"test_nrmse": scores[key], "test_acc": mean_cell_acc(pred, target),
                 "target_nrmse": cells.mean(axis=1).tolist(), "lead_nrmse": cells.mean(axis=0).tolist(),
                 "reduction_vs_frozen_percent": 100 * (1 - scores[key] / scores["frozen"]),
                 "target_reduction_vs_frozen_percent": (100 * (1 - cells.mean(axis=1) /
                    cell_rmse(predictions['frozen'], target, scale).mean(axis=1))).tolist(),
                 **hyperparameters.get(key, {})}
        for baseline in ("frozen", "uniform", "xi_shapley", "shap", "univariate", "ei_shapley"):
            entry["gain_over_" + baseline] = {
                "point": scores[baseline] - scores[key],
                "ci95": np.percentile(aggregate[baseline] - aggregate[key], [2.5, 97.5]).tolist(),
                "target_ci95": np.percentile(target_draws[baseline] - target_draws[key], [2.5, 97.5], axis=0).tolist(),
                "lead_ci95": np.percentile(lead_draws[baseline] - lead_draws[key], [2.5, 97.5], axis=0).tolist(),
            }
        entry["reduction_vs_frozen_ci95_percent"] = np.percentile(
            100 * (1 - aggregate[key] / aggregate['frozen']), [2.5, 97.5]).tolist()
        result[key] = entry
    return result, aggregate


def run(output=OUTPUT, random_repeats=200, prior_scope='joint7'):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(DEFAULT_INPUT / "model_inputs.npz", allow_pickle=False) as cache:
        history, all_target = cache["history"].astype(float), cache["targets"].astype(float)
        names = cache["mode_names"].astype(str).tolist()
        metadata = json.loads(str(cache["metadata"]))
    indices = target_indices(names)
    target = all_target[:, indices]
    if metadata.get("normalization_fit_period") != "1980-01/2003-12":
        raise ValueError("Expected fit-only upstream normalization.")
    with np.load(DEFAULT_INPUT / "modeformer_predictions.npz", allow_pickle=False) as cache:
        ensemble, dates = cache["ensemble_prediction"].astype(float), cache["target_dates"].astype(str)
        if names != cache["mode_names"].astype(str).tolist():
            raise ValueError("Forecast source names mismatch.")
    old = json.loads((DEFAULT_CALIBRATION / "summary.json").read_text())
    split = chronological_split(dates, **old["chronological_split"])
    if [len(split.fit), len(split.validation), len(split.test)] != [253, 36, 96]:
        raise ValueError("Unexpected chronological sample counts.")
    _, scale = target_scaling(target, split.fit)
    additive, _ = history_summaries(history)
    designs = prepare_designs(ensemble, target, additive, split)
    priors, prior_audit = load_priors(names, indices, prior_scope=prior_scope)
    searches, independent, independent_hyperparameters = {}, {}, {}
    for key in METHODS:
        best, scores = validation_search(designs, priors[key], target[split.validation], scale)
        score, alpha, gamma, _, prediction = best
        searches[key] = scores
        independent[key] = prediction
        independent_hyperparameters[key] = {"alpha": alpha, "gamma": gamma, "validation_nrmse": score}
        print(f"Validation selected {key}: alpha={alpha:g}, gamma={gamma:g}, score={score:.6f}", flush=True)
    # Uniform has no informative gamma; equal alpha search, no duplicate trials.
    best_uniform, searches["uniform"] = validation_search(designs, np.ones((7, 24, 11)),
        target[split.validation], scale, gammas=(1.,))
    independent['uniform'] = best_uniform[4]
    independent_hyperparameters['uniform'] = dict(alpha=best_uniform[1], gamma=1., validation_nrmse=best_uniform[0])
    uni_best, uni_scores = None, {}
    for alpha in (.01, .1, 1., 10., 100., 1000.):
        val, test = ridge_univariate(ensemble[:, indices], target, split, alpha)
        score = mean_cell_nrmse(val, target[split.validation], scale)
        uni_scores[str(alpha)] = score
        if uni_best is None or score < uni_best[0]:
            uni_best = (score, alpha, test)
    searches['univariate'] = uni_scores
    for key in ('frozen', 'univariate'):
        independent[key] = ensemble[split.test][:, indices] if key == 'frozen' else uni_best[2]
    independent_hyperparameters['univariate'] = dict(alpha=uni_best[1], validation_nrmse=uni_best[0])
    selected = independent_hyperparameters['xi_shapley']
    alpha, gamma = selected['alpha'], selected['gamma']
    shared = {key: independent[key] for key in ('frozen', 'univariate')}
    shared_hyperparameters = {'univariate': independent_hyperparameters['univariate']}
    original_parameters = {key: independent[key] for key in ('frozen', 'univariate')}
    for key in ('uniform', *METHODS):
        prior = np.ones((7, 24, 11)) if key == 'uniform' else priors[key]
        val, shared[key] = predict_generalized_ridge(designs, prior, alpha=alpha, gamma=gamma, floor_fraction=.05)
        _, original_parameters[key] = predict_generalized_ridge(designs, prior, alpha=30000., gamma=3., floor_fraction=.05)
        shared_hyperparameters[key] = dict(alpha=alpha, gamma=gamma,
            validation_nrmse=mean_cell_nrmse(val, target[split.validation], scale))
    with np.load(REFERENCE / 'evaluation_arrays.npz') as cache:
        np.testing.assert_allclose(target[split.test], cache['test_target'][:, indices], atol=1e-6, rtol=0)
        np.testing.assert_array_equal(ensemble[split.test][:, indices], cache['prediction_frozen'][:, indices])
        weights = cache['bootstrap_month_weights'].copy()
        for key in (('xi_shapley', 'ei_shapley', 'surd') if prior_scope == 'scalar7' else ('surd',)):
            np.testing.assert_allclose(original_parameters[key], cache['prediction_'+key][:, indices],
                                       atol=1e-10, rtol=0)
    test_target = target[split.test]
    shared_metrics, draws = summarize(shared, test_target, scale, weights, shared_hyperparameters)
    independent_metrics, _ = summarize(independent, test_target, scale, weights, independent_hyperparameters)
    original_metrics, _ = summarize(original_parameters, test_target, scale, weights, {})
    rng = np.random.default_rng(20261001)
    random_scores = []
    for repeat in range(random_repeats):
        _, pred = predict_generalized_ridge(designs, shuffle_prior(priors['xi_shapley'], rng, prior_scope),
            alpha=alpha, gamma=gamma, floor_fraction=.05)
        random_scores.append(mean_cell_nrmse(pred, test_target, scale))
    xi_score = shared_metrics['xi_shapley']['test_nrmse']
    result = dict(status='completed_candidate', information_prior_scope=prior_scope,
        target_mode_names=list(TARGETS), source_mode_names=names,
        target_indices=indices.tolist(), samples=dict(fit=253, validation=36, test=96),
        chronological_split=old['chronological_split'], prediction_shape=list(shared['xi_shapley'].shape),
        shared_hyperparameters=dict(alpha=alpha, gamma=gamma, floor_fraction=.05),
        search=dict(alphas=ALPHAS, gammas=GAMMAS, selection='mean validation nRMSE over exactly 7x24 cells',
                    validation_scores=searches, test_used_for_selection=False),
        parameter_mode='fixed_seven_target_xi', methods=shared_metrics,
        independently_tuned_methods=independent_metrics, original_parameter_methods=original_metrics,
        prior_audit=prior_audit,
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
            [DEFAULT_INPUT/'model_inputs.npz', DEFAULT_INPUT/'modeformer_predictions.npz',
             REFERENCE/'information_priors.npz', SHAP/'shap_priors.npz', REFERENCE/'evaluation_arrays.npz']},
        bootstrap=dict(replicates=len(weights), block_length_months=12, seed=20261001,
                       shared_original_weights=True, intervals='pointwise 95%, no multiplicity adjustment'),
        shuffled_xi_control=dict(repeats=random_repeats, scores=random_scores,
            parameter_mode='fixed_unshuffled_seven_target_validation_optimum_no_null_retuning',
            shuffle_unit=('one source permutation per lead shared by all outputs' if prior_scope == 'joint7'
                          else 'independent source permutation per target/lead'),
            interpretation='conditional label-shuffle rank; does not account for selection of real-prior hyperparameters',
            fraction_null_at_least_as_good=(1+int(np.count_nonzero(np.asarray(random_scores)<=xi_score)))/(1+random_repeats)),
        limitations=['Exploratory revision of a previously inspected test set.',
            'Shared parameters are selected for Xi; independently tuned comparison is also reported.',
            'Phi pair priors use 21 pairs among the seven target modes; extra four sources have zero raw pair score.',
            'Bootstrap conditions on fixed checkpoints/priors and uses the historical 12-month block convention.'],
        elapsed_seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    np.savez_compressed(output/'evaluation_arrays.npz',
        **{'prediction_'+k:v for k,v in shared.items()},
        **{'independently_tuned_prediction_'+k:v for k,v in independent.items()},
        **{'bootstrap_nrmse_'+k:v for k,v in draws.items()},
        test_target=test_target, target_scale=scale, bootstrap_month_weights=weights,
        target_mode_names=np.asarray(TARGETS), source_mode_names=np.asarray(names),
        test_issue_dates=split.issue_dates[split.test].astype(str))
    np.savez_compressed(output/'information_priors.npz', **priors)
    for key, entry in shared_metrics.items():
        print(f"{key}: test={entry['test_nrmse']:.6f}, gain={entry['reduction_vs_frozen_percent']:.2f}%", flush=True)
    print(f"Completed candidate in {result['elapsed_seconds']:.1f}s: {output}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--random-repeats', type=int, default=200)
    parser.add_argument('--prior-scope', choices=('joint7','scalar7'), default='joint7')
    args = parser.parse_args()
    run(args.output_dir, args.random_repeats, args.prior_scope)
