#!/usr/bin/env python3
"""Direct grouped permutation-SHAP on frozen UniCM; fit-only pilot calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_unicm_target_xi_shapley_prior import DEFAULT_INPUT, DEFAULT_CALIBRATION
from scripts.run_unicm_synergy_guided_forecast import (
    chronological_split, history_summaries, target_scaling, cell_rmse,
)
from scripts.run_unicm_synergy_regularized_calibration import prepare_designs
from scripts.run_unicm_information_prior_comparison import fit_prior, COLORS, LABELS
from scripts.run_unicm_observational_prior_comparison import metric_entry, METHOD_LABELS
from scripts.unicm_peid_syn_analysis import (
    MODE_NAMES, resolve_checkpoint_paths, load_unicm_model,
    predict_modeformer_all_modes_from_history, sample_full_history_mode_inputs,
)

REFERENCE = ROOT / 'results/unicm_observational_prior_comparison_fit253_surd_allorders'
OUTPUT = ROOT / 'results/unicm_frozen_shap_prior_fit253_pilot'
SHAP_COLOR = '#BE6B32'
SHAP_LABEL = 'SHAP (direct grouped pilot)'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def permutation_masks(permutations: np.ndarray) -> np.ndarray:
    """Union of full forward/reverse chains, with shared endpoints evaluated once."""
    masks = {0}
    n = permutations.shape[1]
    for order in permutations:
        if sorted(order.tolist()) != list(range(n)):
            raise ValueError('A permutation must contain every group exactly once.')
        for chain in (order, order[::-1]):
            mask = 0
            for source in chain:
                mask |= 1 << int(source)
                masks.add(mask)
    return np.array(sorted(masks), dtype=np.int64)


def masked_histories(foreground: np.ndarray, background: np.ndarray,
                     masks: np.ndarray) -> np.ndarray:
    """Shape [foreground, background, coalition, month, mode]; whole history groups."""
    n = foreground.shape[-1]
    if foreground.shape[1:] != background.shape[1:]:
        raise ValueError('Foreground/background history blocks must have the same shape.')
    keep = ((masks[:, None] >> np.arange(n)) & 1).astype(bool)
    return np.where(keep[None, None, :, None, :],
                    foreground[:, None, None, :, :], background[None, :, None, :, :])


def permutation_shap(values: np.ndarray, masks: np.ndarray,
                     permutations: np.ndarray) -> np.ndarray:
    """Local output attribution [foreground, source, lead, target].

    Coalition predictions have shape [foreground, background, mask, lead, target].
    Average signed values over the empirical background before taking importance.
    """
    averaged = np.mean(values, axis=1)
    lookup = {int(mask): i for i, mask in enumerate(masks)}
    n = permutations.shape[1]
    phi = np.zeros((len(values), n, *values.shape[-2:]), dtype=np.float64)
    for order in permutations:
        for chain in (order, order[::-1]):
            before = 0
            for source in chain:
                after = before | (1 << int(source))
                phi[:, source] += averaged[:, lookup[after]] - averaged[:, lookup[before]]
                before = after
    phi /= 2 * len(permutations)
    expected = averaged[:, lookup[(1 << n) - 1]] - averaged[:, lookup[0]]
    np.testing.assert_allclose(phi.sum(axis=1), expected, rtol=1e-11, atol=1e-10)
    if not np.isfinite(phi).all():
        raise ValueError('Nonfinite direct SHAP attribution.')
    return phi


def importance(phi: np.ndarray) -> np.ndarray:
    # [foreground, source, lead, target] -> [target, lead, source].
    return np.mean(np.abs(phi), axis=0).transpose(2, 1, 0)


def load_data():
    with np.load(DEFAULT_INPUT / 'model_inputs.npz') as d:
        history, target = d['history'].astype(float), d['targets'].astype(float)
        names = d['mode_names'].astype(str).tolist()
        metadata = json.loads(str(d['metadata']))
    with np.load(DEFAULT_INPUT / 'modeformer_predictions.npz') as d:
        ensemble, dates = d['ensemble_prediction'].astype(float), d['target_dates'].astype(str)
        if names != d['mode_names'].astype(str).tolist():
            raise ValueError('Prediction mode order mismatch.')
    expected_names = ['ENSO' if name == 'nino' else name for name in MODE_NAMES]
    if names != expected_names or metadata['normalization_fit_period'] != '1980-01/2003-12':
        raise ValueError('Mode order or fit-only normalization mismatch.')
    calibration = json.loads((DEFAULT_CALIBRATION / 'summary.json').read_text())
    split = chronological_split(dates, **calibration['chronological_split'])
    if tuple(map(len, (split.fit, split.validation, split.test))) != (253, 36, 96):
        raise ValueError('Chronological split mismatch.')
    if not all(np.isfinite(x).all() for x in (history, target, ensemble)):
        raise ValueError('Nonfinite existing inputs.')
    return history, target, ensemble, split, names


def plot_comparison(result: dict, predictions: dict, target, scale, output: Path):
    import matplotlib.pyplot as plt
    colors = {**COLORS, 'shap': SHAP_COLOR}
    labels = {**LABELS, **METHOD_LABELS, 'shap': SHAP_LABEL}
    order = ('frozen', 'xi_shapley', 'shap', 'univariate', 'uniform', 'phi_r',
             'phi_wms', 'phi_si', 'causal_density', 'surd', 'ei_shapley')
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig = plt.figure(figsize=(12.4, 8.4), layout='constrained')
        grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
        ax, diff, lead = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
        for i, key in enumerate(order):
            score = result['methods'][key]['test_nrmse']
            ax.scatter(score, i, color=colors[key], s=30, zorder=3)
            ax.annotate(f'{score:.6f}', (score, i), xytext=(6, 0), textcoords='offset points',
                        fontsize=7, va='center')
        ax.set_yticks(range(len(order)), [labels[k] for k in order])
        ax.invert_yaxis()
        scores = [result['methods'][k]['test_nrmse'] for k in order]
        ax.set_xlim(min(scores) - .015, max(scores) + .022)
        ax.set_xlabel('Test normalized RMSE (lower is better)')
        ax.grid(axis='x', alpha=.15)
        for i, key in enumerate(('uniform', 'xi_shapley', 'surd', 'ei_shapley')):
            gain = result['methods']['shap']['gain_over_' + key]
            diff.plot(gain['ci95'], [i, i], lw=1.5, color=SHAP_COLOR)
            diff.scatter(gain['point'], i, s=30, color=SHAP_COLOR, zorder=3)
        diff.axvline(0, color='#777777', ls='--', lw=.7)
        diff.set_yticks(range(4), ['SHAP vs ' + labels[k] for k in ('uniform','xi_shapley','surd','ei_shapley')])
        diff.invert_yaxis()
        diff.set_xlabel('nRMSE gain from SHAP (positive favors SHAP)\nPaired time-block bootstrap: pointwise 95% intervals')
        diff.grid(axis='x', alpha=.15)
        uniform = cell_rmse(predictions['uniform'], target, scale).mean(axis=0)
        for key in ('xi_shapley', 'shap', 'surd', 'ei_shapley'):
            gain = uniform - cell_rmse(predictions[key], target, scale).mean(axis=0)
            lead.plot(np.arange(1,25), gain, label=labels[key], color=colors[key], lw=1.5)
        lead.axhline(0, color='#777777', ls='--', lw=.7)
        lead.set(xlabel='Prediction lead (months)', ylabel='nRMSE gain over uniform ridge',
                 xticks=[1,4,8,12,16,20,24], xlim=(1,24))
        lead.grid(axis='y', alpha=.15)
        lead.legend(loc='center left', bbox_to_anchor=(1.02,.5), frameon=False)
        for letter, axis in zip('abc', (ax,diff,lead)):
            axis.set_title(letter, loc='left', fontsize=11, fontweight='bold')
        fig.suptitle(r'Direct grouped SHAP pilot | shared $\alpha=30000$, $\gamma=3$ | all 96 test months', fontsize=10)
        fig.savefig(output, dpi=350, bbox_inches='tight')
        plt.close(fig)


def evaluate(prior, half_prior, args, identity, diagnostics):
    history, target, ensemble, split, names = load_data()
    _, scale = target_scaling(target, split.fit)
    additive, _ = history_summaries(history)
    designs = prepare_designs(ensemble, target, additive, split)
    reference = json.loads((REFERENCE / 'summary.json').read_text())
    result = {'method': 'direct_grouped_marginal_permutation_shap_pilot', 'identity': identity,
              'diagnostics': diagnostics, 'shared_hyperparameters': reference['shared_hyperparameters'],
              'parameter_mode': 'fixed_xi', 'samples': reference['samples'], 'mode_names': names,
              'methods': dict(reference['methods']), 'limitations': [
                  'Exploratory SHAP budget: 16 foregrounds, four empirical fit backgrounds, two antithetic permutation pairs.',
                  'Natural marginal replacement SHAP differs from the independent uniform EI/Xi distribution.',
                  'Ordinary output SHAP importance is not interaction-only SHAP or information in bits.',
                  'Hyperparameters were originally selected for Xi; no method-specific tuning.',
                  'Time-block intervals cover test-month resampling, not SHAP Monte Carlo estimation error.']}
    if result['shared_hyperparameters'] != {'alpha': 30000., 'gamma': 3.}:
        raise ValueError('Xi-selected parameter reference changed.')
    with np.load(REFERENCE / 'evaluation_arrays.npz') as d:
        predictions = {k.removeprefix('prediction_'): d[k].copy() for k in d.files if k.startswith('prediction_')}
        np.testing.assert_array_equal(target[split.test], d['test_target'])
        np.testing.assert_array_equal(scale, d['target_scale'])
        counts = d['bootstrap_month_weights'].copy()
        boot = {k.removeprefix('bootstrap_nrmse_'): d[k].copy() for k in d.files if k.startswith('bootstrap_nrmse_')}
    test_target = target[split.test]
    fits = {}
    for key, c in (('shap', prior), ('shap_half_permutations', half_prior)):
        alpha, gamma, tuning, _, pred = fit_prior(designs, c, target[split.validation], scale,
            alphas=[30000.], gammas=[3.], floor_fraction=.05, fixed_parameters=result['shared_hyperparameters'])
        fits[key] = pred
        entry = metric_entry(pred, test_target, scale)
        entry.update(alpha=alpha, gamma=gamma, validation_scores=tuning)
        result['methods'][key] = entry
    predictions['shap'] = fits['shap']
    squared = ((fits['shap'] - test_target) / scale)**2
    cells = np.sqrt((counts @ squared.reshape(len(test_target), -1)).reshape(len(counts),11,24))
    boot['shap'] = cells.mean(axis=(1,2))
    shap_lead_boot = cells.mean(axis=1)
    for key in predictions:
        gain = boot[key] - boot['shap']
        result['methods']['shap']['gain_over_' + key] = {
            'point': result['methods'][key]['test_nrmse'] - result['methods']['shap']['test_nrmse'],
            'ci95': np.percentile(gain, [2.5,97.5]).tolist()}
    result['methods']['shap']['rmse_reduction_vs_frozen_percent'] = 100*(1-result['methods']['shap']['test_nrmse']/result['methods']['frozen']['test_nrmse'])
    full_score = result['methods']['shap']['test_nrmse']
    half_score = result['methods']['shap_half_permutations']['test_nrmse']
    share = prior / prior.sum(axis=-1, keepdims=True)
    half_share = half_prior / half_prior.sum(axis=-1, keepdims=True)
    result['stability'] = {
        'half_permutation_test_nrmse': half_score,
        'full_permutation_test_nrmse': full_score,
        'full_minus_half_test_nrmse': full_score-half_score,
        'source_share_absolute_difference_mean': float(np.mean(np.abs(share-half_share))),
        'source_share_absolute_difference_maximum': float(np.max(np.abs(share-half_share))),
        'parameter_selection_based_on_this_diagnostic': False}
    from scipy.stats import spearmanr
    with np.load(REFERENCE/'information_priors.npz') as d:
        xi_prior = d['xi_shapley']
    xi_share = xi_prior / xi_prior.sum(axis=-1, keepdims=True)
    correlations = np.array([spearmanr(a,b).statistic for a,b in
                            zip(share.reshape(-1,11),xi_share.reshape(-1,11))])
    result['weight_comparison'] = {
        'source_share_average_rule': 'Normalize sources within each target-lead cell, then equally average over all 264 cells.',
        'shap_mean_source_share': dict(zip(names,share.mean(axis=(0,1)).tolist())),
        'xi_mean_source_share': dict(zip(names,xi_share.mean(axis=(0,1)).tolist())),
        'cell_source_rank_spearman_median': float(np.median(correlations)),
        'cell_source_rank_spearman_range': [float(correlations.min()),float(correlations.max())]}
    result['bootstrap'] = {**reference['bootstrap'], 'source': str(REFERENCE/'evaluation_arrays.npz'),
        'paired_existing_weights_reused': True}
    result['syn_validity'] = {'new_syn_estimation': False, 'existing_syn_audit': reference['centrality_audit'],
                            'shap_nonnegative_mapping': 'mean absolute local output attribution; not Syn clipping'}
    np.savez_compressed(args.output_dir/'evaluation_arrays.npz',
        **{'prediction_'+k: v for k,v in predictions.items()},
        prediction_shap_half_permutations=fits['shap_half_permutations'],
        **{'bootstrap_nrmse_'+k: v for k,v in boot.items()},
        bootstrap_lead_nrmse_shap=shap_lead_boot, bootstrap_month_weights=counts,
        test_target=test_target, target_scale=scale)
    plot_comparison(result, predictions, test_target, scale, args.output_dir/'shap_prior_comparison.png')
    (args.output_dir/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'SHAP test nRMSE': full_score, 'SHAP half-budget': half_score,
                      'gain_over_xi': result['methods']['shap']['gain_over_xi_shapley']},ensure_ascii=False), flush=True)


def run(args):
    import torch
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history, _, _, split, _ = load_data()
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(split.fit, size=args.foregrounds+args.backgrounds, replace=False)
    fg_indices, bg_indices = chosen[:args.foregrounds], chosen[args.foregrounds:]
    # A group is a mode's complete 12-month history, matching calibration priors.
    foreground = history[fg_indices].transpose(0,2,1).astype(np.float32)
    background = history[bg_indices].transpose(0,2,1).astype(np.float32)
    permutations = np.array([rng.permutation(11) for _ in range(args.permutations)])
    masks = permutation_masks(permutations)
    paths = resolve_checkpoint_paths(ROOT/'data/UniCM-checkpoint/src/experiments', (1,2,3))
    identity = {'algorithm_version': 'grouped_marginal_antithetic_shap_v1',
        'input_sha256': digest(DEFAULT_INPUT/'model_inputs.npz'),
        'evaluation_sha256': digest(REFERENCE/'evaluation_arrays.npz'),
        'checkpoint_sha256': {str(k):digest(v) for k,v in paths.items()},
        'seed': args.seed, 'foreground_fit_indices':fg_indices.tolist(),
        'background_fit_indices': bg_indices.tolist(), 'permutations':permutations.tolist(),
        'masks': masks.tolist(), 'start_month':0, 'device':'cpu', 'torch_threads':1,
        'foreground_samples':args.foregrounds, 'background_samples':args.backgrounds,
        'permutation_pairs':args.permutations, 'fit_only_prior':True,
        'prior_units':'model-output units; used only through within-cell relative weights'}
    identity_json = json.dumps(identity, sort_keys=True)
    config_path = args.output_dir/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != identity:
        raise ValueError('Output cache provenance mismatch; choose a fresh directory.')
    config_path.write_text(json.dumps(identity,indent=2)+'\n')
    batch = masked_histories(foreground,background,masks)
    expected_shape = (args.foregrounds,args.backgrounds,len(masks),24,11)
    print(f'Run: {len(masks)} unique coalition masks, {np.prod(expected_shape[:3])} histories/checkpoint; 3 checkpoints.',flush=True)
    by_cp, half_cp, diagnostics = [], [], {}
    for cp, path in paths.items():
        cache = args.output_dir/f'checkpoint{cp}_coalition_predictions.npz'
        if cache.exists():
            with np.load(cache) as d:
                if str(d['config']) != identity_json:
                    raise ValueError('Checkpoint coalition cache provenance mismatch.')
                values = d['predictions'].copy()
                diagnostics[str(cp)] = json.loads(str(d['diagnostics']))
            print(f'Checkpoint {cp}: reusing direct coalition predictions.',flush=True)
        else:
            print(f'Checkpoint {cp}: loading frozen weights and verifying cached prediction identity.',flush=True)
            model = load_unicm_model(path,'cpu')
            probe_history = sample_full_history_mode_inputs(n_samples=16384,intervention_bound=4.,seed=20260901)[:8]
            probe_start = time.perf_counter()
            probe = predict_modeformer_all_modes_from_history(model,probe_history,device='cpu',batch_size=8,start_month=0)
            old_cache = ROOT/f'results/unicm_xi_hierarchy_uniform_n16384/cache/checkpoint{cp}_samples16384_sampling20260901_bound4_fullhist12_start0_cpu.npz'
            with np.load(old_cache) as d:
                probe_error = float(np.max(np.abs(probe-d['all_mode_targets'][:8])))
            if probe_error > 1e-4:
                raise ValueError(f'Frozen inference differs from existing reference by {probe_error:g}.')
            diagnostics[str(cp)] = {'probe_maximum_cache_error':probe_error,
                                    'probe_seconds':time.perf_counter()-probe_start}
            print(f'Checkpoint {cp}: probe passed, maximum error {probe_error:.3g}; direct SHAP inference begins.',flush=True)
            start = time.perf_counter()
            values = predict_modeformer_all_modes_from_history(model,batch.reshape(-1,12,11),
                device='cpu',batch_size=128,start_month=0).reshape(expected_shape)
            diagnostics[str(cp)]['coalition_inference_seconds'] = time.perf_counter()-start
            np.savez_compressed(cache,predictions=values,config=identity_json,
                                diagnostics=json.dumps(diagnostics[str(cp)]))
            del model
            print(f'Checkpoint {cp}: completed and cached ({diagnostics[str(cp)]["coalition_inference_seconds"]:.1f}s).',flush=True)
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError('Invalid direct coalition prediction cache.')
        phi = permutation_shap(values,masks,permutations)
        half_phi = permutation_shap(values,masks,permutations[:max(1,args.permutations//2)])
        by_cp.append(importance(phi)); half_cp.append(importance(half_phi))
        averaged = values.mean(axis=1)
        error = np.max(np.abs(phi.sum(axis=1)-(averaged[:,-1]-averaged[:,0])))
        diagnostics[str(cp)]['maximum_local_additivity_error'] = float(error)
    prior,half_prior = np.mean(by_cp,axis=0),np.mean(half_cp,axis=0)
    if np.any(prior.sum(axis=-1)<=0):
        raise ValueError('A target-lead SHAP importance vector has zero mass.')
    np.savez_compressed(args.output_dir/'shap_priors.npz',shap=prior,half_permutations=half_prior,
                        by_checkpoint=np.array(by_cp),config=identity_json)
    evaluate(prior,half_prior,args,identity,diagnostics)
    print('Completed direct grouped SHAP pilot.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    parser.add_argument('--seed',type=int,default=20261002)
    parser.add_argument('--foregrounds',type=int,default=16)
    parser.add_argument('--backgrounds',type=int,default=4)
    parser.add_argument('--permutations',type=int,default=2)
    args = parser.parse_args()
    if min(args.foregrounds,args.backgrounds,args.permutations) < 1:
        parser.error('SHAP sampling budgets must be positive.')
    run(args)
