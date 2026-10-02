#!/usr/bin/env python3
"""Compare intervention EI/Xi against fit-only observational priors and full SURD."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_unicm_information_prior_comparison import (
    COLORS, LABELS, EXCLUDED_METHODS, fit_prior, paired_bootstrap, plot_results,
)
from scripts.run_unicm_target_xi_shapley_prior import (
    DEFAULT_INPUT, DEFAULT_CALIBRATION, DEFAULT_CACHE, DEFAULT_OUTPUT as XI_REFERENCE,
)
from scripts.run_unicm_synergy_guided_forecast import (
    chronological_split, target_scaling, history_summaries, mean_cell_nrmse,
    mean_cell_acc, cell_rmse,
)
from scripts.run_unicm_synergy_regularized_calibration import prepare_designs
from scripts.unicm_peid_syn_analysis import MODE_NAMES, sample_full_history_mode_inputs
from scripts.unicm_observational_information import observational_prior, intervention_prior

OUTPUT = ROOT / 'results/unicm_observational_prior_comparison_fit253_surd_allorders'
PREVIOUS = ROOT / 'results/unicm_information_prior_comparison_literature_shared_xi_n16384'
METHODS = ('xi_shapley', 'phi_r', 'ei_shapley', 'phi_wms', 'phi_si', 'causal_density', 'surd')
METHOD_LABELS = {
    **LABELS,
    'phi_r': r'$\Phi^R$ (observed pair)',
    'phi_wms': r'$\Phi^{\mathrm{WMS}}$ (observed pair)',
    'phi_si': r'$\Phi_{\mathrm{SI}}$ (observed pair)',
    'causal_density': 'Causal density (observed)',
    'surd': 'Gaussian SURD (all orders)',
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_entry(prediction: np.ndarray, target: np.ndarray, scale: np.ndarray) -> dict:
    return dict(test_nrmse=mean_cell_nrmse(prediction, target, scale),
                test_acc=mean_cell_acc(prediction, target),
                lead_nrmse=cell_rmse(prediction, target, scale).mean(axis=0).tolist(),
                target_nrmse=cell_rmse(prediction, target, scale).mean(axis=1).tolist())


def summarize_timing(trials: list[dict]) -> dict:
    result = {'repetitions': len(trials), 'trials': trials}
    for key in trials[0]:
        values = [row[key] for row in trials]
        result[key] = {'median': float(np.median(values)), 'minimum': min(values), 'maximum': max(values)}
    return result


def plot_runtime(timing: dict, output: Path) -> None:
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), layout='constrained',
                                 gridspec_kw={'width_ratios': [1.3, 1]})
        for row, key in enumerate(METHODS):
            values = timing['information_core'][key]['core_total_seconds']
            axes[0].plot([values['minimum'], values['maximum']], [row, row], color=COLORS[key], lw=1.5)
            axes[0].scatter(values['median'], row, color=COLORS[key], s=30)
            axes[0].annotate(f"{values['median']:.4f}s", (values['maximum'], row),
                             xytext=(6, 0), textcoords='offset points', va='center', fontsize=8)
        axes[0].set_yticks(range(len(METHODS)), [METHOD_LABELS[k] for k in METHODS])
        axes[0].invert_yaxis()
        axes[0].set_xscale('log')
        vals = [timing['information_core'][k]['core_total_seconds']['median'] for k in METHODS]
        axes[0].set_xlim(min(vals) / 1.8, max(vals) * 2.6)
        axes[0].set_xlabel('Fresh information + attribution time (s, log scale)')
        axes[0].grid(axis='x', alpha=.15)
        axes[0].set_title('a   Three serial repetitions', loc='left', fontsize=10)
        stages = ['density_fit_seconds', 'information_seconds', 'attribution_seconds']
        stage_labels = ['Density fit / input preparation', 'Information values', 'Source attribution']
        palette = ['#798FA6', '#D5A150', '#42877E']
        bottom = np.zeros(len(METHODS))
        for stage, label, color in zip(stages, stage_labels, palette):
            values = np.asarray([timing['information_core'][k][stage]['median'] for k in METHODS])
            axes[1].barh(range(len(METHODS)), values, left=bottom, color=color, label=label)
            bottom += values
        axes[1].set_yticks(range(len(METHODS)), [r'$\Xi$', r'$\Phi^R$', 'EI', 'WMS', 'SI', 'CD', 'SURD'])
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Median stage time (s)')
        axes[1].set_title('b   Stage medians', loc='left', fontsize=10)
        axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.16), frameon=False, fontsize=8)
        fig.suptitle('EI/Xi: 3 x 16,384 intervention histories; observed priors: 253 fit pairs', fontsize=10)
        fig.savefig(output, dpi=350, bbox_inches='tight')
        plt.close(fig)


def benchmark_inference(history: np.ndarray, cached_outputs: list[np.ndarray], n_probe: int = 128) -> dict:
    """Measured small-batch inference, with full-run projection explicitly labeled."""
    from scripts.unicm_peid_syn_analysis import (
        load_unicm_model, resolve_checkpoint_paths, predict_modeformer_all_modes_from_history,
    )
    import torch
    torch.set_num_threads(1)
    paths = resolve_checkpoint_paths(ROOT / 'data/UniCM-checkpoint/src/experiments', (1, 2, 3))
    result = {'device': 'cpu', 'torch_threads': torch.get_num_threads(), 'batch_size': n_probe,
              'probe_histories_per_checkpoint': n_probe, 'full_histories_per_checkpoint': len(history),
              'full_generation_status': 'extrapolated from three repeated fixed-size batches; NOT full-run measured',
              'checkpoints': {}}
    for index, seed in enumerate((1, 2, 3)):
        start = time.perf_counter()
        model = load_unicm_model(paths[seed], 'cpu')
        load_seconds = time.perf_counter() - start
        seconds, max_difference = [], 0.0
        for repetition in range(3):
            start = time.perf_counter()
            output = predict_modeformer_all_modes_from_history(model, history[:n_probe],
                        device='cpu', batch_size=n_probe, start_month=0)
            seconds.append(time.perf_counter() - start)
            max_difference = max(max_difference, float(np.max(np.abs(output - cached_outputs[index][:n_probe]))))
        if max_difference > 1e-4:
            raise ValueError(f'Inference probe cache identity differs by {max_difference:g}.')
        projection = np.median(seconds) * len(history) / n_probe
        result['checkpoints'][str(seed)] = {'load_seconds_measured': load_seconds,
             'batch_seconds_measured': seconds, 'batch_seconds_median': float(np.median(seconds)),
             'full_inference_seconds_projected': float(projection), 'maximum_cache_difference': max_difference}
        print(f'Inference checkpoint {seed}: batch median {np.median(seconds):.3f}s; '
              f'full-run projection {projection:.1f}s (not full measured)', flush=True)
        del model
    result['full_generation_seconds_projected'] = sum(
        row['full_inference_seconds_projected'] + row['load_seconds_measured']
        for row in result['checkpoints'].values())
    return result


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(DEFAULT_INPUT / 'model_inputs.npz') as archive:
        history = archive['history'].astype(float)
        target = archive['targets'].astype(float)
        names = archive['mode_names'].astype(str).tolist()
        metadata = json.loads(str(archive['metadata']))
    with np.load(DEFAULT_INPUT / 'modeformer_predictions.npz') as archive:
        ensemble = archive['ensemble_prediction'].astype(float)
        dates = archive['target_dates'].astype(str)
        prediction_names = archive['mode_names'].astype(str).tolist()
    expected_names = ['ENSO' if name == 'nino' else name for name in MODE_NAMES]
    if names != expected_names or prediction_names != names:
        raise ValueError('Observed and intervention mode order mismatch.')
    if metadata.get('normalization_fit_period') != '1980-01/2003-12':
        raise ValueError('Expected fit-only upstream normalization.')
    if not all(np.isfinite(value).all() for value in (history, target, ensemble)):
        raise ValueError('Nonfinite observed inputs or evaluation predictions.')
    calibration = json.loads((DEFAULT_CALIBRATION / 'summary.json').read_text())
    split = chronological_split(dates, **calibration['chronological_split'])
    if [len(split.fit), len(split.validation), len(split.test)] != [253, 36, 96]:
        raise ValueError('Unexpected chronological split.')
    observed_history, observed_target = history[split.fit], target[split.fit]
    source_generation_start = time.perf_counter()
    intervention_history = sample_full_history_mode_inputs(n_samples=16384, intervention_bound=4., seed=20260901)
    source_generation_seconds = time.perf_counter() - source_generation_start
    paths = [DEFAULT_CACHE / f'checkpoint{seed}_samples16384_sampling20260901_bound4_fullhist12_start0_cpu.npz'
             for seed in (1, 2, 3)]
    cache_start = time.perf_counter()
    outputs = []
    for path in paths:
        with np.load(path) as archive:
            outputs.append(archive['all_mode_targets'].astype(float))
    prediction_cache_read_seconds = time.perf_counter() - cache_start
    reference_cache = PREVIOUS / 'information_centralities.npz'
    with np.load(reference_cache) as archive:
        expected = {key: archive[key].mean(axis=0) for key in ('xi_shapley', 'ei_shapley')}
    trials = {key: [] for key in METHODS}
    priors, audits, native_arrays = {}, {}, {}
    rng = np.random.default_rng(20261002)
    for repetition in range(args.timing_repetitions):
        for key in rng.permutation(METHODS):
            if key in ('xi_shapley', 'ei_shapley'):
                prior, audit = intervention_prior(intervention_history, outputs, key)
                difference = float(np.max(np.abs(prior - expected[key])))
                if difference > 1e-10:
                    raise ValueError(f'{key} cache reproduction differs by {difference:g} bit.')
                audit['maximum_reference_cache_difference_bit'] = difference
                native = {}
            else:
                prior, native, audit = observational_prior(observed_history, observed_target, key)
            trials[key].append(audit['timing'])
            if repetition == 0:
                priors[key], audits[key] = prior, audit
                native_arrays.update({key + '_' + name: value for name, value in native.items()
                                      if isinstance(value, np.ndarray)})
            else:
                np.testing.assert_array_equal(prior, priors[key])
            print(f'Core repetition {repetition + 1}: {key} '
                  f'{audit["timing"]["core_total_seconds"]:.4f}s', flush=True)
    timing = {'hardware': {'system': platform.platform(), 'machine': platform.machine(),
                          'processor': platform.processor(), 'python': platform.python_version(),
                          'numpy': np.__version__, 'thread_environment': {k: os.environ.get(k) for k in
                           ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}},
              'protocol': 'Serial, randomized method order, fresh recomputation; density + information + attribution; I/O and inference separate.',
              'information_core': {key: summarize_timing(trials[key]) for key in METHODS},
              'intervention_sampling_seconds_measured': source_generation_seconds,
              'prediction_cache_read_seconds_measured': prediction_cache_read_seconds,
              'sample_counts': {'intervention_per_checkpoint': 16384, 'checkpoints': 3, 'observational_fit_pairs': 253},
              'common_downstream': {}}
    if not args.skip_inference_probe:
        probe_cache = args.output_dir / 'inference_timing.json'
        cache_identity = {'probe_samples': args.inference_probe_samples,
                          'prediction_cache_sha256': [sha256(p) for p in paths],
                          'hardware': timing['hardware']}
        if probe_cache.exists():
            saved_probe = json.loads(probe_cache.read_text())
            if saved_probe['identity'] != cache_identity:
                raise ValueError('Inference timing cache configuration changed; use a new output directory.')
            timing['inference'] = saved_probe['measurement']
            timing['inference']['timing_cache_reused'] = True
        else:
            timing['inference'] = benchmark_inference(intervention_history, outputs, args.inference_probe_samples)
            probe_cache.write_text(json.dumps({'identity': cache_identity, 'measurement': timing['inference']}, indent=2) + '\n')
    else:
        timing['inference'] = {'full_generation_status': 'not measured; cache I/O is not inference cost'}
    # Free large intervention arrays before downstream calibration and bootstrap.
    del outputs, intervention_history
    _, scale = target_scaling(target, split.fit)
    additive, _ = history_summaries(history)
    start = time.perf_counter()
    designs = prepare_designs(ensemble, target, additive, split)
    timing['common_downstream']['design_preparation_seconds'] = time.perf_counter() - start
    reference = json.loads((XI_REFERENCE / 'summary.json').read_text())
    fixed = reference['selected_hyperparameters']
    if fixed != {'alpha': 30000.0, 'gamma': 3.0}:
        raise ValueError('Original Xi-selected parameters changed.')
    test_target = target[split.test]
    predictions = {'frozen': ensemble[split.test]}
    with np.load(DEFAULT_CALIBRATION / 'evaluation_arrays.npz') as archive:
        np.testing.assert_allclose(archive['test_target'], test_target, atol=1e-6, rtol=0)
        predictions['univariate'] = archive['prediction_univariate'].astype(float)
    result = {'status': 'completed', 'parameter_mode': 'fixed_xi', 'shared_hyperparameters': fixed,
              'included_methods': list(METHODS), 'excluded_methods': list(EXCLUDED_METHODS),
              'method_labels': {key: METHOD_LABELS[key] for key in METHODS},
              'estimator': 'observed-correlated-affine-tm + Gaussian full-coalition SURD; intervention affine-tm EI/Xi',
              'samples': {'fit': 253, 'validation': 36, 'test': 96}, 'mode_names': names,
              'chronological_split': calibration['chronological_split'], 'input_preprocessing': metadata,
              'prior_distributions': {key: ('independent_uniform_intervention_history_and_frozen_output' if key in
                   ('xi_shapley', 'ei_shapley') else 'fit_only_natural_history_and_actual_future') for key in METHODS},
              'prior_data_audit': {'history_source': str(DEFAULT_INPUT / 'model_inputs.npz'),
                   'input_sha256': sha256(DEFAULT_INPUT / 'model_inputs.npz'),
                   'fit_indices': split.fit.tolist(), 'last_fit_issue': str(split.issue_dates[split.fit[-1]]),
                   'latest_information_target': str(np.max(dates[split.fit].astype('datetime64[M]'))),
                   'source_history_months': 12, 'actual_target_leads': list(range(1, 25)),
                   'uses_validation_or_test_for_prior': False, 'uses_model_predictions_for_observational_prior': False,
                   'source_covariance_identity_replacement': False, 'observed_covariance_ridge': 1e-6,
                   'surd_nonempty_coalition_count': 2047, 'surd_maximum_order': 11,
                   'surd_scope': 'all-order Gaussian-density SURD; synergy atoms split equally among participating sources',
                   'intervention_prediction_cache_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in paths}},
              'centrality_audit': audits, 'methods': {}, 'bootstrap': {'replicates': 4000,
                   'block_length_months': 12, 'seed': 20261001, 'paired_shared_indices': True,
                   'intervals': 'pointwise percentile 95%; not multiplicity adjusted'},
              'historical_intervention_comparison': str(PREVIOUS / 'summary.json'),
              'runtime': timing, 'figure_panel_parameter_modes': {'a': 'fixed_xi', 'b': 'fixed_xi', 'c': 'fixed_xi'}}
    for key in ('uniform', *METHODS):
        start = time.perf_counter()
        centrality = np.ones((11, 24, 11)) if key == 'uniform' else priors[key]
        alpha, gamma, tuning, _, prediction = fit_prior(designs, centrality, target[split.validation], scale,
             alphas=[30000.], gammas=[3.], floor_fraction=.05, fixed_parameters=fixed)
        timing['common_downstream'][key + '_calibration_seconds'] = time.perf_counter() - start
        predictions[key] = prediction
        result['methods'][key] = {'alpha': alpha, 'gamma': gamma, 'validation_scores': tuning,
                                 'validation_nrmse': next(iter(tuning.values()))}
    if abs(mean_cell_nrmse(predictions['xi_shapley'], test_target, scale)
           - reference['test_nrmse']['target_xi_shapley_prior']) > 1e-9:
        raise ValueError('Unchanged Xi calibration reference did not reproduce.')
    start = time.perf_counter()
    aggregate, lead_bootstrap, counts = paired_bootstrap(predictions, test_target, scale, 4000, 12, 20261001)
    timing['common_downstream']['paired_bootstrap_all_methods_seconds'] = time.perf_counter() - start
    scores = {key: mean_cell_nrmse(value, test_target, scale) for key, value in predictions.items()}
    for key, prediction in predictions.items():
        entry = result['methods'].setdefault(key, {})
        entry.update(metric_entry(prediction, test_target, scale))
        entry['rmse_reduction_vs_frozen_percent'] = 100 * (1 - scores[key] / scores['frozen'])
        for baseline in ('uniform', 'xi_shapley'):
            gain = aggregate[baseline] - aggregate[key]
            entry['gain_over_' + ('xi' if baseline == 'xi_shapley' else baseline)] = {
                'point': scores[baseline] - scores[key], 'ci95': np.percentile(gain, [2.5, 97.5]).tolist(),
                'lead_ci95': np.percentile(lead_bootstrap[baseline] - lead_bootstrap[key], [2.5, 97.5], axis=0).tolist()}
        print(f'{key}: test nRMSE={scores[key]:.9f}', flush=True)
    old = json.loads((PREVIOUS / 'summary.json').read_text())
    result['distribution_revision_deltas'] = {key: {'old_intervention_nrmse': old['methods'][key]['test_nrmse'],
        'new_observational_nrmse': scores[key], 'new_minus_old': scores[key] - old['methods'][key]['test_nrmse']}
        for key in ('phi_r', 'phi_wms', 'phi_si', 'causal_density')}
    np.savez_compressed(args.output_dir / 'information_priors.npz', **priors, **native_arrays,
                        config=json.dumps(result['prior_data_audit'], sort_keys=True))
    np.savez_compressed(args.output_dir / 'evaluation_arrays.npz',
        **{'prediction_' + k: v for k, v in predictions.items()},
        **{'bootstrap_nrmse_' + k: v for k, v in aggregate.items()},
        **{'bootstrap_lead_nrmse_' + k: v for k, v in lead_bootstrap.items()},
        bootstrap_month_weights=counts, test_target=test_target, target_scale=scale,
        test_issue_dates=split.issue_dates[split.test].astype(str))
    figure = args.output_dir / 'information_prior_comparison.png'
    plot_results(result, predictions, test_target, scale, figure)
    plot_runtime(timing, args.output_dir / 'information_runtime.png')
    result['figure'] = str(figure)
    result['elapsed_seconds'] = time.perf_counter() - started
    (args.output_dir / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f'Completed {args.output_dir} in {result["elapsed_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--timing-repetitions', type=int, default=3)
    parser.add_argument('--skip-inference-probe', action='store_true')
    parser.add_argument('--inference-probe-samples', type=int, default=128)
    run(parser.parse_args())
