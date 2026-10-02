"""Combine synchronized UniCM intervention panels and observational validation."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_earth_system_main_figures import (
    configure_matplotlib,
    plot_unicm_figure,
    plot_unicm_rmse_display_comparison,
    UNICM_INFORMATION_PRIOR_SUMMARY,
    UNICM_MAIN_PRIOR_METHODS,
    UNICM_MAIN_METHOD_KEYS,
    UNICM_SHAP_PRIOR_SUMMARY,
)

OUT=ROOT/'docs/reports/assets/unicm_main_with_hypergraph.png'
FIGURE_OUT=ROOT/'fig/earth_unicm_hierarchy_overview.png'

def main():
    comparison=json.loads(UNICM_INFORMATION_PRIOR_SUMMARY.read_text())
    shap_prior=json.loads(UNICM_SHAP_PRIOR_SUMMARY.read_text())
    tree=json.loads((ROOT/'results/unicm_xi_hierarchy_uniform_n16384/summary.json').read_text())
    shap=json.loads((ROOT/'results/unicm_11mode_shapley_affine_n16384_independent/summary.json').read_text())
    hyper=json.loads((ROOT/'results/unicm_pair_hypergraph_independent/audit.json').read_text())
    with np.load(ROOT/'results/unicm_spt_order_mass/order_mass.npz') as d:
        order=json.loads(str(d['metadata']))
        order_tolerance=float(order['syn_tolerance_bits'])
    keys=['n_samples','sampling_seed','intervention_bound','start_month','estimator']
    assert all(order[k]==hyper[k] for k in keys)
    assert all(tree[k]==hyper[k] for k in keys)
    assert all(shap['method'][k]==hyper[k] for k in keys)
    tolerances = {
        'order_bits': order_tolerance,
        'spt_bits': float(tree['syn_tolerance_bits']),
        'hypergraph_bits': float(hyper['tolerance_bits']),
        'shapley_bits': float(shap['audit']['syn_nonnegative_tolerance_bits']),
    }
    assert len(set(tolerances.values())) == 1
    caches=[]
    for cp in [1,2,3]:
        p=ROOT/f'results/unicm_xi_hierarchy_uniform_n16384/cache/checkpoint{cp}_samples16384_sampling20260901_bound4_fullhist12_start0_cpu.npz'
        with np.load(p) as d:
            metadata=json.loads(str(d['metadata']))
            assert d['all_mode_targets'].shape==(16384,24,11)
            assert all(metadata[k]==hyper[k] for k in ['n_samples','sampling_seed','intervention_bound','start_month'])
        caches.append(dict(checkpoint=cp,path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    report=dict(all_intervention_panels_same_samples_and_parameters=True,
        intervention_panel_scope='a-h; panel j combines intervention Xi, direct grouped SHAP pilot and natural observational comparators; EI-Shapley retained in appendix B.5',
        synchronized_intervention_parameters={k:hyper[k] for k in keys},
        synchronized_nonnegative_tolerance=tolerances,
        shared_prediction_caches=caches,
        sample_identity_evidence='same deterministic sampler, full seed/configuration and exact prediction cache files; cached source arrays are not stored',
        tree=dict(n_samples=tree['n_samples'],sampling_seed=tree['sampling_seed'],checkpoint_displayed=2,algorithm=tree['estimator']),
        shapley=shap['method'],
        target_definitions=dict(hypergraph='one scalar future mode',spt_order_shapley='all 11 future modes jointly'),
        forecast_validation=dict(
            kind='observational held-out prediction evaluation',
            fit_samples=253,
            validation_samples=36,
            test_samples=96,
            data_period='1980-01/2018-12',
            normalization_fit_period='1980-01/2003-12',
            prior='target- and lead-specific exact Xi-Shapley generalized-ridge weights',
            intervention_samples=16384,
            summary='results/unicm_target_xi_shapley_prior_normfit_1980_2003_n16384/summary.json',
            panel_j=dict(summary=str(UNICM_INFORMATION_PRIOR_SUMMARY.relative_to(ROOT)),
                         displayed_metric='100 * (1 - mean_cell_nrmse_method / mean_cell_nrmse_frozen)',
                         metric_unit='percent; higher is better; negative means worse than Frozen',
                         aggregation='same mean over 264 normalized target-lead RMSE cells as source summaries',
                         sha256=hashlib.sha256(UNICM_INFORMATION_PRIOR_SUMMARY.read_bytes()).hexdigest(),
                         parameter_mode=comparison['parameter_mode'],
                         shared_hyperparameters=comparison['shared_hyperparameters'],
                         included_methods=list(UNICM_MAIN_PRIOR_METHODS),
                         displayed_methods=list(UNICM_MAIN_METHOD_KEYS),
                         computed_methods=[*comparison['included_methods'],'shap'],
                         appendix_only_methods=['ei_shapley'],
                         excluded_methods=comparison['excluded_methods'],
                         prior_distributions={**comparison['prior_distributions'],
                                              'shap':'fit_only_natural_history_background_replacement_and_frozen_output'},
                         shap_summary=str(UNICM_SHAP_PRIOR_SUMMARY.relative_to(ROOT)),
                         shap_summary_sha256=hashlib.sha256(UNICM_SHAP_PRIOR_SUMMARY.read_bytes()).hexdigest(),
                         shap_sampling=shap_prior['identity'],
                         prior_data_audit=comparison['prior_data_audit'],
                         references=['frozen','univariate','uniform']),
            panel_g='Tree-wise SPT order mass; original panel g.',
            panel_i='Original panel j: Xi-Shapley shuffled-prior control; percentage reduction versus frozen ensemble.',
            note='EI/Xi use 16,384 independent intervention histories per checkpoint. PhiR, WMS, SI, causal density and all-order Gaussian SURD use only 253 natural fit histories paired with actual future observations; no frozen predictions enter these observational priors. Direct grouped SHAP uses 16 fit foregrounds and 4 fit backgrounds, with two antithetic permutation pairs per frozen checkpoint; its approximation is declared in appendix B.7. Calibration evaluation retains its chronological fit/validation/test split.'),
        action='SPT, SPT-selected order mass, exact Shapley, and pair hypergraphs share the corrected estimator and identical intervention samples.')
    audit=ROOT/'results/unicm_pair_hypergraph_independent/composite_provenance_audit.json'
    audit.write_text(json.dumps(report,indent=2)+'\n')
    configure_matplotlib()
    plot_unicm_figure(OUT.with_suffix(""))
    plot_unicm_rmse_display_comparison(OUT.parent / 'unicm_rmse_display_comparison')
    FIGURE_OUT.parent.mkdir(parents=True,exist_ok=True)
    FIGURE_OUT.write_bytes(OUT.read_bytes())
    print(OUT)

if __name__=='__main__': main()
