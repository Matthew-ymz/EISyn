#!/usr/bin/env python3
"""Rescore original predictions on seven modes; do not change priors or tuning."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_unicm_information_prior_comparison import sha256, paired_bootstrap, plot_results
from scripts.run_unicm_observational_prior_comparison import metric_entry
from scripts.run_unicm_frozen_shap_prior import plot_comparison
from scripts.run_unicm_target_xi_shapley_prior import DEFAULT_INPUT, DEFAULT_CALIBRATION, DEFAULT_OUTPUT as XI
from scripts.run_unicm_synergy_guided_forecast import (
    chronological_split, history_summaries, target_scaling, mean_cell_nrmse, mean_cell_acc, cell_rmse,
)
from scripts.run_unicm_synergy_regularized_calibration import (
    prepare_designs, predict_generalized_ridge, shuffled_centrality,
)

OUTPUT = ROOT/'results/unicm_forecast_scoring_seven_modes'
OBSERVED = ROOT/'results/unicm_observational_prior_comparison_fit253_surd_allorders'
SHAP = ROOT/'results/unicm_frozen_shap_prior_fit253_pilot'
SCORED_MODES = ('ENSO','NPMM','SPMM','IOB','IOD','SIOD','TNA')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    reference=json.loads((OBSERVED/'summary.json').read_text())
    shap_reference=json.loads((SHAP/'summary.json').read_text())
    xi_reference=json.loads((XI/'summary.json').read_text())
    names=reference['mode_names']
    indices=np.asarray([names.index(n) for n in SCORED_MODES])
    if reference['shared_hyperparameters'] != {'alpha':30000.,'gamma':3.}:
        raise ValueError('Original calibration parameters changed.')
    with np.load(SHAP/'evaluation_arrays.npz') as cache:
        full_target=cache['test_target'].copy()
        full_scale=cache['target_scale'].copy()
        full_predictions={k.removeprefix('prediction_'):cache[k].copy() for k in cache.files
                          if k.startswith('prediction_') and k!='prediction_shap_half_permutations'}
        half=cache['prediction_shap_half_permutations'][:,indices].copy()
        weights=cache['bootstrap_month_weights'].copy()
    # Fail if an earlier stage has changed instead of silently accepting a new fit.
    with np.load(OBSERVED/'evaluation_arrays.npz') as cache:
        np.testing.assert_array_equal(full_target,cache['test_target'])
        np.testing.assert_array_equal(full_scale,cache['target_scale'])
        for key in reference['methods']:
            np.testing.assert_array_equal(full_predictions[key],cache['prediction_'+key])
    target,scale=full_target[:,indices],full_scale[:,indices]
    predictions={k:v[:,indices] for k,v in full_predictions.items()}
    draws,lead_draws,reconstructed_weights=paired_bootstrap(predictions,target,scale,4000,12,20261001)
    np.testing.assert_array_equal(reconstructed_weights,weights)
    result={**reference,'status':'completed','scoring_revision':'evaluation_only_seven_modes',
        'scored_mode_names':list(SCORED_MODES),'scored_mode_indices':indices.tolist(),
        'scored_cell_count':168,'scored_prediction_observation_pairs':16128,
        'upstream_mode_names':names,'mode_names':list(SCORED_MODES),
        'upstream_parameter_selection':'original eleven-mode validation selection; not repeated',
        'fit_or_tuning_changed':False,'priors_changed':False,'new_syn_estimation':False,
        'methods':{},'bootstrap':{**reference['bootstrap'],'original_month_weights_reused':True},
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in
            (OBSERVED/'summary.json',OBSERVED/'evaluation_arrays.npz',OBSERVED/'information_priors.npz',
             SHAP/'summary.json',SHAP/'evaluation_arrays.npz',SHAP/'shap_priors.npz',XI/'summary.json')},
        'scope_note':'Only final metrics select seven outputs. All eleven-target predictions, priors, validation choices, and mechanism figures remain unchanged.'}
    scores={key:mean_cell_nrmse(pred,target,scale) for key,pred in predictions.items()}
    for key,pred in predictions.items():
        source_entry=shap_reference['methods'][key]
        entry={**source_entry,**metric_entry(pred,target,scale),
               'raw_rmse':float(np.sqrt(np.mean((pred-target)**2,axis=0)).mean()),
               'rmse_reduction_vs_frozen_percent':100*(1-scores[key]/scores['frozen'])}
        for baseline in predictions:
            entry['gain_over_'+baseline]={
                'point':scores[baseline]-scores[key],
                'ci95':np.percentile(draws[baseline]-draws[key],[2.5,97.5]).tolist(),
                'lead_ci95':np.percentile(lead_draws[baseline]-lead_draws[key],[2.5,97.5],axis=0).tolist()}
        entry['gain_over_xi']=entry['gain_over_xi_shapley']
        entry['rmse_reduction_vs_frozen_ci95_percent']=np.percentile(
            100*(1-draws[key]/draws['frozen']),[2.5,97.5]).tolist()
        result['methods'][key]=entry
    # Null predictions were not cached. Reconstruct the SAME original fits on all
    # eleven outputs, verify their original scores, then score the seven outputs.
    with np.load(DEFAULT_INPUT/'model_inputs.npz') as cache:
        history,all_target=cache['history'].astype(float),cache['targets'].astype(float)
    with np.load(DEFAULT_INPUT/'modeformer_predictions.npz') as cache:
        ensemble,dates=cache['ensemble_prediction'].astype(float),cache['target_dates'].astype(str)
    calibration=json.loads((DEFAULT_CALIBRATION/'summary.json').read_text())
    split=chronological_split(dates,**calibration['chronological_split'])
    np.testing.assert_array_equal(all_target[split.test],full_target)
    _,reconstructed_scale=target_scaling(all_target,split.fit)
    np.testing.assert_array_equal(reconstructed_scale,full_scale)
    additive,_=history_summaries(history)
    designs=prepare_designs(ensemble,all_target,additive,split)
    with np.load(XI/'target_xi_shapley_centrality.npz') as cache:
        centrality=cache['centrality'].copy()
    rng=np.random.default_rng(20260908)
    original_null=xi_reference['shuffled_xi_control']
    null_scores=[]
    maximum_error=0.
    for expected,hp in zip(original_null['scores'],original_null['hyperparameters'],strict=True):
        _,prediction=predict_generalized_ridge(designs,shuffled_centrality(centrality,rng),
            alpha=hp['alpha'],gamma=hp['gamma'],floor_fraction=.05)
        old_score=mean_cell_nrmse(prediction,full_target,full_scale)
        maximum_error=max(maximum_error,abs(old_score-expected))
        if abs(old_score-expected)>1e-10:
            raise ValueError(f'Original shuffled prediction reproduction failed: {old_score} vs {expected}.')
        null_scores.append(mean_cell_nrmse(prediction[:,indices],target,scale))
    null=np.asarray(null_scores)
    result['shuffled_xi_control']={**original_null,'scores':null_scores,
        'mean':float(null.mean()),'std':float(null.std(ddof=1)),
        'fraction_null_at_least_as_good':float((1+np.count_nonzero(null<=scores['xi_shapley']))/(1+len(null))),
        'original_11mode_score_maximum_reproduction_error':maximum_error,
        'reconstruction':'same seed=20260908, full eleven-target designs, original per-null alpha/gamma; no search'}
    with np.load(DEFAULT_CALIBRATION/'evaluation_arrays.npz') as cache:
        historical=cache['prediction_uniform'][:,indices].copy()
    hist_draw,hist_lead,_=paired_bootstrap({'uniform':historical},target,scale,4000,12,20261001)
    history_score=mean_cell_nrmse(historical,target,scale)
    hist_lead_gain=cell_rmse(historical,target,scale).mean(axis=0)-np.asarray(result['methods']['xi_shapley']['lead_nrmse'])
    result['historical_uniform']={
        'alpha':1000.,'test_nrmse':history_score,'test_acc':mean_cell_acc(historical,target),
        'xi_gain':history_score-scores['xi_shapley'],
        'xi_relative_reduction_percent':100*(1-scores['xi_shapley']/history_score),
        'xi_gain_ci95':np.percentile(hist_draw['uniform']-draws['xi_shapley'],[2.5,97.5]).tolist(),
        'lead_7_10_xi_gain':float(hist_lead_gain[6:10].mean()),
        'lead_7_10_xi_gain_ci95':np.percentile((hist_lead['uniform']-lead_draws['xi_shapley'])[:,6:10].mean(axis=1),[2.5,97.5]).tolist()}
    result['shap_half_budget']={'test_nrmse':mean_cell_nrmse(half,target,scale),
        'full_minus_half_nrmse':scores['shap']-mean_cell_nrmse(half,target,scale)}
    np.savez_compressed(OUTPUT/'evaluation_arrays.npz',
        **{'prediction_'+k:v for k,v in predictions.items()},
        **{'bootstrap_nrmse_'+k:v for k,v in draws.items()},
        **{'bootstrap_lead_nrmse_'+k:v for k,v in lead_draws.items()},
        bootstrap_month_weights=weights,test_target=target,target_scale=scale,
        scored_mode_names=SCORED_MODES,prediction_historical_uniform=historical)
    result['figure']=str(OUTPUT/'information_prior_comparison.png')
    (OUTPUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    plot_results(result,predictions,target,scale,OUTPUT/'information_prior_comparison.png')
    plot_comparison(result,predictions,target,scale,OUTPUT/'shap_prior_comparison.png')
    for key,entry in result['methods'].items():
        print(f"{key}: {entry['test_nrmse']:.6f}; reduction {entry['rmse_reduction_vs_frozen_percent']:.2f}%")
    print('Original null reproduction maximum error:',maximum_error)


if __name__=='__main__':
    run()
