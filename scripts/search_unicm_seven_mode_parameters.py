#!/usr/bin/env python3
"""Validation-only search on seven outputs using the original information priors.

Screen, refine, then evaluate selected configurations. Test observations are not
read by the search objectives. The manuscript figure/report are never modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from scripts.run_unicm_target_xi_shapley_prior import DEFAULT_INPUT, DEFAULT_CALIBRATION
from scripts.run_unicm_synergy_guided_forecast import chronological_split, history_summaries, target_scaling, ridge_univariate
from scripts.run_unicm_synergy_regularized_calibration import prepare_designs, feature_penalty, predict_generalized_ridge
from scripts.run_unicm_information_prior_comparison import nonnegative, sha256

OUT=ROOT/'results/unicm_seven_mode_tuning'
LOG=ROOT/'docs/log/unicm_seven_mode_tuning'
TARGETS=('ENSO','NPMM','SPMM','IOB','IOD','SIOD','TNA')
METHODS=('xi_shapley','ei_shapley','shap','surd','phi_r','phi_wms','phi_si','causal_density')
PRIORS=ROOT/'results/unicm_observational_prior_comparison_fit253_surd_allorders'
SHAP=ROOT/'results/unicm_frozen_shap_prior_fit253_pilot'
ALPHAS=np.asarray([.1,1,10,30,100,300,1000,3000,10000,30000,100000,300000,1000000.])
GAMMAS=(.125,.25,.5,1.,2.,3.,4.,6.,8.)
FLOORS=(.005,.05,.5,5.)


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def load_search_data():
    with np.load(DEFAULT_INPUT/'model_inputs.npz') as d:
        history,target=d['history'].astype(float),d['targets'].astype(float)
        names=d['mode_names'].astype(str).tolist()
        preprocessing=json.loads(str(d['metadata']))
    ix=np.asarray([names.index(n) for n in TARGETS])
    with np.load(DEFAULT_INPUT/'modeformer_predictions.npz') as d:
        base,dates=d['ensemble_prediction'].astype(float),d['target_dates'].astype(str)
        if names!=d['mode_names'].astype(str).tolist(): raise ValueError('Mode order mismatch.')
    old=json.loads((DEFAULT_CALIBRATION/'summary.json').read_text())
    split=chronological_split(dates,**old['chronological_split'])
    if [len(split.fit),len(split.validation),len(split.test)]!=[253,36,96]: raise ValueError('Split changed.')
    if preprocessing['normalization_fit_period']!='1980-01/2003-12': raise ValueError('Normalization changed.')
    target=target[:,ix]
    _,scale=target_scaling(target,split.fit)
    add,_=history_summaries(history)
    designs=prepare_designs(base,target,add,split)
    ordered=[designs[t,l] for t in range(7) for l in range(24)]
    audit={}
    with np.load(PRIORS/'information_priors.npz') as d:
        priors={k:nonnegative(d[k][ix],k,audit) for k in METHODS if k!='shap'}
    with np.load(SHAP/'shap_priors.npz') as d: priors['shap']=d['shap'][ix].copy()
    prior_metadata=json.loads((PRIORS/'summary.json').read_text())
    shap_metadata=json.loads((SHAP/'summary.json').read_text())
    if prior_metadata['mode_names']!=names or shap_metadata['mode_names']!=names:
        raise ValueError('Prior source/target order mismatch.')
    if prior_metadata['prior_data_audit']['input_sha256']!=sha256(DEFAULT_INPUT/'model_inputs.npz'):
        raise ValueError('Prior input file differs from calibration inputs.')
    if prior_metadata['prior_data_audit']['fit_indices']!=split.fit.tolist():
        raise ValueError('Prior fit samples differ from calibration fit samples.')
    if not shap_metadata['identity']['fit_only_prior']:
        raise ValueError('SHAP prior is not fit-only.')
    if shap_metadata['identity']['input_sha256']!=sha256(DEFAULT_INPUT/'model_inputs.npz'):
        raise ValueError('SHAP input file differs from calibration inputs.')
    if any(p.shape!=(7,24,11) or not np.isfinite(p).all() or np.any(p<0) for p in priors.values()):
        raise ValueError('Invalid prior shape or values.')
    arrays=dict(gram=np.stack([d.gram for d in ordered]),rhs=np.stack([d.rhs for d in ordered]),
        x_validation=np.stack([d.x_validation for d in ordered],axis=1),
        x_test=np.stack([d.x_test for d in ordered],axis=1),
        y_mean=np.asarray([d.y_mean for d in ordered]),
        validation_target=target[split.validation].reshape(36,168),
        scale=np.broadcast_to(scale,(1,7,24)).reshape(168))
    anchor=np.zeros((168,44))
    for t,source in enumerate(ix):
        for lead in range(24):
            sd=float(base[split.fit,source,lead].std())
            anchor[t*24+lead,source]=1. if sd<1e-8 else sd
    arrays['frozen_anchor_coefficients']=anchor
    identity=dict(mode_names=names,scored_mode_names=list(TARGETS),chronological_split=old['chronological_split'],
        samples=dict(fit=253,validation=36,test=96),syn_tolerance_bit=1e-8,prior_nonnegative_audit=audit,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in
            (DEFAULT_INPUT/'model_inputs.npz',DEFAULT_INPUT/'modeformer_predictions.npz',
             PRIORS/'information_priors.npz',SHAP/'shap_priors.npz')},
        manuscript=dict(parent='P6UJCVG8',main_attachment='DXGC7JEA',supplement='MWIWKSVG',
            sections='S12.1.2 pp21-22; S10.3; original scalar-target game retained',
            checked='2026-10-23',version='no explicit manuscript version/date; current attachments rechecked'))
    return arrays,priors,identity,designs,(base[:,ix],target,split,scale)


def penalties(prior,gamma,floor,forecast_ratio=1.):
    raw=np.asarray(prior).reshape(-1,11)
    mean=raw.mean(axis=1,keepdims=True)
    epsilon=np.maximum(mean*floor,1e-12)  # declared numerical floor in prior-to-penalty map
    relative=(raw+epsilon)/(mean+epsilon)
    penalty=np.tile(relative**(-gamma),(1,4))
    penalty/=penalty.mean(axis=1,keepdims=True)
    # Optional block ratio changes relative shrinkage of frozen forecasts and
    # historical summaries. It does not alter the information prior itself.
    penalty[:,:11]*=forecast_ratio
    return penalty/penalty.mean(axis=1,keepdims=True)


def solve_alpha_batch(arrays,penalty,alphas):
    g=arrays['gram'];r=arrays['rhs']
    matrix=np.broadcast_to(g,(len(alphas),*g.shape)).copy()
    diagonal=np.arange(g.shape[-1])
    matrix[:,:,diagonal,diagonal]+=np.asarray(alphas)[:,None,None]*penalty[None]
    coefficient=np.linalg.solve(matrix,np.broadcast_to(r,(len(alphas),*r.shape))[...,None])[...,0]
    prediction=np.einsum('ncf,acf->anc',arrays['x_validation'],coefficient)+arrays['y_mean']
    cells=np.sqrt(np.mean(((prediction-arrays['validation_target'][None])/arrays['scale'])**2,axis=1))
    if not np.isfinite(cells).all(): raise ValueError('Nonfinite validation score.')
    return cells.reshape(len(alphas),7,24),coefficient


def solve_configurations(arrays,prior,configurations):
    """Score parameter tuples, optionally adding a frozen-predictor ridge centre."""
    configurations=np.asarray(configurations,float)
    if configurations.ndim!=3 or configurations.shape[1]!=7 or configurations.shape[2] not in (4,5):
        raise ValueError('Expected [configuration, seven targets, four or five parameters].')
    scores=[]
    mean=prior.mean(axis=2,keepdims=True)
    for begin in range(0,len(configurations),16):
        c=configurations[begin:begin+16]
        epsilon=np.maximum(mean[None]*c[:,:,2,None,None],1e-12)
        relative=(prior[None]+epsilon)/(mean[None]+epsilon)
        penalty=np.tile(relative**(-c[:,:,1,None,None]),(1,1,1,4))
        penalty[:,:,:,:11]*=c[:,:,3,None,None]
        penalty/=penalty.mean(axis=3,keepdims=True)
        g=np.broadcast_to(arrays['gram'],(len(c),168,44,44)).copy()
        diagonal=np.arange(44)
        g[:,:,diagonal,diagonal]+=(c[:,:,0,None,None]*penalty).reshape(len(c),168,44)
        rhs=np.broadcast_to(arrays['rhs'],(len(c),168,44)).copy()
        if c.shape[2]==5:
            ridge_diagonal=(c[:,:,0,None,None]*penalty).reshape(len(c),168,44)
            center=(c[:,:,4,None,None]*arrays['frozen_anchor_coefficients'].reshape(1,7,24,44)).reshape(len(c),168,44)
            rhs+=ridge_diagonal*center
        coefficient=np.linalg.solve(g,rhs[...,None])[...,0]
        prediction=np.einsum('ncf,acf->anc',arrays['x_validation'],coefficient)+arrays['y_mean']
        cells=np.sqrt(np.mean(((prediction-arrays['validation_target'][None])/arrays['scale'])**2,axis=1))
        if not np.isfinite(cells).all(): raise ValueError('Nonfinite refinement validation score.')
        scores.append(cells.reshape(len(c),7,24))
    return np.concatenate(scores)


def encode(config):
    return np.asarray([config[k] for k in ('alpha','gamma','floor','forecast_ratio')],float)


def decode(config):
    return dict(zip(('alpha','gamma','floor','forecast_ratio','anchor_strength'),np.asarray(config,float).tolist()))


def refine():
    """Same block search and Sobol refinements for every information prior."""
    from scipy.stats import qmc
    started=time.perf_counter()
    arrays,priors,identity,_,_=load_search_data()
    screen=json.loads((OUT/'screen_summary.json').read_text())
    if identity!=screen['identity']: raise ValueError('Screen source identity mismatch.')
    ratios=(.01,.03,.1,.3,1.,3.,10.,30.,100.)
    alphas=np.logspace(-1,8,19)
    # Bounds and refinement radii fixed before looking at any test score.
    bounds=np.log10(np.asarray([[.001,.0625,.0001,.001],[1e9,16.,50.,1000.]]))
    radii=np.asarray([.8,.25,.7,.7])
    sobol=qmc.Sobol(d=4,scramble=True,seed=20261023).random_base2(8)*2-1
    global_sobol=qmc.Sobol(d=4,scramble=True,seed=20261024).random_base2(8)*2-1
    records={};saved={};plans={}
    for key in (*METHODS,'uniform'):
        begin=time.perf_counter()
        prior=np.ones((7,24,11)) if key=='uniform' else priors[key]
        old=screen['methods'][key]
        families={}
        for family in ('standard','block'):
            target_center=np.asarray([encode(p) for p in old['target_parameters']])
            global_center=encode(old['global_parameters'])
            candidates=[]
            # Explore alpha/ratio while retaining screened target-specific vs
            # global gamma/floor. All methods receive the identical layout.
            for center in (target_center,np.broadcast_to(global_center,(7,4))):
                for ratio in (ratios if family=='block' else (1.,)):
                    for alpha in alphas:
                        candidate=center.copy()
                        candidate[:,0]=alpha;candidate[:,3]=ratio
                        candidates.append(candidate)
            candidates.extend([target_center,np.broadcast_to(global_center,(7,4)).copy()])
            candidates=np.asarray(candidates)
            cells=solve_configurations(arrays,prior,candidates)
            all_candidates=[candidates];all_cells=[cells]
            for iteration in range(2):
                by_target=cells.mean(axis=2).argmin(axis=0)
                target_center=candidates[by_target,np.arange(7)].copy()
                # To keep true global configs global, choose among candidates
                # where every target has the same four parameters.
                global_mask=np.max(np.abs(candidates-candidates[:,:1]),axis=(1,2))<1e-12
                global_candidates=candidates[global_mask]
                global_cells=cells[global_mask]
                idx=int(global_cells.mean(axis=(1,2)).argmin())
                global_center=global_candidates[idx,0]
                factor=.5**iteration
                target_plan=10**np.clip(np.log10(target_center)[None]+sobol[:,None,:]*radii*factor,
                                       bounds[0],bounds[1])
                global_plan=10**np.clip(np.log10(global_center)[None]+global_sobol*radii*factor,
                                       bounds[0],bounds[1])
                global_plan=np.broadcast_to(global_plan[:,None],(len(global_plan),7,4)).copy()
                new=np.concatenate([target_plan,global_plan])
                if family=='standard': new[:,:,3]=1.
                if key=='uniform': new[:,:,1]=1.;new[:,:,2]=.05
                scored=solve_configurations(arrays,prior,new)
                all_candidates.append(new);all_cells.append(scored)
                candidates=np.concatenate([candidates,new]);cells=np.concatenate([cells,scored])
            by_target=cells.mean(axis=2).argmin(axis=0)
            target_config=candidates[by_target,np.arange(7)]
            target_cells=cells[by_target,np.arange(7)]
            same=np.max(np.abs(candidates-candidates[:,:1]),axis=(1,2))<1e-12
            global_index=np.flatnonzero(same)[cells[same].mean(axis=(1,2)).argmin()]
            families[family]=dict(global_parameters=decode(candidates[global_index,0]),
                global_validation_nrmse=float(cells[global_index].mean()),
                target_parameters=[decode(c) for c in target_config],
                target_validation_nrmse=target_cells.mean(axis=1).tolist(),
                targetwise_validation_nrmse=float(target_cells.mean()),
                evaluations=len(candidates))
            saved[key+'_'+family]=cells
            plans[key+'_'+family]=candidates
            print(f"{key}/{family}: global={families[family]['global_validation_nrmse']:.6f}, targetwise={target_cells.mean():.6f}; {len(candidates)}",flush=True)
        records[key]=families
        with (LOG/'run_history.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage='refine',method=key,results=families,elapsed_seconds=time.perf_counter()-begin))+'\n')
    write_json(OUT/'refine_summary.json',dict(status='refinement_complete_no_test_evaluation',identity=identity,
        methods=records,local_bounds=(10**bounds).tolist(),refinement_radii_log10=radii.tolist(),
        sobol_seed=20261023,global_sobol_seed=20261024,refinement_rounds=2,
        family_selection='block targetwise is primary; standard targetwise/global and block global are diagnostics',
        elapsed_seconds=time.perf_counter()-started))
    # Retain validation surfaces so no expensive search has to be repeated.
    np.savez_compressed(OUT/'refine_validation_scores.npz',**saved)
    np.savez_compressed(OUT/'refine_configurations.npz',**plans)


def anchor_search():
    """Further validation-driven search of the ridge centre; same capacity/budget."""
    from scipy.stats import qmc
    start=time.perf_counter()
    arrays,priors,identity,_,_=load_search_data()
    previous=json.loads((OUT/'refine_summary.json').read_text())
    if previous['identity']!=identity: raise ValueError('Source identity changed before anchor search.')
    rng=qmc.Sobol(5,scramble=True,seed=20261025).random_base2(8)*2-1
    bounds=np.log10(np.asarray([[.001,.0625,.0001,.001],[1e9,16.,50.,1000.]]))
    radii=np.asarray([.6,.2,.5,.5])
    records={}
    for key in (*METHODS,'uniform'):
        prior=np.ones((7,24,11)) if key=='uniform' else priors[key]
        old=previous['methods'][key]['block']
        target=np.asarray([encode(p) for p in old['target_parameters']])
        global_=encode(old['global_parameters'])
        candidates=[]
        for center in (target,np.broadcast_to(global_,(7,4))):
            for strength in (0.,.25,.5,.75,1.):
                candidates.append(np.column_stack([center,np.full(7,strength)]))
        candidates=np.asarray(candidates)
        cells=solve_configurations(arrays,prior,candidates)
        for iteration in range(2):
            ix=cells.mean(axis=2).argmin(axis=0)
            target=candidates[ix,np.arange(7)]
            mask=np.max(np.abs(candidates-candidates[:,:1]),axis=(1,2))<1e-12
            global_=candidates[np.flatnonzero(mask)[cells[mask].mean(axis=(1,2)).argmin()],0]
            centers=np.stack([target,np.broadcast_to(global_,(7,5))])
            plans=[]
            for center in centers:
                plan=np.empty((len(rng),7,5))
                plan[:,:,:4]=10**np.clip(np.log10(center[:,:4])[None]+rng[:,None,:4]*radii*(.5**iteration),bounds[0],bounds[1])
                plan[:,:,4]=np.clip(center[:,4][None]+rng[:,4,None]*.5*(.5**iteration),0,1)
                if key=='uniform': plan[:,:,1]=1.;plan[:,:,2]=.05
                plans.append(plan)
            new=np.concatenate(plans)
            cells=np.concatenate([cells,solve_configurations(arrays,prior,new)])
            candidates=np.concatenate([candidates,new])
        ix=cells.mean(axis=2).argmin(axis=0)
        target=candidates[ix,np.arange(7)]
        mask=np.max(np.abs(candidates-candidates[:,:1]),axis=(1,2))<1e-12
        gi=np.flatnonzero(mask)[cells[mask].mean(axis=(1,2)).argmin()]
        records[key]=dict(global_parameters=decode(candidates[gi,0]),
            global_validation_nrmse=float(cells[gi].mean()),
            target_parameters=[decode(c) for c in target],
            target_validation_nrmse=cells[ix,np.arange(7)].mean(axis=1).tolist(),
            targetwise_validation_nrmse=float(cells[ix,np.arange(7)].mean()),evaluations=len(candidates))
        np.savez_compressed(OUT/('anchor_validation_'+key+'.npz'),configurations=candidates,cell_scores=cells)
        with (LOG/'run_history.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage='anchor',method=key,results=records[key]))+'\n')
        print(f"{key}/anchor: global={records[key]['global_validation_nrmse']:.6f}, targetwise={records[key]['targetwise_validation_nrmse']:.6f}; {len(candidates)}",flush=True)
    write_json(OUT/'anchor_summary.json',dict(status='anchor_complete_no_test_evaluation',identity=identity,
        methods=records,anchor_definition='ridge centre = kappa * own frozen forecast slope; target mean intercept remains fit-only',
        anchor_strength_bounds=[0,1],sobol_seed=20261025,evaluations_per_method=1034,
        primary_family='targetwise anchored block; global and unanchored families are diagnostics',
        elapsed_seconds=time.perf_counter()-start))


def fit_selected(arrays,prior,parameters):
    """Fit frozen validation-selected parameters and emit validation/test outputs."""
    c=np.asarray([[p[k] for k in ('alpha','gamma','floor','forecast_ratio')]+[p.get('anchor_strength',0.)]
                  for p in parameters])
    mean=prior.mean(axis=2,keepdims=True)
    epsilon=np.maximum(mean*c[:,2,None,None],1e-12)
    relative=(prior+epsilon)/(mean+epsilon)
    penalty=np.tile(relative**(-c[:,1,None,None]),(1,1,4))
    penalty[:,:,:11]*=c[:,3,None,None]
    penalty/=penalty.mean(axis=2,keepdims=True)
    diagonal=(c[:,0,None,None]*penalty).reshape(168,44)
    matrix=arrays['gram'].copy()
    f=np.arange(44);matrix[:,f,f]+=diagonal
    center=(c[:,4,None,None]*arrays['frozen_anchor_coefficients'].reshape(7,24,44)).reshape(168,44)
    coefficient=np.linalg.solve(matrix,(arrays['rhs']+diagonal*center)[...,None])[...,0]
    return tuple((np.einsum('ncf,cf->nc',arrays['x_'+part],coefficient)+arrays['y_mean']).reshape(-1,7,24)
                 for part in ('validation','test'))


def freeze():
    """Persist every selected configuration before any new test scoring."""
    arrays,priors,identity,_,_=load_search_data()
    ref=json.loads((OUT/'refine_summary.json').read_text())
    anchor=json.loads((OUT/'anchor_summary.json').read_text())
    if ref['identity']!=identity or anchor['identity']!=identity: raise ValueError('Search source changed.')
    families={key:{} for key in ('standard_global','standard_targetwise','block_global','block_targetwise',
                                 'anchor_global','anchor_targetwise')}
    for method in (*METHODS,'uniform'):
        for variant in ('standard','block','anchor'):
            item=anchor['methods'][method] if variant=='anchor' else ref['methods'][method][variant]
            families[variant+'_global'][method]=dict(parameters=[item['global_parameters']]*7,
                                                     validation_nrmse=item['global_validation_nrmse'])
            families[variant+'_targetwise'][method]=dict(parameters=item['target_parameters'],
                validation_nrmse=item['targetwise_validation_nrmse'])
    # Univariate has a single coefficient per output/lead; search its entire
    # relevant regularization range directly, without pointless gamma/floor axes.
    _,_,_,_,(base,target,split,scale)=load_search_data()
    alphas=np.logspace(-4,8,49)
    cells=[]
    for alpha in alphas:
        val,_=ridge_univariate(base,target,split,float(alpha))
        cells.append(np.sqrt(np.mean(((val-target[split.validation])/scale)**2,axis=0)))
    cells=np.asarray(cells);gi=int(cells.mean(axis=(1,2)).argmin());ti=cells.mean(axis=2).argmin(axis=0)
    univariate=dict(global_parameters=dict(alpha=float(alphas[gi])),
        target_parameters=[dict(alpha=float(alphas[i])) for i in ti],
        global_validation_nrmse=float(cells[gi].mean()),targetwise_validation_nrmse=float(cells[ti,np.arange(7)].mean()),
        evaluations=len(alphas))
    for name in families:
        global_=name.endswith('_global')
        families[name]['univariate']=dict(parameters=([univariate['global_parameters']]*7 if global_ else univariate['target_parameters']),
            validation_nrmse=univariate['global_validation_nrmse' if global_ else 'targetwise_validation_nrmse'])
    output=dict(status='configurations_frozen_before_test_scoring',identity=identity,
        primary_family='anchor_targetwise',secondary_families=list(families),families=families,
        choice_basis='validation only; anchored block was added based on validation findings before test evaluation',
        test_used_to_choose_family=False,univariate_search=univariate,
        uncertainty_plan=dict(primary_block_months=12,sensitivity_block_months=24,replicates=4000,
            comparisons=list(METHODS[1:])+['uniform','univariate'],
            simultaneous_intervals='Bonferroni percentile intervals across nine Xi-versus-comparator contrasts'))
    write_json(OUT/'frozen_configurations.json',output)
    print('Frozen primary anchor_targetwise; six declared families; no test scoring.',flush=True)


def evaluate():
    from scripts.run_unicm_information_prior_comparison import paired_bootstrap
    from scripts.run_unicm_synergy_guided_forecast import mean_cell_nrmse,mean_cell_acc,cell_rmse
    arrays,priors,identity,_,(base,target,split,scale)=load_search_data()
    frozen_path=OUT/'frozen_configurations.json'
    frozen=json.loads(frozen_path.read_text())
    if identity!=frozen['identity']: raise ValueError('Source changed after configuration freeze.')
    frozen_hash=sha256(frozen_path)
    old=ROOT/'results/unicm_forecast_scoring_seven_modes'
    with np.load(old/'evaluation_arrays.npz') as cache:
        original={k.removeprefix('prediction_'):cache[k].copy() for k in cache.files
                  if k.startswith('prediction_') and k!='prediction_historical_uniform'}
        np.testing.assert_array_equal(target[split.test],cache['test_target'])
        np.testing.assert_array_equal(scale,cache['target_scale'])
    predictions={};families={};boot_saved={}
    test_target=target[split.test]
    for family,selected in frozen['families'].items():
        family_predictions={'frozen':original['frozen']}
        for method,item in selected.items():
            if method=='univariate':
                pred=np.empty((len(split.test),7,24));val=np.empty((len(split.validation),7,24))
                for t,p in enumerate(item['parameters']):
                    v,q=ridge_univariate(base,target,split,p['alpha'])
                    val[:,t]=v[:,t];pred[:,t]=q[:,t]
            else:
                prior=np.ones((7,24,11)) if method=='uniform' else priors[method]
                val,pred=fit_selected(arrays,prior,item['parameters'])
            validation=mean_cell_nrmse(val,target[split.validation],scale)
            if abs(validation-item['validation_nrmse'])>1e-10:
                raise ValueError(f'{family}/{method} validation fit mismatch {validation} vs {item["validation_nrmse"]}.')
            family_predictions[method]=pred
        # Include the original fixed Xi prediction to measure improvement over
        # the user's current 10.65% result under identical months and scaling.
        family_predictions['original_xi']=original['xi_shapley']
        aggregate,lead_draws,weights=paired_bootstrap(family_predictions,test_target,scale,4000,12,20261001)
        metrics={}
        scores={k:mean_cell_nrmse(p,test_target,scale) for k,p in family_predictions.items()}
        for method,pred in family_predictions.items():
            cells=cell_rmse(pred,test_target,scale)
            entry=dict(test_nrmse=scores[method],test_acc=mean_cell_acc(pred,test_target),
                reduction_vs_frozen_percent=100*(1-scores[method]/scores['frozen']),
                reduction_vs_frozen_ci95_percent=np.percentile(100*(1-aggregate[method]/aggregate['frozen']),[2.5,97.5]).tolist(),
                target_nrmse=cells.mean(axis=1).tolist(),lead_nrmse=cells.mean(axis=0).tolist(),
                validation_nrmse=selected.get(method,{}).get('validation_nrmse'),
                parameters=selected.get(method,{}).get('parameters'))
            for baseline in ('frozen','uniform','univariate','ei_shapley','shap','surd','original_xi'):
                delta=aggregate[baseline]-aggregate[method]
                entry['gain_over_'+baseline]=dict(point=scores[baseline]-scores[method],
                    ci95=np.percentile(delta,[2.5,97.5]).tolist())
            metrics[method]=entry
        families[family]=dict(methods=metrics)
        predictions.update({family+'__'+k:v for k,v in family_predictions.items()})
        if family==frozen['primary_family']:
            comparisons=[k for k in METHODS if k!='xi_shapley']+['uniform','univariate']
            simultaneous={}
            for key in comparisons:
                delta=aggregate[key]-aggregate['xi_shapley']
                tail=100*.05/(2*len(comparisons))
                simultaneous[key]=dict(point=scores[key]-scores['xi_shapley'],
                    pointwise_ci95=np.percentile(delta,[2.5,97.5]).tolist(),
                    bonferroni_family_ci95=np.percentile(delta,[tail,100-tail]).tolist())
            families[family]['xi_simultaneous_comparisons']=simultaneous
            alt,_,_=paired_bootstrap(family_predictions,test_target,scale,4000,24,20261001)
            families[family]['block24_sensitivity']={key:dict(point=scores[key]-scores['xi_shapley'],
                ci95=np.percentile(alt[key]-alt['xi_shapley'],[2.5,97.5]).tolist())
                for key in comparisons+['frozen','original_xi']}
            boot_saved={**{'block12_'+k:v for k,v in aggregate.items()},**{'block24_'+k:v for k,v in alt.items()}}
            np.testing.assert_allclose(weights.sum(axis=1),1.)
        print(f"{family}: Xi={scores['xi_shapley']:.6f} ({metrics['xi_shapley']['reduction_vs_frozen_percent']:.2f}%), EI={scores['ei_shapley']:.6f}, SHAP={scores['shap']:.6f}, Uniform={scores['uniform']:.6f}",flush=True)
    result=dict(status='completed_search_and_evaluation',identity=identity,primary_family=frozen['primary_family'],
        configuration_freeze_sha256=frozen_hash,families=families,
        fixed_original_methods=json.loads((old/'summary.json').read_text())['methods'],
        limitations=['Exploratory: this historical test period has been viewed in previous research.',
            'Hyperparameters and model family were selected using the same 36 overlapping validation issue months.',
            'Bootstrap conditions on selected parameters and fixed information priors; it does not cover selection uncertainty.',
            'All competitors receive their own validation tuning and the same refinement rules; no test-based model selection.',
            'Anchor penalty extends the original zero-centred calibration method; coefficient capacity remains 44 per target/lead.'],
        bootstrap=dict(replicates=4000,primary_block_months=12,sensitivity_block_months=24,seed=20261001),
        official_figure_and_report_modified=False)
    if frozen_hash!=sha256(frozen_path): raise ValueError('Configurations changed during test evaluation.')
    write_json(OUT/'evaluation_summary.json',result)
    np.savez_compressed(OUT/'selected_predictions.npz',**predictions,test_target=test_target,target_scale=scale,
                        target_mode_names=TARGETS,test_issue_dates=split.issue_dates[split.test].astype(str))
    np.savez_compressed(OUT/'bootstrap_primary.npz',**boot_saved)


def screen():
    started=time.perf_counter()
    arrays,priors,identity,designs,univariate=load_search_data()
    # Numerical equivalence smoke test to the original implementation, before any search.
    check=penalties(priors['xi_shapley'],3.,.05)
    for t,l in ((0,0),(3,7),(6,23)):
        np.testing.assert_allclose(check[t*24+l],feature_penalty(priors['xi_shapley'][t,l],3.,floor_fraction=.05),atol=1e-12,rtol=1e-12)
    val,_=predict_generalized_ridge(designs,priors['xi_shapley'],alpha=30000,gamma=3,floor_fraction=.05)
    cell,_=solve_alpha_batch(arrays,check,np.asarray([30000.]))
    np.testing.assert_allclose(cell[0],np.sqrt(np.mean(((val-univariate[1][univariate[2].validation])/univariate[3])**2,axis=0)),atol=1e-10,rtol=0)
    records={};save={};configs=[]
    for floor in FLOORS:
        for gamma in GAMMAS:
            configs.extend(dict(alpha=float(a),gamma=gamma,floor=floor,forecast_ratio=1.) for a in ALPHAS)
    for key in (*METHODS,'uniform'):
        begin=time.perf_counter()
        if key=='uniform':
            cells,_=solve_alpha_batch(arrays,np.ones((168,44)),ALPHAS)
            parameters=[dict(alpha=float(a),gamma=1.,floor=.05,forecast_ratio=1.) for a in ALPHAS]
        else:
            blocks=[]
            for floor in FLOORS:
                for gamma in GAMMAS:
                    cells,_=solve_alpha_batch(arrays,penalties(priors[key],gamma,floor),ALPHAS)
                    blocks.append(cells)
            cells=np.concatenate(blocks);parameters=configs
        global_scores=cells.mean(axis=(1,2));target_scores=cells.mean(axis=2)
        best=int(global_scores.argmin());by_target=target_scores.argmin(axis=0)
        records[key]=dict(global_parameters=parameters[best],global_validation_nrmse=float(global_scores[best]),
            target_parameters=[parameters[i] for i in by_target],
            target_validation_nrmse=[float(target_scores[i,t]) for t,i in enumerate(by_target)],
            targetwise_validation_nrmse=float(np.mean(target_scores[by_target,np.arange(7)])),
            evaluations=len(parameters),elapsed_seconds=time.perf_counter()-begin)
        save[key]=cells
        print(f"{key}: val global={global_scores[best]:.6f}, targetwise={records[key]['targetwise_validation_nrmse']:.6f}; {len(parameters)} configs; {time.perf_counter()-begin:.1f}s",flush=True)
        with (LOG/'run_history.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage='screen',method=key,**records[key]))+'\n')
    base,target,split,scale=univariate
    uni_cells=[]
    for alpha in ALPHAS:
        val,_=ridge_univariate(base,target,split,float(alpha))
        uni_cells.append(np.sqrt(np.mean(((val-target[split.validation])/scale)**2,axis=0)))
    uni_cells=np.asarray(uni_cells)
    scores=uni_cells.mean(axis=(1,2));by_target=uni_cells.mean(axis=2).argmin(axis=0)
    records['univariate']=dict(global_parameters=dict(alpha=float(ALPHAS[scores.argmin()])),
        global_validation_nrmse=float(scores.min()),
        target_parameters=[dict(alpha=float(ALPHAS[i])) for i in by_target],
        target_validation_nrmse=[float(uni_cells[i,t].mean()) for t,i in enumerate(by_target)],
        targetwise_validation_nrmse=float(uni_cells[by_target,np.arange(7)].mean()),evaluations=len(ALPHAS))
    save['univariate']=uni_cells
    write_json(OUT/'screen_summary.json',dict(status='screen_complete_no_test_evaluation',identity=identity,
        selection='validation-only mean over seven outputs and 24 leads',
        primary_family='one parameter triple per target, fixed across all 24 leads',
        configs=configs,alpha_grid=ALPHAS.tolist(),gamma_grid=GAMMAS,floor_grid=FLOORS,
        methods=records,elapsed_seconds=time.perf_counter()-started))
    np.savez_compressed(OUT/'screen_cell_scores.npz',**save)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('screen','refine','anchor','freeze','evaluate'),default='screen')
    args=parser.parse_args()
    {'screen':screen,'refine':refine,'anchor':anchor_search,'freeze':freeze,'evaluate':evaluate}[args.stage]()
