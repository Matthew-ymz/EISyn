"""Bounded full-dimensional quadratic TM feasibility probe; not a full Syn estimate.

Probe the last triangular component of IOB | (Y, other ten modes, earlier IOB
history). Its degree-2 design is the largest encountered in the full 11-mode
conditional map. Use the same split and polynomial ridge for degree 1 and 2.
A failed held-out component blocks scaling this density family to full CMI.
"""
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.linalg import cho_factor,cho_solve
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from exp.TM.transport_map_density import _polynomial_design,_polynomial_exponents
from scripts.unicm_peid_syn_analysis import sample_full_history_mode_inputs
OUT=ROOT/'results/unicm_high_order_tm_pilot'


def fit_component(predictors,response,fit,evaluate,degree):
    started=time.monotonic()
    mean=predictors[fit].mean(0); scale=predictors[fit].std(0)
    if (scale<=1e-8).any(): raise ValueError('Degenerate predictor')
    exponents=_polynomial_exponents(predictors.shape[1],degree)
    design=_polynomial_design(predictors[fit],exponents=exponents,mean=mean,scale=scale)
    gram=design.T@design
    gram.flat[::len(gram)+1]+=1e-6
    gram[0,0]-=1e-6
    rhs=design.T@response[fit]
    coefficient=cho_solve(cho_factor(gram,overwrite_a=True,check_finite=False),rhs,check_finite=False)
    error=response[fit]-design@coefficient
    variance=float(np.var(error))
    if not np.isfinite(variance) or variance<=0: raise ValueError('Invalid residual variance')
    train_mse=float(np.mean(error**2));del design,gram
    predictions=[]
    for start in range(0,len(evaluate),256):
        design=_polynomial_design(predictors[evaluate[start:start+256]],exponents=exponents,mean=mean,scale=scale)
        predictions.append(design@coefficient)
    error=response[evaluate]-np.concatenate(predictions)
    log_prob=-.5*(np.log(2*np.pi*variance)+error**2/variance)
    result=dict(degree=degree,predictor_dimension=predictors.shape[1],feature_count=len(exponents),
                train_mse=train_mse,test_mse=float(np.mean(error**2)),
                variance_inflation=float(np.mean(error**2)/variance),
                heldout_log_prob_bits=float(log_prob.mean()/np.log(2)),
                elapsed_seconds=time.monotonic()-started)
    return result,log_prob,coefficient


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    n=16384;cp=1;lead=18
    history=sample_full_history_mode_inputs(n_samples=n,intervention_bound=4,seed=20260901).astype(float)
    path=ROOT/f'results/unicm_xi_hierarchy_uniform_n16384/cache/checkpoint{cp}_samples{n}_sampling20260901_bound4_fullhist12_start0_cpu.npz'
    with np.load(path) as d: y=d['all_mode_targets'][:,lead-1,:].astype(float)
    permutation=np.random.default_rng(20260905).permutation(n)
    fit=permutation[:12288];evaluate=permutation[12288:]
    own=history[:,:,3];others=history[:,:,np.arange(11)!=3].reshape(n,-1)
    predictors={'reduced':np.c_[y,own[:,:-1]],'full':np.c_[y,others,own[:,:-1]]}
    contract=dict(checkpoint=cp,lead=lead,n_samples=n,train_samples=len(fit),evaluate_samples=len(evaluate),
                  sampling_seed=20260901,split_seed=20260905,intervention_bound=4,source_history_months=12,
                  target_modes=11,source_coalition_order=11,partition='IOB | other ten modes',
                  probe='last IOB history coordinate conditional density only; NOT full CMI or minimum Syn',
                  degree_levels=[1,2],ridge=1e-6,source_sampling='independent uniform',
                  syn_tolerance_bits=1e-4,projection='none',
                  stop_rule='Do not extend to full CMI if degree 2 worsens held-out full-component log score by more than 3 paired standard errors',
                  limitation='conditional density diagnostic does not impose a coherent independent prior on all fitted marginals')
    rows=[];scores={};weights={}
    for degree in [1,2]:
        for kind,p in predictors.items():
            r,s,w=fit_component(p,own[:,-1],fit,evaluate,degree)
            r['conditioning']=kind;rows.append(r);scores[f'{kind}_{degree}']=s;weights[f'{kind}_{degree}']=w
            print(json.dumps(r),flush=True)
            (OUT/'summary.json').write_text(json.dumps(dict(status='running',contract=contract,rows=rows),indent=2))
    delta=(scores['full_2']-scores['full_1'])/np.log(2)
    mean=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
    failed=mean < -3*se
    result=dict(status='pilot_complete',contract=contract,rows=rows,
                quadratic_minus_affine_heldout_bits=mean,paired_standard_error_bits=se,
                expand_full_cmi=not failed,
                conclusion='stop: quadratic conditional density generalizes worse' if failed else 'component check passed; full CMI still untested')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(OUT/'component_fit.npz',**weights,**{f'logprob_{k}':v for k,v in scores.items()},fit=fit,evaluate=evaluate)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
