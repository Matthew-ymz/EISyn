"""Statewise SURD of the frozen matched gLV intervention joint, robust proposals."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.spt_matched_common import FOLDER,deadline,fitted_model,model_sha,read,state,guarded,diagnostic_issues,quality
from scripts.spt_glv_finite_tm import proposal_sample,proposal_log_prob,LOG_TWO_PI
from scripts.spt_glv_group_tm import RealGroupIntegrator,conditional_proposal
from scripts.spt_glv_surd import PILOT_MASKS,audit_and_allocate
from scripts.compare_spt_surd_three_cases import maximum_weight_tree
from scripts.spt_kuramoto_finite_tm import check_time,save_json


class Specific:
    def __init__(self,model,outer,post,inner,repeat,end):
        check_time(end)
        self.outer,self.post,self.inner,self.end=outer,post,inner,end
        seed=20261015+1000*repeat
        self.meta=dict(version='matched-surd-group-proposal-v1',model_sha=model_sha(model),outer=outer,
                       post=post,inner=inner,seed=seed,posterior='90% bounded optimized Gaussian +10% std-times2',
                       conditional='existing real-group-v4 full missing 80/15/5 with row refinement',
                       source='same factorized uniform intervention',target='fixed complete nine-dimensional future')
        token=hashlib.sha256(json.dumps(self.meta,sort_keys=True).encode()).hexdigest()[:24]
        self.folder=FOLDER/'surd_cache'/token;self.folder.mkdir(parents=True,exist_ok=True)
        self.base=RealGroupIntegrator(model,outer=outer,max_particles=256,seed=seed,deadline=end,
                                     cache_dir=self.folder/'base',proposal='full')
        self.denominator,den_diag=self.base.marginal(0,511,256)
        center,chol,step=conditional_proposal(model,self.base.source,self.base.target,np.arange(9),np.arange(9))
        u=qmc.Sobol(10,scramble=True,seed=seed+300000).random_base2(int(np.log2(outer*128))).reshape(outer,128,10)[:,:post]
        local=proposal_sample(center,chol,u[:,:,1:]);broad=proposal_sample(center,chol*2,u[:,:,1:])
        sources=np.where((u[:,:,0]<.1)[:,:,None],broad,local)
        logq=np.logaddexp(np.log(.9)+proposal_log_prob(sources,center,chol),np.log(.1)+proposal_log_prob(sources,center,chol*2))
        flat=sources.reshape(-1,9);target=np.repeat(self.base.target,post,axis=0)
        logw=model.log_prob(flat,target).reshape(outer,post)-9*LOG_TWO_PI-logq
        normalized=logw-logsumexp(logw,axis=1)[:,None];self.weights=np.exp(normalized)
        posterior=dict(ess_minimum=float(np.exp(-logsumexp(2*normalized,axis=1)).min()),
                       maximum_normalized_weight=float(self.weights.max()),max_projected_step=step)
        self.common=dict(denominator=den_diag,posterior=posterior)
        self.common_issues=sorted({s for d in self.common.values() for s in diagnostic_issues(d)})
        self.meta['posterior_rows_sha256']=hashlib.sha256(flat.tobytes()+target.tobytes()+self.weights.tobytes()).hexdigest()
        self.conditional=RealGroupIntegrator(model,outer=outer*post,max_particles=256,seed=seed+400000,
                                            deadline=end,cache_dir=self.folder/('conditional_'+self.meta['posterior_rows_sha256'][:16]),proposal='full')
        # These are fixed posterior-evaluation rows, not uniform intervention rows.
        # Their complete identity is bound by this directory's grid/model token.
        self.conditional.source=flat;self.conditional.target=target

    def specific(self,mask):
        check_time(self.end);path=self.folder/f'specific_{mask:03d}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as data:
                if json.loads(str(data['meta']))!=self.meta:raise ValueError('Specific cache provenance mismatch')
                return data['specific'].copy(),json.loads(str(data['diagnostic']))
        values,d=self.conditional.marginal(mask,511,self.inner)
        specific=np.sum(self.weights*(values.reshape(self.outer,self.post)-self.denominator[:,None]),axis=1)
        if not np.isfinite(specific).all():raise ArithmeticError('Nonfinite target-specific KL')
        issues=sorted(set(self.common_issues+diagnostic_issues(d)))
        violations=int((specific < -.01).sum())
        if violations:issues.append('specific_below_negative_tolerance')
        diagnostic=dict(**d,issues=issues,minimum_specific_nats=float(specific.min()),
                        significant_specific_violation_count=violations,tolerance_nats=.01,clipping=False)
        temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,specific=specific,meta=json.dumps(self.meta),diagnostic=json.dumps(diagnostic));temporary.replace(path)
        self.conditional.cache.clear()
        return specific,diagnostic


def pilot(model,end,result):
    path=FOLDER/'surd_pilot.json';payload=read(path) if path.exists() else dict(status='running',records=[])
    grids=((64,64,64),(64,128,64),(128,128,64),(128,128,256))
    for n,p,k in grids:
        for repeat in range(4):
            if any((r['outer'],r['post'],r['inner'],r['repeat'])==(n,p,k,repeat) for r in payload['records']):continue
            estimator=Specific(model,n,p,k,repeat,end);row=dict(outer=n,post=p,inner=k,repeat=repeat,information_nats={},diagnostics={},common=estimator.common)
            for mask in PILOT_MASKS:
                value,d=estimator.specific(mask);row['information_nats'][str(mask)]=float(value.mean());row['diagnostics'][str(mask)]=d
            payload['records'].append(row);save_json(path,payload)
            result['pilot_completed_settings']=len(payload['records']);state('surd',result)
    arrays=[np.array([[r['information_nats'][str(m)] for m in PILOT_MASKS] for r in payload['records'] if (r['outer'],r['post'],r['inner'])==grid]) for grid in grids]
    changes=[float(abs(a.mean(0)-b.mean(0)).max()) for a,b in zip(arrays,arrays[1:])]
    final=[r for r in payload['records'] if (r['outer'],r['post'],r['inner'])==grids[-1]]
    issues=sorted({s for r in final for d in r['diagnostics'].values() for s in d['issues']})
    sem=float(arrays[-1].std(0,ddof=1).max()/2)
    if any(v>=.01 for v in changes):issues.append('grid_mean_change>=.01')
    if sem>=.01:issues.append('information_SEM>=.01')
    payload.update(status='precision_passed' if not issues else 'numerically_unresolved',failed_checks=issues,
                   maximum_changes_nats=changes,maximum_sem_nats=sem,grid=dict(outer=128,post=128,inner=256))
    save_json(path,payload);return payload


def worker():
    end=deadline();model=fitted_model();path=FOLDER/'surd.json';result=read(path) if path.exists() else dict(status='running',replicates=[])
    result.update(status='running',stage='precision_pilot',model_sha=model_sha(model),tolerance_nats=.01,clipping=False,
                  adaptation='SURD of intervention joint, statewise allocation then average, global maximum-weight laminar tree')
    state('surd',result)
    try:
        checked=pilot(model,end,result);result['pilot_status']=checked['status']
        if checked['status']!='precision_passed':
            result.update(status='not_evaluated_precision_failed',failed_checks=checked['failed_checks'],metrics=None);state('surd',result);return
        result['stage']='full_specific_tables';state('surd',result)
        for repeat in range(4):
            row=next((r for r in result['replicates'] if r['scramble']==repeat),None)
            if row and row.get('complete'):continue
            if row is None:
                row=dict(scramble=repeat,subsets=0,diagnostics={});result['replicates'].append(row)
            estimator=Specific(model,128,128,256,repeat,end)
            specific=np.full((128,512),np.nan);specific[:,0]=0.
            for mask in range(1,512):
                value,d=estimator.specific(mask);specific[:,mask]=value;row['subsets']=mask;row['diagnostics'][str(mask)]=d
                if mask%16==0:state('surd',result);print(f'SURD repeat={repeat} subsets={mask}/511',flush=True)
            red,syn,audit=audit_and_allocate(specific)
            row.update(complete=True,redundancy_unique_nats=red.tolist(),synergy_nats=syn.tolist(),mi_nats=specific.mean(0).tolist(),audit=audit)
            state('surd',result)
        rows=result['replicates'];issues=sorted({s for r in rows for d in r['diagnostics'].values() for s in d['issues']})
        mi=np.array([r['mi_nats'] for r in rows]);sem=float(mi.std(0,ddof=1).max()/2)
        if sem>=.01:issues.append('information_SEM>=.01')
        result.update(status='screen_passed' if not issues else 'numerically_unresolved',failed_checks=issues,maximum_information_sem_nats=sem,stage='ended',metrics=None)
        if not issues:
            syn=np.array([r['synergy_nats'] for r in rows]);tree=maximum_weight_tree(syn.mean(0))
            result.update(metrics=quality(tree['root_split'],tree['clades']),integration_replicates=[quality(t['root_split'],t['clades']) for t in map(maximum_weight_tree,syn)])
        state('surd',result)
    except Exception as error:
        result.update(status='budget_exhausted' if isinstance(error,TimeoutError) else 'failed',error=f'{type(error).__name__}: {error}',metrics=None)
        state('surd',result)
        if not isinstance(error,TimeoutError):raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    worker() if args.worker else guarded('surd',__file__,worker,resume=args.resume)
