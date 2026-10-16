"""Read-only frozen Kuramoto models/data, repaired natural WMS within total budget."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.spt_matched_common import FOLDER,deadline,read,state,guarded
from scripts.spt_kuramoto_natural_baselines import NaturalInformation,natural_proposal,information_audit,fixed_target_wms
from scripts.spt_kuramoto_natural_source import RotationalSource
from scripts.spt_kuramoto_flow_tm import PeriodicFlowConditionalTM
from scripts.spt_kuramoto_finite_tm import proposal_sample,proposal_log_prob,TWO_PI,DEFENSIVE,WIDE_FRACTION,WIDE_SCALE,check_time
from scripts.spt_continuous_phase_benchmark import score
from scripts.spt_local_hyperedge_kuramoto import SUPPORTS


class Context:
    def __init__(self,case,end):
        self.case,self.end=case,end
        self.original=ROOT/'results/spt_local_kuramoto_tm/ac_comparison'/case
        self.folder=FOLDER/f'kuramoto_{case}';self.folder.mkdir(exist_ok=True)
        frozen=read(self.original/'frozen_model.json');self.model=PeriodicFlowConditionalTM.load(frozen['model'])
        self.digest=hashlib.sha256(json.dumps(frozen['model'],sort_keys=True).encode()).hexdigest()
        prior=read(self.original/'natural_source.json')
        actual=hashlib.sha256((self.original/'natural_chains.npz').read_bytes()).hexdigest()
        if prior['natural_data_sha256']!=actual:raise ValueError('Frozen natural data changed')
        self.source_record=prior;self.prior=RotationalSource(prior['source']['centers'],prior['source']['kappa'])
    def load_model(self):return self.model,None,self.digest
    def deadline(self):return self.end
    def scored(self,root,clades):return score(dict(status='computed',root_split=root,clades=clades),SUPPORTS[self.case])


class RepairedInformation(NaturalInformation):
    def __init__(self,context,prior,outer,seed):
        super().__init__(context,prior,outer=outer,seed=seed)
        self.old_directory=context.original/'natural_marginals'/self.directory.name

    def values(self,mask,rows,particles,u,center,chol):
        missing=np.array([i for i in range(9) if not mask>>i&1])
        c,l=center[rows],chol[rows];normal=u[:,:,1+missing]
        sample=proposal_sample(c,l,normal);wide=proposal_sample(c,WIDE_SCALE*l,normal);branch=u[:,:,0]
        sample=np.where(((branch>=DEFENSIVE)&(branch<DEFENSIVE+WIDE_FRACTION))[:,:,None],wide,sample)
        sample=np.where((branch<DEFENSIVE)[:,:,None],normal*TWO_PI-np.pi,sample)
        logq=np.logaddexp(np.logaddexp(np.log(1-DEFENSIVE-WIDE_FRACTION)+proposal_log_prob(sample,c,l),
            np.log(WIDE_FRACTION)+proposal_log_prob(sample,c,WIDE_SCALE*l)),np.log(DEFENSIVE)-len(missing)*np.log(TWO_PI))
        x=np.repeat(self.source[rows,None,:],particles,axis=1);x[:,:,missing]=sample
        flat=x.reshape(-1,9);y=np.repeat(self.target[rows],particles,axis=0)
        weights=(self.model.log_prob(flat,y)+self.prior.log_prob(flat)).reshape(len(rows),particles)-self.prior.log_prob(self.source[rows],mask)[:,None]-logq
        normalizer=logsumexp(weights,axis=1);normalized=weights-normalizer[:,None]
        return normalizer-np.log(particles),np.exp(-logsumexp(2*normalized,axis=1)),np.exp(normalized.max(1))

    def marginal(self,mask,particles):
        repaired=self.directory/f'{mask}_{particles}_row_refined.npz'
        meta={**self.meta,'row_refinement':'fresh nested1024/4096/16384/65536; same periodic natural proposal and exact weights'}
        if repaired.exists():
            with np.load(repaired,allow_pickle=False) as data:
                if json.loads(str(data['meta']))!=meta:raise ValueError('Refined periodic provenance mismatch')
                return data['log_marginal'].copy(),json.loads(str(data['diagnostic']))
        native=self.directory/f'{mask}_{particles}.npz';original=self.old_directory/native.name
        if not native.exists() and original.exists():
            with np.load(original,allow_pickle=False) as data:
                if json.loads(str(data['meta']))!=self.meta:raise ValueError('Original periodic cache provenance mismatch')
            shutil.copy2(original,native)
        base,d=super().marginal(mask,particles)
        if d.get('ess_minimum') is None or (d['ess_minimum']>=10 and d['maximum_weight']<.25):return base,d
        check_time(self.context.deadline())
        missing=np.array([i for i in range(9) if not mask>>i&1]);center,chol=natural_proposal(self.model,self.prior,self.source,self.target,missing)
        values,ess,weight=self.values(mask,np.arange(self.outer),particles,self.inner[:,:particles],center,chol)
        if not np.allclose(values,base,rtol=0,atol=1e-9):raise ArithmeticError('Periodic native pointwise replay mismatch')
        records=[]
        for row in np.flatnonzero((ess<10)|(weight>=.25)):
            seed=int(np.random.SeedSequence([self.meta['seed'],mask,int(row),5522]).generate_state(1)[0])
            for p in (1024,4096,16384,65536):
                check_time(self.context.deadline());u=(qmc.Sobol(10,scramble=True,seed=seed,bits=30).random_base2(int(np.log2(p)))+.5/2**30)[None]
                estimate,e,w=self.values(mask,np.array([row]),p,u,center,chol)
                if e[0]>=10 and w[0]<.25:break
            values[row],ess[row],weight[row]=estimate[0],e[0],w[0]
            records.append(dict(row=int(row),particles=p,ess=float(e[0]),maximum_weight=float(w[0])))
        diagnostic={**d,'initial_diagnostic':d,'ess_minimum':float(ess.min()),'maximum_weight':float(weight.max()),'refined_rows':records}
        temporary=repaired.with_suffix('.tmp.npz');np.savez_compressed(temporary,log_marginal=values,diagnostic=json.dumps(diagnostic),meta=json.dumps(meta));temporary.replace(repaired)
        self.cache[mask,particles]=values,diagnostic
        return values,diagnostic


def worker(case):
    end=deadline();context=Context(case,end);stage=f'kuramoto_wms_{case}';path=FOLDER/f'{stage}.json'
    result=read(path) if path.exists() else dict(status='running',case=case,levels=[])
    result.update(status='running',source_data_sha256=context.source_record['natural_data_sha256'],model_sha256=context.digest,
        scope='same frozen natural joint; N1024/4096/8192 conditional numerical refinement, no source refit',metrics=None)
    state(stage,result)
    try:
        for outer in (1024,4096,8192):
            result['stage']=f'full_n{outer}';state(stage,result)
            level=next((r for r in result['levels'] if r['outer']==outer),None)
            if level is None:level=dict(outer=outer,replicates=[]);result['levels'].append(level)
            for repeat in range(4):
                row=next((r for r in level['replicates'] if r['scramble']==repeat),None)
                if row is None:row=dict(scramble=repeat,information_nats={},diagnostics={});level['replicates'].append(row)
                estimator=RepairedInformation(context,context.prior,outer,20261015+1000*repeat)
                for mask in range(1,512):
                    if str(mask) in row['information_nats']:continue
                    value,_,ds=estimator.information(mask,64);row['information_nats'][str(mask)]=value;row['diagnostics'][str(mask)]=ds
                    if mask%32==0:state(stage,result);print(f'{case} WMS N={outer} repeat={repeat} subsets={mask}',flush=True)
                state(stage,result)
            samples=np.array([[r['information_nats'][str(m)] for m in range(1,512)] for r in level['replicates']]);sem=float(samples.std(0,ddof=1).max()/2)
            changes=[]
            for repeat in range(4):
                estimator=RepairedInformation(context,context.prior,outer,20261015+1000*repeat)
                for mask in (1,3,7,24,31,96,384,480,511):
                    value,_,_=estimator.information(mask,256);changes.append(abs(value-level['replicates'][repeat]['information_nats'][str(mask)]))
            ds=[d for row in level['replicates'] for pair in row['diagnostics'].values() for d in pair if d.get('ess_minimum') is not None]
            audit=information_audit(samples);issues=[]
            if sem>=.01:issues.append('information_SEM>=.01')
            if max(changes)>=.01:issues.append('inner64_to256_change>=.01')
            if min(d['ess_minimum'] for d in ds)<10 or max(d['maximum_weight'] for d in ds)>=.25:issues.append('importance_weights_failed')
            if not audit['passed']:issues.append('MI_audit_failed')
            mean=samples.mean(0);previous=result['levels'][-2] if len(result['levels'])>1 else None
            outer_change=float(abs(mean-np.array(previous['mean_information_nats'])).max()) if previous else None
            if previous and outer_change>=.01:issues.append('outer_mean_change>=.01')
            level.update(complete=True,maximum_information_sem_nats=sem,maximum_inner_change_nats=max(changes),
                minimum_ess=min(d['ess_minimum'] for d in ds),maximum_weight=max(d['maximum_weight'] for d in ds),
                information_audit=audit,mean_information_nats=mean.tolist(),maximum_outer_change_nats=outer_change,failed_checks=issues)
            state(stage,result)
            if not issues:
                metric,wms_audit=fixed_target_wms(context,dict(zip(range(1,512),mean)))
                result.update(status='screen_passed',metrics=metric,audit=wms_audit,integration_replicates=[fixed_target_wms(context,{m:r['information_nats'][str(m)] for m in range(1,512)})[0] for r in level['replicates']],stage='ended')
                state(stage,result);return
        result.update(status='numerically_unresolved',failed_checks=level['failed_checks'],stage='ended');state(stage,result)
    except Exception as error:
        result.update(status='budget_exhausted' if isinstance(error,TimeoutError) else 'failed',error=f'{type(error).__name__}: {error}')
        state(stage,result)
        if not isinstance(error,TimeoutError):raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',choices=['A','C'],required=True);parser.add_argument('--worker',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    worker(args.case) if args.worker else guarded(f'kuramoto_wms_{args.case}',__file__,lambda:worker(args.case),worker_args=('--case',args.case),resume=args.resume)
