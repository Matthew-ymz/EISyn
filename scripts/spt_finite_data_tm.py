"""Finite-sample, data-only conditional affine TM -> EI -> SPT pilot.

The fitter receives paired samples and the registered coordinate domain only.
All EI queries are marginals of ONE fitted model under the product intervention.
Truth is used only by the simulator and post-fit diagnostics / support scoring.
For circular B/C, a fixed probit chart preserves the registered source support;
the affine conditional family is deliberately an approximation to their
non-Gaussian higher-order dynamics. This is not the oracle benchmark rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import ndtri

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from exp.TM.transport_map_density import AffineTransportMapDensityEstimator
from scripts.spt import all_nonempty_subsets
from scripts.spt_continuous_phase_benchmark import (
    SOURCES,TWO_PI,TOLERANCE_NATS,InformationOracle,PhaseKernel,block_tree,counterfactuals,evaluate,score,
)
from scripts.spt_nested_broadcast_case import NestedGaussianBroadcast

TRAIN_SIZES=(4096,16384,65536)
RIDGE=1e-6


def encode(samples,domain):
    x=np.asarray(samples,dtype=float)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite samples')
    if domain=='real': return x
    if domain!='phase': raise ValueError('Unknown coordinate domain')
    if np.any((x<=0)|(x>=TWO_PI)):
        raise ValueError('Phase chart reached an endpoint; no silent boundary clipping')
    return ndtri(x/TWO_PI)


def phase_log_jacobian(encoded):
    return .5*np.sum(encoded**2-np.log(TWO_PI),axis=1)


class FittedConditionalTM:
    """Learned dense transition, with one coherent independent-input joint."""
    def __init__(self,coefficient,intercept,noise,sample_size,domain,node_width):
        self.coefficient=np.asarray(coefficient)
        self.intercept=np.asarray(intercept)
        self.noise=np.asarray(noise)
        self.sample_size=sample_size; self.domain=domain; self.node_width=node_width
        self.dimension=len(intercept)
        if self.coefficient.shape!=(self.dimension,self.dimension) or self.dimension!=9*node_width:
            raise ValueError('Expected matching complete nine-node source / future dimensions')
        self.noise_cholesky=np.linalg.cholesky(noise)
        self.future=self.coefficient@self.coefficient.T+self.noise
        self.future_cholesky=np.linalg.cholesky(self.future)
        self.posterior=np.eye(self.dimension)-self.coefficient.T@cho_solve((self.future_cholesky,True),self.coefficient)
        self.posterior=(self.posterior+self.posterior.T)/2  # floating symmetry only, no spectrum / Syn projection
        joint=np.block([[np.eye(self.dimension),self.coefficient.T],[self.coefficient,self.future]])
        self.density=AffineTransportMapDensityEstimator(
            np.r_[np.zeros(self.dimension),self.intercept],joint,sample_size)
        self.columns={bits:np.array([j for i in range(9) if bits>>i&1
                                    for j in range(i*node_width,(i+1)*node_width)]) for bits in range(512)}

    @classmethod
    def fit(cls,source,target,*,domain):
        source=np.asarray(source); target=np.asarray(target)
        if source.shape!=target.shape or source.ndim!=3 or source.shape[1]!=9:
            raise ValueError('Expected paired complete nine-node arrays')
        n=len(source); width=source.shape[2]
        x=encode(source.reshape(n,-1),domain); y=encode(target.reshape(n,-1),domain)
        p=x.shape[1]
        if n<=p+1: raise ValueError('More training samples than regression parameters are required')
        xm=x.mean(0); ym=y.mean(0); xc=x-xm; yc=y-ym
        beta=np.linalg.solve(xc.T@xc+RIDGE*n*np.eye(p),xc.T@yc)
        intercept=ym-xm@beta
        residual=yc-xc@beta
        noise=residual.T@residual/(n-p-1)+RIDGE*np.eye(p)
        return cls(beta.T,intercept,noise,n,domain,width)

    @staticmethod
    def logdet(covariance):
        return float(2*np.log(np.diag(np.linalg.cholesky(covariance))).sum())

    @staticmethod
    def mask(nodes):
        return sum(1<<(i-1) for i in set(nodes))

    @lru_cache(maxsize=None)
    def target_logdet(self,target_mask):
        indices=self.columns[target_mask]
        return self.logdet(self.future[np.ix_(indices,indices)])

    @lru_cache(maxsize=None)
    def information(self,source_mask,target_mask):
        if not source_mask: return 0.
        source=self.columns[source_mask]
        if target_mask==511:
            return -.5*self.logdet(self.posterior[np.ix_(source,source)])
        target=self.columns[target_mask]; missing=self.columns[511^source_mask]
        remaining=self.coefficient[np.ix_(target,missing)]
        conditional=self.noise[np.ix_(target,target)]+remaining@remaining.T
        return .5*(self.target_logdet(target_mask)-self.logdet(conditional))

    def info(self,source,target=SOURCES):
        return self.information(self.mask(source),self.mask(target))

    def future_mi(self,a,b):
        aa,bb=1<<(a-1),1<<(b-1)
        return .5*(self.target_logdet(aa)+self.target_logdet(bb)-self.target_logdet(aa|bb))

    def heldout(self,source,target,true_log_prob):
        n=len(source)
        x=encode(source.reshape(n,-1),self.domain); y=encode(target.reshape(n,-1),self.domain)
        residual=y-(x@self.coefficient.T+self.intercept)
        z=solve_triangular(self.noise_cholesky,residual.T,lower=True).T
        log_prob=-.5*np.sum(z*z+np.log(TWO_PI),axis=1)-np.log(np.diag(self.noise_cholesky)).sum()
        if self.domain=='phase': log_prob+=phase_log_jacobian(y)
        difference=true_log_prob-log_prob
        marginal_z=solve_triangular(self.future_cholesky,(y-self.intercept).T,lower=True).T
        marginal=-.5*np.sum(marginal_z**2+np.log(TWO_PI),axis=1)-np.log(np.diag(self.future_cholesky)).sum()
        if self.domain=='phase': marginal+=phase_log_jacobian(y)
        # Held-out log ratio is a density-fit diagnostic under the real data
        # law, distinct from the model-mediated MI used for the tree.
        return {'samples':n,'nll_nats':float(-log_prob.mean()),
                'oracle_nll_nats':float(-true_log_prob.mean()),
                'excess_nll_nats':float(difference.mean()),
                'excess_nll_iid_sem_nats':float(difference.std(ddof=1)/np.sqrt(n)),
                'full_source_log_ratio_nats':float((log_prob-marginal).mean()),
                'reference_mean_max_abs':float(np.max(abs(z.mean(0)))),
                'reference_variance_max_error':float(np.max(abs(z.var(0)-1)))}


def sample_dataset(case,n_train,n_test,seed):
    # B/C share actual source draws. Their different kernels consume different
    # edge-noise streams, so only the inputs (not all noises) are paired.
    seed_group=0 if case=='A' else 1
    streams=np.random.SeedSequence([20261014,seed,seed_group]).spawn(4)
    if case=='A':
        model=NestedGaussianBroadcast(); width=model.coordinates_per_node
        x=np.random.default_rng(streams[0]).normal(size=(n_train,9,width))
        tx=np.random.default_rng(streams[2]).normal(size=(n_test,9,width))
        y=model.step(x,np.random.default_rng(streams[1])); ty=model.step(tx,np.random.default_rng(streams[3]))
        z=model.conditional_tm(tx,ty)
        true_log=-.5*np.sum(z*z+np.log(TWO_PI),axis=1)-np.log(np.diag(model.noise_cholesky)).sum()
        truth_info=model.info; supports=model.supports; domain='real'
    else:
        supports=counterfactuals()[case][0]; model=PhaseKernel(supports)
        x=np.random.default_rng(streams[0]).uniform(0,TWO_PI,size=(n_train,9,9))
        tx=np.random.default_rng(streams[2]).uniform(0,TWO_PI,size=(n_test,9,9))
        y=model.step(x,np.random.default_rng(streams[1])); ty=model.step(tx,np.random.default_rng(streams[3]))
        true_log=model.log_conditional(tx,ty); truth_info=InformationOracle(model).info; domain='phase'
    truth_ei={s:truth_info(s) for s in all_nonempty_subsets(SOURCES)}
    return x,y,tx,ty,true_log,truth_ei,supports,domain


def analyze_fit(case,seed,n,x,y,tx,ty,true_log,truth_ei,supports,domain):
    start=time.monotonic()
    # This call has no access to simulator, support, oracle, or test samples.
    fitted=FittedConditionalTM.fit(x[:n],y[:n],domain=domain)
    fit_seconds=time.monotonic()-start
    record=evaluate(fitted.info,fitted.future_mi)
    root,clades,values=block_tree(fitted.info,SOURCES,normalize=True)
    if min(values)<-TOLERANCE_NATS:
        raise ArithmeticError(f'Normalized Block candidate: minimum={min(values)}; '
                              f'threshold={-TOLERANCE_NATS}; affected_count={sum(v < -TOLERANCE_NATS for v in values)}')
    record['methods']['normalized_block_phi_r']={'root_split':root,'clades':clades,'status':'resolved'}
    record['methods']={m:score(v,supports) for m,v in record['methods'].items()}
    estimated={s:fitted.info(s) for s in truth_ei}
    errors=np.array([estimated[s]-truth_ei[s] for s in truth_ei])
    true_xi=truth_ei[SOURCES]-sum(truth_ei[(i,)] for i in SOURCES)
    record.update(case=case,seed=seed,n_train=n,domain=domain,status='valid',
        fit_seconds=fit_seconds,total_seconds=time.monotonic()-start,
        ei_nats={','.join(map(str,s)):v for s,v in estimated.items()},
        truth_ei_nats={','.join(map(str,s)):v for s,v in truth_ei.items()},
        error={'ei_rmse_nats':float(np.sqrt(np.mean(errors**2))),
               'ei_max_abs_nats':float(np.max(abs(errors))),
               'full_ei_error_nats':estimated[SOURCES]-truth_ei[SOURCES],
               'xi_error_nats':record['xi_nats']-true_xi,'truth_xi_nats':true_xi},
        heldout=fitted.heldout(tx,ty,true_log),
        parameter_digest=hashlib.sha256(fitted.coefficient.tobytes()+fitted.noise.tobytes()).hexdigest(),
        training_prefix_digest=hashlib.sha256(x[:n].tobytes()+y[:n].tobytes()).hexdigest(),
        test_digest=hashlib.sha256(tx.tobytes()+ty.tobytes()).hexdigest(),
        noise_minimum_eigenvalue=float(np.linalg.eigvalsh(fitted.noise).min()))
    record['total_seconds']=time.monotonic()-start
    return record


def run(output,*,sizes=TRAIN_SIZES,seeds=(0,1,2),cases=('A','B','C'),test_samples=8192,budget_seconds=600):
    start=time.monotonic()
    result={'protocol':{'train_sizes':sizes,'seeds':seeds,'cases':cases,'test_samples':test_samples,
        'variant':0,'labels':'identity, no label-repeat inflation','nodes':9,'horizon_steps':1,
        'source_target_coordinates':{'A':126,'B':81,'C':81},
        'source_distribution':'A iid N(0,1); B/C iid Uniform(0,2*pi)',
        'estimator':'dense fitted conditional affine triangular TM with registered independent source prior',
        'phase_chart':'fixed ndtri(theta/(2*pi)), inverse preserves bounded support; Jacobian included in raw density',
        'ei_definition':'mutual information under fitted conditional TM and registered input; model-mediated estimate',
        'si_estimator_difference':'not separate empirical source/joint fits or Eq(S13) bias correction',
        'ridge':RIDGE,'noise_divisor':'n - source_dimension - 1','syn_tolerance_nats':TOLERANCE_NATS,
        'units':'nats','clipping':False,'source_factorization':'exact known product intervention in latent coordinates',
        'training_information':'paired full-state arrays and domain; no truth supports or coefficients',
        'replication':'three independent sampled datasets, nested train prefixes, disjoint held-out data',
        'scope':'correct affine family A; model-expression diagnostic for nonlinear circular B/C',
        'spt':'canonical exact recursive raw residual; no planted split constraints',
        'budget_seconds':budget_seconds,
        'manuscript':{'parent':'P6UJCVG8','main':'DXGC7JEA','supplement':'MWIWKSVG','checked':'2026-10-14',
            'version':'sole available main/SI; no explicit manuscript date/version',
            'locations':'Methods Eqs(10)-(12); SI S1.2 Eqs(S7)-(S15), S5.1'}},'records':[],'status':'running'}
    output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        result['elapsed_seconds']=time.monotonic()-start
        temporary=output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result,indent=2,allow_nan=False)); temporary.replace(output)
    save()
    for case in cases:
        for seed in seeds:
            dataset=sample_dataset(case,max(sizes),test_samples,seed)
            for n in sizes:
                if time.monotonic()-start>budget_seconds:
                    result['status']='budget_exhausted'; save(); return result
                try:
                    record=analyze_fit(case,seed,n,*dataset)
                except (ArithmeticError,np.linalg.LinAlgError) as error:
                    record={'case':case,'seed':seed,'n_train':n,'status':'invalid_estimate','error':str(error)}
                result['records'].append(record); save()
                print(json.dumps({'case':case,'seed':seed,'n_train':n,'status':record['status'],
                    'spt_f1':record.get('methods',{}).get('spt',{}).get('clade_f1'),
                    'ei_rmse_nats':record.get('error',{}).get('ei_rmse_nats') if isinstance(record.get('error'),dict) else None,
                    'seconds':record.get('total_seconds')}),flush=True)
            del dataset
    result['status']='complete'; save(); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/spt_finite_data_tm/summary.json')
    parser.add_argument('--sizes',type=int,nargs='+',default=TRAIN_SIZES)
    parser.add_argument('--seeds',type=int,nargs='+',default=(0,1,2))
    parser.add_argument('--cases',nargs='+',choices=list('ABC'),default=list('ABC'))
    parser.add_argument('--test-samples',type=int,default=8192)
    parser.add_argument('--budget-seconds',type=int,default=600)
    args=parser.parse_args()
    result=run(args.output,sizes=tuple(sorted(args.sizes)),seeds=tuple(args.seeds),cases=tuple(args.cases),
               test_samples=args.test_samples,budget_seconds=args.budget_seconds)
    if result['status']!='complete' or any(r['status']!='valid' for r in result['records']): sys.exit(2)


if __name__=='__main__': main()
