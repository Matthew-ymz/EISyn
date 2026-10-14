"""Nine-node nested extension of the weighted Gaussian public-readout case.

Seven independent channels carry the actual 2/3/4/5/9-source mechanisms.
Every node contains seven independent drivers and seven noisy readouts.
Information is evaluated by exact, factorized Cholesky TMs of the known
Gaussian joint. The all-source target remains the full 126-dimensional future.
No fitted covariance approximation, parameter scan, or hidden Syn projection.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import multivariate_normal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.spt import all_nonempty_subsets, nontrivial_bipartitions
from scripts.spt_continuous_phase_benchmark import SOURCES, TOLERANCE_NATS, block_tree, evaluate, score

SUPPORTS = ((1,2,3),(4,5),(6,7),(8,9),(1,2,3,4,5),(6,7,8,9),SOURCES)


class NestedGaussianBroadcast:
    sources = SOURCES
    supports = SUPPORTS
    channels = len(SUPPORTS)
    coordinates_per_node = 2*channels
    dimension = len(SOURCES)*coordinates_per_node

    def __init__(self):
        self.weights = np.zeros((self.channels,len(SOURCES)))
        for g,edge in enumerate(SUPPORTS):
            coefficients = ([1.,.5,.5] if g==0 else [1.,.5]) if g<4 else .25 if g<6 else .125
            self.weights[g,np.asarray(edge)-1] = coefficients
        self.variances = np.sum(self.weights**2,axis=1)
        self.transition = np.zeros((self.dimension,self.dimension))
        self.noise_covariance = np.eye(self.dimension)
        for g in range(self.channels):
            drivers = np.arange(9)*self.coordinates_per_node+g
            readouts = drivers+self.channels
            self.transition[np.ix_(readouts,drivers)] = self.weights[g][None,:]
            self.noise_covariance[np.ix_(readouts,readouts)] += 1.
        self.noise_cholesky = np.linalg.cholesky(self.noise_covariance)
        self.future_covariance = self.transition@self.transition.T+self.noise_covariance
        self.joint_covariance = np.block([[np.eye(self.dimension),self.transition.T],
                                         [self.transition,self.future_covariance]])
        self.joint_cholesky = np.linalg.cholesky(self.joint_covariance)

    @staticmethod
    def logdet(covariance):
        return float(2*np.log(np.diag(np.linalg.cholesky(covariance))).sum())

    @lru_cache(maxsize=None)
    def _target_logdet(self,g,k):
        return self.logdet(np.eye(k)+(self.variances[g]+1)*np.ones((k,k)))

    @lru_cache(maxsize=None)
    def _factor_info(self,source,k):
        idx = np.asarray(source)-1
        values = []
        for g in range(self.channels):
            cross = np.broadcast_to(self.weights[g,idx],(k,len(idx)))
            future = np.eye(k)+(self.variances[g]+1)*np.ones((k,k))
            joint = np.block([[np.eye(len(idx)),cross.T],[cross,future]])
            values.append(.5*(self._target_logdet(g,k)-self.logdet(joint)))
        # Independent driver/readout channels justify exact factorization;
        # unused past readouts and fresh future drivers cancel in MI.
        return math.fsum(values)

    def info(self,source,target=SOURCES):
        return self._factor_info(tuple(sorted(set(source))),len(set(target)))

    def formula_info(self,source,target=SOURCES):
        q = np.sum(self.weights[:,np.asarray(tuple(source))-1]**2,axis=1)
        a = 1+1/len(set(target))
        return float(.5*np.log1p(q/(a+self.variances-q)).sum())

    def future_mi(self,a,b):
        if a==b:
            raise ValueError('Expected two distinct future nodes')
        return float(.5*np.log1p((self.variances+1)**2/(2*self.variances+3)).sum())

    def step(self,source,rng,*,return_drive=False):
        if source.shape[1:]!=(9,self.coordinates_per_node):
            raise ValueError('Expected nine full node states')
        drive = np.einsum('nig,gi->ng',source[:,:,:self.channels],self.weights)
        signals = drive+rng.normal(size=drive.shape)
        target = rng.normal(size=source.shape)
        target[:,:,self.channels:] += signals[:,None,:]
        return (target,drive) if return_drive else target

    def conditional_tm(self,source,target):
        residual = target.reshape(-1,self.dimension)-source.reshape(-1,self.dimension)@self.transition.T
        return solve_triangular(self.noise_cholesky,residual.T,lower=True).T


def validate(model):
    subsets = all_nonempty_subsets(SOURCES)
    error = max(abs(model.info(s,SOURCES[:k])-model.formula_info(s,SOURCES[:k]))
                for s in subsets for k in range(1,10))
    # The future nodes are exchangeable readouts. All 511 source subsets and
    # nine target sizes cover every one of the 511^2 information queries.
    if error>1e-11:
        raise ArithmeticError(f'Factorized TM / sufficient-statistic mismatch: {error}')
    dense_error = 0.
    rng = np.random.default_rng(20261014)
    for _ in range(32):
        source = tuple(sorted(rng.choice(SOURCES,size=int(rng.integers(1,10)),replace=False)))
        target = tuple(sorted(rng.choice(SOURCES,size=int(rng.integers(1,10)),replace=False)))
        a = tuple(j for i in source for j in range((i-1)*model.coordinates_per_node,i*model.coordinates_per_node))
        b = tuple(model.dimension+j for i in target for j in range((i-1)*model.coordinates_per_node,i*model.coordinates_per_node))
        covariance = model.joint_covariance
        dense = .5*(model.logdet(covariance[np.ix_(a,a)])+model.logdet(covariance[np.ix_(b,b)])
                    -model.logdet(covariance[np.ix_(a+b,a+b)]))
        dense_error = max(dense_error,abs(dense-model.info(source,target)))
    if dense_error>1e-10:
        raise ArithmeticError(f'Full-joint TM / factorization mismatch: {dense_error}')
    future_error = 0.
    for left in range(1,10):
        for right in range(left+1,10):
            a = tuple(range((left-1)*model.coordinates_per_node,left*model.coordinates_per_node))
            b = tuple(range((right-1)*model.coordinates_per_node,right*model.coordinates_per_node))
            covariance = model.future_covariance
            dense = .5*(model.logdet(covariance[np.ix_(a,a)])+model.logdet(covariance[np.ix_(b,b)])
                        -model.logdet(covariance[np.ix_(a+b,a+b)]))
            future_error = max(future_error,abs(dense-model.future_mi(left,right)))
    if future_error>1e-11:
        raise ArithmeticError(f'Future MI / full covariance mismatch: {future_error}')
    source = rng.normal(size=(8192,9,model.coordinates_per_node))
    target = model.step(source,rng)
    reference = model.conditional_tm(source,target)
    if np.max(abs(reference.mean(0)))>.08 or np.max(abs(reference.var(0)-1))>.10:
        raise ArithmeticError('Real sampler / conditional TM mismatch')
    residual = target[:64].reshape(64,-1)-source[:64].reshape(64,-1)@model.transition.T
    direct = multivariate_normal(cov=model.noise_covariance).logpdf(residual)
    tm_density = (-.5*np.sum(reference[:64]**2+np.log(2*np.pi),axis=1)
                  -np.log(np.diag(model.noise_cholesky)).sum())
    density_error = float(np.max(abs(direct-tm_density)))
    if density_error>1e-10:
        raise ArithmeticError(f'Conditional TM density mismatch: {density_error}')
    return {'joint_dimension':2*model.dimension,'joint_rank':int(np.linalg.matrix_rank(model.joint_covariance)),
            'minimum_joint_eigenvalue':float(np.linalg.eigvalsh(model.joint_covariance).min()),
            'source_target_size_queries':4599,'queries_covered_by_target_exchangeability':511**2,
            'tm_formula_max_error_nats':error,'full_joint_checks':32,'full_joint_max_error_nats':dense_error,
            'all_future_pairs_max_error_nats':future_error,
            'conditional_density_max_error_nats':density_error,'sampler_iid_samples':8192}


def run(permutations=None):
    model = NestedGaussianBroadcast()
    if permutations is None:
        rng = np.random.default_rng(20261013)
        permutations = [SOURCES,*(tuple(map(int,rng.permutation(SOURCES))) for _ in range(19))]
    result = {'protocol':{'case':'A','model':'nested_gaussian_public_readout','nodes':9,
        'coordinates_per_node':model.coordinates_per_node,'source_coordinates':model.dimension,
        'target_coordinates':model.dimension,'intervention':'iid N(0,1)^126, fixed mean zero / variance one',
        'horizon_steps':1,'shared_noise_variance':1.,'observation_noise_variance':1.,
        'channel_weights':model.weights.tolist(),'units':'nats','syn_tolerance_nats':TOLERANCE_NATS,
        'clipping':False,'estimator':'exact known-joint Cholesky TM, factored across independent channels',
        'label_check':'identity plus 19 common bijections, same schedule as phase B/C; no physical repetitions',
        'selection':'constructive nested extension, fixed rational weights, no parameter scan',
        'normalization':'Phi_R/min(|L|,|R|), each recursive node; target is its local future block',
        'scoring':'same nonroot support metric as B/C; internal refinements of primitive supports ignored',
        'manuscript':{'parent':'P6UJCVG8','main':'DXGC7JEA','supplement':'MWIWKSVG','checked':'2026-10-14',
            'version':'sole main/SI, no explicit manuscript date/version',
            'locations':'Methods Eqs(10)-(12), SI S1.2 and S5.1'}},
        'validation':validate(model),'records':[]}
    result['root_candidates'] = []
    for left,right in nontrivial_bipartitions(SOURCES):
        phi = model.info(SOURCES)-model.info(left,left)-model.info(right,right)+min(
            model.info(u,v) for u in (left,right) for v in (left,right))
        result['root_candidates'].append({'left':left,'right':right,
            'syn_nats':model.info(SOURCES)-model.info(left)-model.info(right),
            'block_phi_r_nats':phi,'normalized_block_phi_r':phi/min(len(left),len(right))})
    for permutation_id,permutation in enumerate(permutations):
        original = dict(zip(SOURCES,permutation)); labels = {v:k for k,v in original.items()}
        def info(a,b=SOURCES):
            return model.info(tuple(original[i] for i in a),tuple(original[i] for i in b))
        record = evaluate(info,lambda a,b:model.future_mi(original[a],original[b]),sources=SOURCES)
        supports = tuple(tuple(sorted(labels[i] for i in e)) for e in SUPPORTS)
        root,clades,values = block_tree(info,SOURCES,normalize=True)
        if min(values)<-TOLERANCE_NATS:
            raise ArithmeticError(f'Normalized Block raw PhiR violation: minimum={min(values)}; '
                                  f'threshold={-TOLERANCE_NATS}; affected_count={sum(v < -TOLERANCE_NATS for v in values)}')
        record['methods']['normalized_block_phi_r'] = {'root_split':root,'clades':clades,'status':'resolved'}
        record['methods'] = {m:score(v,supports) for m,v in record['methods'].items()}
        record.update(case='A',variant=0,permutation_id=permutation_id,permutation=permutation,
                      hyperedges=supports,model='nested_gaussian_public_readout')
        result['records'].append(record)
    return result


def simulate(steps=80,samples=4096,seed=20261014):
    model = NestedGaussianBroadcast()
    sequence = np.random.SeedSequence(seed).spawn(steps+3)
    state = np.random.default_rng(sequence[0]).normal(size=(1,9,model.coordinates_per_node))
    states,means = [state[0].copy()],[]
    for t in range(steps):
        state = model.step(state,np.random.default_rng(sequence[t+1]))
        states.append(state[0].copy()); means.append(state[0,:,model.channels:].mean(0))
    source = np.random.default_rng(sequence[-2]).normal(size=(samples,9,model.coordinates_per_node))
    future,drive = model.step(source,np.random.default_rng(sequence[-1]),return_drive=True)
    # The largest proper mechanism matches the phase panels' selection rule.
    slot = max((g for g,e in enumerate(SUPPORTS) if len(e)<9),key=lambda g:len(SUPPORTS[g]))
    return {'model':'nested_gaussian_public_readout','supports':SUPPORTS,'states':np.asarray(states),
            'joint':np.asarray(means),'response_edge':SUPPORTS[slot],'source_sum':drive[:,slot],
            'response':future[:,:,model.channels+slot].mean(1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/spt_nested_broadcast_case/summary.json')
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'validation':result['validation'],
        'mean_f1':{m:float(np.mean([r['methods'][m]['clade_f1'] for r in result['records']])) for m in result['records'][0]['methods']},
        'minimum_syn_nats':min(r['audit']['minimum_syn_nats'] for r in result['records'])}))


if __name__=='__main__':
    main()
