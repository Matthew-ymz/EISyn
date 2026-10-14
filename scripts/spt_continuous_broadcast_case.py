"""Full-rank Gaussian public-readout counterexamples, symmetric and weighted.

Four nodes, three real coordinates each; independent standard-Gaussian
interventions on all twelve coordinates. Information queries use the exact
Cholesky triangular TM of one known 24-dimensional cross-time joint. This is
an exact Gaussian model, not a covariance approximation to nonlinear data.
Fixed sigma^2=delta^2=1, horizon one, complete future target for SPT.
--weighted uses gains (1, 1/2, 1, 1/2), defeating the specific normalization
Phi_R/min(|L|,|R|). The default preserves the earlier symmetric control.
"""
from __future__ import annotations

import argparse
import itertools
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
from scripts.spt_continuous_phase_benchmark import evaluate, block_tree, TOLERANCE_NATS

SOURCES = (1,2,3,4)
SUPPORTS = ((1,2),(3,4))
WEIGHTED_GAINS = (1., .5, 1., .5)


class GaussianBroadcast:
    supports = SUPPORTS
    sources = SOURCES

    def __init__(self, gains=(1.,1.,1.,1.)):
        self.gains = tuple(float(v) for v in gains)
        if len(self.gains)!=4 or not all(np.isfinite(v) and v>0 for v in self.gains):
            raise ValueError('Expected four finite positive gains; both true supports must remain present')
        self.transition = np.zeros((12,12))
        self.noise_covariance = np.zeros((12,12))
        for i in range(4):
            self.noise_covariance[3*i,3*i] = 1.
            for g in range(2):
                row = 3*i+g+1
                self.transition[row,6*g] = self.gains[2*g]
                self.transition[row,6*g+3] = self.gains[2*g+1]
                for j in range(4):
                    self.noise_covariance[row,3*j+g+1] = 1.+float(i==j)
        self.future_covariance = self.transition@self.transition.T+self.noise_covariance
        self.joint_covariance = np.block([[np.eye(12),self.transition.T],
                                          [self.transition,self.future_covariance]])
        self.joint_cholesky = np.linalg.cholesky(self.joint_covariance)
        self.noise_cholesky = np.linalg.cholesky(self.noise_covariance)

    @staticmethod
    def indices(nodes):
        nodes = tuple(sorted(set(nodes)))
        if not nodes or not set(nodes)<=set(SOURCES):
            raise ValueError("Expected a nonempty subset of four source/target nodes")
        return tuple(j for i in nodes for j in range(3*(i-1),3*i))

    def step(self,source,rng,*,return_signals=False):
        source = np.asarray(source)
        if source.ndim!=3 or source.shape[1:]!=(4,3):
            raise ValueError("Expected four three-coordinate node states")
        n = len(source)
        sums = (source[:,:,0]*np.asarray(self.gains)).reshape(n,2,2).sum(2)
        signals = sums+rng.normal(size=(n,2))
        target = np.empty_like(source)
        target[:,:,0] = rng.normal(size=(n,4))
        target[:,:,1:] = signals[:,None,:]+rng.normal(size=(n,4,2))
        return (target,signals,sums) if return_signals else target

    def conditional_tm(self,source,target):
        mean = source.reshape(-1,12)@self.transition.T
        return solve_triangular(self.noise_cholesky,(target.reshape(-1,12)-mean).T,lower=True).T

    def conditional_log_prob(self,source,target):
        reference = self.conditional_tm(source,target)
        return (-.5*np.sum(reference**2+np.log(2*np.pi),axis=1)
                -np.log(np.diag(self.noise_cholesky)).sum())

    @lru_cache(maxsize=None)
    def _logdet(self,indices):
        covariance = self.joint_covariance[np.ix_(indices,indices)]
        return float(2*np.log(np.diag(np.linalg.cholesky(covariance))).sum())

    def _mi(self,a,b):
        return .5*(self._logdet(a)+self._logdet(b)-self._logdet(a+b))

    def info(self,source,target=SOURCES):
        a = self.indices(source)
        b = tuple(12+i for i in self.indices(target))
        return self._mi(a,b)

    def future_mi(self,a,b):
        return self._mi(tuple(12+i for i in self.indices((a,))),
                        tuple(12+i for i in self.indices((b,))))

    def formula_info(self,source,target=SOURCES):
        # k noisy replicate reads reduce observation noise to 1/k; the
        # shared channel noise retains variance one. Unobserved drivers
        # contribute their known gain-squared variances to the denominator.
        a = 1.+1./len(set(target))
        present = set(source)
        return math.fsum(.5*np.log1p(
            sum(self.gains[i-1]**2 for i in e if i in present)/
            (a+sum(self.gains[i-1]**2 for i in e if i not in present))) for e in SUPPORTS)


def score_four(output,supports):
    truth = {frozenset(e) for e in supports}
    found = {frozenset(e) for e in output['clades'] if 1<len(e)<4}
    common = len(truth&found)
    precision = common/len(found) if found else 0.
    recall = common/len(truth)
    return {**output,'clade_precision':precision,'clade_recall':recall,
            'clade_f1':2*precision*recall/(precision+recall) if precision+recall else 0.,
            'correct_root':{frozenset(e) for e in output['root_split']}==truth,
            'upper_support_recall':None}


def validate(model):
    subsets = all_nonempty_subsets(SOURCES)
    maximum = max(abs(model.info(a,b)-model.formula_info(a,b)) for a in subsets for b in subsets)
    if maximum>1e-12:
        raise ArithmeticError("Full-joint TM and independent sufficient-statistic information disagree")
    rng = np.random.default_rng(20261014)
    source = rng.normal(size=(8192,4,3))
    target = model.step(source,rng)
    reference = model.conditional_tm(source,target)
    if np.max(abs(reference.mean(0)))>.08 or np.max(abs(reference.var(0)-1))>.10:
        raise ArithmeticError("Sampler and conditional TM reference disagree")
    means = source[:64].reshape(64,12)@model.transition.T
    direct = np.array([multivariate_normal.logpdf(y,mean=m,cov=model.noise_covariance)
                       for m,y in zip(means,target[:64].reshape(64,12))])
    density_error = float(np.max(abs(model.conditional_log_prob(source[:64],target[:64])-direct)))
    if density_error>1e-10:
        raise ArithmeticError("Exact TM density and independent Gaussian density disagree")
    return {'joint_dimension':24,'joint_rank':int(np.linalg.matrix_rank(model.joint_covariance)),
            'minimum_joint_eigenvalue':float(np.linalg.eigvalsh(model.joint_covariance).min()),
            'cholesky_reconstruction_max_error':float(np.max(abs(model.joint_cholesky@model.joint_cholesky.T-model.joint_covariance))),
            'all_225_group_target_mi_max_error_nats':maximum,
            'conditional_density_max_error_nats':density_error,'sampler_check_iid_samples':8192}


def run(*, gains=(1.,1.,1.,1.)):
    model = GaussianBroadcast(gains)
    weighted = model.gains!= (1.,1.,1.,1.)
    result = {'protocol':{'case':'A','model':'full_rank_gaussian_public_readout','nodes':4,
        'coordinates_per_node':3,'source_coordinates':12,'target_coordinates':12,
        'intervention':'iid N(0,1)^12, fixed mean zero and variance one per coordinate',
        'horizon_steps':1,'shared_noise_variance':1.,'observation_noise_variance':1.,
        'driver_gains':model.gains,
        'estimator':'exact Cholesky triangular TM of the known 24-dimensional joint',
        'units':'nats','syn_tolerance_nats':TOLERANCE_NATS,'clipping':False,
        'label_check':'all 24 bijections; label/tie sensitivity, not physical repetitions',
        'truth':'two independent source mixing supports {1,2},{3,4}; noisy broadcast readout is shown separately',
        'normalization':'Phi_R(L,R)/min(number of nodes in L, number of nodes in R); minimize at each recursive split',
        'scope':('raw and min-size-normalized recursive Block-PhiR; no claim about all normalizations' if weighted
                 else 'unnormalized recursive Block-PhiR; normalized method succeeds in this symmetric control'),
        'selection':'constructive example with fixed rational gains; not a confirmatory random-system benchmark',
        'manuscript':{'parent':'P6UJCVG8','main':'DXGC7JEA','supplement':'MWIWKSVG',
            'checked':'2026-10-14','version':'sole available main/SI; no explicit manuscript date/version',
            'locations':'Methods pp15-17 Eqs(4)-(12); SI S1.2 pp3-4 and S5 pp8-9'},
        'design':('docs/reports/spt_continuous_hierarchy_refinement.md section 8' if weighted
                  else 'docs/ref/spt_hierarchy_comparison_design.md section 8.1')},
        'validation':validate(model),'records':[]}
    ei = {s:model.info(s) for s in all_nonempty_subsets(SOURCES)}
    result['all_block_candidates'] = []
    for group in all_nonempty_subsets(SOURCES):
        for left,right in nontrivial_bipartitions(group):
            redundancy = min(model.info(u,v) for u in (left,right) for v in (left,right))
            phi = model.info(group,group)-model.info(left,left)-model.info(right,right)+redundancy
            result['all_block_candidates'].append({'group':group,'left':left,'right':right,
                'syn_nats':ei[group]-ei[left]-ei[right],'block_phi_r_nats':phi,
                'normalized_block_phi_r':phi/min(len(left),len(right))})
    result['root_candidates'] = [c for c in result['all_block_candidates'] if c['group']==SOURCES]
    for permutation in itertools.permutations(SOURCES):
        original = dict(zip(SOURCES,permutation))
        labels = {v:k for k,v in original.items()}
        def info(a,b=SOURCES):
            return model.info(tuple(original[i] for i in a),tuple(original[i] for i in b))
        record = evaluate(info,lambda a,b:model.future_mi(original[a],original[b]),sources=SOURCES)
        supports = tuple(tuple(sorted(labels[i] for i in e)) for e in SUPPORTS)
        record.update(case='A',variant=0,permutation=permutation,hyperedges=supports,
                      model='gaussian_public_readout',driver_gains=tuple(model.gains[original[i]-1] for i in SOURCES))
        record['methods'] = {m:score_four(v,supports) for m,v in record['methods'].items()}
        root,clades,values = block_tree(info,SOURCES,normalize=True)
        if min(values)<-TOLERANCE_NATS:
            raise ArithmeticError(f'Normalized Block raw PhiR below tolerance: minimum={min(values)}; '
                                  f'threshold={-TOLERANCE_NATS}; affected_count={sum(v < -TOLERANCE_NATS for v in values)}')
        record['normalized_block_control'] = score_four({'root_split':root,'clades':clades,'status':'resolved'},supports)
        record['methods']['normalized_block_phi_r'] = record['normalized_block_control']
        record['audit']['normalized_block_phi_minimum_raw_nats'] = min(values)
        result['records'].append(record)
    return result


def simulate(steps=80,samples=4096,seed=20261014,*,gains=(1.,1.,1.,1.)):
    model = GaussianBroadcast(gains)
    sequence = np.random.SeedSequence(seed).spawn(steps+3)
    state = np.random.default_rng(sequence[0]).normal(size=(1,4,3))
    states,means,signals,drives = [state[0].copy()],[],[],[]
    for t in range(steps):
        future,z,total = model.step(state,np.random.default_rng(sequence[t+1]),return_signals=True)
        states.append(future[0].copy())
        means.append(future[0,:,1:].mean(0))
        signals.append(z[0]); drives.append(total[0])
        state = future
    intervention = np.random.default_rng(sequence[-2]).normal(size=(samples,4,3))
    future,z,total = model.step(intervention,np.random.default_rng(sequence[-1]),return_signals=True)
    return {'model':'gaussian_public_readout','supports':SUPPORTS,'driver_gains':model.gains,'states':np.asarray(states),
            'joint':np.asarray(means),'signal':np.asarray(signals),'drive':np.asarray(drives),
            'response_edge':(1,2),'source_sum':total[:,0],'response':future[:,:,1].mean(1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--weighted',action='store_true',help='Use the heterogeneous-gain normalized-Block counterexample')
    args = parser.parse_args()
    if args.output is None:
        folder = 'spt_normalized_block_counterexample' if args.weighted else 'spt_continuous_broadcast_case'
        args.output = ROOT/'results'/folder/'summary.json'
    result = run(gains=WEIGHTED_GAINS if args.weighted else (1.,1.,1.,1.))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'validation':result['validation'],'root_candidates':result['root_candidates'],
                     'mean_f1':{m:float(np.mean([r['methods'][m]['clade_f1'] for r in result['records']]))
                                for m in result['records'][0]['methods']}}))


if __name__=='__main__':
    main()
