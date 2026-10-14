"""Fixed finite-data B/C: sample -> learned coherent TM -> exact model EI -> SPT.

Truth is restricted to sampling and post-fit diagnostics. C learns a nonlinear
coordinate map from future training values, never the generating inverse.
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
from scipy.linalg import solve_triangular
from scipy.special import ndtri

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt import all_nonempty_subsets, nontrivial_bipartitions
from scripts.spt_continuous_phase_benchmark import SOURCES, TOLERANCE_NATS, evaluate, block_tree, score
from scripts.spt_finite_data_tm import FittedConditionalTM, RIDGE
from scripts.spt_nested_broadcast_case import NestedGaussianBroadcast, validate

SIZES = (4096, 16384, 65536)
SEEDS = (101, 102, 103)
KNOTS = 257
METHODS = ('spt', 'pair_phi_r_hac', 'pair_xi_hac', 'future_mi_hac',
           'block_phi_r', 'normalized_block_phi_r')
SUPPORTS = {
    'B': ((1,2),(1,2,3),(1,2,3,4),(1,2,3,4,5),(6,7),(6,7,8),(6,7,8,9),SOURCES),
    'C': ((1,2,3),(4,5,6),(7,8,9),(1,2,3,4,5,6),SOURCES),
}


class ReplacementBroadcast(NestedGaussianBroadcast):
    """Registered generator. Fitting routines receive none of this object's data."""
    def __init__(self, case):
        self.case = case
        self.supports = SUPPORTS[case]
        self.channels = len(self.supports)
        self.coordinates_per_node = 2*self.channels
        self.dimension = 9*self.coordinates_per_node
        self.weights = np.zeros((self.channels,9))
        for g, edge in enumerate(self.supports[:-1]):
            coefficient = (2. if len(edge)==2 else 3.) if case=='B' else (2. if len(edge)==3 else 1.)
            self.weights[g,np.asarray(edge)-1] = coefficient
        self.weights[-1,:] = .05
        self.weights[-1,[0,5] if case=='B' else [0,6]] = 3. if case=='B' else 1.5
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

    def step(self, source, rng, *, return_drive=False):
        target, drive = super().step(source,rng,return_drive=True)
        if self.case=='C':
            target[:,:,self.channels:] = np.arcsinh(target[:,:,self.channels:])
        return (target,drive) if return_drive else target

    def conditional_tm(self, source, target):
        latent = np.asarray(target).copy()
        if self.case=='C':
            latent[:,:,self.channels:] = np.sinh(latent[:,:,self.channels:])
        return super().conditional_tm(source,latent)

    def true_log_conditional(self, source, target):
        z = self.conditional_tm(source,target)
        value = -.5*np.sum(z*z+np.log(2*np.pi),axis=1)-np.log(np.diag(self.noise_cholesky)).sum()
        if self.case=='C':
            # log cosh computed stably. Only the truth diagnostic reads this.
            y = target[:,:,self.channels:]
            value += np.sum(np.logaddexp(y,-y)-np.log(2),axis=(1,2))
        return value

    def info(self, source, target=SOURCES):
        return self._factor_info(tuple(sorted(set(source))),len(set(target)))


class LearnedMonotoneTM:
    """Strictly increasing continuous map R^d -> R^d, with linear tails.

    Fixed evenly spaced order-statistic ranks, 257 knots per coordinate. Knot
    ordinates are normal scores. Neither inverse generator nor truth is used.
    """
    def __init__(self, raw_knots, scores):
        self.raw_knots = np.asarray(raw_knots)
        self.scores = np.asarray(scores)
        increments = np.diff(self.raw_knots,axis=0)
        if np.any(increments<=0):
            raise ArithmeticError('Normal-score knots must be strictly increasing; no tied-knot clipping')
        self.slopes = np.diff(self.scores)[:,None]/increments
        if not np.isfinite(self.slopes).all() or np.any(self.slopes<=0):
            raise ArithmeticError('Normal-score knots must be strictly increasing; no tied-knot clipping')

    @classmethod
    def fit(cls, target):
        target = np.asarray(target)
        n = len(target)
        if n<KNOTS or not np.isfinite(target).all():
            raise ValueError('Need finite training values and at least 257 samples')
        ranks = np.rint(np.linspace(0,n-1,KNOTS)).astype(int)
        return cls(np.sort(target,axis=0)[ranks],ndtri((ranks+.5)/n))

    def transform(self, target, *, jacobian=False):
        target = np.asarray(target)
        if target.ndim!=2 or target.shape[1]!=self.raw_knots.shape[1] or not np.isfinite(target).all():
            raise ValueError('Expected finite complete future coordinates')
        z = np.empty_like(target,dtype=float)
        log_jac = np.zeros(len(target))
        for j in range(target.shape[1]):
            # Restrict segment index, never value/domain. End segments extend
            # with their positive slope to +/- infinity.
            segment = np.searchsorted(self.raw_knots[:,j],target[:,j],side='right')-1
            segment = np.minimum(np.maximum(segment,0),len(self.scores)-2)
            slope = self.slopes[segment,j]
            z[:,j] = self.scores[segment]+slope*(target[:,j]-self.raw_knots[segment,j])
            log_jac += np.log(slope)
        return (z,log_jac) if jacobian else z

    def inverse(self, z):
        z = np.asarray(z)
        result = np.empty_like(z,dtype=float)
        for j in range(z.shape[1]):
            segment = np.searchsorted(self.scores,z[:,j],side='right')-1
            segment = np.minimum(np.maximum(segment,0),len(self.scores)-2)
            result[:,j] = self.raw_knots[segment,j]+(z[:,j]-self.scores[segment])/self.slopes[segment,j]
        return result


class LearnedConditionalTM:
    def __init__(self, gaussian, encoder):
        self.gaussian = gaussian
        self.encoder = encoder

    @classmethod
    def fit(cls, source, target, *, nonlinear):
        # Only observed training arrays and a registered model-family flag.
        encoder = LearnedMonotoneTM.fit(target.reshape(len(target),-1)) if nonlinear else None
        encoded = encoder.transform(target.reshape(len(target),-1)).reshape(target.shape) if nonlinear else target
        return cls(FittedConditionalTM.fit(source,encoded,domain='real'),encoder)

    def heldout(self, source, target, true_log):
        shape = target.shape
        encoded = target.reshape(len(target),-1)
        jac = np.zeros(len(target))
        if self.encoder is not None:
            encoded,jac = self.encoder.transform(encoded,jacobian=True)
        result = self.gaussian.heldout(source,encoded.reshape(shape),true_log-jac)
        result['nll_nats'] -= float(jac.mean())
        result['oracle_nll_nats'] -= float(jac.mean())
        result['raw_log_jacobian_mean'] = float(jac.mean())
        if self.encoder is not None:
            restored = self.encoder.inverse(encoded)
            error = float(np.max(abs(restored-target.reshape(len(target),-1))))
            if error>1e-9:
                raise ArithmeticError(f'Learned TM inversion failed: {error}')
            result['inverse_max_abs_error'] = error
            lo,hi = self.encoder.raw_knots[[0,-1]]
            result['linear_tail_value_count'] = int(np.count_nonzero((target.reshape(len(target),-1)<lo)|(target.reshape(len(target),-1)>hi)))
            result['minimum_map_slope'] = float(self.encoder.slopes.min())
        return result


def label_schedule():
    rng = np.random.default_rng(20261013)
    return [SOURCES,*(tuple(map(int,rng.permutation(SOURCES))) for _ in range(19))]


def evaluate_labels(fitted, supports, case, seed, n, *, deadline):
    records = []
    for index, permutation in enumerate(label_schedule()):
        if time.monotonic()>deadline:
            raise TimeoutError('Cumulative experiment compute budget reached')
        to_original = dict(zip(SOURCES,permutation))
        to_label = {v:k for k,v in to_original.items()}
        def info(a,b=SOURCES):
            return fitted.info(tuple(to_original[i] for i in a),tuple(to_original[i] for i in b))
        record = evaluate(info,lambda a,b:fitted.future_mi(to_original[a],to_original[b]))
        root,clades,values = block_tree(info,SOURCES,normalize=True)
        minimum = min(values)
        count = sum(v < -TOLERANCE_NATS for v in values)
        if count:
            raise ArithmeticError(f'Normalized Block violation: minimum={minimum}; threshold={-TOLERANCE_NATS}; affected_count={count}')
        record['audit']['normalized_block_minimum_nats'] = minimum
        record['audit']['normalized_block_tolerance_negative_count'] = sum(-TOLERANCE_NATS<=v<0 for v in values)
        record['methods']['normalized_block_phi_r'] = {'root_split':root,'clades':clades,'status':'resolved'}
        labeled = tuple(tuple(sorted(to_label[i] for i in e)) for e in supports)
        record['methods'] = {m:score(v,labeled) for m,v in record['methods'].items()}
        record.update(case=case,seed=seed,n_train=n,variant=0,permutation_id=index,
                      permutation=permutation,hyperedges=labeled)
        records.append(record)
    return records


def analyze(case, seed, n, dataset, *, deadline):
    start = time.monotonic()
    x,y,tx,ty,true_log,model = dataset
    learned = LearnedConditionalTM.fit(x[:n],y[:n],nonlinear=case=='C')
    fit_seconds = time.monotonic()-start
    fitted = learned.gaussian
    heldout = learned.heldout(tx,ty,true_log)
    label_records = evaluate_labels(fitted,model.supports,case,seed,n,deadline=deadline)
    subsets = all_nonempty_subsets(SOURCES)
    truth = {s:model.info(s) for s in subsets}
    estimate = {s:fitted.info(s) for s in subsets}
    errors = np.array([estimate[s]-truth[s] for s in subsets])
    root_candidates = sorted(fitted.info(SOURCES)-fitted.info(l)-fitted.info(r)
                             for l,r in nontrivial_bipartitions(SOURCES))
    payload = fitted.coefficient.tobytes()+fitted.noise.tobytes()
    if learned.encoder is not None:
        payload += learned.encoder.raw_knots.tobytes()+learned.encoder.scores.tobytes()
    result = {'case':case,'seed':seed,'n_train':n,'status':'valid','fit_seconds':fit_seconds,
        'total_seconds':time.monotonic()-start,'heldout':heldout,
        'error':{'ei_rmse_nats':float(np.sqrt(np.mean(errors**2))),
                 'ei_max_abs_nats':float(np.max(abs(errors))),
                 'full_ei_error_nats':estimate[SOURCES]-truth[SOURCES]},
        'ei_nats':{','.join(map(str,s)):v for s,v in estimate.items()},
        'truth_ei_nats':{','.join(map(str,s)):v for s,v in truth.items()},
        'root_margin_nats':root_candidates[1]-root_candidates[0],
        'noise_minimum_eigenvalue':float(np.linalg.eigvalsh(fitted.noise).min()),
        'parameter_digest':hashlib.sha256(payload).hexdigest(),
        'training_prefix_digest':hashlib.sha256(x[:n].tobytes()+y[:n].tobytes()).hexdigest(),
        'test_digest':hashlib.sha256(tx.tobytes()+ty.tobytes()).hexdigest(),
        'label_records':label_records,
        'mean_methods':{m:{metric:float(np.mean([r['methods'][m][metric] for r in label_records]))
                          for metric in ('clade_f1','clade_precision','clade_recall','upper_support_recall','correct_root')}
                        for m in METHODS}}
    # Release unbounded method caches after each fit, rather than retain 18 models.
    fitted.information.cache_clear(); fitted.target_logdet.cache_clear()
    return result


def sample_dataset(case, seed, n, test_samples):
    model = ReplacementBroadcast(case)
    streams = np.random.SeedSequence([20261014,seed,ord(case)]).spawn(4)
    x = np.random.default_rng(streams[0]).normal(size=(n,9,model.coordinates_per_node))
    y = model.step(x,np.random.default_rng(streams[1]))
    tx = np.random.default_rng(streams[2]).normal(size=(test_samples,9,model.coordinates_per_node))
    ty = model.step(tx,np.random.default_rng(streams[3]))
    return x,y,tx,ty,model.true_log_conditional(tx,ty),model


def acceptance(records, sizes=SIZES):
    result = {}
    for case in 'BC':
        result[case] = {}
        for n in sizes:
            rows = [r for r in records if r['case']==case and r['n_train']==n]
            valid = len(rows)==3 and all(r['status']=='valid' for r in rows)
            means = {m:float(np.mean([r['mean_methods'][m]['clade_f1'] for r in rows])) for m in METHODS} if valid else {}
            best = max((v for m,v in means.items() if m!='spt'),default=None)
            root = all(r['mean_methods']['spt']['correct_root']==1 for r in rows) if valid else False
            result[case][str(n)] = {'complete':valid,'mean_f1':means,'strongest_baseline_f1':best,
                'spt_gap':means['spt']-best if valid else None,'spt_all_roots_correct':root,
                'passes':valid and means['spt']>=.95 and root and means['spt']-best>=.20}
    return {'by_case_size':result,'replace_main_figure':all(result[c].get(str(n),{}).get('passes',False)
                                                         for c in 'BC' for n in (16384,65536))}


def run(output, *, sizes=SIZES, seeds=SEEDS, budget_seconds=1200, test_samples=8192, prior_compute_seconds=0.):
    start = time.monotonic(); deadline = start+budget_seconds-prior_compute_seconds
    result = {'status':'running','records':[],'protocol':{
        'fixed_cases':{c:{'supports':SUPPORTS[c],'weights':ReplacementBroadcast(c).weights.tolist(),
                          'node_width':ReplacementBroadcast(c).coordinates_per_node} for c in 'BC'},
        'train_sizes':sizes,'seeds':seeds,'test_samples':test_samples,'permutations':label_schedule(),
        'source_distribution':'registered iid N(0,1) complete state; including unused past readouts',
        'horizon_steps':1,'methods':METHODS,'ridge':RIDGE,'noise_divisor':'n-p-1',
        'estimator_B':'dense conditional affine triangular TM',
        'estimator_C':'learned marginal monotone normal-score TM composed with dense conditional affine TM',
        'normal_score_knots':KNOTS,'knot_ranks':'round(linspace(0,n-1,257)); scores ndtri((rank+.5)/n)',
        'tails':'positive-slope linear tails on R; exact Jacobian in raw density',
        'fitting_access':'training paired arrays only; no true transform, weights, support or test samples',
        'EI':'exact marginals of fitted complete coherent joint; model-mediated EI in nats',
        'si_difference':'conditional fitted model + registered source prior; no separate empirical fits or Wishart correction',
        'replication':'three independent datasets; nested prefixes; 20 labels are within-fit rechecks, not independent replication',
        'syn_tolerance_nats':TOLERANCE_NATS,'clipping':False,'budget_seconds':budget_seconds,
        'prior_smoke_compute_seconds':prior_compute_seconds,'smoke_excluded_from_official':True,
        'acceptance':'N=16384 and 65536: mean SPT F1>=.95, all roots correct, strongest-baseline mean gap>=.20',
        'selection':'two predeclared constructive cases; no finite-data parameter or seed selection',
        'manuscript':{'parent':'P6UJCVG8','main':'DXGC7JEA','SI':'MWIWKSVG','checked':'2026-10-14',
                     'version':'sole main/SI pair, no explicit manuscript date/version',
                     'locations':'Methods Eqs10-12; SI S1.2 EqsS7-S15 and S5.1'}}}
    output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        result['elapsed_seconds'] = time.monotonic()-start
        result['cumulative_compute_seconds'] = result['elapsed_seconds']+prior_compute_seconds
        temporary = output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result,indent=2,allow_nan=False)); temporary.replace(output)
    save()
    for case in 'BC':
        # The dense-validation function checks the latent Gaussian generator.
        latent = ReplacementBroadcast(case)
        latent.case = 'B'  # disable only observation transform, retain C weights/support
        result.setdefault('generator_validation',{})[case] = validate(latent)
        for seed in seeds:
            dataset = sample_dataset(case,seed,max(sizes),test_samples)
            for n in sizes:
                if time.monotonic()>deadline:
                    result['status']='budget_exhausted'; save(); return result
                try:
                    record = analyze(case,seed,n,dataset,deadline=deadline)
                except TimeoutError as error:
                    result['records'].append({'case':case,'seed':seed,'n_train':n,'status':'budget_exhausted','error':str(error)})
                    result['status']='budget_exhausted'; save(); return result
                except (ArithmeticError,np.linalg.LinAlgError) as error:
                    record = {'case':case,'seed':seed,'n_train':n,'status':'invalid_estimate','error':str(error)}
                result['records'].append(record); save()
                print(json.dumps({'case':case,'seed':seed,'n_train':n,'status':record['status'],
                    'spt_f1':record.get('mean_methods',{}).get('spt',{}).get('clade_f1'),
                    'seconds':record.get('total_seconds')}),flush=True)
            del dataset
    result['status']='complete'; result['acceptance']=acceptance(result['records'],sizes); save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/spt_finite_replacement_bc/summary.json')
    parser.add_argument('--budget-seconds',type=float,default=1200)
    args = parser.parse_args()
    smoke = args.output.parent/'smoke.json'
    prior = json.loads(smoke.read_text())['elapsed_seconds'] if smoke.exists() else 0.
    result = run(args.output,budget_seconds=args.budget_seconds,prior_compute_seconds=prior)
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],
                      'acceptance':result.get('acceptance')},ensure_ascii=False))


if __name__=='__main__':
    main()
