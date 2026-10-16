"""Data-only circular conditional TM and precision-gated Kuramoto experiment.

The known simulator supplies observations only. Each fitted mean family uses
the same full candidate dictionary for every case; no planted supports enter density fitting,
importance proposals, or tree construction. Information concerns future phases,
although a source-dependent translation is used inside conditional densities.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import i0e, i1e, log_ndtr, logsumexp, ndtr, ndtri
from scipy.optimize import brentq
from scipy.stats import qmc, vonmises, kstest
from sklearn.linear_model import Lasso

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_local_hyperedge_kuramoto import LocalHyperedgeKuramoto, SOURCES, wrap
from scripts.spt import all_nonempty_subsets, nontrivial_bipartitions
from scripts.spt_continuous_hierarchy_pilot import audit_table

TWO_PI = 2*np.pi
LOG_TWO_PI = float(np.log(TWO_PI))
SYN_TOLERANCE = .01
DEFENSIVE = .05
WIDE_FRACTION = .15
WIDE_SCALE = 2.
ALPHAS = (2e-3, 5e-4, 1e-4)
DT = .0025


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Approved cumulative compute budget reached')


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def digest(*arrays):
    return hashlib.sha256(b''.join(np.asarray(a).tobytes() for a in arrays)).hexdigest()


def dictionary(receiver, harmonics=(1, 2, 3)):
    """Every other-node subset; no architecture is accepted by this function."""
    other = [j for j in range(9) if j != receiver]
    vectors = []
    for count in range(1, 9):
        for subset in itertools.combinations(other, count):
            v = np.zeros(9)
            v[list(subset)] = 1
            v[receiver] = -count
            vectors.extend(h*v for h in harmonics)
    return np.array(vectors)


def vm_log_density(residual, kappa):
    # Stable even when kappa is thousands; log(i0)=kappa+log(i0e).
    return kappa*(np.cos(residual)-1)-np.log(i0e(kappa))-LOG_TWO_PI


def fit_concentration(residual):
    resultant = float(np.mean(np.cos(residual)))
    if not 0 < resultant < 1:
        raise ArithmeticError(f'Invalid residual cosine resultant: {resultant}')
    upper = 1/(1-resultant)
    return float(brentq(lambda k: i1e(k)/i0e(k)-resultant, 1e-8, upper))


class PeriodicConditionalTM:
    """Exactly normalized product-von-Mises conditional, fitted from arrays.

    For each fixed x, the circular conditional CDF followed by Phi^-1 is a
    diagonal (therefore triangular) transport. A conditional mean translation
    does not change the EI target to a phase increment.
    """
    def __init__(self, vectors, sine, cosine, intercept, kappa):
        self.vectors = [np.asarray(a, dtype=float) for a in vectors]
        self.sine = [np.asarray(a, dtype=float) for a in sine]
        self.cosine = [np.asarray(a, dtype=float) for a in cosine]
        self.intercept = np.asarray(intercept, dtype=float)
        self.kappa = np.asarray(kappa, dtype=float)
        if np.any(self.kappa <= 0) or not np.isfinite(self.kappa).all():
            raise ValueError('Concentrations must be finite and positive')

    def mean(self, source, *, jacobian=False):
        x = np.asarray(source, dtype=float)
        value = x+self.intercept
        jac = np.broadcast_to(np.eye(9), (len(x), 9, 9)).copy() if jacobian else None
        for i, vectors in enumerate(self.vectors):
            if not len(vectors):
                continue
            phase = x@vectors.T
            sine, cosine = np.sin(phase), np.cos(phase)
            value[:, i] += sine@self.sine[i]+cosine@self.cosine[i]
            if jacobian:
                jac[:, i, :] += (cosine*self.sine[i]-sine*self.cosine[i])@vectors
        return (value, jac) if jacobian else value

    def log_prob(self, source, target, targets=None):
        indices = np.arange(9) if targets is None else np.asarray(targets, dtype=int)
        mu = self.mean(source)
        return vm_log_density(np.asarray(target)[:, indices]-mu[:, indices], self.kappa[indices]).sum(1)

    def sample(self, source, uniforms):
        # Tabulated inverse CDF avoids thousands of scalar root solves. Grid
        # resolution is in residual standard-deviation units and independent
        # of the experimental architecture. Validate against the same CDF at
        # every call; this is numerical inversion, not a Gaussian noise swap.
        if not hasattr(self,'_inverse_cdf'):
            self._inverse_cdf=[]
            for kappa in self.kappa:
                extent=min(np.pi,12/np.sqrt(kappa))
                angles=np.r_[-np.pi,np.linspace(-extent,extent,65537),np.pi]
                cdf=vonmises.cdf(angles,kappa)
                unique,index=np.unique(cdf,return_index=True)
                self._inverse_cdf.append((unique,angles[index]))
        residual=np.column_stack([np.interp(uniforms[:,j],*self._inverse_cdf[j]) for j in range(9)])
        error=float(np.max(abs(vonmises.cdf(residual,self.kappa)-uniforms)))
        if error>1e-8:
            raise ArithmeticError(f'Circular inverse-CDF accuracy failed: {error}')
        return wrap(self.mean(source)+residual)

    def conditional_tm(self, source, target):
        # CDF with a fixed target chart [-pi,pi); integrates exactly to 1.
        mu = self.mean(source)
        u = (vonmises.cdf(target, self.kappa, loc=mu)
             - vonmises.cdf(-np.pi, self.kappa, loc=mu))
        if np.any((u <= 0) | (u >= 1)):
            raise ArithmeticError('Conditional CDF reached a numerical endpoint')
        return ndtri(u)

    def payload(self):
        return {'vectors': [v.tolist() for v in self.vectors],
                'sine': [v.tolist() for v in self.sine],
                'cosine': [v.tolist() for v in self.cosine],
                'intercept': self.intercept.tolist(), 'kappa': self.kappa.tolist()}

    @classmethod
    def load(cls, payload):
        if payload.get('family')=='natural_periodic_pair_mlp_v1':
            from scripts.spt_kuramoto_pair_mlp import PairMLPConditionalTM
            return PairMLPConditionalTM.load(payload)
        if payload.get('family')=='natural_periodic_mlp_v1':
            from scripts.spt_kuramoto_natural_mlp import NaturalMLPConditionalTM
            return NaturalMLPConditionalTM.load(payload)
        if payload.get('family')=='learned_flow':
            from scripts.spt_kuramoto_flow_tm import PeriodicFlowConditionalTM
            return PeriodicFlowConditionalTM.load(payload)
        return cls(**payload)

    @classmethod
    def fit(cls, source, target, validation_source, validation_target, *, deadline):
        """Arrays only; validation MSE selects regularization per output."""
        displacement = wrap(target-source)
        validation_displacement = wrap(validation_target-validation_source)
        vectors, sine, cosine, intercept, kappa, records = [], [], [], [], [], []
        for receiver in range(9):
            check_time(deadline)
            v = dictionary(receiver)
            phase = source@v.T
            design = np.asfortranarray(np.c_[np.sin(phase), np.cos(phase)])
            vp = validation_source@v.T
            validation = np.c_[np.sin(vp), np.cos(vp)]
            fit = Lasso(fit_intercept=True, max_iter=2500, tol=1e-7,
                        warm_start=True, selection='cyclic')
            candidates = []
            for alpha in ALPHAS:
                check_time(deadline)
                fit.set_params(alpha=alpha)
                fit.fit(design, displacement[:, receiver])
                residual = validation_displacement[:, receiver]-fit.predict(validation)
                candidates.append((float(np.mean(residual**2)), alpha,
                                   fit.coef_.copy(), float(fit.intercept_), int(fit.n_iter_)))
            mse, alpha, beta, bias, iterations = min(candidates, key=lambda r:r[0])
            nonzero = (beta[:len(v)] != 0) | (beta[len(v):] != 0)
            chosen_v = v[nonzero]
            chosen_s = beta[:len(v)][nonzero]
            chosen_c = beta[len(v):][nonzero]
            # Concentration uses training residual, not the final test set.
            train_residual = displacement[:, receiver]-(design@beta+bias)
            concentration = fit_concentration(train_residual)
            vectors.append(chosen_v); sine.append(chosen_s); cosine.append(chosen_c)
            intercept.append(bias); kappa.append(concentration)
            record = {'node':receiver+1, 'alpha':alpha, 'active_frequencies':int(nonzero.sum()),
                      'validation_rmse_rad':float(np.sqrt(mse)), 'kappa':concentration,
                      'iterations':iterations,
                      'candidates':[{'alpha':c[1], 'validation_mse_rad2':c[0]} for c in candidates]}
            records.append(record)
            print(json.dumps({'stage':'fit', **record}), flush=True)
        return cls(vectors,sine,cosine,intercept,kappa), records


def transition_with_increments(model, source, increments, dt):
    value = np.asarray(source).copy()
    for noise in increments:
        k1=model.drift(value); k2=model.drift(value+.5*dt*k1)
        k3=model.drift(value+.5*dt*k2); k4=model.drift(value+dt*k3)
        value=wrap(value+dt*(k1+2*k2+2*k3+k4)/6+model.noise*noise)
    return value


def step_precision(case, seed=20261015, n=512):
    model=LocalHyperedgeKuramoto(case)
    rng=np.random.default_rng(seed)
    source=model.sample_source(n,rng)
    fine_noise=np.sqrt(.00125)*rng.normal(size=(160,n,9))
    states=[]
    for block in (8,4,2,1):
        increments=fine_noise.reshape(160//block,block,n,9).sum(1)
        states.append(transition_with_increments(model,source,increments,.00125*block))
    differences=[wrap(states[i]-states[i+1]) for i in (0,1,2)]
    result={'case':case,'samples':n,'dt':(.01,.005,.0025,.00125),
            'paired_brownian':True,
            'rms_difference_rad':[float(np.sqrt(np.mean(d*d))) for d in differences],
            'max_difference_rad':[float(np.max(abs(d))) for d in differences],
            'selected_dt':DT}
    result['passed']=result['rms_difference_rad'][-1]<1e-4
    return result


def test_diagnostics(model, source, target, repeated_source, repeated_targets):
    residual=wrap(target-model.mean(source))
    standardized=residual*np.sqrt(model.kappa)
    circular_mean=wrap(np.angle(np.mean(np.exp(1j*residual),axis=0)))
    pit=vonmises.cdf(residual,model.kappa)
    correlation=np.corrcoef(standardized,rowvar=False)-np.eye(9)
    # Repeated transitions are independent Brownian paths at the same initial x.
    repeated_mean=np.angle(np.mean(np.exp(1j*repeated_targets),axis=0))
    repeat_noise=wrap(repeated_targets-repeated_mean)
    mean_error=wrap(repeated_mean-model.mean(repeated_source))
    n_repeat=len(repeated_targets)
    noise_variance=np.mean(repeat_noise**2,axis=(0,1))*n_repeat/(n_repeat-1)
    # Remove the finite-repetition variance in each estimated conditional mean.
    raw_mean_error_variance=np.mean(mean_error**2,axis=0)
    corrected=raw_mean_error_variance-noise_variance/n_repeat
    bins=[]
    predicted_increment=wrap(model.mean(source)-source)
    for j in range(9):
        cuts=np.quantile(predicted_increment[:,j],np.linspace(0,1,6))
        groups=np.searchsorted(cuts[1:-1],predicted_increment[:,j],side='right')
        bins.append([float(np.mean(residual[groups==g,j]**2)) for g in range(5)])
    return {'test_samples':len(source),'test_digest':digest(source,target),
            'conditional_nll_nats':float(-model.log_prob(source,target).mean()),
            'circular_mean_residual_rad':circular_mean.tolist(),
            'standardized_second_moment':np.mean(standardized**2,axis=0).tolist(),
            'maximum_residual_correlation':float(np.max(abs(correlation))),
            'residual_correlation':correlation.tolist(),
            'pit_ks_statistic':[float(kstest(pit[:,i],'uniform').statistic) for i in range(9)],
            'conditional_variance_bins_rad2':bins,
            'repeat_count':n_repeat,'repeat_initial_states':len(repeated_source),
            'irreducible_noise_variance_rad2':noise_variance.tolist(),
            'conditional_mean_mse_raw_rad2':raw_mean_error_variance.tolist(),
            'conditional_mean_mse_noise_corrected_rad2':corrected.tolist(),
            'model_family_limitation':'independent constant-concentration conditional outputs; test diagnostics do not establish exact SDE density'}


def normal_interval_logprob(a,b):
    """Stable log(Phi(b)-Phi(a)); a<b, including remote tails."""
    a,b=np.broadcast_arrays(a,b)
    result=np.empty_like(a,dtype=float)
    negative=b<=0; positive=a>=0; middle=~(negative|positive)
    for mask,left,right in ((negative,a,b),(positive,-b,-a)):
        if mask.any():
            hi=log_ndtr(right[mask]); lo=log_ndtr(left[mask])
            result[mask]=hi+np.log(-np.expm1(lo-hi))
    result[middle]=np.log(ndtr(b[middle])-ndtr(a[middle]))
    return result


def proposal_sample(center,cholesky,uniforms):
    """Sequentially truncated correlated normal on [-pi,pi)^d, then rotate.

    Each conditional is truncated to the coordinate's fixed support. This is
    exactly normalized, including near a torus chart boundary; it is not a
    nearest-image approximation to a wrapped multivariate normal.
    """
    n,p,d=uniforms.shape
    z=np.zeros((n,p,d)); displacement=np.zeros_like(z)
    for j in range(d):
        mean=np.einsum('nq,npq->np',cholesky[:,j,:j],z[:,:,:j])
        std=cholesky[:,j,j,None]
        lo=(-np.pi-mean)/std; hi=(np.pi-mean)/std
        probability=ndtr(lo)+uniforms[:,:,j]*(ndtr(hi)-ndtr(lo))
        if np.any((probability<=0)|(probability>=1)):
            raise ArithmeticError('Proposal inverse CDF reached numerical endpoint')
        z[:,:,j]=ndtri(probability)
        displacement[:,:,j]=mean+std*z[:,:,j]
    return wrap(center[:,None,:]+displacement)


def proposal_log_prob(values,center,cholesky):
    displacement=wrap(values-center[:,None,:])
    z=np.zeros_like(displacement); result=np.zeros(displacement.shape[:2])
    for j in range(values.shape[2]):
        mean=np.einsum('nq,npq->np',cholesky[:,j,:j],z[:,:,:j])
        std=cholesky[:,j,j,None]
        z[:,:,j]=(displacement[:,:,j]-mean)/std
        lognorm=normal_interval_logprob((-np.pi-mean)/std,(np.pi-mean)/std)
        result+=-.5*(z[:,:,j]**2+LOG_TWO_PI)-np.log(std)-lognorm
    return result


def posterior_proposal(model, source, target, missing, targets, *, iterations=8, initial_offset=0.):
    """Local Gauss-Newton using y and kept x only, never missing source truth."""
    value=source.copy()
    value[:,missing]=wrap(target[:,missing]+initial_offset)
    for _ in range(iterations):
        mean,jac=model.mean(value,jacobian=True)
        residual=wrap(target[:,targets]-mean[:,targets])
        derivative=jac[:,targets][:,:,missing]
        precision=np.einsum('nti,t,ntj->nij',derivative,model.kappa[targets],derivative)
        gradient=np.einsum('nti,t,nt->ni',derivative,model.kappa[targets],np.sin(residual))
        try:
            step=np.linalg.solve(precision,gradient)
        except np.linalg.LinAlgError as error:
            raise ArithmeticError('Singular posterior proposal; no arbitrary ridge added') from error
        # Bound a numerical optimization step, not a metric or source support.
        scale=np.maximum(1,np.max(abs(step),axis=1)/.5)
        value[:,missing]=wrap(value[:,missing]+step/scale[:,None])
    mean,jac=model.mean(value,jacobian=True)
    derivative=jac[:,targets][:,:,missing]
    precision=np.einsum('nti,t,ntj->nij',derivative,model.kappa[targets],derivative)
    covariance=np.linalg.inv(precision)
    cholesky=np.linalg.cholesky(1.2*covariance)
    gradient=np.einsum('nti,t,nt->ni',derivative,model.kappa[targets],np.sin(wrap(target[:,targets]-mean[:,targets])))
    # Small Newton step at the final center is a scale-independent diagnostic.
    remaining=np.linalg.solve(precision,gradient)
    return value[:,missing],cholesky,{'max_remaining_step_rad':float(np.max(abs(remaining))),
        'max_proposal_std_rad':float(np.sqrt(np.diagonal(1.2*covariance,axis1=1,axis2=2)).max())}


class FullTargetIntegrator:
    """RQMC outer expectation, exact importance weights for one fitted joint."""
    def __init__(self,model,*,outer,max_particles,seed,deadline):
        if outer&(outer-1) or max_particles&(max_particles-1):
            raise ValueError('RQMC sample counts must be powers of two')
        self.model=model; self.outer=outer; self.max_particles=max_particles
        self.deadline=deadline; self.cache={}; self.proposals={}
        points=qmc.Sobol(18,scramble=True,seed=seed).random_base2(int(np.log2(outer)))
        self.source=(points[:,:9]-.5)*TWO_PI
        self.target=model.sample(self.source,points[:,9:])
        # One P-point net, different scramble per outer point would be expensive.
        # One N*P-point scrambled net in (mixture,9 Gaussian) dimensions, then
        # reshape; prefixes are nested within every outer observation.
        inner=qmc.Sobol(10,scramble=True,seed=seed+1).random_base2(int(np.log2(outer*max_particles)))
        self.inner=inner.reshape(outer,max_particles,10)
        self.targets=np.arange(9)

    def marginal(self,mask,particles):
        key=(mask,particles)
        if key in self.cache: return self.cache[key]
        check_time(self.deadline)
        if mask==511:
            result=self.model.log_prob(self.source,self.target),{'ess_minimum':None,'ess_median':None}
            self.cache[key]=result; return result
        missing=np.array([i for i in range(9) if not mask>>i&1],dtype=int)
        if mask not in self.proposals:
            self.proposals[mask]=posterior_proposal(self.model,self.source,self.target,missing,self.targets)
        center,chol,diagnostic=self.proposals[mask]
        values=[]; effective=[]; maximum=[]
        for start in range(0,self.outer,max(1,32768//particles)):
            check_time(self.deadline)
            stop=min(self.outer,start+max(1,32768//particles))
            uniforms=self.inner[start:stop,:particles]
            proposed=proposal_sample(center[start:stop],chol[start:stop],uniforms[:,:,1:len(missing)+1])
            wide=proposal_sample(center[start:stop],WIDE_SCALE*chol[start:stop],uniforms[:,:,1:len(missing)+1])
            prior=uniforms[:,:,1:len(missing)+1]*TWO_PI-np.pi
            use_prior=uniforms[:,:,0]<DEFENSIVE
            use_wide=(uniforms[:,:,0]>=DEFENSIVE)&(uniforms[:,:,0]<DEFENSIVE+WIDE_FRACTION)
            proposed=np.where(use_wide[:,:,None],wide,proposed)
            proposed=np.where(use_prior[:,:,None],prior,proposed)
            logproposal=proposal_log_prob(proposed,center[start:stop],chol[start:stop])
            logwide=proposal_log_prob(proposed,center[start:stop],WIDE_SCALE*chol[start:stop])
            logmixture=np.logaddexp(np.logaddexp(np.log(1-DEFENSIVE-WIDE_FRACTION)+logproposal,
                                               np.log(WIDE_FRACTION)+logwide),
                                   np.log(DEFENSIVE)-len(missing)*LOG_TWO_PI)
            completed=np.repeat(self.source[start:stop,None,:],particles,axis=1)
            completed[:,:,missing]=proposed
            target=np.repeat(self.target[start:stop],particles,axis=0)
            likelihood=self.model.log_prob(completed.reshape(-1,9),target).reshape(stop-start,particles)
            weights=likelihood-len(missing)*LOG_TWO_PI-logmixture
            normalizer=logsumexp(weights,axis=1)
            normalized=weights-normalizer[:,None]
            values.append(normalizer-np.log(particles))
            effective.extend(np.exp(-logsumexp(2*normalized,axis=1)))
            maximum.extend(np.exp(np.max(normalized,axis=1)))
        result=np.concatenate(values),{**diagnostic,'ess_minimum':float(np.min(effective)),
            'ess_median':float(np.median(effective)),'maximum_normalized_weight':float(np.max(maximum))}
        self.cache[key]=result
        return result

    def information(self,masks,particles):
        denominator,diagnostic=self.marginal(0,particles)
        estimates={}; pointwise={}; queries={}
        for mask in masks:
            numerator,query=self.marginal(mask,particles)
            pointwise[mask]=numerator-denominator
            estimates[mask]=float(np.mean(pointwise[mask]))
            queries[str(mask)]=query
        return estimates,pointwise,{'denominator':diagnostic,'queries':queries}


def masks_to_table(values):
    return {tuple(i+1 for i in range(9) if mask>>i&1):float(value) for mask,value in values.items() if mask}


def pilot_masks(case):
    # Structural locations are a predeclared precision audit, never fitter input.
    left=(1,2,3,4,5) if case in 'AB' else (1,2,3,4,5,6)
    right=tuple(i for i in SOURCES if i not in left)
    candidates=[SOURCES,left,right,*[(i,) for i in SOURCES],(1,2),(1,2,3),(6,7),(7,8,9)]
    return sorted({sum(1<<(i-1) for i in s) for s in candidates})


def summarize_precision(replicates,masks,case):
    groups={}
    for row in replicates:
        groups.setdefault((row['outer'],row['particles']),[]).append(row)
    results=[]
    for (outer,particles),rows in sorted(groups.items()):
        data=np.array([[r['ei_nats'][str(m)] for m in masks] for r in rows])
        mean=data.mean(0)
        sem=data.std(0,ddof=1)/np.sqrt(len(rows)) if len(rows)>1 else np.full(len(masks),np.nan)
        results.append({'outer':outer,'particles':particles,'replicates':len(rows),
            'mean_ei_nats':dict(zip(map(str,masks),mean.tolist())),
            'ei_sem_nats':dict(zip(map(str,masks),sem.tolist())) if len(rows)>1 else None,
            'maximum_ei_sem_nats':float(sem.max()) if len(rows)>1 else None})
    return results


def critical_syn_audit(replicates):
    """Direct paired residual audit on available queries; never a full audit."""
    if not replicates: return {'split_count':0,'summary':[],'passed':False}
    available=set(map(int,replicates[0]['ei_nats']))
    splits=[]
    for parent in sorted(available):
        nodes=tuple(i+1 for i in range(9) if parent>>i&1)
        if len(nodes)<2: continue
        for left,right in nontrivial_bipartitions(nodes):
            l=sum(1<<(i-1) for i in left); r=sum(1<<(i-1) for i in right)
            if l in available and r in available: splits.append((parent,l,r))
    groups={}
    for row in replicates:
        groups.setdefault((row['outer'],row['particles']),[]).append(row)
    summaries=[]
    for (outer,particles),rows in sorted(groups.items()):
        if len(rows)!=4: continue
        residuals=np.array([[z['ei_nats'][str(a)]-z['ei_nats'][str(l)]-z['ei_nats'][str(r)]
                             for a,l,r in splits] for z in rows])
        mean=residuals.mean(0); sem=residuals.std(0,ddof=1)/np.sqrt(len(rows))
        summaries.append({'outer':outer,'particles':particles,'replicates':len(rows),
            'mean_syn_nats':{f'{a}:{l}:{r}':float(v) for (a,l,r),v in zip(splits,mean)},
            'syn_sem_nats':{f'{a}:{l}:{r}':float(v) for (a,l,r),v in zip(splits,sem)},
            'maximum_syn_sem_nats':float(sem.max()),'minimum_raw_syn_nats':float(residuals.min()),
            'tolerance_negative_count':int(np.count_nonzero((residuals<0)&(residuals>=-SYN_TOLERANCE))),
            'significant_violation_count':int(np.count_nonzero(residuals < -SYN_TOLERANCE))})
    by={(s['outer'],s['particles']):s for s in summaries}
    result={'split_count':len(splits),'summary':summaries,
            'scope':'only available critical-query bipartitions, not all 9330'}
    if all(k in by for k in ((256,64),(256,256),(1024,256),(4096,256))):
        def difference(a,b):
            return max(abs(by[a]['mean_syn_nats'][k]-by[b]['mean_syn_nats'][k]) for k in by[a]['mean_syn_nats'])
        inner=difference((256,64),(256,256)); outer=difference((1024,256),(4096,256))
        high=by[4096,256]
        result.update(inner_change_nats=inner,outer_change_nats=outer,
            passed=(inner<.01 and outer<.01 and high['maximum_syn_sem_nats']<.01
                    and high['significant_violation_count']==0))
    else: result['passed']=False
    return result


def run(args):
    started=time.monotonic(); deadline=started+args.budget_seconds
    output=Path(args.output)
    payload={'protocol':{'created':'2026-10-15','cases':args.cases,'seed':args.seed,
        'train_samples':args.train,'validation_samples':args.validation,'test_samples':args.test,
        'source':'iid Uniform(-pi,pi)^9','target':'complete future nine phases','tau':.2,'dt':DT,
        'coupling':1.,'decay':.5,'noise':.08,'units':'nats','syn_tolerance_nats':SYN_TOLERANCE,
        'defensive_prior_fraction':DEFENSIVE,'precision_target_nats':.01,
        'dictionary':('all 502 non-singleton node subsets, shared attractive basic-sine vector-field coefficients'
                      if args.mean_family=='flow' else
                      'all 255 nonempty other-node subsets per receiver, harmonics 1/2/3, sine/cosine'),
        'alphas':list(ALPHAS),'mean_family':args.mean_family,
        'model':'periodic finite-time mean + independent constant-concentration von Mises conditional TM',
        'proposal':'80% local + 15% two-scale sequential truncated correlated normal on torus + 5% uniform',
        'budget_seconds':args.budget_seconds,'clipping':False,
        'manuscript':{'parent':'P6UJCVG8','main':'DXGC7JEA','si':'MWIWKSVG',
            'version':'sole attachment group, no explicit manuscript date/version',
            'locations':'Methods (4)-(12), SI S1.2, S5.1, S7 (S67)-(S71)'}},'records':[], 'status':'running'}
    if output.exists():
        old=json.loads(output.read_text())
        if old['protocol']!=payload['protocol']:
            raise ValueError('Existing cache protocol differs; choose a new output path')
        payload=old
        started-=payload.get('elapsed_compute_seconds',0)
        deadline=started+args.budget_seconds
    elif args.fit_cache:
        fit_cache=json.loads(Path(args.fit_cache).read_text())
        for key in ('cases','seed','train_samples','validation_samples','test_samples','source','target',
                    'tau','dt','coupling','decay','noise','mean_family'):
            if fit_cache['protocol'][key]!=payload['protocol'][key]:
                raise ValueError(f'Fitted cache mismatch: {key}')
        fields=('case','model','fit','training_digest','validation_digest','heldout','step_precision','inverse_check')
        payload['records']=[{**{k:r[k] for k in fields if k in r},'status':'fitted'} for r in fit_cache['records']]
        payload['fit_cache_provenance']={'path':str(args.fit_cache),
            'sha256':hashlib.sha256(Path(args.fit_cache).read_bytes()).hexdigest(),
            'note':'only fitted model/data diagnostics reused; all importance estimates recomputed under refined proposal'}
    def save():
        payload['elapsed_compute_seconds']=time.monotonic()-started
        save_json(output,payload)
    save()
    for case in args.cases:
        check_time(deadline)
        existing=next((r for r in payload['records'] if r['case']==case),None)
        if existing is not None and existing.get('status') in ('precision_failed','complete'):
            continue
        record=existing or {'case':case,'status':'preflight'}
        if existing is None: payload['records'].append(record)
        try:
            if 'step_precision' not in record:
                record['step_precision']=step_precision(case)
                save(); print(json.dumps({'stage':'step_precision',**record['step_precision']}),flush=True)
            if not record['step_precision']['passed']:
                record['status']='step_precision_failed'; save(); continue
            if 'model' not in record:
                model=LocalHyperedgeKuramoto(case)
                streams=np.random.SeedSequence([20261015,args.seed]).spawn(8)
                total=args.train+args.validation
                source=model.sample_source(total,np.random.default_rng(streams[0]))
                target=model.step(source,np.random.default_rng(streams[1]),dt=DT)
                tx=model.sample_source(args.test,np.random.default_rng(streams[2]))
                ty=model.step(tx,np.random.default_rng(streams[3]),dt=DT)
                rx=model.sample_source(256,np.random.default_rng(streams[4]))
                ry=np.array([model.step(rx,np.random.default_rng(np.random.SeedSequence([20261015,args.seed,500+i])),dt=DT)
                             for i in range(16)])
                fitter=PeriodicConditionalTM
                if args.mean_family=='flow':
                    from scripts.spt_kuramoto_flow_tm import PeriodicFlowConditionalTM
                    fitter=PeriodicFlowConditionalTM
                fitted,history=fitter.fit(source[:args.train],target[:args.train],
                    source[args.train:],target[args.train:],deadline=deadline)
                record.update(model=fitted.payload(),fit=history,
                    training_digest=digest(source[:args.train],target[:args.train]),
                    validation_digest=digest(source[args.train:],target[args.train:]),
                    heldout=test_diagnostics(fitted,tx,ty,rx,ry))
                record['status']='fitted'; save()
            fitted=PeriodicConditionalTM.load(record['model'])
            if 'inverse_check' not in record:
                check_source=np.random.default_rng(20261016).uniform(-np.pi,np.pi,(128,9))
                check_target=fitted.sample(check_source,np.random.default_rng(20261017).uniform(size=(128,9)))
                a=posterior_proposal(fitted,check_source,check_target,np.arange(9),np.arange(9),iterations=16)
                b=posterior_proposal(fitted,check_source,check_target,np.arange(9),np.arange(9),iterations=16,initial_offset=.15)
                _,jac=fitted.mean(check_source,jacobian=True)
                record['inverse_check']={'second_initialization_offset_rad':.15,
                    'max_center_difference_rad':float(np.max(abs(wrap(a[0]-b[0])))),
                    'jacobian_minimum_singular_value':float(np.linalg.svd(jac,compute_uv=False).min()),
                    'jacobian_nonpositive_determinant_count':int(np.count_nonzero(np.linalg.det(jac)<=0)),
                    'first':a[2],'second':b[2]}
                save()
            if (record['inverse_check']['max_center_difference_rad']>=1e-4
                    or record['inverse_check']['jacobian_nonpositive_determinant_count']):
                record['status']='inverse_proposal_failed'
                record['precision_passed']=False
                record['error']='Local posterior proposal cannot certify mode coverage; no tree is built'
                save(); continue
            if args.fit_only:
                record['status']='fit_complete'; save(); continue
            masks=pilot_masks(case)
            # A modest first precision gate; no ranking-based tuning.
            layouts=[(256,64),(256,256),(1024,256),(4096,256)]
            record.setdefault('precision',[])
            for outer,particles in layouts:
                for scramble in range(args.scrambles):
                    if any((r['outer'],r['particles'],r['scramble'])==(outer,particles,scramble) for r in record['precision']):
                        continue
                    integrator=FullTargetIntegrator(fitted,outer=outer,max_particles=256,
                        seed=20261015+1000*scramble,deadline=deadline)
                    estimates,pointwise,diagnostic=integrator.information(masks,particles)
                    record['precision'].append({'outer':outer,'particles':particles,'scramble':scramble,
                        'ei_nats':{str(k):v for k,v in estimates.items()},'diagnostics':diagnostic})
                    record['status']='precision_running'; save()
                    print(json.dumps({'stage':'precision','case':case,'outer':outer,'particles':particles,
                        'scramble':scramble,'full_ei_nats':estimates[511],
                        'denominator_ess':diagnostic['denominator']['ess_median']}),flush=True)
            summary=summarize_precision(record['precision'],masks,case)
            record['precision_summary']=summary
            differences=[]
            for a,b in zip(summary,summary[1:]):
                differences.append(max(abs(a['mean_ei_nats'][str(m)]-b['mean_ei_nats'][str(m)]) for m in masks))
            record['precision_changes_nats']=differences
            # Initial coarse levels are diagnostics; acceptance is based on
            # the last particle comparison and highest two outer resolutions.
            record['maximum_precision_change_nats']=max(differences)
            diagnostics=[v for r in record['precision'] if r['outer']==4096
                         for v in [r['diagnostics']['denominator'],*r['diagnostics']['queries'].values()]
                         if v['ess_minimum'] is not None]
            record['proposal_audit']={'minimum_ess':min(v['ess_minimum'] for v in diagnostics),
                'maximum_normalized_weight':max(v['maximum_normalized_weight'] for v in diagnostics),
                'maximum_remaining_newton_step_rad':max(v['max_remaining_step_rad'] for v in diagnostics)}
            passed=(differences[0]<.01 and differences[-1]<.01
                    and summary[-1]['maximum_ei_sem_nats'] is not None
                    and summary[-1]['maximum_ei_sem_nats']<.01
                    and record['proposal_audit']['minimum_ess']>=10
                    and record['proposal_audit']['maximum_normalized_weight']<.25
                    and record['proposal_audit']['maximum_remaining_newton_step_rad']<1e-3
                    and record['inverse_check']['max_center_difference_rad']<1e-4
                    and record['inverse_check']['jacobian_nonpositive_determinant_count']==0)
            record['precision_passed']=passed
            record['critical_syn_audit']=critical_syn_audit(record['precision'])
            record['precision_passed']=passed and record['critical_syn_audit']['passed']
            record['status']='precision_passed' if record['precision_passed'] else 'precision_failed'
            save()
        except TimeoutError as error:
            record['status']='budget_exhausted'; record['error']=str(error)
            payload['status']='budget_exhausted'; save(); return payload
        except (ArithmeticError,np.linalg.LinAlgError,ValueError) as error:
            record['status']='numerical_failure'; record['error']=str(error); save()
    payload['status']='pilot_complete'; save()
    return payload


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'results/spt_local_kuramoto_tm/flow_mixture_dt0025.json'))
    parser.add_argument('--cases',nargs='+',default=list('ABC'))
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--train',type=int,default=16384)
    parser.add_argument('--validation',type=int,default=4096)
    parser.add_argument('--test',type=int,default=4096)
    parser.add_argument('--scrambles',type=int,default=4)
    parser.add_argument('--budget-seconds',type=float,default=2250)
    parser.add_argument('--mean-family',choices=('fourier','flow'),default='flow')
    parser.add_argument('--fit-only',action='store_true')
    parser.add_argument('--fit-cache')
    return parser.parse_args()


if __name__=='__main__':
    run(parse_args())
