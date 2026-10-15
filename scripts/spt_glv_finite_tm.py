"""HO-gLV parallel to the frozen local-Kuramoto finite-data precision pilot.

Shared seeds/data counts, sampling horizon, RQMC layouts and stopping gates.
Necessary family adaptations: real Gaussian conditional CDF TM, all-candidate
abundance monomial mean, absolute (not torus-rotated) bounded-source proposal.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr,ndtri,logsumexp
from scipy.stats import kstest,qmc,truncnorm
from sklearn.linear_model import Lasso

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.spt_local_hyperedge_glv import LocalHyperedgeGLV,abundance
from scripts.spt_kuramoto_finite_tm import (
    check_time as _check_monotonic_time,save_json,digest,ALPHAS,SYN_TOLERANCE,DEFENSIVE,
    normal_interval_logprob,pilot_masks,summarize_precision,
)

TWO_PI=2*np.pi
LOG_TWO_PI=float(np.log(TWO_PI))


@dataclass(frozen=True)
class WallClockDeadline:
    epoch: float


def check_time(deadline):
    if isinstance(deadline,WallClockDeadline):
        if time.time()>=deadline.epoch:
            raise TimeoutError('Persisted absolute wall-clock budget exhausted')
    else:
        _check_monotonic_time(deadline)


def dictionary():
    vectors=[]
    for count in range(1,10):
        for subset in itertools.combinations(range(9),count):
            v=np.zeros(9); v[list(subset)]=1/np.pi
            vectors.extend(power*v for power in (1,2,3))
    return np.asarray(vectors)


class RealConditionalTM:
    """Learned nonlinear conditional mean + normalized diagonal Gaussian TM."""
    def __init__(self,vectors,coefficients,intercept,variance):
        self.vectors=[np.asarray(v,dtype=float).reshape(-1,9) for v in vectors]
        self.coefficients=[np.asarray(c,dtype=float) for c in coefficients]
        self.intercept=np.asarray(intercept,dtype=float)
        self.variance=np.asarray(variance,dtype=float)
        if np.any(self.variance<=0) or not np.isfinite(self.variance).all():
            raise ArithmeticError('Invalid conditional residual variance; no noise floor added')
        self.precision=1/self.variance
        self._native_terms=None

    def mean(self,source,*,jacobian=False):
        from scripts.spt_glv_native import pack,mean
        if self._native_terms is None:
            self._native_terms=pack(self.vectors,self.coefficients)
        return mean(source,self.intercept,self._native_terms,jacobian=jacobian)

    def mean_selected(self,source,targets,variables=None):
        from scripts.spt_glv_native import pack,subset_mean
        if self._native_terms is None:
            self._native_terms=pack(self.vectors,self.coefficients)
        return subset_mean(source,self.intercept,self._native_terms,targets,variables)

    def mean_numpy(self,source,*,jacobian=False):
        x=np.asarray(source,dtype=float)
        value=x+self.intercept
        jac=np.broadcast_to(np.eye(9),(len(x),9,9)).copy() if jacobian else None
        for i,vectors in enumerate(self.vectors):
            if not len(vectors): continue
            features=np.exp(x@vectors.T)
            value[:,i]+=features@self.coefficients[i]
            if jacobian:
                jac[:,i,:]+=(features*self.coefficients[i])@vectors
        return (value,jac) if jacobian else value

    def log_prob(self,source,target,targets=None):
        columns=np.arange(9) if targets is None else np.asarray(targets,dtype=int)
        residual=np.asarray(target)[:,columns]-self.mean_selected(source,columns)
        return -.5*np.sum(residual**2/self.variance[columns]+np.log(self.variance[columns])+LOG_TWO_PI,axis=1)

    def sample(self,source,uniforms):
        if np.any((uniforms<=0)|(uniforms>=1)):
            raise ArithmeticError('Gaussian inverse CDF reached endpoint')
        return self.mean(source)+np.sqrt(self.variance)*ndtri(uniforms)

    def conditional_tm(self,source,target):
        return (target-self.mean(source))/np.sqrt(self.variance)

    def payload(self):
        return {'vectors':[v.tolist() for v in self.vectors],
                'coefficients':[c.tolist() for c in self.coefficients],
                'intercept':self.intercept.tolist(),'variance':self.variance.tolist()}

    @classmethod
    def load(cls,payload):
        return cls(**payload)

    @classmethod
    def fit(cls,source,target,vsource,vtarget,*,deadline):
        # Same data-only Lasso/validation-MSE protocol as Kuramoto. All 511
        # candidate subsets, powers 1/2/3; no mechanism supports are accepted.
        vectors=dictionary()
        raw=np.exp(source@vectors.T); vraw=np.exp(vsource@vectors.T)
        center=raw.mean(0); scale=np.std(raw,axis=0)*np.sqrt(2.)
        if np.any(scale<=0): raise ArithmeticError('Degenerate candidate feature')
        design=np.asfortranarray((raw-center)/scale)
        validation=(vraw-center)/scale
        gram=design.T@design
        displacement=target-source; vd=vtarget-vsource
        selected_vectors=[]; coefficients=[]; intercept=[]; variance=[]; records=[]
        for receiver in range(9):
            check_time(deadline)
            fit=Lasso(fit_intercept=True,max_iter=2500,tol=1e-7,warm_start=True,selection='cyclic',precompute=gram)
            candidates=[]
            for alpha in ALPHAS:
                check_time(deadline); fit.set_params(alpha=alpha)
                fit.fit(design,displacement[:,receiver])
                residual=vd[:,receiver]-fit.predict(validation)
                candidates.append((float(np.mean(residual**2)),alpha,fit.coef_.copy(),
                                   float(fit.intercept_),int(fit.n_iter_)))
            mse,alpha,beta,bias,iterations=min(candidates,key=lambda r:r[0])
            active=beta!=0; raw_beta=beta[active]/scale[active]
            raw_bias=float(bias-center[active]@raw_beta)
            residual=displacement[:,receiver]-(design@beta+bias)
            var=float(np.mean(residual**2))
            selected_vectors.append(vectors[active]); coefficients.append(raw_beta)
            intercept.append(raw_bias); variance.append(var)
            record={'node':receiver+1,'alpha':alpha,'active_features':int(active.sum()),
                'validation_rmse_coordinate':float(np.sqrt(mse)),'variance':var,'iterations':iterations,
                'candidates':[{'alpha':c[1],'validation_mse_coordinate2':c[0]} for c in candidates]}
            records.append(record); print(json.dumps({'stage':'fit',**record}),flush=True)
        return cls(selected_vectors,coefficients,intercept,variance),records


def transition_with_increments(model,source,increments,dt):
    value=np.asarray(source).copy()
    for noise in increments:
        k1=model.drift(value); k2=model.drift(value+.5*dt*k1)
        k3=model.drift(value+.5*dt*k2); k4=model.drift(value+dt*k3)
        value+=dt*(k1+2*k2+2*k3+k4)/6+model.noise*noise
    abundance(value)
    return value


def step_precision(case,seed=20261015,n=512):
    model=LocalHyperedgeGLV(case); rng=np.random.default_rng(seed)
    source=model.sample_source(n,rng)
    fine_noise=np.sqrt(.00125)*rng.normal(size=(160,n,9))
    states=[]
    for block in (8,4,2,1):
        increments=fine_noise.reshape(160//block,block,n,9).sum(1)
        states.append(transition_with_increments(model,source,increments,.00125*block))
    difference=[states[i]-states[i+1] for i in (0,1,2)]
    result={'case':case,'samples':n,'dt':[.01,.005,.0025,.00125],'paired_brownian':True,
        'source_digest':digest(source),'brownian_digest':digest(fine_noise),
        'rms_difference_coordinate':[float(np.sqrt(np.mean(d*d))) for d in difference],
        'max_difference_coordinate':[float(np.max(abs(d))) for d in difference]}
    result['passed']=result['rms_difference_coordinate'][-1]<1e-4
    return result


def test_diagnostics(model,source,target,rx,ry):
    residual=target-model.mean(source)
    standardized=residual/np.sqrt(model.variance)
    correlation=np.corrcoef(standardized,rowvar=False)-np.eye(9)
    repeat_mean=ry.mean(0); repeat_noise=ry-repeat_mean
    noise_variance=np.mean(repeat_noise**2,axis=(0,1))*len(ry)/(len(ry)-1)
    raw_mean_mse=np.mean((repeat_mean-model.mean(rx))**2,axis=0)
    pit=ndtr(standardized)
    return {'test_samples':len(source),'test_digest':digest(source,target),
        'conditional_nll_nats':float(-model.log_prob(source,target).mean()),
        'mean_residual_coordinate':residual.mean(0).tolist(),
        'standardized_second_moment':np.mean(standardized**2,axis=0).tolist(),
        'maximum_residual_correlation':float(np.max(abs(correlation))),
        'residual_correlation':correlation.tolist(),
        'pit_ks_statistic':[float(kstest(pit[:,i],'uniform').statistic) for i in range(9)],
        'repeat_count':len(ry),'repeat_initial_states':len(rx),
        'irreducible_noise_variance_coordinate2':noise_variance.tolist(),
        'conditional_mean_mse_raw_coordinate2':raw_mean_mse.tolist(),
        'conditional_mean_mse_noise_corrected_coordinate2':(raw_mean_mse-noise_variance/len(ry)).tolist(),
        'model_family_limitation':'nonlinear learned monomial mean with independent constant Gaussian residuals; finite-time SDE approximation'}


def proposal_sample(center,cholesky,uniforms,*,return_log_prob=False):
    """Sequential normal conditionals truncated to the absolute source cube."""
    n,p,d=uniforms.shape; z=np.zeros((n,p,d)); value=np.zeros_like(z)
    log_prob=np.zeros((n,p)) if return_log_prob else None
    for j in range(d):
        mean=center[:,j,None]+np.einsum('nq,npq->np',cholesky[:,j,:j],z[:,:,:j])
        std=cholesky[:,j,j,None]
        lo=(-np.pi-mean)/std; hi=(np.pi-mean)/std
        lower=ndtr(lo); width=ndtr(hi)-lower
        probability=lower+uniforms[:,:,j]*width
        z[:,:,j]=ndtri(probability); value[:,:,j]=mean+std*z[:,:,j]
        tail=(probability<=0)|(probability>=1)
        if np.any(tail):
            # Stable log-CDF inverse for remote sequential conditional tails;
            # same truncated Gaussian, no endpoint clipping or density change.
            lower_bound=np.broadcast_to(lo,probability.shape)
            upper_bound=np.broadcast_to(hi,probability.shape)
            z[:,:,j][tail]=truncnorm.ppf(uniforms[:,:,j][tail],lower_bound[tail],upper_bound[tail])
            value[:,:,j]=mean+std*z[:,:,j]
        if not np.isfinite(z[:,:,j]).all():
            raise ArithmeticError('Nonfinite real bounded proposal inverse CDF')
        if return_log_prob:
            # Reuse CDF width where subtraction is well conditioned. Preserve
            # the stable log-CDF difference for narrow/tail intervals.
            safe=width>1e-3
            lognorm=np.empty_like(width); lognorm[safe]=np.log(width[safe])
            lognorm[~safe]=normal_interval_logprob(lo[~safe],hi[~safe])
            log_prob+=-.5*(z[:,:,j]**2+LOG_TWO_PI)-np.log(std)-lognorm
    return (value,log_prob) if return_log_prob else value


def proposal_log_prob(values,center,cholesky):
    z=np.zeros_like(values); result=np.zeros(values.shape[:2])
    for j in range(values.shape[2]):
        mean=center[:,j,None]+np.einsum('nq,npq->np',cholesky[:,j,:j],z[:,:,:j])
        std=cholesky[:,j,j,None]
        z[:,:,j]=(values[:,:,j]-mean)/std
        lognorm=normal_interval_logprob((-np.pi-mean)/std,(np.pi-mean)/std)
        result+=-.5*(z[:,:,j]**2+LOG_TWO_PI)-np.log(std)-lognorm
    return np.where(np.all((values>=-np.pi)&(values<np.pi),axis=2),result,-np.inf)


def posterior_proposal(model,source,target,missing,targets,*,iterations=8,initial_offset=0.):
    value=source.copy()
    # Bounds apply to numerical proposal optimization, never EI/Syn or data.
    value[:,missing]=np.minimum(np.maximum(target[:,missing]+initial_offset,-np.pi+1e-10),np.pi-1e-10)
    for _ in range(iterations):
        mean,jac=model.mean(value,jacobian=True)
        derivative=jac[:,targets][:,:,missing]
        precision=np.einsum('nti,t,ntj->nij',derivative,model.precision[targets],derivative)
        gradient=np.einsum('nti,t,nt->ni',derivative,model.precision[targets],target[:,targets]-mean[:,targets])
        step=np.linalg.solve(precision,gradient)
        scale=np.maximum(1,np.max(abs(step),axis=1)/.5)
        value[:,missing]=np.minimum(np.maximum(value[:,missing]+step/scale[:,None],-np.pi+1e-10),np.pi-1e-10)
    mean,jac=model.mean(value,jacobian=True)
    derivative=jac[:,targets][:,:,missing]
    precision=np.einsum('nti,t,ntj->nij',derivative,model.precision[targets],derivative)
    covariance=np.linalg.inv(precision)
    chol=np.linalg.cholesky(1.2*covariance)
    gradient=np.einsum('nti,t,nt->ni',derivative,model.precision[targets],target[:,targets]-mean[:,targets])
    remaining=np.linalg.solve(precision,gradient)
    projected=np.minimum(np.maximum(value[:,missing]+remaining,-np.pi+1e-10),np.pi-1e-10)-value[:,missing]
    return value[:,missing],chol,{'max_remaining_step_coordinate':float(np.max(abs(remaining))),
        'max_projected_remaining_step_coordinate':float(np.max(abs(projected))),
        'max_proposal_std_coordinate':float(np.sqrt(np.diagonal(1.2*covariance,axis1=1,axis2=2)).max())}


class RealFullTargetIntegrator:
    def __init__(self,model,*,outer,max_particles,seed,deadline,source=None,target=None):
        # Frozen common point construction; domain-specific steps are below.
        check_time(deadline)
        if outer&(outer-1) or max_particles&(max_particles-1):
            raise ValueError('RQMC sample counts must be powers of two')
        self.model=model; self.outer=outer; self.max_particles=max_particles
        self.deadline=deadline; self.cache={}; self.proposals={}
        if (source is None)!=(target is None): raise ValueError('Supply both fixed source and target rows')
        if source is None:
            points=qmc.Sobol(18,scramble=True,seed=seed).random_base2(int(np.log2(outer)))
            self.source=(points[:,:9]-.5)*TWO_PI
            self.target=model.sample(self.source,points[:,9:])
        else:
            self.source=np.asarray(source,dtype=float); self.target=np.asarray(target,dtype=float)
            if self.source.shape!=(outer,9) or self.target.shape!=(outer,9):
                raise ValueError('Invalid fixed conditional-evaluation rows')
        inner=qmc.Sobol(10,scramble=True,seed=seed+1).random_base2(int(np.log2(outer*max_particles)))
        self.inner=inner.reshape(outer,max_particles,10); self.targets=np.arange(9)

    def information(self,masks,particles):
        denominator,diagnostic=self.marginal(0,particles)
        estimates={}; pointwise={}; queries={}
        for mask in masks:
            numerator,query=self.marginal(mask,particles)
            pointwise[mask]=numerator-denominator
            estimates[mask]=float(np.mean(pointwise[mask])); queries[str(mask)]=query
        return estimates,pointwise,{'denominator':diagnostic,'queries':queries}

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
            check_time(self.deadline); stop=min(self.outer,start+max(1,32768//particles))
            uniforms=self.inner[start:stop,:particles]
            proposed,logproposal=proposal_sample(center[start:stop],chol[start:stop],uniforms[:,:,1:len(missing)+1],return_log_prob=True)
            prior=uniforms[:,:,1:len(missing)+1]*TWO_PI-np.pi
            from_prior=uniforms[:,:,0]<DEFENSIVE
            proposed=np.where(from_prior[:,:,None],prior,proposed)
            rows,cols=np.nonzero(from_prior)
            if len(rows):
                logproposal[rows,cols]=proposal_log_prob(prior[rows,cols][:,None,:],
                    center[start:stop][rows],chol[start:stop][rows])[:,0]
            mixture=np.logaddexp(np.log(1-DEFENSIVE)+logproposal,np.log(DEFENSIVE)-len(missing)*LOG_TWO_PI)
            completed=np.repeat(self.source[start:stop,None,:],particles,axis=1)
            completed[:,:,missing]=proposed
            targets=np.repeat(self.target[start:stop],particles,axis=0)
            likelihood=self.model.log_prob(completed.reshape(-1,9),targets).reshape(stop-start,particles)
            weights=likelihood-len(missing)*LOG_TWO_PI-mixture
            normalizer=logsumexp(weights,axis=1); normalized=weights-normalizer[:,None]
            values.append(normalizer-np.log(particles))
            effective.extend(np.exp(-logsumexp(2*normalized,axis=1)))
            maximum.extend(np.exp(np.max(normalized,axis=1)))
        result=np.concatenate(values),{**diagnostic,'ess_minimum':float(np.min(effective)),
            'ess_median':float(np.median(effective)),'maximum_normalized_weight':float(np.max(maximum))}
        self.cache[key]=result
        return result


def run(args):
    started=time.monotonic()-args.prior_compute_seconds; deadline=started+args.budget_seconds; output=Path(args.output)
    protocol={'created':'2026-10-15','cases':args.cases,'seed':args.seed,
        'train_samples':args.train,'validation_samples':args.validation,'test_samples':args.test,
        'source':'iid Uniform(-pi,pi)^9 in u=pi*log(x)','target':'complete future nine log-abundance coordinates',
        'tau':.2,'dt':.0025,'coupling':1.,'decay':.5,'noise':.08,'self_limit':1.,
        'units':'nats','syn_tolerance_nats':SYN_TOLERANCE,'defensive_prior_fraction':DEFENSIVE,
        'precision_target_nats':.01,'dictionary':'all 511 nonempty node subsets, abundance powers1/2/3; 1533 data-standardized features',
        'alphas':ALPHAS,'model':'nonlinear conditional mean + independent constant Gaussian CDF TM',
        'lasso_precompute':'shared exact training Gram, same objective/alpha/tolerance/iteration cap',
        'proposal':'inverse/Gauss-Newton sequential Gaussian conditionals truncated to absolute source cube + uniform',
        'precision_gate':'inner256x64->256x256; outer1024x256->4096x256; highest SEM<.01; ESS>=10; maxweight<.25; projected Newton<1e-3; inverse starts gap<1e-4; nonpositive Jacobian count0',
        'budget_seconds':args.budget_seconds,'clipping':False,
        'reference_protocol':'docs/reports/spt_local_kuramoto_validation.md',
        'reference_code_sha256':hashlib.sha256((ROOT/'scripts/spt_kuramoto_finite_tm.py').read_bytes()).hexdigest(),
        'generator_code_sha256':hashlib.sha256((ROOT/'scripts/spt_local_hyperedge_glv.py').read_bytes()).hexdigest(),
        'manuscript':{'parent':'P6UJCVG8','main':'DXGC7JEA','si':'MWIWKSVG',
            'version':'sole attachments; no explicit manuscript date/version','checked':'2026-10-15',
            'locations':'Methods Eqs4-12; SI S1.2, S5.1, S7'}}
    protocol=json.loads(json.dumps(protocol))
    payload={'protocol':protocol,'records':[],'status':'running'}
    if output.exists():
        old=json.loads(output.read_text())
        # The parallel owner may add unrelated optional estimators to its file.
        # Preserve the original reference hash; all scientific settings and our
        # generator hash must still match. Our integrator point construction
        # and domain code are independent, with analytic-reference tests.
        comparable=lambda p:{k:v for k,v in p.items() if k!='reference_code_sha256'}
        if comparable(old['protocol'])!=comparable(protocol):
            raise ValueError('Cache scientific protocol differs; use a new output path')
        payload=old
        payload.setdefault('reference_versions_observed',[])
        if protocol['reference_code_sha256'] not in payload['reference_versions_observed']:
            payload['reference_versions_observed'].append(protocol['reference_code_sha256'])
        started=time.monotonic()-max(payload.get('elapsed_compute_seconds',0),args.prior_compute_seconds)
        deadline=started+args.budget_seconds
    def save():
        payload['elapsed_compute_seconds']=time.monotonic()-started; save_json(output,payload)
    save()
    for case in args.cases:
        check_time(deadline)
        existing=next((r for r in payload['records'] if r['case']==case),None)
        if existing is not None and existing.get('status') in ('precision_failed','complete'): continue
        record=existing or {'case':case,'status':'preflight'}
        if existing is None: payload['records'].append(record)
        try:
            if 'step_precision' not in record:
                record['step_precision']=step_precision(case); save()
                print(json.dumps({'stage':'step_precision',**record['step_precision']}),flush=True)
            if not record['step_precision']['passed']:
                record['status']='step_precision_failed'; save(); continue
            generator=LocalHyperedgeGLV(case)
            if 'model' not in record:
                streams=np.random.SeedSequence([20261015,args.seed]).spawn(8)
                source=generator.sample_source(args.train+args.validation,np.random.default_rng(streams[0]))
                target=generator.step(source,np.random.default_rng(streams[1]),dt=.0025)
                tx=generator.sample_source(args.test,np.random.default_rng(streams[2]))
                ty=generator.step(tx,np.random.default_rng(streams[3]),dt=.0025)
                rx=generator.sample_source(256,np.random.default_rng(streams[4]))
                ry=np.array([generator.step(rx,np.random.default_rng(np.random.SeedSequence([20261015,args.seed,500+i])),dt=.0025)
                             for i in range(16)])
                fitted,history=RealConditionalTM.fit(source[:args.train],target[:args.train],source[args.train:],target[args.train:],deadline=deadline)
                record.update(model=fitted.payload(),fit=history,generator=generator.metadata(),
                    source_digest=digest(source),training_digest=digest(source[:args.train],target[:args.train]),
                    validation_digest=digest(source[args.train:],target[args.train:]),
                    heldout=test_diagnostics(fitted,tx,ty,rx,ry))
                record['status']='fitted'; save()
            fitted=RealConditionalTM.load(record['model']); masks=pilot_masks(case)
            if 'inverse_check' not in record:
                check_source=np.random.default_rng(20261016).uniform(-np.pi,np.pi,(128,9))
                check_target=fitted.sample(check_source,np.random.default_rng(20261017).uniform(size=(128,9)))
                a=posterior_proposal(fitted,check_source,check_target,np.arange(9),np.arange(9),iterations=16)
                b=posterior_proposal(fitted,check_source,check_target,np.arange(9),np.arange(9),iterations=16,initial_offset=.15)
                _,jac=fitted.mean(check_source,jacobian=True)
                record['inverse_check']={'second_initialization_offset_coordinate':.15,
                    'maximum_center_difference_coordinate':float(np.max(abs(a[0]-b[0]))),
                    'nonpositive_jacobian_count':int(np.count_nonzero(np.linalg.det(jac)<=0)),
                    'minimum_jacobian_determinant':float(np.linalg.det(jac).min()),'first':a[2],'second':b[2]}
                save()
            if (record['inverse_check']['maximum_center_difference_coordinate']>=1e-4
                    or record['inverse_check']['nonpositive_jacobian_count']):
                record.update(status='inverse_proposal_failed',precision_passed=False,
                    error='Local posterior proposal cannot certify mode coverage; no tree is built')
                save(); continue
            record.setdefault('precision',[])
            for outer,particles in ((256,64),(256,256),(1024,256),(4096,256)):
                for scramble in range(args.scrambles):
                    if any((r['outer'],r['particles'],r['scramble'])==(outer,particles,scramble) for r in record['precision']): continue
                    integrator=RealFullTargetIntegrator(fitted,outer=outer,max_particles=256,seed=20261015+1000*scramble,deadline=deadline)
                    estimates,pointwise,diagnostic=integrator.information(masks,particles)
                    record['precision'].append({'outer':outer,'particles':particles,'scramble':scramble,
                        'ei_nats':{str(k):v for k,v in estimates.items()},'diagnostics':diagnostic})
                    record['status']='precision_running'; save()
                    print(json.dumps({'stage':'precision','case':case,'outer':outer,'particles':particles,
                        'scramble':scramble,'full_ei_nats':estimates[511],'denominator_ess':diagnostic['denominator']['ess_median']}),flush=True)
            summary=summarize_precision(record['precision'],masks,case)
            changes=[max(abs(a['mean_ei_nats'][str(m)]-b['mean_ei_nats'][str(m)]) for m in masks) for a,b in zip(summary,summary[1:])]
            diagnostics=[v for row in record['precision'] if row['outer']==4096
                         for v in [row['diagnostics']['denominator'],*row['diagnostics']['queries'].values()]
                         if v['ess_minimum'] is not None]
            record['proposal_audit']={'minimum_ess':min(v['ess_minimum'] for v in diagnostics),
                'maximum_normalized_weight':max(v['maximum_normalized_weight'] for v in diagnostics),
                'maximum_projected_newton_step_coordinate':max(v['max_projected_remaining_step_coordinate'] for v in diagnostics)}
            passed=(changes[0]<.01 and changes[-1]<.01 and summary[-1]['maximum_ei_sem_nats'] is not None
                    and summary[-1]['maximum_ei_sem_nats']<.01
                    and record['proposal_audit']['minimum_ess']>=10
                    and record['proposal_audit']['maximum_normalized_weight']<.25
                    and record['proposal_audit']['maximum_projected_newton_step_coordinate']<1e-3
                    and record['inverse_check']['maximum_center_difference_coordinate']<1e-4
                    and record['inverse_check']['nonpositive_jacobian_count']==0)
            record.update(precision_summary=summary,precision_changes_nats=changes,maximum_precision_change_nats=max(changes),precision_passed=passed,
                          status='precision_passed' if passed else 'precision_failed'); save()
        except TimeoutError as error:
            record['status']='budget_exhausted'; record['error']=str(error); payload['status']='budget_exhausted'; save(); return payload
        except (ArithmeticError,np.linalg.LinAlgError,ValueError) as error:
            record['status']='numerical_failure'; record['error']=str(error); save()
    payload['status']='pilot_complete'; save(); return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/spt_local_glv_tm/pilot.json')
    parser.add_argument('--train',type=int,default=16384); parser.add_argument('--validation',type=int,default=4096)
    parser.add_argument('--test',type=int,default=4096); parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--cases',nargs='+',default=list('ABC')); parser.add_argument('--scrambles',type=int,default=4)
    parser.add_argument('--budget-seconds',type=float,default=3600)
    parser.add_argument('--prior-compute-seconds',type=float,default=0.,help='Include interrupted attempt time in the same cumulative budget')
    args=parser.parse_args(); result=run(args)
    print(json.dumps({'status':result['status'],'seconds':result['elapsed_compute_seconds']}))


if __name__=='__main__': main()
