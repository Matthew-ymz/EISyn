"""All-BOLD appendix sensitivity: fitted-model intervention Xi, natural baselines.

The input is one fixed-length 100-ROI BOLD-like sequence. A shared first-order
delta-Ridge model (alpha=1) predicts the same full 100-ROI state at 300 ms. Xi
uses a factorized unit-variance prior (Gaussian moments of independent uniform
[-sqrt(3),sqrt(3)] standardized inputs); baselines retain the empirical correlated
prior. Both joint densities use the repository affine triangular TM estimator.
These are counterfactuals of a BOLD-fitted surrogate, not physical DMF/BOLD do's.
"""
from __future__ import annotations
import numpy as np
from exp.TM.transport_map_density import AffineTransportMapDensityEstimator
from scripts.dmf_joint_readout import ld,audit_nonnegative

TOL_NATS=1e-8


def information(covariance):
    cov=np.asarray(covariance,float);d=len(cov)//2
    prior,cross,future=cov[:d,:d],cov[:d,d:],cov[d:,d:]
    conditional=prior-cross@np.linalg.solve(future,cross.T)
    conditional=.5*(conditional+conditional.T)
    whole=.5*(ld(prior)-ld(conditional))
    singles=.5*(np.log(np.diag(prior))-np.log(np.diag(conditional)))
    return whole,singles


def hemisphere_phi_r(covariance):
    cov=np.asarray(covariance,float);d=len(cov)//2
    scale=np.sqrt(np.diag(cov));corr=cov/scale[:,None]/scale[None,:]
    def mi(past,future):
        future=future+d;both=np.r_[past,future]
        return .5*(ld(corr[np.ix_(past,past)])+ld(corr[np.ix_(future,future)])-ld(corr[np.ix_(both,both)]))
    left,right=np.arange(d//2),np.arange(d//2,d)
    blocks=np.array([[mi(a,b) for b in (left,right)] for a in (left,right)])
    whole=mi(np.arange(d),np.arange(d))
    phi=whole-np.trace(blocks)+blocks.min()
    return float(phi),blocks


def bold_metrics(bold,lag_steps=300,alpha=1.,ridge=1e-6):
    b=np.asarray(bold,float)
    if b.ndim!=2 or b.shape[1]%2 or b.shape[0]<=lag_steps+20 or not np.isfinite(b).all():
        raise ValueError('Expected finite even-dimensional BOLD sequence and enough lagged samples')
    x,y=b[:-lag_steps],b[lag_steps:];cut=int(.75*len(x))
    if cut<=b.shape[1] or len(x)-cut<=b.shape[1]:
        raise ValueError('Insufficient fit/holdout samples for full-ROI model')
    mean=x[:cut].mean(0);scale=x[:cut].std(0,ddof=1)
    if np.any(scale<=1e-12): raise ValueError('Near-constant BOLD source channel')
    xs,ys=(x-mean)/scale,(y-mean)/scale
    train,truth=xs[:cut],ys[:cut];delta=truth-train;intercept=delta.mean(0)
    coefficient=np.linalg.solve(train.T@train+alpha*np.eye(b.shape[1]),train.T@(delta-intercept))
    transition=np.eye(b.shape[1])+coefficient
    residual=truth-train@transition-intercept
    noise=np.cov(residual,rowvar=False)+ridge*np.eye(b.shape[1])
    natural_prior=np.cov(train,rowvar=False)+ridge*np.eye(b.shape[1])
    intervention_prior=np.diag(np.diag(natural_prior))
    def joint(prior):
        cross=prior@transition
        future=transition.T@cross+noise
        cov=np.block([[prior,cross],[cross.T,future]])
        return AffineTransportMapDensityEstimator(np.zeros(len(cov)),cov,cut).covariance
    natural=joint(natural_prior);intervention=joint(intervention_prior)
    ei,partial=information(intervention);xi=ei-partial.sum()
    natural_whole,natural_single=information(natural)
    wms=natural_whole-natural_single.sum()
    phi,blocks=hemisphere_phi_r(natural)
    audit=audit_nonnegative([xi,ei,*partial,phi,natural_whole,*natural_single,*blocks.ravel()],TOL_NATS)
    # Purge a horizon-sized gap: no training target occurs in held-out sources.
    holdout_start=cut+lag_steps
    if len(x)-holdout_start<=b.shape[1]:
        raise ValueError('Insufficient held-out data after the forecast gap')
    holdout=ys[holdout_start:]-xs[holdout_start:]@transition-intercept
    rmse=np.sqrt(np.mean(holdout**2,axis=0))
    persistence=np.sqrt(np.mean((ys[holdout_start:]-xs[holdout_start:])**2,axis=0))
    return dict(xi_nats=float(xi),phi_r_nats=phi,wms_nats=float(wms),audit=audit,
                whole_ei_nats=float(ei),partial_ei_sum_nats=float(partial.sum()),
                natural_whole_mi_nats=float(natural_whole),natural_single_mi_sum_nats=float(natural_single.sum()),
                fit_samples=cut,holdout_samples=len(x)-holdout_start,holdout_gap_steps=lag_steps,lagged_samples=len(x),
                mean_holdout_RMSE_standardized=float(rmse.mean()),
                mean_persistence_RMSE_standardized=float(persistence.mean()),
                natural_covariance=natural,intervention_covariance=intervention)
