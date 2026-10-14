"""Equivalent batched solves and OLS-moment recovery for continuous DMF curves.

This changes numerical linear algebra only. The estimator, fixed penalties,
ROI order, source prior and residual ridge are identical to the paired pilot.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve, solve_triangular

from exp.TM.transport_map_density import AffineTransportMapDensityEstimator
from scripts.dmf_continuous_affine import ContinuousAffine, Moments, fit_continuous_affine
from scripts.dmf_response_benchmark import dmf_source_components


def recover_moments(covariance, x, *, ridge=1e-6):
    """Recover standardized paired-sample moments from an unrestricted OLS cache.

    C_xy=C_xx B and C_yy=R_unregularized+B^T C_xx B by OLS orthogonality.
    Native target mean/scale are unavailable: these recovered moments support
    information readouts only, and must not be used for native-unit prediction.
    """
    x=np.asarray(x,float); cov=np.asarray(covariance,float); d=x.shape[1]
    if cov.shape!=(2*d,2*d) or len(x)<=d or not np.isfinite(cov).all():
        raise ValueError('Full unrestricted OLS density and original source samples required')
    prior=cov[:d,:d]
    if not np.allclose(prior,np.diag(np.diag(prior)),atol=1e-12,rtol=0):
        raise ValueError('Known diagonal source prior required')
    xs=x.std(0,ddof=1); xm=x.mean(0); xz=(x-xm)/xs
    xx=xz.T@xz/(len(x)-1)
    b=cov[:d,d:]/np.diag(prior)[:,None]
    residual=cov[d:,d:]-b.T@prior@b-ridge*np.eye(d)
    xy=xx@b; yy=residual+b.T@xy; yy=.5*(yy+yy.T)
    if not np.allclose(np.diag(yy),1,atol=1e-10,rtol=0):
        raise ValueError('Cache is not the unrestricted standardized paired OLS fit')
    return Moments(xx,xy,yy,prior.copy(),xm,xs,
                   np.full(d,np.nan),np.full(d,np.nan),len(x))


def fit_continuous_batched(m, sc, g, mean_penalty=.1, noise_penalty=10., ridge=1e-6):
    """Woodbury mean solves and batched modified-Cholesky residual solves.

    For tiny G use the reference solver to avoid subtractive cancellation in
    the Woodbury denominators. Both branches solve the same fixed objective.
    """
    if g < .05:
        return fit_continuous_affine(m,sc,g,mean_penalty,noise_penalty,ridge)
    if not np.isfinite([g,mean_penalty,noise_penalty,ridge]).all() or min(mean_penalty,noise_penalty,ridge)<=0:
        raise ValueError('Finite positive penalties/ridge required')
    d=len(m.xx); n=d//2; blocks=dmf_source_components(sc,1.)
    b=np.zeros((d,d))
    for block in blocks:
        indices=np.asarray(sorted(block),int); k=len(indices)
        xx=m.xx[np.ix_(indices,indices)]; xy=m.xy[np.ix_(indices,indices)]
        alpha=mean_penalty/(g*g)
        inverse=solve(xx+alpha*np.eye(k),np.eye(k),assume_a='pos')
        base=inverse@xy
        for roi in sorted(i for i in indices if i<n):
            local=np.array([np.flatnonzero(indices==roi)[0],np.flatnonzero(indices==roi+n)[0]])
            correction=solve(np.eye(2)-alpha*inverse[np.ix_(local,local)],base[np.ix_(local,local)],assume_a='pos')
            fitted=base[:,local]+alpha*inverse[:,local]@correction
            b[np.ix_(indices,indices[local])]=fitted
    residual=m.yy-b.T@m.xy-m.xy.T@b+b.T@m.xx@b
    residual=.5*(residual+residual.T)
    noise=np.zeros((d,d)); factor_native=np.zeros((d,d)); minimum=np.inf
    for block in blocks:
        rois=sorted(i for i in block if i<n)
        order=np.array([j for i in rois for j in [i,i+n]],int); k=len(order)
        q=residual[np.ix_(order,order)]; sd=np.sqrt(np.diag(q))
        if not np.isfinite(sd).all() or np.any(sd<=0):
            raise ArithmeticError('Invalid residual variance')
        c=q/sd[:,None]/sd[None,:]; alpha=noise_penalty/(g*g)
        chol=np.linalg.cholesky(c+alpha*np.eye(k))
        inverse_chol=solve_triangular(chol,np.eye(k),lower=True)
        regression=-inverse_chol/np.diag(inverse_chol)[:,None]
        np.fill_diagonal(regression,0.)
        for j in range(1,k,2):
            # Previous E coordinate is the sole within-ROI predecessor of I.
            column=inverse_chol[j-1,:j]*inverse_chol[j-1,j-1]
            denominator=1-alpha*column[-1]
            if denominator<=0:
                raise ArithmeticError('Nonpositive Woodbury denominator')
            regression[j,:j]+=alpha*column*regression[j,j-1]/denominator
        innovation=1-2*np.sum(regression*c,axis=1)+np.sum((regression@c)*regression,axis=1)
        innovation*=sd*sd
        if not np.isfinite(innovation).all() or np.any(innovation<=0):
            raise ArithmeticError('Nonpositive innovation variance')
        minimum=min(minimum,float(innovation.min()))
        regression*=sd[:,None]/sd[None,:]
        factor=solve_triangular(np.eye(k)-regression,np.diag(np.sqrt(innovation)),lower=True)
        factor_native[np.ix_(order,order)]=factor
        noise[np.ix_(order,order)]=factor@factor.T+ridge*np.eye(k)
    cross=m.prior@b
    joint=np.block([[m.prior,cross],[cross.T,b.T@cross+noise]])
    covariance=AffineTransportMapDensityEstimator(np.zeros(2*d),joint,m.count).covariance
    mask=np.arange(d)[:,None]%n==np.arange(d)[None,:]%n
    off_b=b.copy(); off_b[mask]=0
    off_l=factor_native.copy(); off_l[mask]=0
    diagnostics=dict(fixed_sc_components=len(blocks),mean_cross_norm=float(np.linalg.norm(off_b)),
        noise_cross_factor_norm=float(np.linalg.norm(off_l)),
        mean_unscaled_cross_norm=float(np.linalg.norm(off_b)/g),
        noise_unscaled_cross_norm=float(np.linalg.norm(off_l)/g),minimum_innovation_variance=minimum,
        factor_covariance_error=float(np.abs(factor_native@factor_native.T+ridge*np.eye(d)-noise).max()))
    return ContinuousAffine(covariance,b,noise,factor_native,m,g,mean_penalty,noise_penalty,diagnostics)
