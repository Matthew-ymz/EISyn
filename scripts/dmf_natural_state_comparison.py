"""Natural-distribution full-E/I PhiR; preserve observational source WMS.

This module prepares/benchmarks conditions only. It does not start a cohort sweep.
Xi retains its existing interventional cache; neither baseline factorizes the
natural prior. PhiR uses the fixed complete E|I vector bipartition, not ROI-pair
averaging or MIP. The original source-WMS query regularization is preserved and
its small difference from coherent joint-density MI is explicitly measured.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import qr
from scripts.dmf_joint_readout import audit_nonnegative, ld
from scripts.dmf_response_benchmark import simulate_sources
from scripts.run_dmf_subject_curve_baselines import native_wms
from scripts.validate_dmf_83_region_oracle_phi_eid import standardize, load_dmf_module

ROOT = Path(__file__).resolve().parents[1]
DENSE = ROOT / 'results/dmf_schaefer100/subject_curves_93_dense'
TOL_NATS = 1e-8


def fixed_EI_phi_r(covariance):
    cov = np.asarray(covariance, float)
    if (cov.ndim != 2 or cov.shape[0] != cov.shape[1] or len(cov)%4
            or not np.isfinite(cov).all()
            or not np.allclose(cov, cov.T, rtol=0, atol=1e-12)):
        raise ValueError('Expected finite symmetric past E/I, future E/I covariance')
    ld(cov)  # fail invalid covariance; no repair, floor or projection
    scale = np.sqrt(np.diag(cov))
    corr = cov / scale[:,None] / scale[None,:]
    d = len(cov)//2

    def mi(past, future):
        target = future+d
        both = np.r_[past,target]
        return .5*(ld(corr[np.ix_(past,past)])+ld(corr[np.ix_(target,target)])
                    -ld(corr[np.ix_(both,both)]))

    e, i = np.arange(d//2), np.arange(d//2,d)
    cross = np.array([[mi(x,y) for y in (e,i)] for x in (e,i)])
    whole = mi(np.arange(d),np.arange(d))
    phi = whole-np.trace(cross)+cross.min()
    audit = audit_nonnegative([whole,phi,*cross.ravel()], TOL_NATS)
    return dict(phi_r_nats=float(phi), whole_mi_nats=float(whole),
                partition_mi_nats=cross, nonnegative_audit=audit)


def affine_fixed_EI_phi_r(source, coefficient, noise):
    """Query the same correlated affine Gaussian using square-root factors.

    X = L z and Y = B.T L z + R epsilon, where LL.T=C and RR.T=N.
    QR evaluates covariance determinants without forming a badly scaled joint
    matrix or subtracting explained covariance. Reordering the source Cholesky
    gives the exact conditional source factor for each fixed E/I block. No
    eigenvalue floor, extra ridge, prior factorization or projection is used.
    ``noise`` includes both inherited native target/noise ridge contributions.
    """
    source, coefficient, noise = map(lambda a: np.asarray(a, float),
                                     (source, coefficient, noise))
    d = len(source)
    if (d % 2 or any(a.shape != (d,d) for a in (source,coefficient,noise))
            or not all(np.isfinite(a).all() for a in (source,coefficient,noise))
            or not all(np.allclose(a,a.T,rtol=0,atol=1e-12) for a in (source,noise))):
        raise ValueError('Expected finite full E/I affine Gaussian factors')
    source_root = np.linalg.cholesky(source)
    noise_root = np.linalg.cholesky(noise)

    def log_gram(root):
        scale = np.linalg.norm(root,axis=1)
        if not np.isfinite(scale).all() or (scale<=0).any():
            raise ArithmeticError('Invalid Gaussian square-root scale')
        upper = qr((root/scale[:,None]).T,mode='economic',check_finite=True)[1]
        diagonal = np.abs(np.diag(upper))
        if (diagonal<=0).any():
            raise np.linalg.LinAlgError('Rank-deficient Gaussian square root')
        return float(2*(np.log(scale).sum()+np.log(diagonal).sum()))

    indices = np.arange(d)
    blocks = (indices[:d//2],indices[d//2:])
    target_root = coefficient.T@source_root
    whole = .5*(log_gram(np.c_[target_root,noise_root])-log_gram(noise_root))
    target_logdet = [log_gram(np.c_[target_root[q],noise_root[q]]) for q in blocks]
    cross = np.empty((2,2))
    for a,p in enumerate(blocks):
        order = np.r_[p,np.setdiff1d(indices,p)]
        conditional_root = np.linalg.cholesky(source[np.ix_(order,order)])[:,len(p):]
        remaining_signal = coefficient[order].T@conditional_root
        for b,q in enumerate(blocks):
            cross[a,b] = .5*(target_logdet[b]
                            -log_gram(np.c_[remaining_signal[q],noise_root[q]]))
    phi = whole-np.trace(cross)+cross.min()
    audit = audit_nonnegative([whole,phi,*cross.ravel()], TOL_NATS)
    return dict(phi_r_nats=float(phi),whole_mi_nats=float(whole),
                partition_mi_nats=cross,nonnegative_audit=audit,
                numerical_method='affine-square-root-QR')


def natural_metrics(x,y,ridge=1e-6):
    """Keep the empirical correlated prior and the native least-squares model."""
    x,y = standardize(x)[0],standardize(y)[0]
    if x.shape != y.shape or x.shape[1]%2:
        raise ValueError('Expected same full E/I source/target coordinates')
    coefficient = np.linalg.lstsq(x,y,rcond=None)[0]
    source = np.cov(x,rowvar=False)+ridge*np.eye(x.shape[1])
    residual = y-x@coefficient
    noise = np.cov(residual,rowvar=False)+ridge*np.eye(y.shape[1])
    target = coefficient.T@source@coefficient+noise
    target = .5*(target+target.T)+ridge*np.eye(y.shape[1])
    cross = source@coefficient
    cov = np.block([[source,cross],[cross.T,target]])
    # The second ridge already present in target is part of conditional noise.
    effective_noise = noise+ridge*np.eye(y.shape[1])
    phi = affine_fixed_EI_phi_r(source,coefficient,effective_noise)
    native, audit = native_wms(x,y)
    # Signed WMS is NOT PEID Syn: do not apply the nonnegative audit to it.
    wms = native['raw_phi']*np.log(2)
    return dict(**phi, source_wms_nats=float(wms),
                native_whole_mi_nats=float(native['joint_ei']*np.log(2)),
                native_singleton_sum_nats=float(native['singleton_ei_sum']*np.log(2)),
                native_wms_audit=audit, covariance=cov,
                source_covariance=source,transition_coefficient=coefficient,
                effective_noise_covariance=effective_noise,
                native_whole_vs_joint_mi_error_nats=float(abs(native['joint_ei']*np.log(2)-phi['whole_mi_nats'])),
                source_correlation_max=float(abs(np.corrcoef(x,rowvar=False)-np.eye(x.shape[1])).max()))


def condition(subject,g,seed,scheme='original_sampling'):
    """One bounded condition for review/cost estimation; never launches a sweep.

    original_sampling: exact existing natural-tail draws + independent 300-ms
    futures, with a source-hash and WMS bridge to the existing condition.
    trajectory_pairs: one continuous natural trajectory, 300-ms-separated pairs;
    faster, but intentionally a different sampling protocol and effective N.
    """
    c = json.loads((DENSE/'contract.json').read_text())
    with np.load(DENSE/'inputs.npz') as a:
        idx = a['subject_ids'].tolist().index(subject)
        sc,jf = a['connectivity'][idx].copy(),a['j_fic'].copy()
    native = c['native_protocol']
    dmf = load_dmf_module()
    p = dmf.DMFParameters(**native['native_parameters'])
    stabilization = dmf.StabilizationParameters(**native['stabilization'])
    started = time.perf_counter()
    sim = dmf.simulate_dmf(sc,g,jf,parameters=p,stabilization_parameters=stabilization,
                          seed=62000+seed,record_state_trace=True,record_rate_trace=True)
    states = np.column_stack((sim['state_se_trace'],sim['state_si_trace']))
    if not np.isfinite(states).all() or states.min()<0 or states.max()>1:
        raise ArithmeticError('Invalid natural states')
    rates = np.asarray(sim['region_rate_trace_hz'])
    if not np.isfinite(rates).all() or rates.min()<0 or rates.max()>500:
        raise ArithmeticError('Invalid natural rates')
    start = int(sim['stabilization_start_step'])
    tail = states[start:]
    natural_seconds = time.perf_counter()-started
    if scheme=='original_sampling':
        indices = np.random.default_rng(63000+seed).integers(0,len(tail),size=2048)
        x = tail[indices].copy()
        y,diagnostic = simulate_sources(dmf,x,sc,jf,g,p,64000+seed)
        if diagnostic['outside_state_count'] or diagnostic['abnormal_rate_count']:
            raise ArithmeticError(f'Invalid future states: {diagnostic}')
    elif scheme=='trajectory_pairs':
        if len(tail)<=300:
            raise ArithmeticError('Natural tail too short for 300-ms pairs')
        x,y = tail[:-300].copy(),tail[300:].copy()
        diagnostic = dict(scheme='continuous trajectory; inherited natural-state clipping')
    else:
        raise ValueError(scheme)
    future_seconds = time.perf_counter()-started-natural_seconds
    result = natural_metrics(x,y)
    source_sha = hashlib.sha256(x.tobytes()).hexdigest()
    if scheme=='original_sampling':
        with np.load(DENSE/'conditions'/f'{subject}_G{g:.2f}_seed{seed}.npz') as a:
            if str(a['native_source_sha256'])!=source_sha:
                raise ValueError('Reconstructed natural source differs from original cache')
            ref = json.loads(str(a['native_metrics_json']))
        bridge = abs(result['source_wms_nats']-ref['wms_bits']*np.log(2))
        if bridge>TOL_NATS:
            raise ArithmeticError(f'Natural WMS bridge failed: {bridge} nats')
        result['native_wms_bridge_error_nats'] = float(bridge)
    result.update(subject=subject,G=g,seed=seed,scheme=scheme,source_sha256=source_sha,
                  sample_count=len(x),unique_source_count=len(np.unique(x,axis=0)),
                  natural_seconds=natural_seconds,future_seconds=future_seconds,
                  elapsed_seconds=time.perf_counter()-started,diagnostics=diagnostic)
    return result
