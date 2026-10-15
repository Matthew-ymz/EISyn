"""Positive HO-gLV counterpart to the nine-node local Kuramoto benchmark.

Shared intervention/observation coordinates u=pi*log(x), source iid U(-pi,pi).
Noise is additive sigma*dW in u, multiplicative (sigma/pi)*x*dW in abundance.
Hyperedge effects are centered and scaled analytically under this intervention
to match the zero mean and per-receiver RMS of K_e*sin(...) in Kuramoto.
No truth structure is intended as input to a density fitter.
"""
from __future__ import annotations

import numpy as np

from scripts.spt_local_hyperedge_kuramoto import SOURCES, SUPPORTS, nesting_levels

LOG_SCALE=np.pi
DEFAULT_SELF_LIMIT=1.


def product_moments(order):
    """E[P], Var(P) for P=prod of order-1 independent exp(U[-1,1])."""
    if not 2<=order<=9:
        raise ValueError('Expected an interaction involving 2..9 distinct nodes')
    mean=float(np.sinh(1.)**(order-1))
    variance=float((np.sinh(2.)/2)**(order-1)-mean**2)
    return mean,variance


def abundance(coordinates):
    coordinates=np.asarray(coordinates,dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError('Nonfinite log-abundance coordinates')
    with np.errstate(over='raise',under='raise',invalid='raise'):
        result=np.exp(coordinates/LOG_SCALE)
    if np.any(result<=0):
        raise ArithmeticError('Loss of positivity; no abundance floor is applied')
    return result


def edge_force(coordinates,edge):
    """Normalized competitive hyperedge force in log-abundance coordinates."""
    values=np.asarray(coordinates,dtype=float)
    if values.ndim!=2 or values.shape[1]!=9 or not np.isfinite(values).all():
        raise ValueError('Expected finite (samples,9) log-abundance coordinates')
    edge=tuple(edge)
    if len(edge)<2 or len(set(edge))!=len(edge) or not set(edge)<=set(SOURCES):
        raise ValueError('Expected distinct valid source labels, at least two')
    indices=np.asarray(edge,dtype=int)-1
    local=values[:,indices]/LOG_SCALE
    with np.errstate(over='raise',under='raise',invalid='raise'):
        products=np.exp(local.sum(axis=1,keepdims=True)-local)
    mean,variance=product_moments(len(edge))
    result=np.zeros_like(values)
    result[:,indices]=-(products-mean)/np.sqrt(2*variance)
    return result


class LocalHyperedgeGLV:
    domain='real'

    def __init__(self,case,*,coupling=1.,decay=.5,noise=.08,frequencies=None,
                 self_limit=DEFAULT_SELF_LIMIT):
        if case not in SUPPORTS:
            raise ValueError('Expected A, B, or C')
        parameters=[coupling,decay,noise,self_limit]
        if not np.isfinite(parameters).all() or coupling<=0 or not 0<decay<1 or noise<0 or self_limit<=0:
            raise ValueError('Invalid coupling, decay, noise or self limitation')
        self.case=case; self.supports=SUPPORTS[case]
        self.levels=nesting_levels(self.supports)
        self.weights={edge:float(coupling*decay**self.levels[edge]) for edge in self.supports}
        self.noise=float(noise); self.self_limit=float(self_limit)
        self.frequencies=np.linspace(-.2,.2,9) if frequencies is None else np.asarray(frequencies,dtype=float)
        if self.frequencies.shape!=(9,) or not np.isfinite(self.frequencies).all():
            raise ValueError('Expected nine finite baseline drift offsets')
        # du baseline = omega + pi*d*(E[x]-x), also centered under q.
        # The corresponding raw-abundance dynamics is exactly HO-gLV.
        self.growth=np.full(9,self_limit*np.sinh(1.))+self.frequencies/LOG_SCALE
        self.coefficients={}
        for edge,weight in self.weights.items():
            mean,variance=product_moments(len(edge))
            coefficient=-weight/(LOG_SCALE*np.sqrt(2*variance))
            self.coefficients[edge]=float(coefficient)
            self.growth[np.asarray(edge)-1]-=coefficient*mean

    def drift(self,coordinates):
        values=np.asarray(coordinates,dtype=float)
        if values.ndim!=2 or values.shape[1]!=9 or not np.isfinite(values).all():
            raise ValueError('Expected finite (samples,9) log-abundance coordinates')
        value=self.frequencies+LOG_SCALE*self.self_limit*(np.sinh(1.)-abundance(values))
        for edge,weight in self.weights.items():
            value+=weight*edge_force(values,edge)
        return value

    def raw_drift(self,populations,*,ito=True):
        """Abundance SDE drift, including the explicit Ito transform correction."""
        populations=np.asarray(populations,dtype=float)
        if populations.ndim!=2 or populations.shape[1]!=9 or not np.isfinite(populations).all() or np.any(populations<=0):
            raise ValueError('Expected positive finite (samples,9) abundances')
        per_capita=np.broadcast_to(self.growth,populations.shape).copy()-self.self_limit*populations
        for edge,coefficient in self.coefficients.items():
            indices=np.asarray(edge)-1
            local=populations[:,indices]
            per_capita[:,indices]+=coefficient*np.prod(local,axis=1,keepdims=True)/local
        if ito:
            per_capita+=.5*(self.noise/LOG_SCALE)**2
        return populations*per_capita

    def sample_source(self,n,rng):
        if n<1:
            raise ValueError('Need at least one source state')
        return rng.uniform(-np.pi,np.pi,size=(n,9))

    def step(self,coordinates,rng,*,horizon=.20,dt=.01,increments=None):
        """Same RK4 drift/noise splitting as Kuramoto, on the real state space."""
        ratio=horizon/dt if dt>0 else -1
        steps=int(round(ratio))
        if horizon<=0 or dt<=0 or steps<1 or not np.isclose(ratio,steps,rtol=0,atol=1e-10):
            raise ValueError('Require positive horizon and an integer number of steps')
        value=np.asarray(coordinates,dtype=float).copy()
        if increments is not None and np.asarray(increments).shape!=(steps,*value.shape):
            raise ValueError('Noise increments must have (steps,samples,9) shape')
        for t in range(steps):
            k1=self.drift(value); k2=self.drift(value+.5*dt*k1)
            k3=self.drift(value+.5*dt*k2); k4=self.drift(value+dt*k3)
            value+=dt*(k1+2*k2+2*k3+k4)/6
            value+=self.noise*np.sqrt(dt)*rng.normal(size=value.shape) if increments is None else increments[t]
            if not np.isfinite(value).all():
                raise ArithmeticError('Nonfinite HO-gLV trajectory; no clipping or sample deletion')
        abundance(value)  # fail on underflow/overflow instead of hiding it
        return value

    def metadata(self):
        return {'case':self.case,'domain':'real','coordinate':'u=pi*log(x)',
            'physical_intervention':'independent log-uniform x in [exp(-1),exp(1)]',
            'source_coordinate_intervention':'iid Uniform(-pi,pi)',
            'supports':self.supports,'weights':{','.join(map(str,e)):k for e,k in self.weights.items()},
            'growth':self.growth.tolist(),'self_limit':self.self_limit,
            'raw_coefficients':{','.join(map(str,e)):a for e,a in self.coefficients.items()},
            'noise_coordinate_sd':self.noise,'noise_abundance':'(sigma/pi)*x*dW',
            'ito_correction_in_raw_drift':.5*(self.noise/LOG_SCALE)**2,
            'calibration':'each edge centered under common q; coordinate RMS K_e/sqrt(2)',
            'clipping':False,'source_labels':SOURCES}
