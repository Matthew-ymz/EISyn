"""Nine scalar phases with local, irreducible hyperedge sine interactions.

Design-stage generator only: no EI, learned TM, SPT, or replacement scores.
The m=3 kernel matches the form of manuscript SI S7 Eq(S69), up to strength.
General m>3 is an explicitly proposed extension, not an established manuscript
benchmark. Source sets preserve the current A/B/C diagram layouts.
"""
from __future__ import annotations

import numpy as np

SOURCES=tuple(range(1,10))
SUPPORTS={
    'A':((1,2,3),(4,5),(6,7),(8,9),(1,2,3,4,5),(6,7,8,9),SOURCES),
    'B':((1,2),(1,2,3),(1,2,3,4),(1,2,3,4,5),(6,7),(6,7,8),(6,7,8,9),SOURCES),
    'C':((1,2,3),(4,5,6),(7,8,9),(1,2,3,4,5,6),SOURCES),
}


def wrap(phases):
    return (np.asarray(phases)+np.pi)%(2*np.pi)-np.pi


def nesting_levels(supports):
    """Innermost edges have level 0; each strict inclusion adds a level."""
    levels={}
    for edge in sorted(supports,key=lambda e:(len(e),e)):
        levels[edge]=1+max((levels[child] for child in levels if set(child)<set(edge)),default=-1)
    return levels


def edge_force(phases,edge):
    """Unit-strength force: zero outside e; sin(sum_e theta - |e| theta_i) inside."""
    phases=np.asarray(phases,dtype=float)
    edge=tuple(edge)
    if phases.ndim!=2 or phases.shape[1]!=9 or not np.isfinite(phases).all():
        raise ValueError('Expected finite (samples,9) phase states')
    if len(edge)<2 or len(set(edge))!=len(edge) or not set(edge)<=set(SOURCES):
        raise ValueError('Expected distinct valid source labels, at least two')
    indices=np.asarray(edge,dtype=int)-1
    local=phases[:,indices]
    force=np.zeros_like(phases)
    force[:,indices]=np.sin(local.sum(axis=1,keepdims=True)-len(edge)*local)
    return force


class LocalHyperedgeKuramoto:
    def __init__(self,case,*,coupling=1.,decay=.5,noise=.08,frequencies=None):
        if case not in SUPPORTS:
            raise ValueError('Expected A, B, or C')
        if not np.isfinite([coupling,decay,noise]).all() or coupling<=0 or not 0<decay<1 or noise<0:
            raise ValueError('Need positive coupling, 0<decay<1, and nonnegative noise')
        self.case=case; self.supports=SUPPORTS[case]
        self.levels=nesting_levels(self.supports)
        self.weights={edge:float(coupling*decay**self.levels[edge]) for edge in self.supports}
        self.noise=float(noise)
        # Shared fixed frequency profile for all three candidate architectures.
        self.frequencies=np.linspace(-.2,.2,9) if frequencies is None else np.asarray(frequencies,dtype=float)
        if self.frequencies.shape!=(9,) or not np.isfinite(self.frequencies).all():
            raise ValueError('Expected nine finite scalar frequencies')

    def drift(self,phases):
        phases=np.asarray(phases,dtype=float)
        if phases.ndim!=2 or phases.shape[1]!=9 or not np.isfinite(phases).all():
            raise ValueError('Expected finite (samples,9) phase states')
        value=np.broadcast_to(self.frequencies,phases.shape).copy()
        for edge,weight in self.weights.items():
            value+=weight*edge_force(phases,edge)
        return value

    def sample_source(self,n,rng):
        """Registered product intervention on the nine scalar source phases."""
        if n<1:
            raise ValueError('Need at least one source state')
        return rng.uniform(-np.pi,np.pi,size=(n,9))

    def step(self,phases,rng,*,horizon=.20,dt=.01):
        """RK4 drift + Gaussian diffusion increments, wrapped after each step.

        Numerical approximation to dtheta=f(theta)dt+sigma dW, not an exact
        finite-time transition or a fourth-order stochastic solver.
        """
        ratio=horizon/dt if dt>0 else -1
        steps=int(round(ratio))
        if horizon<=0 or dt<=0 or steps<1 or not np.isclose(ratio,steps,rtol=0,atol=1e-10):
            raise ValueError('Require positive horizon and an integer number of time steps')
        value=wrap(phases)
        for _ in range(steps):
            k1=self.drift(value)
            k2=self.drift(value+.5*dt*k1)
            k3=self.drift(value+.5*dt*k2)
            k4=self.drift(value+dt*k3)
            value=value+dt*(k1+2*k2+2*k3+k4)/6
            value=wrap(value+self.noise*np.sqrt(dt)*rng.normal(size=value.shape))
        return value
