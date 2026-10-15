"""Data-only finite-time flow mean for the circular conditional TM.

Every one of the 502 possible non-singleton hyperedges is a candidate. Paired
observations select coefficients; the planted architecture is never accepted.
An integrated smooth vector field avoids fitting a folded near-identity map.
The independent von Mises observation approximation remains explicit.
"""
from __future__ import annotations

import itertools
import time
import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import Lasso

from scripts.spt_local_hyperedge_kuramoto import wrap
from scripts.spt_kuramoto_finite_tm import PeriodicConditionalTM, check_time, fit_concentration, ALPHAS

_NATIVE=None


def native_kernel():
    global _NATIVE
    if _NATIVE is None:
        source=Path(__file__).with_name('spt_kuramoto_flow_kernel.cpp')
        folder=Path(__file__).resolve().parents[1]/'results/spt_local_kuramoto_tm'
        folder.mkdir(parents=True,exist_ok=True)
        library=folder/('flow_kernel.dylib' if sys.platform=='darwin' else 'flow_kernel.so')
        if not library.exists() or library.stat().st_mtime<source.stat().st_mtime:
            subprocess.run(['c++','-O3','-std=c++17','-dynamiclib' if sys.platform=='darwin' else '-shared',
                            '-fPIC','-pthread',str(source),'-o',str(library)],check=True,capture_output=True)
        _NATIVE=ctypes.CDLL(str(library)).kuramoto_flow
        pointer=ctypes.c_void_p
        _NATIVE.argtypes=[ctypes.c_int,pointer,pointer,pointer,pointer,pointer,
                         ctypes.c_int,pointer,pointer,ctypes.c_double,ctypes.c_int,ctypes.c_int]
        _NATIVE.restype=None
    return _NATIVE


def candidate_edges():
    return tuple(e for size in range(2,10) for e in itertools.combinations(range(9),size))


class PeriodicFlowConditionalTM(PeriodicConditionalTM):
    def __init__(self, edges, weights, frequencies, kappa, horizon=.2, steps=8):
        self.edges=tuple(np.asarray(e,dtype=int) for e in edges)
        self.weights=np.asarray(weights,dtype=float)
        self.frequencies=np.asarray(frequencies,dtype=float)
        self.kappa=np.asarray(kappa,dtype=float)
        self.horizon=float(horizon); self.steps=int(steps)
        if np.any(self.weights<0) or np.any(self.kappa<=0):
            raise ValueError('Attractive flow weights nonnegative, concentrations positive')

    def _rhs(self,value,sensitivity,parameter_sensitivity=False):
        force=np.broadcast_to(self.frequencies,value.shape).copy()
        jac=np.zeros((len(value),9,9)) if sensitivity is not None else None
        direct=np.zeros_like(sensitivity) if parameter_sensitivity else None
        for k,(indices,weight) in enumerate(zip(self.edges,self.weights)):
            local=value[:,indices]
            phase=local.sum(1,keepdims=True)-len(indices)*local
            sine=np.sin(phase)
            force[:,indices]+=weight*sine
            if sensitivity is not None:
                cosine=weight*np.cos(phase)
                for receiver,i in enumerate(indices):
                    jac[:,i,indices]+=cosine[:,receiver,None]
                    jac[:,i,i]-=len(indices)*cosine[:,receiver]
            if parameter_sensitivity:
                direct[:,indices,k]=sine
        if sensitivity is None:
            return force,None
        derivative=np.einsum('nij,njp->nip',jac,sensitivity,optimize=True)
        if parameter_sensitivity:
            direct[:,:,len(self.edges):]+=np.eye(9)
            derivative+=direct
        return force,derivative

    def propagate(self,source,*,source_jacobian=False,parameter_jacobian=False,steps=None):
        if source_jacobian and parameter_jacobian:
            raise ValueError('Choose one sensitivity mode')
        value=np.asarray(source,dtype=float).copy()
        sensitivity=None
        if source_jacobian:
            sensitivity=np.broadcast_to(np.eye(9),(len(value),9,9)).copy()
        if parameter_jacobian:
            sensitivity=np.zeros((len(value),9,len(self.edges)+9))
        steps=self.steps if steps is None else steps
        dt=self.horizon/steps
        for _ in range(steps):
            k1,j1=self._rhs(value,sensitivity,parameter_jacobian)
            k2,j2=self._rhs(value+.5*dt*k1,None if sensitivity is None else sensitivity+.5*dt*j1,parameter_jacobian)
            k3,j3=self._rhs(value+.5*dt*k2,None if sensitivity is None else sensitivity+.5*dt*j2,parameter_jacobian)
            k4,j4=self._rhs(value+dt*k3,None if sensitivity is None else sensitivity+dt*j3,parameter_jacobian)
            value+=dt*(k1+2*k2+2*k3+k4)/6
            if sensitivity is not None:
                sensitivity+=dt*(j1+2*j2+2*j3+j4)/6
        return (value,sensitivity) if sensitivity is not None else value

    def mean(self,source,*,jacobian=False):
        source=np.ascontiguousarray(source,dtype=np.float64)
        output=np.empty_like(source)
        jac=np.empty((len(source),9,9)) if jacobian else None
        edges=np.zeros((len(self.edges),9),dtype=np.int32)
        sizes=np.array([len(e) for e in self.edges],dtype=np.int32)
        for j,e in enumerate(self.edges): edges[j,:len(e)]=e
        native_kernel()(len(source),source.ctypes.data,output.ctypes.data,
            None if jac is None else jac.ctypes.data,edges.ctypes.data,sizes.ctypes.data,len(edges),
            self.weights.ctypes.data,self.frequencies.ctypes.data,self.horizon,self.steps,4)
        return (output,jac) if jacobian else output

    def payload(self):
        return {'family':'learned_flow','edges':[e.tolist() for e in self.edges],
            'weights':self.weights.tolist(),'frequencies':self.frequencies.tolist(),
            'kappa':self.kappa.tolist(),'horizon':self.horizon,'steps':self.steps}

    @classmethod
    def load(cls,payload):
        return cls(**{k:v for k,v in payload.items() if k!='family'})

    @classmethod
    def fit(cls,source,target,validation_source,validation_target,*,deadline):
        started=time.monotonic()
        displacement=wrap(target-source)
        validation_displacement=wrap(validation_target-validation_source)
        n=len(source); edges=candidate_edges(); horizon=.2
        bias=displacement.mean(0)
        design=np.zeros((n*9,len(edges)),order='F')
        # Fixed candidate family. No support or generator parameters enter.
        for k,edge in enumerate(edges):
            check_time(deadline)
            idx=np.asarray(edge)
            phase=source[:,idx].sum(1,keepdims=True)-len(idx)*source[:,idx]
            for j,i in enumerate(idx):
                design[i::9,k]=horizon*np.sin(phase[:,j])
        labels=(displacement-bias).ravel()
        regression=Lasso(fit_intercept=False,positive=True,max_iter=2000,tol=1e-7,warm_start=True)
        candidates=[]
        for alpha in ALPHAS:
            check_time(deadline)
            regression.set_params(alpha=alpha)
            regression.fit(design,labels)
            beta=regression.coef_.copy()
            active=np.flatnonzero(beta)
            probe=cls([edges[k] for k in active],beta[active],bias/horizon,np.ones(9))
            residual=wrap(validation_target-probe.mean(validation_source))
            candidates.append((float(np.mean(residual**2)),alpha,active,beta[active]))
        del design
        mse,alpha,active,initial=min(candidates,key=lambda r:r[0])
        if not len(active):
            raise ArithmeticError('Flow screening selected no active candidate interactions')
        model=cls([edges[k] for k in active],initial,bias/horizon,np.ones(9))
        parameters=np.r_[initial,bias/horizon]
        lower=np.r_[np.zeros(len(active)),np.full(9,-np.inf)]
        upper=np.full(len(parameters),np.inf)
        # A deterministic training prefix limits fit cost; final least squares
        # then uses all registered training samples. Test data are never used.
        stages=(min(2048,n),n)
        history=[]
        for count in stages:
            cache={}
            def calculate(parameters):
                check_time(deadline)
                if 'parameters' in cache and np.array_equal(parameters,cache['parameters']):
                    return cache['residual'],cache['jacobian']
                model.weights=parameters[:len(active)].copy()
                model.frequencies=parameters[len(active):].copy()
                prediction,jac=model.propagate(source[:count],parameter_jacobian=True)
                residual=wrap(prediction-target[:count]).ravel()/np.sqrt(count)
                cache.update(parameters=parameters.copy(),residual=residual,
                             jacobian=jac.reshape(count*9,-1)/np.sqrt(count))
                return cache['residual'],cache['jacobian']
            fit=least_squares(lambda p:calculate(p)[0],parameters,jac=lambda p:calculate(p)[1],
                bounds=(lower,upper),max_nfev=16,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            parameters=fit.x
            model.weights=parameters[:len(active)].copy()
            model.frequencies=parameters[len(active):].copy()
            row={'samples':count,'function_evaluations':fit.nfev,'converged':bool(fit.success),
                 'training_mse_rad2':float(2*fit.cost/9),'optimality':float(fit.optimality)}
            history.append(row)
            print({'stage':'flow_fit',**row},flush=True)
        residual=wrap(target-model.mean(source))
        model.kappa=np.array([fit_concentration(residual[:,i]) for i in range(9)])
        validation_error=wrap(validation_target-model.mean(validation_source))
        coarse=model.propagate(validation_source[:256])
        fine=model.propagate(validation_source[:256],steps=2*model.steps)
        integration_difference=wrap(coarse-fine)
        report={'family':'data-only integrated autonomous Fourier vector field',
            'candidate_edges':len(edges),'screening_alpha':alpha,'active_candidate_edges':len(active),
            'screening_candidates':[{'alpha':r[1],'validation_mse_rad2':r[0], 'active':len(r[2])} for r in candidates],
            'stages':history,'fit_seconds':time.monotonic()-started,
            'validation_rmse_rad':np.sqrt(np.mean(validation_error**2,axis=0)).tolist(),
            'mean_integrator_steps':model.steps,'mean_integrator_max_halfstep_difference_rad':float(np.max(abs(integration_difference))),
            'estimator_difference':'all-candidate shared attractive vector-field family replaces finite-map dictionary; independent constant VM noise approximation remains'}
        if report['mean_integrator_max_halfstep_difference_rad']>1e-5:
            raise ArithmeticError('Learned mean flow integration has not converged to 1e-5 rad')
        return model,report
