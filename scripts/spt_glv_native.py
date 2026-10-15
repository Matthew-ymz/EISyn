"""Float64 same-function acceleration for learned abundance monomial means."""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import shutil
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
_FUNCTION=None
_SUBSET_FUNCTION=None


def function():
    global _FUNCTION
    if _FUNCTION is not None: return _FUNCTION
    source=Path(__file__).with_name('spt_glv_mean_kernel.cpp')
    stamp=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    output=ROOT/'results/spt_local_glv_tm/native'/f'mean_{stamp}.so'
    if not output.exists():
        compiler=shutil.which('clang++') or shutil.which('c++')
        if compiler is None: raise RuntimeError('Native mean acceleration needs a local C++ compiler')
        output.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run([compiler,'-O3','-ffp-contract=off','-std=c++17','-shared','-fPIC','-pthread',str(source),'-o',str(output)],
                       check=True,capture_output=True,text=True)
    library=ctypes.CDLL(str(output)); native=library.glv_mean
    ptr=ctypes.c_void_p
    native.argtypes=[ctypes.c_long,ptr,ptr,ctypes.c_int,ptr,ptr,ptr,ptr,ptr,ptr,ctypes.c_int]
    native.restype=ctypes.c_int
    native._library=library
    _FUNCTION=native
    return native


def pack(vectors,coefficients):
    masks=[]; powers=[]; outputs=[]; values=[]
    for i,(rows,beta) in enumerate(zip(vectors,coefficients)):
        for vector,coefficient in zip(rows,beta):
            nonzero=np.flatnonzero(vector)
            if not len(nonzero): raise ValueError('Constant belongs in the intercept')
            power=int(round(float(vector[nonzero[0]]*np.pi)))
            if power not in (1,2,3) or not np.allclose(vector[nonzero]*np.pi,power,rtol=0,atol=1e-12):
                raise ValueError('Unsupported vector for monomial kernel')
            masks.append(sum(1<<int(j) for j in nonzero)); powers.append(power); outputs.append(i); values.append(coefficient)
    return (np.asarray(masks,dtype=np.int32),np.asarray(powers,dtype=np.int32),
            np.asarray(outputs,dtype=np.int32),np.asarray(values,dtype=np.float64))


def mean(source,intercept,terms,*,jacobian=False):
    source=np.ascontiguousarray(source,dtype=np.float64)
    intercept=np.ascontiguousarray(intercept,dtype=np.float64)
    result=np.empty_like(source)
    jac=np.empty((len(source),9,9),dtype=np.float64) if jacobian else None
    masks,powers,outputs,values=terms
    arrays=[source,intercept,masks,powers,outputs,values,result]
    pointers=[ctypes.c_void_p(a.ctypes.data) for a in arrays]
    error=function()(len(source),pointers[0],pointers[1],len(values),*pointers[2:],
                     ctypes.c_void_p(jac.ctypes.data) if jacobian else None,4)
    if error: raise ArithmeticError('Nonfinite compiled monomial mean')
    return (result,jac) if jacobian else result


def subset_mean(source,intercept,terms,targets,variables=None):
    global _SUBSET_FUNCTION
    if _SUBSET_FUNCTION is None:
        native=function()._library.glv_subset
        ptr=ctypes.c_void_p
        native.argtypes=[ctypes.c_long,ptr,ptr,ctypes.c_int,ptr,ptr,ptr,ptr,
                         ctypes.c_int,ptr,ctypes.c_int,ptr,ptr,ptr,ctypes.c_int]
        native.restype=ctypes.c_int
        _SUBSET_FUNCTION=native
    source=np.ascontiguousarray(source,dtype=np.float64)
    intercept=np.ascontiguousarray(intercept,dtype=np.float64)
    targets=np.ascontiguousarray(targets,dtype=np.int32)
    if targets.ndim!=1 or not len(targets) or np.any((targets<0)|(targets>=9)):
        raise ValueError('Invalid target columns')
    derivative=variables is not None
    variables=np.ascontiguousarray([] if variables is None else variables,dtype=np.int32)
    if variables.ndim!=1 or np.any((variables<0)|(variables>=9)):
        raise ValueError('Invalid derivative columns')
    result=np.empty((len(source),len(targets)),dtype=np.float64)
    jac=np.empty((len(source),len(targets),len(variables)),dtype=np.float64) if derivative else None
    masks,powers,outputs,values=terms
    ptr=lambda a:ctypes.c_void_p(a.ctypes.data)
    error=_SUBSET_FUNCTION(len(source),ptr(source),ptr(intercept),len(values),
        ptr(masks),ptr(powers),ptr(outputs),ptr(values),len(targets),ptr(targets),
        len(variables),ptr(variables),ptr(result),ptr(jac) if derivative else None,4)
    if error: raise ArithmeticError('Nonfinite selected monomial mean or derivative')
    return (result,jac) if derivative else result
