"""Future-pair MI of the same frozen TM by rotational Fourier quadrature.

Common phase rotations make each Yi uniform. For D=Yi-Yj, MI is the KL
divergence of p(D) from Uniform; VM noise is integrated analytically.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import ive,i0e
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.spt_kuramoto_b_comparison import FOLDER,load_model,budget_deadline
from scripts.spt_kuramoto_finite_tm import save_json,check_time,TWO_PI
from scripts.spt_multilevel_parity_case import graph_tree
from scripts.spt_continuous_phase_benchmark import score


def information_matrix(coefficients,order,grid):
    angles=np.arange(grid)*TWO_PI/grid
    basis=np.exp(-1j*np.outer(np.arange(1,order+1),angles))
    result=np.zeros((9,9)); minimum=1.; normalization_error=0.
    for i in range(9):
        for j in range(i+1,9):
            relative=1+2*np.real(coefficients[1:order+1,i,j]@basis)
            minimum=min(minimum,float(relative.min()))
            if np.any(relative<=0):
                raise ArithmeticError(f'Nonpositive Fourier density for future pair {i+1},{j+1}: {relative.min()}')
            normalization_error=max(normalization_error,abs(float(relative.mean())-1))
            result[i,j]=result[j,i]=np.mean(relative*np.log(relative))
    return result,{'minimum_relative_density':minimum,'normalization_error':normalization_error}


def tree(matrix):
    root,clades=graph_tree(matrix)
    # Scoring truth is imported only after the information matrix is fixed.
    from scripts.spt_local_hyperedge_kuramoto import SUPPORTS
    return score({'status':'exploratory','root_split':root,'clades':clades},SUPPORTS['B'])


def run(folder=FOLDER,power_high=18):
    output=folder/('future_mi.json' if power_high==18 else f'future_mi_refined_p{power_high}.json')
    if output.exists() and json.loads(output.read_text()).get('status')=='complete':
        return json.loads(output.read_text())
    started=time.monotonic(); deadline=budget_deadline(folder)
    model,_,digest=load_model()
    n_low=2**(power_high-2); n_high=2**power_high; order_high=32
    data={'protocol':{'model_sha256':digest,'source_counts':[n_low,n_high],
        'orders':[16,32],'grids':[1024,2048],'scrambles':4,'seed_base':20261015,
        'seed_stride':1000,'matrix_precision_target_nats':1e-4,
        'target':'future-pair MI; same frozen product-VM joint, VM noise integrated analytically',
        'replication':'numerical scrambles of one fitted physical dataset; not independent datasets'},
        'records':[],'status':'running'}
    probe=np.random.default_rng(20261015).uniform(-np.pi,np.pi,(32,9))
    shift=.731
    equivariance=float(np.max(abs(model.mean(probe+shift)-model.mean(probe)-shift)))
    if equivariance>1e-10: raise ArithmeticError('Rotational invariance required for pair reduction')
    data['rotational_equivariance_error_rad']=equivariance
    noise=ive(np.arange(order_high+1)[:,None],model.kappa)/i0e(model.kappa)
    try:
        for scramble in range(4):
            check_time(deadline)
            points=qmc.Sobol(18,scramble=True,seed=20261015+1000*scramble).random_base2(power_high)
            source=(points[:,:9]-.5)*TWO_PI
            mean=model.mean(source); base=np.exp(1j*mean); power=np.ones_like(base)
            coefficients={n:np.empty((order_high+1,9,9),dtype=complex) for n in (n_low,n_high)}
            for n in coefficients: coefficients[n][0]=1
            for k in range(1,order_high+1):
                check_time(deadline); power*=base
                for n in coefficients:
                    c=power[:n].T@power[:n].conj()/n
                    coefficients[n][k]=c*np.outer(noise[k],noise[k])
            rows=[]
            for n in (n_low,n_high):
                for order,grid in ((16,2048),(32,1024),(32,2048)):
                    matrix,diagnostic=information_matrix(coefficients[n],order,grid)
                    rows.append({'sources':n,'order':order,'grid':grid,
                        'matrix_nats':matrix.tolist(),'diagnostics':diagnostic})
            data['records'].append({'scramble':scramble,'layouts':rows})
            data['elapsed_seconds']=time.monotonic()-started; save_json(output,data)
            print(json.dumps({'stage':'B future MI','scramble':scramble,'source_count':n_high}),flush=True)
        def get(n,k,g):
            return np.array([next(v['matrix_nats'] for v in r['layouts']
                if (v['sources'],v['order'],v['grid'])==(n,k,g)) for r in data['records']])
        high=get(n_high,32,2048); mean=high.mean(0); sem=high.std(0,ddof=1)/2
        changes={'source_count':float(np.max(abs(mean-get(n_low,32,2048).mean(0)))),
                 'fourier_order':float(np.max(abs(mean-get(n_high,16,2048).mean(0)))),
                 'angle_grid':float(np.max(abs(mean-get(n_high,32,1024).mean(0))))}
        metrics=tree(mean); replicate_metrics=[tree(m) for m in high]
        data.update(status='complete',matrix_nats=mean.tolist(),matrix_sem_nats=sem.tolist(),
            changes_nats=changes,maximum_matrix_sem_nats=float(sem.max()),
            precision_passed=max(*changes.values(),float(sem.max()))<1e-4,
            metrics=metrics,numerical_replicate_metrics=replicate_metrics)
        # Identity mapping: nonzero Fourier modes of independent phase
        # differences vanish exactly; noise factors preserve the zero MI.
        control=np.zeros((33,9,9),dtype=complex); control[0]=1
        control_mi,_=information_matrix(control,32,2048)
        data['independent_uniform_identity_control_maximum_mi_nats']=float(control_mi.max())
    except (TimeoutError,ArithmeticError) as error:
        data.update(status='budget_exhausted' if isinstance(error,TimeoutError) else 'numerical_failure',error=str(error))
    data['elapsed_seconds']=time.monotonic()-started; save_json(output,data)
    return data


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--power-high',type=int,default=18)
    args=parser.parse_args()
    result=run(power_high=args.power_high)
    print(json.dumps({k:result[k] for k in ('status','elapsed_seconds','precision_passed','metrics') if k in result}))
