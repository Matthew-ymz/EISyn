"""Fixed-outer paired P=512 -> 1024 check; uses cached lower-P prefixes."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri
from threadpoolctl import threadpool_limits

from scripts.refine_spt_third_case import OUT, SEEDS, one_integral, read, fixed_tree_shares
from scripts.kuramoto_six_topology_core import MLPChannel, features, save
from scripts.spt_bounded_joint_integration import zero_mean_projection_controls
from scripts.summarize_spt_weak_coupling import fit_fourier_control


def main():
    deadline = read(OUT/'budget.json')['deadline_epoch']
    save(OUT/'inner_process.json', dict(pid=__import__('os').getpid(), deadline_epoch=deadline))
    pending = [(512, 1024, seed, deadline) for seed in SEEDS
               if not (OUT/f'ei_N512_P1024_R{seed}.npz').exists()]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(one_integral, arg) for arg in pending]
        for f in as_completed(futures):
            row = f.result()
            print(json.dumps(dict(seed=row['scramble'], seconds=row['seconds'])), flush=True)
    model = MLPChannel(read(OUT/'model.json'))
    fourier = fit_fourier_control(model)
    lower, upper, differences = [], [], []
    for seed in SEEDS:
        lo = np.load(OUT/f'ei_N2048_P512_R{seed}.npz')['pointwise_ei'][:, :512]
        hi = np.load(OUT/f'ei_N512_P1024_R{seed}.npz')['pointwise_ei']
        u = qmc.Sobol(12, scramble=True, seed=seed).random_base2(9)
        x, z = (u[:, :6]-.5)*2*np.pi, ndtri(u[:, 6:])
        control = zero_mean_projection_controls(model, x, z)
        if fourier and fourier['enabled']:
            control -= (features(x, kind=fourier['selection']['feature_kind'])@
                        np.array(fourier['coefficients'])).T
        lower.append((lo+control).mean(1))
        upper.append((hi+control).mean(1))
        differences.append((hi-lo).mean(1))
    lower, upper, differences = np.array(lower), np.array(upper), np.array(differences)
    tree = read(OUT/'spt_projection.json')['tree']
    lo_shares, hi_shares = fixed_tree_shares(tree, lower.mean(0)), fixed_tree_shares(tree, upper.mean(0))
    max_ei = float(abs(differences.mean(0)).max())
    max_sem = float((differences.std(0, ddof=1)/2).max())
    max_share = float(max(abs(lo_shares[key]-hi_shares[key]) for key in lo_shares))
    result = dict(fixed_outer=512, inner_particles=[512, 1024], scrambles=SEEDS,
        lower_source='first 512 outer points of each completed N2048/P512 cache',
        maximum_mean_ei_change_nats=max_ei, maximum_ei_difference_sem_nats=max_sem,
        maximum_node_share_change_percentage_points=max_share,
        lower_shares=lo_shares, upper_shares=hi_shares,
        fixed_common_tree=True, no_F1_selection=True,
        passed=bool(max_ei < .01 and max_sem < .01 and max_share <= 1.), finished_epoch=time.time())
    save(OUT/'inner_precision_check.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
