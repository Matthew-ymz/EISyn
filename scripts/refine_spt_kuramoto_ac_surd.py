"""Extend the same SURD Sobol streams from 1024 to 4096 targets, reusing prefixes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_kuramoto_ac_selected import SelectedCase
from scripts.spt_kuramoto_ac_surd import posterior_moments, marginal_specific, specific_audit
from scripts.spt_kuramoto_finite_tm import FullTargetIntegrator, posterior_proposal, check_time, save_json
from scripts.surd_batch_allocation import allocate_batch
from scripts.compare_spt_surd_three_cases import maximum_weight_tree


def run(case):
    context = SelectedCase(case)
    folder = context.folder
    base = json.loads((folder / 'surd_selected_n1024_p8192.json').read_text())
    if base['status'] != 'complete':
        raise ValueError('Complete midpoint-protocol N1024 SURD first')
    expected = {'case': case, 'model_sha256': context.digest, 'outer': 1024,
                'particles': 8192, 'scrambles': 4,
                'posterior_uniforms': '30-bit scrambled Sobol cell midpoints; open interval'}
    if any(base['protocol'].get(k) != v for k, v in expected.items()):
        raise ValueError('SURD prefix does not match the frozen midpoint/model protocol')
    output = folder / 'surd_selected_refined_n4096_p8192.json'
    protocol = {**base['protocol'], 'outer': 4096, 'extension': 'reuse N1024; append same streams in chunks of 256'}
    data = json.loads(output.read_text()) if output.exists() else {'protocol': protocol, 'records': [], 'status': 'running'}
    if data['protocol'] != protocol:
        raise ValueError('Refined SURD cache protocol mismatch')
    model, _, _ = context.load_model()
    try:
        for r in range(4):
            if any(z['scramble'] == r for z in data['records']):
                continue
            seed = 20261015 + 1000 * r
            stream = FullTargetIntegrator(model, outer=4096, max_particles=1, seed=seed, deadline=context.deadline())
            initial = np.load(folder / f'surd_posterior_n1024_p8192_r{r}.npz')
            specific = {p: [initial[f'specific_p{p}']] for p in (4096, 8192)}
            defects = {p: [initial[f'gaussian_kl_p{p}']] for p in (4096, 8192)}
            audits = [next(z for z in base['records'] if z['scramble'] == r)['proposal_audit']]
            for start in range(1024, 4096, 256):
                check_time(context.deadline())
                path = folder / f'surd_extension_n4096_p8192_r{r}_start{start}.npz'
                if path.exists():
                    chunk = np.load(path)
                    if any(chunk[f'specific_p{p}'].shape != (256, 512)
                           or chunk[f'gaussian_kl_p{p}'].shape != (256,) for p in (4096, 8192)):
                        raise ValueError(f'Invalid extension chunk shape: {path.name}')
                    audits.append(json.loads(str(chunk['proposal_audit_json'])))
                else:
                    target = stream.target[start:start + 256]
                    center, chol, inversion = posterior_proposal(model, np.zeros((256, 9)), target,
                        np.arange(9), np.arange(9), iterations=16)
                    fit = posterior_moments(model, target, center, chol, outer=256, particles=8192,
                        seed=seed + 300000, deadline=context.deadline(), stream_offset=start)
                    evaluation = posterior_moments(model, target, center, chol, outer=256, particles=8192,
                        seed=seed + 400000, deadline=context.deadline(), stream_offset=start)
                    chunk = {}
                    for p in (4096, 8192):
                        f, e = fit[p], evaluation[p]
                        sp = marginal_specific(f['mean'], f['covariance'], e['mean'], e['covariance'])
                        defect = e['direct_specific'] - sp[:, 511]
                        specific_audit(sp, defect)
                        chunk[f'specific_p{p}'], chunk[f'gaussian_kl_p{p}'] = sp, defect
                    high = evaluation[8192]
                    audit = {'minimum_ess': float(min(fit[8192]['ess'].min(), high['ess'].min())),
                        'maximum_normalized_weight': float(max(fit[8192]['maximum_weight'].max(), high['maximum_weight'].max())),
                        'maximum_six_sd_chart_extent_rad': float(np.max(abs(high['mean']) + 6 * np.sqrt(np.diagonal(high['covariance'], axis1=1, axis2=2)))),
                        **inversion}
                    chunk['proposal_audit_json'] = np.array(json.dumps(audit))
                    np.savez_compressed(path, **chunk)
                    audits.append(audit)
                    print(json.dumps({'case': case, 'stage': 'SURD extension', 'scramble': r,
                                      'targets': start + 256}), flush=True)
                for p in (4096, 8192):
                    specific[p].append(chunk[f'specific_p{p}'])
                    defects[p].append(chunk[f'gaussian_kl_p{p}'])
            layouts = []
            for p in (4096, 8192):
                sp = np.concatenate(specific[p]); defect = np.concatenate(defects[p])
                for n in (1024, 4096):
                    atoms = allocate_batch(sp[:n]); weights = atoms['synergy_sum'] / n
                    tree = maximum_weight_tree(weights)
                    layouts.append({'outer': n, 'particles': p, 'synergy_atoms_nats': weights.tolist(),
                        'mean_specific_nats': (atoms['mi_sum'] / n).tolist(),
                        'metrics': context.scored(tree['root_split'], tree['clades']), 'tree': tree,
                        'allocation_audit': atoms['audit'], 'specific_audit': specific_audit(sp[:n], defect[:n])})
            proposal = {'minimum_ess': min(a['minimum_ess'] for a in audits),
                'maximum_normalized_weight': max(a['maximum_normalized_weight'] for a in audits),
                'maximum_six_sd_chart_extent_rad': max(a['maximum_six_sd_chart_extent_rad'] for a in audits)}
            data['records'].append({'scramble': r, 'layouts': layouts, 'proposal_audit': proposal})
            save_json(output, data)
        high = [next(z for z in r['layouts'] if (z['outer'], z['particles']) == (4096, 8192)) for r in data['records']]
        weights = np.asarray([z['synergy_atoms_nats'] for z in high])
        tree = maximum_weight_tree(weights.mean(0))
        data.update(status='complete', metrics=context.scored(tree['root_split'], tree['clades']),
            tree=tree, synergy_atoms_nats=weights.mean(0).tolist(),
            numerical_replicate_metrics=[z['metrics'] for z in high],
            maximum_atom_sem_nats=float((weights.std(0, ddof=1) / 2).max()),
            approximation_audits=[z['specific_audit'] for z in high])
    except (TimeoutError, ValueError, ArithmeticError, np.linalg.LinAlgError) as e:
        data.update(status='budget_exhausted' if isinstance(e, TimeoutError) else 'numerical_failure', error=str(e))
    save_json(output, data)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('A', 'C'), required=True)
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        result = run(args.case)
    print(json.dumps({'case': args.case, 'status': result['status']}))
