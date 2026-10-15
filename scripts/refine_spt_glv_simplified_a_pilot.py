"""Refine only failed outer rows within the existing approved pilot deadline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_glv_finite_tm import (
    RealConditionalTM, RealFullTargetIntegrator, proposal_sample, proposal_log_prob,
    DEFENSIVE, LOG_TWO_PI, TWO_PI,
)
from scripts.spt_glv_simplified_a_pilot import OUTPUT, read, syn_audit, require_valid_syn
from scripts.spt_kuramoto_finite_tm import check_time, save_json
from scripts.spt_glv_group_tm import RealGroupIntegrator, conditional_proposal

RESULT = OUTPUT.with_name('proposal_refinement.json')


def row_integrals(integrator, mask, rows, particles, uniforms):
    missing = np.array([j for j in range(9) if not mask >> j & 1])
    center, chol, _ = integrator.proposals[mask]
    proposed, logproposal = proposal_sample(center[rows], chol[rows],
        uniforms[:, :, 1:len(missing)+1], return_log_prob=True)
    prior = uniforms[:, :, 1:len(missing)+1] * TWO_PI - np.pi
    from_prior = uniforms[:, :, 0] < DEFENSIVE
    proposed = np.where(from_prior[:, :, None], prior, proposed)
    rr, cc = np.nonzero(from_prior)
    if len(rr):
        logproposal[rr, cc] = proposal_log_prob(prior[rr, cc][:, None, :], center[rows][rr], chol[rows][rr])[:, 0]
    mixture = np.logaddexp(np.log(1-DEFENSIVE) + logproposal,
                          np.log(DEFENSIVE) - len(missing) * LOG_TWO_PI)
    completed = np.repeat(integrator.source[rows, None, :], particles, axis=1)
    completed[:, :, missing] = proposed
    targets = np.repeat(integrator.target[rows], particles, axis=0)
    likelihood = integrator.model.log_prob(completed.reshape(-1, 9), targets).reshape(len(rows), particles)
    weights = likelihood - len(missing) * LOG_TWO_PI - mixture
    normalizer = logsumexp(weights, axis=1)
    normalized = weights - normalizer[:, None]
    return normalizer - np.log(particles), np.exp(-logsumexp(2*normalized, axis=1)), np.exp(normalized.max(1))


def worker(polished=False):
    pilot = read(OUTPUT)
    budget = read(OUTPUT.with_name('budget.json'))
    started = time.time()
    deadline = time.monotonic() + budget['deadline_epoch'] - time.time()
    payload = {'status': 'running', 'original_deadline_epoch': budget['deadline_epoch'],
               'rule': ('ESS<10 or maxweight>=.25; existing real-group-v4 bounded optimized 80/15/5 proposal, same learned channel and exact weights'
                        if polished else 'ESS<10 or maxweight>=.25; same learned channel and proposal; fresh nested per-row Sobol1024/4096/16384/65536'),
               'records': [], 'modifies_native_pilot': False}
    def save():
        payload['elapsed_seconds'] = time.time() - started
        payload['total_trial_wall_seconds'] = time.time() - budget['started_epoch']
        save_json(RESULT, payload)
    save()
    try:
        model = RealConditionalTM.load(pilot['model'])
        finest = [r for r in pilot['precision'] if r['outer'] == 4096]
        updated = [dict(r['ei_nats']) for r in finest]
        new_diagnostics = []
        for index, row in enumerate(finest):
            check_time(deadline)
            denominator = row['diagnostics']['denominator']
            if denominator['ess_minimum'] < 10 or denominator['maximum_normalized_weight'] >= .25:
                raise ValueError('This bounded diagnostic does not cover a failed shared denominator')
            new_diagnostics.append(denominator)
            bad = [int(k) for k, d in row['diagnostics']['queries'].items()
                   if d.get('ess_minimum') is not None and (d['ess_minimum'] < 10 or d['maximum_normalized_weight'] >= .25)]
            new_diagnostics.extend(d for k, d in row['diagnostics']['queries'].items()
                                   if int(k) not in bad and d.get('ess_minimum') is not None)
            if not bad:
                continue
            integrator = RealFullTargetIntegrator(model, outer=4096, max_particles=256,
                seed=20261015 + 1000*row['scramble'], deadline=deadline)
            denominator_values, _ = integrator.marginal(0, 256)
            for mask in bad:
                check_time(deadline)
                native, diagnostic = integrator.marginal(mask, 256)
                baseline_mean = float(np.mean(native - denominator_values))
                if abs(baseline_mean - row['ei_nats'][str(mask)]) > 1e-11:
                    raise ArithmeticError('Native pointwise marginal replay mismatch')
                all_rows = np.arange(4096)
                base, ess, maximum = row_integrals(integrator, mask, all_rows, 256, integrator.inner)
                if not np.allclose(base, native, rtol=0, atol=1e-11):
                    raise ArithmeticError('Native importance-weight replay mismatch')
                affected = np.flatnonzero((ess < 10) | (maximum >= .25))
                record = {'scramble': row['scramble'], 'mask': mask, 'affected_rows': affected.tolist(),
                          'baseline_information_nats': baseline_mean, 'refinements': []}
                payload['records'].append(record)
                save()
                for outer_row in affected:
                    if polished:
                        check_time(deadline)
                        guided = np.array([j for j in range(9) if not mask >> j & 1])
                        repair = RealGroupIntegrator(model, outer=1, max_particles=1024,
                            seed=20261015 + 1000*row['scramble'] + int(outer_row), deadline=deadline, proposal='full')
                        repair.source = integrator.source[outer_row:outer_row+1].copy()
                        repair.target = integrator.target[outer_row:outer_row+1].copy()
                        _, _, step = conditional_proposal(model, repair.source, repair.target, np.arange(9), guided)
                        value, row_ess, row_maximum, particles = repair.refine_row(mask, 511, 0, guided, 1024)
                        record['refinements'].append({'outer_row': int(outer_row), 'particles': particles,
                            'ess': row_ess, 'maximum_weight': row_maximum, 'log_marginal': value,
                            'proposal_projected_step': step, 'proposal': 'existing real-group-v4 80/15/5'})
                        save()
                        if row_ess < 10 or row_maximum >= .25 or step >= 1e-3:
                            raise ArithmeticError(f'Optimized proposal unresolved for mask={mask}, outer_row={outer_row}')
                        native[outer_row], ess[outer_row], maximum[outer_row] = value, row_ess, row_maximum
                        continue
                    seed = int(np.random.SeedSequence([20261015, row['scramble'], mask, int(outer_row)]).generate_state(1)[0])
                    for particles in (1024, 4096, 16384, 65536):
                        check_time(deadline)
                        uniforms = qmc.Sobol(10, scramble=True, seed=seed).random_base2(int(np.log2(particles)))[None]
                        value, row_ess, row_maximum = row_integrals(integrator, mask, np.array([outer_row]), particles, uniforms)
                        detail = {'outer_row': int(outer_row), 'particles': particles, 'ess': float(row_ess[0]),
                                  'maximum_weight': float(row_maximum[0]), 'log_marginal': float(value[0])}
                        record['refinements'].append(detail)
                        save()
                        if row_ess[0] >= 10 and row_maximum[0] < .25:
                            native[outer_row] = value[0]
                            ess[outer_row] = row_ess[0]
                            maximum[outer_row] = row_maximum[0]
                            break
                    else:
                        raise ArithmeticError(f'Weight precision unresolved for mask={mask}, outer_row={outer_row}')
                new_mean = float(np.mean(native - denominator_values))
                updated[index][str(mask)] = new_mean
                record.update(information_nats=new_mean, change_nats=new_mean-baseline_mean,
                              minimum_ess=float(ess.min()), maximum_weight=float(maximum.max()))
                new_diagnostics.append({**diagnostic, 'ess_minimum': float(ess.min()),
                                        'maximum_normalized_weight': float(maximum.max())})
                save()
        values = np.array([[r[str(m)] for m in pilot['masks']] for r in updated])
        mean = values.mean(0)
        sem = values.std(0, ddof=1) / 2
        coarse = next(x for x in pilot['precision_result']['summary'] if x['outer'] == 1024)
        change = float(np.max(abs(mean - np.array([coarse['mean_ei_nats'][str(m)] for m in pilot['masks']]))))
        audits = [syn_audit(row) for row in updated]
        for audit in audits:
            require_valid_syn(audit)
        minimum_ess = min(d['ess_minimum'] for d in new_diagnostics)
        maximum_weight = max(d['maximum_normalized_weight'] for d in new_diagnostics)
        passed = (minimum_ess >= 10 and maximum_weight < .25 and sem.max() < .01 and change < .01
                  and pilot['precision_result']['changes_nats'][0] < .01
                  and pilot['precision_result']['proposal_audit']['maximum_projected_step_coordinate'] < 1e-3)
        payload.update(status='feasibility_passed' if passed else 'precision_failed',
                       updated_finest_information_nats=updated, maximum_information_sem_nats=float(sem.max()),
                       information_sem_nats=dict(zip(map(str, pilot['masks']), sem.tolist())),
                       outer1024_to4096_change_nats=change, minimum_ess=minimum_ess,
                       maximum_weight=maximum_weight, syn_audits=audits)
        save()
    except Exception as error:
        payload.update(status='budget_exhausted' if isinstance(error, TimeoutError) else 'precision_failed',
                       error=f'{type(error).__name__}: {error}')
        save()


def main():
    global RESULT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--polished', action='store_true', help='Use the existing bounded group proposal on failed rows')
    args = parser.parse_args()
    if args.polished:
        RESULT = OUTPUT.with_name('proposal_refinement_polished.json')
    if args.worker:
        worker(args.polished)
        return
    ledger = read(OUTPUT.with_name('budget.json'))
    remaining = ledger['deadline_epoch'] - time.time()
    if remaining <= 0:
        raise TimeoutError('Existing trial deadline reached; no new budget is authorized')
    claim = RESULT.with_name(RESULT.stem + '_launch.json')
    with claim.open('x') as stream:
        json.dump({'started_epoch': time.time(), 'deadline_epoch': ledger['deadline_epoch']}, stream)
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}
    with RESULT.with_suffix('.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
                                  *(['--polished'] if args.polished else [])],
                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        launch = read(claim)
        launch['child_pid'] = child.pid
        save_json(claim, launch)
        try:
            child.wait(timeout=max(.001, ledger['deadline_epoch'] - time.time()))
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            payload = read(RESULT) if RESULT.exists() else {}
            payload.update(status='budget_exhausted', error='Original approved pilot deadline reached')
            save_json(RESULT, payload)
        launch.update(exit_code=child.returncode, finished_epoch=time.time())
        save_json(claim, launch)
    result = read(RESULT)
    print(json.dumps({'status': result['status'], 'error': result.get('error'),
                      'records': len(result.get('records', [])), 'output': str(RESULT)}))


if __name__ == '__main__':
    main()
