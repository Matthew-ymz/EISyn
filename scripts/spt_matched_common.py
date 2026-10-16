"""Shared frozen-model protocol and one-deadline execution for matched cases."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_glv_finite_tm import RealConditionalTM
from scripts.spt_glv_group_tm import RealGroupIntegrator
from scripts.spt_kuramoto_finite_tm import check_time, save_json
from scripts.spt_glv_a_comparison import quality, labels, splits

FOLDER = ROOT / 'results/spt_matched_three_case/full_comparison'
PILOT = ROOT / 'results/spt_matched_three_case/glv_simplified_A/pilot.json'
TOLERANCE = .01
VERSION = 'matched-simplified-a-v1'


def read(path):
    return json.loads(Path(path).read_text())


def deadline():
    budget = read(FOLDER / 'budget.json')
    if budget['budget_seconds'] != 28800:
        raise ValueError('Expected the explicitly approved eight-hour total budget')
    return time.monotonic() + budget['deadline_epoch'] - time.time()


def fitted_model():
    from scripts.spt_glv_simplified_a_pilot import MatchedSimplifiedGLV
    if read(PILOT.with_name('proposal_refinement_polished.json'))['status'] != 'feasibility_passed':
        raise ValueError('Matched pilot proposal repair did not pass')
    pilot = read(PILOT)
    expected = {','.join(map(str, e)): w for e, w in MatchedSimplifiedGLV(
        {tuple(x['edge']): x['weight'] for x in read(ROOT / 'results/spt_local_kuramoto_tm/ac_comparison/A/frozen_model.json')['network']}).weights.items()}
    if pilot['generator']['weights'] != expected:
        raise ValueError('Frozen gLV and Kuramoto A weighted networks differ')
    return RealConditionalTM.load(pilot['model'])


def model_sha(model):
    return hashlib.sha256(json.dumps(model.payload(), sort_keys=True).encode()).hexdigest()


def json_native(value):
    """Preserve numerical values while serializing NumPy audit scalars."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return json_native(value.tolist())
    if isinstance(value, dict):
        return {key: json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(item) for item in value]
    return value


def state(stage, payload):
    payload.update(process_id=os.getpid(), updated_epoch=time.time())
    save_json(FOLDER / f'{stage}.json', json_native(payload))


def diagnostic_issues(d):
    issues = []
    if d.get('ess_minimum') is not None and d['ess_minimum'] < 10:
        issues.append('ESS<10')
    weight = d.get('maximum_normalized_weight', d.get('maximum_weight'))
    if weight is not None and weight >= .25:
        issues.append('weight>=.25')
    if d.get('max_projected_step', d.get('max_projected_remaining_step_coordinate', 0)) >= 1e-3:
        issues.append('proposal_step>=1e-3')
    return issues


class Oracle:
    """All marginals belong to the same fitted joint; no truth enters a query."""
    def __init__(self, model, outer, end, particles=256):
        self.model, self.outer, self.end = model, outer, end
        self.particles = particles
        self.protocol = dict(version=VERSION, model_sha=model_sha(model), outer=outer,
                             particles=particles, scrambles=4, source='iid Uniform(-pi,pi)^9',
                             target_rule='explicit query target; full511 for SPT', units='nats',
                             proposal='existing real-group-v4 full missing 80/15/5 with recorded row refinement')
        self.path = FOLDER / (f'queries_n{outer}.jsonl' if particles == 256 else f'queries_n{outer}_p{particles}.jsonl')
        self.records = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                if row['protocol'] != self.protocol:
                    raise ValueError('Query scientific provenance mismatch')
                self.records[row['key']] = row
        self.integrators = None
        self.used = set()

    def query(self, source, target):
        check_time(self.end)
        key = f'{source}:{target}'
        self.used.add(key)
        if key in self.records:
            return np.asarray(self.records[key]['values_nats'])
        if self.integrators is None:
            self.integrators = [RealGroupIntegrator(self.model, outer=self.outer, max_particles=max(256,self.particles),
                seed=20261015+1000*r, deadline=self.end, cache_dir=FOLDER/'glv_marginals', proposal='full')
                for r in range(4)]
        values, diagnostics, prefixes = [], [], []
        started = time.monotonic()
        for repeat, estimator in enumerate(self.integrators):
            value, ds = estimator.information(source, target, self.particles)
            values.append(value)
            prefixes.append(ds[0]['inner_prefix_change_nats'] - ds[1]['inner_prefix_change_nats'])
            diagnostics.append({'scramble': repeat, 'marginals': ds})
        values = np.asarray(values)
        if not np.isfinite(values).all():
            raise ArithmeticError(f'Nonfinite MI for {key}')
        issues = sorted({issue for row in diagnostics for d in row['marginals'] for issue in diagnostic_issues(d)})
        sem = float(values.std(ddof=1)/2)
        if sem >= .01:
            issues.append('information_SEM>=.01')
        if abs(np.mean(prefixes)) >= .01:
            issues.append('inner64_to256_mean_change>=.01')
        if values.min() < -TOLERANCE:
            issues.append(f'MI below -{TOLERANCE}: minimum={values.min()}, affected={int((values < -TOLERANCE).sum())}')
        row = dict(key=key, protocol=self.protocol, values_nats=values.tolist(), mean_nats=float(values.mean()),
                   sem_nats=sem, inner_mean_change_nats=float(np.mean(prefixes)), diagnostics=diagnostics,
                   issues=issues, valid=not issues, elapsed_seconds=time.monotonic()-started)
        with self.path.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        self.records[key] = row
        return values


def guarded(stage, script, worker, *, worker_args=(), resume=False):
    """Guard one stage with the shared immutable deadline, including native calls."""
    launch_path = FOLDER / f'{stage}_launch.json'
    budget = read(FOLDER/'budget.json')
    remaining = budget['deadline_epoch'] - time.time()
    if remaining <= 0:
        raise TimeoutError('Shared eight-hour deadline has passed')
    if launch_path.exists() and not resume:
        raise ValueError(f'{stage} already launched; an explicit cache-preserving resume is required')
    if launch_path.exists():
        old = read(launch_path)
        pid = old.get('child_pid')
        if pid and subprocess.run(['ps', '-p', str(pid), '-o', 'pid='], capture_output=True, text=True).stdout.strip():
            raise RuntimeError(f'{stage} worker still alive')
    else:
        old = {}
    launch = {'started_epoch': time.time(), 'deadline_epoch': budget['deadline_epoch'],
              'attempt': old.get('attempt', 0)+1, 'previous_attempt': old or None,
              'script_sha256': hashlib.sha256(Path(script).read_bytes()).hexdigest()}
    if not resume:
        with launch_path.open('x') as stream:
            json.dump(launch, stream, indent=2)
    else:
        save_json(launch_path, launch)
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}
    with (FOLDER/f'{stage}.log').open('a') as log:
        child = subprocess.Popen([sys.executable, '-u', str(Path(script).resolve()), '--worker', *worker_args],
                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        launch['child_pid'] = child.pid
        save_json(launch_path, launch)
        try:
            child.wait(timeout=max(.001, budget['deadline_epoch']-time.time()))
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            payload = read(FOLDER/f'{stage}.json') if (FOLDER/f'{stage}.json').exists() else {}
            payload.update(status='budget_exhausted', error='Shared eight-hour hard deadline reached')
            state(stage, payload)
        launch.update(exit_code=child.returncode, finished_epoch=time.time())
        save_json(launch_path, launch)
    print(json.dumps({'stage': stage, 'exit_code': child.returncode, 'result': str(FOLDER/f'{stage}.json')}))
