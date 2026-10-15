"""Continue the authorized natural stage after both retained base comparisons end.

The original ledger is read-only. One shared four-hour natural ledger is created
once, and each child owns a case-specific cache. No other experiment is started.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'results/spt_local_kuramoto_tm/ac_comparison'
STATE = FOLDER / 'natural_stage.json'


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def original_processes():
    output = subprocess.check_output(['ps', '-axo', 'pid=,args='], text=True)
    names = ('scripts/spt_kuramoto_ac_selected.py', 'scripts/spt_kuramoto_ac_surd.py',
             'scripts/spt_kuramoto_ac_block.py', 'scripts/refine_spt_kuramoto_ac_block.py',
             'scripts/refine_spt_kuramoto_ac_surd.py')
    return [line.strip() for line in output.splitlines()
            if any(name in line for name in names) and 'python' in line]


def run():
    authorization = read(FOLDER / 'natural_authorization.json')
    if authorization.get('authorized_additional_wall_clock_hours') != 4:
        raise ValueError('The natural stage requires its explicit four-hour authorization')
    original = read(FOLDER / 'budget.json')
    state = {'status': 'waiting_for_original_four_methods', 'cases': ['A', 'C'],
             'authorization': authorization, 'original_deadline_epoch_seconds': original['deadline_epoch_seconds']}
    save(STATE, state)
    while original_processes():
        time.sleep(30)
    # A terminal numerical failure is retained as such, rather than being
    # converted to a zero or blocking the separately requested natural methods.
    for case in 'AC':
        coarse = read(FOLDER / case / 'block_selected_n256_p256.json')
        high = read(FOLDER / case / 'block_selected_refinement.json')
        if coarse.get('status') in (None, 'running'):
            if time.time() < original['deadline_epoch_seconds']:
                raise RuntimeError(f'{case}: Block exited without a terminal artifact')
        if coarse.get('status') == 'complete' and high.get('status') in (None, 'running'):
            if time.time() < original['deadline_epoch_seconds']:
                raise RuntimeError(f'{case}: scheduled Block refinement has no terminal artifact')
    budget_path = FOLDER / 'natural_budget.json'
    if not budget_path.exists():
        started = time.time()
        names = ('spt_kuramoto_natural_baselines.py', 'spt_kuramoto_natural_source.py')
        ledger = {'start_epoch_seconds': started, 'deadline_epoch_seconds': started + 4 * 3600,
                  'wall_clock_budget_hours': 4, 'cases': ['A', 'C'],
                  'methods': authorization['methods'], 'original_budget_unchanged': True,
                  'pilot_sha256': original['pilot_sha256'],
                  'source_sha256': {name: hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest()
                                    for name in names}}
        with budget_path.open('x') as stream:
            json.dump(ledger, stream, indent=2)
    budget = read(budget_path)
    deadline = budget['deadline_epoch_seconds']
    if deadline - budget['start_epoch_seconds'] != 4 * 3600:
        raise ValueError('The shared natural deadline differs from the approved duration')
    state.update(status='running', budget=budget, processes={})
    children = {}
    handles = []
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    try:
        for case in 'AC':
            if time.time() >= deadline:
                path = FOLDER / case / 'natural_baselines.json'
                record = read(path)
                if record.get('status') != 'complete':
                    record.update(status='budget_exhausted',
                                  error='Shared four-hour natural-stage deadline reached before launch')
                    save(path, record)
                continue
            handle = (FOLDER / case / 'natural_baselines.log').open('a')
            handles.append(handle)
            child = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u',
                'scripts/spt_kuramoto_natural_baselines.py', '--case', case],
                cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            children[case] = child
            state['processes'][case] = {'pid': child.pid, 'log': str(handle.name)}
        save(STATE, state)
        print(json.dumps({'stage': 'natural methods started', 'processes': state['processes'],
                          'deadline_epoch_seconds': deadline}), flush=True)
        while any(child.poll() is None for child in children.values()) and time.time() < deadline:
            time.sleep(min(20, max(.01, deadline - time.time())))
        for case, child in children.items():
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=30)
                path = FOLDER / case / 'natural_baselines.json'
                record = read(path)
                record.update(status='budget_exhausted',
                              error='Shared four-hour natural-stage deadline reached')
                save(path, record)
            state['processes'][case]['exit_code'] = child.returncode
        state.update(status='local_natural_stage_ended', natural_finished_epoch_seconds=time.time(),
            native_status={case: read(FOLDER / case / 'natural_baselines.json').get('status', 'not_started')
                           for case in 'AC'})
    finally:
        for handle in handles:
            handle.close()
        save(STATE, state)
    subprocess.run([str(ROOT / '.venv/bin/python'), 'scripts/summarize_spt_kuramoto_ac_selected.py'],
                   cwd=ROOT, env=env, check=True)
    glv = ROOT / 'results/spt_local_glv_tm/a_comparison'
    glv_budget = read(glv / 'budget.json')
    glv_result = read(glv / 'comparison_1024.json')
    # The third case belongs to another chat. Wait read-only for its existing
    # job/budget; do not launch, stop, or modify any part of that experiment.
    if str(glv_result.get('status', '')).startswith('running'):
        state['status'] = 'waiting_for_glv_readonly'
        save(STATE, state)
    while (str(glv_result.get('status', '')).startswith('running')
           and time.time() < glv_budget.get('deadline_epoch', 0) + 60):
        time.sleep(30)
        glv_result = read(glv / 'comparison_1024.json')
    subprocess.run([str(ROOT / '.venv/bin/python'), 'scripts/summarize_spt_glv_kuramoto_six.py'],
                   cwd=ROOT, env=env, check=True)
    state.update(status='all_local_jobs_ended', finished_epoch_seconds=time.time(),
                 glv_native_status=glv_result.get('status', 'missing'))
    save(STATE, state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        state = read(STATE)
        state.update(status='orchestration_failed', error=str(error), finished_epoch_seconds=time.time())
        save(STATE, state)
        raise
